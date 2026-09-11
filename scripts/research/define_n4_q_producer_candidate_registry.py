"""Freeze the existing q-producer candidates for the N4 V2 comparison.

This definition-only consumer searches and records existing producer code and
contracts.  It does not load q values, fit a producer, sample a model bank,
invoke an evaluator, or read sealed confirmation values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, canonical_sha256, file_sha256
from scripts.research import define_n3_development_bank_v2 as source_definition


TASK_ID = "N4_Q_PRODUCER_CANDIDATE_REGISTRY"
GATE = TASK_ID
SCHEMA = "n4_q_producer_candidate_registry.v1"
EXPECTED_BRANCH = source_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = source_definition.EXPECTED_STARTING_HEAD

REGISTRY_OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_candidate_registry_v1_attempt_02"
)
REGISTRY_NAME = "n4_q_producer_candidate_registry.v1.json"
EXECUTOR_PATH = Path("scripts/research/execute_n4_q_producer_development_comparison.py")

SOURCE_MANIFEST = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_source_manifest_v1_attempt_01/"
    "n3_development_bank_v2_source_manifest.v1.json"
)
Q_ARTIFACT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_q_materialization_v1_attempt_01/"
    "n3_development_bank_v2_q.v1.pt"
)
FOLD_MANIFEST = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_five_fold_protocol_v1_attempt_01/"
    "n3_development_bank_v2_fold_manifest.v1.json"
)
CEILING_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)

AUTHORITY_SHA256 = {
    SOURCE_MANIFEST.as_posix(): "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812",
    Q_ARTIFACT.as_posix(): "fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf",
    FOLD_MANIFEST.as_posix(): "4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e",
    CEILING_REPORT.as_posix(): "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826",
}

COMPARISON_SOURCE = Path("scripts/research/compare_n3_minimal_competing_producer_baselines_v1.py")
COPULA_SOURCE = Path("scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py")
COPULA_CONTRACT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_contract_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_contract.v1.json"
)
COPULA_CONTRACT_SHA256 = "3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4"
BASELINE_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_minimal_competing_producer_baselines_v1_attempt_01/"
    "n3_minimal_competing_producer_baselines_report.v1.json"
)
BASELINE_REPORT_SHA256 = "af7611cfbca211e666d7773e81b230a1b380b91583817c3c4ab80925f2a16440"
COPULA_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_report.v1.json"
)
COPULA_REPORT_SHA256 = "78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d"

DIMENSION = 100
MODEL_SAMPLE_COUNT = 200
PROPOSAL_BUDGET = 1_000_000
PROPOSAL_BATCH_SIZE = 4096


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_state(repo_root: Path) -> dict[str, Any]:
    return source_definition._git_state(repo_root)


def _ref(repo_root: Path, path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"candidate reference is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None:
        _require(actual == expected_sha256, f"candidate reference hash changed: {target}")
    return {
        "path": _relative(repo_root, target),
        "sha256": actual,
    }


def _candidate_entries(repo_root: Path, git_head: str) -> list[dict[str, Any]]:
    comparison_ref = _ref(repo_root, COMPARISON_SOURCE)
    copula_ref = _ref(repo_root, COPULA_SOURCE)
    baseline_ref = _ref(repo_root, BASELINE_REPORT, BASELINE_REPORT_SHA256)
    copula_report_ref = _ref(repo_root, COPULA_REPORT, COPULA_REPORT_SHA256)
    copula_contract_ref = _ref(repo_root, COPULA_CONTRACT, COPULA_CONTRACT_SHA256)

    common_sampling = {
        "model_sample_count": MODEL_SAMPLE_COUNT,
        "rng_device": "cpu",
        "rng_generator": "torch.Generator(device='cpu')",
        "dtype": "float64",
        "samplewise_normalization": False,
        "best_of_n": False,
        "seed_retry": False,
        "heldout_seed_tuning": False,
    }
    return [
        {
            "candidate_id": "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
            "candidate_version": "v1",
            "implementation_path": comparison_ref["path"],
            "implementation_sha256": comparison_ref["sha256"],
            "git_commit": git_head,
            "model_family": "elliptical Gaussian amplitude with analytic covariance shrinkage",
            "input_q_dimensionality": DIMENSION,
            "output_q_dimensionality": DIMENSION,
            "deterministic": {"fitting": True, "sampling": False},
            "stochastic": {"fitting": False, "sampling": True},
            "fitting_algorithm": "existing _fit_fold -> analytic mean/covariance and Ledoit-Wolf-style plug-in shrinkage",
            "fixed_hyperparameters": {
                "covariance_denominator": "n_fit-1",
                "shrinkage_alpha": "clip(beta/delta,0,1); alpha=0 when delta=0",
                "dimension": DIMENSION,
                "fit_rows": 4000,
            },
            "initialization_rule": "none; closed-form fit",
            "rng_source": {
                **common_sampling,
                "namespace": "N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1_CPU",
                "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
            },
            "compute_cap": {
                "fit": "one analytic pass plus positive-definite checks",
                "proposal_budget": PROPOSAL_BUDGET,
                "proposal_batch_size": PROPOSAL_BATCH_SIZE,
            },
            "convergence_rule": "analytic fitting; finite covariance and successful Cholesky are required",
            "fallback_behavior": "finite q>=0 rejection only; no clipping, repair, retry, or adaptive budget",
            "serialization_format": "torch.save mapping with float64 fitted tensors and frozen metadata",
            "analytical_fitting": True,
            "gradient_training": False,
            "sampling_contract": {
                "law": "mu + Normal(0,Sigma_shrunk)",
                "positivity_rule": "accept finite q>=0 elementwise",
                "rejection_is_fixed_support_filter": True,
                "rejection_is_not_best_of_n": True,
                **common_sampling,
            },
            "historical_lineage": {"development_comparison_report": baseline_ref},
            "v2_binding": "fit receives the exact 4000-row V2 q_fit for each fold; only the model bank has 200 rows",
        },
        {
            "candidate_id": "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
            "candidate_version": "v1",
            "implementation_path": comparison_ref["path"],
            "implementation_sha256": comparison_ref["sha256"],
            "git_commit": git_head,
            "model_family": "elliptical Gaussian amplitude with analytic radial scale-mixture extension",
            "input_q_dimensionality": DIMENSION,
            "output_q_dimensionality": DIMENSION,
            "deterministic": {"fitting": True, "sampling": False},
            "stochastic": {"fitting": False, "sampling": True},
            "fitting_algorithm": "existing _fit_fold -> shrinkage covariance then fit-only radial variance and nu",
            "fixed_hyperparameters": {
                "covariance_denominator": "n_fit-1",
                "radial_identification_predicate": "radial_variance > 2*d",
                "nu_formula": "(4*radial_variance + 2*d*(d-2))/(radial_variance-2*d)",
                "dimension": DIMENSION,
                "fit_rows": 4000,
            },
            "initialization_rule": "none; closed-form fit",
            "rng_source": {
                **common_sampling,
                "namespace": "N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1_CPU",
                "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
            },
            "compute_cap": {
                "fit": "one analytic pass plus fit-only radial moment check",
                "proposal_budget": PROPOSAL_BUDGET,
                "proposal_batch_size": PROPOSAL_BATCH_SIZE,
            },
            "convergence_rule": "analytic fitting; nu must be finite and strictly greater than 4",
            "fallback_behavior": "RADIAL_EXTENSION_NOT_IDENTIFIED_ON_FIT is a candidate-fold scientific failure; no parameter is invented",
            "serialization_format": "torch.save mapping with float64 fitted tensors and frozen metadata",
            "analytical_fitting": True,
            "gradient_training": False,
            "sampling_contract": {
                "law": "mu + sqrt((nu-2)/chi_square_nu) * Normal(0,Sigma_shrunk)",
                "positivity_rule": "accept finite q>=0 elementwise",
                "rejection_is_fixed_support_filter": True,
                "rejection_is_not_best_of_n": True,
                **common_sampling,
            },
            "historical_lineage": {"development_comparison_report": baseline_ref},
            "v2_binding": "fit receives the exact 4000-row V2 q_fit for each fold; radial identification is fit-only",
        },
        {
            "candidate_id": "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
            "candidate_version": "v1",
            "implementation_path": copula_ref["path"],
            "implementation_sha256": copula_ref["sha256"],
            "git_commit": git_head,
            "model_family": "fit-only empirical marginal plus Gaussianized-score dependence",
            "input_q_dimensionality": DIMENSION,
            "output_q_dimensionality": DIMENSION,
            "deterministic": {"fitting": True, "sampling": False},
            "stochastic": {"fitting": False, "sampling": True},
            "fitting_algorithm": "existing _fit_fold -> stable midrank empirical marginals, Phi^-1 Gaussianization, analytic R_shrunk",
            "fixed_hyperparameters": {
                "plotting_positions": "(rank-0.5)/n_fit",
                "tie_rule": "average rank with stable value-then-original-row order",
                "quantile_interpolation": "piecewise-linear with observed endpoint tails",
                "standardization_denominator": "n_fit-1",
                "dependence_denominator": "n_fit-1",
                "unit_diagonal_renormalization": False,
                "epsilon_clamp": False,
                "fit_rows": 4000,
            },
            "initialization_rule": "none; closed-form rank transform",
            "rng_source": {
                **common_sampling,
                "namespace": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_CPU",
                "seed_derivation": "sha256(namespace|role|fold|replicate) first 64 bits masked to signed 63-bit",
            },
            "compute_cap": {
                "fit": "one stable sort per coordinate plus analytic dependence fit",
                "model_bank": "one 200-row Gaussian bank",
            },
            "convergence_rule": "finite Gaussianized scores, positive-definite R_shrunk, and unit-diagonal check",
            "fallback_behavior": "no clipping, epsilon repair, positivity rejection, retry, or adaptive sampling",
            "serialization_format": "torch.save mapping with sorted empirical marginals, cholesky, and frozen metadata",
            "analytical_fitting": True,
            "gradient_training": False,
            "sampling_contract": {
                "law": "z~Normal(0,R_shrunk); u=Phi(z); q_j=Q_j(u_j)",
                "positivity_rule": "finite q>=0 is required by the existing implementation",
                "rejection_is_fixed_support_filter": False,
                "rejection_is_not_best_of_n": True,
                **common_sampling,
            },
            "historical_lineage": {
                "producer_contract": copula_contract_ref,
                "development_report": copula_report_ref,
            },
            "v2_binding": "the existing fit-count assertion is bound to 4000 before fitting; algorithm and sampling semantics are unchanged",
        },
    ]


def _excluded_existing_implementations(repo_root: Path) -> list[dict[str, Any]]:
    paths_and_reasons = [
        (
            Path("scripts/research/fit_n3_amplitude_producer_baseline.py"),
            "historical r=99 state fitter has no existing q sampling function and is bound to a separate old authority",
        ),
        (
            Path("scripts/research/execute_n3_local_location_mixture_selection_v1.py"),
            "historical selection wrapper is bound to its 800/200 contract; no V2 count adapter was pre-registered",
        ),
        (
            Path("scripts/research/execute_n3_local_shrinkage_covariance_mixture_selection_v1.py"),
            "historical component-local covariance wrapper hardcodes the old 800-row fit and would require implementation changes",
        ),
        (
            Path("scripts/research/execute_n3_oracle_guided_minimal_intervention_comparison_v2.py"),
            "registered as a diagnostic intervention, not an independent q-producer family",
        ),
    ]
    result: list[dict[str, Any]] = []
    for path, reason in paths_and_reasons:
        item: dict[str, Any] = {"implementation_path": path.as_posix(), "reason_not_registered": reason}
        target = repo_root / path
        if target.is_file():
            item["implementation_sha256"] = file_sha256(target)
        result.append(item)
    return result


def build_registry(repo_root: Path) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N4 authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "current HEAD changed from the V2 authority base")
    for path_text, expected in AUTHORITY_SHA256.items():
        target = repo_root / Path(path_text)
        _require(target.is_file(), f"V2 authority is missing: {target}")
        _require(file_sha256(target) == expected, f"V2 authority hash changed: {target}")
    executor = _ref(repo_root, EXECUTOR_PATH)
    script_path = Path(__file__).resolve()
    candidates = _candidate_entries(repo_root, git["head"])
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    _require(len(candidate_ids) == len(set(candidate_ids)), "candidate registry contains duplicate ids")
    return {
        "schema": SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "FROZEN",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            **git,
            "definition_script": _relative(repo_root, script_path),
            "definition_script_sha256": file_sha256(script_path),
            "executor_script": executor["path"],
            "executor_script_sha256": executor["sha256"],
        },
        "v2_authorities": [
            {"path": path_text, "sha256": expected}
            for path_text, expected in AUTHORITY_SHA256.items()
        ],
        "scope": {
            "coordinate": "R2_AMPLITUDE",
            "input_q_dimensionality": DIMENSION,
            "output_q_dimensionality": DIMENSION,
            "candidate_selection_rule": "existing q/amplitude implementation with a frozen fit and sampling function that can be bound to the exact V2 q_fit without changing its model family",
            "no_new_model_family": True,
            "no_image_or_optical_evaluator": True,
            "excluded_existing_implementations": _excluded_existing_implementations(repo_root),
        },
        "candidates": candidates,
        "candidate_count": len(candidates),
        "freeze": {
            "registry_frozen_before_first_evaluator_call": True,
            "candidate_addition_after_freeze": False,
            "candidate_deletion_after_freeze": False,
            "parameter_mutation_after_freeze": False,
            "fit_method_mutation_after_freeze": False,
            "heldout_driven_tuning": False,
            "tie_break_registered": False,
            "registry_self_hash_is_recorded_by_downstream_lineage": True,
        },
        "governance": {
            "q_values_loaded": False,
            "sealed_confirmation_values_loaded": False,
            "selected_producer": None,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "not_executed": [
            "V2 q loading",
            "producer fitting",
            "model-q sampling",
            "heldout compatibility evaluation",
            "sealed confirmation loading or materialization",
            "image-space or optical evaluation",
            "training",
        ],
    }


def write_registry(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once registry output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        payload = build_registry(repo_root)
        registry_path = temporary / REGISTRY_NAME
        atomic_write_new_json(registry_path, payload, artifact_kind="N4 candidate registry")
        if output_dir.exists():
            raise FileExistsError(f"registry output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    final_path = output_dir / REGISTRY_NAME
    return {
        "status": payload["status"],
        "candidate_count": payload["candidate_count"],
        "candidate_ids": [item["candidate_id"] for item in payload["candidates"]],
        "registry": _relative(repo_root, final_path),
        "registry_sha256": file_sha256(final_path),
        "git_head": payload["git"]["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REGISTRY_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(write_registry(repo_root, output_dir))


if __name__ == "__main__":
    main()
