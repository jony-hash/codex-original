"""Execute the frozen N3 unsupervised local-location mixture contract once.

This consumer is development-only.  It reads the existing 1000-row N3
development pool, reuses the frozen five-fold assignment and analytic
shrinkage covariance, and never reads the reserved confirmation source,
materializes OLS states, invokes an optical/evaluator path, or uses CUDA.
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
from scripts.research import (
    compare_n3_minimal_competing_producer_baselines_v1 as comparison,
    define_n3_unsupervised_local_location_mixture_producer_contract_v1 as contract_impl,
)


TASK_ID = "EXECUTE_N3_LOCAL_LOCATION_MIXTURE_SELECTION_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_local_location_mixture_selection_report.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"

CONTRACT_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_unsupervised_local_location_mixture_producer_contract_v1_attempt_01/"
    "n3_unsupervised_local_location_mixture_producer_contract.v1.json"
)
EXPECTED_CONTRACT_SHA256 = "38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449"
REPORT_NAME = "n3_local_location_mixture_selection_report.v1.json"

CLASSIFICATION_SELECTED = "N3_LOCAL_LOCATION_MIXTURE_MODEL_ORDER_SELECTED"
CLASSIFICATION_NOT_SUPPORTED = "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED"
CLASSIFICATION_INCONCLUSIVE = "N3_LOCAL_LOCATION_MIXTURE_INCONCLUSIVE"

FOLD_COUNT = 5
DIMENSION = 100
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
SEED_DERIVATION = "sha256(namespace|K|role|fold|replicate) first 64 bits masked to signed 63-bit"


class CandidateFoldFailure(RuntimeError):
    """A preregistered candidate/fold scientific failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scientific_require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateFoldFailure(message)


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


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(repo_root, CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "frozen local-mixture contract")
    _require(contract.get("schema") == "n3_unsupervised_local_location_mixture_producer_contract.v1", "contract schema changed")
    _require(contract.get("task_id") == "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1", "contract task changed")
    _require(contract.get("gate") == contract["task_id"], "contract gate changed")
    _require(
        contract.get("classification") == "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY",
        "contract is not ready",
    )
    _require(
        contract.get("claim_classification") == "N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED",
        "contract claim boundary changed",
    )

    domain = _mapping(contract.get("component_count_domain"), "component_count_domain")
    _require(domain.get("candidate_K") == [2, 4, 8], "candidate K domain changed")
    _require(domain.get("post_result_K_addition") is False, "post-result K addition is not forbidden")
    _require(domain.get("post_result_initialization_change") is False, "post-result initialization change is not forbidden")
    _require(domain.get("post_result_threshold_change") is False, "post-result threshold change is not forbidden")

    development = _mapping(contract.get("development_identity"), "development_identity")
    _require(development.get("count") == 1000, "development pool count changed")
    fold_identity = _mapping(development.get("fold_assignment"), "contract fold assignment")
    _require(fold_identity.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "contract fold identity changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "contract fold count changed")
    _require(fold_identity.get("q_values_used") is False, "contract fold construction used q values")
    _require(fold_identity.get("ols_values_used") is False, "contract fold construction used OLS values")

    shared = _mapping(contract.get("shared_covariance_contract"), "shared_covariance_contract")
    _require(shared.get("fit_rows_per_fold") == 800, "shared covariance fit count changed")
    _require(shared.get("heldout_rows_per_fold") == 200, "shared covariance held-out count changed")
    _require(shared.get("shared_covariance_fixed_during_EM") is True, "shared covariance is not fixed")
    _require(
        shared.get("parent_report_sha256") == contract_impl.COMPARISON_REPORT_SHA256,
        "shared covariance parent identity changed",
    )

    initialization = _mapping(contract.get("initialization_contract"), "initialization_contract")
    _require(initialization.get("label_input") is False, "labels are permitted in initialization")
    _require(initialization.get("random_restarts") is False, "random restarts are permitted")
    _require("farthest-point" in str(initialization.get("subsequent_centers")), "farthest-point initialization changed")

    fit = _mapping(contract.get("fit_contract"), "fit_contract")
    _require(fit.get("algorithm") == "deterministic CPU float64 EM with fixed Sigma_shared", "fit algorithm changed")
    _require(fit.get("fitted_parameters_only") == ["pi_k", "mu_k"], "fitted parameter set changed")
    _require(fit.get("max_iterations") == 200, "EM horizon changed")
    _require(float(fit.get("convergence_tolerance")) == 1.0e-10, "EM convergence tolerance changed")
    _require(float(fit.get("monotonicity_tolerance_per_row")) == 1.0e-10, "EM monotonicity tolerance changed")
    _require(fit.get("convergence_required") is True, "EM convergence is not required")

    collapse = _mapping(contract.get("component_collapse_contract"), "component_collapse_contract")
    _require(float(collapse.get("mass_floor")) == 1.0e-8, "component mass floor changed")
    _require(collapse.get("action") == "candidate/fold scientific failure", "collapse action changed")
    _require(collapse.get("change_K") is False, "collapse may change K")
    _require(collapse.get("random_restart") is False, "collapse may restart")

    sampling = _mapping(contract.get("sampling_contract"), "sampling_contract")
    expected_sampling = {
        "target_accepted": comparison.TARGET_ACCEPTED,
        "proposal_budget": comparison.PROPOSAL_BUDGET,
        "proposal_batch_size": comparison.PROPOSAL_BATCH_SIZE,
        "data_vs_model_replicates": comparison.DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": comparison.MODEL_VS_MODEL_REPLICATES,
    }
    for key, expected in expected_sampling.items():
        _require(sampling.get(key) == expected, f"sampling setting changed: {key}")
    _require(sampling.get("rng_device") == "cpu", "sampling device changed")
    _require(sampling.get("rng_generator") == "torch.Generator(device='cpu')", "sampling generator changed")
    _require(sampling.get("legal_state") == "finite(q) and q>=0 elementwise", "positivity predicate changed")
    _require(sampling.get("clamp") is False, "clamp is permitted")
    _require(sampling.get("abs") is False, "abs repair is permitted")
    _require(sampling.get("samplewise_normalization") is False, "samplewise normalization is permitted")
    _require(isinstance(sampling.get("rng_namespace"), str) and sampling["rng_namespace"], "RNG namespace missing")
    _require(sampling.get("seed_substreams") == ["K", "fold", "data-vs-model replicate", "model-vs-model replicate"], "RNG substreams changed")

    evaluation = _mapping(contract.get("development_evaluation_contract"), "development_evaluation_contract")
    _require(evaluation.get("fit_rows") == 800 and evaluation.get("heldout_rows") == 200, "evaluation split changed")
    _require(evaluation.get("labels_as_model_inputs") is False, "labels are permitted as model inputs")
    _require(evaluation.get("optical_forward") is False and evaluation.get("evaluator") is False, "optical/evaluator execution is permitted")
    _require(evaluation.get("sealed_confirmation") is False, "sealed confirmation is permitted")

    compatibility = _mapping(contract.get("compatibility_contract"), "compatibility_contract")
    _require(float(compatibility.get("q")) == 0.95, "compatibility quantile changed")
    _require(compatibility.get("all_folds_required") is True, "all-fold requirement changed")
    _require(compatibility.get("own_model_reference_required") is True, "own model reference requirement changed")
    _require(compatibility.get("absolute_metric_thresholds") is False, "absolute thresholds were introduced")
    _require("covariance/norm" in str(compatibility.get("rule")), "compatibility guardrails changed")

    sealed = _mapping(contract.get("sealed_confirmation_firewall"), "sealed_confirmation_firewall")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    _require(sealed.get("sample_values_loaded") is False, "sealed sample loading is permitted")
    _require(sealed.get("confirmation_evaluation_authorized") is False, "sealed evaluation is authorized")
    governance = _mapping(contract.get("governance"), "governance")
    _require(governance.get("training_authorized") is False, "training authorization changed")
    _require(governance.get("formal_promotion") is False, "formal promotion changed")

    contract_git = _mapping(contract.get("git"), "contract git")
    script_path = repo_root / str(contract_git.get("contract_script"))
    _require(script_path.is_file(), "contract script is missing")
    _require(file_sha256(script_path) == contract_git.get("contract_script_sha256"), "contract script changed")
    return contract


def _verify_input_authorities(repo_root: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_authorities = comparison._verify_authorities(repo_root)
    actual_boundaries = comparison._verify_historical_boundaries(repo_root)
    expected_authorities = _mapping(contract.get("authorities"), "contract authorities")
    _require(actual_authorities == expected_authorities["producer_data_authority"], "producer data authority changed")
    _require(actual_boundaries == expected_authorities["historical_boundaries"], "historical/sealed authority changed")
    return dict(actual_authorities), dict(actual_boundaries)


def _load_development(repo_root: Path, contract: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    q, source_indices, labels, pool_identity = comparison._load_development_pool(repo_root)
    fold_ids, fold_identity = comparison._assign_folds(source_indices, labels)
    expected_development = _mapping(contract["development_identity"], "contract development identity")
    for key in ("count", "source_indices_sha256", "labels_sha256", "source_row_positions_sha256"):
        _require(pool_identity.get(key) == expected_development.get(key), f"development identity changed: {key}")
    expected_fold = _mapping(expected_development["fold_assignment"], "contract fold assignment")
    _require(fold_identity == expected_fold, "frozen development fold assignment changed")
    _require(tuple(q.shape) == (1000, DIMENSION), "development q shape changed")
    return q, source_indices, labels, fold_ids, pool_identity, fold_identity


def _prepare_fold(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
) -> dict[str, Any]:
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    q_fit = q.index_select(0, fit_positions).to(dtype=torch.float64).contiguous()
    fit_source_indices = source_indices.index_select(0, fit_positions).contiguous()
    heldout = q.index_select(0, heldout_positions).to(dtype=torch.float64).contiguous()
    shared = comparison._fit_shrinkage(q_fit)
    centered = q_fit - shared["mu"]
    whitened = torch.linalg.solve_triangular(
        shared["chol"], centered.transpose(0, 1), upper=False
    ).transpose(0, 1).contiguous()
    _require(bool(torch.isfinite(whitened).all()), "shared-whitened fitting coordinates are non-finite")
    return {
        "fold": fold,
        "fit_positions": fit_positions,
        "heldout_positions": heldout_positions,
        "fit_source_indices": fit_source_indices,
        "heldout": heldout,
        "shared": shared,
        "whitened": whitened,
    }


def _log_density(
    whitened: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    chol: torch.Tensor,
) -> torch.Tensor:
    _scientific_require(bool(torch.isfinite(centers).all()), "component centers are non-finite")
    _scientific_require(bool(torch.isfinite(weights).all()) and bool(weights.gt(0.0).all()), "component weights are invalid")
    logdet = torch.log(torch.diagonal(chol)).sum()
    normalizer = logdet + 0.5 * float(DIMENSION) * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
    squared_distance = whitened[:, None, :].sub(centers[None, :, :]).square().sum(dim=2)
    values = torch.log(weights)[None, :] - 0.5 * squared_distance - normalizer
    _scientific_require(bool(torch.isfinite(values).all()), "component log densities are non-finite")
    return values


def _fit_mixture(context: Mapping[str, Any], *, fold: int, k: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    _require(k in tuple(settings["candidate_K"]), "K is outside the frozen domain")
    shared = context["shared"]
    whitened = context["whitened"]
    fit_source_indices = context["fit_source_indices"]
    initial_positions = contract_impl.initial_center_positions(
        whitened,
        fit_source_indices,
        fold=fold,
        k=k,
    )
    centers = whitened.index_select(0, torch.tensor(initial_positions, dtype=torch.int64)).contiguous()
    weights = torch.full((k,), 1.0 / float(k), dtype=torch.float64)
    initial_order = contract_impl.canonical_component_order(centers)
    initial_order_tensor = torch.tensor(initial_order, dtype=torch.int64)
    centers = centers.index_select(0, initial_order_tensor).contiguous()
    weights = weights.index_select(0, initial_order_tensor).contiguous()

    tolerance = float(settings["convergence_tolerance"])
    monotonicity_tolerance = float(settings["monotonicity_tolerance_per_row"])
    mass_floor = float(settings["mass_floor"])
    max_iterations = int(settings["max_iterations"])
    history: list[dict[str, float]] = []
    converged = False
    final_delta = None
    final_log_likelihood = None

    for iteration in range(1, max_iterations + 1):
        log_density = _log_density(whitened, centers, weights, shared["chol"])
        current_log_likelihood = torch.logsumexp(log_density, dim=1).mean()
        responsibilities = torch.softmax(log_density, dim=1)
        _scientific_require(bool(torch.isfinite(responsibilities).all()), "responsibilities are non-finite")
        component_mass = responsibilities.sum(dim=0)
        _scientific_require(bool(torch.isfinite(component_mass).all()), "component masses are non-finite")
        _scientific_require(bool((component_mass / float(whitened.shape[0]) > mass_floor).all()), "component collapse")

        new_weights = component_mass / float(whitened.shape[0])
        new_centers = responsibilities.transpose(0, 1).mm(whitened).div(component_mass[:, None])
        _scientific_require(bool(torch.isfinite(new_weights).all()) and bool(torch.isfinite(new_centers).all()), "M-step state is non-finite")
        new_order = contract_impl.canonical_component_order(new_centers)
        new_order_tensor = torch.tensor(new_order, dtype=torch.int64)
        new_weights = new_weights.index_select(0, new_order_tensor).contiguous()
        new_centers = new_centers.index_select(0, new_order_tensor).contiguous()
        new_log_density = _log_density(whitened, new_centers, new_weights, shared["chol"])
        new_log_likelihood = torch.logsumexp(new_log_density, dim=1).mean()
        delta = float(new_log_likelihood - current_log_likelihood)
        _scientific_require(torch.isfinite(new_log_likelihood), "log likelihood is non-finite")
        _scientific_require(delta >= -monotonicity_tolerance, "EM monotonicity failure")
        history.append(
            {
                "iteration": float(iteration),
                "log_likelihood_per_row": float(new_log_likelihood),
                "delta_per_row": delta,
            }
        )
        centers = new_centers
        weights = new_weights
        final_delta = delta
        final_log_likelihood = float(new_log_likelihood)
        if abs(delta) <= tolerance:
            converged = True
            break

    _scientific_require(converged, "EM did not converge within the frozen iteration horizon")
    component_means = shared["mu"] + centers.mm(shared["chol"].transpose(0, 1))
    _scientific_require(bool(torch.isfinite(component_means).all()), "component means are non-finite")
    return {
        "shared": shared,
        "centers": centers,
        "weights": weights,
        "component_means": component_means,
        "summary": {
            "fit_count": int(whitened.shape[0]),
            "initial_center_positions": [int(position) for position in initial_positions],
            "initial_center_source_indices": [int(fit_source_indices[position]) for position in initial_positions],
            "initial_canonical_order": [int(position) for position in initial_order],
            "em_iterations": len(history),
            "converged": converged,
            "final_delta_per_row": final_delta,
            "final_log_likelihood_per_row": final_log_likelihood,
            "log_likelihood_history": history,
            "component_weights": [float(value) for value in weights.tolist()],
            "shared_mu_sha256": tensor_sha256(shared["mu"]),
            "shared_sigma_sha256": tensor_sha256(shared["sigma"]),
            "component_centers_sha256": tensor_sha256(centers),
            "component_means_sha256": tensor_sha256(component_means),
            "shrinkage_alpha": float(shared["alpha"]),
            "shrinkage_tau": float(shared["tau"]),
        },
    }


def _seed(namespace: str, k: int, role: str, fold: int, replicate: int) -> int:
    material = f"{namespace}|K={k}|{role}|fold={fold}|replicate={replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_positive(
    fit: Mapping[str, Any],
    *,
    namespace: str,
    k: int,
    role: str,
    fold: int,
    replicate: int,
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_accepted = int(settings["target_accepted"])
    proposal_budget = int(settings["proposal_budget"])
    batch_size = int(settings["proposal_batch_size"])
    seed = _seed(namespace, k, role, fold, replicate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = fit["weights"]
    component_means = fit["component_means"]
    chol = fit["shared"]["chol"]
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    proposals = 0
    termination = "proposal_budget_exhausted"
    while proposals < proposal_budget and accepted_count < target_accepted:
        block_count = min(batch_size, proposal_budget - proposals)
        component_ids = torch.multinomial(weights, block_count, replacement=True, generator=generator)
        gaussian = torch.randn((block_count, DIMENSION), generator=generator, dtype=torch.float64)
        q = component_means.index_select(0, component_ids) + gaussian.mm(chol.transpose(0, 1))
        valid = torch.isfinite(q).all(dim=1) & q.ge(0.0).all(dim=1)
        positions = valid.nonzero(as_tuple=False).reshape(-1)
        needed = target_accepted - accepted_count
        if positions.numel() >= needed:
            selected = positions[:needed]
            accepted.append(q.index_select(0, selected).contiguous())
            proposals += int(selected[-1].item()) + 1
            accepted_count += needed
            termination = "target_accepted_reached"
            break
        if positions.numel() > 0:
            accepted.append(q.index_select(0, positions).contiguous())
            accepted_count += int(positions.numel())
        proposals += block_count
    result = torch.cat(accepted, dim=0) if accepted else torch.empty((0, DIMENSION), dtype=torch.float64)
    _require(tuple(result.shape) == (accepted_count, DIMENSION), "accepted q shape changed")
    if accepted_count:
        _require(bool(torch.isfinite(result).all()) and bool(result.ge(0.0).all()), "accepted q violates positivity")
    return result, {
        "role": role,
        "fold": fold,
        "replicate": replicate,
        "K": k,
        "seed": seed,
        "rng_namespace": namespace,
        "rng": "torch.Generator(device='cpu')",
        "target_accepted": target_accepted,
        "proposal_budget": proposal_budget,
        "proposal_batch_size": batch_size,
        "proposal_count": proposals,
        "accepted_count": accepted_count,
        "rejected_count": proposals - accepted_count,
        "acceptance_fraction": float(accepted_count / proposals) if proposals else 0.0,
        "termination": termination,
        "feasible": accepted_count == target_accepted,
    }


def _positivity_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    fractions = [float(record["acceptance_fraction"]) for record in records]
    return {
        "replicate_count": len(records),
        "all_feasible": all(bool(record["feasible"]) for record in records),
        "acceptance_fraction_mean": float(sum(fractions) / len(fractions)) if fractions else None,
        "acceptance_fraction_min": float(min(fractions)) if fractions else None,
        "acceptance_fraction_max": float(max(fractions)) if fractions else None,
        "proposal_count_total": int(sum(int(record["proposal_count"]) for record in records)),
        "accepted_count_total": int(sum(int(record["accepted_count"]) for record in records)),
    }


def _evaluate_fold(context: Mapping[str, Any], *, fold: int, k: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    shared = context["shared"]
    shared_summary = {
        "fit_count": int(context["whitened"].shape[0]),
        "mu_sha256": tensor_sha256(shared["mu"]),
        "sigma_sha256": tensor_sha256(shared["sigma"]),
        "alpha": float(shared["alpha"]),
        "tau": float(shared["tau"]),
        "sigma_trace": float(torch.trace(shared["sigma"])),
        "sigma_min_eigenvalue": float(torch.linalg.eigvalsh(shared["sigma"]).min()),
    }
    try:
        fitted = _fit_mixture(context, fold=fold, k=k, settings=settings)
    except CandidateFoldFailure as exc:
        return {
            "fold": fold,
            "K": k,
            "operationally_valid": False,
            "compatibility": False,
            "fit": {"shared_covariance": shared_summary, "failure": str(exc)},
            "sampling": [],
            "reason": "CANDIDATE_FOLD_SCIENTIFIC_FAILURE",
        }

    heldout = context["heldout"]
    namespace = str(settings["rng_namespace"])
    sampling_records: list[dict[str, Any]] = []
    data_metrics: list[dict[str, float]] = []
    data_sampling: list[dict[str, Any]] = []
    for replicate in range(int(settings["data_vs_model_replicates"])):
        model_q, sampling = _sample_positive(
            fitted,
            namespace=namespace,
            k=k,
            role="data-vs-model",
            fold=fold,
            replicate=replicate,
            settings=settings,
        )
        sampling_records.append(sampling)
        data_sampling.append(sampling)
        if sampling["feasible"]:
            data_metrics.append(comparison._pair_metrics(heldout, model_q))

    model_banks: list[torch.Tensor] = []
    model_sampling: list[dict[str, Any]] = []
    for replicate in range(int(settings["model_vs_model_replicates"])):
        model_q, sampling = _sample_positive(
            fitted,
            namespace=namespace,
            k=k,
            role="model-vs-model",
            fold=fold,
            replicate=replicate,
            settings=settings,
        )
        sampling_records.append(sampling)
        model_sampling.append(sampling)
        if sampling["feasible"]:
            model_banks.append(model_q)

    operationally_valid = (
        len(data_metrics) == int(settings["data_vs_model_replicates"])
        and len(model_banks) == int(settings["model_vs_model_replicates"])
    )
    fit_summary = {
        "shared_covariance": shared_summary,
        "mixture": fitted["summary"],
    }
    if not operationally_valid:
        return {
            "fold": fold,
            "K": k,
            "operationally_valid": False,
            "compatibility": False,
            "fit": fit_summary,
            "sampling": sampling_records,
            "positivity": {
                "data_vs_model": _positivity_summary(data_sampling),
                "model_vs_model": _positivity_summary(model_sampling),
            },
            "reason": "FIXED_POSITIVITY_BANK_NOT_REACHED",
        }

    reference_pairs = [
        comparison._pair_metrics(model_banks[index], model_banks[index + 1])
        for index in range(0, int(settings["model_vs_model_replicates"]), 2)
    ]
    data_mean = comparison._mean_metrics(data_metrics)
    reference_q95 = comparison._quantile_metrics(reference_pairs, float(settings["compatibility_q"]))
    guardrail_names = (
        "energy_distance",
        "covariance_relative_frobenius",
        "norm_energy_distance",
        "squared_norm_energy_distance",
    )
    compatibility_components = {
        name: data_mean[name] <= reference_q95[name] for name in guardrail_names
    }
    return {
        "fold": fold,
        "K": k,
        "operationally_valid": True,
        "compatibility": all(compatibility_components.values()),
        "fit": fit_summary,
        "sampling": sampling_records,
        "positivity": {
            "data_vs_model": _positivity_summary(data_sampling),
            "model_vs_model": _positivity_summary(model_sampling),
        },
        "data_vs_model_replicates": data_metrics,
        "data_vs_model_mean": data_mean,
        "model_vs_model_reference_pairs": reference_pairs,
        "model_vs_model_reference_q95": reference_q95,
        "compatibility_rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary plus covariance/norm guardrails",
        "compatibility_components": compatibility_components,
    }


def _evaluate_k(contexts: list[Mapping[str, Any]], *, k: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    folds = [
        _evaluate_fold(context, fold=fold, k=k, settings=settings)
        for fold, context in enumerate(contexts)
    ]
    valid_folds = [item for item in folds if item["operationally_valid"]]
    primary_values = [item["data_vs_model_mean"]["energy_distance"] for item in valid_folds]
    covariance_values = [item["data_vs_model_mean"]["covariance_relative_frobenius"] for item in valid_folds]
    norm_values = [
        item["data_vs_model_mean"]["norm_energy_distance"]
        + item["data_vs_model_mean"]["squared_norm_energy_distance"]
        for item in valid_folds
    ]
    return {
        "K": k,
        "operationally_valid": len(valid_folds) == FOLD_COUNT,
        "compatible": len(valid_folds) == FOLD_COUNT and all(bool(item["compatibility"]) for item in folds),
        "compatible_fold_count": sum(bool(item["compatibility"]) for item in folds),
        "operationally_valid_fold_count": len(valid_folds),
        "primary_mean": float(sum(primary_values) / len(primary_values)) if primary_values else None,
        "covariance_mean": float(sum(covariance_values) / len(covariance_values)) if covariance_values else None,
        "norm_guardrail_mean": float(sum(norm_values) / len(norm_values)) if norm_values else None,
        "folds": folds,
    }


def _select_model_order(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    compatible = [int(item["K"]) for item in candidates if bool(item["compatible"])]
    if compatible:
        selected = min(compatible)
        return {
            "classification": CLASSIFICATION_SELECTED,
            "K_selected": selected,
            "compatible_K": sorted(compatible),
            "rationale": "smallest K satisfies all frozen-fold compatibility checks",
        }
    if all(bool(item["operationally_valid"]) for item in candidates):
        return {
            "classification": CLASSIFICATION_NOT_SUPPORTED,
            "K_selected": None,
            "compatible_K": [],
            "rationale": "no frozen K satisfies all five-fold compatibility checks",
        }
    return {
        "classification": CLASSIFICATION_INCONCLUSIVE,
        "K_selected": None,
        "compatible_K": [],
        "rationale": "at least one candidate/fold did not yield a complete valid measurement, so no deterministic K decision is available",
    }


def _write_report(repo_root: Path, output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(temporary / REPORT_NAME, report, artifact_kind="N3 local-mixture selection report")
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
    _require(git["worktree_clean"], "consumer must start from a clean committed state")
    contract = _load_contract(repo_root)
    authorities, boundaries = _verify_input_authorities(repo_root, contract)
    q, source_indices, labels, fold_ids, pool_identity, fold_identity = _load_development(repo_root, contract)
    contexts = [
        _prepare_fold(q, source_indices, fold_ids, fold)
        for fold in range(FOLD_COUNT)
    ]

    contract_sampling = _mapping(contract["sampling_contract"], "contract sampling")
    contract_fit = _mapping(contract["fit_contract"], "contract fit")
    contract_collapse = _mapping(contract["component_collapse_contract"], "contract collapse")
    contract_compatibility = _mapping(contract["compatibility_contract"], "contract compatibility")
    settings = {
        "candidate_K": [int(value) for value in contract["component_count_domain"]["candidate_K"]],
        "max_iterations": int(contract_fit["max_iterations"]),
        "convergence_tolerance": float(contract_fit["convergence_tolerance"]),
        "monotonicity_tolerance_per_row": float(contract_fit["monotonicity_tolerance_per_row"]),
        "mass_floor": float(contract_collapse["mass_floor"]),
        "rng_namespace": str(contract_sampling["rng_namespace"]),
        "target_accepted": int(contract_sampling["target_accepted"]),
        "proposal_budget": int(contract_sampling["proposal_budget"]),
        "proposal_batch_size": int(contract_sampling["proposal_batch_size"]),
        "data_vs_model_replicates": int(contract_sampling["data_vs_model_replicates"]),
        "model_vs_model_replicates": int(contract_sampling["model_vs_model_replicates"]),
        "compatibility_q": float(contract_compatibility["q"]),
    }
    candidates = [
        _evaluate_k(contexts, k=k, settings=settings)
        for k in settings["candidate_K"]
    ]
    selection = _select_model_order(candidates)
    script_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": selection["classification"],
        "selection": selection,
        "git": {
            **git,
            "consumer_script": _relative(repo_root, script_path),
            "consumer_script_sha256": file_sha256(script_path),
        },
        "contract": {
            "path": _relative(repo_root, repo_root / CONTRACT_PATH),
            "sha256": EXPECTED_CONTRACT_SHA256,
            "classification": contract["classification"],
            "claim_classification": contract["claim_classification"],
            "starting_committed_head": contract["git"]["starting_committed_head"],
        },
        "authorities": authorities,
        "historical_boundaries": boundaries,
        "development_pool": {
            **pool_identity,
            "labels_as_model_inputs": False,
            "q_values_used_for_fit_and_evaluation": True,
        },
        "development_fold_identity": fold_identity,
        "execution_settings": {
            **settings,
            "seed_derivation": SEED_DERIVATION,
            "fold_count": FOLD_COUNT,
            "dimension": DIMENSION,
            "fit_rows_per_fold": 800,
            "heldout_rows_per_fold": 200,
            "labels_as_model_inputs": False,
            "shared_covariance_fixed_during_EM": True,
            "no_random_restarts": True,
        },
        "candidate_K_results": candidates,
        "claim_boundary": {
            "development_contract_supported_or_failed": True,
            "selected_K_is_development_only": selection["K_selected"] is not None,
            "independent_confirmation": False,
            "true_number_of_OLS_components": False,
            "semantic_component_interpretation": False,
            "full_complex_OLS_distribution": False,
            "novel_OLS_validity": False,
            "generation_readiness": "not_ready",
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "selection_scope": "DEVELOPMENT_ONLY",
        },
        "sealed_confirmation": {
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "materialized": False,
            "sample_values_loaded": False,
            "future_use_requires_separate_frozen_producer_confirmation_contract": True,
        },
        "not_executed": [
            "final all-1000-row producer fit",
            "sealed confirmation source loading or materialization",
            "independent confirmation",
            "selector V3",
            "label-conditioned producer",
            "component-specific covariance",
            "local rank/MFA",
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
    report_path = _write_report(repo_root, output_dir, report)
    return {
        "classification": report["classification"],
        "K_selected": selection["K_selected"],
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "worktree_clean_at_run": git["worktree_clean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "n3_local_location_mixture_selection_v1_attempt_01"
        ),
        help="fresh write-once selection output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
