"""Execute the frozen N3 local-shrinkage-covariance mixture contract once.

This consumer reuses the previous local-location Stage-1 partition exactly,
then estimates one analytic-shrinkage covariance per frozen hard-assigned
component.  It is a development-only CPU consumer: it never reads the sealed
confirmation source, materializes OLS states, invokes optics/evaluation, or
uses CUDA.
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

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, tensor_sha256
from scripts.research import (
    compare_n3_minimal_competing_producer_baselines_v1 as comparison,
    execute_n3_local_location_mixture_selection_v1 as previous_selection,
)


TASK_ID = "EXECUTE_N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_SELECTION_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_local_shrinkage_covariance_mixture_selection_report.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"

CONTRACT_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract_v1_attempt_01/"
    "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1.json"
)
EXPECTED_CONTRACT_SHA256 = "2cb596abe61deeaa847b90c74654c5abe0e94aae18a6f252170f8149ad1c1550"
PARENT_CONTRACT_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_unsupervised_local_location_mixture_producer_contract_v1_attempt_01/"
    "n3_unsupervised_local_location_mixture_producer_contract.v1.json"
)
PARENT_CONTRACT_SHA256 = "38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449"
PARENT_SELECTION_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_local_location_mixture_selection_v1_attempt_01/"
    "n3_local_location_mixture_selection_report.v1.json"
)
PARENT_SELECTION_SHA256 = "5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080"
REPORT_NAME = "n3_local_shrinkage_covariance_mixture_selection_report.v1.json"

CLASSIFICATION_SELECTED = "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_MODEL_ORDER_SELECTED"
CLASSIFICATION_NOT_SUPPORTED = "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_NOT_SUPPORTED"
CLASSIFICATION_INCONCLUSIVE = "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE"
COMPONENT_INSUFFICIENT = "LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT"

FOLD_COUNT = 5
DIMENSION = 100
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
SEED_DERIVATION = "sha256(namespace|K|role|fold|replicate) first 64 bits masked to signed 63-bit"


CandidateFoldFailure = previous_selection.CandidateFoldFailure


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


def _load_parent_lineage(repo_root: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = previous_selection._load_contract(repo_root)
    _require(parent.get("schema") == "n3_unsupervised_local_location_mixture_producer_contract.v1", "parent contract schema changed")
    _require(parent.get("classification") == "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY", "parent contract is not ready")
    selection = _read_json_exact(repo_root, PARENT_SELECTION_PATH, PARENT_SELECTION_SHA256, "parent local-location selection")
    _require(selection.get("schema") == "n3_local_location_mixture_selection_report.v1", "parent selection schema changed")
    _require(selection.get("classification") == "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED", "parent local-location result changed")
    _require(selection.get("contract", {}).get("sha256") == PARENT_CONTRACT_SHA256, "parent selection contract identity changed")
    selection_summary = _mapping(selection.get("selection"), "parent selection")
    _require(selection_summary.get("K_selected") is None, "parent selection unexpectedly selected K")
    candidates = selection.get("candidate_K_results")
    _require(isinstance(candidates, list) and [item.get("K") for item in candidates] == [2, 4, 8], "parent candidate domain changed")
    _require(all(item.get("compatible") is False and item.get("compatible_fold_count") == 0 for item in candidates), "parent compatibility result changed")

    lineage = _mapping(contract.get("previous_local_location_lineage"), "previous local-location lineage")
    _require(lineage.get("contract_path") == _relative(repo_root, repo_root / PARENT_CONTRACT_PATH), "parent contract path changed")
    _require(lineage.get("contract_sha256") == PARENT_CONTRACT_SHA256, "parent contract hash changed")
    _require(lineage.get("selection_report_path") == _relative(repo_root, repo_root / PARENT_SELECTION_PATH), "parent selection path changed")
    _require(lineage.get("selection_report_sha256") == PARENT_SELECTION_SHA256, "parent selection hash changed")
    _require(lineage.get("selection_classification") == "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED", "parent selection classification changed")
    _require(lineage.get("selected_K") is None, "parent selected K changed")
    _require(lineage.get("fold_assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "parent fold lineage changed")
    return parent, selection


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(repo_root, CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "frozen local shrinkage-covariance contract")
    _require(contract.get("schema") == "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1", "contract schema changed")
    _require(contract.get("task_id") == "DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1", "contract task changed")
    _require(contract.get("gate") == contract["task_id"], "contract gate changed")
    _require(contract.get("classification") == "DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_READY", "contract is not ready")
    _require(contract.get("claim_classification") == "N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED", "contract claim boundary changed")

    domain = _mapping(contract.get("component_count_domain"), "component_count_domain")
    _require(domain.get("candidate_K") == [2, 4, 8], "candidate K domain changed")
    for key in ("post_result_K_addition", "post_result_reassignment_change", "post_result_covariance_estimator_change", "post_result_threshold_change"):
        _require(domain.get(key) is False, f"post-result change is permitted: {key}")

    development = _mapping(contract.get("development_identity"), "development_identity")
    _require(development.get("count") == 1000, "development pool count changed")
    fold_identity = _mapping(development.get("fold_assignment"), "contract fold assignment")
    _require(fold_identity.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "contract fold identity changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "contract fold count changed")
    _require(fold_identity.get("q_values_used") is False and fold_identity.get("ols_values_used") is False, "fold construction used producer values")

    stage1 = _mapping(contract.get("stage1_local_partition_contract"), "stage1_local_partition_contract")
    _require(stage1.get("parent_contract_sha256") == PARENT_CONTRACT_SHA256, "Stage-1 parent identity changed")
    _require(stage1.get("fit_rows_per_fold") == 800 and stage1.get("heldout_rows_per_fold") == 200, "Stage-1 split changed")
    _require(stage1.get("coordinate") == "R2_AMPLITUDE", "Stage-1 coordinate changed")
    _require(stage1.get("labels_as_inputs") is False and stage1.get("heldout_parameter_leakage") is False, "Stage-1 leakage boundary changed")
    _require(stage1.get("no_second_joint_EM") is True, "Stage-1 permits a second joint EM")
    _require("final stabilized E-step" in str(stage1.get("responsibilities")), "final E-step contract changed")

    assignment = _mapping(contract.get("hard_assignment_contract"), "hard_assignment_contract")
    _require(assignment.get("rule") == "a_i=argmax_k r_ik over final Stage-1 responsibilities", "assignment rule changed")
    _require(assignment.get("tie_rule") == "lowest canonical component index wins", "assignment tie rule changed")
    _require(assignment.get("labels_as_inputs") is False and assignment.get("fit_rows_only") is True, "assignment input boundary changed")
    _require(assignment.get("assignment_frozen_before_local_covariance") is True, "assignment is not frozen before Stage-2")
    _require(assignment.get("reassignment_after_local_covariance") is False, "Stage-2 reassignment is permitted")

    stage2 = _mapping(contract.get("stage2_local_covariance_contract"), "stage2_local_covariance_contract")
    expected_stage2 = {
        "insufficient_count_predicate": "n_k < 2",
        "insufficient_count_classification": COMPONENT_INSUFFICIENT,
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
        "heldout_parameter_leakage": False,
        "second_EM": False,
        "component_covariance_specific": True,
        "weight": "pi_k=n_k/800",
    }
    for key, expected in expected_stage2.items():
        _require(stage2.get(key) == expected, f"Stage-2 contract changed: {key}")

    sampling = _mapping(contract.get("sampling_contract"), "sampling_contract")
    expected_sampling = {
        "target_accepted": comparison.TARGET_ACCEPTED,
        "proposal_budget": comparison.PROPOSAL_BUDGET,
        "proposal_batch_size": comparison.PROPOSAL_BATCH_SIZE,
        "data_vs_model_replicates": comparison.DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": comparison.MODEL_VS_MODEL_REPLICATES,
        "rng_namespace": "N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_CPU",
    }
    for key, expected in expected_sampling.items():
        _require(sampling.get(key) == expected, f"sampling setting changed: {key}")
    _require(sampling.get("rng_device") == "cpu" and sampling.get("rng_generator") == "torch.Generator(device='cpu')", "sampling device changed")
    _require(sampling.get("legal_state") == "finite(q) and q>=0 elementwise", "positivity predicate changed")
    for key in ("clamp", "abs", "relu", "softplus", "epsilon_repair", "samplewise_normalization", "budget_adaptation"):
        _require(sampling.get(key) is False, f"sampling repair/adaptation is permitted: {key}")
    _require(sampling.get("seed_substreams") == ["K", "fold", "data-vs-model replicate", "model-vs-model replicate"], "RNG substreams changed")

    evaluation = _mapping(contract.get("development_evaluation_contract"), "development_evaluation_contract")
    _require(evaluation.get("fit_rows") == 800 and evaluation.get("heldout_rows") == 200, "evaluation split changed")
    _require(evaluation.get("labels_as_model_inputs") is False, "labels are permitted as model inputs")
    _require(evaluation.get("optical_forward") is False and evaluation.get("evaluator") is False, "optical/evaluator execution is permitted")
    _require(evaluation.get("sealed_confirmation") is False, "sealed confirmation is permitted")
    _require(evaluation.get("secondary_guardrails") == [
        "covariance_relative_frobenius",
        "norm_energy_distance",
        "squared_norm_energy_distance",
        "positivity_acceptance_fraction",
        "marginal_mean_l2",
        "marginal_std_l2",
    ], "secondary guardrails changed")

    compatibility = _mapping(contract.get("compatibility_contract"), "compatibility_contract")
    _require(float(compatibility.get("q")) == 0.95, "compatibility quantile changed")
    _require(compatibility.get("all_folds_required") is True and compatibility.get("own_model_reference_required") is True, "compatibility fold/reference rule changed")
    _require(compatibility.get("absolute_metric_thresholds") is False, "absolute thresholds were introduced")
    _require(compatibility.get("less_bad_than_failed_prior_family_is_sufficient") is False, "relative improvement can substitute for compatibility")
    _require("covariance/norm" in str(compatibility.get("rule")), "compatibility guardrails changed")

    complexity = _mapping(contract.get("complexity_selection_contract"), "complexity_selection_contract")
    _require(complexity.get("candidate_orders") == [2, 4, 8], "complexity candidate domain changed")
    _require(complexity.get("selection") == "smallest globally compatible K", "complexity selection rule changed")
    _require(complexity.get("none_compatible") == CLASSIFICATION_NOT_SUPPORTED, "not-supported classification changed")
    _require(complexity.get("mixed_without_deterministic_K") == CLASSIFICATION_INCONCLUSIVE, "inconclusive classification changed")
    _require(complexity.get("exactly_one_selected") == CLASSIFICATION_SELECTED, "selected classification changed")
    for key in ("label_based_selection", "image_based_selection", "generator_based_selection", "sealed_confirmation_based_selection"):
        _require(complexity.get(key) is False, f"forbidden selection source permitted: {key}")

    sealed = _mapping(contract.get("sealed_confirmation_firewall"), "sealed_confirmation_firewall")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    _require(sealed.get("sample_values_loaded") is False and sealed.get("confirmation_evaluation_authorized") is False, "sealed confirmation is permitted")
    _require(sealed.get("selection_source_exclusion") is True, "sealed source exclusion changed")
    governance = _mapping(contract.get("governance"), "governance")
    _require(governance.get("training_authorized") is False and governance.get("formal_promotion") is False, "governance authorization changed")

    contract_git = _mapping(contract.get("git"), "contract git")
    contract_script = Path(str(contract_git.get("contract_script")))
    if not contract_script.is_absolute():
        contract_script = repo_root / contract_script
    _require(contract_script.is_file(), "contract definition script is missing")
    _load_parent_lineage(repo_root, contract)
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


def _prepare_fold(q: torch.Tensor, source_indices: torch.Tensor, fold_ids: torch.Tensor, fold: int) -> dict[str, Any]:
    context = previous_selection._prepare_fold(q, source_indices, fold_ids, fold)
    context["q_fit"] = q.index_select(0, context["fit_positions"]).to(dtype=torch.float64).contiguous()
    return context


def hard_assignment_from_responsibilities(responsibilities: torch.Tensor) -> torch.Tensor:
    values = responsibilities.to(dtype=torch.float64).contiguous()
    _require(values.ndim == 2 and values.shape[1] >= 1, "responsibilities shape is invalid")
    _require(bool(torch.isfinite(values).all()) and bool(values.ge(0.0).all()), "responsibilities are invalid")
    _require(bool(torch.allclose(values.sum(dim=1), torch.ones(values.shape[0], dtype=torch.float64))), "responsibilities do not sum to one")
    # torch.argmax returns the first maximum, which is the lowest canonical index.
    return torch.argmax(values, dim=1).to(dtype=torch.int64).contiguous()


def component_insufficiency_status(assigned_count: int) -> str | None:
    return COMPONENT_INSUFFICIENT if int(assigned_count) < 2 else None


def _final_responsibilities(context: Mapping[str, Any], fitted: Mapping[str, Any]) -> torch.Tensor:
    log_density = previous_selection._log_density(
        context["whitened"], fitted["centers"], fitted["weights"], fitted["shared"]["chol"]
    )
    responsibilities = torch.softmax(log_density, dim=1).to(dtype=torch.float64).contiguous()
    _scientific_require(bool(torch.isfinite(responsibilities).all()), "final responsibilities are non-finite")
    _scientific_require(bool(torch.allclose(responsibilities.sum(dim=1), torch.ones(responsibilities.shape[0], dtype=torch.float64))), "final responsibilities do not sum to one")
    return responsibilities


def _component_assignment_summary(assignments: torch.Tensor, k: int) -> dict[str, Any]:
    counts = torch.bincount(assignments, minlength=k)
    _require(tuple(counts.shape) == (k,), "component assignment shape changed")
    return {
        "component_count": k,
        "assigned_counts": [int(value) for value in counts.tolist()],
        "assigned_weight_sum": int(counts.sum()),
        "hard_assignment_sha256": tensor_sha256(assignments),
        "insufficient_components": [index for index, value in enumerate(counts.tolist()) if int(value) < 2],
        "tie_rule": "lowest canonical component index wins",
    }


def _fit_local_components(q_fit: torch.Tensor, assignments: torch.Tensor, k: int) -> dict[str, Any]:
    _require(tuple(q_fit.shape) == (800, DIMENSION), "local covariance fit shape changed")
    counts = torch.bincount(assignments, minlength=k)
    _require(int(counts.sum()) == 800, "local assignment counts do not cover the fit rows")
    if bool(counts.lt(2).any()):
        raise CandidateFoldFailure(COMPONENT_INSUFFICIENT)

    components: list[dict[str, Any]] = []
    means: list[torch.Tensor] = []
    cholesky: list[torch.Tensor] = []
    for component in range(k):
        positions = (assignments == component).nonzero(as_tuple=False).reshape(-1)
        local_q = q_fit.index_select(0, positions).to(dtype=torch.float64).contiguous()
        fitted = comparison._fit_shrinkage(local_q)
        means.append(fitted["mu"])
        cholesky.append(fitted["chol"])
        eigenvalues = torch.linalg.eigvalsh(fitted["sigma"])
        _require(bool(torch.isfinite(eigenvalues).all()) and bool(eigenvalues.gt(0.0).all()), "local covariance spectrum is invalid")
        components.append(
            {
                "component": component,
                "assigned_count": int(counts[component]),
                "weight": float(counts[component].item() / 800.0),
                "mu_sha256": tensor_sha256(fitted["mu"]),
                "sample_covariance_unbiased_sha256": tensor_sha256(fitted["sample_covariance_unbiased"]),
                "sigma_sha256": tensor_sha256(fitted["sigma"]),
                "cholesky_sha256": tensor_sha256(fitted["chol"]),
                "shrinkage_alpha": float(fitted["alpha"]),
                "shrinkage_tau": float(fitted["tau"]),
                "sigma_trace": float(torch.trace(fitted["sigma"])),
                "sigma_min_eigenvalue": float(eigenvalues.min()),
                "finite": True,
                "positive_definite": True,
            }
        )
    weights = counts.to(dtype=torch.float64).div(800.0).contiguous()
    _require(bool(torch.isfinite(weights).all()) and bool(torch.allclose(weights.sum(), torch.ones((), dtype=torch.float64))), "local mixture weights are invalid")
    return {
        "weights": weights,
        "component_means": torch.stack(means, dim=0).contiguous(),
        "cholesky": torch.stack(cholesky, dim=0).contiguous(),
        "components": components,
        "summary": {
            "estimator": "existing analytic component-local shrinkage",
            "covariance_denominator": "n_k-1",
            "weight": "pi_k=n_k/800",
            "component_specific": True,
            "components": components,
            "weights_sum": float(weights.sum()),
        },
    }


def _seed(namespace: str, k: int, role: str, fold: int, replicate: int) -> int:
    material = f"{namespace}|K={k}|{role}|fold={fold}|replicate={replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_positive_local(
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
    means = fit["component_means"]
    cholesky = fit["cholesky"]
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    proposals = 0
    termination = "proposal_budget_exhausted"
    while proposals < proposal_budget and accepted_count < target_accepted:
        block_count = min(batch_size, proposal_budget - proposals)
        component_ids = torch.multinomial(weights, block_count, replacement=True, generator=generator)
        gaussian = torch.randn((block_count, DIMENSION), generator=generator, dtype=torch.float64)
        selected_means = means.index_select(0, component_ids)
        selected_cholesky = cholesky.index_select(0, component_ids)
        q = selected_means + torch.bmm(gaussian.unsqueeze(1), selected_cholesky.transpose(1, 2)).squeeze(1)
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
        stage1 = previous_selection._fit_mixture(context, fold=fold, k=k, settings=settings)
    except CandidateFoldFailure as exc:
        return {
            "fold": fold,
            "K": k,
            "operationally_valid": False,
            "compatibility": False,
            "stage1": {"shared_covariance": shared_summary, "failure": str(exc)},
            "reason": "STAGE1_CANDIDATE_FOLD_SCIENTIFIC_FAILURE",
        }

    responsibilities = _final_responsibilities(context, stage1)
    assignments = hard_assignment_from_responsibilities(responsibilities)
    assignment_summary = _component_assignment_summary(assignments, k)
    if assignment_summary["insufficient_components"]:
        return {
            "fold": fold,
            "K": k,
            "operationally_valid": False,
            "compatibility": False,
            "stage1": {"shared_covariance": shared_summary, "mixture": stage1["summary"]},
            "final_e_step": {
                "responsibilities_sha256": tensor_sha256(responsibilities),
                "row_count": int(responsibilities.shape[0]),
                "frozen": True,
            },
            "assignment": assignment_summary,
            "local_covariance": {"failure": COMPONENT_INSUFFICIENT},
            "reason": COMPONENT_INSUFFICIENT,
            "insufficient_components": assignment_summary["insufficient_components"],
        }

    local = _fit_local_components(context["q_fit"], assignments, k)
    namespace = str(settings["rng_namespace"])
    data_sampling: list[dict[str, Any]] = []
    data_metrics: list[dict[str, float]] = []
    for replicate in range(int(settings["data_vs_model_replicates"])):
        model_q, sampling = _sample_positive_local(
            local,
            namespace=namespace,
            k=k,
            role="data-vs-model",
            fold=fold,
            replicate=replicate,
            settings=settings,
        )
        data_sampling.append(sampling)
        if sampling["feasible"]:
            data_metrics.append(comparison._pair_metrics(context["heldout"], model_q))

    model_banks: list[torch.Tensor] = []
    model_sampling: list[dict[str, Any]] = []
    for replicate in range(int(settings["model_vs_model_replicates"])):
        model_q, sampling = _sample_positive_local(
            local,
            namespace=namespace,
            k=k,
            role="model-vs-model",
            fold=fold,
            replicate=replicate,
            settings=settings,
        )
        model_sampling.append(sampling)
        if sampling["feasible"]:
            model_banks.append(model_q)

    operationally_valid = (
        len(data_metrics) == int(settings["data_vs_model_replicates"])
        and len(model_banks) == int(settings["model_vs_model_replicates"])
    )
    base = {
        "fold": fold,
        "K": k,
        "stage1": {"shared_covariance": shared_summary, "mixture": stage1["summary"]},
        "final_e_step": {
            "responsibilities_sha256": tensor_sha256(responsibilities),
            "row_count": int(responsibilities.shape[0]),
            "frozen": True,
        },
        "assignment": assignment_summary,
        "local_covariance": local["summary"],
        "positivity": {
            "data_vs_model": _positivity_summary(data_sampling),
            "model_vs_model": _positivity_summary(model_sampling),
        },
        "sampling": data_sampling + model_sampling,
    }
    if not operationally_valid:
        return {
            **base,
            "operationally_valid": False,
            "compatibility": False,
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
    compatibility_components = {name: data_mean[name] <= reference_q95[name] for name in guardrail_names}
    return {
        **base,
        "operationally_valid": True,
        "compatibility": all(compatibility_components.values()),
        "data_vs_model_replicates": data_metrics,
        "data_vs_model_mean": data_mean,
        "model_vs_model_reference_pairs": reference_pairs,
        "model_vs_model_reference_q95": reference_q95,
        "compatibility_rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary plus covariance/norm guardrails",
        "compatibility_components": compatibility_components,
    }


def _evaluate_k(contexts: list[Mapping[str, Any]], *, k: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    folds = [_evaluate_fold(context, fold=fold, k=k, settings=settings) for fold, context in enumerate(contexts)]
    valid_folds = [item for item in folds if item["operationally_valid"]]
    primary_values = [item["data_vs_model_mean"]["energy_distance"] for item in valid_folds]
    covariance_values = [item["data_vs_model_mean"]["covariance_relative_frobenius"] for item in valid_folds]
    norm_values = [
        item["data_vs_model_mean"]["norm_energy_distance"] + item["data_vs_model_mean"]["squared_norm_energy_distance"]
        for item in valid_folds
    ]
    marginal_mean_values = [item["data_vs_model_mean"]["marginal_mean_l2"] for item in valid_folds]
    marginal_std_values = [item["data_vs_model_mean"]["marginal_std_l2"] for item in valid_folds]
    return {
        "K": k,
        "operationally_valid": len(valid_folds) == FOLD_COUNT,
        "compatible": len(valid_folds) == FOLD_COUNT and all(bool(item["compatibility"]) for item in folds),
        "compatible_fold_count": sum(bool(item["compatibility"]) for item in folds),
        "operationally_valid_fold_count": len(valid_folds),
        "primary_mean": float(sum(primary_values) / len(primary_values)) if primary_values else None,
        "covariance_mean": float(sum(covariance_values) / len(covariance_values)) if covariance_values else None,
        "norm_guardrail_mean": float(sum(norm_values) / len(norm_values)) if norm_values else None,
        "marginal_mean_l2_mean": float(sum(marginal_mean_values) / len(marginal_mean_values)) if marginal_mean_values else None,
        "marginal_std_l2_mean": float(sum(marginal_std_values) / len(marginal_std_values)) if marginal_std_values else None,
        "folds": folds,
    }


def _select_model_order(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    compatible = [int(item["K"]) for item in candidates if bool(item["compatible"])]
    if compatible:
        return {
            "classification": CLASSIFICATION_SELECTED,
            "K_selected": min(compatible),
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
        atomic_write_new_json(temporary / REPORT_NAME, report, artifact_kind="N3 local shrinkage-covariance mixture selection report")
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
    contexts = [_prepare_fold(q, source_indices, fold_ids, fold) for fold in range(FOLD_COUNT)]

    contract_sampling = _mapping(contract["sampling_contract"], "contract sampling")
    parent_contract = _read_json_exact(repo_root, PARENT_CONTRACT_PATH, PARENT_CONTRACT_SHA256, "parent contract")
    parent_fit = _mapping(parent_contract["fit_contract"], "parent fit")
    parent_collapse = _mapping(parent_contract["component_collapse_contract"], "parent collapse")
    contract_compatibility = _mapping(contract["compatibility_contract"], "contract compatibility")
    settings = {
        "candidate_K": [int(value) for value in contract["component_count_domain"]["candidate_K"]],
        "max_iterations": int(parent_fit["max_iterations"]),
        "convergence_tolerance": float(parent_fit["convergence_tolerance"]),
        "monotonicity_tolerance_per_row": float(parent_fit["monotonicity_tolerance_per_row"]),
        "mass_floor": float(parent_collapse["mass_floor"]),
        "rng_namespace": str(contract_sampling["rng_namespace"]),
        "target_accepted": int(contract_sampling["target_accepted"]),
        "proposal_budget": int(contract_sampling["proposal_budget"]),
        "proposal_batch_size": int(contract_sampling["proposal_batch_size"]),
        "data_vs_model_replicates": int(contract_sampling["data_vs_model_replicates"]),
        "model_vs_model_replicates": int(contract_sampling["model_vs_model_replicates"]),
        "compatibility_q": float(contract_compatibility["q"]),
    }
    candidates = [_evaluate_k(contexts, k=k, settings=settings) for k in settings["candidate_K"]]
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
            "contract_script_sha256_observed": file_sha256(Path(str(contract["git"]["contract_script"]))),
        },
        "previous_local_location_result": {
            "contract_sha256": PARENT_CONTRACT_SHA256,
            "selection_report_sha256": PARENT_SELECTION_SHA256,
            "classification": "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED",
            "selected_K": None,
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
            "stage1_shared_covariance_fixed_during_EM": True,
            "stage2_assignment_frozen_before_local_covariance": True,
            "stage2_second_EM": False,
            "no_random_restarts": True,
        },
        "candidate_K_results": candidates,
        "claim_boundary": {
            "development_contract_supported_or_failed": True,
            "selected_K_is_development_only": selection["K_selected"] is not None,
            "true_number_of_components": False,
            "semantic_component_interpretation": False,
            "independent_confirmation": False,
            "full_complex_OLS_distribution": False,
            "novel_OLS_validity": False,
            "generation_readiness": "not_ready",
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "selection_scope": "DEVELOPMENT_ONLY",
            "real_n3_development_selection_executed": True,
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
            "joint unrestricted full-covariance EM",
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
            "n3_local_shrinkage_covariance_mixture_selection_v1_attempt_01"
        ),
        help="fresh write-once selection output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
