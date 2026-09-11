"""Define the delta-only V2 repair for the N3 minimal-intervention contract.

This definition consumer supersedes only Candidate M's invalid untruncated
log1p-Gaussian support semantics.  It binds the immutable V1 contract and its
INVALID_HARNESS report, defines Candidate M2 as a zero-truncated marginal,
and proves the support boundary with deterministic synthetic scalar tests.
It never loads q tensors, fits either candidate, samples scientific banks, or
reads sealed confirmation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256
from scripts.research import define_n3_oracle_guided_minimal_intervention_contract_v1 as base


TASK_ID = "DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_oracle_guided_minimal_intervention_contract.v2"
READY_CLASSIFICATION = "DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_READY"
CLAIM_CLASSIFICATION = "N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_DEFINED"
EXPECTED_BRANCH = base.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = base.EXPECTED_STARTING_HEAD

BASE_CONTRACT = base.OUTPUT_DIR / base.CONTRACT_NAME
BASE_CONTRACT_SHA256 = "d640335c298afed696c64250acd15ed4767e8222a8e962f54fe6cd05c8a0f6b1"
INVALID_HARNESS_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_oracle_guided_minimal_intervention_comparison_v1_attempt_01/"
    "n3_oracle_guided_minimal_intervention_comparison_report.v1.json"
)
INVALID_HARNESS_REPORT_SHA256 = "ff8404e4c6b26b66d9292e728e30031cc5b14b7c1d1396d5713e6a2bdae77c17"

OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_oracle_guided_minimal_intervention_contract_v2_attempt_01"
)
CONTRACT_NAME = "n3_oracle_guided_minimal_intervention_contract.v2.json"

CLASS_COUNT = base.CLASS_COUNT
DIMENSION = base.DIMENSION
POOL_COUNT = base.POOL_COUNT
FOLD_COUNT = base.FOLD_COUNT
FIT_COUNT_PER_FOLD = base.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = base.HELDOUT_COUNT_PER_FOLD
GENERATED_BANK_SIZE = base.GENERATED_BANK_SIZE
DATA_VS_MODEL_REPLICATES = base.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = base.MODEL_VS_MODEL_REPLICATES
MODEL_REFERENCE_PAIRS = base.MODEL_REFERENCE_PAIRS
COMPATIBILITY_Q = base.COMPATIBILITY_Q
FOLD_ASSIGNMENT_SHA256 = base.FOLD_ASSIGNMENT_SHA256
RNG_NAMESPACE = base.RNG_NAMESPACE
STUDENT_T_DEGREES_OF_FREEDOM = base.STUDENT_T_DEGREES_OF_FREEDOM
BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT = base.BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT

CANDIDATE_M2 = "M2"
CANDIDATE_D = base.CANDIDATE_D
CANDIDATE_M2_FAMILY = "ZERO_TRUNCATED_LOG1P_GAUSSIAN_MARGINAL_V1"
CANDIDATE_D_FAMILY = base.CANDIDATE_D_FAMILY

Q_ROUNDOFF_TOLERANCE = 1.0e-12
Y_ROUNDOFF_TOLERANCE = 1.0e-12
SIGMA_DEGENERACY_THRESHOLD = 0.0
NORMAL = NormalDist()

SYNTHETIC_CASES = (
    {"name": "lower_boundary", "mu": 0.0, "sigma": 1.0, "u": math.nextafter(0.0, 1.0)},
    {"name": "near_lower_negative_mu", "mu": -2.0, "sigma": 1.0, "u": 1.0e-12},
    {"name": "interior_positive_mu", "mu": 2.0, "sigma": 0.5, "u": 0.5},
    {"name": "near_upper_negative_mu", "mu": -1.0, "sigma": 1.5, "u": 1.0 - 1.0e-12},
    {"name": "narrow_scale", "mu": 0.25, "sigma": 0.125, "u": 1.0 - 1.0e-6},
)

SAFE_DIRTY_PATHS = set(base.SAFE_DIRTY_PATHS) | {
    "scripts/research/execute_n3_oracle_guided_minimal_intervention_comparison_v1.py",
    "tests/v2/test_execute_n3_oracle_guided_minimal_intervention_comparison_v1.py",
    "scripts/research/define_n3_oracle_guided_minimal_intervention_contract_v2.py",
    "tests/v2/test_n3_oracle_guided_minimal_intervention_contract_v2.py",
}


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
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return dict(value)


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


def _normal_cdf(value: float) -> float:
    result = NORMAL.cdf(value)
    _require(math.isfinite(result), "synthetic normal CDF is non-finite")
    return result


def _normal_inverse(value: float) -> float:
    _require(math.isfinite(value) and 0.0 < value < 1.0, "synthetic normal inverse input is invalid")
    result = NORMAL.inv_cdf(value)
    _require(math.isfinite(result), "synthetic normal inverse result is non-finite")
    return result


def _m2_inverse_quantile(mu: float, sigma: float, u: float) -> dict[str, Any]:
    """Evaluate the fully specified zero-truncated log1p inverse CDF."""

    _require(math.isfinite(mu), "M2 mu is non-finite")
    _require(math.isfinite(sigma) and sigma > SIGMA_DEGENERACY_THRESHOLD, "M2 sigma is invalid")
    _require(math.isfinite(u) and 0.0 < u < 1.0, "M2 uniform input is not strictly interior")
    boundary_standard = (0.0 - mu) / sigma
    _require(math.isfinite(boundary_standard), "M2 truncation boundary standardization is non-finite")
    alpha = _normal_cdf(boundary_standard)
    _require(math.isfinite(alpha) and 0.0 <= alpha < 1.0, "M2 alpha is outside its defined numeric domain")
    p = alpha + (1.0 - alpha) * u
    _require(math.isfinite(p) and 0.0 < p < 1.0, "M2 inverse-CDF probability is not strictly interior")
    _require(p >= alpha, "M2 inverse-CDF probability crossed the truncation boundary")
    latent_standard = _normal_inverse(p)
    y_raw = mu + sigma * latent_standard
    _require(math.isfinite(y_raw), "M2 y output is non-finite")
    try:
        q_raw = math.expm1(y_raw)
    except (OverflowError, ValueError) as error:
        raise ValueError("M2 q output is not representable") from error
    _require(math.isfinite(q_raw), "M2 q output is non-finite")

    canonicalized = False
    q = q_raw
    if q_raw < 0.0:
        _require(
            q_raw >= -Q_ROUNDOFF_TOLERANCE and y_raw >= -Y_ROUNDOFF_TOLERANCE,
            "M2 q output is materially negative",
        )
        q = 0.0
        canonicalized = True
    _require(math.isfinite(q) and q >= 0.0, "M2 q output violates support")
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "u": float(u),
        "alpha": float(alpha),
        "p": float(p),
        "y_raw": float(y_raw),
        "q_raw": float(q_raw),
        "q": float(q),
        "canonicalized_roundoff": canonicalized,
        "support_q_nonnegative": True,
        "finite": True,
    }


def _run_synthetic_support_tests() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in SYNTHETIC_CASES:
        first = _m2_inverse_quantile(case["mu"], case["sigma"], case["u"])
        second = _m2_inverse_quantile(case["mu"], case["sigma"], case["u"])
        _require(first == second, f"synthetic case is not deterministic: {case['name']}")
        _require(first["support_q_nonnegative"] is True, f"synthetic support failed: {case['name']}")
        _require(first["finite"] is True, f"synthetic finiteness failed: {case['name']}")
        _require(first["p"] >= first["alpha"], f"synthetic truncation ordering failed: {case['name']}")
        _require(first["y_raw"] >= -Y_ROUNDOFF_TOLERANCE, f"synthetic y support failed: {case['name']}")
        cases.append(
            {
                "name": case["name"],
                "inputs": {"mu": case["mu"], "sigma": case["sigma"], "u": case["u"]},
                "output": first,
                "deterministic_repeated_evaluation": True,
                "support_invariant_verified": True,
            }
        )
    _require(cases[0]["output"]["q"] <= Q_ROUNDOFF_TOLERANCE, "lower boundary case did not reach q=0 boundary")
    return {
        "status": "passed",
        "case_count": len(cases),
        "cases": cases,
        "uses_development_q": False,
        "uses_sealed_confirmation": False,
        "uses_candidate_fit_or_sampling": False,
    }


def _validate_invalid_harness_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema") == "n3_oracle_guided_minimal_intervention_comparison_report.v1", "invalid-harness report schema changed")
    _require(report.get("task_id") == "EXECUTE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_COMPARISON_V1", "invalid-harness report task changed")
    _require(report.get("status") == "blocked" and report.get("result") == "BLOCKED", "invalid-harness report status changed")
    _require(report.get("classification") is None, "invalid-harness report emitted a classification")
    _require(report.get("scientific_classification_emitted") is False, "invalid-harness report emitted scientific evidence")
    blocker = _mapping(report.get("blocker"), "invalid-harness blocker")
    _require(blocker.get("type") == "INVALID_HARNESS", "invalid-harness blocker type changed")
    details = _mapping(blocker.get("details"), "invalid-harness details")
    for key, expected in {
        "candidate": "M",
        "role": "data-vs-model",
        "fold": 0,
        "replicate": 0,
        "invalid_count": 351,
        "negative_count": 351,
        "nonfinite_count": 0,
        "q_min": -0.42263710721584474,
        "q_max": 7.528838701593836,
        "failure_semantics": "INVALID_HARNESS",
        "contract_action": "stop before scientific compatibility or selection",
    }.items():
        _require(details.get(key) == expected, f"invalid-harness detail changed: {key}")
    report_contract = _mapping(report.get("contract"), "invalid-harness report contract")
    _require(report_contract.get("sha256") == BASE_CONTRACT_SHA256, "invalid-harness report base contract changed")
    _require(report.get("candidate_results") is None, "invalid-harness report contains partial candidate results")
    comparison = _mapping(report.get("comparison"), "invalid-harness comparison")
    _require(comparison.get("no_partial_candidate_selection") is True, "invalid-harness report contains partial selection")
    _require(comparison.get("selection_applied") is False, "invalid-harness report applied selection")
    selection = _mapping(report.get("selection"), "invalid-harness selection")
    _require(selection.get("classification") is None and selection.get("selected_candidate") is None, "invalid-harness report selected a candidate")
    _require(selection.get("selection_applied") is False, "invalid-harness report applied candidate selection")
    governance = _mapping(report.get("governance"), "invalid-harness governance")
    for key, expected in {
        "comparison_executed": False,
        "candidate_selected": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"invalid-harness governance changed: {key}")


def _validate_v1_authority(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    v1 = _read_json_exact(repo_root, BASE_CONTRACT, BASE_CONTRACT_SHA256, "V1 minimal-intervention contract")
    _require(v1.get("schema") == base.CONTRACT_SCHEMA, "V1 contract schema changed")
    _require(v1.get("task_id") == base.TASK_ID, "V1 contract task changed")
    _require(v1.get("classification") == base.READY_CLASSIFICATION, "V1 contract is not ready")
    _require(v1.get("claim_classification") == base.CLAIM_CLASSIFICATION, "V1 contract claim changed")
    _require(v1.get("definition_only") is True, "V1 contract definition boundary changed")
    git = _mapping(v1.get("git"), "V1 contract git")
    _require(git.get("branch") == EXPECTED_BRANCH, "V1 contract branch changed")
    _require(git.get("starting_committed_head") == EXPECTED_STARTING_HEAD, "V1 contract starting HEAD changed")
    _require(git.get("contract_script_sha256") == file_sha256(Path(base.__file__)), "V1 contract script hash changed")
    _require(v1.get("candidate_M", {}).get("id") == base.CANDIDATE_M, "V1 Candidate M identity changed")
    _require(v1.get("candidate_D", {}).get("id") == base.CANDIDATE_D, "V1 Candidate D identity changed")
    _require(v1.get("complexity", {}).get("candidate_M", {}).get("total_effective_fitted_scalar_count") == 5250, "V1 M complexity changed")
    _require(v1.get("complexity", {}).get("candidate_D", {}).get("total_effective_fitted_scalar_count") == 85050, "V1 D complexity changed")
    _require(v1.get("complexity", {}).get("preferred_candidate_if_both_supported") == base.CANDIDATE_M, "V1 tie-break changed")
    _require(v1.get("execution_boundary", {}).get("q_tensor_loaded_by_this_definition_consumer") is False, "V1 definition loaded q")
    base._validate_oracle_inputs(repo_root)
    base._validate_attribution_inputs(repo_root)
    base._validate_baseline_inputs(repo_root)
    report = _read_json_exact(repo_root, INVALID_HARNESS_REPORT, INVALID_HARNESS_REPORT_SHA256, "V1 invalid-harness report")
    _validate_invalid_harness_report(report)
    return v1, report


def _candidate_m2(v1_m: Mapping[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(v1_m))
    candidate["id"] = CANDIDATE_M2
    candidate["family"] = CANDIDATE_M2_FAMILY
    candidate["changed_assumption"] = (
        "For each coordinate j, Y_j=log1p(q_j) follows the V1 fitted Gaussian location-scale "
        "law conditioned on the fixed support boundary Y_j>=0; the joint dependence remains "
        "the frozen Gaussian copula."
    )
    candidate["parameterization"] = {
        **copy.deepcopy(dict(v1_m["parameterization"])),
        "distribution": "Y_j~Normal(mu_j,sigma_j^2) conditioned on Y_j>=0",
        "coordinate_support": "q_j>=0 (equivalently y_j>=0)",
        "cdf": "F_M2,j(q)=[Phi((log1p(q)-mu_j)/sigma_j)-alpha_j]/(1-alpha_j) for q>=0; F_M2,j(q)=0 for q<0",
        "quantile": "Q_M2,j(u)=expm1(mu_j+sigma_j*Phi^-1(alpha_j+(1-alpha_j)u)) for 0<u<1",
        "truncation_boundary": "y_j=0, fixed family constraint, not fitted",
        "truncation_probability": "alpha_j=Phi((0-mu_j)/sigma_j)",
        "inverse_probability": "p_j=alpha_j+(1-alpha_j)u_j",
        "support_proof": (
            "For sigma_j>0 and u_j in (0,1), p_j>=alpha_j and Phi^-1 is monotone, so "
            "y_j=mu_j+sigma_j*Phi^-1(p_j)>=mu_j+sigma_j*Phi^-1(alpha_j)=0; expm1 is "
            "monotone, hence q_j>=0 by construction."
        ),
        "fixed_truncation_fitted_scalar_count": 0,
        "zero_value_semantics": "q_j=0 maps to y_j=0 exactly; the support boundary is fixed and no positive offset is fitted",
    }
    candidate["sampling"] = {
        **copy.deepcopy(dict(v1_m["sampling"])),
        "law": "z_new~Normal(0,R_shrunk); u_new,j=Phi(z_new,j); alpha_j=Phi(-mu_j/sigma_j); p_j=alpha_j+(1-alpha_j)u_new,j; y_new,j=mu_j+sigma_j Phi^-1(p_j); q_new,j=expm1(y_new,j)",
        "support_boundary": "y_new,j>=0 and q_new,j>=0",
        "support_by_construction": True,
        "roundoff_canonicalization": {
            "raw_q_symbol": "q_raw=expm1(y_raw)",
            "q_absolute_tolerance": Q_ROUNDOFF_TOLERANCE,
            "y_absolute_tolerance": Y_ROUNDOFF_TOLERANCE,
            "condition": "only when q_raw<0, q_raw>=-1e-12 and y_raw>=-1e-12",
            "action": "set q=0.0 and record canonicalized_roundoff=true",
            "material_negative_action": "INVALID_HARNESS",
            "not_a_distributional_repair": True,
        },
        "sigma_degeneracy_threshold": SIGMA_DEGENERACY_THRESHOLD,
        "alpha_numeric_domain": "alpha finite and 0<=alpha<1; alpha=1 is INVALID_HARNESS because p cannot be guaranteed strictly interior",
        "p_numeric_domain": "p finite, p>=alpha and strictly inside (0,1); no clamp",
    }
    candidate["numerical_failure_semantics"] = {
        **copy.deepcopy(dict(v1_m["numerical_failure_semantics"])),
        "required_transform": "log1p(q_fit), V1 mu_j and sigma_j, alpha_j=Phi(-mu_j/sigma_j), p_j=alpha_j+(1-alpha_j)u_j, Phi^-1(p_j), y_j and q_raw=expm1(y_j) all satisfy their finite-domain checks",
        "strict_scale": "sigma_j finite and strictly greater than the V1 registered degeneracy threshold 0.0; no additional threshold or fitted truncation parameter",
        "required_alpha": "alpha_j finite and 0<=alpha_j<1",
        "strict_probability": "u_j and p_j finite and strictly between zero and one, with p_j>=alpha_j; no clamp",
        "required_y": "y_j finite and y_j>=-1e-12; a value below -1e-12 is INVALID_HARNESS",
        "strict_output": "q_raw finite; q_raw in [-1e-12,0) is canonicalized to exactly 0 only when y_j>=-1e-12; q_raw<-1e-12 is INVALID_HARNESS",
        "roundoff_policy": "explicit absolute float64 boundary policy: q tolerance=1e-12 and y tolerance=1e-12; no clamp, abs, ReLU, softplus, epsilon repair or proposal rejection",
    }
    candidate["unchanged_component"] = copy.deepcopy(dict(v1_m["unchanged_component"]))
    candidate["fitting"] = copy.deepcopy(dict(v1_m["fitting"]))
    return candidate


def _semantic_diff(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    unchanged_paths = [
        "authorities.oracle_diagnostic_contract",
        "authorities.oracle_diagnostic_report",
        "authorities.attribution_contract_v2",
        "authorities.attribution_report",
        "authorities.development_data",
        "authorities.fold_assignment",
        "authorities.implementation_sources",
        "authorities.original_producer_contract",
        "authorities.original_producer_report",
        "authorities.sealed_confirmation_exclusion",
        "oracle_basis",
        "baseline_frozen",
        "development_evaluation",
        "candidate_D",
        "component_isolation.candidate_D_changed_component",
        "component_isolation.candidate_D_frozen_component",
        "component_isolation.joint_marginal_dependence_family",
        "component_isolation.labels_conditioning",
        "component_isolation.component_scope_is_machine_defined",
        "component_isolation.cross_component_post_result_retuning",
        "complexity.counting_convention",
        "complexity.shared_baseline_dependence_matrix_unique_entries",
        "complexity.candidate_D",
        "complexity.post_result_subjective_score",
        "complexity.numerical_failure_surface_used_for_ordering",
        "one_shot_semantics",
        "execution_boundary",
        "governance",
    ]

    def value_at(payload: Mapping[str, Any], path: str) -> Any:
        value: Any = payload
        for part in path.split("."):
            value = value[part]
        return value

    for path in unchanged_paths:
        _require(value_at(v1, path) == value_at(v2, path), f"V2 changed frozen semantic path: {path}")
    selection_v1 = _mapping(v1["selection"], "V1 selection")
    selection_v2 = _mapping(v2["selection"], "V2 selection")
    for key in ("mapping_function", "mapping", "no_weighted_score", "at_most_one_candidate_selected", "invalid_harness_action"):
        _require(selection_v1[key] == selection_v2[key], f"V2 changed selection semantic: {key}")
    scope_v1 = _mapping(v1["scope"], "V1 scope")
    scope_v2 = _mapping(v2["scope"], "V2 scope")
    for key in (
        "selection_scope",
        "coordinate",
        "dimension",
        "exactly_one_family_per_candidate",
        "labels_as_model_inputs",
        "optical_forward",
        "evaluator",
        "image_or_generator_metric",
        "producer_family_selection_before_evaluation",
        "new_candidate_results_observed_by_definition_consumer",
    ):
        _require(scope_v1[key] == scope_v2[key], f"V2 changed scope semantic: {key}")
    _require(v2["candidate_M2"]["unchanged_component"] == v1["candidate_M"]["unchanged_component"], "V2 M2 dependence changed")
    _require(v2["candidate_M2"]["fitting"] == v1["candidate_M"]["fitting"], "V2 M2 fitting changed")
    return {
        "status": "passed",
        "unchanged_paths": unchanged_paths,
        "unchanged_selection_semantics": [
            "mapping_function",
            "mapping",
            "no_weighted_score",
            "at_most_one_candidate_selected",
            "invalid_harness_action",
        ],
        "unchanged_scope_semantics": [
            "selection_scope",
            "coordinate",
            "dimension",
            "labels_as_model_inputs",
            "optical_forward",
            "evaluator",
            "image_or_generator_metric",
            "producer_family_selection_before_evaluation",
        ],
        "allowed_changes": [
            "contract schema/task/classification/semantic authority version",
            "Candidate M renamed to M2 and replaced by the requested zero-truncated marginal family",
            "candidate id references in scope, complexity ordering and selection both-supported tie-break",
            "explicit V1 supersession and invalid-harness provenance bindings",
            "explicit numerical boundary/canonicalization policy and synthetic-support evidence",
        ],
        "candidate_D_byte_semantic_equivalence": True,
        "comparison_semantics_unchanged": True,
    }


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    v1: Mapping[str, Any],
    invalid_report: Mapping[str, Any],
    synthetic_support: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(v1))
    payload["schema"] = CONTRACT_SCHEMA
    payload["task_id"] = TASK_ID
    payload["gate"] = GATE
    payload["classification"] = READY_CLASSIFICATION
    payload["claim_classification"] = CLAIM_CLASSIFICATION
    payload["semantic_authority_id"] = TASK_ID
    payload["new_contract"] = True
    payload["definition_only"] = True
    payload["candidate_M2"] = _candidate_m2(v1["candidate_M"])
    del payload["candidate_M"]
    payload["candidate_D"] = copy.deepcopy(dict(v1["candidate_D"]))

    payload["base_v1_contract"] = {
        "path": _relative(repo_root, repo_root / BASE_CONTRACT),
        "schema": v1["schema"],
        "sha256": BASE_CONTRACT_SHA256,
        "classification": v1["classification"],
        "claim_classification": v1["claim_classification"],
        "immutable": True,
        "superseded_only_for": "Candidate M support semantics",
    }
    payload["invalid_harness_authority"] = {
        "path": _relative(repo_root, repo_root / INVALID_HARNESS_REPORT),
        "schema": invalid_report["schema"],
        "sha256": INVALID_HARNESS_REPORT_SHA256,
        "status": invalid_report["status"],
        "result": invalid_report["result"],
        "classification": invalid_report["classification"],
        "blocker": "INVALID_HARNESS",
        "candidate": "M",
        "candidate_D_executed": False,
        "scientific_classification_emitted": False,
        "immutable": True,
    }
    payload["supersession"] = {
        "status": "V2_DELTA_ONLY_REPAIR",
        "base_contract_sha256": BASE_CONTRACT_SHA256,
        "invalid_harness_report_sha256": INVALID_HARNESS_REPORT_SHA256,
        "supersedes_candidate": "M",
        "replacement_candidate": CANDIDATE_M2,
        "changed_component": "marginal support/inverse-CDF semantics only",
        "unchanged_components": ["Gaussian copula dependence", "Candidate D", "all comparison semantics", "all governance boundaries"],
        "v1_contract_immutable": True,
        "v1_invalid_harness_immutable": True,
        "no_development_outcomes_observed": True,
    }

    payload["scope"]["candidate_ids"] = [CANDIDATE_M2, CANDIDATE_D]
    payload["scope"]["candidate_family_count"] = 2
    payload["scope"]["v1_candidate_ids"] = [base.CANDIDATE_M, base.CANDIDATE_D]
    payload["scope"]["v2_delta"] = "M -> M2 zero-truncated support repair only"

    complexity = copy.deepcopy(dict(v1["complexity"]))
    complexity["candidate_M2"] = {
        **copy.deepcopy(dict(complexity["candidate_M"])),
        "fixed_truncation_boundary_scalars": 0,
        "fixed_family_constraint": "support boundary y=0; no fitted degree of freedom",
        "numerical_failure_surface": "log1p/mu-sigma/alpha-p-Phi-inverse-y-expm1 finite-domain checks with explicit 1e-12 boundary canonicalization; no distributional repair",
    }
    del complexity["candidate_M"]
    complexity["candidate_ordering_tuples"][CANDIDATE_M2] = [
        complexity["candidate_M2"]["total_effective_fitted_scalar_count"],
        complexity["candidate_M2"]["separately_fitted_component_count"],
        0,
        0,
        CANDIDATE_M2,
    ]
    del complexity["candidate_ordering_tuples"][base.CANDIDATE_M]
    complexity["preferred_candidate_if_both_supported"] = CANDIDATE_M2
    complexity["ordering_tuple"][-1] = "candidate_id in fixed order M2 before D"
    complexity["tie_break"] = "lexicographically smallest frozen ordering tuple; any exact residual tie resolves M2 before D"
    complexity["design_decision"] = "M2 is preferred over D only because 5250 < 85050; this is a complexity design decision, not a mechanism claim"
    payload["complexity"] = complexity

    selection = copy.deepcopy(dict(v1["selection"]))
    selection["valid_execution_precondition"] = "both Candidate M2 and Candidate D complete all five folds without INVALID_HARNESS"
    selection["both_supported_result"] = "SELECT_COMPLEXITY_PREFERRED_MINIMAL_INTERVENTION with selected_candidate=M2"
    payload["selection"] = selection

    payload["claim_boundary"]["not_established"] = [
        "Candidate M2 or Candidate D development support",
        "a selected producer family before one-shot comparison",
        "a valid copula or tail family in observed results",
        "generalization or population truth",
        "full complex OLS distribution",
        "sealed confirmation",
        "N4/N5/N6 support",
        "semantic generation",
        "generation readiness",
    ]
    payload["contract_totality"]["validated_domains"]["candidate_ids"] = [CANDIDATE_M2, CANDIDATE_D]
    payload["contract_totality"]["scientific_classification_requires"] = (
        "a separate later valid one-shot M2-vs-D development execution after every authority, identity, "
        "numerical and compatibility gate passes"
    )
    payload["not_executed"] = [
        "Candidate M2 fitting",
        "Candidate M2 scientific sampling",
        "Candidate D fitting",
        "Candidate D scientific sampling",
        "new candidate q development results",
        "candidate support classification",
        "candidate selection",
        "all-1000 final fit",
        "sealed confirmation loading or materialization",
        "complex OLS",
        "N4/N5/N6",
        "generator/image metric",
        "CUDA",
        "optimizer",
        "backward",
        "training",
        "formal promotion",
        "generation",
    ]
    payload["synthetic_support_tests"] = dict(synthetic_support)
    payload["git"] = {
        "branch": git["branch"],
        "starting_committed_head": git["head"],
        "contract_script": _relative(repo_root, Path(__file__).resolve()),
        "contract_script_sha256": file_sha256(Path(__file__).resolve()),
        "dirty_paths": git["dirty_paths"],
        "worktree_clean": git["worktree_clean"],
    }
    payload["artifact_policy"] = {
        **copy.deepcopy(dict(v1["artifact_policy"])),
        "definition_is_delta_only": True,
        "base_v1_contract_modified": False,
        "invalid_harness_report_modified": False,
        "synthetic_support_tests_embedded": True,
    }
    payload["governance"] = copy.deepcopy(dict(v1["governance"]))
    payload["semantic_diff"] = _semantic_diff(v1, payload)
    return payload


def _write_contract(repo_root: Path, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once contract output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / CONTRACT_NAME,
            payload,
            artifact_kind="N3 oracle-guided minimal intervention contract V2",
        )
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared during V2 definition: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / CONTRACT_NAME


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "starting committed HEAD changed")
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in SAFE_DIRTY_PATHS]
    _require(not unexpected_dirty, f"unexpected dirty paths before V2 contract definition: {unexpected_dirty}")
    _require(not output_dir.exists(), f"write-once V2 contract output already exists: {output_dir}")

    v1, invalid_report = _validate_v1_authority(repo_root)
    synthetic_support = _run_synthetic_support_tests()
    payload = _build_payload(repo_root, git, v1, invalid_report, synthetic_support)
    contract_path = _write_contract(repo_root, output_dir, payload)
    return {
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "semantic_authority": TASK_ID,
        "contract": _relative(repo_root, contract_path),
        "contract_sha256": file_sha256(contract_path),
        "base_v1_contract_sha256": BASE_CONTRACT_SHA256,
        "invalid_harness_report_sha256": INVALID_HARNESS_REPORT_SHA256,
        "git_branch": git["branch"],
        "git_starting_head": EXPECTED_STARTING_HEAD,
        "git_ending_head": _git_state(repo_root)["head"],
        "candidate_M2_fit_or_sampling": False,
        "candidate_D_fit_or_sampling": False,
        "new_candidate_results_observed": False,
        "q_tensors_loaded_as_tensors": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
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
