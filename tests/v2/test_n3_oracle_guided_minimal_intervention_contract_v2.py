from __future__ import annotations

import json
from pathlib import Path

from ldm_project.v2.evidence import file_sha256
from scripts.research import define_n3_oracle_guided_minimal_intervention_contract_v1 as base
from scripts.research import define_n3_oracle_guided_minimal_intervention_contract_v2 as contract


EXPECTED_ARTIFACT_SHA256 = "f297a1ffbf84bec84a15525be97ba1a5fd897d32beefd0fe22138bff0012ce2e"


def _load_artifact() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / contract.OUTPUT_DIR / contract.CONTRACT_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _load_base_artifact() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / base.OUTPUT_DIR / base.CONTRACT_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_artifact_is_immutable_and_binds_v1_failure() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / contract.OUTPUT_DIR / contract.CONTRACT_NAME
    artifact = _load_artifact()

    assert file_sha256(path) == EXPECTED_ARTIFACT_SHA256
    assert artifact["schema"] == contract.CONTRACT_SCHEMA
    assert artifact["semantic_authority_id"] == contract.TASK_ID
    assert artifact["classification"] == contract.READY_CLASSIFICATION
    assert artifact["claim_classification"] == contract.CLAIM_CLASSIFICATION
    assert artifact["definition_only"] is True
    assert artifact["contract_totality"]["status"] == "passed"
    assert artifact["contract_totality"]["future_runtime_math_choices"] == []
    assert artifact["git"]["contract_script_sha256"] == file_sha256(Path(contract.__file__))

    assert artifact["base_v1_contract"]["sha256"] == contract.BASE_CONTRACT_SHA256
    assert artifact["base_v1_contract"]["immutable"] is True
    assert artifact["base_v1_contract"]["superseded_only_for"] == "Candidate M support semantics"
    assert artifact["invalid_harness_authority"]["sha256"] == contract.INVALID_HARNESS_REPORT_SHA256
    assert artifact["invalid_harness_authority"]["blocker"] == "INVALID_HARNESS"
    assert artifact["invalid_harness_authority"]["candidate_D_executed"] is False
    assert artifact["invalid_harness_authority"]["scientific_classification_emitted"] is False


def test_m2_support_formula_and_numerical_policy_are_exact() -> None:
    artifact = _load_artifact()
    v1 = _load_base_artifact()
    m2 = artifact["candidate_M2"]
    v1_m = v1["candidate_M"]

    assert artifact["scope"]["candidate_ids"] == [contract.CANDIDATE_M2, contract.CANDIDATE_D]
    assert m2["id"] == contract.CANDIDATE_M2
    assert m2["family"] == contract.CANDIDATE_M2_FAMILY
    assert m2["changed_component"] == "marginal representation only"
    assert m2["unchanged_component"] == v1_m["unchanged_component"]
    assert m2["fitting"] == v1_m["fitting"]

    parameters = m2["parameterization"]
    assert parameters["distribution"] == "Y_j~Normal(mu_j,sigma_j^2) conditioned on Y_j>=0"
    assert parameters["truncation_boundary"] == "y_j=0, fixed family constraint, not fitted"
    assert parameters["fixed_truncation_fitted_scalar_count"] == 0
    assert parameters["quantile"] == (
        "Q_M2,j(u)=expm1(mu_j+sigma_j*Phi^-1(alpha_j+(1-alpha_j)u)) for 0<u<1"
    )
    assert "hence q_j>=0 by construction" in parameters["support_proof"]
    assert m2["numerical_failure_semantics"]["invalid_status"] == "INVALID_HARNESS"
    assert m2["numerical_failure_semantics"]["fallbacks"] is False
    assert m2["sampling"]["support_by_construction"] is True
    assert m2["sampling"]["sigma_degeneracy_threshold"] == 0.0
    assert m2["sampling"]["roundoff_canonicalization"]["q_absolute_tolerance"] == 1.0e-12
    assert m2["sampling"]["roundoff_canonicalization"]["y_absolute_tolerance"] == 1.0e-12
    assert m2["sampling"]["roundoff_canonicalization"]["not_a_distributional_repair"] is True
    assert m2["sampling"]["clamp"] is False
    assert m2["sampling"]["proposal_rejection"] is False
    assert m2["sampling"]["rng_namespace"] == base.RNG_NAMESPACE
    assert artifact["complexity"]["candidate_M2"]["total_effective_fitted_scalar_count"] == 5250
    assert artifact["complexity"]["candidate_M2"]["fixed_truncation_boundary_scalars"] == 0
    assert artifact["complexity"]["candidate_D"]["total_effective_fitted_scalar_count"] == 85050
    assert artifact["complexity"]["preferred_candidate_if_both_supported"] == contract.CANDIDATE_M2
    assert artifact["complexity"]["design_decision"].startswith("M2 is preferred over D only because 5250 < 85050")


def test_candidate_d_and_comparison_semantics_are_unchanged() -> None:
    artifact = _load_artifact()
    v1 = _load_base_artifact()

    assert artifact["candidate_D"] == v1["candidate_D"]
    assert artifact["development_evaluation"] == v1["development_evaluation"]
    assert artifact["baseline_frozen"] == v1["baseline_frozen"]
    assert artifact["oracle_basis"] == v1["oracle_basis"]
    assert artifact["execution_boundary"] == v1["execution_boundary"]
    assert artifact["one_shot_semantics"] == v1["one_shot_semantics"]
    assert artifact["semantic_diff"]["status"] == "passed"
    assert artifact["semantic_diff"]["candidate_D_byte_semantic_equivalence"] is True
    assert artifact["semantic_diff"]["comparison_semantics_unchanged"] is True

    for key in ("mapping_function", "mapping", "no_weighted_score", "at_most_one_candidate_selected", "invalid_harness_action"):
        assert artifact["selection"][key] == v1["selection"][key]
    assert artifact["selection"]["valid_execution_precondition"] == (
        "both Candidate M2 and Candidate D complete all five folds without INVALID_HARNESS"
    )
    assert artifact["selection"]["both_supported_result"].endswith("selected_candidate=M2")
    assert artifact["complexity"]["candidate_ordering_tuples"]["M2"] < artifact["complexity"]["candidate_ordering_tuples"]["D"]


def test_synthetic_support_boundary_and_determinism_pass() -> None:
    artifact = _load_artifact()
    synthetic = artifact["synthetic_support_tests"]

    assert synthetic["status"] == "passed"
    assert synthetic["case_count"] == 5
    assert synthetic["uses_development_q"] is False
    assert synthetic["uses_sealed_confirmation"] is False
    assert synthetic["uses_candidate_fit_or_sampling"] is False
    assert {case["name"] for case in synthetic["cases"]} == {
        "lower_boundary",
        "near_lower_negative_mu",
        "interior_positive_mu",
        "near_upper_negative_mu",
        "narrow_scale",
    }
    assert all(case["deterministic_repeated_evaluation"] is True for case in synthetic["cases"])
    assert all(case["support_invariant_verified"] is True for case in synthetic["cases"])
    assert all(case["output"]["q"] >= 0.0 for case in synthetic["cases"])
    assert synthetic["cases"][0]["output"]["q"] == 0.0

    assert contract._run_synthetic_support_tests() == synthetic


def test_definition_consumer_has_no_candidate_result_or_tensor_path() -> None:
    source = Path(contract.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.load" not in source
    assert "safe_torch_load" not in source
    artifact = _load_artifact()
    boundary = artifact["execution_boundary"]
    assert boundary["q_tensor_loaded_by_this_definition_consumer"] is False
    assert boundary["candidate_M_fit"] is False
    assert boundary["candidate_M_sampling"] is False
    assert boundary["candidate_D_fit"] is False
    assert boundary["candidate_D_sampling"] is False
    assert boundary["new_candidate_results_observed"] is False
    assert boundary["new_candidate_q_results_read"] is False
    assert boundary["sealed_confirmation_read"] is False
    assert boundary["cuda"] is False
    assert boundary["training"] is False
    assert "Candidate M2 fitting" in artifact["not_executed"]
    assert "Candidate D scientific sampling" in artifact["not_executed"]


def test_project_state_and_governance_remain_closed() -> None:
    artifact = _load_artifact()
    assert artifact["governance"]["candidate_selected"] is False
    assert artifact["governance"]["sealed_confirmation_status"] == "RESERVED_NOT_MATERIALIZED"
    assert artifact["governance"]["sealed_confirmation_consumed"] is False
    assert artifact["governance"]["training_authorized"] is False
    assert artifact["governance"]["formal_promotion"] is False
    assert artifact["governance"]["generation_readiness"] == "not_ready"
    assert artifact["supersession"]["v1_contract_immutable"] is True
    assert artifact["supersession"]["v1_invalid_harness_immutable"] is True
    assert artifact["supersession"]["no_development_outcomes_observed"] is True
    assert artifact["governance"]["oracle_classification"] == "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT"
    assert artifact["governance"]["attribution_classification"] == "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED"
    assert artifact["governance"]["current_copula_classification"] == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED"
