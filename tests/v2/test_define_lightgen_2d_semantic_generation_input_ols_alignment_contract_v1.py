"""Targeted tests for the LightGen 2-D input/OLS alignment definition."""

from __future__ import annotations

import unittest

from scripts.research import define_lightgen_2d_semantic_generation_input_ols_alignment_contract_v1 as contract


class LightGen2DAlignmentContractTests(unittest.TestCase):
    def test_architecture_and_ols_dimensions_are_frozen(self) -> None:
        self.assertEqual(contract.TASK_ID, "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1")
        self.assertEqual(contract.ARCHITECTURE_DECISION, "LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST")
        self.assertEqual(contract.NEXT_ACTION, "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1")
        self.assertEqual(contract.OLS_GRID_SHAPE, (10, 10))
        self.assertEqual(contract.PORT_COUNT, 100)
        self.assertEqual(contract.OLS_PROJECTION_FORMULA, "z_j = <E_encoder, u_j>")
        self.assertEqual(contract.OLS_REEMISSION_FORMULA, "W_OLS(z) = sum_j (Tz)_j v_j")
        self.assertEqual(contract.OLS_TRANSMISSION_DECISION, "T=I_100 in coefficient space")

    def test_definition_totality_is_definition_only(self) -> None:
        totality = contract._contract_totality_checks()
        self.assertEqual(totality["status"], "passed")
        self.assertTrue(totality["projection_required"])
        self.assertTrue(totality["reemission_required"])
        self.assertTrue(totality["same_writer_for_real_and_generated_ols"])
        self.assertFalse(totality["whole_field_identity_bypass"])
        self.assertFalse(totality["q_is_authoritative_latent"])
        self.assertFalse(totality["training_executed"])
        self.assertFalse(totality["scientific_generation_executed"])
        self.assertFalse(totality["sealed_values_loaded"])
        self.assertFalse(totality["governance_mutated"])

    def test_source_taxonomy_keeps_sqrt_as_project_decision(self) -> None:
        decisions = contract._project_decisions()
        sqrt_decision = next(item for item in decisions if item["id"] == "PROJECT_DECISION_SLM_MAPPING")
        self.assertEqual(sqrt_decision["category"], "PROJECT_DECISION")
        self.assertIn("sqrt", sqrt_decision["decision"])
        self.assertNotIn("CHEN_FACT", sqrt_decision["decision"])

    def test_paths_converge_without_phase_repair_or_controller(self) -> None:
        payload = contract._build_payload(
            contract._repo_root(),
            {"branch": contract.EXPECTED_BRANCH, "head": contract.EXPECTED_STARTING_HEAD, "dirty_paths": [], "worktree_clean": False},
            {"paper_pdf": {"path": "research_assets/lightgen/science_adv7434/paper/sciencesm.pdf", "sha256": contract.PAPER_PDF_SHA256}},
            {},
            {"classification": "N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED"},
            {"training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
            contract._validate_source_code_authorities(contract._repo_root()),
            contract._contract_totality_checks(),
        )
        writer = payload["generator_input_writer"]
        self.assertTrue(writer["convergence_invariant"].startswith("The two paths share the same W_OLS"))
        self.assertNotIn("image encoder", writer["future_generated_ols_path"])
        self.assertIn("phase-repair module", writer["generation_prerequisites_forbidden"])

    def test_current_result_and_authorities_are_read_only_valid(self) -> None:
        root = contract._repo_root()
        result = contract._validate_latest_result(root)
        self.assertEqual(result["candidate"], "0/5")
        self.assertEqual(result["control"], "5/5")
        self.assertFalse(result["sealed"]["values_loaded"])
        self.assertFalse(result["sealed"]["reused"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(contract._validate_source_identity(root)["identity"]["doi"], "10.1126/science.adv7434")


if __name__ == "__main__":
    unittest.main()
