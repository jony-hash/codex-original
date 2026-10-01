"""Tests for the definition-only LightGen complex-OLS training contract."""

from __future__ import annotations

import unittest

from scripts.research.lightgen import define_lightgen_2d_semantic_complex_ols_training_contract_v1 as contract


class LightGenComplexOLSTrainingContractTests(unittest.TestCase):
    def test_outer_contract_is_definition_only_and_full_complex(self) -> None:
        self.assertEqual(contract.TASK_ID, "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1")
        self.assertEqual(contract.NEXT_ACTION, "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1")
        totality = contract._contract_totality_checks()
        self.assertTrue(totality["definition_only"])
        self.assertFalse(totality["training_executed"])
        self.assertFalse(totality["cuda_used"])
        self.assertFalse(totality["optimizer_created"])
        self.assertFalse(totality["scientific_generation_executed"])
        self.assertFalse(totality["sealed_values_loaded"])
        self.assertFalse(totality["q_authoritative"])
        self.assertTrue(totality["same_real_and_generated_w_ols"])

    def test_authorities_and_parameter_surface_are_current(self) -> None:
        root = contract._repo_root()
        alignment = contract._validate_alignment_authority(root)
        runtime = contract._validate_runtime_authority(root)
        parameter_surface = contract._inspect_trainable_parameter_surface()
        self.assertEqual(alignment["classification"], "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY")
        self.assertEqual(runtime["runtime"]["canonical_ols"], "complex [B,100], 10x10, row-major j=10*r+c")
        self.assertEqual(parameter_surface["trainable_parameter_names"], ["encoder.phase", "generator.phase"])
        self.assertEqual(parameter_surface["parameter_count"], 40_000)

    def test_prior_and_latent_objective_are_complete_and_not_q_only(self) -> None:
        self.assertEqual(contract.PRIOR_SIGMA, 1.0e-3)
        self.assertEqual(contract.MMD_BANDWIDTH_SQUARED, 200.0)
        decisions = contract._project_decisions()
        prior = next(item for item in decisions if item["id"] == "PROJECT_DECISION_FULL_COMPLEX_PRIOR")
        self.assertIn("C^100", prior["decision"])
        self.assertIn("not a Chen fact", prior["decision"])
        latent = next(item for item in decisions if item["id"] == "PROJECT_DECISION_LATENT_MMD")
        self.assertIn("full-complex", latent["decision"])
        self.assertNotIn("q-only", latent["decision"])

    def test_project_complex_authority_is_gauge_only(self) -> None:
        authority = contract._validate_project_complex_ols_authority(contract._repo_root())
        self.assertEqual(authority["semantic_authority_id"], "PROJECT_COMPLEX_OLS_V1")
        self.assertIn("global phase", authority["reused_boundary"])
        self.assertIn("not reused", authority["native_runtime_scaling"])

    def test_current_n3_failure_and_sealed_boundary_are_preserved(self) -> None:
        result = contract._validate_latest_scientific_boundary(contract._repo_root())
        self.assertEqual(result["candidate"], "0/5")
        self.assertEqual(result["control"], "5/5")
        self.assertEqual(result["candidate_p_mc"], [0.001] * 5)
        self.assertTrue(result["sealed"]["consumed"])
        self.assertFalse(result["sealed"]["values_loaded"])
        self.assertFalse(result["sealed"]["reused"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["generation_readiness"], "not_ready")

    def test_dataset_contract_uses_train_dev_and_protected_test_roles(self) -> None:
        data = contract._validate_dataset_authority(contract._repo_root())
        self.assertEqual(data["split"]["train_count"], 55_000)
        self.assertEqual(data["split"]["development_count"], 5_000)
        self.assertEqual(data["roles"]["protected_heldout"]["count"], 10_000)
        self.assertFalse(data["preprocessing"]["label_or_controller_input"])
        self.assertEqual(data["roles"]["calibration"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
