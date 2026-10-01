"""Tests for the definition-only LightGen post-training evaluation contract."""

from __future__ import annotations

import unittest

from scripts.research.lightgen import define_lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1 as contract


class LightGenPostTrainingEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = contract._repo_root()
        cls.alignment_payload, cls.alignment = contract._validate_alignment_authority(cls.root)
        cls.training = contract._validate_training_authority(cls.root)
        cls.evaluator = contract._validate_evaluator_authority(cls.root)
        cls.data = contract._validate_dataset_authority(cls.root, cls.training)

    def test_contract_is_definition_only_and_fail_closed(self) -> None:
        totality = contract._contract_totality_checks()
        self.assertTrue(totality["definition_only"])
        self.assertFalse(totality["training_executed"])
        self.assertFalse(totality["scientific_generation_executed"])
        self.assertFalse(totality["heldout_evaluation_executed"])
        self.assertFalse(totality["protected_values_read"])
        self.assertFalse(totality["fid_executed"])
        self.assertFalse(totality["kid_executed"])
        self.assertFalse(totality["mmd_executed"])
        self.assertFalse(totality["sealed_values_reused"])
        self.assertFalse(totality["training_authorized"])
        self.assertEqual(totality["generation_readiness"], "not_ready")

    def test_valid_training_lineage_binds_only_step_100(self) -> None:
        self.assertEqual(self.training["checkpoint"]["sha256"], contract.CHECKPOINT_SHA256)
        self.assertEqual(self.training["checkpoint_policy"]["eligible_checkpoint"], "step_100 only")
        self.assertFalse(self.training["checkpoint_policy"]["selection_search"])
        self.assertFalse(self.training["checkpoint_policy"]["fallback_checkpoint"])
        self.assertFalse(self.training["training_boundary"]["post_training_evaluation_executed"])

    def test_evaluator_authority_and_negative_control_are_explicit(self) -> None:
        selected = self.evaluator["selected_authority"]
        self.assertEqual(selected["architecture_kind"], "frozen_mnist_evaluator_v1")
        self.assertGreaterEqual(selected["validation_accuracy"], contract.EVALUATOR_QUALIFICATION_THRESHOLD)
        self.assertFalse(selected["trained_on_generator_checkpoint"])
        self.assertFalse(selected["protected_test_used_for_training"])
        rejected = self.evaluator["output_domain_candidate_rejected"]
        self.assertFalse(rejected["used_by_this_contract"])
        self.assertEqual(rejected["classification"], "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED")

    def test_digest_metadata_does_not_lock_an_explicit_authority_path(self) -> None:
        payload = contract._read_json(
            self.root,
            contract.ALIGNMENT_CONTRACT_PATH,
            "aligned contract",
            "0" * 64,
        )
        self.assertEqual(payload["schema"], "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1")
        with self.assertRaises(contract.DefinitionBlocked):
            contract._read_json(self.root, contract.Path("missing/authority.json"), "missing authority")

    def test_data_roles_reserve_calibration_and_protected_boundary(self) -> None:
        self.assertEqual(self.data["roles"]["train_reference"]["count"], 55_000)
        self.assertEqual(self.data["roles"]["development_checkpoint_selection"]["count"], 1_000)
        self.assertEqual(self.data["roles"]["development_calibration"]["count"], 4_000)
        self.assertEqual(self.data["roles"]["development_calibration"]["per_class"], 400)
        self.assertEqual(
            self.data["roles"]["development_calibration"]["indices_sha256"],
            contract.DEVELOPMENT_CALIBRATION_INDICES_SHA256,
        )
        protected = self.data["roles"]["protected_semantic_generation_evaluation"]
        self.assertEqual(protected["count"], 10_000)
        self.assertFalse(protected["read_by_definition"])
        self.assertFalse(self.data["roles"]["sealed"]["read_by_future_evaluation"])

    def test_protocol_has_one_hard_primary_per_distribution_stage(self) -> None:
        protocol = contract._evaluation_protocol()
        self.assertEqual(protocol["checkpoint"]["eligible_count"], 1)
        self.assertEqual(protocol["stage_2_latent_compatibility"]["data_panel"]["count"], 1_000)
        self.assertEqual(protocol["stage_3_direct_prior_generation"]["banks"]["total_samples"], 1_000)
        semantic = protocol["stage_4_semantic_distribution"]
        self.assertEqual(semantic["status"], "HARD_GATE_AFTER_GENERATION")
        self.assertIn("MMD2_feature", semantic["observed"])
        self.assertNotIn("FID", semantic)
        self.assertNotIn("KID", semantic)
        diversity = protocol["stage_4_diversity"]
        novelty = protocol["stage_4_novelty"]
        self.assertEqual(diversity["status"], "HARD_GATE_AFTER_SEMANTIC_DISTRIBUTION")
        self.assertEqual(novelty["status"], "HARD_GATE_AFTER_DIVERSITY")
        self.assertTrue(novelty["retention"].startswith("all generated samples remain"))

    def test_architecture_claim_and_next_action_are_bounded(self) -> None:
        totality = contract._contract_totality_checks()
        payload = contract._build_payload(
            self.root,
            {"branch": contract.EXPECTED_BRANCH, "head": contract.EXPECTED_STARTING_HEAD},
            self.alignment_payload,
            self.alignment,
            self.training,
            self.evaluator,
            self.data,
            {"state_mutated": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
            totality,
        )
        self.assertEqual(payload["result"], "READY")
        self.assertEqual(payload["classification"], contract.READY_CLASSIFICATION)
        self.assertEqual(payload["next_action"], contract.NEXT_ACTION)
        self.assertEqual(payload["ols_mathematical_definition"]["shape"], ["N", 100])
        self.assertTrue(payload["generator_input_writer"]["same_writer"])
        self.assertTrue(payload["generator_input_writer"]["same_generator"])
        self.assertFalse(payload["training_authorized"])
        self.assertFalse(payload["formal_promotion"])
        self.assertEqual(payload["generation_readiness"], "not_ready")
        self.assertEqual(
            payload["outer_gate"]["hard_stop_conditions"]["semantic_evaluator_authority_unresolved"],
            False,
        )


if __name__ == "__main__":
    unittest.main()
