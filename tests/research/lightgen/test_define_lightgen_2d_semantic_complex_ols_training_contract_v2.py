"""Tests for the materialized LightGen V2 training-consumer contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import CHECKPOINT_STEPS, EPSILON_SCALE, LEARNING_RATE, LOGICAL_BATCH_SIZE, MMD_BANDWIDTH_SQUARED, PRIOR_NAMESPACE, SEMANTIC_THRESHOLD, STAGE_A_STEPS, STAGE_B_STEPS, V2_CONTRACT_RELATIVE_PATH, V2_CONTRACT_SHA256, load_v2_training_contract


ROOT = Path(__file__).resolve().parents[3]


class LightGenV2TrainingContractTests(unittest.TestCase):
    def test_materialized_contract_freezes_definition_only_boundary(self) -> None:
        contract = load_v2_training_contract(repo_root=ROOT)
        self.assertEqual(len(contract.sha256), len(V2_CONTRACT_SHA256))
        self.assertEqual(contract.path, (ROOT / V2_CONTRACT_RELATIVE_PATH).resolve())
        self.assertEqual(contract.payload["classification"], "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V2_READY")
        self.assertFalse(contract.payload["training_authorized"])
        self.assertFalse(contract.payload["formal_promotion"])
        self.assertEqual(contract.payload["generation_readiness"], "not_ready")

    def test_exact_stage_scale_prior_and_optimizer_parameters_are_bound(self) -> None:
        payload = json.loads((ROOT / V2_CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(payload["stage_a"]["steps"], STAGE_A_STEPS)
        self.assertEqual(payload["stage_a"]["checkpoint_steps"], list(CHECKPOINT_STEPS))
        self.assertEqual(payload["stage_b"]["steps"], STAGE_B_STEPS)
        self.assertEqual(payload["stage_b"]["checkpoint_steps"], list(CHECKPOINT_STEPS))
        self.assertEqual(payload["optimizer_and_precision"]["logical_batch_size"], LOGICAL_BATCH_SIZE)
        self.assertEqual(payload["optimizer_and_precision"]["optimizer"]["learning_rate"], LEARNING_RATE)
        self.assertEqual(payload["scale_contract"]["epsilon_scale"], EPSILON_SCALE)
        self.assertEqual(payload["normalized_latent_contract"]["prior"]["namespace"], PRIOR_NAMESPACE)
        self.assertIn(f"l^2={MMD_BANDWIDTH_SQUARED:g}", payload["normalized_latent_contract"]["mmd"]["kernel"])
        self.assertEqual(payload["semantic_evaluator_contract"]["threshold"]["value"], SEMANTIC_THRESHOLD)
        self.assertEqual(payload["trainable_parameter_contract"]["exact_trainable_parameters"], ["encoder.phase", "generator.phase"])

    def test_implementation_artifacts_are_non_scientific_and_governance_closed(self) -> None:
        output = ROOT / "outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02"
        report = json.loads((output / "lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json").read_text(encoding="utf-8"))
        preflight = json.loads((output / "lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1.json").read_text(encoding="utf-8"))
        self.assertEqual(report["classification"], "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2_IMPLEMENTED")
        self.assertEqual(preflight["classification"], "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_INPUT_READY")
        self.assertEqual(report["smoke"]["scientific_steps_executed"], 0)
        self.assertEqual(report["smoke"]["real_data_rows_read"], 0)
        self.assertEqual(report["smoke"]["protected_test_rows_read"], 0)
        self.assertEqual(report["smoke"]["sealed_rows_read"], 0)
        self.assertFalse(preflight["claim_boundary"]["training_authorized"])
        self.assertFalse(preflight["claim_boundary"]["formal_promotion"])
        self.assertEqual(preflight["claim_boundary"]["generation_readiness"], "not_ready")
        self.assertEqual(preflight["planned_budget"]["stage_a_steps"], STAGE_A_STEPS)
        self.assertEqual(preflight["planned_budget"]["stage_b_steps"], STAGE_B_STEPS)


if __name__ == "__main__":
    unittest.main()
