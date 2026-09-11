"""Tests for the bounded LightGen formal-training execution boundary."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from scripts.research import execute_lightgen_2d_semantic_complex_ols_formal_training_v1 as execution


ROOT = Path(__file__).resolve().parents[2]


class LightGenFormalTrainingExecutionTests(unittest.TestCase):
    def test_authorities_are_hash_bound_without_opening_protected_data(self) -> None:
        contract, authorities = execution._validate_authorities(ROOT)
        self.assertEqual(contract.sha256, execution.TRAINING_CONTRACT_SHA256)
        self.assertEqual(
            authorities["long_training_preflight"]["sha256"],
            execution.PREFLIGHT_SHA256,
        )
        self.assertTrue(authorities["hash_on_trust_boundary"])

    def test_development_target_is_fixed_full_complex_and_deterministic(self) -> None:
        first, first_identity = execution._sample_development_target()
        second, second_identity = execution._sample_development_target()
        self.assertEqual(tuple(first.shape), (1000, 100))
        self.assertTrue(torch.is_complex(first))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(first_identity["seed"], execution.DEVELOPMENT_TARGET_SEED)
        self.assertEqual(first_identity["namespace"], execution.PRIOR_NAMESPACE)

    def test_output_path_cannot_escape_outputs_root(self) -> None:
        with self.assertRaises(execution.InvalidTrainingExecution):
            execution._resolve_output_dir(ROOT, ROOT / "outside-training", None)


if __name__ == "__main__":
    unittest.main()
