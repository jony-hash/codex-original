"""Tests for the bounded LightGen target-device preflight consumer."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.research.execute_lightgen_2d_semantic_complex_ols_long_training_preflight_v1 as preflight
from ldm_project.v2.student.evaluator import load_local_mnist, stratified_mnist_indices


ROOT = Path(__file__).resolve().parents[2]


class LightGenLongTrainingPreflightTests(unittest.TestCase):
    def test_frozen_authorities_and_consumer_identity_are_hash_bound(self) -> None:
        contract, authorities = preflight._validate_authorities(ROOT)
        self.assertEqual(contract.sha256, preflight.TRAINING_CONTRACT_SHA256)
        self.assertEqual(
            authorities["implemented_consumer_preflight"]["sha256"],
            preflight.PREVIOUS_PREFLIGHT_ARTIFACT_SHA256,
        )
        self.assertTrue(authorities["hash_on_trust_boundary"])

    def test_tampered_implemented_preflight_is_rejected_before_execution(self) -> None:
        original_hash = preflight.file_sha256

        def tampered_hash(path: Path) -> str:
            if "lightgen_2d_semantic_complex_ols_training_preflight_input" in str(path):
                return "0" * 64
            return original_hash(path)

        with patch.object(preflight, "file_sha256", side_effect=tampered_hash):
            with self.assertRaises(preflight.InvalidPreflight):
                preflight._validate_authorities(ROOT)

    def test_development_selection_is_100_per_class_and_does_not_read_test_split(self) -> None:
        images, labels = load_local_mnist(ROOT / "data", train=True)
        train_indices, development_indices = stratified_mnist_indices(
            labels, seed=1234, validation_per_label=500
        )
        selected = preflight._fixed_development_selection(labels, development_indices)
        self.assertEqual(tuple(images.shape), (60_000, 1, 28, 28))
        self.assertEqual(selected.numel(), 1_000)
        self.assertTrue(
            torch.equal(
                torch.bincount(labels.index_select(0, selected), minlength=10),
                torch.full((10,), 100, dtype=torch.long),
            )
        )
        self.assertEqual(train_indices.numel(), 55_000)

    def test_output_path_cannot_escape_outputs_root(self) -> None:
        with self.assertRaises(preflight.InvalidPreflight):
            preflight._resolve_output_dir(ROOT, ROOT / "outside-preflight")


if __name__ == "__main__":
    unittest.main()
