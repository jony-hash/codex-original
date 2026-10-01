"""Harness tests for the LightGen V2 target-device preflight."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import scripts.research.lightgen.execute_lightgen_2d_semantic_complex_ols_long_training_preflight_v2 as preflight
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import FrozenPhysicalOLSScale, TrainingContractError, load_scale_artifact, save_scale_artifact


ROOT = Path(__file__).resolve().parents[3]


class LightGenV2LongTrainingPreflightTests(unittest.TestCase):
    def test_authorities_use_explicit_paths_and_scientific_roles(self) -> None:
        contract, authorities = preflight._validate_authorities(ROOT)
        self.assertEqual(contract.path, ROOT / preflight.CONTRACT_RELATIVE_PATH)
        self.assertEqual(authorities["correction_authority"]["path"], preflight.CORRECTION_AUTHORITY_RELATIVE_PATH.as_posix())
        self.assertTrue(authorities["semantic_equivalence"]["all_scientific_fields_equal"])
        self.assertEqual(authorities["historical_blocked_preflight"]["disposition"], "historical/superseded-by-authority-correction")
        self.assertEqual(authorities["consumer"]["path"], preflight.CONSUMER_RELATIVE_PATH.as_posix())
        self.assertEqual(authorities["aligned_runtime"]["path"], preflight.ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix())
        self.assertEqual(authorities["evaluator"]["checkpoint"], preflight.EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix())
        self.assertFalse(authorities["dataset_authority"]["protected_test_read"])
        self.assertFalse(authorities["dataset_authority"]["sealed_read"])
        self.assertFalse(authorities["hash_on_every_trust_boundary"])

    def test_corrected_preflight_input_is_bound_and_supersedes_only_the_blocked_attempt(self) -> None:
        contract, authorities = preflight._validate_authorities(ROOT)
        payload = preflight._preflight_input_payload(ROOT, contract, authorities, preflight.git_facts(ROOT))
        self.assertEqual(payload["schema"], preflight.PREFLIGHT_INPUT_SCHEMA)
        self.assertEqual(payload["result"], "READY_FOR_EXECUTION")
        self.assertEqual(payload["contract"]["path"], preflight.CONTRACT_RELATIVE_PATH.as_posix())
        self.assertEqual(payload["correction_authority"]["path"], preflight.CORRECTION_AUTHORITY_RELATIVE_PATH.as_posix())
        self.assertEqual(payload["canonical_training_source"]["training"], preflight.TRAINING_DATA_RELATIVE_PATH.as_posix())
        self.assertEqual(payload["consumer"]["path"], preflight.CONSUMER_RELATIVE_PATH.as_posix())
        self.assertEqual(payload["protocol"]["target_device_grid"], 2048)
        self.assertEqual(payload["protocol"]["logical_batch_size"], 10)
        self.assertEqual(payload["protocol"]["stage_a"]["future_steps"], 300)
        self.assertEqual(payload["protocol"]["stage_b"]["future_steps"], 300)
        self.assertEqual(payload["supersedes"]["classification"], "BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH")
        self.assertFalse(payload["supersedes"]["invalidated_or_deleted"])
        self.assertFalse(payload["claim_boundary"]["training_authorized"])
        self.assertFalse(payload["claim_boundary"]["protected_accessed"])
        self.assertFalse(payload["claim_boundary"]["sealed_accessed"])

    def test_train_dev_partition_preserves_observed_source_counts(self) -> None:
        images, labels, train_indices, development_indices, _selection_indices, _dataset = preflight._load_dataset(ROOT, object())
        self.assertEqual(tuple(images.shape), (60_000, 1, 28, 28))
        source_counts = torch.bincount(labels, minlength=10)
        train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
        development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
        self.assertTrue(torch.equal(train_counts + development_counts, source_counts))
        self.assertFalse(torch.equal(source_counts, torch.full((10,), 6_000, dtype=torch.long)))

    def test_development_selection_is_100_per_class(self) -> None:
        labels = torch.arange(60_000, dtype=torch.long).remainder(10)
        development_indices = torch.arange(5_000, dtype=torch.long)
        selected = preflight._fixed_development_selection(labels, development_indices)
        self.assertEqual(selected.numel(), 1_000)
        self.assertTrue(torch.equal(torch.bincount(labels.index_select(0, selected), minlength=10), torch.full((10,), 100, dtype=torch.long)))

    def test_preflight_scale_is_not_loadable_as_formal_scale(self) -> None:
        scale = FrozenPhysicalOLSScale(
            torch.ones(100, dtype=torch.float64),
            {"role": "NON_SCIENTIFIC_PREFLIGHT_REFERENCE"},
            {"role": "TRAIN", "rows": 55_000},
            data_role=preflight.SCALE_SMOKE_CLASSIFICATION,
            classification=preflight.PREFLIGHT_SCALE_CLASSIFICATION,
        )
        with tempfile.TemporaryDirectory(prefix="lightgen-v2-preflight-test-") as temporary:
            artifact = save_scale_artifact(Path(temporary) / "preflight-scale.v2.pt", scale)
            loaded = preflight._load_preflight_scale_artifact(Path(artifact.artifact_path))
            self.assertEqual(loaded.classification, preflight.PREFLIGHT_SCALE_CLASSIFICATION)
            self.assertEqual(loaded.vector_sha256, artifact.vector_sha256)
            with self.assertRaisesRegex(TrainingContractError, "classification is invalid"):
                load_scale_artifact(artifact.artifact_path)

    def test_output_path_cannot_escape_outputs(self) -> None:
        with self.assertRaises(preflight.InvalidPreflight):
            preflight._resolve_output_dir(ROOT, ROOT / "outside-preflight-v2")


if __name__ == "__main__":
    unittest.main()
