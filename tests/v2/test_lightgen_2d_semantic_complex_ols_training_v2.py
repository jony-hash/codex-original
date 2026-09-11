"""Engineering tests for the LightGen V2 complex-OLS training consumer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    ALLOWED_TRAINABLE_PARAMETER_NAMES,
    COMPLETE,
    DeterministicTrainRoleSampler,
    FINAL_SELECT,
    FrozenPhysicalOLSScale,
    LightGen2DComplexOLSTrainingConsumerV2,
    SCALE_SMOKE_CLASSIFICATION,
    STAGE_A_FAILURE,
    STAGE_A_SEMANTIC_CHECK,
    STAGE_A_TRAIN,
    STAGE_B_FAILURE,
    STAGE_B_SEMANTIC_CHECK,
    STAGE_B_TRAIN,
    V2TrainingStateMachine,
    checkpoint_payload,
    extract_train_only_ols_scale,
    load_checkpoint,
    load_scale_artifact,
    load_v2_training_contract,
    normalized_full_complex_mmd_vstat,
    normalized_gauge_representation,
    normalized_ols,
    physical_ols_from_normalized,
    sample_normalized_prior,
    save_checkpoint,
    save_scale_artifact,
    select_stage_a_checkpoint,
    select_stage_b_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract():
    return load_v2_training_contract(repo_root=ROOT)


def _consumer() -> LightGen2DComplexOLSTrainingConsumerV2:
    return LightGen2DComplexOLSTrainingConsumerV2(
        _contract(),
        device=torch.device("cpu"),
        propagation_grid_size=128,
    )


def _fixture_images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


class LightGenV2TrainingConsumerTests(unittest.TestCase):
    def test_parameter_firewall_prior_and_gauge_invariant_mmd(self) -> None:
        consumer = _consumer()
        manifest = consumer.parameter_manifest()
        self.assertEqual(manifest["trainable_parameter_names"], list(ALLOWED_TRAINABLE_PARAMETER_NAMES))
        self.assertEqual(manifest["optimizer_parameter_names"], list(ALLOWED_TRAINABLE_PARAMETER_NAMES))
        self.assertEqual(manifest["trainable_parameter_count"], 40_000)
        first = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
        replay = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
        self.assertEqual(first.coefficients.shape, (10, 100))
        self.assertEqual(first.coefficients.dtype, torch.complex64)
        self.assertTrue(torch.equal(first.coefficients, replay.coefficients))
        rotated = first.coefficients * torch.polar(torch.tensor(1.0), torch.tensor(0.37))
        self.assertTrue(torch.allclose(normalized_gauge_representation(first.coefficients), normalized_gauge_representation(rotated), atol=2.0e-6, rtol=2.0e-6))
        self.assertTrue(torch.allclose(normalized_full_complex_mmd_vstat(first.coefficients, first.coefficients), normalized_full_complex_mmd_vstat(rotated, first.coefficients), atol=2.0e-6, rtol=2.0e-6))

    def test_stage_a_and_stage_b_update_boundaries_and_frozen_scale(self) -> None:
        consumer = _consumer()
        optimizer_a = consumer.create_optimizer()
        update_a = consumer.update(_fixture_images(), stage="A", global_step=0, stage_local_step=0, optimizer=optimizer_a)
        self.assertTrue(update_a.autograd.stage_a_no_mmd_gradient)
        self.assertIsNone(update_a.loss.prior)
        self.assertAlmostEqual(float(update_a.loss.total), float(update_a.loss.image), places=7)
        self.assertTrue(update_a.autograd.modal_projection_active)
        self.assertTrue(update_a.autograd.w_ols_active)
        self.assertTrue(update_a.frozen_state_unchanged)
        self.assertTrue(optimizer_a.state)

        smoke_scale = extract_train_only_ols_scale(
            consumer,
            _fixture_images(),
            data_role=SCALE_SMOKE_CLASSIFICATION,
            checkpoint_identity={"role": SCALE_SMOKE_CLASSIFICATION, "stage_local_step": 200},
            data_identity={"role": SCALE_SMOKE_CLASSIFICATION},
        )
        self.assertEqual(smoke_scale.classification, SCALE_SMOKE_CLASSIFICATION)
        self.assertEqual(smoke_scale.values.dtype, torch.float64)
        self.assertEqual(tuple(smoke_scale.values.shape), (100,))
        self.assertFalse(smoke_scale.values.requires_grad)
        with self.assertRaisesRegex(ValueError, "rejects development"):
            extract_train_only_ols_scale(
                consumer,
                _fixture_images(),
                data_role="development",
                checkpoint_identity={},
                data_identity={},
            )

        with tempfile.TemporaryDirectory(prefix="lightgen-v2-test-") as temporary:
            scale = save_scale_artifact(Path(temporary) / "scale.v2.pt", smoke_scale)
            self.assertEqual(load_scale_artifact(scale.artifact_path, expected_sha256=scale.artifact_sha256).vector_sha256, scale.vector_sha256)
            consumer.set_scale(scale)
            optimizer_b, marker = consumer.reset_stage_b_optimizer()
            self.assertTrue(marker["optimizer_reset"])
            self.assertTrue(marker["new_optimizer_state_empty"])
            self.assertFalse(marker["stage_a_optimizer_moments_restored"])
            self.assertFalse(optimizer_b.state)
            update_b = consumer.update(_fixture_images(), stage="B", global_step=301, stage_local_step=1, optimizer=optimizer_b, scale=scale)
            self.assertTrue(update_b.autograd.stage_b_normalized_mmd_active)
            self.assertTrue(update_b.autograd.scale_gradient_active)
            self.assertTrue(update_b.autograd.frozen_scale_no_grad)
            self.assertAlmostEqual(float(update_b.loss.total), float(update_b.loss.image + update_b.loss.latent + update_b.loss.scale), places=7)
            recovered = physical_ols_from_normalized(normalized_ols(update_b.loss.result.z_ols.coefficients, scale), scale)
            self.assertTrue(torch.allclose(recovered, update_b.loss.result.z_ols.coefficients, rtol=2.0e-5, atol=2.0e-7))

    def test_state_machine_sampler_and_selection_are_fail_closed(self) -> None:
        records_a = [
            {"stage_local_step": 100, "development_macro_recall": 0.95, "development_L_image": 0.30},
            {"stage_local_step": 200, "development_macro_recall": 0.96, "development_L_image": 0.20},
            {"stage_local_step": 300, "development_macro_recall": 0.94, "development_L_image": 0.40},
        ]
        self.assertEqual(select_stage_a_checkpoint(records_a)["stage_local_step"], 200)
        self.assertIsNone(select_stage_a_checkpoint([{**record, "development_macro_recall": 0.1} for record in records_a]))
        records_b = [{**record, "development_L_total": record["development_L_image"]} for record in records_a]
        self.assertEqual(select_stage_b_checkpoint(records_b)["stage_local_step"], 200)
        labels = torch.arange(60_000, dtype=torch.long).remainder(10)
        train_indices = torch.arange(55_000, dtype=torch.long)
        sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=300)
        first_indices, first_labels = sampler.next_batch()
        state = sampler.state_dict()
        restored_sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=300)
        restored_sampler.load_state_dict(state)
        self.assertTrue(torch.equal(restored_sampler.next_batch()[0], sampler.next_batch()[0]))
        self.assertTrue(torch.equal(first_labels, torch.arange(10, dtype=torch.long)))

        machine = V2TrainingStateMachine(stage=STAGE_A_TRAIN)
        machine.transition(STAGE_A_SEMANTIC_CHECK)
        machine.stage_a_checkpoint_checked(final_checkpoint=True, eligible=False)
        self.assertEqual(machine.terminal_classification, STAGE_A_FAILURE)
        failed_b = V2TrainingStateMachine(stage=STAGE_B_SEMANTIC_CHECK)
        failed_b.stage_b_checkpoint_checked(final_checkpoint=True, eligible=False)
        self.assertEqual(failed_b.terminal_classification, STAGE_B_FAILURE)

        complete = V2TrainingStateMachine(stage=STAGE_B_SEMANTIC_CHECK)
        complete.stage_b_checkpoint_checked(final_checkpoint=True, eligible=True)
        self.assertEqual(complete.stage, FINAL_SELECT)
        complete.transition(COMPLETE)
        self.assertEqual(complete.stage, COMPLETE)

    def test_stage_b_checkpoint_round_trip_binds_scale_and_optimizer_reset(self) -> None:
        consumer = _consumer()
        optimizer_a = consumer.create_optimizer()
        consumer.update(_fixture_images(), stage="A", global_step=0, stage_local_step=0, optimizer=optimizer_a)
        scale = FrozenPhysicalOLSScale.from_tensor(
            torch.ones(100, dtype=torch.float64),
            checkpoint_identity={"role": SCALE_SMOKE_CLASSIFICATION},
            data_identity={"role": SCALE_SMOKE_CLASSIFICATION},
        )
        with tempfile.TemporaryDirectory(prefix="lightgen-v2-test-") as temporary:
            scale = save_scale_artifact(Path(temporary) / "scale.v2.pt", scale)
            consumer.set_scale(scale)
            optimizer_b, marker = consumer.reset_stage_b_optimizer()
            consumer.update(_fixture_images(), stage="B", global_step=301, stage_local_step=1, optimizer=optimizer_b, scale=scale)
            state = V2TrainingStateMachine(
                stage=STAGE_B_TRAIN,
                global_step=301,
                stage_local_step=1,
                selected_stage_a_checkpoint={"stage_local_step": 200},
                scale_identity=scale.identity(),
                optimizer_reset_marker=marker,
            )
            payload = checkpoint_payload(
                consumer,
                optimizer_b,
                state=state,
                sampler_state={"schema": "fixture"},
                semantic_selection_state={"threshold": 0.937},
                git={"branch": "fixture", "head": "fixture", "worktree_dirty": True},
                scale=scale,
                selected_stage_a_checkpoint={"stage_local_step": 200},
                optimizer_reset_marker=marker,
            )
            checkpoint = Path(temporary) / "checkpoint.v2.pt"
            save_checkpoint(checkpoint, payload)
            restored = _consumer()
            restored_optimizer = restored.create_optimizer()
            loaded = load_checkpoint(checkpoint, restored, restored_optimizer, expected_scale=scale)
            self.assertEqual(loaded.stage, STAGE_B_TRAIN)
            self.assertEqual(loaded.scale.vector_sha256, scale.vector_sha256)
            self.assertEqual(loaded.scale.artifact_sha256, scale.artifact_sha256)
            self.assertEqual(loaded.optimizer_reset_marker["optimizer_reset"], True)
            for name, value in consumer.trainable_state_dict().items():
                self.assertTrue(torch.equal(value, restored.trainable_state_dict()[name]))
            self.assertTrue(_nested_equal(restored_optimizer.state_dict(), optimizer_b.state_dict()))

            tampered_payload = dict(payload)
            tampered_artifact = dict(payload["scale_artifact"])
            tampered_artifact["sha256"] = "0" * 64
            tampered_payload["scale_artifact"] = tampered_artifact
            tampered_checkpoint = Path(temporary) / "tampered-checkpoint.v2.pt"
            save_checkpoint(tampered_checkpoint, tampered_payload)
            tampered_consumer = _consumer()
            tampered_optimizer = tampered_consumer.create_optimizer()
            with self.assertRaisesRegex(ValueError, "scale artifact SHA mismatch"):
                load_checkpoint(tampered_checkpoint, tampered_consumer, tampered_optimizer)


if __name__ == "__main__":
    unittest.main()
