"""Engineering tests for the LightGen full-complex OLS training consumer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import (
    ALLOWED_TRAINABLE_PARAMETER_NAMES,
    FrozenComplexOLSPrior,
    LightGen2DComplexOLSTrainingConsumer,
    checkpoint_payload,
    full_complex_mmd_vstat,
    gauge_invariant_ols_representation,
    load_frozen_training_contract,
    load_smoke_checkpoint,
    save_smoke_checkpoint,
    validate_legacy_exclusion,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract():
    return load_frozen_training_contract(repo_root=ROOT)


def _consumer() -> LightGen2DComplexOLSTrainingConsumer:
    return LightGen2DComplexOLSTrainingConsumer(
        _contract(),
        device=torch.device("cpu"),
        propagation_grid_size=128,
    )


def _fixture_images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


class LightGenComplexOLSTrainingConsumerTests(unittest.TestCase):
    def test_contract_firewall_and_optimizer_manifest_are_exact(self) -> None:
        consumer = _consumer()
        manifest = consumer.parameter_manifest()
        self.assertEqual(manifest["trainable_parameter_names"], list(ALLOWED_TRAINABLE_PARAMETER_NAMES))
        self.assertEqual(manifest["optimizer_parameter_names"], list(ALLOWED_TRAINABLE_PARAMETER_NAMES))
        self.assertEqual(manifest["trainable_parameter_count"], 40_000)
        self.assertEqual(manifest["frozen_parameter_count"], 0)
        self.assertEqual(manifest["parameter_dtypes"], {name: "torch.float32" for name in ALLOWED_TRAINABLE_PARAMETER_NAMES})
        optimizer = consumer.create_optimizer()
        self.assertEqual(consumer.optimizer_manifest(optimizer)["parameter_names"], list(ALLOWED_TRAINABLE_PARAMETER_NAMES))
        self.assertTrue(validate_legacy_exclusion()["passed"])

    def test_prior_replay_shape_and_dtype_are_exact(self) -> None:
        prior = FrozenComplexOLSPrior()
        first = prior.sample(10, global_step=3, device=torch.device("cpu"))
        replay = prior.sample(10, global_step=3, device=torch.device("cpu"))
        self.assertEqual(first.seed, replay.seed)
        self.assertEqual(first.state.coefficients.shape, (10, 100))
        self.assertEqual(first.state.coefficients.dtype, torch.complex64)
        self.assertTrue(torch.equal(first.state.coefficients, replay.state.coefficients))
        self.assertTrue(torch.equal(first.generator_state, replay.generator_state))

    def test_mmd_is_global_phase_invariant_but_keeps_relative_complex_structure(self) -> None:
        prior = FrozenComplexOLSPrior().sample(10, global_step=0, device=torch.device("cpu")).state.coefficients
        encoded = prior.clone()
        phase = torch.polar(torch.tensor(1.0), torch.tensor(0.37))
        rotated = encoded * phase
        self.assertTrue(torch.allclose(gauge_invariant_ols_representation(encoded), gauge_invariant_ols_representation(rotated), atol=2.0e-6, rtol=2.0e-6))
        self.assertTrue(torch.allclose(full_complex_mmd_vstat(encoded, prior), full_complex_mmd_vstat(rotated, prior), atol=2.0e-6, rtol=2.0e-6))
        reference = torch.zeros((10, 100), dtype=torch.complex64)
        reference[:, 0] = 1.0e-3
        changed = reference.clone()
        changed[:, 1] = 1.0e-3
        self.assertGreater(float(full_complex_mmd_vstat(changed, reference)), 1.0e-8)

    def test_cpu_update_proves_modal_writer_and_gradient_firewall(self) -> None:
        consumer = _consumer()
        optimizer = consumer.create_optimizer()
        update = consumer.update(_fixture_images(), global_step=0, optimizer=optimizer)
        self.assertTrue(update.autograd.modal_projection_active)
        self.assertTrue(update.autograd.w_ols_active)
        self.assertTrue(update.autograd.trainable_gradients_finite)
        self.assertTrue(update.autograd.forbidden_gradients_absent)
        self.assertTrue(update.frozen_state_unchanged)
        self.assertTrue(all(value > 0.0 for value in update.parameter_update_norms.values()))
        self.assertTrue(all(torch.isfinite(value) for value in (update.loss.image, update.loss.latent, update.loss.optical, update.loss.total)))
        self.assertAlmostEqual(float(update.loss.total), float(update.loss.image + update.loss.latent), places=7)

    def test_smoke_checkpoint_safe_load_preserves_parameters_optimizer_and_rng(self) -> None:
        consumer = _consumer()
        optimizer = consumer.create_optimizer()
        update = consumer.update(_fixture_images(), global_step=0, optimizer=optimizer)
        sampler_state = {
            "schema": "lightgen_train_role_sampler.v1",
            "seed": 1234,
            "cursor": 1,
            "class_count": 10,
            "rows_per_class": 5_500,
            "permutations": [torch.arange(5_500, dtype=torch.long) for _ in range(10)],
        }
        git = {"branch": "fixture", "head": "fixture", "worktree_dirty": True}
        payload = checkpoint_payload(
            consumer,
            optimizer,
            global_step=0,
            sampler_state=sampler_state,
            prior=update.loss.prior,
            dataset_roles={"roles": {"train": {"count": 55_000}}, "heldout_consumed": False, "sealed_consumed": False},
            git=git,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "smoke.pt"
            save_smoke_checkpoint(path, payload)
            restored = _consumer()
            restored_optimizer = restored.create_optimizer()
            loaded = load_smoke_checkpoint(path, restored, restored_optimizer)
            for name, value in consumer.trainable_state_dict().items():
                self.assertTrue(torch.equal(value, restored.trainable_state_dict()[name]))
            self.assertEqual(loaded.global_step, 0)
            self.assertEqual(loaded.completed_steps, 1)
            self.assertTrue(torch.equal(payload["rng_state"]["cpu"], torch.random.get_rng_state()))
            self.assertTrue(torch.equal(payload["prior_rng"]["state"], loaded.prior_rng["state"]))


if __name__ == "__main__":
    unittest.main()
