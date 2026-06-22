from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for training smoke tests")
class ConditionedTrainingSmokeTests(unittest.TestCase):
    def test_first_stage_loss_smoke_returns_foreground_metrics(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.training import first_stage_loss

        first_stage = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        batch = torch.rand(2, 1, 200, 200)
        outputs = first_stage(batch)
        losses = first_stage_loss(
            outputs,
            batch,
            global_step=0,
            total_steps=8,
            target_beta=1.0e-4,
            warmup_ratio=0.1,
        )
        losses["loss"].backward()

        for key in (
            "loss",
            "recon_loss",
            "image_recon_loss",
            "image_fg_mse",
            "image_bg_mse",
            "camera_recon_loss",
            "camera_fg_mse",
            "camera_bg_mse",
            "structure_loss",
            "image_structure_loss",
            "camera_structure_loss",
            "shape_loss",
            "image_shape_loss",
            "camera_shape_loss",
            "camera_foreground_dice_loss",
            "camera_foreground_dice",
            "field_smoothness_loss",
            "field_tv",
            "phase_smoothness_loss",
            "phase_tv",
            "background_leakage_loss",
            "camera_background_leakage",
            "camera_hotspot_loss",
            "camera_hotspot_ratio",
            "camera_saturation_ratio",
            "camera_topology_loss",
            "camera_topology_base",
            "camera_saturation_loss",
            "camera_saturation_penalty",
            "camera_artifact_loss",
            "camera_foreground_hole_loss",
            "camera_foreground_hole_ratio",
            "camera_foreground_roughness",
            "camera_label_margin_loss",
            "camera_label_margin_base",
            "camera_label_margin_mean",
            "camera_label_margin_requested_corr",
            "camera_label_margin_best_other_corr",
            "camera_label_margin_sample_ratio",
            "kl_loss",
            "fg_pixel_ratio",
            "pred_fg_energy_ratio",
            "hard_class_weight_mean",
        ):
            self.assertEqual(losses[key].ndim, 0, key)
            self.assertTrue(torch.isfinite(losses[key]).item(), key)

    def test_first_stage_loss_adds_weighted_structure_term(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        input_batch[:, :, 1:4, 1:4] = 1.0
        prediction = torch.zeros_like(input_batch)
        outputs = {
            "x_hat": prediction,
            "image_recon": prediction,
            "camera_plane": prediction,
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }

        base_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.5,
            camera_recon_weight=1.0,
            structure_recon_weight=0.0,
        )
        structure_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.5,
            camera_recon_weight=1.0,
            structure_recon_weight=0.2,
        )

        self.assertGreater(structure_losses["image_structure_loss"].item(), 0.0)
        self.assertGreater(structure_losses["camera_structure_loss"].item(), 0.0)
        self.assertGreater(structure_losses["structure_loss"].item(), 0.0)
        self.assertAlmostEqual(
            structure_losses["loss"].item(),
            base_losses["loss"].item() + structure_losses["structure_loss"].item(),
            places=6,
        )

    def test_first_stage_loss_adds_weighted_shape_term(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        input_batch[:, :, 1:4, 1:4] = 1.0
        prediction = torch.zeros_like(input_batch)
        outputs = {
            "x_hat": prediction,
            "image_recon": prediction,
            "camera_plane": prediction,
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }

        base_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.5,
            camera_recon_weight=1.0,
            shape_recon_weight=0.0,
        )
        shape_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.5,
            camera_recon_weight=1.0,
            shape_recon_weight=0.2,
        )

        self.assertGreater(shape_losses["image_shape_loss"].item(), 0.0)
        self.assertGreater(shape_losses["camera_shape_loss"].item(), 0.0)
        self.assertGreater(shape_losses["shape_loss"].item(), 0.0)
        self.assertAlmostEqual(
            shape_losses["loss"].item(),
            base_losses["loss"].item() + shape_losses["shape_loss"].item(),
            places=6,
        )

    def test_first_stage_hard_class_weights_increase_camera_terms_only_for_selected_labels(self) -> None:
        import torch

        from ldm_project.first_stage_losses import hard_class_sample_weights
        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((2, 1, 6, 6), dtype=torch.float32)
        input_batch[0, :, 1:5, 1:5] = 1.0
        input_batch[1, :, 2:4, 2:4] = 0.2
        prediction = torch.zeros_like(input_batch)
        prediction[1, :, 2:4, 2:4] = 0.2
        outputs = {
            "x_hat": prediction,
            "image_recon": prediction,
            "camera_plane": prediction,
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }
        labels = torch.tensor([4, 1], dtype=torch.long)

        weights = hard_class_sample_weights(
            labels,
            hard_class_weight_labels=(4,),
            hard_class_weight=2.0,
            dtype=torch.float32,
        )
        self.assertEqual(weights.tolist(), [2.0, 1.0])

        base_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
            shape_recon_weight=0.5,
            camera_topology_weight=0.25,
            camera_artifact_weight=0.25,
            class_labels=labels,
            hard_class_weight_labels=(4,),
            hard_class_weight=1.0,
        )
        weighted_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
            shape_recon_weight=0.5,
            camera_topology_weight=0.25,
            camera_artifact_weight=0.25,
            class_labels=labels,
            hard_class_weight_labels=(4,),
            hard_class_weight=2.0,
        )

        self.assertGreater(weighted_losses["camera_recon_loss"].item(), base_losses["camera_recon_loss"].item())
        self.assertGreater(weighted_losses["shape_loss"].item(), base_losses["shape_loss"].item())
        self.assertGreater(weighted_losses["camera_topology_loss"].item(), base_losses["camera_topology_loss"].item())
        self.assertNotAlmostEqual(
            weighted_losses["camera_artifact_loss"].item(),
            base_losses["camera_artifact_loss"].item(),
            places=6,
        )
        self.assertEqual(base_losses["hard_class_weight_mean"].item(), 1.0)
        self.assertAlmostEqual(weighted_losses["hard_class_weight_mean"].item(), 1.5, places=6)
        self.assertTrue(torch.isfinite(weighted_losses["loss"]).item())

    def test_first_stage_loss_adds_prototype_margin_for_requested_labels(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        prototype_0 = torch.zeros((1, 8, 8), dtype=torch.float32)
        prototype_0[:, 1:4, 1:4] = 1.0
        prototype_1 = torch.zeros((1, 8, 8), dtype=torch.float32)
        prototype_1[:, 4:7, 4:7] = 1.0
        prototypes = torch.stack([prototype_0, prototype_1], dim=0)
        labels = torch.tensor([0, 1], dtype=torch.long)
        matching_camera = prototypes.index_select(0, labels).clone()
        wrong_camera = prototypes.index_select(0, torch.tensor([1, 0], dtype=torch.long)).clone().requires_grad_(True)

        def make_outputs(camera_plane: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "x_hat": torch.zeros_like(camera_plane),
                "image_recon": torch.zeros_like(camera_plane),
                "camera_plane": camera_plane,
                "mu": torch.zeros_like(camera_plane),
                "logvar": torch.zeros_like(camera_plane),
            }

        matching_losses = first_stage_loss(
            make_outputs(matching_camera),
            torch.zeros_like(matching_camera),
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=0.0,
            class_labels=labels,
            camera_label_margin_weight=2.0,
            camera_label_margin_target=0.08,
            camera_label_margin_labels=(0, 1),
            camera_label_margin_prototypes=prototypes,
        )
        wrong_losses = first_stage_loss(
            make_outputs(wrong_camera),
            torch.zeros_like(wrong_camera),
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=0.0,
            class_labels=labels,
            camera_label_margin_weight=2.0,
            camera_label_margin_target=0.08,
            camera_label_margin_labels=(0, 1),
            camera_label_margin_prototypes=prototypes,
        )

        self.assertGreater(matching_losses["camera_label_margin_mean"].item(), 0.08)
        self.assertEqual(matching_losses["camera_label_margin_loss"].item(), 0.0)
        self.assertGreater(wrong_losses["camera_label_margin_loss"].item(), 0.0)
        self.assertLess(wrong_losses["camera_label_margin_mean"].item(), matching_losses["camera_label_margin_mean"].item())
        wrong_losses["loss"].backward()
        self.assertIsNotNone(wrong_camera.grad)
        self.assertGreater(wrong_camera.grad.abs().sum().item(), 0.0)

    def test_first_stage_hard_class_config_validates_labels_and_weight(self) -> None:
        from ldm_project.config import load_training_config

        base_yaml = """
training:
  mode: first_stage
first_stage:
  hard_class_weight_labels: [{label}]
  hard_class_weight: {weight}
diffusion:
  num_classes: 10
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_label = Path(tmpdir) / "invalid_label.yaml"
            invalid_label.write_text(base_yaml.format(label=10, weight=2.0), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hard_class_weight_labels"):
                load_training_config(invalid_label)

            invalid_weight = Path(tmpdir) / "invalid_weight.yaml"
            invalid_weight.write_text(base_yaml.format(label=4, weight=0.0), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hard_class_weight"):
                load_training_config(invalid_weight)

    def test_first_stage_loss_adds_topology_and_saturation_terms(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        input_batch[:, :, 3:13, 4:6] = 1.0
        input_batch[:, :, 3:5, 4:12] = 1.0
        input_batch[:, :, 8:10, 4:12] = 1.0
        input_batch[:, :, 11:13, 4:12] = 1.0
        shifted_prediction = torch.roll(input_batch, shifts=3, dims=-1)
        mu = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        logvar = torch.zeros_like(mu)
        matching_outputs = {
            "x_hat": input_batch,
            "image_recon": input_batch,
            "camera_plane": input_batch,
            "camera_plane_loss": input_batch,
            "mu": mu,
            "logvar": logvar,
        }
        shifted_outputs = {
            "x_hat": shifted_prediction,
            "image_recon": shifted_prediction,
            "camera_plane": shifted_prediction,
            "camera_plane_loss": shifted_prediction + 0.4,
            "mu": mu,
            "logvar": logvar,
        }

        matching_losses = first_stage_loss(
            matching_outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            camera_topology_weight=0.5,
            camera_saturation_weight=0.25,
        )
        shifted_losses = first_stage_loss(
            shifted_outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            camera_topology_weight=0.5,
            camera_saturation_weight=0.25,
        )

        for key in ("camera_topology_loss", "camera_topology_base", "camera_saturation_loss", "camera_saturation_penalty"):
            self.assertTrue(torch.isfinite(shifted_losses[key]).item(), key)
        self.assertGreater(shifted_losses["camera_topology_base"].item(), matching_losses["camera_topology_base"].item())
        self.assertGreater(
            shifted_losses["camera_saturation_penalty"].item(),
            matching_losses["camera_saturation_penalty"].item(),
        )
        self.assertGreater(shifted_losses["loss"].item(), matching_losses["loss"].item())

    def test_first_stage_loss_adds_foreground_artifact_term(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        input_batch[:, :, 4:12, 4:12] = 1.0
        smooth_prediction = input_batch * 0.85
        artifact_prediction = smooth_prediction.clone()
        artifact_prediction[:, :, 7:9, 7:9] = 0.0
        artifact_prediction[:, :, 4:12, 5:12:2] *= 0.2
        mu = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        logvar = torch.zeros_like(mu)

        def make_outputs(camera_plane: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "x_hat": camera_plane,
                "image_recon": camera_plane,
                "camera_plane": camera_plane,
                "mu": mu,
                "logvar": logvar,
            }

        default_losses = first_stage_loss(
            make_outputs(artifact_prediction),
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=0.0,
        )
        smooth_losses = first_stage_loss(
            make_outputs(smooth_prediction),
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=0.0,
            camera_artifact_weight=0.5,
        )
        artifact_losses = first_stage_loss(
            make_outputs(artifact_prediction),
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=0.0,
            camera_artifact_weight=0.5,
        )

        self.assertEqual(default_losses["camera_artifact_loss"].item(), 0.0)
        self.assertEqual(default_losses["loss"].item(), 0.0)
        self.assertGreater(artifact_losses["camera_artifact_loss"].item(), smooth_losses["camera_artifact_loss"].item())
        self.assertGreater(
            artifact_losses["camera_foreground_hole_ratio"].item(),
            smooth_losses["camera_foreground_hole_ratio"].item(),
        )
        self.assertGreater(
            artifact_losses["camera_foreground_roughness"].item(),
            smooth_losses["camera_foreground_roughness"].item(),
        )
        self.assertAlmostEqual(
            artifact_losses["loss"].item(),
            artifact_losses["camera_artifact_loss"].item(),
            places=6,
        )

    def test_first_stage_loss_adds_field_phase_and_background_terms(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        camera_plane = torch.zeros_like(input_batch)
        camera_plane[:, :, :, 2:] = 0.5
        camera_plane_loss = camera_plane.clone()
        camera_plane_loss[:, :, :, 3:] = 1.5
        chip_real = torch.zeros_like(input_batch)
        chip_real[:, :, :, 2:] = 1.0
        chip_imag = torch.zeros_like(input_batch)
        chip_phase = torch.zeros_like(input_batch)
        chip_phase[:, :, 2:, :] = torch.pi
        outputs = {
            "x_hat": camera_plane,
            "image_recon": torch.zeros_like(input_batch),
            "camera_plane": camera_plane,
            "camera_plane_loss": camera_plane_loss,
            "chip_real": chip_real,
            "chip_imag": chip_imag,
            "chip_phase": chip_phase,
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }

        base_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
        )
        regularized_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            field_smoothness_weight=0.2,
            phase_smoothness_weight=0.3,
            background_leakage_weight=0.4,
            field_energy_weight=0.7,
            camera_hotspot_weight=0.5,
            camera_saturation_weight=0.6,
        )

        self.assertGreater(regularized_losses["field_smoothness_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["phase_smoothness_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["background_leakage_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["field_energy_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["camera_hotspot_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["camera_saturation_loss"].item(), 0.0)
        self.assertGreater(regularized_losses["camera_background_leakage"].item(), 0.0)
        self.assertGreater(regularized_losses["camera_hotspot_ratio"].item(), 0.0)
        self.assertGreater(regularized_losses["field_tv"].item(), 0.0)
        self.assertGreater(regularized_losses["field_energy"].item(), 0.0)
        self.assertGreater(regularized_losses["phase_tv"].item(), 0.0)
        self.assertGreaterEqual(regularized_losses["camera_saturation_ratio"].item(), 0.0)
        self.assertGreater(regularized_losses["loss"].item(), base_losses["loss"].item())

    def test_first_stage_loss_uses_clamped_camera_for_mnist_reconstruction(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        outputs = {
            "x_hat": torch.ones_like(input_batch),
            "image_recon": torch.ones_like(input_batch),
            "camera_plane": torch.ones_like(input_batch),
            "camera_plane_loss": torch.full_like(input_batch, 2.0),
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }

        losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
            camera_hotspot_weight=0.0,
        )
        hotspot_losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
            camera_hotspot_weight=0.5,
        )

        self.assertEqual(losses["camera_recon_loss"].item(), 0.0)
        self.assertGreater(hotspot_losses["camera_hotspot_loss"].item(), 0.0)
        self.assertGreater(hotspot_losses["loss"].item(), losses["loss"].item())

    def test_conditioned_diffusion_loss_smoke(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig, FirstStageConfig, PhysicsConfig
        from ldm_project.diffusion import LatentDiffusionUNet, LinearNoiseScheduler
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.training import diffusion_loss, estimate_latent_scale

        first_stage = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        diffusion_model = LatentDiffusionUNet(4, DiffusionConfig())
        scheduler = LinearNoiseScheduler(timesteps=16)

        batch = torch.rand(2, 1, 200, 200)
        labels = torch.tensor([1, 8], dtype=torch.long)
        latent_scale = estimate_latent_scale(first_stage, batch)

        diffusion_model.train()
        loss = diffusion_loss(first_stage, diffusion_model, scheduler, batch, labels, latent_scale)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_diffusion_loss_upweights_requested_classes(self) -> None:
        from unittest import mock

        import torch

        import ldm_project.diffusion_losses as diffusion_losses

        labels = torch.tensor([3, 4], dtype=torch.long)
        per_sample_losses = torch.tensor([1.0, 2.0])
        with mock.patch.object(diffusion_losses, "diffusion_per_sample_losses", return_value=per_sample_losses):
            unweighted = diffusion_losses.diffusion_loss(
                first_stage=None,
                diffusion_model=None,
                scheduler=None,
                batch=torch.zeros((2, 1, 2, 2)),
                class_labels=labels,
                latent_scale=torch.tensor(1.0),
            )
            weighted = diffusion_losses.diffusion_loss(
                first_stage=None,
                diffusion_model=None,
                scheduler=None,
                batch=torch.zeros((2, 1, 2, 2)),
                class_labels=labels,
                latent_scale=torch.tensor(1.0),
                class_weight_labels=(4,),
                class_weight=3.0,
            )

        self.assertAlmostEqual(unweighted.item(), 1.5, places=6)
        self.assertAlmostEqual(weighted.item(), 1.75, places=6)
        self.assertGreater(weighted.item(), unweighted.item())

    def test_conditioned_sampling_and_decode_smoke(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig, FirstStageConfig, PhysicsConfig
        from ldm_project.diffusion import LatentDiffusionUNet, LinearNoiseScheduler
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.training import estimate_latent_scale, sample_latents, sample_latents_with_trace

        first_stage = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        diffusion_model = LatentDiffusionUNet(4, DiffusionConfig()).eval()
        scheduler = LinearNoiseScheduler(timesteps=16)

        reference_batch = torch.rand(2, 1, 200, 200)
        labels = torch.tensor([2, 9], dtype=torch.long)
        latent_scale = estimate_latent_scale(first_stage, reference_batch)
        latent_shape = (2, 4, 25, 25)

        latents = sample_latents(
            diffusion_model,
            scheduler,
            latent_shape,
            torch.device("cpu"),
            labels,
            guidance_scale=2.0,
        )
        decoded = first_stage.decode(latents * latent_scale)

        self.assertEqual(latents.shape, latent_shape)
        self.assertEqual(decoded.shape, (2, 1, 200, 200))

        traced_latents, trace = sample_latents_with_trace(
            diffusion_model,
            scheduler,
            latent_shape,
            torch.device("cpu"),
            labels,
            guidance_scale=2.0,
        )

        self.assertEqual(traced_latents.shape, latent_shape)
        self.assertEqual(list(trace.keys()), ["noise", "t750", "t500", "t250", "t100", "t050", "t000"])
        for trace_latents in trace.values():
            self.assertEqual(trace_latents.shape, latent_shape)

        fast_latents = sample_latents(
            diffusion_model,
            scheduler,
            latent_shape,
            torch.device("cpu"),
            labels,
            guidance_scale=2.0,
            sampling_steps=4,
        )

        self.assertEqual(fast_latents.shape, latent_shape)
        self.assertTrue(torch.isfinite(fast_latents).all().item())

    def test_first_stage_loss_weights_foreground_errors_more_than_background(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.tensor([[[[0.1, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        mu = torch.zeros_like(input_batch)
        logvar = torch.zeros_like(input_batch)
        fg_outputs = {
            "x_hat": torch.tensor([[[[0.3, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            "image_recon": torch.tensor([[[[0.3, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            "camera_plane": torch.tensor([[[[0.3, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            "mu": mu,
            "logvar": logvar,
        }
        bg_outputs = {
            "x_hat": torch.tensor([[[[0.1, 0.2], [0.0, 0.0]]]], dtype=torch.float32),
            "image_recon": torch.tensor([[[[0.1, 0.2], [0.0, 0.0]]]], dtype=torch.float32),
            "camera_plane": torch.tensor([[[[0.1, 0.2], [0.0, 0.0]]]], dtype=torch.float32),
            "mu": mu,
            "logvar": logvar,
        }

        fg_losses = first_stage_loss(
            fg_outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            foreground_threshold=0.05,
            foreground_weight=4.0,
            background_weight=1.0,
        )
        bg_losses = first_stage_loss(
            bg_outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            foreground_threshold=0.05,
            foreground_weight=4.0,
            background_weight=1.0,
        )

        self.assertGreater(fg_losses["recon_loss"].item(), bg_losses["recon_loss"].item())
        self.assertAlmostEqual(fg_losses["fg_pixel_ratio"].item(), 0.25, places=6)
        self.assertGreater(fg_losses["image_fg_mse"].item(), 0.0)
        self.assertEqual(bg_losses["image_fg_mse"].item(), 0.0)

    def test_first_stage_loss_keeps_camera_branch_primary(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        mu = torch.zeros_like(input_batch)
        logvar = torch.zeros_like(input_batch)
        outputs = {
            "x_hat": torch.zeros_like(input_batch),
            "image_recon": torch.zeros_like(input_batch),
            "camera_plane": input_batch.clone(),
            "mu": mu,
            "logvar": logvar,
        }

        losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            foreground_threshold=0.05,
            foreground_weight=4.0,
            background_weight=1.0,
            image_recon_weight=0.25,
            camera_recon_weight=1.0,
        )

        self.assertGreater(losses["image_recon_loss"].item(), 0.0)
        self.assertEqual(losses["camera_recon_loss"].item(), 0.0)
        self.assertAlmostEqual(
            losses["recon_loss"].item(),
            losses["image_recon_loss"].item() * 0.25,
            places=6,
        )

    def test_first_stage_loss_all_black_input_keeps_metrics_finite(self) -> None:
        import torch

        from ldm_project.training import first_stage_loss

        input_batch = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        outputs = {
            "x_hat": torch.zeros_like(input_batch),
            "image_recon": torch.zeros_like(input_batch),
            "camera_plane": torch.zeros_like(input_batch),
            "mu": torch.zeros_like(input_batch),
            "logvar": torch.zeros_like(input_batch),
        }
        losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=1,
            target_beta=0.0,
            warmup_ratio=0.1,
            foreground_threshold=0.05,
            foreground_weight=4.0,
            background_weight=1.0,
        )

        for key in (
            "recon_loss",
            "image_fg_mse",
            "image_bg_mse",
            "camera_fg_mse",
            "camera_bg_mse",
            "pred_fg_energy_ratio",
        ):
            self.assertTrue(torch.isfinite(losses[key]).item(), key)

    def test_first_stage_overfit_regression_recovers_sparse_patterns(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.training import first_stage_loss

        torch.manual_seed(0)
        config = FirstStageConfig(
            image_size=32,
            latent_channels=2,
            encoder_channels=(8, 16, 32),
        )
        model = OpticalPathVAE(PhysicsConfig(), config)
        optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-3)

        batch = torch.zeros(4, 1, 32, 32)
        batch[0, 0, 6:26, 8:10] = 1.0
        batch[0, 0, 6:8, 8:22] = 1.0
        batch[1, 0, 6:26, 14:18] = 1.0
        batch[2, 0, 6:8, 8:24] = 1.0
        batch[2, 0, 14:16, 8:24] = 1.0
        batch[2, 0, 24:26, 8:24] = 1.0
        batch[2, 0, 8:14, 20:24] = 1.0
        batch[2, 0, 16:24, 8:12] = 1.0
        batch[3, 0, 6:8, 8:24] = 1.0
        batch[3, 0, 14:16, 8:24] = 1.0
        batch[3, 0, 24:26, 8:24] = 1.0
        batch[3, 0, 8:14, 20:24] = 1.0
        batch[3, 0, 16:24, 20:24] = 1.0

        # 为了让回归测试稳定，这里固定用 mu 作为 latent，专注检查结构是否具备过拟合数字形状的能力。
        model.reparameterize = lambda mu, logvar: mu

        camera_losses: list[float] = []
        model.train()
        for step in range(40):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = first_stage_loss(
                outputs,
                batch,
                global_step=step,
                total_steps=40,
                target_beta=0.0,
                warmup_ratio=0.1,
                foreground_threshold=config.recon_foreground_threshold,
                foreground_weight=config.recon_foreground_weight,
                background_weight=config.recon_background_weight,
                image_recon_weight=config.image_recon_weight,
                camera_recon_weight=config.camera_recon_weight,
            )
            losses["loss"].backward()
            optimizer.step()
            camera_losses.append(losses["camera_recon_loss"].item())

        self.assertLess(camera_losses[-1], camera_losses[0] * 0.6)


if __name__ == "__main__":
    unittest.main()
