from __future__ import annotations

import importlib.util
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
            "kl_loss",
            "fg_pixel_ratio",
            "pred_fg_energy_ratio",
        ):
            self.assertEqual(losses[key].ndim, 0, key)
            self.assertTrue(torch.isfinite(losses[key]).item(), key)

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

    def test_conditioned_sampling_and_decode_smoke(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig, FirstStageConfig, PhysicsConfig
        from ldm_project.diffusion import LatentDiffusionUNet, LinearNoiseScheduler
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.training import estimate_latent_scale, sample_latents

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
