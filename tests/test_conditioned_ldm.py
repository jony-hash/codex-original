from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import warnings


def load_train_module():
    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_under_test", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train.py from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiffusionConfigTests(unittest.TestCase):
    def test_load_training_config_reads_new_diffusion_fields(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
training:
  mode: diffusion
  visual_save_epoch_interval: 3
first_stage:
  recon_foreground_threshold: 0.08
  recon_foreground_weight: 5.0
  recon_background_weight: 0.5
  encoder_fusion_mode: optical_only
  optical_field_mode: complex_cartesian
  primary_output: camera
  image_recon_weight: 0.2
  camera_recon_weight: 1.3
  electronic_domain:
    phase_scale_rad: 1.57
    phase_bias_rad: 0.25
diffusion:
  base_channels: 32
  channel_multipliers: [1, 2, 4]
  num_res_blocks: 3
  attention_levels: [1, 2]
  num_heads: 8
  context_dim: 64
  num_classes: 10
  cond_dropout_prob: 0.25
  latent_source: mu
  guidance_scale: 2.5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            loaded = load_training_config(config_path)

        self.assertEqual(loaded.diffusion.base_channels, 32)
        self.assertEqual(loaded.diffusion.channel_multipliers, (1, 2, 4))
        self.assertEqual(loaded.diffusion.num_res_blocks, 3)
        self.assertEqual(loaded.diffusion.attention_levels, (1, 2))
        self.assertEqual(loaded.diffusion.num_heads, 8)
        self.assertEqual(loaded.diffusion.context_dim, 64)
        self.assertEqual(loaded.diffusion.num_classes, 10)
        self.assertEqual(loaded.diffusion.cond_dropout_prob, 0.25)
        self.assertEqual(loaded.diffusion.latent_source, "mu")
        self.assertAlmostEqual(loaded.diffusion.guidance_scale, 2.5, places=6)
        self.assertEqual(loaded.diffusion.timesteps, 1000)
        self.assertEqual(loaded.visual_save_epoch_interval, 3)
        self.assertAlmostEqual(loaded.data.validation_fraction, 0.1, places=6)
        self.assertEqual(loaded.data.validation_split_seed, 42)
        self.assertAlmostEqual(loaded.first_stage.recon_foreground_threshold, 0.08, places=6)
        self.assertAlmostEqual(loaded.first_stage.recon_foreground_weight, 5.0, places=6)
        self.assertAlmostEqual(loaded.first_stage.recon_background_weight, 0.5, places=6)
        self.assertEqual(loaded.first_stage.encoder_fusion_mode, "optical_only")
        self.assertEqual(loaded.first_stage.optical_field_mode, "complex_cartesian")
        self.assertEqual(loaded.first_stage.primary_output, "camera")
        self.assertAlmostEqual(loaded.first_stage.image_recon_weight, 0.2, places=6)
        self.assertAlmostEqual(loaded.first_stage.camera_recon_weight, 1.3, places=6)
        self.assertEqual(loaded.first_stage.electronic_domain.mode, "mnist_complex_slm")
        self.assertEqual(loaded.first_stage.electronic_domain.background_floor, 0.05)
        self.assertEqual(loaded.first_stage.electronic_domain.gamma, 1.0)
        self.assertEqual(loaded.first_stage.electronic_domain.polarity, "non_inverted")
        self.assertAlmostEqual(loaded.first_stage.electronic_domain.phase_scale_rad, 1.57, places=6)
        self.assertAlmostEqual(loaded.first_stage.electronic_domain.phase_bias_rad, 0.25, places=6)

    def test_training_defaults_reflect_epoch_first(self) -> None:
        from ldm_project.config import DiffusionConfig, PhysicsConfig, TrainingConfig
        from ldm_project.diffusion import LinearNoiseScheduler

        training_cfg = TrainingConfig()
        diffusion_cfg = DiffusionConfig()
        scheduler = LinearNoiseScheduler()
        physics_cfg = PhysicsConfig()

        self.assertEqual(training_cfg.mode, "first_stage")
        self.assertEqual(training_cfg.visual_save_epoch_interval, 1)
        self.assertEqual(training_cfg.first_stage_epochs, 10)
        self.assertEqual(training_cfg.diffusion_epochs, 20)
        self.assertAlmostEqual(training_cfg.data.validation_fraction, 0.1, places=6)
        self.assertEqual(training_cfg.data.validation_split_seed, 42)
        self.assertAlmostEqual(training_cfg.first_stage.recon_foreground_threshold, 0.05, places=6)
        self.assertAlmostEqual(training_cfg.first_stage.recon_foreground_weight, 4.0, places=6)
        self.assertAlmostEqual(training_cfg.first_stage.recon_background_weight, 1.0, places=6)
        self.assertEqual(training_cfg.first_stage.encoder_fusion_mode, "optical_only")
        self.assertEqual(training_cfg.first_stage.optical_field_mode, "complex_cartesian")
        self.assertEqual(training_cfg.first_stage.primary_output, "camera")
        self.assertAlmostEqual(training_cfg.first_stage.image_recon_weight, 0.25, places=6)
        self.assertAlmostEqual(training_cfg.first_stage.camera_recon_weight, 1.0, places=6)
        self.assertEqual(diffusion_cfg.latent_source, "mu")
        self.assertAlmostEqual(diffusion_cfg.guidance_scale, 3.0, places=6)
        self.assertEqual(diffusion_cfg.timesteps, 1000)
        self.assertEqual(scheduler.timesteps, 1000)
        self.assertEqual(physics_cfg.optics.wavelength, 532.0e-9)
        self.assertEqual(training_cfg.first_stage.electronic_domain.mode, "mnist_complex_slm")
        self.assertAlmostEqual(training_cfg.first_stage.electronic_domain.phase_scale_rad, math.pi, places=6)
        self.assertAlmostEqual(training_cfg.first_stage.electronic_domain.phase_bias_rad, 0.0, places=6)

    def test_hidden_channels_maps_to_base_channels(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
training:
  mode: diffusion
diffusion:
  hidden_channels: 48
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            loaded = load_training_config(config_path)

        self.assertEqual(loaded.diffusion.base_channels, 48)

    def test_legacy_amplitude_slm_mode_still_loads(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
first_stage:
  electronic_domain:
    mode: mnist_amplitude_slm
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            loaded = load_training_config(config_path)

        self.assertEqual(loaded.first_stage.electronic_domain.mode, "mnist_amplitude_slm")

    def test_legacy_smoke_steps_raises_migration_error(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
training:
  smoke_steps: 1000
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "training.smoke_steps"):
                load_training_config(config_path)

    def test_legacy_visual_save_interval_raises_migration_error(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
training:
  visual_save_interval: 200
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "training.visual_save_interval"):
                load_training_config(config_path)

    def test_invalid_validation_fraction_raises_clear_error(self) -> None:
        from ldm_project.config import load_training_config

        config_text = """
data:
  validation_fraction: 1.0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "data.validation_fraction"):
                load_training_config(config_path)

    def test_resolve_class_labels_broadcasts_and_validates(self) -> None:
        import torch

        train_module = load_train_module()
        build_visualization_labels = train_module.build_visualization_labels
        resolve_class_labels = train_module.resolve_class_labels
        resolve_device = train_module.resolve_device

        labels, source = resolve_class_labels(torch, [2], num_samples=3, num_classes=10, device=torch.device("cpu"))
        self.assertEqual(labels.tolist(), [2, 2, 2])
        self.assertEqual(source, "user_provided")

        random_labels, random_source = resolve_class_labels(
            torch,
            None,
            num_samples=4,
            num_classes=10,
            device=torch.device("cpu"),
        )
        self.assertEqual(random_labels.shape, (4,))
        self.assertTrue(torch.all((random_labels >= 0) & (random_labels < 10)).item())
        self.assertEqual(random_source, "random_uniform")

        with self.assertRaisesRegex(ValueError, "class-labels"):
            resolve_class_labels(torch, [1, 2], num_samples=3, num_classes=10, device=torch.device("cpu"))

        resolved_device = resolve_device(torch, "auto")
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolved_device.type, expected_type)

        visualization_labels = build_visualization_labels(torch, 8, 3, torch.device("cpu"))
        self.assertEqual(visualization_labels.tolist(), [0, 1, 2, 0, 1, 2, 0, 1])

    def test_epoch_visual_save_config_and_logic(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from ldm_project.data import DatasetBundle

        train_module = load_train_module()
        build_first_stage_visualization_tensors = train_module.build_first_stage_visualization_tensors
        build_diffusion_visualization_tensors = train_module.build_diffusion_visualization_tensors
        resolve_stage_total_steps = train_module.resolve_stage_total_steps
        select_reference_images_for_labels = train_module.select_reference_images_for_labels
        should_save_epoch_visual = train_module.should_save_epoch_visual
        validate_stage_requirements = train_module.validate_stage_requirements

        self.assertTrue(should_save_epoch_visual(0, 3))
        self.assertFalse(should_save_epoch_visual(1, 3))
        self.assertTrue(should_save_epoch_visual(2, 3))
        with self.assertRaisesRegex(ValueError, "visual_save_epoch_interval"):
            should_save_epoch_visual(0, 0)

        visuals = build_first_stage_visualization_tensors(
            torch.rand(2, 1, 4, 4),
            {
                "pf2_intensity": torch.rand(2, 1, 4, 4),
                "image_recon": torch.rand(2, 1, 4, 4),
                "chip_amplitude": torch.rand(2, 1, 4, 4),
                "chip_phase": torch.rand(2, 1, 4, 4) * math.pi,
                "probe_plane": torch.rand(2, 1, 4, 4),
                "camera_plane": torch.rand(2, 1, 4, 4),
            },
        )
        self.assertEqual(
            list(visuals.keys()),
            ["input", "camera_hat", "image_hat", "pf2", "chip_amplitude", "chip_phase", "probe"],
        )

        diffusion_visuals = build_diffusion_visualization_tensors(
            torch.rand(2, 1, 4, 4),
            {
                "image_recon": torch.rand(2, 1, 4, 4),
                "chip_amplitude": torch.rand(2, 1, 4, 4),
                "chip_phase": torch.rand(2, 1, 4, 4) * math.pi,
                "probe_plane": torch.rand(2, 1, 4, 4),
                "camera_plane": torch.rand(2, 1, 4, 4),
            },
        )
        self.assertEqual(
            list(diffusion_visuals.keys()),
            [
                "reference (参考)",
                "generated_camera (生成相机)",
                "generated_image (生成图像)",
                "chip_amplitude (芯片振幅)",
                "chip_phase (芯片相位)",
                "probe (探测平面)",
            ],
        )
        self.assertEqual(
            train_module.format_output_title("images/first_stage/epoch_0001.png", "first_stage epoch 1"),
            "images/first_stage/epoch_0001.png | first_stage epoch 1",
        )

        train_images = torch.stack(
            [
                torch.zeros(1, 4, 4),
                torch.full((1, 4, 4), 0.25),
            ]
        )
        train_labels = torch.tensor([0, 3], dtype=torch.long)
        validation_images = torch.stack(
            [
                torch.full((1, 4, 4), 0.5),
                torch.full((1, 4, 4), 0.75),
            ]
        )
        validation_labels = torch.tensor([2, 1], dtype=torch.long)
        train_bundle = DatasetBundle(
            dataset=TensorDataset(train_images, train_labels),
            loader=DataLoader(TensorDataset(train_images, train_labels), batch_size=2, shuffle=False),
            effective_size=2,
            total_size=2,
        )
        validation_bundle = DatasetBundle(
            dataset=TensorDataset(validation_images, validation_labels),
            loader=DataLoader(TensorDataset(validation_images, validation_labels), batch_size=2, shuffle=False),
            effective_size=2,
            total_size=2,
        )
        reference_images, reference_labels, reference_source = select_reference_images_for_labels(
            torch,
            torch.tensor([0, 2, 3, 0], dtype=torch.long),
            validation_bundle,
            train_bundle,
            torch.device("cpu"),
        )
        self.assertEqual(reference_labels.tolist(), [0, 2, 3, 0])
        self.assertEqual(reference_source, "validation_then_train_fallback")
        self.assertTrue(torch.equal(reference_images[0], train_images[0]))
        self.assertTrue(torch.equal(reference_images[1], validation_images[0]))
        self.assertTrue(torch.equal(reference_images[2], train_images[1]))
        self.assertTrue(torch.equal(reference_images[3], train_images[0]))

        self.assertEqual(resolve_stage_total_steps(loader_length=4, epochs=2), 8)
        with self.assertRaisesRegex(ValueError, "stage epochs"):
            resolve_stage_total_steps(loader_length=4, epochs=0)
        with self.assertRaisesRegex(ValueError, "Training loader is empty"):
            resolve_stage_total_steps(loader_length=0, epochs=1)
        with self.assertRaisesRegex(ValueError, "first-stage-checkpoint"):
            validate_stage_requirements("diffusion", None)
        validate_stage_requirements("full", None)
        validate_stage_requirements("diffusion", "checkpoints/first_stage_latest.pt")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for checkpoint tests")
class CheckpointCompatibilityTests(unittest.TestCase):
    def test_first_stage_checkpoint_continues_to_load(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import (
            FIRST_STAGE_MODEL_VERSION,
            OpticalPathVAE,
            load_first_stage_checkpoint,
        )

        source_model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        target_model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "first_stage.pt"
            torch.save(
                {
                    "model_state_dict": source_model.state_dict(),
                    "model_version": FIRST_STAGE_MODEL_VERSION,
                },
                checkpoint_path,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                payload = torch.load(checkpoint_path, map_location="cpu")

        load_first_stage_checkpoint(target_model, payload)

    def test_old_diffusion_checkpoint_raises_clear_error(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig
        from ldm_project.diffusion import LatentDiffusionUNet, load_diffusion_checkpoint
        from ldm_project.first_stage import FIRST_STAGE_MODEL_VERSION

        model = LatentDiffusionUNet(4, DiffusionConfig())
        old_payload = {"model_state_dict": model.state_dict(), "latent_scale": torch.tensor(1.0)}

        with self.assertRaisesRegex(ValueError, "Incompatible diffusion checkpoint"):
            load_diffusion_checkpoint(model, old_payload, FIRST_STAGE_MODEL_VERSION)

    def test_old_first_stage_checkpoint_raises_clear_error(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE, load_first_stage_checkpoint

        model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        old_payload = {"model_state_dict": model.state_dict()}

        with self.assertRaisesRegex(ValueError, "Incompatible first-stage checkpoint"):
            load_first_stage_checkpoint(model, old_payload)


if __name__ == "__main__":
    unittest.main()
