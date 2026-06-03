from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


def _cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for one-shot optical student smoke tests.")
    return torch.device("cuda")


def _dummy_one_shot_outputs(batch_size: int, image_size: int, grid_size: int):
    import torch

    camera = torch.linspace(0.05, 0.95, steps=batch_size * image_size * image_size).reshape(
        batch_size, 1, image_size, image_size
    )
    full_phase = torch.linspace(-torch.pi, torch.pi, steps=camera.numel()).reshape_as(camera)
    grid = torch.linspace(0.0, 2.0, steps=batch_size * grid_size * grid_size).reshape(
        batch_size, 1, grid_size, grid_size
    )
    grid_phase = torch.linspace(-torch.pi, torch.pi, steps=grid.numel()).reshape_as(grid)
    raw_camera = torch.linspace(0.1, 1.1, steps=batch_size * grid_size * grid_size).reshape(
        batch_size, 1, grid_size, grid_size
    )
    layer_energy = torch.ones(batch_size, 2, dtype=torch.float32)
    return {
        "camera_plane": camera,
        "slm_command": camera.clamp(0.0, 1.0),
        "slm_amplitude": camera.clamp(0.0, 1.0),
        "slm_phase": torch.zeros_like(camera),
        "encoder_input_amplitude": camera * 2.0,
        "encoder_input_phase": full_phase,
        "encoder_input_grid_amplitude": grid * 0.8,
        "encoder_input_grid_phase": grid_phase,
        "encoded_amplitude": grid,
        "encoded_phase": grid_phase,
        "ols_input_amplitude": grid,
        "ols_input_phase": grid_phase,
        "ols_output_amplitude": grid * 0.5,
        "ols_output_phase": -grid_phase,
        "ols_amplitude": grid * 0.5,
        "ols_phase": -grid_phase,
        "decoder_input_amplitude": grid * 0.5,
        "decoder_input_phase": -grid_phase,
        "decoder_output_amplitude": grid * 1.25,
        "decoder_output_phase": grid_phase.roll(shifts=1, dims=-1),
        "chip_amplitude": grid * 3.0,
        "chip_phase": grid_phase,
        "camera_phase": grid_phase,
        "diffractive_camera_intensity_raw": raw_camera,
        "diffractive_camera_plane_raw": raw_camera.clamp(0.0, 1.0),
        "diffractive_camera_plane_loss_raw": raw_camera,
        "diffractive_encoder_layer_amplitude": grid.unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_encoder_layer_phase": grid_phase.unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_decoder_layer_amplitude": (grid * 1.1).unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_decoder_layer_phase": grid_phase.neg().unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_encoder_mask_amplitude": grid.new_ones(2, 1, grid_size, grid_size),
        "diffractive_encoder_mask_phase": grid_phase[:2],
        "diffractive_decoder_mask_amplitude": grid.new_ones(2, 1, grid_size, grid_size),
        "diffractive_decoder_mask_phase": -grid_phase[:2],
        "diffractive_encoder_crop_input_energy": layer_energy * 4.0,
        "diffractive_encoder_crop_padded_input_energy": layer_energy * 4.0,
        "diffractive_encoder_crop_propagated_full_energy": layer_energy * 3.0,
        "diffractive_encoder_crop_cropped_energy": layer_energy,
        "diffractive_encoder_crop_crop_retention": layer_energy * 0.25,
        "diffractive_encoder_crop_propagation_energy_ratio": layer_energy * 0.75,
        "diffractive_decoder_crop_input_energy": layer_energy * 2.0,
        "diffractive_decoder_crop_padded_input_energy": layer_energy * 2.0,
        "diffractive_decoder_crop_propagated_full_energy": layer_energy * 1.5,
        "diffractive_decoder_crop_cropped_energy": layer_energy * 0.5,
        "diffractive_decoder_crop_crop_retention": layer_energy * 0.25,
        "diffractive_decoder_crop_propagation_energy_ratio": layer_energy * 0.75,
    }


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class OneShotOpticalStudentTests(unittest.TestCase):
    def test_generator_outputs_amplitude_slm_command_on_cuda(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE, IdealAmplitudeSLMModel

        device = _cuda_device()
        generator = ElectronicGeneratorE(
            image_size=16,
            grid_size=4,
            code_dim=8,
            num_classes=3,
            label_embed_dim=4,
            hidden_dim=16,
        ).to(device)
        code = torch.randn(4, 8, device=device)
        labels = torch.tensor([0, 1, 2, 1], device=device)

        command = generator(code, labels)
        slm = IdealAmplitudeSLMModel(background_floor=0.05).to(device)
        slm_outputs = slm(command)

        self.assertEqual(tuple(command.shape), (4, 1, 16, 16))
        self.assertTrue(torch.all(command >= 0.0).item())
        self.assertTrue(torch.all(command <= 1.0).item())
        self.assertTrue(torch.all(slm_outputs["slm_amplitude"] >= 0.05).item())
        self.assertTrue(torch.all(slm_outputs["slm_amplitude"] <= 1.0).item())
        self.assertTrue(torch.allclose(slm_outputs["slm_phase"], torch.zeros_like(command)))
        self.assertTrue(torch.is_complex(slm_outputs["slm_transfer"]))

    def test_one_shot_forward_shapes_are_finite_on_cuda(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        device = _cuda_device()
        config = OneShotConfig(
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            metasurface_grid_size=8,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=16).to(device)
        code = torch.randn(2, config.code_dim, device=device)
        labels = torch.tensor([0, 2], device=device)

        outputs = model(code, labels)

        for key in (
            "camera_plane",
            "slm_command",
            "slm_amplitude",
            "slm_phase",
            "encoder_input_amplitude",
            "encoder_input_phase",
            "chip_amplitude",
            "chip_phase",
            "probe_plane",
            "camera_plane_loss",
        ):
            self.assertIn(key, outputs)
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 16, 16))
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
        for key in (
            "encoder_input_grid_amplitude",
            "encoder_input_grid_phase",
            "encoded_amplitude",
            "encoded_phase",
            "ols_input_amplitude",
            "ols_input_phase",
            "ols_output_amplitude",
            "ols_output_phase",
            "ols_amplitude",
            "ols_phase",
            "decoder_input_amplitude",
            "decoder_input_phase",
            "decoder_output_amplitude",
            "decoder_output_phase",
        ):
            self.assertIn(key, outputs)
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 8, 8))
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)

    def test_one_shot_loss_backpropagates_on_cuda(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.training import one_shot_loss

        device = _cuda_device()
        config = OneShotConfig(
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            metasurface_grid_size=8,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=16).to(device)
        code = torch.randn(2, config.code_dim, device=device)
        labels = torch.tensor([0, 1], device=device)
        target = torch.rand(2, 1, 16, 16, device=device)

        losses = one_shot_loss(model(code, labels), target)
        losses["loss"].backward()

        self.assertGreater(losses["loss"].item(), 0.0)
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_diffractive_one_shot_forward_and_gradients_on_cpu(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.training import one_shot_loss

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="identity",
            diffractive_grid_size=32,
            diffractive_propagation_grid_size=64,
            encoder_diffractive_layers=2,
            decoder_diffractive_layers=2,
            diffractive_layer_spacing=0.004,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=16)
        code = torch.randn(2, config.code_dim)
        labels = torch.tensor([0, 2])
        target = torch.rand(2, 1, 16, 16)

        outputs = model(code, labels)
        self.assertEqual(tuple(model.encoder.phase.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(model.decoder.phase.shape), (2, 1, 32, 32))
        self.assertGreaterEqual(float(model.encoder.phase.min().item()), -math.pi)
        self.assertLessEqual(float(model.encoder.phase.max().item()), math.pi)
        self.assertEqual(tuple(outputs["camera_plane"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(outputs["camera_plane_loss"].shape), (2, 1, 16, 16))
        for key in (
            "encoder_input_grid_amplitude",
            "decoder_output_amplitude",
            "decoder_output_phase",
            "chip_amplitude",
            "chip_phase",
            "chip_real",
            "chip_imag",
            "diffractive_camera_intensity_raw",
            "diffractive_camera_plane_raw",
            "diffractive_camera_plane_loss_raw",
        ):
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 32, 32), key)
        self.assertEqual(tuple(outputs["diffractive_encoder_layer_amplitude"].shape), (2, 2, 1, 32, 32))
        self.assertEqual(tuple(outputs["diffractive_decoder_layer_phase"].shape), (2, 2, 1, 32, 32))
        for key in (
            "diffractive_encoder_crop_input_energy",
            "diffractive_encoder_crop_padded_input_energy",
            "diffractive_encoder_crop_propagated_full_energy",
            "diffractive_encoder_crop_cropped_energy",
            "diffractive_encoder_crop_crop_retention",
            "diffractive_encoder_crop_propagation_energy_ratio",
            "diffractive_decoder_crop_input_energy",
            "diffractive_decoder_crop_padded_input_energy",
            "diffractive_decoder_crop_propagated_full_energy",
            "diffractive_decoder_crop_cropped_energy",
            "diffractive_decoder_crop_crop_retention",
            "diffractive_decoder_crop_propagation_energy_ratio",
        ):
            self.assertEqual(tuple(outputs[key].shape), (2, 2), key)
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
        for key in (
            "camera_plane",
            "camera_plane_loss",
            "diffractive_camera_intensity_raw",
            "diffractive_camera_plane_raw",
            "chip_amplitude",
        ):
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
        self.assertTrue(
            torch.allclose(
                outputs["diffractive_encoder_mask_amplitude"],
                torch.ones_like(outputs["diffractive_encoder_mask_amplitude"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["diffractive_decoder_mask_amplitude"],
                torch.ones_like(outputs["diffractive_decoder_mask_amplitude"]),
            )
        )
        self.assertTrue(torch.allclose(outputs["ols_input_amplitude"], outputs["ols_output_amplitude"]))
        self.assertTrue(torch.allclose(outputs["ols_input_phase"], outputs["ols_output_phase"]))

        losses = one_shot_loss(outputs, target)
        losses["loss"].backward()
        self.assertIsNotNone(model.encoder.phase.grad)
        self.assertIsNotNone(model.decoder.phase.grad)
        self.assertTrue(any(param.grad is not None for param in model.generator_e.parameters()))

    def test_diffractive_phase_initialization_modes(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        small_model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                diffractive_phase_init="small",
            ),
            image_size=8,
        )
        self.assertGreaterEqual(float(small_model.encoder.phase.min().item()), -0.05)
        self.assertLessEqual(float(small_model.encoder.phase.max().item()), 0.05)

        pi_model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                diffractive_phase_init="uniform_pi",
            ),
            image_size=8,
        )
        self.assertEqual(tuple(pi_model.encoder.phase.shape), (2, 1, 8, 8))
        self.assertGreaterEqual(float(pi_model.encoder.phase.min().item()), -torch.pi)
        self.assertLessEqual(float(pi_model.encoder.phase.max().item()), torch.pi)
        self.assertGreater(float(pi_model.encoder.phase.std(unbiased=False).item()), 0.5)

        with self.assertRaisesRegex(ValueError, "diffractive_phase_init"):
            OneShotOpticalGenerator(
                PhysicsConfig(),
                OneShotConfig(
                    optical_core_arch="diffractive_phase_only",
                    code_dim=4,
                    label_embed_dim=2,
                    hidden_dim=8,
                    diffractive_grid_size=8,
                    diffractive_propagation_grid_size=8,
                    diffractive_phase_init="bad",
                ),
                image_size=8,
            )

    def test_old_one_shot_checkpoint_version_is_rejected(self) -> None:
        from ldm_project.one_shot import validate_one_shot_checkpoint_payload

        with self.assertRaisesRegex(ValueError, "unexpected model_version"):
            validate_one_shot_checkpoint_payload(
                {
                    "model_version": "ldm_distilled_single_shot_optical_student_v2",
                    "model_state_dict": {},
                }
            )


class OneShotConfigTests(unittest.TestCase):
    @staticmethod
    def _minmax_norm(tensor):
        mn = tensor.amin(dim=(-2, -1), keepdim=True)
        mx = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - mn) / (mx - mn).clamp_min(1e-6)

    def test_smoke_config_loads_one_shot_fields(self) -> None:
        from ldm_project.config import load_training_config

        d2nn_config = load_training_config(Path("configs/train_one_shot_d2nn_smoke.yaml"))

        self.assertEqual(d2nn_config.mode, "one_shot")
        self.assertEqual(d2nn_config.one_shot_epochs, 1)
        self.assertEqual(d2nn_config.one_shot.code_dim, 8)
        self.assertEqual(d2nn_config.one_shot.label_embed_dim, 4)
        self.assertEqual(d2nn_config.one_shot.hidden_dim, 16)
        self.assertEqual(d2nn_config.one_shot.num_classes, 10)
        self.assertEqual(d2nn_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(d2nn_config.one_shot.ols_kind, "identity")
        self.assertEqual(d2nn_config.one_shot.d2nn.grid_size, 16)
        self.assertEqual(d2nn_config.one_shot.d2nn.propagation_grid_size, 64)
        self.assertEqual(d2nn_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(d2nn_config.one_shot.d2nn.decoder_layers, 2)
        self.assertAlmostEqual(d2nn_config.one_shot.d2nn.layer_spacing, 0.004, places=8)
        self.assertEqual(d2nn_config.one_shot.d2nn.modulation, "phase_only")
        self.assertEqual(d2nn_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertEqual(d2nn_config.one_shot.sample_class_labels, (0, 1))

    def test_distillation_configs_load(self) -> None:
        from ldm_project.config import load_training_config

        export_3k_fast200_config = load_training_config(Path("configs/export_teacher_distill_mnist_3k_fast200.yaml"))
        filter_config = load_training_config(Path("configs/filter_teacher_distill_mnist_3k_fast200_top2k.yaml"))
        train_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml"))
        feasibility_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_gpu_feasibility.yaml"))
        long_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_long.yaml"))

        self.assertEqual(export_3k_fast200_config.mode, "teacher_distill_export")
        self.assertEqual(export_3k_fast200_config.distillation.samples_per_class, 300)
        self.assertEqual(export_3k_fast200_config.distillation.sampling_steps, 200)
        self.assertEqual(filter_config.mode, "teacher_distill_filter")
        self.assertEqual(filter_config.distillation.samples_per_class, 300)
        self.assertEqual(filter_config.distillation.filter_keep_per_class, 200)
        self.assertAlmostEqual(filter_config.distillation.filter_foreground_trim_fraction, 0.05, places=8)
        self.assertAlmostEqual(filter_config.distillation.filter_contrast_drop_fraction, 0.10, places=8)
        self.assertAlmostEqual(filter_config.distillation.filter_foreground_threshold, 0.2, places=8)
        self.assertEqual(train_config.mode, "one_shot_distill")
        self.assertEqual(train_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(train_config.one_shot.ols_kind, "identity")
        self.assertIsNotNone(train_config.one_shot.d2nn)
        self.assertEqual(train_config.one_shot.d2nn.grid_size, 200)
        self.assertEqual(train_config.one_shot.d2nn.propagation_grid_size, 1024)
        self.assertEqual(train_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(train_config.one_shot.d2nn.decoder_layers, 2)
        self.assertAlmostEqual(train_config.one_shot.d2nn.layer_spacing, 0.005, places=8)
        self.assertEqual(train_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertEqual(train_config.first_stage.image_size, 200)
        self.assertEqual(train_config.one_shot.metasurface_grid_size, 40)
        self.assertAlmostEqual(train_config.one_shot.slm_smoothness_weight, 1.0e-4, places=8)
        self.assertAlmostEqual(train_config.one_shot.camera_saturation_weight, 1.0e-3, places=8)
        self.assertEqual(train_config.one_shot.checkpoint_selection_metric, "quality")
        self.assertEqual(train_config.one_shot.class_recon_weight_labels, (3, 4, 5, 6, 7, 8, 9))
        self.assertAlmostEqual(train_config.one_shot.shape_loss_weight, 5.0e-2, places=8)
        self.assertAlmostEqual(train_config.one_shot.edge_loss_weight, 2.0e-2, places=8)
        self.assertEqual(train_config.distillation.samples_per_class, 300)
        self.assertTrue(train_config.distillation.paired_class_batches)
        self.assertEqual(train_config.distillation.class_labels, tuple(range(10)))
        self.assertEqual(feasibility_config.mode, "one_shot_distill")
        self.assertEqual(feasibility_config.batch_size, 1)
        self.assertEqual(feasibility_config.one_shot_epochs, 1)
        self.assertEqual(feasibility_config.distillation.train_subset_size, 20)
        self.assertEqual(feasibility_config.one_shot.ols_kind, "identity")
        self.assertIsNotNone(feasibility_config.one_shot.d2nn)
        self.assertEqual(feasibility_config.one_shot.d2nn.grid_size, 200)
        self.assertEqual(feasibility_config.one_shot.d2nn.propagation_grid_size, 1024)
        self.assertAlmostEqual(feasibility_config.one_shot.d2nn.layer_spacing, 0.005, places=8)
        self.assertEqual(feasibility_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertAlmostEqual(feasibility_config.one_shot.cross_class_pairwise_weight, 1.0e-2, places=8)
        self.assertAlmostEqual(feasibility_config.one_shot.slm_contrast_weight, 1.0e-3, places=8)
        self.assertTrue(feasibility_config.distillation.paired_class_batches)
        self.assertTrue(feasibility_config.distillation.multiclass_batches)
        self.assertEqual(long_config.mode, "one_shot_distill")
        self.assertEqual(long_config.step_log_interval, 50)
        self.assertEqual(long_config.visual_save_epoch_interval, 5)
        self.assertEqual(long_config.validation_epoch_interval, 2)
        self.assertEqual(long_config.batch_size, 5)
        self.assertEqual(long_config.distillation.train_subset_size, 1000)
        self.assertAlmostEqual(long_config.one_shot.slm_contrast_weight, 1.0e-3, places=8)
        self.assertAlmostEqual(long_config.one_shot.cross_class_pairwise_weight, 1.0e-2, places=8)

    def test_distillation_train_subset_reduces_payload_deterministically(self) -> None:
        import torch

        from ldm_project.config import (
            DataConfig,
            DistillationConfig,
            FirstStageConfig,
            OneShotConfig,
            TrainingConfig,
        )
        from ldm_project.one_shot_distill_data import _subset_distill_payload_for_training

        payload = {
            "codes": torch.arange(30, dtype=torch.float32).reshape(10, 3),
            "class_labels": torch.arange(10, dtype=torch.long) % 2,
            "teacher_camera": torch.arange(10 * 16, dtype=torch.float32).reshape(10, 1, 4, 4),
            "teacher_initial_noise": torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1),
            "config": {"source": "unit-test"},
        }
        train_cfg = TrainingConfig(
            data=DataConfig(subset_seed=123, validation_fraction=0.2),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=3, num_classes=2),
            distillation=DistillationConfig(train_subset_size=4),
        )

        subset, source_size, effective_size = _subset_distill_payload_for_training(torch, payload, train_cfg)
        expected_indices = torch.randperm(10, generator=torch.Generator().manual_seed(123))[:4]

        self.assertEqual(source_size, 10)
        self.assertEqual(effective_size, 4)
        self.assertTrue(torch.equal(subset["codes"], payload["codes"].index_select(0, expected_indices)))
        self.assertTrue(
            torch.equal(
                subset["teacher_camera"],
                payload["teacher_camera"].index_select(0, expected_indices),
            )
        )
        self.assertEqual(subset["config"], payload["config"])

    def test_distillation_train_subset_rejects_invalid_sizes(self) -> None:
        import torch

        from ldm_project.config import DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _subset_distill_payload_for_training

        payload = {
            "codes": torch.randn(3, 2),
            "class_labels": torch.tensor([0, 1, 0]),
            "teacher_camera": torch.randn(3, 1, 4, 4),
        }
        for invalid_size in (0, 4):
            train_cfg = TrainingConfig(
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2),
                distillation=DistillationConfig(train_subset_size=invalid_size),
            )
            with self.assertRaisesRegex(ValueError, "distillation.train_subset_size"):
                _subset_distill_payload_for_training(torch, payload, train_cfg)

    def test_paired_class_distill_loader_enables_diversity_pairs(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders
        from ldm_project.training import one_shot_loss

        payload = {
            "codes": torch.randn(4, 3),
            "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "teacher_camera": torch.tensor(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]],
                    [[[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 0.0]]],
                ],
                dtype=torch.float32,
            ),
        }
        train_cfg = TrainingConfig(
            batch_size=1,
            data=DataConfig(validation_fraction=0.0, subset_seed=123),
            first_stage=FirstStageConfig(image_size=2),
            one_shot=OneShotConfig(code_dim=3, num_classes=2),
            distillation=DistillationConfig(paired_class_batches=True),
        )

        train_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
        _, labels, targets = next(iter(train_loader))

        self.assertGreaterEqual(int(labels.shape[0]), 2)
        self.assertLess(int(labels.unique().numel()), int(labels.numel()))

        losses = one_shot_loss(
            {
                "camera_plane": torch.zeros_like(targets),
                "slm_amplitude": torch.ones_like(targets),
            },
            targets,
            class_labels=labels,
            code_diversity_weight=1.0,
            teacher_pairwise_diversity_weight=1.0,
            diversity_min_teacher_mse=1.0e-8,
        )
        self.assertGreater(float(losses["diversity_pair_count"].item()), 0.0)
        self.assertGreater(float(losses["code_diversity_loss"].item()), 0.0)
        self.assertGreater(float(losses["teacher_pairwise_mse"].item()), 0.0)

    def test_multiclass_distill_loader_enables_cross_class_pairs(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders
        from ldm_project.training import one_shot_loss

        labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        target = torch.zeros(6, 1, 2, 2, dtype=torch.float32)
        for index in range(6):
            target[index, 0, index // 2 % 2, index % 2] = 1.0
        payload = {
            "codes": torch.randn(6, 3),
            "class_labels": labels,
            "teacher_camera": target,
        }
        train_cfg = TrainingConfig(
            batch_size=3,
            data=DataConfig(validation_fraction=0.0, subset_seed=321),
            first_stage=FirstStageConfig(image_size=2),
            one_shot=OneShotConfig(code_dim=3, num_classes=3),
            distillation=DistillationConfig(multiclass_batches=True),
        )

        train_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
        first_batch = next(iter(train_loader))
        second_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
        second_first_batch = next(iter(second_loader))
        _, batch_labels, batch_targets = first_batch

        self.assertEqual(batch_labels.tolist(), second_first_batch[1].tolist())
        self.assertGreaterEqual(int(batch_labels.unique().numel()), 2)

        collapsed = torch.full_like(batch_targets, 0.5, requires_grad=True)
        losses = one_shot_loss(
            {
                "camera_plane": collapsed,
                "slm_amplitude": torch.rand_like(batch_targets),
            },
            batch_targets,
            class_labels=batch_labels,
            cross_class_pairwise_weight=0.1,
            diversity_min_teacher_mse=1.0e-8,
        )

        self.assertGreater(float(losses["cross_class_pair_count"].item()), 0.0)
        self.assertGreater(float(losses["cross_class_pairwise_loss"].item()), 0.0)
        self.assertGreater(float(losses["cross_class_teacher_pairwise_mse"].item()), 0.0)
        self.assertEqual(float(losses["cross_class_student_pairwise_mse"].item()), 0.0)
        self.assertEqual(float(losses["cross_class_diversity_ratio"].item()), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(collapsed.grad)

    def test_one_shot_distill_warm_start_latest_and_crop_history(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_training import run_one_shot_distill_training
        from ldm_project.one_shot_training_utils import _one_shot_d2nn_crop_retention_metrics
        from ldm_project.stage_constants import ONE_SHOT_DISTILL_LATEST_CHECKPOINT
        from ldm_project.training import one_shot_loss

        class DummyOneShot(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, codes, labels):
                outputs = _dummy_one_shot_outputs(batch_size=int(codes.shape[0]), image_size=4, grid_size=2)
                for key in (
                    "slm_phase",
                    "encoder_input_phase",
                    "encoder_input_grid_phase",
                    "encoded_phase",
                    "ols_input_phase",
                    "ols_output_phase",
                    "ols_phase",
                    "decoder_input_phase",
                    "decoder_output_phase",
                    "chip_phase",
                    "camera_phase",
                    "diffractive_encoder_layer_amplitude",
                    "diffractive_encoder_layer_phase",
                    "diffractive_decoder_layer_amplitude",
                    "diffractive_decoder_layer_phase",
                    "diffractive_encoder_mask_amplitude",
                    "diffractive_encoder_mask_phase",
                    "diffractive_decoder_mask_amplitude",
                    "diffractive_decoder_mask_phase",
                ):
                    outputs.pop(key, None)
                outputs["camera_plane"] = torch.sigmoid(self.scale) * outputs["camera_plane"]
                return outputs

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.events = []
                self.history = {}
                self.meta_payload = {}
                self.checkpoints = []

            def save_config_snapshot(self, name, data):
                return self.configs_dir / f"{name}.json"

            def append_history(self, name, payload):
                self.history.setdefault(name, []).append(dict(payload))

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
                self.events.append(("grid", Path(destination).name))
                return Path(destination)

            def save_artifact_json(self, name, payload):
                self.events.append(("artifact", name))
                return self.artifacts_dir / f"{name}.json"

            def save_checkpoint(self, name, payload):
                path = self.run_dir / "checkpoints" / name
                record = (name, dict(payload), path)
                self.events.append(("checkpoint", name))
                self.checkpoints.append(record)
                return path

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            distill_path = run_dir / "distill.pt"
            warm_start_path = run_dir / "one_shot.pt"
            labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
            torch.save(
                {
                    "codes": torch.randn(4, 3),
                    "class_labels": labels,
                    "teacher_camera": torch.rand(4, 1, 4, 4),
                },
                distill_path,
            )
            warm_start_payload = {"model_state_dict": {"sentinel": torch.tensor(1.0)}}
            torch.save(warm_start_payload, warm_start_path)
            train_cfg = TrainingConfig(
                batch_size=2,
                step_log_interval=1,
                visual_save_epoch_interval=5,
                validation_epoch_interval=1,
                one_shot_epochs=2,
                data=DataConfig(validation_fraction=0.5, validation_split_seed=11),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(),
            )
            run_manager = RecordingRunManager(run_dir)
            loaded_payloads = []

            def load_checkpoint(model, payload):
                loaded_payloads.append(payload)

            run_one_shot_distill_training(
                torch,
                one_shot_model=DummyOneShot(),
                one_shot_loss_fn=one_shot_loss,
                run_manager=run_manager,
                train_cfg=train_cfg,
                run_meta={},
                physics={"unit": "test"},
                device=torch.device("cpu"),
                use_pin_memory=False,
                one_shot_model_version="unit-test",
                distill_dataset=str(distill_path),
                one_shot_checkpoint=str(warm_start_path),
                load_one_shot_checkpoint_fn=load_checkpoint,
            )

            self.assertEqual(len(loaded_payloads), 1)
            self.assertEqual(loaded_payloads[0]["model_state_dict"]["sentinel"].item(), 1.0)
            latest_checkpoints = [
                item for item in run_manager.checkpoints if item[0] == ONE_SHOT_DISTILL_LATEST_CHECKPOINT
            ]
            self.assertEqual(len(latest_checkpoints), 2)
            latest_payload = latest_checkpoints[-1][1]
            self.assertEqual(latest_payload["epoch"], 2)
            self.assertEqual(latest_payload["stage_step"], 2)
            self.assertEqual(latest_payload["total_steps"], 2)
            self.assertEqual(latest_payload["distill_dataset"], str(distill_path))
            self.assertEqual(run_manager.meta_payload["one_shot_distill_warm_start_checkpoint"], str(warm_start_path))
            self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_checkpoint"], latest_checkpoints[-1][2])
            self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_epoch"], 2)
            self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_step"], 2)
            first_checkpoint_event = next(index for index, event in enumerate(run_manager.events) if event[0] == "checkpoint")
            first_grid_event = next(index for index, event in enumerate(run_manager.events) if event[0] == "grid")
            self.assertLess(first_checkpoint_event, first_grid_event)
            train_history = run_manager.history["one_shot_distill"]
            validation_history = run_manager.history["one_shot_distill_validation"]
            for payload in (train_history[0], validation_history[0]):
                self.assertAlmostEqual(payload["crop_energy_retention_mean"], 0.25, places=6)
                self.assertAlmostEqual(payload["encoder_crop_energy_retention_mean"], 0.25, places=6)
                self.assertAlmostEqual(payload["decoder_crop_energy_retention_mean"], 0.25, places=6)
                self.assertAlmostEqual(payload["encoder_l1_crop_energy_retention"], 0.25, places=6)
                self.assertAlmostEqual(payload["decoder_l2_crop_energy_retention"], 0.25, places=6)
                self.assertNotIn("encoder_crop_input_energy", payload)
                self.assertNotIn("decoder_crop_cropped_energy", payload)
            self.assertEqual(_one_shot_d2nn_crop_retention_metrics({"camera_plane": torch.ones(1, 1, 2, 2)}), {})

        with self.assertRaisesRegex(ValueError, "warm-start"):
            run_one_shot_distill_training(
                torch,
                one_shot_model=DummyOneShot(),
                one_shot_loss_fn=one_shot_loss,
                run_manager=RecordingRunManager(Path(".")),
                train_cfg=train_cfg,
                run_meta={},
                physics={},
                device=torch.device("cpu"),
                use_pin_memory=False,
                one_shot_model_version="unit-test",
                distill_dataset=str(distill_path),
                one_shot_checkpoint=str(warm_start_path),
                load_one_shot_checkpoint_fn=None,
            )

    def test_one_shot_distill_quality_selection_score(self) -> None:
        from ldm_project.one_shot_quality import (
            _one_shot_distill_balanced_quality_score,
            _one_shot_distill_quality_score,
            _one_shot_distill_strict_balanced_quality_score,
            _resolve_one_shot_distill_selection,
        )
        from ldm_project.stage_constants import (
            ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC,
            ONE_SHOT_DISTILL_BEST_METRIC,
            ONE_SHOT_DISTILL_QUALITY_METRIC,
            ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC,
        )

        one_shot = type("OneShotSelectionConfig", (), {"checkpoint_selection_metric": "quality"})()
        train_cfg = type("TrainingSelectionConfig", (), {"one_shot": one_shot})()
        metrics = {
            "loss": 0.5,
            "camera_recon_loss": 0.1,
            "camera_foreground_dice": 0.8,
            "camera_saturation_ratio": 0.4,
            "foreground_hole_ratio": 0.25,
            "slm_tv": 0.2,
        }

        score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

        self.assertEqual(metric, ONE_SHOT_DISTILL_QUALITY_METRIC)
        self.assertAlmostEqual(score, 0.191, places=6)
        self.assertAlmostEqual(metrics["quality_score"], 0.191, places=6)
        self.assertAlmostEqual(_one_shot_distill_quality_score(metrics), 0.191, places=6)

        one_shot.checkpoint_selection_metric = "loss"
        score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

        self.assertEqual(metric, ONE_SHOT_DISTILL_BEST_METRIC)
        self.assertAlmostEqual(score, 0.5, places=6)

        one_shot.checkpoint_selection_metric = "balanced_quality"
        score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

        self.assertEqual(metric, ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC)
        self.assertAlmostEqual(score, 0.1575, places=6)
        self.assertAlmostEqual(metrics["balanced_quality_score"], 0.1575, places=6)
        self.assertAlmostEqual(_one_shot_distill_balanced_quality_score(metrics), 0.1575, places=6)

        one_shot.checkpoint_selection_metric = "strict_balanced_quality"
        score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

        self.assertEqual(metric, ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC)
        self.assertAlmostEqual(score, 0.1695, places=6)
        self.assertAlmostEqual(metrics["strict_balanced_quality_score"], 0.1695, places=6)
        self.assertAlmostEqual(_one_shot_distill_strict_balanced_quality_score(metrics), 0.1695, places=6)

    def test_one_shot_forward_exposes_encoder_diagnostics_on_cpu(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        config = OneShotConfig(
            code_dim=4,
            label_embed_dim=4,
            hidden_dim=8,
            num_classes=2,
            metasurface_grid_size=4,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=8)
        outputs = model(torch.randn(2, config.code_dim), torch.tensor([0, 1]))

        full_size_keys = (
            "encoder_input_amplitude",
            "encoder_input_phase",
            "chip_amplitude",
            "chip_phase",
        )
        grid_size_keys = (
            "encoder_input_grid_amplitude",
            "encoder_input_grid_phase",
            "encoded_amplitude",
            "encoded_phase",
            "ols_input_amplitude",
            "ols_input_phase",
            "ols_output_amplitude",
            "ols_output_phase",
            "ols_amplitude",
            "ols_phase",
            "decoder_input_amplitude",
            "decoder_input_phase",
            "decoder_output_amplitude",
            "decoder_output_phase",
        )
        for key in full_size_keys:
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 8, 8))
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
        for key in grid_size_keys:
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 4, 4))
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)

    def test_teacher_distill_filter_preserves_bundle_and_balances_classes(self) -> None:
        import torch

        from ldm_project.config import (
            DataConfig,
            DiffusionConfig,
            DistillationConfig,
            FirstStageConfig,
            OneShotConfig,
            TrainingConfig,
        )
        from ldm_project.teacher_distill import run_teacher_distill_filter

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.meta_payload = None
                self.metrics_payload = None
                self.bundle_path = None

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

            def save_config_snapshot(self, name, data):
                path = self.configs_dir / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            def save_tensor_bundle(self, name, payload):
                path = self.artifacts_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(payload), path)
                self.bundle_path = path
                return path

            def save_artifact_json(self, name, payload):
                self.metrics_payload = payload
                path = self.artifacts_dir / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4):
                path = Path(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            source_path = run_dir / "source_distill.pt"
            torch.manual_seed(7)
            labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
            teacher_camera = torch.rand(20, 1, 8, 8, dtype=torch.float32) * 0.8 + 0.1
            payload = {
                "codes": torch.randn(20, 4),
                "class_labels": labels,
                "teacher_camera": teacher_camera,
                "teacher_image": teacher_camera.clone(),
                "teacher_latents": torch.randn(20, 1, 1, 1),
                "teacher_initial_noise": torch.randn(20, 1, 1, 1),
                "guidance_scale": torch.tensor(2.0),
                "latent_scale": torch.tensor(1.0),
                "config": {"source": "unit-test"},
            }
            torch.save(payload, source_path)
            train_cfg = TrainingConfig(
                data=DataConfig(validation_fraction=0.1),
                first_stage=FirstStageConfig(image_size=8, latent_channels=1),
                diffusion=DiffusionConfig(num_classes=2),
                one_shot=OneShotConfig(code_dim=4, num_classes=2),
                distillation=DistillationConfig(
                    samples_per_class=10,
                    class_labels=(0, 1),
                    filter_keep_per_class=4,
                    filter_foreground_threshold=0.2,
                ),
            )
            run_manager = RecordingRunManager(run_dir)
            run_meta = {}

            run_teacher_distill_filter(
                torch,
                train_cfg=train_cfg,
                run_manager=run_manager,
                run_meta=run_meta,
                distill_dataset=str(source_path),
            )

            filtered = torch.load(run_manager.bundle_path, map_location="cpu", weights_only=False)
            self.assertEqual(tuple(filtered["codes"].shape), (8, 4))
            self.assertEqual(tuple(filtered["teacher_camera"].shape), (8, 1, 8, 8))
            self.assertEqual(tuple(filtered["teacher_initial_noise"].shape), (8, 1, 1, 1))
            self.assertIn("filter", filtered["config"])
            self.assertEqual((filtered["class_labels"] == 0).sum().item(), 4)
            self.assertEqual((filtered["class_labels"] == 1).sum().item(), 4)
            self.assertEqual(run_manager.metrics_payload["class_counts"], {"0": 4, "1": 4})
            self.assertEqual(run_manager.meta_payload["teacher_distill_filter_sample_count"], 8)

    def test_one_shot_visualization_includes_raw_and_display_camera(self) -> None:
        import torch

        from ldm_project.d2nn_diagnostics import build_one_shot_d2nn_diagnostics
        from ldm_project.d2nn_visualization import (
            build_one_shot_d2nn_focus_visualization_tensors,
            build_one_shot_d2nn_visualization_tensors,
            build_one_shot_slm_to_encoder_focus_visualization_tensors,
        )
        from ldm_project.one_shot_visualization import (
            build_one_shot_display_diagnostic_tensors,
            build_one_shot_shared_scale_visualization_tensors,
            build_one_shot_visual_diagnostics,
            build_one_shot_visualization_tensors,
        )

        reference = torch.rand(2, 1, 16, 16)
        camera = torch.rand(2, 1, 16, 16)
        slm_amplitude = torch.rand(2, 1, 16, 16)
        codes = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
        chip_amplitude = torch.cat(
            [
                torch.linspace(0.0, 2.0, steps=256).reshape(1, 1, 16, 16),
                torch.linspace(0.0, 200.0, steps=256).reshape(1, 1, 16, 16),
            ],
            dim=0,
        )
        outputs = _dummy_one_shot_outputs(batch_size=2, image_size=16, grid_size=8)
        outputs["camera_plane"] = camera
        outputs["slm_command"] = slm_amplitude
        outputs["slm_amplitude"] = slm_amplitude
        outputs["chip_amplitude"] = chip_amplitude
        outputs["chip_phase"] = torch.zeros(2, 1, 16, 16)

        tensors = build_one_shot_visualization_tensors(reference, outputs, codes=codes)
        diagnostics = build_one_shot_display_diagnostic_tensors(outputs)

        self.assertEqual(
            list(tensors.keys()),
            ["amplitude", "phase"],
        )
        self.assertEqual(
            list(tensors["amplitude"].keys()),
            [
                "noise code",
                "reference",
                "camera",
                "slm command",
                "slm amplitude grid",
                "slm out amp",
                "encoder in amp",
                "encoder grid in amp",
                "encoder mask amp = 1 L1",
                "encoder field after L1",
                "encoder mask amp = 1 L2",
                "encoder field after L2",
                "encoder out amp",
                "ols in amp",
                "ols out amp",
                "decoder in amp",
                "decoder mask amp = 1 L1",
                "decoder field after L1",
                "decoder mask amp = 1 L2",
                "decoder field after L2",
                "decoder out amp",
                "chip amp",
                "chip I",
            ],
        )
        self.assertEqual(
            list(tensors["phase"].keys()),
            [
                "noise code",
                "reference",
                "camera",
                "slm command",
                "slm amplitude grid",
                "slm out amp",
                "encoder in amp",
                "encoder grid in amp",
                "encoder mask phase L1",
                "encoder field after L1",
                "encoder mask phase L2",
                "encoder field after L2",
                "encoder out amp",
                "ols in amp",
                "ols out amp",
                "decoder in amp",
                "decoder mask phase L1",
                "decoder field after L1",
                "decoder mask phase L2",
                "decoder field after L2",
                "decoder out amp",
                "chip amp",
                "chip I",
            ],
        )
        self.assertNotIn("chip_amplitude_display", diagnostics)
        for key in ("slm command", "slm out amp"):
            normalized = tensors["amplitude"][key]
            self.assertTrue(torch.all(normalized >= 0.0).item(), key)
            self.assertTrue(torch.all(normalized <= 1.0).item(), key)
            for sample_idx in range(normalized.shape[0]):
                sample = normalized[sample_idx]
                self.assertAlmostEqual(sample.amin().item(), 0.0, delta=1e-5, msg=f"{key}[{sample_idx}] min")
                self.assertAlmostEqual(sample.amax().item(), 1.0, delta=1e-5, msg=f"{key}[{sample_idx}] max")
        self.assertTrue(torch.all(tensors["amplitude"]["camera"] >= 0.0).item())
        self.assertTrue(torch.all(tensors["amplitude"]["camera"] <= 1.0).item())
        for key in (
            "slm_phase_display",
            "encoder_input_amplitude_display",
            "encoder_input_phase_display",
            "encoder_input_grid_amplitude_display",
            "encoder_input_grid_phase_display",
            "encoded_amplitude_display",
            "encoded_phase_display",
            "ols_input_amplitude_display",
            "ols_input_phase_display",
            "ols_output_amplitude_display",
            "ols_output_phase_display",
            "ols_amplitude_display",
            "ols_phase_display",
            "decoder_input_amplitude_display",
            "decoder_input_phase_display",
            "decoder_output_amplitude_display",
            "decoder_output_phase_display",
            "chip_field_amplitude_display",
            "chip_intensity_display",
            "chip_phase_display",
        ):
            self.assertTrue(torch.all(diagnostics[key] >= 0.0).item(), key)
            self.assertTrue(torch.all(diagnostics[key] <= 1.0).item(), key)
        flat_amp = chip_amplitude.flatten(start_dim=1)
        amp_lo = flat_amp.quantile(0.02, dim=1).reshape(2, 1, 1, 1)
        amp_hi = flat_amp.quantile(0.98, dim=1).reshape(2, 1, 1, 1)
        expected_amplitude = ((chip_amplitude - amp_lo) / (amp_hi - amp_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        chip_intensity = chip_amplitude.pow(2)
        flat_int = chip_intensity.flatten(start_dim=1)
        int_lo = flat_int.quantile(0.02, dim=1).reshape(2, 1, 1, 1)
        int_hi = flat_int.quantile(0.98, dim=1).reshape(2, 1, 1, 1)
        expected_intensity = ((chip_intensity - int_lo) / (int_hi - int_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(diagnostics["chip_field_amplitude_display"], expected_amplitude))
        self.assertTrue(torch.allclose(diagnostics["chip_intensity_display"], expected_intensity))
        visual_diagnostics = build_one_shot_visual_diagnostics(outputs, diagnostics)
        self.assertEqual(visual_diagnostics["view"], "compact_full_chain")
        self.assertEqual(
            visual_diagnostics["alias_diffs"]["encoded_to_ols_input_amplitude"]["max_abs_diff"],
            0.0,
        )
        self.assertIn("encoder_out_ols_in", visual_diagnostics["amplitude_phase_similarity"])
        self.assertIn("pearson", visual_diagnostics["amplitude_phase_similarity"]["chip"])
        self.assertEqual(visual_diagnostics["tensor_stats"]["chip_amplitude"]["shape"], [2, 1, 16, 16])

        shared = build_one_shot_shared_scale_visualization_tensors(reference, outputs, codes=codes)
        shared_amplitude = shared["amplitude"]
        self.assertEqual(
            list(shared_amplitude.keys()),
            [
                "noise code",
                "reference",
                "slm in cmd",
                "slm in cmd contrast",
                "slm amplitude grid",
                "slm out amp",
                "encoder in amp",
                "encoder grid in amp",
                "encoder mask amp = 1 L1",
                "encoder field after L1",
                "encoder mask amp = 1 L2",
                "encoder field after L2",
                "encoder out amp",
                "ols in amp",
                "ols out amp",
                "decoder in amp",
                "decoder mask amp = 1 L1",
                "decoder field after L1",
                "decoder mask amp = 1 L2",
                "decoder field after L2",
                "decoder out amp",
                "chip amp",
                "chip I",
                "camera",
            ],
        )
        self.assertTrue(torch.allclose(shared_amplitude["slm in cmd"], slm_amplitude))
        self.assertTrue(torch.all(shared_amplitude["slm in cmd contrast"] >= 0.0).item())
        self.assertTrue(torch.all(shared_amplitude["slm in cmd contrast"] <= 1.0).item())
        shared_scale = shared["diagnostics"]["amplitude_scale"]
        self.assertAlmostEqual(float(shared_amplitude["chip amp"].max().item()), 1.0, places=6)
        self.assertTrue(torch.allclose(shared_amplitude["slm out amp"], self._minmax_norm(slm_amplitude)))
        self.assertTrue(torch.allclose(shared_amplitude["encoder in amp"], self._minmax_norm(outputs["encoder_input_amplitude"])))
        self.assertTrue(
            torch.allclose(shared_amplitude["encoder grid in amp"], self._minmax_norm(outputs["encoder_input_grid_amplitude"]))
        )
        self.assertTrue(torch.allclose(shared_amplitude["chip amp"], (chip_amplitude / shared_scale).clamp(0.0, 1.0)))
        self.assertEqual(list(shared["phase"].keys())[0], "noise code")
        self.assertEqual(shared["diagnostics"]["view"], "shared_scale_full_chain")
        self.assertNotIn("slm out amp", shared["diagnostics"]["amplitude_scale_columns"])
        self.assertIn("slm out amp", shared["diagnostics"]["early_raw_amplitude_columns"])
        self.assertIn("encoder grid in amp", shared["diagnostics"]["early_raw_amplitude_columns"])
        self.assertIn("p99_minus_p01_mean", shared["diagnostics"]["slm_command_stats"])
        self.assertIn("encoder grid in phase", shared["diagnostics"]["phase_stats"])

        d2nn = build_one_shot_d2nn_visualization_tensors(outputs)
        self.assertEqual(
            list(d2nn["layers"]["amplitude"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        self.assertEqual(
            list(d2nn["layers"]["phase"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        self.assertEqual(
            list(d2nn["masks"]["amplitude"].keys()),
            ["encoder mask amp = 1 L1", "encoder mask amp = 1 L2", "decoder mask amp = 1 L1", "decoder mask amp = 1 L2"],
        )
        self.assertEqual(
            list(d2nn["masks"]["phase"].keys()),
            ["encoder mask phase L1", "encoder mask phase L2", "decoder mask phase L1", "decoder mask phase L2"],
        )
        self.assertEqual(tuple(d2nn["layers"]["amplitude"]["encoder field after L1"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(d2nn["layers"]["phase"]["decoder field after L2"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(d2nn["masks"]["amplitude"]["encoder mask amp = 1 L1"].shape), (1, 1, 8, 8))
        self.assertEqual(tuple(tensors["amplitude"]["encoder mask amp = 1 L1"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(tensors["phase"]["decoder mask phase L2"].shape), (2, 1, 8, 8))
        focus = build_one_shot_d2nn_focus_visualization_tensors(outputs)
        self.assertEqual(list(focus["amplitude"].keys()), ["encoder field after L1 amplitude"])
        self.assertEqual(list(focus["phase"].keys()), ["encoder field after L1 phase"])
        self.assertEqual(tuple(focus["amplitude"]["encoder field after L1 amplitude"].shape), (1, 1, 8, 8))
        self.assertEqual(tuple(focus["phase"]["encoder field after L1 phase"].shape), (1, 1, 8, 8))
        slm_focus = build_one_shot_slm_to_encoder_focus_visualization_tensors(outputs)
        self.assertEqual(
            list(slm_focus["amplitude"].keys()),
            [
                "SLM plane before propagation amplitude",
                "encoder input after propagation amplitude",
                "encoder grid after propagation amplitude",
            ],
        )
        self.assertEqual(
            list(slm_focus["phase"].keys()),
            [
                "SLM plane before propagation phase",
                "encoder input after propagation phase",
                "encoder grid after propagation phase",
            ],
        )
        self.assertEqual(tuple(slm_focus["amplitude"]["SLM plane before propagation amplitude"].shape), (1, 1, 16, 16))
        self.assertEqual(
            tuple(slm_focus["amplitude"]["encoder input after propagation amplitude"].shape),
            (1, 1, 16, 16),
        )
        self.assertEqual(
            tuple(slm_focus["amplitude"]["encoder grid after propagation amplitude"].shape),
            (1, 1, 8, 8),
        )
        self.assertEqual(tuple(slm_focus["phase"]["encoder grid after propagation phase"].shape), (1, 1, 8, 8))
        self.assertTrue(torch.all(d2nn["masks"]["amplitude"]["encoder mask amp = 1 L1"] >= 0.0).item())
        self.assertTrue(torch.allclose(d2nn["masks"]["amplitude"]["encoder mask amp = 1 L1"], torch.ones(1, 1, 8, 8)))
        self.assertTrue(torch.all(d2nn["masks"]["phase"]["decoder mask phase L2"] <= 1.0).item())
        dense_only = {key: value for key, value in outputs.items() if not key.startswith("diffractive_")}
        dense_d2nn = build_one_shot_d2nn_visualization_tensors(dense_only)
        self.assertFalse(dense_d2nn["layers"]["amplitude"])
        self.assertFalse(dense_d2nn["layers"]["phase"])
        self.assertFalse(dense_d2nn["masks"]["amplitude"])
        self.assertFalse(dense_d2nn["masks"]["phase"])
        d2nn_diagnostics = build_one_shot_d2nn_diagnostics(outputs)
        self.assertTrue(d2nn_diagnostics["available"])
        self.assertEqual(d2nn_diagnostics["view"], "d2nn_phase_only_feasibility")
        self.assertTrue(
            d2nn_diagnostics["phase_only_mask_checks"]["diffractive_encoder_mask_amplitude"][
                "allclose_to_one_atol_1e-6"
            ]
        )
        self.assertEqual(
            d2nn_diagnostics["identity_checks"]["ols_input_to_output_amplitude"]["max_abs_diff"],
            1.0,
        )
        self.assertIn("p99_to_mean_ratio_mean", d2nn_diagnostics["layers"]["amplitude"]["encoder field after L1"])
        self.assertIn("finite_ratio", d2nn_diagnostics["masks"]["phase"]["decoder mask phase L2"])
        self.assertIn("local_contrast_mean", d2nn_diagnostics["masks"]["phase"]["decoder mask phase L2"])
        self.assertEqual(
            list(d2nn_diagnostics["energy_retention"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        self.assertEqual(
            list(d2nn_diagnostics["crop_energy_retention"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        for energy_stats in d2nn_diagnostics["energy_retention"].values():
            self.assertIsNotNone(energy_stats["ratio_mean"])
            self.assertGreaterEqual(energy_stats["ratio_mean"], 0.0)
            self.assertGreaterEqual(energy_stats["ratio_min"], 0.0)
            self.assertGreaterEqual(energy_stats["ratio_max"], 0.0)
        for crop_stats in d2nn_diagnostics["crop_energy_retention"].values():
            self.assertIsNotNone(crop_stats["ratio_mean"])
            self.assertGreaterEqual(crop_stats["ratio_mean"], 0.0)
            self.assertIn("propagation_energy_ratio_mean", crop_stats)
            self.assertIn("propagated_full_energy_mean", crop_stats)
        self.assertEqual(
            list(d2nn_diagnostics["layers"]["amplitude"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        self.assertEqual(
            list(d2nn_diagnostics["masks"]["amplitude"].keys()),
            ["encoder mask amp = 1 L1", "encoder mask amp = 1 L2", "decoder mask amp = 1 L1", "decoder mask amp = 1 L2"],
        )
        dense_diagnostics = build_one_shot_d2nn_diagnostics(dense_only)
        self.assertFalse(dense_diagnostics["available"])
        self.assertEqual(dense_diagnostics["energy_retention"], {})
        self.assertEqual(dense_diagnostics["crop_energy_retention"], {})
        self.assertFalse(build_one_shot_d2nn_focus_visualization_tensors(dense_only)["amplitude"])

    def test_phase_visualization_wraps_to_zero_to_two_pi(self) -> None:
        import torch

        from ldm_project.one_shot_visualization import build_one_shot_display_diagnostic_tensors

        phase = torch.tensor([[-torch.pi, 0.0], [torch.pi, 2.0 * torch.pi]], dtype=torch.float32).reshape(1, 1, 2, 2)
        outputs = _dummy_one_shot_outputs(batch_size=1, image_size=2, grid_size=2)
        for key in (
            "slm_phase",
            "encoder_input_phase",
            "encoder_input_grid_phase",
            "encoded_phase",
            "ols_input_phase",
            "ols_output_phase",
            "ols_phase",
            "decoder_input_phase",
            "decoder_output_phase",
            "chip_phase",
        ):
            outputs[key] = phase

        diagnostics = build_one_shot_display_diagnostic_tensors(outputs)

        expected = torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float32).reshape(1, 1, 2, 2)
        self.assertTrue(torch.allclose(diagnostics["slm_phase_display"], expected))
        self.assertTrue(torch.allclose(diagnostics["chip_phase_display"], expected))
        self.assertTrue(torch.allclose(diagnostics["encoder_input_phase_display"], expected))

    def test_one_shot_d2nn_focus_helper_skips_missing_d2nn_tensors(self) -> None:
        import struct

        if importlib.util.find_spec("matplotlib") is None:
            raise unittest.SkipTest("matplotlib is required for D2NN focus PNG output.")

        from ldm_project.one_shot_eval_artifacts import (
            _save_one_shot_d2nn_focus_visualization,
            _save_one_shot_slm_to_encoder_focus_visualization,
        )

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"

        outputs = _dummy_one_shot_outputs(batch_size=2, image_size=8, grid_size=4)
        dense_only = {key: value for key, value in outputs.items() if not key.startswith("diffractive_")}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_manager = RecordingRunManager(Path(tmpdir))
            skipped = _save_one_shot_d2nn_focus_visualization(
                run_manager,
                dense_only,
                run_manager.images_dir / "one_shot_distill" / "d2nn_focus" / "epoch_0001_encoder_L1_focus.png",
                title_label="one_shot_distill epoch 1 encoder L1 focus",
                row_label="label 0",
            )
            self.assertEqual(skipped, {})

            saved = _save_one_shot_d2nn_focus_visualization(
                run_manager,
                outputs,
                run_manager.images_dir / "one_shot_distill" / "d2nn_focus" / "epoch_0001_encoder_L1_focus.png",
                title_label="one_shot_distill epoch 1 encoder L1 focus",
                row_label="label 0",
            )
            self.assertEqual(set(saved.keys()), {"amplitude", "phase"})
            self.assertEqual(saved["amplitude"].name, "epoch_0001_encoder_L1_focus_amplitude.png")
            self.assertEqual(saved["phase"].name, "epoch_0001_encoder_L1_focus_phase.png")
            self.assertEqual(saved["amplitude"].parent.name, "d2nn_focus")
            for path in saved.values():
                self.assertTrue(path.exists())
                with path.open("rb") as handle:
                    header = handle.read(24)
                self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
                width, height = struct.unpack(">II", header[16:24])
                self.assertGreaterEqual(width, 800)
                self.assertGreaterEqual(height, 800)

            slm_focus_saved = _save_one_shot_slm_to_encoder_focus_visualization(
                run_manager,
                outputs,
                run_manager.images_dir / "one_shot_distill" / "d2nn_focus" / "epoch_0001_slm_to_encoder_focus.png",
                title_label="one_shot_distill epoch 1 SLM to encoder focus",
                row_label="label 0",
            )
            self.assertEqual(set(slm_focus_saved.keys()), {"amplitude", "phase"})
            for path in slm_focus_saved.values():
                self.assertTrue(path.exists())
                with path.open("rb") as handle:
                    header = handle.read(24)
                width, height = struct.unpack(">II", header[16:24])
                self.assertGreaterEqual(width, 2400)
                self.assertGreaterEqual(height, 800)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            relative_run_dir = Path(tmpdir).relative_to(Path.cwd())
            run_manager = RecordingRunManager(relative_run_dir)
            destination = run_manager.images_dir / "one_shot_distill" / "d2nn_focus" / "epoch_0001_encoder_L1_focus.png"
            saved = _save_one_shot_d2nn_focus_visualization(
                run_manager,
                outputs,
                destination,
                title_label="one_shot_distill epoch 1 encoder L1 focus",
                row_label="label 0",
            )
            self.assertEqual(saved["amplitude"], destination.with_name("epoch_0001_encoder_L1_focus_amplitude.png"))
            self.assertTrue(saved["amplitude"].exists())
            nested = relative_run_dir / relative_run_dir / "images"
            self.assertFalse(nested.exists())

    def test_run_manager_saves_compact_tensor_grid_png(self) -> None:
        import importlib.util
        import struct

        import torch

        if importlib.util.find_spec("matplotlib") is None:
            raise unittest.SkipTest("matplotlib is required for PNG grid output.")

        from ldm_project.run_manager import RunManager

        with tempfile.TemporaryDirectory() as tmpdir:
            run_manager = RunManager(Path(tmpdir))
            destination = run_manager.images_dir / "compact_grid.png"

            path = run_manager.save_tensor_grid(
                {
                    "left": torch.rand(2, 1, 8, 8),
                    "right": torch.rand(2, 1, 8, 8),
                },
                destination,
                max_items=2,
                row_labels=["0", "1"],
                phase_colorbar=True,
                cmap="twilight",
            )

            self.assertEqual(path, destination)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            with path.open("rb") as handle:
                header = handle.read(24)
            self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", header[16:24])
            self.assertLessEqual(width, 420)
            self.assertLessEqual(height, 320)

    def test_one_shot_distill_eval_selection_respects_requested_sample_count(self) -> None:
        import torch

        from ldm_project.config import DataConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _select_distill_eval_indices

        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        train_cfg = TrainingConfig(
            data=DataConfig(validation_fraction=0.0),
            one_shot=OneShotConfig(
                num_classes=2,
                sample_class_labels=(0, 1),
                sample_repeats_per_class=1,
            ),
        )

        selected, summary = _select_distill_eval_indices(torch, labels, train_cfg)

        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["requested_samples_per_class"], 1)
        self.assertEqual(summary["selected_count"], 2)
        self.assertEqual(summary["per_class"]["0"]["selected"], 1)
        self.assertEqual(summary["per_class"]["1"]["selected"], 1)

    def test_one_shot_distill_eval_selection_rejects_non_positive_sample_count(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _select_distill_eval_indices

        train_cfg = TrainingConfig(
            one_shot=OneShotConfig(
                num_classes=2,
                sample_class_labels=(0, 1),
                sample_repeats_per_class=0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "sample_repeats_per_class"):
            _select_distill_eval_indices(torch, torch.tensor([0, 1], dtype=torch.long), train_cfg)

    def test_one_shot_distill_eval_preview_indices_balance_classes(self) -> None:
        import torch

        from ldm_project.sampling_reports import select_balanced_visual_preview_indices

        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.long)

        selected, summary = select_balanced_visual_preview_indices(labels, max_per_class=2)

        self.assertEqual(selected, [0, 1, 3, 4, 6])
        self.assertEqual(summary["visual_preview_count"], 5)
        self.assertEqual(summary["visual_preview_labels"], [0, 0, 1, 1, 2])
        self.assertEqual(summary["per_class"]["0"]["selected"], 2)
        self.assertEqual(summary["per_class"]["1"]["selected"], 2)
        self.assertEqual(summary["per_class"]["2"]["selected"], 1)

    def test_one_shot_distill_eval_diagnostics_writes_focus_preview_without_mutating_source(self) -> None:
        import torch

        from ldm_project import RunManager
        from ldm_project.one_shot_eval import run_one_shot_distill_eval_diagnostics_stage

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_path = root / "one_shot_distill_eval_tensors.pt"
            metrics_path = root / "one_shot_distill_eval_metrics.json"
            labels = torch.tensor([label for label in range(10) for _ in range(3)], dtype=torch.long)
            base = torch.linspace(0.0, 1.0, steps=labels.numel() * 16).reshape(labels.numel(), 1, 4, 4)
            tensor_payload = {
                "class_labels": labels,
                "teacher_camera_display": base,
                "student_camera_display": base.flip(-1),
                "abs_error": (base - base.flip(-1)).abs(),
                "codes_display": torch.linspace(0.0, 1.0, steps=labels.numel() * 4).reshape(labels.numel(), 1, 1, 4),
                "slm_phase_display": base.roll(shifts=1, dims=-2),
                "encoder_input_amplitude_display": base * 0.9,
                "encoder_input_phase_display": base.roll(shifts=2, dims=-1),
                "encoder_input_grid_amplitude_display": base * 0.8,
                "encoder_input_grid_phase_display": base.roll(shifts=3, dims=-1),
                "encoded_amplitude_display": base * 0.7,
                "encoded_phase_display": base.roll(shifts=1, dims=-1),
                "ols_input_amplitude_display": base * 0.6,
                "ols_input_phase_display": base.roll(shifts=2, dims=-2),
                "ols_output_amplitude_display": base * 0.5,
                "ols_output_phase_display": base.roll(shifts=3, dims=-2),
                "decoder_input_amplitude_display": base * 0.4,
                "decoder_input_phase_display": base.flip(-2),
                "decoder_output_amplitude_display": base * 0.3,
                "decoder_output_phase_display": base.flip(-1),
                "chip_field_amplitude_display": base.sqrt(),
                "chip_intensity_display": base.pow(2),
                "chip_phase_display": base.roll(shifts=1, dims=-1),
            }
            torch.save(tensor_payload, bundle_path)
            source_size = bundle_path.stat().st_size
            metrics_path.write_text(
                json.dumps(
                    {
                        "overall": {"count": int(labels.numel()), "paired_dice": 0.88},
                        "per_class": {
                            str(label): {
                                "paired_dice": 0.80 + 0.01 * label,
                                "paired_corr": 0.90 + 0.001 * label,
                                "foreground_hole_ratio": 0.01,
                                "student_saturation_ratio": 0.38,
                                "student_low_saturation_ratio": 0.37,
                                "student_high_saturation_ratio": 0.005,
                                "diversity_ratio": 0.70 + 0.01 * label,
                            }
                            for label in range(10)
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_manager = RunManager(root / "run")

            run_one_shot_distill_eval_diagnostics_stage(
                torch,
                run_manager=run_manager,
                run_meta={},
                eval_tensor_bundle=str(bundle_path),
                eval_metrics=str(metrics_path),
                diagnostic_class_labels=tuple(range(10)),
                diagnostic_max_per_class=20,
            )

            amplitude_path = run_manager.images_dir / "one_shot_distill_eval_focus_0_1_2_3_4_5_6_7_8_9_amplitude.png"
            phase_path = run_manager.images_dir / "one_shot_distill_eval_focus_0_1_2_3_4_5_6_7_8_9_phase.png"
            artifact_path = run_manager.artifacts_dir / "one_shot_distill_eval_focus_diagnostics.json"
            config_path = run_manager.configs_dir / "one_shot_distill_eval_diagnostics_config.json"
            self.assertTrue(amplitude_path.is_file())
            self.assertGreater(amplitude_path.stat().st_size, 0)
            self.assertTrue(phase_path.is_file())
            self.assertGreater(phase_path.stat().st_size, 0)
            self.assertTrue(artifact_path.is_file())
            self.assertTrue(config_path.is_file())
            self.assertEqual(bundle_path.stat().st_size, source_size)
            with artifact_path.open("r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            self.assertEqual(
                diagnostics["selection"]["visual_preview_labels"],
                [label for label in range(10) for _ in range(2)],
            )
            self.assertEqual(diagnostics["selection"]["visual_preview_count"], 20)
            self.assertEqual(diagnostics["classes"]["5"]["preview_indices"], [15, 16])
            self.assertAlmostEqual(diagnostics["classes"]["9"]["paired_dice"], 0.89, places=6)
            self.assertIn("amplitude", diagnostics["images"])
            self.assertIn("phase", diagnostics["images"])
            self.assertEqual(diagnostics["visual_diagnostics"]["view"], "compact_full_chain")
            self.assertIn("encoder_out_ols_in", diagnostics["visual_diagnostics"]["amplitude_phase_similarity"])
            self.assertIn("top25_iou", diagnostics["visual_diagnostics"]["amplitude_phase_similarity"]["chip"])

    def test_one_shot_sample_artifact_preserves_raw_and_display_diagnostics(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.data import DatasetBundle
        from ldm_project.one_shot_eval import run_one_shot_sample_stage

        def diagnostic_outputs(batch_size: int, image_size: int, grid_size: int) -> dict[str, torch.Tensor]:
            return _dummy_one_shot_outputs(batch_size=batch_size, image_size=image_size, grid_size=grid_size)

        class DummyOneShot:
            def eval(self) -> None:
                pass

            def __call__(self, codes, labels):
                return diagnostic_outputs(int(codes.shape[0]), 4, 2)

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.grid_calls = []
                self.bundle_payload = None
                self.config_payloads = {}
                self.artifact_payloads = {}

            def save_config_snapshot(self, name, data):
                self.config_payloads[name] = dict(data)
                return self.configs_dir / f"{name}.json"

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
                self.grid_calls.append((Path(destination).name, dict(tensor_map), max_items, dict(kwargs)))
                return Path(destination)

            def save_tensor_bundle(self, name, payload):
                self.bundle_payload = dict(payload)
                return self.artifacts_dir / name

            def save_artifact_json(self, name, payload):
                self.artifact_payloads[name] = dict(payload)
                return self.artifacts_dir / f"{name}.json"

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint = run_dir / "one_shot.pt"
            torch.save({}, checkpoint)
            images = torch.rand(2, 1, 4, 4)
            labels = torch.tensor([0, 1], dtype=torch.long)
            dataset = TensorDataset(images, labels)
            bundle = DatasetBundle(
                dataset=dataset,
                loader=DataLoader(dataset, batch_size=2, shuffle=False),
                effective_size=2,
                total_size=2,
            )
            train_cfg = TrainingConfig(
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                diffusion=DiffusionConfig(num_classes=2),
                one_shot=OneShotConfig(
                    code_dim=3,
                    num_classes=2,
                    sample_class_labels=(0, 1),
                    sample_repeats_per_class=4,
                ),
            )
            run_manager = RecordingRunManager(run_dir)

            run_one_shot_sample_stage(
                torch,
                one_shot_model=DummyOneShot(),
                train_bundle=bundle,
                validation_bundle=bundle,
                train_cfg=train_cfg,
                run_manager=run_manager,
                run_meta={},
                device=torch.device("cpu"),
                one_shot_checkpoint=str(checkpoint),
                load_one_shot_checkpoint_fn=lambda model, payload: None,
            )

            for key in (
                "codes_display",
                "slm_phase_display",
                "chip_field_amplitude_display",
                "chip_intensity_display",
                "encoder_input_amplitude_display",
                "encoder_input_grid_amplitude_display",
                "encoded_phase_display",
                "ols_input_phase_display",
                "ols_output_phase_display",
                "ols_phase_display",
                "decoder_input_amplitude_display",
                "decoder_output_phase_display",
            ):
                self.assertIn(key, run_manager.bundle_payload)
            amplitude_call = next(call for call in run_manager.grid_calls if call[0] == "one_shot_sample_amplitude.png")
            shared_phase_call = next(
                call for call in run_manager.grid_calls if call[0] == "one_shot_sample_shared_scale_phase.png"
            )
            self.assertEqual(
                list(amplitude_call[1].keys()),
                [
                    "noise code",
                    "reference",
                    "camera",
                    "slm command",
                    "slm amplitude grid",
                    "slm out amp",
                    "encoder in amp",
                    "encoder grid in amp",
                    "encoder mask amp = 1 L1",
                    "encoder field after L1",
                    "encoder mask amp = 1 L2",
                    "encoder field after L2",
                    "encoder out amp",
                    "ols in amp",
                    "ols out amp",
                    "decoder in amp",
                    "decoder mask amp = 1 L1",
                    "decoder field after L1",
                    "decoder mask amp = 1 L2",
                    "decoder field after L2",
                    "decoder out amp",
                    "chip amp",
                    "chip I",
                ],
            )
            self.assertEqual(
                list(shared_phase_call[1].keys()),
                [
                    "noise code",
                    "reference",
                    "slm in cmd",
                    "slm in cmd contrast",
                    "slm amplitude grid",
                    "slm out amp",
                    "encoder in amp",
                    "encoder grid in amp",
                    "encoder mask phase L1",
                    "encoder field after L1",
                    "encoder mask phase L2",
                    "encoder field after L2",
                    "encoder out amp",
                    "ols in amp",
                    "ols out amp",
                    "decoder in amp",
                    "decoder mask phase L1",
                    "decoder field after L1",
                    "decoder mask phase L2",
                    "decoder field after L2",
                    "decoder out amp",
                    "chip amp",
                    "chip I",
                    "camera",
                ],
            )
            self.assertIn("chip amp", amplitude_call[1])
            self.assertIn("chip amp", shared_phase_call[1])
            self.assertIn("encoder field after L1", amplitude_call[1])
            self.assertIn("decoder field after L2", shared_phase_call[1])
            self.assertEqual(tuple(amplitude_call[1]["encoder field after L1"].shape), (8, 1, 2, 2))
            self.assertEqual(tuple(amplitude_call[1]["encoder mask amp = 1 L1"].shape), (8, 1, 2, 2))
            self.assertEqual(tuple(shared_phase_call[1]["decoder mask phase L2"].shape), (8, 1, 2, 2))
            self.assertFalse(any("_d2nn_" in call[0] for call in run_manager.grid_calls))
            self.assertEqual(amplitude_call[2], 8)
            self.assertEqual(amplitude_call[3]["row_labels"], ["0", "0", "0", "0", "1", "1", "1", "1"])
            sample_visual_diagnostics = run_manager.artifact_payloads["one_shot_sample_visual_diagnostics"]
            self.assertEqual(sample_visual_diagnostics["view"], "compact_full_chain")
            self.assertEqual(
                sample_visual_diagnostics["alias_diffs"]["encoded_to_ols_input_amplitude"]["max_abs_diff"],
                0.0,
            )
            self.assertIn("top25_iou", sample_visual_diagnostics["amplitude_phase_similarity"]["chip"])
            sample_d2nn_diagnostics = run_manager.artifact_payloads["one_shot_sample_d2nn_diagnostics"]
            self.assertEqual(sample_d2nn_diagnostics["context"]["stage"], "one_shot_sample")
            self.assertTrue(sample_d2nn_diagnostics["available"])
            self.assertTrue(
                sample_d2nn_diagnostics["phase_only_mask_checks"]["diffractive_decoder_mask_amplitude"][
                    "allclose_to_one_atol_1e-6"
                ]
            )
            self.assertEqual(
                sample_d2nn_diagnostics["identity_checks"]["ols_input_to_output_amplitude"]["max_abs_diff"],
                1.0,
            )
            shared_diagnostics = run_manager.artifact_payloads["one_shot_sample_shared_scale_diagnostics"]
            self.assertEqual(shared_diagnostics["view"], "shared_scale_full_chain")
            self.assertIn("slm out amp", shared_diagnostics["early_raw_amplitude_columns"])
            self.assertIn("encoder grid in amp", shared_diagnostics["early_raw_amplitude_columns"])
            self.assertNotIn("slm out amp", shared_diagnostics["amplitude_scale_columns"])
            self.assertIn("encoder grid in phase", shared_diagnostics["phase_stats"])
            self.assertIn("p99_minus_p01_mean", shared_diagnostics["slm_command_stats"])
            self.assertFalse(any("chip_amplitude_display" in call[1] for call in run_manager.grid_calls))
            self.assertNotIn("chip_amplitude_display", run_manager.bundle_payload)
            for key in (
                "slm_command",
                "chip_amplitude",
                "chip_phase",
                "encoder_input_amplitude",
                "encoder_input_grid_phase",
                "encoded_phase",
                "ols_input_amplitude",
                "ols_output_phase",
                "decoder_input_amplitude",
                "decoder_output_phase",
                "slm_phase",
                "diffractive_encoder_layer_amplitude",
                "diffractive_decoder_mask_phase",
                "diffractive_camera_intensity_raw",
                "diffractive_camera_plane_raw",
                "diffractive_camera_plane_loss_raw",
            ):
                self.assertIn(key, run_manager.bundle_payload)

    def test_one_shot_distill_eval_artifact_preserves_raw_and_display_diagnostics(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_eval import run_one_shot_distill_eval_stage

        def diagnostic_outputs(batch_size: int, image_size: int, grid_size: int) -> dict[str, torch.Tensor]:
            return _dummy_one_shot_outputs(batch_size=batch_size, image_size=image_size, grid_size=grid_size)

        class DummyOneShot:
            def eval(self) -> None:
                pass

            def __call__(self, codes, labels):
                return diagnostic_outputs(int(codes.shape[0]), 4, 2)

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.grid_calls = []
                self.bundle_payload = None
                self.config_payloads = {}
                self.artifact_payloads = {}

            def save_config_snapshot(self, name, data):
                self.config_payloads[name] = dict(data)
                return self.configs_dir / f"{name}.json"

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
                self.grid_calls.append((Path(destination).name, dict(tensor_map), max_items, dict(kwargs)))
                return Path(destination)

            def save_tensor_bundle(self, name, payload):
                self.bundle_payload = dict(payload)
                return self.artifacts_dir / name

            def save_artifact_json(self, name, payload):
                self.artifact_payloads[name] = dict(payload)
                return self.artifacts_dir / f"{name}.json"

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint = run_dir / "one_shot.pt"
            distill_dataset = run_dir / "distill.pt"
            torch.save({}, checkpoint)
            labels = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
            torch.save(
                {
                    "codes": torch.randn(10, 3),
                    "class_labels": labels,
                    "teacher_camera": torch.rand(10, 1, 4, 4),
                },
                distill_dataset,
            )
            train_cfg = TrainingConfig(
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                diffusion=DiffusionConfig(num_classes=2),
                one_shot=OneShotConfig(
                    code_dim=3,
                    num_classes=2,
                    sample_class_labels=(0, 1),
                    sample_repeats_per_class=4,
                ),
            )
            run_manager = RecordingRunManager(run_dir)

            run_one_shot_distill_eval_stage(
                torch,
                one_shot_model=DummyOneShot(),
                train_cfg=train_cfg,
                run_manager=run_manager,
                run_meta={},
                device=torch.device("cpu"),
                one_shot_checkpoint=str(checkpoint),
                distill_dataset=str(distill_dataset),
                load_one_shot_checkpoint_fn=lambda model, payload: None,
            )

            for key in (
                "codes_display",
                "slm_phase_display",
                "chip_field_amplitude_display",
                "chip_intensity_display",
                "encoder_input_phase_display",
                "encoder_input_grid_phase_display",
                "encoded_amplitude_display",
                "ols_input_amplitude_display",
                "ols_output_amplitude_display",
                "ols_amplitude_display",
                "decoder_input_phase_display",
                "decoder_output_amplitude_display",
            ):
                self.assertIn(key, run_manager.bundle_payload)
            amplitude_call = next(
                call for call in run_manager.grid_calls if call[0] == "one_shot_distill_eval_amplitude.png"
            )
            shared_phase_call = next(
                call
                for call in run_manager.grid_calls
                if call[0] == "one_shot_distill_eval_shared_scale_phase.png"
            )
            self.assertEqual(
                list(amplitude_call[1].keys()),
                [
                    "digit",
                    "noise code",
                    "teacher",
                    "slm command",
                    "slm amplitude grid",
                    "slm out amp",
                    "encoder in amp",
                    "encoder grid in amp",
                    "encoder mask amp = 1 L1",
                    "encoder field after L1",
                    "encoder mask amp = 1 L2",
                    "encoder field after L2",
                    "encoder out amp",
                    "ols in amp",
                    "ols out amp",
                    "decoder in amp",
                    "decoder mask amp = 1 L1",
                    "decoder field after L1",
                    "decoder mask amp = 1 L2",
                    "decoder field after L2",
                    "decoder out amp",
                    "chip amp",
                    "chip I",
                    "student",
                    "abs error",
                ],
            )
            self.assertEqual(
                list(shared_phase_call[1].keys()),
                [
                    "digit",
                    "noise code",
                    "teacher",
                    "slm in cmd",
                    "slm in cmd contrast",
                    "slm amplitude grid",
                    "slm out amp",
                    "encoder in amp",
                    "encoder grid in amp",
                    "encoder mask phase L1",
                    "encoder field after L1",
                    "encoder mask phase L2",
                    "encoder field after L2",
                    "encoder out amp",
                    "ols in amp",
                    "ols out amp",
                    "decoder in amp",
                    "decoder mask phase L1",
                    "decoder field after L1",
                    "decoder mask phase L2",
                    "decoder field after L2",
                    "decoder out amp",
                    "chip amp",
                    "chip I",
                    "student",
                    "abs error",
                ],
            )
            self.assertIn("teacher", amplitude_call[1])
            self.assertIn("student", amplitude_call[1])
            self.assertIn("chip amp", amplitude_call[1])
            self.assertIn("chip amp", shared_phase_call[1])
            self.assertIn("encoder field after L1", amplitude_call[1])
            self.assertIn("decoder field after L2", shared_phase_call[1])
            self.assertEqual(tuple(amplitude_call[1]["encoder field after L1"].shape), (4, 1, 2, 2))
            self.assertEqual(tuple(amplitude_call[1]["decoder mask amp = 1 L2"].shape), (4, 1, 2, 2))
            self.assertEqual(tuple(shared_phase_call[1]["encoder mask phase L1"].shape), (4, 1, 2, 2))
            self.assertFalse(any("_d2nn_" in call[0] for call in run_manager.grid_calls))
            self.assertNotIn("chip_amplitude_display", amplitude_call[1])
            self.assertNotIn("chip_amplitude_display", shared_phase_call[1])
            self.assertNotIn("chip_amplitude_display", run_manager.bundle_payload)
            for key in (
                "slm_command",
                "chip_amplitude",
                "chip_phase",
                "encoder_input_phase",
                "encoder_input_grid_amplitude",
                "encoded_amplitude",
                "ols_input_phase",
                "ols_output_amplitude",
                "decoder_input_phase",
                "decoder_output_amplitude",
                "slm_phase",
                "diffractive_encoder_layer_phase",
                "diffractive_decoder_mask_amplitude",
                "diffractive_camera_intensity_raw",
                "diffractive_camera_plane_raw",
                "diffractive_camera_plane_loss_raw",
            ):
                self.assertIn(key, run_manager.bundle_payload)
            eval_config = run_manager.config_payloads["one_shot_distill_eval_config"]
            metrics = run_manager.artifact_payloads["one_shot_distill_eval_metrics"]
            self.assertEqual(metrics["overall"]["count"], 8)
            self.assertEqual(metrics["selection"]["selected_count"], 8)
            self.assertEqual(eval_config["visual_preview_count"], 4)
            self.assertEqual(eval_config["visual_preview_labels"], [0, 0, 1, 1])
            eval_visual_diagnostics = run_manager.artifact_payloads["one_shot_distill_eval_visual_diagnostics"]
            self.assertEqual(eval_visual_diagnostics["view"], "compact_full_chain")
            self.assertEqual(
                eval_visual_diagnostics["alias_diffs"]["ols_output_to_decoder_input_phase"]["max_abs_diff"],
                0.0,
            )
            self.assertIn("pearson", eval_visual_diagnostics["amplitude_phase_similarity"]["decoder_out"])
            eval_d2nn_diagnostics = run_manager.artifact_payloads["one_shot_distill_eval_d2nn_diagnostics"]
            self.assertEqual(eval_d2nn_diagnostics["context"]["stage"], "one_shot_distill_eval")
            self.assertEqual(eval_d2nn_diagnostics["context"]["visual_preview_labels"], [0, 0, 1, 1])
            self.assertTrue(eval_d2nn_diagnostics["available"])
            self.assertIn("encoder field after L1", eval_d2nn_diagnostics["layers"]["amplitude"])
            self.assertIn("decoder mask phase L2", eval_d2nn_diagnostics["masks"]["phase"])
            shared_diagnostics = run_manager.artifact_payloads["one_shot_distill_eval_shared_scale_diagnostics"]
            self.assertEqual(shared_diagnostics["view"], "shared_scale_full_chain")
            self.assertIn("slm out amp", shared_diagnostics["early_raw_amplitude_columns"])
            self.assertIn("decoder out amp", shared_diagnostics["amplitude_scale_columns"])
            self.assertIn("encoder out phase", shared_diagnostics["phase_stats"])
            self.assertEqual(amplitude_call[2], 4)
            self.assertEqual(amplitude_call[3]["row_labels"], ["0", "0", "1", "1"])
            self.assertEqual(tuple(amplitude_call[1]["teacher"].shape), (4, 1, 4, 4))
            self.assertEqual(tuple(run_manager.bundle_payload["teacher_camera"].shape), (8, 1, 4, 4))

    def test_one_shot_distill_eval_metrics_include_diversity_and_hardware_terms(self) -> None:
        import torch

        from ldm_project.one_shot_eval_metrics import compute_one_shot_distill_eval_metrics

        teacher = torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[0.0, 0.8], [0.1, 1.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.8, 0.0], [1.0, 0.1]]],
            ],
            dtype=torch.float32,
        )
        student = teacher.clone()
        student[1] = student[1].mean()
        student[3] = student[3].mean()
        slm = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).reshape(4, 1, 2, 2)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        metrics = compute_one_shot_distill_eval_metrics(
            torch,
            teacher_camera=teacher,
            student_camera=student,
            slm_amplitude=slm,
            class_labels=labels,
        )

        self.assertIn("overall", metrics)
        self.assertIn("per_class", metrics)
        self.assertIn("0", metrics["per_class"])
        self.assertIn("1", metrics["per_class"])
        self.assertGreater(metrics["overall"]["paired_mse"], 0.0)
        self.assertGreaterEqual(metrics["overall"]["student_saturation_ratio"], 0.0)
        for key in ("student_low_saturation_ratio", "student_high_saturation_ratio", "foreground_hole_ratio"):
            self.assertIn(key, metrics["overall"])
            self.assertTrue(math.isfinite(metrics["overall"][key]))
            self.assertGreaterEqual(metrics["overall"][key], 0.0)
            self.assertIn(key, metrics["per_class"]["0"])
            self.assertTrue(math.isfinite(metrics["per_class"]["0"][key]))
            self.assertGreaterEqual(metrics["per_class"]["0"][key], 0.0)
        self.assertIsNotNone(metrics["overall"]["teacher_pairwise_mse"])
        self.assertIsNotNone(metrics["overall"]["student_pairwise_mse"])
        self.assertIsNotNone(metrics["overall"]["diversity_ratio"])

    def test_one_shot_loss_penalizes_same_class_code_collapse(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        collapsed = torch.full_like(target, 0.5, requires_grad=True)
        outputs = {
            "camera_plane": collapsed,
            "slm_amplitude": torch.ones_like(target),
        }
        labels = torch.tensor([0, 0], dtype=torch.long)

        losses = one_shot_loss(
            outputs,
            target,
            class_labels=labels,
            code_diversity_weight=0.1,
            teacher_pairwise_diversity_weight=0.02,
            diversity_min_teacher_mse=1.0e-4,
        )

        self.assertGreater(losses["code_diversity_penalty"].item(), 0.0)
        self.assertGreater(losses["teacher_pairwise_mse"].item(), losses["student_pairwise_mse"].item())
        self.assertEqual(losses["diversity_ratio"].item(), 0.0)
        self.assertEqual(losses["cross_class_pair_count"].item(), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(collapsed.grad)

    def test_one_shot_loss_supports_weak_class_shape_and_edge_terms(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.ones(2, 1, 4, 4, dtype=torch.float32)
        prediction = target.clone()
        prediction[0] = 0.0
        prediction.requires_grad_(True)
        outputs = {
            "camera_plane": prediction,
            "slm_amplitude": torch.linspace(0.0, 1.0, steps=32, dtype=torch.float32).reshape(2, 1, 4, 4),
        }
        labels = torch.tensor([3, 0], dtype=torch.long)

        unweighted = one_shot_loss(outputs, target, class_labels=labels)
        weighted = one_shot_loss(
            outputs,
            target,
            class_labels=labels,
            class_recon_weight_labels=(3,),
            class_recon_weight=3.0,
            shape_loss_weight=0.1,
            edge_loss_weight=0.01,
        )

        self.assertGreater(weighted["camera_recon_loss"].item(), unweighted["camera_recon_loss"].item())
        self.assertGreater(weighted["shape_loss"].item(), 0.0)
        self.assertGreaterEqual(weighted["edge_loss"].item(), 0.0)
        self.assertGreater(weighted["class_weight_mean"].item(), 1.0)
        weighted["loss"].backward()
        self.assertIsNotNone(prediction.grad)
