from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for one-shot optical student smoke tests.")
    return torch.device("cuda")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class OneShotOpticalStudentTests(unittest.TestCase):
    def test_generator_outputs_amplitude_slm_command_on_cuda(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE, IdealAmplitudeSLMModel

        device = _cuda_device()
        generator = ElectronicGeneratorE(
            image_size=16,
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
            "slm_amplitude",
            "chip_amplitude",
            "chip_phase",
            "probe_plane",
            "camera_plane_loss",
        ):
            self.assertIn(key, outputs)
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 16, 16))
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


class OneShotConfigTests(unittest.TestCase):
    def test_smoke_config_loads_one_shot_fields(self) -> None:
        from ldm_project.config import load_training_config

        config = load_training_config(Path("configs/train_one_shot_smoke.yaml"))

        self.assertEqual(config.mode, "one_shot")
        self.assertEqual(config.one_shot_epochs, 1)
        self.assertEqual(config.one_shot.code_dim, 16)
        self.assertEqual(config.one_shot.label_embed_dim, 8)
        self.assertEqual(config.one_shot.hidden_dim, 32)
        self.assertEqual(config.one_shot.num_classes, 10)
        self.assertEqual(config.one_shot.sample_class_labels, (0, 1))

    def test_distillation_configs_load(self) -> None:
        from ldm_project.config import load_training_config

        export_config = load_training_config(Path("configs/export_teacher_distill_mnist_2k.yaml"))
        train_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16.yaml"))
        sat_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16_sat.yaml"))
        div_light_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16_div_w003.yaml"))
        div_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16_div_w010.yaml"))
        div_sat_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16_div_w010_sat.yaml"))
        shape_config = load_training_config(Path("configs/train_one_shot_distill_mnist_2k_grid16_shape_w003.yaml"))
        shape_class_config = load_training_config(
            Path("configs/train_one_shot_distill_mnist_2k_grid16_shape_class.yaml")
        )
        shape_class_smooth_config = load_training_config(
            Path("configs/train_one_shot_distill_mnist_2k_grid16_shape_class_smooth.yaml")
        )
        grid32_shape_class_config = load_training_config(
            Path("configs/train_one_shot_distill_mnist_2k_grid32_shape_class.yaml")
        )
        grid32_shape_class_slm_config = load_training_config(
            Path("configs/train_one_shot_distill_mnist_2k_grid32_shape_class_slm1e3.yaml")
        )

        self.assertEqual(export_config.mode, "teacher_distill_export")
        self.assertEqual(export_config.distillation.samples_per_class, 200)
        self.assertEqual(export_config.one_shot.code_dim, 128)
        self.assertEqual(train_config.mode, "one_shot_distill")
        self.assertEqual(train_config.one_shot.metasurface_grid_size, 16)
        self.assertAlmostEqual(train_config.one_shot.slm_smoothness_weight, 1.0e-4, places=8)
        self.assertAlmostEqual(train_config.one_shot.camera_saturation_weight, 1.0e-3, places=8)
        self.assertEqual(sat_config.mode, "one_shot_distill")
        self.assertEqual(sat_config.one_shot.metasurface_grid_size, 16)
        self.assertAlmostEqual(sat_config.one_shot.slm_smoothness_weight, 5.0e-4, places=8)
        self.assertAlmostEqual(sat_config.one_shot.camera_saturation_weight, 1.0e-2, places=8)
        self.assertAlmostEqual(div_light_config.one_shot.code_diversity_weight, 3.0e-2, places=8)
        self.assertAlmostEqual(div_light_config.one_shot.teacher_pairwise_diversity_weight, 5.0e-3, places=8)
        self.assertAlmostEqual(div_config.one_shot.code_diversity_weight, 1.0e-1, places=8)
        self.assertAlmostEqual(div_config.one_shot.teacher_pairwise_diversity_weight, 2.0e-2, places=8)
        self.assertAlmostEqual(div_sat_config.one_shot.camera_saturation_weight, 1.0e-2, places=8)
        self.assertAlmostEqual(div_sat_config.one_shot.code_diversity_weight, 1.0e-1, places=8)
        self.assertAlmostEqual(shape_config.one_shot.shape_loss_weight, 5.0e-2, places=8)
        self.assertAlmostEqual(shape_config.one_shot.edge_loss_weight, 1.0e-2, places=8)
        self.assertEqual(shape_class_config.one_shot.class_recon_weight_labels, (3, 4, 5, 6))
        self.assertAlmostEqual(shape_class_config.one_shot.class_recon_weight, 2.0, places=8)
        self.assertAlmostEqual(shape_class_smooth_config.one_shot.slm_smoothness_weight, 5.0e-4, places=8)
        self.assertAlmostEqual(shape_class_smooth_config.one_shot.shape_loss_weight, 5.0e-2, places=8)
        self.assertEqual(grid32_shape_class_config.one_shot.metasurface_grid_size, 32)
        self.assertEqual(grid32_shape_class_config.one_shot.class_recon_weight_labels, (3, 4, 5, 6))
        self.assertAlmostEqual(grid32_shape_class_config.one_shot.code_diversity_weight, 3.0e-2, places=8)
        self.assertAlmostEqual(grid32_shape_class_config.one_shot.shape_loss_weight, 0.0, places=8)
        self.assertEqual(grid32_shape_class_slm_config.one_shot.metasurface_grid_size, 32)
        self.assertAlmostEqual(grid32_shape_class_slm_config.one_shot.slm_smoothness_weight, 1.0e-3, places=8)
        self.assertAlmostEqual(grid32_shape_class_slm_config.one_shot.camera_saturation_weight, 1.0e-3, places=8)
        self.assertAlmostEqual(grid32_shape_class_slm_config.one_shot.edge_loss_weight, 0.0, places=8)

    def test_one_shot_visualization_includes_raw_and_display_camera(self) -> None:
        import torch

        from ldm_project.sampling import build_one_shot_visualization_tensors

        reference = torch.rand(2, 1, 16, 16)
        camera = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": camera,
            "slm_amplitude": torch.rand(2, 1, 16, 16),
            "chip_amplitude": torch.rand(2, 1, 16, 16),
            "chip_phase": torch.zeros(2, 1, 16, 16),
        }

        tensors = build_one_shot_visualization_tensors(reference, outputs)

        self.assertIn("generated_camera_raw", tensors)
        self.assertIn("generated_camera_display", tensors)
        self.assertTrue(torch.allclose(tensors["generated_camera_raw"], camera))
        self.assertTrue(torch.all(tensors["generated_camera_display"] >= 0.0).item())
        self.assertTrue(torch.all(tensors["generated_camera_display"] <= 1.0).item())

    def test_one_shot_distill_eval_metrics_include_diversity_and_hardware_terms(self) -> None:
        import torch

        from ldm_project.stage_runners import compute_one_shot_distill_eval_metrics

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
