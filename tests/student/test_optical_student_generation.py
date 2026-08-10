# ruff: noqa: F401,F403,F405,F811
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from tests.tests_support import (
    _assert_nonblank_phase_columns_are_labeled,
    _cuda_device,
    _dummy_one_shot_outputs,
)
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class OneShotOpticalStudentGenerationTests(unittest.TestCase):
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
    def test_template_residual_generator_zero_scale_matches_base(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE

        generator = ElectronicGeneratorE(
            image_size=8,
            grid_size=4,
            code_dim=5,
            num_classes=3,
            label_embed_dim=4,
            hidden_dim=12,
            arch="template_residual_mlp",
            code_residual_scale=0.0,
        )
        code = torch.randn(2, 5)
        labels = torch.tensor([0, 2], dtype=torch.long)

        command, diagnostics = generator.forward_with_diagnostics(code, labels)
        base = diagnostics["generator_base_command"]

        self.assertTrue(torch.allclose(command, base))
        self.assertEqual(tuple(diagnostics["generator_residual_map"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(diagnostics["generator_residual_abs_mean"].shape), (2,))
    def test_label_template_inverse_outputs_actual_command_and_ignores_code(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE, IdealAmplitudeSLMModel

        generator = ElectronicGeneratorE(
            image_size=4,
            grid_size=4,
            code_dim=5,
            num_classes=2,
            label_embed_dim=3,
            hidden_dim=8,
            arch="label_template",
            command_transform="intensity_to_amplitude",
            background_floor=0.05,
            template_init="mnist_prototype",
        )
        templates = torch.stack(
            [
                torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
                torch.linspace(1.0, 0.0, 16).reshape(1, 4, 4),
            ]
        )
        generator.initialize_label_templates(templates)
        labels = torch.tensor([0, 0, 1])
        code_a = torch.randn(3, 5)
        code_b = torch.randn(3, 5)

        command_a, diagnostics_a = generator.forward_with_diagnostics(code_a, labels)
        command_b, diagnostics_b = generator.forward_with_diagnostics(code_b, labels)
        slm_intensity = IdealAmplitudeSLMModel(0.05)(command_a)["slm_amplitude"].square()

        torch.testing.assert_close(command_a, command_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            diagnostics_a["generator_desired_intensity"],
            diagnostics_b["generator_desired_intensity"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.equal(command_a[0], command_a[1]))
        self.assertFalse(torch.equal(command_a[0], command_a[2]))
        expected_intensity = templates.index_select(0, labels).clamp_min(0.05**2)
        torch.testing.assert_close(slm_intensity, expected_intensity, atol=2.0e-6, rtol=0.0)
        self.assertEqual(int(diagnostics_a["generator_command_transform"].item()), 1)
        self.assertFalse(bool(diagnostics_a["generator_code_conditioning_active"].item()))
    def test_label_template_code_residual_zero_scale_matches_template(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE

        common = dict(
            image_size=4,
            grid_size=4,
            code_dim=5,
            num_classes=2,
            label_embed_dim=3,
            hidden_dim=8,
            command_transform="intensity_to_amplitude",
            background_floor=0.05,
            template_init="mnist_prototype",
        )
        template = ElectronicGeneratorE(arch="label_template", **common)
        residual = ElectronicGeneratorE(
            arch="label_template_code_residual",
            code_residual_scale=0.0,
            **common,
        )
        templates = torch.stack(
            [
                torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
                torch.linspace(1.0, 0.0, 16).reshape(1, 4, 4),
            ]
        )
        template.initialize_label_templates(templates)
        residual.initialize_label_templates(templates)
        labels = torch.tensor([0, 1], dtype=torch.long)
        code = torch.randn(2, 5)

        template_command, template_diag = template.forward_with_diagnostics(code, labels)
        residual_command, residual_diag = residual.forward_with_diagnostics(code, labels)

        torch.testing.assert_close(residual_command, template_command, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            residual_diag["generator_desired_intensity"],
            template_diag["generator_desired_intensity"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(tuple(residual_diag["generator_residual_map"].shape), (2, 1, 4, 4))
        self.assertTrue(bool(residual_diag["generator_code_conditioning_active"].item()))
    def test_label_template_code_residual_nonzero_scale_changes_with_code(self) -> None:
        import torch

        from ldm_project.one_shot import ElectronicGeneratorE

        generator = ElectronicGeneratorE(
            image_size=4,
            grid_size=4,
            code_dim=5,
            num_classes=2,
            label_embed_dim=3,
            hidden_dim=8,
            arch="label_template_code_residual",
            code_residual_scale=0.2,
            command_transform="intensity_to_amplitude",
            background_floor=0.05,
            template_init="mnist_prototype",
        )
        templates = torch.stack(
            [
                torch.full((1, 4, 4), 0.25),
                torch.full((1, 4, 4), 0.75),
            ]
        )
        generator.initialize_label_templates(templates)
        labels = torch.tensor([0, 0], dtype=torch.long)
        codes = torch.stack([torch.zeros(5), torch.ones(5) * 3.0], dim=0)

        command, diagnostics = generator.forward_with_diagnostics(codes, labels)

        self.assertTrue(torch.isfinite(command).all().item())
        self.assertGreaterEqual(float(command.min().item()), 0.0)
        self.assertLessEqual(float(command.max().item()), 1.0)
        self.assertFalse(torch.allclose(command[0], command[1]))
        self.assertGreater(float(diagnostics["generator_residual_abs_mean"].mean().item()), 0.0)
    def test_diagnostic_slm_override_is_default_off_and_can_perturb_command(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        common = dict(
            optical_core_arch="camera_direct_clean_oracle",
            generator_e_arch="label_template",
            generator_command_transform="intensity_to_amplitude",
            generator_template_init="mnist_prototype",
            distill_target_source="mnist_prototype",
            code_dim=8,
            num_classes=2,
            label_embed_dim=3,
            hidden_dim=8,
        )
        templates = torch.stack(
            [
                torch.full((1, 8, 8), 0.35),
                torch.full((1, 8, 8), 0.65),
            ]
        )
        labels = torch.tensor([0, 0], dtype=torch.long)
        codes = torch.stack([torch.zeros(8), torch.ones(8) * 2.0], dim=0)

        baseline = OneShotOpticalGenerator(PhysicsConfig(), OneShotConfig(**common), image_size=8)
        baseline.generator_e.initialize_label_templates(templates)
        baseline_outputs = baseline(codes, labels)

        torch.testing.assert_close(baseline_outputs["slm_command"][0], baseline_outputs["slm_command"][1])
        self.assertFalse(bool(baseline_outputs["diagnostic_slm_override_active"].item()))

        oracle = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                **common,
                diagnostic_slm_override_mode="lowfreq_additive",
                diagnostic_slm_override_amplitude=0.10,
            ),
            image_size=8,
        )
        oracle.generator_e.initialize_label_templates(templates)
        oracle_outputs = oracle(codes, labels)

        self.assertTrue(bool(oracle_outputs["diagnostic_slm_override_active"].item()))
        self.assertEqual(tuple(oracle_outputs["diagnostic_slm_override_pattern"].shape), (2, 1, 8, 8))
        self.assertFalse(torch.allclose(oracle_outputs["slm_command"][0], oracle_outputs["slm_command"][1]))
        self.assertTrue(torch.isfinite(oracle_outputs["slm_command"]).all().item())
        self.assertGreaterEqual(float(oracle_outputs["slm_command"].min().item()), 0.0)
        self.assertLessEqual(float(oracle_outputs["slm_command"].max().item()), 1.0)
    def test_label_template_rejects_mlp_checkpoint_structure(self) -> None:
        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint,
        )

        source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(optical_core_arch="camera_direct_clean_oracle", code_dim=4, num_classes=2),
            image_size=4,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="camera_direct_clean_oracle",
                generator_e_arch="label_template",
                generator_command_transform="intensity_to_amplitude",
                generator_template_init="mnist_prototype",
                distill_target_source="mnist_prototype",
                code_dim=4,
                num_classes=2,
            ),
            image_size=4,
        )
        payload = {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()}

        with self.assertRaisesRegex(ValueError, "exact model match"):
            load_one_shot_checkpoint(target, payload)
    def test_label_template_checkpoint_is_self_contained_for_inference(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint,
        )

        config = OneShotConfig(
            optical_core_arch="camera_direct_clean_oracle",
            generator_e_arch="label_template",
            generator_command_transform="intensity_to_amplitude",
            generator_template_init="mnist_prototype",
            distill_target_source="mnist_prototype",
            code_dim=4,
            num_classes=2,
        )
        source = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=4)
        target = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=4)
        templates = torch.stack([torch.eye(4).unsqueeze(0), torch.eye(4).flip(0).unsqueeze(0)])
        source.generator_e.initialize_label_templates(templates)
        payload = {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()}
        load_one_shot_checkpoint(target, payload)
        code = torch.randn(2, 4)
        labels = torch.tensor([0, 1])

        torch.testing.assert_close(
            source(code, labels)["slm_command"],
            target(code, labels)["slm_command"],
            rtol=0.0,
            atol=0.0,
        )
    def test_relay_intensity_psf_sigma_zero_preserves_slm_intensity(self) -> None:
        import torch

        from ldm_project.config import PhysicsConfig
        from ldm_project.optics import FrontOpticalPath

        physics = PhysicsConfig(front_path_model="relay_intensity_psf", front_relay_psf_sigma_px=0.0)
        front_path = FrontOpticalPath(physics)
        slm_transfer = torch.rand(2, 1, 12, 12)

        field = front_path.forward_field(slm_transfer)

        self.assertTrue(torch.is_complex(field))
        self.assertEqual(tuple(field.shape), tuple(slm_transfer.shape))
        torch.testing.assert_close(field.imag, torch.zeros_like(field.imag), rtol=0.0, atol=0.0)
        torch.testing.assert_close(field.abs().pow(2), slm_transfer.pow(2), rtol=1.0e-6, atol=1.0e-6)
    def test_relay_intensity_psf_positive_sigma_blurs_intensity(self) -> None:
        import torch

        from ldm_project.config import PhysicsConfig
        from ldm_project.optics import FrontOpticalPath

        physics = PhysicsConfig(front_path_model="relay_intensity_psf", front_relay_psf_sigma_px=1.0)
        front_path = FrontOpticalPath(physics)
        slm_transfer = torch.zeros(1, 1, 17, 17)
        slm_transfer[:, :, 8, 8] = 1.0

        field = front_path.forward_field(slm_transfer)
        relayed_intensity = field.abs().pow(2)

        self.assertTrue(torch.is_complex(field))
        self.assertEqual(tuple(field.shape), tuple(slm_transfer.shape))
        self.assertGreater(float(relayed_intensity[:, :, 8, 8]), 0.0)
        self.assertLess(float(relayed_intensity[:, :, 8, 8]), 1.0)
        self.assertGreater(float(relayed_intensity[:, :, 8, 9]), 0.0)
    def test_invalid_front_path_model_rejects_unknown_values(self) -> None:
        from ldm_project.config import PhysicsConfig

        with self.assertRaisesRegex(ValueError, "physics.front_path_model"):
            PhysicsConfig(front_path_model="not_a_front_path")
    def test_label_template_desired_intensity_only_appears_in_intensity_visuals(self) -> None:
        import torch

        from ldm_project.one_shot_visualization import (
            build_one_shot_shared_scale_visualization_tensors,
            build_one_shot_visualization_tensors,
        )

        outputs = _dummy_one_shot_outputs(batch_size=2, image_size=8, grid_size=4)
        desired = torch.rand(2, 1, 8, 8)
        outputs["generator_desired_intensity"] = desired
        outputs["generator_command_transform"] = torch.tensor(1)
        outputs["generator_code_conditioning_active"] = torch.tensor(False)
        reference = torch.rand(2, 1, 8, 8)

        visuals = build_one_shot_visualization_tensors(reference, outputs)
        shared = build_one_shot_shared_scale_visualization_tensors(reference, outputs)

        self.assertTrue(torch.equal(visuals["intensity"]["generator desired intensity"], desired))
        self.assertNotIn("generator desired intensity", visuals["amplitude"])
        self.assertNotIn("generator desired intensity", visuals["phase"])
        self.assertEqual(
            shared["diagnostics"]["generator"]["command_transform"],
            "intensity_to_amplitude",
        )
        self.assertFalse(shared["diagnostics"]["generator"]["code_conditioning_active"])
    def test_residual_visualization_builder_skips_missing_residual_keys(self) -> None:
        import torch

        from ldm_project.one_shot_visualization import build_one_shot_distill_residual_visualization_tensors

        reference = torch.zeros(2, 1, 8, 8)
        outputs = {
            "camera_plane": torch.ones(2, 1, 8, 8),
            "slm_amplitude": torch.ones(2, 1, 8, 8),
        }

        self.assertIsNone(build_one_shot_distill_residual_visualization_tensors(reference, outputs))
    def test_residual_visualization_builder_outputs_expected_columns(self) -> None:
        import torch

        from ldm_project.one_shot_visualization import build_one_shot_distill_residual_visualization_tensors
        from ldm_project.visualization_core import VISUALIZATION_INTENSITY_KEY

        reference = torch.zeros(2, 1, 8, 8)
        base = torch.full((2, 1, 8, 8), 0.4)
        residual = torch.linspace(-1.0, 1.0, steps=128).reshape(2, 1, 8, 8)
        outputs = {
            "camera_plane": (base + 0.05 * residual).clamp(0.0, 1.0),
            "generator_base_command": base,
            "generator_residual_map": residual,
            "slm_amplitude": torch.ones(2, 1, 8, 8),
        }

        visualization = build_one_shot_distill_residual_visualization_tensors(reference, outputs)

        self.assertIsNotNone(visualization)
        amplitude = visualization[VISUALIZATION_INTENSITY_KEY]
        self.assertEqual(
            tuple(amplitude.keys()),
            (
                "reference/prototype",
                "student raw",
                "generator_base_command",
                "generator_residual_map",
                "abs residual",
                "student-base diff",
            ),
        )
        for tensor in amplitude.values():
            self.assertEqual(tuple(tensor.shape), (2, 1, 8, 8))
            self.assertGreaterEqual(float(tensor.min().item()), 0.0)
            self.assertLessEqual(float(tensor.max().item()), 1.0)
