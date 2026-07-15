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

    def test_non_camera_primary_readout_resizes_output_and_preserves_native_planes(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig, ReadoutConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.training import one_shot_loss

        for primary_plane in ("chip_intensity", "chip_amplitude", "probe"):
            with self.subTest(primary_plane=primary_plane):
                config = OneShotConfig(
                    optical_core_arch="diffractive_phase_only",
                    code_dim=8,
                    label_embed_dim=4,
                    hidden_dim=16,
                    num_classes=3,
                    ols_kind="identity",
                    diffractive_grid_size=8,
                    diffractive_propagation_grid_size=16,
                    encoder_diffractive_layers=2,
                    decoder_diffractive_layers=2,
                    diffractive_layer_spacing=0.004,
                )
                physics = PhysicsConfig(readout=ReadoutConfig(primary_plane=primary_plane))
                model = OneShotOpticalGenerator(physics, config, image_size=16)
                code = torch.randn(2, config.code_dim)
                labels = torch.tensor([0, 2])
                target = torch.rand(2, 1, 16, 16)

                outputs = model(code, labels)
                self.assertEqual(tuple(outputs["camera_plane"].shape), (2, 1, 16, 16))
                self.assertEqual(tuple(outputs["camera_plane_loss"].shape), (2, 1, 16, 16))
                self.assertEqual(tuple(outputs["primary_readout_plane_native"].shape), (2, 1, 8, 8))
                self.assertEqual(tuple(outputs["primary_readout_plane_loss_native"].shape), (2, 1, 8, 8))
                self.assertEqual(tuple(outputs["chip_intensity_primary"].shape), (2, 1, 8, 8))
                self.assertEqual(tuple(outputs["chip_amplitude"].shape), (2, 1, 8, 8))
                self.assertEqual(tuple(outputs["diffractive_camera_plane_raw"].shape), (2, 1, 8, 8))
                self.assertTrue(torch.isfinite(outputs["camera_plane"]).all().item())

                losses = one_shot_loss(outputs, target)
                losses["loss"].backward()
                self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_v24_diagnostic_switch_defaults_preserve_forward_outputs(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        base_config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            d2nn=None,
            diffractive_grid_size=8,
            diffractive_propagation_grid_size=16,
            encoder_diffractive_layers=2,
            decoder_diffractive_layers=2,
            diffractive_layer_spacing=0.004,
        )
        explicit_config = OneShotConfig(
            **{
                **base_config.__dict__,
                "decoder_diagnostic_mode": "normal",
                "decoder_diagnostic_modulation_override": "none",
                "ols_transfer_calibration": "none",
            }
        )
        torch.manual_seed(123)
        base = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), base_config, image_size=16)
        torch.manual_seed(123)
        explicit = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            explicit_config,
            image_size=16,
        )
        code = torch.randn(2, base_config.code_dim)
        labels = torch.tensor([0, 2])

        base_outputs = base(code, labels)
        explicit_outputs = explicit(code, labels)

        for key in (
            "camera_plane",
            "ols_output_amplitude",
            "decoder_output_amplitude",
            "chip_intensity_primary",
            "physical_camera_plane",
        ):
            torch.testing.assert_close(base_outputs[key], explicit_outputs[key], rtol=0.0, atol=0.0)
        self.assertEqual(explicit_outputs["decoder_diagnostic_mode"], "normal")
        self.assertEqual(explicit_outputs["decoder_diagnostic_modulation_override"], "none")
        self.assertEqual(explicit_outputs["ols_transfer_calibration"], "none")

    def test_v51_optical_field_capture_preserves_real_ols_to_decoder_continuity(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, ReadoutConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            target_code_diagnostic_mode="none",
            target_code_bridge_source="none",
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        model = OneShotOpticalGenerator(
            PhysicsConfig(
                front_path_model="relay_coherent_psf",
                front_relay_psf_sigma_px=0.0,
                back_path_model="relay_psf",
                readout=ReadoutConfig(relay_psf_sigma_px=1.0),
            ),
            config,
            image_size=16,
        )

        outputs = model(torch.randn(2, 8), torch.tensor([0, 2]), return_optical_fields=True)

        fields = outputs["optical_fields"]
        self.assertIn("slm_output", fields)
        self.assertIn("encoder_layer_1", fields)
        self.assertIn("decoder_layer_2", fields)
        self.assertIn("chip", fields)
        self.assertNotIn("camera_field", fields)
        self.assertTrue(fields["ols_output"]["phase_available"])
        self.assertTrue(torch.equal(fields["ols_output"]["real"], fields["decoder_input"]["real"]))
        self.assertTrue(torch.equal(fields["ols_output"]["imag"], fields["decoder_input"]["imag"]))

    def test_v25_decoder_modulation_override_only_adds_decoder_amplitude(self) -> None:
        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            decoder_diagnostic_modulation_override="phase_amplitude",
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)

        self.assertFalse(hasattr(model.encoder, "amplitude_logits"))
        self.assertTrue(hasattr(model.decoder, "amplitude_logits"))
        self.assertEqual(tuple(model.decoder.amplitude_logits.shape), (2, 1, 8, 8))

    def test_v49_decoder_diagnostic_mask_component_applies_only_selected_decoder_component(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        base_kwargs = {
            "optical_core_arch": "diffractive_phase_only",
            "code_dim": 8,
            "label_embed_dim": 4,
            "hidden_dim": 16,
            "num_classes": 3,
            "ols_kind": "tanh",
            "decoder_diagnostic_modulation_override": "phase_amplitude",
            "d2nn": D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
                phase_init="small",
                amplitude_logit_init=1.0,
            ),
        }
        torch.manual_seed(49)
        trained = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs),
            image_size=16,
        )
        phase_only = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs, decoder_diagnostic_mask_component="phase_only"),
            image_size=16,
        )
        amplitude_only = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs, decoder_diagnostic_mask_component="amplitude_only"),
            image_size=16,
        )
        phase_only.load_state_dict(trained.state_dict())
        amplitude_only.load_state_dict(trained.state_dict())
        with torch.no_grad():
            phase_pattern = torch.linspace(-0.8, 0.8, steps=trained.decoder.phase.numel()).reshape_as(
                trained.decoder.phase
            )
            amplitude_pattern = torch.linspace(-2.0, 2.0, steps=trained.decoder.amplitude_logits.numel()).reshape_as(
                trained.decoder.amplitude_logits
            )
            for model in (trained, phase_only, amplitude_only):
                model.decoder.phase.copy_(phase_pattern)
                model.decoder.amplitude_logits.copy_(amplitude_pattern)

        code = torch.randn(2, 8)
        labels = torch.tensor([0, 2])
        trained_outputs = trained(code, labels)
        phase_outputs = phase_only(code, labels)
        amplitude_outputs = amplitude_only(code, labels)

        torch.testing.assert_close(
            phase_outputs["diffractive_decoder_mask_amplitude"],
            torch.ones_like(phase_outputs["diffractive_decoder_mask_amplitude"]),
        )
        torch.testing.assert_close(
            phase_outputs["diffractive_decoder_mask_phase"],
            phase_pattern,
        )
        torch.testing.assert_close(
            amplitude_outputs["diffractive_decoder_mask_amplitude"],
            torch.sigmoid(amplitude_pattern),
        )
        torch.testing.assert_close(
            amplitude_outputs["diffractive_decoder_mask_phase"],
            torch.zeros_like(amplitude_outputs["diffractive_decoder_mask_phase"]),
        )
        torch.testing.assert_close(
            trained_outputs["diffractive_encoder_mask_amplitude"],
            phase_outputs["diffractive_encoder_mask_amplitude"],
        )
        torch.testing.assert_close(
            trained_outputs["diffractive_encoder_mask_phase"],
            amplitude_outputs["diffractive_encoder_mask_phase"],
        )
        self.assertEqual(phase_outputs["decoder_diagnostic_mask_component"], "phase_only")
        self.assertEqual(amplitude_outputs["decoder_diagnostic_mask_component"], "amplitude_only")

    def test_v50_decoder_diagnostic_phase_layer_mask_selects_decoder_layers_only(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        base_kwargs = {
            "optical_core_arch": "diffractive_phase_only",
            "code_dim": 8,
            "label_embed_dim": 4,
            "hidden_dim": 16,
            "num_classes": 3,
            "ols_kind": "tanh",
            "decoder_diagnostic_modulation_override": "phase_amplitude",
            "decoder_diagnostic_mask_component": "phase_only",
            "d2nn": D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
                phase_init="small",
            ),
        }
        torch.manual_seed(50)
        both = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs),
            image_size=16,
        )
        layer1 = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs, decoder_diagnostic_phase_layer_mask=(True, False)),
            image_size=16,
        )
        layer2 = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(**base_kwargs, decoder_diagnostic_phase_layer_mask=(False, True)),
            image_size=16,
        )
        layer1.load_state_dict(both.state_dict())
        layer2.load_state_dict(both.state_dict())
        with torch.no_grad():
            phase_pattern = torch.linspace(-0.8, 0.8, steps=both.decoder.phase.numel()).reshape_as(
                both.decoder.phase
            )
            for model in (both, layer1, layer2):
                model.decoder.phase.copy_(phase_pattern)

        code = torch.randn(2, 8)
        labels = torch.tensor([0, 2])
        both_outputs = both(code, labels)
        layer1_outputs = layer1(code, labels)
        layer2_outputs = layer2(code, labels)

        torch.testing.assert_close(both_outputs["diffractive_decoder_mask_phase"], phase_pattern)
        torch.testing.assert_close(
            layer1_outputs["diffractive_decoder_mask_phase"],
            torch.stack((phase_pattern[0], torch.zeros_like(phase_pattern[1]))),
        )
        torch.testing.assert_close(
            layer2_outputs["diffractive_decoder_mask_phase"],
            torch.stack((torch.zeros_like(phase_pattern[0]), phase_pattern[1])),
        )
        torch.testing.assert_close(
            layer1_outputs["diffractive_decoder_mask_amplitude"],
            torch.ones_like(layer1_outputs["diffractive_decoder_mask_amplitude"]),
        )
        torch.testing.assert_close(
            both_outputs["diffractive_encoder_mask_phase"],
            layer1_outputs["diffractive_encoder_mask_phase"],
        )
        torch.testing.assert_close(
            both_outputs["diffractive_encoder_mask_phase"],
            layer2_outputs["diffractive_encoder_mask_phase"],
        )
        self.assertEqual(tuple(layer1_outputs["decoder_diagnostic_phase_layer_mask"]), (True, False))
        self.assertEqual(tuple(layer2_outputs["decoder_diagnostic_phase_layer_mask"]), (False, True))
        self.assertEqual(tuple(layer1_outputs["diffractive_decoder_crop_input_energy"].shape), (2, 2))
        self.assertTrue(torch.isfinite(layer1_outputs["diffractive_decoder_crop_input_energy"]).all().item())

    def test_v28_decoder_only_training_optimizer_excludes_upstream_parameters(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import _build_one_shot_optimizer

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            decoder_only_training=True,
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        train_cfg = TrainingConfig(learning_rate=1.0e-4, one_shot=config)
        model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)
        optimizer = _build_one_shot_optimizer(torch, model, train_cfg)
        optimized_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        named_parameters = dict(model.named_parameters())

        self.assertTrue(optimized_ids)
        self.assertTrue(all(id(param) in optimized_ids for name, param in named_parameters.items() if name.startswith("decoder.")))
        self.assertTrue(all(id(param) not in optimized_ids for name, param in named_parameters.items() if not name.startswith("decoder.")))

    def test_v29_aux_topology_target_transforms_are_deterministic_and_safe(self) -> None:
        import torch

        from ldm_project.one_shot_losses import transform_aux_topology_target

        target = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
        target[:, :, 8:24, 13:19] = 0.8
        target[:, :, 18:24, 8:24] = 1.0

        unchanged = transform_aux_topology_target(target, "none")
        torch.testing.assert_close(unchanged["target"], target)
        self.assertEqual(unchanged["target"].dtype, target.dtype)
        self.assertEqual(unchanged["target"].device, target.device)

        lowpass_a = transform_aux_topology_target(target, "lowpass")
        lowpass_b = transform_aux_topology_target(target, "lowpass")
        thick = transform_aux_topology_target(target, "thick_binary")

        torch.testing.assert_close(lowpass_a["target"], lowpass_b["target"])
        self.assertEqual(tuple(lowpass_a["target"].shape), tuple(target.shape))
        self.assertEqual(tuple(thick["target"].shape), tuple(target.shape))
        self.assertGreater(float(thick["foreground_ratio"]), float(unchanged["foreground_ratio"]))
        self.assertGreaterEqual(float(lowpass_a["target"].min()), 0.0)
        self.assertLessEqual(float(lowpass_a["target"].max()), 1.0)
        self.assertGreater(float(lowpass_a["target"].sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "aux_topology_target_transform"):
            transform_aux_topology_target(target, "sharpen")

    def test_v29_decoder_trainability_audit_records_updates_and_zero_update_failures(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import (
            _accumulate_target_code_bridge_trainability_gradients,
            _accumulate_decoder_trainability_gradients,
            _build_one_shot_optimizer,
            _finalize_target_code_bridge_trainability_audit,
            _finalize_decoder_trainability_audit,
            _init_target_code_bridge_trainability_audit,
            _init_decoder_trainability_audit,
        )

        torch.manual_seed(7)
        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            decoder_only_training=True,
            decoder_trainability_audit=True,
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        train_cfg = TrainingConfig(learning_rate=1.0e-2, one_shot=config)
        model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)
        optimizer = _build_one_shot_optimizer(torch, model, train_cfg)
        audit = _init_decoder_trainability_audit(model, optimizer, train_cfg)
        loss = sum(param.pow(2).sum() for name, param in model.named_parameters() if name.startswith("decoder."))
        loss.backward()
        _accumulate_decoder_trainability_gradients(audit, model)
        optimizer.step()
        finalized = _finalize_decoder_trainability_audit(audit, model)

        self.assertTrue(finalized["passed"])
        self.assertGreater(finalized["final"]["decoder"]["update_l2_norm"], 0.0)
        self.assertEqual(finalized["final"]["upstream"]["update_l2_norm"], 0.0)
        self.assertGreater(finalized["gradients"]["decoder_grad_l2_max"], 0.0)

        zero_model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)
        zero_optimizer = _build_one_shot_optimizer(torch, zero_model, train_cfg)
        zero_audit = _init_decoder_trainability_audit(zero_model, zero_optimizer, train_cfg)
        zero_finalized = _finalize_decoder_trainability_audit(zero_audit, zero_model)
        self.assertFalse(zero_finalized["passed"])
        self.assertIn("decoder-update-zero", zero_finalized["failure_reasons"])
        self.assertIn("decoder-gradient-zero", zero_finalized["failure_reasons"])

        bridge_config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            target_code_diagnostic_mode="lowfreq_digit_code",
            target_code_bridge_source="encoded_amplitude",
            target_code_bridge_hidden_channels=4,
            target_code_bridge_trainability_audit=True,
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        bridge_train_cfg = TrainingConfig(learning_rate=1.0e-2, one_shot=bridge_config)
        bridge_model = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            bridge_config,
            image_size=16,
        )
        bridge_optimizer = _build_one_shot_optimizer(torch, bridge_model, bridge_train_cfg)
        bridge_audit = _init_target_code_bridge_trainability_audit(
            bridge_model,
            bridge_optimizer,
            bridge_train_cfg,
        )
        bridge_loss = sum(
            param.pow(2).sum()
            for name, param in bridge_model.named_parameters()
            if name.startswith("target_code_bridge_adapter.")
        )
        bridge_loss.backward()
        _accumulate_target_code_bridge_trainability_gradients(bridge_audit, bridge_model)
        bridge_optimizer.step()
        bridge_finalized = _finalize_target_code_bridge_trainability_audit(bridge_audit, bridge_model)
        self.assertTrue(bridge_finalized["passed"])
        self.assertGreater(bridge_finalized["final"]["bridge"]["update_l2_norm"], 0.0)
        self.assertEqual(bridge_finalized["final"]["non_bridge"]["update_l2_norm"], 0.0)
        self.assertGreater(bridge_finalized["gradients"]["bridge_grad_l2_max"], 0.0)

    def test_v30_target_code_modes_are_default_safe_and_shape_stable(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_stage_common import (
            ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS,
            ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS,
        )

        self.assertIn("target_code_amplitude", ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS)
        self.assertIn("generator_base_command", ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS)
        self.assertIn("generator_residual_map", ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS)
        self.assertIn("generator_residual_abs_mean", ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS)
        self.assertIn("generator_residual_bg_abs_mean", ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS)
        self.assertIn("target_code_amplitude_display", ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS)

        base_config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=4,
            ols_kind="tanh",
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        explicit_none = OneShotConfig(**{**base_config.__dict__, "target_code_diagnostic_mode": "none"})
        torch.manual_seed(31)
        base = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), base_config, image_size=16)
        torch.manual_seed(31)
        explicit = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            explicit_none,
            image_size=16,
        )
        code = torch.randn(3, base_config.code_dim)
        labels = torch.tensor([0, 1, 3])

        base_outputs = base(code, labels)
        explicit_outputs = explicit(code, labels)
        for key in ("camera_plane", "decoder_input_amplitude", "decoder_output_amplitude", "physical_camera_plane"):
            torch.testing.assert_close(base_outputs[key], explicit_outputs[key], rtol=0.0, atol=0.0)
        self.assertEqual(explicit_outputs["target_code_diagnostic_mode"], "none")
        self.assertFalse(bool(explicit_outputs["target_code_diagnostic_active"].item()))

        for mode in (
            "spatial_identity_code",
            "lowfreq_digit_code",
            "thick_lowfreq_digit_code",
            "sharp_stroke_digit_code",
            "small_sharp_stroke_digit_code",
            "class_codebook_grid",
        ):
            with self.subTest(mode=mode):
                config = OneShotConfig(**{**base_config.__dict__, "target_code_diagnostic_mode": mode})
                model = OneShotOpticalGenerator(
                    PhysicsConfig(front_path_model="relay_intensity_psf"),
                    config,
                    image_size=16,
                )
                outputs = model(code, labels)
                self.assertEqual(outputs["target_code_diagnostic_mode"], mode)
                self.assertTrue(bool(outputs["target_code_diagnostic_active"].item()))
                if mode == "small_sharp_stroke_digit_code":
                    self.assertAlmostEqual(outputs["target_code_spatial_scale"], 0.72)
                else:
                    self.assertIsNone(outputs["target_code_spatial_scale"])
                self.assertEqual(tuple(outputs["target_code_amplitude"].shape), (3, 1, 8, 8))
                self.assertEqual(tuple(outputs["decoder_input_amplitude"].shape), (3, 1, 8, 8))
                self.assertTrue(torch.isfinite(outputs["target_code_amplitude"]).all().item())
                self.assertGreaterEqual(float(outputs["target_code_amplitude"].min()), 0.0)
                self.assertLessEqual(float(outputs["target_code_amplitude"].max()), 1.0)

        default_small = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(
                **{
                    **base_config.__dict__,
                    "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                }
            ),
            image_size=16,
        )
        explicit_small = OneShotOpticalGenerator(
            PhysicsConfig(front_path_model="relay_intensity_psf"),
            OneShotConfig(
                **{
                    **base_config.__dict__,
                    "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                    "target_code_spatial_scale": 0.60,
                }
            ),
            image_size=16,
        )
        torch.manual_seed(41)
        default_outputs = default_small(code, labels)
        torch.manual_seed(41)
        explicit_outputs = explicit_small(code, labels)
        self.assertAlmostEqual(default_outputs["target_code_spatial_scale"], 0.72)
        self.assertAlmostEqual(explicit_outputs["target_code_spatial_scale"], 0.60)
        self.assertEqual(tuple(explicit_outputs["target_code_amplitude"].shape), (3, 1, 8, 8))
        self.assertTrue(torch.isfinite(explicit_outputs["target_code_amplitude"]).all().item())
        self.assertGreaterEqual(float(explicit_outputs["target_code_amplitude"].min()), 0.0)
        self.assertLessEqual(float(explicit_outputs["target_code_amplitude"].max()), 1.0)
        self.assertFalse(
            torch.allclose(default_outputs["target_code_amplitude"], explicit_outputs["target_code_amplitude"])
        )

        bridge_cases = [
            ("shallow", "lowfreq_digit_code", 4),
            ("deep_residual", "thick_lowfreq_digit_code", 4),
            ("deep_residual", "sharp_stroke_digit_code", 4),
            ("deep_residual", "small_sharp_stroke_digit_code", 4),
            ("multiscale", "spatial_identity_code", 3),
        ]
        for bridge_arch, target_mode, hidden_channels in bridge_cases:
            with self.subTest(bridge_arch=bridge_arch, target_mode=target_mode):
                bridge_config = OneShotConfig(
                    **{
                        **base_config.__dict__,
                        "target_code_diagnostic_mode": target_mode,
                        "target_code_bridge_source": "encoded_amplitude",
                        "target_code_bridge_arch": bridge_arch,
                        "target_code_bridge_hidden_channels": hidden_channels,
                    }
                )
                bridge_model = OneShotOpticalGenerator(
                    PhysicsConfig(front_path_model="relay_intensity_psf"),
                    bridge_config,
                    image_size=16,
                )
                bridge_outputs = bridge_model(code, labels)
                self.assertEqual(bridge_outputs["target_code_diagnostic_mode"], target_mode)
                self.assertEqual(bridge_outputs["target_code_bridge_source"], "encoded_amplitude")
                self.assertEqual(bridge_outputs["target_code_bridge_arch"], bridge_arch)
                self.assertTrue(bool(bridge_outputs["target_code_diagnostic_active"].item()))
                self.assertTrue(bool(bridge_outputs["target_code_bridge_active"].item()))
                self.assertEqual(tuple(bridge_outputs["target_code_amplitude"].shape), (3, 1, 8, 8))
                self.assertEqual(tuple(bridge_outputs["decoder_input_amplitude"].shape), (3, 1, 8, 8))
                self.assertTrue(torch.isfinite(bridge_outputs["target_code_amplitude"]).all().item())
                self.assertGreaterEqual(float(bridge_outputs["target_code_amplitude"].min()), 0.0)
                self.assertLessEqual(float(bridge_outputs["target_code_amplitude"].max()), 1.0)
                self.assertTrue(
                    any(name.startswith("target_code_bridge_adapter.") for name, _ in bridge_model.named_parameters())
                )

    def test_v25_two_layer_checkpoint_loads_into_four_layer_decoder_with_sliced_phase(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint_subset,
        )

        source_cfg = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            d2nn=D2NNConfig(
                grid_size=8,
                propagation_grid_size=16,
                encoder_layers=2,
                decoder_layers=2,
                layer_spacing=0.004,
                modulation="phase_only",
            ),
        )
        target_cfg = OneShotConfig(
            **{
                **source_cfg.__dict__,
                "d2nn": D2NNConfig(
                    grid_size=8,
                    propagation_grid_size=16,
                    encoder_layers=2,
                    decoder_layers=4,
                    layer_spacing=0.004,
                    modulation="phase_only",
                ),
            }
        )
        torch.manual_seed(11)
        source = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), source_cfg, image_size=16)
        torch.manual_seed(22)
        target = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), target_cfg, image_size=16)
        payload = {
            "model_version": ONE_SHOT_MODEL_VERSION,
            "model_state_dict": source.state_dict(),
        }

        summary = load_one_shot_checkpoint_subset(
            target,
            payload,
            include_prefixes=("encoder.", "decoder."),
            description="v25 two-to-four layer optical warm-start",
        )

        self.assertGreaterEqual(summary["sliced"], 1)
        torch.testing.assert_close(target.decoder.phase[:2], source.decoder.phase, rtol=0.0, atol=0.0)

    def test_v25_artifact_metadata_sidecar_round_trips_non_tensor_strings(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot_eval import _one_shot_artifact_metadata
        from ldm_project.run_manager import RunManager

        cfg = TrainingConfig(
            one_shot=OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                decoder_diagnostic_mode="bypass_decoder",
                decoder_diagnostic_modulation_override="phase_amplitude",
                ols_transfer_calibration="phase_screen",
                d2nn=D2NNConfig(
                    grid_size=100,
                    pixel_size=2.0e-6,
                    encoder_layers=2,
                    decoder_layers=4,
                    layer_spacing=1.50e-3,
                    modulation="phase_only",
                ),
            ),
            physics=PhysicsConfig(
                front_path_model="relay_intensity_psf",
                back_path_model="relay_psf",
            ),
        )
        outputs = {
            "decoder_diagnostic_mode": "bypass_decoder",
            "decoder_diagnostic_modulation_override": "phase_amplitude",
            "ols_transfer_calibration": "phase_screen",
            "decoder_bypass_active": torch.tensor(True),
            "primary_readout_plane": "camera",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir) / "run")
            metadata = _one_shot_artifact_metadata(
                stage="one_shot_distill_eval",
                train_cfg=cfg,
                one_shot_checkpoint="outputs/example/checkpoints/one_shot_distill_best.pt",
                distill_dataset="outputs/example/artifacts/val.pt",
                outputs=outputs,
            )
            path = manager.save_artifact_json("one_shot_distill_eval_artifact_metadata", metadata)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["decoder_diagnostic_mode"], "bypass_decoder")
        self.assertEqual(payload["decoder_diagnostic_modulation_override"], "phase_amplitude")
        self.assertEqual(payload["ols_transfer_calibration"], "phase_screen")
        self.assertTrue(payload["decoder_bypass_active"])
        self.assertEqual(payload["d2nn"]["decoder_layers"], 4)

    def test_v24_decoder_bypass_exposes_ols_output_at_decoder_plane(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            decoder_diagnostic_mode="bypass_decoder",
            diffractive_grid_size=8,
            diffractive_propagation_grid_size=16,
            encoder_diffractive_layers=2,
            decoder_diffractive_layers=2,
            diffractive_layer_spacing=0.004,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)
        code = torch.randn(2, config.code_dim)
        labels = torch.tensor([0, 2])

        outputs = model(code, labels)

        torch.testing.assert_close(
            outputs["decoder_output_amplitude"],
            outputs["ols_output_amplitude"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(bool(outputs["decoder_bypass_active"].item()))
        self.assertEqual(outputs["decoder_diagnostic_mode"], "bypass_decoder")

    def test_v42_propagation_only_decoder_outputs_finite_diagnostic_bundle(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        config = OneShotConfig(
            optical_core_arch="diffractive_phase_only",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            ols_kind="tanh",
            decoder_diagnostic_mode="propagation_only_decoder",
            diffractive_grid_size=8,
            diffractive_propagation_grid_size=16,
            encoder_diffractive_layers=2,
            decoder_diffractive_layers=2,
            diffractive_layer_spacing=0.004,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(front_path_model="relay_intensity_psf"), config, image_size=16)
        outputs = model(torch.randn(2, config.code_dim), torch.tensor([0, 2]))

        self.assertEqual(outputs["decoder_diagnostic_mode"], "propagation_only_decoder")
        self.assertFalse(bool(outputs["decoder_bypass_active"].item()))
        self.assertTrue(bool(outputs["decoder_propagation_only_active"].item()))
        self.assertEqual(tuple(outputs["diffractive_decoder_layer_amplitude"].shape), (2, 2, 1, 8, 8))
        for key in (
            "decoder_output_amplitude",
            "physical_camera_plane",
            "diffractive_decoder_crop_crop_retention",
            "diffractive_decoder_crop_propagation_energy_ratio",
        ):
            self.assertTrue(torch.isfinite(outputs[key]).all(), key)

    def test_v24_ols_calibration_screens_have_finite_gradients(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.training import one_shot_loss

        for calibration in ("phase_screen", "amplitude_phase_screen"):
            with self.subTest(calibration=calibration):
                config = OneShotConfig(
                    optical_core_arch="diffractive_phase_only",
                    code_dim=8,
                    label_embed_dim=4,
                    hidden_dim=16,
                    num_classes=3,
                    ols_kind="tanh",
                    ols_transfer_calibration=calibration,
                    diffractive_grid_size=8,
                    diffractive_propagation_grid_size=16,
                    encoder_diffractive_layers=2,
                    decoder_diffractive_layers=2,
                    diffractive_layer_spacing=0.004,
                )
                model = OneShotOpticalGenerator(
                    PhysicsConfig(front_path_model="relay_intensity_psf"),
                    config,
                    image_size=16,
                )
                code = torch.randn(2, config.code_dim)
                labels = torch.tensor([0, 2])
                target = torch.rand(2, 1, 16, 16)

                outputs = model(code, labels)
                self.assertEqual(tuple(outputs["ols_output_amplitude"].shape), (2, 1, 8, 8))
                losses = one_shot_loss(outputs, target)
                losses["loss"].backward()

                self.assertIsNotNone(model.ols_phase_screen)
                self.assertIsNotNone(model.ols_phase_screen.grad)
                self.assertTrue(torch.isfinite(model.ols_phase_screen.grad).all().item())
                if calibration == "amplitude_phase_screen":
                    self.assertIsNotNone(model.ols_amplitude_log)
                    self.assertIsNotNone(model.ols_amplitude_log.grad)
                    self.assertTrue(torch.isfinite(model.ols_amplitude_log.grad).all().item())

    def test_readout_primary_plane_rejects_unknown_value(self) -> None:
        from ldm_project.config import ReadoutConfig

        with self.assertRaisesRegex(ValueError, "readout.primary_plane"):
            ReadoutConfig(primary_plane="not_a_plane")

    def test_direct_camera_oracle_forward_is_eval_compatible_on_cpu(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.training import one_shot_loss

        config = OneShotConfig(
            optical_core_arch="camera_direct_clean_oracle",
            code_dim=8,
            label_embed_dim=4,
            hidden_dim=16,
            num_classes=3,
            metasurface_grid_size=8,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=16)
        code = torch.randn(2, config.code_dim)
        labels = torch.tensor([0, 2])
        target = torch.rand(2, 1, 16, 16)

        outputs = model(code, labels)

        for key in (
            "x_hat",
            "camera_plane",
            "camera_plane_loss",
            "slm_command",
            "slm_amplitude",
            "slm_phase",
            "encoder_input_amplitude",
            "encoded_amplitude",
            "ols_output_amplitude",
            "decoder_output_amplitude",
            "chip_amplitude",
            "probe_plane",
        ):
            self.assertIn(key, outputs)
            self.assertEqual(tuple(outputs[key].shape), (2, 1, 16, 16), key)
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
        losses = one_shot_loss(outputs, target, clean_target_enabled=True, clean_foreground_dice_weight=0.1)
        losses["loss"].backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

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

    def test_phase_amplitude_d2nn_forward_exposes_attenuation_masks(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.d2nn_diagnostics import build_one_shot_d2nn_diagnostics
        from ldm_project.one_shot import OneShotOpticalGenerator

        model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="phase_amplitude_d2nn",
                d2nn=D2NNConfig(
                    grid_size=8,
                    propagation_grid_size=8,
                    encoder_layers=2,
                    decoder_layers=2,
                    modulation="phase_amplitude",
                    amplitude_logit_init=2.0,
                ),
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
            ),
            image_size=8,
        )
        code = torch.randn(2, 4)
        labels = torch.tensor([0, 2], dtype=torch.long)

        outputs = model(code, labels)
        mask_amplitude = outputs["diffractive_encoder_mask_amplitude"]

        self.assertEqual(tuple(outputs["camera_plane"].shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(outputs["camera_plane"]).all().item())
        self.assertEqual(tuple(mask_amplitude.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all(mask_amplitude >= 0.0).item())
        self.assertTrue(torch.all(mask_amplitude <= 1.0).item())
        self.assertFalse(torch.allclose(mask_amplitude, torch.ones_like(mask_amplitude)))
        self.assertAlmostEqual(float(model.encoder.amplitude_logits.mean().item()), 2.0, places=6)
        self.assertAlmostEqual(
            float(mask_amplitude.mean().item()),
            float(torch.sigmoid(torch.tensor(2.0))),
            places=6,
        )
        diagnostics = build_one_shot_d2nn_diagnostics(outputs)
        self.assertEqual(diagnostics["view"], "d2nn_phase_amplitude_feasibility")
        self.assertEqual(diagnostics["mask_modulation"], "phase_amplitude")

    def test_phase_amplitude_d2nn_rejects_phase_only_modulation(self) -> None:
        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator

        with self.assertRaisesRegex(ValueError, "phase_amplitude"):
            OneShotOpticalGenerator(
                PhysicsConfig(),
                OneShotConfig(
                    optical_core_arch="phase_amplitude_d2nn",
                    d2nn=D2NNConfig(grid_size=8, propagation_grid_size=8, modulation="phase_only"),
                    code_dim=4,
                    label_embed_dim=2,
                    hidden_dim=8,
                ),
                image_size=8,
            )

    def test_phase_amplitude_optimizer_uses_independent_amplitude_learning_rate(self) -> None:
        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import _build_one_shot_optimizer

        model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="phase_amplitude_d2nn",
                d2nn=D2NNConfig(grid_size=8, propagation_grid_size=8, modulation="phase_amplitude"),
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
            ),
            image_size=8,
        )
        train_cfg = TrainingConfig(learning_rate=2.0e-5)
        train_cfg.one_shot = model.config
        train_cfg.one_shot.phase_learning_rate = 1.0e-3
        train_cfg.one_shot.amplitude_learning_rate = 5.0e-3

        optimizer = _build_one_shot_optimizer(__import__("torch"), model, train_cfg)
        lr_by_param_id = {
            id(param): group["lr"]
            for group in optimizer.param_groups
            for param in group["params"]
        }

        self.assertAlmostEqual(lr_by_param_id[id(model.encoder.phase)], 1.0e-3, places=10)
        self.assertAlmostEqual(lr_by_param_id[id(model.encoder.amplitude_logits)], 5.0e-3, places=10)
        self.assertAlmostEqual(lr_by_param_id[id(model.decoder.phase)], 1.0e-3, places=10)
        self.assertAlmostEqual(lr_by_param_id[id(model.decoder.amplitude_logits)], 5.0e-3, places=10)

    def test_zero_lr_parameter_groups_remain_exactly_frozen(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import (
            _build_one_shot_optimizer,
            _snapshot_frozen_one_shot_parameters,
            _verify_frozen_one_shot_parameters,
        )

        config = OneShotConfig(
            optical_core_arch="phase_amplitude_d2nn",
            generator_e_arch="label_template",
            generator_command_transform="intensity_to_amplitude",
            generator_template_init="mnist_prototype",
            distill_target_source="mnist_prototype",
            generator_base_lr_scale=0.0,
            generator_residual_lr_scale=0.0,
            phase_learning_rate=0.0,
            amplitude_learning_rate=0.0,
            d2nn=D2NNConfig(grid_size=8, propagation_grid_size=8, modulation="phase_amplitude"),
            code_dim=4,
            label_embed_dim=2,
            hidden_dim=8,
            num_classes=2,
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=8)
        model.generator_e.initialize_label_templates(torch.rand(2, 1, 8, 8))
        train_cfg = TrainingConfig(learning_rate=1.0e-4, one_shot=config)
        snapshots = _snapshot_frozen_one_shot_parameters(model, train_cfg)
        optimizer = _build_one_shot_optimizer(torch, model, train_cfg)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        report = _verify_frozen_one_shot_parameters(model, snapshots)

        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_change"], 0.0)
        self.assertEqual(set(report["groups"]), {"generator_base", "phase", "amplitude"})
        with torch.no_grad():
            model.encoder.phase.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "Frozen one-shot parameters changed"):
            _verify_frozen_one_shot_parameters(model, snapshots)

    def test_phase_amplitude_grad_and_mask_summaries_name_phase_and_amp(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import _mask_amplitude_summary, _phase_grad_summary

        model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="phase_amplitude_d2nn",
                d2nn=D2NNConfig(grid_size=8, propagation_grid_size=8, modulation="phase_amplitude"),
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
            ),
            image_size=8,
        )
        outputs = model(torch.randn(2, 4), torch.tensor([0, 1], dtype=torch.long))
        outputs["camera_plane"].mean().backward()

        grad_summary = _phase_grad_summary(model)
        self.assertIn("encoder_phase_grad=", grad_summary)
        self.assertIn("encoder_amp_grad=", grad_summary)
        self.assertIn("decoder_phase_grad=", grad_summary)
        self.assertIn("decoder_amp_grad=", grad_summary)
        self.assertNotIn("encoder_grad=", grad_summary)

        mask_summary = _mask_amplitude_summary(model)
        self.assertIn("mask_amp_mean=", mask_summary)
        self.assertIn("mask_amp_std=", mask_summary)
        self.assertIn("mask_amp_min=", mask_summary)
        self.assertIn("mask_amp_max=", mask_summary)

    def test_one_shot_optical_identity_audit_variants_are_finite_on_cpu(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_identity_audit import (
            IDENTITY_AUDIT_METRIC_KEYS,
            IDENTITY_AUDIT_VARIANTS,
            build_one_shot_optical_identity_audit,
        )

        model = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="phase_amplitude_d2nn",
                d2nn=D2NNConfig(
                    grid_size=8,
                    propagation_grid_size=8,
                    encoder_layers=1,
                    decoder_layers=1,
                    layer_spacing=0.0005,
                    modulation="phase_amplitude",
                    phase_init="small",
                    amplitude_logit_init=4.0,
                ),
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
            ),
            image_size=8,
        )
        codes = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)

        audit = build_one_shot_optical_identity_audit(torch, one_shot_model=model, codes=codes, labels=labels)

        self.assertEqual(tuple(audit["variants"].keys()), IDENTITY_AUDIT_VARIANTS)
        for variant, tensor in audit["variants"].items():
            self.assertEqual(tuple(tensor.shape), (3, 1, 8, 8), variant)
            self.assertTrue(torch.isfinite(tensor).all().item(), variant)
            self.assertIn(variant, audit["metrics"])
            for key in IDENTITY_AUDIT_METRIC_KEYS:
                self.assertIn(key, audit["metrics"][variant])
                self.assertTrue(math.isfinite(float(audit["metrics"][variant][key])))
        self.assertTrue(audit["zero_mask_checks"]["encoder_mask_amplitude_allclose_one"])
        self.assertTrue(audit["zero_mask_checks"]["decoder_mask_amplitude_allclose_one"])
        self.assertAlmostEqual(audit["zero_mask_checks"]["encoder_mask_phase_max_abs"], 0.0, places=6)
        self.assertAlmostEqual(audit["zero_mask_checks"]["decoder_mask_phase_max_abs"], 0.0, places=6)

    def test_one_shot_identity_audit_artifacts_are_named_as_intensity(self) -> None:
        import torch

        from ldm_project.one_shot_identity_audit import _save_variant_artifacts

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.destinations = []

            def save_tensor_grid(self, tensor_map, destination, **kwargs):
                path = Path(destination)
                self.destinations.append((dict(tensor_map), path, dict(kwargs)))
                return path

        with tempfile.TemporaryDirectory() as tmpdir:
            run_manager = RecordingRunManager(Path(tmpdir))
            paths = _save_variant_artifacts(
                run_manager,
                variant="relay_psf_sigma1",
                output=torch.rand(2, 1, 4, 4),
                reference=torch.rand(2, 1, 4, 4),
                labels=torch.tensor([0, 1]),
            )

        self.assertEqual(set(paths), {"intensity", "shared_scale_intensity"})
        self.assertEqual(
            [destination.name for _, destination, _ in run_manager.destinations],
            ["relay_psf_sigma1_intensity.png", "relay_psf_sigma1_shared_scale_intensity.png"],
        )
        self.assertFalse(any("amplitude" in destination.name for _, destination, _ in run_manager.destinations))

    def test_old_one_shot_checkpoint_version_is_rejected(self) -> None:
        from ldm_project.one_shot import validate_one_shot_checkpoint_payload

        with self.assertRaisesRegex(ValueError, "unexpected model_version"):
            validate_one_shot_checkpoint_payload(
                {
                    "model_version": "ldm_distilled_single_shot_optical_student_v2",
                    "model_state_dict": {},
                }
            )

    def test_one_shot_checkpoint_loads_matching_submodules_across_ablation_architectures(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint,
        )

        source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=2,
                decoder_diffractive_layers=2,
            ),
            image_size=8,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="dense_linear",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
            ),
            image_size=8,
        )
        key = "generator_e.label_embedding.weight"
        expected = source.state_dict()[key].clone()
        target.state_dict()[key].zero_()

        load_one_shot_checkpoint(
            target,
            {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()},
        )

        self.assertTrue(torch.equal(target.state_dict()[key], expected))

    def test_one_shot_checkpoint_subset_loads_generator_and_optical_modules_separately(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint_subset,
        )

        common = dict(
            code_dim=4,
            label_embed_dim=2,
            hidden_dim=8,
            num_classes=3,
            metasurface_grid_size=4,
        )
        generator_source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(optical_core_arch="camera_direct_clean_oracle", **common),
            image_size=8,
        )
        optical_source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=2,
                decoder_diffractive_layers=2,
                **common,
            ),
            image_size=8,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=2,
                decoder_diffractive_layers=2,
                **common,
            ),
            image_size=8,
        )
        with torch.no_grad():
            generator_source.generator_e.net[-1].bias.fill_(0.25)
            optical_source.encoder.phase.fill_(0.75)
            optical_source.decoder.phase.fill_(-0.50)
            target.generator_e.net[-1].bias.zero_()
            target.encoder.phase.zero_()
            target.decoder.phase.zero_()

        load_one_shot_checkpoint_subset(
            target,
            {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": generator_source.state_dict()},
            include_prefixes=("generator_e.",),
            description="unit generator subset",
        )

        self.assertTrue(torch.equal(target.generator_e.net[-1].bias, generator_source.generator_e.net[-1].bias))
        self.assertEqual(float(target.encoder.phase.abs().sum().item()), 0.0)

        load_one_shot_checkpoint_subset(
            target,
            {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": optical_source.state_dict()},
            include_prefixes=("encoder.", "decoder."),
            description="unit optical subset",
        )

        self.assertTrue(torch.equal(target.encoder.phase, optical_source.encoder.phase))
        self.assertTrue(torch.equal(target.decoder.phase, optical_source.decoder.phase))

    def test_phase_only_optical_checkpoint_warm_starts_phase_amplitude_core_phase(self) -> None:
        import torch

        from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint_subset,
        )

        common = dict(
            code_dim=4,
            label_embed_dim=2,
            hidden_dim=8,
            num_classes=3,
            metasurface_grid_size=4,
        )
        optical_source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=2,
                decoder_diffractive_layers=2,
                **common,
            ),
            image_size=8,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="phase_amplitude_d2nn",
                d2nn=D2NNConfig(
                    grid_size=8,
                    propagation_grid_size=8,
                    encoder_layers=2,
                    decoder_layers=2,
                    modulation="phase_amplitude",
                    amplitude_logit_init=2.0,
                ),
                **common,
            ),
            image_size=8,
        )
        with torch.no_grad():
            optical_source.encoder.phase.fill_(0.75)
            optical_source.decoder.phase.fill_(-0.50)
            target.encoder.phase.zero_()
            target.decoder.phase.zero_()

        summary = load_one_shot_checkpoint_subset(
            target,
            {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": optical_source.state_dict()},
            include_prefixes=("encoder.", "decoder."),
            description="unit phase-only optical warm-start",
            expected_missing_prefixes=("encoder.amplitude_logits", "decoder.amplitude_logits"),
        )

        self.assertEqual(summary["loaded"], 2)
        self.assertTrue(torch.equal(target.encoder.phase, optical_source.encoder.phase))
        self.assertTrue(torch.equal(target.decoder.phase, optical_source.decoder.phase))
        self.assertTrue(
            torch.allclose(
                target.encoder.amplitude_logits,
                torch.full_like(target.encoder.amplitude_logits, 2.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                target.decoder.amplitude_logits,
                torch.full_like(target.decoder.amplitude_logits, 2.0),
            )
        )

    def test_one_shot_checkpoint_partially_warm_starts_template_residual_generator(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint,
            load_one_shot_checkpoint_subset,
        )

        source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="camera_direct_clean_oracle",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
            ),
            image_size=8,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="camera_direct_clean_oracle",
                generator_e_arch="template_residual_mlp",
                code_residual_scale=0.02,
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
            ),
            image_size=8,
        )
        key = "generator_e.net.0.weight"
        expected = source.state_dict()[key].clone()
        payload = {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()}

        with self.assertRaisesRegex(ValueError, "full checkpoint load requires an exact model match"):
            load_one_shot_checkpoint(target, payload)

        load_one_shot_checkpoint_subset(
            target,
            payload,
            include_prefixes=("generator_e.",),
            description="unit template residual generator warm-start",
            expected_missing_prefixes=("generator_e.residual_net.",),
        )

        self.assertTrue(torch.equal(target.state_dict()[key], expected))
        self.assertIn("generator_e.residual_net.0.weight", target.state_dict())

    def test_one_shot_optimizer_applies_generator_lr_scales(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig, TrainingConfig
        from ldm_project.one_shot import OneShotOpticalGenerator
        from ldm_project.one_shot_training_utils import _build_one_shot_optimizer, _one_shot_parameter_groups

        train_cfg = TrainingConfig(
            learning_rate=1.0e-4,
            one_shot=OneShotConfig(
                optical_core_arch="camera_direct_clean_oracle",
                generator_e_arch="template_residual_mlp",
                code_residual_scale=0.02,
                generator_base_lr_scale=0.0,
                generator_residual_lr_scale=0.5,
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
            ),
        )
        model = OneShotOpticalGenerator(PhysicsConfig(), train_cfg.one_shot, image_size=8)

        optimizer = _build_one_shot_optimizer(torch, model, train_cfg)
        lrs = sorted({float(group["lr"]) for group in optimizer.param_groups})

        self.assertIn(0.0, lrs)
        self.assertIn(5.0e-5, lrs)

        residual_cfg = TrainingConfig(
            learning_rate=1.0e-4,
            one_shot=OneShotConfig(
                optical_core_arch="camera_direct_clean_oracle",
                generator_e_arch="label_template_code_residual",
                generator_command_transform="intensity_to_amplitude",
                generator_template_init="mnist_prototype",
                distill_target_source="mnist_sample",
                code_residual_scale=0.05,
                generator_base_lr_scale=0.0,
                generator_residual_lr_scale=1.0,
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
            ),
        )
        residual_model = OneShotOpticalGenerator(PhysicsConfig(), residual_cfg.one_shot, image_size=8)
        groups = _one_shot_parameter_groups(residual_model)
        base_names = {name for name, _ in groups["generator_base"]}
        residual_names = {name for name, _ in groups["generator_residual"]}

        self.assertIn("generator_e.template_logits", base_names)
        self.assertTrue(any(name.startswith("generator_e.residual_net.") for name in residual_names))
        self.assertFalse(any(name.startswith("generator_e.residual_net.") for name in base_names))

    def test_one_shot_checkpoint_slices_diffractive_phase_layers_for_fair_shallow_ablation(self) -> None:
        import torch

        from ldm_project.config import OneShotConfig, PhysicsConfig
        from ldm_project.one_shot import (
            ONE_SHOT_MODEL_VERSION,
            OneShotOpticalGenerator,
            load_one_shot_checkpoint,
        )

        source = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=2,
                decoder_diffractive_layers=2,
            ),
            image_size=8,
        )
        target = OneShotOpticalGenerator(
            PhysicsConfig(),
            OneShotConfig(
                optical_core_arch="diffractive_phase_only",
                code_dim=4,
                label_embed_dim=2,
                hidden_dim=8,
                num_classes=3,
                metasurface_grid_size=4,
                diffractive_grid_size=8,
                diffractive_propagation_grid_size=8,
                encoder_diffractive_layers=1,
                decoder_diffractive_layers=1,
            ),
            image_size=8,
        )
        expected_encoder = source.encoder.phase[:1].detach().clone()
        expected_decoder = source.decoder.phase[:1].detach().clone()

        load_one_shot_checkpoint(
            target,
            {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()},
        )

        self.assertTrue(torch.equal(target.encoder.phase.detach(), expected_encoder))
        self.assertTrue(torch.equal(target.decoder.phase.detach(), expected_decoder))
