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


def _assert_nonblank_phase_columns_are_labeled(test_case, tensor_map: dict[str, object]) -> None:
    import torch

    for label, value in tensor_map.items():
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().float()
        if torch.isnan(tensor).all().item():
            continue
        test_case.assertIn("phase", label, f"nonblank phase column must say phase: {label}")


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
        from ldm_project.one_shot_training_utils import _build_one_shot_optimizer

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

    def test_frozen_generator_requires_explicit_warm_start(self) -> None:
        from ldm_project.config import OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_training import _validate_frozen_generator_warm_start

        frozen_cfg = TrainingConfig(one_shot=OneShotConfig(generator_base_lr_scale=0.0))
        with self.assertRaisesRegex(ValueError, "--one-shot-generator-checkpoint"):
            _validate_frozen_generator_warm_start(
                frozen_cfg,
                one_shot_checkpoint=None,
                one_shot_generator_checkpoint=None,
            )

        _validate_frozen_generator_warm_start(
            frozen_cfg,
            one_shot_checkpoint="student.pt",
            one_shot_generator_checkpoint=None,
        )
        _validate_frozen_generator_warm_start(
            frozen_cfg,
            one_shot_checkpoint=None,
            one_shot_generator_checkpoint="generator.pt",
        )
        _validate_frozen_generator_warm_start(
            TrainingConfig(one_shot=OneShotConfig(generator_base_lr_scale=1.0)),
            one_shot_checkpoint=None,
            one_shot_generator_checkpoint=None,
        )
        _validate_frozen_generator_warm_start(
            TrainingConfig(
                one_shot=OneShotConfig(
                    generator_e_arch="label_template",
                    generator_command_transform="intensity_to_amplitude",
                    generator_template_init="mnist_prototype",
                    distill_target_source="mnist_prototype",
                    generator_base_lr_scale=0.0,
                )
            ),
            one_shot_checkpoint=None,
            one_shot_generator_checkpoint=None,
        )

    def test_v9_8_full_relay_short_train_config_requires_generator_warm_start(self) -> None:
        from ldm_project.config import load_training_config

        config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_"
                "front_coherent_psf_back_psf_relay.yaml"
            )
        )

        self.assertEqual(config.one_shot_epochs, 10)
        self.assertEqual(config.physics.front_path_model, "relay_coherent_psf")
        self.assertEqual(config.physics.back_path_model, "relay_psf")
        self.assertEqual(config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(config.one_shot.generator_base_lr_scale, 0.0)
        self.assertEqual(config.one_shot.generator_residual_lr_scale, 0.0)

    def test_v9_10_label_template_inverse_audit_config_contract(self) -> None:
        from ldm_project.config import load_training_config

        config = load_training_config(
            Path("configs/audit_one_shot_d2nn_v9_10_label_template_inverse_prototype_baseline.yaml")
        )

        self.assertEqual(config.one_shot_epochs, 1)
        self.assertEqual(config.one_shot.generator_e_arch, "label_template")
        self.assertEqual(config.one_shot.generator_command_transform, "intensity_to_amplitude")
        self.assertEqual(config.one_shot.generator_template_init, "mnist_prototype")
        self.assertEqual(config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(config.one_shot.generator_base_lr_scale, 0.0)
        self.assertEqual(config.one_shot.generator_residual_lr_scale, 0.0)
        self.assertEqual(config.one_shot.phase_learning_rate, 0.0)
        self.assertEqual(config.one_shot.amplitude_learning_rate, 0.0)
        self.assertEqual(config.physics.front_path_model, "relay_coherent_psf")
        self.assertEqual(config.physics.back_path_model, "relay_psf")
        self.assertEqual(config.one_shot.d2nn.grid_size, 200)
        self.assertEqual(config.one_shot.d2nn.propagation_grid_size, 1024)
        self.assertAlmostEqual(config.one_shot.d2nn.layer_spacing, 0.0005, places=8)

        train_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_"
                "label_template_inverse_prototype_baseline.yaml"
            )
        )
        eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_"
                "label_template_inverse_prototype_baseline_strict_readability_test.yaml"
            )
        )
        for formal_config in (train_config, eval_config):
            self.assertEqual(formal_config.one_shot.generator_e_arch, "label_template")
            self.assertEqual(
                formal_config.one_shot.generator_command_transform,
                "intensity_to_amplitude",
            )
            self.assertEqual(formal_config.one_shot.generator_template_init, "mnist_prototype")
            self.assertEqual(formal_config.physics.front_path_model, "relay_coherent_psf")
            self.assertEqual(formal_config.physics.back_path_model, "relay_psf")
        self.assertEqual(train_config.one_shot_epochs, 1)
        self.assertEqual(eval_config.one_shot.eval_samples_per_class, 10)

    def test_twophoton_100x100_full_connect_config_and_shapes(self) -> None:
        import torch

        from ldm_project.config import PhysicsConfig, load_training_config
        from ldm_project.one_shot import OneShotOpticalGenerator

        train_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_"
                "twophoton_100x100_2um_full_connect.yaml"
            )
        )
        eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_"
                "twophoton_100x100_2um_full_connect_strict_readability_test.yaml"
            )
        )

        for config in (train_config, eval_config):
            d2nn = config.one_shot.d2nn
            self.assertIsNotNone(d2nn)
            self.assertEqual(config.first_stage.image_size, 200)
            self.assertEqual(config.one_shot.optical_core_arch, "diffractive_phase_only")
            self.assertEqual(d2nn.grid_size, 100)
            self.assertEqual(d2nn.propagation_grid_size, 1024)
            self.assertEqual(d2nn.encoder_layers, 2)
            self.assertEqual(d2nn.decoder_layers, 2)
            self.assertEqual(d2nn.modulation, "phase_only")
            self.assertAlmostEqual(d2nn.wavelength, 532.0e-9, places=15)
            self.assertAlmostEqual(d2nn.pixel_size, 2.0e-6, places=15)
            self.assertAlmostEqual(d2nn.layer_spacing, 1.50e-3, places=12)

            neuron_count = d2nn.grid_size * d2nn.grid_size
            minimum_spacing = (
                math.sqrt(neuron_count)
                * d2nn.pixel_size
                * math.sqrt(4.0 * d2nn.pixel_size**2 / d2nn.wavelength**2 - 1.0)
            )
            self.assertAlmostEqual(minimum_spacing, 1.490e-3, delta=1.0e-6)
            self.assertGreaterEqual(d2nn.layer_spacing, minimum_spacing)

        model = OneShotOpticalGenerator(
            PhysicsConfig(),
            train_config.one_shot,
            image_size=train_config.first_stage.image_size,
        ).eval()
        self.assertEqual(tuple(model.encoder.phase.shape), (2, 1, 100, 100))
        self.assertEqual(tuple(model.decoder.phase.shape), (2, 1, 100, 100))
        with torch.no_grad():
            outputs = model(
                torch.zeros(1, train_config.one_shot.code_dim),
                torch.zeros(1, dtype=torch.long),
            )
        self.assertEqual(tuple(outputs["diffractive_camera_intensity_raw"].shape), (1, 1, 100, 100))
        self.assertEqual(tuple(outputs["camera_plane"].shape), (1, 1, 200, 200))
        self.assertTrue(torch.isfinite(outputs["camera_plane"]).all().item())

    def test_distillation_configs_load(self) -> None:
        from ldm_project.config import load_training_config

        export_3k_fast200_config = load_training_config(Path("configs/export_teacher_distill_mnist_3k_fast200.yaml"))
        export_15k_fast200_config = load_training_config(Path("configs/export_teacher_distill_mnist_15k_fast200.yaml"))
        filter_config = load_training_config(Path("configs/filter_teacher_distill_mnist_3k_fast200_top2k.yaml"))
        strict_filter_config = load_training_config(
            Path("configs/filter_teacher_distill_mnist_3k_fast200_label_strict_top1k.yaml")
        )
        readable_split_config = load_training_config(
            Path("configs/split_teacher_distill_readable_top1k_80_10_10.yaml")
        )
        identity_audit_config = load_training_config(
            Path("configs/audit_one_shot_d2nn_v8_7a4_optical_identity_chain.yaml")
        )
        readout_ablation_audit_config = load_training_config(
            Path("configs/audit_one_shot_d2nn_v9_2_readout_plane_ablation.yaml")
        )
        hardclass_recovery_config = load_training_config(
            Path("configs/train_first_stage_hardclass_camera_recovery.yaml")
        )
        margin_recovery_config = load_training_config(
            Path("configs/train_first_stage_hardclass_camera_recovery_margin.yaml")
        )
        acceptance_margin_config = load_training_config(
            Path("configs/train_first_stage_acceptance_margin_recovery.yaml")
        )
        train_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml"))
        strict_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v3_strict_readability.yaml")
        )
        student_v3_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v3.yaml")
        )
        student_v4_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v4_readability.yaml")
        )
        student_v5_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v5_speckle_label.yaml")
        )
        student_v6_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v6_clean_bg.yaml")
        )
        student_v7_1_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_1_clean_target_fixed.yaml")
        )
        student_v7_3_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_3_denoise.yaml")
        )
        student_v7_3_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_3_strict_readability_test.yaml")
        )
        student_v8_identity_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_identity_oracle.yaml")
        )
        student_v8_shallow_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn.yaml")
        )
        student_v8_identity_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_identity_strict_readability_test.yaml")
        )
        student_v8_shallow_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_strict_readability_test.yaml")
        )
        student_v8_shallow_fair_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_fair.yaml")
        )
        student_v8_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_direct_clean_oracle.yaml")
        )
        student_v8_direct_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_direct_clean_oracle_strict_readability_test.yaml")
        )
        student_v8_1_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1_direct_mnist_prototype_oracle.yaml")
        )
        student_v8_1b_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1b_direct_mnist_prototype_oracle_focus1.yaml")
        )
        student_v8_2_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_2_direct_mnist_prototype_oracle_detail_polish.yaml")
        )
        student_v8_2c_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_2c_direct_mnist_prototype_oracle_margin_finish.yaml")
        )
        student_v8_3_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_3_direct_mnist_sample_oracle.yaml")
        )
        student_v8_4_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_4_direct_mnist_nearest_teacher_image_oracle.yaml")
        )
        student_v8_5a_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5a_direct_prototype_residual_safety.yaml")
        )
        student_v8_5b_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5b_direct_prototype_anchor_aux_detail.yaml")
        )
        student_v8_5c_direct_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5c_direct_prototype_residual_probe.yaml")
        )
        student_v8_5d_direct_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5d_direct_prototype_residual_foreground_gate.yaml"
            )
        )
        student_v8_6a_config = load_training_config(
            Path("configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6a_phase2x2_prototype_anchor.yaml")
        )
        student_v8_6b_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6b_phase2x2_random_phase_prototype_anchor.yaml"
            )
        )
        student_v8_7a_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a_phase_amplitude2x2_prototype_anchor.yaml"
            )
        )
        student_v8_7a2_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a2_phase_amplitude_warm_phase_active_amp.yaml"
            )
        )
        student_v8_7a3_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a3_near_identity_phase_amp_feasibility.yaml"
            )
        )
        student_v9_1_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_1_trainable_generator_phase_amp_feasibility.yaml"
            )
        )
        student_v8_1_direct_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1_direct_mnist_prototype_oracle_strict_readability_test.yaml")
        )
        student_v8_3_direct_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_3_direct_mnist_sample_oracle_strict_readability_test.yaml")
        )
        student_v8_4_direct_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_4_direct_mnist_nearest_teacher_image_oracle_strict_readability_test.yaml")
        )
        student_v8_5c_direct_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5c_direct_prototype_residual_probe_strict_readability_test.yaml")
        )
        student_v8_5d_direct_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5d_direct_prototype_residual_foreground_gate_strict_readability_test.yaml"
            )
        )
        student_v8_6a_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6a_phase2x2_prototype_anchor_strict_readability_test.yaml"
            )
        )
        student_v8_6b_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6b_phase2x2_random_phase_prototype_anchor_strict_readability_test.yaml"
            )
        )
        student_v8_7a_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a_phase_amplitude2x2_prototype_anchor_strict_readability_test.yaml"
            )
        )
        student_v8_7a2_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a2_phase_amplitude_warm_phase_active_amp_strict_readability_test.yaml"
            )
        )
        student_v8_7a3_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a3_near_identity_phase_amp_feasibility_strict_readability_test.yaml"
            )
        )
        student_v9_1_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_1_trainable_generator_phase_amp_feasibility_strict_readability_test.yaml"
            )
        )
        student_v9_8_relay_psf_config = load_training_config(
            Path(
                "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_relay_psf_camera_primary.yaml"
            )
        )
        student_v9_8_relay_psf_eval_config = load_training_config(
            Path(
                "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_relay_psf_camera_primary_strict_readability_test.yaml"
            )
        )
        student_v8_shallow_fair_eval_config = load_training_config(
            Path("configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_fair_strict_readability_test.yaml")
        )
        feasibility_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_gpu_feasibility.yaml"))
        long_config = load_training_config(Path("configs/train_one_shot_distill_d2nn_long.yaml"))

        self.assertEqual(export_3k_fast200_config.mode, "teacher_distill_export")
        self.assertEqual(export_3k_fast200_config.distillation.samples_per_class, 300)
        self.assertEqual(export_3k_fast200_config.distillation.sampling_steps, 200)
        self.assertEqual(export_15k_fast200_config.mode, "teacher_distill_export")
        self.assertEqual(export_15k_fast200_config.distillation.samples_per_class, 1500)
        self.assertEqual(export_15k_fast200_config.distillation.sampling_steps, 200)
        self.assertEqual(export_15k_fast200_config.distillation.class_labels, tuple(range(10)))
        self.assertEqual(filter_config.mode, "teacher_distill_filter")
        self.assertEqual(filter_config.distillation.samples_per_class, 300)
        self.assertEqual(filter_config.distillation.filter_keep_per_class, 200)
        self.assertAlmostEqual(filter_config.distillation.filter_foreground_trim_fraction, 0.05, places=8)
        self.assertAlmostEqual(filter_config.distillation.filter_contrast_drop_fraction, 0.10, places=8)
        self.assertAlmostEqual(filter_config.distillation.filter_foreground_threshold, 0.2, places=8)
        self.assertEqual(strict_filter_config.mode, "teacher_distill_label_strict_filter")
        self.assertEqual(strict_filter_config.distillation.label_strict_keep_per_class, 100)
        self.assertEqual(strict_filter_config.distillation.label_audit_prototypes_per_class, 64)
        self.assertEqual(strict_filter_config.distillation.label_audit_batch_size, 32)
        self.assertEqual(readable_split_config.mode, "teacher_distill_readable_split")
        self.assertAlmostEqual(readable_split_config.distillation.readable_min_label_margin, 0.03, places=8)
        self.assertAlmostEqual(readable_split_config.distillation.readable_min_contrast, 0.70, places=8)
        self.assertAlmostEqual(readable_split_config.distillation.readable_max_background_speckle, 0.20, places=8)
        self.assertAlmostEqual(readable_split_config.distillation.readable_max_foreground_hole, 0.08, places=8)
        self.assertEqual(readable_split_config.distillation.readable_train_per_class, 80)
        self.assertEqual(readable_split_config.distillation.readable_val_per_class, 10)
        self.assertEqual(readable_split_config.distillation.readable_test_per_class, 10)
        self.assertEqual(identity_audit_config.mode, "one_shot_optical_identity_audit")
        self.assertEqual(identity_audit_config.batch_size, 10)
        self.assertEqual(identity_audit_config.one_shot_epochs, 0)
        self.assertEqual(identity_audit_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(identity_audit_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(identity_audit_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(identity_audit_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(identity_audit_config.one_shot.d2nn.phase_init, "small")
        self.assertAlmostEqual(identity_audit_config.one_shot.d2nn.layer_spacing, 0.0005, places=8)
        self.assertAlmostEqual(identity_audit_config.one_shot.d2nn.amplitude_logit_init, 4.0, places=8)
        self.assertFalse(identity_audit_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(readout_ablation_audit_config.mode, "one_shot_optical_identity_audit")
        self.assertFalse(readout_ablation_audit_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(readout_ablation_audit_config.physics.readout.primary_plane, "camera")
        self.assertEqual(student_v9_8_relay_psf_config.mode, "one_shot_distill")
        self.assertEqual(student_v9_8_relay_psf_config.one_shot_epochs, 10)
        self.assertEqual(student_v9_8_relay_psf_config.physics.front_path_model, "free_space")
        self.assertEqual(student_v9_8_relay_psf_config.physics.back_path_model, "relay_psf")
        self.assertAlmostEqual(student_v9_8_relay_psf_config.physics.readout.relay_psf_sigma_px, 1.0, places=8)
        self.assertEqual(student_v9_8_relay_psf_eval_config.one_shot_epochs, 0)
        self.assertEqual(student_v9_8_relay_psf_eval_config.physics.front_path_model, "free_space")
        self.assertEqual(student_v9_8_relay_psf_eval_config.physics.back_path_model, "relay_psf")
        self.assertAlmostEqual(student_v9_8_relay_psf_eval_config.physics.readout.relay_psf_sigma_px, 1.0, places=8)
        self.assertEqual(hardclass_recovery_config.mode, "first_stage")
        self.assertEqual(hardclass_recovery_config.first_stage_epochs, 12)
        self.assertAlmostEqual(hardclass_recovery_config.learning_rate, 5.0e-5, places=10)
        self.assertEqual(hardclass_recovery_config.first_stage.hard_class_weight_labels, (2, 3, 4, 5, 7, 8))
        self.assertAlmostEqual(hardclass_recovery_config.first_stage.hard_class_weight, 2.0, places=8)
        self.assertAlmostEqual(hardclass_recovery_config.first_stage.camera_topology_weight, 0.35, places=8)
        self.assertAlmostEqual(hardclass_recovery_config.first_stage.camera_artifact_weight, 0.75, places=8)
        self.assertEqual(margin_recovery_config.mode, "first_stage")
        self.assertEqual(margin_recovery_config.first_stage.optical_encoder_input_mode, "pf2_complex_field")
        self.assertEqual(margin_recovery_config.first_stage.optical_encoder_arch, "linear_metasurface")
        self.assertEqual(margin_recovery_config.first_stage.metasurface_decoder_arch, "linear_metasurface")
        self.assertEqual(margin_recovery_config.first_stage.hard_class_weight_labels, (2, 3, 4, 5, 7, 8))
        self.assertAlmostEqual(margin_recovery_config.first_stage.hard_class_weight, 2.0, places=8)
        self.assertAlmostEqual(margin_recovery_config.first_stage.camera_label_margin_weight, 2.0, places=8)
        self.assertAlmostEqual(margin_recovery_config.first_stage.camera_label_margin_target, 0.08, places=8)
        self.assertEqual(margin_recovery_config.first_stage.camera_label_margin_labels, (2, 3, 4, 5, 7, 8))
        self.assertEqual(margin_recovery_config.first_stage.camera_label_margin_prototypes_per_class, 64)
        self.assertEqual(margin_recovery_config.first_stage.checkpoint_selection_metric, "camera_label_margin_score")
        self.assertEqual(acceptance_margin_config.mode, "first_stage")
        self.assertEqual(acceptance_margin_config.first_stage.optical_encoder_input_mode, "pf2_complex_field")
        self.assertEqual(acceptance_margin_config.first_stage.optical_encoder_arch, "linear_metasurface")
        self.assertEqual(acceptance_margin_config.first_stage.hard_class_weight_labels, (4, 7, 8))
        self.assertAlmostEqual(acceptance_margin_config.first_stage.hard_class_weight, 2.5, places=8)
        self.assertAlmostEqual(acceptance_margin_config.learning_rate, 3.0e-5, places=10)
        self.assertEqual(acceptance_margin_config.first_stage_epochs, 12)
        self.assertAlmostEqual(acceptance_margin_config.first_stage.camera_label_margin_weight, 4.0, places=8)
        self.assertAlmostEqual(acceptance_margin_config.first_stage.camera_label_margin_target, 0.10, places=8)
        self.assertEqual(acceptance_margin_config.first_stage.camera_label_margin_labels, (4, 7, 8))
        self.assertEqual(acceptance_margin_config.first_stage.camera_label_margin_prototypes_per_class, 128)
        self.assertEqual(acceptance_margin_config.first_stage.checkpoint_selection_metric, "camera_label_margin_score")
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
        self.assertTrue(strict_eval_config.one_shot.student_label_audit_enabled)
        self.assertEqual(strict_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertEqual(strict_eval_config.one_shot.sample_repeats_per_class, 2)
        self.assertEqual(strict_eval_config.one_shot.eval_samples_per_class, 5)
        self.assertEqual(strict_eval_config.distillation.label_audit_prototypes_per_class, 128)
        self.assertEqual(student_v3_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v3_config.one_shot.eval_preview_per_class, 2)
        self.assertEqual(student_v3_config.one_shot.eval_samples_per_class, 5)
        self.assertAlmostEqual(student_v3_config.one_shot.shape_loss_weight, 0.12, places=8)
        self.assertAlmostEqual(student_v3_config.one_shot.edge_loss_weight, 0.06, places=8)
        self.assertAlmostEqual(student_v3_config.one_shot.camera_topology_weight, 0.04, places=8)
        self.assertAlmostEqual(student_v3_config.one_shot.camera_readability_weight, 0.8, places=8)
        self.assertAlmostEqual(student_v3_config.one_shot.code_diversity_weight, 0.03, places=8)
        self.assertTrue(student_v3_config.one_shot.student_label_audit_enabled)
        self.assertEqual(student_v4_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v4_config.one_shot.student_label_margin_labels, (2, 3, 4, 5, 8, 9))
        self.assertAlmostEqual(student_v4_config.one_shot.camera_background_speckle_weight, 0.02, places=8)
        self.assertAlmostEqual(student_v4_config.one_shot.camera_readable_contrast_weight, 0.01, places=8)
        self.assertAlmostEqual(student_v4_config.one_shot.student_label_margin_weight, 0.5, places=8)
        self.assertEqual(student_v4_config.one_shot.sample_repeats_per_class, 2)
        self.assertEqual(student_v4_config.one_shot.eval_samples_per_class, 5)
        self.assertEqual(student_v4_config.one_shot.eval_preview_per_class, 2)
        self.assertEqual(student_v5_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v5_config.one_shot.student_label_margin_prototype_source, "mnist")
        self.assertEqual(student_v5_config.one_shot.student_label_margin_labels, (2, 3, 5, 8, 9))
        self.assertEqual(student_v5_config.one_shot.student_label_margin_focus_label, 2)
        self.assertAlmostEqual(student_v5_config.one_shot.student_label_margin_focus_weight, 2.0, places=8)
        self.assertAlmostEqual(student_v5_config.learning_rate, 2.0e-5, places=10)
        self.assertEqual(student_v5_config.one_shot_epochs, 3)
        self.assertAlmostEqual(student_v5_config.one_shot.camera_background_hotspot_weight, 0.08, places=8)
        self.assertAlmostEqual(student_v5_config.one_shot.camera_background_highfreq_weight, 0.05, places=8)
        self.assertEqual(student_v5_config.one_shot.eval_samples_per_class, 5)
        self.assertEqual(student_v6_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v6_config.one_shot.student_label_margin_prototype_source, "mnist")
        self.assertAlmostEqual(student_v6_config.learning_rate, 1.0e-5, places=10)
        self.assertAlmostEqual(student_v6_config.one_shot.phase_learning_rate, 1.0e-3, places=10)
        self.assertEqual(student_v6_config.one_shot_epochs, 3)
        self.assertAlmostEqual(student_v6_config.one_shot.recon_background_weight, 2.0, places=8)
        self.assertAlmostEqual(student_v6_config.one_shot.camera_background_speckle_weight, 0.15, places=8)
        self.assertAlmostEqual(student_v6_config.one_shot.camera_background_hotspot_weight, 2.0, places=8)
        self.assertAlmostEqual(student_v6_config.one_shot.camera_background_hotspot_threshold, 0.05, places=8)
        self.assertAlmostEqual(student_v6_config.one_shot.camera_background_highfreq_weight, 1.0, places=8)
        self.assertEqual(student_v6_config.one_shot.eval_samples_per_class, 5)
        self.assertEqual(student_v6_config.one_shot.eval_preview_per_class, 2)
        # v7.1 clean target fixed config
        self.assertEqual(student_v7_1_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v7_1_config.one_shot.student_label_margin_prototype_source, "mnist")
        self.assertAlmostEqual(student_v7_1_config.one_shot.student_label_margin_weight, 0.12, places=8)
        self.assertAlmostEqual(student_v7_1_config.one_shot.clean_foreground_dice_weight, 0.30, places=8)
        self.assertAlmostEqual(student_v7_1_config.one_shot.clean_foreground_hole_weight, 0.15, places=8)
        self.assertAlmostEqual(student_v7_1_config.one_shot.camera_recon_weight, 0.35, places=8)
        self.assertTrue(student_v7_1_config.one_shot.clean_target_enabled)
        self.assertEqual(student_v7_3_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(
            student_v7_3_config.one_shot.student_pair_contrast_config,
            ((2, 0, 0.05), (2, 3, 0.05), (5, 6, 0.04), (5, 8, 0.03)),
        )
        self.assertAlmostEqual(student_v7_3_config.learning_rate, 3.0e-6, places=10)
        self.assertAlmostEqual(student_v7_3_config.one_shot.phase_learning_rate, 2.0e-4, places=10)
        self.assertEqual(student_v7_3_config.one_shot_epochs, 3)
        self.assertAlmostEqual(student_v7_3_config.one_shot.student_display_speckle_weight, 0.35, places=8)
        self.assertEqual(student_v7_3_config.one_shot.eval_samples_per_class, 20)
        self.assertTrue(student_v7_3_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v7_3_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v7_3_eval_config.one_shot.eval_allow_train_fallback)
        self.assertTrue(student_v7_3_eval_config.one_shot.student_label_audit_enabled)
        self.assertEqual(student_v8_identity_config.one_shot.optical_core_arch, "dense_linear")
        self.assertIsNone(student_v8_identity_config.one_shot.d2nn)
        self.assertEqual(student_v8_identity_config.one_shot.checkpoint_selection_metric, "strict_readability")
        self.assertEqual(student_v8_identity_config.one_shot_epochs, 3)
        self.assertEqual(student_v8_identity_config.one_shot.eval_samples_per_class, 20)
        self.assertTrue(student_v8_identity_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_shallow_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertIsNotNone(student_v8_shallow_config.one_shot.d2nn)
        self.assertEqual(student_v8_shallow_config.one_shot.d2nn.propagation_grid_size, 1536)
        self.assertEqual(student_v8_shallow_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v8_shallow_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v8_shallow_config.one_shot_epochs, 3)
        self.assertEqual(student_v8_identity_eval_config.one_shot.optical_core_arch, "dense_linear")
        self.assertEqual(student_v8_identity_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_identity_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_shallow_eval_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_shallow_eval_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v8_shallow_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_shallow_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_shallow_fair_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_shallow_fair_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v8_shallow_fair_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v8_shallow_fair_config.one_shot.d2nn.propagation_grid_size, 1536)
        self.assertEqual(student_v8_shallow_fair_config.one_shot_epochs, 3)
        self.assertEqual(student_v8_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_direct_config.one_shot_epochs, 5)
        self.assertAlmostEqual(student_v8_direct_config.one_shot.camera_recon_weight, 0.10, places=8)
        self.assertAlmostEqual(student_v8_direct_config.one_shot.clean_foreground_dice_weight, 0.80, places=8)
        self.assertEqual(student_v8_direct_config.one_shot.student_label_margin_focus_label, 5)
        self.assertEqual(student_v8_direct_eval_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_direct_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_1_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_1_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertFalse(student_v8_1_direct_config.one_shot.clean_target_enabled)
        self.assertEqual(student_v8_1_direct_config.one_shot_epochs, 20)
        self.assertAlmostEqual(student_v8_1_direct_config.learning_rate, 1.0e-4, places=10)
        self.assertAlmostEqual(student_v8_1_direct_config.one_shot.camera_recon_weight, 1.0, places=8)
        self.assertAlmostEqual(student_v8_1_direct_config.one_shot.recon_background_weight, 8.0, places=8)
        self.assertEqual(student_v8_1b_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_1b_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_1b_direct_config.one_shot_epochs, 8)
        self.assertAlmostEqual(student_v8_1b_direct_config.learning_rate, 3.0e-5, places=10)
        self.assertAlmostEqual(student_v8_1b_direct_config.one_shot.student_label_margin_target, 0.08, places=8)
        self.assertEqual(student_v8_1b_direct_config.one_shot.student_label_margin_focus_label, 1)
        self.assertAlmostEqual(student_v8_1b_direct_config.one_shot.student_label_margin_focus_weight, 4.0, places=8)
        self.assertEqual(
            student_v8_1b_direct_config.one_shot.student_pair_contrast_config,
            ((1, 8, 0.10), (8, 1, 0.10), (2, 8, 0.06), (4, 0, 0.06), (4, 9, 0.06), (9, 8, 0.06)),
        )
        self.assertEqual(student_v8_2_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_2_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_2_direct_config.one_shot_epochs, 5)
        self.assertAlmostEqual(student_v8_2_direct_config.learning_rate, 1.0e-5, places=10)
        self.assertTrue(student_v8_2_direct_config.one_shot.clean_target_enabled)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.shape_loss_weight, 0.03, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.edge_loss_weight, 0.02, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.camera_topology_weight, 0.01, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.clean_foreground_dice_weight, 0.15, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.clean_foreground_hole_weight, 0.10, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.camera_background_speckle_weight, 0.5, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.camera_background_hotspot_weight, 1.0, places=8)
        self.assertAlmostEqual(student_v8_2_direct_config.one_shot.camera_background_highfreq_weight, 0.25, places=8)
        self.assertEqual(student_v8_2_direct_config.one_shot.student_label_margin_focus_label, 1)
        self.assertEqual(
            student_v8_2_direct_config.one_shot.student_pair_contrast_config,
            student_v8_1b_direct_config.one_shot.student_pair_contrast_config,
        )
        self.assertEqual(student_v8_2c_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_2c_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_2c_direct_config.one_shot_epochs, 3)
        self.assertAlmostEqual(student_v8_2c_direct_config.learning_rate, 5.0e-6, places=11)
        self.assertAlmostEqual(student_v8_2c_direct_config.one_shot.student_label_margin_weight, 0.16, places=8)
        self.assertAlmostEqual(student_v8_2c_direct_config.one_shot.student_label_margin_target, 0.09, places=8)
        self.assertIn((3, 8, 0.08), student_v8_2c_direct_config.one_shot.student_pair_contrast_config)
        self.assertEqual(student_v8_3_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_3_direct_config.one_shot.distill_target_source, "mnist_sample")
        self.assertEqual(student_v8_3_direct_config.one_shot_epochs, 10)
        self.assertAlmostEqual(student_v8_3_direct_config.one_shot.shape_loss_weight, 0.04, places=8)
        self.assertAlmostEqual(student_v8_3_direct_config.one_shot.edge_loss_weight, 0.03, places=8)
        self.assertAlmostEqual(student_v8_3_direct_config.one_shot.clean_foreground_dice_weight, 0.20, places=8)
        self.assertEqual(
            student_v8_3_direct_config.one_shot.student_pair_contrast_config,
            student_v8_2c_direct_config.one_shot.student_pair_contrast_config,
        )
        self.assertEqual(student_v8_1_direct_eval_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_1_direct_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_1_direct_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_1_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_3_direct_eval_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_3_direct_eval_config.one_shot.distill_target_source, "mnist_sample")
        self.assertEqual(student_v8_3_direct_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_3_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_4_direct_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_4_direct_config.one_shot.distill_target_source, "mnist_nearest_teacher_image")
        self.assertEqual(student_v8_4_direct_config.one_shot_epochs, 10)
        self.assertAlmostEqual(student_v8_4_direct_config.learning_rate, 8.0e-6, places=11)
        self.assertAlmostEqual(student_v8_4_direct_config.one_shot.camera_recon_weight, 0.8, places=8)
        self.assertAlmostEqual(student_v8_4_direct_config.one_shot.recon_background_weight, 6.0, places=8)
        self.assertEqual(
            student_v8_4_direct_config.one_shot.student_pair_contrast_config,
            student_v8_2c_direct_config.one_shot.student_pair_contrast_config,
        )
        self.assertEqual(student_v8_4_direct_eval_config.one_shot.optical_core_arch, "camera_direct_clean_oracle")
        self.assertEqual(student_v8_4_direct_eval_config.one_shot.distill_target_source, "mnist_nearest_teacher_image")
        self.assertEqual(student_v8_4_direct_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertFalse(student_v8_4_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_5a_direct_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5a_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5a_direct_config.one_shot.aux_detail_target_source, "none")
        self.assertAlmostEqual(student_v8_5a_direct_config.one_shot.code_residual_scale, 0.02, places=8)
        self.assertAlmostEqual(student_v8_5a_direct_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertEqual(student_v8_5b_direct_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5b_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5b_direct_config.one_shot.aux_detail_target_source, "mnist_nearest_teacher_image")
        self.assertAlmostEqual(student_v8_5b_direct_config.one_shot.aux_detail_edge_weight, 0.01, places=8)
        self.assertAlmostEqual(student_v8_5b_direct_config.one_shot.residual_background_weight, 0.05, places=8)
        self.assertEqual(student_v8_5c_direct_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5c_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5c_direct_config.one_shot.aux_detail_target_source, "mnist_nearest_teacher_image")
        self.assertAlmostEqual(student_v8_5c_direct_config.one_shot.code_residual_scale, 0.08, places=8)
        self.assertAlmostEqual(student_v8_5c_direct_config.one_shot.aux_detail_edge_weight, 0.03, places=8)
        self.assertAlmostEqual(student_v8_5c_direct_config.one_shot.aux_detail_shape_weight, 0.02, places=8)
        self.assertAlmostEqual(student_v8_5c_direct_config.one_shot.residual_l1_weight, 0.005, places=8)
        self.assertAlmostEqual(student_v8_5c_direct_config.one_shot.residual_background_weight, 0.03, places=8)
        self.assertEqual(student_v8_5c_direct_eval_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5c_direct_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5c_direct_eval_config.one_shot.aux_detail_target_source, "mnist_nearest_teacher_image")
        self.assertAlmostEqual(student_v8_5c_direct_eval_config.one_shot.code_residual_scale, 0.08, places=8)
        self.assertFalse(student_v8_5c_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_5d_direct_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5d_direct_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5d_direct_config.one_shot.aux_detail_target_source, "mnist_nearest_teacher_image")
        self.assertAlmostEqual(student_v8_5d_direct_config.one_shot.code_residual_scale, 0.05, places=8)
        self.assertAlmostEqual(student_v8_5d_direct_config.one_shot.residual_background_weight, 0.06, places=8)
        self.assertEqual(student_v8_5d_direct_config.one_shot.aux_detail_foreground_gate_dilation, 15)
        self.assertEqual(student_v8_5d_direct_eval_config.one_shot.generator_e_arch, "template_residual_mlp")
        self.assertEqual(student_v8_5d_direct_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_5d_direct_eval_config.one_shot.aux_detail_target_source, "mnist_nearest_teacher_image")
        self.assertAlmostEqual(student_v8_5d_direct_eval_config.one_shot.code_residual_scale, 0.05, places=8)
        self.assertEqual(student_v8_5d_direct_eval_config.one_shot.aux_detail_foreground_gate_dilation, 15)
        self.assertFalse(student_v8_5d_direct_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_6a_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_6a_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v8_6a_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(student_v8_6a_config.one_shot.d2nn.decoder_layers, 2)
        self.assertEqual(student_v8_6a_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v8_6a_config.one_shot.phase_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v8_6a_config.learning_rate, 2.0e-5, places=10)
        self.assertAlmostEqual(student_v8_6a_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertEqual(student_v8_6a_config.one_shot_epochs, 10)
        self.assertEqual(student_v8_6a_eval_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_6a_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertFalse(student_v8_6a_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_6b_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_6b_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v8_6b_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(student_v8_6b_config.one_shot.d2nn.decoder_layers, 2)
        self.assertEqual(student_v8_6b_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertEqual(student_v8_6b_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v8_6b_config.one_shot.phase_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v8_6b_config.learning_rate, 2.0e-5, places=10)
        self.assertAlmostEqual(student_v8_6b_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertEqual(student_v8_6b_config.one_shot_epochs, 10)
        self.assertEqual(student_v8_6b_eval_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_6b_eval_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertEqual(student_v8_6b_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_6b_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertEqual(student_v8_6b_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertFalse(student_v8_6b_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_7a_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v8_7a_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(student_v8_7a_config.one_shot.d2nn.decoder_layers, 2)
        self.assertEqual(student_v8_7a_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v8_7a_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertEqual(student_v8_7a_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v8_7a_config.one_shot.phase_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v8_7a_config.learning_rate, 2.0e-5, places=10)
        self.assertAlmostEqual(student_v8_7a_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertEqual(student_v8_7a_config.one_shot_epochs, 10)
        self.assertEqual(student_v8_7a2_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a2_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v8_7a2_config.one_shot.d2nn.encoder_layers, 2)
        self.assertEqual(student_v8_7a2_config.one_shot.d2nn.decoder_layers, 2)
        self.assertEqual(student_v8_7a2_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v8_7a2_config.one_shot.d2nn.phase_init, "uniform_pi")
        self.assertAlmostEqual(student_v8_7a2_config.one_shot.d2nn.amplitude_logit_init, 2.0, places=8)
        self.assertEqual(student_v8_7a2_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v8_7a2_config.one_shot.phase_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v8_7a2_config.one_shot.amplitude_learning_rate, 5.0e-3, places=10)
        self.assertAlmostEqual(student_v8_7a2_config.learning_rate, 2.0e-5, places=10)
        self.assertAlmostEqual(student_v8_7a2_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertEqual(student_v8_7a2_config.one_shot_epochs, 10)
        self.assertEqual(student_v8_7a3_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a3_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v8_7a3_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v8_7a3_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v8_7a3_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v8_7a3_config.one_shot.d2nn.phase_init, "small")
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.d2nn.layer_spacing, 0.0005, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.d2nn.amplitude_logit_init, 4.0, places=8)
        self.assertEqual(student_v8_7a3_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.phase_learning_rate, 3.0e-4, places=10)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.amplitude_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.generator_base_lr_scale, 0.0, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.recon_foreground_weight, 6.0, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.recon_background_weight, 12.0, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.shape_loss_weight, 0.06, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.edge_loss_weight, 0.04, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.camera_background_speckle_weight, 1.0, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.camera_background_hotspot_weight, 2.0, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.camera_background_highfreq_weight, 0.5, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.camera_readable_contrast_weight, 0.005, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.clean_foreground_dice_weight, 0.30, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.clean_foreground_hole_weight, 0.25, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.student_label_margin_weight, 0.05, places=8)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.student_label_margin_target, 0.06, places=8)
        self.assertIsNone(student_v8_7a3_config.one_shot.student_label_margin_focus_label)
        self.assertAlmostEqual(student_v8_7a3_config.one_shot.cross_class_pairwise_weight, 0.02, places=8)
        self.assertEqual(student_v8_7a3_config.one_shot_epochs, 10)
        self.assertEqual(student_v9_1_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v9_1_config.one_shot.generator_e_arch, "mlp")
        self.assertEqual(student_v9_1_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v9_1_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v9_1_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v9_1_config.one_shot.d2nn.phase_init, "small")
        self.assertAlmostEqual(student_v9_1_config.one_shot.d2nn.layer_spacing, 0.0005, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.d2nn.amplitude_logit_init, 4.0, places=8)
        self.assertEqual(student_v9_1_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v9_1_config.one_shot.phase_learning_rate, 3.0e-4, places=10)
        self.assertAlmostEqual(student_v9_1_config.one_shot.amplitude_learning_rate, 1.0e-3, places=10)
        self.assertAlmostEqual(student_v9_1_config.one_shot.generator_base_lr_scale, 1.0, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.recon_foreground_weight, 6.0, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.recon_background_weight, 12.0, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.shape_loss_weight, 0.06, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.edge_loss_weight, 0.04, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.camera_background_speckle_weight, 1.0, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.camera_background_hotspot_weight, 2.0, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.camera_background_highfreq_weight, 0.5, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.camera_readable_contrast_weight, 0.005, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.clean_foreground_dice_weight, 0.30, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.clean_foreground_hole_weight, 0.25, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.student_label_margin_weight, 0.05, places=8)
        self.assertAlmostEqual(student_v9_1_config.one_shot.student_label_margin_target, 0.06, places=8)
        self.assertIsNone(student_v9_1_config.one_shot.student_label_margin_focus_label)
        self.assertAlmostEqual(student_v9_1_config.one_shot.cross_class_pairwise_weight, 0.02, places=8)
        self.assertEqual(student_v9_1_config.one_shot_epochs, 10)
        self.assertEqual(student_v9_1_config.one_shot_distill_epoch_checkpoint_interval, 2)
        self.assertEqual(student_v8_7a_eval_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a_eval_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v8_7a_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_7a_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertEqual(student_v8_7a_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertFalse(student_v8_7a_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_7a2_eval_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a2_eval_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertAlmostEqual(student_v8_7a2_eval_config.one_shot.d2nn.amplitude_logit_init, 2.0, places=8)
        self.assertEqual(student_v8_7a2_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_7a2_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertEqual(student_v8_7a2_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertFalse(student_v8_7a2_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_7a3_eval_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v8_7a3_eval_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v8_7a3_eval_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v8_7a3_eval_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v8_7a3_eval_config.one_shot.d2nn.phase_init, "small")
        self.assertAlmostEqual(student_v8_7a3_eval_config.one_shot.d2nn.layer_spacing, 0.0005, places=8)
        self.assertAlmostEqual(student_v8_7a3_eval_config.one_shot.d2nn.amplitude_logit_init, 4.0, places=8)
        self.assertEqual(student_v8_7a3_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertEqual(student_v8_7a3_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertEqual(student_v8_7a3_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertFalse(student_v8_7a3_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v9_1_eval_config.one_shot.optical_core_arch, "phase_amplitude_d2nn")
        self.assertEqual(student_v9_1_eval_config.one_shot.d2nn.modulation, "phase_amplitude")
        self.assertEqual(student_v9_1_eval_config.one_shot.d2nn.encoder_layers, 1)
        self.assertEqual(student_v9_1_eval_config.one_shot.d2nn.decoder_layers, 1)
        self.assertEqual(student_v9_1_eval_config.one_shot.d2nn.phase_init, "small")
        self.assertAlmostEqual(student_v9_1_eval_config.one_shot.d2nn.layer_spacing, 0.0005, places=8)
        self.assertAlmostEqual(student_v9_1_eval_config.one_shot.d2nn.amplitude_logit_init, 4.0, places=8)
        self.assertEqual(student_v9_1_eval_config.one_shot.distill_target_source, "mnist_prototype")
        self.assertAlmostEqual(student_v9_1_eval_config.one_shot.generator_base_lr_scale, 1.0, places=8)
        self.assertEqual(student_v9_1_eval_config.one_shot.eval_samples_per_class, 10)
        self.assertEqual(student_v9_1_eval_config.one_shot.eval_preview_per_class, 2)
        self.assertFalse(student_v9_1_eval_config.one_shot.eval_allow_train_fallback)
        self.assertEqual(student_v8_shallow_fair_eval_config.one_shot.optical_core_arch, "diffractive_phase_only")
        self.assertEqual(student_v8_shallow_fair_eval_config.one_shot.d2nn.encoder_layers, 1)
        self.assertFalse(student_v8_shallow_fair_eval_config.one_shot.eval_allow_train_fallback)
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

    def test_distill_target_source_rejects_unknown_value(self) -> None:
        from ldm_project.config import load_training_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_target_source.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "first_stage:",
                        "  image_size: 4",
                        "one_shot:",
                        "  code_dim: 3",
                        "  num_classes: 2",
                        "  distill_target_source: not_a_target",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "distill_target_source"):
                load_training_config(config_path)

    def test_generator_and_aux_detail_config_reject_unknown_values(self) -> None:
        from ldm_project.config import load_training_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_generator_arch.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  generator_e_arch: not_a_generator",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "generator_e_arch"):
                load_training_config(config_path)

            config_path = Path(tmpdir) / "bad_aux_detail_source.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  aux_detail_target_source: not_a_target",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "aux_detail_target_source"):
                load_training_config(config_path)

            config_path = Path(tmpdir) / "bad_aux_detail_gate.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  aux_detail_foreground_gate_dilation: -1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "aux_detail_foreground_gate_dilation"):
                load_training_config(config_path)

    def test_d2nn_arch_and_modulation_mismatch_is_rejected(self) -> None:
        from ldm_project.config import load_training_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_d2nn_modulation.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  optical_core_arch: phase_amplitude_d2nn",
                        "  d2nn:",
                        "    modulation: phase_only",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "phase_amplitude_d2nn"):
                load_training_config(config_path)

            config_path = Path(tmpdir) / "bad_d2nn_amplitude_init.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  optical_core_arch: phase_amplitude_d2nn",
                        "  d2nn:",
                        "    modulation: phase_amplitude",
                        "    amplitude_logit_init: .inf",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "amplitude_logit_init"):
                load_training_config(config_path)

            config_path = Path(tmpdir) / "bad_amplitude_lr.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  mode: one_shot_distill",
                        "one_shot:",
                        "  amplitude_learning_rate: -0.001",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "amplitude_learning_rate"):
                load_training_config(config_path)

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

    def test_distill_target_source_defaults_to_teacher_camera(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _resolve_distill_targets

        teacher_camera = torch.rand(3, 1, 4, 4)
        payload = {
            "codes": torch.randn(3, 2),
            "class_labels": torch.tensor([0, 1, 0]),
            "teacher_camera": teacher_camera,
        }
        train_cfg = TrainingConfig(
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=2, num_classes=2),
        )

        targets = _resolve_distill_targets(torch, payload, train_cfg)

        self.assertTrue(torch.equal(targets, teacher_camera))

    def test_distill_target_source_mnist_prototype_indexes_normalized_labels(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _resolve_distill_targets

        payload = {
            "codes": torch.randn(4, 2),
            "class_labels": torch.tensor([1, 0, 1, 0]),
            "teacher_camera": torch.zeros(4, 1, 4, 4),
        }
        prototypes = torch.stack(
            [
                torch.linspace(-1.0, 1.0, steps=16).reshape(1, 4, 4),
                torch.linspace(2.0, 6.0, steps=16).reshape(1, 4, 4).flip(-1),
            ],
            dim=0,
        )
        train_cfg = TrainingConfig(
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_prototype"),
        )

        targets = _resolve_distill_targets(torch, payload, train_cfg, mnist_prototypes=prototypes)

        self.assertEqual(tuple(targets.shape), (4, 1, 4, 4))
        self.assertGreaterEqual(float(targets.min().item()), 0.0)
        self.assertLessEqual(float(targets.max().item()), 1.0)
        self.assertTrue(torch.equal(targets[0], targets[2]))
        self.assertTrue(torch.equal(targets[1], targets[3]))
        self.assertFalse(torch.equal(targets[0], targets[1]))

    def test_distill_loader_uses_resolved_mnist_prototype_targets(self) -> None:
        import torch

        from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders

        labels = torch.tensor([0, 1], dtype=torch.long)
        payload = {
            "codes": torch.randn(2, 2),
            "class_labels": labels,
            "teacher_camera": torch.zeros(2, 1, 4, 4),
        }
        prototypes = torch.zeros(2, 1, 4, 4)
        prototypes[0, :, :2, :] = 1.0
        prototypes[1, :, :, :2] = 1.0
        train_cfg = TrainingConfig(
            batch_size=2,
            data=DataConfig(validation_fraction=0.0),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_prototype"),
        )

        train_loader, _, _, _, _ = _build_distill_loaders(
            torch,
            payload,
            train_cfg,
            use_pin_memory=False,
            mnist_prototypes=prototypes,
        )
        _, batch_labels, batch_targets = next(iter(train_loader))

        for index, label in enumerate(batch_labels.tolist()):
            self.assertTrue(torch.equal(batch_targets[index], prototypes[int(label)]))

    def test_distill_loader_keeps_aux_detail_target_separate_from_main_target(self) -> None:
        import torch

        from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders
        from ldm_project.one_shot_losses import _build_clean_readable_target

        labels = torch.tensor([0, 1], dtype=torch.long)
        teacher_camera = torch.zeros(2, 1, 4, 4)
        teacher_camera[:, :, 1:3, 1:3] = 1.0
        payload = {
            "codes": torch.randn(2, 2),
            "class_labels": labels,
            "teacher_camera": teacher_camera,
        }
        prototypes = torch.zeros(2, 1, 4, 4)
        prototypes[0, :, :2, :] = 1.0
        prototypes[1, :, :, :2] = 1.0
        train_cfg = TrainingConfig(
            batch_size=2,
            data=DataConfig(validation_fraction=0.0),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(
                code_dim=2,
                num_classes=2,
                distill_target_source="mnist_prototype",
                aux_detail_target_source="clean_teacher_camera",
            ),
        )

        train_loader, _, _, _, _ = _build_distill_loaders(
            torch,
            payload,
            train_cfg,
            use_pin_memory=False,
            mnist_prototypes=prototypes,
        )
        _, batch_labels, batch_targets, batch_aux_targets = next(iter(train_loader))
        expected_aux = _build_clean_readable_target(teacher_camera)["clean_target"]

        for index, label in enumerate(batch_labels.tolist()):
            self.assertTrue(torch.equal(batch_targets[index], prototypes[int(label)]))
        self.assertTrue(torch.allclose(batch_aux_targets, expected_aux))

    def test_distill_target_source_mnist_sample_is_deterministic_and_per_sample(self) -> None:
        import sys
        import types
        from unittest.mock import patch

        import torch

        from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _resolve_distill_targets

        class FakeMNIST:
            def __init__(self, root, train, download, transform):
                del root, train, download
                self.transform = transform
                self.targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
                self.images = [
                    torch.full((1, 4, 4), 0.10),
                    torch.full((1, 4, 4), 0.20),
                    torch.full((1, 4, 4), 0.30),
                    torch.full((1, 4, 4), 0.70),
                    torch.full((1, 4, 4), 0.80),
                    torch.full((1, 4, 4), 0.90),
                ]

            def __iter__(self):
                for image, label in zip(self.images, self.targets.tolist(), strict=True):
                    yield self.transform(image), label

        class IdentityTransform:
            def __call__(self, image):
                return image

        fake_torchvision = types.ModuleType("torchvision")
        fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
        fake_transforms = types.SimpleNamespace(
            Compose=lambda transforms: IdentityTransform(),
            Resize=lambda size: IdentityTransform(),
            ToTensor=lambda: IdentityTransform(),
        )
        fake_torchvision.datasets = fake_datasets
        fake_torchvision.transforms = fake_transforms

        payload = {
            "codes": torch.randn(4, 2),
            "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "teacher_camera": torch.zeros(4, 1, 4, 4),
            "readability_filter_indices": torch.tensor([0, 1, 3, 4], dtype=torch.long),
        }
        train_cfg = TrainingConfig(
            data=DataConfig(root="unused"),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_sample"),
        )

        with patch.dict(
            sys.modules,
            {
                "torchvision": fake_torchvision,
                "torchvision.datasets": fake_datasets,
                "torchvision.transforms": fake_transforms,
            },
        ):
            targets_a = _resolve_distill_targets(torch, payload, train_cfg)
            targets_b = _resolve_distill_targets(torch, payload, train_cfg)

        self.assertEqual(tuple(targets_a.shape), (4, 1, 4, 4))
        self.assertGreaterEqual(float(targets_a.min().item()), 0.0)
        self.assertLessEqual(float(targets_a.max().item()), 1.0)
        self.assertTrue(torch.equal(targets_a, targets_b))
        self.assertFalse(torch.equal(targets_a[0], targets_a[1]))
        self.assertFalse(torch.equal(targets_a[2], targets_a[3]))
        self.assertAlmostEqual(float(targets_a[0].mean().item()), 0.10, places=6)
        self.assertAlmostEqual(float(targets_a[1].mean().item()), 0.20, places=6)
        self.assertAlmostEqual(float(targets_a[2].mean().item()), 0.70, places=6)
        self.assertAlmostEqual(float(targets_a[3].mean().item()), 0.80, places=6)

    def test_distill_loader_uses_resolved_mnist_sample_targets(self) -> None:
        import sys
        import types
        from unittest.mock import patch

        import torch

        from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders

        class FakeMNIST:
            def __init__(self, root, train, download, transform):
                del root, train, download
                self.transform = transform
                self.targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                self.images = [
                    torch.full((1, 4, 4), 0.15),
                    torch.full((1, 4, 4), 0.25),
                    torch.full((1, 4, 4), 0.75),
                    torch.full((1, 4, 4), 0.85),
                ]

            def __iter__(self):
                for image, label in zip(self.images, self.targets.tolist(), strict=True):
                    yield self.transform(image), label

        class IdentityTransform:
            def __call__(self, image):
                return image

        fake_torchvision = types.ModuleType("torchvision")
        fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
        fake_transforms = types.SimpleNamespace(
            Compose=lambda transforms: IdentityTransform(),
            Resize=lambda size: IdentityTransform(),
            ToTensor=lambda: IdentityTransform(),
        )
        fake_torchvision.datasets = fake_datasets
        fake_torchvision.transforms = fake_transforms

        payload = {
            "codes": torch.randn(4, 2),
            "class_labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            "teacher_camera": torch.zeros(4, 1, 4, 4),
            "readability_filter_indices": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        }
        train_cfg = TrainingConfig(
            batch_size=4,
            data=DataConfig(root="unused", validation_fraction=0.0),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_sample"),
        )

        with patch.dict(
            sys.modules,
            {
                "torchvision": fake_torchvision,
                "torchvision.datasets": fake_datasets,
                "torchvision.transforms": fake_transforms,
            },
        ):
            train_loader, _, _, _, _ = _build_distill_loaders(
                torch,
                payload,
                train_cfg,
                use_pin_memory=False,
            )
            _, batch_labels, batch_targets = next(iter(train_loader))

        means_by_label: dict[int, list[float]] = {0: [], 1: []}
        for label, target in zip(batch_labels.tolist(), batch_targets, strict=True):
            means_by_label[int(label)].append(round(float(target.mean().item()), 6))
            self.assertFalse(torch.equal(target, torch.zeros_like(target)))
        self.assertEqual(sorted(means_by_label[0]), [0.15, 0.25])
        self.assertEqual(sorted(means_by_label[1]), [0.75, 0.85])

    def test_distill_target_source_mnist_nearest_teacher_image_selects_aligned_targets(self) -> None:
        import sys
        import types
        from unittest.mock import patch

        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _resolve_distill_targets

        horizontal = torch.zeros(1, 4, 4)
        horizontal[:, :2, :] = 1.0
        vertical = torch.zeros(1, 4, 4)
        vertical[:, :, :2] = 1.0
        diagonal = torch.eye(4).view(1, 4, 4)
        anti_diagonal = torch.flip(diagonal, dims=(-1,))

        class FakeMNIST:
            def __init__(self, root, train, download, transform):
                del root, train, download
                self.transform = transform
                self.targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                self.images = [horizontal, vertical, diagonal, anti_diagonal]

            def __iter__(self):
                for image, label in zip(self.images, self.targets.tolist(), strict=True):
                    yield self.transform(image), label

        class IdentityTransform:
            def __call__(self, image):
                return image

        fake_torchvision = types.ModuleType("torchvision")
        fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
        fake_transforms = types.SimpleNamespace(
            Compose=lambda transforms: IdentityTransform(),
            Resize=lambda size: IdentityTransform(),
            ToTensor=lambda: IdentityTransform(),
        )
        fake_torchvision.datasets = fake_datasets
        fake_torchvision.transforms = fake_transforms

        payload = {
            "codes": torch.randn(4, 2),
            "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "teacher_camera": torch.zeros(4, 1, 4, 4),
            "teacher_image": torch.stack([horizontal * 2.0 - 0.5, vertical * 3.0 - 1.0, diagonal, anti_diagonal], dim=0),
        }
        train_cfg = TrainingConfig(
            data=DataConfig(root="unused"),
            distillation=DistillationConfig(label_audit_prototypes_per_class=2),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(
                code_dim=2,
                num_classes=2,
                distill_target_source="mnist_nearest_teacher_image",
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "torchvision": fake_torchvision,
                "torchvision.datasets": fake_datasets,
                "torchvision.transforms": fake_transforms,
            },
        ):
            targets_a = _resolve_distill_targets(torch, payload, train_cfg)
            targets_b = _resolve_distill_targets(torch, payload, train_cfg)

        self.assertEqual(tuple(targets_a.shape), (4, 1, 4, 4))
        self.assertGreaterEqual(float(targets_a.min().item()), 0.0)
        self.assertLessEqual(float(targets_a.max().item()), 1.0)
        self.assertTrue(torch.equal(targets_a, targets_b))
        self.assertTrue(torch.equal(targets_a[0], horizontal))
        self.assertTrue(torch.equal(targets_a[1], vertical))
        self.assertTrue(torch.equal(targets_a[2], diagonal))
        self.assertTrue(torch.equal(targets_a[3], anti_diagonal))

    def test_distill_target_source_mnist_nearest_teacher_image_requires_teacher_image(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _resolve_distill_targets

        payload = {
            "codes": torch.randn(2, 2),
            "class_labels": torch.tensor([0, 1], dtype=torch.long),
            "teacher_camera": torch.zeros(2, 1, 4, 4),
        }
        train_cfg = TrainingConfig(
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(
                code_dim=2,
                num_classes=2,
                distill_target_source="mnist_nearest_teacher_image",
            ),
        )

        with self.assertRaisesRegex(ValueError, "teacher_image"):
            _resolve_distill_targets(torch, payload, train_cfg)

    def test_distill_loader_uses_resolved_mnist_nearest_teacher_image_targets(self) -> None:
        import sys
        import types
        from unittest.mock import patch

        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _build_distill_loaders

        horizontal = torch.zeros(1, 4, 4)
        horizontal[:, :2, :] = 1.0
        vertical = torch.zeros(1, 4, 4)
        vertical[:, :, :2] = 1.0

        class FakeMNIST:
            def __init__(self, root, train, download, transform):
                del root, train, download
                self.transform = transform
                self.targets = torch.tensor([0, 0], dtype=torch.long)
                self.images = [horizontal, vertical]

            def __iter__(self):
                for image, label in zip(self.images, self.targets.tolist(), strict=True):
                    yield self.transform(image), label

        class IdentityTransform:
            def __call__(self, image):
                return image

        fake_torchvision = types.ModuleType("torchvision")
        fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
        fake_transforms = types.SimpleNamespace(
            Compose=lambda transforms: IdentityTransform(),
            Resize=lambda size: IdentityTransform(),
            ToTensor=lambda: IdentityTransform(),
        )
        fake_torchvision.datasets = fake_datasets
        fake_torchvision.transforms = fake_transforms

        payload = {
            "codes": torch.randn(2, 2),
            "class_labels": torch.tensor([0, 0], dtype=torch.long),
            "teacher_camera": torch.zeros(2, 1, 4, 4),
            "teacher_image": torch.stack([horizontal, vertical], dim=0),
        }
        train_cfg = TrainingConfig(
            batch_size=2,
            data=DataConfig(root="unused", validation_fraction=0.0),
            distillation=DistillationConfig(label_audit_prototypes_per_class=2),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(
                code_dim=2,
                num_classes=1,
                distill_target_source="mnist_nearest_teacher_image",
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "torchvision": fake_torchvision,
                "torchvision.datasets": fake_datasets,
                "torchvision.transforms": fake_transforms,
            },
        ):
            train_loader, _, _, _, _ = _build_distill_loaders(
                torch,
                payload,
                train_cfg,
                use_pin_memory=False,
            )
            _, _, batch_targets = next(iter(train_loader))

        target_sums = sorted(float(target.sum().item()) for target in batch_targets)
        self.assertEqual(target_sums, [8.0, 8.0])
        self.assertFalse(torch.equal(batch_targets[0], batch_targets[1]))

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
            _one_shot_distill_strict_readability_score,
            _one_shot_distill_strict_balanced_quality_score,
            _resolve_one_shot_distill_selection,
        )
        from ldm_project.stage_constants import (
            ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC,
            ONE_SHOT_DISTILL_BEST_METRIC,
            ONE_SHOT_DISTILL_QUALITY_METRIC,
            ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC,
            ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC,
        )

        one_shot = type("OneShotSelectionConfig", (), {"checkpoint_selection_metric": "quality"})()
        train_cfg = type("TrainingSelectionConfig", (), {"one_shot": one_shot})()
        metrics = {
            "loss": 0.5,
            "camera_recon_loss": 0.1,
            "camera_foreground_dice": 0.8,
            "camera_saturation_ratio": 0.4,
            "foreground_hole_ratio": 0.25,
            "background_speckle_ratio": 0.20,
            "background_hotspot_ratio": 0.35,
            "background_highfreq_energy": 0.004,
            "readable_contrast": 4.2,
            "slm_tv": 0.2,
            "diversity_ratio": 0.30,
            "student_label_correct_ratio": 0.75,
            "student_label_margin_mean": 0.01,
            "paired_dice": 0.80,
            "paired_corr": 0.85,
            "camera_high_saturation_ratio": 0.02,
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

        one_shot.checkpoint_selection_metric = "strict_readability"
        score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

        self.assertEqual(metric, ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC)
        self.assertAlmostEqual(score, 0.1425, places=6)
        self.assertAlmostEqual(metrics["strict_readability_score"], 0.1425, places=6)
        self.assertAlmostEqual(_one_shot_distill_strict_readability_score(metrics), 0.1425, places=6)
        for required_metric in ("paired_corr", "paired_dice", "camera_high_saturation_ratio"):
            incomplete = dict(metrics)
            incomplete.pop(required_metric)
            with self.assertRaisesRegex(ValueError, required_metric):
                _one_shot_distill_strict_readability_score(incomplete)

    def test_one_shot_saturation_split_not_penalizing_black_background(self) -> None:
        from ldm_project.training_loss_common import (
            _camera_low_saturation_ratio,
            _camera_high_saturation_ratio,
            _camera_saturation_ratio,
        )
        from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score
        import torch

        # Pure black background: all pixels at 0.0 → low saturation high, high saturation zero.
        black = torch.zeros(1, 1, 200, 200)
        self.assertAlmostEqual(float(_camera_low_saturation_ratio(black).item()), 1.0, places=6)
        self.assertAlmostEqual(float(_camera_high_saturation_ratio(black).item()), 0.0, places=6)
        self.assertAlmostEqual(float(_camera_saturation_ratio(black).item()), 1.0, places=6)

        # Pure white foreground: all pixels at 1.0 → high saturation high, low saturation zero.
        white = torch.ones(1, 1, 200, 200)
        self.assertAlmostEqual(float(_camera_low_saturation_ratio(white).item()), 0.0, places=6)
        self.assertAlmostEqual(float(_camera_high_saturation_ratio(white).item()), 1.0, places=6)

        # Strict readability score should NOT penalize black background (low sat only).
        base_metrics = {
            "camera_recon_loss": 0.1,
            "camera_foreground_dice": 0.8,
            "paired_dice": 0.80,
            "paired_corr": 0.85,
            "camera_saturation_ratio": 1.0,
            "camera_low_saturation_ratio": 1.0,
            "camera_high_saturation_ratio": 0.0,
            "foreground_hole_ratio": 0.06,
            "background_speckle_ratio": 0.07,
            "background_hotspot_ratio": 0.12,
            "background_highfreq_energy": 0.002,
            "readable_contrast": 4.5,
            "slm_tv": 0.01,
            "student_label_correct_ratio": 0.90,
            "student_label_margin_mean": 0.03,
        }
        score_black = _one_shot_distill_strict_readability_score(base_metrics)
        # Identical metrics but low_saturation=0 → same score because high_saturation=0 in both.
        low_sat_only = dict(base_metrics, camera_low_saturation_ratio=0.0)
        score_no_low_sat = _one_shot_distill_strict_readability_score(low_sat_only)
        self.assertAlmostEqual(score_black, score_no_low_sat, places=6,
                               msg="strict_readability should not distinguish low saturation levels (black bg is valid).")

        # High saturation increases strict_readability score (worse).
        high_sat_metrics = dict(base_metrics, camera_high_saturation_ratio=0.50)
        score_high_sat = _one_shot_distill_strict_readability_score(high_sat_metrics)
        self.assertGreater(score_high_sat, score_black,
                           msg="high saturation should worsen (increase) strict_readability score.")

    def test_one_shot_strict_selection_prefers_better_paired_corr_dice(self) -> None:
        from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

        base = {
            "camera_recon_loss": 0.1,
            "camera_foreground_dice": 0.8,
            "paired_dice": 0.72,
            "paired_corr": 0.78,
            "camera_saturation_ratio": 0.02,
            "camera_low_saturation_ratio": 0.01,
            "camera_high_saturation_ratio": 0.01,
            "foreground_hole_ratio": 0.06,
            "background_speckle_ratio": 0.07,
            "background_hotspot_ratio": 0.12,
            "background_highfreq_energy": 0.002,
            "readable_contrast": 4.5,
            "slm_tv": 0.01,
            "student_label_correct_ratio": 0.90,
            "student_label_margin_mean": 0.03,
        }
        score_ok = _one_shot_distill_strict_readability_score(base)
        # Worse paired_dice → higher score (worse).
        bad_dice = dict(base, paired_dice=0.50)
        score_bad_dice = _one_shot_distill_strict_readability_score(bad_dice)
        self.assertGreater(score_bad_dice, score_ok,
                           msg="lower paired_dice must worsen (increase) strict_readability score.")
        # Worse paired_corr → higher score (worse).
        bad_corr = dict(base, paired_corr=0.50)
        score_bad_corr = _one_shot_distill_strict_readability_score(bad_corr)
        self.assertGreater(score_bad_corr, score_ok,
                           msg="lower paired_corr must worsen (increase) strict_readability score.")

    def test_one_shot_foreground_dice_not_used_in_strict_selection(self) -> None:
        from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

        base = {
            "camera_recon_loss": 0.1,
            "camera_foreground_dice": 0.50,  # poor, but should NOT affect strict score
            "paired_dice": 0.80,
            "paired_corr": 0.85,
            "camera_saturation_ratio": 0.02,
            "camera_low_saturation_ratio": 0.01,
            "camera_high_saturation_ratio": 0.01,
            "foreground_hole_ratio": 0.06,
            "background_speckle_ratio": 0.07,
            "background_hotspot_ratio": 0.12,
            "background_highfreq_energy": 0.002,
            "readable_contrast": 4.5,
            "slm_tv": 0.01,
            "student_label_correct_ratio": 0.90,
            "student_label_margin_mean": 0.03,
        }
        score_poor_fg_dice = _one_shot_distill_strict_readability_score(base)
        good_fg_dice = dict(base, camera_foreground_dice=0.95)
        score_good_fg_dice = _one_shot_distill_strict_readability_score(good_fg_dice)
        # camera_foreground_dice is no longer in the formula → scores must be identical.
        self.assertAlmostEqual(score_poor_fg_dice, score_good_fg_dice, places=6,
                               msg="camera_foreground_dice should NOT affect strict_readability score.")

        diagnostics_changed = dict(
            base,
            student_label_correct_ratio=0.0,
            student_label_margin_mean=-1.0,
            foreground_hole_ratio=1.0,
            background_speckle_ratio=1.0,
            readable_contrast=0.0,
        )
        self.assertAlmostEqual(
            score_poor_fg_dice,
            _one_shot_distill_strict_readability_score(diagnostics_changed),
            places=6,
        )

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

    def test_teacher_distill_label_audit_and_strict_selection_use_requested_label(self) -> None:
        import torch

        from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
        from ldm_project.teacher_distill import (
            _build_label_prototype_bank_from_tensors,
            _select_teacher_distill_label_strict_indices,
            compute_teacher_distill_label_audit,
        )

        proto0 = torch.zeros(1, 4, 4)
        proto0[:, :, :2] = 1.0
        proto1 = torch.zeros(1, 4, 4)
        proto1[:, :, 2:] = 1.0
        prototype_images = torch.stack([proto0, proto0, proto1, proto1], dim=0)
        prototype_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        prototypes = _build_label_prototype_bank_from_tensors(
            torch,
            prototype_images,
            prototype_labels,
            class_labels=(0, 1),
            image_size=4,
        )

        teacher_camera = torch.stack([proto0, proto0 * 0.9, proto1, proto1, proto1, proto0], dim=0)
        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        audit = compute_teacher_distill_label_audit(
            torch,
            teacher_camera,
            labels,
            prototypes,
            class_labels=(0, 1),
            foreground_threshold=0.2,
        )
        self.assertEqual(tuple(prototypes.shape), (2, 1, 4, 4))
        self.assertEqual(audit["nearest_label"].tolist(), [0, 0, 1, 1, 1, 0])
        self.assertTrue(audit["label_correct"][0].item())
        self.assertFalse(audit["label_correct"][2].item())
        self.assertFalse(audit["label_correct"][5].item())

        train_cfg = TrainingConfig(
            one_shot=OneShotConfig(num_classes=2),
            distillation=DistillationConfig(class_labels=(0, 1), label_strict_keep_per_class=2),
        )
        selected, summary = _select_teacher_distill_label_strict_indices(torch, labels, audit, train_cfg)
        self.assertEqual(len(selected), 4)
        self.assertEqual((labels[torch.tensor(selected)] == 0).sum().item(), 2)
        self.assertEqual((labels[torch.tensor(selected)] == 1).sum().item(), 2)
        self.assertFalse(summary["per_class"]["0"]["relaxed_label_filter"])
        self.assertFalse(summary["per_class"]["1"]["relaxed_label_filter"])
        self.assertEqual(summary["per_class"]["0"]["selected_label_correct"], 2)
        self.assertEqual(summary["per_class"]["1"]["selected_label_correct"], 2)

        relaxed_cfg = TrainingConfig(
            one_shot=OneShotConfig(num_classes=2),
            distillation=DistillationConfig(class_labels=(0, 1), label_strict_keep_per_class=3),
        )
        relaxed_selected, relaxed_summary = _select_teacher_distill_label_strict_indices(
            torch,
            labels,
            audit,
            relaxed_cfg,
        )
        self.assertEqual(len(relaxed_selected), 6)
        self.assertTrue(relaxed_summary["per_class"]["0"]["relaxed_label_filter"])
        self.assertTrue(relaxed_summary["per_class"]["1"]["relaxed_label_filter"])

    def test_teacher_distill_label_audit_batched_matches_full_audit(self) -> None:
        import torch

        from ldm_project.teacher_distill import (
            compute_teacher_distill_label_audit,
            compute_teacher_distill_label_audit_batched,
        )

        proto0 = torch.zeros(1, 4, 4)
        proto0[:, :, :2] = 1.0
        proto1 = torch.zeros(1, 4, 4)
        proto1[:, :, 2:] = 1.0
        prototypes = torch.stack([proto0, proto1], dim=0)
        teacher_camera = torch.stack([proto0, proto1, proto0 * 0.8, proto1 * 0.7, proto0], dim=0)
        labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

        full = compute_teacher_distill_label_audit(
            torch,
            teacher_camera,
            labels,
            prototypes,
            class_labels=(0, 1),
            foreground_threshold=0.2,
        )
        batched = compute_teacher_distill_label_audit_batched(
            torch,
            teacher_camera,
            labels,
            prototypes,
            class_labels=(0, 1),
            batch_size=2,
            device=torch.device("cpu"),
            foreground_threshold=0.2,
        )

        self.assertEqual(set(batched), set(full))
        self.assertEqual(batched["nearest_label"].tolist(), full["nearest_label"].tolist())
        self.assertEqual(batched["label_correct"].tolist(), full["label_correct"].tolist())
        torch.testing.assert_close(batched["label_margin"], full["label_margin"])
        torch.testing.assert_close(batched["teacher_readable"], full["teacher_readable"])

    def test_teacher_distill_label_strict_relaxed_filter_preserves_all_correct_samples(self) -> None:
        import torch

        from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
        from ldm_project.teacher_distill import _select_teacher_distill_label_strict_indices

        labels = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
        audit = {
            "label_correct": torch.tensor([True, True, False, False, False]),
            "label_margin": torch.tensor([0.01, 0.02, 10.0, 5.0, 1.0]),
            "requested_label_corr": torch.tensor([0.7, 0.8, 0.95, 0.9, 0.85]),
            "background_speckle_ratio": torch.zeros(5),
            "readable_contrast": torch.zeros(5),
            "foreground_hole_ratio": torch.zeros(5),
        }
        train_cfg = TrainingConfig(
            one_shot=OneShotConfig(num_classes=1),
            distillation=DistillationConfig(class_labels=(0,), label_strict_keep_per_class=3),
        )

        selected, summary = _select_teacher_distill_label_strict_indices(torch, labels, audit, train_cfg)

        self.assertEqual(selected, [0, 1, 2])
        self.assertTrue(summary["per_class"]["0"]["relaxed_label_filter"])
        self.assertEqual(summary["per_class"]["0"]["label_correct_available"], 2)
        self.assertEqual(summary["per_class"]["0"]["selected_label_correct"], 2)
        self.assertEqual(summary["per_class"]["0"]["kept"], 3)

    def test_teacher_distill_label_strict_filter_writes_balanced_bundle_and_audit(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.teacher_distill import run_teacher_distill_label_strict_filter

        proto0 = torch.zeros(1, 4, 4)
        proto0[:, :, :2] = 1.0
        proto1 = torch.zeros(1, 4, 4)
        proto1[:, :, 2:] = 1.0
        prototypes = torch.stack([proto0, proto1], dim=0)

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.meta_payload = None
                self.bundle_path = None
                self.artifact_payloads = {}
                self.config_payloads = {}
                self.grid_calls = []

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

            def save_config_snapshot(self, name, data):
                self.config_payloads[name] = dict(data)
                return self.configs_dir / f"{name}.json"

            def save_tensor_bundle(self, name, payload):
                path = self.artifacts_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(payload), path)
                self.bundle_path = path
                return path

            def save_artifact_json(self, name, payload):
                self.artifact_payloads[name] = payload
                path = self.artifacts_dir / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

            def save_tensor_grid(
                self,
                tensor_map,
                destination,
                title="",
                max_items=4,
                row_labels=None,
                cmap="gray",
                phase_colorbar=False,
            ):
                self.grid_calls.append(
                    (
                        Path(destination).name,
                        list(tensor_map.keys()),
                        {
                            "row_labels": row_labels,
                            "cmap": cmap,
                            "phase_colorbar": phase_colorbar,
                        },
                    )
                )
                path = Path(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            source_path = run_dir / "source_distill.pt"
            labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
            teacher_camera = torch.stack([proto0, proto0, proto1, proto1, proto1, proto0], dim=0)
            torch.save(
                {
                    "codes": torch.arange(18, dtype=torch.float32).reshape(6, 3),
                    "class_labels": labels,
                    "teacher_camera": teacher_camera,
                    "teacher_image": teacher_camera.clone(),
                },
                source_path,
            )
            train_cfg = TrainingConfig(
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(
                    class_labels=(0, 1),
                    label_strict_keep_per_class=2,
                    label_audit_prototypes_per_class=2,
                ),
            )
            run_manager = RecordingRunManager(run_dir)
            with unittest.mock.patch(
                "ldm_project.teacher_distill._build_mnist_label_prototype_bank",
                return_value=prototypes,
            ):
                run_teacher_distill_label_strict_filter(
                    torch,
                    train_cfg=train_cfg,
                    run_manager=run_manager,
                    run_meta={},
                    distill_dataset=str(source_path),
                )

            filtered = torch.load(run_manager.bundle_path, map_location="cpu", weights_only=False)
            self.assertEqual(run_manager.bundle_path.name, "teacher_distill_dataset_label_strict.pt")
            self.assertEqual(tuple(filtered["codes"].shape), (4, 3))
            self.assertEqual((filtered["class_labels"] == 0).sum().item(), 2)
            self.assertEqual((filtered["class_labels"] == 1).sum().item(), 2)
            self.assertEqual(filtered["config"]["filter"]["teacher_filter"], "label_strict")
            self.assertIn("label_audit_nearest_label", filtered)
            metrics = run_manager.artifact_payloads["teacher_distill_label_strict_filter_metrics"]
            self.assertEqual(metrics["teacher_filter"], "label_strict")
            self.assertEqual(metrics["class_counts"], {"0": 2, "1": 2})
            self.assertIn("label_margin", metrics["per_class"]["0"])
            audit = run_manager.artifact_payloads["teacher_distill_label_audit"]
            self.assertIn("per_sample", audit)
            self.assertIn("nearest_label", audit["per_sample"][0])
            self.assertEqual(run_manager.meta_payload["teacher_distill_filter"], "label_strict")
            self.assertIn("teacher_distill_label_audit.png", [call[0] for call in run_manager.grid_calls])

    def test_teacher_distill_readable_split_requires_readable_targets(self) -> None:
        import torch

        from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
        from ldm_project.teacher_distill import _split_readable_teacher_indices

        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
        audit = {
            "label_correct": torch.tensor([True, True, True, True, False, True, True, True, True, False]),
            "label_margin": torch.tensor([0.04, 0.05, 0.06, 0.07, 0.90, 0.04, 0.05, 0.06, 0.07, 0.90]),
            "requested_label_corr": torch.ones(10),
            "nearest_label": labels.clone(),
            "background_speckle_ratio": torch.tensor([0.10, 0.12, 0.11, 0.13, 0.01, 0.10, 0.12, 0.11, 0.13, 0.01]),
            "readable_contrast": torch.tensor([0.80, 0.81, 0.82, 0.83, 0.95, 0.80, 0.81, 0.82, 0.83, 0.95]),
            "foreground_hole_ratio": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.01, 0.01, 0.02, 0.03, 0.04, 0.01]),
        }
        train_cfg = TrainingConfig(
            one_shot=OneShotConfig(num_classes=2),
            distillation=DistillationConfig(
                class_labels=(0, 1),
                readable_train_per_class=2,
                readable_val_per_class=1,
                readable_test_per_class=1,
                readable_split_seed=7,
            ),
        )

        summary = _split_readable_teacher_indices(torch, labels, audit, train_cfg)

        self.assertFalse(summary["failed"])
        self.assertEqual(len(summary["indices"]["train"]), 4)
        self.assertEqual(len(summary["indices"]["val"]), 2)
        self.assertEqual(len(summary["indices"]["test"]), 2)
        self.assertEqual(summary["per_class"]["0"]["readable_available"], 4)
        self.assertEqual(summary["per_class"]["1"]["readable_available"], 4)
        all_indices = (
            summary["indices"]["train"] + summary["indices"]["val"] + summary["indices"]["test"]
        )
        self.assertEqual(len(all_indices), len(set(all_indices)))

        fail_cfg = TrainingConfig(
            one_shot=OneShotConfig(num_classes=2),
            distillation=DistillationConfig(
                class_labels=(0, 1),
                readable_train_per_class=3,
                readable_val_per_class=1,
                readable_test_per_class=1,
            ),
        )
        failed = _split_readable_teacher_indices(torch, labels, audit, fail_cfg)
        self.assertTrue(failed["failed"])
        self.assertTrue(failed["per_class"]["0"]["readability_filter_failed"])
        self.assertEqual(failed["indices"]["train"], [])

    def test_teacher_distill_readable_split_writes_train_val_test_bundles(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.teacher_distill import run_teacher_distill_readable_split

        proto0 = torch.zeros(1, 4, 4)
        proto0[:, :, :2] = 1.0
        proto1 = torch.zeros(1, 4, 4)
        proto1[:, :, 2:] = 1.0
        prototypes = torch.stack([proto0, proto1], dim=0)

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.images_dir = run_dir / "images"
                self.artifacts_dir = run_dir / "artifacts"
                self.configs_dir = run_dir / "configs"
                self.meta_payload = None
                self.bundle_paths = {}
                self.artifact_payloads = {}
                self.config_payloads = {}
                self.grid_calls = []

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

            def save_config_snapshot(self, name, data):
                self.config_payloads[name] = dict(data)
                return self.configs_dir / f"{name}.json"

            def save_tensor_bundle(self, name, payload):
                path = self.artifacts_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(payload), path)
                self.bundle_paths[name] = path
                return path

            def save_artifact_json(self, name, payload):
                self.artifact_payloads[name] = payload
                path = self.artifacts_dir / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

            def save_tensor_grid(
                self,
                tensor_map,
                destination,
                title="",
                max_items=4,
                row_labels=None,
                cmap="gray",
                phase_colorbar=False,
            ):
                self.grid_calls.append((Path(destination).name, list(tensor_map.keys()), row_labels))
                path = Path(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            source_path = run_dir / "source_distill.pt"
            labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
            teacher_camera = torch.stack([proto0, proto0, proto0, proto0, proto1, proto1, proto1, proto1], dim=0)
            torch.save(
                {
                    "codes": torch.arange(24, dtype=torch.float32).reshape(8, 3),
                    "class_labels": labels,
                    "teacher_camera": teacher_camera,
                    "teacher_image": teacher_camera.clone(),
                    "config": {"source": "unit"},
                },
                source_path,
            )
            train_cfg = TrainingConfig(
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(
                    class_labels=(0, 1),
                    label_audit_prototypes_per_class=2,
                    readable_min_label_margin=-1.0,
                    readable_min_contrast=0.0,
                    readable_max_background_speckle=1.0,
                    readable_max_foreground_hole=1.0,
                    readable_train_per_class=2,
                    readable_val_per_class=1,
                    readable_test_per_class=1,
                    readable_split_seed=11,
                ),
            )
            run_manager = RecordingRunManager(run_dir)
            with unittest.mock.patch(
                "ldm_project.teacher_distill._build_mnist_label_prototype_bank",
                return_value=prototypes,
            ):
                run_teacher_distill_readable_split(
                    torch,
                    train_cfg=train_cfg,
                    run_manager=run_manager,
                    run_meta={},
                    distill_dataset=str(source_path),
                    device=torch.device("cpu"),
                )

            self.assertFalse(run_manager.artifact_payloads["teacher_distill_readable_split_metrics"]["readability_filter_failed"])
            self.assertEqual(set(run_manager.bundle_paths), {
                "teacher_distill_dataset_readable_train.pt",
                "teacher_distill_dataset_readable_val.pt",
                "teacher_distill_dataset_readable_test.pt",
            })
            train_payload = torch.load(
                run_manager.bundle_paths["teacher_distill_dataset_readable_train.pt"],
                map_location="cpu",
                weights_only=False,
            )
            val_payload = torch.load(
                run_manager.bundle_paths["teacher_distill_dataset_readable_val.pt"],
                map_location="cpu",
                weights_only=False,
            )
            test_payload = torch.load(
                run_manager.bundle_paths["teacher_distill_dataset_readable_test.pt"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(tuple(train_payload["codes"].shape), (4, 3))
            self.assertEqual(tuple(val_payload["codes"].shape), (2, 3))
            self.assertEqual(tuple(test_payload["codes"].shape), (2, 3))
            self.assertEqual(train_payload["config"]["filter"]["teacher_filter"], "readability_strict_split")
            self.assertIn("label_audit_margin", train_payload)
            split_indices = [
                *train_payload["readability_filter_indices"].tolist(),
                *val_payload["readability_filter_indices"].tolist(),
                *test_payload["readability_filter_indices"].tolist(),
            ]
            self.assertEqual(len(split_indices), len(set(split_indices)))
            self.assertEqual(run_manager.meta_payload["teacher_distill_readability_filter_failed"], False)
            self.assertIn("teacher_distill_readable_split_before_after.png", [call[0] for call in run_manager.grid_calls])

    def test_prepare_distill_loaders_uses_external_validation_dataset(self) -> None:
        import torch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_training import _prepare_distill_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.pt"
            val_path = root / "val.pt"
            torch.save(
                {
                    "codes": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                    "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                    "teacher_camera": torch.rand(4, 1, 4, 4),
                },
                train_path,
            )
            torch.save(
                {
                    "codes": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    "class_labels": torch.tensor([0, 1], dtype=torch.long),
                    "teacher_camera": torch.rand(2, 1, 4, 4),
                },
                val_path,
            )
            train_cfg = TrainingConfig(
                batch_size=2,
                data=DataConfig(validation_fraction=0.5),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(),
            )

            loaders = _prepare_distill_loaders(
                torch,
                distill_dataset=str(train_path),
                distill_validation_dataset=str(val_path),
                train_cfg=train_cfg,
                use_pin_memory=False,
            )

            self.assertEqual(loaders.train_size, 4)
            self.assertEqual(loaders.validation_size, 2)
            self.assertEqual(loaders.total_size, 6)
            self.assertEqual(loaders.validation_source, "external_dataset")
            self.assertEqual(loaders.validation_dataset_path, str(val_path))

    def test_student_label_margin_prototypes_can_use_mnist_source(self) -> None:
        import torch
        from unittest.mock import patch

        from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_training import _build_student_label_margin_prototypes

        prototype_bank = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
        payload = {
            "class_labels": torch.tensor([0, 1], dtype=torch.long),
            "teacher_camera": torch.zeros(2, 1, 4, 4),
        }
        train_cfg = TrainingConfig(
            data=DataConfig(root="data"),
            first_stage=FirstStageConfig(image_size=4),
            one_shot=OneShotConfig(
                num_classes=2,
                student_label_margin_weight=0.1,
                student_label_margin_prototype_source="mnist",
            ),
            distillation=DistillationConfig(label_audit_prototypes_per_class=3),
        )

        with patch(
            "ldm_project.one_shot_distill_training._build_mnist_label_prototype_bank",
            return_value=prototype_bank,
        ) as build_bank:
            prototypes, source = _build_student_label_margin_prototypes(torch, payload, train_cfg)

        self.assertEqual(source, "mnist")
        self.assertTrue(torch.equal(prototypes, prototype_bank))
        build_bank.assert_called_once()
        _, kwargs = build_bank.call_args
        self.assertEqual(kwargs["class_labels"], (0, 1))
        self.assertEqual(kwargs["image_size"], 4)
        self.assertEqual(kwargs["max_per_class"], 3)

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
            sorted(tensors.keys()),
            sorted(["amplitude", "intensity", "phase"]),
        )
        self.assertEqual(
            list(tensors["amplitude"].keys()),
            [
                "SLM command (amplitude control)",
                "SLM out |E|",
                "encoder in |E|",
                "encoder grid in |E|",
                "encoder field after L1 |E|",
                "encoder field after L2 |E|",
                "encoder out |E|",
                "OLS in |E|",
                "OLS out |E|",
                "decoder in |E|",
                "decoder field after L1 |E|",
                "decoder field after L2 |E|",
                "decoder out |E|",
                "chip |E|",
            ],
        )
        # Intensity: |E|² for chain, camera/display for sensor/reference/SLM command.
        self.assertEqual(
            list(tensors["intensity"].keys()),
            [
                "noise code",
                "reference",
                "SLM command (amplitude control)",
                "SLM amplitude grid (control)",
                "SLM out I",
                "encoder in I",
                "encoder grid in I",
                "encoder out I",
                "ols in I",
                "ols out I",
                "decoder in I",
                "decoder out I",
                "chip I",
                "camera intensity",
            ],
        )
        self.assertEqual(
            list(tensors["phase"].keys()),
            [
                "SLM out phase",
                "encoder in phase",
                "encoder grid in phase",
                "encoder field after L1 phase",
                "encoder field after L2 phase",
                "encoder out phase",
                "OLS in phase",
                "OLS out phase",
                "decoder in phase",
                "decoder field after L1 phase",
                "decoder field after L2 phase",
                "decoder out phase",
                "chip phase",
            ],
        )
        _assert_nonblank_phase_columns_are_labeled(self, tensors["phase"])
        self.assertNotIn("chip_amplitude_display", diagnostics)
        for key in ("SLM out |E|", "chip |E|"):
            normalized = tensors["amplitude"][key]
            self.assertTrue(torch.all(normalized >= 0.0).item(), key)
            self.assertTrue(torch.all(normalized <= 1.0).item(), key)
            for sample_idx in range(normalized.shape[0]):
                sample = normalized[sample_idx]
                self.assertAlmostEqual(sample.amin().item(), 0.0, delta=1e-5, msg=f"{key}[{sample_idx}] min")
                self.assertAlmostEqual(sample.amax().item(), 1.0, delta=1e-5, msg=f"{key}[{sample_idx}] max")
        self.assertTrue(torch.all(tensors["intensity"]["camera intensity"] >= 0.0).item())
        self.assertTrue(torch.all(tensors["intensity"]["camera intensity"] <= 1.0).item())
        self.assertTrue(torch.equal(tensors["phase"]["SLM out phase"], torch.zeros_like(camera)))
        self.assertFalse(any("mask" in key for split in tensors.values() for key in split))
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
        self.assertTrue(torch.allclose(tensors["amplitude"]["chip |E|"], expected_amplitude))
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
        _assert_nonblank_phase_columns_are_labeled(self, shared["phase"])
        self.assertEqual(
            list(shared["amplitude"].keys()),
            list(tensors["amplitude"].keys()),
        )
        self.assertEqual(
            list(shared["intensity"].keys()),
            [
                "noise code",
                "reference",
                "SLM command (amplitude control)",
                "SLM command contrast (display)",
                "SLM amplitude grid (control)",
                "SLM out I",
                "encoder in I",
                "encoder grid in I",
                "encoder field after L1 I",
                "encoder field after L2 I",
                "encoder out I",
                "ols in I",
                "ols out I",
                "decoder in I",
                "decoder field after L1 I",
                "decoder field after L2 I",
                "decoder out I",
                "chip I",
                "camera intensity",
            ],
        )
        shared_intensity = shared["intensity"]
        self.assertTrue(torch.allclose(shared_intensity["SLM command (amplitude control)"], slm_amplitude))
        self.assertTrue(torch.all(shared_intensity["SLM command contrast (display)"] >= 0.0).item())
        self.assertTrue(torch.all(shared_intensity["SLM command contrast (display)"] <= 1.0).item())
        shared_scale = shared["diagnostics"]["intensity_scale"]
        self.assertTrue(
            all("mask amplitude" not in key for key in shared["amplitude"]),
            "main physical-chain amplitude view should not include constant mask amplitude columns",
        )
        self.assertAlmostEqual(float(shared_intensity["chip I"].max().item()), 1.0, places=6)
        self.assertTrue(
            torch.allclose(
                shared_intensity["encoder in I"],
                (outputs["encoder_input_amplitude"].pow(2) / shared_scale).clamp(0.0, 1.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                shared_intensity["encoder grid in I"],
                (outputs["encoder_input_grid_amplitude"].pow(2) / shared_scale).clamp(0.0, 1.0),
            )
        )
        self.assertTrue(
            torch.allclose(shared_intensity["chip I"], (chip_amplitude.pow(2) / shared_scale).clamp(0.0, 1.0))
        )
        self.assertNotIn("noise code", shared["phase"],
                         "noise code has no phase source and should be excluded from phase split")
        self.assertEqual(shared["diagnostics"]["view"], "shared_scale_full_chain")
        self.assertEqual(shared["diagnostics"]["visualization_schema_version"], 2)
        self.assertEqual(shared["diagnostics"]["amplitude_display_policy"], "per_plane_robust_normalization")
        self.assertIn("SLM out I", shared["diagnostics"]["intensity_scale_columns"])
        self.assertIn("encoder grid in I", shared["diagnostics"]["intensity_scale_columns"])
        self.assertNotIn("reference", shared["diagnostics"]["intensity_scale_columns"])
        self.assertNotIn("camera intensity", shared["diagnostics"]["intensity_scale_columns"])
        self.assertNotIn("amplitude_scale", shared["diagnostics"])
        self.assertNotIn("early_raw_amplitude_columns", shared["diagnostics"])
        self.assertIn("p99_minus_p01_mean", shared["diagnostics"]["slm_command_stats"])
        self.assertIn("encoder grid in phase", shared["diagnostics"]["phase_stats"])

        coherent_outputs = dict(outputs)
        coherent_outputs.update(
            camera_field_available=True,
            camera_phase_available=True,
            camera_field_real=camera * 3.0,
            camera_field_imag=camera * 4.0,
            camera_phase=torch.zeros_like(camera),
        )
        coherent = build_one_shot_visualization_tensors(reference, coherent_outputs)
        camera_amplitude = camera * 5.0
        flat_camera_amplitude = camera_amplitude.flatten(start_dim=1)
        camera_lo = flat_camera_amplitude.quantile(0.02, dim=1).reshape(2, 1, 1, 1)
        camera_hi = flat_camera_amplitude.quantile(0.98, dim=1).reshape(2, 1, 1, 1)
        expected_camera_amplitude = (
            (camera_amplitude - camera_lo) / (camera_hi - camera_lo).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        self.assertTrue(
            torch.allclose(
                coherent["amplitude"]["camera field |E|"],
                expected_camera_amplitude,
            )
        )
        expected_camera_phase = torch.full_like(camera, math.atan2(4.0, 3.0) / (2.0 * math.pi))
        self.assertTrue(torch.allclose(coherent["phase"]["camera field phase"], expected_camera_phase))
        intensity_only_outputs = dict(coherent_outputs)
        intensity_only_outputs.update(camera_field_available=False, camera_phase_available=False)
        intensity_only = build_one_shot_visualization_tensors(reference, intensity_only_outputs)
        self.assertNotIn("camera field |E|", intensity_only["amplitude"])
        self.assertNotIn("camera field phase", intensity_only["phase"])

        d2nn = build_one_shot_d2nn_visualization_tensors(outputs)
        self.assertEqual(
            list(d2nn["layers"]["amplitude"].keys()),
            ["encoder field after L1", "encoder field after L2", "decoder field after L1", "decoder field after L2"],
        )
        self.assertEqual(
            list(d2nn["layers"]["phase"].keys()),
            [
                "encoder field after L1 phase",
                "encoder field after L2 phase",
                "decoder field after L1 phase",
                "decoder field after L2 phase",
            ],
        )
        self.assertEqual(
            list(d2nn["masks"]["amplitude"].keys()),
            [
                "encoder mask transmittance |t| L1",
                "encoder mask transmittance |t| L2",
                "decoder mask transmittance |t| L1",
                "decoder mask transmittance |t| L2",
            ],
        )
        self.assertEqual(
            list(d2nn["masks"]["phase"].keys()),
            ["encoder mask phase arg(t) L1", "encoder mask phase arg(t) L2", "decoder mask phase arg(t) L1", "decoder mask phase arg(t) L2"],
        )
        self.assertEqual(tuple(d2nn["layers"]["amplitude"]["encoder field after L1"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(d2nn["layers"]["phase"]["decoder field after L2 phase"].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(d2nn["masks"]["amplitude"]["encoder mask transmittance |t| L1"].shape), (1, 1, 8, 8))
        self.assertFalse(any("mask" in key for key in tensors["amplitude"] | tensors["phase"]))
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
        self.assertTrue(torch.all(d2nn["masks"]["amplitude"]["encoder mask transmittance |t| L1"] >= 0.0).item())
        self.assertTrue(
            torch.allclose(d2nn["masks"]["amplitude"]["encoder mask transmittance |t| L1"], torch.ones(1, 1, 8, 8))
        )
        self.assertTrue(torch.all(d2nn["masks"]["phase"]["decoder mask phase arg(t) L2"] <= 1.0).item())
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
        self.assertIn("finite_ratio", d2nn_diagnostics["masks"]["phase"]["decoder mask phase arg(t) L2"])
        self.assertIn("local_contrast_mean", d2nn_diagnostics["masks"]["phase"]["decoder mask phase arg(t) L2"])
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
            [
                "encoder mask transmittance |t| L1",
                "encoder mask transmittance |t| L2",
                "decoder mask transmittance |t| L1",
                "decoder mask transmittance |t| L2",
            ],
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

    def test_relay_psf_back_path_preserves_crop_and_blurs_readout(self) -> None:
        import torch

        from ldm_project.config import PhysicsConfig, ReadoutConfig, load_training_config
        from ldm_project.optics import BackOpticalPath

        def total_variation(image):
            return (image[..., 1:, :] - image[..., :-1, :]).abs().mean() + (
                image[..., :, 1:] - image[..., :, :-1]
            ).abs().mean()

        chip_amplitude = torch.zeros(1, 1, 16, 16)
        chip_amplitude[:, :, 7:9, :] = 1.0
        chip_amplitude[:, :, :, 7:9] = 1.0
        chip_field = torch.complex(chip_amplitude, torch.zeros_like(chip_amplitude))

        identity_physics = PhysicsConfig(back_path_model="relay_identity")
        psf_identity_physics = PhysicsConfig(
            back_path_model="relay_psf",
            readout=ReadoutConfig(relay_psf_sigma_px=0.0),
        )
        blurred_physics = PhysicsConfig(
            back_path_model="relay_psf",
            readout=ReadoutConfig(relay_psf_sigma_px=1.25),
        )

        identity_outputs = BackOpticalPath(identity_physics)(chip_field)
        psf_identity_outputs = BackOpticalPath(psf_identity_physics)(chip_field)
        blurred_outputs = BackOpticalPath(blurred_physics)(chip_field)

        self.assertEqual(tuple(blurred_outputs["camera_plane"].shape), tuple(chip_amplitude.shape))
        self.assertTrue(torch.isfinite(blurred_outputs["camera_plane"]).all().item())
        self.assertTrue(torch.allclose(identity_outputs["camera_plane"], psf_identity_outputs["camera_plane"]))
        self.assertLess(
            total_variation(blurred_outputs["camera_plane"]).item(),
            total_variation(psf_identity_outputs["camera_plane"]).item(),
        )
        self.assertEqual(
            load_training_config(Path("configs/audit_one_shot_d2nn_v9_8_back_relay_psf.yaml")).physics.back_path_model,
            "relay_psf",
        )

        with self.assertRaisesRegex(ValueError, "relay_psf_sigma_px"):
            ReadoutConfig(relay_psf_sigma_px=-0.1)

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

        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        train_cfg = TrainingConfig(
            data=DataConfig(validation_fraction=0.0),
            one_shot=OneShotConfig(
                num_classes=2,
                sample_class_labels=(0, 1),
                sample_repeats_per_class=1,
                eval_samples_per_class=3,
            ),
        )

        selected, summary = _select_distill_eval_indices(torch, labels, train_cfg)

        self.assertEqual(len(selected), 6)
        self.assertEqual(summary["requested_samples_per_class"], 3)
        self.assertEqual(summary["selected_count"], 6)
        self.assertEqual(summary["per_class"]["0"]["selected"], 3)
        self.assertEqual(summary["per_class"]["1"]["selected"], 3)

    def test_one_shot_distill_eval_selection_can_disable_train_fallback(self) -> None:
        import torch

        from ldm_project.config import DataConfig, OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _select_distill_eval_indices

        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        train_cfg = TrainingConfig(
            data=DataConfig(validation_fraction=0.1),
            one_shot=OneShotConfig(
                num_classes=2,
                sample_class_labels=(0, 1),
                eval_samples_per_class=2,
                eval_allow_train_fallback=False,
            ),
        )

        selected, summary = _select_distill_eval_indices(torch, labels, train_cfg)

        self.assertEqual(len(selected), 4)
        self.assertFalse(summary["used_train_fallback"])
        self.assertEqual(summary["validation_size"], len(labels))

        train_cfg.one_shot.eval_samples_per_class = 4
        with self.assertRaisesRegex(ValueError, "eval_allow_train_fallback"):
            _select_distill_eval_indices(torch, labels, train_cfg)

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
                "eval_target_display": base.roll(shifts=1, dims=-1),
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
            intensity_path = run_manager.images_dir / "one_shot_distill_eval_focus_0_1_2_3_4_5_6_7_8_9_intensity.png"
            phase_path = run_manager.images_dir / "one_shot_distill_eval_focus_0_1_2_3_4_5_6_7_8_9_phase.png"
            artifact_path = run_manager.artifacts_dir / "one_shot_distill_eval_focus_diagnostics.json"
            config_path = run_manager.configs_dir / "one_shot_distill_eval_diagnostics_config.json"
            self.assertTrue(amplitude_path.is_file())
            self.assertGreater(amplitude_path.stat().st_size, 0)
            self.assertTrue(intensity_path.is_file())
            self.assertGreater(intensity_path.stat().st_size, 0)
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
            self.assertIn("intensity", diagnostics["images"])
            self.assertIn("phase", diagnostics["images"])
            self.assertEqual(diagnostics["focus_target_display_key"], "eval_target_display")
            self.assertEqual(diagnostics["visual_diagnostics"]["view"], "compact_full_chain")
            self.assertIn("encoder_out_ols_in", diagnostics["visual_diagnostics"]["amplitude_phase_similarity"])
            self.assertIn("top25_iou", diagnostics["visual_diagnostics"]["amplitude_phase_similarity"]["chip"])
            with config_path.open("r", encoding="utf-8") as handle:
                config_snapshot = json.load(handle)
            self.assertEqual(config_snapshot["focus_target_display_key"], "eval_target_display")

            legacy_bundle_path = root / "legacy_one_shot_distill_eval_tensors.pt"
            legacy_payload = dict(tensor_payload)
            legacy_payload.pop("eval_target_display")
            torch.save(legacy_payload, legacy_bundle_path)
            legacy_run_manager = RunManager(root / "legacy_run")

            run_one_shot_distill_eval_diagnostics_stage(
                torch,
                run_manager=legacy_run_manager,
                run_meta={},
                eval_tensor_bundle=str(legacy_bundle_path),
                eval_metrics=str(metrics_path),
                diagnostic_class_labels=(0,),
                diagnostic_max_per_class=1,
            )

            legacy_artifact_path = legacy_run_manager.artifacts_dir / "one_shot_distill_eval_focus_diagnostics.json"
            legacy_config_path = legacy_run_manager.configs_dir / "one_shot_distill_eval_diagnostics_config.json"
            with legacy_artifact_path.open("r", encoding="utf-8") as handle:
                legacy_diagnostics = json.load(handle)
            with legacy_config_path.open("r", encoding="utf-8") as handle:
                legacy_config_snapshot = json.load(handle)
            self.assertEqual(legacy_diagnostics["focus_target_display_key"], "teacher_camera_display")
            self.assertEqual(legacy_config_snapshot["focus_target_display_key"], "teacher_camera_display")

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
            intensity_call = next(call for call in run_manager.grid_calls if call[0] == "one_shot_sample_intensity.png")
            phase_call = next(call for call in run_manager.grid_calls if call[0] == "one_shot_sample_phase.png")
            self.assertFalse(any(call[0] == "one_shot_sample_shared_scale_phase.png" for call in run_manager.grid_calls))
            self.assertEqual(
                list(amplitude_call[1].keys()),
                [
                    "SLM command (amplitude control)",
                    "SLM out |E|",
                    "encoder in |E|",
                    "encoder grid in |E|",
                    "encoder field after L1 |E|",
                    "encoder field after L2 |E|",
                    "encoder out |E|",
                    "OLS in |E|",
                    "OLS out |E|",
                    "decoder in |E|",
                    "decoder field after L1 |E|",
                    "decoder field after L2 |E|",
                    "decoder out |E|",
                    "chip |E|",
                ],
            )
            self.assertEqual(
                list(intensity_call[1].keys()),
                [
                    "noise code",
                    "reference",
                    "SLM command (amplitude control)",
                    "SLM command contrast (display)",
                    "SLM amplitude grid (control)",
                    "SLM out I",
                    "encoder in I",
                    "encoder grid in I",
                    "encoder field after L1 I",
                    "encoder field after L2 I",
                    "encoder out I",
                    "ols in I",
                    "ols out I",
                    "decoder in I",
                    "decoder field after L1 I",
                    "decoder field after L2 I",
                    "decoder out I",
                    "chip I",
                    "camera intensity",
                ],
            )
            self.assertEqual(
                list(phase_call[1].keys()),
                [
                    "SLM out phase",
                    "encoder in phase",
                    "encoder grid in phase",
                    "encoder field after L1 phase",
                    "encoder field after L2 phase",
                    "encoder out phase",
                    "OLS in phase",
                    "OLS out phase",
                    "decoder in phase",
                    "decoder field after L1 phase",
                    "decoder field after L2 phase",
                    "decoder out phase",
                    "chip phase",
                ],
            )
            self.assertIn("chip I", intensity_call[1])
            self.assertIn("chip phase", phase_call[1])
            self.assertIn("encoder field after L1 I", intensity_call[1])
            self.assertIn("decoder field after L2 phase", phase_call[1])
            self.assertEqual(tuple(intensity_call[1]["encoder field after L1 I"].shape), (8, 1, 2, 2))
            self.assertFalse(any("mask amplitude" in key for key in intensity_call[1]))
            self.assertFalse(any("mask" in key for key in amplitude_call[1] | phase_call[1]))
            _assert_nonblank_phase_columns_are_labeled(self, phase_call[1])
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
            self.assertEqual(shared_diagnostics["visualization_schema_version"], 2)
            self.assertIn("SLM out I", shared_diagnostics["intensity_scale_columns"])
            self.assertIn("encoder grid in I", shared_diagnostics["intensity_scale_columns"])
            self.assertNotIn("reference", shared_diagnostics["intensity_scale_columns"])
            self.assertNotIn("camera intensity", shared_diagnostics["intensity_scale_columns"])
            self.assertNotIn("amplitude_scale_columns", shared_diagnostics)
            self.assertNotIn("early_raw_amplitude_columns", shared_diagnostics)
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
        from unittest.mock import patch

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
                    distill_target_source="mnist_prototype",
                    sample_class_labels=(0, 1),
                    sample_repeats_per_class=4,
                    student_label_audit_enabled=True,
                ),
            )
            run_manager = RecordingRunManager(run_dir)
            prototypes = torch.zeros(2, 1, 4, 4)
            prototypes[0, :, :2, :] = 1.0
            prototypes[1, :, :, :2] = 1.0

            with patch("ldm_project.one_shot_eval._build_mnist_label_prototype_bank", return_value=prototypes):
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
            intensity_call = next(
                call for call in run_manager.grid_calls if call[0] == "one_shot_distill_eval_intensity.png"
            )
            phase_call = next(
                call for call in run_manager.grid_calls if call[0] == "one_shot_distill_eval_phase.png"
            )
            self.assertFalse(
                any(call[0] == "one_shot_distill_eval_shared_scale_phase.png" for call in run_manager.grid_calls)
            )
            self.assertEqual(
                list(amplitude_call[1].keys()),
                [
                    "SLM command (amplitude control)",
                    "SLM out |E|",
                    "encoder in |E|",
                    "encoder grid in |E|",
                    "encoder field after L1 |E|",
                    "encoder field after L2 |E|",
                    "encoder out |E|",
                    "OLS in |E|",
                    "OLS out |E|",
                    "decoder in |E|",
                    "decoder field after L1 |E|",
                    "decoder field after L2 |E|",
                    "decoder out |E|",
                    "chip |E|",
                ],
            )
            self.assertEqual(
                list(intensity_call[1].keys()),
                [
                    "digit",
                    "noise code",
                    "teacher intensity target",
                    "SLM command (amplitude control)",
                    "SLM command contrast (display)",
                    "SLM amplitude grid (control)",
                    "SLM out I",
                    "encoder in I",
                    "encoder grid in I",
                    "encoder field after L1 I",
                    "encoder field after L2 I",
                    "encoder out I",
                    "ols in I",
                    "ols out I",
                    "decoder in I",
                    "decoder field after L1 I",
                    "decoder field after L2 I",
                    "decoder out I",
                    "chip I",
                    "student camera intensity",
                    "intensity abs error",
                ],
            )
            self.assertEqual(
                list(phase_call[1].keys()),
                [
                    "SLM out phase",
                    "encoder in phase",
                    "encoder grid in phase",
                    "encoder field after L1 phase",
                    "encoder field after L2 phase",
                    "encoder out phase",
                    "OLS in phase",
                    "OLS out phase",
                    "decoder in phase",
                    "decoder field after L1 phase",
                    "decoder field after L2 phase",
                    "decoder out phase",
                    "chip phase",
                ],
            )
            self.assertIn("teacher intensity target", intensity_call[1])
            self.assertIn("student camera intensity", intensity_call[1])
            self.assertIn("chip I", intensity_call[1])
            self.assertIn("chip phase", phase_call[1])
            self.assertIn("encoder field after L1 I", intensity_call[1])
            self.assertIn("decoder field after L2 phase", phase_call[1])
            self.assertEqual(tuple(intensity_call[1]["encoder field after L1 I"].shape), (4, 1, 2, 2))
            self.assertFalse(any("mask amplitude" in key for key in intensity_call[1]))
            self.assertFalse(any("mask" in key for key in amplitude_call[1] | phase_call[1]))
            _assert_nonblank_phase_columns_are_labeled(self, phase_call[1])
            self.assertFalse(any("_d2nn_" in call[0] for call in run_manager.grid_calls))
            self.assertNotIn("chip_amplitude_display", amplitude_call[1])
            self.assertNotIn("chip_amplitude_display", phase_call[1])
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
            self.assertIn("student_label_correct_ratio", metrics["overall"])
            self.assertIn("min_class_student_label_correct_ratio", metrics["overall"])
            self.assertIn("strict_readability_acceptance", metrics)
            self.assertIn("one_shot_distill_eval_student_label_audit", run_manager.artifact_payloads)
            student_audit = run_manager.artifact_payloads["one_shot_distill_eval_student_label_audit"]
            self.assertEqual(student_audit["sample_count"], 8)
            self.assertIn("nearest_label", student_audit["per_sample"][0])
            self.assertEqual(eval_config["visual_preview_count"], 4)
            self.assertEqual(eval_config["visual_preview_labels"], [0, 0, 1, 1])
            self.assertEqual(eval_config["one_shot_distill_target_source"], "mnist_prototype")
            self.assertTrue(eval_config["student_label_audit_enabled"])
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
            self.assertIn("decoder mask phase arg(t) L2", eval_d2nn_diagnostics["masks"]["phase"])
            shared_diagnostics = run_manager.artifact_payloads["one_shot_distill_eval_shared_scale_diagnostics"]
            self.assertEqual(shared_diagnostics["view"], "shared_scale_full_chain")
            self.assertEqual(shared_diagnostics["visualization_schema_version"], 2)
            self.assertIn("SLM out I", shared_diagnostics["intensity_scale_columns"])
            self.assertIn("decoder out I", shared_diagnostics["intensity_scale_columns"])
            self.assertNotIn("teacher intensity target", shared_diagnostics["intensity_scale_columns"])
            self.assertNotIn("student camera intensity", shared_diagnostics["intensity_scale_columns"])
            self.assertEqual(shared_diagnostics["columns"]["intensity"], list(intensity_call[1]))
            self.assertNotIn("amplitude_scale_columns", shared_diagnostics)
            self.assertNotIn("early_raw_amplitude_columns", shared_diagnostics)
            self.assertIn("encoder out phase", shared_diagnostics["phase_stats"])
            self.assertEqual(amplitude_call[2], 4)
            self.assertEqual(amplitude_call[3]["row_labels"], ["0", "0", "1", "1"])
            self.assertEqual(tuple(intensity_call[1]["teacher intensity target"].shape), (4, 1, 4, 4))
            self.assertEqual(tuple(run_manager.bundle_payload["teacher_camera"].shape), (8, 1, 4, 4))
            self.assertEqual(run_manager.bundle_payload["one_shot_distill_target_source"], "mnist_prototype")
            self.assertEqual(run_manager.meta_payload["one_shot_distill_target_source"], "mnist_prototype")
            expected_targets = prototypes.index_select(0, run_manager.bundle_payload["class_labels"])
            self.assertTrue(torch.equal(run_manager.bundle_payload["eval_target"], expected_targets))
            self.assertFalse(torch.equal(run_manager.bundle_payload["teacher_camera"], expected_targets))

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

    def test_one_shot_loss_supports_aux_detail_and_residual_penalties(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        target[:, :, 1:3, 1:3] = 1.0
        aux_detail = torch.zeros_like(target)
        aux_detail[:, :, :, 1:3] = 1.0
        prediction = target.clone().requires_grad_(True)
        residual = torch.full_like(target, 0.25, requires_grad=True)
        outputs = {
            "camera_plane": prediction,
            "slm_amplitude": torch.ones_like(target),
            "generator_residual_map": residual,
            "generator_base_command": torch.zeros_like(target),
            "generator_residual_abs_mean": residual.detach().abs().flatten(1).mean(dim=1),
            "generator_residual_bg_abs_mean": residual.detach().abs().flatten(1).mean(dim=1),
        }

        losses = one_shot_loss(
            outputs,
            target,
            aux_detail_target=aux_detail,
            aux_detail_shape_weight=0.1,
            aux_detail_edge_weight=0.1,
            residual_l1_weight=0.2,
            residual_background_weight=0.3,
        )

        self.assertGreater(losses["aux_detail_shape_loss"].item(), 0.0)
        self.assertGreater(losses["aux_detail_edge_loss"].item(), 0.0)
        self.assertGreater(losses["residual_l1_loss"].item(), 0.0)
        self.assertGreater(losses["residual_background_loss"].item(), 0.0)
        self.assertGreater(losses["student_base_mse"].item(), 0.0)
        self.assertGreater(losses["student_base_foreground_mse"].item(), 0.0)
        self.assertEqual(losses["student_base_background_mse"].item(), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNotNone(residual.grad)

    def test_one_shot_loss_aux_detail_foreground_gate_blocks_off_template_detail(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
        target[:, :, 1:3, 1:3] = 1.0
        aux_detail = torch.zeros_like(target)
        aux_detail[:, :, 4:, 4:] = 1.0
        prediction = target.clone().requires_grad_(True)
        outputs = {
            "camera_plane": prediction,
            "slm_amplitude": torch.ones_like(target),
        }

        losses = one_shot_loss(
            outputs,
            target,
            aux_detail_target=aux_detail,
            aux_detail_shape_weight=0.1,
            aux_detail_edge_weight=0.1,
            aux_detail_foreground_gate_dilation=1,
        )

        self.assertGreater(losses["aux_detail_gate_fg_ratio"].item(), 0.0)
        self.assertEqual(losses["aux_detail_active_fg_ratio"].item(), 0.0)
        self.assertEqual(losses["aux_detail_shape_loss"].item(), 0.0)
        self.assertEqual(losses["aux_detail_edge_loss"].item(), 0.0)

    def test_one_shot_loss_residual_background_uses_foreground_gate_complement(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
        target[:, :, 2, 2] = 1.0
        residual = torch.full_like(target, 2.0, requires_grad=True)
        with torch.no_grad():
            residual[:, :, 1:4, 1:4] = 1.0
        outputs = {
            "camera_plane": target.clone(),
            "slm_amplitude": torch.ones_like(target),
            "generator_residual_map": residual,
        }

        losses = one_shot_loss(
            outputs,
            target,
            residual_background_weight=0.5,
            aux_detail_foreground_gate_dilation=3,
        )

        self.assertAlmostEqual(losses["aux_detail_gate_fg_ratio"].item(), 9.0 / 25.0, places=6)
        self.assertAlmostEqual(losses["residual_background_mask_ratio"].item(), 16.0 / 25.0, places=6)
        self.assertAlmostEqual(losses["residual_background_penalty"].item(), 2.0, places=6)
        self.assertAlmostEqual(losses["residual_background_loss"].item(), 1.0, places=6)
        losses["loss"].backward()
        self.assertIsNotNone(residual.grad)

    def test_one_shot_loss_student_base_metrics_default_zero_without_base_command(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.ones(1, 1, 4, 4, dtype=torch.float32)
        losses = one_shot_loss(
            {
                "camera_plane": torch.zeros_like(target),
                "slm_amplitude": torch.ones_like(target),
            },
            target,
        )

        self.assertEqual(losses["student_base_mse"].item(), 0.0)
        self.assertEqual(losses["student_base_foreground_mse"].item(), 0.0)
        self.assertEqual(losses["student_base_background_mse"].item(), 0.0)

    def test_one_shot_readability_loss_defaults_off_and_activates(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16),
            "slm_amplitude": torch.ones(2, 1, 16, 16),
        }

        off = one_shot_loss(outputs, target)
        on = one_shot_loss(outputs, target, camera_readability_weight=1.0)

        self.assertEqual(off["camera_readability_loss"].item(), 0.0)
        self.assertEqual(off["camera_readability_penalty"].item(), 0.0)
        self.assertGreater(on["camera_readability_loss"].item(), 0.0)
        self.assertGreater(on["camera_readability_penalty"].item(), 0.0)
        self.assertLessEqual(on["camera_readability_loss"].item(), on["loss"].item())

    def test_one_shot_loss_activates_background_speckle_and_contrast_terms(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
        target[:, :, 2:6, 2:6] = 1.0
        prediction = target * 0.02
        prediction[:, :, 0, 0] = 0.5
        prediction.requires_grad_(True)
        outputs = {
            "camera_plane": prediction,
            "slm_amplitude": torch.ones(2, 1, 8, 8),
        }

        losses = one_shot_loss(
            outputs,
            target,
            camera_background_speckle_weight=0.1,
            camera_background_hotspot_weight=0.1,
            camera_background_hotspot_threshold=0.08,
            camera_background_highfreq_weight=0.1,
            camera_readable_contrast_weight=0.1,
            camera_readable_contrast_target=4.5,
        )

        self.assertGreater(losses["camera_background_speckle_loss"].item(), 0.0)
        self.assertGreater(losses["background_speckle_ratio"].item(), 0.0)
        self.assertGreater(losses["camera_background_hotspot_loss"].item(), 0.0)
        self.assertGreater(losses["background_hotspot_ratio"].item(), 0.0)
        self.assertGreater(losses["camera_background_highfreq_loss"].item(), 0.0)
        self.assertGreater(losses["background_highfreq_energy"].item(), 0.0)
        self.assertGreater(losses["camera_readable_contrast_loss"].item(), 0.0)
        self.assertLess(losses["readable_contrast"].item(), 4.5)
        losses["loss"].backward()
        self.assertIsNotNone(prediction.grad)

    def test_one_shot_loss_reports_readability_diagnostics_when_weights_are_zero(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
        target[:, :, 2:6, 2:6] = 1.0
        prediction = target * 0.02
        prediction[:, :, 0, 0] = 0.5
        outputs = {
            "camera_plane": prediction,
            "slm_amplitude": torch.ones(2, 1, 8, 8),
        }

        losses = one_shot_loss(
            outputs,
            target,
            camera_background_hotspot_threshold=0.08,
            camera_readable_contrast_target=4.5,
        )

        self.assertEqual(losses["camera_artifact_loss"].item(), 0.0)
        self.assertEqual(losses["camera_background_speckle_loss"].item(), 0.0)
        self.assertEqual(losses["camera_background_hotspot_loss"].item(), 0.0)
        self.assertEqual(losses["camera_background_highfreq_loss"].item(), 0.0)
        self.assertEqual(losses["camera_readable_contrast_loss"].item(), 0.0)
        self.assertGreater(losses["foreground_hole_ratio"].item(), 0.0)
        self.assertGreater(losses["background_speckle_ratio"].item(), 0.0)
        self.assertGreater(losses["background_hotspot_ratio"].item(), 0.0)
        self.assertGreater(losses["background_highfreq_energy"].item(), 0.0)
        self.assertLess(losses["readable_contrast"].item(), 4.5)

    def test_save_epoch_distill_checkpoint_persists_epoch_records_to_meta(self) -> None:
        import torch

        from ldm_project.config import TrainingConfig
        from ldm_project.one_shot_distill_training import _save_epoch_distill_checkpoint

        class DummyModel:
            def state_dict(self):
                return {"weight": torch.tensor(1.0)}

        class RecordingRunManager:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.checkpoints_dir = run_dir / "checkpoints"
                self.meta_payload = None

            def save_checkpoint(self, name, payload):
                self.saved_checkpoint = (name, dict(payload))
                return self.checkpoints_dir / name

            def write_meta(self, meta):
                self.meta_payload = dict(meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_manager = RecordingRunManager(Path(tmpdir))
            train_cfg = TrainingConfig(one_shot_distill_epoch_checkpoint_interval=2)
            run_meta: dict[str, object] = {}

            _save_epoch_distill_checkpoint(
                run_manager,
                run_meta,
                one_shot_model=DummyModel(),
                train_cfg=train_cfg,
                physics={},
                epoch_number=2,
                one_shot_model_version="test",
                distill_dataset="distill.pt",
                validation_metrics={"diversity_ratio": 0.7, "eval_corr": 0.8, "eval_dice": 0.9, "best_loss": 0.1},
            )

            self.assertIsNotNone(run_manager.meta_payload)
            self.assertEqual(run_manager.meta_payload["epoch_checkpoints"][0]["epoch"], 2)
            self.assertTrue(
                str(run_manager.meta_payload["epoch_checkpoints"][0]["path"]).endswith(
                    "one_shot_distill_epoch_0002.pt"
                )
            )

    def test_one_shot_loss_student_label_margin_prefers_matching_prototype(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        proto_zero = torch.zeros(1, 1, 4, 4)
        proto_zero[:, :, :2, :2] = 1.0
        proto_one = torch.zeros(1, 1, 4, 4)
        proto_one[:, :, 2:, 2:] = 1.0
        prototypes = torch.cat([proto_zero, proto_one], dim=0)
        labels = torch.tensor([0, 1], dtype=torch.long)
        target = prototypes.index_select(0, labels)
        slm = torch.ones(2, 1, 4, 4)

        matched = one_shot_loss(
            {"camera_plane": target.clone(), "slm_amplitude": slm},
            target,
            class_labels=labels,
            student_label_margin_weight=1.0,
            student_label_margin_target=0.05,
            student_label_margin_prototypes=prototypes,
        )
        swapped = one_shot_loss(
            {"camera_plane": target.flip(0).clone(), "slm_amplitude": slm},
            target,
            class_labels=labels,
            student_label_margin_weight=1.0,
            student_label_margin_target=0.05,
            student_label_margin_prototypes=prototypes,
        )

        self.assertGreater(matched["student_label_correct_ratio"].item(), swapped["student_label_correct_ratio"].item())
        self.assertLess(matched["student_label_margin_loss"].item(), swapped["student_label_margin_loss"].item())
        self.assertGreater(matched["student_label_margin_mean"].item(), swapped["student_label_margin_mean"].item())

    def test_one_shot_distill_eval_readability_metrics_present_and_bounded(self) -> None:
        import torch

        from ldm_project.one_shot_eval_metrics import compute_one_shot_distill_eval_metrics

        teacher = torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        student = teacher.clone()
        student[0, 0, 1, 0] = 0.05  # fake speckle in background
        student[2, 0, 0, 1] = 0.05
        slm = torch.ones(4, 1, 2, 2, dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        metrics = compute_one_shot_distill_eval_metrics(
            torch,
            teacher_camera=teacher,
            student_camera=student,
            slm_amplitude=slm,
            class_labels=labels,
        )

        for key in ("foreground_ratio", "background_speckle_ratio", "readable_contrast"):
            self.assertIn(key, metrics["overall"], f"missing overall.{key}")
            self.assertTrue(math.isfinite(metrics["overall"][key]))
            self.assertGreaterEqual(metrics["overall"][key], 0.0)
            self.assertIn(key, metrics["per_class"]["0"], f"missing per_class.0.{key}")
            self.assertTrue(math.isfinite(metrics["per_class"]["0"][key]))
            self.assertGreaterEqual(metrics["per_class"]["0"][key], 0.0)

        # background_speckle_ratio should be 0 when there is no background region
        no_bg_teacher = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        no_bg_student = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        no_bg_slm = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        no_bg_labels = torch.tensor([0], dtype=torch.long)
        no_bg_metrics = compute_one_shot_distill_eval_metrics(
            torch,
            teacher_camera=no_bg_teacher,
            student_camera=no_bg_student,
            slm_amplitude=no_bg_slm,
            class_labels=no_bg_labels,
        )
        self.assertEqual(no_bg_metrics["overall"]["background_speckle_ratio"], 0.0)

    def test_direct_oracle_eval_visuals_keep_only_readable_columns_and_no_phase(self) -> None:
        from ldm_project.one_shot_eval import _direct_readable_eval_visuals
        from ldm_project.visualization_core import VISUALIZATION_INTENSITY_KEY, VISUALIZATION_PHASE_KEY

        visuals = {
            VISUALIZATION_INTENSITY_KEY: {
                "digit": object(),
                "teacher intensity target": object(),
                "SLM command (amplitude control)": object(),
                "student camera intensity": object(),
                "intensity abs error": object(),
                "encoded out I": object(),
                "ols out I": object(),
                "decoder out I": object(),
            },
            VISUALIZATION_PHASE_KEY: {
                "encoded out phase": object(),
                "ols out phase": object(),
            },
        }

        direct = _direct_readable_eval_visuals(visuals)

        self.assertEqual(
            tuple(direct[VISUALIZATION_INTENSITY_KEY].keys()),
            (
                "digit",
                "teacher intensity target",
                "SLM command (amplitude control)",
                "student camera intensity",
                "intensity abs error",
            ),
        )
        self.assertEqual(direct[VISUALIZATION_PHASE_KEY], {})

    def test_student_label_audit_metrics_and_strict_acceptance(self) -> None:
        from ldm_project.one_shot_eval_metrics import (
            add_student_label_audit_metrics,
            compute_strict_student_readability_acceptance,
        )

        metrics = {
            "overall": {
                "count": 4,
                "paired_corr": 0.80,
                "paired_dice": 0.73,
                "background_speckle_ratio": 0.07,
                "foreground_hole_ratio": 0.05,
                "student_saturation_ratio": 0.07,
                "student_high_saturation_ratio": 0.01,
                "readable_contrast": 4.8,
            },
            "per_class": {
                "0": {"paired_dice": 0.70},
                "1": {"paired_dice": 0.69},
            },
        }
        audit_payload = {
            "label_correct_count": 4,
            "label_correct_ratio": 1.0,
            "overall": {"label_margin": {"mean": 0.06}},
            "per_class": {
                "0": {
                    "count": 2,
                    "label_correct_count": 2,
                    "label_correct_ratio": 1.0,
                    "label_margin": {"mean": 0.05},
                },
                "1": {
                    "count": 2,
                    "label_correct_count": 2,
                    "label_correct_ratio": 1.0,
                    "label_margin": {"mean": 0.04},
                },
            },
            "per_sample": [
                {"requested_label": 0, "nearest_label": 0},
                {"requested_label": 0, "nearest_label": 0},
                {"requested_label": 1, "nearest_label": 1},
                {"requested_label": 1, "nearest_label": 1},
            ],
        }

        add_student_label_audit_metrics(metrics, audit_payload)
        acceptance = compute_strict_student_readability_acceptance(metrics)

        self.assertTrue(acceptance["passed"])
        self.assertEqual(metrics["overall"]["student_label_correct_count"], 4)
        self.assertEqual(metrics["overall"]["min_class_student_label_correct_ratio"], 1.0)
        self.assertAlmostEqual(metrics["overall"]["min_class_student_label_margin_mean"], 0.04, places=6)
        self.assertEqual(metrics["per_class"]["0"]["student_nearest_label_confusion"], {"0": 2})
        self.assertEqual(
            set(acceptance["checks"]),
            {"paired_corr", "paired_dice", "student_high_saturation_ratio"},
        )

        metrics["per_class"]["1"]["student_label_correct_ratio"] = 0.5
        diagnostics_only = compute_strict_student_readability_acceptance(metrics)
        self.assertTrue(diagnostics_only["passed"])
        metrics["overall"]["paired_corr"] = 0.77
        self.assertFalse(compute_strict_student_readability_acceptance(metrics)["passed"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class OneShotCleanTargetTests(unittest.TestCase):
    """Tests for clean readable target generation and associated losses."""

    def _check_soft_mask_range(self, mask) -> None:
        self.assertTrue(mask.isfinite().all())
        self.assertGreaterEqual(mask.min().item(), 0.0)
        self.assertLessEqual(mask.max().item(), 1.0)

    def test_build_clean_readable_target_shape_and_range(self) -> None:
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        teacher = torch.rand(4, 1, 32, 32) * 0.8 + 0.1
        result = _build_clean_readable_target(teacher)

        self.assertIn("clean_target", result)
        self.assertIn("clean_target_fg_ratio", result)
        clean = result["clean_target"]
        self.assertEqual(tuple(clean.shape), (4, 1, 32, 32))
        self._check_soft_mask_range(clean)
        # Foreground ratio should be in reasonable range for random data.
        fg_ratio = float(result["clean_target_fg_ratio"].item())
        self.assertGreater(fg_ratio, 0.0)
        self.assertLessEqual(fg_ratio, 1.0)

    def test_build_clean_readable_target_sharp_edges(self) -> None:
        """Clean target should preserve sharp edges from a step-function input."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        step = torch.zeros(1, 1, 32, 32)
        step[:, :, 8:24, 8:24] = 1.0
        result = _build_clean_readable_target(step)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # Most of the foreground region should be high (>0.5).
        interior = clean[:, :, 10:22, 10:22]
        self.assertGreater(interior.mean().item(), 0.7)
        # Most of the background should be low (<0.5).
        exterior = clean[:, :, :6, :6]
        self.assertLess(exterior.mean().item(), 0.3)
        # Opening shouldn't fully erode the foreground.
        fg_area = (clean > 0.5).float().sum().item()
        expected_area = 16 * 16
        self.assertGreater(fg_area, expected_area * 0.5)
        self.assertLess(fg_area, expected_area * 1.5)

    def test_build_clean_readable_target_closes_small_holes(self) -> None:
        """Morphology close should fill small holes in the foreground."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        ring = torch.ones(1, 1, 32, 32) * 0.1
        ring[:, :, 6:26, 6:26] = 0.8
        ring[:, :, 12:20, 12:20] = 0.1  # hole in foreground
        result = _build_clean_readable_target(ring)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # The hole region should be at least partially filled by closing.
        hole_region = clean[:, :, 14:18, 14:18]
        self.assertGreater(hole_region.mean().item(), 0.2)

    def test_build_clean_readable_target_opens_small_speckles(self) -> None:
        """Morphology open should suppress small background speckles."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        speckled = torch.zeros(1, 1, 32, 32)
        speckled[:, :, 8:24, 8:24] = 0.8  # foreground block
        speckled[:, :, 2:4, 2:4] = 0.9  # small speckle in background
        speckled[:, :, 28:30, 28:30] = 0.9  # small speckle in background
        result = _build_clean_readable_target(speckled)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # Speckle regions should be suppressed below 0.5.
        speckle1 = clean[:, :, 2:4, 2:4]
        speckle2 = clean[:, :, 28:30, 28:30]
        self.assertLess(speckle1.mean().item(), 0.5)
        self.assertLess(speckle2.mean().item(), 0.5)
        # Main foreground block should survive.
        main_block = clean[:, :, 10:22, 10:22]
        self.assertGreater(main_block.mean().item(), 0.5)

    def test_clean_losses_backpropagate_with_clean_target_enabled(self) -> None:
        """All clean-target losses should backpropagate when clean_target_enabled=True."""
        import torch

        from ldm_project.training import one_shot_loss

        target = (torch.rand(2, 1, 16, 16) > 0.3).float()
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "camera_plane_loss": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }
        losses = one_shot_loss(
            outputs,
            target,
            clean_target_enabled=True,
            clean_foreground_dice_weight=0.5,
            clean_foreground_hole_weight=0.3,
            clean_background_speckle_weight=0.1,
            clean_background_hotspot_weight=0.5,
            clean_background_highfreq_weight=0.3,
        )
        self.assertIn("clean_target_fg_ratio", losses)
        self.assertGreater(float(losses["clean_target_fg_ratio"]), 0.0)
        self.assertGreater(float(losses["clean_dice_loss"]), 0.0)
        self.assertGreater(float(losses["clean_fg_hole_loss"]), 0.0)
        self.assertGreater(float(losses["clean_bg_speckle_loss"]), 0.0)
        losses["loss"].backward()
        # Camera plane should have received gradients from clean losses.
        self.assertIsNotNone(outputs["camera_plane"].grad)
        self.assertGreater(outputs["camera_plane"].grad.abs().sum().item(), 0.0)

    def test_clean_losses_noop_when_clean_target_disabled(self) -> None:
        """clean_target_enabled=False should produce zero clean losses."""
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "camera_plane_loss": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }
        losses = one_shot_loss(
            outputs,
            target,
            clean_target_enabled=False,
        )
        self.assertEqual(float(losses["clean_dice_loss"]), 0.0)
        self.assertEqual(float(losses["clean_fg_hole_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_speckle_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_hotspot_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_highfreq_loss"]), 0.0)
        self.assertIn("clean_target_fg_ratio", losses)
        self.assertEqual(float(losses["clean_target_fg_ratio"]), 0.0)

    def test_display_speckle_loss_targets_background(self) -> None:
        import torch

        from ldm_project.one_shot_losses import _student_display_speckle_metrics

        camera = torch.zeros(1, 1, 24, 24, requires_grad=True)
        with torch.no_grad():
            camera[:, :, 8:16, 8:16] = 0.8
            camera[:, :, :6, :6] = 0.4
        foreground = torch.zeros_like(camera)
        foreground[:, :, 8:16, 8:16] = 1.0
        background = 1.0 - foreground

        metrics = _student_display_speckle_metrics(camera, foreground, background)
        self.assertGreater(float(metrics["display_speckle_base"]), 0.0)
        metrics["display_speckle_base"].backward()

        self.assertIsNotNone(camera.grad)
        background_grad = (camera.grad.abs() * background).sum().item()
        foreground_core_grad = camera.grad[:, :, 10:14, 10:14].abs().sum().item()
        self.assertGreater(background_grad, 0.0)
        self.assertLess(foreground_core_grad, background_grad)

    def test_strict_readability_treats_clean_target_metrics_as_diagnostics(self) -> None:
        from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

        good_metrics = {
            "paired_corr": 0.80,
            "paired_dice": 0.75,
            "camera_high_saturation_ratio": 0.01,
            "loss": 0.35,
            "camera_recon_loss": 0.12,
            "camera_foreground_dice": 0.50,
            "camera_saturation_ratio": 0.06,
            "foreground_hole_ratio": 0.05,
            "background_speckle_ratio": 0.06,
            "background_hotspot_ratio": 0.10,
            "background_highfreq_energy": 0.001,
            "readable_contrast": 4.6,
            "slm_tv": 0.35,
            "student_label_correct_ratio": 0.85,
            "student_label_margin_mean": 0.04,
            "clean_foreground_dice": 0.60,
            "clean_fg_hole_ratio": 0.30,
            "clean_target_fg_ratio": 0.33,
        }
        good_score = _one_shot_distill_strict_readability_score(good_metrics)

        poor_metrics = dict(good_metrics)
        poor_metrics["clean_foreground_dice"] = 0.30
        poor_metrics["clean_fg_hole_ratio"] = 0.60
        poor_metrics["clean_target_fg_ratio"] = 0.55
        poor_score = _one_shot_distill_strict_readability_score(poor_metrics)

        self.assertEqual(poor_score, good_score)

    def test_clean_target_fg_ratio_recorded_in_history_keys(self) -> None:
        from ldm_project.one_shot_training_utils import ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS

        self.assertIn("clean_target_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("aux_detail_gate_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("aux_detail_active_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("residual_background_mask_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
