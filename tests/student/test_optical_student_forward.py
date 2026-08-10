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
class OneShotOpticalStudentForwardTests(unittest.TestCase):
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
    def test_diagnostic_switch_defaults_preserve_forward_outputs(self) -> None:
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
    def test_optical_field_capture_preserves_real_ols_to_decoder_continuity(self) -> None:
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
    def test_decoder_modulation_override_only_adds_decoder_amplitude(self) -> None:
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
    def test_decoder_diagnostic_mask_component_applies_only_selected_decoder_component(self) -> None:
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
    def test_decoder_diagnostic_phase_layer_mask_selects_decoder_layers_only(self) -> None:
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
