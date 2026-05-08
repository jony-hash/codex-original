from __future__ import annotations

import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for model smoke tests")
class OpticalPathVAESmokeTests(unittest.TestCase):
    @staticmethod
    def _flatten_correlation(left, right):
        left_flat = left.flatten(1)
        right_flat = right.flatten(1)
        left_centered = left_flat - left_flat.mean(dim=1, keepdim=True)
        right_centered = right_flat - right_flat.mean(dim=1, keepdim=True)
        numerator = (left_centered * right_centered).sum(dim=1)
        denominator = (
            left_centered.pow(2).sum(dim=1).sqrt() * right_centered.pow(2).sum(dim=1).sqrt()
        ).clamp_min(1e-6)
        return numerator / denominator

    @staticmethod
    def _second_moment(image):
        import torch

        height, width = image.shape[-2:]
        y = torch.arange(height, dtype=image.dtype, device=image.device) - (height - 1) / 2.0
        x = torch.arange(width, dtype=image.dtype, device=image.device) - (width - 1) / 2.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        weights = image.clamp_min(0.0)
        moment = ((xx.pow(2) + yy.pow(2)) * weights).sum()
        return moment / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _total_variation(image):
        diff_y = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
        diff_x = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
        return diff_x + diff_y

    def test_forward_and_decode_shapes(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        x = torch.rand(2, 1, 200, 200)
        outputs = model(x)

        self.assertEqual(outputs["mu"].shape, (2, 4, 25, 25))
        self.assertEqual(outputs["logvar"].shape, (2, 4, 25, 25))
        self.assertEqual(outputs["z"].shape, (2, 4, 25, 25))
        self.assertEqual(outputs["x_hat"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["image_recon"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["camera_plane"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["chip_amplitude"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["chip_phase"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["probe_plane"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["probe_plane_loss"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["camera_plane_loss"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["slm_phase"].shape, (2, 1, 200, 200))
        self.assertTrue(torch.equal(outputs["x_hat"], outputs["camera_plane"]))

        decoded = model.decode(outputs["z"])
        self.assertEqual(decoded.shape, (2, 1, 200, 200))
        self.assertTrue(torch.equal(decoded, model.decode_with_intermediates(outputs["z"])["camera_plane"]))

    def test_dual_path_concat_mode_remains_available(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(
            PhysicsConfig(),
            FirstStageConfig(encoder_fusion_mode="dual_path_concat", primary_output="image"),
        )
        x = torch.rand(2, 1, 200, 200)
        outputs = model(x)

        self.assertTrue(torch.equal(outputs["x_hat"], outputs["image_recon"]))

    def test_gated_dual_path_can_decay_to_optical_only(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(
            PhysicsConfig(),
            FirstStageConfig(
                encoder_fusion_mode="gated_dual_path",
                image_fusion_start_weight=1.0,
                image_fusion_end_weight=0.0,
            ),
        )
        model.set_image_fusion_weight(0.0)
        x = torch.rand(2, 1, 200, 200)
        outputs = model(x)

        self.assertEqual(model.image_fusion_weight, 0.0)
        self.assertEqual(outputs["mu"].shape, (2, 4, 25, 25))

    def test_eval_forward_uses_deterministic_mu_latent(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig()).eval()
        x = torch.rand(2, 1, 200, 200)

        with torch.no_grad():
            first = model(x)
            second = model(x)

        self.assertTrue(torch.allclose(first["z"], first["mu"]))
        self.assertTrue(torch.allclose(first["z"], second["z"]))

    def test_decode_with_intermediates_exposes_digit_and_optical_diagnostics(self) -> None:
        import torch
        import torch.nn as nn

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        latents = torch.randn(2, 4, 25, 25)

        decoded = model.decode_with_intermediates(latents)

        self.assertEqual(decoded["image_recon"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["chip_real"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["chip_imag"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["chip_amplitude"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["chip_phase"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["probe_plane"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["probe_plane_loss"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["camera_plane"].shape, (2, 1, 200, 200))
        self.assertEqual(decoded["camera_plane_loss"].shape, (2, 1, 200, 200))

        for key in (
            "image_recon",
            "chip_real",
            "chip_imag",
            "chip_amplitude",
            "chip_phase",
            "probe_plane",
            "probe_plane_loss",
            "camera_plane",
            "camera_plane_loss",
        ):
            self.assertTrue(torch.isfinite(decoded[key]).all().item(), key)
        self.assertFalse(any(isinstance(module, nn.ConvTranspose2d) for module in model.modules()))

    def test_to_electronic_pattern_and_complex_slm_transfer_follow_formula(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        model = OpticalPathVAE(PhysicsConfig(), FirstStageConfig())
        x = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]], dtype=torch.float32)

        electronic_pattern = model.to_electronic_pattern(x)
        slm_phase = model.to_slm_phase(electronic_pattern)
        slm_transfer = model.to_slm_transfer(electronic_pattern, slm_phase)

        self.assertEqual(electronic_pattern.shape, x.shape)
        self.assertTrue(torch.all(electronic_pattern >= 0.0).item())
        self.assertTrue(torch.all(electronic_pattern <= 1.0).item())
        self.assertTrue(torch.all(electronic_pattern[:, :, :, 1:] >= electronic_pattern[:, :, :, :-1]).item())
        self.assertAlmostEqual(electronic_pattern[0, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(electronic_pattern[0, 0, 1, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(slm_phase[0, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(slm_phase[0, 0, 1, 1].item(), math.pi, places=6)
        self.assertTrue(torch.is_complex(slm_transfer))
        self.assertAlmostEqual(slm_transfer.abs()[0, 0, 0, 0].item(), 0.05, places=6)
        self.assertAlmostEqual(slm_transfer.abs()[0, 0, 1, 1].item(), 1.0, places=6)

    def test_percentile_normalization_is_applied_in_optical_readout(self) -> None:
        import torch

        from ldm_project.optics import normalize_intensity

        field = torch.tensor([[[[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 10.0 + 0.0j]]]])

        max_normalized = normalize_intensity(field, normalize_percentile=100.0)
        percentile_normalized = normalize_intensity(field, normalize_percentile=50.0)

        self.assertAlmostEqual(max_normalized[0, 0, 0, 0].item(), 0.01, places=6)
        self.assertAlmostEqual(percentile_normalized[0, 0, 0, 0].item(), 1.0, places=6)

    def test_camera_plane_stays_probe_aligned_under_default_camera_approximation(self) -> None:
        import torch

        from ldm_project.config import PhysicsConfig
        from ldm_project.optics import BackOpticalPath

        chip_amplitude = torch.zeros(1, 1, 200, 200)
        center_y, center_x = 100, 100
        for yy in range(200):
            for xx in range(200):
                radius = ((yy - center_y) ** 2 + (xx - center_x) ** 2) ** 0.5
                if 20 <= radius <= 30 and yy >= 88:
                    chip_amplitude[0, 0, yy, xx] = 1.0
                if 60 <= yy <= 120 and 80 <= xx <= 92:
                    chip_amplitude[0, 0, yy, xx] = 1.0

        physics = PhysicsConfig()
        outputs = BackOpticalPath(physics)(torch.complex(chip_amplitude, torch.zeros_like(chip_amplitude)))

        probe_corr = self._flatten_correlation(chip_amplitude, outputs["probe_plane"])
        camera_probe_corr = self._flatten_correlation(outputs["probe_plane"], outputs["camera_plane"])
        camera_input_corr = self._flatten_correlation(chip_amplitude, outputs["camera_plane"])

        self.assertGreater(probe_corr.item(), 0.80)
        self.assertGreater(camera_probe_corr.item(), 0.62)
        self.assertGreater(camera_input_corr.item(), 0.60)

    def test_camera_magnification_blend_preserves_monotonic_object_growth(self) -> None:
        import torch

        from ldm_project.config import MacroOpticsConfig, PhysicsConfig, ReadoutConfig
        from ldm_project.optics import BackOpticalPath

        chip_amplitude = torch.zeros(1, 1, 200, 200)
        chip_amplitude[:, :, 92:108, 92:108] = 1.0
        chip_field = torch.complex(chip_amplitude, torch.zeros_like(chip_amplitude))

        small_mag_physics = PhysicsConfig(
            readout=ReadoutConfig(camera_magnification_blend=1.0, camera_blur_gain=0.0),
            macro_optics=MacroOpticsConfig(
                dist_plane_to_lens=0.105,
                f_relay=0.075,
                dist_lens_to_cam=0.18,
            ),
        )
        large_mag_physics = PhysicsConfig(
            readout=ReadoutConfig(camera_magnification_blend=1.0, camera_blur_gain=0.0),
            macro_optics=MacroOpticsConfig(
                dist_plane_to_lens=0.105,
                f_relay=0.075,
                dist_lens_to_cam=0.32,
            ),
        )

        small_camera = BackOpticalPath(small_mag_physics)(chip_field)["camera_plane"]
        large_camera = BackOpticalPath(large_mag_physics)(chip_field)["camera_plane"]

        self.assertGreater(self._second_moment(large_camera).item(), self._second_moment(small_camera).item() * 1.15)

    def test_camera_defocus_blur_reduces_high_frequency_energy(self) -> None:
        import torch

        from ldm_project.config import MacroOpticsConfig, PhysicsConfig, ReadoutConfig
        from ldm_project.optics import BackOpticalPath

        chip_amplitude = torch.zeros(1, 1, 200, 200)
        chip_amplitude[:, :, 80:120, 96:104] = 1.0
        chip_amplitude[:, :, 96:104, 80:120] = 1.0
        chip_field = torch.complex(chip_amplitude, torch.zeros_like(chip_amplitude))

        object_distance = 0.105
        focal_length = 0.075
        focus_distance = 1.0 / ((1.0 / focal_length) - (1.0 / object_distance))

        focused_physics = PhysicsConfig(
            readout=ReadoutConfig(camera_magnification_blend=0.35, camera_blur_gain=4.0),
            macro_optics=MacroOpticsConfig(
                dist_plane_to_lens=object_distance,
                f_relay=focal_length,
                dist_lens_to_cam=focus_distance,
            ),
        )
        defocused_physics = PhysicsConfig(
            readout=ReadoutConfig(camera_magnification_blend=0.35, camera_blur_gain=4.0),
            macro_optics=MacroOpticsConfig(
                dist_plane_to_lens=object_distance,
                f_relay=focal_length,
                dist_lens_to_cam=focus_distance + 0.06,
            ),
        )

        focused_camera = BackOpticalPath(focused_physics)(chip_field)["camera_plane"]
        defocused_camera = BackOpticalPath(defocused_physics)(chip_field)["camera_plane"]

        self.assertLess(self._total_variation(defocused_camera).item(), self._total_variation(focused_camera).item() * 0.85)

    def test_pf2_intensity_multiscale_uses_four_encoder_channels(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import MetasurfaceEncoder, OpticalPathVAE

        config = FirstStageConfig(optical_encoder_input_mode="pf2_intensity_multiscale")
        model = OpticalPathVAE(PhysicsConfig(), config)
        x = torch.rand(2, 1, 200, 200)

        outputs = model(x)

        self.assertIsInstance(model.optical_encoder, MetasurfaceEncoder)
        self.assertEqual(model.optical_encoder.input_channels, 4)
        self.assertEqual(outputs["optical_encoder_input"].shape, (2, 4, 200, 200))
        self.assertEqual(outputs["pf2_intensity"].shape, (2, 1, 200, 200))
        self.assertEqual(outputs["mu"].shape, (2, 4, 25, 25))
        self.assertEqual(outputs["x_hat"].shape, (2, 1, 200, 200))

    def test_global_context_encoder_preserves_forward_shapes(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import GlobalContextMetasurfaceEncoder, OpticalPathVAE

        config = FirstStageConfig(
            optical_encoder_input_mode="pf2_intensity_multiscale",
            optical_encoder_arch="global_context",
        )
        model = OpticalPathVAE(PhysicsConfig(), config)
        x = torch.rand(1, 1, 200, 200)

        outputs = model(x)

        self.assertIsInstance(model.optical_encoder, GlobalContextMetasurfaceEncoder)
        self.assertEqual(model.optical_encoder.input_channels, 4)
        self.assertEqual(outputs["optical_encoder_input"].shape, (1, 4, 200, 200))
        self.assertEqual(outputs["mu"].shape, (1, 4, 25, 25))
        self.assertEqual(outputs["x_hat"].shape, (1, 1, 200, 200))

    def test_pf2_intensity_multiscale_v2_uses_eight_global_context_channels(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import GlobalContextMetasurfaceEncoder, OpticalPathVAE

        config = FirstStageConfig(
            optical_encoder_input_mode="pf2_intensity_multiscale_v2",
            optical_encoder_arch="global_context",
        )
        model = OpticalPathVAE(PhysicsConfig(), config)
        x = torch.rand(1, 1, 200, 200)

        outputs = model(x)

        self.assertIsInstance(model.optical_encoder, GlobalContextMetasurfaceEncoder)
        self.assertEqual(model.optical_encoder.input_channels, 8)
        self.assertEqual(outputs["optical_encoder_input"].shape, (1, 8, 200, 200))
        self.assertEqual(outputs["pf2_intensity"].shape, (1, 1, 200, 200))
        self.assertEqual(outputs["mu"].shape, (1, 4, 25, 25))
        self.assertEqual(outputs["x_hat"].shape, (1, 1, 200, 200))

    def test_angular_spectrum_propagate_preserves_shape_dtype_and_finiteness(self) -> None:
        import torch

        from ldm_project.optics import angular_spectrum_propagate

        real = torch.rand(2, 1, 16, 16)
        imag = torch.rand(2, 1, 16, 16)
        field = torch.complex(real, imag)
        propagated = angular_spectrum_propagate(
            field,
            distance=0.003,
            wavelength=532.0e-9,
            pixel_size=3.0e-6,
        )

        self.assertEqual(propagated.shape, field.shape)
        self.assertEqual(propagated.dtype, field.dtype)
        self.assertTrue(torch.isfinite(propagated.real).all().item())
        self.assertTrue(torch.isfinite(propagated.imag).all().item())

    def test_linear_metasurface_replaces_encoder_ols_decoder_with_linear_layers(self) -> None:
        import torch
        import torch.nn as nn

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import (
            ConfigurableOLS,
            LinearMetasurfaceDecoder,
            LinearMetasurfaceEncoder,
            OpticalPathVAE,
        )

        config = FirstStageConfig(
            image_size=32,
            latent_channels=1,
            optical_encoder_input_mode="pf2_complex_field",
            optical_encoder_arch="linear_metasurface",
            metasurface_decoder_arch="linear_metasurface",
            linear_metasurface_grid_size=8,
            linear_metasurface_layer_spacing=0.003,
            ols_kind="gelu",
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
        )
        model = OpticalPathVAE(PhysicsConfig(), config)
        x = torch.rand(2, 1, 32, 32)

        outputs = model(x)

        self.assertIsInstance(model.optical_encoder, LinearMetasurfaceEncoder)
        self.assertIsInstance(model.ols, ConfigurableOLS)
        self.assertIsInstance(model.metasurface_decoder, LinearMetasurfaceDecoder)
        nonlinear_modules = (
            nn.ELU,
            nn.GELU,
            nn.GroupNorm,
            nn.LeakyReLU,
            nn.ReLU,
            nn.SELU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Tanh,
        )
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in model.optical_encoder.modules()), 1)
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in model.metasurface_decoder.modules()), 1)
        self.assertFalse(any(isinstance(module, nonlinear_modules) for module in model.optical_encoder.modules()))
        self.assertFalse(any(isinstance(module, nonlinear_modules) for module in model.metasurface_decoder.modules()))
        self.assertEqual(model.latent_shape, (1, 4, 4))
        self.assertEqual(outputs["mu"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["logvar"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["z"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["optical_encoder_input"].shape, (2, 2, 32, 32))
        self.assertEqual(outputs["image_recon"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["camera_plane"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["chip_real"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["chip_imag"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.equal(outputs["z"], outputs["mu"]))
        self.assertTrue(torch.equal(outputs["image_recon"], outputs["chip_real"]))
        for key in ("image_recon", "camera_plane", "chip_real", "chip_imag"):
            self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)

    def test_linear_metasurface_requires_single_channel_ols(self) -> None:
        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        config = FirstStageConfig(
            image_size=32,
            latent_channels=2,
            optical_encoder_input_mode="pf2_complex_field",
            optical_encoder_arch="linear_metasurface",
            metasurface_decoder_arch="linear_metasurface",
            image_recon_weight=0.0,
            camera_recon_weight=1.0,
        )

        with self.assertRaisesRegex(ValueError, "latent_channels=1"):
            OpticalPathVAE(PhysicsConfig(), config)

    def test_legacy_amplitude_slm_mode_keeps_zero_phase(self) -> None:
        import torch

        from ldm_project.config import ElectronicDomainConfig, FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE

        config = FirstStageConfig(
            electronic_domain=ElectronicDomainConfig(mode="mnist_amplitude_slm"),
        )
        model = OpticalPathVAE(PhysicsConfig(), config)
        control = torch.tensor([[[[0.0, 0.5], [0.75, 1.0]]]], dtype=torch.float32)

        slm_phase = model.to_slm_phase(control)
        slm_transfer = model.to_slm_transfer(control, slm_phase)

        self.assertTrue(torch.allclose(slm_phase, torch.zeros_like(control)))
        self.assertTrue(torch.allclose(slm_transfer.abs(), model.to_slm_amplitude(control)))


if __name__ == "__main__":
    unittest.main()
