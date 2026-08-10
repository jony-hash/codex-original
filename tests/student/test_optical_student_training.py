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
class OneShotOpticalStudentTrainingTests(unittest.TestCase):
    def test_decoder_only_training_optimizer_excludes_upstream_parameters(self) -> None:
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
    def test_aux_topology_target_transforms_are_deterministic_and_safe(self) -> None:
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
    def test_decoder_trainability_audit_records_updates_and_zero_update_failures(self) -> None:
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
    def test_target_code_modes_are_default_safe_and_shape_stable(self) -> None:
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
