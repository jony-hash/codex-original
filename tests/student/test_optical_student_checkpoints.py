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
class OneShotOpticalStudentCheckpointsTests(unittest.TestCase):
    def test_two_layer_checkpoint_loads_into_four_layer_decoder_with_sliced_phase(self) -> None:
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
    def test_artifact_metadata_sidecar_round_trips_non_tensor_strings(self) -> None:
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
    def test_decoder_bypass_exposes_ols_output_at_decoder_plane(self) -> None:
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
    def test_propagation_only_decoder_outputs_finite_diagnostic_bundle(self) -> None:
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
    def test_ols_calibration_screens_have_finite_gradients(self) -> None:
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
