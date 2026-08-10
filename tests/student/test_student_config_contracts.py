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
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class TestStudentConfigContracts(unittest.TestCase):
        def _minmax_norm(tensor):
            mn = tensor.amin(dim=(-2, -1), keepdim=True)
            mx = tensor.amax(dim=(-2, -1), keepdim=True)
            return (tensor - mn) / (mx - mn).clamp_min(1e-6)

        def test_student_prototype_ce_config_defaults_and_validation(self) -> None:
            from ldm_project.config import OneShotConfig, load_training_config

            defaults = OneShotConfig()
            self.assertEqual(defaults.student_prototype_ce_weight, 0.0)
            self.assertEqual(defaults.student_prototype_ce_labels, ())
            self.assertAlmostEqual(defaults.student_prototype_ce_temperature, 0.07, places=8)
            self.assertEqual(defaults.student_negative_label_margin_weight, 0.0)
            self.assertEqual(defaults.student_negative_label_margin_config, ())
            self.assertEqual(defaults.student_roi_prototype_ce_weight, 0.0)
            self.assertEqual(defaults.student_roi_prototype_ce_labels, ())
            self.assertAlmostEqual(defaults.student_roi_prototype_ce_temperature, 0.07, places=8)
            self.assertEqual(defaults.student_roi_negative_label_margin_weight, 0.0)
            self.assertEqual(defaults.student_roi_negative_label_margin_config, ())
            self.assertAlmostEqual(defaults.student_roi_foreground_threshold, 0.20, places=8)
            self.assertEqual(defaults.aux_semantic_plane_key, "")
            self.assertEqual(defaults.aux_semantic_weight, 0.0)
            self.assertEqual(defaults.aux_semantic_prototype_ce_weight, 0.0)
            self.assertEqual(defaults.aux_semantic_negative_label_margin_weight, 0.0)
            self.assertEqual(defaults.aux_topology_plane_key, "")
            self.assertEqual(defaults.aux_topology_weight, 0.0)
            self.assertEqual(defaults.aux_topology_dice_weight, 1.0)
            self.assertEqual(defaults.aux_topology_mse_weight, 1.0)
            self.assertEqual(defaults.aux_topology_target_transform, "none")
            self.assertFalse(defaults.decoder_trainability_audit)
            self.assertEqual(defaults.decoder_diagnostic_mode, "normal")
            self.assertEqual(defaults.decoder_diagnostic_modulation_override, "none")
            self.assertEqual(defaults.decoder_diagnostic_mask_component, "trained")
            self.assertEqual(defaults.decoder_diagnostic_phase_layer_mask, ())
            self.assertEqual(defaults.ols_transfer_calibration, "none")
            self.assertEqual(defaults.target_code_diagnostic_mode, "none")
            self.assertEqual(defaults.target_code_bridge_source, "none")
            self.assertEqual(defaults.target_code_bridge_arch, "shallow")
            self.assertEqual(defaults.target_code_bridge_hidden_channels, 8)
            self.assertFalse(defaults.target_code_bridge_trainability_audit)
            self.assertAlmostEqual(defaults.sample_code_diversity_weight, 0.0, places=8)

            with tempfile.TemporaryDirectory() as tmpdir:
                good_path = Path(tmpdir) / "prototype_ce.yaml"
                good_path.write_text(
                    "\n".join(
                        [
                            "one_shot:",
                            "  student_prototype_ce_weight: 0.08",
                            "  student_prototype_ce_labels: [1, 3, 7, 9]",
                            "  student_prototype_ce_temperature: 0.07",
                            "  student_negative_label_margin_weight: 0.5",
                            "  student_negative_label_margin_config:",
                            "    - [3, 5, 0.08]",
                            "    - [7, 9, 0.08]",
                            "  student_roi_prototype_ce_weight: 0.10",
                            "  student_roi_prototype_ce_labels: [0, 3, 4]",
                            "  student_roi_prototype_ce_temperature: 0.09",
                            "  student_roi_negative_label_margin_weight: 0.25",
                            "  student_roi_negative_label_margin_config:",
                            "    - [0, 5, 0.04]",
                            "    - [4, 9, 0.04]",
                            "  student_roi_foreground_threshold: 0.25",
                            "  aux_semantic_plane_key: chip_intensity_primary",
                            "  aux_semantic_weight: 0.40",
                            "  aux_semantic_prototype_ce_weight: 0.20",
                            "  aux_semantic_negative_label_margin_weight: 0.30",
                            "  aux_topology_plane_key: decoder_output_amplitude",
                            "  aux_topology_weight: 0.50",
                            "  aux_topology_dice_weight: 1.25",
                            "  aux_topology_mse_weight: 0.75",
                            "  aux_topology_target_transform: lowpass",
                            "  decoder_trainability_audit: true",
                            "  target_code_diagnostic_mode: lowfreq_digit_code",
                            "  target_code_bridge_source: ols_output_amplitude",
                            "  target_code_bridge_arch: deep_residual",
                            "  target_code_bridge_hidden_channels: 4",
                            "  target_code_bridge_trainability_audit: true",
                            "  sample_code_diversity_weight: 0.25",
                        ]
                    ),
                    encoding="utf-8",
                )
                cfg = load_training_config(good_path)
                self.assertAlmostEqual(cfg.one_shot.student_prototype_ce_weight, 0.08, places=8)
                self.assertEqual(cfg.one_shot.student_prototype_ce_labels, (1, 3, 7, 9))
                self.assertEqual(
                    cfg.one_shot.student_negative_label_margin_config,
                    ((3, 5, 0.08), (7, 9, 0.08)),
                )
                self.assertAlmostEqual(cfg.one_shot.student_roi_prototype_ce_weight, 0.10, places=8)
                self.assertEqual(cfg.one_shot.student_roi_prototype_ce_labels, (0, 3, 4))
                self.assertAlmostEqual(cfg.one_shot.student_roi_prototype_ce_temperature, 0.09, places=8)
                self.assertAlmostEqual(cfg.one_shot.student_roi_negative_label_margin_weight, 0.25, places=8)
                self.assertEqual(
                    cfg.one_shot.student_roi_negative_label_margin_config,
                    ((0, 5, 0.04), (4, 9, 0.04)),
                )
                self.assertAlmostEqual(cfg.one_shot.student_roi_foreground_threshold, 0.25, places=8)
                self.assertEqual(cfg.one_shot.aux_semantic_plane_key, "chip_intensity_primary")
                self.assertAlmostEqual(cfg.one_shot.aux_semantic_weight, 0.40, places=8)
                self.assertAlmostEqual(cfg.one_shot.aux_semantic_prototype_ce_weight, 0.20, places=8)
                self.assertAlmostEqual(cfg.one_shot.aux_semantic_negative_label_margin_weight, 0.30, places=8)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, "decoder_output_amplitude")
                self.assertAlmostEqual(cfg.one_shot.aux_topology_weight, 0.50, places=8)
                self.assertAlmostEqual(cfg.one_shot.aux_topology_dice_weight, 1.25, places=8)
                self.assertAlmostEqual(cfg.one_shot.aux_topology_mse_weight, 0.75, places=8)
                self.assertEqual(cfg.one_shot.aux_topology_target_transform, "lowpass")
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "lowfreq_digit_code")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "ols_output_amplitude")
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertEqual(cfg.one_shot.target_code_bridge_hidden_channels, 4)
                self.assertTrue(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertAlmostEqual(cfg.one_shot.sample_code_diversity_weight, 0.25, places=8)

                for plane_key in ("target_code_amplitude", "decoder_input_amplitude"):
                    with self.subTest(aux_topology_plane_key=plane_key):
                        aux_plane_path = Path(tmpdir) / f"aux_{plane_key}.yaml"
                        aux_plane_path.write_text(
                            "\n".join(
                                [
                                    "one_shot:",
                                    f"  aux_topology_plane_key: {plane_key}",
                                    "  aux_topology_weight: 0.5",
                                ]
                            ),
                            encoding="utf-8",
                        )
                        aux_cfg = load_training_config(aux_plane_path)
                        self.assertEqual(aux_cfg.one_shot.aux_topology_plane_key, plane_key)

                bad_label_path = Path(tmpdir) / "bad_label.yaml"
                bad_label_path.write_text(
                    "\n".join(["one_shot:", "  student_prototype_ce_labels: [10]"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_prototype_ce_labels"):
                    load_training_config(bad_label_path)

                bad_tuple_path = Path(tmpdir) / "bad_tuple.yaml"
                bad_tuple_path.write_text(
                    "\n".join(
                        [
                            "one_shot:",
                            "  student_negative_label_margin_config:",
                            "    - [3, 5]",
                        ]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_negative_label_margin_config entries"):
                    load_training_config(bad_tuple_path)

                bad_weight_path = Path(tmpdir) / "bad_weight.yaml"
                bad_weight_path.write_text(
                    "\n".join(["one_shot:", "  student_negative_label_margin_weight: -0.1"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_negative_label_margin_weight"):
                    load_training_config(bad_weight_path)

                bad_roi_label_path = Path(tmpdir) / "bad_roi_label.yaml"
                bad_roi_label_path.write_text(
                    "\n".join(["one_shot:", "  student_roi_prototype_ce_labels: [-1]"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_roi_prototype_ce_labels"):
                    load_training_config(bad_roi_label_path)

                bad_roi_tuple_path = Path(tmpdir) / "bad_roi_tuple.yaml"
                bad_roi_tuple_path.write_text(
                    "\n".join(
                        [
                            "one_shot:",
                            "  student_roi_negative_label_margin_config:",
                            "    - [3, 5]",
                        ]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_roi_negative_label_margin_config entries"):
                    load_training_config(bad_roi_tuple_path)

                bad_roi_weight_path = Path(tmpdir) / "bad_roi_weight.yaml"
                bad_roi_weight_path.write_text(
                    "\n".join(["one_shot:", "  student_roi_prototype_ce_weight: -0.1"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "student_roi_prototype_ce_weight"):
                    load_training_config(bad_roi_weight_path)

                bad_aux_plane_path = Path(tmpdir) / "bad_aux_plane.yaml"
                bad_aux_plane_path.write_text(
                    "\n".join(["one_shot:", "  aux_semantic_plane_key: not_a_plane"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_semantic_plane_key"):
                    load_training_config(bad_aux_plane_path)

                bad_aux_blank_path = Path(tmpdir) / "bad_aux_blank.yaml"
                bad_aux_blank_path.write_text(
                    "\n".join(
                        [
                            "one_shot:",
                            "  aux_semantic_weight: 0.5",
                            "  aux_semantic_prototype_ce_weight: 1.0",
                        ]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_semantic_plane_key"):
                    load_training_config(bad_aux_blank_path)

                bad_aux_weight_path = Path(tmpdir) / "bad_aux_weight.yaml"
                bad_aux_weight_path.write_text(
                    "\n".join(["one_shot:", "  aux_semantic_weight: -0.1"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_semantic_weight"):
                    load_training_config(bad_aux_weight_path)

                bad_aux_topology_plane_path = Path(tmpdir) / "bad_aux_topology_plane.yaml"
                bad_aux_topology_plane_path.write_text(
                    "\n".join(["one_shot:", "  aux_topology_plane_key: not_a_plane"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_topology_plane_key"):
                    load_training_config(bad_aux_topology_plane_path)

                bad_aux_topology_blank_path = Path(tmpdir) / "bad_aux_topology_blank.yaml"
                bad_aux_topology_blank_path.write_text(
                    "\n".join(["one_shot:", "  aux_topology_weight: 0.5"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_topology_plane_key"):
                    load_training_config(bad_aux_topology_blank_path)

                bad_aux_topology_weight_path = Path(tmpdir) / "bad_aux_topology_weight.yaml"
                bad_aux_topology_weight_path.write_text(
                    "\n".join(["one_shot:", "  aux_topology_weight: -0.1"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_topology_weight"):
                    load_training_config(bad_aux_topology_weight_path)

                bad_aux_topology_transform_path = Path(tmpdir) / "bad_aux_topology_transform.yaml"
                bad_aux_topology_transform_path.write_text(
                    "\n".join(["one_shot:", "  aux_topology_target_transform: sharpen"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "aux_topology_target_transform"):
                    load_training_config(bad_aux_topology_transform_path)

                bad_decoder_mode_path = Path(tmpdir) / "bad_decoder_mode.yaml"
                bad_decoder_mode_path.write_text(
                    "\n".join(["one_shot:", "  decoder_diagnostic_mode: not_a_mode"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "decoder_diagnostic_mode"):
                    load_training_config(bad_decoder_mode_path)

                bad_decoder_modulation_override_path = Path(tmpdir) / "bad_decoder_modulation_override.yaml"
                bad_decoder_modulation_override_path.write_text(
                    "\n".join(["one_shot:", "  decoder_diagnostic_modulation_override: phase_only"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "decoder_diagnostic_modulation_override"):
                    load_training_config(bad_decoder_modulation_override_path)

                bad_decoder_mask_component_path = Path(tmpdir) / "bad_decoder_mask_component.yaml"
                bad_decoder_mask_component_path.write_text(
                    "\n".join(["one_shot:", "  decoder_diagnostic_mask_component: not_a_component"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "decoder_diagnostic_mask_component"):
                    load_training_config(bad_decoder_mask_component_path)

                bad_decoder_mask_mode_path = Path(tmpdir) / "bad_decoder_mask_mode.yaml"
                bad_decoder_mask_mode_path.write_text(
                    "\n".join(
                        [
                            "one_shot:",
                            "  decoder_diagnostic_mode: propagation_only_decoder",
                            "  decoder_diagnostic_modulation_override: phase_amplitude",
                            "  decoder_diagnostic_mask_component: phase_only",
                        ]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "decoder_diagnostic_mode=normal"):
                    load_training_config(bad_decoder_mask_mode_path)

                bad_decoder_mask_modulation_path = Path(tmpdir) / "bad_decoder_mask_modulation.yaml"
                bad_decoder_mask_modulation_path.write_text(
                    "\n".join(["one_shot:", "  decoder_diagnostic_mask_component: amplitude_only"]),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "phase_amplitude"):
                    load_training_config(bad_decoder_mask_modulation_path)

        def test_smoke_config_loads_one_shot_fields(self) -> None:
            from ldm_project.config import load_training_config

            d2nn_config = load_training_config(Path("configs/v1/smoke/train_one_shot_d2nn_smoke.yaml"))

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
            self.assertEqual(d2nn_config.one_shot.encoder_input_topology_weight, 0.0)
            self.assertEqual(d2nn_config.one_shot.encoder_input_topology_dice_weight, 1.0)
            self.assertEqual(d2nn_config.one_shot.encoder_input_topology_mse_weight, 1.0)

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

        def test_twophoton_100x100_full_connect_config_and_shapes(self) -> None:
            import torch

            from ldm_project.config import PhysicsConfig, load_training_config
            from ldm_project.one_shot import OneShotOpticalGenerator

            train_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_"
                    "twophoton_100x100_2um_full_connect.yaml"
                )
            )
            eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_"
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
            self.assertEqual(train_config.one_shot.eval_samples_per_class, 5)
            self.assertEqual(eval_config.one_shot.eval_samples_per_class, 10)
            self.assertFalse(eval_config.one_shot.eval_allow_train_fallback)

        def test_teacher_distillation_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            export_3k_fast200_config = load_training_config(Path("configs/v1/teacher/export_teacher_distill_mnist_3k_fast200.yaml"))
            export_15k_fast200_config = load_training_config(Path("configs/v1/teacher/export_teacher_distill_mnist_15k_fast200.yaml"))
            filter_config = load_training_config(Path("configs/v1/teacher/filter_teacher_distill_mnist_3k_fast200_top2k.yaml"))
            strict_filter_config = load_training_config(
                Path("configs/v1/teacher/filter_teacher_distill_mnist_3k_fast200_label_strict_top1k.yaml")
            )
            readable_split_config = load_training_config(
                Path("configs/v1/teacher/split_teacher_distill_readable_top1k_80_10_10.yaml")
            )

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

        def test_optical_audit_and_relay_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            identity_audit_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/audit_one_shot_d2nn_v8_7a4_optical_identity_chain.yaml")
            )
            readout_ablation_audit_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/audit_one_shot_d2nn_v9_2_readout_plane_ablation.yaml")
            )
            student_v9_8_relay_psf_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_relay_psf_camera_primary.yaml"
                )
            )
            student_v9_8_relay_psf_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_relay_psf_camera_primary_strict_readability_test.yaml"
                )
            )

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

        def test_first_stage_recovery_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            hardclass_recovery_config = load_training_config(
                Path("configs/v1/teacher/train_first_stage_hardclass_camera_recovery.yaml")
            )
            margin_recovery_config = load_training_config(
                Path("configs/v1/teacher/train_first_stage_hardclass_camera_recovery_margin.yaml")
            )
            acceptance_margin_config = load_training_config(
                Path("configs/v1/teacher/train_first_stage_acceptance_margin_recovery.yaml")
            )

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

        def test_one_shot_v3_to_v6_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            train_config = load_training_config(Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml"))
            strict_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v3_strict_readability.yaml")
            )
            student_v3_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v3.yaml")
            )
            student_v4_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v4_readability.yaml")
            )
            student_v5_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v5_speckle_label.yaml")
            )
            student_v6_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v6_clean_bg.yaml")
            )

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

        def test_one_shot_v7_and_v8_baseline_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            student_v7_1_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_1_clean_target_fixed.yaml")
            )
            student_v7_3_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_3_denoise.yaml")
            )
            student_v7_3_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v7_3_strict_readability_test.yaml")
            )
            student_v8_identity_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_identity_oracle.yaml")
            )
            student_v8_shallow_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn.yaml")
            )
            student_v8_identity_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_identity_strict_readability_test.yaml")
            )
            student_v8_shallow_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_strict_readability_test.yaml")
            )
            student_v8_shallow_fair_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_fair.yaml")
            )
            student_v8_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_direct_clean_oracle.yaml")
            )
            student_v8_direct_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_direct_clean_oracle_strict_readability_test.yaml")
            )

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

        def test_one_shot_v8_direct_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            student_v8_1_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1_direct_mnist_prototype_oracle.yaml")
            )
            student_v8_1b_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1b_direct_mnist_prototype_oracle_focus1.yaml")
            )
            student_v8_2_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_2_direct_mnist_prototype_oracle_detail_polish.yaml")
            )
            student_v8_2c_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_2c_direct_mnist_prototype_oracle_margin_finish.yaml")
            )
            student_v8_3_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_3_direct_mnist_sample_oracle.yaml")
            )
            student_v8_4_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_4_direct_mnist_nearest_teacher_image_oracle.yaml")
            )
            student_v8_5a_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5a_direct_prototype_residual_safety.yaml")
            )
            student_v8_5b_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5b_direct_prototype_anchor_aux_detail.yaml")
            )
            student_v8_5c_direct_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5c_direct_prototype_residual_probe.yaml")
            )
            student_v8_5d_direct_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5d_direct_prototype_residual_foreground_gate.yaml"
                )
            )
            student_v8_1_direct_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_1_direct_mnist_prototype_oracle_strict_readability_test.yaml")
            )
            student_v8_3_direct_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_3_direct_mnist_sample_oracle_strict_readability_test.yaml")
            )
            student_v8_4_direct_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_4_direct_mnist_nearest_teacher_image_oracle_strict_readability_test.yaml")
            )
            student_v8_5c_direct_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5c_direct_prototype_residual_probe_strict_readability_test.yaml")
            )
            student_v8_5d_direct_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_5d_direct_prototype_residual_foreground_gate_strict_readability_test.yaml"
                )
            )

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

        def test_one_shot_v8_diffractive_and_v9_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            student_v8_6a_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6a_phase2x2_prototype_anchor.yaml")
            )
            student_v8_6b_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6b_phase2x2_random_phase_prototype_anchor.yaml"
                )
            )
            student_v8_7a_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a_phase_amplitude2x2_prototype_anchor.yaml"
                )
            )
            student_v8_7a2_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a2_phase_amplitude_warm_phase_active_amp.yaml"
                )
            )
            student_v8_7a3_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a3_near_identity_phase_amp_feasibility.yaml"
                )
            )
            student_v9_1_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_1_trainable_generator_phase_amp_feasibility.yaml"
                )
            )
            student_v8_6a_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6a_phase2x2_prototype_anchor_strict_readability_test.yaml"
                )
            )
            student_v8_6b_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_6b_phase2x2_random_phase_prototype_anchor_strict_readability_test.yaml"
                )
            )
            student_v8_7a_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a_phase_amplitude2x2_prototype_anchor_strict_readability_test.yaml"
                )
            )
            student_v8_7a2_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a2_phase_amplitude_warm_phase_active_amp_strict_readability_test.yaml"
                )
            )
            student_v8_7a3_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_7a3_near_identity_phase_amp_feasibility_strict_readability_test.yaml"
                )
            )
            student_v9_1_eval_config = load_training_config(
                Path(
                    "experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_1_trainable_generator_phase_amp_feasibility_strict_readability_test.yaml"
                )
            )

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

        def test_one_shot_feasibility_and_long_configs_load(self) -> None:
            from ldm_project.config import load_training_config

            student_v8_shallow_fair_eval_config = load_training_config(
                Path("experiments/archive/pre_v56/configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v8_shallow_d2nn_fair_strict_readability_test.yaml")
            )
            feasibility_config = load_training_config(Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_gpu_feasibility.yaml"))
            long_config = load_training_config(Path("experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_long.yaml"))

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
