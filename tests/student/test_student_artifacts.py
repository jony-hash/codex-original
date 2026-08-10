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
class TestStudentArtifacts(unittest.TestCase):
        def _minmax_norm(tensor):
            mn = tensor.amin(dim=(-2, -1), keepdim=True)
            mx = tensor.amax(dim=(-2, -1), keepdim=True)
            return (tensor - mn) / (mx - mn).clamp_min(1e-6)

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

        def test_roi_semantic_losses_request_student_prototypes(self) -> None:
            import torch

            from ldm_project.config import OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_training import _build_student_label_margin_prototypes

            payload = {
                "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "teacher_camera": torch.tensor(
                    [
                        [[[[1.0, 0.0], [0.0, 0.0]]]],
                        [[[[0.0, 1.0], [0.0, 0.0]]]],
                        [[[[0.0, 0.0], [1.0, 0.0]]]],
                        [[[[0.0, 0.0], [0.0, 1.0]]]],
                    ],
                    dtype=torch.float32,
                ).squeeze(1),
            }
            train_cfg = TrainingConfig(
                one_shot=OneShotConfig(
                    num_classes=2,
                    student_roi_prototype_ce_weight=0.1,
                    student_roi_negative_label_margin_weight=0.2,
                    student_label_margin_prototype_source="teacher_camera",
                    checkpoint_selection_metric="loss",
                )
            )

            prototypes, source = _build_student_label_margin_prototypes(torch, payload, train_cfg)

            self.assertEqual(source, "teacher_camera")
            self.assertEqual(tuple(prototypes.shape), (2, 1, 2, 2))
            self.assertTrue(torch.allclose(prototypes[0], payload["teacher_camera"][:2].mean(dim=0)))
            self.assertTrue(torch.allclose(prototypes[1], payload["teacher_camera"][2:].mean(dim=0)))

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
            self.assertFalse(psf_identity_outputs["camera_field_available"])
            self.assertFalse(psf_identity_outputs["camera_phase_available"])
            self.assertNotIn("camera_field_real", psf_identity_outputs)
            self.assertNotIn("camera_field_imag", psf_identity_outputs)
            self.assertNotIn("camera_phase", psf_identity_outputs)
            self.assertLess(
                total_variation(blurred_outputs["camera_plane"]).item(),
                total_variation(psf_identity_outputs["camera_plane"]).item(),
            )
            self.assertEqual(
                load_training_config(Path("experiments/archive/pre_v56/configs/audit_one_shot_d2nn_v9_8_back_relay_psf.yaml")).physics.back_path_model,
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

        def test_run_manager_tensor_grid_defaults_to_nearest_interpolation(self) -> None:
            import importlib.util
            from unittest.mock import patch

            import torch

            if importlib.util.find_spec("matplotlib") is None:
                raise unittest.SkipTest("matplotlib is required for PNG grid output.")

            import matplotlib.axes

            from ldm_project.run_manager import RunManager

            calls = []
            original_imshow = matplotlib.axes.Axes.imshow

            def record_imshow(axis, *args, **kwargs):
                calls.append(kwargs.get("interpolation"))
                return original_imshow(axis, *args, **kwargs)

            with tempfile.TemporaryDirectory() as tmpdir:
                run_manager = RunManager(Path(tmpdir))
                with patch.object(matplotlib.axes.Axes, "imshow", record_imshow):
                    run_manager.save_tensor_grid(
                        {"grid": torch.rand(1, 1, 4, 4)},
                        run_manager.images_dir / "nearest_grid.png",
                        max_items=1,
                    )

            self.assertEqual(calls, ["nearest"])

        def test_one_shot_neuron_lattice_amplitude_tensors_use_grid_cells(self) -> None:
            import torch

            from ldm_project.one_shot_visualization import build_one_shot_neuron_lattice_amplitude_tensors

            outputs = _dummy_one_shot_outputs(batch_size=2, image_size=16, grid_size=8)
            tensors, metadata = build_one_shot_neuron_lattice_amplitude_tensors(outputs, pixel_size=2.0e-6)

            self.assertEqual(metadata["view"], "neuron_lattice_amplitude")
            self.assertEqual(metadata["grid_size"], 8)
            self.assertEqual(metadata["pixel_size_um"], 2.0)
            self.assertEqual(metadata["display_interpolation"], "nearest")
            self.assertEqual(
                list(tensors),
                [
                    "SLM command/control grid",
                    "SLM amplitude grid |E|",
                    "encoder input grid |E|",
                    "encoder out |E|",
                    "decoder in |E|",
                    "decoder out |E|",
                    "chip |E|",
                ],
            )
            for tensor in tensors.values():
                self.assertEqual(tuple(tensor.shape), (2, 1, 8, 8))
                self.assertTrue(torch.all(tensor >= 0.0).item())
                self.assertTrue(torch.all(tensor <= 1.0).item())
            self.assertEqual(
                metadata["columns"]["SLM command/control grid"]["source_policy"],
                "area_resampled_to_lattice",
            )
            self.assertEqual(
                metadata["columns"]["encoder input grid |E|"]["source_policy"],
                "raw_lattice",
            )

        def test_one_shot_slim_cell_chain_preview_uses_requested_columns(self) -> None:
            import torch

            from ldm_project.one_shot_visualization import build_one_shot_slim_cell_chain_preview_tensors

            outputs = _dummy_one_shot_outputs(batch_size=2, image_size=16, grid_size=8)
            tensors, metadata = build_one_shot_slim_cell_chain_preview_tensors(outputs, pixel_size=2.0e-6)

            expected_columns = [
                "SLM command",
                "SLM out amp",
                "Encoder in amp",
                "Encoder out amp",
                "Decoder in amp",
                "Decoder out amp",
                "CMOS intensity",
            ]
            self.assertEqual(list(tensors), expected_columns)
            self.assertEqual(metadata["render_mode"], "v4_style_slim_amplitude_preview")
            self.assertEqual(metadata["columns"], expected_columns)
            self.assertEqual(metadata["grid_size"], 8)
            self.assertEqual(metadata["cell_pitch_um"], 2.0)
            self.assertEqual(metadata["display_interpolation"], "nearest")
            self.assertEqual(metadata["panel_style"], "v4_epoch_amplitude")
            self.assertEqual(metadata["column_metadata"]["SLM command"]["source_policy"], "area_resampled_to_lattice")
            self.assertEqual(metadata["column_metadata"]["SLM out amp"]["source_policy"], "area_resampled_to_lattice")
            self.assertEqual(metadata["column_metadata"]["Encoder in amp"]["source_policy"], "raw_lattice")
            self.assertEqual(metadata["column_metadata"]["Encoder out amp"]["source_policy"], "raw_lattice")
            self.assertEqual(metadata["column_metadata"]["Decoder in amp"]["source_policy"], "raw_lattice")
            self.assertEqual(metadata["column_metadata"]["Decoder out amp"]["source_policy"], "raw_lattice")
            self.assertEqual(
                metadata["column_metadata"]["CMOS intensity"]["source_policy"],
                "area_resampled_camera_to_lattice",
            )
            self.assertEqual(metadata["column_metadata"]["CMOS intensity"]["value_kind"], "intensity")
            for name, tensor in tensors.items():
                self.assertEqual(tuple(tensor.shape), (2, 1, 8, 8))
                self.assertTrue(torch.all(tensor >= 0.0).item(), name)
                self.assertTrue(torch.all(tensor <= 1.0).item(), name)

            offline_bundle = dict(outputs)
            offline_bundle["generated_camera_raw"] = offline_bundle.pop("camera_plane")
            offline_tensors, offline_metadata = build_one_shot_slim_cell_chain_preview_tensors(
                offline_bundle,
                pixel_size=2.0e-6,
            )
            self.assertEqual(list(offline_tensors), expected_columns)
            self.assertEqual(offline_metadata["column_metadata"]["CMOS intensity"]["source_tensor"], "generated_camera_raw")

        def test_one_shot_slim_cell_chain_preview_splits_selected_readout_from_physical_cmos(self) -> None:
            import torch

            from ldm_project.one_shot_visualization import build_one_shot_slim_cell_chain_preview_tensors

            outputs = _dummy_one_shot_outputs(batch_size=2, image_size=16, grid_size=8)
            outputs["primary_readout_plane"] = "chip_intensity"
            outputs["selected_readout"] = torch.ones(2, 1, 16, 16) * 0.9
            outputs["camera_plane"] = outputs["selected_readout"]
            outputs["physical_camera_plane"] = torch.ones(2, 1, 16, 16) * 0.2

            tensors, metadata = build_one_shot_slim_cell_chain_preview_tensors(outputs, pixel_size=2.0e-6)

            self.assertEqual(
                list(tensors),
                [
                    "SLM command",
                    "SLM out amp",
                    "Encoder in amp",
                    "Encoder out amp",
                    "Decoder in amp",
                    "Decoder out amp",
                    "Selected readout (chip_intensity)",
                    "Physical CMOS intensity",
                ],
            )
            self.assertNotIn("CMOS intensity", tensors)
            self.assertEqual(metadata["primary_readout_plane"], "chip_intensity")
            self.assertEqual(metadata["physical_cmos_source_tensor"], "physical_camera_plane")
            self.assertEqual(
                metadata["column_metadata"]["Selected readout (chip_intensity)"]["source_tensor"],
                "selected_readout",
            )
            self.assertEqual(
                metadata["column_metadata"]["Physical CMOS intensity"]["source_tensor"],
                "physical_camera_plane",
            )
            self.assertEqual(tuple(tensors["Selected readout (chip_intensity)"].shape), (2, 1, 8, 8))
            self.assertEqual(tuple(tensors["Physical CMOS intensity"].shape), (2, 1, 8, 8))

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

                def __call__(self, codes, labels, *, return_optical_fields=False):
                    outputs = diagnostic_outputs(int(codes.shape[0]), 4, 2)
                    if return_optical_fields:
                        outputs["optical_fields"] = {}
                    return outputs

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
                    "slm_phase_display",
                    "chip_field_amplitude_display",
                    "encoder_input_amplitude_display",
                    "ols_output_phase_display",
                    "decoder_output_phase_display",
                ):
                    self.assertNotIn(key, run_manager.bundle_payload)
                intensity_call = next(call for call in run_manager.grid_calls if call[0] == "one_shot_sample_intensity.png")
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
                self.assertIn("chip I", intensity_call[1])
                self.assertIn("encoder field after L1 I", intensity_call[1])
                self.assertEqual(tuple(intensity_call[1]["encoder field after L1 I"].shape), (8, 1, 2, 2))
                self.assertFalse(any("mask amplitude" in key for key in intensity_call[1]))
                self.assertFalse(any("_d2nn_" in call[0] for call in run_manager.grid_calls))
                self.assertFalse(any(call[0] == "one_shot_sample_amplitude.png" for call in run_manager.grid_calls))
                self.assertFalse(any(call[0] == "one_shot_sample_phase.png" for call in run_manager.grid_calls))
                self.assertNotIn("one_shot_sample_neuron_lattice_amplitude", run_manager.artifact_payloads)
                self.assertNotIn("one_shot_sample_100x100_cell_chain_preview", run_manager.artifact_payloads)
                provenance = run_manager.artifact_payloads["one_shot_sample_optical_field_provenance"]
                self.assertEqual(provenance["status"], "harness-invalid")
                self.assertFalse(provenance["full_chain_available"])
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
                self.assertNotIn("one_shot_sample_shared_scale_diagnostics", run_manager.artifact_payloads)
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

        def test_one_shot_distill_eval_artifact_preserves_raw_diagnostics_and_strict_field_provenance(self) -> None:
            import torch
            from unittest.mock import patch

            from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_eval import run_one_shot_distill_eval_stage

            def diagnostic_outputs(batch_size: int, image_size: int, grid_size: int) -> dict[str, torch.Tensor]:
                return _dummy_one_shot_outputs(batch_size=batch_size, image_size=image_size, grid_size=grid_size)

            class DummyOneShot:
                def eval(self) -> None:
                    pass

                def __call__(self, codes, labels, *, return_optical_fields=False):
                    outputs = diagnostic_outputs(int(codes.shape[0]), 4, 2)
                    if return_optical_fields:
                        outputs["optical_fields"] = {}
                    return outputs

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
                    "slm_phase_display",
                    "chip_field_amplitude_display",
                    "encoder_input_phase_display",
                    "ols_output_amplitude_display",
                    "decoder_output_amplitude_display",
                ):
                    self.assertNotIn(key, run_manager.bundle_payload)
                intensity_call = next(
                    call for call in run_manager.grid_calls if call[0] == "one_shot_distill_eval_intensity.png"
                )
                self.assertIn("teacher intensity target", intensity_call[1])
                self.assertIn("student camera intensity", intensity_call[1])
                self.assertFalse(any("mask" in key for key in intensity_call[1]))
                self.assertFalse(any(call[0] == "one_shot_distill_eval_amplitude.png" for call in run_manager.grid_calls))
                self.assertFalse(any(call[0] == "one_shot_distill_eval_phase.png" for call in run_manager.grid_calls))
                provenance = run_manager.artifact_payloads["one_shot_distill_eval_optical_field_provenance"]
                self.assertEqual(provenance["status"], "harness-invalid")
                self.assertFalse(provenance["full_chain_available"])
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
                self.assertNotIn("one_shot_distill_eval_neuron_lattice_amplitude", run_manager.artifact_payloads)
                self.assertNotIn("one_shot_distill_eval_100x100_cell_chain_preview", run_manager.artifact_payloads)
                eval_d2nn_diagnostics = run_manager.artifact_payloads["one_shot_distill_eval_d2nn_diagnostics"]
                self.assertEqual(eval_d2nn_diagnostics["context"]["stage"], "one_shot_distill_eval")
                self.assertEqual(eval_d2nn_diagnostics["context"]["visual_preview_labels"], [0, 0, 1, 1])
                self.assertTrue(eval_d2nn_diagnostics["available"])
                self.assertIn("encoder field after L1", eval_d2nn_diagnostics["layers"]["amplitude"])
                self.assertIn("decoder mask phase arg(t) L2", eval_d2nn_diagnostics["masks"]["phase"])
                self.assertNotIn("one_shot_distill_eval_shared_scale_diagnostics", run_manager.artifact_payloads)
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
