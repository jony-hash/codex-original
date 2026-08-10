# ruff: noqa: F401,F403,F405,F811
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings
from tests.tests_support import *  # noqa: F403
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for train main smoke tests")
class TrainScriptStepTests(unittest.TestCase):
    def _assert_twophoton_d2nn_contract(
        self,
        config,
        *,
        decoder_layers: int = 2,
        layer_spacing: float = 1.50e-3,
        propagation_grid_size: int | None = None,
        modulation: str = "phase_only",
    ) -> None:
        d2nn = config.one_shot.d2nn
        self.assertIsNotNone(d2nn)
        assert d2nn is not None
        self.assertEqual(d2nn.grid_size, 100)
        if propagation_grid_size is not None:
            self.assertEqual(d2nn.propagation_grid_size, propagation_grid_size)
        self.assertAlmostEqual(d2nn.pixel_size, 2.0e-6)
        self.assertEqual(d2nn.encoder_layers, 2)
        self.assertEqual(d2nn.decoder_layers, decoder_layers)
        self.assertAlmostEqual(d2nn.layer_spacing, layer_spacing)
        self.assertEqual(d2nn.modulation, modulation)

    def test_output_layout_extracts_version_from_run_name_before_config(self) -> None:
        from ldm_project.output_layout import (
            earliest_run_timestamp,
            extract_output_version,
            resolve_output_layout,
            resolve_version_dir,
            version_directory_name,
            version_sort_key,
        )

        self.assertEqual(extract_output_version("student_v8_7a2_phase_amp"), "v8_7a2")
        self.assertEqual(extract_output_version("v9.10_label_template"), "v9_10")
        explicit = resolve_output_layout(
            "/tmp/manual-root",
            run_name=None,
            train_config="configs/train_student_v7.yaml",
        )
        self.assertEqual(explicit.version, "v7")
        self.assertEqual(explicit.output_root, Path("/tmp/manual-root"))
        self.assertTrue(explicit.explicit)
        versions = ["v10", "v9_10", "v9_8", "v8_7a2", "v8_7a", "v2", "unversioned"]
        self.assertEqual(
            sorted(versions, key=version_sort_key),
            ["unversioned", "v2", "v8_7a", "v8_7a2", "v9_8", "v9_10", "v10"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "outputs"
            version_dir = output_base / "20260620_v9_8" / "runs"
            version_dir.mkdir(parents=True)
            (version_dir / "20260620_102234_train_one_shot_d2nn_v9_8_audit").mkdir()
            (version_dir / "20260620_102936_train_one_shot_d2nn_v9_8_eval").mkdir()
            self.assertEqual(earliest_run_timestamp(output_base / "20260620_v9_8"), "20260620_102234")
            self.assertEqual(version_directory_name("v9_8", date_prefix="20260620"), "20260620_v9_8")
            self.assertEqual(resolve_version_dir("v9_8", output_base=output_base), output_base / "20260620_v9_8")
            self.assertEqual(
                resolve_output_layout(
                    None,
                    run_name="manual_v9_8_eval",
                    train_config="configs/train_student_v7.yaml",
                    output_base=output_base,
                ).output_root,
                output_base / "20260620_v9_8" / "runs",
            )

    def test_resolve_logged_step_resets_global_step_per_train_round(self) -> None:
        resolve_logged_step = load_train_utils_module().resolve_logged_step

        first_epoch_last = resolve_logged_step(batch_index=625, stage_step=625)
        next_epoch_first = resolve_logged_step(batch_index=0, stage_step=626)

        self.assertEqual(first_epoch_last["global_step"], 625)
        self.assertEqual(first_epoch_last["stage_step"], 625)
        self.assertEqual(next_epoch_first["global_step"], 0)
        self.assertEqual(next_epoch_first["stage_step"], 626)

    def test_resolve_logged_step_rejects_negative_values(self) -> None:
        resolve_logged_step = load_train_utils_module().resolve_logged_step

        with self.assertRaises(ValueError):
            resolve_logged_step(batch_index=-1, stage_step=0)

        with self.assertRaises(ValueError):
            resolve_logged_step(batch_index=0, stage_step=-1)

    def test_resolve_stage_epoch_plan_matches_requested_mode(self) -> None:
        resolve_stage_epoch_plan = load_train_utils_module().resolve_stage_epoch_plan

        self.assertEqual(
            resolve_stage_epoch_plan(stage="full", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 10, "diffusion": 20, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(stage="first_stage", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 10, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(stage="diffusion", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 0, "diffusion": 20, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(stage="sample", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(stage="teacher_distill_export", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(stage="teacher_distill_filter", first_stage_epochs=10, diffusion_epochs=20),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="teacher_first_stage_oracle_audit",
                first_stage_epochs=10,
                diffusion_epochs=20,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="teacher_diffusion_latent_audit",
                first_stage_epochs=10,
                diffusion_epochs=20,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="one_shot_distill_eval",
                first_stage_epochs=10,
                diffusion_epochs=20,
                one_shot_epochs=4,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="one_shot_distill_eval_diagnostics",
                first_stage_epochs=10,
                diffusion_epochs=20,
                one_shot_epochs=4,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="one_shot_optical_identity_audit",
                first_stage_epochs=10,
                diffusion_epochs=20,
                one_shot_epochs=4,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 0},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="one_shot",
                first_stage_epochs=10,
                diffusion_epochs=20,
                one_shot_epochs=3,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 3},
        )
        self.assertEqual(
            resolve_stage_epoch_plan(
                stage="one_shot_distill",
                first_stage_epochs=10,
                diffusion_epochs=20,
                one_shot_epochs=4,
            ),
            {"first_stage": 0, "diffusion": 0, "one_shot": 4},
        )

        with self.assertRaises(ValueError):
            resolve_stage_epoch_plan(stage="full", first_stage_epochs=-1, diffusion_epochs=20)

        with self.assertRaises(ValueError):
            resolve_stage_epoch_plan(stage="unknown", first_stage_epochs=10, diffusion_epochs=20)

    def test_resolve_image_fusion_weight_decays_for_gated_mode(self) -> None:
        from ldm_project.config import FirstStageConfig

        train_utils_module = load_train_utils_module()
        evaluation_module = load_evaluation_module()
        compute_first_stage_balanced_metrics = evaluation_module.compute_first_stage_balanced_metrics
        is_best_optical_candidate = train_utils_module.is_best_optical_candidate
        resolve_image_fusion_weight = train_utils_module.resolve_image_fusion_weight
        should_update_best_optical = train_utils_module.should_update_best_optical
        config = FirstStageConfig(
            encoder_fusion_mode="gated_dual_path",
            image_fusion_start_weight=1.0,
            image_fusion_end_weight=0.0,
            image_fusion_decay_epochs=3,
        )

        self.assertAlmostEqual(resolve_image_fusion_weight(config, 0), 1.0, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(config, 1), 0.5, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(config, 2), 0.0, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(config, 5), 0.0, places=6)
        cosine_config = FirstStageConfig(
            encoder_fusion_mode="gated_dual_path",
            image_fusion_start_weight=1.0,
            image_fusion_end_weight=0.0,
            image_fusion_decay_epochs=5,
            image_fusion_schedule="cosine",
        )
        self.assertAlmostEqual(resolve_image_fusion_weight(cosine_config, 0), 1.0, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(cosine_config, 2), 0.5, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(cosine_config, 4), 0.0, places=6)
        fixed_config = FirstStageConfig(
            encoder_fusion_mode="gated_dual_path",
            image_fusion_start_weight=0.3,
            image_fusion_end_weight=0.3,
            image_fusion_decay_epochs=1,
        )
        self.assertAlmostEqual(resolve_image_fusion_weight(fixed_config, 0), 0.3, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(fixed_config, 10), 0.3, places=6)
        self.assertAlmostEqual(resolve_image_fusion_weight(FirstStageConfig(), 0), 0.0, places=6)
        self.assertFalse(is_best_optical_candidate(config, 0.5))
        self.assertTrue(is_best_optical_candidate(config, 0.0))
        self.assertFalse(is_best_optical_candidate(fixed_config, 0.3))
        self.assertTrue(is_best_optical_candidate(FirstStageConfig(encoder_fusion_mode="optical_only"), 1.0))
        self.assertTrue(should_update_best_optical(0.8, None))
        self.assertTrue(should_update_best_optical(0.8, 0.7))
        self.assertFalse(should_update_best_optical(0.7, 0.8))
        balanced = compute_first_stage_balanced_metrics(
            {
                "camera_corr": 0.8,
                "camera_edge_corr": 0.6,
                "camera_foreground_dice": 0.7,
                "camera_background_leakage": 0.2,
                "camera_hotspot_ratio": 0.1,
                "complex_camera_corr": 0.3,
                "complex_camera_edge_corr": 0.1,
                "complex_camera_foreground_dice": 0.2,
                "complex_camera_background_leakage": 0.5,
                "complex_camera_hotspot_ratio": 0.1,
            },
            {
                "2": {
                    "camera_corr": 0.5,
                    "camera_edge_corr": 0.2,
                    "camera_foreground_dice": 0.3,
                    "camera_background_leakage": 0.4,
                    "camera_hotspot_ratio": 0.1,
                },
                "7": {
                    "camera_corr": 0.9,
                    "camera_edge_corr": 0.7,
                    "camera_foreground_dice": 0.8,
                    "camera_background_leakage": 0.1,
                    "camera_hotspot_ratio": 0.01,
                },
            },
        )
        self.assertAlmostEqual(balanced["min_class_camera_corr"], 0.5, places=6)
        self.assertAlmostEqual(balanced["camera_balanced_score"], 0.68, places=6)
        self.assertAlmostEqual(balanced["camera_mnist_score"], 0.66, places=6)
        self.assertAlmostEqual(balanced["min_class_camera_mnist_score"], 0.315, places=6)
        self.assertAlmostEqual(balanced["camera_class_balanced_mnist_score"], 0.4875, places=6)
        self.assertAlmostEqual(balanced["complex_camera_mnist_score"], 0.16, places=6)
        self.assertAlmostEqual(balanced["camera_complex_balanced_mnist_score"], 0.32375, places=6)
        self.assertLess(
            balanced["camera_class_balanced_mnist_score"],
            balanced["camera_mnist_score"],
        )
        self.assertLess(
            balanced["camera_complex_balanced_mnist_score"],
            balanced["camera_class_balanced_mnist_score"],
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for complexity metric tests")
    def test_complex_subset_selects_higher_edge_density_samples(self) -> None:
        import torch

        batch = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
        batch[0, 0, 3:13, 7:9] = 1.0
        batch[1, 0, 3:13, 4:6] = 1.0
        batch[1, 0, 3:5, 4:12] = 1.0
        batch[1, 0, 8:10, 4:12] = 1.0
        batch[1, 0, 11:13, 4:12] = 1.0
        batch[1, 0, 5:8, 10:12] = 1.0

        evaluation_module = load_evaluation_module()
        scores = evaluation_module.compute_input_complexity_scores(torch, batch)
        complex_mask = evaluation_module.select_complex_subset_mask(torch, batch)

        self.assertGreater(scores[1].item(), scores[0].item())
        self.assertEqual(complex_mask.tolist(), [False, True])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for sample metric tests")
    def test_sample_quality_metrics_include_nearest_neighbor_bank(self) -> None:
        import torch

        sampling_module = load_sampling_module()
        generated = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        generated[0, 0, :, 0] = 1.0
        generated[1, 0, :, 3] = 1.0
        fixed_reference = torch.zeros_like(generated)
        fixed_reference[0, 0, :, 3] = 1.0
        fixed_reference[1, 0, :, 0] = 1.0
        reference_bank = {
            4: torch.stack(
                [
                    generated[0],
                    generated[1],
                ]
            )
        }

        sampling_module = load_sampling_module()
        metrics = sampling_module.compute_sample_quality_metrics(
            generated,
            fixed_reference,
            torch.tensor([4, 4], dtype=torch.long),
            reference_image_bank=reference_bank,
            reference_bank_source="validation_only",
        )

        self.assertAlmostEqual(metrics["overall"]["dice"], 0.0, places=6)
        nearest = metrics["nearest_neighbor"]
        self.assertEqual(nearest["reference_source"], "validation_only")
        self.assertEqual(nearest["reference_counts"], {"4": 2})
        self.assertAlmostEqual(nearest["overall"]["dice"], 1.0, places=6)
        self.assertAlmostEqual(nearest["per_class"]["4"]["dice"], 1.0, places=6)
        self.assertEqual(nearest["per_class"]["4"]["reference_count"], 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v16 readout diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v16 readout diagnostics")

    def _v17_reference_images(self, torch_module, labels):
        reference = torch_module.zeros((len(labels), 1, 8, 8), dtype=torch_module.float32)
        patterns = {
            0: [(1, 1), (1, 2), (2, 1), (2, 2), (5, 5)],
            3: [(1, 1), (1, 2), (3, 2), (5, 1), (5, 2)],
            4: [(1, 1), (2, 1), (3, 1), (3, 2), (1, 3)],
            7: [(1, 1), (1, 2), (1, 3), (2, 3), (3, 2)],
            9: [(1, 2), (2, 1), (2, 2), (3, 2), (4, 1)],
        }
        for index, label in enumerate(labels):
            for row, col in patterns[int(label)]:
                reference[index, 0, row, col] = 1.0
        return reference

    def _write_v17_terminal_sample(self, root, name, labels, generated, reference, torch_module):
        sample_run = root / "outputs" / "20260630_v17" / "runs" / name
        artifacts = sample_run / "artifacts"
        images = sample_run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        bundle = {
            "class_labels": label_tensor,
            "reference_labels": label_tensor,
            "reference_images": reference,
            "generated_camera_raw": generated,
        }
        for key in (
            "generator_desired_intensity",
            "slm_command",
            "encoder_input_amplitude",
            "encoded_amplitude",
            "ols_output_amplitude",
            "decoder_output_amplitude",
            "chip_amplitude",
            "diffractive_camera_plane_raw",
            "generated_camera",
            "generated_camera_display",
        ):
            bundle[key] = generated.clone()
        torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
        sampling_module = load_sampling_module()
        metrics = sampling_module.compute_sample_quality_metrics(generated, reference, label_tensor)
        (artifacts / "one_shot_sample_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return sample_run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")

    def _write_v19_front_sample(self, root, run_name, labels, slm, encoder, torch_module):
        sample_run = root / "outputs" / "20260630_v19" / "runs" / run_name
        artifacts = sample_run / "artifacts"
        images = sample_run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        bundle = {
            "class_labels": label_tensor,
            "slm_command": slm.clone(),
            "encoder_input_amplitude": encoder.clone(),
            "encoder_input_grid_amplitude": encoder.clone(),
            "generated_camera": encoder.clone(),
            "generated_camera_display": encoder.clone(),
        }
        torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
        (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
        (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return sample_run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v19 front diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v19 front diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v19 front diagnostics")

    def _write_v20_downstream_run(self, root, run_name, kind, labels, planes, torch_module):
        version = "20260630_v20"
        run = root / "outputs" / version / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        reference = planes["reference"].clone()
        bundle = {
            "class_labels": label_tensor,
            "eval_target" if kind == "eval" else "reference_images": reference,
        }
        for key in (
            "slm_command",
            "encoder_input_amplitude",
            "encoded_amplitude",
            "ols_output_amplitude",
            "decoder_output_amplitude",
            "chip_amplitude",
            "chip_intensity_primary",
            "probe_plane",
            "diffractive_camera_plane_raw",
            "generated_camera",
        ):
            bundle[key] = planes.get(key, reference).clone()
        if kind == "eval":
            bundle["student_camera_raw"] = bundle["generated_camera"].clone()
            torch_module.save(bundle, artifacts / "one_shot_distill_eval_tensors.pt")
            (artifacts / "one_shot_distill_eval_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_distill_eval_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        else:
            bundle["generated_camera_raw"] = bundle["generated_camera"].clone()
            torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
            (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")

    def _write_v21_physical_cmos_run(self, root, run_name, kind, labels, reference, physical, selected, torch_module):
        version = "20260701_v21"
        run = root / "outputs" / version / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        bundle = {
            "class_labels": label_tensor,
            "eval_target" if kind == "eval" else "reference_images": reference.clone(),
            "physical_camera_plane": physical.clone(),
            "diffractive_camera_plane_raw": physical.clone(),
            "generated_camera": physical.clone(),
            "generated_camera_raw": physical.clone(),
            "selected_readout": selected.clone(),
            "selected_readout_raw": selected.clone(),
            "camera_plane": selected.clone(),
            "primary_readout_plane": "chip_intensity",
        }
        if kind == "eval":
            bundle["student_camera_raw"] = physical.clone()
            torch_module.save(bundle, artifacts / "one_shot_distill_eval_tensors.pt")
            (artifacts / "one_shot_distill_eval_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_distill_eval_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        else:
            torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
            (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v21 physical CMOS diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v21 physical CMOS diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v21 physical CMOS diagnostics")

    def _write_v23_decoder_chip_run(self, root, run_name, kind, labels, planes, torch_module):
        version = "20260701_v23"
        run = root / "outputs" / version / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        reference = planes["reference"].clone()
        bundle = {
            "class_labels": label_tensor,
            "eval_target" if kind == "eval" else "reference_images": reference,
        }
        for key in (
            "slm_command",
            "encoder_input_amplitude",
            "encoded_amplitude",
            "ols_output_amplitude",
            "decoder_output_amplitude",
            "chip_intensity_primary",
            "probe_plane",
            "physical_camera_plane",
        ):
            bundle[key] = planes.get(key, reference).clone()
        if kind == "eval":
            torch_module.save(bundle, artifacts / "one_shot_distill_eval_tensors.pt")
            (artifacts / "one_shot_distill_eval_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_distill_eval_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        else:
            torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
            (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v23 decoder/chip diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v23 decoder/chip diagnostics")

    def _write_v24_expressivity_run(
        self,
        root,
        run_name,
        kind,
        labels,
        planes,
        torch_module,
        *,
        decoder_mode="normal",
        ols_calibration="none",
        decoder_bypass_active=False,
        decoder_modulation_override="none",
        decoder_layers=2,
        target_code_mode="none",
        target_code_bridge_source="none",
        optical_core_arch="diffractive_phase_only",
    ):
        version = "20260701_v24"
        run = root / "outputs" / version / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        reference = planes["reference"].clone()
        bundle = {
            "class_labels": label_tensor,
            "eval_target" if kind == "eval" else "reference_images": reference,
            "decoder_diagnostic_mode": decoder_mode,
            "decoder_diagnostic_modulation_override": decoder_modulation_override,
            "ols_transfer_calibration": ols_calibration,
            "decoder_bypass_active": torch_module.tensor(decoder_bypass_active),
            "target_code_diagnostic_mode": target_code_mode,
            "target_code_diagnostic_active": torch_module.tensor(target_code_mode != "none"),
            "target_code_bridge_source": target_code_bridge_source,
            "target_code_bridge_active": torch_module.tensor(target_code_bridge_source != "none"),
        }
        metadata = {
            "stage": "one_shot_distill_eval" if kind == "eval" else "one_shot_sample",
            "decoder_diagnostic_mode": decoder_mode,
            "decoder_diagnostic_modulation_override": decoder_modulation_override,
            "decoder_bypass_active": bool(decoder_bypass_active),
            "ols_transfer_calibration": ols_calibration,
            "target_code_diagnostic_mode": target_code_mode,
            "target_code_diagnostic_active": target_code_mode != "none",
            "target_code_bridge_source": target_code_bridge_source,
            "target_code_bridge_active": target_code_bridge_source != "none",
            "optical_core_arch": optical_core_arch,
            "readout_primary_plane": "camera",
            "d2nn": {
                "grid_size": 100,
                "pixel_size": 2.0e-6,
                "encoder_layers": 2,
                "decoder_layers": int(decoder_layers),
                "layer_spacing": 1.50e-3,
                "modulation": "phase_only",
            },
        }
        bundle["artifact_metadata"] = metadata
        for key in (
            "slm_command",
            "encoder_input_amplitude",
            "encoded_amplitude",
            "ols_output_amplitude",
            "target_code_amplitude",
            "decoder_output_amplitude",
            "chip_intensity_primary",
            "probe_plane",
            "physical_camera_plane",
        ):
            bundle[key] = planes.get(key, reference).clone()
        if kind == "eval":
            torch_module.save(bundle, artifacts / "one_shot_distill_eval_tensors.pt")
            (artifacts / "one_shot_distill_eval_metrics.json").write_text("{}", encoding="utf-8")
            (artifacts / "one_shot_distill_eval_artifact_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (images / "one_shot_distill_eval_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        else:
            torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
            (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
            (artifacts / "one_shot_sample_artifact_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"not-a-real-png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v24 expressivity diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v24 expressivity diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v24 expressivity diagnostics")

    def _write_v25_capacity_summary(self, root, probes):
        summary_path = root / "outputs" / "reports" / "v25_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": {"v24_probe_a_sample_corr": 0.10},
            "probes": [
                {
                    "label": label,
                    "eval_run": eval_run.as_posix(),
                    "sample_eligible": True,
                }
                for label, eval_run, _sample_run in probes
            ],
            "sample_runs": [
                {"label": label, "sample_run": sample_run.as_posix()}
                for label, _eval_run, sample_run in probes
                if sample_run is not None
            ],
            "accepted_summary_written": False,
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return summary_path

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v25 decoder capacity diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v25 decoder capacity diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v25 decoder capacity diagnostics")

    def _v26_oracle_reference(self, torch_module, labels):
        reference = torch_module.zeros((len(labels), 1, 16, 16), dtype=torch_module.float32)
        positions = {
            0: (1, 1),
            1: (1, 5),
            2: (1, 9),
            3: (4, 2),
            4: (4, 6),
            5: (4, 10),
            6: (8, 2),
            7: (8, 6),
            8: (8, 10),
            9: (12, 6),
        }
        for index, label in enumerate(labels):
            row, col = positions[int(label)]
            reference[index, 0, row : row + 2, col : col + 2] = 1.0
        return reference

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v26 OLS oracle diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v26 OLS oracle diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v26 OLS oracle diagnostics")

    def _write_v27_v26_report(self, root):
        report = root / "outputs" / "reports" / "v26_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "accepted_summary_written": False,
                    "decision": {
                        "status": "ols-decodable",
                        "best_oracle_feature": "encoded_amplitude",
                        "best_oracle_accuracy": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return report

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v27 decoder synthesis diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v27 decoder synthesis diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v27 decoder synthesis diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")


    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28.1 replay payload tests")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28.1 same-batch diagnostics")


    def _write_v29_train_run(
        self,
        root,
        run_name,
        *,
        target_transform="none",
        decoder_layers=2,
        decoder_modulation_override="none",
        audit_passed=True,
        decoder_update_norm=1.0,
    ):
        run = root / "outputs" / "20260702_v29" / "runs" / run_name
        artifacts = run / "artifacts"
        configs = run / "configs"
        artifacts.mkdir(parents=True)
        configs.mkdir(parents=True)
        config = {
            "one_shot": {
                "decoder_trainability_audit": True,
                "decoder_only_training": True,
                "aux_topology_target_transform": target_transform,
                "decoder_diagnostic_modulation_override": decoder_modulation_override,
                "d2nn": {
                    "grid_size": 100,
                    "pixel_size": 2.0e-6,
                    "encoder_layers": 2,
                    "decoder_layers": decoder_layers,
                    "layer_spacing": 1.50e-3,
                    "modulation": "phase_only",
                },
            }
        }
        audit = {
            "enabled": True,
            "required_decoder_only_training": True,
            "passed": audit_passed,
            "failure_reasons": [] if audit_passed else ["decoder-update-zero"],
            "decoder_update_norm": float(decoder_update_norm),
            "upstream_update_norm": 0.0,
            "decoder_grad_norm_mean": 1.0 if audit_passed else 0.0,
            "decoder_param_count": 10,
            "upstream_param_count": 20,
        }
        (configs / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
        (artifacts / "one_shot_decoder_trainability_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v29 decoder diagnostics")


    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v30 target-code diagnostics")


    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v31 bridge diagnostics")

    def _write_v32_1_gap_run(
        self,
        root,
        run_name,
        kind,
        labels,
        reference,
        torch_module,
        *,
        planes,
        codes=None,
    ):
        run = root / "outputs" / "20260703_v32" / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        bundle = {
            "class_labels": label_tensor,
            "codes": codes
            if codes is not None
            else torch_module.arange(len(labels) * 4, dtype=torch_module.float32).view(len(labels), 4),
            "target_code_diagnostic_mode": "lowfreq_digit_code",
            "target_code_bridge_source": "ols_output_amplitude",
            "target_code_bridge_arch": "deep_residual",
        }
        if kind == "eval":
            bundle["eval_target"] = reference.clone()
            bundle["teacher_camera"] = reference.clone()
            filename = "one_shot_distill_eval_tensors.pt"
        else:
            bundle["reference_images"] = reference.clone()
            bundle["reference_labels"] = label_tensor.clone()
            filename = "one_shot_sample_tensors.pt"
        for key in (
            "slm_command",
            "encoded_amplitude",
            "target_code_amplitude",
            "decoder_input_amplitude",
            "decoder_output_amplitude",
            "chip_intensity_primary",
            "physical_camera_plane",
        ):
            bundle[key] = planes.get(key, reference).clone()
        torch_module.save(bundle, artifacts / filename)
        return run

    def _write_v32_1_gap_inputs(self, root, eval_run, sample_run):
        reports = root / "outputs" / "reports"
        reports.mkdir(parents=True)
        summary_path = reports / "v32_summary.json"
        bridge_path = reports / "v32_bridge.json"
        summary_path.write_text(
            json.dumps(
                {
                    "version": "v32",
                    "diagnostic_only": True,
                    "accepted_summary_written": False,
                    "probes": [
                        {
                            "label": "probe_a_ols_lowfreq_deep_bridge_same_batch",
                            "test_eval_run": eval_run.relative_to(root).as_posix(),
                            "sample_run": sample_run.relative_to(root).as_posix(),
                            "test_strict_gate_passed": True,
                            "sample_gate_passed": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        bridge_path.write_text(
            json.dumps(
                {
                    "version": "v32",
                    "diagnostic_only": True,
                    "accepted_summary_written": False,
                    "decision": {"status": "bridge-code-candidate-needs-human-review"},
                }
            ),
            encoding="utf-8",
        )
        return summary_path, bridge_path

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v32.1 sample-gap diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v32.1 sample-gap diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v32.1 sample-gap diagnostics")

    def _write_v33_gap_run(
        self,
        root,
        run_name,
        kind,
        labels,
        reference,
        torch_module,
        *,
        generated=None,
        residual=None,
        include_residual=True,
        reference_source=None,
    ):
        run = root / "outputs" / "20260703_v33" / "runs" / run_name
        artifacts = run / "artifacts"
        artifacts.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        generated_tensor = reference.clone() if generated is None else generated.clone()
        residual_tensor = torch_module.ones_like(reference) * 0.25 if residual is None else residual.clone()
        bundle = {
            "class_labels": label_tensor,
            "codes": torch_module.arange(len(labels) * 4, dtype=torch_module.float32).view(len(labels), 4),
            "target_code_amplitude": generated_tensor,
            "decoder_input_amplitude": generated_tensor,
            "decoder_output_amplitude": generated_tensor,
            "chip_intensity_primary": generated_tensor,
            "physical_camera_plane": generated_tensor,
        }
        if include_residual:
            bundle.update(
                {
                    "generator_residual_map": residual_tensor,
                    "generator_residual_abs_mean": residual_tensor.abs().flatten(1).mean(dim=1),
                }
            )
        if kind == "eval":
            bundle["eval_target"] = reference.clone()
            filename = "one_shot_distill_eval_tensors.pt"
        else:
            bundle["reference_images"] = reference.clone()
            bundle["reference_labels"] = label_tensor.clone()
            if reference_source is not None:
                bundle["reference_source"] = reference_source
            filename = "one_shot_sample_tensors.pt"
        torch_module.save(bundle, artifacts / filename)
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33 code diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33 code diagnostics")

    def _write_v33_2_sample_run(
        self,
        root,
        run_name,
        labels,
        reference,
        torch_module,
        *,
        target=None,
        decoder=None,
        physical=None,
        residual=None,
        reference_source="validation_only",
        with_preview=True,
    ):
        run = root / "outputs" / "20260703_v33_2" / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)
        target_tensor = reference.clone() if target is None else target.clone()
        decoder_tensor = target_tensor.clone() if decoder is None else decoder.clone()
        physical_tensor = decoder_tensor.clone() if physical is None else physical.clone()
        residual_tensor = (reference.clone() * 0.2 + 0.05) if residual is None else residual.clone()
        bundle = {
            "class_labels": label_tensor,
            "codes": torch_module.arange(len(labels) * 4, dtype=torch_module.float32).view(len(labels), 4),
            "reference_images": reference.clone(),
            "reference_labels": label_tensor.clone(),
            "reference_source": reference_source,
            "generator_residual_map": residual_tensor,
            "generator_residual_abs_mean": residual_tensor.abs().flatten(1).mean(dim=1),
            "target_code_amplitude": target_tensor,
            "decoder_input_amplitude": target_tensor,
            "decoder_output_amplitude": decoder_tensor,
            "chip_intensity_primary": decoder_tensor,
            "physical_camera_plane": physical_tensor,
        }
        torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
        (artifacts / "one_shot_sample_metrics.json").write_text(
            json.dumps({"corr": 0.1, "dice": 0.1}),
            encoding="utf-8",
        )
        if with_preview:
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"png")
            (artifacts / "one_shot_sample_100x100_cell_chain_preview.json").write_text(
                json.dumps({"columns": ["Target code amp", "Decoder in amp", "CMOS intensity"]}),
                encoding="utf-8",
            )
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33.2 preview diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33.2 preview diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33.2 preview diagnostics")

    def _write_v34_coupling_sample_run(
        self,
        root,
        run_name,
        labels,
        torch_module,
        *,
        active_until,
    ):
        run = root / "outputs" / "20260703_v34" / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        label_tensor = torch_module.tensor(labels, dtype=torch_module.long)

        def active_tensor():
            tensor = torch_module.zeros((len(labels), 1, 8, 8), dtype=torch_module.float32)
            for idx, label in enumerate(labels):
                row = (idx * 2 + label) % 8
                col = (idx * 3 + 1) % 8
                tensor[idx, 0, row, col] = 1.0
                tensor[idx, 0, (row + 1) % 8, col] = 0.5
            return tensor

        def collapsed_tensor():
            tensor = torch_module.zeros((len(labels), 1, 8, 8), dtype=torch_module.float32)
            for idx, label in enumerate(labels):
                tensor[idx, 0, label + 1, 2:6] = 1.0
            return tensor

        order = [
            "slm_command",
            "encoded_amplitude",
            "ols_output_amplitude",
            "target_code_amplitude",
            "decoder_output_amplitude",
            "physical_camera_plane",
        ]
        active_keys = set(order[: active_until])
        bundle = {
            "class_labels": label_tensor,
            "codes": torch_module.arange(len(labels) * 4, dtype=torch_module.float32).view(len(labels), 4),
            "generator_residual_map": active_tensor(),
            "generator_desired_intensity": active_tensor(),
        }
        for key in order:
            bundle[key] = active_tensor() if key in active_keys else collapsed_tensor()
        bundle["decoder_input_amplitude"] = bundle["target_code_amplitude"].clone()
        bundle["chip_intensity_primary"] = bundle["decoder_output_amplitude"].clone()
        torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
        (artifacts / "one_shot_sample_metrics.json").write_text(json.dumps({"diagnostic_only": True}), encoding="utf-8")
        (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v34 code coupling diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v35 code-carrier diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v36 high-gain diagnostics")

    def _write_v37_topology_sample_run(
        self,
        root,
        run_name,
        torch_module,
        *,
        target_ok,
        decoder_ok,
        camera_ok,
        with_preview=True,
        middle_shape=(16, 16),
        physical_shape=None,
        physical_batch=None,
    ):
        run = root / "outputs" / "20260703_v37" / "runs" / run_name
        artifacts = run / "artifacts"
        images = run / "images"
        artifacts.mkdir(parents=True)
        images.mkdir(parents=True)
        physical_shape = physical_shape or middle_shape
        batch = 4 if physical_batch is None else physical_batch

        def digit_tensor(*, shape, n=4):
            height, width = shape
            tensor = torch_module.zeros((n, 1, height, width), dtype=torch_module.float32)
            cx = width // 2
            y0 = max(1, height // 5)
            y1 = min(height - 1, height - height // 5)
            x0 = max(1, cx - width // 8)
            x1 = min(width - 1, cx + width // 8 + 1)
            for idx in range(n):
                tensor[idx, 0, y0:y1, cx] = 1.0
                upper = min(height - 1, y0 + idx % max(1, height // 4))
                lower = max(0, y1 - 1 - idx % max(1, height // 4))
                tensor[idx, 0, upper, x0:x1] = 0.8
                tensor[idx, 0, lower, x0:x1] = 0.8
            return tensor

        def blob_tensor(*, shape, n=4):
            return torch_module.full((n, 1, shape[0], shape[1]), 0.5, dtype=torch_module.float32)

        target = digit_tensor(shape=middle_shape) if target_ok else blob_tensor(shape=middle_shape)
        decoder_input = target.clone()
        decoder_output = digit_tensor(shape=middle_shape) if decoder_ok else blob_tensor(shape=middle_shape)
        physical = (
            digit_tensor(shape=physical_shape, n=batch)
            if camera_ok
            else blob_tensor(shape=physical_shape, n=batch)
        )
        bundle = {
            "class_labels": torch_module.tensor([0, 0, 1, 1], dtype=torch_module.long),
            "reference_source": "validation_only",
            "target_code_amplitude": target,
            "decoder_input_amplitude": decoder_input,
            "decoder_output_amplitude": decoder_output,
            "physical_camera_plane": physical,
        }
        torch_module.save(bundle, artifacts / "one_shot_sample_tensors.pt")
        (artifacts / "one_shot_sample_metrics.json").write_text(json.dumps({"diagnostic_only": True}), encoding="utf-8")
        if with_preview:
            (images / "one_shot_sample_100x100_cell_chain_preview.png").write_bytes(b"png")
        return run

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v37 topology diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v37 topology diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v38 CMOS review")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v38 CMOS review")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v39 topology diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v39 topology diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v40 diversity diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for CMOS review rendering")
    def test_cmos_review_selects_one_sample_per_class_and_rejects_missing_classes(self) -> None:
        import torch

        tool = load_v40_sample_code_diversity_target_representation_tool()
        labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
        physical = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 16, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            review = tool._render_cmos_review(
                "balanced",
                Path("outputs/20260710_v48/runs/balanced"),
                physical,
                Path(tmpdir),
                class_labels=labels,
                review_version="v48",
            )
            self.assertEqual(review["review_class_labels"], list(range(10)))
            self.assertEqual(review["review_indices"], list(range(0, 20, 2)))
            self.assertEqual(review["review_planes"], ["physical_camera_plane"])
            self.assertTrue(review["class_coverage_complete"])
            self.assertTrue(Path(review["cmos_only_review_image"]).is_file())

            with self.assertRaisesRegex(ValueError, "missing required review classes"):
                tool._render_cmos_review(
                    "missing",
                    Path("outputs/20260710_v48/runs/missing"),
                    physical[:-2],
                    Path(tmpdir),
                    class_labels=labels[:-2],
                    review_version="v48",
                )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v48 diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v49 diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v50 diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v41 decoder diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v42 bypass diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v43 geometry diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v44 spatial scaling diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v45 spacing diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v46 semantic diagnostics")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v47 decoder capacity diagnostics")


























    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for post-train sample tests")
    def test_post_train_samples_use_full_sample_count_for_grids(self) -> None:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, TrainingConfig
        from ldm_project.data import DatasetBundle

        sampling_module = load_sampling_module()

        class RecordingRunManager:
            def __init__(self) -> None:
                self.run_dir = Path(".")
                self.samples_dir = Path("images/samples")
                self.grid_calls = []
                self.metrics_payload = None

            def save_config_snapshot(self, name, data):
                return name, data

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
                self.grid_calls.append((Path(destination).name, max_items, list(tensor_map.keys()), dict(kwargs)))
                return destination

            def save_tensor_bundle(self, name, payload):
                return name, payload

            def save_artifact_json(self, name, payload):
                self.metrics_payload = payload
                return Path("artifacts") / f"{name}.json"

        class DummyFirstStage:
            def decode_with_intermediates(self, latents):
                image = F.interpolate(latents[:, :1], size=(8, 8), mode="nearest").clamp(0.0, 1.0)
                zeros = torch.zeros_like(image)
                return {
                    "image_recon": image,
                    "chip_amplitude": image,
                    "chip_phase": zeros,
                    "probe_plane": image,
                    "camera_plane": image,
                }

        class DummyScheduler:
            timesteps = 2

        images = torch.rand(3, 1, 8, 8)
        labels = torch.arange(3, dtype=torch.long)
        dataset = TensorDataset(images, labels)
        bundle = DatasetBundle(
            dataset=dataset,
            loader=DataLoader(dataset, batch_size=3, shuffle=False),
            effective_size=3,
            total_size=3,
        )
        train_cfg = TrainingConfig(
            data=DataConfig(dataset="mnist"),
            first_stage=FirstStageConfig(image_size=8, latent_channels=1),
            diffusion=DiffusionConfig(num_classes=3),
        )

        def sample_fn(_model, _scheduler, shape, device, _labels, guidance_scale):
            return torch.zeros(shape, device=device)

        def trace_fn(_model, _scheduler, shape, device, _labels, guidance_scale, trace_timesteps):
            final = torch.zeros(shape, device=device)
            return final, {"noise": torch.ones(shape, device=device), "t000": final}

        run_manager = RecordingRunManager()
        sampling_module.save_post_train_samples(
            torch,
            run_manager,
            DummyFirstStage(),
            diffusion_model=None,
            scheduler=DummyScheduler(),
            train_bundle=bundle,
            validation_bundle=None,
            train_cfg=train_cfg,
            latent_scale=torch.tensor(1.0),
            device=torch.device("cpu"),
            sample_latents_fn=sample_fn,
            sample_latents_with_trace_fn=trace_fn,
            resolve_trace_steps_fn=lambda *_args: [{"label": "t000", "nominal_timestep": 0, "actual_timestep": 0}],
        )

        self.assertEqual(run_manager.grid_calls[0][0], "post_train_sample_amplitude.png")
        self.assertEqual(run_manager.grid_calls[0][1], 6)
        self.assertEqual(run_manager.grid_calls[0][3]["row_labels"], ["0", "0", "1", "1", "2", "2"])
        self.assertEqual(run_manager.grid_calls[2][0], "post_train_sample_phase.png")
        self.assertEqual(run_manager.grid_calls[1][1], 6)
        self.assertEqual(run_manager.grid_calls[3][0], "post_train_denoising_camera_trace.png")
        self.assertEqual(run_manager.grid_calls[2][1], 6)
        self.assertEqual(run_manager.metrics_payload["overall"]["count"], 6)
        self.assertEqual(sorted(run_manager.metrics_payload["per_class"].keys()), ["0", "1", "2"])
        self.assertEqual(run_manager.metrics_payload["nearest_neighbor"]["reference_source"], "train_only")
        self.assertEqual(run_manager.metrics_payload["nearest_neighbor"]["overall"]["count"], 6)
        self.assertEqual(
            sorted(run_manager.metrics_payload["nearest_neighbor"]["per_class"].keys()),
            ["0", "1", "2"],
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for post-train sample tests")
    def test_post_train_sample_labels_support_weak_class_repeats(self) -> None:
        import torch

        sampling_module = load_sampling_module()
        labels = sampling_module.build_post_train_sample_labels(
            torch,
            num_classes=10,
            device=torch.device("cpu"),
            sample_class_labels=(3, 4, 5),
            sample_repeats_per_class=4,
        )
        default_labels = sampling_module.build_post_train_sample_labels(
            torch,
            num_classes=10,
            device=torch.device("cpu"),
        )

        self.assertEqual(labels.tolist(), [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
        self.assertEqual(default_labels.tolist(), [label for label in range(10) for _ in range(1)])
        with self.assertRaisesRegex(ValueError, "sample_repeats_per_class"):
            sampling_module.build_post_train_sample_labels(
                torch,
                num_classes=10,
                device=torch.device("cpu"),
                sample_class_labels=(3,),
                sample_repeats_per_class=0,
            )
        with self.assertRaisesRegex(ValueError, "sample_class_labels"):
            sampling_module.build_post_train_sample_labels(
                torch,
                num_classes=10,
                device=torch.device("cpu"),
                sample_class_labels=(10,),
            )

    def test_should_validate_epoch_keeps_first_interval_and_last_epoch(self) -> None:
        should_validate_epoch = load_train_utils_module().should_validate_epoch

        self.assertTrue(should_validate_epoch(0, 10, 5))
        self.assertFalse(should_validate_epoch(1, 10, 5))
        self.assertTrue(should_validate_epoch(4, 10, 5))
        self.assertFalse(should_validate_epoch(7, 10, 5))
        self.assertTrue(should_validate_epoch(9, 10, 5))
        with self.assertRaisesRegex(ValueError, "validation_epoch_interval"):
            should_validate_epoch(0, 10, 0)
        with self.assertRaisesRegex(ValueError, "total_epochs"):
            should_validate_epoch(0, 0, 1)

    def test_require_strict_optical_best_checkpoint_rejects_missing_best(self) -> None:
        train_utils_module = load_train_utils_module()

        checkpoint = "/tmp/run/checkpoints/first_stage_best_optical.pt"
        self.assertEqual(train_utils_module.require_strict_optical_best_checkpoint(checkpoint), checkpoint)
        with self.assertRaisesRegex(RuntimeError, "strict optical-only first-stage best checkpoint"):
            train_utils_module.require_strict_optical_best_checkpoint(None)

    def test_validate_stage_requirements_requires_stage_inputs(self) -> None:
        train_utils_module = load_train_utils_module()
        train_utils_module.validate_stage_requirements(
            "sample", "first.pt", "diffusion.pt"
        )
        with self.assertRaisesRegex(ValueError, "requires --first-stage-checkpoint"):
            train_utils_module.validate_stage_requirements("sample", None, None)


    def test_final_pt_checkpoint_resolution_fills_missing_stage_inputs(self) -> None:
        train_utils_module = load_train_utils_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            final_pt_dir = Path(tmpdir) / "final_pt"
            final_pt_dir.mkdir()
            first_stage_path = final_pt_dir / "first_stage_best_optical.pt"
            diffusion_path = final_pt_dir / "diffusion_best.pt"
            first_stage_path.write_text("first", encoding="utf-8")
            diffusion_path.write_text("diffusion", encoding="utf-8")

            resolved = train_utils_module.resolve_final_pt_stage_checkpoints(
                "sample",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(resolved["first_stage_checkpoint"], str(first_stage_path))
            self.assertEqual(resolved["first_stage_checkpoint_source"], "final_pt")
            self.assertEqual(resolved["diffusion_checkpoint"], str(diffusion_path))
            self.assertEqual(resolved["diffusion_checkpoint_source"], "final_pt")

            explicit_first = train_utils_module.resolve_final_pt_stage_checkpoints(
                "sample",
                "manual_first.pt",
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(explicit_first["first_stage_checkpoint"], "manual_first.pt")
            self.assertEqual(explicit_first["first_stage_checkpoint_source"], "cli")
            self.assertEqual(explicit_first["diffusion_checkpoint"], str(diffusion_path))
            self.assertEqual(explicit_first["diffusion_checkpoint_source"], "final_pt")

            diffusion_only = train_utils_module.resolve_final_pt_stage_checkpoints(
                "diffusion",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(diffusion_only["first_stage_checkpoint"], str(first_stage_path))
            self.assertIsNone(diffusion_only["diffusion_checkpoint"])

            distill_export = train_utils_module.resolve_final_pt_stage_checkpoints(
                "teacher_distill_export",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(distill_export["first_stage_checkpoint"], str(first_stage_path))
            self.assertEqual(distill_export["diffusion_checkpoint"], str(diffusion_path))

            oracle_audit = train_utils_module.resolve_final_pt_stage_checkpoints(
                "teacher_first_stage_oracle_audit",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(oracle_audit["first_stage_checkpoint"], str(first_stage_path))
            self.assertIsNone(oracle_audit["diffusion_checkpoint"])

            latent_audit = train_utils_module.resolve_final_pt_stage_checkpoints(
                "teacher_diffusion_latent_audit",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(latent_audit["first_stage_checkpoint"], str(first_stage_path))
            self.assertEqual(latent_audit["diffusion_checkpoint"], str(diffusion_path))

            distill_filter = train_utils_module.resolve_final_pt_stage_checkpoints(
                "teacher_distill_filter",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertIsNone(distill_filter["first_stage_checkpoint"])
            self.assertIsNone(distill_filter["diffusion_checkpoint"])

    def test_apply_cli_overrides_updates_training_config(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_utils_module = load_train_utils_module()
        config = TrainingConfig()
        train_utils_module.apply_cli_overrides(
            config,
            Namespace(
                stage="first_stage",
                batch_size=8,
                seed=123,
                subset_fraction=0.002,
                subset_size=2000,
                validation_fraction=0.0,
                num_workers=4,
                first_stage_epochs=30,
                diffusion_epochs=0,
                one_shot_epochs=2,
                guidance_scale=3.5,
                cond_dropout_prob=0.05,
                sample_class_labels=[3, 4, 5],
                sample_repeats_per_class=4,
                diffusion_class_weight_labels=[3, 4, 5],
                diffusion_class_weight=1.5,
                camera_artifact_weight=0.25,
                encoder_fusion_mode="optical_only",
                image_fusion_weight=0.0,
                optical_encoder_input_mode="pf2_intensity_multiscale_v2",
                optical_encoder_arch="global_context",
                metasurface_decoder_arch="conv",
                distill_samples_per_class=7,
                distill_total_samples=70,
                distill_train_subset_size=12,
                distill_output="outputs/runs/distill.pt",
            ),
        )

        self.assertEqual(config.mode, "teacher_first_stage")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.seed, 123)
        self.assertAlmostEqual(config.data.subset_fraction, 0.002, places=6)
        self.assertEqual(config.data.subset_size, 2000)
        self.assertAlmostEqual(config.data.validation_fraction, 0.0, places=6)
        self.assertEqual(config.data.num_workers, 4)
        self.assertEqual(config.first_stage_epochs, 30)
        self.assertEqual(config.diffusion_epochs, 0)
        self.assertEqual(config.one_shot_epochs, 2)
        self.assertAlmostEqual(config.diffusion.guidance_scale, 3.5, places=6)
        self.assertAlmostEqual(config.diffusion.cond_dropout_prob, 0.05, places=6)
        self.assertEqual(config.diffusion.sample_class_labels, (3, 4, 5))
        self.assertEqual(config.diffusion.sample_repeats_per_class, 4)
        self.assertEqual(config.one_shot.sample_class_labels, (3, 4, 5))
        self.assertEqual(config.one_shot.sample_repeats_per_class, 4)
        self.assertEqual(config.diffusion.class_weight_labels, (3, 4, 5))
        self.assertAlmostEqual(config.diffusion.class_weight, 1.5, places=6)
        self.assertAlmostEqual(config.first_stage.camera_artifact_weight, 0.25, places=6)
        self.assertEqual(config.first_stage.encoder_fusion_mode, "optical_only")
        self.assertEqual(config.first_stage.optical_encoder_input_mode, "pf2_intensity_multiscale_v2")
        self.assertEqual(config.first_stage.optical_encoder_arch, "global_context")
        self.assertEqual(config.first_stage.metasurface_decoder_arch, "conv")
        self.assertEqual(config.distillation.samples_per_class, 7)
        self.assertEqual(config.distillation.total_samples, 70)
        self.assertEqual(config.distillation.train_subset_size, 12)
        self.assertEqual(config.distillation.output_path, "outputs/runs/distill.pt")
        self.assertAlmostEqual(config.first_stage.image_fusion_start_weight, 0.0, places=6)
        self.assertAlmostEqual(config.first_stage.image_fusion_end_weight, 0.0, places=6)
        self.assertEqual(config.first_stage.image_fusion_decay_epochs, 1)

    def test_cli_sample_overrides_drive_one_shot_distill_eval_selection(self) -> None:
        from argparse import Namespace

        import torch

        from ldm_project.config import OneShotConfig, TrainingConfig
        from ldm_project.one_shot_distill_data import _select_distill_eval_indices

        train_utils_module = load_train_utils_module()
        config = TrainingConfig(
            one_shot=OneShotConfig(
                num_classes=3,
                sample_class_labels=(0,),
                sample_repeats_per_class=1,
            ),
        )
        train_utils_module.apply_cli_overrides(
            config,
            Namespace(
                stage="one_shot_distill_eval",
                batch_size=None,
                subset_fraction=None,
                subset_size=None,
                validation_fraction=None,
                num_workers=None,
                first_stage_epochs=None,
                diffusion_epochs=None,
                one_shot_epochs=None,
                guidance_scale=None,
                cond_dropout_prob=None,
                sample_class_labels=[1, 2],
                sample_repeats_per_class=3,
                diffusion_class_weight_labels=None,
                diffusion_class_weight=None,
                camera_artifact_weight=None,
                encoder_fusion_mode=None,
                image_fusion_weight=None,
                optical_encoder_input_mode=None,
                optical_encoder_arch=None,
                metasurface_decoder_arch=None,
                distill_samples_per_class=None,
                distill_total_samples=None,
                distill_output=None,
            ),
        )

        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
        selected, summary = _select_distill_eval_indices(torch, labels, config)

        self.assertEqual(config.one_shot.sample_class_labels, (1, 2))
        self.assertEqual(config.one_shot.sample_repeats_per_class, 3)
        self.assertEqual(len(selected), 6)
        self.assertEqual(summary["requested_samples_per_class"], 3)
        self.assertEqual(summary["selected_count"], 6)
        self.assertEqual(summary["per_class"]["1"]["selected"], 3)
        self.assertEqual(summary["per_class"]["2"]["selected"], 3)

    def test_apply_cli_overrides_rejects_invalid_image_fusion_weight(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_utils_module = load_train_utils_module()
        for invalid_weight in (-0.1, 1.1):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "image_fusion_weight"):
                    train_utils_module.apply_cli_overrides(
                        TrainingConfig(),
                        Namespace(
                            stage=None,
                            batch_size=None,
                            subset_fraction=None,
                            subset_size=None,
                            validation_fraction=None,
                            num_workers=None,
                            first_stage_epochs=None,
                            diffusion_epochs=None,
                            guidance_scale=None,
                            cond_dropout_prob=None,
                            sample_class_labels=None,
                            sample_repeats_per_class=None,
                            diffusion_class_weight_labels=None,
                            diffusion_class_weight=None,
                            camera_artifact_weight=None,
                            encoder_fusion_mode=None,
                            image_fusion_weight=invalid_weight,
                            optical_encoder_input_mode=None,
                            optical_encoder_arch=None,
                            metasurface_decoder_arch=None,
                        ),
                    )

    def test_apply_training_seed_sets_torch_and_cuda_rng(self) -> None:
        train_utils_module = load_train_utils_module()
        calls: list[tuple[str, int]] = []

        class FakeCuda:
            def is_available(self) -> bool:
                return True

            def manual_seed_all(self, seed: int) -> None:
                calls.append(("cuda", seed))

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed: int) -> None:
                calls.append(("cpu", seed))

        applied_seed = train_utils_module.apply_training_seed(FakeTorch, 202)

        self.assertEqual(applied_seed, 202)
        self.assertEqual(calls, [("cpu", 202), ("cuda", 202)])

    def test_apply_cli_overrides_rejects_invalid_diffusion_overrides(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_utils_module = load_train_utils_module()

        def args_with(**overrides):
            values = {
                "stage": None,
                "batch_size": None,
                "subset_fraction": None,
                "subset_size": None,
                "validation_fraction": None,
                "num_workers": None,
                "first_stage_epochs": None,
                "diffusion_epochs": None,
                "guidance_scale": None,
                "cond_dropout_prob": None,
                "sample_class_labels": None,
                "sample_repeats_per_class": None,
                "diffusion_class_weight_labels": None,
                "diffusion_class_weight": None,
                "camera_artifact_weight": None,
                "encoder_fusion_mode": None,
                "image_fusion_weight": None,
                "optical_encoder_input_mode": None,
                "optical_encoder_arch": None,
                "metasurface_decoder_arch": None,
            }
            values.update(overrides)
            return Namespace(**values)

        with self.assertRaisesRegex(ValueError, "guidance_scale"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(guidance_scale=-1.0))
        for invalid_dropout in (-0.1, 1.5):
            with self.subTest(invalid_dropout=invalid_dropout):
                with self.assertRaisesRegex(ValueError, "cond_dropout_prob"):
                    train_utils_module.apply_cli_overrides(
                        TrainingConfig(),
                        args_with(cond_dropout_prob=invalid_dropout),
                    )
        with self.assertRaisesRegex(ValueError, "sample_class_labels"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(sample_class_labels=[10]))
        with self.assertRaisesRegex(ValueError, "sample_repeats_per_class"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(sample_repeats_per_class=0))
        with self.assertRaisesRegex(ValueError, "diffusion_class_weight_labels"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(diffusion_class_weight_labels=[10]))
        with self.assertRaisesRegex(ValueError, "class_weight"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(diffusion_class_weight=0.0))
        with self.assertRaisesRegex(ValueError, "training.seed"):
            train_utils_module.apply_cli_overrides(TrainingConfig(), args_with(seed=-1))

    def test_apply_cli_overrides_rejects_invalid_camera_artifact_weight(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_utils_module = load_train_utils_module()
        with self.assertRaisesRegex(ValueError, "camera_artifact_weight"):
            train_utils_module.apply_cli_overrides(
                TrainingConfig(),
                Namespace(
                    stage=None,
                    batch_size=None,
                    subset_fraction=None,
                    subset_size=None,
                    validation_fraction=None,
                    num_workers=None,
                    first_stage_epochs=None,
                    diffusion_epochs=None,
                    guidance_scale=None,
                    cond_dropout_prob=None,
                    sample_class_labels=None,
                    sample_repeats_per_class=None,
                    diffusion_class_weight_labels=None,
                    diffusion_class_weight=None,
                    camera_artifact_weight=-0.01,
                    encoder_fusion_mode=None,
                    image_fusion_weight=None,
                    optical_encoder_input_mode=None,
                    optical_encoder_arch=None,
                    metasurface_decoder_arch=None,
                ),
            )
