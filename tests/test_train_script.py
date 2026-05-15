from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings


def load_train_module():
    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_under_test", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train.py from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainScriptStepTests(unittest.TestCase):
    def test_resolve_logged_step_resets_global_step_per_train_round(self) -> None:
        resolve_logged_step = load_train_module().resolve_logged_step

        first_epoch_last = resolve_logged_step(batch_index=625, stage_step=625)
        next_epoch_first = resolve_logged_step(batch_index=0, stage_step=626)

        self.assertEqual(first_epoch_last["global_step"], 625)
        self.assertEqual(first_epoch_last["stage_step"], 625)
        self.assertEqual(next_epoch_first["global_step"], 0)
        self.assertEqual(next_epoch_first["stage_step"], 626)

    def test_resolve_logged_step_rejects_negative_values(self) -> None:
        resolve_logged_step = load_train_module().resolve_logged_step

        with self.assertRaises(ValueError):
            resolve_logged_step(batch_index=-1, stage_step=0)

        with self.assertRaises(ValueError):
            resolve_logged_step(batch_index=0, stage_step=-1)

    def test_resolve_stage_epoch_plan_matches_requested_mode(self) -> None:
        resolve_stage_epoch_plan = load_train_module().resolve_stage_epoch_plan

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
                stage="one_shot_distill_eval",
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

        train_module = load_train_module()
        compute_first_stage_balanced_metrics = train_module.compute_first_stage_balanced_metrics
        is_best_optical_candidate = train_module.is_best_optical_candidate
        resolve_image_fusion_weight = train_module.resolve_image_fusion_weight
        should_update_best_optical = train_module.should_update_best_optical
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

        train_module = load_train_module()
        batch = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
        batch[0, 0, 3:13, 7:9] = 1.0
        batch[1, 0, 3:13, 4:6] = 1.0
        batch[1, 0, 3:5, 4:12] = 1.0
        batch[1, 0, 8:10, 4:12] = 1.0
        batch[1, 0, 11:13, 4:12] = 1.0
        batch[1, 0, 5:8, 10:12] = 1.0

        scores = train_module.compute_input_complexity_scores(torch, batch)
        complex_mask = train_module.select_complex_subset_mask(torch, batch)

        self.assertGreater(scores[1].item(), scores[0].item())
        self.assertEqual(complex_mask.tolist(), [False, True])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for sample metric tests")
    def test_sample_quality_metrics_include_nearest_neighbor_bank(self) -> None:
        import torch

        train_module = load_train_module()
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

        metrics = train_module.compute_sample_quality_metrics(
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for post-train sample tests")
    def test_post_train_samples_use_full_sample_count_for_grids(self) -> None:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, TrainingConfig
        from ldm_project.data import DatasetBundle

        train_module = load_train_module()

        class RecordingRunManager:
            def __init__(self) -> None:
                self.run_dir = Path(".")
                self.samples_dir = Path("images/samples")
                self.grid_calls = []
                self.metrics_payload = None

            def save_config_snapshot(self, name, data):
                return name, data

            def save_tensor_grid(self, tensor_map, destination, title="", max_items=4):
                self.grid_calls.append((Path(destination).name, max_items, list(tensor_map.keys())))
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
        train_module.save_post_train_samples(
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

        self.assertEqual(run_manager.grid_calls[0][0], "post_train_sample_grid.png")
        self.assertEqual(run_manager.grid_calls[0][1], 3)
        self.assertEqual(run_manager.grid_calls[1][0], "post_train_denoising_camera_trace.png")
        self.assertEqual(run_manager.grid_calls[1][1], 3)
        self.assertEqual(run_manager.metrics_payload["overall"]["count"], 3)
        self.assertEqual(sorted(run_manager.metrics_payload["per_class"].keys()), ["0", "1", "2"])
        self.assertEqual(run_manager.metrics_payload["nearest_neighbor"]["reference_source"], "train_only")
        self.assertEqual(run_manager.metrics_payload["nearest_neighbor"]["overall"]["count"], 3)
        self.assertEqual(
            sorted(run_manager.metrics_payload["nearest_neighbor"]["per_class"].keys()),
            ["0", "1", "2"],
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for post-train sample tests")
    def test_post_train_sample_labels_support_weak_class_repeats(self) -> None:
        import torch

        train_module = load_train_module()
        labels = train_module.build_post_train_sample_labels(
            torch,
            num_classes=10,
            device=torch.device("cpu"),
            sample_class_labels=(3, 4, 5),
            sample_repeats_per_class=4,
        )
        default_labels = train_module.build_post_train_sample_labels(
            torch,
            num_classes=10,
            device=torch.device("cpu"),
        )

        self.assertEqual(labels.tolist(), [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5])
        self.assertEqual(default_labels.tolist(), list(range(10)))
        with self.assertRaisesRegex(ValueError, "sample_repeats_per_class"):
            train_module.build_post_train_sample_labels(
                torch,
                num_classes=10,
                device=torch.device("cpu"),
                sample_class_labels=(3,),
                sample_repeats_per_class=0,
            )
        with self.assertRaisesRegex(ValueError, "sample_class_labels"):
            train_module.build_post_train_sample_labels(
                torch,
                num_classes=10,
                device=torch.device("cpu"),
                sample_class_labels=(10,),
            )

    def test_should_validate_epoch_keeps_first_interval_and_last_epoch(self) -> None:
        should_validate_epoch = load_train_module().should_validate_epoch

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
        train_module = load_train_module()

        checkpoint = "/tmp/run/checkpoints/first_stage_best_optical.pt"
        self.assertEqual(train_module.require_strict_optical_best_checkpoint(checkpoint), checkpoint)
        with self.assertRaisesRegex(RuntimeError, "strict optical-only first-stage best checkpoint"):
            train_module.require_strict_optical_best_checkpoint(None)

    def test_validate_stage_requirements_enforces_sample_checkpoints(self) -> None:
        validate_stage_requirements = load_train_module().validate_stage_requirements

        validate_stage_requirements("sample", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_sample", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_distill_export", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_distill_filter", None, None, None, "distill.pt")
        validate_stage_requirements("one_shot_distill", "first.pt", "diffusion.pt", None, "distill.pt")
        validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", "one_shot.pt", "distill.pt")
        with self.assertRaisesRegex(ValueError, "first-stage-checkpoint"):
            validate_stage_requirements("sample", None, "diffusion.pt")
        with self.assertRaisesRegex(ValueError, "diffusion-checkpoint"):
            validate_stage_requirements("sample", "first.pt", None)
        with self.assertRaisesRegex(ValueError, "one-shot-checkpoint"):
            validate_stage_requirements("one_shot_sample", "first.pt", "diffusion.pt", None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("one_shot_distill", "first.pt", "diffusion.pt", None, None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("teacher_distill_filter", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "one-shot-checkpoint"):
            validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", None, "distill.pt")
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", "one_shot.pt", None)

    def test_final_pt_checkpoint_resolution_fills_missing_stage_inputs(self) -> None:
        train_module = load_train_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            final_pt_dir = Path(tmpdir) / "final_pt"
            final_pt_dir.mkdir()
            first_stage_path = final_pt_dir / "first_stage_best_optical.pt"
            diffusion_path = final_pt_dir / "diffusion_best.pt"
            first_stage_path.write_text("first", encoding="utf-8")
            diffusion_path.write_text("diffusion", encoding="utf-8")

            resolved = train_module.resolve_final_pt_stage_checkpoints(
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

            explicit_first = train_module.resolve_final_pt_stage_checkpoints(
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

            diffusion_only = train_module.resolve_final_pt_stage_checkpoints(
                "diffusion",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(diffusion_only["first_stage_checkpoint"], str(first_stage_path))
            self.assertIsNone(diffusion_only["diffusion_checkpoint"])

            distill_export = train_module.resolve_final_pt_stage_checkpoints(
                "teacher_distill_export",
                None,
                None,
                final_pt_dir,
                "first_stage_best_optical.pt",
                "diffusion_best.pt",
            )
            self.assertEqual(distill_export["first_stage_checkpoint"], str(first_stage_path))
            self.assertEqual(distill_export["diffusion_checkpoint"], str(diffusion_path))

            distill_filter = train_module.resolve_final_pt_stage_checkpoints(
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

        train_module = load_train_module()
        config = TrainingConfig()
        train_module.apply_cli_overrides(
            config,
            Namespace(
                stage="first_stage",
                batch_size=8,
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
                distill_output="outputs/runs/distill.pt",
            ),
        )

        self.assertEqual(config.mode, "teacher_first_stage")
        self.assertEqual(config.batch_size, 8)
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
        self.assertEqual(config.diffusion.class_weight_labels, (3, 4, 5))
        self.assertAlmostEqual(config.diffusion.class_weight, 1.5, places=6)
        self.assertAlmostEqual(config.first_stage.camera_artifact_weight, 0.25, places=6)
        self.assertEqual(config.first_stage.encoder_fusion_mode, "optical_only")
        self.assertEqual(config.first_stage.optical_encoder_input_mode, "pf2_intensity_multiscale_v2")
        self.assertEqual(config.first_stage.optical_encoder_arch, "global_context")
        self.assertEqual(config.first_stage.metasurface_decoder_arch, "conv")
        self.assertEqual(config.distillation.samples_per_class, 7)
        self.assertEqual(config.distillation.total_samples, 70)
        self.assertEqual(config.distillation.output_path, "outputs/runs/distill.pt")
        self.assertAlmostEqual(config.first_stage.image_fusion_start_weight, 0.0, places=6)
        self.assertAlmostEqual(config.first_stage.image_fusion_end_weight, 0.0, places=6)
        self.assertEqual(config.first_stage.image_fusion_decay_epochs, 1)

    def test_apply_cli_overrides_rejects_invalid_image_fusion_weight(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_module = load_train_module()
        for invalid_weight in (-0.1, 1.1):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "image_fusion_weight"):
                    train_module.apply_cli_overrides(
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

    def test_apply_cli_overrides_rejects_invalid_diffusion_overrides(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_module = load_train_module()

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
            train_module.apply_cli_overrides(TrainingConfig(), args_with(guidance_scale=-1.0))
        for invalid_dropout in (-0.1, 1.5):
            with self.subTest(invalid_dropout=invalid_dropout):
                with self.assertRaisesRegex(ValueError, "cond_dropout_prob"):
                    train_module.apply_cli_overrides(
                        TrainingConfig(),
                        args_with(cond_dropout_prob=invalid_dropout),
                    )
        with self.assertRaisesRegex(ValueError, "sample_class_labels"):
            train_module.apply_cli_overrides(TrainingConfig(), args_with(sample_class_labels=[10]))
        with self.assertRaisesRegex(ValueError, "sample_repeats_per_class"):
            train_module.apply_cli_overrides(TrainingConfig(), args_with(sample_repeats_per_class=0))
        with self.assertRaisesRegex(ValueError, "diffusion_class_weight_labels"):
            train_module.apply_cli_overrides(TrainingConfig(), args_with(diffusion_class_weight_labels=[10]))
        with self.assertRaisesRegex(ValueError, "class_weight"):
            train_module.apply_cli_overrides(TrainingConfig(), args_with(diffusion_class_weight=0.0))

    def test_apply_cli_overrides_rejects_invalid_camera_artifact_weight(self) -> None:
        from argparse import Namespace

        from ldm_project.config import TrainingConfig

        train_module = load_train_module()
        with self.assertRaisesRegex(ValueError, "camera_artifact_weight"):
            train_module.apply_cli_overrides(
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


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for train main smoke tests")
class TrainMainValidationTests(unittest.TestCase):
    def test_main_records_train_val_metadata_and_validation_history(self) -> None:
        import sys
        import io

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        import ldm_project
        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, PhysicsConfig, TrainingConfig
        from ldm_project.data import DatasetBundle, DatasetSplits

        train_module = load_train_module()
        grid_calls = []

        class DummyFirstStage(nn.Module):
            def __init__(self, physics, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.latent_channels = config.latent_channels
                self.image_size = config.image_size

            def forward(self, batch):
                scaled = batch * self.weight
                zeros = torch.zeros_like(batch)
                return {
                    "electronic_pattern": scaled,
                    "slm_phase": zeros,
                    "pf2_intensity": scaled,
                    "x_hat": scaled,
                    "image_recon": scaled,
                    "camera_plane": scaled,
                    "chip_amplitude": scaled.abs(),
                    "chip_phase": zeros,
                    "probe_plane": scaled,
                    "mu": zeros,
                    "logvar": zeros,
                }

            def encode(self, batch):
                spatial_size = max(1, self.image_size // 8)
                base = batch.mean(dim=(-2, -1), keepdim=True)
                z = base.expand(batch.shape[0], self.latent_channels, spatial_size, spatial_size) * self.weight
                return {"mu": z, "z": z}

            def decode_with_intermediates(self, latents):
                image = latents.mean(dim=1, keepdim=True)
                image = F.interpolate(image, size=(self.image_size, self.image_size), mode="nearest")
                zeros = torch.zeros_like(image)
                return {
                    "image_recon": image,
                    "chip_real": image,
                    "chip_imag": zeros,
                    "chip_amplitude": image.abs(),
                    "chip_phase": zeros,
                    "probe_plane": image,
                    "camera_plane": image,
                }

        class DummyDiffusionModel(nn.Module):
            def __init__(self, latent_channels, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.25))
                self.config = config

            def conditioning_metadata(self):
                return {
                    "type": "mnist_class",
                    "num_classes": self.config.num_classes,
                    "context_dim": self.config.context_dim,
                    "cond_dropout_prob": self.config.cond_dropout_prob,
                }

            def encode_condition(self, class_labels, dropout_prob=None, force_unconditional=False):
                return torch.zeros((class_labels.shape[0], 1), device=class_labels.device)

            def forward(self, noisy_latents, timesteps, context):
                return torch.zeros_like(noisy_latents) + self.weight

        class DummyScheduler:
            def __init__(self, timesteps=2, beta_start=1.0e-4, beta_end=2.0e-2) -> None:
                self.timesteps = timesteps

            def to(self, device):
                return self

            def add_noise(self, clean, noise, timesteps):
                return clean + noise

            def step(self, model_output, timesteps, sample):
                return sample

        original_save_tensor_grid = ldm_project.RunManager.save_tensor_grid

        def capture_save_tensor_grid(self, tensor_map, destination, title="", max_items=4):
            grid_calls.append((Path(destination).name, max_items, list(tensor_map.keys())))
            return original_save_tensor_grid(self, tensor_map, destination, title=title, max_items=max_items)

        images = torch.rand(10, 1, 8, 8)
        labels = torch.arange(10, dtype=torch.long)
        train_dataset = TensorDataset(images[:5], labels[:5])
        validation_dataset = TensorDataset(images[5:], labels[5:])
        splits = DatasetSplits(
            train=DatasetBundle(
                dataset=train_dataset,
                loader=DataLoader(train_dataset, batch_size=2, shuffle=False),
                effective_size=5,
                total_size=5,
            ),
            validation=DatasetBundle(
                dataset=validation_dataset,
                loader=DataLoader(validation_dataset, batch_size=2, shuffle=False),
                effective_size=5,
                total_size=5,
            ),
            total_size=10,
        )
        train_config = TrainingConfig(
            mode="full",
            batch_size=2,
            first_stage_epochs=1,
            diffusion_epochs=1,
            data=DataConfig(
                dataset="mnist",
                root="data",
                download=False,
                subset_fraction=1.0,
                validation_fraction=0.5,
                validation_split_seed=9,
                num_workers=0,
                pin_memory=False,
            ),
            first_stage=FirstStageConfig(
                image_size=8,
                latent_channels=1,
                encoder_channels=(4, 8, 16),
            ),
            diffusion=DiffusionConfig(
                timesteps=2,
                base_channels=4,
                channel_multipliers=(1,),
                attention_levels=(),
                context_dim=4,
                num_classes=10,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir, \
            mock.patch.object(ldm_project, "load_physics_config", return_value=PhysicsConfig()), \
            mock.patch.object(ldm_project, "load_training_config", return_value=train_config), \
            mock.patch.object(ldm_project, "build_train_validation_loaders", return_value=splits), \
            mock.patch.object(ldm_project, "OpticalPathVAE", DummyFirstStage), \
            mock.patch.object(ldm_project, "LatentDiffusionUNet", DummyDiffusionModel), \
            mock.patch.object(ldm_project, "LinearNoiseScheduler", DummyScheduler), \
            mock.patch.object(ldm_project.RunManager, "save_tensor_grid", capture_save_tensor_grid), \
            mock.patch.object(sys, "argv", ["train.py", "--device", "cpu", "--output-root", tmpdir, "--stage", "full"]), \
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout_buffer:
            train_module.main()

            run_dirs = list(Path(tmpdir).iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
            first_stage_visual_config = json.loads(
                (run_dir / "configs" / "first_stage_visual_config.json").read_text(encoding="utf-8")
            )
            diffusion_visual_config = json.loads(
                (run_dir / "configs" / "diffusion_visual_config.json").read_text(encoding="utf-8")
            )
            post_train_config = json.loads((run_dir / "configs" / "post_train_sample_config.json").read_text(encoding="utf-8"))
            trace_config = json.loads(
                (run_dir / "configs" / "post_train_denoising_trace_config.json").read_text(encoding="utf-8")
            )
            sample_metrics = json.loads(
                (run_dir / "artifacts" / "post_train_sample_metrics.json").read_text(encoding="utf-8")
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                post_train_tensors = torch.load(run_dir / "artifacts" / "post_train_sample_tensors.pt", map_location="cpu")
                trace_tensors = torch.load(
                    run_dir / "artifacts" / "post_train_denoising_trace_tensors.pt",
                    map_location="cpu",
                )
                best_first_stage = torch.load(
                    run_dir / "checkpoints" / "first_stage_best_optical.pt",
                    map_location="cpu",
                )
                best_diffusion = torch.load(
                    run_dir / "checkpoints" / "diffusion_best.pt",
                    map_location="cpu",
                )
            stdout_text = stdout_buffer.getvalue()

        self.assertEqual(meta["dataset_total_size"], 10)
        self.assertEqual(meta["train_split_size"], 5)
        self.assertEqual(meta["train_effective_size"], 5)
        self.assertEqual(meta["validation_size"], 5)
        self.assertAlmostEqual(meta["validation_fraction"], 0.5, places=6)
        self.assertEqual(meta["validation_split_seed"], 9)
        self.assertEqual(meta["subset_fraction_applies_to"], "train_only")
        self.assertEqual(meta["planned_first_stage_epochs"], 1)
        self.assertEqual(meta["planned_diffusion_epochs"], 1)
        self.assertEqual(meta["visual_save_epoch_interval"], 1)
        self.assertEqual(meta["validation_epoch_interval"], 1)
        self.assertEqual(meta["first_stage_model_version"], "physical_first_linear_metasurface_vae_v5")
        self.assertEqual(meta["first_stage_encoder_fusion_mode"], "optical_only")
        self.assertEqual(meta["first_stage_optical_encoder_input_mode"], "pf2_intensity")
        self.assertEqual(meta["first_stage_optical_encoder_arch"], "conv")
        self.assertEqual(meta["first_stage_metasurface_decoder_arch"], "conv")
        self.assertEqual(meta["first_stage_linear_metasurface_grid_size"], 50)
        self.assertAlmostEqual(meta["first_stage_linear_metasurface_layer_spacing"], 0.003, places=6)
        self.assertAlmostEqual(meta["first_stage_field_energy_weight"], 0.0, places=6)
        self.assertIsNone(meta["subset_size"])
        self.assertAlmostEqual(meta["image_fusion_start_weight"], 1.0, places=6)
        self.assertAlmostEqual(meta["image_fusion_end_weight"], 0.0, places=6)
        self.assertEqual(meta["image_fusion_decay_epochs"], 20)
        self.assertEqual(
            meta["first_stage_best_optical_checkpoint"],
            str(run_dir / "checkpoints" / "first_stage_best_optical.pt"),
        )
        self.assertEqual(meta["first_stage_best_optical_epoch"], 1)
        self.assertEqual(meta["first_stage_best_optical_metric"], "camera_complex_balanced_mnist_score")
        self.assertIsNotNone(meta["first_stage_best_optical_score"])
        self.assertTrue(meta["first_stage_best_optical_loaded_for_diffusion"])
        self.assertEqual(meta["diffusion_latent_source"], "mu")
        self.assertAlmostEqual(meta["diffusion_guidance_scale"], 3.0, places=6)
        self.assertEqual(meta["diffusion_class_weight_labels"], [])
        self.assertAlmostEqual(meta["diffusion_class_weight"], 1.0, places=6)
        self.assertEqual(meta["sample_class_labels"], [])
        self.assertEqual(meta["sample_repeats_per_class"], 1)
        self.assertEqual(
            meta["diffusion_best_checkpoint"],
            str(run_dir / "checkpoints" / "diffusion_best.pt"),
        )
        self.assertEqual(meta["diffusion_best_epoch"], 1)
        self.assertEqual(meta["diffusion_best_metric"], "diffusion_validation.loss")
        self.assertAlmostEqual(meta["diffusion_best_loss"], history["diffusion_validation"][0]["loss"], places=6)
        self.assertTrue(meta["diffusion_best_loaded_for_sampling"])
        self.assertEqual(meta["sample_metric_source"], "artifacts/post_train_sample_metrics.json")
        self.assertIn("first_stage_validation", history)
        self.assertIn("first_stage_validation_per_class", history)
        self.assertIn("first_stage_best_optical", history)
        self.assertIn("diffusion_validation", history)
        self.assertIn("diffusion_validation_per_class", history)
        self.assertIn("diffusion_best", history)
        self.assertEqual(len(history["first_stage_validation"]), 1)
        self.assertEqual(len(history["first_stage_validation_per_class"]), 1)
        self.assertEqual(len(history["first_stage_best_optical"]), 1)
        self.assertEqual(len(history["diffusion_validation"]), 1)
        self.assertEqual(len(history["diffusion_validation_per_class"]), 1)
        self.assertEqual(len(history["diffusion_best"]), 1)
        self.assertEqual(history["first_stage_validation"][0]["epoch"], 1)
        self.assertEqual(history["first_stage_validation_per_class"][0]["epoch"], 1)
        self.assertTrue(history["first_stage_best_optical"][0]["eligible"])
        self.assertTrue(history["first_stage_best_optical"][0]["updated"])
        self.assertEqual(history["first_stage_best_optical"][0]["selection_metric"], "camera_complex_balanced_mnist_score")
        self.assertEqual(
            history["first_stage_best_optical"][0]["checkpoint"],
            str(run_dir / "checkpoints" / "first_stage_best_optical.pt"),
        )
        self.assertEqual(history["diffusion_validation"][0]["epoch"], 1)
        self.assertEqual(history["diffusion_validation_per_class"][0]["epoch"], 1)
        self.assertEqual(history["diffusion_validation_per_class"][0]["classes"]["5"]["count"], 1)
        self.assertIsNotNone(history["diffusion_validation_per_class"][0]["classes"]["5"]["loss"])
        self.assertTrue(history["diffusion_best"][0]["updated"])
        self.assertEqual(history["diffusion_best"][0]["selection_metric"], "diffusion_validation.loss")
        self.assertEqual(
            history["diffusion_best"][0]["checkpoint"],
            str(run_dir / "checkpoints" / "diffusion_best.pt"),
        )
        self.assertIn("image_recon_loss", history["first_stage_validation"][0])
        self.assertIn("camera_recon_loss", history["first_stage_validation"][0])
        self.assertIn("structure_loss", history["first_stage_validation"][0])
        self.assertIn("shape_loss", history["first_stage_validation"][0])
        self.assertIn("field_smoothness_loss", history["first_stage_validation"][0])
        self.assertIn("phase_tv", history["first_stage_validation"][0])
        self.assertIn("field_energy", history["first_stage_validation"][0])
        self.assertIn("field_energy_loss", history["first_stage_validation"][0])
        self.assertIn("background_leakage_loss", history["first_stage_validation"][0])
        self.assertIn("camera_background_leakage", history["first_stage_validation"][0])
        self.assertIn("camera_hotspot_loss", history["first_stage_validation"][0])
        self.assertIn("camera_hotspot_ratio", history["first_stage_validation"][0])
        self.assertIn("camera_foreground_dice", history["first_stage_validation"][0])
        self.assertIn("camera_saturation_ratio", history["first_stage_validation"][0])
        self.assertIn("camera_topology_loss", history["first_stage_validation"][0])
        self.assertIn("camera_topology_base", history["first_stage_validation"][0])
        self.assertIn("camera_saturation_loss", history["first_stage_validation"][0])
        self.assertIn("camera_saturation_penalty", history["first_stage_validation"][0])
        self.assertIn("camera_artifact_loss", history["first_stage_validation"][0])
        self.assertIn("camera_foreground_hole_loss", history["first_stage_validation"][0])
        self.assertIn("camera_foreground_hole_ratio", history["first_stage_validation"][0])
        self.assertIn("camera_foreground_roughness", history["first_stage_validation"][0])
        self.assertIn("camera_corr", history["first_stage_validation"][0])
        self.assertIn("camera_edge_corr", history["first_stage_validation"][0])
        self.assertIn("min_class_camera_corr", history["first_stage_validation"][0])
        self.assertIn("camera_balanced_score", history["first_stage_validation"][0])
        self.assertIn("camera_mnist_score", history["first_stage_validation"][0])
        self.assertIn("min_class_camera_mnist_score", history["first_stage_validation"][0])
        self.assertIn("camera_class_balanced_mnist_score", history["first_stage_validation"][0])
        self.assertIn("complex_camera_corr", history["first_stage_validation"][0])
        self.assertIn("complex_camera_edge_corr", history["first_stage_validation"][0])
        self.assertIn("complex_camera_foreground_dice", history["first_stage_validation"][0])
        self.assertIn("complex_camera_mnist_score", history["first_stage_validation"][0])
        self.assertIn("camera_complex_balanced_mnist_score", history["first_stage_validation"][0])
        self.assertEqual(first_stage_visual_config["class_labels"], list(range(10)))
        self.assertEqual(first_stage_visual_config["reference_source"], "validation_then_train_fallback")
        self.assertEqual(diffusion_visual_config["class_labels"], list(range(10)))
        self.assertEqual(diffusion_visual_config["reference_labels"], list(range(10)))
        self.assertEqual(diffusion_visual_config["reference_source"], "validation_then_train_fallback")
        diffusion_epoch_grids = [
            call for call in grid_calls
            if call[0] == "epoch_0001.png" and "generated_camera" in call[2]
        ]
        self.assertEqual(len(diffusion_epoch_grids), 1)
        self.assertEqual(diffusion_epoch_grids[0][1], 10)
        self.assertEqual(
            sorted(history["first_stage_validation_per_class"][0]["classes"].keys()),
            ["5", "6", "7", "8", "9"],
        )
        self.assertIn(
            "camera_fg_mse",
            history["first_stage_validation_per_class"][0]["classes"]["5"],
        )
        self.assertIn(
            "camera_corr",
            history["first_stage_validation_per_class"][0]["classes"]["5"],
        )
        self.assertIn(
            "camera_foreground_dice",
            history["first_stage_validation_per_class"][0]["classes"]["5"],
        )
        self.assertEqual(post_train_config["reference_labels"], list(range(10)))
        self.assertEqual(post_train_config["reference_source"], "validation_then_train_fallback")
        self.assertIn("reference_images", post_train_tensors)
        self.assertIn("reference_labels", post_train_tensors)
        self.assertIn("generated_camera", post_train_tensors)
        self.assertIn("generated_image", post_train_tensors)
        self.assertIn("chip_amplitude", post_train_tensors)
        self.assertIn("chip_phase", post_train_tensors)
        self.assertEqual(tuple(post_train_tensors["reference_images"].shape), (10, 1, 8, 8))
        self.assertEqual(post_train_tensors["reference_labels"].tolist(), list(range(10)))
        self.assertEqual(post_train_tensors["reference_source"], "validation_then_train_fallback")
        self.assertEqual(tuple(post_train_tensors["generated_camera"].shape), (10, 1, 8, 8))
        self.assertEqual(tuple(post_train_tensors["generated_image"].shape), (10, 1, 8, 8))
        self.assertEqual(trace_config["reference_labels"], list(range(10)))
        self.assertEqual(sample_metrics["overall"]["count"], 10)
        self.assertEqual(sorted(sample_metrics["per_class"].keys()), [str(label) for label in range(10)])
        self.assertEqual(sample_metrics["nearest_neighbor"]["reference_source"], "validation_then_train_fallback")
        self.assertEqual(sample_metrics["nearest_neighbor"]["overall"]["count"], 10)
        self.assertEqual(
            sorted(sample_metrics["nearest_neighbor"]["per_class"].keys()),
            [str(label) for label in range(10)],
        )
        self.assertEqual(trace_config["trace_columns"], ["reference", "noise", "t750", "t000"])
        self.assertEqual([step["label"] for step in trace_config["trace_steps"]], ["t750", "t000"])
        self.assertEqual(trace_tensors["reference_labels"].tolist(), list(range(10)))
        self.assertEqual(list(trace_tensors["latent_trace"].keys()), ["noise", "t750", "t000"])
        self.assertEqual(list(trace_tensors["camera_trace"].keys()), ["noise", "t750", "t000"])
        for camera_trace in trace_tensors["camera_trace"].values():
            self.assertEqual(tuple(camera_trace.shape), (10, 1, 8, 8))
        self.assertEqual(best_first_stage["epoch"], 1)
        self.assertEqual(best_first_stage["selection_metric"], "camera_complex_balanced_mnist_score")
        self.assertIn("validation_metrics", best_first_stage)
        self.assertEqual(best_diffusion["epoch"], 1)
        self.assertEqual(best_diffusion["selection_metric"], "diffusion_validation.loss")
        self.assertAlmostEqual(best_diffusion["selection_loss"], history["diffusion_validation"][0]["loss"], places=6)
        self.assertIn("validation_metrics", best_diffusion)
        self.assertIn("image_fusion_weight", best_first_stage)
        self.assertIn("planned epochs: first_stage=1 diffusion=1", stdout_text)
        self.assertIn("loaded best optical first stage for diffusion", stdout_text)

    def test_main_sample_mode_loads_checkpoints_and_skips_training_history(self) -> None:
        import sys
        import io

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        import ldm_project
        from ldm_project import FIRST_STAGE_MODEL_VERSION
        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, PhysicsConfig, TrainingConfig
        from ldm_project.data import DatasetBundle, DatasetSplits
        from ldm_project.diffusion import DIFFUSION_MODEL_VERSION

        train_module = load_train_module()

        class DummyFirstStage(nn.Module):
            def __init__(self, physics, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.latent_channels = config.latent_channels
                self.image_size = config.image_size

            def set_image_fusion_weight(self, weight) -> None:
                self.image_fusion_weight = float(weight)

            def decode_with_intermediates(self, latents):
                image = latents.mean(dim=1, keepdim=True)
                image = F.interpolate(image, size=(self.image_size, self.image_size), mode="nearest")
                zeros = torch.zeros_like(image)
                return {
                    "image_recon": image,
                    "chip_real": image,
                    "chip_imag": zeros,
                    "chip_amplitude": image.abs(),
                    "chip_phase": zeros,
                    "probe_plane": image,
                    "camera_plane": image,
                }

        class DummyDiffusionModel(nn.Module):
            def __init__(self, latent_channels, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.25))
                self.config = config

            def conditioning_metadata(self):
                return {
                    "type": "mnist_class",
                    "num_classes": self.config.num_classes,
                    "context_dim": self.config.context_dim,
                    "cond_dropout_prob": self.config.cond_dropout_prob,
                }

            def encode_condition(self, class_labels, dropout_prob=None, force_unconditional=False):
                return torch.zeros((class_labels.shape[0], 1), device=class_labels.device)

            def forward(self, sample, timesteps, context):
                return torch.zeros_like(sample) + self.weight

        class DummyScheduler:
            def __init__(self, timesteps=2, beta_start=1.0e-4, beta_end=2.0e-2) -> None:
                self.timesteps = timesteps

            def to(self, device):
                return self

            def step(self, model_output, timesteps, sample):
                return sample

        images = torch.rand(10, 1, 8, 8)
        labels = torch.arange(10, dtype=torch.long)
        train_dataset = TensorDataset(images[:5], labels[:5])
        validation_dataset = TensorDataset(images[5:], labels[5:])
        splits = DatasetSplits(
            train=DatasetBundle(
                dataset=train_dataset,
                loader=DataLoader(train_dataset, batch_size=2, shuffle=False),
                effective_size=5,
                total_size=5,
            ),
            validation=DatasetBundle(
                dataset=validation_dataset,
                loader=DataLoader(validation_dataset, batch_size=2, shuffle=False),
                effective_size=5,
                total_size=5,
            ),
            total_size=10,
        )
        train_config = TrainingConfig(
            mode="first_stage",
            batch_size=2,
            data=DataConfig(
                dataset="mnist",
                root="data",
                download=False,
                subset_fraction=1.0,
                validation_fraction=0.5,
                num_workers=0,
                pin_memory=False,
            ),
            first_stage=FirstStageConfig(
                image_size=8,
                latent_channels=1,
                encoder_channels=(4, 8, 16),
            ),
            diffusion=DiffusionConfig(
                timesteps=2,
                base_channels=4,
                channel_multipliers=(1,),
                attention_levels=(),
                context_dim=4,
                num_classes=10,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            physics = PhysicsConfig()
            first_stage_checkpoint = checkpoint_root / "first_stage.pt"
            diffusion_checkpoint = checkpoint_root / "diffusion.pt"
            torch.save(
                {
                    "model_state_dict": DummyFirstStage(physics, train_config.first_stage).state_dict(),
                    "stage": "first_stage",
                    "model_version": FIRST_STAGE_MODEL_VERSION,
                    "physics_config": physics,
                    "train_config": train_config,
                },
                first_stage_checkpoint,
            )
            torch.save(
                {
                    "model_state_dict": DummyDiffusionModel(
                        train_config.first_stage.latent_channels,
                        train_config.diffusion,
                    ).state_dict(),
                    "latent_scale": torch.tensor(0.75),
                    "latent_source": train_config.diffusion.latent_source,
                    "stage": "diffusion",
                    "model_version": DIFFUSION_MODEL_VERSION,
                    "first_stage_model_version": FIRST_STAGE_MODEL_VERSION,
                    "conditioning": {
                        "type": "mnist_class",
                        "num_classes": train_config.diffusion.num_classes,
                        "context_dim": train_config.diffusion.context_dim,
                        "cond_dropout_prob": train_config.diffusion.cond_dropout_prob,
                    },
                    "physics_config": physics,
                    "train_config": train_config,
                },
                diffusion_checkpoint,
            )

            argv = [
                "train.py",
                "--device",
                "cpu",
                "--output-root",
                str(output_root),
                "--stage",
                "sample",
                "--first-stage-checkpoint",
                str(first_stage_checkpoint),
                "--diffusion-checkpoint",
                str(diffusion_checkpoint),
                "--guidance-scale",
                "3.5",
                "--sample-class-labels",
                "3",
                "4",
                "5",
                "--sample-repeats-per-class",
                "4",
            ]
            with mock.patch.object(ldm_project, "load_physics_config", return_value=physics), \
                mock.patch.object(ldm_project, "load_training_config", return_value=train_config), \
                mock.patch.object(ldm_project, "build_train_validation_loaders", return_value=splits), \
                mock.patch.object(ldm_project, "OpticalPathVAE", DummyFirstStage), \
                mock.patch.object(ldm_project, "LatentDiffusionUNet", DummyDiffusionModel), \
                mock.patch.object(ldm_project, "LinearNoiseScheduler", DummyScheduler), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout_buffer:
                train_module.main()

            run_dirs = list(output_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            post_train_config = json.loads(
                (run_dir / "configs" / "post_train_sample_config.json").read_text(encoding="utf-8")
            )
            sample_metrics = json.loads(
                (run_dir / "artifacts" / "post_train_sample_metrics.json").read_text(encoding="utf-8")
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                post_train_tensors = torch.load(
                    run_dir / "artifacts" / "post_train_sample_tensors.pt",
                    map_location="cpu",
                )
            stdout_text = stdout_buffer.getvalue()
            history_exists = (run_dir / "training_history.json").exists()

        self.assertEqual(meta["mode"], "teacher_sample")
        self.assertEqual(meta["planned_first_stage_epochs"], 0)
        self.assertEqual(meta["planned_diffusion_epochs"], 0)
        self.assertEqual(meta["diffusion_checkpoint"], str(diffusion_checkpoint))
        self.assertAlmostEqual(meta["diffusion_guidance_scale"], 3.5, places=6)
        self.assertEqual(meta["sample_class_labels"], [3, 4, 5])
        self.assertEqual(meta["sample_repeats_per_class"], 4)
        self.assertEqual(meta["sample_metric_source"], "artifacts/post_train_sample_metrics.json")
        self.assertEqual(meta["latent_scale_source"], "diffusion_checkpoint")
        self.assertTrue(meta["diffusion_best_loaded_for_sampling"])
        self.assertFalse(history_exists)
        self.assertEqual(post_train_config["class_labels"], [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5])
        self.assertEqual(post_train_config["reference_labels"], [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5])
        self.assertEqual(post_train_config["sample_class_labels"], [3, 4, 5])
        self.assertEqual(post_train_config["sample_repeats_per_class"], 4)
        self.assertEqual(sample_metrics["overall"]["count"], 12)
        self.assertEqual(sorted(sample_metrics["per_class"].keys()), ["3", "4", "5"])
        self.assertEqual(sample_metrics["nearest_neighbor"]["reference_source"], "validation_then_train_fallback")
        self.assertEqual(sample_metrics["nearest_neighbor"]["overall"]["count"], 12)
        self.assertEqual(sorted(sample_metrics["nearest_neighbor"]["per_class"].keys()), ["3", "4", "5"])
        self.assertAlmostEqual(float(post_train_tensors["guidance_scale"].item()), 3.5, places=6)
        self.assertEqual(tuple(post_train_tensors["generated_camera"].shape), (12, 1, 8, 8))
        self.assertIn("loaded diffusion checkpoint for sampling", stdout_text)

    def test_main_rejects_full_diffusion_without_strict_optical_best(self) -> None:
        import sys
        import io

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        import ldm_project
        from ldm_project.config import DataConfig, DiffusionConfig, FirstStageConfig, PhysicsConfig, TrainingConfig
        from ldm_project.data import DatasetBundle, DatasetSplits

        train_module = load_train_module()

        class DummyFirstStage(nn.Module):
            def __init__(self, physics, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.latent_channels = config.latent_channels
                self.image_size = config.image_size

            def set_image_fusion_weight(self, weight) -> None:
                self.image_fusion_weight = float(weight)

            def forward(self, batch):
                scaled = batch * self.weight
                zeros = torch.zeros_like(batch)
                return {
                    "electronic_pattern": scaled,
                    "slm_phase": zeros,
                    "pf2_intensity": scaled,
                    "x_hat": scaled,
                    "image_recon": scaled,
                    "camera_plane": scaled,
                    "chip_amplitude": scaled.abs(),
                    "chip_phase": zeros,
                    "probe_plane": scaled,
                    "mu": zeros,
                    "logvar": zeros,
                }

        class DummyDiffusionModel(nn.Module):
            def __init__(self, latent_channels, config) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.25))
                self.config = config

            def conditioning_metadata(self):
                return {"type": "mnist_class", "num_classes": self.config.num_classes}

        class DummyScheduler:
            def __init__(self, timesteps=2, beta_start=1.0e-4, beta_end=2.0e-2) -> None:
                self.timesteps = timesteps

            def to(self, device):
                return self

        images = torch.rand(8, 1, 8, 8)
        labels = torch.arange(8, dtype=torch.long)
        train_dataset = TensorDataset(images[:4], labels[:4])
        validation_dataset = TensorDataset(images[4:], labels[4:])
        splits = DatasetSplits(
            train=DatasetBundle(
                dataset=train_dataset,
                loader=DataLoader(train_dataset, batch_size=2, shuffle=False),
                effective_size=4,
                total_size=4,
            ),
            validation=DatasetBundle(
                dataset=validation_dataset,
                loader=DataLoader(validation_dataset, batch_size=2, shuffle=False),
                effective_size=4,
                total_size=4,
            ),
            total_size=8,
        )
        train_config = TrainingConfig(
            mode="full",
            batch_size=2,
            first_stage_epochs=1,
            diffusion_epochs=1,
            data=DataConfig(
                dataset="mnist",
                root="data",
                download=False,
                subset_fraction=1.0,
                validation_fraction=0.5,
                num_workers=0,
                pin_memory=False,
            ),
            first_stage=FirstStageConfig(
                image_size=8,
                latent_channels=1,
                encoder_channels=(4, 8, 16),
                encoder_fusion_mode="gated_dual_path",
                image_fusion_start_weight=0.3,
                image_fusion_end_weight=0.3,
                image_fusion_decay_epochs=1,
            ),
            diffusion=DiffusionConfig(
                timesteps=2,
                base_channels=4,
                channel_multipliers=(1,),
                attention_levels=(),
                context_dim=4,
                num_classes=10,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir, \
            mock.patch.object(ldm_project, "load_physics_config", return_value=PhysicsConfig()), \
            mock.patch.object(ldm_project, "load_training_config", return_value=train_config), \
            mock.patch.object(ldm_project, "build_train_validation_loaders", return_value=splits), \
            mock.patch.object(ldm_project, "OpticalPathVAE", DummyFirstStage), \
            mock.patch.object(ldm_project, "LatentDiffusionUNet", DummyDiffusionModel), \
            mock.patch.object(ldm_project, "LinearNoiseScheduler", DummyScheduler), \
            mock.patch.object(sys, "argv", ["train.py", "--device", "cpu", "--output-root", tmpdir, "--stage", "full"]), \
            mock.patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "strict optical-only first-stage best checkpoint"):
                train_module.main()

            run_dir = next(Path(tmpdir).iterdir())
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))

        self.assertIsNone(meta["first_stage_best_optical_checkpoint"])
        self.assertFalse(history["first_stage_best_optical"][0]["eligible"])


if __name__ == "__main__":
    unittest.main()
