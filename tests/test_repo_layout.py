from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepoLayoutTests(unittest.TestCase):
    def test_expected_scaffold_files_exist(self) -> None:
        expected = [
            ROOT / "README.md",
            ROOT / "pyproject.toml",
            ROOT / "pyrightconfig.json",
            ROOT / "train.py",
            ROOT / "configs" / "physics_cfg.yaml",
            ROOT / "configs" / "train_ldm.yaml",
            ROOT / "configs" / "export_teacher_distill_mnist_3k_fast200.yaml",
            ROOT / "configs" / "export_teacher_distill_mnist_15k_fast200.yaml",
            ROOT / "configs" / "filter_teacher_distill_mnist_3k_fast200_top2k.yaml",
            ROOT / "configs" / "filter_teacher_distill_mnist_3k_fast200_label_strict_top1k.yaml",
            ROOT / "configs" / "split_teacher_distill_readable_top1k_80_10_10.yaml",
            ROOT / "configs" / "audit_one_shot_d2nn_v8_7a4_optical_identity_chain.yaml",
            ROOT
            / "configs"
            / "audit_one_shot_d2nn_v9_10_label_template_inverse_prototype_baseline.yaml",
            ROOT
            / "configs"
            / "train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_label_template_inverse_prototype_baseline.yaml",
            ROOT
            / "configs"
            / "eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_label_template_inverse_prototype_baseline_strict_readability_test.yaml",
            ROOT / "configs" / "train_first_stage_hardclass_camera_recovery.yaml",
            ROOT / "configs" / "train_one_shot_distill_d2nn_mnist_3k_fast200.yaml",
            ROOT / "configs" / "train_one_shot_distill_d2nn_gpu_feasibility.yaml",
            ROOT / "configs" / "train_one_shot_distill_d2nn_long.yaml",
            ROOT / "configs" / "train_one_shot_d2nn_smoke.yaml",
            ROOT / "docs" / "ldm_migration" / "single_shot_ldm_distillation.md",
            ROOT / "docs" / "ldm_migration" / "stage_guide.md",
            ROOT / "docs" / "ldm_migration" / "acceptance_baseline_c99f559.json",
            ROOT / "docs" / "ldm_migration" / "acceptance_baseline_bs128_matched_diffusion.json",
            ROOT / "docs" / "ldm_migration" / "acceptance_seed_sweep_ebba8e9.json",
            ROOT / "docs" / "ldm_migration" / "acceptance_post_refactor_one_shot_smoke_8a307f3.json",
            ROOT / "src" / "ldm_project" / "data.py",
            ROOT / "src" / "ldm_project" / "first_stage.py",
            ROOT / "src" / "ldm_project" / "diffusion.py",
            ROOT / "src" / "ldm_project" / "one_shot.py",
            ROOT / "src" / "ldm_project" / "one_shot_model.py",
            ROOT / "src" / "ldm_project" / "one_shot_optical_layers.py",
            ROOT / "src" / "ldm_project" / "evaluation.py",
            ROOT / "src" / "ldm_project" / "run_manager.py",
            ROOT / "src" / "ldm_project" / "d2nn_diagnostics.py",
            ROOT / "src" / "ldm_project" / "d2nn_visualization.py",
            ROOT / "src" / "ldm_project" / "diffusion_losses.py",
            ROOT / "src" / "ldm_project" / "diffusion_sampling.py",
            ROOT / "src" / "ldm_project" / "first_stage_losses.py",
            ROOT / "src" / "ldm_project" / "one_shot_visualization.py",
            ROOT / "src" / "ldm_project" / "one_shot_losses.py",
            ROOT / "src" / "ldm_project" / "sampling.py",
            ROOT / "src" / "ldm_project" / "sampling_reports.py",
            ROOT / "src" / "ldm_project" / "stage_constants.py",
            ROOT / "src" / "ldm_project" / "stage_runners.py",
            ROOT / "src" / "ldm_project" / "train_cli.py",
            ROOT / "src" / "ldm_project" / "train_orchestration.py",
            ROOT / "src" / "ldm_project" / "train_run_context.py",
            ROOT / "src" / "ldm_project" / "train_utils.py",
            ROOT / "src" / "ldm_project" / "training_loss_common.py",
            ROOT / "src" / "ldm_project" / "visualization_core.py",
            ROOT / "src" / "ldm_project" / "one_shot_distill_data.py",
            ROOT / "src" / "ldm_project" / "one_shot_distill_training.py",
            ROOT / "src" / "ldm_project" / "one_shot_eval.py",
            ROOT / "src" / "ldm_project" / "one_shot_eval_artifacts.py",
            ROOT / "src" / "ldm_project" / "one_shot_eval_metrics.py",
            ROOT / "src" / "ldm_project" / "one_shot_identity_audit.py",
            ROOT / "src" / "ldm_project" / "one_shot_quality.py",
            ROOT / "src" / "ldm_project" / "one_shot_stage_common.py",
            ROOT / "src" / "ldm_project" / "one_shot_training.py",
            ROOT / "src" / "ldm_project" / "one_shot_training_utils.py",
            ROOT / "src" / "ldm_project" / "teacher_distill.py",
            ROOT / "src" / "ldm_project" / "teacher_fault_localization.py",
            ROOT / "src" / "ldm_project" / "teacher_sampling.py",
            ROOT / "src" / "ldm_project" / "teacher_stages.py",
            ROOT / "src" / "ldm_project" / "teacher_training.py",
            ROOT / "src" / "ldm_project" / "teacher_training_utils.py",
            ROOT / "src" / "ldm_project" / "one_shot_stages.py",
            ROOT / "src" / "ldm_project" / "eval_reports.py",
            ROOT / "tests" / "test_teacher_training_utils.py",
            ROOT / "tests" / "test_eval_reports.py",
            ROOT / "tools" / "run_v9_10_label_template_audit.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"missing expected file: {path}")

        self.assertFalse((ROOT / "train_ldm.py").exists(), "legacy entry script should be removed: train_ldm.py")
        self.assertFalse((ROOT / "sample_ldm.py").exists(), "legacy entry script should be removed: sample_ldm.py")

    def test_train_entry_delegates_to_package_cli_main(self) -> None:
        from ldm_project.train_cli import main

        spec = importlib.util.spec_from_file_location("train_entry", ROOT / "train.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIs(module.main, main)

    def test_stage_shims_keep_legacy_imports_available(self) -> None:
        from ldm_project import one_shot_stages, sampling, stage_runners, teacher_stages, training
        from ldm_project.d2nn_diagnostics import build_one_shot_d2nn_diagnostics
        from ldm_project.diffusion_losses import apply_classifier_free_guidance, diffusion_loss
        from ldm_project.diffusion_sampling import sample_latents
        from ldm_project.first_stage_losses import first_stage_loss, select_diffusion_latents
        from ldm_project.one_shot_losses import one_shot_loss
        from ldm_project.one_shot_visualization import build_one_shot_visualization_tensors
        from ldm_project.one_shot_distill_training import run_one_shot_distill_training
        from ldm_project.one_shot_eval import (
            _save_one_shot_d2nn_focus_visualization as legacy_focus_visualization,
            compute_one_shot_distill_eval_metrics as legacy_eval_metrics,
            run_one_shot_sample_stage,
        )
        from ldm_project.one_shot_eval_artifacts import _save_one_shot_d2nn_focus_visualization
        from ldm_project.one_shot_eval_metrics import compute_one_shot_distill_eval_metrics
        from ldm_project.one_shot_training import run_one_shot_training
        from ldm_project.sampling_reports import save_post_train_samples, select_post_train_sample_batch
        from ldm_project.train_cli import main
        from ldm_project.train_orchestration import run_selected_stage
        from ldm_project.teacher_distill import run_teacher_distill_export
        from ldm_project.teacher_sampling import run_sample_stage
        from ldm_project.teacher_training import run_diffusion_training, run_first_stage_training
        from ldm_project.visualization_core import save_visualization_grids

        self.assertEqual(main.__module__, "ldm_project.train_cli")
        self.assertEqual(run_selected_stage.__module__, "ldm_project.train_orchestration")
        self.assertIs(sampling.build_one_shot_d2nn_diagnostics, build_one_shot_d2nn_diagnostics)
        self.assertIs(sampling.build_one_shot_visualization_tensors, build_one_shot_visualization_tensors)
        self.assertIs(sampling.save_post_train_samples, save_post_train_samples)
        self.assertIs(sampling.save_visualization_grids, save_visualization_grids)
        self.assertIs(sampling.select_post_train_sample_batch, select_post_train_sample_batch)
        self.assertIs(training.apply_classifier_free_guidance, apply_classifier_free_guidance)
        self.assertIs(training.diffusion_loss, diffusion_loss)
        self.assertIs(training.first_stage_loss, first_stage_loss)
        self.assertIs(training.one_shot_loss, one_shot_loss)
        self.assertIs(training.sample_latents, sample_latents)
        self.assertIs(training.select_diffusion_latents, select_diffusion_latents)
        self.assertIs(teacher_stages.run_first_stage_training, run_first_stage_training)
        self.assertIs(teacher_stages.run_diffusion_training, run_diffusion_training)
        self.assertIs(teacher_stages.run_sample_stage, run_sample_stage)
        self.assertIs(teacher_stages.run_teacher_distill_export, run_teacher_distill_export)
        self.assertIs(one_shot_stages.run_one_shot_training, run_one_shot_training)
        self.assertIs(one_shot_stages.run_one_shot_distill_training, run_one_shot_distill_training)
        self.assertIs(one_shot_stages.run_one_shot_sample_stage, run_one_shot_sample_stage)
        self.assertIs(legacy_eval_metrics, compute_one_shot_distill_eval_metrics)
        self.assertIs(legacy_focus_visualization, _save_one_shot_d2nn_focus_visualization)
        self.assertIs(stage_runners.run_first_stage_training, run_first_stage_training)
        self.assertIs(stage_runners.run_one_shot_distill_training, run_one_shot_distill_training)

    def test_d2nn_design_doc_is_main_source(self) -> None:
        design_doc = ROOT / "docs" / "ldm_migration" / "single_shot_ldm_distillation.md"
        content = design_doc.read_text(encoding="utf-8")
        self.assertIn("D2NN one-shot", content)
        self.assertIn("configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml", content)
        self.assertIn("diffractive_phase_only", content)
        self.assertIn("N = 200", content)
        self.assertIn("d = 5.0 mm", content)

    def test_one_shot_distill_train_and_eval_configs_do_not_use_relay_identity(self) -> None:
        config_dir = ROOT / "configs"
        checked = 0
        for pattern in ("train_one_shot_distill*.yaml", "eval_one_shot_distill*.yaml"):
            for path in config_dir.glob(pattern):
                checked += 1
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("relay_identity", content, f"{path.name} should use free-space D2NN optics")
        self.assertGreater(checked, 0)

    def test_removed_historical_docs_are_absent(self) -> None:
        removed_docs = [
            ROOT / "docs" / "ldm_migration" / "final_acceptance.md",
            ROOT / "docs" / "ldm_migration" / "final_delivery.md",
            ROOT / "docs" / "ldm_migration" / "ldm_architecture_spec.md",
            ROOT / "docs" / "ldm_migration" / "ldm_project_workflow.md",
            ROOT / "docs" / "ldm_migration" / "ldm_reuse_matrix.md",
        ]
        for path in removed_docs:
            self.assertFalse(path.exists(), f"historical migration doc should be removed: {path}")

    def test_train_config_defaults_to_mnist_subset(self) -> None:
        config_path = ROOT / "configs" / "train_ldm.yaml"
        content = config_path.read_text(encoding="utf-8")
        self.assertIn("dataset: mnist", content)
        self.assertIn("validation_fraction: 0.1", content)
        self.assertIn("subset_fraction: 0.1667", content)
        self.assertIn("encoder_fusion_mode: optical_only", content)
        self.assertIn("optical_encoder_input_mode: pf2_complex_field", content)
        self.assertIn("optical_encoder_arch: linear_metasurface", content)
        self.assertIn("metasurface_decoder_arch: linear_metasurface", content)
        self.assertIn("linear_metasurface_grid_size: 40", content)
        self.assertIn("linear_metasurface_layer_spacing: 0.003", content)
        self.assertIn("image_fusion_start_weight: 0.0", content)
        self.assertIn("image_fusion_end_weight: 0.0", content)
        self.assertIn("image_fusion_decay_epochs: 1", content)
        self.assertIn("diffusion_epochs: 0", content)
        self.assertIn("field_smoothness_weight: 0.01", content)
        self.assertIn("phase_smoothness_weight: 0.005", content)
        self.assertIn("field_energy_weight: 0.01", content)
        self.assertIn("background_leakage_weight: 0.2", content)
        self.assertIn("camera_hotspot_weight: 0.25", content)
        self.assertIn("camera_topology_weight: 0.15", content)
        self.assertIn("camera_saturation_weight: 0.05", content)
        self.assertIn("optical_field_mode: complex_cartesian", content)
        self.assertIn("primary_output: camera", content)
        self.assertIn("mode: mnist_complex_slm", content)
        self.assertIn("latent_source: mu", content)
        self.assertIn("base_channels: 64", content)


if __name__ == "__main__":
    unittest.main()
