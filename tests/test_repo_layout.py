from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepoLayoutTests(unittest.TestCase):
    @staticmethod
    def _project_imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module)
        return {name for name in imports if name == "ldm_project" or name.startswith("ldm_project.")}

    def test_active_scripts_are_not_import_path_mutators(self) -> None:
        script_dirs = (ROOT / "scripts" / "maintenance", ROOT / "scripts" / "visualization")

        offenders = []
        for directory in script_dirs:
            for path in sorted(directory.glob("*.py")):
                if "sys.path" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])
        self.assertTrue((ROOT / "experiments" / "archive" / "v57_v67" / "tools").is_dir())

    def test_research_code_is_grouped_and_closed_families_are_archived(self) -> None:
        families = {"lightgen", "scaled_chen", "optical_manifold", "capacity", "shared"}
        scripts_root = ROOT / "scripts" / "research"
        runtime_root = ROOT / "src" / "ldm_project" / "v2" / "application" / "research"
        configs_root = ROOT / "configs" / "research"
        tests_root = ROOT / "tests" / "research"

        for directory in (scripts_root, runtime_root):
            self.assertFalse(
                [path.name for path in directory.glob("*.py") if path.name != "__init__.py"],
                str(directory.relative_to(ROOT)),
            )
        for directory, expected_families in (
            (scripts_root, families - {"shared"}),
            (runtime_root, families - {"scaled_chen"}),
            (configs_root, families),
            (tests_root, families - {"shared"}),
        ):
            present = {
                path.relative_to(directory).parts[0]
                for path in directory.rglob("*")
                if path.is_dir() and path.name != "__pycache__"
            }
            self.assertTrue(expected_families.issubset(present), str(directory.relative_to(ROOT)))

        active_m0 = ROOT / "scripts" / "research" / "lightgen" / "m0"
        archived_m0 = ROOT / "scripts" / "research" / "archive" / "project_refactor_20261001" / "lightgen" / "m0"
        active_n3 = ROOT / "scripts" / "research" / "lightgen" / "n3"
        archived_n3 = ROOT / "scripts" / "research" / "archive" / "project_refactor_20261001" / "lightgen" / "n3"
        self.assertFalse(active_m0.exists())
        self.assertTrue(any(path.name.endswith(".py.txt") for path in archived_m0.iterdir()))
        self.assertFalse(active_n3.exists())
        self.assertTrue(any(path.name.endswith(".py.txt") for path in archived_n3.iterdir()))

    def test_namespaces_and_legacy_facades_are_isolated(self) -> None:
        package_root = ROOT / "src" / "ldm_project"
        for namespace in ("v1", "v2"):
            self.assertTrue((package_root / namespace / "__init__.py").is_file())

        facade_files = [
            path
            for path in package_root.glob("*.py")
            if path.name not in {"__init__.py", "_compat.py"}
        ]
        for path in facade_files:
            content = path.read_text(encoding="utf-8")
            self.assertTrue(
                "sys.modules[__name__]" in content or "globals().update" in content or "from .v1" in content,
                path.name,
            )

        for package in ("application", "configuration", "diagnostics", "distillation", "evaluation", "generation", "optics", "student", "teacher"):
            content = (package_root / package / "__init__.py").read_text(encoding="utf-8")
            self.assertIn("alias_package", content, package)

        forbidden = []
        for namespace in ("v1", "v2"):
            for path in (package_root / namespace).rglob("*.py"):
                imports = self._project_imports(path)
                if path == package_root / "v2" / "application" / "legacy_adapter.py":
                    self.assertIn("ldm_project.v1.config", imports)
                    imports.discard("ldm_project.v1.config")
                forbidden.extend(
                    f"{path.relative_to(ROOT)} imports {name}"
                    for name in sorted(imports)
                    if (namespace == "v1" and name.startswith("ldm_project.v2"))
                    or (namespace == "v2" and name.startswith("ldm_project.v1"))
                )
        self.assertEqual(forbidden, [])

    def test_current_runtime_is_physically_isolated_and_split(self) -> None:
        application = ROOT / "src" / "ldm_project" / "v2" / "application"
        self.assertLessEqual(len((application / "cli.py").read_text(encoding="utf-8").splitlines()), 250)
        self.assertLessEqual(len((application / "identity_runtime.py").read_text(encoding="utf-8").splitlines()), 100)
        for name in ("runtime.py", "admission.py", "recovery.py", "training.py", "evaluation.py", "dataset.py"):
            path = application / "current" / name
            self.assertTrue(path.is_file(), path)
            self.assertLessEqual(len(path.read_text(encoding="utf-8").splitlines()), 2000)
        for name in ("runtime.py", "admission.py", "recovery.py", "diversity.py", "class_modes.py", "parity.py", "legacy_training.py", "evaluation.py", "multi_anchor.py"):
            path = application / "diagnostic" / name
            self.assertTrue(path.is_file(), path)
            self.assertLessEqual(len(path.read_text(encoding="utf-8").splitlines()), 2000)
        for name in ("cli_impl.py", "teacher.py", "prototypes.py", "geometry.py", "templates.py"):
            path = application / "legacy" / name
            self.assertTrue(path.is_file(), path)
            self.assertLessEqual(len(path.read_text(encoding="utf-8").splitlines()), 2000)
        current_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (application / "current").glob("*.py")
        )
        for forbidden_import in (
            "ldm_project.v1",
            "student.capacity",
            "student.direct_template",
            "optics.lightgen",
            "optics.fiber_ols",
            "from ..diagnostic",
            "from ...diagnostic",
        ):
            self.assertNotIn(forbidden_import, current_source)

    def test_expected_scaffold_files_exist(self) -> None:
        expected = [
            ROOT / "README.md",
            ROOT / "pyproject.toml",
            ROOT / "pyrightconfig.json",
            ROOT / "train.py",
            ROOT / "configs" / "v1" / "formal" / "physics_cfg.yaml",
            ROOT / "configs" / "v1" / "formal" / "train_ldm.yaml",
            ROOT / "configs" / "v1" / "teacher" / "export_teacher_distill_mnist_3k_fast200.yaml",
            ROOT / "configs" / "v1" / "teacher" / "export_teacher_distill_mnist_15k_fast200.yaml",
            ROOT / "configs" / "v1" / "teacher" / "filter_teacher_distill_mnist_3k_fast200_top2k.yaml",
            ROOT / "configs" / "v1" / "teacher" / "filter_teacher_distill_mnist_3k_fast200_label_strict_top1k.yaml",
            ROOT / "configs" / "v1" / "teacher" / "split_teacher_distill_readable_top1k_80_10_10.yaml",
            ROOT / "experiments" / "archive" / "manifest.yaml",
            ROOT / "configs" / "v1" / "teacher" / "train_first_stage_hardclass_camera_recovery.yaml",
            ROOT / "configs" / "v1" / "smoke" / "train_one_shot_d2nn_smoke.yaml",
            ROOT / "docs" / "architecture" / "v1_single_shot_distillation.md",
            ROOT / "docs" / "operations" / "stage_guide.md",
            ROOT / "docs" / "archive" / "acceptance_baseline_c99f559.json",
            ROOT / "docs" / "archive" / "acceptance_baseline_bs128_matched_diffusion.json",
            ROOT / "docs" / "archive" / "acceptance_seed_sweep_ebba8e9.json",
            ROOT / "docs" / "archive" / "acceptance_post_refactor_one_shot_smoke_8a307f3.json",
            ROOT / "src" / "ldm_project" / "v1" / "application" / "cli.py",
            ROOT / "src" / "ldm_project" / "v1" / "optics" / "d2nn.py",
            ROOT / "src" / "ldm_project" / "v1" / "student" / "model.py",
            ROOT / "src" / "ldm_project" / "v2" / "application" / "cli.py",
            ROOT / "src" / "ldm_project" / "data.py",
            ROOT / "src" / "ldm_project" / "first_stage.py",
            ROOT / "src" / "ldm_project" / "diffusion.py",
            ROOT / "src" / "ldm_project" / "one_shot.py",
            ROOT / "src" / "ldm_project" / "one_shot_model.py",
            ROOT / "src" / "ldm_project" / "one_shot_optical_layers.py",
            ROOT / "src" / "ldm_project" / "evaluation" / "__init__.py",
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
            ROOT / "experiments" / "archive" / "v57_v67" / "tools" / "diagnose_v59_upstream_separation_isolation.py",
            ROOT / "scripts" / "maintenance" / "organize_outputs_by_version.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"missing expected file: {path}")

        self.assertFalse((ROOT / "train_ldm.py").exists(), "legacy entry script should be removed: train_ldm.py")
        self.assertFalse((ROOT / "sample_ldm.py").exists(), "legacy entry script should be removed: sample_ldm.py")

    def test_current_legacy_diagnostic_taxonomy(self) -> None:
        current = ROOT / "configs" / "v2" / "current" / "formal_identity_logit_amplitude_finite_crop_identity.yaml"
        legacy = ROOT / "configs" / "v2" / "legacy" / "formal_global_amplitude_finite_crop_identity.yaml"
        diagnostic = ROOT / "configs" / "v2" / "diagnostic" / "lightgen_amplitude_fiber_ols_400.yaml"
        migration = ROOT / "configs" / "v2" / "path_migration_manifest.yaml"
        for path in (current, legacy, diagnostic, migration):
            self.assertTrue(path.is_file(), path)
        self.assertTrue((ROOT / "src" / "ldm_project" / "v1" / "AGENTS.md").is_file())
        self.assertTrue((ROOT / "src" / "ldm_project" / "v2" / "AGENTS.md").is_file())
        active_runtime = (ROOT / "docs" / "architecture" / "v2_runtime.md").read_text(encoding="utf-8")
        self.assertIn("conditional_identity_logit_amplitude_v2", active_runtime)
        self.assertNotIn("teacher pair/target manifest", active_runtime)
        self.assertNotIn("prototype-reachability", active_runtime)
        current_configs = sorted((ROOT / "configs" / "v2" / "current").glob("*.yaml"))
        self.assertIn(current, current_configs)
        self.assertIn(
            ROOT / "configs" / "v2" / "current" / "formal_spatial_quadrature_v3.yaml",
            current_configs,
        )

    def test_train_entry_delegates_to_package_cli_main(self) -> None:
        from ldm_project.train_cli import main

        spec = importlib.util.spec_from_file_location("train_entry", ROOT / "train.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIs(module.main, main)

    def _load_output_tool(self):
        tool_path = ROOT / "scripts" / "maintenance" / "organize_outputs_by_version.py"
        spec = importlib.util.spec_from_file_location("organize_outputs_by_version", tool_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_output_version_reorder_detects_conflicts_without_overwrite(self) -> None:
        module = self._load_output_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs"
            source = root / "v9_8" / "runs" / "20260620_102234_train_one_shot_d2nn_v9_8_audit"
            target = root / "20260620_v9_8" / "runs" / source.name
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            (source / "run_meta.json").write_text("{}", encoding="utf-8")
            (target / "run_meta.json").write_text('{"target": true}', encoding="utf-8")

            dry_run_manifest = module.reorganize_outputs(root, dry_run=True, tracked_files=[])
            self.assertEqual(dry_run_manifest["counts"]["versions_conflict_duplicate_target"], 2)
            self.assertTrue(source.is_dir())
            self.assertEqual((target / "run_meta.json").read_text(encoding="utf-8"), '{"target": true}')

    def test_output_version_reorder_renames_and_rewrites_exact_refs(self) -> None:
        module = self._load_output_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs"
            real_run = root / "v9_8" / "runs" / "20260620_102234_train_one_shot_d2nn_v9_8_audit"
            real_run.mkdir(parents=True)
            doc = Path(tmpdir) / "doc.md"
            doc.write_text(
                "outputs/v9_8/runs/20260620_102234_train_one_shot_d2nn_v9_8_audit\n"
                "outputs/<version>/runs/<run>\n"
                "outputs/v9_8_placeholder\n",
                encoding="utf-8",
            )

            dry_run = module.reorganize_outputs(root, dry_run=True, tracked_files=[doc])
            self.assertEqual(dry_run["counts"]["versions_planned_rename"], 1)
            self.assertEqual(dry_run["counts"]["rewrite_files"], 1)
            self.assertTrue((root / "v9_8").is_dir())

            apply_manifest = module.reorganize_outputs(root, dry_run=False, tracked_files=[doc])
            self.assertEqual(apply_manifest["counts"]["versions_renamed"], 1)
            self.assertFalse((root / "v9_8").exists())
            self.assertTrue((root / "20260620_v9_8" / "runs" / real_run.name).is_dir())
            self.assertIn(
                "outputs/20260620_v9_8/runs/20260620_102234_train_one_shot_d2nn_v9_8_audit",
                doc.read_text(encoding="utf-8"),
            )
            self.assertIn("outputs/<version>/runs/<run>", doc.read_text(encoding="utf-8"))
            self.assertIn("outputs/v9_8_placeholder", doc.read_text(encoding="utf-8"))

    def test_output_version_reorder_refuses_missing_timestamped_runs(self) -> None:
        module = self._load_output_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs"
            real_run = root / "v9_8" / "runs" / "real_run_without_timestamp"
            real_run.mkdir(parents=True)

            dry_run = module.reorganize_outputs(root, dry_run=True, tracked_files=[])
            self.assertEqual(dry_run["counts"]["versions_blocker_missing_timestamped_runs"], 1)
            with self.assertRaisesRegex(RuntimeError, "blockers"):
                module.reorganize_outputs(root, dry_run=False, tracked_files=[])
            self.assertTrue(real_run.is_dir())

    def test_output_version_index_is_oldest_first_by_earliest_run_time(self) -> None:
        module = self._load_output_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs"
            fixtures = {
                "v10": "20260523_224644_train_v10",
                "v9_8": "20260620_102112_train_v9_8",
                "unversioned": "20260514_161039_train_unversioned",
                "v5": "20260520_195637_train_v5",
            }
            for version, run_name in fixtures.items():
                (root / version / "runs" / run_name).mkdir(parents=True)

            json_path, md_path, index = module.write_version_index(root)
            self.assertEqual(
                [entry["directory"] for entry in index],
                ["unversioned", "v5", "v10", "v9_8"],
            )
            markdown = md_path.read_text(encoding="utf-8")
            self.assertLess(markdown.index("`v5`"), markdown.index("`v9_8`"))
            self.assertTrue(json_path.is_file())

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

        from ldm_project.v1.application.cli import main as canonical_main
        from ldm_project.v1.application.stages import run_selected_stage as canonical_stage

        self.assertIs(main, canonical_main)
        self.assertIs(run_selected_stage, canonical_stage)
        self.assertEqual(main.__module__, "ldm_project.v1.application.cli")
        self.assertEqual(run_selected_stage.__module__, "ldm_project.v1.application.stages")
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
        design_doc = ROOT / "docs" / "architecture" / "v1_single_shot_distillation.md"
        content = design_doc.read_text(encoding="utf-8")
        self.assertIn("D2NN one-shot", content)
        self.assertIn("configs/v1/formal/train_v56_coherent_relay_from_scratch.yaml", content)
        self.assertIn("diffractive_phase_only", content)
        self.assertIn("100 × 100 @ 2 µm", content)
        self.assertIn("d = 1.50 mm", content)

    def test_one_shot_distill_train_and_eval_configs_do_not_use_relay_identity(self) -> None:
        config_dir = ROOT / "configs" / "v1" / "formal"
        checked = 0
        for pattern in ("train_v56*.yaml", "eval_v56*.yaml"):
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
        config_path = ROOT / "configs" / "v1" / "formal" / "train_ldm.yaml"
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
