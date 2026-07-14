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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_module():
    import ldm_project.train_cli as train_cli

    return train_cli


def load_train_utils_module():
    import ldm_project.train_utils as train_utils

    return train_utils


def load_sampling_module():
    import ldm_project.sampling as sampling

    return sampling


def load_evaluation_module():
    import ldm_project.evaluation as evaluation

    return evaluation


def _load_tool_module(module_name: str, filename: str):
    tools_dir = PROJECT_ROOT / "tools"
    if "tools" not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            "tools",
            tools_dir / "__init__.py",
            submodule_search_locations=[str(tools_dir)],
        )
        if package_spec is None or package_spec.loader is None:
            raise RuntimeError(f"could not load tools package: {tools_dir}")
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["tools"] = package
        package_spec.loader.exec_module(package)

    tool_path = PROJECT_ROOT / "tools" / filename
    qualified_name = f"tools.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def load_v16_readout_calibration_tool():
    return _load_tool_module("diagnose_v16_readout_calibration", "diagnose_v16_readout_calibration.py")


def load_v17_terminal_readability_tool():
    return _load_tool_module("diagnose_v17_terminal_readability", "diagnose_v17_terminal_readability.py")


def load_v19_front_transfer_tool():
    return _load_tool_module("diagnose_v19_front_transfer", "diagnose_v19_front_transfer.py")


def load_v20_downstream_plane_gate_tool():
    return _load_tool_module("diagnose_v20_downstream_plane_gate", "diagnose_v20_downstream_plane_gate.py")


def load_v21_physical_cmos_readout_tool():
    return _load_tool_module("diagnose_v21_physical_cmos_readout", "diagnose_v21_physical_cmos_readout.py")


def load_v23_decoder_chip_semantic_tool():
    return _load_tool_module(
        "diagnose_v23_decoder_chip_semantic_collapse",
        "diagnose_v23_decoder_chip_semantic_collapse.py",
    )


def load_v24_ols_decoder_expressivity_tool():
    return _load_tool_module("diagnose_v24_ols_decoder_expressivity", "diagnose_v24_ols_decoder_expressivity.py")


def load_v25_decoder_capacity_tool():
    return _load_tool_module("diagnose_v25_decoder_capacity_gate", "diagnose_v25_decoder_capacity_gate.py")


def load_v26_ols_decodability_tool():
    return _load_tool_module("diagnose_v26_ols_decodability_oracle", "diagnose_v26_ols_decodability_oracle.py")


def load_v27_decoder_synthesis_gap_tool():
    return _load_tool_module("diagnose_v27_decoder_synthesis_gap", "diagnose_v27_decoder_synthesis_gap.py")


def load_v28_decoder_controllability_tool():
    return _load_tool_module("diagnose_v28_decoder_controllability", "diagnose_v28_decoder_controllability.py")


def load_v28_1_tiny_replay_tool():
    return _load_tool_module("build_v28_1_tiny_replay_payload", "build_v28_1_tiny_replay_payload.py")


def load_v28_1_same_batch_overfit_tool():
    return _load_tool_module("diagnose_v28_1_same_batch_overfit", "diagnose_v28_1_same_batch_overfit.py")


def load_v29_decoder_trainability_tool():
    return _load_tool_module(
        "diagnose_v29_decoder_trainability_target_encoding",
        "diagnose_v29_decoder_trainability_target_encoding.py",
    )


def load_v30_target_code_alignment_tool():
    return _load_tool_module("diagnose_v30_target_code_alignment", "diagnose_v30_target_code_alignment.py")


def load_v30_1_sample_gate_recovery_tool():
    return _load_tool_module("diagnose_v30_1_sample_gate_recovery", "diagnose_v30_1_sample_gate_recovery.py")


def load_v31_bridge_topology_tool():
    return _load_tool_module("diagnose_v31_bridge_topology", "diagnose_v31_bridge_topology.py")


def load_v32_1_sample_gap_isolation_tool():
    return _load_tool_module("diagnose_v32_1_sample_gap_isolation", "diagnose_v32_1_sample_gap_isolation.py")


def load_v33_code_conditioned_sample_gap_tool():
    return _load_tool_module("diagnose_v33_code_conditioned_sample_gap", "diagnose_v33_code_conditioned_sample_gap.py")


def load_v33_2_real_ols_candidate_preview_tool():
    return _load_tool_module(
        "diagnose_v33_2_real_ols_candidate_preview",
        "diagnose_v33_2_real_ols_candidate_preview.py",
    )


def load_v34_code_coupling_recovery_tool():
    return _load_tool_module("diagnose_v34_code_coupling_recovery", "diagnose_v34_code_coupling_recovery.py")


def load_v35_optical_code_carrier_isolation_tool():
    return _load_tool_module(
        "diagnose_v35_optical_code_carrier_isolation",
        "diagnose_v35_optical_code_carrier_isolation.py",
    )


def load_v36_high_gain_code_coupled_training_tool():
    return _load_tool_module(
        "diagnose_v36_high_gain_code_coupled_training",
        "diagnose_v36_high_gain_code_coupled_training.py",
    )


def load_v37_final_cmos_decoder_topology_tool():
    return _load_tool_module(
        "diagnose_v37_final_cmos_decoder_topology",
        "diagnose_v37_final_cmos_decoder_topology.py",
    )


def load_v38_formal_cmos_manual_review_tool():
    return _load_tool_module(
        "diagnose_v38_formal_cmos_manual_review",
        "diagnose_v38_formal_cmos_manual_review.py",
    )


def load_v39_human_topology_false_positive_tool():
    return _load_tool_module(
        "diagnose_v39_human_topology_false_positive",
        "diagnose_v39_human_topology_false_positive.py",
    )


def load_v40_sample_code_diversity_target_representation_tool():
    return _load_tool_module(
        "diagnose_v40_sample_code_diversity_target_representation",
        "diagnose_v40_sample_code_diversity_target_representation.py",
    )


def load_v41_decoder_physical_topology_recovery_tool():
    return _load_tool_module(
        "diagnose_v41_decoder_physical_topology_recovery",
        "diagnose_v41_decoder_physical_topology_recovery.py",
    )


def load_v42_decoder_bypass_backpath_isolation_tool():
    return _load_tool_module(
        "diagnose_v42_decoder_bypass_backpath_isolation",
        "diagnose_v42_decoder_bypass_backpath_isolation.py",
    )


def load_v43_decoder_propagation_geometry_isolation_tool():
    return _load_tool_module(
        "diagnose_v43_decoder_propagation_geometry_isolation",
        "diagnose_v43_decoder_propagation_geometry_isolation.py",
    )


def load_v44_target_code_spatial_scaling_decoder_recovery_tool():
    return _load_tool_module(
        "diagnose_v44_target_code_spatial_scaling_decoder_recovery",
        "diagnose_v44_target_code_spatial_scaling_decoder_recovery.py",
    )


def load_v45_decoder_spacing_near_field_recovery_tool():
    return _load_tool_module(
        "diagnose_v45_decoder_spacing_near_field_recovery",
        "diagnose_v45_decoder_spacing_near_field_recovery.py",
    )


def load_v46_near_field_semantic_separation_tool():
    return _load_tool_module(
        "diagnose_v46_near_field_semantic_separation",
        "diagnose_v46_near_field_semantic_separation.py",
    )


def load_v47_near_field_decoder_capacity_upper_bound_tool():
    return _load_tool_module(
        "diagnose_v47_near_field_decoder_capacity_upper_bound",
        "diagnose_v47_near_field_decoder_capacity_upper_bound.py",
    )


def load_v48_transparent_decoder_control_tool():
    return _load_tool_module(
        "diagnose_v48_transparent_decoder_control",
        "diagnose_v48_transparent_decoder_control.py",
    )


def load_v49_decoder_mask_component_attribution_tool():
    return _load_tool_module(
        "diagnose_v49_decoder_mask_component_attribution",
        "diagnose_v49_decoder_mask_component_attribution.py",
    )


def load_v50_decoder_phase_layer_attribution_tool():
    return _load_tool_module(
        "diagnose_v50_decoder_phase_layer_attribution",
        "diagnose_v50_decoder_phase_layer_attribution.py",
    )


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
    def test_v16_readout_calibration_tool_writes_diagnostic_report_only(self) -> None:
        import torch

        tool = load_v16_readout_calibration_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_run = root / "outputs" / "20260630_v16" / "runs" / "20260630_000000_train_v16_probe_sample"
            artifacts = sample_run / "artifacts"
            artifacts.mkdir(parents=True)
            labels = torch.tensor([3, 4], dtype=torch.long)
            reference = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
            reference[0, 0, 1:3, 1] = 1.0
            reference[0, 0, 2, 1:3] = 1.0
            reference[1, 0, :, 2] = 1.0
            reference[1, 0, 1, 1:3] = 1.0
            generated = reference.clone()
            torch.save(
                {
                    "class_labels": labels,
                    "reference_labels": labels,
                    "reference_images": reference,
                    "generated_camera_raw": generated,
                },
                artifacts / "one_shot_sample_tensors.pt",
            )

            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_readout_calibration(
                    sample_runs=[Path("outputs/20260630_v16/runs/20260630_000000_train_v16_probe_sample")],
                    output=Path("outputs/reports/v16_readout_test.json"),
                    generate_previews=False,
                )
                repeated = tool.run_readout_calibration(
                    sample_runs=[Path("outputs/20260630_v16/runs/20260630_000000_train_v16_probe_sample")],
                    output=Path("outputs/reports/v16_readout_test_repeat.json"),
                    generate_previews=False,
                )
            finally:
                os.chdir(cwd)

            report_path = root / "outputs" / "reports" / "v16_readout_test.json"
            self.assertTrue(report_path.is_file())
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(written["diagnostic_only"])
            self.assertFalse(written["accepted_summary_written"])
            self.assertEqual(written["decision"]["accepted_summary_written"], False)
            self.assertEqual(summary["decision"], repeated["decision"])
            run = written["runs"][0]
            self.assertTrue(run["decision"]["metric_positive"])
            self.assertFalse(run["top_preview_candidates"][0]["generated"])
            first_transform = run["transforms"][0]
            self.assertIn("exact", first_transform["thresholds"][0])
            self.assertIn("nearest_neighbor", first_transform["thresholds"][0])
            self.assertIn("prototype_label_audit", first_transform["thresholds"][0])
            self.assertIn("3", first_transform["thresholds"][0]["exact"]["focus_classes"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v16 readout diagnostics")
    def test_v16_readout_calibration_tool_fails_loudly_for_bad_inputs(self) -> None:
        import torch

        tool = load_v16_readout_calibration_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_run = root / "outputs" / "20260630_v16" / "runs" / "bad_sample"
            artifacts = bad_run / "artifacts"
            artifacts.mkdir(parents=True)
            torch.save({"class_labels": torch.tensor([3], dtype=torch.long)}, artifacts / "one_shot_sample_tensors.pt")

            with self.assertRaisesRegex(ValueError, "missing required tensors"):
                tool.analyze_sample_run(bad_run, output_dir=root / "outputs" / "reports", generate_previews=False)
            with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                tool.run_readout_calibration(sample_runs=[bad_run], output=Path("bad/report.json"), generate_previews=False)
            with self.assertRaises(FileNotFoundError):
                tool.analyze_sample_run(root / "missing_sample", output_dir=root / "outputs" / "reports", generate_previews=False)

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
    def test_v17_terminal_readability_tool_accepts_perfect_diagnostic_candidate(self) -> None:
        import torch

        tool = load_v17_terminal_readability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v17_terminal_sample(root, "perfect_sample", labels, reference.clone(), reference, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_terminal_readability(
                    sample_runs=[Path("outputs/20260630_v17/runs/perfect_sample")],
                    output=Path("outputs/reports/v17_terminal_perfect.json"),
                )
            finally:
                os.chdir(cwd)

            report_path = root / "outputs" / "reports" / "v17_terminal_perfect.json"
            self.assertTrue(report_path.is_file())
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["accepted_summary_written"])
            self.assertEqual(summary["decision"]["status"], "diagnostic-only-candidate-needs-human-v4-review")
            run = summary["runs"][0]
            self.assertTrue(run["gate"]["candidate"])
            self.assertEqual(run["prototype_label_audit"]["correct_ratio"], 1.0)
            self.assertIsNone(run["first_gross_collapse_plane"])
            self.assertIn("3", run["focus_classes"])
            self.assertEqual(
                run["v4_preview"],
                "outputs/20260630_v17/runs/perfect_sample/images/one_shot_sample_100x100_cell_chain_preview.png",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")
    def test_v17_terminal_readability_tool_rejects_blob_sample(self) -> None:
        import torch

        tool = load_v17_terminal_readability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            generated = torch.ones_like(reference) * 0.5
            self._write_v17_terminal_sample(root, "blob_sample", labels, generated, reference, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_terminal_readability(
                    sample_runs=[Path("outputs/20260630_v17/runs/blob_sample")],
                    output=Path("outputs/reports/v17_terminal_blob.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(summary["decision"]["status"], "diagnostic-only-failed-terminal-readability")
            self.assertFalse(run["gate"]["candidate"])
            self.assertFalse(run["gate"]["checks"]["overall_corr"])
            self.assertIsNotNone(run["first_gross_collapse_plane"])
            self.assertEqual(run["first_gross_collapse_plane"]["plane"], "generator_desired_intensity")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")
    def test_v17_terminal_readability_focus_class_failure_blocks_candidate(self) -> None:
        import torch

        tool = load_v17_terminal_readability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            generated = reference.clone()
            generated[2] = 0.0
            self._write_v17_terminal_sample(root, "focus_fail_sample", labels, generated, reference, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_terminal_readability(
                    sample_runs=[Path("outputs/20260630_v17/runs/focus_fail_sample")],
                    output=Path("outputs/reports/v17_terminal_focus_fail.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertFalse(run["gate"]["candidate"])
            self.assertTrue(run["gate"]["checks"]["overall_corr"])
            self.assertTrue(run["gate"]["checks"]["overall_dice"])
            self.assertFalse(run["gate"]["checks"]["focus_classes"])
            self.assertEqual([failure["label"] for failure in run["gate"]["focus_failures"]], [4])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")
    def test_v17_terminal_readability_tool_fails_loudly_for_bad_inputs(self) -> None:
        import torch

        tool = load_v17_terminal_readability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_run = root / "outputs" / "20260630_v17" / "runs" / "bad_sample"
            artifacts = bad_run / "artifacts"
            artifacts.mkdir(parents=True)
            torch.save({"class_labels": torch.tensor([3], dtype=torch.long)}, artifacts / "one_shot_sample_tensors.pt")
            (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required tensors"):
                tool.analyze_sample_run(bad_run)
            with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                tool.run_terminal_readability(sample_runs=[bad_run], output=Path("bad/report.json"))
            with self.assertRaises(FileNotFoundError):
                tool.analyze_sample_run(root / "outputs/20260630_v17/runs/missing")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v17 terminal diagnostics")
    def test_v17_terminal_readability_summary_extracts_sample_and_eval_runs(self) -> None:
        import torch

        tool = load_v17_terminal_readability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v17_terminal_sample(root, "summary_sample", labels, reference.clone(), reference, torch)
            eval_run = root / "outputs" / "20260630_v17" / "runs" / "summary_eval"
            eval_artifacts = eval_run / "artifacts"
            eval_artifacts.mkdir(parents=True)
            (eval_artifacts / "one_shot_distill_eval_metrics.json").write_text(
                json.dumps({"strict_readability_acceptance": {"passed": True}}),
                encoding="utf-8",
            )
            summary_path = root / "outputs" / "reports" / "launcher_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "sample_runs": [
                            {"sample_run": "outputs/20260630_v17/runs/summary_sample"}
                        ],
                        "probes": [
                            {"eval_run": "outputs/20260630_v17/runs/summary_eval"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_terminal_readability(
                    summary=Path("outputs/reports/launcher_summary.json"),
                    output=Path("outputs/reports/v17_terminal_from_summary.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["source_summary"], "outputs/reports/launcher_summary.json")
            self.assertEqual(summary["sample_runs"], ["outputs/20260630_v17/runs/summary_sample"])
            self.assertEqual(summary["eval_runs"], ["outputs/20260630_v17/runs/summary_eval"])
            self.assertIsNotNone(summary["runs"][0]["eval"]["metrics"])

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
    def test_v19_front_transfer_accepts_retained_encoder_input(self) -> None:
        import torch

        tool = load_v19_front_transfer_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            slm = self._v17_reference_images(torch, labels)
            self._write_v19_front_sample(root, "retained_sample", labels, slm, slm.clone(), torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_front_transfer(
                    sample_runs=[Path("outputs/20260630_v19/runs/retained_sample")],
                    output=Path("outputs/reports/v19_front_retained.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertTrue((root / "outputs/reports/v19_front_retained.json").is_file())
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["accepted_summary_written"])
            self.assertEqual(summary["decision"]["status"], "diagnostic-only-front-transfer-retained")
            run = summary["runs"][0]
            self.assertTrue(run["front_transfer_retained"])
            self.assertIsNone(run["first_gross_collapse_plane"])
            self.assertEqual(run["front_reference_plane"], "slm_command")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v19 front diagnostics")
    def test_v19_front_transfer_flags_encoder_input_collapse(self) -> None:
        import torch

        tool = load_v19_front_transfer_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            slm = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(slm) * 0.5
            self._write_v19_front_sample(root, "collapsed_sample", labels, slm, collapsed, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_front_transfer(
                    sample_runs=[Path("outputs/20260630_v19/runs/collapsed_sample")],
                    output=Path("outputs/reports/v19_front_collapsed.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(summary["decision"]["status"], "diagnostic-only-front-transfer-collapsed")
            self.assertFalse(run["front_transfer_retained"])
            self.assertEqual(run["first_gross_collapse_plane"]["plane"], "encoder_input_amplitude")
            self.assertFalse(run["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v19 front diagnostics")
    def test_v19_front_transfer_fails_loudly_for_bad_inputs(self) -> None:
        import torch

        tool = load_v19_front_transfer_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_run = root / "outputs" / "20260630_v19" / "runs" / "bad_sample"
            artifacts = bad_run / "artifacts"
            artifacts.mkdir(parents=True)
            torch.save({"class_labels": torch.tensor([3], dtype=torch.long)}, artifacts / "one_shot_sample_tensors.pt")
            with self.assertRaisesRegex(ValueError, "missing required tensors"):
                tool.analyze_sample_run(bad_run)
            with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                tool.run_front_transfer(sample_runs=[bad_run], output=Path("bad/report.json"))

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
    def test_v20_downstream_plane_gate_accepts_readable_chain(self) -> None:
        import torch

        tool = load_v20_downstream_plane_gate_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v20_downstream_run(
                root,
                "readable_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_downstream_plane_gate(
                    eval_runs=[Path("outputs/20260630_v20/runs/readable_eval")],
                    output=Path("outputs/reports/v20_downstream_readable.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertTrue((root / "outputs/reports/v20_downstream_readable.json").is_file())
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["accepted_summary_written"])
            run = summary["runs"][0]
            self.assertIsNone(run["first_gross_collapse_plane"])
            self.assertFalse(run["downstream_localized"])
            self.assertEqual(run["prototype_label_audit"]["correct_ratio"], 1.0)
            self.assertEqual(run["best_readable_intermediate_plane"]["plane"], "slm_command")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")
    def test_v20_downstream_plane_gate_reports_encoded_collapse(self) -> None:
        import torch

        tool = load_v20_downstream_plane_gate_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            planes = {"reference": reference}
            for key in (
                "encoded_amplitude",
                "ols_output_amplitude",
                "decoder_output_amplitude",
                "chip_amplitude",
                "chip_intensity_primary",
                "probe_plane",
                "diffractive_camera_plane_raw",
                "generated_camera",
            ):
                planes[key] = collapsed
            self._write_v20_downstream_run(root, "encoded_collapse_eval", "eval", labels, planes, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_downstream_plane_gate(
                    eval_runs=[Path("outputs/20260630_v20/runs/encoded_collapse_eval")],
                    output=Path("outputs/reports/v20_downstream_encoded_collapse.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["first_gross_collapse_plane"]["plane"], "encoded_amplitude")
            self.assertTrue(run["d2nn_failure"])
            self.assertFalse(run["camera_path_failure"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")
    def test_v20_downstream_plane_gate_classifies_camera_path_failure(self) -> None:
        import torch

        tool = load_v20_downstream_plane_gate_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            bad_camera = torch.ones_like(reference) * 0.5
            planes = {
                "reference": reference,
                "diffractive_camera_plane_raw": bad_camera,
                "generated_camera": bad_camera,
            }
            self._write_v20_downstream_run(root, "camera_fail_sample", "sample", labels, planes, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_downstream_plane_gate(
                    sample_runs=[Path("outputs/20260630_v20/runs/camera_fail_sample")],
                    output=Path("outputs/reports/v20_downstream_camera_fail.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertTrue(run["camera_path_failure"])
            self.assertFalse(run["d2nn_failure"])
            self.assertEqual(run["first_gross_collapse_plane"]["plane"], "diffractive_camera_plane_raw")
            self.assertEqual(run["best_readable_intermediate_plane"]["plane"], "slm_command")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")
    def test_v20_downstream_plane_gate_fails_loudly_for_bad_inputs(self) -> None:
        import torch

        tool = load_v20_downstream_plane_gate_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_run = root / "outputs" / "20260630_v20" / "runs" / "bad_eval"
            artifacts = bad_run / "artifacts"
            artifacts.mkdir(parents=True)
            torch.save({"class_labels": torch.tensor([3], dtype=torch.long)}, artifacts / "one_shot_distill_eval_tensors.pt")
            with self.assertRaisesRegex(ValueError, "missing required reference/target images"):
                tool.analyze_run(bad_run, kind="eval")
            with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                tool.run_downstream_plane_gate(eval_runs=[bad_run], output=Path("bad/report.json"))
            with self.assertRaises(FileNotFoundError):
                tool.analyze_run(root / "outputs/20260630_v20/runs/missing_eval", kind="eval")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v20 downstream diagnostics")
    def test_v20_downstream_plane_gate_summary_extracts_eval_and_sample_runs(self) -> None:
        import torch

        tool = load_v20_downstream_plane_gate_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v20_downstream_run(root, "summary_eval", "eval", labels, {"reference": reference}, torch)
            self._write_v20_downstream_run(root, "summary_sample", "sample", labels, {"reference": reference}, torch)
            summary_path = root / "outputs" / "reports" / "v20_launcher_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "probes": [{"eval_run": "outputs/20260630_v20/runs/summary_eval"}],
                        "sample_runs": [{"sample_run": "outputs/20260630_v20/runs/summary_sample"}],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_downstream_plane_gate(
                    summary=Path("outputs/reports/v20_launcher_summary.json"),
                    output=Path("outputs/reports/v20_downstream_from_summary.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["source_summary"], "outputs/reports/v20_launcher_summary.json")
            self.assertEqual(summary["eval_runs"], ["outputs/20260630_v20/runs/summary_eval"])
            self.assertEqual(summary["sample_runs"], ["outputs/20260630_v20/runs/summary_sample"])
            self.assertEqual([run["run_type"] for run in summary["runs"]], ["eval", "sample"])

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
    def test_v21_physical_cmos_readout_flags_selected_vs_physical_mismatch(self) -> None:
        import torch

        tool = load_v21_physical_cmos_readout_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            physical_blob = torch.ones_like(reference) * 0.5
            self._write_v21_physical_cmos_run(
                root,
                "mismatch_eval",
                "eval",
                labels,
                reference,
                physical_blob,
                reference,
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_physical_cmos_readout(
                    eval_runs=[Path("outputs/20260701_v21/runs/mismatch_eval")],
                    output=Path("outputs/reports/v21_physical_mismatch.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["status"], "camera-path-failure")
            self.assertTrue(run["selected_vs_physical_mismatch"])
            self.assertTrue(run["camera_path_failure"])
            self.assertEqual(run["physical_cmos"]["source_tensor"], "physical_camera_plane")
            self.assertEqual(run["selected_readout"]["source_tensor"], "selected_readout")
            self.assertFalse(run["physical_cmos_positive"])
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v21 physical CMOS diagnostics")
    def test_v21_physical_cmos_readout_candidate_requires_human_review(self) -> None:
        import torch

        tool = load_v21_physical_cmos_readout_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v21_physical_cmos_run(
                root,
                "candidate_sample",
                "sample",
                labels,
                reference,
                reference,
                reference,
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_physical_cmos_readout(
                    sample_runs=[Path("outputs/20260701_v21/runs/candidate_sample")],
                    output=Path("outputs/reports/v21_physical_candidate.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["status"], "physical-cmos-candidate-needs-human-review")
            self.assertTrue(run["physical_cmos_positive"])
            self.assertTrue(run["human_readability_required"])
            self.assertEqual(run["prototype_label_audit"]["correct_ratio"], 1.0)
            self.assertEqual(summary["decision"]["status"], "physical-cmos-candidate-needs-human-review")
            self.assertFalse(summary["decision"]["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v21 physical CMOS diagnostics")
    def test_v21_physical_cmos_readout_rejects_bad_output_and_summary_extracts_runs(self) -> None:
        import torch

        tool = load_v21_physical_cmos_readout_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v21_physical_cmos_run(
                root,
                "summary_eval",
                "eval",
                labels,
                reference,
                reference,
                reference,
                torch,
            )
            summary_path = root / "outputs" / "reports" / "v21_launcher_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps({"probes": [{"eval_run": "outputs/20260701_v21/runs/summary_eval"}]}),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_physical_cmos_readout(
                        eval_runs=[Path("outputs/20260701_v21/runs/summary_eval")],
                        output=Path("bad/report.json"),
                    )
                summary = tool.run_physical_cmos_readout(
                    summary=Path("outputs/reports/v21_launcher_summary.json"),
                    output=Path("outputs/reports/v21_physical_from_summary.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["source_summary"], "outputs/reports/v21_launcher_summary.json")
            self.assertEqual(summary["eval_runs"], ["outputs/20260701_v21/runs/summary_eval"])
            self.assertEqual(summary["runs"][0]["run_type"], "eval")

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
    def test_v23_decoder_chip_semantic_collapse_reports_decoder_plane(self) -> None:
        import torch

        tool = load_v23_decoder_chip_semantic_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            planes = {"reference": reference}
            for key in (
                "decoder_output_amplitude",
                "chip_intensity_primary",
                "probe_plane",
                "physical_camera_plane",
            ):
                planes[key] = collapsed
            self._write_v23_decoder_chip_run(root, "decoder_collapse_eval", "eval", labels, planes, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_chip_semantic_collapse(
                    eval_runs=[Path("outputs/20260701_v23/runs/decoder_collapse_eval")],
                    output=Path("outputs/reports/v23_decoder_chip_collapse.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(summary["decision"]["status"], "decoder-chip-collapse")
            self.assertEqual(run["status"], "decoder-chip-collapse")
            self.assertTrue(run["decoder_chip_collapse"])
            self.assertEqual(run["first_semantic_collapse_plane"]["plane"], "decoder_output_amplitude")
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v23 decoder/chip diagnostics")
    def test_v23_decoder_chip_semantic_rejects_bad_output_and_summary_extracts_runs(self) -> None:
        import torch

        tool = load_v23_decoder_chip_semantic_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v23_decoder_chip_run(
                root,
                "summary_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
            )
            self._write_v23_decoder_chip_run(
                root,
                "summary_sample",
                "sample",
                labels,
                {"reference": reference},
                torch,
            )
            summary_path = root / "outputs" / "reports" / "v23_launcher_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "probes": [{"eval_run": "outputs/20260701_v23/runs/summary_eval"}],
                        "sample_runs": [{"sample_run": "outputs/20260701_v23/runs/summary_sample"}],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_decoder_chip_semantic_collapse(
                        eval_runs=[Path("outputs/20260701_v23/runs/summary_eval")],
                        output=Path("bad/report.json"),
                    )
                summary = tool.run_decoder_chip_semantic_collapse(
                    summary=Path("outputs/reports/v23_launcher_summary.json"),
                    output=Path("outputs/reports/v23_decoder_chip_from_summary.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["source_summary"], "outputs/reports/v23_launcher_summary.json")
            self.assertEqual(summary["eval_runs"], ["outputs/20260701_v23/runs/summary_eval"])
            self.assertEqual(summary["sample_runs"], ["outputs/20260701_v23/runs/summary_sample"])
            self.assertEqual([run["run_type"] for run in summary["runs"]], ["eval", "sample"])

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
    def test_v24_ols_decoder_expressivity_classifies_decoder_bottleneck(self) -> None:
        import torch

        tool = load_v24_ols_decoder_expressivity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            planes = {"reference": reference}
            for key in ("decoder_output_amplitude", "chip_intensity_primary", "probe_plane", "physical_camera_plane"):
                planes[key] = collapsed
            self._write_v24_expressivity_run(root, "decoder_collapse_eval", "eval", labels, planes, torch)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_ols_decoder_expressivity(
                    eval_runs=[Path("outputs/20260701_v24/runs/decoder_collapse_eval")],
                    output=Path("outputs/reports/v24_decoder_bottleneck.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["classification"], "decoder-bottleneck")
            self.assertEqual(run["first_semantic_collapse_plane"]["plane"], "decoder_output_amplitude")
            self.assertEqual(summary["decision"]["status"], "decoder-bottleneck")
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v24 expressivity diagnostics")
    def test_v24_ols_decoder_expressivity_classifies_phase_screen_rescue(self) -> None:
        import torch

        tool = load_v24_ols_decoder_expressivity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v24_expressivity_run(
                root,
                "phase_screen_rescue_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
                ols_calibration="phase_screen",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_ols_decoder_expressivity(
                    eval_runs=[Path("outputs/20260701_v24/runs/phase_screen_rescue_eval")],
                    output=Path("outputs/reports/v24_phase_screen_rescue.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["classification"], "ols-transfer-bottleneck")
            self.assertIsNone(run["first_semantic_collapse_plane"])
            self.assertEqual(summary["decision"]["status"], "ols-transfer-bottleneck")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v24 expressivity diagnostics")
    def test_v24_ols_decoder_expressivity_classifies_amp_phase_upper_bound_and_rejects_bad_output(self) -> None:
        import torch

        tool = load_v24_ols_decoder_expressivity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            self._write_v24_expressivity_run(
                root,
                "amp_phase_rescue_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
                ols_calibration="amplitude_phase_screen",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_ols_decoder_expressivity(
                        eval_runs=[Path("outputs/20260701_v24/runs/amp_phase_rescue_eval")],
                        output=Path("bad/report.json"),
                    )
                summary = tool.run_ols_decoder_expressivity(
                    eval_runs=[Path("outputs/20260701_v24/runs/amp_phase_rescue_eval")],
                    output=Path("outputs/reports/v24_amp_phase_rescue.json"),
                )
            finally:
                os.chdir(cwd)

            run = summary["runs"][0]
            self.assertEqual(run["classification"], "upper-bound-only")
            self.assertEqual(summary["decision"]["status"], "upper-bound-only")

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
    def test_v25_decoder_capacity_gate_classifies_eval_positive_sample_failed(self) -> None:
        import torch

        tool = load_v25_decoder_capacity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            eval_run = self._write_v24_expressivity_run(
                root,
                "probe_a_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
                decoder_modulation_override="phase_amplitude",
            )
            sample_run = self._write_v24_expressivity_run(
                root,
                "probe_a_sample",
                "sample",
                labels,
                {
                    "reference": reference,
                    "physical_camera_plane": collapsed,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "probe_plane": collapsed,
                },
                torch,
                decoder_modulation_override="phase_amplitude",
            )
            self._write_v25_capacity_summary(root, [("probe_a", eval_run, sample_run)])
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_capacity_gate(
                    summary=Path("outputs/reports/v25_summary.json"),
                    output=Path("outputs/reports/v25_capacity.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "eval-positive-sample-failed")
            self.assertEqual(summary["probes"][0]["status"], "eval-positive-sample-failed")
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v25 decoder capacity diagnostics")
    def test_v25_decoder_capacity_gate_classifies_phase_only_limited(self) -> None:
        import torch

        tool = load_v25_decoder_capacity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            amp_eval = self._write_v24_expressivity_run(
                root,
                "amp_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
                decoder_modulation_override="phase_amplitude",
            )
            amp_sample = self._write_v24_expressivity_run(
                root,
                "amp_sample",
                "sample",
                labels,
                {"reference": reference},
                torch,
                decoder_modulation_override="phase_amplitude",
            )
            phase_eval = self._write_v24_expressivity_run(
                root,
                "phase_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "physical_camera_plane": collapsed,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "probe_plane": collapsed,
                },
                torch,
                decoder_layers=4,
            )
            phase_sample = self._write_v24_expressivity_run(
                root,
                "phase_sample",
                "sample",
                labels,
                {
                    "reference": reference,
                    "physical_camera_plane": collapsed,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "probe_plane": collapsed,
                },
                torch,
                decoder_layers=4,
            )
            self._write_v25_capacity_summary(
                root,
                [("probe_a", amp_eval, amp_sample), ("probe_b", phase_eval, phase_sample)],
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_decoder_capacity_gate(
                        summary=Path("outputs/reports/v25_summary.json"),
                        output=Path("bad/v25_capacity.json"),
                    )
                summary = tool.run_decoder_capacity_gate(
                    summary=Path("outputs/reports/v25_summary.json"),
                    output=Path("outputs/reports/v25_capacity.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "phase-only-limited")
            self.assertEqual(summary["probes"][0]["metadata"]["decoder_diagnostic_modulation_override"], "phase_amplitude")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v25 decoder capacity diagnostics")
    def test_v25_decoder_capacity_gate_classifies_depth_limited(self) -> None:
        import torch

        tool = load_v25_decoder_capacity_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v17_reference_images(torch, labels)
            eval_run = self._write_v24_expressivity_run(
                root,
                "depth_eval",
                "eval",
                labels,
                {"reference": reference},
                torch,
                decoder_layers=4,
            )
            sample_run = self._write_v24_expressivity_run(
                root,
                "depth_sample",
                "sample",
                labels,
                {"reference": reference},
                torch,
                decoder_layers=4,
            )
            self._write_v25_capacity_summary(root, [("probe_b", eval_run, sample_run)])
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_capacity_gate(
                    summary=Path("outputs/reports/v25_summary.json"),
                    output=Path("outputs/reports/v25_capacity.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "depth-limited")
            self.assertEqual(summary["probes"][0]["decoder_layers"], 4)

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
    def test_v26_ols_oracle_classifies_separable_ols_as_decodable(self) -> None:
        import torch

        tool = load_v26_ols_decodability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            runs = []
            for index in range(2):
                runs.append(
                    self._write_v24_expressivity_run(
                        root,
                        f"v26_separable_{index}_eval",
                        "eval",
                        labels,
                        {
                            "reference": reference,
                            "encoded_amplitude": reference,
                            "ols_output_amplitude": reference,
                            "decoder_output_amplitude": collapsed,
                            "chip_intensity_primary": collapsed,
                            "probe_plane": collapsed,
                            "physical_camera_plane": collapsed,
                        },
                        torch,
                    )
                )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_ols_decodability_oracle(
                    eval_runs=[Path(run.relative_to(root)) for run in runs],
                    output=Path("outputs/reports/v26_ols_decodable.json"),
                    feature_size=16,
                    pca_dim=8,
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "ols-decodable")
            self.assertGreaterEqual(summary["decision"]["best_oracle_accuracy"], 0.70)
            self.assertFalse(summary["accepted_summary_written"])
            self.assertEqual(summary["oracle_config"]["split_policy"], "leave-run-out")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v26 OLS oracle diagnostics")
    def test_v26_ols_oracle_classifies_collapsed_ols_as_not_decodable(self) -> None:
        import torch

        tool = load_v26_ols_decodability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            runs = []
            for index in range(2):
                runs.append(
                    self._write_v24_expressivity_run(
                        root,
                        f"v26_collapsed_{index}_eval",
                        "eval",
                        labels,
                        {
                            "reference": reference,
                            "encoded_amplitude": collapsed,
                            "ols_output_amplitude": collapsed,
                            "decoder_output_amplitude": collapsed,
                            "chip_intensity_primary": collapsed,
                            "probe_plane": collapsed,
                            "physical_camera_plane": collapsed,
                        },
                        torch,
                    )
                )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_ols_decodability_oracle(
                        eval_runs=[Path(run.relative_to(root)) for run in runs],
                        output=Path("bad/v26_ols.json"),
                    )
                summary = tool.run_ols_decodability_oracle(
                    eval_runs=[Path(run.relative_to(root)) for run in runs],
                    output=Path("outputs/reports/v26_ols_not_decodable.json"),
                    feature_size=16,
                    pca_dim=8,
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "ols-not-decodable")
            self.assertLess(summary["decision"]["best_oracle_accuracy"], 0.50)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v26 OLS oracle diagnostics")
    def test_v26_ols_oracle_rejects_same_run_leakage_and_extracts_summary_runs(self) -> None:
        import torch

        tool = load_v26_ols_decodability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            run = self._write_v24_expressivity_run(
                root,
                "v26_single_eval",
                "eval",
                labels,
                {"reference": reference, "encoded_amplitude": reference, "ols_output_amplitude": reference},
                torch,
            )
            summary_path = root / "outputs" / "reports" / "v26_launcher_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps({"probes": [{"eval_run": run.relative_to(root).as_posix()}]}),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_ols_decodability_oracle(
                    summaries=[Path("outputs/reports/v26_launcher_summary.json")],
                    output=Path("outputs/reports/v26_single_run.json"),
                    feature_size=16,
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["eval_runs"], [run.relative_to(root).as_posix()])
            self.assertEqual(summary["decision"]["status"], "insufficient-oracle-data")
            self.assertFalse(summary["linear_oracles"]["ols_output_amplitude"]["valid"])

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
    def test_v27_decoder_synthesis_gap_reports_decoder_gap(self) -> None:
        import torch

        tool = load_v27_decoder_synthesis_gap_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            run = self._write_v24_expressivity_run(
                root,
                "v27_decoder_gap_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            self._write_v27_v26_report(root)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_synthesis_gap(
                    v26_report=Path("outputs/reports/v26_report.json"),
                    eval_runs=[Path(run.relative_to(root))],
                    output=Path("outputs/reports/v27_decoder_gap.json"),
                )
            finally:
                os.chdir(cwd)

            run_record = summary["runs"][0]
            self.assertEqual(summary["decision"]["status"], "physical-decoder-unreachable")
            self.assertEqual(run_record["first_human_topology_failure_plane"]["plane"], "decoder_output_amplitude")
            self.assertEqual(run_record["classification"], "physical-decoder-unreachable")
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v27 decoder synthesis diagnostics")
    def test_v27_decoder_synthesis_gap_reports_camera_readout_gap(self) -> None:
        import torch

        tool = load_v27_decoder_synthesis_gap_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            run = self._write_v24_expressivity_run(
                root,
                "v27_camera_gap_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            self._write_v27_v26_report(root)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_synthesis_gap(
                    v26_report=Path("outputs/reports/v26_report.json"),
                    eval_runs=[Path(run.relative_to(root))],
                    output=Path("outputs/reports/v27_camera_gap.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "decoder-only-rescue")
            self.assertEqual(summary["runs"][0]["classification"], "decoder-only-rescue")
            self.assertEqual(summary["runs"][0]["first_human_topology_failure_plane"]["plane"], "physical_camera_plane")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v27 decoder synthesis diagnostics")
    def test_v27_decoder_synthesis_gap_marks_readable_cmos_candidate_and_rejects_bad_output(self) -> None:
        import torch

        tool = load_v27_decoder_synthesis_gap_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            run = self._write_v24_expressivity_run(
                root,
                "v27_cmos_candidate_sample",
                "sample",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": reference,
                },
                torch,
            )
            summary_path = root / "outputs" / "reports" / "v27_launcher_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps({"sample_runs": [{"sample_run": run.relative_to(root).as_posix()}]}),
                encoding="utf-8",
            )
            self._write_v27_v26_report(root)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_decoder_synthesis_gap(
                        v26_report=Path("outputs/reports/v26_report.json"),
                        summary=Path("outputs/reports/v27_launcher_summary.json"),
                        output=Path("bad/v27.json"),
                    )
                summary = tool.run_decoder_synthesis_gap(
                    v26_report=Path("outputs/reports/v26_report.json"),
                    summary=Path("outputs/reports/v27_launcher_summary.json"),
                    output=Path("outputs/reports/v27_cmos_candidate.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "human-accepted-candidate-needs-v4-review")
            self.assertEqual(summary["sample_runs"], [run.relative_to(root).as_posix()])
            self.assertTrue(summary["human_readability_required"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")
    def test_v28_decoder_controllability_reports_decoder_unreachable(self) -> None:
        import torch

        tool = load_v28_decoder_controllability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            run = self._write_v24_expressivity_run(
                root,
                "v28_decoder_unreachable_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_controllability(
                    eval_runs=[Path(run.relative_to(root))],
                    output=Path("outputs/reports/v28_decoder_unreachable.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "decoder-topology-unreachable")
            self.assertEqual(summary["runs"][0]["first_human_topology_failure_plane"]["plane"], "decoder_output_amplitude")
            self.assertFalse(summary["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")
    def test_v28_decoder_controllability_reports_camera_after_decoder_failure(self) -> None:
        import torch

        tool = load_v28_decoder_controllability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            run = self._write_v24_expressivity_run(
                root,
                "v28_camera_gap_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                summary = tool.run_decoder_controllability(
                    eval_runs=[Path(run.relative_to(root))],
                    output=Path("outputs/reports/v28_camera_gap.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "camera-after-decoder-failure")
            self.assertEqual(summary["runs"][0]["classification"], "camera-after-decoder-failure")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28 decoder diagnostics")
    def test_v28_decoder_controllability_marks_metric_proxy_failure_and_rejects_bad_output(self) -> None:
        import torch

        tool = load_v28_decoder_controllability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            run = self._write_v24_expressivity_run(
                root,
                "v28_proxy_failure_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            oracle = root / "outputs" / "reports" / "v26_oracle.json"
            oracle.parent.mkdir(parents=True, exist_ok=True)
            oracle.write_text(
                json.dumps(
                    {
                        "decision": {
                            "status": "ols-decodable",
                            "best_oracle_feature": "encoded_amplitude",
                            "best_oracle_accuracy": 1.0,
                        },
                        "accepted_summary_written": False,
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_decoder_controllability(
                        oracle_report=Path("outputs/reports/v26_oracle.json"),
                        eval_runs=[Path(run.relative_to(root))],
                        output=Path("bad/v28.json"),
                    )
                summary = tool.run_decoder_controllability(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[Path(run.relative_to(root))],
                    output=Path("outputs/reports/v28_proxy_failure.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary["decision"]["status"], "metric-proxy-failure")
            self.assertEqual(summary["runs"][0]["classification"], "metric-proxy-failure")
            self.assertTrue(summary["runs"][0]["oracle_decodable"])

    def test_v28_decoder_controllability_configs_preserve_contract(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            (
                "train_twophoton_100x100_v28_probe_a_decoder_fixed_upstream_tiny_overfit_decoder_plane_e8.yaml",
                "eval_twophoton_100x100_v28_probe_a_decoder_fixed_upstream_tiny_overfit_decoder_plane.yaml",
                "decoder_output_amplitude",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_b_decoder_fixed_upstream_tiny_overfit_chip_plane_e8.yaml",
                "eval_twophoton_100x100_v28_probe_b_decoder_fixed_upstream_tiny_overfit_chip_plane.yaml",
                "chip_intensity_primary",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_c_decoder_fixed_upstream_tiny_overfit_cmos_plane_e8.yaml",
                "eval_twophoton_100x100_v28_probe_c_decoder_fixed_upstream_tiny_overfit_cmos_plane.yaml",
                "physical_camera_plane",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_d_decoder_phase_amplitude_tiny_upper_bound_e8.yaml",
                "eval_twophoton_100x100_v28_probe_d_decoder_phase_amplitude_tiny_upper_bound.yaml",
                "chip_intensity_primary",
                "phase_amplitude",
            ),
        ]
        for train_name, eval_name, target_plane, override in cases:
            with self.subTest(train_config=train_name):
                train_cfg = load_training_config(config_dir / train_name)
                eval_cfg = load_training_config(config_dir / eval_name)
                for cfg in (train_cfg, eval_cfg):
                    self.assertIsNotNone(cfg.one_shot.d2nn)
                    self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                    self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                    self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                    self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")
                    self.assertEqual(cfg.one_shot.generator_e_arch, "label_template")
                    self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                    self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, override)
                self.assertTrue(train_cfg.one_shot.decoder_only_training)
                self.assertEqual(train_cfg.one_shot.aux_topology_plane_key, target_plane)
                self.assertGreater(train_cfg.one_shot.aux_topology_weight, 0.0)
                self.assertFalse(eval_cfg.one_shot.decoder_only_training)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28.1 replay payload tests")
    def test_v28_1_tiny_replay_payload_is_deterministic_and_balanced(self) -> None:
        import torch

        tool = load_v28_1_tiny_replay_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = torch.tensor([label for label in range(10) for _ in range(6)], dtype=torch.long)
            source = {
                "codes": torch.arange(labels.numel() * 4, dtype=torch.float32).view(labels.numel(), 4),
                "class_labels": labels,
                "teacher_camera": torch.arange(labels.numel() * 16, dtype=torch.float32).view(labels.numel(), 1, 4, 4),
                "readability_filter_indices": torch.arange(1000, 1000 + labels.numel(), dtype=torch.long),
            }
            source_path = root / "source.pt"
            torch.save(source, source_path)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                first = tool.build_tiny_replay_payload(
                    source=source_path,
                    output=Path("outputs/20260702_v28_1/artifacts/replay_a.pt"),
                    per_class=2,
                    num_classes=10,
                )
                second = tool.build_tiny_replay_payload(
                    source=source_path,
                    output=Path("outputs/reports/replay_b.pt"),
                    per_class=2,
                    num_classes=10,
                )
                with self.assertRaisesRegex(ValueError, "outputs/reports"):
                    tool.build_tiny_replay_payload(
                        source=source_path,
                        output=Path("bad/replay.pt"),
                        per_class=2,
                        num_classes=10,
                    )
            finally:
                os.chdir(cwd)

            first_payload = torch.load(root / first["output"], map_location="cpu", weights_only=False)
            second_payload = torch.load(root / second["output"], map_location="cpu", weights_only=False)
            self.assertEqual(first["metadata"]["selected_source_positions"], second["metadata"]["selected_source_positions"])
            self.assertEqual(first["metadata"]["selected_labels"], [label for label in range(10) for _ in range(2)])
            self.assertEqual(first["metadata"]["selected_source_indices"][:2], [1000, 1001])
            self.assertFalse(first["metadata"]["acceptance_eligible"])
            self.assertTrue((root / first["sidecar"]).is_file())
            self.assertEqual(first_payload["class_labels"].tolist(), second_payload["class_labels"].tolist())
            self.assertEqual(first_payload["teacher_camera"].shape[0], 20)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v28.1 same-batch diagnostics")
    def test_v28_1_same_batch_gate_distinguishes_main_statuses(self) -> None:
        import torch

        tool = load_v28_1_same_batch_overfit_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = list(range(10))
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            same_reachable = self._write_v24_expressivity_run(
                root,
                "probe_a_same_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": reference,
                },
                torch,
            )
            test_collapsed = self._write_v24_expressivity_run(
                root,
                "probe_a_test_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            same_camera_failure = self._write_v24_expressivity_run(
                root,
                "probe_b_same_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            same_probe_d = self._write_v24_expressivity_run(
                root,
                "probe_d_phase_amplitude_same_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": reference,
                },
                torch,
                decoder_modulation_override="phase_amplitude",
            )
            same_unreachable = self._write_v24_expressivity_run(
                root,
                "probe_c_same_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
            )
            oracle = root / "outputs" / "reports" / "v26_oracle.json"
            oracle.parent.mkdir(parents=True, exist_ok=True)
            oracle.write_text(
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
            summary_path = root / "outputs" / "reports" / "v28_1_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "probes": [
                            {
                                "same_batch_eval_run": same_reachable.relative_to(root).as_posix(),
                                "test_eval_run": test_collapsed.relative_to(root).as_posix(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_same_batch_overfit_gate(
                        oracle_report=Path("outputs/reports/v26_oracle.json"),
                        same_batch_eval_runs=[same_reachable.relative_to(root)],
                        output=Path("bad/v28_1.json"),
                    )
                summary_result = tool.run_same_batch_overfit_gate(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    summary=Path("outputs/reports/v28_1_summary.json"),
                    output=Path("outputs/reports/v28_1_generalization.json"),
                )
                camera_result = tool.run_same_batch_overfit_gate(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    same_batch_eval_runs=[same_camera_failure.relative_to(root)],
                    output=Path("outputs/reports/v28_1_camera.json"),
                )
                phase_result = tool.run_same_batch_overfit_gate(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    same_batch_eval_runs=[same_probe_d.relative_to(root)],
                    output=Path("outputs/reports/v28_1_phase.json"),
                )
                unreachable_result = tool.run_same_batch_overfit_gate(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    same_batch_eval_runs=[same_unreachable.relative_to(root)],
                    output=Path("outputs/reports/v28_1_unreachable.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(summary_result["decision"]["status"], "same-batch-generalization-failure")
            self.assertEqual(camera_result["decision"]["status"], "same-batch-camera-failure")
            self.assertEqual(phase_result["decision"]["status"], "phase-only-limited")
            self.assertEqual(unreachable_result["decision"]["status"], "decoder-topology-unreachable")
            self.assertFalse(summary_result["accepted_summary_written"])

    def test_v28_1_configs_preserve_contract_and_same_eval_fallback(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            (
                "train_twophoton_100x100_v28_probe_a_decoder_fixed_upstream_tiny_overfit_decoder_plane_e8.yaml",
                "eval_same_twophoton_100x100_v28_1_probe_a_decoder_fixed_upstream_tiny_overfit_decoder_plane.yaml",
                "decoder_output_amplitude",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_b_decoder_fixed_upstream_tiny_overfit_chip_plane_e8.yaml",
                "eval_same_twophoton_100x100_v28_1_probe_b_decoder_fixed_upstream_tiny_overfit_chip_plane.yaml",
                "chip_intensity_primary",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_c_decoder_fixed_upstream_tiny_overfit_cmos_plane_e8.yaml",
                "eval_same_twophoton_100x100_v28_1_probe_c_decoder_fixed_upstream_tiny_overfit_cmos_plane.yaml",
                "physical_camera_plane",
                "none",
            ),
            (
                "train_twophoton_100x100_v28_probe_d_decoder_phase_amplitude_tiny_upper_bound_e8.yaml",
                "eval_same_twophoton_100x100_v28_1_probe_d_decoder_phase_amplitude_tiny_upper_bound.yaml",
                "chip_intensity_primary",
                "phase_amplitude",
            ),
        ]
        for train_name, same_eval_name, target_plane, override in cases:
            with self.subTest(train_config=train_name):
                train_cfg = load_training_config(config_dir / train_name)
                same_eval_cfg = load_training_config(config_dir / same_eval_name)
                for cfg in (train_cfg, same_eval_cfg):
                    self.assertIsNotNone(cfg.one_shot.d2nn)
                    self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                    self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                    self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                    self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")
                    self.assertEqual(cfg.one_shot.generator_e_arch, "label_template")
                    self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                    self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, override)
                self.assertTrue(train_cfg.one_shot.decoder_only_training)
                self.assertEqual(train_cfg.one_shot.aux_topology_plane_key, target_plane)
                self.assertGreater(train_cfg.one_shot.aux_topology_weight, 0.0)
                self.assertFalse(same_eval_cfg.one_shot.decoder_only_training)
                self.assertTrue(same_eval_cfg.one_shot.eval_allow_train_fallback)
                self.assertEqual(same_eval_cfg.one_shot.aux_topology_weight, 0.0)

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
    def test_v29_decoder_trainability_classifies_main_outcomes(self) -> None:
        import torch

        tool = load_v29_decoder_trainability_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            oracle = root / "outputs" / "reports" / "v26_oracle.json"
            oracle.parent.mkdir(parents=True, exist_ok=True)
            oracle.write_text(
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

            fail_train = self._write_v29_train_run(root, "v29_harness_fail_train", audit_passed=False)
            lowpass_train = self._write_v29_train_run(root, "v29_lowpass_train", target_transform="lowpass")
            depth_train = self._write_v29_train_run(root, "v29_depth_train", target_transform="thick_binary", decoder_layers=4)
            phase_train = self._write_v29_train_run(
                root,
                "v29_phase_amp_train",
                target_transform="thick_binary",
                decoder_layers=4,
                decoder_modulation_override="phase_amplitude",
            )
            unreachable_train = self._write_v29_train_run(root, "v29_unreachable_train")
            readable = {"reference": reference}
            collapsed_planes = {
                "reference": reference,
                "decoder_output_amplitude": collapsed,
                "chip_intensity_primary": collapsed,
                "physical_camera_plane": collapsed,
            }
            fail_eval = self._write_v24_expressivity_run(root, "v29_fail_same_eval", "eval", labels, readable, torch)
            lowpass_eval = self._write_v24_expressivity_run(
                root,
                "v29_decoder_lowpass_target_same_batch_eval",
                "eval",
                labels,
                readable,
                torch,
            )
            depth_eval = self._write_v24_expressivity_run(
                root,
                "v29_decoder_4layer_phase_only_thick_binary_same_eval",
                "eval",
                labels,
                readable,
                torch,
                decoder_layers=4,
            )
            phase_eval = self._write_v24_expressivity_run(
                root,
                "v29_decoder_4layer_phase_amplitude_thick_binary_same_eval",
                "eval",
                labels,
                readable,
                torch,
                decoder_modulation_override="phase_amplitude",
                decoder_layers=4,
            )
            unreachable_eval = self._write_v24_expressivity_run(
                root,
                "v29_unreachable_same_eval",
                "eval",
                labels,
                collapsed_planes,
                torch,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v29_decoder_trainability_target_encoding(
                        oracle_report=Path("outputs/reports/v26_oracle.json"),
                        train_runs=[fail_train.relative_to(root)],
                        same_batch_eval_runs=[fail_eval.relative_to(root)],
                        output=Path("bad/v29.json"),
                    )
                harness = tool.run_v29_decoder_trainability_target_encoding(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    train_runs=[fail_train.relative_to(root)],
                    same_batch_eval_runs=[fail_eval.relative_to(root)],
                    output=Path("outputs/reports/v29_harness.json"),
                )
                target = tool.run_v29_decoder_trainability_target_encoding(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    train_runs=[lowpass_train.relative_to(root)],
                    same_batch_eval_runs=[lowpass_eval.relative_to(root)],
                    output=Path("outputs/reports/v29_target.json"),
                )
                depth = tool.run_v29_decoder_trainability_target_encoding(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    train_runs=[depth_train.relative_to(root)],
                    same_batch_eval_runs=[depth_eval.relative_to(root)],
                    output=Path("outputs/reports/v29_depth.json"),
                )
                phase = tool.run_v29_decoder_trainability_target_encoding(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    train_runs=[phase_train.relative_to(root)],
                    same_batch_eval_runs=[phase_eval.relative_to(root)],
                    output=Path("outputs/reports/v29_phase.json"),
                )
                unreachable = tool.run_v29_decoder_trainability_target_encoding(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    train_runs=[unreachable_train.relative_to(root)],
                    same_batch_eval_runs=[unreachable_eval.relative_to(root)],
                    output=Path("outputs/reports/v29_unreachable.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(harness["decision"]["status"], "harness-failure")
            self.assertEqual(target["decision"]["status"], "target-encoding-rescue")
            self.assertEqual(depth["decision"]["status"], "depth-limited")
            self.assertEqual(phase["decision"]["status"], "phase-amplitude-limited")
            self.assertIn(unreachable["decision"]["status"], {"architecture-target-unreachable", "metric-proxy-failure"})
            self.assertFalse(target["accepted_summary_written"])

    def test_v29_configs_preserve_contract_and_mark_upper_bounds(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            ("train_twophoton_100x100_v29_probe_a_decoder_trainability_audit_same_target_e2.yaml", 2, "none", "none", False),
            ("train_twophoton_100x100_v29_probe_b_decoder_lowpass_target_same_batch_e8.yaml", 2, "lowpass", "none", False),
            ("train_twophoton_100x100_v29_probe_c_decoder_thick_binary_target_same_batch_e8.yaml", 2, "thick_binary", "none", False),
            ("train_twophoton_100x100_v29_probe_d_decoder_4layer_phase_only_thick_binary_upper_bound_e8.yaml", 4, "thick_binary", "none", True),
            ("train_twophoton_100x100_v29_probe_e_decoder_4layer_phase_amplitude_thick_binary_upper_bound_e8.yaml", 4, "thick_binary", "phase_amplitude", True),
        ]
        for name, decoder_layers, target_transform, override, upper_bound in cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertIsNotNone(cfg.one_shot.d2nn)
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, decoder_layers)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template")
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.aux_topology_target_transform, target_transform)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, override)
                self.assertEqual(decoder_layers > 2 or override != "none", upper_bound)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v30 target-code diagnostics")
    def test_v30_target_code_alignment_classifies_main_outcomes(self) -> None:
        import torch

        tool = load_v30_target_code_alignment_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            oracle = root / "outputs" / "reports" / "v26_oracle.json"
            oracle.parent.mkdir(parents=True, exist_ok=True)
            oracle.write_text(
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
            readable = {
                "reference": reference,
                "encoded_amplitude": reference,
                "ols_output_amplitude": reference,
                "target_code_amplitude": reference,
                "decoder_output_amplitude": reference,
                "chip_intensity_primary": reference,
                "physical_camera_plane": reference,
            }
            camera_gap = {
                **readable,
                "physical_camera_plane": collapsed,
            }
            collapsed_planes = {
                "reference": reference,
                "encoded_amplitude": reference,
                "ols_output_amplitude": reference,
                "target_code_amplitude": reference,
                "decoder_output_amplitude": collapsed,
                "chip_intensity_primary": collapsed,
                "physical_camera_plane": collapsed,
            }

            spatial = self._write_v24_expressivity_run(
                root,
                "v30_probe_a_spatial_identity_code_decoder_same_batch_eval",
                "eval",
                labels,
                readable,
                torch,
                target_code_mode="spatial_identity_code",
            )
            lowfreq = self._write_v24_expressivity_run(
                root,
                "v30_probe_b_lowfreq_digit_code_decoder_same_batch_eval",
                "eval",
                labels,
                readable,
                torch,
                target_code_mode="lowfreq_digit_code",
                target_code_bridge_source="ols_output_amplitude",
            )
            codebook = self._write_v24_expressivity_run(
                root,
                "v30_probe_c_class_codebook_grid_decoder_same_batch_eval",
                "eval",
                labels,
                readable,
                torch,
                target_code_mode="class_codebook_grid",
            )
            camera = self._write_v24_expressivity_run(
                root,
                "v30_probe_a_spatial_identity_code_camera_gap_same_eval",
                "eval",
                labels,
                camera_gap,
                torch,
                target_code_mode="spatial_identity_code",
            )
            oracle_only = self._write_v24_expressivity_run(
                root,
                "v30_probe_d_camera_direct_target_code_oracle_eval",
                "eval",
                labels,
                readable,
                torch,
                target_code_mode="spatial_identity_code",
                optical_core_arch="camera_direct_clean_oracle",
            )
            unreachable = self._write_v24_expressivity_run(
                root,
                "v30_unreachable_same_batch_eval",
                "eval",
                labels,
                collapsed_planes,
                torch,
                target_code_mode="spatial_identity_code",
            )

            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v30_target_code_alignment(
                        oracle_report=Path("outputs/reports/v26_oracle.json"),
                        eval_runs=[spatial.relative_to(root)],
                        output=Path("bad/v30.json"),
                    )
                spatial_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[spatial.relative_to(root)],
                    output=Path("outputs/reports/v30_spatial.json"),
                )
                lowfreq_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[lowfreq.relative_to(root)],
                    output=Path("outputs/reports/v30_lowfreq.json"),
                )
                codebook_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[codebook.relative_to(root)],
                    output=Path("outputs/reports/v30_codebook.json"),
                )
                camera_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[camera.relative_to(root)],
                    output=Path("outputs/reports/v30_camera.json"),
                )
                oracle_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[oracle_only.relative_to(root)],
                    output=Path("outputs/reports/v30_oracle_only.json"),
                )
                unreachable_summary = tool.run_v30_target_code_alignment(
                    oracle_report=Path("outputs/reports/v26_oracle.json"),
                    eval_runs=[unreachable.relative_to(root)],
                    output=Path("outputs/reports/v30_unreachable.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(spatial_summary["decision"]["status"], "target-code-rescue")
            self.assertEqual(lowfreq_summary["decision"]["status"], "frequency-encoding-bottleneck")
            self.assertEqual(
                lowfreq_summary["runs"][0]["target_code_bridge_source"],
                "ols_output_amplitude",
            )
            self.assertEqual(codebook_summary["decision"]["status"], "semantic-codebook-rescue")
            self.assertEqual(camera_summary["decision"]["status"], "camera-after-code-rescue")
            self.assertEqual(oracle_summary["decision"]["status"], "architecture-unreachable")
            self.assertIn(
                unreachable_summary["decision"]["status"],
                {"architecture-unreachable", "metric-proxy-failure"},
            )
            self.assertFalse(spatial_summary["accepted_summary_written"])

    def test_v30_1_sample_gate_recovery_separates_metric_positive_from_human_candidate(self) -> None:
        tool = load_v30_1_sample_gate_recovery_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = root / "outputs" / "reports"
            reports.mkdir(parents=True)
            summary_path = reports / "v30_summary.json"
            alignment_path = reports / "v30_alignment.json"
            summary_payload = {
                "accepted_summary_written": False,
                "probes": [
                    {
                        "label": "probe_a_spatial_identity_code_decoder",
                        "same_batch_eval_run": "outputs/v30/runs/a_same",
                        "test_eval_run": "outputs/v30/runs/a_test",
                        "sample_run": None,
                        "same_batch_strict_gate_passed": True,
                        "test_strict_gate_passed": True,
                        "target_code_diagnostic_mode": "spatial_identity_code",
                        "optical_core_arch": "diffractive_phase_only",
                    },
                    {
                        "label": "probe_b_lowfreq_digit_code_decoder",
                        "same_batch_eval_run": "outputs/v30/runs/b_same",
                        "test_eval_run": "outputs/v30/runs/b_test",
                        "sample_run": "outputs/v30/runs/b_sample",
                        "same_batch_strict_gate_passed": True,
                        "test_strict_gate_passed": True,
                        "target_code_diagnostic_mode": "lowfreq_digit_code",
                        "optical_core_arch": "diffractive_phase_only",
                    },
                    {
                        "label": "probe_c_class_codebook_grid_decoder",
                        "same_batch_eval_run": "outputs/v30/runs/c_same",
                        "test_eval_run": "outputs/v30/runs/c_test",
                        "sample_run": None,
                        "same_batch_strict_gate_passed": True,
                        "test_strict_gate_passed": True,
                        "target_code_diagnostic_mode": "class_codebook_grid",
                        "optical_core_arch": "diffractive_phase_only",
                    },
                    {
                        "label": "probe_d_camera_direct_target_code_oracle",
                        "same_batch_eval_run": "outputs/v30/runs/d_same",
                        "test_eval_run": "outputs/v30/runs/d_test",
                        "sample_run": None,
                        "same_batch_strict_gate_passed": True,
                        "test_strict_gate_passed": True,
                        "target_code_diagnostic_mode": "spatial_identity_code",
                        "optical_core_arch": "camera_direct_clean_oracle",
                    },
                ],
            }
            alignment_payload = {
                "runs": [
                    {
                        "run": "outputs/v30/runs/a_same",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/a_test",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/b_same",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/b_test",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/b_sample",
                        "run_type": "sample",
                        "classification": "human-accepted-candidate-needs-v4-review",
                    },
                    {
                        "run": "outputs/v30/runs/c_same",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/c_test",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/d_same",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                    {
                        "run": "outputs/v30/runs/d_test",
                        "run_type": "eval",
                        "classification": "decoder-topology-reachable",
                    },
                ]
            }
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
            alignment_path.write_text(json.dumps(alignment_payload), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v30_1_sample_gate_recovery(
                        summary=Path("outputs/reports/v30_summary.json"),
                        alignment_report=Path("outputs/reports/v30_alignment.json"),
                        output=Path("bad/v30_1.json"),
                    )
                result = tool.run_v30_1_sample_gate_recovery(
                    summary=Path("outputs/reports/v30_summary.json"),
                    alignment_report=Path("outputs/reports/v30_alignment.json"),
                    output=Path("outputs/reports/v30_1.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["status"], "v30-sample-metric-candidate-needs-human-review")
            self.assertIn("outputs/v30/runs/b_sample", result["human_readability_candidate_runs"])
            self.assertIn("outputs/v30/runs/a_test", result["eval_positive_sample_failed_runs"])
            self.assertEqual(
                [entry["label"] for entry in result["eligible_probes"]],
                ["probe_a_spatial_identity_code_decoder", "probe_b_lowfreq_digit_code_decoder"],
            )
            self.assertEqual(
                [entry["label"] for entry in result["excluded_probes"]],
                ["probe_c_class_codebook_grid_decoder", "probe_d_camera_direct_target_code_oracle"],
            )
            self.assertFalse(result["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v31 bridge diagnostics")
    def test_v31_bridge_topology_classifies_bridge_decoder_and_camera_failures(self) -> None:
        import torch

        tool = load_v31_bridge_topology_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 3, 4, 7, 9]
            reference = self._v26_oracle_reference(torch, labels)
            collapsed = torch.ones_like(reference) * 0.5
            bridge_collapse = self._write_v24_expressivity_run(
                root,
                "v31_1_bridge_collapse_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "target_code_amplitude": collapsed,
                    "decoder_input_amplitude": collapsed,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
                target_code_mode="lowfreq_digit_code",
                target_code_bridge_source="ols_output_amplitude",
            )
            decoder_gap = self._write_v24_expressivity_run(
                root,
                "v31_1_decoder_gap_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "target_code_amplitude": reference,
                    "decoder_input_amplitude": reference,
                    "decoder_output_amplitude": collapsed,
                    "chip_intensity_primary": collapsed,
                    "physical_camera_plane": collapsed,
                },
                torch,
                target_code_mode="lowfreq_digit_code",
                target_code_bridge_source="ols_output_amplitude",
            )
            camera_gap = self._write_v24_expressivity_run(
                root,
                "v31_1_camera_gap_eval",
                "eval",
                labels,
                {
                    "reference": reference,
                    "target_code_amplitude": reference,
                    "decoder_input_amplitude": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": collapsed,
                },
                torch,
                target_code_mode="spatial_identity_code",
                target_code_bridge_source="ols_output_amplitude",
            )
            candidate = self._write_v24_expressivity_run(
                root,
                "v31_1_bridge_candidate_sample",
                "sample",
                labels,
                {
                    "reference": reference,
                    "target_code_amplitude": reference,
                    "decoder_input_amplitude": reference,
                    "decoder_output_amplitude": reference,
                    "chip_intensity_primary": reference,
                    "physical_camera_plane": reference,
                },
                torch,
                target_code_mode="lowfreq_digit_code",
                target_code_bridge_source="encoded_amplitude",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v31_bridge_topology(
                        eval_runs=[bridge_collapse.relative_to(root)],
                        output=Path("bad/v31_1.json"),
                    )
                collapse_summary = tool.run_v31_bridge_topology(
                    eval_runs=[bridge_collapse.relative_to(root)],
                    output=Path("outputs/reports/v31_1_bridge_collapse.json"),
                )
                decoder_summary = tool.run_v31_bridge_topology(
                    eval_runs=[decoder_gap.relative_to(root)],
                    output=Path("outputs/reports/v31_1_decoder_gap.json"),
                )
                camera_summary = tool.run_v31_bridge_topology(
                    eval_runs=[camera_gap.relative_to(root)],
                    output=Path("outputs/reports/v31_1_camera_gap.json"),
                )
                candidate_summary = tool.run_v31_bridge_topology(
                    sample_runs=[candidate.relative_to(root)],
                    output=Path("outputs/reports/v31_1_candidate.json"),
                )
                v32_summary = tool.run_v31_bridge_topology(
                    eval_runs=[decoder_gap.relative_to(root)],
                    output=Path("outputs/reports/v32_decoder_gap.json"),
                    version="v32",
                    objective="bridge_target_code_redesign_same_batch_topology_gate",
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(collapse_summary["decision"]["status"], "bridge-collapse")
            self.assertEqual(decoder_summary["decision"]["status"], "decoder-after-bridge-failure")
            self.assertEqual(camera_summary["decision"]["status"], "camera-after-bridge-failure")
            self.assertEqual(candidate_summary["decision"]["status"], "bridge-code-candidate-needs-human-review")
            self.assertEqual(
                decoder_summary["runs"][0]["first_human_topology_failure_plane"]["plane"],
                "decoder_output_amplitude",
            )
            self.assertEqual(v32_summary["version"], "v32")
            self.assertEqual(v32_summary["objective"], "bridge_target_code_redesign_same_batch_topology_gate")
            self.assertFalse(candidate_summary["accepted_summary_written"])

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
    def test_v32_1_sample_gap_isolation_detects_label_template_reference_mismatch(self) -> None:
        import torch

        tool = load_v32_1_sample_gap_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            eval_reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            eval_reference[:2, 0, 1:7, 2:4] = 1.0
            eval_reference[2:, 0, 2:4, 1:7] = 1.0
            sample_reference = torch.zeros_like(eval_reference)
            sample_reference[:2, 0, 1:7, 5:7] = 1.0
            sample_reference[2:, 0, 5:7, 1:7] = 1.0
            generated_template = eval_reference.clone()
            eval_run = self._write_v32_1_gap_run(
                root,
                "probe_a_template_test_eval",
                "eval",
                labels,
                eval_reference,
                torch,
                planes={"physical_camera_plane": generated_template},
            )
            sample_run = self._write_v32_1_gap_run(
                root,
                "probe_a_template_sample",
                "sample",
                labels,
                sample_reference,
                torch,
                planes={"physical_camera_plane": generated_template},
            )
            summary_path, bridge_path = self._write_v32_1_gap_inputs(root, eval_run, sample_run)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v32_1_sample_gap_isolation(
                    summary=summary_path.relative_to(root),
                    bridge_report=bridge_path.relative_to(root),
                    output=Path("outputs/reports/v32_1_template_gap.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "label-template-reference-mismatch")
            self.assertTrue(result["runs"][0]["checks"]["generated_matches_eval"])
            self.assertTrue(result["runs"][0]["checks"]["generated_fails_sample_reference"])
            self.assertFalse(result["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v32.1 sample-gap diagnostics")
    def test_v32_1_sample_gap_isolation_detects_code_insensitive_generator_gap(self) -> None:
        import torch

        tool = load_v32_1_sample_gap_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[:2, 0, 1:7, 2:4] = 1.0
            reference[2:, 0, 2:4, 1:7] = 1.0
            codes = torch.tensor(
                [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                dtype=torch.float32,
            )
            eval_run = self._write_v32_1_gap_run(
                root,
                "probe_a_code_flat_test_eval",
                "eval",
                labels,
                reference,
                torch,
                planes={"physical_camera_plane": reference},
                codes=codes,
            )
            sample_run = self._write_v32_1_gap_run(
                root,
                "probe_a_code_flat_sample",
                "sample",
                labels,
                reference,
                torch,
                planes={"physical_camera_plane": reference},
                codes=codes,
            )
            summary_path, bridge_path = self._write_v32_1_gap_inputs(root, eval_run, sample_run)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v32_1_sample_gap_isolation(
                    summary=summary_path.relative_to(root),
                    bridge_report=bridge_path.relative_to(root),
                    output=Path("outputs/reports/v32_1_code_gap.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "code-insensitive-generator-gap")
            self.assertTrue(result["runs"][0]["checks"]["code_insensitive"])
            self.assertFalse(result["runs"][0]["checks"]["eval_sample_reference_mismatch"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v32.1 sample-gap diagnostics")
    def test_v32_1_sample_gap_isolation_detects_sample_target_code_collapse(self) -> None:
        import torch

        tool = load_v32_1_sample_gap_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[:2, 0, 1:7, 2:4] = 1.0
            reference[2:, 0, 2:4, 1:7] = 1.0
            physical = reference.clone()
            physical[1, 0, 0, 0] = 0.2
            physical[3, 0, 7, 7] = 0.2
            collapsed_target = torch.ones_like(reference) * 0.5
            eval_run = self._write_v32_1_gap_run(
                root,
                "probe_a_target_ok_test_eval",
                "eval",
                labels,
                reference,
                torch,
                planes={"physical_camera_plane": reference, "target_code_amplitude": reference},
            )
            sample_run = self._write_v32_1_gap_run(
                root,
                "probe_a_target_collapsed_sample",
                "sample",
                labels,
                reference,
                torch,
                planes={"physical_camera_plane": physical, "target_code_amplitude": collapsed_target},
            )
            summary_path, bridge_path = self._write_v32_1_gap_inputs(root, eval_run, sample_run)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v32_1_sample_gap_isolation(
                        summary=summary_path.relative_to(root),
                        bridge_report=bridge_path.relative_to(root),
                        output=Path("bad/v32_1.json"),
                    )
                result = tool.run_v32_1_sample_gap_isolation(
                    summary=summary_path.relative_to(root),
                    bridge_report=bridge_path.relative_to(root),
                    output=Path("outputs/reports/v32_1_target_collapse.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "sample-plane-collapse")
            self.assertTrue(result["runs"][0]["checks"]["sample_target_code_collapse"])

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
    def test_v33_code_conditioned_diagnostic_classifies_residual_inactive_and_rescue(self) -> None:
        import torch

        tool = load_v33_code_conditioned_sample_gap_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[:2, 0, 1:7, 2:4] = 1.0
            reference[2:, 0, 2:4, 1:7] = 1.0
            residual = torch.zeros_like(reference)
            residual[0, 0, 0, 0] = 0.3
            residual[1, 0, 0, 1] = 0.6
            residual[2, 0, 7, 7] = 0.3
            residual[3, 0, 7, 6] = 0.6
            inactive = self._write_v33_gap_run(
                root,
                "v33_residual_inactive_eval",
                "eval",
                labels,
                reference,
                torch,
                residual=torch.zeros_like(reference),
            )
            missing_residual = self._write_v33_gap_run(
                root,
                "v33_missing_residual_eval",
                "eval",
                labels,
                reference,
                torch,
                include_residual=False,
            )
            rescue = self._write_v33_gap_run(
                root,
                "v33_rescue_eval",
                "eval",
                labels,
                reference,
                torch,
                residual=residual,
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "under outputs/reports"):
                    tool.run_v33_code_conditioned_sample_gap(
                        eval_runs=[inactive.relative_to(root)],
                        output=Path("bad/v33.json"),
                    )
                inactive_result = tool.run_v33_code_conditioned_sample_gap(
                    eval_runs=[inactive.relative_to(root)],
                    output=Path("outputs/reports/v33_inactive.json"),
                )
                missing_result = tool.run_v33_code_conditioned_sample_gap(
                    eval_runs=[missing_residual.relative_to(root)],
                    output=Path("outputs/reports/v33_missing_residual.json"),
                )
                rescue_result = tool.run_v33_code_conditioned_sample_gap(
                    eval_runs=[rescue.relative_to(root)],
                    output=Path("outputs/reports/v33_rescue.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(inactive_result["decision"]["status"], "residual-inactive")
            self.assertEqual(missing_result["decision"]["status"], "harness-invalid")
            self.assertEqual(missing_result["runs"][0]["reason"], "missing generator_residual_map")
            self.assertEqual(rescue_result["decision"]["status"], "code-conditioned-target-rescue")
            self.assertTrue(rescue_result["runs"][0]["checks"]["residual_active"])
            self.assertFalse(rescue_result["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33 code diagnostics")
    def test_v33_code_conditioned_diagnostic_marks_unpaired_sample_reference_invalid(self) -> None:
        import torch

        tool = load_v33_code_conditioned_sample_gap_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[:2, 0, 1:7, 2:4] = 1.0
            reference[2:, 0, 2:4, 1:7] = 1.0
            generated = torch.zeros_like(reference)
            generated[:2, 0, 1:7, 5:7] = 1.0
            generated[2:, 0, 5:7, 1:7] = 1.0
            residual = torch.zeros_like(reference)
            residual[0, 0, 0, 0] = 0.3
            residual[1, 0, 0, 1] = 0.6
            residual[2, 0, 7, 7] = 0.3
            residual[3, 0, 7, 6] = 0.6
            sample = self._write_v33_gap_run(
                root,
                "v33_unpaired_sample",
                "sample",
                labels,
                reference,
                torch,
                generated=generated,
                residual=residual,
                reference_source="validation_only",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v33_code_conditioned_sample_gap(
                    sample_runs=[sample.relative_to(root)],
                    output=Path("outputs/reports/v33_unpaired_sample.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "sample-reference-metric-invalid")
            self.assertTrue(result["runs"][0]["checks"]["sample_reference_metric_invalid"])

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
    def test_v33_2_real_ols_candidate_preview_requires_manual_review_for_unpaired_candidate(self) -> None:
        import torch

        tool = load_v33_2_real_ols_candidate_preview_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[0, 0, 1:6, 2:4] = 0.8
            reference[1, 0, 2:7, 2:5] = 1.0
            reference[2, 0, 2:4, 1:6] = 0.8
            reference[3, 0, 3:5, 2:7] = 1.0
            sample = self._write_v33_2_sample_run(root, "v33_2_candidate_sample", labels, reference, torch)
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "status": "code-conditioned-target-rescue",
                        "probes": [
                            {"label": "probe_c_clean_teacher_lowfreq_code_residual_s005", "sample_run": sample.as_posix()},
                            {"label": "probe_d_mnist_sample_ols_bridge_code_residual_s005", "checkpoint": "d.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v33_2_real_ols_candidate_preview(
                    summary=summary.relative_to(root),
                    candidate_sample_run=sample.relative_to(root),
                    control_sample_run=sample.relative_to(root),
                    output=Path("outputs/reports/v33_2_candidate.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "manual-preview-required")
            self.assertTrue(result["candidate"]["checks"]["sample_reference_metric_invalid"])
            self.assertTrue(result["candidate"]["preview_package"]["v4_style_preview_exists"])
            self.assertFalse(result["accepted_summary_written"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33.2 preview diagnostics")
    def test_v33_2_real_ols_candidate_preview_detects_sample_code_distribution_gap(self) -> None:
        import torch

        tool = load_v33_2_real_ols_candidate_preview_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.rand((4, 1, 8, 8), dtype=torch.float32)
            collapsed = torch.zeros_like(reference)
            sample = self._write_v33_2_sample_run(
                root,
                "v33_2_target_collapse_sample",
                labels,
                reference,
                torch,
                target=collapsed,
            )
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "code-conditioned-target-rescue", "probes": []}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v33_2_real_ols_candidate_preview(
                    summary=summary.relative_to(root),
                    candidate_sample_run=sample.relative_to(root),
                    output=Path("outputs/reports/v33_2_target_collapse.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "sample-code-distribution-gap")
            self.assertFalse(result["candidate"]["checks"]["bridge_sample_diversity_ok"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v33.2 preview diagnostics")
    def test_v33_2_real_ols_candidate_preview_detects_camera_after_code_failure(self) -> None:
        import torch

        tool = load_v33_2_real_ols_candidate_preview_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            reference = torch.zeros((4, 1, 8, 8), dtype=torch.float32)
            reference[0, 0, 1:6, 2:4] = 0.8
            reference[1, 0, 2:7, 2:5] = 1.0
            reference[2, 0, 2:4, 1:6] = 0.8
            reference[3, 0, 3:5, 2:7] = 1.0
            physical = torch.zeros_like(reference)
            sample = self._write_v33_2_sample_run(
                root,
                "v33_2_camera_failure_sample",
                labels,
                reference,
                torch,
                physical=physical,
            )
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "code-conditioned-target-rescue", "probes": []}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v33_2_real_ols_candidate_preview(
                    summary=summary.relative_to(root),
                    candidate_sample_run=sample.relative_to(root),
                    output=Path("outputs/reports/v33_2_camera_failure.json"),
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["decision"]["status"], "camera-after-code-failure")
            self.assertTrue(result["candidate"]["checks"]["decoder_sample_diversity_ok"])
            self.assertFalse(result["candidate"]["checks"]["physical_cmos_sample_diversity_ok"])

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
    def test_v34_code_coupling_recovery_classifies_first_collapse_plane(self) -> None:
        import torch

        tool = load_v34_code_coupling_recovery_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            cases = [
                ("residual_to_command", 0, "residual-to-command-bottleneck"),
                ("command_to_ols", 1, "command-to-ols-bottleneck"),
                ("ols_to_target", 3, "ols-to-target-code-bottleneck"),
                ("decoder_after_code", 4, "decoder-after-code-bottleneck"),
                ("camera_after_code", 5, "camera-after-code-bottleneck"),
                ("recovered", 6, "code-coupling-recovered-needs-human-review"),
            ]
            runs = [
                self._write_v34_coupling_sample_run(root, f"v34_{name}", labels, torch, active_until=active_until)
                for name, active_until, _ in cases
            ]
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "code-conditioned-target-rescue"}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                for (name, _, expected), run in zip(cases, runs, strict=True):
                    with self.subTest(case=name):
                        result = tool.run_v34_code_coupling_recovery(
                            summary=summary.relative_to(root),
                            sample_runs=[run.relative_to(root)],
                            scales=[0.1],
                            labels=[name],
                            output=Path(f"outputs/reports/{name}.json"),
                        )
                        self.assertEqual(result["decision"]["status"], expected)
                        self.assertFalse(result["accepted_summary_written"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v35 code-carrier diagnostics")
    def test_v35_optical_code_carrier_isolation_classifies_oracle_and_scale_paths(self) -> None:
        import torch

        tool = load_v35_optical_code_carrier_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            scale_weak = self._write_v34_coupling_sample_run(
                root,
                "v35_scale_weak",
                labels,
                torch,
                active_until=0,
            )
            oracle_recovered = self._write_v34_coupling_sample_run(
                root,
                "v35_oracle_recovered",
                labels,
                torch,
                active_until=6,
            )
            oracle_front_fail = self._write_v34_coupling_sample_run(
                root,
                "v35_oracle_front_fail",
                labels,
                torch,
                active_until=1,
            )
            oracle_bridge_fail = self._write_v34_coupling_sample_run(
                root,
                "v35_oracle_bridge_fail",
                labels,
                torch,
                active_until=3,
            )
            scale_recovered = self._write_v34_coupling_sample_run(
                root,
                "v35_scale_recovered",
                labels,
                torch,
                active_until=6,
            )
            for run in (oracle_recovered, oracle_front_fail, oracle_bridge_fail):
                bundle_path = run / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                bundle["diagnostic_slm_override_pattern"] = bundle["slm_command"].clone()
                torch.save(bundle, bundle_path)
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "code-conditioned-target-rescue"}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                generator_limited = tool.run_v35_optical_code_carrier_isolation(
                    summary=summary.relative_to(root),
                    scale_runs=[scale_weak.relative_to(root)],
                    scales=[1.0],
                    scale_labels=["scale_weak"],
                    oracle_runs=[oracle_recovered.relative_to(root)],
                    oracle_amplitudes=[0.05],
                    oracle_labels=["oracle_recovered"],
                    output=Path("outputs/reports/v35_generator_limited.json"),
                )
                self.assertEqual(generator_limited["decision"]["status"], "generator-command-too-weak")

                front_fail = tool.run_v35_optical_code_carrier_isolation(
                    summary=summary.relative_to(root),
                    scale_runs=[scale_weak.relative_to(root)],
                    scales=[1.0],
                    oracle_runs=[oracle_front_fail.relative_to(root)],
                    oracle_amplitudes=[0.05],
                    output=Path("outputs/reports/v35_front_fail.json"),
                )
                self.assertEqual(front_fail["decision"]["status"], "front-ols-code-carrier-failure")

                bridge_fail = tool.run_v35_optical_code_carrier_isolation(
                    summary=summary.relative_to(root),
                    scale_runs=[scale_weak.relative_to(root)],
                    scales=[1.0],
                    oracle_runs=[oracle_bridge_fail.relative_to(root)],
                    oracle_amplitudes=[0.05],
                    output=Path("outputs/reports/v35_bridge_fail.json"),
                )
                self.assertEqual(bridge_fail["decision"]["status"], "ols-to-bridge-carrier-failure")

                recovered = tool.run_v35_optical_code_carrier_isolation(
                    summary=summary.relative_to(root),
                    scale_runs=[scale_recovered.relative_to(root)],
                    scales=[4.0],
                    oracle_runs=[oracle_front_fail.relative_to(root)],
                    oracle_amplitudes=[0.05],
                    output=Path("outputs/reports/v35_recovered.json"),
                )
                self.assertEqual(recovered["decision"]["status"], "code-coupling-recovered-needs-human-review")
                self.assertFalse(recovered["accepted_summary_written"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v36 high-gain diagnostics")
    def test_v36_high_gain_code_coupled_training_classifies_training_outcomes(self) -> None:
        import torch

        tool = load_v36_high_gain_code_coupled_training_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = [0, 0, 1, 1]
            recovered = self._write_v34_coupling_sample_run(
                root,
                "v36_recovered",
                labels,
                torch,
                active_until=6,
            )
            coupling_regression = self._write_v34_coupling_sample_run(
                root,
                "v36_coupling_regression",
                labels,
                torch,
                active_until=1,
            )
            saturation = self._write_v34_coupling_sample_run(
                root,
                "v36_saturation",
                labels,
                torch,
                active_until=6,
            )
            camera_failure = self._write_v34_coupling_sample_run(
                root,
                "v36_camera_failure",
                labels,
                torch,
                active_until=5,
            )
            decoder_failure = self._write_v34_coupling_sample_run(
                root,
                "v36_decoder_failure",
                labels,
                torch,
                active_until=4,
            )
            saturation_bundle_path = saturation / "artifacts" / "one_shot_sample_tensors.pt"
            saturation_bundle = torch.load(saturation_bundle_path, map_location="cpu", weights_only=True)
            saturation_bundle["slm_command"] = torch.ones_like(saturation_bundle["slm_command"])
            torch.save(saturation_bundle, saturation_bundle_path)
            summary = root / "outputs" / "reports" / "v33_1_summary.json"
            v35 = root / "outputs" / "reports" / "v35_report.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "code-conditioned-target-rescue"}), encoding="utf-8")
            v35.write_text(json.dumps({"status": "code-coupling-recovered-needs-human-review"}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                cases = [
                    (recovered, "high-gain-training-rescue-needs-human-review"),
                    (coupling_regression, "code-coupling-regression"),
                    (saturation, "scale-saturation-regression"),
                    (camera_failure, "camera-after-decoder-failure"),
                    (decoder_failure, "decoder-topology-failure"),
                ]
                for run, expected in cases:
                    with self.subTest(expected=expected):
                        result = tool.run_v36_high_gain_code_coupled_training(
                            source_v35_report=v35.relative_to(root),
                            source_v33_1_summary=summary.relative_to(root),
                            probe_entries=[
                                f"{run.name}|2.0|lowfreq_digit_code|{run.relative_to(root)}|||{run.relative_to(root)}"
                            ],
                            output=Path(f"outputs/reports/{expected}.json"),
                        )
                        self.assertEqual(result["decision"]["status"], expected)
                        self.assertFalse(result["accepted_summary_written"])
            finally:
                os.chdir(cwd)

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
    def test_v37_final_cmos_decoder_topology_classifies_plane_failures(self) -> None:
        import torch

        tool = load_v37_final_cmos_decoder_topology_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("target_blob", False, True, False, "target-code-topology-too-blob-like"),
                ("decoder_loss", True, False, False, "decoder-physical-topology-loss"),
                ("camera_loss", True, True, False, "camera-after-decoder-loss"),
                ("manual_candidate", True, True, True, "manual-cmos-review-candidate"),
            ]
            runs = []
            for name, target_ok, decoder_ok, camera_ok, _ in cases:
                runs.append(
                    self._write_v37_topology_sample_run(
                        root,
                        f"v37_{name}",
                        torch,
                        target_ok=target_ok,
                        decoder_ok=decoder_ok,
                        camera_ok=camera_ok,
                    )
                )
            bad_run = self._write_v37_topology_sample_run(
                root,
                "v37_missing_preview",
                torch,
                target_ok=True,
                decoder_ok=True,
                camera_ok=True,
                with_preview=False,
            )
            v36_report = root / "outputs" / "reports" / "v36_report.json"
            v36_report.parent.mkdir(parents=True)
            v36_report.write_text(
                json.dumps(
                    {
                        "status": "high-gain-training-rescue-needs-human-review",
                        "probes": [
                            {"label": name, "sample_run": run.relative_to(root).as_posix()}
                            for (name, *_), run in zip(cases, runs, strict=True)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                for (name, _, _, _, expected), run in zip(cases, runs, strict=True):
                    with self.subTest(case=name):
                        result = tool.run_v37_final_cmos_decoder_topology(
                            v36_report=v36_report.relative_to(root),
                            sample_runs=[run.relative_to(root)],
                            labels=[name],
                            output=Path(f"outputs/reports/{name}.json"),
                            review_dir=Path(f"outputs/reports/{name}_review"),
                            version="v37.1",
                            objective="final_cmos_manual_review_and_decoder_topology_isolation_harness_fix",
                        )
                        self.assertEqual(result["decision"]["status"], expected)
                        self.assertEqual(result["version"], "v37.1")
                        self.assertFalse(result["accepted_summary_written"])
                        self.assertTrue(result["runs"][0]["manual_review_package"]["manual_review_image"])
                        self.assertEqual(result["runs"][0]["sample_reference_metric_status"], "sample-reference-metric-invalid")

                invalid = tool.run_v37_final_cmos_decoder_topology(
                    v36_report=v36_report.relative_to(root),
                    sample_runs=[bad_run.relative_to(root)],
                    labels=["bad"],
                    output=Path("outputs/reports/bad.json"),
                )
                self.assertEqual(invalid["decision"]["status"], "harness-invalid")
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v37 topology diagnostics")
    def test_v37_final_cmos_decoder_topology_allows_distinct_cmos_resolution(self) -> None:
        import torch

        tool = load_v37_final_cmos_decoder_topology_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self._write_v37_topology_sample_run(
                root,
                "v37_1_distinct_cmos_resolution",
                torch,
                target_ok=True,
                decoder_ok=True,
                camera_ok=False,
                middle_shape=(100, 100),
                physical_shape=(200, 200),
            )
            bad_run = self._write_v37_topology_sample_run(
                root,
                "v37_1_batch_mismatch",
                torch,
                target_ok=True,
                decoder_ok=True,
                camera_ok=True,
                middle_shape=(100, 100),
                physical_shape=(200, 200),
                physical_batch=3,
            )
            v36_report = root / "outputs" / "reports" / "v36_report.json"
            v36_report.parent.mkdir(parents=True)
            v36_report.write_text(
                json.dumps(
                    {
                        "status": "high-gain-training-rescue-needs-human-review",
                        "probes": [{"label": "probe", "sample_run": run.relative_to(root).as_posix()}],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v37_final_cmos_decoder_topology(
                    v36_report=v36_report.relative_to(root),
                    sample_runs=[run.relative_to(root)],
                    labels=["distinct"],
                    output=Path("outputs/reports/distinct.json"),
                    review_dir=Path("outputs/reports/distinct_review"),
                    version="v37.1",
                )
                self.assertEqual(result["decision"]["status"], "camera-after-decoder-loss")
                self.assertEqual(
                    result["runs"][0]["plane_topology"]["target_code_amplitude"]["shape"],
                    [4, 1, 100, 100],
                )
                self.assertEqual(
                    result["runs"][0]["plane_topology"]["physical_camera_plane"]["shape"],
                    [4, 1, 200, 200],
                )
                self.assertFalse(result["errors"])

                invalid = tool.run_v37_final_cmos_decoder_topology(
                    v36_report=v36_report.relative_to(root),
                    sample_runs=[bad_run.relative_to(root)],
                    labels=["batch_mismatch"],
                    output=Path("outputs/reports/batch_mismatch.json"),
                )
                self.assertEqual(invalid["decision"]["status"], "harness-invalid")
                self.assertIn("batch size is inconsistent", invalid["errors"][0]["error"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v38 CMOS review")
    def test_v38_formal_cmos_manual_review_builds_cmos_only_package_and_template(self) -> None:
        import torch

        tool = load_v38_formal_cmos_manual_review_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                self._write_v37_topology_sample_run(
                    root,
                    f"v38_probe_{idx}",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                )
                for idx in range(2)
            ]
            v37_report = root / "outputs" / "reports" / "v37_1_report.json"
            v37_report.parent.mkdir(parents=True)
            v37_report.write_text(
                json.dumps(
                    {
                        "version": "v37.1",
                        "status": "manual-cmos-review-candidate",
                        "diagnostic_only": True,
                        "accepted_summary_written": False,
                        "runs": [
                            {
                                "label": f"probe_{idx}",
                                "run": run.relative_to(root).as_posix(),
                                "status": "manual-cmos-review-candidate",
                            }
                            for idx, run in enumerate(runs)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v38_formal_cmos_manual_review(
                    v37_report=v37_report.relative_to(root),
                    output=Path("outputs/reports/v38_report.json"),
                    template_output=Path("outputs/reports/v38_template.json"),
                    review_dir=Path("outputs/reports/v38_review"),
                )
                self.assertEqual(result["status"], "manual-review-required")
                self.assertTrue(result["diagnostic_only"])
                self.assertFalse(result["accepted_summary_written"])
                self.assertEqual(result["final_human_gate_plane"], "physical_camera_plane")
                self.assertEqual(result["review_policy"], "cmos-only")
                self.assertFalse(result["errors"])
                self.assertEqual(len(result["runs"]), 2)
                for run in result["runs"]:
                    self.assertEqual(run["review_planes"], ["physical_camera_plane"])
                    self.assertTrue(Path(run["cmos_only_review_image"]).is_file())
                    self.assertEqual(run["physical_camera_plane_shape"], [4, 1, 200, 200])
                    self.assertNotIn("target_code_amplitude", json.dumps(run))
                self.assertTrue(Path(result["comparison_review_image"]).is_file())

                template = json.loads(Path("outputs/reports/v38_template.json").read_text(encoding="utf-8"))
                self.assertEqual(template["status"], "manual-review-required")
                self.assertFalse(template["accepted_summary_written"])
                self.assertEqual(len(template["manual_review_items"]), 8)
                first = template["manual_review_items"][0]
                self.assertEqual(first["final_human_gate_plane"], "physical_camera_plane")
                self.assertIsNone(first["visible_digit_topology"])
                self.assertIsNone(first["suggested_digit"])
                self.assertIsNone(first["confidence"])
                self.assertEqual(first["notes"], "")
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v38 CMOS review")
    def test_v38_formal_cmos_manual_review_marks_missing_cmos_plane_harness_invalid(self) -> None:
        import torch

        tool = load_v38_formal_cmos_manual_review_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self._write_v37_topology_sample_run(
                root,
                "v38_missing_cmos",
                torch,
                target_ok=True,
                decoder_ok=True,
                camera_ok=True,
            )
            bundle_path = run / "artifacts" / "one_shot_sample_tensors.pt"
            bundle = torch.load(bundle_path, map_location="cpu")
            del bundle["physical_camera_plane"]
            torch.save(bundle, bundle_path)
            v37_report = root / "outputs" / "reports" / "v37_1_report.json"
            v37_report.parent.mkdir(parents=True)
            v37_report.write_text(
                json.dumps(
                    {
                        "version": "v37.1",
                        "status": "manual-cmos-review-candidate",
                        "runs": [
                            {
                                "label": "missing_cmos",
                                "run": run.relative_to(root).as_posix(),
                                "status": "manual-cmos-review-candidate",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v38_formal_cmos_manual_review(
                    v37_report=v37_report.relative_to(root),
                    output=Path("outputs/reports/v38_missing.json"),
                    template_output=Path("outputs/reports/v38_missing_template.json"),
                    review_dir=Path("outputs/reports/v38_missing_review"),
                )
                self.assertEqual(result["status"], "harness-invalid")
                self.assertTrue(result["errors"])
                self.assertIn("physical_camera_plane", result["errors"][0]["error"])
                self.assertFalse(result["accepted_summary_written"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v39 topology diagnostics")
    def test_v39_human_topology_false_positive_classifies_blob_origin(self) -> None:
        import torch

        tool = load_v39_human_topology_false_positive_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("target_blob", False, True, True, "target-code-blob-origin"),
                ("decoder_blob", True, False, True, "decoder-blob-origin"),
                ("camera_blob", True, True, False, "camera-texture-amplification"),
            ]
            runs = [
                self._write_v37_topology_sample_run(
                    root,
                    f"v39_{name}",
                    torch,
                    target_ok=target_ok,
                    decoder_ok=decoder_ok,
                    camera_ok=camera_ok,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                )
                for name, target_ok, decoder_ok, camera_ok, _ in cases
            ]
            v38_report = root / "outputs" / "reports" / "v38_report.json"
            v37_report = root / "outputs" / "reports" / "v37_1_report.json"
            v38_report.parent.mkdir(parents=True)
            report_payload = {
                "version": "v38",
                "status": "manual-review-required",
                "diagnostic_only": True,
                "accepted_summary_written": False,
                "runs": [
                    {
                        "label": name,
                        "run": run.relative_to(root).as_posix(),
                        "status": "manual-review-required",
                    }
                    for (name, *_), run in zip(cases, runs, strict=True)
                ],
            }
            v38_report.write_text(json.dumps(report_payload), encoding="utf-8")
            v37_report.write_text(
                json.dumps(
                    {
                        "version": "v37.1",
                        "status": "manual-cmos-review-candidate",
                        "diagnostic_only": True,
                        "accepted_summary_written": False,
                        "runs": [
                            {
                                "label": name,
                                "run": run.relative_to(root).as_posix(),
                                "status": "manual-cmos-review-candidate",
                            }
                            for (name, *_), run in zip(cases, runs, strict=True)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                for (name, *_rest, expected), run in zip(cases, runs, strict=True):
                    with self.subTest(case=name):
                        result = tool.run_v39_human_topology_false_positive(
                            v38_report=v38_report.relative_to(root),
                            v37_report=v37_report.relative_to(root),
                            sample_runs=[run.relative_to(root)],
                            labels=[name],
                            manual_verdict="final_cmos_blob_failed",
                            output=Path(f"outputs/reports/{name}.json"),
                        )
                        self.assertEqual(result["status"], "manual-review-failed-final-cmos-blob")
                        self.assertEqual(result["localized_blob_origin"], expected)
                        self.assertTrue(result["proxy_false_positive"])
                        self.assertIn("topology-proxy-false-positive", result["classifications"])
                        self.assertFalse(result["accepted_summary_written"])
                        self.assertFalse(result["human_review_passed"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v39 topology diagnostics")
    def test_v39_human_topology_false_positive_detects_label_template_collapse_and_invalid_verdict(self) -> None:
        import torch

        tool = load_v39_human_topology_false_positive_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                self._write_v37_topology_sample_run(
                    root,
                    f"v39_collapse_{idx}",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                )
                for idx in range(3)
            ]
            v38_report = root / "outputs" / "reports" / "v38_report.json"
            v37_report = root / "outputs" / "reports" / "v37_1_report.json"
            v38_report.parent.mkdir(parents=True)
            runs_payload = [
                {"label": f"probe_{idx}", "run": run.relative_to(root).as_posix(), "status": "manual-review-required"}
                for idx, run in enumerate(runs)
            ]
            v38_report.write_text(
                json.dumps(
                    {
                        "version": "v38",
                        "status": "manual-review-required",
                        "diagnostic_only": True,
                        "accepted_summary_written": False,
                        "runs": runs_payload,
                    }
                ),
                encoding="utf-8",
            )
            v37_report.write_text(
                json.dumps(
                    {
                        "version": "v37.1",
                        "status": "manual-cmos-review-candidate",
                        "diagnostic_only": True,
                        "accepted_summary_written": False,
                        "runs": runs_payload,
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = tool.run_v39_human_topology_false_positive(
                    v38_report=v38_report.relative_to(root),
                    v37_report=v37_report.relative_to(root),
                    manual_verdict="final_cmos_blob_failed",
                    output=Path("outputs/reports/collapse.json"),
                )
                self.assertEqual(result["status"], "manual-review-failed-final-cmos-blob")
                self.assertIn("label-template-collapse", result["classifications"])
                self.assertTrue(result["diagnostics"]["label_template_collapse"]["collapsed"])
                self.assertEqual(result["next_gate"], "v40_sample_code_diversity_target_representation")
                self.assertFalse(result["accepted_summary_written"])

                invalid = tool.run_v39_human_topology_false_positive(
                    v38_report=v38_report.relative_to(root),
                    v37_report=v37_report.relative_to(root),
                    manual_verdict="",
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v40 diversity diagnostics")
    def test_v40_sample_code_diversity_target_representation_classifies_probe_states(self) -> None:
        import torch

        tool = load_v40_sample_code_diversity_target_representation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("recovered", True, True, True, "label-template-collapse-recovered"),
                ("still_collapsed", False, False, False, "code-diversity-still-collapsed"),
                ("target_blob", False, True, True, "target-representation-still-blob"),
                ("decoder_loss", True, False, True, "bridge-after-diversity-failure"),
                ("camera_loss", True, True, False, "camera-after-diversity-failure"),
            ]
            runs = []
            for name, target_ok, decoder_ok, camera_ok, _expected in cases:
                run = self._write_v37_topology_sample_run(
                    root,
                    f"v40_{name}",
                    torch,
                    target_ok=target_ok,
                    decoder_ok=decoder_ok,
                    camera_ok=camera_ok,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                )
                runs.append(run)
            def active_constant_plane(shape):
                values = torch.tensor([0.2, 0.8, 0.35, 0.95], dtype=torch.float32).view(4, 1, 1, 1)
                return values.expand(4, 1, shape[0], shape[1]).clone()

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            for name, run in zip([case[0] for case in cases], runs, strict=True):
                bundle_path = run / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                if name == "recovered":
                    middle = active_stroke_plane((100, 100))
                    bundle["target_code_amplitude"] = middle
                    bundle["decoder_input_amplitude"] = middle.clone()
                    bundle["decoder_output_amplitude"] = middle.clone()
                    bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                elif name == "target_blob":
                    plane = active_constant_plane((100, 100))
                    bundle["target_code_amplitude"] = plane
                    bundle["decoder_input_amplitude"] = plane.clone()
                elif name == "decoder_loss":
                    bundle["decoder_output_amplitude"] = active_constant_plane((100, 100))
                elif name == "camera_loss":
                    bundle["physical_camera_plane"] = active_constant_plane((200, 200))
                torch.save(bundle, bundle_path)
            v39_report = root / "outputs" / "reports" / "v39.json"
            v33_1_summary = root / "outputs" / "reports" / "v33_1.json"
            v39_report.parent.mkdir(parents=True, exist_ok=True)
            v39_report.write_text(
                json.dumps(
                    {
                        "version": "v39",
                        "status": "manual-review-failed-final-cmos-blob",
                        "next_gate": "v40_sample_code_diversity_target_representation",
                        "diagnostic_only": True,
                        "accepted_summary_written": False,
                    }
                ),
                encoding="utf-8",
            )
            v33_1_summary.write_text(
                json.dumps({"version": "v33.1", "diagnostic_only": True, "accepted_summary_written": False}),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                for (name, _target_ok, _decoder_ok, _camera_ok, expected), run in zip(cases, runs, strict=True):
                    with self.subTest(case=name):
                        result = tool.run_v40_sample_code_diversity_target_representation(
                            source_v39_report=v39_report.relative_to(root),
                            source_v33_1_summary=v33_1_summary.relative_to(root),
                            probe_entries=[
                                f"{name}|sharp_stroke_digit_code|0.5|{run.relative_to(root)}|{run.relative_to(root)}"
                            ],
                            output=Path(f"outputs/reports/{name}.json"),
                        )
                        self.assertEqual(result["status"], expected)
                        self.assertFalse(result["accepted_summary_written"])
                        self.assertTrue(result["human_readability_required"])

                invalid = tool.run_v40_sample_code_diversity_target_representation(
                    source_v39_report=v39_report.relative_to(root),
                    source_v33_1_summary=v33_1_summary.relative_to(root),
                    probe_entries=["missing|sharp_stroke_digit_code|0.0||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

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
    def test_v48_transparent_decoder_control_validates_inputs_and_maps_manual_verdicts(self) -> None:
        import torch

        tool = load_v48_transparent_decoder_control_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)

            def class_planes(size):
                planes = torch.zeros((20, 1, size, size), dtype=torch.float32)
                for index, class_id in enumerate(labels.tolist()):
                    offset = 2 + class_id * max(1, (size - 4) // 10)
                    offset = min(size - 2, offset)
                    planes[index, 0, :, offset - 1 : offset + 1] = 1.0
                    planes[index, 0, offset - 1 : offset + 1, :] = 0.5
                return planes

            target = class_planes(100)

            def write_run(
                name,
                *,
                kind,
                spacing_mm,
                target_tensor=target,
                run_labels=labels,
                artifact_d2nn=None,
                config_d2nn=None,
            ):
                run = root / "outputs" / "20260710_v48" / "runs" / name
                artifacts = run / "artifacts"
                configs = run / "configs"
                artifacts.mkdir(parents=True)
                configs.mkdir(parents=True)
                sample_count = int(run_labels.numel())
                middle = target_tensor[:sample_count].clone()
                physical = torch.nn.functional.interpolate(middle, size=(200, 200), mode="bilinear")
                full_d2nn = {
                    "grid_size": 100,
                    "pixel_size": 2.0e-6,
                    "propagation_grid_size": 1024,
                    "decoder_layers": 2,
                    "layer_spacing": spacing_mm * 1.0e-3,
                }
                metadata = {
                    "stage": "one_shot_sample",
                    "one_shot_checkpoint": "outputs/20260710_v47/shared_probe_a.pt",
                    "decoder_diagnostic_mode": (
                        "propagation_only_decoder" if kind == "transparent" else "normal"
                    ),
                    "decoder_diagnostic_modulation_override": "phase_amplitude",
                    "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                    "target_code_spatial_scale": 0.72,
                    "target_code_bridge_source": "none",
                    "front_path_model": "relay_intensity_psf",
                    "back_path_model": "relay_psf",
                    "readout_primary_plane": "camera",
                    "d2nn": dict(full_d2nn if artifact_d2nn is None else artifact_d2nn),
                }
                torch.save(
                    {
                        "class_labels": run_labels,
                        "artifact_metadata": metadata,
                        "target_code_amplitude": middle,
                        "decoder_input_amplitude": middle.clone(),
                        "decoder_output_amplitude": middle.clone(),
                        "physical_camera_plane": physical,
                    },
                    artifacts / "one_shot_sample_tensors.pt",
                )
                (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
                train_config = {
                    "one_shot": {
                        "decoder_diagnostic_mode": metadata["decoder_diagnostic_mode"],
                        "decoder_diagnostic_modulation_override": "phase_amplitude",
                        "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                        "target_code_spatial_scale": 0.72,
                        "target_code_bridge_source": "none",
                        "d2nn": dict(full_d2nn if config_d2nn is None else config_d2nn),
                    },
                    "physics": {
                        "front_path_model": "relay_intensity_psf",
                        "back_path_model": "relay_psf",
                        "readout": {"primary_plane": "camera"},
                    },
                }
                (configs / "train_config.json").write_text(json.dumps(train_config), encoding="utf-8")
                return run

            t010 = write_run("t010", kind="transparent", spacing_mm=0.10)
            t025 = write_run("t025", kind="transparent", spacing_mm=0.25)
            m025 = write_run("m025", kind="masked", spacing_mm=0.25)
            entries = [
                f"T010|transparent|0.10|{t010.relative_to(root)}",
                f"T025|transparent|0.25|{t025.relative_to(root)}",
                f"M025|masked|0.25|{m025.relative_to(root)}",
            ]
            cwd = Path.cwd()
            os.chdir(root)
            try:
                pending = tool.run_v48_transparent_decoder_control(
                    probe_entries=entries,
                    output=Path("outputs/reports/pending.json"),
                )
                self.assertEqual(pending["status"], "manual-review-required")
                self.assertEqual(pending["review_class_labels"], list(range(10)))
                self.assertEqual(pending["review_planes"], ["physical_camera_plane"])
                self.assertTrue(Path(pending["comparison_review_image"]).is_file())
                self.assertFalse(pending["accepted_summary_written"])
                self.assertEqual(
                    {record["label"] for record in pending["probes"]},
                    {"T010", "T025", "M025"},
                )
                self.assertTrue(all(record["target_relative"]["physical_camera_plane"]["accuracy"] == 1.0 for record in pending["probes"]))

                artifact_d2nn_without_pg = {
                    "grid_size": 100,
                    "pixel_size": 2.0e-6,
                    "decoder_layers": 2,
                    "layer_spacing": 0.25e-3,
                }
                fallback_t010 = write_run(
                    "fallback_t010",
                    kind="transparent",
                    spacing_mm=0.10,
                    artifact_d2nn={**artifact_d2nn_without_pg, "layer_spacing": 0.10e-3},
                )
                fallback_t025 = write_run(
                    "fallback_t025",
                    kind="transparent",
                    spacing_mm=0.25,
                    artifact_d2nn=artifact_d2nn_without_pg,
                )
                fallback_m025 = write_run(
                    "fallback_m025",
                    kind="masked",
                    spacing_mm=0.25,
                    artifact_d2nn=artifact_d2nn_without_pg,
                )
                fallback = tool.run_v48_transparent_decoder_control(
                    probe_entries=[
                        f"T010|transparent|0.10|{fallback_t010.relative_to(root)}",
                        f"T025|transparent|0.25|{fallback_t025.relative_to(root)}",
                        f"M025|masked|0.25|{fallback_m025.relative_to(root)}",
                    ],
                    output=Path("outputs/reports/config_fallback.json"),
                )
                self.assertEqual(fallback["status"], "manual-review-required")

                verdicts = {
                    "t025_readable_m025_unreadable": "decoder-mask-induced-artifact",
                    "t010_readable_t025_unreadable": "spacing-limited-below-formal-contract",
                    "transparent_unreadable": "transparent-relay-insufficient",
                    "m025_readable": "direct-decoder-human-candidate",
                }
                for verdict, expected in verdicts.items():
                    result = tool.run_v48_transparent_decoder_control(
                        probe_entries=entries,
                        manual_verdict=verdict,
                        output=Path(f"outputs/reports/{verdict}.json"),
                    )
                    self.assertEqual(result["status"], expected)
                    self.assertFalse(result["accepted_summary_written"])

                mismatched = target.clone()
                mismatched[0] = torch.roll(mismatched[0], shifts=3, dims=-1)
                mismatch_run = write_run(
                    "mismatch",
                    kind="transparent",
                    spacing_mm=0.25,
                    target_tensor=mismatched,
                )
                mismatch_entries = [entries[0], f"T025|transparent|0.25|{mismatch_run.relative_to(root)}", entries[2]]
                invalid = tool.run_v48_transparent_decoder_control(
                    probe_entries=mismatch_entries,
                    output=Path("outputs/reports/mismatch.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v49 diagnostics")
    def test_v49_decoder_mask_component_attribution_validates_sources_and_maps_verdicts(self) -> None:
        import torch

        tool = load_v49_decoder_mask_component_attribution_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = Path("outputs/20260710_v47/shared_probe_a.pt")
            labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)

            def class_planes(size):
                planes = torch.zeros((20, 1, size, size), dtype=torch.float32)
                for index, class_id in enumerate(labels.tolist()):
                    offset = min(size - 2, 2 + class_id * max(1, (size - 4) // 10))
                    planes[index, 0, :, offset - 1 : offset + 1] = 1.0
                    planes[index, 0, offset - 1 : offset + 1, :] = 0.5
                return planes

            target = class_planes(100)

            def write_run(name, component, *, run_labels=labels, physical_shape=(200, 200), target_tensor=target):
                run = root / "outputs" / "20260711_v49" / "runs" / name
                artifacts = run / "artifacts"
                configs = run / "configs"
                artifacts.mkdir(parents=True)
                configs.mkdir(parents=True)
                sample_count = int(run_labels.numel())
                middle = target_tensor[:sample_count].clone()
                physical = torch.nn.functional.interpolate(middle, size=physical_shape, mode="bilinear")
                mode = "propagation_only_decoder" if component == "transparent" else "normal"
                metadata = {
                    "stage": "one_shot_sample",
                    "one_shot_checkpoint": checkpoint.as_posix(),
                    "decoder_diagnostic_mode": mode,
                    "decoder_diagnostic_modulation_override": "phase_amplitude",
                    "decoder_diagnostic_mask_component": (
                        "trained" if component in {"transparent", "trained"} else component
                    ),
                    "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                    "target_code_spatial_scale": 0.72,
                    "target_code_bridge_source": "none",
                    "front_path_model": "relay_intensity_psf",
                    "back_path_model": "relay_psf",
                    "readout_primary_plane": "camera",
                    "d2nn": {
                        "grid_size": 100,
                        "pixel_size": 2.0e-6,
                        "propagation_grid_size": 1024,
                        "decoder_layers": 2,
                        "layer_spacing": 0.25e-3,
                    },
                }
                torch.save(
                    {
                        "class_labels": run_labels,
                        "artifact_metadata": metadata,
                        "target_code_amplitude": middle,
                        "decoder_input_amplitude": middle.clone(),
                        "decoder_output_amplitude": middle.clone(),
                        "physical_camera_plane": physical,
                    },
                    artifacts / "one_shot_sample_tensors.pt",
                )
                (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
                (configs / "train_config.json").write_text(
                    json.dumps(
                        {
                            "mode": "one_shot_sample",
                            "seed": 1234,
                            "one_shot": {
                                "decoder_diagnostic_mode": mode,
                                "decoder_diagnostic_modulation_override": "phase_amplitude",
                                "decoder_diagnostic_mask_component": metadata[
                                    "decoder_diagnostic_mask_component"
                                ],
                                "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                                "target_code_spatial_scale": 0.72,
                                "target_code_bridge_source": "none",
                                "d2nn": metadata["d2nn"],
                            },
                            "physics": {
                                "front_path_model": "relay_intensity_psf",
                                "back_path_model": "relay_psf",
                                "readout": {"primary_plane": "camera"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return run

            t025 = write_run("t025", "transparent")
            p025 = write_run("p025", "phase_only")
            a025 = write_run("a025", "amplitude_only")
            m025 = write_run("m025", "trained")
            train_run = root / "outputs" / "20260710_v47" / "runs" / "probe_a_train"
            train_artifacts = train_run / "artifacts"
            diagnostics_dir = train_artifacts / "one_shot_distill"
            diagnostics_dir.mkdir(parents=True)
            (train_artifacts / "one_shot_decoder_trainability_audit.json").write_text(
                json.dumps({"gradients": {"steps": 36}, "final": {"decoder": {"update_l2_norm": 2.0}}}),
                encoding="utf-8",
            )
            (diagnostics_dir / "epoch_0012_d2nn_diagnostics.json").write_text(
                json.dumps(
                    {
                        "masks": {
                            "amplitude": {
                                "decoder mask transmittance |t| L1": {"mean": 0.982, "std": 0.0001},
                                "decoder mask transmittance |t| L2": {"mean": 0.982, "std": 0.0002},
                            },
                            "phase": {
                                "decoder mask phase arg(t) L1": {"std": 0.36},
                                "decoder mask phase arg(t) L2": {"std": 0.28},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            v47_summary = root / "outputs" / "reports" / "v47_summary.json"
            v47_summary.parent.mkdir(parents=True)
            v47_summary.write_text(
                json.dumps(
                    {
                        "status": "capacity-still-collapsed",
                        "probes": [
                            {
                                "label": "probe_a_s025_phase_amplitude_decoder_output",
                                "checkpoint": checkpoint.as_posix(),
                                "train_run": train_run.relative_to(root).as_posix(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            v48_verdict = root / "outputs" / "reports" / "v48_verdict.json"
            v48_verdict.write_text(
                json.dumps(
                    {
                        "status": "decoder-mask-induced-artifact",
                        "errors": [],
                        "probes": [
                            {
                                "label": "T025",
                                "run": t025.relative_to(root).as_posix(),
                                "checkpoint": checkpoint.as_posix(),
                            },
                            {
                                "label": "M025",
                                "run": m025.relative_to(root).as_posix(),
                                "checkpoint": checkpoint.as_posix(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            entries = [
                f"P025|phase_only|{p025.relative_to(root)}",
                f"A025|amplitude_only|{a025.relative_to(root)}",
            ]
            cwd = Path.cwd()
            os.chdir(root)
            try:
                pending = tool.run_v49_decoder_mask_component_attribution(
                    source_v48_verdict=v48_verdict.relative_to(root),
                    source_v47_summary=v47_summary.relative_to(root),
                    probe_entries=entries,
                    output=Path("outputs/reports/pending.json"),
                )
                self.assertEqual(pending["status"], "manual-review-required")
                self.assertEqual([probe["label"] for probe in pending["probes"]], ["T025", "P025", "A025", "M025"])
                self.assertEqual(pending["review_planes"], ["physical_camera_plane"])
                self.assertEqual(pending["review_class_labels"], list(range(10)))
                self.assertTrue(Path(pending["comparison_review_image"]).is_file())
                self.assertEqual(pending["capacity_utilization"]["trainability"]["gradients"]["steps"], 36)
                self.assertIn("decoder mask phase arg(t) L1", pending["capacity_utilization"]["decoder_masks"]["phase"])
                self.assertFalse(pending["accepted_summary_written"])

                verdicts = {
                    "phase_only_unreadable": "phase-mask-induced-artifact",
                    "amplitude_only_unreadable": "amplitude-mask-induced-artifact",
                    "both_components_unreadable": "multiple-mask-components-artifact",
                    "both_components_readable": "phase-amplitude-joint-interaction-artifact",
                    "source_gate_not_reproduced": "source-gate-not-reproduced",
                }
                for verdict, expected in verdicts.items():
                    result = tool.run_v49_decoder_mask_component_attribution(
                        source_v48_verdict=v48_verdict.relative_to(root),
                        source_v47_summary=v47_summary.relative_to(root),
                        probe_entries=entries,
                        manual_verdict=verdict,
                        output=Path(f"outputs/reports/{verdict}.json"),
                    )
                    self.assertEqual(result["status"], expected)

                missing_class_run = write_run("missing_class", "phase_only", run_labels=labels[:-2])
                invalid = tool.run_v49_decoder_mask_component_attribution(
                    source_v48_verdict=v48_verdict.relative_to(root),
                    source_v47_summary=v47_summary.relative_to(root),
                    probe_entries=[
                        f"P025|phase_only|{missing_class_run.relative_to(root)}",
                        entries[1],
                    ],
                    output=Path("outputs/reports/missing_class.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v50 diagnostics")
    def test_v50_decoder_phase_layer_attribution_validates_sources_and_maps_verdicts(self) -> None:
        import torch

        tool = load_v50_decoder_phase_layer_attribution_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = Path("outputs/20260710_v47/shared_probe_a.pt")
            labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)

            def class_planes(size):
                planes = torch.zeros((20, 1, size, size), dtype=torch.float32)
                for index, class_id in enumerate(labels.tolist()):
                    offset = min(size - 2, 2 + class_id * max(1, (size - 4) // 10))
                    planes[index, 0, :, offset - 1 : offset + 1] = 1.0
                    planes[index, 0, offset - 1 : offset + 1, :] = 0.5
                return planes

            target = class_planes(100)

            def write_run(name, kind, *, run_labels=labels, target_tensor=target):
                run = root / "outputs" / "20260711_v50" / "runs" / name
                artifacts = run / "artifacts"
                configs = run / "configs"
                artifacts.mkdir(parents=True)
                configs.mkdir(parents=True)
                sample_count = int(run_labels.numel())
                middle = target_tensor[:sample_count].clone()
                physical = torch.nn.functional.interpolate(middle, size=(200, 200), mode="bilinear")
                if kind == "transparent":
                    mode = "propagation_only_decoder"
                    component = "trained"
                    requested_mask = []
                    applied_mask = []
                elif kind == "both":
                    mode = "normal"
                    component = "phase_only"
                    requested_mask = []
                    applied_mask = [True, True]
                elif kind == "layer1":
                    mode = "normal"
                    component = "phase_only"
                    requested_mask = [True, False]
                    applied_mask = [True, False]
                else:
                    mode = "normal"
                    component = "phase_only"
                    requested_mask = [False, True]
                    applied_mask = [False, True]
                metadata = {
                    "stage": "one_shot_sample",
                    "one_shot_checkpoint": checkpoint.as_posix(),
                    "decoder_diagnostic_mode": mode,
                    "decoder_diagnostic_modulation_override": "phase_amplitude",
                    "decoder_diagnostic_mask_component": component,
                    "decoder_diagnostic_phase_layer_mask": requested_mask,
                    "output_decoder_diagnostic_phase_layer_mask_applied": applied_mask,
                    "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                    "target_code_spatial_scale": 0.72,
                    "target_code_bridge_source": "none",
                    "front_path_model": "relay_intensity_psf",
                    "back_path_model": "relay_psf",
                    "readout_primary_plane": "camera",
                    "d2nn": {
                        "grid_size": 100,
                        "pixel_size": 2.0e-6,
                        "propagation_grid_size": 1024,
                        "decoder_layers": 2,
                        "layer_spacing": 0.25e-3,
                    },
                }
                torch.save(
                    {
                        "class_labels": run_labels,
                        "artifact_metadata": metadata,
                        "target_code_amplitude": middle,
                        "decoder_input_amplitude": middle.clone(),
                        "decoder_output_amplitude": middle.clone(),
                        "physical_camera_plane": physical,
                        "physical_camera_intensity_raw": physical * 2.0,
                    },
                    artifacts / "one_shot_sample_tensors.pt",
                )
                (artifacts / "one_shot_sample_metrics.json").write_text("{}", encoding="utf-8")
                (configs / "train_config.json").write_text(
                    json.dumps(
                        {
                            "mode": "one_shot_sample",
                            "seed": 1234,
                            "one_shot": {
                                "decoder_diagnostic_mode": mode,
                                "decoder_diagnostic_modulation_override": "phase_amplitude",
                                "decoder_diagnostic_mask_component": component,
                                "decoder_diagnostic_phase_layer_mask": requested_mask,
                                "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
                                "target_code_spatial_scale": 0.72,
                                "target_code_bridge_source": "none",
                                "d2nn": metadata["d2nn"],
                            },
                            "physics": {
                                "front_path_model": "relay_intensity_psf",
                                "back_path_model": "relay_psf",
                                "readout": {"primary_plane": "camera"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return run

            t025 = write_run("t025", "transparent")
            l1 = write_run("l1p025", "layer1")
            l2 = write_run("l2p025", "layer2")
            p025 = write_run("p025", "both")
            source = root / "outputs" / "reports" / "v49_verdict.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "status": "phase-mask-induced-artifact",
                        "errors": [],
                        "accepted_summary_written": False,
                        "capacity_utilization": {
                            "decoder_masks": {
                                "phase": {
                                    "decoder mask phase arg(t) L1": {"std": 0.36},
                                    "decoder mask phase arg(t) L2": {"std": 0.28},
                                }
                            }
                        },
                        "probes": [
                            {
                                "label": "T025",
                                "run": t025.relative_to(root).as_posix(),
                                "checkpoint": checkpoint.as_posix(),
                            },
                            {
                                "label": "P025",
                                "run": p025.relative_to(root).as_posix(),
                                "checkpoint": checkpoint.as_posix(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            entries = [
                f"L1P025|{l1.relative_to(root)}",
                f"L2P025|{l2.relative_to(root)}",
            ]
            cwd = Path.cwd()
            os.chdir(root)
            try:
                pending = tool.run_v50_decoder_phase_layer_attribution(
                    source_v49_verdict=source.relative_to(root),
                    probe_entries=entries,
                    output=Path("outputs/reports/pending.json"),
                )
                self.assertEqual(pending["status"], "manual-review-required")
                self.assertEqual([probe["label"] for probe in pending["probes"]], ["T025", "L1P025", "L2P025", "P025"])
                self.assertEqual(pending["review_planes"], ["physical_camera_plane"])
                self.assertEqual(pending["review_class_labels"], list(range(10)))
                self.assertTrue(Path(pending["comparison_review_image"]).is_file())
                self.assertEqual(pending["phase_observability"]["decoder mask phase arg(t) L1"]["std"], 0.36)
                self.assertGreater(pending["probes"][1]["raw_physical_energy"]["sum"], 0.0)
                self.assertEqual(pending["probe_budget"]["new_training_runs"], 0)
                self.assertFalse(pending["accepted_summary_written"])

                verdicts = {
                    "layer1_unreadable": "decoder-phase-layer1-artifact",
                    "layer2_unreadable": "decoder-phase-layer2-artifact",
                    "both_layers_unreadable": "decoder-phase-layers-independently-artifactual",
                    "both_layers_readable": "decoder-phase-layer-interaction-artifact",
                    "source_gate_not_reproduced": "source-gate-not-reproduced",
                }
                for verdict, expected in verdicts.items():
                    result = tool.run_v50_decoder_phase_layer_attribution(
                        source_v49_verdict=source.relative_to(root),
                        probe_entries=entries,
                        manual_verdict=verdict,
                        output=Path(f"outputs/reports/{verdict}.json"),
                    )
                    self.assertEqual(result["status"], expected)

                mismatch = target.clone()
                mismatch[0] = torch.roll(mismatch[0], shifts=4, dims=-1)
                mismatch_run = write_run("mismatch", "layer1", target_tensor=mismatch)
                invalid = tool.run_v50_decoder_phase_layer_attribution(
                    source_v49_verdict=source.relative_to(root),
                    probe_entries=[f"L1P025|{mismatch_run.relative_to(root)}", entries[1]],
                    output=Path("outputs/reports/mismatch.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v41 decoder diagnostics")
    def test_v41_decoder_physical_topology_recovery_classifies_probe_states(self) -> None:
        import torch

        tool = load_v41_decoder_physical_topology_recovery_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("recovered", "phase_only", True, True, "decoder-physical-recovered-needs-human-review"),
                ("camera_loss", "phase_only", True, False, "decoder-output-recovered-cmos-failed"),
                ("unreachable", "phase_only", False, False, "decoder-physical-unreachable"),
                ("phase_amp", "phase_amplitude", True, True, "phase-amplitude-upper-bound-positive"),
                ("four_layer", "four_layer", True, True, "deeper-decoder-upper-bound-positive"),
            ]
            runs = []
            for name, _kind, decoder_ok, camera_ok, _expected in cases:
                run = self._write_v37_topology_sample_run(
                    root,
                    f"v41_{name}",
                    torch,
                    target_ok=True,
                    decoder_ok=decoder_ok,
                    camera_ok=camera_ok,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                )
                runs.append(run)

            def active_constant_plane(shape):
                values = torch.tensor([0.2, 0.8, 0.35, 0.95], dtype=torch.float32).view(4, 1, 1, 1)
                return values.expand(4, 1, shape[0], shape[1]).clone()

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            for name, run in zip([case[0] for case in cases], runs, strict=True):
                if name not in {"recovered", "phase_amp", "four_layer"}:
                    continue
                bundle_path = run / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                middle = active_stroke_plane((100, 100))
                bundle["target_code_amplitude"] = middle
                bundle["decoder_output_amplitude"] = middle.clone()
                bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                torch.save(bundle, bundle_path)

            camera_loss_run = runs[1]
            camera_bundle_path = camera_loss_run / "artifacts" / "one_shot_sample_tensors.pt"
            camera_bundle = torch.load(camera_bundle_path, map_location="cpu", weights_only=True)
            camera_bundle["physical_camera_plane"] = active_constant_plane((200, 200))
            torch.save(camera_bundle, camera_bundle_path)

            v40_report = root / "outputs" / "reports" / "v40.json"
            v40_summary = root / "outputs" / "reports" / "v40_summary.json"
            v40_report.parent.mkdir(parents=True, exist_ok=True)
            v40_report.write_text(
                json.dumps({"version": "v40", "status": "label-template-collapse-recovered"}),
                encoding="utf-8",
            )
            v40_summary.write_text(
                json.dumps({"version": "v40", "diagnostic_only": True, "accepted_summary_written": False}),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            os.chdir(root)
            try:
                for (name, kind, _decoder_ok, _camera_ok, expected), run in zip(cases, runs, strict=True):
                    with self.subTest(case=name):
                        result = tool.run_v41_decoder_physical_topology_recovery(
                            source_v40_report=v40_report.relative_to(root),
                            source_v40_summary=v40_summary.relative_to(root),
                            probe_entries=[f"{name}|{kind}|{run.relative_to(root)}|{run.relative_to(root)}"],
                            output=Path(f"outputs/reports/{name}.json"),
                        )
                        self.assertEqual(result["status"], expected)
                        self.assertFalse(result["accepted_summary_written"])
                        self.assertTrue(result["human_readability_required"])

                invalid = tool.run_v41_decoder_physical_topology_recovery(
                    source_v40_report=v40_report.relative_to(root),
                    source_v40_summary=v40_summary.relative_to(root),
                    probe_entries=["missing|phase_only||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v42 bypass diagnostics")
    def test_v42_decoder_bypass_backpath_isolation_classifies_probe_states(self) -> None:
        import torch

        tool = load_v42_decoder_bypass_backpath_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                "bypass_current": self._write_v37_topology_sample_run(
                    root,
                    "v42_bypass_current",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "bypass_clean": self._write_v37_topology_sample_run(
                    root,
                    "v42_bypass_clean",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "bypass_identity_failed": self._write_v37_topology_sample_run(
                    root,
                    "v42_bypass_identity_failed",
                    torch,
                    target_ok=False,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "propagation_failed": self._write_v37_topology_sample_run(
                    root,
                    "v42_propagation_failed",
                    torch,
                    target_ok=True,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
            }

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            def blob_plane(shape):
                return torch.full((4, 1, shape[0], shape[1]), 0.5, dtype=torch.float32)

            for key in ("bypass_current", "bypass_clean"):
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                middle = active_stroke_plane((100, 100))
                bundle["target_code_amplitude"] = middle
                bundle["decoder_input_amplitude"] = middle.clone()
                bundle["decoder_output_amplitude"] = middle.clone()
                bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                torch.save(bundle, bundle_path)

            prop_path = runs["propagation_failed"] / "artifacts" / "one_shot_sample_tensors.pt"
            prop_bundle = torch.load(prop_path, map_location="cpu", weights_only=True)
            middle = active_stroke_plane((100, 100))
            prop_bundle["target_code_amplitude"] = middle
            prop_bundle["decoder_input_amplitude"] = middle.clone()
            prop_bundle["decoder_output_amplitude"] = blob_plane((100, 100))
            prop_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(prop_bundle, prop_path)

            v41_report = root / "outputs" / "reports" / "v41.json"
            v41_summary = root / "outputs" / "reports" / "v41_summary.json"
            v41_report.parent.mkdir(parents=True, exist_ok=True)
            v41_report.write_text(json.dumps({"version": "v41", "status": "decoder-physical-unreachable"}), encoding="utf-8")
            v41_summary.write_text(json.dumps({"version": "v41", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                recovered = tool.run_v42_decoder_bypass_backpath_isolation(
                    source_v41_report=v41_report.relative_to(root),
                    source_v41_summary=v41_summary.relative_to(root),
                    probe_entries=[
                        f"a|bypass_current|{runs['bypass_current'].relative_to(root)}",
                        f"d|propagation_only|{runs['propagation_failed'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/recovered.json"),
                )
                self.assertEqual(recovered["status"], "decoder-propagation-bottleneck")
                self.assertFalse(recovered["accepted_summary_written"])

                mask = tool.run_v42_decoder_bypass_backpath_isolation(
                    source_v41_report=v41_report.relative_to(root),
                    source_v41_summary=v41_summary.relative_to(root),
                    probe_entries=[
                        f"a|bypass_current|{runs['bypass_current'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/mask.json"),
                )
                self.assertEqual(mask["status"], "decoder-mask-bottleneck")

                backpath = tool.run_v42_decoder_bypass_backpath_isolation(
                    source_v41_report=v41_report.relative_to(root),
                    source_v41_summary=v41_summary.relative_to(root),
                    probe_entries=[
                        f"a|bypass_current|{runs['propagation_failed'].relative_to(root)}",
                        f"b|bypass_clean|{runs['bypass_clean'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/backpath.json"),
                )
                self.assertEqual(backpath["status"], "backpath-camera-blur-bottleneck")

                target_scale = tool.run_v42_decoder_bypass_backpath_isolation(
                    source_v41_report=v41_report.relative_to(root),
                    source_v41_summary=v41_summary.relative_to(root),
                    probe_entries=[
                        f"c|bypass_identity|{runs['bypass_identity_failed'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/target_scale.json"),
                )
                self.assertEqual(target_scale["status"], "target-code-scale-mismatch")

                invalid = tool.run_v42_decoder_bypass_backpath_isolation(
                    source_v41_report=v41_report.relative_to(root),
                    source_v41_summary=v41_summary.relative_to(root),
                    probe_entries=["missing|bypass_current|"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v43 geometry diagnostics")
    def test_v43_decoder_propagation_geometry_isolation_classifies_probe_states(self) -> None:
        import torch

        tool = load_v43_decoder_propagation_geometry_isolation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                "spacing": self._write_v37_topology_sample_run(
                    root,
                    "v43_spacing",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "aperture": self._write_v37_topology_sample_run(
                    root,
                    "v43_aperture",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "target_scale": self._write_v37_topology_sample_run(
                    root,
                    "v43_target_scale",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "failed": self._write_v37_topology_sample_run(
                    root,
                    "v43_failed",
                    torch,
                    target_ok=True,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
            }

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            def blob_plane(shape):
                return torch.full((4, 1, shape[0], shape[1]), 0.5, dtype=torch.float32)

            for key in ("spacing", "aperture", "target_scale"):
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                middle = active_stroke_plane((100, 100))
                bundle["target_code_amplitude"] = middle
                bundle["decoder_input_amplitude"] = middle.clone()
                bundle["decoder_output_amplitude"] = middle.clone()
                bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                bundle["diffractive_decoder_crop_crop_retention"] = torch.full((4, 2), 0.95)
                bundle["diffractive_decoder_crop_propagation_energy_ratio"] = torch.full((4, 2), 1.0)
                torch.save(bundle, bundle_path)

            failed_path = runs["failed"] / "artifacts" / "one_shot_sample_tensors.pt"
            failed_bundle = torch.load(failed_path, map_location="cpu", weights_only=True)
            middle = active_stroke_plane((100, 100))
            failed_bundle["target_code_amplitude"] = middle
            failed_bundle["decoder_input_amplitude"] = middle.clone()
            failed_bundle["decoder_output_amplitude"] = blob_plane((100, 100))
            failed_bundle["physical_camera_plane"] = blob_plane((200, 200))
            failed_bundle["diffractive_decoder_crop_crop_retention"] = torch.full((4, 2), 0.35)
            failed_bundle["diffractive_decoder_crop_propagation_energy_ratio"] = torch.full((4, 2), 1.0)
            torch.save(failed_bundle, failed_path)

            v42_report = root / "outputs" / "reports" / "v42.json"
            v42_summary = root / "outputs" / "reports" / "v42_summary.json"
            v42_report.parent.mkdir(parents=True, exist_ok=True)
            v42_report.write_text(json.dumps({"version": "v42", "status": "target-code-scale-mismatch"}), encoding="utf-8")
            v42_summary.write_text(json.dumps({"version": "v42", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                spacing = tool.run_v43_decoder_propagation_geometry_isolation(
                    source_v42_report=v42_report.relative_to(root),
                    source_v42_summary=v42_summary.relative_to(root),
                    probe_entries=[
                        f"s|spacing|0.10|1024|sharp_stroke_digit_code|{runs['spacing'].relative_to(root)}",
                        f"f|spacing|1.50|1024|sharp_stroke_digit_code|{runs['failed'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/spacing.json"),
                )
                self.assertEqual(spacing["status"], "spacing-propagation-bottleneck")
                self.assertFalse(spacing["accepted_summary_written"])

                aperture = tool.run_v43_decoder_propagation_geometry_isolation(
                    source_v42_report=v42_report.relative_to(root),
                    source_v42_summary=v42_summary.relative_to(root),
                    probe_entries=[
                        f"a|aperture|0.10|1536|sharp_stroke_digit_code|{runs['aperture'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/aperture.json"),
                )
                self.assertEqual(aperture["status"], "aperture-crop-bottleneck")

                target_scale = tool.run_v43_decoder_propagation_geometry_isolation(
                    source_v42_report=v42_report.relative_to(root),
                    source_v42_summary=v42_summary.relative_to(root),
                    probe_entries=[
                        f"t|target_scale|1.50|1024|small_sharp_stroke_digit_code|{runs['target_scale'].relative_to(root)}",
                    ],
                    output=Path("outputs/reports/target_scale.json"),
                )
                self.assertEqual(target_scale["status"], "target-scale-bottleneck")

                invalid = tool.run_v43_decoder_propagation_geometry_isolation(
                    source_v42_report=v42_report.relative_to(root),
                    source_v42_summary=v42_summary.relative_to(root),
                    probe_entries=["missing|spacing|0.10|1024|sharp_stroke_digit_code|"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v44 spatial scaling diagnostics")
    def test_v44_target_code_spatial_scaling_decoder_recovery_classifies_probe_states(self) -> None:
        import torch

        tool = load_v44_target_code_spatial_scaling_decoder_recovery_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                "recovered": self._write_v37_topology_sample_run(
                    root,
                    "v44_recovered",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "phase_amplitude": self._write_v37_topology_sample_run(
                    root,
                    "v44_phase_amplitude",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "short_spacing": self._write_v37_topology_sample_run(
                    root,
                    "v44_short_spacing",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "camera_loss": self._write_v37_topology_sample_run(
                    root,
                    "v44_camera_loss",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "failed": self._write_v37_topology_sample_run(
                    root,
                    "v44_failed",
                    torch,
                    target_ok=True,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
            }

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            def blob_plane(shape):
                return torch.full((4, 1, shape[0], shape[1]), 0.5, dtype=torch.float32)

            for key in ("recovered", "phase_amplitude", "short_spacing"):
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
                bundle["decoder_output_amplitude"] = active_stroke_plane((100, 100))
                bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                bundle["target_code_spatial_scale"] = torch.tensor(0.72)
                torch.save(bundle, bundle_path)

            camera_loss_path = runs["camera_loss"] / "artifacts" / "one_shot_sample_tensors.pt"
            camera_loss_bundle = torch.load(camera_loss_path, map_location="cpu", weights_only=True)
            camera_loss_bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
            camera_loss_bundle["decoder_output_amplitude"] = active_stroke_plane((100, 100))
            camera_loss_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(camera_loss_bundle, camera_loss_path)

            failed_path = runs["failed"] / "artifacts" / "one_shot_sample_tensors.pt"
            failed_bundle = torch.load(failed_path, map_location="cpu", weights_only=True)
            failed_bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
            failed_bundle["decoder_output_amplitude"] = blob_plane((100, 100))
            failed_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(failed_bundle, failed_path)

            v43_report = root / "outputs" / "reports" / "v43.json"
            v43_summary = root / "outputs" / "reports" / "v43_summary.json"
            v43_report.parent.mkdir(parents=True, exist_ok=True)
            v43_report.write_text(json.dumps({"version": "v43", "status": "target-scale-bottleneck"}), encoding="utf-8")
            v43_summary.write_text(json.dumps({"version": "v43", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                recovered = tool.run_v44_target_code_spatial_scaling_decoder_recovery(
                    source_v43_report=v43_report.relative_to(root),
                    source_v43_summary=v43_summary.relative_to(root),
                    probe_entries=[
                        f"a|phase_only|0.72|1.50|{runs['recovered'].relative_to(root)}|train_a",
                    ],
                    output=Path("outputs/reports/recovered.json"),
                )
                self.assertEqual(recovered["status"], "target-scale-decoder-recovered-needs-human-review")
                self.assertFalse(recovered["accepted_summary_written"])

                phase_amp = tool.run_v44_target_code_spatial_scaling_decoder_recovery(
                    source_v43_report=v43_report.relative_to(root),
                    source_v43_summary=v43_summary.relative_to(root),
                    probe_entries=[
                        f"d|phase_amplitude|0.72|1.50|{runs['phase_amplitude'].relative_to(root)}|train_d",
                    ],
                    output=Path("outputs/reports/phase_amp.json"),
                )
                self.assertEqual(phase_amp["status"], "phase-only-decoder-constraint-bottleneck")

                short_spacing = tool.run_v44_target_code_spatial_scaling_decoder_recovery(
                    source_v43_report=v43_report.relative_to(root),
                    source_v43_summary=v43_summary.relative_to(root),
                    probe_entries=[
                        f"e|short_spacing|0.72|0.50|{runs['short_spacing'].relative_to(root)}|train_e",
                    ],
                    output=Path("outputs/reports/short_spacing.json"),
                )
                self.assertEqual(short_spacing["status"], "spacing-propagation-bottleneck")

                camera_loss = tool.run_v44_target_code_spatial_scaling_decoder_recovery(
                    source_v43_report=v43_report.relative_to(root),
                    source_v43_summary=v43_summary.relative_to(root),
                    probe_entries=[
                        f"c|phase_only_physical_cmos|0.72|1.50|{runs['camera_loss'].relative_to(root)}|train_c",
                    ],
                    output=Path("outputs/reports/camera_loss.json"),
                )
                self.assertEqual(camera_loss["status"], "camera-after-decoder-loss")

                invalid = tool.run_v44_target_code_spatial_scaling_decoder_recovery(
                    source_v43_report=v43_report.relative_to(root),
                    source_v43_summary=v43_summary.relative_to(root),
                    probe_entries=["missing|phase_only|0.72|1.50||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v45 spacing diagnostics")
    def test_v45_decoder_spacing_near_field_recovery_classifies_probe_states(self) -> None:
        import torch

        tool = load_v45_decoder_spacing_near_field_recovery_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                "near_field": self._write_v37_topology_sample_run(
                    root,
                    "v45_near_field",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "physical_loss": self._write_v37_topology_sample_run(
                    root,
                    "v45_physical_loss",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "texture_limited": self._write_v37_topology_sample_run(
                    root,
                    "v45_texture_limited",
                    torch,
                    target_ok=True,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
                "camera_loss": self._write_v37_topology_sample_run(
                    root,
                    "v45_camera_loss",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                ),
            }

            def active_stroke_plane(shape):
                height, width = shape
                tensor = torch.zeros((4, 1, height, width), dtype=torch.float32)
                for idx in range(4):
                    offset = idx % 4
                    tensor[idx, 0, :, offset::4] = 1.0
                    tensor[idx, 0, height // 4 : height // 4 + 2, :] = 0.8
                    tensor[idx, 0, 3 * height // 4 : 3 * height // 4 + 2, :] = 0.8
                return tensor

            def blob_plane(shape):
                return torch.full((4, 1, shape[0], shape[1]), 0.5, dtype=torch.float32)

            for key in ("near_field", "physical_loss"):
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
                bundle["decoder_output_amplitude"] = active_stroke_plane((100, 100))
                bundle["physical_camera_plane"] = active_stroke_plane((200, 200))
                torch.save(bundle, bundle_path)

            texture_path = runs["texture_limited"] / "artifacts" / "one_shot_sample_tensors.pt"
            texture_bundle = torch.load(texture_path, map_location="cpu", weights_only=True)
            texture_bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
            texture_bundle["decoder_output_amplitude"] = blob_plane((100, 100))
            texture_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(texture_bundle, texture_path)

            camera_path = runs["camera_loss"] / "artifacts" / "one_shot_sample_tensors.pt"
            camera_bundle = torch.load(camera_path, map_location="cpu", weights_only=True)
            camera_bundle["target_code_amplitude"] = active_stroke_plane((100, 100))
            camera_bundle["decoder_output_amplitude"] = active_stroke_plane((100, 100))
            camera_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(camera_bundle, camera_path)

            v44_report = root / "outputs" / "reports" / "v44.json"
            v44_summary = root / "outputs" / "reports" / "v44_summary.json"
            v44_report.parent.mkdir(parents=True, exist_ok=True)
            v44_report.write_text(json.dumps({"version": "v44", "status": "spacing-propagation-bottleneck"}), encoding="utf-8")
            v44_summary.write_text(json.dumps({"version": "v44", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                near_field = tool.run_v45_decoder_spacing_near_field_recovery(
                    source_v44_report=v44_report.relative_to(root),
                    source_v44_summary=v44_summary.relative_to(root),
                    probe_entries=[
                        f"a|spacing|0.50|decoder_output_amplitude|0.72|{runs['near_field'].relative_to(root)}|train_a",
                    ],
                    output=Path("outputs/reports/near_field.json"),
                )
                self.assertEqual(near_field["status"], "near-field-spacing-recovered-needs-human-review")
                self.assertFalse(near_field["accepted_summary_written"])

                physical_loss = tool.run_v45_decoder_spacing_near_field_recovery(
                    source_v44_report=v44_report.relative_to(root),
                    source_v44_summary=v44_summary.relative_to(root),
                    probe_entries=[
                        f"e|physical_cmos_loss|0.50|physical_camera_plane|0.72|{runs['physical_loss'].relative_to(root)}|train_e",
                    ],
                    output=Path("outputs/reports/physical_loss.json"),
                )
                self.assertEqual(physical_loss["status"], "physical-cmos-loss-helped")

                texture_limited = tool.run_v45_decoder_spacing_near_field_recovery(
                    source_v44_report=v44_report.relative_to(root),
                    source_v44_summary=v44_summary.relative_to(root),
                    probe_entries=[
                        f"b|spacing|0.75|decoder_output_amplitude|0.72|{runs['texture_limited'].relative_to(root)}|train_b",
                    ],
                    output=Path("outputs/reports/texture_limited.json"),
                )
                self.assertEqual(texture_limited["status"], "spacing-still-texture-limited")

                camera_loss = tool.run_v45_decoder_spacing_near_field_recovery(
                    source_v44_report=v44_report.relative_to(root),
                    source_v44_summary=v44_summary.relative_to(root),
                    probe_entries=[
                        f"c|spacing|0.25|decoder_output_amplitude|0.72|{runs['camera_loss'].relative_to(root)}|train_c",
                    ],
                    output=Path("outputs/reports/camera_loss.json"),
                )
                self.assertEqual(camera_loss["status"], "camera-after-decoder-loss")

                invalid = tool.run_v45_decoder_spacing_near_field_recovery(
                    source_v44_report=v44_report.relative_to(root),
                    source_v44_summary=v44_summary.relative_to(root),
                    probe_entries=["missing|spacing|0.50|decoder_output_amplitude|0.72||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v46 semantic diagnostics")
    def test_v46_near_field_semantic_separation_classifies_probe_states(self) -> None:
        import torch

        tool = load_v46_near_field_semantic_separation_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                "recovered": self._write_v37_topology_sample_run(
                    root,
                    "v46_recovered",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                    physical_batch=6,
                ),
                "collapse": self._write_v37_topology_sample_run(
                    root,
                    "v46_collapse",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                    physical_batch=6,
                ),
                "texture": self._write_v37_topology_sample_run(
                    root,
                    "v46_texture",
                    torch,
                    target_ok=True,
                    decoder_ok=False,
                    camera_ok=False,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                    physical_batch=6,
                ),
                "final_plane": self._write_v37_topology_sample_run(
                    root,
                    "v46_final_plane",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                    physical_batch=6,
                ),
            }

            def separated_plane(shape):
                height, width = shape
                tensor = torch.zeros((6, 1, height, width), dtype=torch.float32)
                labels = [0, 0, 1, 1, 2, 2]
                for idx, label in enumerate(labels):
                    stroke = max(2, width // 30)
                    if label == 0:
                        x0 = width // 3
                        tensor[idx, 0, height // 4 : 3 * height // 4, x0 : x0 + stroke] = 1.0
                        tensor[idx, 0, height // 4 : height // 4 + stroke, x0 : 2 * width // 3] = 0.8
                        tensor[idx, 0, 3 * height // 4 - stroke : 3 * height // 4, x0 : 2 * width // 3] = 0.8
                    elif label == 1:
                        tensor[idx, 0, height // 4 : height // 4 + stroke, width // 4 : 3 * width // 4] = 1.0
                        tensor[idx, 0, 3 * height // 4 - stroke : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
                        tensor[idx, 0, height // 4 : 3 * height // 4, width // 2 : width // 2 + stroke] = 0.8
                    else:
                        x0 = 2 * width // 3
                        tensor[idx, 0, height // 4 : 3 * height // 4, x0 : x0 + stroke] = 1.0
                        tensor[idx, 0, height // 2 : height // 2 + stroke, width // 3 : x0 + stroke] = 0.8
                return tensor

            def collapsed_plane(shape):
                height, width = shape
                tensor = torch.zeros((6, 1, height, width), dtype=torch.float32)
                stroke = max(2, width // 24)
                tensor[:, 0, height // 4 : 3 * height // 4, width // 2 : width // 2 + stroke] = 1.0
                tensor[:, 0, height // 2 : height // 2 + stroke, width // 4 : 3 * width // 4] = 0.8
                tensor[:, 0, height // 4 : height // 4 + stroke, width // 4 : 3 * width // 4] = 0.8
                return tensor

            def blob_plane(shape):
                return torch.full((6, 1, shape[0], shape[1]), 0.5, dtype=torch.float32)

            for key, plane in {
                "recovered": separated_plane,
                "final_plane": separated_plane,
            }.items():
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                bundle["class_labels"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                bundle["target_code_amplitude"] = plane((100, 100))
                bundle["decoder_output_amplitude"] = plane((100, 100))
                bundle["physical_camera_plane"] = plane((200, 200))
                torch.save(bundle, bundle_path)

            collapse_path = runs["collapse"] / "artifacts" / "one_shot_sample_tensors.pt"
            collapse_bundle = torch.load(collapse_path, map_location="cpu", weights_only=True)
            collapse_bundle["class_labels"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
            collapse_bundle["target_code_amplitude"] = separated_plane((100, 100))
            collapse_bundle["decoder_output_amplitude"] = collapsed_plane((100, 100))
            collapse_bundle["physical_camera_plane"] = collapsed_plane((200, 200))
            torch.save(collapse_bundle, collapse_path)

            texture_path = runs["texture"] / "artifacts" / "one_shot_sample_tensors.pt"
            texture_bundle = torch.load(texture_path, map_location="cpu", weights_only=True)
            texture_bundle["class_labels"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
            texture_bundle["target_code_amplitude"] = separated_plane((100, 100))
            texture_bundle["decoder_output_amplitude"] = blob_plane((100, 100))
            texture_bundle["physical_camera_plane"] = blob_plane((200, 200))
            torch.save(texture_bundle, texture_path)

            v45_report = root / "outputs" / "reports" / "v45.json"
            v45_summary = root / "outputs" / "reports" / "v45_summary.json"
            v45_report.parent.mkdir(parents=True, exist_ok=True)
            v45_report.write_text(json.dumps({"version": "v45", "status": "near-field-spacing-recovered-needs-human-review"}), encoding="utf-8")
            v45_summary.write_text(json.dumps({"version": "v45", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                recovered = tool.run_v46_near_field_semantic_separation(
                    source_v45_report=v45_report.relative_to(root),
                    source_v45_summary=v45_summary.relative_to(root),
                    probe_entries=[
                        f"a|longer_decoder|decoder_output_amplitude|0.25|{runs['recovered'].relative_to(root)}|train_a",
                    ],
                    output=Path("outputs/reports/recovered.json"),
                )
                self.assertEqual(recovered["status"], "semantic-separation-recovered-needs-human-review")
                self.assertFalse(recovered["accepted_summary_written"])

                collapse = tool.run_v46_near_field_semantic_separation(
                    source_v45_report=v45_report.relative_to(root),
                    source_v45_summary=v45_summary.relative_to(root),
                    probe_entries=[
                        f"b|cross_class|decoder_output_amplitude|0.25|{runs['collapse'].relative_to(root)}|train_b",
                    ],
                    output=Path("outputs/reports/collapse.json"),
                )
                self.assertEqual(collapse["status"], "semantic-collapse-still-high")
                self.assertGreater(collapse["probes"][0]["class_correlation"]["max_offdiag_corr"], 0.90)

                texture = tool.run_v46_near_field_semantic_separation(
                    source_v45_report=v45_report.relative_to(root),
                    source_v45_summary=v45_summary.relative_to(root),
                    probe_entries=[
                        f"c|hard_pair|decoder_output_amplitude|0.25|{runs['texture'].relative_to(root)}|train_c",
                    ],
                    output=Path("outputs/reports/texture.json"),
                )
                self.assertEqual(texture["status"], "texture-speckle-limited")

                final_plane = tool.run_v46_near_field_semantic_separation(
                    source_v45_report=v45_report.relative_to(root),
                    source_v45_summary=v45_summary.relative_to(root),
                    probe_entries=[
                        f"d|final_plane|physical_camera_plane|0.25|{runs['final_plane'].relative_to(root)}|train_d",
                    ],
                    output=Path("outputs/reports/final_plane.json"),
                )
                self.assertEqual(final_plane["status"], "final-plane-loss-helped")

                invalid = tool.run_v46_near_field_semantic_separation(
                    source_v45_report=v45_report.relative_to(root),
                    source_v45_summary=v45_summary.relative_to(root),
                    probe_entries=["missing|cross_class|decoder_output_amplitude|0.25||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for v47 decoder capacity diagnostics")
    def test_v47_near_field_decoder_capacity_upper_bound_classifies_probe_states(self) -> None:
        import torch

        tool = load_v47_near_field_decoder_capacity_upper_bound_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = {
                key: self._write_v37_topology_sample_run(
                    root,
                    f"v47_{key}",
                    torch,
                    target_ok=True,
                    decoder_ok=True,
                    camera_ok=True,
                    middle_shape=(100, 100),
                    physical_shape=(200, 200),
                    physical_batch=6,
                )
                for key in ("modulation", "depth", "padding", "collapsed")
            }

            def separated_plane(shape):
                height, width = shape
                tensor = torch.zeros((6, 1, height, width), dtype=torch.float32)
                labels = [0, 0, 1, 1, 2, 2]
                for idx, label in enumerate(labels):
                    stroke = max(2, width // 30)
                    if label == 0:
                        x0 = width // 3
                        tensor[idx, 0, height // 4 : 3 * height // 4, x0 : x0 + stroke] = 1.0
                        tensor[idx, 0, height // 4 : height // 4 + stroke, x0 : 2 * width // 3] = 0.8
                        tensor[idx, 0, 3 * height // 4 - stroke : 3 * height // 4, x0 : 2 * width // 3] = 0.8
                    elif label == 1:
                        tensor[idx, 0, height // 4 : height // 4 + stroke, width // 4 : 3 * width // 4] = 1.0
                        tensor[idx, 0, 3 * height // 4 - stroke : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
                        tensor[idx, 0, height // 4 : 3 * height // 4, width // 2 : width // 2 + stroke] = 0.8
                    else:
                        x0 = 2 * width // 3
                        tensor[idx, 0, height // 4 : 3 * height // 4, x0 : x0 + stroke] = 1.0
                        tensor[idx, 0, height // 2 : height // 2 + stroke, width // 3 : x0 + stroke] = 0.8
                return tensor

            def collapsed_plane(shape):
                height, width = shape
                tensor = torch.zeros((6, 1, height, width), dtype=torch.float32)
                stroke = max(2, width // 24)
                tensor[:, 0, height // 4 : 3 * height // 4, width // 2 : width // 2 + stroke] = 1.0
                tensor[:, 0, height // 2 : height // 2 + stroke, width // 4 : 3 * width // 4] = 0.8
                tensor[:, 0, height // 4 : height // 4 + stroke, width // 4 : 3 * width // 4] = 0.8
                return tensor

            for key in ("modulation", "depth", "padding"):
                bundle_path = runs[key] / "artifacts" / "one_shot_sample_tensors.pt"
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
                bundle["class_labels"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                bundle["target_code_amplitude"] = separated_plane((100, 100))
                bundle["decoder_output_amplitude"] = separated_plane((100, 100))
                bundle["physical_camera_plane"] = separated_plane((200, 200))
                bundle["decoder_output_phase"] = torch.zeros((6, 1, 100, 100), dtype=torch.float32)
                torch.save(bundle, bundle_path)

            collapsed_path = runs["collapsed"] / "artifacts" / "one_shot_sample_tensors.pt"
            collapsed_bundle = torch.load(collapsed_path, map_location="cpu", weights_only=True)
            collapsed_bundle["class_labels"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
            collapsed_bundle["target_code_amplitude"] = separated_plane((100, 100))
            collapsed_bundle["decoder_output_amplitude"] = collapsed_plane((100, 100))
            collapsed_bundle["physical_camera_plane"] = collapsed_plane((200, 200))
            torch.save(collapsed_bundle, collapsed_path)

            v45_report = root / "outputs" / "reports" / "v45.json"
            v46_report = root / "outputs" / "reports" / "v46.json"
            v46_summary = root / "outputs" / "reports" / "v46_summary.json"
            v45_report.parent.mkdir(parents=True, exist_ok=True)
            v45_report.write_text(json.dumps({"version": "v45", "status": "near-field-spacing-recovered-needs-human-review"}), encoding="utf-8")
            v46_report.write_text(json.dumps({"version": "v46", "status": "semantic-collapse-still-high"}), encoding="utf-8")
            v46_summary.write_text(json.dumps({"version": "v46", "diagnostic_only": True}), encoding="utf-8")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                modulation = tool.run_v47_near_field_decoder_capacity_upper_bound(
                    source_v45_report=v45_report.relative_to(root),
                    source_v46_report=v46_report.relative_to(root),
                    source_v46_summary=v46_summary.relative_to(root),
                    probe_entries=[
                        f"a|phase_amplitude|decoder_output_amplitude|phase_amplitude|2|1024|{runs['modulation'].relative_to(root)}|train_a",
                    ],
                    output=Path("outputs/reports/modulation.json"),
                )
                self.assertEqual(modulation["status"], "modulation-constraint-recovered-needs-human-review")
                self.assertFalse(modulation["accepted_summary_written"])

                depth = tool.run_v47_near_field_decoder_capacity_upper_bound(
                    source_v45_report=v45_report.relative_to(root),
                    source_v46_report=v46_report.relative_to(root),
                    source_v46_summary=v46_summary.relative_to(root),
                    probe_entries=[
                        f"b|depth_4layer|decoder_output_amplitude|none|4|1024|{runs['depth'].relative_to(root)}|train_b",
                    ],
                    output=Path("outputs/reports/depth.json"),
                )
                self.assertEqual(depth["status"], "decoder-depth-recovered-needs-human-review")

                padding = tool.run_v47_near_field_decoder_capacity_upper_bound(
                    source_v45_report=v45_report.relative_to(root),
                    source_v46_report=v46_report.relative_to(root),
                    source_v46_summary=v46_summary.relative_to(root),
                    probe_entries=[
                        f"c|padding_grid|decoder_output_amplitude|none|2|1536|{runs['padding'].relative_to(root)}|train_c",
                    ],
                    output=Path("outputs/reports/padding.json"),
                )
                self.assertEqual(padding["status"], "padding-recovered-needs-human-review")

                collapsed = tool.run_v47_near_field_decoder_capacity_upper_bound(
                    source_v45_report=v45_report.relative_to(root),
                    source_v46_report=v46_report.relative_to(root),
                    source_v46_summary=v46_summary.relative_to(root),
                    probe_entries=[
                        f"d|phase_amplitude_final|physical_camera_plane|phase_amplitude|2|1024|{runs['collapsed'].relative_to(root)}|train_d",
                    ],
                    output=Path("outputs/reports/collapsed.json"),
                )
                self.assertEqual(collapsed["status"], "capacity-still-collapsed")
                self.assertGreater(collapsed["probes"][0]["class_correlation"]["max_offdiag_corr"], 0.90)

                invalid = tool.run_v47_near_field_decoder_capacity_upper_bound(
                    source_v45_report=v45_report.relative_to(root),
                    source_v46_report=v46_report.relative_to(root),
                    source_v46_summary=v46_summary.relative_to(root),
                    probe_entries=["missing|padding_grid|decoder_output_amplitude|none|2|1536||"],
                    output=Path("outputs/reports/invalid.json"),
                )
                self.assertEqual(invalid["status"], "harness-invalid")
                self.assertTrue(invalid["errors"])
            finally:
                os.chdir(cwd)

    def test_v30_configs_preserve_contract_and_launcher_is_diagnostic_only(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            (
                "train_twophoton_100x100_v30_probe_a_spatial_identity_code_decoder_same_batch_e4.yaml",
                "spatial_identity_code",
                "diffractive_phase_only",
            ),
            (
                "train_twophoton_100x100_v30_probe_b_lowfreq_digit_code_decoder_same_batch_e4.yaml",
                "lowfreq_digit_code",
                "diffractive_phase_only",
            ),
            (
                "train_twophoton_100x100_v30_probe_c_class_codebook_grid_decoder_same_batch_e4.yaml",
                "class_codebook_grid",
                "diffractive_phase_only",
            ),
            (
                "train_twophoton_100x100_v30_probe_d_camera_direct_target_code_oracle_e2.yaml",
                "spatial_identity_code",
                "camera_direct_clean_oracle",
            ),
        ]
        for name, target_code_mode, optical_core_arch in cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.optical_core_arch, optical_core_arch)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_code_mode)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template")
                if cfg.one_shot.d2nn is not None:
                    self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                    self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                    self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                    self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                    self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")
        launcher = config_dir / "run_twophoton_100x100_v30_target_code_architecture_commands.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("diagnose_v30_target_code_alignment.py", text)
        self.assertIn("accepted_summary_written", text)
        self.assertIn("SAMPLE_ELIGIBLE", text)
        self.assertIn("same-batch", text)

    def test_v30_1_and_v31_configs_preserve_contract_and_sample_gate_policy(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v31_cases = [
            (
                "train_twophoton_100x100_v31_probe_a_encoded_to_lowfreq_bridge_e4.yaml",
                "encoded_amplitude",
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v31_probe_b_ols_to_lowfreq_bridge_e4.yaml",
                "ols_output_amplitude",
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v31_probe_c_ols_to_spatial_bridge_e4.yaml",
                "ols_output_amplitude",
                "spatial_identity_code",
            ),
        ]
        for name, bridge_source, target_mode in v31_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.optical_core_arch, "diffractive_phase_only")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template")
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self._assert_twophoton_d2nn_contract(cfg)

        v30_1_launcher = config_dir / "run_twophoton_100x100_v30_1_sample_gate_recovery_commands.sh"
        v30_1_text = v30_1_launcher.read_text(encoding="utf-8")
        self.assertIn("sample_gate_recovery_no_retrain", v30_1_text)
        self.assertIn("diagnose_v30_1_sample_gate_recovery.py", v30_1_text)
        self.assertIn("probe_a_spatial_identity_code_decoder", v30_1_text)
        self.assertIn("probe_b_lowfreq_digit_code_decoder", v30_1_text)
        self.assertIn('label not in {"probe_a_spatial_identity_code_decoder", "probe_b_lowfreq_digit_code_decoder"}', v30_1_text)
        self.assertNotIn("--stage one_shot_distill \\", v30_1_text)

        v31_launcher = config_dir / "run_twophoton_100x100_v31_lowfreq_target_code_bridge_commands.sh"
        v31_text = v31_launcher.read_text(encoding="utf-8")
        self.assertIn("REPLAY_PAYLOAD", v31_text)
        self.assertIn("--distill-dataset \"${REPLAY_PAYLOAD}\"", v31_text)
        self.assertIn("--distill-validation-dataset \"${REPLAY_PAYLOAD}\"", v31_text)
        self.assertIn("No complete v31 eval runs were produced", v31_text)
        self.assertIn("diagnostic-skipped-not-same-batch", v31_text)
        self.assertIn("--one-shot-generator-checkpoint", v31_text)
        self.assertIn("--one-shot-optical-checkpoint", v31_text)
        self.assertIn("diagnose_v30_target_code_alignment.py", v31_text)
        self.assertIn("accepted_summary_written", v31_text)
        self.assertIn("target_code_bridge_source", v31_text)

    def test_v31_1_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v31_1_cases = [
            (
                "train_twophoton_100x100_v31_1_probe_a_encoded_to_lowfreq_bridge_same_batch_e8.yaml",
                "encoded_amplitude",
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v31_1_probe_b_ols_to_lowfreq_bridge_same_batch_e8.yaml",
                "ols_output_amplitude",
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v31_1_probe_c_ols_to_spatial_bridge_same_batch_e8.yaml",
                "ols_output_amplitude",
                "spatial_identity_code",
            ),
        ]
        for name, bridge_source, target_mode in v31_1_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertTrue(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, "target_code_amplitude")
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v31_1_launcher = config_dir / "run_twophoton_100x100_v31_1_bridge_same_batch_commands.sh"
        v31_1_text = v31_1_launcher.read_text(encoding="utf-8")
        self.assertIn("diagnose_v31_bridge_topology.py", v31_1_text)
        self.assertIn("same_batch_bridge_gate_passed", v31_1_text)
        self.assertIn("--stage one_shot_sample", v31_1_text)
        self.assertIn("accepted_summary_written", v31_1_text)

    def test_v32_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v32_cases = [
            (
                "train_twophoton_100x100_v32_probe_a_ols_lowfreq_deep_bridge_same_batch_e10.yaml",
                "ols_output_amplitude",
                "lowfreq_digit_code",
                "deep_residual",
                32,
            ),
            (
                "train_twophoton_100x100_v32_probe_b_ols_spatial_multiscale_bridge_same_batch_e10.yaml",
                "ols_output_amplitude",
                "spatial_identity_code",
                "multiscale",
                24,
            ),
            (
                "train_twophoton_100x100_v32_probe_c_ols_thick_lowfreq_deep_bridge_same_batch_e10.yaml",
                "ols_output_amplitude",
                "thick_lowfreq_digit_code",
                "deep_residual",
                32,
            ),
            (
                "train_twophoton_100x100_v32_probe_d_encoded_thick_lowfreq_deep_bridge_same_batch_e10.yaml",
                "encoded_amplitude",
                "thick_lowfreq_digit_code",
                "deep_residual",
                32,
            ),
        ]
        for name, bridge_source, target_mode, bridge_arch, hidden_channels in v32_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 10)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, bridge_arch)
                self.assertEqual(cfg.one_shot.target_code_bridge_hidden_channels, hidden_channels)
                self.assertTrue(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, "target_code_amplitude")
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v32_launcher = config_dir / "run_twophoton_100x100_v32_bridge_redesign_commands.sh"
        v32_text = v32_launcher.read_text(encoding="utf-8")
        self.assertIn("bridge_target_code_redesign_same_batch_hard_gate", v32_text)
        self.assertIn("--version v32", v32_text)
        self.assertIn("target_code_bridge_arch", v32_text)
        self.assertIn("same_batch_bridge_gate_passed", v32_text)
        self.assertIn("--stage one_shot_sample", v32_text)
        self.assertIn("accepted_summary_written", v32_text)

        v32_1_launcher = config_dir / "run_twophoton_100x100_v32_1_sample_gap_isolation_commands.sh"
        v32_1_text = v32_1_launcher.read_text(encoding="utf-8")
        self.assertIn("sample_gap_isolation", v32_1_text)
        self.assertIn("diagnose_v32_1_sample_gap_isolation.py", v32_1_text)
        self.assertIn("twophoton_100x100_v32_bridge_redesign_summary_20260703_150903.json", v32_1_text)
        self.assertIn("twophoton_100x100_v32_bridge_topology_20260703_150903.json", v32_1_text)
        self.assertIn("accepted_summary_written=false", v32_1_text)
        self.assertNotIn("--stage one_shot_distill", v32_1_text)
        self.assertNotIn("--stage one_shot_sample", v32_1_text)

    def test_v33_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v33_cases = [
            (
                "train_twophoton_100x100_v33_probe_a_mnist_sample_lowfreq_code_residual_s005_e8.yaml",
                "mnist_sample",
                "none",
                0.05,
            ),
            (
                "train_twophoton_100x100_v33_probe_b_mnist_sample_lowfreq_code_residual_s010_e8.yaml",
                "mnist_sample",
                "none",
                0.10,
            ),
            (
                "train_twophoton_100x100_v33_probe_c_clean_teacher_lowfreq_code_residual_s005_e8.yaml",
                "clean_teacher_camera",
                "none",
                0.05,
            ),
            (
                "train_twophoton_100x100_v33_probe_d_mnist_sample_ols_bridge_code_residual_s005_e8.yaml",
                "mnist_sample",
                "ols_output_amplitude",
                0.05,
            ),
        ]
        for name, target_source, bridge_source, residual_scale in v33_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.generator_template_init, "mnist_prototype")
                self.assertEqual(cfg.one_shot.generator_command_transform, "intensity_to_amplitude")
                self.assertEqual(cfg.one_shot.distill_target_source, target_source)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "lowfreq_digit_code")
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, residual_scale)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 1.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(cfg)

        v33_launcher = config_dir / "run_twophoton_100x100_v33_code_conditioned_target_commands.sh"
        v33_text = v33_launcher.read_text(encoding="utf-8")
        self.assertIn("diagnose_v33_code_conditioned_sample_gap.py", v33_text)
        self.assertIn("code_conditioned_target_redesign", v33_text)
        self.assertIn("accepted_summary_written=false", v33_text)
        self.assertIn("same_batch_code_target_gate_passed", v33_text)
        self.assertIn("--one-shot-optical-checkpoint", v33_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v33_text)
        self.assertIn('if [[ "${same_gate}" == "true" && "${test_pass}" == "true" ]]', v33_text)

    def test_v33_1_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v33_1_cases = [
            (
                "train_twophoton_100x100_v33_1_probe_a_mnist_sample_lowfreq_code_residual_s005_e8.yaml",
                "mnist_sample",
                "none",
                0.05,
            ),
            (
                "train_twophoton_100x100_v33_1_probe_b_mnist_sample_lowfreq_code_residual_s010_e8.yaml",
                "mnist_sample",
                "none",
                0.10,
            ),
            (
                "train_twophoton_100x100_v33_1_probe_c_clean_teacher_lowfreq_code_residual_s005_e8.yaml",
                "clean_teacher_camera",
                "none",
                0.05,
            ),
            (
                "train_twophoton_100x100_v33_1_probe_d_mnist_sample_ols_bridge_code_residual_s005_e8.yaml",
                "mnist_sample",
                "ols_output_amplitude",
                0.05,
            ),
        ]
        for name, target_source, bridge_source, residual_scale in v33_1_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.generator_template_init, "mnist_prototype")
                self.assertEqual(cfg.one_shot.generator_command_transform, "intensity_to_amplitude")
                self.assertEqual(cfg.one_shot.distill_target_source, target_source)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertFalse(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "lowfreq_digit_code")
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, residual_scale)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 1.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v33_1_launcher = config_dir / "run_twophoton_100x100_v33_1_code_conditioned_target_auditfix_commands.sh"
        v33_1_text = v33_1_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v33_1_code_conditioned_target_summary", v33_1_text)
        self.assertIn("code_conditioned_target_redesign_auditfix", v33_1_text)
        self.assertIn("accepted_summary_written=false", v33_1_text)
        self.assertIn("v33_1_probe_a_mnist_sample_lowfreq_code_residual_s005_e8", v33_1_text)
        self.assertIn("v33_1_probe_d_mnist_sample_ols_bridge_code_residual_s005_e8", v33_1_text)
        self.assertNotIn("v33_probe_a_mnist_sample_lowfreq_code_residual_s005_e8", v33_1_text)
        self.assertNotIn("v33_probe_d_mnist_sample_ols_bridge_code_residual_s005_e8", v33_1_text)
        self.assertIn("--one-shot-optical-checkpoint", v33_1_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v33_1_text)
        self.assertIn('if [[ "${same_gate}" == "true" && "${test_pass}" == "true" ]]', v33_1_text)

    def test_v33_2_launcher_contract(self) -> None:
        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v33_2_launcher = config_dir / "run_twophoton_100x100_v33_2_real_ols_candidate_preview_commands.sh"
        v33_2_text = v33_2_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v33_2_real_ols_candidate_preview", v33_2_text)
        self.assertIn("diagnose_v33_2_real_ols_candidate_preview.py", v33_2_text)
        self.assertIn("accepted_summary_written=false", v33_2_text)
        self.assertIn("v33_2_probe_d_mnist_sample_ols_bridge_code_residual_s005_sample", v33_2_text)
        self.assertIn("--stage one_shot_sample", v33_2_text)
        self.assertNotIn("--stage one_shot_distill", v33_2_text)
        self.assertNotIn("--stage one_shot_distill_eval", v33_2_text)
        self.assertIn("--one-shot-checkpoint", v33_2_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v33_2_text)

    def test_v34_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v34_cases = [
            ("sample_twophoton_100x100_v34_scale_s005_code_coupling_recovery.yaml", 0.05),
            ("sample_twophoton_100x100_v34_scale_s010_code_coupling_recovery.yaml", 0.10),
            ("sample_twophoton_100x100_v34_scale_s020_code_coupling_recovery.yaml", 0.20),
            ("sample_twophoton_100x100_v34_scale_s050_code_coupling_recovery.yaml", 0.50),
        ]
        for name, residual_scale in v34_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.distill_target_source, "mnist_sample")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "ols_output_amplitude")
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertFalse(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, residual_scale)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v34_launcher = config_dir / "run_twophoton_100x100_v34_code_coupling_recovery_commands.sh"
        v34_text = v34_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v34_code_coupling_recovery", v34_text)
        self.assertIn("diagnose_v34_code_coupling_recovery.py", v34_text)
        self.assertIn("accepted_summary_written=false", v34_text)
        self.assertIn('local run_name="v34_${label}_probe_d_mnist_sample_ols_bridge_code_residual"', v34_text)
        self.assertIn('run_scale_sample "scale_s005" "0.05"', v34_text)
        self.assertIn('run_scale_sample "scale_s050" "0.50"', v34_text)
        self.assertIn("--stage one_shot_sample", v34_text)
        self.assertNotIn("--stage one_shot_distill", v34_text)
        self.assertNotIn("--stage one_shot_distill_eval", v34_text)
        self.assertIn("--one-shot-checkpoint", v34_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v34_text)

    def test_v35_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v35_scale_cases = [
            ("sample_twophoton_100x100_v35_scale_s100_optical_code_carrier.yaml", 1.00),
            ("sample_twophoton_100x100_v35_scale_s200_optical_code_carrier.yaml", 2.00),
            ("sample_twophoton_100x100_v35_scale_s400_optical_code_carrier.yaml", 4.00),
        ]
        for name, residual_scale in v35_scale_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "ols_output_amplitude")
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, residual_scale)
                self.assertEqual(cfg.one_shot.diagnostic_slm_override_mode, "none")
                self.assertEqual(cfg.one_shot.diagnostic_slm_override_amplitude, 0.0)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v35_oracle_cases = [
            ("sample_twophoton_100x100_v35_oracle_p002_optical_code_carrier.yaml", 0.02),
            ("sample_twophoton_100x100_v35_oracle_p005_optical_code_carrier.yaml", 0.05),
            ("sample_twophoton_100x100_v35_oracle_p010_optical_code_carrier.yaml", 0.10),
        ]
        for name, amplitude in v35_oracle_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "ols_output_amplitude")
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, 0.0)
                self.assertEqual(cfg.one_shot.diagnostic_slm_override_mode, "lowfreq_additive")
                self.assertAlmostEqual(cfg.one_shot.diagnostic_slm_override_amplitude, amplitude)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v35_launcher = config_dir / "run_twophoton_100x100_v35_optical_code_carrier_isolation_commands.sh"
        v35_text = v35_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v35_optical_code_carrier_isolation", v35_text)
        self.assertIn("diagnose_v35_optical_code_carrier_isolation.py", v35_text)
        self.assertIn("accepted_summary_written=false", v35_text)
        self.assertIn('run_sample "scale" "scale_s100" "1.00"', v35_text)
        self.assertIn('run_sample "oracle" "oracle_p010" "0.10"', v35_text)
        self.assertIn("--stage one_shot_sample", v35_text)
        self.assertNotIn("--stage one_shot_distill", v35_text)
        self.assertNotIn("--stage one_shot_distill_eval", v35_text)
        self.assertIn("--one-shot-checkpoint", v35_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v35_text)

    def test_v36_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v36_cases = [
            (
                "train_twophoton_100x100_v36_probe_a_mnist_sample_lowfreq_ols_bridge_code_residual_s200_e8.yaml",
                2.00,
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v36_probe_b_mnist_sample_lowfreq_ols_bridge_code_residual_s400_e8.yaml",
                4.00,
                "lowfreq_digit_code",
            ),
            (
                "train_twophoton_100x100_v36_probe_c_mnist_sample_thick_lowfreq_ols_bridge_code_residual_s200_e8.yaml",
                2.00,
                "thick_lowfreq_digit_code",
            ),
        ]
        for name, residual_scale, target_mode in v36_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.generator_template_init, "mnist_prototype")
                self.assertEqual(cfg.one_shot.generator_command_transform, "intensity_to_amplitude")
                self.assertEqual(cfg.one_shot.distill_target_source, "mnist_sample")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "ols_output_amplitude")
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertFalse(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, residual_scale)
                self.assertEqual(cfg.one_shot.diagnostic_slm_override_mode, "none")
                self.assertEqual(cfg.one_shot.diagnostic_slm_override_amplitude, 0.0)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 1.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v36_launcher = config_dir / "run_twophoton_100x100_v36_high_gain_code_coupled_training_commands.sh"
        v36_text = v36_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v36_high_gain_code_coupled_training", v36_text)
        self.assertIn("diagnose_v36_high_gain_code_coupled_training.py", v36_text)
        self.assertIn("accepted_summary_written=false", v36_text)
        self.assertIn("v36_probe_a_mnist_sample_lowfreq_ols_bridge_code_residual_s200", v36_text)
        self.assertIn("v36_probe_c_mnist_sample_thick_lowfreq_ols_bridge_code_residual_s200", v36_text)
        self.assertIn("--stage one_shot_distill", v36_text)
        self.assertIn("--stage one_shot_distill_eval", v36_text)
        self.assertIn("--stage one_shot_sample", v36_text)
        self.assertIn("--one-shot-optical-checkpoint", v36_text)
        self.assertIn("--one-shot-checkpoint", v36_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v36_text)
        self.assertIn("diagnostic_slm_override", v36_text)

    def test_v37_launcher_contracts(self) -> None:
        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v37_launcher = config_dir / "run_twophoton_100x100_v37_final_cmos_decoder_topology_commands.sh"
        v37_text = v37_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v37_final_cmos_decoder_topology", v37_text)
        self.assertIn("diagnose_v37_final_cmos_decoder_topology.py", v37_text)
        self.assertIn("accepted_summary_written=false", v37_text)
        self.assertIn("--v36-report", v37_text)
        self.assertNotIn("--stage one_shot_distill", v37_text)
        self.assertNotIn("--stage one_shot_distill_eval", v37_text)
        self.assertNotIn("--stage one_shot_sample", v37_text)
        self.assertNotIn("--one-shot-checkpoint", v37_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v37_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v37_text)

        v37_1_launcher = config_dir / "run_twophoton_100x100_v37_1_final_cmos_decoder_topology_commands.sh"
        v37_1_text = v37_1_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v37_1_final_cmos_decoder_topology", v37_1_text)
        self.assertIn("diagnose_v37_final_cmos_decoder_topology.py", v37_1_text)
        self.assertIn("accepted_summary_written=false", v37_1_text)
        self.assertIn("diagnostic_only", v37_1_text)
        self.assertIn("--v36-report", v37_1_text)
        self.assertIn('--version "v37.1"', v37_1_text)
        self.assertNotIn("--stage one_shot_distill", v37_1_text)
        self.assertNotIn("--stage one_shot_distill_eval", v37_1_text)
        self.assertNotIn("--stage one_shot_sample", v37_1_text)
        self.assertNotIn("--one-shot-checkpoint", v37_1_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v37_1_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v37_1_text)

    def test_v38_launcher_contract(self) -> None:
        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v38_launcher = config_dir / "run_twophoton_100x100_v38_formal_cmos_manual_review_commands.sh"
        v38_text = v38_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v38_formal_cmos_manual_review", v38_text)
        self.assertIn("diagnose_v38_formal_cmos_manual_review.py", v38_text)
        self.assertIn("accepted_summary_written=false", v38_text)
        self.assertIn("diagnostic_only", v38_text)
        self.assertIn("--v37-report", v38_text)
        self.assertIn("--template-output", v38_text)
        self.assertIn("--review-dir", v38_text)
        self.assertNotIn("--stage one_shot_distill", v38_text)
        self.assertNotIn("--stage one_shot_distill_eval", v38_text)
        self.assertNotIn("--stage one_shot_sample", v38_text)
        self.assertNotIn("--one-shot-checkpoint", v38_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v38_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v38_text)

    def test_v39_launcher_contract(self) -> None:
        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v39_launcher = config_dir / "run_twophoton_100x100_v39_human_topology_false_positive_commands.sh"
        v39_text = v39_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v39_human_topology_false_positive", v39_text)
        self.assertIn("diagnose_v39_human_topology_false_positive.py", v39_text)
        self.assertIn("accepted_summary_written=false", v39_text)
        self.assertIn("diagnostic_only", v39_text)
        self.assertIn("--v38-report", v39_text)
        self.assertIn("--v37-report", v39_text)
        self.assertIn("--manual-verdict final_cmos_blob_failed", v39_text)
        self.assertNotIn("--stage one_shot_distill", v39_text)
        self.assertNotIn("--stage one_shot_distill_eval", v39_text)
        self.assertNotIn("--stage one_shot_sample", v39_text)
        self.assertNotIn("--one-shot-checkpoint", v39_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v39_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v39_text)

    def test_v40_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v40_cases = [
            (
                "train_twophoton_100x100_v40_probe_a_mnist_sample_sharp_stroke_direct_code_residual_s200_e8.yaml",
                "none",
                "sharp_stroke_digit_code",
                0.0,
            ),
            (
                "train_twophoton_100x100_v40_probe_b_mnist_sample_sharp_stroke_ols_bridge_code_residual_s200_e8.yaml",
                "ols_output_amplitude",
                "sharp_stroke_digit_code",
                0.0,
            ),
            (
                "train_twophoton_100x100_v40_probe_c_mnist_sample_sharp_stroke_ols_bridge_code_residual_s200_div050_e8.yaml",
                "ols_output_amplitude",
                "sharp_stroke_digit_code",
                0.5,
            ),
            (
                "train_twophoton_100x100_v40_probe_d_mnist_sample_thick_lowfreq_ols_bridge_code_residual_s200_div050_e8.yaml",
                "ols_output_amplitude",
                "thick_lowfreq_digit_code",
                0.5,
            ),
        ]
        for name, bridge_source, target_mode, diversity_weight in v40_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.generator_template_init, "mnist_prototype")
                self.assertEqual(cfg.one_shot.generator_command_transform, "intensity_to_amplitude")
                self.assertEqual(cfg.one_shot.distill_target_source, "mnist_sample")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, bridge_source)
                self.assertEqual(cfg.one_shot.target_code_bridge_arch, "deep_residual")
                self.assertFalse(cfg.one_shot.target_code_bridge_trainability_audit)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertAlmostEqual(cfg.one_shot.code_residual_scale, 2.0)
                self.assertAlmostEqual(cfg.one_shot.sample_code_diversity_weight, diversity_weight)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 1.0)
                self.assertEqual(cfg.one_shot.phase_learning_rate, 0.0)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)
                self.assertEqual(cfg.one_shot.d2nn.modulation, "phase_only")

        v40_launcher = config_dir / "run_twophoton_100x100_v40_sample_code_diversity_target_representation_commands.sh"
        v40_text = v40_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v40_sample_code_diversity_target_representation", v40_text)
        self.assertIn("diagnose_v40_sample_code_diversity_target_representation.py", v40_text)
        self.assertIn("accepted_summary_written=false", v40_text)
        self.assertIn("diagnostic_only", v40_text)
        self.assertIn("sample_code_diversity_weight", v40_text)
        self.assertIn("v40_probe_a_mnist_sample_sharp_stroke_direct_code_residual_s200", v40_text)
        self.assertIn("v40_probe_d_mnist_sample_thick_lowfreq_ols_bridge_code_residual_s200_div050", v40_text)
        self.assertIn("--stage one_shot_distill", v40_text)
        self.assertIn("--stage one_shot_distill_eval", v40_text)
        self.assertIn("--stage one_shot_sample", v40_text)
        self.assertIn("--one-shot-optical-checkpoint", v40_text)
        self.assertIn("--one-shot-checkpoint", v40_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v40_text)

    def test_v41_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v41_cases = [
            (
                "train_twophoton_100x100_v41_probe_a_direct_sharp_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                "none",
                2,
                "phase_only",
            ),
            (
                "train_twophoton_100x100_v41_probe_b_direct_sharp_physical_cmos_phase_only_e8.yaml",
                "physical_camera_plane",
                "none",
                2,
                "phase_only",
            ),
            (
                "train_twophoton_100x100_v41_probe_c_direct_sharp_decoder_output_phase_amplitude_e8.yaml",
                "decoder_output_amplitude",
                "phase_amplitude",
                2,
                "phase_only",
            ),
            (
                "train_twophoton_100x100_v41_probe_d_direct_sharp_decoder_output_phase_only_4layer_e8.yaml",
                "decoder_output_amplitude",
                "none",
                4,
                "phase_only",
            ),
        ]
        for name, aux_plane, modulation_override, decoder_layers, modulation in v41_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.generator_template_init, "mnist_prototype")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "sharp_stroke_digit_code")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, aux_plane)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, modulation_override)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertEqual(cfg.learning_rate, 0.0)
                self.assertAlmostEqual(cfg.one_shot.phase_learning_rate, 5.0e-4)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(
                    cfg,
                    decoder_layers=decoder_layers,
                    modulation=modulation,
                )

        v41_launcher = config_dir / "run_twophoton_100x100_v41_decoder_physical_topology_recovery_commands.sh"
        v41_text = v41_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v41_decoder_physical_topology_recovery", v41_text)
        self.assertIn("diagnose_v41_decoder_physical_topology_recovery.py", v41_text)
        self.assertIn("accepted_summary_written=false", v41_text)
        self.assertIn("diagnostic_only", v41_text)
        self.assertIn("decoder_only_training", v41_text)
        self.assertIn("probe_a_direct_sharp_decoder_output_phase_only", v41_text)
        self.assertIn("probe_d_direct_sharp_decoder_output_phase_only_4layer", v41_text)
        self.assertIn("--stage one_shot_distill", v41_text)
        self.assertIn("--stage one_shot_distill_eval", v41_text)
        self.assertIn("--stage one_shot_sample", v41_text)
        self.assertIn("--one-shot-optical-checkpoint", v41_text)
        self.assertIn("--one-shot-checkpoint", v41_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v41_text)

    def test_v42_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v42_cases = [
            (
                "sample_twophoton_100x100_v42_probe_a_bypass_current_relay_psf_s100.yaml",
                "bypass_decoder",
                "relay_psf",
                1.0,
            ),
            (
                "sample_twophoton_100x100_v42_probe_b_bypass_clean_relay_psf_s000.yaml",
                "bypass_decoder",
                "relay_psf",
                0.0,
            ),
            (
                "sample_twophoton_100x100_v42_probe_c_bypass_relay_identity_upper_bound.yaml",
                "bypass_decoder",
                "relay_identity",
                0.0,
            ),
            (
                "sample_twophoton_100x100_v42_probe_d_propagation_only_current_relay_psf_s100.yaml",
                "propagation_only_decoder",
                "relay_psf",
                1.0,
            ),
        ]
        for name, decoder_mode, back_path, sigma in v42_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.mode, "one_shot_sample")
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "sharp_stroke_digit_code")
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, decoder_mode)
                self.assertEqual(cfg.physics.back_path_model, back_path)
                self.assertAlmostEqual(cfg.physics.readout.relay_psf_sigma_px, sigma)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.encoder_layers, 2)
                self.assertEqual(cfg.one_shot.d2nn.decoder_layers, 2)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, 1.50e-3)

        v42_launcher = config_dir / "run_twophoton_100x100_v42_decoder_bypass_backpath_isolation_commands.sh"
        v42_text = v42_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v42_decoder_bypass_backpath_isolation", v42_text)
        self.assertIn("diagnose_v42_decoder_bypass_backpath_isolation.py", v42_text)
        self.assertIn("accepted_summary_written=false", v42_text)
        self.assertIn("diagnostic_only", v42_text)
        self.assertIn("bypass_current", v42_text)
        self.assertIn("propagation_only", v42_text)
        self.assertNotIn("--stage one_shot_distill", v42_text)
        self.assertNotIn("--stage one_shot_distill_eval", v42_text)
        self.assertIn("--stage one_shot_sample", v42_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v42_text)
        self.assertIn("--one-shot-checkpoint", v42_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v42_text)

    def test_v43_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v43_cases = [
            ("sample_twophoton_100x100_v43_spacing_s010_pg1024_sharp.yaml", 0.10e-3, 1024, "sharp_stroke_digit_code"),
            ("sample_twophoton_100x100_v43_spacing_s025_pg1024_sharp.yaml", 0.25e-3, 1024, "sharp_stroke_digit_code"),
            ("sample_twophoton_100x100_v43_spacing_s050_pg1024_sharp.yaml", 0.50e-3, 1024, "sharp_stroke_digit_code"),
            ("sample_twophoton_100x100_v43_spacing_s075_pg1024_sharp.yaml", 0.75e-3, 1024, "sharp_stroke_digit_code"),
            ("sample_twophoton_100x100_v43_spacing_s150_pg1024_sharp.yaml", 1.50e-3, 1024, "sharp_stroke_digit_code"),
            ("sample_twophoton_100x100_v43_aperture_s010_pg1536_sharp.yaml", 0.10e-3, 1536, "sharp_stroke_digit_code"),
            (
                "sample_twophoton_100x100_v43_targetscale_s010_pg1024_small_sharp.yaml",
                0.10e-3,
                1024,
                "small_sharp_stroke_digit_code",
            ),
            (
                "sample_twophoton_100x100_v43_targetscale_s150_pg1024_small_sharp.yaml",
                1.50e-3,
                1024,
                "small_sharp_stroke_digit_code",
            ),
        ]
        for name, spacing, propagation_grid, target_mode in v43_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.mode, "one_shot_sample")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "propagation_only_decoder")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, target_mode)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertEqual(cfg.physics.back_path_model, "relay_psf")
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self.assertIsNotNone(cfg.one_shot.d2nn)
                assert cfg.one_shot.d2nn is not None
                self.assertEqual(cfg.one_shot.d2nn.grid_size, 100)
                self.assertAlmostEqual(cfg.one_shot.d2nn.pixel_size, 2.0e-6)
                self.assertEqual(cfg.one_shot.d2nn.propagation_grid_size, propagation_grid)
                self.assertAlmostEqual(cfg.one_shot.d2nn.layer_spacing, spacing)
                self.assertGreater(cfg.one_shot.d2nn.layer_spacing, 0.0)

        v43_launcher = config_dir / "run_twophoton_100x100_v43_decoder_propagation_geometry_isolation_commands.sh"
        v43_text = v43_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v43_decoder_propagation_geometry_isolation", v43_text)
        self.assertIn("diagnose_v43_decoder_propagation_geometry_isolation.py", v43_text)
        self.assertIn("accepted_summary_written=false", v43_text)
        self.assertIn("diagnostic_only", v43_text)
        self.assertIn("spacing_s010", v43_text)
        self.assertIn("aperture_s010_pg1536", v43_text)
        self.assertIn("targetscale_s150", v43_text)
        self.assertNotIn("--stage one_shot_distill", v43_text)
        self.assertNotIn("--stage one_shot_distill_eval", v43_text)
        self.assertIn("--stage one_shot_sample", v43_text)
        self.assertNotIn("--one-shot-optical-checkpoint", v43_text)
        self.assertIn("--one-shot-checkpoint", v43_text)

    def test_v44_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v44_cases = [
            (
                "train_twophoton_100x100_v44_probe_a_small_sharp_s072_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                0.72,
                1.50e-3,
                "none",
            ),
            (
                "train_twophoton_100x100_v44_probe_b_small_sharp_s060_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                0.60,
                1.50e-3,
                "none",
            ),
            (
                "train_twophoton_100x100_v44_probe_c_small_sharp_s072_physical_cmos_phase_only_e8.yaml",
                "physical_camera_plane",
                0.72,
                1.50e-3,
                "none",
            ),
            (
                "train_twophoton_100x100_v44_probe_d_small_sharp_s072_decoder_output_phase_amplitude_e8.yaml",
                "decoder_output_amplitude",
                0.72,
                1.50e-3,
                "phase_amplitude",
            ),
            (
                "train_twophoton_100x100_v44_probe_e_small_sharp_s072_decoder_output_phase_only_short_spacing_e8.yaml",
                "decoder_output_amplitude",
                0.72,
                0.50e-3,
                "none",
            ),
        ]
        for name, aux_plane, target_scale, spacing, modulation_override in v44_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, target_scale)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, modulation_override)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, aux_plane)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertEqual(cfg.learning_rate, 0.0)
                self.assertAlmostEqual(cfg.one_shot.phase_learning_rate, 5.0e-4)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(cfg, layer_spacing=spacing)

        v44_launcher = config_dir / "run_twophoton_100x100_v44_target_code_spatial_scaling_decoder_recovery_commands.sh"
        v44_text = v44_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v44_target_code_spatial_scaling_decoder_recovery", v44_text)
        self.assertIn("diagnose_v44_target_code_spatial_scaling_decoder_recovery.py", v44_text)
        self.assertIn("accepted_summary_written=false", v44_text)
        self.assertIn("diagnostic_only", v44_text)
        self.assertIn("small_sharp_stroke_digit_code", v44_text)
        self.assertIn("probe_a_small_sharp_s072_decoder_output_phase_only", v44_text)
        self.assertIn("probe_e_small_sharp_s072_decoder_output_phase_only_short_spacing", v44_text)
        self.assertIn("--stage one_shot_distill", v44_text)
        self.assertIn("--stage one_shot_distill_eval", v44_text)
        self.assertIn("--stage one_shot_sample", v44_text)
        self.assertIn("--one-shot-optical-checkpoint", v44_text)
        self.assertIn("--one-shot-checkpoint", v44_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v44_text)

    def test_v45_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v45_cases = [
            (
                "train_twophoton_100x100_v45_probe_a_small_sharp_s072_spacing_s025_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                0.25e-3,
            ),
            (
                "train_twophoton_100x100_v45_probe_b_small_sharp_s072_spacing_s050_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                0.50e-3,
            ),
            (
                "train_twophoton_100x100_v45_probe_c_small_sharp_s072_spacing_s075_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                0.75e-3,
            ),
            (
                "train_twophoton_100x100_v45_probe_d_small_sharp_s072_spacing_s100_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                1.00e-3,
            ),
            (
                "train_twophoton_100x100_v45_probe_e_small_sharp_s072_spacing_s050_physical_cmos_phase_only_e8.yaml",
                "physical_camera_plane",
                0.50e-3,
            ),
        ]
        for name, aux_plane, spacing in v45_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 8)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, "none")
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, aux_plane)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertEqual(cfg.learning_rate, 0.0)
                self.assertAlmostEqual(cfg.one_shot.phase_learning_rate, 5.0e-4)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(cfg, layer_spacing=spacing)
                self.assertGreater(cfg.one_shot.d2nn.layer_spacing, 0.0)

        v45_launcher = config_dir / "run_twophoton_100x100_v45_decoder_spacing_near_field_recovery_commands.sh"
        v45_text = v45_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v45_decoder_spacing_near_field_recovery", v45_text)
        self.assertIn("diagnose_v45_decoder_spacing_near_field_recovery.py", v45_text)
        self.assertIn("accepted_summary_written=false", v45_text)
        self.assertIn("diagnostic_only", v45_text)
        self.assertIn("small_sharp_stroke_digit_code", v45_text)
        self.assertIn("probe_a_small_sharp_s072_spacing_s025_decoder_output_phase_only", v45_text)
        self.assertIn("probe_e_small_sharp_s072_spacing_s050_physical_cmos_phase_only", v45_text)
        self.assertIn("v40_probe_a_direct_sharp", v45_text)
        self.assertNotIn("v44_${label}", v45_text)
        self.assertNotIn("v41_", v45_text)
        self.assertIn("--stage one_shot_distill", v45_text)
        self.assertIn("--stage one_shot_distill_eval", v45_text)
        self.assertIn("--stage one_shot_sample", v45_text)
        self.assertIn("--one-shot-optical-checkpoint", v45_text)
        self.assertIn("--one-shot-checkpoint", v45_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v45_text)

    def test_v46_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v46_cases = [
            (
                "train_twophoton_100x100_v46_probe_a_s025_longer_decoder_output_phase_only_e20.yaml",
                "decoder_output_amplitude",
                20,
                0.0,
                (),
                0.0,
                4.5,
            ),
            (
                "train_twophoton_100x100_v46_probe_b_s025_cross_class_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                8,
                0.02,
                (),
                0.0,
                4.5,
            ),
            (
                "train_twophoton_100x100_v46_probe_c_s025_hard_pair_decoder_output_phase_only_e8.yaml",
                "decoder_output_amplitude",
                8,
                0.0,
                ((3, 5, 0.10), (5, 8, 0.10), (4, 9, 0.10), (7, 9, 0.10), (1, 8, 0.10), (2, 3, 0.10)),
                0.0,
                4.5,
            ),
            (
                "train_twophoton_100x100_v46_probe_d_s025_physical_cmos_cross_class_contrast_phase_only_e8.yaml",
                "physical_camera_plane",
                8,
                0.02,
                (),
                0.01,
                4.5,
            ),
        ]
        for name, aux_plane, epochs, cross_weight, pair_config, contrast_weight, contrast_target in v46_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, epochs)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, "none")
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, aux_plane)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertAlmostEqual(cfg.one_shot.cross_class_pairwise_weight, cross_weight)
                self.assertEqual(cfg.one_shot.student_pair_contrast_config, pair_config)
                self.assertAlmostEqual(cfg.one_shot.camera_readable_contrast_weight, contrast_weight)
                self.assertAlmostEqual(cfg.one_shot.camera_readable_contrast_target, contrast_target)
                self.assertEqual(cfg.learning_rate, 0.0)
                self.assertAlmostEqual(cfg.one_shot.phase_learning_rate, 5.0e-4)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(cfg, layer_spacing=0.25e-3)

        v46_launcher = config_dir / "run_twophoton_100x100_v46_near_field_semantic_separation_commands.sh"
        v46_text = v46_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v46_near_field_semantic_separation", v46_text)
        self.assertIn("diagnose_v46_near_field_semantic_separation.py", v46_text)
        self.assertIn("accepted_summary_written=false", v46_text)
        self.assertIn("diagnostic_only", v46_text)
        self.assertIn("v45_probe_a_checkpoint", v46_text)
        self.assertIn("v40_probe_a_checkpoint", v46_text)
        self.assertIn("probe_a_s025_longer_decoder_output_phase_only", v46_text)
        self.assertIn("probe_d_s025_physical_cmos_cross_class_contrast_phase_only", v46_text)
        self.assertIn("--stage one_shot_distill", v46_text)
        self.assertIn("--stage one_shot_distill_eval", v46_text)
        self.assertIn("--stage one_shot_sample", v46_text)
        self.assertIn("--one-shot-optical-checkpoint", v46_text)
        self.assertIn("--one-shot-checkpoint", v46_text)
        self.assertNotIn("v44_", v46_text)
        self.assertNotIn("v41_", v46_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v46_text)

    def test_v47_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        v47_cases = [
            (
                "train_twophoton_100x100_v47_probe_a_s025_phase_amplitude_decoder_output_e12.yaml",
                "decoder_output_amplitude",
                "phase_amplitude",
                2,
                1024,
            ),
            (
                "train_twophoton_100x100_v47_probe_b_s025_phase_only_4layer_decoder_output_e12.yaml",
                "decoder_output_amplitude",
                "none",
                4,
                1024,
            ),
            (
                "train_twophoton_100x100_v47_probe_c_s025_phase_only_pg1536_decoder_output_e12.yaml",
                "decoder_output_amplitude",
                "none",
                2,
                1536,
            ),
            (
                "train_twophoton_100x100_v47_probe_d_s025_phase_amplitude_physical_cmos_e12.yaml",
                "physical_camera_plane",
                "phase_amplitude",
                2,
                1024,
            ),
        ]
        for name, aux_plane, modulation_override, decoder_layers, propagation_grid_size in v47_cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.one_shot_epochs, 12)
                self.assertEqual(cfg.one_shot.generator_e_arch, "label_template_code_residual")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertTrue(cfg.one_shot.decoder_only_training)
                self.assertTrue(cfg.one_shot.decoder_trainability_audit)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, modulation_override)
                self.assertEqual(cfg.one_shot.aux_topology_plane_key, aux_plane)
                self.assertEqual(cfg.one_shot.generator_base_lr_scale, 0.0)
                self.assertEqual(cfg.one_shot.generator_residual_lr_scale, 0.0)
                self.assertEqual(cfg.learning_rate, 0.0)
                self.assertAlmostEqual(cfg.one_shot.phase_learning_rate, 5.0e-4)
                if modulation_override == "phase_amplitude":
                    self.assertAlmostEqual(cfg.one_shot.amplitude_learning_rate, 5.0e-4)
                self.assertEqual(cfg.physics.readout.primary_plane, "camera")
                self._assert_twophoton_d2nn_contract(
                    cfg,
                    decoder_layers=decoder_layers,
                    layer_spacing=0.25e-3,
                    propagation_grid_size=propagation_grid_size,
                )

        v47_launcher = config_dir / "run_twophoton_100x100_v47_near_field_decoder_capacity_upper_bound_commands.sh"
        v47_text = v47_launcher.read_text(encoding="utf-8")
        self.assertIn("twophoton_100x100_v47_near_field_decoder_capacity_upper_bound", v47_text)
        self.assertIn("diagnose_v47_near_field_decoder_capacity_upper_bound.py", v47_text)
        self.assertIn("accepted_summary_written=false", v47_text)
        self.assertIn("diagnostic_only", v47_text)
        self.assertIn("v40_probe_a_checkpoint", v47_text)
        self.assertIn("V45_REPORT", v47_text)
        self.assertIn("V46_REPORT", v47_text)
        self.assertIn("probe_a_s025_phase_amplitude_decoder_output", v47_text)
        self.assertIn("probe_d_s025_phase_amplitude_physical_cmos", v47_text)
        self.assertIn("--stage one_shot_distill", v47_text)
        self.assertIn("--stage one_shot_distill_eval", v47_text)
        self.assertIn("--stage one_shot_sample", v47_text)
        self.assertIn("--one-shot-optical-checkpoint", v47_text)
        self.assertIn("--one-shot-checkpoint", v47_text)
        self.assertNotIn("v45_probe_a_checkpoint", v47_text)
        self.assertNotIn("v46_probe", v47_text)
        self.assertNotIn("--one-shot-generator-checkpoint", v47_text)

    def test_v48_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            ("sample_twophoton_100x100_v48_t010_transparent_s072.yaml", 0.10e-3),
            ("sample_twophoton_100x100_v48_t025_transparent_s072.yaml", 0.25e-3),
        ]
        for name, spacing in cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.mode, "one_shot_sample")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "propagation_only_decoder")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, "phase_amplitude")
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.target_code_bridge_source, "none")
                self.assertEqual(cfg.one_shot.sample_class_labels, tuple(range(10)))
                self.assertEqual(cfg.one_shot.sample_repeats_per_class, 2)
                self.assertEqual(cfg.physics.back_path_model, "relay_psf")
                self._assert_twophoton_d2nn_contract(
                    cfg,
                    layer_spacing=spacing,
                    propagation_grid_size=1024,
                )

        launcher = config_dir / "run_twophoton_100x100_v48_transparent_decoder_control_commands.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("diagnose_v48_transparent_decoder_control.py", text)
        self.assertIn("accepted_summary_written=false", text)
        self.assertIn("--stage one_shot_sample", text)
        self.assertIn("--device cuda", text)
        self.assertIn("--one-shot-checkpoint", text)
        self.assertIn("T010", text)
        self.assertIn("T025", text)
        self.assertIn("M025", text)
        self.assertNotIn("--stage one_shot_distill ", text)
        self.assertNotIn("--stage one_shot_distill_eval", text)

    def test_v49_decoder_mask_component_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            ("sample_twophoton_100x100_v49_p025_phase_only_s072.yaml", "phase_only"),
            ("sample_twophoton_100x100_v49_a025_amplitude_only_s072.yaml", "amplitude_only"),
        ]
        for name, component in cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.mode, "one_shot_sample")
                self.assertEqual(cfg.seed, 1234)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, "phase_amplitude")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mask_component, component)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.sample_class_labels, tuple(range(10)))
                self.assertEqual(cfg.one_shot.sample_repeats_per_class, 2)
                self.assertEqual(cfg.physics.back_path_model, "relay_psf")
                self._assert_twophoton_d2nn_contract(
                    cfg,
                    layer_spacing=0.25e-3,
                    propagation_grid_size=1024,
                )

        launcher = config_dir / "run_twophoton_100x100_v49_decoder_mask_component_attribution_commands.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("tools.diagnose_v49_decoder_mask_component_attribution", text)
        self.assertIn("decoder-mask-induced-artifact", text)
        self.assertEqual(text.count("--stage one_shot_sample"), 2)
        self.assertEqual(text.count("--device cuda"), 2)
        self.assertIn("P025", text)
        self.assertIn("A025", text)
        self.assertIn("T025", text)
        self.assertIn("M025", text)
        self.assertNotIn("--stage one_shot_distill ", text)
        self.assertNotIn("--stage one_shot_distill_eval", text)

    def test_v50_decoder_phase_layer_config_and_launcher_contracts(self) -> None:
        from ldm_project.config import load_training_config

        config_dir = Path("outputs/reports/twophoton_100x100_probe_configs")
        cases = [
            ("sample_twophoton_100x100_v50_l1p025_phase_only_s072.yaml", (True, False)),
            ("sample_twophoton_100x100_v50_l2p025_phase_only_s072.yaml", (False, True)),
        ]
        for name, layer_mask in cases:
            with self.subTest(config=name):
                cfg = load_training_config(config_dir / name)
                self.assertEqual(cfg.mode, "one_shot_sample")
                self.assertEqual(cfg.seed, 1234)
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mode, "normal")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_modulation_override, "phase_amplitude")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_mask_component, "phase_only")
                self.assertEqual(cfg.one_shot.decoder_diagnostic_phase_layer_mask, layer_mask)
                self.assertEqual(cfg.one_shot.target_code_diagnostic_mode, "small_sharp_stroke_digit_code")
                self.assertAlmostEqual(cfg.one_shot.target_code_spatial_scale, 0.72)
                self.assertEqual(cfg.one_shot.sample_class_labels, tuple(range(10)))
                self.assertEqual(cfg.one_shot.sample_repeats_per_class, 2)
                self._assert_twophoton_d2nn_contract(
                    cfg,
                    layer_spacing=0.25e-3,
                    propagation_grid_size=1024,
                )

        launcher = config_dir / "run_twophoton_100x100_v50_decoder_phase_layer_attribution_commands.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("tools.diagnose_v50_decoder_phase_layer_attribution", text)
        self.assertIn("phase-mask-induced-artifact", text)
        self.assertEqual(text.count("--stage one_shot_sample"), 2)
        self.assertEqual(text.count("--device cuda"), 2)
        self.assertIn("T025", text)
        self.assertIn("L1P025", text)
        self.assertIn("L2P025", text)
        self.assertIn("P025", text)
        self.assertNotIn("--stage one_shot_distill ", text)
        self.assertNotIn("--stage one_shot_distill_eval", text)
        self.assertNotIn("checkpoint_builder", text)


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

    def test_validate_stage_requirements_enforces_sample_checkpoints(self) -> None:
        validate_stage_requirements = load_train_utils_module().validate_stage_requirements

        validate_stage_requirements("sample", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_sample", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_distill_export", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_diffusion_latent_audit", "first.pt", "diffusion.pt")
        validate_stage_requirements("teacher_first_stage_oracle_audit", "first.pt")
        validate_stage_requirements("teacher_distill_filter", None, None, None, "distill.pt")
        validate_stage_requirements("teacher_distill_label_audit", None, None, None, "distill.pt")
        validate_stage_requirements("teacher_distill_label_strict_filter", None, None, None, "distill.pt")
        validate_stage_requirements("teacher_distill_readable_split", None, None, None, "distill.pt")
        validate_stage_requirements("one_shot_distill", "first.pt", "diffusion.pt", None, "distill.pt")
        validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", "one_shot.pt", "distill.pt")
        validate_stage_requirements("one_shot_distill_eval_diagnostics", None, None, None, None)
        validate_stage_requirements("one_shot_optical_identity_audit", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "first-stage-checkpoint"):
            validate_stage_requirements("sample", None, "diffusion.pt")
        with self.assertRaisesRegex(ValueError, "diffusion-checkpoint"):
            validate_stage_requirements("sample", "first.pt", None)
        with self.assertRaisesRegex(ValueError, "first-stage-checkpoint"):
            validate_stage_requirements("teacher_first_stage_oracle_audit", None)
        with self.assertRaisesRegex(ValueError, "diffusion-checkpoint"):
            validate_stage_requirements("teacher_diffusion_latent_audit", "first.pt", None)
        with self.assertRaisesRegex(ValueError, "one-shot-checkpoint"):
            validate_stage_requirements("one_shot_sample", "first.pt", "diffusion.pt", None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("one_shot_distill", "first.pt", "diffusion.pt", None, None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("teacher_distill_filter", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("teacher_distill_label_audit", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("teacher_distill_label_strict_filter", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("teacher_distill_readable_split", None, None, None, None)
        with self.assertRaisesRegex(ValueError, "one-shot-checkpoint"):
            validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", None, "distill.pt")
        with self.assertRaisesRegex(ValueError, "distill-dataset"):
            validate_stage_requirements("one_shot_distill_eval", "first.pt", "diffusion.pt", "one_shot.pt", None)

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

        def capture_save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
            grid_calls.append((Path(destination).name, max_items, list(tensor_map.keys()), dict(kwargs)))
            return original_save_tensor_grid(
                self,
                tensor_map,
                destination,
                title=title,
                max_items=max_items,
                **kwargs,
            )

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
            seed=77,
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
        self.assertEqual(meta["output_version"], "unversioned")
        self.assertEqual(meta["output_root"], tmpdir)
        self.assertTrue(meta["output_root_explicit"])
        self.assertEqual(meta["training_seed"], 77)
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
        self.assertEqual(meta["sample_repeats_per_class"], 2)
        self.assertEqual(meta["one_shot_sample_class_labels"], [])
        self.assertEqual(meta["one_shot_sample_repeats_per_class"], 2)
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
        expected_visual_labels = [label for label in range(10) for _ in range(2)]
        self.assertEqual(first_stage_visual_config["class_labels"], expected_visual_labels)
        self.assertEqual(first_stage_visual_config["reference_source"], "validation_then_train_fallback")
        self.assertEqual(diffusion_visual_config["class_labels"], expected_visual_labels)
        self.assertEqual(diffusion_visual_config["reference_labels"], expected_visual_labels)
        self.assertEqual(diffusion_visual_config["reference_source"], "validation_then_train_fallback")
        diffusion_epoch_grids = [
            call for call in grid_calls
            if call[0] == "epoch_0001_intensity.png" and "reference image" in call[2]
        ]
        self.assertEqual(len(diffusion_epoch_grids), 1)
        self.assertEqual(diffusion_epoch_grids[0][1], 20)
        self.assertEqual(diffusion_epoch_grids[0][3]["row_labels"], [str(label) for label in expected_visual_labels])
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
        self.assertEqual(post_train_config["class_labels"], expected_visual_labels)
        self.assertEqual(post_train_config["reference_labels"], expected_visual_labels)
        self.assertEqual(post_train_config["reference_source"], "validation_then_train_fallback")
        self.assertIn("reference_images", post_train_tensors)
        self.assertIn("reference_labels", post_train_tensors)
        self.assertIn("generated_camera", post_train_tensors)
        self.assertIn("generated_image", post_train_tensors)
        self.assertIn("chip_amplitude", post_train_tensors)
        self.assertIn("chip_phase", post_train_tensors)
        self.assertEqual(tuple(post_train_tensors["reference_images"].shape), (20, 1, 8, 8))
        self.assertEqual(post_train_tensors["reference_labels"].tolist(), expected_visual_labels)
        self.assertEqual(post_train_tensors["reference_source"], "validation_then_train_fallback")
        self.assertEqual(tuple(post_train_tensors["generated_camera"].shape), (20, 1, 8, 8))
        self.assertEqual(tuple(post_train_tensors["generated_image"].shape), (20, 1, 8, 8))
        self.assertEqual(trace_config["reference_labels"], expected_visual_labels)
        self.assertEqual(sample_metrics["overall"]["count"], 20)
        self.assertEqual(sorted(sample_metrics["per_class"].keys()), [str(label) for label in range(10)])
        self.assertEqual(sample_metrics["nearest_neighbor"]["reference_source"], "validation_then_train_fallback")
        self.assertEqual(sample_metrics["nearest_neighbor"]["overall"]["count"], 20)
        self.assertEqual(
            sorted(sample_metrics["nearest_neighbor"]["per_class"].keys()),
            [str(label) for label in range(10)],
        )
        self.assertEqual(trace_config["trace_columns"], ["reference", "noise", "t750", "t000"])
        self.assertEqual([step["label"] for step in trace_config["trace_steps"]], ["t750", "t000"])
        self.assertEqual(trace_tensors["reference_labels"].tolist(), expected_visual_labels)
        self.assertEqual(list(trace_tensors["latent_trace"].keys()), ["noise", "t750", "t000"])
        self.assertEqual(list(trace_tensors["camera_trace"].keys()), ["noise", "t750", "t000"])
        for camera_trace in trace_tensors["camera_trace"].values():
            self.assertEqual(tuple(camera_trace.shape), (20, 1, 8, 8))
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
        self.assertEqual(meta["output_version"], "unversioned")
        self.assertEqual(meta["output_root"], str(output_root))
        self.assertTrue(meta["output_root_explicit"])
        self.assertEqual(meta["planned_first_stage_epochs"], 0)
        self.assertEqual(meta["planned_diffusion_epochs"], 0)
        self.assertEqual(meta["diffusion_checkpoint"], str(diffusion_checkpoint))
        self.assertAlmostEqual(meta["diffusion_guidance_scale"], 3.5, places=6)
        self.assertEqual(meta["sample_class_labels"], [3, 4, 5])
        self.assertEqual(meta["sample_repeats_per_class"], 4)
        self.assertEqual(meta["one_shot_sample_class_labels"], [3, 4, 5])
        self.assertEqual(meta["one_shot_sample_repeats_per_class"], 4)
        self.assertEqual(meta["sample_metric_source"], "artifacts/post_train_sample_metrics.json")
        self.assertEqual(meta["latent_scale_source"], "diffusion_checkpoint")
        self.assertTrue(meta["diffusion_best_loaded_for_sampling"])
        self.assertFalse(history_exists)
        expected_sample_labels = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        self.assertEqual(post_train_config["class_labels"], expected_sample_labels)
        self.assertEqual(post_train_config["reference_labels"], expected_sample_labels)
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
