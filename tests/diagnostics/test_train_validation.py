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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    active_tools_dir = PROJECT_ROOT / "tools"
    tools_dir = PROJECT_ROOT / "experiments" / "archive" / "pre_v57" / "tools"
    if "tools" not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            "tools",
            active_tools_dir / "__init__.py",
            submodule_search_locations=[str(tools_dir), str(active_tools_dir)],
        )
        if package_spec is None or package_spec.loader is None:
            raise RuntimeError(f"could not load tools package: {tools_dir}")
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["tools"] = package
        package_spec.loader.exec_module(package)
    else:
        package_paths = sys.modules["tools"].__path__
        for directory in (tools_dir, active_tools_dir):
            if str(directory) not in package_paths:
                package_paths.insert(0, str(directory))

    tool_path = tools_dir / filename
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
