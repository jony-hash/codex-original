# ruff: noqa: F401,F403,F405,F811
"""Shared test fixtures and tool loaders for classified suites."""
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

def _cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for one-shot optical student smoke tests.")
    return torch.device("cuda")


def _dummy_one_shot_outputs(batch_size: int, image_size: int, grid_size: int):
    import torch

    camera = torch.linspace(0.05, 0.95, steps=batch_size * image_size * image_size).reshape(
        batch_size, 1, image_size, image_size
    )
    full_phase = torch.linspace(-torch.pi, torch.pi, steps=camera.numel()).reshape_as(camera)
    grid = torch.linspace(0.0, 2.0, steps=batch_size * grid_size * grid_size).reshape(
        batch_size, 1, grid_size, grid_size
    )
    grid_phase = torch.linspace(-torch.pi, torch.pi, steps=grid.numel()).reshape_as(grid)
    raw_camera = torch.linspace(0.1, 1.1, steps=batch_size * grid_size * grid_size).reshape(
        batch_size, 1, grid_size, grid_size
    )
    layer_energy = torch.ones(batch_size, 2, dtype=torch.float32)
    return {
        "camera_plane": camera,
        "slm_command": camera.clamp(0.0, 1.0),
        "slm_amplitude": camera.clamp(0.0, 1.0),
        "slm_phase": torch.zeros_like(camera),
        "encoder_input_amplitude": camera * 2.0,
        "encoder_input_phase": full_phase,
        "encoder_input_grid_amplitude": grid * 0.8,
        "encoder_input_grid_phase": grid_phase,
        "encoded_amplitude": grid,
        "encoded_phase": grid_phase,
        "ols_input_amplitude": grid,
        "ols_input_phase": grid_phase,
        "ols_output_amplitude": grid * 0.5,
        "ols_output_phase": -grid_phase,
        "ols_amplitude": grid * 0.5,
        "ols_phase": -grid_phase,
        "decoder_input_amplitude": grid * 0.5,
        "decoder_input_phase": -grid_phase,
        "decoder_output_amplitude": grid * 1.25,
        "decoder_output_phase": grid_phase.roll(shifts=1, dims=-1),
        "chip_amplitude": grid * 3.0,
        "chip_phase": grid_phase,
        "camera_phase": grid_phase,
        "diffractive_camera_intensity_raw": raw_camera,
        "diffractive_camera_plane_raw": raw_camera.clamp(0.0, 1.0),
        "diffractive_camera_plane_loss_raw": raw_camera,
        "diffractive_encoder_layer_amplitude": grid.unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_encoder_layer_phase": grid_phase.unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_decoder_layer_amplitude": (grid * 1.1).unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_decoder_layer_phase": grid_phase.neg().unsqueeze(1).expand(batch_size, 2, 1, grid_size, grid_size),
        "diffractive_encoder_mask_amplitude": grid.new_ones(2, 1, grid_size, grid_size),
        "diffractive_encoder_mask_phase": grid_phase[:2],
        "diffractive_decoder_mask_amplitude": grid.new_ones(2, 1, grid_size, grid_size),
        "diffractive_decoder_mask_phase": -grid_phase[:2],
        "diffractive_encoder_crop_input_energy": layer_energy * 4.0,
        "diffractive_encoder_crop_padded_input_energy": layer_energy * 4.0,
        "diffractive_encoder_crop_propagated_full_energy": layer_energy * 3.0,
        "diffractive_encoder_crop_cropped_energy": layer_energy,
        "diffractive_encoder_crop_crop_retention": layer_energy * 0.25,
        "diffractive_encoder_crop_propagation_energy_ratio": layer_energy * 0.75,
        "diffractive_decoder_crop_input_energy": layer_energy * 2.0,
        "diffractive_decoder_crop_padded_input_energy": layer_energy * 2.0,
        "diffractive_decoder_crop_propagated_full_energy": layer_energy * 1.5,
        "diffractive_decoder_crop_cropped_energy": layer_energy * 0.5,
        "diffractive_decoder_crop_crop_retention": layer_energy * 0.25,
        "diffractive_decoder_crop_propagation_energy_ratio": layer_energy * 0.75,
    }


def _assert_nonblank_phase_columns_are_labeled(test_case, tensor_map: dict[str, object]) -> None:
    import torch

    for label, value in tensor_map.items():
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().float()
        if torch.isnan(tensor).all().item():
            continue
        test_case.assertIn("phase", label, f"nonblank phase column must say phase: {label}")



@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")

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
