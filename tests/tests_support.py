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


def load_v40_sample_code_diversity_target_representation_tool():
    return _load_tool_module(
        "diagnose_v40_sample_code_diversity_target_representation",
        "diagnose_v40_sample_code_diversity_target_representation.py",
    )
