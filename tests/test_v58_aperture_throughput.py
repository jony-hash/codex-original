from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ldm_project.one_shot_optical_layers import PhaseOnlyDiffractiveStack


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v58_aperture_throughput_upper_bound.py"


def _module():
    spec = importlib.util.spec_from_file_location("v58_aperture", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_interlayer_crop_stack_retains_full_propagation_grid():
    module = _module()
    source = PhaseOnlyDiffractiveStack(
        num_layers=2, grid_size=4, layer_spacing=1e-4, wavelength=532e-9,
        pixel_size=2e-6, propagation_grid_size=8, phase_init="small",
    )
    stack = module._NoInterlayerCropStack(source)
    field = torch.ones(2, 1, 4, 4, dtype=torch.complex64)
    output, layers, diagnostics = stack(field)
    assert output.shape == (2, 1, 8, 8)
    assert layers.shape == (2, 2, 1, 8, 8)
    assert torch.allclose(diagnostics["crop_retention"], torch.ones(2, 2))
    assert torch.all(diagnostics["propagation_energy_ratio"] > 0.999)


def test_v58_requires_balanced_saved_sample():
    module = _module()
    payload = {"codes": torch.zeros(20, 4), "class_labels": torch.arange(10).repeat_interleave(2)}
    codes, labels = module._validate_sample(payload)
    assert codes.shape == (20, 4)
    assert labels.tolist()[:4] == [0, 0, 1, 1]
