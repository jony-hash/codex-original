from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v61_offline_decodability.py"


def _module():
    spec = importlib.util.spec_from_file_location("v61_decodability", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_and_bootstrap_controls_are_well_defined():
    module = _module()
    image = torch.rand(4, 1, 200, 200)
    summary = module._summary(image, image)
    assert summary["corr"] > 0.999
    assert summary["dice"] == 1.0
    assert module.paired_bootstrap_delta([1.0, 2.0], [0.0, 1.0], seed=1, iterations=32)["ci_low"] > 0.0


def test_candidate_plane_uses_boundary_or_chip_camera():
    module = _module()
    no_boundary = {"decision": {}, "planes": {"test": {"slm_command": {}, "chip": {}, "camera_intensity": {}}}}
    assert module._candidate_planes(no_boundary) == ("chip", "camera_intensity")
    boundary = {"decision": {"first_plane": "chip"}, "planes": {"test": {"slm_command": {}, "chip": {}, "camera_intensity": {}}}}
    assert module._candidate_planes(boundary) == ("slm_command", "chip")


def test_code_inputs_do_not_expand_the_feature_dimension():
    module = _module()
    split = {"codes": torch.zeros(2, 128), "labels": torch.tensor([0, 1]), "planes": {"slm_command": torch.zeros(2, 1, 200, 200)}}
    assert module._input_from_cache(split, "slm_command", "code_only").shape == (2, 128)
    assert module._input_from_cache(split, "slm_command", "code_label").shape == (2, 138)
