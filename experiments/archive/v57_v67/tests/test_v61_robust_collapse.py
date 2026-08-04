from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v61_robust_collapse.py"


def _module():
    spec = importlib.util.spec_from_file_location("v61_robust", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_ci_and_relative_retention_are_deterministic():
    module = _module()
    assert module.bootstrap_ci([1.0, 2.0, 3.0], iterations=32, seed=7) == module.bootstrap_ci([1.0, 2.0, 3.0], iterations=32, seed=7)
    summary = {"slm_command": {"separation": {name: {"signal": 1.0, "signal_ci_low": 0.8, "signal_ci_high": 1.2} for name in module.SEPARATION_METRICS}}, "camera_field": {"separation": {name: {"signal": 0.01, "signal_ci_low": 0.0, "signal_ci_high": 0.02} for name in module.SEPARATION_METRICS}}}
    module._add_relative_retention(summary)
    assert summary["camera_field"]["relative_retention"]["complex"]["ci_high"] == 0.02


def test_robust_decision_requires_both_splits_and_two_metrics():
    module = _module()
    def plane(retention: float):
        return {"relative_retention": {name: {"ci_low": retention, "ci_high": retention} for name in module.SEPARATION_METRICS}}
    val = {"slm_command": plane(1.0), "chip": plane(0.2), "camera_field": plane(0.05)}
    test = {"slm_command": plane(1.0), "chip": plane(0.2), "camera_field": plane(0.05)}
    assert module.decision(val, test)["decision"] == "robust-localized-collapse"
    test["camera_field"] = plane(0.2)
    assert module.decision(val, test)["decision"] == "no-robust-localized-collapse"


def test_complex_distance_preserves_imaginary_component():
    module = _module()
    left = torch.tensor([[1.0 + 0.0j]])
    right = torch.tensor([[0.0 + 1.0j]])
    assert module._distance(left, right)["complex"] < 1e-6  # global phase is removed
