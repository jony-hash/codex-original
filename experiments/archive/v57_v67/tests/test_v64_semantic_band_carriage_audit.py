from __future__ import annotations

import importlib.util

import pytest
import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v64_semantic_band_carriage_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("v64_semantic_band", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sinusoid(cycles_per_mm: float) -> torch.Tensor:
    pixels = torch.arange(200, dtype=torch.float32)
    phase = 2.0 * torch.pi * cycles_per_mm * (pixels * 2.5e-3)
    image = torch.sin(phase)[None, :].expand(200, -1)
    return image[None, None]


def test_native_frequency_masks_are_disjoint_and_complete():
    module = _module()
    masks = module.frequency_band_masks()
    total = sum(mask.int() for mask in masks.values())
    assert int(total.min()) == 1
    assert int(total.max()) == 1
    assert sum(int(mask.sum()) for mask in masks.values()) == 200 * 200


def test_physical_cutoff_routes_low_and_high_synthetic_carriers():
    module = _module()
    masks = module.frequency_band_masks()
    low = module.apply_frequency_mask(_sinusoid(10.0), masks["semantic_0_20_cycles_mm"])
    high = module.apply_frequency_mask(_sinusoid(120.0), masks["semantic_0_20_cycles_mm"])
    assert float(low.square().mean()) > 0.40
    assert float(high.square().mean()) < 1.0e-5


def test_three_native_bands_reconstruct_the_original_intensity():
    module = _module()
    masks = module.frequency_band_masks()
    values = torch.randn(2, 1, 200, 200)
    rebuilt = sum(module.apply_frequency_mask(values, mask) for mask in masks.values()) - 2.0 * values.mean((-2, -1), keepdim=True)
    assert torch.allclose(rebuilt, values, atol=2.0e-5, rtol=2.0e-5)


def test_teacher_cutoff_reports_semantic_energy_and_failure():
    module = _module()
    masks = module.frequency_band_masks()
    low_report = module.teacher_cutoff_report(_sinusoid(10.0), masks)
    high_report = module.teacher_cutoff_report(_sinusoid(120.0), masks)
    assert low_report["cutoff_valid"] is True
    assert high_report["cutoff_valid"] is False


def test_spectral_energy_fractions_are_normalized():
    module = _module()
    masks = module.frequency_band_masks()
    fractions = module.spectral_energy_fractions(torch.randn(4, 1, 200, 200), masks)
    assert sum(fractions.values()) == pytest.approx(1.0, abs=1.0e-6)


def test_decision_covers_semantic_out_of_band_and_distributed_branches():
    module = _module()
    semantic = module.decide(cutoff_valid=True, code_low_fraction=0.7, semantic_probe_retained=True, nonsemantic_probe_retained=False)
    assert semantic["status"] == "semantic-band-carrier-retained"
    out_of_band = module.decide(cutoff_valid=True, code_low_fraction=0.2, semantic_probe_retained=False, nonsemantic_probe_retained=True)
    assert out_of_band["status"] == "out-of-semantic-band-subvisual-carrier-candidate"
    distributed = module.decide(cutoff_valid=True, code_low_fraction=0.7, semantic_probe_retained=True, nonsemantic_probe_retained=True)
    assert distributed["status"] == "distributed-subvisual-carrier"
    distributed_without_low_carrier = module.decide(cutoff_valid=True, code_low_fraction=0.2, semantic_probe_retained=True, nonsemantic_probe_retained=True)
    assert distributed_without_low_carrier["status"] == "distributed-subvisual-carrier"
    invalid = module.decide(cutoff_valid=False, code_low_fraction=0.7, semantic_probe_retained=True, nonsemantic_probe_retained=True)
    assert invalid["status"] == "harness-invalid"


def test_probe_retention_requires_both_metrics_and_positive_rotation_ci():
    module = _module()
    baseline = {"target_corr_mean": 0.9, "thresholded_dice_mean": 0.8}
    retained = {"test": {"target_corr_mean": 0.82, "thresholded_dice_mean": 0.73}, "paired_rotation": {"ci_low_positive": True}}
    assert module._probe_retained(retained, baseline) is True
    retained["paired_rotation"]["ci_low_positive"] = False
    assert module._probe_retained(retained, baseline) is False


def test_tool_has_no_formal_resize_or_training_path():
    source = TOOL.read_text(encoding="utf-8").lower()
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "grid_sample" not in source
    assert "pca" not in source
    assert "optimizer" not in source.replace("optimizer_created", "")
