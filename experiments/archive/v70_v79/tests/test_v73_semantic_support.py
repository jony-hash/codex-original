from __future__ import annotations

import json

import torch

from tools.audit_v73_semantic_support import (
    calibrate_semantic_support_amplitude,
    semantic_support_pattern,
)
from tools.audit_v70_slm_transport import build_lowfreq_code_pattern
from tools.decide_v73_semantic_support import decide
from tools.v68_b_acceptance_utils import balanced_factor_decomposition


def test_semantic_support_is_centered_and_deterministic() -> None:
    generator = torch.Generator().manual_seed(1234)
    codes = torch.randn((100, 128), generator=generator)
    prototypes = torch.rand((10, 1, 200, 200), generator=generator)
    pattern_a, support_a = semantic_support_pattern(codes, prototypes)
    pattern_b, support_b = semantic_support_pattern(codes, prototypes)
    assert torch.equal(pattern_a, pattern_b)
    assert torch.equal(support_a, support_b)
    assert torch.isfinite(pattern_a).all()
    assert torch.allclose(pattern_a.flatten(2).sum(2), torch.zeros((100, 10)), atol=1e-3)
    assert float(pattern_a.flatten(2).abs().amax(2).max()) <= 1.0 + 1e-6


def test_physical_intensity_bisection_hits_target() -> None:
    generator = torch.Generator().manual_seed(42)
    base = torch.rand((100, 10, 1, 16, 16), generator=generator) * 0.5 + 0.2
    codes = torch.randn((100, 128), generator=generator)
    pattern = build_lowfreq_code_pattern(codes, size=16).unsqueeze(1).expand(100, 10, 1, 16, 16)
    result = calibrate_semantic_support_amplitude(base, pattern, iterations=30)
    assert result["finite"]
    assert result["absolute_error"] <= 1e-4
    command = (base + result["amplitude"] * pattern).clamp(0.0, 1.0)
    intensity = (0.05 + 0.95 * command).square()
    assert abs(balanced_factor_decomposition(intensity)[0]["code_fraction"] - 0.25) <= 1e-4


def _case(tmp_path, *, factor: float, code_ss: float, label_ss: float, readable: list[int], valid: bool = True) -> dict:
    tmp_path.mkdir()
    v72 = {"status": "phase-readability-code-tradeoff"}
    v72_report = {"status": "v72-phase-attribution-complete"}
    admission = {"status": "v73-cpu-admission-passed"}
    metrics = {"clean_prototype_corr": 0.9, "clean_prototype_dice": 0.9, "prototype_accuracy": 1.0, "high_saturation": 0.01}
    report = {
        "status": "v73-semantic-support-audit-complete",
        "diagnostic_only": True,
        "test_split_used": False,
        "return_optical_fields": False,
        "full_field_saved": False,
        "active_view_only": True,
        "batch_size": 1,
        "finite": valid,
        "energy_pass": valid,
        "parameters_unchanged": valid,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "teacher_alignment_computed": False,
        "calibration": {"finite": True, "absolute_error": 1e-6},
        "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
        "camera_code_factor": factor,
        "camera_code_effect_ss": code_ss,
        "camera_label_effect_ss": label_ss,
        "v72_baseline_camera_code_effect_ss": 100.0,
        "v72_baseline_camera_label_effect_ss": 100.0,
        "absolute_code_ss_pass": code_ss >= 100.0,
        "label_ss_retention": label_ss / 100.0,
        "label_ss_retention_pass": label_ss >= 90.0,
        "intensity_code_factors": {"camera": factor},
        "first_intensity_plane_below_25_percent": "camera",
        "crossed_metrics": metrics,
        "diagonal_metrics": metrics,
    }
    manual = {"readable_labels": readable}
    paths = {}
    for name, value in (("v72", v72), ("v72_report", v72_report), ("admission", admission), ("report", report), ("manual", manual)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return decide(
        v72_decision=paths["v72"], v72_report=paths["v72_report"], cpu_admission=paths["admission"],
        audit_report=paths["report"], manual_review=paths["manual"], output=tmp_path / "decision.json",
    )


def test_decision_branches(tmp_path) -> None:
    all_labels = list(range(10))
    assert _case(tmp_path / "capable", factor=0.30, code_ss=120.0, label_ss=100.0, readable=all_labels)["status"] == "semantic-support-code-transport-capable"
    assert _case(tmp_path / "ratio", factor=0.30, code_ss=80.0, label_ss=50.0, readable=all_labels)["status"] == "semantic-support-factor-ratio-artifact"
    assert _case(tmp_path / "readability", factor=0.30, code_ss=120.0, label_ss=100.0, readable=[0, 1])["status"] == "semantic-support-readability-tradeoff"
    assert _case(tmp_path / "attenuated", factor=0.20, code_ss=120.0, label_ss=100.0, readable=all_labels)["status"] == "semantic-support-code-factor-attenuated"
    assert _case(tmp_path / "invalid", factor=0.30, code_ss=120.0, label_ss=100.0, readable=all_labels, valid=False)["status"] == "harness-invalid"
