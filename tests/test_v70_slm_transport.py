from __future__ import annotations

import json

import torch

from tools.audit_v70_slm_transport import build_lowfreq_code_pattern, calibrate_override_amplitude
from tools.decide_v70_slm_transport import decide


def test_lowfreq_pattern_is_deterministic_and_code_conditioned() -> None:
    codes = torch.arange(200, dtype=torch.float32).reshape(2, 100)
    first = build_lowfreq_code_pattern(codes, size=16)
    second = build_lowfreq_code_pattern(codes, size=16)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert torch.isfinite(first).all()


def test_bisection_hits_balanced_slm_target() -> None:
    generator = torch.Generator().manual_seed(1234)
    codes = torch.randn(100, 100, generator=generator)
    pattern = build_lowfreq_code_pattern(codes, size=16).unsqueeze(1).expand(100, 10, 1, 16, 16)
    base = torch.rand(pattern.shape, generator=generator) * 0.6 + 0.2
    result = calibrate_override_amplitude(base, pattern, iterations=30)
    assert result["finite"]
    assert result["absolute_error"] <= 1e-4
    assert 0.0 <= result["zero_fraction"] <= 1.0
    assert 0.0 <= result["one_fraction"] <= 1.0


def _write_case(tmp_path, camera: float, readable: list[int], *, valid: bool = True) -> dict:
    prior = {"status": "v69-prototype-only"}
    crossed = {"status": "v69-crossed-audit-complete"}
    metrics = {
        "clean_prototype_corr": 0.9,
        "clean_prototype_dice": 0.9,
        "prototype_accuracy": 1.0,
        "high_saturation": 0.01,
    }
    report = {
        "status": "v70-slm-intervention-complete",
        "diagnostic_only": True,
        "test_split_used": False,
        "return_optical_fields": False,
        "batch_size": 1,
        "parameters_unchanged": True,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "finite": True,
        "energy_pass": True,
        "calibration": {"finite": True, "absolute_error": 1e-6},
        "slm_code_factor": 0.25,
        "camera_code_factor": camera,
        "crossed_metrics": metrics,
        "diagonal_metrics": metrics,
        "first_plane_below_25_percent": "camera",
    }
    if not valid:
        report["parameters_unchanged"] = False
    paths = {}
    for name, value in (("prior", prior), ("crossed", crossed), ("report", report), ("manual", {"readable_labels": readable})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    out = tmp_path / "decision.json"
    return decide(
        v69_decision=paths["prior"], v69_crossed_report=paths["crossed"],
        isolation_report=paths["report"], manual_review=paths["manual"], output=out,
    )


def test_decision_statuses(tmp_path) -> None:
    for name in ("capable", "attenuated", "tradeoff", "invalid"):
        (tmp_path / name).mkdir()
    assert _write_case(tmp_path / "capable", 0.3, list(range(10))[0:10])["status"] == "post-slm-code-transport-capable"
    assert _write_case(tmp_path / "attenuated", 0.1, list(range(10)))["status"] == "post-slm-code-factor-attenuated"
    assert _write_case(tmp_path / "tradeoff", 0.3, [0, 1])["status"] == "post-slm-code-readability-tradeoff"
    assert _write_case(tmp_path / "invalid", 0.3, list(range(10)), valid=False)["status"] == "harness-invalid"
