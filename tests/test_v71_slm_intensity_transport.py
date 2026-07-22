from __future__ import annotations

import json

import torch

from tools.audit_v71_slm_intensity_transport import calibrate_physical_intensity_amplitude
from tools.audit_v70_slm_transport import build_lowfreq_code_pattern
from tools.decide_v71_slm_intensity_transport import decide
from tools.v68_b_acceptance_utils import balanced_factor_decomposition


def test_physical_intensity_calibration_and_domain_split() -> None:
    generator = torch.Generator().manual_seed(1234)
    codes = torch.randn((100, 128), generator=generator)
    pattern = build_lowfreq_code_pattern(codes, size=16).unsqueeze(1).expand(100, 10, 1, 16, 16)
    base = torch.rand((100, 10, 1, 16, 16), generator=generator) * 0.6 + 0.2
    result = calibrate_physical_intensity_amplitude(base, pattern, iterations=30)
    assert result["finite"]
    assert result["absolute_error"] <= 1e-4
    command = (base + result["amplitude"] * pattern).clamp(0.0, 1.0)
    amplitude = 0.05 + 0.95 * command
    assert abs(balanced_factor_decomposition(amplitude)[0]["code_fraction"] - 0.25) > 1e-3
    assert abs(balanced_factor_decomposition(amplitude.square())[0]["code_fraction"] - 0.25) <= 1e-4


def test_complex_factor_stream_contract() -> None:
    values = torch.zeros((100, 10, 2, 4, 4))
    values[:, :, 0, 0, 0] = torch.arange(100).view(100, 1)
    factors = balanced_factor_decomposition(values)[0]
    assert factors["finite"]
    assert factors["code_fraction"] > 0.99


def _case(tmp_path, *, camera: float, pre: float, chip: float, readable: list[int], valid: bool = True) -> dict:
    prior = {"status": "v69-prototype-only"}
    v70 = {"status": "post-slm-code-factor-attenuated"}
    metrics = {"clean_prototype_corr": 0.9, "clean_prototype_dice": 0.9, "prototype_accuracy": 1.0, "high_saturation": 0.01}
    report = {
        "status": "v71-slm-intensity-isolation-complete",
        "diagnostic_only": True,
        "test_split_used": False,
        "return_optical_fields": False,
        "batch_size": 1,
        "calibration_semantics": "slm_amplitude_squared",
        "parameters_unchanged": valid,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "finite": True,
        "energy_pass": True,
        "calibration": {"finite": True, "absolute_error": 1e-6, "achieved_code_fraction": 0.25},
        "intensity_code_factors": {"slm_physical_intensity": 0.25, "pre_encoder": pre, "chip": chip, "camera": camera},
        "first_intensity_plane_below_25_percent": "camera",
        "crossed_metrics": metrics,
        "diagonal_metrics": metrics,
        "complex_factor_decomposition": {"pre_encoder": {"code_fraction": pre}},
    }
    paths = {}
    for name, value in (("prior", prior), ("v70", v70), ("report", report), ("manual", {"readable_labels": readable})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return decide(
        v69_decision=paths["prior"], v70_decision=paths["v70"], isolation_report=paths["report"],
        manual_review=paths["manual"], output=tmp_path / "decision.json",
    )


def test_all_decision_branches(tmp_path) -> None:
    for name in ("capable", "tradeoff", "front", "d2nn", "camera", "invalid"):
        (tmp_path / name).mkdir()
    assert _case(tmp_path / "capable", camera=0.3, pre=0.3, chip=0.3, readable=list(range(10)))["status"] == "physical-slm-to-camera-transport-capable"
    assert _case(tmp_path / "tradeoff", camera=0.3, pre=0.3, chip=0.3, readable=[0, 1])["status"] == "physical-slm-code-readability-tradeoff"
    assert _case(tmp_path / "front", camera=0.1, pre=0.2, chip=0.2, readable=list(range(10)))["status"] == "front-relay-code-factor-attenuated"
    assert _case(tmp_path / "d2nn", camera=0.2, pre=0.3, chip=0.2, readable=list(range(10)))["status"] == "d2nn-code-factor-attenuated"
    assert _case(tmp_path / "camera", camera=0.2, pre=0.3, chip=0.3, readable=list(range(10)))["status"] == "camera-relay-code-factor-attenuated"
    assert _case(tmp_path / "invalid", camera=0.3, pre=0.3, chip=0.3, readable=list(range(10)), valid=False)["status"] == "harness-invalid"
