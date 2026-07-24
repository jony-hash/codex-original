from __future__ import annotations

import json

import torch

from tools.audit_v76_dual_quadrature_readout import _StreamingFactor, balanced_ports
from tools.decide_v76_dual_quadrature_readout import decide


def test_balanced_ports_recover_real_and_quadrature_interference() -> None:
    generator = torch.Generator().manual_seed(1234)
    carrier = torch.randn((12, 12), generator=generator) + 1j * torch.randn((12, 12), generator=generator)
    residual = torch.randn((12, 12), generator=generator) + 1j * torch.randn((12, 12), generator=generator)
    values = balanced_ports(carrier + residual, carrier)
    product = carrier.conj() * residual
    assert torch.allclose(values["H_R"], 2 * product.real, rtol=1e-5, atol=1e-5)
    assert torch.allclose(values["H_J"], 2 * product.imag, rtol=1e-5, atol=1e-5)
    assert torch.allclose(values["r_energy_sum_residual"], torch.zeros_like(carrier.real), atol=5e-6)
    assert torch.allclose(values["j_energy_sum_residual"], torch.zeros_like(carrier.real), atol=5e-6)


def test_streaming_factor_matches_materialized() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn((100, 10, 2, 4, 4), generator=generator)
    stream = _StreamingFactor((2, 4, 4))
    for code in range(100):
        for label in range(10):
            stream.add(code, label, values[code, label])
    report = stream.report()
    grand = values.mean((0, 1))
    code = values.mean(1)
    label = values.mean(0)
    expected_code = 10 * (code - grand).square().sum()
    expected_total = values.sub(grand).square().sum()
    assert abs(report["code_effect_ss"] - float(expected_code)) / float(expected_code) < 1e-6
    assert abs(report["total_energy"] - float(expected_total)) / float(expected_total) < 1e-6


def _case(tmp_path, *, factor: float = 0.30, absolute: float = 60000.0, saturation: float = 0.01) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values = {
        "prior_admission": {"status": "v75-interference-admission-passed"},
        "prior_report": {"status": "v75-carrier-residual-interference-complete"},
        "prior_decision": {"status": "carrier-residual-quadrature-rotation-confirmed"},
        "admission": {"status": "v76-dual-quadrature-admission-passed"},
        "manual": {"readable_labels": list(range(10))},
        "v75_report": {"status": "v75-carrier-residual-interference-complete"},
    }
    report = {
        "status": "v76-dual-quadrature-readout-complete", "diagnostic_only": True,
        "return_optical_fields": False, "test_split_used": False, "transient_carrier_used": True,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only", "carrier_persisted": False,
        "identity_pass": True, "replay_pass": True, "finite": True, "energy_pass": True,
        "parameters_unchanged": True, "optimizer_created": False, "parameter_update_count": 0,
        "teacher_alignment_computed": False, "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
        "joint_factor": {"code_fraction": factor, "code_effect_ss": absolute},
        "absolute_ss": {"v72_baseline_code_ss": 53141.618758854296, "joint_vs_v75_raw": absolute / 39513.27116313368},
        "port_saturation": {name: {"high_saturation": saturation, "finite": True} for name in ("r_plus", "r_minus", "j_plus", "j_minus")},
        "v75_replay": {"raw_metrics_match": True},
        "camera_metrics": {"clean_prototype_corr": 0.95, "clean_prototype_dice": 0.94, "prototype_accuracy": 1.0},
    }
    values["report"] = report
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths


def test_decision_branches(tmp_path) -> None:
    cases = (
        ("passed", {}, "dual-quadrature-readout-upper-bound-passed"),
        ("ratio", {"factor": 0.30, "absolute": 40000.0}, "dual-quadrature-factor-ratio-artifact"),
        ("dynamic", {"saturation": 0.10}, "dual-quadrature-dynamic-range-tradeoff"),
        ("insufficient", {"factor": 0.20}, "dual-quadrature-readout-insufficient"),
    )
    for name, overrides, expected in cases:
        paths = _case(tmp_path / name, **overrides)
        result = decide(
            v73_manual_review=paths["manual"], v75_admission=paths["prior_admission"],
            v75_report=paths["prior_report"], v75_decision=paths["prior_decision"],
            admission=paths["admission"], audit_report=paths["report"], output=tmp_path / name / "decision.json",
        )
        assert result["status"] == expected


def test_decision_fail_closed(tmp_path) -> None:
    paths = _case(tmp_path / "invalid")
    report = json.loads(paths["report"].read_text())
    report["identity_pass"] = False
    paths["report"].write_text(json.dumps(report))
    result = decide(
        v73_manual_review=paths["manual"], v75_admission=paths["prior_admission"],
        v75_report=paths["prior_report"], v75_decision=paths["prior_decision"],
        admission=paths["admission"], audit_report=paths["report"], output=tmp_path / "invalid" / "decision.json",
    )
    assert result["status"] == "harness-invalid"
