from __future__ import annotations

import json

import torch

from tools.audit_v75_carrier_residual_interference import _region_sum, carrier_residual_terms
from tools.decide_v75_carrier_residual_interference import decide


def test_carrier_residual_exact_reconstruction_and_quadrature() -> None:
    generator = torch.Generator().manual_seed(1234)
    carrier = torch.randn((8, 8), generator=generator) + 1j * torch.randn((8, 8), generator=generator)
    residual = torch.randn((8, 8), generator=generator) + 1j * torch.randn((8, 8), generator=generator)
    terms = carrier_residual_terms(carrier + residual, carrier)
    assert torch.allclose(terms["delta_I"], terms["R"] + terms["Q"], rtol=1e-6, atol=1e-6)
    assert torch.allclose(terms["residual"], residual, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(terms["J"]).all()


def test_carrier_residual_zero_residual() -> None:
    carrier = torch.ones((4, 4), dtype=torch.complex64)
    terms = carrier_residual_terms(carrier, carrier)
    for key in ("residual", "R", "J", "Q", "delta_I"):
        assert torch.count_nonzero(terms[key]) == 0


def test_active_exterior_partition_is_exact() -> None:
    value = torch.ones((128, 128))
    parts = _region_sum(value, 128)
    assert parts["full"] == parts["active"] + parts["exterior"]
    assert parts["active"] == 10000.0


def _write_case(tmp_path, *, mechanism: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prior73 = {"status": "semantic-support-code-factor-attenuated"}
    prior74 = {"status": "full-state-phase-to-intensity-redistribution-confirmed"}
    v74 = {"status": "v74-full-state-partition-complete"}
    admission = {"status": "v75-interference-admission-passed"}
    manual = {"readable_labels": list(range(10))}
    if mechanism == "quadrature":
        first = {"delta": 100.0, "r": 100.0, "q": 0.0, "cov": 0.0, "within_r": 100.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}
        second = {"delta": 40.0, "r": 70.0, "q": 0.0, "cov": 0.0, "within_r": 95.0, "coherence": 1.0, "quadrature": 0.2, "residual_energy": 100.0}
    elif mechanism == "cross":
        first = {"delta": 100.0, "r": 100.0, "q": 0.0, "cov": 0.0, "within_r": 100.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}
        second = {"delta": 40.0, "r": 80.0, "q": 0.0, "cov": 0.0, "within_r": 95.0, "coherence": 0.8, "quadrature": 0.0, "residual_energy": 100.0}
    elif mechanism == "self":
        first = {"delta": 100.0, "r": 100.0, "q": 100.0, "cov": 0.0, "within_r": 100.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}
        second = {"delta": 40.0, "r": 100.0, "q": 40.0, "cov": -50.0, "within_r": 100.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}
    else:
        first = {"delta": 100.0, "r": 100.0, "q": 0.0, "cov": 0.0, "within_r": 100.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}
        second = {"delta": 40.0, "r": 70.0, "q": 0.0, "cov": 0.0, "within_r": 50.0, "coherence": 1.0, "quadrature": 0.0, "residual_energy": 100.0}

    def plane(item):
        return {
            "delta_I": {"code_effect_ss": {"full": item["delta"]}},
            "R": {"code_effect_ss": {"full": item["r"]}},
            "Q": {"code_effect_ss": {"full": item["q"]}},
            "within_label_R_ss": {"full": item["within_r"]}, "label_coherence": item["coherence"],
            "quadrature_share": item["quadrature"], "residual_complex_energy": item["residual_energy"],
            "covariance_ss": item["cov"], "reconstruction_error": {"max": 0.0},
        }
    report = {
        "status": "v75-carrier-residual-interference-complete", "diagnostic_only": True,
        "test_split_used": False, "return_optical_fields": True, "transient_full_state_used": True,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only", "carrier_persisted": False,
        "alias_replay_pass": True, "reconstruction_pass": True, "finite": True, "energy_pass": True,
        "parameters_unchanged": True, "optimizer_created": False, "parameter_update_count": 0,
        "teacher_alignment_computed": False, "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
        "amplitude": 0.4027685085311532, "v74_active_replay": {"v74_status_linked": True, "pass": True},
        "planes": {"encoder_full_grid_input": plane(first), "encoder_layer_1": plane(second), "camera_field": plane(second)},
    }
    paths = {}
    for name, value in (("prior73", prior73), ("prior74", prior74), ("v74", v74), ("admission", admission), ("manual", manual), ("report", report)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths


def test_decision_mechanism_branches(tmp_path) -> None:
    for mechanism, expected in (("quadrature", "carrier-residual-quadrature-rotation-confirmed"), ("cross", "cross-label-interference-cancellation-confirmed"), ("self", "residual-self-interference-cancellation-confirmed"), ("none", "carrier-residual-interference-inconclusive")):
        paths = _write_case(tmp_path / mechanism, mechanism=mechanism)
        assert decide(v73_decision=paths["prior73"], v74_decision=paths["prior74"], v73_manual_review=paths["manual"], v74_report=paths["v74"], admission=paths["admission"], audit_report=paths["report"], output=tmp_path / mechanism / "decision.json")["status"] == expected


def test_decision_is_fail_closed_on_provenance(tmp_path) -> None:
    paths = _write_case(tmp_path / "invalid", mechanism="none")
    report = json.loads(paths["report"].read_text())
    report["alias_replay_pass"] = False
    paths["report"].write_text(json.dumps(report))
    result = decide(v73_decision=paths["prior73"], v74_decision=paths["prior74"], v73_manual_review=paths["manual"], v74_report=paths["v74"], admission=paths["admission"], audit_report=paths["report"], output=tmp_path / "invalid" / "decision.json")
    assert result["status"] == "harness-invalid"
