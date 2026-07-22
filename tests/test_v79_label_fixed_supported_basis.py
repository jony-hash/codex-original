from __future__ import annotations

import json

import torch

from ldm_project.student.semantic_command import semantic_basis
from tools.audit_v79_label_fixed_supported_basis import _StreamingStage
from tools.decide_v79_label_fixed_supported_basis import decide


def test_label_fixed_basis_bound_is_independent_of_code():
    basis = semantic_basis(16)
    support = torch.rand(10, 16, 16)
    mean = (support[:, None] * basis[None]).sum((-2, -1)) / support.sum((-2, -1))[:, None]
    psi = support[:, None] * (basis[None] - mean[:, :, None, None])
    normalizer = psi.abs().sum(1).flatten(1).amax(1)
    fixed = psi / normalizer[:, None, None, None].clamp_min(1.0e-8)
    coeff = torch.tanh(torch.randn(100, 10, 8))
    q = torch.einsum("clk,lkhw->clhw", coeff, fixed)
    assert float(q.abs().max()) <= 1.0 + 1.0e-5


def test_v79_streaming_stage_matches_materialized():
    values = torch.arange(100 * 10 * 3, dtype=torch.float32).reshape(100, 10, 3)
    stream = _StreamingStage()
    for start in range(0, 100, 10):
        stream.add(values[start : start + 10])
    report = stream.report()
    grand = values.mean((0, 1))
    code_ss = 10.0 * (values.mean(1) - grand).square().sum().item()
    total_ss = values.sub(grand).square().sum().item()
    assert abs(report["code_effect_ss"] - code_ss) / code_ss < 1.0e-6
    assert abs(report["total_ss"] - total_ss) / total_ss < 1.0e-6


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_v79_decision_branches(tmp_path):
    def run(capacity, learned, **overrides):
        arm = {
            "finite": True,
            "code_fraction": capacity,
            "code_effect_ss": 300000.0,
            "label_effect_ss": 700000.0,
            "prototype_support_command_delta_energy_fraction": 0.96,
            "low_frequency_energy_fraction": 0.995,
            "one_clipping_fraction": 0.01,
            "max_abs_q": 0.8,
            "formula_errors": {"basis": 0.0, "weighted_mean": 0.0, "normalization": 0.0, "q": 0.0, "command": 0.0, "intensity": 0.0},
        }
        report = {
            "status": "v79-label-fixed-supported-basis-audit-complete",
            "finite": True,
            "parameters_unchanged": True,
            "state_keys_equal": True,
            "parameter_hash_before": "same",
            "parameter_hash_after": "same",
            "checkpoint_state_keys_compatible": True,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "optical_forward_used": False,
            "cuda_used": False,
            "persistent_full_field_saved": False,
            "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
            "arms": {"deterministic_capacity": {**arm, "code_fraction": capacity}, "learned_replay": {**arm, "code_fraction": learned}},
        }
        report.update(overrides)
        rp = _write(tmp_path / f"report_{len(list(tmp_path.iterdir()))}.json", report)
        return decide(report=rp, output=tmp_path / "decision.json")["status"]

    assert run(0.1, 0.1) == "label-fixed-supported-basis-insufficient"
    assert run(0.3, 0.1) == "label-fixed-basis-capable-warmstart-insufficient"
    assert run(0.3, 0.3) == "label-fixed-supported-basis-capable"
    assert run(0.3, 0.3, state_keys_equal=False) == "harness-invalid"
