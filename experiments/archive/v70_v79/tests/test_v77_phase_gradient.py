from __future__ import annotations

import json

import torch

from tools.audit_v77_phase_gradient_compatibility import (
    balanced_code_stats,
    project_code_ascent,
    streaming_code_weights,
)
from tools.decide_v77_phase_gradient import decide


def test_streaming_code_weights_match_materialized_anova_gradient() -> None:
    values = torch.arange(1000 * 2, dtype=torch.float32).reshape(100, 10, 2).requires_grad_()
    materialized = balanced_code_stats(values)
    stream = streaming_code_weights(values.detach().sum(dim=1), values.detach().sum(dim=(0, 1)))
    assert torch.allclose(materialized["weights"], stream["weights"])
    assert torch.allclose(materialized["code_effect_ss"], stream["code_effect_ss"])
    materialized["code_effect_ss"].backward()
    weighted = (values * stream["weights"].detach()[:, None]).sum()
    weighted_grad = torch.autograd.grad(weighted, values, retain_graph=False)[0]
    assert torch.allclose(values.grad, weighted_grad, atol=1.0e-5, rtol=1.0e-5)


def test_projection_handles_nonconflict_conflict_and_full_cancellation() -> None:
    nonconflict = project_code_ascent(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    assert nonconflict["projected_code_drive_ratio"] == 1.0
    assert nonconflict["first_order_readability_change"] < 0.0
    orthogonal = project_code_ascent(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert orthogonal["projected_code_drive_ratio"] == 1.0
    assert abs(orthogonal["first_order_readability_change"]) < 1.0e-8
    cancellation = project_code_ascent(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert cancellation["projected_code_drive_ratio"] == 0.0


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_v77_decision_branches(tmp_path) -> None:
    base = {
        "v73_admission": {"status": "v73-cpu-admission-passed"},
        "v73_report": {"status": "v73-semantic-support-audit-complete"},
        "v73_decision": {"status": "semantic-support-code-factor-attenuated"},
        "manual": {"readable_labels": list(range(10))},
        "v75_admission": {"status": "v75-interference-admission-passed"},
        "v75_report": {"status": "v75-carrier-residual-interference-complete"},
        "v75_decision": {"status": "carrier-residual-quadrature-rotation-confirmed"},
        "v76_admission": {"status": "v76-dual-quadrature-admission-passed"},
        "v76_report": {"status": "v76-dual-quadrature-readout-complete"},
        "v76_decision": {"status": "dual-quadrature-dynamic-range-tradeoff"},
        "admission": {"status": "v77-phase-gradient-admission-passed"},
    }
    names = ("encoder_layer_1", "encoder_layer_2", "decoder_layer_1", "decoder_layer_2")

    def run(metrics, *, audit_overrides=None):
        audit = {
            "schema": "v77_phase_gradient_compatibility_audit_v1",
            "diagnostic_only": True, "return_optical_fields": False, "test_split_used": False,
            "finite": True, "parameters_unchanged": True, "optimizer_created": False,
            "parameter_update_count": 0, "full_field_saved": False, "persistent_full_field_saved": False,
            "teacher_alignment_computed": False, "coherent_receiver_branch_closed": True,
            "learned_code_transport_proven": False, "gradient_metrics": metrics,
        }
        if audit_overrides:
            audit.update(audit_overrides)
        paths = {}
        for key, value in base.items():
            paths[key] = _write_json(tmp_path / f"{key}_{len(list(tmp_path.iterdir()))}.json", value)
        paths["smoke_report"] = _write_json(tmp_path / f"smoke_{len(list(tmp_path.iterdir()))}.json", {"status": "v77-phase-gradient-smoke-passed"})
        paths["audit_report"] = _write_json(tmp_path / f"audit_{len(list(tmp_path.iterdir()))}.json", audit)
        paths["output"] = tmp_path / f"decision_{len(list(tmp_path.iterdir()))}.json"
        return decide(
            v73_admission=paths["v73_admission"], v73_report=paths["v73_report"], v73_decision=paths["v73_decision"],
            v73_manual_review=paths["manual"], v75_admission=paths["v75_admission"], v75_report=paths["v75_report"],
            v75_decision=paths["v75_decision"], v76_admission=paths["v76_admission"], v76_report=paths["v76_report"],
            v76_decision=paths["v76_decision"], admission=paths["admission"], smoke_report=paths["smoke_report"],
            audit_report=paths["audit_report"], output=paths["output"],
        )

    def metric(ratio, *, nonzero=True, readability=0.0):
        return {"finite": True, "code_gradient_norm": 1.0 if nonzero else 0.0, "code_gradient_suppressed": not nonzero,
                "projected_code_drive_ratio": ratio, "first_order_readability_change": readability}

    good = {name: metric(0.5) for name in (*names, "joint")}
    assert run(good)["status"] == "phase-aware-local-co-design-admissible"
    conflict = {name: metric(0.1 if name == "joint" else 0.5) for name in (*names, "joint")}
    assert run(conflict)["status"] == "phase-readability-gradient-conflict"
    encoder_fail = {name: metric(0.1 if name == "encoder_layer_1" else 0.5) for name in (*names, "joint")}
    assert run(encoder_fail)["status"] == "encoder-l1-phase-control-insufficient"
    suppressed = {name: metric(0.0, nonzero=False) for name in (*names, "joint")}
    assert run(suppressed)["status"] == "phase-code-gradient-suppressed"
    assert run(good, audit_overrides={"parameters_unchanged": False})["status"] == "harness-invalid"

