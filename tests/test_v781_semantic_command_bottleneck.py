from __future__ import annotations

import json

import torch

from tools.audit_v781_semantic_command_bottleneck import _StreamingStage
from tools.decide_v781_semantic_command_bottleneck import decide


def test_streaming_stage_matches_materialized_code_ss():
    values = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    stream = _StreamingStage(4, 2)
    stream.add(0, values[:2])
    stream.add(2, values[2:])
    report = stream.report()
    grand = values.mean((0, 1))
    code_mean = values.mean(1)
    total_ss = values.sub(grand).square().sum().item()
    code_ss = 2.0 * code_mean.sub(grand).square().sum().item()
    assert abs(report["total_ss"] - total_ss) < 1.0e-8
    assert abs(report["code_effect_ss"] - code_ss) < 1.0e-8
    assert abs(report["code_fraction"] - code_ss / total_ss) < 1.0e-8


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_v781_decision_branches(tmp_path):
    def run(projected, supported, normalized, **overrides):
        stages = {
            "projected_pattern": {"code_fraction": projected, "finite": True},
            "template_support_pattern": {"code_fraction": supported, "finite": True},
            "normalized_pattern": {"code_fraction": normalized, "finite": True},
        }
        report = {
            "status": "v781-semantic-command-bottleneck-complete",
            "finite": True, "parameters_unchanged": True,
            "state_keys_equal": True, "parameter_hash_before": "same", "parameter_hash_after": "same",
            "optimizer_created": False, "parameter_update_count": 0,
            "persistent_full_field_saved": False,
            "reconstruction_errors": {"basis": 0.0, "weighted_mean": 0.0, "normalization": 0.0, "command": 0.0},
            "stages": stages,
        }
        report.update(overrides)
        rp = _write(tmp_path / f"report_{len(list(tmp_path.iterdir()))}.json", report)
        out = tmp_path / f"out_{len(list(tmp_path.iterdir()))}.json"
        return decide(report=rp, output=out)["status"]

    assert run(0.1, 0.05, 0.01) == "basis-projected-code-insufficient"
    assert run(0.6, 0.1, 0.04) == "mixed-support-normalization-code-suppression"
    assert run(0.6, 0.1, 0.09) == "semantic-support-code-suppression"
    assert run(0.6, 0.5, 0.1) == "per-sample-normalization-code-suppression"
    assert run(0.6, 0.5, 0.4, finite=False) == "harness-invalid"
