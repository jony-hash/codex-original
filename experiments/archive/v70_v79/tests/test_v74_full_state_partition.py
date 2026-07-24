from __future__ import annotations

import json

import torch

from tools.audit_v74_full_state_partition import _CodeSS
from tools.decide_v74_full_state_partition import decide


def test_streaming_code_ss_partitions_active_and_exterior() -> None:
    stream = _CodeSS(128, torch.device("cpu"), active_size=8, shell_sizes=(8, 16, 32, 64, 96, 128))
    generator = torch.Generator().manual_seed(1234)
    values = []
    for code in range(100):
        value = torch.randn((128, 128), generator=generator)
        values.append(value)
        stream.add_code_mean(value)
    report = stream.report()
    assert report["finite"]
    full = report["code_effect_ss"]["full"]
    assert abs(full - (report["code_effect_ss"]["active"] + report["code_effect_ss"]["exterior"])) / full < 1e-5
    shells = sum(report["code_effect_ss"][f"shell_{size}"] for size in (8, 16, 32, 64, 96, 128))
    assert abs(shells - full) / full < 1e-5


def test_streaming_matches_materialized_small() -> None:
    stream = _CodeSS(128, torch.device("cpu"), active_size=8, shell_sizes=(8, 16, 32, 64, 96, 128))
    generator = torch.Generator().manual_seed(4)
    values = torch.randn((100, 128, 128), generator=generator)
    for value in values:
        stream.add_code_mean(value)
    report = stream.report()
    grand = values.mean(0)
    expected = 10.0 * ((values - grand) ** 2).sum()
    assert abs(report["code_effect_ss"]["full"] - float(expected)) / float(expected) < 1e-5


def _case(tmp_path, *, status: str, raw: float = 1.0, gauge: float = 1.0, intensity: float = 0.8, active: float = 0.5, explanation: float = 1.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prior = {"status": "semantic-support-code-factor-attenuated"}
    prior_report = {"status": "v73-semantic-support-audit-complete"}
    admission = {"status": "v74-full-state-admission-passed"}
    transition = {
        "from": "encoder_full_grid_input",
        "intensity": {"full_retention": intensity, "active_retention": active, "exterior_gain": 1.0, "active_loss": 1.0, "exterior_explanation_ratio": explanation},
        "raw_complex": {"full_retention": raw, "active_retention": 1.0, "exterior_gain": 1.0, "active_loss": 1.0, "exterior_explanation_ratio": explanation},
        "gauge_complex": {"full_retention": gauge, "active_retention": 1.0, "exterior_gain": 1.0, "active_loss": 1.0, "exterior_explanation_ratio": explanation},
    }
    report = {
        "status": "v74-full-state-partition-complete", "diagnostic_only": True, "test_split_used": False,
        "return_optical_fields": True, "transient_full_state_used": True, "persistent_full_field_saved": False,
        "saved_artifacts": "scalar_statistics_only", "batch_size": 1, "finite": True, "parameters_unchanged": True,
        "energy_pass": True,
        "optimizer_created": False, "parameter_update_count": 0, "teacher_alignment_computed": False,
        "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2", "amplitude": 0.4027685085311532,
        "v73_active_replay_pass": True,
        "transitions": {"encoder_layer_1": transition},
    }
    manual = {"readable_labels": list(range(10))}
    paths = {}
    for name, value in (("prior", prior), ("prior_report", prior_report), ("admission", admission), ("report", report), ("manual", manual)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return decide(v73_decision=paths["prior"], v73_report=paths["prior_report"], v73_manual_review=paths["manual"], admission=paths["admission"], audit_report=paths["report"], output=tmp_path / "decision.json")["status"]


def test_decision_branches(tmp_path) -> None:
    assert _case(tmp_path / "escape", status="", intensity=0.95, active=0.5, explanation=1.0) == "active-window-code-escape-confirmed"
    assert _case(tmp_path / "phase", status="", intensity=0.8, active=0.5, explanation=0.1) == "full-state-phase-to-intensity-redistribution-confirmed"
    assert _case(tmp_path / "mixed", status="", intensity=0.8, active=0.5, explanation=1.0) == "mixed-active-exterior-phase-redistribution"
    assert _case(tmp_path / "complex", status="", raw=0.8, gauge=0.8) == "state-grid-or-bandlimit-code-loss"
    assert _case(tmp_path / "inconclusive", status="", intensity=0.95, active=0.95, explanation=0.1) == "full-state-attribution-inconclusive"
