"""Joint decision for the diagnostic v74 full-state partition audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COMPLEX_RETENTION_MIN = 0.99
INTENSITY_RETENTION_MIN = 0.90
ESCAPE_EXPLANATION_MIN = 0.90


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _transition_valid(transition: dict[str, Any]) -> bool:
    for feature in ("intensity", "raw_complex", "gauge_complex"):
        item = transition.get(feature, {})
        for key in ("full_retention", "active_retention", "exterior_gain", "active_loss", "exterior_explanation_ratio"):
            value = item.get(key)
            if not isinstance(value, (int, float)) or value != value:
                return False
    return True


def decide(*, v73_decision: str | Path, v73_report: str | Path, v73_manual_review: str | Path, admission: str | Path, audit_report: str | Path, output: str | Path) -> dict[str, Any]:
    prior_decision = _read(v73_decision)
    prior_report = _read(v73_report)
    manual = _read(v73_manual_review)
    admission_report = _read(admission)
    report = _read(audit_report)
    readable = sorted(int(value) for value in manual.get("readable_labels", [])) == list(range(10))
    transitions = report.get("transitions", {})
    valid_transitions = bool(transitions) and all(_transition_valid(value) for value in transitions.values())
    provenance_pass = (
        prior_decision.get("status") == "semantic-support-code-factor-attenuated"
        and prior_report.get("status") == "v73-semantic-support-audit-complete"
        and admission_report.get("status") == "v74-full-state-admission-passed"
        and report.get("status") == "v74-full-state-partition-complete"
        and report.get("diagnostic_only") is True
        and report.get("test_split_used") is False
        and report.get("return_optical_fields") is True
        and report.get("transient_full_state_used") is True
        and report.get("persistent_full_field_saved") is False
        and report.get("saved_artifacts") == "scalar_statistics_only"
        and report.get("batch_size") == 1
        and report.get("finite") is True
        and report.get("energy_pass") is True
        and report.get("parameters_unchanged") is True
        and report.get("optimizer_created") is False
        and report.get("parameter_update_count") == 0
        and report.get("teacher_alignment_computed") is False
        and report.get("v73_active_replay_pass") is True
        and report.get("checkpoint_boundary_contract") == "ideal_transparent_thin_phase_v2"
        and abs(float(report.get("amplitude", 0.0)) - 0.4027685085311532) <= 1.0e-9
        and valid_transitions
    )
    first_transition_name = next(iter(transitions), None)
    first = transitions.get(first_transition_name, {}) if first_transition_name else {}
    raw_ret = float(first.get("raw_complex", {}).get("full_retention", 0.0))
    gauge_ret = float(first.get("gauge_complex", {}).get("full_retention", 0.0))
    intensity_ret = float(first.get("intensity", {}).get("full_retention", 0.0))
    active_intensity_ret = float(first.get("intensity", {}).get("active_retention", 0.0))
    explanation = float(first.get("intensity", {}).get("exterior_explanation_ratio", 0.0))
    complex_retained = raw_ret >= COMPLEX_RETENTION_MIN and gauge_ret >= COMPLEX_RETENTION_MIN
    active_loss = active_intensity_ret < INTENSITY_RETENTION_MIN
    exterior_escape = explanation >= ESCAPE_EXPLANATION_MIN
    if not provenance_pass:
        status = "harness-invalid"
    elif not complex_retained:
        status = "state-grid-or-bandlimit-code-loss"
    elif intensity_ret >= INTENSITY_RETENTION_MIN and active_loss and exterior_escape:
        status = "active-window-code-escape-confirmed"
    elif intensity_ret < INTENSITY_RETENTION_MIN and exterior_escape:
        status = "mixed-active-exterior-phase-redistribution"
    elif intensity_ret < INTENSITY_RETENTION_MIN:
        status = "full-state-phase-to-intensity-redistribution-confirmed"
    elif not active_loss:
        status = "full-state-attribution-inconclusive"
    else:
        status = "full-state-attribution-inconclusive"
    decision = {
        "schema": "v74_full_state_partition_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "first_transition": first_transition_name,
        "raw_complex_full_retention": raw_ret,
        "gauge_complex_full_retention": gauge_ret,
        "intensity_full_retention": intensity_ret,
        "intensity_active_retention": active_intensity_ret,
        "exterior_explanation_ratio": explanation,
        "complex_retained": complex_retained,
        "active_intensity_loss": active_loss,
        "exterior_escape": exterior_escape,
        "manual_readability_reused": readable,
        "provenance_pass": provenance_pass,
        "transitions": transitions,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
        "next_action": {
            "active-window-code-escape-confirmed": "freeze geometry; design review only for controllable active support",
            "full-state-phase-to-intensity-redistribution-confirmed": "continue read-only phase-to-intensity attribution; no training",
            "mixed-active-exterior-phase-redistribution": "continue read-only spatial/phase partition; no training",
            "state-grid-or-bandlimit-code-loss": "inspect transient FFT crop/bandlimit code partition; no training",
            "full-state-attribution-inconclusive": "freeze v74 and review audit statistics; no training",
            "harness-invalid": "repair and rerun the complete v74 audit",
        }[status],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v73-decision", required=True, type=Path)
    parser.add_argument("--v73-report", required=True, type=Path)
    parser.add_argument("--v73-manual-review", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = decide(**vars(parser.parse_args()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
