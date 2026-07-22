"""Joint decision for the diagnostic v76 dual-quadrature readout upper bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ABSOLUTE_CODE_SS_MIN = 53141.618758854296
HIGH_SATURATION_MAX = 0.08
FACTOR_MIN = 0.25


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _report_valid(report: dict[str, Any]) -> bool:
    required = {
        "status": "v76-dual-quadrature-readout-complete",
        "diagnostic_only": True,
        "return_optical_fields": False,
        "test_split_used": False,
        "transient_carrier_used": True,
        "persistent_full_field_saved": False,
        "saved_artifacts": "scalar_statistics_only",
        "carrier_persisted": False,
        "identity_pass": True,
        "replay_pass": True,
        "finite": True,
        "energy_pass": True,
        "parameters_unchanged": True,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "teacher_alignment_computed": False,
        "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
    }
    return all(report.get(key) == value for key, value in required.items())


def decide(*, v73_manual_review: str | Path, v75_admission: str | Path,
           v75_report: str | Path, v75_decision: str | Path,
           admission: str | Path, audit_report: str | Path, output: str | Path) -> dict[str, Any]:
    manual = _read(v73_manual_review)
    prior_admission = _read(v75_admission)
    prior_report = _read(v75_report)
    prior_decision = _read(v75_decision)
    admission_report = _read(admission)
    report = _read(audit_report)
    readable = sorted(int(value) for value in manual.get("readable_labels", [])) == list(range(10))
    provenance_pass = (
        prior_admission.get("status") == "v75-interference-admission-passed"
        and prior_report.get("status") == "v75-carrier-residual-interference-complete"
        and prior_decision.get("status") == "carrier-residual-quadrature-rotation-confirmed"
        and admission_report.get("status") == "v76-dual-quadrature-admission-passed"
        and _report_valid(report)
        and readable
    )
    joint = report.get("joint_factor", {})
    joint_factor = float(joint.get("code_fraction", 0.0))
    joint_code_ss = float(joint.get("code_effect_ss", 0.0))
    absolute = report.get("absolute_ss", {})
    baseline = float(absolute.get("v72_baseline_code_ss", ABSOLUTE_CODE_SS_MIN))
    raw_replay = report.get("v75_replay", {})
    saturation = report.get("port_saturation", {})
    port_saturation_pass = bool(saturation) and all(float(item.get("high_saturation", 1.0)) <= HIGH_SATURATION_MAX and bool(item.get("finite", False)) for item in saturation.values())
    factor_pass = joint_factor >= FACTOR_MIN
    absolute_pass = joint_code_ss >= baseline
    raw_readability_pass = (
        readable
        and bool(raw_replay.get("raw_metrics_match"))
        and float(report.get("camera_metrics", {}).get("clean_prototype_corr", 0.0)) >= 0.70
        and float(report.get("camera_metrics", {}).get("clean_prototype_dice", 0.0)) >= 0.68
        and float(report.get("camera_metrics", {}).get("prototype_accuracy", 0.0)) >= 0.80
    )
    if not provenance_pass:
        status = "harness-invalid"
    elif not raw_readability_pass:
        status = "harness-invalid"
    elif factor_pass and not port_saturation_pass:
        status = "dual-quadrature-dynamic-range-tradeoff"
    elif factor_pass and absolute_pass and port_saturation_pass:
        status = "dual-quadrature-readout-upper-bound-passed"
    elif factor_pass and not absolute_pass:
        status = "dual-quadrature-factor-ratio-artifact"
    else:
        status = "dual-quadrature-readout-insufficient"
    decision = {
        "schema": "v76_dual_quadrature_readout_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "joint_camera_code_factor": joint_factor,
        "joint_camera_absolute_code_ss": joint_code_ss,
        "v72_baseline_absolute_code_ss": baseline,
        "joint_vs_v72_baseline": joint_code_ss / max(baseline, 1.0e-12),
        "joint_vs_v75_raw": float(absolute.get("joint_vs_v75_raw", 0.0)),
        "factor_pass": factor_pass,
        "absolute_ss_pass": absolute_pass,
        "port_saturation_pass": port_saturation_pass,
        "raw_readability_pass": raw_readability_pass,
        "manual_readability_reused": readable,
        "provenance_pass": provenance_pass,
        "port_saturation": saturation,
        "next_action": {
            "dual-quadrature-readout-upper-bound-passed": "design review for a physical four-port single-frame coherent receiver; do not implement or train yet",
            "dual-quadrature-factor-ratio-artifact": "freeze the readout upper bound; no LO/gain/phase sweep; review absolute code transport",
            "dual-quadrature-dynamic-range-tradeoff": "freeze the readout upper bound; design a constrained receiver dynamic-range contract",
            "dual-quadrature-readout-insufficient": "close coherent-reference readout variations; review phase-aware D2NN co-design versus receiver redesign",
            "harness-invalid": "repair and rerun the complete v76 two-pass audit",
        }[status],
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v73-manual-review", required=True, type=Path)
    parser.add_argument("--v75-admission", required=True, type=Path)
    parser.add_argument("--v75-report", required=True, type=Path)
    parser.add_argument("--v75-decision", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(decide(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
