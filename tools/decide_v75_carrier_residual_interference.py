"""Joint decision for the diagnostic v75 carrier/residual interference audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_AMPLITUDE = 0.4027685085311532
RETENTION_MIN = 0.99


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_report(report: dict[str, Any]) -> bool:
    if report.get("status") != "v75-carrier-residual-interference-complete":
        return False
    required = {
        "diagnostic_only": True, "test_split_used": False, "return_optical_fields": True,
        "transient_full_state_used": True, "persistent_full_field_saved": False,
        "saved_artifacts": "scalar_statistics_only", "carrier_persisted": False,
        "alias_replay_pass": True, "reconstruction_pass": True, "finite": True,
        "energy_pass": True, "parameters_unchanged": True, "optimizer_created": False,
        "parameter_update_count": 0, "teacher_alignment_computed": False,
        "checkpoint_boundary_contract": "ideal_transparent_thin_phase_v2",
    }
    return all(report.get(key) == value for key, value in required.items()) and abs(float(report.get("amplitude", 0.0)) - EXPECTED_AMPLITUDE) <= 1e-9


def _plane_metrics(report: dict[str, Any], plane: str) -> dict[str, float]:
    item = report.get("planes", {}).get(plane, {})
    delta = item.get("delta_I", {}).get("code_effect_ss", {})
    r = item.get("R", {}).get("code_effect_ss", {})
    q = item.get("Q", {}).get("code_effect_ss", {})
    cov = float(item.get("covariance_ss", 0.0))
    return {
        "delta": float(delta.get("full", 0.0)), "r": float(r.get("full", 0.0)),
        "q": float(q.get("full", 0.0)), "cov": cov,
        "within_r": float(item.get("within_label_R_ss", {}).get("full", 0.0)),
        "coherence": float(item.get("label_coherence", 0.0)),
        "quadrature": float(item.get("quadrature_share", 0.0)),
        "residual_energy": float(item.get("residual_complex_energy", 0.0)),
    }


def decide(*, v73_decision: str | Path, v74_decision: str | Path,
           v73_manual_review: str | Path, v74_report: str | Path,
           admission: str | Path, audit_report: str | Path, output: str | Path) -> dict[str, Any]:
    prior73 = _read(v73_decision)
    prior74 = _read(v74_decision)
    v74 = _read(v74_report)
    admission_report = _read(admission)
    report = _read(audit_report)
    manual = _read(v73_manual_review)
    readable = sorted(int(x) for x in manual.get("readable_labels", [])) == list(range(10))
    provenance_pass = (
        prior73.get("status") == "semantic-support-code-factor-attenuated"
        and prior74.get("status") == "full-state-phase-to-intensity-redistribution-confirmed"
        and v74.get("status") == "v74-full-state-partition-complete"
        and admission_report.get("status") == "v75-interference-admission-passed"
        and _finite_report(report)
        and report.get("v74_active_replay", {}).get("pass") is True
        and abs(float(report.get("amplitude", 0.0)) - EXPECTED_AMPLITUDE) <= 1e-9
    )
    planes = report.get("planes", {})
    complete = all(name in planes for name in ("encoder_full_grid_input", "encoder_layer_1", "camera_field"))
    provenance_pass = provenance_pass and complete
    first = _plane_metrics(report, "encoder_full_grid_input")
    l1 = _plane_metrics(report, "encoder_layer_1")
    if first["delta"] == 0.0:
        delta_retention = 0.0
        residual_retention = 0.0
    else:
        delta_retention = l1["delta"] / first["delta"]
        residual_retention = l1["residual_energy"] / max(first["residual_energy"], 1e-12)
    intensity_loss = first["delta"] - l1["delta"]
    r_loss = first["r"] - l1["r"]
    self_cov_loss = (first["q"] + 2.0 * first["cov"]) - (l1["q"] + 2.0 * l1["cov"])
    real_explanation = r_loss / max(intensity_loss, 1e-12)
    self_cov_explanation = self_cov_loss / max(intensity_loss, 1e-12)
    q_share_increase = l1["quadrature"] - first["quadrature"]
    within_retention = l1["within_r"] / max(first["within_r"], 1e-12)
    coherence_drop = first["coherence"] - l1["coherence"]
    residual_anomaly = residual_retention < RETENTION_MIN
    decomposition_inconclusive = delta_retention >= 0.90
    quadrature = q_share_increase >= 0.10 and real_explanation >= 0.50
    cross_label = within_retention >= 0.90 and (l1["r"] / max(first["r"], 1e-12)) < 0.90 and coherence_drop >= 0.10
    self_term = self_cov_explanation >= 0.50
    if not provenance_pass:
        status = "harness-invalid"
    elif residual_anomaly:
        status = "complex-residual-transport-anomaly"
    elif decomposition_inconclusive:
        status = "phase-intensity-decomposition-inconclusive"
    elif sum((quadrature, cross_label, self_term)) >= 2:
        status = "mixed-carrier-interference-cancellation"
    elif quadrature:
        status = "carrier-residual-quadrature-rotation-confirmed"
    elif cross_label:
        status = "cross-label-interference-cancellation-confirmed"
    elif self_term:
        status = "residual-self-interference-cancellation-confirmed"
    else:
        status = "carrier-residual-interference-inconclusive"
    decision = {
        "schema": "v75_carrier_residual_interference_decision_v1", "diagnostic_only": True, "status": status,
        "first_transition": "encoder_full_grid_input->encoder_layer_1", "delta_intensity_full_retention": delta_retention,
        "residual_complex_full_retention": residual_retention, "quadrature_share_increase": q_share_increase,
        "real_interference_loss_explanation": real_explanation, "self_plus_covariance_loss_explanation": self_cov_explanation,
        "within_label_real_interference_retention": within_retention, "label_coherence_drop": coherence_drop,
        "mechanism_gates": {"quadrature_rotation": quadrature, "cross_label_cancellation": cross_label, "residual_self_interference": self_term},
        "provenance_pass": provenance_pass, "manual_readability_reused": readable, "planes": planes,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
        "next_action": {
            "complex-residual-transport-anomaly": "inspect transient 3072 crop/bandlimit; no training",
            "phase-intensity-decomposition-inconclusive": "repair replay/decomposition harness; no training",
            "carrier-residual-quadrature-rotation-confirmed": "design one fixed dual-quadrature coherent-reference upper bound; do not run yet",
            "cross-label-interference-cancellation-confirmed": "freeze and document cross-label cancellation; no training",
            "residual-self-interference-cancellation-confirmed": "freeze and document residual self-term mechanism; no training",
            "mixed-carrier-interference-cancellation": "freeze and document combined mechanisms; no training",
            "carrier-residual-interference-inconclusive": "freeze v75 and inspect mechanism attribution; no training",
            "harness-invalid": "repair and rerun the complete two-pass v75 audit",
        }[status],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v73-decision", required=True, type=Path)
    parser.add_argument("--v74-decision", required=True, type=Path)
    parser.add_argument("--v73-manual-review", required=True, type=Path)
    parser.add_argument("--v74-report", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(decide(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
