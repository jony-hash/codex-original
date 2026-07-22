"""Joint decision for the read-only v77 phase-gradient compatibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MIN_PROJECTED_RATIO = 0.25
GRADIENT_TOLERANCE = 1.0e-12


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid_audit(report: dict[str, Any], *, smoke: dict[str, Any]) -> bool:
    required = {
        "diagnostic_only": True,
        "return_optical_fields": False,
        "test_split_used": False,
        "finite": True,
        "parameters_unchanged": True,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "full_field_saved": False,
        "persistent_full_field_saved": False,
        "teacher_alignment_computed": False,
        "coherent_receiver_branch_closed": True,
        "learned_code_transport_proven": False,
    }
    return report.get("schema") == "v77_phase_gradient_compatibility_audit_v1" and all(report.get(k) == v for k, v in required.items()) and smoke.get("status") == "v77-phase-gradient-smoke-passed"


def decide(*, v73_admission: str | Path, v73_report: str | Path, v73_decision: str | Path,
           v73_manual_review: str | Path, v75_admission: str | Path, v75_report: str | Path,
           v75_decision: str | Path, v76_admission: str | Path, v76_report: str | Path,
           v76_decision: str | Path, admission: str | Path, smoke_report: str | Path,
           audit_report: str | Path, output: str | Path) -> dict[str, Any]:
    prior = {"v73_admission": _read(v73_admission), "v73_report": _read(v73_report), "v73_decision": _read(v73_decision),
             "manual": _read(v73_manual_review), "v75_admission": _read(v75_admission), "v75_report": _read(v75_report),
             "v75_decision": _read(v75_decision), "v76_admission": _read(v76_admission), "v76_report": _read(v76_report),
             "v76_decision": _read(v76_decision), "admission": _read(admission), "smoke": _read(smoke_report), "audit": _read(audit_report)}
    manual_readable = sorted(int(x) for x in prior["manual"].get("readable_labels", [])) == list(range(10))
    provenance = (
        prior["v73_admission"].get("status") == "v73-cpu-admission-passed"
        and prior["v73_report"].get("status") == "v73-semantic-support-audit-complete"
        and prior["v73_decision"].get("status") == "semantic-support-code-factor-attenuated"
        and prior["v75_admission"].get("status") == "v75-interference-admission-passed"
        and prior["v75_report"].get("status") == "v75-carrier-residual-interference-complete"
        and prior["v75_decision"].get("status") == "carrier-residual-quadrature-rotation-confirmed"
        and prior["v76_admission"].get("status") == "v76-dual-quadrature-admission-passed"
        and prior["v76_report"].get("status") == "v76-dual-quadrature-readout-complete"
        and prior["v76_decision"].get("status") == "dual-quadrature-dynamic-range-tradeoff"
        and prior["admission"].get("status") == "v77-phase-gradient-admission-passed"
        and manual_readable
        and _valid_audit(prior["audit"], smoke=prior["smoke"])
    )
    metrics = prior["audit"].get("gradient_metrics", {})
    layer_names = ("encoder_layer_1", "encoder_layer_2", "decoder_layer_1", "decoder_layer_2")
    finite_metrics = all(bool(metrics.get(name, {}).get("finite", False)) for name in (*layer_names, "joint"))
    nonzero = all(float(metrics.get(name, {}).get("code_gradient_norm", 0.0)) > GRADIENT_TOLERANCE and not bool(metrics.get(name, {}).get("code_gradient_suppressed", True)) for name in (*layer_names, "joint"))
    joint_ratio = float(metrics.get("joint", {}).get("projected_code_drive_ratio", 0.0))
    encoder_l1_ratio = float(metrics.get("encoder_layer_1", {}).get("projected_code_drive_ratio", 0.0))
    readability_nonincrease = all(float(metrics.get(name, {}).get("first_order_readability_change", 1.0)) <= 1.0e-8 for name in (*layer_names, "joint"))
    if not provenance or not finite_metrics or not bool(prior["audit"].get("parameters_unchanged")):
        status = "harness-invalid"
    elif not nonzero:
        status = "phase-code-gradient-suppressed"
    elif joint_ratio < MIN_PROJECTED_RATIO:
        status = "phase-readability-gradient-conflict"
    elif encoder_l1_ratio < MIN_PROJECTED_RATIO:
        status = "encoder-l1-phase-control-insufficient"
    elif not readability_nonincrease:
        status = "phase-readability-gradient-conflict"
    else:
        status = "phase-aware-local-co-design-admissible"
    decision = {
        "schema": "v77_phase_gradient_decision_v1", "diagnostic_only": True, "status": status,
        "provenance_pass": provenance, "finite_metrics_pass": finite_metrics, "phase_gradient_nonzero": nonzero,
        "joint_projected_code_drive_ratio": joint_ratio, "encoder_l1_projected_code_drive_ratio": encoder_l1_ratio,
        "readability_first_order_nonincrease": readability_nonincrease, "gradient_metrics": metrics,
        "coherent_receiver_branch_closed": True, "learned_code_transport_proven": False,
        "next_action": {
            "phase-aware-local-co-design-admissible": "design one experimental semantic-gated direct-command residual plus phase joint-training contract; do not train yet",
            "phase-code-gradient-suppressed": "freeze phase co-design; perform architecture-level review without receiver or phase sweeps",
            "phase-readability-gradient-conflict": "freeze phase co-design; do not trade readability for code factor",
            "encoder-l1-phase-control-insufficient": "freeze local phase co-design; review upstream generator/injection contract",
            "harness-invalid": "repair and rerun complete v77 admission, smoke and audit",
        }[status],
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v73_admission", "v73_report", "v73_decision", "v73_manual_review", "v75_admission", "v75_report", "v75_decision", "v76_admission", "v76_report", "v76_decision", "admission", "smoke_report", "audit_report", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    values = vars(args)
    output = values.pop("output")
    result = decide(output=output, **values)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
