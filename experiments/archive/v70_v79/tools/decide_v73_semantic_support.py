"""Joint decision for the diagnostic v73 semantic-support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGET = 0.25


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics_pass(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("clean_prototype_corr", 0.0)) >= 0.70
        and float(metrics.get("clean_prototype_dice", 0.0)) >= 0.68
        and float(metrics.get("prototype_accuracy", 0.0)) >= 0.80
        and float(metrics.get("high_saturation", 1.0)) <= 0.08
    )


def decide(
    *,
    v72_decision: str | Path,
    v72_report: str | Path,
    cpu_admission: str | Path,
    audit_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    v72 = _read(v72_decision)
    prior_report = _read(v72_report)
    admission = _read(cpu_admission)
    report = _read(audit_report)
    manual = _read(manual_review)
    readable_labels = sorted(int(label) for label in manual.get("readable_labels", []))
    provenance_pass = (
        v72.get("status") == "phase-readability-code-tradeoff"
        and prior_report.get("status") == "v72-phase-attribution-complete"
        and admission.get("status") == "v73-cpu-admission-passed"
        and report.get("status") == "v73-semantic-support-audit-complete"
        and report.get("diagnostic_only") is True
        and report.get("test_split_used") is False
        and report.get("return_optical_fields") is False
        and report.get("full_field_saved") is False
        and report.get("active_view_only") is True
        and report.get("batch_size") == 1
        and report.get("finite") is True
        and report.get("energy_pass") is True
        and report.get("parameters_unchanged") is True
        and report.get("optimizer_created") is False
        and report.get("parameter_update_count") == 0
        and report.get("teacher_alignment_computed") is False
        and report.get("calibration", {}).get("absolute_error", 1.0) <= 1.0e-4
        and report.get("calibration", {}).get("finite") is True
        and report.get("checkpoint_boundary_contract") == "ideal_transparent_thin_phase_v2"
    )
    auto_pass = _metrics_pass(report.get("crossed_metrics", {})) and _metrics_pass(report.get("diagonal_metrics", {}))
    manual_pass = readable_labels == list(range(10))
    factor = float(report.get("camera_code_factor", 0.0))
    factor_pass = factor >= TARGET
    absolute_code_pass = bool(report.get("absolute_code_ss_pass", False))
    label_retention_pass = bool(report.get("label_ss_retention_pass", False))
    anti_ratio_pass = absolute_code_pass and label_retention_pass
    readability_pass = auto_pass and manual_pass
    if not provenance_pass:
        status = "harness-invalid"
    elif not factor_pass:
        status = "semantic-support-code-factor-attenuated"
    elif not readability_pass:
        status = "semantic-support-readability-tradeoff"
    elif not anti_ratio_pass:
        status = "semantic-support-factor-ratio-artifact"
    else:
        status = "semantic-support-code-transport-capable"
    report_factor = report.get("intensity_code_factors", {})
    first_below = report.get("first_intensity_plane_below_25_percent")
    decision = {
        "schema": "v73_semantic_support_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "v72_status": v72.get("status"),
        "camera_code_factor": factor,
        "camera_factor_pass": factor_pass,
        "camera_absolute_code_effect_ss": report.get("camera_code_effect_ss"),
        "camera_absolute_code_ss_pass": absolute_code_pass,
        "camera_label_effect_ss": report.get("camera_label_effect_ss"),
        "camera_label_ss_retention": report.get("label_ss_retention"),
        "camera_label_ss_retention_pass": label_retention_pass,
        "anti_ratio_artifact_pass": anti_ratio_pass,
        "auto_readability_pass": auto_pass,
        "manual_readability_pass": manual_pass,
        "readable_labels": readable_labels,
        "first_intensity_plane_below_25_percent": first_below,
        "intensity_code_factors": report_factor,
        "partial_absolute_code_improvement_vs_v72": float(report.get("camera_code_effect_ss", 0.0)) > float(report.get("v72_baseline_camera_code_effect_ss", 0.0)),
        "provenance_pass": provenance_pass,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v72-decision", required=True, type=Path)
    parser.add_argument("--v72-report", required=True, type=Path)
    parser.add_argument("--cpu-admission", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--manual-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = decide(**vars(parser.parse_args()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
