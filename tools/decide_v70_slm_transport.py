"""Decide the diagnostic v70 post-SLM transport isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_pass(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("clean_prototype_corr", 0.0)) >= 0.70
        and float(metrics.get("clean_prototype_dice", 0.0)) >= 0.68
        and float(metrics.get("prototype_accuracy", 0.0)) >= 0.80
        and float(metrics.get("high_saturation_fraction", metrics.get("high_saturation", 1.0))) <= 0.08
    )


def decide(
    *,
    v69_decision: str | Path,
    v69_crossed_report: str | Path,
    isolation_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prior = _read(v69_decision)
    crossed = _read(v69_crossed_report)
    report = _read(isolation_report)
    manual = _read(manual_review)
    labels = sorted(int(x) for x in manual.get("readable_labels", []))
    provenance_ok = (
        prior.get("status") == "v69-prototype-only"
        and crossed.get("status") == "v69-crossed-audit-complete"
        and report.get("status") == "v70-slm-intervention-complete"
        and report.get("diagnostic_only") is True
        and report.get("test_split_used") is False
        and report.get("return_optical_fields") is False
        and report.get("batch_size") == 1
        and report.get("parameters_unchanged") is True
        and report.get("optimizer_created") is False
        and report.get("parameter_update_count") == 0
        and report.get("finite") is True
        and report.get("energy_pass") is True
        and report.get("calibration", {}).get("finite") is True
        and float(report.get("calibration", {}).get("absolute_error", 1.0)) <= 1e-4
        and abs(float(report.get("slm_code_factor", 0.0)) - 0.25) <= 1e-4
    )
    metrics_ok = _metric_pass(report.get("crossed_metrics", {})) and _metric_pass(report.get("diagonal_metrics", {}))
    readable = labels == list(range(10))
    camera_factor = float(report.get("camera_code_factor", 0.0))
    if not provenance_ok:
        status = "harness-invalid"
    elif camera_factor >= 0.25 and metrics_ok and readable:
        status = "post-slm-code-transport-capable"
    elif camera_factor >= 0.25:
        status = "post-slm-code-readability-tradeoff"
    else:
        status = "post-slm-code-factor-attenuated"
    result = {
        "schema": "v70_slm_transport_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "v69_status": prior.get("status"),
        "camera_code_factor": camera_factor,
        "slm_code_factor": report.get("slm_code_factor"),
        "first_plane_below_25_percent": report.get("first_plane_below_25_percent"),
        "provenance_pass": provenance_ok,
        "metrics_pass": metrics_ok,
        "manual_readability_pass": readable,
        "readable_labels": labels,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v69-decision", "v69-crossed-report", "isolation-report", "manual-review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(decide(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
