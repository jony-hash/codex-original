"""Joint decision for the diagnostic v71 physical SLM-intensity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    v69_decision: str | Path,
    v70_decision: str | Path,
    isolation_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prior = _read(v69_decision)
    v70 = _read(v70_decision)
    report = _read(isolation_report)
    manual = _read(manual_review)
    labels = sorted(int(item) for item in manual.get("readable_labels", []))
    provenance_pass = (
        prior.get("status") == "v69-prototype-only"
        and v70.get("status") == "post-slm-code-factor-attenuated"
        and report.get("status") == "v71-slm-intensity-isolation-complete"
        and report.get("diagnostic_only") is True
        and report.get("test_split_used") is False
        and report.get("return_optical_fields") is False
        and report.get("batch_size") == 1
        and report.get("calibration_semantics") == "slm_amplitude_squared"
        and report.get("parameters_unchanged") is True
        and report.get("optimizer_created") is False
        and report.get("parameter_update_count") == 0
        and report.get("finite") is True
        and report.get("energy_pass") is True
        and report.get("calibration", {}).get("finite") is True
        and float(report.get("calibration", {}).get("absolute_error", 1.0)) <= 1e-4
        and abs(float(report.get("calibration", {}).get("achieved_code_fraction", 0.0)) - 0.25) <= 1e-4
    )
    metrics_pass = _metrics_pass(report.get("crossed_metrics", {})) and _metrics_pass(report.get("diagonal_metrics", {}))
    readable_pass = labels == list(range(10))
    factors = report.get("intensity_code_factors", {})
    camera = float(factors.get("camera", 0.0))
    first_below = report.get("first_intensity_plane_below_25_percent")
    pre = float(factors.get("pre_encoder", 0.0))
    chip = float(factors.get("chip", 0.0))
    if not provenance_pass:
        status = "harness-invalid"
    elif camera >= 0.25 and metrics_pass and readable_pass:
        status = "physical-slm-to-camera-transport-capable"
    elif camera >= 0.25:
        status = "physical-slm-code-readability-tradeoff"
    elif pre < 0.25:
        status = "front-relay-code-factor-attenuated"
    elif chip < 0.25:
        status = "d2nn-code-factor-attenuated"
    else:
        status = "camera-relay-code-factor-attenuated"
    result = {
        "schema": "v71_slm_intensity_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "v69_status": prior.get("status"),
        "v70_status": v70.get("status"),
        "camera_code_factor": camera,
        "first_intensity_plane_below_25_percent": first_below,
        "provenance_pass": provenance_pass,
        "metrics_pass": metrics_pass,
        "manual_readability_pass": readable_pass,
        "readable_labels": labels,
        "complex_factor_decomposition": report.get("complex_factor_decomposition", {}),
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v69-decision", "v70-decision", "isolation-report", "manual-review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(decide(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
