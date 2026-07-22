"""Joint v68.6 B code-conditioning/readability decision gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "v68_b_decision_v1"


def _finite(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _metrics_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "corr": float(metrics.get("clean_prototype_corr", float("nan"))) >= 0.70,
        "dice": float(metrics.get("clean_prototype_dice", float("nan"))) >= 0.68,
        "accuracy": float(metrics.get("prototype_accuracy", float("nan"))) >= 0.80,
        "high_saturation": float(metrics.get("high_saturation", float("nan"))) <= 0.08,
    }


def run(
    *,
    a2_decision: str | Path,
    warmstart_report: str | Path,
    training_history: str | Path,
    crossed_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    a2 = json.loads(Path(a2_decision).read_text())
    smoke = json.loads(Path(warmstart_report).read_text())
    crossed = json.loads(Path(crossed_report).read_text())
    history = json.loads(Path(training_history).read_text())
    manual = json.loads(Path(manual_review).read_text())
    provenance = {
        "a2": a2.get("status") == "a2-raw-readability-passed"
        and a2.get("b_allowed") is True,
        "warmstart": smoke.get("schema") == "v68_b_warmstart_smoke_v1"
        and smoke.get("status") == "b-warmstart-smoke-passed",
        "crossed": crossed.get("schema") == "v68_b_crossed_raw_cmos_v1"
        and crossed.get("status") == "b-crossed-audit-complete"
        and crossed.get("test_split_used") is False,
        "finite": bool(smoke.get("finite"))
        and bool(crossed.get("finite"))
        and _finite(crossed),
        "parameters_unchanged": bool(smoke.get("parameters_unchanged"))
        and bool(crossed.get("parameters_unchanged")),
    }
    rows = history.get("one_shot_distill_validation", [])
    rows = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: int(row.get("epoch", -1)),
    )
    last = rows[-2:] if len(rows) >= 2 else []
    history_gate = {
        "finite": len(last) == 2 and _finite(last),
        "epoch5_not_below_epoch4": len(last) == 2
        and all(
            float(last[-1].get(key, float("nan")))
            >= float(last[-2].get(key, float("nan")))
            for key in ("paired_corr", "paired_dice")
        ),
    }
    metric_gates = {
        name: _metrics_gate(crossed.get(name + "_metrics", {}))
        for name in ("crossed", "diagonal")
    }
    energy = crossed.get("energy_records_mean", {})
    energy_values = [
        float(v) for k, v in energy.items() if "guard" in k or "discarded" in k
    ]
    metric_gates["energy"] = {
        "pass": bool(energy_values) and max(energy_values) <= 1e-4
    }
    metric_gates["camera_code_factor"] = {
        "pass": float(crossed.get("camera_code_factor", float("nan"))) >= 0.25
    }
    manual_gate = {
        "readable_labels": manual.get("readable_labels"),
        "passed": manual.get("readable_labels") == list(range(10)),
        "notes": manual.get("notes", ""),
    }
    readability = (
        all(
            all(values.values())
            for name, values in metric_gates.items()
            if name in ("crossed", "diagonal")
        )
        and metric_gates["energy"]["pass"]
        and history_gate["epoch5_not_below_epoch4"]
        and manual_gate["passed"]
    )
    harness_invalid = not all(provenance.values()) or not history_gate["finite"]
    if harness_invalid:
        status = "harness-invalid"
    elif not readability:
        status = "b-readability-regression"
    elif not metric_gates["camera_code_factor"]["pass"]:
        status = "b-prototype-imaging-only"
    else:
        status = "b-code-conditioned-readability-passed"
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": status,
        "provenance_gate": provenance,
        "history_gate": history_gate,
        "metric_gates": metric_gates,
        "manual_gate": manual_gate,
        "generation_claim_allowed": status == "b-code-conditioned-readability-passed",
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "a2-decision",
        "warmstart-report",
        "training-history",
        "crossed-report",
        "manual-review",
        "output",
    ):
        p.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(run(**vars(p.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
