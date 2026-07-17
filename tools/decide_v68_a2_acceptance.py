"""Joint v68.5 A2 raw-CMOS, history and manual-readability decision gate.

The tool never edits a checkpoint and never infers manual readability from a
correlation score.  ``manual_review`` must explicitly list every readable
digit, in the range 0..9, after inspecting the fixed-exposure raw preview.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "v68_a2_decision_v1"


def _last_two_validation(history: dict[str, Any]) -> list[dict[str, Any]]:
    records = history.get("one_shot_distill_validation")
    if not isinstance(records, list):
        raise ValueError("training history lacks one_shot_distill_validation")
    rows = [row for row in records if isinstance(row, dict) and "epoch" in row]
    rows.sort(key=lambda row: int(row["epoch"]))
    if len(rows) < 2:
        raise ValueError("A2 decision requires validation records for epochs 4 and 5")
    return rows[-2:]


def _history_gate(history: dict[str, Any]) -> dict[str, Any]:
    rows = _last_two_validation(history)
    # paired metrics are the training contract's stability metrics.  The raw
    # audit below remains the source of the locked clean-prototype gate.
    corr = [float(row.get("paired_corr", float("nan"))) for row in rows]
    dice = [float(row.get("paired_dice", float("nan"))) for row in rows]
    finite = all(value == value and abs(value) != float("inf") for value in (*corr, *dice))
    return {
        "epochs": [int(row["epoch"]) for row in rows],
        "paired_corr": corr,
        "paired_dice": dice,
        "finite": finite,
        "epoch5_not_below_epoch4": bool(finite and corr[-1] >= corr[-2] and dice[-1] >= dice[-2]),
    }


def _manual_gate(path: str | Path) -> dict[str, Any]:
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = review.get("readable_labels")
    valid = labels == list(range(10))
    return {
        "readable_labels": labels,
        "required_labels": list(range(10)),
        "passed": bool(valid),
        "notes": review.get("notes", ""),
    }


def _linked_gate(
    path: str | Path | None,
    *,
    accepted_statuses: tuple[str, ...] = (),
    accepted_decisions: tuple[str, ...] = (),
) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": False, "reason": "linked report required"}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    status = payload.get("status")
    decision = payload.get("decision")
    passed = status in accepted_statuses or decision in accepted_decisions
    if "finite" in payload:
        passed = passed and bool(payload["finite"])
    if "gradient_pass" in payload:
        passed = passed and bool(payload["gradient_pass"])
    if "memory_pass" in payload:
        passed = passed and bool(payload["memory_pass"])
    return {
        "provided": True,
        "passed": passed,
        "status": status,
        "decision": decision,
        "path": str(path),
    }


def run(
    *,
    training_history: str | Path,
    raw_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
    warmstart_report: str | Path | None = None,
    numerical_report: str | Path | None = None,
    training_admission_report: str | Path | None = None,
) -> dict[str, Any]:
    history = json.loads(Path(training_history).read_text(encoding="utf-8"))
    raw = json.loads(Path(raw_report).read_text(encoding="utf-8"))
    history_gate = _history_gate(history)
    manual_gate = _manual_gate(manual_review)
    metrics = raw.get("metrics", {})
    clean = metrics.get("clean_prototype", {})
    energy = raw.get("energy_records_mean", {})
    energy_values = [
        float(value)
        for key, value in energy.items()
        if ("guard" in key or "discarded" in key) and isinstance(value, (int, float))
    ]
    finite = bool(raw.get("finite", True))
    raw_gate = {
        "clean_corr": float(clean.get("fixed_exposure_corr", float("nan"))) >= 0.70,
        "clean_dice": float(clean.get("fixed_exposure_dice", float("nan"))) >= 0.68,
        "prototype_accuracy": float(metrics.get("prototype_label_accuracy", float("nan"))) >= 0.80,
        "high_saturation": float(metrics.get("high_saturation_ratio", float("nan"))) <= 0.08,
        "energy_guard_discarded": bool(energy_values) and max(energy_values) <= 1.0e-4,
        "automated_report_passed": bool(raw.get("automated_gate_passed", False)),
        "finite": finite,
    }
    linked = {
        "warmstart": _linked_gate(
            warmstart_report, accepted_decisions=("a2-warmstart-smoke-passed",)
        ),
        "numerical": _linked_gate(
            numerical_report, accepted_statuses=("numerical-converged",)
        ),
        "training_admission": _linked_gate(
            training_admission_report,
            accepted_decisions=("ready-for-a-training",),
        ),
    }
    decision = bool(all(raw_gate.values()) and history_gate["epoch5_not_below_epoch4"] and manual_gate["passed"] and all(item["passed"] for item in linked.values()))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "a2-raw-readability-passed" if decision else "a2-readability-failed",
        "raw_gate": raw_gate,
        "history_gate": history_gate,
        "manual_gate": manual_gate,
        "linked_reports": linked,
        "b_allowed": decision,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "failure_policy": "stop-and-run-readout-isolation" if not decision else "proceed-to-b-crossed-code-audit",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-history", required=True, type=Path)
    parser.add_argument("--raw-report", required=True, type=Path)
    parser.add_argument("--manual-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmstart-report", type=Path)
    parser.add_argument("--numerical-report", type=Path)
    parser.add_argument("--training-admission-report", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
