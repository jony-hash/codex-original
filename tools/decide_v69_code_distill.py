"""Joint raw-CMOS acceptance decision for experimental v69 code distillation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "v69_code_distill_decision_v1"


def _finite(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _image_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "corr": float(metrics.get("clean_prototype_corr", float("nan"))) >= 0.70,
        "dice": float(metrics.get("clean_prototype_dice", float("nan"))) >= 0.68,
        "accuracy": float(metrics.get("prototype_accuracy", float("nan"))) >= 0.80,
        "high_saturation": float(metrics.get("high_saturation", float("nan"))) <= 0.08,
    }


def run(
    *,
    postmortem: str | Path,
    warmstart_report: str | Path,
    training_history: str | Path,
    crossed_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    closure = _read(postmortem)
    smoke = _read(warmstart_report)
    history = _read(training_history)
    crossed = _read(crossed_report)
    manual = _read(manual_review)
    provenance = {
        "v68_closed": closure.get("status")
        == "generator-injection-contract-insufficient"
        and closure.get("v68_closed") is True,
        "warmstart": smoke.get("schema") == "v69_code_delta_warmstart_v1"
        and smoke.get("status") == "v69-warmstart-smoke-passed",
        "crossed": crossed.get("schema") == "v69_crossed_raw_cmos_v1"
        and crossed.get("status") == "v69-crossed-audit-complete"
        and crossed.get("test_split_used") is False
        and crossed.get("return_optical_fields") is False,
        "finite": bool(smoke.get("finite"))
        and bool(crossed.get("finite"))
        and _finite(crossed),
        "frozen_parameters": bool(smoke.get("frozen_hash_unchanged"))
        and bool(smoke.get("parameters_unchanged"))
        and bool(crossed.get("parameters_unchanged")),
    }
    rows = sorted(
        (
            row
            for row in history.get("one_shot_distill_validation", [])
            if isinstance(row, dict)
        ),
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
    image = {
        name: _image_gate(crossed.get(name + "_metrics", {}))
        for name in ("crossed", "diagonal")
    }
    energy_values = [
        float(value)
        for key, value in crossed.get("energy_records_mean", {}).items()
        if "guard" in key or "discarded" in key
    ]
    energy_gate = bool(energy_values) and max(energy_values) <= 1e-4
    code_gate = float(crossed.get("camera_code_factor", float("nan"))) >= 0.25
    alignment = crossed.get("paired_teacher_alignment", {})
    teacher_gate = {
        "corr": float(alignment.get("diagonal_teacher_corr", float("nan"))) >= 0.70,
        "dice": float(alignment.get("diagonal_teacher_dice", float("nan"))) >= 0.68,
        "advantage_mean": float(alignment.get("teacher_advantage_mean", float("nan")))
        > 0.0,
        "advantage_ci95_lower": float(
            alignment.get("teacher_advantage_ci95_lower", float("nan"))
        )
        > 0.0,
        "bootstrap_contract": alignment.get("bootstrap_seed") == 1234
        and alignment.get("bootstrap_resamples") == 10_000,
    }
    manual_gate = {
        "readable_labels": manual.get("readable_labels"),
        "passed": manual.get("readable_labels") == list(range(10)),
        "notes": manual.get("notes", ""),
    }
    readable = (
        all(all(values.values()) for values in image.values())
        and energy_gate
        and history_gate["epoch5_not_below_epoch4"]
        and manual_gate["passed"]
    )
    harness_invalid = not all(provenance.values()) or not history_gate["finite"]
    if harness_invalid:
        status = "harness-invalid"
    elif not readable:
        status = "v69-readability-regression"
    elif not code_gate:
        status = "v69-prototype-only"
    elif not all(teacher_gate.values()):
        status = "v69-code-transport-unaligned"
    else:
        status = "v69-code-distillation-passed"
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": status,
        "provenance_gate": provenance,
        "history_gate": history_gate,
        "image_gates": image,
        "energy_gate": energy_gate,
        "camera_code_gate": code_gate,
        "teacher_alignment_gate": teacher_gate,
        "manual_gate": manual_gate,
        "generation_claim_allowed": status == "v69-code-distillation-passed",
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "postmortem",
        "warmstart-report",
        "training-history",
        "crossed-report",
        "manual-review",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
