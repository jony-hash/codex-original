#!/usr/bin/env python3
"""v36 high-gain code-coupled training diagnostic report.

Aggregates v36 high-gain residual+bridge training probes, compares their
sample-time code coupling against v35, and decides whether a true CMOS preview
manual review is warranted. This tool is diagnostic-only and never writes an
accepted summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v35_optical_code_carrier_isolation import analyze_scale_run


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v36_high_gain_code_coupled_training_{timestamp}.json"


def _read_eval_metrics(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "artifacts" / "one_shot_distill_eval_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def _strict_pass(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return bool(
        metrics.get("strict_readability_acceptance", {}).get("passed")
        or metrics.get("strict_gate", {}).get("passed")
        or metrics.get("strict_readability", {}).get("passed")
    )


def _sample_metrics(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "artifacts" / "one_shot_sample_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def _classify_sample(scale_record: dict[str, Any]) -> tuple[str, str]:
    saturation = scale_record.get("slm_saturation", {})
    near_one = float(saturation.get("near_one_fraction") or 0.0)
    if near_one > 0.05:
        return "scale-saturation-regression", "SLM command is dominated by high-end clipping/saturation"
    status = scale_record.get("status")
    chain = scale_record.get("chain", {})
    if status in {"generator-command-too-weak", "front-ols-code-carrier-failure"}:
        return "code-coupling-regression", "trained high-gain path no longer carries code variation through OLS"
    if status == "ols-to-bridge-carrier-failure":
        return "decoder-topology-failure", "bridge/target-code plane collapsed before decoder review"
    if status == "decoder-after-code-bottleneck":
        return "decoder-topology-failure", "decoder output lost target/code topology"
    if status == "camera-after-code-bottleneck":
        return "camera-after-decoder-failure", "decoder plane varied but true CMOS did not"
    if status == "code-coupling-recovered-needs-human-review":
        physical = chain.get("physical_camera_plane", {})
        decoder = chain.get("decoder_output_amplitude", {})
        target = chain.get("target_code_amplitude", {})
        if target.get("active") and decoder.get("active") and not physical.get("active"):
            return "camera-after-decoder-failure", "target/decoder active but true CMOS inactive"
        return "high-gain-training-rescue-needs-human-review", "trained high-gain path reaches true CMOS preview"
    return "harness-invalid", f"unhandled sample status: {status!r}"


def analyze_probe(
    *,
    label: str,
    scale: float,
    target_mode: str,
    train_run: Path,
    same_eval_run: Path | None,
    test_eval_run: Path | None,
    sample_run: Path | None,
) -> dict[str, Any]:
    same_metrics = _read_eval_metrics(same_eval_run)
    test_metrics = _read_eval_metrics(test_eval_run)
    sample_payload = _sample_metrics(sample_run)
    if sample_run is None:
        return {
            "label": label,
            "scale": float(scale),
            "target_code_diagnostic_mode": target_mode,
            "train_run": train_run.as_posix(),
            "same_eval_run": same_eval_run.as_posix() if same_eval_run else None,
            "test_eval_run": test_eval_run.as_posix() if test_eval_run else None,
            "sample_run": None,
            "same_strict_gate_passed": _strict_pass(same_metrics),
            "test_strict_gate_passed": _strict_pass(test_metrics),
            "sample_metrics": sample_payload,
            "status": "harness-invalid",
            "reason": "missing sample run for high-gain CMOS review",
            "accepted_summary_written": False,
            "human_readability_required": True,
        }
    sample_record = analyze_scale_run(sample_run, scale=scale, label=label)
    status, reason = _classify_sample(sample_record)
    return {
        "label": label,
        "scale": float(scale),
        "target_code_diagnostic_mode": target_mode,
        "train_run": train_run.as_posix(),
        "same_eval_run": same_eval_run.as_posix() if same_eval_run else None,
        "test_eval_run": test_eval_run.as_posix() if test_eval_run else None,
        "sample_run": sample_run.as_posix(),
        "same_strict_gate_passed": _strict_pass(same_metrics),
        "test_strict_gate_passed": _strict_pass(test_metrics),
        "sample_metrics": sample_payload,
        "sample_code_coupling": sample_record,
        "status": status,
        "classification": status,
        "reason": reason,
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record.get("sample_run") or record["train_run"])
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
    elif by_status.get("high-gain-training-rescue-needs-human-review"):
        status = "high-gain-training-rescue-needs-human-review"
    elif by_status.get("camera-after-decoder-failure"):
        status = "camera-after-decoder-failure"
    elif by_status.get("decoder-topology-failure"):
        status = "decoder-topology-failure"
    elif by_status.get("scale-saturation-regression"):
        status = "scale-saturation-regression"
    elif by_status.get("code-coupling-regression"):
        status = "code-coupling-regression"
    else:
        status = "sample-reference-metric-invalid"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": (
            "human_v4_cmos_preview_review"
            if status == "high-gain-training-rescue-needs-human-review"
            else "diagnostic_followup_before_human_acceptance"
        ),
    }


def run_v36_high_gain_code_coupled_training(
    *,
    source_v35_report: Path | None = None,
    source_v33_1_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    source_v35_status = None
    if source_v35_report is not None:
        payload = _load_json(source_v35_report)
        assert payload is not None
        source_v35_status = payload.get("status")
    records = []
    errors = []
    for entry in probe_entries:
        parts = entry.split("|")
        while len(parts) < 7:
            parts.append("")
        label, scale, target_mode, train_run, same_eval_run, test_eval_run, sample_run = parts[:7]
        try:
            records.append(
                analyze_probe(
                    label=label,
                    scale=float(scale),
                    target_mode=target_mode,
                    train_run=Path(train_run),
                    same_eval_run=Path(same_eval_run) if same_eval_run else None,
                    test_eval_run=Path(test_eval_run) if test_eval_run else None,
                    sample_run=Path(sample_run) if sample_run else None,
                )
            )
        except Exception as exc:
            errors.append({"probe": label, "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label,
                    "scale": float(scale),
                    "target_code_diagnostic_mode": target_mode,
                    "train_run": train_run,
                    "same_eval_run": same_eval_run or None,
                    "test_eval_run": test_eval_run or None,
                    "sample_run": sample_run or None,
                    "status": "harness-invalid",
                    "reason": "probe analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": "v36",
        "objective": "high_gain_code_coupled_training_cmos_review",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v35_report": source_v35_report.as_posix() if source_v35_report else None,
        "source_v35_status": source_v35_status,
        "source_v33_1_summary": source_v33_1_summary.as_posix() if source_v33_1_summary else None,
        "probes": records,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane v4-style previews can satisfy human review.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v35-report", type=Path)
    parser.add_argument("--source-v33-1-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v36_high_gain_code_coupled_training(
        source_v35_report=args.source_v35_report,
        source_v33_1_summary=args.source_v33_1_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
