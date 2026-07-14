#!/usr/bin/env python3
"""v30.1 sample-gate recovery diagnostic.

This tool audits whether v30 target-code eval-positive probes actually reached
sample review.  It never writes accepted summaries; human readability remains a
manual v4 CMOS preview gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output


REACHABLE_EVAL = {
    "decoder-topology-reachable",
    "camera-after-decoder-failure",
    "human-accepted-candidate-needs-v4-review",
}
SAMPLE_CANDIDATE = "human-accepted-candidate-needs-v4-review"


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v30_1_sample_gate_recovery_{timestamp}.json"


def _preview_path(run: str | None, *, kind: str) -> str | None:
    if not run:
        return None
    filename = (
        "one_shot_sample_100x100_cell_chain_preview.png"
        if kind == "sample"
        else "one_shot_distill_eval_100x100_cell_chain_preview.png"
    )
    path = Path(run) / "images" / filename
    return path.as_posix() if path.is_file() else None


def _run_records(alignment_report: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(alignment_report)
    assert payload is not None
    return {
        str(record.get("run")): record
        for record in payload.get("runs", [])
        if isinstance(record, dict) and record.get("run")
    }


def _sample_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records.values() if record.get("run_type") == "sample"]


def _is_non_oracle_ab(probe: dict[str, Any]) -> bool:
    label = str(probe.get("label", ""))
    arch = str(probe.get("optical_core_arch", ""))
    mode = str(probe.get("target_code_diagnostic_mode", ""))
    return (
        arch != "camera_direct_clean_oracle"
        and (label.startswith("probe_a_") or label.startswith("probe_b_"))
        and mode in {"spatial_identity_code", "lowfreq_digit_code"}
    )


def run_v30_1_sample_gate_recovery(
    *,
    summary: Path,
    alignment_report: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    summary_payload = _load_json(summary)
    assert summary_payload is not None
    records = _run_records(alignment_report)
    probes = [probe for probe in summary_payload.get("probes", []) if isinstance(probe, dict)]

    sample_records = _sample_records(records)
    sample_positive_runs = [
        str(record["run"])
        for record in sample_records
        if record.get("classification") == SAMPLE_CANDIDATE
    ]
    metric_positive_runs = [
        run
        for run, record in records.items()
        if record.get("classification") in REACHABLE_EVAL
    ]

    eligible_probes = []
    missing_or_failed_samples = []
    excluded_probes = []
    for probe in probes:
        if not _is_non_oracle_ab(probe):
            excluded_probes.append(
                {
                    "label": probe.get("label"),
                    "reason": "not-non-oracle-v30-a-b",
                    "sample_run": probe.get("sample_run"),
                }
            )
            continue
        same_run = str(probe.get("same_batch_eval_run") or "")
        test_run = str(probe.get("test_eval_run") or "")
        sample_run = str(probe.get("sample_run") or "") if probe.get("sample_run") else None
        same_class = records.get(same_run, {}).get("classification")
        test_class = records.get(test_run, {}).get("classification")
        sample_class = records.get(sample_run, {}).get("classification") if sample_run else None
        eval_positive = (
            bool(probe.get("same_batch_strict_gate_passed"))
            and bool(probe.get("test_strict_gate_passed"))
            and same_class in REACHABLE_EVAL
            and test_class in REACHABLE_EVAL
        )
        entry = {
            "label": probe.get("label"),
            "same_batch_eval_run": same_run,
            "same_batch_classification": same_class,
            "test_eval_run": test_run,
            "test_classification": test_class,
            "sample_run": sample_run,
            "sample_classification": sample_class,
            "eval_positive": eval_positive,
            "same_preview": _preview_path(same_run, kind="eval"),
            "test_preview": _preview_path(test_run, kind="eval"),
            "sample_preview": _preview_path(sample_run, kind="sample"),
        }
        if eval_positive:
            eligible_probes.append(entry)
            if sample_class != SAMPLE_CANDIDATE:
                missing_or_failed_samples.append(entry)

    human_readability_candidate_runs = sample_positive_runs
    if human_readability_candidate_runs:
        status = "v30-sample-metric-candidate-needs-human-review"
    elif missing_or_failed_samples:
        status = "eval-positive-sample-failed"
    elif eligible_probes:
        status = "eval-positive-sample-failed"
    else:
        status = "no-sample-eligible"

    payload = {
        "version": "v30.1",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": summary.as_posix(),
        "alignment_report": alignment_report.as_posix(),
        "status": status,
        "human_readability_required": True,
        "metric_positive_runs": metric_positive_runs,
        "human_readability_candidate_runs": human_readability_candidate_runs,
        "eval_positive_sample_failed_runs": [
            entry["sample_run"] or entry["test_eval_run"] for entry in missing_or_failed_samples
        ],
        "eligible_probes": eligible_probes,
        "excluded_probes": excluded_probes,
        "decision": {
            "status": status,
            "accepted_summary_written": False,
            "human_readability_required": True,
            "metric_positive_runs": metric_positive_runs,
            "human_readability_candidate_runs": human_readability_candidate_runs,
            "eval_positive_sample_failed_runs": [
                entry["sample_run"] or entry["test_eval_run"] for entry in missing_or_failed_samples
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v30_1_sample_gate_recovery(
        summary=args.summary,
        alignment_report=args.alignment_report,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
