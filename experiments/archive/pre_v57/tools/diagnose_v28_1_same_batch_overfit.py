#!/usr/bin/env python3
"""v28.1 same-batch decoder overfit hard-gate diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v28_decoder_controllability import analyze_run, _load_oracle_baseline


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v28_1_same_batch_overfit_{timestamp}.json"


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _extract_tagged_runs(payload: Any, same_batch: list[Path], test: list[Path], sample: list[Path]) -> None:
    if isinstance(payload, dict):
        if payload.get("same_batch_eval_run"):
            same_batch.append(Path(str(payload["same_batch_eval_run"])))
        if payload.get("test_eval_run"):
            test.append(Path(str(payload["test_eval_run"])))
        if payload.get("sample_run"):
            sample.append(Path(str(payload["sample_run"])))
        for value in payload.values():
            _extract_tagged_runs(value, same_batch, test, sample)
    elif isinstance(payload, list):
        for item in payload:
            _extract_tagged_runs(item, same_batch, test, sample)


def _runs_from_summary(summary: Path) -> tuple[list[Path], list[Path], list[Path]]:
    payload = _load_json(summary)
    assert payload is not None
    same_batch: list[Path] = []
    test: list[Path] = []
    sample: list[Path] = []
    _extract_tagged_runs(payload, same_batch, test, sample)
    return _dedupe(same_batch), _dedupe(test), _dedupe(sample)


def _is_reachable(record: dict[str, Any]) -> bool:
    return record.get("classification") in {
        "decoder-topology-reachable",
        "camera-after-decoder-failure",
        "human-accepted-candidate-needs-v4-review",
    }


def _is_camera_failure(record: dict[str, Any]) -> bool:
    return record.get("classification") == "camera-after-decoder-failure"


def _is_probe_d(record: dict[str, Any]) -> bool:
    return "probe_d" in str(record.get("run", "")) or "phase_amplitude" in str(record.get("run", ""))


def _decision(same_records: list[dict[str, Any]], test_records: list[dict[str, Any]], sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    same_reachable = [record for record in same_records if _is_reachable(record)]
    test_reachable = [record for record in test_records if _is_reachable(record)]
    sample_positive = [
        record
        for record in sample_records
        if record.get("classification") == "human-accepted-candidate-needs-v4-review"
    ]
    if sample_positive:
        status = "human-accepted-candidate-needs-v4-review"
    elif any(_is_camera_failure(record) for record in same_reachable):
        status = "same-batch-camera-failure"
    elif same_reachable and all(_is_probe_d(record) for record in same_reachable):
        status = "phase-only-limited"
    elif same_reachable and not test_reachable:
        status = "same-batch-generalization-failure"
    elif same_reachable:
        status = "same-batch-decoder-reachable"
    else:
        status = "decoder-topology-unreachable"
    return {
        "status": status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "same_batch_reachable_runs": [record["run"] for record in same_reachable],
        "test_reachable_runs": [record["run"] for record in test_reachable],
        "sample_positive_runs": [record["run"] for record in sample_positive],
    }


def run_same_batch_overfit_gate(
    *,
    oracle_report: Path | None = None,
    summary: Path | None = None,
    same_batch_eval_runs: list[Path] | None = None,
    test_eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    same_runs = list(same_batch_eval_runs or [])
    test_runs = list(test_eval_runs or [])
    sample_run_paths = list(sample_runs or [])
    if summary is not None and not (same_runs or test_runs or sample_run_paths):
        same_runs, test_runs, sample_run_paths = _runs_from_summary(summary)
    same_runs = _dedupe(same_runs)
    test_runs = _dedupe(test_runs)
    sample_run_paths = _dedupe(sample_run_paths)
    if not same_runs:
        raise ValueError("at least one same-batch eval run is required")
    oracle_baseline = _load_oracle_baseline(oracle_report, summary)
    same_records = [analyze_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in same_runs]
    test_records = [analyze_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in test_runs]
    sample_records = [analyze_run(run, kind="sample", oracle_baseline=oracle_baseline) for run in sample_run_paths]
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v28.1",
        "objective": "same_batch_decoder_overfit_hard_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "oracle_report": oracle_report.as_posix() if oracle_report is not None else None,
        "oracle_baseline": oracle_baseline,
        "same_batch_eval_runs": [run.as_posix() for run in same_runs],
        "test_eval_runs": [run.as_posix() for run in test_runs],
        "sample_runs": [run.as_posix() for run in sample_run_paths],
        "same_batch_records": same_records,
        "test_records": test_records,
        "sample_records": sample_records,
        "decision": _decision(same_records, test_records, sample_records),
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--same-batch-eval-run", action="append", type=Path, default=None)
    parser.add_argument("--test-eval-run", action="append", type=Path, default=None)
    parser.add_argument("--sample-run", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_same_batch_overfit_gate(
        oracle_report=args.oracle_report,
        summary=args.summary,
        same_batch_eval_runs=args.same_batch_eval_run,
        test_eval_runs=args.test_eval_run,
        sample_runs=args.sample_run,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
