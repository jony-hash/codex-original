#!/usr/bin/env python3
"""v29 decoder trainability / target-encoding diagnostic."""

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
    return Path("outputs/reports") / f"twophoton_100x100_v29_decoder_trainability_target_encoding_{timestamp}.json"


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _extract_tagged_runs(
    payload: Any,
    train_runs: list[Path],
    same_eval_runs: list[Path],
    test_eval_runs: list[Path],
    sample_runs: list[Path],
) -> None:
    if isinstance(payload, dict):
        if payload.get("train_run"):
            train_runs.append(Path(str(payload["train_run"])))
        if payload.get("same_batch_eval_run"):
            same_eval_runs.append(Path(str(payload["same_batch_eval_run"])))
        if payload.get("test_eval_run"):
            test_eval_runs.append(Path(str(payload["test_eval_run"])))
        if payload.get("sample_run"):
            sample_runs.append(Path(str(payload["sample_run"])))
        for value in payload.values():
            _extract_tagged_runs(value, train_runs, same_eval_runs, test_eval_runs, sample_runs)
    elif isinstance(payload, list):
        for item in payload:
            _extract_tagged_runs(item, train_runs, same_eval_runs, test_eval_runs, sample_runs)


def _runs_from_summary(summary: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    payload = _load_json(summary)
    assert payload is not None
    train_runs: list[Path] = []
    same_eval_runs: list[Path] = []
    test_eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    _extract_tagged_runs(payload, train_runs, same_eval_runs, test_eval_runs, sample_runs)
    return _dedupe(train_runs), _dedupe(same_eval_runs), _dedupe(test_eval_runs), _dedupe(sample_runs)


def _run_label(path: Path) -> str:
    name = path.name
    marker = "_train_v29_"
    if marker in name:
        return name.split(marker, 1)[1]
    for prefix in ("v29_", "train_v29_"):
        if prefix in name:
            return name.split(prefix, 1)[1]
    return name


def _load_trainability_audit(train_run: Path) -> dict[str, Any]:
    artifact = train_run / "artifacts" / "one_shot_decoder_trainability_audit.json"
    if artifact.is_file():
        payload = _load_json(artifact)
        assert isinstance(payload, dict)
        payload["source"] = artifact.as_posix()
        return payload
    meta = _load_json(train_run / "run_meta.json", required=False)
    if isinstance(meta, dict) and isinstance(meta.get("one_shot_decoder_trainability_audit"), dict):
        payload = dict(meta["one_shot_decoder_trainability_audit"])
        payload["source"] = (train_run / "run_meta.json").as_posix()
        return payload
    return {
        "enabled": False,
        "passed": None,
        "failure_reasons": ["missing-trainability-audit"],
        "source": None,
    }


def _load_train_config(train_run: Path) -> dict[str, Any]:
    config = _load_json(train_run / "configs" / "train_config.json", required=False)
    return config if isinstance(config, dict) else {}


def _target_preview_paths(train_run: Path) -> list[str]:
    image_dir = train_run / "images" / "one_shot_distill"
    return [path.as_posix() for path in sorted(image_dir.glob("*_aux_topology_target.png"))]


def _is_reachable(record: dict[str, Any]) -> bool:
    return record.get("classification") in {
        "decoder-topology-reachable",
        "camera-after-decoder-failure",
        "human-accepted-candidate-needs-v4-review",
    }


def _label_matches(record: dict[str, Any], *needles: str) -> bool:
    run = str(record.get("run", ""))
    return any(needle in run for needle in needles)


def _decision(train_records: list[dict[str, Any]], same_records: list[dict[str, Any]], sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    harness_failures = [
        record
        for record in train_records
        if record.get("trainability_audit", {}).get("passed") is False
        or (
            record.get("trainability_required")
            and record.get("trainability_audit", {}).get("enabled") is not True
        )
    ]
    reachable = [record for record in same_records if _is_reachable(record)]
    target_rescue = [
        record
        for record in reachable
        if _label_matches(record, "decoder_lowpass_target", "decoder_thick_binary_target")
    ]
    depth_rescue = [
        record
        for record in reachable
        if _label_matches(record, "decoder_4layer_phase_only")
    ]
    phase_rescue = [
        record
        for record in reachable
        if _label_matches(record, "decoder_4layer_phase_amplitude")
    ]
    sample_positive = [
        record
        for record in sample_records
        if record.get("classification") == "human-accepted-candidate-needs-v4-review"
    ]
    if harness_failures:
        status = "harness-failure"
    elif sample_positive:
        status = "human-accepted-candidate-needs-v4-review"
    elif target_rescue:
        status = "target-encoding-rescue"
    elif depth_rescue:
        status = "depth-limited"
    elif phase_rescue:
        status = "phase-amplitude-limited"
    elif same_records and all(record.get("classification") == "metric-proxy-failure" for record in same_records):
        status = "metric-proxy-failure"
    else:
        status = "architecture-target-unreachable"
    return {
        "status": status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "harness_failure_runs": [record["train_run"] for record in harness_failures],
        "target_encoding_rescue_runs": [record["run"] for record in target_rescue],
        "depth_limited_runs": [record["run"] for record in depth_rescue],
        "phase_amplitude_limited_runs": [record["run"] for record in phase_rescue],
        "sample_positive_runs": [record["run"] for record in sample_positive],
        "same_batch_reachable_runs": [record["run"] for record in reachable],
    }


def run_v29_decoder_trainability_target_encoding(
    *,
    oracle_report: Path | None = None,
    summary: Path | None = None,
    train_runs: list[Path] | None = None,
    same_batch_eval_runs: list[Path] | None = None,
    test_eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_train = list(train_runs or [])
    resolved_same = list(same_batch_eval_runs or [])
    resolved_test = list(test_eval_runs or [])
    resolved_sample = list(sample_runs or [])
    if summary is not None and not (resolved_train or resolved_same or resolved_test or resolved_sample):
        resolved_train, resolved_same, resolved_test, resolved_sample = _runs_from_summary(summary)
    resolved_train = _dedupe(resolved_train)
    resolved_same = _dedupe(resolved_same)
    resolved_test = _dedupe(resolved_test)
    resolved_sample = _dedupe(resolved_sample)
    if not resolved_same:
        raise ValueError("at least one same-batch eval run is required")
    oracle_baseline = _load_oracle_baseline(oracle_report, summary)
    train_records = []
    for run in resolved_train:
        config = _load_train_config(run)
        one_shot = config.get("one_shot", {}) if isinstance(config, dict) else {}
        label = _run_label(run)
        train_records.append(
            {
                "label": label,
                "train_run": run.as_posix(),
                "trainability_required": bool(one_shot.get("decoder_trainability_audit", False)),
                "decoder_only_training": bool(one_shot.get("decoder_only_training", False)),
                "aux_topology_target_transform": one_shot.get("aux_topology_target_transform", "none"),
                "decoder_diagnostic_modulation_override": one_shot.get(
                    "decoder_diagnostic_modulation_override",
                    "none",
                ),
                "decoder_layers": (one_shot.get("d2nn") or {}).get("decoder_layers"),
                "trainability_audit": _load_trainability_audit(run),
                "target_transform_previews": _target_preview_paths(run),
            }
        )
    same_records = [analyze_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in resolved_same]
    test_records = [analyze_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in resolved_test]
    sample_records = [analyze_run(run, kind="sample", oracle_baseline=oracle_baseline) for run in resolved_sample]
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v29",
        "objective": "decoder_trainability_target_encoding_hard_isolation",
        "source_summary": summary.as_posix() if summary is not None else None,
        "oracle_report": oracle_report.as_posix() if oracle_report is not None else None,
        "oracle_baseline": oracle_baseline,
        "train_runs": [run.as_posix() for run in resolved_train],
        "same_batch_eval_runs": [run.as_posix() for run in resolved_same],
        "test_eval_runs": [run.as_posix() for run in resolved_test],
        "sample_runs": [run.as_posix() for run in resolved_sample],
        "train_records": train_records,
        "same_batch_records": same_records,
        "test_records": test_records,
        "sample_records": sample_records,
        "decision": _decision(train_records, same_records, sample_records),
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--train-run", action="append", type=Path, default=None)
    parser.add_argument("--same-batch-eval-run", action="append", type=Path, default=None)
    parser.add_argument("--test-eval-run", action="append", type=Path, default=None)
    parser.add_argument("--sample-run", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v29_decoder_trainability_target_encoding(
        oracle_report=args.oracle_report,
        summary=args.summary,
        train_runs=args.train_run,
        same_batch_eval_runs=args.same_batch_eval_run,
        test_eval_runs=args.test_eval_run,
        sample_runs=args.sample_run,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
