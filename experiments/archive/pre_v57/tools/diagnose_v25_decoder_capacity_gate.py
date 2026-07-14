#!/usr/bin/env python3
"""Offline v25 decoder-capacity hard gate diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
import tools.diagnose_v21_physical_cmos_readout as physical_cmos_tool
import tools.diagnose_v24_ols_decoder_expressivity as expressivity_tool


BASELINE_V24_PROBE_A_SAMPLE_CORR = 0.5968586027622222
WORTH_AUDIT = 0.50
WORTH_CORR_GAIN = 0.08


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v25_decoder_capacity_gate_{timestamp}.json"


def _runs_from_summary(summary_path: Path) -> tuple[list[Path], list[Path], dict[str, str], dict[str, Any]]:
    summary = _load_json(summary_path)
    assert summary is not None
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    labels_by_run: dict[str, str] = {}
    for probe in summary.get("probes", []):
        if not isinstance(probe, dict):
            continue
        label = str(probe.get("label") or probe.get("probe") or "")
        eval_run = probe.get("eval_run")
        if eval_run:
            path = Path(eval_run)
            eval_runs.append(path)
            labels_by_run[path.as_posix()] = label
    for item in summary.get("sample_runs", []):
        if isinstance(item, dict) and item.get("sample_run"):
            path = Path(item["sample_run"])
            sample_runs.append(path)
            labels_by_run[path.as_posix()] = str(item.get("label") or labels_by_run.get(path.as_posix()) or "")
        elif isinstance(item, str):
            path = Path(item)
            sample_runs.append(path)
    return eval_runs, sample_runs, labels_by_run, summary


def _fallback_label(run: str) -> str:
    name = Path(run).name
    for marker in ("train_v25_", "v25_"):
        if marker in name:
            return name.split(marker, 1)[1].replace("_sample", "").replace("_eval", "")
    return name


def _label_for(run: str, labels_by_run: dict[str, str]) -> str:
    return labels_by_run.get(run) or _fallback_label(run)


def _overall_corr(record: dict[str, Any]) -> float | None:
    overall = record.get("physical_cmos", {}).get("overall", {})
    corr = overall.get("corr")
    return None if corr is None else float(corr)


def _audit(record: dict[str, Any]) -> float | None:
    value = record.get("prototype_label_audit_accuracy")
    return None if value is None else float(value)


def _run_kind_map(records: list[dict[str, Any]], labels_by_run: dict[str, str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        label = _label_for(record["run"], labels_by_run)
        grouped.setdefault(label, {})[record["run_type"]] = record
    return grouped


def _metadata(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    return dict(record.get("artifact_metadata") or {})


def _decoder_layers(metadata: dict[str, Any]) -> int | None:
    d2nn = metadata.get("d2nn")
    if isinstance(d2nn, dict) and d2nn.get("decoder_layers") is not None:
        return int(d2nn["decoder_layers"])
    return None


def _decoder_override(metadata: dict[str, Any]) -> str:
    value = metadata.get("decoder_diagnostic_modulation_override")
    if isinstance(value, str):
        return value
    value = metadata.get("output_decoder_diagnostic_modulation_override")
    return value if isinstance(value, str) else "none"


def _physical_positive(record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("physical_cmos_positive"))


def _worth_following(sample: dict[str, Any] | None, *, baseline_corr: float) -> bool:
    if not sample:
        return False
    audit = _audit(sample)
    corr = _overall_corr(sample)
    audit_worth = audit is not None and audit >= WORTH_AUDIT
    corr_worth = corr is not None and (corr - baseline_corr) >= WORTH_CORR_GAIN
    return bool(audit_worth or corr_worth)


def _probe_record(
    label: str,
    grouped_physical: dict[str, Any],
    grouped_expressivity: dict[str, Any],
    *,
    baseline_corr: float,
) -> dict[str, Any]:
    eval_physical = grouped_physical.get("eval")
    sample_physical = grouped_physical.get("sample")
    eval_expr = grouped_expressivity.get("eval")
    sample_expr = grouped_expressivity.get("sample")
    metadata = _metadata(sample_expr) or _metadata(eval_expr)
    sample_positive = _physical_positive(sample_physical)
    eval_positive = _physical_positive(eval_physical)
    worth = _worth_following(sample_physical, baseline_corr=baseline_corr)
    if eval_positive and not sample_positive:
        status = "eval-positive-sample-failed"
    elif worth:
        status = "capacity-rescue-but-human-failed"
    else:
        status = "no-capacity-rescue"
    return {
        "label": label,
        "metadata": metadata,
        "metadata_source": (sample_expr or eval_expr or {}).get("artifact_metadata_source"),
        "decoder_layers": _decoder_layers(metadata),
        "decoder_diagnostic_modulation_override": _decoder_override(metadata),
        "eval_run": eval_physical.get("run") if eval_physical else None,
        "sample_run": sample_physical.get("run") if sample_physical else None,
        "eval_physical_cmos": eval_physical,
        "sample_physical_cmos": sample_physical,
        "eval_expressivity": eval_expr,
        "sample_expressivity": sample_expr,
        "eval_physical_cmos_positive": eval_positive,
        "sample_physical_cmos_positive": sample_positive,
        "sample_corr_gain_over_v24_probe_a_sample": (
            None
            if _overall_corr(sample_physical or {}) is None
            else _overall_corr(sample_physical or {}) - baseline_corr
        ),
        "sample_prototype_label_audit_accuracy": _audit(sample_physical or {}),
        "sample_v4_preview": sample_physical.get("v4_preview") if sample_physical else None,
        "eval_v4_preview": eval_physical.get("v4_preview") if eval_physical else None,
        "first_semantic_collapse_plane": (
            (sample_expr or eval_expr or {}).get("first_semantic_collapse_plane")
        ),
        "worth_following": worth,
        "human_readability_required": True,
        "status": status,
    }


def _decision(probes: list[dict[str, Any]]) -> dict[str, Any]:
    eval_sample_failed = [probe["label"] for probe in probes if probe["status"] == "eval-positive-sample-failed"]
    worth = [probe for probe in probes if probe["worth_following"]]
    depth = [
        probe["label"]
        for probe in worth
        if probe.get("decoder_layers") and int(probe["decoder_layers"]) > 2
        and probe.get("decoder_diagnostic_modulation_override") == "none"
    ]
    phase_amp = [
        probe["label"]
        for probe in worth
        if probe.get("decoder_diagnostic_modulation_override") == "phase_amplitude"
    ]
    phase_only_worth = [
        probe
        for probe in worth
        if probe.get("decoder_diagnostic_modulation_override") == "none"
    ]
    if eval_sample_failed:
        status = "eval-positive-sample-failed"
    elif depth:
        status = "depth-limited"
    elif phase_amp and not phase_only_worth:
        status = "phase-only-limited"
    elif worth:
        status = "capacity-rescue-but-human-failed"
    else:
        status = "no-capacity-rescue"
    return {
        "status": status,
        "eval_positive_sample_failed_runs": eval_sample_failed,
        "depth_limited_runs": depth,
        "phase_only_limited_runs": phase_amp if status == "phase-only-limited" else [],
        "capacity_rescue_but_human_failed_runs": [probe["label"] for probe in worth],
        "no_capacity_rescue_runs": [probe["label"] for probe in probes if not probe["worth_following"]],
        "human_readability_required": True,
        "accepted_summary_written": False,
    }


def run_decoder_capacity_gate(
    *,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    labels_by_run: dict[str, str] = {}
    source_summary: dict[str, Any] = {}
    resolved_eval_runs = list(eval_runs or [])
    resolved_sample_runs = list(sample_runs or [])
    if summary is not None and not (resolved_eval_runs or resolved_sample_runs):
        resolved_eval_runs, resolved_sample_runs, labels_by_run, source_summary = _runs_from_summary(summary)
    if not resolved_eval_runs and not resolved_sample_runs:
        raise ValueError("at least one --eval-run, --sample-run, or a summary with runs is required")
    baseline_corr = float(
        source_summary.get("baseline", {}).get(
            "v24_probe_a_sample_corr",
            BASELINE_V24_PROBE_A_SAMPLE_CORR,
        )
    )
    physical_records = [
        physical_cmos_tool.analyze_run(Path(run), kind="eval") for run in resolved_eval_runs
    ]
    physical_records.extend(
        physical_cmos_tool.analyze_run(Path(run), kind="sample") for run in resolved_sample_runs
    )
    expressivity_records = [
        expressivity_tool.analyze_run(Path(run), kind="eval") for run in resolved_eval_runs
    ]
    expressivity_records.extend(
        expressivity_tool.analyze_run(Path(run), kind="sample") for run in resolved_sample_runs
    )
    physical_by_label = _run_kind_map(physical_records, labels_by_run)
    expressivity_by_label = _run_kind_map(expressivity_records, labels_by_run)
    labels = sorted(set(physical_by_label) | set(expressivity_by_label))
    probes = [
        _probe_record(
            label,
            physical_by_label.get(label, {}),
            expressivity_by_label.get(label, {}),
            baseline_corr=baseline_corr,
        )
        for label in labels
    ]
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v25",
        "objective": "decoder_capacity_hard_human_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "baseline": {
            "v24_probe_a_sample_corr": baseline_corr,
        },
        "thresholds": {
            "worth_audit": WORTH_AUDIT,
            "worth_corr_gain": WORTH_CORR_GAIN,
            "human_readability_required": True,
        },
        "probes": probes,
        "decision": _decision(probes),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Existing one-shot eval run directory.")
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional launcher summary JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_decoder_capacity_gate(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
