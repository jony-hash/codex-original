#!/usr/bin/env python3
"""Offline v28 decoder controllability / human-topology reachability diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import (
    AUDIT_PASS_ACC,
    FOCUS_CLASSES,
    FOCUS_PASS_CORR,
    FOCUS_PASS_DICE,
    PASS_CORR,
    PASS_DICE,
    _load_json,
    _metric_record,
    _prototype_label_audit,
    _require_report_output,
    _resize_like,
    _torch_load,
)


PLANE_ORDER = (
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
REFERENCE_KEYS = (
    "reference_images",
    "eval_target",
    "teacher_camera",
    "original_digits",
    "target_images",
    "targets",
)
LABEL_KEYS = ("class_labels", "labels", "sample_labels")
ORACLE_DECODED_THRESHOLD = 0.70
LOW_HUMAN_AUDIT_THRESHOLD = 0.50


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v28_decoder_controllability_{timestamp}.json"


def _tensor_bundle_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_tensors.pt" if kind == "eval" else "one_shot_sample_tensors.pt"
    return run / "artifacts" / filename


def _metrics_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_metrics.json" if kind == "eval" else "one_shot_sample_metrics.json"
    return run / "artifacts" / filename


def _v4_preview_path(run: Path, kind: str) -> Path:
    filename = (
        "one_shot_distill_eval_100x100_cell_chain_preview.png"
        if kind == "eval"
        else "one_shot_sample_100x100_cell_chain_preview.png"
    )
    return run / "images" / filename


def _first_tensor(bundle: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, torch.Tensor | None]:
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, value
    return None, None


def _load_bundle(run: Path, kind: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str, str]:
    path = _tensor_bundle_path(run, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = _torch_load(path)
    label_key, labels = _first_tensor(bundle, LABEL_KEYS)
    reference_key, reference = _first_tensor(bundle, REFERENCE_KEYS)
    if labels is None or reference is None:
        missing = []
        if labels is None:
            missing.append("class labels")
        if reference is None:
            missing.append("reference/target images")
        raise ValueError(f"{kind} tensor bundle is missing required {' and '.join(missing)}: {path}")
    return bundle, labels.detach().cpu().long(), reference.detach().float().cpu(), str(label_key), str(reference_key)


def _focus_failures(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    per_class = metrics.get("per_class", {})
    for label in FOCUS_CLASSES:
        record = per_class.get(str(label), {})
        corr = record.get("corr")
        dice = record.get("dice")
        if corr is None or dice is None or float(corr) < FOCUS_PASS_CORR or float(dice) < FOCUS_PASS_DICE:
            failures.append(
                {
                    "class": label,
                    "corr": corr,
                    "dice": dice,
                    "thresholds": {"corr": FOCUS_PASS_CORR, "dice": FOCUS_PASS_DICE},
                }
            )
    return failures


def _plane_record(
    key: str,
    tensor: torch.Tensor,
    reference: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    plane = _resize_like(tensor, reference)
    metrics = _metric_record(plane, reference, labels)
    audit = _prototype_label_audit(plane, reference, labels)
    overall = metrics["overall"]
    focus_failures = _focus_failures(metrics)
    topology_pass = bool(
        float(overall.get("corr") or 0.0) >= PASS_CORR
        and float(overall.get("dice") or 0.0) >= PASS_DICE
        and not focus_failures
        and float(audit.get("correct_ratio") or 0.0) >= AUDIT_PASS_ACC
    )
    return {
        "plane": key,
        "shape": list(tensor.shape),
        "overall": overall,
        "per_class": metrics["per_class"],
        "focus_classes": metrics["focus_classes"],
        "focus_failures": focus_failures,
        "prototype_label_audit_accuracy": audit.get("correct_ratio"),
        "prototype_label_confusion": audit.get("confusion"),
        "topology_gate_passed": topology_pass,
    }


def _plane(planes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((record for record in planes if record["plane"] == name), None)


def _first_failure(planes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in planes:
        if not record["topology_gate_passed"]:
            return {
                "plane": record["plane"],
                "overall": record["overall"],
                "prototype_label_audit_accuracy": record["prototype_label_audit_accuracy"],
                "focus_failures": record["focus_failures"],
                "thresholds": {
                    "corr": PASS_CORR,
                    "dice": PASS_DICE,
                    "focus_corr": FOCUS_PASS_CORR,
                    "focus_dice": FOCUS_PASS_DICE,
                    "prototype_audit_accuracy": AUDIT_PASS_ACC,
                },
            }
    return None


def _oracle_decodable(oracle_baseline: dict[str, Any]) -> bool:
    decision = oracle_baseline.get("decision", oracle_baseline)
    accuracy = decision.get("best_oracle_accuracy")
    status = decision.get("status")
    return status == "ols-decodable" and accuracy is not None and float(accuracy) >= ORACLE_DECODED_THRESHOLD


def _low_audit(record: dict[str, Any] | None) -> bool:
    if record is None:
        return True
    accuracy = record.get("prototype_label_audit_accuracy")
    return accuracy is None or float(accuracy) < LOW_HUMAN_AUDIT_THRESHOLD


def _classify(planes: list[dict[str, Any]], *, run_type: str, oracle_decodable: bool) -> str:
    decoder = _plane(planes, "decoder_output_amplitude")
    chip = _plane(planes, "chip_intensity_primary")
    cmos = _plane(planes, "physical_camera_plane")
    decoder_pass = bool(decoder and decoder["topology_gate_passed"])
    chip_pass = bool(chip and chip["topology_gate_passed"])
    cmos_pass = bool(cmos and cmos["topology_gate_passed"])
    if cmos_pass:
        return "human-accepted-candidate-needs-v4-review" if run_type == "sample" else "decoder-topology-reachable"
    if decoder_pass and chip_pass and not cmos_pass:
        return "camera-after-decoder-failure"
    if decoder_pass or chip_pass:
        return "decoder-topology-reachable"
    if oracle_decodable and _low_audit(decoder) and _low_audit(chip) and _low_audit(cmos):
        return "metric-proxy-failure"
    return "decoder-topology-unreachable"


def analyze_run(run: Path, *, kind: str, oracle_baseline: dict[str, Any]) -> dict[str, Any]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle, labels, reference, label_key, reference_key = _load_bundle(run, kind)
    planes = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            planes.append(_plane_record(key, value.detach().float().cpu(), reference, labels))
        else:
            missing_planes.append(key)
    oracle_decodable = _oracle_decodable(oracle_baseline)
    classification = _classify(planes, run_type=kind, oracle_decodable=oracle_decodable)
    return {
        "run_type": kind,
        "run": run.as_posix(),
        "tensor_bundle": _tensor_bundle_path(run, kind).as_posix(),
        "metrics_path": _metrics_path(run, kind).as_posix(),
        "stored_metrics": _load_json(_metrics_path(run, kind), required=False),
        "v4_preview": _v4_preview_path(run, kind).as_posix(),
        "label_key": label_key,
        "reference_key": reference_key,
        "oracle_decodable": oracle_decodable,
        "missing_planes": missing_planes,
        "planes": planes,
        "first_human_topology_failure_plane": _first_failure(planes),
        "classification": classification,
        "status": classification,
        "human_readability_required": True,
        "accepted_summary_written": False,
    }


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _extract_run_paths(payload: Any, eval_runs: list[Path], sample_runs: list[Path]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "eval_run" and value:
                eval_runs.append(Path(str(value)))
            elif key == "sample_run" and value:
                sample_runs.append(Path(str(value)))
            else:
                _extract_run_paths(value, eval_runs, sample_runs)
    elif isinstance(payload, list):
        for item in payload:
            _extract_run_paths(item, eval_runs, sample_runs)


def _runs_from_summary(summary: Path) -> tuple[list[Path], list[Path]]:
    payload = _load_json(summary)
    assert payload is not None
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    _extract_run_paths(payload, eval_runs, sample_runs)
    return _dedupe(eval_runs), _dedupe(sample_runs)


def _load_oracle_baseline(path: Path | None, summary: Path | None) -> dict[str, Any]:
    if path is not None:
        payload = _load_json(path)
        assert payload is not None
        return payload
    if summary is not None:
        payload = _load_json(summary)
        assert payload is not None
        for key in ("v26_baseline", "baseline"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    classification_runs: dict[str, list[str]] = {}
    for record in records:
        classification_runs.setdefault(record["classification"], []).append(record["run"])
    if classification_runs.get("human-accepted-candidate-needs-v4-review"):
        status = "human-accepted-candidate-needs-v4-review"
    elif classification_runs.get("camera-after-decoder-failure"):
        status = "camera-after-decoder-failure"
    elif classification_runs.get("decoder-topology-reachable"):
        status = "decoder-topology-reachable"
    elif classification_runs.get("metric-proxy-failure"):
        status = "metric-proxy-failure"
    else:
        status = "decoder-topology-unreachable"
    return {
        "status": status,
        "classification_runs": classification_runs,
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def run_decoder_controllability(
    *,
    oracle_report: Path | None = None,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_eval_runs = list(eval_runs or [])
    resolved_sample_runs = list(sample_runs or [])
    if summary is not None and not (resolved_eval_runs or resolved_sample_runs):
        resolved_eval_runs, resolved_sample_runs = _runs_from_summary(summary)
    resolved_eval_runs = _dedupe(resolved_eval_runs)
    resolved_sample_runs = _dedupe(resolved_sample_runs)
    if not resolved_eval_runs and not resolved_sample_runs:
        raise ValueError("at least one --eval-run, --sample-run, or --summary with runs is required")
    oracle_baseline = _load_oracle_baseline(oracle_report, summary)
    records = [analyze_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in resolved_eval_runs]
    records.extend(analyze_run(run, kind="sample", oracle_baseline=oracle_baseline) for run in resolved_sample_runs)
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v28",
        "objective": "decoder_controllability_human_topology_reachability",
        "source_summary": summary.as_posix() if summary is not None else None,
        "oracle_report": oracle_report.as_posix() if oracle_report is not None else None,
        "oracle_baseline": oracle_baseline,
        "eval_runs": [run.as_posix() for run in resolved_eval_runs],
        "sample_runs": [run.as_posix() for run in resolved_sample_runs],
        "thresholds": {
            "corr": PASS_CORR,
            "dice": PASS_DICE,
            "focus_corr": FOCUS_PASS_CORR,
            "focus_dice": FOCUS_PASS_DICE,
            "prototype_audit_accuracy": AUDIT_PASS_ACC,
            "oracle_decodable_accuracy": ORACLE_DECODED_THRESHOLD,
            "low_human_audit": LOW_HUMAN_AUDIT_THRESHOLD,
        },
        "runs": records,
        "decision": _decision(records),
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, default=None, help="Optional v26/v27 OLS oracle report JSON.")
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Existing one-shot eval run.")
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional launcher summary JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_decoder_controllability(
        oracle_report=args.oracle_report,
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
