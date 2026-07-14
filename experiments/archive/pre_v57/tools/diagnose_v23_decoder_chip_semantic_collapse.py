#!/usr/bin/env python3
"""Offline v23 decoder/chip semantic-collapse diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import (
    _load_json,
    _metric_record,
    _prototype_label_audit,
    _require_report_output,
    _resize_like,
    _torch_load,
)


PLANE_ORDER = (
    "slm_command",
    "encoder_input_amplitude",
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "probe_plane",
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
PRE_DECODER_PLANES = {"slm_command", "encoder_input_amplitude", "encoded_amplitude", "ols_output_amplitude"}
DECODER_CHIP_PLANES = {"decoder_output_amplitude", "chip_intensity_primary"}
PRE_COLLAPSE_AUDIT = 0.60
COLLAPSE_AUDIT = 0.50


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v23_decoder_chip_semantic_collapse_{timestamp}.json"


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


def _plane_record(key: str, tensor: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    plane = _resize_like(tensor, reference)
    metrics = _metric_record(plane, reference, labels)
    audit = _prototype_label_audit(plane, reference, labels)
    return {
        "plane": key,
        "shape": list(tensor.shape),
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "focus_classes": metrics["focus_classes"],
        "prototype_label_audit_accuracy": audit.get("correct_ratio"),
        "prototype_label_confusion": audit.get("confusion"),
        "prototype_label_audit": audit,
    }


def _first_semantic_collapse(planes: list[dict[str, Any]]) -> dict[str, Any] | None:
    previous_good: dict[str, Any] | None = None
    for record in planes:
        accuracy = record.get("prototype_label_audit_accuracy")
        if accuracy is None:
            continue
        accuracy_f = float(accuracy)
        if record["plane"] in PRE_DECODER_PLANES and accuracy_f >= PRE_COLLAPSE_AUDIT:
            previous_good = record
        if previous_good is not None and accuracy_f < COLLAPSE_AUDIT:
            return {
                "plane": record["plane"],
                "prototype_label_audit_accuracy": accuracy_f,
                "previous_retained_plane": previous_good["plane"],
                "previous_retained_accuracy": previous_good.get("prototype_label_audit_accuracy"),
                "thresholds": {
                    "previous_retained_accuracy": PRE_COLLAPSE_AUDIT,
                    "collapse_accuracy": COLLAPSE_AUDIT,
                },
            }
    return None


def analyze_run(run: Path, *, kind: str) -> dict[str, Any]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle, labels, reference, label_key, reference_key = _load_bundle(run, kind)
    plane_records = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            plane_records.append(_plane_record(key, value.detach().float().cpu(), reference, labels))
        else:
            missing_planes.append(key)
    first_collapse = _first_semantic_collapse(plane_records)
    decoder_chip_collapse = bool(
        first_collapse is not None and first_collapse["plane"] in DECODER_CHIP_PLANES
    )
    status = "decoder-chip-collapse" if decoder_chip_collapse else "diagnostic-only-no-decoder-chip-collapse"
    return {
        "run_type": kind,
        "run": run.as_posix(),
        "tensor_bundle": _tensor_bundle_path(run, kind).as_posix(),
        "metrics_path": _metrics_path(run, kind).as_posix(),
        "stored_metrics": _load_json(_metrics_path(run, kind), required=False),
        "v4_preview": _v4_preview_path(run, kind).as_posix(),
        "label_key": label_key,
        "reference_key": reference_key,
        "plane_order": list(PLANE_ORDER),
        "missing_planes": missing_planes,
        "planes": plane_records,
        "first_semantic_collapse_plane": first_collapse,
        "decoder_chip_collapse": decoder_chip_collapse,
        "status": status,
        "accepted_summary_written": False,
    }


def _runs_from_summary(summary_path: Path) -> tuple[list[Path], list[Path]]:
    summary = _load_json(summary_path)
    assert summary is not None
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    for probe in summary.get("probes", []):
        if isinstance(probe, dict) and probe.get("eval_run"):
            eval_runs.append(Path(probe["eval_run"]))
    for item in summary.get("sample_runs", []):
        if isinstance(item, dict) and item.get("sample_run"):
            sample_runs.append(Path(item["sample_run"]))
        elif isinstance(item, str):
            sample_runs.append(Path(item))
    return eval_runs, sample_runs


def run_decoder_chip_semantic_collapse(
    *,
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
    if not resolved_eval_runs and not resolved_sample_runs:
        raise ValueError("at least one --eval-run, --sample-run, or a summary with runs is required")
    run_records = [analyze_run(Path(run), kind="eval") for run in resolved_eval_runs]
    run_records.extend(analyze_run(Path(run), kind="sample") for run in resolved_sample_runs)
    collapse_runs = [record for record in run_records if record["decoder_chip_collapse"]]
    status = "decoder-chip-collapse" if collapse_runs else "diagnostic-only-no-decoder-chip-collapse"
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v23",
        "objective": "decoder_chip_semantic_collapse_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "thresholds": {
            "pre_decoder_retained_audit": PRE_COLLAPSE_AUDIT,
            "collapse_audit": COLLAPSE_AUDIT,
        },
        "runs": run_records,
        "decision": {
            "status": status,
            "decoder_chip_collapse_runs": [record["run"] for record in collapse_runs],
            "accepted_summary_written": False,
        },
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
    summary = run_decoder_chip_semantic_collapse(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
