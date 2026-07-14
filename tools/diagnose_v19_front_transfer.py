#!/usr/bin/env python3
"""Offline v19 SLM-to-encoder front-transfer diagnostic.

This tool reads existing one-shot sample artifacts and writes a diagnostic-only
report under outputs/reports/. It does not train, run CUDA, create runs, or
write accepted summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch
import torch.nn.functional as F

from ldm_project.sampling_reports import compute_sample_quality_metrics


FOCUS_CLASSES = (3, 4, 7, 9)
FRONT_PLANE_KEYS = (
    "slm_command",
    "slm_amplitude",
    "encoder_input_amplitude",
    "encoder_input_grid_amplitude",
    "generated_camera",
    "generated_camera_display",
)
FRONT_RETAIN_CORR = 0.50
FRONT_RETAIN_DICE = 0.50


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v19_front_transfer_{timestamp}.json"


def _require_report_output(path: Path) -> None:
    normalized = path.as_posix()
    if path.is_absolute() or not normalized.startswith("outputs/reports/"):
        raise ValueError("v19 front-transfer output must be a relative path under outputs/reports/")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"tensor bundle must contain a dict payload: {path}")
    return payload


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _load_sample_bundle(sample_run: Path) -> dict[str, Any]:
    bundle_path = sample_run / "artifacts" / "one_shot_sample_tensors.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    bundle = _torch_load(bundle_path)
    required = ("class_labels", "slm_command", "encoder_input_amplitude")
    missing = [key for key in required if key not in bundle or not torch.is_tensor(bundle[key])]
    if missing:
        raise ValueError(f"sample tensor bundle is missing required tensors {missing}: {bundle_path}")
    return bundle


def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    resized = tensor.detach().float().cpu()
    if resized.ndim != reference.ndim:
        raise ValueError(f"plane tensor rank must match reference rank: {resized.shape} != {reference.shape}")
    if resized.shape[0] != reference.shape[0]:
        raise ValueError(f"plane tensor batch must match reference batch: {resized.shape} != {reference.shape}")
    if resized.shape[-2:] != reference.shape[-2:]:
        resized = F.interpolate(resized, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return resized.clamp(0.0, 1.0)


def _metric_record(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics.get("overall", {})
    per_class = metrics.get("per_class", {})
    return {
        "overall": {
            "corr": overall.get("corr"),
            "dice": overall.get("dice"),
            "hole_proxy": overall.get("hole_proxy"),
        },
        "per_class": per_class,
        "focus_classes": {
            str(label): per_class.get(str(label), {})
            for label in FOCUS_CLASSES
            if str(label) in per_class
        },
    }


def _plane_audit(bundle: dict[str, Any], labels: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    slm_reference = bundle["slm_command"].detach().float().cpu().clamp(0.0, 1.0)
    records: list[dict[str, Any]] = []
    first_collapse: dict[str, Any] | None = None
    for key in FRONT_PLANE_KEYS:
        value = bundle.get(key)
        if not torch.is_tensor(value):
            records.append({"plane": key, "available": False, "reason": "missing"})
            continue
        plane = _resize_like(value, slm_reference)
        metrics = compute_sample_quality_metrics(plane, slm_reference, labels)
        record = {"plane": key, "available": True, **_metric_record(metrics)}
        overall = record["overall"]
        collapsed = bool(
            float(overall.get("corr") or 0.0) < FRONT_RETAIN_CORR
            or float(overall.get("dice") or 0.0) < FRONT_RETAIN_DICE
        )
        record["gross_collapse"] = collapsed
        if collapsed and first_collapse is None:
            first_collapse = {
                "plane": key,
                "corr": overall.get("corr"),
                "dice": overall.get("dice"),
                "thresholds": {"corr": FRONT_RETAIN_CORR, "dice": FRONT_RETAIN_DICE},
            }
        records.append(record)
    return records, first_collapse


def analyze_sample_run(sample_run: Path) -> dict[str, Any]:
    sample_metrics_path = sample_run / "artifacts" / "one_shot_sample_metrics.json"
    bundle = _load_sample_bundle(sample_run)
    labels = bundle["class_labels"].detach().cpu().long()
    plane_records, first_collapse = _plane_audit(bundle, labels)
    encoder_record = next(
        item for item in plane_records
        if item["plane"] == "encoder_input_amplitude" and item.get("available")
    )
    encoder_overall = encoder_record["overall"]
    front_transfer_retained = bool(
        float(encoder_overall.get("corr") or 0.0) >= FRONT_RETAIN_CORR
        and float(encoder_overall.get("dice") or 0.0) >= FRONT_RETAIN_DICE
    )
    return {
        "sample_run": sample_run.as_posix(),
        "sample_tensor_bundle": (sample_run / "artifacts" / "one_shot_sample_tensors.pt").as_posix(),
        "sample_metrics_path": sample_metrics_path.as_posix(),
        "stored_sample_metrics": _load_json(sample_metrics_path, required=False),
        "v4_preview": (sample_run / "images" / "one_shot_sample_100x100_cell_chain_preview.png").as_posix(),
        "count": int(labels.shape[0]),
        "class_labels": labels.tolist(),
        "front_reference_plane": "slm_command",
        "plane_audit": plane_records,
        "encoder_input_transfer": encoder_record,
        "first_gross_collapse_plane": first_collapse,
        "front_transfer_retained": front_transfer_retained,
        "status": (
            "diagnostic-only-front-transfer-retained"
            if front_transfer_retained
            else "diagnostic-only-front-transfer-collapsed"
        ),
        "accepted_summary_written": False,
    }


def _runs_from_summary(summary_path: Path) -> list[Path]:
    summary = _load_json(summary_path)
    assert summary is not None
    sample_runs: list[Path] = []
    for item in summary.get("sample_runs", []):
        if isinstance(item, dict) and item.get("sample_run"):
            sample_runs.append(Path(item["sample_run"]))
        elif isinstance(item, str):
            sample_runs.append(Path(item))
    return sample_runs


def run_front_transfer(
    *,
    sample_runs: list[Path] | None = None,
    summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_sample_runs = list(sample_runs or [])
    if summary is not None and not resolved_sample_runs:
        resolved_sample_runs = _runs_from_summary(summary)
    if not resolved_sample_runs:
        raise ValueError("at least one --sample-run or a summary with sample_runs is required")

    run_records = [analyze_sample_run(Path(sample_run)) for sample_run in resolved_sample_runs]
    retained_runs = [run["sample_run"] for run in run_records if run["front_transfer_retained"]]
    collapsed_runs = [run["sample_run"] for run in run_records if not run["front_transfer_retained"]]
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v19",
        "objective": "slm_to_encoder_front_transfer_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "front_reference_plane": "slm_command",
        "thresholds": {
            "front_retain_corr": FRONT_RETAIN_CORR,
            "front_retain_dice": FRONT_RETAIN_DICE,
        },
        "plane_keys": list(FRONT_PLANE_KEYS),
        "runs": run_records,
        "decision": {
            "retained_runs": retained_runs,
            "collapsed_runs": collapsed_runs,
            "front_transfer_retained": bool(retained_runs) and not collapsed_runs,
            "status": (
                "diagnostic-only-front-transfer-retained"
                if retained_runs and not collapsed_runs
                else "diagnostic-only-front-transfer-collapsed"
            ),
            "accepted_summary_written": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional launcher summary JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_front_transfer(
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
