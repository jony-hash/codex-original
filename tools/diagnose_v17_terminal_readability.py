#!/usr/bin/env python3
"""Offline v17 terminal CMOS readability diagnostic.

This tool reads existing one-shot sample/eval artifacts and writes a
diagnostic-only report under outputs/reports/. It does not train, run CUDA, or
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
PLANE_KEYS = (
    "generator_desired_intensity",
    "slm_command",
    "encoder_input_amplitude",
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_amplitude",
    "diffractive_camera_plane_raw",
    "generated_camera",
    "generated_camera_display",
)
CORR_GATE = 0.78
DICE_GATE = 0.72
FOCUS_CORR_GATE = 0.60
FOCUS_DICE_GATE = 0.60
PROTOTYPE_AUDIT_GATE = 0.70
GROSS_COLLAPSE_CORR = 0.50
GROSS_COLLAPSE_DICE = 0.50


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v17_terminal_readability_{timestamp}.json"


def _require_report_output(path: Path) -> None:
    normalized = path.as_posix()
    if path.is_absolute() or not normalized.startswith("outputs/reports/"):
        raise ValueError("v17 terminal readability output must be a relative path under outputs/reports/")


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
    required = ("class_labels", "reference_images")
    missing = [key for key in required if key not in bundle or not torch.is_tensor(bundle[key])]
    if missing:
        raise ValueError(f"sample tensor bundle is missing required tensors {missing}: {bundle_path}")
    camera = bundle.get("generated_camera_raw", bundle.get("generated_camera", bundle.get("camera_plane")))
    if not torch.is_tensor(camera):
        raise ValueError(f"sample tensor bundle is missing generated camera tensor: {bundle_path}")
    bundle["_camera_for_audit"] = camera
    return bundle


def _reference_bank(reference_images: torch.Tensor, reference_labels: torch.Tensor | None, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    label_source = reference_labels if reference_labels is not None else labels
    bank: dict[int, torch.Tensor] = {}
    for label in sorted({int(item) for item in label_source.detach().cpu().tolist()}):
        mask = label_source.detach().cpu().long() == label
        if bool(mask.any()):
            bank[label] = reference_images.detach().cpu().index_select(0, torch.nonzero(mask, as_tuple=False).flatten())
    return bank


def _normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scaled = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return F.normalize(scaled, dim=1, eps=1.0e-6)


def _prototype_label_audit(camera: torch.Tensor, labels: torch.Tensor, reference_bank: dict[int, torch.Tensor]) -> dict[str, Any]:
    if not reference_bank:
        return {"available": False, "reason": "empty reference bank", "correct_ratio": None, "per_class": {}}
    class_labels = sorted(reference_bank)
    prototypes = torch.stack([reference_bank[label].detach().float().mean(dim=0) for label in class_labels])
    if prototypes.shape[-2:] != camera.shape[-2:]:
        prototypes = F.interpolate(prototypes, size=camera.shape[-2:], mode="bilinear", align_corners=False)
    correlations = _normalize_features(camera) @ _normalize_features(prototypes).t()
    nearest_indices = correlations.argmax(dim=1)
    requested = labels.detach().cpu().long()
    nearest_labels = torch.tensor([class_labels[int(index)] for index in nearest_indices.tolist()], dtype=torch.long)
    correct = nearest_labels == requested
    per_class: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for label in sorted({int(item) for item in requested.tolist()}):
        mask = requested == label
        nearest_for_class = nearest_labels[mask]
        counts: dict[str, int] = {}
        for item in nearest_for_class.tolist():
            counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
        confusion[str(label)] = counts
        per_class[str(label)] = {
            "count": int(mask.sum().item()),
            "correct_ratio": float(correct[mask].float().mean().item()) if bool(mask.any()) else 0.0,
            "nearest_label_counts": counts,
        }
    return {
        "available": True,
        "class_labels": class_labels,
        "correct_ratio": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "confusion": confusion,
        "per_class": per_class,
    }


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


def _plane_audit(bundle: dict[str, Any], reference: torch.Tensor, labels: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    first_collapse: dict[str, Any] | None = None
    for key in PLANE_KEYS:
        value = bundle.get(key)
        if not torch.is_tensor(value):
            records.append({"plane": key, "available": False, "reason": "missing"})
            continue
        plane = _resize_like(value, reference)
        metrics = compute_sample_quality_metrics(plane, reference, labels)
        record = {"plane": key, "available": True, **_metric_record(metrics)}
        overall = record["overall"]
        collapsed = bool(
            float(overall.get("corr") or 0.0) < GROSS_COLLAPSE_CORR
            or float(overall.get("dice") or 0.0) < GROSS_COLLAPSE_DICE
        )
        record["gross_collapse"] = collapsed
        if collapsed and first_collapse is None:
            first_collapse = {
                "plane": key,
                "corr": overall.get("corr"),
                "dice": overall.get("dice"),
                "thresholds": {"corr": GROSS_COLLAPSE_CORR, "dice": GROSS_COLLAPSE_DICE},
            }
        records.append(record)
    return records, first_collapse


def _focus_failures(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    per_class = metrics.get("per_class", {})
    failures: list[dict[str, Any]] = []
    for label in FOCUS_CLASSES:
        record = per_class.get(str(label))
        if not record:
            failures.append({"label": label, "reason": "missing"})
            continue
        corr = float(record.get("corr") or 0.0)
        dice = float(record.get("dice") or 0.0)
        if corr < FOCUS_CORR_GATE or dice < FOCUS_DICE_GATE:
            failures.append({"label": label, "corr": corr, "dice": dice})
    return failures


def _gate(metrics: dict[str, Any], prototype_audit: dict[str, Any]) -> dict[str, Any]:
    overall = metrics.get("overall", {})
    corr = float(overall.get("corr") or 0.0)
    dice = float(overall.get("dice") or 0.0)
    focus_failures = _focus_failures(metrics)
    audit_accuracy = prototype_audit.get("correct_ratio")
    prototype_passed = bool(audit_accuracy is not None and float(audit_accuracy) >= PROTOTYPE_AUDIT_GATE)
    checks = {
        "overall_corr": corr >= CORR_GATE,
        "overall_dice": dice >= DICE_GATE,
        "focus_classes": not focus_failures,
        "prototype_label_audit": prototype_passed,
    }
    candidate = all(checks.values())
    return {
        "candidate": candidate,
        "status": (
            "diagnostic-only-candidate-needs-human-v4-review"
            if candidate
            else "diagnostic-only-failed-terminal-readability"
        ),
        "checks": checks,
        "thresholds": {
            "overall_corr": CORR_GATE,
            "overall_dice": DICE_GATE,
            "focus_corr": FOCUS_CORR_GATE,
            "focus_dice": FOCUS_DICE_GATE,
            "prototype_label_audit_accuracy": PROTOTYPE_AUDIT_GATE,
        },
        "values": {
            "overall_corr": overall.get("corr"),
            "overall_dice": overall.get("dice"),
            "prototype_label_audit_accuracy": audit_accuracy,
        },
        "focus_failures": focus_failures,
        "human_readability_required": True,
        "accepted_summary_written": False,
    }


def _eval_artifacts(eval_run: Path) -> dict[str, Any]:
    metrics_path = eval_run / "artifacts" / "one_shot_distill_eval_metrics.json"
    audit_path = eval_run / "artifacts" / "one_shot_distill_eval_student_label_audit.json"
    return {
        "eval_run": eval_run.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "metrics": _load_json(metrics_path, required=False),
        "student_label_audit_path": audit_path.as_posix(),
        "student_label_audit": _load_json(audit_path, required=False),
        "v4_preview": (eval_run / "images" / "one_shot_distill_eval_100x100_cell_chain_preview.png").as_posix(),
    }


def analyze_sample_run(sample_run: Path, *, eval_run: Path | None = None) -> dict[str, Any]:
    sample_metrics_path = sample_run / "artifacts" / "one_shot_sample_metrics.json"
    stored_metrics = _load_json(sample_metrics_path)
    bundle = _load_sample_bundle(sample_run)
    labels = bundle["class_labels"].detach().cpu().long()
    reference = bundle["reference_images"].detach().float().cpu()
    reference_labels = bundle.get("reference_labels")
    if torch.is_tensor(reference_labels):
        reference_labels = reference_labels.detach().cpu().long()
    else:
        reference_labels = labels
    camera = _resize_like(bundle["_camera_for_audit"], reference)
    bank = _reference_bank(reference, reference_labels, labels)
    terminal_metrics = compute_sample_quality_metrics(camera, reference, labels)
    prototype_audit = _prototype_label_audit(camera, labels, bank)
    plane_records, first_collapse = _plane_audit(bundle, reference, labels)
    gate = _gate(terminal_metrics, prototype_audit)
    gate["first_gross_collapse_plane"] = first_collapse
    return {
        "sample_run": sample_run.as_posix(),
        "sample_tensor_bundle": (sample_run / "artifacts" / "one_shot_sample_tensors.pt").as_posix(),
        "sample_metrics_path": sample_metrics_path.as_posix(),
        "stored_sample_metrics": stored_metrics,
        "v4_preview": (sample_run / "images" / "one_shot_sample_100x100_cell_chain_preview.png").as_posix(),
        "count": int(labels.shape[0]),
        "class_labels": labels.tolist(),
        "reference_counts": {str(label): int(images.shape[0]) for label, images in sorted(bank.items())},
        "terminal_exact": _metric_record(terminal_metrics),
        "focus_classes": _metric_record(terminal_metrics)["focus_classes"],
        "prototype_label_audit": prototype_audit,
        "plane_audit": plane_records,
        "first_gross_collapse_plane": first_collapse,
        "eval": _eval_artifacts(eval_run) if eval_run is not None else None,
        "gate": gate,
    }


def _runs_from_summary(summary_path: Path) -> tuple[list[Path], list[Path]]:
    summary = _load_json(summary_path)
    assert summary is not None
    sample_runs: list[Path] = []
    sample_labels: list[str | None] = []
    for item in summary.get("sample_runs", []):
        if isinstance(item, dict) and item.get("sample_run"):
            sample_runs.append(Path(item["sample_run"]))
            sample_labels.append(str(item.get("label")) if item.get("label") is not None else None)
        elif isinstance(item, str):
            sample_runs.append(Path(item))
            sample_labels.append(None)
    eval_by_label = {
        str(probe.get("label")): Path(probe["eval_run"])
        for probe in summary.get("probes", [])
        if isinstance(probe, dict) and probe.get("label") is not None and probe.get("eval_run")
    }
    eval_runs: list[Path] = []
    if sample_runs and eval_by_label:
        for label in sample_labels:
            if label is not None and label in eval_by_label:
                eval_runs.append(eval_by_label[label])
    else:
        for probe in summary.get("probes", []):
            if isinstance(probe, dict) and probe.get("eval_run"):
                eval_runs.append(Path(probe["eval_run"]))
    return sample_runs, eval_runs


def run_terminal_readability(
    *,
    sample_runs: list[Path] | None = None,
    eval_runs: list[Path] | None = None,
    summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_sample_runs = list(sample_runs or [])
    resolved_eval_runs = list(eval_runs or [])
    if summary is not None:
        summary_samples, summary_evals = _runs_from_summary(summary)
        if not resolved_sample_runs:
            resolved_sample_runs = summary_samples
        if not resolved_eval_runs:
            resolved_eval_runs = summary_evals
    if not resolved_sample_runs:
        raise ValueError("at least one --sample-run or a summary with sample_runs is required")
    run_records = []
    for index, sample_run in enumerate(resolved_sample_runs):
        eval_run = resolved_eval_runs[index] if index < len(resolved_eval_runs) else None
        run_records.append(analyze_sample_run(Path(sample_run), eval_run=eval_run))
    candidate_runs = [run["sample_run"] for run in run_records if run["gate"]["candidate"]]
    failed_runs = [run["sample_run"] for run in run_records if not run["gate"]["candidate"]]
    summary_payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v17.1",
        "objective": "terminal_cmos_readability_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "thresholds": {
            "overall_corr": CORR_GATE,
            "overall_dice": DICE_GATE,
            "focus_corr": FOCUS_CORR_GATE,
            "focus_dice": FOCUS_DICE_GATE,
            "prototype_label_audit_accuracy": PROTOTYPE_AUDIT_GATE,
            "gross_collapse_corr": GROSS_COLLAPSE_CORR,
            "gross_collapse_dice": GROSS_COLLAPSE_DICE,
        },
        "plane_keys": list(PLANE_KEYS),
        "runs": run_records,
        "decision": {
            "candidate_runs": candidate_runs,
            "failed_runs": failed_runs,
            "candidate": bool(candidate_runs) and not failed_runs,
            "status": (
                "diagnostic-only-candidate-needs-human-v4-review"
                if candidate_runs and not failed_runs
                else "diagnostic-only-failed-terminal-readability"
            ),
            "human_readability_required": True,
            "accepted_summary_written": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Optional eval run directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional v17 launcher summary JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_terminal_readability(
        sample_runs=args.sample_run,
        eval_runs=args.eval_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
