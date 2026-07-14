#!/usr/bin/env python3
"""Offline v21 physical-CMOS readout diagnostic.

This tool reads existing one-shot eval/sample tensor bundles and writes a
diagnostic-only report under outputs/reports/. It does not train, run CUDA,
create runs, or write accepted summaries.
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
REFERENCE_KEYS = (
    "reference_images",
    "eval_target",
    "teacher_camera",
    "original_digits",
    "target_images",
    "targets",
)
LABEL_KEYS = ("class_labels", "labels", "sample_labels")
PHYSICAL_CMOS_KEYS = (
    "physical_camera_plane",
    "diffractive_camera_plane_raw",
    "generated_camera_raw",
    "generated_camera",
    "student_camera_raw",
)
SELECTED_READOUT_KEYS = (
    "selected_readout",
    "selected_readout_raw",
    "camera_plane",
)
PASS_CORR = 0.78
PASS_DICE = 0.72
FOCUS_PASS_CORR = 0.60
FOCUS_PASS_DICE = 0.60
AUDIT_PASS_ACC = 0.70
READABLE_SELECTED_CORR = 0.60
READABLE_SELECTED_DICE = 0.55


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v21_physical_cmos_readout_{timestamp}.json"


def _require_report_output(path: Path) -> None:
    normalized = path.as_posix()
    if path.is_absolute() or not normalized.startswith("outputs/reports/"):
        raise ValueError("v21 physical CMOS readout output must be a relative path under outputs/reports/")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
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


def _first_tensor(bundle: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, torch.Tensor] | tuple[None, None]:
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, value
    return None, None


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


def _load_bundle(run: Path, kind: str) -> dict[str, Any]:
    bundle_path = _tensor_bundle_path(run, kind)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    bundle = _torch_load(bundle_path)
    label_key, labels = _first_tensor(bundle, LABEL_KEYS)
    reference_key, reference = _first_tensor(bundle, REFERENCE_KEYS)
    if labels is None or reference is None:
        missing = []
        if labels is None:
            missing.append("class labels")
        if reference is None:
            missing.append("reference/target images")
        raise ValueError(f"{kind} tensor bundle is missing required {' and '.join(missing)}: {bundle_path}")
    bundle["_v21_label_key"] = label_key
    bundle["_v21_reference_key"] = reference_key
    return bundle


def _primary_readout_name(bundle: dict[str, Any]) -> str:
    value = bundle.get("primary_readout_plane", "camera")
    return value if isinstance(value, str) else "camera"


def _physical_cmos(bundle: dict[str, Any]) -> tuple[str | None, torch.Tensor | None]:
    source, value = _first_tensor(bundle, PHYSICAL_CMOS_KEYS)
    if value is not None:
        return source, value
    if _primary_readout_name(bundle) == "camera":
        return _first_tensor(bundle, ("camera_plane",))
    return None, None


def _selected_readout(bundle: dict[str, Any]) -> tuple[str | None, torch.Tensor | None]:
    source, value = _first_tensor(bundle, SELECTED_READOUT_KEYS)
    if value is not None:
        return source, value
    return _physical_cmos(bundle)


def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    resized = tensor.detach().float().cpu()
    if resized.ndim != reference.ndim:
        raise ValueError(f"plane tensor rank must match reference rank: {resized.shape} != {reference.shape}")
    if resized.shape[0] != reference.shape[0]:
        raise ValueError(f"plane tensor batch must match reference batch: {resized.shape} != {reference.shape}")
    if resized.shape[-2:] != reference.shape[-2:]:
        resized = F.interpolate(resized, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return resized.clamp(0.0, 1.0)


def _metric_record(plane: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    metrics = compute_sample_quality_metrics(plane, reference, labels)
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


def _normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scaled = centered / flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return F.normalize(scaled, dim=1, eps=1.0e-6)


def _prototype_label_audit(plane: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    requested = labels.detach().cpu().long()
    class_labels = sorted({int(item) for item in requested.tolist()})
    prototypes = []
    for label in class_labels:
        mask = requested == label
        prototypes.append(reference.index_select(0, torch.nonzero(mask, as_tuple=False).flatten()).mean(dim=0))
    if not prototypes:
        return {"available": False, "reason": "empty reference bank", "correct_ratio": None, "confusion": {}}
    prototype_tensor = torch.stack(prototypes)
    if prototype_tensor.shape[-2:] != plane.shape[-2:]:
        prototype_tensor = F.interpolate(prototype_tensor, size=plane.shape[-2:], mode="bilinear", align_corners=False)
    correlations = _normalize_features(plane) @ _normalize_features(prototype_tensor).t()
    nearest_indices = correlations.argmax(dim=1)
    nearest_labels = torch.tensor([class_labels[int(index)] for index in nearest_indices.tolist()], dtype=torch.long)
    correct = nearest_labels == requested
    confusion: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, Any]] = {}
    for label in class_labels:
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


def _readable(record: dict[str, Any] | None, *, corr: float, dice: float) -> bool:
    if not record:
        return False
    overall = record.get("overall", {})
    return bool(
        float(overall.get("corr") or 0.0) >= corr
        and float(overall.get("dice") or 0.0) >= dice
    )


def _focus_failures(record: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    per_class = record.get("per_class", {})
    for label in FOCUS_CLASSES:
        metrics = per_class.get(str(label), {})
        corr = metrics.get("corr")
        dice = metrics.get("dice")
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


def analyze_run(run: Path, *, kind: str) -> dict[str, Any]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle = _load_bundle(run, kind)
    label_key = bundle["_v21_label_key"]
    reference_key = bundle["_v21_reference_key"]
    labels = bundle[label_key].detach().cpu().long()
    reference = bundle[reference_key].detach().float().cpu()
    physical_source, physical = _physical_cmos(bundle)
    selected_source, selected = _selected_readout(bundle)
    if not torch.is_tensor(physical):
        raise ValueError(f"{kind} tensor bundle is missing a physical CMOS tensor: {_tensor_bundle_path(run, kind)}")
    physical_plane = _resize_like(physical, reference)
    physical_metrics = _metric_record(physical_plane, reference, labels)
    selected_metrics = None
    if torch.is_tensor(selected):
        selected_metrics = _metric_record(_resize_like(selected, reference), reference, labels)
    prototype_audit = _prototype_label_audit(physical_plane, reference, labels)
    focus_failures = _focus_failures(physical_metrics)
    physical_pass = _readable(physical_metrics, corr=PASS_CORR, dice=PASS_DICE)
    selected_readable = _readable(selected_metrics, corr=READABLE_SELECTED_CORR, dice=READABLE_SELECTED_DICE)
    audit_accuracy = prototype_audit.get("correct_ratio")
    audit_pass = audit_accuracy is not None and float(audit_accuracy) >= AUDIT_PASS_ACC
    candidate = bool(physical_pass and not focus_failures and audit_pass)
    mismatch = bool(
        selected_readable
        and not _readable(physical_metrics, corr=READABLE_SELECTED_CORR, dice=READABLE_SELECTED_DICE)
        and selected_source != physical_source
    )
    if candidate:
        status = "physical-cmos-candidate-needs-human-review"
    elif mismatch:
        status = "camera-path-failure"
    else:
        status = "diagnostic-only-failed-physical-cmos"
    return {
        "run_type": kind,
        "run": run.as_posix(),
        "tensor_bundle": _tensor_bundle_path(run, kind).as_posix(),
        "metrics_path": _metrics_path(run, kind).as_posix(),
        "stored_metrics": _load_json(_metrics_path(run, kind), required=False),
        "v4_preview": _v4_preview_path(run, kind).as_posix(),
        "count": int(labels.shape[0]),
        "class_labels": labels.tolist(),
        "reference_key": reference_key,
        "primary_readout_plane": _primary_readout_name(bundle),
        "physical_cmos": {
            "source_tensor": physical_source,
            **physical_metrics,
        },
        "selected_readout": (
            {
                "source_tensor": selected_source,
                **selected_metrics,
            }
            if selected_metrics is not None
            else {"source_tensor": selected_source, "available": False}
        ),
        "prototype_label_audit": prototype_audit,
        "prototype_label_audit_accuracy": audit_accuracy,
        "focus_failures": focus_failures,
        "selected_vs_physical_mismatch": mismatch,
        "camera_path_failure": mismatch,
        "physical_cmos_positive": candidate,
        "human_readability_required": True,
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


def run_physical_cmos_readout(
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
    candidates = [run for run in run_records if run["physical_cmos_positive"]]
    camera_failures = [run for run in run_records if run["camera_path_failure"]]
    failed = [run for run in run_records if run["status"] == "diagnostic-only-failed-physical-cmos"]
    if candidates:
        status = "physical-cmos-candidate-needs-human-review"
    elif camera_failures:
        status = "camera-path-failure"
    else:
        status = "diagnostic-only-failed-physical-cmos"
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v21",
        "objective": "physical_cmos_readout_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "thresholds": {
            "physical_cmos_corr": PASS_CORR,
            "physical_cmos_dice": PASS_DICE,
            "focus_corr": FOCUS_PASS_CORR,
            "focus_dice": FOCUS_PASS_DICE,
            "prototype_audit_accuracy": AUDIT_PASS_ACC,
        },
        "runs": run_records,
        "decision": {
            "status": status,
            "physical_cmos_positive_runs": [run["run"] for run in candidates],
            "camera_path_failure_runs": [run["run"] for run in camera_failures],
            "failed_physical_cmos_runs": [run["run"] for run in failed],
            "human_readability_required": True,
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
    summary = run_physical_cmos_readout(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
