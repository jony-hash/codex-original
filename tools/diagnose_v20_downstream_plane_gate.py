#!/usr/bin/env python3
"""Offline v20 downstream D2NN/readout plane-gate diagnostic.

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
PLANE_KEYS = (
    "slm_command",
    "encoder_input_amplitude",
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_amplitude",
    "chip_intensity_primary",
    "probe_plane",
    "diffractive_camera_plane_raw",
    "generated_camera",
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
PLANE_ALIASES = {
    "chip_intensity_primary": ("chip_intensity_primary", "chip_intensity_display", "chip_intensity"),
    "probe_plane": ("probe_plane", "probe_plane_loss"),
    "generated_camera": ("generated_camera", "generated_camera_raw", "student_camera_raw", "camera_plane"),
}
GROSS_COLLAPSE_CORR = 0.50
GROSS_COLLAPSE_DICE = 0.50
READABLE_PLANE_CORR = 0.60
READABLE_PLANE_DICE = 0.55


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v20_downstream_plane_gate_{timestamp}.json"


def _require_report_output(path: Path) -> None:
    normalized = path.as_posix()
    if path.is_absolute() or not normalized.startswith("outputs/reports/"):
        raise ValueError("v20 downstream plane-gate output must be a relative path under outputs/reports/")


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


def _first_tensor(bundle: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, torch.Tensor] | tuple[None, None]:
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, value
    return None, None


def _resolve_plane(bundle: dict[str, Any], key: str) -> tuple[str | None, torch.Tensor | None]:
    aliases = PLANE_ALIASES.get(key, (key,))
    return _first_tensor(bundle, aliases)


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
    bundle["_v20_label_key"] = label_key
    bundle["_v20_reference_key"] = reference_key
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


def _plane_audit(bundle: dict[str, Any], reference: torch.Tensor, labels: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    first_collapse: dict[str, Any] | None = None
    for key in PLANE_KEYS:
        source_key, value = _resolve_plane(bundle, key)
        if not torch.is_tensor(value):
            records.append({"plane": key, "available": False, "reason": "missing"})
            continue
        plane = _resize_like(value, reference)
        metrics = compute_sample_quality_metrics(plane, reference, labels)
        record = {
            "plane": key,
            "source_key": source_key,
            "available": True,
            **_metric_record(metrics),
        }
        overall = record["overall"]
        collapsed = bool(
            float(overall.get("corr") or 0.0) < GROSS_COLLAPSE_CORR
            or float(overall.get("dice") or 0.0) < GROSS_COLLAPSE_DICE
        )
        record["gross_collapse"] = collapsed
        record["readable_intermediate"] = bool(
            float(overall.get("corr") or 0.0) >= READABLE_PLANE_CORR
            and float(overall.get("dice") or 0.0) >= READABLE_PLANE_DICE
        )
        if collapsed and first_collapse is None:
            first_collapse = {
                "plane": key,
                "source_key": source_key,
                "corr": overall.get("corr"),
                "dice": overall.get("dice"),
                "thresholds": {"corr": GROSS_COLLAPSE_CORR, "dice": GROSS_COLLAPSE_DICE},
            }
        records.append(record)
    return records, first_collapse


def _record_for(records: list[dict[str, Any]], plane: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("plane") == plane and record.get("available")), None)


def _is_readable(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    overall = record.get("overall", {})
    return bool(
        float(overall.get("corr") or 0.0) >= READABLE_PLANE_CORR
        and float(overall.get("dice") or 0.0) >= READABLE_PLANE_DICE
    )


def _best_readable_intermediate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        record for record in records
        if record.get("available") and record.get("plane") != "generated_camera"
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda record: float(record["overall"].get("corr") or 0.0) + float(record["overall"].get("dice") or 0.0),
    )
    return {
        "plane": best["plane"],
        "source_key": best.get("source_key"),
        "corr": best["overall"].get("corr"),
        "dice": best["overall"].get("dice"),
        "readable_intermediate": _is_readable(best),
    }


def _collapse_after(first_collapse: dict[str, Any] | None, plane: str) -> bool:
    if first_collapse is None:
        return True
    try:
        return PLANE_KEYS.index(first_collapse["plane"]) > PLANE_KEYS.index(plane)
    except ValueError:
        return False


def _classify(records: list[dict[str, Any]], first_collapse: dict[str, Any] | None) -> dict[str, Any]:
    encoder_readable = _is_readable(_record_for(records, "encoder_input_amplitude"))
    chip_readable = _is_readable(_record_for(records, "chip_amplitude")) or _is_readable(_record_for(records, "chip_intensity_primary"))
    probe_readable = _is_readable(_record_for(records, "probe_plane"))
    camera_readable = _is_readable(_record_for(records, "generated_camera"))
    first_plane = first_collapse.get("plane") if first_collapse else None
    d2nn_planes = {
        "encoded_amplitude",
        "ols_output_amplitude",
        "decoder_output_amplitude",
        "chip_amplitude",
        "chip_intensity_primary",
    }
    return {
        "downstream_localized": first_collapse is not None,
        "camera_path_failure": bool((chip_readable or probe_readable) and not camera_readable),
        "d2nn_failure": bool(encoder_readable and first_plane in d2nn_planes),
        "collapse_moved_past_decoder_output_amplitude": _collapse_after(first_collapse, "decoder_output_amplitude"),
        "front_to_encoder_readable": encoder_readable,
        "chip_or_probe_readable": bool(chip_readable or probe_readable),
        "generated_camera_readable": camera_readable,
    }


def analyze_run(run: Path, *, kind: str) -> dict[str, Any]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle = _load_bundle(run, kind)
    label_key = bundle["_v20_label_key"]
    reference_key = bundle["_v20_reference_key"]
    labels = bundle[label_key].detach().cpu().long()
    reference = bundle[reference_key].detach().float().cpu()
    records, first_collapse = _plane_audit(bundle, reference, labels)
    generated_source, generated = _resolve_plane(bundle, "generated_camera")
    prototype_audit = (
        _prototype_label_audit(_resize_like(generated, reference), reference, labels)
        if torch.is_tensor(generated)
        else {"available": False, "reason": "generated_camera plane missing", "correct_ratio": None, "confusion": {}}
    )
    best_intermediate = _best_readable_intermediate(records)
    classification = _classify(records, first_collapse)
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
        "generated_camera_source_key": generated_source,
        "plane_audit": records,
        "prototype_label_audit": prototype_audit,
        "first_gross_collapse_plane": first_collapse,
        "best_readable_intermediate_plane": best_intermediate,
        **classification,
        "status": (
            "diagnostic-only-downstream-plane-localized"
            if first_collapse is not None
            else "diagnostic-only-no-gross-collapse-detected"
        ),
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


def run_downstream_plane_gate(
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
    localized = [run for run in run_records if run["downstream_localized"]]
    camera_failures = [run for run in run_records if run["camera_path_failure"]]
    d2nn_failures = [run for run in run_records if run["d2nn_failure"]]
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v20",
        "objective": "downstream_d2nn_readout_plane_gate",
        "source_summary": summary.as_posix() if summary is not None else None,
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "thresholds": {
            "gross_collapse_corr": GROSS_COLLAPSE_CORR,
            "gross_collapse_dice": GROSS_COLLAPSE_DICE,
            "readable_plane_corr": READABLE_PLANE_CORR,
            "readable_plane_dice": READABLE_PLANE_DICE,
        },
        "plane_keys": list(PLANE_KEYS),
        "runs": run_records,
        "decision": {
            "downstream_localized": bool(localized),
            "localized_runs": [run["run"] for run in localized],
            "camera_path_failure_runs": [run["run"] for run in camera_failures],
            "d2nn_failure_runs": [run["run"] for run in d2nn_failures],
            "accepted_summary_written": False,
            "status": (
                "diagnostic-only-downstream-localized"
                if localized
                else "diagnostic-only-no-downstream-collapse-localized"
            ),
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
    summary = run_downstream_plane_gate(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
