#!/usr/bin/env python3
"""Offline v16 readout calibration audit for failed one-shot sample runs.

This diagnostic only reads existing sample tensor bundles and writes a report
under outputs/reports/. It does not train, promote checkpoints, or write an
accepted summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any


import torch

from ldm_project.sampling_reports import compute_sample_quality_metrics
from ldm_project.visualization_core import (
    enhance_camera_for_display,
    readable_camera_for_display,
)


DEFAULT_SAMPLE_RUNS = (
    Path("outputs/20260625_v15/runs/20260629_153304_844767_train_v15_ac_roi_diagnostic_sample"),
    Path(
        "outputs/20260630_v16/runs/"
        "20260630_143402_638868_train_v16_probe_a_sample_background_recover_from_ac_e6_sample"
    ),
    Path(
        "outputs/20260630_v16/runs/"
        "20260630_143451_151228_train_v16_probe_b_foreground_shape_recover_from_ac_e6_sample"
    ),
)

QUANTILE_WINDOWS = ((0.01, 0.995), (0.02, 0.98), (0.05, 0.95))
GAMMAS = (0.5, 0.65, 0.8, 1.0)
SMOOTH_SIGMAS = (0.0, 0.4, 0.6, 1.0)
FOREGROUND_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)
FOCUS_CLASSES = (3, 4, 7, 9)
CORR_GATE = 0.78
DICE_GATE = 0.72


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v16_readout_calibration_{timestamp}.json"


def _require_report_output(path: Path) -> None:
    normalized = path.as_posix()
    if path.is_absolute() or not normalized.startswith("outputs/reports/"):
        raise ValueError("v16 readout calibration output must be a relative path under outputs/reports/")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"tensor bundle must contain a dict payload: {path}")
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


def _reference_bank(reference_images: torch.Tensor, reference_labels: torch.Tensor | None, labels: torch.Tensor):
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
    return torch.nn.functional.normalize(scaled, dim=1, eps=1.0e-6)


def _prototype_label_audit(camera: torch.Tensor, labels: torch.Tensor, reference_bank: dict[int, torch.Tensor]) -> dict[str, Any]:
    if not reference_bank:
        return {"available": False, "reason": "empty reference bank"}
    class_labels = sorted(reference_bank)
    prototypes = torch.stack([reference_bank[label].detach().float().mean(dim=0) for label in class_labels])
    if prototypes.shape[-2:] != camera.shape[-2:]:
        prototypes = torch.nn.functional.interpolate(prototypes, size=camera.shape[-2:], mode="bilinear", align_corners=False)
    correlations = _normalize_features(camera) @ _normalize_features(prototypes).t()
    nearest_indices = correlations.argmax(dim=1)
    requested = labels.detach().cpu().long()
    nearest_labels = torch.tensor([class_labels[int(index)] for index in nearest_indices.tolist()], dtype=torch.long)
    correct = nearest_labels == requested
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted({int(item) for item in requested.tolist()}):
        mask = requested == label
        nearest_for_class = nearest_labels[mask]
        counts: dict[str, int] = {}
        for item in nearest_for_class.tolist():
            counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
        per_class[str(label)] = {
            "count": int(mask.sum().item()),
            "correct_ratio": float(correct[mask].float().mean().item()) if bool(mask.any()) else 0.0,
            "nearest_label_counts": counts,
        }
    return {
        "available": True,
        "class_labels": class_labels,
        "correct_ratio": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "per_class": per_class,
    }


def _quantile_gamma_transform(camera: torch.Tensor, low_q: float, high_q: float, gamma: float, smooth_sigma: float) -> torch.Tensor:
    raw = camera.detach().float()
    flat = raw.flatten(start_dim=-2)
    low = flat.quantile(low_q, dim=-1, keepdim=True).unsqueeze(-1)
    high = flat.quantile(high_q, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = ((raw - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    if smooth_sigma > 0.0:
        scaled = readable_camera_for_display(
            scaled,
            lower_quantile=0.0,
            upper_quantile=1.0,
            gamma=1.0,
            smooth_sigma=smooth_sigma,
            foreground_boost=1.0,
        )
    if gamma != 1.0:
        scaled = scaled.pow(gamma)
    return scaled.clamp(0.0, 1.0)


def _iter_transforms(camera: torch.Tensor):
    yield {
        "id": "raw_clamped",
        "kind": "raw_clamped",
        "config": {},
        "camera": camera.detach().float().clamp(0.0, 1.0),
    }
    yield {
        "id": "enhance_camera_for_display_default",
        "kind": "enhance_camera_for_display",
        "config": {"lower_quantile": 0.01, "upper_quantile": 0.995, "gamma": 0.75},
        "camera": enhance_camera_for_display(camera),
    }
    yield {
        "id": "readable_camera_for_display_default",
        "kind": "readable_camera_for_display",
        "config": {
            "lower_quantile": 0.02,
            "upper_quantile": 0.98,
            "gamma": 0.65,
            "smooth_sigma": 0.6,
            "foreground_boost": 1.3,
        },
        "camera": readable_camera_for_display(camera),
    }
    for low_q, high_q in QUANTILE_WINDOWS:
        for gamma in GAMMAS:
            for smooth_sigma in SMOOTH_SIGMAS:
                transform_id = f"q{low_q:g}_{high_q:g}_g{gamma:g}_s{smooth_sigma:g}".replace(".", "p")
                yield {
                    "id": transform_id,
                    "kind": "quantile_gamma_smooth",
                    "config": {
                        "lower_quantile": low_q,
                        "upper_quantile": high_q,
                        "gamma": gamma,
                        "smooth_sigma": smooth_sigma,
                    },
                    "camera": _quantile_gamma_transform(camera, low_q, high_q, gamma, smooth_sigma),
                }


def _metric_record(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics.get("overall", {})
    per_class = metrics.get("per_class", {})
    return {
        "overall": {
            "corr": overall.get("corr"),
            "dice": overall.get("dice"),
            "hole_proxy": overall.get("hole_proxy"),
        },
        "focus_classes": {
            str(label): per_class.get(str(label), {})
            for label in FOCUS_CLASSES
            if str(label) in per_class
        },
    }


def _score_for_sort(record: dict[str, Any], source: str) -> tuple[float, float]:
    metrics = record[source]["overall"]
    return (float(metrics.get("corr") or 0.0), float(metrics.get("dice") or 0.0))


def _save_preview_image(path: Path, labels: torch.Tensor, reference: torch.Tensor, raw: torch.Tensor, calibrated: torch.Tensor) -> None:
    matplotlib_cache = path.parent.parent / "matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(20, int(labels.shape[0]))
    fig, axes = plt.subplots(count, 3, figsize=(4.5, max(2.0, 1.05 * count)))
    if count == 1:
        axes = axes.reshape(1, 3)
    columns = ("reference", "raw CMOS", "calibrated CMOS")
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=8)
    for row in range(count):
        row_tensors = (reference[row, 0], raw[row, 0], calibrated[row, 0])
        for col, tensor in enumerate(row_tensors):
            axes[row, col].imshow(tensor.detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(str(int(labels[row].item())), rotation=0, labelpad=8, fontsize=8, va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _best_gate_status(record: dict[str, Any]) -> dict[str, Any]:
    exact = record["exact"]["overall"]
    nearest = record["nearest_neighbor"]["overall"]
    exact_pass = bool(float(exact.get("corr") or 0.0) >= CORR_GATE and float(exact.get("dice") or 0.0) >= DICE_GATE)
    nearest_pass = bool(
        float(nearest.get("corr") or 0.0) >= CORR_GATE and float(nearest.get("dice") or 0.0) >= DICE_GATE
    )
    return {
        "exact_metric_passed": exact_pass,
        "nearest_neighbor_metric_passed": nearest_pass,
        "metric_passed": exact_pass or nearest_pass,
        "reference_mismatch_finding": bool(nearest_pass and not exact_pass),
    }


def analyze_sample_run(sample_run: Path, *, output_dir: Path, generate_previews: bool = True) -> dict[str, Any]:
    bundle = _load_sample_bundle(sample_run)
    labels = bundle["class_labels"].detach().cpu().long()
    reference = bundle["reference_images"].detach().float().cpu()
    reference_labels = bundle.get("reference_labels")
    if torch.is_tensor(reference_labels):
        reference_labels = reference_labels.detach().cpu().long()
    else:
        reference_labels = labels
    camera = bundle["_camera_for_audit"].detach().float().cpu()
    if camera.shape != reference.shape:
        raise ValueError(f"generated camera and reference images must share shape for {sample_run}: {camera.shape} != {reference.shape}")
    bank = _reference_bank(reference, reference_labels, labels)

    transform_records = []
    for transform in _iter_transforms(camera):
        calibrated = transform["camera"].detach().float().cpu().clamp(0.0, 1.0)
        threshold_records = []
        for threshold in FOREGROUND_THRESHOLDS:
            metrics = compute_sample_quality_metrics(
                calibrated,
                reference,
                labels,
                foreground_threshold=threshold,
                reference_image_bank=bank,
                reference_bank_source="sample_reference_images_by_class",
            )
            nearest_metrics = metrics.get("nearest_neighbor", {})
            record = {
                "foreground_threshold": threshold,
                "exact": _metric_record(metrics),
                "nearest_neighbor": _metric_record(nearest_metrics),
                "prototype_label_audit": _prototype_label_audit(calibrated, labels, bank),
            }
            record["gate"] = _best_gate_status(record)
            threshold_records.append(record)
        best_exact = max(threshold_records, key=lambda item: _score_for_sort(item, "exact"))
        best_nearest = max(threshold_records, key=lambda item: _score_for_sort(item, "nearest_neighbor"))
        transform_records.append(
            {
                "id": transform["id"],
                "kind": transform["kind"],
                "config": transform["config"],
                "thresholds": threshold_records,
                "best_exact": best_exact,
                "best_nearest_neighbor": best_nearest,
                "_camera": calibrated,
            }
        )

    top_records = sorted(
        transform_records,
        key=lambda item: max(
            _score_for_sort(item["best_exact"], "exact"),
            _score_for_sort(item["best_nearest_neighbor"], "nearest_neighbor"),
        ),
        reverse=True,
    )[:3]
    preview_records = []
    preview_dir = output_dir / "v16_readout_calibration_previews"
    for rank, record in enumerate(top_records, start=1):
        image_path = preview_dir / f"{sample_run.name}_rank{rank}_{record['id']}.png"
        if generate_previews:
            _save_preview_image(image_path, labels, reference, camera.clamp(0.0, 1.0), record["_camera"])
        preview_records.append(
            {
                "rank": rank,
                "transform_id": record["id"],
                "image": image_path.as_posix(),
                "generated": bool(generate_previews),
            }
        )

    metric_positive = [
        {
            "transform_id": record["id"],
            "foreground_threshold": threshold["foreground_threshold"],
            **threshold["gate"],
        }
        for record in transform_records
        for threshold in record["thresholds"]
        if threshold["gate"]["metric_passed"]
    ]
    sanitized_transforms = []
    for record in transform_records:
        public_record = {key: value for key, value in record.items() if key != "_camera"}
        sanitized_transforms.append(public_record)
    return {
        "sample_run": sample_run.as_posix(),
        "sample_tensor_bundle": (sample_run / "artifacts" / "one_shot_sample_tensors.pt").as_posix(),
        "sample_metrics": (sample_run / "artifacts" / "one_shot_sample_metrics.json").as_posix(),
        "count": int(labels.shape[0]),
        "class_labels": labels.tolist(),
        "reference_bank_source": "sample_reference_images_by_class",
        "reference_counts": {str(label): int(images.shape[0]) for label, images in sorted(bank.items())},
        "transforms": sanitized_transforms,
        "top_preview_candidates": preview_records,
        "decision": {
            "metric_positive_candidates": metric_positive,
            "metric_positive": bool(metric_positive),
            "human_readability_required": True,
            "accepted_summary_written": False,
            "sample_promotion": "diagnostic-only",
            "calibration_status": (
                "metric-positive-needs-human-review" if metric_positive else "calibration-negative"
            ),
        },
    }


def run_readout_calibration(
    *,
    sample_runs: list[Path] | None = None,
    eval_runs: list[Path] | None = None,
    output: Path | None = None,
    generate_previews: bool = True,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    output_dir = output_path.parent
    runs = sample_runs or list(DEFAULT_SAMPLE_RUNS)
    run_summaries = [
        analyze_sample_run(Path(run), output_dir=output_dir, generate_previews=generate_previews)
        for run in runs
    ]
    metric_positive_runs = [
        run["sample_run"]
        for run in run_summaries
        if run["decision"]["metric_positive"]
    ]
    summary = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v16",
        "objective": "readout_calibration_audit",
        "sample_runs": [Path(run).as_posix() for run in runs],
        "eval_runs": [Path(run).as_posix() for run in (eval_runs or [])],
        "thresholds": {
            "corr": CORR_GATE,
            "dice": DICE_GATE,
            "foreground_thresholds": list(FOREGROUND_THRESHOLDS),
        },
        "transform_sweep": {
            "quantile_windows": [list(item) for item in QUANTILE_WINDOWS],
            "gammas": list(GAMMAS),
            "smooth_sigmas": list(SMOOTH_SIGMAS),
            "includes": [
                "raw_clamped",
                "enhance_camera_for_display_default",
                "readable_camera_for_display_default",
            ],
        },
        "runs": run_summaries,
        "decision": {
            "metric_positive_runs": metric_positive_runs,
            "metric_positive": bool(metric_positive_runs),
            "human_readability_required": True,
            "accepted_summary_written": False,
            "recommendation": (
                "Inspect top preview candidates before formalizing a fixed readout calibration."
                if metric_positive_runs
                else "Calibration-negative: stop v16 loss/config micro-tuning and open v17 readout/camera/OLS path."
            ),
            "nearest_neighbor_success_policy": (
                "Nearest-class metric success is a reference-mismatch finding, not old exact-reference promotion."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Optional eval run recorded in the report.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    parser.add_argument("--no-previews", action="store_true", help="Skip PNG preview generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_readout_calibration(
        sample_runs=args.sample_run,
        eval_runs=args.eval_run,
        output=args.output,
        generate_previews=not args.no_previews,
    )
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
