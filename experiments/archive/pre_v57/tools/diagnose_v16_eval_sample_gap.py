#!/usr/bin/env python3
"""Offline v16 eval-vs-sample gap diagnostic for the 100x100 phase-only path."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_AC_EVAL_RUN = Path(
    "outputs/20260625_v15/runs/"
    "20260627_161344_557545_train_v15_probe_ac_background_recover_from_ab_e5_eval"
)
DEFAULT_AC_SAMPLE_RUN = Path(
    "outputs/20260625_v15/runs/"
    "20260629_153304_844767_train_v15_ac_roi_diagnostic_sample"
)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    data = tensor.detach().float().cpu()
    flat = data.flatten()
    if flat.numel() == 0:
        return {"shape": list(data.shape), "mean": 0.0, "std": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "shape": list(data.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "p01": float(flat.quantile(0.01).item()),
        "p50": float(flat.quantile(0.50).item()),
        "p99": float(flat.quantile(0.99).item()),
    }


def _centered_features(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scaled = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return F.normalize(scaled, dim=1, eps=1.0e-6)


def _pair_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    count = min(int(a.shape[0]), int(b.shape[0]))
    if count <= 0:
        return {"count": 0, "mean_corr": 0.0, "mean_abs_diff": 0.0}
    left = a[:count].detach().float().cpu()
    right = b[:count].detach().float().cpu()
    if left.shape[-2:] != right.shape[-2:]:
        right = F.interpolate(right, size=left.shape[-2:], mode="area")
    corr = (_centered_features(left) * _centered_features(right)).sum(dim=1)
    return {
        "count": count,
        "mean_corr": float(corr.mean().item()),
        "min_corr": float(corr.min().item()),
        "mean_abs_diff": float((left - right).abs().mean().item()),
    }


def _metrics_summary(path: Path) -> dict[str, Any]:
    metrics = _json_load(path)
    overall = metrics.get("overall", {})
    strict = metrics.get("strict_readability_acceptance", {})
    return {
        "path": path.as_posix(),
        "corr": overall.get("paired_corr", overall.get("corr")),
        "dice": overall.get("paired_dice", overall.get("dice")),
        "high_sat": overall.get("student_high_saturation_ratio", overall.get("high_saturation_ratio")),
        "label_acc": overall.get("student_label_correct_ratio"),
        "min_class_label_acc": overall.get("min_class_student_label_correct_ratio"),
        "background_speckle": overall.get("background_speckle_ratio"),
        "foreground_hole": overall.get("foreground_hole_ratio"),
        "readable_contrast": overall.get("readable_contrast"),
        "strict_passed": strict.get("passed"),
    }


def _chain_stats(bundle: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "slm_command",
        "slm_amplitude",
        "encoder_input_grid_amplitude",
        "encoded_amplitude",
        "decoder_input_amplitude",
        "decoder_output_amplitude",
        "student_camera_raw",
        "generated_camera_raw",
        "camera_plane",
    )
    return {key: _tensor_stats(bundle[key]) for key in keys if key in bundle and torch.is_tensor(bundle[key])}


def _chain_similarity(eval_bundle: dict[str, Any], sample_bundle: dict[str, Any]) -> dict[str, Any]:
    pairs = (
        ("slm_command", "slm_command"),
        ("slm_amplitude", "slm_amplitude"),
        ("encoder_input_grid_amplitude", "encoder_input_grid_amplitude"),
        ("encoded_amplitude", "encoded_amplitude"),
        ("decoder_input_amplitude", "decoder_input_amplitude"),
        ("decoder_output_amplitude", "decoder_output_amplitude"),
    )
    out = {}
    for eval_key, sample_key in pairs:
        if eval_key in eval_bundle and sample_key in sample_bundle:
            out[eval_key] = _pair_similarity(eval_bundle[eval_key], sample_bundle[sample_key])
    eval_camera = eval_bundle.get("student_camera_raw", eval_bundle.get("student_camera_display"))
    sample_camera = sample_bundle.get("generated_camera_raw", sample_bundle.get("camera_plane"))
    if torch.is_tensor(eval_camera) and torch.is_tensor(sample_camera):
        out["camera"] = _pair_similarity(eval_camera, sample_camera)
    return out


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v16_eval_sample_gap_summary_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run", type=Path, default=DEFAULT_AC_EVAL_RUN)
    parser.add_argument("--sample-run", type=Path, default=DEFAULT_AC_SAMPLE_RUN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or _default_output()
    if not str(output).startswith("outputs/reports/"):
        raise ValueError("v16 eval/sample gap output must be under outputs/reports/")
    eval_run = args.eval_run
    sample_run = args.sample_run
    eval_bundle_path = eval_run / "artifacts" / "one_shot_distill_eval_tensors.pt"
    sample_bundle_path = sample_run / "artifacts" / "one_shot_sample_tensors.pt"
    eval_metrics_path = eval_run / "artifacts" / "one_shot_distill_eval_metrics.json"
    sample_metrics_path = sample_run / "artifacts" / "one_shot_sample_metrics.json"
    for path in (eval_bundle_path, sample_bundle_path, eval_metrics_path, sample_metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    eval_bundle = _torch_load(eval_bundle_path)
    sample_bundle = _torch_load(sample_bundle_path)
    eval_metrics = _metrics_summary(eval_metrics_path)
    sample_metrics = _metrics_summary(sample_metrics_path)
    sample_corr = float(sample_metrics.get("corr") or 0.0)
    sample_dice = float(sample_metrics.get("dice") or 0.0)
    sample_strict = bool(sample_corr >= 0.78 and sample_dice >= 0.72)
    eval_strict = bool(eval_metrics.get("strict_passed"))
    camera_similarity = _chain_similarity(eval_bundle, sample_bundle).get("camera", {})
    camera_corr = float(camera_similarity.get("mean_corr", 0.0))
    decision = {
        "eval_strict_passed": eval_strict,
        "sample_strict_passed": sample_strict,
        "sample_training_needed": bool(eval_strict and not sample_strict),
        "reference_metric_mismatch_suspected": bool(eval_strict and not sample_strict and camera_corr >= 0.80),
        "recommendation": (
            "Train v16 sample/readout probes A/B."
            if eval_strict and not sample_strict and camera_corr < 0.80
            else "Inspect sample audit/reference mismatch before training."
            if eval_strict and not sample_strict
            else "No v16 training needed from this diagnostic."
        ),
    }
    summary = {
        "diagnostic_only": True,
        "version": "v16",
        "objective": "eval_vs_sample_gap",
        "accepted_summary_written": False,
        "eval_run": eval_run.as_posix(),
        "sample_run": sample_run.as_posix(),
        "eval_metrics": eval_metrics,
        "sample_metrics": sample_metrics,
        "eval_chain_stats": _chain_stats(eval_bundle),
        "sample_chain_stats": _chain_stats(sample_bundle),
        "eval_sample_chain_similarity": _chain_similarity(eval_bundle, sample_bundle),
        "v4_style_preview": {
            "sample_image": (sample_run / "images" / "one_shot_sample_100x100_cell_chain_preview.png").as_posix(),
            "sample_artifact": (sample_run / "artifacts" / "one_shot_sample_100x100_cell_chain_preview.json").as_posix(),
        },
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_path={output}")


if __name__ == "__main__":
    main()
