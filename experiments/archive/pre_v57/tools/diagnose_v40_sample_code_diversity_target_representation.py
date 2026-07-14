#!/usr/bin/env python3
"""v40 sample-code diversity and target-representation diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output  # noqa: E402
from tools.diagnose_v34_code_coupling_recovery import _labels, _load_sample_bundle, _preview_paths  # noqa: E402


TARGET_PLANE = "target_code_amplitude"
DECODER_INPUT_PLANE = "decoder_input_amplitude"
DECODER_OUTPUT_PLANE = "decoder_output_amplitude"
PHYSICAL_PLANE = "physical_camera_plane"
PLANES = (TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)

DIVERSITY_MSE_THRESHOLD = 1.0e-5
DIVERSITY_STD_THRESHOLD = 3.0e-4


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / (
        f"twophoton_100x100_v40_sample_code_diversity_target_representation_{timestamp}.json"
    )


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    while value.ndim > 4:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"expected tensor with 2-4 dims, got {tuple(tensor.shape)}")
    mn = value.amin(dim=(-2, -1), keepdim=True)
    mx = value.amax(dim=(-2, -1), keepdim=True)
    return (value - mn) / (mx - mn).clamp_min(1.0e-6)


def _as_flat(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(1)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim not in {2, 4}:
        while value.ndim > 4:
            value = value[:, 0]
    return value.reshape(value.shape[0], -1)


def _same_label_diversity(tensor: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    samples = _as_flat(tensor)
    labels = labels[: samples.shape[0]]
    records = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        if int(mask.sum().item()) < 2:
            continue
        group = samples[mask]
        centered = group - group.mean(dim=0, keepdim=True)
        records.append(
            {
                "label": int(label.item()),
                "count": int(mask.sum().item()),
                "mean_pixel_std": float(group.std(dim=0, unbiased=False).mean().item()),
                "mean_mse_to_class_mean": float(centered.square().mean(dim=1).mean().item()),
            }
        )
    mean_std = sum(item["mean_pixel_std"] for item in records) / len(records) if records else None
    mean_mse = sum(item["mean_mse_to_class_mean"] for item in records) / len(records) if records else None
    active = (
        mean_std is not None
        and mean_mse is not None
        and mean_std >= DIVERSITY_STD_THRESHOLD
        and mean_mse >= DIVERSITY_MSE_THRESHOLD
    )
    return {
        "available": bool(records),
        "active": bool(active),
        "mean_pixel_std": mean_std,
        "mean_mse_to_class_mean": mean_mse,
        "thresholds": {
            "mean_pixel_std": DIVERSITY_STD_THRESHOLD,
            "mean_mse_to_class_mean": DIVERSITY_MSE_THRESHOLD,
        },
        "per_label": records,
    }


def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_n = _normalize(source)
    target_n = _normalize(target)
    if source_n.shape[-2:] != target_n.shape[-2:]:
        source_n = F.interpolate(source_n, size=target_n.shape[-2:], mode="area")
    return source_n


def _corr_mse(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_n = _resize_like(left, right)
    right_n = _normalize(right)
    count = min(int(left_n.shape[0]), int(right_n.shape[0]))
    left_flat = left_n[:count].flatten(1)
    right_flat = right_n[:count].flatten(1)
    left_centered = left_flat - left_flat.mean(dim=1, keepdim=True)
    right_centered = right_flat - right_flat.mean(dim=1, keepdim=True)
    corr = (left_centered * right_centered).sum(dim=1) / (
        left_centered.norm(dim=1) * right_centered.norm(dim=1)
    ).clamp_min(1.0e-8)
    mse = (left_flat - right_flat).square().mean(dim=1)
    return {"corr": float(corr.mean().item()), "mse": float(mse.mean().item())}


def _human_topology_proxy_v2(tensor: torch.Tensor) -> dict[str, Any]:
    value = _normalize(tensor)
    threshold = torch.maximum(
        value.new_full((value.shape[0], 1, 1, 1), 0.45),
        value.mean(dim=(-2, -1), keepdim=True)
        + 0.40 * value.std(dim=(-2, -1), keepdim=True, unbiased=False),
    )
    fg = value >= threshold
    fg_fraction = fg.float().mean(dim=(-2, -1)).flatten()
    contrast = value.flatten(1).std(dim=1, unbiased=False)
    padded = F.pad(fg.float(), (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    edges = (
        (center - padded[..., :-2, 1:-1]).abs()
        + (center - padded[..., 2:, 1:-1]).abs()
        + (center - padded[..., 1:-1, :-2]).abs()
        + (center - padded[..., 1:-1, 2:]).abs()
    ) > 0
    edge_per_foreground = edges.float().flatten(1).mean(dim=1) / fg_fraction.clamp_min(1.0e-6)
    valid_fg = ((fg_fraction >= 0.01) & (fg_fraction <= 0.35)).float()
    score = (
        0.35 * valid_fg
        + 0.30 * torch.clamp(contrast / 0.22, 0.0, 1.0)
        + 0.35 * torch.clamp(edge_per_foreground / 1.6, 0.0, 1.0)
    )
    return {
        "shape": list(tensor.shape),
        "score": float(score.mean().item()),
        "readable": bool(score.mean().item() >= 0.55),
        "mean_foreground_fraction": float(fg_fraction.mean().item()),
        "mean_contrast": float(contrast.mean().item()),
        "mean_edge_per_foreground": float(edge_per_foreground.mean().item()),
        "threshold": 0.55,
    }


def _sample_metrics(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "artifacts" / "one_shot_sample_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def _render_cmos_review(
    label: str,
    run: Path,
    physical: torch.Tensor,
    output_dir: Path | None,
    *,
    class_labels: torch.Tensor,
    review_version: str,
    strict_class_coverage: bool = True,
) -> dict[str, Any]:
    if output_dir is None:
        return {"available": False, "reason": "review output directory was not requested"}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_cmos_only_review.png"
    values = _normalize(physical)
    labels = class_labels.detach().long().cpu().reshape(-1)
    if labels.numel() != values.shape[0]:
        raise ValueError(
            "class labels must match physical_camera_plane batch size for CMOS review, "
            f"got {labels.numel()} labels and {values.shape[0]} images"
        )
    required_classes = tuple(range(10))
    missing_classes = [class_id for class_id in required_classes if not bool((labels == class_id).any())]
    if missing_classes:
        if strict_class_coverage:
            raise ValueError(f"missing required review classes: {missing_classes}")
        present_classes = tuple(int(value) for value in torch.unique(labels, sorted=True).tolist())
        review_classes = present_classes
    else:
        review_classes = required_classes
    review_indices = [int((labels == class_id).nonzero(as_tuple=False)[0].item()) for class_id in review_classes]
    count = len(review_indices)
    fig, axes = plt.subplots(1, count, figsize=(max(2, count * 1.4), 1.8))
    if count == 1:
        axes = [axes]
    for axis, class_id, sample_index in zip(axes, review_classes, review_indices, strict=True):
        axis.imshow(values[sample_index, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(str(class_id), fontsize=8)
    fig.suptitle(f"{review_version} CMOS-only review: {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "available": True,
        "run": run.as_posix(),
        "cmos_only_review_image": path.as_posix(),
        "final_human_gate_plane": PHYSICAL_PLANE,
        "final_human_gate_label": "CMOS intensity",
        "review_planes": [PHYSICAL_PLANE],
        "review_class_labels": list(review_classes),
        "review_indices": review_indices,
        "missing_review_classes": missing_classes,
        "class_coverage_complete": not missing_classes,
        "auxiliary_planes_excluded": [TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE],
    }


def analyze_sample_run(
    run: Path,
    *,
    label: str,
    target_mode: str,
    diversity_weight: float,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    physical_vs_decoder = _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE])
    preview = _preview_paths(run)
    target_active = diversity[TARGET_PLANE]["active"]
    decoder_active = diversity[DECODER_OUTPUT_PLANE]["active"]
    physical_active = diversity[PHYSICAL_PLANE]["active"]
    target_readable = topology[TARGET_PLANE]["readable"] and topology[DECODER_INPUT_PLANE]["readable"]
    decoder_readable = topology[DECODER_OUTPUT_PLANE]["readable"]
    physical_readable = topology[PHYSICAL_PLANE]["readable"]
    if not target_active or not decoder_active or not physical_active:
        status = "code-diversity-still-collapsed"
        reason = "same-label sample diversity remains collapsed before final review"
    elif not target_readable:
        status = "target-representation-still-blob"
        reason = "target-code or decoder-input topology remains blob-like"
    elif not decoder_readable:
        status = "bridge-after-diversity-failure"
        reason = "target-code is readable but decoder output loses topology"
    elif not physical_readable:
        status = "camera-after-diversity-failure"
        reason = "decoder output is readable but final CMOS remains blob-like"
    else:
        status = "human-review-required"
        reason = "diversity and topology proxies passed; final CMOS still requires human review"
    return {
        "label": label,
        "run": run.as_posix(),
        "target_code_diagnostic_mode": target_mode,
        "sample_code_diversity_weight": float(diversity_weight),
        "status": status,
        "classification": status,
        "reason": reason,
        "same_label_diversity": diversity,
        "human_topology_proxy_v2": topology,
        "physical_vs_decoder_corr_mse": physical_vs_decoder,
        "sample_metrics": _sample_metrics(run),
        "preview_package": preview,
        "cmos_only_review_package": _render_cmos_review(
            label,
            run,
            bundle[PHYSICAL_PLANE],
            review_dir,
            class_labels=labels,
            review_version="v40",
            strict_class_coverage=False,
        ),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _cross_probe_similarity(records: list[dict[str, Any]]) -> dict[str, Any]:
    physicals = []
    for record in records:
        try:
            bundle = _load_sample_bundle(Path(record["run"]))
            physicals.append((record["label"], bundle[PHYSICAL_PLANE]))
        except Exception:
            continue
    pairs = []
    for left_index in range(len(physicals)):
        for right_index in range(left_index + 1, len(physicals)):
            left_label, left_tensor = physicals[left_index]
            right_label, right_tensor = physicals[right_index]
            metrics = _corr_mse(left_tensor, right_tensor)
            pairs.append({"left": left_label, "right": right_label, **metrics})
    if not pairs:
        return {"available": False, "collapsed": False, "pairs": []}
    mean_corr = sum(pair["corr"] for pair in pairs) / len(pairs)
    mean_mse = sum(pair["mse"] for pair in pairs) / len(pairs)
    return {
        "available": True,
        "collapsed": bool(mean_corr >= 0.98 or mean_mse <= 1.0e-3),
        "mean_corr": mean_corr,
        "mean_mse": mean_mse,
        "pairs": pairs,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    cross_probe = _cross_probe_similarity(records)
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
        next_gate = "repair_v40_inputs"
    elif by_status.get("human-review-required"):
        status = "label-template-collapse-recovered"
        next_gate = "v41_formal_cmos_manual_review_candidate"
    elif by_status.get("camera-after-diversity-failure"):
        status = "camera-after-diversity-failure"
        next_gate = "physical_cmos_readout_followup"
    elif by_status.get("bridge-after-diversity-failure"):
        status = "bridge-after-diversity-failure"
        next_gate = "bridge_or_decoder_topology_followup"
    elif by_status.get("target-representation-still-blob"):
        status = "target-representation-still-blob"
        next_gate = "target_code_topology_redesign"
    else:
        status = "code-diversity-still-collapsed"
        next_gate = "non_template_generator_or_command_head_redesign"
    return {
        "status": status,
        "status_runs": by_status,
        "cross_probe_similarity": cross_probe,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def run_v40_sample_code_diversity_target_representation(
    *,
    source_v39_report: Path | None = None,
    source_v33_1_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v39_status = None
    if source_v39_report is not None:
        payload = _load_json(source_v39_report)
        assert payload is not None
        source_v39_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        parts = entry.split("|")
        while len(parts) < 5:
            parts.append("")
        label, target_mode, diversity_weight, sample_run, train_run = parts[:5]
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    target_mode=target_mode,
                    diversity_weight=float(diversity_weight or 0.0),
                    review_dir=review_dir,
                )
            )
        except Exception as exc:
            errors.append({"probe": label, "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label,
                    "target_code_diagnostic_mode": target_mode,
                    "sample_code_diversity_weight": float(diversity_weight or 0.0),
                    "train_run": train_run or None,
                    "run": sample_run or None,
                    "status": "harness-invalid",
                    "reason": "probe analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": "v40",
        "objective": "sample_code_diversity_target_representation_redesign",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v39_report": source_v39_report.as_posix() if source_v39_report else None,
        "source_v39_status": source_v39_status,
        "source_v33_1_summary": source_v33_1_summary.as_posix() if source_v33_1_summary else None,
        "probes": records,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane human review can satisfy acceptance.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v39-report", type=Path)
    parser.add_argument("--source-v33-1-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v40_sample_code_diversity_target_representation(
        source_v39_report=args.source_v39_report,
        source_v33_1_summary=args.source_v33_1_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
