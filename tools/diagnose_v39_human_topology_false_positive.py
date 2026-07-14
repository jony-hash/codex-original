#!/usr/bin/env python3
"""v39 human-topology false-positive and blob-origin diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v34_code_coupling_recovery import _load_sample_bundle


VALID_MANUAL_VERDICT = "final_cmos_blob_failed"
TARGET_PLANE = "target_code_amplitude"
DECODER_INPUT_PLANE = "decoder_input_amplitude"
DECODER_OUTPUT_PLANE = "decoder_output_amplitude"
PHYSICAL_PLANE = "physical_camera_plane"
PLANES = (TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v39_human_topology_false_positive_{timestamp}.json"


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    while value.ndim > 4:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"expected plane tensor with 2-4 dims, got {tuple(tensor.shape)}")
    mn = value.amin(dim=(-2, -1), keepdim=True)
    mx = value.amax(dim=(-2, -1), keepdim=True)
    return (value - mn) / (mx - mn).clamp_min(1.0e-6)


def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_n = _normalize(source)
    target_n = _normalize(target)
    if source_n.shape[-2:] != target_n.shape[-2:]:
        source_n = F.interpolate(source_n, size=target_n.shape[-2:], mode="area")
    return source_n


def _corr_mse(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_n = _resize_like(a, b)
    b_n = _normalize(b)
    count = min(int(a_n.shape[0]), int(b_n.shape[0]))
    a_flat = a_n[:count].flatten(1)
    b_flat = b_n[:count].flatten(1)
    a_centered = a_flat - a_flat.mean(dim=1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=1, keepdim=True)
    corr = (a_centered * b_centered).sum(dim=1) / (
        a_centered.norm(dim=1) * b_centered.norm(dim=1)
    ).clamp_min(1.0e-8)
    mse = (a_flat - b_flat).square().mean(dim=1)
    return {"corr": float(corr.mean().item()), "mse": float(mse.mean().item())}


def _human_topology_proxy_v2(tensor: torch.Tensor) -> dict[str, Any]:
    value = _normalize(tensor)
    threshold = torch.maximum(
        value.new_full((value.shape[0], 1, 1, 1), 0.45),
        value.mean(dim=(-2, -1), keepdim=True) + 0.40 * value.std(dim=(-2, -1), keepdim=True, unbiased=False),
    )
    fg = value >= threshold
    fg_fraction = fg.float().mean(dim=(-2, -1)).flatten()
    contrast = value.flatten(1).std(dim=1, unbiased=False)
    height = fg.shape[-2]
    width = fg.shape[-1]
    yy = torch.arange(height).view(1, 1, height, 1).to(fg.device)
    xx = torch.arange(width).view(1, 1, 1, width).to(fg.device)
    mass = fg.float().sum(dim=(-2, -1)).clamp_min(1.0)
    y_mean = (fg.float() * yy).sum(dim=(-2, -1)) / mass
    x_mean = (fg.float() * xx).sum(dim=(-2, -1)) / mass
    y_var = (fg.float() * (yy - y_mean[..., None, None]).square()).sum(dim=(-2, -1)) / mass
    x_var = (fg.float() * (xx - x_mean[..., None, None]).square()).sum(dim=(-2, -1)) / mass
    elongation = torch.sqrt(torch.maximum(x_var, y_var) / torch.minimum(x_var, y_var).clamp_min(1.0e-6)).flatten()
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
    contrast_score = torch.clamp(contrast / 0.22, 0.0, 1.0)
    elongation_score = torch.clamp((elongation - 1.15) / 1.35, 0.0, 1.0)
    stroke_score = torch.clamp(edge_per_foreground / 1.6, 0.0, 1.0)
    score = 0.30 * valid_fg + 0.25 * contrast_score + 0.25 * elongation_score + 0.20 * stroke_score
    return {
        "shape": list(tensor.shape),
        "score": float(score.mean().item()),
        "readable": bool(score.mean().item() >= 0.52),
        "mean_foreground_fraction": float(fg_fraction.mean().item()),
        "mean_contrast": float(contrast.mean().item()),
        "mean_elongation": float(elongation.mean().item()),
        "mean_edge_per_foreground": float(edge_per_foreground.mean().item()),
        "threshold": 0.52,
    }


def _same_label_pair_mse(plane: torch.Tensor, labels: torch.Tensor | None) -> dict[str, Any]:
    if labels is None or not torch.is_tensor(labels):
        return {"available": False, "mean": None}
    value = _normalize(plane)
    mses = []
    for label in labels.detach().cpu().unique():
        indexes = (labels.detach().cpu() == label).nonzero(as_tuple=True)[0]
        if int(indexes.numel()) >= 2:
            mses.append(float((value[indexes[0]] - value[indexes[1]]).square().mean().item()))
    mean = sum(mses) / len(mses) if mses else None
    return {"available": bool(mses), "mean": mean, "values": mses}


def _entries_from_report(report: dict[str, Any]) -> list[tuple[str, Path]]:
    entries = []
    for record in report.get("runs", []):
        run = record.get("run")
        if run:
            entries.append((str(record.get("label") or Path(str(run)).name), Path(run)))
    return entries


def _load_run(label: str, run: Path) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    labels = bundle.get("class_labels")
    plane_proxy = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    same_label = {key: _same_label_pair_mse(bundle[key], labels) for key in (TARGET_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)}
    physical_vs_decoder = _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE])
    if not plane_proxy[TARGET_PLANE]["readable"] or not plane_proxy[DECODER_INPUT_PLANE]["readable"]:
        origin = "target-code-blob-origin"
    elif not plane_proxy[DECODER_OUTPUT_PLANE]["readable"]:
        origin = "decoder-blob-origin"
    else:
        origin = "camera-texture-amplification"
    return {
        "label": label,
        "run": run.as_posix(),
        "localized_blob_origin": origin,
        "human_topology_proxy_v2": plane_proxy,
        "same_label_pair_mse": same_label,
        "physical_vs_decoder_corr_mse": physical_vs_decoder,
        "_physical": bundle[PHYSICAL_PLANE],
    }


def _cross_probe_similarity(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left = records[left_index]
            right = records[right_index]
            metrics = _corr_mse(left["_physical"], right["_physical"])
            pairs.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "corr": metrics["corr"],
                    "mse": metrics["mse"],
                }
            )
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


def _label_template_collapse(records: list[dict[str, Any]], cross_probe: dict[str, Any]) -> dict[str, Any]:
    same_label_values = [
        record["same_label_pair_mse"][PHYSICAL_PLANE]["mean"]
        for record in records
        if record["same_label_pair_mse"][PHYSICAL_PLANE]["mean"] is not None
    ]
    same_label_mean = sum(same_label_values) / len(same_label_values) if same_label_values else None
    same_label_collapsed = same_label_mean is not None and same_label_mean <= 1.0e-4
    collapsed = bool(same_label_collapsed and (cross_probe.get("collapsed") or len(records) == 1))
    return {
        "collapsed": collapsed,
        "same_label_mean_mse": same_label_mean,
        "same_label_collapsed": same_label_collapsed,
        "cross_probe_collapsed": bool(cross_probe.get("collapsed")),
    }


def _majority_origin(records: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        origin = record["localized_blob_origin"]
        counts[origin] = counts.get(origin, 0) + 1
    if not counts:
        return "harness-invalid"
    priority = ["target-code-blob-origin", "decoder-blob-origin", "camera-texture-amplification"]
    return max(priority, key=lambda key: (counts.get(key, 0), -priority.index(key)))


def run_v39_human_topology_false_positive(
    *,
    v38_report: Path,
    v37_report: Path,
    manual_verdict: str,
    output: Path | None = None,
    sample_runs: list[Path] | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    errors: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    source_v38 = _load_json(v38_report)
    source_v37 = _load_json(v37_report)
    if manual_verdict != VALID_MANUAL_VERDICT:
        errors.append({"error": f"invalid manual verdict: {manual_verdict!r}"})
    if source_v38 is None:
        errors.append({"error": f"missing v38 report: {v38_report}"})
    if source_v37 is None:
        errors.append({"error": f"missing v37.1 report: {v37_report}"})
    if not errors:
        if sample_runs:
            if labels is not None and len(labels) != len(sample_runs):
                errors.append({"error": "labels and sample_runs must have the same length"})
                entries = []
            else:
                entries = [
                    (labels[index] if labels is not None else run.name, run)
                    for index, run in enumerate(sample_runs)
                ]
        else:
            assert source_v38 is not None
            entries = _entries_from_report(source_v38)
        for label, run in entries:
            try:
                records.append(_load_run(label, run))
            except Exception as exc:
                errors.append({"label": label, "run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    cross_probe = _cross_probe_similarity(records)
    label_collapse = _label_template_collapse(records, cross_probe)
    proxy_false_positive = bool(
        source_v37 is not None
        and source_v37.get("status") == "manual-cmos-review-candidate"
        and manual_verdict == VALID_MANUAL_VERDICT
    )
    localized_origin = _majority_origin(records)
    classifications = []
    if manual_verdict == VALID_MANUAL_VERDICT:
        classifications.append("manual-review-failed-final-cmos-blob")
    if proxy_false_positive:
        classifications.append("topology-proxy-false-positive")
    if localized_origin != "harness-invalid":
        classifications.append(localized_origin)
    if label_collapse["collapsed"]:
        classifications.append("label-template-collapse")
    status = "harness-invalid" if errors or not records else "manual-review-failed-final-cmos-blob"
    result_records = []
    for record in records:
        cleaned = dict(record)
        cleaned.pop("_physical", None)
        result_records.append(cleaned)
    next_gate_by_origin = {
        "target-code-blob-origin": "v40_target_code_topology_redesign",
        "decoder-blob-origin": "v40_decoder_physical_reachability",
        "camera-texture-amplification": "v40_physical_cmos_readout",
        "harness-invalid": "repair_v39_inputs",
    }
    next_gate = next_gate_by_origin.get(localized_origin, "diagnostic_followup_before_human_acceptance")
    if label_collapse["collapsed"] and localized_origin != "harness-invalid":
        next_gate = "v40_sample_code_diversity_target_representation"
    result = {
        "version": "v39",
        "objective": "human_topology_false_positive_and_blob_origin_isolation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "human_review_passed": False,
        "manual_verdict": manual_verdict,
        "source_v38_report": v38_report.as_posix(),
        "source_v37_report": v37_report.as_posix(),
        "source_v38_status": None if source_v38 is None else source_v38.get("status"),
        "source_v37_status": None if source_v37 is None else source_v37.get("status"),
        "status": status,
        "localized_blob_origin": localized_origin,
        "proxy_false_positive": proxy_false_positive,
        "classifications": classifications,
        "diagnostics": {
            "cross_probe_similarity": cross_probe,
            "label_template_collapse": label_collapse,
        },
        "runs": result_records,
        "errors": errors,
        "next_gate": next_gate,
        "human_gate_note": "v38 human review failed final CMOS readability; no accepted summary can be written.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v38-report", type=Path, required=True)
    parser.add_argument("--v37-report", type=Path, required=True)
    parser.add_argument("--manual-verdict", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v39_human_topology_false_positive(
        v38_report=args.v38_report,
        v37_report=args.v37_report,
        manual_verdict=args.manual_verdict,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "localized_blob_origin": result["localized_blob_origin"],
                "proxy_false_positive": result["proxy_false_positive"],
                "next_gate": result["next_gate"],
                "accepted_summary_written": result["accepted_summary_written"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
