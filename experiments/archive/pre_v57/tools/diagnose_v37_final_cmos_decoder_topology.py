#!/usr/bin/env python3
"""v37 final-CMOS manual review and decoder-topology isolation.

This offline diagnostic reuses v36 sample tensor bundles to determine whether
the current failure is already present in target-code topology, introduced by
the decoder, or introduced after decoder output on the true CMOS plane. It
also writes a human review package that prioritizes final CMOS intensity.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v34_code_coupling_recovery import _load_sample_bundle, _preview_paths


PLANES = (
    "target_code_amplitude",
    "decoder_input_amplitude",
    "decoder_output_amplitude",
    "physical_camera_plane",
)
FINAL_PLANE = "physical_camera_plane"
TARGET_PLANE = "target_code_amplitude"
DECODER_INPUT_PLANE = "decoder_input_amplitude"
DECODER_OUTPUT_PLANE = "decoder_output_amplitude"

TOPOLOGY_SCORE_THRESHOLD = 0.34


def _default_output(*, version_slug: str = "v37") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_{version_slug}_final_cmos_decoder_topology_{timestamp}.json"


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


def _binary_edges(binary: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(binary.float(), (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    diffs = (
        (center - padded[..., :-2, 1:-1]).abs()
        + (center - padded[..., 2:, 1:-1]).abs()
        + (center - padded[..., 1:-1, :-2]).abs()
        + (center - padded[..., 1:-1, 2:]).abs()
    )
    return (diffs > 0).float()


def _plane_topology_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = _normalize(tensor)
    threshold = torch.maximum(
        value.new_full((value.shape[0], 1, 1, 1), 0.35),
        value.mean(dim=(-2, -1), keepdim=True) + 0.25 * value.std(dim=(-2, -1), keepdim=True, unbiased=False),
    )
    fg = value >= threshold
    fg_fraction = fg.float().mean(dim=(-2, -1))
    edges = _binary_edges(fg)
    edge_density = edges.flatten(1).mean(dim=1)
    edge_per_foreground = edge_density / fg_fraction.flatten(1).clamp_min(1.0e-6).squeeze(1)
    contrast = value.flatten(1).std(dim=1, unbiased=False)
    not_full_blob = ((fg_fraction.flatten(1).squeeze(1) >= 0.015) & (fg_fraction.flatten(1).squeeze(1) <= 0.55)).float()
    thin_structure = torch.clamp(edge_per_foreground / 1.25, 0.0, 1.0)
    contrast_score = torch.clamp(contrast / 0.18, 0.0, 1.0)
    score = 0.45 * thin_structure + 0.35 * contrast_score + 0.20 * not_full_blob
    return {
        "available": True,
        "shape": list(tensor.shape),
        "mean_foreground_fraction": float(fg_fraction.mean().item()),
        "mean_edge_density": float(edge_density.mean().item()),
        "mean_edge_per_foreground": float(edge_per_foreground.mean().item()),
        "mean_contrast": float(contrast.mean().item()),
        "topology_score": float(score.mean().item()),
        "topology_ok": bool(score.mean().item() >= TOPOLOGY_SCORE_THRESHOLD),
        "threshold": TOPOLOGY_SCORE_THRESHOLD,
    }


def _load_required_planes(run: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    bundle = _load_sample_bundle(run)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing topology planes: {', '.join(missing)}")
    planes = {key: bundle[key] for key in PLANES}
    first_batch = int(_normalize(planes[PLANES[0]]).shape[0])
    for key, tensor in planes.items():
        batch = int(_normalize(tensor).shape[0])
        if batch != first_batch:
            raise ValueError(
                "plane batch size is inconsistent: "
                f"{key} has batch {batch}, expected {first_batch}; shape={tuple(tensor.shape)}"
            )
    return bundle, planes


def _render_review_package(run: Path, label: str, planes: dict[str, torch.Tensor], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_final_cmos_decoder_topology_review.png"
    max_items = min(10, int(planes[FINAL_PLANE].shape[0]))
    final = _normalize(planes[FINAL_PLANE])[:max_items, 0]
    target = _normalize(planes[TARGET_PLANE])[:max_items, 0]
    dec_in = _normalize(planes[DECODER_INPUT_PLANE])[:max_items, 0]
    dec_out = _normalize(planes[DECODER_OUTPUT_PLANE])[:max_items, 0]

    fig, axes = plt.subplots(max_items, 4, figsize=(8, max(2, max_items * 1.5)))
    if max_items == 1:
        axes = axes.reshape(1, -1)
    columns = [
        ("CMOS intensity", final),
        ("Target code amp", target),
        ("Decoder input amp", dec_in),
        ("Decoder output amp", dec_out),
    ]
    for col, (title, values) in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)
        for row in range(max_items):
            axes[row, col].imshow(values[row], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"v37 manual review package: {label} | final CMOS is acceptance plane", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "manual_review_image": path.as_posix(),
        "final_human_gate_plane": FINAL_PLANE,
        "final_human_gate_label": "CMOS intensity",
    }


def analyze_sample_run(run: Path, *, label: str, output_dir: Path | None = None) -> dict[str, Any]:
    bundle, planes = _load_required_planes(run)
    plane_summaries = {key: _plane_topology_summary(value) for key, value in planes.items()}
    preview = _preview_paths(run)
    if not preview["v4_style_preview_exists"]:
        return {
            "label": label,
            "run": run.as_posix(),
            "status": "harness-invalid",
            "reason": "missing v4-style preview image",
            "plane_topology": plane_summaries,
            "preview_package": preview,
            "accepted_summary_written": False,
            "human_readability_required": True,
        }
    target_ok = plane_summaries[TARGET_PLANE]["topology_ok"]
    dec_in_ok = plane_summaries[DECODER_INPUT_PLANE]["topology_ok"]
    dec_out_ok = plane_summaries[DECODER_OUTPUT_PLANE]["topology_ok"]
    final_ok = plane_summaries[FINAL_PLANE]["topology_ok"]
    if final_ok:
        status = "manual-cmos-review-candidate"
        reason = "final CMOS topology score is above diagnostic review threshold"
    elif not target_ok or not dec_in_ok:
        status = "target-code-topology-too-blob-like"
        reason = "target-code or decoder-input topology is already too blob-like"
    elif not dec_out_ok:
        status = "decoder-physical-topology-loss"
        reason = "target-code survives to decoder input but decoder output loses topology"
    else:
        status = "camera-after-decoder-loss"
        reason = "decoder output is topology-positive but true final CMOS is not"
    review_package = {}
    if output_dir is not None:
        review_package = _render_review_package(run, label, planes, output_dir)
    exact_metrics_invalid = not isinstance(bundle.get("reference_source"), str) or bundle.get("reference_source") != "paired"
    return {
        "label": label,
        "run": run.as_posix(),
        "status": status,
        "classification": status,
        "reason": reason,
        "plane_topology": plane_summaries,
        "preview_package": preview,
        "manual_review_package": review_package,
        "sample_reference_metric_status": "sample-reference-metric-invalid" if exact_metrics_invalid else "paired-reference",
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
    elif by_status.get("manual-cmos-review-candidate"):
        status = "manual-cmos-review-candidate"
    elif by_status.get("camera-after-decoder-loss"):
        status = "camera-after-decoder-loss"
    elif by_status.get("decoder-physical-topology-loss"):
        status = "decoder-physical-topology-loss"
    else:
        status = "target-code-topology-too-blob-like"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": (
            "formal_manual_cmos_review_without_auto_acceptance"
            if status == "manual-cmos-review-candidate"
            else "diagnostic_followup_before_human_acceptance"
        ),
    }


def _sample_runs_from_v36_report(report: Path) -> list[tuple[str, Path]]:
    payload = _load_json(report)
    assert payload is not None
    entries = []
    for probe in payload.get("probes", []):
        sample_run = probe.get("sample_run")
        label = str(probe.get("label") or Path(str(sample_run)).name)
        if sample_run:
            entries.append((label, Path(sample_run)))
    return entries


def run_v37_final_cmos_decoder_topology(
    *,
    v36_report: Path,
    sample_runs: list[Path] | None = None,
    labels: list[str] | None = None,
    output: Path | None = None,
    review_dir: Path | None = None,
    version: str = "v37",
    objective: str = "final_cmos_manual_review_and_decoder_topology_isolation",
) -> dict[str, Any]:
    version_slug = version.replace(".", "_")
    output_path = output or _default_output(version_slug=version_slug)
    _require_report_output(output_path)
    source = _load_json(v36_report)
    assert source is not None
    if sample_runs is None or not sample_runs:
        entries = _sample_runs_from_v36_report(v36_report)
    else:
        if labels is not None and len(labels) != len(sample_runs):
            raise ValueError("labels and sample_runs must have the same length")
        entries = [
            (labels[index] if labels is not None else run.name, run)
            for index, run in enumerate(sample_runs)
        ]
    records = []
    errors = []
    review_output_dir = review_dir or (output_path.parent / output_path.stem)
    for label, run in entries:
        try:
            records.append(analyze_sample_run(run, label=label, output_dir=review_output_dir))
        except Exception as exc:
            errors.append({"label": label, "run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label,
                    "run": run.as_posix(),
                    "status": "harness-invalid",
                    "reason": "sample topology analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": version,
        "objective": objective,
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v36_report": v36_report.as_posix(),
        "source_v36_status": source.get("status"),
        "review_dir": review_output_dir.as_posix(),
        "runs": records,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane v4-style previews can satisfy human review.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-report", type=Path, required=True)
    parser.add_argument("--sample-run", action="append", type=Path)
    parser.add_argument("--label", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--version", default="v37")
    parser.add_argument("--objective", default="final_cmos_manual_review_and_decoder_topology_isolation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v37_final_cmos_decoder_topology(
        v36_report=args.v36_report,
        sample_runs=args.sample_run,
        labels=args.label,
        output=args.output,
        review_dir=args.review_dir,
        version=args.version,
        objective=args.objective,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
