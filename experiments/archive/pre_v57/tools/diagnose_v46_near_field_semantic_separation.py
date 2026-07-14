#!/usr/bin/env python3
"""v46 near-field semantic separability and texture suppression diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output  # noqa: E402
from tools.diagnose_v34_code_coupling_recovery import _labels, _load_sample_bundle, _preview_paths  # noqa: E402
from tools.diagnose_v40_sample_code_diversity_target_representation import (  # noqa: E402
    DECODER_OUTPUT_PLANE,
    PHYSICAL_PLANE,
    TARGET_PLANE,
    _corr_mse,
    _human_topology_proxy_v2,
    _render_cmos_review,
    _same_label_diversity,
)


PLANES = (TARGET_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v46_near_field_semantic_separation_{timestamp}.json"


def _sample_metrics(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "artifacts" / "one_shot_sample_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def _class_correlation(plane: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().long().reshape(-1)
    if labels.numel() != plane.shape[0]:
        raise ValueError("class labels must match plane batch size for class correlation")
    flat = plane.detach().float().reshape(plane.shape[0], -1)
    flat = (flat - flat.mean(dim=1, keepdim=True)) / flat.std(dim=1, keepdim=True).clamp_min(1.0e-8)
    class_ids = sorted(int(value) for value in labels.unique(sorted=True).tolist())
    means = []
    for class_id in class_ids:
        means.append(flat[labels == class_id].mean(dim=0))
    if len(means) < 2:
        return {
            "class_ids": class_ids,
            "matrix": [[1.0]],
            "mean_offdiag_corr": 0.0,
            "max_offdiag_corr": 0.0,
            "top_confused_pairs": [],
            "semantic_separation_passed": True,
        }
    class_matrix = torch.stack(means)
    corr = (class_matrix @ class_matrix.t()) / float(class_matrix.shape[1])
    offdiag = corr[~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)]
    pairs = []
    for i, class_a in enumerate(class_ids):
        for j, class_b in enumerate(class_ids):
            if j <= i:
                continue
            pairs.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "corr": float(corr[i, j].item()),
                }
            )
    pairs.sort(key=lambda item: item["corr"], reverse=True)
    max_offdiag = float(offdiag.max().item())
    mean_offdiag = float(offdiag.mean().item())
    return {
        "class_ids": class_ids,
        "matrix": [[float(value) for value in row] for row in corr.cpu().tolist()],
        "mean_offdiag_corr": mean_offdiag,
        "max_offdiag_corr": max_offdiag,
        "top_confused_pairs": pairs[:8],
        "semantic_separation_passed": max_offdiag < 0.88 and mean_offdiag < 0.65,
    }


def _probe_status(
    *,
    topology: dict[str, dict[str, Any]],
    class_correlation: dict[str, Any],
    probe_kind: str,
) -> tuple[str, str]:
    if not bool(topology[TARGET_PLANE]["readable"]):
        return "target-code-lost", "target code is no longer readable"
    if not bool(topology[DECODER_OUTPUT_PLANE]["readable"]):
        return "texture-speckle-limited", "decoder output is still texture/blob limited"
    if not bool(topology[PHYSICAL_PLANE]["readable"]):
        return "camera-after-decoder-loss", "decoder output is readable but final CMOS is not"
    if not bool(class_correlation["semantic_separation_passed"]):
        return "semantic-collapse-still-high", "final CMOS class correlation remains too high"
    if probe_kind == "final_plane":
        return "final-plane-loss-helped", "final-plane objective recovered semantic separation"
    return "semantic-separation-recovered-needs-human-review", "near-field final CMOS semantics improved; human review required"


def analyze_sample_run(
    run: Path,
    *,
    label: str,
    probe_kind: str,
    aux_plane: str | None,
    spacing_mm: float | None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    class_correlation = _class_correlation(bundle[PHYSICAL_PLANE], labels)
    status, reason = _probe_status(
        topology=topology,
        class_correlation=class_correlation,
        probe_kind=probe_kind,
    )
    return {
        "label": label,
        "run": run.as_posix(),
        "probe_kind": probe_kind,
        "spacing_mm": spacing_mm,
        "aux_topology_plane_key": aux_plane,
        "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
        "target_code_spatial_scale": 0.72,
        "status": status,
        "classification": status,
        "reason": reason,
        "human_gate_status": "manual-review-required",
        "class_correlation": class_correlation,
        "same_label_diversity": diversity,
        "human_topology_proxy_v2": topology,
        "decoder_output_vs_target_corr_mse": _corr_mse(bundle[DECODER_OUTPUT_PLANE], bundle[TARGET_PLANE]),
        "physical_vs_decoder_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
        "sample_metrics": _sample_metrics(run),
        "preview_package": _preview_paths(run),
        "cmos_only_review_package": _render_cmos_review(
            label,
            run,
            bundle[PHYSICAL_PLANE],
            review_dir,
            class_labels=labels,
            review_version="v46",
            strict_class_coverage=False,
        ),
        "plane_shapes": {key: list(bundle[key].shape) for key in PLANES},
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(str(record.get("run")))
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
        next_gate = "repair_v46_inputs"
    elif by_status.get("semantic-separation-recovered-needs-human-review"):
        status = "semantic-separation-recovered-needs-human-review"
        next_gate = "v47_formal_manual_review_candidate"
    elif by_status.get("final-plane-loss-helped"):
        status = "final-plane-loss-helped"
        next_gate = "final_plane_loss_at_0p25mm_followup"
    elif by_status.get("semantic-collapse-still-high"):
        status = "semantic-collapse-still-high"
        next_gate = "near_field_semantic_or_pair_contrast_followup"
    elif by_status.get("texture-speckle-limited"):
        status = "texture-speckle-limited"
        next_gate = "near_field_speckle_regularization_or_aperture_followup"
    elif by_status.get("camera-after-decoder-loss"):
        status = "camera-after-decoder-loss"
        next_gate = "physical_cmos_readout_followup"
    else:
        status = "near-field-semantic-unresolved"
        next_gate = "aperture_pixel_size_or_non_propagation_decoder_upper_bound"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def run_v46_near_field_semantic_separation(
    *,
    source_v45_report: Path | None = None,
    source_v45_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v45_status = None
    if source_v45_report is not None:
        payload = _load_json(source_v45_report)
        assert payload is not None
        source_v45_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        label, probe_kind, aux_plane, spacing_mm, sample_run, train_run = _parse_probe_entry(entry)
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    probe_kind=probe_kind or "unknown",
                    aux_plane=aux_plane,
                    spacing_mm=spacing_mm,
                    review_dir=review_dir,
                )
            )
        except Exception as exc:
            errors.append({"probe": label, "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label,
                    "probe_kind": probe_kind or "unknown",
                    "run": sample_run or None,
                    "train_run": train_run or None,
                    "status": "harness-invalid",
                    "reason": "probe analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": "v46",
        "objective": "near_field_semantic_separation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v45_report": source_v45_report.as_posix() if source_v45_report else None,
        "source_v45_summary": source_v45_summary.as_posix() if source_v45_summary else None,
        "source_v45_status": source_v45_status,
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


def _parse_probe_entry(entry: str) -> tuple[str, str, str, float | None, str, str]:
    parts = entry.split("|")
    while len(parts) < 6:
        parts.append("")
    label, probe_kind, aux_plane, spacing_mm, sample_run, train_run = parts[:6]
    return (
        label,
        probe_kind,
        aux_plane,
        float(spacing_mm) if spacing_mm else None,
        sample_run,
        train_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v45-report", type=Path)
    parser.add_argument("--source-v45-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v46_near_field_semantic_separation(
        source_v45_report=args.source_v45_report,
        source_v45_summary=args.source_v45_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
