#!/usr/bin/env python3
"""v45 decoder spacing / near-field recovery diagnostic."""

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
    return Path("outputs/reports") / f"twophoton_100x100_v45_decoder_spacing_near_field_recovery_{timestamp}.json"


def _sample_metrics(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "artifacts" / "one_shot_sample_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def _probe_status(
    *,
    target_readable: bool,
    decoder_readable: bool,
    physical_readable: bool,
    probe_kind: str,
) -> tuple[str, str]:
    if not target_readable:
        return "target-scale-still-unreadable", "target code no longer remains readable"
    if not decoder_readable:
        return "spacing-still-texture-limited", "decoder output remains texture/blob limited"
    if not physical_readable:
        return "camera-after-decoder-loss", "decoder output is readable but final CMOS is not"
    if probe_kind == "physical_cmos_loss":
        return "physical-cmos-loss-helped", "final-plane loss recovered the short-spacing CMOS proxy"
    return "near-field-spacing-recovered-needs-human-review", "near-field spacing recovered final CMOS proxy; human review required"


def analyze_sample_run(
    run: Path,
    *,
    label: str,
    probe_kind: str,
    spacing_mm: float | None,
    aux_plane: str | None,
    target_scale: float | None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    status, reason = _probe_status(
        target_readable=bool(topology[TARGET_PLANE]["readable"]),
        decoder_readable=bool(topology[DECODER_OUTPUT_PLANE]["readable"]),
        physical_readable=bool(topology[PHYSICAL_PLANE]["readable"]),
        probe_kind=probe_kind,
    )
    return {
        "label": label,
        "run": run.as_posix(),
        "probe_kind": probe_kind,
        "spacing_mm": spacing_mm,
        "aux_topology_plane_key": aux_plane,
        "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
        "target_code_spatial_scale": target_scale,
        "status": status,
        "classification": status,
        "reason": reason,
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
            review_version="v45",
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
        next_gate = "repair_v45_inputs"
    elif by_status.get("near-field-spacing-recovered-needs-human-review"):
        status = "near-field-spacing-recovered-needs-human-review"
        next_gate = "v46_formal_manual_review_candidate"
    elif by_status.get("physical-cmos-loss-helped"):
        status = "physical-cmos-loss-helped"
        next_gate = "final_plane_loss_at_best_spacing"
    elif by_status.get("camera-after-decoder-loss"):
        status = "camera-after-decoder-loss"
        next_gate = "physical_cmos_readout_followup"
    elif by_status.get("target-scale-still-unreadable"):
        status = "target-scale-still-unreadable"
        next_gate = "target_code_spatial_scaling_followup"
    else:
        status = "spacing-still-texture-limited"
        next_gate = "aperture_pixel_size_or_non_propagation_decoder_upper_bound"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def run_v45_decoder_spacing_near_field_recovery(
    *,
    source_v44_report: Path | None = None,
    source_v44_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v44_status = None
    if source_v44_report is not None:
        payload = _load_json(source_v44_report)
        assert payload is not None
        source_v44_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        label, probe_kind, spacing_mm, aux_plane, target_scale, sample_run, train_run = _parse_probe_entry(entry)
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    probe_kind=probe_kind or "spacing",
                    spacing_mm=spacing_mm,
                    aux_plane=aux_plane,
                    target_scale=target_scale,
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
        "version": "v45",
        "objective": "decoder_spacing_near_field_recovery",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v44_report": source_v44_report.as_posix() if source_v44_report else None,
        "source_v44_summary": source_v44_summary.as_posix() if source_v44_summary else None,
        "source_v44_status": source_v44_status,
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


def _parse_probe_entry(entry: str) -> tuple[str, str, float | None, str, float | None, str, str]:
    parts = entry.split("|")
    while len(parts) < 7:
        parts.append("")
    label, probe_kind, spacing_mm, aux_plane, target_scale, sample_run, train_run = parts[:7]
    return (
        label,
        probe_kind,
        float(spacing_mm) if spacing_mm else None,
        aux_plane,
        float(target_scale) if target_scale else None,
        sample_run,
        train_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v44-report", type=Path)
    parser.add_argument("--source-v44-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v45_decoder_spacing_near_field_recovery(
        source_v44_report=args.source_v44_report,
        source_v44_summary=args.source_v44_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
