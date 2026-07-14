#!/usr/bin/env python3
"""v43 decoder propagation geometry and target-scale isolation diagnostic."""

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
    DECODER_INPUT_PLANE,
    DECODER_OUTPUT_PLANE,
    PHYSICAL_PLANE,
    TARGET_PLANE,
    _corr_mse,
    _human_topology_proxy_v2,
    _render_cmos_review,
    _same_label_diversity,
)


PLANES = (TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)
CROP_KEYS = (
    "diffractive_decoder_crop_crop_retention",
    "diffractive_decoder_crop_propagation_energy_ratio",
)


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v43_decoder_propagation_geometry_isolation_{timestamp}.json"


def analyze_sample_run(
    run: Path,
    *,
    label: str,
    probe_kind: str,
    spacing_mm: float | None,
    propagation_grid_size: int | None,
    target_mode: str,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    status = _probe_status(topology, diversity)
    return {
        "label": label,
        "run": run.as_posix(),
        "probe_kind": probe_kind,
        "spacing_mm": spacing_mm,
        "propagation_grid_size": propagation_grid_size,
        "target_code_diagnostic_mode": target_mode,
        "status": status,
        "classification": status,
        "same_label_diversity": diversity,
        "human_topology_proxy_v2": topology,
        "decoder_output_vs_input_corr_mse": _corr_mse(bundle[DECODER_OUTPUT_PLANE], bundle[DECODER_INPUT_PLANE]),
        "physical_vs_decoder_output_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
        "physical_vs_decoder_input_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_INPUT_PLANE]),
        "crop_diagnostics": _crop_diagnostics(bundle),
        "preview_package": _preview_paths(run),
        "cmos_only_review_package": _render_cmos_review(
            label,
            run,
            bundle[PHYSICAL_PLANE],
            review_dir,
            class_labels=labels,
            review_version="v43",
            strict_class_coverage=False,
        ),
        "plane_shapes": {key: list(bundle[key].shape) for key in PLANES},
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _probe_status(topology: dict[str, dict[str, Any]], diversity: dict[str, dict[str, Any]]) -> str:
    if not bool(topology[DECODER_INPUT_PLANE]["readable"]):
        return "target-scale-bottleneck"
    if bool(topology[DECODER_OUTPUT_PLANE]["readable"]) and bool(topology[PHYSICAL_PLANE]["readable"]):
        return "propagation-geometry-recovered-needs-human-review"
    return "propagation-geometry-failed"


def _crop_diagnostics(bundle: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in CROP_KEYS:
        value = bundle.get(key)
        if torch.is_tensor(value):
            result[key] = {
                "available": True,
                "mean": float(value.detach().float().mean().item()),
                "min": float(value.detach().float().min().item()),
            }
        else:
            result[key] = {"available": False}
    return result


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(str(record.get("run")))
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
        next_gate = "repair_v43_inputs"
    elif _any_recovered(records, probe_kind="target_scale"):
        status = "target-scale-bottleneck"
        next_gate = "v44_target_code_spatial_scaling_redesign"
    elif _any_recovered(records, probe_kind="aperture"):
        status = "aperture-crop-bottleneck"
        next_gate = "v44_aperture_padding_crop_redesign"
    elif _any_recovered(records, probe_kind="spacing"):
        status = "spacing-propagation-bottleneck"
        next_gate = "v44_decoder_spacing_or_relay_decoder_redesign"
    elif by_status.get("propagation-geometry-recovered-needs-human-review"):
        status = "propagation-geometry-recovered-needs-human-review"
        next_gate = "formal_cmos_human_review_without_auto_acceptance"
    else:
        status = "propagation-geometry-unresolved"
        next_gate = "v44_non_propagation_decoder_or_larger_aperture_upper_bound"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def _any_recovered(records: list[dict[str, Any]], *, probe_kind: str) -> bool:
    return any(
        record.get("probe_kind") == probe_kind
        and record.get("status") == "propagation-geometry-recovered-needs-human-review"
        for record in records
    )


def run_v43_decoder_propagation_geometry_isolation(
    *,
    source_v42_report: Path | None = None,
    source_v42_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v42_status = None
    if source_v42_report is not None:
        payload = _load_json(source_v42_report)
        assert payload is not None
        source_v42_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        label, probe_kind, spacing_mm, grid_size, target_mode, sample_run = _parse_probe_entry(entry)
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    probe_kind=probe_kind,
                    spacing_mm=spacing_mm,
                    propagation_grid_size=grid_size,
                    target_mode=target_mode,
                    review_dir=review_dir,
                )
            )
        except Exception as exc:
            errors.append({"probe": label, "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label,
                    "probe_kind": probe_kind,
                    "run": sample_run or None,
                    "status": "harness-invalid",
                    "reason": "probe analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": "v43",
        "objective": "decoder_propagation_geometry_isolation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v42_report": source_v42_report.as_posix() if source_v42_report else None,
        "source_v42_summary": source_v42_summary.as_posix() if source_v42_summary else None,
        "source_v42_status": source_v42_status,
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


def _parse_probe_entry(entry: str) -> tuple[str, str, float | None, int | None, str, str]:
    parts = entry.split("|")
    while len(parts) < 6:
        parts.append("")
    label, kind, spacing, grid, target_mode, sample_run = parts[:6]
    spacing_mm = float(spacing) if spacing else None
    grid_size = int(grid) if grid else None
    return label, kind or "unknown", spacing_mm, grid_size, target_mode or "unknown", sample_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v42-report", type=Path)
    parser.add_argument("--source-v42-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v43_decoder_propagation_geometry_isolation(
        source_v42_report=args.source_v42_report,
        source_v42_summary=args.source_v42_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
