#!/usr/bin/env python3
"""v42 decoder-bypass and back-path isolation diagnostic."""

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


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v42_decoder_bypass_backpath_isolation_{timestamp}.json"


def _sample_metrics(run: Path) -> dict[str, Any] | None:
    path = run / "artifacts" / "one_shot_sample_metrics.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    assert payload is not None
    return payload


def analyze_sample_run(
    run: Path,
    *,
    label: str,
    probe_kind: str,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    missing = [key for key in PLANES if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError(f"sample bundle missing required planes: {', '.join(missing)}")
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    return {
        "label": label,
        "run": run.as_posix(),
        "probe_kind": probe_kind,
        "status": _probe_status(probe_kind, topology, diversity),
        "classification": _probe_status(probe_kind, topology, diversity),
        "same_label_diversity": diversity,
        "human_topology_proxy_v2": topology,
        "physical_vs_decoder_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
        "physical_vs_decoder_input_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_INPUT_PLANE]),
        "decoder_output_vs_input_corr_mse": _corr_mse(bundle[DECODER_OUTPUT_PLANE], bundle[DECODER_INPUT_PLANE]),
        "sample_metrics": _sample_metrics(run),
        "preview_package": _preview_paths(run),
        "cmos_only_review_package": _render_cmos_review(
            label,
            run,
            bundle[PHYSICAL_PLANE],
            review_dir,
            class_labels=labels,
            review_version="v42",
            strict_class_coverage=False,
        ),
        "plane_shapes": {key: list(bundle[key].shape) for key in PLANES},
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _probe_status(
    probe_kind: str,
    topology: dict[str, dict[str, Any]],
    diversity: dict[str, dict[str, Any]],
) -> str:
    target_readable = bool(topology[TARGET_PLANE]["readable"])
    input_readable = bool(topology[DECODER_INPUT_PLANE]["readable"])
    output_readable = bool(topology[DECODER_OUTPUT_PLANE]["readable"])
    physical_readable = bool(topology[PHYSICAL_PLANE]["readable"])
    physical_active = bool(diversity[PHYSICAL_PLANE]["active"])
    if not target_readable or not input_readable:
        return "target-code-scale-mismatch"
    if probe_kind == "propagation_only":
        if output_readable and physical_readable and physical_active:
            return "propagation-only-recovered-needs-human-review"
        return "decoder-propagation-bottleneck"
    if output_readable and physical_readable and physical_active:
        return "decoder-bypass-recovered-needs-human-review"
    if output_readable and not physical_readable:
        return "backpath-camera-blur-bottleneck"
    return "target-code-scale-mismatch"


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(str(record.get("run")))
        by_kind[str(record.get("probe_kind"))] = record
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
        next_gate = "repair_v42_inputs"
    elif _is_recovered(by_kind.get("bypass_current")) and by_kind.get("propagation_only") and not _is_recovered(
        by_kind.get("propagation_only")
    ):
        status = "decoder-propagation-bottleneck"
        next_gate = "v43_decoder_spacing_aperture_isolation"
    elif _is_recovered(by_kind.get("bypass_current")):
        status = "decoder-mask-bottleneck"
        next_gate = "v43_decoder_architecture_redesign"
    elif _is_recovered(by_kind.get("bypass_clean")) or _is_recovered(by_kind.get("bypass_identity")):
        status = "backpath-camera-blur-bottleneck"
        next_gate = "v43_cmos_readout_calibration"
    elif by_kind.get("bypass_identity") and not _is_recovered(by_kind.get("bypass_identity")):
        status = "target-code-scale-mismatch"
        next_gate = "v43_target_representation_spatial_scaling"
    else:
        status = "decoder-bypass-backpath-unresolved"
        next_gate = "manual_review_then_decoder_or_target_followup"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def _is_recovered(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("status") in {
        "decoder-bypass-recovered-needs-human-review",
        "propagation-only-recovered-needs-human-review",
    }


def run_v42_decoder_bypass_backpath_isolation(
    *,
    source_v41_report: Path | None = None,
    source_v41_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v41_status = None
    if source_v41_report is not None:
        payload = _load_json(source_v41_report)
        assert payload is not None
        source_v41_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        label, probe_kind, sample_run = _parse_probe_entry(entry)
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    probe_kind=probe_kind,
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
        "version": "v42",
        "objective": "decoder_bypass_backpath_isolation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v41_report": source_v41_report.as_posix() if source_v41_report else None,
        "source_v41_summary": source_v41_summary.as_posix() if source_v41_summary else None,
        "source_v41_status": source_v41_status,
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


def _parse_probe_entry(entry: str) -> tuple[str, str, str]:
    parts = entry.split("|")
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1] or "unknown", parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v41-report", type=Path)
    parser.add_argument("--source-v41-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v42_decoder_bypass_backpath_isolation(
        source_v41_report=args.source_v41_report,
        source_v41_summary=args.source_v41_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
