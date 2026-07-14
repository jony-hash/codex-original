#!/usr/bin/env python3
"""v41 decoder physical topology recovery diagnostic."""

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
    return Path("outputs/reports") / f"twophoton_100x100_v41_decoder_physical_topology_recovery_{timestamp}.json"


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
    decoder_active: bool,
    physical_active: bool,
    decoder_readable: bool,
    physical_readable: bool,
    probe_kind: str,
) -> tuple[str, str]:
    if not decoder_active or not decoder_readable:
        return "decoder-physical-unreachable", "decoder output remains collapsed or blob-like"
    if not physical_active or not physical_readable:
        return "decoder-output-recovered-cmos-failed", "decoder output improved but final CMOS did not"
    if probe_kind == "phase_amplitude":
        return "phase-amplitude-upper-bound-positive", "phase-amplitude decoder upper bound reaches final CMOS proxy"
    if probe_kind == "four_layer":
        return "deeper-decoder-upper-bound-positive", "4-layer decoder upper bound reaches final CMOS proxy"
    return "decoder-physical-recovered-needs-human-review", "decoder training reaches final CMOS proxy; human review required"


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
    diversity = {key: _same_label_diversity(bundle[key], labels) for key in PLANES}
    topology = {key: _human_topology_proxy_v2(bundle[key]) for key in PLANES}
    status, reason = _probe_status(
        decoder_active=bool(diversity[DECODER_OUTPUT_PLANE]["active"]),
        physical_active=bool(diversity[PHYSICAL_PLANE]["active"]),
        decoder_readable=bool(topology[DECODER_OUTPUT_PLANE]["readable"]),
        physical_readable=bool(topology[PHYSICAL_PLANE]["readable"]),
        probe_kind=probe_kind,
    )
    return {
        "label": label,
        "run": run.as_posix(),
        "probe_kind": probe_kind,
        "status": status,
        "classification": status,
        "reason": reason,
        "same_label_diversity": diversity,
        "human_topology_proxy_v2": topology,
        "physical_vs_decoder_corr_mse": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
        "sample_metrics": _sample_metrics(run),
        "preview_package": _preview_paths(run),
        "cmos_only_review_package": _render_cmos_review(
            label,
            run,
            bundle[PHYSICAL_PLANE],
            review_dir,
            class_labels=labels,
            review_version="v41",
            strict_class_coverage=False,
        ),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
        next_gate = "repair_v41_inputs"
    elif by_status.get("decoder-physical-recovered-needs-human-review"):
        status = "decoder-physical-recovered-needs-human-review"
        next_gate = "formal_cmos_human_review_without_auto_acceptance"
    elif by_status.get("phase-amplitude-upper-bound-positive"):
        status = "phase-amplitude-upper-bound-positive"
        next_gate = "fabrication_constraint_or_phase_amplitude_decoder_followup"
    elif by_status.get("deeper-decoder-upper-bound-positive"):
        status = "deeper-decoder-upper-bound-positive"
        next_gate = "decoder_layer_count_physical_reachability_followup"
    elif by_status.get("decoder-output-recovered-cmos-failed"):
        status = "decoder-output-recovered-cmos-failed"
        next_gate = "physical_cmos_readout_followup"
    else:
        status = "decoder-physical-unreachable"
        next_gate = "decoder_architecture_or_target_topology_redesign"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def run_v41_decoder_physical_topology_recovery(
    *,
    source_v40_report: Path | None = None,
    source_v40_summary: Path | None = None,
    probe_entries: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v40_status = None
    if source_v40_report is not None:
        payload = _load_json(source_v40_report)
        assert payload is not None
        source_v40_status = payload.get("status")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        parts = entry.split("|")
        while len(parts) < 4:
            parts.append("")
        label, probe_kind, sample_run, train_run = parts[:4]
        try:
            if not sample_run:
                raise ValueError("missing sample run")
            records.append(
                analyze_sample_run(
                    Path(sample_run),
                    label=label,
                    probe_kind=probe_kind or "phase_only",
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
        "version": "v41",
        "objective": "decoder_physical_topology_recovery",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v40_report": source_v40_report.as_posix() if source_v40_report else None,
        "source_v40_summary": source_v40_summary.as_posix() if source_v40_summary else None,
        "source_v40_status": source_v40_status,
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
    parser.add_argument("--source-v40-report", type=Path)
    parser.add_argument("--source-v40-summary", type=Path)
    parser.add_argument("--probe-entry", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v41_decoder_physical_topology_recovery(
        source_v40_report=args.source_v40_report,
        source_v40_summary=args.source_v40_summary,
        probe_entries=args.probe_entry,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
