#!/usr/bin/env python3
"""v35 optical code-carrier isolation diagnostic.

This no-retrain diagnostic compares high-gain code-residual samples with a
direct diagnostic SLM perturbation oracle. It locates whether sample-time code
variation is limited by the generator command head, the front/OLS carrier, the
bridge, decoder, or true CMOS readout. It never promotes checkpoints or writes
accepted summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v34_code_coupling_recovery import (
    DECODER_PLANE,
    ENCODED_PLANE,
    OLS_PLANE,
    PHYSICAL_PLANE,
    TARGET_PLANE,
    _code_summary,
    _labels,
    _load_sample_bundle,
    _preview_paths,
    _tensor_summary,
    analyze_sample_run,
)

COMMAND_PLANE = "slm_command"
ORACLE_PATTERN = "diagnostic_slm_override_pattern"


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v35_optical_code_carrier_isolation_{timestamp}.json"


def _saturation_stats(bundle: dict[str, Any], key: str = COMMAND_PLANE) -> dict[str, Any]:
    tensor = bundle.get(key)
    if not torch.is_tensor(tensor):
        return {"available": False}
    value = tensor.detach().float().cpu()
    return {
        "available": True,
        "near_zero_fraction": float((value <= 1.0e-4).float().mean().item()),
        "near_one_fraction": float((value >= 1.0 - 1.0e-4).float().mean().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def _oracle_status(chain: dict[str, Any]) -> tuple[str, str]:
    if not chain["codes"]["active"]:
        return "harness-invalid", "codes are inactive"
    if not chain[ORACLE_PATTERN]["active"] or not chain[COMMAND_PLANE]["active"]:
        return "harness-invalid", "diagnostic SLM perturbation did not reach SLM command"
    if not chain[ENCODED_PLANE]["active"] or not chain[OLS_PLANE]["active"]:
        return "front-ols-code-carrier-failure", "direct SLM variation does not survive encoder/OLS"
    if not chain[TARGET_PLANE]["active"]:
        return "ols-to-bridge-carrier-failure", "OLS carrier variation does not reach target-code bridge"
    if not chain[DECODER_PLANE]["active"]:
        return "decoder-after-code-bottleneck", "target/code variation does not survive decoder"
    if not chain[PHYSICAL_PLANE]["active"]:
        return "camera-after-code-bottleneck", "decoder variation does not survive true CMOS plane"
    return "code-coupling-recovered-needs-human-review", "direct SLM code variation reaches true CMOS preview"


def analyze_oracle_run(run: Path, *, amplitude: float, label: str | None = None) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    chain: dict[str, Any] = {
        "codes": _code_summary(bundle, labels),
        ORACLE_PATTERN: _tensor_summary(bundle, labels, ORACLE_PATTERN),
    }
    missing = []
    for key in (
        COMMAND_PLANE,
        ENCODED_PLANE,
        OLS_PLANE,
        TARGET_PLANE,
        DECODER_PLANE,
        PHYSICAL_PLANE,
    ):
        record = _tensor_summary(bundle, labels, key)
        chain[key] = record
        if not record["available"]:
            missing.append(key)
    if missing:
        status = "harness-invalid"
        reason = f"missing oracle chain tensors: {', '.join(missing)}"
    else:
        status, reason = _oracle_status(chain)
    return {
        "kind": "direct_slm_perturbation_oracle",
        "label": label or run.name,
        "amplitude": float(amplitude),
        "run": run.as_posix(),
        "status": status,
        "classification": status,
        "reason": reason,
        "chain": chain,
        "slm_saturation": _saturation_stats(bundle),
        "missing_planes": missing,
        "preview_package": _preview_paths(run),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def analyze_scale_run(run: Path, *, scale: float, label: str | None = None) -> dict[str, Any]:
    record = analyze_sample_run(run, scale=scale, label=label)
    bundle = _load_sample_bundle(run)
    mapped = {
        "residual-to-command-bottleneck": "generator-command-too-weak",
        "command-to-ols-bottleneck": "front-ols-code-carrier-failure",
        "ols-to-target-code-bottleneck": "ols-to-bridge-carrier-failure",
    }.get(record["status"], record["status"])
    record.update(
        {
            "kind": "high_gain_code_residual_sample",
            "status": mapped,
            "classification": mapped,
            "slm_saturation": _saturation_stats(bundle),
        }
    )
    return record


def _decision(scale_records: list[dict[str, Any]], oracle_records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    records = scale_records + oracle_records
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    scale_recovered = any(record["status"] == "code-coupling-recovered-needs-human-review" for record in scale_records)
    oracle_recovered = any(record["status"] == "code-coupling-recovered-needs-human-review" for record in oracle_records)
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
    elif scale_recovered:
        status = "code-coupling-recovered-needs-human-review"
    elif by_status.get("camera-after-code-bottleneck"):
        status = "camera-after-code-bottleneck"
    elif by_status.get("decoder-after-code-bottleneck"):
        status = "decoder-after-code-bottleneck"
    elif by_status.get("ols-to-bridge-carrier-failure"):
        status = "ols-to-bridge-carrier-failure"
    elif oracle_recovered:
        status = "generator-command-too-weak"
    elif any(record["status"] == "front-ols-code-carrier-failure" for record in oracle_records):
        status = "front-ols-code-carrier-failure"
    elif by_status.get("front-ols-code-carrier-failure"):
        status = "front-ols-code-carrier-failure"
    else:
        status = "generator-command-too-weak"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": (
            "human_v4_cmos_preview_review"
            if status == "code-coupling-recovered-needs-human-review"
            else "diagnostic_followup_before_human_acceptance"
        ),
    }


def run_v35_optical_code_carrier_isolation(
    *,
    summary: Path | None = None,
    scale_runs: list[Path],
    scales: list[float],
    scale_labels: list[str] | None = None,
    oracle_runs: list[Path],
    oracle_amplitudes: list[float],
    oracle_labels: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    if len(scale_runs) != len(scales):
        raise ValueError("scale_runs and scales must have the same length")
    if len(oracle_runs) != len(oracle_amplitudes):
        raise ValueError("oracle_runs and oracle_amplitudes must have the same length")
    if scale_labels is not None and len(scale_labels) != len(scale_runs):
        raise ValueError("scale_labels must have the same length as scale_runs")
    if oracle_labels is not None and len(oracle_labels) != len(oracle_runs):
        raise ValueError("oracle_labels must have the same length as oracle_runs")
    output_path = output or _default_output()
    _require_report_output(output_path)
    source_summary = None
    source_status = None
    if summary is not None:
        payload = _load_json(summary)
        assert payload is not None
        source_summary = summary.as_posix()
        source_status = payload.get("status")
    errors: list[dict[str, str]] = []
    scale_records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    for index, (run, scale) in enumerate(zip(scale_runs, scales, strict=True)):
        label = scale_labels[index] if scale_labels is not None else None
        try:
            scale_records.append(analyze_scale_run(run, scale=scale, label=label))
        except Exception as exc:
            errors.append({"run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    for index, (run, amplitude) in enumerate(zip(oracle_runs, oracle_amplitudes, strict=True)):
        label = oracle_labels[index] if oracle_labels is not None else None
        try:
            oracle_records.append(analyze_oracle_run(run, amplitude=amplitude, label=label))
        except Exception as exc:
            errors.append({"run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    decision = _decision(scale_records, oracle_records, errors)
    result = {
        "version": "v35",
        "objective": "optical_code_carrier_isolation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": source_summary,
        "source_status": source_status,
        "scale_runs": scale_records,
        "oracle_runs": oracle_records,
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
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--scale-run", action="append", type=Path, default=[])
    parser.add_argument("--scale", action="append", type=float, default=[])
    parser.add_argument("--scale-label", action="append")
    parser.add_argument("--oracle-run", action="append", type=Path, default=[])
    parser.add_argument("--oracle-amplitude", action="append", type=float, default=[])
    parser.add_argument("--oracle-label", action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v35_optical_code_carrier_isolation(
        summary=args.summary,
        scale_runs=args.scale_run,
        scales=args.scale,
        scale_labels=args.scale_label,
        oracle_runs=args.oracle_run,
        oracle_amplitudes=args.oracle_amplitude,
        oracle_labels=args.oracle_label,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
