#!/usr/bin/env python3
"""v31.1 bridge topology / same-batch target-code hard-gate diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output, _resize_like
from tools.diagnose_v28_decoder_controllability import (
    _dedupe,
    _load_bundle,
    _load_oracle_baseline,
    _plane_record,
    analyze_run,
)


PLANE_ORDER = (
    "encoded_amplitude",
    "ols_output_amplitude",
    "target_code_amplitude",
    "decoder_input_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
BRIDGE_PLANES = {"target_code_amplitude", "decoder_input_amplitude"}
DECODER_PLANES = {"decoder_output_amplitude", "chip_intensity_primary"}
PHYSICAL_PLANE = "physical_camera_plane"


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v31_bridge_topology_{timestamp}.json"


def _extract_runs(payload: Any, train_runs: list[Path], eval_runs: list[Path], sample_runs: list[Path]) -> None:
    if isinstance(payload, dict):
        for key in ("train_run",):
            if payload.get(key):
                train_runs.append(Path(str(payload[key])))
        for key in ("same_batch_eval_run", "test_eval_run", "eval_run"):
            if payload.get(key):
                eval_runs.append(Path(str(payload[key])))
        if payload.get("sample_run"):
            sample_runs.append(Path(str(payload["sample_run"])))
        for value in payload.values():
            _extract_runs(value, train_runs, eval_runs, sample_runs)
    elif isinstance(payload, list):
        for item in payload:
            _extract_runs(item, train_runs, eval_runs, sample_runs)


def _runs_from_summary(summary: Path) -> tuple[list[Path], list[Path], list[Path]]:
    payload = _load_json(summary)
    assert payload is not None
    train_runs: list[Path] = []
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    _extract_runs(payload, train_runs, eval_runs, sample_runs)
    return _dedupe(train_runs), _dedupe(eval_runs), _dedupe(sample_runs)


def _preview_path(run: Path, kind: str) -> str:
    filename = (
        "one_shot_distill_eval_100x100_cell_chain_preview.png"
        if kind == "eval"
        else "one_shot_sample_100x100_cell_chain_preview.png"
    )
    return (run / "images" / filename).as_posix()


def _metadata_for_run(run: Path, kind: str, bundle: dict[str, Any]) -> dict[str, Any]:
    suffix = "one_shot_distill_eval_artifact_metadata.json" if kind == "eval" else "one_shot_sample_artifact_metadata.json"
    metadata = _load_json(run / "artifacts" / suffix, required=False)
    result = metadata if isinstance(metadata, dict) else {}
    bundle_metadata = bundle.get("artifact_metadata")
    if isinstance(bundle_metadata, dict):
        result = {**bundle_metadata, **result}
    return result


def _load_bridge_trainability_audit(train_run: Path) -> dict[str, Any]:
    artifact = train_run / "artifacts" / "one_shot_target_code_bridge_trainability_audit.json"
    if artifact.is_file():
        payload = _load_json(artifact)
        assert isinstance(payload, dict)
        payload["source"] = artifact.as_posix()
        return payload
    meta = _load_json(train_run / "run_meta.json", required=False)
    if isinstance(meta, dict) and isinstance(meta.get("one_shot_target_code_bridge_trainability_audit"), dict):
        payload = dict(meta["one_shot_target_code_bridge_trainability_audit"])
        payload["source"] = (train_run / "run_meta.json").as_posix()
        return payload
    return {
        "enabled": False,
        "passed": None,
        "failure_reasons": ["missing-bridge-trainability-audit"],
        "source": None,
    }


def _run_label(path: Path) -> str:
    name = path.name
    for marker in ("_train_v31_1_", "v31_1_", "_train_v31_", "v31_"):
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def _first_failure(planes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in planes:
        if not record["topology_gate_passed"]:
            return {
                "plane": record["plane"],
                "overall": record["overall"],
                "prototype_label_audit_accuracy": record["prototype_label_audit_accuracy"],
                "focus_failures": record["focus_failures"],
            }
    return None


def _bridge_audit_failure(run: Path, train_audits: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    run_text = run.as_posix()
    label = _run_label(run)
    for train_run, audit in train_audits.items():
        if label and label in train_run and audit.get("passed") is False:
            return audit
        if run_text.replace("_eval", "") in train_run and audit.get("passed") is False:
            return audit
    return None


def _plane_pass(planes: list[dict[str, Any]], names: set[str]) -> bool:
    return any(record["plane"] in names and record["topology_gate_passed"] for record in planes)


def _classify(
    planes: list[dict[str, Any]],
    *,
    kind: str,
    base_classification: str,
    bridge_audit: dict[str, Any] | None,
) -> str:
    if bridge_audit is not None:
        return "harness-failure"
    bridge_pass = _plane_pass(planes, BRIDGE_PLANES)
    decoder_pass = _plane_pass(planes, DECODER_PLANES)
    physical_pass = _plane_pass(planes, {PHYSICAL_PLANE})
    if physical_pass:
        return "bridge-code-candidate-needs-human-review" if kind == "sample" else "bridge-code-candidate-needs-human-review"
    if decoder_pass and not physical_pass:
        return "camera-after-bridge-failure"
    if bridge_pass and not decoder_pass:
        return "decoder-after-bridge-failure"
    if base_classification == "metric-proxy-failure":
        return "metric-proxy-failure"
    return "bridge-collapse"


def analyze_bridge_run(
    run: Path,
    *,
    kind: str,
    oracle_baseline: dict[str, Any],
    train_audits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle, labels, reference, label_key, reference_key = _load_bundle(run, kind)
    metadata = _metadata_for_run(run, kind, bundle)
    base = analyze_run(run, kind=kind, oracle_baseline=oracle_baseline)
    planes = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            plane = _resize_like(value.detach().float().cpu(), reference)
            planes.append(_plane_record(key, plane, reference, labels))
        else:
            missing_planes.append(key)
    bridge_audit = _bridge_audit_failure(run, train_audits or {})
    classification = _classify(
        planes,
        kind=kind,
        base_classification=base.get("classification", ""),
        bridge_audit=bridge_audit,
    )
    return {
        "run_type": kind,
        "run": run.as_posix(),
        "label": _run_label(run),
        "tensor_bundle": base.get("tensor_bundle"),
        "metrics_path": base.get("metrics_path"),
        "v4_preview": _preview_path(run, kind),
        "label_key": label_key,
        "reference_key": reference_key,
        "metadata": {
            "target_code_diagnostic_mode": metadata.get("output_target_code_diagnostic_mode")
            or metadata.get("target_code_diagnostic_mode"),
            "target_code_bridge_source": metadata.get("output_target_code_bridge_source")
            or metadata.get("target_code_bridge_source"),
            "target_code_bridge_arch": metadata.get("output_target_code_bridge_arch")
            or metadata.get("target_code_bridge_arch"),
            "target_code_bridge_trainability_audit": metadata.get("target_code_bridge_trainability_audit"),
            "readout_primary_plane": metadata.get("readout_primary_plane"),
            "optical_core_arch": metadata.get("optical_core_arch"),
        },
        "base_decoder_controllability_classification": base.get("classification"),
        "missing_planes": missing_planes,
        "planes": planes,
        "first_human_topology_failure_plane": _first_failure(planes),
        "bridge_topology_passed": _plane_pass(planes, BRIDGE_PLANES),
        "decoder_or_chip_topology_passed": _plane_pass(planes, DECODER_PLANES),
        "physical_cmos_topology_passed": _plane_pass(planes, {PHYSICAL_PLANE}),
        "bridge_trainability_failure": bridge_audit,
        "classification": classification,
        "status": classification,
        "human_readability_required": True,
        "accepted_summary_written": False,
    }


def _decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    classification_runs: dict[str, list[str]] = {}
    for record in records:
        classification_runs.setdefault(record["classification"], []).append(record["run"])
    if classification_runs.get("bridge-code-candidate-needs-human-review"):
        status = "bridge-code-candidate-needs-human-review"
    elif classification_runs.get("camera-after-bridge-failure"):
        status = "camera-after-bridge-failure"
    elif classification_runs.get("decoder-after-bridge-failure"):
        status = "decoder-after-bridge-failure"
    elif classification_runs.get("harness-failure"):
        status = "harness-failure"
    elif classification_runs.get("metric-proxy-failure"):
        status = "metric-proxy-failure"
    else:
        status = "bridge-collapse"
    return {
        "status": status,
        "classification_runs": classification_runs,
        "bridge_code_reachable_runs": classification_runs.get("bridge-code-candidate-needs-human-review", []),
        "decoder_after_bridge_failure_runs": classification_runs.get("decoder-after-bridge-failure", []),
        "camera_after_bridge_failure_runs": classification_runs.get("camera-after-bridge-failure", []),
        "bridge_collapse_runs": classification_runs.get("bridge-collapse", []),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def run_v31_bridge_topology(
    *,
    oracle_report: Path | None = None,
    summary: Path | None = None,
    train_runs: list[Path] | None = None,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    output: Path | None = None,
    version: str = "v31.1",
    objective: str = "bridge_observability_same_batch_target_code_topology_gate",
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_train = list(train_runs or [])
    resolved_eval = list(eval_runs or [])
    resolved_sample = list(sample_runs or [])
    if summary is not None and not (resolved_train or resolved_eval or resolved_sample):
        resolved_train, resolved_eval, resolved_sample = _runs_from_summary(summary)
    resolved_train = _dedupe(resolved_train)
    resolved_eval = _dedupe(resolved_eval)
    resolved_sample = _dedupe(resolved_sample)
    if not resolved_eval and not resolved_sample:
        raise ValueError("at least one eval/sample run or summary with runs is required")
    oracle_baseline = _load_oracle_baseline(oracle_report, summary)
    train_audits = {run.as_posix(): _load_bridge_trainability_audit(run) for run in resolved_train}
    records = [
        analyze_bridge_run(run, kind="eval", oracle_baseline=oracle_baseline, train_audits=train_audits)
        for run in resolved_eval
    ]
    records.extend(
        analyze_bridge_run(run, kind="sample", oracle_baseline=oracle_baseline, train_audits=train_audits)
        for run in resolved_sample
    )
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": version,
        "objective": objective,
        "source_summary": summary.as_posix() if summary is not None else None,
        "oracle_report": oracle_report.as_posix() if oracle_report is not None else None,
        "train_runs": [run.as_posix() for run in resolved_train],
        "eval_runs": [run.as_posix() for run in resolved_eval],
        "sample_runs": [run.as_posix() for run in resolved_sample],
        "trainability_audits": train_audits,
        "runs": records,
        "decision": _decision(records),
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, default=None, help="Optional v26 OLS oracle report JSON.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional launcher summary JSON.")
    parser.add_argument("--train-run", action="append", type=Path, default=None, help="Existing bridge train run.")
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Existing bridge eval run.")
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing bridge sample run.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    parser.add_argument("--version", default="v31.1", help="Version label to write into the diagnostic report.")
    parser.add_argument(
        "--objective",
        default="bridge_observability_same_batch_target_code_topology_gate",
        help="Objective label to write into the diagnostic report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v31_bridge_topology(
        oracle_report=args.oracle_report,
        summary=args.summary,
        train_runs=args.train_run,
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        output=args.output,
        version=args.version,
        objective=args.objective,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
