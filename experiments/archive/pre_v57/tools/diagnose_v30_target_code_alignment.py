#!/usr/bin/env python3
"""v30 target-code / D2NN architecture hard-isolation diagnostic."""

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
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
REACHABLE = {
    "decoder-topology-reachable",
    "camera-after-decoder-failure",
    "human-accepted-candidate-needs-v4-review",
}
PROBE_A_MARKERS = ("spatial_identity_code", "probe_a", "spatial_identity")
PROBE_B_MARKERS = ("lowfreq_digit_code", "probe_b", "lowfreq")
PROBE_C_MARKERS = ("class_codebook_grid", "probe_c", "codebook")
PROBE_D_MARKERS = ("camera_direct", "probe_d", "oracle")


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v30_target_code_alignment_{timestamp}.json"


def _extract_runs(payload: Any, eval_runs: list[Path], sample_runs: list[Path], train_runs: list[Path]) -> None:
    if isinstance(payload, dict):
        if payload.get("same_batch_eval_run"):
            eval_runs.append(Path(str(payload["same_batch_eval_run"])))
        if payload.get("test_eval_run"):
            eval_runs.append(Path(str(payload["test_eval_run"])))
        if payload.get("eval_run"):
            eval_runs.append(Path(str(payload["eval_run"])))
        if payload.get("sample_run"):
            sample_runs.append(Path(str(payload["sample_run"])))
        if payload.get("train_run"):
            train_runs.append(Path(str(payload["train_run"])))
        for value in payload.values():
            _extract_runs(value, eval_runs, sample_runs, train_runs)
    elif isinstance(payload, list):
        for item in payload:
            _extract_runs(item, eval_runs, sample_runs, train_runs)


def _runs_from_summary(summary: Path) -> tuple[list[Path], list[Path], list[Path]]:
    payload = _load_json(summary)
    assert payload is not None
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    train_runs: list[Path] = []
    _extract_runs(payload, eval_runs, sample_runs, train_runs)
    return _dedupe(eval_runs), _dedupe(sample_runs), _dedupe(train_runs)


def _metadata_for_run(run: Path, kind: str, bundle: dict[str, Any]) -> dict[str, Any]:
    suffix = "one_shot_distill_eval_artifact_metadata.json" if kind == "eval" else "one_shot_sample_artifact_metadata.json"
    sidecar = _load_json(run / "artifacts" / suffix, required=False)
    metadata = sidecar if isinstance(sidecar, dict) else {}
    bundle_metadata = bundle.get("artifact_metadata")
    if isinstance(bundle_metadata, dict):
        metadata = {**bundle_metadata, **metadata}
    return metadata


def _string_from_bundle(bundle: dict[str, Any], key: str) -> str | None:
    value = bundle.get(key)
    return value if isinstance(value, str) else None


def _target_code_mode(bundle: dict[str, Any], metadata: dict[str, Any]) -> str:
    for key in (
        "output_target_code_diagnostic_mode",
        "target_code_diagnostic_mode",
        "one_shot_target_code_diagnostic_mode",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return _string_from_bundle(bundle, "target_code_diagnostic_mode") or "none"


def _target_code_bridge_source(bundle: dict[str, Any], metadata: dict[str, Any]) -> str:
    for key in (
        "output_target_code_bridge_source",
        "target_code_bridge_source",
        "one_shot_target_code_bridge_source",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return _string_from_bundle(bundle, "target_code_bridge_source") or "none"


def _optical_arch(metadata: dict[str, Any]) -> str:
    value = metadata.get("optical_core_arch") or metadata.get("one_shot_optical_core_arch")
    return str(value) if value else "unknown"


def _high_frequency_ratio(tensor: torch.Tensor) -> float:
    x = tensor.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]).mean(dim=1, keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1)), dim=(-2, -1)).abs().pow(2)
    h, w = spectrum.shape[-2:]
    yy = torch.arange(h, device=spectrum.device).view(h, 1)
    xx = torch.arange(w, device=spectrum.device).view(1, w)
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    radius = torch.sqrt((yy - center_y).float().pow(2) + (xx - center_x).float().pow(2))
    mask = radius > (min(h, w) * 0.25)
    total = spectrum.sum(dim=(-2, -1)).clamp_min(1.0e-8)
    high = spectrum[..., mask].sum(dim=-1)
    return float((high / total).mean().item())


def _alignment_summary(key: str, tensor: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    plane = _resize_like(tensor.detach().float().cpu(), reference)
    record = _plane_record(key, plane, reference, labels)
    return {
        "plane": key,
        "overall": record["overall"],
        "prototype_label_audit_accuracy": record["prototype_label_audit_accuracy"],
        "focus_failures": record["focus_failures"],
        "high_frequency_energy_ratio": _high_frequency_ratio(plane),
        "topology_gate_passed": record["topology_gate_passed"],
    }


def _run_label(path: Path) -> str:
    name = path.name
    for marker in ("_train_v30_", "v30_"):
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def _matches(record: dict[str, Any], markers: tuple[str, ...]) -> bool:
    haystack = " ".join(
        [
            str(record.get("run", "")),
            str(record.get("label", "")),
            str(record.get("target_code_diagnostic_mode", "")),
            str(record.get("optical_core_arch", "")),
        ]
    )
    return any(marker in haystack for marker in markers)


def analyze_alignment_run(run: Path, *, kind: str, oracle_baseline: dict[str, Any]) -> dict[str, Any]:
    base = analyze_run(run, kind=kind, oracle_baseline=oracle_baseline)
    bundle, labels, reference, _, _ = _load_bundle(run, kind)
    metadata = _metadata_for_run(run, kind, bundle)
    alignments = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            alignments.append(_alignment_summary(key, value, reference, labels))
        else:
            missing_planes.append(key)
    base.update(
        {
            "label": _run_label(run),
            "target_code_diagnostic_mode": _target_code_mode(bundle, metadata),
            "target_code_diagnostic_active": _target_code_mode(bundle, metadata) != "none",
            "target_code_bridge_source": _target_code_bridge_source(bundle, metadata),
            "target_code_bridge_active": _target_code_bridge_source(bundle, metadata) != "none",
            "optical_core_arch": _optical_arch(metadata),
            "alignment_planes": alignments,
            "alignment_missing_planes": missing_planes,
        }
    )
    return base


def _decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    reachable = [record for record in records if record.get("classification") in REACHABLE]
    sample_positive = [
        record for record in records if record.get("classification") == "human-accepted-candidate-needs-v4-review"
    ]
    camera_after = [record for record in records if record.get("classification") == "camera-after-decoder-failure"]
    non_oracle_reachable = [
        record
        for record in reachable
        if record.get("optical_core_arch") != "camera_direct_clean_oracle" and not _matches(record, PROBE_D_MARKERS)
    ]
    spatial = [record for record in non_oracle_reachable if _matches(record, PROBE_A_MARKERS)]
    lowfreq = [record for record in non_oracle_reachable if _matches(record, PROBE_B_MARKERS)]
    codebook = [record for record in non_oracle_reachable if _matches(record, PROBE_C_MARKERS)]
    oracle_only = [record for record in reachable if record not in non_oracle_reachable]
    if sample_positive:
        status = "human-accepted-candidate-needs-v4-review"
    elif camera_after:
        status = "camera-after-code-rescue"
    elif lowfreq and not (spatial and codebook):
        status = "frequency-encoding-bottleneck"
    elif codebook and not (spatial and lowfreq):
        status = "semantic-codebook-rescue"
    elif non_oracle_reachable:
        status = "target-code-rescue"
    elif records and all(record.get("classification") == "metric-proxy-failure" for record in records):
        status = "metric-proxy-failure"
    else:
        status = "architecture-unreachable"
    if oracle_only and not non_oracle_reachable and status in {"architecture-unreachable", "metric-proxy-failure"}:
        status = "architecture-unreachable"
    return {
        "status": status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "target_code_rescue_runs": [record["run"] for record in non_oracle_reachable],
        "frequency_encoding_bottleneck_runs": [record["run"] for record in lowfreq],
        "semantic_codebook_rescue_runs": [record["run"] for record in codebook],
        "camera_after_code_rescue_runs": [record["run"] for record in camera_after],
        "oracle_only_runs": [record["run"] for record in oracle_only],
        "sample_positive_runs": [record["run"] for record in sample_positive],
        "classification_runs": {
            status_key: [record["run"] for record in records if record.get("classification") == status_key]
            for status_key in sorted({str(record.get("classification")) for record in records})
        },
    }


def run_v30_target_code_alignment(
    *,
    oracle_report: Path | None = None,
    summary: Path | None = None,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    train_runs: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_eval = list(eval_runs or [])
    resolved_sample = list(sample_runs or [])
    resolved_train = list(train_runs or [])
    if summary is not None and not (resolved_eval or resolved_sample or resolved_train):
        resolved_eval, resolved_sample, resolved_train = _runs_from_summary(summary)
    resolved_eval = _dedupe(resolved_eval)
    resolved_sample = _dedupe(resolved_sample)
    resolved_train = _dedupe(resolved_train)
    if not resolved_eval and not resolved_sample:
        raise ValueError("at least one eval/sample run or summary with runs is required")
    oracle_baseline = _load_oracle_baseline(oracle_report, summary)
    records = [analyze_alignment_run(run, kind="eval", oracle_baseline=oracle_baseline) for run in resolved_eval]
    records.extend(analyze_alignment_run(run, kind="sample", oracle_baseline=oracle_baseline) for run in resolved_sample)
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v30",
        "objective": "target_code_d2nn_architecture_hard_isolation",
        "source_summary": summary.as_posix() if summary is not None else None,
        "oracle_report": oracle_report.as_posix() if oracle_report is not None else None,
        "train_runs": [run.as_posix() for run in resolved_train],
        "eval_runs": [run.as_posix() for run in resolved_eval],
        "sample_runs": [run.as_posix() for run in resolved_sample],
        "runs": records,
        "decision": _decision(records),
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--train-run", action="append", type=Path, default=None)
    parser.add_argument("--eval-run", action="append", type=Path, default=None)
    parser.add_argument("--sample-run", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v30_target_code_alignment(
        oracle_report=args.oracle_report,
        summary=args.summary,
        train_runs=args.train_run,
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
