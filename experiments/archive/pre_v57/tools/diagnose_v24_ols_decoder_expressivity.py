#!/usr/bin/env python3
"""Offline v24 OLS/decoder expressivity diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import (
    _load_json,
    _metric_record,
    _prototype_label_audit,
    _require_report_output,
    _resize_like,
    _torch_load,
)


PLANE_ORDER = (
    "slm_command",
    "encoder_input_amplitude",
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "probe_plane",
    "physical_camera_plane",
)
REFERENCE_KEYS = (
    "reference_images",
    "eval_target",
    "teacher_camera",
    "original_digits",
    "target_images",
    "targets",
)
LABEL_KEYS = ("class_labels", "labels", "sample_labels")
PRE_DECODER_PLANES = {"slm_command", "encoder_input_amplitude", "encoded_amplitude", "ols_output_amplitude"}
CHIP_PROBE_PLANES = ("chip_intensity_primary", "probe_plane")
RETAINED_AUDIT = 0.60
COLLAPSE_AUDIT = 0.50
NO_RESCUE_AUDIT = 0.30


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v24_ols_decoder_expressivity_{timestamp}.json"


def _tensor_bundle_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_tensors.pt" if kind == "eval" else "one_shot_sample_tensors.pt"
    return run / "artifacts" / filename


def _metrics_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_metrics.json" if kind == "eval" else "one_shot_sample_metrics.json"
    return run / "artifacts" / filename


def _v4_preview_path(run: Path, kind: str) -> Path:
    filename = (
        "one_shot_distill_eval_100x100_cell_chain_preview.png"
        if kind == "eval"
        else "one_shot_sample_100x100_cell_chain_preview.png"
    )
    return run / "images" / filename


def _metadata_path(run: Path, kind: str) -> Path:
    filename = (
        "one_shot_distill_eval_artifact_metadata.json"
        if kind == "eval"
        else "one_shot_sample_artifact_metadata.json"
    )
    return run / "artifacts" / filename


def _load_metadata(run: Path, kind: str, bundle: dict[str, Any]) -> tuple[dict[str, Any], str]:
    sidecar = _metadata_path(run, kind)
    if sidecar.exists():
        payload = _load_json(sidecar, required=True)
        if isinstance(payload, dict):
            return payload, sidecar.as_posix()
    payload = bundle.get("artifact_metadata")
    if isinstance(payload, dict):
        return payload, "tensor_bundle.artifact_metadata"
    return {}, "missing"


def _first_tensor(bundle: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, torch.Tensor | None]:
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, value
    return None, None


def _string_value(bundle: dict[str, Any], key: str, default: str) -> str:
    value = bundle.get(key, default)
    return value if isinstance(value, str) else default


def _metadata_string(metadata: dict[str, Any], bundle: dict[str, Any], key: str, default: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    output_key = f"output_{key}"
    output_value = metadata.get(output_key)
    if isinstance(output_value, str):
        return output_value
    return _string_value(bundle, key, default)


def _bool_value(bundle: dict[str, Any], key: str) -> bool:
    value = bundle.get(key)
    if torch.is_tensor(value):
        return bool(value.detach().cpu().item())
    return bool(value)


def _metadata_bool(metadata: dict[str, Any], bundle: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    output_key = f"output_{key}"
    output_value = metadata.get(output_key)
    if isinstance(output_value, bool):
        return output_value
    return _bool_value(bundle, key)


def _load_bundle(run: Path, kind: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str, str]:
    path = _tensor_bundle_path(run, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = _torch_load(path)
    label_key, labels = _first_tensor(bundle, LABEL_KEYS)
    reference_key, reference = _first_tensor(bundle, REFERENCE_KEYS)
    if labels is None or reference is None:
        missing = []
        if labels is None:
            missing.append("class labels")
        if reference is None:
            missing.append("reference/target images")
        raise ValueError(f"{kind} tensor bundle is missing required {' and '.join(missing)}: {path}")
    return bundle, labels.detach().cpu().long(), reference.detach().float().cpu(), str(label_key), str(reference_key)


def _energy_contrast(plane: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    foreground = reference > 0.20
    background = ~foreground
    record = {
        "mean": float(plane.mean().item()),
        "std": float(plane.std(unbiased=False).item()),
        "min": float(plane.min().item()),
        "max": float(plane.max().item()),
        "l2_energy": float(plane.pow(2).mean().item()),
        "foreground_mean": None,
        "background_mean": None,
        "foreground_background_contrast": None,
    }
    if bool(foreground.any()) and bool(background.any()):
        fg = plane[foreground].mean()
        bg = plane[background].mean()
        record["foreground_mean"] = float(fg.item())
        record["background_mean"] = float(bg.item())
        record["foreground_background_contrast"] = float((fg - bg).item())
    return record


def _plane_record(key: str, tensor: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    plane = _resize_like(tensor, reference)
    metrics = _metric_record(plane, reference, labels)
    audit = _prototype_label_audit(plane, reference, labels)
    return {
        "plane": key,
        "shape": list(tensor.shape),
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "focus_classes": metrics["focus_classes"],
        "energy_contrast": _energy_contrast(plane, reference),
        "prototype_label_audit_accuracy": audit.get("correct_ratio"),
        "prototype_label_confusion": audit.get("confusion"),
        "prototype_label_audit": audit,
    }


def _plane_by_name(planes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((record for record in planes if record["plane"] == name), None)


def _audit(record: dict[str, Any] | None) -> float | None:
    if record is None or record.get("prototype_label_audit_accuracy") is None:
        return None
    return float(record["prototype_label_audit_accuracy"])


def _first_semantic_collapse(planes: list[dict[str, Any]]) -> dict[str, Any] | None:
    previous_good: dict[str, Any] | None = None
    for record in planes:
        accuracy = _audit(record)
        if accuracy is None:
            continue
        if record["plane"] in PRE_DECODER_PLANES and accuracy >= RETAINED_AUDIT:
            previous_good = record
        if previous_good is not None and accuracy < COLLAPSE_AUDIT:
            return {
                "plane": record["plane"],
                "prototype_label_audit_accuracy": accuracy,
                "previous_retained_plane": previous_good["plane"],
                "previous_retained_accuracy": previous_good.get("prototype_label_audit_accuracy"),
                "thresholds": {
                    "retained_audit": RETAINED_AUDIT,
                    "collapse_audit": COLLAPSE_AUDIT,
                },
            }
    return None


def _best_chip_probe_audit(planes: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        record
        for record in planes
        if record["plane"] in CHIP_PROBE_PLANES and _audit(record) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record["prototype_label_audit_accuracy"]))


def _collapse_moved_past_decoder(first_collapse: dict[str, Any] | None) -> bool:
    if first_collapse is None:
        return True
    order = {name: index for index, name in enumerate(PLANE_ORDER)}
    return order.get(first_collapse["plane"], -1) > order["decoder_output_amplitude"]


def _classify_run(
    *,
    planes: list[dict[str, Any]],
    first_collapse: dict[str, Any] | None,
    decoder_bypass_active: bool,
    ols_transfer_calibration: str,
) -> str:
    best_chip_probe = _best_chip_probe_audit(planes)
    best_chip_probe_accuracy = _audit(best_chip_probe)
    chip_probe_rescued = best_chip_probe_accuracy is not None and best_chip_probe_accuracy >= RETAINED_AUDIT
    moved_past_decoder = _collapse_moved_past_decoder(first_collapse)

    if decoder_bypass_active and (chip_probe_rescued or moved_past_decoder):
        return "decoder-bottleneck"
    if (
        ols_transfer_calibration == "none"
        and first_collapse is not None
        and first_collapse.get("plane") == "decoder_output_amplitude"
    ):
        return "decoder-bottleneck"
    if ols_transfer_calibration == "phase_screen" and (chip_probe_rescued or moved_past_decoder):
        return "ols-transfer-bottleneck"
    if ols_transfer_calibration == "amplitude_phase_screen" and (chip_probe_rescued or moved_past_decoder):
        return "upper-bound-only"

    decoder = _plane_by_name(planes, "decoder_output_amplitude")
    chip = _plane_by_name(planes, "chip_intensity_primary")
    if max(_audit(decoder) or 0.0, _audit(chip) or 0.0) <= NO_RESCUE_AUDIT:
        return "no-expressivity-rescue"
    return "diagnostic-only-inconclusive"


def analyze_run(run: Path, *, kind: str) -> dict[str, Any]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle, labels, reference, label_key, reference_key = _load_bundle(run, kind)
    metadata, metadata_source = _load_metadata(run, kind, bundle)
    plane_records = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            plane_records.append(_plane_record(key, value.detach().float().cpu(), reference, labels))
        else:
            missing_planes.append(key)
    first_collapse = _first_semantic_collapse(plane_records)
    decoder_bypass_active = _metadata_bool(metadata, bundle, "decoder_bypass_active")
    ols_transfer_calibration = _metadata_string(metadata, bundle, "ols_transfer_calibration", "none")
    decoder_diagnostic_mode = _metadata_string(metadata, bundle, "decoder_diagnostic_mode", "normal")
    decoder_modulation_override = _metadata_string(
        metadata,
        bundle,
        "decoder_diagnostic_modulation_override",
        "none",
    )
    classification = _classify_run(
        planes=plane_records,
        first_collapse=first_collapse,
        decoder_bypass_active=decoder_bypass_active,
        ols_transfer_calibration=ols_transfer_calibration,
    )
    best_chip_probe = _best_chip_probe_audit(plane_records)
    return {
        "run_type": kind,
        "run": run.as_posix(),
        "tensor_bundle": _tensor_bundle_path(run, kind).as_posix(),
        "metrics_path": _metrics_path(run, kind).as_posix(),
        "stored_metrics": _load_json(_metrics_path(run, kind), required=False),
        "v4_preview": _v4_preview_path(run, kind).as_posix(),
        "label_key": label_key,
        "reference_key": reference_key,
        "artifact_metadata": metadata,
        "artifact_metadata_source": metadata_source,
        "decoder_diagnostic_mode": decoder_diagnostic_mode,
        "decoder_diagnostic_modulation_override": decoder_modulation_override,
        "decoder_bypass_active": decoder_bypass_active,
        "ols_transfer_calibration": ols_transfer_calibration,
        "plane_order": list(PLANE_ORDER),
        "missing_planes": missing_planes,
        "planes": plane_records,
        "first_semantic_collapse_plane": first_collapse,
        "best_chip_probe_plane": best_chip_probe,
        "collapse_moved_past_decoder": _collapse_moved_past_decoder(first_collapse),
        "classification": classification,
        "status": classification,
        "accepted_summary_written": False,
    }


def _runs_from_summary(summary_path: Path) -> tuple[list[Path], list[Path]]:
    summary = _load_json(summary_path)
    assert summary is not None
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    for probe in summary.get("probes", []):
        if isinstance(probe, dict) and probe.get("eval_run"):
            eval_runs.append(Path(probe["eval_run"]))
    for item in summary.get("sample_runs", []):
        if isinstance(item, dict) and item.get("sample_run"):
            sample_runs.append(Path(item["sample_run"]))
        elif isinstance(item, str):
            sample_runs.append(Path(item))
    return eval_runs, sample_runs


def run_ols_decoder_expressivity(
    *,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_eval_runs = list(eval_runs or [])
    resolved_sample_runs = list(sample_runs or [])
    if summary is not None and not (resolved_eval_runs or resolved_sample_runs):
        resolved_eval_runs, resolved_sample_runs = _runs_from_summary(summary)
    if not resolved_eval_runs and not resolved_sample_runs:
        raise ValueError("at least one --eval-run, --sample-run, or a summary with runs is required")
    run_records = [analyze_run(Path(run), kind="eval") for run in resolved_eval_runs]
    run_records.extend(analyze_run(Path(run), kind="sample") for run in resolved_sample_runs)
    classification_runs: dict[str, list[str]] = {}
    for record in run_records:
        classification_runs.setdefault(record["classification"], []).append(record["run"])
    sample_no_rescue = [
        record["run"]
        for record in run_records
        if record["run_type"] == "sample" and record["classification"] == "no-expressivity-rescue"
    ]
    if sample_no_rescue:
        status = "no-expressivity-rescue"
    else:
        priority = (
            "decoder-bottleneck",
            "ols-transfer-bottleneck",
            "upper-bound-only",
            "no-expressivity-rescue",
            "diagnostic-only-inconclusive",
        )
        status = next((item for item in priority if classification_runs.get(item)), "diagnostic-only-inconclusive")
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v24",
        "objective": "ols_decoder_expressivity_isolation",
        "source_summary": summary.as_posix() if summary is not None else None,
        "eval_runs": [Path(run).as_posix() for run in resolved_eval_runs],
        "sample_runs": [Path(run).as_posix() for run in resolved_sample_runs],
        "thresholds": {
            "retained_audit": RETAINED_AUDIT,
            "collapse_audit": COLLAPSE_AUDIT,
            "no_rescue_audit": NO_RESCUE_AUDIT,
        },
        "runs": run_records,
        "decision": {
            "status": status,
            "classification_runs": classification_runs,
            "accepted_summary_written": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Existing one-shot eval run directory.")
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional launcher summary JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ols_decoder_expressivity(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summary=args.summary,
        output=args.output,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
