#!/usr/bin/env python3
"""v48 transparent-decoder spacing / mask root-cause isolation diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output  # noqa: E402
from tools.diagnose_v34_code_coupling_recovery import _labels, _load_sample_bundle  # noqa: E402
from tools.diagnose_v40_sample_code_diversity_target_representation import (  # noqa: E402
    DECODER_INPUT_PLANE,
    DECODER_OUTPUT_PLANE,
    PHYSICAL_PLANE,
    TARGET_PLANE,
    _corr_mse,
    _normalize,
    _render_cmos_review,
)


PLANES = (TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE, PHYSICAL_PLANE)
REQUIRED_CLASSES = tuple(range(10))
VERDICT_TO_STATUS = {
    "t025_readable_m025_unreadable": "decoder-mask-induced-artifact",
    "t010_readable_t025_unreadable": "spacing-limited-below-formal-contract",
    "transparent_unreadable": "transparent-relay-insufficient",
    "m025_readable": "direct-decoder-human-candidate",
}


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v48_transparent_decoder_control_{timestamp}.json"


def _parse_probe_entry(entry: str) -> tuple[str, str, float | None, Path | None]:
    parts = entry.split("|")
    while len(parts) < 4:
        parts.append("")
    label, kind, spacing_mm, run = parts[:4]
    return label, kind, float(spacing_mm) if spacing_mm else None, Path(run) if run else None


def _metadata(run: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle.get("artifact_metadata")
    result = dict(metadata) if isinstance(metadata, dict) else {}
    path = run / "artifacts" / "one_shot_sample_artifact_metadata.json"
    if path.is_file():
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            result.update(loaded)
    config_path = run / "configs" / "train_config.json"
    if config_path.is_file():
        loaded = _load_json(config_path)
        if isinstance(loaded, dict):
            one_shot = loaded.get("one_shot")
            physics = loaded.get("physics")
            if isinstance(one_shot, dict):
                result.setdefault("decoder_diagnostic_mode", one_shot.get("decoder_diagnostic_mode"))
                result.setdefault(
                    "decoder_diagnostic_modulation_override",
                    one_shot.get("decoder_diagnostic_modulation_override"),
                )
                result.setdefault("target_code_diagnostic_mode", one_shot.get("target_code_diagnostic_mode"))
                result.setdefault("target_code_spatial_scale", one_shot.get("target_code_spatial_scale"))
                result.setdefault("target_code_bridge_source", one_shot.get("target_code_bridge_source"))
                config_d2nn = one_shot.get("d2nn")
                if isinstance(config_d2nn, dict):
                    artifact_d2nn = result.get("d2nn")
                    if isinstance(artifact_d2nn, dict):
                        merged_d2nn = dict(config_d2nn)
                        merged_d2nn.update(artifact_d2nn)
                        result["d2nn"] = merged_d2nn
                    else:
                        result["d2nn"] = config_d2nn
            if isinstance(physics, dict):
                result.setdefault("front_path_model", physics.get("front_path_model"))
                result.setdefault("back_path_model", physics.get("back_path_model"))
                readout = physics.get("readout")
                if isinstance(readout, dict):
                    result.setdefault("readout_primary_plane", readout.get("primary_plane"))
    return result


def _nested(metadata: dict[str, Any], *keys: str) -> Any:
    current: Any = metadata
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _metadata_value(metadata: dict[str, Any], key: str, *nested_keys: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    for prefix in ("training", "physics", "one_shot"):
        value = _nested(metadata, prefix, key)
        if value is not None:
            return value
    if nested_keys:
        return _nested(metadata, *nested_keys)
    return None


def _float_close(left: Any, right: float, *, atol: float = 1.0e-9) -> bool:
    try:
        return abs(float(left) - right) <= atol
    except (TypeError, ValueError):
        return False


def _validate_classes(labels: torch.Tensor) -> list[int]:
    present = {int(value) for value in labels.tolist()}
    return [class_id for class_id in REQUIRED_CLASSES if class_id not in present]


def _class_prototypes(tensor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = _normalize(tensor)
    prototypes = []
    for class_id in REQUIRED_CLASSES:
        mask = labels[: values.shape[0]] == class_id
        if not bool(mask.any()):
            raise ValueError(f"missing required review classes: {[class_id]}")
        prototypes.append(values[mask].mean(dim=0))
    return torch.stack(prototypes)


def _prototype_classifier_metrics(plane: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    values = _normalize(plane)
    prototypes = _class_prototypes(target, labels)
    if prototypes.shape[-2:] != values.shape[-2:]:
        prototypes = F.interpolate(prototypes, size=values.shape[-2:], mode="bilinear", align_corners=False)
    sample_count = min(int(values.shape[0]), int(labels.numel()))
    flat = values[:sample_count].flatten(1)
    proto_flat = prototypes.flatten(1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    proto_flat = proto_flat - proto_flat.mean(dim=1, keepdim=True)
    logits = flat @ proto_flat.T
    logits = logits / (flat.norm(dim=1, keepdim=True) * proto_flat.norm(dim=1).unsqueeze(0)).clamp_min(1.0e-8)
    top2 = torch.topk(logits, k=2, dim=1).values
    predictions = logits.argmax(dim=1)
    expected = labels[:sample_count]
    return {
        "accuracy": float((predictions == expected).float().mean().item()),
        "mean_requested_corr": float(logits[torch.arange(sample_count), expected].mean().item()),
        "mean_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
        "predictions": [int(value) for value in predictions.tolist()],
    }


def _validate_record(
    *,
    label: str,
    kind: str,
    spacing_mm: float | None,
    run: Path,
    bundle: dict[str, Any],
    labels: torch.Tensor,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if set(_validate_classes(labels)):
        errors.append(f"{label}: missing required review classes: {_validate_classes(labels)}")
    for plane in PLANES:
        value = bundle.get(plane)
        if not torch.is_tensor(value):
            errors.append(f"{label}: missing required tensor {plane}")
            continue
        if int(value.shape[0]) != int(labels.numel()):
            errors.append(f"{label}: {plane} batch does not match labels")
    for plane in (TARGET_PLANE, DECODER_INPUT_PLANE, DECODER_OUTPUT_PLANE):
        value = bundle.get(plane)
        if torch.is_tensor(value) and tuple(value.shape[-2:]) != (100, 100):
            errors.append(f"{label}: {plane} shape must end with 100x100, got {tuple(value.shape)}")
    physical = bundle.get(PHYSICAL_PLANE)
    if torch.is_tensor(physical) and tuple(physical.shape[-2:]) != (200, 200):
        errors.append(f"{label}: {PHYSICAL_PLANE} shape must end with 200x200, got {tuple(physical.shape)}")

    expected_mode = "propagation_only_decoder" if kind == "transparent" else "normal"
    checks = {
        "stage": "one_shot_sample",
        "decoder_diagnostic_mode": expected_mode,
        "decoder_diagnostic_modulation_override": "phase_amplitude",
        "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
        "target_code_bridge_source": "none",
        "front_path_model": "relay_intensity_psf",
        "back_path_model": "relay_psf",
        "readout_primary_plane": "camera",
    }
    for key, expected in checks.items():
        value = _metadata_value(metadata, key)
        if value != expected:
            errors.append(f"{label}: metadata {key}={value!r}, expected {expected!r}")
    if not _float_close(_metadata_value(metadata, "target_code_spatial_scale"), 0.72, atol=1.0e-6):
        errors.append(f"{label}: target_code_spatial_scale must be 0.72")
    d2nn = metadata.get("d2nn") if isinstance(metadata.get("d2nn"), dict) else {}
    if d2nn.get("grid_size") != 100:
        errors.append(f"{label}: d2nn.grid_size must be 100")
    if not _float_close(d2nn.get("pixel_size"), 2.0e-6, atol=1.0e-12):
        errors.append(f"{label}: d2nn.pixel_size must be 2e-6")
    if d2nn.get("propagation_grid_size") != 1024:
        errors.append(f"{label}: d2nn.propagation_grid_size must be 1024")
    if d2nn.get("decoder_layers") != 2:
        errors.append(f"{label}: d2nn.decoder_layers must be 2")
    if spacing_mm is not None and not _float_close(d2nn.get("layer_spacing"), spacing_mm * 1.0e-3, atol=1.0e-9):
        errors.append(f"{label}: d2nn.layer_spacing does not match probe entry")
    if kind == "transparent" and spacing_mm not in {0.10, 0.25}:
        errors.append(f"{label}: transparent spacing must be 0.10 or 0.25 mm")
    if kind == "masked" and spacing_mm != 0.25:
        errors.append(f"{label}: masked spacing must be 0.25 mm")
    if not run.is_dir():
        errors.append(f"{label}: run directory does not exist")
    return errors


def _comparison_review(records: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "v48_t010_t025_m025_cmos_only_comparison.png"
    fig, axes = plt.subplots(len(REQUIRED_CLASSES), len(records), figsize=(5.4, 18.0))
    for col, record in enumerate(records):
        bundle = _load_sample_bundle(Path(record["run"]))
        labels = _labels(bundle)
        values = _normalize(bundle[PHYSICAL_PLANE])
        for row, class_id in enumerate(REQUIRED_CLASSES):
            sample_index = int((labels == class_id).nonzero(as_tuple=False)[0].item())
            axis = axes[row][col]
            axis.imshow(values[sample_index, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(record["label"], fontsize=8)
            if col == 0:
                axis.set_ylabel(str(class_id), fontsize=8)
    fig.suptitle("v48 CMOS-only physical_camera_plane review", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()


def _decision(manual_verdict: str | None, errors: list[dict[str, str]]) -> dict[str, Any]:
    if errors:
        status = "harness-invalid"
        next_gate = "repair_v48_inputs"
    elif manual_verdict is None:
        status = "manual-review-required"
        next_gate = "human_review_t010_t025_m025_physical_camera_plane"
    elif manual_verdict not in VERDICT_TO_STATUS:
        status = "harness-invalid"
        next_gate = "provide_complete_v48_manual_verdict"
    else:
        status = VERDICT_TO_STATUS[manual_verdict]
        next_gate = "root_cause_followup"
    return {
        "status": status,
        "manual_verdict": manual_verdict,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": next_gate,
    }


def analyze_probe(
    *,
    label: str,
    kind: str,
    spacing_mm: float | None,
    run: Path,
    review_dir: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    metadata = _metadata(run, bundle)
    errors = _validate_record(
        label=label,
        kind=kind,
        spacing_mm=spacing_mm,
        run=run,
        bundle=bundle,
        labels=labels,
        metadata=metadata,
    )
    target_relative = {
        plane: _prototype_classifier_metrics(bundle[plane], bundle[TARGET_PLANE], labels)
        for plane in PLANES
        if torch.is_tensor(bundle.get(plane))
    }
    plane_corr_mse = {
        "decoder_output_vs_target": _corr_mse(bundle[DECODER_OUTPUT_PLANE], bundle[TARGET_PLANE]),
        "physical_vs_decoder": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
    }
    return (
        {
            "label": label,
            "run": run.as_posix(),
            "probe_kind": kind,
            "spacing_mm": spacing_mm,
            "checkpoint": metadata.get("one_shot_checkpoint"),
            "status": "harness-invalid" if errors else "manual-review-required",
            "target_code_diagnostic_mode": "small_sharp_stroke_digit_code",
            "target_code_spatial_scale": 0.72,
            "target_code_bridge_source": "none",
            "decoder_diagnostic_mode": metadata.get("decoder_diagnostic_mode"),
            "decoder_diagnostic_modulation_override": metadata.get("decoder_diagnostic_modulation_override"),
            "target_relative": target_relative,
            "plane_corr_mse": plane_corr_mse,
            "cmos_only_review_package": _render_cmos_review(
                label,
                run,
                bundle[PHYSICAL_PLANE],
                review_dir,
                class_labels=labels,
                review_version="v48",
            ),
            "plane_shapes": {plane: list(bundle[plane].shape) for plane in PLANES if torch.is_tensor(bundle.get(plane))},
            "accepted_summary_written": False,
            "human_readability_required": True,
        },
        errors,
    )


def _check_cross_probe_contract(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected = {"T010": ("transparent", 0.10), "T025": ("transparent", 0.25), "M025": ("masked", 0.25)}
    observed = {record["label"]: (record["probe_kind"], record["spacing_mm"]) for record in records}
    for label, contract in expected.items():
        if observed.get(label) != contract:
            errors.append(f"{label}: expected probe contract {contract}, got {observed.get(label)}")
    checkpoints = {record.get("checkpoint") for record in records}
    if len(checkpoints) != 1:
        errors.append(f"checkpoint mismatch across probes: {sorted(str(value) for value in checkpoints)}")
    prototypes = []
    for record in records:
        bundle = _load_sample_bundle(Path(record["run"]))
        labels = _labels(bundle)
        prototypes.append((record["label"], _class_prototypes(bundle[TARGET_PLANE], labels)))
    if prototypes:
        baseline_label, baseline = prototypes[0]
        for label, candidate in prototypes[1:]:
            if not torch.allclose(baseline, candidate, rtol=1.0e-5, atol=1.0e-6):
                errors.append(f"target-code source mismatch: {baseline_label} vs {label}")
    return errors


def run_v48_transparent_decoder_control(
    *,
    probe_entries: list[str],
    source_v47_report: Path | None = None,
    manual_verdict: str | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    source_v47_status = None
    if source_v47_report is not None:
        payload = _load_json(source_v47_report)
        assert payload is not None
        source_v47_status = payload.get("status")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in probe_entries:
        label, kind, spacing_mm, run = _parse_probe_entry(entry)
        try:
            if run is None:
                raise ValueError("missing sample run")
            record, probe_errors = analyze_probe(
                label=label,
                kind=kind,
                spacing_mm=spacing_mm,
                run=run,
                review_dir=review_dir,
            )
            records.append(record)
            errors.extend({"probe": label, "error": error} for error in probe_errors)
        except Exception as exc:
            errors.append({"probe": label or "unknown", "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label or "unknown",
                    "probe_kind": kind or "unknown",
                    "spacing_mm": spacing_mm,
                    "run": run.as_posix() if run else None,
                    "status": "harness-invalid",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )

    if not errors and len(records) == 3:
        errors.extend({"probe": "cross_probe", "error": error} for error in _check_cross_probe_contract(records))
    elif len(records) != 3:
        errors.append({"probe": "cross_probe", "error": f"expected exactly 3 probes, got {len(records)}"})

    comparison_review_image = None
    if not errors:
        comparison_review_image = _comparison_review(records, review_dir)
    decision = _decision(manual_verdict, errors)
    result = {
        "version": "v48",
        "objective": "transparent_decoder_spacing_mask_root_cause_isolation",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_v47_report": source_v47_report.as_posix() if source_v47_report else None,
        "source_v47_status": source_v47_status,
        "probes": records,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "review_planes": [PHYSICAL_PLANE],
        "review_class_labels": list(REQUIRED_CLASSES),
        "comparison_review_image": comparison_review_image,
        "manual_verdict_options": VERDICT_TO_STATUS,
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane human review can satisfy this gate.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", dest="probe_entries", action="append", required=True)
    parser.add_argument("--source-v47-report", type=Path)
    parser.add_argument("--manual-verdict", choices=sorted(VERDICT_TO_STATUS))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_v48_transparent_decoder_control(
        probe_entries=args.probe_entries,
        source_v47_report=args.source_v47_report,
        manual_verdict=args.manual_verdict,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output or "")}, sort_keys=True))
    return 0 if result["status"] != "harness-invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
