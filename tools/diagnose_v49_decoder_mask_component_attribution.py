#!/usr/bin/env python3
"""v49 decoder phase/amplitude mask-component attribution diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch

from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v34_code_coupling_recovery import _labels, _load_sample_bundle
from tools.diagnose_v40_sample_code_diversity_target_representation import (
    DECODER_INPUT_PLANE,
    DECODER_OUTPUT_PLANE,
    PHYSICAL_PLANE,
    TARGET_PLANE,
    _corr_mse,
    _normalize,
    _render_cmos_review,
)
from tools.diagnose_v48_transparent_decoder_control import (
    PLANES,
    REQUIRED_CLASSES,
    _class_prototypes,
    _float_close,
    _metadata,
    _metadata_value,
    _prototype_classifier_metrics,
    _validate_classes,
)


VERDICT_TO_STATUS = {
    "phase_only_unreadable": "phase-mask-induced-artifact",
    "amplitude_only_unreadable": "amplitude-mask-induced-artifact",
    "both_components_unreadable": "multiple-mask-components-artifact",
    "both_components_readable": "phase-amplitude-joint-interaction-artifact",
    "source_gate_not_reproduced": "source-gate-not-reproduced",
}
EXPECTED_COMPONENTS = {
    "T025": "transparent",
    "P025": "phase_only",
    "A025": "amplitude_only",
    "M025": "trained",
}


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v49_decoder_mask_component_attribution_{timestamp}.json"


def _parse_probe_entry(entry: str) -> tuple[str, str, Path | None]:
    parts = entry.split("|", maxsplit=2)
    while len(parts) < 3:
        parts.append("")
    label, component, run = parts
    return label, component, Path(run) if run else None


def _source_probe(payload: dict[str, Any], label: str) -> dict[str, Any] | None:
    for probe in payload.get("probes", []):
        if isinstance(probe, dict) and probe.get("label") == label:
            return probe
    return None


def _v47_probe_a(payload: dict[str, Any]) -> dict[str, Any] | None:
    for probe in payload.get("probes", []):
        if not isinstance(probe, dict):
            continue
        if "probe_a_s025_phase_amplitude_decoder_output" in str(probe.get("label", "")):
            return probe
    return None


def _training_seed(run: Path) -> int | None:
    path = run / "configs" / "train_config.json"
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    if payload.get("seed") is not None:
        return int(payload["seed"])
    training = payload.get("training")
    if isinstance(training, dict) and training.get("seed") is not None:
        return int(training["seed"])
    return None


def _validate_probe(
    *,
    label: str,
    component: str,
    run: Path,
    bundle: dict[str, Any],
    labels: torch.Tensor,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    missing_classes = _validate_classes(labels)
    if missing_classes:
        errors.append(f"{label}: missing required review classes: {missing_classes}")
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

    expected_mode = "propagation_only_decoder" if component == "transparent" else "normal"
    expected_mask_component = "trained" if component in {"transparent", "trained"} else component
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
    observed_component = _metadata_value(metadata, "decoder_diagnostic_mask_component") or "trained"
    if observed_component != expected_mask_component:
        errors.append(
            f"{label}: decoder_diagnostic_mask_component={observed_component!r}, "
            f"expected {expected_mask_component!r}"
        )
    if not _float_close(_metadata_value(metadata, "target_code_spatial_scale"), 0.72, atol=1.0e-6):
        errors.append(f"{label}: target_code_spatial_scale must be 0.72")
    if _training_seed(run) != 1234:
        errors.append(f"{label}: training.seed must be 1234")
    d2nn = metadata.get("d2nn") if isinstance(metadata.get("d2nn"), dict) else {}
    if d2nn.get("grid_size") != 100:
        errors.append(f"{label}: d2nn.grid_size must be 100")
    if not _float_close(d2nn.get("pixel_size"), 2.0e-6, atol=1.0e-12):
        errors.append(f"{label}: d2nn.pixel_size must be 2e-6")
    if d2nn.get("propagation_grid_size") != 1024:
        errors.append(f"{label}: d2nn.propagation_grid_size must be 1024")
    if d2nn.get("decoder_layers") != 2:
        errors.append(f"{label}: d2nn.decoder_layers must be 2")
    if not _float_close(d2nn.get("layer_spacing"), 0.25e-3, atol=1.0e-9):
        errors.append(f"{label}: d2nn.layer_spacing must be 0.25 mm")
    if not run.is_dir():
        errors.append(f"{label}: run directory does not exist")
    return errors


def _analyze_probe(
    *,
    label: str,
    component: str,
    run: Path,
    review_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    metadata = _metadata(run, bundle)
    errors = _validate_probe(
        label=label,
        component=component,
        run=run,
        bundle=bundle,
        labels=labels,
        metadata=metadata,
    )
    return (
        {
            "label": label,
            "run": run.as_posix(),
            "mask_component": component,
            "checkpoint": metadata.get("one_shot_checkpoint"),
            "seed": _training_seed(run),
            "status": "harness-invalid" if errors else "manual-review-required",
            "target_relative": {
                plane: _prototype_classifier_metrics(bundle[plane], bundle[TARGET_PLANE], labels)
                for plane in PLANES
                if torch.is_tensor(bundle.get(plane))
            },
            "plane_corr_mse": {
                "decoder_output_vs_target": _corr_mse(bundle[DECODER_OUTPUT_PLANE], bundle[TARGET_PLANE]),
                "physical_vs_decoder": _corr_mse(bundle[PHYSICAL_PLANE], bundle[DECODER_OUTPUT_PLANE]),
            },
            "plane_shapes": {
                plane: list(bundle[plane].shape)
                for plane in PLANES
                if torch.is_tensor(bundle.get(plane))
            },
            "cmos_only_review_package": _render_cmos_review(
                label,
                run,
                bundle[PHYSICAL_PLANE],
                review_dir,
                class_labels=labels,
                review_version="v49",
            ),
            "accepted_summary_written": False,
            "human_readability_required": True,
        },
        errors,
    )


def _comparison_review(records: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "v49_t025_p025_a025_m025_cmos_only_comparison.png"
    fig, axes = plt.subplots(len(REQUIRED_CLASSES), len(records), figsize=(7.0, 18.0))
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
    fig.suptitle("v49 CMOS-only physical_camera_plane review", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()


def _capacity_utilization(v47_probe: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    train_run = Path(str(v47_probe.get("train_run") or ""))
    audit_path = train_run / "artifacts" / "one_shot_decoder_trainability_audit.json"
    diagnostics_paths = sorted((train_run / "artifacts" / "one_shot_distill").glob("epoch_*_d2nn_diagnostics.json"))
    trainability = _load_json(audit_path)
    if not isinstance(trainability, dict):
        errors.append(f"missing v47 decoder trainability audit: {audit_path}")
        trainability = {}
    diagnostics = _load_json(diagnostics_paths[-1]) if diagnostics_paths else None
    if not isinstance(diagnostics, dict):
        errors.append(f"missing v47 final D2NN diagnostics under {train_run}")
        diagnostics = {}
    masks = diagnostics.get("masks") if isinstance(diagnostics.get("masks"), dict) else {}
    decoder_masks: dict[str, dict[str, Any]] = {}
    for component in ("phase", "amplitude"):
        values = masks.get(component) if isinstance(masks.get(component), dict) else {}
        decoder_masks[component] = {
            key: value for key, value in values.items() if str(key).startswith("decoder mask")
        }
    return (
        {
            "train_run": train_run.as_posix(),
            "trainability": trainability,
            "final_d2nn_diagnostics": diagnostics_paths[-1].as_posix() if diagnostics_paths else None,
            "decoder_masks": decoder_masks,
            "automatic_capacity_threshold": None,
        },
        errors,
    )


def _cross_probe_contract(records: list[dict[str, Any]], expected_checkpoint: str) -> list[str]:
    errors: list[str] = []
    observed = {record.get("label"): record.get("mask_component") for record in records}
    if observed != EXPECTED_COMPONENTS:
        errors.append(f"probe contract mismatch: expected {EXPECTED_COMPONENTS}, got {observed}")
    checkpoints = {record.get("checkpoint") for record in records}
    if checkpoints != {expected_checkpoint}:
        errors.append(f"checkpoint mismatch across probes: {sorted(str(value) for value in checkpoints)}")
    seeds = {record.get("seed") for record in records}
    if seeds != {1234}:
        errors.append(f"seed mismatch across probes: {sorted(str(value) for value in seeds)}")
    prototypes = []
    for record in records:
        bundle = _load_sample_bundle(Path(record["run"]))
        prototypes.append((record["label"], _class_prototypes(bundle[TARGET_PLANE], _labels(bundle))))
    baseline_label, baseline = prototypes[0]
    for label, candidate in prototypes[1:]:
        if not torch.allclose(baseline, candidate, rtol=1.0e-5, atol=1.0e-6):
            errors.append(f"target-code source mismatch: {baseline_label} vs {label}")
    return errors


def _control_relative_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    bundles = {record["label"]: _load_sample_bundle(Path(record["run"])) for record in records}
    return {
        label: {
            "physical_vs_T025": _corr_mse(bundles[label][PHYSICAL_PLANE], bundles["T025"][PHYSICAL_PLANE]),
            "physical_vs_M025": _corr_mse(bundles[label][PHYSICAL_PLANE], bundles["M025"][PHYSICAL_PLANE]),
        }
        for label in ("P025", "A025")
    }


def _decision(manual_verdict: str | None, errors: list[dict[str, str]], *, preflight_only: bool) -> dict[str, Any]:
    if errors:
        status = "harness-invalid"
        next_gate = "repair_v49_inputs"
    elif preflight_only:
        status = "preflight-valid"
        next_gate = "run_p025_a025_samples"
    elif manual_verdict is None:
        status = "manual-review-required"
        next_gate = "human_review_t025_p025_a025_m025_physical_camera_plane"
    elif manual_verdict not in VERDICT_TO_STATUS:
        status = "harness-invalid"
        next_gate = "provide_complete_v49_manual_verdict"
    else:
        status = VERDICT_TO_STATUS[manual_verdict]
        next_gate = "stop_v49_after_manual_gate"
    return {
        "status": status,
        "manual_verdict": manual_verdict,
        "accepted_summary_written": False,
        "human_readability_required": not preflight_only,
        "next_gate": next_gate,
    }


def run_v49_decoder_mask_component_attribution(
    *,
    source_v48_verdict: Path,
    source_v47_summary: Path,
    probe_entries: list[str],
    manual_verdict: str | None = None,
    preflight_only: bool = False,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    review_dir = output_path.with_suffix("")
    errors: list[dict[str, str]] = []
    v48 = _load_json(source_v48_verdict)
    v47 = _load_json(source_v47_summary)
    if not isinstance(v48, dict) or v48.get("status") != "decoder-mask-induced-artifact" or v48.get("errors"):
        errors.append({"probe": "source", "error": "v48 source verdict must be decoder-mask-induced-artifact with no errors"})
        v48 = v48 if isinstance(v48, dict) else {}
    if not isinstance(v47, dict):
        errors.append({"probe": "source", "error": "v47 summary is missing or invalid"})
        v47 = {}
    v47_probe = _v47_probe_a(v47)
    if v47_probe is None:
        errors.append({"probe": "source", "error": "missing v47 Probe A source contract"})
        v47_probe = {}
    expected_checkpoint = str(v47_probe.get("checkpoint") or "")
    capacity, capacity_errors = _capacity_utilization(v47_probe)
    errors.extend({"probe": "capacity", "error": error} for error in capacity_errors)

    source_runs: dict[str, tuple[str, Path]] = {}
    for label, component in (("T025", "transparent"), ("M025", "trained")):
        source = _source_probe(v48, label)
        run_value = source.get("run") if isinstance(source, dict) else None
        if not run_value:
            errors.append({"probe": label, "error": f"missing {label} run in v48 verdict"})
            continue
        if str(source.get("checkpoint") or "") != expected_checkpoint:
            errors.append({"probe": label, "error": f"{label} checkpoint does not match v47 Probe A"})
        source_runs[label] = (component, Path(str(run_value)))

    parsed_entries = [_parse_probe_entry(entry) for entry in probe_entries]
    if not preflight_only and len(parsed_entries) != 2:
        errors.append({"probe": "cross_probe", "error": f"expected exactly 2 new probes, got {len(parsed_entries)}"})
    new_runs = {
        label: (component, run)
        for label, component, run in parsed_entries
        if run is not None
    }
    if not preflight_only and set(new_runs) != {"P025", "A025"}:
        errors.append({"probe": "cross_probe", "error": "new probes must be exactly P025 and A025"})

    ordered_runs = [
        ("T025", *source_runs["T025"]) if "T025" in source_runs else None,
        ("P025", *new_runs["P025"]) if "P025" in new_runs else None,
        ("A025", *new_runs["A025"]) if "A025" in new_runs else None,
        ("M025", *source_runs["M025"]) if "M025" in source_runs else None,
    ]
    if preflight_only:
        ordered_runs = [ordered_runs[0], ordered_runs[3]]

    records: list[dict[str, Any]] = []
    for item in ordered_runs:
        if item is None:
            continue
        label, component, run = item
        try:
            record, probe_errors = _analyze_probe(
                label=label,
                component=component,
                run=run,
                review_dir=review_dir,
            )
            records.append(record)
            errors.extend({"probe": label, "error": error} for error in probe_errors)
        except Exception as exc:
            errors.append({"probe": label, "error": f"{type(exc).__name__}: {exc}"})

    comparison_review_image = None
    control_relative = None
    if not errors and not preflight_only:
        errors.extend(
            {"probe": "cross_probe", "error": error}
            for error in _cross_probe_contract(records, expected_checkpoint)
        )
    if not errors and not preflight_only:
        comparison_review_image = _comparison_review(records, review_dir)
        control_relative = _control_relative_metrics(records)

    decision = _decision(manual_verdict, errors, preflight_only=preflight_only)
    result = {
        "version": "v49",
        "objective": "decoder_mask_phase_amplitude_component_attribution",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "probe_budget": {"new_training_runs": 0, "new_sample_runs": 2, "numeric_sweeps": 0},
        "source_v48_verdict": source_v48_verdict.as_posix(),
        "source_v47_summary": source_v47_summary.as_posix(),
        "capacity_utilization": capacity,
        "probes": records,
        "control_relative_corr_mse": control_relative,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "review_planes": [PHYSICAL_PLANE],
        "review_class_labels": list(REQUIRED_CLASSES),
        "comparison_review_image": comparison_review_image,
        "manual_verdict_options": VERDICT_TO_STATUS,
        "human_readability_required": not preflight_only,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane human review can satisfy this gate.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v48-verdict", type=Path, required=True)
    parser.add_argument("--source-v47-summary", type=Path, required=True)
    parser.add_argument("--probe", dest="probe_entries", action="append", default=[])
    parser.add_argument("--manual-verdict", choices=sorted(VERDICT_TO_STATUS))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_v49_decoder_mask_component_attribution(
        source_v48_verdict=args.source_v48_verdict,
        source_v47_summary=args.source_v47_summary,
        probe_entries=args.probe_entries,
        manual_verdict=args.manual_verdict,
        preflight_only=args.preflight_only,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output or "")}, sort_keys=True))
    return 0 if result["status"] != "harness-invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
