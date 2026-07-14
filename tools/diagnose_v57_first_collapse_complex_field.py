#!/usr/bin/env python3
"""Read-only v57 first-collapse analysis for a class-balanced strict field sample."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import torch


PLANE_ORDER = (
    "slm_output",
    "pre_encoder_relay_image",
    "encoder_input",
    "encoder_layer_1",
    "encoder_layer_2",
    "encoder_output",
    "ols_input",
    "ols_output",
    "decoder_input",
    "decoder_layer_1",
    "decoder_layer_2",
    "decoder_output",
    "chip",
    "camera_field",
)
METRICS = ("amplitude", "complex", "intensity")


def _pair_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    # Preserve the propagated complex field.  Calling ``double()`` on a
    # complex tensor silently discards its imaginary component, which would
    # turn the complex/phase diagnostics into amplitude-only diagnostics.
    a = a.detach().to(torch.complex128)
    b = b.detach().to(torch.complex128)
    a_amp, b_amp = a.abs(), b.abs()
    amp_denom = (a_amp.norm() + b_amp.norm()).mul(0.5).clamp_min(1e-12)
    amp_distance = float((a_amp - b_amp).norm() / amp_denom)
    phase = torch.angle((a.conj() * b).sum())
    aligned = a * torch.exp(1j * phase).to(a.dtype)
    complex_denom = (a.norm() + b.norm()).mul(0.5).clamp_min(1e-12)
    complex_distance = float((aligned - b).norm() / complex_denom)
    a_intensity, b_intensity = a_amp.square(), b_amp.square()
    intensity_denom = (a_intensity.norm() + b_intensity.norm()).mul(0.5).clamp_min(1e-12)
    intensity_distance = float((a_intensity - b_intensity).norm() / intensity_denom)
    valid = (a_intensity >= a_intensity.amax() * 1e-6) & (b_intensity >= b_intensity.amax() * 1e-6)
    phase_distance = float(torch.angle(aligned[valid] * b[valid].conj()).abs().mean()) if bool(valid.any()) else 0.0
    return {
        "amplitude": amp_distance,
        "complex": complex_distance,
        "intensity": intensity_distance,
        "phase_mae_rad": phase_distance,
    }


def _field_tensor(record: dict[str, Any]) -> torch.Tensor:
    real, imag = record.get("real"), record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag):
        raise ValueError("field record is missing real/imag tensors")
    if real.shape != imag.shape:
        raise ValueError(f"real/imag shape mismatch: {tuple(real.shape)} vs {tuple(imag.shape)}")
    if not torch.isfinite(real).all() or not torch.isfinite(imag).all():
        raise ValueError("field contains non-finite values")
    return torch.complex(real, imag)


def _summarize_pairs(fields: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    same: dict[str, list[float]] = {metric: [] for metric in (*METRICS, "phase_mae_rad")}
    cross: dict[str, list[float]] = {metric: [] for metric in (*METRICS, "phase_mae_rad")}
    for i in range(fields.shape[0]):
        for j in range(i + 1, fields.shape[0]):
            distance = _pair_distance(fields[i, 0], fields[j, 0])
            target = same if int(labels[i]) == int(labels[j]) else cross
            for metric, value in distance.items():
                target[metric].append(value)

    result: dict[str, Any] = {
        "sample_count": int(fields.shape[0]),
        "same_class_pair_count": len(same["complex"]),
        "cross_class_pair_count": len(cross["complex"]),
        "same_class_median": {metric: median(values) if values else None for metric, values in same.items()},
        "cross_class_median": {metric: median(values) if values else None for metric, values in cross.items()},
    }
    result["separation_signal"] = {
        metric: (
            result["cross_class_median"][metric] - result["same_class_median"][metric]
            if result["same_class_median"][metric] is not None and result["cross_class_median"][metric] is not None
            else None
        )
        for metric in (*METRICS, "phase_mae_rad")
    }
    return result


def _validate_inputs(
    fields_payload: dict[str, Any], provenance: dict[str, Any], labels: torch.Tensor, fields_path: Path,
) -> list[str]:
    reasons: list[str] = []
    if provenance.get("field_chain_status") != "valid":
        reasons.append(f"strict provenance is {provenance.get('field_chain_status')!r}")
    if provenance.get("accepted_summary_written") is not False or provenance.get("diagnostic_only") is not True:
        reasons.append("provenance acceptance metadata is invalid")
    if fields_payload.get("status") not in {"manual-review-required", "valid"}:
        reasons.append(f"optical field bundle status is {fields_payload.get('status')!r}")
    if not torch.is_tensor(labels) or labels.ndim != 1:
        reasons.append("class labels are missing or not one-dimensional")
        return reasons
    counts = {int(label): int((labels == label).sum()) for label in torch.unique(labels)}
    if set(counts) != set(range(10)) or any(counts.get(label, 0) != 2 for label in range(10)):
        reasons.append(f"expected exactly two samples for each class 0-9, got {counts}")
    fields = fields_payload.get("fields")
    if not isinstance(fields, dict):
        reasons.append("optical field bundle has no fields mapping")
        return reasons
    missing = [name for name in PLANE_ORDER if name not in fields]
    if missing:
        reasons.append(f"missing required planes: {missing}")
    if "lightgen_input" in fields:
        reasons.append("legacy lightgen_input plane is present")
    for name, record in fields.items():
        try:
            value = _field_tensor(record)
        except ValueError as exc:
            reasons.append(f"{name}: {exc}")
            continue
        if value.shape[0] != labels.shape[0]:
            reasons.append(f"{name}: sample count {value.shape[0]} differs from labels {labels.shape[0]}")
        if record.get("operation") in {"generic_resize", "sample_center_aperture", "physical_aperture_sampling"}:
            reasons.append(f"{name}: forbidden operation {record.get('operation')}")
    if not fields_path.exists():
        reasons.append("field bundle path does not exist")
    return reasons


def _load_d2nn_diagnostics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "energy_retention": data.get("energy_retention", {}),
        "crop_energy_retention": data.get("crop_energy_retention", {}),
    }


def _decision(plane_results: list[dict[str, Any]], d2nn: dict[str, Any] | None) -> dict[str, Any]:
    if not plane_results:
        return {"decision": "harness-invalid", "reason": "no plane results"}
    baseline = plane_results[0]["separation_signal"]
    if all(baseline.get(metric) is None or baseline.get(metric, 0.0) <= 0.0 for metric in METRICS):
        return {"decision": "upstream-generator-or-objective-collapse", "first_plane": "slm_output"}
    for item in plane_results[1:]:
        item["relative_separation_retention"] = {
            metric: (
                item["separation_signal"].get(metric) / baseline.get(metric)
                if baseline.get(metric) and item["separation_signal"].get(metric) is not None
                else None
            )
            for metric in METRICS
        }
    for index, item in enumerate(plane_results[1:], start=1):
        retention = item["relative_separation_retention"]
        collapsed = [value is not None and value < 0.10 for value in retention.values()]
        later = plane_results[index:]
        if sum(collapsed) >= 2 and all(
            sum(
                later_item["relative_separation_retention"].get(metric) is not None
                and later_item["relative_separation_retention"].get(metric) < 0.10
                for later_item in later
            ) >= 2
            for metric in METRICS
        ):
            if item["plane"] == "pre_encoder_relay_image":
                decision = "coherent-relay-information-collapse"
            elif item["plane"].startswith("encoder_"):
                decision = "encoder-d2nn-first-collapse"
            elif item["plane"].startswith("decoder_") or item["plane"] == "chip":
                decision = "decoder-d2nn-first-collapse"
            else:
                decision = "no-localized-collapse"
            return {"decision": decision, "first_plane": item["plane"], "retention": retention}
    chip = next((item for item in plane_results if item["plane"] == "chip"), None)
    camera = next((item for item in plane_results if item["plane"] == "camera_field"), None)
    if chip and camera and chip.get("relative_separation_retention", {}).get("complex", 0.0) >= 0.10 and camera.get("relative_separation_retention", {}).get("intensity", 0.0) < 0.10:
        return {"decision": "phase-to-intensity-readout-collapse", "first_plane": "camera_field"}
    if d2nn and d2nn.get("crop_energy_retention"):
        ratios = [float(item.get("ratio_mean", 1.0)) for item in d2nn["crop_energy_retention"].values()]
        if ratios and math.prod(ratios) < 0.01:
            return {"decision": "aperture-throughput-limited-no-localized-collapse", "cumulative_crop_retention": math.prod(ratios)}
    return {"decision": "no-localized-collapse"}


def run(
    fields_path: str | Path,
    provenance_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    d2nn_diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    fields_path, provenance_path, labels_path, output_path = map(Path, (fields_path, provenance_path, labels_path, output_path))
    fields_payload = torch.load(fields_path, map_location="cpu", weights_only=False)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    labels_payload = torch.load(labels_path, map_location="cpu", weights_only=False)
    labels = labels_payload["class_labels"] if isinstance(labels_payload, dict) else labels_payload
    reasons = _validate_inputs(fields_payload, provenance, labels, fields_path)
    report: dict[str, Any] = {
        "status": "manual-review-required" if not reasons else "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {"fields": str(fields_path), "provenance": str(provenance_path), "labels": str(labels_path)},
        "reasons": reasons,
        "planes": [],
    }
    if not reasons:
        fields = fields_payload["fields"]
        d2nn = _load_d2nn_diagnostics(Path(d2nn_diagnostics_path) if d2nn_diagnostics_path else None)
        for plane in PLANE_ORDER:
            result = {"plane": plane, **_summarize_pairs(_field_tensor(fields[plane]), labels)}
            energy = _field_tensor(fields[plane]).abs().square().flatten(1).sum(1)
            result["energy_mean"] = float(energy.mean())
            result["energy_cv"] = float(energy.std() / energy.mean().clamp_min(1e-12))
            report["planes"].append(result)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["decision"] = _decision(report["planes"], d2nn)
        report["d2nn_diagnostics"] = d2nn
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--d2nn-diagnostics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.fields, args.provenance, args.labels, args.output, args.d2nn_diagnostics)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
