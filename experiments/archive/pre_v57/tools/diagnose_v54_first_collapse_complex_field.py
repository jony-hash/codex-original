#!/usr/bin/env python3
"""Locate the first plane where class-dependent complex-field diversity collapses.

This is a read-only diagnostic for an existing strict optical-field bundle.  It
never builds an all-pairs field tensor: each pair is reduced immediately and
the report is flushed after every plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


PLANES = (
    "slm_output", "lightgen_input", "encoder_input", "encoder_grid",
    "encoder_layer_1", "encoder_layer_2", "encoder_output", "ols_input",
    "ols_output", "decoder_input", "decoder_layer_1", "decoder_layer_2",
    "decoder_output", "chip", "camera_field",
)
REQUIRED_SAMPLING = ("shape", "pixel_size_x", "pixel_size_y", "fov_x", "fov_y")
FLOOR = 1.0e-6  # -60 dB intensity threshold


def _complex(record: dict[str, Any]) -> torch.Tensor:
    real, imag = record.get("real"), record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag):
        raise ValueError("each plane must contain real and imag tensors")
    if real.shape != imag.shape:
        raise ValueError(f"real/imag shape mismatch: {tuple(real.shape)} vs {tuple(imag.shape)}")
    if not torch.isfinite(real).all() or not torch.isfinite(imag).all():
        raise ValueError("non-finite real/imag field")
    return torch.complex(real, imag)


def _labels(bundle: dict[str, Any]) -> torch.Tensor:
    labels = bundle.get("class_labels")
    if not torch.is_tensor(labels):
        raise ValueError("sample tensor bundle is missing class_labels")
    labels = labels.detach().cpu().long().flatten()
    counts = {int(k): int((labels == k).sum()) for k in range(10)}
    if counts != {k: 2 for k in range(10)}:
        raise ValueError(f"expected classes 0-9 exactly twice, got {counts}")
    return labels


def _validate(
    field_bundle: dict[str, Any], provenance: dict[str, Any], tensor_bundle: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if provenance.get("field_chain_status") != "valid" or not provenance.get("full_chain_available"):
        errors.append("strict provenance is not a valid full chain")
    if provenance.get("accepted_summary_written") is not False:
        errors.append("accepted_summary_written must remain false")
    fields = field_bundle.get("fields")
    if not isinstance(fields, dict):
        return ["optical field bundle has no fields mapping"]
    missing = [name for name in PLANES if name not in fields]
    errors.extend(f"missing plane: {name}" for name in missing)
    for name, record in fields.items():
        if name not in PLANES:
            continue
        try:
            field = _complex(record)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if field.ndim < 3 or field.shape[0] != 20:
            errors.append(f"{name}: expected 20 samples, got {tuple(field.shape)}")
        sampling = record.get("sampling")
        if not isinstance(sampling, dict):
            errors.append(f"{name}: missing physical sampling")
        else:
            for key in REQUIRED_SAMPLING:
                if key not in sampling:
                    errors.append(f"{name}: missing sampling.{key}")
        if not bool(record.get("phase_available", False)):
            errors.append(f"{name}: phase is unavailable")
    try:
        labels = _labels(tensor_bundle)
        if labels.numel() != 20:
            errors.append("class_labels must contain 20 samples")
    except ValueError as exc:
        errors.append(str(exc))
    if "ols_input" in fields and "ols_output" in fields:
        a, b = _complex(fields["ols_input"]), _complex(fields["ols_output"])
        if not torch.equal(a, b):
            errors.append("OLS input/output are not exactly identical")
    return errors


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.reshape(-1), b.reshape(-1)
    ea, eb = a.abs().square().sum(), b.abs().square().sum()
    denom = max(float(((ea + eb) * 0.5).item()), 1.0e-20)
    inner = torch.sum(torch.conj(b) * a)
    theta = torch.angle(inner)
    aligned = b * torch.exp(1j * theta)
    complex_rel = float(((a - aligned).abs().square().sum() / denom).item())
    aa, bb = a.abs(), b.abs()
    ac, bc = aa - aa.mean(), bb - bb.mean()
    amp_corr = float((torch.sum(ac * bc) / torch.sqrt(torch.sum(ac.square()) * torch.sum(bc.square()) + 1e-20)).item())
    valid = (aa.square() >= aa.square().max() * FLOOR) & (bb.square() >= bb.square().max() * FLOOR)
    phase_mae = float(torch.abs(torch.angle(a[valid] * torch.conj(aligned[valid]))).mean().item()) if valid.any() else float("nan")
    return {"complex_relative_l2": complex_rel, "amplitude_corr": amp_corr, "phase_mae_rad": phase_mae}


def _summarize(field: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    field = field.detach().cpu()
    n = field.shape[0]
    energy = field.abs().square().flatten(1).sum(1)
    pair_values: dict[str, list[float]] = {"same": [], "cross": []}
    amp_values: dict[str, list[float]] = {"same": [], "cross": []}
    phase_values: dict[str, list[float]] = {"same": [], "cross": []}
    for i in range(n):
        for j in range(i + 1, n):
            kind = "same" if int(labels[i]) == int(labels[j]) else "cross"
            metrics = _pair_metrics(field[i], field[j])
            pair_values[kind].append(metrics["complex_relative_l2"])
            amp_values[kind].append(metrics["amplitude_corr"])
            phase_values[kind].append(metrics["phase_mae_rad"])

    def mean(values: Iterable[float]) -> float | None:
        values = [v for v in values if math.isfinite(v)]
        return sum(values) / len(values) if values else None

    same, cross = mean(pair_values["same"]), mean(pair_values["cross"])
    separation = None if same is None or cross is None else cross - same
    return {
        "sample_count": n,
        "power_mean": float(energy.mean().item()),
        "power_min": float(energy.min().item()),
        "power_max": float(energy.max().item()),
        "same_class": {"complex_relative_l2": same, "amplitude_corr": mean(amp_values["same"]), "phase_mae_rad": mean(phase_values["same"])},
        "cross_class": {"complex_relative_l2": cross, "amplitude_corr": mean(amp_values["cross"]), "phase_mae_rad": mean(phase_values["cross"])},
        "class_separation_signal": separation,
    }


def _energy_retention(fields: dict[str, Any], name: str) -> float | None:
    record = fields[name]
    upstream = record.get("upstream")
    if upstream not in fields:
        return None
    current = _complex(record)
    previous = _complex(fields[upstream])
    sampling = record.get("sampling") or {}
    previous_sampling = fields[upstream].get("sampling") or {}
    dxy = float(sampling.get("pixel_size_x", 1.0)) * float(sampling.get("pixel_size_y", 1.0))
    prev_dxy = float(previous_sampling.get("pixel_size_x", 1.0)) * float(previous_sampling.get("pixel_size_y", 1.0))
    return float((current.abs().square().flatten(1).sum(1).mean() * dxy / (previous.abs().square().flatten(1).sum(1).mean() * prev_dxy + 1e-30)).item())


def _decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r.get("class_separation_signal") for r in results]
    finite = [float(x) if x is not None and math.isfinite(float(x)) else 0.0 for x in scores]
    baseline = finite[0] if finite else 0.0
    if baseline <= 1.0e-8:
        return {"status": "no-localized-collapse", "reason": "class separation is absent at SLM output"}
    for i in range(1, len(finite)):
        if finite[i] <= 0.1 * baseline and max(finite[:i]) >= 1.0e-6:
            name = results[i]["plane"]
            if name in {"lightgen_input", "encoder_input", "encoder_grid"}:
                status = "macro-front-aperture-first-collapse"
            elif name.startswith("encoder_"):
                status = "encoder-d2nn-first-collapse"
            elif name in {"ols_input", "ols_output", "decoder_input"}:
                status = "ols-harness-invalid" if name == "ols_output" else "no-localized-collapse"
            elif name.startswith("decoder_") or name == "chip":
                status = "decoder-d2nn-first-collapse"
            else:
                status = "readout-intensity-phase-loss-collapse"
            return {"status": status, "first_candidate_plane": name, "baseline_signal": baseline, "candidate_signal": finite[i]}
    return {"status": "no-localized-collapse", "reason": "no clear 90% signal drop"}


def run_diagnostic(field_path: Path, provenance_path: Path, tensor_path: Path, output: Path, progress: Path) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    field_bundle = torch.load(field_path, map_location="cpu", weights_only=False)
    tensor_bundle = torch.load(tensor_path, map_location="cpu", weights_only=False)
    errors = _validate(field_bundle, provenance, tensor_bundle)
    report: dict[str, Any] = {"status": "manual-review-required", "diagnostic_only": True, "accepted_summary_written": False, "field_chain_status": "valid", "planes": [], "errors": errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report["status"] = "harness-invalid"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        progress.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    fields = field_bundle["fields"]
    labels = _labels(tensor_bundle)
    for name in PLANES:
        item = {"plane": name, "sampling": fields[name].get("sampling"), "operation": fields[name].get("operation"), "energy_retention_from_upstream": _energy_retention(fields, name)}
        item.update(_summarize(_complex(fields[name]), labels))
        report["planes"].append(item)
        report["progress"] = {"completed_plane": name, "completed_count": len(report["planes"]), "total": len(PLANES)}
        progress.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["decision"] = _decision(report["planes"])
    report["progress"]["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("plane", "power_mean", "energy_retention_from_upstream", "same_complex", "cross_complex", "class_separation_signal"))
        writer.writeheader()
        for item in report["planes"]:
            writer.writerow({"plane": item["plane"], "power_mean": item["power_mean"], "energy_retention_from_upstream": item["energy_retention_from_upstream"], "same_complex": item["same_class"]["complex_relative_l2"], "cross_complex": item["cross_class"]["complex_relative_l2"], "class_separation_signal": item["class_separation_signal"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, default=None)
    args = parser.parse_args()
    progress = args.progress or args.output.with_name(args.output.stem + "_progress.json")
    report = run_diagnostic(args.fields, args.provenance, args.tensors, args.output, progress)
    print(json.dumps({"status": report["status"], "decision": report.get("decision"), "output": str(args.output), "progress": str(progress)}, sort_keys=True))
    raise SystemExit(0 if report["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
