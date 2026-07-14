#!/usr/bin/env python3
"""CUDA-only v56 preflight for the coherent-imaging relay contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.one_shot import OneShotOpticalGenerator
from ldm_project.one_shot_eval import build_strict_optical_field_artifact


EXPECTED_SEGMENTS = [0.350, 0.370, 0.025, 0.025, 0.125, 0.07321428571428572, 0.3125, 0.0892857142857143]


def _invalid(reason: str) -> dict[str, Any]:
    return {
        "status": "physical-forward-model-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "reasons": [reason],
    }


def _sampling_matches(record: dict[str, Any], shape: tuple[int, int], pixel: float, fov: float) -> bool:
    sampling = record.get("sampling")
    if not isinstance(sampling, dict):
        return False
    return (
        sampling.get("shape") == list(shape)
        and abs(float(sampling.get("pixel_size_x", 0.0)) - pixel) <= 1e-12
        and abs(float(sampling.get("pixel_size_y", 0.0)) - pixel) <= 1e-12
        and abs(float(sampling.get("fov_x", 0.0)) - fov) <= 1e-12
        and abs(float(sampling.get("fov_y", 0.0)) - fov) <= 1e-12
    )


def run_preflight(
    config_path: str | Path,
    reference_report_path: str | Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    try:
        reference = json.loads(Path(reference_report_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return _invalid(f"cannot read v58.2 relay reference report: {exc}")
    if reference.get("status") != "physical-forward-model-valid" or reference.get("validation_revision") != "v58.2-layered-b0-relay-validation":
        return _invalid("v58.2 relay reference report is not independently physical-forward-model-valid")
    if reference.get("accepted_summary_written") is not False or reference.get("diagnostic_only") is not True:
        return _invalid("v58.2 relay reference report has invalid acceptance metadata")
    if device_name != "cuda":
        return _invalid("v56 preflight requires --device cuda")
    if not torch.cuda.is_available():
        return _invalid("CUDA is unavailable")

    config = load_training_config(config_path)
    reasons: list[str] = []
    try:
        validate_physical_forward_contract(config, config.physics)
    except ValueError as exc:
        return _invalid(str(exc))
    if config.physics.front_path_model != "coherent_imaging_4f":
        reasons.append("v56 requires coherent_imaging_4f front path")
    if config.physics.back_path_model != "free_space":
        reasons.append("v56 requires free_space back path")
    if config.one_shot.target_code_diagnostic_mode != "none" or config.one_shot.target_code_bridge_source != "none":
        reasons.append("target-code intervention is enabled")
    if config.one_shot.decoder_diagnostic_modulation_override != "none":
        reasons.append("decoder modulation override is enabled")
    if config.one_shot.ols_transfer_calibration != "none" or config.one_shot.ols_kind != "identity":
        reasons.append("OLS is not uncalibrated identity")
    if reasons:
        return _invalid("; ".join(reasons))

    device = torch.device("cuda")
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(device).eval()
    with torch.no_grad():
        output = model(
            torch.zeros(1, config.one_shot.code_dim, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            return_optical_fields=True,
        )
    fields = output.get("optical_fields")
    if not isinstance(fields, dict):
        return _invalid("model did not return strict optical fields")
    artifact = build_strict_optical_field_artifact(
        fields,
        front_path_model=config.physics.front_path_model,
        target_code_active=False,
        bridge_active=False,
    )
    if artifact.get("field_chain_status") != "valid":
        reasons.append(f"strict provenance invalid: {artifact.get('reasons') or artifact.get('errors')}")
    if "pre_encoder_relay_image" not in fields or "lightgen_input" in fields:
        reasons.append("front provenance must use pre_encoder_relay_image and never lightgen_input")
    for name, record in fields.items():
        real, imag = record.get("real"), record.get("imag")
        if not torch.is_tensor(real) or not torch.is_tensor(imag):
            reasons.append(f"{name} lacks real/imag tensors")
            continue
        if real.shape != imag.shape or not torch.isfinite(real).all() or not torch.isfinite(imag).all():
            reasons.append(f"{name} has invalid complex field shape or values")
        if float((real.square() + imag.square()).sum()) <= 0.0:
            reasons.append(f"{name} has zero optical energy")
        if record.get("operation") in {"generic_resize", "sample_center_aperture", "physical_aperture_sampling"}:
            reasons.append(f"{name} uses forbidden operation {record.get('operation')}")
    if not _sampling_matches(fields.get("slm_output", {}), (200, 200), 2.5e-6, 500e-6):
        reasons.append("SLM sampling contract mismatch")
    if not _sampling_matches(fields.get("encoder_input", {}), (100, 100), 2.0e-6, 200e-6):
        reasons.append("encoder sampling contract mismatch")
    if not _sampling_matches(fields.get("camera_field", {}), (200, 200), 2.5e-6, 500e-6):
        reasons.append("CMOS sampling contract mismatch")
    if not torch.equal(fields["ols_input"]["real"], fields["ols_output"]["real"]) or not torch.equal(fields["ols_input"]["imag"], fields["ols_output"]["imag"]):
        reasons.append("OLS field is not elementwise identity")
    front = output.get("front_path_diagnostics", {})
    if front.get("segments_m") != EXPECTED_SEGMENTS:
        reasons.append(f"front relay trace mismatch: {front.get('segments_m')}")
    relay = front.get("front_relay", {})
    if abs(float(relay.get("A", 0.0)) + 0.4) > 1e-8 or abs(float(relay.get("B", 1.0))) > 1e-8:
        reasons.append(f"front relay ABCD mismatch: {relay}")
    status = "physical-forward-model-valid" if not reasons else "physical-forward-model-invalid"
    return {
        "status": status,
        "manual_review_required": True,
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "config": str(config_path),
        "reference_report": str(reference_report_path),
        "checks": {
            "strict_artifact": artifact,
            "fields": sorted(fields),
            "front_path_diagnostics": front,
            "d2nn": {
                "grid_size": config.one_shot.d2nn.grid_size,
                "pixel_size": config.one_shot.d2nn.pixel_size,
                "encoder_layers": config.one_shot.d2nn.encoder_layers,
                "decoder_layers": config.one_shot.d2nn.decoder_layers,
                "layer_spacing": config.one_shot.d2nn.layer_spacing,
            },
        },
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_preflight(args.config, args.reference_report, device_name=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, default=str))
    raise SystemExit(0 if report["status"] == "physical-forward-model-valid" else 1)


if __name__ == "__main__":
    main()
