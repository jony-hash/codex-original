"""Run the v53 CUDA-only physical-forward preflight.

The preflight constructs a fresh student model and executes one no-grad forward.
It does not load a checkpoint, train parameters, or run evaluation.  A failure is
an invalid physical harness and must stop the v53 training command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.one_shot import OneShotOpticalGenerator
from ldm_project.one_shot_eval import build_strict_optical_field_artifact


EXPECTED_SEGMENTS = [0.350, 0.370, 0.025, 0.025, 0.125, 0.475]
EXPECTED_FIELDS = {
    "slm_output",
    "lightgen_input",
    "encoder_input",
    "encoder_output",
    "ols_input",
    "ols_output",
    "decoder_input",
    "chip",
    "camera_field",
}


def _sampling_matches(record: dict[str, Any], shape: tuple[int, int], pixel: float, fov: float) -> bool:
    sampling = record.get("sampling")
    if not isinstance(sampling, dict):
        return False
    return (
        sampling.get("shape") == list(shape)
        and abs(float(sampling.get("pixel_size_x", 0.0)) - pixel) <= 1.0e-12
        and abs(float(sampling.get("pixel_size_y", 0.0)) - pixel) <= 1.0e-12
        and abs(float(sampling.get("fov_x", 0.0)) - fov) <= 1.0e-12
        and abs(float(sampling.get("fov_y", 0.0)) - fov) <= 1.0e-12
    )


def run_preflight(config_path: str | Path, *, device_name: str = "cuda") -> dict[str, Any]:
    if device_name != "cuda":
        return {
            "status": "physical-forward-model-invalid",
            "reason": "v53 CUDA preflight requires --device cuda",
            "diagnostic_only": True,
            "accepted_summary_written": False,
        }
    if not torch.cuda.is_available():
        return {
            "status": "physical-forward-model-invalid",
            "reason": "CUDA is unavailable",
            "diagnostic_only": True,
            "accepted_summary_written": False,
        }

    config = load_training_config(config_path)
    physics = config.physics
    if physics is None:
        return {
            "status": "physical-forward-model-invalid",
            "reason": "v53 config has no physics section",
            "diagnostic_only": True,
            "accepted_summary_written": False,
        }
    try:
        validate_physical_forward_contract(config, physics)
    except ValueError as exc:
        return {
            "status": "physical-forward-model-invalid",
            "reason": str(exc),
            "diagnostic_only": True,
            "accepted_summary_written": False,
        }

    device = torch.device("cuda")
    model = OneShotOpticalGenerator(physics, config.one_shot, image_size=config.first_stage.image_size).to(device)
    model.eval()
    code = torch.zeros(1, config.one_shot.code_dim, device=device)
    labels = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(code, labels, return_optical_fields=True)

    fields = outputs.get("optical_fields")
    if not isinstance(fields, dict):
        return {
            "status": "physical-forward-model-invalid",
            "reason": "model did not return optical_fields",
            "diagnostic_only": True,
            "accepted_summary_written": False,
        }
    artifact = build_strict_optical_field_artifact(
        fields,
        front_path_model=physics.front_path_model,
        target_code_active=bool(outputs.get("target_code_diagnostic_active", False)),
        bridge_active=bool(outputs.get("target_code_bridge_active", False)),
    )
    reasons: list[str] = []
    missing = sorted(EXPECTED_FIELDS - set(fields))
    if missing:
        reasons.append(f"missing optical fields: {missing}")
    for name, record in fields.items():
        real, imag = record.get("real"), record.get("imag")
        if not torch.is_tensor(real) or not torch.is_tensor(imag):
            reasons.append(f"{name} is missing real/imag tensors")
            continue
        if not torch.isfinite(real).all() or not torch.isfinite(imag).all():
            reasons.append(f"{name} contains non-finite values")
        if float((real.square() + imag.square()).sum()) <= 0.0:
            reasons.append(f"{name} has zero optical energy")
        if record.get("operation") == "generic_resize":
            reasons.append(f"{name} uses forbidden generic_resize")

    if "ols_input" in fields and "ols_output" in fields and fields.get("ols_output", {}).get("real") is not None:
        if not torch.equal(fields["ols_input"]["real"], fields["ols_output"]["real"]):
            reasons.append("OLS real field is not identity")
        if not torch.equal(fields["ols_input"]["imag"], fields["ols_output"]["imag"]):
            reasons.append("OLS imag field is not identity")
    if "slm_output" in fields and not _sampling_matches(fields["slm_output"], (200, 200), 2.5e-6, 500e-6):
        reasons.append("SLM sampling contract mismatch")
    if "encoder_input" in fields and not _sampling_matches(fields["encoder_input"], (100, 100), 2.0e-6, 200e-6):
        reasons.append("LightGen sampling contract mismatch")
    if "camera_field" in fields and not _sampling_matches(fields["camera_field"], (200, 200), 2.5e-6, 500e-6):
        reasons.append("CMOS sampling contract mismatch")
    trace = outputs.get("front_path_diagnostics", {})
    if trace.get("segments_m") != EXPECTED_SEGMENTS:
        reasons.append("macro propagation segment trace mismatch")
    if artifact["status"] != "manual-review-required" or artifact["field_chain_status"] != "valid":
        reasons.append(f"strict provenance status is {artifact['status']}")

    status = "physical-forward-model-valid" if not reasons else "physical-forward-model-invalid"
    return {
        "status": status,
        "manual_review_required": True,
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "device": str(device),
        "config": str(config_path),
        "checks": {
            "fields": sorted(fields),
            "strict_artifact": artifact,
            "front_path_diagnostics": trace,
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
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_preflight(args.config, device_name=args.device)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if report["status"] != "physical-forward-model-valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
