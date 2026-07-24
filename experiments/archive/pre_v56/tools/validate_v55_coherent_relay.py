#!/usr/bin/env python3
"""CUDA-free structural preflight for the v55 coherent relay contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.one_shot_model import OneShotOpticalGenerator
from ldm_project.one_shot_eval import build_strict_optical_field_artifact


def run(config_path: str | Path, device: str = "cpu") -> dict[str, object]:
    config = load_training_config(config_path)
    reasons: list[str] = []
    try:
        validate_physical_forward_contract(config, config.physics)
    except ValueError as exc:
        reasons.append(str(exc))
    if config.physics is None:
        reasons.append("missing physics")
        return {"status": "physical-forward-model-invalid", "reasons": reasons, "diagnostic_only": True, "accepted_summary_written": False}
    try:
        model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(device).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, config.one_shot.code_dim, device=device), torch.zeros(1, dtype=torch.long, device=device), return_optical_fields=True)
        fields = output.get("optical_fields", {})
        artifact = build_strict_optical_field_artifact(
            fields,
            front_path_model=config.physics.front_path_model,
            target_code_active=False,
            bridge_active=False,
        )
        if artifact.get("field_chain_status") != "valid":
            reasons.append(f"strict optical provenance invalid: {artifact.get('reasons') or artifact.get('errors')}")
        if "pre_encoder_relay_image" not in fields or "lightgen_input" in fields:
            reasons.append("front provenance must expose pre_encoder_relay_image and no lightgen_input")
        record = fields.get("encoder_input", {})
        sampling = record.get("sampling", {})
        if tuple(record.get("shape", ())) != (1, 1, 100, 100):
            reasons.append(f"encoder input shape is {record.get('shape')}, expected [1,1,100,100]")
        if abs(float(sampling.get("fov_x", 0.0)) - 200e-6) > 1e-12 or abs(float(sampling.get("pixel_size_x", 0.0)) - 2e-6) > 1e-12:
            reasons.append("encoder sampling is not 100x100 @ 2 um")
        relay = output.get("front_path_diagnostics", {}).get("front_relay", {})
        if abs(float(relay.get("A", 0.0)) + 0.4) > 1e-8 or abs(float(relay.get("B", 1.0))) > 1e-8:
            reasons.append(f"front ABCD is not A=-0.4,B=0: {relay}")
        if not fields.get("pre_encoder_relay_image", {}).get("relay_trace"):
            reasons.append("front relay lens/pupil trace missing from optical-field metadata")
        components = fields.get("pre_encoder_relay_image", {}).get("relay_trace", {}).get("components") or []
        expected = [
            ("SLM", "BS", 0.025),
            ("BS", "M3", 0.125),
            ("M3", "L1", 0.07321428571428572),
            ("L1", "L2", 0.3125),
            ("L2", "Encoder", 0.0892857142857143),
        ]
        actual = [(item.get("from"), item.get("to"), float(item.get("distance_m", -1.0))) for item in components]
        if len(actual) != len(expected) or any(
            a[0] != e[0] or a[1] != e[1] or abs(a[2] - e[2]) > 1e-9 for a, e in zip(actual, expected)
        ):
            reasons.append(f"front relay component trace mismatch: {actual}")
    except Exception as exc:  # structural preflight must report invalidity, not hide it
        reasons.append(f"forward failed: {exc}")
    return {"status": "physical-forward-model-valid" if not reasons else "physical-forward-model-invalid", "reasons": reasons, "diagnostic_only": True, "accepted_summary_written": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "physical-forward-model-valid" else 1)


if __name__ == "__main__":
    main()
