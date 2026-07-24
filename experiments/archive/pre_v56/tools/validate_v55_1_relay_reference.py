#!/usr/bin/env python3
"""Independent v55.1 coherent-relay reference validation.

This is a read-only forward check.  It never loads a checkpoint, trains, or
starts an eval/sample stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.optics import angular_spectrum_propagate, coherent_imaging_4f_field
from ldm_project.relay_reference import (
    collins_b0_relay_reference,
    direct_rayleigh_sommerfeld,
    four_f_abcd,
    four_f_direct_rayleigh_sommerfeld_reference,
    four_f_scaled_fresnel_reference,
)


def _align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    phase = torch.angle((a.conj() * b).sum())
    return a * torch.exp(1j * phase).to(a.dtype)


def _relay_args(device: torch.device, *, input_size: int = 32, output_size: int = 16) -> dict[str, object]:
    f1 = 0.22321428571428573
    f2 = 0.0892857142857143
    abcd = four_f_abcd(f1, f2)
    return {
        "magnification": abcd[0][0],
        "output_size": (output_size, output_size),
        "input_pixel_size": 500e-6 / input_size,
        "output_pixel_size": 200e-6 / output_size,
        "wavelength": 532e-9,
        "c_abcd": abcd[1][0],
        "pupil_diameter": 0.02375,
        "pupil_focal_length": f2,
    }


def _relay_case_metrics(field: torch.Tensor, args: dict[str, object]) -> dict[str, float]:
    production = coherent_imaging_4f_field(field, **args)
    reference = collins_b0_relay_reference(field, **args)
    scaled = four_f_scaled_fresnel_reference(
        field,
        f1=0.22321428571428573,
        f2=0.0892857142857143,
        wavelength=532e-9,
        input_pixel_size=float(args["input_pixel_size"]),
        output_size=tuple(args["output_size"]),
        output_pixel_size=float(args["output_pixel_size"]),
        pupil_diameter=float(args["pupil_diameter"]),
    )
    aligned = _align(production, reference)
    aligned_scaled = _align(production, scaled)
    relative_l2 = float((aligned - reference).norm() / reference.norm().clamp_min(1e-12))
    valid = reference.abs().square() >= reference.abs().square().amax() * 1.0e-6
    phase_error = torch.angle(aligned[valid] * reference[valid].conj()).abs().mean()
    energy_ratio = float(production.abs().square().sum() / reference.abs().square().sum().clamp_min(1e-12))
    return {
        "complex_relative_l2": relative_l2,
        "wrapped_phase_mae_rad": float(phase_error),
        "energy_ratio": energy_ratio,
        "scaled_fresnel_complex_relative_l2": float((aligned_scaled - scaled).norm() / scaled.norm().clamp_min(1e-12)),
        "scaled_fresnel_energy_ratio": float(production.abs().square().sum() / scaled.abs().square().sum().clamp_min(1e-12)),
    }


def _rs_aperture_check(device: torch.device) -> dict[str, float]:
    size = 96
    pixel = 1e-6
    wavelength = 532e-9
    distance = 200e-6
    coord = (torch.arange(size, device=device, dtype=torch.float64) - size // 2) * pixel
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    aperture = ((xx.square() + yy.square()) <= (10e-6) ** 2).to(torch.complex128)[None, None]
    asm = angular_spectrum_propagate(aperture, distance, wavelength, pixel)[0, 0]
    rs = direct_rayleigh_sommerfeld(aperture, distance=distance, wavelength=wavelength, input_pixel_size=pixel)[0, 0]
    start, stop = size // 2 - 12, size // 2 + 12
    asm_i = asm[start:stop, start:stop].abs().square()
    rs_i = rs[start:stop, start:stop].abs().square()
    asm_i = asm_i / asm_i.amax().clamp_min(1e-12)
    rs_i = rs_i / rs_i.amax().clamp_min(1e-12)
    relative_l2 = float((asm_i - rs_i).norm() / rs_i.norm().clamp_min(1e-12))
    return {"aperture_central_intensity_relative_l2": relative_l2}


def _direct_4f_check(device: torch.device) -> dict[str, float]:
    size = 24
    input_pixel = 500e-6 / size
    output_pixel = 200e-6 / size
    coord = (torch.arange(size, device=device, dtype=torch.float64) - (size - 1) / 2.0) * input_pixel
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    field = torch.exp(-(xx.square() + yy.square()) / (120e-6**2)).to(torch.complex128)[None, None]
    args = _relay_args(device)
    args.update(output_size=(size, size), input_pixel_size=input_pixel, output_pixel_size=output_pixel)
    production = coherent_imaging_4f_field(field, **args)
    reference = four_f_direct_rayleigh_sommerfeld_reference(
        field, f1=0.22321428571428573, f2=0.0892857142857143, wavelength=532e-9,
        input_pixel_size=input_pixel, output_size=(size, size), output_pixel_size=output_pixel,
        pupil_diameter=0.02375,
    )
    aligned = _align(production, reference)
    return {"complex_relative_l2": float((aligned - reference).norm() / reference.norm().clamp_min(1e-12))}


def run(config_path: str | Path, device: str = "cpu") -> dict[str, object]:
    config = load_training_config(config_path)
    reasons: list[str] = []
    structural: dict[str, object]
    try:
        validate_physical_forward_contract(config, config.physics)
        macro = config.physics.macro_optics
        matrix = four_f_abcd(macro.front_lens1_focal_length, macro.front_lens2_focal_length)
        if abs(matrix[0][0] + 0.4) > 1e-10 or abs(matrix[0][1]) > 1e-10:
            reasons.append(f"reference ABCD mismatch: {matrix}")
        if abs(2.0 * (macro.front_lens1_focal_length + macro.front_lens2_focal_length) - 0.625) > 1e-9:
            reasons.append("front mechanical length is not 625 mm")
        structural = {
            "status": "valid",
            "ols": {"kind": config.one_shot.ols_kind, "linear": config.one_shot.ols_kind == "identity"},
            "sampling": {"slm": {"shape": [200, 200], "pixel_size_m": 2.5e-6, "fov_m": 500e-6},
                         "encoder": {"shape": [100, 100], "pixel_size_m": 2.0e-6, "fov_m": 200e-6}},
            "front_abcd": {"A": matrix[0][0], "B": matrix[0][1]},
        }
    except Exception as exc:
        structural = {"status": "invalid", "error": str(exc)}
        reasons.append(str(exc))
    reference_metrics: dict[str, object] = {}
    try:
        target_device = torch.device(device)
        args = _relay_args(target_device)
        size = 32
        yy, xx = torch.meshgrid(
            (torch.arange(size, device=target_device, dtype=torch.float32) - (size - 1) / 2) * float(args["input_pixel_size"]),
            (torch.arange(size, device=target_device, dtype=torch.float32) - (size - 1) / 2) * float(args["input_pixel_size"]),
            indexing="ij",
        )
        cases = {
            "plane_wave": torch.ones(1, 1, size, size, device=target_device, dtype=torch.complex64),
            "tilted_plane_wave": torch.exp(1j * 2.0 * torch.pi * 25_000.0 * xx).to(torch.complex64)[None, None],
            "gaussian_beam": torch.exp(-(xx.square() + yy.square()) / (30e-6) ** 2).to(torch.complex64)[None, None],
            "circular_aperture": (((xx.square() + yy.square()) <= (100e-6) ** 2).to(torch.complex64))[None, None],
            "checkerboard": ((((torch.arange(size, device=target_device)[:, None] + torch.arange(size, device=target_device)[None, :]) % 2) == 0).to(torch.complex64))[None, None],
        }
        reference_metrics["relay_cases"] = {name: _relay_case_metrics(field, args) for name, field in cases.items()}
        reference_metrics["direct_rs"] = _rs_aperture_check(target_device)
        reference_metrics["direct_rs_full_4f"] = _direct_4f_check(target_device)
        worst_l2 = max(item["complex_relative_l2"] for item in reference_metrics["relay_cases"].values())
        worst_phase = max(item["wrapped_phase_mae_rad"] for item in reference_metrics["relay_cases"].values())
        energy_errors = [abs(item["energy_ratio"] - 1.0) for item in reference_metrics["relay_cases"].values()]
        scaled_l2 = max(item["scaled_fresnel_complex_relative_l2"] for item in reference_metrics["relay_cases"].values())
        scaled_energy_errors = [abs(item["scaled_fresnel_energy_ratio"] - 1.0) for item in reference_metrics["relay_cases"].values()]
        if worst_l2 > 1e-2:
            reasons.append(f"relay complex relative L2 exceeds 1e-2: {worst_l2}")
        if worst_phase > 0.02:
            reasons.append(f"relay wrapped phase MAE exceeds 0.02 rad: {worst_phase}")
        if max(energy_errors) > 0.02:
            reasons.append(f"relay energy ratio error exceeds 2%: {max(energy_errors)}")
        if scaled_l2 > 1e-2:
            reasons.append(f"independent scaled-Fresnel complex relative L2 exceeds 1e-2: {scaled_l2}")
        if max(scaled_energy_errors) > 0.02:
            reasons.append(f"independent scaled-Fresnel energy ratio error exceeds 2%: {max(scaled_energy_errors)}")
        if reference_metrics["direct_rs"]["aperture_central_intensity_relative_l2"] > 0.01:
            reasons.append("direct RS aperture comparison exceeds 1%")
        if reference_metrics["direct_rs_full_4f"]["complex_relative_l2"] > 1e-2:
            reasons.append("direct RS full 4f comparison exceeds 1e-2")
    except Exception as exc:
        reference_metrics["status"] = "invalid"
        reference_metrics["error"] = str(exc)
        reasons.append(f"reference forward failed: {exc}")
    hardware = {
        "status": "assumption-only",
        "front_clear_aperture_m": 0.02375,
        "back_clear_aperture_m": 0.0266,
        "note": "pupil/NA values are design lower bounds, not measured hardware specifications",
    }
    structural_status = structural.get("status") == "valid"
    reference_harness_valid = bool(
        reference_metrics.get("direct_rs", {}).get("aperture_central_intensity_relative_l2", float("inf")) <= 0.01
        and reference_metrics.get("direct_rs_full_4f", {}).get("complex_relative_l2", float("inf")) <= 0.01
        and reference_metrics.get("status", "valid") != "invalid"
    )
    if not reference_harness_valid:
        status = "reference-harness-invalid"
    elif reasons:
        status = "production-relay-invalid"
    elif not structural_status:
        status = "physical-forward-model-invalid"
    else:
        status = "physical-forward-model-valid"
    report = {
        "status": status,
        "structural_preflight": structural,
        "reference_propagation_validation": reference_metrics,
        "reference_harness_valid": reference_harness_valid,
        "hardware_assumptions": hardware,
        "reasons": reasons,
        "diagnostic_only": True,
        "accepted_summary_written": False,
    }
    return report


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
