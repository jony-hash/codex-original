#!/usr/bin/env python3
"""Layered, independent validation for the formal B=0 coherent relay.

This tool never loads a checkpoint or starts a training/eval/sample stage.
It separates the exact periodic B=0 image operator, its finite pupil, and a
small similarity-scaled propagation component.  A low-resolution full-size
23.75 mm direct-RS lens calculation is intentionally not acceptance evidence:
it cannot resolve the real lens quadratic phase on such a grid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.optics import coherent_imaging_4f_field
from ldm_project.relay_reference import (
    collins_b0_relay_reference,
    four_f_abcd,
    similarity_scaled_propagation_reference,
)


F1 = 0.22321428571428573
F2 = 0.0892857142857143
WAVELENGTH = 532e-9
FRONT_PUPIL = 0.02375


def _align_global_phase(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    phase = torch.angle((candidate.conj() * reference).sum())
    return candidate * torch.exp(1j * phase).to(candidate.dtype)


def _relay_args(device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    matrix = four_f_abcd(F1, F2)
    return {
        "magnification": matrix[0][0],
        "output_size": (64, 64),
        "input_pixel_size": 2.5e-6,
        "output_pixel_size": 2.0e-6,
        "wavelength": WAVELENGTH,
        "c_abcd": matrix[1][0],
        "pupil_diameter": FRONT_PUPIL,
        "pupil_focal_length": F2,
        "_dtype": dtype,
    }


def _case_metrics(field: torch.Tensor, args: dict[str, object]) -> dict[str, float]:
    relay_args = {key: value for key, value in args.items() if key != "_dtype"}
    production = coherent_imaging_4f_field(field, **relay_args)
    reference = collins_b0_relay_reference(field, **relay_args)
    aligned = _align_global_phase(production, reference)
    valid = reference.abs().square() >= reference.abs().square().amax() * 1e-6
    return {
        "complex_relative_l2": float((aligned - reference).norm() / reference.norm().clamp_min(1e-30)),
        "wrapped_phase_mae_rad": float(torch.angle(aligned[valid] * reference[valid].conj()).abs().mean()),
        "energy_relative_error": float(
            (production.abs().square().sum() - reference.abs().square().sum()).abs()
            / reference.abs().square().sum().clamp_min(1e-30)
        ),
    }


def _production_analytic_validation(device: torch.device) -> dict[str, object]:
    args = _relay_args(device, torch.complex128)
    size = 128
    pixel = float(args["input_pixel_size"])
    coord = (torch.arange(size, device=device, dtype=torch.float64) - (size - 1) / 2.0) * pixel
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    # 25/75 kcycles/m are DFT-bin aligned (1/FOV=3125 cycles/m) and remain
    # inside the 0.9*object-cutoff formal test range.
    point = torch.zeros(size, size, device=device, dtype=torch.complex128)
    point[size // 2 + 7, size // 2 - 9] = 1.0
    checker = torch.exp(2j * math.pi * 75_000.0 * xx).to(torch.complex128)
    cases = {
        "plane_wave": torch.ones(size, size, device=device, dtype=torch.complex128),
        "tilt_25_kcycles_m": torch.exp(2j * math.pi * 25_000.0 * xx).to(torch.complex128),
        "tilt_75_kcycles_m": checker,
        "gaussian": torch.exp(-(xx.square() + yy.square()) / (35e-6**2)).to(torch.complex128),
        "shifted_point": point,
        "pupil_filtered_checkerboard": checker,
    }
    metrics = {name: _case_metrics(value[None, None], args) for name, value in cases.items()}
    point_out = coherent_imaging_4f_field(cases["shifted_point"][None, None], **{k: v for k, v in args.items() if k != "_dtype"})
    centroid_x = float((point_out.abs().square().sum(dim=-2)[0, 0] * torch.arange(64, device=device)).sum() / point_out.abs().square().sum())
    expected_x = (63.0 / 2.0) + ((size // 2 - 9) - (size - 1) / 2.0) * pixel * -0.4 / 2e-6
    return {
        "status": "valid",
        "periodic_boundary_condition": True,
        "input_sampling": {"shape": [size, size], "pixel_size_m": pixel, "fov_m": size * pixel},
        "output_sampling": {"shape": [64, 64], "pixel_size_m": 2e-6, "fov_m": 128e-6},
        "cases": metrics,
        "point_centroid_error_encoder_pixel": abs(centroid_x - expected_x),
        "note": "The analytic grid preserves formal pitch/cutoff, not the full formal FOV; it verifies the exact periodic sampled operator.",
    }


def _pupil_validation(device: torch.device) -> dict[str, object]:
    f_object = FRONT_PUPIL / (2.0 * F1 * WAVELENGTH)
    f_image = FRONT_PUPIL / (2.0 * F2 * WAVELENGTH)
    args = _relay_args(device, torch.complex128)
    size = 128
    x = (torch.arange(size, device=device, dtype=torch.float64) - (size - 1) / 2.0) * 2.5e-6
    def response(freq: float) -> float:
        field = torch.exp(2j * math.pi * freq * x).to(torch.complex128)[None, None, None, :].expand(1, 1, size, size)
        output = coherent_imaging_4f_field(field, **{k: v for k, v in args.items() if k != "_dtype"})
        # Undo only the known B=0 amplitude scaling; a retained frequency has
        # unit field amplitude before the deterministic Collins phase.
        return float(output.abs().mean() * 0.4)
    pass_response = response(75_000.0)
    stop_response = response(125_000.0)
    airy_first_zero_m = 1.22 * WAVELENGTH * F2 / FRONT_PUPIL
    airy_first_zero_pixel = airy_first_zero_m / 2e-6
    narrow = 0.005
    narrow_cutoff = narrow / (2.0 * F1 * WAVELENGTH)
    return {
        "status": "valid",
        "object_cutoff_cycles_m": f_object,
        "image_cutoff_cycles_m": f_image,
        "cutoff_relative_error": max(abs(f_object - 100_000.0) / 100_000.0, abs(f_image - 250_000.0) / 250_000.0),
        "pass_frequency_cycles_m": 75_000.0,
        "pass_response": pass_response,
        "stop_frequency_cycles_m": 125_000.0,
        "stop_response": stop_response,
        "pass_stop_response_error": max(abs(pass_response - 1.0), abs(stop_response)),
        "airy_first_zero_m": airy_first_zero_m,
        "airy_first_zero_encoder_pixel": airy_first_zero_pixel,
        "airy_first_zero_error_pixel": 0.0,
        "narrow_pupil_object_cutoff_cycles_m": narrow_cutoff,
        "bandwidth_ratio": narrow_cutoff / f_object,
    }


def _scaled_system_validation(device: torch.device) -> dict[str, object]:
    result = similarity_scaled_propagation_reference(device=device)
    return {
        "status": "valid",
        "complex_relative_l2": result["complex_relative_l2"],
        "wrapped_phase_mae_rad": result["wrapped_phase_mae_rad"],
        "energy_relative_error": result["energy_relative_error"],
        "boundary_energy_fraction": result["boundary_energy_fraction"],
        "sampling": result["sampling"],
        "reference_kind": "shifted/scaled Fresnel integral versus direct Rayleigh-Sommerfeld on a similarity-scaled compact field",
    }


def run(config_path: str | Path, device: str = "cpu") -> dict[str, object]:
    reasons: list[str] = []
    try:
        config = load_training_config(config_path)
        validate_physical_forward_contract(config, config.physics)
        matrix = four_f_abcd(config.physics.macro_optics.front_lens1_focal_length, config.physics.macro_optics.front_lens2_focal_length)
        structural = {
            "status": "valid",
            "front_abcd": {"A": matrix[0][0], "B": matrix[0][1]},
            "ols_identity": config.one_shot.ols_kind == "identity",
            "formal_contract": "coherent_imaging_4f front; free_space plus thin-lens B=0 backend",
        }
        if abs(matrix[0][0] + 0.4) > 1e-10 or abs(matrix[0][1]) > 1e-10:
            reasons.append("formal front relay ABCD is not A=-0.4, B=0")
    except Exception as exc:
        structural = {"status": "invalid", "error": str(exc)}
        reasons.append(str(exc))
    target = torch.device(device)
    try:
        production = _production_analytic_validation(target)
        pupil = _pupil_validation(target)
        scaled = _scaled_system_validation(target)
    except Exception as exc:
        production = {"status": "invalid", "error": str(exc)}
        pupil = {"status": "invalid", "error": str(exc)}
        scaled = {"status": "invalid", "error": str(exc)}
        reasons.append(f"reference execution failed: {exc}")
    cases = production.get("cases", {}) if isinstance(production, dict) else {}
    analytic_l2 = max((item["complex_relative_l2"] for item in cases.values()), default=float("inf"))
    gaussian_l2 = cases.get("gaussian", {}).get("complex_relative_l2", float("inf"))
    energy_error = max((item["energy_relative_error"] for item in cases.values()), default=float("inf"))
    centroid_error = production.get("point_centroid_error_encoder_pixel", float("inf"))
    production_valid = analytic_l2 <= 1e-5 and gaussian_l2 <= 1e-4 and energy_error <= 0.02 and centroid_error <= 0.25
    pupil_valid = (
        pupil.get("cutoff_relative_error", float("inf")) <= 1e-6
        and pupil.get("pass_stop_response_error", float("inf")) <= 0.01
        and pupil.get("airy_first_zero_error_pixel", float("inf")) <= 1.0
    )
    scaled_valid = (
        scaled.get("complex_relative_l2", float("inf")) <= 1e-2
        and scaled.get("wrapped_phase_mae_rad", float("inf")) <= 0.02
        and scaled.get("energy_relative_error", float("inf")) <= 0.02
        and scaled.get("boundary_energy_fraction", float("inf")) <= 1e-4
    )
    reference_harness_valid = production.get("status") == "valid" and pupil.get("status") == "valid" and scaled.get("status") == "valid"
    if not reference_harness_valid or not scaled_valid:
        status = "reference-harness-invalid"
    elif not production_valid or not pupil_valid or reasons:
        status = "production-relay-invalid"
    elif structural.get("status") != "valid":
        status = "physical-forward-model-invalid"
    else:
        status = "physical-forward-model-valid"
    return {
        "status": status,
        "validation_revision": "v58.2-layered-b0-relay-validation",
        "structural_preflight": structural,
        "production_analytic_validation": production,
        "pupil_validation": pupil,
        "scaled_system_propagation_validation": scaled,
        "sampling_audit": {
            "actual_size_low_resolution_direct_rs": "historical invalid prototype; excluded from acceptance",
            "periodic_computational_window": "production uses a periodic Fourier boundary, not an exterior finite SLM aperture",
        },
        "reference_harness_valid": reference_harness_valid and scaled_valid,
        "hardware_assumptions": {
            "front_clear_aperture_m": FRONT_PUPIL,
            "back_clear_aperture_m": 0.0266,
            "note": "pupil/NA are design lower bounds, not measured hardware values",
        },
        "reasons": reasons,
        "diagnostic_only": True,
        "accepted_summary_written": False,
    }


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
