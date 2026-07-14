#!/usr/bin/env python3
"""Full-FOV acceptance harness for the formal ideal B=0 front relay.

This is a read-only CUDA/CPU validator.  It does not load checkpoints, train,
evaluate a dataset, or generate a sample run.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.optics import coherent_imaging_4f_field
from ldm_project.optics.reference import (
    collins_b0_relay_reference,
    four_f_abcd,
    measure_circular_pupil_first_zero,
    similarity_scaled_propagation_reference,
)


F1 = 0.22321428571428573
F2 = 0.0892857142857143
WAVELENGTH = 532e-9
FRONT_PUPIL = 0.02375
MAGNIFICATION = -0.4
INPUT_SIZE = 200
OUTPUT_SIZE = 100
INPUT_PIXEL = 2.5e-6
OUTPUT_PIXEL = 2.0e-6


def _args() -> dict[str, object]:
    matrix = four_f_abcd(F1, F2)
    return {
        "magnification": matrix[0][0],
        "output_size": (OUTPUT_SIZE, OUTPUT_SIZE),
        "input_pixel_size": INPUT_PIXEL,
        "output_pixel_size": OUTPUT_PIXEL,
        "wavelength": WAVELENGTH,
        "c_abcd": matrix[1][0],
        "pupil_diameter": FRONT_PUPIL,
        "pupil_focal_length": F2,
    }


def _align(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    phase = torch.angle((candidate.conj() * reference).sum())
    return candidate * torch.exp(1j * phase).to(candidate.dtype)


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    aligned = _align(candidate, reference)
    valid = reference.abs().square() >= reference.abs().square().amax() * 1.0e-6
    return {
        "complex_relative_l2": float((aligned - reference).norm() / reference.norm().clamp_min(1e-30)),
        "wrapped_phase_mae_rad": float(torch.angle(aligned[valid] * reference[valid].conj()).abs().mean()),
        "energy_relative_error": float(
            (candidate.abs().square().sum() - reference.abs().square().sum()).abs()
            / reference.abs().square().sum().clamp_min(1e-30)
        ),
    }


def _input_grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = (torch.arange(INPUT_SIZE, device=device, dtype=torch.float64) - (INPUT_SIZE - 1) / 2.0) * INPUT_PIXEL
    return torch.meshgrid(axis, axis, indexing="ij")


def _output_grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = (torch.arange(OUTPUT_SIZE, device=device, dtype=torch.float64) - (OUTPUT_SIZE - 1) / 2.0) * OUTPUT_PIXEL
    return torch.meshgrid(axis, axis, indexing="ij")


def _closed_form_case(field_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    yy_in, xx_in = _input_grid(device)
    yy_out, xx_out = _output_grid(device)
    if field_name == "plane":
        source = torch.ones_like(xx_in, dtype=torch.complex128)
        expected = torch.ones_like(xx_out, dtype=torch.complex128) / abs(MAGNIFICATION)
    else:
        frequency = {"tilt_24k": 24_000.0, "tilt_74k": 74_000.0}[field_name]
        source = torch.exp(2j * math.pi * frequency * xx_in).to(torch.complex128)
        expected = torch.exp(2j * math.pi * frequency * (xx_out / MAGNIFICATION)).to(torch.complex128)
        expected = expected / abs(MAGNIFICATION)
    return source[None, None], expected[None, None]


def _full_fov_validation(device: torch.device) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    args = _args()
    cases: dict[str, dict[str, float]] = {}
    fields: dict[str, torch.Tensor] = {}
    for name in ("plane", "tilt_24k", "tilt_74k"):
        source, expected = _closed_form_case(name, device)
        actual = coherent_imaging_4f_field(source, **args)
        cases[name] = _metrics(actual, expected)
        fields[name] = actual
    yy, xx = _input_grid(device)
    gaussian = torch.exp(-(xx.square() + yy.square()) / (35e-6**2)).to(torch.complex128)[None, None]
    point = torch.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), device=device, dtype=torch.complex128)
    point[..., INPUT_SIZE // 2 + 14, INPUT_SIZE // 2 - 18] = 1.0
    checker = torch.exp(2j * math.pi * 74_000.0 * xx).to(torch.complex128)[None, None]
    # 25/75 kcycles/m are required diagnostic cases, but neither is an FFT bin
    # on a 500 µm FOV (bin spacing = 2 kcycles/m).  They remain independently
    # compared rather than being used as closed-form periodic-field gates.
    diagnostics = {
        "tilt_25k_nonbin": torch.exp(2j * math.pi * 25_000.0 * xx).to(torch.complex128)[None, None],
        "tilt_75k_nonbin": torch.exp(2j * math.pi * 75_000.0 * xx).to(torch.complex128)[None, None],
    }
    for name, source in {"gaussian": gaussian, "shifted_point": point, "pupil_filtered_checkerboard": checker, **diagnostics}.items():
        actual = coherent_imaging_4f_field(source, **args)
        independent = collins_b0_relay_reference(source, **args)
        cases[name] = _metrics(actual, independent)
        fields[name] = actual
    intensity = fields["shifted_point"].abs().square()[0, 0]
    x_centroid = float((intensity.sum(dim=0) * torch.arange(OUTPUT_SIZE, device=device)).sum() / intensity.sum())
    expected_x = (OUTPUT_SIZE - 1) / 2.0 + ((INPUT_SIZE // 2 - 18) - (INPUT_SIZE - 1) / 2.0) * INPUT_PIXEL * MAGNIFICATION / OUTPUT_PIXEL
    return {
        "status": "valid",
        "input_sampling": {"shape": [INPUT_SIZE, INPUT_SIZE], "pixel_size_m": INPUT_PIXEL, "fov_m": INPUT_SIZE * INPUT_PIXEL},
        "output_sampling": {"shape": [OUTPUT_SIZE, OUTPUT_SIZE], "pixel_size_m": OUTPUT_PIXEL, "fov_m": OUTPUT_SIZE * OUTPUT_PIXEL},
        "magnification": MAGNIFICATION,
        "cases": cases,
        "point_centroid_error_encoder_pixel": abs(x_centroid - expected_x),
    }, fields


def _pupil_validation(device: torch.device) -> dict[str, object]:
    args = _args()
    _, xx = _input_grid(device)
    def response(frequency: float) -> float:
        source = torch.exp(2j * math.pi * frequency * xx).to(torch.complex128)[None, None]
        output = coherent_imaging_4f_field(source, **args)
        return float(output.abs().mean() * abs(MAGNIFICATION))
    wide = measure_circular_pupil_first_zero(
        source_size=INPUT_SIZE, input_pixel_size=INPUT_PIXEL, magnification=MAGNIFICATION,
        wavelength=WAVELENGTH, pupil_diameter=FRONT_PUPIL, pupil_focal_length=F2,
        radial_step=0.05e-6, radial_extent=10e-6, device=device,
    )
    narrow_diameter = 0.005
    narrow = measure_circular_pupil_first_zero(
        source_size=INPUT_SIZE, input_pixel_size=INPUT_PIXEL, magnification=MAGNIFICATION,
        wavelength=WAVELENGTH, pupil_diameter=narrow_diameter, pupil_focal_length=F2,
        radial_step=0.05e-6, radial_extent=30e-6, device=device,
    )
    predicted = 1.22 * WAVELENGTH * F2 / FRONT_PUPIL
    predicted_narrow = 1.22 * WAVELENGTH * F2 / narrow_diameter
    object_cutoff = FRONT_PUPIL / (2.0 * F1 * WAVELENGTH)
    image_cutoff = FRONT_PUPIL / (2.0 * F2 * WAVELENGTH)
    return {
        "status": "valid",
        "object_cutoff_cycles_m": object_cutoff,
        "image_cutoff_cycles_m": image_cutoff,
        "cutoff_relative_error": max(abs(object_cutoff - 100_000.0) / 100_000.0, abs(image_cutoff - 250_000.0) / 250_000.0),
        "pass_frequency_cycles_m": 80_000.0,
        "pass_response": response(80_000.0),
        "stop_frequency_cycles_m": 120_000.0,
        "stop_response": response(120_000.0),
        "pass_stop_response_error": max(abs(response(80_000.0) - 1.0), abs(response(120_000.0))),
        "cutoff_boundary_cycles_m": 100_000.0,
        "wide_psf": {**wide, "predicted_first_zero_m": predicted, "error_encoder_pixel": abs(float(wide["measured_first_zero_m"]) - predicted) / OUTPUT_PIXEL},
        "narrow_psf": {**narrow, "predicted_first_zero_m": predicted_narrow, "error_encoder_pixel": abs(float(narrow["measured_first_zero_m"]) - predicted_narrow) / OUTPUT_PIXEL},
        "measured_zero_ratio": float(narrow["measured_first_zero_m"]) / float(wide["measured_first_zero_m"]),
        "predicted_zero_ratio": FRONT_PUPIL / narrow_diameter,
    }


def _fault_injection(device: torch.device) -> dict[str, bool]:
    source, expected = _closed_form_case("tilt_74k", device)
    yy_out, xx_out = _output_grid(device)
    wrong_magnification = torch.exp(2j * math.pi * 74_000.0 * (xx_out / 0.4)).to(torch.complex128)[None, None] / 0.4
    # 倒像与倍率是不同的 B=0 成像条件：前者只错误地丢掉负号，
    # 后者保留倒像但使用错误的绝对倍率。二者必须分别触发验收故障。
    wrong_inversion = torch.exp(2j * math.pi * 74_000.0 * (xx_out / abs(MAGNIFICATION))).to(torch.complex128)[None, None] / abs(MAGNIFICATION)
    wrong_amplitude = expected * abs(MAGNIFICATION)
    spurious_phase = expected * torch.exp(1j * 1.0e12 * (xx_out.square() + yy_out.square())).to(torch.complex128)[None, None]
    # A 120 kcycles/m field is outside the object pupil and must not survive.
    _, xx_in = _input_grid(device)
    high_source = torch.exp(2j * math.pi * 120_000.0 * xx_in).to(torch.complex128)[None, None]
    pupil_omitted = torch.exp(2j * math.pi * 120_000.0 * (xx_out / MAGNIFICATION)).to(torch.complex128)[None, None] / abs(MAGNIFICATION)
    actual_high = coherent_imaging_4f_field(high_source, **_args())
    return {
        "wrong_magnification_detected": _metrics(wrong_magnification, expected)["complex_relative_l2"] > 1e-2,
        "wrong_inversion_detected": _metrics(wrong_inversion, expected)["complex_relative_l2"] > 1e-2,
        "wrong_amplitude_detected": _metrics(wrong_amplitude, expected)["energy_relative_error"] > 0.02,
        "spurious_quadratic_phase_detected": _metrics(spurious_phase, expected)["wrapped_phase_mae_rad"] > 0.02,
        "pupil_omission_detected": float(actual_high.abs().mean() * abs(MAGNIFICATION)) < 1e-6 and float(pupil_omitted.abs().mean()) > 1.0,
    }


def _independence_audit() -> dict[str, object]:
    production = inspect.getsource(coherent_imaging_4f_field)
    reference = inspect.getsource(collins_b0_relay_reference)
    return {
        "production_has_no_grid_sample": "grid_sample" not in production,
        "production_has_no_finite_sinc": "torch.sinc" not in production,
        "production_has_no_generic_resize": "F.interpolate" not in production and "_resize_complex_field" not in production,
        "reference_imports_no_production_relay": "coherent_imaging_4f_field" not in reference,
        "reference_is_canonical_optics_module": collins_b0_relay_reference.__module__ == "ldm_project.optics.reference",
        "v58_2_interpretation": "historical CUDA numerical self-consistency only; not v58.3 acceptance evidence",
    }


def _report_contract() -> dict[str, object]:
    """Machine-checkable v58.3 constants consumed by the CUDA preflight."""
    return {
        "input_sampling": {"shape": [INPUT_SIZE, INPUT_SIZE], "pixel_size_m": INPUT_PIXEL, "fov_m": INPUT_SIZE * INPUT_PIXEL},
        "output_sampling": {"shape": [OUTPUT_SIZE, OUTPUT_SIZE], "pixel_size_m": OUTPUT_PIXEL, "fov_m": OUTPUT_SIZE * OUTPUT_PIXEL},
        "wavelength_m": WAVELENGTH,
        "front_magnification": MAGNIFICATION,
        "front_abcd": {"A": MAGNIFICATION, "B": 0.0, "C": 0.0},
        "front_pupil_diameter_m": FRONT_PUPIL,
        "fault_keys": sorted(_fault_injection(torch.device("cpu")).keys()),
    }


def run(config_path: str | Path, device: str = "cpu") -> dict[str, object]:
    reasons: list[str] = []
    config = load_training_config(config_path)
    try:
        validate_physical_forward_contract(config, config.physics)
        matrix = four_f_abcd(config.physics.macro_optics.front_lens1_focal_length, config.physics.macro_optics.front_lens2_focal_length)
        formal = {
            "status": "valid",
            "front_abcd": {"A": matrix[0][0], "B": matrix[0][1], "C": matrix[1][0]},
            "ols_identity": config.one_shot.ols_kind == "identity",
        }
        if abs(matrix[0][0] - MAGNIFICATION) > 1e-10 or abs(matrix[0][1]) > 1e-10 or abs(matrix[1][0]) > 1e-10:
            reasons.append("formal front relay is not A=-0.4, B=0, C=0")
    except Exception as exc:
        formal = {"status": "invalid", "error": str(exc)}
        reasons.append(str(exc))
    target = torch.device(device)
    try:
        full, _ = _full_fov_validation(target)
        pupil = _pupil_validation(target)
        component = similarity_scaled_propagation_reference(device=target)
        audit = _independence_audit()
        faults = _fault_injection(target)
    except Exception as exc:
        full, pupil, component, audit, faults = ({"status": "invalid", "error": str(exc)},) * 5
        reasons.append(f"reference execution failed: {exc}")
    cases = full.get("cases", {})
    plane_tilt_l2 = max((cases.get(name, {}).get("complex_relative_l2", float("inf")) for name in ("plane", "tilt_24k", "tilt_74k")), default=float("inf"))
    gaussian_l2 = cases.get("gaussian", {}).get("complex_relative_l2", float("inf"))
    energy = max((item["energy_relative_error"] for item in cases.values()), default=float("inf"))
    full_valid = plane_tilt_l2 <= 1e-5 and gaussian_l2 <= 1e-4 and energy <= 0.02 and full.get("point_centroid_error_encoder_pixel", float("inf")) <= 0.25
    pupil_valid = (
        pupil.get("cutoff_relative_error", float("inf")) <= 1e-6
        and pupil.get("pass_stop_response_error", float("inf")) <= 0.01
        and pupil.get("wide_psf", {}).get("error_encoder_pixel", float("inf")) <= 1.0
        and pupil.get("narrow_psf", {}).get("error_encoder_pixel", float("inf")) <= 1.0
        and abs(pupil.get("measured_zero_ratio", 0.0) / pupil.get("predicted_zero_ratio", 1.0) - 1.0) <= 0.2
    )
    audit_valid = all(value is True for key, value in audit.items() if key != "v58_2_interpretation") and all(faults.values())
    component_report = {
        "status": "diagnostic-only",
        "complex_relative_l2": component.get("complex_relative_l2"),
        "wrapped_phase_mae_rad": component.get("wrapped_phase_mae_rad"),
        "energy_relative_error": component.get("energy_relative_error"),
        "boundary_energy_fraction": component.get("boundary_energy_fraction"),
        "sampling": component.get("sampling"),
        "scope": "single propagation component only; not evidence for the full actual-pupil two-lens relay",
    }
    if formal.get("status") != "valid" or full.get("status") != "valid" or pupil.get("status") != "valid" or not audit_valid:
        status = "reference-harness-invalid"
    elif not full_valid or not pupil_valid or reasons:
        status = "production-relay-invalid"
    else:
        status = "physical-forward-model-valid"
    return {
        "status": status,
        "validation_revision": "v58.3-full-fov-b0-relay-acceptance",
        "formal_full_fov_validation": formal | full,
        "independent_analytic_validation": {"cases": full.get("cases"), "fault_injection": faults},
        "measured_pupil_psf_validation": pupil,
        "propagation_component_diagnostic": component_report,
        "sampling_and_independence_audit": audit,
        "report_contract": _report_contract(),
        "reasons": reasons,
        "diagnostic_only": True,
        "accepted_summary_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "physical-forward-model-valid" else 1)


if __name__ == "__main__":
    main()
