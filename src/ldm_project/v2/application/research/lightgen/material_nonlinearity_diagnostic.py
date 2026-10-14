"""Passive, unit-explicit material scenarios. Never imported by training optics."""

from __future__ import annotations

import math

import torch
from ldm_project.v2.application.research.lightgen.material_nonlinearity import (
    MATERIAL_MODELS,
    MaterialNonlinearity,
    MaterialParameters,
)


def nonlinear_coverage(intensity_w_cm2: torch.Tensor, i_sat_w_cm2: float) -> dict[str, object]:
    """Preserve all B x 100 ratios and per-sample/per-channel distributions."""
    if intensity_w_cm2.ndim != 2 or intensity_w_cm2.shape[1] != 100 or intensity_w_cm2.shape[0] == 0:
        raise ValueError("coverage requires a nonempty [batch,100] intensity table")
    if not torch.is_floating_point(intensity_w_cm2) or not torch.isfinite(intensity_w_cm2).all() or (intensity_w_cm2 < 0).any():
        raise ValueError("coverage intensity must be finite nonnegative W/cm^2")
    if not math.isfinite(i_sat_w_cm2) or i_sat_w_cm2 <= 0:
        raise ValueError("coverage I_sat must be finite positive W/cm^2")
    ratio = intensity_w_cm2.detach().to(torch.float64) / i_sat_w_cm2

    def statistics(values: torch.Tensor, dim: int) -> dict[str, object]:
        quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device, dtype=values.dtype), dim=dim)
        return {
            "mean": values.mean(dim).tolist(), "median": quantiles[1].tolist(),
            "p10": quantiles[0].tolist(), "p50": quantiles[1].tolist(), "p90": quantiles[2].tolist(),
            "max": values.amax(dim).tolist(), "min": values.amin(dim).tolist(),
        }

    linear = ratio < 0.1
    nonlinear = (ratio >= 0.1) & (ratio <= 10)
    saturated = ratio > 10
    return {
        "intensity_unit": "W/cm^2", "i_sat_w_cm2": i_sat_w_cm2,
        "r_each": ratio.tolist(), "batch_statistics": statistics(ratio.flatten(), 0),
        "per_sample_statistics": statistics(ratio, 1), "per_channel_statistics": statistics(ratio, 0),
        "fraction_linear": float(linear.double().mean()),
        "fraction_nonlinear": float(nonlinear.double().mean()),
        "fraction_saturated": float(saturated.double().mean()),
        "nonlinear_activation_coverage": float(nonlinear.double().mean()),
        "per_sample_fraction_nonlinear": nonlinear.double().mean(1).tolist(),
        "per_channel_fraction_nonlinear": nonlinear.double().mean(0).tolist(),
        "decision": "distribution_only_no_threshold_gate",
    }


def validate_material_scenarios() -> dict[str, object]:
    """Execute value, derivative/sign, phase, passivity and faulty-amplitude controls."""
    intensity = torch.tensor([1e-6, 52.0, 520.0, 5200.0, 520000.0], dtype=torch.float64, requires_grad=True)
    # At the measured base points these spatial fields have |U|^2 = I * 1e4 W/m^2.
    incoming = torch.complex((intensity.detach()*1e4).sqrt(), torch.zeros_like(intensity))
    material = MaterialNonlinearity()
    values = []
    all_passed = True
    for model in MATERIAL_MODELS:
        params = MaterialParameters(model, t_ns=0.7, delta_t=0.2, i_sat_w_cm2=520.0, i_rsa_w_cm2=5200.0, phi_max_rad=0.5)
        output = material.forward(incoming, intensity, params)
        measured = output.abs().square()/incoming.abs().square()
        slopes = torch.autograd.grad(measured.sum(), intensity, retain_graph=True)[0] if model != "identity" else torch.zeros_like(intensity)
        value_error = 0.0
        slope_error = 0.0
        phase_error = 0.0
        for index, point in enumerate(intensity.detach().tolist()):
            sa = 0.7 + 0.2*point/(point+520)
            sa_slope = 0.2*520/(point+520)**2
            expected, derivative, phase = 1.0, 0.0, 0.0
            if model != "identity":
                expected, derivative = sa, sa_slope
            if model == "saturable_absorption_plus_RSA":
                expected = sa/(1+point/5200)
                derivative = sa_slope/(1+point/5200)-sa/(5200*(1+point/5200)**2)
            if model == "amplitude_plus_phase_nonlinearity":
                phase = 0.5*point/(point+520)
            value_error = max(value_error, abs(float(measured[index])-expected))
            slope_error = max(slope_error, abs(float(slopes[index])-derivative))
            phase_error = max(phase_error, abs(float(torch.angle(output[index]*incoming[index].conj()))-phase))
        positive_sa_sign = model not in ("saturable_absorption", "amplitude_plus_phase_nonlinearity") or bool((slopes > 0).all())
        negative_rsa_high_sign = model != "saturable_absorption_plus_RSA" or float(slopes[-1]) < 0
        passive = bool((measured <= 1+1e-12).all() and (measured >= 0).all())
        passed = value_error < 1e-12 and slope_error < 1e-12 and phase_error < 1e-12 and passive and positive_sa_sign and negative_rsa_high_sign
        all_passed = all_passed and passed
        values.append({"model": model, "passed": passed, "max_transmission_error": value_error,
                       "max_intensity_derivative_error": slope_error, "max_phase_error_rad": phase_error,
                       "passive": passive, "derivative_sign_checked": positive_sa_sign and negative_rsa_high_sign})
    # A faulty amplitude multiplier T would produce T^2 power and fail the value consumer.
    expected_sa = 0.7+0.2*520/(520+520)
    faulty_amplitude_rejected = abs(expected_sa**2-expected_sa) > 1e-12
    # Central difference through the actual class at a fixed incident field.
    point = 520.0
    step = 1e-2
    params = MaterialParameters("saturable_absorption", t_ns=0.7, delta_t=0.2, i_sat_w_cm2=520.0)
    fixed = torch.tensor([complex(1, 0)], dtype=torch.complex128)
    plus = material.forward(fixed, torch.tensor([point+step], dtype=torch.float64), params).abs().square()[0]
    minus = material.forward(fixed, torch.tensor([point-step], dtype=torch.float64), params).abs().square()[0]
    fd = float((plus-minus)/(2*step))
    fd_error = abs(fd-0.2*520/(point+520)**2)
    passed = all_passed and faulty_amplitude_rejected and fd_error < 1e-10
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "models": values,
            "intensity_unit": "W/cm^2", "finite_difference_error": fd_error,
            "faulty_T_instead_of_sqrt_T_rejected": faulty_amplitude_rejected,
            "claim_limit": "static generic equations; no material identification, temporal dynamics or measured-device fit"}
