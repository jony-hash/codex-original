from __future__ import annotations

import json
import math

import pytest
import torch

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from ldm_project.v2.application.research.lightgen.material_nonlinearity_diagnostic import (
    MATERIAL_MODELS, MaterialNonlinearity, MaterialParameters, nonlinear_coverage, validate_material_scenarios,
)
from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import (
    array_support, calibrate_source, gaussian, no_energy_creation,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import physical_plane_power
from ldm_project.v2.contracts import PlaneSpec
from scripts.research.lightgen.power_budget import load_evidence


def plane():
    return PlaneSpec("test_power", (100, 100), (2e-6, 2e-6), 532e-9, 0, "diagnostic")


def test_source_watts_and_layer_power_transmission():
    source = calibrate_source(gaussian(plane(), 20e-6), plane(), 1e-3)
    assert float(physical_plane_power(source)[0]) == pytest.approx(1e-3, rel=1e-12)
    # Independent direct integral; 0.889 is a power transmission, not amplitude.
    array = source.tensor.numpy()
    power = float((array.real**2+array.imag**2).sum()*4e-12)
    assert power == pytest.approx(1e-3, rel=1e-12)
    output_power = float((source.tensor*math.sqrt(0.889)).abs().square().sum()*4e-12)
    assert output_power/power == pytest.approx(0.889, rel=1e-12)
    assert 0.889**2 != pytest.approx(output_power/power, rel=1e-5)


def test_overlap_producer_through_no_energy_consumer_rejects_missing_area():
    optical_plane = plane()
    operator = GaussianFiberArrayOLS(optical_plane, GaussianFiberGeometry())
    source = calibrate_source(gaussian(optical_plane, 3.5e-6, x_m=-10e-6, y_m=-10e-6), optical_plane, 1e-3)
    coefficients = operator.transmit_coefficients(operator.project_coefficients(source))
    before = physical_plane_power(source)
    assert no_energy_creation(before, coefficients.abs().square())["passed"]
    faulty_coefficients = coefficients/math.prod(optical_plane.pixel_pitch_m)
    assert not no_energy_creation(before, faulty_coefficients.abs().square())["passed"]


def test_pad_does_not_make_nominal_array_fit_coupling_plane():
    active = array_support(200e-6, 125e-6, 125e-6)
    padded = array_support(4096e-6, 125e-6, 125e-6)
    assert active["status"] == "FAIL"
    assert active["margin_m"]["left"] == pytest.approx(-525e-6)
    assert padded["status"] == "PASS"


@pytest.mark.parametrize("model", MATERIAL_MODELS)
def test_material_values_phase_passivity_and_dark_input(model):
    intensity = torch.tensor([0, 520, 52000], dtype=torch.float64)
    incoming = torch.complex((intensity*1e4).sqrt(), torch.zeros_like(intensity))
    params = MaterialParameters(model, t_ns=0.7, delta_t=0.2, i_sat_w_cm2=520, i_rsa_w_cm2=5200, phi_max_rad=0.5)
    output = MaterialNonlinearity().forward(incoming, intensity, params)
    assert torch.isfinite(output).all() and output[0] == 0
    assert (output.abs().square() <= incoming.abs().square()*(1+1e-12)).all()
    if model == "saturable_absorption":
        assert float(output[1].abs().square()/incoming[1].abs().square()) == pytest.approx(0.8)
    if model == "amplitude_plus_phase_nonlinearity":
        assert float(torch.angle(output[1]*incoming[1].conj())) == pytest.approx(0.25)


def test_material_executed_reference_includes_derivatives_and_faulty_amplitude(monkeypatch):
    assert validate_material_scenarios()["passed"]
    original = MaterialNonlinearity.forward

    def faulty(self, field, intensity_w_cm2, params):
        correct = original(self, field, intensity_w_cm2, params)
        if params.model == "identity":
            return correct
        # Inject T amplitude instead of sqrt(T) without rewriting the reference.
        ratio = correct.abs()/field.abs()
        return correct*ratio

    monkeypatch.setattr(MaterialNonlinearity, "forward", faulty)
    assert not validate_material_scenarios()["passed"]


def test_coverage_boundaries_keep_dark_and_saturated_channels():
    ratios = torch.tensor([[0.0]*25+[0.1]*25+[10.0]*25+[10.001]*25], dtype=torch.float64)
    result = nonlinear_coverage(ratios*520, 520)
    assert result["fraction_linear"] == 0.25
    assert result["fraction_nonlinear"] == 0.5
    assert result["fraction_saturated"] == 0.25
    assert len(result["r_each"][0]) == 100
    assert len(result["per_channel_statistics"]["mean"]) == 100


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_source_rejects_invalid_physical_power(bad):
    with pytest.raises(ValueError, match="source power"):
        calibrate_source(gaussian(plane(), 20e-6), plane(), bad)


def test_passive_material_rejects_gain_and_undefined_rsa():
    with pytest.raises(ValueError, match="passive SA"):
        MaterialParameters("saturable_absorption", t_ns=0.9, delta_t=0.2, i_sat_w_cm2=520)
    with pytest.raises(ValueError, match="I_RSA"):
        MaterialParameters("saturable_absorption_plus_RSA", i_sat_w_cm2=520)


def test_budget_boundary_loads_production_coefficients_and_rejects_energy_creation(tmp_path):
    optical_plane = plane()
    operator = GaussianFiberArrayOLS(optical_plane, GaussianFiberGeometry())
    source = calibrate_source(gaussian(optical_plane, 3.5e-6, x_m=-10e-6, y_m=-10e-6), optical_plane, 1e-3)
    coefficients = operator.project_coefficients(source)
    powers = coefficients.abs().square()
    before = physical_plane_power(source)
    payload = {"schema": "physical_power_component_evidence.v1", "source_power_w": 1e-3,
               "profile_ids": ["matched"], "basis_results": {"gaussian_s12_v1": {
                   "coefficients": coefficients, "channel_power_w": powers, "before_fiber_w": before,
                   "peak_mode_density_per_m2": torch.full((100,), 2/(math.pi*(3.5e-6)**2), dtype=torch.float64)}}}
    report = {"source_power_w": 1e-3, "power_unit": "W", "field_unit": "sqrt(W)/m",
              "basis_results": {"gaussian_s12_v1": {"channel_power_w_each": powers.tolist()}}}
    (tmp_path/"diagnostics.json").write_text(json.dumps({"schema": "physical_power_diagnostics.v1", "numeric_component_checks_passed": True}), encoding="utf-8")
    (tmp_path/"absolute_power.json").write_text(json.dumps(report), encoding="utf-8")
    torch.save(payload, tmp_path/"component_power_evidence.pt")
    assert load_evidence(tmp_path)[1]["source_power_w"] == 1e-3
    payload["basis_results"]["gaussian_s12_v1"]["coefficients"] = coefficients*2
    payload["basis_results"]["gaussian_s12_v1"]["channel_power_w"] = powers*4
    report["basis_results"]["gaussian_s12_v1"]["channel_power_w_each"] = (powers*4).tolist()
    (tmp_path/"absolute_power.json").write_text(json.dumps(report), encoding="utf-8")
    torch.save(payload, tmp_path/"component_power_evidence.pt")
    with pytest.raises(ValueError, match="non-passive channel evidence"):
        load_evidence(tmp_path)
