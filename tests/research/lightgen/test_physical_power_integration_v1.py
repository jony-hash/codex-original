from __future__ import annotations

import math

import pytest
import torch

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.application.research.lightgen.material_nonlinearity import MaterialParameters
from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import (
    calibrate_source,
    encoder_power_radius_diagnostic,
    fiber_reference_plane_diagnostic,
    gaussian,
)
from ldm_project.v2.contracts import PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack


def _physical_config(*, material: MaterialParameters | None = None) -> LightGenPhysicalOpticsConfig:
    return LightGenPhysicalOpticsConfig(
        input_power_w=1.0e-3,
        layer_power_transmission=0.889,
        global_fiber_geometry=True,
        fiber_pitch_m=125.0e-6,
        fiber_cladding_diameter_m=125.0e-6,
        fiber_mfd_m=14.222e-6,
        fiber_length_m=8.5e-3,
        fiber_attenuation_db_per_km=0.0,
        material=material,
    )


def test_reference_plane_uses_centered_125um_coordinates_and_global_boundary():
    plane = PlaneSpec("global", (2048, 2048), (2e-6, 2e-6), 532e-9, 0, "global")
    report = fiber_reference_plane_diagnostic(
        plane,
        trainable_width_m=200e-6,
        fiber_pitch_m=125e-6,
        cladding_diameter_m=100e-6,
    )
    assert report["status"] == "PASS"
    assert report["fiber_center_span_m"] == pytest.approx(1.125e-3)
    assert report["fiber_outer_extent_m"] == pytest.approx(1.225e-3)
    assert report["fiber_center_x_m_row_major"][0] == pytest.approx(-562.5e-6)
    assert report["fiber_center_x_m_row_major"][-1] == pytest.approx(562.5e-6)
    assert report["left_margin_m"] == pytest.approx((4.096e-3 - 1.225e-3) / 2.0)
    assert report["cladding_pitch_independent"] is True


def test_gaussian_overlap_matches_one_global_array_mode_without_energy_creation():
    plane = PlaneSpec("global", (2048, 2048), (2e-6, 2e-6), 532e-9, 0, "global")
    geometry = GaussianFiberGeometry(port_pitch_m=125e-6, waist_m=3.5e-6)
    operator = GaussianFiberArrayOLS(plane, geometry)
    center_y, center_x = operator.centers_m[44].tolist()
    incident = calibrate_source(gaussian(plane, geometry.waist_m, x_m=center_x, y_m=center_y), plane, 1e-3)
    coefficients = operator.project_coefficients(incident)
    input_power = float(incident.tensor.abs().square().sum() * math.prod(plane.pixel_pitch_m))
    each_power = coefficients.abs().square()
    assert float(each_power[0, 44] / input_power) == pytest.approx(1.0, abs=2e-5)
    assert float(each_power.sum()) <= input_power * (1.0 + 1e-5) + 1e-12


def test_physical_forward_consumes_calibrated_power_global_overlap_and_layer_transmission():
    config = LightGen2DAlignmentRuntimeConfig(
        propagation_grid_size=1024,
        physical_optics=_physical_config(),
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(config).eval()
    image = torch.ones(1, 1, 28, 28)
    with torch.no_grad():
        result = runtime.forward_from_image(image)
    assert result.ols_input_field is not None
    assert result.ols_input_field.plane.shape == (1024, 1024)
    assert result.generator_input_field.plane.shape == (100, 100)
    trace = result.physical_power_trace
    assert trace is not None
    assert float(trace["P_input_W"][0]) == pytest.approx(1e-3, rel=1e-6)
    assert float(trace["P_after_slm_W"][0]) == pytest.approx(1e-3, rel=1e-6)
    assert float(trace["P_encoder_1_W"][0]) < float(trace["P_encoder_entrance_W"][0])
    assert float(trace["P_encoder_2_W"][0]) <= float(trace["P_encoder_1_W"][0])
    assert trace["fiber_coupled_power_W"].shape == (1, 100)
    assert torch.all(trace["fiber_coupled_power_W"] >= 0)
    assert float(trace["P_fiber_total_W"][0]) <= float(trace["P_before_fiber_W"][0]) * (1.0 + 1e-5) + 1e-12
    assert torch.allclose(trace["P_fiber_output_total_W"], trace["P_fiber_total_W"], rtol=1e-6, atol=1e-12)
    assert torch.isfinite(trace["I_peak_W_m2"]).all()
    assert result.ols_input_field.provenance.parameters["modal_projection_boundary"] is True
    radii = encoder_power_radius_diagnostic(
        result.ols_input_field,
        fiber_pitch_m=125e-6,
        cladding_diameter_m=125e-6,
    )
    assert radii["power_radii_m"]["r99_m"][0] < radii["fiber_array_half_width_m"]
    assert result.z_ols.mode_geometry["fiber_center_span_m"] == pytest.approx(1.125e-3)


def test_identity_material_matches_a3_linear_output_and_sa_is_passive():
    a3 = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=1024, physical_optics=_physical_config())
    ).eval()
    a4_identity = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(
            propagation_grid_size=1024,
            physical_optics=_physical_config(material=MaterialParameters("identity")),
        )
    ).eval()
    a4_identity.load_state_dict(a3.state_dict())
    image = torch.linspace(0.0, 1.0, 28 * 28).reshape(1, 1, 28, 28)
    with torch.no_grad():
        baseline = a3.forward_from_image(image)
        identity = a4_identity.forward_from_image(image)
    assert torch.equal(baseline.generator_input_field.tensor, identity.generator_input_field.tensor)
    assert torch.equal(baseline.raw_camera.tensor, identity.raw_camera.tensor)
    assert torch.equal(
        baseline.physical_power_trace["fiber_output_power_W"],
        identity.physical_power_trace["P_after_material_each_W"],
    )
    assert a3.ols.operator.output_spec.waist_m == pytest.approx(7.111e-6)

    sa = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(
            propagation_grid_size=1024,
            physical_optics=_physical_config(
                material=MaterialParameters(
                    "saturable_absorption",
                    t_ns=0.7,
                    delta_t=0.2,
                    i_sat_w_cm2=520.0,
                )
            ),
        )
    ).eval()
    with torch.no_grad():
        activated = sa.forward_from_image(image)
    sa_trace = activated.physical_power_trace
    intensity = sa_trace["I_peak_W_cm2"]
    expected_transmission = 0.7 + 0.2 * intensity / (intensity + 520.0)
    expected_power = sa_trace["fiber_output_power_W"] * expected_transmission
    assert torch.allclose(sa_trace["P_after_material_each_W"], expected_power, rtol=1e-6, atol=1e-12)
    assert torch.all(sa_trace["P_after_material_each_W"] <= sa_trace["fiber_output_power_W"] + 1e-12)


def test_layer_power_transmission_multiplies_field_by_sqrt_transmission():
    plane = PlaneSpec("active", (100, 100), (2e-6, 2e-6), 532e-9, 0, "active")
    stack = FiniteCropPhaseStack(
        layers=1,
        active_shape=(100, 100),
        propagation_shape=(128, 128),
        spacing_m=1.5e-3,
        layer_power_transmission=0.889,
        preserve_global_output=True,
    )
    field = torch.ones(1, 1, 100, 100, dtype=torch.complex64)
    from ldm_project.v2.contracts import ComplexField, OperationProvenance

    input_field = ComplexField(field, plane, OperationProvenance("test", None, None))
    with torch.no_grad():
        output, records, stats = stack(input_field, "test")
    before = float(field.abs().square().sum() * 4e-12)
    assert float(stats["layer_power"][0, 0, 1]) == pytest.approx(before * 0.889, rel=1e-6)
    assert float(stats["layer_power"][0, 0, 2]) == pytest.approx(before * 0.889, rel=1e-5)
    assert output.plane.shape == (128, 128)
    assert records[0].plane.shape == (128, 128)
