"""Non-scientific contract tests for the LightGen-aligned 2-D runtime."""

from __future__ import annotations

import inspect

import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import ComplexOLSState, LightGen2DAlignmentRuntimeConfig, LightGen2DSemanticGenerationInputOLSAlignment, ROW_MAJOR_PORT_ORDER


def _runtime() -> LightGen2DSemanticGenerationInputOLSAlignment:
    # The production default remains the 2048 propagation grid.  A smaller
    # explicit grid keeps these interface tests CPU-only and non-scientific.
    return LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=128)
    )


def test_runtime_freezes_complex_ols_interface_and_metadata() -> None:
    runtime = _runtime()
    assert LightGen2DAlignmentRuntimeConfig().propagation_grid_size == 2048
    assert runtime.ols.operator.input_plane.shape == (100, 100)
    assert runtime.ols.operator.input_plane.pixel_pitch_m == (2.0e-6, 2.0e-6)
    assert runtime.ols.transport.shape == (100, 100)
    assert torch.equal(runtime.ols.transport, torch.eye(100, dtype=torch.complex64))
    assert runtime.encoder.layers == 2
    assert runtime.generator.layers == 2
    assert runtime.alignment_metadata["CHEN_FACT"]
    assert runtime.alignment_metadata["CHEN_UNRESOLVED"]
    assert runtime.alignment_metadata["PROJECT_DECISION"]


def test_modal_projection_is_complex_active_and_row_major() -> None:
    runtime = _runtime()
    mode = runtime.ols.operator.input_modes[0]
    phase = torch.tensor(0.37, dtype=torch.float32)
    phase_factor = torch.polar(torch.ones_like(mode.real), phase)
    field = ComplexField(
        (mode * phase_factor).unsqueeze(0).unsqueeze(0),
        runtime.ols.operator.input_plane,
        OperationProvenance("fixture_modal_input", None, None),
    )

    state = runtime.ols.project_state(field)

    assert state.coefficients.shape == (1, 100)
    assert torch.is_complex(state.coefficients)
    assert state.port_order == ROW_MAJOR_PORT_ORDER
    assert state.source == "real_encoded"
    assert torch.isclose(state.coefficients[0, 0].abs(), torch.tensor(1.0), atol=2e-5)
    assert torch.isclose(state.coefficients[0, 0].angle(), phase, atol=2e-5)
    assert float(state.coefficients[0, 1:].abs().max()) < 2e-5


def test_image_adapter_exposes_project_decision_without_hiding_field_semantics() -> None:
    runtime = _runtime()
    image = torch.full((1, 1, 7, 9), 0.25, dtype=torch.float32)

    optical_input = runtime.input_adapter(image)

    assert optical_input.slm_intensity.shape == (1, 1, 200, 200)
    assert optical_input.slm_amplitude.shape == optical_input.slm_intensity.shape
    assert torch.allclose(
        optical_input.slm_amplitude,
        optical_input.slm_intensity.sqrt(),
    )
    assert torch.is_complex(optical_input.slm_field.tensor)
    assert torch.allclose(optical_input.slm_field.tensor.imag, torch.zeros_like(optical_input.slm_field.tensor.imag))
    assert optical_input.slm_field.provenance.parameters["mapping_status"] == "PROJECT_DECISION"


def test_modal_writer_is_coefficient_space_t_then_reemission() -> None:
    runtime = _runtime()
    coefficients = torch.randn(2, 100, dtype=torch.complex64)
    state = ComplexOLSState.from_external(coefficients)

    field = runtime.w_ols(state)
    transmitted = coefficients @ runtime.ols.transport.T
    expected = torch.einsum(
        "bm,mhw->bhw",
        transmitted,
        runtime.ols.operator.output_modes.to(coefficients),
    )

    assert field.plane == runtime.generator_entrance_plane
    assert torch.allclose(field.tensor[:, 0], expected)
    assert field.provenance.parameters["transport"] == "T=I_100_coefficient_space"
    assert field.provenance.parameters["whole_field_identity_bypass"] is False


def test_image_and_direct_paths_converge_at_the_same_writer_and_generator() -> None:
    runtime = _runtime().eval()
    image = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

    image_result = runtime.forward_from_image(image)
    direct_result = runtime.forward_from_ols(image_result.z_ols)

    assert image_result.image_input is not None
    assert image_result.encoder_output_field is not None
    assert image_result.ols_input_field is not None
    assert image_result.ols_input_field.plane == runtime.ols.operator.input_plane
    assert direct_result.image_input is None
    assert direct_result.encoder_output_field is None
    assert direct_result.ols_input_field is None
    assert image_result.generator_input_field.plane == direct_result.generator_input_field.plane
    assert torch.equal(image_result.generator_input_field.tensor, direct_result.generator_input_field.tensor)
    assert torch.equal(image_result.raw_camera.tensor, direct_result.raw_camera.tensor)
    assert image_result.raw_camera.tensor.ndim == 4
    assert image_result.raw_camera.tensor.shape[-2:] == (200, 200)


def test_direct_path_rejects_q_only_and_has_no_controller_prerequisite() -> None:
    runtime = _runtime()
    with pytest.raises(TypeError, match="complex"):
        runtime.forward_from_ols(torch.ones(1, 100))

    parameters = tuple(inspect.signature(runtime.forward_from_ols).parameters)
    assert parameters == ("z",)

    with pytest.raises(ValueError, match="row-major"):
        ComplexOLSState(torch.zeros(1, 100, dtype=torch.complex64), port_order=tuple(reversed(ROW_MAJOR_PORT_ORDER)))


def test_writer_rejects_incompatible_generator_plane_and_nonfinite_state() -> None:
    runtime = _runtime()
    bad_plane = PlaneSpec(
        "bad_generator_plane",
        (100, 100),
        (3.0e-6, 3.0e-6),
        532.0e-9,
        runtime.generator_entrance_plane.z_m,
        "generator",
    )
    with pytest.raises(ValueError, match="sampling"):
        runtime.ols.write_to_generator_input(
            ComplexOLSState.from_external(torch.zeros(1, 100, dtype=torch.complex64)),
            bad_plane,
        )
    with pytest.raises(ValueError, match="finite"):
        ComplexOLSState.from_external(
            torch.full((1, 100), complex(float("nan"), 0.0), dtype=torch.complex64)
        )
