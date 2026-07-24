import pytest
import torch

from ldm_project.optical_sampling import (
    PlaneSampling,
    aperture_sampling,
    center_aperture_indices,
    sample_center_aperture,
    validate_sampling_transition,
)


def test_plane_sampling_reports_physical_fov():
    plane = PlaneSampling("d2nn", height=100, width=100, pixel_size_y=2e-6, pixel_size_x=2e-6)

    assert plane.shape == (100, 100)
    assert plane.fov_y == pytest.approx(200e-6)
    assert plane.fov_x == pytest.approx(200e-6)


def test_sampling_transition_rejects_unannotated_grid_resize():
    source = PlaneSampling("slm", height=200, width=200, pixel_size_y=2.5e-6, pixel_size_x=2.5e-6)
    target = PlaneSampling("d2nn", height=100, width=100, pixel_size_y=2e-6, pixel_size_x=2e-6)

    with pytest.raises(ValueError, match="physical mapping"):
        validate_sampling_transition(source, target, operation="generic_resize")


def test_aperture_sampling_requires_physical_source_extent():
    source = PlaneSampling("propagated", height=200, width=200, pixel_size_y=2.5e-6, pixel_size_x=2.5e-6)
    target = PlaneSampling("lightgen", height=100, width=100, pixel_size_y=2e-6, pixel_size_x=2e-6)

    transition = aperture_sampling(source, target, aperture_width=200e-6, aperture_height=200e-6)

    assert transition["operation"] == "physical_aperture_sampling"
    assert transition["source_fov_x"] == pytest.approx(500e-6)
    assert transition["target_fov_x"] == pytest.approx(200e-6)


def test_center_aperture_indices_map_200um_into_500um_source():
    source = PlaneSampling("slm", height=200, width=200, pixel_size_y=2.5e-6, pixel_size_x=2.5e-6)

    top, left, height, width = center_aperture_indices(source, aperture_width=200e-6, aperture_height=200e-6)

    assert (top, left, height, width) == (60, 60, 80, 80)


def test_sample_center_aperture_preserves_complex_field_and_target_shape():
    source = PlaneSampling("slm", height=20, width=20, pixel_size_y=2.5e-6, pixel_size_x=2.5e-6)
    field = torch.ones(2, 1, 20, 20, dtype=torch.complex64)
    sampled = sample_center_aperture(field, source, target_size=(10, 10), target_pixel_size=2e-6)

    assert tuple(sampled.shape) == (2, 1, 10, 10)
    assert torch.is_complex(sampled)
    assert torch.allclose(sampled.real, torch.ones_like(sampled.real))
    assert torch.allclose(sampled.imag, torch.zeros_like(sampled.imag))


def test_full_free_space_front_reports_all_macro_segments():
    from ldm_project.config import PhysicsConfig
    from ldm_project.optics import FrontOpticalPath

    physics = PhysicsConfig(front_path_model="free_space", physical_forward_contract=True)
    path = FrontOpticalPath(physics)
    slm = torch.ones(1, 1, 16, 16, dtype=torch.complex64)

    field, diagnostics = path.forward_field_with_diagnostics(slm, propagation_grid_size=64)

    assert tuple(field.shape[-2:]) == (64, 64)
    assert diagnostics["operation"] == "macro_free_space"
    assert diagnostics["segments_m"] == [0.350, 0.370, 0.025, 0.025, 0.125, 0.475]
    assert diagnostics["post_slm_distance_m"] == pytest.approx(0.625)
    assert len(diagnostics["segment_energy"]) == 4
    assert all(float(energy) > 0.0 for energy in diagnostics["segment_energy"])


def test_macro_segment_composition_matches_single_625mm_angular_spectrum_step():
    from ldm_project.optics import angular_spectrum_propagate

    size, pixel_size, wavelength = 64, 2.5e-6, 532e-9
    coordinates = (torch.arange(size, dtype=torch.float64) - size // 2) * pixel_size
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    field = torch.exp(
        1j * 2.0 * torch.pi * (4.0 * xx + 3.0 * yy) / (size * pixel_size)
    ).to(torch.complex128)[None, None]
    segmented = field
    for distance in (0.025, 0.125, 0.475):
        segmented = angular_spectrum_propagate(segmented, distance, wavelength, pixel_size)
    equivalent = angular_spectrum_propagate(field, 0.625, wavelength, pixel_size)

    assert torch.allclose(segmented, equivalent, atol=1e-9, rtol=1e-9)


def test_free_space_back_path_reports_lens_magnification_and_camera_sampling():
    from ldm_project.config import PhysicsConfig
    from ldm_project.optics import BackOpticalPath

    path = BackOpticalPath(PhysicsConfig(back_path_model="free_space", physical_forward_contract=True))
    chip = torch.ones(1, 1, 16, 16, dtype=torch.complex64)

    outputs = path(chip, input_pixel_size=2e-6, output_size=(20, 20), output_pixel_size=2.5e-6)

    assert tuple(outputs["camera_intensity_raw"].shape[-2:]) == (20, 20)
    assert outputs["camera_magnification"] == pytest.approx(-2.5)
    assert outputs["camera_sampling"]["pixel_size_x"] == pytest.approx(2.5e-6)
    assert outputs["camera_sampling"]["fov_x"] == pytest.approx(50e-6)
    assert outputs["camera_field_available"] is True


def test_physical_forward_contract_rejects_non_identity_ols_and_wrong_d2nn_geometry():
    from ldm_project.config import D2NNConfig, OneShotConfig, PhysicsConfig, TrainingConfig, validate_physical_forward_contract

    config = TrainingConfig(
        one_shot=OneShotConfig(
            physical_forward_contract=True,
            ols_kind="tanh",
            d2nn=D2NNConfig(grid_size=200, pixel_size=2.5e-6, layer_spacing=0.25e-3),
        )
    )

    with pytest.raises(ValueError, match="identity OLS"):
        validate_physical_forward_contract(config, PhysicsConfig(front_path_model="free_space", back_path_model="free_space"))


def test_identity_ols_is_complex_linear():
    from ldm_project.one_shot_optical_layers import _apply_complex_ols

    e1 = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    e2 = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    a, b = 0.37, -1.2
    actual = _apply_complex_ols(a * e1 + b * e2, "identity")

    assert torch.equal(actual, a * _apply_complex_ols(e1, "identity") + b * _apply_complex_ols(e2, "identity"))


def test_v52_physical_config_loads_with_contract_metadata():
    from ldm_project.config import load_training_config

    config = load_training_config("configs/v1/diagnostic/validate_v52_physical_forward_model.yaml")
    assert config.one_shot.ols_kind == "identity"
    assert config.one_shot.d2nn is not None
    assert config.one_shot.d2nn.grid_size == 100
    assert config.one_shot.d2nn.pixel_size == pytest.approx(2.0e-6)
    assert config.one_shot.physical_forward_contract is False
