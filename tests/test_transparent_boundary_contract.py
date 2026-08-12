from __future__ import annotations

import torch

from ldm_project.configuration.schema import D2NNConfig
from ldm_project.configuration.schema import OneShotConfig, PhysicsConfig
from ldm_project.student.formal_contract import validate_formal_student_config
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from ldm_project.student.experimental.transparent_phase import (
    TransparentIdentityDiffractiveStack,
)


def test_boundary_contract_is_explicit_and_backward_compatible() -> None:
    assert D2NNConfig().exterior_boundary == "finite_crop"
    assert D2NNConfig().transparent_propagation_method == "periodic_asm"
    assert D2NNConfig().transparent_fft_grid_size is None
    assert D2NNConfig().transparent_edge_guard_fraction == 0.05
    assert D2NNConfig(exterior_boundary="transparent_identity").exterior_boundary == (
        "transparent_identity"
    )


def test_transparent_transfer_is_identity_outside_active_window() -> None:
    stack = TransparentIdentityDiffractiveStack(
        num_layers=1,
        grid_size=4,
        layer_spacing=1.0e-6,
        wavelength=532.0e-9,
        pixel_size=2.0e-6,
        propagation_grid_size=8,
        phase_init="small",
    )
    transfers = stack.phase_transfers()
    top = left = 2
    outside = torch.ones_like(transfers[0], dtype=torch.bool)
    outside[..., top : top + 4, left : left + 4] = False
    assert torch.allclose(transfers[0][outside], torch.ones_like(transfers[0][outside]))
    assert torch.any((transfers[0][..., top : top + 4, left : left + 4] - 1).abs() > 0)


def test_transparent_stack_retains_full_grid_without_crop() -> None:
    stack = TransparentIdentityDiffractiveStack(
        num_layers=2,
        grid_size=4,
        layer_spacing=1.0e-4,
        wavelength=532.0e-9,
        pixel_size=2.0e-6,
        propagation_grid_size=8,
        phase_init="small",
    )
    field = torch.ones(2, 1, 4, 4, dtype=torch.complex64)
    output, layers, diagnostics = stack(field)
    assert output.shape == (2, 1, 8, 8)
    assert layers.shape == (2, 2, 1, 8, 8)
    assert torch.allclose(diagnostics["crop_retention"], torch.ones(2, 2))
    assert torch.all(diagnostics["propagation_energy_ratio"] > 0.999)


def test_padded_transparent_stack_crops_only_back_to_state_grid() -> None:
    stack = TransparentIdentityDiffractiveStack(
        num_layers=2,
        grid_size=4,
        layer_spacing=1.0e-5,
        wavelength=532.0e-9,
        pixel_size=2.0e-6,
        propagation_grid_size=8,
        phase_init="small",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=16,
    )
    field = torch.ones(2, 1, 4, 4, dtype=torch.complex64)
    output, layers, diagnostics = stack(field)
    assert output.shape == (2, 1, 8, 8)
    assert layers.shape == (2, 2, 1, 8, 8)
    assert stack.boundary_contract == "ideal_transparent_thin_phase_v2"
    assert diagnostics["transparent_propagation_method"] == "padded_band_limited_asm"
    assert diagnostics["transparent_state_grid_size"].item() == 8
    assert diagnostics["transparent_fft_grid_size"].item() == 16
    for key in (
        "input_energy",
        "padded_input_energy",
        "propagated_full_energy",
        "cropped_energy",
        "retained_energy",
        "discarded_energy",
        "discarded_energy_ratio",
        "guard_band_energy_ratio",
    ):
        assert diagnostics[key].shape == (2, 2), key
        assert torch.isfinite(diagnostics[key]).all(), key
    assert torch.allclose(
        diagnostics["cropped_energy"], diagnostics["retained_energy"]
    )
    assert torch.all(diagnostics["discarded_energy"] >= 0)


def test_padded_transparent_grid_requires_explicit_fft_size() -> None:
    try:
        TransparentIdentityDiffractiveStack(
            num_layers=1,
            grid_size=4,
            layer_spacing=1.0e-5,
            wavelength=532.0e-9,
            pixel_size=2.0e-6,
            propagation_grid_size=8,
            transparent_propagation_method="padded_band_limited_asm",
        )
    except ValueError as exc:
        assert "transparent_fft_grid_size" in str(exc)
    else:
        raise AssertionError("padded method accepted an unspecified FFT grid")


def test_formal_contract_rejects_transparent_boundary() -> None:
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        d2nn=D2NNConfig(
            grid_size=4,
            propagation_grid_size=8,
            exterior_boundary="transparent_identity",
        ),
    )
    try:
        validate_formal_student_config(PhysicsConfig(physical_forward_contract=True), config)
    except ValueError as exc:
        assert "exterior_boundary" in str(exc)
    else:
        raise AssertionError("formal contract accepted transparent diagnostic boundary")


def test_formal_contract_rejects_padded_transparent_settings_even_with_finite_crop() -> None:
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        d2nn=D2NNConfig(
            grid_size=4,
            propagation_grid_size=8,
            transparent_propagation_method="padded_band_limited_asm",
            transparent_fft_grid_size=16,
        ),
    )
    try:
        validate_formal_student_config(PhysicsConfig(physical_forward_contract=True), config)
    except ValueError as exc:
        assert "transparent_propagation_method" in str(exc)
    else:
        raise AssertionError("formal contract accepted padded transparent settings")


def test_transparent_boundary_keeps_physical_relay_but_routes_experimental() -> None:
    physics = PhysicsConfig(physical_forward_contract=True)
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        d2nn=D2NNConfig(
            grid_size=4,
            propagation_grid_size=8,
            exterior_boundary="transparent_identity",
        ),
    )
    model = OneShotOpticalGenerator(physics, config, image_size=8)
    assert model.__class__.__name__ == "_LegacyDiagnosticOneShotOpticalGenerator"
    assert model.physics.physical_forward_contract is True

