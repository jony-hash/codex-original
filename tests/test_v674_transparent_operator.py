from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from ldm_project.student.experimental.transparent_phase import TransparentIdentityDiffractiveStack
from ldm_project.student.experimental.model import _LegacyDiagnosticOneShotOpticalGenerator
from tools.diagnose_v674_transparent_operator import _set_phase_baseline, _transfer_audit


def test_v674_full_domain_stack_keeps_full_grid_between_layers() -> None:
    stack = TransparentIdentityDiffractiveStack(
        num_layers=2,
        grid_size=4,
        layer_spacing=1.5e-3,
        wavelength=532e-9,
        pixel_size=2e-6,
        propagation_grid_size=8,
        phase_init="small",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=8,
    )
    with torch.no_grad():
        stack.phase.zero_()
    field = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    output, layers, diagnostics = stack(field)
    assert output.shape[-2:] == (8, 8)
    assert layers.shape[-2:] == (8, 8)
    assert torch.allclose(diagnostics["discarded_energy_ratio"], torch.zeros_like(diagnostics["discarded_energy_ratio"]))


def test_v674_transparent_exterior_is_identity_on_full_grid() -> None:
    stack = TransparentIdentityDiffractiveStack(
        num_layers=1,
        grid_size=4,
        layer_spacing=1.5e-3,
        wavelength=532e-9,
        pixel_size=2e-6,
        propagation_grid_size=8,
        phase_init="small",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=8,
    )
    transfer = stack.phase_transfers()[0, 0]
    mask = torch.ones_like(transfer, dtype=torch.bool)
    mask[2:6, 2:6] = False
    assert torch.equal(transfer[mask], torch.ones_like(transfer[mask]))


def test_v674_phase_baseline_is_reproducible() -> None:
    first = TransparentIdentityDiffractiveStack(
        2, 4, 1.5e-3, 532e-9, 2e-6, 8, "small",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=8,
    )
    second = TransparentIdentityDiffractiveStack(
        2, 4, 1.5e-3, 532e-9, 2e-6, 8, "uniform_pi",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=8,
    )
    first_model = nn.Module()
    second_model = nn.Module()
    first_model.encoder, first_model.decoder = first, first
    second_model.encoder, second_model.decoder = second, second
    _set_phase_baseline(first_model, "small")
    _set_phase_baseline(second_model, "small")
    assert torch.equal(first.phase, second.phase)


def test_v674_transfer_audit_uses_shared_physical_frequency_bins() -> None:
    report = _transfer_audit(candidate_grid=12, reference_grid=16)
    assert report["shared_bin_count"] == 4
    assert report["transfer_phase_max_error_rad_on_shared_propagating_bins"] < 1.0e-10


def test_v674_artifact_view_crops_only_after_internal_field_is_available() -> None:
    owner = SimpleNamespace(
        _d2nn={"exterior_boundary": "transparent_identity", "grid_size": 4}
    )
    field = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).to(torch.complex64)
    view = _LegacyDiagnosticOneShotOpticalGenerator._transparent_active_view(owner, field)
    assert view.shape[-2:] == (4, 4)
    assert torch.equal(view, field[..., 2:6, 2:6])
