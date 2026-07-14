import pytest
import torch
from pathlib import Path

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.optics import coherent_imaging_4f_field
from ldm_project.relay_reference import (
    collins_b0_relay_reference,
    direct_rayleigh_sommerfeld,
    four_f_direct_rayleigh_sommerfeld_reference,
    four_f_scaled_fresnel_reference,
    four_f_abcd,
)


CONFIG = "configs/validate_v55_coherent_relay.yaml"


def _relative_global_phase(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    phase = torch.angle((a.conj() * b).sum())
    return a * torch.exp(1j * phase).to(a.dtype)


def test_four_f_abcd_is_independent_and_exact():
    matrix = four_f_abcd(0.22321428571428573, 0.0892857142857143)
    assert matrix[0][0] == pytest.approx(-0.4)
    assert matrix[0][1] == pytest.approx(0.0, abs=1e-12)


def test_independent_collins_reference_is_physical_coordinate_based():
    torch.manual_seed(55)
    source = torch.randn(1, 1, 200, 200, dtype=torch.complex128)
    args = dict(
        magnification=-0.4,
        output_size=(100, 100),
        input_pixel_size=2.5e-6,
        output_pixel_size=2.0e-6,
        wavelength=532e-9,
        c_abcd=-(0.22321428571428573 + 0.0892857142857143) /
        (0.22321428571428573 * 0.0892857142857143),
        pupil_diameter=0.02375,
        pupil_focal_length=0.0892857142857143,
    )
    production = coherent_imaging_4f_field(source, **args)
    reference = collins_b0_relay_reference(source, **args)
    assert torch.isfinite(production.real).all() and torch.isfinite(production.imag).all()
    assert torch.isfinite(reference.real).all() and torch.isfinite(reference.imag).all()
    source = (Path(__file__).resolve().parents[1] / "src/ldm_project/relay_reference.py").read_text()
    assert "grid_sample" not in source


def test_independent_full_4f_references_are_finite():
    size = 12
    field = torch.ones(1, 1, size, size, dtype=torch.complex128)
    common = dict(
        f1=0.22321428571428573,
        f2=0.0892857142857143,
        wavelength=532e-9,
        input_pixel_size=500e-6 / size,
        output_size=(size, size),
        output_pixel_size=200e-6 / size,
        pupil_diameter=0.02375,
    )
    fresnel = four_f_scaled_fresnel_reference(field, **common)
    direct = four_f_direct_rayleigh_sommerfeld_reference(field, **common)
    assert torch.isfinite(fresnel.real).all() and torch.isfinite(fresnel.imag).all()
    assert torch.isfinite(direct.real).all() and torch.isfinite(direct.imag).all()


def test_direct_rs_plane_wave_is_finite_and_aperture_reference_is_nonzero():
    size = 12
    field = torch.ones(1, 1, size, size, dtype=torch.complex128)
    propagated = direct_rayleigh_sommerfeld(
        field, distance=200e-6, wavelength=532e-9, input_pixel_size=1e-6
    )
    assert torch.isfinite(propagated.real).all()
    assert torch.isfinite(propagated.imag).all()
    assert float(propagated.abs().amax()) > 0.0


def test_v55_reference_geometry_has_full_fov_and_pupil_contract():
    cfg = load_training_config(CONFIG)
    validate_physical_forward_contract(cfg, cfg.physics)
    macro = cfg.physics.macro_optics
    assert macro.front_lens1_clear_aperture >= 0.02375
    assert macro.back_relay_clear_aperture >= 0.0266
    assert cfg.physics.optics.pixel_size * 200 == pytest.approx(500e-6)
    assert cfg.one_shot.d2nn.pixel_size * 100 == pytest.approx(200e-6)
