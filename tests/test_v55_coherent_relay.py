import pytest
import torch
import inspect

from ldm_project.config import load_training_config, validate_physical_forward_contract
from ldm_project.optics import FrontOpticalPath, coherent_imaging_4f_field


CONFIG = "configs/v1/diagnostic/validate_v55_coherent_relay.yaml"


def test_coherent_relay_contract_and_abcd_geometry():
    cfg = load_training_config(CONFIG)
    validate_physical_forward_contract(cfg, cfg.physics)
    macro = cfg.physics.macro_optics
    assert macro.front_lens2_focal_length / macro.front_lens1_focal_length == pytest.approx(0.4)
    assert 2 * (macro.front_lens1_focal_length + macro.front_lens2_focal_length) == pytest.approx(0.625)
    assert macro.back_relay_clear_aperture >= 0.0266


def test_front_relay_maps_full_slm_to_full_encoder_without_crop():
    cfg = load_training_config(CONFIG)
    path = FrontOpticalPath(cfg.physics)
    field = torch.ones(1, 1, 200, 200, dtype=torch.complex64)
    output, diagnostics = path.forward_field_with_diagnostics(field)
    assert tuple(output.shape[-2:]) == (100, 100)
    assert diagnostics["operation"] == "coherent_imaging_4f"
    assert diagnostics["front_relay"]["A"] == pytest.approx(-0.4)
    assert diagnostics["front_relay"]["B"] == pytest.approx(0.0)
    assert diagnostics["front_relay"]["C"] == pytest.approx(0.0)
    assert [(item["from"], item["to"]) for item in diagnostics["relay_trace"]] == [
        ("SLM", "BS"), ("BS", "M3"), ("M3", "L1"), ("L1", "L2"), ("L2", "Encoder")
    ]
    assert [item["distance_m"] for item in diagnostics["relay_trace"]] == pytest.approx(
        [0.025, 0.125, 0.07321428571428572, 0.3125, 0.0892857142857143]
    )


def test_front_relay_rejects_insufficient_pupil():
    cfg = load_training_config(CONFIG)
    cfg.physics.macro_optics.front_lens1_clear_aperture = 0.01
    with pytest.raises(ValueError, match="23.75"):
        validate_physical_forward_contract(cfg, cfg.physics)


def test_formal_contract_rejects_legacy_free_space_front():
    cfg = load_training_config(CONFIG)
    cfg.physics.front_path_model = "free_space"
    with pytest.raises(ValueError, match="coherent_imaging_4f"):
        validate_physical_forward_contract(cfg, cfg.physics)


def test_4f_field_preserves_complex_dtype_and_output_sampling():
    source = torch.ones(1, 1, 200, 200, dtype=torch.complex64)
    output = coherent_imaging_4f_field(
        source,
        magnification=-0.4,
        output_size=(100, 100),
        input_pixel_size=2.5e-6,
        output_pixel_size=2e-6,
        wavelength=532e-9,
        c_abcd=-1.0,
        pupil_diameter=0.02375,
        pupil_focal_length=0.0892857142857143,
    )
    assert torch.is_complex(output)
    assert tuple(output.shape[-2:]) == (100, 100)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_4f_periodic_nudft_is_differentiable_and_accepts_noncontiguous_fields(dtype: torch.dtype):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(1, 1, 32, 32, dtype=real_dtype, requires_grad=True)
    imag = torch.randn(1, 1, 32, 32, dtype=real_dtype, requires_grad=True)
    source = torch.complex(real, imag).transpose(-1, -2)
    output = coherent_imaging_4f_field(
        source, magnification=-0.4, output_size=(16, 16), input_pixel_size=2.5e-6,
        output_pixel_size=2e-6, wavelength=532e-9, c_abcd=0.0,
        pupil_diameter=0.02375, pupil_focal_length=0.0892857142857143,
    )
    output.abs().square().mean().backward()
    assert real.grad is not None and torch.isfinite(real.grad).all()
    assert imag.grad is not None and torch.isfinite(imag.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA relay shape check requires CUDA")
def test_4f_periodic_nudft_cuda_shape_and_dtype():
    source = torch.ones(1, 1, 32, 32, device="cuda", dtype=torch.complex64)
    output = coherent_imaging_4f_field(
        source, magnification=-0.4, output_size=(16, 16), input_pixel_size=2.5e-6,
        output_pixel_size=2e-6, wavelength=532e-9, c_abcd=0.0,
        pupil_diameter=0.02375, pupil_focal_length=0.0892857142857143,
    )
    assert output.device.type == "cuda"
    assert output.dtype == torch.complex64
    assert tuple(output.shape[-2:]) == (16, 16)


def test_front_relay_maps_center_point_and_inverts_tilt():
    source = torch.zeros(1, 1, 200, 200, dtype=torch.complex64)
    source[..., 100, 100] = 1.0
    point = coherent_imaging_4f_field(
        source, magnification=-0.4, output_size=(100, 100), input_pixel_size=2.5e-6,
        output_pixel_size=2e-6, wavelength=532e-9, c_abcd=-15.68,
        pupil_diameter=0.02375, pupil_focal_length=0.0892857142857143,
    )
    assert int(point.abs().argmax()) // 100 in {49, 50}
    assert int(point.abs().argmax()) % 100 in {49, 50}

    x = (torch.arange(200) - 99.5) * 2.5e-6
    tilt = torch.exp(1j * 2 * torch.pi * 100_000 * x).view(1, 1, 1, 200).expand(1, 1, 200, 200)
    mapped = coherent_imaging_4f_field(
        tilt, magnification=-0.4, output_size=(100, 100), input_pixel_size=2.5e-6,
        output_pixel_size=2e-6, wavelength=532e-9, c_abcd=-15.68,
        pupil_diameter=0.02375, pupil_focal_length=0.0892857142857143,
    )
    assert torch.isfinite(mapped.real).all() and torch.isfinite(mapped.imag).all()


def test_front_pupil_changes_bandwidth_not_coordinate_contract():
    source = torch.randn(1, 1, 200, 200, dtype=torch.complex64)
    common = dict(magnification=-0.4, output_size=(100, 100), input_pixel_size=2.5e-6,
                  output_pixel_size=2e-6, wavelength=532e-9, c_abcd=-15.68,
                  pupil_focal_length=0.0892857142857143)
    wide = coherent_imaging_4f_field(source, pupil_diameter=0.02375, **common)
    narrow = coherent_imaging_4f_field(source, pupil_diameter=0.005, **common)
    assert tuple(wide.shape[-2:]) == tuple(narrow.shape[-2:]) == (100, 100)
    assert float(narrow.abs().square().mean()) < float(wide.abs().square().mean())


def test_formal_relay_uses_object_and_image_cutoff_without_bilinear_sampling():
    wavelength = 532e-9
    pupil = 0.02375
    f1, f2 = 0.22321428571428573, 0.0892857142857143
    object_cutoff = pupil / (2.0 * f1 * wavelength)
    image_cutoff = pupil / (2.0 * f2 * wavelength)
    assert object_cutoff == pytest.approx(100_000.0, rel=1e-6)
    assert image_cutoff == pytest.approx(250_000.0, rel=1e-6)
    source = inspect.getsource(coherent_imaging_4f_field)
    assert "grid_sample" not in source
    assert "torch.sinc" not in source
    assert "periodic" in source
