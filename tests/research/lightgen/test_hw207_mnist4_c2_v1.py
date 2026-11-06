import importlib.util
import sys
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "scripts/research/lightgen/hw207_mnist4_c2_v1.py"
spec = importlib.util.spec_from_file_location("c2", path)
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)
sys.modules.setdefault("hw207_mnist4_c2_v1", c2)
pupil_spec = importlib.util.spec_from_file_location(
    "c2_pupil", path.with_name("hw207_mnist4_c2_pupil_v1.py"))
c2_pupil = importlib.util.module_from_spec(pupil_spec)
pupil_spec.loader.exec_module(c2_pupil)
rotation_spec = importlib.util.spec_from_file_location(
    "c2_rotation_exact", path.with_name("hw207_mnist4_c2_rotation_exact_v1.py"))
c2_rotation = importlib.util.module_from_spec(rotation_spec)
rotation_spec.loader.exec_module(c2_rotation)
grating_spec = importlib.util.spec_from_file_location(
    "c2_grating", path.with_name("hw207_mnist4_c2_grating_v1.py"))
c2_grating = importlib.util.module_from_spec(grating_spec)
grating_spec.loader.exec_module(c2_grating)
psf_spec = importlib.util.spec_from_file_location(
    "c2_psf", path.with_name("hw207_mnist4_c2_psf_v1.py"))
c2_psf = importlib.util.module_from_spec(psf_spec)
psf_spec.loader.exec_module(c2_psf)
power_spec = importlib.util.spec_from_file_location(
    "c2_pupil_energy", path.with_name("hw207_mnist4_c2_pupil_energy_v1.py"))
c2_power = importlib.util.module_from_spec(power_spec)
power_spec.loader.exec_module(c2_power)
sys.modules.setdefault("hw207_mnist4_c0_v1", importlib.import_module("scripts.research.lightgen.hw207_mnist4_c0_v1"))
sys.modules.setdefault("hw207_mnist4_c1_v1", importlib.import_module("scripts.research.lightgen.hw207_mnist4_c1_v1"))
shape_spec = importlib.util.spec_from_file_location(
    "c2_shape", path.with_name("hw207_mnist4_c2_shape_v1.py"))
c2_shape = importlib.util.module_from_spec(shape_spec)
shape_spec.loader.exec_module(c2_shape)
readout_spec = importlib.util.spec_from_file_location(
    "c2_readout", path.with_name("hw207_mnist4_c2_readout_v1.py"))
c2_readout = importlib.util.module_from_spec(readout_spec)
readout_spec.loader.exec_module(c2_readout)
sys.modules.setdefault("hw207_mnist4_c2_shape_v1", c2_shape)
shape_population_spec = importlib.util.spec_from_file_location(
    "c2_shape_population", path.with_name("hw207_mnist4_c2_shape_population_v1.py"))
c2_shape_population = importlib.util.module_from_spec(shape_population_spec)
shape_population_spec.loader.exec_module(c2_shape_population)


def test_padded_field_keeps_off_window_light_and_mirror_area():
    pattern = torch.ones((1, 26, 26))
    field = c2_pupil.padded_axis_field(pattern, 1.0, 1.0, 1.0, 400.0, (20.0, 0.0))
    assert field[0, :, 300:].sum() > 0
    assert field.sum().item() == pytest.approx(26 ** 2 * 7.56 ** 2, rel=1e-12)
    central = field[:, 100:300, 100:300]
    assert central.sum() < field.sum()


def test_padded_field_rejects_invalid_grid():
    with pytest.raises(ValueError):
        c2_pupil.padded_axis_field(torch.zeros((1, 26, 26)), 1, .92, 0.3, 400)


def test_padded_producer_through_pupil_conserves_or_rejects_power():
    pattern = torch.zeros((1, 26, 26))
    pattern[:, 12:14, 12:14] = 1
    field = c2_pupil.padded_axis_field(pattern, 1, 1, 2, 400).to(torch.complex128)
    output = c2.coherent_pupil(field, 2, 0.1)
    assert 0 < output.abs().square().sum() < field.abs().square().sum()
    torch.testing.assert_close(c2.coherent_pupil(field, 2, 1).abs().square().sum(),
                               field.abs().square().sum())
    zero = c2_pupil.padded_axis_field(torch.zeros_like(pattern), 1, .92, 2, 400)
    assert c2.coherent_pupil(zero.to(torch.complex128), 2, .1).count_nonzero() == 0


def test_rotated_polygon_overlap_matches_axis_and_conserves_isolated_area():
    pattern = torch.zeros((1, 26, 26))
    pattern[0, 12, 13] = 1
    matrix = c2_rotation.rotated_coverage_matrix(26, 1, .92, 0)
    actual = c2_rotation.rotated_intensity(pattern, 1, matrix)
    expected = c2.axis_aligned_pixel_integrals(pattern, 1, .92)["mean_intensity"]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    rotated = c2_rotation.rotated_coverage_matrix(26, 1, .92, .17)
    assert (c2_rotation.rotated_intensity(pattern, 1, rotated).sum() * 4).item() == pytest.approx(
        7.56 ** 2 * .92, rel=1e-12)
    many = torch.arange(28 * 28).reshape(1, 28, 28).remainder(3).eq(0).float()
    mag = 200 / (28 * 7.56)
    for sign in (1, -1):
        matrix = c2_rotation.rotated_coverage_matrix(28, sign * mag, .92, 0)
        expected = c2.axis_aligned_pixel_integrals(many, sign * mag, .92)["mean_intensity"]
        torch.testing.assert_close(c2_rotation.rotated_intensity(many, sign * mag, matrix),
                                   expected, atol=1e-11, rtol=1e-11)


def test_grating_fundamental_modulation_analytic_reference():
    x = torch.arange(100, dtype=torch.float64) * 2 - 99
    image = (2 + .6 * torch.cos(2 * torch.pi * x / 32))[None, :].expand(100, -1)
    assert c2_grating.modulation(image, 8, 4, "vertical", 200) == pytest.approx(.3, abs=1e-12)
    assert c2_grating.modulation(image, 8, 4, "horizontal", 200) == pytest.approx(0, abs=1e-12)


def test_spatial_pupil_reference_impulse_and_defocus_limit():
    radius = torch.tensor([0., 1., 4., 12.], dtype=torch.float64)
    exact = c2_psf.radial_psf(radius, .1, .532, 0)
    assert exact[0].real.item() == pytest.approx(torch.pi * .1 ** 2, abs=1e-14)
    nearby = c2_psf.radial_psf(radius, .1, .532, 1e-5, quadrature=2049)
    torch.testing.assert_close(nearby.real, exact.real, atol=2e-8, rtol=1e-5)
    impulse = torch.zeros((1, 64, 64), dtype=torch.complex128)
    impulse[0, 32, 32] = 1
    output = c2_psf.pupil_convolve(impulse, 1, .1)
    transfer = c2_psf.pupil_transfer(64, 1, .1, dtype=torch.float64)
    torch.testing.assert_close(c2_psf.pupil_convolve(impulse, 1, .1, transfer=transfer), output)
    torch.testing.assert_close(output[0, 32, 32], exact[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(output[0, 32, 36], exact[2], atol=1e-12, rtol=1e-12)


def test_pupil_energy_off_single_and_monotonic_aperture():
    pattern = torch.zeros((2, 26, 26))
    pattern[1, 12, 13] = 1
    small = c2_power.circular_pupil_energy(pattern, 1, .92, .03, 200)
    large = c2_power.circular_pupil_energy(pattern, 1, .92, .1, 200)
    assert small[0] == large[0] == 0
    assert 0 < small[1] < large[1] < 7.56 ** 2 * .92
    shifted = pattern.roll(3, 1)
    torch.testing.assert_close(c2_power.circular_pupil_energy(shifted, 1, .92, .1, 200), large)


def test_shape_proxy_detects_broken_stroke():
    ref = torch.zeros((7, 7), dtype=torch.bool)
    ref[3, 1:6] = True
    broken = ref.float()
    broken[3, 3] = 0
    result = c2_shape.metrics(ref, broken)
    assert result["reference_components"] == 1 and result["candidate_components"] == 2
    assert result["thin_pixel_retention"] == pytest.approx(.8)
    assert c2_shape.metrics(ref, ref.float())["dice"] == 1
    original = torch.zeros((1, 28, 28))
    original[0, 14, 5:23] = 1
    assert torch.equal(c2_shape_population.original_thin(original),
                       sys.modules["hw207_mnist4_c0_v1"].map_image(original, True).bool())


def test_auxiliary_readout_real_three_step_updates():
    generator = torch.Generator().manual_seed(12)
    x = torch.randn((768, 8), generator=generator)
    y = torch.arange(768) % 4
    _, result = c2_readout.fit(x, y, x[:32], y[:32], 11, 0, max_steps=3)
    assert result["steps"] == 3 and result["updated"] == {"weight": True, "bias": True}


def test_geometry_and_binary_encoding():
    assert c2.geometry(26, 1)["array_width_um"] == pytest.approx(196.56)
    assert c2.geometry(28, 200 / (28 * 7.56))["array_width_um"] == pytest.approx(200)
    image = torch.zeros((1, 28, 28))
    image[:, 3:12, 5:10] = 1
    for method in ("nearest", "area"):
        pattern = c2.encode(image, 26, method)
        assert pattern.shape == (1, 26, 26)
        assert set(pattern.unique().tolist()) <= {0, 1}


def test_encoding_uses_fixed_physical_field():
    image = torch.zeros((1, 28, 28))
    image[0, 0, 0] = 1
    magnification = 200 / (28 * 7.56)
    for method in ("nearest", "area"):
        assert torch.equal(c2.encode(image, 28, method, magnification=magnification), image)
    mapped = c2.encode(torch.ones_like(image), 26, "area", magnification=1.0)
    assert torch.equal(mapped, torch.ones_like(mapped))
    assert c2.geometry(26, 1.0)["array_width_um"] < c2.TARGET_WIDTH_UM


def test_finite_array_and_orientation():
    pattern = torch.zeros((1, 26, 26))
    pattern[0, 2, 4] = 1
    positive, _ = c2.target_field(pattern, 1, 4)
    negative, _ = c2.target_field(pattern, -1, 4)
    torch.testing.assert_close(negative, positive.flip((-2, -1)))
    zeros, _ = c2.target_field(torch.zeros_like(pattern), 1, 2)
    assert zeros.count_nonzero() == 0


def test_area_fraction_and_finite_boundary():
    all_on = torch.ones((1, 26, 26))
    full, _ = c2.target_field(all_on, 1, 8, 1.0)
    filled, _ = c2.target_field(all_on, 1, 8, 0.92)
    assert full.mean().item() == pytest.approx((196.56 / 200) ** 2, abs=0.01)
    assert 0 < filled.mean() < full.mean()


def test_subpixel_energy_uses_physical_area_and_magnification():
    pattern = torch.ones((1, 28, 28))
    magnification = 200 / (28 * 7.56)
    samples, _ = c2.target_plane_samples(pattern, magnification, 8, 1.0)
    ledger = c2.energy_account(pattern, samples, magnification, 1.0)
    torch.testing.assert_close(ledger["target_window_um2"], ledger["incident_um2"],
                               rtol=2e-4, atol=0)
    assert ledger["target_pixel_area_um2"] == pytest.approx(0.25 ** 2)


def test_coherent_pupil_uniform_and_rejects_high_frequency():
    uniform = torch.ones((1, 32, 32), dtype=torch.complex128)
    output = c2.coherent_pupil(uniform, 2.0, 0.1)
    torch.testing.assert_close(output, uniform)
    high = ((-1.0) ** torch.arange(32))[None, None, :].expand(1, 32, 32).to(torch.complex128)
    filtered = c2.coherent_pupil(high, 2.0, 0.1)
    assert filtered.abs().max() < 1e-12
    defocused = c2.coherent_pupil(uniform, 2.0, 0.1, defocus_um=5)
    torch.testing.assert_close(defocused.abs(), uniform.abs())


def test_calibration_and_intensity_metrics():
    patterns = c2.calibration_patterns(26)
    assert len(patterns) == 13
    assert patterns["all_off"].sum() == 0
    assert patterns["all_on"].sum() == 26 * 26
    assert patterns["isolated"].sum() == 1
    assert patterns["adjacent"].sum() == 2
    assert not torch.equal(patterns["asymmetric"], patterns["asymmetric"].T)
    image = patterns["asymmetric"][None]
    scores = c2.intensity_metrics(image, image * 0.5)
    assert scores["nrmse"].item() == pytest.approx(0.5)
    assert scores["normalized_correlation"].item() == pytest.approx(1)
    assert scores["candidate_sum"].item() == pytest.approx(scores["reference_sum"].item() * 0.5)


def test_frozen_readout_preserves_weights_and_normalizes_energy():
    model = torch.nn.Linear(10000, 4)
    model.requires_grad_(False)
    before = model.weight.detach().clone()
    head = {"scale": 2.0, "models": [{"seed": 11, "model": model}]}
    intensity = torch.zeros((2, 100, 100))
    intensity[1, 25:30, 30:35] = 4
    scores, zero = c2.frozen_scores(intensity, head, normalize=True)
    assert zero.tolist() == [True, False]
    assert scores[0]["prediction"].shape == (2,)
    assert not scores[0]["margin"].requires_grad
    assert torch.equal(before, model.weight)


def test_representative_image_has_independent_storage(tmp_path):
    images = torch.zeros((256, 100, 100))
    representative = images[:1].clone()
    path = tmp_path / "representative.pt"
    torch.save({"representative_intensity": representative}, path)
    assert path.stat().st_size < 100_000


def test_exact_axis_aligned_energy_and_unresolved_field():
    pattern = torch.ones((1, 28, 28))
    magnification = 200 / (28 * 7.56)
    result = c2.axis_aligned_pixel_integrals(pattern, magnification, 0.92)
    energy = result["mean_intensity"].sum() * 4
    expected = 28 * 28 * 7.56 ** 2 * 0.92
    assert energy.item() == pytest.approx(expected, rel=1e-12)
    assert result["mean_field"].square().sum() < result["mean_intensity"].sum()


def test_exact_axis_orientation_and_subpixel_reference():
    pattern = torch.zeros((1, 26, 26))
    pattern[0, 2, 4] = 1
    positive = c2.axis_aligned_pixel_integrals(pattern, 1, 1.0)["mean_intensity"]
    negative = c2.axis_aligned_pixel_integrals(pattern, -1, 1.0)["mean_intensity"]
    torch.testing.assert_close(negative, positive.flip((-2, -1)))
    samples, _ = c2.target_plane_samples(pattern, 1, 32, 1.0)
    numerical = torch.nn.functional.avg_pool2d(samples[:, None].square(), 32)[:, 0]
    assert (positive - numerical).abs().sum().item() * 4 / (7.56 ** 2) < 0.02
