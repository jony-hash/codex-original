import math

import torch

from ldm_project.optics import (
    angular_spectrum_propagate,
    band_limited_angular_spectrum_propagate,
)


def _grid(size: int, pixel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    coords = (torch.arange(size, dtype=torch.float64) - size // 2) * pixel_size
    return torch.meshgrid(coords, coords, indexing="ij")


def test_plane_wave_preserves_amplitude_and_has_analytic_phase():
    size = 64
    pixel_size = 2e-6
    wavelength = 532e-9
    distance = 1.5e-3
    field = torch.ones(1, 1, size, size, dtype=torch.complex128)

    propagated = angular_spectrum_propagate(field, distance, wavelength, pixel_size)
    expected_phase = 2.0 * math.pi * distance / wavelength
    expected = torch.full_like(field, torch.exp(torch.tensor(1j * expected_phase, dtype=torch.complex128)))

    assert torch.allclose(propagated.abs(), torch.ones_like(propagated.abs()), atol=1e-10)
    assert torch.allclose(propagated, expected, atol=1e-10, rtol=1e-10)


def test_fft_bin_aligned_tilted_wave_preserves_spatial_frequency():
    size = 64
    pixel_size = 2e-6
    wavelength = 532e-9
    distance = 1.5e-3
    fx = 4.0 / (size * pixel_size)
    fy = -3.0 / (size * pixel_size)
    yy, xx = _grid(size, pixel_size)
    field = torch.exp(1j * 2.0 * math.pi * (fx * xx + fy * yy)).to(torch.complex128)[None, None]

    propagated = band_limited_angular_spectrum_propagate(field, distance, wavelength, pixel_size)
    spectrum = torch.fft.fftshift(torch.fft.fft2(propagated[0, 0]))
    peak = torch.where(spectrum.abs() == spectrum.abs().amax())

    assert int(peak[0][0]) == size // 2 - 3
    assert int(peak[1][0]) == size // 2 + 4
    assert torch.allclose(propagated.abs(), torch.ones_like(propagated.abs()), atol=1e-8)


def test_gaussian_beam_matches_paraxial_width_and_wrapped_phase():
    """The sampled ASM must reproduce a Gaussian beam's analytic complex field."""
    size = 256
    pixel_size = 2e-6
    wavelength = 532e-9
    distance = 1.0e-3
    waist = 30e-6
    k = 2.0 * math.pi / wavelength
    yy, xx = _grid(size, pixel_size)
    radius_sq = xx.square() + yy.square()
    field = torch.exp(-radius_sq / waist**2).to(torch.complex128)[None, None]

    propagated = angular_spectrum_propagate(field, distance, wavelength, pixel_size)[0, 0]
    rayleigh = math.pi * waist**2 / wavelength
    waist_z = waist * math.sqrt(1.0 + (distance / rayleigh) ** 2)
    curvature = distance * (1.0 + (rayleigh / distance) ** 2)
    gouy = math.atan(distance / rayleigh)
    expected = (
        (waist / waist_z)
        * torch.exp(-radius_sq / waist_z**2)
        * torch.exp(1j * (k * distance - gouy + k * radius_sq / (2.0 * curvature)))
    )
    valid = expected.abs().square() >= expected.abs().amax().square() * 1.0e-6
    amplitude_error = (propagated.abs()[valid] - expected.abs()[valid]).norm() / expected.abs()[valid].norm()
    wrapped_phase_error = torch.angle(propagated[valid] * expected[valid].conj()).abs().mean()

    assert float(amplitude_error) <= 1.0e-3
    assert float(wrapped_phase_error) <= 1.0e-2


def test_circular_aperture_matches_direct_rayleigh_sommerfeld_reference():
    """A small near-field aperture is compared against a direct RS sum."""
    size = 96
    pixel_size = 1e-6
    wavelength = 532e-9
    distance = 200e-6
    radius = 10e-6
    k = 2.0 * math.pi / wavelength
    yy, xx = _grid(size, pixel_size)
    aperture = (xx.square() + yy.square() <= radius**2).to(torch.complex128)
    propagated = angular_spectrum_propagate(
        aperture[None, None], distance, wavelength, pixel_size
    )[0, 0]

    roi_start, roi_stop = size // 2 - 12, size // 2 + 12
    reference = torch.empty((roi_stop - roi_start, roi_stop - roi_start), dtype=torch.complex128)
    for out_y, y_index in enumerate(range(roi_start, roi_stop)):
        for out_x, x_index in enumerate(range(roi_start, roi_stop)):
            separation = torch.sqrt(
                (xx - xx[y_index, x_index]).square()
                + (yy - yy[y_index, x_index]).square()
                + distance**2
            )
            reference[out_y, out_x] = (
                torch.sum(
                    aperture
                    * distance
                    / (1j * wavelength * separation.square())
                    * torch.exp(1j * k * separation)
                )
                * pixel_size**2
            )
    central = propagated[roi_start:roi_stop, roi_start:roi_stop].abs().square()
    reference_intensity = reference.abs().square()
    central = central / central.amax()
    reference_intensity = reference_intensity / reference_intensity.amax()
    relative_l2 = (central - reference_intensity).norm() / reference_intensity.norm()
    assert float(relative_l2) <= 0.01

    def first_local_minimum(profile: torch.Tensor) -> int:
        for index in range(2, profile.numel() - 1):
            if profile[index] <= profile[index - 1] and profile[index] <= profile[index + 1]:
                return index
        raise AssertionError("no diffraction minimum found")

    asm_minimum = first_local_minimum(propagated.abs().square()[size // 2, size // 2:])
    rs_axis = reference.abs().square()[reference.shape[0] // 2]
    rs_minimum = first_local_minimum(rs_axis)
    assert abs(asm_minimum - rs_minimum) <= 1
