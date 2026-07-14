"""Deterministic CPU checks for the v52 scalar propagation contract.

This tool is deliberately independent of training and checkpoints.  It compares
the sampled angular-spectrum implementation with analytic plane/Gaussian fields
and a small direct Rayleigh--Sommerfeld aperture reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ldm_project.optics import angular_spectrum_propagate, band_limited_angular_spectrum_propagate


def _grid(size: int, pixel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = (torch.arange(size, dtype=torch.float64) - size // 2) * pixel_size
    return torch.meshgrid(coordinates, coordinates, indexing="ij")


def _direct_rs_aperture(
    aperture: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    *,
    wavelength: float,
    distance: float,
    pixel_size: float,
    roi_start: int,
    roi_stop: int,
) -> torch.Tensor:
    wavenumber = 2.0 * math.pi / wavelength
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
                    * torch.exp(1j * wavenumber * separation)
                )
                * pixel_size**2
            )
    return reference


def run_checks() -> dict[str, object]:
    size, pixel_size, wavelength, distance = 256, 2e-6, 532e-9, 1.5e-3
    plane = torch.ones(1, 1, size, size, dtype=torch.complex128)
    propagated_plane = angular_spectrum_propagate(plane, distance, wavelength, pixel_size)
    plane_phase = 2.0 * math.pi * distance / wavelength
    expected_plane = torch.full_like(
        plane,
        torch.exp(torch.tensor(1j * plane_phase, dtype=torch.complex128)),
    )
    plane_error = float((propagated_plane - expected_plane).norm() / expected_plane.norm())

    frequency_x = 4.0 / (size * pixel_size)
    frequency_y = -3.0 / (size * pixel_size)
    yy, xx = _grid(size, pixel_size)
    tilted = torch.exp(1j * 2.0 * math.pi * (frequency_x * xx + frequency_y * yy)).to(torch.complex128)[None, None]
    propagated_tilted = band_limited_angular_spectrum_propagate(tilted, distance, wavelength, pixel_size)
    tilted_error = float((propagated_tilted.abs() - 1.0).abs().amax())
    tilted_spectrum = torch.fft.fftshift(torch.fft.fft2(propagated_tilted[0, 0]))
    tilted_peak = torch.where(tilted_spectrum.abs() == tilted_spectrum.abs().amax())
    tilted_peak_coordinates = [int(tilted_peak[0][0]), int(tilted_peak[1][0])]
    expected_tilted_peak = [size // 2 - 3, size // 2 + 4]

    waist, gaussian_distance = 30e-6, 1e-3
    radius_sq = xx.square() + yy.square()
    gaussian = torch.exp(-radius_sq / waist**2).to(torch.complex128)[None, None]
    propagated_gaussian = angular_spectrum_propagate(gaussian, gaussian_distance, wavelength, pixel_size)[0, 0]
    rayleigh = math.pi * waist**2 / wavelength
    waist_z = waist * math.sqrt(1.0 + (gaussian_distance / rayleigh) ** 2)
    curvature = gaussian_distance * (1.0 + (rayleigh / gaussian_distance) ** 2)
    expected_gaussian = (
        (waist / waist_z)
        * torch.exp(-radius_sq / waist_z**2)
        * torch.exp(
            1j
            * (
                2.0 * math.pi * gaussian_distance / wavelength
                - math.atan(gaussian_distance / rayleigh)
                + 2.0 * math.pi * radius_sq / (wavelength * 2.0 * curvature)
            )
        )
    )
    valid = expected_gaussian.abs().square() >= expected_gaussian.abs().amax().square() * 1e-6
    gaussian_amplitude_error = float(
        (propagated_gaussian.abs()[valid] - expected_gaussian.abs()[valid]).norm()
        / expected_gaussian.abs()[valid].norm()
    )
    gaussian_phase_error = float(torch.angle(propagated_gaussian[valid] * expected_gaussian[valid].conj()).abs().mean())

    aperture_size, aperture_dx, aperture_z, aperture_radius = 96, 1e-6, 200e-6, 10e-6
    aperture_yy, aperture_xx = _grid(aperture_size, aperture_dx)
    aperture = (aperture_xx.square() + aperture_yy.square() <= aperture_radius**2).to(torch.complex128)
    aperture_field = aperture[None, None]
    asm = angular_spectrum_propagate(aperture_field, aperture_z, wavelength, aperture_dx)[0, 0]
    roi_start, roi_stop = aperture_size // 2 - 12, aperture_size // 2 + 12
    reference = _direct_rs_aperture(
        aperture,
        aperture_xx,
        aperture_yy,
        wavelength=wavelength,
        distance=aperture_z,
        pixel_size=aperture_dx,
        roi_start=roi_start,
        roi_stop=roi_stop,
    )
    asm_intensity = asm[roi_start:roi_stop, roi_start:roi_stop].abs().square()
    reference_intensity = reference.abs().square()
    asm_intensity /= asm_intensity.amax()
    reference_intensity /= reference_intensity.amax()
    aperture_error = float((asm_intensity - reference_intensity).norm() / reference_intensity.norm())

    result = {
        "plane_wave_complex_relative_l2": plane_error,
        "tilted_wave_amplitude_max_error": tilted_error,
        "tilted_wave_peak_coordinates": tilted_peak_coordinates,
        "tilted_wave_expected_peak_coordinates": expected_tilted_peak,
        "gaussian_amplitude_relative_l2": gaussian_amplitude_error,
        "gaussian_wrapped_phase_mae_rad": gaussian_phase_error,
        "aperture_central_intensity_relative_l2": aperture_error,
        "thresholds": {
            "plane_wave_complex_relative_l2": 1e-5,
            "tilted_wave_amplitude_max_error": 1e-4,
            "gaussian_amplitude_relative_l2": 1e-3,
            "gaussian_wrapped_phase_mae_rad": 1e-2,
            "aperture_central_intensity_relative_l2": 1e-2,
        },
    }
    result["status"] = "pass" if (
        plane_error <= 1e-5
        and tilted_error <= 1e-4
        and tilted_peak_coordinates == expected_tilted_peak
        and gaussian_amplitude_error <= 1e-3
        and gaussian_phase_error <= 1e-2
        and aperture_error <= 1e-2
    ) else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()
    report = run_checks()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
