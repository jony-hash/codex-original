"""Nonperiodic coherent circular-pupil reference using a spatial PSF."""
import math

import torch


def radial_psf(radius, cutoff, wavelength, defocus, quadrature=2049):
    """Evaluate 2*pi*integral rho*J0(2*pi*r*rho)*exp(i*kz) d-rho."""
    if cutoff <= 0 or wavelength <= 0 or cutoff * wavelength >= 1 or quadrature < 3:
        raise ValueError("Invalid propagating circular pupil")
    if defocus == 0:
        argument = 2 * math.pi * cutoff * radius
        result = torch.where(radius == 0, torch.full_like(radius, math.pi * cutoff**2),
                             cutoff * torch.special.bessel_j1(argument) / radius.clamp_min(1e-300))
        return torch.complex(result, torch.zeros_like(result))
    rho = torch.linspace(0, cutoff, quadrature, dtype=radius.dtype, device=radius.device)
    phase = 2 * math.pi * defocus / wavelength * torch.sqrt(1 - (wavelength * rho).square())
    factors = rho * torch.exp(1j * phase)
    result = []
    for chunk in radius.split(512):
        bessel = torch.special.bessel_j0(2 * math.pi * chunk[:, None] * rho[None, :])
        result.append(2 * math.pi * torch.trapezoid(bessel * factors, rho, dim=1))
    return torch.cat(result)


def pupil_transfer(n, pitch, cutoff, defocus=0., wavelength=.532, radial_step=None,
                   quadrature=2049, device="cpu", dtype=torch.float32):
    """FFT of the sampled continuous PSF on a doubled grid."""
    if n <= 0 or pitch <= 0:
        raise ValueError("Expected a positive grid and pitch")
    radial_step = pitch / 8 if radial_step is None else radial_step
    if radial_step <= 0:
        raise ValueError("Invalid PSF interpolation step")
    # A 2n grid turns FFT circular convolution into linear convolution for all
    # separations between the central field and its input support.
    m = 2 * n
    coordinate = torch.fft.fftfreq(m, d=1 / m, dtype=dtype, device=device) * pitch
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    radius = torch.sqrt(x.square() + y.square())
    nr = math.ceil(float(radius.max()) / radial_step) + 2
    radii = torch.arange(nr, device=device, dtype=dtype) * radial_step
    lookup = radial_psf(radii, cutoff, wavelength, defocus, quadrature)
    position = radius / radial_step
    lower = position.floor().long()
    weight = position - lower
    kernel = lookup[lower] * (1 - weight) + lookup[lower + 1] * weight
    return torch.fft.fft2(kernel)


def pupil_convolve(field, pitch, cutoff, defocus=0., wavelength=.532, radial_step=None,
                   quadrature=2049, transfer=None):
    """Linear convolution on a padded grid; input must have zero outer margins."""
    if field.ndim != 3 or field.shape[-1] != field.shape[-2] or not field.is_complex() or pitch <= 0:
        raise ValueError("Expected square complex fields and positive pitch")
    n = field.shape[-1]
    m = 2 * n
    if transfer is None:
        transfer = pupil_transfer(n, pitch, cutoff, defocus, wavelength, radial_step,
                                  quadrature, field.device, field.real.dtype)
    if transfer.shape != (m, m):
        raise ValueError("Pupil transfer shape does not match field")
    padded = torch.zeros((len(field), m, m), dtype=field.dtype, device=field.device)
    start = n // 2
    padded[:, start:start+n, start:start+n] = field
    result = torch.fft.ifft2(torch.fft.fft2(padded) * transfer) * pitch**2
    return result[:, start:start+n, start:start+n]
