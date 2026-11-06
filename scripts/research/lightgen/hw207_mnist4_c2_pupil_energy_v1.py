"""Circular-pupil power via the analytic Fourier transform of square mirrors."""
import math

import torch

from hw207_mnist4_c2_v1 import DMD_PITCH_UM, geometry


def circular_pupil_energy(pattern, magnification, fill_factor, cutoff, frequency_window_um):
    """Numerically integrate continuous-field spectral power inside a pupil.

    A larger frequency_window_um refines the 2-D frequency quadrature; it is
    not a physical relay dimension. This assumes coherent zero-phase ON mirrors.
    """
    if pattern.ndim != 3 or pattern.shape[-1] != pattern.shape[-2]:
        raise ValueError("Expected a batch of square mirror patterns")
    if not ((pattern == 0) | (pattern == 1)).all():
        raise ValueError("Mirror commands must be binary")
    if not 0 < fill_factor <= 1 or cutoff <= 0 or frequency_window_um <= 0:
        raise ValueError("Invalid fill, cutoff or frequency sampling")
    count = pattern.shape[-1]
    pitch = geometry(count, magnification)["pitch_um"]
    size = pitch * math.sqrt(fill_factor)
    n = math.ceil(cutoff * frequency_window_um)
    f = torch.arange(-n, n+1, dtype=torch.float64, device=pattern.device) / frequency_window_um
    centers = (torch.arange(count, dtype=torch.float64, device=pattern.device) + .5 - count/2) * pitch
    phase = torch.exp(-2j * math.pi * f[:, None] * centers[None, :])
    commands = pattern.flip((-2, -1)) if magnification < 0 else pattern
    coefficients = phase @ commands.to(torch.complex128) @ phase.T
    envelope = (size**2 / abs(magnification)) * torch.sinc(size * f)
    coefficients = coefficients * envelope[None, :, None] * torch.sinc(size * f)[None, None, :]
    pupil = f[:, None].square() + f[None, :].square() <= cutoff**2
    return (coefficients.abs().square() * pupil).sum((-2, -1)) / frequency_window_um**2
