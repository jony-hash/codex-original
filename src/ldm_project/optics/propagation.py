"""Scalar coherent-field propagation operators.

This module deliberately contains no model, training, or artifact imports.  Its
operators work on the supplied physical sampling grid; changing a tensor shape
is therefore not part of propagation.
"""
from __future__ import annotations

import math

import torch


def normalize_intensity(
    field: torch.Tensor,
    normalize_percentile: float = 100.0,
    clamp: bool = True,
) -> torch.Tensor:
    """Convert a complex field to detector intensity before normalization."""
    return normalize_real_intensity(field.abs().pow(2), normalize_percentile, clamp=clamp)


def normalize_real_intensity(
    intensity: torch.Tensor,
    normalize_percentile: float = 100.0,
    clamp: bool = True,
) -> torch.Tensor:
    if normalize_percentile >= 100.0:
        scale = intensity.amax(dim=(-2, -1), keepdim=True)
    else:
        quantile = normalize_percentile / 100.0
        scale = torch.quantile(intensity.flatten(-2), quantile, dim=-1, keepdim=True).unsqueeze(-1)
    normalized = intensity / scale.clamp_min(1e-6)
    return normalized.clamp(0.0, 1.0) if clamp else normalized


def frequency_grid(
    height: int,
    width: int,
    pixel_size: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftfreq(height, d=pixel_size, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=pixel_size, device=device, dtype=dtype)
    return torch.meshgrid(fy, fx, indexing="ij")


def fresnel_propagate(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if distance == 0.0:
        return field
    height, width = field.shape[-2:]
    fy, fx = frequency_grid(height, width, pixel_size, field.device, field.real.dtype)
    phase = torch.exp(-1j * math.pi * wavelength * distance * (fx.pow(2) + fy.pow(2))).to(field.dtype)
    return torch.fft.ifft2(torch.fft.fft2(field) * phase)


def angular_spectrum_propagate(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if distance == 0.0:
        return field
    height, width = field.shape[-2:]
    fy, fx = frequency_grid(height, width, pixel_size, field.device, field.real.dtype)
    frequency_radius = (wavelength * fx).pow(2) + (wavelength * fy).pow(2)
    propagating = frequency_radius <= 1.0
    phase_root = (1.0 - frequency_radius).clamp_min(0.0).sqrt()
    transfer = torch.exp(1j * 2.0 * math.pi * distance * phase_root / wavelength).to(field.dtype)
    return torch.fft.ifft2(torch.fft.fft2(field) * torch.where(propagating, transfer, torch.zeros_like(transfer)))


def band_limited_angular_spectrum_propagate(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
) -> torch.Tensor:
    """ASM with a transfer-function sampling band limit on the input grid."""
    if distance == 0.0:
        return field
    height, width = field.shape[-2:]
    fy, fx = frequency_grid(height, width, pixel_size, field.device, field.real.dtype)
    wavelength = float(wavelength)
    radial = (wavelength * fx).pow(2) + (wavelength * fy).pow(2)
    propagating = radial <= 1.0
    root = (1.0 - radial).clamp_min(0.0).sqrt()
    transfer = torch.exp(1j * 2.0 * math.pi * distance * root / wavelength).to(field.dtype)
    delta_fx = 1.0 / (width * pixel_size)
    delta_fy = 1.0 / (height * pixel_size)
    denominator = root.clamp_min(torch.finfo(root.dtype).eps)
    phase_step_x = 2.0 * abs(distance) * wavelength * abs(fx) * delta_fx / denominator
    phase_step_y = 2.0 * abs(distance) * wavelength * abs(fy) * delta_fy / denominator
    sampled = torch.maximum(phase_step_x, phase_step_y) <= 1.0
    transfer = torch.where(propagating & sampled, transfer, torch.zeros_like(transfer))
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)


def apply_thin_lens(field: torch.Tensor, focal_length: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if focal_length == 0.0:
        return field
    height, width = field.shape[-2:]
    y = (torch.arange(height, device=field.device) - height / 2.0) * pixel_size
    x = (torch.arange(width, device=field.device) - width / 2.0) * pixel_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = torch.exp(-1j * math.pi * (xx.pow(2) + yy.pow(2)) / (wavelength * focal_length)).to(field.dtype)
    return field * phase


# Private compatibility for older internal callers; new code uses frequency_grid.
_frequency_grid = frequency_grid
