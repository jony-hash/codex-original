"""Independent v2 angular-spectrum propagation implementations."""
from __future__ import annotations

import math

import torch

from ..contracts import ComplexField, OperationProvenance


_PRODUCTION_TRANSFER_CACHE: dict[tuple[tuple[int, int], tuple[float, float], float, float, str, str], torch.Tensor] = {}


def clear_production_transfer_cache() -> None:
    """Clear cached production transfer functions, primarily for isolated tests."""
    _PRODUCTION_TRANSFER_CACHE.clear()


def _transfer_function(
    shape: tuple[int, int],
    *,
    pixel_pitch_m: tuple[float, float],
    wavelength_m: float,
    distance_m: float,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    # The propagation phase can be thousands of radians. Build the sampled
    # frequencies and phase in float64 even for complex64 fields, then cast the
    # completed transfer function once to the field's complex precision.
    fy = torch.fft.fftfreq(shape[0], d=pixel_pitch_m[0], device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(shape[1], d=pixel_pitch_m[1], device=device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    argument = 1.0 - (wavelength_m * grid_x).square() - (wavelength_m * grid_y).square()
    phase = (2.0 * math.pi * distance_m / wavelength_m) * torch.sqrt(argument.clamp_min(0.0))
    transfer = torch.polar(torch.ones_like(phase), phase)
    transfer = torch.where(argument >= 0.0, transfer, torch.zeros_like(transfer))
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return transfer.to(dtype=complex_dtype)


def angular_spectrum_propagate(field: ComplexField, distance_m: float, plane_id: str) -> ComplexField:
    """Production FFT ASM; it creates a new physical plane and provenance node."""
    if distance_m < 0.0:
        raise ValueError("formal v2 propagation distance must be non-negative")
    key = (field.plane.shape, field.plane.pixel_pitch_m, field.plane.wavelength_m, distance_m, str(field.tensor.device), str(field.tensor.real.dtype))
    transfer = _PRODUCTION_TRANSFER_CACHE.get(key)
    if transfer is None:
        # The cached kernel is constant. Build it outside inference mode so a
        # prior frozen measurement cannot poison a later autograd forward.
        with torch.inference_mode(False):
            transfer = _transfer_function(
                field.plane.shape,
                pixel_pitch_m=field.plane.pixel_pitch_m,
                wavelength_m=field.plane.wavelength_m,
                distance_m=distance_m,
                device=field.tensor.device,
                real_dtype=field.tensor.real.dtype,
            )
        _PRODUCTION_TRANSFER_CACHE[key] = transfer
    propagated = torch.fft.ifft2(torch.fft.fft2(field.tensor, norm="ortho") * transfer, norm="ortho")
    plane = field.plane.with_plane(plane_id, z_m=field.plane.z_m + distance_m)
    provenance = OperationProvenance(
        "propagate_angular_spectrum",
        field.plane.plane_id,
        distance_m,
        {"wavelength_m": field.plane.wavelength_m, "sampling": plane.to_dict()},
    )
    return ComplexField(propagated, plane, provenance)


def reference_angular_spectrum_propagate(field: ComplexField, distance_m: float, plane_id: str) -> ComplexField:
    """Reference ASM with independently assembled spectral coordinates and transfer."""
    if distance_m < 0.0:
        raise ValueError("reference propagation distance must be non-negative")
    height, width = field.plane.shape
    dy, dx = field.plane.pixel_pitch_m
    dtype = field.tensor.real.dtype
    ky = torch.arange(height, device=field.tensor.device, dtype=dtype)
    kx = torch.arange(width, device=field.tensor.device, dtype=dtype)
    fy = torch.where(ky < (height + 1) // 2, ky, ky - height) / (height * dy)
    fx = torch.where(kx < (width + 1) // 2, kx, kx - width) / (width * dx)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    radicand = 1.0 - (field.plane.wavelength_m * fx_grid).square() - (field.plane.wavelength_m * fy_grid).square()
    phase = 2.0 * math.pi * distance_m * torch.sqrt(radicand.clamp_min(0.0)) / field.plane.wavelength_m
    kernel = torch.complex(torch.cos(phase), torch.sin(phase)) * (radicand >= 0.0)
    spectrum = torch.fft.fft2(field.tensor)
    propagated = torch.fft.ifft2(spectrum * kernel)
    plane = field.plane.with_plane(plane_id, z_m=field.plane.z_m + distance_m)
    provenance = OperationProvenance("propagate_angular_spectrum_reference", field.plane.plane_id, distance_m)
    return ComplexField(propagated, plane, provenance)
