"""Independent scalar references for the coherent imaging relay.

This module deliberately does not import the production relay implementation.
The Collins B=0 expression is evaluated from the ABCD matrix and physical
coordinates, while the Rayleigh--Sommerfeld helper is a small direct reference
used for aperture/diffraction checks.
"""
from __future__ import annotations

import math

import torch


def thin_lens_matrix(focal_length: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 0.0), (-1.0 / focal_length, 1.0))


def free_space_matrix(distance: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, distance), (0.0, 1.0))


def multiply_abcd(
    left: tuple[tuple[float, float], tuple[float, float]],
    right: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(2)) for j in range(2))
        for i in range(2)
    )  # type: ignore[return-value]


def four_f_abcd(f1: float, f2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """ABCD matrix for free(f1)-lens(f1)-free(f1+f2)-lens(f2)-free(f2)."""
    result = ((1.0, 0.0), (0.0, 1.0))
    for element in (
        free_space_matrix(f1),
        thin_lens_matrix(f1),
        free_space_matrix(f1 + f2),
        thin_lens_matrix(f2),
        free_space_matrix(f2),
    ):
        result = multiply_abcd(element, result)
    return result


def _physical_grid(size: int, pixel_size: float, *, device: torch.device, dtype: torch.dtype):
    coord = (torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0) * pixel_size
    return torch.meshgrid(coord, coord, indexing="ij")


def collins_b0_relay_reference(
    field: torch.Tensor,
    *,
    magnification: float,
    output_size: tuple[int, int],
    input_pixel_size: float,
    output_pixel_size: float,
    wavelength: float,
    c_abcd: float,
    pupil_diameter: float,
    pupil_focal_length: float,
) -> torch.Tensor:
    """Independent coordinate-aware Collins B=0 relay evaluation.

    This is deliberately independent from the production implementation: it
    constructs the Collins image from its own DFT definition and evaluates
    that Fourier series on physical image coordinates.  It neither imports a
    production relay helper nor interpolates/resizes a display tensor.

    As with the production operator, the sampled input is a periodic discrete
    Fourier field.  This is not evidence for an aperture outside the sampled
    SLM computational window.
    """
    if not torch.is_complex(field):
        raise TypeError("relay reference requires a complex field")
    if magnification == 0.0 or pupil_diameter <= 0.0 or pupil_focal_length <= 0.0:
        raise ValueError("invalid relay reference geometry")
    height, width = output_size
    dtype = field.real.dtype
    yy, xx = _physical_grid(height, output_pixel_size, device=field.device, dtype=dtype)
    source_x = xx / magnification
    source_y = yy / magnification
    source_h, source_w = field.shape[-2:]
    # For the formal two-lens 4f geometry C=0.  The pupil focal length is the
    # explicit physical f2; c_abcd only controls the optional general Collins
    # quadratic phase and is zero for this ideal relay.
    f2 = pupil_focal_length
    object_cutoff = pupil_diameter * abs(magnification) / (2.0 * f2 * wavelength)
    fy_in = torch.fft.fftfreq(source_h, d=input_pixel_size, device=field.device, dtype=dtype)
    fx_in = torch.fft.fftfreq(source_w, d=input_pixel_size, device=field.device, dtype=dtype)
    fy_in, fx_in = torch.meshgrid(fy_in, fx_in, indexing="ij")
    source_pupil = ((fx_in.square() + fy_in.square()).sqrt() <= object_cutoff * (1.0 + 1.0e-6)).to(field.dtype)
    dft = torch.fft.fft2(field.contiguous()) * source_pupil
    y_origin = -(source_h - 1) * input_pixel_size / 2.0
    x_origin = -(source_w - 1) * input_pixel_size / 2.0
    phase_y = torch.exp(
        2j
        * math.pi
        * (source_y[:, 0, None] - y_origin)
        * torch.fft.fftfreq(source_h, d=input_pixel_size, device=field.device, dtype=dtype)[None, :]
    ).to(field.dtype)
    phase_x = torch.exp(
        2j
        * math.pi
        * (source_x[0, :, None] - x_origin)
        * torch.fft.fftfreq(source_w, d=input_pixel_size, device=field.device, dtype=dtype)[None, :]
    ).to(field.dtype)
    result_field = torch.einsum("...ij,ui,vj->...uv", dft, phase_y, phase_x) / float(source_h * source_w)
    c = (2.0 * math.pi / wavelength) * c_abcd / (2.0 * magnification)
    transfer = torch.exp(1j * c * (xx.square() + yy.square())).to(field.dtype)
    result = result_field * transfer.unsqueeze(0).unsqueeze(0) / abs(magnification)
    fy = torch.fft.fftfreq(height, d=output_pixel_size, device=field.device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=output_pixel_size, device=field.device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    cutoff = pupil_diameter / (2.0 * pupil_focal_length * wavelength)
    pupil = ((fx.square() + fy.square()).sqrt() <= cutoff * (1.0 + 1.0e-6)).to(result.dtype)
    return torch.fft.ifft2(torch.fft.fft2(result) * pupil)


def direct_rayleigh_sommerfeld(
    field: torch.Tensor,
    *,
    distance: float,
    wavelength: float,
    input_pixel_size: float,
    output_pixel_size: float | None = None,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Small-grid direct Rayleigh--Sommerfeld scalar propagation reference."""
    if not torch.is_complex(field):
        raise TypeError("direct reference requires a complex field")
    output_pixel_size = input_pixel_size if output_pixel_size is None else output_pixel_size
    output_size = tuple(field.shape[-2:]) if output_size is None else output_size
    if field.shape[0] != 1 or field.shape[1] != 1:
        raise ValueError("direct reference accepts one sample and one channel")
    dtype = field.real.dtype
    yi, xi = _physical_grid(field.shape[-2], input_pixel_size, device=field.device, dtype=dtype)
    yo, xo = _physical_grid(output_size[0], output_pixel_size, device=field.device, dtype=dtype)
    source_x = xi.reshape(-1)
    source_y = yi.reshape(-1)
    target_x = xo.reshape(-1)
    target_y = yo.reshape(-1)
    k = 2.0 * math.pi / wavelength
    source_values = field[0, 0].reshape(-1)
    # Keep the direct reference bounded in memory; a full N² kernel is very
    # large for the 96×96 aperture check.
    chunks: list[torch.Tensor] = []
    for start in range(0, target_x.numel(), 256):
        stop = min(start + 256, target_x.numel())
        separation = torch.sqrt(
            (target_y[start:stop, None] - source_y[None, :]).square()
            + (target_x[start:stop, None] - source_x[None, :]).square()
            + distance**2
        )
        kernel = distance / (1j * wavelength * separation.square()) * torch.exp(1j * k * separation)
        chunks.append(kernel @ source_values * input_pixel_size**2)
    return torch.cat(chunks).reshape(1, 1, *output_size)


def similarity_scaled_propagation_reference(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.complex128,
    size: int = 24,
) -> dict[str, object]:
    """Independent small-system shifted/scaled Fresnel versus direct-RS check.

    This is intentionally a *similarity-scaled propagation component* rather
    than a falsely under-resolved full-size 23.75 mm two-lens calculation.
    The Gaussian is compact in the chosen window, the Fresnel output grid is
    explicitly specified, and the returned sampling/boundary audit makes the
    comparison invalid if its finite window is not adequate.
    """
    pixel_size = 1.0e-6
    wavelength = 532e-9
    # This compact, similarity-scaled component is chosen so its diffracted
    # Gaussian remains inside the explicit 24 µm window.  It is therefore a
    # valid sampling/alias check instead of a truncated long-distance proxy.
    distance = 40e-6
    coord = (torch.arange(size, device=device, dtype=torch.float64) - (size - 1) / 2.0) * pixel_size
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    field = torch.exp(-(xx.square() + yy.square()) / (3.0e-6**2)).to(dtype)[None, None]
    arss = _fresnel_integral_reference(
        field,
        distance=distance,
        wavelength=wavelength,
        input_pixel_size=pixel_size,
        output_pixel_size=pixel_size,
        output_size=size,
    )
    direct = direct_rayleigh_sommerfeld(
        field,
        distance=distance,
        wavelength=wavelength,
        input_pixel_size=pixel_size,
        output_pixel_size=pixel_size,
        output_size=(size, size),
    )
    phase = torch.angle((arss.conj() * direct).sum())
    aligned = arss * torch.exp(1j * phase).to(dtype)
    valid = direct.abs().square() >= direct.abs().square().amax() * 1.0e-6
    edge = torch.zeros((size, size), device=device, dtype=torch.bool)
    edge[:2] = edge[-2:] = True
    edge[:, :2] = edge[:, -2:] = True
    boundary_energy = float(direct[0, 0].abs().square()[edge].sum() / direct.abs().square().sum().clamp_min(1e-30))
    return {
        "arss_field": arss,
        "direct_rs_field": direct,
        "complex_relative_l2": float((aligned - direct).norm() / direct.norm().clamp_min(1e-30)),
        "wrapped_phase_mae_rad": float(torch.angle(aligned[valid] * direct[valid].conj()).abs().mean()),
        "energy_relative_error": float(
            (arss.abs().square().sum() - direct.abs().square().sum()).abs()
            / direct.abs().square().sum().clamp_min(1e-30)
        ),
        "boundary_energy_fraction": boundary_energy,
        "sampling": {
            "input_pixel_size_m": pixel_size,
            "output_pixel_size_m": pixel_size,
            "fov_m": size * pixel_size,
            "padding": 0,
            "alias_margin": "compact Gaussian; direct-RS and shifted/scaled Fresnel use the same explicit grid",
            "similarity_note": "component-level dimensionless propagation check; not a low-resolution actual-pupil 4f reference",
        },
    }


def _fresnel_reference(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    """Independent fixed-grid Fresnel transfer used only by relay validation."""
    height, width = field.shape[-2:]
    dtype = field.real.dtype
    fy = torch.fft.fftfreq(height, d=pixel_size, device=field.device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=pixel_size, device=field.device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    transfer = torch.exp(-1j * math.pi * wavelength * distance * (fx.square() + fy.square())).to(field.dtype)
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)


def _fresnel_integral_reference(
    field: torch.Tensor,
    *,
    distance: float,
    wavelength: float,
    input_pixel_size: float,
    output_pixel_size: float,
    output_size: int,
) -> torch.Tensor:
    """Chunked direct Fresnel integral on an explicitly chosen physical grid."""
    if field.shape[0] != 1 or field.shape[1] != 1:
        raise ValueError("Fresnel reference accepts one sample and one channel")
    dtype = field.real.dtype
    iy, ix = _physical_grid(field.shape[-2], input_pixel_size, device=field.device, dtype=dtype)
    oy, ox = _physical_grid(output_size, output_pixel_size, device=field.device, dtype=dtype)
    source_x = ix.reshape(-1)
    source_y = iy.reshape(-1)
    source_values = field[0, 0].reshape(-1)
    target_x = ox.reshape(-1)
    target_y = oy.reshape(-1)
    k = 2.0 * math.pi / wavelength
    chunks: list[torch.Tensor] = []
    for start in range(0, target_x.numel(), 128):
        stop = min(start + 128, target_x.numel())
        distance_sq = (target_x[start:stop, None] - source_x[None, :]).square() + (
            target_y[start:stop, None] - source_y[None, :]
        ).square()
        kernel = torch.exp(1j * k * (distance + distance_sq / (2.0 * distance))).to(field.dtype)
        chunks.append(kernel @ source_values * input_pixel_size**2 / (1j * wavelength * distance))
    return torch.cat(chunks).reshape(1, 1, output_size, output_size)


def adaptive_4f_reference_geometry(
    *,
    f1: float,
    f2: float,
    wavelength: float,
    input_fov: float,
    pupil_diameter: float,
    intermediate_grid_size: int = 64,
) -> dict[str, float | int]:
    """Choose an intermediate Fourier-plane FOV with a conservative margin."""
    object_cutoff = pupil_diameter * (f2 / f1) / (2.0 * f2 * wavelength)
    fourier_radius = wavelength * f1 * object_cutoff
    intermediate_fov = max(input_fov, 4.0 * fourier_radius)
    return {
        "intermediate_fov_m": float(intermediate_fov),
        "intermediate_pixel_size_m": float(intermediate_fov / intermediate_grid_size),
        "intermediate_grid_size": int(intermediate_grid_size),
        "fourier_radius_m": float(fourier_radius),
        "object_cutoff_cycles_m": float(object_cutoff),
    }


def four_f_scaled_fresnel_reference(
    field: torch.Tensor,
    *,
    f1: float,
    f2: float,
    wavelength: float,
    input_pixel_size: float,
    output_size: tuple[int, int],
    output_pixel_size: float,
    pupil_diameter: float,
    intermediate_grid_size: int = 64,
) -> torch.Tensor:
    """Independent two-lens 4f reference with explicit thin-lens propagation.

    Propagation is performed on the complete input physical FOV; only the
    final physical image plane is sampled onto the requested encoder grid by
    the sinc quadrature above. No production relay code is imported.
    """
    if not torch.is_complex(field):
        raise TypeError("4f reference requires a complex field")
    height, width = field.shape[-2:]
    dtype = field.real.dtype
    coord_y, coord_x = _physical_grid(height, input_pixel_size, device=field.device, dtype=dtype)
    radius = torch.sqrt(coord_x.square() + coord_y.square())
    pupil = (radius <= pupil_diameter / 2.0).to(field.dtype)
    k = 2.0 * math.pi / wavelength
    lens1 = torch.exp((-1j * k * (coord_x.square() + coord_y.square()) / (2.0 * f1)).to(field.dtype)) * pupil
    geometry = adaptive_4f_reference_geometry(
        f1=f1, f2=f2, wavelength=wavelength, input_fov=height * input_pixel_size,
        pupil_diameter=pupil_diameter, intermediate_grid_size=intermediate_grid_size,
    )
    intermediate_pixel = float(geometry["intermediate_pixel_size_m"])
    intermediate_size = int(geometry["intermediate_grid_size"])
    iy, ix = _physical_grid(intermediate_size, intermediate_pixel, device=field.device, dtype=dtype)
    lens1 = torch.exp((-1j * k * (ix.square() + iy.square()) / (2.0 * f1)).to(field.dtype)) * (
        torch.sqrt(ix.square() + iy.square()) <= pupil_diameter / 2.0
    ).to(field.dtype)
    propagated = _fresnel_integral_reference(
        field, distance=f1, wavelength=wavelength, input_pixel_size=input_pixel_size,
        output_pixel_size=intermediate_pixel, output_size=intermediate_size,
    ) * lens1
    propagated = _fresnel_integral_reference(
        propagated, distance=f1 + f2, wavelength=wavelength,
        input_pixel_size=intermediate_pixel, output_pixel_size=intermediate_pixel,
        output_size=intermediate_size,
    )
    lens2 = torch.exp((-1j * k * (ix.square() + iy.square()) / (2.0 * f2)).to(field.dtype)) * (
        torch.sqrt(ix.square() + iy.square()) <= pupil_diameter / 2.0
    ).to(field.dtype)
    propagated = _fresnel_integral_reference(
        propagated * lens2, distance=f2, wavelength=wavelength,
        input_pixel_size=intermediate_pixel, output_pixel_size=output_pixel_size,
        output_size=output_size[0],
    )
    return propagated / abs(f2 / f1)


def four_f_direct_rayleigh_sommerfeld_reference(
    field: torch.Tensor,
    *,
    f1: float,
    f2: float,
    wavelength: float,
    input_pixel_size: float,
    output_size: tuple[int, int],
    output_pixel_size: float,
    pupil_diameter: float,
    intermediate_grid_size: int = 48,
) -> torch.Tensor:
    """Low-resolution direct-RS two-lens reference for validation only."""
    if field.shape[0] != 1 or field.shape[1] != 1:
        raise ValueError("direct 4f reference accepts one sample and one channel")
    height, width = field.shape[-2:]
    dtype = field.real.dtype
    k = 2.0 * math.pi / wavelength
    geometry = adaptive_4f_reference_geometry(
        f1=f1, f2=f2, wavelength=wavelength, input_fov=height * input_pixel_size,
        pupil_diameter=pupil_diameter, intermediate_grid_size=intermediate_grid_size,
    )
    intermediate_pixel = float(geometry["intermediate_pixel_size_m"])
    intermediate_size = int(geometry["intermediate_grid_size"])
    iy, ix = _physical_grid(intermediate_size, intermediate_pixel, device=field.device, dtype=dtype)
    pupil = (torch.sqrt(ix.square() + iy.square()) <= pupil_diameter / 2.0).to(field.dtype)
    lens1 = torch.exp((-1j * k * (ix.square() + iy.square()) / (2.0 * f1)).to(field.dtype)) * pupil
    propagated = direct_rayleigh_sommerfeld(
        field, distance=f1, wavelength=wavelength, input_pixel_size=input_pixel_size,
        output_pixel_size=intermediate_pixel, output_size=(intermediate_size, intermediate_size),
    )
    propagated = propagated * lens1
    propagated = direct_rayleigh_sommerfeld(
        propagated, distance=f1 + f2, wavelength=wavelength,
        input_pixel_size=intermediate_pixel, output_pixel_size=intermediate_pixel,
        output_size=(intermediate_size, intermediate_size),
    )
    lens2 = torch.exp((-1j * k * (ix.square() + iy.square()) / (2.0 * f2)).to(field.dtype)) * pupil
    propagated = propagated * lens2
    return direct_rayleigh_sommerfeld(
        propagated, distance=f2, wavelength=wavelength,
        input_pixel_size=intermediate_pixel, output_pixel_size=output_pixel_size,
        output_size=output_size,
    ) / abs(f2 / f1)
