"""Matched-intensity coefficient approximation; not a measured material model."""
from __future__ import annotations

import math

import torch

from .spatial_material_interface import SpatialMaterialInterface, SyntheticResponse


def matched_mode_response(
    coefficients: torch.Tensor,
    modes: torch.Tensor,
    shape: tuple[int, int],
    pixel_area_m2: float,
    interface: SpatialMaterialInterface,
    response: SyntheticResponse,
    *,
    mode_chunk_size: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return z'_j=a_j z_j and a_j=integral |v_j|^2 t(|z_j v_j|^2)/norm_j.

The local amplitude response is obtained from the existing M2 producer for
each isolated modal field. No peak-intensity or new reference scale is used.
Each mode is integrated separately: coherent cross terms and shape residuals
are deliberately omitted from the reconstructed coefficient model.
"""
    if (coefficients.ndim != 2 or not torch.is_complex(coefficients)
            or not torch.isfinite(coefficients).all() or modes.ndim != 2
            or not torch.is_complex(modes) or not torch.isfinite(modes).all()
            or modes.shape != (math.prod(shape), coefficients.shape[1])
            or mode_chunk_size <= 0 or not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0):
        raise ValueError("matched M1 requires finite complex [B,N], [H*W,N] modes and positive area")
    z = coefficients.to(torch.complex128)
    basis = modes.to(z)
    norm = basis.abs().square().sum(0)*pixel_area_m2
    if (norm <= 0).any():
        raise ValueError("mode norms must be positive")
    zero_transmission = (math.sqrt(1-response.proxy_strength) if response.family == "saturation"
                         else math.sqrt(response.proxy_linear_transmission) if response.family == "linear"
                         else 1.0)
    factors = []
    for start in range(0, z.shape[1], mode_chunk_size):
        end = min(start+mode_chunk_size, z.shape[1])
        v = basis[:, start:end].T
        density = v.abs().square()
        single = z[:, start:end, None]*v[None]
        fields = single.reshape(-1, 1, *shape)
        output = interface.apply("M2", fields, z[:, start:end].reshape(-1, 1), response)
        amplitude = fields.abs()
        safe_amplitude = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
        transmission = torch.where(amplitude > 0, output.abs()/safe_amplitude, zero_transmission)
        transmission = transmission.reshape(z.shape[0], end-start, -1)
        a = (transmission*density[None]).sum(-1)*pixel_area_m2/norm[start:end]
        factors.append(a)
    a = torch.cat(factors, dim=1)
    return z*a, a
