"""Opt-in, uncalibrated Gaussian OLS for the small LightGen research geometry.

The analytical S12 normalization is retained. Only the common numerical FOV
limits the fields; there are no port cells, discrete renormalization, or
orthogonalization. This module does not change the existing LP01 runtime.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


PASSIVITY_TOLERANCE = 1.0e-5
MODE_NORM_TOLERANCE = 5.0e-6


@dataclass(frozen=True)
class GaussianFiberGeometry:
    port_pitch_m: float = 20.0e-6
    waist_m: float = 3.5e-6

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x <= 0.0 for x in (self.port_pitch_m, self.waist_m)):
            raise ValueError("Gaussian port pitch and waist must be finite and positive")


class GaussianFiberArrayOLS(nn.Module):
    """100 complex Gaussian overlaps and same-plane modal re-emission.

Separable axes keep the independently refined OLS checks small in memory.
The Gram spectrum bounds the complete U U* operator, including re-emission.
"""

    diagnostic_only = True

    def __init__(self, input_plane: PlaneSpec, geometry: GaussianFiberGeometry) -> None:
        super().__init__()
        if not all(math.isfinite(x) for x in (*input_plane.pixel_pitch_m, *input_plane.origin_m, input_plane.wavelength_m)):
            raise ValueError("Gaussian OLS plane geometry must be finite")
        if input_plane.wavelength_m != 532.0e-9:
            raise ValueError("this Gaussian baseline requires 532 nm")
        if 4.5 * geometry.port_pitch_m >= min(input_plane.fov_m) / 2.0:
            raise ValueError("Gaussian port centers must lie inside the common optical FOV")
        self.input_plane = input_plane
        self.geometry = geometry
        axes = []
        centers = []
        amplitude = (2.0 / (math.pi * geometry.waist_m**2)) ** 0.25
        for size, pitch, fov, origin in zip(input_plane.shape, input_plane.pixel_pitch_m, input_plane.fov_m, input_plane.origin_m, strict=True):
            coordinates = (torch.arange(size, dtype=torch.float64) + 0.5) * pitch - fov / 2.0 + origin
            center = (torch.arange(10, dtype=torch.float64) - 4.5) * geometry.port_pitch_m + origin
            axes.append(amplitude * torch.exp(-((coordinates[None, :] - center[:, None]) / geometry.waist_m).square()))
            centers.append(center)
        self.register_buffer("modes_y", axes[0])
        self.register_buffer("modes_x", axes[1])
        cy, cx = torch.meshgrid(*centers, indexing="ij")
        self.register_buffer("centers_m", torch.stack((cy.reshape(-1), cx.reshape(-1)), dim=1))
        self.register_buffer("transport", torch.eye(100, dtype=torch.complex128))
        gy = axes[0] @ axes[0].T * input_plane.pixel_pitch_m[0]
        gx = axes[1] @ axes[1].T * input_plane.pixel_pitch_m[1]
        gram = torch.kron(gy, gx)
        largest = float(torch.linalg.eigvalsh(gram)[-1])
        norm_error = float((gram.diag() - 1.0).abs().max())
        gain = largest**2
        if norm_error > MODE_NORM_TOLERANCE:
            raise ValueError(f"Gaussian mode sampling/FOV error exceeds tolerance: {norm_error:.12g}")
        if gain > 1.0 + PASSIVITY_TOLERANCE:
            raise ValueError(f"Gaussian projection/re-emission is non-passive: gain={gain:.12g}")
        self.geometry_diagnostics = {
            "mode_norm_max_absolute_error": norm_error,
            "gram_largest_eigenvalue": largest,
            "complete_operator_power_gain_bound": gain,
            "max_off_diagonal_overlap": float((gram - torch.diag(gram.diag())).abs().max()),
            "parameter_status": "PROJECT_GEOMETRY_UNMEASURED",
            "normalization": "analytic_infinite_plane_S12",
            "per_port_aperture": False,
            "discrete_renormalization": False,
            "orthogonalization": False,
        }
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _values(self, field: ComplexField) -> torch.Tensor:
        if field.plane != self.input_plane:
            raise ValueError("Gaussian OLS input physical plane mismatch")
        values = field.tensor
        if values.ndim == 4 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 3 or not torch.isfinite(values).all():
            raise ValueError("Gaussian OLS requires finite [B,H,W] or [B,1,H,W] fields")
        return values

    def project_coefficients(self, field: ComplexField) -> torch.Tensor:
        values = self._values(field)
        intermediate = torch.einsum("rh,bhw->brw", self.modes_y.to(values), values) * self.input_plane.pixel_pitch_m[0]
        coefficients = torch.einsum("brw,cw->brc", intermediate, self.modes_x.to(values)) * self.input_plane.pixel_pitch_m[1]
        return coefficients.reshape(values.shape[0], 100)

    def transmit_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != 100 or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
            raise ValueError("Gaussian OLS coefficients must be finite complex [B,100]")
        return coefficients @ self.transport.to(coefficients).T

    def reemit_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != 100 or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
            raise ValueError("Gaussian OLS coefficients must be finite complex [B,100]")
        intermediate = torch.einsum("brc,cw->brw", coefficients.reshape(-1, 10, 10), self.modes_x.to(coefficients))
        return torch.einsum("rh,brw->bhw", self.modes_y.to(coefficients), intermediate)

    def forward(self, field: ComplexField) -> ComplexField:
        values = self._values(field)
        coefficients = self.transmit_coefficients(self.project_coefficients(field))
        emitted = self.reemit_coefficients(coefficients)
        area = math.prod(self.input_plane.pixel_pitch_m)
        input_power = values.abs().square().sum((-2, -1)) * area
        output_power = emitted.abs().square().sum((-2, -1)) * area
        if torch.any(output_power > input_power * (1.0 + PASSIVITY_TOLERANCE) + 1.0e-12):
            raise ValueError("Gaussian OLS input/output power check failed")
        self.last_diagnostics = {"input_power": input_power.detach(), "output_power": output_power.detach()}
        tensor = emitted.unsqueeze(1) if field.tensor.ndim == 4 else emitted
        return ComplexField(tensor, self.input_plane.with_plane("gaussian_ols_output"), OperationProvenance(
            "uncalibrated_gaussian_overlap_and_reemission", self.input_plane.plane_id, 0.0,
            {**asdict(self.geometry), **self.geometry_diagnostics, "port_order": "row_major_j=10*r+c", "transport": "I_100"},
        ))
