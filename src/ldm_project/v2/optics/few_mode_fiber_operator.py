"""Research-only nested few-mode fiber OLS.

The formal v2 optical system is intentionally untouched.  This module adds a
small, deterministic M=2 research boundary which reuses the validated LP01
bank and appends one orthogonal x-oriented LP11-like mode per port.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import ComplexField, OperationProvenance, PlaneSpec
from .fiber_operator import FiberArrayOLS, FiberArraySpec, lp01_eigenpair


@dataclass(frozen=True)
class FewModeFiberArraySpec:
    """Geometry and ordered mode contract for the research FMF boundary."""

    modes: tuple[str, ...] = ("LP01", "LP11a")
    array_shape: tuple[int, int] = (10, 10)
    pitch_m: float = 2.0e-6
    wavelength_m: float = 532.0e-9
    waist_m: float = 7.111e-6
    fill: float = 0.8
    v_number: float = 2.2
    expected_fov_m: tuple[float, float] = (200.0e-6, 200.0e-6)

    def __post_init__(self) -> None:
        if self.array_shape != (10, 10):
            raise ValueError("the research FMF contract is fixed to a 10x10 array")
        if self.modes not in (("LP01",), ("LP01", "LP11a")):
            raise ValueError("supported nested FMF modes are LP01 and LP11a")
        if self.v_number <= 0.0 or self.wavelength_m <= 0.0 or self.pitch_m <= 0.0:
            raise ValueError("FMF geometry values must be positive")
        if self.waist_m <= 0.0 or not 0.0 < self.fill <= 1.0:
            raise ValueError("FMF mode geometry is invalid")
        if len(self.expected_fov_m) != 2 or any(v <= 0.0 for v in self.expected_fov_m):
            raise ValueError("expected_fov_m must contain two positive values")


def _interleaved_modes(lp01: torch.Tensor, lp11a: torch.Tensor | None) -> torch.Tensor:
    if lp11a is None:
        return lp01
    return torch.stack((lp01, lp11a), dim=1).reshape(-1, *lp01.shape[1:])


def _lp11a_bank(plane: PlaneSpec, spec: FewModeFiberArraySpec, lp01: torch.Tensor) -> torch.Tensor:
    """Build an x-oriented LP11-like bank orthogonal to the fixed LP01 bank."""
    h, w = plane.shape
    y = (torch.arange(h, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[0] - plane.fov_m[0] / 2.0
    x = (torch.arange(w, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[1] - plane.fov_m[1] / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    _facet_fov_y, facet_fov_x = plane.fov_m[0] * spec.fill, plane.fov_m[1] * spec.fill
    centers_x = torch.linspace(-facet_fov_x / 2.0, facet_fov_x / 2.0, 10, dtype=torch.float64)
    _u, _w, rho_e = lp01_eigenpair(spec.v_number)
    core_radius = spec.waist_m / rho_e
    candidates: list[torch.Tensor] = []
    for port in range(100):
        yi, xi = divmod(port, 10)
        local_x = (xx - centers_x[xi]) / core_radius
        candidates.append(lp01[port].real.to(torch.float64) * local_x)
    raw = torch.stack(candidates).reshape(100, -1).to(torch.complex128)
    base = lp01.reshape(100, -1).to(torch.complex128)
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    projection = raw @ base.conj().T * area
    residual = raw - projection @ base
    gram = residual @ residual.conj().T * area
    values, vectors = torch.linalg.eigh(gram)
    if float(values.min()) <= 1.0e-12:
        raise ValueError("FMF LP11a residual bank is rank deficient")
    whitened = vectors @ torch.diag(values.rsqrt().to(vectors.dtype)) @ vectors.conj().T @ residual
    return whitened.reshape(100, h, w).to(torch.complex64)


class FewModeFiberArrayOLS(nn.Module):
    """Nested LP01/LP11a port OLS with identity mode transport."""

    diagnostic_only = True

    def __init__(
        self,
        input_plane: PlaneSpec,
        spec: FewModeFiberArraySpec = FewModeFiberArraySpec(),
        transport: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_plane.shape != (100, 100):
            raise ValueError("FMF OLS requires the formal 100x100 encoder plane")
        if input_plane.pixel_pitch_m != (spec.pitch_m, spec.pitch_m):
            raise ValueError("FMF pixel pitch does not match the encoder PlaneSpec")
        if any(abs(a - b) > 1.0e-12 for a, b in zip(input_plane.fov_m, spec.expected_fov_m, strict=True)) or abs(input_plane.wavelength_m - spec.wavelength_m) > 1.0e-15:
            raise ValueError("FMF geometry does not match the encoder PlaneSpec")
        smf_spec = FiberArraySpec(
            pitch_m=spec.pitch_m,
            wavelength_m=spec.wavelength_m,
            waist_m=spec.waist_m,
            fill=spec.fill,
            v_number=spec.v_number,
            expected_fov_m=spec.expected_fov_m,
        )
        smf = FiberArrayOLS(input_plane, smf_spec)
        lp01 = smf.input_modes.detach().clone()
        lp11a = _lp11a_bank(input_plane, spec, lp01) if spec.modes == ("LP01", "LP11a") else None
        mode_bank = _interleaved_modes(lp01, lp11a)
        mode_count = len(spec.modes)
        total_modes = 100 * mode_count
        if transport is None:
            transport = torch.eye(total_modes, dtype=torch.complex64)
        if transport.shape != (total_modes, total_modes) or not torch.is_complex(transport):
            raise ValueError("FMF transport must match the configured complex mode count")
        if float(torch.linalg.svdvals(transport).max()) > 1.0 + 1.0e-5:
            raise ValueError("FMF transport matrix is non-passive")
        self.input_plane = input_plane
        self.output_plane = input_plane.with_plane("ols_output")
        self.spec = spec
        self.mode_names = spec.modes
        self.register_buffer("input_modes", mode_bank)
        self.register_buffer("output_modes", mode_bank.clone())
        self.register_buffer("transport", transport)
        self.last_diagnostics: dict[str, object] = {}

    @property
    def mode_count(self) -> int:
        return len(self.mode_names)

    @property
    def total_mode_count(self) -> int:
        return self.input_modes.shape[0]

    def _validate_field(self, field: ComplexField) -> None:
        if (
            field.plane.shape != self.input_plane.shape
            or any(abs(a - b) > 1.0e-15 for a, b in zip(field.plane.pixel_pitch_m, self.input_plane.pixel_pitch_m, strict=True))
            or abs(field.plane.wavelength_m - self.input_plane.wavelength_m) > 1.0e-15
            or any(abs(a - b) > 1.0e-12 for a, b in zip(field.plane.fov_m, self.input_plane.fov_m, strict=True))
            or field.plane.coordinate_frame != self.input_plane.coordinate_frame
        ):
            raise ValueError("FMF input plane mismatch")
        if field.tensor.ndim not in {3, 4} or (field.tensor.ndim == 4 and field.tensor.shape[1] != 1):
            raise ValueError("FMF OLS requires [batch,H,W] or [batch,1,H,W] fields")
        if not torch.isfinite(field.tensor).all():
            raise ValueError("FMF OLS input must be finite")

    def forward(self, field: ComplexField) -> ComplexField:
        self._validate_field(field)
        values = field.tensor[:, 0] if field.tensor.ndim == 4 else field.tensor
        input_modes = self.input_modes.to(values)
        output_modes = self.output_modes.to(values)
        area = self.input_plane.pixel_pitch_m[0] * self.input_plane.pixel_pitch_m[1]
        coefficients = torch.einsum("mhw,bhw->bm", input_modes.conj(), values) * area
        output_coefficients = coefficients @ self.transport.to(values).T
        output_values = torch.einsum("bm,mhw->bhw", output_coefficients, output_modes)
        input_energy = values.abs().square().sum(dim=(-2, -1)) * area
        output_energy = output_values.abs().square().sum(dim=(-2, -1)) * area
        if torch.any(output_energy > input_energy * (1.0 + 1.0e-5) + 1.0e-12):
            raise ValueError("FMF OLS violates the input/output energy gate")
        per_mode = output_coefficients.reshape(-1, 100, self.mode_count)
        self.last_diagnostics = {
            "input_energy": input_energy.detach(),
            "output_energy": output_energy.detach(),
            "coupling_efficiency": (coefficients.abs().square().sum(dim=1) / input_energy.clamp_min(1.0e-30)).detach(),
            "port_input_coefficients": coefficients.detach(),
            "port_output_coefficients": output_coefficients.detach(),
            "mode_coefficients": per_mode.detach(),
            "mode_power": per_mode.abs().square().sum(dim=1).detach(),
            "active_mode_fraction": (per_mode.abs().square().sum(dim=1) > 1.0e-12).to(torch.float32).mean(dim=1).detach(),
            "transport_singular_max": float(torch.linalg.svdvals(self.transport).max()),
            "mode_count": self.mode_count,
            "total_mode_count": self.total_mode_count,
        }
        output = output_values.unsqueeze(1) if field.tensor.ndim == 4 else output_values
        return ComplexField(
            output,
            self.output_plane,
            OperationProvenance(
                "synthetic_fmf_nested_lp01_lp11a_identity_transport",
                self.input_plane.plane_id,
                0.0,
                {
                    "modes": list(self.mode_names),
                    "mode_count": self.mode_count,
                    "total_mode_count": self.total_mode_count,
                    "transport": "identity",
                    "physical_parameter_status": "diagnostic_project_approximation",
                },
            ),
        )


__all__ = ["FewModeFiberArrayOLS", "FewModeFiberArraySpec"]
