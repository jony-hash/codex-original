"""Typed single-mode-fiber OLS proxy.  It is never an identity OLS fallback."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import ComplexField, OperationProvenance, PlaneSpec


@dataclass(frozen=True)
class FiberModeBankV2:
    modes: torch.Tensor
    centers_m: torch.Tensor
    waist_m: float
    plane: PlaneSpec

    def validate(self) -> None:
        if self.modes.ndim != 3 or self.modes.shape[0] != 100 or tuple(self.modes.shape[-2:]) != self.plane.shape:
            raise ValueError("fiber OLS requires exactly 100 2-D modes")
        if not torch.is_complex(self.modes) or not torch.isfinite(self.modes).all() or self.centers_m.shape != (100, 2):
            raise ValueError("fiber modes must be finite complex fields with physical centers")

    @property
    def gram(self) -> torch.Tensor:
        return torch.einsum("mhw,nhw->mn", self.modes.conj(), self.modes)


@dataclass(frozen=True)
class FiberOLSStateV2:
    coefficients: torch.Tensor
    modes: FiberModeBankV2
    input_plane: PlaneSpec
    provenance: OperationProvenance

    def validate(self) -> None:
        self.modes.validate()
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != 100 or not torch.is_complex(self.coefficients):
            raise ValueError("fiber OLS coefficients must have shape [batch, 100] and complex dtype")
        if not torch.isfinite(self.coefficients).all():
            raise ValueError("fiber OLS coefficients must be finite")


def gaussian_single_mode_bank(plane: PlaneSpec, *, waist_m: float, device: torch.device, dtype: torch.dtype) -> FiberModeBankV2:
    """Production mode construction with 10x10 physical Gaussian single-mode fields."""
    if waist_m <= 0.0:
        raise ValueError("fiber mode waist must be positive")
    fy, fx = plane.fov_m
    yy = (torch.arange(plane.shape[0], device=device, dtype=dtype) + 0.5) * plane.pixel_pitch_m[0] - fy / 2
    xx = (torch.arange(plane.shape[1], device=device, dtype=dtype) + 0.5) * plane.pixel_pitch_m[1] - fx / 2
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    centers_y = torch.linspace(-0.45 * fy, 0.45 * fy, 10, device=device, dtype=dtype)
    centers_x = torch.linspace(-0.45 * fx, 0.45 * fx, 10, device=device, dtype=dtype)
    cy, cx = torch.meshgrid(centers_y, centers_x, indexing="ij")
    centers = torch.stack((cy.reshape(-1), cx.reshape(-1)), dim=1)
    fields = torch.exp(-((grid_y.unsqueeze(0) - centers[:, 0, None, None]).square() + (grid_x.unsqueeze(0) - centers[:, 1, None, None]).square()) / waist_m**2)
    fields = fields / fields.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-12)
    bank = FiberModeBankV2(torch.complex(fields, torch.zeros_like(fields)), centers, waist_m, plane)
    bank.validate()
    return bank


def project_single_mode_fibers(field: ComplexField, bank: FiberModeBankV2) -> FiberOLSStateV2:
    """Production coherent coupling; coefficients retain amplitude and phase."""
    bank.validate()
    if field.plane != bank.plane:
        raise ValueError("fiber mode bank must share the incoming physical plane")
    coefficients = torch.einsum("mhw,bchw->bm", bank.modes.conj(), field.tensor)
    state = FiberOLSStateV2(coefficients, bank, field.plane, OperationProvenance("single_mode_fiber_projection", field.plane.plane_id, None, {"mode_count": 100, "waist_m": bank.waist_m, "centers_m": bank.centers_m.detach().cpu().tolist()}))
    state.validate()
    return state


def reemit_single_mode_fibers(state: FiberOLSStateV2, *, plane_id: str) -> ComplexField:
    state.validate()
    tensor = torch.einsum("bm,mhw->bhw", state.coefficients, state.modes.modes).unsqueeze(1)
    plane = state.input_plane.with_plane(plane_id)
    return ComplexField(tensor, plane, OperationProvenance("single_mode_fiber_reemission", state.input_plane.plane_id, None, {"mode_count": 100, "coupling_efficiency": "represented_by_coefficient_norm"}))


def reference_project_single_mode_fibers(field: ComplexField, bank: FiberModeBankV2) -> torch.Tensor:
    """Independent scalar-loop reference, intentionally not used by production code."""
    bank.validate()
    if field.plane != bank.plane:
        raise ValueError("reference fiber bank plane mismatch")
    coefficients = []
    for mode in bank.modes:
        coefficients.append((field.tensor * mode.conj()).sum(dim=(-3, -2, -1)))
    return torch.stack(coefficients, dim=1)
