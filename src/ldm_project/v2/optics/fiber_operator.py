"""Research-only LP01 port-domain fiber operator.

The formal v2 system remains identity OLS. The research boundary is a
weighted complex overlap operator on the 100x100 OLS plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from ..contracts import ComplexField, OperationProvenance, PlaneSpec
from .fiber_pool import FiberTransportPool


@dataclass(frozen=True)
class FiberArraySpec:
    grid_shape: tuple[int, int] = (10, 10)
    pitch_m: float = 2.0e-6
    wavelength_m: float = 532.0e-9
    waist_m: float = 7.111e-6
    fill: float = 0.8
    v_number: float = 2.2
    facet_magnification: float = 1.0
    lateral_offset_m: tuple[float, float] = (0.0, 0.0)
    mode_model: str = "step_index_lp01_ports"
    expected_fov_m: tuple[float, float] = (200.0e-6, 200.0e-6)

    def __post_init__(self) -> None:
        if self.grid_shape != (10, 10):
            raise ValueError("the v2 research contract is fixed to a 10x10 SMF array")
        if self.v_number <= 0.0 or self.v_number >= 2.4048255577:
            raise ValueError("LP01 requires 0 < V < LP11 cutoff")
        if self.pitch_m <= 0.0 or self.wavelength_m <= 0.0 or self.waist_m <= 0.0:
            raise ValueError("fiber geometry values must be positive")
        if len(self.expected_fov_m) != 2 or any(value <= 0.0 for value in self.expected_fov_m):
            raise ValueError("expected_fov_m must contain two positive values")
        if not 0.0 < self.fill <= 1.0 or self.facet_magnification <= 0.0:
            raise ValueError("fill and facet_magnification must be positive and bounded")
        if len(self.lateral_offset_m) != 2:
            raise ValueError("lateral_offset_m must be an independent (y, x) pair")
        if self.mode_model not in {"step_index_lp01_lowdin", "step_index_lp01_ports"}:
            raise ValueError("unsupported LP01 mode model")


def _characteristic(u: float, v: float) -> float:
    w = math.sqrt(max(v * v - u * u, 1.0e-30))
    u_t = torch.tensor(u, dtype=torch.float64)
    w_t = torch.tensor(w, dtype=torch.float64)
    return float(u_t * torch.special.bessel_j1(u_t) / torch.special.bessel_j0(u_t) - w_t * torch.special.modified_bessel_k1(w_t) / torch.special.modified_bessel_k0(w_t))


@lru_cache(maxsize=32)
def lp01_eigenpair(v_number: float) -> tuple[float, float, float]:
    """Return ``(u, w, rho_1e)`` from the scalar LP01 equation."""
    v = float(v_number)
    if not 0.0 < v < 2.4048255577:
        raise ValueError("LP01 requires 0 < V < LP11 cutoff")
    lo, hi = 1.0e-10, min(v - 1.0e-10, 2.4048255577 - 1.0e-10)
    if _characteristic(lo, v) >= 0.0 or _characteristic(hi, v) <= 0.0:
        raise ValueError("LP01 characteristic equation has no bracket")
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if _characteristic(mid, v) > 0.0:
            hi = mid
        else:
            lo = mid
    u = (lo + hi) / 2.0
    w = math.sqrt(v * v - u * u)
    target = math.exp(-1.0)
    boundary = float(torch.special.bessel_j0(torch.tensor(u, dtype=torch.float64)))
    denom = float(torch.special.modified_bessel_k0(torch.tensor(w, dtype=torch.float64)))
    def profile(rho: float) -> float:
        if rho <= 1.0:
            return float(torch.special.bessel_j0(torch.tensor(u * rho, dtype=torch.float64)))
        return boundary * float(torch.special.modified_bessel_k0(torch.tensor(w * rho, dtype=torch.float64))) / denom
    left, right = 1.0, max(2.0, 4.0 / max(w, 1.0e-12))
    while profile(right) > target:
        right *= 2.0
    for _ in range(80):
        mid = (left + right) / 2.0
        if profile(mid) > target:
            left = mid
        else:
            right = mid
    return u, w, (left + right) / 2.0


def lp01_radial_profile(rho: torch.Tensor, v_number: float = 2.2) -> torch.Tensor:
    """Scalar LP01 amplitude on dimensionless radius ``rho=r/a``."""
    u, w, _ = lp01_eigenpair(float(v_number))
    work = rho.to(torch.float64)
    u_t = torch.tensor(u, dtype=torch.float64, device=rho.device)
    w_t = torch.tensor(w, dtype=torch.float64, device=rho.device)
    j0_u = torch.special.bessel_j0(u_t)
    k0_w = torch.special.modified_bessel_k0(w_t)
    profile = torch.where(work <= 1.0, torch.special.bessel_j0(u_t * work), j0_u * torch.special.modified_bessel_k0(w_t * work) / k0_w)
    return profile.to(dtype=rho.dtype if rho.is_floating_point() else torch.float32)


def lp01_reference_metrics(v_number: float = 2.2) -> dict[str, float]:
    u, w, rho_e = lp01_eigenpair(v_number)
    boundary = float(torch.special.bessel_j0(torch.tensor(u, dtype=torch.float64)))
    denom = float(torch.special.modified_bessel_k0(torch.tensor(w, dtype=torch.float64)))
    inside_derivative = -u * float(torch.special.bessel_j1(torch.tensor(u, dtype=torch.float64)))
    outside_derivative = -boundary * w * float(torch.special.modified_bessel_k1(torch.tensor(w, dtype=torch.float64))) / denom
    return {"u": u, "w": w, "v_number": float(v_number), "closure_error": abs(u * u + w * w - v_number * v_number), "characteristic_residual": abs(_characteristic(u, v_number)), "boundary_derivative_relative_error": abs(inside_derivative - outside_derivative) / max(abs(inside_derivative), 1.0e-30), "amplitude_1e_radius_rho": rho_e}


def lp01_finite_difference_eigenpair(v_number: float = 2.2, *, points: int = 512, radius: float = 8.0) -> tuple[float, float]:
    """Independent radial finite-volume bound-state estimate.

    This deliberately does not call the scalar Bessel characteristic equation.
    It discretizes ``-(r R')'/r - V² 1[r<1] R = lambda R`` on a weighted
    finite-volume grid; the fundamental negative eigenvalue gives ``w`` and
    ``u`` through ``lambda=-w²`` and ``u²+w²=V²``.
    """
    if points < 64 or radius <= 1.0:
        raise ValueError("finite-difference radial grid is too small")
    dr = radius / float(points)
    r = (torch.arange(points, dtype=torch.float64) + 0.5) * dr
    faces = torch.arange(points + 1, dtype=torch.float64) * dr
    mass = r
    stiffness = torch.zeros((points, points), dtype=torch.float64)
    conductance = faces[1:-1] / dr
    diagonal = torch.zeros(points, dtype=torch.float64)
    diagonal[:-1] += conductance / dr
    diagonal[1:] += conductance / dr
    diagonal[-1] += faces[-1] / dr
    stiffness.diagonal().copy_(diagonal - float(v_number * v_number) * mass * (r <= 1.0))
    stiffness.diagonal(1).copy_(-conductance / dr)
    stiffness.diagonal(-1).copy_(-conductance / dr)
    inv_sqrt_mass = mass.rsqrt()
    symmetric = inv_sqrt_mass[:, None] * stiffness * inv_sqrt_mass[None, :]
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    negative = eigenvalues[eigenvalues < 0.0]
    if negative.numel() == 0:
        raise ValueError("finite-difference radial solve found no bound LP01 mode")
    w = float((-negative[0]).sqrt())
    u = math.sqrt(max(float(v_number * v_number) - w * w, 0.0))
    return u, w


def lp01_mode_overlap_metrics(plane: PlaneSpec, spec: FiberArraySpec) -> dict[str, float]:
    """Measure the actual Lowdin-vs-reference overlap used by validation."""
    raw = _mode_bank(plane, spec, offset_m=(0.0, 0.0), lowdin=False)
    lowdin = _mode_bank(plane, spec, offset_m=(0.0, 0.0), lowdin=True)
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    overlaps = (lowdin.flatten(1).conj() * raw.flatten(1)).sum(dim=1).abs() * area
    return {"lowdin_overlap_min": float(overlaps.min()), "lowdin_overlap_median": float(overlaps.median())}


@lru_cache(maxsize=64)
def _mode_bank(plane: PlaneSpec, spec: FiberArraySpec, *, offset_m: tuple[float, float], lowdin: bool = False) -> torch.Tensor:
    h, w = plane.shape
    y = (torch.arange(h, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[0] - plane.fov_m[0] / 2.0
    x = (torch.arange(w, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[1] - plane.fov_m[1] / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    facet_y = spec.facet_magnification * yy - float(offset_m[0])
    facet_x = spec.facet_magnification * xx - float(offset_m[1])
    facet_fov_y, facet_fov_x = plane.fov_m[0] * spec.facet_magnification, plane.fov_m[1] * spec.facet_magnification
    center_y = torch.linspace(-facet_fov_y * spec.fill / 2.0, facet_fov_y * spec.fill / 2.0, 10, dtype=torch.float64)
    center_x = torch.linspace(-facet_fov_x * spec.fill / 2.0, facet_fov_x * spec.fill / 2.0, 10, dtype=torch.float64)
    u, w_const, rho_e = lp01_eigenpair(spec.v_number)
    del u, w_const
    core_radius = spec.waist_m / rho_e
    profiles: list[torch.Tensor] = []
    for yi, cy in enumerate(center_y):
        for xi, cx in enumerate(center_x):
            radius = torch.sqrt((facet_y - cy).square() + (facet_x - cx).square())
            profile = lp01_radial_profile(radius / core_radius, spec.v_number)
            if not lowdin:
                spacing_y = float(center_y[1] - center_y[0])
                spacing_x = float(center_x[1] - center_x[0])
                lower_y = -facet_fov_y / 2.0 if yi == 0 else float((center_y[yi - 1] + cy) / 2.0)
                upper_y = facet_fov_y / 2.0 if yi == 9 else float((cy + center_y[yi + 1]) / 2.0)
                lower_x = -facet_fov_x / 2.0 if xi == 0 else float((center_x[xi - 1] + cx) / 2.0)
                upper_x = facet_fov_x / 2.0 if xi == 9 else float((cx + center_x[xi + 1]) / 2.0)
                del spacing_y, spacing_x
                profile = profile * (facet_y >= lower_y) * (facet_y < upper_y) * (facet_x >= lower_x) * (facet_x < upper_x)
            profiles.append(profile.reshape(-1))
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    bank = torch.stack(profiles, dim=0)
    bank = bank / (bank.square().sum(dim=1, keepdim=True) * area).clamp_min(1.0e-30).sqrt()
    if lowdin:
        gram = bank @ bank.T * area
        values, vectors = torch.linalg.eigh(gram)
        if float(values.min()) <= 1.0e-8:
            raise ValueError("LP01 Löwdin bank is rank deficient")
        bank = vectors @ torch.diag(values.rsqrt()) @ vectors.T @ bank
    return torch.complex(bank.to(torch.float32), torch.zeros_like(bank, dtype=torch.float32)).reshape(100, h, w)


class FiberArrayOLS(nn.Module):
    """100x100 complex field -> 100 LP01 ports -> 100x100 complex field."""

    diagnostic_only = True

    def __init__(self, input_plane: PlaneSpec, spec: FiberArraySpec = FiberArraySpec(), transport: torch.Tensor | None = None) -> None:
        super().__init__()
        if input_plane.shape != (100, 100):
            raise ValueError("fiber OLS requires the formal 100x100 encoder plane")
        if any(abs(a - b) > 1.0e-15 for a, b in zip(input_plane.pixel_pitch_m, (spec.pitch_m, spec.pitch_m), strict=True)):
            raise ValueError("fiber pixel pitch does not match the encoder PlaneSpec")
        if any(abs(a - b) > 1.0e-12 for a, b in zip(input_plane.fov_m, spec.expected_fov_m, strict=True)):
            raise ValueError("fiber FOV does not match the research geometry")
        if abs(input_plane.wavelength_m - spec.wavelength_m) > 1.0e-15:
            raise ValueError("fiber wavelength does not match the encoder PlaneSpec")
        if transport is None:
            transport = torch.eye(100, dtype=torch.complex64)
        if transport.shape != (100, 100) or not torch.is_complex(transport):
            raise ValueError("transport must be a 100x100 complex port matrix")
        if float(torch.linalg.svdvals(transport).max()) > 1.0 + 1.0e-5:
            raise ValueError("transport matrix is non-passive")
        self.input_plane = input_plane
        self.spec = spec
        lowdin = spec.mode_model == "step_index_lp01_lowdin"
        self.register_buffer("input_modes", _mode_bank(input_plane, spec, offset_m=spec.lateral_offset_m, lowdin=lowdin))
        self.register_buffer("output_modes", _mode_bank(input_plane, spec, offset_m=(0.0, 0.0), lowdin=lowdin))
        self.register_buffer("transport", transport)
        self.last_diagnostics: dict[str, object] = {}

    def _banks(self, field: ComplexField) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_modes.to(field.tensor), self.output_modes.to(field.tensor)

    def forward(self, field: ComplexField) -> ComplexField:
        if (field.plane.shape != self.input_plane.shape or field.plane.pixel_pitch_m != self.input_plane.pixel_pitch_m or abs(field.plane.wavelength_m - self.input_plane.wavelength_m) > 1.0e-15 or field.plane.fov_m != self.input_plane.fov_m):
            raise ValueError("fiber input plane mismatch")
        if field.tensor.ndim not in {3, 4} or (field.tensor.ndim == 4 and field.tensor.shape[1] != 1):
            raise ValueError("fiber OLS requires [batch,H,W] or [batch,1,H,W] complex fields")
        values = field.tensor[:, 0] if field.tensor.ndim == 4 else field.tensor
        if not torch.isfinite(values).all():
            raise ValueError("fiber OLS input must be finite")
        input_modes, output_modes = self._banks(field)
        area = self.input_plane.pixel_pitch_m[0] * self.input_plane.pixel_pitch_m[1]
        coefficients = torch.einsum("mhw,bhw->bm", input_modes.conj(), values) * area
        output_coefficients = coefficients @ self.transport.to(values).T
        output_values = torch.einsum("bm,mhw->bhw", output_coefficients, output_modes)
        input_energy = values.abs().square().sum(dim=(-2, -1)) * area
        output_energy = output_values.abs().square().sum(dim=(-2, -1)) * area
        if torch.any(output_energy > input_energy * (1.0 + 1.0e-5) + 1.0e-12):
            raise ValueError("fiber OLS violates the input/output energy gate")
        input_energy_max = float(input_energy.max().detach())
        output_energy_max = float(output_energy.max().detach())
        gram = input_modes.flatten(1).conj() @ input_modes.flatten(1).T * area
        singular_values = torch.linalg.svdvals(self.transport)
        self.last_diagnostics = {"input_energy_max": input_energy_max, "output_energy_max": output_energy_max, "input_energy": input_energy.detach(), "output_energy": output_energy.detach(), "coupling_efficiency": (coefficients.abs().square().sum(dim=1) / input_energy.clamp_min(1.0e-30)).detach(), "transport_singular_max": float(singular_values.max()), "transport_singular_min": float(singular_values.min()), "port_gram_rank": int(torch.linalg.matrix_rank(gram).item()), "port_gram_condition": float(torch.linalg.cond(gram)), "active_offset_m": list(self.spec.lateral_offset_m)}
        output = output_values.unsqueeze(1) if field.tensor.ndim == 4 else output_values
        return ComplexField(output, self.input_plane.with_plane("ols_output"), OperationProvenance("synthetic_smf_lp01_port_transport", self.input_plane.plane_id, 0.0, {"mode_model": self.spec.mode_model, "v_number": self.spec.v_number, "facet_magnification": self.spec.facet_magnification, "lateral_offset_m": list(self.spec.lateral_offset_m), "physical_parameter_status": "synthetic diagnostic-only"}))


class FiberTransportEnsembleOLS(FiberArrayOLS):
    """Fiber operator whose active transport and input offset are selected by ID."""

    def __init__(self, input_plane: PlaneSpec, pool: FiberTransportPool, spec: FiberArraySpec = FiberArraySpec()) -> None:
        record = pool.record("train-00")
        active_spec = FiberArraySpec(**{**spec.__dict__, "lateral_offset_m": record.offset_m})
        super().__init__(input_plane, active_spec, pool.matrix_by_id(record.matrix_id))
        self.pool = pool
        self._active_id = record.matrix_id

    @property
    def active_id(self) -> str:
        return self._active_id

    def select(self, matrix_id: str) -> None:
        record = self.pool.record(matrix_id)
        self.spec = FiberArraySpec(**{**self.spec.__dict__, "lateral_offset_m": record.offset_m})
        self.transport.copy_(self.pool.matrix_by_id(matrix_id).to(self.transport))
        self.input_modes.copy_(_mode_bank(self.input_plane, self.spec, offset_m=record.offset_m, lowdin=self.spec.mode_model == "step_index_lp01_lowdin").to(self.input_modes))
        self._active_id = matrix_id
