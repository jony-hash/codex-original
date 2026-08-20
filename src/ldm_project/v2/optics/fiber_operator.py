"""Research-only 10x10 SMF LP01 port operator."""
from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.v_number <= 0 or self.pitch_m <= 0 or self.wavelength_m <= 0:
            raise ValueError("fiber geometry values must be positive")
        if self.facet_magnification <= 0 or len(self.lateral_offset_m) != 2:
            raise ValueError("invalid facet magnification or (y,x) offset")
        if self.mode_model not in {"step_index_lp01_lowdin", "step_index_lp01_ports"}:
            raise ValueError("unsupported LP01 mode model")


def lp01_radial_profile(rho: torch.Tensor, v_number: float = 2.2) -> torch.Tensor:
    """Normalized LP01 proxy; radial sampling is dimensionless and offline-testable."""
    if abs(v_number - 2.2) > 1e-12:
        raise ValueError("offline LP01 fixture is defined only for V=2.2")
    u, w = 1.675, 1.427
    inside = torch.special.bessel_j0(torch.tensor(u) * rho)
    outside = torch.special.bessel_j0(torch.tensor(u)) * torch.special.modified_bessel_k0(torch.tensor(w) * rho) / torch.special.modified_bessel_k0(torch.tensor(w))
    return torch.where(rho <= 1.0, inside, outside).to(torch.float32)


class FiberArrayOLS(nn.Module):
    """Port-domain diagonal SMF transport with explicit field provenance."""

    def __init__(self, input_plane: PlaneSpec, spec: FiberArraySpec = FiberArraySpec(), transport: torch.Tensor | None = None) -> None:
        super().__init__()
        if input_plane.shape != spec.grid_shape:
            raise ValueError("fiber port operator requires a matching 10x10 physical plane")
        self.input_plane = input_plane
        self.spec = spec
        ports = spec.grid_shape[0] * spec.grid_shape[1]
        if transport is None:
            transport = torch.eye(ports, dtype=torch.complex64)
        if transport.shape != (ports, ports) or not torch.is_complex(transport):
            raise ValueError("transport must be a square complex port matrix")
        if float(torch.linalg.svdvals(transport).max()) > 1.0 + 1e-5:
            raise ValueError("transport matrix is non-passive")
        self.register_buffer("transport", transport)

    def forward(self, field: ComplexField) -> ComplexField:
        if field.plane != self.input_plane:
            raise ValueError("fiber input plane mismatch")
        flat = field.tensor.reshape(*field.tensor.shape[:-2], -1)
        output = torch.matmul(flat, self.transport.T).reshape_as(field.tensor)
        y_offset, x_offset = self.spec.lateral_offset_m
        output_plane = PlaneSpec("fiber_output", field.plane.shape, tuple(p * self.spec.facet_magnification for p in field.plane.pixel_pitch_m), field.plane.wavelength_m, field.plane.z_m, "fiber_ports", (field.plane.origin_m[0] + y_offset, field.plane.origin_m[1] + x_offset))
        return ComplexField(output, output_plane, OperationProvenance("synthetic_smf_port_transport", field.plane.plane_id, 0.0, {"mode_model": self.spec.mode_model, "v_number": self.spec.v_number, "facet_magnification": self.spec.facet_magnification, "lateral_offset_m": list(self.spec.lateral_offset_m), "physical_parameter_status": "synthetic diagnostic-only"}))


class FiberTransportEnsembleOLS(FiberArrayOLS):
    """Same operator with deterministic research pool selection per logical batch."""

    def __init__(self, input_plane: PlaneSpec, pool: FiberTransportPool, spec: FiberArraySpec = FiberArraySpec()) -> None:
        super().__init__(input_plane, spec, pool.matrix("train", 0))
        self._pool = pool
        self._active_id = "train-00"

    @property
    def active_id(self) -> str:
        return self._active_id

    def select(self, matrix_id: str) -> None:
        for pool_name, records in self._pool.records.items():
            for index, record in enumerate(records):
                if record.matrix_id == matrix_id:
                    self.transport.copy_(self._pool.matrix(pool_name, index).to(self.transport))
                    self._active_id = matrix_id
                    return
        raise KeyError(f"unknown transport ID: {matrix_id}")
