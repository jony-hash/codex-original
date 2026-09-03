"""Research-only synthetic nested six-mode-per-port multimode operator.

This module extends the existing diagnostic SMF/FMF modal family without
changing either existing mode bank.  It is a sampled synthetic modal-subspace
expansion, not real MMF propagation: there is no fiber length, beta phase,
dispersion, coupling, loss, or hardware calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import ComplexField, OperationProvenance, PlaneSpec
from .few_mode_fiber_operator import FewModeFiberArrayOLS, FewModeFiberArraySpec
from .fiber_operator import lp01_eigenpair


PORT_COUNT = 100
PORT_GRID = (10, 10)
MODES_PER_PORT = 6
TOTAL_MODE_COUNT = PORT_COUNT * MODES_PER_PORT
MODE_FAMILY_ORDER = ("LP01", "LP11a", "ETA1", "XI2", "XIETA", "ETA2")
MODE_DEGREE_ORDER = (0, 1, 1, 2, 2, 2)
EXTRA_FAMILY_ORDER = ("ETA1", "XI2", "XIETA", "ETA2")
CANONICAL_SIGN_EPSILON = 1.0e-12
RANK_EPSILON = 1.0e-12
PROVENANCE_OPERATION = "synthetic_nested_mmf_m6_second_order_identity_transport"
ORDERING_CONTRACT = (
    "exact_fmf_prefix_interleaved_per_port_then_additional_candidate_blocks_y_then_x"
)


@dataclass(frozen=True)
class NestedMultimodeFiberArraySpec:
    """Fixed geometry and six-mode-per-port M6 diagnostic contract."""

    modes: tuple[str, ...] = MODE_FAMILY_ORDER
    array_shape: tuple[int, int] = PORT_GRID
    pitch_m: float = 2.0e-6
    wavelength_m: float = 532.0e-9
    waist_m: float = 7.111e-6
    fill: float = 0.8
    v_number: float = 2.2
    expected_fov_m: tuple[float, float] = (200.0e-6, 200.0e-6)

    def __post_init__(self) -> None:
        if self.modes != MODE_FAMILY_ORDER:
            raise ValueError("the synthetic MMF contract is fixed to the M6 mode family")
        if self.array_shape != PORT_GRID:
            raise ValueError("the synthetic MMF contract is fixed to a 10x10 array")
        if not 0.0 < self.v_number < 2.4048255577:
            raise ValueError("the inherited LP01 contract requires 0 < V < LP11 cutoff")
        if self.pitch_m <= 0.0 or self.wavelength_m <= 0.0 or self.waist_m <= 0.0:
            raise ValueError("MMF geometry values must be positive")
        if not 0.0 < self.fill <= 1.0:
            raise ValueError("MMF fill must be positive and bounded")
        if len(self.expected_fov_m) != 2 or any(value <= 0.0 for value in self.expected_fov_m):
            raise ValueError("expected_fov_m must contain two positive values")


def _candidate_bank(
    plane: PlaneSpec,
    spec: NestedMultimodeFiberArraySpec,
    lp01: torch.Tensor,
    family: str,
) -> torch.Tensor:
    """Build one deterministic 100-port real candidate block."""
    if family not in EXTRA_FAMILY_ORDER:
        raise ValueError(f"unsupported M6 candidate family: {family}")
    if tuple(lp01.shape) != (PORT_COUNT, *plane.shape):
        raise ValueError("M6 candidate generation requires the 100-port LP01 bank")

    height, width = plane.shape
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[0]
    y = y - plane.fov_m[0] / 2.0
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[1]
    x = x - plane.fov_m[1] / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    facet_fov_y = plane.fov_m[0] * spec.fill
    facet_fov_x = plane.fov_m[1] * spec.fill
    centers_y = torch.linspace(-facet_fov_y / 2.0, facet_fov_y / 2.0, 10, dtype=torch.float64)
    centers_x = torch.linspace(-facet_fov_x / 2.0, facet_fov_x / 2.0, 10, dtype=torch.float64)
    _u, _w, rho_e = lp01_eigenpair(spec.v_number)
    core_radius = spec.waist_m / rho_e
    candidates: list[torch.Tensor] = []
    for port in range(PORT_COUNT):
        port_y, port_x = divmod(port, PORT_GRID[1])
        local_x = (xx - centers_x[port_x]) / core_radius
        local_y = (yy - centers_y[port_y]) / core_radius
        if family == "ETA1":
            multiplier = local_y
        elif family == "XI2":
            multiplier = local_x.square()
        elif family == "XIETA":
            multiplier = local_x * local_y
        else:
            multiplier = local_y.square()
        candidates.append(lp01[port].real.to(torch.float64) * multiplier)
    return torch.stack(candidates, dim=0)


def _orthonormalize_residual_block(
    accepted: torch.Tensor,
    candidates: torch.Tensor,
    area: float,
    family: str,
) -> torch.Tensor:
    """Project a candidate block against accepted modes and whiten its residual.

    The accepted basis is never modified.  The Gram solve makes the projection
    exact for the existing FMF basis' measured numerical Gram matrix rather
    than assuming that its float32 representation is mathematically exact.
    """
    if candidates.ndim != 3 or accepted.ndim != 3:
        raise ValueError("M6 orthogonalization requires [mode,H,W] tensors")
    if candidates.shape[1:] != accepted.shape[1:]:
        raise ValueError("M6 candidate and accepted mode planes must match")
    raw = candidates.reshape(candidates.shape[0], -1)
    base = accepted.reshape(accepted.shape[0], -1)
    base_gram = base @ base.T * area
    if not torch.isfinite(base_gram).all() or int(torch.linalg.matrix_rank(base_gram)) != base.shape[0]:
        raise RuntimeError(f"synthetic_mmf_contract_rank_failure: accepted basis for {family}")
    overlap = raw @ base.T * area
    projection_coefficients = torch.linalg.solve(base_gram, overlap.T).T
    residual = raw - projection_coefficients @ base
    gram = residual @ residual.T * area
    if not torch.isfinite(gram).all():
        raise RuntimeError(f"synthetic_mmf_contract_rank_failure: nonfinite Gram for {family}")
    values, vectors = torch.linalg.eigh(gram)
    if values.numel() != candidates.shape[0] or float(values.min()) <= RANK_EPSILON:
        raise RuntimeError(f"synthetic_mmf_contract_rank_failure: {family} block is not full rank")
    whitened = vectors @ torch.diag(values.rsqrt()) @ vectors.T @ residual
    if not torch.isfinite(whitened).all():
        raise RuntimeError(f"synthetic_mmf_contract_rank_failure: nonfinite modes for {family}")
    flattened = whitened.reshape(whitened.shape[0], -1)
    for row in range(flattened.shape[0]):
        pivots = torch.nonzero(flattened[row].abs() > CANONICAL_SIGN_EPSILON, as_tuple=False)
        if pivots.numel() == 0:
            raise RuntimeError(f"synthetic_mmf_contract_rank_failure: no sign pivot for {family}")
        pivot = int(pivots[0, 0])
        if float(flattened[row, pivot]) < 0.0:
            flattened[row].neg_()
    return whitened.reshape_as(candidates)


def _extend_fmf_prefix(
    plane: PlaneSpec,
    spec: NestedMultimodeFiberArraySpec,
    fmf_modes: torch.Tensor,
) -> torch.Tensor:
    """Append four full-rank degree-two blocks to an unchanged FMF prefix."""
    if tuple(fmf_modes.shape) != (200, *plane.shape):
        raise ValueError("M6 construction requires the 200-mode FMF prefix")
    if not torch.is_complex(fmf_modes) or fmf_modes.dtype != torch.complex64:
        raise ValueError("M6 construction requires the existing complex64 FMF bank")
    if float(fmf_modes.imag.abs().max()) != 0.0:
        raise ValueError("M6 synthetic candidates require the existing real-valued FMF bank")

    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    accepted = fmf_modes.real.to(torch.float64)
    blocks: list[torch.Tensor] = []
    for family in EXTRA_FAMILY_ORDER:
        candidate = _candidate_bank(plane, spec, fmf_modes[0::2], family)
        block = _orthonormalize_residual_block(accepted, candidate, area, family)
        block_complex = torch.complex(
            block.to(torch.float32), torch.zeros_like(block, dtype=torch.float32)
        )
        blocks.append(block_complex)
        accepted = torch.cat((accepted, block), dim=0)
    result = torch.cat((fmf_modes, *blocks), dim=0)
    if tuple(result.shape) != (TOTAL_MODE_COUNT, *plane.shape):
        raise RuntimeError("synthetic_mmf_contract_rank_failure: final bank shape is not [600,H,W]")
    if not torch.isfinite(result).all():
        raise RuntimeError("synthetic_mmf_contract_rank_failure: final bank is not finite")
    return result


def _flat_metadata() -> dict[str, object]:
    """Return metadata matching the block-preserving flat tensor ordering."""
    names: list[str] = []
    degrees: list[int] = []
    ports: list[int] = []
    port_x: list[int] = []
    port_y: list[int] = []
    for port in range(PORT_COUNT):
        y_index, x_index = divmod(port, PORT_GRID[1])
        names.extend(("LP01", "LP11a"))
        degrees.extend((0, 1))
        ports.extend((port, port))
        port_x.extend((x_index, x_index))
        port_y.extend((y_index, y_index))
    for family, degree in zip(EXTRA_FAMILY_ORDER, MODE_DEGREE_ORDER[2:], strict=True):
        for port in range(PORT_COUNT):
            y_index, x_index = divmod(port, PORT_GRID[1])
            names.append(family)
            degrees.append(degree)
            ports.append(port)
            port_x.append(x_index)
            port_y.append(y_index)
    return {
        "mode_family_names": names,
        "mode_degree": degrees,
        "port_index": ports,
        "port_x": port_x,
        "port_y": port_y,
        "mode_family_order": list(MODE_FAMILY_ORDER),
        "ordering_contract": ORDERING_CONTRACT,
    }


class NestedMultimodeFiberArrayOLS(nn.Module):
    """Diagnostic 100-port, six-mode-per-port synthetic nested MMF operator."""

    diagnostic_only = True
    modes_per_port = MODES_PER_PORT
    total_mode_count = TOTAL_MODE_COUNT
    operation = PROVENANCE_OPERATION
    ordering_contract = ORDERING_CONTRACT

    def __init__(
        self,
        input_plane: PlaneSpec,
        spec: NestedMultimodeFiberArraySpec = NestedMultimodeFiberArraySpec(),
    ) -> None:
        super().__init__()
        if input_plane.shape != (100, 100):
            raise ValueError("M6 OLS requires the formal 100x100 encoder plane")
        if input_plane.pixel_pitch_m != (spec.pitch_m, spec.pitch_m):
            raise ValueError("M6 pixel pitch does not match the encoder PlaneSpec")
        if any(abs(a - b) > 1.0e-12 for a, b in zip(input_plane.fov_m, spec.expected_fov_m, strict=True)):
            raise ValueError("M6 FOV does not match the encoder PlaneSpec")
        if abs(input_plane.wavelength_m - spec.wavelength_m) > 1.0e-15:
            raise ValueError("M6 wavelength does not match the encoder PlaneSpec")

        fmf_spec = FewModeFiberArraySpec(
            modes=("LP01", "LP11a"),
            array_shape=spec.array_shape,
            pitch_m=spec.pitch_m,
            wavelength_m=spec.wavelength_m,
            waist_m=spec.waist_m,
            fill=spec.fill,
            v_number=spec.v_number,
            expected_fov_m=spec.expected_fov_m,
        )
        fmf = FewModeFiberArrayOLS(input_plane, fmf_spec)
        inherited_input = fmf.input_modes.detach().clone()
        inherited_output = fmf.output_modes.detach().clone()
        if not torch.equal(inherited_input, inherited_output):
            raise RuntimeError("synthetic_mmf_contract_rank_failure: FMF input/output prefix diverged")
        input_modes = _extend_fmf_prefix(input_plane, spec, inherited_input)

        self.input_plane = input_plane
        self.output_plane = input_plane.with_plane("ols_output")
        self.spec = spec
        self.mode_names = MODE_FAMILY_ORDER
        metadata = _flat_metadata()
        self.mode_family_names = tuple(metadata["mode_family_names"])
        self.mode_degree = tuple(metadata["mode_degree"])
        self.port_index = tuple(metadata["port_index"])
        self.port_x = tuple(metadata["port_x"])
        self.port_y = tuple(metadata["port_y"])
        self.register_buffer("input_modes", input_modes)
        self.register_buffer("output_modes", input_modes.clone())
        self.register_buffer(
            "transport", torch.eye(TOTAL_MODE_COUNT, dtype=torch.complex64)
        )
        self.last_diagnostics: dict[str, object] = {}

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "mode_family_names": list(self.mode_family_names),
            "mode_degree": list(self.mode_degree),
            "port_index": list(self.port_index),
            "port_x": list(self.port_x),
            "port_y": list(self.port_y),
            "mode_family_order": list(MODE_FAMILY_ORDER),
            "ordering_contract": self.ordering_contract,
            "port_count": PORT_COUNT,
            "modes_per_port": MODES_PER_PORT,
            "total_mode_count": TOTAL_MODE_COUNT,
            "canonical_sign_epsilon": CANONICAL_SIGN_EPSILON,
            "rank_epsilon": RANK_EPSILON,
        }

    def _validate_field(self, field: ComplexField) -> None:
        if (
            field.plane.shape != self.input_plane.shape
            or any(
                abs(a - b) > 1.0e-15
                for a, b in zip(field.plane.pixel_pitch_m, self.input_plane.pixel_pitch_m, strict=True)
            )
            or abs(field.plane.wavelength_m - self.input_plane.wavelength_m) > 1.0e-15
            or any(abs(a - b) > 1.0e-12 for a, b in zip(field.plane.fov_m, self.input_plane.fov_m, strict=True))
            or field.plane.coordinate_frame != self.input_plane.coordinate_frame
        ):
            raise ValueError("M6 input plane mismatch")
        if field.tensor.ndim not in {3, 4} or (field.tensor.ndim == 4 and field.tensor.shape[1] != 1):
            raise ValueError("M6 OLS requires [batch,H,W] or [batch,1,H,W] complex fields")
        if not torch.is_complex(field.tensor):
            raise ValueError("M6 OLS requires a complex field")
        if not torch.isfinite(field.tensor).all():
            raise ValueError("M6 OLS input must be finite")

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
            raise ValueError("M6 OLS violates the input/output energy gate")

        family_coefficients = torch.stack(
            (
                output_coefficients[:, 0:200:2],
                output_coefficients[:, 1:200:2],
                output_coefficients[:, 200:300],
                output_coefficients[:, 300:400],
                output_coefficients[:, 400:500],
                output_coefficients[:, 500:600],
            ),
            dim=1,
        )
        singular_values = torch.linalg.svdvals(self.transport)
        self.last_diagnostics = {
            "input_energy": input_energy.detach(),
            "output_energy": output_energy.detach(),
            "coupling_efficiency": (
                coefficients.abs().square().sum(dim=1) / input_energy.clamp_min(1.0e-30)
            ).detach(),
            "coefficients": coefficients.detach(),
            "port_input_coefficients": coefficients.detach(),
            "port_output_coefficients": output_coefficients.detach(),
            "mode_coefficients": family_coefficients.detach(),
            "mode_family_coefficients": family_coefficients.detach(),
            "mode_power": family_coefficients.abs().square().sum(dim=2).detach(),
            "active_mode_fraction": (
                (output_coefficients.abs().square() > 1.0e-12).to(torch.float32).mean(dim=1)
            ).detach(),
            "transport_singular_max": float(singular_values.max()),
            "transport_singular_min": float(singular_values.min()),
            "mode_count": MODES_PER_PORT,
            "total_mode_count": TOTAL_MODE_COUNT,
        }
        output = output_values.unsqueeze(1) if field.tensor.ndim == 4 else output_values
        return ComplexField(
            output,
            self.output_plane,
            OperationProvenance(
                PROVENANCE_OPERATION,
                self.input_plane.plane_id,
                0.0,
                {
                    **self.metadata,
                    "transport": "600x600 complex identity",
                    "physical_parameter_status": "synthetic diagnostic-only",
                    "scientific_boundary": "not_real_mmf_propagation",
                },
            ),
        )


__all__ = ["NestedMultimodeFiberArrayOLS", "NestedMultimodeFiberArraySpec"]
