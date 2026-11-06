"""Finite-pixel Gauss quadrature model for the diagnostic LightGen400 line.

The phase masks remain 400 by 400 physical pixels. Gauss nodes integrate the
Rayleigh--Sommerfeld kernel inside each physical pixel; node count does not add
trainable phase parameters. MNIST intensity is piecewise constant within the
400 physical SLM cells after its one declared bilinear resize.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .chen_lightgen_400_baseline import (
    ACTIVE_SHAPE,
    FIBER_MFD_M,
    FIBER_PITCH_M,
    LAYER_POWER_TRANSMISSION,
    PIXEL_PITCH_M,
    SOURCE_POWER_W,
    WAVELENGTH_M,
)


@dataclass(frozen=True)
class FinitePixelConfig:
    quadrature_order: int = 4
    window_cells: int = 4096
    distance_m: float = 3.0e-3
    checkpoint_propagation: bool = True
    propagation_backend: str = "angular_spectrum"
    rs_compute_precision: str = "complex128"
    rs_pair_block: int = 4
    fft_embedding_extra_cells: int = 0
    window_power_tile_cells: int = 512
    window_power_source_chunk: int = 4
    window_power_target_chunk: int = 4

    def __post_init__(self) -> None:
        if self.quadrature_order not in (2, 4, 8, 12, 16):
            raise ValueError("finite-pixel Gauss order must be 2, 4, 8, 12, or 16")
        if self.window_cells < 2 or self.window_cells % 2:
            raise ValueError("finite-pixel physical window must be even and contain at least two cells")
        if not math.isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("finite-pixel propagation distance must be finite and positive")
        if self.propagation_backend not in ("angular_spectrum", "rs_linear", "rs_linear_cached"):
            raise ValueError("unsupported finite-pixel propagation backend")
        if self.rs_compute_precision not in ("complex128", "complex64"):
            raise ValueError("RS compute precision must be complex128 or complex64")
        if not isinstance(self.rs_pair_block, int) or isinstance(self.rs_pair_block, bool) \
                or self.rs_pair_block < 1:
            raise ValueError("RS pair block must be a positive integer")
        if not isinstance(self.fft_embedding_extra_cells, int) or isinstance(self.fft_embedding_extra_cells, bool) \
                or self.fft_embedding_extra_cells < 0:
            raise ValueError("extra FFT embedding cells cannot be negative")
        if not isinstance(self.window_power_tile_cells, int) or isinstance(self.window_power_tile_cells, bool) \
                or self.window_power_tile_cells < 1:
            raise ValueError("full-window power tile size must be positive")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in
               (self.window_power_source_chunk, self.window_power_target_chunk)):
            raise ValueError("full-window source and target node chunks must be positive")

    @property
    def subgrid_factor(self) -> int:
        return min(self.quadrature_order, 8)

    @property
    def fft_size(self) -> int:
        return self.fft_size_for(ACTIVE_SHAPE)

    def fft_size_for(self, active_shape: tuple[int, int]) -> int:
        # Zero-pad the active source aperture plus the full observation window
        # so the central output is a linear-convolution result, not a periodic
        # copy of the source wrapped through the FFT boundary.
        minimum = (self.window_cells + max(active_shape) - 1) * self.subgrid_factor
        return _next_fast_fft_size(minimum)

    @property
    def sample_pitch_m(self) -> float:
        return PIXEL_PITCH_M / self.subgrid_factor


def _legendre_rule(order: int, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        torch.as_tensor(nodes, dtype=torch.float64, device=device),
        torch.as_tensor(weights / 2.0, dtype=torch.float64, device=device),
    )


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a = candidate.to(torch.complex128).reshape(-1)
    b = reference.to(torch.complex128).reshape(-1)
    overlap = torch.vdot(b, a)
    aligned = a * torch.exp(-1j * torch.angle(overlap))
    return float(torch.linalg.vector_norm(aligned - b) /
                 torch.linalg.vector_norm(b).clamp_min(1.0e-300))


def _next_fast_fft_size(minimum: int) -> int:
    """Return the first size >= minimum whose prime factors are 2, 3, 5, or 7."""
    value = max(2, int(minimum))
    while True:
        remainder = value
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return value
        value += 1


class _UnvalidatedFinitePixelRSLinear(nn.Module):
    """Earlier FFT implementation retained only for source-level comparison."""

    CACHE_LIMIT_BYTES = 3 * 1024**3

    def __init__(self, order: int, active_shape: tuple[int, int], config: FinitePixelConfig) -> None:
        super().__init__()
        self.order = order
        self.active_shape = active_shape
        self.config = config
        nodes, weights = _legendre_rule(order)
        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        self.register_buffer("node_y", yy.reshape(-1), persistent=True)
        self.register_buffer("node_x", xx.reshape(-1), persistent=True)
        self.register_buffer("node_weights", (wy * wx).reshape(-1), persistent=True)
        self.register_buffer("_cache_anchor", torch.empty(0), persistent=False)
        self._kernel_cache: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
        self._kernel_cache_bytes = 0

    @staticmethod
    def _fft_shape(source_shape: tuple[int, int], output_shape: tuple[int, int]) -> tuple[int, int]:
        return tuple(_next_fast_fft_size(a + b - 1) for a, b in zip(source_shape, output_shape, strict=True))

    def _kernel_spectrum(
        self, output_shape: tuple[int, int], beta: int, alpha_start: int, alpha_stop: int,
        offset: tuple[int, int], fft_shape: tuple[int, int], device: torch.device,
        *, dtype: torch.dtype, beta_offset: tuple[float, float] | None = None,
        cache_tag: str = "node",
    ) -> torch.Tensor:
        key = (cache_tag, output_shape, beta, alpha_start, alpha_stop, offset,
               fft_shape, str(device), dtype)
        cached = self._kernel_cache.get(key)
        if cached is not None:
            self._kernel_cache.move_to_end(key)
            return cached

        source_height, source_width = self.active_shape
        output_height, output_width = output_shape
        kernel_height = source_height + output_height - 1
        kernel_width = source_width + output_width - 1
        lag_y = (torch.arange(kernel_height, dtype=torch.float64, device=device) - (source_height - 1)
                 + offset[0]) * PIXEL_PITCH_M
        lag_x = (torch.arange(kernel_width, dtype=torch.float64, device=device) - (source_width - 1)
                 + offset[1]) * PIXEL_PITCH_M
        alpha_y = self.node_y[alpha_start:alpha_stop].to(device=device) * (PIXEL_PITCH_M / 2.0)
        alpha_x = self.node_x[alpha_start:alpha_stop].to(device=device) * (PIXEL_PITCH_M / 2.0)
        if beta_offset is None:
            beta_y = self.node_y[beta].to(device=device) * (PIXEL_PITCH_M / 2.0)
            beta_x = self.node_x[beta].to(device=device) * (PIXEL_PITCH_M / 2.0)
        else:
            beta_y = beta_offset[0] * (PIXEL_PITCH_M / 2.0)
            beta_x = beta_offset[1] * (PIXEL_PITCH_M / 2.0)
        dy = lag_y[None, :, None] + beta_y - alpha_y[:, None, None]
        dx = lag_x[None, None, :] + beta_x - alpha_x[:, None, None]
        radius = torch.sqrt(self.config.distance_m**2 + dy.square() + dx.square())
        amplitude = (self.config.distance_m / radius.square()) * (
            1.0 / (2.0 * math.pi * radius) - 1j / WAVELENGTH_M
        )
        phase = (2.0 * math.pi / WAVELENGTH_M) * radius
        kernel = amplitude * torch.polar(torch.ones_like(phase), phase)
        padded = torch.zeros((alpha_stop - alpha_start, *fft_shape), dtype=dtype, device=device)
        padded[:, :kernel_height, :kernel_width] = kernel.to(dtype)
        spectrum = torch.fft.fft2(padded).detach()

        size_bytes = spectrum.numel() * spectrum.element_size()
        if size_bytes <= self.CACHE_LIMIT_BYTES:
            while self._kernel_cache and self._kernel_cache_bytes + size_bytes > self.CACHE_LIMIT_BYTES:
                _, prior = self._kernel_cache.popitem(last=False)
                self._kernel_cache_bytes -= prior.numel() * prior.element_size()
            self._kernel_cache[key] = spectrum
            self._kernel_cache_bytes += size_bytes
        return spectrum

    def _propagate_observation(
        self, source_spectrum: torch.Tensor, output_shape: tuple[int, int], *,
        capture_nodes: bool, capture_center: bool, full_window_power: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        source_height, source_width = self.active_shape
        batch_size = source_spectrum.shape[0]
        fft_shape = tuple(source_spectrum.shape[-2:])
        if full_window_power:
            work_shape = (self.config.window_cells, self.config.window_cells)
        else:
            work_shape = output_shape
        work_offset = ((source_height - work_shape[0]) // 2, (source_width - work_shape[1]) // 2)
        output_crop = ((work_shape[0] - output_shape[0]) // 2,
                       (work_shape[1] - output_shape[1]) // 2)
        node_fields = torch.empty(
            (batch_size, self.order**2, *output_shape), dtype=source_spectrum.dtype,
            device=source_spectrum.device,
        )
        integrated_power = torch.zeros(batch_size, dtype=torch.float64, device=source_spectrum.device)
        alpha_chunk = 1 if work_shape[0] > 1024 else min(16, self.order**2)

        def propagate_node(beta: int, *, integrate_power: bool) -> torch.Tensor:
            spectrum_sum = torch.zeros((batch_size, *fft_shape), dtype=source_spectrum.dtype,
                                       device=source_spectrum.device)
            for alpha_start in range(0, self.order**2, alpha_chunk):
                alpha_stop = min(alpha_start + alpha_chunk, self.order**2)
                kernels = self._kernel_spectrum(
                    work_shape, beta, alpha_start, alpha_stop, work_offset, fft_shape,
                    source_spectrum.device, dtype=source_spectrum.dtype,
                )
                spectrum_sum = spectrum_sum + (
                    source_spectrum[:, alpha_start:alpha_stop] * kernels.unsqueeze(0)
                ).sum(dim=1)
            propagated = torch.fft.ifft2(spectrum_sum)
            if integrate_power:
                area = PIXEL_PITCH_M**2
                weight = self.node_weights[beta].to(device=propagated.device, dtype=torch.float64)
                integrated_power.add_(
                    propagated.abs().square().sum(dim=(-2, -1), dtype=torch.float64) * area * weight
                )
            y0, x0 = output_crop
            result = propagated[..., y0:y0 + output_shape[0], x0:x0 + output_shape[1]]
            return result

        for beta in range(self.order**2):
            node_fields[:, beta] = propagate_node(beta, integrate_power=full_window_power)
        center_field = None
        if capture_center:
            # The center observer has zero subpixel offset; evaluate it directly,
            # independent of the Gauss nodes used for area integration.
            center_field = self._propagate_center(
                source_spectrum, work_shape, output_shape, work_offset, output_crop,
                fft_shape, alpha_chunk,
            )
        if not full_window_power:
            weights = self.node_weights.to(device=node_fields.device, dtype=torch.float64)
            integrated_power = (
                node_fields.abs().square().to(torch.float64)
                * weights[None, :, None, None]
            ).sum(dim=(1, 2, 3)) * PIXEL_PITCH_M**2
        return node_fields, integrated_power, center_field

    def _propagate_center(
        self, source_spectrum: torch.Tensor, work_shape: tuple[int, int],
        output_shape: tuple[int, int], work_offset: tuple[int, int],
        output_crop: tuple[int, int], fft_shape: tuple[int, int], alpha_chunk: int,
    ) -> torch.Tensor:
        batch_size = source_spectrum.shape[0]
        spectrum_sum = torch.zeros((batch_size, *fft_shape), dtype=source_spectrum.dtype,
                                   device=source_spectrum.device)
        for alpha_start in range(0, self.order**2, alpha_chunk):
            alpha_stop = min(alpha_start + alpha_chunk, self.order**2)
            kernels = self._kernel_spectrum(
                work_shape, 0, alpha_start, alpha_stop, work_offset,
                fft_shape, source_spectrum.device, dtype=source_spectrum.dtype,
                beta_offset=(0.0, 0.0), cache_tag="center",
            )
            spectrum_sum = spectrum_sum + (
                source_spectrum[:, alpha_start:alpha_stop] * kernels.unsqueeze(0)
            ).sum(dim=1)
        propagated = torch.fft.ifft2(spectrum_sum)
        y0, x0 = output_crop
        return propagated[..., y0:y0 + output_shape[0], x0:x0 + output_shape[1]]

    def forward(
        self, values: torch.Tensor, *, output_shape: tuple[int, int] | None = None,
        capture_center: bool = False, full_window_power: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if values.ndim != 4 or values.shape[1] != self.order**2:
            raise ValueError("finite-pixel RS field must be [B, order**2, H, W]")
        if tuple(values.shape[-2:]) != self.active_shape or not torch.is_complex(values):
            raise ValueError("finite-pixel RS field has the wrong active shape or dtype")
        if not torch.isfinite(values).all():
            raise ValueError("finite-pixel RS source contains nonfinite values")
        output_shape = output_shape or self.active_shape
        if any(size > self.config.window_cells for size in output_shape):
            raise ValueError("finite-pixel RS observation exceeds the configured physical window")
        work_shape = ((self.config.window_cells, self.config.window_cells)
                      if full_window_power else output_shape)
        fft_shape = self._fft_shape(self.active_shape, work_shape)
        source_height, source_width = self.active_shape
        source_area = self.node_weights.to(device=values.device, dtype=values.real.dtype) \
            * (PIXEL_PITCH_M**2)
        source = torch.zeros(
            (values.shape[0], self.order**2, *fft_shape), dtype=values.dtype, device=values.device
        )
        source[..., :source_height, :source_width] = values * source_area[None, :, None, None]
        source_spectrum = torch.fft.fft2(source)
        del source
        node_fields, power, center = self._propagate_observation(
            source_spectrum, output_shape, capture_nodes=True, capture_center=capture_center,
            full_window_power=full_window_power,
        )
        return node_fields, power, center


class FinitePixelRSLinear(nn.Module):
    """Production adapter for the directly validated RS Toeplitz operator."""

    def __init__(self, order: int, active_shape: tuple[int, int], config: FinitePixelConfig) -> None:
        super().__init__()
        if any(size > config.window_cells for size in active_shape):
            raise ValueError("finite-pixel RS active support exceeds the physical window")
        from .chen_lightgen_400_rs_convolution import (
            FinitePixelRayleighSommerfeld,
            FinitePixelRayleighSommerfeldCached,
        )

        self.order = order
        self.active_shape = active_shape
        self.config = config
        _, weights = _legendre_rule(order)
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        self.register_buffer("node_weights", (wy * wx).reshape(-1), persistent=True)
        if config.propagation_backend == "rs_linear_cached":
            self.operator = FinitePixelRayleighSommerfeldCached(
                order, distance_m=config.distance_m, pair_block=config.rs_pair_block,
                compute_precision=config.rs_compute_precision
            )
        else:
            self.operator = FinitePixelRayleighSommerfeld(order, distance_m=config.distance_m)

    @staticmethod
    def _fft_shape(source_shape: tuple[int, int], output_shape: tuple[int, int]) -> tuple[int, int]:
        return tuple(_next_fast_fft_size(a + b - 1)
                     for a, b in zip(source_shape, output_shape, strict=True))

    def forward(
        self, values: torch.Tensor, *, output_shape: tuple[int, int] | None = None,
        capture_center: bool = False, full_window_power: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if values.ndim != 4 or values.shape[1] != self.order**2 \
                or tuple(values.shape[-2:]) != self.active_shape or not torch.is_complex(values):
            raise ValueError("finite-pixel RS field must be [B, order**2, H, W] on its active support")
        if not torch.isfinite(values).all():
            raise ValueError("finite-pixel RS source contains nonfinite values")
        output_shape = output_shape or self.active_shape
        if len(output_shape) != 2 or any(size < 1 or size > self.config.window_cells for size in output_shape):
            raise ValueError("finite-pixel RS observation has an invalid physical shape")
        if full_window_power and torch.is_grad_enabled():
            raise ValueError("full-window RS power is an evaluation measurement and requires no_grad")

        fft_shape = self._fft_shape(self.active_shape, output_shape)
        if self.config.fft_embedding_extra_cells:
            extra = self.config.fft_embedding_extra_cells
            fft_shape = (fft_shape[0] + extra, fft_shape[1] + extra)
        node_fields = self.operator(values, output_shape=output_shape, fft_shape=fft_shape)
        node_weights = self.node_weights.to(device=values.device, dtype=values.real.dtype)
        if full_window_power:
            full_shape = (self.config.window_cells, self.config.window_cells)
            power = torch.zeros(values.shape[0], dtype=torch.float64, device=values.device)
            tile_size = self.config.window_power_tile_cells
            target_chunk = self.config.window_power_target_chunk
            source_chunk = self.config.window_power_source_chunk
            for y0 in range(0, full_shape[0], tile_size):
                tile_height = min(tile_size, full_shape[0] - y0)
                offset_y = y0 + (tile_height - full_shape[0]) / 2.0
                for x0 in range(0, full_shape[1], tile_size):
                    tile_width = min(tile_size, full_shape[1] - x0)
                    offset_x = x0 + (tile_width - full_shape[1]) / 2.0
                    for target0 in range(0, self.order**2, target_chunk):
                        target_ids = tuple(range(target0, min(target0 + target_chunk, self.order**2)))
                        tile_fft_shape = self._fft_shape(self.active_shape, (tile_height, tile_width))
                        if self.config.fft_embedding_extra_cells:
                            extra = self.config.fft_embedding_extra_cells
                            tile_fft_shape = (tile_fft_shape[0] + extra, tile_fft_shape[1] + extra)
                        tile_nodes = self.operator(
                            values, output_shape=(tile_height, tile_width), target_indices=target_ids,
                            fft_shape=tile_fft_shape, output_offset_cells=(offset_y, offset_x),
                            source_chunk_size=source_chunk,
                        )
                        weights = node_weights[list(target_ids)].to(torch.float64)
                        power += (
                            tile_nodes.abs().square().to(torch.float64)
                            * weights[None, :, None, None]
                        ).sum(dim=(1, 2, 3)) * PIXEL_PITCH_M**2
                        del tile_nodes
        else:
            power = (node_fields.abs().square().to(torch.float64)
                     * node_weights.to(torch.float64)[None, :, None, None]).sum(dim=(1, 2, 3)) \
                * PIXEL_PITCH_M**2
        center = None
        if capture_center:
            center = self.operator(values, output_shape=output_shape, centered=True,
                                   fft_shape=fft_shape)[:, 0]
        return node_fields, power, center

    def observe_nodes_at_order(
        self, values: torch.Tensor, target_order: int, *, output_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Observe a source-node field at another Gauss rule on the same cells."""
        if target_order not in (2, 4, 8, 12, 16):
            raise ValueError("diagnostic observer order must be 2, 4, 8, 12, or 16")
        if values.ndim != 4 or values.shape[1] != self.order**2 \
                or tuple(values.shape[-2:]) != self.active_shape or not torch.is_complex(values):
            raise ValueError("mixed-order RS source field does not match its source quadrature")
        fft_shape = self._fft_shape(self.active_shape, output_shape)
        if self.config.fft_embedding_extra_cells:
            extra = self.config.fft_embedding_extra_cells
            fft_shape = (fft_shape[0] + extra, fft_shape[1] + extra)
        return self.operator(values, output_shape=output_shape, fft_shape=fft_shape,
                             target_order=target_order)


class FinitePixelAngularSpectrum(nn.Module):
    """Integrate a piecewise-smooth source over physical pixels, then propagate.

    Inputs and outputs have shape ``[batch, order**2, H, W]``. The source
    quadrature is evaluated as weighted off-grid point sources in the Fourier
    sum. Subpixel translations are applied as exact spectral phase ramps. The
    active source aperture and observation window are embedded in a larger
    zero-padded FFT domain whose extent covers their linear-convolution support.
    """

    def __init__(self, order: int, active_shape: tuple[int, int], config: FinitePixelConfig) -> None:
        super().__init__()
        self.order = order
        self.active_shape = active_shape
        self.config = config
        if order not in (2, 4, 8, 12):
            raise ValueError("unsupported finite-pixel quadrature order")
        if any(size > config.window_cells for size in active_shape):
            raise ValueError("active support exceeds propagation window")
        nodes, weights = _legendre_rule(order)
        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        self.register_buffer("nodes_y", yy.reshape(-1), persistent=True)
        self.register_buffer("nodes_x", xx.reshape(-1), persistent=True)
        self.register_buffer("node_weights", (wy * wx).reshape(-1), persistent=True)
        node_values = nodes.tolist()
        self._node_offsets = tuple(
            (float(node_y * PIXEL_PITCH_M / 2.0), float(node_x * PIXEL_PITCH_M / 2.0))
            for node_y in node_values for node_x in node_values
        )
        factor = config.subgrid_factor
        self.fft_size = config.fft_size_for(active_shape)
        # The centered fine grid puts its index factor//2 exactly at the
        # centre of every physical pixel, so Gauss offsets need no correction.
        fft_size = self.fft_size
        self.register_buffer("fy", torch.fft.fftfreq(fft_size, d=config.sample_pitch_m,
                                                       dtype=torch.float64), persistent=True)
        self.register_buffer("fx", torch.fft.fftfreq(fft_size, d=config.sample_pitch_m,
                                                       dtype=torch.float64), persistent=True)
        window_start_y = (fft_size - config.window_cells * factor) // 2
        window_start_x = (fft_size - config.window_cells * factor) // 2
        start_y = window_start_y + ((config.window_cells - active_shape[0]) // 2) * factor
        start_x = window_start_x + ((config.window_cells - active_shape[1]) // 2) * factor
        self.register_buffer(
            "active_y_indices",
            start_y + torch.arange(active_shape[0], dtype=torch.long) * factor + factor // 2,
            persistent=True,
        )
        self.register_buffer(
            "active_x_indices",
            start_x + torch.arange(active_shape[1], dtype=torch.long) * factor + factor // 2,
            persistent=True,
        )
        # q**2 * w_i is the area weight in units of the fine-grid pixel area.
        self.register_buffer("spectral_node_weights", self.node_weights * (factor**2), persistent=True)

    def _phase_ramp(self, spectrum: torch.Tensor, dy: float, dx: float, *, sign: float) -> None:
        n = self.fft_size
        rows_per_chunk = max(1, min(128, 8_388_608 // n))
        fy = self.fy.to(device=spectrum.device, dtype=torch.float64)
        fx = self.fx.to(device=spectrum.device, dtype=torch.float64)
        for start in range(0, n, rows_per_chunk):
            stop = min(start + rows_per_chunk, n)
            phase = (sign * 2.0 * math.pi) * (fy[start:stop, None] * dy + fx[None, :] * dx)
            ramp = torch.polar(torch.ones_like(phase), phase).to(dtype=spectrum.dtype)
            spectrum[:, start:stop].mul_(ramp)

    def _transfer_and_power(self, spectrum: torch.Tensor) -> torch.Tensor:
        n = self.fft_size
        rows_per_chunk = max(1, min(128, 8_388_608 // n))
        fy = self.fy.to(device=spectrum.device, dtype=torch.float64)
        fx = self.fx.to(device=spectrum.device, dtype=torch.float64)
        phase_scale = 2.0 * math.pi * self.config.distance_m / WAVELENGTH_M
        for start in range(0, n, rows_per_chunk):
            stop = min(start + rows_per_chunk, n)
            argument = 1.0 - (WAVELENGTH_M * fy[start:stop, None]).square() \
                - (WAVELENGTH_M * fx[None, :]).square()
            transfer_phase = phase_scale * torch.sqrt(argument.clamp_min(0.0))
            transfer = torch.polar(torch.ones_like(transfer_phase), transfer_phase).to(spectrum.dtype)
            transfer *= (argument >= 0.0)
            spectrum[:, start:stop].mul_(transfer)
        dx = self.config.sample_pitch_m
        power = spectrum.abs().square().sum(dim=(-2, -1)) * (dx * dx / (n * n))
        return power

    def forward(
        self,
        values: torch.Tensor,
        *,
        output_shape: tuple[int, int] | None = None,
        capture_center: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if values.ndim != 4 or values.shape[1] != self.order**2:
            raise ValueError("finite-pixel field must be [B, order**2, H, W]")
        if tuple(values.shape[-2:]) != self.active_shape or not torch.is_complex(values):
            raise ValueError("finite-pixel field has the wrong active shape or dtype")
        if not torch.isfinite(values).all():
            raise ValueError("finite-pixel source contains nonfinite values")
        target_shape = output_shape or self.active_shape
        if any(size > self.config.window_cells for size in target_shape):
            raise ValueError("requested observation exceeds physical propagation window")
        device, dtype = values.device, values.dtype
        n, factor = self.fft_size, self.config.subgrid_factor
        active_y = self.active_y_indices.to(device)
        active_x = self.active_x_indices.to(device)
        batch_size = values.shape[0]
        spectrum_sum = torch.zeros((batch_size, n, n), device=device, dtype=dtype)
        for index in range(self.order**2):
            sparse = torch.zeros_like(spectrum_sum)
            sparse[:, active_y[:, None], active_x[None, :]] = (
                values[:, index] * self.spectral_node_weights[index].to(device=device, dtype=values.real.dtype)
            )
            spectrum = torch.fft.fft2(sparse)
            dy, dx = self._node_offsets[index]
            self._phase_ramp(spectrum, dy, dx, sign=-1.0)
            spectrum_sum = spectrum_sum + spectrum
            del sparse, spectrum
        full_window_power = self._transfer_and_power(spectrum_sum)

        output_height, output_width = target_shape
        out_start_y = (n - self.config.window_cells * factor) // 2
        out_start_x = (n - self.config.window_cells * factor) // 2
        out_start_y += ((self.config.window_cells - output_height) // 2) * factor
        out_start_x += ((self.config.window_cells - output_width) // 2) * factor
        target_y = out_start_y + torch.arange(output_height, device=device) * factor + factor // 2
        target_x = out_start_x + torch.arange(output_width, device=device) * factor + factor // 2
        node_fields = torch.empty(
            (batch_size, self.order**2, output_height, output_width), device=device, dtype=dtype
        )
        for index in range(self.order**2):
            spectrum = spectrum_sum.clone()
            dy, dx = self._node_offsets[index]
            self._phase_ramp(spectrum, dy, dx, sign=1.0)
            propagated = torch.fft.ifft2(spectrum)
            node_fields[:, index] = propagated.index_select(-2, target_y).index_select(-1, target_x)
            del spectrum, propagated
        center_field = None
        if capture_center:
            center_spectrum = spectrum_sum.clone()
            self._phase_ramp(center_spectrum, 0.0, 0.0, sign=1.0)
            center_full = torch.fft.ifft2(center_spectrum)
            center_field = center_full.index_select(-2, target_y).index_select(-1, target_x)
            del center_spectrum, center_full
        del spectrum_sum
        return node_fields, full_window_power, center_field


class FinitePixelChenLightGen400(nn.Module):
    """2+4 phase masks using a finite-pixel quadrature propagation operator."""

    def __init__(self, config: FinitePixelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FinitePixelConfig()
        self.order = self.config.quadrature_order
        nodes, weights = _legendre_rule(self.order)
        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        self.register_buffer("node_y", yy.reshape(-1), persistent=True)
        self.register_buffer("node_x", xx.reshape(-1), persistent=True)
        self.register_buffer("node_weight_2d", (wy * wx).reshape(-1), persistent=True)
        self.encoder_phase = nn.Parameter(torch.zeros(2, 1, *ACTIVE_SHAPE))
        self.decoder_phase = nn.Parameter(torch.zeros(4, 1, *ACTIVE_SHAPE))
        if self.config.propagation_backend in ("rs_linear", "rs_linear_cached"):
            self.propagator = FinitePixelRSLinear(self.order, ACTIVE_SHAPE, self.config)
        else:
            self.propagator = FinitePixelAngularSpectrum(self.order, ACTIVE_SHAPE, self.config)
        self._make_modes()

    def _make_modes(self) -> None:
        q = self.order
        pixel_centres = (torch.arange(ACTIVE_SHAPE[0], dtype=torch.float64) + 0.5) * PIXEL_PITCH_M \
            - (ACTIVE_SHAPE[0] * PIXEL_PITCH_M / 2.0)
        node_offsets = torch.as_tensor(np.polynomial.legendre.leggauss(q)[0], dtype=torch.float64) \
            * (PIXEL_PITCH_M / 2.0)
        y = pixel_centres[:, None] + node_offsets[None, :]
        x = pixel_centres[:, None] + node_offsets[None, :]
        centers = (torch.arange(10, dtype=torch.float64) - 4.5) * FIBER_PITCH_M
        amplitude = (2.0 / (math.pi * (FIBER_MFD_M / 2.0) ** 2)) ** 0.25
        modes_y = amplitude * torch.exp(-((y[None, :, :] - centers[:, None, None]) /
                                            (FIBER_MFD_M / 2.0)).square())
        modes_x = amplitude * torch.exp(-((x[None, :, :] - centers[:, None, None]) /
                                            (FIBER_MFD_M / 2.0)).square())
        one_d_weights = torch.as_tensor(np.polynomial.legendre.leggauss(q)[1], dtype=torch.float64) / 2.0
        physical_weights = one_d_weights * PIXEL_PITCH_M
        gy = torch.einsum("ryq,syq,q->rs", modes_y, modes_y, physical_weights)
        gx = torch.einsum("cxq,dxq,q->cd", modes_x, modes_x, physical_weights)
        gram = torch.kron(gy, gx)
        max_norm_error = float((gram.diag() - 1.0).abs().max())
        largest = float(torch.linalg.eigvalsh(gram)[-1])
        power_gain_bound = largest
        if max_norm_error > 2.0e-6 or power_gain_bound > 1.0 + 1.0e-8:
            raise ValueError("finite-pixel Gaussian OLS quadrature failed norm/passivity")
        self.register_buffer("modes_y", modes_y, persistent=True)
        self.register_buffer("modes_x", modes_x, persistent=True)
        self.register_buffer("gram", gram, persistent=True)
        self.register_buffer("mode_diagnostics", torch.tensor(
            [max_norm_error, largest, power_gain_bound], dtype=torch.float64), persistent=True
        )

    @property
    def pixel_area_m2(self) -> float:
        return PIXEL_PITCH_M**2

    def _weighted_power(self, values: torch.Tensor) -> torch.Tensor:
        weights = self.node_weight_2d.to(device=values.device, dtype=values.real.dtype)
        return (values.abs().square() * weights[None, :, None, None]).sum(dim=(1, 2, 3)) * self.pixel_area_m2

    def _project_coefficients(self, field: torch.Tensor) -> torch.Tensor:
        weights = self.node_weight_2d.to(device=field.device, dtype=field.real.dtype)
        modes_y = self.modes_y.to(device=field.device, dtype=field.dtype)
        modes_x = self.modes_x.to(device=field.device, dtype=field.dtype)
        weighted = field.reshape(field.shape[0], self.order, self.order, *ACTIVE_SHAPE) \
            * weights.reshape(1, self.order, self.order, 1, 1)
        intermediate = torch.einsum("ryi,bijyx->brjx", modes_y, weighted) * PIXEL_PITCH_M
        coeff = torch.einsum("brjx,cxj->brc", intermediate, modes_x) * PIXEL_PITCH_M
        return coeff.reshape(field.shape[0], 100)

    def _reemit(self, coeff: torch.Tensor) -> torch.Tensor:
        modes_y = self.modes_y.to(device=coeff.device, dtype=coeff.dtype)
        modes_x = self.modes_x.to(device=coeff.device, dtype=coeff.dtype)
        coeff_grid = coeff.reshape(coeff.shape[0], 10, 10)
        node_field = torch.einsum("brc,ryi,cxj->bijyx", coeff_grid, modes_y, modes_x)
        return node_field.reshape(coeff.shape[0], self.order**2, *ACTIVE_SHAPE)

    def _phase(self, field: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        transmission = torch.polar(
            torch.full_like(phase, math.sqrt(LAYER_POWER_TRANSMISSION)), phase
        )
        return field * transmission[:, 0].unsqueeze(1)

    def _propagate_plane(
        self,
        field: torch.Tensor,
        *,
        capture: bool,
        full_window_power: bool,
        diagnostic_observer_order: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if not capture:
            active, power, _ = self._propagate(field, full_window_power=full_window_power)
            return active, power, None
        roi_shape = (500, 500)
        roi_nodes, power, center = self._propagate(
            field, output_shape=roi_shape, capture_center=True,
            full_window_power=full_window_power,
        )
        weights = self.node_weight_2d.to(device=field.device, dtype=field.real.dtype)
        average = (roi_nodes * weights[None, :, None, None]).sum(dim=1, keepdim=True)
        margin_y = (roi_shape[0] - ACTIVE_SHAPE[0]) // 2
        margin_x = (roi_shape[1] - ACTIVE_SHAPE[1]) // 2
        active = roi_nodes[..., margin_y:margin_y + ACTIVE_SHAPE[0],
                           margin_x:margin_x + ACTIVE_SHAPE[1]]
        diagnostic_nodes = None
        if diagnostic_observer_order is not None:
            if not isinstance(self.propagator, FinitePixelRSLinear):
                raise ValueError("mixed-order node observations require the RS linear backend")
            diagnostic_nodes = (roi_nodes if diagnostic_observer_order == self.order else
                                self.propagator.observe_nodes_at_order(
                                    field, diagnostic_observer_order, output_shape=roi_shape
                                ))
            diagnostic_nodes = diagnostic_nodes[..., margin_y:margin_y + ACTIVE_SHAPE[0],
                                                margin_x:margin_x + ACTIVE_SHAPE[1]]
        intensity_average = (roi_nodes.abs().square()
                             * weights[None, :, None, None]).sum(dim=1, keepdim=True)
        assert center is not None
        observation = {
            "pixel_center": center,
            "pixel_area_average": average,
            "pixel_intensity_average": intensity_average,
        }
        if diagnostic_nodes is not None:
            observation["diagnostic_node_field"] = diagnostic_nodes
        return active, power.detach(), observation

    def _propagate(
        self, field: torch.Tensor, *, capture_center: bool = False,
        output_shape: tuple[int, int] | None = None, full_window_power: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if isinstance(self.propagator, FinitePixelRSLinear):
            if self.config.checkpoint_propagation and torch.is_grad_enabled() and not capture_center \
                    and not full_window_power:
                node_fields, power = checkpoint(
                    lambda value: self.propagator(value, output_shape=output_shape)[:2],
                    field, use_reentrant=False,
                )
                return node_fields, power.detach(), None
            return self.propagator(
                field, output_shape=output_shape, capture_center=capture_center,
                full_window_power=full_window_power,
            )
        if self.config.checkpoint_propagation and torch.is_grad_enabled() and not capture_center:
            node_fields, full_window_power = checkpoint(
                lambda value: self.propagator(value, output_shape=output_shape)[:2], field,
                use_reentrant=False,
            )
            return node_fields, full_window_power.detach(), None
        return self.propagator(field, output_shape=output_shape, capture_center=capture_center)

    def _initial_node_intensities(self, intensity: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = intensity.shape
        # MNIST is bilinearly resized once before this API receives it.  Each
        # resulting value then describes a piecewise-constant physical SLM
        # pixel; Gauss nodes integrate the propagated kernel inside that pixel,
        # not an interpolated image edge.
        return intensity.expand(batch, self.order**2, height, width)

    def forward_from_image(
        self, intensity: torch.Tensor, *, capture_fields: bool = False,
        measure_full_window_power: bool = False,
        diagnostic_observer_order: int | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | str]:
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(1)
        if tuple(intensity.shape[1:]) != (1, *ACTIVE_SHAPE):
            raise ValueError("LightGen400 input must be [B,1,400,400] intensity")
        if not torch.isfinite(intensity).all() or (intensity < 0).any() or (intensity > 1).any():
            raise ValueError("LightGen400 input intensity must be finite in [0,1]")
        if diagnostic_observer_order is not None and not capture_fields:
            raise ValueError("mixed-order node observations require capture_fields=True")
        node_intensity = self._initial_node_intensities(intensity) \
            if isinstance(self.propagator, FinitePixelRSLinear) else intensity[:, 0].unsqueeze(1).expand(
                -1, self.order**2, -1, -1
            )
        field_dtype = torch.complex64
        if (isinstance(self.propagator, FinitePixelRSLinear)
                and self.config.propagation_backend == "rs_linear_cached"
                and self.config.rs_compute_precision == "complex128"):
            # Preserve the reference precision between propagation planes. The
            # RS operator keeps its input dtype, so a complex64 source would
            # otherwise round every complex128 hop back to complex64.
            field_dtype = torch.complex128
        field = torch.sqrt(node_intensity).to(field_dtype)
        density_scale = math.sqrt(SOURCE_POWER_W / math.prod((ACTIVE_SHAPE[0] * PIXEL_PITCH_M,
                                                               ACTIVE_SHAPE[1] * PIXEL_PITCH_M)))
        field = field * density_scale
        input_power = self._weighted_power(field)
        powers: list[torch.Tensor] = []
        planes: list[dict[str, torch.Tensor]] = []
        diagnostic_node_fields: list[torch.Tensor] = []
        for layer in range(2):
            field = self._phase(field, self.encoder_phase[layer])
            field, full_power, observation = self._propagate_plane(
                field, capture=capture_fields, full_window_power=measure_full_window_power,
                diagnostic_observer_order=diagnostic_observer_order,
            )
            powers.append(full_power)
            if observation is not None:
                if diagnostic_observer_order is not None:
                    diagnostic_node_fields.append(observation.pop("diagnostic_node_field"))
                planes.append(observation)
        coefficients = self._project_coefficients(field)
        ols_input_power = powers[-1]
        ols_output = self._reemit(coefficients)
        ols_output_power = self._weighted_power(ols_output)
        field, generator_full_power, observation = self._propagate_plane(
            ols_output, capture=capture_fields, full_window_power=measure_full_window_power,
            diagnostic_observer_order=diagnostic_observer_order,
        )
        powers.append(generator_full_power)
        if observation is not None:
            if diagnostic_observer_order is not None:
                diagnostic_node_fields.append(observation.pop("diagnostic_node_field"))
            planes.append(observation)
        for layer in range(4):
            field = self._phase(field, self.decoder_phase[layer])
            field, full_power, observation = self._propagate_plane(
                field, capture=capture_fields, full_window_power=measure_full_window_power,
                diagnostic_observer_order=diagnostic_observer_order,
            )
            powers.append(full_power)
            if observation is not None:
                if diagnostic_observer_order is not None:
                    diagnostic_node_fields.append(observation.pop("diagnostic_node_field"))
                planes.append(observation)
        weights = self.node_weight_2d.to(device=field.device, dtype=field.real.dtype)
        camera_intensity = (field.abs().square() * weights[None, :, None, None]).sum(dim=1, keepdim=True)
        physical_w_per_pixel = camera_intensity * self.pixel_area_m2
        if capture_fields:
            camera_roi_intensity = planes[-1]["pixel_intensity_average"]
            camera_roi_w_per_pixel = camera_roi_intensity * self.pixel_area_m2
            margin_y = (camera_roi_w_per_pixel.shape[-2] - ACTIVE_SHAPE[0]) // 2
            margin_x = (camera_roi_w_per_pixel.shape[-1] - ACTIVE_SHAPE[1]) // 2
            physical_w_per_pixel = camera_roi_w_per_pixel[
                ..., margin_y:margin_y + ACTIVE_SHAPE[0], margin_x:margin_x + ACTIVE_SHAPE[1]
            ]
        else:
            camera_roi_w_per_pixel = physical_w_per_pixel
        return {
            "physical_w_per_pixel": physical_w_per_pixel,
            "camera_roi_w_per_pixel": camera_roi_w_per_pixel,
            "raw_intensity_w_per_m2": camera_intensity,
            "ols_coefficients": coefficients,
            "input_power_w": input_power,
            "ols_input_power_w": ols_input_power,
            "ols_output_power_w": ols_output_power,
            "plane_powers_w": torch.stack(powers, dim=1),
            "camera_roi_power_w": camera_roi_w_per_pixel.sum(dim=(1, 2, 3)),
            "camera_observed_power_w": powers[-1],
            "plane_power_scope": ("full_propagation_window" if measure_full_window_power else
                                  "camera_observation_500x500" if capture_fields else
                                  "active_physical_support_400x400"),
            "finite_pixel_plane_values": tuple(planes),
            "diagnostic_node_fields_at_order": tuple(diagnostic_node_fields),
        }

