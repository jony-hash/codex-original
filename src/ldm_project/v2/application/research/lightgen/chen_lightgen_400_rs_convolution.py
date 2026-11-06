"""Diagnostic finite-pixel Rayleigh--Sommerfeld quadrature by linear convolution.

Each node-to-node block is a discrete Toeplitz operator on physical pixel
centres. Its circulant embedding contains every source/target displacement;
no angular-spectrum transfer or interpolated complex field is used here.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .chen_lightgen_400_baseline import ACTIVE_SHAPE, PIXEL_PITCH_M, SOURCE_POWER_W, WAVELENGTH_M
from .chen_lightgen_400_finite_pixel import (
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    _legendre_rule,
    _next_fast_fft_size,
)


class _RSLinearMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, operator, output_shape, centered, fft_shape, target_indices,
                output_offset_cells, source_chunk_size, target_order,
                target_offset_y, target_offset_x):
        ctx.operator = operator
        ctx.input_shape = tuple(values.shape)
        ctx.output_shape = output_shape
        ctx.centered = centered
        ctx.fft_shape = fft_shape
        ctx.target_indices = target_indices
        ctx.output_offset_cells = output_offset_cells
        ctx.source_chunk_size = source_chunk_size
        ctx.target_order = target_order
        ctx.target_offset_y = target_offset_y
        ctx.target_offset_x = target_offset_x
        return operator._apply_convolution(values, output_shape, centered, fft_shape,
                                            adjoint=False, target_indices=target_indices,
                                            output_offset_cells=output_offset_cells,
                                            source_chunk_size=source_chunk_size,
                                            target_order=target_order,
                                            target_offset_y=target_offset_y,
                                            target_offset_x=target_offset_x)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        result = ctx.operator._apply_convolution(
            gradient, ctx.output_shape, ctx.centered, ctx.fft_shape,
            adjoint=True, input_shape=ctx.input_shape, target_indices=ctx.target_indices,
            output_offset_cells=ctx.output_offset_cells, target_order=ctx.target_order,
            target_offset_y=ctx.target_offset_y, target_offset_x=ctx.target_offset_x,
        )
        return result, None, None, None, None, None, None, None, None, None, None


class FinitePixelRayleighSommerfeld(nn.Module):
    """Nyström RS integral and its exact discrete adjoint; fields are sqrt(W)/m.

    ``[batch, order**2, height, width]`` stores Gauss-node fields. All physical
    pixels keep the same 3 um pitch independently of quadrature order. The
    output is another node field, or one centre observation per physical cell.
    This operator reports no full-plane power from a limited observation ROI.
    """

    def __init__(self, order: int, *, distance_m: float = 3.0e-3, pair_block: int = 4):
        super().__init__()
        if order not in (2, 4, 8, 12, 16):
            raise ValueError("RS order must be 2, 4, 8, 12, or reference order 16")
        if not math.isfinite(distance_m) or distance_m <= 0 or pair_block < 1:
            raise ValueError("RS distance and node-pair block must be positive")
        self.order = order
        self.distance_m = distance_m
        self.pair_block = pair_block
        nodes, weights = _legendre_rule(order)
        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        self.register_buffer("offset_y", yy.reshape(-1) * PIXEL_PITCH_M / 2)
        self.register_buffer("offset_x", xx.reshape(-1) * PIXEL_PITCH_M / 2)
        self.register_buffer("source_area", (wy * wx).reshape(-1) * PIXEL_PITCH_M**2)

    def forward(self, values: torch.Tensor, *, output_shape: tuple[int, int] | None = None,
                centered: bool = False, fft_shape: tuple[int, int] | None = None,
                target_indices: tuple[int, ...] | None = None,
                target_order: int | None = None,
                output_offset_cells: tuple[float, float] = (0.0, 0.0),
                source_chunk_size: int | None = None) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.order**2 or not torch.is_complex(values):
            raise ValueError("RS input must be complex [batch, order**2, height, width]")
        if values.dtype not in (torch.complex64, torch.complex128) or not torch.isfinite(values).all():
            raise ValueError("RS input must be finite complex64 or complex128")
        shape = output_shape or tuple(values.shape[-2:])
        if len(shape) != 2 or any(size < 1 for size in shape):
            raise ValueError("RS output shape must have two positive dimensions")
        if len(output_offset_cells) != 2 or not all(math.isfinite(float(value)) for value in output_offset_cells):
            raise ValueError("RS observation offset must contain two finite pixel coordinates")
        target_order = self.order if target_order is None else target_order
        if target_order not in (2, 4, 8, 12, 16):
            raise ValueError("RS target quadrature order must be 2, 4, 8, 12, or 16")
        if source_chunk_size is not None and (
            not isinstance(source_chunk_size, int) or isinstance(source_chunk_size, bool) or source_chunk_size < 1
        ):
            raise ValueError("RS source chunk size must be positive")
        minimum = tuple(a + b - 1 for a, b in zip(values.shape[-2:], shape, strict=True))
        padding = fft_shape or tuple(_next_fast_fft_size(size) for size in minimum)
        if len(padding) != 2 or any(a < b for a, b in zip(padding, minimum, strict=True)):
            raise ValueError("RS embedding must contain all source/target displacements")
        if centered and target_indices is not None:
            raise ValueError("centered RS observations do not accept node target indices")
        if target_order == self.order:
            target_offset_y, target_offset_x = None, None
        else:
            target_nodes, _ = _legendre_rule(target_order, device=values.device)
            target_y_grid, target_x_grid = torch.meshgrid(target_nodes, target_nodes, indexing="ij")
            target_offset_y = target_y_grid.reshape(-1) * PIXEL_PITCH_M / 2
            target_offset_x = target_x_grid.reshape(-1) * PIXEL_PITCH_M / 2
        selected = None if centered else (tuple(range(target_order**2)) if target_indices is None
                                          else tuple(target_indices))
        if selected is not None and (not selected or len(set(selected)) != len(selected)
                                     or any(index < 0 or index >= target_order**2 for index in selected)):
            raise ValueError("RS target-node selection must be nonempty, unique, and in range")
        return _RSLinearMap.apply(values, self, tuple(shape), centered, tuple(padding), selected,
                                  tuple(float(value) for value in output_offset_cells), source_chunk_size,
                                  target_order, target_offset_y, target_offset_x)

    def _kernel_spectrum(self, input_shape, output_shape, fft_shape, targets, sources,
                         centered, device, dtype, output_offset_cells=(0.0, 0.0),
                         target_offset_y=None, target_offset_x=None,
                         include_source_area=True):
        # Signed displacement indices are the Toeplitz-to-circulant embedding.
        axes = []
        valid_axes = []
        for source_cells, target_cells, padded in zip(input_shape, output_shape, fft_shape, strict=True):
            index = torch.arange(padded, device=device)
            signed = torch.where(index < target_cells, index, index - padded)
            valid_axes.append((signed >= -(source_cells - 1)) & (signed < target_cells))
            axes.append((signed.double() + (source_cells - target_cells) / 2
                         + output_offset_cells[len(axes)]) * PIXEL_PITCH_M)
        source_y = self.offset_y[sources].to(device)
        source_x = self.offset_x[sources].to(device)
        if centered:
            target_y = torch.zeros(1, device=device, dtype=torch.float64)
            target_x = target_y
        else:
            target_offset_y = self.offset_y if target_offset_y is None else target_offset_y
            target_offset_x = self.offset_x if target_offset_x is None else target_offset_x
            target_y = target_offset_y[targets].to(device)
            target_x = target_offset_x[targets].to(device)
        dy = axes[0][None, None, :, None] + target_y[:, None, None, None] - source_y[None, :, None, None]
        dx = axes[1][None, None, None, :] + target_x[:, None, None, None] - source_x[None, :, None, None]
        radius2 = self.distance_m**2 + dy.square() + dx.square()
        radius = radius2.sqrt()
        phase = (2 * math.pi / WAVELENGTH_M) * radius
        kernel = torch.polar(torch.ones_like(phase), phase) * (self.distance_m / radius2) \
            * (1 / (2 * math.pi * radius) - 1j / WAVELENGTH_M)
        if include_source_area:
            kernel *= self.source_area[sources].to(device)[None, :, None, None]
        kernel *= (valid_axes[0][:, None] & valid_axes[1][None, :])[None, None]
        return torch.fft.fft2(kernel.to(dtype))

    def _apply_convolution(self, values, output_shape, centered, fft_shape, *, adjoint,
                           input_shape=None, target_indices=None,
                           output_offset_cells=(0.0, 0.0), source_chunk_size=None,
                           target_order=None, target_offset_y=None, target_offset_x=None):
        if adjoint:
            source_shape = input_shape[-2:]
        else:
            source_shape = tuple(values.shape[-2:])
        output_dtype = values.dtype
        compute_dtype = torch.complex128 if output_dtype == torch.complex64 else output_dtype
        source_count = self.order**2
        target_order = self.order if target_order is None else target_order
        target_ids = None if centered else torch.as_tensor(
            target_indices if target_indices is not None else tuple(range(target_order**2)),
            dtype=torch.long, device=values.device)
        target_count = 1 if centered else int(target_ids.numel())
        transformed = None if source_chunk_size is not None else torch.fft.fft2(
            values.to(compute_dtype), s=fft_shape
        )
        result_channels = source_count if adjoint else target_count
        result_shape = source_shape if adjoint else output_shape
        result = torch.empty((values.shape[0], result_channels, *result_shape),
                             device=values.device, dtype=compute_dtype)
        # Recompute kernels in backward instead of retaining q**4 FFT graphs.
        for outer in range(0, result_channels, self.pair_block):
            outer_slice = slice(outer, min(outer + self.pair_block, result_channels))
            accumulated = torch.zeros((values.shape[0], outer_slice.stop - outer, *fft_shape),
                                      device=values.device, dtype=compute_dtype)
            inner_count = target_count if adjoint else source_count
            inner_step = source_chunk_size if source_chunk_size is not None else self.pair_block
            for inner in range(0, inner_count, inner_step):
                inner_slice = slice(inner, min(inner + inner_step, inner_count))
                transformed_chunk = (transformed[:, inner_slice] if transformed is not None else
                                     torch.fft.fft2(values[:, inner_slice].to(compute_dtype), s=fft_shape))
                if adjoint:
                    targets = None if centered else target_ids[inner_slice]
                    sources = outer_slice
                else:
                    targets = None if centered else target_ids[outer_slice]
                    sources = inner_slice
                kernel = self._kernel_spectrum(source_shape, output_shape, fft_shape, targets, sources,
                                               centered, values.device, compute_dtype, output_offset_cells,
                                               target_offset_y, target_offset_x)
                if adjoint:
                    accumulated += torch.einsum("btyx,tsyx->bsyx", transformed_chunk, kernel.conj())
                else:
                    accumulated += torch.einsum("bsyx,tsyx->btyx", transformed_chunk, kernel)
            output = torch.fft.ifft2(accumulated)
            result[:, outer_slice] = output[..., :result_shape[0], :result_shape[1]]
        return result.to(output_dtype)


class FinitePixelRayleighSommerfeldCached(FinitePixelRayleighSommerfeld):
    """RS operator with exact q-node reflection and transpose kernel reuse.

    The cache stores unweighted spectra for the canonical pairs of absolute
    one-dimensional Gauss-node differences. Source quadrature weights are
    applied to the input in the forward map and to the output of its adjoint.
    Unsupported observation geometries use the validated base implementation.
    """

    def __init__(self, order: int, *, distance_m: float = 3.0e-3, pair_block: int = 4,
                 compute_precision: str = "complex128", cache_build_batch: int = 4):
        if compute_precision not in {"complex128", "complex64"}:
            raise ValueError("cached RS compute precision must be complex128 or complex64")
        if not isinstance(cache_build_batch, int) or isinstance(cache_build_batch, bool) or cache_build_batch < 1:
            raise ValueError("cached RS kernel build batch must be positive")
        if order not in (2, 4, 8, 12):
            raise ValueError("cached RS propagation supports candidate orders 2, 4, 8, and 12")
        super().__init__(order, distance_m=distance_m, pair_block=pair_block)
        self.compute_precision = compute_precision
        self.compute_dtype = torch.complex128 if compute_precision == "complex128" else torch.complex64
        self.cache_build_batch = cache_build_batch
        nodes, _ = _legendre_rule(order)
        offsets = nodes * (PIXEL_PITCH_M / 2.0)
        differences = offsets[:, None] - offsets[None, :]
        unique_magnitudes, difference_index = torch.unique(
            differences.abs(), sorted=True, return_inverse=True
        )
        difference_sign = torch.sign(differences).to(torch.int8)
        unique_count = int(unique_magnitudes.numel())
        canonical_pairs = [(left, right) for left in range(unique_count)
                           for right in range(left, unique_count)]
        pair_lookup = torch.full((unique_count, unique_count), -1, dtype=torch.long)
        for cache_index, (left, right) in enumerate(canonical_pairs):
            pair_lookup[left, right] = cache_index
            pair_lookup[right, left] = cache_index

        node_y = torch.arange(order).repeat_interleave(order)
        node_x = torch.arange(order).repeat(order)
        target_y, source_y = node_y[:, None], node_y[None, :]
        target_x, source_x = node_x[:, None], node_x[None, :]
        y_index = difference_index[target_y, source_y]
        x_index = difference_index[target_x, source_x]
        transposed = y_index > x_index
        self._pair_transposed_host = tuple(
            tuple(bool(flag) for flag in row) for row in transposed.tolist()
        )
        self.register_buffer("_pair_cache_index", pair_lookup[
            torch.minimum(y_index, x_index), torch.maximum(y_index, x_index)
        ], persistent=False)
        y_sign = difference_sign[target_y, source_y] < 0
        x_sign = difference_sign[target_x, source_x] < 0
        self.register_buffer("_pair_transposed", transposed, persistent=False)
        # The transpose maps the canonical magnitudes onto the requested y/x
        # axes; reflection signs therefore remain in requested-axis order.
        self.register_buffer("_pair_flip_y", y_sign, persistent=False)
        self.register_buffer("_pair_flip_x", x_sign, persistent=False)
        self.register_buffer("_canonical_offsets", unique_magnitudes, persistent=False)
        self._canonical_pairs = tuple(canonical_pairs)
        self._kernel_cache: torch.Tensor | None = None
        self._kernel_cache_key: tuple[object, ...] | None = None
        self._kernel_cache_seconds = 0.0
        self._frequency_metadata: tuple[torch.Tensor, ...] | None = None
        self._transpose_schedule_cache: dict[
            tuple[tuple[int, ...], bool, int, int, int], tuple[bool, ...]
        ] = {}
        self._transpose_schedule_build_seconds = 0.0
        self._transpose_schedule_build_count = 0
        self._transpose_schedule_cache_hits = 0

    @property
    def cache_info(self) -> dict[str, object]:
        return {
            "order": self.order,
            "distinct_absolute_node_differences": int(self._canonical_offsets.numel()),
            "canonical_kernel_count": len(self._canonical_pairs),
            "compute_precision": self.compute_precision,
            "cached_geometry": self._kernel_cache_key,
            "cache_bytes": 0 if self._kernel_cache is None else self._kernel_cache.numel()
            * self._kernel_cache.element_size(),
            "cache_build_seconds": self._kernel_cache_seconds,
        }

    def _cache_key(self, source_shape, output_shape, fft_shape, device):
        return (self.order, source_shape, output_shape, fft_shape, self.distance_m,
                PIXEL_PITCH_M, WAVELENGTH_M, str(device), self.compute_precision)

    @torch.no_grad()
    def _build_kernel_cache(self, source_shape, output_shape, fft_shape, device) -> torch.Tensor:
        started = time.perf_counter()
        height, width = fft_shape
        signed_y = torch.arange(height, device=device)
        signed_x = torch.arange(width, device=device)
        signed_y = torch.where(signed_y < output_shape[0], signed_y, signed_y - height)
        signed_x = torch.where(signed_x < output_shape[1], signed_x, signed_x - width)
        valid_y = (signed_y >= -(source_shape[0] - 1)) & (signed_y < output_shape[0])
        valid_x = (signed_x >= -(source_shape[1] - 1)) & (signed_x < output_shape[1])
        axis_y = (signed_y.to(torch.float64) + (source_shape[0] - output_shape[0]) / 2.0) \
            * PIXEL_PITCH_M
        axis_x = (signed_x.to(torch.float64) + (source_shape[1] - output_shape[1]) / 2.0) \
            * PIXEL_PITCH_M
        valid = valid_y[:, None] & valid_x[None, :]
        canonical = torch.tensor(self._canonical_pairs, dtype=torch.long, device=device)
        offsets = self._canonical_offsets.to(device=device)
        cache = torch.empty((len(self._canonical_pairs), height, width), dtype=self.compute_dtype,
                            device=device)
        reverse_y = torch.cat((torch.zeros(1, device=device, dtype=torch.long),
                               torch.arange(height - 1, 0, -1, device=device)))
        reverse_x = torch.cat((torch.zeros(1, device=device, dtype=torch.long),
                               torch.arange(width - 1, 0, -1, device=device)))
        phase_y = torch.polar(torch.ones(height, device=device, dtype=torch.float64),
                              -2.0 * math.pi * torch.arange(height, device=device, dtype=torch.float64)
                              * (output_shape[0] - source_shape[0]) / height).to(self.compute_dtype)
        phase_x = torch.polar(torch.ones(width, device=device, dtype=torch.float64),
                              -2.0 * math.pi * torch.arange(width, device=device, dtype=torch.float64)
                              * (output_shape[1] - source_shape[1]) / width).to(self.compute_dtype)
        wave_number = 2.0 * math.pi / WAVELENGTH_M
        for start in range(0, len(self._canonical_pairs), self.cache_build_batch):
            stop = min(start + self.cache_build_batch, len(self._canonical_pairs))
            selected = canonical[start:stop]
            dy = axis_y[None, :, None] + offsets[selected[:, 0], None, None]
            dx = axis_x[None, None, :] + offsets[selected[:, 1], None, None]
            radius2 = self.distance_m**2 + dy.square() + dx.square()
            radius = radius2.sqrt()
            phase = wave_number * radius
            spatial_kernel = torch.polar(torch.ones_like(phase), phase) \
                * (self.distance_m / radius2) \
                * (1.0 / (2.0 * math.pi * radius) - 1j / WAVELENGTH_M)
            spatial_kernel = spatial_kernel * valid[None, :, :]
            spectra = torch.fft.fft2(spatial_kernel.to(torch.complex128))
            cache[start:stop].copy_(spectra.to(self.compute_dtype))
            del dy, dx, radius2, radius, phase, spatial_kernel, spectra
        self._kernel_cache_seconds = time.perf_counter() - started
        self._frequency_metadata = (reverse_y, reverse_x, phase_y, phase_x)
        return cache

    def _get_kernel_cache(self, source_shape, output_shape, fft_shape, device):
        key = self._cache_key(source_shape, output_shape, fft_shape, device)
        if self._kernel_cache_key != key or self._kernel_cache is None:
            self._kernel_cache = None
            self._kernel_cache_key = None
            self._kernel_cache = self._build_kernel_cache(source_shape, output_shape, fft_shape, device)
            self._kernel_cache_key = key
        return self._kernel_cache

    def _spectra_for_pairs(self, cache, target_ids, source_ids, fft_shape, source_shape, output_shape,
                           *, has_transposed_pair: bool):
        cache_indices = self._pair_cache_index.index_select(0, target_ids).index_select(1, source_ids)
        transposed = self._pair_transposed.index_select(0, target_ids).index_select(1, source_ids)
        flip_y = self._pair_flip_y.index_select(0, target_ids).index_select(1, source_ids)
        flip_x = self._pair_flip_x.index_select(0, target_ids).index_select(1, source_ids)
        pair_count_y, pair_count_x = cache_indices.shape
        height, width = fft_shape
        spectra = cache.index_select(0, cache_indices.reshape(-1)).reshape(
            pair_count_y, pair_count_x, height, width
        )
        if has_transposed_pair:
            spectra = torch.where(transposed[..., None, None], spectra.transpose(-2, -1), spectra)

        reverse_y, reverse_x, phase_y, phase_x = self._frequency_metadata
        reflected = spectra.index_select(-2, reverse_y) * phase_y[None, None, :, None]
        spectra = torch.where(flip_y[..., None, None], reflected, spectra)
        reflected = spectra.index_select(-1, reverse_x) * phase_x[None, None, None, :]
        spectra = torch.where(flip_x[..., None, None], reflected, spectra)
        return spectra

    def _apply_convolution(self, values, output_shape, centered, fft_shape, *, adjoint,
                           input_shape=None, target_indices=None, output_offset_cells=(0.0, 0.0),
                           source_chunk_size=None, target_order=None,
                           target_offset_y=None, target_offset_x=None):
        if adjoint:
            source_shape = input_shape[-2:]
        else:
            source_shape = tuple(values.shape[-2:])
        target_order = self.order if target_order is None else target_order
        cache_eligible = (
            not centered and target_order == self.order and target_offset_y is None
            and target_offset_x is None and tuple(output_offset_cells) == (0.0, 0.0)
            and source_shape[0] == source_shape[1] and output_shape[0] == output_shape[1]
            and fft_shape[0] == fft_shape[1]
        )
        if not cache_eligible:
            return super()._apply_convolution(
                values, output_shape, centered, fft_shape, adjoint=adjoint,
                input_shape=input_shape, target_indices=target_indices,
                output_offset_cells=output_offset_cells, source_chunk_size=source_chunk_size,
                target_order=target_order, target_offset_y=target_offset_y,
                target_offset_x=target_offset_x,
            )

        compute_dtype = self.compute_dtype
        cache = self._get_kernel_cache(source_shape, output_shape, fft_shape, values.device)
        target_ids = torch.arange(self.order**2, device=values.device) if centered else torch.as_tensor(
            target_indices if target_indices is not None else tuple(range(target_order**2)),
            dtype=torch.long, device=values.device
        )
        target_count = int(target_ids.numel())
        host_target_ids = tuple(range(target_order**2)) if centered or target_indices is None else tuple(
            int(index) for index in target_indices
        )
        source_count = self.order**2
        result_channels = source_count if adjoint else target_count
        result_shape = source_shape if adjoint else output_shape
        result = torch.empty((values.shape[0], result_channels, *result_shape),
                             device=values.device, dtype=compute_dtype)
        source_weights = self.source_area.to(device=values.device, dtype=torch.float64 if compute_dtype == torch.complex128
                                             else torch.float32)
        transformed = None
        if source_chunk_size is None:
            weighted_values = values.to(compute_dtype)
            if not adjoint:
                weighted_values = weighted_values * source_weights[None, :, None, None]
            transformed = torch.fft.fft2(weighted_values, s=fft_shape)
        inner_step = source_chunk_size if source_chunk_size is not None else self.pair_block
        schedule_key = (host_target_ids, adjoint, source_count, self.pair_block, inner_step)
        transpose_schedule = self._transpose_schedule_cache.get(schedule_key)
        if transpose_schedule is None:
            schedule_started = time.perf_counter()
            schedule: list[bool] = []
            for outer in range(0, result_channels, self.pair_block):
                outer_start = outer
                outer_stop = min(outer + self.pair_block, result_channels)
                for inner in range(0, target_count if adjoint else source_count, inner_step):
                    inner_stop = min(inner + inner_step, target_count if adjoint else source_count)
                    if adjoint:
                        target_block = host_target_ids[inner:inner_stop]
                        source_block = range(outer_start, outer_stop)
                    else:
                        target_block = host_target_ids[outer_start:outer_stop]
                        source_block = range(inner, inner_stop)
                    schedule.append(any(self._pair_transposed_host[target][source]
                                        for target in target_block for source in source_block))
            transpose_schedule = tuple(schedule)
            self._transpose_schedule_cache[schedule_key] = transpose_schedule
            self._transpose_schedule_build_seconds += time.perf_counter() - schedule_started
            self._transpose_schedule_build_count += 1
        else:
            self._transpose_schedule_cache_hits += 1
        schedule_index = 0
        for outer in range(0, result_channels, self.pair_block):
            outer_slice = slice(outer, min(outer + self.pair_block, result_channels))
            accumulated = torch.zeros((values.shape[0], outer_slice.stop - outer, *fft_shape),
                                      device=values.device, dtype=compute_dtype)
            inner_count = target_count if adjoint else source_count
            for inner in range(0, inner_count, inner_step):
                inner_slice = slice(inner, min(inner + inner_step, inner_count))
                if transformed is None:
                    value_chunk = values[:, inner_slice].to(compute_dtype)
                    if not adjoint:
                        value_chunk = value_chunk * source_weights[inner_slice][None, :, None, None]
                    transformed_chunk = torch.fft.fft2(value_chunk, s=fft_shape)
                else:
                    transformed_chunk = transformed[:, inner_slice]
                if adjoint:
                    pair_targets = target_ids[inner_slice]
                    pair_sources = torch.arange(outer_slice.start, outer_slice.stop, device=values.device)
                else:
                    pair_targets = target_ids[outer_slice]
                    pair_sources = torch.arange(inner_slice.start, inner_slice.stop, device=values.device)
                kernel = self._spectra_for_pairs(
                    cache, pair_targets, pair_sources, fft_shape, source_shape, output_shape,
                    has_transposed_pair=transpose_schedule[schedule_index],
                )
                schedule_index += 1
                if adjoint:
                    accumulated += torch.einsum("btyx,tsyx->bsyx", transformed_chunk, kernel.conj())
                else:
                    accumulated += torch.einsum("bsyx,tsyx->btyx", transformed_chunk, kernel)
            output = torch.fft.ifft2(accumulated)
            cropped = output[..., :result_shape[0], :result_shape[1]]
            if adjoint:
                cropped = cropped * source_weights[outer_slice][None, :, None, None]
            result[:, outer_slice] = cropped
        return result.to(values.dtype)


class RSChenLightGen400(FinitePixelChenLightGen400):
    """Frozen-geometry 2+4 resource probe using the RS production operator.

    Only active-aperture powers are returned. Full-window power and training
    qualification require additional measurements and cannot be inferred here.
    """

    def __init__(self, order: int):
        nn.Module.__init__(self)
        self.order = order
        self.config = FinitePixelConfig(quadrature_order=order)
        nodes, weights = _legendre_rule(order)
        wy, wx = torch.meshgrid(weights, weights, indexing="ij")
        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        self.register_buffer("node_weight_2d", (wy * wx).reshape(-1))
        self.register_buffer("node_y", yy.reshape(-1))
        self.register_buffer("node_x", xx.reshape(-1))
        self.encoder_phase = nn.Parameter(torch.zeros(2, 1, *ACTIVE_SHAPE))
        self.decoder_phase = nn.Parameter(torch.zeros(4, 1, *ACTIVE_SHAPE))
        self.propagator = FinitePixelRayleighSommerfeld(order)
        self._make_modes()

    def forward_from_image(self, intensity, *, capture_fields=False):
        if capture_fields:
            raise ValueError("RS resource probe does not claim full-window qualification")
        if intensity.shape[1:] != (1, *ACTIVE_SHAPE) or not torch.isfinite(intensity).all() \
                or bool(((intensity < 0) | (intensity > 1)).any()):
            raise ValueError("RS image must be finite [batch,1,400,400] intensity in [0,1]")
        density = math.sqrt(SOURCE_POWER_W / (ACTIVE_SHAPE[0] * ACTIVE_SHAPE[1] * PIXEL_PITCH_M**2))
        field = intensity[:, 0].sqrt().to(torch.complex64)[:, None].expand(-1, self.order**2, -1, -1) * density
        input_power = self._weighted_power(field)
        powers = []
        for index in range(2):
            field = self.propagator(self._phase(field, self.encoder_phase[index]))
            powers.append(self._weighted_power(field).detach())
        coefficients = self._project_coefficients(field)
        field = self.propagator(self._reemit(coefficients))
        powers.append(self._weighted_power(field).detach())
        for index in range(4):
            field = self.propagator(self._phase(field, self.decoder_phase[index]))
            powers.append(self._weighted_power(field).detach())
        weights = self.node_weight_2d.to(field.device, field.real.dtype)
        camera = (field.abs().square() * weights[None, :, None, None]).sum(1, keepdim=True) * PIXEL_PITCH_M**2
        return {"physical_w_per_pixel": camera, "ols_coefficients": coefficients,
                "input_power_w": input_power, "active_aperture_plane_powers_w": torch.stack(powers, 1)}
