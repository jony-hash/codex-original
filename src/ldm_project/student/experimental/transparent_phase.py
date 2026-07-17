"""Ideal transparent-exterior phase stack for diagnostic-only experiments.

The learned 100x100 phase window is embedded in a larger free-space grid.  The
unprinted exterior is identity transmission, not a zero field or a mechanical
aperture.  This module intentionally does not model glass thickness, Fresnel
loss, absorption, quantisation, or alignment error.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ...optics.propagation import (
    angular_spectrum_propagate,
    band_limited_angular_spectrum_propagate,
)


def _energy(field: torch.Tensor) -> torch.Tensor:
    return field.detach().abs().square().flatten(start_dim=1).sum(dim=1)


def _embed(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if height > size or width > size:
        raise ValueError(f"field {(height, width)} exceeds propagation grid {size}")
    if (height, width) == (size, size):
        return field
    result = field.new_zeros(*field.shape[:-2], size, size)
    top, left = (size - height) // 2, (size - width) // 2
    result[..., top : top + height, left : left + width] = field
    return result


def _center_crop(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"field {(height, width)} is smaller than crop size {size}")
    top, left = (height - size) // 2, (width - size) // 2
    return field[..., top : top + size, left : left + size]


def _guard_band_ratio(field: torch.Tensor, fraction: float) -> torch.Tensor:
    """Energy in the outer state-grid guard band, measured after crop-back."""
    height, width = field.shape[-2:]
    border = max(1, int(round(min(height, width) * fraction)))
    mask = torch.zeros(height, width, device=field.device, dtype=torch.bool)
    mask[:border] = True
    mask[-border:] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    values = field.abs().square()
    return values[..., mask].flatten(start_dim=1).sum(dim=1) / _energy(field).clamp_min(
        1.0e-12
    )


class TransparentIdentityDiffractiveStack(nn.Module):
    """Phase-only masks with identity transmission outside the active window."""

    boundary_contract = "ideal_transparent_thin_phase_v1"

    def __init__(
        self,
        num_layers: int,
        grid_size: int,
        layer_spacing: float,
        wavelength: float,
        pixel_size: float,
        propagation_grid_size: int,
        phase_init: str = "uniform_pi",
        *,
        transparent_propagation_method: str = "periodic_asm",
        transparent_fft_grid_size: int | None = None,
        transparent_edge_guard_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if propagation_grid_size < grid_size:
            raise ValueError("propagation_grid_size must be at least grid_size")
        if transparent_propagation_method not in {
            "periodic_asm",
            "padded_band_limited_asm",
        }:
            raise ValueError(
                "transparent_propagation_method must be periodic_asm or "
                "padded_band_limited_asm"
            )
        if transparent_fft_grid_size is not None and transparent_fft_grid_size < propagation_grid_size:
            raise ValueError(
                "transparent_fft_grid_size must be at least propagation_grid_size"
            )
        if not 0.0 < float(transparent_edge_guard_fraction) < 0.5:
            raise ValueError("transparent_edge_guard_fraction must be in (0, 0.5)")
        self.num_layers = int(num_layers)
        self.grid_size = int(grid_size)
        self.layer_spacing = float(layer_spacing)
        self.wavelength = float(wavelength)
        self.pixel_size = float(pixel_size)
        self.propagation_grid_size = int(propagation_grid_size)
        self.phase_init = phase_init
        self.transparent_propagation_method = transparent_propagation_method
        self.transparent_fft_grid_size = (
            int(transparent_fft_grid_size)
            if transparent_fft_grid_size is not None
            else None
        )
        self.transparent_edge_guard_fraction = float(transparent_edge_guard_fraction)
        if transparent_propagation_method == "padded_band_limited_asm":
            if self.transparent_fft_grid_size is None:
                raise ValueError(
                    "transparent_fft_grid_size is required for "
                    "padded_band_limited_asm"
                )
            self.boundary_contract = "ideal_transparent_thin_phase_v2"
        self.phase = nn.Parameter(torch.empty(num_layers, 1, grid_size, grid_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.phase_init == "uniform_pi":
            nn.init.uniform_(self.phase, -math.pi, math.pi)
        elif self.phase_init == "uniform_2pi":
            nn.init.uniform_(self.phase, 0.0, 2.0 * math.pi)
        elif self.phase_init == "small":
            nn.init.uniform_(self.phase, -0.05, 0.05)
        else:
            raise ValueError(f"Unsupported phase_init: {self.phase_init}")

    def phase_transfers(
        self, mask_component: str = "trained", phase_layer_mask: tuple[bool, ...] = ()
    ) -> torch.Tensor:
        if mask_component not in {"trained", "phase_only", "amplitude_only"}:
            raise ValueError(f"Unsupported mask component: {mask_component!r}")
        phase = torch.zeros_like(self.phase) if mask_component == "amplitude_only" else self.phase
        if phase_layer_mask:
            if len(phase_layer_mask) != self.num_layers:
                raise ValueError("phase_layer_mask length must equal num_layers")
            layer_mask = torch.tensor(
                phase_layer_mask, device=phase.device, dtype=phase.dtype
            ).view(-1, 1, 1, 1)
            phase = phase * layer_mask
        active = torch.polar(torch.ones_like(phase), phase)
        transfers = torch.ones(
            self.num_layers,
            1,
            self.propagation_grid_size,
            self.propagation_grid_size,
            device=phase.device,
            dtype=active.dtype,
        )
        top = (self.propagation_grid_size - self.grid_size) // 2
        left = (self.propagation_grid_size - self.grid_size) // 2
        mask = torch.zeros_like(transfers)
        mask[..., top : top + self.grid_size, left : left + self.grid_size] = 1
        return transfers + mask * (
            _embed(active, self.propagation_grid_size) - 1
        )

    def forward(
        self,
        field: torch.Tensor,
        *,
        mask_component: str = "trained",
        phase_layer_mask: tuple[bool, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        field = _embed(field, self.propagation_grid_size)
        transfers = self.phase_transfers(mask_component, phase_layer_mask).to(field.dtype)
        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, list[torch.Tensor]] = {
            "input_energy": [],
            "padded_input_energy": [],
            "propagated_full_energy": [],
            "cropped_energy": [],
            "retained_energy": [],
            "discarded_energy": [],
            "crop_retention": [],
            "propagation_energy_ratio": [],
            "discarded_energy_ratio": [],
            "guard_band_energy_ratio": [],
        }
        for index in range(self.num_layers):
            field = field * transfers[index].unsqueeze(0)
            input_energy = _energy(field)
            if self.transparent_propagation_method == "periodic_asm":
                propagated = angular_spectrum_propagate(
                    field, self.layer_spacing, self.wavelength, self.pixel_size
                )
                padded_input_energy = input_energy
                propagated_energy = _energy(propagated)
                retained = propagated
                retained_energy = propagated_energy
                discarded_energy = torch.zeros_like(retained_energy)
                crop_retention = torch.ones_like(retained_energy)
                discarded_ratio = torch.zeros_like(retained_energy)
            else:
                # 这里只裁回毫米级完整 state grid，绝不裁回 100x100 学习窗口。
                fft_grid = _embed(field, int(self.transparent_fft_grid_size))
                padded_input_energy = _energy(fft_grid)
                propagated_padded = band_limited_angular_spectrum_propagate(
                    fft_grid,
                    self.layer_spacing,
                    self.wavelength,
                    self.pixel_size,
                )
                propagated_energy = _energy(propagated_padded)
                retained = _center_crop(propagated_padded, self.propagation_grid_size)
                retained_energy = _energy(retained)
                discarded_energy = (propagated_energy - retained_energy).clamp_min(0.0)
                crop_retention = retained_energy / propagated_energy.clamp_min(1.0e-12)
                discarded_ratio = discarded_energy / propagated_energy.clamp_min(1.0e-12)
                propagated = retained
            field = propagated
            outputs.append(field)
            diagnostics["input_energy"].append(input_energy)
            diagnostics["padded_input_energy"].append(padded_input_energy)
            diagnostics["propagated_full_energy"].append(propagated_energy)
            diagnostics["cropped_energy"].append(retained_energy)
            diagnostics["retained_energy"].append(retained_energy)
            diagnostics["discarded_energy"].append(discarded_energy)
            diagnostics["crop_retention"].append(crop_retention)
            diagnostics["propagation_energy_ratio"].append(
                retained_energy / input_energy.clamp_min(1.0e-12)
            )
            diagnostics["discarded_energy_ratio"].append(discarded_ratio)
            diagnostics["guard_band_energy_ratio"].append(
                _guard_band_ratio(field, self.transparent_edge_guard_fraction)
            )
        diagnostics_out = {
            key: torch.stack(values, dim=1) for key, values in diagnostics.items()
        }
        diagnostics_out["transparent_propagation_method"] = self.transparent_propagation_method
        diagnostics_out["transparent_state_grid_size"] = torch.tensor(
            self.propagation_grid_size, device=field.device
        )
        diagnostics_out["transparent_fft_grid_size"] = torch.tensor(
            self.transparent_fft_grid_size or self.propagation_grid_size,
            device=field.device,
        )
        diagnostics_out["transparent_edge_guard_fraction"] = torch.tensor(
            self.transparent_edge_guard_fraction, device=field.device
        )
        return field, torch.stack(outputs, dim=1), diagnostics_out


__all__ = ["TransparentIdentityDiffractiveStack"]
