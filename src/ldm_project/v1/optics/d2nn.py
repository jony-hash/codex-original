"""Formal phase-only D2NN elements and their padding/provenance diagnostics."""
from __future__ import annotations

import math

import torch
from torch import nn

from .propagation import angular_spectrum_propagate


def _center_embed_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = size
    if target_height < height or target_width < width:
        raise ValueError(f"padding size {size} must contain field {(height, width)}")
    if (height, width) == size:
        return field
    padded = field.new_zeros(*field.shape[:-2], target_height, target_width)
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    padded[..., top : top + height, left : left + width] = field
    return padded


def _center_crop_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = size
    if target_height > height or target_width > width:
        raise ValueError(f"crop size {size} exceeds field {(height, width)}")
    if (height, width) == size:
        return field
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return field[..., top : top + target_height, left : left + target_width]


def _complex_field_energy(field: torch.Tensor) -> torch.Tensor:
    return field.detach().abs().pow(2).flatten(start_dim=1).sum(dim=1)


def angular_spectrum_propagate_center_padded(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
    propagation_grid_size: int,
) -> torch.Tensor:
    field_size = field.shape[-2:]
    if propagation_grid_size == field_size[-1] and field_size[-2] == field_size[-1]:
        return angular_spectrum_propagate(field, distance, wavelength, pixel_size)
    padded = _center_embed_complex_field(field, (propagation_grid_size, propagation_grid_size))
    return _center_crop_complex_field(
        angular_spectrum_propagate(padded, distance, wavelength, pixel_size), field_size
    )


def angular_spectrum_propagate_center_padded_with_diagnostics(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
    propagation_grid_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    field_size = field.shape[-2:]
    input_energy = _complex_field_energy(field)
    if propagation_grid_size == field_size[-1] and field_size[-2] == field_size[-1]:
        propagated = angular_spectrum_propagate(field, distance, wavelength, pixel_size)
        propagated_energy = _complex_field_energy(propagated)
        return propagated, {
            "input_energy": input_energy,
            "padded_input_energy": input_energy,
            "propagated_full_energy": propagated_energy,
            "cropped_energy": propagated_energy,
            "crop_retention": torch.ones_like(propagated_energy),
            "propagation_energy_ratio": propagated_energy / input_energy.clamp_min(1.0e-12),
        }
    padded = _center_embed_complex_field(field, (propagation_grid_size, propagation_grid_size))
    padded_energy = _complex_field_energy(padded)
    propagated = angular_spectrum_propagate(padded, distance, wavelength, pixel_size)
    cropped = _center_crop_complex_field(propagated, field_size)
    propagated_energy = _complex_field_energy(propagated)
    cropped_energy = _complex_field_energy(cropped)
    return cropped, {
        "input_energy": input_energy,
        "padded_input_energy": padded_energy,
        "propagated_full_energy": propagated_energy,
        "cropped_energy": cropped_energy,
        "crop_retention": cropped_energy / propagated_energy.clamp_min(1.0e-12),
        "propagation_energy_ratio": propagated_energy / padded_energy.clamp_min(1.0e-12),
    }


class PhaseOnlyDiffractiveStack(nn.Module):
    """Phase modulation → center padding → ASM → center crop at every layer."""

    def __init__(
        self,
        num_layers: int,
        grid_size: int,
        layer_spacing: float,
        wavelength: float,
        pixel_size: float,
        propagation_grid_size: int,
        phase_init: str = "uniform_pi",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if propagation_grid_size < grid_size:
            raise ValueError("propagation_grid_size must be at least grid_size")
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.layer_spacing = layer_spacing
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.propagation_grid_size = propagation_grid_size
        self.phase_init = phase_init
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
            raise ValueError(f"Unsupported one_shot.diffractive_phase_init: {self.phase_init}")

    def phase_transfers(
        self, mask_component: str = "trained", phase_layer_mask: tuple[bool, ...] = ()
    ) -> torch.Tensor:
        if mask_component not in {"trained", "phase_only", "amplitude_only"}:
            raise ValueError(f"Unsupported decoder diagnostic mask component: {mask_component!r}")
        phase = torch.zeros_like(self.phase) if mask_component == "amplitude_only" else self.phase
        if phase_layer_mask:
            mask = torch.tensor(phase_layer_mask, device=phase.device, dtype=phase.dtype).view(-1, 1, 1, 1)
            phase = phase * mask
        return torch.polar(torch.ones_like(self.phase), phase)

    def forward(
        self, field: torch.Tensor, *, mask_component: str = "trained", phase_layer_mask: tuple[bool, ...] = ()
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs = []
        values: dict[str, list[torch.Tensor]] = {
            "input_energy": [], "padded_input_energy": [], "propagated_full_energy": [],
            "cropped_energy": [], "crop_retention": [], "propagation_energy_ratio": [],
        }
        transfers = self.phase_transfers(mask_component, phase_layer_mask).to(field.dtype)
        for layer_index in range(self.num_layers):
            field = field * transfers[layer_index].unsqueeze(0)
            field, diagnostics = angular_spectrum_propagate_center_padded_with_diagnostics(
                field, self.layer_spacing, self.wavelength, self.pixel_size, self.propagation_grid_size
            )
            for key, value in diagnostics.items():
                values[key].append(value)
            outputs.append(field)
        return field, torch.stack(outputs, dim=1), {key: torch.stack(value, dim=1) for key, value in values.items()}
