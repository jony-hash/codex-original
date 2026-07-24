from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ...optics import angular_spectrum_propagate
from ...optics.d2nn import PhaseOnlyDiffractiveStack
from ...optics.devices import IdealAmplitudeSLMModel

__all__ = [
    "IdealAmplitudeSLMModel",
    "PhaseAmplitudeDiffractiveStack",
    "PhaseOnlyDiffractiveStack",
    "_angular_spectrum_propagate_center_padded",
    "_angular_spectrum_propagate_center_padded_with_diagnostics",
    "_apply_complex_ols",
    "_center_crop_complex_field",
    "_center_embed_complex_field",
    "_complex_field_energy",
    "_flatten_complex_field",
    "_resize_complex_field",
    "_resize_real_intensity",
    "_vector_to_complex_field",
]


def _resize_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _resize_real_intensity(
    intensity: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    if tuple(intensity.shape[-2:]) == size:
        return intensity
    return F.interpolate(intensity, size=size, mode="area")


def _center_embed_complex_field(
    field: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = size
    if target_height < height or target_width < width:
        raise ValueError(
            "padding size must be at least the field size, "
            f"got field {(height, width)} and padding size {size}."
        )
    if (height, width) == size:
        return field

    padded = field.new_zeros(*field.shape[:-2], target_height, target_width)
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    padded[..., top : top + height, left : left + width] = field
    return padded


def _center_crop_complex_field(
    field: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = size
    if target_height > height or target_width > width:
        raise ValueError(
            "crop size must be no larger than the field size, "
            f"got field {(height, width)} and crop size {size}."
        )
    if (height, width) == size:
        return field

    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return field[..., top : top + target_height, left : left + target_width]


def _complex_field_energy(field: torch.Tensor) -> torch.Tensor:
    return field.detach().abs().pow(2).flatten(start_dim=1).sum(dim=1)


def _angular_spectrum_propagate_center_padded(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
    propagation_grid_size: int,
) -> torch.Tensor:
    field_size = field.shape[-2:]
    if propagation_grid_size == field_size[-1] and field_size[-2] == field_size[-1]:
        return angular_spectrum_propagate(field, distance, wavelength, pixel_size)
    padded_size = (propagation_grid_size, propagation_grid_size)
    padded_field = _center_embed_complex_field(field, padded_size)
    propagated = angular_spectrum_propagate(
        padded_field, distance, wavelength, pixel_size
    )
    return _center_crop_complex_field(propagated, field_size)


def _angular_spectrum_propagate_center_padded_with_diagnostics(
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
        propagated_full_energy = _complex_field_energy(propagated)
        diagnostics = {
            "input_energy": input_energy,
            "padded_input_energy": input_energy,
            "propagated_full_energy": propagated_full_energy,
            "cropped_energy": propagated_full_energy,
            "crop_retention": torch.ones_like(propagated_full_energy),
            "propagation_energy_ratio": propagated_full_energy
            / input_energy.clamp_min(1.0e-12),
        }
        return propagated, diagnostics

    padded_size = (propagation_grid_size, propagation_grid_size)
    padded_field = _center_embed_complex_field(field, padded_size)
    padded_input_energy = _complex_field_energy(padded_field)
    propagated = angular_spectrum_propagate(
        padded_field, distance, wavelength, pixel_size
    )
    cropped = _center_crop_complex_field(propagated, field_size)
    propagated_full_energy = _complex_field_energy(propagated)
    cropped_energy = _complex_field_energy(cropped)
    diagnostics = {
        "input_energy": input_energy,
        "padded_input_energy": padded_input_energy,
        "propagated_full_energy": propagated_full_energy,
        "cropped_energy": cropped_energy,
        "crop_retention": cropped_energy / propagated_full_energy.clamp_min(1.0e-12),
        "propagation_energy_ratio": propagated_full_energy
        / padded_input_energy.clamp_min(1.0e-12),
    }
    return cropped, diagnostics


def _flatten_complex_field(field: torch.Tensor) -> torch.Tensor:
    return torch.cat([field.real.flatten(1), field.imag.flatten(1)], dim=1)


def _vector_to_complex_field(vector: torch.Tensor, grid_size: int) -> torch.Tensor:
    batch_size = vector.shape[0]
    half_width = grid_size * grid_size
    real = vector[:, :half_width].reshape(batch_size, 1, grid_size, grid_size)
    imag = vector[:, half_width:].reshape(batch_size, 1, grid_size, grid_size)
    return torch.complex(real, imag)


def _apply_complex_ols(field: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "identity":
        return field
    if kind in {"pointwise_tanh", "tanh"}:
        real = torch.tanh(field.real)
        imag = torch.tanh(field.imag)
    elif kind == "gelu":
        real = F.gelu(field.real)
        imag = F.gelu(field.imag)
    elif kind == "silu":
        real = F.silu(field.real)
        imag = F.silu(field.imag)
    elif kind == "relu":
        real = F.relu(field.real)
        imag = F.relu(field.imag)
    else:
        raise ValueError(
            "Unsupported one_shot.ols_kind: "
            f"{kind}; expected gelu, tanh, silu, relu, or identity."
        )
    return torch.complex(real, imag)


class PhaseAmplitudeDiffractiveStack(PhaseOnlyDiffractiveStack):
    def __init__(
        self,
        num_layers: int,
        grid_size: int,
        layer_spacing: float,
        wavelength: float,
        pixel_size: float,
        propagation_grid_size: int,
        phase_init: str = "uniform_pi",
        amplitude_logit_init: float = 4.0,
    ) -> None:
        self.amplitude_logit_init = float(amplitude_logit_init)
        super().__init__(
            num_layers,
            grid_size,
            layer_spacing,
            wavelength,
            pixel_size,
            propagation_grid_size,
            phase_init,
        )
        self.amplitude_logits = nn.Parameter(
            torch.empty(num_layers, 1, grid_size, grid_size)
        )
        self.reset_amplitude_parameters()

    def reset_amplitude_parameters(self) -> None:
        nn.init.constant_(self.amplitude_logits, self.amplitude_logit_init)

    def phase_transfers(
        self,
        mask_component: str = "trained",
        phase_layer_mask: tuple[bool, ...] = (),
    ) -> torch.Tensor:
        if mask_component not in {"trained", "phase_only", "amplitude_only"}:
            raise ValueError(
                f"Unsupported decoder diagnostic mask component: {mask_component!r}"
            )
        if mask_component == "phase_only":
            amplitude = torch.ones_like(self.phase)
        else:
            amplitude = torch.sigmoid(self.amplitude_logits)
        phase = (
            torch.zeros_like(self.phase)
            if mask_component == "amplitude_only"
            else self.phase
        )
        if phase_layer_mask:
            layer_mask = torch.tensor(
                phase_layer_mask, device=phase.device, dtype=phase.dtype
            ).view(-1, 1, 1, 1)
            phase = phase * layer_mask
        return torch.polar(amplitude, phase)
