from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .optics import angular_spectrum_propagate


def _resize_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _resize_real_intensity(intensity: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(intensity.shape[-2:]) == size:
        return intensity
    return F.interpolate(intensity, size=size, mode="area")


def _center_embed_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
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


def _center_crop_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
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
    propagated = angular_spectrum_propagate(padded_field, distance, wavelength, pixel_size)
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
            "propagation_energy_ratio": propagated_full_energy / input_energy.clamp_min(1.0e-12),
        }
        return propagated, diagnostics

    padded_size = (propagation_grid_size, propagation_grid_size)
    padded_field = _center_embed_complex_field(field, padded_size)
    padded_input_energy = _complex_field_energy(padded_field)
    propagated = angular_spectrum_propagate(padded_field, distance, wavelength, pixel_size)
    cropped = _center_crop_complex_field(propagated, field_size)
    propagated_full_energy = _complex_field_energy(propagated)
    cropped_energy = _complex_field_energy(cropped)
    diagnostics = {
        "input_energy": input_energy,
        "padded_input_energy": padded_input_energy,
        "propagated_full_energy": propagated_full_energy,
        "cropped_energy": cropped_energy,
        "crop_retention": cropped_energy / propagated_full_energy.clamp_min(1.0e-12),
        "propagation_energy_ratio": propagated_full_energy / padded_input_energy.clamp_min(1.0e-12),
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


class ElectronicGeneratorE(nn.Module):
    def __init__(
        self,
        image_size: int,
        grid_size: int,
        code_dim: int,
        num_classes: int,
        label_embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(code_dim + label_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, image_size * image_size),
        )

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError(f"code must have shape [batch, {self.code_dim}], got {tuple(code.shape)}")
        labels = class_labels.long()
        if labels.ndim != 1 or labels.shape[0] != code.shape[0]:
            raise ValueError("class_labels must be a 1D tensor with the same batch size as code.")
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"class_labels must be in [0, {self.num_classes - 1}].")
        label_features = self.label_embedding(labels)
        slm_logits = self.net(torch.cat([code, label_features], dim=1))
        slm_grid = torch.sigmoid(slm_logits.reshape(code.shape[0], 1, self.image_size, self.image_size))
        return slm_grid


class IdealAmplitudeSLMModel(nn.Module):
    def __init__(self, background_floor: float, residual_phase_scale_rad: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= background_floor <= 1.0:
            raise ValueError(f"slm_background_floor must be in [0, 1], got {background_floor}")
        self.background_floor = float(background_floor)
        self.residual_phase_scale_rad = float(residual_phase_scale_rad)

    def forward(self, slm_command: torch.Tensor) -> dict[str, torch.Tensor]:
        slm_command = slm_command.clamp(0.0, 1.0)
        amplitude = self.background_floor + (1.0 - self.background_floor) * slm_command
        phase = slm_command.new_full(slm_command.shape, self.residual_phase_scale_rad) * slm_command
        return {
            "slm_command": slm_command,
            "slm_amplitude": amplitude,
            "slm_phase": phase,
            "slm_transfer": torch.polar(amplitude, phase),
        }


class PhaseOnlyDiffractiveStack(nn.Module):
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
            raise ValueError(
                "propagation_grid_size must be at least grid_size, "
                f"got propagation_grid_size={propagation_grid_size} and grid_size={grid_size}."
            )
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
            raise ValueError(
                "Unsupported one_shot.diffractive_phase_init: "
                f"{self.phase_init}; expected uniform_pi, uniform_2pi, or small."
            )

    def phase_transfers(self) -> torch.Tensor:
        amplitude = torch.ones_like(self.phase)
        return torch.polar(amplitude, self.phase)

    def forward(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs = []
        diagnostic_values: dict[str, list[torch.Tensor]] = {
            "input_energy": [],
            "padded_input_energy": [],
            "propagated_full_energy": [],
            "cropped_energy": [],
            "crop_retention": [],
            "propagation_energy_ratio": [],
        }
        transfers = self.phase_transfers().to(field.dtype)
        for layer_index in range(self.num_layers):
            field = field * transfers[layer_index].unsqueeze(0)
            field, layer_diagnostics = _angular_spectrum_propagate_center_padded_with_diagnostics(
                field,
                self.layer_spacing,
                self.wavelength,
                self.pixel_size,
                self.propagation_grid_size,
            )
            for key, value in layer_diagnostics.items():
                diagnostic_values[key].append(value)
            outputs.append(field)
        diagnostics = {key: torch.stack(values, dim=1) for key, values in diagnostic_values.items()}
        return field, torch.stack(outputs, dim=1), diagnostics
