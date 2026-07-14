from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..optics import angular_spectrum_propagate


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


class ElectronicGeneratorE(nn.Module):
    def __init__(
        self,
        image_size: int,
        grid_size: int,
        code_dim: int,
        num_classes: int,
        label_embed_dim: int,
        hidden_dim: int,
        arch: str = "mlp",
        code_residual_scale: float = 0.0,
        command_transform: str = "direct_amplitude",
        background_floor: float = 0.05,
        template_init: str = "none",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.arch = arch
        self.code_residual_scale = float(code_residual_scale)
        self.command_transform = command_transform
        self.background_floor = float(background_floor)
        self.template_init = template_init
        if self.arch not in {
            "label_template",
            "label_template_code_residual",
            "mlp",
            "template_residual_mlp",
        }:
            raise ValueError(
                "Unsupported one_shot.generator_e_arch: "
                f"{self.arch}; expected label_template, label_template_code_residual, "
                "mlp, or template_residual_mlp."
            )
        if self.command_transform not in {"direct_amplitude", "intensity_to_amplitude"}:
            raise ValueError(
                f"Unsupported generator command transform: {self.command_transform}"
            )
        if (
            self.command_transform == "intensity_to_amplitude"
            and self.background_floor >= 1.0
        ):
            raise ValueError("intensity_to_amplitude requires background_floor < 1.0")
        if self.arch in {"label_template", "label_template_code_residual"}:
            if self.template_init != "mnist_prototype":
                raise ValueError(
                    f"{self.arch} requires template_init='mnist_prototype'"
                )
            self.template_logits = nn.Parameter(
                torch.zeros(num_classes, 1, image_size, image_size)
            )
            self.label_embedding = None
            self.net = None
            self.residual_net = None
        else:
            self.register_parameter("template_logits", None)
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
            self.net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )
            self.residual_net = None
        if self.arch == "label_template_code_residual":
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
            self.residual_net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )
        if self.arch == "template_residual_mlp":
            self.residual_net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )

    def initialize_label_templates(self, intensity_bank: torch.Tensor) -> None:
        if (
            self.arch not in {"label_template", "label_template_code_residual"}
            or self.template_logits is None
        ):
            raise ValueError(
                "Label templates can only initialize generator_e_arch='label_template' "
                "or 'label_template_code_residual'."
            )
        expected = (self.num_classes, 1, self.image_size, self.image_size)
        if tuple(intensity_bank.shape) != expected:
            raise ValueError(
                f"label template bank must have shape {expected}, got {tuple(intensity_bank.shape)}"
            )
        bank = intensity_bank.detach().to(
            device=self.template_logits.device,
            dtype=self.template_logits.dtype,
        )
        if not torch.isfinite(bank).all():
            raise ValueError("label template bank must contain only finite values")
        eps = torch.finfo(bank.dtype).eps
        stable = bank.clamp(eps, 1.0 - eps)
        with torch.no_grad():
            self.template_logits.copy_(torch.logit(stable))

    def _validate_inputs(
        self, code: torch.Tensor, class_labels: torch.Tensor
    ) -> torch.Tensor:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError(
                f"code must have shape [batch, {self.code_dim}], got {tuple(code.shape)}"
            )
        labels = class_labels.long()
        if labels.ndim != 1 or labels.shape[0] != code.shape[0]:
            raise ValueError(
                "class_labels must be a 1D tensor with the same batch size as code."
            )
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"class_labels must be in [0, {self.num_classes - 1}].")
        return labels

    def _features(self, code: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        labels = self._validate_inputs(code, class_labels)
        if self.label_embedding is None:
            raise ValueError("label_template does not use MLP features")
        label_features = self.label_embedding(labels)
        return torch.cat([code, label_features], dim=1)

    def _to_amplitude_command(self, desired_intensity: torch.Tensor) -> torch.Tensor:
        if self.command_transform == "direct_amplitude":
            return desired_intensity
        amplitude = desired_intensity.clamp_min(0.0).sqrt()
        return (
            (amplitude - self.background_floor) / (1.0 - self.background_floor)
        ).clamp(0.0, 1.0)

    def _diagnostic_metadata(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "generator_command_transform": reference.new_tensor(
                1 if self.command_transform == "intensity_to_amplitude" else 0,
                dtype=torch.int64,
            ),
            "generator_code_conditioning_active": reference.new_tensor(
                self.arch != "label_template",
                dtype=torch.bool,
            ),
        }

    def forward_with_diagnostics(
        self,
        code: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.arch in {"label_template", "label_template_code_residual"}:
            labels = self._validate_inputs(code, class_labels)
            if self.template_logits is None:
                raise RuntimeError(f"{self.arch} generator is missing template logits")
            template_logits = self.template_logits.index_select(0, labels)
            base_intensity = torch.sigmoid(template_logits)
            if self.arch == "label_template":
                desired_intensity = base_intensity
                residual_map = torch.zeros_like(desired_intensity)
            else:
                if self.residual_net is None:
                    raise RuntimeError(
                        "label_template_code_residual generator is missing residual_net"
                    )
                residual_logits = self.residual_net(self._features(code, class_labels))
                residual_map = torch.tanh(
                    residual_logits.reshape(
                        code.shape[0], 1, self.image_size, self.image_size
                    )
                )
                desired_intensity = torch.sigmoid(
                    template_logits + self.code_residual_scale * residual_map
                )
            command = self._to_amplitude_command(desired_intensity)
            base_command = self._to_amplitude_command(base_intensity)
            background_mask = (base_intensity.detach() < 0.2).float()
            bg_abs = (residual_map.abs() * background_mask).flatten(1).sum(
                dim=1
            ) / background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            return command, {
                "generator_base_command": base_command,
                "generator_desired_intensity": desired_intensity,
                "generator_residual_map": residual_map,
                "generator_residual_abs_mean": residual_map.abs()
                .flatten(1)
                .mean(dim=1),
                "generator_residual_bg_abs_mean": bg_abs,
                **self._diagnostic_metadata(command),
            }
        features = self._features(code, class_labels)
        if self.net is None:
            raise RuntimeError("MLP generator is missing its network")
        base_logits = self.net(features)
        base_grid = torch.sigmoid(
            base_logits.reshape(code.shape[0], 1, self.image_size, self.image_size)
        )
        if self.residual_net is None:
            zero_residual = torch.zeros_like(base_grid)
            command = self._to_amplitude_command(base_grid)
            diagnostics = {
                "generator_base_command": command,
                "generator_residual_map": zero_residual,
                "generator_residual_abs_mean": zero_residual.flatten(1).mean(dim=1),
                "generator_residual_bg_abs_mean": zero_residual.flatten(1).mean(dim=1),
                **self._diagnostic_metadata(command),
            }
            if self.command_transform == "intensity_to_amplitude":
                diagnostics["generator_desired_intensity"] = base_grid
            return command, diagnostics

        residual_logits = self.residual_net(features)
        residual_map = torch.tanh(
            residual_logits.reshape(code.shape[0], 1, self.image_size, self.image_size)
        )
        slm_logits = base_logits.reshape(
            code.shape[0], 1, self.image_size, self.image_size
        )
        desired_intensity = torch.sigmoid(
            slm_logits + self.code_residual_scale * residual_map
        )
        command = self._to_amplitude_command(desired_intensity)
        background_mask = (base_grid.detach() < 0.2).float()
        bg_abs = (residual_map.abs() * background_mask).flatten(1).sum(
            dim=1
        ) / background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
        diagnostics = {
            "generator_base_command": self._to_amplitude_command(base_grid),
            "generator_residual_map": residual_map,
            "generator_residual_abs_mean": residual_map.abs().flatten(1).mean(dim=1),
            "generator_residual_bg_abs_mean": bg_abs,
            **self._diagnostic_metadata(command),
        }
        if self.command_transform == "intensity_to_amplitude":
            diagnostics["generator_desired_intensity"] = desired_intensity
        return command, diagnostics

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        slm_grid, _ = self.forward_with_diagnostics(code, class_labels)
        return slm_grid


class IdealAmplitudeSLMModel(nn.Module):
    def __init__(
        self, background_floor: float, residual_phase_scale_rad: float = 0.0
    ) -> None:
        super().__init__()
        if not 0.0 <= background_floor <= 1.0:
            raise ValueError(
                f"slm_background_floor must be in [0, 1], got {background_floor}"
            )
        self.background_floor = float(background_floor)
        self.residual_phase_scale_rad = float(residual_phase_scale_rad)

    def forward(self, slm_command: torch.Tensor) -> dict[str, torch.Tensor]:
        slm_command = slm_command.clamp(0.0, 1.0)
        amplitude = self.background_floor + (1.0 - self.background_floor) * slm_command
        phase = (
            slm_command.new_full(slm_command.shape, self.residual_phase_scale_rad)
            * slm_command
        )
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

    def phase_transfers(
        self,
        mask_component: str = "trained",
        phase_layer_mask: tuple[bool, ...] = (),
    ) -> torch.Tensor:
        if mask_component not in {"trained", "phase_only", "amplitude_only"}:
            raise ValueError(
                f"Unsupported decoder diagnostic mask component: {mask_component!r}"
            )
        amplitude = torch.ones_like(self.phase)
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

    def forward(
        self,
        field: torch.Tensor,
        *,
        mask_component: str = "trained",
        phase_layer_mask: tuple[bool, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs = []
        diagnostic_values: dict[str, list[torch.Tensor]] = {
            "input_energy": [],
            "padded_input_energy": [],
            "propagated_full_energy": [],
            "cropped_energy": [],
            "crop_retention": [],
            "propagation_energy_ratio": [],
        }
        transfers = self.phase_transfers(mask_component, phase_layer_mask).to(
            field.dtype
        )
        for layer_index in range(self.num_layers):
            field = field * transfers[layer_index].unsqueeze(0)
            field, layer_diagnostics = (
                _angular_spectrum_propagate_center_padded_with_diagnostics(
                    field,
                    self.layer_spacing,
                    self.wavelength,
                    self.pixel_size,
                    self.propagation_grid_size,
                )
            )
            for key, value in layer_diagnostics.items():
                diagnostic_values[key].append(value)
            outputs.append(field)
        diagnostics = {
            key: torch.stack(values, dim=1) for key, values in diagnostic_values.items()
        }
        return field, torch.stack(outputs, dim=1), diagnostics


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
