from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .config import OneShotConfig, PhysicsConfig
from .optics import BackOpticalPath, FrontOpticalPath, angular_spectrum_propagate, normalize_real_intensity


ONE_SHOT_MODEL_VERSION = "ldm_distilled_single_shot_optical_student_v3"


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


class OneShotOpticalGenerator(nn.Module):
    def __init__(
        self,
        physics: PhysicsConfig,
        config: OneShotConfig | None = None,
        image_size: int = 200,
    ) -> None:
        super().__init__()
        self.physics = physics
        self.config = config or OneShotConfig()
        self.image_size = image_size
        self.uses_diffractive_core = self.config.optical_core_arch == "diffractive_phase_only"
        self._validate_config()
        self.generator_e = ElectronicGeneratorE(
            image_size=image_size,
            grid_size=self.config.metasurface_grid_size,
            code_dim=self.config.code_dim,
            num_classes=self.config.num_classes,
            label_embed_dim=self.config.label_embed_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.slm = IdealAmplitudeSLMModel(
            self.config.slm_background_floor,
            self.config.slm_residual_phase_scale_rad,
        )
        d2nn = self._resolve_d2nn_params()
        if self.uses_diffractive_core:
            self.encoder = PhaseOnlyDiffractiveStack(
                d2nn["encoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
            )
            self.decoder = PhaseOnlyDiffractiveStack(
                d2nn["decoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
            )
        else:
            grid_features = 2 * self.config.metasurface_grid_size * self.config.metasurface_grid_size
            self.encoder = nn.Linear(grid_features, grid_features, bias=False)
            self.decoder = nn.Linear(grid_features, grid_features, bias=False)
        self._d2nn = d2nn
        self.front_path = FrontOpticalPath(physics)
        self.back_path = BackOpticalPath(physics, propagation_method="angular_spectrum")
        self._initialize_layers()

    def _resolve_d2nn_params(self) -> dict:
        """从 config.d2nn 解析所有 D2NN 参数（优先），否则回退到散落字段."""
        d2nn_cfg = self.config.d2nn
        if d2nn_cfg is not None:
            return {
                "wavelength": d2nn_cfg.wavelength,
                "pixel_size": d2nn_cfg.pixel_size,
                "grid_size": d2nn_cfg.grid_size,
                "propagation_grid_size": d2nn_cfg.propagation_grid_size,
                "encoder_layers": d2nn_cfg.encoder_layers,
                "decoder_layers": d2nn_cfg.decoder_layers,
                "layer_spacing": d2nn_cfg.layer_spacing,
                "phase_init": d2nn_cfg.phase_init,
            }
        return {
            "wavelength": self.physics.optics.wavelength,
            "pixel_size": self.physics.optics.pixel_size,
            "grid_size": self.config.diffractive_grid_size,
            "propagation_grid_size": self.config.diffractive_propagation_grid_size,
            "encoder_layers": self.config.encoder_diffractive_layers,
            "decoder_layers": self.config.decoder_diffractive_layers,
            "layer_spacing": self.config.diffractive_layer_spacing,
            "phase_init": self.config.diffractive_phase_init,
        }

    def _validate_config(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.config.code_dim <= 0:
            raise ValueError(f"one_shot.code_dim must be positive, got {self.config.code_dim}")
        if self.config.label_embed_dim <= 0:
            raise ValueError(
                f"one_shot.label_embed_dim must be positive, got {self.config.label_embed_dim}"
            )
        if self.config.hidden_dim <= 0:
            raise ValueError(f"one_shot.hidden_dim must be positive, got {self.config.hidden_dim}")
        if self.config.num_classes <= 0:
            raise ValueError(f"one_shot.num_classes must be positive, got {self.config.num_classes}")
        if self.config.ols_kind not in {"pointwise_tanh", "tanh", "gelu", "silu", "relu", "identity"}:
            raise ValueError(
                "Unsupported one_shot.ols_kind: "
                f"{self.config.ols_kind}; expected gelu, tanh, silu, relu, or identity."
            )
        if self.config.optical_core_arch not in {"dense_linear", "diffractive_phase_only"}:
            raise ValueError(
                "Unsupported one_shot.optical_core_arch: "
                f"{self.config.optical_core_arch}; expected dense_linear or diffractive_phase_only."
            )
        if self.config.metasurface_grid_size <= 0:
            raise ValueError(
                "one_shot.metasurface_grid_size must be positive, "
                f"got {self.config.metasurface_grid_size}"
            )
        if self.config.layer_spacing <= 0.0:
            raise ValueError(f"one_shot.layer_spacing must be positive, got {self.config.layer_spacing}")
        if self.config.d2nn is None and self.uses_diffractive_core:
            if self.config.diffractive_grid_size <= 0:
                raise ValueError(
                    "one_shot.diffractive_grid_size must be positive, "
                    f"got {self.config.diffractive_grid_size}"
                )
            if self.config.diffractive_propagation_grid_size < self.config.diffractive_grid_size:
                raise ValueError(
                    "one_shot.diffractive_propagation_grid_size must be at least "
                    "one_shot.diffractive_grid_size, got "
                    f"{self.config.diffractive_propagation_grid_size} and {self.config.diffractive_grid_size}"
                )
            if self.config.encoder_diffractive_layers <= 0:
                raise ValueError(
                    "one_shot.encoder_diffractive_layers must be positive, "
                    f"got {self.config.encoder_diffractive_layers}"
                )
            if self.config.decoder_diffractive_layers <= 0:
                raise ValueError(
                    "one_shot.decoder_diffractive_layers must be positive, "
                    f"got {self.config.decoder_diffractive_layers}"
                )
            if self.config.diffractive_layer_spacing <= 0.0:
                raise ValueError(
                    "one_shot.diffractive_layer_spacing must be positive, "
                    f"got {self.config.diffractive_layer_spacing}"
                )
            if self.config.diffractive_phase_init not in {"uniform_pi", "uniform_2pi", "small"}:
                raise ValueError(
                    "Unsupported one_shot.diffractive_phase_init: "
                    f"{self.config.diffractive_phase_init}; expected uniform_pi, uniform_2pi, or small."
                )

    def _initialize_layers(self) -> None:
        if not self.uses_diffractive_core:
            nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
            nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        slm_command = self.generator_e(code, class_labels)
        slm_outputs = self.slm(slm_command)
        if self.uses_diffractive_core:
            return self._forward_diffractive(slm_outputs)
        slm_transfer = slm_outputs["slm_transfer"]
        encoder_input_field = self.front_path.forward_field(
            slm_transfer,
            propagation_method="angular_spectrum",
        )
        encoder_grid = _resize_complex_field(
            encoder_input_field,
            (self.config.metasurface_grid_size, self.config.metasurface_grid_size),
        )
        encoded_vector = self.encoder(_flatten_complex_field(encoder_grid))
        encoded_field = _vector_to_complex_field(encoded_vector, self.config.metasurface_grid_size)
        ols_field = _apply_complex_ols(encoded_field, self.config.ols_kind)
        decoded_vector = self.decoder(_flatten_complex_field(ols_field))
        chip_grid = _vector_to_complex_field(decoded_vector, self.config.metasurface_grid_size)
        chip_field = _resize_complex_field(chip_grid, (self.image_size, self.image_size))
        optical_readout = self.back_path(chip_field)
        return {
            "x_hat": optical_readout["camera_plane"],
            "camera_plane": optical_readout["camera_plane"],
            "camera_plane_loss": optical_readout["camera_plane_loss"],
            "slm_command": slm_outputs["slm_command"],
            "slm_amplitude": slm_outputs["slm_amplitude"],
            "slm_phase": slm_outputs["slm_phase"],
            "encoder_input_amplitude": encoder_input_field.abs(),
            "encoder_input_phase": torch.angle(encoder_input_field),
            "encoder_input_grid_amplitude": encoder_grid.abs(),
            "encoder_input_grid_phase": torch.angle(encoder_grid),
            "encoded_amplitude": encoded_field.abs(),
            "encoded_phase": torch.angle(encoded_field),
            "ols_input_amplitude": encoded_field.abs(),
            "ols_input_phase": torch.angle(encoded_field),
            "ols_output_amplitude": ols_field.abs(),
            "ols_output_phase": torch.angle(ols_field),
            "ols_amplitude": ols_field.abs(),
            "ols_phase": torch.angle(ols_field),
            "decoder_input_amplitude": ols_field.abs(),
            "decoder_input_phase": torch.angle(ols_field),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "chip_real": chip_field.real,
            "chip_imag": chip_field.imag,
            **optical_readout,
        }

    def _forward_diffractive(self, slm_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d2nn = self._d2nn
        slm_transfer = slm_outputs["slm_transfer"]
        encoder_input_field = self.front_path.forward_field(
            slm_transfer,
            propagation_method="angular_spectrum",
        )
        encoder_grid = _resize_complex_field(
            encoder_input_field,
            (d2nn["grid_size"], d2nn["grid_size"]),
        )
        encoded_field, encoder_layer_fields, encoder_crop_diagnostics = self.encoder(encoder_grid)
        ols_field = _apply_complex_ols(encoded_field, self.config.ols_kind)
        chip_grid, decoder_layer_fields, decoder_crop_diagnostics = self.decoder(ols_field)
        raw_optical_readout = self.back_path(chip_grid)
        camera_intensity = _resize_real_intensity(
            raw_optical_readout["camera_intensity_raw"],
            (self.image_size, self.image_size),
        )
        camera_plane_loss = normalize_real_intensity(
            camera_intensity,
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        optical_readout = {
            **raw_optical_readout,
            "camera_plane": camera_plane_loss.clamp(0.0, 1.0),
            "camera_plane_loss": camera_plane_loss,
        }
        encoder_phase_transfer = self.encoder.phase_transfers()
        decoder_phase_transfer = self.decoder.phase_transfers()
        return {
            "x_hat": optical_readout["camera_plane"],
            "camera_plane": optical_readout["camera_plane"],
            "camera_plane_loss": optical_readout["camera_plane_loss"],
            "slm_command": slm_outputs["slm_command"],
            "slm_amplitude": slm_outputs["slm_amplitude"],
            "slm_phase": slm_outputs["slm_phase"],
            "encoder_input_amplitude": encoder_input_field.abs(),
            "encoder_input_phase": torch.angle(encoder_input_field),
            "encoder_input_grid_amplitude": encoder_grid.abs(),
            "encoder_input_grid_phase": torch.angle(encoder_grid),
            "encoded_amplitude": encoded_field.abs(),
            "encoded_phase": torch.angle(encoded_field),
            "ols_input_amplitude": encoded_field.abs(),
            "ols_input_phase": torch.angle(encoded_field),
            "ols_output_amplitude": ols_field.abs(),
            "ols_output_phase": torch.angle(ols_field),
            "ols_amplitude": ols_field.abs(),
            "ols_phase": torch.angle(ols_field),
            "decoder_input_amplitude": ols_field.abs(),
            "decoder_input_phase": torch.angle(ols_field),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "chip_real": chip_grid.real,
            "chip_imag": chip_grid.imag,
            "diffractive_camera_intensity_raw": raw_optical_readout["camera_intensity_raw"],
            "diffractive_camera_plane_raw": raw_optical_readout["camera_plane"],
            "diffractive_camera_plane_loss_raw": raw_optical_readout["camera_plane_loss"],
            "diffractive_encoder_layer_amplitude": encoder_layer_fields.abs(),
            "diffractive_encoder_layer_phase": torch.angle(encoder_layer_fields),
            "diffractive_decoder_layer_amplitude": decoder_layer_fields.abs(),
            "diffractive_decoder_layer_phase": torch.angle(decoder_layer_fields),
            "diffractive_encoder_mask_amplitude": encoder_phase_transfer.abs(),
            "diffractive_encoder_mask_phase": torch.angle(encoder_phase_transfer),
            "diffractive_decoder_mask_amplitude": decoder_phase_transfer.abs(),
            "diffractive_decoder_mask_phase": torch.angle(decoder_phase_transfer),
            **{
                f"diffractive_encoder_crop_{key}": value
                for key, value in encoder_crop_diagnostics.items()
            },
            **{
                f"diffractive_decoder_crop_{key}": value
                for key, value in decoder_crop_diagnostics.items()
            },
            **optical_readout,
        }


def validate_one_shot_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    model_version = payload.get("model_version")
    if model_version != ONE_SHOT_MODEL_VERSION:
        raise ValueError("Incompatible one-shot checkpoint: unexpected model_version.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Incompatible one-shot checkpoint: missing model_state_dict.")


def load_one_shot_checkpoint(model: OneShotOpticalGenerator, payload: Mapping[str, Any]) -> None:
    validate_one_shot_checkpoint_payload(payload)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError("Incompatible one-shot checkpoint: state dict does not match model config.") from exc
