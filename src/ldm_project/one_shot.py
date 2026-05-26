from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .config import OneShotConfig, PhysicsConfig
from .optics import BackOpticalPath, angular_spectrum_propagate


ONE_SHOT_MODEL_VERSION = "ldm_distilled_single_shot_optical_student_v1"


def _resize_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _flatten_complex_field(field: torch.Tensor) -> torch.Tensor:
    return torch.cat([field.real.flatten(1), field.imag.flatten(1)], dim=1)


def _vector_to_complex_field(vector: torch.Tensor, grid_size: int) -> torch.Tensor:
    batch_size = vector.shape[0]
    half_width = grid_size * grid_size
    real = vector[:, :half_width].reshape(batch_size, 1, grid_size, grid_size)
    imag = vector[:, half_width:].reshape(batch_size, 1, grid_size, grid_size)
    return torch.complex(real, imag)


def _apply_complex_ols(field: torch.Tensor, kind: str) -> torch.Tensor:
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
    elif kind == "identity":
        real = field.real
        imag = field.imag
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
        grid_features = 2 * self.config.metasurface_grid_size * self.config.metasurface_grid_size
        self.encoder = nn.Linear(grid_features, grid_features, bias=False)
        self.decoder = nn.Linear(grid_features, grid_features, bias=False)
        self.back_path = BackOpticalPath(physics, propagation_method="angular_spectrum")
        self._initialize_layers()

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
        if self.config.metasurface_grid_size <= 0:
            raise ValueError(
                "one_shot.metasurface_grid_size must be positive, "
                f"got {self.config.metasurface_grid_size}"
            )
        if self.config.layer_spacing <= 0.0:
            raise ValueError(f"one_shot.layer_spacing must be positive, got {self.config.layer_spacing}")

    def _initialize_layers(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        slm_command = self.generator_e(code, class_labels)
        slm_outputs = self.slm(slm_command)
        slm_transfer = slm_outputs["slm_transfer"]
        incoming_field = torch.ones_like(slm_transfer, dtype=torch.complex64)
        slm_field = incoming_field * slm_transfer.to(incoming_field.dtype)
        encoder_input_field = angular_spectrum_propagate(
            slm_field,
            self.config.layer_spacing,
            self.physics.optics.wavelength,
            self.physics.optics.pixel_size,
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
