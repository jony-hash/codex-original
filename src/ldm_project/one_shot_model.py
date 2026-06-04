from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .config import OneShotConfig, PhysicsConfig
from .one_shot_optical_layers import (
    ElectronicGeneratorE,
    IdealAmplitudeSLMModel,
    PhaseOnlyDiffractiveStack,
    _apply_complex_ols,
    _flatten_complex_field,
    _resize_complex_field,
    _resize_real_intensity,
    _vector_to_complex_field,
)
from .optics import BackOpticalPath, FrontOpticalPath, normalize_real_intensity


ONE_SHOT_MODEL_VERSION = "ldm_distilled_single_shot_optical_student_v3"


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
