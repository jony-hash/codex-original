"""Fixed formal single-exposure optical student.

Only the validated physical chain is implemented here.  Compatibility and
intervention behavior belongs to student.experimental.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..config import OneShotConfig, PhysicsConfig
from ..optics import (
    BackOpticalPath,
    FrontOpticalPath,
    IdealAmplitudeSLMModel,
    PhaseOnlyDiffractiveStack,
    apply_identity_ols,
    normalize_real_intensity,
)
from .contracts import ONE_SHOT_MODEL_VERSION
from .formal_contract import validate_formal_student_config
from .generator import ElectronicGeneratorE
from .output_contract import build_formal_optical_fields

__all__ = ["FormalOneShotOpticalGenerator", "ONE_SHOT_MODEL_VERSION"]


class FormalOneShotOpticalGenerator(nn.Module):
    """The formal amplitude-SLM to CMOS path, without diagnostic interventions."""

    def __init__(
        self,
        physics: PhysicsConfig,
        config: OneShotConfig | None = None,
        image_size: int = 200,
    ) -> None:
        super().__init__()
        self.physics = physics
        self.config = config or OneShotConfig()
        validate_formal_student_config(physics, self.config)
        if self.config.d2nn is None:
            raise ValueError("formal student requires one_shot.d2nn")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = image_size
        d2nn = self.config.d2nn
        self._d2nn = {
            "wavelength": d2nn.wavelength,
            "pixel_size": d2nn.pixel_size,
            "grid_size": d2nn.grid_size,
            "propagation_grid_size": d2nn.propagation_grid_size,
            "encoder_layers": d2nn.encoder_layers,
            "decoder_layers": d2nn.decoder_layers,
            "layer_spacing": d2nn.layer_spacing,
            "phase_init": d2nn.phase_init,
        }
        self.generator_e = ElectronicGeneratorE(
            image_size=image_size,
            grid_size=self.config.metasurface_grid_size,
            code_dim=self.config.code_dim,
            num_classes=self.config.num_classes,
            label_embed_dim=self.config.label_embed_dim,
            hidden_dim=self.config.hidden_dim,
            arch=self.config.generator_e_arch,
            code_residual_scale=self.config.code_residual_scale,
            command_transform=self.config.generator_command_transform,
            background_floor=self.config.slm_background_floor,
            template_init=self.config.generator_template_init,
        )
        self.slm = IdealAmplitudeSLMModel(
            self.config.slm_background_floor,
            self.config.slm_residual_phase_scale_rad,
        )
        self.encoder = PhaseOnlyDiffractiveStack(
            d2nn.encoder_layers,
            d2nn.grid_size,
            d2nn.layer_spacing,
            d2nn.wavelength,
            d2nn.pixel_size,
            d2nn.propagation_grid_size,
            d2nn.phase_init,
        )
        self.decoder = PhaseOnlyDiffractiveStack(
            d2nn.decoder_layers,
            d2nn.grid_size,
            d2nn.layer_spacing,
            d2nn.wavelength,
            d2nn.pixel_size,
            d2nn.propagation_grid_size,
            d2nn.phase_init,
        )
        self.front_path = FrontOpticalPath(physics)
        self.back_path = BackOpticalPath(physics, propagation_method="angular_spectrum")

    @staticmethod
    def _resize_real_intensity(
        intensity: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        if tuple(intensity.shape[-2:]) == size:
            return intensity
        return F.interpolate(intensity, size=size, mode="area")

    def forward(
        self,
        code: torch.Tensor,
        class_labels: torch.Tensor,
        *,
        return_optical_fields: bool = False,
    ) -> dict[str, torch.Tensor]:
        slm_command, generator_diagnostics = self.generator_e.forward_with_diagnostics(
            code, class_labels
        )
        generator_diagnostics = {
            **generator_diagnostics,
            "diagnostic_slm_override_active": slm_command.new_tensor(False, dtype=torch.bool),
            "diagnostic_slm_override_amplitude": slm_command.new_tensor(0.0),
        }
        slm_outputs = self.slm(slm_command)
        d2nn = self._d2nn
        slm_transfer = slm_outputs["slm_transfer"]
        front_field, front_path_diagnostics = (
            self.front_path.forward_field_with_diagnostics(
                slm_transfer,
                propagation_method="angular_spectrum",
            )
        )
        front_path_diagnostics = dict(front_path_diagnostics)
        slm_field = front_path_diagnostics.pop("_slm_field", slm_transfer)
        encoder_input_field = front_field
        encoder_grid = encoder_input_field
        encoded_field, encoder_layer_fields, encoder_crop_diagnostics = self.encoder(
            encoder_grid
        )
        ols_field = apply_identity_ols(encoded_field)
        decoder_input_field = ols_field
        target_code_amplitude = ols_field.abs()
        chip_grid, decoder_layer_fields, decoder_crop_diagnostics = self.decoder(
            decoder_input_field
        )
        raw_optical_readout = self.back_path(
            chip_grid,
            input_pixel_size=d2nn["pixel_size"],
            output_size=(self.image_size, self.image_size),
            output_pixel_size=self.physics.optics.pixel_size,
        )
        camera_intensity = raw_optical_readout["camera_intensity_raw"]
        if tuple(camera_intensity.shape[-2:]) != (self.image_size, self.image_size):
            camera_intensity = self._resize_real_intensity(
                camera_intensity, (self.image_size, self.image_size)
            )
        camera_plane_loss = normalize_real_intensity(
            camera_intensity,
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        physical_camera_plane = camera_plane_loss.clamp(0.0, 1.0)
        chip_amplitude = chip_grid.abs()
        chip_intensity_loss = normalize_real_intensity(
            chip_amplitude.pow(2),
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        chip_amplitude_loss = normalize_real_intensity(
            chip_amplitude,
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        probe_plane_loss = raw_optical_readout["probe_plane_loss"]
        primary_plane = self.physics.readout.primary_plane
        if primary_plane == "camera":
            selected_plane_loss = camera_plane_loss
        elif primary_plane == "probe":
            selected_plane_loss = probe_plane_loss
        elif primary_plane == "chip_amplitude":
            selected_plane_loss = chip_amplitude_loss
        elif primary_plane == "chip_intensity":
            selected_plane_loss = chip_intensity_loss
        else:
            raise ValueError(
                "Unsupported physics.readout.primary_plane: "
                f"{primary_plane}; expected camera, probe, chip_amplitude, or chip_intensity."
            )
        selected_native_plane_loss = selected_plane_loss
        selected_native_plane = selected_native_plane_loss.clamp(0.0, 1.0)
        selected_output_size = (self.image_size, self.image_size)
        selected_plane_loss = self._resize_real_intensity(
            selected_native_plane_loss, selected_output_size
        )
        selected_plane = selected_plane_loss.clamp(0.0, 1.0)
        optical_readout = {
            **raw_optical_readout,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
        }
        encoder_phase_transfer = self.encoder.phase_transfers()
        decoder_phase_transfer = self.decoder.phase_transfers()
        outputs = {
            "x_hat": selected_plane,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
            "primary_readout_plane": primary_plane,
            "primary_readout_plane_native": selected_native_plane,
            "primary_readout_plane_loss_native": selected_native_plane_loss,
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
            "decoder_input_amplitude": decoder_input_field.abs(),
            "decoder_input_phase": torch.angle(decoder_input_field),
            "target_code_amplitude": target_code_amplitude,
            "target_code_diagnostic_mode": "none",
            "target_code_spatial_scale": None,
            "target_code_diagnostic_active": torch.tensor(False, device=chip_grid.device),
            "target_code_bridge_source": "none",
            "target_code_bridge_arch": "shallow",
            "target_code_bridge_active": torch.tensor(False, device=chip_grid.device),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "decoder_diagnostic_mode": "normal",
            "decoder_diagnostic_modulation_override": "none",
            "decoder_diagnostic_mask_component": "trained",
            "decoder_diagnostic_phase_layer_mask": (),
            "decoder_diagnostic_phase_layer_mask_applied": (True,) * int(d2nn["decoder_layers"]),
            "decoder_bypass_active": torch.tensor(False, device=chip_grid.device),
            "decoder_propagation_only_active": torch.tensor(False, device=chip_grid.device),
            "ols_transfer_calibration": "none",
            "chip_real": chip_grid.real,
            "chip_imag": chip_grid.imag,
            "chip_intensity_primary_loss": chip_intensity_loss,
            "chip_intensity_primary": chip_intensity_loss.clamp(0.0, 1.0),
            "chip_amplitude_primary_loss": chip_amplitude_loss,
            "chip_amplitude_primary": chip_amplitude_loss.clamp(0.0, 1.0),
            "physical_camera_plane": physical_camera_plane,
            "physical_camera_plane_loss": camera_plane_loss,
            "physical_camera_intensity_raw": raw_optical_readout[
                "camera_intensity_raw"
            ],
            "front_path_diagnostics": front_path_diagnostics,
            "diffractive_camera_intensity_raw": raw_optical_readout[
                "camera_intensity_raw"
            ],
            "diffractive_camera_plane_raw": raw_optical_readout["camera_plane"],
            "diffractive_camera_plane_loss_raw": raw_optical_readout[
                "camera_plane_loss"
            ],
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
        if return_optical_fields:
            outputs["optical_fields"] = build_formal_optical_fields(
                physics=self.physics,
                d2nn=self._d2nn,
                slm_transfer=slm_transfer,
                slm_field=slm_field,
                front_field=front_field,
                front_path_diagnostics=front_path_diagnostics,
                encoder_input_field=encoder_input_field,
                encoder_grid=encoder_grid,
                encoder_layer_fields=encoder_layer_fields,
                encoded_field=encoded_field,
                ols_field=ols_field,
                decoder_input_field=decoder_input_field,
                decoder_layer_fields=decoder_layer_fields,
                chip_grid=chip_grid,
                raw_optical_readout=raw_optical_readout,
            )
        return {**outputs, **generator_diagnostics}
