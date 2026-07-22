"""The formal amplitude-SLM -> CMOS system with structured results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from ..config import FormalSystemConfig
from ..contracts import AmplitudeCommand, ComplexField, DisplayFrame, ExposureCalibration, IntensityFrame, PlaneSpec, render_fixed_exposure
from .d2nn import FiniteCropPhaseStack
from .devices import IdealAmplitudeSLM, identity_ols
from .relays import CoherentImagingRelay


@dataclass(frozen=True)
class FormalForwardResult:
    raw_camera: IntensityFrame
    loss_camera: IntensityFrame
    display_camera: DisplayFrame | None
    phase_masks: dict[str, torch.Tensor]
    crop_statistics: dict[str, torch.Tensor]
    optical_trace: tuple[ComplexField, ...] | None


class FormalAmplitudeFiniteCropIdentitySystem(nn.Module):
    """Only v2 formal system; transparent and compatibility paths do not exist here."""

    def __init__(self, config: FormalSystemConfig) -> None:
        super().__init__()
        self.config = config
        wavelength = config.wavelength_m
        self.slm_plane = PlaneSpec("slm_output", config.slm_shape, (config.slm_pixel_pitch_m,) * 2, wavelength, 0.0, "slm")
        front_segments = (0.025, 0.125, 0.0732142857142857, 0.3125, 0.0892857142857143)
        back_segments = (0.100, 0.250)
        encoder_z = sum(front_segments)
        chip_z = encoder_z + 4 * config.layer_spacing_m
        camera_z = chip_z + sum(back_segments)
        self.encoder_plane = PlaneSpec("encoder_input", config.metasurface_shape, (config.metasurface_pixel_pitch_m,) * 2, wavelength, encoder_z, "encoder")
        self.chip_plane = PlaneSpec("chip", config.metasurface_shape, (config.metasurface_pixel_pitch_m,) * 2, wavelength, chip_z, "chip")
        self.camera_plane = PlaneSpec("camera", config.slm_shape, (config.slm_pixel_pitch_m,) * 2, wavelength, camera_z, "camera")
        self.slm = IdealAmplitudeSLM()
        self.front = CoherentImagingRelay(self.slm_plane, self.encoder_plane, config.front_magnification, name="front_4f", segments_m=front_segments, matrix_segments_m=(sum(front_segments[:3]), front_segments[3], front_segments[4]), focal_lengths_m=config.front_lens_focal_lengths_m, clear_aperture_m=config.front_clear_aperture_m)
        self.encoder = FiniteCropPhaseStack(layers=config.encoder_layers, active_shape=config.metasurface_shape, propagation_shape=(config.propagation_grid_size,) * 2, spacing_m=config.layer_spacing_m)
        self.decoder = FiniteCropPhaseStack(layers=config.decoder_layers, active_shape=config.metasurface_shape, propagation_shape=(config.propagation_grid_size,) * 2, spacing_m=config.layer_spacing_m)
        self.back = CoherentImagingRelay(self.chip_plane, self.camera_plane, config.back_magnification, name="back_b0", segments_m=back_segments, focal_lengths_m=(config.back_lens_focal_length_m,), clear_aperture_m=config.back_clear_aperture_m)

    def command(self, tensor: torch.Tensor) -> AmplitudeCommand:
        return AmplitudeCommand(tensor, self.slm_plane, IdealAmplitudeSLM.calibration_id)

    def forward(self, command: AmplitudeCommand, *, exposure: ExposureCalibration | None = None, return_optical_trace: bool = False) -> FormalForwardResult:
        slm_field = self.slm(command)
        front = self.front(slm_field)
        encoded, encoder_records, encoder_stats = self.encoder(front, "encoder")
        ols = identity_ols(encoded)
        decoded, decoder_records, decoder_stats = self.decoder(ols, "decoder")
        chip = ComplexField(decoded.tensor, self.chip_plane, decoded.provenance)
        camera_field = self.back(chip)
        raw_camera = IntensityFrame(camera_field.tensor.abs().square(), self.camera_plane, camera_field.provenance, exposure.calibration_id if exposure else None)
        display = render_fixed_exposure(raw_camera, exposure) if exposure is not None else None
        trace = (slm_field, front, *encoder_records, ols, *decoder_records, camera_field) if return_optical_trace else None
        return FormalForwardResult(
            raw_camera=raw_camera,
            loss_camera=raw_camera,
            display_camera=display,
            phase_masks={"encoder": self.encoder.phase, "decoder": self.decoder.phase},
            crop_statistics={"encoder_crop_retention": encoder_stats["crop_retention"], "decoder_crop_retention": decoder_stats["crop_retention"]},
            optical_trace=trace,
        )

    def forward_from_generator(self, generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], code: torch.Tensor, labels: torch.Tensor, *, exposure: ExposureCalibration | None = None, return_optical_trace: bool = False) -> FormalForwardResult:
        return self.forward(self.command(generator(code, labels)), exposure=exposure, return_optical_trace=return_optical_trace)
