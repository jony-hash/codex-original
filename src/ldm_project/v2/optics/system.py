"""The formal amplitude-SLM -> CMOS system with structured results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from ..config import FormalSystemConfig
from ..contracts import (
    AmplitudeCommand,
    ComplexField,
    DisplayFrame,
    ExposureCalibration,
    IntensityFrame,
    OperationProvenance,
    PlaneSpec,
    render_fixed_exposure,
)
from .d2nn import FiniteCropPhaseStack, NearConjugateFiniteCropPhaseStack
from .devices import IdealAmplitudeSLM, IdentityOLS
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
        self.slm_plane = PlaneSpec(
            "slm_output",
            config.slm_shape,
            (config.slm_pixel_pitch_m,) * 2,
            wavelength,
            0.0,
            "slm",
        )
        front_segments = (0.025, 0.125, 0.0732142857142857, 0.3125, 0.0892857142857143)
        back_segments = (0.100, 0.250)
        encoder_z = sum(front_segments)
        near_conjugate = config.kind in {
            "diagnostic_amplitude_near_conjugate_relay_identity_v1",
            "formal_amplitude_near_conjugate_relay_identity_v1",
            "diagnostic_amplitude_near_conjugate_relay_identity_phase_capacity_v7_2",
            "diagnostic_amplitude_near_conjugate_relay_identity_evaluation_qualified_v1",
            "diagnostic_amplitude_near_conjugate_relay_identity_evaluation_qualified_v1_1",
        }
        internal_distance = (config.encoder_layers + config.decoder_layers) * (
            99.75e-3 if near_conjugate else config.layer_spacing_m
        )
        chip_z = encoder_z + internal_distance
        camera_z = chip_z + sum(back_segments)
        self.encoder_plane = PlaneSpec(
            "encoder_input",
            config.metasurface_shape,
            (config.metasurface_pixel_pitch_m,) * 2,
            wavelength,
            encoder_z,
            "encoder",
        )
        self.chip_plane = PlaneSpec(
            "chip",
            config.metasurface_shape,
            (config.metasurface_pixel_pitch_m,) * 2,
            wavelength,
            chip_z,
            "chip",
        )
        self.camera_plane = PlaneSpec(
            "camera",
            config.slm_shape,
            (config.slm_pixel_pitch_m,) * 2,
            wavelength,
            camera_z,
            "camera",
        )
        self.slm = IdealAmplitudeSLM()
        self.ols = IdentityOLS()
        self.illumination_intensity_transmission = 0.80 if near_conjugate else 1.0
        self.front = CoherentImagingRelay(
            self.slm_plane,
            self.encoder_plane,
            config.front_magnification,
            name="front_4f",
            segments_m=front_segments,
            matrix_segments_m=(
                sum(front_segments[:3]),
                front_segments[3],
                front_segments[4],
            ),
            focal_lengths_m=config.front_lens_focal_lengths_m,
            clear_aperture_m=config.front_clear_aperture_m,
        )
        stack = (
            NearConjugateFiniteCropPhaseStack
            if near_conjugate
            else FiniteCropPhaseStack
        )
        if near_conjugate:
            self.encoder = stack(
                layers=config.encoder_layers,
                active_shape=config.metasurface_shape,
                propagation_shape=(config.propagation_grid_size,) * 2,
                input_plane=self.encoder_plane,
                prefix="encoder",
            )
            self.decoder = stack(
                layers=config.decoder_layers,
                active_shape=config.metasurface_shape,
                propagation_shape=(config.propagation_grid_size,) * 2,
                input_plane=self.encoder_plane,
                prefix="decoder",
            )
        else:
            self.encoder = stack(
                layers=config.encoder_layers,
                active_shape=config.metasurface_shape,
                propagation_shape=(config.propagation_grid_size,) * 2,
                spacing_m=config.layer_spacing_m,
            )
            self.decoder = stack(
                layers=config.decoder_layers,
                active_shape=config.metasurface_shape,
                propagation_shape=(config.propagation_grid_size,) * 2,
                spacing_m=config.layer_spacing_m,
            )
        self.back = CoherentImagingRelay(
            self.chip_plane,
            self.camera_plane,
            config.back_magnification,
            name="back_b0",
            segments_m=back_segments,
            focal_lengths_m=(config.back_lens_focal_length_m,),
            clear_aperture_m=config.back_clear_aperture_m,
        )

    def command(self, tensor: torch.Tensor) -> AmplitudeCommand:
        return AmplitudeCommand(
            tensor, self.slm_plane, IdealAmplitudeSLM.calibration_id
        )

    def forward(
        self,
        command: AmplitudeCommand,
        *,
        exposure: ExposureCalibration | None = None,
        return_optical_trace: bool = False,
    ) -> FormalForwardResult:
        slm_field = self.slm(command)
        if self.illumination_intensity_transmission != 1.0:
            slm_field = ComplexField(
                slm_field.tensor * self.illumination_intensity_transmission**0.5,
                slm_field.plane,
                OperationProvenance(
                    "global_illumination_attenuation",
                    slm_field.plane.plane_id,
                    None,
                    {
                        "intensity_transmission": self.illumination_intensity_transmission,
                        "field_amplitude_factor": self.illumination_intensity_transmission
                        ** 0.5,
                        "per_sample": False,
                    },
                ),
            )
        front = self.front(slm_field)
        encoded, encoder_records, encoder_stats = self.encoder(front, "encoder")
        ols = self.ols(encoded)
        decoded, decoder_records, decoder_stats = self.decoder(ols, "decoder")
        chip = ComplexField(decoded.tensor, self.chip_plane, decoded.provenance)
        camera_field = self.back(chip)
        front_source_retention = self.front.last_diagnostic.get("source_pupil_retention")
        front_target_retention = self.front.last_diagnostic.get("target_pupil_retention")
        back_source_retention = self.back.last_diagnostic.get("source_pupil_retention")
        back_target_retention = self.back.last_diagnostic.get("target_pupil_retention")
        relay_stats = {
            "front_source_pupil_retention": front_source_retention,
            "front_target_pupil_retention": front_target_retention,
            "back_source_pupil_retention": back_source_retention,
            "back_target_pupil_retention": back_target_retention,
        }
        for name, value in relay_stats.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1.0 + 1.0e-4).any():
                raise ValueError(f"formal relay pupil retention is invalid: {name}")
        raw_camera = IntensityFrame(
            camera_field.tensor.abs().square(),
            self.camera_plane,
            camera_field.provenance,
            exposure.calibration_id if exposure else None,
        )
        display = (
            render_fixed_exposure(raw_camera, exposure)
            if exposure is not None
            else None
        )
        trace = (
            (slm_field, front, *encoder_records, ols, *decoder_records, camera_field)
            if return_optical_trace
            else None
        )
        return FormalForwardResult(
            raw_camera=raw_camera,
            loss_camera=raw_camera,
            display_camera=display,
            phase_masks={"encoder": self.encoder.phase, "decoder": self.decoder.phase},
            crop_statistics={
                "encoder_crop_retention": encoder_stats["crop_retention"],
                "decoder_crop_retention": decoder_stats["crop_retention"],
                **relay_stats,
            },
            optical_trace=trace,
        )

    def forward_from_generator(
        self,
        generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        code: torch.Tensor,
        labels: torch.Tensor,
        *,
        exposure: ExposureCalibration | None = None,
        return_optical_trace: bool = False,
    ) -> FormalForwardResult:
        return self.forward(
            self.command(generator(code, labels)),
            exposure=exposure,
            return_optical_trace=return_optical_trace,
        )
