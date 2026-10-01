"""Isolated LightGen-inspired architecture proxy; it cannot alter formal v2 optics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import LightGenSystemConfig
from ..contracts import AmplitudeCommand, ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from .d2nn import FiniteCropPhaseStack
from .devices import IdealAmplitudeSLM
from .fiber_ols import FiberModeBankV2, FiberOLSStateV2, gaussian_single_mode_bank, project_single_mode_fibers, reemit_single_mode_fibers


@dataclass(frozen=True)
class LightGenForwardResultV2:
    raw_camera: IntensityFrame
    ols: FiberOLSStateV2
    optical_trace: tuple[ComplexField, ...]
    crop_statistics: dict[str, torch.Tensor]


class LightGenAmplitudeFiberOLSSystemV2(nn.Module):
    """2 phase layers -> 100 complex fiber OLS modes -> 4 phase layers architecture proxy."""

    def __init__(self, config: LightGenSystemConfig) -> None:
        super().__init__()
        self.config = config
        shape = config.slm_shape
        pitch = (config.slm_pixel_pitch_m,) * 2
        wavelength = config.proxy_wavelength_m
        self.slm_plane = PlaneSpec("lightgen_slm", shape, pitch, wavelength, 0.0, "lightgen_proxy_slm")
        self.ols_plane = PlaneSpec("lightgen_fiber_input", shape, pitch, wavelength, config.encoder_layers * config.proxy_layer_spacing_m, "lightgen_proxy_fiber")
        self.camera_plane = PlaneSpec("lightgen_raw_sensor", shape, pitch, wavelength, (config.encoder_layers + config.decoder_layers) * config.proxy_layer_spacing_m, "lightgen_proxy_sensor")
        self.slm = IdealAmplitudeSLM()
        self.encoder = FiniteCropPhaseStack(layers=2, active_shape=shape, propagation_shape=(config.propagation_grid_size,) * 2, spacing_m=config.proxy_layer_spacing_m)
        self.decoder = FiniteCropPhaseStack(layers=4, active_shape=shape, propagation_shape=(config.propagation_grid_size,) * 2, spacing_m=config.proxy_layer_spacing_m)
        # Stored buffers make mode coordinates and coupling calibration auditable.
        bank = gaussian_single_mode_bank(self.ols_plane, waist_m=config.fiber_mode_waist_m, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("fiber_modes", bank.modes)
        self.register_buffer("fiber_centers_m", bank.centers_m)

    def _bank(self, reference: torch.Tensor) -> FiberModeBankV2:
        return FiberModeBankV2(self.fiber_modes.to(reference), self.fiber_centers_m.to(reference.real), self.config.fiber_mode_waist_m, self.ols_plane)

    def command(self, seed: torch.Tensor) -> AmplitudeCommand:
        return AmplitudeCommand(seed, self.slm_plane, IdealAmplitudeSLM.calibration_id)

    def _decode_ols(
        self,
        ols: FiberOLSStateV2,
        *,
        input_trace: tuple[ComplexField, ...],
        encoder_crop_retention: torch.Tensor | None,
    ) -> LightGenForwardResultV2:
        emitted = reemit_single_mode_fibers(ols, plane_id="lightgen_fiber_reemission")
        decoded, dec_records, dec_stats = self.decoder(emitted, "lightgen_generator")
        camera = ComplexField(decoded.tensor, self.camera_plane, OperationProvenance("lightgen_proxy_raw_sensor", decoded.plane.plane_id, 0.0, {"physical_parameter_status": "unknown", "tier": "architecture-proxy"}))
        raw = IntensityFrame(camera.tensor.abs().square(), self.camera_plane, camera.provenance)
        crop_statistics = {"decoder_crop_retention": dec_stats["crop_retention"]}
        if encoder_crop_retention is not None:
            crop_statistics["encoder_crop_retention"] = encoder_crop_retention
        return LightGenForwardResultV2(raw, ols, (*input_trace, emitted, *dec_records, camera), crop_statistics)

    def forward(self, command: AmplitudeCommand) -> LightGenForwardResultV2:
        field = self.slm(command)
        encoded, enc_records, enc_stats = self.encoder(field, "lightgen_encoder")
        encoded = ComplexField(encoded.tensor, self.ols_plane, OperationProvenance("lightgen_encoder_to_fiber", encoded.plane.plane_id, 0.0, {"physical_parameter_status": "unknown", "tier": "architecture-proxy"}))
        ols = project_single_mode_fibers(encoded, self._bank(encoded.tensor))
        return self._decode_ols(ols, input_trace=(field, *enc_records, encoded), encoder_crop_retention=enc_stats["crop_retention"])

    def forward_from_ols(self, coefficients: torch.Tensor) -> LightGenForwardResultV2:
        """Render diagnostic complex OLS coefficients through the shared generator."""
        if not isinstance(coefficients, torch.Tensor) or not torch.is_complex(coefficients):
            raise TypeError("external OLS coefficients must be a complex tensor")
        if coefficients.ndim != 2 or coefficients.shape[1] != 100:
            raise ValueError("external OLS coefficients require shape [batch, 100]")
        if coefficients.dtype != self.fiber_modes.dtype or coefficients.device != self.fiber_modes.device:
            raise ValueError("external OLS coefficients must match the model's complex dtype and device")
        ols = FiberOLSStateV2(
            coefficients,
            self._bank(coefficients),
            self.ols_plane,
            OperationProvenance("external_ols_coefficients", None, None, {"physical_parameter_status": "unknown", "tier": "architecture-proxy"}),
        )
        ols.validate()
        return self._decode_ols(ols, input_trace=(), encoder_crop_retention=None)
