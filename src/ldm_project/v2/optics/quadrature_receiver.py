"""Fixed single-exposure four-port quadrature receiver.

The receiver is a deterministic physical readout boundary.  It performs no
learned reconstruction and contains no trainable parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import ComplexField, IntensityFrame, OperationProvenance, PlaneSpec


@dataclass(frozen=True)
class QuadratureReadout:
    r_plus: torch.Tensor
    r_minus: torch.Tensor
    j_plus: torch.Tensor
    j_minus: torch.Tensor
    packed: IntensityFrame
    r_difference: torch.Tensor
    j_difference: torch.Tensor


class BalancedFourPortReceiver(nn.Module):
    """Ideal balanced R/J homodyne receiver packed into one CMOS exposure."""

    receiver_kind = "single_exposure_balanced_four_port_v1"
    sensor_shape = (400, 400)
    port_shape = (200, 200)
    pixel_pitch_m = 2.5e-6

    def __init__(
        self,
        *,
        exposure: float = 1.0,
        lo_amplitude: float = 1.0,
        beamsplitter_amplitude_factor: float = 0.5,
        r_calibration_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if exposure <= 0.0 or lo_amplitude <= 0.0 or r_calibration_scale <= 0.0:
            raise ValueError("quadrature receiver calibration values must be positive")
        if beamsplitter_amplitude_factor != 0.5:
            raise ValueError("quadrature receiver locks a 0.5 amplitude splitter factor")
        self.exposure = float(exposure)
        self.lo_amplitude = float(lo_amplitude)
        self.beamsplitter_amplitude_factor = float(beamsplitter_amplitude_factor)
        self.r_calibration_scale = float(r_calibration_scale)

    def _sensor_plane(self, field: ComplexField) -> PlaneSpec:
        return PlaneSpec(
            "quadrature_cmos_400x400",
            self.sensor_shape,
            (self.pixel_pitch_m, self.pixel_pitch_m),
            field.plane.wavelength_m,
            field.plane.z_m,
            "quadrature_receiver",
        )

    def forward(self, field: ComplexField) -> QuadratureReadout:
        if tuple(field.tensor.shape[-2:]) != self.port_shape:
            raise ValueError("quadrature receiver requires a 200x200 camera field")
        if not torch.is_complex(field.tensor) or not torch.isfinite(field.tensor).all():
            raise ValueError("quadrature receiver requires a finite complex camera field")
        lo = torch.full_like(field.tensor, complex(self.lo_amplitude, 0.0))
        factor = self.exposure * self.beamsplitter_amplitude_factor**2
        r_plus = factor * (field.tensor + lo).abs().square()
        r_minus = factor * (field.tensor - lo).abs().square()
        j_plus = factor * (field.tensor + 1j * lo).abs().square()
        j_minus = factor * (field.tensor - 1j * lo).abs().square()
        packed = torch.cat(
            (
                torch.cat((r_plus, r_minus), dim=-1),
                torch.cat((j_plus, j_minus), dim=-1),
            ),
            dim=-2,
        )
        if tuple(packed.shape[-2:]) != self.sensor_shape:
            raise RuntimeError("quadrature receiver produced an invalid packed sensor shape")
        provenance = OperationProvenance(
            self.receiver_kind,
            field.plane.plane_id,
            None,
            {
                "operation": "single_exposure_balanced_quadrature",
                "port_shape": list(self.port_shape),
                "sensor_shape": list(self.sensor_shape),
                "pixel_pitch_m": self.pixel_pitch_m,
                "reference_phase_radians": 0.0,
                "reference_phase_j_radians": 1.5707963267948966,
                "upstream_plane": field.plane.plane_id,
            },
        )
        packed_frame = IntensityFrame(packed, self._sensor_plane(field), provenance)
        return QuadratureReadout(
            r_plus,
            r_minus,
            j_plus,
            j_minus,
            packed_frame,
            r_plus - r_minus,
            j_plus - j_minus,
        )

    def formal_view(self, readout: QuadratureReadout, *, output_size: int = 28) -> torch.Tensor:
        if output_size <= 0:
            raise ValueError("quadrature formal view output_size must be positive")
        normalized = readout.r_difference / self.r_calibration_scale
        view = (0.5 + 0.5 * normalized).clamp(0.0, 1.0)
        view = view[..., : self.port_shape[0], : self.port_shape[1]]
        return torch.nn.functional.adaptive_avg_pool2d(view, (output_size, output_size))

    def code_measurement(self, readout: QuadratureReadout) -> torch.Tensor:
        return readout.r_difference / self.r_calibration_scale

    def energy_identity(self, field: ComplexField, readout: QuadratureReadout) -> dict[str, torch.Tensor]:
        lo_intensity = self.lo_amplitude**2
        expected = 2.0 * self.exposure * self.beamsplitter_amplitude_factor**2 * (
            field.tensor.abs().square() + lo_intensity
        )
        return {
            "r": (readout.r_plus + readout.r_minus) - expected,
            "j": (readout.j_plus + readout.j_minus) - expected,
        }

    @staticmethod
    def sign_inversion(readout: QuadratureReadout) -> tuple[torch.Tensor, torch.Tensor]:
        return readout.r_minus - readout.r_plus, readout.j_minus - readout.j_plus

    @staticmethod
    def quadrature_swap(readout: QuadratureReadout) -> tuple[torch.Tensor, torch.Tensor]:
        return readout.j_difference, -readout.r_difference
