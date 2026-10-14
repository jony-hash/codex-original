"""Unit-explicit passive material response for the LightGen research path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


MATERIAL_MODELS = (
    "identity",
    "saturable_absorption",
    "saturable_absorption_plus_RSA",
    "amplitude_plus_phase_nonlinearity",
)


@dataclass(frozen=True)
class MaterialParameters:
    model: str = "identity"
    t_ns: float = 1.0
    delta_t: float = 0.0
    i_sat_w_cm2: float | None = None
    i_rsa_w_cm2: float | None = None
    phi_max_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in MATERIAL_MODELS:
            raise ValueError("unsupported material model")
        if not all(math.isfinite(value) for value in (self.t_ns, self.delta_t, self.phi_max_rad)):
            raise ValueError("material transmission parameters must be finite")
        if self.t_ns < 0.0 or self.delta_t < 0.0 or self.t_ns + self.delta_t > 1.0:
            raise ValueError("passive SA requires 0 <= T_ns, DeltaT and T_ns+DeltaT <= 1")
        for name, value in (("I_sat", self.i_sat_w_cm2), ("I_RSA", self.i_rsa_w_cm2)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite positive W/cm^2")
        if self.model != "identity" and self.i_sat_w_cm2 is None:
            raise ValueError("non-identity scenarios require explicit I_sat in W/cm^2")
        if self.model == "saturable_absorption_plus_RSA" and self.i_rsa_w_cm2 is None:
            raise ValueError("SA+RSA requires explicit I_RSA in W/cm^2")


class MaterialNonlinearity:
    """Apply a static passive material response using supplied W/cm^2.

    The SA equation is ``T=T_ns+DeltaT*I/(I+I_sat)``. Optional diagnostic
    scenarios add a phenomenological RSA denominator or phase response.
    ``T_ns`` is low-intensity power transmittance, not nonsaturable loss.
    """

    def forward(
        self,
        field: torch.Tensor,
        intensity_w_cm2: torch.Tensor,
        params: MaterialParameters,
    ) -> torch.Tensor:
        if not torch.is_complex(field) or not torch.isfinite(field).all():
            raise ValueError("material input field must be finite complex values")
        if (
            not torch.is_floating_point(intensity_w_cm2)
            or not torch.isfinite(intensity_w_cm2).all()
            or (intensity_w_cm2 < 0).any()
        ):
            raise ValueError("material intensity must be finite nonnegative W/cm^2")
        if intensity_w_cm2.shape != field.shape or intensity_w_cm2.device != field.device:
            raise ValueError("intensity and field must share shape and device")
        if params.model == "identity":
            return field
        saturation = intensity_w_cm2 / (intensity_w_cm2 + params.i_sat_w_cm2)
        power_transmission = params.t_ns + params.delta_t * saturation
        if params.model == "saturable_absorption_plus_RSA":
            power_transmission = power_transmission / (1.0 + intensity_w_cm2 / params.i_rsa_w_cm2)
        output = field * power_transmission.sqrt()
        if params.model == "amplitude_plus_phase_nonlinearity":
            phase = params.phi_max_rad * saturation
            output = output * torch.polar(torch.ones_like(phase), phase)
        if not torch.isfinite(output).all():
            raise ValueError("material output is nonfinite")
        return output
