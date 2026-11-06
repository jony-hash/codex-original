"""Static, passive synthetic material diagnostics; no measured material model.

Fields have shape [B,1,H,W] and |E|^2 is W/m^2. Projection uses the
actual sampled modes and pixel area, never an orthonormal-mode assumption.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .material_nonlinearity import MaterialNonlinearity, MaterialParameters


@dataclass(frozen=True)
class SyntheticResponse:
    family: str = "identity"
    proxy_strength: float = 0.0
    proxy_saturation_scale: float = 1.0
    proxy_linear_transmission: float = 1.0  # power, not amplitude

    def __post_init__(self) -> None:
        if self.family not in {"identity", "linear", "saturation", "suppression"}:
            raise ValueError("only static amplitude proxy families are supported")
        values = (self.proxy_strength, self.proxy_saturation_scale, self.proxy_linear_transmission)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("proxy parameters must be finite")
        if not 0 <= self.proxy_strength <= 1 or self.proxy_saturation_scale <= 0:
            raise ValueError("invalid synthetic response strength or scale")
        if not 0 <= self.proxy_linear_transmission <= 1:
            raise ValueError("passive power transmission must lie in [0,1]")


class WeightedModeProjector:
    """Area-weighted least-squares projection onto sampled columns of V."""

    def __init__(self, modes: torch.Tensor, pixel_area_m2: float) -> None:
        if modes.ndim != 2 or not torch.is_complex(modes) or not torch.isfinite(modes).all():
            raise ValueError("modes must be finite complex [pixels,modes]")
        if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
            raise ValueError("pixel area must be finite positive m^2")
        self.modes = modes.to(torch.complex128)
        self.area = pixel_area_m2
        self.gram = self.modes.mH @ self.modes * self.area
        self.gram_pinv = torch.linalg.pinv(self.gram, hermitian=True, rtol=1e-12)

    def project(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (field.ndim != 4 or field.shape[1] != 1
                or field[0].numel() != self.modes.shape[0]
                or not torch.is_complex(field) or not torch.isfinite(field).all()):
            raise ValueError("field must be finite complex [B,1,H,W] on the sampled mode plane")
        flat = field.to(torch.complex128).flatten(1)
        overlaps = flat @ self.modes.conj() * self.area
        coefficients = overlaps @ self.gram_pinv.T
        parallel = (coefficients @ self.modes.T).reshape(field.shape)
        return parallel, field.to(torch.complex128) - parallel

    def metrics(self, field: torch.Tensor) -> dict[str, torch.Tensor]:
        parallel, residual = self.project(field)
        power = field.to(torch.complex128).abs().square().flatten(1).sum(1) * self.area
        p_parallel = parallel.abs().square().flatten(1).sum(1) * self.area
        p_perp = residual.abs().square().flatten(1).sum(1) * self.area
        return {"power_w": power, "parallel_power_w": p_parallel,
                "perp_power_w": p_perp, "eta_perp": p_perp / power.clamp_min(1e-30),
                "closure_relative": (power - p_parallel - p_perp).abs() / power.clamp_min(1e-30)}


class SpatialMaterialInterface:
    """Parallel M0/M1/M2 entry, leaving the historical producer unchanged.

M1 delegates to MaterialNonlinearity with the old peak-intensity convention.
The supplied re-emitter is the existing OLS operator, not a new basis renderer.
"""

    def __init__(self, reference_intensity_w_m2: float, legacy_mfd_m: float,
                 reemit: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if not math.isfinite(reference_intensity_w_m2) or reference_intensity_w_m2 <= 0:
            raise ValueError("reference intensity must be positive W/m^2")
        if not math.isfinite(legacy_mfd_m) or legacy_mfd_m <= 0:
            raise ValueError("legacy MFD must be positive m")
        self.reference = reference_intensity_w_m2
        self.legacy_mfd = legacy_mfd_m
        self.reemit = reemit

    def apply(self, model: str, field: torch.Tensor, coefficients: torch.Tensor,
              response: SyntheticResponse) -> torch.Tensor:
        if (not torch.is_complex(field) or not torch.isfinite(field).all()
                or field.ndim != 4 or field.shape[1] != 1):
            raise ValueError("material field must be finite complex [B,1,H,W]")
        if model not in {"M0", "M1", "M2"}:
            raise ValueError("unknown material interface")
        if model == "M0" and response.family not in {"identity", "linear"}:
            raise ValueError("M0 only supports constant transmission")
        if model == "M1":
            if (coefficients.ndim != 2 or coefficients.shape[0] != field.shape[0]
                    or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all()):
                raise ValueError("M1 needs finite complex [B,N] coefficients")
            intensity = 2 * coefficients.abs().square() / (math.pi * (self.legacy_mfd / 2)**2) / 1e4
            scale = self.reference / 1e4 * response.proxy_saturation_scale
            if response.family == "identity":
                params = MaterialParameters()
            elif response.family == "linear":
                params = MaterialParameters(model="saturable_absorption",
                                            t_ns=response.proxy_linear_transmission,
                                            delta_t=0.0, i_sat_w_cm2=scale)
            elif response.family == "saturation":
                params = MaterialParameters(model="saturable_absorption",
                                            t_ns=1-response.proxy_strength,
                                            delta_t=response.proxy_strength, i_sat_w_cm2=scale)
            elif response.proxy_strength == 0:
                params = MaterialParameters()
            else:
                params = MaterialParameters(model="saturable_absorption_plus_RSA", t_ns=1,
                                            delta_t=0, i_sat_w_cm2=scale,
                                            i_rsa_w_cm2=scale/response.proxy_strength)
            return self.reemit(MaterialNonlinearity().forward(coefficients, intensity, params)).unsqueeze(1)
        if response.family == "identity":
            return field
        if response.family == "linear":
            return field * math.sqrt(response.proxy_linear_transmission)
        x = field.abs().square() / self.reference / response.proxy_saturation_scale
        if response.family == "saturation":
            transmission = 1-response.proxy_strength + response.proxy_strength*x/(1+x)
        else:
            transmission = 1/(1+response.proxy_strength*x)
        return field * transmission.sqrt()


def diagnostic_decision(eta: float, field_difference: float, output_difference: float,
                        stable: bool, *, eta_tolerance: float = 1e-8,
                        difference_tolerance: float = 1e-3) -> tuple[str, str]:
    """Consume verified production metrics; relevance means output change, not benefit."""
    if not stable:
        return "D", "M2_RESULTS_INCONCLUSIVE"
    if eta <= eta_tolerance and field_difference <= difference_tolerance:
        return "A", "M2_INTERFACE_VALID_NO_MATERIAL_DISTINCTION"
    if output_difference <= difference_tolerance:
        return "B", "M2_INTERFACE_VALID_FIELD_EFFECT_ONLY"
    return "C", "M2_INTERFACE_VALID_TASK_RELEVANT"
