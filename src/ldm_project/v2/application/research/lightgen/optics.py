"""Explicit diagnostic-only optical upper-bound operators.

These operators reuse the formal finite-crop propagation implementation and
therefore keep physical plane metadata.  They are intentionally isolated from
``v2.optics``: complex modulation and the 1 um spatial upper bound cannot be
selected by the formal runtime.
"""

from __future__ import annotations

import torch
from torch import nn

from ldm_project.v2.contracts import ComplexField, IntensityFrame, OperationProvenance
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.propagation import angular_spectrum_propagate
from ldm_project.v2.optics.system import FormalAmplitudeFiniteCropIdentitySystem, FormalForwardResult


class ComplexControlFiniteCropStack(FiniteCropPhaseStack):
    """Finite-crop stack with independently trainable amplitude and phase.

    This is an ideal, diagnostic upper bound; ``amplitude_logits`` is retained
    in evidence and never maps to a fabrication or formal-v2 parameter.
    """

    diagnostic_kind = "complex_control_upper_bound"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # logit(approximately 1) keeps the complex upper bound locally
        # identity at initialization.  Zero would impose 0.5 field amplitude
        # per layer and manufacture a four-layer attenuation failure.
        self.amplitude_logits = nn.Parameter(torch.full_like(self.phase, 8.0))

    def forward(self, field: ComplexField, prefix: str):
        if field.plane.shape != self.active_shape:
            raise ValueError("complex-control field does not match active shape")
        records, retention = [], []
        current = field
        for index in range(self.layers):
            transfer = torch.sigmoid(self.amplitude_logits[index]) * torch.polar(
                torch.ones_like(self.phase[index]), self.phase[index]
            ).to(current.tensor.dtype)
            modulated = ComplexField(
                current.tensor * transfer,
                current.plane,
                OperationProvenance(
                    "diagnostic_complex_modulation",
                    current.plane.plane_id,
                    None,
                    {"layer": index + 1, "diagnostic_only": True},
                ),
            )
            padding_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
            padding_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
            padded_plane = current.plane.with_plane(
                f"{prefix}_layer_{index + 1}_padded",
                shape=self.propagation_shape,
                coordinate_frame=f"{prefix}_propagation_grid",
            )
            padded = ComplexField(
                torch.nn.functional.pad(modulated.tensor, (padding_x, padding_x, padding_y, padding_y)),
                padded_plane,
                OperationProvenance("finite_crop_center_pad", current.plane.plane_id, None, {"active_shape": list(self.active_shape)}),
            )
            propagated = angular_spectrum_propagate(
                padded, self.spacing_m, f"{prefix}_layer_{index + 1}_propagated"
            )
            cropped = propagated.tensor[..., padding_y : padding_y + self.active_shape[0], padding_x : padding_x + self.active_shape[1]]
            current = ComplexField(
                cropped,
                current.plane.with_plane(f"{prefix}_layer_{index + 1}", z_m=propagated.plane.z_m),
                OperationProvenance("finite_crop_hard_aperture", propagated.plane.plane_id, self.spacing_m, {"layer": index + 1, "active_shape": list(self.active_shape)}),
            )
            retention.append(current.tensor.abs().square().sum(dim=(-3, -2, -1)) / propagated.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1e-12))
            records.append(current)
        return current, tuple(records), {"crop_retention": torch.stack(retention, dim=1), "phase": self.phase, "amplitude_logits": self.amplitude_logits}


class DiagnosticPhaseStackBypass(nn.Module):
    """Front/back formal relays with an explicitly idealized chip bypass.

    This is O1's diagnostic representation upper bound.  The identity link is
    tagged in provenance and cannot be confused with a zero-phase 2+2 stack or
    with a fabricable propagation claim.
    """

    diagnostic_kind = "phase_stack_bypass_upper_bound"

    def __init__(self, system: FormalAmplitudeFiniteCropIdentitySystem) -> None:
        super().__init__()
        self.system = system

    def command(self, tensor: torch.Tensor):
        return self.system.command(tensor)

    def forward(self, command) -> FormalForwardResult:
        slm_field = self.system.slm(command)
        front = self.system.front(slm_field)
        bypass = ComplexField(
            front.tensor,
            self.system.chip_plane,
            OperationProvenance(
                "diagnostic_phase_stack_bypass",
                front.plane.plane_id,
                self.system.chip_plane.z_m - front.plane.z_m,
                {"diagnostic_only": True, "replaces": "encoder_identity_ols_decoder"},
            ),
        )
        camera_field = self.system.back(bypass)
        raw = IntensityFrame(camera_field.tensor.abs().square(), self.system.camera_plane, camera_field.provenance)
        empty = torch.ones(command.tensor.shape[0], 1, device=command.tensor.device, dtype=command.tensor.dtype)
        return FormalForwardResult(raw, raw, None, {}, {"encoder_crop_retention": empty, "decoder_crop_retention": empty}, None)
