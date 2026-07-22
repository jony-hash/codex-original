"""Formal v2 SLM and OLS devices."""
from __future__ import annotations

import torch

from ..contracts import AmplitudeCommand, ComplexField, OperationProvenance


class IdealAmplitudeSLM:
    """A real amplitude command mapped to a zero-residual-phase complex field."""

    calibration_id = "ideal-amplitude-v2"

    def __call__(self, command: AmplitudeCommand) -> ComplexField:
        if command.calibration_id != self.calibration_id:
            raise ValueError("formal v2 SLM requires the ideal-amplitude-v2 calibration")
        if not torch.isfinite(command.tensor).all() or command.tensor.min() < 0.0 or command.tensor.max() > 1.0:
            raise ValueError("formal amplitude SLM command must be finite and in [0, 1]")
        field = torch.complex(command.tensor, torch.zeros_like(command.tensor))
        return ComplexField(field, command.plane, OperationProvenance("amplitude_slm", None, None, {"calibration_id": command.calibration_id}))


def identity_ols(field: ComplexField) -> ComplexField:
    """Strict complex-linear identity OLS preserving the complex-field interface."""
    return ComplexField(field.tensor, field.plane.with_plane("ols_output"), OperationProvenance("identity_ols", field.plane.plane_id, None))
