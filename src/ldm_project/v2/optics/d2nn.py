"""Formal finite-crop phase-only D2NN implementation."""
from __future__ import annotations

import torch
from torch import nn

from ..contracts import ComplexField, OperationProvenance
from .propagation import angular_spectrum_propagate


class FiniteCropPhaseStack(nn.Module):
    """phase modulation -> center pad -> ASM -> center crop for every layer."""

    def __init__(self, *, layers: int, active_shape: tuple[int, int], propagation_shape: tuple[int, int], spacing_m: float) -> None:
        super().__init__()
        if layers <= 0 or any(p < a for p, a in zip(propagation_shape, active_shape, strict=True)):
            raise ValueError("invalid finite-crop D2NN dimensions")
        self.layers = layers
        self.active_shape = active_shape
        self.propagation_shape = propagation_shape
        self.spacing_m = spacing_m
        self.phase = nn.Parameter(torch.zeros(layers, 1, *active_shape))

    def forward(self, field: ComplexField, prefix: str) -> tuple[ComplexField, tuple[ComplexField, ...], dict[str, torch.Tensor]]:
        if field.plane.shape != self.active_shape:
            raise ValueError("finite-crop D2NN field does not match active shape")
        records: list[ComplexField] = []
        retention: list[torch.Tensor] = []
        current = field
        for index in range(self.layers):
            transfer = torch.polar(torch.ones_like(self.phase[index]), self.phase[index]).to(current.tensor.dtype)
            modulated = ComplexField(current.tensor * transfer, current.plane, OperationProvenance("phase_only_modulation", current.plane.plane_id, None, {"layer": index + 1}))
            padding_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
            padding_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
            padded_tensor = torch.nn.functional.pad(modulated.tensor, (padding_x, padding_x, padding_y, padding_y))
            padded_plane = current.plane.with_plane(
                f"{prefix}_layer_{index + 1}_padded",
                shape=self.propagation_shape,
                coordinate_frame=f"{prefix}_propagation_grid",
            )
            padded = ComplexField(padded_tensor, padded_plane, OperationProvenance("finite_crop_center_pad", current.plane.plane_id, None, {"active_shape": list(self.active_shape)}))
            propagated = angular_spectrum_propagate(padded, self.spacing_m, f"{prefix}_layer_{index + 1}_propagated")
            crop_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
            crop_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
            cropped_tensor = propagated.tensor[..., crop_y:crop_y + self.active_shape[0], crop_x:crop_x + self.active_shape[1]]
            active_plane = current.plane.with_plane(f"{prefix}_layer_{index + 1}", z_m=propagated.plane.z_m)
            current = ComplexField(cropped_tensor, active_plane, OperationProvenance("finite_crop_hard_aperture", propagated.plane.plane_id, self.spacing_m, {"layer": index + 1, "active_shape": list(self.active_shape)}))
            retention.append(current.tensor.abs().square().sum(dim=(-3, -2, -1)) / propagated.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12))
            records.append(current)
        return current, tuple(records), {"crop_retention": torch.stack(retention, dim=1), "phase": self.phase}
