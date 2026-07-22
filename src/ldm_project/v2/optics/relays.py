"""Physical scaled-imaging relays with explicit sampling and ABCD provenance."""
from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..contracts import ComplexField, OperationProvenance, PlaneSpec


def _scaled_grid(
    source: PlaneSpec,
    target: PlaneSpec,
    magnification: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if magnification == 0.0:
        raise ValueError("relay magnification cannot be zero")
    target_y = (torch.arange(target.shape[0], device=device, dtype=dtype) - (target.shape[0] - 1) / 2.0) * target.pixel_pitch_m[0]
    target_x = (torch.arange(target.shape[1], device=device, dtype=dtype) - (target.shape[1] - 1) / 2.0) * target.pixel_pitch_m[1]
    source_y, source_x = torch.meshgrid(target_y / magnification, target_x / magnification, indexing="ij")
    norm_y = source_y / (((source.shape[0] - 1) / 2.0) * source.pixel_pitch_m[0])
    norm_x = source_x / (((source.shape[1] - 1) / 2.0) * source.pixel_pitch_m[1])
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


class CoherentImagingRelay:
    """Named coherent scaled-imaging relay; not a generic complex resize."""

    def __init__(self, source: PlaneSpec, target: PlaneSpec, magnification: float, *, name: str, segments_m: tuple[float, ...]) -> None:
        self.source = source
        self.target = target
        self.magnification = magnification
        self.name = name
        self.segments_m = segments_m
        source_fov = source.fov_m
        target_fov = target.fov_m
        if any(abs(abs(magnification) * a - b) > 1.0e-12 for a, b in zip(source_fov, target_fov, strict=True)):
            raise ValueError("relay target FOV must equal source FOV scaled by magnification")

    def __call__(self, field: ComplexField) -> ComplexField:
        if field.plane != self.source:
            raise ValueError(f"{self.name} received an unexpected source plane")
        grid = _scaled_grid(self.source, self.target, self.magnification, device=field.tensor.device, dtype=field.tensor.real.dtype)
        grid = grid.expand(field.tensor.shape[0], -1, -1, -1)
        real = functional.grid_sample(field.tensor.real, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        imag = functional.grid_sample(field.tensor.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        output = torch.complex(real, imag)
        return ComplexField(
            output,
            self.target,
            OperationProvenance(
                "coherent_scaled_imaging",
                field.plane.plane_id,
                sum(self.segments_m),
                {
                    "relay": self.name,
                    "magnification": self.magnification,
                    "segments_m": list(self.segments_m),
                    "source_sampling": self.source.to_dict(),
                    "target_sampling": self.target.to_dict(),
                },
            ),
        )


def reference_nearest_scaled_imaging(field: ComplexField, target: PlaneSpec, magnification: float, *, name: str) -> ComplexField:
    """Independent nearest-neighbour reference for relay invariant tests."""
    source = field.plane
    y = torch.arange(target.shape[0], device=field.tensor.device, dtype=field.tensor.real.dtype)
    x = torch.arange(target.shape[1], device=field.tensor.device, dtype=field.tensor.real.dtype)
    sy = torch.round((y - (target.shape[0] - 1) / 2.0) * target.pixel_pitch_m[0] / magnification / source.pixel_pitch_m[0] + (source.shape[0] - 1) / 2.0).long()
    sx = torch.round((x - (target.shape[1] - 1) / 2.0) * target.pixel_pitch_m[1] / magnification / source.pixel_pitch_m[1] + (source.shape[1] - 1) / 2.0).long()
    sy = sy.clamp(0, source.shape[0] - 1)
    sx = sx.clamp(0, source.shape[1] - 1)
    output = field.tensor.index_select(-2, sy).index_select(-1, sx)
    return ComplexField(output, target, OperationProvenance("reference_nearest_scaled_imaging", source.plane_id, None, {"relay": name, "magnification": magnification}))
