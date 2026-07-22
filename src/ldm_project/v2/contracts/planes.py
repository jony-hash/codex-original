"""Typed physical-plane values used by the v2 formal optical path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class PlaneSpec:
    plane_id: str
    shape: tuple[int, int]
    pixel_pitch_m: tuple[float, float]
    wavelength_m: float
    z_m: float
    coordinate_frame: str
    origin_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.plane_id or not self.coordinate_frame:
            raise ValueError("plane_id and coordinate_frame are required")
        if len(self.shape) != 2 or any(value <= 0 for value in self.shape):
            raise ValueError("plane shape must contain two positive values")
        if len(self.pixel_pitch_m) != 2 or any(value <= 0.0 for value in self.pixel_pitch_m):
            raise ValueError("plane pixel_pitch_m must contain two positive values")
        if self.wavelength_m <= 0.0:
            raise ValueError("plane wavelength_m must be positive")

    @property
    def fov_m(self) -> tuple[float, float]:
        return tuple(size * pitch for size, pitch in zip(self.shape, self.pixel_pitch_m, strict=True))

    def with_plane(
        self,
        plane_id: str,
        *,
        z_m: float | None = None,
        shape: tuple[int, int] | None = None,
        pixel_pitch_m: tuple[float, float] | None = None,
        coordinate_frame: str | None = None,
    ) -> "PlaneSpec":
        return PlaneSpec(
            plane_id=plane_id,
            shape=self.shape if shape is None else shape,
            pixel_pitch_m=self.pixel_pitch_m if pixel_pitch_m is None else pixel_pitch_m,
            wavelength_m=self.wavelength_m,
            z_m=self.z_m if z_m is None else z_m,
            coordinate_frame=self.coordinate_frame if coordinate_frame is None else coordinate_frame,
            origin_m=self.origin_m,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "plane_id": self.plane_id,
            "shape": list(self.shape),
            "pixel_pitch_m": list(self.pixel_pitch_m),
            "fov_m": list(self.fov_m),
            "wavelength_m": self.wavelength_m,
            "z_m": self.z_m,
            "coordinate_frame": self.coordinate_frame,
            "origin_m": list(self.origin_m),
        }


@dataclass(frozen=True)
class OperationProvenance:
    operation: str
    upstream_plane: str | None
    propagation_distance_m: float | None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("operation is required")
        if self.operation.startswith("propagate") and self.propagation_distance_m is None:
            raise ValueError("propagation operations require propagation_distance_m")
        if self.propagation_distance_m is not None and self.propagation_distance_m < 0.0:
            raise ValueError("propagation_distance_m must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "upstream_plane": self.upstream_plane,
            "propagation_distance_m": self.propagation_distance_m,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ComplexField:
    tensor: torch.Tensor
    plane: PlaneSpec
    provenance: OperationProvenance

    def __post_init__(self) -> None:
        if not torch.is_complex(self.tensor):
            raise TypeError("ComplexField requires a complex tensor")
        if tuple(self.tensor.shape[-2:]) != self.plane.shape:
            raise ValueError("ComplexField tensor shape does not match PlaneSpec")

    @property
    def amplitude(self) -> torch.Tensor:
        return self.tensor.abs()

    @property
    def intensity(self) -> "IntensityFrame":
        return IntensityFrame(self.tensor.abs().square(), self.plane, self.provenance)


@dataclass(frozen=True)
class AmplitudeCommand:
    tensor: torch.Tensor
    plane: PlaneSpec
    calibration_id: str

    def __post_init__(self) -> None:
        if torch.is_complex(self.tensor):
            raise TypeError("AmplitudeCommand must be real-valued")
        if tuple(self.tensor.shape[-2:]) != self.plane.shape:
            raise ValueError("AmplitudeCommand tensor shape does not match PlaneSpec")
        if not self.calibration_id:
            raise ValueError("AmplitudeCommand requires calibration_id")


@dataclass(frozen=True)
class IntensityFrame:
    tensor: torch.Tensor
    plane: PlaneSpec
    provenance: OperationProvenance
    exposure_id: str | None = None

    def __post_init__(self) -> None:
        if torch.is_complex(self.tensor):
            raise TypeError("IntensityFrame must be real-valued")
        if tuple(self.tensor.shape[-2:]) != self.plane.shape:
            raise ValueError("IntensityFrame tensor shape does not match PlaneSpec")


@dataclass(frozen=True)
class ExposureCalibration:
    calibration_id: str
    scale: float
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not self.calibration_id or self.scale <= 0.0 or self.gamma != 1.0:
            raise ValueError("exposure requires id, positive fixed scale, and gamma=1")


@dataclass(frozen=True)
class DisplayFrame:
    tensor: torch.Tensor
    source: IntensityFrame
    calibration: ExposureCalibration

    def __post_init__(self) -> None:
        if torch.is_complex(self.tensor):
            raise TypeError("DisplayFrame must be real-valued")
        if self.tensor.shape != self.source.tensor.shape:
            raise ValueError("DisplayFrame must preserve the source tensor shape")
        if not torch.isfinite(self.tensor).all() or self.tensor.min() < 0.0 or self.tensor.max() > 1.0:
            raise ValueError("DisplayFrame must be finite and clipped to [0, 1]")


def render_fixed_exposure(frame: IntensityFrame, calibration: ExposureCalibration) -> DisplayFrame:
    if frame.exposure_id not in {None, calibration.calibration_id}:
        raise ValueError("intensity frame was captured with another exposure calibration")
    return DisplayFrame((frame.tensor / calibration.scale).clamp(0.0, 1.0), frame, calibration)
