from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlaneSampling:
    """Physical sampling contract for one transverse optical-field plane."""

    name: str
    height: int
    width: int
    pixel_size_y: float
    pixel_size_x: float
    origin_y: float = 0.0
    origin_x: float = 0.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("plane sampling shape must be positive")
        if self.pixel_size_y <= 0.0 or self.pixel_size_x <= 0.0:
            raise ValueError("plane sampling pixel sizes must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def fov_y(self) -> float:
        return self.height * self.pixel_size_y

    @property
    def fov_x(self) -> float:
        return self.width * self.pixel_size_x

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": [self.height, self.width],
            "pixel_size_y": self.pixel_size_y,
            "pixel_size_x": self.pixel_size_x,
            "fov_y": self.fov_y,
            "fov_x": self.fov_x,
            "origin_y": self.origin_y,
            "origin_x": self.origin_x,
        }


def validate_sampling_transition(
    source: PlaneSampling,
    target: PlaneSampling,
    *,
    operation: str,
) -> None:
    if operation == "generic_resize":
        raise ValueError(
            f"{source.name}->{target.name} requires an explicit physical mapping; "
            "generic_resize is not an optical operation"
        )
    if operation not in {"same_fov_resample", "physical_aperture_sampling", "scaled_imaging"}:
        raise ValueError(f"unsupported physical sampling transition: {operation!r}")


def aperture_sampling(
    source: PlaneSampling,
    target: PlaneSampling,
    *,
    aperture_width: float,
    aperture_height: float,
) -> dict[str, object]:
    if aperture_width <= 0.0 or aperture_height <= 0.0:
        raise ValueError("physical aperture dimensions must be positive")
    if aperture_width > source.fov_x or aperture_height > source.fov_y:
        raise ValueError("physical aperture must fit inside source field of view")
    validate_sampling_transition(source, target, operation="physical_aperture_sampling")
    return {
        "operation": "physical_aperture_sampling",
        "source_plane": source.name,
        "target_plane": target.name,
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "source_fov_x": source.fov_x,
        "source_fov_y": source.fov_y,
        "target_fov_x": target.fov_x,
        "target_fov_y": target.fov_y,
        "aperture_width": aperture_width,
        "aperture_height": aperture_height,
    }


def center_aperture_indices(
    source: PlaneSampling,
    *,
    aperture_width: float,
    aperture_height: float,
) -> tuple[int, int, int, int]:
    if aperture_width <= 0.0 or aperture_height <= 0.0:
        raise ValueError("physical aperture dimensions must be positive")
    if aperture_width > source.fov_x or aperture_height > source.fov_y:
        raise ValueError("physical aperture must fit inside source field of view")
    width = int(round(aperture_width / source.pixel_size_x))
    height = int(round(aperture_height / source.pixel_size_y))
    if width <= 0 or height <= 0:
        raise ValueError("physical aperture must cover at least one source pixel")
    left = (source.width - width) // 2
    top = (source.height - height) // 2
    return top, left, height, width


def sample_center_aperture(
    field: torch.Tensor,
    source: PlaneSampling,
    *,
    target_size: tuple[int, int],
    target_pixel_size: float,
) -> torch.Tensor:
    """Sample a centered physical aperture from a complex field."""
    if not torch.is_complex(field):
        raise ValueError("physical aperture sampling requires a complex field")
    aperture_height = target_size[0] * target_pixel_size
    aperture_width = target_size[1] * target_pixel_size
    top, left, height, width = center_aperture_indices(
        source,
        aperture_width=aperture_width,
        aperture_height=aperture_height,
    )
    cropped = field[..., top : top + height, left : left + width]
    if tuple(cropped.shape[-2:]) == target_size:
        return cropped
    real = F.interpolate(cropped.real, size=target_size, mode="bilinear", align_corners=False)
    imag = F.interpolate(cropped.imag, size=target_size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)
