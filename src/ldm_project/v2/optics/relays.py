"""Explicit B=0 coherent imaging relays with independent reference paths."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as functional

from ..contracts import ComplexField, OperationProvenance, PlaneSpec


@dataclass(frozen=True)
class ABCDMatrix:
    a: float
    b_m: float
    c_per_m: float
    d: float

    @property
    def determinant(self) -> float:
        return self.a * self.d - self.b_m * self.c_per_m

    def to_dict(self) -> dict[str, float]:
        return {"A": self.a, "B_m": self.b_m, "C_per_m": self.c_per_m, "D": self.d, "determinant": self.determinant}


def _multiply(left: ABCDMatrix, right: ABCDMatrix) -> ABCDMatrix:
    return ABCDMatrix(left.a * right.a + left.b_m * right.c_per_m, left.a * right.b_m + left.b_m * right.d, left.c_per_m * right.a + left.d * right.c_per_m, left.c_per_m * right.b_m + left.d * right.d)


def free_space(distance_m: float) -> ABCDMatrix:
    if distance_m < 0.0:
        raise ValueError("free-space distance must be non-negative")
    return ABCDMatrix(1.0, distance_m, 0.0, 1.0)


def thin_lens(focal_length_m: float) -> ABCDMatrix:
    if focal_length_m <= 0.0:
        raise ValueError("lens focal length must be positive")
    return ABCDMatrix(1.0, 0.0, -1.0 / focal_length_m, 1.0)


def path_abcd(segments_m: tuple[float, ...], focal_lengths_m: tuple[float, ...]) -> ABCDMatrix:
    """Return propagation/lens ray matrix for a relay path."""
    if len(segments_m) != len(focal_lengths_m) + 1:
        raise ValueError("relay path requires one more segment than lenses")
    matrix = free_space(segments_m[0])
    for focal_length, segment in zip(focal_lengths_m, segments_m[1:], strict=True):
        matrix = _multiply(thin_lens(focal_length), matrix)
        matrix = _multiply(free_space(segment), matrix)
    return matrix


def _physical_grid(source: PlaneSpec, target: PlaneSpec, magnification: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    target_y = (torch.arange(target.shape[0], device=device, dtype=dtype) - (target.shape[0] - 1.0) / 2.0) * target.pixel_pitch_m[0]
    target_x = (torch.arange(target.shape[1], device=device, dtype=dtype) - (target.shape[1] - 1.0) / 2.0) * target.pixel_pitch_m[1]
    source_y, source_x = torch.meshgrid(target_y / magnification, target_x / magnification, indexing="ij")
    norm_y = source_y / (((source.shape[0] - 1.0) / 2.0) * source.pixel_pitch_m[0])
    norm_x = source_x / (((source.shape[1] - 1.0) / 2.0) * source.pixel_pitch_m[1])
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


def _quadratic_phase(target: PlaneSpec, matrix: ABCDMatrix, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = (torch.arange(target.shape[0], device=device, dtype=dtype) - (target.shape[0] - 1.0) / 2.0) * target.pixel_pitch_m[0]
    x = (torch.arange(target.shape[1], device=device, dtype=dtype) - (target.shape[1] - 1.0) / 2.0) * target.pixel_pitch_m[1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = math.pi * matrix.c_per_m * (xx.square() + yy.square()) / (target.wavelength_m * matrix.a)
    return torch.polar(torch.ones_like(phase), phase)


class CoherentImagingRelay:
    """Production B=0 coherent imaging; interpolation is only physical resampling."""

    def __init__(self, source: PlaneSpec, target: PlaneSpec, magnification: float, *, name: str, segments_m: tuple[float, ...], focal_lengths_m: tuple[float, ...], clear_aperture_m: float, matrix_segments_m: tuple[float, ...] | None = None) -> None:
        self.source, self.target, self.magnification, self.name = source, target, magnification, name
        self.segments_m, self.focal_lengths_m, self.clear_aperture_m = segments_m, focal_lengths_m, clear_aperture_m
        self.matrix_segments_m = matrix_segments_m or segments_m
        self.abcd = path_abcd(self.matrix_segments_m, focal_lengths_m)
        if abs(self.abcd.a - magnification) > 1.0e-9 or abs(self.abcd.b_m) > 1.0e-9:
            raise ValueError("relay path must realize the declared B=0 magnification")
        if abs(self.abcd.determinant - 1.0) > 1.0e-9:
            raise ValueError("lossless relay ABCD determinant must equal one")
        if clear_aperture_m <= 0.0:
            raise ValueError("relay clear aperture must be positive")
        if any(abs(abs(magnification) * source_fov - target_fov) > 1.0e-12 for source_fov, target_fov in zip(source.fov_m, target.fov_m, strict=True)):
            raise ValueError("relay target FOV must equal source FOV scaled by magnification")
        if abs((target.z_m - source.z_m) - sum(segments_m)) > 1.0e-12:
            raise ValueError("relay plane z coordinates must equal its physical segment sum")

    def __call__(self, field: ComplexField) -> ComplexField:
        if field.plane != self.source:
            raise ValueError(f"{self.name} received an unexpected source plane")
        grid = _physical_grid(self.source, self.target, self.magnification, device=field.tensor.device, dtype=field.tensor.real.dtype).expand(field.tensor.shape[0], -1, -1, -1)
        sampled = torch.complex(functional.grid_sample(field.tensor.real, grid, mode="bilinear", padding_mode="zeros", align_corners=True), functional.grid_sample(field.tensor.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=True))
        output = sampled * _quadratic_phase(self.target, self.abcd, device=field.tensor.device, dtype=field.tensor.real.dtype) * (1.0 / abs(self.abcd.a))
        return ComplexField(output, self.target, OperationProvenance("coherent_b0_imaging", field.plane.plane_id, sum(self.segments_m), {"relay": self.name, "magnification": self.magnification, "segments_m": list(self.segments_m), "matrix_segments_m": list(self.matrix_segments_m), "focal_lengths_m": list(self.focal_lengths_m), "clear_aperture_m": self.clear_aperture_m, "abcd": self.abcd.to_dict(), "source_sampling": self.source.to_dict(), "target_sampling": self.target.to_dict()}))


def reference_b0_imaging(field: ComplexField, target: PlaneSpec, matrix: ABCDMatrix, *, name: str) -> ComplexField:
    """Independent closed-form B=0 reference using explicit nearest physical samples."""
    if abs(matrix.b_m) > 1.0e-9:
        raise ValueError("reference only supports B=0 imaging")
    source = field.plane
    y = torch.arange(target.shape[0], device=field.tensor.device, dtype=field.tensor.real.dtype)
    x = torch.arange(target.shape[1], device=field.tensor.device, dtype=field.tensor.real.dtype)
    sy = torch.round((y - (target.shape[0] - 1.0) / 2.0) * target.pixel_pitch_m[0] / matrix.a / source.pixel_pitch_m[0] + (source.shape[0] - 1.0) / 2.0).long()
    sx = torch.round((x - (target.shape[1] - 1.0) / 2.0) * target.pixel_pitch_m[1] / matrix.a / source.pixel_pitch_m[1] + (source.shape[1] - 1.0) / 2.0).long()
    valid = ((sy >= 0) & (sy < source.shape[0]))[:, None] & ((sx >= 0) & (sx < source.shape[1]))[None, :]
    sampled = field.tensor.index_select(-2, sy.clamp(0, source.shape[0] - 1)).index_select(-1, sx.clamp(0, source.shape[1] - 1))
    output = sampled * valid.to(sampled.dtype) * _quadratic_phase(target, matrix, device=field.tensor.device, dtype=field.tensor.real.dtype) * (1.0 / abs(matrix.a))
    return ComplexField(output, target, OperationProvenance("reference_closed_form_b0_imaging", source.plane_id, target.z_m - source.z_m, {"relay": name, "abcd": matrix.to_dict()}))
