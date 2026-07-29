"""Explicit B=0 coherent imaging relays with independent reference paths."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as functional

from ..contracts import ComplexField, OperationProvenance, PlaneSpec
from .propagation import angular_spectrum_propagate


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


class NearConjugateRelay:
    """A physical two-lens 1:1 relay with a declared, signed residual B.

    The production operator is the LCT-equivalent parity plus residual
    diffraction.  ``effective_b_m`` is *not* submitted to the free-space
    propagator as a physical negative distance: the physical construction is
    recorded separately as three positive mechanical segments.
    """

    focal_length_m = 25.0e-3
    segments_m = (25.0e-3, 50.0e-3, 24.75e-3)
    clear_aperture_m = 12.7e-3
    effective_b_m = 0.25e-3

    def __init__(self, source: PlaneSpec, target: PlaneSpec, *, name: str) -> None:
        self.source, self.target, self.name = source, target, name
        self.abcd = path_abcd(self.segments_m, (self.focal_length_m, self.focal_length_m))
        expected = ABCDMatrix(-1.0, self.effective_b_m, 0.0, -1.0)
        if any(abs(a - b) > 1.0e-12 for a, b in zip((self.abcd.a, self.abcd.b_m, self.abcd.c_per_m, self.abcd.d), (expected.a, expected.b_m, expected.c_per_m, expected.d), strict=True)):
            raise ValueError("near-conjugate mechanical relay does not realize its fixed ABCD matrix")
        if abs(self.abcd.determinant - 1.0) > 1.0e-12:
            raise ValueError("near-conjugate relay determinant must equal one")
        if source.shape != target.shape or source.pixel_pitch_m != target.pixel_pitch_m:
            raise ValueError("near-conjugate relay requires a unit-sampling conjugate plane")
        if abs((target.z_m - source.z_m) - sum(self.segments_m)) > 1.0e-12:
            raise ValueError("near-conjugate target z must equal the mechanical relay length")

    @staticmethod
    def _signed_residual(field: ComplexField, distance_m: float) -> torch.Tensor:
        """Numerical LCT residual; deliberately distinct from physical ASM."""
        height, width = field.plane.shape
        dy, dx = field.plane.pixel_pitch_m
        dtype = field.tensor.real.dtype
        fy = torch.fft.fftfreq(height, d=dy, device=field.tensor.device, dtype=dtype)
        fx = torch.fft.fftfreq(width, d=dx, device=field.tensor.device, dtype=dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radicand = 1.0 - (field.plane.wavelength_m * xx).square() - (field.plane.wavelength_m * yy).square()
        phase = 2.0 * math.pi * distance_m * torch.sqrt(radicand.clamp_min(0.0)) / field.plane.wavelength_m
        transfer = torch.polar(torch.ones_like(phase), phase) * (radicand >= 0.0)
        return torch.fft.ifft2(torch.fft.fft2(field.tensor, norm="ortho") * transfer, norm="ortho")

    def __call__(self, field: ComplexField) -> ComplexField:
        if field.plane != self.source:
            raise ValueError(f"{self.name} received an unexpected source plane")
        # [-1,+B;0,-1] = parity followed by a signed residual propagation.
        residual = self._signed_residual(field, -self.effective_b_m)
        output = torch.flip(residual, dims=(-2, -1))
        return ComplexField(
            output,
            self.target,
            OperationProvenance(
                "near_conjugate_relay_lct",
                field.plane.plane_id,
                sum(self.segments_m),
                {
                    "relay": self.name,
                    "abcd": self.abcd.to_dict(),
                    "mechanical_segments_m": list(self.segments_m),
                    "focal_lengths_m": [self.focal_length_m, self.focal_length_m],
                    "effective_B_m": self.effective_b_m,
                    "clear_aperture_m": self.clear_aperture_m,
                    "spatial_inversion": True,
                    "implementation": "parity_plus_signed_lct_residual_not_negative_free_space",
                    "source_sampling": self.source.to_dict(),
                    "target_sampling": self.target.to_dict(),
                },
            ),
        )


def reference_near_conjugate_lens_relay(field: ComplexField, target: PlaneSpec, *, name: str) -> ComplexField:
    """Independent positive-distance, two-thin-lens reference for audits."""
    if field.plane.shape != target.shape or field.plane.pixel_pitch_m != target.pixel_pitch_m:
        raise ValueError("reference near-conjugate relay requires matched planes")
    current = angular_spectrum_propagate(field, 25.0e-3, f"{name}_reference_lens_1")
    def lens(in_field: ComplexField, plane_id: str) -> ComplexField:
        y = (torch.arange(in_field.plane.shape[0], device=in_field.tensor.device, dtype=in_field.tensor.real.dtype) - (in_field.plane.shape[0] - 1) / 2) * in_field.plane.pixel_pitch_m[0]
        x = (torch.arange(in_field.plane.shape[1], device=in_field.tensor.device, dtype=in_field.tensor.real.dtype) - (in_field.plane.shape[1] - 1) / 2) * in_field.plane.pixel_pitch_m[1]
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        phase = -math.pi * (xx.square() + yy.square()) / (in_field.plane.wavelength_m * 25.0e-3)
        return ComplexField(in_field.tensor * torch.polar(torch.ones_like(phase), phase), in_field.plane.with_plane(plane_id), OperationProvenance("thin_lens_reference", in_field.plane.plane_id, None, {"focal_length_m": 25.0e-3}))
    current = lens(current, f"{name}_reference_lens_1_phase")
    current = angular_spectrum_propagate(current, 50.0e-3, f"{name}_reference_lens_2")
    current = lens(current, f"{name}_reference_lens_2_phase")
    current = angular_spectrum_propagate(current, 24.75e-3, target.plane_id)
    return ComplexField(current.tensor, target, OperationProvenance("near_conjugate_relay_lens_reference", field.plane.plane_id, 99.75e-3, {"relay": name, "abcd": path_abcd(NearConjugateRelay.segments_m, (25.0e-3, 25.0e-3)).to_dict()}))
