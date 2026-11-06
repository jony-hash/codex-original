"""Paper-referenced six-layer LightGen MNIST baseline, kept diagnostic-only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
    GaussianFiberGeometry,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import angular_spectrum_propagate


ACTIVE_SHAPE = (400, 400)
PIXEL_PITCH_M = 3.0e-6
WAVELENGTH_M = 532.0e-9
LAYER_SPACING_M = 3.0e-3
LAYER_POWER_TRANSMISSION = 0.889
SOURCE_POWER_W = 249.3e-6
FIBER_PITCH_M = 125.0e-6
FIBER_MFD_M = 14.222e-6
FIBER_LENGTH_M = 8.5e-3
OLS_PORT_COUNT = 100
PHASE_LEVELS = torch.tensor(
    [0.0, 0.82, 1.63, 2.45, 3.26, 4.08, 4.89, 5.71], dtype=torch.float64
)


@dataclass(frozen=True)
class LightGen400Config:
    propagation_grid_size: int = 2048
    checkpoint_propagation: bool = True
    pixel_sampling_factor: int = 1

    def __post_init__(self) -> None:
        if self.propagation_grid_size < 512 or self.propagation_grid_size % 2:
            raise ValueError("propagation grid must be even and at least 512 pixels")
        if self.pixel_sampling_factor not in (1, 2, 4):
            raise ValueError("physical phase-pixel sampling factor must be one, two, or four")
        if self.propagation_grid_size < max(ACTIVE_SHAPE) * self.pixel_sampling_factor:
            raise ValueError("propagation grid must contain the sampled physical aperture")
        fov = self.propagation_grid_size * self.pixel_pitch_m
        if fov <= 9.0 * FIBER_PITCH_M + 125.0e-6:
            raise ValueError("propagation grid does not contain the assumed 10x10 fiber array")

    @property
    def active_shape(self) -> tuple[int, int]:
        return tuple(size * self.pixel_sampling_factor for size in ACTIVE_SHAPE)

    @property
    def pixel_pitch_m(self) -> float:
        return PIXEL_PITCH_M / self.pixel_sampling_factor


def integrated_power(field: ComplexField) -> torch.Tensor:
    """Integrate complex field power over the declared physical pixel area."""
    if not torch.is_complex(field.tensor):
        raise TypeError("physical optical power requires a complex field")
    dy, dx = field.plane.pixel_pitch_m
    return field.tensor.abs().square().reshape(field.tensor.shape[0], -1).sum(dim=1) * dy * dx


def integrate_camera_pixel_intensity(subpixel_intensity: torch.Tensor, sampling_factor: int) -> torch.Tensor:
    """Area-integrate fine-grid intensity into physical CMOS pixels.

    The input is intensity, not complex field. Averaging complex amplitudes
    before squaring would discard interference-free detector power.
    """
    if sampling_factor not in (1, 2, 4):
        raise ValueError("camera sampling factor must be one, two, or four")
    if subpixel_intensity.ndim != 4 or subpixel_intensity.shape[1] != 1:
        raise ValueError("camera intensity must have shape [B,1,H,W]")
    if subpixel_intensity.shape[-2] % sampling_factor or subpixel_intensity.shape[-1] % sampling_factor:
        raise ValueError("camera intensity dimensions must be divisible by the sampling factor")
    return torch.nn.functional.avg_pool2d(subpixel_intensity, kernel_size=sampling_factor)


def _center_crop(tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    crop_height, crop_width = shape
    if crop_height > height or crop_width > width:
        raise ValueError("crop shape exceeds the physical propagation plane")
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return tensor[..., top : top + crop_height, left : left + crop_width]


class ChenLightGen400Baseline(nn.Module):
    """Two encoder and four generator phase masks around a Gaussian 10x10 OLS.

    All unresolved SMF dimensions stay explicit project assumptions.  The
    learned phase masks are 400x400 at 3 um; FFT windows are numerical grids.
    """

    def __init__(self, config: LightGen400Config | None = None) -> None:
        super().__init__()
        self.config = config or LightGen400Config()
        grid = self.config.propagation_grid_size
        active_shape = self.config.active_shape
        pitch = self.config.pixel_pitch_m
        self.active_plane = PlaneSpec(
            "chen_lightgen400_encoder_aperture",
            active_shape,
            (pitch, pitch),
            WAVELENGTH_M,
            0.0,
            "lightgen400",
        )
        self.grid_plane = self.active_plane.with_plane(
            "chen_lightgen400_propagation_grid",
            shape=(grid, grid),
            coordinate_frame="lightgen400_global",
        )
        self.ols_input_plane = self.grid_plane.with_plane(
            "chen_lightgen400_ols_input", z_m=2.0 * LAYER_SPACING_M
        )
        # The physical phase pixels remain 400x400.  The optional factor-two
        # reference oversamples each held phase pixel; it does not add trainable
        # degrees of freedom.
        self.encoder_phase = nn.Parameter(torch.zeros(2, 1, *ACTIVE_SHAPE))
        self.decoder_phase = nn.Parameter(torch.zeros(4, 1, *ACTIVE_SHAPE))
        self.ols = GaussianFiberArrayOLS(
            self.ols_input_plane,
            GaussianFiberGeometry(
                port_pitch_m=FIBER_PITCH_M,
                waist_m=FIBER_MFD_M / 2.0,
            ),
        )

    @staticmethod
    def _field(tensor: torch.Tensor, plane: PlaneSpec, operation: str) -> ComplexField:
        return ComplexField(tensor, plane, OperationProvenance(operation, plane.plane_id, None))

    def _propagate(
        self, tensor: torch.Tensor, prefix: str, hop: int, *, input_z_m: float
    ) -> ComplexField:
        source_plane = self.grid_plane.with_plane(
            f"{prefix}_phase_output", z_m=input_z_m
        )
        source = self._field(tensor, source_plane, f"{prefix}_phase_output")

        def operation(value: torch.Tensor) -> torch.Tensor:
            source_field = self._field(value, source_plane, f"{prefix}_phase_output")
            return angular_spectrum_propagate(
                source_field,
                LAYER_SPACING_M,
                f"{prefix}_hop_{hop}",
            ).tensor

        propagated_tensor = (
            checkpoint(operation, source.tensor, use_reentrant=False)
            if self.config.checkpoint_propagation and torch.is_grad_enabled()
            else operation(source.tensor)
        )
        plane = self.grid_plane.with_plane(
            f"{prefix}_hop_{hop}", z_m=input_z_m + LAYER_SPACING_M
        )
        return ComplexField(
            propagated_tensor,
            plane,
            OperationProvenance("propagate_angular_spectrum", self.grid_plane.plane_id, LAYER_SPACING_M),
        )

    def _pad_active(self, tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
        pad = (grid_size - self.config.active_shape[0]) // 2
        return torch.nn.functional.pad(tensor, (pad, pad, pad, pad))

    def _phase(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = _center_crop(tensor, self.config.active_shape)
        phase = mask
        for _ in range(int(math.log2(self.config.pixel_sampling_factor))):
            phase = phase.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        transmission = torch.polar(
            torch.full_like(phase, math.sqrt(LAYER_POWER_TRANSMISSION)), phase
        ).to(dtype=tensor.dtype)
        return self._pad_active(active * transmission, self.config.propagation_grid_size)

    def _active_field(self, tensor: torch.Tensor, plane_id: str, z_m: float) -> ComplexField:
        active = _center_crop(tensor, self.config.active_shape)
        plane = self.active_plane.with_plane(plane_id, z_m=z_m)
        return self._field(active, plane, "finite_crop_hard_aperture")

    def forward_from_image(
        self,
        intensity: torch.Tensor,
        *,
        field_callback: Callable[[str, ComplexField], None] | None = None,
    ) -> dict[str, object]:
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(1)
        if tuple(intensity.shape[1:]) != (1, *ACTIVE_SHAPE):
            raise ValueError("LightGen 400 input must be [B,1,400,400] intensity")
        if not torch.isfinite(intensity).all() or (intensity < 0).any() or (intensity > 1).any():
            raise ValueError("LightGen 400 input intensity must be finite in [0,1]")
        for _ in range(int(math.log2(self.config.pixel_sampling_factor))):
            intensity = intensity.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        density = intensity.to(torch.float32) * (SOURCE_POWER_W / math.prod(self.active_plane.fov_m))
        active_input = self._field(density.sqrt().to(torch.complex64), self.active_plane, "slm_input")
        input_power_w = integrated_power(active_input)
        current = self._pad_active(active_input.tensor, self.config.propagation_grid_size)
        powers: list[torch.Tensor] = []
        encoder_fields: list[ComplexField] = []
        for layer in range(2):
            current = self._phase(current, self.encoder_phase[layer])
            propagated = self._propagate(
                current,
                "encoder",
                layer + 1,
                input_z_m=layer * LAYER_SPACING_M,
            )
            current = propagated.tensor
            if field_callback is None:
                encoder_fields.append(propagated)
            else:
                field_callback(f"encoder_{layer + 1}", propagated)
            powers.append(integrated_power(propagated))

        ols_input = self._field(current, self.ols_input_plane, "encoder_to_gaussian_ols")
        if field_callback is not None:
            field_callback("ols_input", ols_input)
        coefficients = self.ols.project_coefficients(ols_input)
        ols_output = self.ols(ols_input)
        ols_input_power_w = integrated_power(ols_input)
        ols_output_power_w = integrated_power(ols_output)
        if field_callback is not None:
            field_callback("ols_output", ols_output)
        generator_current = ols_output.tensor
        generator_plane = self.grid_plane.with_plane(
            "chen_lightgen400_generator_entrance", z_m=3.0 * LAYER_SPACING_M
        )
        generator_current = angular_spectrum_propagate(
            ComplexField(
                generator_current,
                ols_output.plane,
                OperationProvenance("ols_identity_transport", ols_output.plane.plane_id, None),
            ),
            LAYER_SPACING_M,
            generator_plane.plane_id,
        ).tensor
        generator_entry = self._active_field(
            generator_current, "chen_lightgen400_generator_entrance_aperture", 3.0 * LAYER_SPACING_M
        )
        if field_callback is not None:
            generator_full_plane = self.grid_plane.with_plane(
                "chen_lightgen400_generator_entrance_full_plane", z_m=3.0 * LAYER_SPACING_M
            )
            field_callback("generator_entry", self._field(
                generator_current, generator_full_plane, "generator_entrance_full_plane"
            ))
        powers.append(integrated_power(generator_entry))

        decoder_fields: list[ComplexField] = []
        for layer in range(4):
            generator_current = self._phase(generator_current, self.decoder_phase[layer])
            propagated = self._propagate(
                generator_current,
                "decoder",
                layer + 1,
                input_z_m=(3 + layer) * LAYER_SPACING_M,
            )
            generator_current = propagated.tensor
            if field_callback is None:
                decoder_fields.append(propagated)
            else:
                field_callback(f"decoder_{layer + 1}", propagated)
            powers.append(integrated_power(self._active_field(
                propagated.tensor,
                f"chen_lightgen400_decoder_{layer + 1}_aperture",
                propagated.plane.z_m,
            )))

        camera_subpixel_intensity = _center_crop(generator_current.abs().square(), self.config.active_shape)
        camera_intensity = integrate_camera_pixel_intensity(
            camera_subpixel_intensity, self.config.pixel_sampling_factor
        )
        camera_plane = PlaneSpec(
            "chen_lightgen400_raw_cmos",
            ACTIVE_SHAPE,
            (PIXEL_PITCH_M, PIXEL_PITCH_M),
            WAVELENGTH_M,
            7.0 * LAYER_SPACING_M,
            "camera",
        )
        camera_area = PIXEL_PITCH_M**2
        physical_w_per_pixel = camera_intensity * camera_area
        if field_callback is not None:
            camera_full_plane = self.grid_plane.with_plane(
                "chen_lightgen400_camera_field", z_m=7.0 * LAYER_SPACING_M
            )
            field_callback("camera_full_field", self._field(generator_current, camera_full_plane, "camera_field"))
        camera_roi_power_w = physical_w_per_pixel.sum(dim=(1, 2, 3))
        camera_full_field_power_w = integrated_power(self._field(
            generator_current, self.grid_plane.with_plane("chen_lightgen400_camera_full_plane", z_m=7.0 * LAYER_SPACING_M),
            "camera_full_plane",
        ))
        return {
            "raw_intensity_w_per_m2": camera_intensity,
            "physical_w_per_pixel": physical_w_per_pixel,
            "camera_plane": camera_plane,
            "ols_coefficients": coefficients,
            "input_power_w": input_power_w,
            "ols_input_power_w": ols_input_power_w,
            "ols_output_power_w": ols_output_power_w,
            "plane_powers_w": torch.stack(powers, dim=1),
            "encoder_fields": tuple(encoder_fields),
            "decoder_fields": tuple(decoder_fields),
            "ols_input_field": ols_input if field_callback is None else None,
            "ols_output_field": ols_output if field_callback is None else None,
            "generator_entry_field": generator_entry if field_callback is None else None,
            "camera_roi_power_w": camera_roi_power_w,
            "camera_full_field_power_w": camera_full_field_power_w,
            "camera_outside_roi_power_w": (camera_full_field_power_w - camera_roi_power_w).clamp_min(0.0),
            "display_intensity": camera_intensity,
        }

    def forward_from_ols(
        self,
        coefficients: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        field_callback: Callable[[str, ComplexField], None] | None = None,
    ) -> dict[str, object]:
        if coefficients.ndim != 2 or coefficients.shape[1] != OLS_PORT_COUNT:
            raise ValueError("generated OLS input must have shape [B,100]")
        if not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
            raise ValueError("generated OLS coefficients must be finite complex values")
        reemitted = self.ols.reemit_coefficients(self.ols.transmit_coefficients(coefficients))
        ols_plane = self.ols_input_plane.with_plane("chen_lightgen400_generated_ols_output")
        entered = angular_spectrum_propagate(
            ComplexField(reemitted.unsqueeze(1), ols_plane, OperationProvenance("generated_ols_reemission", ols_plane.plane_id, None)),
            LAYER_SPACING_M,
            "chen_lightgen400_generated_decoder_entrance",
        ).tensor
        current = entered
        generator_plane = self.grid_plane.with_plane(
            "chen_lightgen400_generated_decoder_entrance", z_m=3.0 * LAYER_SPACING_M
        )
        generator_entry = self._field(
            entered,
            generator_plane,
            "generated_ols_to_decoder_entrance",
        )
        if field_callback is not None:
            field_callback("generator_entry", generator_entry)
        generator_entry_active = self._active_field(
            entered, "chen_lightgen400_generated_decoder_entrance_aperture", 3.0 * LAYER_SPACING_M
        )
        ols_reemitted_power_w = integrated_power(self._field(reemitted.unsqueeze(1), ols_plane, "generated_ols_output"))
        del reemitted, entered
        decoder_fields: list[ComplexField] = []
        powers = [integrated_power(generator_entry_active)]
        for layer in range(4):
            current = self._phase(current, self.decoder_phase[layer])
            propagated = self._propagate(
                current,
                "generated_decoder",
                layer + 1,
                input_z_m=(3 + layer) * LAYER_SPACING_M,
            )
            current = propagated.tensor
            if return_diagnostics:
                if field_callback is None:
                    decoder_fields.append(propagated)
                else:
                    field_callback(f"decoder_{layer + 1}", propagated)
                active = self._active_field(
                    current,
                    f"chen_lightgen400_generated_decoder_{layer + 1}_aperture",
                    propagated.plane.z_m,
                )
                powers.append(integrated_power(active))
        camera_subpixel_intensity = _center_crop(current.abs().square(), self.config.active_shape)
        intensity = integrate_camera_pixel_intensity(camera_subpixel_intensity, self.config.pixel_sampling_factor)
        if field_callback is not None:
            camera_plane = self.grid_plane.with_plane(
                "chen_lightgen400_generated_camera_field", z_m=7.0 * LAYER_SPACING_M
            )
            field_callback("camera_full_field", self._field(current, camera_plane, "camera_field"))
        result: dict[str, object] = {
            "raw_intensity_w_per_m2": intensity,
            "physical_w_per_pixel": intensity * PIXEL_PITCH_M**2,
        }
        if return_diagnostics:
            camera_roi_power_w = (intensity * PIXEL_PITCH_M**2).sum(dim=(1, 2, 3))
            camera_full_field_power_w = integrated_power(self._field(
                current, self.grid_plane.with_plane("chen_lightgen400_generated_camera_full_plane",
                                                    z_m=7.0 * LAYER_SPACING_M),
                "generated_camera_full_plane",
            ))
            result.update(
                {
                    "ols_modal_power_w": coefficients.abs().square().sum(dim=1),
                    "ols_reemitted_power_w": ols_reemitted_power_w,
                    "camera_roi_power_w": camera_roi_power_w,
                    "camera_full_field_power_w": camera_full_field_power_w,
                    "camera_outside_roi_power_w": (camera_full_field_power_w - camera_roi_power_w).clamp_min(0.0),
                    "generator_entry_field": generator_entry if field_callback is None else None,
                    "generator_entry_active_field": generator_entry_active,
                    "decoder_fields": tuple(decoder_fields),
                    "plane_powers_w": torch.stack(powers, dim=1),
                }
            )
        return result

    @torch.no_grad()
    def quantized_phase_copy(self) -> "ChenLightGen400Baseline":
        quantized = ChenLightGen400Baseline(self.config)
        quantized.load_state_dict(self.state_dict())
        levels = PHASE_LEVELS.to(device=quantized.encoder_phase.device)
        for phase in (quantized.encoder_phase, quantized.decoder_phase):
            distance = torch.remainder(phase.unsqueeze(-1) - levels, 2.0 * math.pi).abs()
            circular_distance = torch.minimum(distance, 2.0 * math.pi - distance)
            indices = circular_distance.argmin(dim=-1)
            phase.copy_(levels[indices])
        return quantized
