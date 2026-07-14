from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PhysicsConfig
from .devices import IdealIdentityElement, ReflectiveAmplitudeSLM
from .propagation import (
    _frequency_grid,
    angular_spectrum_propagate,
    apply_thin_lens,
    band_limited_angular_spectrum_propagate,
    fresnel_propagate,
    normalize_intensity,
    normalize_real_intensity,
)


def resize_complex_field_diagnostic(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Diagnostic-only complex interpolation; never a formal plane transition."""
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


# Kept only for historical direct imports; formal relays use explicit B=0 mappings.
_resize_complex_field = resize_complex_field_diagnostic


def _center_embed_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = size
    if target_height < height or target_width < width:
        raise ValueError("complex-field propagation grid must contain the source field")
    if (height, width) == size:
        return field
    padded = field.new_zeros(*field.shape[:-2], target_height, target_width)
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    padded[..., top : top + height, left : left + width] = field
    return padded


def _sample_complex_field_to_fixed_grid(field: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    if math.isclose(scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return field

    batch_size = field.shape[0]
    theta = torch.tensor(
        [[[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0]]],
        device=field.device,
        dtype=field.real.dtype,
    ).expand(batch_size, -1, -1)
    output_size = [int(dim) for dim in field.real.shape]
    grid = F.affine_grid(theta, output_size, align_corners=False)
    real = F.grid_sample(field.real, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    imag = F.grid_sample(field.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.complex(real, imag)


def _gaussian_blur_complex_field(field: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply a Gaussian blur independently to the real and imag parts of a complex field.

    For sigma=0 the field is returned unchanged (identity).
    The same real-valued kernel is applied to both channels so the PSF
    preserves the coherent nature of the field (linear in complex amplitude).

    Args:
        field: complex tensor (B, C, H, W) in complex64/complex128.
        sigma_px: standard deviation in pixels.
    Returns:
        Blurred complex field, same shape and dtype as input.
    """
    if sigma_px <= 1e-6:
        return field
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    coords = torch.arange(-radius, radius + 1, device=field.device, dtype=field.real.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma_px).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1)
    real_blurred = F.conv2d(field.real, kernel, padding=radius, groups=field.shape[1])
    imag_blurred = F.conv2d(field.imag, kernel, padding=radius, groups=field.shape[1])
    return torch.complex(real_blurred, imag_blurred)


def _gaussian_blur_intensity(intensity: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px <= 1e-6:
        return intensity

    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    coords = torch.arange(-radius, radius + 1, device=intensity.device, dtype=intensity.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma_px).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(intensity.shape[1], -1, -1, -1)
    return F.conv2d(intensity, kernel, padding=radius, groups=intensity.shape[1])


def ideal_image_field(field: torch.Tensor, magnification: float, magnification_blend: float) -> torch.Tensor:
    if magnification == 0.0:
        raise ValueError("magnification must be non-zero.")
    if not 0.0 <= magnification_blend <= 1.0:
        raise ValueError("magnification_blend must be within [0, 1].")

    effective_scale = abs(magnification) ** magnification_blend
    # 固定输出网格上只保留部分几何放大量，避免全量放大时主体被固定视场大面积裁掉。
    image_field = _sample_complex_field_to_fixed_grid(field, effective_scale)
    # camera 输出采用与 probe 对齐的屏幕坐标约定，因此这里不再额外做倒像翻转。
    return image_field


def _camera_blur_sigma_pixels(
    magnification: float,
    dist_relay_lens_to_cmos: float,
    focus_image_distance: float,
    blur_gain: float,
) -> float:
    if blur_gain < 0.0:
        raise ValueError("blur_gain must be non-negative.")
    if focus_image_distance <= 0.0:
        raise ValueError("focus_image_distance must be positive.")
    defocus = abs(dist_relay_lens_to_cmos - focus_image_distance)
    return blur_gain * abs(magnification) * defocus / focus_image_distance


def _scaled_imaging_field(
    field: torch.Tensor,
    *,
    magnification: float,
    output_size: tuple[int, int],
    input_pixel_size: float,
    output_pixel_size: float,
    wavelength: float,
    c_abcd: float,
    pupil_diameter: float | None = None,
    pupil_focal_length: float | None = None,
) -> torch.Tensor:
    """Apply the B=0 paraxial imaging operator on a new physical sampling grid."""
    height, width = output_size
    dtype = field.real.dtype
    y = (torch.arange(height, device=field.device, dtype=dtype) - height / 2.0) * output_pixel_size
    x = (torch.arange(width, device=field.device, dtype=dtype) - width / 2.0) * output_pixel_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    source_x = xx / magnification
    source_y = yy / magnification
    source_height, source_width = field.shape[-2:]
    source_extent_y = source_height * input_pixel_size
    source_extent_x = source_width * input_pixel_size
    grid_x = 2.0 * source_x / source_extent_x
    grid_y = 2.0 * source_y / source_extent_y
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(field.shape[0], -1, -1, -1)
    real = F.grid_sample(field.real, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    imag = F.grid_sample(field.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    phase = (2.0 * math.pi / wavelength) * c_abcd / (2.0 * magnification) * (xx.square() + yy.square())
    transfer = torch.exp(1j * phase).to(field.dtype).unsqueeze(0).unsqueeze(0)
    image = torch.complex(real, imag) * transfer / abs(magnification)
    if pupil_diameter is not None and pupil_focal_length is not None:
        fy, fx = _frequency_grid(height, width, output_pixel_size, field.device, dtype)
        cutoff = pupil_diameter / (2.0 * pupil_focal_length * wavelength)
        pupil = ((fx.square() + fy.square()).sqrt() <= cutoff).to(image.dtype)
        image = torch.fft.ifft2(torch.fft.fft2(image) * pupil)
    return image


def coherent_imaging_4f_field(
    field: torch.Tensor,
    *,
    magnification: float,
    output_size: tuple[int, int],
    input_pixel_size: float,
    output_pixel_size: float,
    wavelength: float,
    c_abcd: float,
    pupil_diameter: float,
    pupil_focal_length: float,
) -> torch.Tensor:
    """Coordinate-aware coherent B=0 4f relay with an explicit pupil.

    The relay is evaluated as a band-limited physical imaging operator.  The
    SLM field is first filtered by the object-side coherent pupil and then
    evaluated on encoder coordinates by exact evaluation of the *periodic*
    band-limited discrete Fourier series.  This is a coordinate transform,
    not a tensor resize and not a finite cardinal-sinc approximation.

    The finite computational window has periodic Fourier boundary conditions.
    It therefore represents the sampled SLM plane only; it must not be read
    as a model of an exterior finite SLM aperture.
    """
    if magnification == 0.0 or pupil_diameter <= 0.0 or pupil_focal_length <= 0.0:
        raise ValueError("4f relay requires non-zero magnification and positive pupil/focal length")
    height, width = output_size
    dtype = field.real.dtype
    y = (torch.arange(height, device=field.device, dtype=dtype) - (height - 1) / 2.0) * output_pixel_size
    x = (torch.arange(width, device=field.device, dtype=dtype) - (width - 1) / 2.0) * output_pixel_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    source_x = xx / magnification
    source_y = yy / magnification
    source_h, source_w = field.shape[-2:]
    # A true free(f1)-L1-free(f1+f2)-L2-free(f2) 4f relay has C=0.  The
    # public c_abcd argument remains for compatibility and for the general
    # B=0 Collins phase expression, but the physical focal length is supplied
    # by the explicit pupil plane.  Deriving f2 from C is invalid for this
    # actual 4f geometry because it would divide by zero.
    f2 = pupil_focal_length
    f1 = -f2 / magnification
    if f1 <= 0.0:
        raise ValueError("4f relay magnification/pupil focal length imply non-positive f1")
    object_cutoff = pupil_diameter * abs(magnification) / (2.0 * f2 * wavelength)
    fy_in, fx_in = _frequency_grid(source_h, source_w, input_pixel_size, field.device, dtype)
    object_pupil = ((fx_in.square() + fy_in.square()).sqrt() <= object_cutoff * (1.0 + 1.0e-6)).to(field.dtype)
    spectrum = torch.fft.fft2(field.contiguous()) * object_pupil
    # torch.fft uses coordinates n * dx.  Physical centred coordinates differ
    # by the same constant origin at all samples, so explicitly retain that
    # offset when evaluating the Fourier series at x_object=x_image/M.
    source_y0 = -(source_h - 1) * input_pixel_size / 2.0
    source_x0 = -(source_w - 1) * input_pixel_size / 2.0
    fy_axis = torch.fft.fftfreq(source_h, d=input_pixel_size, device=field.device, dtype=dtype)
    fx_axis = torch.fft.fftfreq(source_w, d=input_pixel_size, device=field.device, dtype=dtype)
    phase_y = torch.exp(2j * math.pi * (source_y[:, 0, None] - source_y0) * fy_axis[None, :]).to(field.dtype)
    phase_x = torch.exp(2j * math.pi * (source_x[0, :, None] - source_x0) * fx_axis[None, :]).to(field.dtype)
    image_field = torch.einsum("...ij,ui,vj->...uv", spectrum, phase_y, phase_x)
    image_field = image_field / float(source_h * source_w)
    phase = (2.0 * math.pi / wavelength) * c_abcd / (2.0 * magnification) * (xx.square() + yy.square())
    image = image_field * torch.exp(1j * phase).to(field.dtype).unsqueeze(0).unsqueeze(0)
    image = image / abs(magnification)
    fy, fx = _frequency_grid(height, width, output_pixel_size, field.device, dtype)
    cutoff = pupil_diameter / (2.0 * pupil_focal_length * wavelength)
    pupil = ((fx.square() + fy.square()).sqrt() <= cutoff * (1.0 + 1.0e-6)).to(image.dtype)
    return torch.fft.ifft2(torch.fft.fft2(image) * pupil)


def _resolve_propagator(propagation_method: str):
    if propagation_method == "fresnel":
        return fresnel_propagate
    if propagation_method == "angular_spectrum":
        return angular_spectrum_propagate
    raise ValueError(
        "Unsupported propagation_method: "
        f"{propagation_method}; expected fresnel or angular_spectrum."
    )


class FrontOpticalPath(nn.Module):
    def __init__(self, physics: PhysicsConfig) -> None:
        super().__init__()
        self.physics = physics
        self.front_path_model = physics.front_path_model

        self.m1 = IdealIdentityElement("m1")
        self.bs_in = IdealIdentityElement("bs_in")
        self.slm = ReflectiveAmplitudeSLM()
        self.bs_out = IdealIdentityElement("bs_out")
        self.m3 = IdealIdentityElement("m3")
        self.lightgen_input = IdealIdentityElement("lightgen_input")

    def forward_field(self, slm_transfer: torch.Tensor, propagation_method: str = "fresnel") -> torch.Tensor:
        field, _ = self.forward_field_with_diagnostics(slm_transfer, propagation_method=propagation_method)
        return field

    def forward_field_with_diagnostics(
        self,
        slm_transfer: torch.Tensor,
        propagation_method: str = "fresnel",
        propagation_grid_size: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if self.front_path_model == "relay_identity":
            # Ideal relay: plane-wave illumination + amplitude SLM modulation only.
            # No free-space propagation — the SLM pattern is imaged directly to the
            # encoder input plane via an ideal 4f relay system.
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            slm_field = self.slm(slm_transfer, field)
            return slm_field, {"operation": "relay_identity", "_slm_field": slm_field}

        if self.front_path_model == "relay_coherent_psf":
            # Coherent PSF relay: same as relay_identity (plane-wave + SLM) but
            # then apply a Gaussian blur to the complex field (real/imag separately).
            # sigma=0 is strictly equivalent to relay_identity.
            sigma_px = self.physics.front_relay_psf_sigma_px
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            field = self.slm(slm_transfer, field)
            return _gaussian_blur_complex_field(field, sigma_px), {"operation": "relay_coherent_psf", "_slm_field": field}

        if self.front_path_model == "relay_intensity_psf":
            # Intensity relay: image SLM intensity to the encoder plane, with an
            # optional incoherent PSF, then expose it as a real nonnegative field.
            slm_intensity = slm_transfer.abs().pow(2).float()
            relayed_intensity = _gaussian_blur_intensity(
                slm_intensity,
                self.physics.front_relay_psf_sigma_px,
            )
            amplitude = relayed_intensity.clamp_min(0.0).sqrt().to(slm_transfer.real.dtype)
            slm_field = torch.complex(slm_transfer.abs(), torch.zeros_like(slm_transfer.real))
            return torch.complex(amplitude, torch.zeros_like(amplitude)), {"operation": "relay_intensity_psf", "_slm_field": slm_field}

        if self.front_path_model == "coherent_imaging_4f":
            macro = self.physics.macro_optics
            optics = self.physics.optics
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            field = angular_spectrum_propagate(field, macro.dist_laser_to_m1, optics.wavelength, optics.pixel_size)
            field = self.m1(field)
            field = angular_spectrum_propagate(field, macro.dist_m1_to_bs, optics.wavelength, optics.pixel_size)
            field = self.bs_in(field)
            field = angular_spectrum_propagate(field, macro.dist_bs_to_slm, optics.wavelength, optics.pixel_size)
            slm_field = self.slm(slm_transfer, field)
            f1 = macro.front_lens1_focal_length
            f2 = macro.front_lens2_focal_length
            relay_length = (
                macro.dist_slm_to_bs
                + macro.dist_bs_to_m3
                + macro.dist_m3_to_front_lens1
                + macro.dist_front_lens1_to_lens2
                + macro.dist_front_lens2_to_encoder
            )
            if not math.isclose(relay_length, 2.0 * (f1 + f2), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError("front 4f relay mechanical distance must equal 2*(f1+f2)")
            magnification = -f2 / f1
            c_abcd = 0.0
            field = coherent_imaging_4f_field(
                slm_field,
                magnification=magnification,
                output_size=(100, 100),
                input_pixel_size=optics.pixel_size,
                output_pixel_size=2.0e-6,
                wavelength=optics.wavelength,
                c_abcd=c_abcd,
                pupil_diameter=min(macro.front_lens1_clear_aperture, macro.front_lens2_clear_aperture),
                pupil_focal_length=f2,
            )
            relay_trace = [
                {"from": "SLM", "to": "BS", "distance_m": macro.dist_slm_to_bs, "operation": "fold_return"},
                {"from": "BS", "to": "M3", "distance_m": macro.dist_bs_to_m3, "operation": "fold"},
                {"from": "M3", "to": "L1", "distance_m": macro.dist_m3_to_front_lens1, "operation": "free_space"},
                {"from": "L1", "to": "L2", "distance_m": macro.dist_front_lens1_to_lens2, "operation": "free_space"},
                {"from": "L2", "to": "Encoder", "distance_m": macro.dist_front_lens2_to_encoder, "operation": "free_space"},
            ]
            return field, {
                "operation": "coherent_imaging_4f",
                "segments_m": [macro.dist_laser_to_m1, macro.dist_m1_to_bs, macro.dist_bs_to_slm]
                + [item["distance_m"] for item in relay_trace],
                "relay_trace": relay_trace,
                "front_relay": {"f1_m": f1, "f2_m": f2, "magnification": magnification, "A": magnification, "B": 0.0, "C": c_abcd, "D": -f1 / f2, "pupil_diameter_m": min(macro.front_lens1_clear_aperture, macro.front_lens2_clear_aperture)},
                "_slm_field": slm_field,
            }

        # Default free-space front path.
        macro = self.physics.macro_optics
        optics = self.physics.optics
        propagate = _resolve_propagator(propagation_method)

        if not self.physics.physical_forward_contract:
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            field = propagate(field, macro.dist_laser_to_m1, optics.wavelength, optics.pixel_size)
            field = self.m1(field)
            field = propagate(field, macro.dist_m1_to_bs, optics.wavelength, optics.pixel_size)
            field = self.bs_in(field)
            field = propagate(field, macro.dist_bs_to_slm, optics.wavelength, optics.pixel_size)
            field = self.slm(slm_transfer, field)
            field = propagate(field, macro.dist_slm_to_bs, optics.wavelength, optics.pixel_size)
            field = self.bs_out(field)
            field = propagate(field, macro.dist_bs_to_m3, optics.wavelength, optics.pixel_size)
            field = self.m3(field)
            field = propagate(field, macro.dist_m3_to_lg, optics.wavelength, optics.pixel_size)
            return self.lightgen_input(field), {"operation": "legacy_free_space", "_slm_field": field}

        segments = [
            macro.dist_laser_to_m1,
            macro.dist_m1_to_bs,
            macro.dist_bs_to_slm,
            macro.dist_slm_to_bs,
            macro.dist_bs_to_m3,
            macro.dist_m3_to_lg,
        ]
        # Large CPU FFTs in some PyTorch/MKL builds lose complex64 energy at
        # 2048².  Keep the physical contract numerically auditable in CPU
        # smoke runs by using complex128; CUDA keeps the model's native dtype.
        field_dtype = torch.complex128 if slm_transfer.device.type == "cpu" else torch.complex64
        field = torch.ones_like(slm_transfer, dtype=field_dtype)
        field = propagate(field, segments[0], optics.wavelength, optics.pixel_size)
        field = self.m1(field)
        field = propagate(field, segments[1], optics.wavelength, optics.pixel_size)
        field = self.bs_in(field)
        field = propagate(field, segments[2], optics.wavelength, optics.pixel_size)
        field = self.slm(slm_transfer, field)
        slm_field = field
        grid_size = int(propagation_grid_size or macro.propagation_grid_size)
        field = _center_embed_complex_field(field, (grid_size, grid_size))
        segment_energy = [float(field.abs().square().sum().detach().cpu())]
        field = band_limited_angular_spectrum_propagate(field, segments[3], optics.wavelength, optics.pixel_size)
        segment_energy.append(float(field.abs().square().sum().detach().cpu()))
        field = self.bs_out(field)
        field = band_limited_angular_spectrum_propagate(field, segments[4], optics.wavelength, optics.pixel_size)
        segment_energy.append(float(field.abs().square().sum().detach().cpu()))
        field = self.m3(field)
        field = band_limited_angular_spectrum_propagate(field, segments[5], optics.wavelength, optics.pixel_size)
        segment_energy.append(float(field.abs().square().sum().detach().cpu()))
        field = self.lightgen_input(field)
        return field, {
            "operation": "macro_free_space",
            "segments_m": segments,
            "post_slm_distance_m": sum(segments[3:]),
            "propagation_grid_size": grid_size,
            "pixel_size_m": optics.pixel_size,
            "segment_energy": segment_energy,
            "compute_dtype": str(field.dtype),
            "_slm_field": slm_field,
        }

    def forward(self, slm_transfer: torch.Tensor) -> torch.Tensor:
        field = self.forward_field(slm_transfer, propagation_method="fresnel")
        return normalize_intensity(field, self.physics.readout.normalize_percentile)


class BackOpticalPath(nn.Module):
    def __init__(self, physics: PhysicsConfig, propagation_method: str = "fresnel") -> None:
        super().__init__()
        self.physics = physics
        self.propagation_method = propagation_method
        _resolve_propagator(propagation_method)
        readout = physics.readout
        if not 0.0 <= readout.camera_magnification_blend <= 1.0:
            raise ValueError("readout.camera_magnification_blend must be within [0, 1].")
        if readout.camera_blur_gain < 0.0:
            raise ValueError("readout.camera_blur_gain must be non-negative.")

        if physics.back_path_model in {"relay_identity", "relay_psf"}:
            # relay_identity/relay_psf: no free-space propagation or lens.
            # All macro-optic validations are skipped — distances not used.
            self.focus_image_distance = 0.0
            self.camera_magnification = 1.0
            self.camera_magnification_blend = readout.camera_magnification_blend
            self.camera_blur_gain = readout.camera_blur_gain
        else:
            macro = physics.macro_optics
            if macro.dist_lg_to_relay_lens == 0.0:
                raise ValueError("macro_optics.dist_lg_to_relay_lens must be non-zero for camera imaging.")
            lens_denominator = (1.0 / macro.f_relay) - (1.0 / macro.dist_lg_to_relay_lens)
            if math.isclose(lens_denominator, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("macro_optics.dist_lg_to_relay_lens must differ from macro_optics.f_relay.")
            self.focus_image_distance = 1.0 / lens_denominator
            if self.focus_image_distance <= 0.0:
                raise ValueError("Relay lens image distance must be positive for camera imaging.")
            self.camera_magnification = -macro.dist_relay_lens_to_cmos / macro.dist_lg_to_relay_lens
            self.camera_magnification_blend = readout.camera_magnification_blend
            self.camera_blur_gain = readout.camera_blur_gain

    def forward(
        self,
        chip_field: torch.Tensor,
        *,
        input_pixel_size: float | None = None,
        output_size: tuple[int, int] | None = None,
        output_pixel_size: float | None = None,
    ) -> dict[str, torch.Tensor | dict[str, object] | float | bool]:
        normalize_percentile = self.physics.readout.normalize_percentile
        field = chip_field.to(torch.complex64)

        if self.physics.back_path_model == "relay_identity":
            # Ideal relay / readout: chip intensity imaged directly to camera plane,
            # no free-space propagation, no lens phase, no defocus blur.
            camera_intensity_raw = field.abs().pow(2)
            camera_plane_loss = normalize_real_intensity(camera_intensity_raw, normalize_percentile, clamp=False)
            camera_plane = camera_plane_loss.clamp(0.0, 1.0)
            # Probe plane mirrors camera plane as ideal-readout diagnostic.
            probe_plane_loss = camera_plane_loss
            return {
                "chip_amplitude": field.abs(),
                "chip_phase": torch.angle(field),
                "probe_plane": camera_plane,
                "probe_plane_loss": probe_plane_loss,
                "camera_field_real": field.real,
                "camera_field_imag": field.imag,
                "camera_phase": torch.angle(field),
                "camera_field_available": True,
                "camera_phase_available": True,
                "camera_intensity_raw": camera_intensity_raw,
                "camera_plane": camera_plane,
                "camera_plane_loss": camera_plane_loss,
            }

        if self.physics.back_path_model == "relay_psf":
            chip_intensity_raw = field.abs().pow(2)
            camera_intensity_raw = _gaussian_blur_intensity(
                chip_intensity_raw,
                self.physics.readout.relay_psf_sigma_px,
            )
            camera_plane_loss = normalize_real_intensity(camera_intensity_raw, normalize_percentile, clamp=False)
            camera_plane = camera_plane_loss.clamp(0.0, 1.0)
            return {
                "chip_amplitude": field.abs(),
                "chip_phase": torch.angle(field),
                "probe_plane": camera_plane,
                "probe_plane_loss": camera_plane_loss,
                "camera_field_available": False,
                "camera_phase_available": False,
                "camera_intensity_raw": camera_intensity_raw,
                "camera_plane": camera_plane,
                "camera_plane_loss": camera_plane_loss,
            }

        # --- free_space (default) ---
        macro = self.physics.macro_optics
        optics = self.physics.optics
        if not self.physics.physical_forward_contract:
            propagate = _resolve_propagator(self.propagation_method)
            probe_field = propagate(field, macro.dist_lg_to_relay_lens, optics.wavelength, optics.pixel_size)
            probe_plane_loss = normalize_intensity(probe_field, normalize_percentile, clamp=False)
            lens_field = apply_thin_lens(probe_field, macro.f_relay, optics.wavelength, optics.pixel_size)
            camera_field = propagate(lens_field, macro.dist_relay_lens_to_cmos, optics.wavelength, optics.pixel_size)
            camera_sigma_px = _camera_blur_sigma_pixels(
                self.camera_magnification,
                macro.dist_relay_lens_to_cmos,
                self.focus_image_distance,
                self.camera_blur_gain,
            )
            camera_intensity = _gaussian_blur_intensity(camera_field.abs().pow(2), camera_sigma_px)
            camera_plane_loss = normalize_real_intensity(camera_intensity, normalize_percentile, clamp=False)
            return {
                "chip_amplitude": field.abs(),
                "chip_phase": torch.angle(field),
                "probe_plane": probe_plane_loss.clamp(0.0, 1.0),
                "probe_plane_loss": probe_plane_loss,
                "camera_field_real": camera_field.real,
                "camera_field_imag": camera_field.imag,
                "camera_phase": torch.angle(camera_field),
                "camera_field_available": True,
                "camera_phase_available": True,
                "camera_intensity_raw": camera_intensity,
                "camera_plane": camera_plane_loss.clamp(0.0, 1.0),
                "camera_plane_loss": camera_plane_loss,
            }
        input_pixel_size = float(input_pixel_size or optics.pixel_size)
        output_size = output_size or tuple(int(value) for value in chip_field.shape[-2:])
        output_pixel_size = float(output_pixel_size or optics.pixel_size)
        u = macro.dist_lg_to_relay_lens
        v = macro.dist_relay_lens_to_cmos
        f = macro.f_relay
        a_abcd = 1.0 - v / f
        b_abcd = u + v - u * v / f
        c_abcd = -1.0 / f
        if not math.isclose(b_abcd, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("free_space back path currently requires a B=0 relay imaging contract")
        camera_field = _scaled_imaging_field(
            field,
            magnification=a_abcd,
            output_size=output_size,
            input_pixel_size=input_pixel_size,
            output_pixel_size=output_pixel_size,
            wavelength=optics.wavelength,
            c_abcd=c_abcd,
            pupil_diameter=macro.back_relay_clear_aperture,
            pupil_focal_length=f,
        )
        probe_plane_loss = normalize_intensity(field, normalize_percentile, clamp=False)
        camera_sigma_px = 0.0
        camera_intensity = camera_field.abs().pow(2)
        camera_plane_loss = normalize_real_intensity(camera_intensity, normalize_percentile, clamp=False)
        return {
            "chip_amplitude": field.abs(),
            "chip_phase": torch.angle(field),
            "probe_plane": probe_plane_loss.clamp(0.0, 1.0),
            "probe_plane_loss": probe_plane_loss,
            "camera_field_real": camera_field.real,
            "camera_field_imag": camera_field.imag,
            "camera_phase": torch.angle(camera_field),
            "camera_field_available": True,
            "camera_phase_available": True,
            "camera_intensity_raw": camera_intensity,
            "camera_plane": camera_plane_loss.clamp(0.0, 1.0),
            "camera_plane_loss": camera_plane_loss,
            "camera_magnification": a_abcd,
            "camera_sampling": {
                "shape": list(output_size),
                "pixel_size_x": output_pixel_size,
                "pixel_size_y": output_pixel_size,
                "fov_x": output_size[1] * output_pixel_size,
                "fov_y": output_size[0] * output_pixel_size,
            },
            "camera_operation_trace": [
                {"operation": "free_space", "distance_m": u},
                {"operation": "thin_lens", "focal_length_m": f},
                {"operation": "free_space", "distance_m": v},
            ],
            "camera_abcd": {"A": a_abcd, "B": b_abcd, "C": c_abcd, "D": 1.0 - u / f},
        }
