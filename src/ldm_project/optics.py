from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import PhysicsConfig


def normalize_intensity(
    field: torch.Tensor,
    normalize_percentile: float = 100.0,
    clamp: bool = True,
) -> torch.Tensor:
    intensity = field.abs().pow(2)
    return normalize_real_intensity(intensity, normalize_percentile, clamp=clamp)


def normalize_real_intensity(
    intensity: torch.Tensor,
    normalize_percentile: float = 100.0,
    clamp: bool = True,
) -> torch.Tensor:
    if normalize_percentile >= 100.0:
        scale = intensity.amax(dim=(-2, -1), keepdim=True)
    else:
        quantile = normalize_percentile / 100.0
        # 按每个样本/通道的空间分位数归一化，避免极少数尖峰把整幅图压得过暗。
        scale = torch.quantile(intensity.flatten(-2), quantile, dim=-1, keepdim=True).unsqueeze(-1)
    normalized = intensity / scale.clamp_min(1e-6)
    return normalized.clamp(0.0, 1.0) if clamp else normalized


def _frequency_grid(height: int, width: int, pixel_size: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftfreq(height, d=pixel_size, device=device)
    fx = torch.fft.fftfreq(width, d=pixel_size, device=device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    return fy_grid, fx_grid


def fresnel_propagate(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if distance == 0.0:
        return field

    height, width = field.shape[-2:]
    fy, fx = _frequency_grid(height, width, pixel_size, field.device)
    phase = torch.exp(
        -1j * math.pi * wavelength * distance * (fx.pow(2) + fy.pow(2))
    ).to(field.dtype)
    spectrum = torch.fft.fft2(field)
    return torch.fft.ifft2(spectrum * phase)


def angular_spectrum_propagate(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if distance == 0.0:
        return field

    height, width = field.shape[-2:]
    fy, fx = _frequency_grid(height, width, pixel_size, field.device)
    frequency_radius = (wavelength * fx).pow(2) + (wavelength * fy).pow(2)
    propagating = frequency_radius <= 1.0
    phase_root = (1.0 - frequency_radius).clamp_min(0.0).sqrt()
    phase_angle = 2.0 * math.pi * distance * phase_root / wavelength
    transfer = torch.exp(1j * phase_angle).to(field.dtype)
    transfer = torch.where(propagating, transfer, torch.zeros_like(transfer))
    spectrum = torch.fft.fft2(field)
    return torch.fft.ifft2(spectrum * transfer)


def apply_thin_lens(field: torch.Tensor, focal_length: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    if focal_length == 0.0:
        return field

    height, width = field.shape[-2:]
    y = (torch.arange(height, device=field.device) - height / 2.0) * pixel_size
    x = (torch.arange(width, device=field.device) - width / 2.0) * pixel_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = torch.exp(
        -1j * math.pi * (xx.pow(2) + yy.pow(2)) / (wavelength * focal_length)
    ).to(field.dtype)
    return field * phase


def _resize_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


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


class IdealIdentityElement(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field


class ReflectiveAmplitudeSLM(nn.Module):
    def forward(self, slm_transfer: torch.Tensor, incoming_field: torch.Tensor) -> torch.Tensor:
        # 当前 SLM 采用工程近似：电子控制图先映射成复振幅透过率，再与入射光场逐点相乘。
        return incoming_field * slm_transfer.to(incoming_field.dtype)


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
        if self.front_path_model == "relay_identity":
            # Ideal relay: plane-wave illumination + amplitude SLM modulation only.
            # No free-space propagation — the SLM pattern is imaged directly to the
            # encoder input plane via an ideal 4f relay system.
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            return self.slm(slm_transfer, field)

        if self.front_path_model == "relay_coherent_psf":
            # Coherent PSF relay: same as relay_identity (plane-wave + SLM) but
            # then apply a Gaussian blur to the complex field (real/imag separately).
            # sigma=0 is strictly equivalent to relay_identity.
            sigma_px = self.physics.front_relay_psf_sigma_px
            field = torch.ones_like(slm_transfer, dtype=torch.complex64)
            field = self.slm(slm_transfer, field)
            return _gaussian_blur_complex_field(field, sigma_px)

        # Default free-space front path.
        macro = self.physics.macro_optics
        optics = self.physics.optics
        propagate = _resolve_propagator(propagation_method)

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
        return self.lightgen_input(field)

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

    def forward(self, chip_field: torch.Tensor) -> dict[str, torch.Tensor]:
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
            camera_phase = torch.zeros_like(camera_intensity_raw)
            camera_amplitude = camera_intensity_raw.clamp_min(0.0).sqrt()
            return {
                "chip_amplitude": field.abs(),
                "chip_phase": torch.angle(field),
                "probe_plane": camera_plane,
                "probe_plane_loss": camera_plane_loss,
                "camera_field_real": camera_amplitude,
                "camera_field_imag": torch.zeros_like(camera_amplitude),
                "camera_phase": camera_phase,
                "camera_field_available": False,
                "camera_phase_available": False,
                "camera_intensity_raw": camera_intensity_raw,
                "camera_plane": camera_plane,
                "camera_plane_loss": camera_plane_loss,
            }

        # --- free_space (default) ---
        macro = self.physics.macro_optics
        optics = self.physics.optics
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
        # relay lens 后已经做复场传播；这里保留可选传感器/失焦强度域模糊诊断。
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
