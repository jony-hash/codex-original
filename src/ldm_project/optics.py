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
    dist_lens_to_cam: float,
    focus_image_distance: float,
    blur_gain: float,
) -> float:
    if blur_gain < 0.0:
        raise ValueError("blur_gain must be non-negative.")
    if focus_image_distance <= 0.0:
        raise ValueError("focus_image_distance must be positive.")
    defocus = abs(dist_lens_to_cam - focus_image_distance)
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


class Ideal4fRelay(nn.Module):
    def __init__(self, focal_length: float, wavelength: float, pixel_size: float) -> None:
        super().__init__()
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.pixel_size = pixel_size

    def forward(self, field: torch.Tensor, propagation_method: str = "fresnel") -> torch.Tensor:
        propagate = _resolve_propagator(propagation_method)
        field = propagate(field, self.focal_length, self.wavelength, self.pixel_size)
        field = apply_thin_lens(field, self.focal_length, self.wavelength, self.pixel_size)
        field = propagate(field, 2.0 * self.focal_length, self.wavelength, self.pixel_size)
        field = apply_thin_lens(field, self.focal_length, self.wavelength, self.pixel_size)
        field = propagate(field, self.focal_length, self.wavelength, self.pixel_size)
        return field


class IdealRelayLens(nn.Module):
    def __init__(self, focal_length: float, wavelength: float, pixel_size: float) -> None:
        super().__init__()
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.pixel_size = pixel_size

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return apply_thin_lens(field, self.focal_length, self.wavelength, self.pixel_size)


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
        optics = physics.optics
        macro = physics.macro_optics

        self.m1 = IdealIdentityElement("m1")
        self.beam_expander = IdealIdentityElement("beam_expander")
        self.m2 = IdealIdentityElement("m2")
        self.pf1 = IdealIdentityElement("pf1")
        self.pbs = IdealIdentityElement("pbs")
        self.bs_in = IdealIdentityElement("bs_in")
        self.slm = ReflectiveAmplitudeSLM()
        self.bs_out = IdealIdentityElement("bs_out")
        self.m3 = IdealIdentityElement("m3")
        self.relay_4f = Ideal4fRelay(
            focal_length=macro.f_4f,
            wavelength=optics.wavelength,
            pixel_size=optics.pixel_size,
        )
        self.pf2 = IdealIdentityElement("pf2")

    def forward_field(self, slm_transfer: torch.Tensor, propagation_method: str = "fresnel") -> torch.Tensor:
        macro = self.physics.macro_optics
        optics = self.physics.optics
        propagate = _resolve_propagator(propagation_method)

        field = torch.ones_like(slm_transfer, dtype=torch.complex64)
        field = propagate(field, macro.dist_laser_to_m1, optics.wavelength, optics.pixel_size)
        field = self.m1(field)
        field = self.beam_expander(field)
        field = propagate(field, macro.dist_m1_to_m2, optics.wavelength, optics.pixel_size)
        field = self.m2(field)
        field = propagate(field, macro.dist_m2_to_pf1, optics.wavelength, optics.pixel_size)
        field = self.pf1(field)
        field = propagate(field, macro.dist_pf1_to_pbs, optics.wavelength, optics.pixel_size)
        field = self.pbs(field)
        field = propagate(field, macro.dist_pbs_to_bs, optics.wavelength, optics.pixel_size)
        field = self.bs_in(field)
        field = propagate(field, macro.dist_bs_to_slm, optics.wavelength, optics.pixel_size)
        field = self.slm(slm_transfer, field)
        field = self.bs_out(field)
        field = propagate(field, macro.dist_slm_to_m3, optics.wavelength, optics.pixel_size)
        field = self.m3(field)
        field = propagate(field, macro.dist_m3_to_4f, optics.wavelength, optics.pixel_size)
        field = self.relay_4f(field, propagation_method=propagation_method)
        field = propagate(field, macro.dist_4f_to_pf2, optics.wavelength, optics.pixel_size)
        field = self.pf2(field)
        return field

    def forward(self, slm_transfer: torch.Tensor) -> torch.Tensor:
        field = self.forward_field(slm_transfer, propagation_method="fresnel")
        return normalize_intensity(field, self.physics.readout.normalize_percentile)


class BackOpticalPath(nn.Module):
    def __init__(self, physics: PhysicsConfig, propagation_method: str = "fresnel") -> None:
        super().__init__()
        self.physics = physics
        self.propagation_method = propagation_method
        _resolve_propagator(propagation_method)
        macro = physics.macro_optics
        readout = physics.readout
        if macro.dist_plane_to_lens == 0.0:
            raise ValueError("macro_optics.dist_plane_to_lens must be non-zero for camera imaging.")
        if not 0.0 <= readout.camera_magnification_blend <= 1.0:
            raise ValueError("readout.camera_magnification_blend must be within [0, 1].")
        if readout.camera_blur_gain < 0.0:
            raise ValueError("readout.camera_blur_gain must be non-negative.")

        lens_denominator = (1.0 / macro.f_relay) - (1.0 / macro.dist_plane_to_lens)
        if math.isclose(lens_denominator, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("macro_optics.dist_plane_to_lens must differ from macro_optics.f_relay.")
        self.focus_image_distance = 1.0 / lens_denominator
        if self.focus_image_distance <= 0.0:
            raise ValueError("Relay lens image distance must be positive for camera imaging.")

        self.camera_magnification = -macro.dist_lens_to_cam / macro.dist_plane_to_lens
        self.camera_magnification_blend = readout.camera_magnification_blend
        self.camera_blur_gain = readout.camera_blur_gain

    def forward(self, chip_field: torch.Tensor) -> dict[str, torch.Tensor]:
        macro = self.physics.macro_optics
        optics = self.physics.optics
        normalize_percentile = self.physics.readout.normalize_percentile
        propagate = _resolve_propagator(self.propagation_method)

        field = chip_field.to(torch.complex64)
        probe_field = propagate(field, macro.dist_chip_to_plane, optics.wavelength, optics.pixel_size)
        probe_plane_loss = normalize_intensity(probe_field, normalize_percentile, clamp=False)
        camera_field = ideal_image_field(probe_field, self.camera_magnification, self.camera_magnification_blend)
        camera_sigma_px = _camera_blur_sigma_pixels(
            self.camera_magnification,
            macro.dist_lens_to_cam,
            self.focus_image_distance,
            self.camera_blur_gain,
        )
        # 模糊放在强度域，表示传感器看到的是失焦后的能量扩散，而不是再引入一段额外复场传播。
        camera_intensity = _gaussian_blur_intensity(camera_field.abs().pow(2), camera_sigma_px)
        camera_plane_loss = normalize_real_intensity(camera_intensity, normalize_percentile, clamp=False)
        return {
            "chip_amplitude": field.abs(),
            "chip_phase": torch.angle(field),
            "probe_plane": probe_plane_loss.clamp(0.0, 1.0),
            "probe_plane_loss": probe_plane_loss,
            "camera_plane": camera_plane_loss.clamp(0.0, 1.0),
            "camera_plane_loss": camera_plane_loss,
        }
