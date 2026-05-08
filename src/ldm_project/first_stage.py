from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .config import FirstStageConfig, PhysicsConfig
from .optics import BackOpticalPath, FrontOpticalPath, normalize_intensity


FIRST_STAGE_MODEL_VERSION = "physical_first_linear_metasurface_vae_v5"


def _group_norm(channels: int) -> nn.GroupNorm:
    group_count = min(8, channels)
    while channels % group_count != 0:
        group_count -= 1
    return nn.GroupNorm(group_count, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            _group_norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class ConfigurableOLS(nn.Module):
    def __init__(self, kind: str = "pointwise_tanh") -> None:
        super().__init__()
        self.kind = kind

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind in {"pointwise_tanh", "tanh"}:
            return torch.tanh(z)
        if self.kind == "sigmoid":
            return torch.sigmoid(z)
        if self.kind == "silu":
            return F.silu(z)
        if self.kind == "gelu":
            return F.gelu(z)
        if self.kind == "relu":
            return F.relu(z)
        if self.kind == "identity":
            return z
        raise ValueError(f"Unsupported OLS kind: {self.kind}")


class MetasurfaceEncoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], input_channels: int = 1) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.input_channels = input_channels
        self.net = nn.Sequential(
            ConvBlock(input_channels, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            _group_norm(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(8, channels // 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class GlobalContextMetasurfaceEncoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], input_channels: int = 1) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.input_channels = input_channels
        self.down1 = ConvBlock(input_channels, c1)
        self.down2 = ConvBlock(c1, c2)
        self.context50 = nn.Sequential(
            DilatedContextBlock(c2, dilation=2),
            ChannelAttention(c2),
        )
        self.down3 = ConvBlock(c2, c3)
        self.context25 = nn.Sequential(
            DilatedContextBlock(c3, dilation=2),
            DilatedContextBlock(c3, dilation=4),
            ChannelAttention(c3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.context50(x)
        x = self.down3(x)
        return self.context25(x)


class MetasurfaceDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], latent_channels: int, tail_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, c3, kernel_size=1),
            _group_norm(c3),
            nn.SiLU(),
        )
        self.up1 = UpsampleConvBlock(c3, c2)
        self.up2 = UpsampleConvBlock(c2, c1)
        self.up3 = UpsampleConvBlock(c1, tail_channels)
        self.image_head = nn.Conv2d(tail_channels, 1, kernel_size=3, padding=1)
        self.optical_field_head = nn.Conv2d(tail_channels, 2, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.project(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        image_recon = torch.sigmoid(self.image_head(h))
        optical_field = torch.tanh(self.optical_field_head(h))
        return {
            "image_recon": image_recon,
            "chip_real": optical_field[:, :1],
            "chip_imag": optical_field[:, 1:],
        }


def _resize_complex_field(field: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(field.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(field.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _flatten_complex_field(field: torch.Tensor) -> torch.Tensor:
    return torch.cat([field.real.flatten(1), field.imag.flatten(1)], dim=1)


def _vector_to_complex_field(vector: torch.Tensor, grid_size: int) -> torch.Tensor:
    batch_size = vector.shape[0]
    half_width = grid_size * grid_size
    real = vector[:, :half_width].reshape(batch_size, 1, grid_size, grid_size)
    imag = vector[:, half_width:].reshape(batch_size, 1, grid_size, grid_size)
    return torch.complex(real, imag)


class LinearMetasurfaceEncoder(nn.Module):
    def __init__(
        self,
        physics: PhysicsConfig,
        image_size: int,
        latent_channels: int,
        latent_spatial_size: int,
        grid_size: int,
        layer_spacing: float,
    ) -> None:
        super().__init__()
        self.input_channels = 2
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.grid_size = grid_size
        self.layer_spacing = layer_spacing
        self.normalize_percentile = physics.readout.normalize_percentile
        grid_features = 2 * grid_size * grid_size
        latent_features = latent_channels * latent_spatial_size * latent_spatial_size
        # 线性 metasurface encoder 暂定只保留一层纯线性映射，非线性只交给 OLS。
        self.linear = nn.Linear(grid_features, 2 * latent_features, bias=False)
        self._initialize_layers()

    def _initialize_layers(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def forward(self, pf2_field: torch.Tensor) -> dict[str, torch.Tensor]:
        field = _resize_complex_field(pf2_field, (self.grid_size, self.grid_size))
        latent_stats = self.linear(_flatten_complex_field(field))
        latent_features = self.latent_channels * self.latent_spatial_size * self.latent_spatial_size
        mu = latent_stats[:, :latent_features].reshape(
            -1,
            self.latent_channels,
            self.latent_spatial_size,
            self.latent_spatial_size,
        )
        logvar = latent_stats[:, latent_features:].reshape(
            -1,
            self.latent_channels,
            self.latent_spatial_size,
            self.latent_spatial_size,
        )
        return {"mu": mu, "logvar": logvar}


class LinearMetasurfaceDecoder(nn.Module):
    def __init__(
        self,
        physics: PhysicsConfig,
        image_size: int,
        latent_channels: int,
        latent_spatial_size: int,
        grid_size: int,
        layer_spacing: float,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.grid_size = grid_size
        self.layer_spacing = layer_spacing
        self.normalize_percentile = physics.readout.normalize_percentile
        latent_features = latent_channels * latent_spatial_size * latent_spatial_size
        grid_features = 2 * grid_size * grid_size
        # decoder 同样只保留一层纯线性映射，避免在 OLS 之外引入可学习非线性。
        self.linear = nn.Linear(latent_features, grid_features, bias=False)
        self._initialize_layers()

    def _initialize_layers(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        chip_grid = _vector_to_complex_field(self.linear(z.flatten(1)), self.grid_size)
        chip_field = _resize_complex_field(chip_grid, (self.image_size, self.image_size))
        return {
            "image_recon": chip_field.real,
            "chip_real": chip_field.real,
            "chip_imag": chip_field.imag,
        }


class OpticalPathVAE(nn.Module):
    def __init__(self, physics: PhysicsConfig, config: FirstStageConfig | None = None) -> None:
        super().__init__()
        self.physics = physics
        self.config = config or FirstStageConfig()
        self._validate_model_config()
        self.front_path = FrontOpticalPath(physics)
        self.uses_linear_metasurface = self.config.optical_encoder_arch == "linear_metasurface"
        self.optical_encoder_input_channels = self._resolve_optical_encoder_input_channels()
        self.optical_encoder = self._build_optical_encoder()
        self.image_fusion_weight = 0.0
        self.image_encoder: MetasurfaceEncoder | None
        self.encoder_fusion: nn.Module
        self.mu_head: nn.Module
        self.logvar_head: nn.Module
        if self.uses_linear_metasurface:
            self.image_encoder = None
            self.encoder_fusion = nn.Identity()
            self.mu_head = nn.Identity()
            self.logvar_head = nn.Identity()
        else:
            encoder_out_channels = self.config.encoder_channels[-1]
            if self.config.encoder_fusion_mode in {"dual_path_concat", "gated_dual_path"}:
                self.image_encoder = MetasurfaceEncoder(self.config.encoder_channels)
            else:
                self.image_encoder = None
            if self.config.encoder_fusion_mode == "dual_path_concat":
                self.encoder_fusion = nn.Sequential(
                    nn.Conv2d(encoder_out_channels * 2, encoder_out_channels, kernel_size=1),
                    _group_norm(encoder_out_channels),
                    nn.SiLU(),
                )
            elif self.config.encoder_fusion_mode == "gated_dual_path":
                self.encoder_fusion = nn.Identity()
                self.set_image_fusion_weight(self.config.image_fusion_start_weight)
            else:
                self.encoder_fusion = nn.Identity()
            self.mu_head = nn.Conv2d(encoder_out_channels, self.config.latent_channels, kernel_size=1)
            self.logvar_head = nn.Conv2d(encoder_out_channels, self.config.latent_channels, kernel_size=1)
        self.ols = ConfigurableOLS(self.config.ols_kind)
        self.metasurface_decoder = self._build_metasurface_decoder()
        self.back_path = BackOpticalPath(
            physics,
            propagation_method="angular_spectrum" if self.uses_linear_metasurface else "fresnel",
        )

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        downsampled = self.config.image_size // 8
        return (self.config.latent_channels, downsampled, downsampled)

    def _validate_model_config(self) -> None:
        if self.config.encoder_fusion_mode not in {"dual_path_concat", "optical_only", "gated_dual_path"}:
            raise ValueError(
                "Unsupported encoder_fusion_mode: "
                f"{self.config.encoder_fusion_mode}; expected dual_path_concat, gated_dual_path, or optical_only."
            )
        for name, value in (
            ("image_fusion_start_weight", self.config.image_fusion_start_weight),
            ("image_fusion_end_weight", self.config.image_fusion_end_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.config.image_fusion_decay_epochs < 0:
            raise ValueError(
                "image_fusion_decay_epochs must be non-negative, "
                f"got {self.config.image_fusion_decay_epochs}"
            )
        if self.config.image_fusion_schedule not in {"linear", "cosine"}:
            raise ValueError(
                "Unsupported image_fusion_schedule: "
                f"{self.config.image_fusion_schedule}; expected linear or cosine."
            )
        if self.config.optical_field_mode != "complex_cartesian":
            raise ValueError(
                "Unsupported optical_field_mode: "
                f"{self.config.optical_field_mode}; expected complex_cartesian."
            )
        valid_input_modes = {
            "pf2_intensity",
            "pf2_intensity_multiscale",
            "pf2_intensity_multiscale_v2",
            "pf2_complex_field",
        }
        if self.config.optical_encoder_input_mode not in valid_input_modes:
            raise ValueError(
                "Unsupported optical_encoder_input_mode: "
                f"{self.config.optical_encoder_input_mode}; expected "
                "pf2_intensity, pf2_intensity_multiscale, pf2_intensity_multiscale_v2, "
                "or pf2_complex_field."
            )
        if self.config.optical_encoder_arch not in {"conv", "global_context", "linear_metasurface"}:
            raise ValueError(
                "Unsupported optical_encoder_arch: "
                f"{self.config.optical_encoder_arch}; expected conv, global_context, or linear_metasurface."
            )
        if self.config.metasurface_decoder_arch not in {"conv", "linear_metasurface"}:
            raise ValueError(
                "Unsupported metasurface_decoder_arch: "
                f"{self.config.metasurface_decoder_arch}; expected conv or linear_metasurface."
            )
        if (self.config.optical_encoder_arch == "linear_metasurface") != (
            self.config.metasurface_decoder_arch == "linear_metasurface"
        ):
            raise ValueError("linear_metasurface encoder and decoder must be enabled together.")
        if self.config.optical_encoder_arch == "linear_metasurface":
            if self.config.encoder_fusion_mode != "optical_only":
                raise ValueError("linear_metasurface first stage only supports encoder_fusion_mode=optical_only.")
            if self.config.optical_encoder_input_mode != "pf2_complex_field":
                raise ValueError("linear_metasurface first stage requires optical_encoder_input_mode=pf2_complex_field.")
            if self.config.latent_channels != 1:
                raise ValueError(
                    "linear_metasurface first stage requires latent_channels=1 so OLS stays single-channel."
                )
        elif self.config.optical_encoder_input_mode == "pf2_complex_field":
            raise ValueError("pf2_complex_field input requires optical_encoder_arch=linear_metasurface.")
        if self.config.ols_kind not in {"pointwise_tanh", "tanh", "sigmoid", "silu", "gelu", "relu", "identity"}:
            raise ValueError(
                "Unsupported OLS kind: "
                f"{self.config.ols_kind}; expected tanh, sigmoid, silu, gelu, relu, or identity."
            )
        if self.config.primary_output not in {"image", "camera"}:
            raise ValueError(f"Unsupported primary_output: {self.config.primary_output}; expected image or camera.")
        if self.config.image_recon_weight < 0.0:
            raise ValueError(f"image_recon_weight must be non-negative, got {self.config.image_recon_weight}")
        if self.config.camera_recon_weight < 0.0:
            raise ValueError(f"camera_recon_weight must be non-negative, got {self.config.camera_recon_weight}")
        if self.config.image_recon_weight + self.config.camera_recon_weight <= 0.0:
            raise ValueError("At least one first-stage reconstruction weight must be positive.")
        if self.config.decoder_tail_channels <= 0:
            raise ValueError(f"decoder_tail_channels must be positive, got {self.config.decoder_tail_channels}")
        if self.config.structure_recon_weight < 0.0:
            raise ValueError(f"structure_recon_weight must be non-negative, got {self.config.structure_recon_weight}")
        if self.config.shape_recon_weight < 0.0:
            raise ValueError(f"shape_recon_weight must be non-negative, got {self.config.shape_recon_weight}")
        if self.config.field_smoothness_weight < 0.0:
            raise ValueError(f"field_smoothness_weight must be non-negative, got {self.config.field_smoothness_weight}")
        if self.config.phase_smoothness_weight < 0.0:
            raise ValueError(f"phase_smoothness_weight must be non-negative, got {self.config.phase_smoothness_weight}")
        if self.config.background_leakage_weight < 0.0:
            raise ValueError(
                "background_leakage_weight must be non-negative, "
                f"got {self.config.background_leakage_weight}"
            )
        if self.config.camera_hotspot_weight < 0.0:
            raise ValueError(f"camera_hotspot_weight must be non-negative, got {self.config.camera_hotspot_weight}")
        if self.config.camera_topology_weight < 0.0:
            raise ValueError(f"camera_topology_weight must be non-negative, got {self.config.camera_topology_weight}")
        if self.config.camera_saturation_weight < 0.0:
            raise ValueError(f"camera_saturation_weight must be non-negative, got {self.config.camera_saturation_weight}")
        if self.config.field_energy_weight < 0.0:
            raise ValueError(f"field_energy_weight must be non-negative, got {self.config.field_energy_weight}")
        if self.config.linear_metasurface_grid_size <= 0:
            raise ValueError(
                "linear_metasurface_grid_size must be positive, "
                f"got {self.config.linear_metasurface_grid_size}"
            )
        if self.config.linear_metasurface_layer_spacing <= 0.0:
            raise ValueError(
                "linear_metasurface_layer_spacing must be positive, "
                f"got {self.config.linear_metasurface_layer_spacing}"
            )

    def set_image_fusion_weight(self, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"image_fusion_weight must be within [0, 1], got {weight}")
        self.image_fusion_weight = float(weight)

    def _resolve_optical_encoder_input_channels(self) -> int:
        if self.config.optical_encoder_input_mode == "pf2_complex_field":
            return 2
        if self.config.optical_encoder_input_mode == "pf2_intensity_multiscale_v2":
            return 8
        if self.config.optical_encoder_input_mode == "pf2_intensity_multiscale":
            return 4
        return 1

    def _build_optical_encoder(self) -> nn.Module:
        if self.config.optical_encoder_arch == "linear_metasurface":
            return LinearMetasurfaceEncoder(
                self.physics,
                self.config.image_size,
                self.config.latent_channels,
                self.config.image_size // 8,
                self.config.linear_metasurface_grid_size,
                self.config.linear_metasurface_layer_spacing,
            )
        if self.config.optical_encoder_arch == "global_context":
            return GlobalContextMetasurfaceEncoder(
                self.config.encoder_channels,
                input_channels=self.optical_encoder_input_channels,
            )
        return MetasurfaceEncoder(
            self.config.encoder_channels,
            input_channels=self.optical_encoder_input_channels,
        )

    def _build_metasurface_decoder(self) -> nn.Module:
        if self.config.metasurface_decoder_arch == "linear_metasurface":
            return LinearMetasurfaceDecoder(
                self.physics,
                self.config.image_size,
                self.config.latent_channels,
                self.config.image_size // 8,
                self.config.linear_metasurface_grid_size,
                self.config.linear_metasurface_layer_spacing,
            )
        return MetasurfaceDecoder(
            self.config.encoder_channels,
            self.config.latent_channels,
            self.config.decoder_tail_channels,
        )

    def build_optical_encoder_input(
        self,
        pf2_intensity: torch.Tensor,
        pf2_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.optical_encoder_input_mode == "pf2_complex_field":
            if pf2_field is None:
                raise ValueError("pf2_field is required for pf2_complex_field encoder input.")
            return torch.cat([pf2_field.real, pf2_field.imag], dim=1)
        if self.config.optical_encoder_input_mode == "pf2_intensity":
            return pf2_intensity

        # 只从可测 PF2 intensity 构造多尺度读出，不把原图或复场信息接进 encoder。
        local_lowpass = F.avg_pool2d(pf2_intensity, kernel_size=5, stride=1, padding=2)
        broad_lowpass = F.avg_pool2d(pf2_intensity, kernel_size=15, stride=1, padding=7)
        diff_y = F.pad(
            pf2_intensity[..., 1:, :] - pf2_intensity[..., :-1, :],
            (0, 0, 1, 0),
        )
        diff_x = F.pad(
            pf2_intensity[..., :, 1:] - pf2_intensity[..., :, :-1],
            (1, 0, 0, 0),
        )
        edge_magnitude = (diff_x.square() + diff_y.square()).sqrt()
        if self.config.optical_encoder_input_mode == "pf2_intensity_multiscale":
            return torch.cat([pf2_intensity, local_lowpass, broad_lowpass, edge_magnitude], dim=1)

        log_intensity = torch.log1p(pf2_intensity.clamp_min(0.0))
        log_intensity = log_intensity / log_intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        local_contrast = pf2_intensity - local_lowpass
        dog_bandpass = local_lowpass - broad_lowpass
        dog_diff_y = F.pad(
            dog_bandpass[..., 1:, :] - dog_bandpass[..., :-1, :],
            (0, 0, 1, 0),
        )
        dog_diff_x = F.pad(
            dog_bandpass[..., :, 1:] - dog_bandpass[..., :, :-1],
            (1, 0, 0, 0),
        )
        dog_edge = (dog_diff_x.square() + dog_diff_y.square()).sqrt()
        # v2 仍只读 PF2 intensity，把动态范围、局部残差和带通边缘显式交给 encoder。
        return torch.cat(
            [
                pf2_intensity,
                local_lowpass,
                broad_lowpass,
                edge_magnitude,
                log_intensity,
                local_contrast,
                dog_bandpass,
                dog_edge,
            ],
            dim=1,
        )

    def _validate_electronic_domain_config(self) -> None:
        electronic_cfg = self.config.electronic_domain
        if electronic_cfg.mode not in {"mnist_complex_slm", "mnist_amplitude_slm"}:
            raise ValueError(f"Unsupported electronic_domain mode: {electronic_cfg.mode}")
        if electronic_cfg.gamma <= 0.0:
            raise ValueError(f"electronic_domain.gamma must be positive, got {electronic_cfg.gamma}")
        if electronic_cfg.polarity != "non_inverted":
            raise ValueError(f"Unsupported electronic_domain polarity: {electronic_cfg.polarity}")
        if not 0.0 <= electronic_cfg.background_floor <= 1.0:
            raise ValueError(
                "electronic_domain.background_floor must be in [0, 1], "
                f"got {electronic_cfg.background_floor}"
            )

    def to_electronic_pattern(self, input_batch: torch.Tensor) -> torch.Tensor:
        self._validate_electronic_domain_config()
        # electronic_pattern 保留为电子域控制图；它还不是光场，只是后续生成振幅/相位的控制量。
        normalized = input_batch.clamp(0.0, 1.0)
        return normalized.pow(self.config.electronic_domain.gamma)

    def to_slm_amplitude(self, electronic_pattern: torch.Tensor) -> torch.Tensor:
        electronic_cfg = self.config.electronic_domain
        slm_amplitude = electronic_cfg.background_floor + (1.0 - electronic_cfg.background_floor) * electronic_pattern
        return slm_amplitude.clamp(0.0, 1.0)

    def to_slm_phase(self, electronic_pattern: torch.Tensor) -> torch.Tensor:
        electronic_cfg = self.config.electronic_domain
        if electronic_cfg.mode == "mnist_amplitude_slm":
            return torch.zeros_like(electronic_pattern)
        return electronic_cfg.phase_bias_rad + electronic_cfg.phase_scale_rad * electronic_pattern

    def to_slm_transfer(self, electronic_pattern: torch.Tensor, slm_phase: torch.Tensor | None = None) -> torch.Tensor:
        if slm_phase is None:
            slm_phase = self.to_slm_phase(electronic_pattern)
        slm_amplitude = self.to_slm_amplitude(electronic_pattern)
        return torch.polar(slm_amplitude, slm_phase)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input_batch: torch.Tensor, deterministic: bool | None = None) -> dict[str, torch.Tensor]:
        # 物理优先模式只从 PF2 光学读出编码；gated 模式只在早期用原图特征扶一把。
        electronic_pattern = self.to_electronic_pattern(input_batch)
        slm_phase = self.to_slm_phase(electronic_pattern)
        slm_transfer = self.to_slm_transfer(electronic_pattern, slm_phase)
        if self.uses_linear_metasurface:
            pf2_field = self.front_path.forward_field(slm_transfer, propagation_method="angular_spectrum")
            pf2_intensity = normalize_intensity(pf2_field, self.physics.readout.normalize_percentile)
            optical_encoder_input = self.build_optical_encoder_input(pf2_intensity, pf2_field=pf2_field)
            encoded_stats = self.optical_encoder(pf2_field)
            mu = encoded_stats["mu"]
            logvar = encoded_stats["logvar"]
        else:
            pf2_intensity = self.front_path(slm_transfer)
            optical_encoder_input = self.build_optical_encoder_input(pf2_intensity)
            optical_features = self.optical_encoder(optical_encoder_input)
            if self.config.encoder_fusion_mode == "dual_path_concat":
                if self.image_encoder is None:
                    raise RuntimeError("image_encoder is required for dual_path_concat mode.")
                image_features = self.image_encoder(input_batch)
                fused_features = self.encoder_fusion(torch.cat([optical_features, image_features], dim=1))
            elif self.config.encoder_fusion_mode == "gated_dual_path":
                if self.image_encoder is None:
                    raise RuntimeError("image_encoder is required for gated_dual_path mode.")
                image_features = self.image_encoder(input_batch)
                fused_features = optical_features + self.image_fusion_weight * image_features
            else:
                fused_features = self.encoder_fusion(optical_features)
            mu = self.mu_head(fused_features)
            logvar = self.logvar_head(fused_features)
        if deterministic is None:
            use_deterministic = self.uses_linear_metasurface or not self.training
        else:
            use_deterministic = deterministic
        z = mu if use_deterministic else self.reparameterize(mu, logvar)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "electronic_pattern": electronic_pattern,
            "slm_phase": slm_phase,
            "pf2_intensity": pf2_intensity,
            "optical_encoder_input": optical_encoder_input,
        }

    def decode_with_intermediates(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # 解码先走共享 trunk 得到“数字图像头 + 复 chip 场头”，再把复场送进后级光路。
        z_after_ols = self.ols(z)
        decoded = self.metasurface_decoder(z_after_ols)
        chip_field = torch.complex(decoded["chip_real"], decoded["chip_imag"])
        optical_readout = self.back_path(chip_field)
        return {
            "z_after_ols": z_after_ols,
            "image_recon": decoded["image_recon"],
            **decoded,
            **optical_readout,
        }

    def _primary_output(self, decoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.primary_output == "camera":
            return decoded["camera_plane"]
        return decoded["image_recon"]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._primary_output(self.decode_with_intermediates(z))

    def forward(self, input_batch: torch.Tensor, deterministic: bool | None = None) -> dict[str, torch.Tensor]:
        # 第一阶段训练时，x_hat 跟随 primary_output；物理优先默认直接监督 camera_plane。
        encoded = self.encode(input_batch, deterministic=deterministic)
        decoded = self.decode_with_intermediates(encoded["z"])
        return {
            "x_hat": self._primary_output(decoded),
            "image_recon": decoded["image_recon"],
            "camera_plane": decoded["camera_plane"],
            "camera_plane_loss": decoded["camera_plane_loss"],
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
            "z": encoded["z"],
            "electronic_pattern": encoded["electronic_pattern"],
            "slm_phase": encoded["slm_phase"],
            "pf2_intensity": encoded["pf2_intensity"],
            "optical_encoder_input": encoded["optical_encoder_input"],
            "chip_real": decoded["chip_real"],
            "chip_imag": decoded["chip_imag"],
            "chip_amplitude": decoded["chip_amplitude"],
            "chip_phase": decoded["chip_phase"],
            "probe_plane": decoded["probe_plane"],
            "probe_plane_loss": decoded["probe_plane_loss"],
        }


def validate_first_stage_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    model_version = payload.get("model_version")
    if model_version != FIRST_STAGE_MODEL_VERSION:
        raise ValueError(
            "Incompatible first-stage checkpoint: expected physical-first hybrid OpticalPathVAE checkpoint metadata."
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Incompatible first-stage checkpoint: missing model_state_dict.")


def load_first_stage_checkpoint(model: OpticalPathVAE, payload: Mapping[str, Any]) -> None:
    validate_first_stage_checkpoint_payload(payload)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            "Incompatible first-stage checkpoint: checkpoint state does not match the physical-first hybrid OpticalPathVAE."
        ) from exc
