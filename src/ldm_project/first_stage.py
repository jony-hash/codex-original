from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .config import FirstStageConfig, PhysicsConfig
from .optics import BackOpticalPath, FrontOpticalPath


FIRST_STAGE_MODEL_VERSION = "physical_first_hybrid_vae_v3"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class PointwiseOLS(nn.Module):
    def __init__(self, kind: str = "pointwise_tanh") -> None:
        super().__init__()
        self.kind = kind

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind in {"pointwise_tanh", "tanh"}:
            return torch.tanh(z)
        if self.kind == "sigmoid":
            return torch.sigmoid(z)
        raise ValueError(f"Unsupported OLS kind: {self.kind}")


class MetasurfaceEncoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetasurfaceDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], latent_channels: int, tail_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, c3, kernel_size=1),
            nn.GroupNorm(8, c3),
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


class OpticalPathVAE(nn.Module):
    def __init__(self, physics: PhysicsConfig, config: FirstStageConfig | None = None) -> None:
        super().__init__()
        self.physics = physics
        self.config = config or FirstStageConfig()
        self._validate_model_config()
        self.front_path = FrontOpticalPath(physics)
        self.optical_encoder = MetasurfaceEncoder(self.config.encoder_channels)
        encoder_out_channels = self.config.encoder_channels[-1]
        self.image_fusion_weight = 0.0
        self.image_encoder: MetasurfaceEncoder | None
        if self.config.encoder_fusion_mode in {"dual_path_concat", "gated_dual_path"}:
            self.image_encoder = MetasurfaceEncoder(self.config.encoder_channels)
        else:
            self.image_encoder = None
        self.encoder_fusion: nn.Module
        if self.config.encoder_fusion_mode == "dual_path_concat":
            self.encoder_fusion = nn.Sequential(
                nn.Conv2d(encoder_out_channels * 2, encoder_out_channels, kernel_size=1),
                nn.GroupNorm(8, encoder_out_channels),
                nn.SiLU(),
            )
        elif self.config.encoder_fusion_mode == "gated_dual_path":
            self.encoder_fusion = nn.Identity()
            self.set_image_fusion_weight(self.config.image_fusion_start_weight)
        else:
            self.encoder_fusion = nn.Identity()
        self.mu_head = nn.Conv2d(encoder_out_channels, self.config.latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(encoder_out_channels, self.config.latent_channels, kernel_size=1)
        self.ols = PointwiseOLS(self.config.ols_kind)
        self.metasurface_decoder = MetasurfaceDecoder(
            self.config.encoder_channels,
            self.config.latent_channels,
            self.config.decoder_tail_channels,
        )
        self.back_path = BackOpticalPath(physics)

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

    def set_image_fusion_weight(self, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"image_fusion_weight must be within [0, 1], got {weight}")
        self.image_fusion_weight = float(weight)

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
        pf2_intensity = self.front_path(slm_transfer)
        optical_features = self.optical_encoder(pf2_intensity)
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
        use_deterministic = not self.training if deterministic is None else deterministic
        z = mu if use_deterministic else self.reparameterize(mu, logvar)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "electronic_pattern": electronic_pattern,
            "slm_phase": slm_phase,
            "pf2_intensity": pf2_intensity,
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
