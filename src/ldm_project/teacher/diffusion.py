from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DiffusionConfig


DIFFUSION_MODEL_VERSION = "conditioned_ldm_unet_v1"


def _expand_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def _num_groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        exponent = torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half, 1)
        frequencies = torch.exp(-math.log(10000.0) * exponent)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding.shape[1] < self.channels:
            padding = torch.zeros(embedding.shape[0], 1, device=timesteps.device)
            embedding = torch.cat([embedding, padding], dim=1)
        return embedding


class ClassConditionEncoder(nn.Module):
    def __init__(
        self, num_classes: int, context_dim: int, cond_dropout_prob: float
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim
        self.cond_dropout_prob = cond_dropout_prob
        self.class_embedding = nn.Embedding(num_classes, context_dim)
        self.null_embedding = nn.Parameter(torch.zeros(1, 1, context_dim))
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.null_embedding, mean=0.0, std=0.02)

    def null_context(self, batch_size: int) -> torch.Tensor:
        return self.null_embedding.expand(batch_size, -1, -1)

    def forward(
        self,
        class_labels: torch.Tensor,
        force_unconditional: bool = False,
        dropout_prob: float | None = None,
    ) -> torch.Tensor:
        labels = class_labels.long()
        if force_unconditional:
            return self.null_context(labels.shape[0])

        context = self.class_embedding(labels).unsqueeze(1)
        if dropout_prob is None:
            dropout_prob = self.cond_dropout_prob if self.training else 0.0
        if dropout_prob <= 0.0:
            return context

        dropout_mask = torch.rand(labels.shape[0], device=labels.device) < dropout_prob
        if not dropout_mask.any():
            return context
        null_context = self.null_context(labels.shape[0])
        return torch.where(dropout_mask.view(-1, 1, 1), null_context, context)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip: nn.Module
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + _expand_to(self.time_proj(time_embedding), h)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class AttentionBlock(nn.Module):
    def __init__(
        self, channels: int, num_heads: int, context_dim: int, enabled: bool
    ) -> None:
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.self_norm = nn.LayerNorm(channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        batch_size, channels, height, width = x.shape
        residual = x
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        self_tokens = self.self_norm(tokens)
        self_out, _ = self.self_attn(
            self_tokens, self_tokens, self_tokens, need_weights=False
        )
        tokens = tokens + self_out

        cross_tokens = self.cross_norm(tokens)
        cross_out, _ = self.cross_attn(
            cross_tokens, context, context, need_weights=False
        )
        tokens = tokens + cross_out
        attended = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return attended + residual


class UNetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        num_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(current_channels, out_channels, time_channels)
            )
            self.attention_blocks.append(
                AttentionBlock(out_channels, num_heads, context_dim, use_attention)
            )
            current_channels = out_channels

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, time_embedding)
            x = attention_block(x, context)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


class LatentDiffusionUNet(nn.Module):
    def __init__(
        self, latent_channels: int, config: DiffusionConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        self.latent_channels = latent_channels
        self.condition_encoder = ClassConditionEncoder(
            num_classes=self.config.num_classes,
            context_dim=self.config.context_dim,
            cond_dropout_prob=self.config.cond_dropout_prob,
        )

        channels_by_level = [
            self.config.base_channels * multiplier
            for multiplier in self.config.channel_multipliers
        ]
        if len(channels_by_level) < 2:
            raise ValueError("channel_multipliers must contain at least two levels.")

        time_channels = self.config.base_channels * 4
        self.time_embedding = SinusoidalTimeEmbedding(self.config.base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        self.in_conv = nn.Conv2d(
            latent_channels, channels_by_level[0], kernel_size=3, padding=1
        )
        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels_by_level[0]
        for level, next_channels in enumerate(channels_by_level[1:]):
            self.down_stages.append(
                UNetStage(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    time_channels=time_channels,
                    num_res_blocks=self.config.num_res_blocks,
                    use_attention=level in self.config.attention_levels,
                    num_heads=self.config.num_heads,
                    context_dim=self.config.context_dim,
                )
            )
            self.downsamples.append(DownsampleBlock(current_channels, next_channels))
            current_channels = next_channels

        bottleneck_level = len(channels_by_level) - 1
        self.mid_stage = UNetStage(
            in_channels=current_channels,
            out_channels=current_channels,
            time_channels=time_channels,
            num_res_blocks=self.config.num_res_blocks,
            use_attention=bottleneck_level in self.config.attention_levels,
            num_heads=self.config.num_heads,
            context_dim=self.config.context_dim,
        )

        self.upsamples = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        for level in reversed(range(len(channels_by_level) - 1)):
            up_in_channels = channels_by_level[level + 1]
            up_out_channels = channels_by_level[level]
            self.upsamples.append(UpsampleBlock(up_in_channels, up_out_channels))
            self.up_stages.append(
                UNetStage(
                    in_channels=up_out_channels * 2,
                    out_channels=up_out_channels,
                    time_channels=time_channels,
                    num_res_blocks=self.config.num_res_blocks,
                    use_attention=level in self.config.attention_levels,
                    num_heads=self.config.num_heads,
                    context_dim=self.config.context_dim,
                )
            )

        self.out_norm = nn.GroupNorm(
            _num_groups(channels_by_level[0]), channels_by_level[0]
        )
        self.out_conv = nn.Conv2d(
            channels_by_level[0], latent_channels, kernel_size=3, padding=1
        )

    def encode_condition(
        self,
        class_labels: torch.Tensor,
        force_unconditional: bool = False,
        dropout_prob: float | None = None,
    ) -> torch.Tensor:
        return self.condition_encoder(
            class_labels,
            force_unconditional=force_unconditional,
            dropout_prob=dropout_prob,
        )

    def conditioning_metadata(self) -> dict[str, Any]:
        return {
            "type": "mnist_class",
            "num_classes": self.config.num_classes,
            "context_dim": self.config.context_dim,
            "cond_dropout_prob": self.config.cond_dropout_prob,
        }

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []

        for down_stage, downsample in zip(self.down_stages, self.downsamples):
            h = down_stage(h, time_embedding, context)
            skips.append(h)
            h = downsample(h)

        h = self.mid_stage(h, time_embedding, context)

        for upsample, up_stage in zip(self.upsamples, self.up_stages):
            skip = skips.pop()
            h = upsample(h, size=skip.shape[-2:])
            h = torch.cat([h, skip], dim=1)
            h = up_stage(h, time_embedding, context)

        return self.out_conv(F.silu(self.out_norm(h)))


class LinearNoiseScheduler(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "alphas_cumprod_prev", torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = _expand_to(self.sqrt_alphas_cumprod[timesteps], clean)
        sqrt_one_minus = _expand_to(
            self.sqrt_one_minus_alphas_cumprod[timesteps], clean
        )
        return sqrt_alpha * clean + sqrt_one_minus * noise

    @torch.no_grad()
    def step(
        self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        beta_t = _expand_to(self.betas[timesteps], sample)
        alpha_t = _expand_to(self.alphas[timesteps], sample)
        alpha_bar_t = _expand_to(self.alphas_cumprod[timesteps], sample)
        alpha_bar_prev = _expand_to(self.alphas_cumprod_prev[timesteps], sample)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(
            alpha_bar_t
        )
        mean = (
            torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * pred_x0
            + torch.sqrt(alpha_t)
            * (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
            * sample
        )

        noise = torch.randn_like(sample)
        nonzero = (timesteps > 0).float()
        while nonzero.ndim < sample.ndim:
            nonzero = nonzero.unsqueeze(-1)
        return mean + nonzero * torch.sqrt(beta_t) * noise


def validate_diffusion_checkpoint_payload(
    payload: Mapping[str, Any],
    expected_config: DiffusionConfig,
    expected_first_stage_model_version: str,
) -> None:
    model_version = payload.get("model_version")
    if model_version != DIFFUSION_MODEL_VERSION:
        raise ValueError(
            "Incompatible diffusion checkpoint: expected conditioned LDM U-Net checkpoint metadata."
        )

    conditioning = payload.get("conditioning")
    if not isinstance(conditioning, dict):
        raise ValueError(
            "Incompatible diffusion checkpoint: missing conditioning metadata."
        )

    if conditioning.get("num_classes") != expected_config.num_classes:
        raise ValueError(
            "Incompatible diffusion checkpoint: num_classes does not match the current config."
        )
    if conditioning.get("context_dim") != expected_config.context_dim:
        raise ValueError(
            "Incompatible diffusion checkpoint: context_dim does not match the current config."
        )
    if payload.get("first_stage_model_version") != expected_first_stage_model_version:
        raise ValueError(
            "Incompatible diffusion checkpoint: first-stage model version does not match the digit-first hybrid OpticalPathVAE."
        )


def load_diffusion_checkpoint(
    model: LatentDiffusionUNet,
    payload: Mapping[str, Any],
    expected_first_stage_model_version: str,
) -> None:
    validate_diffusion_checkpoint_payload(
        payload, model.config, expected_first_stage_model_version
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Incompatible diffusion checkpoint: missing model_state_dict.")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            "Incompatible diffusion checkpoint: checkpoint state does not match the conditioned LDM U-Net."
        ) from exc
