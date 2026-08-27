"""Offline software first-stage VAE for Package B3.

This module is intentionally independent of the optical first-stage.  It is
an offline teacher component only: no optical path, OLS, evaluator, student,
or diffusion object is constructed here.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MODEL_VERSION = "software_multiscale_teacher_vae_v1"
LATENT_SHAPE = (1, 25, 25)
IMAGE_SIZE = 200


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = _group_norm(out_channels)
        self.block = ResidualBlock(out_channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(F.silu(self.norm(self.downsample(value))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = _group_norm(out_channels)
        self.block = ResidualBlock(out_channels)

    def forward(self, value: Tensor) -> Tensor:
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(F.silu(self.norm(self.conv(value))))


class SoftwareMultiscaleTeacherVAE(nn.Module):
    """Unconditional 200x200 software VAE with a 1x25x25 latent.

    Labels are deliberately absent from the call signature.  Class balancing
    belongs to the training sampler/loss and cannot become a decoder shortcut.
    """

    model_version = MODEL_VERSION
    latent_shape = LATENT_SHAPE
    image_size = IMAGE_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            _group_norm(32),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            DownsampleBlock(32, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 128),
            ResidualBlock(128),
        )
        self.mu_head = nn.Conv2d(128, LATENT_SHAPE[0], 1)
        self.logvar_head = nn.Conv2d(128, LATENT_SHAPE[0], 1)

        self.decoder_in = nn.Sequential(
            nn.Conv2d(LATENT_SHAPE[0], 128, 3, padding=1),
            _group_norm(128),
            nn.SiLU(),
            ResidualBlock(128),
        )
        self.decoder = nn.Sequential(
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
        )
        self.output_head = nn.Conv2d(32, 1, 3, padding=1)

    def encode(self, images: Tensor) -> dict[str, Tensor]:
        _validate_images(images)
        hidden = self.encoder(self.encoder_stem(images))
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-30.0, 20.0)
        return {"mu": mu, "logvar": logvar}

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != LATENT_SHAPE:
            raise ValueError(
                f"software VAE latent must have shape [N, {LATENT_SHAPE}], got {tuple(latent.shape)}"
            )
        hidden = self.decoder(self.decoder_in(latent))
        output = torch.sigmoid(self.output_head(hidden))
        if tuple(output.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"software VAE decoder produced {tuple(output.shape)}")
        return output

    def forward(self, images: Tensor, *, deterministic: bool = False) -> dict[str, Tensor]:
        encoded = self.encode(images)
        latent = encoded["mu"] if deterministic else self.reparameterize(encoded["mu"], encoded["logvar"])
        return {**encoded, "latent": latent, "reconstruction": self.decode(latent)}


def _validate_images(images: Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"software VAE images must have shape [N, 1, {IMAGE_SIZE}, {IMAGE_SIZE}], got {tuple(images.shape)}"
        )
    if not torch.is_floating_point(images):
        raise TypeError("software VAE images must be floating point")


def sobel(images: Tensor) -> tuple[Tensor, Tensor]:
    kernel_x = images.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return F.conv2d(images, kernel_x, padding=1), F.conv2d(images, kernel_y, padding=1)


def _foreground_dice_per_sample(
    reconstruction: Tensor, target: Tensor, threshold: float = 0.05
) -> Tensor:
    predicted = reconstruction.clamp_min(0.0)
    truth = target.clamp_min(0.0)
    predicted_mask = (predicted > threshold).to(predicted.dtype)
    truth_mask = (truth > threshold).to(truth.dtype)
    intersection = (predicted_mask * truth_mask).flatten(1).sum(1)
    denominator = predicted_mask.flatten(1).sum(1) + truth_mask.flatten(1).sum(1)
    return 1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)


def foreground_dice(reconstruction: Tensor, target: Tensor, threshold: float = 0.05) -> Tensor:
    return _foreground_dice_per_sample(reconstruction, target, threshold).mean()


def _samplewise_terms(reconstruction: Tensor, target: Tensor) -> Tensor:
    recon_x, recon_y = sobel(reconstruction)
    target_x, target_y = sobel(target)
    terms = [
        F.smooth_l1_loss(reconstruction, target, reduction="none").flatten(1).mean(1),
        F.smooth_l1_loss(
            F.avg_pool2d(reconstruction, 2), F.avg_pool2d(target, 2), reduction="none"
        ).flatten(1).mean(1),
        F.smooth_l1_loss(
            F.avg_pool2d(reconstruction, 4), F.avg_pool2d(target, 4), reduction="none"
        ).flatten(1).mean(1),
        F.smooth_l1_loss(recon_x, target_x, reduction="none").flatten(1).mean(1),
        F.smooth_l1_loss(recon_y, target_y, reduction="none").flatten(1).mean(1),
    ]
    return torch.stack(terms, dim=1)


def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    labels: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    temperature: float = 0.1,
    kl_beta: float = 1.0e-6,
    multiscale_weight: float = 1.0,
    sobel_weight: float = 0.5,
    foreground_dice_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    _validate_images(reconstruction)
    _validate_images(target)
    if labels.ndim != 1 or labels.shape[0] != target.shape[0]:
        raise ValueError("labels must be one-dimensional and match the image batch")
    if temperature <= 0.0 or kl_beta < 0.0 or min(multiscale_weight, sobel_weight, foreground_dice_weight) < 0.0:
        raise ValueError("temperature must be positive and loss weights cannot be negative")
    pixel_terms = _samplewise_terms(reconstruction, target)
    multiscale = pixel_terms[:, :3].mean(dim=1)
    sobel_terms = pixel_terms[:, 3:].mean(dim=1)
    dice_per_sample = _foreground_dice_per_sample(reconstruction, target)
    sample_loss = multiscale_weight * multiscale + sobel_weight * sobel_terms + foreground_dice_weight * dice_per_sample
    class_losses = []
    for label in range(10):
        selected = sample_loss[labels == label]
        if selected.numel():
            class_losses.append(selected.mean())
    if not class_losses:
        raise ValueError("reconstruction loss requires at least one known MNIST class")
    class_tensor = torch.stack(class_losses)
    worst_class = temperature * torch.logsumexp(class_tensor / temperature, dim=0)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
    total = worst_class + kl_beta * kl
    return total, {
        # Keep metrics detached but on the source device.  Converting these to
        # Python scalars here synchronizes every training batch with the host.
        # Callers that need JSON should materialize them at an evidence
        # boundary, not in the loss hot path.
        "total": total.detach(),
        "worst_class_reconstruction": worst_class.detach(),
        "foreground_dice": dice_per_sample.mean().detach(),
        "kl": kl.detach(),
    }


def state_content_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_content_hash(parameters: Iterable[Tensor]) -> str:
    return state_content_hash({str(index): value for index, value in enumerate(parameters)})
