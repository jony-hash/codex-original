from __future__ import annotations

import torch
import torch.nn.functional as F

from .diffusion import LatentDiffusionUNet, LinearNoiseScheduler
from .first_stage import OpticalPathVAE
from .first_stage_losses import select_diffusion_latents


@torch.no_grad()
def estimate_latent_scale(first_stage: OpticalPathVAE, batch: torch.Tensor, latent_source: str = "mu") -> torch.Tensor:
    encoded = first_stage.encode(batch)
    latents = select_diffusion_latents(encoded, latent_source)
    return latents.detach().pow(2).mean().sqrt().clamp_min(1e-6)


def apply_classifier_free_guidance(
    conditioned_prediction: torch.Tensor,
    unconditioned_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    return unconditioned_prediction + guidance_scale * (conditioned_prediction - unconditioned_prediction)


def diffusion_per_sample_losses(
    first_stage: OpticalPathVAE,
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    batch: torch.Tensor,
    class_labels: torch.Tensor,
    latent_scale: torch.Tensor,
    latent_source: str = "mu",
) -> torch.Tensor:
    with torch.no_grad():
        encoded = first_stage.encode(batch)
        latents = select_diffusion_latents(encoded, latent_source) / latent_scale

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.timesteps, (latents.shape[0],), device=latents.device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    context = diffusion_model.encode_condition(class_labels)
    predicted_noise = diffusion_model(noisy_latents, timesteps, context)
    return F.mse_loss(predicted_noise, noise, reduction="none").flatten(1).mean(dim=1)


def _resolve_diffusion_loss_weights(
    class_labels: torch.Tensor,
    class_weight_labels: tuple[int, ...] = (),
    class_weight: float = 1.0,
) -> torch.Tensor:
    weights = torch.ones(class_labels.shape[0], device=class_labels.device, dtype=torch.float32)
    if not class_weight_labels or class_weight == 1.0:
        return weights

    weak_class_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    for label in class_weight_labels:
        weak_class_mask |= class_labels == int(label)
    return torch.where(weak_class_mask, weights.new_full((), float(class_weight)), weights)


def diffusion_loss(
    first_stage: OpticalPathVAE,
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    batch: torch.Tensor,
    class_labels: torch.Tensor,
    latent_scale: torch.Tensor,
    latent_source: str = "mu",
    class_weight_labels: tuple[int, ...] = (),
    class_weight: float = 1.0,
) -> torch.Tensor:
    per_sample_losses = diffusion_per_sample_losses(
        first_stage,
        diffusion_model,
        scheduler,
        batch,
        class_labels,
        latent_scale,
        latent_source=latent_source,
    )
    weights = _resolve_diffusion_loss_weights(class_labels, class_weight_labels, class_weight)
    return (per_sample_losses * weights).sum() / weights.sum().clamp_min(1.0)

