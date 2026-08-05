"""Fixed teacher-free adversarial losses for v2.3."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .o0_models import ProjectionDiscriminator


def discriminator_loss(
    discriminator: ProjectionDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    *,
    r1_weight: float = 10.0,
    apply_r1: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute fixed hinge/R1 discriminator loss without touching optimizer state."""
    real = real.detach().requires_grad_(apply_r1)
    fake = fake.detach()
    real_score, _, _, _ = discriminator(real, labels)
    fake_score, _, _, _ = discriminator(fake, labels)
    hinge_real = F.relu(1.0 - real_score).mean()
    hinge_fake = F.relu(1.0 + fake_score).mean()
    r1 = real.new_zeros(())
    if apply_r1:
        gradient = torch.autograd.grad(real_score.sum(), real, create_graph=True, only_inputs=True)[0]
        r1 = gradient.flatten(1).square().sum(1).mean() * (r1_weight * 0.5)
    loss = hinge_real + hinge_fake + r1
    return loss, {"hinge_real": float(hinge_real.detach()), "hinge_fake": float(hinge_fake.detach()), "r1": float(r1.detach())}


def generator_loss(
    discriminator: ProjectionDiscriminator,
    generated: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    *,
    real_features: torch.Tensor | None = None,
    anchor_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fixed data-adversarial generator objective; evaluator is never an input."""
    score, logits, predicted_code, features = discriminator(generated, labels)
    adversarial = -score.mean()
    feature_matching = generated.new_zeros(()) if real_features is None else F.l1_loss(features, real_features.detach())
    classification = F.cross_entropy(logits, labels)
    latent = F.smooth_l1_loss(predicted_code, codes)
    saturation = (generated >= 1.0).float().mean()
    anchor = generated.new_zeros(()) if anchor_value is None else anchor_value
    loss = adversarial + 10.0 * feature_matching + classification + latent + 0.1 * saturation + anchor
    return loss, {
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
        "classification": float(classification.detach()),
        "latent": float(latent.detach()),
        "saturation": float(saturation.detach()),
        "anchor": float(anchor.detach()),
    }


def soft_anchor_loss(generator: nn.Module, labels: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """The C2/C3 anchor is a training-only zero-code penalty."""
    code_dim = int(getattr(generator, "code_dim"))
    zero = torch.zeros((labels.shape[0], code_dim), device=device)
    generated = generator(zero, labels)
    carrier = torch.sigmoid(generator.carrier_logits.index_select(0, labels))
    carrier = F.interpolate(carrier, size=generated.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(generated, carrier.detach())


def detach_feature_cache(discriminator: ProjectionDiscriminator, real: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cache real discriminator features without building a real-data graph."""
    with torch.no_grad():
        return discriminator(real, labels)[3].detach()


def validate_loss_contract() -> Mapping[str, float]:
    return {
        "generator_adversarial_weight": 1.0,
        "feature_matching_weight": 10.0,
        "classification_weight": 1.0,
        "latent_weight": 1.0,
        "saturation_weight": 0.1,
        "r1_weight": 10.0,
        "r1_interval": 16.0,
        "optimizer_lr": 2.0e-4,
    }
