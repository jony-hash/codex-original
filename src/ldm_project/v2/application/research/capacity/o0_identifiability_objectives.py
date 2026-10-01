"""Losses and parameter-boundary helpers for the v2.4 audit."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.capacity.o0_models import ProjectionDiscriminator


def discriminator_q_loss(
    discriminator: ProjectionDiscriminator,
    real: torch.Tensor,
    fake_detached: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    *,
    r1_weight: float = 10.0,
    apply_r1: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    real = real.detach().requires_grad_(apply_r1)
    fake_detached = fake_detached.detach()
    real_score, real_logits, _, _ = discriminator(real, labels)
    fake_score, _, fake_code, _ = discriminator(fake_detached, labels)
    hinge_real = F.relu(1.0 - real_score).mean()
    hinge_fake = F.relu(1.0 + fake_score).mean()
    real_ce = F.cross_entropy(real_logits, labels)
    latent = F.smooth_l1_loss(fake_code, codes.detach())
    r1 = real.new_zeros(())
    if apply_r1:
        grad = torch.autograd.grad(
            real_score.sum(), real, create_graph=True, only_inputs=True
        )[0]
        r1 = 0.5 * r1_weight * grad.flatten(1).square().sum(1).mean()
    loss = hinge_real + hinge_fake + r1 + real_ce + latent
    return loss, {
        "hinge_real": float(hinge_real.detach()),
        "hinge_fake": float(hinge_fake.detach()),
        "r1": float(r1.detach()),
        "real_classifier": float(real_ce.detach()),
        "latent_q": float(latent.detach()),
    }


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


def generator_identifiability_loss(
    discriminator: ProjectionDiscriminator,
    generated: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    *,
    real_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    score, logits, predicted_code, features = discriminator(generated, labels)
    adversarial = -score.mean()
    feature_matching = (
        generated.new_zeros(())
        if real_features is None
        else F.l1_loss(features, real_features.detach())
    )
    classification = F.cross_entropy(logits, labels)
    latent = F.smooth_l1_loss(predicted_code, codes)
    saturation = (generated >= 1.0).float().mean()
    loss = (
        adversarial
        + 10.0 * feature_matching
        + classification
        + latent
        + 0.1 * saturation
    )
    return loss, {
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
        "classification": float(classification.detach()),
        "latent": float(latent.detach()),
        "saturation": float(saturation.detach()),
    }


def parameter_hash(module: nn.Module) -> str:
    from ldm_project.v2.evidence import tensor_sha256, canonical_sha256

    return canonical_sha256(
        {
            name: tensor_sha256(value.detach().cpu())
            for name, value in module.state_dict().items()
        }
    )
