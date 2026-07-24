"""Teacher-free conditional optical generation primitives.

This module is intentionally independent of v2's historical teacher manifests.
Every image loss begins at raw CMOS intensity and gradients then flow through the
formal optical system to ``ConditionalGlobalAmplitudeGenerator``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .generator import ConditionalGlobalAmplitudeGenerator


@dataclass(frozen=True)
class ConditionalLossSummary:
    total: float
    sinkhorn: float
    classification: float
    diversity: float
    saturation: float
    mode_orthogonality: float
    phase_tv: float


def fixed_exposure_view(raw_cmos: torch.Tensor, exposure_scale: float, *, output_size: int = 28) -> torch.Tensor:
    """A globally calibrated detector view; never normalizes individual samples."""
    if raw_cmos.ndim != 4 or raw_cmos.shape[1] != 1 or exposure_scale <= 0.0:
        raise ValueError("raw CMOS must be [batch, 1, H, W] with a positive fixed exposure")
    normalized = (raw_cmos / exposure_scale).clamp(0.0, 1.0)
    return torch.nn.functional.adaptive_avg_pool2d(normalized, (output_size, output_size))


def phase_total_variation(system: nn.Module) -> torch.Tensor:
    values = []
    for name, parameter in system.named_parameters():
        if name.endswith("phase") and parameter.ndim == 4:
            values.extend(((parameter[..., 1:, :] - parameter[..., :-1, :]).square().mean(), (parameter[..., :, 1:] - parameter[..., :, :-1]).square().mean()))
    if not values:
        raise ValueError("conditional optical system has no trainable phase masks")
    return sum(values)


def log_sinkhorn(cost: torch.Tensor, *, epsilon: float = 0.05, iterations: int = 20) -> torch.Tensor:
    """Balanced entropic OT in log-domain, with no third-party dependency."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] < 2:
        raise ValueError("Sinkhorn requires a square cost matrix with at least two samples")
    if epsilon <= 0.0 or iterations < 1:
        raise ValueError("invalid Sinkhorn parameters")
    count = cost.shape[0]
    log_mass = -torch.log(torch.tensor(float(count), device=cost.device, dtype=cost.dtype))
    log_kernel = -cost / epsilon
    u = torch.zeros(count, device=cost.device, dtype=cost.dtype)
    v = torch.zeros_like(u)
    for _ in range(iterations):
        u = log_mass - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_mass - torch.logsumexp(log_kernel + u[:, None], dim=0)
    transport = torch.exp(log_kernel + u[:, None] + v[None, :])
    return (transport * cost).sum()


def conditional_sinkhorn_loss(
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Class-conditional set matching; deliberately has no sample-index pairing."""
    if generated_features.shape != real_features.shape or generated_view.shape != real_view.shape:
        raise ValueError("generated and real batches must have matching shapes")
    terms = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("each conditional Sinkhorn batch needs at least two samples per label")
        feature_cost = torch.cdist(generated_features.index_select(0, selected), real_features.index_select(0, selected)).square()
        generated_pixels = generated_view.index_select(0, selected).flatten(1)
        real_pixels = real_view.index_select(0, selected).flatten(1)
        pixel_cost = torch.cdist(generated_pixels, real_pixels, p=1) / generated_pixels.shape[1]
        terms.append(log_sinkhorn(feature_cost + 0.25 * pixel_cost, epsilon=epsilon, iterations=iterations))
    return torch.stack(terms).mean()


def diversity_hinge(features: torch.Tensor, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Require code-separated samples to remain separated in CMOS semantic space."""
    terms = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        feature_distance = torch.pdist(features.index_select(0, selected), p=2)
        code_distance = torch.pdist(codes.index_select(0, selected), p=2)
        terms.append(torch.relu(0.25 * code_distance - feature_distance).mean())
    return torch.stack(terms).mean()


def raw_from_generator(system: nn.Module, generator: ConditionalGlobalAmplitudeGenerator, codes: torch.Tensor, labels: torch.Tensor, *, activation_checkpoint: bool) -> torch.Tensor:
    def forward(command: torch.Tensor) -> torch.Tensor:
        return system.forward(system.command(command)).raw_camera.tensor

    command = generator(codes, labels)
    if activation_checkpoint and torch.is_grad_enabled():
        return checkpoint(forward, command, use_reentrant=False)
    return forward(command)


def raw_from_generator_batched(system: nn.Module, generator: ConditionalGlobalAmplitudeGenerator, codes: torch.Tensor, labels: torch.Tensor, *, microbatch_size: int = 1, activation_checkpoint: bool) -> torch.Tensor:
    if microbatch_size < 1:
        raise ValueError("optical microbatch_size must be positive")
    rows = []
    for start in range(0, codes.shape[0], microbatch_size):
        rows.append(raw_from_generator(system, generator, codes[start : start + microbatch_size], labels[start : start + microbatch_size], activation_checkpoint=activation_checkpoint))
    return torch.cat(rows, dim=0)


def conditional_loss(
    *,
    system: nn.Module,
    generator: ConditionalGlobalAmplitudeGenerator,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    exposure_scale: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
    activation_checkpoint: bool,
    microbatch_size: int = 1,
) -> tuple[torch.Tensor, ConditionalLossSummary, torch.Tensor]:
    raw = raw_from_generator_batched(system, generator, codes, labels, microbatch_size=microbatch_size, activation_checkpoint=activation_checkpoint)
    generated_view = fixed_exposure_view(raw, exposure_scale)
    generated_features, logits = evaluator(generated_view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(generated_features, generated_view, real_features, real_views, labels, epsilon=sinkhorn_epsilon, iterations=sinkhorn_iterations)
    classification = torch.nn.functional.cross_entropy(logits, labels)
    diversity = diversity_hinge(generated_features, codes, labels)
    saturation = torch.relu(raw / exposure_scale - 1.0).square().mean()
    orthogonality = generator.mode_orthogonality_penalty()
    phase_tv = phase_total_variation(system)
    total = sinkhorn + 0.5 * classification + 0.1 * diversity + 0.1 * saturation + 1.0e-3 * orthogonality + 1.0e-5 * phase_tv
    summary = ConditionalLossSummary(*(float(value.detach().item()) for value in (total, sinkhorn, classification, diversity, saturation, orthogonality, phase_tv)))
    return total, summary, raw
