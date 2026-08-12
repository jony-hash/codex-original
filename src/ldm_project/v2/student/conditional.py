"""Teacher-free conditional optical generation primitives.

This module is intentionally independent of v2's historical teacher manifests.
Every image loss begins at raw CMOS intensity and gradients then flow through the
formal optical system to the configured amplitude-SLM generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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


def sinkhorn_cost_view(view: torch.Tensor, *, output_size: int = 14) -> torch.Tensor:
    """Return the fixed lower-resolution intensity view used by OT costs."""
    if view.ndim != 4 or view.shape[1] != 1:
        raise ValueError("view must be [batch, 1, H, W]")
    return torch.nn.functional.adaptive_avg_pool2d(view, (output_size, output_size))


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
        generated_features_label = torch.nn.functional.normalize(generated_features.index_select(0, selected), dim=1)
        real_features_label = torch.nn.functional.normalize(real_features.index_select(0, selected), dim=1)
        feature_cost = torch.cdist(generated_features_label, real_features_label).square()
        generated_pixels = sinkhorn_cost_view(generated_view.index_select(0, selected)).flatten(1)
        real_pixels = sinkhorn_cost_view(real_view.index_select(0, selected)).flatten(1)
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


def relative_diversity_ratios(features: torch.Tensor, real_features: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Per-class generated/real median feature-distance ratios.

    Both feature sets are L2-normalized before distances, matching the
    Sinkhorn feature contract and making this statistic independent of the
    evaluator's arbitrary feature scale.
    """
    if features.shape != real_features.shape or features.ndim != 2:
        raise ValueError("feature batches must have matching [N,D] shapes")
    generated = torch.nn.functional.normalize(features, dim=1)
    real = torch.nn.functional.normalize(real_features, dim=1)
    ratios: list[float] = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("relative diversity needs at least two samples per class")
        generated_distance = torch.pdist(generated.index_select(0, selected), p=2).median()
        real_distance = torch.pdist(real.index_select(0, selected), p=2).median()
        ratios.append(float((generated_distance / real_distance.clamp_min(1.0e-8)).detach().item()))
    return ratios


def relative_diversity_loss(features: torch.Tensor, real_features: torch.Tensor, labels: torch.Tensor, *, lower: float = 0.70, upper: float = 1.30) -> torch.Tensor:
    """Smooth bounded relative-diversity penalty used only by recovery training."""
    if not (0.0 < lower < upper):
        raise ValueError("relative-diversity bounds must satisfy 0 < lower < upper")
    generated = torch.nn.functional.normalize(features, dim=1)
    real = torch.nn.functional.normalize(real_features, dim=1)
    terms = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("relative diversity needs at least two samples per class")
        generated_distance = torch.pdist(generated.index_select(0, selected), p=2).median()
        real_distance = torch.pdist(real.index_select(0, selected), p=2).median().detach()
        ratio = generated_distance / real_distance.clamp_min(1.0e-8)
        terms.append(torch.relu(torch.as_tensor(lower, device=ratio.device, dtype=ratio.dtype) - ratio).square() + 0.25 * torch.relu(ratio - upper).square())
    return torch.stack(terms).mean()


def pairwise_diversity_distribution_loss(features: torch.Tensor, real_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Match sorted within-class feature-distance distributions."""
    generated = torch.nn.functional.normalize(features, dim=1)
    real = torch.nn.functional.normalize(real_features, dim=1)
    terms = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("pairwise diversity needs at least two samples per class")
        generated_distance = torch.sort(torch.pdist(generated.index_select(0, selected), p=2)).values
        real_distance = torch.sort(torch.pdist(real.index_select(0, selected), p=2)).values.detach()
        terms.append(torch.nn.functional.smooth_l1_loss(generated_distance, real_distance))
    return torch.stack(terms).mean()


def coverage_ratio_upper_loss(
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
    target: float = 1.35,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Differentiable generated/real coverage ratio with a detached real baseline."""
    if target <= 0.0 or temperature <= 0.0:
        raise ValueError("coverage target and temperature must be positive")
    def class_sinkhorn(generated_f: torch.Tensor, generated_v: torch.Tensor, real_f: torch.Tensor, real_v: torch.Tensor) -> torch.Tensor:
        generated_n = torch.nn.functional.normalize(generated_f, dim=1)
        real_n = torch.nn.functional.normalize(real_f, dim=1)
        feature_cost = torch.cdist(generated_n, real_n).square()
        generated_pixels = sinkhorn_cost_view(generated_v).flatten(1)
        real_pixels = sinkhorn_cost_view(real_v).flatten(1)
        pixel_cost = torch.cdist(generated_pixels, real_pixels, p=1) / generated_pixels.shape[1]
        return log_sinkhorn(feature_cost + 0.25 * pixel_cost, epsilon=epsilon, iterations=iterations)

    observed_terms, baseline_terms = [], []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 4:
            raise ValueError("coverage objective needs at least four samples per class")
        left, right = selected[::2], selected[1::2]
        observed_terms.append(class_sinkhorn(
            generated_features.index_select(0, selected), generated_view.index_select(0, selected),
            real_features.index_select(0, right), real_view.index_select(0, right),
        ))
        baseline_terms.append(class_sinkhorn(
            real_features.index_select(0, left), real_view.index_select(0, left),
            real_features.index_select(0, right), real_view.index_select(0, right),
        ).detach())
    ratio = torch.stack(observed_terms).mean() / torch.stack(baseline_terms).mean().clamp_min(1.0e-12)
    return torch.nn.functional.softplus((ratio - target) / temperature) * temperature


def camera_code_fraction(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiable sample-centered ANOVA code fraction on raw CMOS."""
    if values.ndim != 4 or values.shape[1] != 1 or labels.ndim != 1 or values.shape[0] != labels.numel():
        raise ValueError("camera code fraction expects raw [N,1,H,W] and labels [N]")
    means = torch.stack([values[labels == label].mean(dim=0) for label in range(10)])
    residual = values - means.index_select(0, labels)
    centered = values - values.mean(dim=0, keepdim=True)
    return residual.square().sum() / centered.square().sum().clamp_min(1.0e-12)


def camera_code_fraction_loss(raw: torch.Tensor, labels: torch.Tensor, *, target: float = 0.25) -> torch.Tensor:
    if not 0.0 < target < 1.0:
        raise ValueError("camera code fraction target must lie in (0,1)")
    fraction = camera_code_fraction(raw, labels)
    return torch.relu(torch.as_tensor(target, device=raw.device, dtype=raw.dtype) - fraction).square()


def camera_code_safety_loss(raw: torch.Tensor, labels: torch.Tensor, *, target: float = 0.30, temperature: float = 0.02) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("camera safety temperature must be positive")
    gap = torch.as_tensor(target, device=raw.device, dtype=raw.dtype) - camera_code_fraction(raw, labels)
    return torch.nn.functional.softplus(gap / temperature) * temperature


def raw_from_generator(system: nn.Module, generator: nn.Module, codes: torch.Tensor, labels: torch.Tensor, *, activation_checkpoint: bool) -> torch.Tensor:
    def forward(command: torch.Tensor) -> torch.Tensor:
        return system.forward(system.command(command)).raw_camera.tensor

    command = generator(codes, labels)
    if activation_checkpoint and torch.is_grad_enabled():
        return checkpoint(forward, command, use_reentrant=False)
    return forward(command)


def raw_from_generator_batched(system: nn.Module, generator: nn.Module, codes: torch.Tensor, labels: torch.Tensor, *, microbatch_size: int = 1, activation_checkpoint: bool) -> torch.Tensor:
    if microbatch_size < 1:
        raise ValueError("optical microbatch_size must be positive")
    rows = []
    for start in range(0, codes.shape[0], microbatch_size):
        rows.append(raw_from_generator(system, generator, codes[start : start + microbatch_size], labels[start : start + microbatch_size], activation_checkpoint=activation_checkpoint))
    return torch.cat(rows, dim=0)


def conditional_loss(
    *,
    system: nn.Module,
    generator: nn.Module,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    exposure_scale: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
    activation_checkpoint: bool,
    microbatch_size: int = 1,
    objective_kind: str = "A",
    camera_code_safety_weight: float = 0.0,
    pairwise_diversity_weight: float = 0.0,
    coverage_ratio_weight: float = 0.0,
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
    if objective_kind not in {"A", "B", "C"}:
        raise ValueError("unknown conditional objective kind")
    if objective_kind in {"B", "C"}:
        total = total + camera_code_safety_weight * camera_code_safety_loss(raw, labels, target=0.30, temperature=0.02)
        total = total + pairwise_diversity_weight * pairwise_diversity_distribution_loss(generated_features, real_features, labels)
    if objective_kind == "C":
        total = total + coverage_ratio_weight * coverage_ratio_upper_loss(
            generated_features, generated_view, real_features, real_views, labels,
            epsilon=sinkhorn_epsilon, iterations=sinkhorn_iterations,
        )
    summary = ConditionalLossSummary(*(float(value.detach().item()) for value in (total, sinkhorn, classification, diversity, saturation, orthogonality, phase_tv)))
    return total, summary, raw
