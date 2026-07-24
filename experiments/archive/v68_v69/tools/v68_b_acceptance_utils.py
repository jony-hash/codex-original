"""Small, dependency-light helpers shared by the v68.6 B acceptance tools."""

from __future__ import annotations

from typing import Any

import torch


def balanced_factor_decomposition(
    values: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """ANOVA energy fractions for a balanced ``[code, label, ...]`` tensor.

    Fractions are computed on centered effects and therefore remain meaningful
    for raw intensities without normalising individual samples.
    """
    if values.ndim < 3 or values.shape[:2] != (100, 10):
        raise ValueError("expected balanced values with shape [100, 10, ...]")
    x = values.float()
    grand = x.mean((0, 1))
    code = x.mean(1)
    label = x.mean(0)
    code_effect = code - grand
    label_effect = label - grand
    interaction = x - code[:, None] - label[None] + grand
    total_ss = x.sub(grand).square().sum().item()
    code_ss = code_effect.square().sum().item() * 10.0
    label_ss = label_effect.square().sum().item() * 100.0
    interaction_ss = interaction.square().sum().item()
    denom = max(total_ss, 1.0e-12)
    factors = {
        "total_energy": float(total_ss),
        "code_fraction": float(code_ss / denom),
        "label_fraction": float(label_ss / denom),
        "interaction_fraction": float(interaction_ss / denom),
        "finite": bool(torch.isfinite(x).all().item()),
    }
    effects = {"grand": grand, "code": code, "label": label, "interaction": interaction}
    return factors, effects


def sample_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = prediction.float().flatten(1)
    t = target.float().flatten(1)
    p = p - p.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    den = p.square().mean(1).sqrt() * t.square().mean(1).sqrt()
    return (p * t).mean(1) / den.clamp_min(1.0e-8)


def sample_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = prediction.float().clamp(0, 1) > 0.2
    t = target.float().clamp(0, 1) > 0.2
    dims = tuple(range(1, p.ndim))
    inter = (p & t).float().sum(dims)
    den = p.float().sum(dims) + t.float().sum(dims)
    return 2 * inter / den.clamp_min(1.0)


def metric_bundle(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: torch.Tensor,
) -> dict[str, Any]:
    corr = sample_correlation(prediction, targets)
    dice = sample_dice(prediction, targets)
    flat_p = prediction.float().flatten(1)
    flat_t = prototype_bank.float().flatten(1)
    flat_p = flat_p - flat_p.mean(1, keepdim=True)
    flat_t = flat_t - flat_t.mean(1, keepdim=True)
    scores = (flat_p @ flat_t.T) / flat_p.shape[1]
    scores = scores / (
        flat_p.square().mean(1, keepdim=True).sqrt() * flat_t.square().mean(1).sqrt()
    ).clamp_min(1e-8)
    predicted = scores.argmax(1)
    per_label: dict[str, Any] = {}
    for label in range(10):
        mask = labels == label
        if bool(mask.any()):
            per_label[str(label)] = {
                "sample_count": int(mask.sum()),
                "corr": float(corr[mask].mean()),
                "dice": float(dice[mask].mean()),
                "accuracy": float((predicted[mask] == label).float().mean()),
            }
    return {
        "clean_prototype_corr": float(corr.mean()),
        "clean_prototype_dice": float(dice.mean()),
        "prototype_accuracy": float((predicted == labels).float().mean()),
        "high_saturation": float((prediction >= 0.99).float().mean()),
        "low_saturation": float((prediction <= 0.01).float().mean()),
        "per_label": per_label,
    }


def validate_b_architecture(
    config: Any, expected_scale: float = 0.4913053810596466
) -> None:
    if config.one_shot.generator_e_arch != "label_template_code_residual":
        raise ValueError("B audit requires label_template_code_residual")
    if config.one_shot.diagnostic_camera_normalization != "fixed_exposure":
        raise ValueError("B audit requires fixed_exposure normalization")
    scale = config.one_shot.diagnostic_camera_exposure_scale
    if scale is None or abs(float(scale) - expected_scale) > 1e-12:
        raise ValueError("B audit requires the registered locked exposure scale")
