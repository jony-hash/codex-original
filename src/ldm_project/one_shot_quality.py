from __future__ import annotations

from .stage_constants import (
    ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC,
    ONE_SHOT_DISTILL_BALANCED_QUALITY_SATURATION_TARGET,
    ONE_SHOT_DISTILL_BEST_METRIC,
    ONE_SHOT_DISTILL_QUALITY_METRIC,
    ONE_SHOT_DISTILL_QUALITY_SATURATION_TARGET,
    ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC,
    ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_SATURATION_TARGET,
    ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC,
)


def _one_shot_distill_quality_score(validation_metrics: dict[str, float]) -> float:
    saturation_overage = max(
        0.0,
        validation_metrics["camera_saturation_ratio"] - ONE_SHOT_DISTILL_QUALITY_SATURATION_TARGET,
    )
    return (
        validation_metrics["camera_recon_loss"]
        + 0.03 * (1.0 - validation_metrics["camera_foreground_dice"])
        + saturation_overage
        + 0.05 * validation_metrics["slm_tv"]
    )

def _one_shot_distill_balanced_quality_score(validation_metrics: dict[str, float]) -> float:
    saturation_overage = max(
        0.0,
        validation_metrics["camera_saturation_ratio"] - ONE_SHOT_DISTILL_BALANCED_QUALITY_SATURATION_TARGET,
    )
    return (
        validation_metrics["camera_recon_loss"]
        + 0.05 * (1.0 - validation_metrics["camera_foreground_dice"])
        + 0.15 * validation_metrics.get("foreground_hole_ratio", 0.0)
        + 0.4 * saturation_overage
        + 0.03 * validation_metrics["slm_tv"]
    )

def _one_shot_distill_strict_balanced_quality_score(validation_metrics: dict[str, float]) -> float:
    saturation_overage = max(
        0.0,
        validation_metrics["camera_saturation_ratio"] - ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_SATURATION_TARGET,
    )
    return (
        validation_metrics["camera_recon_loss"]
        + 0.05 * (1.0 - validation_metrics["camera_foreground_dice"])
        + 0.15 * validation_metrics.get("foreground_hole_ratio", 0.0)
        + 0.4 * saturation_overage
        + 0.03 * validation_metrics["slm_tv"]
    )

def _one_shot_distill_strict_readability_score(validation_metrics: dict[str, float]) -> float:
    required_metrics = (
        "paired_corr",
        "paired_dice",
        "camera_high_saturation_ratio",
    )
    missing = [key for key in required_metrics if key not in validation_metrics]
    if missing:
        raise ValueError(
            "strict_readability checkpoint selection requires paired readability metrics; "
            f"missing {missing}"
        )
    # 仅惩罚高饱和（过曝），低饱和（黑色背景）是有效信号。
    saturation_overage = max(0.0, validation_metrics["camera_high_saturation_ratio"] - 0.08)
    return (
        0.45 * (1.0 - validation_metrics["paired_dice"])
        + 0.35 * (1.0 - validation_metrics["paired_corr"])
        + 0.20 * saturation_overage
    )

def _resolve_one_shot_distill_selection(train_cfg, validation_metrics: dict[str, float]) -> tuple[float, str]:
    selection_metric = train_cfg.one_shot.checkpoint_selection_metric
    if selection_metric == "loss":
        return validation_metrics["loss"], ONE_SHOT_DISTILL_BEST_METRIC
    if selection_metric == "quality":
        quality_score = _one_shot_distill_quality_score(validation_metrics)
        validation_metrics["quality_score"] = quality_score
        return quality_score, ONE_SHOT_DISTILL_QUALITY_METRIC
    if selection_metric == "balanced_quality":
        balanced_quality_score = _one_shot_distill_balanced_quality_score(validation_metrics)
        validation_metrics["balanced_quality_score"] = balanced_quality_score
        return balanced_quality_score, ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC
    if selection_metric == "strict_balanced_quality":
        strict_balanced_quality_score = _one_shot_distill_strict_balanced_quality_score(validation_metrics)
        validation_metrics["strict_balanced_quality_score"] = strict_balanced_quality_score
        return strict_balanced_quality_score, ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC
    if selection_metric == "strict_readability":
        strict_readability_score = _one_shot_distill_strict_readability_score(validation_metrics)
        validation_metrics["strict_readability_score"] = strict_readability_score
        return strict_readability_score, ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC
    raise ValueError(
        "one_shot.checkpoint_selection_metric must be 'loss', 'quality', 'balanced_quality', "
        "'strict_balanced_quality', or 'strict_readability', "
        f"got {selection_metric!r}"
    )
