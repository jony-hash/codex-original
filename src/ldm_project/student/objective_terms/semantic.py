"""semantic objective terms."""
# ruff: noqa: F403,F401,F405
from __future__ import annotations

from ._shared import *

def _normalize_for_label_margin(tensor: torch.Tensor) -> torch.Tensor:
    flattened = tensor.float().flatten(1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    normalized = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(
        1.0e-6
    )
    return F.normalize(normalized, dim=1, eps=1.0e-6)

def _student_prototype_correlations(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = class_labels.detach().reshape(-1).to(device=camera_prediction.device)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError(
            "class_labels must match one_shot_loss batch size for student prototype losses."
        )
    if prototypes.ndim != 4 or prototypes.shape[1] != camera_prediction.shape[1]:
        raise ValueError(
            "student_label_margin_prototypes must be [num_classes, channels, height, width] "
            f"with matching channels, got {tuple(prototypes.shape)}"
        )
    prototype_batch = prototypes.to(
        device=camera_prediction.device, dtype=camera_prediction.dtype
    )
    if prototype_batch.shape[-2:] != camera_prediction.shape[-2:]:
        prototype_batch = F.interpolate(
            prototype_batch,
            size=camera_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if (
        int(labels.max().item()) >= prototype_batch.shape[0]
        or int(labels.min().item()) < 0
    ):
        raise ValueError(
            "class_labels exceed student_label_margin_prototypes class dimension."
        )

    sample_features = _normalize_for_label_margin(camera_prediction)
    prototype_features = _normalize_for_label_margin(prototype_batch)
    correlations = sample_features @ prototype_features.t()
    return labels, correlations

def _student_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    target: float,
    focus_label: int | None,
    focus_weight: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_label_margin_base": zero,
            "student_label_margin_mean": zero,
            "student_label_correct_ratio": zero,
        }
    labels, correlations = _student_prototype_correlations(
        camera_prediction, class_labels, prototypes
    )

    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_label_margin_base": zero,
            "student_label_margin_mean": zero,
            "student_label_correct_ratio": zero,
        }

    target_corr = correlations.gather(1, labels.view(-1, 1)).squeeze(1)
    other_corr = correlations.clone()
    other_corr.scatter_(1, labels.view(-1, 1), -2.0)
    best_other = other_corr.max(dim=1).values
    margins = target_corr - best_other
    active_margins = margins[valid_mask]
    active_labels = labels[valid_mask]
    nearest_labels = correlations.argmax(dim=1)
    active_correct = (nearest_labels[valid_mask] == labels[valid_mask]).float()
    penalties = F.relu(float(target) - active_margins)
    if focus_label is not None and float(focus_weight) != 1.0:
        weights = torch.where(
            active_labels == int(focus_label),
            penalties.new_full(penalties.shape, float(focus_weight)),
            torch.ones_like(penalties),
        )
        base = (penalties * weights).sum() / weights.sum().clamp_min(1.0e-6)
    else:
        base = penalties.mean()
    return {
        "student_label_margin_base": base,
        "student_label_margin_mean": active_margins.mean().detach(),
        "student_label_correct_ratio": active_correct.mean().detach(),
    }

def _student_roi_prototype_correlations(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    prototypes: torch.Tensor,
    foreground_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = class_labels.detach().reshape(-1).to(device=camera_prediction.device)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError(
            "class_labels must match one_shot_loss batch size for student ROI prototype losses."
        )
    if prototypes.ndim != 4 or prototypes.shape[1] != camera_prediction.shape[1]:
        raise ValueError(
            "student_label_margin_prototypes must be [num_classes, channels, height, width] "
            f"with matching channels, got {tuple(prototypes.shape)}"
        )
    prototype_batch = prototypes.to(
        device=camera_prediction.device, dtype=camera_prediction.dtype
    )
    if prototype_batch.shape[-2:] != camera_prediction.shape[-2:]:
        prototype_batch = F.interpolate(
            prototype_batch,
            size=camera_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if (
        int(labels.max().item()) >= prototype_batch.shape[0]
        or int(labels.min().item()) < 0
    ):
        raise ValueError(
            "class_labels exceed student_label_margin_prototypes class dimension."
        )

    sample_flat = camera_prediction.float().flatten(1)
    prototype_flat = prototype_batch.float().flatten(1)
    roi_source = prototype_batch.index_select(0, labels).float()
    roi_mask = (roi_source > float(foreground_threshold)).float().flatten(1)
    mask_sum = roi_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    sample_mean = (sample_flat * roi_mask).sum(dim=1, keepdim=True) / mask_sum
    sample_centered = (sample_flat - sample_mean) * roi_mask
    prototype_mean = (prototype_flat.unsqueeze(0) * roi_mask.unsqueeze(1)).sum(
        dim=2
    ) / mask_sum
    prototype_centered = (
        prototype_flat.unsqueeze(0) - prototype_mean.unsqueeze(2)
    ) * roi_mask.unsqueeze(1)

    numerator = (sample_centered.unsqueeze(1) * prototype_centered).sum(dim=2)
    sample_norm = sample_centered.pow(2).sum(dim=1, keepdim=True).sqrt()
    prototype_norm = prototype_centered.pow(2).sum(dim=2).sqrt()
    correlations = numerator / (sample_norm * prototype_norm).clamp_min(1.0e-6)
    return labels, correlations

def _student_roi_prototype_ce_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    temperature: float,
    foreground_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_roi_prototype_ce_base": zero,
            "student_roi_prototype_ce_correct_ratio": zero,
            "student_roi_prototype_ce_count": zero,
        }
    labels, correlations = _student_roi_prototype_correlations(
        camera_prediction,
        class_labels,
        prototypes,
        foreground_threshold,
    )
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_roi_prototype_ce_base": zero,
            "student_roi_prototype_ce_correct_ratio": zero,
            "student_roi_prototype_ce_count": zero,
        }
    logits = correlations[valid_mask] / float(temperature)
    active_labels = labels[valid_mask]
    predictions = logits.argmax(dim=1)
    return {
        "student_roi_prototype_ce_base": F.cross_entropy(logits, active_labels),
        "student_roi_prototype_ce_correct_ratio": (predictions == active_labels)
        .float()
        .mean()
        .detach(),
        "student_roi_prototype_ce_count": torch.tensor(
            float(active_labels.numel()), device=camera_prediction.device
        ),
    }

def _student_roi_negative_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    margin_config: tuple[tuple[int, int, float], ...],
    foreground_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None or not margin_config:
        return {
            "student_roi_negative_label_margin_base": zero,
            "student_roi_negative_label_margin_mean": zero,
            "student_roi_negative_label_margin_count": zero,
        }
    labels, correlations = _student_roi_prototype_correlations(
        camera_prediction,
        class_labels,
        prototypes,
        foreground_threshold,
    )
    penalties = []
    margins = []
    for target_class, negative_class, target_margin in margin_config:
        active = labels == int(target_class)
        if not active.any():
            continue
        target_corr = correlations[active, int(target_class)]
        negative_corr = correlations[active, int(negative_class)]
        pair_margins = target_corr - negative_corr
        margins.append(pair_margins)
        penalties.append(F.relu(float(target_margin) - pair_margins))
    if not penalties:
        return {
            "student_roi_negative_label_margin_base": zero,
            "student_roi_negative_label_margin_mean": zero,
            "student_roi_negative_label_margin_count": zero,
        }
    penalty_tensor = torch.cat(penalties)
    margin_tensor = torch.cat(margins)
    return {
        "student_roi_negative_label_margin_base": penalty_tensor.mean(),
        "student_roi_negative_label_margin_mean": margin_tensor.mean().detach(),
        "student_roi_negative_label_margin_count": torch.tensor(
            float(penalty_tensor.numel()),
            device=camera_prediction.device,
        ),
    }

def _student_prototype_ce_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    temperature: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_prototype_ce_base": zero,
            "student_prototype_ce_correct_ratio": zero,
            "student_prototype_ce_count": zero,
        }
    labels, correlations = _student_prototype_correlations(
        camera_prediction, class_labels, prototypes
    )
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_prototype_ce_base": zero,
            "student_prototype_ce_correct_ratio": zero,
            "student_prototype_ce_count": zero,
        }
    logits = correlations[valid_mask] / float(temperature)
    active_labels = labels[valid_mask]
    predictions = logits.argmax(dim=1)
    return {
        "student_prototype_ce_base": F.cross_entropy(logits, active_labels),
        "student_prototype_ce_correct_ratio": (predictions == active_labels)
        .float()
        .mean()
        .detach(),
        "student_prototype_ce_count": torch.tensor(
            float(active_labels.numel()), device=camera_prediction.device
        ),
    }

def _student_negative_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    margin_config: tuple[tuple[int, int, float], ...],
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None or not margin_config:
        return {
            "student_negative_label_margin_base": zero,
            "student_negative_label_margin_mean": zero,
            "student_negative_label_margin_count": zero,
        }
    labels, correlations = _student_prototype_correlations(
        camera_prediction, class_labels, prototypes
    )
    penalties = []
    margins = []
    for target_class, negative_class, target_margin in margin_config:
        active = labels == int(target_class)
        if not active.any():
            continue
        target_corr = correlations[active, int(target_class)]
        negative_corr = correlations[active, int(negative_class)]
        pair_margins = target_corr - negative_corr
        margins.append(pair_margins)
        penalties.append(F.relu(float(target_margin) - pair_margins))
    if not penalties:
        return {
            "student_negative_label_margin_base": zero,
            "student_negative_label_margin_mean": zero,
            "student_negative_label_margin_count": zero,
        }
    penalty_tensor = torch.cat(penalties)
    margin_tensor = torch.cat(margins)
    return {
        "student_negative_label_margin_base": penalty_tensor.mean(),
        "student_negative_label_margin_mean": margin_tensor.mean().detach(),
        "student_negative_label_margin_count": torch.tensor(
            float(penalty_tensor.numel()), device=camera_prediction.device
        ),
    }

def _resize_aux_semantic_plane(
    outputs: dict[str, torch.Tensor],
    plane_key: str,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    if not plane_key:
        raise ValueError(
            "aux_semantic_plane_key is required when aux semantic losses are enabled."
        )
    plane = outputs.get(plane_key)
    if not torch.is_tensor(plane):
        raise ValueError(
            f"aux_semantic_plane_key={plane_key!r} was not found in one-shot outputs."
        )
    plane = plane.float()
    if plane.ndim != target_batch.ndim:
        raise ValueError(
            "aux semantic plane rank must match target_batch rank, "
            f"got {tuple(plane.shape)} and {tuple(target_batch.shape)}"
        )
    if (
        plane.shape[0] != target_batch.shape[0]
        or plane.shape[1] != target_batch.shape[1]
    ):
        raise ValueError(
            "aux semantic plane batch/channel must match target_batch, "
            f"got {tuple(plane.shape)} and {tuple(target_batch.shape)}"
        )
    if plane.shape[-2:] != target_batch.shape[-2:]:
        plane = F.interpolate(
            plane, size=target_batch.shape[-2:], mode="bilinear", align_corners=False
        )
    return plane.clamp(0.0, 1.0)

def _student_label_margin_terms(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    weight: float,
    labels: tuple[int, ...],
    target: float,
    focus_label: int | None,
    focus_weight: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if weight != 0.0 or prototypes is not None:
        if weight != 0.0 and prototypes is None:
            raise ValueError(
                "student_label_margin_weight requires student_label_margin_prototypes."
            )
        metrics = _student_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            labels,
            target,
            focus_label,
            focus_weight,
        )
        base = metrics["student_label_margin_base"]
        margin_mean = metrics["student_label_margin_mean"]
        correct_ratio = metrics["student_label_correct_ratio"]
    else:
        base = zero
        margin_mean = zero
        correct_ratio = zero
    return {
        "base": base,
        "loss": weight * base,
        "margin_mean": margin_mean,
        "correct_ratio": correct_ratio,
    }

def _student_prototype_objective_terms(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    prototype_ce_weight: float,
    prototype_ce_labels: tuple[int, ...],
    prototype_ce_temperature: float,
    negative_margin_weight: float,
    negative_margin_config: tuple[tuple[int, int, float], ...],
    roi_prototype_ce_weight: float,
    roi_prototype_ce_labels: tuple[int, ...],
    roi_prototype_ce_temperature: float,
    roi_negative_margin_weight: float,
    roi_negative_margin_config: tuple[tuple[int, int, float], ...],
    roi_foreground_threshold: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        prototype_ce_weight != 0.0
        or negative_margin_weight != 0.0
        or roi_prototype_ce_weight != 0.0
        or roi_negative_margin_weight != 0.0
    ) and prototypes is None:
        raise ValueError(
            "student prototype label objectives "
            "require student_label_margin_prototypes."
        )
    if prototype_ce_weight != 0.0:
        prototype_metrics = _student_prototype_ce_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            prototype_ce_labels,
            prototype_ce_temperature,
        )
        prototype_base = prototype_metrics["student_prototype_ce_base"]
        prototype_correct_ratio = prototype_metrics[
            "student_prototype_ce_correct_ratio"
        ]
        prototype_count = prototype_metrics["student_prototype_ce_count"]
    else:
        prototype_base = zero
        prototype_correct_ratio = zero
        prototype_count = zero
    if negative_margin_weight != 0.0:
        negative_metrics = _student_negative_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            negative_margin_config,
        )
        negative_base = negative_metrics["student_negative_label_margin_base"]
        negative_mean = negative_metrics["student_negative_label_margin_mean"]
        negative_count = negative_metrics["student_negative_label_margin_count"]
    else:
        negative_base = zero
        negative_mean = zero
        negative_count = zero
    if roi_prototype_ce_weight != 0.0:
        roi_prototype_metrics = _student_roi_prototype_ce_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            roi_prototype_ce_labels,
            roi_prototype_ce_temperature,
            roi_foreground_threshold,
        )
        roi_prototype_base = roi_prototype_metrics["student_roi_prototype_ce_base"]
        roi_prototype_correct_ratio = roi_prototype_metrics[
            "student_roi_prototype_ce_correct_ratio"
        ]
        roi_prototype_count = roi_prototype_metrics["student_roi_prototype_ce_count"]
    else:
        roi_prototype_base = zero
        roi_prototype_correct_ratio = zero
        roi_prototype_count = zero
    if roi_negative_margin_weight != 0.0:
        roi_negative_metrics = _student_roi_negative_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            roi_negative_margin_config,
            roi_foreground_threshold,
        )
        roi_negative_base = roi_negative_metrics[
            "student_roi_negative_label_margin_base"
        ]
        roi_negative_mean = roi_negative_metrics[
            "student_roi_negative_label_margin_mean"
        ]
        roi_negative_count = roi_negative_metrics[
            "student_roi_negative_label_margin_count"
        ]
    else:
        roi_negative_base = zero
        roi_negative_mean = zero
        roi_negative_count = zero
    return {
        "prototype_base": prototype_base,
        "prototype_loss": prototype_ce_weight * prototype_base,
        "prototype_correct_ratio": prototype_correct_ratio,
        "prototype_count": prototype_count,
        "negative_base": negative_base,
        "negative_loss": negative_margin_weight * negative_base,
        "negative_mean": negative_mean,
        "negative_count": negative_count,
        "roi_prototype_base": roi_prototype_base,
        "roi_prototype_loss": roi_prototype_ce_weight * roi_prototype_base,
        "roi_prototype_correct_ratio": roi_prototype_correct_ratio,
        "roi_prototype_count": roi_prototype_count,
        "roi_negative_base": roi_negative_base,
        "roi_negative_loss": roi_negative_margin_weight * roi_negative_base,
        "roi_negative_mean": roi_negative_mean,
        "roi_negative_count": roi_negative_count,
    }

def _aux_semantic_objective_terms(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    plane_key: str,
    weight: float,
    prototype_ce_weight: float,
    negative_margin_weight: float,
    prototype_ce_labels: tuple[int, ...],
    prototype_ce_temperature: float,
    negative_margin_config: tuple[tuple[int, int, float], ...],
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    active = weight != 0.0 and (
        prototype_ce_weight != 0.0 or negative_margin_weight != 0.0
    )
    if active:
        if prototypes is None:
            raise ValueError(
                "aux semantic objectives require student_label_margin_prototypes."
            )
        semantic_plane = _resize_aux_semantic_plane(outputs, plane_key, target_batch)
        if prototype_ce_weight != 0.0:
            prototype_metrics = _student_prototype_ce_metrics(
                semantic_plane,
                class_labels,
                prototypes,
                prototype_ce_labels,
                prototype_ce_temperature,
            )
            prototype_base = prototype_metrics["student_prototype_ce_base"]
            prototype_correct_ratio = prototype_metrics[
                "student_prototype_ce_correct_ratio"
            ]
            prototype_count = prototype_metrics["student_prototype_ce_count"]
        else:
            prototype_base = zero
            prototype_correct_ratio = zero
            prototype_count = zero
        if negative_margin_weight != 0.0:
            negative_metrics = _student_negative_label_margin_metrics(
                semantic_plane,
                class_labels,
                prototypes,
                negative_margin_config,
            )
            negative_base = negative_metrics["student_negative_label_margin_base"]
            negative_mean = negative_metrics["student_negative_label_margin_mean"]
            negative_count = negative_metrics["student_negative_label_margin_count"]
        else:
            negative_base = zero
            negative_mean = zero
            negative_count = zero
    else:
        prototype_base = zero
        prototype_correct_ratio = zero
        prototype_count = zero
        negative_base = zero
        negative_mean = zero
        negative_count = zero
    prototype_loss = weight * prototype_ce_weight * prototype_base
    negative_loss = weight * negative_margin_weight * negative_base
    return {
        "loss": prototype_loss + negative_loss,
        "prototype_base": prototype_base,
        "prototype_loss": prototype_loss,
        "prototype_correct_ratio": prototype_correct_ratio,
        "prototype_count": prototype_count,
        "negative_base": negative_base,
        "negative_loss": negative_loss,
        "negative_mean": negative_mean,
        "negative_count": negative_count,
    }
