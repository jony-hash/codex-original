"""topology objective terms."""
# ruff: noqa: F403,F401,F405
from __future__ import annotations

from ._shared import *
from .readability import _gaussian_blur_2d

def transform_aux_topology_target(
    target: torch.Tensor,
    transform: str = "none",
    foreground_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Return a diagnostic-only target variant for aux topology supervision."""
    if transform == "none":
        transformed = target.float().clamp(0.0, 1.0)
    elif transform == "lowpass":
        transformed = _gaussian_blur_2d(
            target.float().clamp(0.0, 1.0), sigma=3.0
        ).clamp(0.0, 1.0)
    elif transform == "thick_binary":
        base = (target.float() > foreground_threshold).float()
        thick = F.max_pool2d(base, kernel_size=9, stride=1, padding=4)
        transformed = _gaussian_blur_2d(thick, sigma=1.2).clamp(0.0, 1.0)
    else:
        raise ValueError(
            "aux_topology_target_transform must be one of ['none', 'lowpass', 'thick_binary'], "
            f"got {transform!r}"
        )
    foreground_mask = (transformed > foreground_threshold).float()
    return {
        "target": transformed,
        "foreground_mask": foreground_mask,
        "foreground_ratio": foreground_mask.mean().detach(),
    }

def _encoder_input_topology_metrics(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    planes = []
    for key in ("encoder_input_amplitude", "encoder_input_grid_amplitude"):
        plane = outputs.get(key)
        if plane is None:
            continue
        resized = plane.float()
        if resized.shape[-2:] != target.shape[-2:]:
            resized = F.interpolate(
                resized, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        planes.append(resized.clamp(0.0, 1.0))
    if not planes:
        raise ValueError(
            "encoder_input_topology_weight requires encoder_input_amplitude "
            "or encoder_input_grid_amplitude."
        )

    dice_scores = [_foreground_dice_score(plane, foreground_mask) for plane in planes]
    mse_scores = [torch.mean((plane - target.float()).pow(2)) for plane in planes]
    dice = torch.stack(dice_scores).mean()
    mse = torch.stack(mse_scores).mean()
    return {
        "dice": dice,
        "dice_penalty": 1.0 - dice,
        "mse": mse,
    }

def _aux_topology_metrics(
    outputs: dict[str, torch.Tensor],
    plane_key: str,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not plane_key:
        raise ValueError(
            "aux_topology_plane_key is required when aux topology loss is enabled."
        )
    plane = outputs.get(plane_key)
    if not torch.is_tensor(plane):
        raise ValueError(
            f"aux_topology_plane_key={plane_key!r} was not found in one-shot outputs."
        )
    resized = plane.float()
    if resized.ndim != target.ndim:
        raise ValueError(
            "aux topology plane rank must match target_batch rank, "
            f"got {tuple(resized.shape)} and {tuple(target.shape)}"
        )
    if resized.shape[0] != target.shape[0] or resized.shape[1] != target.shape[1]:
        raise ValueError(
            "aux topology plane batch/channel must match target_batch, "
            f"got {tuple(resized.shape)} and {tuple(target.shape)}"
        )
    if resized.shape[-2:] != target.shape[-2:]:
        resized = F.interpolate(
            resized, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    resized = resized.clamp(0.0, 1.0)
    dice = _foreground_dice_score(resized, foreground_mask)
    mse = torch.mean((resized - target.float()).pow(2))
    return {
        "dice": dice,
        "dice_penalty": 1.0 - dice,
        "mse": mse,
    }

def _resolve_sample_weights(
    reference: torch.Tensor,
    class_labels: torch.Tensor | None,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
) -> torch.Tensor:
    weights = torch.ones(
        reference.shape[0], device=reference.device, dtype=reference.dtype
    )
    if class_labels is None or not class_weight_labels or class_weight == 1.0:
        return weights
    labels = class_labels.detach().reshape(-1).to(device=reference.device)
    if labels.shape[0] != reference.shape[0]:
        raise ValueError(
            "class_labels must match one_shot_loss batch size when class weighting is enabled."
        )
    weak_mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in class_weight_labels:
        weak_mask |= labels == int(label)
    return torch.where(weak_mask, weights * float(class_weight), weights)

def _one_shot_edge_shape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    target_soft = target.clamp(0.0, 1.0)
    foreground_context = F.max_pool2d(
        foreground_mask, kernel_size=5, stride=1, padding=2
    )
    edge_error = (
        _sobel_magnitude(prediction_soft) - _sobel_magnitude(target_soft)
    ).abs()
    local_prediction = F.avg_pool2d(prediction_soft, kernel_size=3, stride=1, padding=1)
    local_target = F.avg_pool2d(target_soft, kernel_size=3, stride=1, padding=1)
    local_error = (local_prediction - local_target).abs()
    edge_samples = _masked_mean_per_sample(edge_error, foreground_context)
    local_samples = _masked_mean_per_sample(local_error, foreground_context)
    return _weighted_sample_mean(edge_samples + 0.5 * local_samples, sample_weights)

def _dilate_foreground_mask(
    foreground_mask: torch.Tensor, dilation: int
) -> torch.Tensor:
    dilation = int(dilation)
    if dilation <= 0:
        return foreground_mask.float()
    if dilation % 2 == 0:
        dilation += 1
    return F.max_pool2d(
        foreground_mask.float(),
        kernel_size=dilation,
        stride=1,
        padding=dilation // 2,
    ).clamp(0.0, 1.0)

def _resolve_clean_target_masks(
    target_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    enabled: bool,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled:
        return target_batch, foreground_mask, background_mask, zero
    clean_result = _build_clean_readable_target(target_batch.detach())
    clean_target = clean_result["clean_target"]
    clean_foreground_mask = clean_target.float()
    return (
        clean_target,
        clean_foreground_mask,
        1.0 - clean_foreground_mask,
        clean_result["clean_target_fg_ratio"],
    )

def _aux_topology_terms(
    outputs: dict[str, torch.Tensor],
    topology_target: torch.Tensor,
    *,
    plane_key: str,
    weight: float,
    dice_weight: float,
    mse_weight: float,
    target_transform: str,
    foreground_threshold: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    target_info = transform_aux_topology_target(
        topology_target,
        target_transform,
        foreground_threshold,
    )
    if weight != 0.0:
        metrics = _aux_topology_metrics(
            outputs,
            plane_key,
            target_info["target"],
            target_info["foreground_mask"],
        )
        dice = metrics["dice"]
        mse = metrics["mse"]
        base = float(dice_weight) * metrics["dice_penalty"] + float(mse_weight) * mse
    else:
        dice = zero
        mse = zero
        base = zero
    return {
        "target_info": target_info,
        "dice": dice,
        "mse": mse,
        "loss": weight * base,
    }

def _aux_detail_terms(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    aux_detail_target: torch.Tensor | None,
    clean_foreground_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    foreground_threshold: float,
    gate_dilation: int,
    shape_weight: float,
    edge_weight: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    gate_dilation = int(gate_dilation)
    if gate_dilation > 0:
        gate_mask = _dilate_foreground_mask(clean_foreground_mask, gate_dilation)
        gate_foreground_ratio = gate_mask.mean().detach()
    else:
        gate_mask = None
        gate_foreground_ratio = zero
    if aux_detail_target is not None and (shape_weight != 0.0 or edge_weight != 0.0):
        if aux_detail_target.shape != target_batch.shape:
            raise ValueError(
                "aux_detail_target must match target_batch shape, "
                f"got {tuple(aux_detail_target.shape)} and {tuple(target_batch.shape)}"
            )
        aux_detail = aux_detail_target.float().clamp(0.0, 1.0)
        aux_detail_foreground = (aux_detail > foreground_threshold).float()
        if gate_mask is not None:
            aux_detail = aux_detail * gate_mask
            aux_detail_foreground = aux_detail_foreground * gate_mask
        active_foreground_ratio = aux_detail_foreground.mean().detach()
        has_foreground = bool(aux_detail_foreground.detach().sum().item() > 0)
        shape_base = (
            _foreground_shape_loss_for_samples(
                camera_prediction, aux_detail_foreground, sample_weights
            )
            if shape_weight != 0.0 and has_foreground
            else zero
        )
        edge_base = (
            _one_shot_edge_shape_loss(
                camera_prediction,
                aux_detail,
                aux_detail_foreground,
                sample_weights,
            )
            if edge_weight != 0.0 and has_foreground
            else zero
        )
    else:
        shape_base = zero
        edge_base = zero
        active_foreground_ratio = zero
    return {
        "gate_mask": gate_mask,
        "gate_foreground_ratio": gate_foreground_ratio,
        "active_foreground_ratio": active_foreground_ratio,
        "shape_base": shape_base,
        "shape_loss": shape_weight * shape_base,
        "edge_base": edge_base,
        "edge_loss": edge_weight * edge_base,
    }
