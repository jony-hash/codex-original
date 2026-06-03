from __future__ import annotations

import torch
import torch.nn.functional as F

from .training_loss_common import (
    _camera_artifact_metrics,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _foreground_dice_score,
    _foreground_shape_loss_for_samples,
    _masked_mean_per_sample,
    _sobel_magnitude,
    _total_variation,
    _weighted_reconstruction_metrics,
    _weighted_reconstruction_metrics_for_samples,
    _weighted_sample_mean,
    _zero_like_loss,
)


def _one_shot_pairwise_diversity_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    min_teacher_mse: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or camera_prediction.shape[0] < 2:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when diversity weights are enabled.")

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_distances = []
    teacher_distances = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        student_class = prediction_flat.index_select(0, indices)
        teacher_class = target_flat.index_select(0, indices)
        student_pairwise = (student_class[:, None, :] - student_class[None, :, :]).pow(2).mean(dim=-1)
        teacher_pairwise = (teacher_class[:, None, :] - teacher_class[None, :, :]).pow(2).mean(dim=-1)
        pair_mask = torch.triu(
            torch.ones(indices.numel(), indices.numel(), dtype=torch.bool, device=camera_prediction.device),
            diagonal=1,
        )
        student_distances.append(student_pairwise[pair_mask])
        teacher_distances.append(teacher_pairwise[pair_mask])

    if not student_distances:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    student_distance = torch.cat(student_distances)
    teacher_distance = torch.cat(teacher_distances)
    active = teacher_distance >= float(min_teacher_mse)
    if not active.any():
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": teacher_distance.mean().detach(),
            "student_pairwise_mse": student_distance.mean().detach(),
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    active_student = student_distance[active]
    active_teacher = teacher_distance[active].clamp_min(1.0e-8)
    # 只在 teacher 自身存在差异的同类样本对上惩罚塌缩，避免鼓励无意义噪声。
    diversity_ratio = active_student / active_teacher
    code_diversity_base = F.relu(1.0 - diversity_ratio).pow(2).mean()
    teacher_pairwise_base = F.smooth_l1_loss(
        torch.log(active_student.clamp_min(1.0e-8)),
        torch.log(active_teacher),
    )
    return {
        "code_diversity_base": code_diversity_base,
        "teacher_pairwise_diversity_base": teacher_pairwise_base,
        "teacher_pairwise_mse": active_teacher.mean().detach(),
        "student_pairwise_mse": active_student.mean().detach(),
        "diversity_ratio": diversity_ratio.mean().detach(),
        "diversity_pair_count": active.float().sum().detach(),
    }


def _one_shot_cross_class_pairwise_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    min_teacher_mse: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or camera_prediction.shape[0] < 2:
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": zero,
            "cross_class_student_pairwise_mse": zero,
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when cross-class diversity is enabled.")

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_pairwise = (prediction_flat[:, None, :] - prediction_flat[None, :, :]).pow(2).mean(dim=-1)
    teacher_pairwise = (target_flat[:, None, :] - target_flat[None, :, :]).pow(2).mean(dim=-1)
    different_class = labels[:, None] != labels[None, :]
    pair_mask = torch.triu(
        torch.ones(labels.numel(), labels.numel(), dtype=torch.bool, device=camera_prediction.device),
        diagonal=1,
    ) & different_class
    if not pair_mask.any():
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": zero,
            "cross_class_student_pairwise_mse": zero,
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    student_distance = student_pairwise[pair_mask]
    teacher_distance = teacher_pairwise[pair_mask]
    active = teacher_distance >= float(min_teacher_mse)
    if not active.any():
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": teacher_distance.mean().detach(),
            "cross_class_student_pairwise_mse": student_distance.mean().detach(),
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    active_student = student_distance[active]
    active_teacher = teacher_distance[active].clamp_min(1.0e-8)
    pairwise_base = F.smooth_l1_loss(
        torch.log(active_student.clamp_min(1.0e-8)),
        torch.log(active_teacher),
    )
    return {
        "cross_class_pairwise_base": pairwise_base,
        "cross_class_teacher_pairwise_mse": active_teacher.mean().detach(),
        "cross_class_student_pairwise_mse": active_student.mean().detach(),
        "cross_class_diversity_ratio": (active_student / active_teacher).mean().detach(),
        "cross_class_pair_count": active.float().sum().detach(),
    }


def _pairwise_mse_summary(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] < 2:
        return _zero_like_loss(tensor)
    flattened = tensor.detach().float().flatten(1)
    pairwise = (flattened[:, None, :] - flattened[None, :, :]).pow(2).mean(dim=-1)
    pair_mask = torch.triu(
        torch.ones(tensor.shape[0], tensor.shape[0], dtype=torch.bool, device=tensor.device),
        diagonal=1,
    )
    return pairwise[pair_mask].mean()


def _per_class_std_summary(tensor: torch.Tensor, class_labels: torch.Tensor | None) -> torch.Tensor:
    if class_labels is None:
        return _zero_like_loss(tensor)
    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != tensor.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size for SLM diagnostics.")
    flattened = tensor.detach().float().flatten(1)
    class_stds = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        class_stds.append(flattened.index_select(0, indices).std(dim=0, unbiased=False).mean())
    if not class_stds:
        return _zero_like_loss(tensor)
    return torch.stack(class_stds).mean()


def _camera_hotspot_loss(camera_prediction: torch.Tensor) -> torch.Tensor:
    return F.relu(camera_prediction - 1.0).pow(2).mean()


def _camera_hotspot_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    return (camera_prediction > 1.0).float().mean()


def _resolve_sample_weights(
    reference: torch.Tensor,
    class_labels: torch.Tensor | None,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
) -> torch.Tensor:
    weights = torch.ones(reference.shape[0], device=reference.device, dtype=reference.dtype)
    if class_labels is None or not class_weight_labels or class_weight == 1.0:
        return weights
    labels = class_labels.detach().reshape(-1).to(device=reference.device)
    if labels.shape[0] != reference.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when class weighting is enabled.")
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
    foreground_context = F.max_pool2d(foreground_mask, kernel_size=5, stride=1, padding=2)
    edge_error = (_sobel_magnitude(prediction_soft) - _sobel_magnitude(target_soft)).abs()
    local_prediction = F.avg_pool2d(prediction_soft, kernel_size=3, stride=1, padding=1)
    local_target = F.avg_pool2d(target_soft, kernel_size=3, stride=1, padding=1)
    local_error = (local_prediction - local_target).abs()
    edge_samples = _masked_mean_per_sample(edge_error, foreground_context)
    local_samples = _masked_mean_per_sample(local_error, foreground_context)
    return _weighted_sample_mean(edge_samples + 0.5 * local_samples, sample_weights)



def one_shot_loss(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    foreground_threshold: float = 0.05,
    foreground_weight: float = 3.0,
    background_weight: float = 1.0,
    class_recon_weight_labels: tuple[int, ...] = (),
    class_recon_weight: float = 1.0,
    shape_loss_weight: float = 0.0,
    edge_loss_weight: float = 0.0,
    slm_smoothness_weight: float = 0.0,
    slm_contrast_weight: float = 0.0,
    camera_saturation_weight: float = 0.0,
    camera_topology_weight: float = 0.0,
    camera_artifact_weight: float = 0.0,
    code_diversity_weight: float = 0.0,
    teacher_pairwise_diversity_weight: float = 0.0,
    cross_class_pairwise_weight: float = 0.0,
    diversity_min_teacher_mse: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    foreground_mask = (target_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    sample_weights = _resolve_sample_weights(
        target_batch,
        class_labels,
        class_recon_weight_labels,
        class_recon_weight,
    )
    has_class_weights = class_labels is not None and bool(class_recon_weight_labels) and class_recon_weight != 1.0
    if has_class_weights:
        camera_metrics = _weighted_reconstruction_metrics_for_samples(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
            sample_weights,
        )
    else:
        camera_metrics = _weighted_reconstruction_metrics(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
        )
    zero = _zero_like_loss(camera_prediction)
    shape_base = (
        _foreground_shape_loss_for_samples(camera_prediction, foreground_mask, sample_weights)
        if shape_loss_weight != 0.0
        else zero
    )
    shape_loss = shape_loss_weight * shape_base
    edge_base = (
        _one_shot_edge_shape_loss(camera_prediction, target_batch, foreground_mask, sample_weights)
        if edge_loss_weight != 0.0
        else zero
    )
    edge_loss = edge_loss_weight * edge_base
    slm_tv = _total_variation(outputs["slm_amplitude"])
    slm_smoothness = slm_smoothness_weight * slm_tv
    slm_amplitude = outputs["slm_amplitude"]
    if slm_contrast_weight != 0.0:
        grid_tensor = outputs.get("encoder_input_grid_amplitude")
        if grid_tensor is not None:
            grid_h, grid_w = grid_tensor.shape[-2:]
            slm_grid = F.interpolate(slm_amplitude, size=(grid_h, grid_w), mode="area")
        else:
            slm_grid = slm_amplitude
        per_sample_slm_std = slm_grid.reshape(slm_amplitude.shape[0], -1).std(dim=1)
        slm_contrast_base = F.relu(0.12 - per_sample_slm_std).mean()
        slm_contrast = slm_contrast_weight * slm_contrast_base
    else:
        per_sample_slm_std = slm_amplitude.reshape(slm_amplitude.shape[0], -1).std(dim=1)
        slm_contrast_base = zero
        slm_contrast = zero
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    # 拓扑约束：保持笔画连接和弯折的结构，对复杂数字(3/4/5)尤为重要。
    camera_topology_base = (
        _camera_topology_loss(camera_prediction, target_batch, foreground_mask)
        if camera_topology_weight != 0.0
        else zero
    )
    camera_topology = camera_topology_weight * camera_topology_base
    # 抗黑斑：惩罚前景笔画内部的暗斑和粗糙纹理。
    if camera_artifact_weight != 0.0:
        artifact_metrics = _camera_artifact_metrics(
            camera_loss_prediction, target_batch, foreground_mask
        )
        camera_artifact_base = artifact_metrics["artifact_base"]
        foreground_hole_ratio = artifact_metrics["foreground_hole_ratio"]
    else:
        camera_artifact_base = zero
        foreground_hole_ratio = zero
    camera_artifact = camera_artifact_weight * camera_artifact_base
    diversity_metrics = _one_shot_pairwise_diversity_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        diversity_min_teacher_mse,
    )
    code_diversity = code_diversity_weight * diversity_metrics["code_diversity_base"]
    teacher_pairwise_diversity = (
        teacher_pairwise_diversity_weight * diversity_metrics["teacher_pairwise_diversity_base"]
    )
    cross_class_metrics = _one_shot_cross_class_pairwise_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        diversity_min_teacher_mse,
    )
    cross_class_pairwise = cross_class_pairwise_weight * cross_class_metrics["cross_class_pairwise_base"]
    slm_pairwise_mse = _pairwise_mse_summary(slm_amplitude)
    slm_per_class_std_mean = _per_class_std_summary(slm_amplitude, class_labels)
    total = (
        camera_metrics["recon_loss"]
        + shape_loss
        + edge_loss
        + slm_smoothness
        + slm_contrast
        + camera_saturation
        + camera_topology
        + camera_artifact
        + code_diversity
        + teacher_pairwise_diversity
        + cross_class_pairwise
    )
    return {
        "loss": total,
        "camera_recon_loss": camera_metrics["recon_loss"].detach(),
        "camera_fg_mse": camera_metrics["fg_mse"],
        "camera_bg_mse": camera_metrics["bg_mse"],
        "shape_loss": shape_loss.detach(),
        "shape_penalty": shape_base.detach(),
        "edge_loss": edge_loss.detach(),
        "edge_penalty": edge_base.detach(),
        "slm_smoothness_loss": slm_smoothness.detach(),
        "slm_tv": slm_tv.detach(),
        "slm_contrast_loss": slm_contrast.detach(),
        "slm_contrast_penalty": slm_contrast_base.detach(),
        "slm_per_sample_std_mean": per_sample_slm_std.mean().detach(),
        "slm_per_sample_std_min": per_sample_slm_std.min().detach(),
        "camera_saturation_loss": camera_saturation.detach(),
        "camera_saturation_penalty": camera_saturation_base.detach(),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "camera_topology_loss": camera_topology.detach(),
        "camera_artifact_loss": camera_artifact.detach(),
        "foreground_hole_ratio": foreground_hole_ratio.detach() if hasattr(foreground_hole_ratio, 'detach') else foreground_hole_ratio,
        "camera_foreground_dice": _foreground_dice_score(camera_prediction.detach(), foreground_mask),
        "code_diversity_loss": code_diversity.detach(),
        "code_diversity_penalty": diversity_metrics["code_diversity_base"].detach(),
        "teacher_pairwise_diversity_loss": teacher_pairwise_diversity.detach(),
        "teacher_pairwise_diversity_penalty": diversity_metrics["teacher_pairwise_diversity_base"].detach(),
        "teacher_pairwise_mse": diversity_metrics["teacher_pairwise_mse"].detach(),
        "student_pairwise_mse": diversity_metrics["student_pairwise_mse"].detach(),
        "diversity_ratio": diversity_metrics["diversity_ratio"].detach(),
        "diversity_pair_count": diversity_metrics["diversity_pair_count"].detach(),
        "cross_class_pairwise_loss": cross_class_pairwise.detach(),
        "cross_class_pairwise_penalty": cross_class_metrics["cross_class_pairwise_base"].detach(),
        "cross_class_teacher_pairwise_mse": cross_class_metrics["cross_class_teacher_pairwise_mse"].detach(),
        "cross_class_student_pairwise_mse": cross_class_metrics["cross_class_student_pairwise_mse"].detach(),
        "cross_class_diversity_ratio": cross_class_metrics["cross_class_diversity_ratio"].detach(),
        "cross_class_pair_count": cross_class_metrics["cross_class_pair_count"].detach(),
        "slm_pairwise_mse": slm_pairwise_mse.detach(),
        "slm_per_class_std_mean": slm_per_class_std_mean.detach(),
        "class_weight_mean": sample_weights.mean().detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
    }

