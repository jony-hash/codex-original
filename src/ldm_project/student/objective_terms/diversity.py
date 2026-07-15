"""diversity objective terms."""
# ruff: noqa: F403,F401,F405
from __future__ import annotations

from ._shared import *

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
        raise ValueError(
            "class_labels must match one_shot_loss batch size when diversity weights are enabled."
        )

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
        student_pairwise = (
            (student_class[:, None, :] - student_class[None, :, :]).pow(2).mean(dim=-1)
        )
        teacher_pairwise = (
            (teacher_class[:, None, :] - teacher_class[None, :, :]).pow(2).mean(dim=-1)
        )
        pair_mask = torch.triu(
            torch.ones(
                indices.numel(),
                indices.numel(),
                dtype=torch.bool,
                device=camera_prediction.device,
            ),
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
        raise ValueError(
            "class_labels must match one_shot_loss batch size when cross-class diversity is enabled."
        )

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_pairwise = (
        (prediction_flat[:, None, :] - prediction_flat[None, :, :]).pow(2).mean(dim=-1)
    )
    teacher_pairwise = (
        (target_flat[:, None, :] - target_flat[None, :, :]).pow(2).mean(dim=-1)
    )
    different_class = labels[:, None] != labels[None, :]
    pair_mask = (
        torch.triu(
            torch.ones(
                labels.numel(),
                labels.numel(),
                dtype=torch.bool,
                device=camera_prediction.device,
            ),
            diagonal=1,
        )
        & different_class
    )
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
        "cross_class_diversity_ratio": (active_student / active_teacher)
        .mean()
        .detach(),
        "cross_class_pair_count": active.float().sum().detach(),
    }

def _pairwise_mse_summary(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] < 2:
        return _zero_like_loss(tensor)
    flattened = tensor.detach().float().flatten(1)
    pairwise = (flattened[:, None, :] - flattened[None, :, :]).pow(2).mean(dim=-1)
    pair_mask = torch.triu(
        torch.ones(
            tensor.shape[0], tensor.shape[0], dtype=torch.bool, device=tensor.device
        ),
        diagonal=1,
    )
    return pairwise[pair_mask].mean()

def _per_class_std_summary(
    tensor: torch.Tensor, class_labels: torch.Tensor | None
) -> torch.Tensor:
    if class_labels is None:
        return _zero_like_loss(tensor)
    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != tensor.shape[0]:
        raise ValueError(
            "class_labels must match one_shot_loss batch size for SLM diagnostics."
        )
    flattened = tensor.detach().float().flatten(1)
    class_stds = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        class_stds.append(
            flattened.index_select(0, indices).std(dim=0, unbiased=False).mean()
        )
    if not class_stds:
        return _zero_like_loss(tensor)
    return torch.stack(class_stds).mean()

def _diversity_terms(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    slm_amplitude: torch.Tensor,
    *,
    code_weight: float,
    sample_code_weight: float,
    teacher_pairwise_weight: float,
    cross_class_weight: float,
    minimum_teacher_mse: float,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    pairwise = _one_shot_pairwise_diversity_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        minimum_teacher_mse,
    )
    cross_class = _one_shot_cross_class_pairwise_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        minimum_teacher_mse,
    )
    return {
        "pairwise": pairwise,
        "cross_class": cross_class,
        "code_loss": code_weight * pairwise["code_diversity_base"],
        "sample_code_loss": sample_code_weight * pairwise["code_diversity_base"],
        "teacher_pairwise_loss": teacher_pairwise_weight
        * pairwise["teacher_pairwise_diversity_base"],
        "cross_class_loss": cross_class_weight
        * cross_class["cross_class_pairwise_base"],
        "slm_pairwise_mse": _pairwise_mse_summary(slm_amplitude),
        "slm_per_class_std_mean": _per_class_std_summary(slm_amplitude, class_labels),
    }
