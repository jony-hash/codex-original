from __future__ import annotations

import math
from pathlib import Path

from .evaluation import compute_first_stage_balanced_metrics, evaluate_diffusion_with_per_class, evaluate_first_stage_with_per_class
from .sampling import (
    build_diffusion_visualization_tensors,
    build_first_stage_visualization_tensors,
    build_one_shot_visualization_tensors,
    compute_sample_quality_metrics,
    enhance_camera_for_display,
    format_output_title,
    save_post_train_samples,
    select_post_train_sample_batch,
)
from .train_utils import (
    estimate_latent_scale_from_loader,
    is_best_optical_candidate,
    load_latent_scale_from_diffusion_checkpoint,
    require_strict_optical_best_checkpoint,
    resolve_image_fusion_weight,
    resolve_logged_step,
    resolve_stage_total_steps,
    set_first_stage_image_fusion_weight,
    should_log_step,
    should_save_epoch_visual,
    should_update_best_optical,
    should_validate_epoch,
    validate_class_label_values,
)


FIRST_STAGE_BEST_OPTICAL_CHECKPOINT = "first_stage_best_optical.pt"
FIRST_STAGE_BEST_OPTICAL_METRIC = "camera_complex_balanced_mnist_score"
DIFFUSION_BEST_CHECKPOINT = "diffusion_best.pt"
DIFFUSION_BEST_METRIC = "diffusion_validation.loss"
ONE_SHOT_BEST_CHECKPOINT = "one_shot_best.pt"
ONE_SHOT_BEST_METRIC = "one_shot_validation.loss"
ONE_SHOT_DISTILL_BEST_CHECKPOINT = "one_shot_distill_best.pt"
ONE_SHOT_DISTILL_LATEST_CHECKPOINT = "one_shot_distill_latest.pt"
ONE_SHOT_DISTILL_BEST_METRIC = "one_shot_distill_validation.loss"
ONE_SHOT_DISTILL_QUALITY_METRIC = "one_shot_distill_validation.quality"
ONE_SHOT_DISTILL_QUALITY_SATURATION_TARGET = 0.325


def _sample_one_shot_codes(torch_module, batch_size: int, train_cfg, device):
    return torch_module.randn(batch_size, train_cfg.one_shot.code_dim, device=device)


def _sample_one_shot_codes_with_seed(torch_module, sample_count: int, code_dim: int, seed: int, device):
    generator = torch_module.Generator().manual_seed(int(seed))
    return torch_module.randn(sample_count, code_dim, generator=generator).to(device)


def _distill_class_label_sequence(torch_module, train_cfg, device):
    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.diffusion.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(range(train_cfg.diffusion.num_classes))
    labels = [label for label in class_labels for _ in range(train_cfg.distillation.samples_per_class)]
    if train_cfg.distillation.total_samples is not None:
        if not labels:
            raise ValueError("distillation.class_labels must not be empty when total_samples is set.")
        repeated = []
        while len(repeated) < train_cfg.distillation.total_samples:
            repeated.extend(class_labels)
        labels = repeated[: train_cfg.distillation.total_samples]
    if not labels:
        raise ValueError("Teacher distillation export resolved to zero labels.")
    return torch_module.tensor(labels, device=device, dtype=torch_module.long)


def _project_codes_to_teacher_noise(torch_module, codes, sample_shape: tuple[int, int, int, int], projection_seed: int, device):
    if codes.ndim != 2:
        raise ValueError(f"codes must be [batch, code_dim], got shape {tuple(codes.shape)}")
    flat_size = int(math.prod(sample_shape[1:]))
    generator = torch_module.Generator().manual_seed(int(projection_seed))
    projection = torch_module.randn(codes.shape[1], flat_size, generator=generator, dtype=codes.dtype)
    projected = codes.detach().cpu() @ (projection / math.sqrt(float(codes.shape[1])))
    return projected.to(device=device).reshape(sample_shape)


def _one_shot_loss_kwargs(train_cfg) -> dict[str, object]:
    return {
        "foreground_threshold": train_cfg.one_shot.recon_foreground_threshold,
        "foreground_weight": train_cfg.one_shot.recon_foreground_weight,
        "background_weight": train_cfg.one_shot.recon_background_weight,
        "class_recon_weight_labels": train_cfg.one_shot.class_recon_weight_labels,
        "class_recon_weight": train_cfg.one_shot.class_recon_weight,
        "shape_loss_weight": train_cfg.one_shot.shape_loss_weight,
        "edge_loss_weight": train_cfg.one_shot.edge_loss_weight,
        "slm_smoothness_weight": train_cfg.one_shot.slm_smoothness_weight,
        "camera_saturation_weight": train_cfg.one_shot.camera_saturation_weight,
        "camera_topology_weight": train_cfg.one_shot.camera_topology_weight,
        "camera_artifact_weight": train_cfg.one_shot.camera_artifact_weight,
        "code_diversity_weight": train_cfg.one_shot.code_diversity_weight,
        "teacher_pairwise_diversity_weight": train_cfg.one_shot.teacher_pairwise_diversity_weight,
        "diversity_min_teacher_mse": train_cfg.one_shot.diversity_min_teacher_mse,
    }


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


def _resolve_one_shot_distill_selection(train_cfg, validation_metrics: dict[str, float]) -> tuple[float, str]:
    selection_metric = train_cfg.one_shot.checkpoint_selection_metric
    if selection_metric == "loss":
        return validation_metrics["loss"], ONE_SHOT_DISTILL_BEST_METRIC
    if selection_metric == "quality":
        quality_score = _one_shot_distill_quality_score(validation_metrics)
        validation_metrics["quality_score"] = quality_score
        return quality_score, ONE_SHOT_DISTILL_QUALITY_METRIC
    raise ValueError(
        "one_shot.checkpoint_selection_metric must be 'loss' or 'quality', "
        f"got {selection_metric!r}"
    )


def _select_one_shot_visual_batch(torch_module, validation_bundle, train_bundle, train_cfg, device):
    sample_labels, reference_images, _, _ = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.one_shot.num_classes,
        device,
        sample_class_labels=train_cfg.one_shot.sample_class_labels,
        sample_repeats_per_class=1,
    )
    visual_codes = _sample_one_shot_codes(torch_module, int(sample_labels.shape[0]), train_cfg, device)
    return visual_codes, sample_labels, reference_images


def _evaluate_one_shot(
    torch_module,
    one_shot_model,
    one_shot_loss_fn,
    loader,
    train_cfg,
    device,
    use_pin_memory: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    one_shot_model.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            target_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            codes = _sample_one_shot_codes(torch_module, target_batch.shape[0], train_cfg, device)
            outputs = one_shot_model(codes, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
    if count <= 0:
        raise ValueError("Validation loader is empty.")
    return {key: value / float(count) for key, value in totals.items()}


def _load_distill_payload(torch_module, distill_dataset: str | Path) -> dict[str, object]:
    path = Path(distill_dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Distillation dataset does not exist: {path}")
    payload = torch_module.load(path, map_location="cpu", weights_only=False)
    required = ("codes", "class_labels", "teacher_camera")
    for key in required:
        if key not in payload:
            raise ValueError(f"Distillation dataset is missing required tensor: {key}")
    return payload


def _validated_distill_tensors(torch_module, payload: dict[str, object], train_cfg):
    codes = payload["codes"].float()
    labels = payload["class_labels"].long()
    targets = payload["teacher_camera"].float()
    if codes.ndim != 2:
        raise ValueError(f"distill codes must be [N, code_dim], got {tuple(codes.shape)}")
    if codes.shape[1] != train_cfg.one_shot.code_dim:
        raise ValueError(
            "distill code_dim does not match one_shot.code_dim: "
            f"{codes.shape[1]} != {train_cfg.one_shot.code_dim}"
        )
    if labels.ndim != 1 or labels.shape[0] != codes.shape[0] or targets.shape[0] != codes.shape[0]:
        raise ValueError("distill codes, class_labels, and teacher_camera must share the same batch dimension.")
    expected_target_shape = (1, train_cfg.first_stage.image_size, train_cfg.first_stage.image_size)
    if tuple(targets.shape[1:]) != expected_target_shape:
        raise ValueError(
            "distill teacher_camera shape does not match configured one-shot image size: "
            f"{tuple(targets.shape[1:])} != {expected_target_shape}"
        )
    validate_class_label_values(tuple(int(label) for label in labels.tolist()), train_cfg.one_shot.num_classes, "class_labels")
    return codes, labels, targets


def compute_teacher_distill_filter_metrics(
    torch_module,
    teacher_camera,
    *,
    foreground_threshold: float = 0.2,
) -> dict[str, object]:
    camera = teacher_camera.detach().float().clamp(0.0, 1.0).cpu()
    if camera.ndim != 4:
        raise ValueError(f"teacher_camera must be [N, C, H, W], got {tuple(camera.shape)}")
    flattened = camera.flatten(1)
    foreground_mask = camera > float(foreground_threshold)
    mask_flattened = foreground_mask.float().flatten(1)
    mask = foreground_mask.float()
    mask_tv_y = (mask[..., 1:, :] - mask[..., :-1, :]).abs().flatten(1).mean(dim=1)
    mask_tv_x = (mask[..., :, 1:] - mask[..., :, :-1]).abs().flatten(1).mean(dim=1)
    return {
        "foreground_ratio": mask_flattened.mean(dim=1),
        "saturation_ratio": ((camera >= 0.99) | (camera <= 0.01)).float().flatten(1).mean(dim=1),
        "mask_tv": mask_tv_x + mask_tv_y,
        "contrast_std": flattened.std(dim=1, unbiased=False),
    }


def _metric_summary(values) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def _class_relative_minmax(values):
    values = values.detach().float()
    spread = values.max() - values.min()
    if float(spread.item()) <= 1.0e-8:
        return values.new_zeros(values.shape)
    return (values - values.min()) / spread


def _select_teacher_distill_filter_indices(torch_module, labels, metric_tensors: dict[str, object], train_cfg):
    keep_per_class = int(train_cfg.distillation.filter_keep_per_class)
    if keep_per_class <= 0:
        raise ValueError(f"distillation.filter_keep_per_class must be positive, got {keep_per_class}")
    trim_fraction = float(train_cfg.distillation.filter_foreground_trim_fraction)
    contrast_drop_fraction = float(train_cfg.distillation.filter_contrast_drop_fraction)
    if trim_fraction < 0.0 or trim_fraction >= 0.5:
        raise ValueError("distillation.filter_foreground_trim_fraction must be in [0.0, 0.5).")
    if contrast_drop_fraction < 0.0 or contrast_drop_fraction >= 1.0:
        raise ValueError("distillation.filter_contrast_drop_fraction must be in [0.0, 1.0).")

    labels_cpu = labels.detach().long().cpu()
    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.one_shot.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(sorted({int(label) for label in labels_cpu.tolist()}))

    selected_indices: list[int] = []
    per_class: dict[str, dict[str, object]] = {}
    for label in class_labels:
        class_indices = torch_module.nonzero(labels_cpu == int(label), as_tuple=False).flatten()
        class_count = int(class_indices.numel())
        if class_count < keep_per_class:
            raise ValueError(
                f"Class {label} has only {class_count} teacher samples; "
                f"cannot keep {keep_per_class} filtered samples."
            )

        fg = metric_tensors["foreground_ratio"].index_select(0, class_indices)
        sat = metric_tensors["saturation_ratio"].index_select(0, class_indices)
        mask_tv = metric_tensors["mask_tv"].index_select(0, class_indices)
        contrast = metric_tensors["contrast_std"].index_select(0, class_indices)
        fg_low = torch_module.quantile(fg, trim_fraction)
        fg_high = torch_module.quantile(fg, 1.0 - trim_fraction)
        contrast_floor = torch_module.quantile(contrast, contrast_drop_fraction)
        candidate_mask = (fg >= fg_low) & (fg <= fg_high) & (contrast >= contrast_floor)
        candidate_indices = class_indices[candidate_mask]
        relaxed_filters = False
        if int(candidate_indices.numel()) < keep_per_class:
            candidate_indices = class_indices
            relaxed_filters = True

        candidate_sat = metric_tensors["saturation_ratio"].index_select(0, candidate_indices)
        candidate_tv = metric_tensors["mask_tv"].index_select(0, candidate_indices)
        candidate_contrast = metric_tensors["contrast_std"].index_select(0, candidate_indices)
        score = (
            _class_relative_minmax(candidate_sat)
            + _class_relative_minmax(candidate_tv)
            + (1.0 - _class_relative_minmax(candidate_contrast))
        )
        order = torch_module.argsort(score)
        chosen = candidate_indices.index_select(0, order[:keep_per_class])
        selected_indices.extend(int(index) for index in chosen.tolist())

        selected_local = torch_module.isin(class_indices, chosen)
        per_class[str(label)] = {
            "available": class_count,
            "kept": int(chosen.numel()),
            "candidate_count": int(candidate_indices.numel()),
            "relaxed_filters": relaxed_filters,
            "foreground_low_threshold": float(fg_low.item()),
            "foreground_high_threshold": float(fg_high.item()),
            "contrast_floor": float(contrast_floor.item()),
            "rejected_by_foreground_trim": int(((fg < fg_low) | (fg > fg_high)).sum().item()),
            "rejected_by_contrast": int((contrast < contrast_floor).sum().item()),
            "rejected_total": class_count - int(chosen.numel()),
            "before": {
                "foreground_ratio": _metric_summary(fg),
                "saturation_ratio": _metric_summary(sat),
                "mask_tv": _metric_summary(mask_tv),
                "contrast_std": _metric_summary(contrast),
            },
            "after": {
                "foreground_ratio": _metric_summary(fg[selected_local]),
                "saturation_ratio": _metric_summary(sat[selected_local]),
                "mask_tv": _metric_summary(mask_tv[selected_local]),
                "contrast_std": _metric_summary(contrast[selected_local]),
            },
        }

    return selected_indices, {
        "class_labels": list(class_labels),
        "keep_per_class": keep_per_class,
        "selected_count": len(selected_indices),
        "foreground_trim_fraction": trim_fraction,
        "contrast_drop_fraction": contrast_drop_fraction,
        "per_class": per_class,
    }


def _filter_distill_payload_by_indices(torch_module, payload: dict[str, object], indices, *, source_size: int):
    index_tensor = torch_module.tensor(indices, dtype=torch_module.long)
    filtered: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, torch_module.Tensor) and value.ndim >= 1 and int(value.shape[0]) == source_size:
            filtered[key] = value.index_select(0, index_tensor).clone()
        else:
            filtered[key] = value
    return filtered, index_tensor


def _balanced_preview_indices(torch_module, labels, *, max_per_class: int = 2, max_total: int = 20):
    labels_cpu = labels.detach().long().cpu()
    preview: list[int] = []
    for label in sorted({int(item) for item in labels_cpu.tolist()}):
        class_indices = torch_module.nonzero(labels_cpu == label, as_tuple=False).flatten().tolist()
        preview.extend(int(index) for index in class_indices[:max_per_class])
        if len(preview) >= max_total:
            break
    return torch_module.tensor(preview[:max_total], dtype=torch_module.long)


def _build_distill_loaders(torch_module, payload: dict[str, object], train_cfg, use_pin_memory: bool):
    codes, labels, targets = _validated_distill_tensors(torch_module, payload, train_cfg)
    dataset = torch_module.utils.data.TensorDataset(codes, labels, targets)
    total_size = len(dataset)
    validation_size = int(total_size * train_cfg.data.validation_fraction)
    if validation_size >= total_size:
        validation_size = total_size - 1
    train_size = total_size - validation_size
    if train_size <= 0:
        raise ValueError("Distillation dataset must leave at least one training sample.")
    generator = torch_module.Generator().manual_seed(train_cfg.data.validation_split_seed)
    if validation_size > 0:
        train_dataset, validation_dataset = torch_module.utils.data.random_split(
            dataset,
            (train_size, validation_size),
            generator=generator,
        )
    else:
        train_dataset = dataset
        validation_dataset = None
    train_loader = torch_module.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.data.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = torch_module.utils.data.DataLoader(
            validation_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
    return train_loader, validation_loader, total_size, train_size, validation_size


def _evaluate_one_shot_distill(
    torch_module,
    one_shot_model,
    one_shot_loss_fn,
    loader,
    train_cfg,
    device,
    use_pin_memory: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    one_shot_model.eval()
    with torch_module.no_grad():
        for codes, labels, targets in loader:
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            outputs = one_shot_model(code_batch, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
    if count <= 0:
        raise ValueError("Distillation validation loader is empty.")
    return {key: value / float(count) for key, value in totals.items()}


def _eval_correlation(torch_module, prediction, target):
    prediction_flat = prediction.detach().float().flatten(1)
    target_flat = target.detach().float().flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    denominator = prediction_centered.pow(2).mean(dim=1).sqrt() * target_centered.pow(2).mean(dim=1).sqrt()
    return ((prediction_centered * target_centered).mean(dim=1) / denominator.clamp_min(1.0e-8)).mean()


def _eval_foreground_dice(prediction, target, threshold: float = 0.2):
    prediction_mask = prediction.detach().float().clamp(0.0, 1.0) > threshold
    target_mask = target.detach().float().clamp(0.0, 1.0) > threshold
    dims = tuple(range(1, prediction_mask.ndim))
    intersection = (prediction_mask & target_mask).float().sum(dim=dims)
    denominator = prediction_mask.float().sum(dim=dims) + target_mask.float().sum(dim=dims)
    return (2.0 * intersection / denominator.clamp_min(1.0)).mean()


def _eval_camera_saturation_ratio(camera_prediction):
    camera = camera_prediction.detach().float()
    saturated = (camera >= 0.99) | (camera <= 0.01)
    return saturated.float().mean()


def _eval_total_variation(tensor):
    diff_y = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean()
    diff_x = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()
    return diff_x + diff_y


def _mean_pairwise_mse(torch_module, samples) -> float | None:
    sample_count = int(samples.shape[0])
    if sample_count < 2:
        return None
    flattened = samples.detach().float().flatten(1)
    pairwise = (flattened[:, None, :] - flattened[None, :, :]).pow(2).mean(dim=-1)
    upper = torch_module.triu(torch_module.ones(sample_count, sample_count, dtype=torch_module.bool), diagonal=1)
    return float(pairwise[upper].mean().item())


def _mean_pairwise_corr(torch_module, samples) -> float | None:
    sample_count = int(samples.shape[0])
    if sample_count < 2:
        return None
    flattened = samples.detach().float().flatten(1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    rms = centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    normalized = centered / rms
    pairwise = (normalized[:, None, :] * normalized[None, :, :]).mean(dim=-1)
    upper = torch_module.triu(torch_module.ones(sample_count, sample_count, dtype=torch_module.bool), diagonal=1)
    return float(pairwise[upper].mean().item())


def _select_distill_eval_indices(torch_module, labels, train_cfg) -> tuple[list[int], dict[str, object]]:
    label_values = labels.detach().cpu()
    requested = tuple(train_cfg.one_shot.sample_class_labels)
    if requested:
        class_labels = validate_class_label_values(requested, train_cfg.one_shot.num_classes, "one_shot.sample_class_labels")
    else:
        class_labels = tuple(sorted({int(label) for label in label_values.tolist()}))
    samples_per_class = max(5, int(train_cfg.one_shot.sample_repeats_per_class))
    total_size = int(label_values.shape[0])
    validation_size = int(total_size * train_cfg.data.validation_fraction)
    if validation_size >= total_size:
        validation_size = total_size - 1
    train_size = total_size - validation_size
    generator = torch_module.Generator().manual_seed(train_cfg.data.validation_split_seed)
    permutation = torch_module.randperm(total_size, generator=generator).tolist()
    validation_indices = set(permutation[train_size:]) if validation_size > 0 else set()

    selected: list[int] = []
    per_class: dict[str, dict[str, int]] = {}
    used_train_fallback = False
    for label in class_labels:
        all_for_label = [index for index, item in enumerate(label_values.tolist()) if int(item) == int(label)]
        if not all_for_label:
            raise ValueError(f"Distillation dataset has no samples for requested eval class label: {label}")
        validation_for_label = [index for index in all_for_label if index in validation_indices]
        chosen = validation_for_label[:samples_per_class]
        if len(chosen) < min(samples_per_class, len(all_for_label)):
            fallback = [index for index in all_for_label if index not in set(chosen)]
            chosen.extend(fallback[: samples_per_class - len(chosen)])
            used_train_fallback = True
        selected.extend(chosen)
        per_class[str(label)] = {
            "available": len(all_for_label),
            "selected": len(chosen),
            "validation_selected": min(len(validation_for_label), len(chosen)),
        }

    return selected, {
        "class_labels": list(class_labels),
        "requested_samples_per_class": samples_per_class,
        "selected_count": len(selected),
        "validation_size": validation_size,
        "used_train_fallback": used_train_fallback,
        "per_class": per_class,
    }


def compute_one_shot_distill_eval_metrics(
    torch_module,
    *,
    teacher_camera,
    student_camera,
    slm_amplitude,
    class_labels,
    foreground_threshold: float = 0.2,
) -> dict[str, object]:
    teacher = teacher_camera.detach().float().clamp(0.0, 1.0).cpu()
    student = student_camera.detach().float().clamp(0.0, 1.0).cpu()
    slm = slm_amplitude.detach().float().cpu()
    labels = [int(label) for label in class_labels.detach().cpu().tolist()]
    if teacher.shape != student.shape:
        raise ValueError(f"teacher_camera and student_camera must share shape, got {teacher.shape} != {student.shape}")
    if teacher.shape[0] != len(labels):
        raise ValueError("teacher_camera and class_labels must share batch size.")

    mse = float((student - teacher).pow(2).mean().item())
    corr = float(_eval_correlation(torch_module, student, teacher).item())
    dice = float(_eval_foreground_dice(student, teacher, threshold=foreground_threshold).item())
    saturation = float(_eval_camera_saturation_ratio(student).item())
    slm_tv = float(_eval_total_variation(slm).item())

    per_class: dict[str, dict[str, float | int | None]] = {}
    teacher_diversities: list[float] = []
    student_diversities: list[float] = []
    ratios: list[float] = []
    for label in sorted(set(labels)):
        indices = [index for index, item in enumerate(labels) if item == label]
        index_tensor = torch_module.tensor(indices, dtype=torch_module.long)
        teacher_class = teacher.index_select(0, index_tensor)
        student_class = student.index_select(0, index_tensor)
        slm_class = slm.index_select(0, index_tensor)
        teacher_pairwise_mse = _mean_pairwise_mse(torch_module, teacher_class)
        student_pairwise_mse = _mean_pairwise_mse(torch_module, student_class)
        teacher_pairwise_corr = _mean_pairwise_corr(torch_module, teacher_class)
        student_pairwise_corr = _mean_pairwise_corr(torch_module, student_class)
        diversity_ratio = None
        if teacher_pairwise_mse is not None and student_pairwise_mse is not None:
            diversity_ratio = student_pairwise_mse / max(teacher_pairwise_mse, 1.0e-8)
            teacher_diversities.append(teacher_pairwise_mse)
            student_diversities.append(student_pairwise_mse)
            ratios.append(diversity_ratio)
        per_class[str(label)] = {
            "count": len(indices),
            "paired_mse": float((student_class - teacher_class).pow(2).mean().item()),
            "paired_corr": float(_eval_correlation(torch_module, student_class, teacher_class).item()),
            "paired_dice": float(_eval_foreground_dice(student_class, teacher_class, threshold=foreground_threshold).item()),
            "student_saturation_ratio": float(_eval_camera_saturation_ratio(student_class).item()),
            "slm_tv": float(_eval_total_variation(slm_class).item()),
            "teacher_pairwise_mse": teacher_pairwise_mse,
            "student_pairwise_mse": student_pairwise_mse,
            "diversity_ratio": diversity_ratio,
            "teacher_pairwise_corr": teacher_pairwise_corr,
            "student_pairwise_corr": student_pairwise_corr,
        }

    return {
        "foreground_threshold": foreground_threshold,
        "overall": {
            "count": len(labels),
            "paired_mse": mse,
            "paired_corr": corr,
            "paired_dice": dice,
            "student_saturation_ratio": saturation,
            "slm_tv": slm_tv,
            "teacher_pairwise_mse": sum(teacher_diversities) / len(teacher_diversities) if teacher_diversities else None,
            "student_pairwise_mse": sum(student_diversities) / len(student_diversities) if student_diversities else None,
            "diversity_ratio": sum(ratios) / len(ratios) if ratios else None,
        },
        "per_class": per_class,
    }


def run_sample_stage(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    diffusion_checkpoint: str,
    first_stage_model_version: str,
    load_diffusion_checkpoint_fn,
    sample_latents_fn,
    sample_latents_with_trace_fn,
    resolve_denoising_trace_steps_fn,
) -> None:
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, train_cfg.first_stage.image_fusion_end_weight)
    diffusion_payload = torch_module.load(diffusion_checkpoint, map_location=device, weights_only=False)
    load_diffusion_checkpoint_fn(diffusion_model, diffusion_payload, first_stage_model_version)
    latent_scale = load_latent_scale_from_diffusion_checkpoint(torch_module, diffusion_payload, device)
    diffusion_model.eval()
    run_meta.update(
        {
            "latent_scale_source": "diffusion_checkpoint",
            "diffusion_best_loaded_for_sampling": True,
        }
    )
    run_manager.write_meta(run_meta)
    print(f"loaded diffusion checkpoint for sampling: {diffusion_checkpoint}")
    sample_artifacts = save_post_train_samples(
        torch_module,
        run_manager,
        first_stage,
        diffusion_model,
        scheduler,
        train_bundle,
        validation_bundle,
        train_cfg,
        latent_scale,
        device,
        sample_latents_fn,
        sample_latents_with_trace_fn,
        resolve_denoising_trace_steps_fn,
    )
    run_meta.update(sample_artifacts)
    run_manager.write_meta(run_meta)


def run_teacher_distill_export(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    diffusion_checkpoint: str,
    first_stage_model_version: str,
    load_diffusion_checkpoint_fn,
    sample_latents_fn,
) -> None:
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, train_cfg.first_stage.image_fusion_end_weight)
    diffusion_payload = torch_module.load(diffusion_checkpoint, map_location=device, weights_only=False)
    load_diffusion_checkpoint_fn(diffusion_model, diffusion_payload, first_stage_model_version)
    latent_scale = load_latent_scale_from_diffusion_checkpoint(torch_module, diffusion_payload, device)
    diffusion_model.eval()
    sample_labels = _distill_class_label_sequence(torch_module, train_cfg, device)
    sample_count = int(sample_labels.shape[0])
    codes = _sample_one_shot_codes_with_seed(
        torch_module,
        sample_count,
        train_cfg.one_shot.code_dim,
        train_cfg.distillation.code_seed,
        device,
    )
    sample_shape = (
        sample_count,
        train_cfg.first_stage.latent_channels,
        train_cfg.first_stage.image_size // 8,
        train_cfg.first_stage.image_size // 8,
    )
    teacher_latents = []
    teacher_initial_noise = []
    teacher_camera = []
    teacher_image = []
    batch_size = train_cfg.batch_size
    with torch_module.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_labels = sample_labels[start:end]
            batch_codes = codes[start:end]
            batch_shape = (end - start, *sample_shape[1:])
            initial_noise = _project_codes_to_teacher_noise(
                torch_module,
                batch_codes,
                batch_shape,
                train_cfg.distillation.code_projection_seed,
                device,
            )
            batch_latents = sample_latents_fn(
                diffusion_model,
                scheduler,
                batch_shape,
                device,
                batch_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
                initial_noise=initial_noise,
                sampling_steps=train_cfg.distillation.sampling_steps,
            )
            outputs = first_stage.decode_with_intermediates(batch_latents * latent_scale)
            teacher_initial_noise.append(initial_noise.detach().cpu())
            teacher_latents.append(batch_latents.detach().cpu())
            teacher_camera.append(outputs["camera_plane"].detach().cpu())
            teacher_image.append(outputs["image_recon"].detach().cpu())
            print(f"[teacher_distill_export] generated {end}/{sample_count}")

    payload = {
        "codes": codes.detach().cpu(),
        "class_labels": sample_labels.detach().cpu(),
        "teacher_camera": torch_module.cat(teacher_camera, dim=0),
        "teacher_image": torch_module.cat(teacher_image, dim=0),
        "teacher_latents": torch_module.cat(teacher_latents, dim=0),
        "teacher_initial_noise": torch_module.cat(teacher_initial_noise, dim=0),
        "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
        "latent_scale": latent_scale.detach().cpu(),
        "config": {
            "samples_per_class": train_cfg.distillation.samples_per_class,
            "total_samples": train_cfg.distillation.total_samples,
            "class_labels": list(train_cfg.distillation.class_labels),
            "code_dim": train_cfg.one_shot.code_dim,
            "code_seed": train_cfg.distillation.code_seed,
            "code_projection_seed": train_cfg.distillation.code_projection_seed,
            "sampling_steps": train_cfg.distillation.sampling_steps,
            "teacher_sample_shape": sample_shape,
        },
    }
    run_manager.save_config_snapshot(
        "teacher_distill_export_config",
        {
            **payload["config"],
            "num_samples": sample_count,
            "source": "train.py_teacher_distill_export",
        },
    )
    artifact_path: Path
    if train_cfg.distillation.output_path:
        artifact_path = Path(train_cfg.distillation.output_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(payload, artifact_path)
    else:
        artifact_path = run_manager.save_tensor_bundle("teacher_distill_dataset.pt", payload)
    run_manager.save_tensor_grid(
        {
            "teacher_camera_raw": payload["teacher_camera"],
            "teacher_camera_display": enhance_camera_for_display(payload["teacher_camera"]),
            "teacher_image": payload["teacher_image"],
        },
        run_manager.images_dir / "teacher_distill_export.png",
        title=format_output_title("images/teacher_distill_export.png", "teacher distillation targets"),
        max_items=min(sample_count, 20),
    )
    run_meta.update(
        {
            "latent_scale_source": "diffusion_checkpoint",
            "diffusion_best_loaded_for_sampling": True,
            "teacher_distill_dataset": str(artifact_path),
            "teacher_distill_sample_count": sample_count,
        }
    )
    run_manager.write_meta(run_meta)


def run_teacher_distill_filter(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
    codes, labels, teacher_camera = _validated_distill_tensors(torch_module, payload, train_cfg)
    metric_tensors = compute_teacher_distill_filter_metrics(
        torch_module,
        teacher_camera,
        foreground_threshold=train_cfg.distillation.filter_foreground_threshold,
    )
    selected_indices, selection_summary = _select_teacher_distill_filter_indices(
        torch_module,
        labels,
        metric_tensors,
        train_cfg,
    )
    filtered_payload, index_tensor = _filter_distill_payload_by_indices(
        torch_module,
        payload,
        selected_indices,
        source_size=int(codes.shape[0]),
    )
    filtered_metrics = {
        key: value.index_select(0, index_tensor)
        for key, value in metric_tensors.items()
    }
    source_config = payload.get("config", {})
    filtered_config = dict(source_config) if isinstance(source_config, dict) else {"source_config": source_config}
    filtered_config["filter"] = {
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "selected_count": len(selected_indices),
        "keep_per_class": train_cfg.distillation.filter_keep_per_class,
        "foreground_threshold": train_cfg.distillation.filter_foreground_threshold,
        "foreground_trim_fraction": train_cfg.distillation.filter_foreground_trim_fraction,
        "contrast_drop_fraction": train_cfg.distillation.filter_contrast_drop_fraction,
    }
    filtered_payload["config"] = filtered_config
    filtered_payload["filter_indices"] = index_tensor

    before_summary = {
        key: _metric_summary(value)
        for key, value in metric_tensors.items()
    }
    after_summary = {
        key: _metric_summary(value)
        for key, value in filtered_metrics.items()
    }
    class_counts = {
        str(label): int((filtered_payload["class_labels"].long() == int(label)).sum().item())
        for label in sorted({int(label) for label in filtered_payload["class_labels"].long().tolist()})
    }
    metrics_payload = {
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "selected_count": len(selected_indices),
        "class_counts": class_counts,
        "before": before_summary,
        "after": after_summary,
        **selection_summary,
    }

    run_manager.save_config_snapshot(
        "teacher_distill_filter_config",
        {
            "distill_dataset": distill_dataset,
            "filter": filtered_config["filter"],
            "source": "train.py_teacher_distill_filter",
        },
    )
    artifact_path = run_manager.save_tensor_bundle("teacher_distill_dataset_filtered.pt", filtered_payload)
    metrics_path = run_manager.save_artifact_json("teacher_distill_filter_metrics", metrics_payload)
    source_preview = _balanced_preview_indices(torch_module, labels)
    filtered_preview = _balanced_preview_indices(torch_module, filtered_payload["class_labels"])
    run_manager.save_tensor_grid(
        {
            "source_teacher_camera_display": enhance_camera_for_display(
                teacher_camera.index_select(0, source_preview)
            ),
            "filtered_teacher_camera_display": enhance_camera_for_display(
                filtered_payload["teacher_camera"].index_select(0, filtered_preview)
            ),
        },
        run_manager.images_dir / "teacher_distill_filter_before_after.png",
        title=format_output_title("images/teacher_distill_filter_before_after.png", "teacher distill filter"),
        max_items=min(20, len(selected_indices)),
    )
    print(
        "[teacher_distill_filter] "
        f"selected {len(selected_indices)}/{int(codes.shape[0])} samples "
        f"to {artifact_path}"
    )
    run_meta.update(
        {
            "teacher_distill_source_dataset": distill_dataset,
            "teacher_distill_filtered_dataset": str(artifact_path),
            "teacher_distill_filter_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
            "teacher_distill_filter_sample_count": len(selected_indices),
            "teacher_distill_filter_class_counts": class_counts,
        }
    )
    run_manager.write_meta(run_meta)


def run_one_shot_sample_stage(
    torch_module,
    *,
    one_shot_model,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_checkpoint: str,
    load_one_shot_checkpoint_fn,
) -> None:
    one_shot_payload = torch_module.load(one_shot_checkpoint, map_location=device, weights_only=False)
    load_one_shot_checkpoint_fn(one_shot_model, one_shot_payload)
    one_shot_model.eval()
    print(f"loaded one-shot checkpoint for sampling: {one_shot_checkpoint}")
    sample_labels, reference_images, reference_labels, reference_source = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.one_shot.num_classes,
        device,
        sample_class_labels=train_cfg.one_shot.sample_class_labels,
        sample_repeats_per_class=train_cfg.one_shot.sample_repeats_per_class,
    )
    sample_count = int(sample_labels.shape[0])
    codes = _sample_one_shot_codes(torch_module, sample_count, train_cfg, device)
    with torch_module.no_grad():
        sample_outputs = one_shot_model(codes, sample_labels)
    run_manager.save_config_snapshot(
        "one_shot_sample_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "sample_class_labels": list(train_cfg.one_shot.sample_class_labels),
            "sample_repeats_per_class": train_cfg.one_shot.sample_repeats_per_class,
            "source": "train.py_one_shot_sample",
        },
    )
    run_manager.save_tensor_grid(
        build_one_shot_visualization_tensors(reference_images, sample_outputs),
        run_manager.images_dir / "one_shot_sample.png",
        title=format_output_title("images/one_shot_sample.png", "one-shot sample outputs"),
        max_items=sample_count,
    )
    run_manager.save_tensor_bundle(
        "one_shot_sample_tensors.pt",
        {
            "codes": codes.detach().cpu(),
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "generated_camera": sample_outputs["camera_plane"].detach().cpu(),
            "generated_camera_raw": sample_outputs["camera_plane"].detach().cpu(),
            "generated_camera_display": enhance_camera_for_display(sample_outputs["camera_plane"]).detach().cpu(),
            "slm_amplitude": sample_outputs["slm_amplitude"].detach().cpu(),
            "chip_amplitude": sample_outputs["chip_amplitude"].detach().cpu(),
            "chip_phase": sample_outputs["chip_phase"].detach().cpu(),
        },
    )
    sample_metrics = compute_sample_quality_metrics(sample_outputs["camera_plane"], reference_images, sample_labels)
    metrics_path = run_manager.save_artifact_json("one_shot_sample_metrics", sample_metrics)
    run_meta.update(
        {
            "one_shot_checkpoint": one_shot_checkpoint,
            "one_shot_sample_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
        }
    )
    run_manager.write_meta(run_meta)


def run_one_shot_distill_eval_stage(
    torch_module,
    *,
    one_shot_model,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_checkpoint: str,
    distill_dataset: str,
    load_one_shot_checkpoint_fn,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
    codes, labels, teacher_camera = _validated_distill_tensors(torch_module, payload, train_cfg)
    selected_indices, selection_summary = _select_distill_eval_indices(torch_module, labels, train_cfg)
    if not selected_indices:
        raise ValueError("one_shot_distill_eval selected zero samples.")

    index_tensor = torch_module.tensor(selected_indices, dtype=torch_module.long)
    eval_codes = codes.index_select(0, index_tensor)
    eval_labels = labels.index_select(0, index_tensor)
    eval_teacher = teacher_camera.index_select(0, index_tensor)

    one_shot_payload = torch_module.load(one_shot_checkpoint, map_location=device, weights_only=False)
    load_one_shot_checkpoint_fn(one_shot_model, one_shot_payload)
    one_shot_model.eval()
    print(f"loaded one-shot checkpoint for distill eval: {one_shot_checkpoint}")
    with torch_module.no_grad():
        eval_outputs = one_shot_model(eval_codes.to(device), eval_labels.to(device))
    student_camera = eval_outputs["camera_plane"].detach().cpu()
    slm_amplitude = eval_outputs["slm_amplitude"].detach().cpu()
    abs_error = (student_camera - eval_teacher).abs()
    metrics = compute_one_shot_distill_eval_metrics(
        torch_module,
        teacher_camera=eval_teacher,
        student_camera=student_camera,
        slm_amplitude=slm_amplitude,
        class_labels=eval_labels,
    )
    metrics["selection"] = selection_summary

    run_manager.save_config_snapshot(
        "one_shot_distill_eval_config",
        {
            "distill_dataset": distill_dataset,
            "one_shot_checkpoint": one_shot_checkpoint,
            "selection": selection_summary,
            "source": "train.py_one_shot_distill_eval",
        },
    )
    run_manager.save_tensor_grid(
        {
            "teacher_camera_display": enhance_camera_for_display(eval_teacher),
            "student_camera_raw": student_camera,
            "student_camera_display": enhance_camera_for_display(student_camera),
            "abs_error": abs_error,
            "slm_amplitude": slm_amplitude,
        },
        run_manager.images_dir / "one_shot_distill_eval.png",
        title=format_output_title("images/one_shot_distill_eval.png", "one-shot distill paired eval"),
        max_items=int(eval_labels.shape[0]),
    )
    tensor_path = run_manager.save_tensor_bundle(
        "one_shot_distill_eval_tensors.pt",
        {
            "indices": index_tensor,
            "codes": eval_codes,
            "class_labels": eval_labels,
            "teacher_camera": eval_teacher,
            "teacher_camera_display": enhance_camera_for_display(eval_teacher),
            "student_camera_raw": student_camera,
            "student_camera_display": enhance_camera_for_display(student_camera),
            "abs_error": abs_error,
            "slm_amplitude": slm_amplitude,
        },
    )
    metrics_path = run_manager.save_artifact_json("one_shot_distill_eval_metrics", metrics)
    overall = metrics["overall"]
    print(
        "[one_shot_distill_eval] "
        f"mse={overall['paired_mse']:.6f} corr={overall['paired_corr']:.4f} "
        f"dice={overall['paired_dice']:.4f} sat={overall['student_saturation_ratio']:.4f} "
        f"slm_tv={overall['slm_tv']:.6f} diversity_ratio={overall['diversity_ratio']}"
    )
    run_meta.update(
        {
            "one_shot_checkpoint": one_shot_checkpoint,
            "distill_dataset": distill_dataset,
            "one_shot_distill_eval_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
            "one_shot_distill_eval_tensor_source": str(tensor_path.relative_to(run_manager.run_dir)),
        }
    )
    run_manager.write_meta(run_meta)


def run_one_shot_training(
    torch_module,
    *,
    one_shot_model,
    one_shot_loss_fn,
    run_manager,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    one_shot_model_version: str,
) -> None:
    optimizer = torch_module.optim.Adam(one_shot_model.parameters(), lr=train_cfg.learning_rate)
    loader_length = len(train_bundle.loader)
    total_steps = resolve_stage_total_steps(loader_length, train_cfg.one_shot_epochs)
    visual_codes, visual_labels, visual_references = _select_one_shot_visual_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg,
        device,
    )
    one_shot_stage_step = 0
    best_validation_loss = None
    best_checkpoint_path = None
    one_shot_model.train()
    for epoch_index in range(train_cfg.one_shot_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (images, labels) in enumerate(train_bundle.loader):
            target_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            codes = _sample_one_shot_codes(torch_module, target_batch.shape[0], train_cfg, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = one_shot_model(codes, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            losses["loss"].backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, one_shot_stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                print(
                    f"[one_shot] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"slm_tv={losses['slm_tv'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"div={losses['code_diversity_penalty'].item():.4f}"
                )
                run_manager.append_history(
                    "one_shot",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        "loss": losses["loss"].item(),
                        "camera_recon_loss": losses["camera_recon_loss"].item(),
                        "camera_fg_mse": losses["camera_fg_mse"].item(),
                        "camera_bg_mse": losses["camera_bg_mse"].item(),
                        "camera_foreground_dice": losses["camera_foreground_dice"].item(),
                        "slm_smoothness_loss": losses["slm_smoothness_loss"].item(),
                        "slm_tv": losses["slm_tv"].item(),
                        "camera_saturation_loss": losses["camera_saturation_loss"].item(),
                        "camera_saturation_ratio": losses["camera_saturation_ratio"].item(),
                        "code_diversity_loss": losses["code_diversity_loss"].item(),
                        "code_diversity_penalty": losses["code_diversity_penalty"].item(),
                        "teacher_pairwise_diversity_loss": losses["teacher_pairwise_diversity_loss"].item(),
                        "teacher_pairwise_diversity_penalty": losses["teacher_pairwise_diversity_penalty"].item(),
                        "teacher_pairwise_mse": losses["teacher_pairwise_mse"].item(),
                        "student_pairwise_mse": losses["student_pairwise_mse"].item(),
                        "diversity_pair_count": losses["diversity_pair_count"].item(),
                        "fg_pixel_ratio": losses["fg_pixel_ratio"].item(),
                    },
                )
            one_shot_stage_step += 1

        if should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval):
            one_shot_model.eval()
            with torch_module.no_grad():
                visual_outputs = one_shot_model(visual_codes, visual_labels)
                run_manager.save_tensor_grid(
                    build_one_shot_visualization_tensors(visual_references, visual_outputs),
                    run_manager.images_dir / "one_shot" / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/one_shot/epoch_{epoch_number:04d}.png",
                        f"one_shot epoch {epoch_number}",
                    ),
                    max_items=visual_labels.shape[0],
                )
            one_shot_model.train()

        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.one_shot_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics = _evaluate_one_shot(
                torch_module,
                one_shot_model,
                one_shot_loss_fn,
                validation_bundle.loader,
                train_cfg,
                device,
                use_pin_memory,
            )
            selection_loss = validation_metrics["loss"]
            updated_best = best_validation_loss is None or selection_loss < best_validation_loss
            print(
                f"[one_shot][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"slm_tv={validation_metrics['slm_tv']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"div={validation_metrics['code_diversity_penalty']:.4f}"
            )
            run_manager.append_history(
                "one_shot_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            if updated_best:
                best_validation_loss = selection_loss
                best_checkpoint_path = run_manager.save_checkpoint(
                    ONE_SHOT_BEST_CHECKPOINT,
                    {
                        "model_state_dict": one_shot_model.state_dict(),
                        "stage": "one_shot",
                        "model_version": one_shot_model_version,
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": ONE_SHOT_BEST_METRIC,
                        "selection_loss": selection_loss,
                        "validation_metrics": validation_metrics,
                    },
                )
                run_meta.update(
                    {
                        "one_shot_best_checkpoint": best_checkpoint_path,
                        "one_shot_best_epoch": epoch_number,
                        "one_shot_best_loss": best_validation_loss,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "one_shot_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best,
                    "selection_metric": ONE_SHOT_BEST_METRIC,
                    "selection_loss": selection_loss,
                    "best_loss": best_validation_loss,
                    "checkpoint": best_checkpoint_path,
                },
            )
            one_shot_model.train()

    run_manager.save_checkpoint(
        "one_shot_latest.pt",
        {
            "model_state_dict": one_shot_model.state_dict(),
            "stage": "one_shot",
            "model_version": one_shot_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
            "total_steps": total_steps,
        },
    )


def run_one_shot_distill_training(
    torch_module,
    *,
    one_shot_model,
    one_shot_loss_fn,
    run_manager,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    one_shot_model_version: str,
    distill_dataset: str,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
    train_loader, validation_loader, total_size, train_size, validation_size = _build_distill_loaders(
        torch_module,
        payload,
        train_cfg,
        use_pin_memory,
    )
    optimizer = torch_module.optim.Adam(one_shot_model.parameters(), lr=train_cfg.learning_rate)
    loader_length = len(train_loader)
    total_steps = resolve_stage_total_steps(loader_length, train_cfg.one_shot_epochs)
    visual_source = validation_loader if validation_loader is not None else train_loader
    visual_codes, visual_labels, visual_references = next(iter(visual_source))
    visual_codes = visual_codes.to(device, non_blocking=use_pin_memory)
    visual_labels = visual_labels.to(device, non_blocking=use_pin_memory)
    visual_references = visual_references.to(device, non_blocking=use_pin_memory)
    stage_step = 0
    best_selection_score = None
    best_checkpoint_path = None
    run_manager.save_config_snapshot(
        "one_shot_distill_config",
        {
            "distill_dataset": distill_dataset,
            "dataset_total_size": total_size,
            "train_size": train_size,
            "validation_size": validation_size,
            "source": "train.py_one_shot_distill",
        },
    )
    run_meta.update(
        {
            "distill_dataset": distill_dataset,
            "distill_dataset_total_size": total_size,
            "distill_train_size": train_size,
            "distill_validation_size": validation_size,
        }
    )
    run_manager.write_meta(run_meta)

    one_shot_model.train()
    for epoch_index in range(train_cfg.one_shot_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (codes, labels, targets) in enumerate(train_loader):
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            outputs = one_shot_model(code_batch, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            losses["loss"].backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                print(
                    f"[one_shot_distill] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"shape={losses['shape_penalty'].item():.4f} "
                    f"edge={losses['edge_penalty'].item():.4f} "
                    f"slm_tv={losses['slm_tv'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f}"
                )
                run_manager.append_history(
                    "one_shot_distill",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        "loss": losses["loss"].item(),
                        "camera_recon_loss": losses["camera_recon_loss"].item(),
                        "camera_fg_mse": losses["camera_fg_mse"].item(),
                        "camera_bg_mse": losses["camera_bg_mse"].item(),
                        "camera_foreground_dice": losses["camera_foreground_dice"].item(),
                        "shape_loss": losses["shape_loss"].item(),
                        "shape_penalty": losses["shape_penalty"].item(),
                        "edge_loss": losses["edge_loss"].item(),
                        "edge_penalty": losses["edge_penalty"].item(),
                        "class_weight_mean": losses["class_weight_mean"].item(),
                        "slm_smoothness_loss": losses["slm_smoothness_loss"].item(),
                        "slm_tv": losses["slm_tv"].item(),
                        "camera_saturation_loss": losses["camera_saturation_loss"].item(),
                        "camera_saturation_ratio": losses["camera_saturation_ratio"].item(),
                        "fg_pixel_ratio": losses["fg_pixel_ratio"].item(),
                    },
                )
            stage_step += 1

        if should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval):
            one_shot_model.eval()
            with torch_module.no_grad():
                visual_outputs = one_shot_model(visual_codes, visual_labels)
                run_manager.save_tensor_grid(
                    build_one_shot_visualization_tensors(visual_references, visual_outputs),
                    run_manager.images_dir / "one_shot_distill" / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/one_shot_distill/epoch_{epoch_number:04d}.png",
                        f"one_shot_distill epoch {epoch_number}",
                    ),
                    max_items=visual_labels.shape[0],
                )
            one_shot_model.train()

        if validation_loader is not None and should_validate_epoch(
            epoch_index,
            train_cfg.one_shot_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics = _evaluate_one_shot_distill(
                torch_module,
                one_shot_model,
                one_shot_loss_fn,
                validation_loader,
                train_cfg,
                device,
                use_pin_memory,
            )
            selection_score, selection_metric = _resolve_one_shot_distill_selection(train_cfg, validation_metrics)
            updated_best = best_selection_score is None or selection_score < best_selection_score
            print(
                f"[one_shot_distill][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"shape={validation_metrics['shape_penalty']:.4f} "
                f"edge={validation_metrics['edge_penalty']:.4f} "
                f"slm_tv={validation_metrics['slm_tv']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"selection={selection_score:.6f} ({selection_metric})"
            )
            run_manager.append_history(
                "one_shot_distill_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            if updated_best:
                best_selection_score = selection_score
                best_checkpoint_path = run_manager.save_checkpoint(
                    ONE_SHOT_DISTILL_BEST_CHECKPOINT,
                    {
                        "model_state_dict": one_shot_model.state_dict(),
                        "stage": "one_shot_distill",
                        "model_version": one_shot_model_version,
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": selection_metric,
                        "selection_loss": selection_score,
                        "selection_score": selection_score,
                        "validation_metrics": validation_metrics,
                        "distill_dataset": distill_dataset,
                    },
                )
                run_meta.update(
                    {
                        "one_shot_best_checkpoint": best_checkpoint_path,
                        "one_shot_best_epoch": epoch_number,
                        "one_shot_best_loss": best_selection_score,
                        "one_shot_best_score": best_selection_score,
                        "one_shot_distill_best_checkpoint": best_checkpoint_path,
                        "one_shot_distill_best_epoch": epoch_number,
                        "one_shot_distill_best_loss": best_selection_score,
                        "one_shot_distill_best_score": best_selection_score,
                        "one_shot_distill_best_metric": selection_metric,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "one_shot_distill_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best,
                    "selection_metric": selection_metric,
                    "selection_loss": selection_score,
                    "selection_score": selection_score,
                    "best_loss": best_selection_score,
                    "best_score": best_selection_score,
                    "checkpoint": best_checkpoint_path,
                },
            )
            one_shot_model.train()

    run_manager.save_checkpoint(
        ONE_SHOT_DISTILL_LATEST_CHECKPOINT,
        {
            "model_state_dict": one_shot_model.state_dict(),
            "stage": "one_shot_distill",
            "model_version": one_shot_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
            "total_steps": total_steps,
            "distill_dataset": distill_dataset,
        },
    )


def run_first_stage_training(
    torch_module,
    *,
    first_stage,
    first_stage_loss_fn,
    first_stage_visual_batch,
    run_manager,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    first_stage_model_version: str,
) -> None:
    optimizer = torch_module.optim.Adam(first_stage.parameters(), lr=train_cfg.learning_rate)
    first_stage.train()
    loader_length = len(train_bundle.loader)
    first_stage_total_steps = resolve_stage_total_steps(
        loader_length,
        train_cfg.first_stage_epochs,
    )
    first_stage_stage_step = 0
    best_optical_score = None
    best_optical_checkpoint_path = None
    for epoch_index in range(train_cfg.first_stage_epochs):
        epoch_number = epoch_index + 1
        image_fusion_weight = resolve_image_fusion_weight(train_cfg.first_stage, epoch_index)
        set_first_stage_image_fusion_weight(first_stage, image_fusion_weight)
        for batch_index, (images, _) in enumerate(train_bundle.loader):
            # 物理优先训练把 camera_plane 作为主读出，image_recon 仅保留为辅助诊断。
            input_batch = images.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            outputs = first_stage(input_batch)
            losses = first_stage_loss_fn(
                outputs,
                input_batch,
                global_step=first_stage_stage_step,
                total_steps=first_stage_total_steps,
                target_beta=train_cfg.first_stage.kl_beta,
                warmup_ratio=train_cfg.first_stage.kl_warmup_ratio,
                foreground_threshold=train_cfg.first_stage.recon_foreground_threshold,
                foreground_weight=train_cfg.first_stage.recon_foreground_weight,
                background_weight=train_cfg.first_stage.recon_background_weight,
                image_recon_weight=train_cfg.first_stage.image_recon_weight,
                camera_recon_weight=train_cfg.first_stage.camera_recon_weight,
                structure_recon_weight=train_cfg.first_stage.structure_recon_weight,
                shape_recon_weight=train_cfg.first_stage.shape_recon_weight,
                field_smoothness_weight=train_cfg.first_stage.field_smoothness_weight,
                phase_smoothness_weight=train_cfg.first_stage.phase_smoothness_weight,
                background_leakage_weight=train_cfg.first_stage.background_leakage_weight,
                field_energy_weight=train_cfg.first_stage.field_energy_weight,
                camera_hotspot_weight=train_cfg.first_stage.camera_hotspot_weight,
                camera_topology_weight=train_cfg.first_stage.camera_topology_weight,
                camera_saturation_weight=train_cfg.first_stage.camera_saturation_weight,
                camera_artifact_weight=train_cfg.first_stage.camera_artifact_weight,
            )
            losses["loss"].backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, first_stage_stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                print(
                    f"[first_stage] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} recon_w={losses['recon_loss'].item():.6f} "
                    f"image_recon={losses['image_recon_loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"image_fg_mse={losses['image_fg_mse'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"hotspot={losses['camera_hotspot_ratio'].item():.4f} "
                    f"topology={losses['camera_topology_loss'].item():.6f} "
                    f"structure={losses['structure_loss'].item():.6f} "
                    f"shape={losses['shape_loss'].item():.6f} "
                    f"field={losses['field_smoothness_loss'].item():.6f} "
                    f"phase_tv={losses['phase_tv'].item():.6f} "
                    f"field_energy={losses['field_energy'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"artifact={losses['camera_artifact_loss'].item():.6f} "
                    f"hole={losses['camera_foreground_hole_ratio'].item():.4f} "
                    f"image_fusion={image_fusion_weight:.4f} "
                    f"kl={losses['kl_loss'].item():.6f}"
                )
                run_manager.append_history(
                    "first_stage",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        "loss": losses["loss"].item(),
                        "recon_loss": losses["recon_loss"].item(),
                        "image_recon_loss": losses["image_recon_loss"].item(),
                        "image_fg_mse": losses["image_fg_mse"].item(),
                        "image_bg_mse": losses["image_bg_mse"].item(),
                        "camera_recon_loss": losses["camera_recon_loss"].item(),
                        "camera_fg_mse": losses["camera_fg_mse"].item(),
                        "camera_bg_mse": losses["camera_bg_mse"].item(),
                        "camera_foreground_dice_loss": losses["camera_foreground_dice_loss"].item(),
                        "camera_foreground_dice": losses["camera_foreground_dice"].item(),
                        "structure_loss": losses["structure_loss"].item(),
                        "image_structure_loss": losses["image_structure_loss"].item(),
                        "camera_structure_loss": losses["camera_structure_loss"].item(),
                        "shape_loss": losses["shape_loss"].item(),
                        "image_shape_loss": losses["image_shape_loss"].item(),
                        "camera_shape_loss": losses["camera_shape_loss"].item(),
                        "field_smoothness_loss": losses["field_smoothness_loss"].item(),
                        "field_tv": losses["field_tv"].item(),
                        "phase_smoothness_loss": losses["phase_smoothness_loss"].item(),
                        "phase_tv": losses["phase_tv"].item(),
                        "field_energy_loss": losses["field_energy_loss"].item(),
                        "field_energy": losses["field_energy"].item(),
                        "background_leakage_loss": losses["background_leakage_loss"].item(),
                        "camera_background_leakage": losses["camera_background_leakage"].item(),
                        "camera_hotspot_loss": losses["camera_hotspot_loss"].item(),
                        "camera_hotspot_ratio": losses["camera_hotspot_ratio"].item(),
                        "camera_saturation_ratio": losses["camera_saturation_ratio"].item(),
                        "camera_topology_loss": losses["camera_topology_loss"].item(),
                        "camera_topology_base": losses["camera_topology_base"].item(),
                        "camera_saturation_loss": losses["camera_saturation_loss"].item(),
                        "camera_saturation_penalty": losses["camera_saturation_penalty"].item(),
                        "camera_artifact_loss": losses["camera_artifact_loss"].item(),
                        "camera_foreground_hole_loss": losses["camera_foreground_hole_loss"].item(),
                        "camera_foreground_hole_ratio": losses["camera_foreground_hole_ratio"].item(),
                        "camera_foreground_roughness": losses["camera_foreground_roughness"].item(),
                        "kl_loss": losses["kl_loss"].item(),
                        "fg_pixel_ratio": losses["fg_pixel_ratio"].item(),
                        "pred_fg_energy_ratio": losses["pred_fg_energy_ratio"].item(),
                        "beta": losses["beta"].item(),
                        "image_fusion_weight": image_fusion_weight,
                    },
                )
            first_stage_stage_step += 1

        # 仅在 epoch 结束时保存固定验证样本，方便横向观察复杂数字结构是否改善。
        if first_stage_visual_batch is not None and should_save_epoch_visual(
            epoch_index,
            train_cfg.visual_save_epoch_interval,
        ):
            first_stage.eval()
            with torch_module.no_grad():
                visual_outputs = first_stage(first_stage_visual_batch)
                run_manager.save_tensor_grid(
                    build_first_stage_visualization_tensors(first_stage_visual_batch, visual_outputs),
                    run_manager.first_stage_images_dir / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/first_stage/epoch_{epoch_number:04d}.png",
                        f"first_stage epoch {epoch_number}",
                    ),
                    max_items=first_stage_visual_batch.shape[0],
                )
            first_stage.train()
        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.first_stage_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics, per_class_metrics = evaluate_first_stage_with_per_class(
                torch_module,
                first_stage,
                first_stage_loss_fn,
                validation_bundle.loader,
                device,
                use_pin_memory,
                first_stage_stage_step,
                first_stage_total_steps,
                train_cfg,
            )
            validation_metrics.update(compute_first_stage_balanced_metrics(validation_metrics, per_class_metrics))
            print(
                f"[first_stage][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"recon_w={validation_metrics['recon_loss']:.6f} "
                f"image_recon={validation_metrics['image_recon_loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"image_fg_mse={validation_metrics['image_fg_mse']:.6f} "
                f"camera_fg_mse={validation_metrics['camera_fg_mse']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"hotspot={validation_metrics['camera_hotspot_ratio']:.4f} "
                f"topology={validation_metrics['camera_topology_loss']:.6f} "
                f"structure={validation_metrics['structure_loss']:.6f} "
                f"shape={validation_metrics['shape_loss']:.6f} "
                f"field={validation_metrics['field_smoothness_loss']:.6f} "
                f"phase_tv={validation_metrics['phase_tv']:.6f} "
                f"field_energy={validation_metrics['field_energy']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"artifact={validation_metrics['camera_artifact_loss']:.6f} "
                f"hole={validation_metrics['camera_foreground_hole_ratio']:.4f} "
                f"camera_corr={validation_metrics['camera_corr']:.4f} "
                f"camera_edge_corr={validation_metrics['camera_edge_corr']:.4f} "
                f"min_class_corr={validation_metrics['min_class_camera_corr']:.4f} "
                f"balanced={validation_metrics['camera_balanced_score']:.4f} "
                f"mnist_score={validation_metrics['camera_mnist_score']:.4f} "
                f"min_class_mnist={validation_metrics['min_class_camera_mnist_score']:.4f} "
                f"class_balanced_mnist={validation_metrics['camera_class_balanced_mnist_score']:.4f} "
                f"complex_corr={validation_metrics['complex_camera_corr']:.4f} "
                f"complex_mnist={validation_metrics['complex_camera_mnist_score']:.4f} "
                f"complex_balanced_mnist={validation_metrics['camera_complex_balanced_mnist_score']:.4f} "
                f"kl={validation_metrics['kl_loss']:.6f}"
            )
            run_manager.append_history(
                "first_stage_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            run_manager.append_history(
                "first_stage_validation_per_class",
                {
                    "epoch": epoch_number,
                    "classes": per_class_metrics,
                },
            )
            optical_candidate = is_best_optical_candidate(train_cfg.first_stage, image_fusion_weight)
            selection_score = validation_metrics[FIRST_STAGE_BEST_OPTICAL_METRIC]
            updated_best = optical_candidate and should_update_best_optical(selection_score, best_optical_score)
            if updated_best:
                best_optical_score = selection_score
                best_optical_checkpoint_path = run_manager.save_checkpoint(
                    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
                    {
                        "model_state_dict": first_stage.state_dict(),
                        "stage": "first_stage",
                        "model_version": first_stage_model_version,
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
                        "selection_score": selection_score,
                        "validation_metrics": validation_metrics,
                        "image_fusion_weight": image_fusion_weight,
                    },
                )
                run_meta.update(
                    {
                        "first_stage_best_optical_checkpoint": best_optical_checkpoint_path,
                        "first_stage_best_optical_epoch": epoch_number,
                        "first_stage_best_optical_score": best_optical_score,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "first_stage_best_optical",
                {
                    "epoch": epoch_number,
                    "eligible": optical_candidate,
                    "updated": updated_best,
                    "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
                    "selection_score": selection_score,
                    "best_score": best_optical_score,
                    "image_fusion_weight": image_fusion_weight,
                    "checkpoint": best_optical_checkpoint_path,
                },
            )
            first_stage.train()

    run_manager.save_checkpoint(
        "first_stage_latest.pt",
        {
            "model_state_dict": first_stage.state_dict(),
            "stage": "first_stage",
            "model_version": first_stage_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
        },
    )


def run_diffusion_training(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    run_manager,
    physics,
    device,
    use_pin_memory: bool,
    first_stage_model_version: str,
    diffusion_model_version: str,
    diffusion_visual_labels,
    diffusion_visual_reference_images,
    load_first_stage_checkpoint_fn,
    load_diffusion_checkpoint_fn,
    diffusion_loss_fn,
    diffusion_per_sample_losses_fn,
    sample_latents_fn,
    sample_latents_with_trace_fn,
    resolve_denoising_trace_steps_fn,
) -> None:
    if train_cfg.mode == "teacher_full":
        best_checkpoint_path = require_strict_optical_best_checkpoint(
            run_meta.get("first_stage_best_optical_checkpoint")
        )
        # full 训练的第二阶段必须使用验证集上最好的纯 optical first stage。
        best_first_stage_payload = torch_module.load(best_checkpoint_path, map_location=device, weights_only=False)
        load_first_stage_checkpoint_fn(first_stage, best_first_stage_payload)
        run_meta["first_stage_best_optical_loaded_for_diffusion"] = True
        run_manager.write_meta(run_meta)
        print(f"loaded best optical first stage for diffusion: {best_checkpoint_path}")
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, train_cfg.first_stage.image_fusion_end_weight)
    if train_cfg.first_stage.freeze_after_pretrain:
        for param in first_stage.parameters():
            param.requires_grad_(False)

    latent_scale = estimate_latent_scale_from_loader(
        torch_module,
        first_stage,
        train_bundle.loader,
        device,
        use_pin_memory,
        train_cfg.diffusion.latent_source,
    )
    optimizer = torch_module.optim.Adam(diffusion_model.parameters(), lr=train_cfg.learning_rate)
    diffusion_model.train()
    loader_length = len(train_bundle.loader)
    resolve_stage_total_steps(
        loader_length,
        train_cfg.diffusion_epochs,
    )
    diffusion_stage_step = 0
    best_diffusion_loss = None
    best_diffusion_checkpoint_path = None
    for epoch_index in range(train_cfg.diffusion_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (images, labels) in enumerate(train_bundle.loader):
            # 第二阶段不再重建输入图，而是学习“给定类别条件时如何在 latent 空间去噪”。
            batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss_fn(
                first_stage,
                diffusion_model,
                scheduler,
                batch,
                class_labels,
                latent_scale,
                latent_source=train_cfg.diffusion.latent_source,
                class_weight_labels=train_cfg.diffusion.class_weight_labels,
                class_weight=train_cfg.diffusion.class_weight,
            )
            loss.backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, diffusion_stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                print(
                    f"[diffusion] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={loss.item():.6f} latent_scale={latent_scale.item():.6f}"
                )
                run_manager.append_history(
                    "diffusion",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        "loss": loss.item(),
                        "latent_scale": latent_scale.item(),
                    },
                )
            diffusion_stage_step += 1

        if (
            diffusion_visual_labels is not None
            and diffusion_visual_reference_images is not None
            and should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval)
        ):
            with torch_module.no_grad():
                sample_count = int(diffusion_visual_labels.shape[0])
                sample_batch = sample_latents_fn(
                    diffusion_model,
                    scheduler,
                    (
                        sample_count,
                        train_cfg.first_stage.latent_channels,
                        train_cfg.first_stage.image_size // 8,
                        train_cfg.first_stage.image_size // 8,
                    ),
                    device,
                    diffusion_visual_labels,
                    guidance_scale=train_cfg.diffusion.guidance_scale,
                )
                sample_outputs = first_stage.decode_with_intermediates(sample_batch * latent_scale)
                run_manager.save_tensor_grid(
                    build_diffusion_visualization_tensors(diffusion_visual_reference_images, sample_outputs),
                    run_manager.diffusion_images_dir / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/diffusion/epoch_{epoch_number:04d}.png",
                        f"diffusion epoch {epoch_number}",
                    ),
                    max_items=sample_count,
                )
        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.diffusion_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics, per_class_metrics = evaluate_diffusion_with_per_class(
                torch_module,
                first_stage,
                diffusion_model,
                scheduler,
                diffusion_per_sample_losses_fn,
                validation_bundle.loader,
                device,
                use_pin_memory,
                latent_scale,
                train_cfg.diffusion.latent_source,
                train_cfg.diffusion.num_classes,
            )
            print(
                f"[diffusion][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"latent_scale={validation_metrics['latent_scale']:.6f}"
            )
            run_manager.append_history(
                "diffusion_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            run_manager.append_history(
                "diffusion_validation_per_class",
                {
                    "epoch": epoch_number,
                    "classes": per_class_metrics,
                },
            )
            selection_loss = validation_metrics["loss"]
            updated_best_diffusion = best_diffusion_loss is None or selection_loss < best_diffusion_loss
            if updated_best_diffusion:
                best_diffusion_loss = selection_loss
                best_diffusion_checkpoint_path = run_manager.save_checkpoint(
                    DIFFUSION_BEST_CHECKPOINT,
                    {
                        "model_state_dict": diffusion_model.state_dict(),
                        "latent_scale": latent_scale.detach().cpu(),
                        "latent_source": train_cfg.diffusion.latent_source,
                        "stage": "diffusion",
                        "model_version": diffusion_model_version,
                        "first_stage_model_version": first_stage_model_version,
                        "conditioning": diffusion_model.conditioning_metadata(),
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": DIFFUSION_BEST_METRIC,
                        "selection_loss": selection_loss,
                        "validation_metrics": validation_metrics,
                    },
                )
                run_meta.update(
                    {
                        "diffusion_best_checkpoint": best_diffusion_checkpoint_path,
                        "diffusion_best_epoch": epoch_number,
                        "diffusion_best_loss": best_diffusion_loss,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "diffusion_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best_diffusion,
                    "selection_metric": DIFFUSION_BEST_METRIC,
                    "selection_loss": selection_loss,
                    "best_loss": best_diffusion_loss,
                    "checkpoint": best_diffusion_checkpoint_path,
                },
            )
            diffusion_model.train()

    run_manager.save_checkpoint(
        "diffusion_latest.pt",
        {
            "model_state_dict": diffusion_model.state_dict(),
            "latent_scale": latent_scale.detach().cpu(),
            "latent_source": train_cfg.diffusion.latent_source,
            "stage": "diffusion",
            "model_version": diffusion_model_version,
            "first_stage_model_version": first_stage_model_version,
            "conditioning": diffusion_model.conditioning_metadata(),
            "physics_config": physics,
            "train_config": train_cfg,
        },
    )

    diffusion_model.eval()
    if best_diffusion_checkpoint_path is not None:
        # 最终样图用验证集上最好的 diffusion 权重，避免最后一个 epoch 回退污染展示。
        best_diffusion_payload = torch_module.load(
            best_diffusion_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        load_diffusion_checkpoint_fn(diffusion_model, best_diffusion_payload, first_stage_model_version)
        run_meta["diffusion_best_loaded_for_sampling"] = True
        run_manager.write_meta(run_meta)
        print(f"loaded best diffusion for sampling: {best_diffusion_checkpoint_path}")
        diffusion_model.eval()
    sample_artifacts = save_post_train_samples(
        torch_module,
        run_manager,
        first_stage,
        diffusion_model,
        scheduler,
        train_bundle,
        validation_bundle,
        train_cfg,
        latent_scale,
        device,
        sample_latents_fn,
        sample_latents_with_trace_fn,
        resolve_denoising_trace_steps_fn,
    )
    run_meta.update(sample_artifacts)
    run_manager.write_meta(run_meta)
