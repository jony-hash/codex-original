from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def compute_camera_mnist_score(
    camera_corr: float,
    camera_edge_corr: float,
    camera_foreground_dice: float,
    camera_background_leakage: float,
    camera_hotspot_ratio: float,
    class_floor_corr: float,
) -> float:
    return (
        0.35 * camera_corr
        + 0.25 * camera_edge_corr
        + 0.30 * camera_foreground_dice
        + 0.10 * class_floor_corr
        - 0.10 * camera_background_leakage
        - 0.10 * camera_hotspot_ratio
    )


def compute_first_stage_balanced_metrics(
    validation_metrics: dict[str, float],
    per_class_metrics: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    camera_corr = float(validation_metrics["camera_corr"])
    camera_edge_corr = float(validation_metrics["camera_edge_corr"])
    class_scores = [
        float(metrics["camera_corr"])
        for metrics in per_class_metrics.values()
        if "camera_corr" in metrics
    ]
    min_class_camera_corr = min(class_scores) if class_scores else camera_corr
    # 兼顾整体相似度、边缘结构和最弱类别，避免只优化容易数字。
    camera_balanced_score = 0.5 * camera_corr + 0.3 * camera_edge_corr + 0.2 * min_class_camera_corr
    camera_foreground_dice = float(validation_metrics["camera_foreground_dice"])
    camera_background_leakage = float(validation_metrics["camera_background_leakage"])
    camera_hotspot_ratio = float(validation_metrics["camera_hotspot_ratio"])
    camera_mnist_score = compute_camera_mnist_score(
        camera_corr,
        camera_edge_corr,
        camera_foreground_dice,
        camera_background_leakage,
        camera_hotspot_ratio,
        min_class_camera_corr,
    )
    class_mnist_scores = []
    for metrics in per_class_metrics.values():
        if not {
            "camera_corr",
            "camera_edge_corr",
            "camera_foreground_dice",
            "camera_background_leakage",
            "camera_hotspot_ratio",
        }.issubset(metrics):
            continue
        class_score = compute_camera_mnist_score(
            float(metrics["camera_corr"]),
            float(metrics["camera_edge_corr"]),
            float(metrics["camera_foreground_dice"]),
            float(metrics["camera_background_leakage"]),
            float(metrics["camera_hotspot_ratio"]),
            float(metrics["camera_corr"]),
        )
        metrics["camera_mnist_score"] = class_score
        class_mnist_scores.append(class_score)
    min_class_camera_mnist_score = min(class_mnist_scores) if class_mnist_scores else camera_mnist_score
    camera_class_balanced_mnist_score = 0.5 * camera_mnist_score + 0.5 * min_class_camera_mnist_score
    complex_camera_corr = float(validation_metrics.get("complex_camera_corr", camera_corr))
    complex_camera_edge_corr = float(validation_metrics.get("complex_camera_edge_corr", camera_edge_corr))
    complex_camera_foreground_dice = float(
        validation_metrics.get("complex_camera_foreground_dice", camera_foreground_dice)
    )
    complex_camera_background_leakage = float(
        validation_metrics.get("complex_camera_background_leakage", camera_background_leakage)
    )
    complex_camera_hotspot_ratio = float(
        validation_metrics.get("complex_camera_hotspot_ratio", camera_hotspot_ratio)
    )
    complex_class_floor_corr = min(min_class_camera_corr, complex_camera_corr)
    complex_camera_mnist_score = compute_camera_mnist_score(
        complex_camera_corr,
        complex_camera_edge_corr,
        complex_camera_foreground_dice,
        complex_camera_background_leakage,
        complex_camera_hotspot_ratio,
        complex_class_floor_corr,
    )
    # v3 best 选择同时看全体验证和复杂子集，避免简单数字把 checkpoint 分数抬高。
    camera_complex_balanced_mnist_score = 0.5 * camera_class_balanced_mnist_score + 0.5 * complex_camera_mnist_score
    return {
        "min_class_camera_corr": min_class_camera_corr,
        "camera_balanced_score": camera_balanced_score,
        "camera_mnist_score": camera_mnist_score,
        "min_class_camera_mnist_score": min_class_camera_mnist_score,
        "camera_class_balanced_mnist_score": camera_class_balanced_mnist_score,
        "complex_camera_mnist_score": complex_camera_mnist_score,
        "camera_complex_balanced_mnist_score": camera_complex_balanced_mnist_score,
    }


def average_scalar_metrics(metric_totals: dict[str, float], example_count: int) -> dict[str, float]:
    if example_count <= 0:
        raise ValueError("Validation loader is empty.")
    return {name: total / float(example_count) for name, total in metric_totals.items()}


def _mean_flattened_correlation(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    prediction_flat = prediction.detach().float().flatten(1)
    target_flat = target.detach().float().flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (prediction_centered * target_centered).mean(dim=1)
    denominator = (
        prediction_centered.pow(2).mean(dim=1).sqrt() * target_centered.pow(2).mean(dim=1).sqrt()
    ).clamp_min(1.0e-6)
    return (numerator / denominator).mean()


def _edge_magnitude_like(torch_module, tensor: "torch.Tensor") -> "torch.Tensor":
    tensor = tensor.detach().float()
    edge = torch_module.zeros_like(tensor)
    diff_y = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs()
    diff_x = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs()
    edge[..., 1:, :] = edge[..., 1:, :] + diff_y
    edge[..., :-1, :] = edge[..., :-1, :] + diff_y
    edge[..., :, 1:] = edge[..., :, 1:] + diff_x
    edge[..., :, :-1] = edge[..., :, :-1] + diff_x
    return edge


def compute_input_complexity_scores(
    torch_module,
    input_batch: "torch.Tensor",
    foreground_threshold: float = 0.05,
) -> "torch.Tensor":
    edge_density = _edge_magnitude_like(torch_module, input_batch).flatten(1).mean(dim=1)
    foreground_density = (input_batch > foreground_threshold).float().flatten(1).mean(dim=1)
    return edge_density + 0.25 * foreground_density


def select_complex_subset_mask(
    torch_module,
    input_batch: "torch.Tensor",
    foreground_threshold: float = 0.05,
) -> "torch.Tensor":
    scores = compute_input_complexity_scores(torch_module, input_batch, foreground_threshold)
    if scores.numel() == 0:
        return scores.bool()
    keep_count = max(1, int(math.ceil(scores.numel() * 0.5)))
    keep_indices = torch_module.topk(scores, k=keep_count).indices
    mask = torch_module.zeros_like(scores, dtype=torch_module.bool)
    mask[keep_indices] = True
    return mask


def first_stage_quality_metrics(
    torch_module,
    outputs: dict[str, "torch.Tensor"],
    input_batch: "torch.Tensor",
    foreground_threshold: float = 0.05,
) -> dict[str, "torch.Tensor"]:
    camera_plane = outputs["camera_plane"]
    camera_loss_plane = outputs.get("camera_plane_loss", camera_plane)
    image_recon = outputs["image_recon"]
    foreground_mask = (input_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    dims = tuple(range(1, camera_plane.ndim))
    dice_intersection = (camera_plane.clamp(0.0, 1.0) * foreground_mask).sum(dim=dims)
    dice_denominator = camera_plane.clamp(0.0, 1.0).sum(dim=dims) + foreground_mask.sum(dim=dims)
    camera_foreground_dice = ((2.0 * dice_intersection + 1.0e-6) / (dice_denominator + 1.0e-6)).mean()
    # 这些指标直接服务 MNIST 可读性：背景漏光和未 clamp 热点都会压低 checkpoint 分数。
    camera_background_leakage = (
        camera_plane.clamp_min(0.0) * background_mask
    ).sum() / background_mask.sum().clamp_min(1e-6)
    camera_hotspot_ratio = (camera_loss_plane > 1.0).float().mean()
    return {
        "camera_corr": _mean_flattened_correlation(camera_plane, input_batch),
        "image_corr": _mean_flattened_correlation(image_recon, input_batch),
        "camera_edge_corr": _mean_flattened_correlation(
            _edge_magnitude_like(torch_module, camera_plane),
            _edge_magnitude_like(torch_module, input_batch),
        ),
        "camera_foreground_dice": camera_foreground_dice,
        "camera_background_leakage": camera_background_leakage,
        "camera_hotspot_ratio": camera_hotspot_ratio,
    }


def evaluate_first_stage(
    torch_module,
    first_stage,
    first_stage_loss_fn: Callable[..., dict[str, "torch.Tensor"]],
    loader,
    device,
    use_pin_memory: bool,
    stage_step: int,
    total_steps: int,
    train_cfg,
) -> dict[str, float]:
    overall_metrics, _ = evaluate_first_stage_with_per_class(
        torch_module,
        first_stage,
        first_stage_loss_fn,
        loader,
        device,
        use_pin_memory,
        stage_step,
        total_steps,
        train_cfg,
    )
    return overall_metrics


def evaluate_first_stage_with_per_class(
    torch_module,
    first_stage,
    first_stage_loss_fn: Callable[..., dict[str, "torch.Tensor"]],
    loader,
    device,
    use_pin_memory: bool,
    stage_step: int,
    total_steps: int,
    train_cfg,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    metric_names = (
        "loss",
        "recon_loss",
        "image_recon_loss",
        "image_fg_mse",
        "image_bg_mse",
        "camera_recon_loss",
        "camera_fg_mse",
        "camera_bg_mse",
        "structure_loss",
        "image_structure_loss",
        "camera_structure_loss",
        "shape_loss",
        "image_shape_loss",
        "camera_shape_loss",
        "camera_foreground_dice_loss",
        "camera_foreground_dice",
        "field_smoothness_loss",
        "field_tv",
        "phase_smoothness_loss",
        "phase_tv",
        "field_energy_loss",
        "field_energy",
        "background_leakage_loss",
        "camera_background_leakage",
        "camera_hotspot_loss",
        "camera_hotspot_ratio",
        "camera_saturation_ratio",
        "camera_topology_loss",
        "camera_topology_base",
        "camera_saturation_loss",
        "camera_saturation_penalty",
        "camera_artifact_loss",
        "camera_foreground_hole_loss",
        "camera_foreground_hole_ratio",
        "camera_foreground_roughness",
        "kl_loss",
        "fg_pixel_ratio",
        "pred_fg_energy_ratio",
        "beta",
        "camera_corr",
        "image_corr",
        "camera_edge_corr",
    )
    per_class_metric_names = (
        "camera_fg_mse",
        "image_fg_mse",
        "camera_recon_loss",
        "image_recon_loss",
        "camera_corr",
        "image_corr",
        "camera_edge_corr",
        "camera_foreground_dice",
        "camera_background_leakage",
        "camera_hotspot_ratio",
    )
    complex_metric_names = (
        "camera_corr",
        "camera_edge_corr",
        "camera_foreground_dice",
        "camera_background_leakage",
        "camera_hotspot_ratio",
    )
    metric_totals = {name: 0.0 for name in metric_names}
    example_count = 0
    class_count = train_cfg.diffusion.num_classes
    class_metric_totals = {label: {name: 0.0 for name in per_class_metric_names} for label in range(class_count)}
    class_counts = {label: 0 for label in range(class_count)}
    complex_metric_totals = {name: 0.0 for name in complex_metric_names}
    complex_example_count = 0
    validation_step = max(stage_step - 1, 0)
    first_stage.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            input_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            outputs = first_stage(input_batch)
            losses = first_stage_loss_fn(
                outputs,
                input_batch,
                global_step=validation_step,
                total_steps=total_steps,
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
            quality_metrics = first_stage_quality_metrics(
                torch_module,
                outputs,
                input_batch,
                train_cfg.first_stage.recon_foreground_threshold,
            )
            batch_size = input_batch.shape[0]
            for name in metric_names:
                if name in quality_metrics:
                    metric_totals[name] += quality_metrics[name].item() * batch_size
                    continue
                metric_totals[name] += losses[name].item() * batch_size
            example_count += batch_size
            complex_mask = select_complex_subset_mask(
                torch_module,
                input_batch,
                train_cfg.first_stage.recon_foreground_threshold,
            )
            complex_count = int(complex_mask.sum().item())
            if complex_count > 0:
                complex_quality_metrics = first_stage_quality_metrics(
                    torch_module,
                    {key: value[complex_mask] for key, value in outputs.items()},
                    input_batch[complex_mask],
                    train_cfg.first_stage.recon_foreground_threshold,
                )
                complex_example_count += complex_count
                for name in complex_metric_names:
                    complex_metric_totals[name] += complex_quality_metrics[name].item() * complex_count
            for label in range(class_count):
                mask = class_labels == label
                count = int(mask.sum().item())
                if count == 0:
                    continue
                label_losses = first_stage_loss_fn(
                    {key: value[mask] for key, value in outputs.items()},
                    input_batch[mask],
                    global_step=validation_step,
                    total_steps=total_steps,
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
                label_quality_metrics = first_stage_quality_metrics(
                    torch_module,
                    {key: value[mask] for key, value in outputs.items()},
                    input_batch[mask],
                    train_cfg.first_stage.recon_foreground_threshold,
                )
                class_counts[label] += count
                for name in per_class_metric_names:
                    if name in label_quality_metrics:
                        class_metric_totals[label][name] += label_quality_metrics[name].item() * count
                    else:
                        class_metric_totals[label][name] += label_losses[name].item() * count

    per_class: dict[str, dict[str, float | int]] = {}
    for label in range(class_count):
        count = class_counts[label]
        if count == 0:
            continue
        per_class[str(label)] = {
            "count": count,
            **{name: class_metric_totals[label][name] / float(count) for name in per_class_metric_names},
        }
    overall_metrics = average_scalar_metrics(metric_totals, example_count)
    if complex_example_count > 0:
        overall_metrics.update(
            {
                f"complex_{name}": complex_metric_totals[name] / float(complex_example_count)
                for name in complex_metric_names
            }
        )
    return overall_metrics, per_class


def evaluate_diffusion_with_per_class(
    torch_module,
    first_stage,
    diffusion_model,
    scheduler,
    diffusion_per_sample_loss_fn: Callable[..., "torch.Tensor"],
    loader,
    device,
    use_pin_memory: bool,
    latent_scale,
    latent_source: str,
    class_count: int,
) -> tuple[dict[str, float], dict[str, dict[str, float | int | None]]]:
    metric_totals = {"loss": 0.0}
    class_totals = {label: 0.0 for label in range(class_count)}
    class_counts = {label: 0 for label in range(class_count)}
    example_count = 0
    diffusion_model.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            per_sample_losses = diffusion_per_sample_loss_fn(
                first_stage,
                diffusion_model,
                scheduler,
                batch,
                class_labels,
                latent_scale,
                latent_source=latent_source,
            )
            batch_size = batch.shape[0]
            metric_totals["loss"] += per_sample_losses.sum().item()
            example_count += batch_size
            for label in range(class_count):
                mask = class_labels == label
                count = int(mask.sum().item())
                if count == 0:
                    continue
                class_counts[label] += count
                class_totals[label] += per_sample_losses[mask].sum().item()

    averaged = average_scalar_metrics(metric_totals, example_count)
    averaged["latent_scale"] = latent_scale.item()
    per_class = {
        str(label): {
            "loss": class_totals[label] / float(class_counts[label]) if class_counts[label] > 0 else None,
            "count": class_counts[label],
        }
        for label in range(class_count)
    }
    return averaged, per_class
