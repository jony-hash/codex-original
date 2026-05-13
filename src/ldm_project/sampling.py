from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .train_utils import validate_class_label_values

if TYPE_CHECKING:
    import torch


DEFAULT_DENOISING_TRACE_TIMESTEPS = (750, 500, 250, 100, 50, 0)
DEFAULT_NEAREST_REFERENCE_LIMIT = 256


def build_visualization_labels(torch_module, num_samples: int, num_classes: int, device):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    return torch_module.arange(num_samples, device=device, dtype=torch_module.long) % num_classes


def build_post_train_sample_labels(
    torch_module,
    num_classes: int,
    device,
    sample_class_labels: list[int] | tuple[int, ...] | None = None,
    sample_repeats_per_class: int = 1,
):
    if sample_repeats_per_class <= 0:
        raise ValueError(f"sample_repeats_per_class must be positive, got {sample_repeats_per_class}")

    requested_labels = tuple(sample_class_labels or ())
    if not requested_labels:
        return build_visualization_labels(torch_module, num_classes, num_classes, device)

    labels = validate_class_label_values(requested_labels, num_classes, "sample_class_labels")
    # 多次采样时按“0-9 一组”循环排列，最终网格更容易逐组横向比较。
    repeated_labels = [label for _ in range(sample_repeats_per_class) for label in labels]
    return torch_module.tensor(repeated_labels, device=device, dtype=torch_module.long)


def _normalize_for_visualization(tensor):
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def enhance_camera_for_display(tensor, lower_quantile: float = 0.01, upper_quantile: float = 0.995, gamma: float = 0.75):
    if lower_quantile < 0.0 or upper_quantile > 1.0 or lower_quantile >= upper_quantile:
        raise ValueError("display quantiles must satisfy 0 <= lower < upper <= 1")
    if gamma <= 0.0:
        raise ValueError(f"display gamma must be positive, got {gamma}")
    raw = tensor.detach().float()
    flat = raw.flatten(start_dim=-2)
    low = flat.quantile(lower_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    high = flat.quantile(upper_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = (raw - low) / (high - low).clamp_min(1e-6)
    return scaled.clamp(0.0, 1.0).pow(gamma)


def format_output_title(relative_path: str, label: str) -> str:
    return f"{relative_path} | {label}"


def build_first_stage_visualization_tensors(input_batch, outputs):
    # 物理优先先看相机平面是否承载数字，再用 decoder image 诊断 latent 语义。
    return {
        "input": input_batch,
        "camera_hat": outputs["camera_plane"],
        "image_hat": outputs["image_recon"],
        "pf2": outputs["pf2_intensity"],
        "chip_amplitude": _normalize_for_visualization(outputs["chip_amplitude"]),
        "chip_phase": ((outputs["chip_phase"] / math.pi) + 1.0).clamp(0.0, 2.0) * 0.5,
        "probe": outputs["probe_plane"],
    }


def build_diffusion_visualization_tensors(reference_images, sample_outputs):
    # diffusion 阶段先看生成 latent 经物理后级读出的 camera，再看 decoder image 作为辅助。
    return {
        "reference": reference_images,
        "generated_camera": sample_outputs["camera_plane"],
        "generated_image": sample_outputs["image_recon"],
        "chip_amplitude": _normalize_for_visualization(sample_outputs["chip_amplitude"]),
        "chip_phase": ((sample_outputs["chip_phase"] / math.pi) + 1.0).clamp(0.0, 2.0) * 0.5,
        "probe": sample_outputs["probe_plane"],
    }


def build_one_shot_visualization_tensors(reference_images, sample_outputs):
    # one-shot 阶段直接看单次光路 camera，同时保留 SLM 和 chip 诊断。
    camera_raw = sample_outputs["camera_plane"]
    return {
        "reference": reference_images,
        "generated_camera_raw": camera_raw,
        "generated_camera_display": enhance_camera_for_display(camera_raw),
        "slm_amplitude": sample_outputs["slm_amplitude"],
        "chip_amplitude": _normalize_for_visualization(sample_outputs["chip_amplitude"]),
        "chip_phase": ((sample_outputs["chip_phase"] / math.pi) + 1.0).clamp(0.0, 2.0) * 0.5,
    }


def build_denoising_trace_visualization_tensors(reference_images, camera_trace: dict[str, "torch.Tensor"]):
    # 每列展示同一条采样轨迹在指定去噪阶段的相机读出，方便直接看噪声如何变成数字。
    return {
        "reference": reference_images,
        **{label: tensor for label, tensor in camera_trace.items()},
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _compute_reference_similarity_metrics(
    prediction,
    references,
    foreground_threshold: float,
) -> dict[str, float | int]:
    prediction = prediction.detach().float().clamp(0.0, 1.0).cpu()
    references = references.detach().float().clamp(0.0, 1.0).cpu()
    if references.ndim == prediction.ndim:
        references = references.unsqueeze(0)
    if references.shape[0] <= 0:
        raise ValueError("nearest reference bank must contain at least one image per requested label.")

    prediction_mask = prediction > foreground_threshold
    reference_masks = references > foreground_threshold
    intersections = (reference_masks & prediction_mask.unsqueeze(0)).float().flatten(1).sum(dim=1)
    denominator = (
        prediction_mask.float().sum()
        + reference_masks.float().flatten(1).sum(dim=1)
    ).clamp_min(1.0)
    dice_scores = 2.0 * intersections / denominator
    best_dice, best_dice_index = dice_scores.max(dim=0)

    prediction_flat = prediction.flatten()
    references_flat = references.flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean()
    references_centered = references_flat - references_flat.mean(dim=1, keepdim=True)
    corr_denominator = (
        prediction_centered.pow(2).mean().sqrt()
        * references_centered.pow(2).mean(dim=1).sqrt()
    )
    corr_scores = (
        (references_centered * prediction_centered.unsqueeze(0)).mean(dim=1)
        / corr_denominator.clamp_min(1.0e-8)
    )
    best_corr, best_corr_index = corr_scores.max(dim=0)
    return {
        "dice": float(best_dice.item()),
        "corr": float(best_corr.item()),
        "dice_reference_index": int(best_dice_index.item()),
        "corr_reference_index": int(best_corr_index.item()),
    }


def _summarize_nearest_neighbor_metrics(
    generated,
    labels: list[int],
    reference_image_bank: dict[int, "torch.Tensor"],
    reference_bank_source: str | None,
    foreground_threshold: float,
) -> dict[str, object]:
    per_sample: list[dict[str, float | int]] = []
    per_class_values: dict[str, dict[str, list[float] | int]] = {}
    reference_counts = {
        str(label): int(images.shape[0])
        for label, images in sorted(reference_image_bank.items())
    }
    for index, label in enumerate(labels):
        if label not in reference_image_bank:
            raise ValueError(f"Missing nearest-neighbor reference images for class label: {label}")
        nearest = _compute_reference_similarity_metrics(
            generated[index],
            reference_image_bank[label],
            foreground_threshold,
        )
        sample_record = {
            "index": index,
            "label": label,
            **nearest,
        }
        per_sample.append(sample_record)
        class_record = per_class_values.setdefault(str(label), {"count": 0, "dice": [], "corr": []})
        class_record["count"] = int(class_record["count"]) + 1
        class_record["dice"].append(float(sample_record["dice"]))
        class_record["corr"].append(float(sample_record["corr"]))

    per_class = {}
    for label, values in sorted(per_class_values.items(), key=lambda item: int(item[0])):
        per_class[label] = {
            "count": int(values["count"]),
            "reference_count": reference_counts[label],
            "dice": _safe_mean(values["dice"]),
            "corr": _safe_mean(values["corr"]),
        }

    return {
        "reference_source": reference_bank_source,
        "reference_counts": reference_counts,
        "overall": {
            "count": len(per_sample),
            "dice": _safe_mean([float(item["dice"]) for item in per_sample]),
            "corr": _safe_mean([float(item["corr"]) for item in per_sample]),
        },
        "per_class": per_class,
        "per_sample": per_sample,
    }


def compute_sample_quality_metrics(
    generated_camera,
    reference_images,
    class_labels,
    foreground_threshold: float = 0.2,
    reference_image_bank: dict[int, "torch.Tensor"] | None = None,
    reference_bank_source: str | None = None,
) -> dict[str, object]:
    generated = generated_camera.detach().float().clamp(0.0, 1.0).cpu()
    reference = reference_images.detach().float().clamp(0.0, 1.0).cpu()
    labels = [int(label) for label in class_labels.detach().cpu().tolist()]
    if generated.shape[0] != reference.shape[0] or generated.shape[0] != len(labels):
        raise ValueError("generated_camera, reference_images, and class_labels must have matching batch size.")

    per_sample: list[dict[str, float | int]] = []
    per_class_values: dict[str, dict[str, list[float] | int]] = {}
    for index, label in enumerate(labels):
        prediction = generated[index]
        target = reference[index]
        prediction_mask = prediction > foreground_threshold
        target_mask = target > foreground_threshold
        intersection = (prediction_mask & target_mask).float().sum()
        denominator = prediction_mask.float().sum() + target_mask.float().sum()
        dice = (2.0 * intersection / denominator.clamp_min(1.0)).item()

        prediction_flat = prediction.flatten()
        target_flat = target.flatten()
        prediction_centered = prediction_flat - prediction_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr_denominator = prediction_centered.pow(2).mean().sqrt() * target_centered.pow(2).mean().sqrt()
        corr = ((prediction_centered * target_centered).mean() / corr_denominator.clamp_min(1.0e-8)).item()
        hole_proxy = (
            (target_mask & (prediction <= foreground_threshold)).float().sum()
            / target_mask.float().sum().clamp_min(1.0)
        ).item()

        sample_record = {
            "index": index,
            "label": label,
            "dice": float(dice),
            "corr": float(corr),
            "hole_proxy": float(hole_proxy),
        }
        per_sample.append(sample_record)
        class_record = per_class_values.setdefault(str(label), {"count": 0, "dice": [], "corr": [], "hole_proxy": []})
        class_record["count"] = int(class_record["count"]) + 1
        for metric_name in ("dice", "corr", "hole_proxy"):
            class_record[metric_name].append(float(sample_record[metric_name]))

    per_class = {}
    for label, values in sorted(per_class_values.items(), key=lambda item: int(item[0])):
        per_class[label] = {
            "count": int(values["count"]),
            "dice": _safe_mean(values["dice"]),
            "corr": _safe_mean(values["corr"]),
            "hole_proxy": _safe_mean(values["hole_proxy"]),
        }

    metrics = {
        "foreground_threshold": foreground_threshold,
        "overall": {
            "count": len(per_sample),
            "dice": _safe_mean([float(item["dice"]) for item in per_sample]),
            "corr": _safe_mean([float(item["corr"]) for item in per_sample]),
            "hole_proxy": _safe_mean([float(item["hole_proxy"]) for item in per_sample]),
        },
        "per_class": per_class,
        "per_sample": per_sample,
    }
    if reference_image_bank is not None:
        metrics["nearest_neighbor"] = _summarize_nearest_neighbor_metrics(
            generated,
            labels,
            reference_image_bank,
            reference_bank_source,
            foreground_threshold,
        )
    return metrics


def _resolve_label_value(label) -> int:
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _collect_reference_images_from_dataset(
    torch_module,
    dataset,
    requested_labels: list[int],
) -> dict[int, "torch.Tensor"]:
    requested_set = set(requested_labels)
    found_images: dict[int, "torch.Tensor"] = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label_value = _resolve_label_value(label)
        if label_value not in requested_set or label_value in found_images:
            continue
        found_images[label_value] = torch_module.as_tensor(image).detach().float().cpu()
        if len(found_images) == len(requested_set):
            break
    return found_images


def _collect_reference_image_bank_from_dataset(
    torch_module,
    dataset,
    requested_labels: list[int],
    max_items_per_label: int = DEFAULT_NEAREST_REFERENCE_LIMIT,
) -> dict[int, "torch.Tensor"]:
    requested_set = set(requested_labels)
    found_images: dict[int, list["torch.Tensor"]] = {label: [] for label in requested_set}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label_value = _resolve_label_value(label)
        if label_value not in requested_set:
            continue
        label_images = found_images[label_value]
        if len(label_images) >= max_items_per_label:
            continue
        label_images.append(torch_module.as_tensor(image).detach().float().cpu())
        if all(len(images) >= max_items_per_label for images in found_images.values()):
            break
    return {
        label: torch_module.stack(images)
        for label, images in found_images.items()
        if images
    }


def build_reference_image_bank_for_labels(
    torch_module,
    class_labels,
    validation_bundle,
    train_bundle,
    max_items_per_label: int = DEFAULT_NEAREST_REFERENCE_LIMIT,
) -> tuple[dict[int, "torch.Tensor"], str]:
    label_values = sorted({int(label) for label in class_labels.detach().cpu().tolist()})
    validation_bank: dict[int, "torch.Tensor"] = {}
    if validation_bundle is not None:
        validation_bank = _collect_reference_image_bank_from_dataset(
            torch_module,
            validation_bundle.dataset,
            label_values,
            max_items_per_label=max_items_per_label,
        )

    found_bank = dict(validation_bank)
    missing_labels = sorted(label for label in label_values if label not in found_bank)
    if missing_labels:
        found_bank.update(
            _collect_reference_image_bank_from_dataset(
                torch_module,
                train_bundle.dataset,
                missing_labels,
                max_items_per_label=max_items_per_label,
            )
        )

    unresolved = [label for label in label_values if label not in found_bank]
    if unresolved:
        raise ValueError(f"Unable to find nearest-neighbor reference images for class labels: {unresolved}")
    if validation_bundle is None:
        reference_bank_source = "train_only"
    elif all(label in validation_bank for label in label_values):
        reference_bank_source = "validation_only"
    else:
        reference_bank_source = "validation_then_train_fallback"
    return found_bank, reference_bank_source


def _collect_label_values_from_dataset(dataset) -> set[int]:
    label_values: set[int] = set()
    for index in range(len(dataset)):
        _, label = dataset[index]
        label_values.add(_resolve_label_value(label))
    return label_values


def select_reference_images_for_labels(
    torch_module,
    class_labels,
    validation_bundle,
    train_bundle,
    device,
):
    label_values = [_resolve_label_value(label) for label in class_labels.detach().cpu()]
    validation_found: dict[int, "torch.Tensor"] = {}
    if validation_bundle is not None:
        validation_found = _collect_reference_images_from_dataset(
            torch_module,
            validation_bundle.dataset,
            label_values,
        )

    found_images = dict(validation_found)
    missing_labels = sorted({label for label in label_values if label not in found_images})
    if missing_labels:
        found_images.update(
            _collect_reference_images_from_dataset(
                torch_module,
                train_bundle.dataset,
                missing_labels,
            )
        )

    unresolved = [label for label in label_values if label not in found_images]
    if unresolved:
        raise ValueError(f"Unable to find reference images for class labels: {sorted(set(unresolved))}")

    reference_images = torch_module.stack([found_images[label] for label in label_values]).to(device=device)
    reference_labels = torch_module.tensor(label_values, device=device, dtype=torch_module.long)
    if validation_bundle is None:
        reference_source = "train_only"
    elif all(label in validation_found for label in set(label_values)):
        reference_source = "validation_only"
    else:
        reference_source = "validation_then_train_fallback"
    return reference_images, reference_labels, reference_source


def select_first_stage_visual_batch(
    torch_module,
    validation_bundle,
    train_bundle,
    num_classes: int,
    device,
):
    # 固定 0-9 验证样本，避免每个 epoch 因训练 batch 变化而误判结构改进。
    visual_labels = build_visualization_labels(torch_module, num_classes, num_classes, device)
    try:
        return select_reference_images_for_labels(
            torch_module,
            visual_labels,
            validation_bundle,
            train_bundle,
            device,
        )
    except ValueError:
        available_labels = set()
        if validation_bundle is not None:
            available_labels.update(_collect_label_values_from_dataset(validation_bundle.dataset))
        available_labels.update(_collect_label_values_from_dataset(train_bundle.dataset))
        fallback_labels = sorted(label for label in available_labels if 0 <= label < num_classes)
        if not fallback_labels:
            raise
        fallback_tensor = torch_module.tensor(fallback_labels, device=device, dtype=torch_module.long)
        return select_reference_images_for_labels(
            torch_module,
            fallback_tensor,
            validation_bundle,
            train_bundle,
            device,
        )


def select_post_train_sample_batch(
    torch_module,
    validation_bundle,
    train_bundle,
    num_classes: int,
    device,
    sample_class_labels: list[int] | tuple[int, ...] | None = None,
    sample_repeats_per_class: int = 1,
):
    # 正式 MNIST 采样固定 0-9；小测试集缺类时退回到实际存在的类别，避免烟测被样本构造限制卡住。
    sample_labels = build_post_train_sample_labels(
        torch_module,
        num_classes,
        device,
        sample_class_labels=sample_class_labels,
        sample_repeats_per_class=sample_repeats_per_class,
    )
    try:
        reference_images, reference_labels, reference_source = select_reference_images_for_labels(
            torch_module,
            sample_labels,
            validation_bundle,
            train_bundle,
            device,
        )
        return sample_labels, reference_images, reference_labels, reference_source
    except ValueError:
        if sample_class_labels:
            raise
        available_labels = set()
        if validation_bundle is not None:
            available_labels.update(_collect_label_values_from_dataset(validation_bundle.dataset))
        available_labels.update(_collect_label_values_from_dataset(train_bundle.dataset))
        fallback_labels = sorted(label for label in available_labels if 0 <= label < num_classes)
        if not fallback_labels:
            raise
        fallback_tensor = torch_module.tensor(fallback_labels, device=device, dtype=torch_module.long)
        reference_images, reference_labels, reference_source = select_reference_images_for_labels(
            torch_module,
            fallback_tensor,
            validation_bundle,
            train_bundle,
            device,
        )
        return fallback_tensor, reference_images, reference_labels, reference_source


def save_post_train_samples(
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
    sample_latents_with_trace_fn=None,
    resolve_trace_steps_fn=None,
) -> dict[str, str]:
    sample_labels, reference_images, reference_labels, reference_source = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.diffusion.num_classes,
        device,
        sample_class_labels=train_cfg.diffusion.sample_class_labels,
        sample_repeats_per_class=train_cfg.diffusion.sample_repeats_per_class,
    )
    reference_image_bank, reference_bank_source = build_reference_image_bank_for_labels(
        torch_module,
        sample_labels,
        validation_bundle,
        train_bundle,
    )
    sample_count = int(sample_labels.shape[0])
    run_manager.save_config_snapshot(
        "post_train_sample_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "guidance_scale": train_cfg.diffusion.guidance_scale,
            "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
            "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
            "source": "train.py_post_train_visualization",
        },
    )
    sample_shape = (
        sample_count,
        train_cfg.first_stage.latent_channels,
        train_cfg.first_stage.image_size // 8,
        train_cfg.first_stage.image_size // 8,
    )
    with torch_module.no_grad():
        if sample_latents_with_trace_fn is None:
            final_sample_batch = sample_latents_fn(
                diffusion_model,
                scheduler,
                sample_shape,
                device,
                sample_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
            )
            trace_latents = {"t000": final_sample_batch}
        else:
            final_sample_batch, trace_latents = sample_latents_with_trace_fn(
                diffusion_model,
                scheduler,
                sample_shape,
                device,
                sample_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
                trace_timesteps=DEFAULT_DENOISING_TRACE_TIMESTEPS,
            )
        final_sample_outputs = first_stage.decode_with_intermediates(final_sample_batch * latent_scale)
        camera_trace = {
            label: first_stage.decode_with_intermediates(latents * latent_scale)["camera_plane"]
            for label, latents in trace_latents.items()
        }
    # 训练完成后的最终 sample 统一保存在 train run 目录，避免再创建独立 sample run。
    run_manager.save_tensor_grid(
        build_diffusion_visualization_tensors(reference_images, final_sample_outputs),
        run_manager.samples_dir / "post_train_sample_grid.png",
        title=format_output_title("images/samples/post_train_sample_grid.png", "post-train sample outputs"),
        max_items=sample_count,
    )
    trace_steps = (
        resolve_trace_steps_fn(scheduler.timesteps, DEFAULT_DENOISING_TRACE_TIMESTEPS)
        if resolve_trace_steps_fn is not None
        else []
    )
    run_manager.save_config_snapshot(
        "post_train_denoising_trace_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "guidance_scale": train_cfg.diffusion.guidance_scale,
            "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
            "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
            "trace_columns": ["reference", *camera_trace.keys()],
            "trace_steps": trace_steps,
            "source": "train.py_post_train_denoising_trace",
        },
    )
    run_manager.save_tensor_grid(
        build_denoising_trace_visualization_tensors(reference_images, camera_trace),
        run_manager.samples_dir / "post_train_denoising_camera_trace.png",
        title=format_output_title(
            "images/samples/post_train_denoising_camera_trace.png",
            "post-train denoising camera trace",
        ),
        max_items=sample_count,
    )
    run_manager.save_tensor_bundle(
        "post_train_sample_tensors.pt",
        {
            "latents": final_sample_batch.detach().cpu(),
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
            "generated_camera": final_sample_outputs["camera_plane"].detach().cpu(),
            "generated_image": final_sample_outputs["image_recon"].detach().cpu(),
            "chip_amplitude": final_sample_outputs["chip_amplitude"].detach().cpu(),
            "chip_phase": final_sample_outputs["chip_phase"].detach().cpu(),
            "probe_plane": final_sample_outputs["probe_plane"].detach().cpu(),
            "camera_plane": final_sample_outputs["camera_plane"].detach().cpu(),
        },
    )
    run_manager.save_tensor_bundle(
        "post_train_denoising_trace_tensors.pt",
        {
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
            "trace_steps": trace_steps,
            "latent_trace": {label: latents.detach().cpu() for label, latents in trace_latents.items()},
            "camera_trace": {label: camera.detach().cpu() for label, camera in camera_trace.items()},
        },
    )
    sample_metrics = compute_sample_quality_metrics(
        final_sample_outputs["camera_plane"],
        reference_images,
        sample_labels,
        reference_image_bank=reference_image_bank,
        reference_bank_source=reference_bank_source,
    )
    metrics_path = run_manager.save_artifact_json("post_train_sample_metrics", sample_metrics)
    return {
        "sample_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
    }
