from __future__ import annotations

import math
from pathlib import Path

from .one_shot_distill_data import _filter_distill_payload_by_indices, _load_distill_payload, _validated_distill_tensors
from .one_shot_stage_common import _sample_one_shot_codes_with_seed
from .sampling import (
    enhance_camera_for_display,
    format_output_title,
    save_visualization_grids,
)
from .train_utils import (
    load_latent_scale_from_diffusion_checkpoint,
    set_first_stage_image_fusion_weight,
    validate_class_label_values,
)


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

def _balanced_preview_indices(torch_module, labels, *, max_per_class: int = 2, max_total: int = 20):
    labels_cpu = labels.detach().long().cpu()
    preview: list[int] = []
    for label in sorted({int(item) for item in labels_cpu.tolist()}):
        class_indices = torch_module.nonzero(labels_cpu == label, as_tuple=False).flatten().tolist()
        preview.extend(int(index) for index in class_indices[:max_per_class])
        if len(preview) >= max_total:
            break
    return torch_module.tensor(preview[:max_total], dtype=torch_module.long)

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
