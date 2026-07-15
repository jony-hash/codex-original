# ruff: noqa: F401,E402
from __future__ import annotations

from pathlib import Path

from ..distillation.dataset import (
    filter_distill_payload_by_indices,
    load_distill_payload,
    validated_distill_tensors,
)
from ..generation.codes import (
    sample_one_shot_codes_with_seed,
    project_codes_to_teacher_noise,
)
from ..visualization_core import (
    _normalize_for_visualization,
    enhance_camera_for_display,
    format_output_title,
    readable_camera_for_display,
)
from ..train_utils import (
    load_latent_scale_from_diffusion_checkpoint,
    set_first_stage_image_fusion_weight,
    validate_class_label_values,
)

_project_codes_to_teacher_noise = project_codes_to_teacher_noise

from .distill_support import _balanced_preview_indices, _class_relative_minmax, _metric_summary
from .label_audit import _resolve_teacher_label_audit_inputs, _save_teacher_label_audit_artifacts

def compute_teacher_distill_filter_metrics(
    torch_module,
    teacher_camera,
    *,
    foreground_threshold: float = 0.2,
) -> dict[str, object]:
    camera = teacher_camera.detach().float().clamp(0.0, 1.0).cpu()
    if camera.ndim != 4:
        raise ValueError(
            f"teacher_camera must be [N, C, H, W], got {tuple(camera.shape)}"
        )
    flattened = camera.flatten(1)
    foreground_mask = camera > float(foreground_threshold)
    mask_flattened = foreground_mask.float().flatten(1)
    mask = foreground_mask.float()
    mask_tv_y = (mask[..., 1:, :] - mask[..., :-1, :]).abs().flatten(1).mean(dim=1)
    mask_tv_x = (mask[..., :, 1:] - mask[..., :, :-1]).abs().flatten(1).mean(dim=1)
    return {
        "foreground_ratio": mask_flattened.mean(dim=1),
        "saturation_ratio": ((camera >= 0.99) | (camera <= 0.01))
        .float()
        .flatten(1)
        .mean(dim=1),
        "mask_tv": mask_tv_x + mask_tv_y,
        "contrast_std": flattened.std(dim=1, unbiased=False),
    }

def _select_teacher_distill_label_strict_indices(
    torch_module, labels, audit_tensors: dict[str, object], train_cfg
):
    keep_per_class = int(train_cfg.distillation.label_strict_keep_per_class)
    if keep_per_class <= 0:
        raise ValueError(
            f"distillation.label_strict_keep_per_class must be positive, got {keep_per_class}"
        )
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
        class_indices = torch_module.nonzero(
            labels_cpu == int(label), as_tuple=False
        ).flatten()
        class_count = int(class_indices.numel())
        if class_count == 0:
            per_class[str(label)] = {
                "available": 0,
                "kept": 0,
                "label_correct_available": 0,
                "relaxed_label_filter": True,
            }
            continue
        class_correct = audit_tensors["label_correct"].index_select(0, class_indices)
        correct_indices = class_indices[class_correct]
        relaxed = int(correct_indices.numel()) < keep_per_class
        if relaxed:
            incorrect_indices = class_indices[~class_correct]
            remaining = keep_per_class - int(correct_indices.numel())
            margin = audit_tensors["label_margin"].index_select(0, incorrect_indices)
            contrast = audit_tensors["readable_contrast"].index_select(
                0, incorrect_indices
            )
            speckle = audit_tensors["background_speckle_ratio"].index_select(
                0, incorrect_indices
            )
            holes = audit_tensors["foreground_hole_ratio"].index_select(
                0, incorrect_indices
            )
            score = margin + 0.25 * contrast - 0.25 * speckle - 0.25 * holes
            order = torch_module.argsort(score, descending=True)
            filler = incorrect_indices.index_select(0, order[:remaining])
            chosen = torch_module.cat([correct_indices, filler], dim=0)
        else:
            margin = audit_tensors["label_margin"].index_select(0, correct_indices)
            contrast = audit_tensors["readable_contrast"].index_select(
                0, correct_indices
            )
            speckle = audit_tensors["background_speckle_ratio"].index_select(
                0, correct_indices
            )
            holes = audit_tensors["foreground_hole_ratio"].index_select(
                0, correct_indices
            )
            score = margin + 0.25 * contrast - 0.25 * speckle - 0.25 * holes
            order = torch_module.argsort(score, descending=True)
            chosen = correct_indices.index_select(0, order[:keep_per_class])
        selected_indices.extend(int(index) for index in chosen.tolist())
        per_class[str(label)] = {
            "available": class_count,
            "kept": int(chosen.numel()),
            "label_correct_available": int(correct_indices.numel()),
            "relaxed_label_filter": relaxed,
            "selected_label_correct": int(
                audit_tensors["label_correct"].index_select(0, chosen).sum().item()
            ),
            "label_margin": _metric_summary(
                audit_tensors["label_margin"].index_select(0, chosen)
            ),
            "requested_label_corr": _metric_summary(
                audit_tensors["requested_label_corr"].index_select(0, chosen)
            ),
            "background_speckle_ratio": _metric_summary(
                audit_tensors["background_speckle_ratio"].index_select(0, chosen)
            ),
            "readable_contrast": _metric_summary(
                audit_tensors["readable_contrast"].index_select(0, chosen)
            ),
            "foreground_hole_ratio": _metric_summary(
                audit_tensors["foreground_hole_ratio"].index_select(0, chosen)
            ),
        }
    return selected_indices, {
        "teacher_filter": "label_strict",
        "class_labels": list(class_labels),
        "keep_per_class": keep_per_class,
        "selected_count": len(selected_indices),
        "per_class": per_class,
    }

def _select_teacher_distill_filter_indices(
    torch_module, labels, metric_tensors: dict[str, object], train_cfg
):
    keep_per_class = int(train_cfg.distillation.filter_keep_per_class)
    if keep_per_class <= 0:
        raise ValueError(
            f"distillation.filter_keep_per_class must be positive, got {keep_per_class}"
        )
    trim_fraction = float(train_cfg.distillation.filter_foreground_trim_fraction)
    contrast_drop_fraction = float(train_cfg.distillation.filter_contrast_drop_fraction)
    if trim_fraction < 0.0 or trim_fraction >= 0.5:
        raise ValueError(
            "distillation.filter_foreground_trim_fraction must be in [0.0, 0.5)."
        )
    if contrast_drop_fraction < 0.0 or contrast_drop_fraction >= 1.0:
        raise ValueError(
            "distillation.filter_contrast_drop_fraction must be in [0.0, 1.0)."
        )

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
        class_indices = torch_module.nonzero(
            labels_cpu == int(label), as_tuple=False
        ).flatten()
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

        candidate_sat = metric_tensors["saturation_ratio"].index_select(
            0, candidate_indices
        )
        candidate_tv = metric_tensors["mask_tv"].index_select(0, candidate_indices)
        candidate_contrast = metric_tensors["contrast_std"].index_select(
            0, candidate_indices
        )
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
            "rejected_by_foreground_trim": int(
                ((fg < fg_low) | (fg > fg_high)).sum().item()
            ),
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

def run_teacher_distill_label_strict_filter(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
    device=None,
    prototype_builder=None,
) -> None:
    payload = load_distill_payload(torch_module, distill_dataset)
    (
        codes,
        labels,
        teacher_camera,
        prototypes,
        audit_tensors,
        audit_payload,
        class_labels,
    ) = _resolve_teacher_label_audit_inputs(
        torch_module,
        payload,
        train_cfg,
        device=device,
        prototype_builder=prototype_builder,
    )
    selected_indices, selection_summary = _select_teacher_distill_label_strict_indices(
        torch_module,
        labels,
        audit_tensors,
        train_cfg,
    )
    filtered_payload, index_tensor = filter_distill_payload_by_indices(
        torch_module,
        payload,
        selected_indices,
        source_size=int(codes.shape[0]),
    )
    source_config = payload.get("config", {})
    filtered_config = (
        dict(source_config)
        if isinstance(source_config, dict)
        else {"source_config": source_config}
    )
    filtered_config["filter"] = {
        "teacher_filter": "label_strict",
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "selected_count": len(selected_indices),
        "keep_per_class": int(train_cfg.distillation.label_strict_keep_per_class),
        "foreground_threshold": train_cfg.distillation.filter_foreground_threshold,
        "prototype_source": "train_mnist_mean_by_label",
        "prototypes_per_class": int(
            train_cfg.distillation.label_audit_prototypes_per_class
        ),
    }
    filtered_payload["config"] = filtered_config
    filtered_payload["filter_indices"] = index_tensor
    filtered_payload["label_audit_requested_corr"] = audit_tensors[
        "requested_label_corr"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_nearest_label"] = audit_tensors[
        "nearest_label"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_margin"] = audit_tensors["label_margin"].index_select(
        0, index_tensor
    )

    class_counts = {
        str(label): int(
            (filtered_payload["class_labels"].long() == int(label)).sum().item()
        )
        for label in sorted(
            {int(label) for label in filtered_payload["class_labels"].long().tolist()}
        )
    }
    metrics_payload = {
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "selected_count": len(selected_indices),
        "class_counts": class_counts,
        "audit": {
            key: value for key, value in audit_payload.items() if key != "per_sample"
        },
        **selection_summary,
    }
    run_manager.save_config_snapshot(
        "teacher_distill_label_strict_filter_config",
        {
            "distill_dataset": distill_dataset,
            "audit_device": str(device) if device is not None else "cpu",
            "filter": filtered_config["filter"],
            "source": "train.py_teacher_distill_label_strict_filter",
        },
    )
    artifact_path = run_manager.save_tensor_bundle(
        "teacher_distill_dataset_label_strict.pt", filtered_payload
    )
    metrics_path = run_manager.save_artifact_json(
        "teacher_distill_label_strict_filter_metrics", metrics_payload
    )
    audit_path, audit_image_path = _save_teacher_label_audit_artifacts(
        torch_module,
        run_manager=run_manager,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        labels=labels,
        prototypes=prototypes,
        audit_tensors=audit_tensors,
        audit_payload=audit_payload,
        class_labels=class_labels,
        artifact_name="teacher_distill_label_audit",
        image_name="teacher_distill_label_audit.png",
    )
    source_preview = _balanced_preview_indices(torch_module, labels)
    filtered_preview = _balanced_preview_indices(
        torch_module, filtered_payload["class_labels"]
    )
    run_manager.save_tensor_grid(
        {
            "source teacher readable": readable_camera_for_display(
                teacher_camera.index_select(0, source_preview)
            ),
            "strict teacher readable": readable_camera_for_display(
                filtered_payload["teacher_camera"].index_select(0, filtered_preview)
            ),
        },
        run_manager.images_dir / "teacher_distill_label_strict_before_after.png",
        title=format_output_title(
            "images/teacher_distill_label_strict_before_after.png",
            "teacher label-strict distill filter",
        ),
        max_items=min(20, len(selected_indices)),
    )
    print(
        "[teacher_distill_label_strict_filter] "
        f"selected {len(selected_indices)}/{int(codes.shape[0])} samples "
        f"to {artifact_path}"
    )
    run_meta.update(
        {
            "teacher_distill_source_dataset": distill_dataset,
            "teacher_distill_filtered_dataset": str(artifact_path),
            "teacher_distill_filter": "label_strict",
            "teacher_distill_filter_metric_source": str(
                metrics_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_label_audit_metric_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_label_audit_image": str(
                audit_image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_label_audit_device": str(device)
            if device is not None
            else "cpu",
            "teacher_distill_filter_sample_count": len(selected_indices),
            "teacher_distill_filter_class_counts": class_counts,
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
    payload = load_distill_payload(torch_module, distill_dataset)
    codes, labels, teacher_camera = validated_distill_tensors(
        torch_module, payload, train_cfg
    )
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
    filtered_payload, index_tensor = filter_distill_payload_by_indices(
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
    filtered_config = (
        dict(source_config)
        if isinstance(source_config, dict)
        else {"source_config": source_config}
    )
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
        key: _metric_summary(value) for key, value in metric_tensors.items()
    }
    after_summary = {
        key: _metric_summary(value) for key, value in filtered_metrics.items()
    }
    class_counts = {
        str(label): int(
            (filtered_payload["class_labels"].long() == int(label)).sum().item()
        )
        for label in sorted(
            {int(label) for label in filtered_payload["class_labels"].long().tolist()}
        )
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
    artifact_path = run_manager.save_tensor_bundle(
        "teacher_distill_dataset_filtered.pt", filtered_payload
    )
    metrics_path = run_manager.save_artifact_json(
        "teacher_distill_filter_metrics", metrics_payload
    )
    source_preview = _balanced_preview_indices(torch_module, labels)
    filtered_preview = _balanced_preview_indices(
        torch_module, filtered_payload["class_labels"]
    )
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
        title=format_output_title(
            "images/teacher_distill_filter_before_after.png", "teacher distill filter"
        ),
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
            "teacher_distill_filter_metric_source": str(
                metrics_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_filter_sample_count": len(selected_indices),
            "teacher_distill_filter_class_counts": class_counts,
        }
    )
    run_manager.write_meta(run_meta)

__all__ = ["compute_teacher_distill_filter_metrics","run_teacher_distill_filter","run_teacher_distill_label_strict_filter"]
