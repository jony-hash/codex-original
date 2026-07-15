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

from .distill_support import (
    _balanced_preview_indices,
    _build_label_prototype_bank_from_tensors,
    _build_mnist_label_prototype_bank,
    _masked_ratio,
    _metric_summary,
    _prototype_correlation,
)

def compute_teacher_distill_label_audit(
    torch_module,
    teacher_camera,
    labels,
    prototypes,
    *,
    class_labels,
    foreground_threshold: float = 0.2,
    speckle_threshold: float = 0.15,
    hole_threshold: float = 0.1,
):
    label_to_index = {int(label): index for index, label in enumerate(class_labels)}
    missing = sorted(
        {
            int(label)
            for label in labels.detach().cpu().tolist()
            if int(label) not in label_to_index
        }
    )
    if missing:
        raise ValueError(f"Audit labels are not present in prototype bank: {missing}")

    readable = readable_camera_for_display(teacher_camera).detach().float()
    if readable.ndim == 3:
        readable = readable.unsqueeze(0)
    prototypes = prototypes.detach().float().to(device=readable.device)
    labels_cpu = labels.detach().long().cpu()
    corr = _prototype_correlation(torch_module, readable, prototypes)
    requested_indices = torch_module.tensor(
        [label_to_index[int(label)] for label in labels_cpu.tolist()],
        device=readable.device,
        dtype=torch_module.long,
    )
    requested_corr = corr.gather(1, requested_indices[:, None]).squeeze(1)
    nearest_corr, nearest_indices = corr.max(dim=1)
    nearest_labels = torch_module.tensor(
        [
            int(class_labels[int(index)])
            for index in nearest_indices.detach().cpu().tolist()
        ],
        device=readable.device,
        dtype=torch_module.long,
    )
    other_corr = corr.clone()
    other_corr[
        torch_module.arange(other_corr.shape[0], device=readable.device),
        requested_indices,
    ] = -2.0
    best_other_corr = other_corr.max(dim=1).values
    label_margin = requested_corr - best_other_corr

    requested_prototypes = prototypes.index_select(0, requested_indices)
    foreground_mask = requested_prototypes > float(foreground_threshold)
    background_mask = ~foreground_mask
    foreground_ratio = (
        (readable > float(foreground_threshold)).float().flatten(1).mean(dim=1)
    )
    background_speckle_ratio = _masked_ratio(
        (readable > float(speckle_threshold)) & background_mask, background_mask
    )
    foreground_hole_ratio = _masked_ratio(
        (readable < float(hole_threshold)) & foreground_mask, foreground_mask
    )
    fg_mean = _masked_ratio(readable * foreground_mask.float(), foreground_mask)
    bg_mean = _masked_ratio(readable * background_mask.float(), background_mask)
    readable_contrast = (fg_mean - bg_mean) / (fg_mean + bg_mean + 1.0e-8)

    return {
        "teacher_readable": readable.detach().cpu(),
        "prototype_corr": corr.detach().cpu(),
        "requested_label_corr": requested_corr.detach().cpu(),
        "nearest_label": nearest_labels.detach().cpu(),
        "nearest_label_corr": nearest_corr.detach().cpu(),
        "best_other_corr": best_other_corr.detach().cpu(),
        "label_margin": label_margin.detach().cpu(),
        "label_correct": (nearest_labels.detach().cpu() == labels_cpu),
        "foreground_ratio": foreground_ratio.detach().cpu(),
        "background_speckle_ratio": background_speckle_ratio.detach().cpu(),
        "readable_contrast": readable_contrast.detach().cpu(),
        "foreground_hole_ratio": foreground_hole_ratio.detach().cpu(),
    }

def compute_teacher_distill_label_audit_batched(
    torch_module,
    teacher_camera,
    labels,
    prototypes,
    *,
    class_labels,
    batch_size: int,
    device,
    foreground_threshold: float = 0.2,
):
    if batch_size <= 0:
        raise ValueError(f"label audit batch_size must be positive, got {batch_size}")
    audit_parts: dict[str, list[object]] = {}
    with torch_module.no_grad():
        for start in range(0, int(labels.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(labels.shape[0]))
            batch_audit = compute_teacher_distill_label_audit(
                torch_module,
                teacher_camera[start:end].to(device=device),
                labels[start:end].to(device=device),
                prototypes.to(device=device),
                class_labels=class_labels,
                foreground_threshold=foreground_threshold,
            )
            for key, value in batch_audit.items():
                audit_parts.setdefault(key, []).append(value)
    return {key: torch_module.cat(values, dim=0) for key, values in audit_parts.items()}

def _teacher_label_audit_json(
    labels, audit_tensors, *, class_labels
) -> dict[str, object]:
    labels_cpu = labels.detach().long().cpu()
    per_sample = []
    for index, requested in enumerate(labels_cpu.tolist()):
        per_sample.append(
            {
                "index": index,
                "requested_label": int(requested),
                "nearest_label": int(audit_tensors["nearest_label"][index].item()),
                "requested_label_corr": float(
                    audit_tensors["requested_label_corr"][index].item()
                ),
                "nearest_label_corr": float(
                    audit_tensors["nearest_label_corr"][index].item()
                ),
                "best_other_corr": float(
                    audit_tensors["best_other_corr"][index].item()
                ),
                "label_margin": float(audit_tensors["label_margin"][index].item()),
                "label_correct": bool(audit_tensors["label_correct"][index].item()),
                "foreground_ratio": float(
                    audit_tensors["foreground_ratio"][index].item()
                ),
                "background_speckle_ratio": float(
                    audit_tensors["background_speckle_ratio"][index].item()
                ),
                "readable_contrast": float(
                    audit_tensors["readable_contrast"][index].item()
                ),
                "foreground_hole_ratio": float(
                    audit_tensors["foreground_hole_ratio"][index].item()
                ),
            }
        )
    per_class = {}
    for label in class_labels:
        indices = [
            index
            for index, item in enumerate(labels_cpu.tolist())
            if int(item) == int(label)
        ]
        if not indices:
            continue
        index_tensor = labels_cpu.new_tensor(indices)
        correct = audit_tensors["label_correct"].index_select(0, index_tensor).float()
        margins = audit_tensors["label_margin"].index_select(0, index_tensor)
        per_class[str(label)] = {
            "count": len(indices),
            "label_correct_count": int(correct.sum().item()),
            "label_correct_ratio": float(correct.mean().item()),
            "label_margin": _metric_summary(margins),
            "requested_label_corr": _metric_summary(
                audit_tensors["requested_label_corr"].index_select(0, index_tensor)
            ),
            "background_speckle_ratio": _metric_summary(
                audit_tensors["background_speckle_ratio"].index_select(0, index_tensor)
            ),
            "readable_contrast": _metric_summary(
                audit_tensors["readable_contrast"].index_select(0, index_tensor)
            ),
            "foreground_hole_ratio": _metric_summary(
                audit_tensors["foreground_hole_ratio"].index_select(0, index_tensor)
            ),
        }
    correct_all = audit_tensors["label_correct"].float()
    return {
        "class_labels": [int(label) for label in class_labels],
        "sample_count": int(labels_cpu.numel()),
        "label_correct_count": int(correct_all.sum().item()),
        "label_correct_ratio": float(correct_all.mean().item())
        if int(labels_cpu.numel())
        else 0.0,
        "overall": {
            "label_margin": _metric_summary(audit_tensors["label_margin"]),
            "requested_label_corr": _metric_summary(
                audit_tensors["requested_label_corr"]
            ),
            "background_speckle_ratio": _metric_summary(
                audit_tensors["background_speckle_ratio"]
            ),
            "readable_contrast": _metric_summary(audit_tensors["readable_contrast"]),
            "foreground_hole_ratio": _metric_summary(
                audit_tensors["foreground_hole_ratio"]
            ),
        },
        "per_class": per_class,
        "per_sample": per_sample,
    }

def _teacher_label_audit_visual_tensors(
    torch_module,
    labels,
    prototypes,
    audit_tensors,
    *,
    class_labels,
    max_items: int = 20,
):
    labels_cpu = labels.detach().long().cpu()
    preview = _balanced_preview_indices(torch_module, labels_cpu, max_total=max_items)
    label_to_index = {int(label): index for index, label in enumerate(class_labels)}
    requested_indices = torch_module.tensor(
        [label_to_index[int(labels_cpu[index].item())] for index in preview],
        dtype=torch_module.long,
    )
    nearest_indices = torch_module.tensor(
        [
            label_to_index[int(audit_tensors["nearest_label"][index].item())]
            for index in preview
        ],
        dtype=torch_module.long,
    )
    margins = (
        audit_tensors["label_margin"].detach().float().cpu().index_select(0, preview)
    )
    margin_tiles = margins.reshape(-1, 1, 1, 1).expand(
        -1, 1, prototypes.shape[-2], prototypes.shape[-1]
    )
    return (
        {
            "mnist exemplar": prototypes.index_select(0, requested_indices),
            "teacher readable": audit_tensors["teacher_readable"].index_select(
                0, preview
            ),
            "requested label": prototypes.index_select(0, requested_indices),
            "nearest label": prototypes.index_select(0, nearest_indices),
            "label margin": _normalize_for_visualization(margin_tiles),
        },
        [
            f"r{int(labels_cpu[index].item())} n{int(audit_tensors['nearest_label'][index].item())} "
            f"m={float(audit_tensors['label_margin'][index].item()):.2f}"
            for index in preview.tolist()
        ],
    )

def _resolve_teacher_label_audit_inputs(
    torch_module, payload, train_cfg, *, device=None, prototype_builder=None
):
    codes, labels, teacher_camera = validated_distill_tensors(
        torch_module, payload, train_cfg
    )
    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.one_shot.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(
            sorted({int(label) for label in labels.detach().cpu().tolist()})
        )
    builder = prototype_builder or _build_mnist_label_prototype_bank
    prototypes = builder(
        torch_module,
        class_labels=class_labels,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    )
    audit_device = device if device is not None else teacher_camera.device
    audit_batch_size = (
        train_cfg.distillation.label_audit_batch_size or train_cfg.batch_size
    )
    audit_tensors = compute_teacher_distill_label_audit_batched(
        torch_module,
        teacher_camera,
        labels,
        prototypes,
        class_labels=class_labels,
        batch_size=int(audit_batch_size),
        device=audit_device,
        foreground_threshold=train_cfg.distillation.filter_foreground_threshold,
    )
    audit_payload = _teacher_label_audit_json(
        labels, audit_tensors, class_labels=class_labels
    )
    return (
        codes,
        labels,
        teacher_camera,
        prototypes,
        audit_tensors,
        audit_payload,
        class_labels,
    )

def _save_teacher_label_audit_artifacts(
    torch_module,
    *,
    run_manager,
    distill_dataset: str,
    train_cfg,
    labels,
    prototypes,
    audit_tensors,
    audit_payload,
    class_labels,
    artifact_name: str = "teacher_distill_label_audit",
    image_name: str = "teacher_distill_label_audit.png",
):
    audit_path = run_manager.save_artifact_json(
        artifact_name,
        {
            "distill_dataset": distill_dataset,
            "prototype_source": "train_mnist_mean_by_label",
            "prototypes_per_class": int(
                train_cfg.distillation.label_audit_prototypes_per_class
            ),
            **audit_payload,
        },
    )
    visual_tensors, row_labels = _teacher_label_audit_visual_tensors(
        torch_module,
        labels,
        prototypes,
        audit_tensors,
        class_labels=class_labels,
    )
    image_path = run_manager.save_tensor_grid(
        visual_tensors,
        run_manager.images_dir / image_name,
        title=format_output_title(
            f"images/{image_name}", "teacher distill label audit"
        ),
        max_items=min(20, int(labels.shape[0])),
        row_labels=row_labels,
    )
    return audit_path, image_path

def run_teacher_distill_label_audit(
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
    _, labels, _, prototypes, audit_tensors, audit_payload, class_labels = (
        _resolve_teacher_label_audit_inputs(
            torch_module,
            payload,
            train_cfg,
            device=device,
            prototype_builder=prototype_builder,
        )
    )
    run_manager.save_config_snapshot(
        "teacher_distill_label_audit_config",
        {
            "distill_dataset": distill_dataset,
            "audit_device": str(device) if device is not None else "cpu",
            "prototype_source": "train_mnist_mean_by_label",
            "prototypes_per_class": train_cfg.distillation.label_audit_prototypes_per_class,
            "foreground_threshold": train_cfg.distillation.filter_foreground_threshold,
            "source": "train.py_teacher_distill_label_audit",
        },
    )
    audit_path, image_path = _save_teacher_label_audit_artifacts(
        torch_module,
        run_manager=run_manager,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        labels=labels,
        prototypes=prototypes,
        audit_tensors=audit_tensors,
        audit_payload=audit_payload,
        class_labels=class_labels,
    )
    print(
        "[teacher_distill_label_audit] "
        f"label_correct={audit_payload['label_correct_count']}/{audit_payload['sample_count']} "
        f"artifact={audit_path}"
    )
    run_meta.update(
        {
            "teacher_distill_label_audit_dataset": distill_dataset,
            "teacher_distill_label_audit_metric_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_label_audit_image": str(
                image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_label_audit_device": str(device)
            if device is not None
            else "cpu",
            "teacher_distill_label_correct_count": audit_payload["label_correct_count"],
            "teacher_distill_label_correct_ratio": audit_payload["label_correct_ratio"],
        }
    )
    run_manager.write_meta(run_meta)

__all__ = ["compute_teacher_distill_label_audit","compute_teacher_distill_label_audit_batched","run_teacher_distill_label_audit"]
