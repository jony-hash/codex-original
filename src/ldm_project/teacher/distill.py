from __future__ import annotations

import math
from pathlib import Path

from ..distillation.dataset import (
    _filter_distill_payload_by_indices,
    _load_distill_payload,
    _validated_distill_tensors,
)
from ..generation.codes import _sample_one_shot_codes_with_seed
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
    labels = [
        label
        for label in class_labels
        for _ in range(train_cfg.distillation.samples_per_class)
    ]
    if train_cfg.distillation.total_samples is not None:
        if not labels:
            raise ValueError(
                "distillation.class_labels must not be empty when total_samples is set."
            )
        repeated = []
        while len(repeated) < train_cfg.distillation.total_samples:
            repeated.extend(class_labels)
        labels = repeated[: train_cfg.distillation.total_samples]
    if not labels:
        raise ValueError("Teacher distillation export resolved to zero labels.")
    return torch_module.tensor(labels, device=device, dtype=torch_module.long)


def _project_codes_to_teacher_noise(
    torch_module,
    codes,
    sample_shape: tuple[int, int, int, int],
    projection_seed: int,
    device,
):
    if codes.ndim != 2:
        raise ValueError(
            f"codes must be [batch, code_dim], got shape {tuple(codes.shape)}"
        )
    flat_size = int(math.prod(sample_shape[1:]))
    generator = torch_module.Generator().manual_seed(int(projection_seed))
    projection = torch_module.randn(
        codes.shape[1], flat_size, generator=generator, dtype=codes.dtype
    )
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


def _build_label_prototype_bank_from_tensors(
    torch_module, images, labels, *, class_labels, image_size: int
):
    if images.ndim != 4:
        raise ValueError(
            f"prototype images must be [N, C, H, W], got {tuple(images.shape)}"
        )
    if int(images.shape[-1]) != image_size or int(images.shape[-2]) != image_size:
        images = torch_module.nn.functional.interpolate(
            images.detach().float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    labels_cpu = labels.detach().long().cpu()
    prototypes = []
    for label in class_labels:
        indices = torch_module.nonzero(
            labels_cpu == int(label), as_tuple=False
        ).flatten()
        if int(indices.numel()) == 0:
            raise ValueError(f"No prototype samples found for label {label}.")
        prototype = (
            images.detach()
            .float()
            .cpu()
            .index_select(0, indices)
            .mean(dim=0, keepdim=True)
        )
        prototypes.append(_normalize_for_visualization(prototype).squeeze(0))
    return torch_module.stack(prototypes, dim=0)


def _build_mnist_label_prototype_bank(
    torch_module, *, class_labels, image_size: int, root: str, max_per_class: int
):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to build MNIST label prototypes."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    mnist = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    requested = tuple(int(label) for label in class_labels)
    buckets = {label: [] for label in requested}
    for image, label in mnist:
        label = int(label)
        if label not in buckets or len(buckets[label]) >= int(max_per_class):
            continue
        buckets[label].append(image)
        if all(len(items) >= int(max_per_class) for items in buckets.values()):
            break
    missing = [label for label, items in buckets.items() if not items]
    if missing:
        raise ValueError(f"MNIST prototype bank is missing labels: {missing}")
    prototype_images = []
    prototype_labels = []
    for label in requested:
        prototype_images.extend(buckets[label])
        prototype_labels.extend([label] * len(buckets[label]))
    return _build_label_prototype_bank_from_tensors(
        torch_module,
        torch_module.stack(prototype_images, dim=0),
        torch_module.tensor(prototype_labels, dtype=torch_module.long),
        class_labels=requested,
        image_size=image_size,
    )


def _prototype_correlation(torch_module, samples, prototypes):
    sample_flat = samples.detach().float().flatten(1)
    prototype_flat = prototypes.detach().float().flatten(1)
    sample_centered = sample_flat - sample_flat.mean(dim=1, keepdim=True)
    prototype_centered = prototype_flat - prototype_flat.mean(dim=1, keepdim=True)
    sample_norm = (
        sample_centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    )
    prototype_norm = (
        prototype_centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    )
    return (
        (sample_centered / sample_norm)
        @ (prototype_centered / prototype_norm).T
        / float(sample_flat.shape[1])
    )


def _masked_ratio(values, mask):
    return values.float().sum(dim=(1, 2, 3)) / mask.float().sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)


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


def _readable_gate_thresholds(train_cfg) -> dict[str, float]:
    return {
        "min_label_margin": float(train_cfg.distillation.readable_min_label_margin),
        "min_readable_contrast": float(train_cfg.distillation.readable_min_contrast),
        "max_background_speckle_ratio": float(
            train_cfg.distillation.readable_max_background_speckle
        ),
        "max_foreground_hole_ratio": float(
            train_cfg.distillation.readable_max_foreground_hole
        ),
    }


def _readable_candidate_score(audit_tensors, indices):
    margin = audit_tensors["label_margin"].index_select(0, indices)
    contrast = audit_tensors["readable_contrast"].index_select(0, indices)
    speckle = audit_tensors["background_speckle_ratio"].index_select(0, indices)
    holes = audit_tensors["foreground_hole_ratio"].index_select(0, indices)
    return margin + 0.25 * contrast - 0.25 * speckle - 0.25 * holes


def _split_readable_teacher_indices(
    torch_module, labels, audit_tensors: dict[str, object], train_cfg
):
    train_per_class = int(train_cfg.distillation.readable_train_per_class)
    val_per_class = int(train_cfg.distillation.readable_val_per_class)
    test_per_class = int(train_cfg.distillation.readable_test_per_class)
    required_per_class = train_per_class + val_per_class + test_per_class
    if (
        train_per_class < 0
        or val_per_class < 0
        or test_per_class < 0
        or required_per_class <= 0
    ):
        raise ValueError(
            "readable split counts must be non-negative and sum to a positive value."
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

    thresholds = _readable_gate_thresholds(train_cfg)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    selected_indices: list[int] = []
    per_class: dict[str, dict[str, object]] = {}
    failed = False
    generator = torch_module.Generator().manual_seed(
        int(train_cfg.distillation.readable_split_seed)
    )

    for label in class_labels:
        class_indices = torch_module.nonzero(
            labels_cpu == int(label), as_tuple=False
        ).flatten()
        class_count = int(class_indices.numel())
        if class_count == 0:
            failed = True
            per_class[str(label)] = {
                "available": 0,
                "readable_available": 0,
                "readability_filter_failed": True,
            }
            continue

        label_correct = audit_tensors["label_correct"].index_select(0, class_indices)
        margin = audit_tensors["label_margin"].index_select(0, class_indices)
        contrast = audit_tensors["readable_contrast"].index_select(0, class_indices)
        speckle = audit_tensors["background_speckle_ratio"].index_select(
            0, class_indices
        )
        holes = audit_tensors["foreground_hole_ratio"].index_select(0, class_indices)
        readable_mask = (
            label_correct
            & (margin >= thresholds["min_label_margin"])
            & (contrast >= thresholds["min_readable_contrast"])
            & (speckle <= thresholds["max_background_speckle_ratio"])
            & (holes <= thresholds["max_foreground_hole_ratio"])
        )
        readable_indices = class_indices[readable_mask]
        readable_count = int(readable_indices.numel())
        class_failed = readable_count < required_per_class
        failed = failed or class_failed

        chosen = readable_indices
        split_counts = {"train": 0, "val": 0, "test": 0}
        if not class_failed:
            score = _readable_candidate_score(audit_tensors, readable_indices)
            order = torch_module.argsort(score, descending=True)
            chosen = readable_indices.index_select(0, order[:required_per_class])
            permutation = torch_module.randperm(required_per_class, generator=generator)
            chosen = chosen.index_select(0, permutation)
            train_part = chosen[:train_per_class]
            val_part = chosen[train_per_class : train_per_class + val_per_class]
            test_part = chosen[train_per_class + val_per_class :]
            train_indices.extend(int(index) for index in train_part.tolist())
            val_indices.extend(int(index) for index in val_part.tolist())
            test_indices.extend(int(index) for index in test_part.tolist())
            selected_indices.extend(int(index) for index in chosen.tolist())
            split_counts = {
                "train": int(train_part.numel()),
                "val": int(val_part.numel()),
                "test": int(test_part.numel()),
            }

        per_class[str(label)] = {
            "available": class_count,
            "label_correct_available": int(label_correct.sum().item()),
            "readable_available": readable_count,
            "required": required_per_class,
            "readability_filter_failed": class_failed,
            "split_counts": split_counts,
            "rejected_label_incorrect": int((~label_correct).sum().item()),
            "rejected_low_margin": int(
                (label_correct & (margin < thresholds["min_label_margin"])).sum().item()
            ),
            "rejected_low_contrast": int(
                (label_correct & (contrast < thresholds["min_readable_contrast"]))
                .sum()
                .item()
            ),
            "rejected_high_speckle": int(
                (label_correct & (speckle > thresholds["max_background_speckle_ratio"]))
                .sum()
                .item()
            ),
            "rejected_high_hole": int(
                (label_correct & (holes > thresholds["max_foreground_hole_ratio"]))
                .sum()
                .item()
            ),
            "readable_label_margin": _metric_summary(
                audit_tensors["label_margin"].index_select(0, readable_indices)
            ),
            "readable_contrast": _metric_summary(
                audit_tensors["readable_contrast"].index_select(0, readable_indices)
            ),
            "readable_background_speckle_ratio": _metric_summary(
                audit_tensors["background_speckle_ratio"].index_select(
                    0, readable_indices
                )
            ),
            "readable_foreground_hole_ratio": _metric_summary(
                audit_tensors["foreground_hole_ratio"].index_select(0, readable_indices)
            ),
        }

    if failed:
        train_indices = []
        val_indices = []
        test_indices = []
        selected_indices = []

    return {
        "failed": failed,
        "thresholds": thresholds,
        "class_labels": list(class_labels),
        "split_seed": int(train_cfg.distillation.readable_split_seed),
        "required_per_class": required_per_class,
        "split_counts_per_class": {
            "train": train_per_class,
            "val": val_per_class,
            "test": test_per_class,
        },
        "indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "selected": selected_indices,
        },
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


def _balanced_preview_indices(
    torch_module, labels, *, max_per_class: int = 2, max_total: int = 20
):
    labels_cpu = labels.detach().long().cpu()
    preview: list[int] = []
    for label in sorted({int(item) for item in labels_cpu.tolist()}):
        class_indices = (
            torch_module.nonzero(labels_cpu == label, as_tuple=False).flatten().tolist()
        )
        preview.extend(int(index) for index in class_indices[:max_per_class])
        if len(preview) >= max_total:
            break
    return torch_module.tensor(preview[:max_total], dtype=torch_module.long)


def _resolve_teacher_label_audit_inputs(
    torch_module, payload, train_cfg, *, device=None
):
    codes, labels, teacher_camera = _validated_distill_tensors(
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
    prototypes = _build_mnist_label_prototype_bank(
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
    set_first_stage_image_fusion_weight(
        first_stage, train_cfg.first_stage.image_fusion_end_weight
    )
    diffusion_payload = torch_module.load(
        diffusion_checkpoint, map_location=device, weights_only=False
    )
    load_diffusion_checkpoint_fn(
        diffusion_model, diffusion_payload, first_stage_model_version
    )
    latent_scale = load_latent_scale_from_diffusion_checkpoint(
        torch_module, diffusion_payload, device
    )
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
            outputs = first_stage.decode_with_intermediates(
                batch_latents * latent_scale
            )
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
        artifact_path = run_manager.save_tensor_bundle(
            "teacher_distill_dataset.pt", payload
        )
    run_manager.save_tensor_grid(
        {
            "teacher_camera_raw": payload["teacher_camera"],
            "teacher_camera_display": enhance_camera_for_display(
                payload["teacher_camera"]
            ),
            "teacher_image": payload["teacher_image"],
        },
        run_manager.images_dir / "teacher_distill_export.png",
        title=format_output_title(
            "images/teacher_distill_export.png", "teacher distillation targets"
        ),
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


def run_teacher_distill_label_audit(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
    device=None,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
    _, labels, _, prototypes, audit_tensors, audit_payload, class_labels = (
        _resolve_teacher_label_audit_inputs(
            torch_module,
            payload,
            train_cfg,
            device=device,
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


def run_teacher_distill_label_strict_filter(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
    device=None,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
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
    )
    selected_indices, selection_summary = _select_teacher_distill_label_strict_indices(
        torch_module,
        labels,
        audit_tensors,
        train_cfg,
    )
    filtered_payload, index_tensor = _filter_distill_payload_by_indices(
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


def _add_readability_audit_fields(
    filtered_payload, audit_tensors, index_tensor
) -> None:
    filtered_payload["label_audit_requested_corr"] = audit_tensors[
        "requested_label_corr"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_nearest_label"] = audit_tensors[
        "nearest_label"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_margin"] = audit_tensors["label_margin"].index_select(
        0, index_tensor
    )
    filtered_payload["readability_audit_contrast"] = audit_tensors[
        "readable_contrast"
    ].index_select(0, index_tensor)
    filtered_payload["readability_audit_background_speckle_ratio"] = audit_tensors[
        "background_speckle_ratio"
    ].index_select(0, index_tensor)
    filtered_payload["readability_audit_foreground_hole_ratio"] = audit_tensors[
        "foreground_hole_ratio"
    ].index_select(
        0,
        index_tensor,
    )


def _readable_split_class_counts(
    torch_module, payload: dict[str, object]
) -> dict[str, int]:
    labels = payload["class_labels"].detach().long().cpu()
    return {
        str(label): int((labels == int(label)).sum().item())
        for label in sorted({int(item) for item in labels.tolist()})
    }


def run_teacher_distill_readable_split(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
    device=None,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
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
    )
    split_summary = _split_readable_teacher_indices(
        torch_module, labels, audit_tensors, train_cfg
    )
    metrics_payload = {
        "teacher_filter": "readability_strict_split",
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "readability_filter_failed": bool(split_summary["failed"]),
        "audit": {
            key: value for key, value in audit_payload.items() if key != "per_sample"
        },
        **{key: value for key, value in split_summary.items() if key != "indices"},
    }
    run_manager.save_config_snapshot(
        "teacher_distill_readable_split_config",
        {
            "distill_dataset": distill_dataset,
            "audit_device": str(device) if device is not None else "cpu",
            "thresholds": split_summary["thresholds"],
            "split_counts_per_class": split_summary["split_counts_per_class"],
            "split_seed": split_summary["split_seed"],
            "source": "train.py_teacher_distill_readable_split",
        },
    )
    metrics_path = run_manager.save_artifact_json(
        "teacher_distill_readable_split_metrics", metrics_payload
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
        artifact_name="teacher_distill_readable_split_audit",
        image_name="teacher_distill_readable_split_audit.png",
    )

    split_paths: dict[str, str | None] = {"train": None, "val": None, "test": None}
    split_class_counts: dict[str, dict[str, int]] = {}
    if not split_summary["failed"]:
        source_config = payload.get("config", {})
        for split_name in ("train", "val", "test"):
            split_indices = split_summary["indices"][split_name]
            filtered_payload, index_tensor = _filter_distill_payload_by_indices(
                torch_module,
                payload,
                split_indices,
                source_size=int(codes.shape[0]),
            )
            filtered_config = (
                dict(source_config)
                if isinstance(source_config, dict)
                else {"source_config": source_config}
            )
            filtered_config["filter"] = {
                "teacher_filter": "readability_strict_split",
                "split": split_name,
                "source_dataset": distill_dataset,
                "source_count": int(codes.shape[0]),
                "selected_count": len(split_indices),
                "thresholds": split_summary["thresholds"],
                "split_counts_per_class": split_summary["split_counts_per_class"],
                "split_seed": split_summary["split_seed"],
            }
            filtered_payload["config"] = filtered_config
            filtered_payload["readability_split"] = split_name
            filtered_payload["readability_filter_indices"] = index_tensor
            _add_readability_audit_fields(filtered_payload, audit_tensors, index_tensor)
            artifact_path = run_manager.save_tensor_bundle(
                f"teacher_distill_dataset_readable_{split_name}.pt",
                filtered_payload,
            )
            split_paths[split_name] = str(artifact_path)
            split_class_counts[split_name] = _readable_split_class_counts(
                torch_module, filtered_payload
            )

        source_preview = _balanced_preview_indices(torch_module, labels)
        selected_index_tensor = torch_module.tensor(
            split_summary["indices"]["selected"], dtype=torch_module.long
        )
        selected_preview = _balanced_preview_indices(
            torch_module, labels.index_select(0, selected_index_tensor)
        )
        selected_preview = selected_index_tensor.index_select(0, selected_preview)
        run_manager.save_tensor_grid(
            {
                "source teacher readable": readable_camera_for_display(
                    teacher_camera.index_select(0, source_preview)
                ),
                "readable selected": readable_camera_for_display(
                    teacher_camera.index_select(0, selected_preview)
                ),
            },
            run_manager.images_dir / "teacher_distill_readable_split_before_after.png",
            title=format_output_title(
                "images/teacher_distill_readable_split_before_after.png",
                "teacher readable split filter",
            ),
            max_items=min(20, len(split_summary["indices"]["selected"])),
        )

    print(
        "[teacher_distill_readable_split] "
        f"failed={bool(split_summary['failed'])} "
        f"train={len(split_summary['indices']['train'])} "
        f"val={len(split_summary['indices']['val'])} "
        f"test={len(split_summary['indices']['test'])}"
    )
    run_meta.update(
        {
            "teacher_distill_source_dataset": distill_dataset,
            "teacher_distill_filter": "readability_strict_split",
            "teacher_distill_readability_filter_failed": bool(split_summary["failed"]),
            "teacher_distill_readable_split_metric_source": str(
                metrics_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_split_audit_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_split_audit_image": str(
                audit_image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_train_dataset": split_paths["train"],
            "teacher_distill_readable_val_dataset": split_paths["val"],
            "teacher_distill_readable_test_dataset": split_paths["test"],
            "teacher_distill_readable_split_class_counts": split_class_counts,
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
    codes, labels, teacher_camera = _validated_distill_tensors(
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
