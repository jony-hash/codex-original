"""Label/readability audit shared by teacher target and student camera outputs."""

from __future__ import annotations

from ..visualization_core import readable_camera_for_display


def _metric_summary(values) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


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


def compute_label_audit(
    torch_module,
    camera,
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
    readable = readable_camera_for_display(camera).detach().float()
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
    return {
        "teacher_readable": readable.detach().cpu(),
        "prototype_corr": corr.detach().cpu(),
        "requested_label_corr": requested_corr.detach().cpu(),
        "nearest_label": nearest_labels.detach().cpu(),
        "nearest_label_corr": nearest_corr.detach().cpu(),
        "best_other_corr": best_other_corr.detach().cpu(),
        "label_margin": (requested_corr - best_other_corr).detach().cpu(),
        "label_correct": nearest_labels.detach().cpu() == labels_cpu,
        "foreground_ratio": foreground_ratio.detach().cpu(),
        "background_speckle_ratio": background_speckle_ratio.detach().cpu(),
        "readable_contrast": ((fg_mean - bg_mean) / (fg_mean + bg_mean + 1.0e-8))
        .detach()
        .cpu(),
        "foreground_hole_ratio": foreground_hole_ratio.detach().cpu(),
    }


def compute_label_audit_batched(
    torch_module,
    camera,
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
    parts: dict[str, list[object]] = {}
    with torch_module.no_grad():
        for start in range(0, int(labels.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(labels.shape[0]))
            audit = compute_label_audit(
                torch_module,
                camera[start:end].to(device=device),
                labels[start:end].to(device=device),
                prototypes.to(device=device),
                class_labels=class_labels,
                foreground_threshold=foreground_threshold,
            )
            for key, value in audit.items():
                parts.setdefault(key, []).append(value)
    return {key: torch_module.cat(values, dim=0) for key, values in parts.items()}


def label_audit_json(labels, audit_tensors, *, class_labels) -> dict[str, object]:
    labels_cpu = labels.detach().long().cpu()
    per_sample = [
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
            "best_other_corr": float(audit_tensors["best_other_corr"][index].item()),
            "label_margin": float(audit_tensors["label_margin"][index].item()),
            "label_correct": bool(audit_tensors["label_correct"][index].item()),
            "foreground_ratio": float(audit_tensors["foreground_ratio"][index].item()),
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
        for index, requested in enumerate(labels_cpu.tolist())
    ]
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
        per_class[str(label)] = {
            "count": len(indices),
            "label_correct_count": int(correct.sum().item()),
            "label_correct_ratio": float(correct.mean().item()),
            "label_margin": _metric_summary(
                audit_tensors["label_margin"].index_select(0, index_tensor)
            ),
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


# Legacy teacher names remain aliases so payload field semantics remain stable.
compute_teacher_distill_label_audit = compute_label_audit
compute_teacher_distill_label_audit_batched = compute_label_audit_batched
_teacher_label_audit_json = label_audit_json
