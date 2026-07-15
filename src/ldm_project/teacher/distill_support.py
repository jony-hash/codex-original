from __future__ import annotations

from ..generation.codes import project_codes_to_teacher_noise
from ..train_utils import validate_class_label_values
from ..visualization_core import _normalize_for_visualization

_project_codes_to_teacher_noise = project_codes_to_teacher_noise


def distill_class_label_sequence(torch_module, train_cfg, device):
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



def metric_summary(values) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }

def class_relative_minmax(values):
    values = values.detach().float()
    spread = values.max() - values.min()
    if float(spread.item()) <= 1.0e-8:
        return values.new_zeros(values.shape)
    return (values - values.min()) / spread

def build_label_prototype_bank_from_tensors(
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

def build_mnist_label_prototype_bank(
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
    return build_label_prototype_bank_from_tensors(
        torch_module,
        torch_module.stack(prototype_images, dim=0),
        torch_module.tensor(prototype_labels, dtype=torch_module.long),
        class_labels=requested,
        image_size=image_size,
    )

def prototype_correlation(torch_module, samples, prototypes):
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

def masked_ratio(values, mask):
    return values.float().sum(dim=(1, 2, 3)) / mask.float().sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)

def balanced_preview_indices(
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

# Historical private names remain aliases in the facade/module namespace.
_distill_class_label_sequence = distill_class_label_sequence
_metric_summary = metric_summary
_class_relative_minmax = class_relative_minmax
_build_label_prototype_bank_from_tensors = build_label_prototype_bank_from_tensors
_build_mnist_label_prototype_bank = build_mnist_label_prototype_bank
_prototype_correlation = prototype_correlation
_masked_ratio = masked_ratio
_balanced_preview_indices = balanced_preview_indices

__all__ = [
    "distill_class_label_sequence", "metric_summary", "class_relative_minmax",
    "build_label_prototype_bank_from_tensors", "build_mnist_label_prototype_bank",
    "prototype_correlation", "masked_ratio", "balanced_preview_indices",
    "project_codes_to_teacher_noise", "_project_codes_to_teacher_noise",
    "_distill_class_label_sequence", "_metric_summary", "_class_relative_minmax",
    "_build_label_prototype_bank_from_tensors", "_build_mnist_label_prototype_bank",
    "_prototype_correlation", "_masked_ratio", "_balanced_preview_indices",
    "_normalize_for_visualization",
]
