"""MNIST prototype-bank construction shared by teacher and student workflows."""

from __future__ import annotations

from ..visualization_core import _normalize_for_visualization


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
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
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


# Compatibility names are intentionally public at the neutral boundary.
_build_label_prototype_bank_from_tensors = build_label_prototype_bank_from_tensors
_build_mnist_label_prototype_bank = build_mnist_label_prototype_bank
