from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DataConfig, FirstStageConfig


@dataclass
class DatasetBundle:
    dataset: Dataset
    loader: DataLoader
    effective_size: int
    total_size: int


@dataclass
class DatasetSplits:
    train: DatasetBundle
    validation: DatasetBundle | None
    total_size: int


def _build_mnist_dataset(root: str, train: bool, image_size: int, download: bool) -> Dataset:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load the MNIST dataset.") from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)


def _apply_subset(dataset: Dataset, subset_fraction: float, subset_seed: int) -> Dataset:
    if subset_fraction <= 0.0 or subset_fraction > 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    if subset_fraction >= 1.0:
        return dataset

    subset_size = max(1, int(len(dataset) * subset_fraction))
    generator = torch.Generator().manual_seed(subset_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def _build_split_indices(total_size: int, validation_fraction: float, validation_split_seed: int) -> tuple[list[int], list[int]]:
    if validation_fraction == 0.0:
        return list(range(total_size)), []

    validation_size = int(total_size * validation_fraction)
    if validation_size <= 0:
        raise ValueError(
            "data.validation_fraction produced an empty validation split; "
            "increase the fraction or disable validation with 0.0."
        )
    if validation_size >= total_size:
        raise ValueError(
            "data.validation_fraction must leave at least one training sample; "
            f"got validation_size={validation_size} for total_size={total_size}."
        )

    generator = torch.Generator().manual_seed(validation_split_seed)
    shuffled_indices = torch.randperm(total_size, generator=generator).tolist()
    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]
    return training_indices, validation_indices


def _apply_subset_to_indices(indices: list[int], subset_fraction: float, subset_seed: int) -> list[int]:
    if subset_fraction <= 0.0 or subset_fraction > 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    if subset_fraction >= 1.0:
        return indices

    subset_size = max(1, int(len(indices) * subset_fraction))
    generator = torch.Generator().manual_seed(subset_seed)
    selected_offsets = torch.randperm(len(indices), generator=generator)[:subset_size].tolist()
    # 训练子集在 split 之后再抽，确保验证集稳定且不受 subset_fraction 影响。
    return [indices[offset] for offset in selected_offsets]


def build_train_validation_loaders(
    data_config: DataConfig,
    first_stage_config: FirstStageConfig,
    batch_size: int,
) -> DatasetSplits:
    dataset_name = data_config.dataset.lower()
    if dataset_name != "mnist":
        raise ValueError(f"Unsupported dataset: {data_config.dataset}")

    dataset = _build_mnist_dataset(
        root=data_config.root,
        train=True,
        image_size=first_stage_config.image_size,
        download=data_config.download,
    )
    total_size = len(dataset)
    train_indices, validation_indices = _build_split_indices(
        total_size=total_size,
        validation_fraction=data_config.validation_fraction,
        validation_split_seed=data_config.validation_split_seed,
    )
    effective_train_indices = _apply_subset_to_indices(
        train_indices,
        subset_fraction=data_config.subset_fraction,
        subset_seed=data_config.subset_seed,
    )

    train_dataset = Subset(dataset, effective_train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
    )
    validation_bundle: DatasetBundle | None = None
    if validation_indices:
        validation_dataset = Subset(dataset, validation_indices)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=False,
        )
        validation_bundle = DatasetBundle(
            dataset=validation_dataset,
            loader=validation_loader,
            effective_size=len(validation_indices),
            total_size=len(validation_indices),
        )

    return DatasetSplits(
        train=DatasetBundle(
            dataset=train_dataset,
            loader=train_loader,
            effective_size=len(effective_train_indices),
            total_size=len(train_indices),
        ),
        validation=validation_bundle,
        total_size=total_size,
    )


def build_training_loader(
    data_config: DataConfig,
    first_stage_config: FirstStageConfig,
    batch_size: int,
    train: bool = True,
) -> DatasetBundle:
    if train:
        return build_train_validation_loaders(
            data_config=data_config,
            first_stage_config=first_stage_config,
            batch_size=batch_size,
        ).train

    dataset_name = data_config.dataset.lower()
    if dataset_name != "mnist":
        raise ValueError(f"Unsupported dataset: {data_config.dataset}")

    dataset = _build_mnist_dataset(
        root=data_config.root,
        train=False,
        image_size=first_stage_config.image_size,
        download=data_config.download,
    )
    total_size = len(dataset)
    dataset = _apply_subset(dataset, data_config.subset_fraction, data_config.subset_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
    )
    return DatasetBundle(
        dataset=dataset,
        loader=loader,
        effective_size=len(dataset),
        total_size=total_size,
    )
