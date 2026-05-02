from __future__ import annotations

from collections.abc import Sized
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


def _dataset_len(dataset: Dataset) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must provide __len__ for train/validation splitting.")
    return len(dataset)


def _build_mnist_dataset(root: str, train: bool, image_size: int, download: bool) -> Dataset:
    try:
        from torchvision import datasets, transforms  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load the MNIST dataset.") from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)


def _resolve_subset_size(total_size: int, subset_fraction: float, subset_size: int | None) -> int:
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {subset_size}")
        if subset_size > total_size:
            raise ValueError(f"subset_size must be <= dataset size ({total_size}), got {subset_size}")
        return subset_size
    if subset_fraction <= 0.0 or subset_fraction > 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    return max(1, int(total_size * subset_fraction))


def _apply_subset(dataset: Dataset, subset_fraction: float, subset_seed: int, subset_size: int | None = None) -> Dataset:
    dataset_size = _dataset_len(dataset)
    resolved_subset_size = _resolve_subset_size(dataset_size, subset_fraction, subset_size)
    if resolved_subset_size >= dataset_size:
        return dataset

    generator = torch.Generator().manual_seed(subset_seed)
    indices = torch.randperm(dataset_size, generator=generator)[:resolved_subset_size].tolist()
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


def _apply_subset_to_indices(
    indices: list[int],
    subset_fraction: float,
    subset_seed: int,
    subset_size: int | None = None,
) -> list[int]:
    resolved_subset_size = _resolve_subset_size(len(indices), subset_fraction, subset_size)
    if resolved_subset_size >= len(indices):
        return indices

    generator = torch.Generator().manual_seed(subset_seed)
    selected_offsets = torch.randperm(len(indices), generator=generator)[:resolved_subset_size].tolist()
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
    total_size = _dataset_len(dataset)
    train_indices, validation_indices = _build_split_indices(
        total_size=total_size,
        validation_fraction=data_config.validation_fraction,
        validation_split_seed=data_config.validation_split_seed,
    )
    effective_train_indices = _apply_subset_to_indices(
        train_indices,
        subset_fraction=data_config.subset_fraction,
        subset_seed=data_config.subset_seed,
        subset_size=data_config.subset_size,
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
    total_size = _dataset_len(dataset)
    dataset = _apply_subset(
        dataset,
        data_config.subset_fraction,
        data_config.subset_seed,
        subset_size=data_config.subset_size,
    )
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
        effective_size=_dataset_len(dataset),
        total_size=total_size,
    )
