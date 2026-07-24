"""Frozen MNIST semantic evaluator used only during conditional training/evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from ..evidence import atomic_torch_save, canonical_sha256, file_sha256, tensor_sha256


class FrozenMNISTEvaluator(nn.Module):
    """A compact CNN exposing a 128D feature vector before its classifier head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.embedding(self.features(images))
        return features, self.classifier(features)

    def freeze(self) -> "FrozenMNISTEvaluator":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


@dataclass(frozen=True)
class MNISTSplitManifest:
    schema_version: int
    dataset_root: str
    seed: int
    train_indices_hash: str
    validation_indices_hash: str
    train_count: int
    validation_count: int
    dataset_content_hash: str = ""

    @property
    def manifest_hash(self) -> str:
        payload = asdict(self)
        if self.schema_version >= 2:
            payload.pop("dataset_root", None)
        else:
            payload.pop("dataset_content_hash", None)
        return canonical_sha256(payload)


def stratified_mnist_indices(labels: torch.Tensor, *, seed: int = 1234, validation_per_label: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 1 or labels.numel() != 60000 or validation_per_label != 500:
        raise ValueError("conditional MNIST split requires the 60,000-example train split and 500 validation samples per label")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train, validation = [], []
    for label in range(10):
        matches = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        validation.append(shuffled[:validation_per_label])
        train.append(shuffled[validation_per_label:])
    return torch.cat(train), torch.cat(validation)


def load_local_mnist(dataset_root: str | Path, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("torchvision is required for local MNIST") from error
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError("local MNIST root does not exist")
    try:
        dataset = MNIST(root=str(root), train=train, download=False)
    except (RuntimeError, FileNotFoundError) as error:
        raise FileNotFoundError("local MNIST is required; download=False is enforced") from error
    return dataset.data.unsqueeze(1).to(torch.float32).div_(255.0), dataset.targets.to(torch.long)


def balanced_batches(labels: torch.Tensor, *, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
    if batch_size % 10 or batch_size < 50:
        raise ValueError("conditional batch size must be divisible by ten and contain at least five examples per label")
    per_label = batch_size // 10
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = []
    for label in range(10):
        indices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        groups.append(indices.index_select(0, torch.randperm(indices.numel(), generator=generator)))
    count = min(group.numel() for group in groups)
    for start in range(0, count - per_label + 1, per_label):
        yield torch.cat([group[start : start + per_label] for group in groups])


def save_evaluator_checkpoint(path: str | Path, evaluator: FrozenMNISTEvaluator, manifest: MNISTSplitManifest, *, validation_accuracy: float) -> None:
    payload = {"architecture_kind": "frozen_mnist_evaluator_v1", "state_dict": evaluator.state_dict(), "split_manifest": asdict(manifest), "split_manifest_hash": manifest.manifest_hash, "validation_accuracy": validation_accuracy}
    atomic_torch_save(path, payload)


def load_frozen_evaluator(path: str | Path, *, device: torch.device, min_validation_accuracy: float = 0.985) -> tuple[FrozenMNISTEvaluator, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != "frozen_mnist_evaluator_v1":
        raise ValueError("evaluator checkpoint has an incompatible architecture")
    accuracy = float(payload.get("validation_accuracy", 0.0))
    if accuracy < min_validation_accuracy:
        raise ValueError("evaluator validation accuracy is below the 98.5% qualification gate")
    model = FrozenMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.freeze(), payload


def evaluator_checkpoint_hash(path: str | Path) -> str:
    return file_sha256(path)
