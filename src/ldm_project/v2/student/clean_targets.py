"""Content-addressed local MNIST clean-target export for v2 prototype admission."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..evidence import ArtifactRefV2, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256


@dataclass(frozen=True)
class CleanTargetManifestV2:
    schema_version: int
    dataset_root: str
    dataset_split: str
    samples_per_label: int
    labels: tuple[int, ...]
    targets: ArtifactRefV2
    targets_hash: str
    shape: tuple[int, int, int, int]
    dtype: str
    config_hash: str
    git_commit: str
    system_contract_hash: str | None = None
    producer_config_hash: str | None = None

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = self.targets.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CleanTargetManifestV2":
        values = dict(payload)
        values.pop("manifest_hash", None)
        target_ref = dict(values["targets"])
        if target_ref.get("shape") is not None:
            target_ref["shape"] = tuple(target_ref["shape"])
        values["targets"] = ArtifactRefV2(**target_ref)
        values["labels"] = tuple(values["labels"])
        values["shape"] = tuple(values["shape"])
        return cls(**values)

    def validate(self) -> None:
        if self.schema_version != 2 or self.dataset_split != "train":
            raise ValueError("unsupported clean-target manifest")
        if self.samples_per_label != 128 or self.labels != tuple(range(10)):
            raise ValueError("formal clean targets require labels 0-9 with 128 samples each")
        self.targets.validate()
        if file_sha256(self.targets.path) != self.targets.sha256:
            raise ValueError("clean-target artifact hash changed")
        value = safe_torch_load(self.targets.path, expected_sha256=self.targets.sha256)
        if not torch.is_tensor(value) or tuple(value.shape) != self.shape:
            raise ValueError("clean-target tensor shape changed")
        if self.shape[0:2] != (10, 1) or not torch.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
            raise ValueError("clean targets must be finite normalized [10, 1, H, W] tensors")
        if str(value.dtype) != self.dtype or tensor_sha256(value) != self.targets_hash:
            raise ValueError("clean-target tensor content changed")
        if self.system_contract_hash is not None and not self.system_contract_hash:
            raise ValueError("clean-target system contract hash must be non-empty")
        if self.producer_config_hash is not None and not self.producer_config_hash:
            raise ValueError("clean-target producer config hash must be non-empty")


def build_clean_targets(*, dataset_root: str | Path, image_size: int, samples_per_label: int = 128) -> torch.Tensor:
    """Average the first fixed local MNIST samples per label without downloading."""
    if samples_per_label != 128:
        raise ValueError("v2 clean targets lock samples_per_label=128")
    try:
        from torchvision import datasets, transforms
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("torchvision is required to export local MNIST clean targets") from error
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"MNIST root does not exist: {root}")
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    try:
        dataset = datasets.MNIST(root=str(root), train=True, download=False, transform=transform)
    except (RuntimeError, FileNotFoundError) as error:
        raise FileNotFoundError("local MNIST training data is required; download=False was enforced") from error
    buckets: dict[int, list[torch.Tensor]] = {label: [] for label in range(10)}
    for image, label in dataset:
        label = int(label)
        if len(buckets[label]) < samples_per_label:
            buckets[label].append(image)
        if all(len(values) == samples_per_label for values in buckets.values()):
            break
    if any(len(values) != samples_per_label for values in buckets.values()):
        raise ValueError("local MNIST data does not contain 128 samples for every label")
    return torch.stack([torch.stack(buckets[label]).mean(dim=0) for label in range(10)]).contiguous()
