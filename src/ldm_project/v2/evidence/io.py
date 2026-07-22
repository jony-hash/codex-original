"""Content hashes and atomic artifact writes shared by v2 subsystems."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


def canonical_sha256(value: Mapping[str, Any] | list[Any] | str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) if not isinstance(value, str) else value
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRefV2:
    path: str
    sha256: str
    role: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tensor_artifact_ref(path: str | Path, *, role: str, git_commit: str | None = None) -> ArtifactRefV2:
    target = Path(path)
    value = torch.load(target, map_location="cpu", weights_only=False)
    if not torch.is_tensor(value):
        raise ValueError(f"{target} must contain one tensor")
    return ArtifactRefV2(str(target.resolve()), file_sha256(target), role, tuple(value.shape), str(value.dtype), git_commit)


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        return Path(handle.name)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
