"""Content hashes and atomic artifact writes shared by v2 subsystems."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import torch


def _canonical_value(value: Any) -> Any:
    """Convert checkpoint-era dataclasses and tensors to stable JSON values."""
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value) if hasattr(value, field.name)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _canonical_value(vars(value))
    raise TypeError(f"cannot canonically serialize {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a v2 tensor/container artifact with an optional pre-load digest check."""
    target = Path(path)
    if expected_sha256 is not None and file_sha256(target) != expected_sha256:
        raise ValueError(f"artifact content no longer matches {target}")
    try:
        return torch.load(target, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"artifact {target} is not safely loadable") from exc


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

    def validate(self) -> None:
        target = Path(self.path)
        if self.shape is not None or self.dtype is not None:
            try:
                value = safe_torch_load(target, expected_sha256=self.sha256)
            except ValueError as exc:
                if "content no longer matches" in str(exc):
                    raise ValueError(f"artifact content no longer matches {self.role}") from exc
                raise ValueError(f"tensor artifact {self.role} is not safely loadable") from exc
            except Exception as exc:
                raise ValueError(f"tensor artifact {self.role} is not safely loadable") from exc
            if not torch.is_tensor(value):
                raise ValueError(f"tensor artifact {self.role} is no longer a tensor")
            if self.shape is not None and tuple(value.shape) != self.shape:
                raise ValueError(f"tensor artifact {self.role} shape changed")
            if self.dtype is not None and str(value.dtype) != self.dtype:
                raise ValueError(f"tensor artifact {self.role} dtype changed")
            return
        if not target.is_file() or file_sha256(target) != self.sha256:
            raise ValueError(f"artifact content no longer matches {self.role}")


def tensor_artifact_ref(path: str | Path, *, role: str, git_commit: str | None = None) -> ArtifactRefV2:
    target = Path(path)
    try:
        value = safe_torch_load(target)
    except Exception as exc:
        raise ValueError(f"{target} is not safely loadable as a tensor") from exc
    if not torch.is_tensor(value):
        raise ValueError(f"{target} must contain one tensor")
    return ArtifactRefV2(str(target.resolve()), file_sha256(target), role, tuple(value.shape), str(value.dtype), git_commit)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor content plus shape/dtype, independent of torch serialization bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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


def atomic_write_new_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    artifact_kind: str = "artifact",
) -> None:
    """Publish one complete JSON artifact without replacing an existing file."""
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {artifact_kind}: {target}") from exc
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
