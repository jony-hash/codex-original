"""Strict v2 checkpoints; migrations are explicit and never implicit full loads."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..evidence import ArtifactRefV2, atomic_torch_save, canonical_sha256, file_sha256, tensor_sha256


def state_signature(state: Mapping[str, torch.Tensor]) -> str:
    descriptor = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
    }
    return canonical_sha256(descriptor)


def state_content_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointManifestV2:
    schema_version: int
    system_kind: str
    boundary_contract: str
    system_signature: str
    config_hash: str
    dataset_hash: str
    teacher_manifest_hash: str | None
    calibration_hash: str
    basis_hash: str
    state_signature: str
    state_content_hash: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def save_exact_checkpoint(path: str | Path, model: torch.nn.Module, manifest: CheckpointManifestV2, *, extra: Mapping[str, Any] | None = None) -> None:
    state = model.state_dict()
    if manifest.schema_version != 2 or manifest.state_signature != state_signature(state) or manifest.state_content_hash != state_content_hash(state):
        raise ValueError("checkpoint manifest does not exactly describe model state")
    payload = {"manifest": manifest.to_dict(), "state_dict": state, "extra": dict(extra or {})}
    atomic_torch_save(path, payload)


def load_exact_checkpoint(path: str | Path, model: torch.nn.Module, expected: CheckpointManifestV2) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("invalid v2 checkpoint payload")
    actual = CheckpointManifestV2(**payload["manifest"])
    if actual != expected:
        raise ValueError("v2 checkpoint manifest does not match the requested formal system")
    state = payload["state_dict"]
    if state_signature(state) != actual.state_signature or state_content_hash(state) != actual.state_content_hash or state_signature(model.state_dict()) != actual.state_signature:
        raise ValueError("v2 checkpoint state signature is not exact")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError("strict v2 checkpoint load found state mismatch")
    return payload.get("extra", {})


def checkpoint_file_hash(path: str | Path) -> str:
    return file_sha256(path)


@dataclass(frozen=True)
class WarmStartManifestV2:
    schema_version: int
    source_checkpoint: ArtifactRefV2
    source_system_signature: str
    target_system_signature: str
    loaded_key_signatures: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_checkpoint"] = self.source_checkpoint.to_dict()
        return payload


def load_prototype_warm_start(path: str | Path, model: torch.nn.Module, *, expected_system_signature: str) -> WarmStartManifestV2:
    """Load only the auditable prototype/template and phase state boundary."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    manifest_payload = payload.get("manifest") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not isinstance(manifest_payload, dict):
        raise ValueError("prototype checkpoint is invalid")
    source_manifest = CheckpointManifestV2(**manifest_payload)
    if source_manifest.system_signature != expected_system_signature:
        raise ValueError("prototype checkpoint system signature does not match warm-start target")
    allowed = ("generator.template_logits", "system.encoder.phase", "system.decoder.phase")
    target = model.state_dict()
    loaded: dict[str, str] = {}
    for key in allowed:
        if key not in state or key not in target or state[key].shape != target[key].shape:
            raise ValueError(f"prototype warm-start lacks compatible {key}")
        target[key].copy_(state[key])
        loaded[key] = state_signature({key: state[key]})
    model.load_state_dict(target, strict=True)
    return WarmStartManifestV2(
        schema_version=2,
        source_checkpoint=ArtifactRefV2(str(Path(path).resolve()), file_sha256(path), "prototype_checkpoint"),
        source_system_signature=source_manifest.system_signature,
        target_system_signature=expected_system_signature,
        loaded_key_signatures=loaded,
    )
