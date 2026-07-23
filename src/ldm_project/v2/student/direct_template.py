"""Immutable diagnostic artifacts for admitted direct-template commands."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from ..evidence import ArtifactRefV2, canonical_sha256


@dataclass(frozen=True)
class DirectTemplatePrototypeArtifactV2:
    """Content-addressed, read-only prototype commands; never a student checkpoint."""

    schema_version: int
    system_signature: str
    config_hash: str
    clean_target_manifest_hash: str
    shape_checkpoint_sha256: str
    source_admission_evidence_hash: str
    source_replay_evidence_hash: str
    fixed_exposure_scale: float
    commands: ArtifactRefV2
    raw_cmos: ArtifactRefV2
    supports: ArtifactRefV2

    @property
    def artifact_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DirectTemplatePrototypeArtifactV2":
        values = dict(payload)
        for name in ("commands", "raw_cmos", "supports"):
            values[name] = ArtifactRefV2(**values[name])
        values.pop("artifact_hash", None)
        return cls(**values)

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported direct-template prototype artifact schema")
        for reference in (self.commands, self.raw_cmos, self.supports):
            reference.validate()
        commands = torch.load(self.commands.path, map_location="cpu", weights_only=False)
        raw = torch.load(self.raw_cmos.path, map_location="cpu", weights_only=False)
        supports = torch.load(self.supports.path, map_location="cpu", weights_only=False)
        if not all(torch.is_tensor(value) for value in (commands, raw, supports)):
            raise ValueError("direct-template artifact payloads must be tensors")
        if commands.ndim != 4 or commands.shape[:2] != (10, 1) or raw.shape != commands.shape:
            raise ValueError("direct-template commands/raw tensors must be [10, 1, H, W]")
        if supports.shape != (10, *commands.shape[-2:]):
            raise ValueError("direct-template support tensor must be [10, H, W]")
        if commands.dtype != torch.float32 or not torch.isfinite(commands).all() or bool((commands < 0).any()) or bool((commands > 1).any()):
            raise ValueError("direct-template commands violate the amplitude contract")
        if not torch.isfinite(raw).all() or not torch.isfinite(supports).all():
            raise ValueError("direct-template artifact tensors must be finite")
        expected_supports = (commands[:, 0] >= 0.2).to(supports.dtype)
        if not torch.equal(supports, expected_supports):
            raise ValueError("direct-template support is not the locked command >= 0.2 support")
