"""Teacher checkpoint pairing adapter; this is the only v2-to-v1 model boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..evidence import canonical_sha256, file_sha256


@dataclass(frozen=True)
class TeacherPairManifestV2:
    schema_version: int
    first_stage_path: str
    diffusion_path: str
    first_stage_sha256: str
    diffusion_sha256: str
    first_stage_model_version: str
    diffusion_model_version: str
    config_hash: str
    latent_geometry: str
    target_plane: str
    seed: int
    guidance_scale: float
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != 2 or self.target_plane != "teacher_camera":
            raise ValueError("unsupported v2 teacher manifest")
        if self.latent_geometry != "matched_first_stage":
            raise ValueError("teacher manifest must use matched_first_stage geometry")
        if not Path(self.first_stage_path).is_file() or not Path(self.diffusion_path).is_file():
            raise FileNotFoundError("teacher manifest checkpoint is missing")
        if file_sha256(self.first_stage_path) != self.first_stage_sha256 or file_sha256(self.diffusion_path) != self.diffusion_sha256:
            raise ValueError("teacher checkpoint hash does not match manifest")


def _payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("teacher checkpoint payload must be a mapping")
    return payload


def build_teacher_pair_manifest(
    first_stage_path: str | Path,
    diffusion_path: str | Path,
    *,
    target_plane: str,
    seed: int,
    guidance_scale: float,
    split: str,
    latent_geometry: str = "matched_first_stage",
) -> TeacherPairManifestV2:
    """Inspect existing v1 payloads without changing their network or serialization."""
    first_stage = _payload(first_stage_path)
    diffusion = _payload(diffusion_path)
    first_version = first_stage.get("model_version")
    diffusion_version = diffusion.get("model_version")
    if not isinstance(first_version, str) or not isinstance(diffusion_version, str):
        raise ValueError("teacher checkpoints must declare model_version")
    expected_first_version = diffusion.get("first_stage_model_version")
    if expected_first_version is not None and expected_first_version != first_version:
        raise ValueError("diffusion checkpoint was trained against another first-stage version")
    config_hash = canonical_sha256({"first_stage": first_stage.get("train_config"), "diffusion": diffusion.get("train_config")})
    manifest = TeacherPairManifestV2(
        schema_version=2,
        first_stage_path=str(Path(first_stage_path)),
        diffusion_path=str(Path(diffusion_path)),
        first_stage_sha256=file_sha256(first_stage_path),
        diffusion_sha256=file_sha256(diffusion_path),
        first_stage_model_version=first_version,
        diffusion_model_version=diffusion_version,
        config_hash=config_hash,
        latent_geometry=latent_geometry,
        target_plane=target_plane,
        seed=seed,
        guidance_scale=guidance_scale,
        split=split,
    )
    manifest.validate()
    return manifest
