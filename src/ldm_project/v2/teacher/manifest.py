"""Teacher checkpoint pairing adapter; this is the only v2-to-v1 model boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..evidence import ArtifactRefV2, canonical_sha256, file_sha256, tensor_sha256


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherPairManifestV2":
        values = dict(payload)
        values.pop("manifest_hash", None)
        return cls(**values)

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


@dataclass(frozen=True)
class TeacherTargetManifestV2:
    """Content-addressed export bundle consumed by v2 training and evaluation."""

    schema_version: int
    teacher_pair_manifest_hash: str
    dataset: ArtifactRefV2
    dataset_hash: str
    codes_hash: str
    labels_hash: str
    teacher_camera_hash: str
    sample_count: int
    samples_per_label: int
    latent_shape: tuple[int, int, int]
    target_plane: str
    seed: int
    guidance_scale: float
    split: str

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"] = self.dataset.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherTargetManifestV2":
        values = dict(payload)
        values.pop("manifest_hash", None)
        values["dataset"] = ArtifactRefV2(**values["dataset"])
        values["latent_shape"] = tuple(values["latent_shape"])
        return cls(**values)

    def validate(self) -> None:
        if self.schema_version != 2 or self.target_plane != "teacher_camera":
            raise ValueError("unsupported teacher target manifest")
        self.dataset.validate()
        if file_sha256(self.dataset.path) != self.dataset_hash:
            raise ValueError("teacher dataset hash changed")
        payload = torch.load(self.dataset.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("teacher dataset must be a tensor bundle")
        required = {"codes", "class_labels", "teacher_camera", "teacher_latents", "guidance_scale"}
        if not required.issubset(payload):
            raise ValueError("teacher dataset is missing required export tensors")
        codes, labels, camera, latents = (payload["codes"], payload["class_labels"], payload["teacher_camera"], payload["teacher_latents"])
        if not all(torch.is_tensor(item) for item in (codes, labels, camera, latents)):
            raise ValueError("teacher dataset values must be tensors")
        if codes.shape[0] != labels.numel() or camera.shape[0] != labels.numel() or latents.shape[0] != labels.numel():
            raise ValueError("teacher dataset sample axes disagree")
        if tuple(latents.shape[1:]) != self.latent_shape or labels.numel() != self.sample_count:
            raise ValueError("teacher dataset geometry changed")
        if not torch.isfinite(codes).all() or not torch.isfinite(camera).all() or camera.min() < 0.0 or camera.max() > 1.0:
            raise ValueError("teacher targets must be finite nonnegative normalized intensities")
        if labels.dtype not in {torch.int64, torch.int32} or labels.min() < 0 or labels.max() > 9:
            raise ValueError("teacher dataset labels must be integers in [0, 9]")
        counts = torch.bincount(labels.to(torch.int64), minlength=10)
        if not torch.equal(counts, torch.full_like(counts, self.samples_per_label)):
            raise ValueError("teacher dataset label counts do not match manifest")
        if tensor_sha256(codes) != self.codes_hash or tensor_sha256(labels) != self.labels_hash or tensor_sha256(camera) != self.teacher_camera_hash:
            raise ValueError("teacher dataset tensor content changed")


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


def build_teacher_target_manifest(dataset_path: str | Path, pair: TeacherPairManifestV2) -> TeacherTargetManifestV2:
    """Validate a legacy export once, then expose a strict v2 dataset boundary."""
    pair.validate()
    target = Path(dataset_path)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("teacher export must be a tensor bundle")
    try:
        codes = payload["codes"]
        labels = payload["class_labels"].to(torch.int64)
        camera = payload["teacher_camera"]
        latents = payload["teacher_latents"]
        guidance = float(payload["guidance_scale"].item())
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("teacher export has an incompatible payload") from error
    if guidance != pair.guidance_scale:
        raise ValueError("teacher export guidance scale does not match pair manifest")
    if labels.numel() == 0 or labels.numel() % 10:
        raise ValueError("teacher export must contain balanced ten-label samples")
    dataset_ref = ArtifactRefV2(str(target.resolve()), file_sha256(target), "teacher_target_dataset")
    manifest = TeacherTargetManifestV2(
        schema_version=2,
        teacher_pair_manifest_hash=pair.manifest_hash,
        dataset=dataset_ref,
        dataset_hash=dataset_ref.sha256,
        codes_hash=tensor_sha256(codes),
        labels_hash=tensor_sha256(labels),
        teacher_camera_hash=tensor_sha256(camera),
        sample_count=int(labels.numel()),
        samples_per_label=int(labels.numel() // 10),
        latent_shape=tuple(int(value) for value in latents.shape[1:]),
        target_plane=pair.target_plane,
        seed=pair.seed,
        guidance_scale=pair.guidance_scale,
        split=pair.split,
    )
    manifest.validate()
    return manifest
