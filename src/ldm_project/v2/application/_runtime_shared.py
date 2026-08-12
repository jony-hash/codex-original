"""Architecture-neutral v2 runtime helpers.

These helpers own no architecture dispatch, training command, or acceptance
decision.  Current and diagnostic runtimes import them without depending on
each other.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from ..config import V2Config
from ..evidence import EvidenceContextV2, EvidenceRecordV2, atomic_write_json, canonical_sha256, tensor_sha256
from ..student.evaluator import MNISTSplitManifest, stratified_mnist_indices
from .runs import V2Run
from .runtime_lineage import lineage_inputs


def system_signature(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def build_context(
    config: V2Config,
    *,
    evaluator_hash: str | None = None,
    split_hash: str | None = None,
    checkpoint_hash: str | None = None,
    candidate_manifest_hash: str | None = None,
) -> EvidenceContextV2:
    settings = config.conditional_generation
    assert settings is not None
    return EvidenceContextV2(
        system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        checkpoint_hash=checkpoint_hash,
        evaluator_checkpoint_hash=evaluator_hash,
        dataset_split_manifest_hash=split_hash,
        exposure_scale=settings.fixed_exposure_scale,
        candidate_dataset_manifest_hash=candidate_manifest_hash,
    )


def write_evidence(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    input_hashes = dict(record.input_hashes)
    input_hashes.update(lineage_inputs(Path.cwd(), run.git_commit, record.evidence_kind))
    record = EvidenceRecordV2(
        record.schema_version,
        record.evidence_kind,
        record.tier,
        record.status,
        record.git_commit,
        record.dirty_patch_hash,
        record.config_hash,
        record.context,
        input_hashes,
        record.gates,
        record.permitted_next_actions,
        record.invalidation_reason,
    )
    record.validate()
    path = run.root / name
    atomic_write_json(path, record.to_dict())
    return path


def new_run(config: V2Config) -> V2Run:
    if config.run.evidence_tier == "acceptance":
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("acceptance operations require a clean git commit")
    return V2Run.create(
        output_root=config.run.output_root,
        run_name=config.run.run_name,
        config_payload=config.model_dump(mode="json"),
    )


def split_mnist(
    images: torch.Tensor, labels: torch.Tensor, dataset_root: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MNISTSplitManifest]:
    train_indices, validation_indices = stratified_mnist_indices(labels)
    manifest = MNISTSplitManifest(
        2,
        str(Path(dataset_root).resolve()),
        1234,
        canonical_sha256(train_indices),
        canonical_sha256(validation_indices),
        train_indices.numel(),
        validation_indices.numel(),
        canonical_sha256({"images": tensor_sha256(images), "labels": tensor_sha256(labels)}),
    )
    return (
        images.index_select(0, train_indices),
        labels.index_select(0, train_indices),
        images.index_select(0, validation_indices),
        labels.index_select(0, validation_indices),
        manifest,
    )


def carrier_logits(
    images: torch.Tensor, labels: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    means = torch.stack([images[labels == label].mean(dim=0) for label in range(10)])
    enlarged = torch.nn.functional.interpolate(
        means, size=shape, mode="bilinear", align_corners=False
    )
    amplitude = enlarged.sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(amplitude).squeeze(1)


def random_codes(count: int, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(count, 8, generator=generator).mul_(2.0).sub_(1.0).to(device)
