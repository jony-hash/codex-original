from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import atomic_write_json, tensor_artifact_ref
from ldm_project.v2.student.checkpoints import CheckpointManifestV2, load_exact_checkpoint, save_exact_checkpoint, state_content_hash, state_signature
from ldm_project.v2.teacher import build_teacher_pair_manifest, build_teacher_target_manifest


def _teacher_payload(model_version: str, *, first_stage_model_version: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"model_version": model_version, "train_config": {"name": model_version}}
    if first_stage_model_version is not None:
        payload["first_stage_model_version"] = first_stage_model_version
    return payload


def test_teacher_manifest_requires_matched_pair_and_stable_hash(tmp_path: Path) -> None:
    first_stage = tmp_path / "first.pt"
    diffusion = tmp_path / "diffusion.pt"
    torch.save(_teacher_payload("first-v1"), first_stage)
    torch.save(_teacher_payload("diffusion-v1", first_stage_model_version="first-v1"), diffusion)
    manifest = build_teacher_pair_manifest(first_stage, diffusion, target_plane="teacher_camera", seed=1234, guidance_scale=4.0, split="train")
    manifest.validate()
    assert len(manifest.manifest_hash) == 64
    torch.save(_teacher_payload("diffusion-v1", first_stage_model_version="other"), diffusion)
    with pytest.raises(ValueError, match="another first-stage"):
        build_teacher_pair_manifest(first_stage, diffusion, target_plane="teacher_camera", seed=1234, guidance_scale=4.0, split="train")


def test_exact_checkpoint_refuses_any_contract_or_state_change(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    manifest = CheckpointManifestV2(2, "formal_amplitude_finite_crop_identity", "finite_crop", "system", "config", "dataset", "teacher", "calibration", "basis", state_signature(model.state_dict()), state_content_hash(model.state_dict()))
    path = tmp_path / "checkpoint.pt"
    save_exact_checkpoint(path, model, manifest, extra={"kind": "unit"})
    restored = torch.nn.Linear(3, 2)
    assert load_exact_checkpoint(path, restored, manifest)["kind"] == "unit"
    incompatible = CheckpointManifestV2(2, "formal_amplitude_finite_crop_identity", "finite_crop", "other-system", "config", "dataset", "teacher", "calibration", "basis", state_signature(model.state_dict()), state_content_hash(model.state_dict()))
    with pytest.raises(ValueError, match="manifest"):
        load_exact_checkpoint(path, restored, incompatible)
    with pytest.raises(ValueError, match="signature"):
        load_exact_checkpoint(path, torch.nn.Linear(3, 3), manifest)
    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["weight"].add_(1.0)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="signature"):
        load_exact_checkpoint(path, torch.nn.Linear(3, 2), manifest)


def test_atomic_json_write_replaces_only_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    atomic_write_json(path, {"status": "passed"})
    assert path.read_text(encoding="utf-8").strip().endswith("}")


def test_tensor_artifact_ref_is_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "tensor.pt"
    torch.save(torch.ones(2, 3), path)
    first = tensor_artifact_ref(path, role="clean_targets")
    torch.save(torch.zeros(2, 3), path)
    second = tensor_artifact_ref(path, role="clean_targets")
    assert first.sha256 != second.sha256
    assert first.shape == second.shape == (2, 3)


def test_formal_teacher_manifest_requires_exact_8d_100_per_label(tmp_path: Path) -> None:
    first, diffusion = tmp_path / "first.pt", tmp_path / "diffusion.pt"
    torch.save(_teacher_payload("first-v1"), first)
    torch.save(_teacher_payload("diffusion-v1", first_stage_model_version="first-v1"), diffusion)
    pair = build_teacher_pair_manifest(first, diffusion, target_plane="teacher_camera", seed=1234, guidance_scale=4.0, split="train")
    labels = torch.arange(10).repeat_interleave(100)
    payload = {"codes": torch.randn(1000, 128), "class_labels": labels, "teacher_camera": torch.rand(1000, 1, 2, 2), "teacher_latents": torch.zeros(1000, 1, 1, 1), "teacher_initial_noise": torch.zeros(1000, 1, 1, 1), "guidance_scale": torch.tensor(4.0), "config": {"code_seed": 1234, "code_projection_seed": 2026, "sampling_steps": 200, "class_labels": list(range(10))}}
    dataset = tmp_path / "historical_128d.pt"
    torch.save(payload, dataset)
    with pytest.raises(ValueError, match="code_dim=8"):
        build_teacher_target_manifest(dataset, pair, formal=True)
    payload["codes"] = torch.randn(1000, 8)
    torch.save(payload, dataset)
    manifest = build_teacher_target_manifest(dataset, pair, formal=True, generated_git_commit="unit", v2_config_hash="config", legacy_export_config_hash="legacy")
    manifest.validate(formal=True)
    payload["config"]["sampling_steps"] = 199
    torch.save(payload, dataset)
    with pytest.raises(ValueError, match="content"):
        manifest.validate(formal=True)
