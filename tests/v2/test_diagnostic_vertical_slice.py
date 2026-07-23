from __future__ import annotations

import json
from pathlib import Path

import torch

from ldm_project.v2.application.cli import main
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import canonical_sha256, file_sha256, tensor_artifact_ref, tensor_sha256
from ldm_project.v2.student.clean_targets import CleanTargetManifestV2
from ldm_project.v2.teacher import build_teacher_pair_manifest, build_teacher_target_manifest


def _checkpoint(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


def _config(path: Path, output_root: Path, first: Path, diffusion: Path) -> None:
    path.write_text(
        f"""schema_version: 2
system:
  kind: diagnostic_amplitude_finite_crop_identity
  slm_shape: [8, 8]
  slm_pixel_pitch_m: 0.0000625
  metasurface_shape: [4, 4]
  metasurface_pixel_pitch_m: 0.000050
  propagation_grid_size: 8
teacher:
  first_stage_checkpoint: {first}
  diffusion_checkpoint: {diffusion}
  target_plane: teacher_camera
  guidance_scale: 4.0
  seed: 1234
  latent_geometry: matched_first_stage
prototype_calibration:
  epochs: 2
  learning_rate: 0.0005
  fixed_exposure_scale: 1.0
capacity_audit:
  dct_modes: 8
  selected_basis_dim: 8
  max_cycles_per_mm: 20.0
  codes_per_label: 2
  seed: 1234
  finite_difference_epsilon: 0.001
  command_gain: 0.1
student_training:
  epochs: 1
  learning_rate: 0.0001
  code_dim: 8
  batch_size: 20
run:
  run_name: vertical
  output_root: {output_root}
  evidence_tier: diagnostic
  device: cpu
""",
        encoding="utf-8",
    )


def _artifact(root: Path, name: str) -> Path:
    return next(root.rglob(name))


def test_cpu_diagnostic_cli_vertical_slice_is_never_acceptance(tmp_path: Path) -> None:
    first, diffusion = tmp_path / "first.pt", tmp_path / "diffusion.pt"
    _checkpoint(first, {"model_version": "first-v1", "train_config": {"first": 1}})
    _checkpoint(diffusion, {"model_version": "diffusion-v1", "first_stage_model_version": "first-v1", "train_config": {"diffusion": 1}})
    config, output_root = tmp_path / "diagnostic.yaml", tmp_path / "outputs"
    _config(config, output_root, first, diffusion)
    assert main(["student", "preflight", "--config", str(config)]) == 0
    preflight = _artifact(output_root, "formal_execution_preflight_evidence.json")
    assert json.loads(preflight.read_text(encoding="utf-8"))["status"] == "diagnostic"
    pair = build_teacher_pair_manifest(first, diffusion, target_plane="teacher_camera", seed=1234, guidance_scale=4.0, split="train")
    codes = torch.randn(20, 8)
    labels = torch.arange(10).repeat_interleave(2)
    dataset = tmp_path / "teacher_dataset.pt"
    torch.save({"codes": codes, "class_labels": labels, "teacher_camera": torch.full((20, 1, 8, 8), 0.5), "teacher_latents": torch.zeros(20, 1, 1, 1), "teacher_initial_noise": torch.zeros(20, 1, 1, 1), "guidance_scale": torch.tensor(4.0), "config": {"code_seed": 1234, "code_projection_seed": 2026, "sampling_steps": 200, "class_labels": list(range(10))}}, dataset)
    target = build_teacher_target_manifest(dataset, pair)
    target_path = tmp_path / "teacher_target_manifest.json"
    target_path.write_text(json.dumps(target.to_dict() | {"manifest_hash": target.manifest_hash}), encoding="utf-8")
    clean = tmp_path / "clean.pt"
    torch.save(torch.full((10, 1, 8, 8), 0.5), clean)
    loaded = load_config(config)
    clean_manifest = CleanTargetManifestV2(2, str(tmp_path), "train", 128, tuple(range(10)), tensor_artifact_ref(clean, role="clean_mnist_targets", git_commit="unit"), tensor_sha256(torch.load(clean, weights_only=False)), (10, 1, 8, 8), "torch.float32", canonical_sha256(loaded.model_dump(mode="json")), "unit")
    clean_path = tmp_path / "clean_manifest.json"
    clean_path.write_text(json.dumps(clean_manifest.to_dict() | {"manifest_hash": clean_manifest.manifest_hash}), encoding="utf-8")

    assert main(["student", "calibrate-prototypes", "--config", str(config), "--clean-target-manifest", str(clean_path)]) == 0
    calibration = _artifact(output_root, "prototype_calibration_evidence.json")
    calibration_root = calibration.parent
    assert main(["audit", "prototype-topology", "--config", str(config), "--clean-target-manifest", str(clean_path), "--prototype-checkpoint", str(calibration_root / "prototype_bundle.pt")]) == 0
    topology = _artifact(output_root, "prototype_topology_evidence.json")
    assert json.loads(topology.read_text(encoding="utf-8"))["status"] == "diagnostic"
    assert main(["audit", "prototype-geometry-upper-bound", "--config", str(config), "--clean-target-manifest", str(clean_path), "--shape-evidence", str(calibration), "--shape-checkpoint", str(calibration_root / "prototype_bundle.pt")]) == 0
    geometry = _artifact(output_root, "prototype_geometry_upper_bound_evidence.json")
    assert json.loads(geometry.read_text(encoding="utf-8"))["permitted_next_actions"] == []
    raw = torch.load(calibration_root / "prototype_raw_cmos.pt", weights_only=False)
    review = tmp_path / "prototype_review.json"
    preview = calibration_root / "prototype_fixed_exposure.png"
    review.write_text(json.dumps({"schema_version": 2, "review_kind": "prototype", "reviewed_tensor_sha256": tensor_sha256(raw), "reviewed_preview_sha256": file_sha256(preview), "readable_labels": list(range(10)), "reviewer": "unit"}), encoding="utf-8")
    assert main(["student", "admit-prototypes", "--config", str(config), "--calibration-evidence", str(calibration), "--prototype-checkpoint", str(calibration_root / "prototype_bundle.pt"), "--prototype-raw-cmos", str(calibration_root / "prototype_raw_cmos.pt"), "--prototype-preview", str(preview), "--measurements", str(calibration_root / "prototype_measurements.json"), "--clean-target-manifest", str(clean_path), "--manual-review", str(review)]) == 0
    prototype = _artifact(output_root, "prototype_evidence.json")
    assert main(["audit", "capacity", "--config", str(config), "--prototype-evidence", str(prototype), "--prototype-checkpoint", str(calibration_root / "prototype_bundle.pt")]) == 0
    capacity = _artifact(output_root, "capacity_evidence.json")
    capacity_root = capacity.parent
    assert main(["student", "train", "--config", str(config), "--prototype-evidence", str(prototype), "--capacity-evidence", str(capacity), "--code-basis", str(capacity_root / "code_basis.pt"), "--prototype-checkpoint", str(calibration_root / "prototype_bundle.pt"), "--teacher-target-manifest", str(target_path)]) == 0
    training = _artifact(output_root, "code_training_evidence.json")
    training_root = training.parent
    sample = tmp_path / "sample.pt"
    code_path, label_path = tmp_path / "codes.pt", tmp_path / "labels.pt"
    torch.save(codes[:2], code_path)
    torch.save(labels[:2], label_path)
    assert main(["student", "sample", "--config", str(config), "--code-basis", str(capacity_root / "code_basis.pt"), "--checkpoint", str(training_root / "student_bundle.pt"), "--codes", str(code_path), "--labels", str(label_path), "--output", str(sample)]) == 0
    report = tmp_path / "evaluation.json"
    assert main(["student", "evaluate", "--config", str(config), "--training-evidence", str(training), "--code-basis", str(capacity_root / "code_basis.pt"), "--checkpoint", str(training_root / "student_bundle.pt"), "--teacher-target-manifest", str(target_path), "--output", str(report)]) == 0
    assert torch.load(sample, weights_only=False).shape == (2, 1, 8, 8)
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "diagnostic"
