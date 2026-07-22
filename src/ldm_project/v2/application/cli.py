"""Explicit v2 subcommand CLI. v1 ``train.py`` remains untouched."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from ..config import V2Config, load_config
from ..evidence import EvidenceRecordV2, GateResult, atomic_write_json, canonical_sha256, require_passed_evidence
from ..optics import FormalAmplitudeFiniteCropIdentitySystem
from ..student.admission import CapacityMetrics, PrototypeMetrics, capacity_gate, prototype_gate
from ..student.capacity import CodeBasisArtifactV2, build_code_basis, capacity_metrics, finite_difference_response_matrices
from ..student.checkpoints import CheckpointManifestV2, load_exact_checkpoint, save_exact_checkpoint, state_signature
from ..student.generator import LabelPrototypeBasisGenerator
from ..student.metrics import correlation, foreground_dice
from ..student.training import calibrate_prototypes, train_code_conditioned
from ..teacher import build_teacher_pair_manifest
from .runs import V2Run


class StudentBundle(nn.Module):
    def __init__(self, config: V2Config, basis: CodeBasisArtifactV2) -> None:
        super().__init__()
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        self.generator = LabelPrototypeBasisGenerator(code_dim=config.student_training.code_dim if config.student_training else 8, basis_artifact=basis)


def _tensor(path: str, *, device: torch.device) -> torch.Tensor:
    value = torch.load(path, map_location=device, weights_only=False)
    if not torch.is_tensor(value):
        raise ValueError(f"{path} must contain one tensor")
    return value.to(device)


def _record(path: str) -> EvidenceRecordV2:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvidenceRecordV2.from_dict(payload)


def _write_record(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    record.validate()
    path = run.root / name
    atomic_write_json(path, record.to_dict())
    return path


def _zero_basis(shape: tuple[int, int]) -> CodeBasisArtifactV2:
    basis = torch.zeros(10, 8, *shape)
    return CodeBasisArtifactV2(basis, torch.ones(10, 1, 1), 8, 1234, "prototype-calibration-zero-basis")


def _bundle_manifest(config: V2Config, bundle: StudentBundle, *, dataset_hash: str, calibration_hash: str) -> CheckpointManifestV2:
    return CheckpointManifestV2(2, config.system.kind, config.system.boundary, canonical_sha256(config.model_dump(mode="json")), dataset_hash, calibration_hash, state_signature(bundle.state_dict()))


def _manual_labels(path: str | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("readable_labels")
    return labels if isinstance(labels, list) else None


def _legacy_teacher(args: argparse.Namespace, config: V2Config, stage: str) -> None:
    command = [sys.executable, "train.py", "--train-config", args.legacy_train_config, "--stage", stage, "--device", config.run.device]
    subprocess.run(command, check=True)


def _teacher_manifest(args: argparse.Namespace, config: V2Config) -> int:
    if config.teacher is None:
        raise ValueError("teacher command requires teacher config")
    run = V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))
    manifest = build_teacher_pair_manifest(config.teacher.first_stage_checkpoint, config.teacher.diffusion_checkpoint, target_plane=config.teacher.target_plane, seed=config.teacher.seed, guidance_scale=config.teacher.guidance_scale, split=args.split, latent_geometry=config.teacher.latent_geometry)
    atomic_write_json(run.root / "teacher_pair_manifest.json", manifest.to_dict() | {"manifest_hash": manifest.manifest_hash})
    print(run.root / "teacher_pair_manifest.json")
    return 0


def _calibrate(args: argparse.Namespace, config: V2Config) -> int:
    if config.prototype_calibration is None:
        raise ValueError("student calibrate-prototypes requires prototype_calibration config")
    device = torch.device(config.run.device)
    clean_targets = _tensor(args.clean_targets, device=device)
    run = V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    summary = calibrate_prototypes(bundle.system, bundle.generator, clean_targets, exposure_scale=config.prototype_calibration.fixed_exposure_scale, epochs=config.prototype_calibration.epochs, learning_rate=config.prototype_calibration.learning_rate)
    metrics = PrototypeMetrics(summary.clean_corr, summary.clean_dice, args.prototype_accuracy, summary.high_saturation, torch.isfinite(torch.tensor(summary.loss)).item(), summary.gradients_finite, True, True)
    gate = prototype_gate(metrics, _manual_labels(args.manual_review))
    status = "passed" if gate.passed and config.run.evidence_tier == "acceptance" and run.dirty_patch_hash is None else "diagnostic"
    record = EvidenceRecordV2(2, "prototype", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, {"clean_targets": canonical_sha256(str(Path(args.clean_targets).resolve()))}, (gate,), ("audit_capacity",) if status == "passed" else ())
    _write_record(run, "prototype_evidence.json", record)
    manifest = _bundle_manifest(config, bundle, dataset_hash=canonical_sha256(str(Path(args.clean_targets).resolve())), calibration_hash=basis.artifact_hash)
    save_exact_checkpoint(run.root / "prototype_bundle.pt", bundle, manifest, extra={"basis_artifact_hash": basis.artifact_hash})
    print(run.root)
    return 0


def _capacity(args: argparse.Namespace, config: V2Config) -> int:
    if config.capacity_audit is None:
        raise ValueError("audit capacity requires capacity_audit config")
    prototype = _record(args.prototype_evidence)
    require_passed_evidence(prototype, kind="prototype")
    device = torch.device(config.run.device)
    supports = _tensor(args.supports, device=device)
    commands = _tensor(args.commands, device=device)
    slm_intensity = _tensor(args.slm_intensity, device=device)
    camera = _tensor(args.raw_camera, device=device)
    parameters_unchanged = True
    if args.response_matrices is not None:
        responses = _tensor(args.response_matrices, device=device)
    else:
        if args.prototype_checkpoint is None:
            raise ValueError("finite-difference capacity audit requires --prototype-checkpoint")
        bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
        _warm_start_prototype(bundle, args.prototype_checkpoint)
        responses, parameters_unchanged = finite_difference_response_matrices(
            bundle.system,
            _tensor(args.prototype_commands, device=device),
            supports,
            candidate_count=config.capacity_audit.dct_modes,
            epsilon=config.capacity_audit.finite_difference_epsilon,
        )
    artifact = build_code_basis(responses, supports, candidate_count=config.capacity_audit.dct_modes, seed=config.capacity_audit.seed, source_hash=prototype.evidence_hash)
    slm_factor, camera_factor, support_energy, clipping = capacity_metrics(commands, slm_intensity, camera, supports)
    gate = capacity_gate(CapacityMetrics(slm_factor, camera_factor, support_energy, clipping, parameters_unchanged, False, True), prototype.gates[0])
    run = V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))
    status = "passed" if gate.passed and config.run.evidence_tier == "acceptance" and run.dirty_patch_hash is None else "diagnostic"
    record = EvidenceRecordV2(2, "capacity", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, {"prototype_evidence": prototype.evidence_hash, "basis": artifact.artifact_hash}, (gate,), ("train",) if status == "passed" else ())
    _write_record(run, "capacity_evidence.json", record)
    artifact.save(str(run.root / "code_basis.pt"))
    print(run.root)
    return 0


def _warm_start_prototype(bundle: StudentBundle, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("prototype checkpoint is invalid")
    allowed = ("generator.template_logits", "system.encoder.phase", "system.decoder.phase")
    target = bundle.state_dict()
    for key in allowed:
        if key not in state or key not in target or state[key].shape != target[key].shape:
            raise ValueError(f"prototype warm-start lacks compatible {key}")
        target[key].copy_(state[key])
    bundle.load_state_dict(target, strict=True)


def _train(args: argparse.Namespace, config: V2Config) -> int:
    if config.student_training is None:
        raise ValueError("student train requires student_training config")
    prototype = _record(args.prototype_evidence)
    capacity = _record(args.capacity_evidence)
    require_passed_evidence(prototype, kind="prototype")
    require_passed_evidence(capacity, kind="capacity")
    device = torch.device(config.run.device)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    bundle = StudentBundle(config, basis).to(device)
    _warm_start_prototype(bundle, args.prototype_checkpoint)
    summary = train_code_conditioned(bundle.system, bundle.generator, _tensor(args.codes, device=device), _tensor(args.labels, device=device).long(), _tensor(args.teacher_targets, device=device), exposure_scale=args.exposure_scale, prototype_evidence=prototype, capacity_evidence=capacity, expected_basis_hash=basis.artifact_hash, epochs=config.student_training.epochs, learning_rate=config.student_training.learning_rate)
    run = V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))
    manifest = _bundle_manifest(config, bundle, dataset_hash=canonical_sha256(str(Path(args.teacher_targets).resolve())), calibration_hash=basis.artifact_hash)
    save_exact_checkpoint(run.root / "student_bundle.pt", bundle, manifest, extra={"summary": summary.__dict__, "basis_artifact_hash": basis.artifact_hash})
    record = EvidenceRecordV2(2, "code_training", "diagnostic", "diagnostic", run.git_commit, run.dirty_patch_hash, run.config_hash, {"prototype": prototype.evidence_hash, "capacity": capacity.evidence_hash, "basis": basis.artifact_hash}, (GateResult("training_completed", True, {"loss_finite": torch.isfinite(torch.tensor(summary.loss)).item()}),), ())
    _write_record(run, "code_training_evidence.json", record)
    print(run.root)
    return 0


def _sample(args: argparse.Namespace, config: V2Config) -> int:
    device = torch.device(config.run.device)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = CheckpointManifestV2(**payload["manifest"])
    expected = _bundle_manifest(config, bundle, dataset_hash=manifest.dataset_hash, calibration_hash=basis.artifact_hash)
    load_exact_checkpoint(args.checkpoint, bundle, expected)
    result = bundle.system.forward_from_generator(bundle.generator, _tensor(args.codes, device=device), _tensor(args.labels, device=device).long())
    torch.save(result.raw_camera.tensor.detach().cpu(), args.output)
    return 0


def _evaluate(args: argparse.Namespace, config: V2Config) -> int:
    prediction = _tensor(args.predictions, device=torch.device("cpu"))
    target = _tensor(args.targets, device=torch.device("cpu"))
    metrics = {"corr": float(correlation(prediction, target).mean().item()), "dice": float(foreground_dice(prediction, target).mean().item()), "diagnostic_only": True}
    atomic_write_json(args.output, metrics)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ldm_project.v2")
    root = parser.add_subparsers(dest="domain", required=True)
    teacher = root.add_parser("teacher").add_subparsers(dest="action", required=True)
    for action, stage in (("train-first-stage", "first_stage"), ("train-diffusion", "diffusion"), ("export-targets", "teacher_distill_export")):
        command = teacher.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--legacy-train-config", required=True)
        command.set_defaults(handler=lambda args, stage=stage: _legacy_teacher(args, load_config(args.config), stage))
    manifest = teacher.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--split", default="train")
    manifest.set_defaults(handler=lambda args: _teacher_manifest(args, load_config(args.config)))
    student = root.add_parser("student").add_subparsers(dest="action", required=True)
    calibrate = student.add_parser("calibrate-prototypes")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--clean-targets", required=True)
    calibrate.add_argument("--manual-review")
    calibrate.add_argument("--prototype-accuracy", type=float, default=0.0)
    calibrate.set_defaults(handler=lambda args: _calibrate(args, load_config(args.config)))
    train = student.add_parser("train")
    for name in ("prototype_evidence", "capacity_evidence", "code_basis", "prototype_checkpoint", "codes", "labels", "teacher_targets"):
        train.add_argument("--" + name.replace("_", "-"), required=True)
    train.add_argument("--exposure-scale", type=float, required=True)
    train.add_argument("--config", required=True)
    train.set_defaults(handler=lambda args: _train(args, load_config(args.config)))
    sample = student.add_parser("sample")
    for name in ("code_basis", "checkpoint", "codes", "labels", "output"):
        sample.add_argument("--" + name.replace("_", "-"), required=True)
    sample.add_argument("--config", required=True)
    sample.set_defaults(handler=lambda args: _sample(args, load_config(args.config)))
    evaluate = student.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=lambda args: _evaluate(args, load_config(args.config)))
    audit = root.add_parser("audit").add_subparsers(dest="action", required=True)
    capacity = audit.add_parser("capacity")
    for name in ("prototype_evidence", "supports", "commands", "slm_intensity", "raw_camera"):
        capacity.add_argument("--" + name.replace("_", "-"), required=True)
    response_source = capacity.add_mutually_exclusive_group(required=True)
    response_source.add_argument("--response-matrices")
    response_source.add_argument("--prototype-commands")
    capacity.add_argument("--prototype-checkpoint")
    capacity.add_argument("--config", required=True)
    capacity.set_defaults(handler=lambda args: _capacity(args, load_config(args.config)))
    evidence = root.add_parser("evidence").add_subparsers(dest="action", required=True)
    summarize = evidence.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    summarize.set_defaults(handler=lambda args: print(Path(args.run).read_text(encoding="utf-8")) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = args.handler(args)
    return int(result or 0)
