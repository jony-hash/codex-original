"""Explicit v2 CLI with content-addressed admission boundaries; v1 remains untouched."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn

from ..config import V2Config, load_config
from ..evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, load_manual_review, require_evidence_action, tensor_artifact_ref, tensor_sha256
from ..optics import FormalAmplitudeFiniteCropIdentitySystem
from ..student.admission import CapacityMetrics, PrototypeMetrics, capacity_gate, prototype_gate
from ..student.capacity import CodeBasisArtifactV2, build_code_basis, capacity_metrics, finite_difference_response_matrices, replay_capacity
from ..student.checkpoints import CheckpointManifestV2, load_exact_checkpoint, load_prototype_warm_start, save_exact_checkpoint, state_content_hash, state_signature
from ..student.clean_targets import CleanTargetManifestV2, build_clean_targets
from ..student.evaluation import CodeAcceptanceMetrics, acceptance_metrics, code_acceptance_gate, paired_advantage_lower_bound
from ..student.generator import LabelPrototypeBasisGenerator
from ..student.training import calibrate_prototypes, train_code_conditioned
from ..teacher import TeacherPairManifestV2, TeacherTargetManifestV2, build_teacher_pair_manifest, build_teacher_target_manifest
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
    return EvidenceRecordV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _pair_manifest(path: str) -> TeacherPairManifestV2:
    manifest = TeacherPairManifestV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    manifest.validate()
    return manifest


def _target_manifest(path: str) -> TeacherTargetManifestV2:
    manifest = TeacherTargetManifestV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    manifest.validate()
    return manifest


def _validate_target_for_config(target: TeacherTargetManifestV2, config: V2Config) -> None:
    if config.teacher is None:
        raise ValueError("v2 student stages require a teacher configuration")
    pair = build_teacher_pair_manifest(config.teacher.first_stage_checkpoint, config.teacher.diffusion_checkpoint, target_plane=config.teacher.target_plane, seed=config.teacher.seed, guidance_scale=config.teacher.guidance_scale, split=target.split, latent_geometry=config.teacher.latent_geometry)
    if pair.manifest_hash != target.teacher_pair_manifest_hash:
        raise ValueError("teacher target manifest is not paired with current teacher checkpoints/config")
    target.validate(formal=config.system.is_formal)


def _write_record(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    record.validate()
    output = run.root / name
    atomic_write_json(output, record.to_dict())
    return output


def _system_signature(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def _context(config: V2Config, *, teacher_manifest_hash: str | None = None, dataset_hash: str | None = None, calibration_hash: str | None = None, checkpoint_hash: str | None = None, basis_hash: str | None = None) -> EvidenceContextV2:
    return EvidenceContextV2(_system_signature(config), canonical_sha256(config.model_dump(mode="json")), teacher_manifest_hash, dataset_hash, calibration_hash, checkpoint_hash, basis_hash)


def _require_clean_acceptance(config: V2Config) -> None:
    if config.run.evidence_tier != "acceptance":
        return
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    if dirty:
        raise ValueError("acceptance operations require a clean git commit")


def _new_run(config: V2Config) -> V2Run:
    _require_clean_acceptance(config)
    return V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))


def _zero_basis(shape: tuple[int, int]) -> CodeBasisArtifactV2:
    return CodeBasisArtifactV2(torch.zeros(10, 8, *shape), torch.ones(10, 1, 1), 8, 1234, "prototype-calibration-zero-basis")


def _bundle_manifest(config: V2Config, bundle: StudentBundle, *, dataset_hash: str, teacher_manifest_hash: str | None, calibration_hash: str, basis_hash: str) -> CheckpointManifestV2:
    state = bundle.state_dict()
    return CheckpointManifestV2(2, config.system.kind, config.system.boundary, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), dataset_hash, teacher_manifest_hash, calibration_hash, basis_hash, state_signature(state), state_content_hash(state))


def _memory_report(device: torch.device, *, baseline_reserved: int, baseline_allocated: int) -> tuple[bool, dict[str, float | bool]]:
    if device.type != "cuda":
        return False, {"cuda_measured": False}
    properties = torch.cuda.get_device_properties(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    delta = max(0, peak_reserved - baseline_reserved)
    free_after_model = max(1, properties.total_memory - baseline_reserved)
    passed = delta <= 0.85 * free_after_model and peak_reserved <= 0.90 * properties.total_memory
    return passed, {"cuda_measured": True, "allocated_baseline": float(baseline_allocated), "reserved_baseline": float(baseline_reserved), "allocated_peak": float(peak_allocated), "reserved_peak": float(peak_reserved), "reserved_delta_peak": float(delta), "total_memory": float(properties.total_memory)}


def _legacy_teacher(args: argparse.Namespace, config: V2Config, stage: str) -> int:
    command = [sys.executable, "train.py", "--train-config", args.legacy_train_config, "--stage", stage, "--device", config.run.device]
    subprocess.run(command, check=True)
    return 0


def _teacher_manifest(args: argparse.Namespace, config: V2Config) -> int:
    if config.teacher is None:
        raise ValueError("teacher command requires teacher config")
    run = _new_run(config)
    manifest = build_teacher_pair_manifest(config.teacher.first_stage_checkpoint, config.teacher.diffusion_checkpoint, target_plane=config.teacher.target_plane, seed=config.teacher.seed, guidance_scale=config.teacher.guidance_scale, split=args.split, latent_geometry=config.teacher.latent_geometry)
    atomic_write_json(run.root / "teacher_pair_manifest.json", manifest.to_dict() | {"manifest_hash": manifest.manifest_hash})
    print(run.root / "teacher_pair_manifest.json")
    return 0


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as handle:
        return Path(handle.name)


def _teacher_export(args: argparse.Namespace, config: V2Config) -> int:
    if config.teacher is None:
        raise ValueError("teacher export requires teacher config")
    pair = _pair_manifest(args.teacher_manifest)
    if pair.first_stage_path != config.teacher.first_stage_checkpoint or pair.diffusion_path != config.teacher.diffusion_checkpoint or pair.seed != config.teacher.seed or pair.guidance_scale != config.teacher.guidance_scale:
        raise ValueError("teacher pair manifest does not match v2 teacher configuration")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("teacher target output already exists")
    temporary = _temporary_output(output)
    try:
        command = [sys.executable, "train.py", "--train-config", args.legacy_train_config, "--stage", "teacher_distill_export", "--device", config.run.device, "--first-stage-checkpoint", pair.first_stage_path, "--diffusion-checkpoint", pair.diffusion_path, "--seed", str(pair.seed), "--guidance-scale", str(pair.guidance_scale), "--distill-output", str(temporary)]
        subprocess.run(command, check=True)
        target = build_teacher_target_manifest(temporary, pair, formal=config.system.is_formal)
        os.replace(temporary, output)
        target = build_teacher_target_manifest(output, pair, formal=config.system.is_formal)
        target_path = Path(args.target_manifest)
        if target_path.exists():
            raise FileExistsError("teacher target manifest already exists")
        atomic_write_json(target_path, target.to_dict() | {"manifest_hash": target.manifest_hash})
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
    return 0


def _export_clean_targets(args: argparse.Namespace, config: V2Config) -> int:
    output, manifest_path = Path(args.output), Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("clean-target output and manifest paths must be new")
    run = _new_run(config)
    targets = build_clean_targets(dataset_root=args.dataset_root, image_size=config.system.slm_shape[0])
    if tuple(targets.shape) != (10, 1, *config.system.slm_shape):
        raise ValueError("clean-target geometry does not match the configured SLM plane")
    atomic_torch_save(output, targets)
    reference = tensor_artifact_ref(output, role="clean_mnist_targets", git_commit=run.git_commit)
    manifest = CleanTargetManifestV2(
        schema_version=2,
        dataset_root=str(Path(args.dataset_root).resolve()),
        dataset_split="train",
        samples_per_label=128,
        labels=tuple(range(10)),
        targets=reference,
        targets_hash=tensor_sha256(targets),
        shape=tuple(targets.shape),
        dtype=str(targets.dtype),
        config_hash=run.config_hash,
        git_commit=run.git_commit,
    )
    manifest.validate()
    atomic_write_json(manifest_path, manifest.to_dict() | {"manifest_hash": manifest.manifest_hash})
    print(output)
    return 0


def _calibrate(args: argparse.Namespace, config: V2Config) -> int:
    if config.prototype_calibration is None:
        raise ValueError("student calibrate-prototypes requires prototype_calibration config")
    device = torch.device(config.run.device)
    clean_targets = _tensor(args.clean_targets, device=device)
    run = _new_run(config)
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated, baseline_reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    else:
        baseline_allocated = baseline_reserved = 0
    summary = calibrate_prototypes(bundle.system, bundle.generator, clean_targets, exposure_scale=config.prototype_calibration.fixed_exposure_scale, epochs=config.prototype_calibration.epochs, learning_rate=config.prototype_calibration.learning_rate)
    memory_pass, memory = _memory_report(device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    labels = torch.arange(10, device=device)
    with torch.no_grad():
        raw = bundle.system.forward(bundle.system.command(bundle.generator.prototype_command(labels))).raw_camera.tensor.detach().cpu()
    raw_path = run.root / "prototype_raw_cmos.pt"
    atomic_torch_save(raw_path, raw)
    stable = summary.epoch_history[-1]["loss"] <= summary.epoch_history[-2]["loss"] and summary.epoch_history[-1]["corr"] >= summary.epoch_history[-2]["corr"] and summary.epoch_history[-1]["dice"] >= summary.epoch_history[-2]["dice"]
    metrics = PrototypeMetrics(summary.clean_corr, summary.clean_dice, summary.prototype_accuracy, summary.high_saturation, torch.isfinite(torch.tensor(summary.loss)).item() and torch.isfinite(raw).all().item(), summary.gradients_finite, memory_pass, stable)
    atomic_write_json(run.root / "prototype_measurements.json", {"summary": summary.__dict__, "memory": memory, "raw_tensor_sha256": tensor_sha256(raw)})
    checkpoint = run.root / "prototype_bundle.pt"
    manifest = _bundle_manifest(config, bundle, dataset_hash=file_sha256(args.clean_targets), teacher_manifest_hash=None, calibration_hash=basis.artifact_hash, basis_hash=basis.artifact_hash)
    save_exact_checkpoint(checkpoint, bundle, manifest, extra={"epoch_history": summary.epoch_history, "memory": memory, "clean_targets_hash": file_sha256(args.clean_targets), "prototype_raw_hash": tensor_sha256(raw)})
    checkpoint_hash = file_sha256(checkpoint)
    context = _context(config, dataset_hash=file_sha256(args.clean_targets), calibration_hash=basis.artifact_hash, checkpoint_hash=checkpoint_hash)
    status = "diagnostic" if not config.system.is_formal else "pending_review"
    actions = ("diagnostic_capacity_smoke",) if status == "diagnostic" else ("admit_prototypes",)
    record = EvidenceRecordV2(2, "prototype_calibration", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"clean_targets": file_sha256(args.clean_targets), "prototype_checkpoint": checkpoint_hash, "prototype_raw_cmos": file_sha256(raw_path)}, (prototype_gate(metrics, None),), actions)
    _write_record(run, "prototype_calibration_evidence.json", record)
    print(run.root)
    return 0


def _admit_prototypes(args: argparse.Namespace, config: V2Config) -> int:
    source = _record(args.calibration_evidence)
    source.validate()
    if source.evidence_kind != "prototype_calibration" or source.status not in {"pending_review", "diagnostic"}:
        raise ValueError("prototype admission requires a calibration record")
    if source.context.system_signature != _system_signature(config) or source.context.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("prototype calibration belongs to another v2 configuration")
    if file_sha256(args.prototype_checkpoint) != source.context.checkpoint_hash:
        raise ValueError("prototype checkpoint content does not match calibration evidence")
    device = torch.device(config.run.device)
    if file_sha256(args.clean_targets) != source.context.dataset_hash:
        raise ValueError("clean targets content does not match calibration evidence")
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(args.prototype_checkpoint, map_location="cpu", weights_only=False)
    actual = CheckpointManifestV2(**payload["manifest"])
    load_exact_checkpoint(args.prototype_checkpoint, bundle, actual)
    raw = bundle.system.forward(bundle.system.command(bundle.generator.prototype_command(torch.arange(10, device=device)))).raw_camera.tensor.detach().cpu()
    raw_hash = tensor_sha256(raw)
    saved_raw = _tensor(args.prototype_raw_cmos, device=torch.device("cpu"))
    if tensor_sha256(saved_raw) != raw_hash:
        raise ValueError("prototype replay does not match the saved raw-CMOS evidence")
    review, review_hash = load_manual_review(args.manual_review, kind="prototype", tensor_hash=raw_hash)
    measurement = json.loads(Path(args.measurements).read_text(encoding="utf-8"))
    summary = measurement["summary"]
    memory = measurement["memory"]
    stable = summary["epoch_history"][-1]["loss"] <= summary["epoch_history"][-2]["loss"] and summary["epoch_history"][-1]["corr"] >= summary["epoch_history"][-2]["corr"] and summary["epoch_history"][-1]["dice"] >= summary["epoch_history"][-2]["dice"]
    memory_pass = bool(memory.get("cuda_measured")) and bool(memory.get("reserved_delta_peak", float("inf")) <= 0.85 * max(1.0, float(memory.get("total_memory", 0.0)) - float(memory.get("reserved_baseline", 0.0)))) and bool(memory.get("reserved_peak", float("inf")) <= 0.90 * float(memory.get("total_memory", 0.0)))
    metrics = PrototypeMetrics(float(summary["clean_corr"]), float(summary["clean_dice"]), float(summary["prototype_accuracy"]), float(summary["high_saturation"]), bool(torch.isfinite(raw).all().item()), bool(summary["gradients_finite"]), memory_pass, stable)
    gate = prototype_gate(metrics, review.readable_labels)
    run = _new_run(config)
    diagnostic = not config.system.is_formal or config.run.evidence_tier == "diagnostic"
    status = "diagnostic" if diagnostic else ("passed" if gate.passed else "failed")
    actions = ("diagnostic_capacity_smoke",) if diagnostic else (("audit_capacity",) if status == "passed" else ())
    context = _context(config, dataset_hash=file_sha256(args.clean_targets), calibration_hash=source.context.calibration_hash, checkpoint_hash=file_sha256(args.prototype_checkpoint))
    record = EvidenceRecordV2(2, "prototype", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"calibration_evidence": source.evidence_hash, "prototype_checkpoint": file_sha256(args.prototype_checkpoint), "prototype_raw_cmos": file_sha256(args.prototype_raw_cmos), "manual_review": review_hash}, (gate,), actions, None if gate.passed else "prototype gate failed")
    _write_record(run, "prototype_evidence.json", record)
    print(run.root)
    return 0


def _capacity(args: argparse.Namespace, config: V2Config) -> int:
    if config.capacity_audit is None:
        raise ValueError("audit capacity requires capacity_audit config")
    prototype = _record(args.prototype_evidence)
    diagnostic = not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    action = "diagnostic_capacity_smoke" if diagnostic else "audit_capacity"
    require_evidence_action(prototype, kind="prototype", action=action, expected_context=_context(config, checkpoint_hash=file_sha256(args.prototype_checkpoint)), allow_diagnostic=diagnostic)
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    warm = load_prototype_warm_start(args.prototype_checkpoint, bundle, expected_system_signature=_system_signature(config))
    labels = torch.arange(10, device=device)
    prototype_commands = bundle.generator.prototype_command(labels).detach()
    supports = (prototype_commands[:, 0] >= 0.2).to(prototype_commands.dtype)
    audit = config.capacity_audit
    responses, unchanged, before_hash, after_hash = finite_difference_response_matrices(bundle.system, prototype_commands, supports, candidate_count=audit.dct_modes, epsilon=audit.finite_difference_epsilon, pixel_pitch_m=(config.system.slm_pixel_pitch_m, config.system.slm_pixel_pitch_m), max_cycles_per_mm=audit.max_cycles_per_mm)
    artifact = build_code_basis(responses, supports, candidate_count=audit.dct_modes, seed=audit.seed, source_hash=prototype.evidence_hash, pixel_pitch_m=(config.system.slm_pixel_pitch_m, config.system.slm_pixel_pitch_m), max_cycles_per_mm=audit.max_cycles_per_mm)
    commands, slm_intensity, camera, coefficients = replay_capacity(bundle.system, prototype_commands, artifact, supports, codes_per_label=audit.codes_per_label, command_gain=audit.command_gain)
    trace = bundle.system.forward(bundle.system.command(prototype_commands[:1]), return_optical_trace=True).optical_trace
    provenance_complete = trace is not None and all(field.provenance.upstream_plane is not None or field.provenance.operation == "ideal_amplitude_slm" for field in trace)
    slm_factor, camera_factor, support_energy, clipping = capacity_metrics(commands, slm_intensity, camera, supports)
    gate = capacity_gate(CapacityMetrics(slm_factor, camera_factor, support_energy, clipping, unchanged, False, provenance_complete), prototype.gates[0])
    run = _new_run(config)
    basis_path = run.root / "code_basis.pt"
    replay_path = run.root / "capacity_replay.pt"
    artifact.save(str(basis_path))
    atomic_torch_save(replay_path, {"unclamped_commands": commands.cpu(), "slm_intensity": slm_intensity.cpu(), "raw_camera": camera.cpu(), "coefficients": coefficients.cpu(), "supports": supports.cpu(), "responses": responses.cpu(), "parameter_before_hash": before_hash, "parameter_after_hash": after_hash, "warm_start": warm.to_dict()})
    status = "diagnostic" if diagnostic else ("passed" if gate.passed else "failed")
    actions = ("diagnostic_train_smoke",) if diagnostic else (("train",) if status == "passed" else ())
    context = _context(config, dataset_hash=prototype.context.dataset_hash, calibration_hash=prototype.context.calibration_hash, checkpoint_hash=file_sha256(args.prototype_checkpoint), basis_hash=artifact.artifact_hash)
    record = EvidenceRecordV2(2, "capacity", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"prototype_evidence": prototype.evidence_hash, "prototype_checkpoint": file_sha256(args.prototype_checkpoint), "basis": artifact.artifact_hash, "basis_file": file_sha256(basis_path), "replay": file_sha256(replay_path), "responses": tensor_sha256(responses), "supports": tensor_sha256(supports)}, (gate,), actions, None if gate.passed else "capacity gate failed")
    _write_record(run, "capacity_evidence.json", record)
    print(run.root)
    return 0


def _teacher_dataset(manifest: TeacherTargetManifestV2, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(manifest.dataset.path, map_location=device, weights_only=False)
    return payload["codes"].to(device), payload["class_labels"].to(device).long(), payload["teacher_camera"].to(device)


def _train(args: argparse.Namespace, config: V2Config) -> int:
    if config.student_training is None or config.prototype_calibration is None:
        raise ValueError("student train requires training and prototype config")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    prototype, capacity = _record(args.prototype_evidence), _record(args.capacity_evidence)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    diagnostic = not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    action = "diagnostic_train_smoke" if diagnostic else "train"
    require_evidence_action(prototype, kind="prototype", action="diagnostic_capacity_smoke" if diagnostic else "audit_capacity", expected_context=_context(config, checkpoint_hash=file_sha256(args.prototype_checkpoint)), allow_diagnostic=diagnostic)
    require_evidence_action(capacity, kind="capacity", action=action, expected_context=_context(config, calibration_hash=prototype.context.calibration_hash, checkpoint_hash=file_sha256(args.prototype_checkpoint), basis_hash=basis.artifact_hash), allow_diagnostic=diagnostic)
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, basis).to(device)
    warm = load_prototype_warm_start(args.prototype_checkpoint, bundle, expected_system_signature=_system_signature(config))
    codes, labels, targets = _teacher_dataset(target, device)
    settings = config.student_training
    summary = train_code_conditioned(bundle.system, bundle.generator, codes, labels, targets, exposure_scale=config.prototype_calibration.fixed_exposure_scale, prototype_evidence=prototype, capacity_evidence=capacity, expected_basis_hash=basis.artifact_hash, epochs=settings.epochs, learning_rate=settings.learning_rate, batch_size=settings.batch_size, weights={"prototype": settings.prototype_fidelity_weight, "paired_residual": settings.paired_teacher_residual_weight, "code_transport": settings.code_transport_weight, "clipping": settings.clipping_weight, "saturation": settings.saturation_weight, "phase_tv": settings.phase_tv_weight}, allow_diagnostic_evidence=diagnostic)
    run = _new_run(config)
    checkpoint = run.root / "student_bundle.pt"
    manifest = _bundle_manifest(config, bundle, dataset_hash=target.dataset_hash, teacher_manifest_hash=target.teacher_pair_manifest_hash, calibration_hash=prototype.context.calibration_hash or "", basis_hash=basis.artifact_hash)
    save_exact_checkpoint(checkpoint, bundle, manifest, extra={"summary": summary.__dict__, "basis_artifact_hash": basis.artifact_hash, "warm_start": warm.to_dict()})
    checkpoint_hash = file_sha256(checkpoint)
    status = "diagnostic" if diagnostic else "passed"
    actions = ("diagnostic_evaluate_smoke",) if diagnostic else ("evaluate",)
    context = _context(config, teacher_manifest_hash=target.teacher_pair_manifest_hash, dataset_hash=target.dataset_hash, calibration_hash=prototype.context.calibration_hash, checkpoint_hash=checkpoint_hash, basis_hash=basis.artifact_hash)
    record = EvidenceRecordV2(2, "code_training", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"prototype": prototype.evidence_hash, "capacity": capacity.evidence_hash, "teacher_dataset": target.dataset_hash, "student_checkpoint": checkpoint_hash}, (GateResult("training_completed", torch.isfinite(torch.tensor(summary.loss)).item(), {"loss": summary.loss, "paired_corr": summary.paired_corr, "paired_dice": summary.paired_dice, "code_variance": summary.code_variance, "clipping": summary.clipping_fraction}),), actions)
    _write_record(run, "code_training_evidence.json", record)
    print(run.root)
    return 0


def _load_student_bundle(config: V2Config, basis_path: str, checkpoint_path: str, *, device: torch.device) -> tuple[StudentBundle, CheckpointManifestV2]:
    basis = CodeBasisArtifactV2.load(basis_path)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual = CheckpointManifestV2(**payload["manifest"])
    if actual.system_signature != _system_signature(config) or actual.config_hash != canonical_sha256(config.model_dump(mode="json")) or actual.basis_hash != basis.artifact_hash:
        raise ValueError("student checkpoint does not match config/system/basis")
    load_exact_checkpoint(checkpoint_path, bundle, actual)
    return bundle, actual


def _sample(args: argparse.Namespace, config: V2Config) -> int:
    device = torch.device(config.run.device)
    bundle, _ = _load_student_bundle(config, args.code_basis, args.checkpoint, device=device)
    result = bundle.system.forward_from_generator(bundle.generator, _tensor(args.codes, device=device), _tensor(args.labels, device=device).long())
    atomic_torch_save(args.output, result.raw_camera.tensor.detach().cpu())
    return 0


def _evaluate(args: argparse.Namespace, config: V2Config) -> int:
    if config.prototype_calibration is None:
        raise ValueError("student evaluate requires prototype calibration config")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    training = _record(args.training_evidence)
    device = torch.device(config.run.device)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    checkpoint_hash = file_sha256(args.checkpoint)
    diagnostic = not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    require_evidence_action(training, kind="code_training", action="diagnostic_evaluate_smoke" if diagnostic else "evaluate", expected_context=_context(config, teacher_manifest_hash=target.teacher_pair_manifest_hash, dataset_hash=target.dataset_hash, checkpoint_hash=checkpoint_hash, basis_hash=basis.artifact_hash), allow_diagnostic=diagnostic)
    bundle, _ = _load_student_bundle(config, args.code_basis, args.checkpoint, device=device)
    codes, labels, teacher = _teacher_dataset(target, device)
    raw = bundle.system.forward_from_generator(bundle.generator, codes, labels).raw_camera.tensor
    normalized = (raw / config.prototype_calibration.fixed_exposure_scale).clamp(0.0, 1.0)
    camera_fraction, teacher_corr, teacher_dice = acceptance_metrics(raw, teacher, labels)
    lower_bound = paired_advantage_lower_bound(normalized, teacher, labels)
    run = _new_run(config)
    raw_path = run.root / "evaluation_raw_cmos.pt"
    atomic_torch_save(raw_path, raw.detach().cpu())
    raw_hash = tensor_sha256(raw.detach().cpu())
    review_labels = None
    review_hash = "missing"
    if args.manual_review:
        review, review_hash = load_manual_review(args.manual_review, kind="code_acceptance", tensor_hash=raw_hash)
        review_labels = review.readable_labels
    gate = code_acceptance_gate(CodeAcceptanceMetrics(camera_fraction, teacher_corr, teacher_dice, lower_bound, 1234, 10000), review_labels)
    if diagnostic:
        status, actions = "diagnostic", ()
    elif args.manual_review is None:
        status, actions = "pending_review", ()
    else:
        status, actions = ("passed", ("manual_promotion_review",)) if gate.passed else ("failed", ())
    context = _context(config, teacher_manifest_hash=target.teacher_pair_manifest_hash, dataset_hash=target.dataset_hash, calibration_hash=training.context.calibration_hash, checkpoint_hash=checkpoint_hash, basis_hash=basis.artifact_hash)
    record = EvidenceRecordV2(2, "code_acceptance", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"training_evidence": training.evidence_hash, "checkpoint": checkpoint_hash, "basis": basis.artifact_hash, "teacher_dataset": target.dataset_hash, "raw_cmos": file_sha256(raw_path), "manual_review": review_hash}, (gate,), actions, None if gate.passed else "code-conditioned acceptance gate is incomplete or failed")
    _write_record(run, "code_acceptance_evidence.json", record)
    atomic_write_json(args.output, record.to_dict())
    print(run.root)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ldm_project.v2")
    root = parser.add_subparsers(dest="domain", required=True)
    teacher = root.add_parser("teacher").add_subparsers(dest="action", required=True)
    for action, stage in (("train-first-stage", "first_stage"), ("train-diffusion", "diffusion")):
        command = teacher.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--legacy-train-config", required=True)
        command.set_defaults(handler=lambda args, stage=stage: _legacy_teacher(args, load_config(args.config), stage))
    export = teacher.add_parser("export-targets")
    for name in ("config", "legacy_train_config", "teacher_manifest", "output", "target_manifest"):
        export.add_argument("--" + name.replace("_", "-"), required=True)
    export.set_defaults(handler=lambda args: _teacher_export(args, load_config(args.config)))
    manifest = teacher.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--split", default="train")
    manifest.set_defaults(handler=lambda args: _teacher_manifest(args, load_config(args.config)))
    student = root.add_parser("student").add_subparsers(dest="action", required=True)
    calibrate = student.add_parser("calibrate-prototypes")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--clean-targets", required=True)
    calibrate.set_defaults(handler=lambda args: _calibrate(args, load_config(args.config)))
    clean = student.add_parser("export-clean-targets")
    for name in ("config", "dataset_root", "output", "manifest"):
        clean.add_argument("--" + name.replace("_", "-"), required=True)
    clean.set_defaults(handler=lambda args: _export_clean_targets(args, load_config(args.config)))
    admit = student.add_parser("admit-prototypes")
    for name in ("config", "calibration_evidence", "prototype_checkpoint", "prototype_raw_cmos", "measurements", "clean_targets", "manual_review"):
        admit.add_argument("--" + name.replace("_", "-"), required=True)
    admit.set_defaults(handler=lambda args: _admit_prototypes(args, load_config(args.config)))
    train = student.add_parser("train")
    for name in ("config", "prototype_evidence", "capacity_evidence", "code_basis", "prototype_checkpoint", "teacher_target_manifest"):
        train.add_argument("--" + name.replace("_", "-"), required=True)
    train.set_defaults(handler=lambda args: _train(args, load_config(args.config)))
    sample = student.add_parser("sample")
    for name in ("config", "code_basis", "checkpoint", "codes", "labels", "output"):
        sample.add_argument("--" + name.replace("_", "-"), required=True)
    sample.set_defaults(handler=lambda args: _sample(args, load_config(args.config)))
    evaluate = student.add_parser("evaluate")
    for name in ("config", "training_evidence", "code_basis", "checkpoint", "teacher_target_manifest", "output"):
        evaluate.add_argument("--" + name.replace("_", "-"), required=True)
    evaluate.add_argument("--manual-review")
    evaluate.set_defaults(handler=lambda args: _evaluate(args, load_config(args.config)))
    audit = root.add_parser("audit").add_subparsers(dest="action", required=True)
    capacity = audit.add_parser("capacity")
    for name in ("config", "prototype_evidence", "prototype_checkpoint"):
        capacity.add_argument("--" + name.replace("_", "-"), required=True)
    capacity.set_defaults(handler=lambda args: _capacity(args, load_config(args.config)))
    evidence = root.add_parser("evidence").add_subparsers(dest="action", required=True)
    summarize = evidence.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    summarize.set_defaults(handler=lambda args: print(Path(args.run).read_text(encoding="utf-8")) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args) or 0)
