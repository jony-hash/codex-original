"""Explicit v2 CLI with content-addressed admission boundaries; v1 remains untouched."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import V2Config, load_config
from ..evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, load_manual_review, require_evidence_action, tensor_artifact_ref, tensor_sha256
from ..optics import FormalAmplitudeFiniteCropIdentitySystem
from ..student.admission import CapacityMetrics, PrototypeMetrics, capacity_gate, prototype_gate, prototype_measurement_gate
from ..student.capacity import CodeBasisArtifactV2, build_code_basis, capacity_metrics, finite_difference_response_matrices, replay_capacity
from ..student.checkpoints import CheckpointManifestV2, load_exact_checkpoint, load_prototype_warm_start, save_exact_checkpoint, state_content_hash, state_signature
from ..student.clean_targets import CleanTargetManifestV2, build_clean_targets
from ..student.evaluation import CodeAcceptanceMetrics, acceptance_metrics, code_acceptance_gate, paired_advantage_lower_bound
from ..student.generator import LabelPrototypeBasisGenerator
from ..student.review_artifacts import write_review_artifacts
from ..student.training import _differentiable_correlation, calibrate_prototypes, train_code_conditioned
from ..student.capacity import supported_dct_candidates
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


def _clean_target_manifest(path: str, config: V2Config) -> CleanTargetManifestV2:
    manifest = CleanTargetManifestV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    manifest.validate()
    if manifest.system_contract_hash is not None:
        if manifest.system_contract_hash != _system_signature(config):
            raise ValueError("clean-target manifest belongs to another physical system")
    elif manifest.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("legacy clean-target manifest requires explicit migration")
    if manifest.shape != (10, 1, *config.system.slm_shape):
        raise ValueError("clean-target manifest geometry does not match the formal SLM plane")
    return manifest


def _validate_target_for_config(target: TeacherTargetManifestV2, config: V2Config) -> None:
    if config.teacher is None:
        raise ValueError("v2 student stages require a teacher configuration")
    pair = build_teacher_pair_manifest(config.teacher.first_stage_checkpoint, config.teacher.diffusion_checkpoint, target_plane=config.teacher.target_plane, seed=config.teacher.seed, guidance_scale=config.teacher.guidance_scale, split=target.split, latent_geometry=config.teacher.latent_geometry)
    if pair.manifest_hash != target.teacher_pair_manifest_hash:
        raise ValueError("teacher target manifest is not paired with current teacher checkpoints/config")
    if target.system_contract_hash is not None and target.system_contract_hash != _system_signature(config):
        raise ValueError("teacher target manifest belongs to another physical system")
    if target.teacher_export_contract_hash is None:
        if not config.system.is_formal:
            target.validate(formal=False)
            return
        raise ValueError("teacher target manifest requires explicit scoped migration")
    expected_export_hash = _teacher_export_contract_hash(config, pair, target.legacy_export_config_hash or "")
    if target.teacher_export_contract_hash != expected_export_hash:
        raise ValueError("teacher target manifest belongs to another teacher export contract")
    target.validate(formal=config.system.is_formal)


def _write_record(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    record.validate()
    output = run.root / name
    atomic_write_json(output, record.to_dict())
    return output


def _system_signature(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def _teacher_export_contract_hash(config: V2Config, pair: TeacherPairManifestV2, legacy_hash: str) -> str:
    if config.teacher is None:
        raise ValueError("teacher export contract requires teacher configuration")
    return canonical_sha256({
        "teacher_pair_manifest_hash": pair.manifest_hash,
        "legacy_export_config_hash": legacy_hash,
        "target_plane": config.teacher.target_plane,
        "latent_geometry": config.teacher.latent_geometry,
        "seed": config.teacher.seed,
        "guidance_scale": config.teacher.guidance_scale,
    })


def _context(config: V2Config, *, teacher_manifest_hash: str | None = None, dataset_hash: str | None = None, calibration_hash: str | None = None, checkpoint_hash: str | None = None, basis_hash: str | None = None) -> EvidenceContextV2:
    return EvidenceContextV2(_system_signature(config), canonical_sha256(config.model_dump(mode="json")), teacher_manifest_hash, dataset_hash, calibration_hash, checkpoint_hash, basis_hash)


def _require_clean_acceptance(config: V2Config) -> None:
    if config.run.evidence_tier != "acceptance":
        return
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("acceptance operations require a clean git commit")


def _legacy_export_contract(path: str, config: V2Config) -> str:
    from ldm_project.config import load_training_config

    legacy = load_training_config(path)
    expected = ("teacher_distill_export", 8, 100, 1234, 2026, 200, 4.0, tuple(range(10)), False)
    actual = (legacy.mode, legacy.one_shot.code_dim, legacy.distillation.samples_per_class, legacy.distillation.code_seed, legacy.distillation.code_projection_seed, legacy.distillation.sampling_steps, legacy.diffusion.guidance_scale, tuple(legacy.distillation.class_labels), legacy.data.download)
    if actual != expected or config.teacher is None or config.teacher.seed != 1234 or config.teacher.guidance_scale != 4.0:
        raise ValueError("formal teacher export requires the locked 8D/100-per-label legacy export contract")
    return file_sha256(path)


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


def _preflight_stage(name: str, operation, *, device: torch.device, baseline_reserved: int, baseline_allocated: int) -> tuple[GateResult, dict[str, float | bool]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_pass, memory = _memory_report(device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    memory["elapsed_seconds"] = time.perf_counter() - started
    return GateResult(f"preflight_{name}", memory_pass, memory, None if memory_pass else "CUDA memory gate was not measured or failed"), memory


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
    _require_clean_acceptance(config)
    legacy_hash = _legacy_export_contract(args.legacy_train_config, config)
    pair = _pair_manifest(args.teacher_manifest)
    if pair.first_stage_path != config.teacher.first_stage_checkpoint or pair.diffusion_path != config.teacher.diffusion_checkpoint or pair.seed != config.teacher.seed or pair.guidance_scale != config.teacher.guidance_scale:
        raise ValueError("teacher pair manifest does not match v2 teacher configuration")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("teacher target output already exists")
    temporary = _temporary_output(output)
    try:
        legacy_output_root = output.parent.parent
        command = [sys.executable, "train.py", "--train-config", args.legacy_train_config, "--stage", "teacher_distill_export", "--device", config.run.device, "--output-root", str(legacy_output_root), "--first-stage-checkpoint", pair.first_stage_path, "--diffusion-checkpoint", pair.diffusion_path, "--seed", str(pair.seed), "--guidance-scale", str(pair.guidance_scale), "--distill-output", str(temporary)]
        subprocess.run(command, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        scoped_export_hash = _teacher_export_contract_hash(config, pair, legacy_hash)
        target = build_teacher_target_manifest(temporary, pair, formal=config.system.is_formal, generated_git_commit=commit, v2_config_hash=canonical_sha256(config.model_dump(mode="json")), legacy_export_config_hash=legacy_hash, system_contract_hash=_system_signature(config), teacher_export_contract_hash=scoped_export_hash)
        os.replace(temporary, output)
        target = build_teacher_target_manifest(output, pair, formal=config.system.is_formal, generated_git_commit=commit, v2_config_hash=canonical_sha256(config.model_dump(mode="json")), legacy_export_config_hash=legacy_hash, system_contract_hash=_system_signature(config), teacher_export_contract_hash=scoped_export_hash)
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
        system_contract_hash=_system_signature(config),
        producer_config_hash=run.config_hash,
    )
    manifest.validate()
    atomic_write_json(manifest_path, manifest.to_dict() | {"manifest_hash": manifest.manifest_hash})
    print(output)
    return 0


def _calibrate(args: argparse.Namespace, config: V2Config) -> int:
    if config.prototype_calibration is None:
        raise ValueError("student calibrate-prototypes requires prototype_calibration config")
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    clean_targets = _tensor(clean_manifest.targets.path, device=device)
    run = _new_run(config)
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated, baseline_reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    else:
        baseline_allocated = baseline_reserved = 0
    summary = calibrate_prototypes(
        bundle.system,
        bundle.generator,
        clean_targets,
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
        epochs=config.prototype_calibration.epochs,
        learning_rate=config.prototype_calibration.learning_rate,
        microbatch_size=config.execution.prototype_microbatch_size,
        activation_checkpoint=config.execution.activation_checkpoint,
        initialization=config.prototype_calibration.initialization,
        optimizer_steps_per_epoch=config.prototype_calibration.optimizer_steps_per_epoch,
        correlation_weight=config.prototype_calibration.prototype_correlation_weight,
        dice_weight=config.prototype_calibration.prototype_dice_weight,
        ce_weight=config.prototype_calibration.prototype_ce_weight,
        ce_temperature=config.prototype_calibration.prototype_ce_temperature,
    )
    memory_pass, memory = _memory_report(device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    raw = _stream_prototype_raw(bundle, device=device, microbatch_size=config.execution.prototype_microbatch_size)
    raw_path = run.root / "prototype_raw_cmos.pt"
    atomic_torch_save(raw_path, raw)
    review_artifacts = write_review_artifacts(root=run.root, stem="prototype", kind="prototype", raw_cmos=raw, targets=clean_targets.detach().cpu(), exposure_scale=config.prototype_calibration.fixed_exposure_scale)
    stable = summary.epoch_history[-1]["loss"] <= summary.epoch_history[-2]["loss"] and summary.epoch_history[-1]["corr"] >= summary.epoch_history[-2]["corr"] and summary.epoch_history[-1]["dice"] >= summary.epoch_history[-2]["dice"]
    metrics = PrototypeMetrics(summary.clean_corr, summary.clean_dice, summary.prototype_accuracy, summary.high_saturation, torch.isfinite(torch.tensor(summary.loss)).item() and torch.isfinite(raw).all().item(), summary.gradients_finite, memory_pass, stable)
    atomic_write_json(run.root / "prototype_measurements.json", {"summary": summary.__dict__, "memory": memory, "raw_tensor_sha256": tensor_sha256(raw)})
    checkpoint = run.root / "prototype_bundle.pt"
    manifest = _bundle_manifest(config, bundle, dataset_hash=clean_manifest.targets.sha256, teacher_manifest_hash=None, calibration_hash=basis.artifact_hash, basis_hash=basis.artifact_hash)
    save_exact_checkpoint(checkpoint, bundle, manifest, extra={"epoch_history": summary.epoch_history, "memory": memory, "clean_targets_hash": clean_manifest.targets.sha256, "clean_target_manifest_hash": clean_manifest.manifest_hash, "prototype_raw_hash": tensor_sha256(raw)})
    checkpoint_hash = file_sha256(checkpoint)
    context = _context(config, dataset_hash=clean_manifest.targets.sha256, calibration_hash=basis.artifact_hash, checkpoint_hash=checkpoint_hash)
    measurement_gate = prototype_measurement_gate(metrics)
    diagnostic = not config.system.is_formal or config.run.evidence_tier == "diagnostic"
    status = "diagnostic" if diagnostic else ("pending_review" if measurement_gate.passed else "failed")
    actions = ("diagnostic_capacity_smoke",) if diagnostic and measurement_gate.passed else (("admit_prototypes",) if status == "pending_review" else ())
    record = EvidenceRecordV2(2, "prototype_calibration", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"clean_targets": clean_manifest.targets.sha256, "clean_target_manifest": clean_manifest.manifest_hash, "prototype_checkpoint": checkpoint_hash, "prototype_raw_cmos": file_sha256(raw_path), "prototype_preview": review_artifacts.preview_sha256}, (measurement_gate,), actions, None if measurement_gate.passed else "prototype automatic measurement gate failed")
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
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    if clean_manifest.targets.sha256 != source.context.dataset_hash or clean_manifest.manifest_hash != source.input_hashes.get("clean_target_manifest"):
        raise ValueError("clean targets content does not match calibration evidence")
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(args.prototype_checkpoint, map_location="cpu", weights_only=False)
    actual = CheckpointManifestV2(**payload["manifest"])
    load_exact_checkpoint(args.prototype_checkpoint, bundle, actual)
    raw = _stream_prototype_raw(bundle, device=device, microbatch_size=config.execution.prototype_microbatch_size)
    raw_hash = tensor_sha256(raw)
    saved_raw = _tensor(args.prototype_raw_cmos, device=torch.device("cpu"))
    if tensor_sha256(saved_raw) != raw_hash:
        raise ValueError("prototype replay does not match the saved raw-CMOS evidence")
    preview_hash = file_sha256(args.prototype_preview)
    if preview_hash != source.input_hashes.get("prototype_preview"):
        raise ValueError("prototype preview content does not match calibration evidence")
    review, review_hash = load_manual_review(args.manual_review, kind="prototype", tensor_hash=raw_hash, preview_hash=preview_hash)
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
    context = _context(config, dataset_hash=clean_manifest.targets.sha256, calibration_hash=source.context.calibration_hash, checkpoint_hash=file_sha256(args.prototype_checkpoint))
    record = EvidenceRecordV2(2, "prototype", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"calibration_evidence": source.evidence_hash, "clean_target_manifest": clean_manifest.manifest_hash, "prototype_checkpoint": file_sha256(args.prototype_checkpoint), "prototype_raw_cmos": file_sha256(args.prototype_raw_cmos), "prototype_preview": preview_hash, "manual_review": review_hash}, (gate,), actions, None if gate.passed else "prototype gate failed")
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
    responses, unchanged, before_hash, after_hash = finite_difference_response_matrices(bundle.system, prototype_commands, supports, candidate_count=audit.dct_modes, epsilon=audit.finite_difference_epsilon, pixel_pitch_m=(config.system.slm_pixel_pitch_m, config.system.slm_pixel_pitch_m), max_cycles_per_mm=audit.max_cycles_per_mm, microbatch_size=config.execution.capacity_microbatch_size)
    artifact = build_code_basis(responses, supports.cpu(), candidate_count=audit.dct_modes, seed=audit.seed, source_hash=prototype.evidence_hash, pixel_pitch_m=(config.system.slm_pixel_pitch_m, config.system.slm_pixel_pitch_m), max_cycles_per_mm=audit.max_cycles_per_mm)
    commands, slm_intensity, camera, coefficients = replay_capacity(bundle.system, prototype_commands, artifact, supports, codes_per_label=audit.codes_per_label, command_gain=audit.command_gain, microbatch_size=config.execution.capacity_microbatch_size)
    trace = bundle.system.forward(bundle.system.command(prototype_commands[:1]), return_optical_trace=True).optical_trace
    provenance_complete = trace is not None and all(field.provenance.upstream_plane is not None or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"} for field in trace)
    slm_factor, camera_factor, support_energy, clipping = capacity_metrics(commands, slm_intensity, camera, supports.cpu())
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


def _teacher_dataset(manifest: TeacherTargetManifestV2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(manifest.dataset.path, map_location="cpu", weights_only=False)
    return payload["codes"].cpu(), payload["class_labels"].cpu().long(), payload["teacher_camera"].cpu()


@torch.no_grad()
def _stream_raw(bundle: StudentBundle, codes: torch.Tensor, labels: torch.Tensor, *, device: torch.device, microbatch_size: int) -> torch.Tensor:
    if microbatch_size < 1:
        raise ValueError("evaluation microbatch must be positive")
    rows = []
    for start in range(0, codes.shape[0], microbatch_size):
        result = bundle.system.forward_from_generator(bundle.generator, codes[start : start + microbatch_size].to(device), labels[start : start + microbatch_size].to(device))
        rows.append(result.raw_camera.tensor.detach().cpu())
    return torch.cat(rows)


@torch.no_grad()
def _stream_prototype_raw(bundle: StudentBundle, *, device: torch.device, microbatch_size: int) -> torch.Tensor:
    labels = torch.arange(10, device=device)
    rows = []
    for start in range(0, labels.numel(), microbatch_size):
        command = bundle.generator.prototype_command(labels[start : start + microbatch_size])
        rows.append(bundle.system.forward(bundle.system.command(command)).raw_camera.tensor.detach().cpu())
    return torch.cat(rows)


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
    codes, labels, targets = _teacher_dataset(target)
    settings = config.student_training
    summary = train_code_conditioned(bundle.system, bundle.generator, codes, labels, targets, exposure_scale=config.prototype_calibration.fixed_exposure_scale, prototype_evidence=prototype, capacity_evidence=capacity, expected_basis_hash=basis.artifact_hash, epochs=settings.epochs, learning_rate=settings.learning_rate, batch_size=settings.batch_size, weights={"prototype": settings.prototype_fidelity_weight, "paired_residual": settings.paired_teacher_residual_weight, "code_transport": settings.code_transport_weight, "clipping": settings.clipping_weight, "saturation": settings.saturation_weight, "phase_tv": settings.phase_tv_weight}, microbatch_size=config.execution.training_microbatch_size, activation_checkpoint=config.execution.activation_checkpoint, allow_diagnostic_evidence=diagnostic)
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
    codes, labels, teacher = _teacher_dataset(target)
    raw = _stream_raw(bundle, codes, labels, device=device, microbatch_size=config.execution.evaluation_microbatch_size)
    normalized = (raw / config.prototype_calibration.fixed_exposure_scale).clamp(0.0, 1.0)
    camera_fraction, teacher_corr, teacher_dice = acceptance_metrics(raw, teacher, labels)
    lower_bound = paired_advantage_lower_bound(normalized, teacher, labels)
    run = _new_run(config)
    raw_path = run.root / "evaluation_raw_cmos.pt"
    atomic_torch_save(raw_path, raw.detach().cpu())
    review_artifacts = write_review_artifacts(root=run.root, stem="evaluation", kind="code_acceptance", raw_cmos=raw, labels=labels, targets=teacher, exposure_scale=config.prototype_calibration.fixed_exposure_scale)
    gate = code_acceptance_gate(CodeAcceptanceMetrics(camera_fraction, teacher_corr, teacher_dice, lower_bound, 1234, 10000), None)
    if diagnostic:
        status, actions = "diagnostic", ()
    else:
        status, actions = "pending_review", ("admit_evaluation",)
    context = _context(config, teacher_manifest_hash=target.teacher_pair_manifest_hash, dataset_hash=target.dataset_hash, calibration_hash=training.context.calibration_hash, checkpoint_hash=checkpoint_hash, basis_hash=basis.artifact_hash)
    record = EvidenceRecordV2(2, "code_acceptance", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"training_evidence": training.evidence_hash, "checkpoint": checkpoint_hash, "basis": basis.artifact_hash, "teacher_dataset": target.dataset_hash, "raw_cmos": file_sha256(raw_path), "evaluation_preview": review_artifacts.preview_sha256}, (gate,), actions, "manual review is required before code acceptance")
    _write_record(run, "code_acceptance_evidence.json", record)
    atomic_write_json(args.output, record.to_dict())
    print(run.root)
    return 0


def _admit_evaluation(args: argparse.Namespace, config: V2Config) -> int:
    source = _record(args.evaluation_evidence)
    if source.evidence_kind != "code_acceptance" or source.status not in {"pending_review", "diagnostic"}:
        raise ValueError("evaluation admission requires pending evaluation evidence")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(config, teacher_manifest_hash=target.teacher_pair_manifest_hash, dataset_hash=target.dataset_hash, checkpoint_hash=checkpoint_hash, basis_hash=basis.artifact_hash)
    source.context.assert_matches(expected)
    device = torch.device(config.run.device)
    _load_student_bundle(config, args.code_basis, args.checkpoint, device=device)
    raw = _tensor(args.evaluation_raw_cmos, device=torch.device("cpu"))
    raw_hash = tensor_sha256(raw)
    if file_sha256(args.evaluation_raw_cmos) != source.input_hashes.get("raw_cmos"):
        raise ValueError("evaluation raw-CMOS file does not match pending evidence")
    preview_hash = file_sha256(args.evaluation_preview)
    if preview_hash != source.input_hashes.get("evaluation_preview"):
        raise ValueError("evaluation preview does not match pending evidence")
    review, review_hash = load_manual_review(args.manual_review, kind="code_acceptance", tensor_hash=raw_hash, preview_hash=preview_hash)
    gate = source.gates[0]
    completed = code_acceptance_gate(CodeAcceptanceMetrics(float(gate.metrics["camera_code_fraction"]), float(gate.metrics["teacher_corr"]), float(gate.metrics["teacher_dice"]), float(gate.metrics["paired_advantage_lower_bound"]), 1234, 10000), review.readable_labels)
    run = _new_run(config)
    diagnostic = not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    status = "diagnostic" if diagnostic else ("passed" if completed.passed else "failed")
    actions = () if diagnostic else (("manual_promotion_review",) if status == "passed" else ())
    record = EvidenceRecordV2(2, "code_acceptance", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, source.context, {**source.input_hashes, "manual_review": review_hash}, (completed,), actions, None if completed.passed else "code-conditioned acceptance gate failed")
    _write_record(run, "code_acceptance_evidence.json", record)
    atomic_write_json(args.output, record.to_dict())
    print(run.root)
    return 0


def _preflight(args: argparse.Namespace, config: V2Config) -> int:
    """Bounded no-update physical execution probe; it is diagnostic by definition."""
    device = torch.device(config.run.device)
    run = _new_run(config)
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated, baseline_reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    else:
        baseline_allocated = baseline_reserved = 0
    before = state_content_hash(bundle.state_dict())
    stages: list[GateResult] = []
    reports: dict[str, dict[str, float | bool]] = {}
    label = torch.zeros(1, dtype=torch.long, device=device)

    def prototype() -> None:
        bundle.zero_grad(set_to_none=True)
        command = bundle.generator(torch.zeros(1, 8, device=device), label)
        raw = checkpoint(lambda value: bundle.system.forward(bundle.system.command(value), return_optical_trace=True).raw_camera.tensor, command, use_reentrant=False)
        raw.square().mean().backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in bundle.parameters()):
            raise ValueError("prototype preflight produced non-finite gradients")
        bundle.zero_grad(set_to_none=True)

    gate, report = _preflight_stage("prototype_backward", prototype, device=device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    stages.append(gate)
    reports["prototype_backward"] = report

    def capacity() -> None:
        with torch.no_grad():
            base = bundle.generator.prototype_command(label)
            support = torch.ones(*config.system.slm_shape, device=device)
            direction = supported_dct_candidates(support, candidate_count=8, pixel_pitch_m=(config.system.slm_pixel_pitch_m, config.system.slm_pixel_pitch_m), max_cycles_per_mm=config.capacity_audit.max_cycles_per_mm if config.capacity_audit else 20.0).directions[0].unsqueeze(0).unsqueeze(0)
            epsilon = config.capacity_audit.finite_difference_epsilon if config.capacity_audit else 1.0e-3
            plus = bundle.system.forward(bundle.system.command((base + epsilon * direction).clamp(0.0, 1.0))).raw_camera.tensor
            minus = bundle.system.forward(bundle.system.command((base - epsilon * direction).clamp(0.0, 1.0))).raw_camera.tensor
            if not torch.isfinite((plus - minus) / (2.0 * epsilon)).all():
                raise ValueError("capacity preflight response is non-finite")

    gate, report = _preflight_stage("capacity_pair", capacity, device=device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    stages.append(gate)
    reports["capacity_pair"] = report

    def training() -> None:
        bundle.zero_grad(set_to_none=True)
        labels = torch.arange(10, device=device).repeat_interleave(2)
        codes = torch.randn(20, 8, device=device)
        values = []
        for start in range(0, 20, config.execution.training_microbatch_size):
            command = bundle.generator(codes[start : start + config.execution.training_microbatch_size], labels[start : start + config.execution.training_microbatch_size])
            values.append(checkpoint(lambda value: bundle.system.forward(bundle.system.command(value)).raw_camera.tensor, command, use_reentrant=False))
        torch.cat(values).square().mean().backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in bundle.parameters()):
            raise ValueError("training preflight produced non-finite gradients")
        bundle.zero_grad(set_to_none=True)

    gate, report = _preflight_stage("logical_training_backward", training, device=device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    stages.append(gate)
    reports["logical_training_backward"] = report

    def evaluation() -> None:
        with torch.no_grad():
            bundle.system.forward_from_generator(bundle.generator, torch.zeros(1, 8, device=device), label)

    gate, report = _preflight_stage("evaluation_microbatch", evaluation, device=device, baseline_reserved=baseline_reserved, baseline_allocated=baseline_allocated)
    stages.append(gate)
    reports["evaluation_microbatch"] = report
    after = state_content_hash(bundle.state_dict())
    trace = bundle.system.forward(bundle.system.command(bundle.generator.prototype_command(label)), return_optical_trace=True).optical_trace
    provenance_complete = trace is not None and all(field.provenance.upstream_plane is not None or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"} for field in trace)
    unchanged = before == after
    stages.append(GateResult("preflight_parameter_unchanged", unchanged, {"before_after_equal": unchanged}, None if unchanged else "preflight modified parameters"))
    stages.append(GateResult("preflight_provenance", provenance_complete, {"provenance_complete": provenance_complete}, None if provenance_complete else "optical trace is incomplete"))
    record = EvidenceRecordV2(2, "formal_execution_preflight", "diagnostic", "diagnostic", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, calibration_hash=basis.artifact_hash, basis_hash=basis.artifact_hash), {}, tuple(stages), (), None if all(gate.passed for gate in stages) else "formal execution preflight did not pass")
    _write_record(run, "formal_execution_preflight_evidence.json", record)
    atomic_write_json(run.root / "formal_execution_preflight_memory.json", reports)
    print(run.root)
    return 0


def _reachability(args: argparse.Namespace, config: V2Config) -> int:
    if config.prototype_calibration is None:
        raise ValueError("prototype reachability requires prototype_calibration config")
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    clean = _tensor(clean_manifest.targets.path, device=torch.device(config.run.device))
    device = torch.device(config.run.device)
    failed_payload = torch.load(args.failed_prototype_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(failed_payload, dict) or "state_dict" not in failed_payload:
        raise ValueError("failed prototype checkpoint payload is invalid")
    labels = torch.arange(10, device=device)

    def bundle_with(kind: str) -> StudentBundle:
        bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
        if kind == "failed_replay":
            bundle.load_state_dict(failed_payload["state_dict"], strict=True)
        elif kind == "clean_intensity_to_amplitude":
            with torch.no_grad():
                amplitude = clean[:, 0].sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
                bundle.generator.template_logits.copy_(torch.logit(amplitude))
        elif kind != "uniform_half":
            raise ValueError(f"unknown reachability state: {kind}")
        return bundle

    def replay(bundle: StudentBundle, *, trace: bool = False) -> tuple[torch.Tensor, tuple[object, ...] | None]:
        rows, traces = [], None
        with torch.no_grad():
            for label in labels:
                result = bundle.system.forward(bundle.system.command(bundle.generator.prototype_command(label[None])), return_optical_trace=trace)
                rows.append(result.raw_camera.tensor)
                if trace and traces is None:
                    traces = [[] for _ in result.optical_trace or ()]
                if trace and result.optical_trace is not None:
                    for bucket, field in zip(traces, result.optical_trace):
                        bucket.append(field)
        return torch.cat(rows), tuple(torch.cat([field.tensor.abs().square().cpu() for field in bucket]) for bucket in traces) if traces is not None else None

    def camera_metrics(raw: torch.Tensor) -> dict[str, float]:
        normalized = (raw / config.prototype_calibration.fixed_exposure_scale).clamp(0.0, 1.0)
        matrix = _differentiable_correlation(normalized[:, None].expand(10, 10, *normalized.shape[1:]).reshape(100, *normalized.shape[1:]), clean[None].expand(10, 10, *clean.shape[1:]).reshape(100, *clean.shape[1:])).reshape(10, 10)
        dice = (2.0 * (normalized * clean).flatten(1).sum(dim=1) + 1.0e-6) / (normalized.square().flatten(1).sum(dim=1) + clean.square().flatten(1).sum(dim=1) + 1.0e-6)
        margin = matrix.diag() - (matrix + torch.eye(10, device=device) * -2.0).max(dim=1).values
        return {"clean_corr": float(_differentiable_correlation(normalized, clean).mean().item()), "clean_dice_soft": float(dice.mean().item()), "prototype_accuracy": float((matrix.argmax(dim=1) == labels).to(torch.float32).mean().item()), "min_target_margin": float(margin.min().item()), "high_saturation": float((raw / config.prototype_calibration.fixed_exposure_scale > 1.0).to(torch.float32).mean().item())}

    states = {}
    for kind in ("uniform_half", "failed_replay", "clean_intensity_to_amplitude"):
        bundle = bundle_with(kind)
        raw, trace = replay(bundle, trace=True)
        states[kind] = {"metrics": camera_metrics(raw), "trace_plane_ids": [field.plane.plane_id for field in bundle.system.forward(bundle.system.command(bundle.generator.prototype_command(labels[:1])), return_optical_trace=True).optical_trace or ()], "trace": trace}

    clean_bundle = bundle_with("clean_intensity_to_amplitude")
    before = state_content_hash(clean_bundle.state_dict())
    template = clean_bundle.generator.template_logits
    slm_error = float((clean_bundle.generator.prototype_command(labels).square() - clean).square().max().sqrt().item())
    raw, trace = replay(clean_bundle, trace=True)
    direction_metrics: dict[str, dict[str, float | bool]] = {}
    groups = {
        "template": [template],
        "encoder": [clean_bundle.system.encoder.phase],
        "decoder": [clean_bundle.system.decoder.phase],
        "joint": [template, clean_bundle.system.encoder.phase, clean_bundle.system.decoder.phase],
    }
    base_margin = camera_metrics(raw)["min_target_margin"]
    for name, parameters in groups.items():
        clean_bundle.zero_grad(set_to_none=True)
        current_rows = []
        for label in labels:
            current_rows.append(clean_bundle.system.forward(clean_bundle.system.command(clean_bundle.generator.prototype_command(label[None]))).raw_camera.tensor)
        current_raw = torch.cat(current_rows)
        normalized = (current_raw / config.prototype_calibration.fixed_exposure_scale).clamp(0.0, 1.0)
        loss = torch.nn.functional.mse_loss(normalized, clean) + 0.25 * (1.0 - _differentiable_correlation(normalized, clean).mean())
        loss.backward()
        finite = all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in parameters)
        snapshots = [parameter.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(-1.0e-3 * parameter.grad / parameter.grad.square().mean().sqrt().clamp_min(1.0e-12))
        candidate_raw, _ = replay(clean_bundle)
        improvement = camera_metrics(candidate_raw)["min_target_margin"] - base_margin
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
        direction_metrics[name] = {"gradient_finite": bool(finite), "margin_improvement": float(improvement), "margin_improved": bool(improvement > 0.0)}
    after = state_content_hash(clean_bundle.state_dict())
    crop_values = []
    if trace is not None:
        for plane_id, bucket in zip(states["clean_intensity_to_amplitude"]["trace_plane_ids"], trace):
            if "layer_" in plane_id and "propagated" not in plane_id:
                crop_values.append(float(bucket.square().mean().item()))
    front_retention = 1.0
    if trace is not None and len(trace) >= 2:
        base = trace[0].flatten(1)
        front = trace[1].flatten(1)
        base = base / base.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
        front = front / front.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
        front_retention = float((front @ front.T).abs().mean().item() / (base @ base.T).abs().mean().clamp_min(1.0e-12).item())
    passed = slm_error <= 1.0e-6 and front_retention >= 0.95 and all(metric["gradient_finite"] for metric in direction_metrics.values()) and any(metric["margin_improved"] for metric in direction_metrics.values()) and before == after
    run = _new_run(config)
    evidence = {"schema_version": 2, "evidence_kind": "prototype_reachability", "status": "diagnostic", "tier": "diagnostic", "git_commit": run.git_commit, "config_hash": run.config_hash, "input_hashes": {"clean_target_manifest": clean_manifest.manifest_hash, "failed_prototype_checkpoint": file_sha256(args.failed_prototype_checkpoint)}, "states": {name: {"metrics": value["metrics"], "trace_plane_ids": value["trace_plane_ids"]} for name, value in states.items()}, "slm_intensity_max_error": slm_error, "front_retention": front_retention, "crop_statistics": crop_values, "direction_metrics": direction_metrics, "parameters_unchanged": before == after, "passed": passed, "permitted_next_actions": ["calibrate_prototypes"] if passed else [], "invalidation_reason": None if passed else "prototype reachability isolation gate failed"}
    atomic_write_json(run.root / "prototype_reachability_evidence.json", evidence)
    print(run.root)
    return 0


def _rebind_manifest(args: argparse.Namespace, config: V2Config) -> int:
    """Re-emit a verified input manifest without touching its tensor payload."""
    _require_clean_acceptance(config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("rebound manifest output already exists")
    source = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if args.kind == "clean-target":
        old = CleanTargetManifestV2.from_dict(source)
        old.validate()
        targets = torch.load(old.targets.path, map_location="cpu", weights_only=False)
        reference = tensor_artifact_ref(old.targets.path, role="clean_mnist_targets", git_commit=commit)
        manifest = CleanTargetManifestV2(2, old.dataset_root, old.dataset_split, old.samples_per_label, old.labels, reference, tensor_sha256(targets), tuple(targets.shape), str(targets.dtype), canonical_sha256(config.model_dump(mode="json")), commit, _system_signature(config), canonical_sha256(config.model_dump(mode="json")))
        manifest.validate()
    else:
        pair = _pair_manifest(args.teacher_manifest)
        legacy_hash = _legacy_export_contract(args.legacy_train_config, config)
        old = TeacherTargetManifestV2.from_dict(source)
        old.validate(formal=config.system.is_formal)
        if Path(old.dataset.path).resolve() != Path(args.dataset).resolve():
            raise ValueError("teacher source manifest dataset does not match --dataset")
        scoped_export_hash = _teacher_export_contract_hash(config, pair, legacy_hash)
        manifest = build_teacher_target_manifest(old.dataset.path, pair, formal=config.system.is_formal, generated_git_commit=commit, v2_config_hash=canonical_sha256(config.model_dump(mode="json")), legacy_export_config_hash=legacy_hash, system_contract_hash=_system_signature(config), teacher_export_contract_hash=scoped_export_hash)
    atomic_write_json(output, manifest.to_dict() | {"manifest_hash": manifest.manifest_hash})
    print(output)
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
    calibrate.add_argument("--clean-target-manifest", required=True)
    calibrate.set_defaults(handler=lambda args: _calibrate(args, load_config(args.config)))
    clean = student.add_parser("export-clean-targets")
    for name in ("config", "dataset_root", "output", "manifest"):
        clean.add_argument("--" + name.replace("_", "-"), required=True)
    clean.set_defaults(handler=lambda args: _export_clean_targets(args, load_config(args.config)))
    admit = student.add_parser("admit-prototypes")
    for name in ("config", "calibration_evidence", "prototype_checkpoint", "prototype_raw_cmos", "prototype_preview", "measurements", "clean_target_manifest", "manual_review"):
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
    preflight = student.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.set_defaults(handler=lambda args: _preflight(args, load_config(args.config)))
    evaluate = student.add_parser("evaluate")
    for name in ("config", "training_evidence", "code_basis", "checkpoint", "teacher_target_manifest", "output"):
        evaluate.add_argument("--" + name.replace("_", "-"), required=True)
    evaluate.set_defaults(handler=lambda args: _evaluate(args, load_config(args.config)))
    admit_evaluation = student.add_parser("admit-evaluation")
    for name in ("config", "evaluation_evidence", "evaluation_raw_cmos", "evaluation_preview", "manual_review", "code_basis", "checkpoint", "teacher_target_manifest", "output"):
        admit_evaluation.add_argument("--" + name.replace("_", "-"), required=True)
    admit_evaluation.set_defaults(handler=lambda args: _admit_evaluation(args, load_config(args.config)))
    audit = root.add_parser("audit").add_subparsers(dest="action", required=True)
    capacity = audit.add_parser("capacity")
    for name in ("config", "prototype_evidence", "prototype_checkpoint"):
        capacity.add_argument("--" + name.replace("_", "-"), required=True)
    capacity.set_defaults(handler=lambda args: _capacity(args, load_config(args.config)))
    reachability = audit.add_parser("prototype-reachability")
    for name in ("config", "clean_target_manifest", "failed_calibration_evidence", "failed_prototype_checkpoint"):
        reachability.add_argument("--" + name.replace("_", "-"), required=True)
    reachability.set_defaults(handler=lambda args: _reachability(args, load_config(args.config)))
    evidence = root.add_parser("evidence").add_subparsers(dest="action", required=True)
    summarize = evidence.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    summarize.set_defaults(handler=lambda args: print(Path(args.run).read_text(encoding="utf-8")) or 0)
    rebind = evidence.add_parser("rebind-manifest")
    rebind.add_argument("--config", required=True)
    rebind.add_argument("--kind", choices=("clean-target", "teacher-target"), required=True)
    rebind.add_argument("--source-manifest", required=True)
    rebind.add_argument("--output", required=True)
    rebind.add_argument("--dataset", default="")
    rebind.add_argument("--teacher-manifest", default="")
    rebind.add_argument("--legacy-train-config", default="")
    rebind.set_defaults(handler=lambda args: _rebind_manifest(args, load_config(args.config)))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args) or 0)
