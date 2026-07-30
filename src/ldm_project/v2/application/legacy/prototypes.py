# ruff: noqa: F401,F403,F405,F821,E402
"""Historical CLI implementation segment, loaded only for legacy commands."""

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

from .._cli_dependencies import install as _install_dependencies

_DEPENDENCIES_INSTALLED = False


def _ensure_dependencies() -> None:
    """Load the historical execution surface only when a legacy command runs."""
    global _DEPENDENCIES_INSTALLED
    if not _DEPENDENCIES_INSTALLED:
        _install_dependencies(globals())
        from . import teacher as _teacher

        globals().update(
            {name: value for name, value in _teacher.__dict__.items() if not name.startswith("__")}
        )
        _DEPENDENCIES_INSTALLED = True

def _calibrate(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.prototype_calibration is None:
        raise ValueError(
            "student calibrate-prototypes requires prototype_calibration config"
        )
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    if config.system.is_formal:
        if not args.reachability_evidence:
            raise ValueError(
                "formal prototype calibration requires --reachability-evidence"
            )
        reachability = _record(args.reachability_evidence)
        reachability.validate()
        expected = _context(config, dataset_hash=clean_manifest.targets.sha256)
        if (
            reachability.evidence_kind != "prototype_reachability"
            or reachability.tier != "diagnostic"
            or reachability.status != "diagnostic"
        ):
            raise ValueError(
                "formal prototype calibration requires diagnostic reachability evidence"
            )
        reachability.context.assert_matches(
            expected, fields=("system_signature", "config_hash", "dataset_hash")
        )
        if (
            not all(gate.passed for gate in reachability.gates)
            or "calibrate_prototypes" not in reachability.permitted_next_actions
        ):
            raise ValueError(
                "reachability evidence did not authorize prototype calibration"
            )
        if (
            reachability.input_hashes.get("clean_target_manifest")
            != clean_manifest.manifest_hash
        ):
            raise ValueError("reachability evidence uses another clean-target manifest")
        failed_checkpoint = reachability.input_hashes.get("failed_prototype_checkpoint")
        if (
            not failed_checkpoint
            or failed_checkpoint != reachability.context.checkpoint_hash
        ):
            raise ValueError(
                "reachability evidence lacks a verified failed-checkpoint lineage"
            )
    else:
        reachability = None
    clean_targets = _tensor(clean_manifest.targets.path, device=device)
    run = _new_run(config)
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated, baseline_reserved = (
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
        )
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
        contract_kind=config.prototype_calibration.prototype_contract_kind,
        foreground_weight=config.prototype_calibration.foreground_weight,
        background_weight=config.prototype_calibration.background_weight,
        roi_threshold=config.prototype_calibration.roi_threshold,
        background_energy_weight=config.prototype_calibration.background_energy_weight,
    )
    memory_pass, memory = _memory_report(
        device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    raw = _stream_prototype_raw(
        bundle,
        device=device,
        microbatch_size=config.execution.prototype_microbatch_size,
    )
    raw_path = run.root / "prototype_raw_cmos.pt"
    atomic_torch_save(raw_path, raw)
    review_artifacts = write_review_artifacts(
        root=run.root,
        stem="prototype",
        kind="prototype",
        raw_cmos=raw,
        targets=clean_targets.detach().cpu(),
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
    )
    stable = (
        summary.epoch_history[-1]["loss"] <= summary.epoch_history[-2]["loss"]
        and summary.epoch_history[-1]["corr"] >= summary.epoch_history[-2]["corr"]
        and summary.epoch_history[-1]["dice"] >= summary.epoch_history[-2]["dice"]
    )
    metrics = PrototypeMetrics(
        summary.clean_corr,
        summary.clean_dice,
        summary.prototype_accuracy,
        summary.high_saturation,
        torch.isfinite(torch.tensor(summary.loss)).item()
        and torch.isfinite(raw).all().item(),
        summary.gradients_finite,
        memory_pass,
        stable,
    )
    atomic_write_json(
        run.root / "prototype_measurements.json",
        {
            "summary": summary.__dict__,
            "memory": memory,
            "raw_tensor_sha256": tensor_sha256(raw),
        },
    )
    checkpoint = run.root / "prototype_bundle.pt"
    manifest = _bundle_manifest(
        config,
        bundle,
        dataset_hash=clean_manifest.targets.sha256,
        teacher_manifest_hash=None,
        calibration_hash=basis.artifact_hash,
        basis_hash=basis.artifact_hash,
    )
    save_exact_checkpoint(
        checkpoint,
        bundle,
        manifest,
        extra={
            "epoch_history": summary.epoch_history,
            "memory": memory,
            "clean_targets_hash": clean_manifest.targets.sha256,
            "clean_target_manifest_hash": clean_manifest.manifest_hash,
            "prototype_raw_hash": tensor_sha256(raw),
        },
    )
    checkpoint_hash = file_sha256(checkpoint)
    context = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        calibration_hash=basis.artifact_hash,
        checkpoint_hash=checkpoint_hash,
    )
    measurement_gate = prototype_measurement_gate(metrics)
    gates: tuple[GateResult, ...] = (measurement_gate,)
    if (
        config.prototype_calibration.prototype_contract_kind
        == "formal_prototype_shape_guard_v1"
    ):
        normalized = (
            raw.to(device) / config.prototype_calibration.fixed_exposure_scale
        ).clamp(0.0, 1.0)
        matrix = prototype_correlation_matrix(normalized, clean_targets)
        margins = prototype_margins(matrix)
        shape_checks = {
            "full_frame_accuracy": bool(
                (matrix.argmax(dim=1) == torch.arange(10, device=device)).all().item()
            ),
            "minimum_full_frame_margin": bool((margins >= 0.03).all().item()),
            "foreground_hole": bool(
                (foreground_hole_ratio(normalized, clean_targets) <= 0.08).all().item()
            ),
            "background_speckle": bool(
                (background_speckle_ratio(normalized, clean_targets) <= 0.07)
                .all()
                .item()
            ),
        }
        shape_gate = GateResult(
            "prototype_shape_guard",
            all(shape_checks.values()),
            {
                "min_full_frame_margin": float(margins.min().item()),
                "foreground_hole": float(
                    foreground_hole_ratio(normalized, clean_targets).mean().item()
                ),
                "background_speckle": float(
                    background_speckle_ratio(normalized, clean_targets).mean().item()
                ),
                **shape_checks,
            },
            None if all(shape_checks.values()) else "raw shape guard failed",
        )
        gates = (measurement_gate, shape_gate)
    automatic_passed = all(gate.passed for gate in gates)
    diagnostic = not config.system.is_formal or config.run.evidence_tier == "diagnostic"
    status = (
        "diagnostic"
        if diagnostic
        else ("pending_review" if automatic_passed else "failed")
    )
    actions = (
        ("diagnostic_capacity_smoke",)
        if diagnostic and automatic_passed
        else (("admit_prototypes",) if status == "pending_review" else ())
    )
    input_hashes = {
        "clean_targets": clean_manifest.targets.sha256,
        "clean_target_manifest": clean_manifest.manifest_hash,
        "prototype_checkpoint": checkpoint_hash,
        "prototype_raw_cmos": file_sha256(raw_path),
        "prototype_preview": review_artifacts.preview_sha256,
    }
    if reachability is not None:
        input_hashes["reachability_evidence"] = reachability.evidence_hash
        input_hashes["failed_prototype_checkpoint"] = (
            reachability.context.checkpoint_hash or ""
        )
    record = EvidenceRecordV2(
        2,
        "prototype_calibration",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        input_hashes,
        gates,
        actions,
        None
        if automatic_passed
        else "prototype automatic measurement/shape gate failed",
    )
    _write_record(run, "prototype_calibration_evidence.json", record)
    print(run.root)
    return 0


def _admit_prototypes(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    source = _record(args.calibration_evidence)
    source.validate()
    if source.evidence_kind != "prototype_calibration" or source.status not in {
        "pending_review",
        "diagnostic",
    }:
        raise ValueError("prototype admission requires a calibration record")
    if source.context.system_signature != _system_signature(
        config
    ) or source.context.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("prototype calibration belongs to another v2 configuration")
    if file_sha256(args.prototype_checkpoint) != source.context.checkpoint_hash:
        raise ValueError(
            "prototype checkpoint content does not match calibration evidence"
        )
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    if (
        clean_manifest.targets.sha256 != source.context.dataset_hash
        or clean_manifest.manifest_hash
        != source.input_hashes.get("clean_target_manifest")
    ):
        raise ValueError("clean targets content does not match calibration evidence")
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(
        args.prototype_checkpoint, map_location="cpu", weights_only=False
    )
    actual = CheckpointManifestV2(**payload["manifest"])
    load_exact_checkpoint(args.prototype_checkpoint, bundle, actual)
    raw = _stream_prototype_raw(
        bundle,
        device=device,
        microbatch_size=config.execution.prototype_microbatch_size,
    )
    raw_hash = tensor_sha256(raw)
    saved_raw = _tensor(args.prototype_raw_cmos, device=torch.device("cpu"))
    if tensor_sha256(saved_raw) != raw_hash:
        raise ValueError("prototype replay does not match the saved raw-CMOS evidence")
    preview_hash = file_sha256(args.prototype_preview)
    if preview_hash != source.input_hashes.get("prototype_preview"):
        raise ValueError(
            "prototype preview content does not match calibration evidence"
        )
    review, review_hash = load_manual_review(
        args.manual_review,
        kind="prototype",
        tensor_hash=raw_hash,
        preview_hash=preview_hash,
    )
    measurement = json.loads(Path(args.measurements).read_text(encoding="utf-8"))
    summary = measurement["summary"]
    memory = measurement["memory"]
    stable = (
        summary["epoch_history"][-1]["loss"] <= summary["epoch_history"][-2]["loss"]
        and summary["epoch_history"][-1]["corr"] >= summary["epoch_history"][-2]["corr"]
        and summary["epoch_history"][-1]["dice"] >= summary["epoch_history"][-2]["dice"]
    )
    memory_pass = (
        bool(memory.get("cuda_measured"))
        and bool(
            memory.get("reserved_delta_peak", float("inf"))
            <= 0.85
            * max(
                1.0,
                float(memory.get("total_memory", 0.0))
                - float(memory.get("reserved_baseline", 0.0)),
            )
        )
        and bool(
            memory.get("reserved_peak", float("inf"))
            <= 0.90 * float(memory.get("total_memory", 0.0))
        )
    )
    metrics = PrototypeMetrics(
        float(summary["clean_corr"]),
        float(summary["clean_dice"]),
        float(summary["prototype_accuracy"]),
        float(summary["high_saturation"]),
        bool(torch.isfinite(raw).all().item()),
        bool(summary["gradients_finite"]),
        memory_pass,
        stable,
    )
    gate = prototype_gate(metrics, review.readable_labels)
    run = _new_run(config)
    diagnostic = not config.system.is_formal or config.run.evidence_tier == "diagnostic"
    status = "diagnostic" if diagnostic else ("passed" if gate.passed else "failed")
    actions = (
        ("diagnostic_capacity_smoke",)
        if diagnostic
        else (("audit_capacity",) if status == "passed" else ())
    )
    context = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        calibration_hash=source.context.calibration_hash,
        checkpoint_hash=file_sha256(args.prototype_checkpoint),
    )
    record = EvidenceRecordV2(
        2,
        "prototype",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "calibration_evidence": source.evidence_hash,
            "clean_target_manifest": clean_manifest.manifest_hash,
            "prototype_checkpoint": file_sha256(args.prototype_checkpoint),
            "prototype_raw_cmos": file_sha256(args.prototype_raw_cmos),
            "prototype_preview": preview_hash,
            "manual_review": review_hash,
        },
        (gate,),
        actions,
        None if gate.passed else "prototype gate failed",
    )
    _write_record(run, "prototype_evidence.json", record)
    print(run.root)
    return 0


def _capacity(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.capacity_audit is None:
        raise ValueError("audit capacity requires capacity_audit config")
    prototype = _record(args.prototype_evidence)
    diagnostic = (
        not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    )
    action = "diagnostic_capacity_smoke" if diagnostic else "audit_capacity"
    require_evidence_action(
        prototype,
        kind="prototype",
        action=action,
        expected_context=_context(
            config, checkpoint_hash=file_sha256(args.prototype_checkpoint)
        ),
        allow_diagnostic=diagnostic,
    )
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    warm = load_prototype_warm_start(
        args.prototype_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    labels = torch.arange(10, device=device)
    prototype_commands = bundle.generator.prototype_command(labels).detach()
    supports = (prototype_commands[:, 0] >= 0.2).to(prototype_commands.dtype)
    audit = config.capacity_audit
    responses, unchanged, before_hash, after_hash = finite_difference_response_matrices(
        bundle.system,
        prototype_commands,
        supports,
        candidate_count=audit.dct_modes,
        epsilon=audit.finite_difference_epsilon,
        pixel_pitch_m=(
            config.system.slm_pixel_pitch_m,
            config.system.slm_pixel_pitch_m,
        ),
        max_cycles_per_mm=audit.max_cycles_per_mm,
        microbatch_size=config.execution.capacity_microbatch_size,
    )
    artifact = build_code_basis(
        responses,
        supports.cpu(),
        candidate_count=audit.dct_modes,
        seed=audit.seed,
        source_hash=prototype.evidence_hash,
        pixel_pitch_m=(
            config.system.slm_pixel_pitch_m,
            config.system.slm_pixel_pitch_m,
        ),
        max_cycles_per_mm=audit.max_cycles_per_mm,
    )
    commands, slm_intensity, camera, coefficients = replay_capacity(
        bundle.system,
        prototype_commands,
        artifact,
        supports,
        codes_per_label=audit.codes_per_label,
        command_gain=audit.command_gain,
        microbatch_size=config.execution.capacity_microbatch_size,
    )
    trace = bundle.system.forward(
        bundle.system.command(prototype_commands[:1]), return_optical_trace=True
    ).optical_trace
    provenance_complete = trace is not None and all(
        field.provenance.upstream_plane is not None
        or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"}
        for field in trace
    )
    slm_factor, camera_factor, support_energy, clipping = capacity_metrics(
        commands, slm_intensity, camera, supports.cpu()
    )
    gate = capacity_gate(
        CapacityMetrics(
            slm_factor,
            camera_factor,
            support_energy,
            clipping,
            unchanged,
            False,
            provenance_complete,
        ),
        prototype.gates[0],
    )
    run = _new_run(config)
    basis_path = run.root / "code_basis.pt"
    replay_path = run.root / "capacity_replay.pt"
    artifact.save(str(basis_path))
    atomic_torch_save(
        replay_path,
        {
            "unclamped_commands": commands.cpu(),
            "slm_intensity": slm_intensity.cpu(),
            "raw_camera": camera.cpu(),
            "coefficients": coefficients.cpu(),
            "supports": supports.cpu(),
            "responses": responses.cpu(),
            "parameter_before_hash": before_hash,
            "parameter_after_hash": after_hash,
            "warm_start": warm.to_dict(),
        },
    )
    status = "diagnostic" if diagnostic else ("passed" if gate.passed else "failed")
    actions = (
        ("diagnostic_train_smoke",)
        if diagnostic
        else (("train",) if status == "passed" else ())
    )
    context = _context(
        config,
        dataset_hash=prototype.context.dataset_hash,
        calibration_hash=prototype.context.calibration_hash,
        checkpoint_hash=file_sha256(args.prototype_checkpoint),
        basis_hash=artifact.artifact_hash,
    )
    record = EvidenceRecordV2(
        2,
        "capacity",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "prototype_evidence": prototype.evidence_hash,
            "prototype_checkpoint": file_sha256(args.prototype_checkpoint),
            "basis": artifact.artifact_hash,
            "basis_file": file_sha256(basis_path),
            "replay": file_sha256(replay_path),
            "responses": tensor_sha256(responses),
            "supports": tensor_sha256(supports),
        },
        (gate,),
        actions,
        None if gate.passed else "capacity gate failed",
    )
    _write_record(run, "capacity_evidence.json", record)
    print(run.root)
    return 0


def _teacher_dataset(
    manifest: TeacherTargetManifestV2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(manifest.dataset.path, map_location="cpu", weights_only=False)
    return (
        payload["codes"].cpu(),
        payload["class_labels"].cpu().long(),
        payload["teacher_camera"].cpu(),
    )


@torch.no_grad()
def _stream_raw(
    bundle: StudentBundle,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    microbatch_size: int,
) -> torch.Tensor:
    if microbatch_size < 1:
        raise ValueError("evaluation microbatch must be positive")
    rows = []
    for start in range(0, codes.shape[0], microbatch_size):
        result = bundle.system.forward_from_generator(
            bundle.generator,
            codes[start : start + microbatch_size].to(device),
            labels[start : start + microbatch_size].to(device),
        )
        rows.append(result.raw_camera.tensor.detach().cpu())
    return torch.cat(rows)


@torch.no_grad()
def _stream_prototype_raw(
    bundle: StudentBundle, *, device: torch.device, microbatch_size: int
) -> torch.Tensor:
    labels = torch.arange(10, device=device)
    rows = []
    for start in range(0, labels.numel(), microbatch_size):
        command = bundle.generator.prototype_command(
            labels[start : start + microbatch_size]
        )
        rows.append(
            bundle.system.forward(bundle.system.command(command))
            .raw_camera.tensor.detach()
            .cpu()
        )
    return torch.cat(rows)


def _prototype_shape_summary(
    raw: torch.Tensor,
    clean: torch.Tensor,
    *,
    exposure_scale: float,
    device: torch.device,
) -> tuple[dict[str, float | list[int]], GateResult]:
    """One full-frame metric contract shared by prototype diagnostic audits."""
    labels = torch.arange(10, device=device)
    normalized = (raw.to(device) / exposure_scale).clamp(0.0, 1.0)
    matrix = prototype_correlation_matrix(normalized, clean)
    margins = prototype_margins(matrix)
    values: dict[str, float | list[int]] = {
        "clean_corr": float(correlation(normalized, clean).mean().item()),
        "hard_dice": float(foreground_dice(normalized, clean).mean().item()),
        "standard_accuracy": float(
            (matrix.argmax(dim=1) == labels).to(torch.float32).mean().item()
        ),
        "minimum_full_frame_margin": float(margins.min().item()),
        "foreground_hole": float(
            foreground_hole_ratio(normalized, clean).mean().item()
        ),
        "background_speckle": float(
            background_speckle_ratio(normalized, clean).mean().item()
        ),
        "high_saturation": float(
            high_saturation_ratio(raw.to(device), exposure_scale).item()
        ),
        "confusion": [int(value) for value in matrix.argmax(dim=1).cpu().tolist()],
    }
    checks = {
        "clean_corr": values["clean_corr"] >= 0.70,
        "hard_dice": values["hard_dice"] >= 0.68,
        "standard_accuracy": values["standard_accuracy"] == 1.0,
        "minimum_full_frame_margin": values["minimum_full_frame_margin"] >= 0.03,
        "foreground_hole": values["foreground_hole"] <= 0.08,
        "background_speckle": values["background_speckle"] <= 0.07,
        "high_saturation": values["high_saturation"] <= 0.08,
        "finite": bool(torch.isfinite(raw).all().item()),
    }
    return values, GateResult(
        "prototype_raw_shape",
        all(checks.values()),
        {
            key: float(value) if isinstance(value, (int, float)) else bool(value)
            for key, value in values.items()
            if key != "confusion"
        }
        | checks,
        None if all(checks.values()) else "full-frame prototype shape gate failed",
    )


def _trace_plane_report(
    fields: tuple[ComplexField, ...],
) -> tuple[list[dict[str, object]], str | None]:
    """Report physical planes and the first persistent material separation collapse."""
    rows: list[dict[str, object]] = []
    distances: list[float] = []
    for field in fields:
        intensity = field.tensor.abs().square().detach()
        flat = intensity.flatten(1)
        unit = flat / flat.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
        pair = torch.cdist(unit, unit)
        minimum = float(
            pair[~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)]
            .min()
            .item()
        )
        distances.append(minimum)
        rows.append(
            {
                "plane": field.plane.to_dict(),
                "provenance": field.provenance.to_dict(),
                "mean_energy": float(intensity.sum(dim=(-3, -2, -1)).mean().item()),
                "minimum_class_distance": minimum,
            }
        )
    for index, row in enumerate(rows):
        previous = distances[index - 1] if index else distances[0]
        row["adjacent_distance_retention"] = (
            1.0 if index == 0 else distances[index] / max(previous, 1.0e-12)
        )
        row["cumulative_distance_retention"] = distances[index] / max(
            distances[0], 1.0e-12
        )
    for index in range(1, len(rows) - 1):
        if (
            rows[index]["adjacent_distance_retention"] < 0.85
            and rows[index + 1]["cumulative_distance_retention"]
            <= rows[index]["cumulative_distance_retention"]
        ):
            return rows, str(rows[index]["plane"]["plane_id"])
    return rows, None


def _train(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.student_training is None or config.prototype_calibration is None:
        raise ValueError("student train requires training and prototype config")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    prototype, capacity = (
        _record(args.prototype_evidence),
        _record(args.capacity_evidence),
    )
    basis = CodeBasisArtifactV2.load(args.code_basis)
    diagnostic = (
        not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    )
    action = "diagnostic_train_smoke" if diagnostic else "train"
    require_evidence_action(
        prototype,
        kind="prototype",
        action="diagnostic_capacity_smoke" if diagnostic else "audit_capacity",
        expected_context=_context(
            config, checkpoint_hash=file_sha256(args.prototype_checkpoint)
        ),
        allow_diagnostic=diagnostic,
    )
    require_evidence_action(
        capacity,
        kind="capacity",
        action=action,
        expected_context=_context(
            config,
            calibration_hash=prototype.context.calibration_hash,
            checkpoint_hash=file_sha256(args.prototype_checkpoint),
            basis_hash=basis.artifact_hash,
        ),
        allow_diagnostic=diagnostic,
    )
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, basis).to(device)
    warm = load_prototype_warm_start(
        args.prototype_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    codes, labels, targets = _teacher_dataset(target)
    settings = config.student_training
    summary = train_code_conditioned(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        targets,
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
        prototype_evidence=prototype,
        capacity_evidence=capacity,
        expected_basis_hash=basis.artifact_hash,
        epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        weights={
            "prototype": settings.prototype_fidelity_weight,
            "paired_residual": settings.paired_teacher_residual_weight,
            "code_transport": settings.code_transport_weight,
            "clipping": settings.clipping_weight,
            "saturation": settings.saturation_weight,
            "phase_tv": settings.phase_tv_weight,
        },
        microbatch_size=config.execution.training_microbatch_size,
        activation_checkpoint=config.execution.activation_checkpoint,
        allow_diagnostic_evidence=diagnostic,
    )
    run = _new_run(config)
    checkpoint = run.root / "student_bundle.pt"
    manifest = _bundle_manifest(
        config,
        bundle,
        dataset_hash=target.dataset_hash,
        teacher_manifest_hash=target.teacher_pair_manifest_hash,
        calibration_hash=prototype.context.calibration_hash or "",
        basis_hash=basis.artifact_hash,
    )
    save_exact_checkpoint(
        checkpoint,
        bundle,
        manifest,
        extra={
            "summary": summary.__dict__,
            "basis_artifact_hash": basis.artifact_hash,
            "warm_start": warm.to_dict(),
        },
    )
    checkpoint_hash = file_sha256(checkpoint)
    status = "diagnostic" if diagnostic else "passed"
    actions = ("diagnostic_evaluate_smoke",) if diagnostic else ("evaluate",)
    context = _context(
        config,
        teacher_manifest_hash=target.teacher_pair_manifest_hash,
        dataset_hash=target.dataset_hash,
        calibration_hash=prototype.context.calibration_hash,
        checkpoint_hash=checkpoint_hash,
        basis_hash=basis.artifact_hash,
    )
    record = EvidenceRecordV2(
        2,
        "code_training",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "prototype": prototype.evidence_hash,
            "capacity": capacity.evidence_hash,
            "teacher_dataset": target.dataset_hash,
            "student_checkpoint": checkpoint_hash,
        },
        (
            GateResult(
                "training_completed",
                torch.isfinite(torch.tensor(summary.loss)).item(),
                {
                    "loss": summary.loss,
                    "paired_corr": summary.paired_corr,
                    "paired_dice": summary.paired_dice,
                    "code_variance": summary.code_variance,
                    "clipping": summary.clipping_fraction,
                },
            ),
        ),
        actions,
    )
    _write_record(run, "code_training_evidence.json", record)
    print(run.root)
    return 0


def _load_student_bundle(
    config: V2Config, basis_path: str, checkpoint_path: str, *, device: torch.device
) -> tuple[StudentBundle, CheckpointManifestV2]:
    basis = CodeBasisArtifactV2.load(basis_path)
    bundle = StudentBundle(config, basis).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual = CheckpointManifestV2(**payload["manifest"])
    if (
        actual.system_signature != _system_signature(config)
        or actual.config_hash != canonical_sha256(config.model_dump(mode="json"))
        or actual.basis_hash != basis.artifact_hash
    ):
        raise ValueError("student checkpoint does not match config/system/basis")
    load_exact_checkpoint(checkpoint_path, bundle, actual)
    return bundle, actual


def _sample(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    device = torch.device(config.run.device)
    bundle, _ = _load_student_bundle(
        config, args.code_basis, args.checkpoint, device=device
    )
    result = bundle.system.forward_from_generator(
        bundle.generator,
        _tensor(args.codes, device=device),
        _tensor(args.labels, device=device).long(),
    )
    atomic_torch_save(args.output, result.raw_camera.tensor.detach().cpu())
    return 0


def _evaluate(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.prototype_calibration is None:
        raise ValueError("student evaluate requires prototype calibration config")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    training = _record(args.training_evidence)
    device = torch.device(config.run.device)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    checkpoint_hash = file_sha256(args.checkpoint)
    diagnostic = (
        not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    )
    require_evidence_action(
        training,
        kind="code_training",
        action="diagnostic_evaluate_smoke" if diagnostic else "evaluate",
        expected_context=_context(
            config,
            teacher_manifest_hash=target.teacher_pair_manifest_hash,
            dataset_hash=target.dataset_hash,
            checkpoint_hash=checkpoint_hash,
            basis_hash=basis.artifact_hash,
        ),
        allow_diagnostic=diagnostic,
    )
    bundle, _ = _load_student_bundle(
        config, args.code_basis, args.checkpoint, device=device
    )
    codes, labels, teacher = _teacher_dataset(target)
    raw = _stream_raw(
        bundle,
        codes,
        labels,
        device=device,
        microbatch_size=config.execution.evaluation_microbatch_size,
    )
    normalized = (raw / config.prototype_calibration.fixed_exposure_scale).clamp(
        0.0, 1.0
    )
    camera_fraction, teacher_corr, teacher_dice = acceptance_metrics(
        raw, teacher, labels
    )
    lower_bound = paired_advantage_lower_bound(normalized, teacher, labels)
    run = _new_run(config)
    raw_path = run.root / "evaluation_raw_cmos.pt"
    atomic_torch_save(raw_path, raw.detach().cpu())
    review_artifacts = write_review_artifacts(
        root=run.root,
        stem="evaluation",
        kind="code_acceptance",
        raw_cmos=raw,
        labels=labels,
        targets=teacher,
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
    )
    gate = code_acceptance_gate(
        CodeAcceptanceMetrics(
            camera_fraction, teacher_corr, teacher_dice, lower_bound, 1234, 10000
        ),
        None,
    )
    if diagnostic:
        status, actions = "diagnostic", ()
    else:
        status, actions = "pending_review", ("admit_evaluation",)
    context = _context(
        config,
        teacher_manifest_hash=target.teacher_pair_manifest_hash,
        dataset_hash=target.dataset_hash,
        calibration_hash=training.context.calibration_hash,
        checkpoint_hash=checkpoint_hash,
        basis_hash=basis.artifact_hash,
    )
    record = EvidenceRecordV2(
        2,
        "code_acceptance",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "training_evidence": training.evidence_hash,
            "checkpoint": checkpoint_hash,
            "basis": basis.artifact_hash,
            "teacher_dataset": target.dataset_hash,
            "raw_cmos": file_sha256(raw_path),
            "evaluation_preview": review_artifacts.preview_sha256,
        },
        (gate,),
        actions,
        "manual review is required before code acceptance",
    )
    _write_record(run, "code_acceptance_evidence.json", record)
    atomic_write_json(args.output, record.to_dict())
    print(run.root)
    return 0


def _admit_evaluation(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    source = _record(args.evaluation_evidence)
    if source.evidence_kind != "code_acceptance" or source.status not in {
        "pending_review",
        "diagnostic",
    }:
        raise ValueError("evaluation admission requires pending evaluation evidence")
    target = _target_manifest(args.teacher_target_manifest)
    _validate_target_for_config(target, config)
    basis = CodeBasisArtifactV2.load(args.code_basis)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        teacher_manifest_hash=target.teacher_pair_manifest_hash,
        dataset_hash=target.dataset_hash,
        checkpoint_hash=checkpoint_hash,
        basis_hash=basis.artifact_hash,
    )
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
    review, review_hash = load_manual_review(
        args.manual_review,
        kind="code_acceptance",
        tensor_hash=raw_hash,
        preview_hash=preview_hash,
    )
    gate = source.gates[0]
    completed = code_acceptance_gate(
        CodeAcceptanceMetrics(
            float(gate.metrics["camera_code_fraction"]),
            float(gate.metrics["teacher_corr"]),
            float(gate.metrics["teacher_dice"]),
            float(gate.metrics["paired_advantage_lower_bound"]),
            1234,
            10000,
        ),
        review.readable_labels,
    )
    run = _new_run(config)
    diagnostic = (
        not config.system.is_formal and config.run.evidence_tier == "diagnostic"
    )
    status = (
        "diagnostic" if diagnostic else ("passed" if completed.passed else "failed")
    )
    actions = (
        ()
        if diagnostic
        else (("manual_promotion_review",) if status == "passed" else ())
    )
    record = EvidenceRecordV2(
        2,
        "code_acceptance",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        source.context,
        {**source.input_hashes, "manual_review": review_hash},
        (completed,),
        actions,
        None if completed.passed else "code-conditioned acceptance gate failed",
    )
    _write_record(run, "code_acceptance_evidence.json", record)
    atomic_write_json(args.output, record.to_dict())
    print(run.root)
    return 0


def _preflight(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    """Bounded no-update physical execution probe; it is diagnostic by definition."""
    device = torch.device(config.run.device)
    run = _new_run(config)
    basis = _zero_basis(config.system.slm_shape)
    bundle = StudentBundle(config, basis).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated, baseline_reserved = (
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
        )
    else:
        baseline_allocated = baseline_reserved = 0
    before = state_content_hash(bundle.state_dict())
    stages: list[GateResult] = []
    reports: dict[str, dict[str, float | bool]] = {}
    label = torch.zeros(1, dtype=torch.long, device=device)

    def prototype() -> None:
        bundle.zero_grad(set_to_none=True)
        command = bundle.generator(torch.zeros(1, 8, device=device), label)
        raw = checkpoint(
            lambda value: (
                bundle.system.forward(
                    bundle.system.command(value), return_optical_trace=True
                ).raw_camera.tensor
            ),
            command,
            use_reentrant=False,
        )
        raw.square().mean().backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in bundle.parameters()
        ):
            raise ValueError("prototype preflight produced non-finite gradients")
        bundle.zero_grad(set_to_none=True)

    gate, report = _preflight_stage(
        "prototype_backward",
        prototype,
        device=device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    stages.append(gate)
    reports["prototype_backward"] = report

    def capacity() -> None:
        with torch.no_grad():
            base = bundle.generator.prototype_command(label)
            support = torch.ones(*config.system.slm_shape, device=device)
            direction = (
                supported_dct_candidates(
                    support,
                    candidate_count=8,
                    pixel_pitch_m=(
                        config.system.slm_pixel_pitch_m,
                        config.system.slm_pixel_pitch_m,
                    ),
                    max_cycles_per_mm=config.capacity_audit.max_cycles_per_mm
                    if config.capacity_audit
                    else 20.0,
                )
                .directions[0]
                .unsqueeze(0)
                .unsqueeze(0)
            )
            epsilon = (
                config.capacity_audit.finite_difference_epsilon
                if config.capacity_audit
                else 1.0e-3
            )
            plus = bundle.system.forward(
                bundle.system.command((base + epsilon * direction).clamp(0.0, 1.0))
            ).raw_camera.tensor
            minus = bundle.system.forward(
                bundle.system.command((base - epsilon * direction).clamp(0.0, 1.0))
            ).raw_camera.tensor
            if not torch.isfinite((plus - minus) / (2.0 * epsilon)).all():
                raise ValueError("capacity preflight response is non-finite")

    gate, report = _preflight_stage(
        "capacity_pair",
        capacity,
        device=device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    stages.append(gate)
    reports["capacity_pair"] = report

    def training() -> None:
        bundle.zero_grad(set_to_none=True)
        labels = torch.arange(10, device=device).repeat_interleave(2)
        codes = torch.randn(20, 8, device=device)
        values = []
        for start in range(0, 20, config.execution.training_microbatch_size):
            command = bundle.generator(
                codes[start : start + config.execution.training_microbatch_size],
                labels[start : start + config.execution.training_microbatch_size],
            )
            values.append(
                checkpoint(
                    lambda value: (
                        bundle.system.forward(
                            bundle.system.command(value)
                        ).raw_camera.tensor
                    ),
                    command,
                    use_reentrant=False,
                )
            )
        torch.cat(values).square().mean().backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in bundle.parameters()
        ):
            raise ValueError("training preflight produced non-finite gradients")
        bundle.zero_grad(set_to_none=True)

    gate, report = _preflight_stage(
        "logical_training_backward",
        training,
        device=device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    stages.append(gate)
    reports["logical_training_backward"] = report

    def evaluation() -> None:
        with torch.no_grad():
            bundle.system.forward_from_generator(
                bundle.generator, torch.zeros(1, 8, device=device), label
            )

    gate, report = _preflight_stage(
        "evaluation_microbatch",
        evaluation,
        device=device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    stages.append(gate)
    reports["evaluation_microbatch"] = report
    after = state_content_hash(bundle.state_dict())
    trace = bundle.system.forward(
        bundle.system.command(bundle.generator.prototype_command(label)),
        return_optical_trace=True,
    ).optical_trace
    provenance_complete = trace is not None and all(
        field.provenance.upstream_plane is not None
        or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"}
        for field in trace
    )
    unchanged = before == after
    stages.append(
        GateResult(
            "preflight_parameter_unchanged",
            unchanged,
            {"before_after_equal": unchanged},
            None if unchanged else "preflight modified parameters",
        )
    )
    stages.append(
        GateResult(
            "preflight_provenance",
            provenance_complete,
            {"provenance_complete": provenance_complete},
            None if provenance_complete else "optical trace is incomplete",
        )
    )
    record = EvidenceRecordV2(
        2,
        "formal_execution_preflight",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config, calibration_hash=basis.artifact_hash, basis_hash=basis.artifact_hash
        ),
        {},
        tuple(stages),
        (),
        None
        if all(gate.passed for gate in stages)
        else "formal execution preflight did not pass",
    )
    _write_record(run, "formal_execution_preflight_evidence.json", record)
    atomic_write_json(run.root / "formal_execution_preflight_memory.json", reports)
    print(run.root)
    return 0


def _reachability(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.prototype_calibration is None:
        raise ValueError("prototype reachability requires prototype_calibration config")
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    device = torch.device(config.run.device)
    clean = _tensor(clean_manifest.targets.path, device=device)
    source = _record(args.failed_calibration_evidence)
    source.validate()
    if source.evidence_kind != "prototype_calibration" or source.status != "failed":
        raise ValueError("reachability requires failed prototype-calibration evidence")
    if source.context.system_signature != _system_signature(config):
        raise ValueError(
            "failed calibration evidence belongs to another physical contract"
        )
    if (
        source.context.dataset_hash != clean_manifest.targets.sha256
        or source.input_hashes.get("clean_target_manifest")
        != clean_manifest.manifest_hash
    ):
        raise ValueError(
            "failed calibration evidence does not match clean-target manifest"
        )
    failed_checkpoint_hash = file_sha256(args.failed_prototype_checkpoint)
    if failed_checkpoint_hash != source.context.checkpoint_hash:
        raise ValueError(
            "failed prototype checkpoint does not match failed calibration evidence"
        )
    failed_payload = torch.load(
        args.failed_prototype_checkpoint, map_location="cpu", weights_only=False
    )
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

    def replay(
        bundle: StudentBundle, *, trace: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, torch.Tensor]]:
        rows, traces, crop_rows = (
            [],
            None,
            {"encoder_crop_retention": [], "decoder_crop_retention": []},
        )
        with torch.no_grad():
            for label in labels:
                result = bundle.system.forward(
                    bundle.system.command(
                        bundle.generator.prototype_command(label[None])
                    ),
                    return_optical_trace=trace,
                )
                rows.append(result.raw_camera.tensor)
                for name, value in result.crop_statistics.items():
                    crop_rows[name].append(value.detach().cpu())
                if trace and traces is None:
                    traces = [[] for _ in result.optical_trace or ()]
                if trace and result.optical_trace is not None:
                    for bucket, field in zip(traces, result.optical_trace):
                        bucket.append(field)
        crops = {name: torch.cat(values, dim=0) for name, values in crop_rows.items()}
        return (
            torch.cat(rows),
            tuple(
                torch.cat([field.tensor.abs().square().cpu() for field in bucket])
                for bucket in traces
            )
            if traces is not None
            else None,
            crops,
        )

    def camera_metrics(raw: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
        normalized = (raw / config.prototype_calibration.fixed_exposure_scale).clamp(
            0.0, 1.0
        )
        matrix = _differentiable_correlation(
            normalized[:, None]
            .expand(10, 10, *normalized.shape[1:])
            .reshape(100, *normalized.shape[1:]),
            clean[None].expand(10, 10, *clean.shape[1:]).reshape(100, *clean.shape[1:]),
        ).reshape(10, 10)
        dice = (2.0 * (normalized * clean).flatten(1).sum(dim=1) + 1.0e-6) / (
            normalized.square().flatten(1).sum(dim=1)
            + clean.square().flatten(1).sum(dim=1)
            + 1.0e-6
        )
        margin = (
            matrix.diag()
            - (matrix + torch.eye(10, device=device) * -2.0).max(dim=1).values
        )
        return {
            "clean_corr": float(
                _differentiable_correlation(normalized, clean).mean().item()
            ),
            "clean_dice_soft": float(dice.mean().item()),
            "prototype_accuracy": float(
                (matrix.argmax(dim=1) == labels).to(torch.float32).mean().item()
            ),
            "min_target_margin": float(margin.min().item()),
            "high_saturation": float(
                (raw / config.prototype_calibration.fixed_exposure_scale > 1.0)
                .to(torch.float32)
                .mean()
                .item()
            ),
        }, margin.detach()

    def min_pair_distance(values: torch.Tensor) -> float:
        flat = values.flatten(1)
        normalized = flat / flat.square().sum(dim=1, keepdim=True).sqrt().clamp_min(
            1.0e-12
        )
        distances = torch.cdist(normalized, normalized)
        return float(
            distances[
                ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
            ]
            .min()
            .item()
        )

    states = {}
    for kind in ("uniform_half", "failed_replay", "clean_intensity_to_amplitude"):
        bundle = bundle_with(kind)
        raw, trace, crops = replay(bundle, trace=True)
        metrics, _ = camera_metrics(raw)
        states[kind] = {
            "metrics": metrics,
            "trace_plane_ids": [
                field.plane.plane_id
                for field in bundle.system.forward(
                    bundle.system.command(
                        bundle.generator.prototype_command(labels[:1])
                    ),
                    return_optical_trace=True,
                ).optical_trace
                or ()
            ],
            "trace": trace,
            "crops": crops,
        }

    clean_bundle = bundle_with("clean_intensity_to_amplitude")
    before = state_content_hash(clean_bundle.state_dict())
    template = clean_bundle.generator.template_logits
    slm_error = float(
        (clean_bundle.generator.prototype_command(labels).square() - clean)
        .square()
        .max()
        .sqrt()
        .item()
    )
    raw, trace, crops = replay(clean_bundle, trace=True)
    direction_metrics: dict[str, dict[str, object]] = {}
    groups = {
        "template": [template],
        "encoder": [clean_bundle.system.encoder.phase],
        "decoder": [clean_bundle.system.decoder.phase],
        "joint": [
            template,
            clean_bundle.system.encoder.phase,
            clean_bundle.system.decoder.phase,
        ],
    }
    _, base_margin = camera_metrics(raw)
    for name, parameters in groups.items():
        clean_bundle.zero_grad(set_to_none=True)
        current_rows = []
        for label in labels:
            current_rows.append(
                clean_bundle.system.forward(
                    clean_bundle.system.command(
                        clean_bundle.generator.prototype_command(label[None])
                    )
                ).raw_camera.tensor
            )
        current_raw = torch.cat(current_rows)
        normalized = (
            current_raw / config.prototype_calibration.fixed_exposure_scale
        ).clamp(0.0, 1.0)
        loss = torch.nn.functional.mse_loss(normalized, clean) + 0.25 * (
            1.0 - _differentiable_correlation(normalized, clean).mean()
        )
        loss.backward()
        finite = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
        )
        snapshots = [parameter.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(
                        -1.0e-3
                        * parameter.grad
                        / parameter.grad.square().mean().sqrt().clamp_min(1.0e-12)
                    )
        candidate_raw, _, _ = replay(clean_bundle)
        _, candidate_margin = camera_metrics(candidate_raw)
        improvement = candidate_margin - base_margin
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
        direction_metrics[name] = {
            "gradient_finite": bool(finite),
            "per_label_margin_improvement": [
                float(value) for value in improvement.cpu().tolist()
            ],
            "all_margins_strictly_improved": bool((improvement > 0.0).all().item()),
        }
    after = state_content_hash(clean_bundle.state_dict())
    crop_statistics = {
        name: {
            "min": float(value.min().item()),
            "mean": float(value.mean().item()),
            "max": float(value.max().item()),
            "values": [float(item) for item in value.flatten().tolist()],
        }
        for name, value in crops.items()
    }
    crop_values = [item for values in crops.values() for item in values.flatten()]
    front_retention = 0.0
    if trace is not None and len(trace) >= 2:
        front_retention = min_pair_distance(trace[1]) / max(
            min_pair_distance(trace[0]), 1.0e-12
        )
    crop_gate = (
        bool(crop_values)
        and all(0.0 <= float(value) <= 1.0 for value in crop_values)
        and min(float(value) for value in crop_values) >= 0.98
    )
    direction_gate = any(
        bool(direction_metrics[name]["all_margins_strictly_improved"])
        for name in ("template", "joint")
    )
    gates = (
        GateResult(
            "slm_clean_intensity", slm_error <= 1.0e-6, {"max_error": slm_error}
        ),
        GateResult(
            "front_class_distance_retention",
            front_retention >= 0.95,
            {"worst_pair_distance_retention": front_retention},
        ),
        GateResult(
            "finite_crop_retention",
            crop_gate,
            {
                "minimum_retention": min(float(value) for value in crop_values)
                if crop_values
                else -1.0
            },
        ),
        GateResult(
            "directional_margin_reachability",
            direction_gate,
            {
                "template_all_labels": bool(
                    direction_metrics["template"]["all_margins_strictly_improved"]
                ),
                "joint_all_labels": bool(
                    direction_metrics["joint"]["all_margins_strictly_improved"]
                ),
            },
        ),
        GateResult(
            "finite_gradients",
            all(
                bool(metric["gradient_finite"]) for metric in direction_metrics.values()
            ),
            {
                "all_groups_finite": all(
                    bool(metric["gradient_finite"])
                    for metric in direction_metrics.values()
                )
            },
        ),
        GateResult(
            "parameters_unchanged",
            before == after,
            {"before_after_equal": before == after},
        ),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    context = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=failed_checkpoint_hash,
    )
    record = EvidenceRecordV2(
        2,
        "prototype_reachability",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "clean_target_manifest": clean_manifest.manifest_hash,
            "failed_calibration_evidence": source.evidence_hash,
            "failed_prototype_checkpoint": failed_checkpoint_hash,
        },
        gates,
        ("calibrate_prototypes",) if passed else (),
        None if passed else "prototype reachability isolation gate failed",
    )
    _write_record(run, "prototype_reachability_evidence.json", record)
    atomic_write_json(
        run.root / "prototype_reachability_report.json",
        {
            "supersedes": {
                "evidence": str(Path(args.failed_calibration_evidence)),
                "reason": "earlier reachability harness used intensity means and similarity ratios instead of physical retention/distance gates",
            },
            "states": {
                name: {
                    "metrics": value["metrics"],
                    "trace_plane_ids": value["trace_plane_ids"],
                }
                for name, value in states.items()
            },
            "slm_intensity_max_error": slm_error,
            "front_class_distance_retention": front_retention,
            "crop_statistics": crop_statistics,
            "direction_metrics": direction_metrics,
            "parameters_unchanged": before == after,
        },
    )
    print(run.root)
    return 0
