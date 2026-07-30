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

_install_dependencies(globals())

from . import teacher as _teacher
from . import prototypes as _prototypes
from . import geometry as _geometry

globals().update({name: value for name, value in _teacher.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _prototypes.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _geometry.__dict__.items() if not name.startswith("__")})

def _direct_template_replay(args: argparse.Namespace, config: V2Config) -> int:
    """Replay a successful amplitude-control oracle without any optimizer."""
    oracle = _record(args.control_evidence)
    oracle.validate()
    if (
        oracle.evidence_kind != "prototype_control_oracle"
        or oracle.status != "diagnostic"
        or not oracle.gates
        or not oracle.gates[0].metrics.get("amplitude_only_passed", False)
    ):
        raise ValueError(
            "direct-template replay requires a successful diagnostic amplitude control oracle"
        )
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    expected = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
    )
    oracle.context.assert_matches(
        expected,
        fields=("system_signature", "config_hash", "dataset_hash", "checkpoint_hash"),
    )
    report_path = Path(args.control_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arm = report.get("arms", {}).get("formal_amplitude_only", {})
    if (
        report.get("decision") != "direct_template_replay_candidate"
        or not report.get("parameters_unchanged")
        or arm.get("input_control") != "formal_amplitude_only"
        or not arm.get("shape_gate", {}).get("passed")
    ):
        raise ValueError(
            "control report does not establish amplitude direct-template replay"
        )
    if arm.get("commands") != file_sha256(args.amplitude_commands) or arm.get(
        "raw_cmos"
    ) != file_sha256(args.oracle_raw_cmos):
        raise ValueError(
            "control report artifact hashes do not match the supplied command/raw tensors"
        )
    commands = _tensor(args.amplitude_commands, device=torch.device("cpu"))
    oracle_raw = _tensor(args.oracle_raw_cmos, device=torch.device("cpu"))
    if (
        tuple(commands.shape) != (10, 1, *config.system.slm_shape)
        or tuple(oracle_raw.shape) != tuple(commands.shape)
        or commands.dtype != torch.float32
        or not torch.isfinite(commands).all()
        or bool((commands < 0.0).any())
        or bool((commands > 1.0).any())
    ):
        raise ValueError(
            "direct-template commands/raw tensors violate the locked amplitude contract"
        )
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    load_prototype_warm_start(
        args.shape_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    before = state_content_hash(bundle.state_dict())
    rows = []
    with torch.no_grad():
        for label in range(10):
            rows.append(
                bundle.system.forward(
                    bundle.system.command(commands[label : label + 1].to(device))
                ).raw_camera.tensor.cpu()
            )
    raw = torch.cat(rows)
    after = state_content_hash(bundle.state_dict())
    bit_exact = tensor_sha256(raw) == tensor_sha256(oracle_raw)
    metrics, shape_gate = _prototype_shape_summary(
        raw,
        _tensor(clean_manifest.targets.path, device=device),
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
        device=device,
    )
    gate = GateResult(
        "direct_template_replay",
        bool(bit_exact and before == after and shape_gate.passed),
        {
            "bit_exact_raw_replay": bit_exact,
            "parameters_unchanged": before == after,
            **dict(shape_gate.metrics),
        },
        None
        if bit_exact and before == after and shape_gate.passed
        else "direct-template replay gate failed",
    )
    run = _new_run(config)
    command_path, raw_path = (
        run.root / "direct_template_commands.pt",
        run.root / "direct_template_raw_cmos.pt",
    )
    atomic_torch_save(command_path, commands)
    atomic_torch_save(raw_path, raw)
    review = write_review_artifacts(
        root=run.root,
        stem="direct_template_replay",
        kind="direct_template_replay",
        raw_cmos=raw,
        targets=_tensor(clean_manifest.targets.path, device=torch.device("cpu")),
        exposure_scale=config.prototype_calibration.fixed_exposure_scale,
    )
    manifest = {
        "schema_version": 2,
        "control_evidence": oracle.evidence_hash,
        "control_report_sha256": file_sha256(report_path),
        "system_signature": _system_signature(config),
        "config_hash": canonical_sha256(config.model_dump(mode="json")),
        "checkpoint_sha256": checkpoint_hash,
        "clean_target_manifest": clean_manifest.manifest_hash,
        "commands": {
            "path": str(command_path),
            "sha256": file_sha256(command_path),
            "tensor_sha256": tensor_sha256(commands),
            "shape": list(commands.shape),
            "dtype": str(commands.dtype),
        },
        "raw_cmos": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
            "tensor_sha256": tensor_sha256(raw),
            "preview_sha256": review.preview_sha256,
        },
        "metrics": metrics,
        "parameters_before": before,
        "parameters_after": after,
    }
    atomic_write_json(run.root / "direct_template_command_manifest.json", manifest)
    status = "pending_review" if gate.passed else "failed"
    record = EvidenceRecordV2(
        2,
        "direct_template_replay",
        "diagnostic",
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "control_evidence": oracle.evidence_hash,
            "control_report": file_sha256(report_path),
            "shape_checkpoint": checkpoint_hash,
            "clean_target_manifest": clean_manifest.manifest_hash,
            "commands": file_sha256(command_path),
            "raw_cmos": file_sha256(raw_path),
            "preview": review.preview_sha256,
        },
        (gate,),
        ("admit_direct_template_replay",) if gate.passed else (),
        None if gate.passed else "direct-template automatic replay gate failed",
    )
    _write_record(run, "direct_template_replay_evidence.json", record)
    print(run.root)
    return 0


def _admit_direct_template_replay(args: argparse.Namespace, config: V2Config) -> int:
    source = _record(args.replay_evidence)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    expected = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
    )
    source.validate()
    if (
        source.evidence_kind != "direct_template_replay"
        or source.tier != "diagnostic"
        or source.status != "pending_review"
        or "admit_direct_template_replay" not in source.permitted_next_actions
    ):
        raise ValueError(
            "direct-template replay evidence cannot authorize manual admission"
        )
    source.context.assert_matches(expected)
    manifest = json.loads(Path(args.command_manifest).read_text(encoding="utf-8"))
    raw = _tensor(args.replay_raw_cmos, device=torch.device("cpu"))
    if manifest.get("raw_cmos", {}).get("tensor_sha256") != tensor_sha256(
        raw
    ) or source.input_hashes.get("raw_cmos") != file_sha256(args.replay_raw_cmos):
        raise ValueError(
            "direct-template replay raw-CMOS content does not match its manifest/evidence"
        )
    preview_hash = file_sha256(args.replay_preview)
    if preview_hash != manifest.get("raw_cmos", {}).get(
        "preview_sha256"
    ) or preview_hash != source.input_hashes.get("preview"):
        raise ValueError(
            "direct-template replay preview content does not match its manifest/evidence"
        )
    review, review_hash = load_manual_review(
        args.manual_review,
        kind="direct_template_replay",
        tensor_hash=tensor_sha256(raw),
        preview_hash=preview_hash,
    )
    readable = tuple(review.readable_labels) == tuple(range(10))
    gate = GateResult(
        "direct_template_manual_readability",
        readable,
        {"all_labels_readable": readable},
        None if readable else "manual review did not approve labels 0-9",
    )
    run = _new_run(config)
    actions = ("freeze_direct_template_prototype",) if gate.passed else ()
    record = EvidenceRecordV2(
        2,
        "direct_template_control_demonstrated",
        "diagnostic",
        "diagnostic" if gate.passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "replay_evidence": source.evidence_hash,
            "command_manifest": file_sha256(args.command_manifest),
            "raw_cmos": file_sha256(args.replay_raw_cmos),
            "preview": preview_hash,
            "manual_review": review_hash,
        },
        (source.gates[0], gate),
        actions,
        None if gate.passed else "direct-template manual readability gate failed",
    )
    _write_record(run, "direct_template_admission_evidence.json", record)
    print(run.root)
    return 0


def _freeze_direct_template_prototype(
    args: argparse.Namespace, config: V2Config
) -> int:
    """Freeze a manually admitted amplitude command bank without creating a model checkpoint."""
    admission = _record(args.admission_evidence)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    expected = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
    )
    require_evidence_action(
        admission,
        kind="direct_template_control_demonstrated",
        action="freeze_direct_template_prototype",
        expected_context=expected,
        allow_diagnostic=True,
    )
    if not all(gate.passed for gate in admission.gates):
        raise ValueError("direct-template admission contains an unsuccessful gate")
    replay = _record(args.replay_evidence)
    replay.validate()
    if (
        replay.evidence_kind != "direct_template_replay"
        or replay.status != "pending_review"
        or admission.input_hashes.get("replay_evidence") != replay.evidence_hash
    ):
        raise ValueError(
            "direct-template admission is not bound to the supplied replay evidence"
        )
    replay.context.assert_matches(expected)
    manifest_path = Path(args.command_manifest)
    if admission.input_hashes.get("command_manifest") != file_sha256(manifest_path):
        raise ValueError(
            "direct-template command manifest does not match admission evidence"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = _tensor(args.commands, device=torch.device("cpu"))
    saved_raw = _tensor(args.raw_cmos, device=torch.device("cpu"))
    if (
        manifest.get("commands", {}).get("sha256") != file_sha256(args.commands)
        or manifest.get("raw_cmos", {}).get("sha256") != file_sha256(args.raw_cmos)
        or manifest.get("commands", {}).get("tensor_sha256") != tensor_sha256(commands)
        or manifest.get("raw_cmos", {}).get("tensor_sha256") != tensor_sha256(saved_raw)
    ):
        raise ValueError(
            "direct-template command/raw artifacts do not match their manifest"
        )
    if (
        tuple(commands.shape) != (10, 1, *config.system.slm_shape)
        or saved_raw.shape != commands.shape
        or commands.dtype != torch.float32
    ):
        raise ValueError(
            "direct-template command/raw geometry does not match the formal SLM"
        )
    if (
        not torch.isfinite(commands).all()
        or bool((commands < 0.0).any())
        or bool((commands > 1.0).any())
    ):
        raise ValueError(
            "direct-template commands violate the bounded amplitude contract"
        )
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    load_prototype_warm_start(
        args.shape_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    before = state_content_hash(bundle.state_dict())
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for label in range(10):
            rows.append(
                bundle.system.forward(
                    bundle.system.command(commands[label : label + 1].to(device))
                )
                .raw_camera.tensor.detach()
                .cpu()
            )
    replay_raw = torch.cat(rows)
    after = state_content_hash(bundle.state_dict())
    bit_exact = tensor_sha256(replay_raw) == tensor_sha256(saved_raw)
    supports = (commands[:, 0] >= 0.2).to(torch.float32)
    gate = GateResult(
        "direct_template_prototype_frozen",
        bit_exact and before == after and bool(torch.isfinite(replay_raw).all().item()),
        {
            "bit_exact_raw_replay": bit_exact,
            "parameters_unchanged": before == after,
            "finite": bool(torch.isfinite(replay_raw).all().item()),
        },
        None
        if bit_exact and before == after
        else "frozen direct-template replay changed content or parameters",
    )
    run = _new_run(config)
    command_path, raw_path, support_path = (
        run.root / "frozen_direct_template_commands.pt",
        run.root / "frozen_direct_template_raw_cmos.pt",
        run.root / "frozen_direct_template_supports.pt",
    )
    atomic_torch_save(command_path, commands)
    atomic_torch_save(raw_path, replay_raw)
    atomic_torch_save(support_path, supports)
    artifact = DirectTemplatePrototypeArtifactV2(
        2,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        clean_manifest.manifest_hash,
        checkpoint_hash,
        admission.evidence_hash,
        replay.evidence_hash,
        config.prototype_calibration.fixed_exposure_scale,
        tensor_artifact_ref(
            command_path,
            role="frozen_direct_template_commands",
            git_commit=run.git_commit,
        ),
        tensor_artifact_ref(
            raw_path, role="frozen_direct_template_raw_cmos", git_commit=run.git_commit
        ),
        tensor_artifact_ref(
            support_path,
            role="frozen_direct_template_support",
            git_commit=run.git_commit,
        ),
    )
    artifact.validate()
    artifact_path = run.root / "direct_template_prototype_artifact.json"
    atomic_write_json(
        artifact_path, artifact.to_dict() | {"artifact_hash": artifact.artifact_hash}
    )
    record = EvidenceRecordV2(
        2,
        "direct_template_prototype",
        "diagnostic",
        "diagnostic" if gate.passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission_evidence": admission.evidence_hash,
            "replay_evidence": replay.evidence_hash,
            "command_manifest": file_sha256(manifest_path),
            "shape_checkpoint": checkpoint_hash,
            "artifact": file_sha256(artifact_path),
            "commands": file_sha256(command_path),
            "raw_cmos": file_sha256(raw_path),
            "supports": file_sha256(support_path),
        },
        (gate,),
        ("diagnostic_local_capacity_upper_bound",) if gate.passed else (),
        None if gate.passed else "direct-template prototype freeze failed",
    )
    _write_record(run, "direct_template_prototype_evidence.json", record)
    print(run.root)
    return 0


def _direct_template_local_capacity(args: argparse.Namespace, config: V2Config) -> int:
    """Run the fixed 8D capacity upper bound around immutable admitted commands."""
    if config.capacity_audit is None:
        raise ValueError(
            "direct-template local capacity requires capacity_audit config"
        )
    frozen = _record(args.prototype_evidence)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    expected = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
    )
    require_evidence_action(
        frozen,
        kind="direct_template_prototype",
        action="diagnostic_local_capacity_upper_bound",
        expected_context=expected,
        allow_diagnostic=True,
    )
    artifact_path = Path(args.prototype_artifact)
    artifact = DirectTemplatePrototypeArtifactV2.from_dict(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    artifact.validate()
    if artifact.artifact_hash != json.loads(
        artifact_path.read_text(encoding="utf-8")
    ).get("artifact_hash"):
        raise ValueError("direct-template prototype artifact hash mismatch")
    if (
        artifact.system_signature != _system_signature(config)
        or artifact.config_hash != canonical_sha256(config.model_dump(mode="json"))
        or artifact.clean_target_manifest_hash != clean_manifest.manifest_hash
        or artifact.shape_checkpoint_sha256 != checkpoint_hash
        or artifact.source_admission_evidence_hash
        != frozen.input_hashes.get("admission_evidence")
        or frozen.input_hashes.get("artifact") != file_sha256(artifact_path)
    ):
        raise ValueError(
            "direct-template prototype artifact does not match evidence/context"
        )
    commands = _tensor(artifact.commands.path, device=torch.device("cpu"))
    supports = _tensor(artifact.supports.path, device=torch.device("cpu"))
    if tuple(commands.shape) != (10, 1, *config.system.slm_shape) or supports.shape != (
        10,
        *config.system.slm_shape,
    ):
        raise ValueError(
            "direct-template artifact geometry does not match the configured system"
        )
    device = torch.device(config.run.device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    load_prototype_warm_start(
        args.shape_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    commands_device, supports_device = commands.to(device), supports.to(device)
    audit = config.capacity_audit
    state_before = state_content_hash(bundle.state_dict())
    responses, unchanged_response, before_hash, response_hash = (
        finite_difference_response_matrices(
            bundle.system,
            commands_device,
            supports_device,
            candidate_count=audit.dct_modes,
            epsilon=audit.finite_difference_epsilon,
            pixel_pitch_m=(
                config.system.slm_pixel_pitch_m,
                config.system.slm_pixel_pitch_m,
            ),
            max_cycles_per_mm=audit.max_cycles_per_mm,
            microbatch_size=config.execution.capacity_microbatch_size,
        )
    )
    basis = build_code_basis(
        responses,
        supports,
        candidate_count=audit.dct_modes,
        seed=audit.seed,
        source_hash=artifact.artifact_hash,
        pixel_pitch_m=(
            config.system.slm_pixel_pitch_m,
            config.system.slm_pixel_pitch_m,
        ),
        max_cycles_per_mm=audit.max_cycles_per_mm,
    )
    unclamped, slm_intensity, raw_camera, coefficients = replay_capacity(
        bundle.system,
        commands_device,
        basis,
        supports_device,
        codes_per_label=audit.codes_per_label,
        command_gain=audit.command_gain,
        microbatch_size=config.execution.capacity_microbatch_size,
    )
    after_hash = state_content_hash(bundle.state_dict())
    unchanged = unchanged_response and state_before == after_hash
    trace = bundle.system.forward(
        bundle.system.command(commands_device[:1]), return_optical_trace=True
    ).optical_trace
    provenance_complete = trace is not None and all(
        field.provenance.upstream_plane is not None
        or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"}
        for field in trace
    )
    slm_factor, camera_factor, support_energy, clipping = capacity_metrics(
        unclamped, slm_intensity, raw_camera, supports
    )
    finite = bool(
        torch.isfinite(responses).all().item()
        and torch.isfinite(unclamped).all().item()
        and torch.isfinite(raw_camera).all().item()
    )
    base_gate = capacity_gate(
        CapacityMetrics(
            slm_factor,
            camera_factor,
            support_energy,
            clipping,
            unchanged,
            False,
            provenance_complete,
        ),
        frozen.gates[0],
    )
    gate = GateResult(
        "direct_template_local_capacity",
        base_gate.passed and finite,
        dict(base_gate.metrics) | {"finite": finite},
        None
        if base_gate.passed and finite
        else "direct-template local capacity upper bound failed",
    )
    run = _new_run(config)
    basis_path, replay_path = (
        run.root / "direct_template_local_code_basis.pt",
        run.root / "direct_template_local_capacity_replay.pt",
    )
    basis.save(str(basis_path))
    atomic_torch_save(
        replay_path,
        {
            "unclamped_commands": unclamped,
            "physical_slm_intensity": slm_intensity,
            "raw_camera": raw_camera,
            "coefficients": coefficients,
            "supports": supports.cpu(),
            "responses": responses,
            "parameter_before_hash": state_before,
            "finite_difference_parameter_before_hash": before_hash,
            "parameter_after_response_hash": response_hash,
            "parameter_after_hash": after_hash,
            "optimizer_created": False,
            "provenance_complete": provenance_complete,
        },
    )
    atomic_write_json(
        run.root / "direct_template_local_capacity_report.json",
        {
            "diagnostic_only": True,
            "contract": {
                "dct_modes": audit.dct_modes,
                "selected_basis_dim": audit.selected_basis_dim,
                "max_cycles_per_mm": audit.max_cycles_per_mm,
                "codes_per_label": audit.codes_per_label,
                "seed": audit.seed,
                "finite_difference_epsilon": audit.finite_difference_epsilon,
                "command_gain": audit.command_gain,
                "microbatch_size": config.execution.capacity_microbatch_size,
            },
            "metrics": dict(gate.metrics),
            "basis_hash": basis.artifact_hash,
            "parameters_unchanged": unchanged,
            "optimizer_created": False,
            "provenance_complete": provenance_complete,
        },
    )
    context = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
        basis_hash=basis.artifact_hash,
    )
    record = EvidenceRecordV2(
        2,
        "direct_template_local_capacity",
        "diagnostic",
        "diagnostic" if gate.passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "prototype_evidence": frozen.evidence_hash,
            "prototype_artifact": file_sha256(artifact_path),
            "shape_checkpoint": checkpoint_hash,
            "clean_target_manifest": clean_manifest.manifest_hash,
            "basis": basis.artifact_hash,
            "basis_file": file_sha256(basis_path),
            "replay": file_sha256(replay_path),
            "responses": tensor_sha256(responses),
        },
        (gate,),
        (),
        None if gate.passed else "direct-template local capacity gate failed",
    )
    _write_record(run, "direct_template_local_capacity_evidence.json", record)
    print(run.root)
    return 0


def _rebind_manifest(args: argparse.Namespace, config: V2Config) -> int:
    """Re-emit a verified input manifest without touching its tensor payload."""
    _require_clean_acceptance(config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("rebound manifest output already exists")
    source = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if args.kind == "clean-target":
        old = CleanTargetManifestV2.from_dict(source)
        old.validate()
        targets = torch.load(old.targets.path, map_location="cpu", weights_only=False)
        reference = tensor_artifact_ref(
            old.targets.path, role="clean_mnist_targets", git_commit=commit
        )
        manifest = CleanTargetManifestV2(
            2,
            old.dataset_root,
            old.dataset_split,
            old.samples_per_label,
            old.labels,
            reference,
            tensor_sha256(targets),
            tuple(targets.shape),
            str(targets.dtype),
            canonical_sha256(config.model_dump(mode="json")),
            commit,
            _system_signature(config),
            canonical_sha256(config.model_dump(mode="json")),
        )
        manifest.validate()
    else:
        pair = _pair_manifest(args.teacher_manifest)
        legacy_hash = _legacy_export_contract(args.legacy_train_config, config)
        old = TeacherTargetManifestV2.from_dict(source)
        old.validate(formal=config.system.is_formal)
        if Path(old.dataset.path).resolve() != Path(args.dataset).resolve():
            raise ValueError("teacher source manifest dataset does not match --dataset")
        scoped_export_hash = _teacher_export_contract_hash(config, pair, legacy_hash)
        manifest = build_teacher_target_manifest(
            old.dataset.path,
            pair,
            formal=config.system.is_formal,
            generated_git_commit=commit,
            v2_config_hash=canonical_sha256(config.model_dump(mode="json")),
            legacy_export_config_hash=legacy_hash,
            system_contract_hash=_system_signature(config),
            teacher_export_contract_hash=scoped_export_hash,
        )
    atomic_write_json(
        output, manifest.to_dict() | {"manifest_hash": manifest.manifest_hash}
    )
    print(output)
    return 0


def _supersede_reachability(args: argparse.Namespace, config: V2Config) -> int:
    """Content-bind an invalidation without mutating historical evidence."""
    old = Path(args.old_evidence)
    payload = json.loads(old.read_text(encoding="utf-8"))
    if payload.get("evidence_kind") != "prototype_reachability":
        raise ValueError("only prototype reachability evidence can be superseded here")
    run = _new_run(config)
    record = EvidenceRecordV2(
        2,
        "prototype_reachability_supersession",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config),
        {"superseded_reachability": file_sha256(old)},
        (
            GateResult(
                "historical_harness_invalid", True, {"immutable_historical_file": True}
            ),
        ),
        (),
        "superseded evidence used plane intensity means, a similarity ratio, and no per-label directional gate",
    )
    _write_record(run, "prototype_reachability_supersession_evidence.json", record)
    atomic_write_json(
        run.root / "prototype_reachability_supersession.json",
        {
            "superseded_path": str(old.resolve()),
            "superseded_sha256": file_sha256(old),
            "superseded_evidence_hash": canonical_sha256(payload),
            "classification": "harness-invalid",
            "reason": record.invalidation_reason,
        },
    )
    print(run.root)
    return 0


def _dispatch_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.architecture == "conditional_identity_logit_amplitude_v2":
        required = (
            "dataset_root",
            "evaluator_checkpoint",
            "initialization_checkpoint",
            "injection_evidence",
            "observability_evidence",
            "semantic_recovery_evidence",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(
                f"identity conditional train requires: {', '.join('--' + item.replace('_', '-') for item in missing)}"
            )
        return conditional_train(args, config)
    if config.architecture in {
        "conditional_global_amplitude_v1",
        "conditional_identity_logit_amplitude_v2",
        "conditional_class_conditioned_modes_v3",
        "conditional_parity_stabilized_modes_v4",
        "conditional_multi_anchor_local_modes_v5",
        "conditional_pca_transport_codesign_v6",
        "conditional_pca_self_imaging_transport_v6_2",
        "conditional_pca_formal_spacing_self_imaging_v6_3",
        "conditional_pca_near_conjugate_relay_v6_4",
    }:
        if config.architecture == "conditional_multi_anchor_local_modes_v5":
            raise ValueError(
                "v5 training must use student train-multi-anchor with its passed smoke evidence"
            )
        required = (
            "dataset_root",
            "evaluator_checkpoint",
            "observability_evidence",
            "controllability_evidence",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(
                f"conditional student train requires: {', '.join('--' + item.replace('_', '-') for item in missing)}"
            )
        return conditional_train(args, config)
    required = (
        "prototype_evidence",
        "capacity_evidence",
        "code_basis",
        "prototype_checkpoint",
        "teacher_target_manifest",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(
            f"legacy student train requires: {', '.join('--' + item.replace('_', '-') for item in missing)}"
        )
    return _train(args, config)


def _dispatch_sample(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.architecture in {
        "conditional_global_amplitude_v1",
        "conditional_identity_logit_amplitude_v2",
        "conditional_class_conditioned_modes_v3",
        "conditional_parity_stabilized_modes_v4",
        "conditional_multi_anchor_local_modes_v5",
        "conditional_pca_transport_codesign_v6",
        "conditional_pca_self_imaging_transport_v6_2",
        "conditional_pca_formal_spacing_self_imaging_v6_3",
        "conditional_pca_near_conjugate_relay_v6_4",
    }:
        return sample_conditional(args, config)
    if not args.code_basis:
        raise ValueError("legacy student sample requires --code-basis")
    return _sample(args, config)


def _dispatch_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.architecture in {
        "conditional_global_amplitude_v1",
        "conditional_identity_logit_amplitude_v2",
        "conditional_class_conditioned_modes_v3",
        "conditional_parity_stabilized_modes_v4",
        "conditional_multi_anchor_local_modes_v5",
        "conditional_pca_formal_spacing_self_imaging_v6_3",
        "conditional_pca_near_conjugate_relay_v6_4",
    }:
        required = (
            "training_evidence",
            "checkpoint",
            "dataset_root",
            "evaluator_checkpoint",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(
                f"conditional student evaluate requires: {', '.join('--' + item.replace('_', '-') for item in missing)}"
            )
        return conditional_evaluate(args, config)
    required = (
        "training_evidence",
        "code_basis",
        "checkpoint",
        "teacher_target_manifest",
        "output",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(
            f"legacy student evaluate requires: {', '.join('--' + item.replace('_', '-') for item in missing)}"
        )
    return _evaluate(args, config)
