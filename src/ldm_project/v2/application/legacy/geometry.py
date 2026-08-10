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
from .serialization import _trusted_legacy_torch_load

_CONTROL_ORACLE_ARMS = ("formal_amplitude_only", "diagnostic_complex_slm")
_CONTROL_ORACLE_STEPS = 400
_CONTROL_ORACLE_LR = 1.0e-2
_CONTROL_ORACLE_PROGRESS_INTERVAL = 40

_DEPENDENCIES_INSTALLED = False


def _ensure_dependencies() -> None:
    global _DEPENDENCIES_INSTALLED
    if not _DEPENDENCIES_INSTALLED:
        _install_dependencies(globals())
        from . import teacher as _teacher
        from . import prototypes as _prototypes

        globals().update(
            {name: value for name, value in _teacher.__dict__.items() if not name.startswith("__")}
        )
        globals().update(
            {name: value for name, value in _prototypes.__dict__.items() if not name.startswith("__")}
        )
        _DEPENDENCIES_INSTALLED = True

def _prototype_topology(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    """Frozen 2x2 template/phase isolation.  It intentionally creates no optimizer."""
    if config.prototype_calibration is None:
        raise ValueError("prototype topology requires prototype calibration config")
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    clean = _tensor(clean_manifest.targets.path, device=device)
    payload = _trusted_legacy_torch_load(
        args.prototype_checkpoint, map_location="cpu"
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state_dict"), dict)
        or not isinstance(payload.get("manifest"), dict)
    ):
        raise ValueError("prototype topology requires a v2 prototype checkpoint")
    source_manifest = CheckpointManifestV2(**payload["manifest"])
    if source_manifest.system_signature != _system_signature(config):
        raise ValueError(
            "prototype topology checkpoint belongs to another physical system"
        )
    state = payload["state_dict"]
    labels = torch.arange(10, device=device)

    def build(*, trained_template: bool, trained_phase: bool) -> StudentBundle:
        bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
        with torch.no_grad():
            if trained_template:
                bundle.generator.template_logits.copy_(
                    state["generator.template_logits"].to(device)
                )
            else:
                bundle.generator.template_logits.copy_(
                    torch.logit(clean[:, 0].sqrt().clamp(1.0e-4, 1.0 - 1.0e-4))
                )
            if trained_phase:
                bundle.system.encoder.phase.copy_(
                    state["system.encoder.phase"].to(device)
                )
                bundle.system.decoder.phase.copy_(
                    state["system.decoder.phase"].to(device)
                )
        return bundle

    def replay(
        bundle: StudentBundle,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        tuple[torch.Tensor, ...],
        tuple[ComplexField, ...],
    ]:
        raw_rows, crops, trace_rows, trace_fields = (
            [],
            {"encoder_crop_retention": [], "decoder_crop_retention": []},
            None,
            None,
        )
        with torch.no_grad():
            for label in labels:
                result = bundle.system.forward(
                    bundle.system.command(
                        bundle.generator.prototype_command(label[None])
                    ),
                    return_optical_trace=True,
                )
                raw_rows.append(result.raw_camera.tensor.detach().cpu())
                for name, value in result.crop_statistics.items():
                    crops[name].append(value.detach().cpu())
                if trace_rows is None:
                    trace_rows = [[] for _ in result.optical_trace or ()]
                    trace_fields = result.optical_trace
                for bucket, field in zip(
                    trace_rows or [], result.optical_trace or (), strict=True
                ):
                    bucket.append(field.tensor.abs().square().detach().cpu())
        return (
            torch.cat(raw_rows),
            {name: torch.cat(values) for name, values in crops.items()},
            tuple(torch.cat(bucket) for bucket in trace_rows or []),
            tuple(trace_fields or ()),
        )

    def summary(
        raw: torch.Tensor,
        crops: dict[str, torch.Tensor],
        trace: tuple[torch.Tensor, ...],
        trace_fields: tuple[ComplexField, ...],
    ) -> dict[str, object]:
        normalized = (
            raw.to(device) / config.prototype_calibration.fixed_exposure_scale
        ).clamp(0.0, 1.0)
        matrix = prototype_correlation_matrix(normalized, clean)
        margins = prototype_margins(matrix)
        roi = clean >= 0.2
        roi_scores = []
        for label in range(10):
            masked_prediction = normalized[label] * roi[label]
            roi_scores.append(
                torch.stack(
                    [
                        correlation(masked_prediction, clean[target] * roi[target])[0]
                        for target in range(10)
                    ]
                )
            )
        roi_matrix = torch.stack(roi_scores)
        trace_for_report = tuple(
            ComplexField(
                value.to(device).sqrt().to(torch.complex64),
                field.plane,
                field.provenance,
            )
            for value, field in zip(trace, trace_fields, strict=True)
        )
        planes, first_collapse = _trace_plane_report(trace_for_report)
        return {
            "clean_corr": float(correlation(normalized, clean).mean().item()),
            "hard_dice": float(foreground_dice(normalized, clean).mean().item()),
            "soft_dice": float(
                (
                    (2.0 * (normalized * clean).flatten(1).sum(dim=1) + 1.0e-6)
                    / (
                        normalized.square().flatten(1).sum(dim=1)
                        + clean.square().flatten(1).sum(dim=1)
                        + 1.0e-6
                    )
                )
                .mean()
                .item()
            ),
            "standard_accuracy": float(
                (matrix.argmax(dim=1) == labels).to(torch.float32).mean().item()
            ),
            "confusion": [int(value) for value in matrix.argmax(dim=1).cpu().tolist()],
            "full_frame_margins": [float(value) for value in margins.cpu().tolist()],
            "roi_accuracy_diagnostic_only": float(
                (roi_matrix.argmax(dim=1) == labels).to(torch.float32).mean().item()
            ),
            "roi_margins_diagnostic_only": [
                float(value) for value in prototype_margins(roi_matrix).cpu().tolist()
            ],
            "foreground_hole": float(
                foreground_hole_ratio(normalized, clean).mean().item()
            ),
            "background_speckle": float(
                background_speckle_ratio(normalized, clean).mean().item()
            ),
            "high_saturation": float(
                high_saturation_ratio(
                    raw.to(device), config.prototype_calibration.fixed_exposure_scale
                ).item()
            ),
            "crop_retention": {
                name: [float(value) for value in values.flatten().tolist()]
                for name, values in crops.items()
            },
            "plane_report": planes,
            "first_material_collapse_plane": first_collapse,
        }

    arms = {
        "sqrt_template_zero_phase": (False, False),
        "trained_template_zero_phase": (True, False),
        "sqrt_template_trained_phase": (False, True),
        "trained_template_trained_phase": (True, True),
    }
    outputs: dict[str, dict[str, object]] = {}
    all_unchanged = True
    run = _new_run(config)
    for name, (trained_template, trained_phase) in arms.items():
        bundle = build(trained_template=trained_template, trained_phase=trained_phase)
        before = state_content_hash(bundle.state_dict())
        raw, crops, trace, trace_fields = replay(bundle)
        after = state_content_hash(bundle.state_dict())
        all_unchanged = all_unchanged and before == after
        raw_path = run.root / f"topology_{name}_raw_cmos.pt"
        atomic_torch_save(raw_path, raw)
        review = write_review_artifacts(
            root=run.root,
            stem=f"topology_{name}",
            kind="prototype_topology_diagnostic",
            raw_cmos=raw,
            targets=clean.detach().cpu(),
            exposure_scale=config.prototype_calibration.fixed_exposure_scale,
        )
        outputs[name] = summary(raw, crops, trace, trace_fields) | {
            "raw_cmos": file_sha256(raw_path),
            "preview": review.preview_sha256,
            "parameters_unchanged": before == after,
        }
    joint = outputs["trained_template_trained_phase"]
    best_arm = max(outputs, key=lambda name: float(outputs[name]["standard_accuracy"]))
    classification = (
        "joint-separation-improving-but-topology-unresolved"
        if best_arm == "trained_template_trained_phase"
        and float(joint["standard_accuracy"]) < 1.0
        else f"component-or-baseline-preferred:{best_arm}"
    )
    gate = GateResult(
        "topology_isolation_parameters_unchanged",
        all_unchanged,
        {"parameters_unchanged": all_unchanged},
    )
    record = EvidenceRecordV2(
        2,
        "prototype_topology",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            dataset_hash=clean_manifest.targets.sha256,
            checkpoint_hash=file_sha256(args.prototype_checkpoint),
        ),
        {
            "clean_target_manifest": clean_manifest.manifest_hash,
            "prototype_checkpoint": file_sha256(args.prototype_checkpoint),
        },
        (gate,),
        (),
        "topology isolation is diagnostic only; ROI metrics cannot authorize acceptance",
    )
    _write_record(run, "prototype_topology_evidence.json", record)
    atomic_write_json(
        run.root / "prototype_topology_report.json",
        {
            "classification": classification,
            "arms": outputs,
            "acceptance_note": "ROI results and topology proxies are diagnostic only; fixed-exposure raw previews require human judgement.",
        },
    )
    print(run.root)
    return 0


def _geometry_forward(
    system: FormalAmplitudeFiniteCropIdentitySystem, command: torch.Tensor, *, arm: str
) -> tuple[torch.Tensor, tuple[ComplexField, ...], dict[str, torch.Tensor]]:
    """Diagnostic decoder-only counterfactual with legal, re-derived physical planes."""
    slm = system.slm(system.command(command))
    front = system.front(slm)
    encoded, encoder_records, encoder_stats = system.encoder(front, "encoder")
    ols = identity_ols(encoded)
    if arm == "trained_decoder_1500um":
        decoded, decoder_records, decoder_stats = system.decoder(ols, "decoder")
    elif arm in {"zero_phase_decoder_1500um", "zero_phase_decoder_0250um"}:
        spacing = 1.50e-3 if arm.endswith("1500um") else 0.25e-3
        decoder = FiniteCropPhaseStack(
            layers=system.config.decoder_layers,
            active_shape=system.config.metasurface_shape,
            propagation_shape=(system.config.propagation_grid_size,) * 2,
            spacing_m=spacing,
        ).to(command.device)
        decoded, decoder_records, decoder_stats = decoder(ols, "diagnostic_decoder")
    elif arm == "decoder_bypass":
        decoded, decoder_records, decoder_stats = (
            ols,
            (),
            {"crop_retention": torch.ones(command.shape[0], 0, device=command.device)},
        )
    else:
        raise ValueError(f"unknown decoder geometry arm: {arm}")
    chip_plane = PlaneSpec(
        f"{arm}_chip",
        system.config.metasurface_shape,
        (system.config.metasurface_pixel_pitch_m,) * 2,
        system.config.wavelength_m,
        decoded.plane.z_m,
        "diagnostic_chip",
    )
    chip = ComplexField(
        decoded.tensor,
        chip_plane,
        OperationProvenance(
            "diagnostic_decoder_geometry",
            decoded.plane.plane_id,
            None,
            {
                "arm": arm,
                "decoder_spacing_m": None
                if arm == "decoder_bypass"
                else (1.50e-3 if arm.endswith("1500um") else 0.25e-3),
            },
        ),
    )
    camera_plane = PlaneSpec(
        f"{arm}_camera",
        system.config.slm_shape,
        (system.config.slm_pixel_pitch_m,) * 2,
        system.config.wavelength_m,
        chip_plane.z_m + sum(system.back.segments_m),
        "diagnostic_camera",
    )
    back = CoherentImagingRelay(
        chip_plane,
        camera_plane,
        system.config.back_magnification,
        name=f"diagnostic_{arm}_back_b0",
        segments_m=system.back.segments_m,
        focal_lengths_m=system.back.focal_lengths_m,
        clear_aperture_m=system.back.clear_aperture_m,
    )
    camera = back(chip)
    return (
        camera.tensor.abs().square(),
        (slm, front, *encoder_records, ols, *decoder_records, camera),
        {
            "encoder_crop_retention": encoder_stats["crop_retention"],
            "decoder_crop_retention": decoder_stats["crop_retention"],
        },
    )


def _prototype_geometry_upper_bound(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    if config.prototype_calibration is None:
        raise ValueError(
            "prototype geometry upper bound requires prototype calibration config"
        )
    source = _record(args.shape_evidence)
    source.validate()
    if source.evidence_kind != "prototype_calibration" or source.status not in {
        "failed",
        "diagnostic",
    }:
        raise ValueError(
            "geometry upper bound requires a failed prototype calibration record"
        )
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    clean = _tensor(clean_manifest.targets.path, device=device)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    if (
        source.context.checkpoint_hash != checkpoint_hash
        or source.context.dataset_hash != clean_manifest.targets.sha256
    ):
        raise ValueError(
            "shape checkpoint or clean targets do not match shape evidence"
        )
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    load_prototype_warm_start(
        args.shape_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    labels = torch.arange(10, device=device)
    commands = bundle.generator.prototype_command(labels).detach()
    before = state_content_hash(bundle.state_dict())
    arms = (
        "trained_decoder_1500um",
        "zero_phase_decoder_1500um",
        "decoder_bypass",
        "zero_phase_decoder_0250um",
    )
    run = _new_run(config)
    outputs: dict[str, dict[str, object]] = {}
    for arm in arms:
        raw_rows, crop_rows, trace_rows, trace_fields = (
            [],
            {"encoder_crop_retention": [], "decoder_crop_retention": []},
            None,
            None,
        )
        with torch.no_grad():
            for label in range(10):
                raw, trace, crops = _geometry_forward(
                    bundle.system, commands[label : label + 1], arm=arm
                )
                raw_rows.append(raw.detach().cpu())
                for name, value in crops.items():
                    crop_rows[name].append(value.detach().cpu())
                if trace_rows is None:
                    trace_rows, trace_fields = [[] for _ in trace], trace
                for bucket, field in zip(trace_rows or [], trace, strict=True):
                    bucket.append(field.tensor.abs().square().detach().cpu())
        raw = torch.cat(raw_rows)
        raw_path = run.root / f"geometry_{arm}_raw_cmos.pt"
        atomic_torch_save(raw_path, raw)
        review = write_review_artifacts(
            root=run.root,
            stem=f"geometry_{arm}",
            kind="prototype_geometry_diagnostic",
            raw_cmos=raw,
            targets=clean.detach().cpu(),
            exposure_scale=config.prototype_calibration.fixed_exposure_scale,
        )
        metrics, shape_gate = _prototype_shape_summary(
            raw,
            clean,
            exposure_scale=config.prototype_calibration.fixed_exposure_scale,
            device=device,
        )
        aggregated = tuple(
            ComplexField(
                torch.cat(bucket).to(device).sqrt().to(torch.complex64),
                field.plane,
                field.provenance,
            )
            for bucket, field in zip(trace_rows or [], trace_fields or (), strict=True)
        )
        planes, collapse = _trace_plane_report(aggregated)
        crops = {
            name: [float(item) for row in values for item in row.flatten().tolist()]
            for name, values in crop_rows.items()
        }
        outputs[arm] = {
            "metrics": metrics,
            "shape_gate": {
                "passed": shape_gate.passed,
                "reason": shape_gate.reason,
                "metrics": dict(shape_gate.metrics),
            },
            "crop_retention": crops,
            "plane_report": planes,
            "first_material_collapse_plane": collapse,
            "raw_cmos": file_sha256(raw_path),
            "preview": review.preview_sha256,
        }
    after = state_content_hash(bundle.state_dict())
    unchanged = before == after
    short_gate = (
        outputs["zero_phase_decoder_0250um"]["shape_gate"]["passed"] and unchanged
    )
    gate = GateResult(
        "decoder_geometry_upper_bound",
        bool(short_gate),
        {
            "short_decoder_full_shape_gate": bool(
                outputs["zero_phase_decoder_0250um"]["shape_gate"]["passed"]
            ),
            "parameters_unchanged": unchanged,
        },
    )
    next_actions = (
        ("manual_geometry_review",) if gate.passed else ("diagnostic_control_oracle",)
    )
    record = EvidenceRecordV2(
        2,
        "prototype_geometry_upper_bound",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            dataset_hash=clean_manifest.targets.sha256,
            checkpoint_hash=checkpoint_hash,
        ),
        {
            "shape_evidence": source.evidence_hash,
            "shape_checkpoint": checkpoint_hash,
            "clean_target_manifest": clean_manifest.manifest_hash,
        },
        (gate,),
        next_actions,
        None
        if gate.passed
        else "short-decoder geometry upper bound did not recover the full raw-CMOS shape gate",
    )
    _write_record(run, "prototype_geometry_upper_bound_evidence.json", record)
    atomic_write_json(
        run.root / "prototype_geometry_upper_bound_report.json",
        {
            "arms": outputs,
            "formal_contract_unchanged": True,
            "decoder025_diagnostic_only": True,
            "parameters_unchanged": unchanged,
            "decision": "pending_manual_geometry_review"
            if gate.passed
            else "prepare_bounded_control_oracle",
        },
    )
    print(run.root)
    return 0


def _prototype_control_oracle(args: argparse.Namespace, config: V2Config) -> int:
    _ensure_dependencies()
    """Bounded per-label control upper bound; diagnostic only and never a model checkpoint."""
    if config.prototype_calibration is None:
        raise ValueError(
            "prototype control oracle requires prototype calibration config"
        )
    geometry = _record(args.geometry_evidence)
    geometry.validate()
    if (
        geometry.evidence_kind != "prototype_geometry_upper_bound"
        or geometry.status != "diagnostic"
    ):
        raise ValueError("control oracle requires geometry upper-bound evidence")
    device = torch.device(config.run.device)
    clean_manifest = _clean_target_manifest(args.clean_target_manifest, config)
    checkpoint_hash = file_sha256(args.shape_checkpoint)
    expected_context = _context(
        config,
        dataset_hash=clean_manifest.targets.sha256,
        checkpoint_hash=checkpoint_hash,
    )
    require_evidence_action(
        geometry,
        kind="prototype_geometry_upper_bound",
        action="diagnostic_control_oracle",
        expected_context=expected_context,
        allow_diagnostic=True,
    )
    if (
        geometry.input_hashes.get("clean_target_manifest")
        != clean_manifest.manifest_hash
    ):
        raise ValueError("geometry evidence does not match the clean-target manifest")
    if geometry.input_hashes.get("shape_checkpoint") != checkpoint_hash:
        raise ValueError("geometry evidence does not match the shape checkpoint")
    clean = _tensor(clean_manifest.targets.path, device=device)
    bundle = StudentBundle(config, _zero_basis(config.system.slm_shape)).to(device)
    load_prototype_warm_start(
        args.shape_checkpoint,
        bundle,
        expected_system_signature=_system_signature(config),
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    before = state_content_hash(bundle.state_dict())
    exposure = config.prototype_calibration.fixed_exposure_scale
    run = _new_run(config)
    contract = {
        "arms": list(_CONTROL_ORACLE_ARMS),
        "labels": list(range(10)),
        "steps_per_label": _CONTROL_ORACLE_STEPS,
        "adam_learning_rate": _CONTROL_ORACLE_LR,
        "progress_interval_steps": _CONTROL_ORACLE_PROGRESS_INTERVAL,
        "fixed_exposure_scale": exposure,
        "initialization": "sqrt_clean_intensity_logit",
        "objective": {
            "raw_mse": 1.0,
            "correlation": 0.25,
            "soft_dice": 0.30,
            "background": 0.10,
        },
        "parameters_before": before,
        "geometry_evidence": geometry.evidence_hash,
    }
    atomic_write_json(run.root / "prototype_control_oracle_contract.json", contract)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def write_progress(
        *,
        status: str,
        arm: str | None = None,
        label: int | None = None,
        step: int | None = None,
        objective_value: float | None = None,
        error: str | None = None,
    ) -> None:
        memory: dict[str, int] = {}
        if device.type == "cuda":
            memory = {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            }
        atomic_write_json(
            run.root / "prototype_control_oracle_progress.json",
            {
                "status": status,
                "arm": arm,
                "label": label,
                "step": step,
                "objective": objective_value,
                "elapsed_seconds": time.perf_counter() - started,
                "optical_parameter_hash": state_content_hash(bundle.state_dict()),
                "cuda_memory": memory,
                "contract": contract,
                "error": error,
            },
        )

    def complex_forward(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        field = ComplexField(
            torch.polar(amplitude, phase).to(torch.complex64),
            bundle.system.slm_plane,
            OperationProvenance("diagnostic_complex_slm_control", None, None, {}),
        )
        front = bundle.system.front(field)
        encoded, _, _ = bundle.system.encoder(front, "encoder")
        decoded, _, _ = bundle.system.decoder(identity_ols(encoded), "decoder")
        chip = ComplexField(
            decoded.tensor, bundle.system.chip_plane, decoded.provenance
        )
        return bundle.system.back(chip).tensor.abs().square()

    def objective(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        normalized = (raw / exposure).clamp(0.0, 1.0)
        foreground = (target >= 0.20).to(normalized.dtype)
        weights = 1.0 + 2.0 * foreground
        mse = (
            (normalized - target).square() * weights
        ).sum() / weights.sum().clamp_min(1.0)
        soft = (2.0 * (normalized * target).flatten(1).sum(dim=1) + 1.0e-6) / (
            normalized.square().flatten(1).sum(dim=1)
            + target.square().flatten(1).sum(dim=1)
            + 1.0e-6
        )
        return (
            mse
            + 0.25 * (1.0 - _differentiable_correlation(normalized, target).mean())
            + 0.30 * (1.0 - soft.mean())
            + 0.10 * (normalized * (1.0 - foreground)).square().mean()
        )

    def optimize(kind: str) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
        commands, raws, history = [], [], []
        for label in range(10):
            initial = clean[label : label + 1, 0].sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
            logits = torch.nn.Parameter(torch.logit(initial))
            phase_logits = (
                torch.nn.Parameter(torch.zeros_like(initial))
                if kind == "diagnostic_complex_slm"
                else None
            )
            optimizer = torch.optim.Adam(
                [logits] + ([phase_logits] if phase_logits is not None else []),
                lr=_CONTROL_ORACLE_LR,
            )
            target = clean[label : label + 1]
            label_history: list[float] = []
            for step in range(1, _CONTROL_ORACLE_STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                amplitude = torch.sigmoid(logits).unsqueeze(1)
                raw = (
                    bundle.system.forward(
                        bundle.system.command(amplitude)
                    ).raw_camera.tensor
                    if phase_logits is None
                    else complex_forward(
                        amplitude, torch.pi * torch.tanh(phase_logits).unsqueeze(1)
                    )
                )
                loss = objective(raw, target)
                loss.backward()
                optimizer.step()
                value = float(loss.detach().item())
                label_history.append(value)
                if (
                    step % _CONTROL_ORACLE_PROGRESS_INTERVAL == 0
                    or step == _CONTROL_ORACLE_STEPS
                ):
                    write_progress(
                        status="running",
                        arm=kind,
                        label=label,
                        step=step,
                        objective_value=value,
                    )
            with torch.no_grad():
                amplitude = torch.sigmoid(logits).unsqueeze(1)
                raw = (
                    bundle.system.forward(
                        bundle.system.command(amplitude)
                    ).raw_camera.tensor
                    if phase_logits is None
                    else complex_forward(
                        amplitude, torch.pi * torch.tanh(phase_logits).unsqueeze(1)
                    )
                )
                commands.append(amplitude.detach().cpu())
                raws.append(raw.detach().cpu())
                label_history.append(float(objective(raw, target).item()))
                history.append(label_history)
        return torch.cat(commands), torch.cat(raws), history

    arms: dict[str, dict[str, object]] = {}
    try:
        for kind in _CONTROL_ORACLE_ARMS:
            commands, raw, history = optimize(kind)
            command_path, raw_path = (
                run.root / f"control_oracle_{kind}_commands.pt",
                run.root / f"control_oracle_{kind}_raw_cmos.pt",
            )
            atomic_torch_save(command_path, commands)
            atomic_torch_save(raw_path, raw)
            review = write_review_artifacts(
                root=run.root,
                stem=f"control_oracle_{kind}",
                kind="prototype_control_oracle",
                raw_cmos=raw,
                targets=clean.detach().cpu(),
                exposure_scale=exposure,
            )
            metrics, gate = _prototype_shape_summary(
                raw, clean, exposure_scale=exposure, device=device
            )
            arms[kind] = {
                "input_control": kind,
                "metrics": metrics,
                "shape_gate": {"passed": gate.passed, "metrics": dict(gate.metrics)},
                "objective_history_per_label": history,
                "commands": file_sha256(command_path),
                "raw_cmos": file_sha256(raw_path),
                "preview": review.preview_sha256,
            }
    except BaseException as error:
        write_progress(status="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    unchanged = before == state_content_hash(bundle.state_dict())
    amplitude_passed = bool(
        arms["formal_amplitude_only"]["shape_gate"]["passed"] and unchanged
    )
    complex_passed = bool(
        arms["diagnostic_complex_slm"]["shape_gate"]["passed"] and unchanged
    )
    decision = (
        "direct_template_replay_candidate"
        if amplitude_passed
        else (
            "amplitude_control_limited"
            if complex_passed
            else "control_oracle_not_demonstrated"
        )
    )
    gate = GateResult(
        "bounded_control_oracle",
        amplitude_passed,
        {
            "amplitude_only_passed": amplitude_passed,
            "complex_diagnostic_passed": complex_passed,
            "parameters_unchanged": unchanged,
        },
    )
    record = EvidenceRecordV2(
        2,
        "prototype_control_oracle",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            dataset_hash=clean_manifest.targets.sha256,
            checkpoint_hash=checkpoint_hash,
        ),
        {
            "geometry_evidence": geometry.evidence_hash,
            "shape_checkpoint": checkpoint_hash,
            "clean_target_manifest": clean_manifest.manifest_hash,
        },
        (gate,),
        (),
        decision,
    )
    _write_record(run, "prototype_control_oracle_evidence.json", record)
    atomic_write_json(
        run.root / "prototype_control_oracle_report.json",
        {
            "contract": contract,
            "arms": arms,
            "decision": decision,
            "parameters_before": before,
            "parameters_after": state_content_hash(bundle.state_dict()),
            "parameters_unchanged": unchanged,
            "diagnostic_only": True,
        },
    )
    write_progress(status="completed")
    print(run.root)
    return 0
