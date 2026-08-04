"""Evaluator-qualified 2+2 transport diagnostics.

This module intentionally starts with the read-only curvature baseline.  It
never treats historical v7.1/v7.2 failures as authorizations.
"""

from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from ...evidence import (
    EvidenceRecordV2,
    GateResult,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ...student.checkpoints import load_conditional_checkpoint, save_conditional_checkpoint, state_content_hash
from ...student.conditional import camera_code_fraction_loss, conditional_sinkhorn_loss, fixed_exposure_view, phase_total_variation, raw_from_generator_batched, relative_diversity_loss
from ...student.evaluator import (
    MNISTSplitManifest,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .expanded_manifold import _build_artifact, _source
from .continuous_audit import _lowpass_modes, _pca
from .support import (
    ConditionalBundle,
    EVALUATION_QUALIFIED_TRANSPORT_V11_ARCHITECTURE,
    _context,
    _new_run,
    _split,
    _write_evidence,
)

from .transport_aligned_manifold import (
    _device,
    _feature_targets,
    _fit_transport_mixing,
    _logit_delta,
    _memory,
    _relay_provenance,
)
from .pca_transport import _training_codes
from .evaluation_qualified_runtime import (
    ADMISSION_CONTRACT,
    SMOKE_CONTRACT,
    TANGENT_CONTRACT,
    admission_gate_dict,
    admission_values,
    load_evaluators,
    load_record,
    manifest as checkpoint_manifest,
)

EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE = EVALUATION_QUALIFIED_TRANSPORT_V11_ARCHITECTURE


CURVATURE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_evaluator_curvature_baseline_v1",
        "architecture": EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE,
        "epsilon_logit": 0.01,
        "candidate_directions": 32,
        "operational_directions": 8,
        "optimizer": False,
        "checkpoint": False,
    }
)
DESIGN_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_evaluation_qualified_transport_design_v1_1",
        "architecture": EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE,
        "components": 32,
        "directions": 8,
        "scale_bisection_steps": 20,
        "optimizer": False,
        "checkpoint": False,
    }
)


def _require(config) -> None:
    if config.architecture != EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE:
        raise ValueError(
            "evaluation-qualified stages require conditional_transport_aligned_evaluation_qualified_v1"
        )


def _record(path: str, kind: str, status: str = "passed") -> EvidenceRecordV2:
    return load_record(path, kind, status=status)


def _load_bundle(args, config, split_hash: str):
    # Compatibility wrapper for the read-only curvature stage.  All mutable
    # stages call the shared runtime directly.
    _, _, smooth, qualification, bundle = load_evaluators(args, config, split_hash)
    return qualification, bundle, smooth


def _ratio(
    plus: torch.Tensor, minus: torch.Tensor, zero: torch.Tensor
) -> dict[str, float]:
    numerator = float((plus + minus - 2 * zero).norm())
    denominator = float((plus - minus).norm().clamp_min(1e-12))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / denominator,
    }


def _summary(rows: list[dict[str, float]]) -> dict[str, object]:
    ratios = torch.tensor([row["ratio"] for row in rows])
    index = int(ratios.argmax())
    return {
        "p50": float(ratios.quantile(0.5)),
        "p90": float(ratios.quantile(0.9)),
        "p95": float(ratios.quantile(0.95)),
        "max": float(ratios[index]),
        "argmax_direction": index,
        "argmax": rows[index],
    }


def _capture(system, primary, smooth, command: torch.Tensor, exposure: float):
    forward = system.forward(system.command(command), return_optical_trace=True)
    trace = forward.optical_trace or ()
    ols = next(
        item.tensor for item in trace if item.provenance.operation == "identity_ols"
    )
    view = fixed_exposure_view(forward.raw_camera.tensor, exposure)
    return {
        "command": command,
        "ols_field": ols,
        "camera_field": trace[-1].tensor,
        "raw_intensity": forward.raw_camera.tensor,
        "fixed_exposure": view,
        "primary_feature": primary(view)[0],
        "smooth_feature": smooth(view)[0],
        "low_frequency": F.avg_pool2d(view, 2),
    }


def _measure(system, primary, smooth, center, directions, exposure: float):
    stages = {
        name: []
        for name in (
            "command",
            "ols_field",
            "camera_field",
            "raw_intensity",
            "fixed_exposure",
            "primary_feature",
            "smooth_feature",
            "low_frequency",
        )
    }
    for direction in directions:
        values = {
            name: _capture(
                system,
                primary,
                smooth,
                torch.sigmoid(
                    center.unsqueeze(0) + sign * 0.01 * direction.unsqueeze(0)
                ).unsqueeze(1),
                exposure,
            )
            for name, sign in (("plus", 1), ("minus", -1), ("zero", 0))
        }
        for stage in stages:
            stages[stage].append(
                _ratio(
                    values["plus"][stage], values["minus"][stage], values["zero"][stage]
                )
            )
    return {stage: _summary(rows) for stage, rows in stages.items()}


def conditional_evaluator_curvature_baseline(args, config) -> int:
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(
        images, labels, args.dataset_root
    )
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    primary, payload = load_frozen_evaluator(
        args.primary_evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("primary evaluator split mismatch")
    qualification, _, smooth = _load_bundle(args, config, split.manifest_hash)
    if "conditional-evaluator-curvature-baseline" not in qualification.permitted_next_actions:
        raise ValueError("evaluator qualification does not authorize curvature baseline")
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.electronic_initialization),
    )
    qualification.context.assert_matches(
        expected,
        fields=(
            "evaluator_checkpoint_hash",
            "dataset_split_manifest_hash",
            "exposure_scale",
        ),
    )
    source = _source(args.electronic_initialization)
    v7 = _record(
        args.v7_capacity_evidence, "conditional_nonlinear_expanded_manifold_capacity"
    )
    artifact = torch.load(
        args.v7_basis_artifact, map_location="cpu", weights_only=False
    )
    if not isinstance(artifact, dict) or v7.input_hashes.get(
        "basis_artifact"
    ) != file_sha256(args.v7_basis_artifact):
        raise ValueError("v7 basis evidence/artifact mismatch")
    regenerated, _ = _build_artifact(train_images, train_labels, source, config)
    if tensor_sha256(regenerated["basis"]) != tensor_sha256(artifact["basis"]):
        raise ValueError("v7 basis regeneration mismatch")
    selector = torch.zeros(10, 32, 8)
    selector[:, torch.arange(8), torch.arange(8)] = 1
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=source["carrier"],
            class_modes=regenerated["basis"],
            manifold_scales=source["scales"],
            transport_mixing=selector,
        )
        .to(device)
        .eval()
    )
    before = state_content_hash(bundle.state_dict())
    exposure = config.conditional_generation.fixed_exposure_scale
    mixing, _ = _fit_transport_mixing(
        bundle,
        primary,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        _feature_targets(primary, train_images, train_labels, device),
        exposure,
    )
    candidate = bundle.generator.normalized_modes(regenerated["basis"].to(device))
    operational = bundle.generator.normalized_modes(
        torch.einsum(
            "ymk,ymhw->ykhw", mixing.to(device), regenerated["basis"].to(device)
        )
    )
    report = {
        "contract": CURVATURE_CONTRACT,
        "candidate": {},
        "operational": {},
        "real_pca": {},
        "nearest_chord": {},
    }
    for label in range(10):
        center = source["carrier"][label].to(device)
        report["candidate"][str(label)] = _measure(
            bundle.system, primary, smooth, center, candidate[label], exposure
        )
        report["operational"][str(label)] = _measure(
            bundle.system, primary, smooth, center, operational[label], exposure
        )
        real_images = train_images[train_labels == label]
        _, _, vectors = _pca(real_images.sqrt().flatten(1), components=32)
        directions = _lowpass_modes(
            vectors.T.reshape(32, 28, 28),
            shape=config.system.slm_shape,
            pixel_pitch_m=config.system.slm_pixel_pitch_m,
            max_cycles_per_mm=20.0,
        ).to(device)
        real_center = real_images.mean(0).squeeze(0).to(device).clamp(0.01, 0.99)
        real_center = (
            F.interpolate(
                real_center[None, None],
                size=config.system.slm_shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .logit()
        )
        report["real_pca"][str(label)] = _measure(
            bundle.system, primary, smooth, real_center, directions, exposure
        )
        class_validation = validation_images[validation_labels == label].to(device)
        low = F.avg_pool2d(class_validation, 2).flatten(1)
        distance = torch.cdist(low, low)
        distance.fill_diagonal_(float("inf"))
        first, second = divmod(int(distance.argmin()), distance.shape[1])
        pair = (
            class_validation[torch.tensor([first, second], device=device)]
            .squeeze(1)
            .clamp(0.01, 0.99)
        )
        pair = (
            F.interpolate(
                pair.unsqueeze(1),
                size=config.system.slm_shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .logit()
        )
        chord = F.normalize((pair[1] - pair[0]).unsqueeze(0), dim=(1, 2))
        report["nearest_chord"][str(label)] = _measure(
            bundle.system, primary, smooth, pair.mean(0), chord, exposure
        )
    after, memory, provenance = (
        state_content_hash(bundle.state_dict()),
        _memory(device),
        _relay_provenance(bundle, device),
    )
    run = _new_run(config)
    report_path = run.root / "conditional_evaluator_curvature_baseline_report.json"
    atomic_write_json(report_path, report)
    gates = (
        GateResult(
            "qualified_bundle",
            True,
            {
                "qualification": file_sha256(args.evaluator_qualification_evidence),
                "bundle": file_sha256(args.evaluator_bundle),
            },
        ),
        GateResult(
            "state_memory",
            before == after and bool(memory["passed"]),
            {"state_unchanged": before == after, **memory},
        ),
        GateResult(
            "relay_provenance",
            provenance["near_conjugate_count"] == 4 and provenance["identity_ols"],
            provenance,
        ),
        GateResult(
            "no_optimizer_checkpoint",
            True,
            {"optimizer_created": False, "checkpoint_created": False},
        ),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_evaluator_curvature_baseline",
        config.run.evidence_tier,
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "qualification": file_sha256(args.evaluator_qualification_evidence),
            "bundle": file_sha256(args.evaluator_bundle),
            "smooth": file_sha256(args.smooth_evaluator_checkpoint),
            "v7_capacity": file_sha256(args.v7_capacity_evidence),
            "v7_basis": file_sha256(args.v7_basis_artifact),
            "report": file_sha256(report_path),
        },
        gates,
        ("conditional-evaluation-qualified-transport-design",),
        "curvature_diagnostic_only",
    )
    _write_evidence(
        run, "conditional_evaluator_curvature_baseline_evidence.json", record
    )
    print(run.root)
    return 0


def _design_inputs(args, config):
    """Rebuild the immutable source/basis/bundle shared by curvature and design."""
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    device = _device(config)
    primary, payload = load_frozen_evaluator(
        args.primary_evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("primary evaluator split mismatch")
    qualification, _, smooth = _load_bundle(args, config, split.manifest_hash)
    curvature = _record(
        args.curvature_evidence,
        "conditional_evaluator_curvature_baseline",
        "diagnostic",
    )
    if (
        "conditional-evaluation-qualified-transport-design"
        not in curvature.permitted_next_actions
    ):
        raise ValueError("curvature evidence does not authorize design")
    source = _source(args.electronic_initialization)
    v7 = _record(
        args.v7_capacity_evidence, "conditional_nonlinear_expanded_manifold_capacity"
    )
    artifact = torch.load(
        args.v7_basis_artifact, map_location="cpu", weights_only=False
    )
    if not isinstance(artifact, dict) or v7.input_hashes.get(
        "basis_artifact"
    ) != file_sha256(args.v7_basis_artifact):
        raise ValueError("v7 basis evidence/artifact mismatch")
    regenerated, _ = _build_artifact(train_images, train_labels, source, config)
    if tensor_sha256(regenerated["basis"]) != tensor_sha256(artifact["basis"]):
        raise ValueError("v7 basis regeneration mismatch")
    selector = torch.zeros(10, 32, 8)
    selector[:, torch.arange(8), torch.arange(8)] = 1
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=source["carrier"],
            class_modes=regenerated["basis"],
            manifold_scales=source["scales"],
            transport_mixing=selector,
        )
        .to(device)
        .eval()
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.electronic_initialization),
    )
    return (
        train_images,
        train_labels,
        split,
        device,
        primary,
        smooth,
        qualification,
        curvature,
        source,
        regenerated,
        bundle,
        expected,
    )


def _dual_scale_calibration(
    bundle, primary, smooth, carrier, candidates, mixing, exposure
):
    modes = bundle.generator.normalized_modes(
        torch.einsum("ymk,ymhw->ykhw", mixing, candidates)
    )
    scales, report = torch.empty(10, 8), {}
    for label in range(10):
        rows = []
        for axis in range(8):
            low, high, trace = 0.05, 0.50, []
            mode = modes[label, axis : axis + 1]
            def evaluate(value):
                outcomes = []
                for sign in (-1.0, 1.0):
                    command = torch.sigmoid(carrier[label : label + 1] + sign * value * mode).unsqueeze(1)
                    with torch.no_grad():
                        raw = bundle.system.forward(bundle.system.command(command)).raw_camera.tensor
                        zero = bundle.system.forward(bundle.system.command(torch.sigmoid(carrier[label : label + 1]).unsqueeze(1))).raw_camera.tensor
                        view, zero_view = fixed_exposure_view(raw, exposure), fixed_exposure_view(zero, exposure)
                        p_feature, p_logits = primary(view)
                        s_feature, s_logits = smooth(view)
                        zp, _ = primary(zero_view)
                        zs, _ = smooth(zero_view)
                    outcomes.append({"sign": sign, "primary": bool(p_logits.argmax(1).item() == label), "smooth": bool(s_logits.argmax(1).item() == label), "saturation": float((raw / exposure > 1).float().mean()), "raw_response": float((raw - zero).norm()), "primary_response": float((p_feature - zp).norm()), "smooth_response": float((s_feature - zs).norm()), "low_frequency_response": float((F.avg_pool2d(view, 2) - F.avg_pool2d(zero_view, 2)).norm())})
                return outcomes
            def acceptable(outcomes):
                return all(item["primary"] and item["saturation"] <= .08 and all(item[key] > 1e-8 for key in ("raw_response", "primary_response", "smooth_response", "low_frequency_response")) for item in outcomes)
            floor_outcomes = evaluate(low)
            floor_passed = acceptable(floor_outcomes)
            trace.append({"scale": low, "passed": floor_passed, "floor_probe": True, "outcomes": floor_outcomes})
            if not floor_passed:
                scales[label, axis] = low
                rows.append({"floor_passed": False, "final_passed": False, "trace": trace})
                continue
            for _ in range(20):
                value = (low + high) / 2
                outcomes = evaluate(value)
                passed = acceptable(outcomes)
                trace.append({"scale": value, "passed": passed, "outcomes": outcomes})
                if passed:
                    low = value
                else:
                    high = value
            final_outcomes = evaluate(low)
            final_passed = acceptable(final_outcomes)
            trace.append({"scale": low, "passed": final_passed, "final_probe": True, "outcomes": final_outcomes})
            scales[label, axis] = low
            rows.append({"floor_passed": True, "final_passed": final_passed, "trace": trace})
        report[str(label)] = rows
    return scales, report


def conditional_evaluation_qualified_transport_design(args, config) -> int:
    _require(config)
    torch.cuda.reset_peak_memory_stats(_device(config))
    (
        train_images,
        train_labels,
        split,
        device,
        primary,
        smooth,
        qualification,
        curvature,
        source,
        regenerated,
        bundle,
        expected,
    ) = _design_inputs(args, config)
    before = state_content_hash(bundle.state_dict())
    exposure = config.conditional_generation.fixed_exposure_scale
    mixing, fit = _fit_transport_mixing(
        bundle,
        primary,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        _feature_targets(primary, train_images, train_labels, device),
        exposure,
    )
    scales, calibration = _dual_scale_calibration(
        bundle,
        primary,
        smooth,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        mixing,
        exposure,
    )
    provenance, memory = _relay_provenance(bundle, device), _memory(device)
    labels = torch.arange(10, device=device)
    zero = bool(
        torch.equal(
            bundle.generator(torch.zeros(10, 8, device=device), labels),
            bundle.generator.class_carrier_command(labels),
        )
    )
    after = state_content_hash(bundle.state_dict())
    gates = (
        GateResult(
            "curvature_input", True, {"evidence": file_sha256(args.curvature_evidence)}
        ),
        GateResult(
            "response_rank_8",
            all(
                row["rank"]["rank"] == 8 and row["rank"]["sigma8_over_sigma1"] >= 0.05
                for row in fit.values()
            ),
            fit,
        ),
        GateResult(
            "positive_negative_scale",
            all(row.get("floor_passed") and row.get("final_passed") for rows in calibration.values() for row in rows),
            {"steps": 20, "calibration": calibration},
        ),
        GateResult("zero_carrier", zero, {}),
        GateResult(
            "relay_provenance",
            provenance["near_conjugate_count"] == 4
            and provenance["identity_ols"]
            and provenance["operations"].count("finite_crop_hard_aperture") == 4,
            provenance,
        ),
        GateResult(
            "execution_integrity",
            before == after and bool(memory["passed"]),
            {"state_unchanged": before == after, **memory},
        ),
        GateResult(
            "no_optimizer_checkpoint",
            True,
            {"optimizer_created": False, "checkpoint_created": False},
        ),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    report = run.root / "conditional_evaluation_qualified_transport_design_report.json"
    atomic_write_json(
        report,
        {
            "contract": DESIGN_CONTRACT,
            "fit": fit,
            "calibration": calibration,
            "curvature_report": curvature.input_hashes["report"],
            "artifact_published": passed,
        },
    )
    inputs = {
        "curvature": file_sha256(args.curvature_evidence),
        "qualification": file_sha256(args.evaluator_qualification_evidence),
        "bundle": file_sha256(args.evaluator_bundle),
        "electronic_initialization": file_sha256(args.electronic_initialization),
        "v7_capacity": file_sha256(args.v7_capacity_evidence),
        "v7_basis": file_sha256(args.v7_basis_artifact),
        "report": file_sha256(report),
    }
    if passed:
        artifact = {
            "schema_version": 1,
            "contract": DESIGN_CONTRACT,
            "architecture": EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE,
            "carrier": source["carrier"].float(),
            "candidate_modes": regenerated["basis"].float(),
            "transport_mixing": mixing.cpu(),
            "manifold_scales": scales.cpu(),
            "carrier_hash": tensor_sha256(source["carrier"].float()),
            "candidate_basis_hash": tensor_sha256(regenerated["basis"]),
            "mixing_hash": tensor_sha256(mixing.cpu()),
            "scale_hash": tensor_sha256(scales.cpu()),
            "curvature_evidence_hash": file_sha256(args.curvature_evidence),
            "evaluator_bundle_hash": file_sha256(args.evaluator_bundle),
            "electronic_initialization_hash": file_sha256(args.electronic_initialization),
            "v7_capacity_evidence_hash": file_sha256(args.v7_capacity_evidence),
            "v7_basis_artifact_hash": file_sha256(args.v7_basis_artifact),
            "qualification_evidence_hash": file_sha256(args.evaluator_qualification_evidence),
        }
        artifact_path = (
            run.root / "conditional_evaluation_qualified_transport_design.pt"
        )
        from ...evidence import atomic_torch_save

        atomic_torch_save(artifact_path, artifact)
        inputs["design_artifact"] = file_sha256(artifact_path)
    record = EvidenceRecordV2(
        2,
        "conditional_evaluation_qualified_transport_design",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        inputs,
        gates,
        ("conditional-evaluation-qualified-transport-admission",) if passed else (),
        None if passed else "evaluation_qualified_transport_design_failed",
    )
    _write_evidence(
        run, "conditional_evaluation_qualified_transport_design_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def _artifact(args, action: str):
    evidence = load_record(args.design_evidence, "conditional_evaluation_qualified_transport_design", action=action)
    artifact = torch.load(args.design_artifact, map_location="cpu", weights_only=False)
    required = {"contract", "architecture", "carrier", "candidate_modes", "transport_mixing", "manifold_scales", "carrier_hash", "candidate_basis_hash", "mixing_hash", "scale_hash", "curvature_evidence_hash", "evaluator_bundle_hash", "electronic_initialization_hash", "v7_capacity_evidence_hash", "v7_basis_artifact_hash", "qualification_evidence_hash"}
    if not isinstance(artifact, dict) or not required.issubset(artifact) or artifact["contract"] != DESIGN_CONTRACT or artifact["architecture"] != EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE:
        raise ValueError("invalid evaluation-qualified design artifact")
    hashes = {"carrier": "carrier_hash", "candidate_modes": "candidate_basis_hash", "transport_mixing": "mixing_hash", "manifold_scales": "scale_hash"}
    if any(artifact[key] != tensor_sha256(artifact[tensor]) for tensor, key in hashes.items()):
        raise ValueError("evaluation-qualified design artifact internal hash mismatch")
    if evidence.input_hashes.get("design_artifact") != file_sha256(args.design_artifact):
        raise ValueError("design evidence does not bind design artifact")
    return evidence, artifact


def _bundle_from_design(config, artifact, device):
    return ConditionalBundle(config, carrier_logits=artifact["carrier"], class_modes=artifact["candidate_modes"], transport_mixing=artifact["transport_mixing"], manifold_scales=artifact["manifold_scales"]).to(device).eval()


def _admission_gates(values, state_ok, memory, checkpoint_ok=True):
    flags = admission_gate_dict(values)
    details = {
        "camera": {name: row["camera_fraction"] for name, row in values["metrics"].items()},
        "delta": {name: row["delta_correlation"] for name, row in values["metrics"].items()},
        "saturation": {name: row["saturation"] for name, row in values["metrics"].items()},
        "crop": values["crops"],
        "zero_code": {"accuracy": values["zero_accuracy"]},
        "directions": {"command": values["command_rank"], "raw": values["raw_rank"]},
    }
    return tuple(
        GateResult(name, passed, values if name in {"semantics", "distribution", "anti_template"} else details.get(name, {}))
        for name, passed in flags.items()
    ) + (
        GateResult("execution_integrity", state_ok and bool(memory["passed"]), {"state_unchanged": state_ok, **memory}),
        GateResult("strict_checkpoint", checkpoint_ok, {}),
    )


def conditional_evaluation_qualified_transport_admission(args, config) -> int:
    _require(config)
    design, artifact = _artifact(args, "conditional-evaluation-qualified-transport-admission")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    device, primary, smooth, qualification, _ = load_evaluators(args, config, split.manifest_hash)
    expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint), split_hash=split.manifest_hash)
    design.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "exposure_scale"))
    torch.cuda.reset_peak_memory_stats(device)
    bundle = _bundle_from_design(config, artifact, device)
    before = state_content_hash(bundle.state_dict())
    values = admission_values(bundle, primary, smooth, validation_images, validation_labels, config.conditional_generation.fixed_exposure_scale, device)
    after, memory = state_content_hash(bundle.state_dict()), _memory(device)
    provisional = _admission_gates(values, before == after, memory)
    run, checkpoint_ok, checkpoint = _new_run(config), True, None
    if all(gate.passed for gate in provisional):
        manifest = checkpoint_manifest(config, split.manifest_hash, evaluator_checkpoint_hash(args.primary_evaluator_checkpoint), bundle, file_sha256(args.design_artifact))
        extra = {"design_artifact": file_sha256(args.design_artifact), "evaluator_bundle": file_sha256(args.evaluator_bundle), "qualification": file_sha256(args.evaluator_qualification_evidence), "architecture": EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE, "admission_only": True}
        temporary = run.root / "evaluation_qualified_initialization.verify.pt"
        save_conditional_checkpoint(temporary, bundle, manifest, extra=extra)
        restored = _bundle_from_design(config, artifact, device)
        try:
            extra_read = load_conditional_checkpoint(temporary, restored, manifest)
            checkpoint_ok = state_content_hash(restored.state_dict()) == before and extra_read == extra
        finally:
            if temporary.exists():
                temporary.unlink()
        if checkpoint_ok:
            checkpoint = run.root / "evaluation_qualified_initialization.pt"
            save_conditional_checkpoint(checkpoint, bundle, manifest, extra=extra)
    gates = _admission_gates(values, before == after, memory, checkpoint_ok)
    passed = all(gate.passed for gate in gates) and checkpoint is not None
    report = run.root / "conditional_evaluation_qualified_transport_admission_report.json"
    atomic_write_json(report, {"contract": ADMISSION_CONTRACT, "values": values, "state_hash_before": before, "state_hash_after": after, "memory": memory, "optimizer_created": False, "checkpoint_created": checkpoint is not None})
    inputs = {"design_evidence": file_sha256(args.design_evidence), "design_artifact": file_sha256(args.design_artifact), "qualification": file_sha256(args.evaluator_qualification_evidence), "bundle": file_sha256(args.evaluator_bundle), "report": file_sha256(report)}
    if checkpoint is not None:
        inputs["initialization_checkpoint"] = file_sha256(checkpoint)
    record = EvidenceRecordV2(2, "conditional_evaluation_qualified_transport_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, inputs, gates, ("conditional-evaluation-qualified-transport-tangent",) if passed else (), None if passed else "evaluation_qualified_transport_not_admitted")
    _write_evidence(run, "conditional_evaluation_qualified_transport_admission_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _load_initialization(args, config, action: str):
    admission = load_record(args.admission_evidence, "conditional_evaluation_qualified_transport_admission", action=action)
    if admission.input_hashes.get("initialization_checkpoint") != file_sha256(args.checkpoint):
        raise ValueError("admission evidence does not bind supplied initialization")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    device, primary, smooth, _, _ = load_evaluators(args, config, split.manifest_hash)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict) or payload["manifest"].get("architecture_kind") != EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE:
        raise ValueError("checkpoint architecture is incompatible before optimizer construction")
    state = payload["state_dict"]
    required = ("generator.class_carrier_logits", "generator.candidate_modes", "generator.transport_mixing", "generator.manifold_scales")
    if any(key not in state for key in required):
        raise ValueError("initialization checkpoint lacks evaluation-qualified generator state")
    bundle = ConditionalBundle(config, carrier_logits=state[required[0]], class_modes=state[required[1]], transport_mixing=state[required[2]], manifold_scales=state[required[3]]).to(device).eval()
    manifest = checkpoint_manifest(config, split.manifest_hash, evaluator_checkpoint_hash(args.primary_evaluator_checkpoint), bundle, payload["extra"].get("design_artifact", ""))
    extra = load_conditional_checkpoint(args.checkpoint, bundle, manifest)
    if not isinstance(extra, dict) or extra.get("architecture") != EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE:
        raise ValueError("initialization checkpoint extra contract mismatch")
    return admission, bundle, device, primary, smooth, train_images, train_labels, validation_images, validation_labels, split


def _tangent_objective(bundle, primary, codes, labels, reference, exposure):
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=1, activation_checkpoint=True)
    feature, logits = primary(fixed_exposure_view(raw, exposure))
    target, _ = primary(reference)
    return conditional_sinkhorn_loss(feature, fixed_exposure_view(raw, exposure), target, reference, labels, epsilon=.05, iterations=20) + F.cross_entropy(logits, labels) + relative_diversity_loss(feature, target, labels) + 2 * camera_code_fraction_loss(raw, labels, target=.25)


def conditional_evaluation_qualified_transport_tangent(args, config) -> int:
    _require(config)
    admission, bundle, device, primary, smooth, train_images, train_labels, validation_images, validation_labels, split = _load_initialization(args, config, "conditional-evaluation-qualified-transport-tangent")
    torch.cuda.reset_peak_memory_stats(device)
    before = state_content_hash(bundle.state_dict())
    exposure = config.conditional_generation.fixed_exposure_scale
    reference = __import__("ldm_project.v2.application.diagnostic.expanded_manifold", fromlist=["_validation_reference"])._validation_reference(train_images, train_labels, device)
    panels = __import__("ldm_project.v2.application.diagnostic.expanded_manifold", fromlist=["_panels"])._panels(device)
    baseline = admission_values(bundle, primary, smooth, validation_images, validation_labels, exposure, device)
    baseline_snapshots, _ = __import__("ldm_project.v2.application.diagnostic.evaluation_qualified_runtime", fromlist=["snapshots"]).snapshots(bundle, primary, smooth, __import__("ldm_project.v2.application.diagnostic.expanded_manifold", fromlist=["_validation_reference"])._validation_reference(validation_images, validation_labels, device), exposure, device)
    params = (bundle.generator.transport_mixing, bundle.system.encoder.phase, bundle.system.decoder.phase)
    objective = sum(_tangent_objective(bundle, primary, codes, labels, reference, exposure) for codes, labels in panels.values())
    gradients = torch.autograd.grad(objective, params)
    variants = {"transport_mixing": (gradients[0], torch.zeros_like(gradients[1]), torch.zeros_like(gradients[2])), "encoder_phase": (torch.zeros_like(gradients[0]), gradients[1], torch.zeros_like(gradients[2])), "decoder_phase": (torch.zeros_like(gradients[0]), torch.zeros_like(gradients[1]), gradients[2]), "joint": gradients}
    results = {}
    for name, direction in variants.items():
        valid = all(torch.isfinite(value).all() for value in direction) and any(value.norm() > 1e-12 for value in direction)
        if not valid:
            results[name] = {"valid": False}
            continue
        unit = tuple(-value / value.norm().clamp_min(1e-12) if value.norm() > 1e-12 else value for value in direction)
        probe = copy.deepcopy(bundle)
        with torch.no_grad():
            probe.generator.transport_mixing.add_(unit[0] * .001)
            probe.system.encoder.phase.add_(unit[1] * .001)
            probe.system.decoder.phase.add_(unit[2] * .001)
        trial = admission_values(probe, primary, smooth, validation_images, validation_labels, exposure, device)
        trial_snapshots, _ = __import__("ldm_project.v2.application.diagnostic.evaluation_qualified_runtime", fromlist=["snapshots"]).snapshots(probe, primary, smooth, __import__("ldm_project.v2.application.diagnostic.expanded_manifold", fromlist=["_validation_reference"])._validation_reference(validation_images, validation_labels, device), exposure, device)
        base_score = sum(min(row["composite"]) for row in baseline["metrics"].values()) / 4
        trial_score = sum(min(row["composite"]) for row in trial["metrics"].values()) / 4
        declines = [min(trial["metrics"][panel]["composite"]) - min(baseline["metrics"][panel]["composite"]) for panel in baseline["metrics"]]
        raw_rms = max(float((trial_snapshots[p].raw - baseline_snapshots[p].raw).square().mean().sqrt() / baseline_snapshots[p].raw.square().mean().sqrt().clamp_min(1e-12)) for p in baseline_snapshots)
        phase_rms = float(torch.sqrt(probe.system.encoder.phase.square().mean() + probe.system.decoder.phase.square().mean()).item())
        semantic_delta = min(trial["metrics"][p]["accuracy"] for p in trial["metrics"]) - min(baseline["metrics"][p]["accuracy"] for p in baseline["metrics"])
        logit_rms = max(_logit_delta(bundle, probe, codes, labels) for codes, labels in panels.values())
        results[name] = {"valid": True, "aggregate_composite_delta": trial_score - base_score, "panel_deltas": declines, "raw_rms": raw_rms, "phase_rms": phase_rms, "logit_rms": logit_rms, "semantic_delta": semantic_delta, "safe": trial_score >= base_score + .005 and min(declines) >= -.005 and semantic_delta >= -.015 and raw_rms <= .01 and phase_rms <= .005 and logit_rms <= .05 and all(value <= .08 for value in (row["saturation"] for row in trial["metrics"].values()))}
    after, memory = state_content_hash(bundle.state_dict()), _memory(device)
    passed = before == after and bool(memory["passed"]) and any(row.get("safe", False) for row in results.values())
    run = _new_run(config)
    report = run.root / "conditional_evaluation_qualified_transport_tangent_report.json"
    atomic_write_json(report, {"contract": TANGENT_CONTRACT, "results": results, "state_hash_before": before, "state_hash_after": after, "optimizer_created": False, "checkpoint_created": False, "memory": memory})
    gates = (GateResult("state_unchanged", before == after, {}), GateResult("safe_aggregate_tangent", any(row.get("safe", False) for row in results.values()), results), GateResult("cuda", bool(memory["passed"]), memory))
    expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint))
    record = EvidenceRecordV2(2, "conditional_evaluation_qualified_transport_tangent", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"admission_evidence": file_sha256(args.admission_evidence), "checkpoint": file_sha256(args.checkpoint), "report": file_sha256(report)}, gates, ("evaluation-qualified-transport-smoke",) if passed else (), None if passed else "evaluation_qualified_distribution_tangent_absent")
    _write_evidence(run, "conditional_evaluation_qualified_transport_tangent_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _smoke_loss(bundle, primary, codes, labels, real, exposure):
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=1, activation_checkpoint=True)
    view = fixed_exposure_view(raw, exposure)
    features, logits = primary(view)
    real_features, _ = primary(real)
    sinkhorn = conditional_sinkhorn_loss(features, view, real_features, real, labels, epsilon=.05, iterations=20)
    diversity = relative_diversity_loss(features, real_features, labels)
    camera = camera_code_fraction_loss(raw, labels, target=.25)
    zero = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros_like(codes), labels, microbatch_size=1, activation_checkpoint=True)
    delta = 1 - F.cosine_similarity((raw - zero).flatten(1), (.80 * bundle.generator(codes, labels).square() - .80 * bundle.generator(torch.zeros_like(codes), labels).square()).flatten(1), dim=1).mean()
    saturation = torch.relu(raw / exposure - 1).square().mean()
    total = sinkhorn + F.cross_entropy(logits, labels) + diversity + 2 * camera + .1 * saturation + 1e-5 * phase_total_variation(bundle.system) + delta
    return total, {"total": float(total.detach()), "sinkhorn": float(sinkhorn.detach()), "relative_diversity": float(diversity.detach()), "delta": float(delta.detach())}


def evaluation_qualified_transport_smoke(args, config) -> int:
    _require(config)
    tangent = load_record(args.tangent_evidence, "conditional_evaluation_qualified_transport_tangent", action="evaluation-qualified-transport-smoke")
    if tangent.input_hashes.get("checkpoint") != file_sha256(args.checkpoint):
        raise ValueError("tangent evidence does not bind admission initialization")
    # Load once for immutable input validation; each seed reconstructs its own state.
    _, _, device, _, _, train_images, train_labels, _, _, split = _load_initialization(args, config, "conditional-evaluation-qualified-transport-tangent")
    expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint))
    torch.cuda.reset_peak_memory_stats(device)
    run, all_results = _new_run(config), {}
    for seed in (1234, 4321):
        torch.manual_seed(seed)
        _, bundle, _, primary, smooth, _, _, validation_images, validation_labels, _ = _load_initialization(args, config, "conditional-evaluation-qualified-transport-tangent")
        initial = {key: value.detach().cpu().clone() for key, value in bundle.state_dict().items()}
        optimizer = torch.optim.Adam([{"params": [bundle.generator.transport_mixing], "lr": 2e-4}, {"params": [bundle.system.encoder.phase, bundle.system.decoder.phase], "lr": 5e-5}])
        history, probes = [], {}
        batches = __import__("ldm_project.v2.student.evaluator", fromlist=["balanced_batches"]).balanced_batches(train_labels, batch_size=50, seed=seed)
        for step in range(401):
            if step in {0, 50, 100, 200, 400}:
                values = admission_values(bundle, primary, smooth, validation_images, validation_labels, config.conditional_generation.fixed_exposure_scale, device)
                flags = admission_gate_dict(values, final=step == 400)
                probes[str(step)] = {"flags": flags, "passed": all(flags.values()) if step == 400 else flags["semantics"] and flags["distribution"] and flags["saturation"] and flags["crop"]}
            if step == 400:
                break
            codes, schedule_labels = _training_codes(step, device)
            labels = schedule_labels
            indices = next(batches)
            optimizer.zero_grad(set_to_none=True)
            loss, row = _smoke_loss(bundle, primary, codes, labels, train_images.index_select(0, indices).to(device), config.conditional_generation.fixed_exposure_scale)
            loss.backward()
            optimizer.step()
            history.append(row | {"step": step + 1})
        updates = {"mixing": float((bundle.generator.transport_mixing.detach().cpu() - initial["generator.transport_mixing"]).norm()), "encoder": float((bundle.system.encoder.phase.detach().cpu() - initial["system.encoder.phase"]).norm()), "decoder": float((bundle.system.decoder.phase.detach().cpu() - initial["system.decoder.phase"]).norm())}
        trend = {name: {"first50": sum(row[name] for row in history[:50]) / 50, "last50": sum(row[name] for row in history[-50:]) / 50} for name in ("sinkhorn", "relative_diversity", "delta")}
        frozen = all(torch.equal(value.cpu(), initial[key]) for key, value in bundle.state_dict().items() if key not in {"generator.transport_mixing", "system.encoder.phase", "system.decoder.phase"})
        all_results[str(seed)] = {"probes": probes, "updates": updates, "frozen": frozen, "trend": trend, "passed": all(item["passed"] for item in probes.values()) and all(value > 0 for value in updates.values()) and frozen and all(value["last50"] <= value["first50"] for value in trend.values())}
    memory = _memory(device)
    passed = all(row["passed"] for row in all_results.values()) and bool(memory["passed"])
    history_path = run.root / "evaluation_qualified_transport_smoke_history.json"
    atomic_write_json(history_path, {"contract": SMOKE_CONTRACT, "seeds": all_results, "memory": memory})
    classification = None if passed else ("optimizer_instability_not_formally_qualified" if sum(row["passed"] for row in all_results.values()) == 1 else "evaluation_qualified_transport_smoke_failed")
    gates = (GateResult("two_seed_independent", all(row["passed"] for row in all_results.values()), all_results), GateResult("cuda", bool(memory["passed"]), memory))
    record = EvidenceRecordV2(2, "evaluation_qualified_transport_smoke", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"tangent_evidence": file_sha256(args.tangent_evidence), "checkpoint": file_sha256(args.checkpoint), "history": file_sha256(history_path)}, gates, ("train-evaluation-qualified-transport",) if passed else (), classification)
    _write_evidence(run, "evaluation_qualified_transport_smoke_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def train_evaluation_qualified_transport(args, config) -> int:
    _require(config)
    raise RuntimeError("evaluation-qualified formal training is intentionally fail-closed in this iteration")
