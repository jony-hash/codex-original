"""Content-bound v7.1 transport-aligned electronic-manifold stages.

The sequence is deliberately linear: design, no-update admission, no-update
tangent, then one bounded smoke.  A scientific gate failure writes evidence
and returns non-zero; malformed provenance fails before creating a run.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import (
    EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json,
    canonical_sha256, file_sha256, tensor_sha256,
)
from ...student.checkpoints import (
    ConditionalCheckpointManifestV2, load_conditional_checkpoint,
    save_conditional_checkpoint, state_content_hash, state_signature,
)
from ...student.conditional import (
    camera_code_fraction_loss, conditional_sinkhorn_loss, fixed_exposure_view,
    phase_total_variation, raw_from_generator_batched, relative_diversity_loss,
)
from ...student.evaluator import (
    MNISTSplitManifest, balanced_batches, evaluator_checkpoint_hash,
    load_frozen_evaluator, load_local_mnist,
)
from .expanded_manifold import (
    _build_artifact, _crop_minimum, _panels, _rank_summary, _source,
    _validation_reference,
)
from .pca_transport import _metrics, _training_codes
from .support import (
    ConditionalBundle, TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE, _context,
    _new_run, _split, _write_evidence,
)


DESIGN_CONTRACT = canonical_sha256({"kind": "conditional_transport_aligned_manifold_design_v7_1", "epsilon_logit": .01, "ridge": "1e-3*trace(JTJ)/32", "components": 32, "directions": 8, "scale_bisection_steps": 20, "positive_and_negative": True, "optimizer": False, "checkpoint": False})
ADMISSION_CONTRACT = canonical_sha256({"kind": "conditional_transport_aligned_manifold_admission_v7_1", "panels": ["axis", "uniform_1234", "uniform_5678", "holdout_9012"], "no_update": True, "raw_unique": 170, "delta_floor": .80})
TANGENT_CONTRACT = canonical_sha256({"kind": "conditional_transport_aligned_manifold_tangent_v7_1", "groups": ["transport_mixing", "encoder_phase", "decoder_phase", "joint"], "logit_rms": .05, "phase_rms": .005, "raw_rms": .01, "optimizer": False})
SMOKE_CONTRACT = canonical_sha256({"kind": "transport_aligned_manifold_smoke_v7_1", "steps": 400, "probes": [0, 50, 100, 200, 400], "seed": 1234, "mixing_lr": 2e-4, "phase_lr": 5e-5, "one_shot": True})
TRAIN_CONTRACT = canonical_sha256({"kind": "train_transport_aligned_manifold_v7_1", "epochs": 5, "steps": 5500, "one_shot": True})
GENERATOR_CONTRACT = canonical_sha256({"architecture": TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE, "formula": "sigmoid(By+sum(s*tanh(z)*normalize(WP)))", "frozen": ["By", "P", "s"]})
METRIC_CONTRACT = canonical_sha256({"delta": "cosine(raw(z)-raw(0),.80*A2(z)-.80*A2(0))", "exposure": "fixed", "loss": ["sinkhorn", "classification", "relative_diversity", "camera", "saturation", "phase_tv", "delta"]})


def _require(config) -> None:
    if config.architecture != TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("transport-aligned manifold command requires conditional_transport_aligned_manifold_v7_1")


def _device(config) -> torch.device:
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("transport-aligned scientific gates require configured CUDA")
    return device


def _memory(device: torch.device) -> dict[str, float | bool]:
    peak = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return {"reserved": float(peak), "total": float(total), "passed": bool(peak <= .92 * total)}


def _load_evidence(path: str, *, kind: str, action: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed")
    if action is not None and action not in record.permitted_next_actions:
        raise ValueError(f"{kind} evidence does not authorize {action}")
    return record


def _manifest(config, split_hash: str, evaluator_hash: str, bundle, design_hash: str) -> ConditionalCheckpointManifestV2:
    state = bundle.state_dict()
    return ConditionalCheckpointManifestV2(
        2, TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE,
        canonical_sha256(config.system.model_dump(mode="json")),
        canonical_sha256(config.model_dump(mode="json")), split_hash, evaluator_hash,
        config.conditional_generation.fixed_exposure_scale, state_signature(state),
        state_content_hash(state), GENERATOR_CONTRACT, METRIC_CONTRACT,
        TRAIN_CONTRACT, 1234, state_content_hash(state),
        config_contract_hash=canonical_sha256(config.model_dump(mode="json")),
        pca_basis_hash=tensor_sha256(bundle.generator.candidate_modes),
        lowpass_contract_hash=canonical_sha256({"cycles_per_mm": 20.0}),
        scale_contract_hash=tensor_sha256(bundle.generator.manifold_scales),
        route_contract_hash=canonical_sha256({"code": "tanh", "tau": .80}),
        semantic_source_checkpoint_hash=design_hash,
        safe_scale_evidence_hash=design_hash,
        first_400_schedule_hash=SMOKE_CONTRACT,
    )


def _artifact_hashes(artifact: dict) -> bool:
    return (
        artifact.get("carrier_hash") == tensor_sha256(artifact["carrier"])
        and artifact.get("candidate_basis_hash") == tensor_sha256(artifact["candidate_modes"])
        and artifact.get("mixing_hash") == tensor_sha256(artifact["transport_mixing"])
        and artifact.get("scale_hash") == tensor_sha256(artifact["manifold_scales"])
    )


def _bundle_from_artifact(config, artifact: dict, device: torch.device) -> ConditionalBundle:
    return ConditionalBundle(
        config, carrier_logits=artifact["carrier"], class_modes=artifact["candidate_modes"],
        transport_mixing=artifact["transport_mixing"],
        manifold_scales=artifact["manifold_scales"],
    ).to(device).eval()


def _load_checkpoint_bundle(path: str, config, *, device: torch.device, evaluator_hash: str, split_hash: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("invalid transport-aligned checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    state = payload["state_dict"]
    required = ("generator.class_carrier_logits", "generator.candidate_modes", "generator.transport_mixing", "generator.manifold_scales")
    if manifest.architecture_kind != TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE or any(key not in state for key in required):
        raise ValueError("checkpoint is not a transport-aligned manifold state")
    if manifest.evaluator_checkpoint_hash != evaluator_hash or manifest.dataset_split_manifest_hash != split_hash:
        raise ValueError("checkpoint evaluator/split mismatch")
    bundle = ConditionalBundle(config, carrier_logits=state[required[0]], class_modes=state[required[1]], transport_mixing=state[required[2]], manifold_scales=state[required[3]]).to(device).eval()
    extra = load_conditional_checkpoint(path, bundle, manifest)
    if not isinstance(extra, dict) or not isinstance(extra.get("design_artifact"), str):
        raise ValueError("transport checkpoint lacks content-bound design artifact")
    return bundle, manifest, extra


def _delta_correlation(bundle, codes, labels, raw) -> float:
    with torch.no_grad():
        zero = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros_like(codes), labels, microbatch_size=1, activation_checkpoint=False)
        command = .80 * bundle.generator(codes, labels).square()
        zero_command = .80 * bundle.generator(torch.zeros_like(codes), labels).square()
        return float(F.cosine_similarity((raw - zero).flatten(1), (command - zero_command).flatten(1), dim=1).mean().item())


def _direction_ranks(bundle, device: torch.device) -> tuple[dict, dict]:
    command, raw = {}, {}
    with torch.no_grad():
        for label in range(10):
            command_rows, raw_rows = [], []
            labels = torch.tensor([label], device=device)
            for axis in range(8):
                plus, minus = torch.zeros(1, 8, device=device), torch.zeros(1, 8, device=device)
                plus[0, axis], minus[0, axis] = 1., -1.
                command_rows.append((bundle.generator(plus, labels) - bundle.generator(minus, labels)).flatten())
                plus_raw = raw_from_generator_batched(bundle.system, bundle.generator, plus, labels, microbatch_size=1, activation_checkpoint=False)
                minus_raw = raw_from_generator_batched(bundle.system, bundle.generator, minus, labels, microbatch_size=1, activation_checkpoint=False)
                raw_rows.append((plus_raw - minus_raw).flatten())
            command[str(label)] = _rank_summary(torch.stack(command_rows))
            raw[str(label)] = _rank_summary(torch.stack(raw_rows))
    return command, raw


def _relay_provenance(bundle, device: torch.device) -> dict[str, object]:
    with torch.no_grad():
        trace = bundle.system.forward_from_generator(bundle.generator, torch.zeros(1, 8, device=device), torch.zeros(1, dtype=torch.long, device=device), return_optical_trace=True).optical_trace or ()
    operations = [field.provenance.operation for field in trace]
    attenuation = [field.provenance.parameters for field in trace if field.provenance.operation == "global_illumination_attenuation"]
    return {"operations": operations, "near_conjugate_count": operations.count("near_conjugate_relay_lct"), "identity_ols": "identity_ols" in operations, "attenuation": attenuation}


def _mode_response(bundle, evaluator, carrier, mode, label: int, epsilon: float, exposure: float):
    plus_command = torch.sigmoid(carrier[label:label + 1] + epsilon * mode).unsqueeze(1)
    minus_command = torch.sigmoid(carrier[label:label + 1] - epsilon * mode).unsqueeze(1)
    with torch.no_grad():
        plus = bundle.system.forward(bundle.system.command(plus_command)).raw_camera.tensor
        minus = bundle.system.forward(bundle.system.command(minus_command)).raw_camera.tensor
        zero = bundle.system.forward(bundle.system.command(torch.sigmoid(carrier[label:label + 1]).unsqueeze(1))).raw_camera.tensor
        f_plus, _ = evaluator(fixed_exposure_view(plus, exposure))
        f_minus, _ = evaluator(fixed_exposure_view(minus, exposure))
        f_zero, _ = evaluator(fixed_exposure_view(zero, exposure))
    return (f_plus - f_minus).flatten() / (2 * epsilon), float(((f_plus + f_minus - 2 * f_zero).norm() / (f_plus - f_minus).norm().clamp_min(1e-12)).item())


def _feature_targets(evaluator, images, labels, device):
    targets = []
    with torch.no_grad():
        for label in range(10):
            values = []
            selected = images[labels == label]
            for start in range(0, selected.shape[0], 128):
                features, _ = evaluator(selected[start:start + 128].to(device))
                values.append(features.cpu())
            centered = torch.cat(values).double()
            centered -= centered.mean(0, keepdim=True)
            targets.append(torch.linalg.svd(centered, full_matrices=False)[2][:8].T.float())
    return torch.stack(targets).to(device)


def _fit_transport_mixing(bundle, evaluator, carrier, candidates, targets, exposure):
    mixing, reports = [], {}
    normalized = bundle.generator.normalized_modes(candidates)
    for label in range(10):
        columns, errors = [], []
        for mode in normalized[label]:
            derivative, error = _mode_response(bundle, evaluator, carrier, mode.unsqueeze(0), label, .01, exposure)
            columns.append(derivative)
            errors.append(error)
        jacobian = torch.stack(columns, dim=1)
        ridge = 1e-3 * torch.trace(jacobian.T @ jacobian) / 32.
        solution = torch.linalg.solve(jacobian.T @ jacobian + ridge * torch.eye(32, device=jacobian.device), jacobian.T @ targets[label])
        solution = F.normalize(solution, dim=0)
        response = jacobian @ solution
        solution *= torch.sign((response * targets[label]).sum(0)).clamp_min(0).mul(2).sub(1)
        mixing.append(solution)
        reports[str(label)] = {"rank": _rank_summary(response.T), "nonlinearity": max(errors), "ridge": float(ridge.item())}
    return torch.stack(mixing), reports


def _calibrate_scale(acceptable):
    lower, upper, trace = .05, .50, []
    for _ in range(20):
        midpoint = (lower + upper) / 2
        passed, details = acceptable(midpoint)
        trace.append({"scale": midpoint, "passed": bool(passed), **details})
        if passed:
            lower = midpoint
        else:
            upper = midpoint
    return lower, trace


def _calibrate_scales(bundle, evaluator, carrier, candidates, mixing, exposure):
    aligned = bundle.generator.normalized_modes(torch.einsum("ymk,ymhw->ykhw", mixing, candidates))
    scales, report = torch.empty(10, 8), {}
    for label in range(10):
        rows = []
        for direction in range(8):
            mode = aligned[label, direction:direction + 1]
            def acceptable(scale):
                outcomes = []
                for sign in (-1., 1.):
                    command = torch.sigmoid(carrier[label:label + 1] + sign * scale * mode).unsqueeze(1)
                    with torch.no_grad():
                        raw = bundle.system.forward(bundle.system.command(command)).raw_camera.tensor
                        zero = bundle.system.forward(bundle.system.command(torch.sigmoid(carrier[label:label + 1]).unsqueeze(1))).raw_camera.tensor
                        features, logits = evaluator(fixed_exposure_view(raw, exposure))
                        zero_features, _ = evaluator(fixed_exposure_view(zero, exposure))
                    outcomes.append({"sign": sign, "semantic": bool(logits.argmax(1).item() == label), "saturation": float((raw / exposure > 1).float().mean().item()), "raw_response": float((raw - zero).norm().item()), "feature_response": float((features - zero_features).norm().item())})
                passed = all(item["semantic"] and item["saturation"] <= .08 and item["raw_response"] > 1e-8 and item["feature_response"] > 1e-8 for item in outcomes)
                return passed, {"outcomes": outcomes}
            scale, trace = _calibrate_scale(acceptable)
            scales[label, direction] = scale
            rows.append(trace)
        report[str(label)] = rows
    return scales, report


def conditional_transport_aligned_manifold_design(args, config) -> int:
    _require(config)
    source = _source(args.electronic_initialization)
    v7 = _load_evidence(args.v7_capacity_evidence, kind="conditional_nonlinear_expanded_manifold_capacity")
    v7_artifact = torch.load(args.v7_basis_artifact, map_location="cpu", weights_only=False)
    if not isinstance(v7_artifact, dict) or v7.input_hashes.get("basis_artifact") != file_sha256(args.v7_basis_artifact) or v7.context.checkpoint_hash != file_sha256(args.electronic_initialization):
        raise ValueError("v7 capacity inputs are not content-bound to this source")
    if v7_artifact.get("source_basis_hash") != source["basis_hash"] or v7_artifact.get("source_electronic_hash") != source["source_checkpoint_hash"]:
        raise ValueError("v7 basis artifact is not derived from the supplied frozen source")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    v7_regenerated, _ = _build_artifact(train_images, train_labels, source, config)
    if tensor_sha256(v7_regenerated["basis"]) != tensor_sha256(v7_artifact["basis"]):
        raise ValueError("regenerated candidate pool does not match v7 artifact")
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, evaluator_payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**evaluator_payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("frozen evaluator split mismatch")
    selector = torch.zeros(10, 32, 8)
    selector[:, torch.arange(8), torch.arange(8)] = 1.
    bundle = ConditionalBundle(config, carrier_logits=source["carrier"], class_modes=v7_regenerated["basis"], manifold_scales=source["scales"], transport_mixing=selector).to(device).eval()
    before = state_content_hash(bundle.state_dict())
    mixing, fit = _fit_transport_mixing(bundle, evaluator, source["carrier"].to(device), v7_regenerated["basis"].to(device), _feature_targets(evaluator, train_images, train_labels, device), config.conditional_generation.fixed_exposure_scale)
    scales, calibration = _calibrate_scales(bundle, evaluator, source["carrier"].to(device), v7_regenerated["basis"].to(device), mixing, config.conditional_generation.fixed_exposure_scale)
    provenance = _relay_provenance(bundle, device)
    zero_labels = torch.arange(10, device=device)
    zero_equal = bool(torch.equal(bundle.generator(torch.zeros(10, 8, device=device), zero_labels), bundle.generator.class_carrier_command(zero_labels)))
    memory = _memory(device)
    design = {"schema_version": 1, "contract": DESIGN_CONTRACT, "carrier": source["carrier"].float(), "candidate_modes": v7_regenerated["basis"].float(), "transport_mixing": mixing.cpu(), "manifold_scales": scales.cpu(), "carrier_hash": tensor_sha256(source["carrier"].float()), "source_basis_hash": source["basis_hash"], "source_electronic_hash": source["source_checkpoint_hash"], "candidate_basis_hash": tensor_sha256(v7_regenerated["basis"]), "mixing_hash": tensor_sha256(mixing.cpu()), "scale_hash": tensor_sha256(scales.cpu()), "v7_capacity_evidence_hash": file_sha256(args.v7_capacity_evidence), "v7_basis_artifact_hash": file_sha256(args.v7_basis_artifact), "calibration": calibration}
    after = state_content_hash(bundle.state_dict())
    gates = (
        GateResult("v7_input_fact", True, {"evidence": v7.evidence_hash}),
        GateResult("response_rank_8", all(value["rank"]["rank"] == 8 and value["rank"]["sigma8_over_sigma1"] >= .05 for value in fit.values()), fit),
        GateResult("central_difference_linearity", all(value["nonlinearity"] <= .10 for value in fit.values()), fit),
        GateResult("positive_negative_scale", bool(((scales >= .05) & (scales <= .50)).all()), {"steps": 20}),
        GateResult("zero_carrier", zero_equal, {}),
        GateResult("relay_provenance", bool(provenance["identity_ols"] and provenance["near_conjugate_count"] == 4 and any(item.get("intensity_transmission") == .80 and item.get("per_sample") is False for item in provenance["attenuation"])), provenance),
        GateResult("execution_integrity", before == after and bool(memory["passed"]), {"state_unchanged": before == after, **memory}),
        GateResult("no_optimizer_checkpoint", True, {"optimizer_created": False, "checkpoint_created": False}),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    report = run.root / "conditional_transport_aligned_manifold_design_report.json"
    atomic_write_json(report, {"contract": DESIGN_CONTRACT, "fit": fit, "calibration": calibration, "provenance": provenance, "memory": memory, "state_hash_before": before, "state_hash_after": after, "artifact_published": passed})
    inputs = {"electronic_initialization": file_sha256(args.electronic_initialization), "v7_capacity_evidence": file_sha256(args.v7_capacity_evidence), "v7_basis_artifact": file_sha256(args.v7_basis_artifact), "report": file_sha256(report)}
    if passed:
        artifact_path = run.root / "conditional_transport_aligned_manifold_design.pt"
        atomic_torch_save(artifact_path, design)
        inputs["design_artifact"] = file_sha256(artifact_path)
    context = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_initialization))
    record = EvidenceRecordV2(2, "conditional_transport_aligned_manifold_design", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, inputs, gates, ("conditional-transport-aligned-manifold-admission",) if passed else (), None if passed else "transport_design_gate_failed")
    _write_evidence(run, "conditional_transport_aligned_manifold_design_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _admission_values(bundle, evaluator, reference, device, exposure):
    panels, metrics, feature_rank, unique, crops = _panels(device), {}, {}, {}, {}
    for name, (codes, labels) in panels.items():
        values = _metrics(bundle, evaluator, codes, labels, reference, exposure)
        metrics[name] = {key: value for key, value in values.items() if key not in {"raw", "command", "features", "logits"}}
        feature_rank[name] = {str(label): _rank_summary(values["features"][labels == label] - values["features"][labels == label].mean(0, keepdim=True)) for label in range(10)}
        unique[name] = len({tensor_sha256(row.detach().cpu()) for row in values["raw"]})
        crops[name] = _crop_minimum(bundle, codes, labels)
        metrics[name]["delta_correlation"] = _delta_correlation(bundle, codes, labels, values["raw"])
    command_rank, raw_rank = _direction_ranks(bundle, device)
    zero_labels = torch.arange(10, device=device)
    zero_raw = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros(10, 8, device=device), zero_labels, microbatch_size=1, activation_checkpoint=False)
    _, zero_logits = evaluator(fixed_exposure_view(zero_raw, exposure))
    return metrics, feature_rank, unique, crops, command_rank, raw_rank, float((zero_logits.argmax(1) == zero_labels).float().mean().item())


def _admission_gates(metrics, feature_rank, unique, crops, command_rank, raw_rank, zero_accuracy, state_ok, memory, *, checkpoint_ok=True):
    semantics = all(row["accuracy"] >= .95 and min(row["recall"]) >= .90 for row in metrics.values())
    fraction = metrics["axis"]["camera_fraction"] >= .25 and all(metrics[name]["camera_fraction"] >= .40 for name in ("uniform_1234", "uniform_5678", "holdout_9012"))
    ratios = min(metrics["axis"]["ratios"]) >= .50 and all(min(metrics[name]["ratios"]) >= .60 for name in ("uniform_1234", "uniform_5678", "holdout_9012"))
    directions = all(row["rank"] == 8 and row["sigma8_over_sigma1"] >= 1e-3 for rows in (command_rank, raw_rank) for row in rows.values())
    noncollapsed = all(item["rank"] >= 8 and item["sigma8_over_sigma1"] >= 1e-3 for panel in feature_rank.values() for item in panel.values()) and all(count == 170 for count in unique.values())
    return (
        GateResult("semantics", semantics, metrics), GateResult("zero_code", zero_accuracy >= .95, {"accuracy": zero_accuracy}),
        GateResult("camera_fraction", fraction, {name: row["camera_fraction"] for name, row in metrics.items()}),
        GateResult("feature_ratio", ratios, {name: min(row["ratios"]) for name, row in metrics.items()}),
        GateResult("delta_correlation", all(row["delta_correlation"] >= .80 for row in metrics.values()), {name: row["delta_correlation"] for name, row in metrics.items()}),
        GateResult("command_raw_rank", directions, {"command": command_rank, "raw": raw_rank}),
        GateResult("anti_template", noncollapsed, {"feature_rank": feature_rank, "unique": unique}),
        GateResult("saturation", all(row["saturation"] <= .08 for row in metrics.values()), {name: row["saturation"] for name, row in metrics.items()}),
        GateResult("relay_crop", all(value >= .90 for value in crops.values()), crops),
        GateResult("execution_integrity", state_ok and bool(memory["passed"]), {"state_unchanged": state_ok, **memory}),
        GateResult("strict_checkpoint", checkpoint_ok, {}),
    )


def conditional_transport_aligned_manifold_admission(args, config) -> int:
    _require(config)
    design_evidence = _load_evidence(args.design_evidence, kind="conditional_transport_aligned_manifold_design", action="conditional-transport-aligned-manifold-admission")
    artifact = torch.load(args.design_artifact, map_location="cpu", weights_only=False)
    required = {"contract", "carrier", "candidate_modes", "transport_mixing", "manifold_scales", "carrier_hash", "source_basis_hash", "source_electronic_hash", "candidate_basis_hash", "mixing_hash", "scale_hash", "v7_capacity_evidence_hash", "v7_basis_artifact_hash"}
    if not isinstance(artifact, dict) or not required.issubset(artifact) or artifact["contract"] != DESIGN_CONTRACT or not _artifact_hashes(artifact):
        raise ValueError("invalid or internally inconsistent transport design artifact")
    if design_evidence.input_hashes.get("design_artifact") != file_sha256(args.design_artifact):
        raise ValueError("design artifact hash does not match design evidence")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash)
    design_evidence.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "exposure_scale"))
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("frozen evaluator split mismatch")
    bundle = _bundle_from_artifact(config, artifact, device)
    before = state_content_hash(bundle.state_dict())
    values = _admission_values(bundle, evaluator, _validation_reference(validation_images, validation_labels, device), device, config.conditional_generation.fixed_exposure_scale)
    after = state_content_hash(bundle.state_dict())
    memory = _memory(device)
    run = _new_run(config)
    checkpoint_ok, checkpoint = True, None
    provisional = _admission_gates(*values, before == after, memory, checkpoint_ok=True)
    if all(gate.passed for gate in provisional):
        temporary = run.root / "conditional_transport_aligned_manifold_initialization.verify.pt"
        manifest = _manifest(config, split.manifest_hash, evaluator_checkpoint_hash(args.evaluator_checkpoint), bundle, file_sha256(args.design_artifact))
        save_conditional_checkpoint(temporary, bundle, manifest, extra={"design_artifact": file_sha256(args.design_artifact), "admission_only": True})
        try:
            restored, restored_manifest, extra = _load_checkpoint_bundle(str(temporary), config, device=device, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash)
            checkpoint_ok = restored_manifest == manifest and state_content_hash(restored.state_dict()) == state_content_hash(bundle.state_dict()) and extra.get("design_artifact") == file_sha256(args.design_artifact)
        finally:
            if temporary.exists():
                temporary.unlink()
        if checkpoint_ok:
            checkpoint = run.root / "conditional_transport_aligned_manifold_initialization.pt"
            save_conditional_checkpoint(checkpoint, bundle, manifest, extra={"design_artifact": file_sha256(args.design_artifact), "admission_only": True})
    gates = _admission_gates(*values, before == after, memory, checkpoint_ok=checkpoint_ok)
    passed = all(gate.passed for gate in gates) and checkpoint is not None
    report = run.root / "conditional_transport_aligned_manifold_admission_report.json"
    atomic_write_json(report, {"contract": ADMISSION_CONTRACT, "metrics": values[0], "feature_rank": values[1], "unique": values[2], "crops": values[3], "command_rank": values[4], "raw_rank": values[5], "zero_accuracy": values[6], "memory": memory, "state_hash_before": before, "state_hash_after": after, "optimizer_created": False, "checkpoint_created": checkpoint is not None})
    inputs = {"design_evidence": file_sha256(args.design_evidence), "design_artifact": file_sha256(args.design_artifact), "report": file_sha256(report)}
    if checkpoint is not None:
        inputs["initialization_checkpoint"] = file_sha256(checkpoint)
    record = EvidenceRecordV2(2, "conditional_transport_aligned_manifold_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, inputs, gates, ("conditional-transport-aligned-manifold-tangent",) if passed else (), None if passed else "transport_aligned_manifold_not_admitted")
    _write_evidence(run, "conditional_transport_aligned_manifold_admission_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _tangent_loss(bundle, evaluator, codes, labels, real, exposure):
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=1, activation_checkpoint=True)
    features, logits = evaluator(fixed_exposure_view(raw, exposure))
    real_features, _ = evaluator(real)
    sinkhorn = conditional_sinkhorn_loss(features, fixed_exposure_view(raw, exposure), real_features, real, labels, epsilon=.05, iterations=20)
    diversity = relative_diversity_loss(features, real_features, labels)
    camera = camera_code_fraction_loss(raw, labels, target=.25)
    zero = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros_like(codes), labels, microbatch_size=1, activation_checkpoint=True)
    delta = 1 - F.cosine_similarity((raw - zero).flatten(1), (.80 * bundle.generator(codes, labels).square() - .80 * bundle.generator(torch.zeros_like(codes), labels).square()).flatten(1), dim=1).mean()
    saturation = torch.relu(raw / exposure - 1).square().mean()
    total = sinkhorn + F.cross_entropy(logits, labels) + diversity + 2 * camera + .1 * saturation + 1e-5 * phase_total_variation(bundle.system) + delta
    return total


def _logit_delta(bundle, probe, codes, labels) -> float:
    with torch.no_grad():
        def logits(item):
            modes = item.generator.aligned_modes().index_select(0, labels)
            return item.generator.class_carrier_logits.index_select(0, labels) + torch.einsum("bk,bk,bkhw->bhw", torch.tanh(codes), item.generator.manifold_scales.index_select(0, labels), modes)
        return float((logits(probe) - logits(bundle)).square().mean().sqrt().item())


def conditional_transport_aligned_manifold_tangent(args, config) -> int:
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    admission = _load_evidence(args.admission_evidence, kind="conditional_transport_aligned_manifold_admission", action="conditional-transport-aligned-manifold-tangent")
    if admission.input_hashes.get("initialization_checkpoint") != file_sha256(args.checkpoint):
        raise ValueError("admission evidence does not bind supplied checkpoint")
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("frozen evaluator split mismatch")
    bundle, manifest, _ = _load_checkpoint_bundle(args.checkpoint, config, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint))
    admission.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "exposure_scale"))
    before = state_content_hash(bundle.state_dict())
    reference, panels, exposure = _validation_reference(train_images, train_labels, device), _panels(device), config.conditional_generation.fixed_exposure_scale
    baseline = {name: _metrics(bundle, evaluator, codes, panel_labels, reference, exposure) for name, (codes, panel_labels) in panels.items()}
    params = {"transport_mixing": bundle.generator.transport_mixing, "encoder_phase": bundle.system.encoder.phase, "decoder_phase": bundle.system.decoder.phase}
    for parameter in params.values():
        parameter.grad = None
    objective = sum(_tangent_loss(bundle, evaluator, codes, panel_labels, reference, exposure) for codes, panel_labels in panels.values())
    gradients = torch.autograd.grad(objective, tuple(params.values()), allow_unused=False)
    directions = {"transport_mixing": (gradients[0], torch.zeros_like(gradients[1]), torch.zeros_like(gradients[2])), "encoder_phase": (torch.zeros_like(gradients[0]), gradients[1], torch.zeros_like(gradients[2])), "decoder_phase": (torch.zeros_like(gradients[0]), torch.zeros_like(gradients[1]), gradients[2]), "joint": gradients}
    results = {}
    for name, direction in directions.items():
        if not all(torch.isfinite(item).all() for item in direction) or all(item.norm() <= 1e-12 for item in direction):
            results[name] = {"valid": False, "reason": "zero_or_nonfinite_gradient"}
            continue
        normalized = tuple(-item / item.norm().clamp_min(1e-12) if item.norm() > 1e-12 else item for item in direction)
        probe = copy.deepcopy(bundle)
        with torch.no_grad():
            probe.generator.transport_mixing.add_(normalized[0])
            probe.system.encoder.phase.add_(normalized[1])
            probe.system.decoder.phase.add_(normalized[2])
        logit = max(_logit_delta(bundle, probe, codes, panel_labels) for codes, panel_labels in panels.values())
        phase = float(torch.sqrt(normalized[1].square().mean() + normalized[2].square().mean()).item())
        scale = min(1., .05 / max(logit, 1e-12), .005 / max(phase, 1e-12))
        probe = copy.deepcopy(bundle)
        with torch.no_grad():
            probe.generator.transport_mixing.add_(normalized[0] * scale)
            probe.system.encoder.phase.add_(normalized[1] * scale)
            probe.system.decoder.phase.add_(normalized[2] * scale)
        trial = {panel: _metrics(probe, evaluator, codes, panel_labels, reference, exposure) for panel, (codes, panel_labels) in panels.items()}
        raw_fraction = max(float(((trial[panel]["raw"] - baseline[panel]["raw"]).square().mean().sqrt() / baseline[panel]["raw"].square().mean().sqrt().clamp_min(1e-12)).item()) for panel in panels)
        if raw_fraction > .01:
            scale *= .99 * .01 / raw_fraction
            probe = copy.deepcopy(bundle)
            with torch.no_grad():
                probe.generator.transport_mixing.add_(normalized[0] * scale)
                probe.system.encoder.phase.add_(normalized[1] * scale)
                probe.system.decoder.phase.add_(normalized[2] * scale)
            trial = {panel: _metrics(probe, evaluator, codes, panel_labels, reference, exposure) for panel, (codes, panel_labels) in panels.items()}
            raw_fraction = max(float(((trial[panel]["raw"] - baseline[panel]["raw"]).square().mean().sqrt() / baseline[panel]["raw"].square().mean().sqrt().clamp_min(1e-12)).item()) for panel in panels)
        for panel, (codes, panel_labels) in panels.items():
            trial[panel]["delta_correlation"] = _delta_correlation(probe, codes, panel_labels, trial[panel]["raw"])
            baseline[panel]["delta_correlation"] = _delta_correlation(bundle, codes, panel_labels, baseline[panel]["raw"])
        protected = raw_fraction <= .01 and max(_logit_delta(bundle, probe, codes, panel_labels) for codes, panel_labels in panels.values()) <= .05 and phase * scale <= .005 and all(min(trial[panel]["ratios"]) >= min(baseline[panel]["ratios"]) + .01 and all(value >= base for value, base in zip(trial[panel]["recall"], baseline[panel]["recall"], strict=True)) and trial[panel]["camera_fraction"] >= baseline[panel]["camera_fraction"] - .005 and trial[panel]["delta_correlation"] >= baseline[panel]["delta_correlation"] - .02 and trial[panel]["saturation"] <= .08 for panel in panels)
        results[name] = {"valid": True, "protected_improvement": protected, "scale": scale, "phase_rms": phase * scale, "raw_rms_fraction": raw_fraction, "logit_rms": max(_logit_delta(bundle, probe, codes, panel_labels) for codes, panel_labels in panels.values())}
    after = state_content_hash(bundle.state_dict())
    memory = _memory(device)
    safe = any(item.get("protected_improvement", False) for item in results.values())
    gates = (GateResult("state_unchanged", before == after, {"before": before, "after": after}), GateResult("finite_directions", all(item.get("valid", False) for item in results.values()), results), GateResult("safety_limits", all(not item.get("valid", False) or (item["phase_rms"] <= .005 and item["raw_rms_fraction"] <= .01 and item["logit_rms"] <= .05) for item in results.values()), results), GateResult("safe_distribution_tangent", safe, results), GateResult("cuda", bool(memory["passed"]), memory))
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    report = run.root / "conditional_transport_aligned_manifold_tangent_report.json"
    atomic_write_json(report, {"contract": TANGENT_CONTRACT, "results": results, "state_hash_before": before, "state_hash_after": after, "optimizer_created": False, "checkpoint_created": False, "memory": memory})
    record = EvidenceRecordV2(2, "conditional_transport_aligned_manifold_tangent", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"admission_evidence": file_sha256(args.admission_evidence), "checkpoint": file_sha256(args.checkpoint), "report": file_sha256(report)}, gates, ("transport-aligned-manifold-smoke",) if passed else (), None if passed else "transport_aligned_distribution_tangent_absent")
    _write_evidence(run, "conditional_transport_aligned_manifold_tangent_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _smoke_loss(bundle, evaluator, codes, labels, real, exposure):
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=1, activation_checkpoint=True)
    features, logits = evaluator(fixed_exposure_view(raw, exposure))
    real_features, _ = evaluator(real)
    sinkhorn = conditional_sinkhorn_loss(features, fixed_exposure_view(raw, exposure), real_features, real, labels, epsilon=.05, iterations=20)
    diversity = relative_diversity_loss(features, real_features, labels)
    camera = camera_code_fraction_loss(raw, labels, target=.25)
    zero = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros_like(codes), labels, microbatch_size=1, activation_checkpoint=True)
    delta = 1 - F.cosine_similarity((raw - zero).flatten(1), (.80 * bundle.generator(codes, labels).square() - .80 * bundle.generator(torch.zeros_like(codes), labels).square()).flatten(1), dim=1).mean()
    saturation, tv = torch.relu(raw / exposure - 1).square().mean(), phase_total_variation(bundle.system)
    total = sinkhorn + F.cross_entropy(logits, labels) + diversity + 2 * camera + .1 * saturation + 1e-5 * tv + delta
    return total, {"total": float(total.detach()), "sinkhorn": float(sinkhorn.detach()), "classification": float(F.cross_entropy(logits, labels).detach()), "relative_diversity": float(diversity.detach()), "camera": float(camera.detach()), "saturation": float(saturation.detach()), "phase_tv": float(tv.detach()), "delta": float(delta.detach())}


def transport_aligned_manifold_smoke(args, config) -> int:
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    tangent = _load_evidence(args.tangent_evidence, kind="conditional_transport_aligned_manifold_tangent", action="transport-aligned-manifold-smoke")
    if tangent.input_hashes.get("checkpoint") != file_sha256(args.checkpoint):
        raise ValueError("tangent evidence does not bind supplied checkpoint")
    device = _device(config)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("frozen evaluator split mismatch")
    bundle, _, _ = _load_checkpoint_bundle(args.checkpoint, config, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint))
    tangent.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash", "exposure_scale"))
    initial = {key: value.detach().clone() for key, value in bundle.state_dict().items()}
    reference, exposure = _validation_reference(validation_images, validation_labels, device), config.conditional_generation.fixed_exposure_scale
    run = _new_run(config)
    probes, history = {}, []
    def probe(step):
        before = state_content_hash(bundle.state_dict())
        values = _admission_values(bundle, evaluator, reference, device, exposure)
        memory = _memory(device)
        gates = _admission_gates(*values, before == state_content_hash(bundle.state_dict()), memory)
        for name, (codes, panel_labels) in _panels(device).items():
            raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, panel_labels, microbatch_size=1, activation_checkpoint=False)
            atomic_torch_save(run.root / f"transport_aligned_{name}_step_{step}_raw.pt", raw.cpu())
        return {"gates": [{"name": gate.name, "passed": gate.passed, "metrics": gate.metrics} for gate in gates], "passed": all(gate.passed for gate in gates)}
    probes["0"] = probe(0)
    optimizer = torch.optim.Adam([{"params": [bundle.generator.transport_mixing], "lr": 2e-4}, {"params": [bundle.system.encoder.phase, bundle.system.decoder.phase], "lr": 5e-5}])
    batches = list(balanced_batches(train_labels, batch_size=50, seed=1234))
    for step in range(400):
        indices = batches[step % len(batches)]
        codes, batch_labels = _training_codes(step, device)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _smoke_loss(bundle, evaluator, codes, batch_labels, train_images.index_select(0, indices).to(device), exposure)
        loss.backward()
        gradients = {"mixing": float(bundle.generator.transport_mixing.grad.norm()), "encoder": float(bundle.system.encoder.phase.grad.norm()), "decoder": float(bundle.system.decoder.phase.grad.norm())}
        optimizer.step()
        history.append(summary | {"step": step + 1, "code_hash": tensor_sha256(codes.cpu()), "gradients": gradients})
        if step + 1 in {50, 100, 200, 400}:
            probes[str(step + 1)] = probe(step + 1)
    updates = {"mixing": float((bundle.generator.transport_mixing - initial["generator.transport_mixing"].to(device)).norm()), "encoder": float((bundle.system.encoder.phase - initial["system.encoder.phase"].to(device)).norm()), "decoder": float((bundle.system.decoder.phase - initial["system.decoder.phase"].to(device)).norm())}
    frozen = all(torch.equal(value.cpu(), initial[key].cpu()) for key, value in bundle.state_dict().items() if key not in {"generator.transport_mixing", "system.encoder.phase", "system.decoder.phase"})
    trend = {name: {"first50": sum(row[name] for row in history[:50]) / 50, "last50": sum(row[name] for row in history[-50:]) / 50} for name in ("sinkhorn", "relative_diversity", "delta")}
    memory = _memory(device)
    gates = (GateResult("all_probe_admission", all(item["passed"] for item in probes.values()), probes), GateResult("finite_loss", bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all()), {}), GateResult("allowed_updates", all(value > 0 for value in updates.values()) and frozen, {"updates": updates, "frozen": frozen}), GateResult("loss_trend", all(row["last50"] <= row["first50"] for row in trend.values()), trend), GateResult("cuda", bool(memory["passed"]), memory))
    passed = all(gate.passed for gate in gates)
    history_path = run.root / "conditional_transport_aligned_manifold_smoke_history.json"
    atomic_write_json(history_path, {"contract": SMOKE_CONTRACT, "history": history, "probes": probes, "updates": updates, "trend": trend, "memory": memory})
    inputs = {"tangent_evidence": file_sha256(args.tangent_evidence), "checkpoint": file_sha256(args.checkpoint), "history": file_sha256(history_path)}
    if passed:
        final_path = run.root / "conditional_transport_aligned_manifold_smoke_audit.pt"
        manifest = _manifest(config, split.manifest_hash, evaluator_hash, bundle, file_sha256(args.checkpoint))
        save_conditional_checkpoint(final_path, bundle, manifest, extra={"diagnostic_only": True, "never_warm_start": True, "admission_initialization": file_sha256(args.checkpoint)})
        inputs["diagnostic_final"] = file_sha256(final_path)
    record = EvidenceRecordV2(2, "conditional_transport_aligned_manifold_smoke", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, inputs, gates, ("train-transport-aligned-manifold",) if passed else (), None if passed else "transport_aligned_manifold_smoke_failed")
    _write_evidence(run, "conditional_transport_aligned_manifold_smoke_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def train_transport_aligned_manifold(args, config) -> int:
    _require(config)
    raise RuntimeError("v7.1 formal training is intentionally fail-closed in this iteration")
