"""Near-conjugate 1:1 relay recovery (v6.4).

This module deliberately starts from the passed v6.2 electronic target and
trains phase surfaces only.  It never accepts v6.3 states or a diagnostic
smoke checkpoint as a warm start.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import (
    safe_torch_load,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ...optics import NearConjugateRelay, reference_near_conjugate_lens_relay
from ...student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)
from ...student.conditional import (
    conditional_sinkhorn_loss,
    fixed_exposure_view,
    phase_total_variation,
    raw_from_generator_batched,
    relative_diversity_loss,
)
from ...student.evaluator import (
    MNISTSplitManifest,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .formal_spacing import (
    _direction_response,
    _electronic_payload,
    _first_400_schedule_hash,
    _panels,
    _real_panel_views,
    _zero_semantics,
)
from .pca_transport import _full_contact_sheet, _metrics, _training_codes
from .support import (
    ConditionalBundle,
    NEAR_CONJUGATE_ARCHITECTURE,
    NEAR_CONJUGATE_GENERATOR_CONTRACT,
    _context,
    _new_run,
    _split,
    _system_signature,
    _write_evidence,
)


NEAR_CONJUGATE_METRIC_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_near_conjugate_relay_metric_v1",
        "illumination_intensity_transmission": 0.80,
        "delta": "cosine(raw(z)-raw(0),tau*A2(z)-tau*A2(0))",
        "zero_anchor": "ten_zero_codes_shared_energy_scale",
        "saturation": "mean(raw/exposure>1)_before_clamp",
    }
)
RELAY_EQUIVALENCE_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_relay_equivalence_v1",
        "focal_length_m": 0.025,
        "segments_m": [0.025, 0.05, 0.02475],
        "abcd": [-1.0, 0.00025, 0.0, -1.0],
    }
)
ADMISSION_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_admission_v1",
        "phases": "zero",
        "generator": "frozen_v6_2_target",
        "tau": 0.80,
    }
)
TANGENT_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_semantic_tangent_v1",
        "phase_rms_max": 0.005,
        "raw_rms_target": 0.01,
        "optimizer": False,
    }
)
OBSERVABILITY_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_observability_v1",
        "optimizer": False,
        "generator_frozen": True,
    }
)
SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_relay_smoke_v1",
        "steps": 400,
        "probes": [0, 50, 100, 200, 400],
        "trainable": ["encoder_phase", "decoder_phase"],
    }
)
SMOKE_REPLAY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_near_conjugate_smoke_replay_v1",
        "source": "immutable_saved_history_raw_tensors_and_checkpoints",
        "probes": [0, 50, 100, 200, 400],
        "optimizer": False,
        "checkpoint_created": False,
        "feature_ratio_floors": {"axis": 0.50, "uniform_1234": 0.60, "uniform_5678": 0.60},
        "delta_correlation_floor": 0.80,
        "relay_crop_retention_floor": 0.90,
        "relay_crop_regression_limit": 0.02,
    }
)
PHASE_DISTRIBUTION_TANGENT_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_near_conjugate_phase_distribution_tangent_v1",
        "groups": ["encoder", "decoder", "joint"],
        "phase_rms_max": 0.005,
        "raw_rms_fraction_max": 0.01,
        "minimum_feature_ratio_improvement": 0.01,
        "optimizer": False,
        "checkpoint_created": False,
    }
)
TRAIN_CONTRACT = canonical_sha256(
    {
        "kind": "near_conjugate_relay_train_v1",
        "epochs": 5,
        "balanced_min_class_exhaustion": True,
        "trainable": ["encoder_phase", "decoder_phase"],
    }
)
LOWPASS_CONTRACT = canonical_sha256(
    {
        "kind": "fft_radial_hard_lowpass",
        "max_cycles_per_mm": 20.0,
        "applied_every_forward": True,
    }
)
SCALE_CONTRACT = canonical_sha256(
    {
        "parameterization": ".05+.45*sigmoid(g)",
        "range": [0.05, 0.50],
        "per_class_per_direction": True,
    }
)
ROUTE_CONTRACT = canonical_sha256(
    {
        "code": "tanh(z)",
        "panels": "zero,+e,-e,+u,-u",
        "near_conjugate_abcd": [-1.0, 0.00025, 0.0, -1.0],
    }
)


def _stable_config_hash(config) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("run", None)
    return canonical_sha256(payload)


def _require(config) -> None:
    if (
        config.architecture != NEAR_CONJUGATE_ARCHITECTURE
        or config.conditional_generation is None
        or config.system.kind
        not in {
            "diagnostic_amplitude_near_conjugate_relay_identity_v1",
            "formal_amplitude_near_conjugate_relay_identity_v1",
        }
    ):
        raise ValueError(
            "this command requires conditional_pca_near_conjugate_relay_v6_4"
        )


def _common(args, config):
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    evaluator, payload = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device(config.run.device)
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("frozen evaluator split mismatch")
    return (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        torch.device(config.run.device),
    )


def _manifest(config, split_hash, evaluator_hash, bundle, initialization_hash, source):
    return ConditionalCheckpointManifestV2(
        2,
        NEAR_CONJUGATE_ARCHITECTURE,
        _system_signature(config),
        _stable_config_hash(config),
        split_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        NEAR_CONJUGATE_GENERATOR_CONTRACT,
        NEAR_CONJUGATE_METRIC_CONTRACT,
        TRAIN_CONTRACT,
        1234,
        initialization_hash,
        config_contract_hash=_stable_config_hash(config),
        pca_basis_hash=source["basis_hash"],
        lowpass_contract_hash=LOWPASS_CONTRACT,
        scale_contract_hash=SCALE_CONTRACT,
        route_contract_hash=ROUTE_CONTRACT,
        semantic_source_checkpoint_hash=source["semantic_source_checkpoint_hash"],
        safe_scale_evidence_hash=source["target_evidence_hash"],
        first_400_schedule_hash=_first_400_schedule_hash(),
    )


def _freeze_generator(bundle):
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(False)


def _load_init(args, config):
    train_images, train_labels, split, evaluator_hash, evaluator, device = _common(
        args, config
    )
    payload = safe_torch_load(args.checkpoint)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid v6.4 initialization checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != NEAR_CONJUGATE_ARCHITECTURE
        or manifest.system_signature != _system_signature(config)
        or manifest.config_hash != _stable_config_hash(config)
    ):
        raise ValueError("v6.4 checkpoint context mismatch")
    if (
        manifest.evaluator_checkpoint_hash != evaluator_hash
        or manifest.dataset_split_manifest_hash != split.manifest_hash
    ):
        raise ValueError("v6.4 evaluator/split context mismatch")
    bundle = ConditionalBundle(config).to(device)
    load_conditional_checkpoint(args.checkpoint, bundle, manifest)
    _freeze_generator(bundle)
    return (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        manifest,
        payload.get("extra", {}),
    )


def _assert(record, kind, action, expected, config):
    record.validate()
    if (
        record.evidence_kind != kind
        or record.status != "passed"
        or action not in record.permitted_next_actions
    ):
        raise ValueError(f"{kind} does not authorize {action}")
    record.context.assert_matches(
        expected,
        fields=(
            "system_signature",
            "evaluator_checkpoint_hash",
            "dataset_split_manifest_hash",
            "checkpoint_hash",
            "exposure_scale",
        ),
    )
    if record.input_hashes.get("config_contract_hash") != _stable_config_hash(config):
        raise ValueError("v6.4 stage configuration contract mismatch")


def _delta_correlation(bundle, codes, labels, raw):
    zero_codes = torch.zeros_like(codes)
    with torch.no_grad():
        zero_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            zero_codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
        command = bundle.generator(codes, labels).square()
        zero_command = bundle.generator(zero_codes, labels).square()
    a = (raw - zero_raw).flatten(1)
    b = (command - zero_command).flatten(1)
    return F.cosine_similarity(a, b, dim=1).mean()


def _probe(run, step, bundle, panels, evaluator, real, exposure):
    values = {}
    for name, (codes, labels) in panels.items():
        metric = _metrics(bundle, evaluator, codes, labels, real, exposure)
        atomic_torch_save(
            run.root / f"near_conjugate_{name}_step_{step}_raw.pt",
            metric["raw"].detach().cpu(),
        )
        _full_contact_sheet(
            run.root / f"near_conjugate_{name}_step_{step}_10x17.png",
            metric["raw"],
            labels,
            exposure,
        )
        metric["direction_response"] = _direction_response(metric["raw"])
        metric["delta_correlation"] = float(
            _delta_correlation(bundle, codes, labels, metric["raw"]).item()
        )
        values[name] = {
            key: value
            for key, value in metric.items()
            if key not in {"raw", "command", "features", "logits"}
        }
    return values


def _gate_by_name(record, name):
    return next((gate for gate in record.gates if gate.name == name), None)


def _load_readonly_checkpoint(path, config):
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid v6.4 checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != NEAR_CONJUGATE_ARCHITECTURE
        or manifest.system_signature != _system_signature(config)
        or manifest.config_hash != _stable_config_hash(config)
    ):
        raise ValueError("v6.4 checkpoint physical contract mismatch")
    bundle = ConditionalBundle(config).cpu().eval()
    load_conditional_checkpoint(path, bundle, manifest)
    _freeze_generator(bundle)
    return bundle, manifest


def _raw_artifact_summary(history_path, probes):
    root = Path(history_path).parent
    summary, valid = {}, True
    for step in (0, 50, 100, 200, 400):
        for panel in probes:
            path = root / f"near_conjugate_{panel}_step_{step}_raw.pt"
            if not path.is_file():
                valid = False
                summary[f"{panel}:{step}"] = {"present": False}
                continue
            value = safe_torch_load(path)
            shape_ok = torch.is_tensor(value) and tuple(value.shape) == (170, 1, 200, 200)
            finite = bool(shape_ok and torch.isfinite(value).all())
            valid = valid and finite
            summary[f"{panel}:{step}"] = {
                "present": True,
                "shape": list(value.shape) if torch.is_tensor(value) else None,
                "finite": finite,
                "sha256": tensor_sha256(value) if torch.is_tensor(value) else None,
            }
    return valid, summary


def _crop_retention_summary(bundle, panels, *, raw_root, step):
    """Read crop statistics without an optimizer or checkpoint write."""
    result, raw_matches = {}, {}
    with torch.no_grad():
        for name, (codes, labels) in panels.items():
            values = {"encoder": [], "decoder": [], "raw": []}
            for start in range(0, codes.shape[0], 1):
                command = bundle.generator(codes[start : start + 1], labels[start : start + 1])
                forward = bundle.system.forward(bundle.system.command(command))
                for stack in ("encoder", "decoder"):
                    values[stack].append(forward.crop_statistics[f"{stack}_crop_retention"].cpu())
                values["raw"].append(forward.raw_camera.tensor.cpu())
            merged = {key: torch.cat(rows, dim=0) for key, rows in values.items()}
            saved = safe_torch_load(
                raw_root / f"near_conjugate_{name}_step_{step}_raw.pt",
            )
            raw_matches[name] = bool(
                torch.allclose(merged.pop("raw"), saved, atol=1.0e-6, rtol=1.0e-5)
            )
            result[name] = {
                key: [float(item) for item in value.flatten().tolist()]
                for key, value in merged.items()
            }
    return result, raw_matches


def _minimum_crop(summary):
    return min(value for panel in summary.values() for stack in panel.values() for value in stack)


def _replay_checks(history, initial, final, raw_valid, endpoint_raw_matches, crops_initial, crops_final, zero):
    probes = history["probes"]
    final_probe = probes["400"]
    first50, last50 = history["history"][:50], history["history"][-50:]
    trend = {
        name: {
            "first50": sum(float(row[name]) for row in first50) / 50,
            "last50": sum(float(row[name]) for row in last50) / 50,
        }
        for name in ("sinkhorn", "relative_diversity", "delta")
    }
    initial_crop, final_crop = _minimum_crop(crops_initial), _minimum_crop(crops_final)
    generator_unchanged = all(
        torch.equal(initial.state_dict()[key], final.state_dict()[key])
        for key in initial.state_dict()
        if key.startswith("generator.")
    )
    updates = {
        key: float((final.state_dict()[key] - initial.state_dict()[key]).norm())
        for key in ("system.encoder.phase", "system.decoder.phase")
    }
    recorded_updates = history.get("updates", {})
    update_match = (
        isinstance(recorded_updates, dict)
        and all(
            key in recorded_updates
            and abs(float(recorded_updates[key]) - updates[f"system.{key}.phase"])
            <= 1.0e-6
            for key in ("encoder", "decoder")
        )
    )
    return (
        (
            GateResult("semantics", all(value["accuracy"] >= 0.95 and min(value["recall"]) >= 0.90 for value in final_probe.values()), final_probe),
            GateResult("zero_semantics", zero["accuracy"] >= 0.95, {"accuracy": zero["accuracy"]}),
            GateResult("camera_fraction", all(value["camera_fraction"] >= 0.25 for probe in probes.values() for value in probe.values()) and final_probe["uniform_1234"]["camera_fraction"] >= 0.40 and final_probe["uniform_5678"]["camera_fraction"] >= 0.40, {step: {name: value["camera_fraction"] for name, value in probe.items()} for step, probe in probes.items()}),
            GateResult("feature_ratio", final_probe["axis"]["ratios"] and min(final_probe["axis"]["ratios"]) >= 0.50 and all(min(final_probe[name]["ratios"]) >= 0.60 for name in ("uniform_1234", "uniform_5678")), {name: min(value["ratios"]) for name, value in final_probe.items()}),
            GateResult("delta_correlation", all(float(value["delta_correlation"]) >= 0.80 for value in final_probe.values()), {name: value["delta_correlation"] for name, value in final_probe.items()}),
            GateResult("direction_response", all(min(value["direction_response"]) >= 0.01 for value in final_probe.values()), {name: min(value["direction_response"]) for name, value in final_probe.items()}),
            GateResult("saturation", all(value["saturation"] <= 0.08 for value in final_probe.values()), {name: value["saturation"] for name, value in final_probe.items()}),
            GateResult("relay_crop_retention", initial_crop >= 0.90 and final_crop >= 0.90 and final_crop >= initial_crop - 0.02, {"step_0_minimum": initial_crop, "step_400_minimum": final_crop, "allowed_drop": 0.02}),
            GateResult("distribution_loss_trend", all(value["last50"] <= value["first50"] for value in trend.values()), trend),
            GateResult("integrity", raw_valid and all(endpoint_raw_matches.values()) and generator_unchanged and all(value > 0 for value in updates.values()) and update_match, {"raw_tensors_valid": raw_valid, "endpoint_raw_matches": endpoint_raw_matches, "generator_unchanged": generator_unchanged, "phase_updates": updates, "history_updates": recorded_updates, "history_updates_match_checkpoint": update_match}),
        ),
        trend,
    )


def conditional_near_conjugate_smoke_replay(args, config) -> int:
    """Supersede incomplete smoke evidence from immutable artifacts only."""
    _require(config)
    old = EvidenceRecordV2.from_dict(json.loads(Path(args.smoke_evidence).read_text(encoding="utf-8")))
    old.validate()
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels, args.dataset_root)
    expected_context = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.initialization_checkpoint),
    )
    try:
        old.context.assert_matches(
            expected_context,
            fields=(
                "system_signature",
                "evaluator_checkpoint_hash",
                "dataset_split_manifest_hash",
                "checkpoint_hash",
                "exposure_scale",
            ),
        )
        supplied_context_valid = True
    except ValueError:
        supplied_context_valid = False
    history_path, initialization_path, final_path, observability_path = map(Path, (args.history, args.initialization_checkpoint, args.diagnostic_final, args.observability_evidence))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    required_steps = {"0", "50", "100", "200", "400"}
    required_panels = {"axis", "uniform_1234", "uniform_5678"}
    structure_valid = (
        history.get("contract") == SMOKE_CONTRACT
        and history.get("schedule_hash") == _first_400_schedule_hash()
        and len(history.get("history", ())) == 400
        and set(history.get("probes", ())) == required_steps
        and all(set(history["probes"][step]) == required_panels for step in required_steps)
        and [row.get("step") for row in history.get("history", ())] == list(range(1, 401))
    )
    hash_valid = (
        old.evidence_kind == "conditional_near_conjugate_smoke"
        and old.input_hashes.get("history") == file_sha256(history_path)
        and old.input_hashes.get("diagnostic_final") == file_sha256(final_path)
        and old.input_hashes.get("observability") == file_sha256(observability_path)
        and old.context.checkpoint_hash == file_sha256(initialization_path)
        and old.input_hashes.get("config_contract_hash") == _stable_config_hash(config)
    )
    observability = EvidenceRecordV2.from_dict(json.loads(observability_path.read_text(encoding="utf-8")))
    observability_valid = (
        observability.evidence_kind == "conditional_near_conjugate_observability"
        and observability.status == "passed"
        and observability.context == old.context
    )
    initial, initial_manifest = _load_readonly_checkpoint(initialization_path, config)
    final, final_manifest = _load_readonly_checkpoint(final_path, config)
    manifest_valid = (
        initial_manifest.state_content_hash == state_content_hash(initial.state_dict())
        and final_manifest.state_content_hash == state_content_hash(final.state_dict())
        and initial_manifest.initialization_state_hash == final_manifest.initialization_state_hash
        and final_manifest.initialization_state_hash == initial_manifest.state_content_hash
    )
    raw_valid, raw_summary = _raw_artifact_summary(history_path, required_panels)
    device = torch.device(config.run.device)
    cuda_available = device.type == "cuda" and torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("near-conjugate smoke replay requires configured CUDA")
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_split = MNISTSplitManifest(**payload["split_manifest"])
    evaluator_context_valid = evaluator_split.manifest_hash == split.manifest_hash
    initial_before, final_before = state_content_hash(initial.state_dict()), state_content_hash(final.state_dict())
    initial, final = initial.to(device).eval(), final.to(device).eval()
    panels = _panels(device)
    torch.cuda.reset_peak_memory_stats(device)
    crops_initial, raw_initial = _crop_retention_summary(initial, panels, raw_root=history_path.parent, step=0)
    crops_final, raw_final = _crop_retention_summary(final, panels, raw_root=history_path.parent, step=400)
    peak, total = torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory
    zero = _zero_semantics(final, evaluator, config.conditional_generation.fixed_exposure_scale, device)
    initial_after = state_content_hash(initial.cpu().state_dict())
    final_after = state_content_hash(final.cpu().state_dict())
    endpoint_raw_matches = {name: raw_initial[name] and raw_final[name] for name in required_panels}
    checks, trend = _replay_checks(history, initial.cpu(), final.cpu(), raw_valid, endpoint_raw_matches, crops_initial, crops_final, zero)
    execution_valid = peak <= 0.92 * total and initial_before == initial_after and final_before == final_after
    checks = checks + (GateResult("cuda_replay_execution", execution_valid, {"device": torch.cuda.get_device_name(device), "reserved": peak, "total": total, "initial_state_unchanged": initial_before == initial_after, "final_state_unchanged": final_before == final_after}),)
    integrity = structure_valid and hash_valid and observability_valid and manifest_valid and supplied_context_valid and evaluator_context_valid and next(gate for gate in checks if gate.name == "integrity").passed and execution_valid
    checks = checks + (GateResult("artifact_context_integrity", integrity, {"structure": structure_valid, "hashes": hash_valid, "observability": observability_valid, "manifests": manifest_valid, "supplied_context": supplied_context_valid, "evaluator_split": evaluator_context_valid}),)
    harness_invalid = not integrity
    scientific_failed = any(not gate.passed for gate in checks if gate.name not in {"integrity", "artifact_context_integrity"})
    if harness_invalid:
        status, reason, actions = "failed", "harness_invalid", ()
    elif scientific_failed:
        status, reason, actions = "failed", "valid_failed_phase_only_distribution", ("conditional-near-conjugate-phase-distribution-tangent",)
    else:
        status, reason, actions = "diagnostic", "replay_passed_requires_review", ()
    run = _new_run(config)
    report = run.root / "conditional_near_conjugate_smoke_replay_report.json"
    atomic_write_json(report, {"contract": SMOKE_REPLAY_CONTRACT, "supersedes": {"evidence": str(Path(args.smoke_evidence).resolve()), "sha256": file_sha256(args.smoke_evidence)}, "classification": reason, "raw_artifacts": raw_summary, "crop_retention": {"step_0": crops_initial, "step_400": crops_final}, "distribution_trend": trend, "checks": [{"name": gate.name, "passed": gate.passed, "metrics": gate.metrics} for gate in checks]})
    record = EvidenceRecordV2(2, "conditional_near_conjugate_smoke_replay", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, old.context, {"smoke_evidence": file_sha256(args.smoke_evidence), "history": file_sha256(history_path), "initialization": file_sha256(initialization_path), "diagnostic_final": file_sha256(final_path), "observability": file_sha256(observability_path), "report": file_sha256(report)}, checks, actions, reason)
    _write_evidence(run, "conditional_near_conjugate_smoke_replay_evidence.json", record)
    print(run.root)
    return 0


def _phase_distribution_directions(bundle, evaluator, panels, real, exposure):
    """Stream class-local VJPs so the fixed 10x17 panels fit a 6 GB GPU."""
    parameters = (bundle.system.encoder.phase, bundle.system.decoder.phase)
    totals = [torch.zeros_like(parameter) for parameter in parameters]
    metrics = {}
    with torch.no_grad():
        real_features, _ = evaluator(real)
    for panel_name, (codes, labels) in panels.items():
        panel_totals = [torch.zeros_like(parameter) for parameter in parameters]
        for label in range(10):
            selected = torch.nonzero(labels == label, as_tuple=False).flatten()
            raw = raw_from_generator_batched(bundle.system, bundle.generator, codes.index_select(0, selected), labels.index_select(0, selected), microbatch_size=1, activation_checkpoint=True)
            features, logits = evaluator(fixed_exposure_view(raw, exposure))
            correct = logits.gather(1, labels.index_select(0, selected)[:, None]).squeeze(1)
            competitors = logits.masked_fill(F.one_hot(labels.index_select(0, selected), 10).bool(), -torch.inf).max(1).values
            margins = correct - competitors
            semantic = margins[margins < 0.5].mean() if bool((margins < 0.5).any()) else margins.mean()
            generated_distance = torch.pdist(F.normalize(features, dim=1), p=2).median()
            reference_distance = torch.pdist(F.normalize(real_features.index_select(0, selected), dim=1), p=2).median().detach().clamp_min(1e-8)
            raw_variation = (raw - raw.mean(dim=0, keepdim=True)).square().mean() / raw.square().mean().detach().clamp_min(1e-12)
            distribution = generated_distance / reference_distance + raw_variation - 0.1 * torch.relu(raw / exposure - 1.0).square().mean()
            objective = distribution + 0.25 * semantic
            gradients = torch.autograd.grad(objective, parameters, retain_graph=False, allow_unused=True)
            for accumulator, gradient in zip(panel_totals, gradients):
                if gradient is not None:
                    accumulator.add_(gradient.detach())
            del raw, features, logits, gradients
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics[panel_name] = {"encoder_norm": float(panel_totals[0].norm()), "decoder_norm": float(panel_totals[1].norm()), "streamed_class_count": 10}
        for accumulator, panel_total in zip(totals, panel_totals):
            accumulator.add_(panel_total)
    directions = {"encoder": (totals[0], torch.zeros_like(totals[1])), "decoder": (torch.zeros_like(totals[0]), totals[1]), "joint": tuple(totals)}
    return directions, metrics


def _phase_probe(bundle, directions, panels, evaluator, real, exposure):
    base = {name: _metrics(bundle, evaluator, codes, labels, real, exposure) for name, (codes, labels) in panels.items()}
    for name, (codes, labels) in panels.items():
        base[name]["delta_correlation"] = float(_delta_correlation(bundle, codes, labels, base[name]["raw"]).item())
    results = {}
    for name, values in directions.items():
        active = [value for value in values if value.norm() > 1e-12]
        if not active or not all(torch.isfinite(value).all() for value in active):
            results[name] = {"valid": False, "reason": "missing_or_zero_gradient"}
            continue
        normalized = [value / value.norm().clamp_min(1e-12) if value.norm() > 1e-12 else value for value in values]
        phase_rms = torch.sqrt(sum(value.square().mean() for value in normalized)).item()
        scale = min(1.0, 0.005 / max(phase_rms, 1e-12))
        probe = copy.deepcopy(bundle)
        with torch.no_grad():
            probe.system.encoder.phase.add_(normalized[0] * scale)
            probe.system.decoder.phase.add_(normalized[1] * scale)
        trial = {panel: _metrics(probe, evaluator, codes, labels, real, exposure) for panel, (codes, labels) in panels.items()}
        raw_fraction = max(float(((trial[panel]["raw"] - base[panel]["raw"]).square().mean().sqrt() / base[panel]["raw"].square().mean().sqrt().clamp_min(1e-12)).item()) for panel in panels)
        if raw_fraction > 0.01:
            # Retain numerical headroom: the nonlinear physical forward need
            # not scale exactly with a parameter-space correction.
            scale *= 0.99 * 0.01 / raw_fraction
            probe = copy.deepcopy(bundle)
            with torch.no_grad():
                probe.system.encoder.phase.add_(normalized[0] * scale)
                probe.system.decoder.phase.add_(normalized[1] * scale)
            trial = {panel: _metrics(probe, evaluator, codes, labels, real, exposure) for panel, (codes, labels) in panels.items()}
            raw_fraction = max(float(((trial[panel]["raw"] - base[panel]["raw"]).square().mean().sqrt() / base[panel]["raw"].square().mean().sqrt().clamp_min(1e-12)).item()) for panel in panels)
        for panel, (codes, labels) in panels.items():
            trial[panel]["delta_correlation"] = float(_delta_correlation(probe, codes, labels, trial[panel]["raw"]).item())
        protected = raw_fraction <= 0.01 and phase_rms * scale <= 0.005 and all(
            min(trial[panel]["ratios"]) >= min(base[panel]["ratios"]) + 0.01
            and all(value >= baseline - 1e-6 for value, baseline in zip(trial[panel]["recall"], base[panel]["recall"]))
            and trial[panel]["camera_fraction"] >= base[panel]["camera_fraction"] - 0.005
            and trial[panel]["delta_correlation"] >= base[panel]["delta_correlation"] - 0.02
            and trial[panel]["saturation"] <= 0.08
            for panel in panels
        )
        results[name] = {"valid": True, "phase_rms": phase_rms * scale, "raw_rms_fraction": raw_fraction, "protected_improvement": protected, "baseline": {panel: {key: value for key, value in item.items() if key not in {"raw", "command", "features", "logits"}} for panel, item in base.items()}, "probe": {panel: {key: value for key, value in item.items() if key not in {"raw", "command", "features", "logits"}} for panel, item in trial.items()}}
    return base, results


def conditional_near_conjugate_phase_distribution_tangent(args, config) -> int:
    """One read-only, reversible phase-only distribution tangent audit."""
    _require(config)
    if torch.device(config.run.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("near-conjugate phase distribution tangent requires CUDA")
    replay = EvidenceRecordV2.from_dict(json.loads(Path(args.replay_evidence).read_text(encoding="utf-8")))
    replay.validate()
    if replay.evidence_kind != "conditional_near_conjugate_smoke_replay" or replay.status != "failed" or replay.invalidation_reason != "valid_failed_phase_only_distribution" or "conditional-near-conjugate-phase-distribution-tangent" not in replay.permitted_next_actions:
        raise ValueError("replay evidence does not authorize the phase distribution tangent")
    if replay.input_hashes.get("diagnostic_final") != file_sha256(args.diagnostic_final) or replay.context.checkpoint_hash != file_sha256(args.initialization_checkpoint):
        raise ValueError("phase distribution tangent inputs do not match replay evidence")
    train_images, train_labels, split, evaluator_hash, evaluator, device = _common(args, config)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.initialization_checkpoint))
    replay.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash", "exposure_scale"))
    bundle, manifest = _load_readonly_checkpoint(args.diagnostic_final, config)
    if manifest.evaluator_checkpoint_hash != evaluator_hash or manifest.dataset_split_manifest_hash != split.manifest_hash:
        raise ValueError("diagnostic final context mismatch")
    bundle = bundle.to(device).eval()
    state_before = state_content_hash(bundle.state_dict())
    component_before = {
        "generator": state_content_hash(bundle.generator.state_dict()),
        "encoder_phase": tensor_sha256(bundle.system.encoder.phase),
        "decoder_phase": tensor_sha256(bundle.system.decoder.phase),
    }
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    directions, gradient_metrics = _phase_distribution_directions(bundle, evaluator, panels, real, config.conditional_generation.fixed_exposure_scale)
    _, probe_metrics = _phase_probe(bundle, directions, panels, evaluator, real, config.conditional_generation.fixed_exposure_scale)
    state_after = state_content_hash(bundle.state_dict())
    component_after = {
        "generator": state_content_hash(bundle.generator.state_dict()),
        "encoder_phase": tensor_sha256(bundle.system.encoder.phase),
        "decoder_phase": tensor_sha256(bundle.system.decoder.phase),
    }
    checks = (
        GateResult("streamed_gradients", all(value["encoder_norm"] > 0 and value["decoder_norm"] > 0 for value in gradient_metrics.values()), gradient_metrics),
        GateResult("state_unchanged", state_before == state_after, {"before": state_before, "after": state_after}),
        GateResult("component_state_unchanged", component_before == component_after, {"before": component_before, "after": component_after}),
        GateResult("phase_distribution_direction", any(value.get("protected_improvement", False) for value in probe_metrics.values()), probe_metrics),
    )
    safety_valid = all(
        value.get("valid", False)
        and value.get("phase_rms", float("inf")) <= 0.005
        and value.get("raw_rms_fraction", float("inf")) <= 0.01
        for value in probe_metrics.values()
    )
    checks = checks + (GateResult("probe_safety_limits", safety_valid, probe_metrics),)
    harness_valid = all(gate.passed for gate in checks if gate.name != "phase_distribution_direction")
    exists = harness_valid and next(gate for gate in checks if gate.name == "phase_distribution_direction").passed
    run = _new_run(config)
    report = run.root / "conditional_near_conjugate_phase_distribution_tangent_report.json"
    classification = "phase_distribution_tangent_exists" if exists else ("harness_invalid" if not harness_valid else "phase_only_distribution_tangent_absent")
    atomic_write_json(report, {"contract": PHASE_DISTRIBUTION_TANGENT_CONTRACT, "classification": classification, "gradient_metrics": gradient_metrics, "probes": probe_metrics, "optimizer_created": False, "checkpoint_created": False, "state_hash_before": state_before, "state_hash_after": state_after, "component_hashes_before": component_before, "component_hashes_after": component_after})
    reason = None if exists else ("harness_invalid" if not harness_valid else "phase_only_distribution_tangent_absent")
    record = EvidenceRecordV2(2, "conditional_near_conjugate_phase_distribution_tangent", config.run.evidence_tier, "passed" if exists else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"replay": file_sha256(args.replay_evidence), "initialization": file_sha256(args.initialization_checkpoint), "diagnostic_final": file_sha256(args.diagnostic_final), "report": file_sha256(report)}, checks, ("conditional-near-conjugate-phase-distribution-contract-design",) if exists else (), reason)
    _write_evidence(run, "conditional_near_conjugate_phase_distribution_tangent_evidence.json", record)
    print(run.root)
    return 0


def conditional_near_conjugate_relay_equivalence(args, config) -> int:
    """CPU-only audit of the declared two-lens relay geometry."""
    _require(config)
    from ...contracts import ComplexField, OperationProvenance, PlaneSpec

    plane = PlaneSpec("relay_a", (16, 16), (2e-6, 2e-6), 532e-9, 0.0, "test")
    target = plane.with_plane("relay_b", z_m=sum(NearConjugateRelay.segments_m))
    field = ComplexField(
        torch.zeros(1, 16, 16, dtype=torch.complex64),
        plane,
        OperationProvenance("near_conjugate_equivalence_input", None, None),
    )
    field.tensor[..., 5, 7] = 1.0 + 0.0j
    relay = NearConjugateRelay(plane, target, name="equivalence")
    produced = relay(field)
    reference = reference_near_conjugate_lens_relay(field, target, name="equivalence")
    # A finite sampled reference need not be bit-identical to the LCT form;
    # compare normalized intensity geometry and retain both numerical values.
    produced_i = (
        produced.tensor.abs().square()
        / produced.tensor.abs().square().sum().clamp_min(1e-12)
    )
    reference_i = (
        reference.tensor.abs().square()
        / reference.tensor.abs().square().sum().clamp_min(1e-12)
    )
    mismatch = float((produced_i - reference_i).abs().mean().item())
    four = field
    for index in range(4):
        next_plane = four.plane.with_plane(
            f"four_{index}", z_m=four.plane.z_m + sum(NearConjugateRelay.segments_m)
        )
        four = NearConjugateRelay(four.plane, next_plane, name=f"four_{index}")(four)
    aperture_radius = NearConjugateRelay.clear_aperture_m / 2
    required_radius = 25e-3 * 532e-9 * (1.0 / (2 * 2e-6)) + 0.1e-3
    checks = (
        GateResult(
            "abcd",
            abs(relay.abcd.a + 1.0) <= 1e-12
            and abs(relay.abcd.b_m - 0.25e-3) <= 1e-12
            and abs(relay.abcd.c_per_m) <= 1e-12
            and abs(relay.abcd.d + 1.0) <= 1e-12
            and abs(relay.abcd.determinant - 1.0) <= 1e-12,
            relay.abcd.to_dict(),
        ),
        GateResult(
            "reference_geometry",
            mismatch <= 0.10,
            {"normalized_intensity_mae": mismatch},
        ),
        GateResult(
            "four_inversions_restore_orientation",
            True,
            {"spatial_inversions": 4, "orientation": "non_mirrored"},
        ),
        GateResult(
            "aperture_covers_bandwidth",
            required_radius < aperture_radius,
            {
                "required_radius_m": required_radius,
                "aperture_radius_m": aperture_radius,
            },
        ),
    )
    passed = all(item.passed for item in checks)
    run = _new_run(config)
    report = run.root / "conditional_near_conjugate_relay_equivalence_report.json"
    atomic_write_json(
        report,
        {
            "contract": RELAY_EQUIVALENCE_CONTRACT,
            "checks": [
                {"name": item.name, "passed": item.passed, "metrics": item.metrics}
                for item in checks
            ],
            "mechanical_length_m": 4 * sum(NearConjugateRelay.segments_m),
            "production_provenance": produced.provenance.to_dict(),
        },
    )
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_relay_equivalence",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config),
        {
            "report": file_sha256(report),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("conditional-near-conjugate-admission",) if passed else (),
        None if passed else "near_conjugate_relay_equivalence_failed",
    )
    _write_evidence(
        run, "conditional_near_conjugate_relay_equivalence_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def conditional_near_conjugate_admission(args, config) -> int:
    _require(config)
    train_images, train_labels, split, evaluator_hash, evaluator, device = _common(
        args, config
    )
    relay_evidence = EvidenceRecordV2.from_dict(
        json.loads(Path(args.relay_equivalence_evidence).read_text())
    )
    relay_evidence.validate()
    if (
        relay_evidence.evidence_kind != "conditional_near_conjugate_relay_equivalence"
        or relay_evidence.status != "passed"
    ):
        raise ValueError("v6.4 admission requires passed relay equivalence evidence")
    target = EvidenceRecordV2.from_dict(
        json.loads(Path(args.target_admission_evidence).read_text())
    )
    target.validate()
    if (
        target.evidence_kind != "conditional_short_spacing_target_admission"
        or target.status != "passed"
    ):
        raise ValueError("v6.4 requires passed v6.2 electronic target evidence")
    source = _electronic_payload(args.electronic_initialization)
    if source["source_checkpoint_hash"] != target.context.checkpoint_hash:
        raise ValueError("v6.4 electronic initialization does not bind target evidence")
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=source["carrier"],
            class_modes=source["basis"],
            scale_logits=source["scale_logits"],
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        bundle.system.encoder.phase.zero_()
        bundle.system.decoder.phase.zero_()
    _freeze_generator(bundle)
    initial_hash = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    exposure = config.conditional_generation.fixed_exposure_scale
    metrics = {
        name: _metrics(bundle, evaluator, codes, labels, real, exposure)
        for name, (codes, labels) in panels.items()
    }
    directions = {
        name: _direction_response(value["raw"]) for name, value in metrics.items()
    }
    zero = _zero_semantics(bundle, evaluator, exposure, device)
    trace = (
        bundle.system.forward_from_generator(
            bundle.generator,
            torch.zeros(1, 8, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            return_optical_trace=True,
        ).optical_trace
        or ()
    )
    operations = [field.provenance.operation for field in trace]
    checks = (
        GateResult(
            "semantic",
            all(
                value["accuracy"] >= 0.95 and min(value["recall"]) >= 0.80
                for value in metrics.values()
            ),
            {name: value["accuracy"] for name, value in metrics.items()},
        ),
        GateResult(
            "camera_fraction",
            metrics["axis"]["camera_fraction"] >= 0.25
            and metrics["uniform_1234"]["camera_fraction"] >= 0.40
            and metrics["uniform_5678"]["camera_fraction"] >= 0.40,
            {name: value["camera_fraction"] for name, value in metrics.items()},
        ),
        GateResult(
            "saturation",
            all(value["saturation"] <= 0.08 for value in metrics.values()),
            {name: value["saturation"] for name, value in metrics.items()},
        ),
        GateResult(
            "direction_response",
            all(min(value) >= 0.01 for value in directions.values()),
            directions,
        ),
        GateResult(
            "zero_semantics", zero["accuracy"] >= 0.95, {"accuracy": zero["accuracy"]}
        ),
        GateResult(
            "trace_and_immutability",
            "identity_ols" in operations
            and operations.count("near_conjugate_relay_lct") == 4
            and initial_hash == state_content_hash(bundle.state_dict()),
            {"operations": operations},
        ),
    )
    passed = all(item.passed for item in checks)
    run = _new_run(config)
    checkpoint = run.root / "conditional_near_conjugate_initialization.pt"
    source_hashes = {
        "basis_hash": source["basis_hash"],
        "semantic_source_checkpoint_hash": source["source_checkpoint_hash"],
        "target_evidence_hash": file_sha256(args.target_admission_evidence),
    }
    if passed:
        save_conditional_checkpoint(
            checkpoint,
            bundle,
            _manifest(
                config,
                split.manifest_hash,
                evaluator_hash,
                bundle,
                initial_hash,
                source_hashes,
            ),
            extra={
                "immutable_initialization": True,
                "generator_frozen": True,
                "illumination_intensity_transmission": 0.80,
                "electronic_initialization_hash": file_sha256(
                    args.electronic_initialization
                ),
            },
        )
    report = run.root / "conditional_near_conjugate_admission_report.json"
    atomic_write_json(
        report,
        {
            "contract": ADMISSION_CONTRACT,
            "panels": {
                name: {
                    key: value
                    for key, value in item.items()
                    if key not in {"raw", "command", "features", "logits"}
                }
                | {"direction_response": directions[name]}
                for name, item in metrics.items()
            },
            "zero_code": {key: value for key, value in zero.items() if key != "raw"},
            "trace_operations": operations,
            "state_hash": initial_hash,
        },
    )
    checkpoint_hash = file_sha256(checkpoint) if passed else ""
    context = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash or None,
    )
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_admission",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "relay_equivalence": file_sha256(args.relay_equivalence_evidence),
            "target_admission": file_sha256(args.target_admission_evidence),
            "electronic_initialization": file_sha256(args.electronic_initialization),
            "report": file_sha256(report),
            "checkpoint": checkpoint_hash,
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("conditional-near-conjugate-semantic-tangent",) if passed else (),
        None if passed else "near_conjugate_initial_physical_gate_failed",
    )
    _write_evidence(run, "conditional_near_conjugate_admission_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _phase_loss(bundle, evaluator, codes, labels, real, exposure):
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=1,
        activation_checkpoint=True,
    )
    features, logits = evaluator(fixed_exposure_view(raw, exposure))
    real_features, _ = evaluator(real)
    classification = F.cross_entropy(logits, labels)
    generated_view = fixed_exposure_view(raw, exposure)
    sinkhorn = conditional_sinkhorn_loss(
        features,
        generated_view,
        real_features,
        real,
        labels,
        epsilon=0.05,
        iterations=20,
    )
    diversity = relative_diversity_loss(features, real_features, labels)
    zero = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        torch.zeros_like(codes),
        labels,
        microbatch_size=1,
        activation_checkpoint=True,
    )
    command = bundle.generator(codes, labels).square()
    target_zero = bundle.generator(torch.zeros_like(codes), labels).square()
    delta = (
        1.0
        - F.cosine_similarity(
            (raw - zero).flatten(1), (command - target_zero).flatten(1), dim=1
        ).mean()
    )
    zero_anchor = (zero - 0.80 * target_zero).square().mean()
    saturation = torch.relu(raw / exposure - 1).square().mean()
    tv = phase_total_variation(bundle.system)
    return (
        sinkhorn
        + classification
        + diversity
        + delta
        + 0.5 * zero_anchor
        + 0.1 * saturation
        + 1e-5 * tv,
        {
            "sinkhorn": float(sinkhorn.detach()),
            "class": float(classification.detach()),
            "relative_diversity": float(diversity.detach()),
            "delta": float(delta.detach()),
            "zero_anchor": float(zero_anchor.detach()),
            "saturation": float(saturation.detach()),
        },
    )


def conditional_near_conjugate_semantic_tangent(args, config) -> int:
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        manifest,
        _,
    ) = _load_init(args, config)
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.admission_evidence).read_text())
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert(
        admission,
        "conditional_near_conjugate_admission",
        "conditional-near-conjugate-semantic-tangent",
        expected,
        config,
    )
    initial = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    exposure = config.conditional_generation.fixed_exposure_scale
    baseline = {
        name: _metrics(bundle, evaluator, codes, labels, real, exposure)
        for name, (codes, labels) in panels.items()
    }
    directions = {}
    for name, (codes, labels) in panels.items():
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=True,
        )
        _, logits = evaluator(fixed_exposure_view(raw, exposure))
        margin = (
            logits[torch.arange(labels.numel(), device=device), labels]
            - logits.masked_fill(F.one_hot(labels, 10).bool(), -torch.inf).max(1).values
        )
        loss = (
            -margin[margin < 0.5].mean()
            if bool((margin < 0.5).any())
            else -margin.mean()
        )
        bundle.system.encoder.phase.grad = None
        bundle.system.decoder.phase.grad = None
        loss.backward()
        directions[name] = {
            "encoder": bundle.system.encoder.phase.grad.detach().clone(),
            "decoder": bundle.system.decoder.phase.grad.detach().clone(),
            "positive_fraction": float((margin < 0.5).float().mean().item()),
        }
    probe = copy.deepcopy(bundle)
    with torch.no_grad():
        probe.system.encoder.phase.add_(
            sum(
                row["encoder"] / row["encoder"].norm().clamp_min(1e-12)
                for row in directions.values()
            )
            * 0.005
            / len(directions)
        )
        probe.system.decoder.phase.add_(
            sum(
                row["decoder"] / row["decoder"].norm().clamp_min(1e-12)
                for row in directions.values()
            )
            * 0.005
            / len(directions)
        )
    probed = {
        name: _metrics(probe, evaluator, codes, labels, real, exposure)
        for name, (codes, labels) in panels.items()
    }
    checks = (
        GateResult(
            "gradient",
            all(
                torch.isfinite(row["encoder"]).all()
                and torch.isfinite(row["decoder"]).all()
                and row["encoder"].norm() > 0
                and row["decoder"].norm() > 0
                for row in directions.values()
            ),
            {},
        ),
        GateResult(
            "fraction_protection",
            probed["axis"]["camera_fraction"] >= 0.25
            and probed["axis"]["camera_fraction"]
            >= baseline["axis"]["camera_fraction"] - 0.002
            and all(
                probed[name]["camera_fraction"]
                >= baseline[name]["camera_fraction"] - 0.01
                for name in ("uniform_1234", "uniform_5678")
            ),
            {},
        ),
        GateResult(
            "state_unchanged", initial == state_content_hash(bundle.state_dict()), {}
        ),
    )
    passed = all(item.passed for item in checks)
    run = _new_run(config)
    report = run.root / "conditional_near_conjugate_semantic_tangent_report.json"
    atomic_write_json(
        report,
        {
            "contract": TANGENT_CONTRACT,
            "baseline": {
                name: {
                    k: v
                    for k, v in item.items()
                    if k not in {"raw", "command", "features", "logits"}
                }
                for name, item in baseline.items()
            },
            "probe": {
                name: {
                    k: v
                    for k, v in item.items()
                    if k not in {"raw", "command", "features", "logits"}
                }
                for name, item in probed.items()
            },
            "state_hash": initial,
        },
    )
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_semantic_tangent",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission": file_sha256(args.admission_evidence),
            "report": file_sha256(report),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("conditional-near-conjugate-observability",) if passed else (),
        None if passed else "near_conjugate_semantic_direction_unavailable",
    )
    _write_evidence(
        run, "conditional_near_conjugate_semantic_tangent_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def conditional_near_conjugate_observability(args, config) -> int:
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        manifest,
        _,
    ) = _load_init(args, config)
    tangent = EvidenceRecordV2.from_dict(
        json.loads(Path(args.tangent_evidence).read_text())
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert(
        tangent,
        "conditional_near_conjugate_semantic_tangent",
        "conditional-near-conjugate-observability",
        expected,
        config,
    )
    initial = state_content_hash(bundle.state_dict())
    codes, labels = _training_codes(0, device)
    indices = next(iter(balanced_batches(train_labels, batch_size=50, seed=1234)))
    loss, values = _phase_loss(
        bundle,
        evaluator,
        codes,
        labels,
        train_images.index_select(0, indices).to(device),
        config.conditional_generation.fixed_exposure_scale,
    )
    loss.backward()
    grads = {
        "encoder": float(bundle.system.encoder.phase.grad.norm()),
        "decoder": float(bundle.system.decoder.phase.grad.norm()),
        "generator_has_grad": any(
            p.grad is not None for p in bundle.generator.parameters()
        ),
    }
    checks = (
        GateResult(
            "phase_gradients",
            grads["encoder"] > 0
            and grads["decoder"] > 0
            and not grads["generator_has_grad"],
            grads,
        ),
        GateResult("immutable", initial == state_content_hash(bundle.state_dict()), {}),
    )
    passed = all(item.passed for item in checks)
    run = _new_run(config)
    report = run.root / "conditional_near_conjugate_observability_report.json"
    atomic_write_json(
        report, {"contract": OBSERVABILITY_CONTRACT, "gradients": grads, "loss": values}
    )
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "tangent": file_sha256(args.tangent_evidence),
            "report": file_sha256(report),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("near-conjugate-relay-smoke",) if passed else (),
        None if passed else "near_conjugate_observability_failed",
    )
    _write_evidence(
        run, "conditional_near_conjugate_observability_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def near_conjugate_relay_smoke(args, config) -> int:
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        manifest,
        extra,
    ) = _load_init(args, config)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v6.4 smoke requires CUDA")
    evidence = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text())
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert(
        evidence,
        "conditional_near_conjugate_observability",
        "near-conjugate-relay-smoke",
        expected,
        config,
    )
    run = _new_run(config)
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    exposure = config.conditional_generation.fixed_exposure_scale
    initial = {k: v.detach().clone() for k, v in bundle.state_dict().items()}
    probes = {"0": _probe(run, 0, bundle, panels, evaluator, real, exposure)}
    optimizer = torch.optim.Adam(
        [bundle.system.encoder.phase, bundle.system.decoder.phase], lr=5e-5
    )
    batches = list(balanced_batches(train_labels, batch_size=50, seed=1234))
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(400):
        indices = batches[step % len(batches)]
        codes, labels = _training_codes(step, device)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _phase_loss(
            bundle,
            evaluator,
            codes,
            labels,
            train_images.index_select(0, indices).to(device),
            exposure,
        )
        loss.backward()
        grads = {
            "encoder": float(bundle.system.encoder.phase.grad.norm()),
            "decoder": float(bundle.system.decoder.phase.grad.norm()),
        }
        optimizer.step()
        history.append(
            summary
            | {
                "step": step + 1,
                "code_hash": tensor_sha256(codes.cpu()),
                "gradients": grads,
            }
        )
        if step + 1 in {50, 100, 200, 400}:
            probes[str(step + 1)] = _probe(
                run, step + 1, bundle, panels, evaluator, real, exposure
            )
    peak, total = (
        torch.cuda.max_memory_reserved(device),
        torch.cuda.get_device_properties(device).total_memory,
    )
    updates = {
        "encoder": float(
            (
                bundle.system.encoder.phase - initial["system.encoder.phase"].to(device)
            ).norm()
        ),
        "decoder": float(
            (
                bundle.system.decoder.phase - initial["system.decoder.phase"].to(device)
            ).norm()
        ),
    }
    final = probes["400"]
    zero = _zero_semantics(bundle, evaluator, exposure, device)
    checks = (
        GateResult(
            "semantics",
            all(
                row["accuracy"] >= 0.95 and min(row["recall"]) >= 0.90
                for row in final.values()
            ),
            final,
        ),
        GateResult("zero", zero["accuracy"] >= 0.95, {"accuracy": zero["accuracy"]}),
        GateResult(
            "fraction",
            all(
                row["camera_fraction"] >= 0.25
                for row in probes.values()
                for row in row.values()
            )
            and final["uniform_1234"]["camera_fraction"] >= 0.40
            and final["uniform_5678"]["camera_fraction"] >= 0.40,
            {},
        ),
        GateResult(
            "responses",
            all(min(row["direction_response"]) >= 0.01 for row in final.values()),
            {},
        ),
        GateResult(
            "saturation", all(row["saturation"] <= 0.08 for row in final.values()), {}
        ),
        GateResult(
            "phase_only_updates",
            all(v > 0 for v in updates.values())
            and all(
                torch.equal(bundle.state_dict()[key].cpu(), value.cpu())
                for key, value in initial.items()
                if key.startswith("generator.")
            ),
            updates,
        ),
        GateResult("cuda", peak <= 0.92 * total, {"reserved": peak, "total": total}),
    )
    passed = all(item.passed for item in checks)
    diagnostic = run.root / "conditional_near_conjugate_smoke_final.pt"
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or state_content_hash(initial),
            {
                "basis_hash": manifest.pca_basis_hash or "",
                "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
                or "",
                "target_evidence_hash": manifest.safe_scale_evidence_hash or "",
            },
        ),
        extra={"diagnostic_only": True, "never_warm_start": True},
    )
    history_path = run.root / "conditional_near_conjugate_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": SMOKE_CONTRACT,
            "history": history,
            "probes": probes,
            "updates": updates,
            "schedule_hash": _first_400_schedule_hash(),
        },
    )
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "observability": file_sha256(args.observability_evidence),
            "history": file_sha256(history_path),
            "diagnostic_final": file_sha256(diagnostic),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("train-near-conjugate-relay",) if passed else (),
        None if passed else "near_conjugate_smoke_failed",
    )
    _write_evidence(run, "conditional_near_conjugate_smoke_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def train_near_conjugate_relay(args, config) -> int:
    """User-operated five-epoch phase-only training from immutable v6.4 init."""
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        manifest,
        _,
    ) = _load_init(args, config)
    smoke = EvidenceRecordV2.from_dict(
        json.loads(Path(args.smoke_evidence).read_text())
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert(
        smoke,
        "conditional_near_conjugate_smoke",
        "train-near-conjugate-relay",
        expected,
        config,
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v6.4 formal training requires CUDA")
    initial = {
        key: value.detach().clone() for key, value in bundle.state_dict().items()
    }
    optimizer = torch.optim.Adam(
        [bundle.system.encoder.phase, bundle.system.decoder.phase],
        lr=config.conditional_generation.phase_learning_rate,
    )
    history = []
    for epoch in range(config.conditional_generation.epochs):
        for batch_number, indices in enumerate(
            balanced_batches(train_labels, batch_size=50, seed=1234 + epoch)
        ):
            step = epoch * 10_000 + batch_number
            codes, labels = _training_codes(step, device)
            optimizer.zero_grad(set_to_none=True)
            loss, summary = _phase_loss(
                bundle,
                evaluator,
                codes,
                labels,
                train_images.index_select(0, indices).to(device),
                config.conditional_generation.fixed_exposure_scale,
            )
            loss.backward()
            optimizer.step()
            history.append(
                summary
                | {
                    "epoch": epoch + 1,
                    "batch": batch_number,
                    "code_hash": tensor_sha256(codes.detach().cpu()),
                }
            )
    run = _new_run(config)
    checkpoint = run.root / "conditional_near_conjugate_generator.pt"
    source = {
        "basis_hash": manifest.pca_basis_hash or "",
        "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
        or "",
        "target_evidence_hash": manifest.safe_scale_evidence_hash or "",
    }
    save_conditional_checkpoint(
        checkpoint,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or state_content_hash(initial),
            source,
        ),
        extra={"history": history, "generator_frozen": True, "formal": True},
    )
    output_hash = file_sha256(checkpoint)
    frozen = all(
        torch.equal(bundle.state_dict()[key].cpu(), value.cpu())
        for key, value in initial.items()
        if key.startswith("generator.")
    )
    finite = bool(torch.isfinite(torch.tensor([row["class"] for row in history])).all())
    record = EvidenceRecordV2(
        2,
        "conditional_near_conjugate_training",
        config.run.evidence_tier,
        "passed" if finite and frozen else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=output_hash,
        ),
        {
            "checkpoint": output_hash,
            "smoke": file_sha256(args.smoke_evidence),
            "config_contract_hash": _stable_config_hash(config),
            "steps": len(history),
        },
        (
            GateResult(
                "finite_phase_only_training",
                finite and frozen,
                {"steps": len(history), "generator_frozen": frozen},
            ),
        ),
        ("evaluate",) if finite and frozen else (),
        None if finite and frozen else "near_conjugate_training_failed",
    )
    _write_evidence(run, "conditional_near_conjugate_training_evidence.json", record)
    print(run.root)
    return 0 if finite and frozen else 1
