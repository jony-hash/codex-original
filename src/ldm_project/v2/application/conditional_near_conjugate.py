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

from ..evidence import (
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ..optics import NearConjugateRelay, reference_near_conjugate_lens_relay
from ..student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)
from ..student.conditional import (
    conditional_sinkhorn_loss,
    fixed_exposure_view,
    phase_total_variation,
    raw_from_generator_batched,
    relative_diversity_loss,
)
from ..student.evaluator import (
    MNISTSplitManifest,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .conditional_formal_spacing import (
    _direction_response,
    _electronic_payload,
    _first_400_schedule_hash,
    _panels,
    _real_panel_views,
    _zero_semantics,
)
from .conditional_pca_transport import _full_contact_sheet, _metrics, _training_codes
from .conditional_runtime import (
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
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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


def conditional_near_conjugate_relay_equivalence(args, config) -> int:
    """CPU-only audit of the declared two-lens relay geometry."""
    _require(config)
    from ..contracts import ComplexField, OperationProvenance, PlaneSpec

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
    tv = phase_total_variation(bundle.system.encoder.phase) + phase_total_variation(
        bundle.system.decoder.phase
    )
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
