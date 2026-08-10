"""Hardware-feasible 1.50 mm PCA self-imaging recovery (v6.3).

The v6.3 line is intentionally separate from the 0.25 mm near-field
diagnostic.  It explicitly imports only the content-addressed electronic
command state, resets all four phase surfaces to zero, and retains the formal
finite-crop 2+2 optical system.
"""

from __future__ import annotations

import json
import copy
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
from .pca_transport import (
    _full_contact_sheet,
    _metrics,
    _panels,
    _real_panel_views,
    _training_codes,
)
from .support import (
    ConditionalBundle,
    FORMAL_SPACING_ARCHITECTURE,
    FORMAL_SPACING_GENERATOR_CONTRACT,
    _context,
    _new_run,
    _split,
    _write_evidence,
    _system_signature,
)


FORMAL_SPACING_METRIC_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_self_imaging_metric_v1",
        "target": "kappa*A^2",
        "delta": "same_code_delta",
        "exposure": "global_fixed",
        "target_detached": True,
        "saturation": "mean(raw/exposure>1)",
    }
)
FORMAL_SPACING_ADMISSION_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_admission_v1",
        "layer_spacing_m": 1.50e-3,
        "phase_initialization": "zero",
        "source": "explicit_v6_2_electronic_state",
    }
)
FORMAL_SPACING_TANGENT_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_semantic_tangent_v1",
        "carrier_logit_rms_max": 0.02,
        "phase_rms_max": 0.005,
        "raw_relative_rms_target": 0.01,
        "optimizer": False,
    }
)
FORMAL_SPACING_OBSERVABILITY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_observability_v1",
        "losses": ["image", "delta", "class"],
        "optimizer": False,
    }
)
FORMAL_SPACING_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_self_imaging_smoke_v1",
        "steps": 400,
        "phase_warmup": 100,
        "probes": [0, 50, 100, 200, 400],
    }
)
FORMAL_SPACING_TRAIN_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_formal_spacing_self_imaging_train_v1",
        "epochs": 5,
        "phase_warmup": 100,
        "balanced_min_class_exhaustion": True,
    }
)
FORMAL_SPACING_LOWPASS_CONTRACT = canonical_sha256(
    {
        "kind": "fft_radial_hard_lowpass",
        "max_cycles_per_mm": 20.0,
        "applied_every_forward": True,
    }
)
FORMAL_SPACING_SCALE_CONTRACT = canonical_sha256(
    {
        "parameterization": ".05+.45*sigmoid(g)",
        "range": [0.05, 0.50],
        "per_class_per_direction": True,
    }
)
FORMAL_SPACING_ROUTE_CONTRACT = canonical_sha256(
    {
        "code": "tanh(z)",
        "code_dim": 8,
        "panels": "zero,+e,-e,+u,-u",
        "physical_spacing_m": 1.50e-3,
    }
)


def _stable_config_hash(config) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("run", None)
    return canonical_sha256(payload)


def _first_400_schedule_hash() -> str:
    rows = []
    for step in range(400):
        codes, labels = _training_codes(step, torch.device("cpu"))
        rows.append(
            {
                "step": step,
                "codes": tensor_sha256(codes),
                "labels": tensor_sha256(labels),
            }
        )
    return canonical_sha256(rows)


def _require(config) -> None:
    if (
        config.architecture != FORMAL_SPACING_ARCHITECTURE
        or config.conditional_generation is None
        or not config.system.is_formal
    ):
        raise ValueError(
            "this command requires formal conditional_pca_formal_spacing_self_imaging_v6_3"
        )


def _electronic_payload(path: str) -> dict:
    payload = safe_torch_load(path)
    required = {
        "architecture_kind",
        "carrier",
        "basis",
        "scale_logits",
        "scales",
        "source_checkpoint_hash",
        "basis_hash",
        "scale_hash",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("invalid explicit v6.2 electronic initialization")
    if payload["architecture_kind"] != "conditional_pca_self_imaging_transport_v6_2":
        raise ValueError(
            "v6.3 electronic source must be the passed v6.2 target artifact"
        )
    if (
        tuple(payload["carrier"].shape) != (10, 200, 200)
        or tuple(payload["basis"].shape) != (10, 8, 200, 200)
        or tuple(payload["scales"].shape) != (10, 8)
    ):
        raise ValueError("electronic source has incompatible PCA state shapes")
    if not bool(
        torch.isfinite(payload["carrier"]).all()
        and torch.isfinite(payload["basis"]).all()
        and torch.isfinite(payload["scales"]).all()
    ):
        raise ValueError("electronic source contains non-finite tensors")
    if not bool(((payload["scales"] >= 0.05) & (payload["scales"] <= 0.50)).all()):
        raise ValueError(
            "electronic source violates the per-class per-direction scale contract"
        )
    return payload


def _direction_response(raw: torch.Tensor) -> list[float]:
    panel = raw.view(10, 17, *raw.shape[1:])
    return [
        float((panel[:, 1 + axis] - panel[:, 9 + axis]).square().mean().sqrt().item())
        for axis in range(8)
    ]


def _zero_semantics(
    bundle, evaluator, exposure: float, device: torch.device
) -> dict[str, object]:
    labels = torch.arange(10, device=device)
    codes = torch.zeros(10, 8, device=device)
    with torch.no_grad():
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
        _, logits = evaluator(fixed_exposure_view(raw, exposure))
    prediction = logits.argmax(1)
    return {
        "raw": raw,
        "accuracy": float((prediction == labels).float().mean().item()),
        "recall": [
            float((prediction[labels == y] == y).float().mean().item())
            for y in range(10)
        ],
    }


def _kappa(bundle, device: torch.device) -> float:
    labels = torch.arange(10, device=device)
    codes = torch.zeros(10, 8, device=device)
    with torch.no_grad():
        command = bundle.generator(codes, labels)
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
    return float(
        (raw.flatten(1).sum(1) / command.square().flatten(1).sum(1).clamp_min(1e-8))
        .median()
        .item()
    )


def _manifest(
    config,
    split_hash: str,
    evaluator_hash: str,
    bundle,
    initialization_hash: str,
    *,
    source: dict[str, str],
) -> ConditionalCheckpointManifestV2:
    return ConditionalCheckpointManifestV2(
        2,
        FORMAL_SPACING_ARCHITECTURE,
        _system_signature(config),
        _stable_config_hash(config),
        split_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        FORMAL_SPACING_GENERATOR_CONTRACT,
        FORMAL_SPACING_METRIC_CONTRACT,
        FORMAL_SPACING_TRAIN_CONTRACT,
        1234,
        initialization_hash,
        config_contract_hash=_stable_config_hash(config),
        pca_basis_hash=source["basis_hash"],
        lowpass_contract_hash=FORMAL_SPACING_LOWPASS_CONTRACT,
        scale_contract_hash=FORMAL_SPACING_SCALE_CONTRACT,
        route_contract_hash=FORMAL_SPACING_ROUTE_CONTRACT,
        semantic_source_checkpoint_hash=source["semantic_source_checkpoint_hash"],
        safe_scale_evidence_hash=source["target_evidence_hash"],
        first_400_schedule_hash=_first_400_schedule_hash(),
    )


def _load_common(args, config):
    _require(config)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, all_labels, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("frozen evaluator MNIST split does not match local data")
    return train_images, train_labels, split, evaluator_hash, evaluator, device


def _load_checkpoint(args, config):
    train_images, train_labels, split, evaluator_hash, evaluator, device = _load_common(
        args, config
    )
    payload = safe_torch_load(args.checkpoint)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid v6.3 initialization checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != FORMAL_SPACING_ARCHITECTURE
        or manifest.system_signature != _system_signature(config)
        or manifest.config_hash != _stable_config_hash(config)
        or manifest.evaluator_checkpoint_hash != evaluator_hash
        or manifest.dataset_split_manifest_hash != split.manifest_hash
    ):
        raise ValueError("v6.3 checkpoint context mismatch")
    bundle = ConditionalBundle(config).to(device)
    load_conditional_checkpoint(args.checkpoint, bundle, manifest)
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


def _assert_stage(
    record: EvidenceRecordV2, *, kind: str, action: str, expected_context, config
) -> None:
    record.validate()
    if (
        record.evidence_kind != kind
        or record.status != "passed"
        or action not in record.permitted_next_actions
    ):
        raise ValueError(f"{kind} evidence does not authorize {action}")
    record.context.assert_matches(
        expected_context,
        fields=(
            "system_signature",
            "evaluator_checkpoint_hash",
            "dataset_split_manifest_hash",
            "checkpoint_hash",
            "exposure_scale",
        ),
    )
    if record.input_hashes.get("config_contract_hash") != _stable_config_hash(config):
        raise ValueError(
            "stage evidence has a different physical v6.3 configuration contract"
        )


def conditional_formal_spacing_admission(args, config) -> int:
    """Derive zero-phase formal v6.3 initialization from a passed electronic source."""
    _require(config)
    train_images, train_labels, split, evaluator_hash, evaluator, device = _load_common(
        args, config
    )
    source_evidence = EvidenceRecordV2.from_dict(
        json.loads(Path(args.target_admission_evidence).read_text(encoding="utf-8"))
    )
    source_evidence.validate()
    if (
        source_evidence.evidence_kind != "conditional_short_spacing_target_admission"
        or source_evidence.status != "passed"
    ):
        raise ValueError("v6.3 requires passed v6.2 electronic target evidence")
    if (
        source_evidence.context.evaluator_checkpoint_hash != evaluator_hash
        or source_evidence.context.dataset_split_manifest_hash != split.manifest_hash
    ):
        raise ValueError("electronic target evidence evaluator/split context mismatch")
    source = _electronic_payload(args.electronic_initialization)
    if source["source_checkpoint_hash"] != source_evidence.context.checkpoint_hash:
        raise ValueError(
            "electronic initialization does not bind the target evidence semantic source"
        )
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
    initial_hash = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    metrics = {
        name: _metrics(
            bundle,
            evaluator,
            codes,
            labels,
            real,
            config.conditional_generation.fixed_exposure_scale,
        )
        for name, (codes, labels) in panels.items()
    }
    directions = {
        name: _direction_response(value["raw"]) for name, value in metrics.items()
    }
    zero = _zero_semantics(
        bundle, evaluator, config.conditional_generation.fixed_exposure_scale, device
    )
    kappa = _kappa(bundle, device)
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
            "formal_1p50mm_system",
            config.system.layer_spacing_m == 1.50e-3 and config.system.is_formal,
            {},
        ),
        GateResult(
            "command_and_raw_code_fraction",
            all(value["camera_fraction"] >= 0.25 for value in metrics.values()),
            {name: value["camera_fraction"] for name, value in metrics.items()},
        ),
        GateResult(
            "actual_saturation",
            all(value["saturation"] <= 0.08 for value in metrics.values()),
            {name: value["saturation"] for name, value in metrics.items()},
        ),
        GateResult(
            "raw_direction_response",
            all(min(rows) >= 1e-8 for rows in directions.values()),
            directions,
        ),
        GateResult(
            "rank_eight",
            all(
                int(
                    torch.linalg.matrix_rank(
                        bundle.generator.lowpass_normalized_modes()[label].flatten(1)
                    ).item()
                )
                == 8
                for label in range(10)
            ),
            {},
        ),
        GateResult(
            "zero_code_semantics",
            zero["accuracy"] >= 0.70,
            {"accuracy": zero["accuracy"]},
        ),
        GateResult(
            "complete_trace",
            any("identity_ols" in item for item in operations),
            {"operations": len(operations)},
        ),
        GateResult(
            "finite_and_immutable",
            bool(torch.isfinite(torch.tensor(kappa)))
            and initial_hash == state_content_hash(bundle.state_dict()),
            {"kappa": kappa},
        ),
    )
    passed = all(item.passed for item in checks)
    run = _new_run(config)
    source_hashes = {
        "basis_hash": source["basis_hash"],
        "semantic_source_checkpoint_hash": source["source_checkpoint_hash"],
        "target_evidence_hash": file_sha256(args.target_admission_evidence),
    }
    checkpoint = run.root / "conditional_formal_spacing_initialization.pt"
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
                source=source_hashes,
            ),
            extra={
                "immutable_initialization": True,
                "kappa": kappa,
                "electronic_initialization_hash": file_sha256(
                    args.electronic_initialization
                ),
                "initial_scales": source["scales"],
                "phase_initialization": "zero",
            },
        )
    report = run.root / "conditional_formal_spacing_admission_report.json"
    atomic_write_json(
        report,
        {
            "contract": FORMAL_SPACING_ADMISSION_CONTRACT,
            "classification": "formal_spacing_semantic_alignment_required"
            if passed and min(value["accuracy"] for value in metrics.values()) < 0.90
            else "formal_spacing_semantics_ready",
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
            "kappa": kappa,
            "trace_operations": operations,
            "state_hash": initial_hash,
            "initialization_checkpoint": str(checkpoint) if passed else None,
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
        "conditional_formal_spacing_admission",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "target_admission": file_sha256(args.target_admission_evidence),
            "electronic_initialization": file_sha256(args.electronic_initialization),
            "report": file_sha256(report),
            "checkpoint": checkpoint_hash,
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("conditional-formal-spacing-semantic-tangent",) if passed else (),
        None if passed else "formal_spacing_physical_admission_failed",
    )
    _write_evidence(run, "conditional_formal_spacing_admission_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _panel_margin(raw, labels, evaluator, exposure):
    _, logits = evaluator(fixed_exposure_view(raw, exposure))
    selected = logits[torch.arange(labels.numel(), device=labels.device), labels]
    others = logits.masked_fill(F.one_hot(labels, 10).bool(), -torch.inf).max(1).values
    return logits, selected - others


def _probe_copy(
    bundle, directions, *, codes, labels, evaluator, exposure, baseline_raw
):
    probe = copy.deepcopy(bundle).eval()
    with torch.no_grad():
        probe.generator.class_carrier_logits.add_(directions["carrier"])
        probe.system.encoder.phase.add_(directions["encoder_phase"])
        probe.system.decoder.phase.add_(directions["decoder_phase"])
        raw = raw_from_generator_batched(
            probe.system,
            probe.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
    relative = float(
        (raw - baseline_raw)
        .square()
        .mean()
        .sqrt()
        .div(baseline_raw.square().mean().sqrt().clamp_min(1e-8))
        .item()
    )
    if relative > 0.01:
        scale = 0.01 / relative
        with torch.no_grad():
            probe.generator.class_carrier_logits.copy_(
                bundle.generator.class_carrier_logits + directions["carrier"] * scale
            )
            probe.system.encoder.phase.copy_(
                bundle.system.encoder.phase + directions["encoder_phase"] * scale
            )
            probe.system.decoder.phase.copy_(
                bundle.system.decoder.phase + directions["decoder_phase"] * scale
            )
            raw = raw_from_generator_batched(
                probe.system,
                probe.generator,
                codes,
                labels,
                microbatch_size=1,
                activation_checkpoint=False,
            )
        relative = float(
            (raw - baseline_raw)
            .square()
            .mean()
            .sqrt()
            .div(baseline_raw.square().mean().sqrt().clamp_min(1e-8))
            .item()
        )
    return probe, raw, relative


def conditional_formal_spacing_semantic_tangent(args, config) -> int:
    """Read-only local semantic direction check for the formal 1.50 mm state."""
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        _,
        extra,
    ) = _load_checkpoint(args, config)
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.admission_evidence).read_text(encoding="utf-8"))
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert_stage(
        admission,
        kind="conditional_formal_spacing_admission",
        action="conditional-formal-spacing-semantic-tangent",
        expected_context=expected,
        config=config,
    )
    before = state_content_hash(bundle.state_dict())
    real = _real_panel_views(train_images, train_labels, device=device)
    panels = _panels(device)
    result = {}
    for name, (codes, labels) in panels.items():
        with torch.no_grad():
            baseline_raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes,
                labels,
                microbatch_size=1,
                activation_checkpoint=False,
            )
            baseline_logits, baseline_margin = _panel_margin(
                baseline_raw,
                labels,
                evaluator,
                config.conditional_generation.fixed_exposure_scale,
            )
        low = baseline_margin < 0.5
        if not bool(low.any()):
            low = torch.ones_like(low, dtype=torch.bool)
        for parameter in bundle.parameters():
            parameter.requires_grad_(False)
        groups = {
            "carrier": bundle.generator.class_carrier_logits,
            "encoder_phase": bundle.system.encoder.phase,
            "decoder_phase": bundle.system.decoder.phase,
        }
        for parameter in groups.values():
            parameter.requires_grad_(True)
        # Stream by class: each graph is released after its class margin VJP.
        aggregate = {key: torch.zeros_like(value) for key, value in groups.items()}
        for label in range(10):
            selected = torch.nonzero((labels == label) & low, as_tuple=False).flatten()
            if not selected.numel():
                continue
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes.index_select(0, selected),
                labels.index_select(0, selected),
                microbatch_size=1,
                activation_checkpoint=True,
            )
            _, margin = _panel_margin(
                raw,
                labels.index_select(0, selected),
                evaluator,
                config.conditional_generation.fixed_exposure_scale,
            )
            gradients = torch.autograd.grad(
                margin.mean(), tuple(groups.values()), allow_unused=True
            )
            for key, gradient in zip(groups, gradients):
                if gradient is not None:
                    aggregate[key] += gradient.detach()
            del raw, gradients
            if device.type == "cuda":
                torch.cuda.empty_cache()
        directions = {}
        caps = {"carrier": 0.02, "encoder_phase": 0.005, "decoder_phase": 0.005}
        for key, gradient in aggregate.items():
            rms = gradient.square().mean().sqrt().clamp_min(1e-12)
            directions[key] = gradient * (caps[key] / rms)
        for parameter in groups.values():
            parameter.requires_grad_(False)
        temporary, raw, raw_relative = _probe_copy(
            bundle,
            directions,
            codes=codes,
            labels=labels,
            evaluator=evaluator,
            exposure=config.conditional_generation.fixed_exposure_scale,
            baseline_raw=baseline_raw,
        )
        probe = _metrics(
            temporary,
            evaluator,
            codes,
            labels,
            real,
            config.conditional_generation.fixed_exposure_scale,
        )
        probe_logits, probe_margin = _panel_margin(
            probe["raw"],
            labels,
            evaluator,
            config.conditional_generation.fixed_exposure_scale,
        )
        result[name] = {
            "baseline": {
                key: value
                for key, value in _metrics(
                    bundle,
                    evaluator,
                    codes,
                    labels,
                    real,
                    config.conditional_generation.fixed_exposure_scale,
                ).items()
                if key not in {"raw", "command", "features", "logits"}
            },
            "low_margin_count": int(low.sum().item()),
            "positive_fraction": float(
                (probe_margin[low] > baseline_margin[low]).float().mean().item()
            ),
            "median_margin_delta": float(
                (probe_margin[low] - baseline_margin[low]).median().item()
            ),
            "correct_count_delta": int(
                (probe_logits.argmax(1) == labels).sum().item()
                - (baseline_logits.argmax(1) == labels).sum().item()
            ),
            "probe": {
                key: value
                for key, value in probe.items()
                if key not in {"raw", "command", "features", "logits"}
            },
            "raw_relative_rms": raw_relative,
            "gradient_norms": {
                key: float(value.norm().item()) for key, value in aggregate.items()
            },
        }
    after = state_content_hash(bundle.state_dict())

    def ok(name):
        row = result[name]
        threshold = 0.80
        return (
            row["positive_fraction"] >= threshold
            and row["median_margin_delta"] >= 0.01
            and row["correct_count_delta"] >= 0
            and row["probe"]["camera_fraction"]
            >= row["baseline"]["camera_fraction"] - 0.01
            and row["probe"]["saturation"] <= 0.08
        )

    passed = after == before and all(ok(name) for name in result)
    run = _new_run(config)
    report = run.root / "conditional_formal_spacing_semantic_tangent_report.json"
    atomic_write_json(
        report,
        {
            "contract": FORMAL_SPACING_TANGENT_CONTRACT,
            "classification": "formal_spacing_semantic_direction_available"
            if passed
            else "formal_spacing_semantic_direction_unavailable",
            "panels": result,
            "state_before": before,
            "state_after": after,
            "optimizer_created": False,
        },
    )
    checks = (
        GateResult("state_unchanged", before == after, {}),
        GateResult("joint_semantic_direction", passed, result),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_formal_spacing_semantic_tangent",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission": file_sha256(args.admission_evidence),
            "report": file_sha256(report),
            "checkpoint": file_sha256(args.checkpoint),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("conditional-formal-spacing-observability",) if passed else (),
        None if passed else "formal_spacing_semantic_direction_unavailable",
    )
    _write_evidence(
        run, "conditional_formal_spacing_semantic_tangent_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def _self_objective(
    bundle,
    evaluator,
    codes,
    labels,
    real_views,
    *,
    kappa: float,
    initial_scales,
    config,
):
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=1,
        activation_checkpoint=True,
    )
    command = bundle.generator(codes, labels)
    target = (kappa * command.square()).detach()
    view = fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    image = (raw - target).abs().mean()
    panel_raw, panel_target = (
        raw.view(10, 5, *raw.shape[1:]),
        target.view(10, 5, *target.shape[1:]),
    )
    delta = (
        (
            (panel_raw[:, 1:] - panel_raw[:, :1])
            - (panel_target[:, 1:] - panel_target[:, :1])
        )
        .abs()
        .mean()
    )
    sinkhorn = conditional_sinkhorn_loss(
        features,
        view,
        real_features,
        real_views,
        labels,
        epsilon=config.conditional_generation.sinkhorn_epsilon,
        iterations=config.conditional_generation.sinkhorn_iterations,
    )
    classification = F.cross_entropy(logits, labels)
    diversity = relative_diversity_loss(features, real_features, labels)
    saturation = (
        torch.relu(raw / config.conditional_generation.fixed_exposure_scale - 1)
        .square()
        .mean()
    )
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    phase_tv = phase_total_variation(bundle.system)
    anchor = (bundle.generator.scales() - initial_scales.to(raw.device)).square().mean()
    total = (
        image
        + delta
        + sinkhorn
        + classification
        + diversity
        + 0.1 * saturation
        + 1e-3 * orthogonality
        + 1e-5 * phase_tv
        + 1e-2 * anchor
    )
    return total, {
        "total": float(total.detach()),
        "image": float(image.detach()),
        "delta": float(delta.detach()),
        "sinkhorn": float(sinkhorn.detach()),
        "class": float(classification.detach()),
        "relative_diversity": float(diversity.detach()),
        "saturation": float(saturation.detach()),
        "orthogonality": float(orthogonality.detach()),
        "phase_tv": float(phase_tv.detach()),
        "scale_anchor": float(anchor.detach()),
    }


def conditional_formal_spacing_observability(args, config) -> int:
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        device,
        bundle,
        _,
        extra,
    ) = _load_checkpoint(args, config)
    tangent = EvidenceRecordV2.from_dict(
        json.loads(Path(args.tangent_evidence).read_text(encoding="utf-8"))
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert_stage(
        tangent,
        kind="conditional_formal_spacing_semantic_tangent",
        action="conditional-formal-spacing-observability",
        expected_context=expected,
        config=config,
    )
    before = state_content_hash(bundle.state_dict())
    indices = next(balanced_batches(train_labels, batch_size=50, seed=1234))
    codes, labels = _training_codes(0, device)
    real = train_images.index_select(0, indices).to(device)
    for parameter in bundle.parameters():
        parameter.requires_grad_(True)
    loss, summary = _self_objective(
        bundle,
        evaluator,
        codes,
        labels,
        real,
        kappa=float(extra["kappa"]),
        initial_scales=extra["initial_scales"],
        config=config,
    )
    groups = {
        "carrier": bundle.generator.class_carrier_logits,
        "modes": bundle.generator.class_modes,
        "scales": bundle.generator.scale_logits,
        "encoder_phase": bundle.system.encoder.phase,
        "decoder_phase": bundle.system.decoder.phase,
    }
    gradients = torch.autograd.grad(loss, tuple(groups.values()), allow_unused=True)
    norms = {
        name: float(value.norm().item()) if value is not None else 0.0
        for name, value in zip(groups, gradients)
    }
    after = state_content_hash(bundle.state_dict())
    passed = after == before and all(
        torch.isfinite(torch.tensor(value)) and value > 1e-12
        for value in norms.values()
    )
    run = _new_run(config)
    report = run.root / "conditional_formal_spacing_observability_report.json"
    atomic_write_json(
        report,
        {
            "contract": FORMAL_SPACING_OBSERVABILITY_CONTRACT,
            "loss": summary,
            "gradient_norms": norms,
            "target_detached": True,
            "state_before": before,
            "state_after": after,
            "optimizer_created": False,
        },
    )
    checks = (
        GateResult(
            "five_group_gradients",
            all(value > 1e-12 for value in norms.values()),
            norms,
        ),
        GateResult("state_unchanged", before == after, {}),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_formal_spacing_observability",
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
        ("formal-spacing-self-imaging-smoke",) if passed else (),
        None if passed else "formal_spacing_observability_failed",
    )
    _write_evidence(
        run, "conditional_formal_spacing_observability_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def _probe(run, step, bundle, panels, evaluator, real, exposure):
    values = {}
    for name, (codes, labels) in panels.items():
        metric = _metrics(bundle, evaluator, codes, labels, real, exposure)
        raw_path = run.root / f"formal_spacing_{name}_step_{step}_raw.pt"
        atomic_torch_save(raw_path, metric["raw"].detach().cpu())
        _full_contact_sheet(
            run.root / f"formal_spacing_{name}_step_{step}_10x17.png",
            metric["raw"],
            labels,
            exposure,
        )
        metric["direction_response"] = _direction_response(metric["raw"])
        values[name] = {
            key: value
            for key, value in metric.items()
            if key not in {"raw", "command", "features", "logits"}
        }
    return values


def formal_spacing_self_imaging_smoke(args, config) -> int:
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
    ) = _load_checkpoint(args, config)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v6.3 smoke requires CUDA")
    observability = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert_stage(
        observability,
        kind="conditional_formal_spacing_observability",
        action="formal-spacing-self-imaging-smoke",
        expected_context=expected,
        config=config,
    )
    torch.cuda.reset_peak_memory_stats(device)
    run = _new_run(config)
    panels = _panels(device)
    real = _real_panel_views(train_images, train_labels, device=device)
    initial = {
        key: value.detach().clone() for key, value in bundle.state_dict().items()
    }
    probes = {
        "0": _probe(
            run,
            0,
            bundle,
            panels,
            evaluator,
            real,
            config.conditional_generation.fixed_exposure_scale,
        )
    }
    history = []
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    bundle.generator.class_carrier_logits,
                    bundle.generator.class_modes,
                    bundle.generator.scale_logits,
                ],
                "lr": 2e-4,
            },
            {
                "params": [bundle.system.encoder.phase, bundle.system.decoder.phase],
                "lr": 5e-5,
            },
        ]
    )
    batches = list(balanced_batches(train_labels, batch_size=50, seed=1234))
    for step in range(400):
        generator_trainable = step >= 100
        for parameter in (
            bundle.generator.class_carrier_logits,
            bundle.generator.class_modes,
            bundle.generator.scale_logits,
        ):
            parameter.requires_grad_(generator_trainable)
        for parameter in (bundle.system.encoder.phase, bundle.system.decoder.phase):
            parameter.requires_grad_(True)
        indices = batches[step % len(batches)]
        codes, labels = _training_codes(step, device)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _self_objective(
            bundle,
            evaluator,
            codes,
            labels,
            train_images.index_select(0, indices).to(device),
            kappa=float(extra["kappa"]),
            initial_scales=extra["initial_scales"],
            config=config,
        )
        loss.backward()
        gradients = {
            "carrier": float(bundle.generator.class_carrier_logits.grad.norm().item())
            if bundle.generator.class_carrier_logits.grad is not None
            else 0.0,
            "modes": float(bundle.generator.class_modes.grad.norm().item())
            if bundle.generator.class_modes.grad is not None
            else 0.0,
            "scales": float(bundle.generator.scale_logits.grad.norm().item())
            if bundle.generator.scale_logits.grad is not None
            else 0.0,
            "encoder_phase": float(bundle.system.encoder.phase.grad.norm().item()),
            "decoder_phase": float(bundle.system.decoder.phase.grad.norm().item()),
        }
        optimizer.step()
        history.append(
            summary
            | {
                "step": step + 1,
                "code_hash": tensor_sha256(codes.detach().cpu()),
                "gradients": gradients,
                "phase_only": not generator_trainable,
            }
        )
        if step + 1 in {50, 100, 200, 400}:
            probes[str(step + 1)] = _probe(
                run,
                step + 1,
                bundle,
                panels,
                evaluator,
                real,
                config.conditional_generation.fixed_exposure_scale,
            )
    peak, total_memory = (
        torch.cuda.max_memory_reserved(device),
        torch.cuda.get_device_properties(device).total_memory,
    )
    updates = {
        key: float((value - initial[key].to(value.device)).norm().item())
        for key, value in bundle.state_dict().items()
        if key
        in {
            "generator.class_carrier_logits",
            "generator.class_modes",
            "generator.scale_logits",
            "system.encoder.phase",
            "system.decoder.phase",
        }
    }
    final = probes["400"]
    zero = _zero_semantics(
        bundle, evaluator, config.conditional_generation.fixed_exposure_scale, device
    )
    checks = (
        GateResult(
            "semantic",
            all(
                value["accuracy"] >= 0.90 and min(value["recall"]) >= 0.80
                for value in final.values()
            ),
            final,
        ),
        GateResult(
            "zero_code_accuracy",
            zero["accuracy"] >= 0.95,
            {"accuracy": zero["accuracy"]},
        ),
        GateResult(
            "camera_fraction",
            all(value["camera_fraction"] >= 0.25 for value in final.values()),
            {name: value["camera_fraction"] for name, value in final.items()},
        ),
        GateResult(
            "intermediate_camera_floor",
            all(
                value["camera_fraction"] >= 0.20
                for step, panel in probes.items()
                if step != "0"
                for value in panel.values()
            ),
            {},
        ),
        GateResult(
            "diversity_floor",
            all(min(value["ratios"]) >= 0.35 for value in final.values()),
            {},
        ),
        GateResult(
            "direction_response",
            all(min(value["direction_response"]) >= 0.01 for value in final.values()),
            {},
        ),
        GateResult(
            "saturation",
            all(value["saturation"] <= 0.08 for value in final.values()),
            {},
        ),
        GateResult(
            "five_updates", all(value > 0 for value in updates.values()), updates
        ),
        GateResult(
            "cuda_memory",
            peak <= 0.92 * total_memory,
            {"reserved": peak, "total": total_memory},
        ),
    )
    diagnostic = run.root / "conditional_formal_spacing_smoke_final.pt"
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or state_content_hash(initial),
            source={
                "basis_hash": manifest.pca_basis_hash or "",
                "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
                or "",
                "target_evidence_hash": manifest.safe_scale_evidence_hash or "",
            },
        ),
        extra={
            "diagnostic_only": True,
            "never_warm_start": True,
            "contract": FORMAL_SPACING_SMOKE_CONTRACT,
        },
    )
    history_path = run.root / "conditional_formal_spacing_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": FORMAL_SPACING_SMOKE_CONTRACT,
            "history": history,
            "probes": probes,
            "updates": updates,
            "schedule_hash": _first_400_schedule_hash(),
        },
    )
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_formal_spacing_smoke",
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
        ("train-formal-spacing-self-imaging",) if passed else (),
        None if passed else "formal_spacing_smoke_failed",
    )
    _write_evidence(run, "conditional_formal_spacing_smoke_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def train_formal_spacing_self_imaging(args, config) -> int:
    """User-invoked five-epoch formal recovery, always from immutable v6.3 init."""
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
    ) = _load_checkpoint(args, config)
    smoke = EvidenceRecordV2.from_dict(
        json.loads(Path(args.smoke_evidence).read_text(encoding="utf-8"))
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _assert_stage(
        smoke,
        kind="conditional_formal_spacing_smoke",
        action="train-formal-spacing-self-imaging",
        expected_context=expected,
        config=config,
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    bundle.generator.class_carrier_logits,
                    bundle.generator.class_modes,
                    bundle.generator.scale_logits,
                ],
                "lr": 2e-4,
            },
            {
                "params": [bundle.system.encoder.phase, bundle.system.decoder.phase],
                "lr": 5e-5,
            },
        ]
    )
    history = []
    global_step = 0
    for epoch in range(5):
        for indices in balanced_batches(train_labels, batch_size=50, seed=1234 + epoch):
            trainable = global_step >= 100
            for parameter in (
                bundle.generator.class_carrier_logits,
                bundle.generator.class_modes,
                bundle.generator.scale_logits,
            ):
                parameter.requires_grad_(trainable)
            for parameter in (bundle.system.encoder.phase, bundle.system.decoder.phase):
                parameter.requires_grad_(True)
            codes, labels = _training_codes(global_step, device)
            optimizer.zero_grad(set_to_none=True)
            loss, summary = _self_objective(
                bundle,
                evaluator,
                codes,
                labels,
                train_images.index_select(0, indices).to(device),
                kappa=float(extra["kappa"]),
                initial_scales=extra["initial_scales"],
                config=config,
            )
            loss.backward()
            optimizer.step()
            history.append(
                summary
                | {
                    "epoch": epoch,
                    "step": global_step + 1,
                    "code_hash": tensor_sha256(codes.detach().cpu()),
                }
            )
            global_step += 1
    panels, real = (
        _panels(device),
        _real_panel_views(train_images, train_labels, device=device),
    )
    final = {
        name: _metrics(
            bundle,
            evaluator,
            codes,
            labels,
            real,
            config.conditional_generation.fixed_exposure_scale,
        )
        for name, (codes, labels) in panels.items()
    }
    zero = _zero_semantics(
        bundle, evaluator, config.conditional_generation.fixed_exposure_scale, device
    )
    checks = (
        GateResult(
            "semantic",
            all(
                value["accuracy"] >= 0.95 and min(value["recall"]) >= 0.90
                for value in final.values()
            ),
            {},
        ),
        GateResult(
            "zero_code", zero["accuracy"] >= 0.95, {"accuracy": zero["accuracy"]}
        ),
        GateResult(
            "camera_fraction",
            all(value["camera_fraction"] >= 0.25 for value in final.values()),
            {},
        ),
        GateResult(
            "diversity",
            all(
                all(0.70 <= ratio <= 1.30 for ratio in value["ratios"])
                for value in final.values()
            ),
            {},
        ),
        GateResult(
            "direction_response",
            all(
                min(_direction_response(value["raw"])) >= 0.01
                for value in final.values()
            ),
            {},
        ),
        GateResult(
            "saturation",
            all(value["saturation"] <= 0.08 for value in final.values()),
            {},
        ),
    )
    run = _new_run(config)
    output = run.root / "conditional_formal_spacing_generator.pt"
    save_conditional_checkpoint(
        output,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or manifest.state_content_hash,
            source={
                "basis_hash": manifest.pca_basis_hash or "",
                "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
                or "",
                "target_evidence_hash": manifest.safe_scale_evidence_hash or "",
            },
        ),
        extra={
            "smoke_evidence": smoke.evidence_hash,
            "history": history,
            "formal_epochs": 5,
        },
    )
    history_path = run.root / "conditional_formal_spacing_training_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": FORMAL_SPACING_TRAIN_CONTRACT,
            "steps": global_step,
            "history": history,
            "panels": {
                name: {
                    key: value
                    for key, value in item.items()
                    if key not in {"raw", "command", "features", "logits"}
                }
                for name, item in final.items()
            },
        },
    )
    passed = all(item.passed for item in checks)
    output_hash = file_sha256(output)
    record = EvidenceRecordV2(
        2,
        "conditional_formal_spacing_training",
        config.run.evidence_tier,
        "passed" if passed else "failed",
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
            "smoke": file_sha256(args.smoke_evidence),
            "checkpoint": output_hash,
            "history": file_sha256(history_path),
            "config_contract_hash": _stable_config_hash(config),
        },
        checks,
        ("evaluate",) if passed else (),
        None if passed else "formal_spacing_training_failed",
    )
    _write_evidence(run, "conditional_formal_spacing_training_evidence.json", record)
    print(run.root)
    return 0 if passed else 1
