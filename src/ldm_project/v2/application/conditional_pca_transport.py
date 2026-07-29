"""v6 PCA transport co-design gates and the single bounded recovery smoke.

This module intentionally keeps v6 separate from the historical generators.
It derives a new checkpoint only through explicit carrier/phase copying and a
deterministic PCA basis; no previous generator state is partially loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
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
from ..student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)
from ..student.conditional import (
    camera_code_fraction,
    camera_code_fraction_loss,
    conditional_sinkhorn_loss,
    fixed_exposure_view,
    raw_from_generator_batched,
    relative_diversity_loss,
    relative_diversity_ratios,
    phase_total_variation,
    sinkhorn_cost_view,
)
from ..student.evaluator import (
    MNISTSplitManifest,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .conditional_continuous_reachability import (
    EXPECTED_CONTINUOUS_BASIS_HASH,
    SAFE_PROBE_ALPHA,
    _ContinuousProbeGenerator,
    _build_modes,
    _full_contact_sheet,
)
from .conditional_runtime import (
    BALANCED_TRAINING_SCHEDULE_CONTRACT,
    CONDITIONAL_CONTROL_METRIC_CONTRACT,
    IDENTITY_ARCHITECTURE,
    PCA_TRANSPORT_ARCHITECTURE,
    PCA_TRANSPORT_GENERATOR_CONTRACT,
    ConditionalBundle,
    _context,
    _fixed_control_probe,
    _new_run,
    _random_codes,
    _split,
    _write_evidence,
)


PCA_TRANSPORT_ADMISSION_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_pca_transport_admission_v6",
        "initial_scale": SAFE_PROBE_ALPHA,
        "max_scale": 0.20,
        "l2_budget": 0.30,
        "lowpass_cycles_per_mm": 20.0,
    }
)
PCA_TRANSPORT_OBSERVABILITY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_pca_transport_observability_v6",
        "panels": ["axis", "uniform_1234", "uniform_5678"],
        "streamed_class_size": 17,
    }
)
PCA_TRANSPORT_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_pca_transport_smoke_v6",
        "steps": 400,
        "probes": [0, 50, 100, 200, 400],
    }
)
PCA_TRANSPORT_LOWPASS_CONTRACT = canonical_sha256(
    {
        "kind": "fft_radial_hard_lowpass",
        "max_cycles_per_mm": 20.0,
        "applied_every_forward": True,
    }
)
PCA_TRANSPORT_SCALE_CONTRACT = canonical_sha256(
    {
        "parameterization": "0.20*sigmoid(g)",
        "initial_axis_logit_rms": 0.05,
        "initial_scale": SAFE_PROBE_ALPHA,
        "l2_budget": 0.30,
    }
)
PCA_TRANSPORT_ROUTE_CONTRACT = canonical_sha256(
    {"code": "tanh(z)", "code_dim": 8, "panels": "zero,+e,-e,+u,-u"}
)


def _config_contract_hash(config) -> str:
    """Stable v6 contract: run metadata cannot invalidate a physical state."""
    payload = config.model_dump(mode="json")
    payload.pop("run", None)
    return canonical_sha256(payload)


def _first_400_schedule_hash() -> str:
    codes = []
    for step in range(400):
        value, labels = _training_codes(step, torch.device("cpu"))
        codes.append(
            {
                "step": step,
                "codes": tensor_sha256(value),
                "labels": tensor_sha256(labels),
            }
        )
    return canonical_sha256(codes)


def _require_stage_evidence(
    record: EvidenceRecordV2,
    *,
    kind: str,
    action: str,
    config,
    evaluator_hash: str,
    split_hash: str,
    checkpoint_hash: str,
) -> None:
    """Allow diagnostic→formal v6 handoff only across identical physics contracts."""
    record.validate()
    context = record.context
    if (
        record.evidence_kind != kind
        or record.status != "passed"
        or action not in record.permitted_next_actions
        or context.system_signature != _context(config).system_signature
        or context.evaluator_checkpoint_hash != evaluator_hash
        or context.dataset_split_manifest_hash != split_hash
        or context.checkpoint_hash != checkpoint_hash
        or record.input_hashes.get("config_contract_hash")
        != _config_contract_hash(config)
    ):
        raise ValueError(
            "v6 evidence is not bound to this physical/configuration contract"
        )


def _require(config) -> None:
    if (
        config.architecture != PCA_TRANSPORT_ARCHITECTURE
        or config.conditional_generation is None
    ):
        raise ValueError("this command requires conditional_pca_transport_codesign_v6")


def _panels(device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    axis_codes, labels = _fixed_control_probe(device=device)
    return {
        "axis": (axis_codes, labels),
        "uniform_1234": (_random_codes(170, device=device, seed=1234), labels),
        "uniform_5678": (_random_codes(170, device=device, seed=5678), labels),
    }


def _real_panel_views(
    train_images: torch.Tensor, train_labels: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Deterministic 10x17 real reference aligned with every control panel."""
    indices = []
    for label in range(10):
        selected = torch.nonzero(train_labels == label, as_tuple=False).flatten()
        if selected.numel() < 17:
            raise ValueError("fixed MNIST split lacks 17 examples for a label")
        indices.append(selected[:17])
    return train_images.index_select(0, torch.cat(indices)).to(device)


def _source_state(
    path: str, *, expected_system: str, evaluator_hash: str, split_hash: str
) -> tuple[dict, ConditionalCheckpointManifestV2]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state_dict"), dict)
        or not isinstance(payload.get("manifest"), dict)
    ):
        raise ValueError("invalid semantic source checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != IDENTITY_ARCHITECTURE
        or manifest.system_signature != expected_system
        or manifest.evaluator_checkpoint_hash != evaluator_hash
        or manifest.dataset_split_manifest_hash != split_hash
    ):
        raise ValueError("v6 requires the content-bound semantic identity checkpoint")
    return payload["state_dict"], manifest


def _metrics(
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    exposure: float,
    *,
    grad: bool = False,
) -> dict[str, object]:
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=grad,
        )
        command = bundle.generator(codes, labels)
        view = fixed_exposure_view(raw, exposure)
        features, logits = evaluator(view)
        prediction = logits.argmax(1)
        real_features, _ = evaluator(real_views)
        fraction = camera_code_fraction(raw, labels)
        ratios = relative_diversity_ratios(
            F.normalize(features, dim=1), F.normalize(real_features, dim=1), labels
        )
        saturation = torch.relu(raw / exposure - 1.0).gt(0).float().mean()
    return {
        "raw": raw,
        "command": command,
        "features": features,
        "logits": logits,
        "accuracy": float((prediction == labels).float().mean().item()),
        "recall": [
            float((prediction[labels == y] == y).float().mean().item())
            for y in range(10)
        ],
        "camera_fraction": float(fraction.item()),
        "ratios": [float(x) for x in ratios],
        "saturation": float(saturation.item()),
        "finite": bool(torch.isfinite(raw).all().item()),
    }


def _streamed_panel_gradient_norms(
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    groups: dict[str, torch.Tensor],
    exposure: float,
) -> dict[str, float]:
    """Exact panel ANOVA gradient without retaining 170 optical graphs.

    The first no-grad pass supplies the class/global means.  In the second
    pass those means may be treated as constants: the derivative terms from
    centering cancel within each class and globally, so the accumulated VJP is
    the same as a full-panel ANOVA derivative.
    """
    with torch.no_grad():
        raw_parts = [
            raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes[i * 17 : (i + 1) * 17],
                labels[i * 17 : (i + 1) * 17],
                microbatch_size=1,
                activation_checkpoint=False,
            )
            for i in range(10)
        ]
        baseline = torch.cat(raw_parts, dim=0)
        global_mean = baseline.mean(dim=0, keepdim=True)
        class_means = [item.mean(dim=0, keepdim=True) for item in raw_parts]
        denominator = (
            (baseline - global_mean).square().sum().detach().clamp_min(1.0e-12)
        )
    norms = {name: 0.0 for name in groups}
    for label in range(10):
        start, stop = label * 17, (label + 1) * 17
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes[start:stop],
            labels[start:stop],
            microbatch_size=1,
            activation_checkpoint=True,
        )
        _, logits = evaluator(fixed_exposure_view(raw, exposure))
        camera_term = (raw - class_means[label]).square().sum() / denominator
        objective = F.cross_entropy(logits, labels[start:stop]) - 0.1 * camera_term
        gradients = torch.autograd.grad(
            objective, tuple(groups.values()), allow_unused=True
        )
        for name, value in zip(groups, gradients):
            if value is not None:
                norms[name] += float(value.detach().norm().item())
        del raw, logits, gradients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return norms


def _manifest(
    config,
    split_hash: str,
    evaluator_hash: str,
    bundle: ConditionalBundle,
    init_hash: str,
    *,
    provenance: dict[str, str],
) -> ConditionalCheckpointManifestV2:
    return ConditionalCheckpointManifestV2(
        2,
        PCA_TRANSPORT_ARCHITECTURE,
        _context(config).system_signature,
        _config_contract_hash(config),
        split_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        PCA_TRANSPORT_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        init_hash,
        config_contract_hash=_config_contract_hash(config),
        pca_basis_hash=provenance["pca_basis_hash"],
        lowpass_contract_hash=PCA_TRANSPORT_LOWPASS_CONTRACT,
        scale_contract_hash=PCA_TRANSPORT_SCALE_CONTRACT,
        route_contract_hash=PCA_TRANSPORT_ROUTE_CONTRACT,
        semantic_source_checkpoint_hash=provenance["semantic_source_checkpoint_hash"],
        safe_scale_evidence_hash=provenance["safe_scale_evidence_hash"],
        first_400_schedule_hash=_first_400_schedule_hash(),
    )


def conditional_pca_transport_admission(args, config) -> int:
    """Derive the immutable v6 initial state from the safe-scale failure."""
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    evaluator, evaluator_payload = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device("cpu")
    )
    if (
        MNISTSplitManifest(**evaluator_payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("evaluator split mismatch")
    failure = EvidenceRecordV2.from_dict(
        json.loads(Path(args.safe_scale_evidence).read_text(encoding="utf-8"))
    )
    failure.validate()
    if (
        failure.evidence_kind != "conditional_continuous_mode_reachability"
        or failure.status != "failed"
        or failure.invalidation_reason != "safe_scale_transport_insufficient"
    ):
        raise ValueError(
            "v6 requires the immutable safe_scale_transport_insufficient evidence"
        )
    if (
        failure.context.system_signature != _context(config).system_signature
        or failure.context.evaluator_checkpoint_hash != evaluator_hash
        or failure.context.dataset_split_manifest_hash != split.manifest_hash
        or failure.context.checkpoint_hash != file_sha256(args.semantic_checkpoint)
    ):
        raise ValueError(
            "safe-scale evidence is not content-bound to the supplied semantic checkpoint"
        )
    source, source_manifest = _source_state(
        args.semantic_checkpoint,
        expected_system=_context(config).system_signature,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    basis = _build_modes(train_images, train_labels, evaluator, config)
    basis_hash = tensor_sha256(basis)
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "v6 physical admission requires CUDA; CPU is not an acceptance substitute"
        )
    evaluator = evaluator.to(device).eval()
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=source["generator.class_carrier_logits"],
            class_modes=basis,
        )
        .to(device)
        .eval()
    )
    target = bundle.state_dict()
    for name in ("system.encoder.phase", "system.decoder.phase"):
        target[name].copy_(source[name])
    bundle.load_state_dict(target, strict=True)
    before = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    real_views = _real_panel_views(train_images, train_labels, device=device)
    safe_reference = (
        _ContinuousProbeGenerator(
            source["generator.class_carrier_logits"], basis, gain=SAFE_PROBE_ALPHA
        )
        .to(device)
        .eval()
    )
    initial_equivalence: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, (codes, panel_labels) in panels.items():
            candidate_command = bundle.generator(codes, panel_labels)
            reference_command = safe_reference(codes, panel_labels)
            candidate_raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes,
                panel_labels,
                microbatch_size=1,
                activation_checkpoint=False,
            )
            reference_raw = raw_from_generator_batched(
                bundle.system,
                safe_reference,
                codes,
                panel_labels,
                microbatch_size=1,
                activation_checkpoint=False,
            )
            initial_equivalence[name] = {
                "command_max_abs": float(
                    (candidate_command - reference_command).abs().max().item()
                ),
                "raw_max_abs": float(
                    (candidate_raw - reference_raw).abs().max().item()
                ),
            }
    panel_metrics = {
        name: _metrics(
            bundle,
            evaluator,
            codes,
            panel_labels,
            real_views,
            config.conditional_generation.fixed_exposure_scale,
        )
        for name, (codes, panel_labels) in panels.items()
    }
    scales = bundle.generator.scales().detach()
    checks = (
        GateResult(
            "basis_hash",
            basis_hash == EXPECTED_CONTINUOUS_BASIS_HASH,
            {"observed": basis_hash},
        ),
        GateResult(
            "initial_scale",
            bool(
                torch.allclose(
                    scales, torch.full_like(scales, SAFE_PROBE_ALPHA), atol=1e-7
                )
            ),
            {"min": float(scales.min()), "max": float(scales.max())},
        ),
        GateResult(
            "safe_probe_equivalence",
            all(
                item["command_max_abs"] <= 1.0e-6 and item["raw_max_abs"] <= 1.0e-6
                for item in initial_equivalence.values()
            ),
            initial_equivalence,
        ),
        GateResult(
            "scale_bounds_and_budget",
            bool(
                (scales < 0.20).all()
                and (scales.square().sum(1).sqrt() <= 0.30 + 1e-6).all()
            ),
            {"max_l2": float(scales.square().sum(1).sqrt().max())},
        ),
        GateResult(
            "pca_rank",
            all(
                int(
                    torch.linalg.matrix_rank(
                        bundle.generator.lowpass_normalized_modes()[y].flatten(1)
                    ).item()
                )
                == 8
                for y in range(10)
            ),
            {},
        ),
        GateResult(
            "safe_semantics",
            all(
                float(panel_metrics[name]["accuracy"]) >= threshold
                for name, threshold in {
                    "axis": 0.85,
                    "uniform_1234": 0.80,
                    "uniform_5678": 0.80,
                }.items()
            ),
            {k: v["accuracy"] for k, v in panel_metrics.items()},
        ),
        GateResult(
            "safe_saturation",
            all(float(v["saturation"]) <= 0.08 for v in panel_metrics.values()),
            {},
        ),
        GateResult(
            "parameters_unchanged",
            before == state_content_hash(bundle.state_dict()),
            {},
        ),
    )
    run = _new_run(config)
    out = run.root / "conditional_pca_transport_initialization.pt"
    provenance = {
        "pca_basis_hash": basis_hash,
        "semantic_source_checkpoint_hash": file_sha256(args.semantic_checkpoint),
        "safe_scale_evidence_hash": file_sha256(args.safe_scale_evidence),
    }
    manifest = _manifest(
        config,
        split.manifest_hash,
        evaluator_hash,
        bundle,
        before,
        provenance=provenance,
    )
    save_conditional_checkpoint(
        out,
        bundle,
        manifest,
        extra={
            "semantic_source": file_sha256(args.semantic_checkpoint),
            "safe_scale_evidence": file_sha256(args.safe_scale_evidence),
            "basis_hash": basis_hash,
            "immutable_initialization": True,
        },
    )
    report = run.root / "conditional_pca_transport_admission_report.json"
    atomic_write_json(
        report,
        {
            "contract": PCA_TRANSPORT_ADMISSION_CONTRACT,
            "basis_hash": basis_hash,
            "source_state_hash": source_manifest.state_content_hash,
            "initial_state_hash": before,
            "panels": {
                k: {
                    n: v
                    for n, v in x.items()
                    if n not in {"raw", "command", "features", "logits"}
                }
                for k, x in panel_metrics.items()
            },
            "safe_probe_equivalence": initial_equivalence,
            "optimizer_created": False,
        },
    )
    out_hash = file_sha256(out)
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_pca_transport_admission",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=out_hash,
        ),
        {
            "checkpoint": out_hash,
            "safe_scale_evidence": file_sha256(args.safe_scale_evidence),
            "semantic_source": file_sha256(args.semantic_checkpoint),
            "report": file_sha256(report),
            "config_contract_hash": _config_contract_hash(config),
        },
        checks,
        ("conditional_pca_transport_observability",) if passed else (),
        None if passed else "pca transport initialization gate failed",
    )
    _write_evidence(run, "conditional_pca_transport_admission_evidence.json", record)
    print(run.root)
    return 0


def _load(args, config):
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("evaluator split mismatch")
    bundle = ConditionalBundle(config).to(device)
    # Loading is strict but the initialization-state hash is checked from its own manifest below.
    payload_cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actual = ConditionalCheckpointManifestV2(**payload_cp["manifest"])
    if (
        actual.architecture_kind != PCA_TRANSPORT_ARCHITECTURE
        or actual.system_signature != _context(config).system_signature
        or actual.config_hash != _config_contract_hash(config)
        or actual.evaluator_checkpoint_hash != evaluator_hash
        or actual.dataset_split_manifest_hash != split.manifest_hash
    ):
        raise ValueError("v6 checkpoint context mismatch")
    load_conditional_checkpoint(args.checkpoint, bundle, actual)
    return (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        bundle,
        actual,
        device,
    )


def conditional_pca_transport_observability(args, config) -> int:
    _require(config)
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.admission_evidence).read_text())
    )
    admission.validate()
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        bundle,
        manifest,
        device,
    ) = _load(args, config)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _require_stage_evidence(
        admission,
        kind="conditional_pca_transport_admission",
        action="conditional_pca_transport_observability",
        config=config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    before = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    groups = {
        "class_carrier": bundle.generator.class_carrier_logits,
        "pca_modes": bundle.generator.class_modes,
        "mode_scales": bundle.generator.scale_logits,
        "encoder_phase": bundle.system.encoder.phase,
        "decoder_phase": bundle.system.decoder.phase,
    }
    observations = {}
    valid = True
    for name, (codes, panel_labels) in panels.items():
        norms = _streamed_panel_gradient_norms(
            bundle,
            evaluator,
            codes,
            panel_labels,
            groups,
            config.conditional_generation.fixed_exposure_scale,
        )
        observations[name] = norms
        valid &= all(
            torch.isfinite(torch.tensor(value)) and value > 1e-12
            for value in norms.values()
        )
    unchanged = before == state_content_hash(bundle.state_dict())
    run = _new_run(config)
    report = run.root / "conditional_pca_transport_observability_report.json"
    atomic_write_json(
        report,
        {
            "contract": PCA_TRANSPORT_OBSERVABILITY_CONTRACT,
            "gradients": observations,
            "state_before": before,
            "state_after": state_content_hash(bundle.state_dict()),
            "optimizer_created": False,
        },
    )
    checks = (
        GateResult("five_group_gradients", valid, observations),
        GateResult("state_unchanged", unchanged, {}),
    )
    passed = all(c.passed for c in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_pca_transport_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission": admission.evidence_hash,
            "report": file_sha256(report),
            "config_contract_hash": _config_contract_hash(config),
        },
        checks,
        ("pca_transport_smoke",) if passed else (),
        None if passed else "joint transport gradients unavailable",
    )
    _write_evidence(
        run, "conditional_pca_transport_observability_evidence.json", record
    )
    print(run.root)
    return 0


def _training_codes(
    step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fixed 5-code-per-class schedule shared by smoke and formal train."""
    generator = torch.Generator(device="cpu").manual_seed(100000 + step)
    rows, labels = [], []
    for label in range(10):
        axis = step % 8
        u = torch.rand(8, generator=generator).mul_(2).sub_(1)
        plus = torch.zeros(8)
        plus[axis] = 1
        rows.extend((torch.zeros(8), plus, -plus, u, -u))
        labels.extend([label] * 5)
    return torch.stack(rows).to(device), torch.tensor(labels, device=device)


def _objective(
    bundle, evaluator, codes, generated_labels, real_view, real_labels, config
):
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        generated_labels,
        microbatch_size=1,
        activation_checkpoint=True,
    )
    view = fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(real_view)
    sinkhorn = conditional_sinkhorn_loss(
        F.normalize(features, dim=1),
        sinkhorn_cost_view(view),
        F.normalize(real_features, dim=1),
        sinkhorn_cost_view(real_view),
        generated_labels,
        epsilon=config.conditional_generation.sinkhorn_epsilon,
        iterations=config.conditional_generation.sinkhorn_iterations,
    )
    classification = F.cross_entropy(logits, generated_labels)
    diversity = relative_diversity_loss(
        F.normalize(features, dim=1),
        F.normalize(real_features, dim=1),
        generated_labels,
    )
    camera = camera_code_fraction_loss(raw, generated_labels)
    saturation = (
        torch.relu(raw / config.conditional_generation.fixed_exposure_scale - 1)
        .square()
        .mean()
    )
    orthogonal = bundle.generator.mode_orthogonality_penalty()
    phase_tv = phase_total_variation(bundle.system)
    budget = bundle.generator.scale_budget_penalty()
    loss = (
        sinkhorn
        + classification
        + diversity
        + 2 * camera
        + 0.1 * saturation
        + 1e-3 * orthogonal
        + 1e-5 * phase_tv
        + 1e-2 * budget
    )
    return loss, {
        "total": float(loss.detach()),
        "sinkhorn": float(sinkhorn.detach()),
        "class": float(classification.detach()),
        "relative_diversity": float(diversity.detach()),
        "camera_fraction": float(camera.detach()),
        "saturation": float(saturation.detach()),
        "orthogonality": float(orthogonal.detach()),
        "phase_tv": float(phase_tv.detach()),
        "scale_budget": float(budget.detach()),
    }


def pca_transport_smoke(args, config) -> int:
    """The one and only 400-step CUDA smoke for the v6 contract."""
    _require(config)
    obs = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text())
    )
    obs.validate()
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        bundle,
        manifest,
        device,
    ) = _load(args, config)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    _require_stage_evidence(
        obs,
        kind="conditional_pca_transport_observability",
        action="pca_transport_smoke",
        config=config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v6 smoke requires CUDA; CPU is diagnostic-only")
    torch.cuda.reset_peak_memory_stats(device)
    initial = state_content_hash(bundle.state_dict())
    panels = _panels(device)
    real_probe = _real_panel_views(train_images, train_labels, device=device)
    run = _new_run(config)
    probes, history = {}, []
    initial_parameters = {k: v.detach().clone() for k, v in bundle.state_dict().items()}
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
    if not batches:
        raise ValueError("empty balanced training schedule")

    def probe(step: int):
        payload = {}
        for name, (codes, labels) in panels.items():
            metric = _metrics(
                bundle,
                evaluator,
                codes,
                labels,
                real_probe,
                config.conditional_generation.fixed_exposure_scale,
            )
            raw_path = run.root / f"pca_transport_{name}_step_{step}_raw.pt"
            atomic_torch_save(raw_path, metric["raw"].detach().cpu())
            _full_contact_sheet(
                run.root / f"pca_transport_{name}_step_{step}_10x17.png",
                metric["raw"],
                labels,
                config.conditional_generation.fixed_exposure_scale,
            )
            payload[name] = {
                k: v
                for k, v in metric.items()
                if k not in {"raw", "command", "features", "logits"}
            }
        probes[str(step)] = payload

    probe(0)
    for step in range(400):
        indices = batches[step % len(batches)]
        real_view = train_images.index_select(0, indices).to(device)
        real_labels = train_labels.index_select(0, indices).to(device)
        codes, generated_labels = _training_codes(step, device)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _objective(
            bundle, evaluator, codes, generated_labels, real_view, real_labels, config
        )
        loss.backward()
        grad_norms = {
            "carrier": float(bundle.generator.class_carrier_logits.grad.norm()),
            "modes": float(bundle.generator.class_modes.grad.norm()),
            "scales": float(bundle.generator.scale_logits.grad.norm()),
            "encoder": float(bundle.system.encoder.phase.grad.norm()),
            "decoder": float(bundle.system.decoder.phase.grad.norm()),
        }
        optimizer.step()
        history.append(
            summary
            | {
                "step": step + 1,
                "code_hash": tensor_sha256(codes.detach().cpu()),
                "gradients": grad_norms,
            }
        )
        if step + 1 in {50, 100, 200, 400}:
            probe(step + 1)
    peak = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    final = state_content_hash(bundle.state_dict())
    updates = {
        k: float((v - initial_parameters[k].to(v.device)).norm().item())
        for k, v in bundle.state_dict().items()
    }
    final_panels = probes["400"]
    checks = (
        GateResult(
            "semantic_accuracy",
            all(
                x["accuracy"] >= 0.90 and min(x["recall"]) >= 0.80
                for x in final_panels.values()
            ),
            final_panels,
        ),
        GateResult(
            "camera_fraction",
            all(x["camera_fraction"] >= 0.10 for x in final_panels.values()),
            final_panels,
        ),
        GateResult(
            "diversity_floor",
            all(min(x["ratios"]) >= 0.35 for x in final_panels.values()),
            final_panels,
        ),
        GateResult(
            "saturation",
            all(x["saturation"] <= 0.08 for x in final_panels.values()),
            final_panels,
        ),
        GateResult(
            "scale_budget",
            bool(
                (bundle.generator.scales() < 0.20).all()
                and (bundle.generator.scales().square().sum(1).sqrt() <= 0.3001).all()
            ),
            {},
        ),
        GateResult("updates", all(value > 0 for value in updates.values()), updates),
        GateResult(
            "cuda_memory", peak <= 0.92 * total, {"reserved": peak, "total": total}
        ),
    )
    diagnostic = run.root / "pca_transport_smoke_final.pt"
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or initial,
            provenance={
                "pca_basis_hash": manifest.pca_basis_hash or "",
                "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
                or "",
                "safe_scale_evidence_hash": manifest.safe_scale_evidence_hash or "",
            },
        ),
        extra={
            "diagnostic_only": True,
            "never_warm_start": True,
            "contract": PCA_TRANSPORT_SMOKE_CONTRACT,
        },
    )
    history_path = run.root / "conditional_pca_transport_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": PCA_TRANSPORT_SMOKE_CONTRACT,
            "history": history,
            "probes": probes,
            "updates": updates,
            "initial_hash": initial,
            "final_hash": final,
        },
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_pca_transport_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "observability": obs.evidence_hash,
            "history": file_sha256(history_path),
            "diagnostic_final_checkpoint": file_sha256(diagnostic),
            "config_contract_hash": _config_contract_hash(config),
        },
        checks,
        ("train_pca_transport",) if passed else (),
        None if passed else "PCA transport smoke failed",
    )
    _write_evidence(run, "conditional_pca_transport_smoke_evidence.json", record)
    print(run.root)
    return 0


def train_pca_transport(args, config) -> int:
    """User-invoked five-epoch run; always starts from immutable v6 init."""
    _require(config)
    smoke = EvidenceRecordV2.from_dict(
        json.loads(Path(args.smoke_evidence).read_text())
    )
    smoke.validate()
    (
        train_images,
        train_labels,
        split,
        evaluator_hash,
        evaluator,
        bundle,
        manifest,
        device,
    ) = _load(args, config)
    _require_stage_evidence(
        smoke,
        kind="conditional_pca_transport_smoke",
        action="train_pca_transport",
        config=config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
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
            real_view = train_images.index_select(0, indices).to(device)
            codes, generated_labels = _training_codes(global_step, device)
            optimizer.zero_grad(set_to_none=True)
            loss, summary = _objective(
                bundle,
                evaluator,
                codes,
                generated_labels,
                real_view,
                train_labels.index_select(0, indices).to(device),
                config,
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
    panels = _panels(device)
    real_probe = _real_panel_views(train_images, train_labels, device=device)
    final = {
        name: _metrics(
            bundle,
            evaluator,
            codes,
            labels,
            real_probe,
            config.conditional_generation.fixed_exposure_scale,
        )
        for name, (codes, labels) in panels.items()
    }
    checks = (
        GateResult(
            "semantic",
            all(
                x["accuracy"] >= 0.95 and min(x["recall"]) >= 0.90
                for x in final.values()
            ),
            {k: x["accuracy"] for k, x in final.items()},
        ),
        GateResult(
            "camera_fraction",
            all(x["camera_fraction"] >= 0.25 for x in final.values()),
            {k: x["camera_fraction"] for k, x in final.items()},
        ),
        GateResult(
            "diversity",
            all(
                all(0.70 <= ratio <= 1.30 for ratio in x["ratios"])
                for x in final.values()
            ),
            {k: x["ratios"] for k, x in final.items()},
        ),
        GateResult(
            "saturation", all(x["saturation"] <= 0.08 for x in final.values()), {}
        ),
    )
    run = _new_run(config)
    output = run.root / "conditional_pca_transport_generator.pt"
    save_conditional_checkpoint(
        output,
        bundle,
        _manifest(
            config,
            split.manifest_hash,
            evaluator_hash,
            bundle,
            manifest.initialization_state_hash or manifest.state_content_hash,
            provenance={
                "pca_basis_hash": manifest.pca_basis_hash or "",
                "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash
                or "",
                "safe_scale_evidence_hash": manifest.safe_scale_evidence_hash or "",
            },
        ),
        extra={
            "history": history,
            "smoke_evidence": smoke.evidence_hash,
            "formal_epochs": 5,
        },
    )
    history_path = run.root / "conditional_pca_transport_training_history.json"
    atomic_write_json(
        history_path,
        {
            "history": history,
            "steps": global_step,
            "panels": {
                k: {
                    n: v
                    for n, v in x.items()
                    if n not in {"raw", "command", "features", "logits"}
                }
                for k, x in final.items()
            },
        },
    )
    passed = all(check.passed for check in checks)
    out_hash = file_sha256(output)
    record = EvidenceRecordV2(
        2,
        "conditional_pca_transport_training",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=out_hash,
        ),
        {
            "checkpoint": out_hash,
            "smoke": smoke.evidence_hash,
            "history": file_sha256(history_path),
            "config_contract_hash": _config_contract_hash(config),
        },
        checks,
        ("evaluate",) if passed else (),
        None if passed else "PCA transport formal training gate failed",
    )
    _write_evidence(run, "conditional_pca_transport_training_evidence.json", record)
    print(run.root)
    return 0
