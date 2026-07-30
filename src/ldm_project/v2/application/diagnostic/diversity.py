# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from . import recovery as _recovery

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})

def _diversity_parameters(
    bundle: ConditionalBundle,
) -> dict[str, tuple[nn.Parameter, ...]]:
    return {
        "global_modes": (bundle.generator.global_modes,),
        "code_projection": (bundle.generator.code_projection.weight,),
        "class_heads": (bundle.generator.class_heads,),
    }


def _aligned_codes(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Five deterministic codes per label: zero and two antithetic pairs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for _ in range(10):
        vectors = torch.rand(2, 8, generator=generator).mul_(2.0).sub_(1.0)
        rows.append(torch.cat((torch.zeros(1, 8), vectors, -vectors), dim=0))
    panel = torch.cat(rows, dim=0).to(labels.device)
    if labels.numel() != panel.shape[0]:
        raise ValueError("aligned code schedule requires a logical batch of 50")
    return panel


def _aligned_objective(
    *,
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    settings,
    config: V2Config,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=config.execution.training_microbatch_size,
        activation_checkpoint=config.execution.activation_checkpoint,
    )
    view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
    generated_features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(
        generated_features,
        view,
        real_features,
        real_views,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    classification = torch.nn.functional.cross_entropy(logits, labels)
    relative = relative_diversity_loss(generated_features, real_features, labels)
    camera_loss = camera_code_fraction_loss(raw, labels)
    saturation = torch.relu(raw / settings.fixed_exposure_scale - 1.0).square().mean()
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    total = (
        sinkhorn
        + 0.5 * classification
        + relative
        + 2.0 * camera_loss
        + 0.1 * saturation
        + 1.0e-3 * orthogonality
    )
    return (
        total,
        {
            "total": float(total.detach()),
            "sinkhorn": float(sinkhorn.detach()),
            "classification": float(classification.detach()),
            "relative_diversity": float(relative.detach()),
            "camera_fraction_loss": float(camera_loss.detach()),
            "camera_fraction": float(camera_code_fraction(raw, labels).detach()),
            "saturation": float(saturation.detach()),
            "mode_orthogonality": float(orthogonality.detach()),
        },
        raw,
    )


def _reanchor_codes(labels: torch.Tensor, *, batch_number: int) -> torch.Tensor:
    """Five codes per class: one rotating axis pair and one dense antithetic pair."""
    generator = torch.Generator(device="cpu").manual_seed(100000 + batch_number)
    direction = batch_number % 8
    rows = []
    for _ in range(10):
        dense = torch.rand(8, generator=generator).mul_(2.0).sub_(1.0)
        axis = torch.zeros(8)
        axis[direction] = 1.0
        rows.append(torch.stack((torch.zeros(8), axis, -axis, dense, -dense)))
    codes = torch.cat(rows).to(labels.device)
    if codes.shape[0] != labels.numel():
        raise ValueError("reanchor schedule requires a logical batch of 50")
    return codes


def _reanchor_schedule_hash(train_labels: torch.Tensor, *, count: int = 400) -> str:
    """Hash the deterministic labels and mixed axis/uniform code schedule."""
    entries = []
    for batch_number, indices in enumerate(
        balanced_batches(train_labels, batch_size=50, seed=1234)
    ):
        if batch_number >= count:
            break
        labels = train_labels.index_select(0, indices)
        entries.append(
            {
                "indices": tensor_sha256(indices),
                "labels": tensor_sha256(labels),
                "codes": tensor_sha256(
                    _reanchor_codes(labels, batch_number=batch_number)
                ),
            }
        )
    if len(entries) != count:
        raise ValueError(
            f"reanchor schedule requires {count} batches, got {len(entries)}"
        )
    return canonical_sha256(entries)


def _reanchor_objective(
    *,
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    settings,
    config: V2Config,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=config.execution.training_microbatch_size,
        activation_checkpoint=config.execution.activation_checkpoint,
    )
    view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
    generated_features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(
        generated_features,
        view,
        real_features,
        real_views,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    classification = torch.nn.functional.cross_entropy(logits, labels)
    relative = relative_diversity_loss(generated_features, real_features, labels)
    camera_loss = camera_code_fraction_loss(raw, labels)
    saturation = torch.relu(raw / settings.fixed_exposure_scale - 1.0).square().mean()
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    total = (
        sinkhorn
        + classification
        + relative
        + 2.0 * camera_loss
        + 0.1 * saturation
        + 1.0e-3 * orthogonality
    )
    return (
        total,
        {
            "total": float(total.detach()),
            "sinkhorn": float(sinkhorn.detach()),
            "classification": float(classification.detach()),
            "relative_diversity": float(relative.detach()),
            "camera_fraction_loss": float(camera_loss.detach()),
            "camera_fraction": float(camera_code_fraction(raw, labels).detach()),
            "saturation": float(saturation.detach()),
            "mode_orthogonality": float(orthogonality.detach()),
        },
        raw,
    )


def _zero_reanchor(bundle: ConditionalBundle) -> None:
    with torch.no_grad():
        bundle.generator.class_heads.zero_()
    bundle.generator.class_heads.requires_grad_(False)
    bundle.generator.code_projection.weight.requires_grad_(False)


def _panel_real_features(
    evaluator: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    real = torch.cat(
        [train_images[train_labels == label][:17] for label in range(10)], dim=0
    ).to(device)
    with torch.no_grad():
        features, _ = evaluator(real)
    return features


def _reanchor_panel_metrics(
    raw: torch.Tensor,
    labels: torch.Tensor,
    evaluator: nn.Module,
    real_features: torch.Tensor,
    exposure_scale: float,
) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure_scale)
    with torch.no_grad():
        features, logits = evaluator(view)
    predictions = logits.argmax(dim=1)
    metrics = {
        "accuracy": float((predictions == labels).float().mean()),
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100)
        .reshape(10, 10)
        .cpu()
        .tolist(),
        "code_fraction": float(camera_code_fraction(raw, labels)),
        "feature_code_fraction": balanced_code_fraction(features.reshape(10, 17, -1)),
        "diversity_ratios": relative_diversity_ratios(features, real_features, labels),
        "saturation": float(
            torch.relu(raw / exposure_scale - 1.0).gt(0).float().mean()
        ),
    }
    metrics["recalls"] = [
        float(
            metrics["confusion"][label][label]
            / max(1, sum(metrics["confusion"][label]))
        )
        for label in range(10)
    ]
    return metrics


def _diversity_objective(
    *,
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    settings,
    config: V2Config,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=config.execution.training_microbatch_size,
        activation_checkpoint=config.execution.activation_checkpoint,
    )
    generated_view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
    generated_features, logits = evaluator(generated_view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(
        generated_features,
        generated_view,
        real_features,
        real_views,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    classification = torch.nn.functional.cross_entropy(logits, labels)
    diversity = relative_diversity_loss(generated_features, real_features, labels)
    saturation = torch.relu(raw / settings.fixed_exposure_scale - 1.0).square().mean()
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    total = (
        sinkhorn
        + 0.5 * classification
        + diversity
        + 0.1 * saturation
        + 1.0e-3 * orthogonality
    )
    return (
        total,
        {
            "total": float(total.detach()),
            "sinkhorn": float(sinkhorn.detach()),
            "classification": float(classification.detach()),
            "relative_diversity": float(diversity.detach()),
            "saturation": float(saturation.detach()),
            "mode_orthogonality": float(orthogonality.detach()),
        },
        raw,
    )


def conditional_diversity_observability(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    corrected_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=corrected_hash,
    )
    completion = EvidenceRecordV2.from_dict(
        json.loads(Path(args.training_completion_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        completion,
        kind="conditional_training_completion",
        action="diversity_observability",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    if manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT:
        raise ValueError(
            "diversity observability requires the corrected balanced schedule checkpoint"
        )
    groups = _diversity_parameters(bundle)
    codes, probe_labels = _fixed_control_probe(device=device)
    with torch.no_grad():
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            probe_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
    state_before = state_content_hash(bundle.state_dict())
    indices = next(iter(balanced_batches(train_labels, batch_size=50, seed=1234)))
    settings = config.conditional_generation
    assert settings is not None
    train_codes = _random_codes(50, device=device, seed=100000)
    train_labels_device = train_labels.index_select(0, indices).to(device)
    real_views = train_images.index_select(0, indices).to(device)
    total, _, _ = _diversity_objective(
        bundle=bundle,
        evaluator=evaluator,
        codes=train_codes,
        labels=train_labels_device,
        real_views=real_views,
        settings=settings,
        config=config,
    )
    grads = torch.autograd.grad(
        total,
        [parameter for params in groups.values() for parameter in params],
        allow_unused=True,
    )
    gradient_gate = all(
        g is not None and torch.isfinite(g).all() and g.detach().norm() > 0
        for g in grads
    )
    state_after = state_content_hash(bundle.state_dict())
    coeff = bundle.generator.coefficients(codes, probe_labels).reshape(10, 17, 8)
    checks = (
        GateResult("finite_optical_response", bool(torch.isfinite(raw).all()), {}),
        GateResult("code_parameter_gradients", gradient_gate, {}),
        GateResult(
            "coefficient_rank_eight",
            int(torch.linalg.matrix_rank(coeff[:, 1:].reshape(-1, 8)).item()) == 8,
            {},
        ),
        GateResult("parameters_unchanged", state_before == state_after, {}),
    )
    run = _new_run(config)
    report_path = run.root / "conditional_diversity_observability_report.json"
    atomic_write_json(
        report_path,
        {
            "raw_code_fraction": _code_fraction(raw, probe_labels),
            "rank": int(torch.linalg.matrix_rank(coeff[:, 1:].reshape(-1, 8)).item()),
            "gradient_gate": gradient_gate,
            "parameters_unchanged": state_before == state_after,
            "diversity_loss_contract": DIVERSITY_LOSS_CONTRACT,
        },
    )
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "completion": file_sha256(args.training_completion_evidence),
            "checkpoint": corrected_hash,
            "report": file_sha256(report_path),
        },
        checks,
        ("diversity_recovery_smoke",) if passed else (),
        None if passed else "diversity observability failed",
    )
    _write_evidence(run, "conditional_diversity_observability_evidence.json", record)
    print(run.root)
    return 0
def conditional_diversity_alignment_admission(args, config: V2Config) -> int:
    """Read-only admission for the gate-aligned diversity contract."""
    _require_identity(config)
    failed = EvidenceRecordV2.from_dict(
        json.loads(Path(args.failed_recovery_evidence).read_text(encoding="utf-8"))
    )
    failed.validate()
    if (
        failed.evidence_kind != "conditional_diversity_recovery_smoke"
        or failed.status != "failed"
    ):
        raise ValueError(
            "alignment admission requires the immutable failed diversity recovery evidence"
        )
    failed_gates = {gate.name: gate.passed for gate in failed.gates}
    if any(
        failed_gates.get(name) is not True
        for name in (
            "semantic_accuracy",
            "per_class_recall",
            "relative_diversity",
            "high_saturation",
            "frozen_parameters",
            "code_parameter_updates",
            "zero_code_unchanged",
            "finite_losses",
        )
    ):
        raise ValueError(
            "failed recovery contains failures beyond the camera-fraction gate"
        )
    history_path = Path(args.failed_recovery_history)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    probes = history.get("probes", {})
    required_steps = ("step_0", "step_100", "step_200", "step_400")
    if any(step not in probes for step in required_steps):
        raise ValueError(
            "failed recovery history lacks the fixed 0/100/200/400 probe panel"
        )
    camera_values = [float(probes[step]["code_fraction"]) for step in required_steps]
    ratio_values = [
        min(float(v) for v in probes[step]["diversity_ratios"])
        for step in required_steps
    ]
    if not all(
        camera_values[i] <= camera_values[i + 1]
        and ratio_values[i] <= ratio_values[i + 1]
        for i in range(3)
    ):
        raise ValueError("failed recovery trend is not monotonic")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, _ = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    for parameter in (
        bundle.generator.class_carrier_logits,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    indices = next(iter(balanced_batches(train_labels, batch_size=50, seed=1234)))
    labels = train_labels.index_select(0, indices).to(device)
    codes = _aligned_codes(labels, seed=100000)
    settings = config.conditional_generation
    assert settings is not None
    before = state_content_hash(bundle.state_dict())
    objective, summary, _ = _aligned_objective(
        bundle=bundle,
        evaluator=evaluator,
        codes=codes,
        labels=labels,
        real_views=train_images.index_select(0, indices).to(device),
        settings=settings,
        config=config,
    )
    grads = torch.autograd.grad(
        objective,
        [parameter for params in groups.values() for parameter in params],
        allow_unused=True,
    )
    gradient_gate = all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.detach().norm() > 0
        for gradient in grads
    )
    after = state_content_hash(bundle.state_dict())
    checks = (
        GateResult(
            "only_camera_fraction_failure",
            True,
            {"failed_gate": "camera_code_fraction"},
        ),
        GateResult(
            "monotonic_probe_trend",
            True,
            {
                "camera_start": camera_values[0],
                "camera_end": camera_values[-1],
                "ratio_start": ratio_values[0],
                "ratio_end": ratio_values[-1],
            },
        ),
        GateResult("aligned_loss_gradients", gradient_gate, {}),
        GateResult("parameters_unchanged", before == after, {}),
    )
    run = _new_run(config)
    report_path = run.root / "conditional_diversity_alignment_admission_report.json"
    atomic_write_json(
        report_path,
        {
            "metric_contract": ALIGNED_DIVERSITY_METRIC_CONTRACT,
            "loss_contract": ALIGNED_DIVERSITY_LOSS_CONTRACT,
            "camera_values": camera_values,
            "ratio_values": ratio_values,
            "aligned_summary": summary,
            "parameters_unchanged": before == after,
        },
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_alignment_admission",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "failed_recovery_evidence": file_sha256(args.failed_recovery_evidence),
            "failed_recovery_history": file_sha256(history_path),
            "checkpoint": checkpoint_hash,
            "report": file_sha256(report_path),
        },
        checks,
        ("diversity_aligned_smoke",) if passed else (),
        None if passed else "aligned diversity admission failed",
    )
    _write_evidence(
        run, "conditional_diversity_alignment_admission_evidence.json", record
    )
    print(run.root)
    return 0

def conditional_identity_reanchor(args, config: V2Config) -> int:
    """Derive a Wy=0 checkpoint without optimizer or optical parameter updates."""
    _require_identity(config)
    aligned = EvidenceRecordV2.from_dict(
        json.loads(Path(args.aligned_smoke_evidence).read_text(encoding="utf-8"))
    )
    if (
        aligned.evidence_kind != "conditional_diversity_aligned_smoke"
        or aligned.status != "failed"
    ):
        raise ValueError(
            "reanchor requires the immutable failed aligned smoke evidence"
        )
    failed = {gate.name: gate.passed for gate in aligned.gates}
    if any(
        failed.get(name) is not True
        for name in (
            "semantic_accuracy",
            "per_class_recall",
            "high_saturation",
            "coefficient_rank_eight",
            "zero_code_unchanged",
            "frozen_parameters",
            "finite_losses",
            "code_parameter_updates",
        )
    ):
        raise ValueError("aligned smoke has failures beyond diversity alignment")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    source_hash = file_sha256(args.checkpoint)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, source_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    before = {
        name: value.detach().clone() for name, value in bundle.state_dict().items()
    }
    _zero_reanchor(bundle)
    after = bundle.state_dict()
    changed = [
        name
        for name in before
        if not torch.equal(before[name].cpu(), after[name].detach().cpu())
    ]
    expected_changed = ["generator.class_heads"]
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    uniform_codes = _random_codes(170, device=device, seed=1234)
    uniform_labels = torch.arange(10, device=device).repeat_interleave(17)
    real_features = _panel_real_features(evaluator, train_images, train_labels, device)
    with torch.no_grad():
        axis_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            axis_codes,
            axis_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        uniform_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            uniform_codes,
            uniform_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
    axis_metrics = _reanchor_panel_metrics(
        axis_raw,
        axis_labels,
        evaluator,
        real_features,
        config.conditional_generation.fixed_exposure_scale,
    )
    uniform_metrics = _reanchor_panel_metrics(
        uniform_raw,
        uniform_labels,
        evaluator,
        real_features,
        config.conditional_generation.fixed_exposure_scale,
    )
    checks = (
        GateResult(
            "only_class_heads_zeroed", changed == expected_changed, {"changed": changed}
        ),
        GateResult(
            "axis_accuracy",
            axis_metrics["accuracy"] >= 0.85,
            {"accuracy": axis_metrics["accuracy"]},
        ),
        GateResult(
            "axis_camera_fraction",
            axis_metrics["code_fraction"] >= 0.25,
            {"value": axis_metrics["code_fraction"]},
        ),
        GateResult(
            "axis_min_diversity",
            min(axis_metrics["diversity_ratios"]) >= 0.50,
            {"value": min(axis_metrics["diversity_ratios"])},
        ),
        GateResult(
            "uniform_accuracy",
            uniform_metrics["accuracy"] >= 0.70,
            {"accuracy": uniform_metrics["accuracy"]},
        ),
        GateResult(
            "uniform_camera_fraction",
            uniform_metrics["code_fraction"] >= 0.25,
            {"value": uniform_metrics["code_fraction"]},
        ),
        GateResult(
            "finite_saturation",
            axis_metrics["saturation"] <= 0.08
            and uniform_metrics["saturation"] <= 0.08,
            {},
        ),
        GateResult(
            "rank_eight",
            int(
                torch.linalg.matrix_rank(
                    bundle.generator.coefficients(axis_codes, axis_labels).reshape(
                        -1, 8
                    )
                ).item()
            )
            == 8,
            {},
        ),
    )
    run = _new_run(config)
    path = run.root / "conditional_identity_reanchor.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        IDENTITY_ARCHITECTURE,
        source_manifest.system_signature,
        source_manifest.config_hash,
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        IDENTITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        source_manifest.initialization_state_hash,
    )
    save_conditional_checkpoint(
        path,
        bundle,
        manifest,
        extra={
            "reanchor_contract": REANCHOR_CONTRACT,
            "source_checkpoint": source_hash,
            "zeroed_keys": expected_changed,
            "diagnostic_only_until_observability": True,
        },
    )
    out_hash = file_sha256(path)
    report = run.root / "conditional_identity_reanchor_report.json"
    atomic_write_json(
        report,
        {
            "contract": REANCHOR_CONTRACT,
            "source_checkpoint": source_hash,
            "axis": axis_metrics,
            "uniform": uniform_metrics,
            "changed_keys": changed,
            "wy_zero": True,
        },
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_identity_reanchor_admission",
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
            "source_checkpoint": source_hash,
            "checkpoint": out_hash,
            "aligned_smoke": file_sha256(args.aligned_smoke_evidence),
            "report": file_sha256(report),
        },
        checks,
        ("conditional_reanchor_observability",) if passed else (),
        None if passed else "identity reanchor admission failed",
    )
    _write_evidence(
        run, "conditional_identity_reanchor_admission_evidence.json", record
    )
    print(run.root)
    return 0


def conditional_reanchor_observability(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash,
    )
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.reanchor_admission_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        admission,
        kind="conditional_identity_reanchor_admission",
        action="conditional_reanchor_observability",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    if manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT:
        raise ValueError("reanchor observability requires balanced schedule")
    _zero_reanchor(bundle)
    bundle.generator.global_modes.requires_grad_(True)
    before = state_content_hash(bundle.state_dict())
    settings = config.conditional_generation
    assert settings is not None
    indices = next(iter(balanced_batches(train_labels, batch_size=50, seed=1234)))
    labels = train_labels.index_select(0, indices).to(device)
    codes = _reanchor_codes(labels, batch_number=0)
    objective, summary, raw = _reanchor_objective(
        bundle=bundle,
        evaluator=evaluator,
        codes=codes,
        labels=labels,
        real_views=train_images.index_select(0, indices).to(device),
        settings=settings,
        config=config,
    )
    gradient = torch.autograd.grad(
        objective, bundle.generator.global_modes, allow_unused=True
    )[0]
    after = state_content_hash(bundle.state_dict())
    trace = (
        bundle.system.forward_from_generator(
            bundle.generator, codes[:2], labels[:2], return_optical_trace=True
        ).optical_trace
        or ()
    )
    checks = (
        GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}),
        GateResult(
            "global_mode_gradient",
            bool(
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                and bool(gradient.detach().norm() > 0)
            ),
            {},
        ),
        GateResult(
            "class_heads_zero",
            bool(
                torch.equal(
                    bundle.generator.class_heads.detach(),
                    torch.zeros_like(bundle.generator.class_heads),
                )
            ),
            {},
        ),
        GateResult(
            "complete_trace",
            any("identity_ols" in field.provenance.operation for field in trace),
            {},
        ),
        GateResult("parameters_unchanged", before == after, {}),
    )
    run = _new_run(config)
    report = run.root / "conditional_reanchor_observability_report.json"
    atomic_write_json(
        report,
        {
            "contract": REANCHOR_CONTRACT,
            "summary": summary,
            "gradient_norm": float(gradient.norm()) if gradient is not None else None,
            "trace_operations": [field.provenance.operation for field in trace],
            "parameters_unchanged": before == after,
        },
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_reanchor_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission": file_sha256(args.reanchor_admission_evidence),
            "checkpoint": checkpoint_hash,
            "report": file_sha256(report),
        },
        checks,
        ("identity_reanchor_smoke",) if passed else (),
        None if passed else "reanchor observability failed",
    )
    _write_evidence(run, "conditional_reanchor_observability_evidence.json", record)
    print(run.root)
    return 0


def identity_reanchor_smoke(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash,
    )
    obs = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        obs,
        kind="conditional_reanchor_observability",
        action="identity_reanchor_smoke",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, init_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    _zero_reanchor(bundle)
    bundle.generator.global_modes.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [bundle.generator.global_modes],
        lr=config.conditional_generation.generator_learning_rate,
    )
    settings = config.conditional_generation
    assert settings is not None
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    uniform_codes = _random_codes(170, device=device, seed=1234)
    uniform_labels = torch.arange(10, device=device).repeat_interleave(17)
    real_features = _panel_real_features(evaluator, train_images, train_labels, device)
    initial_state = {
        name: value.detach().clone() for name, value in bundle.state_dict().items()
    }
    zero_raw = (
        raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            torch.zeros(10, 8, device=device),
            torch.arange(10, device=device),
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
    )
    panels = {
        "axis": (axis_codes, axis_labels),
        "uniform": (uniform_codes, uniform_labels),
    }
    probe_history = {}
    loss_history = []
    update_history = []
    raw_history = {}

    def probe():
        result = {}
        raws = {}
        with torch.no_grad():
            for name, (codes, labels) in panels.items():
                raw = raw_from_generator_batched(
                    bundle.system,
                    bundle.generator,
                    codes,
                    labels,
                    microbatch_size=config.execution.evaluation_microbatch_size,
                    activation_checkpoint=False,
                )
                result[name] = _reanchor_panel_metrics(
                    raw, labels, evaluator, real_features, settings.fixed_exposure_scale
                )
                raws[name] = raw.detach().cpu()
        return result, raws

    metrics, raws = probe()
    probe_history["step_0"] = metrics
    raw_history[0] = raws
    for step, indices in zip(
        range(400),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=step)
        optimizer.zero_grad(set_to_none=True)
        total, summary, raw = _reanchor_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=settings,
            config=config,
        )
        total.backward()
        before = bundle.generator.global_modes.detach().clone()
        optimizer.step()
        update_history.append(
            {
                "global_modes": float(
                    torch.linalg.vector_norm(
                        bundle.generator.global_modes.detach() - before
                    )
                )
            }
        )
        loss_history.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            metrics, raws = probe()
            probe_history[f"step_{step + 1}"] = metrics
            raw_history[step + 1] = raws
    final = probe_history["step_400"]
    axis400 = final["axis"]
    uni400 = final["uniform"]
    axis_ratio400 = min(axis400["diversity_ratios"])
    cls0 = sum(x["classification"] for x in loss_history[:50]) / 50
    cls1 = sum(x["classification"] for x in loss_history[-50:]) / 50
    frozen = all(
        torch.equal(initial_state[name].cpu(), value.detach().cpu())
        for name, value in bundle.state_dict().items()
        if name.startswith("generator.class_carrier")
        or name.startswith("generator.code_projection")
        or name.startswith("generator.class_heads")
        or name.startswith("system.")
    )
    zero_after = (
        raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            torch.zeros(10, 8, device=device),
            torch.arange(10, device=device),
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
    )
    checks = (
        GateResult(
            "axis_accuracy",
            axis400["accuracy"] >= 0.95,
            {"accuracy": axis400["accuracy"]},
        ),
        GateResult("axis_recall", min(axis400["recalls"]) >= 0.90, {}),
        GateResult("axis_code_fraction", axis400["code_fraction"] >= 0.25, {}),
        GateResult("axis_diversity", axis_ratio400 >= 0.60, {"minimum": axis_ratio400}),
        GateResult(
            "uniform_accuracy",
            uni400["accuracy"] >= 0.90,
            {"accuracy": uni400["accuracy"]},
        ),
        GateResult("uniform_recall", min(uni400["recalls"]) >= 0.80, {}),
        GateResult("uniform_code_fraction", uni400["code_fraction"] >= 0.25, {}),
        GateResult("uniform_diversity", min(uni400["diversity_ratios"]) >= 0.60, {}),
        GateResult(
            "saturation",
            axis400["saturation"] <= 0.08 and uni400["saturation"] <= 0.08,
            {},
        ),
        GateResult("zero_code_unchanged", bool(torch.equal(zero_raw, zero_after)), {}),
        GateResult("frozen_parameters", frozen, {}),
        GateResult(
            "classification_progress",
            cls1 <= 0.85 * cls0,
            {"first_50": cls0, "last_50": cls1},
        ),
        GateResult(
            "finite_losses",
            bool(
                torch.isfinite(torch.tensor([x["total"] for x in loss_history])).all()
            ),
            {},
        ),
        GateResult(
            "global_mode_updates",
            all(x["global_modes"] > 0 for x in update_history),
            {},
        ),
    )
    run = _new_run(config)
    artifacts = {}
    for step, panel_raws in raw_history.items():
        artifacts[str(step)] = {}
        for name, raw in panel_raws.items():
            path = run.root / f"identity_reanchor_{name}_step_{step}_raw_cmos.pt"
            atomic_torch_save(path, raw)
            preview = write_review_artifacts(
                root=run.root,
                stem=f"identity_reanchor_{name}_step_{step}",
                kind="conditional_identity_reanchor_smoke",
                raw_cmos=raw,
                labels=axis_labels.cpu() if name == "axis" else uniform_labels.cpu(),
                exposure_scale=settings.fixed_exposure_scale,
            )
            artifacts[str(step)][name] = {
                "raw": file_sha256(path),
                "preview": preview.preview_sha256,
            }
    schedule_hash = _reanchor_schedule_hash(train_labels)
    history_path = run.root / "identity_reanchor_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": REANCHOR_SMOKE_CONTRACT,
            "probes": probe_history,
            "loss": loss_history,
            "updates": update_history,
            "artifacts": artifacts,
            "first_400_schedule_hash": schedule_hash,
        },
    )
    diagnostic = run.root / "identity_reanchor_smoke_final.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        IDENTITY_ARCHITECTURE,
        init_manifest.system_signature,
        init_manifest.config_hash,
        split.manifest_hash,
        evaluator_hash,
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        IDENTITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        init_manifest.initialization_state_hash,
    )
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        manifest,
        extra={
            "diagnostic_only": True,
            "never_warm_start": True,
            "contract": REANCHOR_SMOKE_CONTRACT,
            "first_400_schedule_hash": schedule_hash,
        },
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_identity_reanchor_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "observability": file_sha256(args.observability_evidence),
            "history": file_sha256(history_path),
            "diagnostic_final_checkpoint": file_sha256(diagnostic),
            "contract": REANCHOR_SMOKE_CONTRACT,
            "first_400_schedule_hash": schedule_hash,
        },
        checks,
        ("train_identity_reanchor",) if passed else (),
        None if passed else "identity reanchor smoke failed",
    )
    _write_evidence(run, "conditional_identity_reanchor_smoke_evidence.json", record)
    print(run.root)
    return 0


def train_identity_reanchor(args, config: V2Config) -> int:
    _require_identity(config)
    smoke = _passed_evidence(
        args.smoke_evidence, kind="conditional_identity_reanchor_smoke"
    )
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash,
    )
    require_evidence_action(
        smoke,
        kind="conditional_identity_reanchor_smoke",
        action="train_identity_reanchor",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, init_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    _zero_reanchor(bundle)
    bundle.generator.global_modes.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [bundle.generator.global_modes],
        lr=config.conditional_generation.generator_learning_rate,
    )
    settings = config.conditional_generation
    assert settings is not None
    history = []
    for batch_number, indices in enumerate(
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        )
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=batch_number)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _reanchor_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=settings,
            config=config,
        )
        loss.backward()
        optimizer.step()
        history.append(summary | {"step": float(batch_number + 1)})
    schedule_hash = _reanchor_schedule_hash(train_labels)
    run = _new_run(config)
    path = run.root / "conditional_identity_reanchored_generator.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        IDENTITY_ARCHITECTURE,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_hash,
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        IDENTITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        init_manifest.initialization_state_hash,
    )
    save_conditional_checkpoint(
        path,
        bundle,
        manifest,
        extra={
            "contract": REANCHOR_CONTRACT,
            "smoke_evidence": file_sha256(args.smoke_evidence),
            "steps": len(history),
            "first_400_schedule_hash": schedule_hash,
        },
    )
    out_hash = file_sha256(path)
    finite = bool(torch.isfinite(torch.tensor([x["total"] for x in history])).all())
    record = EvidenceRecordV2(
        2,
        "conditional_identity_reanchor_training",
        config.run.evidence_tier,
        "passed" if finite else "failed",
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
            "smoke": file_sha256(args.smoke_evidence),
            "steps": len(history),
            "first_400_schedule_hash": schedule_hash,
        },
        (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),),
        ("evaluate",) if finite else (),
        None if finite else "reanchor training produced non-finite loss",
    )
    _write_evidence(run, "conditional_identity_reanchor_training_evidence.json", record)
    print(run.root)
    return 0
