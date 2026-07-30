# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from . import recovery as _recovery
from . import diversity as _diversity
from . import class_modes as _class_modes

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _diversity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _class_modes.__dict__.items() if not name.startswith("__")})

def conditional_parity_conflict(args, config: V2Config) -> int:
    """Read-only test that odd signed modes conflict with semantic gradients."""
    _require_parity(config)
    conflict = EvidenceRecordV2.from_dict(
        json.loads(Path(args.failed_smoke_evidence).read_text(encoding="utf-8"))
    )
    if (
        conflict.evidence_kind != "conditional_class_mode_recovery_smoke"
        or conflict.status != "failed"
    ):
        raise ValueError("parity conflict audit requires the valid failed v3 smoke")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, _ = _load_parity_source(config, args.checkpoint, device=device)
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    bundle.generator.class_modes.requires_grad_(True)
    bundle.generator.semantic_stabilizers.requires_grad_(True)
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    odd_cosines, even_cosines, rows = [], [], []
    before = state_content_hash(bundle.state_dict())
    for label in range(10):
        for direction in range(8):
            plus = torch.zeros(1, 8, device=device)
            plus[0, direction] = 1.0
            minus = -plus
            grads = []
            for code in (plus, minus):
                lab = torch.tensor([label], device=device)
                raw = raw_from_generator_batched(
                    bundle.system,
                    bundle.generator,
                    code,
                    lab,
                    microbatch_size=1,
                    activation_checkpoint=False,
                )
                _, logits = evaluator(
                    fixed_exposure_view(
                        raw, config.conditional_generation.fixed_exposure_scale
                    )
                )
                loss = torch.nn.functional.cross_entropy(logits, lab)
                gp, gq = torch.autograd.grad(
                    loss,
                    (
                        bundle.generator.class_modes,
                        bundle.generator.semantic_stabilizers,
                    ),
                    retain_graph=False,
                )
                grads.append(
                    (
                        gp[label, direction].detach().flatten(),
                        gq[label, direction].detach().flatten(),
                    )
                )
            odd_cosine = float(
                torch.nn.functional.cosine_similarity(
                    grads[0][0], grads[1][0], dim=0
                ).cpu()
            )
            even_cosine = float(
                torch.nn.functional.cosine_similarity(
                    grads[0][1], grads[1][1], dim=0
                ).cpu()
            )
            odd_cancel_pair = float(
                (
                    1.0
                    - torch.linalg.vector_norm(grads[0][0] + grads[1][0])
                    / (
                        torch.linalg.vector_norm(grads[0][0])
                        + torch.linalg.vector_norm(grads[1][0])
                    ).clamp_min(1.0e-12)
                ).cpu()
            )
            even_cancel_pair = float(
                (
                    1.0
                    - torch.linalg.vector_norm(grads[0][1] + grads[1][1])
                    / (
                        torch.linalg.vector_norm(grads[0][1])
                        + torch.linalg.vector_norm(grads[1][1])
                    ).clamp_min(1.0e-12)
                ).cpu()
            )
            odd_cosines.append(odd_cosine)
            even_cosines.append(even_cosine)
            rows.append(
                {
                    "label": label,
                    "direction": direction,
                    "odd_cosine": odd_cosine,
                    "even_cosine": even_cosine,
                    "odd_pair_cancellation": odd_cancel_pair,
                    "even_pair_cancellation": even_cancel_pair,
                }
            )
    after = state_content_hash(bundle.state_dict())
    conflict_count = sum(value <= -0.10 for value in odd_cosines)
    affected = {
        label: any(row["label"] == label and row["odd_cosine"] <= -0.10 for row in rows)
        for label in (5, 7)
    }
    odd_cancel = sum(row["odd_pair_cancellation"] for row in rows) / len(rows)
    even_cancel = sum(row["even_pair_cancellation"] for row in rows) / len(rows)
    checks = (
        GateResult(
            "finite_nonzero_gradients",
            all(torch.isfinite(torch.tensor(odd_cosines + even_cosines)))
            and all(abs(x) > 1.0e-8 for x in odd_cosines + even_cosines),
            {},
        ),
        GateResult(
            "odd_signed_conflict",
            conflict_count >= 20,
            {"count": float(conflict_count)},
        ),
        GateResult(
            "affected_labels_5_7",
            all(affected.values()),
            {str(k): v for k, v in affected.items()},
        ),
        GateResult(
            "even_pair_alignment", sum(x > 0.10 for x in even_cosines) >= 20, {}
        ),
        GateResult(
            "even_cancellation_improves",
            even_cancel + 0.15 <= odd_cancel,
            {"odd": odd_cancel, "even": even_cancel},
        ),
        GateResult("parameters_unchanged", before == after, {}),
    )
    run = _new_run(config)
    report = run.root / "conditional_parity_conflict_report.json"
    atomic_write_json(
        report,
        {
            "classification": "odd_even_semantic_conflict_confirmed"
            if all(x.passed for x in checks)
            else "parity_cause_unconfirmed",
            "rows": rows,
            "odd_cancellation": odd_cancel,
            "even_cancellation": even_cancel,
        },
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_parity_conflict",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "source_checkpoint": file_sha256(args.checkpoint),
            "failed_smoke": file_sha256(args.failed_smoke_evidence),
            "report": file_sha256(report),
        },
        checks,
        ("conditional_parity_mode_admission",) if passed else (),
        None if passed else "odd-even semantic conflict was not confirmed",
    )
    _write_evidence(run, "conditional_parity_conflict_evidence.json", record)
    print(run.root)
    return 0
def conditional_parity_mode_admission(args, config: V2Config) -> int:
    _require_parity(config)
    conflict = EvidenceRecordV2.from_dict(
        json.loads(Path(args.conflict_evidence).read_text(encoding="utf-8"))
    )
    if (
        conflict.evidence_kind != "conditional_parity_conflict"
        or conflict.status != "passed"
        or "conditional_parity_mode_admission" not in conflict.permitted_next_actions
    ):
        raise ValueError("parity admission requires passed conflict evidence")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, source_payload = _load_parity_source(config, args.checkpoint, device=device)
    source_hash = file_sha256(args.checkpoint)
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    panels = {
        "axis": (axis_codes, axis_labels),
        "uniform_1234": (_random_codes(170, device=device, seed=1234), axis_labels),
        "uniform_5678": (_random_codes(170, device=device, seed=5678), axis_labels),
    }
    initial = {
        name: raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
        for name, (codes, labels) in panels.items()
    }
    checks = [
        GateResult(
            "zero_stabilizers",
            bool(torch.count_nonzero(bundle.generator.semantic_stabilizers) == 0),
            {},
        ),
        GateResult(
            "finite_modes", bool(torch.isfinite(bundle.generator.class_modes).all()), {}
        ),
    ]
    run = _new_run(config)
    path = run.root / "conditional_parity_stabilized_initialization.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        PARITY_ARCHITECTURE,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        PARITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        None,
    )
    save_conditional_checkpoint(
        path,
        bundle,
        manifest,
        extra={
            "source_checkpoint": source_hash,
            "conflict_evidence": file_sha256(args.conflict_evidence),
            "semantic_stabilizers_zero": True,
            "initial_panel_hashes": {k: tensor_sha256(v) for k, v in initial.items()},
        },
    )
    out_hash = file_sha256(path)
    checks.append(GateResult("derived_checkpoint", True, {"checkpoint": out_hash}))
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_parity_mode_admission",
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
            "conflict_evidence": file_sha256(args.conflict_evidence),
            "initial_panel_hashes": {k: tensor_sha256(v) for k, v in initial.items()},
        },
        tuple(checks),
        ("conditional_parity_observability",) if passed else (),
        None if passed else "parity admission failed",
    )
    _write_evidence(run, "conditional_parity_mode_admission_evidence.json", record)
    print(run.root)
    return 0

def conditional_parity_observability(args, config: V2Config) -> int:
    _require_parity(config)
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.admission_evidence).read_text(encoding="utf-8"))
    )
    if (
        admission.evidence_kind != "conditional_parity_mode_admission"
        or admission.status != "passed"
        or "conditional_parity_observability" not in admission.permitted_next_actions
    ):
        raise ValueError("parity observability requires passed parity admission")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
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
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.generator.semantic_stabilizers.requires_grad_(True)
    indices = next(
        balanced_batches(
            train_labels,
            batch_size=config.conditional_generation.logical_batch_size,
            seed=1234,
        )
    )
    labels = train_labels.index_select(0, indices).to(device)
    codes = _reanchor_codes(labels, batch_number=0)
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=1,
        activation_checkpoint=False,
    )
    _, logits = evaluator(
        fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    )
    loss = torch.nn.functional.cross_entropy(logits, labels)
    grad = torch.autograd.grad(
        loss, bundle.generator.semantic_stabilizers, allow_unused=True
    )[0]
    nonzero = (
        [bool(torch.count_nonzero(grad[label]) > 0) for label in range(10)]
        if grad is not None
        else [False] * 10
    )
    checks = (
        GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}),
        GateResult("all_label_stabilizer_gradients", all(nonzero), {}),
        GateResult(
            "frozen_parameters",
            all(
                not p.requires_grad
                for n, p in bundle.named_parameters()
                if n != "generator.semantic_stabilizers"
            ),
            {},
        ),
        GateResult("no_optimizer_update", True, {}),
    )
    run = _new_run(config)
    report = run.root / "conditional_parity_observability_report.json"
    atomic_write_json(
        report,
        {
            "gradient_norm": float(grad.norm()) if grad is not None else 0.0,
            "per_label_nonzero": nonzero,
        },
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_parity_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "admission": file_sha256(args.admission_evidence),
            "report": file_sha256(report),
        },
        checks,
        ("parity_stabilizer_smoke",) if passed else (),
        None if passed else "parity observability failed",
    )
    _write_evidence(run, "conditional_parity_observability_evidence.json", record)
    print(run.root)
    return 0


def _parity_objective(
    *,
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    settings,
    config: V2Config,
) -> tuple[torch.Tensor, dict[str, float]]:
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
    total = sinkhorn + classification + relative + 2.0 * camera_loss + 0.1 * saturation
    return total, {
        "total": float(total.detach()),
        "sinkhorn": float(sinkhorn.detach()),
        "classification": float(classification.detach()),
        "relative_diversity": float(relative.detach()),
        "camera_fraction_loss": float(camera_loss.detach()),
        "camera_fraction": float(camera_code_fraction(raw, labels).detach()),
        "saturation": float(saturation.detach()),
    }


def parity_stabilizer_smoke(args, config: V2Config) -> int:
    _require_parity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    obs = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        obs,
        kind="conditional_parity_observability",
        action="parity_stabilizer_smoke",
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
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.generator.semantic_stabilizers.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [bundle.generator.semantic_stabilizers],
        lr=config.conditional_generation.generator_learning_rate,
    )
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    panels = {
        "axis": (axis_codes, axis_labels),
        "uniform_1234": (_random_codes(170, device=device, seed=1234), axis_labels),
        "uniform_5678": (_random_codes(170, device=device, seed=5678), axis_labels),
    }
    real_features = _panel_real_features(evaluator, train_images, train_labels, device)
    initial_state = {n: v.detach().clone() for n, v in bundle.state_dict().items()}
    zero_raw = (
        raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            torch.zeros(10, 8, device=device),
            torch.arange(10, device=device),
            microbatch_size=1,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
    )

    def probe():
        metrics, raws = {}, {}
        with torch.no_grad():
            for name, (codes, labels) in panels.items():
                raw = raw_from_generator_batched(
                    bundle.system,
                    bundle.generator,
                    codes,
                    labels,
                    microbatch_size=1,
                    activation_checkpoint=False,
                )
                metrics[name] = _reanchor_panel_metrics(
                    raw,
                    labels,
                    evaluator,
                    real_features,
                    config.conditional_generation.fixed_exposure_scale,
                )
                raws[name] = raw.detach().cpu()
        return metrics, raws

    probe_history, raw_history = {}, {}
    metrics, raws = probe()
    probe_history["step_0"], raw_history[0] = metrics, raws
    loss_history, updates = [], []
    for step, indices in zip(
        range(400),
        balanced_batches(
            train_labels,
            batch_size=config.conditional_generation.logical_batch_size,
            seed=1234,
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _parity_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=config.conditional_generation,
            config=config,
        )
        loss.backward()
        before = bundle.generator.semantic_stabilizers.detach().clone()
        optimizer.step()
        updates.append(
            float(
                torch.linalg.vector_norm(
                    bundle.generator.semantic_stabilizers.detach() - before
                )
            )
        )
        loss_history.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            metrics, raws = probe()
            probe_history[f"step_{step + 1}"], raw_history[step + 1] = metrics, raws
    final = probe_history["step_400"]
    cls_first = sum(x["classification"] for x in loss_history[:50]) / 50
    cls_last = sum(x["classification"] for x in loss_history[-50:]) / 50
    frozen = all(
        torch.equal(initial_state[n].cpu(), v.detach().cpu())
        for n, v in bundle.state_dict().items()
        if n != "generator.semantic_stabilizers"
    )
    zero_after = (
        raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            torch.zeros(10, 8, device=device),
            torch.arange(10, device=device),
            microbatch_size=1,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
    )
    checks = []
    for name, amin, rmin, dmin in (
        ("axis", 0.95, 0.90, 0.60),
        ("uniform_1234", 0.90, 0.80, 0.70),
        ("uniform_5678", 0.90, 0.80, 0.70),
    ):
        m = final[name]
        checks.extend(
            (
                GateResult(
                    f"{name}_accuracy",
                    m["accuracy"] >= amin,
                    {"accuracy": m["accuracy"]},
                ),
                GateResult(f"{name}_recall", min(m["recalls"]) >= rmin, {}),
                GateResult(f"{name}_code_fraction", m["code_fraction"] >= 0.25, {}),
                GateResult(f"{name}_diversity", min(m["diversity_ratios"]) >= dmin, {}),
                GateResult(f"{name}_saturation", m["saturation"] <= 0.08, {}),
            )
        )
    checks.extend(
        (
            GateResult(
                "zero_code_unchanged", bool(torch.equal(zero_raw, zero_after)), {}
            ),
            GateResult("frozen_odd_and_optics", frozen, {}),
            GateResult("stabilizer_updates", all(x > 0 for x in updates), {}),
            GateResult(
                "classification_progress",
                cls_last <= 0.85 * cls_first,
                {"first_50": cls_first, "last_50": cls_last},
            ),
            GateResult(
                "finite_losses",
                bool(
                    torch.isfinite(
                        torch.tensor([x["total"] for x in loss_history])
                    ).all()
                ),
                {},
            ),
        )
    )
    run = _new_run(config)
    history_path = run.root / "parity_stabilizer_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": PARITY_SMOKE_CONTRACT,
            "probes": probe_history,
            "loss": loss_history,
            "updates": updates,
            "first_400_schedule_hash": _reanchor_schedule_hash(train_labels),
        },
    )
    diagnostic = run.root / "parity_stabilizer_smoke_final.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        PARITY_ARCHITECTURE,
        init_manifest.system_signature,
        init_manifest.config_hash,
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        PARITY_GENERATOR_CONTRACT,
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
            "contract": PARITY_SMOKE_CONTRACT,
        },
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_parity_stabilizer_smoke",
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
            "contract": PARITY_SMOKE_CONTRACT,
        },
        tuple(checks),
        ("train_parity_stabilizer",) if passed else (),
        None if passed else "parity stabilizer smoke failed",
    )
    _write_evidence(run, "conditional_parity_stabilizer_smoke_evidence.json", record)
    print(run.root)
    return 0


def train_parity_stabilizer(args, config: V2Config) -> int:
    _require_parity(config)
    smoke = _passed_evidence(
        args.smoke_evidence, kind="conditional_parity_stabilizer_smoke"
    )
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    require_evidence_action(
        smoke,
        kind="conditional_parity_stabilizer_smoke",
        action="train_parity_stabilizer",
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
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.generator.semantic_stabilizers.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [bundle.generator.semantic_stabilizers],
        lr=config.conditional_generation.generator_learning_rate,
    )
    history = []
    steps = _balanced_epoch_steps(
        train_labels, config.conditional_generation.logical_batch_size
    )
    for step, indices in enumerate(
        balanced_batches(
            train_labels,
            batch_size=config.conditional_generation.logical_batch_size,
            seed=1234,
        )
    ):
        if step >= steps:
            break
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary = _parity_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=config.conditional_generation,
            config=config,
        )
        loss.backward()
        optimizer.step()
        history.append(summary | {"step": float(step + 1)})
    run = _new_run(config)
    path = run.root / "conditional_parity_stabilized_generator.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        PARITY_ARCHITECTURE,
        init_manifest.system_signature,
        init_manifest.config_hash,
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        PARITY_GENERATOR_CONTRACT,
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
            "history": history,
            "smoke_evidence": file_sha256(args.smoke_evidence),
            "steps": len(history),
            "first_400_schedule_hash": _reanchor_schedule_hash(train_labels),
        },
    )
    out_hash = file_sha256(path)
    finite = bool(torch.isfinite(torch.tensor([x["total"] for x in history])).all())
    record = EvidenceRecordV2(
        2,
        "conditional_parity_stabilizer_training",
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
        },
        (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),),
        ("evaluate",) if finite else (),
        None if finite else "parity training produced non-finite loss",
    )
    _write_evidence(run, "conditional_parity_stabilizer_training_evidence.json", record)
    print(run.root)
    return 0


def class_mode_recovery_smoke(args, config: V2Config) -> int:
    _require_class_modes(config)
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
    observability = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        observability,
        kind="conditional_class_mode_observability",
        action="class_mode_recovery_smoke",
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
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    bundle.generator.class_modes.requires_grad_(True)
    settings = config.conditional_generation
    assert settings is not None
    optimizer = torch.optim.Adam(
        [bundle.generator.class_modes], lr=settings.generator_learning_rate
    )
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    panels = {
        "axis": (axis_codes, axis_labels),
        "uniform_1234": (_random_codes(170, device=device, seed=1234), axis_labels),
        "uniform_5678": (_random_codes(170, device=device, seed=5678), axis_labels),
    }
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
    probe_history, raw_history, loss_history, update_history = {}, {}, [], []

    def probe():
        metrics, raws = {}, {}
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
                metrics[name] = _reanchor_panel_metrics(
                    raw, labels, evaluator, real_features, settings.fixed_exposure_scale
                )
                raws[name] = raw.detach().cpu()
        return metrics, raws

    metrics, raws = probe()
    probe_history["step_0"], raw_history[0] = metrics, raws
    for step, indices in zip(
        range(400),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _class_mode_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=settings,
            config=config,
        )
        loss.backward()
        before = bundle.generator.class_modes.detach().clone()
        optimizer.step()
        update_history.append(
            float(
                torch.linalg.vector_norm(bundle.generator.class_modes.detach() - before)
            )
        )
        loss_history.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            metrics, raws = probe()
            probe_history[f"step_{step + 1}"], raw_history[step + 1] = metrics, raws
    final = probe_history["step_400"]
    cls0 = sum(row["classification"] for row in loss_history[:50]) / 50
    cls1 = sum(row["classification"] for row in loss_history[-50:]) / 50
    frozen = all(
        torch.equal(initial_state[name].cpu(), value.detach().cpu())
        for name, value in bundle.state_dict().items()
        if name != "generator.class_modes"
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
    checks = []
    for name, minimum_accuracy, minimum_recall, minimum_ratio in (
        ("axis", 0.95, 0.90, 0.60),
        ("uniform_1234", 0.90, 0.80, 0.70),
        ("uniform_5678", 0.90, 0.80, 0.70),
    ):
        metrics = final[name]
        checks.extend(
            (
                GateResult(
                    f"{name}_accuracy",
                    metrics["accuracy"] >= minimum_accuracy,
                    {"accuracy": metrics["accuracy"]},
                ),
                GateResult(
                    f"{name}_recall", min(metrics["recalls"]) >= minimum_recall, {}
                ),
                GateResult(
                    f"{name}_code_fraction", metrics["code_fraction"] >= 0.25, {}
                ),
                GateResult(
                    f"{name}_diversity",
                    min(metrics["diversity_ratios"]) >= minimum_ratio,
                    {},
                ),
                GateResult(f"{name}_saturation", metrics["saturation"] <= 0.08, {}),
            )
        )
    checks.extend(
        (
            GateResult(
                "rank_eight_per_label",
                all(
                    int(
                        torch.linalg.matrix_rank(
                            bundle.generator.normalized_modes(
                                bundle.generator.class_modes
                            )[label].flatten(1)
                        ).item()
                    )
                    == 8
                    for label in range(10)
                ),
                {},
            ),
            GateResult(
                "zero_code_unchanged", bool(torch.equal(zero_raw, zero_after)), {}
            ),
            GateResult("frozen_parameters", frozen, {}),
            GateResult(
                "classification_progress",
                cls1 <= 0.85 * cls0,
                {"first_50": cls0, "last_50": cls1},
            ),
            GateResult(
                "finite_losses",
                bool(
                    torch.isfinite(
                        torch.tensor([row["total"] for row in loss_history])
                    ).all()
                ),
                {},
            ),
            GateResult(
                "class_mode_updates", all(value > 0 for value in update_history), {}
            ),
        )
    )
    run = _new_run(config)
    artifacts = {}
    for step, panel_raws in raw_history.items():
        artifacts[str(step)] = {}
        for name, raw in panel_raws.items():
            raw_path = run.root / f"class_mode_{name}_step_{step}_raw_cmos.pt"
            atomic_torch_save(raw_path, raw)
            preview = write_review_artifacts(
                root=run.root,
                stem=f"class_mode_{name}_step_{step}",
                kind="conditional_class_mode_recovery_smoke",
                raw_cmos=raw,
                labels=axis_labels.cpu(),
                exposure_scale=settings.fixed_exposure_scale,
            )
            artifacts[str(step)][name] = {
                "raw": file_sha256(raw_path),
                "preview": preview.preview_sha256,
            }
    schedule_hash = _reanchor_schedule_hash(train_labels)
    history_path = run.root / "class_mode_recovery_smoke_history.json"
    atomic_write_json(
        history_path,
        {
            "contract": CLASS_MODE_SMOKE_CONTRACT,
            "probes": probe_history,
            "loss": loss_history,
            "updates": update_history,
            "artifacts": artifacts,
            "first_400_schedule_hash": schedule_hash,
        },
    )
    diagnostic = run.root / "class_mode_recovery_smoke_final.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        CLASS_MODE_ARCHITECTURE,
        init_manifest.system_signature,
        init_manifest.config_hash,
        split.manifest_hash,
        evaluator_hash,
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        CLASS_MODE_GENERATOR_CONTRACT,
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
            "contract": CLASS_MODE_SMOKE_CONTRACT,
            "first_400_schedule_hash": schedule_hash,
        },
    )
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_class_mode_recovery_smoke",
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
            "contract": CLASS_MODE_SMOKE_CONTRACT,
            "first_400_schedule_hash": schedule_hash,
        },
        tuple(checks),
        ("train_class_modes",) if passed else (),
        None if passed else "class-mode recovery smoke failed",
    )
    _write_evidence(run, "conditional_class_mode_recovery_smoke_evidence.json", record)
    print(run.root)
    return 0


def train_class_modes(args, config: V2Config) -> int:
    _require_class_modes(config)
    smoke = _passed_evidence(
        args.smoke_evidence, kind="conditional_class_mode_recovery_smoke"
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
        kind="conditional_class_mode_recovery_smoke",
        action="train_class_modes",
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
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    bundle.generator.class_modes.requires_grad_(True)
    settings = config.conditional_generation
    assert settings is not None
    optimizer = torch.optim.Adam(
        [bundle.generator.class_modes], lr=settings.generator_learning_rate
    )
    history = []
    for batch_number, indices in enumerate(
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        )
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _reanchor_codes(labels, batch_number=batch_number)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _class_mode_objective(
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
    path = run.root / "conditional_class_mode_generator.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        CLASS_MODE_ARCHITECTURE,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_hash,
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        CLASS_MODE_GENERATOR_CONTRACT,
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
            "smoke_evidence": file_sha256(args.smoke_evidence),
            "steps": len(history),
            "first_400_schedule_hash": schedule_hash,
        },
    )
    out_hash = file_sha256(path)
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    record = EvidenceRecordV2(
        2,
        "conditional_class_mode_training",
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
        None if finite else "class-mode training produced non-finite loss",
    )
    _write_evidence(run, "conditional_class_mode_training_evidence.json", record)
    print(run.root)
    return 0
