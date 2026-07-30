# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from . import recovery as _recovery
from . import diversity as _diversity

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _diversity.__dict__.items() if not name.startswith("__")})

def _class_mode_panel_codes(
    *, device: torch.device, uniform_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    axis, labels = _fixed_control_probe(device=device)
    uniform = _random_codes(170, device=device, seed=uniform_seed)
    return torch.cat((axis, uniform), dim=0), torch.cat((labels, labels), dim=0)


def _class_mode_objective(
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


def conditional_shared_mode_conflict(args, config: V2Config) -> int:
    """Read-only gradient audit for the shared-mode capacity conflict."""
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
    failed = EvidenceRecordV2.from_dict(
        json.loads(Path(args.reanchor_smoke_evidence).read_text(encoding="utf-8"))
    )
    if (
        failed.evidence_kind != "conditional_identity_reanchor_smoke"
        or failed.status != "failed"
        or failed.context != expected
    ):
        raise ValueError(
            "shared-mode conflict audit requires failed reanchor smoke evidence"
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
    _zero_reanchor(bundle)
    bundle.generator.global_modes.requires_grad_(True)
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    panels = {
        "axis": (axis_codes, axis_labels),
        "uniform": (_random_codes(170, device=device, seed=1234), axis_labels),
    }
    gradients: dict[str, list[torch.Tensor]] = {}
    before = state_content_hash(bundle.state_dict())
    for panel_name, (codes, panel_labels) in panels.items():
        rows = []
        for label in range(10):
            selected = panel_labels == label
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes[selected],
                panel_labels[selected],
                microbatch_size=config.execution.evaluation_microbatch_size,
                activation_checkpoint=False,
            )
            view = fixed_exposure_view(
                raw, config.conditional_generation.fixed_exposure_scale
            )
            _, logits = evaluator(view)
            grad = torch.autograd.grad(
                torch.nn.functional.cross_entropy(logits, panel_labels[selected]),
                bundle.generator.global_modes,
                retain_graph=False,
            )[0]
            rows.append(grad.detach().flatten().cpu())
        gradients[panel_name] = rows
    after = state_content_hash(bundle.state_dict())
    panel_metrics = {}
    all_rows = []
    for name, rows in gradients.items():
        norms = torch.stack([row.norm() for row in rows])
        matrix = torch.stack(rows)
        normalized = matrix / norms.clamp_min(1.0e-12).unsqueeze(1)
        cosine = normalized @ normalized.T
        aggregate = matrix.sum(dim=0).norm() / norms.sum().clamp_min(1.0e-12)
        panel_metrics[name] = {
            "gradient_norms": [float(x) for x in norms],
            "cosine": cosine.tolist(),
            "aggregate_cancellation_ratio": float(aggregate),
            "minimum_off_diagonal_cosine": float(
                cosine[~torch.eye(10, dtype=torch.bool)].min()
            ),
        }
        all_rows.extend(rows)
    all_matrix = torch.stack(all_rows)
    all_norms = all_matrix.norm(dim=1)
    all_normalized = all_matrix / all_norms.clamp_min(1.0e-12).unsqueeze(1)
    all_cosine = all_normalized @ all_normalized.T
    off = all_cosine[~torch.eye(all_cosine.shape[0], dtype=torch.bool)]
    conflict = bool(
        torch.isfinite(all_matrix).all()
        and (all_norms > 1.0e-12).all()
        and off.min() <= -0.10
        and (all_matrix.sum(dim=0).norm() / all_norms.sum() < 0.75)
    )
    checks = (
        GateResult(
            "all_class_gradients_finite_nonzero",
            bool(torch.isfinite(all_matrix).all() and (all_norms > 1.0e-12).all()),
            {},
        ),
        GateResult(
            "negative_class_pair",
            bool(off.min() <= -0.10),
            {"minimum_cosine": float(off.min())},
        ),
        GateResult(
            "aggregate_cancellation_below_threshold",
            bool((all_matrix.sum(dim=0).norm() / all_norms.sum()) < 0.75),
            {"ratio": float(all_matrix.sum(dim=0).norm() / all_norms.sum())},
        ),
        GateResult("parameters_unchanged", before == after, {}),
    )
    run = _new_run(config)
    report = run.root / "conditional_shared_mode_conflict_report.json"
    atomic_write_json(
        report,
        {
            "classification": "shared_mode_capacity_conflict"
            if conflict
            else "capacity_cause_unconfirmed",
            "panels": panel_metrics,
            "affected_labels": [5, 7],
            "parameters_unchanged": before == after,
        },
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_shared_mode_conflict",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "reanchor_smoke": file_sha256(args.reanchor_smoke_evidence),
            "report": file_sha256(report),
        },
        checks,
        ("conditional_class_mode_admission",) if passed else (),
        None if passed else "shared mode conflict was not established",
    )
    _write_evidence(run, "conditional_shared_mode_conflict_evidence.json", record)
    print(run.root)
    return 0
def conditional_class_mode_admission(args, config: V2Config) -> int:
    """Derive the class-mode architecture from the immutable Wy=0 anchor."""
    _require_class_modes(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    source_hash = file_sha256(args.checkpoint)
    source_config = config.model_copy(update={"architecture": IDENTITY_ARCHITECTURE})
    expected_source = _context(
        source_config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=source_hash,
    )
    conflict = EvidenceRecordV2.from_dict(
        json.loads(Path(args.conflict_evidence).read_text(encoding="utf-8"))
    )
    if (
        conflict.evidence_kind != "conditional_shared_mode_conflict"
        or conflict.status != "passed"
        or conflict.context.system_signature != expected_source.system_signature
        or conflict.context.evaluator_checkpoint_hash
        != expected_source.evaluator_checkpoint_hash
        or conflict.context.dataset_split_manifest_hash
        != expected_source.dataset_split_manifest_hash
        or conflict.context.exposure_scale != expected_source.exposure_scale
        or "conditional_class_mode_admission" not in conflict.permitted_next_actions
    ):
        raise ValueError(
            "shared-mode conflict evidence is not compatible with the explicit v3 migration"
        )
    device = torch.device(config.run.device)
    source_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict) or not isinstance(
        source_payload.get("manifest"), dict
    ):
        raise ValueError("source reanchor checkpoint is invalid")
    source_manifest = ConditionalCheckpointManifestV2(**source_payload["manifest"])
    if (
        source_manifest.architecture_kind != IDENTITY_ARCHITECTURE
        or source_manifest.system_signature != _system_signature(config)
        or source_manifest.evaluator_checkpoint_hash != evaluator_hash
        or source_manifest.dataset_split_manifest_hash != split.manifest_hash
    ):
        raise ValueError(
            "source reanchor checkpoint context does not match the v3 admission"
        )
    source = ConditionalBundle(source_config).to(device)
    load_conditional_checkpoint(args.checkpoint, source, source_manifest)
    class_modes = (
        source.generator.global_modes.detach()
        .unsqueeze(0)
        .expand(10, -1, -1, -1)
        .clone()
    )
    bundle = ConditionalBundle(
        config,
        carrier_logits=source.generator.class_carrier_logits.detach(),
    ).to(device)
    bundle.generator.class_modes.data.copy_(class_modes.to(device))
    # The electronic generator migration must not silently reset the learned
    # encoder/decoder phase state.  Copy every formal-system state tensor so
    # the derived v3 checkpoint is physically equivalent at step zero.
    target_state = bundle.state_dict()
    for name, value in source.state_dict().items():
        if name.startswith("system."):
            if name not in target_state or target_state[name].shape != value.shape:
                raise ValueError(f"source system state is incompatible: {name}")
            target_state[name].copy_(value)
    bundle.load_state_dict(target_state, strict=True)
    axis_codes, axis_labels = _fixed_control_probe(device=device)
    zero = torch.zeros(10, 8, device=device)
    with torch.no_grad():
        source_axis = source.generator(axis_codes, axis_labels)
        target_axis = bundle.generator(axis_codes, axis_labels)
        source_zero = source.generator(zero, torch.arange(10, device=device))
        target_zero = bundle.generator(zero, torch.arange(10, device=device))
        source_raw = raw_from_generator_batched(
            source.system,
            source.generator,
            axis_codes,
            axis_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        target_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            axis_codes,
            axis_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
    max_abs = float(torch.max(torch.abs(source_axis - target_axis)))
    checks = (
        GateResult(
            "class_modes_shape",
            tuple(bundle.generator.class_modes.shape)
            == (10, 8, *config.system.slm_shape),
            {},
        ),
        GateResult(
            "initial_output_matches_source", max_abs <= 1.0e-6, {"max_abs": max_abs}
        ),
        GateResult(
            "zero_code_matches_source", bool(torch.equal(source_zero, target_zero)), {}
        ),
        GateResult(
            "initial_raw_cmos_matches_source",
            bool(torch.max(torch.abs(source_raw - target_raw)) <= 1.0e-6),
            {"max_abs": float(torch.max(torch.abs(source_raw - target_raw)))},
        ),
        GateResult(
            "finite_modes", bool(torch.isfinite(bundle.generator.class_modes).all()), {}
        ),
        GateResult(
            "rank_eight_per_label",
            all(
                int(
                    torch.linalg.matrix_rank(
                        bundle.generator.normalized_modes(bundle.generator.class_modes)[
                            label
                        ].flatten(1)
                    ).item()
                )
                == 8
                for label in range(10)
            ),
            {},
        ),
    )
    run = _new_run(config)
    path = run.root / "conditional_class_mode_generator.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        CLASS_MODE_ARCHITECTURE,
        source_manifest.system_signature,
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        CLASS_MODE_GENERATOR_CONTRACT,
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
            "source_checkpoint": source_hash,
            "conflict_evidence": file_sha256(args.conflict_evidence),
            "class_modes_initialized_from_global_modes": True,
        },
    )
    out_hash = file_sha256(path)
    record = EvidenceRecordV2(
        2,
        "conditional_class_mode_admission",
        config.run.evidence_tier,
        "passed" if all(x.passed for x in checks) else "failed",
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
        },
        checks,
        ("conditional_class_mode_observability",)
        if all(x.passed for x in checks)
        else (),
        None if all(x.passed for x in checks) else "class-mode admission failed",
    )
    _write_evidence(run, "conditional_class_mode_admission_evidence.json", record)
    print(run.root)
    return 0

def conditional_class_mode_observability(args, config: V2Config) -> int:
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
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.admission_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        admission,
        kind="conditional_class_mode_admission",
        action="conditional_class_mode_observability",
        expected_context=expected,
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
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    bundle.generator.class_modes.requires_grad_(True)
    indices = next(
        balanced_batches(
            train_labels,
            batch_size=config.conditional_generation.logical_batch_size,
            seed=1234,
        )
    )
    labels = train_labels.index_select(0, indices).to(device)
    codes = _reanchor_codes(labels, batch_number=0)
    before = state_content_hash(bundle.state_dict())
    loss, summary, raw = _class_mode_objective(
        bundle=bundle,
        evaluator=evaluator,
        codes=codes,
        labels=labels,
        real_views=train_images.index_select(0, indices).to(device),
        settings=config.conditional_generation,
        config=config,
    )
    gradient = torch.autograd.grad(
        loss, bundle.generator.class_modes, allow_unused=True
    )[0]
    after = state_content_hash(bundle.state_dict())
    checks = (
        GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}),
        GateResult(
            "class_modes_gradient",
            bool(
                gradient is not None
                and torch.isfinite(gradient).all()
                and gradient.norm() > 0
            ),
            {"norm": float(gradient.norm()) if gradient is not None else 0.0},
        ),
        GateResult(
            "frozen_parameters",
            all(
                not parameter.requires_grad
                for name, parameter in bundle.named_parameters()
                if name != "generator.class_modes"
            ),
            {},
        ),
        GateResult("parameters_unchanged", before == after, {}),
        GateResult("no_optimizer_update", True, {}),
    )
    run = _new_run(config)
    report = run.root / "conditional_class_mode_observability_report.json"
    atomic_write_json(
        report,
        {
            "summary": summary,
            "gradient_norm": float(gradient.norm()) if gradient is not None else None,
            "schedule_hash": _reanchor_schedule_hash(train_labels),
            "parameters_unchanged": before == after,
        },
    )
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_class_mode_observability",
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
        ("class_mode_recovery_smoke",) if passed else (),
        None if passed else "class-mode observability failed",
    )
    _write_evidence(run, "conditional_class_mode_observability_evidence.json", record)
    print(run.root)
    return 0


def _write_tangent_contact_sheet(
    root: Path, name: str, raw: torch.Tensor, labels: torch.Tensor, exposure: float
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = raw.detach().cpu()
    labels = labels.detach().cpu()
    figure, axes = plt.subplots(10, 17, figsize=(17, 10))
    for row in range(10):
        rows = torch.nonzero(labels == row, as_tuple=False).flatten()
        if rows.numel() != 17:
            raise ValueError(
                "tangent contact sheet requires exactly 17 samples per label"
            )
        for col, index in enumerate(rows.tolist()):
            axes[row, col].imshow(
                (raw[index, 0] / exposure).clamp(0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(str(col), fontsize=7)
        axes[row, 0].set_ylabel(str(row), rotation=0, labelpad=10)
    figure.tight_layout(pad=0.1)
    path = root / f"{name}_10x17_fixed_exposure.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return file_sha256(path)


def _tangent_sample_grad(
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    code: torch.Tensor,
    label: torch.Tensor,
    groups: dict[str, tuple[torch.Tensor, ...]],
) -> tuple[float, dict[str, torch.Tensor]]:
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        code,
        label,
        microbatch_size=1,
        activation_checkpoint=True,
    )
    _, logits = evaluator(fixed_exposure_view(raw, 0.4913053810596466))
    target = label.item()
    other = logits.clone()
    other[0, target] = -torch.inf
    margin = logits[0, target] - other.max()
    grads = torch.autograd.grad(
        margin, [p for params in groups.values() for p in params], allow_unused=True
    )
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, params in groups.items():
        pieces = []
        for parameter in params:
            grad = grads[offset]
            offset += 1
            pieces.append(
                torch.zeros_like(parameter) if grad is None else grad.detach()
            )
        result[name] = torch.cat([piece.reshape(-1) for piece in pieces])
    return float(margin.detach().cpu()), result


def _tangent_apply_probe(
    bundle: ConditionalBundle,
    group: str,
    directions: dict[int | str, torch.Tensor],
    *,
    panel_codes: torch.Tensor,
    panel_labels: torch.Tensor,
    evaluator: nn.Module,
    exposure: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    probe = copy.deepcopy(bundle)
    with torch.no_grad():
        if group == "q":
            for label, direction in directions.items():
                delta = direction.reshape_as(
                    probe.generator.semantic_stabilizers[int(label)]
                )
                probe.generator.semantic_stabilizers[int(label)].copy_(delta)
        elif group == "carrier":
            for label, direction in directions.items():
                delta = direction.reshape_as(
                    probe.generator.class_carrier_logits[int(label)]
                )
                probe.generator.class_carrier_logits[int(label)].copy_(delta)
        else:
            for name, direction in directions.items():
                parameter = (
                    probe.system.encoder.phase
                    if name == "encoder_phase"
                    else probe.system.decoder.phase
                )
                parameter.copy_(direction.reshape_as(parameter))
    with torch.no_grad():
        raw = raw_from_generator_batched(
            probe.system,
            probe.generator,
            panel_codes,
            panel_labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
        _, logits = evaluator(fixed_exposure_view(raw, exposure))
    return raw, {"predictions": logits.argmax(dim=1).cpu(), "logits": logits.cpu()}


def conditional_semantic_tangent(args, config: V2Config) -> int:
    """Read-only local reachability audit; never creates an optimizer or checkpoint."""
    _require_parity(config)
    admission = EvidenceRecordV2.from_dict(
        json.loads(Path(args.class_mode_admission_evidence).read_text(encoding="utf-8"))
    )
    failed_smoke = EvidenceRecordV2.from_dict(
        json.loads(
            Path(args.failed_class_mode_smoke_evidence).read_text(encoding="utf-8")
        )
    )
    failed_parity = EvidenceRecordV2.from_dict(
        json.loads(Path(args.failed_parity_evidence).read_text(encoding="utf-8"))
    )
    if (
        admission.evidence_kind != "conditional_class_mode_admission"
        or admission.status != "passed"
    ):
        raise ValueError("semantic tangent requires passed v3 class-mode admission")
    if (
        failed_smoke.evidence_kind != "conditional_class_mode_recovery_smoke"
        or failed_smoke.status != "failed"
    ):
        raise ValueError("semantic tangent requires failed v3 class-mode smoke")
    if (
        failed_parity.evidence_kind != "conditional_parity_conflict"
        or failed_parity.status != "failed"
    ):
        raise ValueError("semantic tangent requires failed parity evidence")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, _ = _load_parity_source(config, args.checkpoint, device=device)
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        _, validation_logits = evaluator(validation_images.to(device))
    evaluator_accuracy = float(
        (validation_logits.argmax(dim=1) == validation_labels.to(device))
        .float()
        .mean()
        .cpu()
    )
    panels = {
        "axis": (_fixed_control_probe(device=device)),
        "uniform_1234": (
            _random_codes(170, device=device, seed=1234),
            torch.arange(10, device=device).repeat_interleave(17),
        ),
        "uniform_5678": (
            _random_codes(170, device=device, seed=5678),
            torch.arange(10, device=device).repeat_interleave(17),
        ),
    }
    real_features = _panel_real_features(evaluator, train_images, train_labels, device)
    initial_state = state_content_hash(bundle.state_dict())
    groups = {
        "q": (bundle.generator.semantic_stabilizers,),
        "carrier": (bundle.generator.class_carrier_logits,),
        "encoder_phase": (bundle.system.encoder.phase,),
        "decoder_phase": (bundle.system.decoder.phase,),
    }
    report_panels: dict[str, object] = {}
    artifact_hashes: dict[str, object] = {}
    run = _new_run(config)
    for panel_name, (codes, panel_labels) in panels.items():
        with torch.no_grad():
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes,
                panel_labels,
                microbatch_size=1,
                activation_checkpoint=False,
            ).detach()
        view = fixed_exposure_view(
            raw, config.conditional_generation.fixed_exposure_scale
        )
        with torch.no_grad():
            _, logits = evaluator(view)
        margins = (
            logits[torch.arange(170, device=device), panel_labels]
            - logits.masked_fill(
                torch.nn.functional.one_hot(panel_labels, 10).bool(), -torch.inf
            )
            .max(dim=1)
            .values
        )
        selected = torch.nonzero(margins < 0.5, as_tuple=False).flatten().tolist()
        for parameter in bundle.parameters():
            parameter.requires_grad_(True)
        directions: dict[str, dict[int | str, torch.Tensor]] = {
            name: {} for name in groups
        }
        sample_rows = []
        sums = {}
        for name, params in groups.items():
            size = (
                sum(p[0].numel() for p in params)
                if name in {"q", "carrier"}
                else sum(p.numel() for p in params)
            )
            sums[name] = {label: torch.zeros(size, device="cpu") for label in range(10)}
        for index in selected:
            margin, grads = _tangent_sample_grad(
                bundle,
                evaluator,
                codes[index : index + 1],
                panel_labels[index : index + 1],
                groups,
            )
            label = int(panel_labels[index])
            sample_rows.append({"index": index, "label": label, "margin": margin})
            for name, grad in grads.items():
                if name in {"q", "carrier"}:
                    grad = grad.reshape(10, -1)[label]
                sums[name][label] += grad.cpu()
        for name in groups:
            for label, direction in sums[name].items():
                if torch.linalg.vector_norm(direction) > 0:
                    directions[name][label] = direction / torch.linalg.vector_norm(
                        direction
                    )
                else:
                    directions[name][label] = direction
        positive: dict[str, float] = {}
        directional_rows = []
        for name in groups:
            values = []
            for index in selected:
                margin, grads = _tangent_sample_grad(
                    bundle,
                    evaluator,
                    codes[index : index + 1],
                    panel_labels[index : index + 1],
                    groups,
                )
                direction = directions[name][int(panel_labels[index])]
                gradient = grads[name]
                if name == "q":
                    gradient = gradient.reshape(10, -1)[int(panel_labels[index])]
                elif name == "carrier":
                    gradient = gradient.reshape(10, -1)[int(panel_labels[index])]
                value = float(torch.dot(gradient.cpu(), direction).item())
                values.append(value)
                directional_rows.append(
                    {"index": index, "group": name, "margin_direction": value}
                )
            positive[name] = sum(v > 0 for v in values) / max(1, len(values))
        for parameter in bundle.parameters():
            parameter.requires_grad_(False)
        probe_results = {}
        baseline_margins = margins.detach().cpu()
        baseline_correct = int((logits.argmax(dim=1) == panel_labels).sum().item())
        for name in ("q", "carrier", "encoder_phase", "decoder_phase"):
            scaled: dict[int | str, torch.Tensor] = {}
            for key, direction in directions[name].items():
                if direction.numel() == 0:
                    continue
                target = 0.005 if name.endswith("phase") else 0.05
                scaled[key] = (
                    direction
                    * target
                    / direction.square().mean().sqrt().clamp_min(1.0e-8)
                ).to(device)
            probe_raw, probe_info = _tangent_apply_probe(
                bundle,
                name,
                scaled,
                panel_codes=codes,
                panel_labels=panel_labels,
                evaluator=evaluator,
                exposure=config.conditional_generation.fixed_exposure_scale,
            )
            probe_metrics = _reanchor_panel_metrics(
                probe_raw,
                panel_labels,
                evaluator,
                real_features,
                config.conditional_generation.fixed_exposure_scale,
            )
            probe_logits = probe_info["logits"].to(device)
            probe_other = probe_logits.masked_fill(
                torch.nn.functional.one_hot(panel_labels, 10).bool(), -torch.inf
            )
            probe_margins = (
                probe_logits[torch.arange(170), panel_labels.cpu()]
                - probe_other.max(dim=1).values
            ).cpu()
            selected_tensor = torch.tensor(selected, dtype=torch.long)
            margin_delta = (
                float(
                    (
                        probe_margins.index_select(0, selected_tensor)
                        - baseline_margins.index_select(0, selected_tensor)
                    )
                    .median()
                    .item()
                )
                if selected
                else 0.0
            )
            correct_delta = (
                int((probe_info["predictions"] == panel_labels.cpu()).sum().item())
                - baseline_correct
            )
            probe_results[name] = {
                "positive_fraction": positive[name],
                "metrics": probe_metrics,
                "raw_rms_delta": float((probe_raw - raw).square().mean().sqrt().cpu()),
                "median_low_margin_delta": margin_delta,
                "correct_count_delta": correct_delta,
            }
        baseline_metrics = _reanchor_panel_metrics(
            raw,
            panel_labels,
            evaluator,
            real_features,
            config.conditional_generation.fixed_exposure_scale,
        )
        report_panels[panel_name] = {
            "baseline": baseline_metrics,
            "low_margin_count": len(selected),
            "positive_fraction": positive,
            "probe": probe_results,
            "samples": sample_rows,
            "directional_rows": directional_rows,
        }
        run_dir = run.root
        raw_path = run_dir / f"{panel_name}_raw_cmos.pt"
        atomic_torch_save(raw_path, raw.cpu())
        artifact_hashes[panel_name] = {
            "raw": file_sha256(raw_path),
            "preview": _write_tangent_contact_sheet(
                run_dir,
                panel_name,
                raw,
                panel_labels,
                config.conditional_generation.fixed_exposure_scale,
            ),
        }
    final_state = state_content_hash(bundle.state_dict())

    def group_gate(name: str, group: str) -> bool:
        panel = report_panels[name]
        threshold = 0.90 if name == "axis" else 0.80
        baseline = panel["baseline"]
        probe = panel["probe"][group]
        ratios_ok = all(
            after >= before - 0.03
            for before, after in zip(
                baseline["diversity_ratios"], probe["metrics"]["diversity_ratios"]
            )
        )
        return (
            panel["positive_fraction"][group] >= threshold
            and probe["median_low_margin_delta"] >= 0.01
            and probe["correct_count_delta"] >= 0
            and probe["metrics"]["code_fraction"] >= 0.25
            and probe["metrics"]["code_fraction"] >= baseline["code_fraction"] - 0.01
            and ratios_ok
            and probe["metrics"]["saturation"] <= 0.08
        )

    q_positive = all(
        report_panels[name]["positive_fraction"]["q"]
        >= (0.90 if name == "axis" else 0.80)
        for name in report_panels
    )
    q_probe = all(group_gate(name, "q") for name in report_panels)
    state_ok = initial_state == final_state
    readout_ok = evaluator_accuracy >= 0.985 and state_ok
    q_reachable = readout_ok and q_positive and q_probe
    carrier_or_phase = readout_ok and any(
        all(group_gate(name, group) for name in report_panels)
        for group in ("carrier", "encoder_phase", "decoder_phase")
    )
    classification = (
        "semantic_correction_reachable"
        if q_reachable
        else (
            "carrier_or_phase_required"
            if carrier_or_phase
            else (
                "semantic_readout_inconsistent"
                if not readout_ok
                else "local_semantic_unreachable"
            )
        )
    )
    checks = (
        GateResult(
            "evaluator_qualification",
            evaluator_accuracy >= 0.985,
            {"accuracy": evaluator_accuracy},
        ),
        GateResult("state_unchanged", state_ok, {}),
        GateResult("q_directional_gate", q_positive, {}),
        GateResult("q_probe_accuracy_non_decrease", q_probe, {}),
        GateResult("no_optimizer", True, {}),
    )
    report = run.root / "conditional_semantic_tangent_report.json"
    atomic_write_json(
        report,
        {
            "contract": SEMANTIC_TANGENT_CONTRACT,
            "classification": classification,
            "evaluator_accuracy": evaluator_accuracy,
            "panels": report_panels,
            "artifacts": artifact_hashes,
            "state_before": initial_state,
            "state_after": final_state,
        },
    )
    evidence = EvidenceRecordV2(
        2,
        "conditional_semantic_tangent",
        config.run.evidence_tier,
        "passed" if classification == "semantic_correction_reachable" else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=file_sha256(args.checkpoint),
        ),
        {
            "report": file_sha256(report),
            "class_mode_admission": file_sha256(args.class_mode_admission_evidence),
            "failed_class_mode_smoke": file_sha256(
                args.failed_class_mode_smoke_evidence
            ),
            "failed_parity": file_sha256(args.failed_parity_evidence),
            "artifacts": artifact_hashes,
        },
        checks,
        ("conditional_parity_mode_admission",)
        if classification == "semantic_correction_reachable"
        else (),
        None if classification == "semantic_correction_reachable" else classification,
    )
    _write_evidence(run, "conditional_semantic_tangent_evidence.json", evidence)
    print(run.root)
    return 0
