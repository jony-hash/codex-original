# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from .._conditional_admission_shared import ConditionalRuntimeBindings
from .._conditional_recovery_shared import bind_recovery

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})

def _conditional_optimizer(
    bundle: ConditionalBundle, config: V2Config
) -> torch.optim.Optimizer:
    settings = config.conditional_generation
    assert settings is not None
    return torch.optim.Adam(
        [
            {
                "params": [
                    bundle.generator.class_carrier_logits,
                    bundle.generator.global_modes,
                    bundle.generator.code_projection.weight,
                    bundle.generator.class_heads,
                ],
                "lr": settings.generator_learning_rate,
            },
            {
                "params": [bundle.system.encoder.phase, bundle.system.decoder.phase],
                "lr": settings.phase_learning_rate,
            },
        ]
    )


def conditional_train(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    if config.architecture == IDENTITY_ARCHITECTURE:
        images, all_labels = load_local_mnist(args.dataset_root, train=True)
        train_images, train_labels, _, _, split = _split(
            images, all_labels, args.dataset_root
        )
        expected = _context(
            config,
            evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
            checkpoint_hash=file_sha256(args.initialization_checkpoint),
        )
        injection = EvidenceRecordV2.from_dict(
            json.loads(Path(args.injection_evidence).read_text(encoding="utf-8"))
        )
        observability = EvidenceRecordV2.from_dict(
            json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
        )
        recovery = EvidenceRecordV2.from_dict(
            json.loads(
                Path(args.semantic_recovery_evidence).read_text(encoding="utf-8")
            )
        )
        require_evidence_action(
            injection,
            kind="conditional_injection_admission",
            action="conditional_observability",
            expected_context=expected,
        )
        require_evidence_action(
            observability,
            kind="conditional_observability",
            action="conditional_control_isolation",
            expected_context=expected,
        )
        require_evidence_action(
            recovery,
            kind="conditional_semantic_recovery_smoke",
            action="train",
            expected_context=expected,
        )
        if recovery.input_hashes.get("first_400_schedule_hash") != _schedule_hash(
            train_labels, SEMANTIC_RECOVERY_STEPS
        ):
            raise ValueError(
                "semantic recovery does not reproduce the formal training first-400 batch/code schedule"
            )
    else:
        _passed_evidence(args.observability_evidence, kind="conditional_observability")
        _passed_evidence(args.controllability_evidence, kind="controllability_smoke")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, all_labels, args.dataset_root
    )
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if config.architecture == IDENTITY_ARCHITECTURE:
        bundle, init_manifest = _load_bundle(
            config,
            args.initialization_checkpoint,
            device=device,
            evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
        )
    else:
        bundle = ConditionalBundle(
            config,
            carrier_logits=_carrier_logits(
                train_images, train_labels, config.system.slm_shape
            ),
        ).to(device)
        init_manifest = None
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    history: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        for batch_number, indices in enumerate(
            balanced_batches(
                train_labels, batch_size=settings.logical_batch_size, seed=1234 + epoch
            )
        ):
            labels = train_labels.index_select(0, indices).to(device)
            codes = _random_codes(
                labels.numel(), device=device, seed=(epoch + 1) * 100000 + batch_number
            )
            optimizer.zero_grad(set_to_none=True)
            loss, summary, _ = conditional_loss(
                system=bundle.system,
                generator=bundle.generator,
                evaluator=evaluator,
                codes=codes,
                labels=labels,
                real_views=train_images.index_select(0, indices).to(device),
                exposure_scale=settings.fixed_exposure_scale,
                sinkhorn_epsilon=settings.sinkhorn_epsilon,
                sinkhorn_iterations=settings.sinkhorn_iterations,
                activation_checkpoint=config.execution.activation_checkpoint,
                microbatch_size=config.execution.training_microbatch_size,
            )
            loss.backward()
            optimizer.step()
            history.append(asdict(summary) | {"epoch": float(epoch + 1)})
    run = _new_run(config)
    checkpoint_path = run.root / "conditional_optical_generator.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        config.architecture,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_checkpoint_hash(args.evaluator_checkpoint),
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        IDENTITY_GENERATOR_CONTRACT
        if config.architecture == IDENTITY_ARCHITECTURE
        else None,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
        if config.architecture == IDENTITY_ARCHITECTURE
        else None,
        TRAINING_SCHEDULE_CONTRACT
        if config.architecture == IDENTITY_ARCHITECTURE
        else None,
        1234 if config.architecture == IDENTITY_ARCHITECTURE else None,
        init_manifest.initialization_state_hash if init_manifest is not None else None,
    )
    save_conditional_checkpoint(
        checkpoint_path, bundle, manifest, extra={"history": history}
    )
    checkpoint_hash = file_sha256(checkpoint_path)
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    permitted = (
        ("evaluate",) if finite and config.run.evidence_tier == "acceptance" else ()
    )
    inputs = {
        "checkpoint": checkpoint_hash,
        "evaluator": manifest.evaluator_checkpoint_hash,
    }
    if config.architecture == IDENTITY_ARCHITECTURE:
        inputs["first_100_schedule_hash"] = _first_hundred_schedule_hash(train_labels)
    record = EvidenceRecordV2(
        2,
        "conditional_training",
        config.run.evidence_tier,
        "passed" if finite else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=manifest.evaluator_checkpoint_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=checkpoint_hash,
        ),
        inputs,
        (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),),
        permitted,
        None if finite else "conditional training produced non-finite loss",
    )
    _write_evidence(run, "conditional_training_evidence.json", record)
    print(run.root)
    return 0


def _balanced_epoch_steps(labels: torch.Tensor, batch_size: int = 50) -> int:
    return sum(1 for _ in balanced_batches(labels, batch_size=batch_size, seed=1234))


def conditional_training_completion(args, config: V2Config) -> int:
    """Repair only the schedule metadata of the completed semantic checkpoint."""
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    training = _passed_evidence(args.training_evidence, kind="conditional_training")
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=checkpoint_hash,
    )
    if (
        training.context != expected
        or training.input_hashes.get("checkpoint") != checkpoint_hash
    ):
        raise ValueError(
            "training evidence does not bind the requested semantic checkpoint context"
        )
    payload = safe_torch_load(args.checkpoint)
    history = (
        payload.get("extra", {}).get("history") if isinstance(payload, dict) else None
    )
    expected_steps = config.conditional_generation.epochs * _balanced_epoch_steps(
        train_labels, config.conditional_generation.logical_batch_size
    )
    if not isinstance(history, list) or len(history) != expected_steps:
        raise ValueError(
            f"completed checkpoint history must contain exactly {expected_steps} balanced steps"
        )
    if any(
        not torch.isfinite(torch.tensor(float(row.get("total", float("nan")))))
        for row in history
        if isinstance(row, dict)
    ):
        raise ValueError("completed checkpoint contains non-finite history")
    bundle, old_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=torch.device("cpu"),
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    run = _new_run(config)
    corrected = run.root / "conditional_optical_generator_balanced.pt"
    new_manifest = ConditionalCheckpointManifestV2(
        2,
        IDENTITY_ARCHITECTURE,
        old_manifest.system_signature,
        old_manifest.config_hash,
        old_manifest.dataset_split_manifest_hash,
        old_manifest.evaluator_checkpoint_hash,
        old_manifest.exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        IDENTITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        old_manifest.initialization_state_hash,
    )
    save_conditional_checkpoint(
        corrected,
        bundle,
        new_manifest,
        extra={
            "history": history,
            "schedule_steps_per_epoch": expected_steps
            // config.conditional_generation.epochs,
            "source_checkpoint": checkpoint_hash,
            "metadata_only_correction": True,
        },
    )
    checks = (
        GateResult(
            "balanced_steps_per_epoch",
            expected_steps // config.conditional_generation.epochs == 984,
            {
                "steps_per_epoch": float(
                    expected_steps // config.conditional_generation.epochs
                )
            },
        ),
        GateResult("history_finite", True, {"steps": float(len(history))}),
        GateResult(
            "state_content_preserved",
            new_manifest.state_content_hash == old_manifest.state_content_hash,
            {},
        ),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_training_completion",
        config.run.evidence_tier,
        "passed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=file_sha256(corrected),
        ),
        {
            "source_checkpoint": checkpoint_hash,
            "checkpoint": file_sha256(corrected),
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
        },
        checks,
        ("diversity_observability",),
        None,
    )
    _write_evidence(run, "conditional_training_completion_evidence.json", record)
    print(run.root)
    return 0


# Formal and diagnostic commands bind the same recovery workflows.  The local
# training compatibility functions remain diagnostic-only and retain their
# broader-architecture behavior.
_recovery_bindings = ConditionalRuntimeBindings(vars(_runtime) | vars(_admission))
globals().update(bind_recovery(_recovery_bindings))
