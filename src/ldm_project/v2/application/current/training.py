# ruff: noqa: F401
from __future__ import annotations

from .runtime import (
    BALANCED_TRAINING_SCHEDULE_CONTRACT, CONDITIONAL_CONTROL_METRIC_CONTRACT,
    CONDITIONAL_CONTROL_ADMISSION_CONTRACT,
    ConditionalBundle, ConditionalCheckpointManifestV2, EvidenceRecordV2,
    GateResult, IDENTITY_ARCHITECTURE, IDENTITY_GENERATOR_CONTRACT, Path,
    SEMANTIC_RECOVERY_STEPS, TRAINING_SCHEDULE_CONTRACT, V2Config,
    _carrier_logits, _context, _load_bundle, _load_bundle_with_payload, _new_run, _random_codes,
    _require_conditional, _require_current_acceptance, _require_identity, _split,
    _system_signature, _validate_conditional_lineage, _validate_current_evidence_source,
    _write_evidence, asdict, balanced_batches,
    balanced_schedule_steps,
    canonical_sha256, conditional_loss, evaluator_checkpoint_hash, file_sha256,
    json, load_frozen_evaluator, load_local_mnist, require_evidence_action,
    require_input_hashes,
    safe_torch_load, save_conditional_checkpoint, state_content_hash, state_signature, torch,
)
from .admission import (
    _conditional_parameter_groups,
    _first_hundred_schedule_hash,
    _passed_evidence,
    _schedule_hash,
)
from .recovery import _conditional_optimizer


def _validate_two_step_parameter_updates(
    bundle: ConditionalBundle,
    groups: dict[str, tuple[torch.nn.Parameter, ...]],
    before: dict[int, torch.Tensor],
    frozen_before: dict[int, torch.Tensor],
    gradient_seen: set[int],
) -> None:
    declared = {id(parameter) for parameters in groups.values() for parameter in parameters}
    if set(groups) != {
        "class_carrier",
        "global_modes",
        "code_projection",
        "class_heads",
        "encoder_phase",
        "decoder_phase",
    }:
        raise ValueError("current optimizer parameter groups changed")
    for name, parameters in groups.items():
        if not any(id(parameter) in gradient_seen for parameter in parameters):
            raise ValueError(f"current training group {name} received no finite gradient")
        if not any(
            not torch.equal(before[id(parameter)], parameter.detach())
            for parameter in parameters
        ):
            raise ValueError(f"current training group {name} did not update")
    for parameter in bundle.parameters():
        if id(parameter) in declared:
            continue
        if not torch.equal(frozen_before[id(parameter)], parameter.detach()):
            raise ValueError("frozen current parameter changed during training")


def _validate_current_optimizer_coverage(
    optimizer: torch.optim.Optimizer,
    groups: dict[str, tuple[torch.nn.Parameter, ...]],
) -> None:
    """Require the production optimizer to cover exactly the declared groups."""
    expected = {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
    actual = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"current optimizer parameter coverage changed (missing={len(missing)}, extra={len(extra)})"
        )


def conditional_train(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    if config.architecture == IDENTITY_ARCHITECTURE:
        controllability_path = getattr(args, "controllability_evidence", None)
        recovery_path = getattr(args, "semantic_recovery_evidence", None)
        if bool(controllability_path) == bool(recovery_path):
            raise ValueError(
                "identity training requires exactly one controllability or "
                "semantic-recovery evidence"
            )
        images, all_labels = load_local_mnist(args.dataset_root, train=True)
        train_images, train_labels, _, _, split = _split(
            images, all_labels, args.dataset_root
        )
        injection = EvidenceRecordV2.from_dict(
            json.loads(Path(args.injection_evidence).read_text(encoding="utf-8"))
        )
        observability = EvidenceRecordV2.from_dict(
            json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
        )
        authorization: EvidenceRecordV2
        initialization_checkpoint_hash = injection.input_hashes.get(
            "initialization_checkpoint"
        )
        if not isinstance(initialization_checkpoint_hash, str):
            raise ValueError("injection evidence lacks initialization checkpoint binding")
        expected = _context(
            config,
            evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
            checkpoint_hash=initialization_checkpoint_hash,
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
        if controllability_path:
            control = _passed_evidence(
                controllability_path, kind="controllability_smoke"
            )
            authorization = control
            require_evidence_action(
                control,
                kind="controllability_smoke",
                action="train",
                expected_context=expected,
            )
            initialization_state_hash = injection.input_hashes.get(
                "initialization_state_hash"
            )
            if not isinstance(initialization_state_hash, str):
                raise ValueError("injection evidence lacks initialization state binding")
            require_input_hashes(
                control,
                {
                    "initialization_checkpoint": initialization_checkpoint_hash,
                    "initialization_state_hash": initialization_state_hash,
                    "metric_contract": canonical_sha256(
                        CONDITIONAL_CONTROL_ADMISSION_CONTRACT
                    ),
                    "first_100_schedule_hash": _first_hundred_schedule_hash(
                        train_labels
                    ),
                },
            )
            if control.input_hashes.get("initialization_state_hash") != initialization_state_hash:
                raise ValueError(
                    "controllability evidence initialization state does not match admission"
                )
        else:
            recovery = _passed_evidence(recovery_path, kind="conditional_semantic_recovery_smoke")
            authorization = recovery
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
        _validate_current_evidence_source(injection, observability, authorization)
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
            checkpoint_sha256=initialization_checkpoint_hash,
        )
        _validate_conditional_lineage(
            config,
            init_manifest,
            checkpoint_hash=initialization_checkpoint_hash,
            evidence=injection,
        )
        require_input_hashes(
            injection,
            {
                "initialization_checkpoint": initialization_checkpoint_hash,
                "initialization_seed": str(init_manifest.initialization_seed),
                "initialization_state_hash": init_manifest.initialization_state_hash,
            },
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
    update_groups = (
        _conditional_parameter_groups(bundle)
        if config.architecture == IDENTITY_ARCHITECTURE
        else {}
    )
    if config.architecture == IDENTITY_ARCHITECTURE:
        _validate_current_optimizer_coverage(optimizer, update_groups)
    update_before = {
        id(parameter): parameter.detach().clone()
        for parameters in update_groups.values()
        for parameter in parameters
    }
    update_declared = set(update_before)
    frozen_before = {
        id(parameter): parameter.detach().clone()
        for parameter in bundle.parameters()
        if id(parameter) not in update_declared
    }
    gradient_seen: set[int] = set()
    update_steps = 0
    update_audit_completed = False
    expected_steps_per_epoch, expected_total_steps = balanced_schedule_steps(
        train_labels,
        batch_size=settings.logical_batch_size,
        epochs=settings.epochs,
    )
    epoch_step_counts: list[int] = []
    for epoch in range(settings.epochs):
        epoch_steps = 0
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
            if config.architecture == IDENTITY_ARCHITECTURE:
                for parameters in update_groups.values():
                    for parameter in parameters:
                        if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                            gradient_seen.add(id(parameter))
            optimizer.step()
            update_steps += 1
            epoch_steps += 1
            if config.architecture == IDENTITY_ARCHITECTURE and update_steps == 2:
                _validate_two_step_parameter_updates(
                    bundle,
                    update_groups,
                    update_before,
                    frozen_before,
                    gradient_seen,
                )
                update_audit_completed = True
            history.append(asdict(summary) | {"epoch": float(epoch + 1)})
        epoch_step_counts.append(epoch_steps)
    if config.architecture == IDENTITY_ARCHITECTURE and not update_audit_completed:
        raise ValueError("current training requires at least two optimizer steps")
    if config.architecture == IDENTITY_ARCHITECTURE and (
        update_steps != expected_total_steps
        or epoch_step_counts != [expected_steps_per_epoch] * settings.epochs
    ):
        raise ValueError(
            "current training history does not match the balanced schedule "
            f"({expected_steps_per_epoch} steps/epoch, {expected_total_steps} total)"
        )
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
        BALANCED_TRAINING_SCHEDULE_CONTRACT
        if config.architecture == IDENTITY_ARCHITECTURE
        else None,
        settings.initialization_seed if config.architecture == IDENTITY_ARCHITECTURE else None,
        init_manifest.initialization_state_hash if init_manifest is not None else None,
    )
    save_conditional_checkpoint(
        checkpoint_path,
        bundle,
        manifest,
        extra={
            "history": history,
            "schedule_steps_per_epoch": expected_steps_per_epoch
            if config.architecture == IDENTITY_ARCHITECTURE
            else None,
            "schedule_total_steps": expected_total_steps
            if config.architecture == IDENTITY_ARCHITECTURE
            else None,
        },
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
        inputs["initialization_checkpoint"] = initialization_checkpoint_hash
        inputs["initialization_state_hash"] = manifest.initialization_state_hash
        inputs["schedule_contract"] = BALANCED_TRAINING_SCHEDULE_CONTRACT
        inputs["schedule_steps_per_epoch"] = str(expected_steps_per_epoch)
        inputs["schedule_total_steps"] = str(expected_total_steps)
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
    _validate_current_evidence_source(training)
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
    bundle, old_manifest, payload = _load_bundle_with_payload(
        config,
        args.checkpoint,
        device=torch.device("cpu"),
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_sha256=checkpoint_hash,
    )
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
        old_manifest.initialization_seed,
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
    corrected_hash = file_sha256(corrected)
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
            checkpoint_hash=corrected_hash,
        ),
        {
            "source_checkpoint": checkpoint_hash,
            "checkpoint": corrected_hash,
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
        },
        checks,
        ("diversity_observability",),
        None,
    )
    _write_evidence(run, "conditional_training_completion_evidence.json", record)
    print(run.root)
    return 0
