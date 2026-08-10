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


def controllability_smoke(args, config: V2Config) -> int:
    _require_conditional(config)
    if config.architecture == IDENTITY_ARCHITECTURE:
        if args.steps != 100:
            raise ValueError(
                "identity semantic-alignment smoke is locked to exactly 100 steps"
            )
        images, all_labels = load_local_mnist(args.dataset_root, train=True)
        train_images, train_labels, _, _, split = _split(
            images, all_labels, args.dataset_root
        )
        evaluator, _ = load_frozen_evaluator(
            args.evaluator_checkpoint, device=torch.device(config.run.device)
        )
        expected = _context(
            config,
            evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
            checkpoint_hash=file_sha256(args.checkpoint),
        )
        injection = EvidenceRecordV2.from_dict(
            json.loads(Path(args.injection_evidence).read_text(encoding="utf-8"))
        )
        observability = EvidenceRecordV2.from_dict(
            json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
        )
        isolation = EvidenceRecordV2.from_dict(
            json.loads(Path(args.isolation_evidence).read_text(encoding="utf-8"))
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
        if "controllability_smoke" in isolation.permitted_next_actions:
            require_evidence_action(
                isolation,
                kind="conditional_control_isolation",
                action="controllability_smoke",
                expected_context=expected,
                allow_diagnostic=True,
            )
        else:
            localization_path = getattr(args, "semantic_localization_evidence", None)
            if not localization_path:
                raise ValueError(
                    "semantic-alignment isolation requires semantic-localization evidence before smoke"
                )
            require_evidence_action(
                isolation,
                kind="conditional_control_isolation",
                action="conditional_semantic_localization",
                expected_context=expected,
                allow_diagnostic=True,
            )
            localization = EvidenceRecordV2.from_dict(
                json.loads(Path(localization_path).read_text(encoding="utf-8"))
            )
            require_evidence_action(
                localization,
                kind="conditional_semantic_localization",
                action="controllability_smoke",
                expected_context=expected,
                allow_diagnostic=True,
            )
        if isolation.input_hashes.get(
            "control_admission_metric_contract"
        ) != canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT):
            raise ValueError(
                "legacy control-isolation metric contract cannot authorize identity smoke"
            )
    else:
        _passed_evidence(args.observability_evidence, kind="conditional_observability")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, all_labels, args.dataset_root
    )
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if config.architecture == IDENTITY_ARCHITECTURE:
        bundle, initialization_manifest = _load_bundle(
            config,
            args.checkpoint,
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
        initialization_manifest = None
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    real_probe_images = torch.cat(
        [train_images[train_labels == label][:17] for label in range(10)], dim=0
    ).to(device)
    with torch.no_grad():
        real_probe_features, _ = evaluator(real_probe_images)
    probe_history: dict[str, dict[str, object]] = {}
    probe_raw_by_step: dict[int, torch.Tensor] = {}
    loss_history: list[dict[str, float]] = []
    group_history: list[dict[str, object]] = []
    schedule_hash = (
        _first_hundred_schedule_hash(train_labels)
        if config.architecture == IDENTITY_ARCHITECTURE
        else None
    )

    def evaluate_probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            probe_raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                probe_codes,
                probe_labels,
                microbatch_size=config.execution.training_microbatch_size,
                activation_checkpoint=False,
            )
            return probe_raw, _probe_metrics(
                probe_raw, probe_labels, evaluator, settings.fixed_exposure_scale
            )

    initial_raw, initial_metrics = evaluate_probe()
    probe_history["step_0"] = initial_metrics
    probe_raw_by_step[0] = initial_raw.detach().cpu()
    for step, indices in zip(
        range(args.steps),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(
            labels.numel(),
            device=device,
            seed=100000 + step
            if config.architecture == IDENTITY_ARCHITECTURE
            else 1234 + step,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, summary, raw = conditional_loss(
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
        groups = _conditional_parameter_groups(bundle)
        grad_metrics = {}
        snapshots = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in groups.items()
        }
        for name, parameters in groups.items():
            norms = [
                parameter.grad.detach().norm()
                if parameter.grad is not None
                else torch.tensor(float("nan"), device=device)
                for parameter in parameters
            ]
            grad_metrics[name] = {
                "gradient_norm": float(torch.stack(norms).norm().item()),
                "gradient_finite": bool(all(torch.isfinite(value) for value in norms)),
                "gradient_nonzero": bool(any(value > 0.0 for value in norms)),
            }
        optimizer.step()
        update_metrics = {}
        for name, parameters in groups.items():
            deltas = [
                parameter.detach() - before
                for parameter, before in zip(parameters, snapshots[name], strict=True)
            ]
            update_metrics[name] = {
                "update_norm": float(
                    torch.stack([delta.norm() for delta in deltas]).norm().item()
                ),
                "update_nonzero": bool(
                    any(delta.abs().sum() > 0.0 for delta in deltas)
                ),
            }
        group_history.append(
            {"step": step + 1, "gradients": grad_metrics, "updates": update_metrics}
        )
        loss_history.append(asdict(summary) | {"step": float(step + 1)})
        if step + 1 in {10, args.steps}:
            probe_raw, metrics = evaluate_probe()
            probe_history[f"step_{step + 1}"] = metrics
            probe_raw_by_step[step + 1] = probe_raw.detach().cpu()
    if args.steps < 1:
        raise ValueError("controllability smoke received no balanced MNIST batches")
    final_raw, final_metrics = evaluate_probe()
    probe_history[f"step_{args.steps}"] = final_metrics
    probe_raw_by_step[args.steps] = final_raw.detach().cpu()
    final_groups = group_history[-1]
    step_ten_groups = next((row for row in group_history if row["step"] == 10), None)
    gradient_checks = [
        value["gradient_finite"] and value["gradient_nonzero"]
        for value in final_groups["gradients"].values()
    ]
    update_checks = [
        value["update_nonzero"] for value in final_groups["updates"].values()
    ]
    code_projection_recovered = bool(
        step_ten_groups is not None
        and step_ten_groups["gradients"]["code_projection"]["gradient_finite"]
        and step_ten_groups["gradients"]["code_projection"]["gradient_nonzero"]
        and step_ten_groups["updates"]["code_projection"]["update_nonzero"]
        and final_groups["gradients"]["code_projection"]["gradient_finite"]
        and final_groups["gradients"]["code_projection"]["gradient_nonzero"]
        and final_groups["updates"]["code_projection"]["update_nonzero"]
    )
    checks = (
        GateResult(
            "fixed_probe_complete",
            final_metrics["accuracy"] is not None,
            {"samples": 170.0},
        ),
        GateResult(
            "semantic_accuracy",
            float(final_metrics["accuracy"]) >= 0.30,
            {"accuracy": float(final_metrics["accuracy"])},
        ),
        GateResult(
            "camera_code_fraction",
            float(final_metrics["code_fraction"]) >= 0.10,
            {"camera_code_fraction": float(final_metrics["code_fraction"])},
        ),
        GateResult(
            "finite_nonzero_gradients",
            bool(all(gradient_checks)),
            {
                "groups": float(sum(gradient_checks)),
                "total": float(len(gradient_checks)),
            },
        ),
        GateResult(
            "nonzero_parameter_updates",
            bool(all(update_checks)),
            {"groups": float(sum(update_checks)), "total": float(len(update_checks))},
        ),
        GateResult(
            "identity_wz_recovered_by_step_10_and_100",
            code_projection_recovered,
            {"recovered": code_projection_recovered},
        ),
        GateResult(
            "high_saturation",
            float(final_metrics["saturation"]) <= 0.08,
            {"high_saturation": float(final_metrics["saturation"])},
        ),
    )
    run = _new_run(config)
    raw_path = run.root / "controllability_probe_raw_cmos.pt"
    labels_path = run.root / "controllability_probe_labels.pt"
    atomic_torch_save(raw_path, final_raw.cpu())
    atomic_torch_save(labels_path, probe_labels.cpu())
    review_artifacts = write_conditional_review_artifacts(
        root=run.root,
        raw_cmos=final_raw.cpu(),
        labels=probe_labels.cpu(),
        exposure_scale=settings.fixed_exposure_scale,
    )
    step_probe_artifacts: dict[str, dict[str, str]] = {}
    for probe_step in (0, 10, args.steps):
        probe_raw = probe_raw_by_step[probe_step]
        probe_raw_path = (
            run.root / f"controllability_probe_step_{probe_step}_raw_cmos.pt"
        )
        atomic_torch_save(probe_raw_path, probe_raw)
        preview = write_review_artifacts(
            root=run.root,
            stem=f"controllability_probe_step_{probe_step}",
            kind="conditional_controllability_smoke",
            raw_cmos=probe_raw,
            labels=probe_labels.cpu(),
            exposure_scale=settings.fixed_exposure_scale,
        )
        step_probe_artifacts[str(probe_step)] = {
            "raw_cmos": file_sha256(probe_raw_path),
            "preview": preview.preview_sha256,
        }
    smoke_checkpoint_path = run.root / "controllability_smoke_final.pt"
    if config.architecture == IDENTITY_ARCHITECTURE:
        assert initialization_manifest is not None
        smoke_manifest = ConditionalCheckpointManifestV2(
            2,
            IDENTITY_ARCHITECTURE,
            _system_signature(config),
            canonical_sha256(config.model_dump(mode="json")),
            split.manifest_hash,
            evaluator_checkpoint_hash(args.evaluator_checkpoint),
            settings.fixed_exposure_scale,
            state_signature(bundle.state_dict()),
            state_content_hash(bundle.state_dict()),
            IDENTITY_GENERATOR_CONTRACT,
            canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
            TRAINING_SCHEDULE_CONTRACT,
            1234,
            initialization_manifest.initialization_state_hash,
        )
        save_conditional_checkpoint(
            smoke_checkpoint_path,
            bundle,
            smoke_manifest,
            extra={
                "diagnostic_only": True,
                "never_warm_start": True,
                "steps": args.steps,
            },
        )
    history_path = run.root / "controllability_probe_history.json"
    atomic_write_json(
        history_path,
        {
            "metric_contract": CONDITIONAL_CONTROL_ADMISSION_CONTRACT
            if config.architecture == IDENTITY_ARCHITECTURE
            else CONDITIONAL_CONTROL_METRIC_CONTRACT,
            "loss": loss_history,
            "probes": probe_history,
            "parameter_groups": group_history,
            "step_probe_artifacts": step_probe_artifacts,
            "first_100_schedule_hash": schedule_hash,
        },
    )
    passed = all(item.passed for item in checks)
    context = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint)
        if config.architecture == IDENTITY_ARCHITECTURE
        else None,
    )
    permitted = (
        ("train",) if passed and config.run.evidence_tier == "acceptance" else ()
    )
    inputs = {
        "observability": file_sha256(args.observability_evidence),
        "evaluator": evaluator_checkpoint_hash(args.evaluator_checkpoint),
        "metric_contract": canonical_sha256(
            CONDITIONAL_CONTROL_ADMISSION_CONTRACT
            if config.architecture == IDENTITY_ARCHITECTURE
            else CONDITIONAL_CONTROL_METRIC_CONTRACT
        ),
        "probe_raw_cmos": file_sha256(raw_path),
        "probe_labels": file_sha256(labels_path),
        "probe_preview": review_artifacts.preview_sha256,
        "probe_history": file_sha256(history_path),
    }
    if config.architecture == IDENTITY_ARCHITECTURE:
        inputs.update(
            {
                "injection": file_sha256(args.injection_evidence),
                "isolation": file_sha256(args.isolation_evidence),
                "initialization_checkpoint": file_sha256(args.checkpoint),
                "first_100_schedule_hash": schedule_hash,
                "diagnostic_final_checkpoint": file_sha256(smoke_checkpoint_path),
            }
        )
    record = EvidenceRecordV2(
        2,
        "controllability_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        inputs,
        checks,
        permitted,
        None if passed else "conditional controllability smoke failed",
    )
    _write_evidence(run, "controllability_smoke_evidence.json", record)
    print(run.root)
    return 0
def semantic_recovery_smoke(args, config: V2Config) -> int:
    """Run the single registered 400-step recovery smoke from initialization."""
    _require_identity(config)
    if getattr(args, "steps", SEMANTIC_RECOVERY_STEPS) != SEMANTIC_RECOVERY_STEPS:
        raise ValueError("semantic recovery smoke is locked to exactly 400 steps")
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, all_labels, args.dataset_root
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.initialization_checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=init_hash,
    )
    injection = EvidenceRecordV2.from_dict(
        json.loads(Path(args.injection_evidence).read_text(encoding="utf-8"))
    )
    observability = EvidenceRecordV2.from_dict(
        json.loads(Path(args.observability_evidence).read_text(encoding="utf-8"))
    )
    isolation = EvidenceRecordV2.from_dict(
        json.loads(Path(args.isolation_evidence).read_text(encoding="utf-8"))
    )
    localization = EvidenceRecordV2.from_dict(
        json.loads(
            Path(args.semantic_localization_evidence).read_text(encoding="utf-8")
        )
    )
    failed_smoke = EvidenceRecordV2.from_dict(
        json.loads(
            Path(args.failed_controllability_evidence).read_text(encoding="utf-8")
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
        isolation,
        kind="conditional_control_isolation",
        action="conditional_semantic_localization",
        expected_context=expected,
        allow_diagnostic=True,
    )
    require_evidence_action(
        localization,
        kind="conditional_semantic_localization",
        action="controllability_smoke",
        expected_context=expected,
        allow_diagnostic=True,
    )
    failed_smoke.validate()
    if (
        failed_smoke.evidence_kind != "controllability_smoke"
        or failed_smoke.status != "failed"
    ):
        raise ValueError(
            "semantic recovery requires the immutable failed 100-step smoke evidence"
        )
    if (
        failed_smoke.context != expected
        or failed_smoke.input_hashes.get("initialization_checkpoint") != init_hash
    ):
        raise ValueError(
            "failed smoke does not belong to the requested initialization context"
        )
    diagnostic_final = failed_smoke.input_hashes.get("diagnostic_final_checkpoint")
    if diagnostic_final and diagnostic_final == init_hash:
        raise ValueError(
            "semantic recovery cannot warm-start from a smoke final checkpoint"
        )
    if failed_smoke.input_hashes.get(
        "first_100_schedule_hash"
    ) != _first_hundred_schedule_hash(train_labels):
        raise ValueError(
            "failed smoke does not reproduce the formal first-100 schedule"
        )
    if failed_smoke.input_hashes.get("metric_contract") != canonical_sha256(
        CONDITIONAL_CONTROL_ADMISSION_CONTRACT
    ):
        raise ValueError("failed smoke uses an incompatible control metric contract")

    bundle, initialization_manifest = _load_bundle(
        config,
        args.initialization_checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in bundle.named_parameters()
    }
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    probe_steps = (0, 10, 100, 200, 300, SEMANTIC_RECOVERY_STEPS)
    probe_history: dict[str, dict[str, object]] = {}
    probe_raw_by_step: dict[int, torch.Tensor] = {}
    loss_history: list[dict[str, float]] = []
    group_history: list[dict[str, object]] = []
    groups = _conditional_parameter_groups(bundle)

    def evaluate_probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                probe_codes,
                probe_labels,
                microbatch_size=config.execution.training_microbatch_size,
                activation_checkpoint=False,
            )
            return raw, _probe_metrics(
                raw, probe_labels, evaluator, settings.fixed_exposure_scale
            )

    raw0, metrics0 = evaluate_probe()
    probe_history["step_0"] = metrics0
    probe_raw_by_step[0] = raw0.detach().cpu()
    for step, indices in zip(
        range(SEMANTIC_RECOVERY_STEPS),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + step)
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
        gradient_metrics: dict[str, dict[str, object]] = {}
        snapshots = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in groups.items()
        }
        for name, parameters in groups.items():
            norms = [
                parameter.grad.detach().norm()
                if parameter.grad is not None
                else torch.tensor(float("nan"), device=device)
                for parameter in parameters
            ]
            gradient_metrics[name] = {
                "gradient_norm": float(torch.stack(norms).norm().item()),
                "gradient_finite": bool(all(torch.isfinite(value) for value in norms)),
                "gradient_nonzero": bool(any(value > 0.0 for value in norms)),
            }
        optimizer.step()
        update_metrics: dict[str, dict[str, object]] = {}
        for name, parameters in groups.items():
            deltas = [
                parameter.detach() - before
                for parameter, before in zip(parameters, snapshots[name], strict=True)
            ]
            update_metrics[name] = {
                "update_norm": float(
                    torch.stack([delta.norm() for delta in deltas]).norm().item()
                ),
                "update_nonzero": bool(
                    any(delta.abs().sum() > 0.0 for delta in deltas)
                ),
            }
        # Use explicit initial tensors for the parameter-group aliases.
        for name, parameters in groups.items():
            initial_values = {
                "class_carrier": initial_state["generator.class_carrier_logits"],
                "global_modes": initial_state["generator.global_modes"],
                "code_projection": initial_state["generator.code_projection.weight"],
                "class_heads": initial_state["generator.class_heads"],
                "encoder_phase": initial_state["system.encoder.phase"],
                "decoder_phase": initial_state["system.decoder.phase"],
            }[name]
            cumulative = [
                parameter.detach() - initial_values for parameter in parameters
            ]
            update_metrics[name]["cumulative_update_norm"] = float(
                torch.stack([delta.norm() for delta in cumulative]).norm().item()
            )
            update_metrics[name]["cumulative_update_nonzero"] = bool(
                any(delta.abs().sum() > 0.0 for delta in cumulative)
            )
        group_history.append(
            {"step": step + 1, "gradients": gradient_metrics, "updates": update_metrics}
        )
        loss_history.append(asdict(summary) | {"step": float(step + 1)})
        if step + 1 in probe_steps[1:]:
            probe_raw, probe_metrics = evaluate_probe()
            probe_history[f"step_{step + 1}"] = probe_metrics
            probe_raw_by_step[step + 1] = probe_raw.detach().cpu()
    final_metrics = probe_history[f"step_{SEMANTIC_RECOVERY_STEPS}"]
    first_cls = sum(float(item["classification"]) for item in loss_history[:50]) / 50.0
    last_cls = sum(float(item["classification"]) for item in loss_history[-50:]) / 50.0
    relative_raw_change = float(
        (probe_raw_by_step[SEMANTIC_RECOVERY_STEPS] - probe_raw_by_step[0])
        .square()
        .mean()
        .sqrt()
        .div(probe_raw_by_step[0].square().mean().sqrt().clamp_min(1.0e-12))
        .item()
    )
    checkpoints = [row for row in group_history if int(row["step"]) in probe_steps[1:]]
    gradient_gate = all(
        value["gradient_finite"] and value["gradient_nonzero"]
        for row in checkpoints
        for value in row["gradients"].values()
    )
    update_gate = all(
        value["update_nonzero"]
        for row in checkpoints
        for value in row["updates"].values()
    )
    wz_recovered = all(
        row["gradients"]["code_projection"]["gradient_nonzero"]
        and row["updates"]["code_projection"]["update_nonzero"]
        for row in checkpoints
    )
    checks = (
        GateResult(
            "fixed_probe_complete",
            len(probe_history) == len(probe_steps),
            {"checkpoints": float(len(probe_history))},
        ),
        GateResult(
            "semantic_accuracy",
            float(final_metrics["accuracy"]) >= 0.30,
            {"accuracy": float(final_metrics["accuracy"])},
        ),
        GateResult(
            "classification_progress",
            last_cls <= 0.90 * first_cls,
            {"first_50": first_cls, "last_50": last_cls},
        ),
        GateResult(
            "camera_code_fraction",
            float(final_metrics["code_fraction"]) >= 0.10,
            {"camera_code_fraction": float(final_metrics["code_fraction"])},
        ),
        GateResult(
            "high_saturation",
            float(final_metrics["saturation"]) <= 0.08,
            {"saturation": float(final_metrics["saturation"])},
        ),
        GateResult("finite_nonzero_gradients", gradient_gate, {}),
        GateResult("nonzero_parameter_updates", update_gate, {}),
        GateResult("identity_wz_recovered", wz_recovered, {}),
        GateResult(
            "finite_raw_change",
            torch.isfinite(torch.tensor(relative_raw_change)).item(),
            {"relative_raw_change": relative_raw_change},
        ),
    )
    run = _new_run(config)
    raw_path = run.root / "semantic_recovery_probe_raw_cmos.pt"
    labels_path = run.root / "semantic_recovery_probe_labels.pt"
    atomic_torch_save(raw_path, probe_raw_by_step[SEMANTIC_RECOVERY_STEPS])
    atomic_torch_save(labels_path, probe_labels.cpu())
    step_artifacts: dict[str, dict[str, str]] = {}
    for probe_step in probe_steps:
        step_path = run.root / f"semantic_recovery_probe_step_{probe_step}_raw_cmos.pt"
        atomic_torch_save(step_path, probe_raw_by_step[probe_step])
        preview = write_review_artifacts(
            root=run.root,
            stem=f"semantic_recovery_probe_step_{probe_step}",
            kind="conditional_semantic_recovery",
            raw_cmos=probe_raw_by_step[probe_step],
            labels=probe_labels.cpu(),
            exposure_scale=settings.fixed_exposure_scale,
        )
        step_artifacts[str(probe_step)] = {
            "raw_cmos": file_sha256(step_path),
            "preview": preview.preview_sha256,
        }
    history_path = run.root / "semantic_recovery_probe_history.json"
    schedule_hash = _schedule_hash(train_labels, SEMANTIC_RECOVERY_STEPS)
    atomic_write_json(
        history_path,
        {
            "metric_contract": SEMANTIC_RECOVERY_CONTRACT,
            "loss": loss_history,
            "probes": probe_history,
            "parameter_groups": group_history,
            "step_probe_artifacts": step_artifacts,
            "first_100_schedule_hash": _first_hundred_schedule_hash(train_labels),
            "first_400_schedule_hash": schedule_hash,
            "relative_raw_change": relative_raw_change,
        },
    )
    final_checkpoint = run.root / "semantic_recovery_final.pt"
    save_conditional_checkpoint(
        final_checkpoint,
        bundle,
        ConditionalCheckpointManifestV2(
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
            TRAINING_SCHEDULE_CONTRACT,
            1234,
            initialization_manifest.initialization_state_hash,
        ),
        extra={
            "diagnostic_only": True,
            "never_warm_start": True,
            "steps": SEMANTIC_RECOVERY_STEPS,
        },
    )
    report_path = run.root / "semantic_recovery_report.json"
    atomic_write_json(
        report_path,
        {
            "metric_contract": SEMANTIC_RECOVERY_CONTRACT,
            "steps": SEMANTIC_RECOVERY_STEPS,
            "first_50_classification": first_cls,
            "last_50_classification": last_cls,
            "relative_raw_change": relative_raw_change,
            "final_metrics": final_metrics,
            "probe_history": probe_history,
            "parameters_started_from_initialization": True,
            "initialization_checkpoint_hash": init_hash,
        },
    )
    passed = all(check.passed for check in checks)
    context = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=init_hash,
    )
    inputs = {
        "injection": file_sha256(args.injection_evidence),
        "observability": file_sha256(args.observability_evidence),
        "isolation": file_sha256(args.isolation_evidence),
        "semantic_localization": file_sha256(args.semantic_localization_evidence),
        "failed_controllability": file_sha256(args.failed_controllability_evidence),
        "initialization_checkpoint": init_hash,
        "evaluator": evaluator_hash,
        "report": file_sha256(report_path),
        "history": file_sha256(history_path),
        "diagnostic_final_checkpoint": file_sha256(final_checkpoint),
        "first_100_schedule_hash": _first_hundred_schedule_hash(train_labels),
        "first_400_schedule_hash": schedule_hash,
        "recovery_contract": canonical_sha256(SEMANTIC_RECOVERY_CONTRACT),
    }
    record = EvidenceRecordV2(
        2,
        "conditional_semantic_recovery_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        inputs,
        checks,
        ("train",) if passed else (),
        None if passed else "conditional semantic recovery smoke failed",
    )
    _write_evidence(run, "semantic_recovery_evidence.json", record)
    print(run.root)
    return 0

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
