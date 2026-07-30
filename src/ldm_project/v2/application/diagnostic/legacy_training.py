# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from . import recovery as _recovery
from . import diversity as _diversity
from . import class_modes as _class_modes
from . import parity as _parity

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _diversity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _class_modes.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _parity.__dict__.items() if not name.startswith("__")})

def diversity_recovery_smoke(args, config: V2Config) -> int:
    """Single fixed 400-step code-only diagnostic from the corrected anchor."""
    _require_identity(config)
    if getattr(args, "steps", 400) != 400:
        raise ValueError("diversity recovery smoke is locked to exactly 400 steps")
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
    completion = EvidenceRecordV2.from_dict(
        json.loads(Path(args.training_completion_evidence).read_text(encoding="utf-8"))
    )
    observability = EvidenceRecordV2.from_dict(
        json.loads(
            Path(args.diversity_observability_evidence).read_text(encoding="utf-8")
        )
    )
    require_evidence_action(
        completion,
        kind="conditional_training_completion",
        action="diversity_observability",
        expected_context=expected,
    )
    require_evidence_action(
        observability,
        kind="conditional_diversity_observability",
        action="diversity_recovery_smoke",
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
        raise ValueError("diversity smoke requires corrected completion checkpoint")
    for parameter in (
        bundle.generator.class_carrier_logits,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam(
        [parameter for params in groups.values() for parameter in params],
        lr=config.conditional_generation.generator_learning_rate,
    )
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    real_probe_images = torch.cat(
        [train_images[train_labels == label][:17] for label in range(10)], dim=0
    ).to(device)
    with torch.no_grad():
        real_probe_features, _ = evaluator(real_probe_images)
    initial_state = {
        name: value.detach().clone() for name, value in bundle.state_dict().items()
    }
    with torch.no_grad():
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
    probe_history: dict[str, object] = {}
    probe_raw_by_step: dict[int, torch.Tensor] = {}
    loss_history: list[dict[str, float]] = []
    update_history: list[dict[str, float]] = []

    def probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                probe_codes,
                probe_labels,
                microbatch_size=config.execution.evaluation_microbatch_size,
                activation_checkpoint=False,
            )
            metrics = _probe_metrics(
                raw, probe_labels, evaluator, settings.fixed_exposure_scale
            )
            generated_features, _ = evaluator(
                fixed_exposure_view(raw, settings.fixed_exposure_scale)
            )
            metrics["diversity_ratios"] = relative_diversity_ratios(
                generated_features, real_probe_features, probe_labels
            )
            return raw.detach(), metrics

    raw0, metrics0 = probe()
    probe_history["step_0"] = metrics0
    probe_raw_by_step[0] = raw0.cpu()
    for step, indices in zip(
        range(400),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _diversity_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=settings,
            config=config,
        )
        loss.backward()
        before = {
            name: [parameter.detach().clone() for parameter in params]
            for name, params in groups.items()
        }
        optimizer.step()
        update_history.append(
            {
                name: float(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(parameter.detach() - old)
                            for parameter, old in zip(
                                groups[name], before[name], strict=True
                            )
                        ]
                    )
                    .norm()
                    .item()
                )
                for name in groups
            }
        )
        loss_history.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            probe_raw, probe_metrics = probe()
            probe_history[f"step_{step + 1}"] = probe_metrics
            probe_raw_by_step[step + 1] = probe_raw.cpu()
    final_raw, final_metrics = probe()
    zero_raw_after = (
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
    ratios = (
        [float(value) for value in final_metrics.get("diversity_ratios", [])]
        if isinstance(final_metrics.get("diversity_ratios"), list)
        else []
    )
    frozen_unchanged = all(
        torch.equal(initial_state[name].detach().cpu(), value.detach().cpu())
        for name, value in bundle.state_dict().items()
        if name in initial_state
        and (
            name.startswith("generator.class_carrier")
            or name.startswith("system.encoder")
            or name.startswith("system.decoder")
        )
    )
    code_updates = all(value > 0.0 for row in update_history for value in row.values())
    initial_ratios = probe_history["step_0"].get("diversity_ratios", [])
    ratios_ok = (
        len(ratios) == 10
        and min(ratios) >= 0.25
        and all(
            float(after) >= float(before)
            for before, after in zip(initial_ratios, ratios, strict=True)
        )
    )
    checks = (
        GateResult(
            "semantic_accuracy",
            float(final_metrics["accuracy"]) >= 0.95,
            {"accuracy": float(final_metrics["accuracy"])},
        ),
        GateResult(
            "per_class_recall",
            min(
                float(final_metrics["confusion"][label][label]) / 17.0
                for label in range(10)
            )
            >= 0.90,
            {},
        ),
        GateResult(
            "relative_diversity",
            ratios_ok,
            {"minimum_ratio": min(ratios) if ratios else 0.0},
        ),
        GateResult(
            "camera_code_fraction",
            float(final_metrics["code_fraction"]) >= 0.15,
            {"value": float(final_metrics["code_fraction"])},
        ),
        GateResult(
            "high_saturation",
            float(final_metrics["saturation"]) <= 0.08,
            {"value": float(final_metrics["saturation"])},
        ),
        GateResult("frozen_parameters", frozen_unchanged, {}),
        GateResult("code_parameter_updates", code_updates, {}),
        GateResult(
            "zero_code_unchanged", bool(torch.equal(zero_raw, zero_raw_after)), {}
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
    )
    run = _new_run(config)
    step_artifacts: dict[str, dict[str, str]] = {}
    for probe_step in (0, 100, 200, 400):
        step_path = (
            run.root / f"conditional_diversity_recovery_step_{probe_step}_raw_cmos.pt"
        )
        atomic_torch_save(step_path, probe_raw_by_step[probe_step])
        preview = write_review_artifacts(
            root=run.root,
            stem=f"conditional_diversity_recovery_step_{probe_step}",
            kind="conditional_diversity_recovery",
            raw_cmos=probe_raw_by_step[probe_step],
            labels=probe_labels.cpu(),
            exposure_scale=settings.fixed_exposure_scale,
        )
        step_artifacts[str(probe_step)] = {
            "raw_cmos": file_sha256(step_path),
            "preview": preview.preview_sha256,
        }
    history_path = run.root / "conditional_diversity_recovery_history.json"
    atomic_write_json(
        history_path,
        {
            "loss": loss_history,
            "probes": probe_history,
            "updates": update_history,
            "step_probe_artifacts": step_artifacts,
            "schedule_hash_first_400": _schedule_hash(train_labels, 400),
            "contract": DIVERSITY_RECOVERY_CONTRACT,
        },
    )
    raw_path = run.root / "conditional_diversity_recovery_probe_raw_cmos.pt"
    atomic_torch_save(raw_path, final_raw.cpu())
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_recovery_smoke",
        config.run.evidence_tier,
        "passed" if all(item.passed for item in checks) else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "checkpoint": checkpoint_hash,
            "history": file_sha256(history_path),
            "raw_cmos": file_sha256(raw_path),
            "schedule_hash_first_400": _schedule_hash(train_labels, 400),
            "loss_contract": DIVERSITY_LOSS_CONTRACT,
        },
        checks,
        ("train_diversity_recovery",) if all(item.passed for item in checks) else (),
        None
        if all(item.passed for item in checks)
        else "conditional diversity recovery smoke failed",
    )
    _write_evidence(run, "conditional_diversity_recovery_evidence.json", record)
    print(run.root)
    return 0
def diversity_aligned_smoke(args, config: V2Config) -> int:
    """Fixed 400-step camera-fraction aligned smoke from the semantic anchor."""
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
        json.loads(Path(args.alignment_admission_evidence).read_text(encoding="utf-8"))
    )
    require_evidence_action(
        admission,
        kind="conditional_diversity_alignment_admission",
        action="diversity_aligned_smoke",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, initialization_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
    )
    if (
        initialization_manifest.training_schedule_hash
        != BALANCED_TRAINING_SCHEDULE_CONTRACT
    ):
        raise ValueError(
            "aligned smoke requires the metadata-corrected semantic checkpoint"
        )
    for parameter in (
        bundle.generator.class_carrier_logits,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam(
        [parameter for params in groups.values() for parameter in params],
        lr=config.conditional_generation.generator_learning_rate,
    )
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    real_probe = torch.cat(
        [train_images[train_labels == label][:17] for label in range(10)], dim=0
    ).to(device)
    with torch.no_grad():
        real_features, _ = evaluator(real_probe)
    initial_state = {
        name: value.detach().clone() for name, value in bundle.state_dict().items()
    }
    with torch.no_grad():
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
    probes: dict[str, dict[str, object]] = {}
    raws: dict[int, torch.Tensor] = {}
    losses: list[dict[str, float]] = []
    updates: list[dict[str, float]] = []

    def evaluate_probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                probe_codes,
                probe_labels,
                microbatch_size=config.execution.evaluation_microbatch_size,
                activation_checkpoint=False,
            )
            view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
            features, logits = evaluator(view)
            ratios = relative_diversity_ratios(features, real_features, probe_labels)
            predictions = logits.argmax(dim=1)
            metrics = _probe_metrics(
                raw, probe_labels, evaluator, settings.fixed_exposure_scale
            )
            metrics["diversity_ratios"] = ratios
            metrics["coefficient_rank"] = int(
                torch.linalg.matrix_rank(
                    bundle.generator.coefficients(probe_codes, probe_labels).reshape(
                        -1, 8
                    )
                ).item()
            )
            metrics["confusion"] = (
                torch.bincount(probe_labels * 10 + predictions, minlength=100)
                .reshape(10, 10)
                .cpu()
                .tolist()
            )
            return raw.detach().cpu(), metrics

    raw0, metric0 = evaluate_probe()
    raws[0] = raw0
    probes["step_0"] = metric0
    for step, indices in zip(
        range(400),
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        ),
    ):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _aligned_codes(labels, seed=100000 + step)
        optimizer.zero_grad(set_to_none=True)
        total, summary, _ = _aligned_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=train_images.index_select(0, indices).to(device),
            settings=settings,
            config=config,
        )
        total.backward()
        before = {
            name: [parameter.detach().clone() for parameter in params]
            for name, params in groups.items()
        }
        optimizer.step()
        updates.append(
            {
                name: float(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(parameter.detach() - old)
                            for parameter, old in zip(
                                groups[name], before[name], strict=True
                            )
                        ]
                    )
                    .norm()
                    .item()
                )
                for name in groups
            }
        )
        losses.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            probe_raw, probe_metrics = evaluate_probe()
            raws[step + 1] = probe_raw
            probes[f"step_{step + 1}"] = probe_metrics
    final = probes["step_400"]
    ratio200 = min(probes["step_200"]["diversity_ratios"])
    ratio400 = min(final["diversity_ratios"])
    camera200 = float(probes["step_200"]["code_fraction"])
    camera400 = float(final["code_fraction"])
    camera_loss_first = sum(row["camera_fraction_loss"] for row in losses[:50]) / 50.0
    camera_loss_last = sum(row["camera_fraction_loss"] for row in losses[-50:]) / 50.0
    frozen = all(
        torch.equal(initial_state[name].detach().cpu(), value.detach().cpu())
        for name, value in bundle.state_dict().items()
        if name.startswith("generator.class_carrier")
        or name.startswith("system.encoder")
        or name.startswith("system.decoder")
    )
    checks = (
        GateResult(
            "semantic_accuracy",
            float(final["accuracy"]) >= 0.95,
            {"accuracy": float(final["accuracy"])},
        ),
        GateResult(
            "per_class_recall",
            min(float(final["confusion"][label][label]) / 17.0 for label in range(10))
            >= 0.90,
            {},
        ),
        GateResult("camera_code_fraction", camera400 >= 0.175, {"value": camera400}),
        GateResult(
            "camera_fraction_progress",
            camera400 - camera200 >= 0.026,
            {"delta": camera400 - camera200},
        ),
        GateResult("relative_diversity", ratio400 >= 0.42, {"minimum": ratio400}),
        GateResult(
            "relative_diversity_progress",
            ratio400 - ratio200 >= 0.10,
            {"delta": ratio400 - ratio200},
        ),
        GateResult("high_saturation", float(final["saturation"]) <= 0.08, {}),
        GateResult("coefficient_rank_eight", int(final["coefficient_rank"]) == 8, {}),
        GateResult(
            "zero_code_unchanged",
            bool(
                torch.equal(
                    zero_raw,
                    raw_from_generator_batched(
                        bundle.system,
                        bundle.generator,
                        torch.zeros(10, 8, device=device),
                        torch.arange(10, device=device),
                        microbatch_size=config.execution.evaluation_microbatch_size,
                        activation_checkpoint=False,
                    )
                    .detach()
                    .cpu(),
                )
            ),
            {},
        ),
        GateResult("frozen_parameters", frozen, {}),
        GateResult(
            "camera_loss_progress",
            camera_loss_last <= 0.90 * camera_loss_first,
            {"first_50": camera_loss_first, "last_50": camera_loss_last},
        ),
        GateResult(
            "finite_losses",
            bool(torch.isfinite(torch.tensor([row["total"] for row in losses])).all()),
            {},
        ),
        GateResult(
            "code_parameter_updates",
            all(value > 0.0 for row in updates for value in row.values()),
            {},
        ),
    )
    run = _new_run(config)
    artifact_hashes = {}
    for step, raw in raws.items():
        path = run.root / f"conditional_diversity_aligned_step_{step}_raw_cmos.pt"
        atomic_torch_save(path, raw)
        preview = write_review_artifacts(
            root=run.root,
            stem=f"conditional_diversity_aligned_step_{step}",
            kind="conditional_diversity_aligned_smoke",
            raw_cmos=raw,
            labels=probe_labels.cpu(),
            exposure_scale=settings.fixed_exposure_scale,
        )
        artifact_hashes[str(step)] = {
            "raw_cmos": file_sha256(path),
            "preview": preview.preview_sha256,
        }
    history_path = run.root / "conditional_diversity_aligned_history.json"
    atomic_write_json(
        history_path,
        {
            "metric_contract": ALIGNED_DIVERSITY_METRIC_CONTRACT,
            "loss_contract": ALIGNED_DIVERSITY_LOSS_CONTRACT,
            "probes": probes,
            "loss": losses,
            "updates": updates,
            "artifacts": artifact_hashes,
            "first_400_schedule_hash": canonical_sha256(
                [
                    {
                        "indices": tensor_sha256(indices),
                        "codes": tensor_sha256(
                            _aligned_codes(
                                train_labels.index_select(0, indices),
                                seed=100000 + number,
                            )
                        ),
                    }
                    for number, indices in zip(
                        range(400),
                        balanced_batches(train_labels, batch_size=50, seed=1234),
                    )
                ]
            ),
        },
    )
    diagnostic = run.root / "conditional_diversity_aligned_diagnostic_final.pt"
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        ConditionalCheckpointManifestV2(
            2,
            IDENTITY_ARCHITECTURE,
            initialization_manifest.system_signature,
            initialization_manifest.config_hash,
            split.manifest_hash,
            evaluator_hash,
            settings.fixed_exposure_scale,
            state_signature(bundle.state_dict()),
            state_content_hash(bundle.state_dict()),
            IDENTITY_GENERATOR_CONTRACT,
            canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
            BALANCED_TRAINING_SCHEDULE_CONTRACT,
            1234,
            initialization_manifest.initialization_state_hash,
        ),
        extra={"diagnostic_only": True, "never_warm_start": True, "steps": 400},
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_aligned_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "alignment_admission": file_sha256(args.alignment_admission_evidence),
            "history": file_sha256(history_path),
            "diagnostic_final_checkpoint": file_sha256(diagnostic),
            "metric_contract": ALIGNED_DIVERSITY_METRIC_CONTRACT,
            "loss_contract": ALIGNED_DIVERSITY_LOSS_CONTRACT,
        },
        checks,
        ("train_diversity_aligned",) if passed else (),
        None if passed else "aligned diversity smoke failed",
    )
    _write_evidence(run, "conditional_diversity_aligned_smoke_evidence.json", record)
    print(run.root)
    return 0

def train_diversity_recovery(args, config: V2Config) -> int:
    """User-run one balanced epoch code-only recovery (not launched here)."""
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
    smoke = _passed_evidence(
        args.diversity_recovery_evidence, kind="conditional_diversity_recovery_smoke"
    )
    require_evidence_action(
        smoke,
        kind="conditional_diversity_recovery_smoke",
        action="train_diversity_recovery",
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
    for parameter in (
        bundle.generator.class_carrier_logits,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam(
        [parameter for params in groups.values() for parameter in params],
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
        codes = _random_codes(labels.numel(), device=device, seed=100000 + batch_number)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _diversity_objective(
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
    run = _new_run(config)
    path = run.root / "conditional_diversity_generator.pt"
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
            "history": history,
            "diversity_loss_contract": DIVERSITY_LOSS_CONTRACT,
            "epochs": 1,
            "steps": len(history),
        },
    )
    checkpoint_out_hash = file_sha256(path)
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_training",
        config.run.evidence_tier,
        "passed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=checkpoint_out_hash,
        ),
        {
            "checkpoint": checkpoint_out_hash,
            "recovery_evidence": file_sha256(args.diversity_recovery_evidence),
            "steps": len(history),
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
        },
        (
            GateResult(
                "finite_training_loss",
                bool(
                    torch.isfinite(
                        torch.tensor([row["total"] for row in history])
                    ).all()
                ),
                {"steps": float(len(history))},
            ),
        ),
        ("evaluate",),
        None,
    )
    _write_evidence(run, "conditional_diversity_training_evidence.json", record)
    print(run.root)
    return 0


def train_diversity_aligned(args, config: V2Config) -> int:
    """User-run one-epoch aligned diversity training from the semantic anchor."""
    _require_identity(config)
    smoke = _passed_evidence(
        args.aligned_smoke_evidence, kind="conditional_diversity_aligned_smoke"
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
        kind="conditional_diversity_aligned_smoke",
        action="train_diversity_aligned",
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
    for parameter in (
        bundle.generator.class_carrier_logits,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam(
        [p for ps in groups.values() for p in ps],
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
        codes = _aligned_codes(labels, seed=100000 + batch_number)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _aligned_objective(
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
    run = _new_run(config)
    path = run.root / "conditional_diversity_aligned_generator.pt"
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
            "history": history,
            "metric_contract": ALIGNED_DIVERSITY_METRIC_CONTRACT,
            "loss_contract": ALIGNED_DIVERSITY_LOSS_CONTRACT,
            "steps": len(history),
            "first_400_schedule_hash": canonical_sha256(
                [
                    {
                        "indices": tensor_sha256(indices),
                        "codes": tensor_sha256(
                            _aligned_codes(
                                train_labels.index_select(0, indices),
                                seed=100000 + number,
                            )
                        ),
                    }
                    for number, indices in zip(
                        range(400),
                        balanced_batches(train_labels, batch_size=50, seed=1234),
                    )
                ]
            ),
        },
    )
    out_hash = file_sha256(path)
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    record = EvidenceRecordV2(
        2,
        "conditional_diversity_aligned_training",
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
            "aligned_smoke": file_sha256(args.aligned_smoke_evidence),
            "steps": len(history),
            "metric_contract": ALIGNED_DIVERSITY_METRIC_CONTRACT,
        },
        (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),),
        ("evaluate",) if finite else (),
        None if finite else "aligned diversity training produced non-finite loss",
    )
    _write_evidence(run, "conditional_diversity_aligned_training_evidence.json", record)
    print(run.root)
    return 0
