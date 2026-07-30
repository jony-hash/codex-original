# ruff: noqa: F401,F403,F405,F821
"""Diagnostic compatibility implementation segment."""

from __future__ import annotations

from . import runtime as _runtime
from . import admission as _admission
from . import recovery as _recovery
from . import diversity as _diversity
from . import class_modes as _class_modes
from . import parity as _parity
from . import legacy_training as _legacy_training

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _diversity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _class_modes.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _parity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _legacy_training.__dict__.items() if not name.startswith("__")})

def conditional_evaluate(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    training_record = EvidenceRecordV2.from_dict(
        json.loads(Path(args.training_evidence).read_text(encoding="utf-8"))
    )
    if (
        training_record.evidence_kind
        not in {
            "conditional_training",
            "conditional_diversity_training",
            "conditional_diversity_aligned_training",
            "conditional_class_mode_training",
            "conditional_parity_stabilizer_training",
            "conditional_multi_anchor_training",
            "conditional_formal_spacing_training",
            "conditional_near_conjugate_training",
        }
        or training_record.status != "passed"
    ):
        raise ValueError(
            "conditional evaluate requires passed conditional training evidence"
        )
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(
        images, all_labels, args.dataset_root
    )
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    bundle, manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=training_record.evidence_kind
        in {
            "conditional_diversity_training",
            "conditional_diversity_aligned_training",
            "conditional_class_mode_training",
            "conditional_parity_stabilizer_training",
            "conditional_multi_anchor_training",
            "conditional_formal_spacing_training",
            "conditional_near_conjugate_training",
        },
    )
    if training_record.context.checkpoint_hash != file_sha256(args.checkpoint):
        raise ValueError("training evidence does not bind the evaluation checkpoint")
    settings = config.conditional_generation
    assert settings is not None
    (
        raw_rows,
        label_rows,
        prediction_rows,
        feature_rows,
        real_feature_rows,
        generated_view_rows,
        real_view_rows,
    ) = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch_number, indices in enumerate(
            balanced_batches(
                validation_labels, batch_size=settings.logical_batch_size, seed=4321
            )
        ):
            labels = validation_labels.index_select(0, indices).to(device)
            codes = _random_codes(
                labels.numel(), device=device, seed=4321 + batch_number
            )
            raw = raw_from_generator_batched(
                bundle.system,
                bundle.generator,
                codes,
                labels,
                microbatch_size=config.execution.evaluation_microbatch_size,
                activation_checkpoint=False,
            )
            generated_view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
            generated_features, logits = evaluator(generated_view)
            real_view = validation_images.index_select(0, indices).to(device)
            real_features, _ = evaluator(real_view)
            raw_rows.append(raw.cpu())
            label_rows.append(labels.cpu())
            prediction_rows.append(logits.argmax(dim=1).cpu())
            feature_rows.append(generated_features.cpu())
            real_feature_rows.append(real_features.cpu())
            generated_view_rows.append(generated_view.cpu())
            real_view_rows.append(real_view.cpu())
    raw, labels, predictions = (
        torch.cat(raw_rows),
        torch.cat(label_rows),
        torch.cat(prediction_rows),
    )
    generated_features, real_features = (
        torch.cat(feature_rows),
        torch.cat(real_feature_rows),
    )
    generated_view, real_view = (
        torch.cat(generated_view_rows),
        torch.cat(real_view_rows),
    )
    accuracy = float((predictions == labels).to(torch.float32).mean().item())
    recalls = [
        float(((predictions[labels == label] == label).to(torch.float32).mean()).item())
        for label in range(10)
    ]
    code_fraction = _code_fraction(raw, labels)
    saturation = float(
        torch.relu(raw / settings.fixed_exposure_scale - 1.0)
        .gt(0)
        .to(torch.float32)
        .mean()
        .item()
    )
    if (
        training_record.evidence_kind == "conditional_diversity_aligned_training"
        and training_record.input_hashes.get("metric_contract")
        != ALIGNED_DIVERSITY_METRIC_CONTRACT
    ):
        raise ValueError(
            "aligned evaluation requires the normalized diversity metric contract"
        )
    diversity_ratios = tuple(
        relative_diversity_ratios(generated_features, real_features, labels)
    )
    coverage_ratio = conditional_coverage_ratio(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    checks = (
        GateResult("semantic_accuracy", accuracy >= 0.95, {"accuracy": accuracy}),
        GateResult(
            "per_class_recall", min(recalls) >= 0.90, {"minimum_recall": min(recalls)}
        ),
        GateResult(
            "camera_code_fraction",
            code_fraction >= 0.25,
            {"camera_code_fraction": code_fraction},
        ),
        GateResult(
            "high_saturation", saturation <= 0.08, {"high_saturation": saturation}
        ),
        GateResult(
            "per_class_feature_diversity",
            all(0.70 <= value <= 1.30 for value in diversity_ratios),
            {f"label_{label}": value for label, value in enumerate(diversity_ratios)},
        ),
        GateResult(
            "sinkhorn_coverage",
            coverage_ratio <= 1.5,
            {"coverage_ratio": coverage_ratio},
        ),
    )
    run = _new_run(config)
    raw_path = run.root / "conditional_evaluation_raw_cmos.pt"
    labels_path = run.root / "conditional_evaluation_labels.pt"
    atomic_torch_save(raw_path, raw)
    atomic_torch_save(labels_path, labels)
    review_artifacts = write_conditional_review_artifacts(
        root=run.root,
        raw_cmos=raw,
        labels=labels,
        exposure_scale=settings.fixed_exposure_scale,
    )
    passed = all(item.passed for item in checks)
    acceptance = config.run.evidence_tier == "acceptance"
    status = (
        "pending_review"
        if passed and acceptance
        else ("diagnostic" if passed else "failed")
    )
    permitted = ("admit_conditional_evaluation",) if status == "pending_review" else ()
    record = EvidenceRecordV2(
        2,
        "conditional_acceptance",
        config.run.evidence_tier,
        status,
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
            "training_evidence": file_sha256(args.training_evidence),
            "checkpoint": file_sha256(args.checkpoint),
            "raw_cmos": file_sha256(raw_path),
            "labels": file_sha256(labels_path),
            "review_raw_cmos": review_artifacts.raw_tensor_sha256,
            "review_preview": review_artifacts.preview_sha256,
        },
        checks,
        permitted,
        None if passed else "conditional automatic acceptance gate failed",
    )
    _write_evidence(run, "conditional_acceptance_evidence.json", record)
    print(run.root)
    return 0


def admit_conditional_evaluation(args, config: V2Config) -> int:
    _require_conditional(config)
    source = EvidenceRecordV2.from_dict(
        json.loads(Path(args.evaluation_evidence).read_text(encoding="utf-8"))
    )
    source.validate()
    if (
        source.evidence_kind != "conditional_acceptance"
        or source.status != "pending_review"
        or "admit_conditional_evaluation" not in source.permitted_next_actions
    ):
        raise ValueError(
            "conditional evaluation evidence cannot authorize manual admission"
        )
    raw = torch.load(args.evaluation_raw_cmos, map_location="cpu", weights_only=False)
    labels = torch.load(args.evaluation_labels, map_location="cpu", weights_only=False)
    if not torch.is_tensor(raw) or file_sha256(
        args.evaluation_raw_cmos
    ) != source.input_hashes.get("raw_cmos"):
        raise ValueError(
            "evaluation raw CMOS does not match conditional acceptance evidence"
        )
    if (
        not torch.is_tensor(labels)
        or labels.ndim != 1
        or labels.shape[0] != raw.shape[0]
        or file_sha256(args.evaluation_labels) != source.input_hashes.get("labels")
    ):
        raise ValueError(
            "evaluation labels do not match conditional acceptance evidence"
        )
    selected = torch.cat(
        [
            torch.nonzero(labels == label, as_tuple=False).flatten()[:16]
            for label in range(10)
        ]
    )
    reviewed = raw.index_select(0, selected)
    preview_hash = file_sha256(args.evaluation_preview)
    if preview_hash != source.input_hashes.get("review_preview"):
        raise ValueError("conditional evaluation preview does not match evidence")
    review, review_hash = load_conditional_manual_review(
        args.manual_review,
        tensor_hash=tensor_sha256(reviewed),
        preview_hash=preview_hash,
    )
    readable = all(count >= 14 for count in review.readable_counts)
    gate = GateResult(
        "manual_readability_14_of_16",
        readable,
        {
            f"label_{label}": float(count)
            for label, count in enumerate(review.readable_counts)
        },
        None if readable else "every label requires at least fourteen readable samples",
    )
    run = _new_run(config)
    status = "passed" if readable else "failed"
    record = EvidenceRecordV2(
        2,
        "conditional_generation_admission",
        config.run.evidence_tier,
        status,
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        source.context,
        {
            **source.input_hashes,
            "evaluation_evidence": source.evidence_hash,
            "manual_review": review_hash,
        },
        (*source.gates, gate),
        ("export",) if readable else (),
        None if readable else "conditional manual readability gate failed",
    )
    _write_evidence(run, "conditional_generation_admission_evidence.json", record)
    print(run.root)
    return 0


def export_conditional_dataset(args, config: V2Config) -> int:
    _require_conditional(config)
    admission = _passed_evidence(
        args.acceptance_evidence, kind="conditional_generation_admission"
    )
    device = torch.device(config.run.device)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device)
    settings = config.conditional_generation
    assert settings is not None
    count = args.samples_per_label or settings.samples_per_label
    if count != settings.samples_per_label:
        raise ValueError(
            "formal export locks samples_per_label to the configured value"
        )
    run = _new_run(config)
    dataset_root = run.root / "conditional_mnist_dataset"
    dataset_root.mkdir()
    shard_hashes = []
    index = 0
    with torch.no_grad():
        for label in range(10):
            for start in range(0, count, settings.shard_size):
                size = min(settings.shard_size, count - start)
                labels = torch.full((size,), label, device=device, dtype=torch.long)
                codes = _random_codes(size, device=device, seed=args.seed + index)
                raw = raw_from_generator_batched(
                    bundle.system,
                    bundle.generator,
                    codes,
                    labels,
                    microbatch_size=config.execution.evaluation_microbatch_size,
                    activation_checkpoint=False,
                )
                view200 = (raw / settings.fixed_exposure_scale).clamp(0.0, 1.0)
                view28 = (
                    (fixed_exposure_view(raw, settings.fixed_exposure_scale) * 255.0)
                    .round()
                    .to(torch.uint8)
                )
                shard = (
                    dataset_root
                    / f"label_{label}_part_{start // settings.shard_size:03d}.pt"
                )
                atomic_torch_save(
                    shard,
                    {
                        "indices": torch.arange(index, index + size),
                        "labels": labels.cpu(),
                        "codes": codes.cpu(),
                        "raw_cmos": raw.cpu(),
                        "fixed_exposure_view": view200.cpu(),
                        "mnist_view_uint8": view28.cpu(),
                        "seed": args.seed,
                    },
                )
                shard_hashes.append(
                    {"path": shard.name, "sha256": file_sha256(shard), "count": size}
                )
                index += size
    manifest_payload = {
        "schema_version": 1,
        "status": "candidate",
        "architecture_kind": manifest.architecture_kind,
        "system_signature": manifest.system_signature,
        "checkpoint_hash": file_sha256(args.checkpoint),
        "evaluator_checkpoint_hash": manifest.evaluator_checkpoint_hash,
        "dataset_split_manifest_hash": manifest.dataset_split_manifest_hash,
        "exposure_scale": settings.fixed_exposure_scale,
        "samples_per_label": count,
        "seed": args.seed,
        "code_distribution": "uniform[-1,1]^8",
        "shards": shard_hashes,
        "raw_shape": [1, *config.system.slm_shape],
        "mnist_shape": [1, 28, 28],
    }
    manifest_hash = canonical_sha256(manifest_payload)
    atomic_write_json(
        dataset_root / "manifest.json",
        manifest_payload | {"manifest_hash": manifest_hash},
    )
    record = EvidenceRecordV2(
        2,
        "conditional_dataset_candidate",
        config.run.evidence_tier,
        "passed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=manifest.evaluator_checkpoint_hash,
            split_hash=manifest.dataset_split_manifest_hash,
            checkpoint_hash=file_sha256(args.checkpoint),
            candidate_manifest_hash=manifest_hash,
        ),
        {
            "generation_admission": admission.evidence_hash,
            "candidate_manifest": file_sha256(dataset_root / "manifest.json"),
        },
        (GateResult("candidate_shards_complete", True, {"samples": float(index)}),),
        ("audit_memorization", "evaluate_utility"),
        None,
    )
    _write_evidence(run, "conditional_dataset_candidate_evidence.json", record)
    print(dataset_root)
    return 0


def audit_memorization(args, config: V2Config) -> int:
    _require_conditional(config)
    candidate = _passed_evidence(
        args.candidate_evidence, kind="conditional_dataset_candidate"
    )
    payload, generated_images, generated_labels = load_candidate_dataset(
        args.candidate_dataset
    )
    manifest_hash = canonical_sha256(payload)
    if candidate.context.candidate_dataset_manifest_hash != manifest_hash:
        raise ValueError(
            "candidate evidence does not match the candidate dataset manifest"
        )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    real_images, real_labels = load_local_mnist(args.dataset_root, train=True)

    def embeddings(images: torch.Tensor) -> torch.Tensor:
        rows = []
        with torch.no_grad():
            for start in range(0, images.shape[0], 256):
                features, _ = evaluator(images[start : start + 256].to(device))
                rows.append(features.cpu())
        return torch.cat(rows)

    report = memorization_statistics(
        embeddings(generated_images),
        generated_labels,
        embeddings(real_images),
        real_labels,
    )
    class_rates = [float(row["anomalous_rate"]) for row in report["per_class"]]
    passed = float(report["overall_anomalous_rate"]) <= 0.01 and all(
        rate <= 0.01 for rate in class_rates
    )
    run = _new_run(config)
    report_path = run.root / "conditional_memorization_report.json"
    atomic_write_json(report_path, report)
    gate = GateResult(
        "nearest_neighbor_memorization",
        passed,
        {
            "overall_anomalous_rate": float(report["overall_anomalous_rate"]),
            **{f"label_{index}": rate for index, rate in enumerate(class_rates)},
        },
        None
        if passed
        else "candidate contains more than one percent anomalously close samples",
    )
    record = EvidenceRecordV2(
        2,
        "conditional_memorization_audit",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        candidate.context,
        {
            "candidate_evidence": candidate.evidence_hash,
            "candidate_manifest": file_sha256(
                Path(args.candidate_dataset) / "manifest.json"
            ),
            "report": file_sha256(report_path),
        },
        (gate,),
        ("admit_dataset",) if passed else (),
        None if passed else "conditional memorization audit failed",
    )
    _write_evidence(run, "conditional_memorization_evidence.json", record)
    print(run.root)
    return 0


def _fixed_low_data_indices(
    labels: torch.Tensor, *, per_label: int = 600
) -> torch.Tensor:
    indices = []
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        if rows.numel() < per_label:
            raise ValueError("MNIST train split lacks required low-data class examples")
        indices.append(rows[:per_label])
    return torch.cat(indices)


def _train_utility_model(
    images: torch.Tensor, labels: torch.Tensor, *, seed: int, device: torch.device
) -> FrozenMNISTEvaluator:
    torch.manual_seed(seed)
    model = FrozenMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(10):
        order = torch.randperm(images.shape[0], generator=generator)
        for start in range(0, order.numel(), 128):
            rows = order[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images.index_select(0, rows).to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, labels.index_select(0, rows).to(device)
            )
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate_utility(args, config: V2Config) -> int:
    _require_conditional(config)
    candidate = _passed_evidence(
        args.candidate_evidence, kind="conditional_dataset_candidate"
    )
    payload, synthetic_images, synthetic_labels = load_candidate_dataset(
        args.candidate_dataset
    )
    manifest_hash = canonical_sha256(payload)
    if candidate.context.candidate_dataset_manifest_hash != manifest_hash:
        raise ValueError(
            "candidate evidence does not match the candidate dataset manifest"
        )
    device = torch.device(config.run.device)
    train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    low_rows = _fixed_low_data_indices(train_labels)
    regimes = {
        "low_data": (
            train_images.index_select(0, low_rows),
            train_labels.index_select(0, low_rows),
            0.0,
        ),
        "full_data": (train_images, train_labels, -0.002),
    }
    report: dict[str, object] = {
        "seeds": [1234, 1235, 1236],
        "test_count": int(test_labels.numel()),
        "regimes": {},
    }
    all_passed = True
    for name, (base_images, base_labels, lower_limit) in regimes.items():
        differences, seed_rows = [], []
        for seed in (1234, 1235, 1236):
            baseline = _train_utility_model(
                base_images, base_labels, seed=seed, device=device
            )
            augmented = _train_utility_model(
                torch.cat((base_images, synthetic_images)),
                torch.cat((base_labels, synthetic_labels)),
                seed=seed,
                device=device,
            )
            with torch.no_grad():
                _, base_logits = baseline(test_images.to(device))
                _, augmented_logits = augmented(test_images.to(device))
            base_correct = base_logits.argmax(dim=1).cpu() == test_labels
            augmented_correct = augmented_logits.argmax(dim=1).cpu() == test_labels
            differences.append(
                augmented_correct.to(torch.float32) - base_correct.to(torch.float32)
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "baseline_accuracy": float(
                        base_correct.to(torch.float32).mean().item()
                    ),
                    "augmented_accuracy": float(
                        augmented_correct.to(torch.float32).mean().item()
                    ),
                    "baseline_prediction_hash": tensor_sha256(
                        base_logits.argmax(dim=1).cpu()
                    ),
                    "augmented_prediction_hash": tensor_sha256(
                        augmented_logits.argmax(dim=1).cpu()
                    ),
                }
            )
        lower = paired_bootstrap_lower_bound(
            torch.cat(differences), seed=1234, resamples=10000
        )
        passed = lower > lower_limit if name == "low_data" else lower >= lower_limit
        report["regimes"][name] = {
            "seeds": seed_rows,
            "bootstrap_lower_bound": lower,
            "limit": lower_limit,
            "passed": passed,
        }
        all_passed = all_passed and passed
    run = _new_run(config)
    report_path = run.root / "conditional_utility_report.json"
    atomic_write_json(report_path, report)
    metrics = {
        f"{name}_lower_bound": float(values["bootstrap_lower_bound"])
        for name, values in report["regimes"].items()
    }
    gate = GateResult(
        "downstream_utility",
        all_passed,
        metrics,
        None if all_passed else "conditional synthetic data did not meet utility gates",
    )
    record = EvidenceRecordV2(
        2,
        "conditional_utility",
        config.run.evidence_tier,
        "passed" if all_passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        candidate.context,
        {
            "candidate_evidence": candidate.evidence_hash,
            "candidate_manifest": file_sha256(
                Path(args.candidate_dataset) / "manifest.json"
            ),
            "report": file_sha256(report_path),
        },
        (gate,),
        ("admit_dataset",) if all_passed else (),
        None if all_passed else "conditional utility gate failed",
    )
    _write_evidence(run, "conditional_utility_evidence.json", record)
    print(run.root)
    return 0


def admit_dataset(args, config: V2Config) -> int:
    _require_conditional(config)
    admission = _passed_evidence(
        args.generation_admission_evidence, kind="conditional_generation_admission"
    )
    memorization = _passed_evidence(
        args.memorization_evidence, kind="conditional_memorization_audit"
    )
    utility = _passed_evidence(args.utility_evidence, kind="conditional_utility")
    payload, _, _ = load_candidate_dataset(args.candidate_dataset)
    manifest_hash = canonical_sha256(payload)
    expected = _context(
        config,
        evaluator_hash=payload["evaluator_checkpoint_hash"],
        split_hash=payload["dataset_split_manifest_hash"],
        checkpoint_hash=payload["checkpoint_hash"],
        candidate_manifest_hash=manifest_hash,
    )
    for record in (memorization, utility):
        record.context.assert_matches(expected)
    if (
        admission.context.system_signature != expected.system_signature
        or admission.context.config_hash != expected.config_hash
        or admission.context.checkpoint_hash != expected.checkpoint_hash
    ):
        raise ValueError(
            "generation admission context does not match candidate dataset"
        )
    accepted = {
        **payload,
        "status": "accepted",
        "candidate_manifest_hash": manifest_hash,
    }
    accepted_hash = canonical_sha256(accepted)
    root = Path(args.candidate_dataset)
    accepted_path = root / "accepted_manifest.json"
    if accepted_path.exists():
        raise FileExistsError("accepted dataset manifest already exists")
    atomic_write_json(accepted_path, accepted | {"manifest_hash": accepted_hash})
    run = _new_run(config)
    record = EvidenceRecordV2(
        2,
        "conditional_dataset_accepted",
        config.run.evidence_tier,
        "passed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "generation_admission": admission.evidence_hash,
            "memorization": memorization.evidence_hash,
            "utility": utility.evidence_hash,
            "candidate_manifest": file_sha256(root / "manifest.json"),
            "accepted_manifest": file_sha256(accepted_path),
        },
        (
            GateResult(
                "all_dataset_acceptance_evidence_bound",
                True,
                {"candidate_manifest_hash_bound": True},
            ),
        ),
        (),
        None,
    )
    _write_evidence(run, "conditional_dataset_accepted_evidence.json", record)
    print(accepted_path)
    return 0


# ---------------------------------------------------------------------------
# v5 multi-anchor/local-mode runtime.  This section is intentionally separate
# from the v1-v4 handlers above: old evidence can be inspected, but cannot
# silently authorize this new generator contract.
