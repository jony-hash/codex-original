# ruff: noqa: F401
from __future__ import annotations

from .runtime import (
    EvidenceRecordV2, FrozenMNISTEvaluator, GateResult, Path, V2Config,
    _context, _load_bundle, _new_run, _random_codes, _require_conditional,
    _write_evidence, atomic_torch_save, atomic_write_json, canonical_sha256,
    file_sha256, fixed_exposure_view, load_candidate_dataset,
    load_frozen_evaluator, load_local_mnist, memorization_statistics,
    paired_bootstrap_lower_bound, raw_from_generator_batched, tensor_sha256,
    torch,
)
from .admission import _passed_evidence

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
