# ruff: noqa: F401
from __future__ import annotations

from .runtime import (
    BALANCED_TRAINING_SCHEDULE_CONTRACT, EvidenceRecordV2, GateResult, Path, V2Config,
    _context, _load_bundle, _load_bundle_with_payload,
    _new_run, _random_codes, _require_conditional, _require_current_acceptance,
    _split, _validate_conditional_lineage, _validate_balanced_training_payload,
    _validate_current_evidence_source,
    _write_evidence, atomic_torch_save, balanced_batches,
    balanced_schedule_steps,
    conditional_coverage_ratio, evaluator_checkpoint_hash, file_sha256,
    fixed_exposure_view, json, load_conditional_manual_review,
    load_frozen_evaluator, load_local_mnist, raw_from_generator_batched,
    relative_diversity_ratios, tensor_sha256, torch,
    safe_torch_load,
    require_evidence_action, require_input_hashes,
    write_conditional_review_artifacts,
)
from .admission import _code_fraction, _passed_evidence
from .measurement import ACCEPTANCE_MEASUREMENT_CONTRACT
from ...student.evaluator import SmoothMNISTEvaluator


def sample_conditional(args, config: V2Config) -> int:
    """Render formal identity samples from an exact identity checkpoint."""
    _require_conditional(config)
    device = torch.device(config.run.device)
    bundle, _ = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        checkpoint_sha256=file_sha256(args.checkpoint),
    )
    codes = safe_torch_load(args.codes)
    labels = safe_torch_load(args.labels)
    if not torch.is_tensor(codes) or not torch.is_tensor(labels):
        raise ValueError("codes and labels must be tensor files")
    raw = (
        raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        .detach()
        .cpu()
    )
    atomic_torch_save(args.output, raw)
    return 0

def conditional_evaluate(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    training_record = EvidenceRecordV2.from_dict(
        json.loads(Path(args.training_evidence).read_text(encoding="utf-8"))
    )
    if (
        training_record.evidence_kind
        != "conditional_training"
        or training_record.status != "passed"
    ):
        raise ValueError(
            "conditional evaluate requires passed conditional training evidence"
        )
    _validate_current_evidence_source(training_record)
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    _, train_labels, validation_images, validation_labels, split = _split(
        images, all_labels, args.dataset_root
    )
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = training_record.input_hashes.get("checkpoint")
    if not isinstance(checkpoint_hash, str):
        raise ValueError("training evidence lacks checkpoint binding")
    require_evidence_action(
        training_record,
        kind="conditional_training",
        action="evaluate",
        expected_context=_context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=checkpoint_hash,
        ),
    )
    bundle, manifest, payload = _load_bundle_with_payload(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        allow_balanced_schedule=True,
        checkpoint_sha256=checkpoint_hash,
    )
    if manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT:
        raise ValueError("current evaluation requires the balanced training schedule")
    _validate_conditional_lineage(
        config,
        manifest,
        checkpoint_hash=checkpoint_hash,
        evidence=training_record,
    )
    _, expected_total_steps = balanced_schedule_steps(
        train_labels,
        batch_size=config.conditional_generation.logical_batch_size,
        epochs=config.conditional_generation.epochs,
        batch_iterator=balanced_batches,
    )
    expected_steps_per_epoch = expected_total_steps // config.conditional_generation.epochs
    _validate_balanced_training_payload(
        payload,
        steps_per_epoch=expected_steps_per_epoch,
        total_steps=expected_total_steps,
    )
    require_input_hashes(
        training_record,
        {
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
            "schedule_steps_per_epoch": str(expected_steps_per_epoch),
            "schedule_total_steps": str(expected_total_steps),
        },
    )
    if not all((args.acceptance_evaluator_checkpoint, args.acceptance_measurement_bundle, args.acceptance_measurement_evidence)):
        raise ValueError("current identity evaluation requires the independent acceptance measurement inputs")
    measurement = EvidenceRecordV2.from_dict(json.loads(Path(args.acceptance_measurement_evidence).read_text(encoding="utf-8")))
    if measurement.evidence_kind != "conditional_acceptance_measurement" or measurement.status != "passed":
        raise ValueError("acceptance measurement evidence is not qualified")
    _validate_current_evidence_source(training_record, measurement)
    measurement_bundle_hash = file_sha256(args.acceptance_measurement_bundle)
    secondary_hash = file_sha256(args.acceptance_evaluator_checkpoint)
    require_input_hashes(measurement, {"measurement_bundle": measurement_bundle_hash})
    bundle_payload = json.loads(Path(args.acceptance_measurement_bundle).read_text(encoding="utf-8"))
    if not isinstance(bundle_payload, dict) or bundle_payload.get("contract") != ACCEPTANCE_MEASUREMENT_CONTRACT:
        raise ValueError("acceptance measurement bundle is malformed")
    if bundle_payload.get("primary_checkpoint") != evaluator_hash or bundle_payload.get("secondary_checkpoint") != secondary_hash:
        raise ValueError("acceptance measurement evaluator hash mismatch")
    manifest_name = bundle_payload.get("test_manifest_filename")
    if not isinstance(manifest_name, str) or not manifest_name or Path(manifest_name).is_absolute() or ".." in Path(manifest_name).parts:
        raise ValueError("acceptance measurement bundle has an unsafe test manifest filename")
    test_manifest_path = Path(args.acceptance_measurement_bundle).parent / manifest_name
    test_manifest_hash = file_sha256(test_manifest_path)
    if test_manifest_hash != bundle_payload.get("test_manifest_hash") or measurement.input_hashes.get("test_manifest") != test_manifest_hash:
        raise ValueError("acceptance measurement test manifest hash mismatch")
    secondary_payload = safe_torch_load(args.acceptance_evaluator_checkpoint, expected_sha256=secondary_hash)
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("acceptance evaluator checkpoint has an incompatible architecture")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    require_input_hashes(
        measurement,
        {
            "primary_evaluator": evaluator_hash,
            "secondary_evaluator": secondary_hash,
            "measurement_bundle": measurement_bundle_hash,
            "test_manifest": test_manifest_hash,
        },
    )
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    if test_manifest.get("count") != int(test_labels.numel()) or test_manifest.get("images_sha256") != tensor_sha256(test_images) or test_manifest.get("labels_sha256") != tensor_sha256(test_labels):
        raise ValueError("local official MNIST test split does not match the qualified manifest")
    require_input_hashes(
        training_record,
        {
            "checkpoint": checkpoint_hash,
            "initialization_state_hash": manifest.initialization_state_hash,
        },
    )
    initialization_checkpoint_hash = training_record.input_hashes.get(
        "initialization_checkpoint"
    )
    if not isinstance(initialization_checkpoint_hash, str) or len(initialization_checkpoint_hash) != 64:
        raise ValueError("training evidence lacks initialization checkpoint binding")
    settings = config.conditional_generation
    assert settings is not None
    (
        raw_rows,
        label_rows,
        prediction_rows,
        secondary_prediction_rows,
        feature_rows,
        real_feature_rows,
        secondary_feature_rows,
        secondary_real_feature_rows,
        generated_view_rows,
        real_view_rows,
    ) = [], [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        batches = (torch.arange(start, min(start + settings.logical_batch_size, test_labels.numel())) for start in range(0, test_labels.numel(), settings.logical_batch_size))
        for batch_number, indices in enumerate(batches):
            labels = test_labels.index_select(0, indices).to(device)
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
            secondary_generated_features, secondary_logits = secondary(generated_view)
            real_view = test_images.index_select(0, indices).to(device)
            real_features, _ = evaluator(real_view)
            secondary_real_features, _ = secondary(real_view)
            raw_rows.append(raw.cpu())
            label_rows.append(labels.cpu())
            prediction_rows.append(logits.argmax(dim=1).cpu())
            secondary_prediction_rows.append(secondary_logits.argmax(dim=1).cpu())
            feature_rows.append(generated_features.cpu())
            real_feature_rows.append(real_features.cpu())
            secondary_feature_rows.append(secondary_generated_features.cpu())
            secondary_real_feature_rows.append(secondary_real_features.cpu())
            generated_view_rows.append(generated_view.cpu())
            real_view_rows.append(real_view.cpu())
    raw, labels, predictions, secondary_predictions = (
        torch.cat(raw_rows),
        torch.cat(label_rows),
        torch.cat(prediction_rows),
        torch.cat(secondary_prediction_rows),
    )
    generated_features, real_features = (
        torch.cat(feature_rows),
        torch.cat(real_feature_rows),
    )
    secondary_generated_features, secondary_real_features = (
        torch.cat(secondary_feature_rows),
        torch.cat(secondary_real_feature_rows),
    )
    generated_view, real_view = (
        torch.cat(generated_view_rows),
        torch.cat(real_view_rows),
    )
    accuracy = float((predictions == labels).to(torch.float32).mean().item())
    secondary_accuracy = float((secondary_predictions == labels).to(torch.float32).mean().item())
    recalls = [
        float(((predictions[labels == label] == label).to(torch.float32).mean()).item())
        for label in range(10)
    ]
    secondary_recalls = [
        float(((secondary_predictions[labels == label] == label).to(torch.float32).mean()).item())
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
    diversity_ratios = tuple(
        relative_diversity_ratios(generated_features, real_features, labels)
    )
    secondary_diversity_ratios = tuple(relative_diversity_ratios(secondary_generated_features, secondary_real_features, labels))
    coverage_ratio = conditional_coverage_ratio(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    secondary_coverage_ratio = conditional_coverage_ratio(
        secondary_generated_features,
        generated_view,
        secondary_real_features,
        real_view,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    checks = (
        GateResult("semantic_accuracy", accuracy >= 0.95, {"accuracy": accuracy}),
        GateResult("secondary_semantic_accuracy", secondary_accuracy >= 0.95, {"accuracy": secondary_accuracy}),
        GateResult(
            "per_class_recall", min(recalls) >= 0.90, {"minimum_recall": min(recalls)}
        ),
        GateResult("secondary_per_class_recall", min(secondary_recalls) >= 0.90, {"minimum_recall": min(secondary_recalls)}),
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
        GateResult("secondary_per_class_feature_diversity", all(0.70 <= value <= 1.30 for value in secondary_diversity_ratios), {f"label_{label}": value for label, value in enumerate(secondary_diversity_ratios)}),
        GateResult(
            "sinkhorn_coverage",
            coverage_ratio <= 1.5,
            {"coverage_ratio": coverage_ratio},
        ),
        GateResult("secondary_sinkhorn_coverage", secondary_coverage_ratio <= 1.5, {"coverage_ratio": secondary_coverage_ratio}),
        GateResult("generated_prediction_agreement", float((predictions == secondary_predictions).float().mean().item()) >= 0.95, {"agreement": float((predictions == secondary_predictions).float().mean().item())}),
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
            checkpoint_hash=checkpoint_hash,
        ),
        {
            "training_evidence": file_sha256(args.training_evidence),
            "checkpoint": checkpoint_hash,
            "initialization_checkpoint": initialization_checkpoint_hash,
            "initialization_state_hash": manifest.initialization_state_hash,
            "raw_cmos": file_sha256(raw_path),
            "labels": file_sha256(labels_path),
            "review_raw_cmos": review_artifacts.raw_tensor_sha256,
            "review_preview": review_artifacts.preview_sha256,
            "primary_evaluator": evaluator_hash,
            "secondary_evaluator": file_sha256(args.acceptance_evaluator_checkpoint),
            "measurement_bundle": measurement_bundle_hash,
            "measurement_evidence": measurement.evidence_hash,
            "test_manifest": file_sha256(test_manifest_path),
            "test_content_manifest": str(test_manifest.get("manifest_hash")),
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
    _validate_current_evidence_source(source)
    source.context.assert_matches(
        _context(
            config,
            evaluator_hash=source.context.evaluator_checkpoint_hash,
            split_hash=source.context.dataset_split_manifest_hash,
            checkpoint_hash=source.context.checkpoint_hash,
        )
    )
    source_checkpoint_hash = source.context.checkpoint_hash
    if not isinstance(source_checkpoint_hash, str) or len(source_checkpoint_hash) != 64:
        raise ValueError("conditional acceptance lacks checkpoint binding")
    source_initialization_state_hash = source.input_hashes.get(
        "initialization_state_hash"
    )
    if (
        not isinstance(source_initialization_state_hash, str)
        or len(source_initialization_state_hash) != 64
    ):
        raise ValueError("conditional acceptance lacks initialization state binding")
    require_input_hashes(
        source,
        {
            "checkpoint": source_checkpoint_hash,
            "initialization_state_hash": source_initialization_state_hash,
            "primary_evaluator": source.context.evaluator_checkpoint_hash,
            "secondary_evaluator": source.input_hashes.get("secondary_evaluator"),
            "measurement_bundle": source.input_hashes.get("measurement_bundle"),
            "measurement_evidence": source.input_hashes.get("measurement_evidence"),
            "test_manifest": source.input_hashes.get("test_manifest"),
        },
    )
    if not isinstance(source.input_hashes.get("initialization_checkpoint"), str):
        raise ValueError("conditional acceptance lacks initialization checkpoint binding")
    raw_hash = source.input_hashes.get("raw_cmos")
    labels_hash = source.input_hashes.get("labels")
    if not isinstance(raw_hash, str) or not isinstance(labels_hash, str):
        raise ValueError("conditional evaluation evidence lacks tensor bindings")
    raw = safe_torch_load(
        args.evaluation_raw_cmos,
        expected_sha256=raw_hash,
    )
    labels = safe_torch_load(
        args.evaluation_labels,
        expected_sha256=labels_hash,
    )
    if not torch.is_tensor(raw):
        raise ValueError(
            "evaluation raw CMOS does not match conditional acceptance evidence"
        )
    if (
        not torch.is_tensor(labels)
        or labels.ndim != 1
        or labels.shape[0] != raw.shape[0]
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
