"""Independent acceptance measurement for the formal conditional line.

The primary frozen evaluator remains the only instrument used by training.  A
smooth-pooling evaluator is qualified here as a second, frozen measurement
instrument and is only consumed by formal acceptance evaluation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import (
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.evaluator import (
    MNISTSplitManifest,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
)
from .runtime import (
    _context,
    _validate_current_evidence_source,
    _new_run,
    _require_current_acceptance,
    _require_identity,
    _split,
    _write_evidence,
    evaluator_checkpoint_hash,
    require_current_gate_action,
    require_input_hashes,
)
from .exposure import exposure_chain_id, EXPOSURE_CONTRACT


ACCEPTANCE_MEASUREMENT_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_acceptance_measurement_v1",
        "primary": "frozen_mnist_evaluator_v1",
        "secondary": SmoothMNISTEvaluator.architecture_kind,
        "secondary_seed": 2234,
        "secondary_epochs": 10,
        "secondary_batch_size": 256,
        "preprocessing": "fixed_exposure_28x28",
        "test_split": "official_mnist_test_10000_v1",
        "exposure_chain": EXPOSURE_CONTRACT,
    }
)


def _variants(images: torch.Tensor) -> dict[str, torch.Tensor]:
    eroded = -F.max_pool2d(-images, 3, stride=1, padding=1)
    return {
        "intensity_minus_2pct": (images * 0.98).clamp(0, 1),
        "intensity_plus_2pct": (images * 1.02).clamp(0, 1),
        "translate_x_plus_1": torch.roll(images, 1, dims=-1),
        "translate_y_plus_1": torch.roll(images, 1, dims=-2),
        "dilate": F.max_pool2d(images, 3, stride=1, padding=1),
        "erode": eroded,
    }


def _accuracy_recall(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, list[float], torch.Tensor]:
    with torch.no_grad():
        _, logits = model(images)
    prediction = logits.argmax(1)
    recall = [
        float((prediction[labels == label] == label).float().mean().item())
        for label in range(10)
    ]
    return float((prediction == labels).float().mean().item()), recall, prediction


def _train_smooth(images: torch.Tensor, labels: torch.Tensor, device: torch.device, *, epochs: int = 10, batch_size: int = 256) -> SmoothMNISTEvaluator:
    torch.manual_seed(2234)
    model = SmoothMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(
            images.shape[0],
            generator=torch.Generator(device="cpu").manual_seed(2234 + epoch),
        )
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images.index_select(0, indices).to(device))
            F.cross_entropy(logits, labels.index_select(0, indices).to(device)).backward()
            optimizer.step()
    return model.freeze()


def _test_manifest(images: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "kind": "official_mnist_test_10000_v1",
        "count": int(labels.numel()),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "images_sha256": tensor_sha256(images),
        "labels_sha256": tensor_sha256(labels),
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    return payload


def _require_official_test_split(images: torch.Tensor, labels: torch.Tensor) -> None:
    if (
        images.ndim != 4
        or tuple(images.shape[1:]) != (1, 28, 28)
        or labels.ndim != 1
        or labels.numel() != 10_000
        or images.shape[0] != 10_000
    ):
        raise ValueError("official MNIST test split must contain exactly 10000 28x28 samples")


def train_acceptance_evaluator(args, config) -> int:
    _require_identity(config)
    _require_current_acceptance(config)
    if args.epochs != 10 or args.batch_size != 256:
        raise ValueError("acceptance evaluator contract fixes epochs=10 and batch_size=256")
    primary_evidence = EvidenceRecordV2.from_dict(
        __import__("json").loads(
            Path(args.evaluator_qualification_evidence).read_text(encoding="utf-8")
        )
    )
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    primary_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    _validate_current_evidence_source(primary_evidence)
    require_current_gate_action(
        primary_evidence,
        kind="evaluator_qualification",
        action="train_acceptance_evaluator",
        expected_context=_context(
            config,
            evaluator_hash=primary_hash,
            split_hash=split.manifest_hash,
        ),
    )
    require_input_hashes(primary_evidence, {"evaluator_checkpoint": primary_hash})
    device = torch.device(config.run.device)
    model = _train_smooth(train_images, train_labels, device, epochs=args.epochs, batch_size=args.batch_size)
    accuracy, recall, _ = _accuracy_recall(model, validation_images.to(device), validation_labels.to(device))
    run = _new_run(config)
    checkpoint = run.root / "smooth_mnist_acceptance_evaluator.pt"
    atomic_torch_save(
        checkpoint,
        {
            "architecture_kind": SmoothMNISTEvaluator.architecture_kind,
            "state_dict": model.state_dict(),
            "split_manifest": asdict(split),
            "split_manifest_hash": split.manifest_hash,
            "validation_accuracy": accuracy,
            "validation_recall": recall,
            "training_seed": 2234,
            "training_epochs": args.epochs,
            "training_batch_size": args.batch_size,
            "contract": ACCEPTANCE_MEASUREMENT_CONTRACT,
        },
    )
    gate = GateResult(
        "secondary_validation",
        accuracy >= 0.985 and sum(recall) / len(recall) >= 0.985 and min(recall) >= 0.95,
        {"accuracy": accuracy, "macro_recall": sum(recall) / len(recall), "recall": recall},
    )
    record = EvidenceRecordV2(
        2,
        "acceptance_evaluator_training",
        config.run.evidence_tier,
        "passed" if gate.passed else "valid_failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config, evaluator_hash=file_sha256(checkpoint), split_hash=split.manifest_hash),
        {
            "acceptance_evaluator": file_sha256(checkpoint),
            "primary_evaluator": primary_hash,
            "primary_evaluator_qualification": primary_evidence.evidence_hash,
            "contract": ACCEPTANCE_MEASUREMENT_CONTRACT,
        },
        (gate,),
        ("qualify_acceptance_measurement",) if gate.passed else (),
        None if gate.passed else "secondary acceptance evaluator qualification failed",
    )
    _write_evidence(run, "acceptance_evaluator_training_evidence.json", record)
    print(run.root)
    return 0 if gate.passed else 1


def qualify_acceptance_measurement(args, config) -> int:
    _require_identity(config)
    _require_current_acceptance(config)
    device = torch.device(config.run.device)
    training_evidence = EvidenceRecordV2.from_dict(
        __import__("json").loads(Path(args.acceptance_evaluator_training_evidence).read_text(encoding="utf-8"))
    )
    _validate_current_evidence_source(training_evidence)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    primary, primary_payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**primary_payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("primary evaluator split mismatch")
    secondary_payload = safe_torch_load(args.acceptance_evaluator_checkpoint)
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("acceptance evaluator checkpoint has an incompatible architecture")
    if secondary_payload.get("contract") != ACCEPTANCE_MEASUREMENT_CONTRACT:
        raise ValueError("acceptance evaluator contract mismatch")
    if MNISTSplitManifest(**secondary_payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("acceptance evaluator split mismatch")
    secondary_hash = file_sha256(args.acceptance_evaluator_checkpoint)
    require_current_gate_action(
        training_evidence,
        kind="acceptance_evaluator_training",
        action="qualify_acceptance_measurement",
        expected_context=_context(
            config,
            evaluator_hash=secondary_hash,
            split_hash=split.manifest_hash,
        ),
    )
    if training_evidence.input_hashes.get("primary_evaluator") != file_sha256(args.evaluator_checkpoint):
        raise ValueError("acceptance evaluator training evidence primary checkpoint mismatch")
    if training_evidence.input_hashes.get("acceptance_evaluator") != secondary_hash:
        raise ValueError("acceptance evaluator training evidence does not bind checkpoint")
    if training_evidence.context.dataset_split_manifest_hash != split.manifest_hash:
        raise ValueError("acceptance evaluator training evidence split mismatch")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    primary_accuracy, primary_recall, primary_prediction = _accuracy_recall(primary, validation_images.to(device), validation_labels.to(device))
    secondary_accuracy, secondary_recall, secondary_prediction = _accuracy_recall(secondary, validation_images.to(device), validation_labels.to(device))
    agreement = float((primary_prediction == secondary_prediction).float().mean().item())
    consistency: dict[str, dict[str, float]] = {}
    validation = validation_images.to(device)
    for name, variant in _variants(validation).items():
        p = _accuracy_recall(primary, variant, validation_labels.to(device))[2]
        s = _accuracy_recall(secondary, variant, validation_labels.to(device))[2]
        consistency[name] = {
            "primary": float((p == primary_prediction).float().mean().item()),
            "secondary": float((s == secondary_prediction).float().mean().item()),
        }
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    _require_official_test_split(test_images, test_labels)
    test_manifest = _test_manifest(test_images, test_labels)
    run = _new_run(config)
    test_manifest_path = run.root / "mnist_test_manifest.json"
    atomic_write_json(test_manifest_path, test_manifest)
    bundle = {
        "schema_version": 1,
        "contract": ACCEPTANCE_MEASUREMENT_CONTRACT,
        "primary_checkpoint": file_sha256(args.evaluator_checkpoint),
        "secondary_checkpoint": secondary_hash,
        "split_manifest_hash": split.manifest_hash,
        "test_manifest_hash": file_sha256(test_manifest_path),
        "test_content_manifest_hash": test_manifest["manifest_hash"],
        "test_manifest_filename": test_manifest_path.name,
        "training_evidence_hash": training_evidence.evidence_hash,
        "preprocessing": "fixed_exposure_28x28",
        "feature_dim": 128,
    }
    bundle_path = run.root / "conditional_acceptance_measurement_bundle.json"
    atomic_write_json(bundle_path, bundle)
    bundle_hash = file_sha256(bundle_path)
    chain_id = exposure_chain_id(
        config,
        primary_evaluator=file_sha256(args.evaluator_checkpoint),
        secondary_evaluator=secondary_hash,
        test_manifest=file_sha256(test_manifest_path),
        measurement_bundle=bundle_hash,
    )
    genesis = EvidenceRecordV2(
        2,
        "formal_evaluation_exposure",
        config.run.evidence_tier,
        "passed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=file_sha256(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
        ),
        {
            "exposure_contract": EXPOSURE_CONTRACT,
            "exposure_chain_id": chain_id,
            "exposure_sequence": "0",
            "primary_evaluator": file_sha256(args.evaluator_checkpoint),
            "secondary_evaluator": secondary_hash,
            "test_manifest": file_sha256(test_manifest_path),
            "measurement_evidence": "genesis",
            "measurement_bundle": bundle_hash,
            "formal_evaluation_id": chain_id,
        },
        (GateResult("exposure_genesis_declared", True, {}),),
        ("record_next_official_test_exposure",),
        None,
    )
    _write_evidence(run, "formal_evaluation_exposure_genesis_evidence.json", genesis)
    gates = (
        GateResult("primary_validation", primary_accuracy >= 0.985 and sum(primary_recall) / 10 >= 0.985 and min(primary_recall) >= 0.95, {"accuracy": primary_accuracy, "macro_recall": sum(primary_recall) / 10, "recall": primary_recall}),
        GateResult("secondary_validation", secondary_accuracy >= 0.985 and sum(secondary_recall) / 10 >= 0.985 and min(secondary_recall) >= 0.95, {"accuracy": secondary_accuracy, "macro_recall": sum(secondary_recall) / 10, "recall": secondary_recall}),
        GateResult("cross_evaluator_agreement", agreement >= 0.98, {"agreement": agreement}),
        GateResult("intensity_consistency", all(consistency[name][which] >= 0.995 for name in ("intensity_minus_2pct", "intensity_plus_2pct") for which in ("primary", "secondary")), consistency),
        GateResult("translation_consistency", all(consistency[name][which] >= 0.975 for name in ("translate_x_plus_1", "translate_y_plus_1") for which in ("primary", "secondary")), consistency),
        GateResult("test_manifest_only", True, {"test_count": float(test_labels.numel()), "test_inference": False}),
    )
    passed = all(gate.passed for gate in gates)
    context = _context(config, evaluator_hash=file_sha256(args.evaluator_checkpoint), split_hash=split.manifest_hash)
    record = EvidenceRecordV2(
        2,
        "conditional_acceptance_measurement",
        config.run.evidence_tier,
        "passed" if passed else "valid_failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "primary_evaluator": file_sha256(args.evaluator_checkpoint),
            "secondary_evaluator": secondary_hash,
            "measurement_bundle": file_sha256(bundle_path),
            "test_manifest": file_sha256(test_manifest_path),
            "test_content_manifest": str(test_manifest["manifest_hash"]),
            "acceptance_evaluator_training_evidence": training_evidence.evidence_hash,
            "contract": ACCEPTANCE_MEASUREMENT_CONTRACT,
            "exposure_genesis": genesis.evidence_hash,
            "exposure_chain_id": chain_id,
        },
        gates,
        ("conditional_injection_admission", "evaluate") if passed else (),
        None if passed else "independent acceptance measurement qualification failed",
    )
    _write_evidence(run, "conditional_acceptance_measurement_evidence.json", record)
    print(run.root)
    return 0 if passed else 1
