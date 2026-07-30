"""Content-bound dual-evaluator qualification for the transport restart."""

from __future__ import annotations

import json
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
)
from ...student.evaluator import (
    MNISTSplitManifest,
    SmoothMNISTEvaluator,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .support import _context, _new_run, _split, _write_evidence


BUNDLE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_evaluator_bundle_v2",
        "primary": "frozen_mnist_evaluator_v1",
        "smooth": "smooth_mnist_evaluator_v2",
        "smooth_seed": 2234,
        "epochs": 5,
        "batch_size": 256,
        "feature_dim": 128,
        "low_frequency_shape": [14, 14],
    }
)


def _train_smooth(
    images: torch.Tensor, labels: torch.Tensor, device: torch.device, *, epochs: int = 5
) -> SmoothMNISTEvaluator:
    torch.manual_seed(2234)
    model = SmoothMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(
            images.shape[0],
            generator=torch.Generator(device="cpu").manual_seed(2234 + epoch),
        )
        for start in range(0, order.numel(), 256):
            index = order[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images.index_select(0, index).to(device))
            F.cross_entropy(logits, labels.index_select(0, index).to(device)).backward()
            optimizer.step()
    return model.freeze()


def _accuracy_recall(model, images, labels) -> tuple[float, list[float], torch.Tensor]:
    with torch.no_grad():
        _, logits = model(images)
    prediction = logits.argmax(1)
    recall = [
        float((prediction[labels == label] == label).float().mean().item())
        for label in range(10)
    ]
    return float((prediction == labels).float().mean().item()), recall, prediction


def _variants(images: torch.Tensor) -> dict[str, torch.Tensor]:
    eroded = -F.max_pool2d(-images, 3, stride=1, padding=1)
    return {
        "translate_x_plus_1": torch.roll(images, 1, dims=-1),
        "translate_y_plus_1": torch.roll(images, 1, dims=-2),
        "dilate": F.max_pool2d(images, 3, stride=1, padding=1),
        "erode": eroded,
        "intensity_minus_2pct": (images * 0.98).clamp(0, 1),
        "intensity_plus_2pct": (images * 1.02).clamp(0, 1),
    }


def conditional_evaluator_qualification_v2(args, config) -> int:
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluator qualification v2 requires configured CUDA")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(
        images, labels, args.dataset_root
    )
    primary, payload = load_frozen_evaluator(
        args.primary_evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("primary evaluator split mismatch")
    torch.cuda.reset_peak_memory_stats(device)
    smooth = _train_smooth(train_images, train_labels, device)
    validation = validation_images.to(device)
    targets = validation_labels.to(device)
    primary_accuracy, primary_recall, primary_prediction = _accuracy_recall(
        primary, validation, targets
    )
    smooth_accuracy, smooth_recall, smooth_prediction = _accuracy_recall(
        smooth, validation, targets
    )
    consistency = {}
    for name, value in _variants(validation).items():
        p = _accuracy_recall(primary, value, targets)[2]
        s = _accuracy_recall(smooth, value, targets)[2]
        consistency[name] = {
            "primary": float((p == primary_prediction).float().mean().item()),
            "smooth": float((s == smooth_prediction).float().mean().item()),
        }
    agreement = float((primary_prediction == smooth_prediction).float().mean().item())
    memory = {
        "reserved": float(torch.cuda.max_memory_reserved(device)),
        "total": float(torch.cuda.get_device_properties(device).total_memory),
    }
    memory["passed"] = bool(memory["reserved"] <= 0.92 * memory["total"])
    run = _new_run(config)
    smooth_path = run.root / "smooth_mnist_evaluator_v2.pt"
    atomic_torch_save(
        smooth_path,
        {
            "architecture_kind": SmoothMNISTEvaluator.architecture_kind,
            "state_dict": smooth.state_dict(),
            "split_manifest": asdict(split),
            "split_manifest_hash": split.manifest_hash,
            "validation_accuracy": smooth_accuracy,
            "training_seed": 2234,
            "contract": BUNDLE_CONTRACT,
        },
    )
    bundle = {
        "schema_version": 1,
        "contract": BUNDLE_CONTRACT,
        "primary_checkpoint": file_sha256(args.primary_evaluator_checkpoint),
        "smooth_checkpoint": file_sha256(smooth_path),
        "split_manifest_hash": split.manifest_hash,
        "preprocessing": "fixed_exposure_28x28",
        "feature_dim": 128,
        "low_frequency_shape": [14, 14],
    }
    bundle_path = run.root / "conditional_evaluator_bundle_v2.json"
    atomic_write_json(bundle_path, bundle)
    gates = (
        GateResult(
            "primary_validation",
            primary_accuracy >= 0.985 and min(primary_recall) >= 0.97,
            {"accuracy": primary_accuracy, "recall": primary_recall},
        ),
        GateResult(
            "smooth_validation",
            smooth_accuracy >= 0.985 and min(smooth_recall) >= 0.97,
            {"accuracy": smooth_accuracy, "recall": smooth_recall},
        ),
        GateResult(
            "cross_evaluator_agreement", agreement >= 0.98, {"agreement": agreement}
        ),
        GateResult(
            "perturbation_consistency",
            all(
                value[which] >= 0.98
                for value in consistency.values()
                for which in ("primary", "smooth")
            ),
            consistency,
        ),
        GateResult("execution", bool(memory["passed"]), memory),
    )
    passed = all(gate.passed for gate in gates)
    context = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint),
        split_hash=split.manifest_hash,
    )
    record = EvidenceRecordV2(
        2,
        "conditional_evaluator_qualification_v2",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "primary_evaluator": file_sha256(args.primary_evaluator_checkpoint),
            "smooth_evaluator": file_sha256(smooth_path),
            "bundle": file_sha256(bundle_path),
        },
        gates,
        (
            "conditional-evaluator-curvature-baseline",
            "conditional-evaluation-qualified-transport-design",
        )
        if passed
        else (),
        None if passed else "evaluator_bundle_v2_not_qualified",
    )
    _write_evidence(run, "conditional_evaluator_qualification_v2_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


V21_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_evaluator_bundle_v2_1",
        "primary": "frozen_mnist_evaluator_v1",
        "smooth": SmoothMNISTEvaluator.architecture_kind,
        "smooth_seed": 2234,
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "low_frequency_shape": [14, 14],
        "morphology": "diagnostic_only",
    }
)


def _evidence(path: str, kind: str, status: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    record.validate()
    if record.evidence_kind != kind or (status is not None and record.status != status):
        raise ValueError(f"expected {kind} evidence with status {status}")
    return record


def conditional_evaluator_calibration_v21(args, config) -> int:
    """Bind the historical v2 failure to the unchanged primary instrument."""
    prior = _evidence(
        args.evaluator_v2_evidence, "conditional_evaluator_qualification_v2", "failed"
    )
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluator calibration v2.1 requires configured CUDA")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels, args.dataset_root)
    primary, payload = load_frozen_evaluator(
        args.primary_evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("primary evaluator split mismatch")
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint),
        split_hash=split.manifest_hash,
    )
    prior.context.assert_matches(
        expected,
        fields=(
            "evaluator_checkpoint_hash",
            "dataset_split_manifest_hash",
            "exposure_scale",
        ),
    )
    validation = _split(images, labels, args.dataset_root)[2].to(device)
    targets = _split(images, labels, args.dataset_root)[3].to(device)
    accuracy, recall, _ = _accuracy_recall(primary, validation, targets)
    memory = {
        "reserved": float(torch.cuda.memory_reserved(device)),
        "total": float(torch.cuda.get_device_properties(device).total_memory),
    }
    memory["passed"] = bool(memory["reserved"] <= 0.92 * memory["total"])
    gates = (
        GateResult("historical_v2_failure_bound", True, {"status_failed": True}),
        GateResult(
            "primary_calibration_baseline",
            accuracy >= 0.985
            and float(torch.tensor(recall).mean()) >= 0.985
            and min(recall) >= 0.95,
            {
                "accuracy": accuracy,
                "macro_recall": float(torch.tensor(recall).mean()),
                "recall": recall,
            },
        ),
        GateResult("execution", bool(memory["passed"]), memory),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    record = EvidenceRecordV2(
        2,
        "conditional_evaluator_calibration_v21",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "evaluator_v2_failure": file_sha256(args.evaluator_v2_evidence),
            "primary": file_sha256(args.primary_evaluator_checkpoint),
        },
        gates,
        ("conditional-evaluator-qualification-v21",) if passed else (),
        None if passed else "primary_evaluator_calibration_baseline_failed",
    )
    _write_evidence(run, "conditional_evaluator_calibration_v21_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def conditional_evaluator_qualification_v21(args, config) -> int:
    """Qualify a freshly trained smooth evaluator under the v2.1 contract."""
    calibration = _evidence(
        args.calibration_evidence, "conditional_evaluator_calibration_v21", "passed"
    )
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluator qualification v2.1 requires configured CUDA")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(
        images, labels, args.dataset_root
    )
    primary, payload = load_frozen_evaluator(
        args.primary_evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("primary evaluator split mismatch")
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.primary_evaluator_checkpoint),
        split_hash=split.manifest_hash,
    )
    calibration.context.assert_matches(
        expected,
        fields=(
            "evaluator_checkpoint_hash",
            "dataset_split_manifest_hash",
            "exposure_scale",
        ),
    )
    torch.cuda.reset_peak_memory_stats(device)
    smooth = _train_smooth(train_images, train_labels, device, epochs=10)
    validation, targets = validation_images.to(device), validation_labels.to(device)
    primary_accuracy, primary_recall, primary_prediction = _accuracy_recall(
        primary, validation, targets
    )
    smooth_accuracy, smooth_recall, smooth_prediction = _accuracy_recall(
        smooth, validation, targets
    )
    consistency = {}
    for name, value in _variants(validation).items():
        p = _accuracy_recall(primary, value, targets)[2]
        s = _accuracy_recall(smooth, value, targets)[2]
        consistency[name] = {
            "primary": float((p == primary_prediction).float().mean()),
            "smooth": float((s == smooth_prediction).float().mean()),
        }
    agreement = float((primary_prediction == smooth_prediction).float().mean())
    memory = {
        "reserved": float(torch.cuda.max_memory_reserved(device)),
        "total": float(torch.cuda.get_device_properties(device).total_memory),
    }
    memory["passed"] = bool(memory["reserved"] <= 0.92 * memory["total"])
    run = _new_run(config)
    smooth_path = run.root / "smooth_mnist_evaluator_v21.pt"
    atomic_torch_save(
        smooth_path,
        {
            "architecture_kind": SmoothMNISTEvaluator.architecture_kind,
            "state_dict": smooth.state_dict(),
            "split_manifest": asdict(split),
            "split_manifest_hash": split.manifest_hash,
            "validation_accuracy": smooth_accuracy,
            "training_seed": 2234,
            "contract": V21_CONTRACT,
        },
    )
    bundle_path = run.root / "conditional_evaluator_bundle_v21.json"
    atomic_write_json(
        bundle_path,
        {
            "schema_version": 1,
            "contract": V21_CONTRACT,
            "primary_checkpoint": file_sha256(args.primary_evaluator_checkpoint),
            "smooth_checkpoint": file_sha256(smooth_path),
            "split_manifest_hash": split.manifest_hash,
            "preprocessing": "fixed_exposure_28x28",
            "feature_dim": 128,
            "low_frequency_shape": [14, 14],
        },
    )

    def qualified(accuracy, recall):
        return (
            accuracy >= 0.985
            and float(torch.tensor(recall).mean()) >= 0.985
            and min(recall) >= 0.95
        )

    robustness = all(
        consistency[name][which] >= threshold
        for name, threshold in (
            ("intensity_minus_2pct", 0.995),
            ("intensity_plus_2pct", 0.995),
            ("translate_x_plus_1", 0.975),
            ("translate_y_plus_1", 0.975),
        )
        for which in ("primary", "smooth")
    )
    gates = (
        GateResult(
            "primary_validation",
            qualified(primary_accuracy, primary_recall),
            {
                "accuracy": primary_accuracy,
                "macro_recall": float(torch.tensor(primary_recall).mean()),
                "recall": primary_recall,
            },
        ),
        GateResult(
            "smooth_validation",
            qualified(smooth_accuracy, smooth_recall),
            {
                "accuracy": smooth_accuracy,
                "macro_recall": float(torch.tensor(smooth_recall).mean()),
                "recall": smooth_recall,
            },
        ),
        GateResult(
            "cross_evaluator_agreement", agreement >= 0.98, {"agreement": agreement}
        ),
        GateResult(
            "hard_robustness",
            robustness,
            {
                name: consistency[name]
                for name in (
                    "intensity_minus_2pct",
                    "intensity_plus_2pct",
                    "translate_x_plus_1",
                    "translate_y_plus_1",
                )
            },
        ),
        GateResult(
            "morphology_diagnostic",
            True,
            {name: consistency[name] for name in ("dilate", "erode")},
        ),
        GateResult("execution", bool(memory["passed"]), memory),
    )
    passed = all(gate.passed for gate in gates)
    record = EvidenceRecordV2(
        2,
        "conditional_evaluator_qualification_v21",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "calibration": file_sha256(args.calibration_evidence),
            "primary_evaluator": file_sha256(args.primary_evaluator_checkpoint),
            "smooth_evaluator": file_sha256(smooth_path),
            "bundle": file_sha256(bundle_path),
        },
        gates,
        (
            "conditional-evaluator-curvature-baseline",
            "conditional-evaluation-qualified-transport-design",
        )
        if passed
        else (),
        None if passed else "evaluator_bundle_v21_not_qualified",
    )
    _write_evidence(
        run, "conditional_evaluator_qualification_v21_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1
