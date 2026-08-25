"""Current v3 lifecycle for hierarchical input and balanced quadrature readout.

This module is deliberately separate from the v2 identity runtime.  It owns
the v3 checkpoint/evidence boundary, while the optical primitives remain in
``optics.quadrature_receiver`` and the generator remains in ``student``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ...config import V2Config
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.conditional import (
    camera_code_fraction,
    conditional_sinkhorn_loss,
    diversity_hinge,
    relative_diversity_ratios,
    relative_diversity_worst_class_loss,
)
from ...student.conditional_acceptance import conditional_coverage_ratio, memorization_statistics
from ...student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from ...student.checkpoints import state_content_hash, state_signature
from .._runtime_shared import split_mnist
from .spatial_quadrature_runtime import (
    SPATIAL_ARCHITECTURE,
    SPATIAL_GENERATOR_CONTRACT,
    SPATIAL_RECEIVER_CONTRACT,
    SpatialQuadratureBundle,
    freeze_spatial_generator,
    seeded_spatial_bundle,
    spatial_codes,
    spatial_forward_batched,
    spatial_parameter_groups,
    unfreeze_spatial_generator,
)


ELECTRONIC_PREFLIGHT_KIND = "spatial_electronic_preflight"
ELECTRONIC_TRAINING_KIND = "spatial_electronic_training"
ELECTRONIC_ADMISSION_KIND = "spatial_electronic_admission"
QUADRATURE_PREFLIGHT_KIND = "spatial_quadrature_preflight"
QUADRATURE_SMOKE_KIND = "spatial_quadrature_smoke"
SPATIAL_TRAINING_KIND = "spatial_quadrature_training"
SPATIAL_ACCEPTANCE_KIND = "spatial_quadrature_acceptance"


def _require_spatial(config: V2Config) -> None:
    if config.architecture != SPATIAL_ARCHITECTURE or config.spatial_quadrature_generation is None:
        raise ValueError("this command requires conditional_spatial_quadrature_v3")


def _device(config: V2Config) -> torch.device:
    return torch.device(config.run.device)


def _config_hash(config: V2Config) -> str:
    return canonical_sha256(config.model_dump(mode="json"))


def _system_hash(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def _git_state() -> tuple[str, str | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    return commit, canonical_sha256(status) if status else None


def _split(dataset_root: str):
    images, labels = load_local_mnist(dataset_root, train=True)
    return (*split_mnist(images, labels, dataset_root), images, labels)


def _context(config: V2Config, *, evaluator_hash: str | None = None, split_hash: str | None = None, checkpoint_hash: str | None = None) -> EvidenceContextV2:
    settings = config.spatial_quadrature_generation
    assert settings is not None
    return EvidenceContextV2(
        _system_hash(config),
        _config_hash(config),
        checkpoint_hash=checkpoint_hash,
        evaluator_checkpoint_hash=evaluator_hash,
        dataset_split_manifest_hash=split_hash,
        exposure_scale=settings.receiver_exposure,
    )


def _write_record(
    config: V2Config,
    output: str | Path,
    *,
    kind: str,
    status: str,
    gates: list[GateResult],
    input_hashes: dict[str, str],
    permitted: tuple[str, ...] = (),
    evaluator_hash: str | None = None,
    split_hash: str | None = None,
    checkpoint_hash: str | None = None,
    reason: str | None = None,
) -> Path:
    commit, dirty = _git_state()
    record = EvidenceRecordV2(
        2,
        kind,
        config.run.evidence_tier,
        status,
        commit,
        dirty,
        _config_hash(config),
        _context(config, evaluator_hash=evaluator_hash, split_hash=split_hash, checkpoint_hash=checkpoint_hash),
        input_hashes,
        tuple(gates),
        permitted,
        reason,
    )
    record.validate()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record.to_dict())
    return path


def _load_record(path: str | Path, *, kind: str) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed")
    return record


def _bank(images: torch.Tensor, labels: torch.Tensor, *, seed: int, half: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, targets = [], []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2 * half:
            raise ValueError("validation bank does not contain two complete class banks")
        shuffled = indices.index_select(0, torch.randperm(indices.numel(), generator=generator))
        rows.append(shuffled[:half])
        targets.append(torch.full((half,), label, dtype=torch.long))
    selected = torch.cat(rows)
    return images.index_select(0, selected), torch.cat(targets)


def _electronic_views(bundle: SpatialQuadratureBundle, codes: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    command = bundle.generator(codes, labels)
    return F.adaptive_avg_pool2d(command, (28, 28)), command


def _metric_summary(
    evaluator: FrozenMNISTEvaluator,
    views: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    *,
    commands: torch.Tensor,
    train_views: torch.Tensor | None = None,
    train_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    features, logits = evaluator(views)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    prediction = logits.argmax(dim=1)
    recalls = [float((prediction[labels == label] == label).float().mean().item()) for label in range(10)]
    summary: dict[str, Any] = {
        "accuracy": float((prediction == labels).float().mean().item()),
        "recall": recalls,
        "minimum_recall": min(recalls),
        "code_fraction": float(camera_code_fraction(commands, labels).item()),
        "diversity": relative_diversity_ratios(features, real_features, labels),
        "finite": bool(torch.isfinite(views).all() and torch.isfinite(features).all() and torch.isfinite(logits).all()),
        "saturation": float(commands.gt(0.99).float().mean().item()),
        "prediction_histogram": torch.bincount(prediction, minlength=10).tolist(),
    }
    summary["coverage"] = conditional_coverage_ratio(features, views, real_features, real_views, labels, epsilon=0.05, iterations=20)
    shifted = (labels + 1) % 10
    summary["label_shift_accuracy"] = float((prediction == shifted).float().mean().item())
    if train_views is not None and train_labels is not None:
        with torch.no_grad():
            train_features, _ = evaluator(train_views)
        summary["memorization"] = memorization_statistics(features, labels, train_features, train_labels)
    return summary


def _load_secondary_evaluator(path: str | Path, *, device: torch.device, split_hash: str) -> SmoothMNISTEvaluator:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("secondary evaluator checkpoint has an incompatible architecture")
    if payload.get("split_manifest_hash") != split_hash:
        raise ValueError("secondary evaluator checkpoint split mismatch")
    if float(payload.get("validation_accuracy", 0.0)) < 0.985:
        raise ValueError("secondary evaluator validation accuracy is below the qualification gate")
    model = SmoothMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.freeze()


def _electronic_pass(metrics: dict[str, Any]) -> bool:
    memo = metrics.get("memorization", {}).get("per_class", [])
    memo_pass = bool(memo) and all(float(row["anomalous_rate"]) <= 0.05 for row in memo)
    return bool(
        metrics["finite"]
        and metrics["accuracy"] >= 0.95
        and metrics["minimum_recall"] >= 0.90
        and metrics["code_fraction"] >= 0.25
        and all(0.70 <= value <= 1.30 for value in metrics["diversity"])
        and metrics["coverage"] <= 1.50
        and metrics["saturation"] <= 0.08
        and memo_pass
        and metrics["label_shift_accuracy"] <= 0.20
    )


def _save_generator_checkpoint(path: str | Path, bundle: SpatialQuadratureBundle, config: V2Config, split_hash: str, evaluator_hash: str, *, extra: dict[str, Any] | None = None) -> str:
    state = bundle.generator.state_dict()
    payload = {
        "manifest": {
            "schema_version": 3,
            "architecture_kind": SPATIAL_ARCHITECTURE,
            "generator_contract": SPATIAL_GENERATOR_CONTRACT,
            "receiver_contract": SPATIAL_RECEIVER_CONTRACT,
            "config_hash": _config_hash(config),
            "system_signature": _system_hash(config),
            "dataset_split_manifest_hash": split_hash,
            "evaluator_checkpoint_hash": evaluator_hash,
            "state_signature": state_signature(state),
            "state_content_hash": state_content_hash(state),
        },
        "state_dict": state,
        "extra": dict(extra or {}),
    }
    atomic_torch_save(path, payload)
    return file_sha256(path)


def _load_generator_checkpoint(path: str | Path, bundle: SpatialQuadratureBundle, config: V2Config, *, split_hash: str | None = None, evaluator_hash: str | None = None) -> dict[str, Any]:
    payload = safe_torch_load(path)
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(manifest, dict) or not isinstance(state, dict):
        raise ValueError("invalid v3 spatial checkpoint")
    if manifest.get("schema_version") != 3 or manifest.get("architecture_kind") != SPATIAL_ARCHITECTURE:
        raise ValueError("checkpoint is not a v3 spatial checkpoint")
    if manifest.get("config_hash") != _config_hash(config) or manifest.get("system_signature") != _system_hash(config):
        raise ValueError("v3 spatial checkpoint config/system lineage mismatch")
    if split_hash is not None and manifest.get("dataset_split_manifest_hash") != split_hash:
        raise ValueError("v3 spatial checkpoint split mismatch")
    if evaluator_hash is not None and manifest.get("evaluator_checkpoint_hash") != evaluator_hash:
        raise ValueError("v3 spatial checkpoint evaluator mismatch")
    if state_signature(state) != manifest.get("state_signature") or state_content_hash(state) != manifest.get("state_content_hash"):
        raise ValueError("v3 spatial checkpoint state hash mismatch")
    bundle.generator.load_state_dict(state, strict=True)
    return payload


def _preflight_batch(labels: torch.Tensor, *, seed: int, device: torch.device) -> torch.Tensor:
    rows = []
    for label in range(10):
        rows.append(torch.full((labels.numel() // 10,), label, dtype=torch.long))
    return torch.cat(rows).to(device)


def spatial_electronic_preflight(args, config: V2Config) -> int:
    _require_spatial(config)
    _, train_labels, _, _, split, _, _ = _split(args.dataset_root)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    labels = _preflight_batch(torch.empty(50, dtype=torch.long), seed=config.spatial_quadrature_generation.training_seed, device=device)
    codes = spatial_codes(labels.numel(), device=device, seed=config.spatial_quadrature_generation.training_seed)
    optimizer = torch.optim.AdamW(bundle.generator.parameters(), lr=config.spatial_quadrature_generation.generator_learning_rate)
    before = {name: value.detach().clone() for name, value in bundle.generator.state_dict().items()}
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        command = bundle.generator(codes, labels)
        loss = command.mean() + 0.01 * bundle.generator.smoothness_penalty()
        if not torch.isfinite(loss):
            raise ValueError("spatial electronic preflight produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    changed = any(not torch.equal(before[name], value.detach()) for name, value in bundle.generator.state_dict().items())
    gates = [
        GateResult("finite_update_loss", bool(all(torch.isfinite(torch.tensor(losses)))), {"steps": 3.0}),
        GateResult("generator_updated", changed, {}),
        GateResult("label_schedule_complete", labels.numel() == 50 and labels.unique().numel() == 10, {"count": float(labels.numel())}),
    ]
    passed = all(gate.passed for gate in gates)
    _write_record(config, args.output, kind=ELECTRONIC_PREFLIGHT_KIND, status="passed" if passed else "invalid_harness", gates=gates, input_hashes={"dataset_split": split.manifest_hash}, split_hash=split.manifest_hash, permitted=("train_spatial_electronic",) if passed else (), reason=None if passed else "spatial electronic preflight failed")
    return 0


def _electronic_training_loss(evaluator: FrozenMNISTEvaluator, bundle: SpatialQuadratureBundle, codes: torch.Tensor, labels: torch.Tensor, real_views: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    views, commands = _electronic_views(bundle, codes, labels)
    features, logits = evaluator(views)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(features, views, real_features, real_views, labels, epsilon=0.05, iterations=20)
    semantic = F.cross_entropy(logits, labels)
    diversity = relative_diversity_worst_class_loss(features, real_features, labels)
    code_term = diversity_hinge(features, codes, labels)
    fraction = torch.relu(0.25 - camera_code_fraction(commands, labels)).square()
    saturation = commands.gt(0.99).float().mean()
    total = sinkhorn + 0.5 * semantic + 0.25 * diversity + 0.1 * code_term + fraction + 0.1 * saturation + 1.0e-5 * bundle.generator.smoothness_penalty()
    return total, {"total": float(total.detach().item()), "sinkhorn": float(sinkhorn.detach().item()), "semantic": float(semantic.detach().item()), "diversity": float(diversity.detach().item())}


def train_spatial_electronic(args, config: V2Config) -> int:
    _require_spatial(config)
    preflight = _load_record(args.preflight_evidence, kind=ELECTRONIC_PREFLIGHT_KIND)
    images, train_labels, _, _, split, _, _ = _split(args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    if preflight.input_hashes.get("dataset_split") != split.manifest_hash:
        raise ValueError("spatial electronic preflight split binding mismatch")
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=_device(config))
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    optimizer = torch.optim.AdamW(bundle.generator.parameters(), lr=config.spatial_quadrature_generation.generator_learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(config.spatial_quadrature_generation.electronic_epochs):
        for batch_number, indices in enumerate(balanced_batches(train_labels, batch_size=config.spatial_quadrature_generation.logical_batch_size, seed=config.spatial_quadrature_generation.training_seed + epoch)):
            labels = train_labels.index_select(0, indices).to(device)
            real_views = images.index_select(0, indices).to(device)
            codes = spatial_codes(labels.numel(), device=device, seed=100000 * (epoch + 1) + batch_number)
            optimizer.zero_grad(set_to_none=True)
            loss, summary = _electronic_training_loss(evaluator, bundle, codes, labels, real_views)
            if not torch.isfinite(loss):
                raise ValueError("spatial electronic training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            history.append({"epoch": float(epoch + 1), "step": float(len(history) + 1), **summary})
    checkpoint_hash = _save_generator_checkpoint(args.output, bundle, config, split.manifest_hash, evaluator_hash, extra={"history": history, "stage": "electronic"})
    gates = [GateResult("finite_training", bool(all(torch.isfinite(torch.tensor(row["total"])).item() for row in history)), {"steps": float(len(history))})]
    _write_record(config, str(Path(args.output).with_suffix(".evidence.json")), kind=ELECTRONIC_TRAINING_KIND, status="passed", gates=gates, input_hashes={"preflight_evidence": preflight.evidence_hash, "checkpoint": checkpoint_hash, "dataset_split": split.manifest_hash, "evaluator": evaluator_hash}, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash, permitted=("admit_spatial_electronic",))
    return 0


def admit_spatial_electronic(args, config: V2Config) -> int:
    _require_spatial(config)
    images, train_labels, validation_images, validation_labels, split, _, _ = _split(args.dataset_root)
    primary_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    secondary_hash = evaluator_checkpoint_hash(args.secondary_evaluator_checkpoint)
    primary, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=_device(config))
    secondary = _load_secondary_evaluator(args.secondary_evaluator_checkpoint, device=_device(config), split_hash=split.manifest_hash)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    _load_generator_checkpoint(args.electronic_checkpoint, bundle, config, split_hash=split.manifest_hash, evaluator_hash=primary_hash)
    bank_metrics = []
    for bank, seed in enumerate(config.spatial_quadrature_generation.validation_bank_seeds):
        bank_images, bank_labels = _bank(validation_images, validation_labels, seed=seed, half=250)
        bank_images = bank_images.to(device)
        bank_labels = bank_labels.to(device)
        codes = spatial_codes(bank_labels.numel(), device=device, seed=int(seed))
        views, commands = _electronic_views(bundle, codes, bank_labels)
        primary_metrics = _metric_summary(primary, views, bank_labels, bank_images, commands=commands, train_views=images.to(device), train_labels=train_labels.to(device))
        secondary_metrics = _metric_summary(secondary, views, bank_labels, bank_images, commands=commands)
        bank_metrics.append({"bank": bank, "seed": seed, "primary": primary_metrics, "secondary": secondary_metrics})
    passed = all(_electronic_pass(row["primary"]) and _electronic_pass(row["secondary"]) for row in bank_metrics)
    gates = [GateResult("two_validation_banks", len(bank_metrics) == 2, {"banks": float(len(bank_metrics))}), GateResult("electronic_semantic_safety", passed, {})]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, {"schema_version": 1, "kind": ELECTRONIC_ADMISSION_KIND, "banks": bank_metrics, "passed": passed})
    _write_record(config, output.with_suffix(".evidence.json"), kind=ELECTRONIC_ADMISSION_KIND, status="passed" if passed else "valid_failed", gates=gates, input_hashes={"checkpoint": file_sha256(args.electronic_checkpoint), "primary_evaluator": primary_hash, "secondary_evaluator": secondary_hash, "dataset_split": split.manifest_hash}, evaluator_hash=primary_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_checkpoint), permitted=("spatial_quadrature_preflight",) if passed else (), reason=None if passed else "hierarchical spatial electronic capacity failed")
    return 0


def spatial_quadrature_preflight(args, config: V2Config) -> int:
    _require_spatial(config)
    admission = _load_record(args.electronic_admission_evidence, kind=ELECTRONIC_ADMISSION_KIND)
    images, train_labels, _, _, split, _, _ = _split(args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    _load_generator_checkpoint(args.electronic_checkpoint, bundle, config, split_hash=split.manifest_hash, evaluator_hash=evaluator_hash)
    labels = _preflight_batch(torch.empty(50, dtype=torch.long), seed=config.spatial_quadrature_generation.training_seed, device=device)
    codes = spatial_codes(labels.numel(), device=device, seed=config.spatial_quadrature_generation.training_seed)
    groups = spatial_parameter_groups(bundle)
    freeze_spatial_generator(bundle)
    phase_parameters = tuple(parameter for name in ("encoder_phase", "decoder_phase") for parameter in groups[name])
    optimizer = torch.optim.AdamW(phase_parameters, lr=config.spatial_quadrature_generation.phase_learning_rate)
    before = {id(parameter): parameter.detach().clone() for parameter in phase_parameters}
    views, commands, packed = spatial_forward_batched(bundle, codes, labels, microbatch_size=1)
    finite_before = bool(torch.isfinite(views).all() and torch.isfinite(commands).all() and packed is not None and torch.isfinite(packed).all())
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        view, _, _ = spatial_forward_batched(bundle, codes[:10], labels[:10], microbatch_size=1, return_packed=False)
        loss = view.mean()
        loss.backward()
        optimizer.step()
    changed = any(not torch.equal(before[identity], parameter.detach()) for identity, parameter in ((id(parameter), parameter) for group in optimizer.param_groups for parameter in group["params"]))
    readout = bundle.receiver(bundle.system(bundle.system.command(bundle.generator(codes[:1], labels[:1])), return_optical_trace=True).optical_trace[-1])
    identity_ok = bool(all(torch.isfinite(value).all() for value in (readout.r_difference, readout.j_difference)))
    gates = [GateResult("electronic_admission_consumed", admission.status == "passed", {}), GateResult("finite_quadrature_forward", finite_before, {}), GateResult("phase_updated", changed, {}), GateResult("packing_shape", tuple(packed.shape[-2:]) == (400, 400), {}), GateResult("difference_finite", identity_ok, {})]
    passed = all(gate.passed for gate in gates)
    _write_record(config, args.output, kind=QUADRATURE_PREFLIGHT_KIND, status="passed" if passed else "invalid_harness", gates=gates, input_hashes={"electronic_admission": admission.evidence_hash, "electronic_checkpoint": file_sha256(args.electronic_checkpoint), "dataset_split": split.manifest_hash}, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_checkpoint), permitted=("spatial_quadrature_smoke",) if passed else (), reason=None if passed else "spatial quadrature preflight failed")
    return 0


def spatial_quadrature_smoke(args, config: V2Config) -> int:
    _require_spatial(config)
    preflight = _load_record(args.quadrature_preflight_evidence, kind=QUADRATURE_PREFLIGHT_KIND)
    images, train_labels, validation_images, validation_labels, split, _, _ = _split(args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    _load_generator_checkpoint(args.electronic_checkpoint, bundle, config, split_hash=split.manifest_hash, evaluator_hash=evaluator_hash)
    freeze_spatial_generator(bundle)
    optimizer = torch.optim.AdamW(tuple(parameter for parameter in bundle.system.parameters()), lr=config.spatial_quadrature_generation.phase_learning_rate)
    history = []
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    schedule = list(balanced_batches(train_labels, batch_size=50, seed=config.spatial_quadrature_generation.training_seed))
    for step in range(args.steps):
        indices = schedule[step % len(schedule)]
        labels = train_labels.index_select(0, indices).to(device)
        codes = spatial_codes(labels.numel(), device=device, seed=100000 + step)
        if step == 400:
            unfreeze_spatial_generator(bundle)
            optimizer = torch.optim.AdamW(bundle.parameters(), lr=config.spatial_quadrature_generation.generator_learning_rate)
        optimizer.zero_grad(set_to_none=True)
        view, commands, _ = spatial_forward_batched(bundle, codes, labels, microbatch_size=1, return_packed=False)
        features, logits = evaluator(view)
        loss = F.cross_entropy(logits, labels) + 0.1 * commands.gt(0.99).float().mean()
        if not torch.isfinite(loss):
            raise ValueError("spatial quadrature smoke produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
    bank_images, bank_labels = _bank(validation_images, validation_labels, seed=config.spatial_quadrature_generation.validation_bank_seeds[0], half=50)
    bank_images = bank_images.to(device)
    bank_labels = bank_labels.to(device)
    codes = spatial_codes(bank_labels.numel(), device=device, seed=9991)
    view, commands, _ = spatial_forward_batched(bundle, codes, bank_labels, microbatch_size=1, return_packed=False)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    metrics = _metric_summary(evaluator, view, bank_labels, bank_images, commands=commands)
    passed = bool(metrics["finite"] and metrics["accuracy"] >= 0.80 and metrics["minimum_recall"] >= 0.60 and metrics["code_fraction"] >= 0.20 and metrics["saturation"] <= 0.08 and metrics["label_shift_accuracy"] <= 0.20)
    gates = [GateResult("preflight_consumed", preflight.status == "passed", {}), GateResult("smoke_semantic_transport", passed, {"accuracy": metrics["accuracy"], "minimum_recall": metrics["minimum_recall"]}), GateResult("smoke_finite", metrics["finite"], {})]
    _write_record(config, args.output, kind=QUADRATURE_SMOKE_KIND, status="passed" if passed else "valid_failed", gates=gates, input_hashes={"quadrature_preflight": preflight.evidence_hash, "electronic_checkpoint": file_sha256(args.electronic_checkpoint), "dataset_split": split.manifest_hash}, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_checkpoint), permitted=("train",) if passed else (), reason=None if passed else "quadrature semantic transport smoke failed")
    return 0


def spatial_train(args, config: V2Config) -> int:
    _require_spatial(config)
    if not args.spatial_electronic_admission_evidence or not args.spatial_quadrature_smoke_evidence or not args.spatial_electronic_checkpoint:
        raise ValueError("v3 training requires electronic admission, smoke evidence, and electronic checkpoint")
    admission = _load_record(args.spatial_electronic_admission_evidence, kind=ELECTRONIC_ADMISSION_KIND)
    smoke = _load_record(args.spatial_quadrature_smoke_evidence, kind=QUADRATURE_SMOKE_KIND)
    images, train_labels, _, _, split, _, _ = _split(args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    _load_generator_checkpoint(args.spatial_electronic_checkpoint, bundle, config, split_hash=split.manifest_hash, evaluator_hash=evaluator_hash)
    freeze_spatial_generator(bundle)
    optimizer = torch.optim.AdamW(tuple(parameter for parameter in bundle.system.parameters()), lr=config.spatial_quadrature_generation.phase_learning_rate)
    history = []
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    for epoch in range(config.spatial_quadrature_generation.joint_epochs):
        for batch_number, indices in enumerate(balanced_batches(train_labels, batch_size=50, seed=config.spatial_quadrature_generation.training_seed + epoch)):
            labels = train_labels.index_select(0, indices).to(device)
            real_views = images.index_select(0, indices).to(device)
            codes = spatial_codes(labels.numel(), device=device, seed=100000 * (epoch + 1) + batch_number)
            if len(history) == 400:
                unfreeze_spatial_generator(bundle)
            optimizer.zero_grad(set_to_none=True)
            view, commands, _ = spatial_forward_batched(bundle, codes, labels, microbatch_size=1, return_packed=False)
            features, logits = evaluator(view)
            with torch.no_grad():
                real_features, _ = evaluator(real_views)
            loss = conditional_sinkhorn_loss(features, view, real_features, real_views, labels, epsilon=0.05, iterations=20) + 0.5 * F.cross_entropy(logits, labels) + 0.25 * relative_diversity_worst_class_loss(features, real_features, labels) + 0.1 * diversity_hinge(features, codes, labels) + torch.relu(0.25 - camera_code_fraction(view, labels)).square() + 0.1 * commands.gt(0.99).float().mean() + 1.0e-5 * bundle.generator.smoothness_penalty()
            if not torch.isfinite(loss):
                raise ValueError("v3 spatial training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            history.append({"epoch": float(epoch + 1), "step": float(len(history) + 1), "loss": float(loss.detach().item())})
    state = bundle.state_dict()
    payload = {"manifest": {"schema_version": 3, "architecture_kind": SPATIAL_ARCHITECTURE, "generator_contract": SPATIAL_GENERATOR_CONTRACT, "receiver_contract": SPATIAL_RECEIVER_CONTRACT, "config_hash": _config_hash(config), "system_signature": _system_hash(config), "dataset_split_manifest_hash": split.manifest_hash, "evaluator_checkpoint_hash": evaluator_hash, "state_signature": state_signature(state), "state_content_hash": state_content_hash(state)}, "state_dict": state, "extra": {"history": history, "calibration": bundle.calibration.to_dict(), "electronic_admission": admission.evidence_hash, "quadrature_smoke": smoke.evidence_hash}}
    atomic_torch_save(args.output, payload)
    checkpoint_hash = file_sha256(args.output)
    _write_record(config, Path(args.output).with_suffix(".evidence.json"), kind=SPATIAL_TRAINING_KIND, status="passed", gates=[GateResult("finite_training", bool(history), {"steps": float(len(history))})], input_hashes={"checkpoint": checkpoint_hash, "electronic_admission": admission.evidence_hash, "quadrature_smoke": smoke.evidence_hash, "dataset_split": split.manifest_hash, "evaluator": evaluator_hash}, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash, permitted=("evaluate",))
    return 0


def _load_full_checkpoint(path: str | Path, bundle: SpatialQuadratureBundle, config: V2Config, *, split_hash: str | None = None, evaluator_hash: str | None = None) -> dict[str, Any]:
    payload = safe_torch_load(path)
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 3 or manifest.get("architecture_kind") != SPATIAL_ARCHITECTURE:
        raise ValueError("invalid v3 full checkpoint")
    if manifest.get("config_hash") != _config_hash(config):
        raise ValueError("v3 full checkpoint lineage mismatch")
    if split_hash is not None and manifest.get("dataset_split_manifest_hash") != split_hash:
        raise ValueError("v3 full checkpoint split mismatch")
    if evaluator_hash is not None and manifest.get("evaluator_checkpoint_hash") != evaluator_hash:
        raise ValueError("v3 full checkpoint evaluator mismatch")
    if not isinstance(state, dict) or state_signature(state) != manifest.get("state_signature") or state_content_hash(state) != manifest.get("state_content_hash"):
        raise ValueError("v3 full checkpoint state hash mismatch")
    bundle.load_state_dict(state, strict=True)
    return payload


def spatial_sample(args, config: V2Config) -> int:
    _require_spatial(config)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    _load_full_checkpoint(args.checkpoint, bundle, config)
    codes = safe_torch_load(args.codes).to(device)
    labels = safe_torch_load(args.labels).to(device)
    _, _, packed = spatial_forward_batched(bundle, codes, labels, microbatch_size=1)
    atomic_torch_save(args.output, packed.detach().cpu())
    if args.view_output:
        view, _, _ = spatial_forward_batched(bundle, codes, labels, microbatch_size=1, return_packed=False)
        atomic_torch_save(args.view_output, view.detach().cpu())
    return 0


def spatial_evaluate(args, config: V2Config) -> int:
    _require_spatial(config)
    if not args.training_evidence or not args.acceptance_evaluator_checkpoint:
        raise ValueError("v3 evaluation requires training evidence and the secondary evaluator")
    training = _load_record(args.training_evidence, kind=SPATIAL_TRAINING_KIND)
    images, train_labels, _, _, split, _, _ = _split(args.dataset_root)
    primary_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    secondary_hash = evaluator_checkpoint_hash(args.acceptance_evaluator_checkpoint)
    device = _device(config)
    bundle = seeded_spatial_bundle(config).to(device)
    checkpoint_hash = training.input_hashes.get("checkpoint")
    if not isinstance(checkpoint_hash, str):
        raise ValueError("v3 training evidence lacks checkpoint binding")
    _load_full_checkpoint(bundle_path := args.checkpoint, bundle, config, split_hash=split.manifest_hash, evaluator_hash=primary_hash)
    if file_sha256(bundle_path) != checkpoint_hash:
        raise ValueError("v3 checkpoint does not match training evidence")
    primary, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    secondary = _load_secondary_evaluator(args.acceptance_evaluator_checkpoint, device=device, split_hash=split.manifest_hash)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    raw_rows, view_rows, label_rows = [], [], []
    settings = config.spatial_quadrature_generation
    assert settings is not None
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, test_labels.numel(), settings.logical_batch_size)):
            labels = test_labels[start : start + settings.logical_batch_size].to(device)
            codes = spatial_codes(labels.numel(), device=device, seed=4321 + batch_number)
            view, _, packed = spatial_forward_batched(bundle, codes, labels, microbatch_size=1)
            if packed is None:
                raise RuntimeError("v3 evaluation did not produce four-port CMOS data")
            view_rows.append(view.cpu())
            raw_rows.append(packed.cpu())
            label_rows.append(labels.cpu())
    views = torch.cat(view_rows)
    raw_cmos = torch.cat(raw_rows)
    labels = torch.cat(label_rows)
    real_test = test_images.to(device)
    primary_metrics = _metric_summary(
        primary,
        views.to(device),
        labels.to(device),
        real_test,
        commands=views.to(device),
        train_views=images.to(device),
        train_labels=train_labels.to(device),
    )
    secondary_metrics = _metric_summary(
        secondary,
        views.to(device),
        labels.to(device),
        real_test,
        commands=views.to(device),
    )
    primary_features, primary_logits = primary(views.to(device))
    secondary_features, secondary_logits = secondary(views.to(device))
    primary_predictions = primary_logits.argmax(dim=1)
    secondary_predictions = secondary_logits.argmax(dim=1)
    zero_labels = torch.arange(10, device=device)
    zero_codes = torch.zeros(10, 32, device=device)
    zero_views, _, _ = spatial_forward_batched(bundle, zero_codes, zero_labels, microbatch_size=1, return_packed=False)
    zero_primary = primary(zero_views)[1].argmax(dim=1)
    zero_secondary = secondary(zero_views)[1].argmax(dim=1)
    unique_count = int(torch.unique(views.reshape(views.shape[0], -1), dim=0).shape[0])
    agreement = float((primary_predictions == secondary_predictions).float().mean().item())
    passed = bool(
        primary_metrics["accuracy"] >= 0.95
        and secondary_metrics["accuracy"] >= 0.95
        and primary_metrics["minimum_recall"] >= 0.90
        and secondary_metrics["minimum_recall"] >= 0.90
        and agreement >= 0.95
        and primary_metrics["code_fraction"] >= 0.25
        and primary_metrics["saturation"] <= 0.08
        and all(0.70 <= value <= 1.30 for value in primary_metrics["diversity"])
        and all(0.70 <= value <= 1.30 for value in secondary_metrics["diversity"])
        and primary_metrics["coverage"] <= 1.50
        and secondary_metrics["coverage"] <= 1.50
        and primary_metrics["label_shift_accuracy"] <= 0.20
        and bool(torch.equal(zero_primary, zero_labels))
        and bool(torch.equal(zero_secondary, zero_labels))
        and unique_count == views.shape[0]
    )
    gates = [
        GateResult("primary_semantics", primary_metrics["accuracy"] >= 0.95 and primary_metrics["minimum_recall"] >= 0.90, {"accuracy": primary_metrics["accuracy"], "minimum_recall": primary_metrics["minimum_recall"]}),
        GateResult("secondary_semantics", secondary_metrics["accuracy"] >= 0.95 and secondary_metrics["minimum_recall"] >= 0.90, {"accuracy": secondary_metrics["accuracy"], "minimum_recall": secondary_metrics["minimum_recall"]}),
        GateResult("evaluator_agreement", agreement >= 0.95, {"agreement": agreement}),
        GateResult("safety_and_diversity", passed, {"code_fraction": primary_metrics["code_fraction"], "coverage": primary_metrics["coverage"], "unique": float(unique_count)}),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw_cmos.pt")
    labels_path = output.with_suffix(".labels.pt")
    atomic_torch_save(raw_path, raw_cmos)
    atomic_torch_save(labels_path, labels)
    atomic_write_json(output.with_suffix(".report.json"), {"primary": primary_metrics, "secondary": secondary_metrics, "agreement": agreement, "zero_code": {"primary": zero_primary.tolist(), "secondary": zero_secondary.tolist()}, "unique_count": unique_count, "raw_cmos_sha256": tensor_sha256(raw_cmos), "view_sha256": tensor_sha256(views)})
    _write_record(config, output, kind=SPATIAL_ACCEPTANCE_KIND, status="pending_review" if passed else "valid_failed", gates=gates, input_hashes={"training_evidence": training.evidence_hash, "checkpoint": checkpoint_hash, "primary_evaluator": primary_hash, "secondary_evaluator": secondary_hash, "dataset_split": split.manifest_hash, "raw_cmos": tensor_sha256(raw_cmos)}, evaluator_hash=primary_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash, permitted=("admit_conditional_evaluation",) if passed else (), reason=None if passed else "v3 formal raw-CMOS acceptance gate failed")
    return 0
