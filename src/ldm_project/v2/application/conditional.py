"""CLI stages for the teacher-free conditional optical generator."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import V2Config
from ..evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, load_conditional_manual_review, tensor_sha256
from ..optics import FormalAmplitudeFiniteCropIdentitySystem
from ..student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint, save_conditional_checkpoint, state_content_hash, state_signature
from ..student.admission import balanced_code_fraction
from ..student.conditional import conditional_loss, fixed_exposure_view, raw_from_generator_batched
from ..student.conditional_acceptance import conditional_coverage_ratio, load_candidate_dataset, memorization_statistics, paired_bootstrap_lower_bound, per_label_feature_diversity_ratio
from ..student.evaluator import FrozenMNISTEvaluator, MNISTSplitManifest, balanced_batches, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist, save_evaluator_checkpoint, stratified_mnist_indices
from ..student.generator import ConditionalGlobalAmplitudeGenerator
from ..student.review_artifacts import write_conditional_review_artifacts
from .runs import V2Run


CONDITIONAL_CONTROL_METRIC_CONTRACT = "sample_centered_anova_v1|sinkhorn_feature_l2_v1|coverage_disjoint_real_ab_v1"


class ConditionalBundle(nn.Module):
    def __init__(self, config: V2Config, *, carrier_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        if config.architecture != "conditional_global_amplitude_v1" or config.conditional_generation is None:
            raise ValueError("conditional bundle requires conditional_global_amplitude_v1 config")
        settings = config.conditional_generation
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        self.generator = ConditionalGlobalAmplitudeGenerator(
            shape=config.system.slm_shape,
            code_dim=settings.code_dim,
            hidden_dim=settings.hidden_dim,
            command_gain=settings.command_gain,
            class_carrier_logits=carrier_logits,
        )


def _require_conditional(config: V2Config) -> None:
    if config.architecture != "conditional_global_amplitude_v1" or config.conditional_generation is None:
        raise ValueError("this command requires architecture: conditional_global_amplitude_v1")


def _system_signature(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def _context(config: V2Config, *, evaluator_hash: str | None = None, split_hash: str | None = None, checkpoint_hash: str | None = None, candidate_manifest_hash: str | None = None) -> EvidenceContextV2:
    settings = config.conditional_generation
    assert settings is not None
    return EvidenceContextV2(
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        checkpoint_hash=checkpoint_hash,
        evaluator_checkpoint_hash=evaluator_hash,
        dataset_split_manifest_hash=split_hash,
        exposure_scale=settings.fixed_exposure_scale,
        candidate_dataset_manifest_hash=candidate_manifest_hash,
    )


def _write_evidence(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    record.validate()
    path = run.root / name
    atomic_write_json(path, record.to_dict())
    return path


def _new_run(config: V2Config) -> V2Run:
    if config.run.evidence_tier == "acceptance":
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], check=True, capture_output=True, text=True).stdout.strip()
        if dirty:
            raise ValueError("acceptance operations require a clean git commit")
    return V2Run.create(output_root=config.run.output_root, run_name=config.run.run_name, config_payload=config.model_dump(mode="json"))


def _split(images: torch.Tensor, labels: torch.Tensor, dataset_root: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MNISTSplitManifest]:
    train_indices, validation_indices = stratified_mnist_indices(labels)
    manifest = MNISTSplitManifest(
        1, str(Path(dataset_root).resolve()), 1234,
        canonical_sha256(train_indices), canonical_sha256(validation_indices), train_indices.numel(), validation_indices.numel(),
    )
    return images.index_select(0, train_indices), labels.index_select(0, train_indices), images.index_select(0, validation_indices), labels.index_select(0, validation_indices), manifest


def _carrier_logits(images: torch.Tensor, labels: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    means = torch.stack([images[labels == label].mean(dim=0) for label in range(10)])
    enlarged = torch.nn.functional.interpolate(means, size=shape, mode="bilinear", align_corners=False)
    amplitude = enlarged.sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(amplitude).squeeze(1)


def _random_codes(count: int, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(count, 8, generator=generator).mul_(2.0).sub_(1.0).to(device)


def _load_bundle(config: V2Config, checkpoint: str, *, device: torch.device, evaluator_hash: str | None = None, split_hash: str | None = None) -> tuple[ConditionalBundle, ConditionalCheckpointManifestV2]:
    _require_conditional(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid conditional checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if manifest.architecture_kind != "conditional_global_amplitude_v1" or manifest.system_signature != _system_signature(config) or manifest.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("conditional checkpoint belongs to a different architecture/system/config")
    if evaluator_hash is not None and manifest.evaluator_checkpoint_hash != evaluator_hash:
        raise ValueError("conditional checkpoint uses another evaluator")
    if split_hash is not None and manifest.dataset_split_manifest_hash != split_hash:
        raise ValueError("conditional checkpoint uses another MNIST split")
    bundle = ConditionalBundle(config).to(device)
    load_conditional_checkpoint(checkpoint, bundle, manifest)
    return bundle, manifest


def train_evaluator(args, config: V2Config) -> int:
    _require_conditional(config)
    device = torch.device(config.run.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    model = FrozenMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train_images.shape[0], generator=torch.Generator(device="cpu").manual_seed(1234 + epoch))
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : min(start + args.batch_size, order.numel())]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(train_images.index_select(0, indices).to(device))
            loss = torch.nn.functional.cross_entropy(logits, train_labels.index_select(0, indices).to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, logits = model(validation_images.to(device))
        accuracy = float((logits.argmax(dim=1) == validation_labels.to(device)).to(torch.float32).mean().item())
    run = _new_run(config)
    checkpoint_path = run.root / "frozen_mnist_evaluator.pt"
    save_evaluator_checkpoint(checkpoint_path, model.freeze(), split, validation_accuracy=accuracy)
    record = EvidenceRecordV2(2, "evaluator_qualification", config.run.evidence_tier, "passed" if accuracy >= 0.985 else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=file_sha256(checkpoint_path), split_hash=split.manifest_hash), {"evaluator_checkpoint": file_sha256(checkpoint_path)}, (GateResult("validation_accuracy", accuracy >= 0.985, {"validation_accuracy": accuracy}),), ("conditional_observability",) if accuracy >= 0.985 else (), None if accuracy >= 0.985 else "frozen evaluator did not reach 98.5% validation accuracy")
    _write_evidence(run, "evaluator_qualification_evidence.json", record)
    atomic_write_json(run.root / "mnist_split_manifest.json", asdict(split) | {"manifest_hash": split.manifest_hash})
    print(run.root)
    return 0


def qualify_evaluator(args, config: V2Config) -> int:
    """Re-check an evaluator checkpoint without retraining it."""
    _require_conditional(config)
    device = torch.device(config.run.device)
    _, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    split_payload = payload.get("split_manifest")
    if not isinstance(split_payload, dict):
        raise ValueError("evaluator checkpoint lacks its MNIST split manifest")
    split = MNISTSplitManifest(**split_payload)
    accuracy = float(payload["validation_accuracy"])
    run = _new_run(config)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    record = EvidenceRecordV2(2, "evaluator_qualification", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash), {"evaluator_checkpoint": evaluator_hash}, (GateResult("validation_accuracy", True, {"validation_accuracy": accuracy}),), ("conditional_observability",) if config.run.evidence_tier == "acceptance" else (), None)
    _write_evidence(run, "evaluator_qualification_evidence.json", record)
    print(run.root)
    return 0


def conditional_observability(args, config: V2Config) -> int:
    _require_conditional(config)
    device = torch.device(config.run.device)
    bundle = ConditionalBundle(config).to(device)
    labels = torch.arange(10, device=device).repeat_interleave(2)
    codes = _random_codes(labels.numel(), device=device, seed=1234)
    before = state_content_hash(bundle.state_dict())
    result = bundle.system.forward_from_generator(bundle.generator, codes, labels, return_optical_trace=True)
    raw = result.raw_camera.tensor
    command_delta = (bundle.generator(codes, labels)[0::2] - bundle.generator(codes, labels)[1::2]).abs().sum(dim=(-3, -2, -1))
    camera_delta = (raw[0::2] - raw[1::2]).abs().sum(dim=(-3, -2, -1))
    objective = raw.mean()
    objective.backward()
    allowed = [bundle.generator.class_carrier_logits, bundle.generator.global_modes, bundle.generator.code_projection.weight, bundle.generator.class_heads, bundle.system.encoder.phase, bundle.system.decoder.phase]
    gradients = all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in allowed)
    trace_operations = tuple(field.provenance.operation for field in result.optical_trace or ())
    after = state_content_hash(bundle.state_dict())
    checks = (
        GateResult("finite_raw_cmos", bool(torch.isfinite(raw).all()), {"finite_raw_cmos": bool(torch.isfinite(raw).all())}),
        GateResult("global_slm_code_response", bool((command_delta > 0).all()), {"minimum_command_delta": float(command_delta.min().item())}),
        GateResult("cmos_code_response", bool((camera_delta > 1.0e-8).all()), {"minimum_camera_delta": float(camera_delta.min().item())}),
        GateResult("complete_encoder_ols_decoder_trace", any("identity_ols" in item for item in trace_operations), {"trace_length": float(len(trace_operations))}),
        GateResult("finite_gradients", gradients, {"finite_gradients": gradients}),
        GateResult("parameters_unchanged", before == after, {"parameters_unchanged": before == after}),
    )
    run = _new_run(config)
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(2, "conditional_observability", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config), {}, checks, ("controllability_smoke",) if passed else (), None if passed else "conditional optical observability failed")
    _write_evidence(run, "conditional_observability_evidence.json", record)
    print(run.root)
    return 0


def sample_conditional(args, config: V2Config) -> int:
    _require_conditional(config)
    device = torch.device(config.run.device)
    bundle, _ = _load_bundle(config, args.checkpoint, device=device)
    codes = torch.load(args.codes, map_location=device, weights_only=False)
    labels = torch.load(args.labels, map_location=device, weights_only=False).long()
    if not torch.is_tensor(codes) or not torch.is_tensor(labels):
        raise ValueError("codes and labels must be tensor files")
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False).detach().cpu()
    atomic_torch_save(args.output, raw)
    return 0


def _passed_evidence(path: str, *, kind: str) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed")
    return record


def _code_fraction(values: torch.Tensor, labels: torch.Tensor) -> float:
    means = torch.stack([values[labels == label].mean(dim=0) for label in range(10)])
    residual = values - means.index_select(0, labels)
    grand_mean = values.mean(dim=0, keepdim=True)
    return float((residual.square().sum() / (values - grand_mean).square().sum().clamp_min(1.0e-12)).item())


def _conditional_parameter_groups(bundle: ConditionalBundle) -> dict[str, tuple[nn.Parameter, ...]]:
    return {
        "class_carrier": (bundle.generator.class_carrier_logits,),
        "global_modes": (bundle.generator.global_modes,),
        "code_projection": (bundle.generator.code_projection.weight,),
        "class_heads": (bundle.generator.class_heads,),
        "encoder_phase": (bundle.system.encoder.phase,),
        "decoder_phase": (bundle.system.decoder.phase,),
    }


def _fixed_control_probe(*, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.eye(8, device=device)
    codes = torch.cat((torch.zeros(1, 8, device=device), axis, -axis), dim=0)
    return codes.repeat(10, 1), torch.arange(10, device=device).repeat_interleave(17)


def _probe_metrics(raw: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, exposure_scale: float) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure_scale)
    features, logits = evaluator(view)
    values = raw.reshape(10, 17, *raw.shape[1:])
    command_fraction = balanced_code_fraction(values)
    code_mean = values.mean(dim=1, keepdim=True)
    direction_rms = (values - code_mean).square().mean(dim=tuple(range(2, values.ndim))).sqrt().mean(dim=0).flatten().tolist()
    return {
        "accuracy": float((logits.argmax(dim=1) == labels).to(torch.float32).mean().item()),
        "confusion": torch.bincount(labels * 10 + logits.argmax(dim=1), minlength=100).reshape(10, 10).cpu().tolist(),
        "code_fraction": command_fraction,
        "label_fraction": 1.0 - command_fraction,
        "direction_rms": [float(value) for value in direction_rms],
        "raw_rms": float(raw.square().mean().sqrt().item()),
        "view_rms": float(view.square().mean().sqrt().item()),
        "saturation": float(torch.relu(raw / exposure_scale - 1.0).gt(0).to(torch.float32).mean().item()),
        "feature_code_fraction": balanced_code_fraction(features.reshape(10, 17, -1)),
    }


def conditional_control_isolation(args, config: V2Config) -> int:
    """Read-only fixed-code panel locating the first control-signal collapse."""
    _require_conditional(config)
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(images, all_labels, config.system.slm_shape)).to(device).eval()
    codes, labels = _fixed_control_probe(device=device)
    settings = config.conditional_generation
    assert settings is not None
    before = state_content_hash(bundle.state_dict())
    with torch.no_grad():
        command = bundle.generator(codes, labels)
        result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
        raw = result.raw_camera.tensor
        view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
        features, logits = evaluator(view)
    after = state_content_hash(bundle.state_dict())
    trace = result.optical_trace or ()
    trace_names = ("slm_field", "pre_encoder", "encoder_layer_1", "encoder_layer_2", "identity_ols", "decoder_layer_1", "decoder_layer_2", "camera_field")
    plane_metrics: list[dict[str, object]] = []
    for name, field in zip(trace_names, trace, strict=False):
        tensor = field.tensor.abs().square()
        grouped = tensor.reshape(10, 17, *tensor.shape[1:])
        fraction = balanced_code_fraction(grouped)
        plane_metrics.append({
            "name": name,
            "code_fraction": fraction,
            "label_fraction": 1.0 - fraction,
            "rms": float(tensor.square().mean().sqrt().item()),
            "mean": float(tensor.mean().item()),
            "max": float(tensor.max().item()),
            "finite": bool(torch.isfinite(tensor).all()),
            "plane": field.plane.to_dict(),
            "provenance": field.provenance.to_dict(),
        })
    command_grouped = command.reshape(10, 17, *command.shape[1:])
    command_fraction = balanced_code_fraction(command_grouped)
    coefficients = bundle.generator.coefficients(codes, labels).reshape(10, 17, 8)
    duplicate = codes[0].repeat(10, 1)
    duplicate_command = bundle.generator(duplicate, labels[:10])
    duplicate_error = float((duplicate_command - command[:10]).abs().max().item())
    report = {
        "metric_contract": CONDITIONAL_CONTROL_METRIC_CONTRACT,
        "code_count_per_label": 17,
        "code_panel": "z0,+e1,-e1,...,+e8,-e8",
        "parameters_unchanged": before == after,
        "trace_operations": [field.provenance.operation for field in trace],
        "trace_plane_ids": [field.plane.plane_id for field in trace],
        "plane_metrics": plane_metrics,
        "command": {
            "code_fraction": command_fraction,
            "label_fraction": 1.0 - command_fraction,
            "coefficient_rms": float(coefficients.square().mean().sqrt().item()),
            "coefficient_std_by_direction": coefficients.reshape(-1, 8).std(dim=0).tolist(),
            "duplicate_zero_code_max_error": duplicate_error,
        },
        "camera": _probe_metrics(raw, labels, evaluator, settings.fixed_exposure_scale),
        "classification": {"accuracy": float((logits.argmax(dim=1) == labels).to(torch.float32).mean().item())},
        "harness": {
            "finite_command": bool(torch.isfinite(command).all()),
            "finite_raw": bool(torch.isfinite(raw).all()),
            "finite_features": bool(torch.isfinite(features).all()),
            "complete_trace": len(trace) == len(trace_names) and any("identity_ols" in item for item in [field.provenance.operation for field in trace]),
        },
    }
    command_path = Path(args.output) if args.output else None
    run = _new_run(config)
    if command_path is None:
        command_path = run.root / "conditional_control_isolation_raw_cmos.pt"
    atomic_torch_save(command_path, raw.cpu())
    atomic_torch_save(run.root / "conditional_control_isolation_codes.pt", codes.cpu())
    atomic_torch_save(run.root / "conditional_control_isolation_labels.pt", labels.cpu())
    report_path = run.root / "conditional_control_isolation_report.json"
    atomic_write_json(report_path, report)
    command_passed = command_fraction >= 0.10
    first_collapse = None
    if not command_passed:
        classification = "generator_injection_insufficient"
    else:
        for item in plane_metrics:
            if item["code_fraction"] < 0.10:
                first_collapse = item["name"]
                break
        if first_collapse is not None:
            classification = "optical_transport_attenuated"
        elif not report["harness"]["complete_trace"] or not report["parameters_unchanged"] or not report["harness"]["finite_raw"]:
            classification = "harness_invalid"
        elif report["camera"]["feature_code_fraction"] < 0.10 or report["classification"]["accuracy"] < 0.30:
            classification = "semantic_readout_insufficient"
        else:
            classification = "controllable_upper_bound"
    report["classification"] = {**report["classification"], "status": classification, "first_collapse_plane": first_collapse}
    atomic_write_json(report_path, report)
    checks = (
        GateResult("fixed_probe_shape", len(codes) == 170 and len(labels) == 170, {"samples": float(len(labels))}),
        GateResult("finite_fields", bool(report["harness"]["finite_command"] and report["harness"]["finite_raw"] and report["harness"]["finite_features"]), {}),
        GateResult("complete_provenance", bool(report["harness"]["complete_trace"]), {"trace_length": float(len(trace))}),
        GateResult("parameters_unchanged", bool(report["parameters_unchanged"]), {}),
        GateResult("command_code_response", command_fraction > 0.0, {"code_fraction": command_fraction}),
    )
    record = EvidenceRecordV2(2, "conditional_control_isolation", "diagnostic", "diagnostic" if all(item.passed for item in checks) else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash), {"report": file_sha256(report_path), "raw_cmos": file_sha256(command_path), "metric_contract": canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)}, checks, (), None if all(item.passed for item in checks) else "conditional control isolation harness failed")
    _write_evidence(run, "conditional_control_isolation_evidence.json", record)
    print(run.root)
    return 0


def _conditional_optimizer(bundle: ConditionalBundle, config: V2Config) -> torch.optim.Optimizer:
    settings = config.conditional_generation
    assert settings is not None
    return torch.optim.Adam([
        {"params": [bundle.generator.class_carrier_logits, bundle.generator.global_modes, bundle.generator.code_projection.weight, bundle.generator.class_heads], "lr": settings.generator_learning_rate},
        {"params": [bundle.system.encoder.phase, bundle.system.decoder.phase], "lr": settings.phase_learning_rate},
    ])


def controllability_smoke(args, config: V2Config) -> int:
    _require_conditional(config)
    _passed_evidence(args.observability_evidence, kind="conditional_observability")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(train_images, train_labels, config.system.slm_shape)).to(device)
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    probe_history: dict[str, dict[str, object]] = {}
    loss_history: list[dict[str, float]] = []
    group_history: list[dict[str, object]] = []

    def evaluate_probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            probe_raw = raw_from_generator_batched(bundle.system, bundle.generator, probe_codes, probe_labels, microbatch_size=config.execution.training_microbatch_size, activation_checkpoint=False)
            return probe_raw, _probe_metrics(probe_raw, probe_labels, evaluator, settings.fixed_exposure_scale)

    initial_raw, initial_metrics = evaluate_probe()
    probe_history["step_0"] = initial_metrics
    for step, indices in zip(range(args.steps), balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234)):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=1234 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, _, raw = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=config.execution.activation_checkpoint, microbatch_size=config.execution.training_microbatch_size)
        loss.backward()
        groups = _conditional_parameter_groups(bundle)
        grad_metrics = {}
        snapshots = {name: [parameter.detach().clone() for parameter in parameters] for name, parameters in groups.items()}
        for name, parameters in groups.items():
            norms = [parameter.grad.detach().norm() if parameter.grad is not None else torch.tensor(float("nan"), device=device) for parameter in parameters]
            grad_metrics[name] = {"gradient_norm": float(torch.stack(norms).norm().item()), "gradient_finite": bool(all(torch.isfinite(value) for value in norms)), "gradient_nonzero": bool(any(value > 0.0 for value in norms))}
        optimizer.step()
        update_metrics = {}
        for name, parameters in groups.items():
            deltas = [parameter.detach() - before for parameter, before in zip(parameters, snapshots[name], strict=True)]
            update_metrics[name] = {"update_norm": float(torch.stack([delta.norm() for delta in deltas]).norm().item()), "update_nonzero": bool(any(delta.abs().sum() > 0.0 for delta in deltas))}
        group_history.append({"step": step + 1, "gradients": grad_metrics, "updates": update_metrics})
        loss_history.append({"step": float(step + 1), "total": float(loss.detach().item())})
        if step + 1 in {10, args.steps}:
            _, metrics = evaluate_probe()
            probe_history[f"step_{step + 1}"] = metrics
    if args.steps < 1:
        raise ValueError("controllability smoke received no balanced MNIST batches")
    final_raw, final_metrics = evaluate_probe()
    probe_history[f"step_{args.steps}"] = final_metrics
    final_groups = group_history[-1]
    gradient_checks = [value["gradient_finite"] and value["gradient_nonzero"] for value in final_groups["gradients"].values()]
    update_checks = [value["update_nonzero"] for value in final_groups["updates"].values()]
    checks = (
        GateResult("fixed_probe_complete", final_metrics["accuracy"] is not None, {"samples": 170.0}),
        GateResult("semantic_accuracy", float(final_metrics["accuracy"]) >= 0.30, {"accuracy": float(final_metrics["accuracy"])}),
        GateResult("camera_code_fraction", float(final_metrics["code_fraction"]) >= 0.10, {"camera_code_fraction": float(final_metrics["code_fraction"])}),
        GateResult("finite_nonzero_gradients", bool(all(gradient_checks)), {"groups": float(sum(gradient_checks)), "total": float(len(gradient_checks))}),
        GateResult("nonzero_parameter_updates", bool(all(update_checks)), {"groups": float(sum(update_checks)), "total": float(len(update_checks))}),
        GateResult("high_saturation", float(final_metrics["saturation"]) <= 0.08, {"high_saturation": float(final_metrics["saturation"])}),
    )
    run = _new_run(config)
    raw_path = run.root / "controllability_probe_raw_cmos.pt"
    labels_path = run.root / "controllability_probe_labels.pt"
    atomic_torch_save(raw_path, final_raw.cpu())
    atomic_torch_save(labels_path, probe_labels.cpu())
    review_artifacts = write_conditional_review_artifacts(root=run.root, raw_cmos=final_raw.cpu(), labels=probe_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
    atomic_write_json(run.root / "controllability_probe_history.json", {"metric_contract": CONDITIONAL_CONTROL_METRIC_CONTRACT, "loss": loss_history, "probes": probe_history, "parameter_groups": group_history})
    passed = all(item.passed for item in checks)
    context = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash)
    permitted = ("train",) if passed and config.run.evidence_tier == "acceptance" else ()
    record = EvidenceRecordV2(2, "controllability_smoke", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"observability": file_sha256(args.observability_evidence), "evaluator": evaluator_checkpoint_hash(args.evaluator_checkpoint), "metric_contract": canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), "probe_raw_cmos": file_sha256(raw_path), "probe_labels": file_sha256(labels_path), "probe_preview": review_artifacts.preview_sha256}, checks, permitted, None if passed else "conditional controllability smoke failed")
    _write_evidence(run, "controllability_smoke_evidence.json", record)
    print(run.root)
    return 0


def conditional_train(args, config: V2Config) -> int:
    _require_conditional(config)
    _passed_evidence(args.observability_evidence, kind="conditional_observability")
    _passed_evidence(args.controllability_evidence, kind="controllability_smoke")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(train_images, train_labels, config.system.slm_shape)).to(device)
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    history: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        for batch_number, indices in enumerate(balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234 + epoch)):
            labels = train_labels.index_select(0, indices).to(device)
            codes = _random_codes(labels.numel(), device=device, seed=(epoch + 1) * 100000 + batch_number)
            optimizer.zero_grad(set_to_none=True)
            loss, summary, _ = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=config.execution.activation_checkpoint, microbatch_size=config.execution.training_microbatch_size)
            loss.backward()
            optimizer.step()
            history.append(asdict(summary) | {"epoch": float(epoch + 1)})
    run = _new_run(config)
    checkpoint_path = run.root / "conditional_optical_generator.pt"
    manifest = ConditionalCheckpointManifestV2(2, "conditional_global_amplitude_v1", _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_checkpoint_hash(args.evaluator_checkpoint), settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()))
    save_conditional_checkpoint(checkpoint_path, bundle, manifest, extra={"history": history})
    checkpoint_hash = file_sha256(checkpoint_path)
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    permitted = ("evaluate",) if finite and config.run.evidence_tier == "acceptance" else ()
    record = EvidenceRecordV2(2, "conditional_training", config.run.evidence_tier, "passed" if finite else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=manifest.evaluator_checkpoint_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash), {"checkpoint": checkpoint_hash, "evaluator": manifest.evaluator_checkpoint_hash}, (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),), permitted, None if finite else "conditional training produced non-finite loss")
    _write_evidence(run, "conditional_training_evidence.json", record)
    print(run.root)
    return 0


def conditional_evaluate(args, config: V2Config) -> int:
    _require_conditional(config)
    _passed_evidence(args.training_evidence, kind="conditional_training")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    settings = config.conditional_generation
    assert settings is not None
    raw_rows, label_rows, prediction_rows, feature_rows, real_feature_rows, generated_view_rows, real_view_rows = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch_number, indices in enumerate(balanced_batches(validation_labels, batch_size=settings.logical_batch_size, seed=4321)):
            labels = validation_labels.index_select(0, indices).to(device)
            codes = _random_codes(labels.numel(), device=device, seed=4321 + batch_number)
            raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
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
    raw, labels, predictions = torch.cat(raw_rows), torch.cat(label_rows), torch.cat(prediction_rows)
    generated_features, real_features = torch.cat(feature_rows), torch.cat(real_feature_rows)
    generated_view, real_view = torch.cat(generated_view_rows), torch.cat(real_view_rows)
    accuracy = float((predictions == labels).to(torch.float32).mean().item())
    recalls = [float(((predictions[labels == label] == label).to(torch.float32).mean()).item()) for label in range(10)]
    code_fraction = _code_fraction(raw, labels)
    saturation = float(torch.relu(raw / settings.fixed_exposure_scale - 1.0).gt(0).to(torch.float32).mean().item())
    diversity_ratios = per_label_feature_diversity_ratio(generated_features, real_features, labels)
    coverage_ratio = conditional_coverage_ratio(generated_features, generated_view, real_features, real_view, labels, epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations)
    checks = (
        GateResult("semantic_accuracy", accuracy >= 0.95, {"accuracy": accuracy}),
        GateResult("per_class_recall", min(recalls) >= 0.90, {"minimum_recall": min(recalls)}),
        GateResult("camera_code_fraction", code_fraction >= 0.25, {"camera_code_fraction": code_fraction}),
        GateResult("high_saturation", saturation <= 0.08, {"high_saturation": saturation}),
        GateResult("per_class_feature_diversity", all(0.70 <= value <= 1.30 for value in diversity_ratios), {f"label_{label}": value for label, value in enumerate(diversity_ratios)}),
        GateResult("sinkhorn_coverage", coverage_ratio <= 1.5, {"coverage_ratio": coverage_ratio}),
    )
    run = _new_run(config)
    raw_path = run.root / "conditional_evaluation_raw_cmos.pt"
    labels_path = run.root / "conditional_evaluation_labels.pt"
    atomic_torch_save(raw_path, raw)
    atomic_torch_save(labels_path, labels)
    review_artifacts = write_conditional_review_artifacts(root=run.root, raw_cmos=raw, labels=labels, exposure_scale=settings.fixed_exposure_scale)
    passed = all(item.passed for item in checks)
    acceptance = config.run.evidence_tier == "acceptance"
    status = "pending_review" if passed and acceptance else ("diagnostic" if passed else "failed")
    permitted = ("admit_conditional_evaluation",) if status == "pending_review" else ()
    record = EvidenceRecordV2(2, "conditional_acceptance", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint)), {"training_evidence": file_sha256(args.training_evidence), "checkpoint": file_sha256(args.checkpoint), "raw_cmos": file_sha256(raw_path), "labels": file_sha256(labels_path), "review_raw_cmos": review_artifacts.raw_tensor_sha256, "review_preview": review_artifacts.preview_sha256}, checks, permitted, None if passed else "conditional automatic acceptance gate failed")
    _write_evidence(run, "conditional_acceptance_evidence.json", record)
    print(run.root)
    return 0


def admit_conditional_evaluation(args, config: V2Config) -> int:
    _require_conditional(config)
    source = EvidenceRecordV2.from_dict(json.loads(Path(args.evaluation_evidence).read_text(encoding="utf-8")))
    source.validate()
    if source.evidence_kind != "conditional_acceptance" or source.status != "pending_review" or "admit_conditional_evaluation" not in source.permitted_next_actions:
        raise ValueError("conditional evaluation evidence cannot authorize manual admission")
    raw = torch.load(args.evaluation_raw_cmos, map_location="cpu", weights_only=False)
    labels = torch.load(args.evaluation_labels, map_location="cpu", weights_only=False)
    if not torch.is_tensor(raw) or file_sha256(args.evaluation_raw_cmos) != source.input_hashes.get("raw_cmos"):
        raise ValueError("evaluation raw CMOS does not match conditional acceptance evidence")
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.shape[0] != raw.shape[0] or file_sha256(args.evaluation_labels) != source.input_hashes.get("labels"):
        raise ValueError("evaluation labels do not match conditional acceptance evidence")
    selected = torch.cat([torch.nonzero(labels == label, as_tuple=False).flatten()[:16] for label in range(10)])
    reviewed = raw.index_select(0, selected)
    preview_hash = file_sha256(args.evaluation_preview)
    if preview_hash != source.input_hashes.get("review_preview"):
        raise ValueError("conditional evaluation preview does not match evidence")
    review, review_hash = load_conditional_manual_review(args.manual_review, tensor_hash=tensor_sha256(reviewed), preview_hash=preview_hash)
    readable = all(count >= 14 for count in review.readable_counts)
    gate = GateResult("manual_readability_14_of_16", readable, {f"label_{label}": float(count) for label, count in enumerate(review.readable_counts)}, None if readable else "every label requires at least fourteen readable samples")
    run = _new_run(config)
    status = "passed" if readable else "failed"
    record = EvidenceRecordV2(2, "conditional_generation_admission", config.run.evidence_tier, status, run.git_commit, run.dirty_patch_hash, run.config_hash, source.context, {**source.input_hashes, "evaluation_evidence": source.evidence_hash, "manual_review": review_hash}, (*source.gates, gate), ("export",) if readable else (), None if readable else "conditional manual readability gate failed")
    _write_evidence(run, "conditional_generation_admission_evidence.json", record)
    print(run.root)
    return 0


def export_conditional_dataset(args, config: V2Config) -> int:
    _require_conditional(config)
    admission = _passed_evidence(args.acceptance_evidence, kind="conditional_generation_admission")
    device = torch.device(config.run.device)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device)
    settings = config.conditional_generation
    assert settings is not None
    count = args.samples_per_label or settings.samples_per_label
    if count != settings.samples_per_label:
        raise ValueError("formal export locks samples_per_label to the configured value")
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
                raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
                view200 = (raw / settings.fixed_exposure_scale).clamp(0.0, 1.0)
                view28 = (fixed_exposure_view(raw, settings.fixed_exposure_scale) * 255.0).round().to(torch.uint8)
                shard = dataset_root / f"label_{label}_part_{start // settings.shard_size:03d}.pt"
                atomic_torch_save(shard, {"indices": torch.arange(index, index + size), "labels": labels.cpu(), "codes": codes.cpu(), "raw_cmos": raw.cpu(), "fixed_exposure_view": view200.cpu(), "mnist_view_uint8": view28.cpu(), "seed": args.seed})
                shard_hashes.append({"path": shard.name, "sha256": file_sha256(shard), "count": size})
                index += size
    manifest_payload = {"schema_version": 1, "status": "candidate", "architecture_kind": manifest.architecture_kind, "system_signature": manifest.system_signature, "checkpoint_hash": file_sha256(args.checkpoint), "evaluator_checkpoint_hash": manifest.evaluator_checkpoint_hash, "dataset_split_manifest_hash": manifest.dataset_split_manifest_hash, "exposure_scale": settings.fixed_exposure_scale, "samples_per_label": count, "seed": args.seed, "code_distribution": "uniform[-1,1]^8", "shards": shard_hashes, "raw_shape": [1, *config.system.slm_shape], "mnist_shape": [1, 28, 28]}
    manifest_hash = canonical_sha256(manifest_payload)
    atomic_write_json(dataset_root / "manifest.json", manifest_payload | {"manifest_hash": manifest_hash})
    record = EvidenceRecordV2(2, "conditional_dataset_candidate", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=manifest.evaluator_checkpoint_hash, split_hash=manifest.dataset_split_manifest_hash, checkpoint_hash=file_sha256(args.checkpoint), candidate_manifest_hash=manifest_hash), {"generation_admission": admission.evidence_hash, "candidate_manifest": file_sha256(dataset_root / "manifest.json")}, (GateResult("candidate_shards_complete", True, {"samples": float(index)}),), ("audit_memorization", "evaluate_utility"), None)
    _write_evidence(run, "conditional_dataset_candidate_evidence.json", record)
    print(dataset_root)
    return 0


def audit_memorization(args, config: V2Config) -> int:
    _require_conditional(config)
    candidate = _passed_evidence(args.candidate_evidence, kind="conditional_dataset_candidate")
    payload, generated_images, generated_labels = load_candidate_dataset(args.candidate_dataset)
    manifest_hash = canonical_sha256(payload)
    if candidate.context.candidate_dataset_manifest_hash != manifest_hash:
        raise ValueError("candidate evidence does not match the candidate dataset manifest")
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
    report = memorization_statistics(embeddings(generated_images), generated_labels, embeddings(real_images), real_labels)
    class_rates = [float(row["anomalous_rate"]) for row in report["per_class"]]
    passed = float(report["overall_anomalous_rate"]) <= 0.01 and all(rate <= 0.01 for rate in class_rates)
    run = _new_run(config)
    report_path = run.root / "conditional_memorization_report.json"
    atomic_write_json(report_path, report)
    gate = GateResult("nearest_neighbor_memorization", passed, {"overall_anomalous_rate": float(report["overall_anomalous_rate"]), **{f"label_{index}": rate for index, rate in enumerate(class_rates)}}, None if passed else "candidate contains more than one percent anomalously close samples")
    record = EvidenceRecordV2(2, "conditional_memorization_audit", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, candidate.context, {"candidate_evidence": candidate.evidence_hash, "candidate_manifest": file_sha256(Path(args.candidate_dataset) / "manifest.json"), "report": file_sha256(report_path)}, (gate,), ("admit_dataset",) if passed else (), None if passed else "conditional memorization audit failed")
    _write_evidence(run, "conditional_memorization_evidence.json", record)
    print(run.root)
    return 0


def _fixed_low_data_indices(labels: torch.Tensor, *, per_label: int = 600) -> torch.Tensor:
    indices = []
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        if rows.numel() < per_label:
            raise ValueError("MNIST train split lacks required low-data class examples")
        indices.append(rows[:per_label])
    return torch.cat(indices)


def _train_utility_model(images: torch.Tensor, labels: torch.Tensor, *, seed: int, device: torch.device) -> FrozenMNISTEvaluator:
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
            loss = torch.nn.functional.cross_entropy(logits, labels.index_select(0, rows).to(device))
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate_utility(args, config: V2Config) -> int:
    _require_conditional(config)
    candidate = _passed_evidence(args.candidate_evidence, kind="conditional_dataset_candidate")
    payload, synthetic_images, synthetic_labels = load_candidate_dataset(args.candidate_dataset)
    manifest_hash = canonical_sha256(payload)
    if candidate.context.candidate_dataset_manifest_hash != manifest_hash:
        raise ValueError("candidate evidence does not match the candidate dataset manifest")
    device = torch.device(config.run.device)
    train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    low_rows = _fixed_low_data_indices(train_labels)
    regimes = {"low_data": (train_images.index_select(0, low_rows), train_labels.index_select(0, low_rows), 0.0), "full_data": (train_images, train_labels, -0.002)}
    report: dict[str, object] = {"seeds": [1234, 1235, 1236], "test_count": int(test_labels.numel()), "regimes": {}}
    all_passed = True
    for name, (base_images, base_labels, lower_limit) in regimes.items():
        differences, seed_rows = [], []
        for seed in (1234, 1235, 1236):
            baseline = _train_utility_model(base_images, base_labels, seed=seed, device=device)
            augmented = _train_utility_model(torch.cat((base_images, synthetic_images)), torch.cat((base_labels, synthetic_labels)), seed=seed, device=device)
            with torch.no_grad():
                _, base_logits = baseline(test_images.to(device))
                _, augmented_logits = augmented(test_images.to(device))
            base_correct = base_logits.argmax(dim=1).cpu() == test_labels
            augmented_correct = augmented_logits.argmax(dim=1).cpu() == test_labels
            differences.append(augmented_correct.to(torch.float32) - base_correct.to(torch.float32))
            seed_rows.append({"seed": seed, "baseline_accuracy": float(base_correct.to(torch.float32).mean().item()), "augmented_accuracy": float(augmented_correct.to(torch.float32).mean().item()), "baseline_prediction_hash": tensor_sha256(base_logits.argmax(dim=1).cpu()), "augmented_prediction_hash": tensor_sha256(augmented_logits.argmax(dim=1).cpu())})
        lower = paired_bootstrap_lower_bound(torch.cat(differences), seed=1234, resamples=10000)
        passed = lower > lower_limit if name == "low_data" else lower >= lower_limit
        report["regimes"][name] = {"seeds": seed_rows, "bootstrap_lower_bound": lower, "limit": lower_limit, "passed": passed}
        all_passed = all_passed and passed
    run = _new_run(config)
    report_path = run.root / "conditional_utility_report.json"
    atomic_write_json(report_path, report)
    metrics = {f"{name}_lower_bound": float(values["bootstrap_lower_bound"]) for name, values in report["regimes"].items()}
    gate = GateResult("downstream_utility", all_passed, metrics, None if all_passed else "conditional synthetic data did not meet utility gates")
    record = EvidenceRecordV2(2, "conditional_utility", config.run.evidence_tier, "passed" if all_passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, candidate.context, {"candidate_evidence": candidate.evidence_hash, "candidate_manifest": file_sha256(Path(args.candidate_dataset) / "manifest.json"), "report": file_sha256(report_path)}, (gate,), ("admit_dataset",) if all_passed else (), None if all_passed else "conditional utility gate failed")
    _write_evidence(run, "conditional_utility_evidence.json", record)
    print(run.root)
    return 0


def admit_dataset(args, config: V2Config) -> int:
    _require_conditional(config)
    admission = _passed_evidence(args.generation_admission_evidence, kind="conditional_generation_admission")
    memorization = _passed_evidence(args.memorization_evidence, kind="conditional_memorization_audit")
    utility = _passed_evidence(args.utility_evidence, kind="conditional_utility")
    payload, _, _ = load_candidate_dataset(args.candidate_dataset)
    manifest_hash = canonical_sha256(payload)
    expected = _context(config, evaluator_hash=payload["evaluator_checkpoint_hash"], split_hash=payload["dataset_split_manifest_hash"], checkpoint_hash=payload["checkpoint_hash"], candidate_manifest_hash=manifest_hash)
    for record in (memorization, utility):
        record.context.assert_matches(expected)
    if admission.context.system_signature != expected.system_signature or admission.context.config_hash != expected.config_hash or admission.context.checkpoint_hash != expected.checkpoint_hash:
        raise ValueError("generation admission context does not match candidate dataset")
    accepted = {**payload, "status": "accepted", "candidate_manifest_hash": manifest_hash}
    accepted_hash = canonical_sha256(accepted)
    root = Path(args.candidate_dataset)
    accepted_path = root / "accepted_manifest.json"
    if accepted_path.exists():
        raise FileExistsError("accepted dataset manifest already exists")
    atomic_write_json(accepted_path, accepted | {"manifest_hash": accepted_hash})
    run = _new_run(config)
    record = EvidenceRecordV2(2, "conditional_dataset_accepted", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"generation_admission": admission.evidence_hash, "memorization": memorization.evidence_hash, "utility": utility.evidence_hash, "candidate_manifest": file_sha256(root / "manifest.json"), "accepted_manifest": file_sha256(accepted_path)}, (GateResult("all_dataset_acceptance_evidence_bound", True, {"candidate_manifest_hash_bound": True}),), (), None)
    _write_evidence(run, "conditional_dataset_accepted_evidence.json", record)
    print(accepted_path)
    return 0
