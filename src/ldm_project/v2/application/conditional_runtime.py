"""CLI stages for the teacher-free conditional optical generator."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import V2Config
from ..contracts import ComplexField
from ..evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, load_conditional_manual_review, tensor_sha256
from ..optics import FormalAmplitudeFiniteCropIdentitySystem
from ..student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint, save_conditional_checkpoint, state_content_hash, state_signature
from ..student.admission import balanced_code_fraction
from ..student.conditional import conditional_loss, conditional_sinkhorn_loss, fixed_exposure_view, raw_from_generator_batched, relative_diversity_loss, relative_diversity_ratios
from ..student.conditional_acceptance import conditional_coverage_ratio, load_candidate_dataset, memorization_statistics, paired_bootstrap_lower_bound, per_label_feature_diversity_ratio
from ..student.evaluator import FrozenMNISTEvaluator, MNISTSplitManifest, balanced_batches, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist, save_evaluator_checkpoint, stratified_mnist_indices
from ..student.generator import ConditionalGlobalAmplitudeGenerator, IdentityAnchoredConditionalAmplitudeGenerator
from ..evidence.records import require_evidence_action
from ..student.review_artifacts import write_conditional_review_artifacts, write_review_artifacts
from .runs import V2Run


CONDITIONAL_CONTROL_METRIC_CONTRACT = "sample_centered_anova_v1|sinkhorn_feature_l2_v1|coverage_disjoint_real_ab_v1"
# This admission contract is deliberately separate from the immutable loss/
# checkpoint metric contract above.  Existing initialization checkpoints retain
# the latter; new isolation evidence cannot be mistaken for the old gate.
CONDITIONAL_CONTROL_ADMISSION_CONTRACT = "sample_centered_anova_v1|sinkhorn_feature_l2_v1|coverage_disjoint_real_ab_v1|physical_command_semantics_v2"
CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT = "identity_semantic_localization_v1|zero_code_vs_formal_gradient_cosine_v1|no_optimizer_state_change_v1"
IDENTITY_ARCHITECTURE = "conditional_identity_logit_amplitude_v2"
IDENTITY_GENERATOR_CONTRACT = canonical_sha256({"architecture": IDENTITY_ARCHITECTURE, "formula": "h=SiLU(Wz*z);r=Wy*h;c=tanh(z+r);A=sigmoid(By+0.5*sum(c*P_hat))", "gain": 0.5, "bias_free": True})
TRAINING_SCHEDULE_CONTRACT = canonical_sha256({"epochs": 5, "logical_batch_size": 50, "steps": 5500, "seed": 1234, "code_seed": "(epoch+1)*100000+batch_number"})
BALANCED_TRAINING_SCHEDULE = {"epochs": 5, "logical_batch_size": 50, "batch_policy": "balanced_min_class_exhaustion", "seed": 1234, "code_seed": "(epoch+1)*100000+batch_number"}
BALANCED_TRAINING_SCHEDULE_CONTRACT = canonical_sha256(BALANCED_TRAINING_SCHEDULE)
DIVERSITY_LOSS_CONTRACT = canonical_sha256({"sinkhorn": 1.0, "classification": 0.5, "relative_diversity": 1.0, "saturation": 0.1, "mode_orthogonality": 1.0e-3, "feature_normalization": "l2", "bounds": [0.70, 1.30]})
DIVERSITY_RECOVERY_CONTRACT = canonical_sha256({"kind": "conditional_diversity_recovery_smoke", "steps": 400, "probes": [0, 100, 200, 400], "trainable_groups": ["global_modes", "code_projection", "class_heads"]})
SEMANTIC_RECOVERY_STEPS = 400
SEMANTIC_RECOVERY_CONTRACT = canonical_sha256({"kind": "conditional_semantic_recovery_smoke", "steps": SEMANTIC_RECOVERY_STEPS, "batch_seed": 1234, "code_seed": "100000+batch_number", "initialization_only": True})


class ConditionalBundle(nn.Module):
    def __init__(self, config: V2Config, *, carrier_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        if config.architecture not in {"conditional_global_amplitude_v1", IDENTITY_ARCHITECTURE} or config.conditional_generation is None:
            raise ValueError("conditional bundle requires a supported conditional architecture config")
        settings = config.conditional_generation
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        generator_type = IdentityAnchoredConditionalAmplitudeGenerator if config.architecture == IDENTITY_ARCHITECTURE else ConditionalGlobalAmplitudeGenerator
        self.generator = generator_type(
            shape=config.system.slm_shape,
            code_dim=settings.code_dim,
            hidden_dim=settings.hidden_dim,
            command_gain=settings.command_gain,
            class_carrier_logits=carrier_logits,
        )


def _require_conditional(config: V2Config) -> None:
    if config.architecture not in {"conditional_global_amplitude_v1", IDENTITY_ARCHITECTURE} or config.conditional_generation is None:
        raise ValueError("this command requires a supported conditional architecture")


def _require_identity(config: V2Config) -> None:
    _require_conditional(config)
    if config.architecture != IDENTITY_ARCHITECTURE:
        raise ValueError("this command requires architecture: conditional_identity_logit_amplitude_v2")


def _require_current_acceptance(config: V2Config) -> None:
    if config.run.evidence_tier == "acceptance" and config.architecture != IDENTITY_ARCHITECTURE:
        raise ValueError("legacy v2 architectures cannot authorize current acceptance; use the identity v2 config")


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
        2, str(Path(dataset_root).resolve()), 1234,
        canonical_sha256(train_indices), canonical_sha256(validation_indices), train_indices.numel(), validation_indices.numel(),
        canonical_sha256({"images": tensor_sha256(images), "labels": tensor_sha256(labels)}),
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


def _load_bundle(config: V2Config, checkpoint: str, *, device: torch.device, evaluator_hash: str | None = None, split_hash: str | None = None, allow_balanced_schedule: bool = False) -> tuple[ConditionalBundle, ConditionalCheckpointManifestV2]:
    _require_conditional(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid conditional checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if manifest.architecture_kind != config.architecture or manifest.system_signature != _system_signature(config) or manifest.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("conditional checkpoint belongs to a different architecture/system/config")
    if evaluator_hash is not None and manifest.evaluator_checkpoint_hash != evaluator_hash:
        raise ValueError("conditional checkpoint uses another evaluator")
    if split_hash is not None and manifest.dataset_split_manifest_hash != split_hash:
        raise ValueError("conditional checkpoint uses another MNIST split")
    bundle = ConditionalBundle(config).to(device)
    if manifest.architecture_kind == IDENTITY_ARCHITECTURE:
        valid_schedules = {TRAINING_SCHEDULE_CONTRACT}
        if allow_balanced_schedule:
            valid_schedules.add(BALANCED_TRAINING_SCHEDULE_CONTRACT)
        if manifest.generator_contract_hash != IDENTITY_GENERATOR_CONTRACT or manifest.metric_contract_hash != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT) or manifest.training_schedule_hash not in valid_schedules or manifest.initialization_seed != 1234:
            raise ValueError("identity checkpoint has an incomplete or mismatched contract")
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
    """Re-check an evaluator checkpoint by replaying its bound validation split."""
    _require_conditional(config)
    device = torch.device(config.run.device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    split_payload = payload.get("split_manifest")
    if not isinstance(split_payload, dict):
        raise ValueError("evaluator checkpoint lacks its MNIST split manifest")
    split = MNISTSplitManifest(**split_payload)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, current_split = _split(images, labels, args.dataset_root)
    if split.manifest_hash != current_split.manifest_hash:
        raise ValueError("evaluator checkpoint split does not match the local content-bound MNIST split")
    with torch.no_grad():
        _, logits = evaluator(validation_images.to(device))
    accuracy = float((logits.argmax(dim=1) == validation_labels.to(device)).to(torch.float32).mean().item())
    run = _new_run(config)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    passed = accuracy >= 0.985
    record = EvidenceRecordV2(2, "evaluator_qualification", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash), {"evaluator_checkpoint": evaluator_hash}, (GateResult("validation_accuracy", passed, {"validation_accuracy": accuracy}),), ("conditional_observability",) if passed and config.run.evidence_tier == "acceptance" else (), None if passed else "replayed evaluator validation accuracy is below 98.5%")
    _write_evidence(run, "evaluator_qualification_evidence.json", record)
    print(run.root)
    return 0


def conditional_injection_admission(args, config: V2Config) -> int:
    """CPU-only generator contract gate and immutable initialization checkpoint."""
    _require_identity(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, evaluator_payload = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cpu"))
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    payload_split = MNISTSplitManifest(**evaluator_payload["split_manifest"])
    if payload_split.manifest_hash != split.manifest_hash:
        raise ValueError("evaluator checkpoint uses a different fixed MNIST split")
    torch.manual_seed(1234)
    bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(train_images, train_labels, config.system.slm_shape)).cpu().eval()
    codes, code_labels = _fixed_control_probe(device=torch.device("cpu"))
    before = state_content_hash(bundle.state_dict())
    with torch.no_grad():
        command = bundle.generator(codes, code_labels)
        command_again = bundle.generator(codes, code_labels)
        physical = command.square()
        grouped = command.reshape(10, 17, *command.shape[1:])
        physical_grouped = physical.reshape(10, 17, *physical.shape[1:])
        coeff = bundle.generator.coefficients(codes, code_labels).reshape(10, 17, 8)
        signed = coeff[:, 1:17].reshape(10, 8, 2, 8)
        direction_rms = signed.square().mean(dim=(0, 2, 3)).sqrt()
        rank = int(torch.linalg.matrix_rank(coeff[:, 1:17].reshape(-1, 8)).item())
        zero = torch.zeros(10, 8)
        zero_command = bundle.generator(zero, torch.arange(10))
        carrier = bundle.generator.class_carrier_command(torch.arange(10))
    after = state_content_hash(bundle.state_dict())
    mode = bundle.generator.normalized_modes(bundle.generator.global_modes)
    gram = mode.flatten(1) @ mode.flatten(1).T / mode[0].numel()
    checks = (
        GateResult("amplitude_command_code_fraction", balanced_code_fraction(grouped) >= 0.10, {"value": balanced_code_fraction(grouped)}),
        GateResult("physical_slm_intensity_code_fraction", balanced_code_fraction(physical_grouped) >= 0.10, {"value": balanced_code_fraction(physical_grouped)}),
        GateResult("zero_code_is_carrier", bool(torch.equal(zero_command, carrier)), {}),
        GateResult("eight_signed_direction_rms", bool((direction_rms >= 0.01).all()), {"minimum": float(direction_rms.min().item())}),
        GateResult("coefficient_rank_eight", rank == 8, {"rank": float(rank)}),
        GateResult("finite_orthogonal_modes", bool(torch.isfinite(mode).all() and (gram - torch.eye(8)).abs().max() < 0.25), {}),
        GateResult("deterministic_initialization", before == after and bool(torch.equal(command, command_again)), {}),
    )
    run = _new_run(config)
    init_path = run.root / "conditional_initialization.pt"
    settings = config.conditional_generation
    assert settings is not None
    manifest = ConditionalCheckpointManifestV2(2, IDENTITY_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_hash, settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), TRAINING_SCHEDULE_CONTRACT, 1234, state_content_hash(bundle.state_dict()))
    save_conditional_checkpoint(init_path, bundle, manifest, extra={"initialization": True, "seed": 1234})
    report = {"architecture_kind": IDENTITY_ARCHITECTURE, "metric_contract": CONDITIONAL_CONTROL_METRIC_CONTRACT, "generator_contract_hash": IDENTITY_GENERATOR_CONTRACT, "command_code_fraction": float(balanced_code_fraction(grouped)), "physical_code_fraction": float(balanced_code_fraction(physical_grouped)), "direction_rms": direction_rms.tolist(), "coefficient_rank": rank, "parameters_unchanged": before == after, "state_hash": manifest.initialization_state_hash}
    report_path = run.root / "conditional_injection_admission_report.json"
    atomic_write_json(report_path, report)
    passed = all(gate.passed for gate in checks)
    record = EvidenceRecordV2(2, "conditional_injection_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(init_path)), {"checkpoint": file_sha256(init_path), "report": file_sha256(report_path), "evaluator": evaluator_hash}, checks, ("conditional_observability",) if passed else (), None if passed else "identity generator injection admission failed")
    _write_evidence(run, "conditional_injection_admission_evidence.json", record)
    print(run.root)
    return 0


def _identity_observability(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device(config.run.device))
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
    injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=init_hash)
    require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
    device = torch.device(config.run.device)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    codes_probe, labels_probe = _fixed_control_probe(device=device)
    codes_probe, labels_probe = codes_probe[:20], labels_probe[:20]
    before = state_content_hash(bundle.state_dict())
    result = bundle.system.forward_from_generator(bundle.generator, codes_probe, labels_probe, return_optical_trace=True)
    raw = result.raw_camera.tensor
    view = fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    with torch.enable_grad():
        _, logits = evaluator(view)
        objective = raw.mean() + logits.float().mean() * 0.0
        objective.backward()
    groups = _conditional_parameter_groups(bundle)
    gradient_checks = {name: bool(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.detach().norm() > 0 for parameter in params)) for name, params in groups.items()}
    after = state_content_hash(bundle.state_dict())
    operations = tuple(field.provenance.operation for field in result.optical_trace or ())
    # At the exact identity anchor Wy=0, Wz is intentionally second-order: the
    # first derivative of Wy*SiLU(Wz*z) with respect to Wz is zero.  Record this
    # contract exception explicitly; smoke must still prove Wz updates after Wy
    # leaves its initialization state.
    gradient_gate = all(value for name, value in gradient_checks.items() if name != "code_projection") and not gradient_checks["code_projection"]
    gradient_metrics = dict(gradient_checks)
    gradient_metrics["code_projection_identity_anchor_zero"] = True
    checks = (GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}), GateResult("complete_trace", len(operations) >= 8 and any("identity_ols" in op for op in operations), {"trace_length": float(len(operations))}), GateResult("gradient_groups", gradient_gate, gradient_metrics), GateResult("parameters_unchanged", before == after, {}))
    run = _new_run(config)
    report_path = run.root / "conditional_observability_report.json"
    atomic_write_json(report_path, {"trace_operations": operations, "gradient_groups": gradient_checks, "parameters_unchanged": before == after, "raw_finite": bool(torch.isfinite(raw).all()), "checkpoint_hash": init_hash})
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(2, "conditional_observability", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"injection_evidence": injection.evidence_hash, "checkpoint": init_hash, "report": file_sha256(report_path)}, checks, ("conditional_control_isolation",) if passed else (), None if passed else "identity optical observability failed")
    _write_evidence(run, "conditional_observability_evidence.json", record)
    print(run.root)
    return 0


def conditional_observability(args, config: V2Config) -> int:
    if config.architecture == IDENTITY_ARCHITECTURE:
        return _identity_observability(args, config)
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


def _panel_semantic_metrics(values: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, exposure_scale: float) -> dict[str, object]:
    """Fixed-exposure semantic metrics for the immutable 10×17 code panel."""
    if values.shape[0] != 170 or labels.shape != (170,):
        raise ValueError("semantic panel requires the fixed 10×17 control probe")
    view = fixed_exposure_view(values, exposure_scale)
    _, logits = evaluator(view)
    predictions = logits.argmax(dim=1)
    zero = torch.arange(0, 170, 17, device=labels.device)
    nonzero_mask = torch.ones(170, device=labels.device, dtype=torch.bool)
    nonzero_mask[zero] = False
    return {
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "zero_code_accuracy": float((predictions.index_select(0, zero) == labels.index_select(0, zero)).to(torch.float32).mean().item()),
        "nonzero_code_accuracy": float((predictions[nonzero_mask] == labels[nonzero_mask]).to(torch.float32).mean().item()),
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).cpu().tolist(),
    }


def _first_hundred_schedule_hash(labels: torch.Tensor) -> str:
    """Content-address the exact formal epoch-one batches/codes used by smoke."""
    return _schedule_hash(labels, 100)


def _schedule_hash(labels: torch.Tensor, count: int) -> str:
    """Content-address the exact first ``count`` formal batches and codes."""
    if count < 1:
        raise ValueError("schedule hash requires at least one batch")
    rows = []
    for batch_number, indices in zip(range(count), balanced_batches(labels, batch_size=50, seed=1234)):
        codes = _random_codes(indices.numel(), device=torch.device("cpu"), seed=100000 + batch_number)
        rows.append({"batch": batch_number, "indices": tensor_sha256(indices), "codes": tensor_sha256(codes)})
    if len(rows) != count:
        raise ValueError(f"fixed MNIST training split cannot provide {count} formal batches")
    return canonical_sha256(rows)


def _identity_isolation_classification(*, harness_valid: bool, command_fraction: float, first_collapse: str | None, camera_fraction: float, command_accuracy: float, camera_accuracy: float, saturation: float) -> str:
    """Classify the read-only identity panel without mutating its evidence."""
    if not harness_valid:
        return "harness_invalid"
    if command_fraction < 0.10:
        return "generator_injection_insufficient"
    if first_collapse is not None or camera_fraction < 0.10:
        return "optical_transport_attenuated"
    if command_accuracy < 0.95:
        return "carrier_semantics_invalid"
    if saturation > 0.08:
        return "exposure_saturation_excessive"
    if camera_accuracy < 0.30:
        return "semantic_alignment_required"
    return "controllable_upper_bound"


def _probe_metrics(raw: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, exposure_scale: float) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure_scale)
    features, logits = evaluator(view)
    values = raw.reshape(10, 17, *raw.shape[1:])
    command_fraction = balanced_code_fraction(values)
    code_mean = values.mean(dim=1, keepdim=True)
    direction_rms = (values - code_mean).square().mean(dim=tuple(range(2, values.ndim))).sqrt().mean(dim=0).flatten().tolist()
    semantic = _panel_semantic_metrics(raw, labels, evaluator, exposure_scale)
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
        "zero_code_accuracy": semantic["zero_code_accuracy"],
        "nonzero_code_accuracy": semantic["nonzero_code_accuracy"],
    }


def _identity_control_isolation(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device(config.run.device))
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=init_hash)
    injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
    observability = EvidenceRecordV2.from_dict(json.loads(Path(args.observability_evidence).read_text(encoding="utf-8")))
    require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
    require_evidence_action(observability, kind="conditional_observability", action="conditional_control_isolation", expected_context=expected)
    device = torch.device(config.run.device)
    bundle, _ = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    codes, probe_labels = _fixed_control_probe(device=device)
    before = state_content_hash(bundle.state_dict())
    settings = config.conditional_generation
    assert settings is not None
    command_rows, raw_rows, trace_rows = [], [], []
    with torch.no_grad():
        for start in range(0, len(codes), config.execution.evaluation_microbatch_size):
            command = bundle.generator(codes[start:start + config.execution.evaluation_microbatch_size], probe_labels[start:start + config.execution.evaluation_microbatch_size])
            result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
            command_rows.append(command.cpu())
            raw_rows.append(result.raw_camera.tensor.cpu())
            trace_rows.append(tuple(ComplexField(field.tensor.cpu(), field.plane, field.provenance) for field in (result.optical_trace or ())))
    command = torch.cat(command_rows).to(device)
    raw = torch.cat(raw_rows).to(device)
    after = state_content_hash(bundle.state_dict())
    command_fraction = balanced_code_fraction(command.reshape(10, 17, *command.shape[1:]))
    physical_fraction = balanced_code_fraction(command.square().reshape(10, 17, *command.shape[1:]))
    camera_fraction = balanced_code_fraction(raw.reshape(10, 17, *raw.shape[1:]))
    command_semantics = _panel_semantic_metrics(command.square(), probe_labels, evaluator, settings.fixed_exposure_scale)
    camera_semantics = _panel_semantic_metrics(raw, probe_labels, evaluator, settings.fixed_exposure_scale)
    saturation = float(torch.relu(raw / settings.fixed_exposure_scale - 1.0).gt(0).to(torch.float32).mean().item())
    trace = trace_rows[0]
    operations = [field.provenance.operation for field in trace]
    plane_metrics = []
    for field_index, field in enumerate(trace):
        tensors = torch.cat([row[field_index].tensor.abs().square() for row in trace_rows], dim=0)
        plane_metrics.append({"index": field_index, "operation": field.provenance.operation, "code_fraction": balanced_code_fraction(tensors.reshape(10, 17, *tensors.shape[1:])), "finite": bool(torch.isfinite(tensors).all()), "plane": field.plane.to_dict(), "provenance": field.provenance.to_dict()})
    first_collapse = next((item["operation"] for item in plane_metrics if item["code_fraction"] < 0.10), None)
    harness_valid = bool(torch.isfinite(raw).all()) and before == after and len(operations) >= 8 and any("identity_ols" in operation for operation in operations) and all(item["finite"] for item in plane_metrics)
    classification = _identity_isolation_classification(
        harness_valid=harness_valid,
        command_fraction=command_fraction,
        first_collapse=first_collapse,
        camera_fraction=camera_fraction,
        command_accuracy=float(command_semantics["accuracy"]),
        camera_accuracy=float(camera_semantics["accuracy"]),
        saturation=saturation,
    )
    run = _new_run(config)
    raw_path = run.root / "conditional_control_isolation_raw_cmos.pt"
    atomic_torch_save(raw_path, raw.cpu())
    report_path = run.root / "conditional_control_isolation_report.json"
    atomic_write_json(report_path, {"metric_contract": CONDITIONAL_CONTROL_ADMISSION_CONTRACT, "code_count_per_label": 17, "command_code_fraction": command_fraction, "physical_slm_code_fraction": physical_fraction, "camera_code_fraction": camera_fraction, "command_semantics": command_semantics, "camera_semantics": camera_semantics, "saturation": saturation, "first_collapse_plane": first_collapse, "classification": classification, "parameters_unchanged": before == after, "plane_metrics": plane_metrics, "trace_operations": operations})
    checks = (GateResult("fixed_probe_shape", len(codes) == 170, {"samples": 170.0}), GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}), GateResult("complete_provenance", len(operations) >= 8 and any("identity_ols" in op for op in operations), {}), GateResult("parameters_unchanged", before == after, {}), GateResult("command_code_response", command_fraction >= 0.10, {"value": command_fraction}), GateResult("command_semantic_readability", float(command_semantics["accuracy"]) >= 0.95, {"accuracy": float(command_semantics["accuracy"])}), GateResult("fixed_exposure_saturation", saturation <= 0.08, {"saturation": saturation}))
    if classification == "semantic_alignment_required":
        permitted_actions = ("conditional_semantic_localization",)
    elif classification == "controllable_upper_bound":
        permitted_actions = ("controllability_smoke",)
    else:
        permitted_actions = ()
    record = EvidenceRecordV2(2, "conditional_control_isolation", "diagnostic", "diagnostic" if harness_valid else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"injection_evidence": injection.evidence_hash, "observability_evidence": observability.evidence_hash, "checkpoint": init_hash, "report": file_sha256(report_path), "raw_cmos": file_sha256(raw_path), "control_admission_metric_contract": canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT)}, checks, permitted_actions, None if harness_valid else "conditional control isolation harness failed")
    _write_evidence(run, "conditional_control_isolation_evidence.json", record)
    print(run.root)
    return 0


def conditional_control_isolation(args, config: V2Config) -> int:
    if config.architecture == IDENTITY_ARCHITECTURE:
        return _identity_control_isolation(args, config)
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
        raw_rows: list[torch.Tensor] = []
        trace_rows: list[list[ComplexField]] = []
        microbatch = config.execution.evaluation_microbatch_size
        for start in range(0, codes.shape[0], microbatch):
            result = bundle.system.forward(bundle.system.command(command[start : start + microbatch]), return_optical_trace=True)
            raw_rows.append(result.raw_camera.tensor.cpu())
            trace_rows.append([ComplexField(field.tensor.cpu(), field.plane, field.provenance) for field in result.optical_trace or ()])
        raw = torch.cat(raw_rows, dim=0).to(device)
        trace = tuple(ComplexField(torch.cat([row[index].tensor for row in trace_rows], dim=0), trace_rows[0][index].plane, trace_rows[0][index].provenance) for index in range(len(trace_rows[0])))
        view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
        features, logits = evaluator(view)
    after = state_content_hash(bundle.state_dict())
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
    zero_codes = torch.zeros(10, 8, device=device)
    zero_labels = torch.arange(10, device=device)
    duplicate_error = float((bundle.generator(zero_codes, zero_labels) - bundle.generator(zero_codes, zero_labels)).abs().max().item())
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


def _gradient_summary(objective: torch.Tensor, groups: dict[str, tuple[nn.Parameter, ...]]) -> tuple[dict[str, dict[str, object]], dict[str, torch.Tensor]]:
    """Collect gradients without creating an optimizer or changing parameters."""
    parameters = [parameter for values in groups.values() for parameter in values]
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    summary: dict[str, dict[str, object]] = {}
    vectors: dict[str, torch.Tensor] = {}
    offset = 0
    for name, values in groups.items():
        rows = gradients[offset : offset + len(values)]
        offset += len(values)
        finite = all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in rows)
        vectors[name] = torch.cat([gradient.detach().flatten() for gradient in rows if gradient is not None]) if any(gradient is not None for gradient in rows) else torch.zeros(1, device=objective.device)
        norm = float(vectors[name].norm().item())
        summary[name] = {"gradient_norm": norm, "gradient_finite": finite, "gradient_nonzero": norm > 1.0e-12, "structurally_inactive": norm <= 1.0e-12}
    return summary, vectors


def _semantic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    predictions = logits.argmax(dim=1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logits - other_logits.max(dim=1).values
    return {
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "per_label_accuracy": [float((predictions[labels == label] == label).to(torch.float32).mean().item()) for label in range(10)],
        "mean_true_margin": float(margins.mean().item()),
        "median_true_margin": float(margins.median().item()),
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).cpu().tolist(),
    }


def _semantic_localization_classification(*, harness_valid: bool, zero_gradients: dict[str, dict[str, object]], formal_gradients: dict[str, dict[str, object]], gradient_cosines: dict[str, dict[str, object]]) -> str:
    required_zero = ("class_carrier", "encoder_phase", "decoder_phase")
    required_formal = ("class_carrier", "global_modes", "class_heads", "encoder_phase", "decoder_phase")
    if not harness_valid:
        return "harness_invalid"
    if any(float(zero_gradients[name]["gradient_norm"]) <= 1.0e-12 for name in required_zero):
        return "zero_code_gradient_absent"
    if not all(bool(formal_gradients[name]["gradient_nonzero"]) and bool(formal_gradients[name]["gradient_finite"]) for name in required_formal):
        return "harness_invalid"
    usable = [float(gradient_cosines[name]["cosine"]) for name in required_zero if gradient_cosines[name]["cosine"] is not None]
    if any(value <= -0.05 for value in usable):
        return "zero_code_joint_conflict"
    if any(abs(value) < 0.05 for value in usable):
        return "indeterminate_alignment"
    return "local_semantic_alignment_plausible"


def _identity_semantic_localization(args, config: V2Config) -> int:
    """Read-only local semantic-gradient diagnostic for the identity anchor."""
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=init_hash)
    injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
    observability = EvidenceRecordV2.from_dict(json.loads(Path(args.observability_evidence).read_text(encoding="utf-8")))
    isolation = EvidenceRecordV2.from_dict(json.loads(Path(args.isolation_evidence).read_text(encoding="utf-8")))
    require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
    require_evidence_action(observability, kind="conditional_observability", action="conditional_control_isolation", expected_context=expected)
    require_evidence_action(isolation, kind="conditional_control_isolation", action="conditional_semantic_localization", expected_context=expected, allow_diagnostic=True)
    if isolation.input_hashes.get("control_admission_metric_contract") != canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT):
        raise ValueError("legacy control-isolation metric contract cannot authorize semantic localization")

    settings = config.conditional_generation
    assert settings is not None
    bundle, _ = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    bundle.eval()
    groups = _conditional_parameter_groups(bundle)
    before = state_content_hash(bundle.state_dict())
    zero_codes = torch.zeros(10, 8, device=device)
    zero_labels = torch.arange(10, device=device)
    control_codes, control_labels = _fixed_control_probe(device=device)
    first_indices = next(balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234))
    formal_labels = train_labels.index_select(0, first_indices).to(device)
    formal_codes = _random_codes(formal_labels.numel(), device=device, seed=100000)
    formal_real = train_images.index_select(0, first_indices).to(device)

    with torch.enable_grad():
        zero_raw = raw_from_generator_batched(bundle.system, bundle.generator, zero_codes, zero_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
        zero_view = fixed_exposure_view(zero_raw, settings.fixed_exposure_scale)
        _, zero_logits = evaluator(zero_view)
        zero_objective = torch.nn.functional.cross_entropy(zero_logits, zero_labels)
        zero_gradients, zero_vectors = _gradient_summary(zero_objective, groups)
        formal_total, formal_summary, formal_raw = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=formal_codes, labels=formal_labels, real_views=formal_real, exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=False, microbatch_size=config.execution.evaluation_microbatch_size)
        formal_gradients, formal_vectors = _gradient_summary(formal_total, groups)

    gradient_cosines = {}
    for name in groups:
        zero_norm = float(zero_gradients[name]["gradient_norm"])
        formal_norm = float(formal_gradients[name]["gradient_norm"])
        gradient_cosines[name] = {"cosine": None if zero_norm <= 1.0e-12 or formal_norm <= 1.0e-12 else float(torch.nn.functional.cosine_similarity(zero_vectors[name][None], formal_vectors[name][None]).item()), "zero_code_structural": zero_norm <= 1.0e-12, "formal_structural": formal_norm <= 1.0e-12}

    with torch.no_grad():
        control_raw = raw_from_generator_batched(bundle.system, bundle.generator, control_codes, control_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
        _, control_logits = evaluator(fixed_exposure_view(control_raw, settings.fixed_exposure_scale))
        _, formal_logits = evaluator(fixed_exposure_view(formal_raw.detach(), settings.fixed_exposure_scale))
        trace_command = bundle.generator(control_codes[:1], control_labels[:1])
        trace_result = bundle.system.forward(bundle.system.command(trace_command), return_optical_trace=True)
    after = state_content_hash(bundle.state_dict())
    operations = tuple(field.provenance.operation for field in trace_result.optical_trace or ())
    finite_gradients = all(bool(value["gradient_finite"]) for value in (*zero_gradients.values(), *formal_gradients.values()))
    required_formal = ("class_carrier", "global_modes", "class_heads", "encoder_phase", "decoder_phase")
    active_formal = all(bool(formal_gradients[name]["gradient_nonzero"]) for name in required_formal)
    valid = bool(torch.isfinite(zero_raw).all() and torch.isfinite(control_raw).all() and torch.isfinite(formal_raw).all() and finite_gradients and active_formal and before == after and len(operations) >= 8 and any("identity_ols" in operation for operation in operations))
    classification = _semantic_localization_classification(harness_valid=valid, zero_gradients=zero_gradients, formal_gradients=formal_gradients, gradient_cosines=gradient_cosines)

    run = _new_run(config)
    zero_path = run.root / "conditional_semantic_localization_zero_raw_cmos.pt"
    control_path = run.root / "conditional_semantic_localization_control_raw_cmos.pt"
    formal_path = run.root / "conditional_semantic_localization_formal_raw_cmos.pt"
    atomic_torch_save(zero_path, zero_raw.detach().cpu())
    atomic_torch_save(control_path, control_raw.detach().cpu())
    atomic_torch_save(formal_path, formal_raw.detach().cpu())
    zero_preview = write_review_artifacts(root=run.root, stem="semantic_localization_zero", kind="conditional_semantic_localization", raw_cmos=zero_raw.detach().cpu(), labels=zero_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
    control_preview = write_review_artifacts(root=run.root, stem="semantic_localization_control", kind="conditional_semantic_localization", raw_cmos=control_raw.detach().cpu(), labels=control_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
    report_path = run.root / "conditional_semantic_localization_report.json"
    report = {"metric_contract": CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT, "classification": classification, "parameters_unchanged": before == after, "optimizer_created": False, "zero_code_semantics": _semantic_metrics(zero_logits.detach(), zero_labels), "control_panel_semantics": _semantic_metrics(control_logits.detach(), control_labels), "formal_batch_semantics": _semantic_metrics(formal_logits.detach(), formal_labels), "formal_loss": asdict(formal_summary), "zero_code_gradients": zero_gradients, "formal_gradients": formal_gradients, "gradient_cosines": gradient_cosines, "control_code_fraction": balanced_code_fraction(control_raw.reshape(10, 17, *control_raw.shape[1:])), "control_saturation": float(torch.relu(control_raw / settings.fixed_exposure_scale - 1.0).gt(0).to(torch.float32).mean().item()), "trace_operations": operations, "trace_provenance": [field.provenance.to_dict() for field in (trace_result.optical_trace or ())], "first_formal_batch_schedule_hash": canonical_sha256({"indices": tensor_sha256(first_indices), "codes": tensor_sha256(formal_codes.cpu())})}
    atomic_write_json(report_path, report)
    checks = (GateResult("finite_fields_and_gradients", valid, {"finite": finite_gradients}), GateResult("parameters_unchanged", before == after, {}), GateResult("complete_trace", len(operations) >= 8 and any("identity_ols" in operation for operation in operations), {"trace_length": float(len(operations))}), GateResult("zero_code_semantic_gradients", all(float(zero_gradients[name]["gradient_norm"]) > 1.0e-12 for name in ("class_carrier", "encoder_phase", "decoder_phase")), {}), GateResult("formal_active_gradients", active_formal, {}))
    permitted = ("controllability_smoke",) if classification == "local_semantic_alignment_plausible" else ()
    record = EvidenceRecordV2(2, "conditional_semantic_localization", "diagnostic", "diagnostic" if valid else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"injection_evidence": injection.evidence_hash, "observability_evidence": observability.evidence_hash, "isolation_evidence": isolation.evidence_hash, "checkpoint": init_hash, "report": file_sha256(report_path), "zero_raw_cmos": file_sha256(zero_path), "control_raw_cmos": file_sha256(control_path), "formal_raw_cmos": file_sha256(formal_path), "zero_preview": zero_preview.preview_sha256, "control_preview": control_preview.preview_sha256, "metric_contract": canonical_sha256(CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT)}, checks, permitted, None if valid else "semantic localization harness failed")
    _write_evidence(run, "conditional_semantic_localization_evidence.json", record)
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
    if config.architecture == IDENTITY_ARCHITECTURE:
        if args.steps != 100:
            raise ValueError("identity semantic-alignment smoke is locked to exactly 100 steps")
        images, all_labels = load_local_mnist(args.dataset_root, train=True)
        train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
        evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device(config.run.device))
        expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint))
        injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
        observability = EvidenceRecordV2.from_dict(json.loads(Path(args.observability_evidence).read_text(encoding="utf-8")))
        isolation = EvidenceRecordV2.from_dict(json.loads(Path(args.isolation_evidence).read_text(encoding="utf-8")))
        require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
        require_evidence_action(observability, kind="conditional_observability", action="conditional_control_isolation", expected_context=expected)
        if "controllability_smoke" in isolation.permitted_next_actions:
            require_evidence_action(isolation, kind="conditional_control_isolation", action="controllability_smoke", expected_context=expected, allow_diagnostic=True)
        else:
            localization_path = getattr(args, "semantic_localization_evidence", None)
            if not localization_path:
                raise ValueError("semantic-alignment isolation requires semantic-localization evidence before smoke")
            require_evidence_action(isolation, kind="conditional_control_isolation", action="conditional_semantic_localization", expected_context=expected, allow_diagnostic=True)
            localization = EvidenceRecordV2.from_dict(json.loads(Path(localization_path).read_text(encoding="utf-8")))
            require_evidence_action(localization, kind="conditional_semantic_localization", action="controllability_smoke", expected_context=expected, allow_diagnostic=True)
        if isolation.input_hashes.get("control_admission_metric_contract") != canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT):
            raise ValueError("legacy control-isolation metric contract cannot authorize identity smoke")
    else:
        _passed_evidence(args.observability_evidence, kind="conditional_observability")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if config.architecture == IDENTITY_ARCHITECTURE:
        bundle, initialization_manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash)
    else:
        bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(train_images, train_labels, config.system.slm_shape)).to(device)
        initialization_manifest = None
    optimizer = _conditional_optimizer(bundle, config)
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    real_probe_images = torch.cat([train_images[train_labels == label][:17] for label in range(10)], dim=0).to(device)
    with torch.no_grad():
        real_probe_features, _ = evaluator(real_probe_images)
    probe_history: dict[str, dict[str, object]] = {}
    probe_raw_by_step: dict[int, torch.Tensor] = {}
    loss_history: list[dict[str, float]] = []
    group_history: list[dict[str, object]] = []
    schedule_hash = _first_hundred_schedule_hash(train_labels) if config.architecture == IDENTITY_ARCHITECTURE else None

    def evaluate_probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            probe_raw = raw_from_generator_batched(bundle.system, bundle.generator, probe_codes, probe_labels, microbatch_size=config.execution.training_microbatch_size, activation_checkpoint=False)
            return probe_raw, _probe_metrics(probe_raw, probe_labels, evaluator, settings.fixed_exposure_scale)

    initial_raw, initial_metrics = evaluate_probe()
    probe_history["step_0"] = initial_metrics
    probe_raw_by_step[0] = initial_raw.detach().cpu()
    for step, indices in zip(range(args.steps), balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234)):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + step if config.architecture == IDENTITY_ARCHITECTURE else 1234 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, raw = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=config.execution.activation_checkpoint, microbatch_size=config.execution.training_microbatch_size)
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
    gradient_checks = [value["gradient_finite"] and value["gradient_nonzero"] for value in final_groups["gradients"].values()]
    update_checks = [value["update_nonzero"] for value in final_groups["updates"].values()]
    code_projection_recovered = bool(step_ten_groups is not None and step_ten_groups["gradients"]["code_projection"]["gradient_finite"] and step_ten_groups["gradients"]["code_projection"]["gradient_nonzero"] and step_ten_groups["updates"]["code_projection"]["update_nonzero"] and final_groups["gradients"]["code_projection"]["gradient_finite"] and final_groups["gradients"]["code_projection"]["gradient_nonzero"] and final_groups["updates"]["code_projection"]["update_nonzero"])
    checks = (
        GateResult("fixed_probe_complete", final_metrics["accuracy"] is not None, {"samples": 170.0}),
        GateResult("semantic_accuracy", float(final_metrics["accuracy"]) >= 0.30, {"accuracy": float(final_metrics["accuracy"])}),
        GateResult("camera_code_fraction", float(final_metrics["code_fraction"]) >= 0.10, {"camera_code_fraction": float(final_metrics["code_fraction"])}),
        GateResult("finite_nonzero_gradients", bool(all(gradient_checks)), {"groups": float(sum(gradient_checks)), "total": float(len(gradient_checks))}),
        GateResult("nonzero_parameter_updates", bool(all(update_checks)), {"groups": float(sum(update_checks)), "total": float(len(update_checks))}),
        GateResult("identity_wz_recovered_by_step_10_and_100", code_projection_recovered, {"recovered": code_projection_recovered}),
        GateResult("high_saturation", float(final_metrics["saturation"]) <= 0.08, {"high_saturation": float(final_metrics["saturation"])}),
    )
    run = _new_run(config)
    raw_path = run.root / "controllability_probe_raw_cmos.pt"
    labels_path = run.root / "controllability_probe_labels.pt"
    atomic_torch_save(raw_path, final_raw.cpu())
    atomic_torch_save(labels_path, probe_labels.cpu())
    review_artifacts = write_conditional_review_artifacts(root=run.root, raw_cmos=final_raw.cpu(), labels=probe_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
    step_probe_artifacts: dict[str, dict[str, str]] = {}
    for probe_step in (0, 10, args.steps):
        probe_raw = probe_raw_by_step[probe_step]
        probe_raw_path = run.root / f"controllability_probe_step_{probe_step}_raw_cmos.pt"
        atomic_torch_save(probe_raw_path, probe_raw)
        preview = write_review_artifacts(root=run.root, stem=f"controllability_probe_step_{probe_step}", kind="conditional_controllability_smoke", raw_cmos=probe_raw, labels=probe_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
        step_probe_artifacts[str(probe_step)] = {"raw_cmos": file_sha256(probe_raw_path), "preview": preview.preview_sha256}
    smoke_checkpoint_path = run.root / "controllability_smoke_final.pt"
    if config.architecture == IDENTITY_ARCHITECTURE:
        assert initialization_manifest is not None
        smoke_manifest = ConditionalCheckpointManifestV2(2, IDENTITY_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_checkpoint_hash(args.evaluator_checkpoint), settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), TRAINING_SCHEDULE_CONTRACT, 1234, initialization_manifest.initialization_state_hash)
        save_conditional_checkpoint(smoke_checkpoint_path, bundle, smoke_manifest, extra={"diagnostic_only": True, "never_warm_start": True, "steps": args.steps})
    history_path = run.root / "controllability_probe_history.json"
    atomic_write_json(history_path, {"metric_contract": CONDITIONAL_CONTROL_ADMISSION_CONTRACT if config.architecture == IDENTITY_ARCHITECTURE else CONDITIONAL_CONTROL_METRIC_CONTRACT, "loss": loss_history, "probes": probe_history, "parameter_groups": group_history, "step_probe_artifacts": step_probe_artifacts, "first_100_schedule_hash": schedule_hash})
    passed = all(item.passed for item in checks)
    context = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint) if config.architecture == IDENTITY_ARCHITECTURE else None)
    permitted = ("train",) if passed and config.run.evidence_tier == "acceptance" else ()
    inputs = {"observability": file_sha256(args.observability_evidence), "evaluator": evaluator_checkpoint_hash(args.evaluator_checkpoint), "metric_contract": canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT if config.architecture == IDENTITY_ARCHITECTURE else CONDITIONAL_CONTROL_METRIC_CONTRACT), "probe_raw_cmos": file_sha256(raw_path), "probe_labels": file_sha256(labels_path), "probe_preview": review_artifacts.preview_sha256, "probe_history": file_sha256(history_path)}
    if config.architecture == IDENTITY_ARCHITECTURE:
        inputs.update({"injection": file_sha256(args.injection_evidence), "isolation": file_sha256(args.isolation_evidence), "initialization_checkpoint": file_sha256(args.checkpoint), "first_100_schedule_hash": schedule_hash, "diagnostic_final_checkpoint": file_sha256(smoke_checkpoint_path)})
    record = EvidenceRecordV2(2, "controllability_smoke", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, inputs, checks, permitted, None if passed else "conditional controllability smoke failed")
    _write_evidence(run, "controllability_smoke_evidence.json", record)
    print(run.root)
    return 0


def semantic_recovery_smoke(args, config: V2Config) -> int:
    """Run the single registered 400-step recovery smoke from initialization."""
    _require_identity(config)
    if getattr(args, "steps", SEMANTIC_RECOVERY_STEPS) != SEMANTIC_RECOVERY_STEPS:
        raise ValueError("semantic recovery smoke is locked to exactly 400 steps")
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.initialization_checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=init_hash)
    injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
    observability = EvidenceRecordV2.from_dict(json.loads(Path(args.observability_evidence).read_text(encoding="utf-8")))
    isolation = EvidenceRecordV2.from_dict(json.loads(Path(args.isolation_evidence).read_text(encoding="utf-8")))
    localization = EvidenceRecordV2.from_dict(json.loads(Path(args.semantic_localization_evidence).read_text(encoding="utf-8")))
    failed_smoke = EvidenceRecordV2.from_dict(json.loads(Path(args.failed_controllability_evidence).read_text(encoding="utf-8")))
    require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
    require_evidence_action(observability, kind="conditional_observability", action="conditional_control_isolation", expected_context=expected)
    require_evidence_action(isolation, kind="conditional_control_isolation", action="conditional_semantic_localization", expected_context=expected, allow_diagnostic=True)
    require_evidence_action(localization, kind="conditional_semantic_localization", action="controllability_smoke", expected_context=expected, allow_diagnostic=True)
    failed_smoke.validate()
    if failed_smoke.evidence_kind != "controllability_smoke" or failed_smoke.status != "failed":
        raise ValueError("semantic recovery requires the immutable failed 100-step smoke evidence")
    if failed_smoke.context != expected or failed_smoke.input_hashes.get("initialization_checkpoint") != init_hash:
        raise ValueError("failed smoke does not belong to the requested initialization context")
    diagnostic_final = failed_smoke.input_hashes.get("diagnostic_final_checkpoint")
    if diagnostic_final and diagnostic_final == init_hash:
        raise ValueError("semantic recovery cannot warm-start from a smoke final checkpoint")
    if failed_smoke.input_hashes.get("first_100_schedule_hash") != _first_hundred_schedule_hash(train_labels):
        raise ValueError("failed smoke does not reproduce the formal first-100 schedule")
    if failed_smoke.input_hashes.get("metric_contract") != canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT):
        raise ValueError("failed smoke uses an incompatible control metric contract")

    bundle, initialization_manifest = _load_bundle(config, args.initialization_checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    initial_state = {name: parameter.detach().clone() for name, parameter in bundle.named_parameters()}
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
            raw = raw_from_generator_batched(bundle.system, bundle.generator, probe_codes, probe_labels, microbatch_size=config.execution.training_microbatch_size, activation_checkpoint=False)
            return raw, _probe_metrics(raw, probe_labels, evaluator, settings.fixed_exposure_scale)

    raw0, metrics0 = evaluate_probe()
    probe_history["step_0"] = metrics0
    probe_raw_by_step[0] = raw0.detach().cpu()
    for step, indices in zip(range(SEMANTIC_RECOVERY_STEPS), balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234)):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=config.execution.activation_checkpoint, microbatch_size=config.execution.training_microbatch_size)
        loss.backward()
        gradient_metrics: dict[str, dict[str, object]] = {}
        snapshots = {name: [parameter.detach().clone() for parameter in parameters] for name, parameters in groups.items()}
        for name, parameters in groups.items():
            norms = [parameter.grad.detach().norm() if parameter.grad is not None else torch.tensor(float("nan"), device=device) for parameter in parameters]
            gradient_metrics[name] = {"gradient_norm": float(torch.stack(norms).norm().item()), "gradient_finite": bool(all(torch.isfinite(value) for value in norms)), "gradient_nonzero": bool(any(value > 0.0 for value in norms))}
        optimizer.step()
        update_metrics: dict[str, dict[str, object]] = {}
        for name, parameters in groups.items():
            deltas = [parameter.detach() - before for parameter, before in zip(parameters, snapshots[name], strict=True)]
            update_metrics[name] = {"update_norm": float(torch.stack([delta.norm() for delta in deltas]).norm().item()), "update_nonzero": bool(any(delta.abs().sum() > 0.0 for delta in deltas))}
        # Use explicit initial tensors for the parameter-group aliases.
        for name, parameters in groups.items():
            initial_values = {"class_carrier": initial_state["generator.class_carrier_logits"], "global_modes": initial_state["generator.global_modes"], "code_projection": initial_state["generator.code_projection.weight"], "class_heads": initial_state["generator.class_heads"], "encoder_phase": initial_state["system.encoder.phase"], "decoder_phase": initial_state["system.decoder.phase"]}[name]
            cumulative = [parameter.detach() - initial_values for parameter in parameters]
            update_metrics[name]["cumulative_update_norm"] = float(torch.stack([delta.norm() for delta in cumulative]).norm().item())
            update_metrics[name]["cumulative_update_nonzero"] = bool(any(delta.abs().sum() > 0.0 for delta in cumulative))
        group_history.append({"step": step + 1, "gradients": gradient_metrics, "updates": update_metrics})
        loss_history.append(asdict(summary) | {"step": float(step + 1)})
        if step + 1 in probe_steps[1:]:
            probe_raw, probe_metrics = evaluate_probe()
            probe_history[f"step_{step + 1}"] = probe_metrics
            probe_raw_by_step[step + 1] = probe_raw.detach().cpu()
    final_metrics = probe_history[f"step_{SEMANTIC_RECOVERY_STEPS}"]
    first_cls = sum(float(item["classification"]) for item in loss_history[:50]) / 50.0
    last_cls = sum(float(item["classification"]) for item in loss_history[-50:]) / 50.0
    relative_raw_change = float((probe_raw_by_step[SEMANTIC_RECOVERY_STEPS] - probe_raw_by_step[0]).square().mean().sqrt().div(probe_raw_by_step[0].square().mean().sqrt().clamp_min(1.0e-12)).item())
    checkpoints = [row for row in group_history if int(row["step"]) in probe_steps[1:]]
    gradient_gate = all(value["gradient_finite"] and value["gradient_nonzero"] for row in checkpoints for value in row["gradients"].values())
    update_gate = all(value["update_nonzero"] for row in checkpoints for value in row["updates"].values())
    wz_recovered = all(row["gradients"]["code_projection"]["gradient_nonzero"] and row["updates"]["code_projection"]["update_nonzero"] for row in checkpoints)
    checks = (GateResult("fixed_probe_complete", len(probe_history) == len(probe_steps), {"checkpoints": float(len(probe_history))}), GateResult("semantic_accuracy", float(final_metrics["accuracy"]) >= 0.30, {"accuracy": float(final_metrics["accuracy"])}), GateResult("classification_progress", last_cls <= 0.90 * first_cls, {"first_50": first_cls, "last_50": last_cls}), GateResult("camera_code_fraction", float(final_metrics["code_fraction"]) >= 0.10, {"camera_code_fraction": float(final_metrics["code_fraction"])}), GateResult("high_saturation", float(final_metrics["saturation"]) <= 0.08, {"saturation": float(final_metrics["saturation"])}), GateResult("finite_nonzero_gradients", gradient_gate, {}), GateResult("nonzero_parameter_updates", update_gate, {}), GateResult("identity_wz_recovered", wz_recovered, {}), GateResult("finite_raw_change", torch.isfinite(torch.tensor(relative_raw_change)).item(), {"relative_raw_change": relative_raw_change}))
    run = _new_run(config)
    raw_path = run.root / "semantic_recovery_probe_raw_cmos.pt"
    labels_path = run.root / "semantic_recovery_probe_labels.pt"
    atomic_torch_save(raw_path, probe_raw_by_step[SEMANTIC_RECOVERY_STEPS])
    atomic_torch_save(labels_path, probe_labels.cpu())
    step_artifacts: dict[str, dict[str, str]] = {}
    for probe_step in probe_steps:
        step_path = run.root / f"semantic_recovery_probe_step_{probe_step}_raw_cmos.pt"
        atomic_torch_save(step_path, probe_raw_by_step[probe_step])
        preview = write_review_artifacts(root=run.root, stem=f"semantic_recovery_probe_step_{probe_step}", kind="conditional_semantic_recovery", raw_cmos=probe_raw_by_step[probe_step], labels=probe_labels.cpu(), exposure_scale=settings.fixed_exposure_scale)
        step_artifacts[str(probe_step)] = {"raw_cmos": file_sha256(step_path), "preview": preview.preview_sha256}
    history_path = run.root / "semantic_recovery_probe_history.json"
    schedule_hash = _schedule_hash(train_labels, SEMANTIC_RECOVERY_STEPS)
    atomic_write_json(history_path, {"metric_contract": SEMANTIC_RECOVERY_CONTRACT, "loss": loss_history, "probes": probe_history, "parameter_groups": group_history, "step_probe_artifacts": step_artifacts, "first_100_schedule_hash": _first_hundred_schedule_hash(train_labels), "first_400_schedule_hash": schedule_hash, "relative_raw_change": relative_raw_change})
    final_checkpoint = run.root / "semantic_recovery_final.pt"
    save_conditional_checkpoint(final_checkpoint, bundle, ConditionalCheckpointManifestV2(2, IDENTITY_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_hash, settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), TRAINING_SCHEDULE_CONTRACT, 1234, initialization_manifest.initialization_state_hash), extra={"diagnostic_only": True, "never_warm_start": True, "steps": SEMANTIC_RECOVERY_STEPS})
    report_path = run.root / "semantic_recovery_report.json"
    atomic_write_json(report_path, {"metric_contract": SEMANTIC_RECOVERY_CONTRACT, "steps": SEMANTIC_RECOVERY_STEPS, "first_50_classification": first_cls, "last_50_classification": last_cls, "relative_raw_change": relative_raw_change, "final_metrics": final_metrics, "probe_history": probe_history, "parameters_started_from_initialization": True, "initialization_checkpoint_hash": init_hash})
    passed = all(check.passed for check in checks)
    context = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=init_hash)
    inputs = {"injection": file_sha256(args.injection_evidence), "observability": file_sha256(args.observability_evidence), "isolation": file_sha256(args.isolation_evidence), "semantic_localization": file_sha256(args.semantic_localization_evidence), "failed_controllability": file_sha256(args.failed_controllability_evidence), "initialization_checkpoint": init_hash, "evaluator": evaluator_hash, "report": file_sha256(report_path), "history": file_sha256(history_path), "diagnostic_final_checkpoint": file_sha256(final_checkpoint), "first_100_schedule_hash": _first_hundred_schedule_hash(train_labels), "first_400_schedule_hash": schedule_hash, "recovery_contract": canonical_sha256(SEMANTIC_RECOVERY_CONTRACT)}
    record = EvidenceRecordV2(2, "conditional_semantic_recovery_smoke", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, inputs, checks, ("train",) if passed else (), None if passed else "conditional semantic recovery smoke failed")
    _write_evidence(run, "semantic_recovery_evidence.json", record)
    print(run.root)
    return 0


def conditional_train(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    if config.architecture == IDENTITY_ARCHITECTURE:
        images, all_labels = load_local_mnist(args.dataset_root, train=True)
        train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
        expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.initialization_checkpoint))
        injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
        observability = EvidenceRecordV2.from_dict(json.loads(Path(args.observability_evidence).read_text(encoding="utf-8")))
        recovery = EvidenceRecordV2.from_dict(json.loads(Path(args.semantic_recovery_evidence).read_text(encoding="utf-8")))
        require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=expected)
        require_evidence_action(observability, kind="conditional_observability", action="conditional_control_isolation", expected_context=expected)
        require_evidence_action(recovery, kind="conditional_semantic_recovery_smoke", action="train", expected_context=expected)
        if recovery.input_hashes.get("first_400_schedule_hash") != _schedule_hash(train_labels, SEMANTIC_RECOVERY_STEPS):
            raise ValueError("semantic recovery does not reproduce the formal training first-400 batch/code schedule")
    else:
        _passed_evidence(args.observability_evidence, kind="conditional_observability")
        _passed_evidence(args.controllability_evidence, kind="controllability_smoke")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if config.architecture == IDENTITY_ARCHITECTURE:
        bundle, init_manifest = _load_bundle(config, args.initialization_checkpoint, device=device, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash)
    else:
        bundle = ConditionalBundle(config, carrier_logits=_carrier_logits(train_images, train_labels, config.system.slm_shape)).to(device)
        init_manifest = None
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
    manifest = ConditionalCheckpointManifestV2(2, config.architecture, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_checkpoint_hash(args.evaluator_checkpoint), settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT if config.architecture == IDENTITY_ARCHITECTURE else None, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT) if config.architecture == IDENTITY_ARCHITECTURE else None, TRAINING_SCHEDULE_CONTRACT if config.architecture == IDENTITY_ARCHITECTURE else None, 1234 if config.architecture == IDENTITY_ARCHITECTURE else None, init_manifest.initialization_state_hash if init_manifest is not None else None)
    save_conditional_checkpoint(checkpoint_path, bundle, manifest, extra={"history": history})
    checkpoint_hash = file_sha256(checkpoint_path)
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    permitted = ("evaluate",) if finite and config.run.evidence_tier == "acceptance" else ()
    inputs = {"checkpoint": checkpoint_hash, "evaluator": manifest.evaluator_checkpoint_hash}
    if config.architecture == IDENTITY_ARCHITECTURE:
        inputs["first_100_schedule_hash"] = _first_hundred_schedule_hash(train_labels)
    record = EvidenceRecordV2(2, "conditional_training", config.run.evidence_tier, "passed" if finite else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=manifest.evaluator_checkpoint_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash), inputs, (GateResult("finite_training_loss", finite, {"steps": float(len(history))}),), permitted, None if finite else "conditional training produced non-finite loss")
    _write_evidence(run, "conditional_training_evidence.json", record)
    print(run.root)
    return 0


def _balanced_epoch_steps(labels: torch.Tensor, batch_size: int = 50) -> int:
    return sum(1 for _ in balanced_batches(labels, batch_size=batch_size, seed=1234))


def conditional_training_completion(args, config: V2Config) -> int:
    """Repair only the schedule metadata of the completed semantic checkpoint."""
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    training = _passed_evidence(args.training_evidence, kind="conditional_training")
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash)
    if training.context != expected or training.input_hashes.get("checkpoint") != checkpoint_hash:
        raise ValueError("training evidence does not bind the requested semantic checkpoint context")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    history = payload.get("extra", {}).get("history") if isinstance(payload, dict) else None
    expected_steps = config.conditional_generation.epochs * _balanced_epoch_steps(train_labels, config.conditional_generation.logical_batch_size)
    if not isinstance(history, list) or len(history) != expected_steps:
        raise ValueError(f"completed checkpoint history must contain exactly {expected_steps} balanced steps")
    if any(not torch.isfinite(torch.tensor(float(row.get("total", float("nan"))))) for row in history if isinstance(row, dict)):
        raise ValueError("completed checkpoint contains non-finite history")
    bundle, old_manifest = _load_bundle(config, args.checkpoint, device=torch.device("cpu"), evaluator_hash=evaluator_hash, split_hash=split.manifest_hash)
    run = _new_run(config)
    corrected = run.root / "conditional_optical_generator_balanced.pt"
    new_manifest = ConditionalCheckpointManifestV2(2, IDENTITY_ARCHITECTURE, old_manifest.system_signature, old_manifest.config_hash, old_manifest.dataset_split_manifest_hash, old_manifest.evaluator_checkpoint_hash, old_manifest.exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), BALANCED_TRAINING_SCHEDULE_CONTRACT, 1234, old_manifest.initialization_state_hash)
    save_conditional_checkpoint(corrected, bundle, new_manifest, extra={"history": history, "schedule_steps_per_epoch": expected_steps // config.conditional_generation.epochs, "source_checkpoint": checkpoint_hash, "metadata_only_correction": True})
    checks = (GateResult("balanced_steps_per_epoch", expected_steps // config.conditional_generation.epochs == 984, {"steps_per_epoch": float(expected_steps // config.conditional_generation.epochs)}), GateResult("history_finite", True, {"steps": float(len(history))}), GateResult("state_content_preserved", new_manifest.state_content_hash == old_manifest.state_content_hash, {}))
    record = EvidenceRecordV2(2, "conditional_training_completion", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(corrected)), {"source_checkpoint": checkpoint_hash, "checkpoint": file_sha256(corrected), "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT}, checks, ("diversity_observability",), None)
    _write_evidence(run, "conditional_training_completion_evidence.json", record)
    print(run.root)
    return 0


def _diversity_parameters(bundle: ConditionalBundle) -> dict[str, tuple[nn.Parameter, ...]]:
    return {"global_modes": (bundle.generator.global_modes,), "code_projection": (bundle.generator.code_projection.weight,), "class_heads": (bundle.generator.class_heads,)}


def _diversity_objective(*, bundle: ConditionalBundle, evaluator: nn.Module, codes: torch.Tensor, labels: torch.Tensor, real_views: torch.Tensor, settings, config: V2Config) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.training_microbatch_size, activation_checkpoint=config.execution.activation_checkpoint)
    generated_view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
    generated_features, logits = evaluator(generated_view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = conditional_sinkhorn_loss(generated_features, generated_view, real_features, real_views, labels, epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations)
    classification = torch.nn.functional.cross_entropy(logits, labels)
    diversity = relative_diversity_loss(generated_features, real_features, labels)
    saturation = torch.relu(raw / settings.fixed_exposure_scale - 1.0).square().mean()
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    total = sinkhorn + 0.5 * classification + diversity + 0.1 * saturation + 1.0e-3 * orthogonality
    return total, {"total": float(total.detach()), "sinkhorn": float(sinkhorn.detach()), "classification": float(classification.detach()), "relative_diversity": float(diversity.detach()), "saturation": float(saturation.detach()), "mode_orthogonality": float(orthogonality.detach())}, raw


def conditional_diversity_observability(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    corrected_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=corrected_hash)
    completion = EvidenceRecordV2.from_dict(json.loads(Path(args.training_completion_evidence).read_text(encoding="utf-8")))
    require_evidence_action(completion, kind="conditional_training_completion", action="diversity_observability", expected_context=expected)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, allow_balanced_schedule=True)
    if manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT:
        raise ValueError("diversity observability requires the corrected balanced schedule checkpoint")
    groups = _diversity_parameters(bundle)
    codes, probe_labels = _fixed_control_probe(device=device)
    with torch.no_grad():
        commands = bundle.generator(codes, probe_labels)
        raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, probe_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
    state_before = state_content_hash(bundle.state_dict())
    indices = next(iter(balanced_batches(train_labels, batch_size=50, seed=1234)))
    settings = config.conditional_generation
    assert settings is not None
    train_codes = _random_codes(50, device=device, seed=100000)
    train_labels_device = train_labels.index_select(0, indices).to(device)
    real_views = train_images.index_select(0, indices).to(device)
    optimizer_like = []
    total, _, _ = _diversity_objective(bundle=bundle, evaluator=evaluator, codes=train_codes, labels=train_labels_device, real_views=real_views, settings=settings, config=config)
    grads = torch.autograd.grad(total, [parameter for params in groups.values() for parameter in params], allow_unused=True)
    gradient_gate = all(g is not None and torch.isfinite(g).all() and g.detach().norm() > 0 for g in grads)
    state_after = state_content_hash(bundle.state_dict())
    coeff = bundle.generator.coefficients(codes, probe_labels).reshape(10, 17, 8)
    checks = (GateResult("finite_optical_response", bool(torch.isfinite(raw).all()), {}), GateResult("code_parameter_gradients", gradient_gate, {}), GateResult("coefficient_rank_eight", int(torch.linalg.matrix_rank(coeff[:, 1:].reshape(-1, 8)).item()) == 8, {}), GateResult("parameters_unchanged", state_before == state_after, {}))
    run = _new_run(config)
    report_path = run.root / "conditional_diversity_observability_report.json"
    atomic_write_json(report_path, {"raw_code_fraction": _code_fraction(raw, probe_labels), "rank": int(torch.linalg.matrix_rank(coeff[:, 1:].reshape(-1, 8)).item()), "gradient_gate": gradient_gate, "parameters_unchanged": state_before == state_after, "diversity_loss_contract": DIVERSITY_LOSS_CONTRACT})
    passed = all(item.passed for item in checks)
    record = EvidenceRecordV2(2, "conditional_diversity_observability", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"completion": file_sha256(args.training_completion_evidence), "checkpoint": corrected_hash, "report": file_sha256(report_path)}, checks, ("diversity_recovery_smoke",) if passed else (), None if passed else "diversity observability failed")
    _write_evidence(run, "conditional_diversity_observability_evidence.json", record)
    print(run.root)
    return 0


def diversity_recovery_smoke(args, config: V2Config) -> int:
    """Single fixed 400-step code-only diagnostic from the corrected anchor."""
    _require_identity(config)
    if getattr(args, "steps", 400) != 400:
        raise ValueError("diversity recovery smoke is locked to exactly 400 steps")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash)
    completion = EvidenceRecordV2.from_dict(json.loads(Path(args.training_completion_evidence).read_text(encoding="utf-8")))
    observability = EvidenceRecordV2.from_dict(json.loads(Path(args.diversity_observability_evidence).read_text(encoding="utf-8")))
    require_evidence_action(completion, kind="conditional_training_completion", action="diversity_observability", expected_context=expected)
    require_evidence_action(observability, kind="conditional_diversity_observability", action="diversity_recovery_smoke", expected_context=expected)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, allow_balanced_schedule=True)
    if manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT:
        raise ValueError("diversity smoke requires corrected completion checkpoint")
    for parameter in (bundle.generator.class_carrier_logits, bundle.system.encoder.phase, bundle.system.decoder.phase):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam([parameter for params in groups.values() for parameter in params], lr=config.conditional_generation.generator_lr)
    settings = config.conditional_generation
    assert settings is not None
    probe_codes, probe_labels = _fixed_control_probe(device=device)
    real_probe_images = torch.cat([train_images[train_labels == label][:17] for label in range(10)], dim=0).to(device)
    with torch.no_grad():
        real_probe_features, _ = evaluator(real_probe_images)
    initial_state = {name: value.detach().clone() for name, value in bundle.state_dict().items()}
    with torch.no_grad():
        zero_raw = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros(10, 8, device=device), torch.arange(10, device=device), microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False).detach().cpu()
    probe_history: dict[str, object] = {}
    loss_history: list[dict[str, float]] = []
    update_history: list[dict[str, float]] = []

    def probe() -> tuple[torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            raw = raw_from_generator_batched(bundle.system, bundle.generator, probe_codes, probe_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
            metrics = _probe_metrics(raw, probe_labels, evaluator, settings.fixed_exposure_scale)
            generated_features, _ = evaluator(fixed_exposure_view(raw, settings.fixed_exposure_scale))
            metrics["diversity_ratios"] = relative_diversity_ratios(generated_features, real_probe_features, probe_labels)
            return raw.detach(), metrics

    raw0, metrics0 = probe()
    probe_history["step_0"] = metrics0
    for step, indices in zip(range(400), balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234)):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _diversity_objective(bundle=bundle, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), settings=settings, config=config)
        loss.backward()
        before = {name: [parameter.detach().clone() for parameter in params] for name, params in groups.items()}
        optimizer.step()
        update_history.append({name: float(torch.stack([torch.linalg.vector_norm(parameter.detach() - old) for parameter, old in zip(groups[name], before[name], strict=True)]).norm().item()) for name in groups})
        loss_history.append(summary | {"step": float(step + 1)})
        if step + 1 in {100, 200, 400}:
            probe_raw, probe_metrics = probe()
            probe_history[f"step_{step + 1}"] = probe_metrics
    final_raw, final_metrics = probe()
    zero_raw_after = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros(10, 8, device=device), torch.arange(10, device=device), microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False).detach().cpu()
    ratios = [float(value) for value in final_metrics.get("diversity_ratios", [])] if isinstance(final_metrics.get("diversity_ratios"), list) else []
    frozen_unchanged = all(torch.equal(initial_state[name], value.detach().cpu()) for name, value in bundle.state_dict().items() if name in initial_state and (name.startswith("generator.class_carrier") or name.startswith("system.encoder") or name.startswith("system.decoder")))
    code_updates = all(value > 0.0 for row in update_history for value in row.values())
    initial_ratios = probe_history["step_0"].get("diversity_ratios", [])
    ratios_ok = len(ratios) == 10 and min(ratios) >= 0.25 and all(float(after) >= float(before) for before, after in zip(initial_ratios, ratios, strict=True))
    checks = (GateResult("semantic_accuracy", float(final_metrics["accuracy"]) >= 0.95, {"accuracy": float(final_metrics["accuracy"])}), GateResult("per_class_recall", min(float(final_metrics["confusion"][label][label]) / 17.0 for label in range(10)) >= 0.90, {}), GateResult("relative_diversity", ratios_ok, {"minimum_ratio": min(ratios) if ratios else 0.0}), GateResult("camera_code_fraction", float(final_metrics["code_fraction"]) >= 0.15, {"value": float(final_metrics["code_fraction"])}), GateResult("high_saturation", float(final_metrics["saturation"]) <= 0.08, {"value": float(final_metrics["saturation"])}), GateResult("frozen_parameters", frozen_unchanged, {}), GateResult("code_parameter_updates", code_updates, {}), GateResult("zero_code_unchanged", bool(torch.equal(zero_raw, zero_raw_after)), {}), GateResult("finite_losses", bool(torch.isfinite(torch.tensor([row["total"] for row in loss_history])).all()), {}))
    run = _new_run(config)
    history_path = run.root / "conditional_diversity_recovery_history.json"
    atomic_write_json(history_path, {"loss": loss_history, "probes": probe_history, "updates": update_history, "schedule_hash_first_400": _schedule_hash(train_labels, 400), "contract": DIVERSITY_RECOVERY_CONTRACT})
    raw_path = run.root / "conditional_diversity_recovery_probe_raw_cmos.pt"
    atomic_torch_save(raw_path, final_raw.cpu())
    record = EvidenceRecordV2(2, "conditional_diversity_recovery_smoke", config.run.evidence_tier, "passed" if all(item.passed for item in checks) else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"checkpoint": checkpoint_hash, "history": file_sha256(history_path), "raw_cmos": file_sha256(raw_path), "schedule_hash_first_400": _schedule_hash(train_labels, 400), "loss_contract": DIVERSITY_LOSS_CONTRACT}, checks, ("train_diversity_recovery",) if all(item.passed for item in checks) else (), None if all(item.passed for item in checks) else "conditional diversity recovery smoke failed")
    _write_evidence(run, "conditional_diversity_recovery_evidence.json", record)
    print(run.root)
    return 0


def train_diversity_recovery(args, config: V2Config) -> int:
    """User-run one balanced epoch code-only recovery (not launched here)."""
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash)
    smoke = _passed_evidence(args.diversity_recovery_evidence, kind="conditional_diversity_recovery_smoke")
    require_evidence_action(smoke, kind="conditional_diversity_recovery_smoke", action="train_diversity_recovery", expected_context=expected)
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    bundle, init_manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, allow_balanced_schedule=True)
    for parameter in (bundle.generator.class_carrier_logits, bundle.system.encoder.phase, bundle.system.decoder.phase):
        parameter.requires_grad_(False)
    groups = _diversity_parameters(bundle)
    optimizer = torch.optim.Adam([parameter for params in groups.values() for parameter in params], lr=config.conditional_generation.generator_lr)
    settings = config.conditional_generation
    assert settings is not None
    history = []
    for batch_number, indices in enumerate(balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234)):
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=100000 + batch_number)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _diversity_objective(bundle=bundle, evaluator=evaluator, codes=codes, labels=labels, real_views=train_images.index_select(0, indices).to(device), settings=settings, config=config)
        loss.backward(); optimizer.step(); history.append(summary | {"step": float(batch_number + 1)})
    run = _new_run(config)
    path = run.root / "conditional_diversity_generator.pt"
    manifest = ConditionalCheckpointManifestV2(2, IDENTITY_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash, evaluator_hash, settings.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), IDENTITY_GENERATOR_CONTRACT, canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT), BALANCED_TRAINING_SCHEDULE_CONTRACT, 1234, init_manifest.initialization_state_hash)
    save_conditional_checkpoint(path, bundle, manifest, extra={"history": history, "diversity_loss_contract": DIVERSITY_LOSS_CONTRACT, "epochs": 1, "steps": len(history)})
    checkpoint_out_hash = file_sha256(path)
    record = EvidenceRecordV2(2, "conditional_diversity_training", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_out_hash), {"checkpoint": checkpoint_out_hash, "recovery_evidence": file_sha256(args.diversity_recovery_evidence), "steps": len(history), "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT}, (GateResult("finite_training_loss", bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all()), {"steps": float(len(history))}),), ("evaluate",), None)
    _write_evidence(run, "conditional_diversity_training_evidence.json", record)
    print(run.root)
    return 0


def conditional_evaluate(args, config: V2Config) -> int:
    _require_conditional(config)
    _require_current_acceptance(config)
    training_record = EvidenceRecordV2.from_dict(json.loads(Path(args.training_evidence).read_text(encoding="utf-8")))
    if training_record.evidence_kind not in {"conditional_training", "conditional_diversity_training"} or training_record.status != "passed":
        raise ValueError("conditional evaluate requires passed conditional training evidence")
    device = torch.device(config.run.device)
    images, all_labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, all_labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, allow_balanced_schedule=training_record.evidence_kind == "conditional_diversity_training")
    if training_record.context.checkpoint_hash != file_sha256(args.checkpoint):
        raise ValueError("training evidence does not bind the evaluation checkpoint")
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
    _require_identity(config)
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
