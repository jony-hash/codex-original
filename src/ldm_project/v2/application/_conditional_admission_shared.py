# ruff: noqa: F401,F821
"""Architecture-neutral conditional admission implementation.

The lifecycle modules bind their own runtime namespace through
``ConditionalRuntimeBindings``.  This keeps the formal and diagnostic import
closures separate while retaining one implementation for identical workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Callable, Mapping


@dataclass(frozen=True)
class ConditionalRuntimeBindings:
    """Explicit runtime symbols used by the neutral admission workflows."""

    namespace: Mapping[str, object]


def bind_admission(bindings: ConditionalRuntimeBindings) -> dict[str, Callable]:
    """Return admission functions bound to one lifecycle runtime namespace."""
    names = _EXPORTED_FUNCTIONS
    namespace = dict(globals())
    namespace.update(bindings.namespace)
    # Runtime symbols are dependencies, not the owner of these functions.
    # Preserve the shared module identity for tracebacks and introspection.
    namespace["__name__"] = __name__
    namespace["__file__"] = __file__
    bound = {
        name: FunctionType(globals()[name].__code__, namespace, name, globals()[name].__defaults__, globals()[name].__closure__)
        for name in names
    }
    namespace.update(bound)
    return bound

def train_evaluator(args, config: V2Config) -> int:
    _require_conditional(config)
    device = torch.device(config.run.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(
        images, labels, args.dataset_root
    )
    model = FrozenMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(
            train_images.shape[0],
            generator=torch.Generator(device="cpu").manual_seed(1234 + epoch),
        )
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : min(start + args.batch_size, order.numel())]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(train_images.index_select(0, indices).to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, train_labels.index_select(0, indices).to(device)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, logits = model(validation_images.to(device))
        accuracy = float(
            (logits.argmax(dim=1) == validation_labels.to(device))
            .to(torch.float32)
            .mean()
            .item()
        )
    run = _new_run(config)
    checkpoint_path = run.root / "frozen_mnist_evaluator.pt"
    save_evaluator_checkpoint(
        checkpoint_path, model.freeze(), split, validation_accuracy=accuracy
    )
    record = EvidenceRecordV2(
        2,
        "evaluator_qualification",
        config.run.evidence_tier,
        "passed" if accuracy >= 0.985 else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=file_sha256(checkpoint_path),
            split_hash=split.manifest_hash,
        ),
        {"evaluator_checkpoint": file_sha256(checkpoint_path)},
        (
            GateResult(
                "validation_accuracy",
                accuracy >= 0.985,
                {"validation_accuracy": accuracy},
            ),
        ),
        ("conditional_observability",) if accuracy >= 0.985 else (),
        None
        if accuracy >= 0.985
        else "frozen evaluator did not reach 98.5% validation accuracy",
    )
    _write_evidence(run, "evaluator_qualification_evidence.json", record)
    atomic_write_json(
        run.root / "mnist_split_manifest.json",
        asdict(split) | {"manifest_hash": split.manifest_hash},
    )
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
    _, _, validation_images, validation_labels, current_split = _split(
        images, labels, args.dataset_root
    )
    if split.manifest_hash != current_split.manifest_hash:
        raise ValueError(
            "evaluator checkpoint split does not match the local content-bound MNIST split"
        )
    with torch.no_grad():
        _, logits = evaluator(validation_images.to(device))
    accuracy = float(
        (logits.argmax(dim=1) == validation_labels.to(device))
        .to(torch.float32)
        .mean()
        .item()
    )
    run = _new_run(config)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    passed = accuracy >= 0.985
    record = EvidenceRecordV2(
        2,
        "evaluator_qualification",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash),
        {"evaluator_checkpoint": evaluator_hash},
        (GateResult("validation_accuracy", passed, {"validation_accuracy": accuracy}),),
        ("conditional_observability",)
        if passed and config.run.evidence_tier == "acceptance"
        else (),
        None if passed else "replayed evaluator validation accuracy is below 98.5%",
    )
    _write_evidence(run, "evaluator_qualification_evidence.json", record)
    print(run.root)
    return 0


def conditional_injection_admission(args, config: V2Config) -> int:
    """CPU-only generator contract gate and immutable initialization checkpoint."""
    _require_identity(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, evaluator_payload = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device("cpu")
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    payload_split = MNISTSplitManifest(**evaluator_payload["split_manifest"])
    if payload_split.manifest_hash != split.manifest_hash:
        raise ValueError("evaluator checkpoint uses a different fixed MNIST split")
    bundle = (
        _seeded_conditional_bundle(
            config,
            carrier_logits=_carrier_logits(
                train_images, train_labels, config.system.slm_shape
            ),
        )
        .cpu()
        .eval()
    )
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
        GateResult(
            "amplitude_command_code_fraction",
            balanced_code_fraction(grouped) >= 0.10,
            {"value": balanced_code_fraction(grouped)},
        ),
        GateResult(
            "physical_slm_intensity_code_fraction",
            balanced_code_fraction(physical_grouped) >= 0.10,
            {"value": balanced_code_fraction(physical_grouped)},
        ),
        GateResult(
            "zero_code_is_carrier", bool(torch.equal(zero_command, carrier)), {}
        ),
        GateResult(
            "eight_signed_direction_rms",
            bool((direction_rms >= 0.01).all()),
            {"minimum": float(direction_rms.min().item())},
        ),
        GateResult("coefficient_rank_eight", rank == 8, {"rank": float(rank)}),
        GateResult(
            "finite_orthogonal_modes",
            bool(
                torch.isfinite(mode).all() and (gram - torch.eye(8)).abs().max() < 0.25
            ),
            {},
        ),
        GateResult(
            "deterministic_initialization",
            before == after and bool(torch.equal(command, command_again)),
            {},
        ),
    )
    run = _new_run(config)
    init_path = run.root / "conditional_initialization.pt"
    settings = config.conditional_generation
    assert settings is not None
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
        TRAINING_SCHEDULE_CONTRACT,
        config.conditional_generation.initialization_seed,
        state_content_hash(bundle.state_dict()),
    )
    save_conditional_checkpoint(
        init_path,
        bundle,
        manifest,
        extra={
            "initialization": True,
            "seed": config.conditional_generation.initialization_seed,
        },
    )
    report = {
        "architecture_kind": IDENTITY_ARCHITECTURE,
        "metric_contract": CONDITIONAL_CONTROL_METRIC_CONTRACT,
        "generator_contract_hash": IDENTITY_GENERATOR_CONTRACT,
        "command_code_fraction": float(balanced_code_fraction(grouped)),
        "physical_code_fraction": float(balanced_code_fraction(physical_grouped)),
        "direction_rms": direction_rms.tolist(),
        "coefficient_rank": rank,
        "parameters_unchanged": before == after,
        "state_hash": manifest.initialization_state_hash,
    }
    report_path = run.root / "conditional_injection_admission_report.json"
    atomic_write_json(report_path, report)
    init_path_hash = file_sha256(init_path)
    passed = all(gate.passed for gate in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_injection_admission",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_hash,
            split_hash=split.manifest_hash,
            checkpoint_hash=init_path_hash,
        ),
        {
            "checkpoint": init_path_hash,
            "initialization_checkpoint": init_path_hash,
            "initialization_seed": str(manifest.initialization_seed),
            "initialization_state_hash": manifest.initialization_state_hash,
            "report": file_sha256(report_path),
            "evaluator": evaluator_hash,
        },
        checks,
        ("conditional_observability",) if passed else (),
        None if passed else "identity generator injection admission failed",
    )
    _write_evidence(run, "conditional_injection_admission_evidence.json", record)
    print(run.root)
    return 0


def _identity_observability(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device(config.run.device)
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
    injection = EvidenceRecordV2.from_dict(
        json.loads(Path(args.injection_evidence).read_text(encoding="utf-8"))
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=init_hash,
    )
    require_evidence_action(
        injection,
        kind="conditional_injection_admission",
        action="conditional_observability",
        expected_context=expected,
    )
    device = torch.device(config.run.device)
    bundle, manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_sha256=init_hash,
    )
    codes_probe, labels_probe = _fixed_control_probe(device=device)
    codes_probe, labels_probe = codes_probe[:20], labels_probe[:20]
    before = state_content_hash(bundle.state_dict())
    result = bundle.system.forward_from_generator(
        bundle.generator, codes_probe, labels_probe, return_optical_trace=True
    )
    raw = result.raw_camera.tensor
    view = fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    with torch.enable_grad():
        _, logits = evaluator(view)
        objective = raw.mean() + logits.float().mean() * 0.0
        objective.backward()
    groups = _conditional_parameter_groups(bundle)
    gradient_checks = {
        name: bool(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and parameter.grad.detach().norm() > 0
                for parameter in params
            )
        )
        for name, params in groups.items()
    }
    after = state_content_hash(bundle.state_dict())
    operations = tuple(
        field.provenance.operation for field in result.optical_trace or ()
    )
    # At the exact identity anchor Wy=0, Wz is intentionally second-order: the
    # first derivative of Wy*SiLU(Wz*z) with respect to Wz is zero.  Record this
    # contract exception explicitly; smoke must still prove Wz updates after Wy
    # leaves its initialization state.
    gradient_gate = (
        all(
            value
            for name, value in gradient_checks.items()
            if name != "code_projection"
        )
        and not gradient_checks["code_projection"]
    )
    gradient_metrics = dict(gradient_checks)
    gradient_metrics["code_projection_identity_anchor_zero"] = True
    checks = (
        GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}),
        GateResult(
            "complete_trace",
            len(operations) >= 8 and any("identity_ols" in op for op in operations),
            {"trace_length": float(len(operations))},
        ),
        GateResult("gradient_groups", gradient_gate, gradient_metrics),
        GateResult("parameters_unchanged", before == after, {}),
    )
    run = _new_run(config)
    report_path = run.root / "conditional_observability_report.json"
    atomic_write_json(
        report_path,
        {
            "trace_operations": operations,
            "gradient_groups": gradient_checks,
            "parameters_unchanged": before == after,
            "raw_finite": bool(torch.isfinite(raw).all()),
            "checkpoint_hash": init_hash,
        },
    )
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "injection_evidence": injection.evidence_hash,
            "checkpoint": init_hash,
            "report": file_sha256(report_path),
        },
        checks,
        ("conditional_control_isolation",) if passed else (),
        None if passed else "identity optical observability failed",
    )
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
    result = bundle.system.forward_from_generator(
        bundle.generator, codes, labels, return_optical_trace=True
    )
    raw = result.raw_camera.tensor
    command_delta = (
        (bundle.generator(codes, labels)[0::2] - bundle.generator(codes, labels)[1::2])
        .abs()
        .sum(dim=(-3, -2, -1))
    )
    camera_delta = (raw[0::2] - raw[1::2]).abs().sum(dim=(-3, -2, -1))
    objective = raw.mean()
    objective.backward()
    allowed = [
        bundle.generator.class_carrier_logits,
        bundle.generator.global_modes,
        bundle.generator.code_projection.weight,
        bundle.generator.class_heads,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ]
    gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in allowed
    )
    trace_operations = tuple(
        field.provenance.operation for field in result.optical_trace or ()
    )
    after = state_content_hash(bundle.state_dict())
    checks = (
        GateResult(
            "finite_raw_cmos",
            bool(torch.isfinite(raw).all()),
            {"finite_raw_cmos": bool(torch.isfinite(raw).all())},
        ),
        GateResult(
            "global_slm_code_response",
            bool((command_delta > 0).all()),
            {"minimum_command_delta": float(command_delta.min().item())},
        ),
        GateResult(
            "cmos_code_response",
            bool((camera_delta > 1.0e-8).all()),
            {"minimum_camera_delta": float(camera_delta.min().item())},
        ),
        GateResult(
            "complete_encoder_ols_decoder_trace",
            any("identity_ols" in item for item in trace_operations),
            {"trace_length": float(len(trace_operations))},
        ),
        GateResult("finite_gradients", gradients, {"finite_gradients": gradients}),
        GateResult(
            "parameters_unchanged",
            before == after,
            {"parameters_unchanged": before == after},
        ),
    )
    run = _new_run(config)
    passed = all(check.passed for check in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config),
        {},
        checks,
        ("controllability_smoke",) if passed else (),
        None if passed else "conditional optical observability failed",
    )
    _write_evidence(run, "conditional_observability_evidence.json", record)
    print(run.root)
    return 0


def sample_conditional(args, config: V2Config) -> int:
    from .evaluation import sample_conditional as _sample_conditional
    return _sample_conditional(args, config)


def _passed_evidence(path: str, *, kind: str) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed")
    return record


def _code_fraction(values: torch.Tensor, labels: torch.Tensor) -> float:
    means = torch.stack([values[labels == label].mean(dim=0) for label in range(10)])
    residual = values - means.index_select(0, labels)
    grand_mean = values.mean(dim=0, keepdim=True)
    return float(
        (
            residual.square().sum()
            / (values - grand_mean).square().sum().clamp_min(1.0e-12)
        ).item()
    )


def _conditional_parameter_groups(
    bundle: ConditionalBundle,
) -> dict[str, tuple[nn.Parameter, ...]]:
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


def _panel_semantic_metrics(
    values: torch.Tensor,
    labels: torch.Tensor,
    evaluator: nn.Module,
    exposure_scale: float,
) -> dict[str, object]:
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
        "zero_code_accuracy": float(
            (predictions.index_select(0, zero) == labels.index_select(0, zero))
            .to(torch.float32)
            .mean()
            .item()
        ),
        "nonzero_code_accuracy": float(
            (predictions[nonzero_mask] == labels[nonzero_mask])
            .to(torch.float32)
            .mean()
            .item()
        ),
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100)
        .reshape(10, 10)
        .cpu()
        .tolist(),
    }


def _first_hundred_schedule_hash(labels: torch.Tensor) -> str:
    """Content-address the exact formal epoch-one batches/codes used by smoke."""
    return _schedule_hash(labels, 100)


def _schedule_hash(labels: torch.Tensor, count: int) -> str:
    """Content-address the exact first ``count`` formal batches and codes."""
    if count < 1:
        raise ValueError("schedule hash requires at least one batch")
    rows = []
    for batch_number, indices in zip(
        range(count), balanced_batches(labels, batch_size=50, seed=1234)
    ):
        codes = _random_codes(
            indices.numel(), device=torch.device("cpu"), seed=100000 + batch_number
        )
        rows.append(
            {
                "batch": batch_number,
                "indices": tensor_sha256(indices),
                "codes": tensor_sha256(codes),
            }
        )
    if len(rows) != count:
        raise ValueError(
            f"fixed MNIST training split cannot provide {count} formal batches"
        )
    return canonical_sha256(rows)


def _identity_isolation_classification(
    *,
    harness_valid: bool,
    command_fraction: float,
    first_collapse: str | None,
    camera_fraction: float,
    command_accuracy: float,
    camera_accuracy: float,
    saturation: float,
) -> str:
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


def _probe_metrics(
    raw: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, exposure_scale: float
) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure_scale)
    features, logits = evaluator(view)
    values = raw.reshape(10, 17, *raw.shape[1:])
    command_fraction = balanced_code_fraction(values)
    code_mean = values.mean(dim=1, keepdim=True)
    direction_rms = (
        (values - code_mean)
        .square()
        .mean(dim=tuple(range(2, values.ndim)))
        .sqrt()
        .mean(dim=0)
        .flatten()
        .tolist()
    )
    semantic = _panel_semantic_metrics(raw, labels, evaluator, exposure_scale)
    return {
        "accuracy": float(
            (logits.argmax(dim=1) == labels).to(torch.float32).mean().item()
        ),
        "confusion": torch.bincount(labels * 10 + logits.argmax(dim=1), minlength=100)
        .reshape(10, 10)
        .cpu()
        .tolist(),
        "code_fraction": command_fraction,
        "label_fraction": 1.0 - command_fraction,
        "direction_rms": [float(value) for value in direction_rms],
        "raw_rms": float(raw.square().mean().sqrt().item()),
        "view_rms": float(view.square().mean().sqrt().item()),
        "saturation": float(
            torch.relu(raw / exposure_scale - 1.0).gt(0).to(torch.float32).mean().item()
        ),
        "feature_code_fraction": balanced_code_fraction(features.reshape(10, 17, -1)),
        "zero_code_accuracy": semantic["zero_code_accuracy"],
        "nonzero_code_accuracy": semantic["nonzero_code_accuracy"],
    }


def _identity_control_isolation(args, config: V2Config) -> int:
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    _, _, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device(config.run.device)
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
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
    device = torch.device(config.run.device)
    bundle, _ = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_sha256=init_hash,
    )
    codes, probe_labels = _fixed_control_probe(device=device)
    before = state_content_hash(bundle.state_dict())
    settings = config.conditional_generation
    assert settings is not None
    command_rows, raw_rows, trace_rows = [], [], []
    with torch.no_grad():
        for start in range(0, len(codes), config.execution.evaluation_microbatch_size):
            command = bundle.generator(
                codes[start : start + config.execution.evaluation_microbatch_size],
                probe_labels[
                    start : start + config.execution.evaluation_microbatch_size
                ],
            )
            result = bundle.system.forward(
                bundle.system.command(command), return_optical_trace=True
            )
            command_rows.append(command.cpu())
            raw_rows.append(result.raw_camera.tensor.cpu())
            trace_rows.append(
                tuple(
                    ComplexField(field.tensor.cpu(), field.plane, field.provenance)
                    for field in (result.optical_trace or ())
                )
            )
    command = torch.cat(command_rows).to(device)
    raw = torch.cat(raw_rows).to(device)
    after = state_content_hash(bundle.state_dict())
    command_fraction = balanced_code_fraction(
        command.reshape(10, 17, *command.shape[1:])
    )
    physical_fraction = balanced_code_fraction(
        command.square().reshape(10, 17, *command.shape[1:])
    )
    camera_fraction = balanced_code_fraction(raw.reshape(10, 17, *raw.shape[1:]))
    command_semantics = _panel_semantic_metrics(
        command.square(), probe_labels, evaluator, settings.fixed_exposure_scale
    )
    camera_semantics = _panel_semantic_metrics(
        raw, probe_labels, evaluator, settings.fixed_exposure_scale
    )
    saturation = float(
        torch.relu(raw / settings.fixed_exposure_scale - 1.0)
        .gt(0)
        .to(torch.float32)
        .mean()
        .item()
    )
    trace = trace_rows[0]
    operations = [field.provenance.operation for field in trace]
    plane_metrics = []
    for field_index, field in enumerate(trace):
        tensors = torch.cat(
            [row[field_index].tensor.abs().square() for row in trace_rows], dim=0
        )
        plane_metrics.append(
            {
                "index": field_index,
                "operation": field.provenance.operation,
                "code_fraction": balanced_code_fraction(
                    tensors.reshape(10, 17, *tensors.shape[1:])
                ),
                "finite": bool(torch.isfinite(tensors).all()),
                "plane": field.plane.to_dict(),
                "provenance": field.provenance.to_dict(),
            }
        )
    first_collapse = next(
        (item["operation"] for item in plane_metrics if item["code_fraction"] < 0.10),
        None,
    )
    harness_valid = (
        bool(torch.isfinite(raw).all())
        and before == after
        and len(operations) >= 8
        and any("identity_ols" in operation for operation in operations)
        and all(item["finite"] for item in plane_metrics)
    )
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
    atomic_write_json(
        report_path,
        {
            "metric_contract": CONDITIONAL_CONTROL_ADMISSION_CONTRACT,
            "code_count_per_label": 17,
            "command_code_fraction": command_fraction,
            "physical_slm_code_fraction": physical_fraction,
            "camera_code_fraction": camera_fraction,
            "command_semantics": command_semantics,
            "camera_semantics": camera_semantics,
            "saturation": saturation,
            "first_collapse_plane": first_collapse,
            "classification": classification,
            "parameters_unchanged": before == after,
            "plane_metrics": plane_metrics,
            "trace_operations": operations,
        },
    )
    checks = (
        GateResult("fixed_probe_shape", len(codes) == 170, {"samples": 170.0}),
        GateResult("finite_fields", bool(torch.isfinite(raw).all()), {}),
        GateResult(
            "complete_provenance",
            len(operations) >= 8 and any("identity_ols" in op for op in operations),
            {},
        ),
        GateResult("parameters_unchanged", before == after, {}),
        GateResult(
            "command_code_response",
            command_fraction >= 0.10,
            {"value": command_fraction},
        ),
        GateResult(
            "command_semantic_readability",
            float(command_semantics["accuracy"]) >= 0.95,
            {"accuracy": float(command_semantics["accuracy"])},
        ),
        GateResult(
            "fixed_exposure_saturation", saturation <= 0.08, {"saturation": saturation}
        ),
    )
    if classification == "semantic_alignment_required":
        permitted_actions = ("conditional_semantic_localization",)
    elif classification == "controllable_upper_bound":
        permitted_actions = ("controllability_smoke",)
    else:
        permitted_actions = ()
    record = EvidenceRecordV2(
        2,
        "conditional_control_isolation",
        "diagnostic",
        "diagnostic" if harness_valid else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "injection_evidence": injection.evidence_hash,
            "observability_evidence": observability.evidence_hash,
            "checkpoint": init_hash,
            "report": file_sha256(report_path),
            "raw_cmos": file_sha256(raw_path),
            "control_admission_metric_contract": canonical_sha256(
                CONDITIONAL_CONTROL_ADMISSION_CONTRACT
            ),
        },
        checks,
        permitted_actions,
        None if harness_valid else "conditional control isolation harness failed",
    )
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
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=_carrier_logits(images, all_labels, config.system.slm_shape),
        )
        .to(device)
        .eval()
    )
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
            result = bundle.system.forward(
                bundle.system.command(command[start : start + microbatch]),
                return_optical_trace=True,
            )
            raw_rows.append(result.raw_camera.tensor.cpu())
            trace_rows.append(
                [
                    ComplexField(field.tensor.cpu(), field.plane, field.provenance)
                    for field in result.optical_trace or ()
                ]
            )
        raw = torch.cat(raw_rows, dim=0).to(device)
        trace = tuple(
            ComplexField(
                torch.cat([row[index].tensor for row in trace_rows], dim=0),
                trace_rows[0][index].plane,
                trace_rows[0][index].provenance,
            )
            for index in range(len(trace_rows[0]))
        )
        view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
        features, logits = evaluator(view)
    after = state_content_hash(bundle.state_dict())
    trace_names = (
        "slm_field",
        "pre_encoder",
        "encoder_layer_1",
        "encoder_layer_2",
        "identity_ols",
        "decoder_layer_1",
        "decoder_layer_2",
        "camera_field",
    )
    plane_metrics: list[dict[str, object]] = []
    for name, field in zip(trace_names, trace, strict=False):
        tensor = field.tensor.abs().square()
        grouped = tensor.reshape(10, 17, *tensor.shape[1:])
        fraction = balanced_code_fraction(grouped)
        plane_metrics.append(
            {
                "name": name,
                "code_fraction": fraction,
                "label_fraction": 1.0 - fraction,
                "rms": float(tensor.square().mean().sqrt().item()),
                "mean": float(tensor.mean().item()),
                "max": float(tensor.max().item()),
                "finite": bool(torch.isfinite(tensor).all()),
                "plane": field.plane.to_dict(),
                "provenance": field.provenance.to_dict(),
            }
        )
    command_grouped = command.reshape(10, 17, *command.shape[1:])
    command_fraction = balanced_code_fraction(command_grouped)
    coefficients = bundle.generator.coefficients(codes, labels).reshape(10, 17, 8)
    zero_codes = torch.zeros(10, 8, device=device)
    zero_labels = torch.arange(10, device=device)
    duplicate_error = float(
        (
            bundle.generator(zero_codes, zero_labels)
            - bundle.generator(zero_codes, zero_labels)
        )
        .abs()
        .max()
        .item()
    )
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
            "coefficient_std_by_direction": coefficients.reshape(-1, 8)
            .std(dim=0)
            .tolist(),
            "duplicate_zero_code_max_error": duplicate_error,
        },
        "camera": _probe_metrics(raw, labels, evaluator, settings.fixed_exposure_scale),
        "classification": {
            "accuracy": float(
                (logits.argmax(dim=1) == labels).to(torch.float32).mean().item()
            )
        },
        "harness": {
            "finite_command": bool(torch.isfinite(command).all()),
            "finite_raw": bool(torch.isfinite(raw).all()),
            "finite_features": bool(torch.isfinite(features).all()),
            "complete_trace": len(trace) == len(trace_names)
            and any(
                "identity_ols" in item
                for item in [field.provenance.operation for field in trace]
            ),
        },
    }
    command_path = Path(args.output) if args.output else None
    run = _new_run(config)
    if command_path is None:
        command_path = run.root / "conditional_control_isolation_raw_cmos.pt"
    atomic_torch_save(command_path, raw.cpu())
    atomic_torch_save(run.root / "conditional_control_isolation_codes.pt", codes.cpu())
    atomic_torch_save(
        run.root / "conditional_control_isolation_labels.pt", labels.cpu()
    )
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
        elif (
            not report["harness"]["complete_trace"]
            or not report["parameters_unchanged"]
            or not report["harness"]["finite_raw"]
        ):
            classification = "harness_invalid"
        elif (
            report["camera"]["feature_code_fraction"] < 0.10
            or report["classification"]["accuracy"] < 0.30
        ):
            classification = "semantic_readout_insufficient"
        else:
            classification = "controllable_upper_bound"
    report["classification"] = {
        **report["classification"],
        "status": classification,
        "first_collapse_plane": first_collapse,
    }
    atomic_write_json(report_path, report)
    checks = (
        GateResult(
            "fixed_probe_shape",
            len(codes) == 170 and len(labels) == 170,
            {"samples": float(len(labels))},
        ),
        GateResult(
            "finite_fields",
            bool(
                report["harness"]["finite_command"]
                and report["harness"]["finite_raw"]
                and report["harness"]["finite_features"]
            ),
            {},
        ),
        GateResult(
            "complete_provenance",
            bool(report["harness"]["complete_trace"]),
            {"trace_length": float(len(trace))},
        ),
        GateResult("parameters_unchanged", bool(report["parameters_unchanged"]), {}),
        GateResult(
            "command_code_response",
            command_fraction > 0.0,
            {"code_fraction": command_fraction},
        ),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_control_isolation",
        "diagnostic",
        "diagnostic" if all(item.passed for item in checks) else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(
            config,
            evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
            split_hash=split.manifest_hash,
        ),
        {
            "report": file_sha256(report_path),
            "raw_cmos": file_sha256(command_path),
            "metric_contract": canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        },
        checks,
        (),
        None
        if all(item.passed for item in checks)
        else "conditional control isolation harness failed",
    )
    _write_evidence(run, "conditional_control_isolation_evidence.json", record)
    print(run.root)
    return 0


def _gradient_summary(
    objective: torch.Tensor, groups: dict[str, tuple[nn.Parameter, ...]]
) -> tuple[dict[str, dict[str, object]], dict[str, torch.Tensor]]:
    """Collect gradients without creating an optimizer or changing parameters."""
    parameters = [parameter for values in groups.values() for parameter in values]
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    summary: dict[str, dict[str, object]] = {}
    vectors: dict[str, torch.Tensor] = {}
    offset = 0
    for name, values in groups.items():
        rows = gradients[offset : offset + len(values)]
        offset += len(values)
        finite = all(
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in rows
        )
        vectors[name] = (
            torch.cat(
                [
                    gradient.detach().flatten()
                    for gradient in rows
                    if gradient is not None
                ]
            )
            if any(gradient is not None for gradient in rows)
            else torch.zeros(1, device=objective.device)
        )
        norm = float(vectors[name].norm().item())
        summary[name] = {
            "gradient_norm": norm,
            "gradient_finite": finite,
            "gradient_nonzero": norm > 1.0e-12,
            "structurally_inactive": norm <= 1.0e-12,
        }
    return summary, vectors


def _semantic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    predictions = logits.argmax(dim=1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logits - other_logits.max(dim=1).values
    return {
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "per_label_accuracy": [
            float(
                (predictions[labels == label] == label).to(torch.float32).mean().item()
            )
            for label in range(10)
        ],
        "mean_true_margin": float(margins.mean().item()),
        "median_true_margin": float(margins.median().item()),
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100)
        .reshape(10, 10)
        .cpu()
        .tolist(),
    }


def _semantic_localization_classification(
    *,
    harness_valid: bool,
    zero_gradients: dict[str, dict[str, object]],
    formal_gradients: dict[str, dict[str, object]],
    gradient_cosines: dict[str, dict[str, object]],
) -> str:
    required_zero = ("class_carrier", "encoder_phase", "decoder_phase")
    required_formal = (
        "class_carrier",
        "global_modes",
        "class_heads",
        "encoder_phase",
        "decoder_phase",
    )
    if not harness_valid:
        return "harness_invalid"
    if any(
        float(zero_gradients[name]["gradient_norm"]) <= 1.0e-12
        for name in required_zero
    ):
        return "zero_code_gradient_absent"
    if not all(
        bool(formal_gradients[name]["gradient_nonzero"])
        and bool(formal_gradients[name]["gradient_finite"])
        for name in required_formal
    ):
        return "harness_invalid"
    usable = [
        float(gradient_cosines[name]["cosine"])
        for name in required_zero
        if gradient_cosines[name]["cosine"] is not None
    ]
    if any(value <= -0.05 for value in usable):
        return "zero_code_joint_conflict"
    if any(abs(value) < 0.05 for value in usable):
        return "indeterminate_alignment"
    return "local_semantic_alignment_plausible"


def _identity_semantic_localization(args, config: V2Config) -> int:
    """Read-only local semantic-gradient diagnostic for the identity anchor."""
    _require_identity(config)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(
        images, labels_all, args.dataset_root
    )
    device = torch.device(config.run.device)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    init_hash = file_sha256(args.checkpoint)
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
    if isolation.input_hashes.get(
        "control_admission_metric_contract"
    ) != canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT):
        raise ValueError(
            "legacy control-isolation metric contract cannot authorize semantic localization"
        )

    settings = config.conditional_generation
    assert settings is not None
    bundle, _ = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_sha256=init_hash,
    )
    bundle.eval()
    groups = _conditional_parameter_groups(bundle)
    before = state_content_hash(bundle.state_dict())
    zero_codes = torch.zeros(10, 8, device=device)
    zero_labels = torch.arange(10, device=device)
    control_codes, control_labels = _fixed_control_probe(device=device)
    first_indices = next(
        balanced_batches(
            train_labels, batch_size=settings.logical_batch_size, seed=1234
        )
    )
    formal_labels = train_labels.index_select(0, first_indices).to(device)
    formal_codes = _random_codes(formal_labels.numel(), device=device, seed=100000)
    formal_real = train_images.index_select(0, first_indices).to(device)

    with torch.enable_grad():
        zero_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            zero_codes,
            zero_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        zero_view = fixed_exposure_view(zero_raw, settings.fixed_exposure_scale)
        _, zero_logits = evaluator(zero_view)
        zero_objective = torch.nn.functional.cross_entropy(zero_logits, zero_labels)
        zero_gradients, zero_vectors = _gradient_summary(zero_objective, groups)
        formal_total, formal_summary, formal_raw = conditional_loss(
            system=bundle.system,
            generator=bundle.generator,
            evaluator=evaluator,
            codes=formal_codes,
            labels=formal_labels,
            real_views=formal_real,
            exposure_scale=settings.fixed_exposure_scale,
            sinkhorn_epsilon=settings.sinkhorn_epsilon,
            sinkhorn_iterations=settings.sinkhorn_iterations,
            activation_checkpoint=False,
            microbatch_size=config.execution.evaluation_microbatch_size,
        )
        formal_gradients, formal_vectors = _gradient_summary(formal_total, groups)

    gradient_cosines = {}
    for name in groups:
        zero_norm = float(zero_gradients[name]["gradient_norm"])
        formal_norm = float(formal_gradients[name]["gradient_norm"])
        gradient_cosines[name] = {
            "cosine": None
            if zero_norm <= 1.0e-12 or formal_norm <= 1.0e-12
            else float(
                torch.nn.functional.cosine_similarity(
                    zero_vectors[name][None], formal_vectors[name][None]
                ).item()
            ),
            "zero_code_structural": zero_norm <= 1.0e-12,
            "formal_structural": formal_norm <= 1.0e-12,
        }

    with torch.no_grad():
        control_raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            control_codes,
            control_labels,
            microbatch_size=config.execution.evaluation_microbatch_size,
            activation_checkpoint=False,
        )
        _, control_logits = evaluator(
            fixed_exposure_view(control_raw, settings.fixed_exposure_scale)
        )
        _, formal_logits = evaluator(
            fixed_exposure_view(formal_raw.detach(), settings.fixed_exposure_scale)
        )
        trace_command = bundle.generator(control_codes[:1], control_labels[:1])
        trace_result = bundle.system.forward(
            bundle.system.command(trace_command), return_optical_trace=True
        )
    after = state_content_hash(bundle.state_dict())
    operations = tuple(
        field.provenance.operation for field in trace_result.optical_trace or ()
    )
    finite_gradients = all(
        bool(value["gradient_finite"])
        for value in (*zero_gradients.values(), *formal_gradients.values())
    )
    required_formal = (
        "class_carrier",
        "global_modes",
        "class_heads",
        "encoder_phase",
        "decoder_phase",
    )
    active_formal = all(
        bool(formal_gradients[name]["gradient_nonzero"]) for name in required_formal
    )
    valid = bool(
        torch.isfinite(zero_raw).all()
        and torch.isfinite(control_raw).all()
        and torch.isfinite(formal_raw).all()
        and finite_gradients
        and active_formal
        and before == after
        and len(operations) >= 8
        and any("identity_ols" in operation for operation in operations)
    )
    classification = _semantic_localization_classification(
        harness_valid=valid,
        zero_gradients=zero_gradients,
        formal_gradients=formal_gradients,
        gradient_cosines=gradient_cosines,
    )

    run = _new_run(config)
    zero_path = run.root / "conditional_semantic_localization_zero_raw_cmos.pt"
    control_path = run.root / "conditional_semantic_localization_control_raw_cmos.pt"
    formal_path = run.root / "conditional_semantic_localization_formal_raw_cmos.pt"
    atomic_torch_save(zero_path, zero_raw.detach().cpu())
    atomic_torch_save(control_path, control_raw.detach().cpu())
    atomic_torch_save(formal_path, formal_raw.detach().cpu())
    zero_preview = write_review_artifacts(
        root=run.root,
        stem="semantic_localization_zero",
        kind="conditional_semantic_localization",
        raw_cmos=zero_raw.detach().cpu(),
        labels=zero_labels.cpu(),
        exposure_scale=settings.fixed_exposure_scale,
    )
    control_preview = write_review_artifacts(
        root=run.root,
        stem="semantic_localization_control",
        kind="conditional_semantic_localization",
        raw_cmos=control_raw.detach().cpu(),
        labels=control_labels.cpu(),
        exposure_scale=settings.fixed_exposure_scale,
    )
    report_path = run.root / "conditional_semantic_localization_report.json"
    report = {
        "metric_contract": CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT,
        "classification": classification,
        "parameters_unchanged": before == after,
        "optimizer_created": False,
        "zero_code_semantics": _semantic_metrics(zero_logits.detach(), zero_labels),
        "control_panel_semantics": _semantic_metrics(
            control_logits.detach(), control_labels
        ),
        "formal_batch_semantics": _semantic_metrics(
            formal_logits.detach(), formal_labels
        ),
        "formal_loss": asdict(formal_summary),
        "zero_code_gradients": zero_gradients,
        "formal_gradients": formal_gradients,
        "gradient_cosines": gradient_cosines,
        "control_code_fraction": balanced_code_fraction(
            control_raw.reshape(10, 17, *control_raw.shape[1:])
        ),
        "control_saturation": float(
            torch.relu(control_raw / settings.fixed_exposure_scale - 1.0)
            .gt(0)
            .to(torch.float32)
            .mean()
            .item()
        ),
        "trace_operations": operations,
        "trace_provenance": [
            field.provenance.to_dict() for field in (trace_result.optical_trace or ())
        ],
        "first_formal_batch_schedule_hash": canonical_sha256(
            {
                "indices": tensor_sha256(first_indices),
                "codes": tensor_sha256(formal_codes.cpu()),
            }
        ),
    }
    atomic_write_json(report_path, report)
    checks = (
        GateResult("finite_fields_and_gradients", valid, {"finite": finite_gradients}),
        GateResult("parameters_unchanged", before == after, {}),
        GateResult(
            "complete_trace",
            len(operations) >= 8
            and any("identity_ols" in operation for operation in operations),
            {"trace_length": float(len(operations))},
        ),
        GateResult(
            "zero_code_semantic_gradients",
            all(
                float(zero_gradients[name]["gradient_norm"]) > 1.0e-12
                for name in ("class_carrier", "encoder_phase", "decoder_phase")
            ),
            {},
        ),
        GateResult("formal_active_gradients", active_formal, {}),
    )
    permitted = (
        ("controllability_smoke",)
        if classification == "local_semantic_alignment_plausible"
        else ()
    )
    record = EvidenceRecordV2(
        2,
        "conditional_semantic_localization",
        "diagnostic",
        "diagnostic" if valid else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "injection_evidence": injection.evidence_hash,
            "observability_evidence": observability.evidence_hash,
            "isolation_evidence": isolation.evidence_hash,
            "checkpoint": init_hash,
            "report": file_sha256(report_path),
            "zero_raw_cmos": file_sha256(zero_path),
            "control_raw_cmos": file_sha256(control_path),
            "formal_raw_cmos": file_sha256(formal_path),
            "zero_preview": zero_preview.preview_sha256,
            "control_preview": control_preview.preview_sha256,
            "metric_contract": canonical_sha256(
                CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT
            ),
        },
        checks,
        permitted,
        None if valid else "semantic localization harness failed",
    )
    _write_evidence(run, "conditional_semantic_localization_evidence.json", record)
    print(run.root)
    return 0


_EXPORTED_FUNCTIONS = (
    "train_evaluator",
    "qualify_evaluator",
    "conditional_injection_admission",
    "_identity_observability",
    "conditional_observability",
    "sample_conditional",
    "_passed_evidence",
    "_code_fraction",
    "_conditional_parameter_groups",
    "_fixed_control_probe",
    "_panel_semantic_metrics",
    "_first_hundred_schedule_hash",
    "_schedule_hash",
    "_identity_isolation_classification",
    "_probe_metrics",
    "_identity_control_isolation",
    "conditional_control_isolation",
    "_gradient_summary",
    "_semantic_metrics",
    "_semantic_localization_classification",
    "_identity_semantic_localization",
)
