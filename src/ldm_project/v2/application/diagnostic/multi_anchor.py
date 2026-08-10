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
from . import evaluation as _evaluation

globals().update({name: value for name, value in _runtime.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _admission.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _recovery.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _diversity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _class_modes.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _parity.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _legacy_training.__dict__.items() if not name.startswith("__")})
globals().update({name: value for name, value in _evaluation.__dict__.items() if not name.startswith("__")})

def _multi_anchor_route_code(
    anchor: int, *, device: torch.device, local: torch.Tensor | None = None
) -> torch.Tensor:
    if anchor not in range(4):
        raise ValueError("anchor must be in [0, 3]")
    signs = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))[anchor]
    value = torch.zeros(8, device=device)
    value[0], value[1] = signs
    if local is not None:
        if local.shape != (6,):
            raise ValueError("local code must have shape [6]")
        value[2:] = local
    return value


def _multi_anchor_batch_codes(
    *, labels: torch.Tensor, batch_number: int, device: torch.device
) -> torch.Tensor:
    """Deterministic 80-sample batch: every class/anchor has an antithetic pair."""
    generator = torch.Generator(device="cpu").manual_seed(100000 + batch_number)
    local = torch.rand(10, 4, 6, generator=generator).mul_(2.0).sub_(1.0)
    rows = []
    for label in range(10):
        for anchor in range(4):
            rows.extend(
                (
                    _multi_anchor_route_code(
                        anchor, device=device, local=local[label, anchor]
                    ),
                    _multi_anchor_route_code(
                        anchor, device=device, local=-local[label, anchor]
                    ),
                )
            )
    return torch.stack(rows).to(device)


def _multi_anchor_panels(
    *, device: torch.device, uniform_seed: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    centers = torch.stack(
        [_multi_anchor_route_code(anchor, device=device) for anchor in range(4)]
    ).repeat(10, 1)
    center_labels = torch.arange(10, device=device).repeat_interleave(4)
    axis_codes, axis_labels = [], []
    for label in range(10):
        for anchor in range(4):
            axis_codes.append(_multi_anchor_route_code(anchor, device=device))
            axis_labels.append(label)
            for direction in range(6):
                local = torch.zeros(6, device=device)
                local[direction] = 1.0
                axis_codes.extend(
                    (
                        _multi_anchor_route_code(anchor, device=device, local=local),
                        _multi_anchor_route_code(anchor, device=device, local=-local),
                    )
                )
                axis_labels.extend((label, label))
    uniform_generator = torch.Generator(device="cpu").manual_seed(uniform_seed)
    local = torch.rand(10, 4, 2, 6, generator=uniform_generator).mul_(2.0).sub_(1.0)
    uniform_codes, uniform_labels = [], []
    for label in range(10):
        for anchor in range(4):
            for row in local[label, anchor]:
                uniform_codes.append(
                    _multi_anchor_route_code(anchor, device=device, local=row)
                )
                uniform_labels.append(label)
    return {
        "anchor_centers": (centers.to(device), center_labels),
        "local_axis": (
            torch.stack(axis_codes),
            torch.tensor(axis_labels, device=device),
        ),
        f"uniform_{uniform_seed}": (
            torch.stack(uniform_codes),
            torch.tensor(uniform_labels, device=device),
        ),
    }


def _multi_anchor_partition(
    features: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int = 1234,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Deterministic four-way feature clustering and its reproducible metadata."""
    if (
        features.ndim != 2
        or images.ndim != 4
        or labels.ndim != 1
        or features.shape[0] != labels.numel()
        or images.shape[0] != labels.numel()
    ):
        raise ValueError("multi-anchor partition inputs have incompatible shapes")
    assignments = torch.full((labels.numel(),), -1, dtype=torch.long)
    summary = {"seed": seed, "clusters": []}
    normalized = torch.nn.functional.normalize(features.cpu(), dim=1)
    for label in range(10):
        indices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        values = normalized.index_select(0, indices)
        mean = values.mean(dim=0, keepdim=True)
        first_local = torch.cdist(values, mean).flatten().argmin().item()
        centers = [values[first_local]]
        for _ in range(3):
            distance = (
                torch.cdist(values, torch.stack(centers)).square().min(dim=1).values
            )
            centers.append(values[distance.argmax()])
        centers_tensor = torch.stack(centers)
        for _ in range(20):
            distance = torch.cdist(values, centers_tensor).square()
            current = distance.argmin(dim=1)
            updated = []
            for cluster in range(4):
                selected = values[current == cluster]
                updated.append(
                    selected.mean(dim=0)
                    if selected.numel()
                    else centers_tensor[cluster]
                )
            new_centers = torch.stack(updated)
            if torch.equal(
                current, torch.cdist(values, new_centers).square().argmin(dim=1)
            ):
                centers_tensor = new_centers
                break
            centers_tensor = new_centers
        current = torch.cdist(values, centers_tensor).square().argmin(dim=1)
        assignments[indices] = current
        counts = [int((current == cluster).sum().item()) for cluster in range(4)]
        summary["clusters"].append(
            {"label": label, "counts": counts, "indices_hash": tensor_sha256(indices)}
        )
    return assignments, summary


def _multi_anchor_modes(
    images: torch.Tensor,
    labels: torch.Tensor,
    assignments: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors, modes = [], []
    for label in range(10):
        class_anchors, class_modes = [], []
        for cluster in range(4):
            selected = torch.nonzero(
                (labels == label) & (assignments == cluster), as_tuple=False
            ).flatten()
            mean = images.index_select(0, selected).mean(dim=0, keepdim=True)
            enlarged = torch.nn.functional.interpolate(
                mean, size=shape, mode="bilinear", align_corners=False
            )
            amplitude = enlarged.sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
            class_anchors.append(torch.logit(amplitude).squeeze(0).squeeze(0))
            flat = images.index_select(0, selected).flatten(1).to(torch.float64)
            centered = flat - flat.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            vectors = vh[:6].to(torch.float32).view(6, 28, 28)
            enlarged_modes = torch.nn.functional.interpolate(
                vectors.unsqueeze(1), size=shape, mode="bilinear", align_corners=False
            ).squeeze(1)
            # A finite SLM-bandwidth low-pass in cycles per active aperture.
            spectrum = torch.fft.fft2(enlarged_modes)
            fy = torch.fft.fftfreq(shape[0]).view(-1, 1) * shape[0]
            fx = torch.fft.fftfreq(shape[1]).view(1, -1) * shape[1]
            spectrum = spectrum * (torch.sqrt(fx.square() + fy.square()) <= 10.0)
            filtered = torch.fft.ifft2(spectrum).real
            filtered = filtered - filtered.mean(dim=(-2, -1), keepdim=True)
            basis = []
            for vector in filtered:
                vector = vector.clone()
                for previous in basis:
                    vector = vector - (vector * previous).mean() * previous
                vector = vector / vector.square().mean().clamp_min(1.0e-8).sqrt()
                basis.append(vector)
            class_modes.append(torch.stack(basis))
        anchors.append(torch.stack(class_anchors))
        modes.append(torch.stack(class_modes))
    return torch.stack(anchors), torch.stack(modes)


def _multi_anchor_source_phases(
    source_checkpoint: str, bundle: ConditionalBundle
) -> None:
    payload = safe_torch_load(source_checkpoint)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), dict)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("semantic source checkpoint is invalid")
    source_architecture = payload["manifest"].get("architecture_kind")
    if source_architecture not in {
        IDENTITY_ARCHITECTURE,
        CLASS_MODE_ARCHITECTURE,
        PARITY_ARCHITECTURE,
    }:
        raise ValueError(
            "multi-anchor initialization requires an explicit v2 semantic optical checkpoint"
        )
    state = payload["state_dict"]
    target = bundle.state_dict()
    for name in ("system.encoder.phase", "system.decoder.phase"):
        if name not in state or state[name].shape != target[name].shape:
            raise ValueError(
                f"semantic source lacks exact optical phase tensor: {name}"
            )
        target[name].copy_(state[name])
    bundle.load_state_dict(target, strict=True)


def _multi_anchor_raw_metrics(
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    exposure: float,
) -> dict[str, object]:
    with torch.no_grad():
        raw = raw_from_generator_batched(
            bundle.system,
            bundle.generator,
            codes,
            labels,
            microbatch_size=1,
            activation_checkpoint=False,
        )
        view = fixed_exposure_view(raw, exposure)
        features, logits = evaluator(view)
    predictions = logits.argmax(dim=1)
    recalls = [
        float((predictions[labels == label] == label).to(torch.float32).mean().item())
        for label in range(10)
    ]
    return {
        "raw": raw.cpu(),
        "features": features.cpu(),
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "recalls": recalls,
        "camera_fraction": float(camera_code_fraction(raw, labels).item()),
        "saturation": float(
            torch.relu(raw / exposure - 1.0).gt(0).to(torch.float32).mean().item()
        ),
    }


def conditional_multi_anchor_capacity(args, config: V2Config) -> int:
    _require_multi_anchor(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device("cpu")
    )
    with torch.no_grad():
        features, _ = evaluator(train_images)
    assignments, summary = _multi_anchor_partition(features, train_images, train_labels)
    reductions, valid = [], True
    normalized = torch.nn.functional.normalize(features, dim=1)
    for label in range(10):
        selected = torch.nonzero(train_labels == label, as_tuple=False).flatten()
        total = torch.pdist(normalized.index_select(0, selected)).median()
        within = []
        for cluster in range(4):
            member = selected[assignments[selected] == cluster]
            within.append(torch.pdist(normalized.index_select(0, member)).median())
            valid = valid and member.numel() >= 256
        within_value = torch.stack(within).mean()
        reductions.append(float((1.0 - within_value / total.clamp_min(1.0e-8)).item()))
    valid = valid and min(reductions) >= 0.15 and sum(reductions) / 10.0 >= 0.30
    run = _new_run(config)
    report = run.root / "conditional_multi_anchor_capacity_report.json"
    atomic_write_json(
        report,
        {
            "architecture_kind": MULTI_ANCHOR_ARCHITECTURE,
            "partition": summary,
            "reductions": reductions,
            "valid": valid,
            "split_hash": split.manifest_hash,
        },
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
        split_hash=split.manifest_hash,
    )
    record = EvidenceRecordV2(
        2,
        "conditional_multi_anchor_capacity",
        config.run.evidence_tier,
        "passed" if valid else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {"report": file_sha256(report)},
        (
            GateResult(
                "cluster_capacity",
                valid,
                {
                    "minimum_reduction": min(reductions),
                    "mean_reduction": sum(reductions) / 10.0,
                },
            ),
        ),
        ("conditional_multi_anchor_admission",) if valid else (),
        None if valid else "four-way class capacity was not established",
    )
    _write_evidence(run, "conditional_multi_anchor_capacity_evidence.json", record)
    print(run.root)
    return 0


def conditional_multi_anchor_admission(args, config: V2Config) -> int:
    _require_multi_anchor(config)
    capacity = _passed_evidence(
        args.capacity_evidence, kind="conditional_multi_anchor_capacity"
    )
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(
        args.evaluator_checkpoint, device=torch.device(config.run.device)
    )
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    with torch.no_grad():
        features, _ = evaluator(train_images.to(torch.device(config.run.device)))
    assignments, summary = _multi_anchor_partition(
        features.cpu(), train_images, train_labels
    )
    anchors, modes = _multi_anchor_modes(
        train_images, train_labels, assignments, config.system.slm_shape
    )
    bundle = ConditionalBundle(config, anchor_logits=anchors, local_modes=modes).to(
        torch.device(config.run.device)
    )
    _multi_anchor_source_phases(args.semantic_checkpoint, bundle)
    panels = _multi_anchor_panels(
        device=torch.device(config.run.device), uniform_seed=1234
    )
    center = _multi_anchor_raw_metrics(
        bundle,
        evaluator,
        panels["anchor_centers"][0],
        panels["anchor_centers"][1],
        device=torch.device(config.run.device),
        exposure=config.conditional_generation.fixed_exposure_scale,
    )
    before = state_content_hash(bundle.state_dict())
    passed = (
        center["accuracy"] >= 0.70
        and min(center["recalls"]) >= 0.50
        and center["camera_fraction"] >= 0.15
        and center["saturation"] <= 0.08
    )
    run = _new_run(config)
    partition_path = run.root / "anchor_partition_manifest.json"
    atomic_write_json(
        partition_path,
        {
            "architecture_kind": MULTI_ANCHOR_ARCHITECTURE,
            "split_hash": split.manifest_hash,
            "partition": summary,
            "assignment_hash": tensor_sha256(assignments),
        },
    )
    path = run.root / "conditional_multi_anchor_initialization.pt"
    manifest = ConditionalCheckpointManifestV2(
        2,
        MULTI_ANCHOR_ARCHITECTURE,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split.manifest_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        MULTI_ANCHOR_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        state_content_hash(bundle.state_dict()),
    )
    save_conditional_checkpoint(
        path,
        bundle,
        manifest,
        extra={
            "capacity_evidence": capacity.evidence_hash,
            "semantic_source_checkpoint": file_sha256(args.semantic_checkpoint),
            "partition": summary,
            "partition_manifest": file_sha256(partition_path),
            "initializer": "cluster_mean_plus_bandlimited_pca",
        },
    )
    out_hash = file_sha256(path)
    report = run.root / "conditional_multi_anchor_admission_report.json"
    atomic_write_json(
        report,
        {
            "center_metrics": {k: v for k, v in center.items() if k != "raw"},
            "partition": summary,
            "partition_manifest": file_sha256(partition_path),
            "source_phase_hash": file_sha256(args.semantic_checkpoint),
            "state_before": before,
            "state_after": state_content_hash(bundle.state_dict()),
        },
    )
    normalized = bundle.generator.normalized_modes(
        bundle.generator.local_modes
    ).flatten(-2)
    rank_six = all(
        int(torch.linalg.matrix_rank(normalized[label, anchor]).item()) == 6
        for label in range(10)
        for anchor in range(4)
    )
    checks = (
        GateResult(
            "anchor_center_semantics",
            passed,
            {"accuracy": center["accuracy"], "minimum_recall": min(center["recalls"])},
        ),
        GateResult("local_mode_rank_six", rank_six, {}),
        GateResult(
            "state_finite",
            all(torch.isfinite(v).all() for v in bundle.state_dict().values()),
            {},
        ),
        GateResult("zero_optimizer", True, {}),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_multi_anchor_admission",
        config.run.evidence_tier,
        "passed" if all(x.passed for x in checks) else "failed",
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
            "capacity": capacity.evidence_hash,
            "report": file_sha256(report),
            "partition_manifest": file_sha256(partition_path),
            "semantic_source": file_sha256(args.semantic_checkpoint),
        },
        checks,
        ("conditional_multi_anchor_observability",)
        if all(x.passed for x in checks)
        else (),
        None
        if all(x.passed for x in checks)
        else "multi-anchor initialization gate failed",
    )
    _write_evidence(run, "conditional_multi_anchor_admission_evidence.json", record)
    print(run.root)
    return 0


def conditional_multi_anchor_observability(args, config: V2Config) -> int:
    _require_multi_anchor(config)
    admission = _passed_evidence(
        args.admission_evidence, kind="conditional_multi_anchor_admission"
    )
    device = torch.device(config.run.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    bundle, manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    require_evidence_action(
        admission,
        kind="conditional_multi_anchor_admission",
        action="conditional_multi_anchor_observability",
        expected_context=expected,
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    bundle.generator.anchor_logits.requires_grad_(True)
    bundle.generator.local_modes.requires_grad_(True)
    indices = next(balanced_batches(train_labels, batch_size=80, seed=1234))
    labels_batch = train_labels.index_select(0, indices).to(device)
    codes = _multi_anchor_batch_codes(
        labels=labels_batch, batch_number=0, device=device
    )
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels_batch,
        microbatch_size=1,
        activation_checkpoint=False,
    )
    _, logits = evaluator(
        fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
    )
    loss = torch.nn.functional.cross_entropy(logits, labels_batch)
    grads = torch.autograd.grad(
        loss,
        (bundle.generator.anchor_logits, bundle.generator.local_modes),
        allow_unused=True,
    )
    before = state_content_hash(bundle.state_dict())
    checks = (
        GateResult("finite_raw", bool(torch.isfinite(raw).all()), {}),
        GateResult(
            "anchor_gradient",
            grads[0] is not None
            and torch.isfinite(grads[0]).all()
            and grads[0].norm() > 0,
            {},
        ),
        GateResult(
            "mode_gradient",
            grads[1] is not None
            and torch.isfinite(grads[1]).all()
            and grads[1].norm() > 0,
            {},
        ),
        GateResult(
            "frozen_phases",
            all(
                not p.requires_grad
                for n, p in bundle.named_parameters()
                if n.startswith("system.")
            ),
            {},
        ),
        GateResult(
            "state_unchanged", before == state_content_hash(bundle.state_dict()), {}
        ),
    )
    run = _new_run(config)
    report = run.root / "conditional_multi_anchor_observability_report.json"
    atomic_write_json(
        report,
        {
            "anchor_gradient_norm": float(grads[0].norm())
            if grads[0] is not None
            else 0.0,
            "mode_gradient_norm": float(grads[1].norm())
            if grads[1] is not None
            else 0.0,
            "state_hash": before,
        },
    )
    passed = all(x.passed for x in checks)
    record = EvidenceRecordV2(
        2,
        "conditional_multi_anchor_observability",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {"admission": admission.evidence_hash, "report": file_sha256(report)},
        checks,
        ("multi_anchor_smoke",) if passed else (),
        None if passed else "multi-anchor observability failed",
    )
    _write_evidence(run, "conditional_multi_anchor_observability_evidence.json", record)
    print(run.root)
    return 0


def _multi_anchor_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    assignments: torch.Tensor,
    *,
    batch_number: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(200000 + batch_number)
    local_generator = torch.Generator(device="cpu").manual_seed(100000 + batch_number)
    codes, real_rows, label_rows = [], [], []
    local = torch.rand(10, 4, 6, generator=local_generator).mul_(2.0).sub_(1.0)
    for label in range(10):
        for anchor in range(4):
            members = torch.nonzero(
                (labels == label) & (assignments == anchor), as_tuple=False
            ).flatten()
            if members.numel() < 2:
                raise ValueError(
                    "each multi-anchor cluster needs at least two training images"
                )
            order = torch.randperm(members.numel(), generator=generator)
            selected = members.index_select(0, order[:2])
            real_rows.append(images.index_select(0, selected))
            label_rows.extend((label, label))
            codes.extend(
                (
                    _multi_anchor_route_code(
                        anchor, device=device, local=local[label, anchor]
                    ),
                    _multi_anchor_route_code(
                        anchor, device=device, local=-local[label, anchor]
                    ),
                )
            )
    return (
        torch.stack(codes),
        torch.cat(real_rows).to(device),
        torch.tensor(label_rows, device=device, dtype=torch.long),
    )


def _multi_anchor_anchor_sinkhorn(
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for start in range(0, generated_features.shape[0], 2):
        gf = torch.nn.functional.normalize(generated_features[start : start + 2], dim=1)
        rf = torch.nn.functional.normalize(real_features[start : start + 2], dim=1)
        feature_cost = torch.cdist(gf, rf).square()
        generated_pixels = sinkhorn_cost_view(
            generated_view[start : start + 2]
        ).flatten(1)
        real_pixels = sinkhorn_cost_view(real_view[start : start + 2]).flatten(1)
        pixel_cost = (
            torch.cdist(generated_pixels, real_pixels, p=1) / generated_pixels.shape[1]
        )
        terms.append(
            log_sinkhorn(feature_cost + 0.25 * pixel_cost, epsilon=0.05, iterations=20)
        )
    return torch.stack(terms).mean()


def _multi_anchor_local_diversity(
    generated_features: torch.Tensor, real_features: torch.Tensor
) -> torch.Tensor:
    generated = torch.nn.functional.normalize(generated_features, dim=1)
    real = torch.nn.functional.normalize(real_features, dim=1)
    ratios = []
    for start in range(0, generated.shape[0], 2):
        gd = torch.pdist(generated[start : start + 2]).squeeze(0)
        rd = torch.pdist(real[start : start + 2]).squeeze(0).detach().clamp_min(1.0e-8)
        ratio = gd / rd
        ratios.append(
            torch.relu(0.70 - ratio).square() + 0.25 * torch.relu(ratio - 1.30).square()
        )
    return torch.stack(ratios).mean()


def _multi_anchor_objective(
    *,
    bundle: ConditionalBundle,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    settings,
    config: V2Config,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=1,
        activation_checkpoint=config.execution.activation_checkpoint,
    )
    view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
    generated_features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    class_sinkhorn = conditional_sinkhorn_loss(
        generated_features,
        view,
        real_features,
        real_views,
        labels,
        epsilon=settings.sinkhorn_epsilon,
        iterations=settings.sinkhorn_iterations,
    )
    anchor_sinkhorn = _multi_anchor_anchor_sinkhorn(
        generated_features, view, real_features, real_views
    )
    classification = torch.nn.functional.cross_entropy(logits, labels)
    global_relative = relative_diversity_loss(generated_features, real_features, labels)
    local_relative = _multi_anchor_local_diversity(generated_features, real_features)
    saturation = torch.relu(raw / settings.fixed_exposure_scale - 1.0).square().mean()
    orthogonality = bundle.generator.mode_orthogonality_penalty()
    phase_tv = phase_total_variation(bundle.system)
    total = (
        class_sinkhorn
        + 0.5 * anchor_sinkhorn
        + classification
        + 0.5 * global_relative
        + 0.25 * local_relative
        + 0.1 * saturation
        + 1.0e-3 * orthogonality
        + 1.0e-5 * phase_tv
    )
    summary = {
        "total": float(total.detach()),
        "class_sinkhorn": float(class_sinkhorn.detach()),
        "anchor_sinkhorn": float(anchor_sinkhorn.detach()),
        "classification": float(classification.detach()),
        "global_relative": float(global_relative.detach()),
        "local_relative": float(local_relative.detach()),
        "saturation": float(saturation.detach()),
        "mode_orthogonality": float(orthogonality.detach()),
        "phase_tv": float(phase_tv.detach()),
    }
    return total, summary, raw


def _multi_anchor_checkpoint_manifest(
    config: V2Config,
    *,
    split_hash: str,
    evaluator_hash: str,
    bundle: ConditionalBundle,
    initial_hash: str,
) -> ConditionalCheckpointManifestV2:
    return ConditionalCheckpointManifestV2(
        2,
        MULTI_ANCHOR_ARCHITECTURE,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        split_hash,
        evaluator_hash,
        config.conditional_generation.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_content_hash(bundle.state_dict()),
        MULTI_ANCHOR_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        BALANCED_TRAINING_SCHEDULE_CONTRACT,
        1234,
        initial_hash,
    )


def multi_anchor_smoke(args, config: V2Config) -> int:
    _require_multi_anchor(config)
    observation = _passed_evidence(
        args.observability_evidence, kind="conditional_multi_anchor_observability"
    )
    device = torch.device(config.run.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    require_evidence_action(
        observation,
        kind="conditional_multi_anchor_observability",
        action="multi_anchor_smoke",
        expected_context=expected,
    )
    bundle, init_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    with torch.no_grad():
        train_features, _ = evaluator(train_images.to(device))
    assignments, partition = _multi_anchor_partition(
        train_features.cpu(), train_images, train_labels
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    for parameter in (
        bundle.generator.anchor_logits,
        bundle.generator.local_modes,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    bundle.generator.anchor_logits,
                    bundle.generator.local_modes,
                ],
                "lr": config.conditional_generation.generator_learning_rate,
            },
            {
                "params": [bundle.system.encoder.phase, bundle.system.decoder.phase],
                "lr": config.conditional_generation.phase_learning_rate,
            },
        ]
    )
    history, probes, update_history = [], {}, []
    panels = {
        "anchor_centers": _multi_anchor_panels(device=device, uniform_seed=1234)[
            "anchor_centers"
        ],
        "local_axis": _multi_anchor_panels(device=device, uniform_seed=1234)[
            "local_axis"
        ],
        "uniform_1234": _multi_anchor_panels(device=device, uniform_seed=1234)[
            "uniform_1234"
        ],
        "uniform_5678": _multi_anchor_panels(device=device, uniform_seed=5678)[
            "uniform_5678"
        ],
    }
    run = _new_run(config)
    for step in range(401):
        if step in (0, 100, 200, 400):
            probes[str(step)] = {}
            for name, (panel_codes, panel_labels) in panels.items():
                metrics = _multi_anchor_raw_metrics(
                    bundle,
                    evaluator,
                    panel_codes,
                    panel_labels,
                    device=device,
                    exposure=config.conditional_generation.fixed_exposure_scale,
                )
                raw_path = run.root / f"multi_anchor_{name}_step_{step}_raw.pt"
                atomic_torch_save(raw_path, metrics["raw"])
                probes[str(step)][name] = {
                    key: value for key, value in metrics.items() if key != "raw"
                } | {"raw_hash": file_sha256(raw_path)}
        if step == 400:
            break
        codes, real_views, batch_labels = _multi_anchor_training_batch(
            train_images, train_labels, assignments, batch_number=step, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss, summary, _ = _multi_anchor_objective(
            bundle=bundle,
            evaluator=evaluator,
            codes=codes,
            labels=batch_labels,
            real_views=real_views,
            settings=config.conditional_generation,
            config=config,
        )
        loss.backward()
        groups = [
            bundle.generator.anchor_logits,
            bundle.generator.local_modes,
            bundle.system.encoder.phase,
            bundle.system.decoder.phase,
        ]
        gradients = [
            float(p.grad.norm().item()) if p.grad is not None else 0.0 for p in groups
        ]
        before = [p.detach().clone() for p in groups]
        optimizer.step()
        updates = [
            float((p.detach() - old).norm().item()) for p, old in zip(groups, before)
        ]
        update_history.append(
            {"step": step + 1, "gradients": gradients, "updates": updates}
        )
        history.append(summary | {"step": step + 1})
    first = sum(row["classification"] for row in history[:50]) / max(
        1, len(history[:50])
    )
    last = sum(row["classification"] for row in history[-50:]) / max(
        1, len(history[-50:])
    )
    final_panels = probes["400"]
    panel_pass = all(
        final_panels[name]["accuracy"] >= (0.95 if name == "anchor_centers" else 0.90)
        and min(final_panels[name]["recalls"])
        >= (0.75 if name == "anchor_centers" else 0.80)
        and final_panels[name]["camera_fraction"]
        >= (0.0 if name == "anchor_centers" else 0.25)
        and final_panels[name]["saturation"] <= 0.08
        for name in final_panels
    )
    valid_updates = all(
        all(
            value > 0.0 and torch.isfinite(torch.tensor(value))
            for value in row["updates"] + row["gradients"]
        )
        for row in update_history
    )
    passed = panel_pass and valid_updates and last <= 0.85 * first
    run_history = run.root / "conditional_multi_anchor_smoke_history.json"
    atomic_write_json(
        run_history,
        {
            "contract": MULTI_ANCHOR_SMOKE_CONTRACT,
            "history": history,
            "probes": probes,
            "updates": update_history,
            "partition": partition,
        },
    )
    diagnostic = run.root / "multi_anchor_smoke_final.pt"
    save_conditional_checkpoint(
        diagnostic,
        bundle,
        _multi_anchor_checkpoint_manifest(
            config,
            split_hash=split.manifest_hash,
            evaluator_hash=evaluator_hash,
            bundle=bundle,
            initial_hash=init_manifest.initialization_state_hash,
        ),
        extra={
            "diagnostic_only": True,
            "never_warm_start": True,
            "contract": MULTI_ANCHOR_SMOKE_CONTRACT,
        },
    )
    checks = (
        GateResult("panel_gates", panel_pass, {}),
        GateResult("parameter_gradients_and_updates", valid_updates, {}),
        GateResult(
            "classification_progress",
            last <= 0.85 * first,
            {"first_50": first, "last_50": last},
        ),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_multi_anchor_smoke",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        {
            "observability": observation.evidence_hash,
            "history": file_sha256(run_history),
            "diagnostic_final_checkpoint": file_sha256(diagnostic),
        },
        checks,
        ("train_multi_anchor",) if passed else (),
        None if passed else "multi-anchor smoke failed",
    )
    _write_evidence(run, "conditional_multi_anchor_smoke_evidence.json", record)
    print(run.root)
    return 0


def train_multi_anchor(args, config: V2Config) -> int:
    _require_multi_anchor(config)
    smoke = _passed_evidence(args.smoke_evidence, kind="conditional_multi_anchor_smoke")
    device = torch.device(config.run.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    expected = _context(
        config,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.checkpoint),
    )
    require_evidence_action(
        smoke,
        kind="conditional_multi_anchor_smoke",
        action="train_multi_anchor",
        expected_context=expected,
    )
    bundle, init_manifest = _load_bundle(
        config,
        args.checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
    )
    with torch.no_grad():
        train_features, _ = evaluator(train_images.to(device))
    assignments, partition = _multi_anchor_partition(
        train_features.cpu(), train_images, train_labels
    )
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    for parameter in (
        bundle.generator.anchor_logits,
        bundle.generator.local_modes,
        bundle.system.encoder.phase,
        bundle.system.decoder.phase,
    ):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    bundle.generator.anchor_logits,
                    bundle.generator.local_modes,
                ],
                "lr": config.conditional_generation.generator_learning_rate,
            },
            {
                "params": [bundle.system.encoder.phase, bundle.system.decoder.phase],
                "lr": config.conditional_generation.phase_learning_rate,
            },
        ]
    )
    steps_per_epoch = (
        min(int((train_labels == label).sum().item()) for label in range(10)) // 8
    )
    history = []
    for epoch in range(config.conditional_generation.epochs):
        for local_step in range(steps_per_epoch):
            batch_number = epoch * steps_per_epoch + local_step
            codes, real_views, batch_labels = _multi_anchor_training_batch(
                train_images,
                train_labels,
                assignments,
                batch_number=batch_number,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, summary, _ = _multi_anchor_objective(
                bundle=bundle,
                evaluator=evaluator,
                codes=codes,
                labels=batch_labels,
                real_views=real_views,
                settings=config.conditional_generation,
                config=config,
            )
            loss.backward()
            optimizer.step()
            history.append(summary | {"epoch": epoch, "step": batch_number + 1})
    run = _new_run(config)
    path = run.root / "conditional_multi_anchor_local_modes_generator.pt"
    manifest = _multi_anchor_checkpoint_manifest(
        config,
        split_hash=split.manifest_hash,
        evaluator_hash=evaluator_hash,
        bundle=bundle,
        initial_hash=init_manifest.initialization_state_hash,
    )
    save_conditional_checkpoint(
        path,
        bundle,
        manifest,
        extra={
            "history": history,
            "partition": partition,
            "steps_per_epoch": steps_per_epoch,
            "smoke_evidence": smoke.evidence_hash,
        },
    )
    finite = bool(torch.isfinite(torch.tensor([row["total"] for row in history])).all())
    out_hash = file_sha256(path)
    record = EvidenceRecordV2(
        2,
        "conditional_multi_anchor_training",
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
        {"checkpoint": out_hash, "smoke": smoke.evidence_hash, "steps": len(history)},
        (
            GateResult(
                "finite_training_loss",
                finite,
                {
                    "steps": float(len(history)),
                    "steps_per_epoch": float(steps_per_epoch),
                },
            ),
        ),
        ("evaluate",) if finite else (),
        None if finite else "multi-anchor training produced non-finite loss",
    )
    _write_evidence(run, "conditional_multi_anchor_training_evidence.json", record)
    print(run.root)
    return 0
