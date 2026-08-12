"""End-to-end, diagnostic-only objective alignment experiments."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)
from ...student.conditional import (
    camera_code_fraction,
    camera_code_safety_loss,
    conditional_loss,
    fixed_exposure_view,
    pairwise_diversity_distribution_loss,
    raw_from_generator_batched,
    relative_diversity_ratios,
    coverage_ratio_upper_loss,
    log_sinkhorn,
)
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from ..current.runtime import (
    _context,
    _load_bundle,
    _new_run,
    _random_codes,
    _require_identity,
    _split,
    _validate_conditional_lineage,
    _write_evidence,
)
from ..current.recovery import _conditional_optimizer
from ..current.training import _conditional_parameter_groups, _validate_current_optimizer_coverage
from ...student.checkpoints import state_content_hash
from ...evidence.records import require_evidence_action, require_input_hashes


OBJECTIVE_DIAGNOSTIC_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_objective_alignment_v2",
        "candidates": ["A", "B", "C"],
        "screen_steps": 100,
        "confirm_steps": 400,
        "seeds": [1234, 1235, 1236],
        "panel_seed": 9012,
        "panel_per_class": 17,
        "reference_per_class": 34,
        "weight_rule": "0.25*base_gradient_norm/auxiliary_gradient_norm",
        "camera_target": 0.30,
        "camera_temperature": 0.02,
        "coverage_target": 1.35,
        "coverage_temperature": 0.10,
        "companion": "five_per_label_excluding_primary_v1",
    }
)


def _record_context(config, evaluator_hash: str, split_hash: str, checkpoint_hash: str | None = None) -> EvidenceContextV2:
    return _context(config, evaluator_hash=evaluator_hash, split_hash=split_hash, checkpoint_hash=checkpoint_hash)


def _load_primary_inputs(args, config):
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    primary, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device(config.run.device))
    primary_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    injection = _load_injection(args, config, split, primary_hash)
    bundle, manifest, init_hash = _load_initialization(args, config, train_labels, split, primary_hash, injection)
    return train_images, train_labels, validation_images, validation_labels, split, primary, injection, bundle, manifest, init_hash, primary_hash


def _load_measurement_inputs(args, config):
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    primary, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device(config.run.device))
    measurement = EvidenceRecordV2.from_dict(json.loads(Path(args.acceptance_measurement_evidence).read_text(encoding="utf-8")))
    if measurement.evidence_kind != "conditional_acceptance_measurement" or measurement.status != "passed":
        raise ValueError("objective diagnostic requires passed acceptance measurement evidence")
    from ..current.runtime import _validate_current_evidence_source
    _validate_current_evidence_source(measurement)
    bundle_path = Path(args.acceptance_measurement_bundle)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if measurement.input_hashes.get("measurement_bundle") != file_sha256(bundle_path):
        raise ValueError("objective diagnostic measurement bundle hash mismatch")
    primary_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    secondary_hash = file_sha256(args.acceptance_evaluator_checkpoint)
    if bundle.get("primary_checkpoint") != primary_hash or bundle.get("secondary_checkpoint") != secondary_hash:
        raise ValueError("objective diagnostic evaluator hash mismatch")
    if bundle.get("split_manifest_hash") != split.manifest_hash:
        raise ValueError("objective diagnostic measurement split mismatch")
    manifest_name = bundle.get("test_manifest_filename")
    if not isinstance(manifest_name, str) or not manifest_name or Path(manifest_name).is_absolute() or ".." in Path(manifest_name).parts:
        raise ValueError("objective diagnostic measurement test manifest filename is unsafe")
    test_manifest = bundle_path.parent / manifest_name
    if file_sha256(test_manifest) != bundle.get("test_manifest_hash") or measurement.input_hashes.get("test_manifest") != file_sha256(test_manifest):
        raise ValueError("objective diagnostic measurement test manifest hash mismatch")
    secondary_payload = safe_torch_load(args.acceptance_evaluator_checkpoint, expected_sha256=secondary_hash)
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("objective diagnostic secondary evaluator architecture mismatch")
    secondary = SmoothMNISTEvaluator().to(torch.device(config.run.device))
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    return train_images, train_labels, validation_images, validation_labels, split, primary, secondary, measurement, primary_hash, secondary_hash


def _load_initialization(args, config, train_labels, split, evaluator_hash, injection):
    init_hash = injection.input_hashes.get("initialization_checkpoint")
    if not isinstance(init_hash, str):
        raise ValueError("injection evidence lacks initialization checkpoint")
    bundle, manifest = _load_bundle(
        config,
        args.initialization_checkpoint,
        device=torch.device(config.run.device),
        evaluator_hash=evaluator_hash,
        split_hash=split.manifest_hash,
        checkpoint_sha256=init_hash,
    )
    _validate_conditional_lineage(config, manifest, checkpoint_hash=init_hash, evidence=injection)
    require_input_hashes(injection, {"initialization_checkpoint": init_hash, "initialization_state_hash": manifest.initialization_state_hash})
    return bundle, manifest, init_hash


def _load_injection(args, config, split, evaluator_hash):
    injection = EvidenceRecordV2.from_dict(json.loads(Path(args.injection_evidence).read_text(encoding="utf-8")))
    if injection.evidence_kind != "conditional_injection_admission" or injection.status != "passed":
        raise ValueError("objective diagnostic requires passed injection admission evidence")
    from ..current.runtime import _validate_current_evidence_source
    _validate_current_evidence_source(injection)
    require_evidence_action(injection, kind="conditional_injection_admission", action="conditional_observability", expected_context=_record_context(config, evaluator_hash, split.manifest_hash, injection.input_hashes.get("initialization_checkpoint")))
    return injection


def _parameter_norm(parameters, gradients) -> torch.Tensor:
    values = [gradient.detach().square().sum() for parameter, gradient in zip(parameters, gradients, strict=True) if gradient is not None]
    if not values:
        return torch.zeros((), device=parameters[0].device)
    return torch.stack(values).sum().sqrt()


def _gradient_vector(parameters, gradients) -> torch.Tensor:
    """Flatten a gradient tuple, preserving zero entries for unused params."""
    return torch.cat([
        (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ])


def _group_gradient_norms(groups, selected, gradients) -> dict[str, float]:
    """Return per-optimizer-group norms without changing the autograd graph."""
    result = {}
    positions = {id(parameter): index for index, parameter in enumerate(selected)}
    for name, parameters in groups.items():
        values = []
        for parameter in parameters:
            index = positions.get(id(parameter))
            if index is None:
                values.append(torch.zeros_like(parameter).reshape(-1))
                continue
            gradient = gradients[index]
            values.append((gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1))
        vector = torch.cat(values)
        result[name] = float(vector.norm().item())
    return result


def _gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = left.norm()
    right_norm = right.norm()
    if not torch.isfinite(torch.stack((left_norm, right_norm))).all() or left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return None
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


def _auxiliary_loss(raw, generated_view, generated_features, real_features, labels, real_views, kind: str, epsilon: float, iterations: int) -> torch.Tensor:
    if kind == "camera":
        return torch.nn.functional.softplus((torch.as_tensor(0.30, device=raw.device, dtype=raw.dtype) - camera_code_fraction(raw, labels)) / 0.02) * 0.02
    if kind == "pairwise":
        return pairwise_diversity_distribution_loss(generated_features, real_features, labels)
    if kind == "coverage":
        return coverage_ratio_upper_loss(generated_features, generated_view, real_features, real_views, labels, epsilon=epsilon, iterations=iterations)
    raise ValueError(f"unknown auxiliary objective {kind}")


def _code_seed(seed: int, step: int) -> int:
    bases = {1234: 100000, 1235: 200000, 1236: 300000}
    if seed not in bases:
        raise ValueError("objective diagnostic seed is not registered")
    return bases[seed] + step


def _gradient_calibration(args, config, train_images, train_labels, split, primary, bundle, evaluator_hash, input_hashes):
    settings = config.conditional_generation
    assert settings is not None
    groups = _conditional_parameter_groups(bundle)
    selected = tuple(parameter for name in ("global_modes", "code_projection", "class_heads") for parameter in groups[name])
    sums = {name: torch.zeros((), device=selected[0].device) for name in ("base", "camera", "pairwise", "coverage")}
    steps = 0
    batch_records = []
    batch_vectors = []
    for epoch in range(1):
        for batch_number, indices in enumerate(balanced_batches(train_labels, batch_size=settings.logical_batch_size, seed=1234 + epoch)):
            if steps >= 4:
                break
            device = selected[0].device
            labels = train_labels.index_select(0, indices).to(device)
            codes = _random_codes(labels.numel(), device=selected[0].device, seed=(epoch + 1) * 100000 + batch_number)
            real_views = train_images.index_select(0, indices).to(device)
            base, _, raw = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=primary, codes=codes, labels=labels, real_views=real_views, exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=False, microbatch_size=config.execution.training_microbatch_size)
            generated_view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
            generated_features, _ = primary(generated_view)
            with torch.no_grad():
                real_features, _ = primary(real_views)
            companion_indices = _companion_indices(train_labels, indices, seed=1234 * 100000 + batch_number)
            companion_views = train_images.index_select(0, companion_indices).to(device)
            companion_labels = train_labels.index_select(0, companion_indices).to(device)
            with torch.no_grad():
                companion_features, _ = primary(companion_views)
            losses = {
                "base": base,
                "camera": _auxiliary_loss(raw, generated_view, generated_features, real_features, labels, real_views, "camera", settings.sinkhorn_epsilon, settings.sinkhorn_iterations),
                "pairwise": _auxiliary_loss(raw, generated_view, generated_features, real_features, labels, real_views, "pairwise", settings.sinkhorn_epsilon, settings.sinkhorn_iterations),
                "coverage": coverage_ratio_upper_loss(
                    generated_features, generated_view, real_features, real_views, labels,
                    epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations,
                    companion_features=companion_features, companion_view=companion_views,
                    companion_labels=companion_labels,
                ),
            }
            gradient_vectors = {}
            batch_norms = {}
            group_norms = {}
            for name, loss in losses.items():
                gradients = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
                norm = _parameter_norm(selected, gradients)
                gradient_vectors[name] = _gradient_vector(selected, gradients)
                batch_norms[name] = float(norm.item())
                group_norms[name] = _group_gradient_norms(groups, selected, gradients)
                sums[name] = sums[name] + norm.square()
            batch_records.append({
                "batch_index_hash": canonical_sha256(indices),
                "code_seed": (epoch + 1) * 100000 + batch_number,
                "gradient_norms": batch_norms,
                "gradient_norms_by_group": group_norms,
                "losses": {name: float(loss.detach().item()) for name, loss in losses.items()},
                "gradient_cosines": {
                    f"{left}_vs_{right}": _gradient_cosine(gradient_vectors[left], gradient_vectors[right])
                    for left, right in (("base", "camera"), ("base", "pairwise"), ("base", "coverage"), ("camera", "pairwise"), ("camera", "coverage"), ("pairwise", "coverage"))
                },
            })
            batch_vectors.append(gradient_vectors)
            steps += 1
        if steps >= 4:
            break
    if steps != 4:
        raise ValueError("gradient calibration did not obtain four balanced batches")
    norms = {name: float((value / 4.0).sqrt().item()) for name, value in sums.items()}
    weights = {name: 0.25 * norms["base"] / norms[name] if torch.isfinite(torch.tensor([norms["base"], norms[name]])).all() and norms[name] > 1.0e-12 else None for name in ("camera", "pairwise", "coverage")}
    violations = {
        name: (value is None or not 1.0e-3 <= value <= 10.0)
        for name, value in weights.items()
    }
    effective = []
    for record, vectors in zip(batch_records, batch_vectors, strict=True):
        base_vector = vectors["base"]
        base_norm = base_vector.norm()
        weighted = {
            name: (vectors[name] * float(weights[name]) if weights[name] is not None else torch.full_like(base_vector, float("nan")))
            for name in ("camera", "pairwise", "coverage")
        }
        ratios = {name: float((value.norm() / base_norm).item()) if torch.isfinite(base_norm) and base_norm > 1.0e-12 else None for name, value in weighted.items()}
        combined = {}
        for kind, names in (("B", ("camera", "pairwise")), ("C", ("camera", "pairwise", "coverage"))):
            vector = base_vector + sum((weighted[name] for name in names), torch.zeros_like(base_vector))
            combined[kind] = float((vector.norm() / base_norm).item()) if torch.isfinite(base_norm) and base_norm > 1.0e-12 else None
        record["weighted_gradient_ratios"] = ratios
        record["combined_gradient_ratios"] = combined
        base_groups = record["gradient_norms_by_group"]["base"]
        record["weighted_gradient_ratios_by_group"] = {
            name: {
                group: (float(weights[name]) * record["gradient_norms_by_group"][name][group] / base_groups[group]
                        if base_groups[group] > 1.0e-12 else None)
                for group in groups
            }
            for name in ("camera", "pairwise", "coverage")
        }
        effective.append({"weighted_gradient_ratios": ratios, "combined_gradient_ratios": combined, "weighted_gradient_ratios_by_group": record["weighted_gradient_ratios_by_group"]})
    aggregate = {
        "weighted_gradient_ratios": {name: float(sum(row["weighted_gradient_ratios"][name] for row in effective) / len(effective)) for name in ("camera", "pairwise", "coverage")},
        "combined_gradient_ratios": {kind: float(sum(row["combined_gradient_ratios"][kind] for row in effective) / len(effective)) for kind in ("B", "C")},
    }
    return norms, weights, {"batches": batch_records, "rms_norms": norms, "raw_weights": weights, "violations": violations, "effective_gradients": aggregate}


def _calibrated_weight(base: float, auxiliary: float) -> float:
    if not torch.isfinite(torch.tensor([base, auxiliary])).all() or auxiliary <= 1.0e-12:
        raise ValueError("objective gradient calibration encountered invalid norm")
    value = 0.25 * base / auxiliary
    if not 1.0e-3 <= value <= 10.0:
        raise ValueError("objective gradient calibration produced a weight outside [1e-3,10]")
    return float(value)


def _effective_gradient_gate(diagnostics: dict[str, object]) -> tuple[bool, dict[str, object]]:
    """Apply scale-invariant per-batch and aggregate gradient budgets."""
    rows = diagnostics.get("batches", [])
    effective = diagnostics.get("effective_gradients", {})
    if not isinstance(rows, list) or not isinstance(effective, dict):
        return False, {"reason": "missing effective gradient diagnostics"}
    weighted = effective.get("weighted_gradient_ratios", {})
    combined = effective.get("combined_gradient_ratios", {})
    batch_safe = all(
        isinstance(row, dict)
        and all(value is not None and float(value) <= 0.50 for value in row.get("weighted_gradient_ratios", {}).values())
        and all(value is None or float(value) <= 1.0 for auxiliary in row.get("weighted_gradient_ratios_by_group", {}).values() for value in auxiliary.values())
        and row.get("combined_gradient_ratios", {}).get("B") is not None
        and float(row["combined_gradient_ratios"]["B"]) <= 0.75
        and row.get("combined_gradient_ratios", {}).get("C") is not None
        and float(row["combined_gradient_ratios"]["C"]) <= 1.00
        for row in rows
    )
    aggregate_safe = (
        set(weighted) == {"camera", "pairwise", "coverage"}
        and all(abs(float(value) - 0.25) <= 0.05 for value in weighted.values())
        and float(combined.get("B", float("inf"))) <= 0.75
        and float(combined.get("C", float("inf"))) <= 1.00
    )
    return batch_safe and aggregate_safe, {
        "batch_safe": batch_safe,
        "aggregate_safe": aggregate_safe,
        "weighted": weighted,
        "combined": combined,
    }


def _fixed_panel(device: torch.device, code_dim: int = 8):
    generator = torch.Generator(device="cpu").manual_seed(9012)
    codes = torch.randn((170, code_dim), generator=generator).to(device)
    labels = torch.arange(10, device=device).repeat_interleave(17)
    return codes, labels


def _panel_coverage(generated_features, generated_view, labels, real_features, real_views, real_labels, epsilon, iterations):
    observed, baseline = [], []
    for label in range(10):
        gi = torch.nonzero(labels == label, as_tuple=False).flatten()
        ri = torch.nonzero(real_labels == label, as_tuple=False).flatten()
        if gi.numel() != 17 or ri.numel() != 34:
            raise ValueError("objective panel requires 17 generated and 34 real samples per label")
        def cost(left_f, left_v, right_f, right_v):
            left_f = torch.nn.functional.normalize(left_f, dim=1)
            right_f = torch.nn.functional.normalize(right_f, dim=1)
            fcost = torch.cdist(left_f, right_f).square()
            lp = left_v.flatten(1)
            rp = right_v.flatten(1)
            return log_sinkhorn(fcost + 0.25 * torch.cdist(lp, rp, p=1) / lp.shape[1], epsilon=epsilon, iterations=iterations)
        first, second = ri[:17], ri[17:]
        observed.append(cost(generated_features[gi], generated_view[gi], real_features[second], real_views[second]))
        baseline.append(cost(real_features[first], real_views[first], real_features[second], real_views[second]).detach())
    return float((torch.stack(observed).mean() / torch.stack(baseline).mean().clamp_min(1.0e-12)).item())


def _measure_panel(bundle, primary, secondary, validation_images, validation_labels, config):
    device = next(bundle.parameters()).device
    settings = config.conditional_generation
    assert settings is not None
    codes, labels = _fixed_panel(device)
    with torch.no_grad():
        raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
        generated_view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
        generated_features, primary_logits = primary(generated_view)
        secondary_features, secondary_logits = secondary(generated_view)
        real_indices = torch.cat([torch.nonzero(validation_labels == label, as_tuple=False).flatten()[:34] for label in range(10)])
        real_view = validation_images.index_select(0, real_indices).to(device)
        real_labels = validation_labels.index_select(0, real_indices).to(device)
        real_features, primary_real_logits = primary(real_view)
        secondary_real_features, secondary_real_logits = secondary(real_view)
    primary_pred, secondary_pred = primary_logits.argmax(1), secondary_logits.argmax(1)
    primary_recall = [float((primary_pred[labels == label] == label).float().mean().item()) for label in range(10)]
    secondary_recall = [float((secondary_pred[labels == label] == label).float().mean().item()) for label in range(10)]
    return {
        "camera_code_fraction": float(camera_code_fraction(raw, labels).item()),
        "saturation": float((raw / settings.fixed_exposure_scale >= 1.0).float().mean().item()),
        "primary_accuracy": float((primary_pred == labels).float().mean().item()),
        "secondary_accuracy": float((secondary_pred == labels).float().mean().item()),
        "primary_min_recall": min(primary_recall),
        "secondary_min_recall": min(secondary_recall),
        "prediction_agreement": float((primary_pred == secondary_pred).float().mean().item()),
        "minimum_diversity_ratio": min(relative_diversity_ratios(generated_features, real_features[:170], labels)),
        "secondary_minimum_diversity_ratio": min(relative_diversity_ratios(secondary_features, secondary_real_features[:170], labels)),
        "coverage_ratio": _panel_coverage(generated_features, generated_view, labels, real_features, real_view, real_labels, settings.sinkhorn_epsilon, settings.sinkhorn_iterations),
        "secondary_coverage_ratio": _panel_coverage(secondary_features, generated_view, labels, secondary_real_features, real_view, real_labels, settings.sinkhorn_epsilon, settings.sinkhorn_iterations),
    }


def _companion_indices(labels: torch.Tensor, primary_indices: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    primary_labels = labels.index_select(0, primary_indices)
    selected: list[torch.Tensor] = []
    for label in range(10):
        primary = primary_indices[primary_labels == label]
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        candidates = candidates[~torch.isin(candidates, primary)]
        if candidates.numel() < primary.numel():
            raise ValueError("companion schedule cannot exclude primary samples")
        order = torch.randperm(candidates.numel(), generator=generator)
        selected.append(candidates.index_select(0, order[: primary.numel()]))
    return torch.cat(selected)


def _scheduled_batches(labels: torch.Tensor, *, batch_size: int, seed: int, start_step: int, steps: int) -> list[torch.Tensor]:
    """Return the exact contiguous slice of the registered balanced schedule."""
    batches = list(balanced_batches(labels, batch_size=batch_size, seed=seed))
    end = start_step + steps
    if start_step < 0 or steps < 0 or end > len(batches):
        raise ValueError("objective diagnostic schedule is shorter than requested range")
    return batches[start_step:end]


def _run_steps(bundle, primary, train_images, train_labels, config, objective_kind, weights, steps, seed, *, optimizer=None, start_step=0, history=None, primary_schedule=None, companion_schedule=None):
    settings = config.conditional_generation
    assert settings is not None
    optimizer = optimizer or _conditional_optimizer(bundle, config)
    groups = _conditional_parameter_groups(bundle)
    _validate_current_optimizer_coverage(optimizer, groups)
    history = [] if history is None else history
    primary_schedule = [] if primary_schedule is None else primary_schedule
    companion_schedule = [] if companion_schedule is None else companion_schedule
    gradient_seen = {name: False for name in groups}
    before = {name: [parameter.detach().clone() for parameter in parameters] for name, parameters in groups.items()}
    device = next(bundle.parameters()).device
    iterator = iter(_scheduled_batches(train_labels, batch_size=settings.logical_batch_size, seed=seed, start_step=start_step, steps=steps))
    for local_step in range(steps):
        step = start_step + local_step
        indices = next(iterator)
        primary_schedule.append({"batch_hash": canonical_sha256(indices), "code_seed": _code_seed(seed, step)})
        labels = train_labels.index_select(0, indices).to(device)
        codes = _random_codes(labels.numel(), device=device, seed=_code_seed(seed, step))
        real_views = train_images.index_select(0, indices).to(device)
        companion_views = None
        companion_labels = None
        if objective_kind == "C":
            companions = _companion_indices(train_labels, indices, seed=seed * 100000 + step)
            companion_schedule.append(canonical_sha256(companions))
            companion_views = train_images.index_select(0, companions).to(device)
            companion_labels = train_labels.index_select(0, companions).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, summary, raw = conditional_loss(
            system=bundle.system,
            generator=bundle.generator,
            evaluator=primary,
            codes=codes,
            labels=labels,
            real_views=real_views,
            exposure_scale=settings.fixed_exposure_scale,
            sinkhorn_epsilon=settings.sinkhorn_epsilon,
            sinkhorn_iterations=settings.sinkhorn_iterations,
            activation_checkpoint=False,
            microbatch_size=config.execution.training_microbatch_size,
            objective_kind=objective_kind,
            camera_code_safety_weight=weights["camera"],
            pairwise_diversity_weight=weights["pairwise"],
            coverage_ratio_weight=weights["coverage"],
            coverage_companion_views=companion_views,
            coverage_companion_labels=companion_labels,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"objective {objective_kind} produced non-finite loss at step {step + 1}")
        loss.backward()
        for name, parameters in groups.items():
            if any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.detach().norm() > 0 for parameter in parameters):
                gradient_seen[name] = True
        optimizer.step()
        with torch.no_grad():
            generated_view = fixed_exposure_view(raw.detach(), settings.fixed_exposure_scale)
            generated_features, _ = primary(generated_view)
            real_features, _ = primary(real_views)
            companion_features = None
            if companion_views is not None:
                companion_features, _ = primary(companion_views)
            auxiliary = {
                "camera": float(camera_code_safety_loss(raw.detach(), labels).item()),
                "pairwise": float(pairwise_diversity_distribution_loss(generated_features, real_features, labels).item()),
                "coverage": float(coverage_ratio_upper_loss(generated_features, generated_view, real_features, real_views, labels, epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations, companion_features=companion_features, companion_view=companion_views, companion_labels=companion_labels).item()),
            }
        history.append({
            "step": step + 1,
            "total": float(summary.total),
            "sinkhorn": float(summary.sinkhorn),
            "classification": float(summary.classification),
            "diversity": float(summary.diversity),
            "saturation": float(summary.saturation),
            "camera_code_fraction": float(camera_code_fraction(raw.detach(), labels).item()),
            "auxiliary_raw": auxiliary,
            "frozen_weights": dict(weights),
            "differentiable_coverage_ratio": auxiliary["coverage"],
        })
    updated = {
        name: gradient_seen[name] and any(not torch.equal(parameter.detach(), previous) for parameter, previous in zip(parameters, before[name], strict=True))
        for name, parameters in groups.items()
    }
    return history, primary_schedule, companion_schedule, optimizer, {"gradient_seen": gradient_seen, "updated": updated}


def _diagnostic_record(config, kind, status, context, input_hashes, gates, permitted=(), reason=None, extra=None, artifact_payloads=None):
    run = _new_run(config)
    artifact_hashes = {}
    for filename, payload in (artifact_payloads or {}).items():
        artifact_path = run.root / filename
        atomic_torch_save(artifact_path, payload)
        artifact_hashes[filename] = file_sha256(artifact_path)
    record = EvidenceRecordV2(2, kind, "diagnostic", status, run.git_commit, run.dirty_patch_hash, run.config_hash, context, {**input_hashes, "artifacts": canonical_sha256(artifact_hashes), **artifact_hashes}, tuple(gates), tuple(permitted) if status == "passed" else (), reason)
    path = run.root / f"{kind}_evidence.json"
    _write_evidence(run, path.name, record)
    if extra:
        atomic_write_json(run.root / f"{kind}_report.json", extra)
    print(run.root)
    return run, record


def conditional_objective_weight_calibration(args, config) -> int:
    _require_identity(config)
    train_images, train_labels, _, _, split, primary, injection, bundle, manifest, init_hash, evaluator_hash = _load_primary_inputs(args, config)
    context = _record_context(config, evaluator_hash, split.manifest_hash, init_hash)
    input_hashes = {"initialization_checkpoint": init_hash, "injection_evidence": file_sha256(args.injection_evidence), "evaluator": evaluator_hash, "split": split.manifest_hash}
    try:
        norms, weights, diagnostics = _gradient_calibration(args, config, train_images, train_labels, split, primary, bundle, evaluator_hash, {"initialization": init_hash})
    except (ValueError, RuntimeError) as error:
        reason = str(error)
        diagnostics = {"batches": [], "rms_norms": {}, "raw_weights": {}, "violations": {}, "execution_error": reason}
        norms, weights = {}, {}
    input_bindings = {"initialization_checkpoint": init_hash, "injection_evidence": file_sha256(args.injection_evidence), "evaluator": evaluator_hash, "split": split.manifest_hash}
    violations = diagnostics.get("violations", {})
    finite = bool(norms) and all(torch.isfinite(torch.tensor(value)) and value > 1.0e-12 for value in norms.values())
    in_range = bool(weights) and not any(violations.values())
    effective_passed, effective_report = _effective_gradient_gate(diagnostics)
    weighted_ratios = effective_report.get("weighted", {})
    combined_ratios = effective_report.get("combined", {})
    batch_safe = bool(effective_report.get("batch_safe", False))
    aggregate_safe = bool(effective_report.get("aggregate_safe", False))
    passed = finite and len(diagnostics.get("batches", [])) == 4 and effective_passed and "execution_error" not in diagnostics
    artifact = {
        "schema_version": 3,
        "contract": OBJECTIVE_DIAGNOSTIC_CONTRACT,
        "formula": "0.25*base_gradient_norm/auxiliary_gradient_norm",
        "target_ratio": 0.25,
        "legacy_raw_weight_range": [1.0e-3, 10.0],
        "norms": norms,
        "weights": weights,
        "diagnostics": diagnostics,
        "frozen": passed,
        "formal_promotion_authorized": False,
        "input_hashes": input_bindings,
    }
    run = _new_run(config)
    artifact_path = run.root / "conditional_objective_weight_calibration.json"
    atomic_write_json(artifact_path, artifact)
    gates = (
        GateResult("four_balanced_batches", len(diagnostics.get("batches", [])) == 4, {"count": float(len(diagnostics.get("batches", [])))}),
        GateResult("finite_gradient_norms", finite, norms),
        GateResult("legacy_raw_weight_range", in_range, {name: value for name, value in weights.items() if value is not None}),
        GateResult("effective_gradient_batches", batch_safe, {"max_weighted_ratio": 0.50, "max_B_ratio": 0.75, "max_C_ratio": 1.00}),
        GateResult("effective_gradient_aggregate", aggregate_safe, {"weighted": weighted_ratios, "combined": combined_ratios}),
    )
    reason = diagnostics.get("execution_error") or ("effective gradient safety gate failed" if not (batch_safe and aggregate_safe) else None)
    record = EvidenceRecordV2(2, "conditional_objective_weight_calibration", "diagnostic", "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"artifact": file_sha256(artifact_path), **input_hashes}, gates, ("screen",) if passed else (), reason)
    _write_evidence(run, "conditional_objective_weight_calibration_evidence.json", record)
    print(run.root)
    return 0 if passed else 1


def _read_calibration(path: str, config, evaluator_hash: str, init_hash: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != OBJECTIVE_DIAGNOSTIC_CONTRACT or payload.get("schema_version") != 3 or payload.get("frozen") is not True:
        raise ValueError("objective calibration contract mismatch")
    inputs = payload.get("input_hashes")
    if not isinstance(inputs, dict) or inputs.get("evaluator") != evaluator_hash or inputs.get("initialization_checkpoint") != init_hash:
        raise ValueError("objective calibration lineage mismatch")
    weights = payload.get("weights")
    if not isinstance(weights, dict) or set(weights) != {"camera", "pairwise", "coverage"}:
        raise ValueError("objective calibration lacks all auxiliary weights")
    return {name: float(weights[name]) for name in ("camera", "pairwise", "coverage")}


def objective_alignment_screen(args, config) -> int:
    _require_identity(config)
    train_images, train_labels, validation_images, validation_labels, split, primary, secondary, measurement, evaluator_hash, secondary_hash = _load_measurement_inputs(args, config)
    injection = _load_injection(args, config, split, evaluator_hash)
    bundle, manifest, init_hash = _load_initialization(args, config, train_labels, split, evaluator_hash, injection)
    calibration_path = args.weight_calibration
    weights = _read_calibration(calibration_path, config, evaluator_hash, init_hash)
    reports = {}
    for kind in ("A", "B", "C"):
        candidate = _load_bundle(config, args.initialization_checkpoint, device=torch.device(config.run.device), evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_sha256=init_hash)[0]
        candidate_weights = {name: 0.0 for name in weights} if kind == "A" else weights
        history, primary_schedule, companion_schedule, _, update_status = _run_steps(candidate, primary, train_images, train_labels, config, kind, candidate_weights, 100, 1234)
        reports[kind] = {"measurement": _measure_panel(candidate, primary, secondary, validation_images, validation_labels, config), "history": history, "primary_schedule_hash": canonical_sha256(primary_schedule), "companion_schedule_hash": canonical_sha256(companion_schedule), "update_status": update_status, "state_hash": state_content_hash(candidate.state_dict()), "state_dict": {name: value.detach().cpu() for name, value in candidate.state_dict().items()}}
    baseline = reports["A"]["measurement"]
    gates = []
    scores = {}
    for kind in ("B", "C"):
        row = reports[kind]["measurement"]
        checks = {
            "camera": row["camera_code_fraction"] >= baseline["camera_code_fraction"] + 0.02,
            "diversity": row["minimum_diversity_ratio"] >= baseline["minimum_diversity_ratio"] + 0.05,
            "coverage": row["coverage_ratio"] <= baseline["coverage_ratio"] - 0.05,
            "accuracy": row["primary_accuracy"] >= baseline["primary_accuracy"] - 0.03 and row["secondary_accuracy"] >= baseline["secondary_accuracy"] - 0.03,
            "recall": row["primary_min_recall"] >= baseline["primary_min_recall"] - 0.05 and row["secondary_min_recall"] >= baseline["secondary_min_recall"] - 0.05,
            "saturation": row["saturation"] <= 0.08,
            "finite": all(torch.isfinite(torch.tensor(value)) for value in row.values()),
        }
        gates.append(GateResult(f"{kind}_screen_gate", all(checks.values()), checks))
        scores[kind] = min((row["camera_code_fraction"] - baseline["camera_code_fraction"]) / 0.02, (row["minimum_diversity_ratio"] - baseline["minimum_diversity_ratio"]) / 0.05, (baseline["coverage_ratio"] - row["coverage_ratio"]) / 0.05)
    update_gate = GateResult("screen_update_groups", all(all(row["update_status"][key].values()) for key in ("gradient_seen", "updated") for row in reports.values()), {kind: row["update_status"] for kind, row in reports.items()})
    gates.append(update_gate)
    schedule_gate = GateResult("screen_primary_schedule", len({row["primary_schedule_hash"] for row in reports.values()}) == 1, {kind: row["primary_schedule_hash"] for kind, row in reports.items()})
    gates.append(schedule_gate)
    candidates = [kind for kind in ("B", "C") if next(g for g in gates if g.name == f"{kind}_screen_gate").passed]
    if not update_gate.passed or not schedule_gate.passed:
        candidates = []
    winner = min(candidates, key=lambda kind: (-scores[kind], kind)) if candidates else None
    gates.append(GateResult("winner_exists", winner is not None, {"winner": winner or "none"}))
    report_payload = {"contract": OBJECTIVE_DIAGNOSTIC_CONTRACT, "winner": winner, "scores": scores, "reports": {kind: {key: value for key, value in row.items() if key != "state_dict"} for kind, row in reports.items()}, "weights": weights, "all_gates": [gate.__dict__ for gate in gates]}
    evidence_gates = tuple(
        gate
        for gate in gates
        if winner is None or gate.name == f"{winner}_screen_gate" or gate.name not in {"B_screen_gate", "C_screen_gate"}
    )
    artifacts = {f"conditional_objective_alignment_{kind}_checkpoint.pt": {"objective_kind": kind, "state_dict": row["state_dict"], "state_content_hash": row["state_hash"]} for kind, row in reports.items()}
    _diagnostic_record(config, "conditional_objective_alignment_screen", "passed" if winner else "failed", _record_context(config, evaluator_hash, split.manifest_hash, init_hash), {"initialization_checkpoint": init_hash, "injection_evidence": file_sha256(args.injection_evidence), "evaluator": evaluator_hash, "secondary_evaluator": secondary_hash, "measurement_evidence": measurement.evidence_hash, "measurement_bundle": file_sha256(args.acceptance_measurement_bundle), "calibration": file_sha256(calibration_path), "companion_schedule": canonical_sha256({kind: reports[kind]["companion_schedule_hash"] for kind in reports})}, evidence_gates, ("confirm",) if winner else (), extra=report_payload, artifact_payloads=artifacts)
    return 0 if winner else 1


def objective_alignment_confirm(args, config) -> int:
    _require_identity(config)
    screen = EvidenceRecordV2.from_dict(json.loads(Path(args.screen_evidence).read_text(encoding="utf-8")))
    if screen.evidence_kind != "conditional_objective_alignment_screen" or screen.status != "passed" or "confirm" not in screen.permitted_next_actions:
        raise ValueError("objective confirmation requires passed screen evidence")
    from ..current.runtime import _validate_current_evidence_source
    _validate_current_evidence_source(screen)
    report_path = Path(args.screen_evidence).with_name("conditional_objective_alignment_screen_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    winner = report.get("winner")
    if winner not in {"B", "C"}:
        raise ValueError("screen evidence lacks a unique B/C winner")
    train_images, train_labels, validation_images, validation_labels, split, primary, secondary, measurement, evaluator_hash, secondary_hash = _load_measurement_inputs(args, config)
    _validate_current_evidence_source(screen, measurement)
    if screen.input_hashes.get("measurement_evidence") != measurement.evidence_hash or screen.input_hashes.get("measurement_bundle") != file_sha256(args.acceptance_measurement_bundle):
        raise ValueError("screen evidence measurement lineage mismatch")
    injection = _load_injection(args, config, split, evaluator_hash)
    bundle, manifest, init_hash = _load_initialization(args, config, train_labels, split, evaluator_hash, injection)
    weights = report.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("screen report lacks frozen weights")
    rows = {}
    gates = []
    for seed in (1234, 1235, 1236):
        seed_rows = {}
        for kind in ("A", winner):
            candidate = _load_bundle(config, args.initialization_checkpoint, device=torch.device(config.run.device), evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_sha256=init_hash)[0]
            candidate_weights = {name: 0.0 for name in weights} if kind == "A" else {name: float(weights[name]) for name in weights}
            history, primary_schedule, companion_schedule, optimizer, update_status_100 = _run_steps(candidate, primary, train_images, train_labels, config, kind, candidate_weights, 100, seed)
            measurement_100 = _measure_panel(candidate, primary, secondary, validation_images, validation_labels, config)
            history, primary_schedule, companion_schedule, _, update_status_400 = _run_steps(candidate, primary, train_images, train_labels, config, kind, candidate_weights, 300, seed, optimizer=optimizer, start_step=100, history=history, primary_schedule=primary_schedule, companion_schedule=companion_schedule)
            seed_rows[kind] = {"measurement_100": measurement_100, "measurement_400": _measure_panel(candidate, primary, secondary, validation_images, validation_labels, config), "history": history, "primary_schedule_hash": canonical_sha256(primary_schedule), "companion_schedule_hash": canonical_sha256(companion_schedule), "update_status_100": update_status_100, "update_status_400": update_status_400, "state_hash": state_content_hash(candidate.state_dict()), "state_dict": {name: value.detach().cpu() for name, value in candidate.state_dict().items()}}
        rows[str(seed)] = seed_rows
        a = seed_rows["A"]["measurement_400"]
        w = seed_rows[winner]["measurement_400"]
        gates.append(GateResult(f"seed_{seed}_improvement", w["camera_code_fraction"] >= a["camera_code_fraction"] + 0.03 and w["minimum_diversity_ratio"] >= a["minimum_diversity_ratio"] + 0.10 and w["coverage_ratio"] <= a["coverage_ratio"] - 0.10, {"camera_delta": w["camera_code_fraction"] - a["camera_code_fraction"], "diversity_delta": w["minimum_diversity_ratio"] - a["minimum_diversity_ratio"], "coverage_delta": w["coverage_ratio"] - a["coverage_ratio"]}))
    joint = [rows[str(seed)][winner]["measurement_400"] for seed in (1234, 1235, 1236)]
    joint_count = sum(item["camera_code_fraction"] >= 0.25 and item["minimum_diversity_ratio"] >= 0.55 and item["coverage_ratio"] <= 1.5 for item in joint)
    gates.append(GateResult("two_seed_absolute_gate", joint_count >= 2, {"count": float(joint_count)}))
    finite = all(torch.isfinite(torch.tensor(list(item["measurement_400"].values()))).all() and torch.isfinite(torch.tensor(list(item["measurement_100"].values()))).all() for item in [rows[str(seed)][winner] for seed in (1234, 1235, 1236)])
    gates.append(GateResult("finite_confirmation", finite, {}))
    for seed in (1234, 1235, 1236):
        row = rows[str(seed)][winner]
        a400 = rows[str(seed)]["A"]["measurement_400"]
        w100 = row["measurement_100"]
        w400 = row["measurement_400"]
        gates.append(GateResult(f"seed_{seed}_trend", w400["camera_code_fraction"] >= w100["camera_code_fraction"] - 0.03, {"camera_delta_100_to_400": w400["camera_code_fraction"] - w100["camera_code_fraction"]}))
        gates.append(GateResult(f"seed_{seed}_primary_schedule", rows[str(seed)]["A"]["primary_schedule_hash"] == row["primary_schedule_hash"], {"A": rows[str(seed)]["A"]["primary_schedule_hash"], "winner": row["primary_schedule_hash"]}))
        for candidate_kind in ("A", winner):
            candidate_row = rows[str(seed)][candidate_kind]
            for label, status in ((f"{candidate_kind}_step_100", candidate_row["update_status_100"]), (f"{candidate_kind}_step_400", candidate_row["update_status_400"])):
                gates.append(GateResult(f"seed_{seed}_{label}_updates", all(status["gradient_seen"].values()) and all(status["updated"].values()), status))
        gates.append(GateResult(f"seed_{seed}_measurement_fidelity", w400["primary_accuracy"] >= a400["primary_accuracy"] - 0.03 and w400["secondary_accuracy"] >= a400["secondary_accuracy"] - 0.03 and w400["primary_min_recall"] >= a400["primary_min_recall"] - 0.05 and w400["secondary_min_recall"] >= a400["secondary_min_recall"] - 0.05 and w400["saturation"] <= 0.08, {"step_100": w100, "step_400": w400}))
    passed = all(gate.passed for gate in gates)
    report_payload = {"contract": OBJECTIVE_DIAGNOSTIC_CONTRACT, "winner": winner, "seeds": {seed: {kind: {key: value for key, value in row.items() if key != "state_dict"} for kind, row in seed_rows.items()} for seed, seed_rows in rows.items()}, "formal_promotion_authorized": False}
    artifacts = {f"conditional_objective_alignment_confirm_{seed}_{kind}_checkpoint.pt": {"objective_kind": kind, "seed": int(seed), "state_dict": row["state_dict"], "state_content_hash": row["state_hash"]} for seed, seed_rows in rows.items() for kind, row in seed_rows.items()}
    _diagnostic_record(config, "conditional_objective_alignment_confirm", "passed" if passed else "failed", _record_context(config, evaluator_hash, split.manifest_hash, init_hash), {"screen_evidence": screen.evidence_hash, "initialization_checkpoint": init_hash, "injection_evidence": file_sha256(args.injection_evidence), "evaluator": evaluator_hash, "secondary_evaluator": secondary_hash, "measurement_evidence": measurement.evidence_hash, "measurement_bundle": file_sha256(args.acceptance_measurement_bundle), "companion_schedule": canonical_sha256({seed: {kind: rows[str(seed)][kind]["companion_schedule_hash"] for kind in ("A", winner)} for seed in (1234, 1235, 1236)})}, tuple(gates), (), extra=report_payload, artifact_payloads=artifacts)
    return 0 if passed else 1
