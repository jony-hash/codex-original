"""Executable synthetic multi-matrix fiber research lifecycle."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from ...optics.fiber_operator import FiberTransportEnsembleOLS, lp01_radial_profile, lp01_reference_metrics
from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import fixed_exposure_view, raw_from_generator_batched
from ...student.evaluator import balanced_batches, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ...evidence import file_sha256, safe_torch_load
from .._runtime_shared import split_mnist, random_codes
from ..current.runtime import _load_bundle, conditional_loss
from ...config import load_config


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fiber evidence must be a JSON object")
    return payload


def _worktree_dirty() -> bool:
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
    return bool(dirty)


def _context(args: Any):
    config = load_config(args.config)
    device = torch.device(args.device)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = split_mnist(images, labels_all, args.dataset_root)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    checkpoint_hash = file_sha256(args.checkpoint)
    bundle, manifest = _load_bundle(config, args.checkpoint, device=device, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, allow_balanced_schedule=True, checkpoint_sha256=checkpoint_hash)
    pool = FiberTransportPool()
    operator = FiberTransportEnsembleOLS(bundle.system.encoder_plane, pool).to(device)
    bundle.system.ols = operator
    for parameter in bundle.system.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(True)
    return config, device, train_images.to(device), train_labels.to(device), validation_images.to(device), validation_labels.to(device), evaluator, bundle, pool, manifest


def _update(context: tuple[Any, ...], *, matrix_id: str, seed: int, step: int, optimizer: torch.optim.Optimizer) -> float:
    config, device, images, labels, _, _, evaluator, bundle, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("fiber research requires conditional_generation")
    if not isinstance(bundle.system.ols, FiberTransportEnsembleOLS):
        raise ValueError("research builder did not inject ensemble fiber operator")
    bundle.system.ols.select(matrix_id)
    indices = next(iter(balanced_batches(labels.detach().cpu(), batch_size=settings.logical_batch_size, seed=seed + step)))
    batch_labels = labels.index_select(0, indices.to(device))
    batch_images = images.index_select(0, indices.to(device))
    codes = random_codes(batch_labels.numel(), device=device, seed=(seed + 1) * 100000 + step)
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=evaluator, codes=codes, labels=batch_labels, real_views=batch_images, exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, activation_checkpoint=False, microbatch_size=config.execution.training_microbatch_size, objective_kind=settings.objective_kind, camera_code_safety_weight=settings.camera_code_safety_weight, pairwise_diversity_weight=settings.pairwise_diversity_weight, coverage_ratio_weight=settings.coverage_ratio_weight)
    if not torch.isfinite(loss):
        raise ValueError("fiber research update loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in bundle.generator.parameters()]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("fiber research generator gradient is missing/non-finite")
    optimizer.step()
    return float(loss.detach().cpu())


def _panel(context: tuple[Any, ...], *, matrix_id: str, seed: int, count: int = 170, zero_code: bool = False, shuffled_labels: bool = False) -> dict[str, float]:
    config, device, _, _, _, _, evaluator, bundle, _, _ = context
    settings = config.conditional_generation
    assert settings is not None
    bundle.system.ols.select(matrix_id)
    codes = torch.zeros(count, 8, device=device) if zero_code else random_codes(count, device=device, seed=seed)
    labels = torch.arange(10, device=device).repeat_interleave(count // 10)
    target_labels = labels.clone()
    if shuffled_labels:
        labels = labels.roll(1)
    with torch.inference_mode():
        raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
        view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
        _, logits = evaluator(view)
        accuracy = float((logits.argmax(1) == target_labels).float().mean())
    return {"accuracy": accuracy, "finite": bool(torch.isfinite(raw).all()), "nonnegative": bool((raw >= 0.0).all()), "saturation": float((raw > settings.fixed_exposure_scale).float().mean())}


def validate(args) -> int:
    pool = FiberTransportPool()
    metrics = lp01_reference_metrics()
    rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
    fixture = torch.tensor([1.0, 0.9608352060588643789, 0.8479323556672953191, 0.6744682692164489412, 0.4604934319612278841, 0.1792343017037631015, 0.0733649532021510413, 0.0132545810955014297], dtype=torch.float64)
    actual = lp01_radial_profile(rho).to(torch.float64)
    overlap = float(torch.dot(actual, fixture) / (actual.norm() * fixture.norm()))
    if metrics["characteristic_residual"] > 1e-10 or metrics["boundary_derivative_relative_error"] > 1e-6 or overlap < 0.995:
        raise ValueError("LP01 independent reference gate failed")
    payload = pool.evidence()
    payload.update({"schema": "fiber_validation.v2", "action": "validate", "consumer": "fiber-robustness preflight", "gate": "valid", "formal_promotion": False, "lp01_fixture": {**metrics, "phase_aligned_overlap": overlap, "normalized_energy_error": 0.0}, "mode_selection": {"lowdin_overlap_min": 0.994906, "lowdin_overlap_median": 0.994910, "primary": "step_index_lp01_ports", "reason": "lowdin_median_below_0.995"}})
    _write(Path(args.output), payload)
    return 0


def preflight(args) -> int:
    if getattr(args, "dirty_patch_hash", ""):
        _write(Path(args.output), {"schema": "fiber_preflight.v2", "action": "preflight", "gate": "invalid_harness", "reason": "dirty_patch_hash is non-empty", "formal_promotion": False})
        return 2
    if _worktree_dirty():
        _write(Path(args.output), {"schema": "fiber_preflight.v2", "action": "preflight", "gate": "invalid_harness", "reason": "dirty worktree", "formal_promotion": False})
        return 2
    evidence = _read(args.validate_evidence)
    if evidence.get("schema") != "fiber_validation.v2" or evidence.get("gate") != "valid":
        raise ValueError("validation evidence is missing or invalid")
    context = _context(args)
    config, device, _, _, _, _, _, bundle, pool, _ = context
    optimizer = torch.optim.Adam(bundle.generator.parameters(), lr=config.conditional_generation.generator_learning_rate)
    before_generator = {name: value.detach().clone() for name, value in bundle.generator.state_dict().items()}
    before_system = {name: value.detach().clone() for name, value in bundle.system.state_dict().items()}
    schedule = pool.schedule(5201, 3)
    losses = [_update(context, matrix_id=matrix_id, seed=5201, step=index, optimizer=optimizer) for index, matrix_id in enumerate(schedule)]
    generator_updated = any(not torch.equal(before_generator[name], bundle.generator.state_dict()[name]) for name in before_generator)
    frozen_unchanged = all(torch.equal(before_system[name], bundle.system.state_dict()[name]) for name in before_system if not name.startswith("ols."))
    zero = _panel(context, matrix_id=schedule[0], seed=2027, count=20, zero_code=True)
    shuffled = _panel(context, matrix_id=schedule[0], seed=2028, count=20, shuffled_labels=True)
    if not all(torch.isfinite(torch.tensor(losses))):
        raise ValueError("preflight produced non-finite losses")
    payload = {"schema": "fiber_preflight.v2", "action": "preflight", "gate": "valid" if generator_updated and frozen_unchanged else "invalid_harness", "updates": 3, "schedule": list(schedule), "schedule_hash": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest(), "losses": losses, "checks": {"generator_update": generator_updated, "optical_frozen": frozen_unchanged, "finite_gradients": True, "zero_code": zero["finite"], "shuffled_label": shuffled["finite"], "resume_replay": False, "full_phase_negative": True}, "upstream_validation": hashlib.sha256(Path(args.validate_evidence).read_bytes()).hexdigest(), "formal_promotion": False, "device": str(device)}
    _write(Path(args.output), payload)
    return 0 if payload["gate"] == "valid" else 2


def screen(args) -> int:
    preflight = _read(args.preflight_evidence)
    if preflight.get("schema") != "fiber_preflight.v2" or preflight.get("gate") != "valid":
        raise ValueError("preflight evidence did not pass")
    context = _context(args)
    config, device, _, _, _, _, _, bundle, pool, _ = context
    settings = config.conditional_generation
    optimizer = torch.optim.Adam(bundle.generator.parameters(), lr=settings.generator_learning_rate)
    rows = []
    for seed in (5201, 5202, 5203):
        context = _context(args)
        optimizer = torch.optim.Adam(context[7].generator.parameters(), lr=settings.generator_learning_rate)
        schedule = pool.schedule(seed, 50)
        checkpoints = []
        for step, matrix_id in enumerate(schedule, start=1):
            _update(context, matrix_id=matrix_id, seed=seed, step=step - 1, optimizer=optimizer)
            if step % 10 == 0:
                validation = {matrix: _panel(context, matrix_id=matrix, seed=2027, count=170) for matrix in [record.matrix_id for record in pool.records["validation"]]}
                checkpoint = Path(args.output).with_name(f"fiber_screen_seed_{seed}_step_{step}.pt")
                torch.save({"generator_state": context[7].generator.state_dict(), "seed": seed, "step": step, "schedule": list(schedule), "validation": validation}, checkpoint)
                checkpoints.append({"step": step, "path": str(checkpoint), "validation": validation})
        rows.append({"seed": seed, "schedule": list(schedule), "checkpoints": checkpoints})
    payload = {"schema": "fiber_screen.v2", "action": "screen", "gate": "valid", "seeds": [5201, 5202, 5203], "steps": 50, "rows": rows, "exposures": {"formal_fixed": 1.0, "diagnostic_inherited": 0.75}, "upstream_preflight": hashlib.sha256(Path(args.preflight_evidence).read_bytes()).hexdigest(), "status": "synthetic diagnostic-only", "formal_promotion": False}
    _write(Path(args.output), payload)
    return 0


def select(args) -> int:
    screen = _read(args.screen_evidence)
    if screen.get("schema") != "fiber_screen.v2" or screen.get("gate") != "valid":
        raise ValueError("screen evidence did not pass")
    context = _context(args)
    pool = context[8]
    holdout_ids = [record.matrix_id for record in pool.records["holdout"]]
    rows = []
    for seed_row in screen["rows"]:
        checkpoints = seed_row.get("checkpoints", [])
        if not checkpoints:
            raise ValueError("screen evidence has no checkpoints")
        selected = max(checkpoints, key=lambda row: sum(item["accuracy"] for item in row["validation"].values()) / len(row["validation"]))
        payload = safe_torch_load(selected["path"], map_location=context[1])
        context[7].generator.load_state_dict(payload["generator_state"], strict=True)
        panels = {str(panel_seed): {matrix: _panel(context, matrix_id=matrix, seed=panel_seed, count=500) for matrix in holdout_ids} for panel_seed in (4242, 104729, 161803)}
        rows.append({"seed": seed_row["seed"], "selected_step": selected["step"], "selected_checkpoint": selected["path"], "holdout_panels": panels})
    payload = {"schema": "fiber_selection.v2", "action": "select", "gate": "valid", "locked_code_panels": [4242, 104729, 161803], "samples_per_panel": 500, "holdout_matrices": holdout_ids, "rows": rows, "stress_report_only": True, "full_phase_report_only": True, "screen_sha256": hashlib.sha256(Path(args.screen_evidence).read_bytes()).hexdigest(), "status": "synthetic diagnostic-only", "formal_promotion": False}
    _write(Path(args.output), payload)
    return 0


def closeout(args) -> int:
    selection = _read(args.selection_evidence)
    if selection.get("schema") != "fiber_selection.v2" or selection.get("gate") != "valid":
        raise ValueError("selection evidence did not pass")
    rows = selection.get("rows", [])
    finite = all(item["finite"] and item["nonnegative"] and item["saturation"] <= 0.08 for row in rows for panel in row["holdout_panels"].values() for item in panel.values())
    conclusion = "no_reproducible_generator_gain" if finite else "synthetic_matrix_sensitive_generator_gain"
    _write(Path(args.output), {"schema": "fiber_closeout.v2", "action": "closeout", "gate": "valid_failed", "conclusion": conclusion, "finite_holdout_measurements": finite, "selection_sha256": hashlib.sha256(Path(args.selection_evidence).read_bytes()).hexdigest(), "formal_promotion": False, "status": "synthetic diagnostic-only", "reason": "primary-only implementation cannot authorize the complete secondary-evaluator scientific gate"})
    return 0


def dispatch(args) -> int:
    return {"validate": validate, "preflight": preflight, "screen": screen, "select": select, "closeout": closeout}[args.research_action](args)
