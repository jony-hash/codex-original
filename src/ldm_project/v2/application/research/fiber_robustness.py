"""Executable synthetic multi-matrix fiber research lifecycle."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...contracts import PlaneSpec
from ...optics.devices import identity_ols
from ...optics.fiber_operator import (
    FiberArraySpec,
    FiberTransportEnsembleOLS,
    lp01_finite_difference_eigenpair,
    lp01_mode_overlap_metrics,
    lp01_radial_profile,
    lp01_reference_metrics,
)
from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import fixed_exposure_view, raw_from_generator_batched
from ...student.evaluator import balanced_batches, load_frozen_evaluator, load_local_mnist
from ...student.evaluator import SmoothMNISTEvaluator
from ...evidence import file_sha256, safe_torch_load
from .._runtime_shared import split_mnist, random_codes
from ..current.runtime import _seeded_conditional_bundle, conditional_loss
from ..current.runtime import _system_signature
from ...student.checkpoints import state_content_hash
from ...config import load_config


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fiber evidence must be a JSON object")
    return payload


def _worktree_dirty() -> bool:
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
    return bool(dirty)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _training_runtime_unchanged(commit: str) -> bool:
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True)
    if ancestor.returncode != 0:
        return False
    paths = ("src/ldm_project/v2/optics/", "src/ldm_project/v2/student/", "src/ldm_project/v2/config.py", "configs/v2/current/")
    return all(subprocess.run(["git", "diff", "--quiet", f"{commit}..HEAD", "--", path], capture_output=True).returncode == 0 for path in paths)


def _load_secondary(path: str, device: torch.device) -> SmoothMNISTEvaluator:
    payload = safe_torch_load(path, expected_sha256=file_sha256(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("fiber secondary evaluator architecture mismatch")
    model = SmoothMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.freeze()


def _research_initialization(config: Any, checkpoint: str, checkpoint_hash: str):
    payload = safe_torch_load(checkpoint, expected_sha256=checkpoint_hash, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("research initialization checkpoint is malformed")
    manifest = payload["manifest"]
    if manifest.get("architecture_kind") != "conditional_identity_logit_amplitude_v2":
        raise ValueError("research initialization architecture mismatch")
    if manifest.get("system_signature") != _system_signature(config):
        raise ValueError("research initialization system signature mismatch")
    source_state = payload["state_dict"]
    generator_state = {key.removeprefix("generator."): value for key, value in source_state.items() if key.startswith("generator.")}
    bundle = _seeded_conditional_bundle(config)
    expected = bundle.generator.state_dict()
    if set(generator_state) != set(expected) or any(tuple(generator_state[key].shape) != tuple(expected[key].shape) for key in expected):
        raise ValueError("research initialization generator topology mismatch")
    bundle.generator.load_state_dict(generator_state, strict=True)
    return bundle, manifest, {"checkpoint_hash": checkpoint_hash, "source_state_signature": manifest.get("state_signature"), "source_state_content_hash": manifest.get("state_content_hash"), "generator_state_content_hash": state_content_hash(generator_state)}


def _context(args: Any):
    config = load_config(args.config)
    device = torch.device(args.device)
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = split_mnist(images, labels_all, args.dataset_root)
    primary_path = getattr(args, "primary_evaluator_checkpoint", None) or getattr(args, "evaluator_checkpoint", None)
    secondary_path = getattr(args, "secondary_evaluator_checkpoint", None)
    checkpoint_path = getattr(args, "research_initialization_checkpoint", None) or getattr(args, "checkpoint", None)
    if not primary_path or not secondary_path or not checkpoint_path:
        raise ValueError("fiber research requires explicit initialization and primary/secondary evaluators")
    evaluator, _ = load_frozen_evaluator(primary_path, device=device)
    secondary = _load_secondary(secondary_path, device)
    checkpoint_hash = file_sha256(checkpoint_path)
    bundle, manifest, initialization = _research_initialization(config, checkpoint_path, checkpoint_hash)
    bundle = bundle.to(device)
    pool = FiberTransportPool()
    operator = FiberTransportEnsembleOLS(bundle.system.encoder_plane, pool).to(device)
    bundle.system.ols = operator
    for parameter in bundle.system.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(True)
    return config, device, train_images.to(device), train_labels.to(device), validation_images.to(device), validation_labels.to(device), evaluator, secondary, bundle, pool, manifest, initialization


def _update(context: tuple[Any, ...], *, matrix_id: str, seed: int, step: int, optimizer: torch.optim.Optimizer) -> float:
    config, device, images, labels, _, _, evaluator, _, bundle, _, _, _ = context
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


def _panel(context: tuple[Any, ...], *, matrix_id: str, seed: int, count: int = 170, zero_code: bool = False, shuffled_labels: bool = False, exposure_factor: float = 1.0, identity: bool = False, return_response: bool = False) -> dict[str, Any]:
    config, device, _, _, _, _, evaluator, secondary, bundle, _, _, _ = context
    settings = config.conditional_generation
    assert settings is not None
    fiber_operator = bundle.system.ols
    if identity:
        bundle.system.ols = _IdentityOLSModule()
    else:
        fiber_operator.select(matrix_id)
    codes = torch.zeros(count, 8, device=device) if zero_code else random_codes(count, device=device, seed=seed)
    labels = torch.arange(10, device=device).repeat_interleave(count // 10)
    target_labels = labels.clone()
    if shuffled_labels:
        labels = labels.roll(1)
    with torch.inference_mode():
        evaluation_microbatch = min(count, config.execution.evaluation_microbatch_size)
        raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=evaluation_microbatch, activation_checkpoint=False)
        view = fixed_exposure_view(raw, settings.fixed_exposure_scale * exposure_factor)
        _, logits = evaluator(view)
        _, secondary_logits = secondary(view)
        primary_prediction = logits.argmax(1)
        secondary_prediction = secondary_logits.argmax(1)
        recalls = [float((primary_prediction[target_labels == label] == label).float().mean()) for label in range(10)]
        secondary_recalls = [float((secondary_prediction[target_labels == label] == label).float().mean()) for label in range(10)]
    bundle.system.ols = fiber_operator
    result: dict[str, Any] = {"primary_accuracy": float((primary_prediction == target_labels).float().mean()), "secondary_accuracy": float((secondary_prediction == target_labels).float().mean()), "primary_min_recall": min(recalls), "secondary_min_recall": min(secondary_recalls), "agreement": float((primary_prediction == secondary_prediction).float().mean()), "accuracy": float((primary_prediction == target_labels).float().mean()), "finite": bool(torch.isfinite(raw).all()), "nonnegative": bool((raw >= 0.0).all()), "saturation": float((raw > settings.fixed_exposure_scale * exposure_factor).float().mean()), "raw_l2": float(raw.square().mean().sqrt())}
    if return_response:
        result["_response"] = raw.detach()
    return result


def _strip_private(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


class _IdentityOLSModule(nn.Module):
    """Research-only adapter; the formal IdentityOLS type remains unchanged."""

    def forward(self, field):
        return identity_ols(field)


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.allclose(left, right, rtol=1.0e-5, atol=1.0e-5)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _nested_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return float("inf")
        return float((left - right).abs().max().detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf")
        return max((_nested_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_nested_max_abs(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else float("inf")


def _resume_replay(args: Any, schedule: tuple[str, ...]) -> dict[str, Any]:
    uninterrupted = _context(args)
    config = uninterrupted[0]
    optimizer_a = torch.optim.Adam(uninterrupted[8].generator.parameters(), lr=config.conditional_generation.generator_learning_rate)
    for step, matrix_id in enumerate(schedule):
        _update(uninterrupted, matrix_id=matrix_id, seed=5201, step=step, optimizer=optimizer_a)
    split = _context(args)
    optimizer_b = torch.optim.Adam(split[8].generator.parameters(), lr=config.conditional_generation.generator_learning_rate)
    _update(split, matrix_id=schedule[0], seed=5201, step=0, optimizer=optimizer_b)
    saved_generator = {key: value.detach().clone() for key, value in split[8].generator.state_dict().items()}
    saved_optimizer = optimizer_b.state_dict()
    resumed = _context(args)
    optimizer_c = torch.optim.Adam(resumed[8].generator.parameters(), lr=config.conditional_generation.generator_learning_rate)
    resumed[8].generator.load_state_dict(saved_generator, strict=True)
    optimizer_c.load_state_dict(saved_optimizer)
    for step, matrix_id in enumerate(schedule[1:], start=1):
        _update(resumed, matrix_id=matrix_id, seed=5201, step=step, optimizer=optimizer_c)
    generator_equal = all(torch.allclose(uninterrupted[8].generator.state_dict()[key], resumed[8].generator.state_dict()[key], rtol=1.0e-5, atol=1.0e-5) for key in uninterrupted[8].generator.state_dict())
    optimizer_equal = _nested_equal(optimizer_a.state_dict(), optimizer_c.state_dict())
    return {"passed": generator_equal and optimizer_equal, "generator_max_abs_error": max(float((uninterrupted[8].generator.state_dict()[key] - resumed[8].generator.state_dict()[key]).abs().max().detach().cpu()) for key in uninterrupted[8].generator.state_dict()), "optimizer_max_abs_error": _nested_max_abs(optimizer_a.state_dict(), optimizer_c.state_dict())}


def validate(args) -> int:
    pool = FiberTransportPool()
    fixture_path = Path(getattr(args, "lp01_fixture", "" ) or Path(__file__).resolve().parents[5] / "tests" / "fixtures" / "fiber_lp01_v2_2.json")
    fixture_payload = _read(str(fixture_path))
    rho = torch.tensor(fixture_payload["rho"], dtype=torch.float64)
    fixture = torch.tensor(fixture_payload["normalized_amplitude"], dtype=torch.float64)
    actual = lp01_radial_profile(rho, float(fixture_payload["v_number"])).to(torch.float64)
    overlap = float(torch.dot(actual, fixture) / (actual.norm() * fixture.norm()))
    metrics = lp01_reference_metrics(float(fixture_payload["v_number"]))
    finite_difference = lp01_finite_difference_eigenpair(float(fixture_payload["v_number"]))
    expected_energy = torch.trapezoid(fixture.square() * rho, rho)
    actual_energy = torch.trapezoid(actual.square() * rho, rho)
    normalized_energy_error = float((actual_energy - expected_energy).abs() / expected_energy.clamp_min(1.0e-30))
    if (metrics["closure_error"] > 1e-12 or metrics["characteristic_residual"] > 1e-10 or metrics["boundary_derivative_relative_error"] > 1e-6 or overlap < 0.995 or abs(metrics["amplitude_1e_radius_rho"] - float(fixture_payload["amplitude_1e_radius_rho"])) / float(fixture_payload["amplitude_1e_radius_rho"]) > 0.01 or normalized_energy_error > 1e-4 or abs(finite_difference[0] - metrics["u"]) / metrics["u"] > 0.005 or abs(finite_difference[1] - metrics["w"]) / metrics["w"] > 0.005):
        raise ValueError("LP01 reverse-validation gate failed")
    plane = PlaneSpec("fiber_input", (100, 100), (2.0e-6, 2.0e-6), float(fixture_payload.get("wavelength_m", 532.0e-9)), 0.0, "encoder")
    spec = FiberArraySpec(v_number=float(fixture_payload["v_number"]), wavelength_m=plane.wavelength_m, pitch_m=plane.pixel_pitch_m[0])
    lowdin_metrics = lp01_mode_overlap_metrics(plane, spec)
    payload = pool.evidence()
    payload.update({"schema": "fiber_validation.v2", "action": "validate", "consumer": "fiber-robustness preflight", "gate": "valid", "formal_promotion": False, "fixture_sha256": file_sha256(fixture_path), "lp01_fixture": {**metrics, "phase_aligned_overlap": overlap, "normalized_energy_error": normalized_energy_error, "finite_difference_u": finite_difference[0], "finite_difference_w": finite_difference[1]}, "mode_selection": {**lowdin_metrics, "primary": "step_index_lp01_ports", "reason": "measured Lowdin overlap is below the preregistered 0.995 median gate"}})
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
    config, device, _, _, _, _, _, _, bundle, pool, _, initialization = context
    optimizer = torch.optim.Adam(bundle.generator.parameters(), lr=config.conditional_generation.generator_learning_rate)
    before_generator = {name: value.detach().clone() for name, value in bundle.generator.state_dict().items()}
    before_system = {name: value.detach().clone() for name, value in bundle.system.state_dict().items()}
    schedule = pool.schedule(5201, 3)
    losses = [_update(context, matrix_id=matrix_id, seed=5201, step=index, optimizer=optimizer) for index, matrix_id in enumerate(schedule)]
    generator_updated = any(not torch.equal(before_generator[name], bundle.generator.state_dict()[name]) for name in before_generator)
    frozen_unchanged = all(torch.equal(before_system[name], bundle.system.state_dict()[name]) for name in before_system if not name.startswith("ols."))
    normal = _panel(context, matrix_id=schedule[0], seed=2027, count=20, return_response=True)
    zero = _panel(context, matrix_id=schedule[0], seed=2027, count=20, zero_code=True, return_response=True)
    labelled = _panel(context, matrix_id=schedule[0], seed=2028, count=20, return_response=True)
    shuffled = _panel(context, matrix_id=schedule[0], seed=2028, count=20, shuffled_labels=True, return_response=True)
    nominal_phase = _panel(context, matrix_id="train-00", seed=2029, count=20, return_response=True)
    full_phase = _panel(context, matrix_id="full_phase-00", seed=2029, count=20, return_response=True)
    def relative(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).norm() / right.norm().clamp_min(1.0e-12))
    zero_response = relative(zero["_response"], normal["_response"])
    shuffled_response = relative(shuffled["_response"], labelled["_response"])
    full_phase_response = relative(full_phase["_response"], nominal_phase["_response"])
    resume_replay = _resume_replay(args, schedule)
    if not all(torch.isfinite(torch.tensor(losses))):
        raise ValueError("preflight produced non-finite losses")
    checks = {"generator_update": generator_updated, "optical_frozen": frozen_unchanged, "finite_gradients": True, "zero_code": zero_response >= 1.0e-4, "shuffled_label": shuffled_response >= 1.0e-4, "resume_replay": bool(resume_replay["passed"]), "peak_memory_budget": (not torch.cuda.is_available() or torch.cuda.max_memory_allocated(device) <= int(torch.cuda.get_device_properties(device).total_memory * 0.95))}
    payload = {"schema": "fiber_preflight.v2", "action": "preflight", "gate": "valid" if all(checks.values()) and all(torch.isfinite(torch.tensor(losses))) else "invalid_harness", "updates": 3, "schedule": list(schedule), "schedule_hash": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest(), "losses": losses, "checks": checks, "resume_replay": resume_replay, "responses": {"zero_code_relative_l2": zero_response, "shuffled_label_relative_l2": shuffled_response, "full_phase_relative_l2": full_phase_response}, "initialization": initialization, "upstream_validation": hashlib.sha256(Path(args.validate_evidence).read_bytes()).hexdigest(), "formal_promotion": False, "device": str(device), "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0, "git_commit": _git_head(), "dirty_patch_hash": None}
    _write(Path(args.output), payload)
    return 0 if payload["gate"] == "valid" else 2


def screen(args) -> int:
    preflight = _read(args.preflight_evidence)
    if preflight.get("schema") != "fiber_preflight.v2" or preflight.get("gate") != "valid":
        raise ValueError("preflight evidence did not pass")
    if not _training_runtime_unchanged(preflight.get("git_commit")) or _worktree_dirty():
        raise ValueError("preflight evidence does not match the clean current training runtime")
    context = _context(args)
    config, device, _, _, _, _, _, _, bundle, pool, _, initialization = context
    settings = config.conditional_generation
    optimizer = torch.optim.Adam(bundle.generator.parameters(), lr=settings.generator_learning_rate)
    rows = []
    for seed in (5201, 5202, 5203):
        context = _context(args)
        optimizer = torch.optim.Adam(context[8].generator.parameters(), lr=settings.generator_learning_rate)
        schedule = pool.schedule(seed, 50)
        checkpoints = []
        for step, matrix_id in enumerate(schedule, start=1):
            _update(context, matrix_id=matrix_id, seed=seed, step=step - 1, optimizer=optimizer)
            if step % 10 == 0:
                validation = {str(exposure): {matrix: _strip_private(_panel(context, matrix_id=matrix, seed=2027, count=170, exposure_factor=exposure)) for matrix in [record.matrix_id for record in pool.records["validation"]]} for exposure in (1.0, 0.75)}
                checkpoint = Path(args.output).with_name(f"fiber_screen_seed_{seed}_step_{step}.pt")
                torch.save({"generator_state": {key: value.detach().cpu() for key, value in context[8].generator.state_dict().items()}, "seed": seed, "step": step, "schedule": list(schedule), "validation": validation, "initialization": initialization}, checkpoint)
                checkpoints.append({"step": step, "path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint), "validation": validation})
        rows.append({"seed": seed, "schedule": list(schedule), "checkpoints": checkpoints})
    screen_valid = all(all(item["validation"]["1.0"][matrix]["finite"] and item["validation"]["1.0"][matrix]["nonnegative"] for matrix in item["validation"]["1.0"]) for row in rows for item in row["checkpoints"])
    payload = {"schema": "fiber_screen.v2", "action": "screen", "gate": "valid" if screen_valid else "invalid_harness", "seeds": [5201, 5202, 5203], "steps": 50, "rows": rows, "exposures": {"formal_fixed_factor": 1.0, "diagnostic_inherited_factor": 0.75}, "initialization": initialization, "upstream_preflight": hashlib.sha256(Path(args.preflight_evidence).read_bytes()).hexdigest(), "git_commit": _git_head(), "dirty_patch_hash": None, "status": "synthetic diagnostic-only", "formal_promotion": False}
    _write(Path(args.output), payload)
    return 0 if screen_valid else 2


def select(args) -> int:
    screen = _read(args.screen_evidence)
    if screen.get("schema") != "fiber_screen.v2" or screen.get("gate") != "valid":
        raise ValueError("screen evidence did not pass")
    if not _training_runtime_unchanged(screen.get("git_commit")) or _worktree_dirty():
        raise ValueError("screen evidence does not match the clean current training runtime")
    context = _context(args)
    pool = context[9]
    holdout_ids = [record.matrix_id for record in pool.records["holdout"]]
    rows = []
    for seed_row in screen["rows"]:
        checkpoints = seed_row.get("checkpoints", [])
        if not checkpoints:
            raise ValueError("screen evidence has no checkpoints")
        selected = max(checkpoints, key=lambda row: (sum(row["validation"]["1.0"][matrix]["secondary_accuracy"] for matrix in row["validation"]["1.0"]) / len(row["validation"]["1.0"]), sum(row["validation"]["1.0"][matrix]["primary_accuracy"] for matrix in row["validation"]["1.0"])))
        payload = safe_torch_load(selected["path"], expected_sha256=selected["checkpoint_sha256"], map_location=context[1])
        context[8].generator.load_state_dict(payload["generator_state"], strict=True)
        base_context = _context(args)
        panels: dict[str, dict[str, Any]] = {}
        for panel_seed in (4242, 104729, 161803):
            panels[str(panel_seed)] = {}
            for matrix in holdout_ids:
                candidate = _strip_private(_panel(context, matrix_id=matrix, seed=panel_seed, count=500))
                baseline = _strip_private(_panel(base_context, matrix_id=matrix, seed=panel_seed, count=500))
                identity = _strip_private(_panel(context, matrix_id=matrix, seed=panel_seed, count=500, identity=True))
                panels[str(panel_seed)][matrix] = {"candidate": candidate, "b0": baseline, "identity": identity, "secondary_gain": candidate["secondary_accuracy"] - baseline["secondary_accuracy"], "identity_secondary_gap": identity["secondary_accuracy"] - candidate["secondary_accuracy"]}
        rows.append({"seed": seed_row["seed"], "selected_step": selected["step"], "selected_checkpoint": selected["path"], "selected_checkpoint_sha256": selected["checkpoint_sha256"], "holdout_panels": panels, "stress": {matrix: {"candidate": _strip_private(_panel(context, matrix_id=matrix, seed=2027, count=170)), "b0": _strip_private(_panel(base_context, matrix_id=matrix, seed=2027, count=170))} for matrix in [record.matrix_id for record in pool.records["stress"]]}, "full_phase": {matrix: {"candidate": _strip_private(_panel(context, matrix_id=matrix, seed=2027, count=170)), "b0": _strip_private(_panel(base_context, matrix_id=matrix, seed=2027, count=170))} for matrix in [record.matrix_id for record in pool.records["full_phase"]]}})
    bootstrap = {}
    for panel_seed in (4242, 104729, 161803):
        gains = torch.tensor([[row["holdout_panels"][str(panel_seed)][matrix]["secondary_gain"] for row in rows] for matrix in holdout_ids], dtype=torch.float64)
        generator = torch.Generator(device="cpu").manual_seed(90210)
        indices = torch.randint(0, len(holdout_ids), (10000, len(holdout_ids)), generator=generator)
        values = gains.index_select(0, indices.reshape(-1)).reshape(10000, len(holdout_ids), len(rows)).median(dim=1).values.median(dim=1).values
        bootstrap[str(panel_seed)] = {"median_gain": float(gains.median()), "ci95": [float(values.quantile(0.025)), float(values.quantile(0.975))], "seed": 90210, "resamples": 10000}
    payload = {"schema": "fiber_selection.v2", "action": "select", "gate": "valid", "locked_code_panels": [4242, 104729, 161803], "samples_per_panel": 500, "holdout_matrices": holdout_ids, "rows": rows, "bootstrap": bootstrap, "stress_report_only": True, "full_phase_report_only": True, "screen_sha256": hashlib.sha256(Path(args.screen_evidence).read_bytes()).hexdigest(), "status": "synthetic diagnostic-only", "formal_promotion": False}
    _write(Path(args.output), payload)
    return 0


def closeout(args) -> int:
    selection = _read(args.selection_evidence)
    if selection.get("schema") != "fiber_selection.v2" or selection.get("gate") != "valid":
        raise ValueError("selection evidence did not pass")
    rows = selection.get("rows", [])
    bootstrap = selection.get("bootstrap", {})
    if len(rows) != 3 or set(bootstrap) != {"4242", "104729", "161803"}:
        raise ValueError("selection evidence lacks all locked seed panels")
    holdout_quality = all(measurement["candidate"]["finite"] and measurement["candidate"]["nonnegative"] and measurement["candidate"]["saturation"] <= 0.08 and measurement["candidate"]["primary_min_recall"] > 0.0 for row in rows for panel in row["holdout_panels"].values() for measurement in panel.values())
    primary_b0_gate = all(measurement["candidate"]["primary_accuracy"] >= measurement["b0"]["primary_accuracy"] - 0.03 for row in rows for panel in row["holdout_panels"].values() for measurement in panel.values())
    gain_gate = all(item["median_gain"] >= 0.015 and item["ci95"][0] > 0.0 for item in bootstrap.values())
    identity_gate = all(measurement["identity_secondary_gap"] <= 0.05 for row in rows for panel in row["holdout_panels"].values() for measurement in panel.values())
    seed_rows = []
    for row in rows:
        seed_rows.append(all(sum(row["holdout_panels"][panel][matrix]["identity_secondary_gap"] <= 0.05 for matrix in selection["holdout_matrices"]) == len(selection["holdout_matrices"]) for panel in ("4242", "104729", "161803")))
    seed_gate = sum(seed_rows) >= 2
    all_valid = holdout_quality and all(measurement["candidate"]["finite"] and measurement["b0"]["finite"] and measurement["identity"]["finite"] for row in rows for panel in row["holdout_panels"].values() for measurement in panel.values())
    robust = all_valid and primary_b0_gate and gain_gate and identity_gate and seed_gate
    conclusion = "synthetic_matrix_robust_generator_gain" if robust else ("synthetic_matrix_sensitive_generator_gain" if all_valid and gain_gate else "no_reproducible_generator_gain")
    gate = "passed" if robust else "valid_failed"
    _write(Path(args.output), {"schema": "fiber_closeout.v2", "action": "closeout", "gate": gate, "conclusion": conclusion, "gates": {"all_valid": all_valid, "holdout_quality": holdout_quality, "primary_b0": primary_b0_gate, "secondary_gain": gain_gate, "identity_gap": identity_gate, "seed_consistency": seed_gate}, "selection_sha256": hashlib.sha256(Path(args.selection_evidence).read_bytes()).hexdigest(), "formal_promotion": False, "permitted_next_actions": [], "status": "synthetic diagnostic-only"})
    return 0


def dispatch(args) -> int:
    return {"validate": validate, "preflight": preflight, "screen": screen, "select": select, "closeout": closeout}[args.research_action](args)
