"""Compare paired reconstruction targets with and without A4700 energy matching."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


TASK_ID = "M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_energy_matched_target_comparison_report.v1"
CHECKPOINT_SCHEMA = "m0_generator_energy_matched_target_checkpoint.v1"
DEFAULT_STROKE = Path("outputs/20261008_m0_generator_stroke_fidelity_comparison_v1/runs/attempt_08")
DEFAULT_OUTPUT = Path("outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_01")
SOURCE_STEP = 4_700
ADDED_UPDATES = 2_000
FINAL_STEP = SOURCE_STEP + ADDED_UPDATES
FIT_COUNT = 200
CONFIRM_COUNT = 300
MEASURE_COUNT = FIT_COUNT + CONFIRM_COUNT
BATCH_SIZE = 10
BATCH_COUNT = FIT_COUNT // BATCH_SIZE
MICROBATCH_SIZE = 2
SSIM_WEIGHT = 0.1
RTOL, ATOL = 2.0e-5, 2.0e-7
DISK_RESERVE = 1_610_612_736
ARM_A = "A_unscaled_target"
ARM_B = "B_energy_matched_target"
ARM_NAMES = (ARM_A, ARM_B)
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
TRAIN_EVALUATORS = ("E_S", "E_I", "E_C")
NUMERIC_MODE = dict(class9.NUMERIC_MODE)


class EnergyTargetError(RuntimeError):
    """Invalid task inputs or failed production invariant."""


class TrainingInterrupted(RuntimeError):
    """An interrupt was received after a durable training checkpoint."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnergyTargetError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _normalize_population_paths(root: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(population)
    for field in ("source_generation_dir", "source_checkpoint", "source_a4700_readout"):
        if field in normalized:
            value = normalized[field]
            _require(isinstance(value, str) and bool(value), f"source population {field} must be an explicit path")
            normalized[field] = _resolve(root, value).as_posix()
    paths = normalized.get("source_input_paths")
    if isinstance(paths, Mapping):
        normalized["source_input_paths"] = {
            str(name): _resolve(root, value).as_posix()
            for name, value in paths.items()
        }
    return normalized


def _validate_reconstructed_population(
    root: Path,
    saved_population: Mapping[str, Any],
    reconstructed_population: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_population_paths(root, saved_population)
    _require(
        reconstructed_population == normalized,
        "reconstructed stroke population differs from explicit attempt_08 population",
    )
    return normalized


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _write_torch_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()


def _tensor_tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _readout_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    tensor_fields = sorted(key for key, value in left.items() if torch.is_tensor(value))
    _require(tensor_fields == sorted(key for key, value in right.items() if torch.is_tensor(value)), "A4700 saved readout fields differ")
    fields: dict[str, Any] = {}
    for key in tensor_fields:
        a, b = left[key], right[key]
        _require(a.shape == b.shape and a.dtype == b.dtype, f"A4700 saved readout shape/dtype differs for {key}")
        _require(torch.equal(a, b), f"stroke task and source checkpoint A4700 readouts differ for {key}")
        fields[key] = {"passed": True, "shape": list(a.shape), "dtype": str(a.dtype)}
    for key in set(left) - set(tensor_fields):
        _require(key in right and left[key] == right[key], f"A4700 readout metadata differs for {key}")
    return {"executed": True, "passed": True, "fields": fields}


def _pixel_area(readout: Mapping[str, Any], key: str) -> tuple[float, list[float]]:
    metadata = readout.get("plane_metadata", {})
    plane = metadata.get(key, {}).get("plane", {}) if key != "camera_plane" else metadata.get("camera_plane", {})
    pitch = plane.get("pixel_pitch_m")
    _require(isinstance(pitch, list) and len(pitch) == 2, f"A4700 readout lacks {key} physical pixel pitch")
    pitch_pair = [float(value) for value in pitch]
    _require(all(math.isfinite(value) and value > 0 for value in pitch_pair), f"A4700 {key} pixel pitch is invalid")
    return pitch_pair[0] * pitch_pair[1], pitch_pair


def _derive_energy_scales(inputs: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    readout = inputs["readout"]
    _require(torch.equal(readout["source_indices"], inputs["rows"]), "A4700 readout rows differ before scale construction")
    _require(torch.equal(readout["labels"], inputs["labels"]), "A4700 readout labels differ before scale construction")
    target_area, target_pitch = _pixel_area(readout, "input_slm_field")
    output_area, output_pitch = _pixel_area(readout, "camera_plane")
    target = readout["target_slm_intensity"][:FIT_COUNT].to(torch.float64)
    output = readout["raw_cmos"][:FIT_COUNT].to(torch.float64)
    target_integral = target.flatten(1).sum(1) * target_area
    output_integral = output.flatten(1).sum(1) * output_area
    _require(bool(torch.isfinite(target_integral).all() and torch.isfinite(output_integral).all()), "fit target or CMOS integrated energy is nonfinite")
    _require(bool((target_integral > 0).all() and (output_integral > 0).all()), "fit target or A4700 CMOS integrated energy must be positive")
    scales = output_integral / target_integral
    _require(bool(torch.isfinite(scales).all() and (scales > 0).all()), "fixed target-energy scale coefficients are invalid")
    evidence = {
        "formula": "(sum(raw_cmos)*camera_pixel_area)/(sum(target_slm_intensity)*slm_pixel_area)",
        "fit_only": True,
        "source_rows": inputs["rows"][:FIT_COUNT].tolist(),
        "labels": inputs["labels"][:FIT_COUNT].tolist(),
        "target_integral": target_integral.tolist(),
        "A4700_cmos_integral": output_integral.tolist(),
        "scale_coefficients": scales.tolist(),
        "coefficient_summary": {
            "mean": float(scales.mean()),
            "min": float(scales.min()),
            "max": float(scales.max()),
        },
        "target_pixel_pitch_m": target_pitch,
        "camera_pixel_pitch_m": output_pitch,
        "target_pixel_area_m2": target_area,
        "camera_pixel_area_m2": output_area,
        "coefficient_is_detached_and_fixed": True,
    }
    return scales.contiguous(), evidence


def _load_population(root: Path, stroke_dir: Path) -> dict[str, Any]:
    population_path = stroke_dir / "input_population.json"
    saved_readout_path = stroke_dir / "readouts_start.pt"
    _require(population_path.is_file() and saved_readout_path.is_file(), "stroke attempt_08 population or A4700 readout is missing")
    saved_population = json.loads(population_path.read_text(encoding="utf-8"))
    _require(saved_population.get("task_id") == stroke.TASK_ID, "explicit source population is not the frozen stroke-fidelity attempt")
    generation_dir = _resolve(root, saved_population["source_generation_dir"])
    inputs = stroke._load_population(root, generation_dir)
    saved_population = _validate_reconstructed_population(root, saved_population, inputs["population"])
    saved_readout = torch.load(saved_readout_path, map_location="cpu", weights_only=True, mmap=True)
    replay_check = _readout_equal(saved_readout, inputs["readout"])
    for key, expected in (("source_indices", inputs["rows"]), ("labels", inputs["labels"])):
        _require(torch.equal(saved_readout[key].to(torch.long), expected.to(torch.long)), f"stroke A4700 readout {key} differs from canonical TRAIN mapping")
    source = inputs["source"]
    _require(source.get("completed_updates") == SOURCE_STEP and source.get("optimizer_step") == SOURCE_STEP, "A4700 source checkpoint step differs")
    _require(source.get("next_batch_index") == 0 and source.get("next_batch_cursor") == 0, "A4700 source cursor is not zero")
    _require(tuple(source["trainable_parameter_state"]["generator.phase"].shape) == (340_000,), "A4700 large Generator parameter layout differs")
    _require(int(float(source["optimizer_state"]["state"][0]["step"].item())) == SOURCE_STEP, "A4700 Generator Adam step differs")
    scales, scale_evidence = _derive_energy_scales(inputs)
    population = copy.deepcopy(saved_population)
    population.update(
        {
            "task_id": TASK_ID,
            "energy_scale_source_task": stroke.TASK_ID,
            "energy_scale_fit_positions": [0, FIT_COUNT],
            "energy_scale_formula": scale_evidence["formula"],
            "energy_scale_coefficients": scale_evidence["scale_coefficients"],
            "energy_scale_source_rows": scale_evidence["source_rows"],
            "energy_scale_labels": scale_evidence["labels"],
            "energy_scale_pixel_pitch_m": {
                "slm": scale_evidence["target_pixel_pitch_m"],
                "camera": scale_evidence["camera_pixel_pitch_m"],
            },
            "energy_scale_unbiased_or_detached": "fixed_cpu_float64_no_gradient",
            "updates_per_arm": ADDED_UPDATES,
            "final_generator_adam_step": FINAL_STEP,
            "energy_matched_target": "s_i * original_target_slm_intensity; only paired MSE+SSIM reconstruction target",
            "SSIM_both_arms": True,
        }
    )
    inputs.update(
        {
            "population": population,
            "readout": saved_readout,
            "scales": scales,
            "scale_evidence": scale_evidence,
            "saved_readout_replay": replay_check,
            "fit_targets_original": saved_readout["target_slm_intensity"][:FIT_COUNT].clone(),
            "fit_targets_scaled": (
                saved_readout["target_slm_intensity"][:FIT_COUNT].to(torch.float64)
                * scales.view(FIT_COUNT, 1, 1, 1)
            ).to(torch.float32),
            "stroke_dir": stroke_dir,
        }
    )
    return inputs


def _targets_for_batch(inputs: Mapping[str, Any], arm: str, batch_index: int, device: torch.device) -> torch.Tensor:
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    selected = inputs["fit_targets_original"] if arm == ARM_A else inputs["fit_targets_scaled"]
    targets = selected[start:stop].to(device=device, dtype=torch.float32)
    _require(targets.shape == (BATCH_SIZE, 1, 200, 200) and not targets.requires_grad, "paired target mapping is not a fixed ten-row tensor")
    return targets


def _fit_inputs(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {"images": inputs["fit_images"], "labels": inputs["fit_labels"], "source_indices": inputs["rows"][:FIT_COUNT]}


def _new_endpoint(inputs: Mapping[str, Any], device: torch.device):
    return stroke._new_endpoint(inputs, device)


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    arm: str,
    local_update: int,
    *,
    verify_vjp: bool = False,
    override_targets: torch.Tensor | None = None,
) -> dict[str, Any]:
    batch_index = local_update % BATCH_COUNT
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    device = consumer.device
    fit = _fit_inputs(inputs)
    images = fit["images"][start:stop].to(device)
    labels_cpu = fit["labels"][start:stop].cpu()
    labels = labels_cpu.to(device)
    targets = _targets_for_batch(inputs, arm, batch_index, device) if override_targets is None else override_targets.to(device=device, dtype=torch.float32)
    scales = inputs["scales"][start:stop]
    fixed_targets = targets.detach()
    _require(torch.equal(fixed_targets, targets), "paired reconstruction target unexpectedly carries gradient")
    powers, losses, gradients, combined = stroke._logical_batch_gradients(
        consumer,
        parameters,
        evaluators,
        images,
        labels,
        structural=True,
        reconstruction_targets=fixed_targets,
        verify_ssim_vjp=verify_vjp,
    )
    phase = parameters["generator.phase"]
    generator_gradient = combined["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the Generator-only optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(float(optimizer.state[phase]["step"].item()))
    _require(step == SOURCE_STEP + local_update + 1, "task Adam step is not source step plus task-local update")
    delta = phase.detach() - before
    old_count = 20_000
    record = {
        "task_local_update": local_update + 1,
        "optimizer_step": step,
        "arm": arm,
        "batch_index": batch_index,
        "fit_positions": list(range(start, stop)),
        "source_row_indices": fit["source_indices"][start:stop].tolist(),
        "labels": labels_cpu.tolist(),
        "target_scale_coefficients": scales.tolist(),
        "target_rule": "original_y" if arm == ARM_A else "fixed_A4700_energy_ratio_times_y",
        "target_scale_applied_only_to_reconstruction": True,
        "three_evaluator_semantic_objective": "joint_normalized_equal_weight_E_S_E_I_E_C_true_class_CE",
        "raw_mse_training_target": losses["raw_mse"],
        "ssim_loss_training_target": losses["ssim_loss"],
        "ssim_weight": SSIM_WEIGHT,
        "reconstruction_objective_training_target": losses["reconstruction_objective"],
        "raw_mse_original_target_is_final_metric_only": True,
        "ce_by_evaluator": {name: losses[f"{name}_ce"] for name in TRAIN_EVALUATORS},
        "l_dist": losses["l_dist"],
        "l_op": losses["l_op"],
        "phase_gradient_rms": float(generator_gradient.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "original_100x100_gradient_norm": capacity._vector_norm(generator_gradient[:old_count]),
        "added_400x400_gradient_norm": capacity._vector_norm(generator_gradient[old_count:]),
        "ssim_gradient_norm": capacity._vector_norm(gradients["ssim"]["generator.phase"]),
        "original_100x100_displacement_rms": float(delta[:old_count].detach().to(torch.float64).square().mean().sqrt().cpu()),
        "added_400x400_displacement_rms": float(delta[old_count:].detach().to(torch.float64).square().mean().sqrt().cpu()),
        "generator_displacement_rms": float(delta.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "five_plane_power_mean": powers.detach().to(torch.float64).mean(0).cpu().tolist(),
        "all_finite": bool(torch.isfinite(phase).all().item()),
    }
    _require(record["all_finite"] and all(parameter.grad is None for parameter in parameters.values()), "training left a nonfinite phase or populated parameter gradient")
    return record


def _independent_reconstruction_delta_vjp(
    consumer: Any,
    parameters: Mapping[str, torch.nn.Parameter],
    images: torch.Tensor,
    original_targets: torch.Tensor,
    scaled_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    accumulated = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
        stop = start + MICROBATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw = result.raw_intensity
        original_objective = stroke.reconstruction_terms(raw, original_targets[start:stop], structural=True)["objective"]
        scaled_objective = stroke.reconstruction_terms(raw, scaled_targets[start:stop], structural=True)["objective"]
        original_pixel_gradient = torch.autograd.grad(original_objective, raw, retain_graph=True)[0]
        scaled_pixel_gradient = torch.autograd.grad(scaled_objective, raw, retain_graph=True)[0]
        difference = scaled_pixel_gradient - original_pixel_gradient
        parameter_gradients = torch.autograd.grad(
            raw,
            tuple(parameters.values()),
            grad_outputs=difference,
            retain_graph=False,
            allow_unused=True,
        )
        fraction = (stop - start) / BATCH_SIZE
        for (name, parameter), gradient in zip(parameters.items(), parameter_gradients, strict=True):
            if gradient is not None:
                accumulated[name] += fraction * gradient.detach()
        del result, raw, original_objective, scaled_objective, original_pixel_gradient, scaled_pixel_gradient, difference, parameter_gradients
    return accumulated


def _run_trial(arm: str, inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], *, updates: int) -> dict[str, Any]:
    consumer, optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    before_encoder = parameters["encoder.phase"].detach().cpu().clone()
    before_frozen = capacity._frozen_runtime_state(consumer)
    before_evaluators = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    before_source_adam = copy.deepcopy(inputs["source"]["optimizer_state"])
    capacity._restore_rng_state(inputs["source"]["rng_state"])
    history = []
    for index in range(updates):
        history.append(_train_step(consumer, optimizer, parameters, evaluators, inputs, arm, index, verify_vjp=index == 0))
    _require(torch.equal(before_encoder, parameters["encoder.phase"].detach().cpu()), f"{arm} disposable Encoder changed")
    capacity._assert_state_equal(before_frozen, capacity._frozen_runtime_state(consumer), f"{arm} disposable frozen runtime")
    for name, model in evaluators.items():
        capacity._assert_state_equal(before_evaluators[name], capacity._state_snapshot(model), f"{arm} disposable evaluator {name}")
    _require(all(parameter.grad is None for parameter in parameters.values()), f"{arm} disposable trial left parameter gradients")
    final_step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(final_step == SOURCE_STEP + updates, f"{arm} disposable trial did not perform its declared Adam steps")
    _require(_tensor_tree_equal(before_source_adam, inputs["source"]["optimizer_state"]), "disposable update aliased the source Adam state")
    return {
        "phase": parameters["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "history": history,
        "new_plane_gradient_norm": float(history[0]["added_400x400_gradient_norm"]),
        "old_plane_gradient_norm": float(history[0]["original_100x100_gradient_norm"]),
        "target_scales": inputs["scales"].clone(),
    }


def _target_control(evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    return stroke._target_positive_control(evaluators, inputs)


def _wrong_label_control(inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    fit = _fit_inputs(inputs)
    images = fit["images"][:BATCH_SIZE].to(consumer.device)
    labels = fit["labels"][:BATCH_SIZE].to(consumer.device)
    original = _targets_for_batch(inputs, ARM_A, 0, consumer.device)
    rotated = torch.roll(labels, shifts=1)
    _require(torch.bincount(rotated.cpu(), minlength=10).tolist() == [1] * 10, "wrong-label control is not balanced")
    _, _, _, correct_gradient = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=original)
    _, _, _, wrong_gradient = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, rotated, structural=True, reconstruction_targets=original)
    difference = capacity._vector_norm(correct_gradient["generator.phase"] - wrong_gradient["generator.phase"])
    _require(difference > 1e-8, "wrong-label counterfactual did not change the production Generator direction")
    return {"executed": True, "passed": True, "generator_gradient_difference_norm": difference, "expected": "wrong labels change the CE contribution while fixed paired targets remain unchanged"}


def _gradient_bundle_comparison(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    _require(set(left) == set(right), "gradient comparison parameter keys differ")
    comparisons: dict[str, Any] = {}
    for name in left:
        first, second = left[name], right[name]
        metadata = first.shape == second.shape and first.dtype == second.dtype
        difference = first.to(torch.float64) - second.to(torch.float64) if metadata else torch.zeros((), dtype=torch.float64)
        max_absolute = float(difference.abs().max().cpu()) if difference.numel() else 0.0
        reference_norm = float(second.to(torch.float64).norm().cpu()) if metadata else 0.0
        relative_l2 = float(difference.norm().cpu()) / max(reference_norm, 1e-30) if metadata else None
        passed = metadata and bool(torch.isfinite(difference).all()) and torch.allclose(first, second, rtol=RTOL, atol=ATOL)
        comparisons[name] = {
            "passed": bool(passed),
            "max_absolute_difference": max_absolute,
            "relative_l2_difference": relative_l2,
            "rtol": RTOL,
            "atol": ATOL,
        }
    return {"passed": all(item["passed"] for item in comparisons.values()), "comparisons": comparisons}


def _target_pair_control(inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    fit = _fit_inputs(inputs)
    images = fit["images"][:BATCH_SIZE].to(consumer.device)
    labels = fit["labels"][:BATCH_SIZE].to(consumer.device)
    original_targets = _targets_for_batch(inputs, ARM_A, 0, consumer.device)
    scaled_targets = _targets_for_batch(inputs, ARM_B, 0, consumer.device)
    # Swap row-specific target-scale assignments while preserving logical batch labels.
    permutation = torch.roll(torch.arange(BATCH_SIZE, device=consumer.device), shifts=1)
    mismatched = scaled_targets.index_select(0, permutation)
    _, _, _, paired = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=scaled_targets)
    _, _, _, mismatched_gradient = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=mismatched)
    difference = capacity._vector_norm(paired["generator.phase"] - mismatched_gradient["generator.phase"])
    _require(difference > 1e-8, "mispaired fixed target did not change the production Generator direction")
    # A fixed s=1 target must be the same producer path as no target override.
    _, losses_none, _, gradient_none = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True)
    _, losses_identity, _, gradient_identity = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=original_targets)
    _require(losses_none == losses_identity, "s=1 override does not reproduce the unscaled production loss")
    identity_gradient_check = _gradient_bundle_comparison(gradient_none, gradient_identity)
    _require(identity_gradient_check["passed"], f"s=1 override does not reproduce the unscaled production direction within replay tolerance: {identity_gradient_check['comparisons']}")
    return {"executed": True, "passed": True, "mispaired_gradient_difference_norm": difference, "s_equals_one_reproduces_A": True, "identity_gradient_comparison": identity_gradient_check}


def _verify_reconstruction_vjp(inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    fit = _fit_inputs(inputs)
    images = fit["images"][:BATCH_SIZE].to(consumer.device)
    labels = fit["labels"][:BATCH_SIZE].to(consumer.device)
    target_a = _targets_for_batch(inputs, ARM_A, 0, consumer.device)
    target_b = _targets_for_batch(inputs, ARM_B, 0, consumer.device)
    _, _, _, gradient_a = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=target_a)
    _, _, _, gradient_b = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, reconstruction_targets=target_b)
    independent = _independent_reconstruction_delta_vjp(consumer, parameters, images, target_a, target_b)
    for name in parameters:
        observed = gradient_b[name] - gradient_a[name]
        expected = capacity.semantic.BETA * independent[name]
        _require(torch.allclose(observed, expected, rtol=5e-3, atol=2e-7), f"energy-target reconstruction direction differs from independent image-space VJP for {name}")
    return {"executed": True, "passed": True, "method": "difference of real production directions vs independent raw-image derivative VJP", "generator_difference_norm": capacity._vector_norm(gradient_b["generator.phase"] - gradient_a["generator.phase"])}


def _recovery_control(inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], target: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    source_rng = copy.deepcopy(inputs["source"]["rng_state"])
    continuous_consumer, continuous_optimizer, continuous_parameters = _new_endpoint(inputs, device)
    capacity._restore_rng_state(source_rng)
    continuous_history = [
        _train_step(continuous_consumer, continuous_optimizer, continuous_parameters, evaluators, inputs, ARM_B, index)
        for index in range(2)
    ]
    continuous_state = {
        "phase": continuous_parameters["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(continuous_optimizer.state_dict())),
        "history": continuous_history,
    }
    del continuous_consumer, continuous_optimizer, continuous_parameters
    gc.collect()
    torch.cuda.empty_cache()

    split_consumer, split_optimizer, split_parameters = _new_endpoint(inputs, device)
    capacity._restore_rng_state(source_rng)
    split_history = [_train_step(split_consumer, split_optimizer, split_parameters, evaluators, inputs, ARM_B, 0)]
    temporary = target / "preflight_recovery_checkpoint.pt"
    payload = _checkpoint_payload(inputs, split_consumer, split_optimizer, ARM_B, 1, split_history)
    _write_torch_new(temporary, payload)
    loaded = torch.load(temporary, map_location="cpu", weights_only=True)
    _restore_checkpoint(loaded, split_consumer, split_optimizer)
    capacity._restore_rng_state(loaded["rng_state"])
    split_history.append(_train_step(split_consumer, split_optimizer, split_parameters, evaluators, inputs, ARM_B, 1))
    split_state = {
        "phase": split_parameters["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(split_optimizer.state_dict())),
        "history": split_history,
    }
    comparison = stroke._trial_repeatability(continuous_state, split_state)
    temporary.unlink()
    _require(comparison["passed"], "continuous and explicit-save/restore training segments differ")
    return {"executed": True, "passed": True, "continuous_equals_save_restore": comparison, "updates": 2, "checkpoint_step": SOURCE_STEP + 1}


def _checkpoint_payload(
    inputs: Mapping[str, Any],
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    local_completed: int,
    history: Sequence[Mapping[str, Any]],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    parameters = capacity._parameters(consumer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "source_step": SOURCE_STEP,
        "completed_updates": SOURCE_STEP + local_completed,
        "task_local_updates": local_completed,
        "optimizer_step": int(float(optimizer.state[parameters["generator.phase"]]["step"].item())),
        "next_batch_index": local_completed % BATCH_COUNT,
        "next_batch_cursor": local_completed % BATCH_COUNT,
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "numeric_mode": NUMERIC_MODE,
        "objective_definition": "raw_MSE_plus_0.1_SSIM_for_both_arms; B_fixed_fit_energy_matched_y",
        "optimizer_parameter_names": ["generator.phase"],
        "optimizer_state": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "trainable_parameter_state": {name: value.detach().cpu().clone() for name, value in parameters.items()},
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": inputs["source"]["references"],
        "weights": inputs["source"]["weights"],
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "population": inputs["population"],
        "target_scale_coefficients": inputs["scales"].clone(),
        "rng_state": capacity._rng_state(),
        "source_rng_state": copy.deepcopy(inputs["source"]["rng_state"]),
        "task_training_history": list(history),
        "source_history_count": SOURCE_STEP,
        "checkpoint_reason": reason or "every_100_updates_or_fixed_budget_endpoint",
    }


def _checkpoint_path(target: Path, arm: str, local_completed: int) -> Path:
    letter = "a" if arm == ARM_A else "b"
    return target / f"arm_{letter}_step{SOURCE_STEP + local_completed:04d}.pt"


def _save_checkpoint(
    target: Path,
    inputs: Mapping[str, Any],
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    local_completed: int,
    history: Sequence[Mapping[str, Any]],
    *,
    reason: str | None = None,
) -> Path:
    path = _checkpoint_path(target, arm, local_completed)
    _write_torch_new(path, _checkpoint_payload(inputs, consumer, optimizer, arm, local_completed, history, reason=reason))
    letter = "a" if arm == ARM_A else "b"
    _replace_json(target / f"training_history_arm_{letter}.json", {"task_id": TASK_ID, "arm": arm, "source_step": SOURCE_STEP, "additional_updates": local_completed, "records": list(history)})
    return path


def _read_checkpoint(path: Path, target: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file() and target.resolve() in path.resolve().parents, "resume requires an explicit checkpoint inside this task output")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("task_id") == TASK_ID, "checkpoint belongs to another task/version")
    _require(payload.get("population") == inputs["population"], "checkpoint population or fixed target coefficients differ")
    _require(payload.get("source_checkpoint") == inputs["population"]["source_checkpoint"] and payload.get("source_step") == SOURCE_STEP, "checkpoint source model differs")
    _require(payload.get("objective_definition") == "raw_MSE_plus_0.1_SSIM_for_both_arms; B_fixed_fit_energy_matched_y", "checkpoint objective differs")
    _require(payload.get("numeric_mode") == NUMERIC_MODE and payload.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "checkpoint numeric mode or Generator layout differs")
    _require(_tensor_tree_equal(payload.get("target_scale_coefficients"), inputs["scales"]), "checkpoint target coefficients differ")
    _require(payload.get("arm") in ARM_NAMES, "checkpoint arm is invalid")
    local = payload.get("task_local_updates")
    _require(type(local) is int and 0 <= local <= ADDED_UPDATES, "checkpoint update count is invalid")
    _require(payload.get("completed_updates") == SOURCE_STEP + local and payload.get("optimizer_step") == SOURCE_STEP + local, "checkpoint source/task Adam count differs")
    _require(payload.get("next_batch_index") == local % BATCH_COUNT and payload.get("next_batch_cursor") == local % BATCH_COUNT, "checkpoint next batch does not match its task-local count")
    _require(path.name == _checkpoint_path(target, payload["arm"], local).name, "explicit checkpoint filename does not match arm and update count")
    _require(len(payload.get("task_training_history", [])) == local, "checkpoint task history length differs from update count")
    history = payload["task_training_history"]
    _require(all(record.get("task_local_update") == index + 1 and record.get("optimizer_step") == SOURCE_STEP + index + 1 and record.get("arm") == payload["arm"] and record.get("batch_index") == index % BATCH_COUNT for index, record in enumerate(history)), "checkpoint history does not match actual task batch mapping")
    for index, record in enumerate(history):
        batch = index % BATCH_COUNT
        start = batch * BATCH_SIZE
        positions = list(range(start, start + BATCH_SIZE))
        _require(record.get("fit_positions") == positions, "checkpoint history fit positions differ from the fixed batch order")
        _require(record.get("source_row_indices") == inputs["rows"][start : start + BATCH_SIZE].tolist(), "checkpoint history source rows differ from the fixed TRAIN mapping")
        _require(record.get("labels") == inputs["labels"][start : start + BATCH_SIZE].tolist(), "checkpoint history labels differ from the fixed batch")
        _require(record.get("target_scale_coefficients") == inputs["scales"][start : start + BATCH_SIZE].tolist(), "checkpoint history target scales differ from the fixed fit-row coefficients")
        expected_target_rule = "original_y" if payload["arm"] == ARM_A else "fixed_A4700_energy_ratio_times_y"
        _require(record.get("target_rule") == expected_target_rule, "checkpoint history reconstruction target differs from the arm definition")
    _require(_tensor_tree_equal(payload.get("source_rng_state"), inputs["source"]["rng_state"]), "checkpoint source RNG differs from A4700 source")
    _require(_tensor_tree_equal(payload.get("references"), inputs["source"]["references"]) and _tensor_tree_equal(payload.get("weights"), inputs["source"]["weights"]), "checkpoint references/weights differ")
    _require(_tensor_tree_equal(payload.get("frozen_runtime_state"), inputs["source"]["frozen_runtime_state"]), "checkpoint frozen OLS state differs")
    params = payload.get("trainable_parameter_state", {})
    _require(set(params) == {"encoder.phase", "generator.phase"}, "checkpoint trainable parameter map is incomplete")
    _require(tuple(params["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(params["generator.phase"].shape) == (340_000,), "checkpoint phase layout differs")
    _require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in params.values()), "checkpoint phase state is nonfinite or has wrong dtype")
    optimizer = payload.get("optimizer_state", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "checkpoint Adam parameter map differs")
    state = states[0]
    group = groups[0]
    _require(float(group.get("lr", -1.0)) == 1e-3 and tuple(group.get("betas", ())) == (0.9, 0.999) and float(group.get("eps", -1.0)) == 1e-8 and float(group.get("weight_decay", -1.0)) == 0.0, "checkpoint Adam configuration differs")
    _require(int(float(state["step"].item())) == SOURCE_STEP + local and tuple(state["exp_avg"].shape) == (340_000,) and tuple(state["exp_avg_sq"].shape) == (340_000,), "checkpoint Adam step/moment shape differs")
    _require(bool(torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()), "checkpoint Adam moments are nonfinite")
    return payload


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = payload["trainable_parameter_state"][name]
            _require(value.shape == parameter.shape and value.dtype == parameter.dtype, f"checkpoint phase layout/dtype differs for {name}")
            parameter.copy_(value.to(parameter.device))
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == payload["optimizer_step"], "restored Generator Adam step differs")
    capacity._assert_state_equal(payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "restored frozen OLS runtime")
    capacity._restore_rng_state(payload["rng_state"])


def _run_arm(
    target: Path,
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, torch.nn.Module],
    arm: str,
    start_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    consumer, optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    if start_payload is None:
        completed = 0
        history: list[dict[str, Any]] = []
        capacity._restore_rng_state(inputs["source"]["rng_state"])
        if not _checkpoint_path(target, arm, 0).exists():
            _save_checkpoint(target, inputs, consumer, optimizer, arm, 0, history, reason="copied_A4700_source_start")
    else:
        _require(start_payload.get("arm") == arm, "explicit resume checkpoint belongs to the other arm")
        _restore_checkpoint(start_payload, consumer, optimizer)
        completed = int(start_payload["task_local_updates"])
        history = list(start_payload["task_training_history"])
    _require(len(history) == completed and 0 <= completed <= ADDED_UPDATES, "restored task-local history/cursor differs")
    letter = "a" if arm == ARM_A else "b"
    _replace_json(target / f"training_history_arm_{letter}.json", {"task_id": TASK_ID, "arm": arm, "source_step": SOURCE_STEP, "additional_updates": completed, "records": list(history)})
    frozen_before = capacity._frozen_runtime_state(consumer)
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    stop_state = {"requested": False}
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        stop_state["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while completed < ADDED_UPDATES:
            history.append(_train_step(consumer, optimizer, parameters, evaluators, inputs, arm, completed))
            completed += 1
            if completed % 100 == 0 or stop_state["requested"]:
                _save_checkpoint(target, inputs, consumer, optimizer, arm, completed, history, reason="interrupt_boundary" if stop_state["requested"] else None)
                telemetry = capacity._gpu_snapshot(torch.device("cuda:0"))
                with (target / "resource_telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"arm": arm, "task_local_update": completed, "optimizer_step": SOURCE_STEP + completed, "gpu_observation": telemetry}, sort_keys=True) + "\n")
            if stop_state["requested"]:
                raise TrainingInterrupted(f"{arm} durably checkpointed after {completed} task-local updates")
        if completed % 100:
            _save_checkpoint(target, inputs, consumer, optimizer, arm, completed, history)
        _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} Encoder changed during training")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{arm} frozen OLS runtime")
        _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == FINAL_STEP, f"{arm} final Adam step differs")
        return _read_checkpoint(_checkpoint_path(target, arm, ADDED_UPDATES), target, inputs)
    finally:
        signal.signal(signal.SIGINT, previous)
        gc.collect()
        torch.cuda.empty_cache()


def _run_training(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], selected: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected is None:
        final_a = _run_arm(target, inputs, evaluators, ARM_A, None)
        final_b = _run_arm(target, inputs, evaluators, ARM_B, None)
    elif selected["arm"] == ARM_A:
        final_a = _run_arm(target, inputs, evaluators, ARM_A, selected)
        bstart = _read_checkpoint(_checkpoint_path(target, ARM_B, 0), target, inputs) if _checkpoint_path(target, ARM_B, 0).is_file() else None
        final_b = _run_arm(target, inputs, evaluators, ARM_B, bstart)
    else:
        _require(_checkpoint_path(target, ARM_A, ADDED_UPDATES).is_file(), "B recovery requires a completed A6700 checkpoint")
        final_a = _read_checkpoint(_checkpoint_path(target, ARM_A, ADDED_UPDATES), target, inputs)
        final_b = _run_arm(target, inputs, evaluators, ARM_B, selected)
    _require(final_a["completed_updates"] == final_b["completed_updates"] == FINAL_STEP, "both arms must reach Generator Adam step6700")
    return {ARM_A: final_a, ARM_B: final_b}


def _readout(consumer: Any, evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    payload = capacity._forward_all(
        consumer,
        evaluators,
        {"images": inputs["images"], "source_indices": inputs["rows"], "labels": inputs["labels"]},
        "B_large",
    )
    _require(torch.equal(payload["source_indices"], inputs["rows"]) and torch.equal(payload["labels"], inputs["labels"]), f"{endpoint} readout row mapping differs")
    return payload


def _endpoint(target: Path, name: str, checkpoint: Mapping[str, Any] | None, inputs: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]):
    consumer, optimizer, _parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    if checkpoint is None:
        capacity._restore_rng_state(inputs["source"]["rng_state"])
    else:
        _restore_checkpoint(checkpoint, consumer, optimizer)
    measured = _readout(consumer, evaluators, inputs, name)
    readout_path = target / f"readouts_{name}.pt"
    if readout_path.exists():
        saved_readout = torch.load(readout_path, map_location="cpu", weights_only=True)
        _require(torch.equal(saved_readout["source_indices"], inputs["rows"]) and torch.equal(saved_readout["labels"], inputs["labels"]), f"saved {name} readout row/label mapping differs")
        stroke._replay_check(measured, saved_readout)
        readout = saved_readout
    else:
        _write_torch_new(readout_path, measured)
        readout = measured
    summary, row_metrics = class9._row_and_image_metrics(readout, inputs["population"])
    summary["endpoint"] = name
    custom_metrics, custom_rows = _energy_image_metrics(readout, inputs)
    summary["energy_metrics"] = custom_metrics
    row_metrics.update(custom_rows)
    metrics_path = target / f"metrics_{name}.json"
    if metrics_path.exists():
        _require(json.loads(metrics_path.read_text(encoding="utf-8")) == summary, f"saved {name} summary differs from its readout")
    else:
        _write_json_new(metrics_path, summary)
    row_path = target / f"row_metrics_{name}.pt"
    if row_path.exists():
        previous = torch.load(row_path, map_location="cpu", weights_only=True)
        _require(set(previous) == set(row_metrics), f"saved {name} row metric fields differ")
        for key, value in row_metrics.items():
            _require(torch.equal(previous[key], value), f"saved {name} row metric differs for {key}")
    else:
        _write_torch_new(row_path, row_metrics)
    return summary, readout, row_metrics


def _energy_image_metrics(readout: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw = readout["raw_cmos"].to(torch.float64).cpu()
    target_raw = readout["target_slm_intensity"].to(torch.float64).cpu()
    target_view = diagnostic.canonical.cmos_view(readout["target_slm_intensity"]).to(torch.float64).cpu()
    generated_view = readout["evaluator_view"].to(torch.float64).cpu()
    _require(raw.shape == target_raw.shape and target_view.shape == generated_view.shape, "readout target/generated dimensions differ")
    area, pitch = _pixel_area(readout, "camera_plane")
    labels = readout["labels"].to(torch.long).cpu()
    flat_generated = generated_view.flatten(1)
    flat_target = target_view.flatten(1)
    generated_mass = flat_generated.sum(1)
    target_mass = flat_target.sum(1)
    q = torch.full((raw.shape[0],), float("nan"), dtype=torch.float64)
    defined = (generated_mass > 0) & (target_mass > 0)
    if bool(defined.any()):
        normalized_generated = flat_generated[defined] / generated_mass[defined, None]
        normalized_target = flat_target[defined] / target_mass[defined, None]
        numerator = (normalized_generated - normalized_target).square().sum(1)
        denominator = normalized_target.square().sum(1)
        _require(bool((denominator > 0).all()), "positive target has zero brightness-normalized Q denominator")
        q[defined] = numerator / denominator
    foreground = target_raw > 0.1
    background = ~foreground
    raw_error = (raw - target_raw).square()
    fg = (raw_error * foreground).flatten(1).sum(1) / foreground.flatten(1).sum(1).clamp_min(1)
    bg = (raw_error * background).flatten(1).sum(1) / background.flatten(1).sum(1).clamp_min(1)
    ssim = stroke.ssim_per_image(generated_view, target_view).flatten().cpu()
    energy = raw.flatten(1).sum(1) * area
    clipped = ((raw < 0) | (raw > 1)).flatten(1).to(torch.float64).mean(1)
    zero = raw.flatten(1).abs().sum(1) == 0
    constant = raw.flatten(1).std(1, unbiased=False) == 0
    row_metrics = {
        "brightness_normalized_stroke_error_Q": q,
        "Q_defined": defined,
        "cmos_integrated_intensity_joule_proxy": energy,
        "raw_clip_fraction_outside_0_1": clipped,
        "raw_zero_image": zero,
        "raw_constant_image": constant,
        "original_target_raw_mse": raw_error.flatten(1).mean(1),
        "original_target_raw_foreground_mse": fg,
        "original_target_raw_background_mse": bg,
        "original_target_ssim_28x28": ssim,
        "generated_view_mass": generated_mass,
        "target_view_mass": target_mass,
    }
    rows_by_role: dict[str, Any] = {}
    role_positions = inputs["population"]["role_positions"]
    for role, bounds in role_positions.items():
        start, stop = map(int, bounds)
        class_stats = {}
        for c in range(10):
            selected = (labels[start:stop] == c).nonzero().flatten() + start
            q_value = None if not bool(defined[selected].all()) else float(q[selected].mean())
            class_stats[str(c)] = {
                "rows": int(selected.numel()),
                "mean_Q": q_value,
                "mean_cmos_integrated_intensity": float(energy[selected].mean()),
                "mean_raw_clip_fraction": float(clipped[selected].mean()),
                "zero_images": int(zero[selected].sum()),
                "constant_images": int(constant[selected].sum()),
                "mean_original_target_raw_mse": float(raw_error.flatten(1).mean(1)[selected].mean()),
                "mean_original_target_raw_foreground_mse": float(fg[selected].mean()),
                "mean_original_target_raw_background_mse": float(bg[selected].mean()),
                "mean_original_target_ssim_28x28": float(ssim[selected].mean()),
            }
        role_defined = bool(defined[start:stop].all())
        rows_by_role[role] = {
            "rows": stop - start,
            "mean_Q": float(q[start:stop].mean()) if role_defined else None,
            "Q_undefined_rows": int((~defined[start:stop]).sum()),
            "mean_cmos_integrated_intensity": float(energy[start:stop].mean()),
            "mean_raw_clip_fraction": float(clipped[start:stop].mean()),
            "zero_images": int(zero[start:stop].sum()),
            "constant_images": int(constant[start:stop].sum()),
            "mean_original_target_raw_mse": float(raw_error.flatten(1).mean(1)[start:stop].mean()),
            "mean_original_target_raw_foreground_mse": float(fg[start:stop].mean()),
            "mean_original_target_raw_background_mse": float(bg[start:stop].mean()),
            "mean_original_target_ssim_28x28": float(ssim[start:stop].mean()),
            "mean_cmos_integrated_intensity_by_class": {key: value["mean_cmos_integrated_intensity"] for key, value in class_stats.items()},
            "by_class": class_stats,
        }
    summary = {
        "energy_pixel_pitch_m": pitch,
        "energy_pixel_area_m2": area,
        "metrics_by_role": rows_by_role,
        "all_rows_finite": bool(torch.isfinite(raw).all() and torch.isfinite(generated_view).all() and torch.isfinite(target_raw).all()),
        "all_rows_zero_images": int(zero.sum()),
        "all_rows_constant_images": int(constant.sum()),
    }
    return summary, row_metrics


def _mean_q(summary: Mapping[str, Any], role: str, class_id: int | None = None) -> float | None:
    values = summary["energy_metrics"]["metrics_by_role"][role]
    if class_id is None:
        return values["mean_Q"]
    return values["by_class"][str(class_id)]["mean_Q"]


def _final_decision(start: Mapping[str, Any], arm_a: Mapping[str, Any], arm_b: Mapping[str, Any], *, valid: bool) -> dict[str, Any]:
    role = "confirmation"
    q_start = _mean_q(start, role)
    q_a = _mean_q(arm_a, role)
    q_b = _mean_q(arm_b, role)
    q_defined = all(value is not None and math.isfinite(float(value)) for value in (q_start, q_a, q_b))
    all_q_defined = q_defined and all(
        value is not None and math.isfinite(float(value))
        for summary in (start, arm_a, arm_b)
        for class_id in range(10)
        for value in (_mean_q(summary, role, class_id),)
    )
    ratios: dict[str, Any] = {}
    focus = {"overall": None, "class_2": None, "class_4": None}
    for key, class_id in (("overall", None), ("class_2", 2), ("class_4", 4)):
        base, a, b = _mean_q(start, role, class_id), _mean_q(arm_a, role, class_id), _mean_q(arm_b, role, class_id)
        if all(value is not None and float(value) > 0 for value in (base, a, b)):
            ratios[key] = {"B_over_A": float(b) / float(a), "B_over_start": float(b) / float(base)}
            focus[key] = {"B_over_A": float(b) / float(a) <= 0.90, "B_over_start": float(b) / float(base) <= 0.90}
        else:
            ratios[key] = None
            focus[key] = {"B_over_A": False, "B_over_start": False}
    target_improvement = all(all(pair.values()) for pair in focus.values())

    q_other_classes = {}
    for class_id in (0, 1, 3, 5, 6, 7, 8, 9):
        base = _mean_q(start, role, class_id)
        a = _mean_q(arm_a, role, class_id)
        b = _mean_q(arm_b, role, class_id)
        ok = all(value is not None and float(value) >= 0 for value in (base, a, b)) and float(b) <= 1.05 * float(a) and float(b) <= 1.05 * float(base)
        q_other_classes[str(class_id)] = {
            "B_over_A": None if not a else float(b) / float(a) if b is not None else None,
            "B_over_start": None if not base else float(b) / float(base) if b is not None else None,
            "passed": bool(ok),
        }

    start_eval = start["evaluators"]["E_D"]["by_role"][role]
    a_eval = arm_a["evaluators"]["E_D"]["by_role"][role]
    b_eval = arm_b["evaluators"]["E_D"]["by_role"][role]
    start_counts, a_counts, b_counts = map(lambda item: list(item["per_class_correct"]), (start_eval, a_eval, b_eval))
    classifier_checks = {
        "B_total_not_below_start": int(b_eval["correct_total"]) >= int(start_eval["correct_total"]),
        "B_total_at_most_six_below_A": int(b_eval["correct_total"]) >= int(a_eval["correct_total"]) - 6,
        "any_class_not_more_than_three_below_A_or_start": all(b_counts[c] >= a_counts[c] - 3 and b_counts[c] >= start_counts[c] - 3 for c in range(10)),
        "class_1_and_7_not_below_start": b_counts[1] >= start_counts[1] and b_counts[7] >= start_counts[7],
    }

    energy_checks: dict[str, Any] = {}
    energy_start = start["energy_metrics"]["metrics_by_role"][role]
    energy_a = arm_a["energy_metrics"]["metrics_by_role"][role]
    energy_b = arm_b["energy_metrics"]["metrics_by_role"][role]
    energy_checks["overall"] = float(energy_b["mean_cmos_integrated_intensity"]) >= 0.95 * float(energy_a["mean_cmos_integrated_intensity"]) and float(energy_b["mean_cmos_integrated_intensity"]) >= 0.95 * float(energy_start["mean_cmos_integrated_intensity"])
    for c in range(10):
        vb = float(energy_b["by_class"][str(c)]["mean_cmos_integrated_intensity"])
        va = float(energy_a["by_class"][str(c)]["mean_cmos_integrated_intensity"])
        vs = float(energy_start["by_class"][str(c)]["mean_cmos_integrated_intensity"])
        energy_checks[str(c)] = vb >= 0.95 * va and vb >= 0.95 * vs
    clip_checks = {
        "overall": float(energy_b["mean_raw_clip_fraction"]) <= 0.01,
        "each_class": all(float(item["mean_raw_clip_fraction"]) <= 0.01 for item in energy_b["by_class"].values()),
    }
    outputs_ok = arm_b.get("all_finite") is True and arm_b["energy_metrics"].get("all_rows_finite") is True and arm_b["energy_metrics"]["all_rows_zero_images"] == 0 and arm_b["energy_metrics"]["all_rows_constant_images"] == 0
    conditions = {
        "Q_defined_for_all_rows": all_q_defined and all(summary["energy_metrics"]["metrics_by_role"][role]["Q_undefined_rows"] == 0 for summary in (start, arm_a, arm_b)),
        "overall_and_class_2_4_Q_at_least_10pct_lower_vs_A_and_start": target_improvement,
        "other_classes_Q_no_more_than_5pct_higher_vs_A_and_start": all(value["passed"] for value in q_other_classes.values()),
        "E_D_total_and_class_protection": all(classifier_checks.values()),
        "mean_cmos_integrated_energy_at_least_95pct_vs_A_and_start": all(energy_checks.values()),
        "raw_clip_fraction_at_most_1pct_overall_and_each_class": all(clip_checks.values()),
        "finite_nonzero_nonconstant_outputs": outputs_ok,
        "budgets_freeze_and_measurement_valid": valid,
    }
    supported = all(conditions.values())
    any_undefined = not all_q_defined or any(item is None for item in ratios.values()) or not outputs_ok
    overall_improved_both = q_defined and q_b is not None and q_a is not None and q_start is not None and float(q_b) < float(q_a) and float(q_b) < float(q_start)
    if not valid:
        classification = "INVALID_HARNESS"
    elif supported:
        classification = "ENERGY_MATCHED_TARGET_STROKE_METRIC_GAIN_SUPPORTED"
    elif any_undefined or overall_improved_both:
        classification = "MIXED_OR_INSUFFICIENT"
    else:
        classification = "ENERGY_MATCHED_TARGET_NOT_SUPPORTED_WITHIN_2000_UPDATES"
    return {
        "classification": classification,
        "conditions": conditions,
        "Q_ratios_B_to_A_and_start": ratios,
        "Q_other_class_checks": q_other_classes,
        "focus_Q_checks": focus,
        "E_D_correct_total": {"start": int(start_eval["correct_total"]), "A": int(a_eval["correct_total"]), "B": int(b_eval["correct_total"])},
        "E_D_per_class_correct": {"start": start_counts, "A": a_counts, "B": b_counts},
        "E_D_checks": classifier_checks,
        "energy_checks": energy_checks,
        "clip_checks": clip_checks,
        "all_conditions_passed": supported,
    }


def _endpoint_summary(readout: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    class_summary, _ = class9._row_and_image_metrics(readout, population)
    return class_summary


def _metrics_from_saved(readout: Mapping[str, Any], inputs: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    summary, _ = class9._row_and_image_metrics(readout, inputs["population"])
    energy, _ = _energy_image_metrics(readout, inputs)
    summary["endpoint"] = endpoint
    summary["energy_metrics"] = energy
    summary["all_finite"] = summary.get("all_finite", False) and energy["all_rows_finite"]
    return summary


def _load_evaluators(inputs: Mapping[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    return diagnostic.load_evaluators(inputs["evaluator_paths"], device)


def _preflight(inputs: Mapping[str, Any], target: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized production preflight")
    evaluators = _load_evaluators(inputs, device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    source_adam_before = copy.deepcopy(inputs["source"]["optimizer_state"])
    source_rng = copy.deepcopy(inputs["source"]["rng_state"])
    checks: dict[str, Any] = {}
    gpu_before = capacity._gpu_snapshot(device)
    try:
        checks["cuda_disk_and_external_io"] = stroke._disk_and_io(target)
        target_control = _target_control(evaluators, inputs)
        checks["four_evaluator_target_positive_control"] = {key: value for key, value in target_control.items() if key not in {"logits", "evaluators"}}
        checks["four_evaluator_target_positive_control"]["per_evaluator"] = {name: {key: value for key, value in item.items() if key != "logits"} for name, item in target_control["evaluators"].items()}
        _require(target_control["passed"], "a frozen evaluator target positive-control threshold failed")
        stroke._write_torch(target / "target_positive_control.pt", {"source_indices": inputs["rows"][FIT_COUNT:], "labels": inputs["labels"][FIT_COUNT:], **{f"{name.lower()}_logits": item["logits"] for name, item in target_control["evaluators"].items()}})

        consumer, _optimizer, _parameters = _new_endpoint(inputs, device)
        baseline = _readout(consumer, evaluators, inputs, "A4700")
        replay = stroke._replay_check(baseline, inputs["readout"])
        checks["a4700_production_replay"] = {"executed": True, **replay, "stroke_readout_identity": inputs["saved_readout_replay"]}
        _require(inputs["saved_readout_replay"]["passed"], "attempt_08 saved start readout differs from source checkpoint A4700")
        stroke._write_torch(target / "readouts_start.pt", baseline)
        del consumer, _optimizer, _parameters
        gc.collect()
        torch.cuda.empty_cache()

        capacity._restore_rng_state(source_rng)
        target_identity = _target_pair_control(inputs, evaluators)
        checks["fixed_target_identity_and_pairing_negative_control"] = {"executed": True, **target_identity}
        checks["reconstruction_difference_image_space_vjp"] = _verify_reconstruction_vjp(inputs, evaluators)

        capacity._restore_rng_state(source_rng)
        trial_a1 = _run_trial(ARM_A, inputs, evaluators, updates=3)
        capacity._restore_rng_state(source_rng)
        trial_a2 = _run_trial(ARM_A, inputs, evaluators, updates=3)
        aa = stroke._aa_preflight_record(trial_a1, trial_a2, updates=3)
        a_groups_updated = trial_a1["old_plane_gradient_norm"] > 0.0 and trial_a1["new_plane_gradient_norm"] > 0.0
        aa["phase_groups_received_nonzero_gradient"] = a_groups_updated
        aa["passed"] = aa["passed"] and a_groups_updated
        checks["arm_a_three_real_updates_and_aa"] = aa
        _require(aa["passed"], "A/A three-step production path differs or a Generator phase group received no gradient")
        del trial_a1, trial_a2
        gc.collect()
        torch.cuda.empty_cache()

        capacity._restore_rng_state(source_rng)
        trial_b = _run_trial(ARM_B, inputs, evaluators, updates=3)
        checks["arm_b_three_real_updates"] = {
            "executed": True,
            "passed": trial_b["new_plane_gradient_norm"] > 0.0 and trial_b["old_plane_gradient_norm"] > 0.0,
            "updates": 3,
            "generator_adam_step": SOURCE_STEP + 3,
            "original_phase_gradient_norm": trial_b["old_plane_gradient_norm"],
            "added_phase_gradient_norm": trial_b["new_plane_gradient_norm"],
            "ssim_gradient_norm": trial_b["history"][0]["ssim_gradient_norm"],
        }
        _require(checks["arm_b_three_real_updates"]["passed"], "B three-step trial did not update both phase groups")
        del trial_b
        gc.collect()
        torch.cuda.empty_cache()

        checks["wrong_label_ce_negative_control"] = _wrong_label_control(inputs, evaluators)
        checks["recovery_control"] = _recovery_control(inputs, evaluators, target)
        checks["freeze_scope"] = {
            "executed": True,
        "passed": all(_tensor_tree_equal(capacity._state_snapshot(model), evaluator_before[name]) for name, model in evaluators.items())
            and _tensor_tree_equal(inputs["source"]["optimizer_state"], source_adam_before)
            and all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()),
            "evaluators_unchanged": all(_tensor_tree_equal(capacity._state_snapshot(model), evaluator_before[name]) for name, model in evaluators.items()),
            "source_adam_unchanged": _tensor_tree_equal(inputs["source"]["optimizer_state"], source_adam_before),
            "evaluator_and_model_parameter_gradients_empty": all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()),
        }
        _require(checks["freeze_scope"]["passed"], "preflight changed a frozen evaluator or source Adam state")
        gpu_after = capacity._gpu_snapshot(device)
        for key, check in checks.items():
            _require(check.get("executed") is True and check.get("passed") is True, f"preflight check did not pass: {key}")
        return {
            "task_id": TASK_ID,
            "status": "READY",
            "checks": checks,
            "cuda_before": gpu_before,
            "cuda_after": gpu_after,
            "numeric_mode": NUMERIC_MODE,
            "source_readout_replay": replay,
            "energy_scale_summary": inputs["scale_evidence"]["coefficient_summary"],
        }, baseline, target_control
    finally:
        capacity._restore_rng_state(source_rng)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight frozen evaluator {name}")
        _require(_tensor_tree_equal(inputs["source"]["optimizer_state"], source_adam_before), "preflight changed source Adam tensors")
        gc.collect()
        torch.cuda.empty_cache()


def _make_identity(inputs: Mapping[str, Any], target: Path) -> ResearchRunIdentity:
    attempt_number = _attempt_number(target)
    attempt_suffix = f"{attempt_number:02d}"
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-{attempt_suffix}",
        trajectory_id="large-generator-A4700-energy-matched-target-2000-updates",
        semantic_authority_id=stroke.TASK_ID,
        data_role="canonical TRAIN fit=[0,200), confirmation=[200,500); scales use fit rows only; confirmation excluded from gradients and coefficient construction",
        config_identity={
            "source_step": SOURCE_STEP,
            "updates_per_arm": ADDED_UPDATES,
            "batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "raw_reconstruction": "MSE_200x200_plus_0.1_SSIM_28x28_both_arms",
            "target_change": "B fixed per-row scale from A4700 CMOS/target integrated energy; only paired reconstruction target",
            "strict_fp32_optics": True,
        },
        model_identity={
            "source_checkpoint": inputs["population"]["source_checkpoint"],
            "architecture": "two_original_100x100_then_two_400x400_phase_planes_flattened",
            "parameter_shapes": inputs["population"]["source_parameter_shapes"],
            "finite": True,
        },
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "reset": False},
        sampler_identity={"fit_rows": inputs["rows"][:FIT_COUNT].tolist(), "fit_batches": BATCH_COUNT, "confirmation_rows": inputs["rows"][FIT_COUNT:].tolist()},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-{attempt_suffix}",
    )


def _attempt_number(target: Path) -> int:
    name = target.name
    prefix = "attempt_"
    _require(name.startswith(prefix) and name[len(prefix) :].isdigit(), "output directory must end with attempt_<number>")
    return int(name[len(prefix) :])


def _make_receipt(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    checks = preflight["checks"]
    _require(checks and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "READY requires every executed check to pass")
    authorization_id = f"{TASK_ID}:authorization:{_attempt_number(target):02d}"
    control = ResearchRunControl(_make_identity(inputs, target), authorization_id=authorization_id)
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "source_replay": checks["a4700_production_replay"]["passed"], "evaluator_target_controls": checks["four_evaluator_target_positive_control"]["passed"], "A_A_real_updates": checks["arm_a_three_real_updates_and_aa"]["passed"], "B_real_updates": checks["arm_b_three_real_updates"]["passed"], "target_identity_pairing": checks["fixed_target_identity_and_pairing_negative_control"]["passed"], "independent_reconstruction_vjp": checks["reconstruction_difference_image_space_vjp"]["passed"], "wrong_label_control": checks["wrong_label_ce_negative_control"]["passed"], "recovery": checks["recovery_control"]["passed"], "frozen_scope": checks["freeze_scope"]["passed"]},
        resource={"passed": checks["cuda_disk_and_external_io"]["passed"], "status": "PASS", "cuda_available": True, "gpu_observation": preflight["cuda_after"], "disk": checks["cuda_disk_and_external_io"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_large_generator_only_strict_fp32", "three_real_updates_per_arm": True, "continuous_restore_equivalence": checks["recovery_control"]["continuous_equals_save_restore"]},
        selected_engine="cuda_2048_large_generator_only_strict_fp32",
        resource_envelope={"cuda": preflight["cuda_after"], "disk": checks["cuda_disk_and_external_io"]},
    )
    return control, receipt


def _stage_ready(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    expected_population = inputs["population"]
    expected = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    population_path, preflight_path = target / "input_population.json", target / "preflight.json"
    if resume:
        _require(population_path.is_file() and json.loads(population_path.read_text(encoding="utf-8")) == expected_population, "saved task population differs")
        _require(preflight_path.is_file() and json.loads(preflight_path.read_text(encoding="utf-8")) == expected, "saved READY receipt differs")
    else:
        if population_path.exists():
            _require(json.loads(population_path.read_text(encoding="utf-8")) == expected_population, "staged population differs")
        else:
            _write_json_new(population_path, expected_population)
        if preflight_path.exists():
            _require(json.loads(preflight_path.read_text(encoding="utf-8")) == expected, "staged READY evidence differs")
        else:
            _write_json_new(preflight_path, expected)


def _validate_ready(target: Path, inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt, dict[str, Any]]:
    payload = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(payload.get("task_id") == TASK_ID and payload.get("status") == "READY", "saved READY is missing or belongs to another task")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == inputs["population"], "population/scale coefficients changed after READY")
    preflight = payload["preflight"]
    _require(preflight.get("status") == "READY" and all(item.get("executed") is True and item.get("passed") is True for item in preflight.get("checks", {}).values()), "READY consumer found a missing or failed measured check")
    authorization_id = f"{TASK_ID}:authorization:{_attempt_number(target):02d}"
    control = ResearchRunControl(_make_identity(inputs, target), authorization_id=authorization_id)
    receipt = PreflightReceipt.from_mapping(payload["receipt"])
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == authorization_id and receipt.status == "READY", "saved receipt identity/status differs")
    _require(all(group.get("passed") is True for group in receipt.checks.values()), "saved receipt contains a missing or failed group")
    control.mark_preflight_ready(receipt)
    return control, receipt, preflight


def _target_directory(root: Path, target: Path, *, resume: bool) -> None:
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output must be inside outputs/")
    parts = target.relative_to(outputs).parts
    _require(len(parts) == 3 and parts[1] == "runs" and parts[0].startswith("20261009_"), "output must be outputs/20261009_name/runs/attempt")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected, restored, or temporary output namespace is forbidden")
    if not target.exists():
        _require(not resume, "cannot resume a task output directory that does not exist")
        target.mkdir(parents=True, exist_ok=False)
        return
    _require(target.is_dir(), "output path is not a directory")
    if resume:
        _require((target / "preflight.json").is_file(), "resume lacks READY evidence")
        return
    allowed = {"input_population.json", "preflight.json", "readouts_start.pt", "target_positive_control.pt", "target_scale_coefficients.json", "source_training_history.json", "preflight_recovery_checkpoint.pt"}
    existing = {path.name for path in target.iterdir()}
    _require(not existing or existing.issubset(allowed), "existing task output is not an empty or READY stage; explicit resume is required")
    _require(not (target / "report.json").is_file(), "completed task requires explicit terminal recovery")


def _save_coefficients(target: Path, inputs: Mapping[str, Any]) -> None:
    path = target / "target_scale_coefficients.json"
    if not path.exists():
        _write_json_new(path, inputs["scale_evidence"])
    else:
        _require(json.loads(path.read_text(encoding="utf-8")) == inputs["scale_evidence"], "saved target coefficients differ from current A4700 fit-only calculation")


def _source_history(target: Path, inputs: Mapping[str, Any]) -> None:
    path = target / "source_training_history.json"
    source_history = inputs["source"].get("training_history", [])
    _require(len(source_history) == SOURCE_STEP, "A4700 source history count differs")
    if not path.exists():
        _write_json_new(path, {"source_checkpoint": inputs["population"]["source_checkpoint"], "source_step": SOURCE_STEP, "records": source_history})
    else:
        saved = json.loads(path.read_text(encoding="utf-8"))
        _require(
            saved.get("source_step") == SOURCE_STEP
            and len(saved.get("records", [])) == SOURCE_STEP
            and saved.get("source_checkpoint") == inputs["population"]["source_checkpoint"]
            and saved.get("records") == source_history,
            "saved source training history differs from the explicitly loaded A4700 checkpoint",
        )


def _gallery(target: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    from PIL import Image, ImageDraw

    page_paths = []
    labels = inputs["labels"].tolist()
    rows = inputs["rows"].tolist()
    start_view = readouts["A4700"]["evaluator_view"]
    a_view = readouts["A6700"]["evaluator_view"]
    b_view = readouts["B6700"]["evaluator_view"]
    target_view = diagnostic.canonical.cmos_view(readouts["A4700"]["target_slm_intensity"])
    scale, cell_h, cell_w = 4, 50, 4 * 120 + 220
    for page, start in enumerate(range(0, MEASURE_COUNT, 25), start=1):
        stop = min(start + 25, MEASURE_COUNT)
        canvas = Image.new("RGB", (cell_w, 36 + (stop - start) * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        headers = ("target", "A4700", "A6700", "B6700")
        for column, label in enumerate(headers):
            draw.text((8 + column * 120, 8), label, fill="black")
        for local, index in enumerate(range(start, stop)):
            y = 34 + local * cell_h
            views = (target_view, start_view, a_view, b_view)
            for column, tensor in enumerate(views):
                pixels = tensor[index, 0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                image = Image.fromarray(pixels, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
                canvas.paste(image, (8 + column * 120, y))
            draw.text((4 * 120 + 8, y + 2), f"pos={index} row={rows[index]} y={labels[index]} role={'fit' if index < FIT_COUNT else 'confirm'}", fill="black")
        path = target / f"gallery_{page:02d}.png"
        if path.exists():
            from PIL import ImageChops

            previous = Image.open(path).convert("RGB")
            _require(previous.size == canvas.size and ImageChops.difference(previous, canvas).getbbox() is None, f"saved gallery page {page} differs from the fixed readouts")
        else:
            canvas.save(path)
        page_paths.append(path.name)
    return page_paths


def _finalize(target: Path, inputs: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]], evaluators: Mapping[str, torch.nn.Module], preflight: Mapping[str, Any], control: ResearchRunControl) -> dict[str, Any]:
    readouts: dict[str, dict[str, Any]] = {"A4700": torch.load(target / "readouts_start.pt", map_location="cpu", weights_only=True)}
    summaries: dict[str, dict[str, Any]] = {"A4700": _metrics_from_saved(readouts["A4700"], inputs, "A4700")}
    for name, arm_key in (("A6700", ARM_A), ("B6700", ARM_B)):
        summary, readout, _rows = _endpoint(target, name, arms[arm_key], inputs, evaluators)
        summaries[name] = _metrics_from_saved(readout, inputs, name)
        readouts[name] = readout
    replay = stroke._replay_check(readouts["A4700"], inputs["readout"])
    _require(replay["passed"], "final A4700 saved start replay differs")
    a_result = summaries["A6700"]
    b_result = summaries["B6700"]
    start_result = summaries["A4700"]
    valid = all(summary.get("all_finite") is True for summary in summaries.values())
    valid = valid and arms[ARM_A]["completed_updates"] == arms[ARM_B]["completed_updates"] == FINAL_STEP
    valid = valid and arms[ARM_A]["optimizer_step"] == arms[ARM_B]["optimizer_step"] == FINAL_STEP
    _require(valid, "final endpoint readout, finite-state, or update-count validity check failed")
    decision = _final_decision(start_result, a_result, b_result, valid=valid)
    gallery_pages = _gallery(target, inputs, readouts)
    control.complete()
    classification = decision["classification"]
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED" if valid else "INVALID_HARNESS",
        "classification": classification,
        "source_status": "parent stroke fidelity remains REVIEW_INCONCLUSIVE / REVIEW_EVIDENCE_INSUFFICIENT; not reclassified",
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "source_step": SOURCE_STEP,
        "updates_per_arm": ADDED_UPDATES,
        "final_generator_adam_step": FINAL_STEP,
        "scientific_updates": 2 * ADDED_UPDATES,
        "arms": {
            ARM_A: {"objective": "raw 200x200 MSE + 0.1*(1-SSIM), paired with original target", "endpoint": "A6700"},
            ARM_B: {"objective": "raw 200x200 MSE + 0.1*(1-SSIM), paired with fixed per-fit-row energy-matched target", "endpoint": "B6700"},
        },
        "target_scale_coefficients": inputs["scale_evidence"],
        "population": inputs["population"],
        "metrics": summaries,
        "decision": decision,
        "source_A4700_replay": replay,
        "preflight": preflight,
        "gallery_pages": gallery_pages,
        "human_review": "none; automatic screening only",
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "stop": True,
        "research_run_state": control.state,
    }
    _write_json_new(target / "report.json", report)
    return report


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    stroke_dir = _resolve(root, args.stroke_dir)
    target = _resolve(root, args.output_dir)
    _require(bool(args.resume) == (args.checkpoint is not None), "--resume requires an explicit --checkpoint and vice versa")
    _require(not (args.preflight_only and args.resume), "--preflight-only cannot be combined with --resume")
    _target_directory(root, target, resume=args.resume)
    inputs = _load_population(root, stroke_dir)
    population_path = target / "input_population.json"
    if population_path.exists():
        _require(json.loads(population_path.read_text(encoding="utf-8")) == inputs["population"], "staged task population differs")
    else:
        _write_json_new(population_path, inputs["population"])
    _save_coefficients(target, inputs)
    _source_history(target, inputs)
    started = time.perf_counter()
    selected = _read_checkpoint(_resolve(root, args.checkpoint), target, inputs) if args.resume else None
    if args.resume and (target / "report.json").is_file():
        _require(selected["arm"] == ARM_B and selected["task_local_updates"] == ADDED_UPDATES, "terminal recovery requires explicit B6700 checkpoint")
        return json.loads((target / "report.json").read_text(encoding="utf-8"))

    device = torch.device("cuda:0")
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == NUMERIC_MODE, "active strict FP32 settings differ")
        evaluators = _load_evaluators(inputs, device)
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        if args.resume:
            _control, _receipt, preflight = _validate_ready(target, inputs)
        elif (target / "preflight.json").is_file():
            _control, _receipt, preflight = _validate_ready(target, inputs)
        else:
            preflight, start_readout, target_control = _preflight(inputs, target)
            control, receipt = _make_receipt(target, inputs, preflight)
            _stage_ready(target, inputs, preflight, receipt, resume=False)
            if args.preflight_only:
                return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "checks": preflight["checks"], "next_action": "NONE"}
        control, receipt, preflight = _validate_ready(target, inputs)
        _require((target / "readouts_start.pt").is_file() and (target / "target_positive_control.pt").is_file(), "READY lacks source readout or evaluator target controls")
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "checks": preflight["checks"], "next_action": "NONE"}
        control.start_execution()
        control.open_scientific_attempt(target, f"{TASK_ID}:attempt-{_attempt_number(target):02d}", receipt, resume=True)
        if selected is None and (target / "report.json").is_file():
            return json.loads((target / "report.json").read_text(encoding="utf-8"))
        arms = _run_training(target, inputs, evaluators, selected)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final frozen evaluator {name}")
        report = _finalize(target, inputs, arms, evaluators, preflight, control)
        report["duration_seconds_observation_only"] = time.perf_counter() - started
        _replace_json(target / "report.json", report)
        return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stroke-dir", type=Path, default=DEFAULT_STROKE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit task-local checkpoint required for recovery")
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    root = _root()
    target = _resolve(root, args.output_dir)
    try:
        return _execute(args)
    except TrainingInterrupted as error:
        return {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume from the explicit durable checkpoint"}
    except Exception as error:
        if target.is_dir() and not (target / "report.json").is_file() and not (target / "invalid_harness.json").is_file():
            checkpoint_names = sorted(path.name for path in target.glob("arm_[ab]_step*.pt"))
            failure = {
                "task_id": TASK_ID,
                "status": "INVALID_HARNESS",
                "classification": None,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "last_durable_checkpoint": checkpoint_names[-1] if checkpoint_names else None,
                "scientific_conclusion": None,
                "next_action": "NONE",
            }
            try:
                _write_json_new(target / "invalid_harness.json", failure)
            except OSError:
                pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
