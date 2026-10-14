"""Run a fixed step-2200 Generator-only feature-distance versus class-CE contrast."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import signal
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_balanced_reconstruction_v1 as balanced


TASK_ID = "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_semantic_target_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_semantic_target_comparison_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_semantic_target_comparison_checkpoint.v1"
FIT_ROWS = 200
HISTORIC_ROWS = 100
CONFIRMATION_ROWS = 300
TOTAL_ROWS = FIT_ROWS + HISTORIC_ROWS + CONFIRMATION_ROWS
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
UPDATES_PER_ARM = 300
CHECKPOINT_INTERVAL = 100
CONFIRMATION_SEED = 2026100405
BOOTSTRAP_SEED = 2026100406
DELTA_SIC = 1.365988374680088e-5
KAPPA = 0.6936758419764932
ALPHA = 0.01596223111734585
BETA = 2.4509953404250195
GAMMA = 0.5330424284576345
ARM_NAMES = ("A_feature", "B_class_ce")
FOCUS_CLASSES = (1, 7)
DISPLAY_CLASSES = (1, 7, 0, 4, 8)
EVALUATOR_NAMES = ("E_S", "E_I", "E_C", "E_D")
ROLE_SLICES = {"fit": (0, FIT_ROWS), "historic_holdout": (FIT_ROWS, FIT_ROWS + HISTORIC_ROWS), "confirmation": (FIT_ROWS + HISTORIC_ROWS, TOTAL_ROWS)}
DEFAULT_DIAGNOSTIC = Path("outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01")
DEFAULT_OUTPUT = Path("outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01")
CHECKPOINT_PATTERN = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_TELEMETRY_NAME = "resource_telemetry_semantic_targets.jsonl"


class SemanticTargetRunError(RuntimeError):
    """Invalid input, failed runtime invariant, or incomplete task result."""


class TrainingInterrupted(RuntimeError):
    """A user interrupt was received and a resumable boundary was saved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticTargetRunError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _cpu_tree(value: Any) -> Any:
    return balanced._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return balanced._tree_equal(left, right)


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return balanced._state_snapshot(module)


def _assert_state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], name: str) -> None:
    balanced._assert_state_equal(left, right, name)


def _rng_state() -> dict[str, Any]:
    return balanced._rng_state()


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    balanced._restore_rng_state(state)


def _sample_resource_telemetry(target: Path, device: torch.device, phase: str, *, progress: Mapping[str, Any] | None = None, started: float | None = None) -> dict[str, Any]:
    sample = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress or {}),
        "elapsed_seconds_observation_only": None if started is None else time.perf_counter() - started,
        "resource": diagnostic._resource_snapshot(device),
    }
    path = target / RESOURCE_TELEMETRY_NAME
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    _require(sample["resource"]["global_used_fraction"] <= 0.90, "PREFLIGHT_BLOCKED(resource): whole-device use exceeded the 90% safety line")
    return sample


def _confirmation_indices(
    source_labels: torch.Tensor,
    original_indices: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    seed: int = CONFIRMATION_SEED,
) -> torch.Tensor:
    _require(source_labels.shape == (60_000,) and source_labels.dtype == torch.long, "source labels must describe canonical 60,000-row MNIST TRAIN")
    _require(original_indices.shape == (300,) and torch.unique(original_indices).numel() == 300, "original diagnostic rows must contain 300 unique indices")
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    original_mask = torch.zeros(60_000, dtype=torch.bool)
    original_mask[original_indices.cpu()] = True
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class: dict[int, torch.Tensor] = {}
    for class_id in range(10):
        candidates = torch.nonzero(train_mask & ~original_mask & (source_labels == class_id), as_tuple=False).flatten()
        candidates = candidates.sort().values
        _require(candidates.numel() >= 30, f"class {class_id} has fewer than 30 unused canonical TRAIN candidates")
        order = torch.randperm(candidates.numel(), generator=generator)
        by_class[class_id] = candidates.index_select(0, order[:30])
    interleaved = torch.stack([torch.stack([by_class[class_id][row] for class_id in range(10)]) for row in range(30)]).reshape(-1)
    _require(interleaved.shape == (CONFIRMATION_ROWS,), "new confirmation population has the wrong size")
    _require(torch.unique(interleaved).numel() == CONFIRMATION_ROWS, "new confirmation rows contain duplicates")
    _require(not set(interleaved.tolist()).intersection(original_indices.tolist()), "new confirmation overlaps the original 300 rows")
    _require(bool(train_mask.index_select(0, interleaved).all().item()), "new confirmation contains a non-TRAIN row")
    _require(torch.bincount(source_labels.index_select(0, interleaved), minlength=10).tolist() == [30] * 10, "new confirmation is not 30 rows per class")
    return interleaved


def training_batch(images: torch.Tensor, labels: torch.Tensor, batch_index: int, *, data_role: str = "train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(data_role == "train", "only canonical TRAIN fit rows may enter an update")
    _require(images.shape[0] == TOTAL_ROWS and labels.shape == (TOTAL_ROWS,), "training input must preserve all 600 fixed rows and roles")
    _require(0 <= batch_index < BATCH_COUNT, "training batch index is outside the 20-batch fit order")
    positions = torch.arange(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE, dtype=torch.long)
    _require(bool((positions < FIT_ROWS).all().item()), "historic holdout and confirmation rows are forbidden in training gradients")
    batch_labels = labels.index_select(0, positions)
    _require(torch.bincount(batch_labels, minlength=10).tolist() == [1] * 10, "fixed fit batch is not class-balanced")
    return images.index_select(0, positions), batch_labels, positions


def _diag_paths(root: Path, diagnostic_dir: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    return balanced._diag_paths(root, diagnostic_dir)


def _load_inputs(root: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    inputs = diagnostic.load_inputs(root, paths)
    data_payload = diagnostic._load_torch(paths["training_data"], mmap=True)
    source_images, source_labels = diagnostic._read_image_labels(data_payload)
    _require(torch.equal(inputs["labels"], source_labels.index_select(0, inputs["source_indices"])), "diagnostic labels no longer match canonical TRAIN metadata")
    train_indices, _validation_indices = diagnostic.stratified_mnist_indices(source_labels, seed=1234, validation_per_label=500)
    confirmation = _confirmation_indices(source_labels, inputs["source_indices"], train_indices)
    source_indices = torch.cat((inputs["source_indices"], confirmation)).contiguous()
    labels = source_labels.index_select(0, source_indices).contiguous()
    images = source_images.index_select(0, source_indices).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(images.shape == (TOTAL_ROWS, 1, 28, 28), "selected images do not match the fixed 600-row population")
    _require(bool(torch.isfinite(images).all().item()) and float(images.min()) >= 0 and float(images.max()) <= 1, "selected TRAIN images are nonfinite or outside [0,1]")
    _require(torch.bincount(labels[:FIT_ROWS], minlength=10).tolist() == [20] * 10, "original fit population changed")
    _require(torch.bincount(labels[FIT_ROWS:FIT_ROWS + HISTORIC_ROWS], minlength=10).tolist() == [10] * 10, "historic holdout population changed")
    _require(torch.bincount(labels[FIT_ROWS + HISTORIC_ROWS:], minlength=10).tolist() == [30] * 10, "new confirmation population changed")
    inputs.update({"source_indices": source_indices, "labels": labels, "images": images, "confirmation_indices": confirmation, "source_labels": source_labels})
    return inputs


def _population_record(inputs: Mapping[str, Any], paths: Mapping[str, Path], diagnostic_dir: Path) -> dict[str, Any]:
    roles = ["fit"] * FIT_ROWS + ["historic_holdout"] * HISTORIC_ROWS + ["confirmation"] * CONFIRMATION_ROWS
    return {
        "task_id": TASK_ID,
        "diagnostic_dir": diagnostic_dir.as_posix(),
        "diagnostic_classification": "STANDARD_DIGIT_PRESENTATION_CANDIDATE",
        "input_paths": {name: paths[name].as_posix() for name in sorted(paths)},
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "roles": roles,
        "original_fit_rows": FIT_ROWS,
        "historic_holdout_rows": HISTORIC_ROWS,
        "new_confirmation_rows": CONFIRMATION_ROWS,
        "confirmation_seed": CONFIRMATION_SEED,
        "confirmation_indices": inputs["confirmation_indices"].tolist(),
        "fit_batch_size": BATCH_SIZE,
        "fit_batch_count": BATCH_COUNT,
        "focus_classes": list(FOCUS_CLASSES),
        "provenance": "the original 300 rows preserve prior roles; the new 300-row population is fixed once from unused canonical TRAIN indices and is not claimed unseen to historical models",
    }


def _new_endpoint(inputs: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.optim.Optimizer, dict[str, torch.nn.Parameter], Any]:
    consumer, optimizer, parameters, references = balanced._new_endpoint(inputs, device)
    _require(consumer.runtime.config.propagation_grid_size == 2048, "the semantic objective contrast requires the production 2048 propagation grid")
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "Encoder Adam state is not at source step 2200")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2200, "Generator Adam state is not at source step 2200")
    return consumer, optimizer, parameters, references


def _evaluator_features_logits(model: torch.nn.Module, view: torch.Tensor, name: str) -> tuple[torch.Tensor | None, torch.Tensor]:
    output = model(view)
    if name == "E_D":
        return None, output
    _require(isinstance(output, (tuple, list)) and len(output) == 2, f"{name} did not return embedding and logits")
    return output[0], output[1]


def _semantic_losses(raw: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, evaluators: Mapping[str, torch.nn.Module]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    view = canonical.cmos_view(raw)
    target_view = canonical.cmos_view(images)
    feature_losses: dict[str, torch.Tensor] = {}
    class_losses: dict[str, torch.Tensor] = {}
    for name, key in (("E_S", "L_S"), ("E_I", "L_I"), ("E_C", "L_C")):
        generated_features, generated_logits = _evaluator_features_logits(evaluators[name], view, name)
        with torch.no_grad():
            target_features, _target_logits = _evaluator_features_logits(evaluators[name], target_view, name)
        _require(generated_features is not None and target_features is not None, f"{name} embedding is unavailable")
        feature_losses[key] = canonical.normalized_feature_loss(generated_features, target_features, key)
        class_losses[key] = F.cross_entropy(generated_logits, labels.to(generated_logits.device), reduction="mean")
    return feature_losses, class_losses


def _gradient(loss: torch.Tensor, parameters: Mapping[str, torch.nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    names = ("encoder.phase", "generator.phase")
    values = torch.autograd.grad(loss, tuple(parameters[name] for name in names), retain_graph=retain_graph, allow_unused=True)
    return {name: torch.zeros_like(parameters[name]) if value is None else value.detach() for name, value in zip(names, values, strict=True)}


def _training_gradients(
    consumer: Any,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    batch: torch.Tensor,
    labels: torch.Tensor,
    arm: str,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    breakdown = consumer.forward_loss(batch, data_role=bogt.TRAIN_ROLE)
    result = breakdown.result
    raw = result.raw_intensity
    target_raw = result.image_input.slm_intensity
    feature_losses, class_losses = _semantic_losses(raw, batch, labels, evaluators)
    selected = feature_losses if arm == ARM_NAMES[0] else class_losses
    losses: dict[str, torch.Tensor] = {"L_dist": breakdown.L_dist, "L_rec": (raw - target_raw).square().mean(), "L_op": breakdown.L_op, **selected}
    loss_names = tuple(losses)
    gradients = {name: _gradient(loss, parameters, retain_graph=index < len(loss_names) - 1) for index, (name, loss) in enumerate(losses.items())}
    canonical_gradient, unavailable = diagnostic._safe_canonical_gradient({name: gradients[name] for name in ("L_S", "L_I", "L_C")})
    _require(canonical_gradient is not None, f"three-evaluator normalized semantic direction is unavailable: {unavailable}")
    weights = consumer.weights
    _require(weights is not None, "BOGT weights are missing")
    combined = {
        name: ALPHA * gradients["L_dist"][name] + BETA * gradients["L_rec"][name] + GAMMA * gradients["L_op"][name] + KAPPA * canonical_gradient[name]
        for name in ("encoder.phase", "generator.phase")
    }
    _require(all(bool(torch.isfinite(value).all().item()) for value in combined.values()), "combined semantic-target direction is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only optical autograd populated parameter .grad buffers")
    return breakdown, losses, gradients, combined


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    update_index: int,
    *,
    arm: str,
    optimizer_labels: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    batch_cpu, true_labels, positions = training_batch(images, labels, update_index % BATCH_COUNT)
    batch = batch_cpu.to(consumer.device)
    objective_labels = true_labels if optimizer_labels is None else optimizer_labels.detach().cpu()
    _require(objective_labels.shape == true_labels.shape, "objective labels do not match the selected fit batch")
    breakdown, losses, gradients, combined = _training_gradients(consumer, parameters, evaluators, batch, objective_labels.to(consumer.device), arm)
    # Objective monitors are recorded for both arms, but only the declared arm loss supplies the normalized gradient.
    feature_losses, class_losses = _semantic_losses(breakdown.result.raw_intensity.detach(), batch, true_labels.to(consumer.device), evaluators)
    encoder_state_before = _cpu_tree(optimizer.state[parameters["encoder.phase"]])
    generator_step_before = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    optimizer.zero_grad(set_to_none=True)
    parameters["generator.phase"].grad = combined["generator.phase"].detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient was assigned to Generator-only Adam")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _require(_tree_equal(encoder_state_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]])), "Encoder Adam state changed during Generator-only update")
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "Encoder Adam step advanced")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == generator_step_before + 1, "Generator Adam did not advance exactly once")
    _require(all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters.values()), "optical phase contains a nonfinite value after Adam")
    generator_dnorm = float(gradients["L_dist"]["generator.phase"].to(torch.float64).norm().cpu())
    _require(generator_dnorm == 0.0, "L_dist produced a Generator gradient despite its declared structural zero")
    record = {
        "update": update_index + 1,
        "batch_index": int(positions[0].item() // BATCH_SIZE),
        "fit_positions": positions.tolist(),
        "source_row_indices": positions.tolist() if source_indices is None else source_indices.index_select(0, positions).tolist(),
        "labels": true_labels.tolist(),
        "optimizer_labels": objective_labels.tolist(),
        "arm": arm,
        "semantic_objective": "three-evaluator_normalized_feature_distance" if arm == ARM_NAMES[0] else "mean_three_evaluator_cross_entropy",
        "semantic_feature_losses": {key: float(value.detach().cpu()) for key, value in feature_losses.items()},
        "semantic_class_cross_entropies_true_labels": {key: float(value.detach().cpu()) for key, value in class_losses.items()},
        "semantic_objective_loss": float(torch.stack([losses[key] for key in ("L_S", "L_I", "L_C")]).detach().mean().cpu()),
        "raw_reconstruction_mse": float(losses["L_rec"].detach().cpu()),
        "l_dist": float(losses["L_dist"].detach().cpu()),
        "l_op": float(losses["L_op"].detach().cpu()),
        "encoder_gradient_norm_read_only": float(combined["encoder.phase"].to(torch.float64).norm().cpu()),
        "generator_gradient_norm": float(combined["generator.phase"].to(torch.float64).norm().cpu()),
        "l_dist_generator_gradient_norm": generator_dnorm,
        "encoder_adam_step": 2200,
        "generator_adam_step": generator_step_before + 1,
    }
    del breakdown, losses, gradients, combined, feature_losses, class_losses
    return record


def _parameters_by_name(consumer: Any) -> dict[str, torch.nn.Parameter]:
    return dict(consumer.runtime.named_parameters())


def _forward_all(consumer: Any, evaluators: Mapping[str, torch.nn.Module], images: torch.Tensor, source_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    rows = images.shape[0]
    z = torch.empty((rows, 100), dtype=torch.complex64)
    raw = torch.empty((rows, 1, 200, 200), dtype=torch.float32)
    target_raw = torch.empty_like(raw)
    power: torch.Tensor | None = None
    logits = {name: torch.empty((rows, 10), dtype=torch.float32) for name in EVALUATOR_NAMES}
    with torch.no_grad():
        for start in range(0, rows, BATCH_SIZE):
            stop = min(rows, start + BATCH_SIZE)
            result = consumer.runtime.forward_from_image(images[start:stop].to(consumer.device))
            view = canonical.cmos_view(result.raw_intensity)
            for name in EVALUATOR_NAMES:
                _features, batch_logits = _evaluator_features_logits(evaluators[name], view, name)
                logits[name][start:stop].copy_(batch_logits.detach().cpu())
            z[start:stop].copy_(result.z_ols.coefficients.detach().cpu())
            raw[start:stop].copy_(result.raw_intensity.detach().cpu())
            current_power = bogt.contracted_plane_powers(result).detach().cpu()
            if power is None:
                power = torch.empty((rows, *current_power.shape[1:]), dtype=current_power.dtype)
            power[start:stop].copy_(current_power)
            _require(result.image_input is not None, "full optical path omitted the target SLM input")
            target_raw[start:stop].copy_(result.image_input.slm_intensity.detach().cpu())
            del result, view, current_power
    _require(power is not None, "optical forward omitted plane-power measurements")
    for name, tensor in (("z", z), ("raw_cmos", raw), ("target_raw", target_raw), ("power_by_plane", power), *logits.items()):
        diagnostic._finite_tensor(tensor.real if torch.is_complex(tensor) else tensor, f"{name} readout")
    for name, model in evaluators.items():
        _require(all(parameter.grad is None for parameter in model.parameters()), f"{name} acquired gradients during endpoint readout")
    return {
        "z": z,
        "raw_cmos": raw,
        "target_slm_intensity": target_raw,
        "power_by_plane": power,
        "source_indices": source_indices.clone(),
        "labels": labels.clone(),
        **{f"{name.lower()}_logits": value for name, value in logits.items()},
    }


def _role_positions() -> dict[str, torch.Tensor]:
    return {name: torch.arange(start, stop, dtype=torch.long) for name, (start, stop) in ROLE_SLICES.items()}


def _class_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=1).cpu()
    labels_cpu = labels.cpu()
    role_metrics = {}
    for role, positions in _role_positions().items():
        role_metrics[role] = diagnostic._metrics_with_intervals(labels_cpu.index_select(0, positions), predictions.index_select(0, positions))
    role_metrics["combined"] = diagnostic._metrics_with_intervals(labels_cpu, predictions)
    return {"by_role": role_metrics, "predictions": predictions}


def _target_logits(evaluators: Mapping[str, torch.nn.Module], images: torch.Tensor) -> dict[str, torch.Tensor]:
    view = canonical.cmos_view(images)
    output: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in EVALUATOR_NAMES:
            model = evaluators[name]
            device = next(model.parameters()).device
            logits_by_batch = []
            for start in range(0, view.shape[0], 128):
                _features, logits = _evaluator_features_logits(model, view[start : start + 128].to(device), name)
                logits_by_batch.append(logits.detach().cpu())
            output[name] = torch.cat(logits_by_batch, dim=0)
    return output


def _positive_control_ok(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    for name in EVALUATOR_NAMES:
        confirmation = metrics[name]["by_role"]["confirmation"]
        if float(confirmation["macro_recall"]) < 0.90:
            return False
        if any(int(confirmation["per_class_correct"][class_id]) < 24 for class_id in FOCUS_CLASSES):
            return False
    return True


def _preflight(inputs: Mapping[str, Any], paths: Mapping[str, Path], target: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "PREFLIGHT_BLOCKED(resource): CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    disk = _check_output_envelope(target, paths["step2200"])
    cuda_before = diagnostic._check_gpu_budget(device)
    started = time.perf_counter()
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    target_logits = _target_logits(evaluators, inputs["images"])
    target_metrics = {name: _class_metrics(inputs["labels"], value) for name, value in target_logits.items()}
    target_control_met = _positive_control_ok(target_metrics)
    _require(target_control_met, "target-image positive control did not reach the declared per-evaluator confirmation thresholds; formal updates are not started")
    initial_rng = _rng_state()
    trials: dict[str, Any] = {}
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for trial_name, arm in (("A_primary", ARM_NAMES[0]), ("A_replay", ARM_NAMES[0]), ("B_class_ce", ARM_NAMES[1])):
        _restore_rng_state(initial_rng)
        consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
        snapshots[trial_name] = []
        records = []
        encoder_state = parameters["encoder.phase"].detach().cpu().clone()
        encoder_adam = _cpu_tree(optimizer.state[parameters["encoder.phase"]])
        frozen = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
        for update in range(3):
            diagnostic._check_gpu_budget(device)
            record = _train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], update, arm=arm, source_indices=inputs["source_indices"])
            records.append(record)
            _require(torch.equal(encoder_state, parameters["encoder.phase"].detach().cpu()), f"{trial_name} changed Encoder phase")
            _assert_state_equal(encoder_adam, _cpu_tree(optimizer.state[parameters["encoder.phase"]]), f"{trial_name} Encoder Adam state")
            _assert_state_equal(frozen, {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}, f"{trial_name} frozen optical state")
            _require(all(parameter.grad is None for parameter in parameters.values()), f"{trial_name} retained optical gradients")
            for name, model in evaluators.items():
                _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} violated the frozen evaluator boundary")
            snapshots[trial_name].append({"model": {name: value.detach().cpu().clone() for name, value in parameters.items()}, "optimizer": _cpu_tree(optimizer.state_dict())})
        trials[trial_name] = {"arm": arm, "updates": 3, "records": records, "encoder_adam_step": 2200, "generator_adam_step": 2203}
        del consumer, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
    repeatability = []
    for first, replay in zip(snapshots["A_primary"], snapshots["A_replay"], strict=True):
        okay, detail = _repeatability_difference(first, replay)
        _require(okay, f"A/A CUDA repeatability check failed: {detail}")
        repeatability.append(detail)

    # Wrong-label negative control: same generated batch and frozen model, different CE target.
    _restore_rng_state(initial_rng)
    consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
    fit_images, true_labels, _positions = training_batch(inputs["images"], inputs["labels"], 0)
    batch = fit_images.to(device)
    true_labels_device = true_labels.to(device)
    wrong_labels = true_labels.roll(1).to(device)
    breakdown = consumer.forward_loss(batch, data_role=bogt.TRAIN_ROLE)
    _features, true_logits_losses = _semantic_losses(breakdown.result.raw_intensity, batch, true_labels_device, evaluators)
    _features_wrong, wrong_logits_losses = _semantic_losses(breakdown.result.raw_intensity, batch, wrong_labels, evaluators)
    true_grad = {name: _gradient(loss, parameters, retain_graph=True) for name, loss in true_logits_losses.items()}
    wrong_grad = {name: _gradient(loss, parameters, retain_graph=True) for name, loss in wrong_logits_losses.items()}
    gradient_differences = {
        name: float((true_grad[name]["generator.phase"] - wrong_grad[name]["generator.phase"]).to(torch.float64).norm().cpu())
        for name in true_grad
    }
    _require(all(value > 0.0 and math.isfinite(value) for value in gradient_differences.values()), "wrong-label negative control did not change every evaluator CE gradient")
    _require(all(parameter.grad is None for parameter in parameters.values()), "negative-control gradient populated optical .grad buffers")
    del consumer, optimizer, parameters, breakdown, true_grad, wrong_grad, true_logits_losses, wrong_logits_losses, _features, _features_wrong
    gc.collect()
    torch.cuda.empty_cache()

    # Save after one disposable A step, reload through the production state restoration path, and compare the next two updates.
    _restore_rng_state(initial_rng)
    consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
    history: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
    history[ARM_NAMES[0]].append(_train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], 0, arm=ARM_NAMES[0], source_indices=inputs["source_indices"]))
    recovery_population = {"task_id": TASK_ID, "recovery_probe": True}
    recovery_payload = _checkpoint_payload(consumer, optimizer, ARM_NAMES[0], 1, history, recovery_population)
    with tempfile.TemporaryDirectory(prefix="lightgen_semantic_recovery_", dir=target.parent) as temp_dir:
        path = Path(temp_dir) / "recovery.pt"
        diagnostic._write_torch_new(path, recovery_payload, "disposable CUDA interruption recovery preflight")
        restored_payload = torch.load(path, map_location="cpu", weights_only=True)
    continuous = []
    for update in (1, 2):
        continuous.append(_train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], update, arm=ARM_NAMES[0], source_indices=inputs["source_indices"]))
    continuous_snapshot = {"model": {name: value.detach().cpu().clone() for name, value in parameters.items()}, "optimizer": _cpu_tree(optimizer.state_dict())}
    del consumer, optimizer, parameters
    consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
    _restore_checkpoint(restored_payload, consumer, optimizer)
    resumed = []
    for update in (1, 2):
        resumed.append(_train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], update, arm=ARM_NAMES[0], source_indices=inputs["source_indices"]))
    resumed_snapshot = {"model": {name: value.detach().cpu().clone() for name, value in parameters.items()}, "optimizer": _cpu_tree(optimizer.state_dict())}
    recovery_ok, recovery_detail = _repeatability_difference(continuous_snapshot, resumed_snapshot)
    _require(recovery_ok, f"checkpoint interruption/resume path did not reproduce continuous updates: {recovery_detail}")
    _require([row["batch_index"] for row in resumed] == [1, 2] and int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2203, "resume did not continue from the saved next-batch cursor and update count")
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()

    # The final decision procedure must refuse to claim support on identical A/A predictions.
    target_predictions = target_logits["E_D"].argmax(dim=1)
    a_a_check = _same_output_decision_check(inputs["labels"], target_predictions)
    _require(a_a_check["support_claim"] is False, "identical A/A predictions incorrectly pass the decision function")
    for name, model in evaluators.items():
        _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"preflight {name} evaluator")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight {name} evaluator is not frozen")
    _restore_rng_state(initial_rng)
    cuda_after = diagnostic._check_gpu_budget(device)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "status": "READY",
        "device": str(device),
        "propagation_grid": 2048,
        "updates_per_arm": UPDATES_PER_ARM,
        "disposable_updates": 14,
        "disk_observation": disk,
        "cuda_before": cuda_before,
        "cuda_after": cuda_after,
        "target_image_positive_control": {name: {"confirmation": target_metrics[name]["by_role"]["confirmation"]} for name in EVALUATOR_NAMES},
        "target_image_positive_control_thresholds_met": target_control_met,
        "three_step_trials": trials,
        "a_a_repeatability": {"passed": True, "tolerances": {"floating_rtol": 2e-5, "floating_atol": 2e-7}, "updates": repeatability},
        "wrong_label_negative_control": {"passed": True, "true_vs_wrong_label_generator_gradient_norm": gradient_differences},
        "interruption_recovery": {"passed": True, "continuous_vs_resumed": recovery_detail, "next_batch_indices": [1, 2], "generator_step": 2203},
        "same_prediction_decision_check": a_a_check,
        "frozen_evaluators_and_scientific_buffers_checked": True,
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    del evaluators
    return preflight


def _check_output_envelope(target: Path, source_checkpoint: Path) -> dict[str, Any]:
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = int(shutil.disk_usage(existing).free)
    # Three six-hundred-row endpoint payloads plus eight full Adam checkpoints, row metrics, montage and logs.
    expected = 8 * source_checkpoint.stat().st_size + 3 * TOTAL_ROWS * 2 * 200 * 200 * 4 + 750_000_000
    safety = 1_610_612_736
    required = expected + safety
    _require(free >= required, f"PREFLIGHT_BLOCKED(resource): output volume has {free} bytes free but needs {required} including the 1.5 GiB safety margin")
    return {"checked_path": existing.as_posix(), "free_bytes": free, "estimated_remaining_output_bytes": expected, "safety_margin_bytes": safety, "required_free_bytes": required}


def _arm_order(arm: str, completed: int) -> int:
    return completed if arm == ARM_NAMES[0] else UPDATES_PER_ARM + 1 + completed


def _checkpoint_name(arm: str, completed: int) -> str:
    letter = "a" if arm == ARM_NAMES[0] else "b"
    return f"arm_{letter}_step{completed:04d}.pt"


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    completed: int,
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = _parameters_by_name(consumer)
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "checkpoint Encoder Adam step changed")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2200 + completed, "checkpoint Generator Adam step differs from completed updates")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "completed_updates": completed,
        "next_batch_index": completed % BATCH_COUNT,
        "source_checkpoint_step": 2200,
        "population": dict(population),
        "trainable_parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_step_by_parameter": {"encoder.phase": 2200, "generator.phase": 2200 + completed},
        "frozen_runtime_state": {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")},
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": {key: [dict(record) for record in records] for key, records in history.items()},
        "rng_state": _rng_state(),
    }


def _write_torch_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    diagnostic._write_torch_new(path, payload, kind)


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Mapping[str, Sequence[Mapping[str, Any]]], population: Mapping[str, Any]) -> Path:
    path = target / _checkpoint_name(arm, completed)
    diagnostic._check_output_budget(target)
    _write_torch_new(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population), "Generator-only semantic-target recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.parent.resolve() == target.resolve() and path.is_file(), "resume checkpoint must be an explicit file inside this attempt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and payload.get("schema") == CHECKPOINT_SCHEMA, "resume checkpoint schema mismatch")
    _require(payload.get("task_id") == TASK_ID and payload.get("source_checkpoint_step") == 2200, "resume checkpoint task or source step mismatch")
    _require(payload.get("population") == dict(population), "resume checkpoint population or data roles changed")
    arm, completed = payload.get("arm"), payload.get("completed_updates")
    _require(arm in ARM_NAMES and isinstance(completed, int) and 0 <= completed <= UPDATES_PER_ARM, "resume checkpoint arm/update count is invalid")
    _require(payload.get("next_batch_index") == completed % BATCH_COUNT, "resume checkpoint batch cursor changed")
    _require(path.name == _checkpoint_name(arm, completed), "resume checkpoint filename and state differ")
    selected_order = _arm_order(arm, completed)
    later = []
    for candidate in target.glob("arm_*_step*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        candidate_arm = ARM_NAMES[0] if match.group(1) == "a" else ARM_NAMES[1]
        candidate_step = int(match.group(2))
        if _arm_order(candidate_arm, candidate_step) > selected_order:
            later.append(candidate.name)
    _require(not later, f"explicit checkpoint is older than already saved state: {sorted(later)}")
    return dict(payload)


def _completed_resume_result(target: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "saved terminal report is invalid")
    _require(checkpoint.get("arm") == ARM_NAMES[1] and int(checkpoint.get("completed_updates", -1)) == UPDATES_PER_ARM, "terminal resume requires the explicit final B checkpoint")
    return dict(report)


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    consumer.load_trainable_state_dict(payload["trainable_parameter_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    parameters = _parameters_by_name(consumer)
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "restored Encoder Adam step changed")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2200 + int(payload["completed_updates"]), "restored Generator Adam step changed")
    _restore_rng_state(payload["rng_state"])
    frozen = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
    _assert_state_equal(payload["frozen_runtime_state"], frozen, "restored scientific buffers")


def _repeatability_difference(left: Any, right: Any, path: str = "root") -> tuple[bool, dict[str, Any]]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype):
            return False, {"path": path, "reason": "tensor shape or dtype differs"}
        if left.is_floating_point() or left.is_complex():
            difference = float((left - right).abs().max().item()) if left.numel() else 0.0
            return torch.allclose(left, right, rtol=2e-5, atol=2e-7), {"path": path, "max_absolute_difference": difference, "rtol": 2e-5, "atol": 2e-7}
        equal = torch.equal(left, right)
        return equal, {"path": path, "exact": equal}
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right)):
            return False, {"path": path, "reason": "mapping keys differ"}
        for key in left:
            equal, detail = _repeatability_difference(left[key], right[key], f"{path}.{key}")
            if not equal:
                return False, detail
        return True, {"path": path, "children_checked": len(left)}
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not (isinstance(left, type(right)) and len(left) == len(right)):
            return False, {"path": path, "reason": "sequence length or type differs"}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            equal, detail = _repeatability_difference(left_item, right_item, f"{path}[{index}]")
            if not equal:
                return False, detail
        return True, {"path": path, "children_checked": len(left)}
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        equal = isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), rel_tol=2e-5, abs_tol=2e-7)
        return equal, {"path": path, "left": left, "right": right}
    return left == right, {"path": path, "left": left, "right": right}


def _run_arm(
    arm: str,
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, torch.nn.Module],
    target: Path,
    population: Mapping[str, Any],
    history: dict[str, list[dict[str, Any]]],
    *,
    start_from: Mapping[str, Any] | None,
    initial_rng: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    if start_from is None:
        _restore_rng_state(initial_rng)
    consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
    if start_from is not None:
        _require(start_from.get("arm") == arm, "resume checkpoint arm is not next in the declared execution order")
        _restore_checkpoint(start_from, consumer, optimizer)
        completed = int(start_from["completed_updates"])
    else:
        completed = 0
        _save_checkpoint(target, consumer, optimizer, arm, 0, history, population)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
    encoder_adam_before = _cpu_tree(optimizer.state[parameters["encoder.phase"]])
    stop_requested = False
    prior_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while completed < UPDATES_PER_ARM:
            diagnostic._check_gpu_budget(device)
            record = _train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], completed, arm=arm)
            completed += 1
            history[arm].append(record)
            interrupted = stop_requested
            if completed % CHECKPOINT_INTERVAL == 0 or interrupted:
                checkpoint = _save_checkpoint(target, consumer, optimizer, arm, completed, history, population)
                _sample_resource_telemetry(target, device, f"{arm}_checkpoint", progress={"arm_updates": completed, "checkpoint": checkpoint.name}, started=started)
                _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} changed Encoder phase")
                _assert_state_equal(encoder_adam_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]]), f"{arm} Encoder Adam")
                _assert_state_equal(frozen_before, {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}, f"{arm} frozen scientific buffers")
                for name, model in evaluators.items():
                    _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} {name} evaluator")
                    _require(all(parameter.grad is None for parameter in model.parameters()), f"{arm} {name} retained gradients")
                gc.collect()
                torch.cuda.empty_cache()
            if interrupted:
                raise TrainingInterrupted(f"user interrupt saved explicit recovery checkpoint {checkpoint.name}")
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    _require(completed == UPDATES_PER_ARM and len(history[arm]) == UPDATES_PER_ARM, f"{arm} did not complete exactly 300 updates")
    _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} changed Encoder parameters")
    _assert_state_equal(encoder_adam_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]]), f"{arm} final Encoder Adam")
    final_path = target / f"readouts_{arm}.pt"
    if final_path.is_file():
        readouts = torch.load(final_path, map_location="cpu", weights_only=True)
        _validate_readout_population(readouts, inputs, arm)
        if not (target / f"image_statistics_{arm}.json").is_file() or not (target / f"row_metrics_{arm}.pt").is_file():
            _save_measurements(target, arm, {"readouts": readouts}, inputs)
    else:
        readouts = _forward_all(consumer, evaluators, inputs["images"], inputs["source_indices"], inputs["labels"])
        result = {"readouts": readouts, "image_statistics": _image_statistics(readouts, inputs)}
        _save_measurements(target, arm, result, inputs)
    result = {"arm": arm, "completed_updates": completed, "readouts": readouts, "image_statistics": json.loads((target / f"image_statistics_{arm}.json").read_text(encoding="utf-8"))}
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _validate_readout_population(readouts: Mapping[str, Any], inputs: Mapping[str, Any], name: str) -> None:
    _require(torch.is_tensor(readouts.get("source_indices")) and torch.equal(readouts["source_indices"].cpu(), inputs["source_indices"]), f"saved {name} row indices changed")
    _require(torch.is_tensor(readouts.get("labels")) and torch.equal(readouts["labels"].cpu(), inputs["labels"]), f"saved {name} row labels changed")
    _require(tuple(readouts.get("raw_cmos", torch.empty(0)).shape) == (TOTAL_ROWS, 1, 200, 200), f"saved {name} raw CMOS shape changed")
    _require(tuple(readouts.get("z", torch.empty(0)).shape) == (TOTAL_ROWS, 100) and readouts["z"].dtype == torch.complex64, f"saved {name} OLS shape/dtype changed")
    for name in EVALUATOR_NAMES:
        _require(tuple(readouts.get(f"{name.lower()}_logits", torch.empty(0)).shape) == (TOTAL_ROWS, 10), f"saved {name} logits shape changed")
    for key in ("raw_cmos", "z", "target_slm_intensity", "power_by_plane", *(f"{name.lower()}_logits" for name in EVALUATOR_NAMES)):
        value = readouts[key]
        diagnostic._finite_tensor(value.real if torch.is_complex(value) else value, f"saved {name} {key}")


def _image_statistics(readouts: Mapping[str, torch.Tensor], inputs: Mapping[str, Any]) -> dict[str, Any]:
    per_row = diagnostic._row_reconstruction_metrics(readouts["raw_cmos"], readouts["target_slm_intensity"], inputs["images"])
    raw_mse = (readouts["raw_cmos"] - readouts["target_slm_intensity"]).square().flatten(1).mean(dim=1)
    generated_view = canonical.cmos_view(readouts["raw_cmos"])
    target_view = canonical.cmos_view(inputs["images"])
    view_mse = (generated_view - target_view).square().flatten(1).mean(dim=1)
    per_row = {**per_row, "raw_mse": raw_mse.cpu(), "evaluator_view_mse": view_mse.cpu()}
    labels = inputs["labels"].cpu()
    roles: dict[str, Any] = {}
    group_positions = {**_role_positions(), "combined": torch.arange(TOTAL_ROWS)}
    for role, role_positions in group_positions.items():
        role_summary: dict[str, Any] = {}
        label_ids = range(10)
        for class_id in label_ids:
            positions = role_positions[labels.index_select(0, role_positions) == class_id]
            selected: dict[str, Any] = {}
            for key in ("raw_mse", "raw_foreground_mse", "raw_background_mse", "evaluator_view_mse", "evaluator_view_foreground_mse", "evaluator_view_background_mse", "raw_fraction_above_one"):
                values = per_row[key].index_select(0, positions).to(torch.float64)
                selected[key] = float(values.mean().item())
            for key in ("raw_zero_image", "raw_constant_image"):
                selected[key] = int(per_row[key].index_select(0, positions).sum().item())
            selected["rows"] = int(positions.numel())
            selected["source_row_indices"] = inputs["source_indices"].index_select(0, positions).tolist()
            selected["labels"] = labels.index_select(0, positions).tolist()
            if role == "combined":
                role_summary[str(class_id)] = selected
            else:
                role_summary[str(class_id)] = selected
        all_selected: dict[str, Any] = {}
        for key in ("raw_mse", "raw_foreground_mse", "raw_background_mse", "evaluator_view_mse", "evaluator_view_foreground_mse", "evaluator_view_background_mse", "raw_fraction_above_one"):
            all_selected[key] = float(per_row[key].index_select(0, role_positions).to(torch.float64).mean().item())
        for key in ("raw_zero_image", "raw_constant_image"):
            all_selected[key] = int(per_row[key].index_select(0, role_positions).sum().item())
        all_selected["rows"] = int(role_positions.numel())
        all_selected["source_row_indices"] = inputs["source_indices"].index_select(0, role_positions).tolist()
        role_summary["overall"] = all_selected
        roles[role] = role_summary
    return {"by_role": roles, "foreground_definition": "target intensity > 0.1; raw uses 200x200 target_slm_intensity and view uses canonical 28x28 target", "raw_software_clipping_definition": "fraction of raw CMOS pixels greater than 1.0"}


def _save_measurements(target: Path, name: str, result: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    readouts = result["readouts"]
    readout_path = target / f"readouts_{name}.pt"
    if readout_path.exists():
        _require(_tree_equal(torch.load(readout_path, map_location="cpu", weights_only=True), readouts), f"saved {name} readouts differ")
    else:
        diagnostic._write_torch_new(readout_path, readouts, "six-hundred-row full optical endpoint readouts")
    row_metrics = diagnostic._row_reconstruction_metrics(readouts["raw_cmos"], readouts["target_slm_intensity"], inputs["images"])
    raw_mse = (readouts["raw_cmos"] - readouts["target_slm_intensity"]).square().flatten(1).mean(1)
    row_metrics["raw_mse"] = raw_mse.cpu()
    row_metrics["evaluator_view_mse"] = (canonical.cmos_view(readouts["raw_cmos"]) - canonical.cmos_view(inputs["images"])).square().flatten(1).mean(1).cpu()
    row_payload = {"source_indices": inputs["source_indices"], "labels": inputs["labels"], **{key: value.cpu() for key, value in row_metrics.items() if value.ndim == 1 and value.shape[0] == TOTAL_ROWS}}
    row_path = target / f"row_metrics_{name}.pt"
    if row_path.exists():
        _require(_tree_equal(torch.load(row_path, map_location="cpu", weights_only=True), row_payload), f"saved {name} per-row metrics differ")
    else:
        diagnostic._write_torch_new(row_path, row_payload, "per-row raw and evaluator-view metrics for all fixed populations")
    statistics = _image_statistics(readouts, inputs)
    stats_path = target / f"image_statistics_{name}.json"
    if stats_path.exists():
        _require(json.loads(stats_path.read_text(encoding="utf-8")) == statistics, f"saved {name} image statistics differ")
    else:
        diagnostic._write_json_new(stats_path, statistics, "role-separated raw and evaluator-view image metrics")
    return statistics


def _save_training_history(target: Path, history: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for arm in ARM_NAMES:
        path = target / f"training_history_{arm}.json"
        payload = {"task_id": TASK_ID, "arm": arm, "updates": len(history[arm]), "records": list(history[arm])}
        if path.exists():
            _require(json.loads(path.read_text(encoding="utf-8")) == payload, f"saved {arm} training history differs")
        else:
            diagnostic._write_json_new(path, payload, "Generator-only semantic objective training history")


def _exact_mcnemar_better(a_correct: torch.Tensor, b_correct: torch.Tensor) -> dict[str, Any]:
    _require(a_correct.shape == b_correct.shape and a_correct.numel() == 30, "paired exact McNemar comparison requires 30 confirmation rows for one class")
    wins = int((b_correct & ~a_correct).sum().item())
    losses = int((a_correct & ~b_correct).sum().item())
    discordant = wins + losses
    pvalue = 1.0 if discordant == 0 else sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant)
    return {"b_only_correct": wins, "a_only_correct": losses, "discordant_pairs": discordant, "one_sided_exact_p": float(min(1.0, pvalue))}


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted


def _paired_macro_bootstrap(labels: torch.Tensor, a_predictions: torch.Tensor, b_predictions: torch.Tensor, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    labels = labels[FIT_ROWS + HISTORIC_ROWS:].cpu()
    a_predictions = a_predictions[FIT_ROWS + HISTORIC_ROWS:].cpu()
    b_predictions = b_predictions[FIT_ROWS + HISTORIC_ROWS:].cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = [torch.nonzero(labels == class_id, as_tuple=False).flatten() for class_id in range(10)]
    _require(all(group.numel() == 30 for group in positions), "paired bootstrap requires 30 confirmation examples per class")
    differences = torch.empty(10_000, dtype=torch.float64)
    for index in range(differences.numel()):
        class_differences = []
        for class_positions in positions:
            sampled = class_positions[torch.randint(0, class_positions.numel(), (class_positions.numel(),), generator=generator)]
            actual = labels.index_select(0, sampled)
            a_rate = (a_predictions.index_select(0, sampled) == actual).to(torch.float64).mean()
            b_rate = (b_predictions.index_select(0, sampled) == actual).to(torch.float64).mean()
            class_differences.append(b_rate - a_rate)
        differences[index] = torch.stack(class_differences).mean()
    bounds = torch.quantile(differences, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"resamples": 10_000, "seed": seed, "paired_stratified_macro_recall_B_minus_A_95_percentile_interval": [float(bounds[0]), float(bounds[1])], "descriptive_only": True}


def _decision(
    target_metrics: Mapping[str, Mapping[str, Any]],
    endpoint_metrics: Mapping[str, Mapping[str, Any]],
    image_statistics: Mapping[str, Mapping[str, Any]],
    paired_tests: Mapping[str, Any],
) -> dict[str, Any]:
    if not _positive_control_ok(target_metrics):
        return {"measurement_qualified": False, "classification": None, "training_recommendation": "NONE", "reason": "one or more target-image evaluator positive controls failed"}
    start = endpoint_metrics["step2200"]["by_role"]["confirmation"]
    a = endpoint_metrics[ARM_NAMES[0]]["by_role"]["confirmation"]
    b = endpoint_metrics[ARM_NAMES[1]]["by_role"]["confirmation"]
    b_counts, a_counts, start_counts = b["per_class_correct"], a["per_class_correct"], start["per_class_correct"]
    pair_gain = all(int(b_counts[c]) >= max(int(a_counts[c]), int(start_counts[c])) + 3 and float(paired_tests["holm_adjusted_p"][str(c)]) <= 0.05 for c in FOCUS_CLASSES)
    macro_gain = float(b["macro_recall"]) >= float(a["macro_recall"]) + 0.05 and float(b["macro_recall"]) >= float(start["macro_recall"])
    retention = all(int(b_counts[c]) >= max(int(a_counts[c]), int(start_counts[c])) - 3 for c in range(10))
    stats = image_statistics
    role = "confirmation"
    b_raw = float(stats[ARM_NAMES[1]]["by_role"][role]["overall"]["raw_mse"])
    a_raw = float(stats[ARM_NAMES[0]]["by_role"][role]["overall"]["raw_mse"])
    start_raw = float(stats["step2200"]["by_role"][role]["overall"]["raw_mse"])
    overall_raw = b_raw <= 1.05 * a_raw and b_raw <= 1.05 * start_raw
    clip = float(stats[ARM_NAMES[1]]["by_role"][role]["overall"]["raw_fraction_above_one"]) <= 0.01
    clip &= all(float(stats[ARM_NAMES[1]]["by_role"][role][str(c)]["raw_fraction_above_one"]) <= 0.01 for c in FOCUS_CLASSES)
    valid_images = all(int(stats[ARM_NAMES[1]]["by_role"][role]["overall"][key]) == 0 for key in ("raw_zero_image", "raw_constant_image"))
    criteria = {
        "class_1_and_7_gain_at_least_3_of_30_vs_A_and_start_with_holm_p_le_0_05": pair_gain,
        "E_D_confirmation_macro_gain_at_least_0_05_vs_A_and_not_below_start": macro_gain,
        "no_class_loses_more_than_3_of_30_vs_A_or_start": retention,
        "confirmation_raw_MSE_no_more_than_5_percent_above_A_or_start": overall_raw,
        "B_raw_clipping_at_most_1_percent_for_classes_1_7_and_overall": clip,
        "confirmation_raw_outputs_are_nonzero_nonconstant_and_finite": valid_images,
    }
    if all(criteria.values()):
        classification = "SEMANTIC_TARGET_TRANSFER_SUPPORTED"
        recommendation = "将三评价器真实类别交叉熵目标作为下一项受控主线研究候选；仅表示本次固定TRAIN确认人口、step2200初始化和300次Generator-only更新下的短程转移，不表示独立泛化或完整语义生成成功。"
        reason = "all preregistered confirmation-population support criteria were met"
    else:
        some_gain = any(int(b_counts[c]) > max(int(a_counts[c]), int(start_counts[c])) for c in FOCUS_CLASSES)
        classification = "MIXED_OR_INSUFFICIENT" if some_gain else "NOT_SUPPORTED_WITHIN_SCOPE"
        recommendation = "NONE"
        reason = "complete valid measurements did not satisfy the preregistered support criteria"
    return {"measurement_qualified": True, "classification": classification, "training_recommendation": recommendation, "reason": reason, "criteria": criteria}


def _readout_report(inputs: Mapping[str, Any], readouts_by_endpoint: Mapping[str, Mapping[str, torch.Tensor]], evaluators: Mapping[str, torch.nn.Module], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint_metrics = {
        endpoint: {name: _class_metrics(inputs["labels"], readouts[f"{name.lower()}_logits"]) for name in EVALUATOR_NAMES}
        for endpoint, readouts in readouts_by_endpoint.items()
    }
    target_logits = _target_logits(evaluators, inputs["images"])
    target_metrics = {name: _class_metrics(inputs["labels"], logits) for name, logits in target_logits.items()}
    return endpoint_metrics, {"logits": target_logits, "metrics": target_metrics}


def _same_output_decision_check(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, Any]:
    correct = predictions.cpu() == labels.cpu()
    confirm = correct[FIT_ROWS + HISTORIC_ROWS:]
    pair = {str(class_id): _exact_mcnemar_better(confirm[labels[FIT_ROWS + HISTORIC_ROWS:] == class_id], confirm[labels[FIT_ROWS + HISTORIC_ROWS:] == class_id]) for class_id in FOCUS_CLASSES}
    adjusted = _holm({key: value["one_sided_exact_p"] for key, value in pair.items()})
    no_gain = all(value == 0 for value in (pair["1"]["b_only_correct"], pair["1"]["a_only_correct"], pair["7"]["b_only_correct"], pair["7"]["a_only_correct"]))
    support_claim = not no_gain and any(value <= 0.05 for value in adjusted.values())
    return {"paired_tests": pair, "holm_adjusted_p": adjusted, "support_claim": support_claim}


def _save_montage(path: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    target_view = canonical.cmos_view(inputs["images"])
    views = {
        "target": target_view,
        "step2200": canonical.cmos_view(readouts["step2200"]["raw_cmos"]),
        ARM_NAMES[0]: canonical.cmos_view(readouts[ARM_NAMES[0]]["raw_cmos"]),
        ARM_NAMES[1]: canonical.cmos_view(readouts[ARM_NAMES[1]]["raw_cmos"]),
    }
    positions = [position for class_id in DISPLAY_CLASSES for position in torch.nonzero(inputs["labels"] == class_id, as_tuple=False).flatten().tolist()]
    _require(len(positions) == 300, "comparison montage must display all 300 rows from classes 1,7,0,4,8")
    side, label_width, gap, columns = 56, 85, 8, 2
    tile_width = label_width + len(views) * side + (len(views) - 1) * gap
    rows = math.ceil(len(positions) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * (side + 8)), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    e_d_prediction = readouts[ARM_NAMES[1]]["e_d_logits"].argmax(dim=1)
    for index, position in enumerate(positions):
        x, y = (index % columns) * tile_width, (index // columns) * (side + 8)
        role = "F" if position < FIT_ROWS else "H" if position < FIT_ROWS + HISTORIC_ROWS else "C"
        label = int(inputs["labels"][position])
        draw.text((x, y + 3), f"{label}{role} p{int(e_d_prediction[position])}", fill=(0, 0, 0))
        for view_index, view in enumerate(views.values()):
            pixels = view[position, 0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            tile = Image.fromarray(pixels, mode="L").resize((side, side), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(tile, (x + label_width + view_index * (side + gap), y))
    if path.exists():
        with Image.open(path) as existing:
            _require(existing.size == canvas.size and list(existing.convert("RGB").getdata()) == list(canvas.getdata()), "saved comparison montage differs from the fixed-scale output")
    else:
        canvas.save(path, format="PNG", optimize=True)
    return {"path": path.name, "rows_displayed": len(positions), "classes": list(DISPLAY_CLASSES), "role_codes": {"F": "original fit rows", "H": "historic 100-row comparison", "C": "new 300-row confirmation"}, "panel_order": list(views), "view": "28x28 clamp[0,1] then adaptive average pooling", "fixed_display_range": [0, 1], "per_image_rescaling": False, "source_positions": positions, "source_row_indices": inputs["source_indices"].index_select(0, torch.tensor(positions)).tolist()}


def _finalize(target: Path, inputs: Mapping[str, Any], history: Mapping[str, Sequence[Mapping[str, Any]]], readouts: Mapping[str, Mapping[str, torch.Tensor]], population: Mapping[str, Any], preflight: Mapping[str, Any], target_control: Mapping[str, Any], endpoint_metrics: Mapping[str, Any], image_statistics: Mapping[str, Any], started: float, receipt: PreflightReceipt, identity: ResearchRunIdentity) -> dict[str, Any]:
    labels = inputs["labels"].cpu()
    confirm = slice(FIT_ROWS + HISTORIC_ROWS, TOTAL_ROWS)
    confirm_labels = labels[confirm]
    a_pred = endpoint_metrics[ARM_NAMES[0]]["E_D"]["predictions"]
    b_pred = endpoint_metrics[ARM_NAMES[1]]["E_D"]["predictions"]
    paired = {
        str(class_id): _exact_mcnemar_better(
            a_pred[confirm][confirm_labels == class_id] == confirm_labels[confirm_labels == class_id],
            b_pred[confirm][confirm_labels == class_id] == confirm_labels[confirm_labels == class_id],
        )
        for class_id in FOCUS_CLASSES
    }
    adjusted = _holm({key: value["one_sided_exact_p"] for key, value in paired.items()})
    paired_statistics = {"raw": paired, "holm_adjusted_p": adjusted, "alternative": "B class-CE objective improves per-row E_D correctness over A within each confirmation class"}
    paired_statistics["macro_recall_bootstrap_vs_A"] = _paired_macro_bootstrap(labels, a_pred, b_pred)
    e_d_metrics = {name: metrics["E_D"] for name, metrics in endpoint_metrics.items()}
    decision = _decision(target_control["metrics"], e_d_metrics, image_statistics, paired_statistics)
    montage = _save_montage(target / "comparison_28x28.png", inputs, readouts)
    comparison_path = target / "comparison_28x28.json"
    if comparison_path.exists():
        _require(json.loads(comparison_path.read_text(encoding="utf-8")) == montage, "saved comparison montage metadata differs")
    else:
        diagnostic._write_json_new(comparison_path, montage, "fixed-scale 28x28 five-class role montage")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED" if decision["classification"] == "SEMANTIC_TARGET_TRANSFER_SUPPORTED" else "VALID_FAILED",
        "classification": decision["classification"],
        "training_recommendation": decision["training_recommendation"],
        "reason": decision["reason"],
        "measurement_qualified": decision["measurement_qualified"],
        "decision_criteria": decision.get("criteria"),
        "scope": {"updates_per_arm": UPDATES_PER_ARM, "fit_rows": FIT_ROWS, "historic_holdout_rows": HISTORIC_ROWS, "new_confirmation_rows": CONFIRMATION_ROWS, "all_rows_are_canonical_train": True, "new_confirmation_is_independent_of_updates_but_not_claimed_unseen_to_historical_models": True, "only_updated_parameter": "generator.phase", "encoder_ols_geometry_calibration_and_evaluators_frozen": True, "protected_test_sealed_dev_and_s1_read": False, "follow_on_training_started": False},
        "population": dict(population),
        "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "delta_SIC": DELTA_SIC, "kappa": KAPPA, "references_and_weights_inherited_from_step1200": True},
        "objectives": {ARM_NAMES[0]: "three evaluator normalized 128D feature distances", ARM_NAMES[1]: "mean of E_S/E_I/E_C cross_entropy(logits, true_label, reduction=mean)", "shared_terms": "alpha*L_dist + beta*raw_200x200_MSE + gamma*L_op + kappa*jointly normalized equal-weight three-evaluator semantic gradient", "evaluator_view": "clamp[0,1] then adaptive average pooling to 28x28", "generator_only_update": True},
        "preflight": dict(preflight),
        "ready_receipt": receipt.to_dict(),
        "research_run_identity": identity.to_dict(),
        "target_image_metrics": {name: {"by_role": {role: {key: value for key, value in summary.items() if key != "confusion_matrix_actual_rows_predicted_columns"} for role, summary in result["by_role"].items()}} for name, result in target_control["metrics"].items()},
        "target_image_positive_controls_met": _positive_control_ok(target_control["metrics"]),
        "evaluator_metrics": {endpoint: {name: {"by_role": metrics["by_role"]} for name, metrics in values.items()} for endpoint, values in endpoint_metrics.items()},
        "image_statistics": image_statistics,
        "paired_statistics": paired_statistics,
        "training_history_artifacts": {arm: f"training_history_{arm}.json" for arm in ARM_NAMES},
        "readout_artifacts": {name: f"readouts_{name}.pt" for name in readouts},
        "row_metric_artifacts": {name: f"row_metrics_{name}.pt" for name in readouts},
        "image_statistics_artifacts": {name: f"image_statistics_{name}.json" for name in readouts},
        "comparison_figure": montage,
        "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_*_step*.pt")),
        "resource_telemetry_artifact": RESOURCE_TELEMETRY_NAME,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "next_action": "NONE",
        "stop": True,
    }
    return report


def _make_identity(inputs: Mapping[str, Any], population: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> ResearchRunIdentity:
    parameters = _parameters_by_name(consumer)
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="step2200-fixed-encoder-ols-generator-only-feature-distance-versus-class-ce",
        semantic_authority_id="M0_CLASS17_MAPPING_DIAGNOSTIC_V2",
        data_role="TRAIN fit=200 update rows; historic holdout=100; fixed independent-of-update confirmation=300, all canonical TRAIN",
        config_identity={"propagation_grid": 2048, "source_step": 2200, "delta_SIC": DELTA_SIC, "kappa": KAPPA, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
        model_identity={"source_checkpoint_step": 2200, "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()}, "encoder_generator_phase_finite": True},
        optimizer_identity={"class": type(optimizer).__name__, "parameter_order": ["encoder.phase", "generator.phase"], "source_step_by_parameter": {"encoder.phase": 2200, "generator.phase": 2200}, "learning_rate": optimizer.param_groups[0]["lr"], "betas": list(optimizer.param_groups[0]["betas"])},
        sampler_identity={"fit_source_rows": population["source_row_indices"][:FIT_ROWS], "confirmation_seed": CONFIRMATION_SEED, "confirmation_rows": population["confirmation_indices"], "batch_size": BATCH_SIZE, "batch_count": BATCH_COUNT, "updates_per_arm": UPDATES_PER_ARM},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    return control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "fixed_600_train_roles": True, "target_positive_controls": preflight["target_image_positive_control_thresholds_met"], "a_a_wrong_label_and_resume_controls": True, "encoder_ols_geometry_evaluators_frozen": True},
        resource={"passed": True, "status": "PASS", "cuda_runtime": True, "whole_device_safety_line": 0.90, "resource_observations": {"before": preflight["cuda_before"], "after": preflight["cuda_after"]}},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_full_2048_generator_only", "three_step_updates_per_arm_and_negative_control": True, "checkpoint_reload_and_readback": True},
        selected_engine="cuda_full_2048_generator_only",
        resource_envelope={"cuda": preflight["cuda_after"], "disk": preflight["disk_observation"]},
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint required together with --resume")
    return parser.parse_args()


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    diagnostic_dir = _resolve(root, arguments.diagnostic_dir)
    target = _resolve(root, arguments.output_dir)
    _require(bool(arguments.resume) == (arguments.checkpoint is not None), "--resume requires one explicit --checkpoint path")
    _require(not (arguments.preflight_only and arguments.resume), "--preflight-only cannot be combined with --resume")
    output_root = (root / "outputs").resolve()
    _require(target != output_root and output_root in target.parents, "output directory must be a named child of repository outputs/")
    _require(not (target.exists() and not arguments.resume and not arguments.preflight_only), f"output directory exists; use explicit --resume and --checkpoint: {target}")
    if arguments.preflight_only:
        _require(not target.exists(), f"preflight-only output must be a new explicit directory: {target}")
        target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    control: ResearchRunControl | None = None
    try:
        paths, _prior_population, _prior_report = _diag_paths(root, diagnostic_dir)
        inputs = _load_inputs(root, paths)
        population = _population_record(inputs, paths, diagnostic_dir)
        if arguments.resume:
            _require(target.is_dir(), "resume output directory is missing")
            _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == population, "explicit resume population, labels, rows or roles differ")
            preflight_saved = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
            _require(preflight_saved.get("task_id") == TASK_ID and preflight_saved.get("status") == "READY", "resume preflight is not READY for this task")
            receipt = PreflightReceipt.from_mapping(preflight_saved["receipt"])
            preflight = preflight_saved["preflight"]
            selected = _read_checkpoint(_resolve(root, arguments.checkpoint), target, population)
            completed_result = _completed_resume_result(target, selected)
            if completed_result is not None:
                return completed_result
        else:
            if arguments.preflight_only:
                preflight_workspace = target
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                preflight_workspace = None
            if arguments.preflight_only:
                preflight = _preflight(inputs, paths, preflight_workspace)
            else:
                with tempfile.TemporaryDirectory(prefix="lightgen_semantic_preflight_", dir=target.parent) as temporary:
                    preflight = _preflight(inputs, paths, Path(temporary))
            probe_consumer, probe_optimizer, _probe_parameters, _probe_references = _new_endpoint(inputs, torch.device("cuda:0"))
            identity = _make_identity(inputs, population, probe_consumer, probe_optimizer)
            del probe_consumer, probe_optimizer
            control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
            receipt = _ready_receipt(control, preflight)
            control.mark_preflight_ready(receipt)
            if arguments.preflight_only:
                diagnostic._write_json_new(target / "input_population.json", population, "fixed semantic-target comparison population")
                diagnostic._write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "standalone preflight result")
                return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "training_updates": 0, "next_action": "NONE"}
            control.start_execution()
            control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt)
            diagnostic._write_json_new(target / "input_population.json", population, "fixed 600-row TRAIN semantic-target population")
            diagnostic._write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "READY receipt and full CUDA preflight")
            initial_rng = _rng_state()
            diagnostic._write_torch_new(target / "initial_rng_state.pt", initial_rng, "common A/B RNG start")
        if arguments.resume:
            # Bind the saved READY receipt to this explicit attempt after validating the loaded step and role semantics.
            probe_consumer, probe_optimizer, _probe_parameters, _probe_references = _new_endpoint(inputs, torch.device("cuda:0"))
            identity = _make_identity(inputs, population, probe_consumer, probe_optimizer)
            del probe_consumer, probe_optimizer
            control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
            control.mark_preflight_ready(receipt)
            control.start_execution()
            control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
            initial_rng = torch.load(target / "initial_rng_state.pt", map_location="cpu", weights_only=True)
        device = torch.device("cuda:0")
        evaluators = diagnostic.load_evaluators(paths, device)
        evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
        history: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
        if arguments.resume:
            history = {arm: [dict(row) for row in selected["training_history"].get(arm, [])] for arm in ARM_NAMES}
        start_readout_path = target / "readouts_step2200.pt"
        if start_readout_path.is_file():
            start_readouts = torch.load(start_readout_path, map_location="cpu", weights_only=True)
            _validate_readout_population(start_readouts, inputs, "step2200")
            if not (target / "image_statistics_step2200.json").is_file() or not (target / "row_metrics_step2200.pt").is_file():
                _save_measurements(target, "step2200", {"readouts": start_readouts}, inputs)
        else:
            start_consumer, start_optimizer, _params, _refs = _new_endpoint(inputs, device)
            start_readouts = _forward_all(start_consumer, evaluators, inputs["images"], inputs["source_indices"], inputs["labels"])
            _save_measurements(target, "step2200", {"readouts": start_readouts}, inputs)
            del start_consumer, start_optimizer
        readouts_by_endpoint: dict[str, Mapping[str, torch.Tensor]] = {"step2200": start_readouts}
        arms: dict[str, dict[str, Any]] = {}
        resume_arm = selected["arm"] if arguments.resume else None
        if not arguments.resume or resume_arm == ARM_NAMES[0]:
            result_a = _run_arm(ARM_NAMES[0], inputs, evaluators, target, population, history, start_from=selected if arguments.resume else None, initial_rng=initial_rng, started=started)
            arms[ARM_NAMES[0]] = result_a
            readouts_by_endpoint[ARM_NAMES[0]] = result_a["readouts"]
        else:
            a_path = target / f"readouts_{ARM_NAMES[0]}.pt"
            _require(a_path.is_file(), "resuming B requires completed A readouts")
            a_readouts = torch.load(a_path, map_location="cpu", weights_only=True)
            _validate_readout_population(a_readouts, inputs, ARM_NAMES[0])
            readouts_by_endpoint[ARM_NAMES[0]] = a_readouts
            arms[ARM_NAMES[0]] = {"arm": ARM_NAMES[0], "completed_updates": UPDATES_PER_ARM, "readouts": a_readouts, "image_statistics": json.loads((target / f"image_statistics_{ARM_NAMES[0]}.json").read_text(encoding="utf-8"))}
        result_b = _run_arm(ARM_NAMES[1], inputs, evaluators, target, population, history, start_from=selected if arguments.resume and resume_arm == ARM_NAMES[1] else None, initial_rng=initial_rng, started=started)
        arms[ARM_NAMES[1]] = result_b
        readouts_by_endpoint[ARM_NAMES[1]] = result_b["readouts"]
        _require(set(readouts_by_endpoint) == {"step2200", *ARM_NAMES}, "one or more final endpoint readouts are missing")
        endpoint_metrics, target_control = _readout_report(inputs, readouts_by_endpoint, evaluators, device)
        if not _positive_control_ok(target_control["metrics"]):
            raise SemanticTargetRunError("target-image evaluator positive control changed below threshold after training; final measurement is invalid harness")
        image_statistics = {name: json.loads((target / f"image_statistics_{name}.json").read_text(encoding="utf-8")) for name in readouts_by_endpoint}
        _save_training_history(target, history)
        for name, model in evaluators.items():
            _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"final {name} evaluator")
            _require(all(parameter.grad is None for parameter in model.parameters()), f"final {name} retained gradients")
        _sample_resource_telemetry(target, device, "measurement_complete", progress={"A_updates": UPDATES_PER_ARM, "B_updates": UPDATES_PER_ARM}, started=started)
        target_payload = {"logits": target_control["logits"], "metrics": target_control["metrics"]}
        target_control_path = target / "target_evaluator_positive_controls.pt"
        if target_control_path.exists():
            _require(_tree_equal(torch.load(target_control_path, map_location="cpu", weights_only=True), target_payload), "saved target evaluator positive controls differ")
        else:
            diagnostic._write_torch_new(target_control_path, target_payload, "fresh target-image evaluator positive controls")
        report = _finalize(target, inputs, history, readouts_by_endpoint, population, preflight, target_control, endpoint_metrics, image_statistics, started, receipt, identity)
        if report["status"] == "VALID_COMPLETED":
            control.complete()
        else:
            control.valid_failed()
        report["research_run_state"] = control.state
        diagnostic._write_json_new(target / "report.json", report, "terminal scientific decision with run-control state")
        del evaluators
        gc.collect()
        torch.cuda.empty_cache()
        return report
    except TrainingInterrupted as error:
        if control is not None:
            control.interrupted()
        payload = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "status": "INTERRUPTED", "classification": None, "training_recommendation": None, "failure": str(error), "next_action": "resume only from an explicit saved checkpoint"}
        if target.is_dir() and not (target / "interrupted.json").exists():
            diagnostic._write_json_new(target / "interrupted.json", payload, "user-interrupted resumable research attempt")
        return payload
    except Exception as error:
        if control is not None and control.state in {"RUN_PREPARING", "RUNNING", "SCIENTIFIC_ACTIVE"}:
            control.invalid_harness()
        _write_invalid(target, "INVALID_HARNESS", error, started)
        raise


def _write_invalid(target: Path, phase: str, error: Exception, started: float) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "invalid_harness.json"
    payload = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "training_recommendation": None, "phase": phase, "failure_type": type(error).__name__, "failure": str(error), "elapsed_seconds_observation_only": time.perf_counter() - started, "next_action": "NONE"}
    if not path.exists():
        diagnostic._write_json_new(path, payload, "invalid scientific harness record")


def main() -> int:
    result = execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
