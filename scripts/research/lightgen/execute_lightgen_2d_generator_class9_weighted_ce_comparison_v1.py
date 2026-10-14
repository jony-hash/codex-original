"""Continue the fixed large three-evaluator Generator to step 3000."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_capacity_accumulation_comparison_v1 as capacity_run
from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as ablation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_ten_class_presentation_accumulation_v1 as presentation


TASK_ID = "M0_GENERATOR_CLASS9_WEIGHTED_CE_COMPARISON_V1"
PARENT_TASK_ID = presentation.TASK_ID
REPORT_SCHEMA = "m0_generator_class9_weighted_ce_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_class9_weighted_ce_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_class9_weighted_ce_checkpoint.v1"
DEFAULT_PRESENTATION = Path("outputs/20261008_m0_generator_ten_class_presentation_accumulation_v1/runs/attempt_06")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_class9_weighted_ce_comparison_v1/runs/attempt_01")
SOURCE_STEP = 3000
FINAL_STEP = 3700
ADDED_UPDATES = FINAL_STEP - SOURCE_STEP
FIT_ROWS = 200
HISTORIC_ROWS = 100
OLD_CONFIRMATION_ROWS = 300
PRIOR_NEW_CONFIRMATION_ROWS = 300
NEW_CONFIRMATION_ROWS = 300
SOURCE_ROWS = FIT_ROWS + HISTORIC_ROWS + OLD_CONFIRMATION_ROWS + PRIOR_NEW_CONFIRMATION_ROWS
OLD_ROWS = SOURCE_ROWS
TOTAL_ROWS = OLD_ROWS + NEW_CONFIRMATION_ROWS
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
CHECKPOINT_INTERVAL = 100
CONFIRMATION_SEED = 2026100810
RTOL, ATOL = ablation.RTOL, ablation.ATOL
ARM_NAMES = ("A_original", "B_class9_weight2")
SOURCE_ARM = ablation.ARM_NAMES[0]
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
ROLE_NAMES = ("fit", "historic", "old_confirmation", "new_confirmation")
NUMERIC_MODE = dict(ablation.NUMERIC_MODE)
CHECKPOINT_RE = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_NAME = "resource_telemetry_class9_weighted.jsonl"


class Class9WeightedRunError(RuntimeError):
    """Invalid source, measurement, checkpoint, or training invariant."""


class TrainingInterrupted(RuntimeError):
    """A signal was observed after a safe checkpoint boundary."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise Class9WeightedRunError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _cpu_tree(value: Any) -> Any:
    return capacity._cpu_tree(value)


def _write_torch(path: Path, value: Mapping[str, Any], kind: str) -> None:
    ablation._write_torch_new(path, value, kind)


def _write_json(path: Path, value: Mapping[str, Any], kind: str) -> None:
    ablation._write_json_new(path, value, kind)


def _new_confirmation_indices(source_labels: torch.Tensor, prior_indices: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    _require(tuple(source_labels.shape) == (60_000,) and source_labels.dtype == torch.long, "canonical TRAIN label metadata is invalid")
    _require(prior_indices.numel() == OLD_ROWS and torch.unique(prior_indices).numel() == OLD_ROWS, "the previous 600-row population is invalid")
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    prior_mask = torch.zeros(60_000, dtype=torch.bool)
    prior_mask[prior_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, prior_indices.cpu()).all()), "prior rows include a non-TRAIN source row")
    generator = torch.Generator(device="cpu").manual_seed(CONFIRMATION_SEED)
    selected_by_class: dict[int, torch.Tensor] = {}
    for class_id in range(10):
        candidates = torch.nonzero(train_mask & ~prior_mask & (source_labels == class_id), as_tuple=False).flatten().sort().values
        _require(candidates.numel() >= 30, f"class {class_id} has too few unused TRAIN candidates")
        order = torch.randperm(candidates.numel(), generator=generator)
        selected_by_class[class_id] = candidates.index_select(0, order[:30])
    selected = torch.stack(
        [torch.stack([selected_by_class[class_id][row] for class_id in range(10)]) for row in range(30)]
    ).reshape(-1)
    _require(selected.numel() == NEW_CONFIRMATION_ROWS and torch.unique(selected).numel() == NEW_CONFIRMATION_ROWS, "new confirmation rows are not unique")
    _require(not set(selected.tolist()).intersection(prior_indices.tolist()), "new confirmation overlaps the prior 600 rows")
    _require(bool(train_mask.index_select(0, selected).all()), "new confirmation includes a non-TRAIN row")
    _require(torch.bincount(source_labels.index_select(0, selected), minlength=10).tolist() == [30] * 10, "new confirmation is not class-balanced")
    return selected


def _population(root: Path, parent_dir: Path):
    population_path, report_path = parent_dir / "input_population.json", parent_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "the explicit 3000-step presentation source is incomplete")
    parent_population = json.loads(population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(parent_population.get("task_id") == PARENT_TASK_ID and parent_report.get("task_id") == PARENT_TASK_ID, "source presentation task identity differs")
    _require(parent_report.get("status") == "VALID_FAILED" and parent_report.get("classification") == "MIXED_OR_INSUFFICIENT", "source presentation attempt is not the specified completed result")
    source_rows = torch.tensor(parent_population.get("source_row_indices", []), dtype=torch.long)
    source_labels = torch.tensor(parent_population.get("labels", []), dtype=torch.long)
    source_roles = parent_population.get("roles")
    _require(source_rows.numel() == SOURCE_ROWS and source_labels.numel() == SOURCE_ROWS and len(source_roles or []) == SOURCE_ROWS, "source 900-row population is incomplete")
    expected_roles = ["fit"] * FIT_ROWS + ["historic"] * HISTORIC_ROWS + ["old_confirmation"] * OLD_CONFIRMATION_ROWS + ["new_confirmation"] * PRIOR_NEW_CONFIRMATION_ROWS
    _require(source_roles == expected_roles and torch.unique(source_rows).numel() == SOURCE_ROWS, "source 900-row role order or uniqueness differs")

    source_checkpoint_path = parent_dir / "generator_step3000.pt"
    _require(source_checkpoint_path.is_file(), "explicit generator_step3000 checkpoint is missing")
    source = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    _require(source.get("task_id") == PARENT_TASK_ID and source.get("arm") == SOURCE_ARM, "source checkpoint task or arm differs")
    _require(source.get("schema") == presentation.CHECKPOINT_SCHEMA and source.get("completed_updates") == SOURCE_STEP, "source checkpoint is not the declared step3000 checkpoint")
    _require(source.get("optimizer_step") == SOURCE_STEP and source.get("optimizer_parameter_names") == ["generator.phase"], "source Generator Adam step or scope differs")
    _require(source.get("next_batch_index") == 0 and source.get("next_batch_cursor") == 0, "source checkpoint next batch is not zero")
    _require(source.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "source large Generator layout differs")
    parameters = source.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping) and set(parameters) == {"encoder.phase", "generator.phase"}, "source checkpoint parameter state is incomplete")
    _require(tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(parameters["generator.phase"].shape) == (340_000,), "source phase parameter shapes differ")
    _require(all(bool(torch.isfinite(value).all()) for value in parameters.values()), "source phase parameters are nonfinite")
    _require(source.get("numeric_mode") == NUMERIC_MODE, "source strict FP32 mode differs")
    _require(len(source.get("training_history", [])) == SOURCE_STEP, "source checkpoint does not contain 3000 update records")
    for update, row in enumerate(source["training_history"], start=1):
        batch_index = (update - 1) % BATCH_COUNT
        positions = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        _require(row.get("update") == update and row.get("batch_index") == batch_index and row.get("fit_positions") == positions, f"source training batch mapping differs at update {update}")
        _require(row.get("source_row_indices") == [source_rows[position].item() for position in positions] and row.get("labels") == [source_labels[position].item() for position in positions], f"source training rows or labels differ at update {update}")
    adam = source["optimizer_state"]
    groups, slots = adam.get("param_groups"), adam.get("state")
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(slots) == {0}, "source Generator Adam mapping differs")
    _require(groups[0].get("lr") == capacity.GENERATOR_LEARNING_RATE and tuple(groups[0].get("betas")) == capacity.GENERATOR_BETAS and groups[0].get("eps") == capacity.GENERATOR_EPS and groups[0].get("weight_decay") == 0.0, "source Adam hyperparameters differ")
    slot = slots[0]
    _require(int(float(slot["step"].item())) == SOURCE_STEP and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "source Adam moments or step differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "source Adam moments are nonfinite")

    ablation_dir = _resolve(root, parent_population["source_ablation_dir"])
    ablation_population_path = ablation_dir / "input_population.json"
    _require(ablation_population_path.is_file(), "explicit source ablation population is missing")
    ablation_population = json.loads(ablation_population_path.read_text(encoding="utf-8"))
    capacity_dir_value = ablation_population.get("source_capacity_dir")
    _require(isinstance(capacity_dir_value, str) and capacity_dir_value, "source ablation does not identify its capacity lineage")
    capacity_dir = _resolve(root, capacity_dir_value)
    fit_inputs, _, paths, base_state, _, _, source_checkpoints, _, _ = ablation._load_parent_inputs(root, capacity_dir)
    _require(torch.equal(fit_inputs["source_indices"], source_rows[:600]) and torch.equal(fit_inputs["labels"], source_labels[:600]), "first 600 source rows differ from the validated capacity lineage")
    _require(_tree_equal(source["references"], source_checkpoints[SOURCE_ARM]["references"]), "source BOGT references differ from existing capacity lineage")
    _require(_tree_equal(source["weights"], source_checkpoints[SOURCE_ARM]["weights"]), "source BOGT weights differ from existing capacity lineage")
    source_report = parent_report
    parent_readout_path = parent_dir / "readouts_A_step3000.pt"
    _require(parent_readout_path.is_file(), "source A3000 readout is missing")
    parent_readout = torch.load(parent_readout_path, map_location="cpu", weights_only=True)
    _require(torch.equal(parent_readout["source_indices"], source_rows) and torch.equal(parent_readout["labels"], source_labels), "source A3000 readout row order or labels differ")
    expected_counts = source_report["endpoint_metrics"]["A_step3000"]["evaluators"]["E_D"]["by_role"]["new_confirmation"]["per_class_correct"]
    prediction = parent_readout["e_d_logits"].to(torch.float64).argmax(1)
    role_start, role_stop = parent_population["role_positions"]["new_confirmation"]
    actual_counts = [int(((source_labels[role_start:role_stop] == c) & (prediction[role_start:role_stop] == c)).sum()) for c in range(10)]
    _require(expected_counts == actual_counts, "source A3000 E_D report and readout disagree")

    image_store = diagnostic._load_torch(paths["training_data"], mmap=True)
    source_images, canonical_labels = diagnostic._read_image_labels(image_store)
    train_indices, _ = diagnostic.stratified_mnist_indices(canonical_labels, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, source_rows).all()), "one or more prior rows are outside canonical TRAIN")
    _require(torch.equal(canonical_labels.index_select(0, source_rows), source_labels), "source labels differ from canonical TRAIN metadata")
    new_indices = _new_confirmation_indices(canonical_labels, source_rows, train_indices)
    new_labels = canonical_labels.index_select(0, new_indices).contiguous()
    all_indices = torch.cat((source_rows, new_indices))
    all_labels = torch.cat((source_labels, new_labels))
    measure_images = source_images.index_select(0, all_indices).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(tuple(measure_images.shape) == (TOTAL_ROWS, 1, 28, 28) and bool(torch.isfinite(measure_images).all()) and float(measure_images.min()) >= 0 and float(measure_images.max()) <= 1, "fixed 1200-row measurement images are invalid")
    roles = expected_roles + ["new_confirmation"] * NEW_CONFIRMATION_ROWS
    role_positions = {
        "fit": [0, FIT_ROWS],
        "historic": [FIT_ROWS, FIT_ROWS + HISTORIC_ROWS],
        "old_confirmation": [FIT_ROWS + HISTORIC_ROWS, FIT_ROWS + HISTORIC_ROWS + OLD_CONFIRMATION_ROWS],
        "prior_new_confirmation": [FIT_ROWS + HISTORIC_ROWS + OLD_CONFIRMATION_ROWS, SOURCE_ROWS],
        "new_confirmation": [SOURCE_ROWS, TOTAL_ROWS],
    }
    measure_inputs = {"images": measure_images, "source_indices": all_indices, "labels": all_labels}
    population = {
        "task_id": TASK_ID,
        "source_presentation_dir": parent_dir.as_posix(),
        "source_presentation_status": source_report["status"],
        "source_presentation_classification": source_report["classification"],
        "source_checkpoint": source_checkpoint_path.as_posix(),
        "source_checkpoint_step": SOURCE_STEP,
        "source_generator_adam_step": SOURCE_STEP,
        "source_parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
        "source_input_paths": {name: value.as_posix() for name, value in paths.items()},
        "source_row_indices": all_indices.tolist(),
        "labels": all_labels.tolist(),
        "roles": roles,
        "role_counts": {"fit": FIT_ROWS, "historic": HISTORIC_ROWS, "old_confirmation": OLD_CONFIRMATION_ROWS, "prior_new_confirmation": PRIOR_NEW_CONFIRMATION_ROWS, "new_confirmation": NEW_CONFIRMATION_ROWS},
        "fit_rows": FIT_ROWS,
        "source_rows": SOURCE_ROWS,
        "new_confirmation_rows": NEW_CONFIRMATION_ROWS,
        "confirmation_seed": CONFIRMATION_SEED,
        "new_confirmation_is_canonical_train_not_claimed_unseen": True,
        "updates_from_source": SOURCE_STEP,
        "additional_updates_per_arm": ADDED_UPDATES,
        "final_generator_adam_step": FINAL_STEP,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
        "weighted_class": 9,
        "weighted_class_factor": 2.0,
        "numeric_mode": NUMERIC_MODE,
        "role_positions": role_positions,
    }
    source = {**source, "checkpoint_path": source_checkpoint_path.as_posix(), "source_task_id": PARENT_TASK_ID}
    source_checkpoints = {arm: source for arm in ARM_NAMES}
    parent_readouts = {"A_step3000": parent_readout}
    return fit_inputs, measure_inputs, population, paths, base_state, source, source_checkpoints, parent_readouts, source_report


def _new_indices_for_test(source_labels: torch.Tensor, prior_indices: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    return _new_confirmation_indices(source_labels, prior_indices, train_indices)


def _target_positive_control(evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    start, stop = OLD_ROWS, TOTAL_ROWS
    images = inputs["images"][start:stop]
    labels = inputs["labels"][start:stop]
    predictions = {name: [] for name in EVALUATOR_NAMES}
    evaluator_logits = {name: [] for name in EVALUATOR_NAMES}
    for offset in range(0, NEW_CONFIRMATION_ROWS, BATCH_SIZE):
        target = images[offset : offset + BATCH_SIZE].to(device)
        view = capacity.canonical.cmos_view(target)
        with torch.no_grad():
            for name in EVALUATOR_NAMES:
                logits = capacity._evaluator_logits(evaluators[name], view, name).detach().cpu()
                evaluator_logits[name].append(logits)
                predictions[name].append(logits.to(torch.float64).argmax(dim=1))
    result: dict[str, Any] = {}
    target_labels = labels.cpu()
    for name in EVALUATOR_NAMES:
        prediction = torch.cat(predictions[name])
        correct = [int(((prediction == target_labels) & (target_labels == class_id)).sum()) for class_id in range(10)]
        macro = sum(correct) / NEW_CONFIRMATION_ROWS
        result[name] = {"per_class_correct": correct, "macro_recall": macro, "passed": macro >= 0.90 and correct[1] >= 24 and correct[7] >= 24 and correct[9] >= 24}
    return {"summary": result, "logits": {name: torch.cat(values) for name, values in evaluator_logits.items()}, "passed": all(value["passed"] for value in result.values())}


def _source_readout_replay(readouts: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    details = {}
    for source_key, key in (("z", "z"), ("raw_cmos", "raw_cmos"), ("evaluator_view", "evaluator_view"), ("target_slm_intensity", "target_slm_intensity"), ("power_by_plane", "power_by_plane"), ("new_phase_plane_power", "new_phase_plane_power")):
        if source_key not in parent or source_key not in readouts:
            continue
        left = readouts[source_key][:OLD_ROWS]
        right = parent[source_key]
        _require(left.shape == right.shape and torch.allclose(left, right, rtol=RTOL, atol=ATOL), f"A1700 source replay differs for {source_key}")
        details[key] = {"passed": True, "max_abs_difference": float((left.to(torch.complex128 if left.is_complex() else torch.float64) - right.to(torch.complex128 if right.is_complex() else torch.float64)).abs().max())}
    for name in EVALUATOR_NAMES:
        key = f"{name.lower()}_logits"
        _require(torch.allclose(readouts[key][:OLD_ROWS], parent[key], rtol=RTOL, atol=ATOL), f"A1700 source replay differs for {key}")
        details[key] = {"passed": True, "max_abs_difference": float((readouts[key][:OLD_ROWS].to(torch.float64) - parent[key].to(torch.float64)).abs().max())}
    return {"passed": True, "rtol": RTOL, "atol": ATOL, "fields": details}


def _measure_forward(consumer: Any, evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows, device = inputs["images"].shape[0], consumer.device
    payload: dict[str, Any] = {
        "z": torch.empty((rows, 100), dtype=torch.complex64),
        "raw_cmos": torch.empty((rows, 1, 200, 200), dtype=torch.float32),
        "evaluator_view": torch.empty((rows, 1, 28, 28), dtype=torch.float32),
        "target_slm_intensity": torch.empty((rows, 1, 200, 200), dtype=torch.float32),
        "power_by_plane": torch.empty((rows, 5), dtype=torch.float64),
        "new_phase_plane_power": torch.empty((rows, 2), dtype=torch.float64),
        "source_indices": inputs["source_indices"].clone(),
        "labels": inputs["labels"].clone(),
    }
    payload.update({f"{name.lower()}_logits": torch.empty((rows, 10), dtype=torch.float32) for name in EVALUATOR_NAMES})
    metadata = None
    for start in range(0, rows, BATCH_SIZE):
        stop = min(rows, start + BATCH_SIZE)
        images = inputs["images"][start:stop].to(device)
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(images)
            view = capacity.canonical.cmos_view(result.raw_intensity)
            payload["z"][start:stop].copy_(result.z_ols.coefficients.detach().cpu())
            payload["raw_cmos"][start:stop].copy_(result.raw_intensity.detach().cpu())
            payload["evaluator_view"][start:stop].copy_(view.detach().cpu())
            _require(result.image_input is not None, "production image path omitted its input image")
            payload["target_slm_intensity"][start:stop].copy_(result.image_input.slm_intensity.detach().cpu())
            payload["power_by_plane"][start:stop].copy_(capacity.bogt.contracted_plane_powers(result).detach().to(torch.float64).cpu())
            generator = consumer.runtime.generator
            _require(generator.last_extra_plane_powers is not None, "production large Generator omitted added-plane power")
            payload["new_phase_plane_power"][start:stop].copy_(generator.last_extra_plane_powers.detach().to(torch.float64).cpu())
            for name in EVALUATOR_NAMES:
                payload[f"{name.lower()}_logits"][start:stop].copy_(capacity._evaluator_logits(evaluators[name], view, name).detach().cpu())
            if metadata is None:
                metadata = capacity_run._fields_metadata(result, "B_large", consumer)
        del result, view, images
    payload["plane_metadata"] = metadata
    for name, value in payload.items():
        if torch.is_tensor(value):
            _require(bool(torch.isfinite(value.real if value.is_complex() else value).all()), f"measurement {name} is nonfinite")
    return payload


def _classification_statistics(labels: torch.Tensor, logits: torch.Tensor, role_positions: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    labels = labels.to(torch.long).cpu()
    logits = logits.to(torch.float64).cpu()
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((10, 10), dtype=torch.long)
    for target, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[target, prediction] += 1
    roles: dict[str, Any] = {}
    for role, bounds in role_positions.items():
        start, stop = int(bounds[0]), int(bounds[1])
        y, pred = labels[start:stop], predictions[start:stop]
        class_rows = torch.bincount(y, minlength=10)
        class_correct = torch.tensor([int(((y == c) & (pred == c)).sum()) for c in range(10)])
        macro = float((class_correct.to(torch.float64) / class_rows.clamp_min(1)).mean())
        roles[role] = {
            "rows": stop - start,
            "per_class_count": class_rows.tolist(),
            "per_class_correct": class_correct.tolist(),
            "per_class_recall": (class_correct.to(torch.float64) / class_rows.clamp_min(1)).tolist(),
            "per_class_wilson_95": {str(c): diagnostic._wilson(int(class_correct[c]), int(class_rows[c])) for c in range(10)},
            "macro_recall": macro,
            "correct_total": int(class_correct.sum()),
            "prediction_counts": torch.bincount(pred, minlength=10).tolist(),
            "confusion_matrix_actual_rows_predicted_columns": torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10).tolist(),
        }
    return {
        "by_role": roles,
        "all_rows_confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
        "all_rows_prediction_counts": torch.bincount(predictions, minlength=10).tolist(),
    }


def _row_and_image_metrics(readouts: Mapping[str, Any], population: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    labels = readouts["labels"].cpu()
    target = readouts["target_slm_intensity"].to(torch.float64)
    raw = readouts["raw_cmos"].to(torch.float64)
    target_view = F.adaptive_avg_pool2d(target.to(torch.float32), (28, 28)).to(torch.float64)
    generated_view = readouts["evaluator_view"].to(torch.float64)
    foreground = target > 0.1
    background = ~foreground
    raw_error = (raw - target).square()
    view_error = (generated_view - target_view).square()
    raw_foreground = (raw_error * foreground).flatten(1).sum(1) / foreground.flatten(1).sum(1).clamp_min(1)
    raw_background = (raw_error * background).flatten(1).sum(1) / background.flatten(1).sum(1).clamp_min(1)
    view_foreground_mask = F.adaptive_max_pool2d(foreground.to(torch.float32), (28, 28)).to(torch.bool)
    view_background_mask = ~view_foreground_mask
    view_foreground = (view_error * view_foreground_mask).flatten(1).sum(1) / view_foreground_mask.flatten(1).sum(1).clamp_min(1)
    view_background = (view_error * view_background_mask).flatten(1).sum(1) / view_background_mask.flatten(1).sum(1).clamp_min(1)
    scores: dict[str, torch.Tensor] = {}
    summaries: dict[str, Any] = {}
    for name in EVALUATOR_NAMES:
        logits = readouts[f"{name.lower()}_logits"].to(torch.float64).cpu()
        y = labels.view(-1, 1)
        true = logits.gather(1, y).squeeze(1)
        wrong_logits = logits.scatter(1, y, -torch.inf)
        margin = true - wrong_logits.max(1).values
        ce = torch.logsumexp(logits, dim=1) - true
        scores[f"{name.lower()}_ce_float64"] = ce
        scores[f"{name.lower()}_margin_float64"] = margin
        scores[f"{name.lower()}_predictions"] = logits.argmax(1)
        summaries[name] = {
            **_classification_statistics(labels, logits, population["role_positions"]),
            "mean_ce_float64": float(ce.mean()),
            "mean_margin_float64": float(margin.mean()),
        }
    image_rows = {
        "raw_mse": raw_error.flatten(1).mean(1),
        "raw_foreground_mse": raw_foreground,
        "raw_background_mse": raw_background,
        "view_mse": view_error.flatten(1).mean(1),
        "view_foreground_mse": view_foreground,
        "view_background_mse": view_background,
        "raw_fraction_above_one": (raw > 1.0).flatten(1).to(torch.float64).mean(1),
        "raw_fraction_below_zero": (raw < 0.0).flatten(1).to(torch.float64).mean(1),
        "raw_max": raw.flatten(1).max(1).values,
        "raw_zero_image": (raw.flatten(1).abs().sum(1) == 0),
        "raw_constant_image": (raw.flatten(1).std(1, unbiased=False) == 0),
        **scores,
    }
    row_summary: dict[str, Any] = {}
    for role, bounds in population["role_positions"].items():
        start, stop = map(int, bounds)
        by_class: dict[str, Any] = {}
        for class_id in range(10):
            selection = torch.arange(start, stop)[labels[start:stop] == class_id]
            values: dict[str, Any] = {}
            for key, vector in image_rows.items():
                if not vector.is_floating_point():
                    continue
                values[key] = float(vector[selection].mean()) if selection.numel() else 0.0
            values["rows"] = int(selection.numel())
            values["zero_images"] = int(image_rows["raw_zero_image"][selection].sum())
            values["constant_images"] = int(image_rows["raw_constant_image"][selection].sum())
            by_class[str(class_id)] = values
        role_values: dict[str, Any] = {"rows": stop - start, "by_class": by_class}
        for key, vector in image_rows.items():
            if vector.is_floating_point():
                role_values[key] = float(vector[start:stop].mean())
            else:
                role_values[key] = int(vector[start:stop].sum())
        row_summary[role] = role_values
    score_summary = {name: value for name, value in summaries.items()}
    return {"evaluators": score_summary, "image_metrics_by_role": row_summary, "all_finite": True}, image_rows


def _final_decision(start: Mapping[str, Any], arm_a: Mapping[str, Any], arm_b: Mapping[str, Any], *, valid: bool) -> dict[str, Any]:
    role = "new_confirmation"
    base = start["evaluators"]["E_D"]["by_role"][role]
    a = arm_a["evaluators"]["E_D"]["by_role"][role]
    b = arm_b["evaluators"]["E_D"]["by_role"][role]
    base_correct = list(base["per_class_correct"])
    a_correct = list(a["per_class_correct"])
    b_correct = list(b["per_class_correct"])
    base_total, a_total, b_total = int(base["correct_total"]), int(a["correct_total"]), int(b["correct_total"])
    base_macro, a_macro, b_macro = float(base["macro_recall"]), float(a["macro_recall"]), float(b["macro_recall"])
    base_image = start["image_metrics_by_role"][role]
    a_image = arm_a["image_metrics_by_role"][role]
    b_image = arm_b["image_metrics_by_role"][role]
    weighted_class_gain = b_correct[9] >= a_correct[9] + 3 and b_correct[9] >= base_correct[9] + 3
    image_ok = (
        b_image["raw_mse"] <= 1.05 * a_image["raw_mse"]
        and b_image["raw_mse"] <= 1.05 * base_image["raw_mse"]
        and b_image["raw_fraction_above_one"] <= 0.01
        and all(b_image["by_class"][str(c)]["raw_fraction_above_one"] <= 0.01 for c in (1, 7, 9))
        and b_image["raw_zero_image"] == 0
        and b_image["raw_constant_image"] == 0
        and arm_b.get("all_finite") is True
    )
    conditions = {
        "class9_at_least_15_and_plus_3_vs_both": b_correct[9] >= 15 and weighted_class_gain,
        "total_at_least_225_and_not_below_either_baseline": b_total >= 225 and b_total >= a_total and b_total >= base_total,
        "each_class_at_least_15": min(b_correct) >= 15,
        "classes_1_and_7_not_below_source": b_correct[1] >= base_correct[1] and b_correct[7] >= base_correct[7],
        "no_class_loses_more_than_3_vs_either_baseline": all(b_correct[c] >= a_correct[c] - 3 and b_correct[c] >= base_correct[c] - 3 for c in range(10)),
        "raw_mse_clipping_and_image_integrity": image_ok,
        "budget_roles_freeze_and_measurement_valid": valid,
    }
    if not valid:
        classification, status = "INVALID_HARNESS", "INVALID_HARNESS"
    elif all(conditions.values()):
        classification, status = "CLASS9_WEIGHTED_CE_GAIN_SUPPORTED", "VALID_COMPLETED"
    elif b_correct[9] > a_correct[9] and b_correct[9] > base_correct[9]:
        classification, status = "MIXED_OR_INSUFFICIENT", "VALID_FAILED"
    else:
        classification, status = "NOT_SUPPORTED_WITHIN_700_UPDATES", "VALID_FAILED"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "new_confirmation": {
            "source_step": SOURCE_STEP,
            "final_step": FINAL_STEP,
            "macro_recall": {"source_step3000": base_macro, "A_step3700": a_macro, "B_step3700": b_macro},
            "correct_total": {"source_step3000": base_total, "A_step3700": a_total, "B_step3700": b_total},
            "per_class_correct": {"source_step3000": base_correct, "A_step3700": a_correct, "B_step3700": b_correct},
            "per_class_delta_B_vs_A": [b_correct[c] - a_correct[c] for c in range(10)],
            "per_class_delta_B_vs_source": [b_correct[c] - base_correct[c] for c in range(10)],
            "class9_gain_over_both_baselines": weighted_class_gain,
            "raw_mse": {"source_step3000": base_image["raw_mse"], "A_step3700": a_image["raw_mse"], "B_step3700": b_image["raw_mse"]},
        },
        "training_recommendation": "CONSIDER_CLASS9_WEIGHTED_CE_IN_A_SEPARATE_CONTROLLED_STUDY" if classification == "CLASS9_WEIGHTED_CE_GAIN_SUPPORTED" else "NONE",
        "next_action": "NONE",
        "stop": True,
    }


def _wilson_display(view: torch.Tensor, scale: int = 4) -> Image.Image:
    image = view.detach().to(torch.float32).cpu().squeeze().clamp(0.0, 1.0)
    array = torch.round((1.0 - image) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _comparison_pages(target: Path, inputs: Mapping[str, Any], start: Mapping[str, Any], arm_a: Mapping[str, Any], arm_b: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    pages: dict[str, list[str]] = {}
    cell, row_height = 112, 150
    x_positions = (350, 465, 580, 695)
    titles = ("Target", "step3000", "A3700", "B3700")
    for role, bounds in population["role_positions"].items():
        role_start, role_stop = map(int, bounds)
        positions = list(range(role_start, role_stop))
        pages[role] = []
        for page_start in range(0, len(positions), 30):
            page_positions = positions[page_start : page_start + 30]
            canvas = Image.new("RGB", (820, 58 + len(page_positions) * row_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{role}; rows {page_start + 1}-{page_start + len(page_positions)} of {len(positions)}; fixed [0,1]", fill="black")
            for x, title in zip(x_positions, titles, strict=True):
                draw.text((x, 35), title, fill="black")
            for index, position in enumerate(page_positions):
                y = 55 + index * row_height
                draw.text((8, y + 8), f"row={int(inputs['source_indices'][position])} y={int(inputs['labels'][position])} {role}", fill="black")
                panels = (inputs["images"][position], start["evaluator_view"][position], arm_a["evaluator_view"][position], arm_b["evaluator_view"][position])
                for col, panel in enumerate(panels):
                    canvas.paste(_wilson_display(panel), (x_positions[col], y))
                for col, readout in enumerate((start, arm_a, arm_b), start=1):
                    pred = int(readout["e_d_logits"][position].to(torch.float64).argmax())
                    draw.text((x_positions[col], y + cell + 2), f"E_D={pred}", fill="black")
            filename = f"comparison_{role}_{page_start // 30 + 1:02d}.png"
            path = target / filename
            if path.exists():
                with Image.open(path) as existing:
                    _require(existing.size == canvas.size and existing.convert("RGB").tobytes() == canvas.tobytes(), f"saved comparison page differs: {filename}")
            else:
                canvas.save(path, optimize=True)
            pages[role].append(filename)
    return {"pages_by_role": pages, "row_counts": population["role_counts"], "panel_order": ["target", "step3000", "A_step3700", "B_step3700"], "fixed_display_range": [0.0, 1.0], "per_image_rescaling": False}


def _prepare_target(root: Path, target: Path, *, resume: bool) -> bool:
    out_root = (root / "outputs").resolve()
    _require(target != out_root and out_root in target.parents, "output must be inside outputs/")
    parts = target.relative_to(out_root).parts
    _require(len(parts) == 3 and re.fullmatch(r"\d{8}_[A-Za-z0-9][A-Za-z0-9_-]*", parts[0]) and parts[1] == "runs", "output must be outputs/YYYYMMDD_<task>/runs/<attempt>")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected output namespace is forbidden")
    if resume:
        _require(target.is_dir(), "resume output directory does not exist")
        return False
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
        return True
    ready_stage_files = {"input_population.json", "preflight.json", "target_image_evaluator_positive_controls.pt", "readouts_step3000.pt"}
    _require(target.is_dir() and {p.name for p in target.iterdir()} == ready_stage_files, "output already contains artifacts; use explicit resume or a new attempt directory")
    return False


def _disk_probe(target: Path, source: Path) -> dict[str, Any]:
    parent = target
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    source_bytes = source.stat().st_size
    estimated = max(1_500_000_000, source_bytes * 4 + TOTAL_ROWS * 200 * 200 * 4 * 4 + 2 * 340_000 * 4)
    reserve = 1_610_612_736
    _require(free >= estimated + reserve, f"output volume free bytes {free}; estimated artifacts plus reserve require {estimated + reserve}")
    temp_root = Path(tempfile.gettempdir()).resolve()
    _require(root_not_inside(target.resolve(), temp_root), "temporary directory must be outside repository output")
    with tempfile.TemporaryDirectory(prefix="lg_class9_external_") as temp_name:
        probe = Path(temp_name) / "probe.bin"
        probe.write_bytes(b"lightgen-class9-recovery")
        _require(probe.read_bytes() == b"lightgen-class9-recovery", "external temporary write/read failed")
    with tempfile.TemporaryDirectory(prefix="lg_class9_output_", dir=target) as temp_name:
        probe = Path(temp_name) / "probe.pt"
        torch.save({"probe": torch.tensor([1.0, 2.0])}, probe)
        _require(torch.equal(torch.load(probe, map_location="cpu", weights_only=True)["probe"], torch.tensor([1.0, 2.0])), "output temporary write/read failed")
    return {"executed": True, "passed": True, "free_bytes_observed": free, "estimated_output_bytes": estimated, "reserve_bytes": reserve, "external_temp_write_read": True, "output_write_read": True}


def root_not_inside(target: Path, temp_root: Path) -> bool:
    return target != temp_root and target not in temp_root.parents


def _checkpoint_name(arm: str, step: int) -> str:
    _require(arm in ARM_NAMES, f"unknown class9-weighted arm: {arm}")
    letter = "a" if arm == ARM_NAMES[0] else "b"
    return f"arm_{letter}_step{step:04d}.pt"


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    _require(arm in ARM_NAMES and SOURCE_STEP <= completed <= FINAL_STEP, "checkpoint arm or update count is invalid")
    parameters = capacity._parameters(consumer)
    step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(step == completed and len(history) == completed, "checkpoint Adam step or training history count differs")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "source_task_id": PARENT_TASK_ID,
        "arm": arm,
        "semantic_weight_policy": "class9_x2" if arm == ARM_NAMES[1] else "uniform_x1",
        "source_checkpoint": population["source_checkpoint"],
        "source_step": SOURCE_STEP,
        "additional_updates": completed - SOURCE_STEP,
        "completed_updates": completed,
        "optimizer_step": step,
        "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": source["architecture_layout"],
        "population": dict(population),
        "trainable_parameter_state": capacity._snapshot_parameters(parameters),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": [dict(row) for row in history],
        "source_history_count": SOURCE_STEP,
        "source_rng_state": source["rng_state"],
        "rng_state": capacity._rng_state(),
        "next_batch_index": completed % BATCH_COUNT,
        "next_batch_cursor": completed % BATCH_COUNT,
        "numeric_mode": dict(NUMERIC_MODE),
    }


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any]) -> Path:
    path = target / _checkpoint_name(arm, completed)
    _require(not path.exists(), f"checkpoint exists and will not be overwritten: {path.name}")
    _write_torch(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population, source), "class9 weighted CE Generator Adam recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any], source: Mapping[str, Any], *, measurement_baseline: bool = False) -> dict[str, Any]:
    _require(path.resolve().parent == target.resolve() and path.is_file(), "resume requires an explicit checkpoint inside this task output")
    value = torch.load(path, map_location="cpu", weights_only=True)
    arm, step = value.get("arm"), value.get("completed_updates")
    _require(value.get("schema") == CHECKPOINT_SCHEMA and value.get("task_id") == TASK_ID and value.get("source_task_id") == PARENT_TASK_ID, "resume checkpoint belongs to another task")
    _require(arm in ARM_NAMES and isinstance(step, int) and SOURCE_STEP <= step <= FINAL_STEP and (step == SOURCE_STEP or step % CHECKPOINT_INTERVAL == 0), "resume arm or cumulative step is invalid")
    _require(value.get("semantic_weight_policy") == ("class9_x2" if arm == ARM_NAMES[1] else "uniform_x1"), "resume semantic weighting policy differs")
    _require(path.name == _checkpoint_name(arm, step) and value.get("source_checkpoint") == source["checkpoint_path"], "resume filename or source path differs")
    _require(value.get("source_step") == SOURCE_STEP and value.get("additional_updates") == step - SOURCE_STEP, "resume source/additional update counts differ")
    _require(value.get("population") == dict(population), "resume population or role mapping differs")
    _require(value.get("optimizer_step") == step and value.get("optimizer_parameter_names") == ["generator.phase"], "resume Adam scope or step differs")
    _require(value.get("next_batch_index") == step % BATCH_COUNT and value.get("next_batch_cursor") == step % BATCH_COUNT, "resume batch cursor differs")
    _require(value.get("numeric_mode") == NUMERIC_MODE and value.get("architecture_layout") == source["architecture_layout"], "resume numeric mode or architecture differs")
    history = value.get("training_history")
    _require(isinstance(history, list) and len(history) == step and _tree_equal(history[:SOURCE_STEP], source["training_history"]), "resume history lacks the exact source trajectory")
    for update, record in enumerate(history[SOURCE_STEP:], start=SOURCE_STEP + 1):
        batch = (update - 1) % BATCH_COUNT
        positions = list(range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE))
        _require(record.get("update") == update and record.get("arm") == arm and record.get("batch_index") == batch and record.get("fit_positions") == positions, f"resume history mapping differs at update {update}")
        _require(record.get("source_row_indices") == [population["source_row_indices"][position] for position in positions] and record.get("labels") == [population["labels"][position] for position in positions], f"resume TRAIN row/label mapping differs at update {update}")
        _require(record.get("training_evaluators") == ["E_S", "E_I", "E_C"], f"resume evaluator set differs at update {update}")
        _require(record.get("semantic_weight_policy") == value["semantic_weight_policy"], f"resume class-weight history differs at update {update}")
        expected_weights = [1.0] * 9 + ([2.0] if arm == ARM_NAMES[1] else [1.0])
        _require(record.get("class_weights_by_label") == expected_weights and record.get("weighted_ce_denominator") == (11.0 if arm == ARM_NAMES[1] else 10.0), f"resume weighted-CE history differs at update {update}")
    parameters = value.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping) and set(parameters) == {"encoder.phase", "generator.phase"}, "resume model state is incomplete")
    for name, tensor in parameters.items():
        expected = source["trainable_parameter_state"][name]
        _require(tuple(tensor.shape) == tuple(expected.shape) and tensor.dtype == expected.dtype and bool(torch.isfinite(tensor).all()), f"resume parameter shape or values are invalid for {name}")
    optimizer_state = value.get("optimizer_state", {})
    groups, slots = optimizer_state.get("param_groups"), optimizer_state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and groups[0].get("params") == [0] and groups[0].get("lr") == capacity.GENERATOR_LEARNING_RATE and tuple(groups[0].get("betas", ())) == capacity.GENERATOR_BETAS and groups[0].get("eps") == capacity.GENERATOR_EPS and groups[0].get("weight_decay") == 0.0, "resume Adam hyperparameters differ")
    _require(isinstance(slots, Mapping) and set(slots) == {0}, "resume Adam state scope differs")
    slot = slots[0]
    _require(int(float(slot["step"].item())) == step and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "resume Adam moments or step differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "resume Adam state is nonfinite")
    _require(isinstance(value.get("rng_state"), Mapping) and len(value["rng_state"]) >= 3, "resume RNG state is incomplete")
    _require(_tree_equal(value.get("frozen_runtime_state"), source.get("frozen_runtime_state")), "resume changed frozen runtime state")
    _require(_tree_equal(value.get("references"), source.get("references")) and _tree_equal(value.get("weights"), source.get("weights")), "resume changed BOGT references or weights")
    if not measurement_baseline:
        later = [candidate.name for candidate in target.glob("arm_[ab]_step*.pt") if (match := CHECKPOINT_RE.fullmatch(candidate.name)) and ("A_original" if match.group(1) == "a" else "B_class9_weight2") == arm and int(match.group(2)) > step]
        _require(not later, f"selected checkpoint predates saved progress for {arm}: {later}")
        if arm == ARM_NAMES[0]:
            _require(not any((target / _checkpoint_name(ARM_NAMES[1], later_step)).exists() for later_step in range(SOURCE_STEP + CHECKPOINT_INTERVAL, FINAL_STEP + 1, CHECKPOINT_INTERVAL)), "A cannot resume after B has started")
        else:
            _require((target / _checkpoint_name(ARM_NAMES[0], FINAL_STEP)).is_file(), "B resume requires the completed A checkpoint")
    return value


def _restore_task_checkpoint(value: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(value["trainable_parameter_state"][name].to(parameter.device))
    ablation._load_adam_state(optimizer, parameters["generator.phase"], value["optimizer_state"], expected_step=int(value["completed_updates"]))
    capacity._restore_rng_state(value["rng_state"])


def _resource(target: Path, phase: str, completed: int | None = None) -> dict[str, Any]:
    device = torch.device("cuda:0")
    value = {"task_id": TASK_ID, "phase": phase, "completed_updates": completed, "elapsed_seconds_observation_only": time.perf_counter() - _STARTED, "resource_observation": capacity.diagnostic._resource_snapshot(device)}
    with (target / RESOURCE_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
    return value


_STARTED = 0.0
_EXECUTED_CHECKS: dict[str, Any] = {}


def _record_check(checks: dict[str, Any], name: str, result: Mapping[str, Any]) -> None:
    value = dict(result)
    checks[name] = value
    _EXECUTED_CHECKS[name] = value


def _aa_repeatability(left: Any, right: Any) -> tuple[bool, dict[str, Any]]:
    """Compare independently repeated production trajectories at the declared numeric tolerance."""
    return capacity._repeatability(left, right)


def _weighted_logits_gradient_reference(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(2026100810)
        logits = torch.randn((BATCH_SIZE, 10), dtype=dtype, device=device, requires_grad=True)
        labels = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
        weights = _class_weights(labels, ARM_NAMES[1]).to(device=device, dtype=dtype)
        denominator = weights.sum()
        accumulated = torch.zeros_like(logits)
        loss_total = torch.zeros((), dtype=dtype, device=device)
        for start in range(0, BATCH_SIZE, 2):
            stop = start + 2
            piece = _weighted_ce_piece(logits[start:stop], labels[start:stop], weights[start:stop], denominator)
            piece_grad = torch.autograd.grad(piece, logits, retain_graph=True)[0]
            accumulated += piece_grad
            loss_total += piece.detach()
        probabilities = torch.softmax(logits.detach(), dim=1)
        one_hot = F.one_hot(labels, num_classes=10).to(dtype=dtype)
        expected = (probabilities - one_hot) * weights[:, None] / denominator
        max_error = float((accumulated - expected).abs().max().detach().cpu())
        _require(torch.allclose(accumulated, expected, rtol=2e-5, atol=2e-7), f"weighted CE logit gradient differs from analytic reference for {dtype}")
        _require(torch.allclose(loss_total, F.cross_entropy(logits, labels, weight=weights, reduction="mean"), rtol=2e-5, atol=2e-7), f"weighted CE value differs from full-batch reference for {dtype}")
        details[str(dtype)] = {"max_abs_gradient_error": max_error, "denominator": float(denominator.detach().cpu()), "passed": True}
        del logits, labels, weights, denominator, accumulated, loss_total
    return {"passed": True, "reference": "analytic weighted-softmax-minus-one-hot VJP", "by_dtype": details, "microbatch_size": 2, "full_logical_batch_rows": 10}


def _run_three_step_trial(fit_inputs: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], arm: str):
    device = torch.device("cuda:0")
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
    before_encoder = params["encoder.phase"].detach().cpu().clone()
    before_frozen = capacity._frozen_runtime_state(consumer)
    before_evaluators = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    records = []
    for completed in range(SOURCE_STEP, SOURCE_STEP + 3):
        if arm == ARM_NAMES[0]:
            row = ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, completed, arm=SOURCE_ARM)
            row["arm"] = arm
        else:
            row = _train_step_weighted(consumer, optimizer, params, evaluators, fit_inputs, completed)
        records.append(row)
        _require(row["generator_adam_step"] == completed + 1 and row["old_100_plane_gradient_norm"] > 0.0 and row["new_400_plane_gradient_norm"] > 0.0, f"{arm} preflight did not update both phase supports")
        _require(torch.equal(params["encoder.phase"].detach().cpu(), before_encoder), f"{arm} preflight changed Encoder")
        capacity._assert_state_equal(before_frozen, capacity._frozen_runtime_state(consumer), f"{arm} preflight changed frozen runtime")
        for name, model in evaluators.items():
            capacity._assert_state_equal(before_evaluators[name], capacity._state_snapshot(model), f"{arm} preflight changed evaluator {name}")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{arm} preflight broke evaluator {name} isolation")
    result = {"records": records, "parameters": capacity._snapshot_parameters(params), "optimizer": _cpu_tree(optimizer.state_dict())}
    del consumer, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _weighted_wrong_label_control(fit_inputs: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
    batch, true_labels, _ = capacity.training_batch(fit_inputs["images"], fit_inputs["labels"], 0)
    images = batch.to(device)
    labels = true_labels.to(device)
    shifted = labels.roll(1)
    _, _, true_gradients, _ = _weighted_microbatch_loss_gradients(consumer, params, evaluators, images, labels)
    _, _, wrong_gradients, _ = _weighted_microbatch_loss_gradients(consumer, params, evaluators, images, shifted)
    true = true_gradients["semantic"]["E_I"]["generator.phase"]
    wrong = wrong_gradients["semantic"]["E_I"]["generator.phase"]
    difference = float(torch.linalg.vector_norm(true - wrong).detach().to(torch.float64).cpu())
    _require(difference > 0.0 and not torch.allclose(true, wrong, rtol=2e-5, atol=2e-7), "weighted wrong-label control did not change semantic gradient")
    _require(all(parameter.grad is None for parameter in params.values()), "weighted negative control populated optical .grad")
    del consumer, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "gradient_difference_norm": difference, "true_labels": labels.detach().cpu().tolist(), "wrong_labels": shifted.detach().cpu().tolist()}


def _recovery_control(fit_inputs: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], population: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
    history = list(source["training_history"])
    first = _train_step_weighted(consumer, optimizer, params, evaluators, fit_inputs, SOURCE_STEP)
    history.append(first)
    checkpoint = _checkpoint_payload(consumer, optimizer, ARM_NAMES[1], SOURCE_STEP + 1, history, population, source)
    with tempfile.TemporaryDirectory(prefix="lg_class9_resume_", dir=Path(tempfile.gettempdir())) as temp_name:
        checkpoint_path = Path(temp_name) / "arm_b_step3001.pt"
        torch.save(checkpoint, checkpoint_path)
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    continuous_records = [_train_step_weighted(consumer, optimizer, params, evaluators, fit_inputs, completed) for completed in (SOURCE_STEP + 1, SOURCE_STEP + 2)]
    continuous = {"parameters": capacity._snapshot_parameters(params), "optimizer": _cpu_tree(optimizer.state_dict()), "records": continuous_records}
    del consumer, optimizer, params
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
    _restore_task_checkpoint(saved_checkpoint, consumer, optimizer)
    resumed_records = [_train_step_weighted(consumer, optimizer, params, evaluators, fit_inputs, completed) for completed in (SOURCE_STEP + 1, SOURCE_STEP + 2)]
    resumed = {"parameters": capacity._snapshot_parameters(params), "optimizer": _cpu_tree(optimizer.state_dict()), "records": resumed_records}
    passed, detail = capacity._repeatability(continuous, resumed)
    _require(passed, f"weighted continuous and resumed trajectories differ: {detail}")
    del consumer, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "source_step": SOURCE_STEP, "saved_step": SOURCE_STEP + 1, "final_step": SOURCE_STEP + 3, "checkpoint_roundtrip": True, "continuous_vs_resumed": detail}


def _preflight(fit_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], paths: Mapping[str, Path], base_state: Mapping[str, Any], source: Mapping[str, Any], parent_readouts: Mapping[str, Any], target: Path) -> dict[str, Any]:
    _EXECUTED_CHECKS.clear()
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    _record_check(checks, "cuda_available", {"executed": True, "passed": True, "device": str(device)})
    disk = _disk_probe(target, _resolve(_root(), population["source_checkpoint"]))
    _record_check(checks, "disk_and_external_io", disk | {"executed": True, "passed": True})
    rng_before = capacity._rng_state()
    source_before = _cpu_tree(source)
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    try:
        reference = _weighted_logits_gradient_reference(device)
        _record_check(checks, "weighted_ce_value_and_logit_gradient_reference", {"executed": True, **reference})
        target_control = _target_positive_control(evaluators, measure_inputs)
        _record_check(checks, "new_confirmation_target_positive_control", {"executed": True, "passed": target_control["passed"], "evaluators": target_control["summary"], "role": "new_confirmation"})
        _require(target_control["passed"], "four-evaluator target positive control missed the declared macro/class thresholds")
        target_payload = {"source_row_indices": measure_inputs["source_indices"][SOURCE_ROWS:].clone(), "labels": measure_inputs["labels"][SOURCE_ROWS:].clone(), **{f"{name.lower()}_logits": target_control["logits"][name] for name in EVALUATOR_NAMES}}
        target_path = target / "target_image_evaluator_positive_controls.pt"
        if target_path.exists():
            _require(_tree_equal(torch.load(target_path, map_location="cpu", weights_only=True), target_payload), "saved target positive-control output differs")
        else:
            _write_torch(target_path, target_payload, "four evaluator target-image controls on the new 300-row canonical TRAIN population")

        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
        parameter_before = capacity._snapshot_parameters(params)
        adam_before = _cpu_tree(optimizer.state_dict())
        frozen_before = capacity._frozen_runtime_state(consumer)
        baseline = _measure_forward(consumer, evaluators, measure_inputs)
        replay = _source_readout_replay(baseline, parent_readouts["A_step3000"])
        baseline_path = target / "readouts_step3000.pt"
        if baseline_path.exists():
            _require(_tree_equal(torch.load(baseline_path, map_location="cpu", weights_only=True), baseline), "saved step3000 baseline readout differs")
        else:
            _write_torch(baseline_path, baseline, "1200-row source step3000 forward replay and new confirmation baseline")
        _require(_tree_equal(parameter_before, capacity._snapshot_parameters(params)), "source forward changed model parameters")
        _require(_tree_equal(adam_before, _cpu_tree(optimizer.state_dict())), "source forward changed Adam state")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "source forward changed frozen runtime")
        _record_check(checks, "source_900_row_production_replay", {"executed": True, "passed": replay["passed"], "rtol": replay["rtol"], "atol": replay["atol"], "fields": replay["fields"], "new_confirmation_rows_measured": NEW_CONFIRMATION_ROWS})
        del consumer, optimizer, params, baseline
        gc.collect()
        torch.cuda.empty_cache()

        batch_cpu, labels_cpu, _ = capacity.training_batch(fit_inputs["images"], fit_inputs["labels"], 0)
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, device)
        batch, labels = batch_cpu.to(device), labels_cpu.to(device)
        _, a_losses, _, a_direction = ablation._large_microbatch_loss_gradients(consumer, params, evaluators, batch, labels, SOURCE_ARM)
        _, ref_losses, _, ref_direction = capacity._large_microbatch_loss_gradients(consumer, params, evaluators, batch, labels)
        reference_error = {name: float((a_direction[name] - ref_direction[name]).abs().max().detach().cpu()) for name in ("encoder.phase", "generator.phase")}
        _require(all(torch.allclose(a_direction[name], ref_direction[name], rtol=2e-5, atol=2e-7) for name in reference_error), "A arm no longer reproduces the existing three-evaluator production direction")
        _record_check(checks, "A_original_rule_gradient_reference", {"executed": True, "passed": True, "reference": "existing capacity three-evaluator microbatch producer", "max_abs_difference": reference_error})
        del consumer, optimizer, params, a_losses, ref_losses, a_direction, ref_direction
        gc.collect()
        torch.cuda.empty_cache()

        trial_a = _run_three_step_trial(fit_inputs, base_state, source, evaluators, ARM_NAMES[0])
        trial_a_repeat = _run_three_step_trial(fit_inputs, base_state, source, evaluators, ARM_NAMES[0])
        aa_passed, aa_detail = capacity._repeatability(trial_a, trial_a_repeat)
        _record_check(checks, "A_A_three_step_repeatability", {"executed": True, "passed": aa_passed, "from_step": SOURCE_STEP, "to_step": SOURCE_STEP + 3, "detail": aa_detail})
        _require(aa_passed, "A/A three-update repeatability failed")
        del trial_a, trial_a_repeat
        trial_b = _run_three_step_trial(fit_inputs, base_state, source, evaluators, ARM_NAMES[1])
        _record_check(checks, "B_weighted_three_step_production_update", {"executed": True, "passed": len(trial_b["records"]) == 3 and all(row["weighted_ce_denominator"] == 11.0 and row["new_400_plane_gradient_norm"] > 0 for row in trial_b["records"]), "steps": [row["generator_adam_step"] for row in trial_b["records"]], "class9_weight": 2.0, "full_batch_denominator": 11.0, "both_generator_phase_groups_updated": True})
        del trial_b
        gc.collect()
        torch.cuda.empty_cache()
        negative = _weighted_wrong_label_control(fit_inputs, base_state, source, evaluators)
        _record_check(checks, "weighted_wrong_label_negative_control", {"executed": True, **negative})
        recovery = _recovery_control(fit_inputs, base_state, source, evaluators, population)
        _record_check(checks, "weighted_interrupted_resume_equivalence", {"executed": True, **recovery})
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight changed evaluator {name}")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight changed evaluator {name} isolation")
        _record_check(checks, "frozen_evaluator_state", {"executed": True, "passed": True, "evaluators": list(EVALUATOR_NAMES)})
        source_unchanged = _tree_equal(source_before, source)
        _record_check(checks, "source_checkpoint_and_adam_immutability", {"executed": True, "passed": source_unchanged, "source_adam_step": SOURCE_STEP})
        _require(source_unchanged, "preflight changed the source checkpoint or Adam state")
        _require(all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "READY lacks an executed passing production check")
        return {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "READY", "device": str(device), "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates_per_arm": ADDED_UPDATES, "population_rows": TOTAL_ROWS, "numeric_mode": NUMERIC_MODE, "disk_observation": disk, "checks": checks, "duration_seconds_observation_only": time.perf_counter() - started}
    finally:
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight final evaluator {name}")
        capacity._restore_rng_state(rng_before)
        del evaluators
        gc.collect()
        torch.cuda.empty_cache()


def _make_run_control(population: Mapping[str, Any], source: Mapping[str, Any], preflight: Mapping[str, Any], fit_inputs: Mapping[str, Any], base_state: Mapping[str, Any]):
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-large-generator-class9-weight2-3000-to-3700",
        semantic_authority_id=PARENT_TASK_ID,
        data_role="canonical TRAIN: fit=200 updated; other 1000 source/new rows excluded from gradients",
        config_identity={"grid": 2048, "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "added_updates_per_arm": ADDED_UPDATES, "batch_size": BATCH_SIZE, "cycles": ADDED_UPDATES // BATCH_COUNT, "microbatch_size": 2, "weighted_class": 9, "weighted_factor": 2.0, "numeric_mode": NUMERIC_MODE, "training_evaluators": ["E_S", "E_I", "E_C"]},
        model_identity={"checkpoint": population["source_checkpoint"], "parameter_shapes": dict(population["source_parameter_shapes"]), "architecture": source["architecture_layout"]},
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": capacity.GENERATOR_LEARNING_RATE, "betas": list(capacity.GENERATOR_BETAS), "eps": capacity.GENERATOR_EPS, "weight_decay": 0.0, "reset": False},
        sampler_identity={"fit_rows": population["source_row_indices"][:FIT_ROWS], "new_confirmation_rows": population["source_row_indices"][SOURCE_ROWS:], "confirmation_seed": CONFIRMATION_SEED, "fit_batch_size": BATCH_SIZE, "fit_batch_count": BATCH_COUNT},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01")
    checks = preflight["checks"]
    _require(bool(checks) and all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY receipt cannot include missing or failed checks")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "target_controls": checks["new_confirmation_target_positive_control"]["passed"], "source_replay": checks["source_900_row_production_replay"]["passed"], "weighted_ce_gradient_reference": checks["weighted_ce_value_and_logit_gradient_reference"]["passed"], "A_original_rule_reference": checks["A_original_rule_gradient_reference"]["passed"], "A_A_three_steps": checks["A_A_three_step_repeatability"]["passed"], "B_weighted_three_steps": checks["B_weighted_three_step_production_update"]["passed"], "negative_control": checks["weighted_wrong_label_negative_control"]["passed"], "resume_equivalence": checks["weighted_interrupted_resume_equivalence"]["passed"]},
        resource={"passed": checks["disk_and_external_io"]["passed"], "status": "PASS", "cuda_available": True, "disk": checks["disk_and_external_io"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_generator_class9_weighted_ce_comparison_3000_to_3700"},
        selected_engine="cuda_2048_generator_class9_weighted_ce_comparison_3000_to_3700",
        resource_envelope={"disk": checks["disk_and_external_io"]},
    )
    return control, receipt


def _ready_files(target: Path, population: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    population_path, preflight_path = target / "input_population.json", target / "preflight.json"
    saved = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}
    if resume:
        _require(json.loads(population_path.read_text(encoding="utf-8")) == dict(population), "saved task population differs")
        prior = json.loads(preflight_path.read_text(encoding="utf-8"))
        _require(prior == saved, "saved READY receipt is missing or differs")
    else:
        if population_path.exists():
            _require(json.loads(population_path.read_text(encoding="utf-8")) == dict(population), "staged population differs")
        else:
            _write_json(population_path, population, "fixed 1200-row canonical TRAIN roles")
        if preflight_path.exists():
            _require(json.loads(preflight_path.read_text(encoding="utf-8")) == saved, "staged READY receipt differs")
        else:
            _write_json(preflight_path, saved, "executed READY preflight receipt")


def _validate_saved_target_positive_control(target: Path, measure_inputs: Mapping[str, Any], population: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = target / "target_image_evaluator_positive_controls.pt"
    _require(path.is_file(), "READY stage lacks the executed target-image positive-control logits")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    start, stop = map(int, population["role_positions"]["new_confirmation"])
    expected_indices = measure_inputs["source_indices"][start:stop].cpu()
    expected_labels = measure_inputs["labels"][start:stop].cpu()
    _require(torch.equal(payload.get("source_row_indices"), expected_indices), "saved positive-control source rows differ from the fixed population")
    _require(torch.equal(payload.get("labels"), expected_labels), "saved positive-control labels differ from the fixed population")
    _require(expected_labels.numel() == NEW_CONFIRMATION_ROWS, "saved positive-control population size differs")
    actual_summary: dict[str, Any] = {}
    for name in EVALUATOR_NAMES:
        key = f"{name.lower()}_logits"
        logits = payload.get(key)
        _require(torch.is_tensor(logits) and tuple(logits.shape) == (NEW_CONFIRMATION_ROWS, 10) and bool(torch.isfinite(logits).all()), f"saved positive-control logits are invalid for {name}")
        prediction = logits.to(torch.float64).argmax(dim=1)
        correct = [int(((prediction == expected_labels) & (expected_labels == class_id)).sum()) for class_id in range(10)]
        macro = sum(correct) / NEW_CONFIRMATION_ROWS
        actual_summary[name] = {"per_class_correct": correct, "macro_recall": macro, "passed": macro >= 0.90 and correct[1] >= 24 and correct[7] >= 24 and correct[9] >= 24}
    saved_check = preflight.get("checks", {}).get("new_confirmation_target_positive_control", {})
    _require(saved_check.get("executed") is True and saved_check.get("passed") is True, "READY receipt does not contain a passed target positive-control check")
    _require(actual_summary == saved_check.get("evaluators"), "saved target positive-control logits do not reproduce READY qualification")
    return {"executed": True, "passed": True, "role": "new_confirmation", "evaluators": actual_summary}


def _consume_saved_ready(target: Path, population: Mapping[str, Any], measure_inputs: Mapping[str, Any], preflight: Mapping[str, Any], control: ResearchRunControl, expected: PreflightReceipt, parent_readouts: Mapping[str, Any]) -> PreflightReceipt:
    ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY" and ready.get("preflight") == preflight, "saved preflight evidence is missing or differs")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == dict(population), "saved population differs from the READY attempt")
    saved = PreflightReceipt.from_mapping(ready.get("receipt", {}))
    _require(saved.source_semantics_id == TASK_ID and saved.authorization_id == expected.authorization_id and saved.status == "READY", "saved READY receipt is bound to another task")
    _require(all(value.get("executed") is True and value.get("passed") is True for value in preflight["checks"].values()), "saved READY receipt omits an executed passing check")
    _validate_saved_target_positive_control(target, measure_inputs, population, preflight)
    baseline_path = target / "readouts_step3000.pt"
    _require(baseline_path.is_file(), "READY stage lacks the executed 1200-row source forward readout")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    replay = _source_readout_replay(baseline, parent_readouts["A_step3000"])
    replay_check = preflight["checks"].get("source_900_row_production_replay", {})
    _require(replay_check.get("executed") is True and replay_check.get("passed") is True and replay["passed"], "READY does not contain a reproducible source replay")
    control.mark_preflight_ready(saved)
    return saved


def _class_weights(labels: torch.Tensor, arm: str) -> torch.Tensor:
    _require(arm in ARM_NAMES, "unknown class9 weighted-CE arm")
    weights = torch.ones_like(labels, dtype=torch.float32)
    if arm == ARM_NAMES[1]:
        weights = torch.where(labels == 9, torch.full_like(weights, 2.0), weights)
    return weights


def _weighted_ce_piece(logits: torch.Tensor, labels: torch.Tensor, row_weights: torch.Tensor, full_batch_weight_sum: torch.Tensor | float) -> torch.Tensor:
    """One microbatch contribution using the complete logical-batch denominator."""
    _require(logits.ndim == 2 and labels.shape == logits.shape[:1] and row_weights.shape == labels.shape, "weighted CE microbatch shapes differ")
    denominator = torch.as_tensor(full_batch_weight_sum, dtype=logits.dtype, device=logits.device)
    _require(bool(torch.isfinite(denominator)) and float(denominator.detach().cpu()) > 0.0, "weighted CE full-batch denominator is invalid")
    per_row = F.cross_entropy(logits, labels, reduction="none")
    return (per_row * row_weights.to(dtype=logits.dtype)).sum() / denominator


def _weighted_microbatch_loss_gradients(consumer: Any, parameters: Mapping[str, torch.nn.Parameter], evaluators: Mapping[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor):
    _require(images.shape[0] == BATCH_SIZE and labels.shape[0] == BATCH_SIZE, "weighted semantic update must contain the full ten-row logical batch")
    _require(torch.bincount(labels.detach().cpu(), minlength=10).tolist() == [1] * 10, "weighted semantic batch must contain exactly one row from every class")
    evaluator_names = ("E_S", "E_I", "E_C")
    references = consumer.references
    _require(references is not None, "frozen BOGT references are missing")
    row_weights = _class_weights(labels, ARM_NAMES[1]).to(images.device)
    full_weight_sum = row_weights.sum()
    _require(float(full_weight_sum.detach().cpu()) == 11.0, "class9-weighted logical batch must have denominator 11")

    coefficient_parts = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
            stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
            result = consumer.runtime.forward_from_image(images[start:stop])
            coefficient_parts.append(result.z_ols.coefficients.detach())
            del result
    coefficients = torch.cat(coefficient_parts, dim=0)
    l_dist, d_dist_d_ols = capacity._dist_loss_and_ols_gradient(coefficients, references.scale)
    parameter_names = ("encoder.phase", "generator.phase")
    dist_gradients = {name: torch.zeros_like(parameters[name]) for name in parameter_names}
    bogt_gradients = {name: torch.zeros_like(parameters[name]) for name in parameter_names}
    semantic_gradients = {name: {key: torch.zeros_like(parameters[key]) for key in parameter_names} for name in evaluator_names}
    loss_totals = {"L_rec": 0.0, "L_op": 0.0, **{f"{name}_ce": 0.0 for name in evaluator_names}}
    power_parts: list[torch.Tensor] = []
    extra_power_parts: list[torch.Tensor] = []
    for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
        fraction = (stop - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw = result.raw_intensity
        target_raw = result.image_input.slm_intensity
        z = result.z_ols.coefficients
        d_dist_chunk = d_dist_d_ols[start:stop].to(dtype=z.dtype, device=z.device)
        l_dist_proxy = torch.real(torch.conj(d_dist_chunk) * z).sum()
        l_rec = (raw - target_raw).square().mean()
        planes = capacity.bogt.contracted_plane_powers(result)
        l_op = capacity.bogt.optical_power_loss_from_powers(planes, references.power_reference)
        view = capacity.canonical.cmos_view(raw)
        ce_losses: dict[str, torch.Tensor] = {}
        for name in evaluator_names:
            logits = capacity._evaluator_logits(evaluators[name], view, name)
            ce_losses[name] = _weighted_ce_piece(logits, labels[start:stop], row_weights[start:stop], full_weight_sum)
        grad_dist = capacity._gradient(l_dist_proxy, parameters, retain_graph=True)
        bogt_loss = capacity.semantic.BETA * l_rec + capacity.semantic.GAMMA * l_op
        grad_bogt = capacity._gradient(bogt_loss, parameters, retain_graph=True)
        grad_semantic = {}
        for index, name in enumerate(evaluator_names):
            grad_semantic[name] = capacity._gradient(ce_losses[name], parameters, retain_graph=index < len(evaluator_names) - 1)
        for name in parameter_names:
            dist_gradients[name] += grad_dist[name]
            bogt_gradients[name] += fraction * grad_bogt[name]
            for evaluator_name in evaluator_names:
                semantic_gradients[evaluator_name][name] += grad_semantic[evaluator_name][name]
        loss_totals["L_rec"] += fraction * float(l_rec.detach().to(torch.float64).cpu())
        loss_totals["L_op"] += fraction * float(l_op.detach().to(torch.float64).cpu())
        for name, loss in ce_losses.items():
            loss_totals[f"{name}_ce"] += float(loss.detach().to(torch.float64).cpu())
        power_parts.append(planes.detach())
        generator = consumer.runtime.generator
        _require(generator.last_extra_plane_powers is not None, "large Generator omitted added-plane power")
        extra_power_parts.append(generator.last_extra_plane_powers.detach())
        del result, raw, target_raw, z, d_dist_chunk, l_dist_proxy, l_rec, planes, l_op, view
        del ce_losses, grad_dist, bogt_loss, grad_bogt, grad_semantic
    semantic_gradient, unavailable = ablation._canonical_semantic_gradient(semantic_gradients, evaluator_names)
    _require(semantic_gradient is not None, f"weighted semantic direction is unavailable: {unavailable}")
    _require(capacity._vector_norm(dist_gradients["generator.phase"]) == 0.0, "L_dist produced a Generator gradient")
    combined = {name: capacity.semantic.ALPHA * dist_gradients[name] + bogt_gradients[name] + capacity.semantic.KAPPA * semantic_gradient[name] for name in parameter_names}
    _require(all(bool(torch.isfinite(value).all()) for value in combined.values()), "weighted combined gradient is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only weighted autograd populated optical .grad")
    losses = {"L_dist": l_dist, **{name: torch.tensor(value, device=images.device) for name, value in loss_totals.items()}}
    consumer.runtime.generator.last_extra_plane_powers = torch.cat(extra_power_parts, dim=0)
    return torch.cat(power_parts, dim=0), losses, {"L_dist": dist_gradients, "semantic": semantic_gradients}, combined


def _train_step_weighted(consumer: Any, optimizer: torch.optim.Optimizer, parameters: Mapping[str, torch.nn.Parameter], evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], completed: int) -> dict[str, Any]:
    batch, labels_cpu, positions = capacity.training_batch(inputs["images"], inputs["labels"], completed % BATCH_COUNT)
    labels = labels_cpu.to(consumer.device)
    planes, losses, gradients, combined = _weighted_microbatch_loss_gradients(consumer, parameters, evaluators, batch.to(consumer.device), labels)
    phase = parameters["generator.phase"]
    generator_gradient = combined["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the weighted optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(float(optimizer.state[phase]["step"].item()))
    _require(step == completed + 1, "weighted Generator Adam step differs from cumulative update")
    delta = phase.detach() - before
    old_gradient, new_gradient = capacity._phase_parts(generator_gradient, "B_large")
    old_delta, new_delta = capacity._phase_parts(delta, "B_large")
    extra_power = consumer.runtime.generator.last_extra_plane_powers
    record = {
        "update": step,
        "arm": ARM_NAMES[1],
        "batch_index": int(positions[0].item() // BATCH_SIZE),
        "fit_positions": positions.tolist(),
        "source_row_indices": inputs["source_indices"].index_select(0, positions).tolist(),
        "labels": labels_cpu.tolist(),
        "training_evaluators": ["E_S", "E_I", "E_C"],
        "semantic_weight_policy": "class9_x2",
        "class_weights_by_label": [1.0] * 9 + [2.0],
        "weighted_ce_denominator": 11.0,
        "ce_by_evaluator": {name: float(losses[f"{name}_ce"].detach().cpu()) for name in ("E_S", "E_I", "E_C")},
        "raw_mse": float(losses["L_rec"].detach().cpu()),
        "l_dist": float(losses["L_dist"].detach().cpu()),
        "l_op": float(losses["L_op"].detach().cpu()),
        "five_plane_power_mean_by_batch": planes.detach().to(torch.float64).mean(dim=0).cpu().tolist(),
        "new_phase_plane_power_mean_by_batch": extra_power.detach().to(torch.float64).mean(dim=0).cpu().tolist(),
        "generator_gradient_norm": capacity._vector_norm(generator_gradient),
        "generator_gradient_rms": capacity._vector_rms(generator_gradient),
        "old_100_plane_gradient_norm": capacity._vector_norm(old_gradient),
        "new_400_plane_gradient_norm": capacity._vector_norm(new_gradient),
        "old_100_plane_displacement_rms": capacity._vector_rms(old_delta),
        "new_400_plane_displacement_rms": capacity._vector_rms(new_delta),
        "generator_displacement_rms": capacity._vector_rms(delta),
        "generator_adam_step": step,
        "all_parameters_finite": bool(torch.isfinite(phase).all().item()),
    }
    _require(record["all_parameters_finite"] and record["new_400_plane_gradient_norm"] > 0.0, "weighted update is nonfinite or omitted added phase planes")
    return record


def _training_loop(target: Path, fit_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], arm: str, start_from: Mapping[str, Any] | None) -> dict[str, Any]:
    _require(arm in ARM_NAMES, "unknown training arm")
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, torch.device("cuda:0"))
    if start_from is not None:
        _require(start_from.get("arm") == arm, "resume checkpoint selected another arm")
        _restore_task_checkpoint(start_from, consumer, optimizer)
    completed = SOURCE_STEP if start_from is None else int(start_from["completed_updates"])
    history = list(source["training_history"] if start_from is None else start_from["training_history"])
    _require(SOURCE_STEP <= completed <= FINAL_STEP and len(history) == completed, "training start count/history differs")
    if not (target / _checkpoint_name(arm, completed)).exists():
        _save_checkpoint(target, consumer, optimizer, arm, completed, history, population, source)
    encoder_before = params["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    stop_requested = False
    prior_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while completed < FINAL_STEP:
            if arm == ARM_NAMES[0]:
                record = ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, completed, arm=SOURCE_ARM)
                record["arm"] = arm
                record["semantic_weight_policy"] = "uniform_x1"
                record["class_weights_by_label"] = [1.0] * 10
                record["weighted_ce_denominator"] = 10.0
            else:
                record = _train_step_weighted(consumer, optimizer, params, evaluators, fit_inputs, completed)
            completed += 1
            history.append(record)
            if completed % CHECKPOINT_INTERVAL == 0 or stop_requested:
                path = _save_checkpoint(target, consumer, optimizer, arm, completed, history, population, source)
                _resource(target, f"{arm}_checkpoint", completed)
                _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed frozen Encoder")
                capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "training frozen OLS/runtime state")
                for name, model in evaluators.items():
                    capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"training evaluator {name}")
                    _require(all(param.grad is None and not param.requires_grad for param in model.parameters()), f"training evaluator {name} isolation changed")
                gc.collect()
                torch.cuda.empty_cache()
                if stop_requested:
                    raise TrainingInterrupted(f"saved {arm} checkpoint {path.name} at a completed update boundary")
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    _require(completed == FINAL_STEP and len(history) == FINAL_STEP, "training did not finish at exactly 3700 updates")
    _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed Encoder at completion")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "frozen runtime at completion")
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"evaluator {name} at completion")
    final_path = target / _checkpoint_name(arm, FINAL_STEP)
    if not final_path.exists():
        _save_checkpoint(target, consumer, optimizer, arm, FINAL_STEP, history, population, source)
    history_payload = {"task_id": TASK_ID, "arm": arm, "source_step": SOURCE_STEP, "additional_updates": ADDED_UPDATES, "final_step": FINAL_STEP, "training_evaluators": ["E_S", "E_I", "E_C"], "semantic_weight_policy": "uniform_x1" if arm == ARM_NAMES[0] else "class9_x2", "records": history}
    history_path = target / ("training_history_A_original.json" if arm == ARM_NAMES[0] else "training_history_B_class9_weight2.json")
    if history_path.exists():
        _require(json.loads(history_path.read_text(encoding="utf-8")) == history_payload, "saved final training history differs")
    else:
        _write_json(history_path, history_payload, "3000-step source plus 700 class9 weighted comparison updates")
    result = torch.load(final_path, map_location="cpu", weights_only=True)
    del consumer, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _ensure_start_checkpoints(target: Path, fit_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    starts = {}
    for arm in ARM_NAMES:
        path = target / _checkpoint_name(arm, SOURCE_STEP)
        if path.is_file():
            checkpoint = _read_checkpoint(path, target, population, source, measurement_baseline=True)
        else:
            consumer, optimizer, _parameters = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, torch.device("cuda:0"))
            payload = _checkpoint_payload(consumer, optimizer, arm, SOURCE_STEP, source["training_history"], population, source)
            _write_torch(path, payload, "independent copied 3000-step start state with inherited Adam and RNG")
            checkpoint = payload
            del consumer, optimizer, _parameters
            gc.collect()
            torch.cuda.empty_cache()
        starts[arm] = checkpoint
    left, right = starts[ARM_NAMES[0]], starts[ARM_NAMES[1]]
    for key in ("trainable_parameter_state", "optimizer_state", "rng_state", "references", "weights", "frozen_runtime_state"):
        _require(_tree_equal(left[key], right[key]), f"A/B start checkpoints differ for {key}")
    return starts


def _endpoint(target: Path, name: str, checkpoint: Mapping[str, Any], arm: str, fit_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = target / f"readouts_{name}.pt"
    if path.is_file():
        readouts = torch.load(path, map_location="cpu", weights_only=True)
        _require(torch.equal(readouts["source_indices"], measure_inputs["source_indices"]) and torch.equal(readouts["labels"], measure_inputs["labels"]), f"saved {name} readout population differs")
    else:
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, SOURCE_ARM, torch.device("cuda:0"))
        if checkpoint.get("task_id") == TASK_ID:
            _require(checkpoint.get("arm") == arm, f"{name} endpoint checkpoint belongs to another arm")
            _restore_task_checkpoint(checkpoint, consumer, optimizer)
        else:
            _require(checkpoint.get("task_id") == PARENT_TASK_ID and checkpoint.get("completed_updates") == SOURCE_STEP, "baseline endpoint is not the fixed source checkpoint")
            _restore_task_checkpoint(checkpoint, consumer, optimizer)
        param_before = capacity._snapshot_parameters(params)
        adam_before = _cpu_tree(optimizer.state_dict())
        frozen_before = capacity._frozen_runtime_state(consumer)
        readouts = _measure_forward(consumer, evaluators, measure_inputs)
        _require(_tree_equal(param_before, capacity._snapshot_parameters(params)), f"{name} forward changed model parameters")
        _require(_tree_equal(adam_before, _cpu_tree(optimizer.state_dict())), f"{name} forward changed Adam state")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{name} forward changed frozen runtime")
        _write_torch(path, readouts, f"1200-row frozen production readout {name}")
        del consumer, optimizer, params
        gc.collect()
        torch.cuda.empty_cache()
    summary, rows = _row_and_image_metrics(readouts, population)
    metric_path = target / f"metrics_{name}.json"
    row_path = target / f"row_metrics_{name}.pt"
    if metric_path.exists():
        _require(json.loads(metric_path.read_text(encoding="utf-8")) == summary, f"saved {name} metrics differ")
    else:
        _write_json(metric_path, summary, "1200-row class and image measurements")
    if row_path.exists():
        _require(_tree_equal(torch.load(row_path, map_location="cpu", weights_only=True), rows), f"saved {name} row metrics differ")
    else:
        _write_torch(row_path, rows, "1200-row float64 per-row scoring and image metrics")
    return summary, readouts


def _main(args: argparse.Namespace) -> dict[str, Any]:
    global _STARTED
    _STARTED = time.perf_counter()
    root = _root()
    presentation_dir = _resolve(root, args.presentation_dir)
    target = _resolve(root, args.output_dir)
    _require(bool(args.resume) == (args.checkpoint is not None), "--resume requires an explicit --checkpoint")
    _require(not (args.preflight_only and args.resume), "--preflight-only cannot be combined with --resume")
    fit_inputs, measure_inputs, population, paths, base_state, source, source_checkpoints, parent_readouts, source_report = _population(root, presentation_dir)
    created_target = _prepare_target(root, target, resume=args.resume)

    if args.resume:
        _require((target / "input_population.json").is_file() and (target / "preflight.json").is_file(), "resume lacks the fixed population or READY receipt")
        _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == population, "resume population differs")
        ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
        _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY", "resume READY receipt is missing")
        preflight = ready["preflight"]
        control, expected_receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
        receipt = _consume_saved_ready(target, population, measure_inputs, preflight, control, expected_receipt, parent_readouts)
        selected = _read_checkpoint(_resolve(root, args.checkpoint), target, population, source)
        report_path = target / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _require(selected["arm"] == ARM_NAMES[1] and selected["completed_updates"] == FINAL_STEP and report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "terminal recovery requires the explicit B3700 checkpoint")
            return report
    else:
        if created_target:
            _write_json(target / "input_population.json", population, "fixed 1200-row canonical TRAIN population and role map")
            with capacity.strict_fp32_mode() as mode:
                _require(mode == NUMERIC_MODE, "active preflight numeric mode differs from strict FP32")
                preflight = _preflight(fit_inputs, measure_inputs, population, paths, base_state, source, parent_readouts, target)
            control, receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
            _ready_files(target, population, preflight, receipt, resume=False)
            selected = None
        else:
            ready_files = {"input_population.json", "preflight.json", "target_image_evaluator_positive_controls.pt", "readouts_step3000.pt"}
            _require({path.name for path in target.iterdir()} == ready_files, "existing output is not an untouched READY stage")
            _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == population, "staged population differs")
            ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
            _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY", "staged READY receipt is missing")
            preflight = ready["preflight"]
            control, expected_receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
            receipt = _consume_saved_ready(target, population, measure_inputs, preflight, control, expected_receipt, parent_readouts)
            selected = None
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}

    control.mark_preflight_ready(receipt)
    control.start_execution()
    control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == NUMERIC_MODE, "active execution numeric mode differs from strict FP32")
        evaluators = diagnostic.load_evaluators(paths, torch.device("cuda:0"))
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        starts = _ensure_start_checkpoints(target, fit_inputs, population, base_state, source)
        if selected is None:
            final_a = _training_loop(target, fit_inputs, population, base_state, source, evaluators, ARM_NAMES[0], starts[ARM_NAMES[0]])
            final_b = _training_loop(target, fit_inputs, population, base_state, source, evaluators, ARM_NAMES[1], starts[ARM_NAMES[1]])
        elif selected["arm"] == ARM_NAMES[0]:
            final_a = _training_loop(target, fit_inputs, population, base_state, source, evaluators, ARM_NAMES[0], selected)
            final_b = _training_loop(target, fit_inputs, population, base_state, source, evaluators, ARM_NAMES[1], starts[ARM_NAMES[1]])
        else:
            final_a = _read_checkpoint(target / _checkpoint_name(ARM_NAMES[0], FINAL_STEP), target, population, source, measurement_baseline=True)
            final_b = _training_loop(target, fit_inputs, population, base_state, source, evaluators, ARM_NAMES[1], selected)
        _require(final_a.get("completed_updates") == FINAL_STEP and final_b.get("completed_updates") == FINAL_STEP, "both arms must end at exactly 3700 updates")
        start_summary, start_readout = _endpoint(target, "step3000", source, SOURCE_ARM, fit_inputs, measure_inputs, population, base_state, source, evaluators)
        summary_a, readout_a = _endpoint(target, "A_step3700", final_a, ARM_NAMES[0], fit_inputs, measure_inputs, population, base_state, source, evaluators)
        summary_b, readout_b = _endpoint(target, "B_step3700", final_b, ARM_NAMES[1], fit_inputs, measure_inputs, population, base_state, source, evaluators)
    replay = _source_readout_replay(start_readout, parent_readouts["A_step3000"])
    decision = _final_decision(start_summary, summary_a, summary_b, valid=True)
    transitions: dict[str, Any] = {}
    for role, bounds in population["role_positions"].items():
        start_pos, stop_pos = map(int, bounds)
        labels = measure_inputs["labels"][start_pos:stop_pos]
        predictions = {
            "step3000": start_readout["e_d_logits"][start_pos:stop_pos].to(torch.float64).argmax(1),
            "A_step3700": readout_a["e_d_logits"][start_pos:stop_pos].to(torch.float64).argmax(1),
            "B_step3700": readout_b["e_d_logits"][start_pos:stop_pos].to(torch.float64).argmax(1),
        }
        transitions[role] = {}
        for left_name, right_name in (("step3000", "A_step3700"), ("step3000", "B_step3700"), ("A_step3700", "B_step3700")):
            left, right = predictions[left_name], predictions[right_name]
            transitions[role][f"{left_name}_to_{right_name}"] = {
                str(class_id): {
                    "correct_to_wrong": int(((labels == class_id) & (left == class_id) & (right != class_id)).sum()),
                    "wrong_to_correct": int(((labels == class_id) & (left != class_id) & (right == class_id)).sum()),
                }
                for class_id in range(10)
            }
    images = _comparison_pages(target, measure_inputs, start_readout, readout_a, readout_b, population)
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final evaluator {name}")
    class_mapping = {str(class_id): {"source_step3000": decision["new_confirmation"]["per_class_correct"]["source_step3000"][class_id], "A_step3700": decision["new_confirmation"]["per_class_correct"]["A_step3700"][class_id], "B_step3700": decision["new_confirmation"]["per_class_correct"]["B_step3700"][class_id]} for class_id in range(10)}
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": decision["status"],
        "classification": decision["classification"],
        "scientific_updates": {"A": ADDED_UPDATES, "B": ADDED_UPDATES},
        "parent": {"task_id": PARENT_TASK_ID, "status": source_report["status"], "classification": source_report["classification"], "source_checkpoint": population["source_checkpoint"]},
        "scope": {"source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates_per_arm": ADDED_UPDATES, "fit_rows": FIT_ROWS, "source_rows": SOURCE_ROWS, "new_confirmation_rows": NEW_CONFIRMATION_ROWS, "total_measurement_rows": TOTAL_ROWS, "all_rows_canonical_train": True, "new_confirmation_unseen_claim": False, "updated_parameter": "generator.phase", "encoder_ols_geometry_buffers_evaluators_frozen": True, "sampler_cycles": ADDED_UPDATES // BATCH_COUNT, "training_evaluators": ["E_S", "E_I", "E_C"], "class9_weight_B": 2.0, "class9_weight_A": 1.0, "numeric_mode": NUMERIC_MODE},
        "source_readout_replay": replay,
        "endpoint_metrics": {"step3000": start_summary, "A_step3700": summary_a, "B_step3700": summary_b},
        "new_confirmation_class_counts_by_id": class_mapping,
        "paired_correctness_transitions": transitions,
        **decision,
        "readout_artifacts": {"step3000": "readouts_step3000.pt", "A_step3700": "readouts_A_step3700.pt", "B_step3700": "readouts_B_step3700.pt"},
        "row_metric_artifacts": {"step3000": "row_metrics_step3000.pt", "A_step3700": "row_metrics_A_step3700.pt", "B_step3700": "row_metrics_B_step3700.pt"},
        "comparison_images": images,
        "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_[ab]_step*.pt")),
        "training_history_artifacts": {"A_original": "training_history_A_original.json", "B_class9_weight2": "training_history_B_class9_weight2.json"},
        "preflight_artifact": "preflight.json",
        "elapsed_seconds_observation_only": time.perf_counter() - _STARTED,
        "next_action": "NONE",
        "stop": True,
    }
    control.complete() if decision["status"] == "VALID_COMPLETED" else control.valid_failed()
    report["research_run_state"] = control.state
    _write_json(target / "report.json", report, "single class9-weighted CE comparison decision")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit task checkpoint required with --resume")
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    root = _root()
    target = _resolve(root, args.output_dir)
    try:
        return _main(args)
    except TrainingInterrupted as error:
        incident = {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume from the explicit checkpoint named in failure"}
        path = target / "interrupted.json"
        if target.is_dir() and not path.exists():
            path.write_text(json.dumps(incident, ensure_ascii=False, indent=2), encoding="utf-8")
        return incident
    except Exception as error:
        path = target / "invalid_harness.json"
        if not target.exists():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir(exist_ok=False)
            except OSError:
                pass
        if target.is_dir() and not (target / "report.json").exists() and not path.exists():
            saved_steps = sorted(
                int(match.group(2))
                for candidate in target.glob("arm_[ab]_step*.pt")
                if (match := CHECKPOINT_RE.fullmatch(candidate.name))
            )
            incident = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "failure_type": type(error).__name__, "failure": str(error), "source_step": SOURCE_STEP, "last_durable_checkpoint_step": max(saved_steps) if saved_steps else None, "scientific_updates": None, "executed_checks": dict(_EXECUTED_CHECKS), "scientific_conclusion": None, "next_action": "NONE"}
            try:
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(incident, handle, ensure_ascii=False, indent=2)
            except FileExistsError:
                pass
        raise


def main() -> None:
    result = execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
