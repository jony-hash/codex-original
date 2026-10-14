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


TASK_ID = "M0_GENERATOR_TEN_CLASS_PRESENTATION_ACCUMULATION_V1"
PARENT_TASK_ID = ablation.TASK_ID
REPORT_SCHEMA = "m0_generator_ten_class_presentation_accumulation_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_ten_class_presentation_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_ten_class_presentation_checkpoint.v1"
DEFAULT_PARENT = Path("outputs/20261008_m0_generator_evaluator_ablation_comparison_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_ten_class_presentation_accumulation_v1/runs/attempt_01")
SOURCE_STEP = 1700
FINAL_STEP = 3000
ADDED_UPDATES = FINAL_STEP - SOURCE_STEP
FIT_ROWS = 200
HISTORIC_ROWS = 100
OLD_CONFIRMATION_ROWS = 300
NEW_CONFIRMATION_ROWS = 300
OLD_ROWS = FIT_ROWS + HISTORIC_ROWS + OLD_CONFIRMATION_ROWS
TOTAL_ROWS = OLD_ROWS + NEW_CONFIRMATION_ROWS
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
CHECKPOINT_INTERVAL = 100
CONFIRMATION_SEED = 2026100809
RTOL, ATOL = ablation.RTOL, ablation.ATOL
ARM = ablation.ARM_NAMES[0]
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
ROLE_NAMES = ("fit", "historic", "old_confirmation", "new_confirmation")
NUMERIC_MODE = dict(ablation.NUMERIC_MODE)
CHECKPOINT_RE = re.compile(r"generator_step(\d{4})\.pt$")
RESOURCE_NAME = "resource_telemetry.jsonl"


class TenClassAccumulationError(RuntimeError):
    """Invalid source, measurement, checkpoint, or training invariant."""


class TrainingInterrupted(RuntimeError):
    """A signal was observed after a safe checkpoint boundary."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TenClassAccumulationError(message)


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
    population_path = parent_dir / "input_population.json"
    report_path = parent_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "explicit evaluator-ablation population or report is missing")
    expected_population = json.loads(population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(parent_report.get("task_id") == PARENT_TASK_ID and expected_population.get("task_id") == PARENT_TASK_ID, "named evaluator-ablation source identity differs")
    _require(parent_report.get("status") == "VALID_FAILED" and parent_report.get("classification") == "NOT_SUPPORTED_WITHIN_700_UPDATES", "source evaluator-ablation attempt is not the specified complete result")
    capacity_dir_value = expected_population.get("source_capacity_dir")
    _require(isinstance(capacity_dir_value, str) and capacity_dir_value, "ablation population lacks its explicit capacity source directory")
    capacity_dir = _resolve(root, capacity_dir_value)
    (
        fit_inputs,
        parent_population,
        paths,
        base_state,
        source_population,
        source_report,
        source_checkpoints,
        parent_readouts,
        parent_summaries,
    ) = ablation._load_parent_inputs(root, capacity_dir)
    _require(parent_population == expected_population, "saved ablation population differs from its explicit capacity lineage")
    source_report = parent_report
    source_path = parent_dir / "arm_a_step1700.pt"
    _require(source_path.is_file(), "explicit A1700 source checkpoint is missing")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    _require(source.get("task_id") == PARENT_TASK_ID and source.get("arm") == ARM, "source checkpoint task or training arm differs")
    _require(source.get("schema") == ablation.CHECKPOINT_SCHEMA and source.get("completed_updates") == SOURCE_STEP, "source checkpoint is not the A1700 task checkpoint")
    _require(source.get("optimizer_step") == SOURCE_STEP and source.get("next_batch_index") == 0 and source.get("next_batch_cursor") == 0, "source Adam step or next batch differs")
    _require(source.get("optimizer_parameter_names") == ["generator.phase"], "source optimizer includes a non-Generator parameter")
    _require(source.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "source Generator architecture differs")
    parameters = source.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping) and set(parameters) == {"encoder.phase", "generator.phase"}, "source model state is incomplete")
    _require(tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(parameters["generator.phase"].shape) == (340_000,), "source model parameter layout differs")
    _require(all(bool(torch.isfinite(value).all()) for value in parameters.values()), "source model parameters are nonfinite")
    _require(len(source.get("training_history", [])) == SOURCE_STEP, "source history does not contain 1700 updates")
    _require(_tree_equal(source.get("references"), source_checkpoints[ARM].get("references")), "source BOGT references differ from the B1000 lineage")
    _require(_tree_equal(source.get("weights"), source_checkpoints[ARM].get("weights")), "source BOGT weights differ from the B1000 lineage")
    state = source["optimizer_state"]
    group, slots = state["param_groups"], state["state"]
    _require(len(group) == 1 and group[0]["params"] == [0] and set(slots) == {0}, "source Adam parameter mapping differs")
    _require(group[0]["lr"] == capacity.GENERATOR_LEARNING_RATE and tuple(group[0]["betas"]) == capacity.GENERATOR_BETAS and group[0]["eps"] == capacity.GENERATOR_EPS and group[0]["weight_decay"] == 0.0, "source Adam hyperparameters differ")
    _require(float(slots[0]["step"].item()) == SOURCE_STEP and tuple(slots[0]["exp_avg"].shape) == (340_000,) and tuple(slots[0]["exp_avg_sq"].shape) == (340_000,), "source Adam moments or step differ")
    _require(bool(torch.isfinite(slots[0]["exp_avg"]).all()) and bool(torch.isfinite(slots[0]["exp_avg_sq"]).all()), "source Adam moments are nonfinite")
    _require(source.get("numeric_mode") == NUMERIC_MODE, "source numeric mode differs from strict FP32")

    row_indices = torch.tensor(parent_population["source_row_indices"], dtype=torch.long)
    labels = torch.tensor(parent_population["labels"], dtype=torch.long)
    _require(row_indices.numel() == OLD_ROWS and torch.equal(row_indices, fit_inputs["source_indices"]), "source row order differs from the 600-row input")
    _require(torch.equal(labels, fit_inputs["labels"]), "source labels differ from the 600-row input")
    _require(parent_population["roles"] == ["fit"] * FIT_ROWS + ["historic_holdout"] * HISTORIC_ROWS + ["confirmation"] * OLD_CONFIRMATION_ROWS, "source row roles differ")
    for step, record in enumerate(source["training_history"], start=1):
        batch_index = (step - 1) % BATCH_COUNT
        positions = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        _require(record.get("update") == step and record.get("batch_index") == batch_index and record.get("fit_positions") == positions, f"source training history batch mapping differs at step {step}")
        _require(record.get("source_row_indices") == [row_indices[pos].item() for pos in positions] and record.get("labels") == [labels[pos].item() for pos in positions], f"source training history rows or labels differ at step {step}")
    saved_path = parent_dir / "readouts_A_step1700.pt"
    _require(saved_path.is_file(), "A1700 saved readout is missing")
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    for key, expected in (("source_indices", row_indices), ("labels", labels)):
        _require(torch.equal(saved[key], expected), f"A1700 readout {key} differs from source population")
    expected_counts = source_report["endpoint_metrics"]["A_step1700"]["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"]
    ed_prediction = saved["e_d_logits"].to(torch.float64).argmax(1)
    actual_counts = [int(((labels[300:600] == class_id) & (ed_prediction[300:600] == class_id)).sum()) for class_id in range(10)]
    _require(expected_counts == actual_counts, "A1700 source report and saved logits disagree")
    parent_readouts["A_step1700"] = saved

    source_labels_all = diagnostic._read_image_labels(diagnostic._load_torch(paths["training_data"], mmap=True))[1]
    train_indices, _ = diagnostic.stratified_mnist_indices(source_labels_all, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    train_membership = torch.zeros(60_000, dtype=torch.bool)
    train_membership[train_indices] = True
    _require(bool(train_membership.index_select(0, row_indices).all()), "prior source population contains non-TRAIN rows")
    new_indices = _new_confirmation_indices(source_labels_all, row_indices, train_indices)
    new_labels = source_labels_all.index_select(0, new_indices).contiguous()
    image_store = diagnostic._load_torch(paths["training_data"], mmap=True)
    source_images, _ = diagnostic._read_image_labels(image_store)
    new_images = source_images.index_select(0, new_indices).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(bool(torch.isfinite(new_images).all()) and float(new_images.min()) >= 0 and float(new_images.max()) <= 1, "new confirmation images are invalid")
    all_indices = torch.cat((row_indices, new_indices))
    all_labels = torch.cat((labels, new_labels))
    roles = ["fit"] * FIT_ROWS + ["historic"] * HISTORIC_ROWS + ["old_confirmation"] * OLD_CONFIRMATION_ROWS + ["new_confirmation"] * NEW_CONFIRMATION_ROWS
    role_mask = {
        "fit": (0, FIT_ROWS),
        "historic": (FIT_ROWS, FIT_ROWS + HISTORIC_ROWS),
        "old_confirmation": (FIT_ROWS + HISTORIC_ROWS, OLD_ROWS),
        "new_confirmation": (OLD_ROWS, TOTAL_ROWS),
    }
    measure_inputs = {
        "images": torch.cat((fit_inputs["images"], new_images), dim=0).contiguous(),
        "source_indices": all_indices,
        "labels": all_labels,
    }
    _require(measure_inputs["images"].shape[0] == TOTAL_ROWS, "900-row measurement population shape differs")
    population = {
        "task_id": TASK_ID,
        "source_ablation_dir": parent_dir.as_posix(),
        "source_ablation_status": source_report["status"],
        "source_ablation_classification": source_report["classification"],
        "source_checkpoint": source_path.as_posix(),
        "source_checkpoint_step": SOURCE_STEP,
        "source_generator_adam_step": SOURCE_STEP,
        "source_parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
        "source_input_paths": dict(parent_population["source_input_paths"]),
        "source_row_indices": all_indices.tolist(),
        "labels": all_labels.tolist(),
        "roles": roles,
        "role_counts": {"fit": FIT_ROWS, "historic": HISTORIC_ROWS, "old_confirmation": OLD_CONFIRMATION_ROWS, "new_confirmation": NEW_CONFIRMATION_ROWS},
        "fit_rows": FIT_ROWS,
        "historic_rows": HISTORIC_ROWS,
        "old_confirmation_rows": OLD_CONFIRMATION_ROWS,
        "new_confirmation_rows": NEW_CONFIRMATION_ROWS,
        "confirmation_seed": CONFIRMATION_SEED,
        "new_confirmation_is_canonical_train_not_claimed_unseen": True,
        "updates_from_source": SOURCE_STEP,
        "additional_updates": ADDED_UPDATES,
        "final_generator_adam_step": FINAL_STEP,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
        "numeric_mode": NUMERIC_MODE,
        "role_positions": {key: [start, stop] for key, (start, stop) in role_mask.items()},
    }
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
        result[name] = {"per_class_correct": correct, "macro_recall": macro, "passed": macro >= 0.90 and correct[1] >= 24 and correct[7] >= 24}
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


def _final_decision(start: Mapping[str, Any], final: Mapping[str, Any], *, valid: bool) -> dict[str, Any]:
    before = start["evaluators"]["E_D"]["by_role"]["new_confirmation"]
    after = final["evaluators"]["E_D"]["by_role"]["new_confirmation"]
    old_correct, new_correct = before["per_class_correct"], after["per_class_correct"]
    base_total, final_total = int(before["correct_total"]), int(after["correct_total"])
    base_macro, final_macro = float(before["macro_recall"]), float(after["macro_recall"])
    image = final["image_metrics_by_role"]["new_confirmation"]
    base_image = start["image_metrics_by_role"]["new_confirmation"]
    image_ok = (
        image["raw_mse"] <= 1.05 * base_image["raw_mse"]
        and image["raw_fraction_above_one"] <= 0.01
        and image["by_class"]["1"]["raw_fraction_above_one"] <= 0.01
        and image["by_class"]["7"]["raw_fraction_above_one"] <= 0.01
        and image["raw_zero_image"] == 0
        and image["raw_constant_image"] == 0
        and final.get("all_finite") is True
    )
    conditions = {
        "new_confirmation_macro_at_least_75_percent_and_gain_at_least_5_points": final_macro >= 0.75 and final_macro >= base_macro + 0.05 and final_total >= 225 and final_total >= base_total + 15,
        "each_class_at_least_15_of_30": min(new_correct) >= 15,
        "classes_1_and_7_do_not_decline": new_correct[1] >= old_correct[1] and new_correct[7] >= old_correct[7],
        "no_class_loses_more_than_3": all(new_correct[c] >= old_correct[c] - 3 for c in range(10)),
        "raw_image_clipping_and_finite_guard": image_ok,
        "budget_roles_freeze_and_measurement_valid": valid,
    }
    if not valid:
        classification, status = "INVALID_HARNESS", "INVALID_HARNESS"
    elif all(conditions.values()):
        classification, status = "TEN_CLASS_PRESENTATION_ACCUMULATION_SUPPORTED", "VALID_COMPLETED"
    elif final_total > base_total or new_correct[1] > old_correct[1] or new_correct[7] > old_correct[7]:
        classification, status = "MIXED_OR_INSUFFICIENT", "VALID_FAILED"
    else:
        classification, status = "NOT_SUPPORTED_WITHIN_3000_UPDATES", "VALID_FAILED"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "new_confirmation": {
            "source_step": SOURCE_STEP,
            "final_step": FINAL_STEP,
            "macro_recall": {"A_step1700": base_macro, "A_step3000": final_macro},
            "correct_total": {"A_step1700": base_total, "A_step3000": final_total},
            "per_class_correct": {"A_step1700": old_correct, "A_step3000": new_correct},
            "per_class_delta": [new_correct[c] - old_correct[c] for c in range(10)],
            "raw_mse": {"A_step1700": base_image["raw_mse"], "A_step3000": image["raw_mse"]},
        },
        "training_recommendation": "CONSIDER_SEPARATE_POST_TRAINING_PRESENTATION_AND_GENERATION_VALIDATION" if classification == "TEN_CLASS_PRESENTATION_ACCUMULATION_SUPPORTED" else "NONE",
        "next_action": "NONE",
        "stop": True,
    }


def _wilson_display(view: torch.Tensor, scale: int = 4) -> Image.Image:
    image = view.detach().to(torch.float32).cpu().squeeze().clamp(0.0, 1.0)
    array = torch.round((1.0 - image) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _comparison_pages(target: Path, inputs: Mapping[str, Any], start: Mapping[str, Any], final: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    role_positions = population["role_positions"]
    pages: dict[str, list[str]] = {}
    cell, row_height = 112, 150
    for role, bounds in role_positions.items():
        role_start, role_stop = map(int, bounds)
        positions = list(range(role_start, role_stop))
        pages[role] = []
        for page_start in range(0, len(positions), 30):
            page_positions = positions[page_start : page_start + 30]
            canvas = Image.new("RGB", (720, 58 + len(page_positions) * row_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{role}; rows {page_start + 1}-{page_start + len(page_positions)} of {len(positions)}; fixed [0,1]", fill="black")
            xs = (345, 465, 585)
            for x, title in zip(xs, ("Target", "A1700", "A3000"), strict=True):
                draw.text((x, 35), title, fill="black")
            for index, position in enumerate(page_positions):
                y = 55 + index * row_height
                draw.text((8, y + 8), f"row={int(inputs['source_indices'][position])} y={int(inputs['labels'][position])} {role}", fill="black")
                for col, panel in enumerate((inputs["images"][position], start["evaluator_view"][position], final["evaluator_view"][position])):
                    canvas.paste(_wilson_display(panel), (xs[col], y))
                for col, readout in enumerate((start, final), start=1):
                    pred = int(readout["e_d_logits"][position].to(torch.float64).argmax())
                    draw.text((xs[col], y + cell + 2), f"E_D={pred}", fill="black")
            filename = f"comparison_{role}_{page_start // 30 + 1:02d}.png"
            path = target / filename
            if path.exists():
                with Image.open(path) as existing:
                    _require(existing.size == canvas.size and existing.convert("RGB").tobytes() == canvas.tobytes(), f"saved comparison page differs: {filename}")
            else:
                canvas.save(path, optimize=True)
            pages[role].append(filename)
    return {"pages_by_role": pages, "row_counts": population["role_counts"], "panel_order": ["target", "A_step1700", "A_step3000"], "fixed_display_range": [0.0, 1.0], "per_image_rescaling": False}


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
    ready_stage_files = {"input_population.json", "preflight.json", "target_image_evaluator_positive_controls.pt"}
    _require(target.is_dir() and {p.name for p in target.iterdir()} == ready_stage_files, "output already contains artifacts; use explicit resume or a new attempt directory")
    return False


def _disk_probe(target: Path, source: Path) -> dict[str, Any]:
    parent = target
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    source_bytes = source.stat().st_size
    estimated = max(1_200_000_000, source_bytes * 22 + 900 * 200 * 200 * 4 * 2)
    reserve = 1_610_612_736
    _require(free >= estimated + reserve, f"output volume free bytes {free}; estimated artifacts plus reserve require {estimated + reserve}")
    temp_root = Path(tempfile.gettempdir()).resolve()
    _require(root_not_inside(target.resolve(), temp_root), "temporary directory must be outside repository output")
    with tempfile.TemporaryDirectory(prefix="lg_tenclass_external_") as temp_name:
        probe = Path(temp_name) / "probe.bin"
        probe.write_bytes(b"lightgen-ten-class-recovery")
        _require(probe.read_bytes() == b"lightgen-ten-class-recovery", "external temporary write/read failed")
    with tempfile.TemporaryDirectory(prefix="lg_tenclass_output_", dir=target) as temp_name:
        probe = Path(temp_name) / "probe.pt"
        torch.save({"probe": torch.tensor([1.0, 2.0])}, probe)
        _require(torch.equal(torch.load(probe, map_location="cpu", weights_only=True)["probe"], torch.tensor([1.0, 2.0])), "output temporary write/read failed")
    return {"executed": True, "passed": True, "free_bytes_observed": free, "estimated_output_bytes": estimated, "reserve_bytes": reserve, "external_temp_write_read": True, "output_write_read": True}


def root_not_inside(target: Path, temp_root: Path) -> bool:
    return target != temp_root and target not in temp_root.parents


def _checkpoint_name(step: int) -> str:
    return f"generator_step{step:04d}.pt"


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Optimizer, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    parameters = capacity._parameters(consumer)
    phase = parameters["generator.phase"]
    actual_step = int(float(optimizer.state[phase]["step"].item()))
    _require(actual_step == completed and len(history) == completed, "checkpoint update count or Adam step differs")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "source_task_id": PARENT_TASK_ID,
        "arm": ARM,
        "source_checkpoint": population["source_checkpoint"],
        "source_step": SOURCE_STEP,
        "additional_updates": completed - SOURCE_STEP,
        "completed_updates": completed,
        "optimizer_step": actual_step,
        "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": source["architecture_layout"],
        "population": dict(population),
        "trainable_parameter_state": capacity._snapshot_parameters(parameters),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": list(history),
        "source_history_count": SOURCE_STEP,
        "source_rng_state": source["rng_state"],
        "rng_state": capacity._rng_state(),
        "next_batch_index": completed % BATCH_COUNT,
        "next_batch_cursor": completed % BATCH_COUNT,
        "numeric_mode": dict(NUMERIC_MODE),
    }


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any]) -> Path:
    path = target / _checkpoint_name(completed)
    _require(not path.exists(), f"checkpoint exists and will not be overwritten: {path.name}")
    _write_torch(path, _checkpoint_payload(consumer, optimizer, completed, history, population, source), "cumulative large Generator Adam checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any], source: Mapping[str, Any], *, measurement_baseline: bool = False) -> dict[str, Any]:
    _require(path.resolve().parent == target.resolve() and path.is_file(), "resume requires an explicit checkpoint in this task output")
    value = torch.load(path, map_location="cpu", weights_only=True)
    step = value.get("completed_updates")
    _require(value.get("schema") == CHECKPOINT_SCHEMA and value.get("task_id") == TASK_ID, "resume checkpoint belongs to another task")
    _require(value.get("source_task_id") == PARENT_TASK_ID and value.get("source_checkpoint") == source["checkpoint_path"], "resume source lineage differs")
    _require(isinstance(step, int) and SOURCE_STEP <= step <= FINAL_STEP and (step == SOURCE_STEP or step % CHECKPOINT_INTERVAL == 0), "resume update count is not a saved task boundary")
    _require(path.name == _checkpoint_name(step) and value.get("additional_updates") == step - SOURCE_STEP, "resume checkpoint name or update count differs")
    _require(value.get("population") == dict(population), "resume population or roles changed")
    _require(value.get("optimizer_step") == step and value.get("optimizer_parameter_names") == ["generator.phase"], "resume Adam scope or step differs")
    _require(value.get("next_batch_index") == step % BATCH_COUNT and value.get("next_batch_cursor") == step % BATCH_COUNT, "resume batch cursor differs")
    _require(value.get("numeric_mode") == NUMERIC_MODE and value.get("architecture_layout") == source["architecture_layout"], "resume numeric mode or architecture differs")
    history = value.get("training_history")
    _require(isinstance(history, list) and len(history) == step and _tree_equal(history[:SOURCE_STEP], source["training_history"]), "resume history lacks the exact source trajectory")
    for update, record in enumerate(history[SOURCE_STEP:], start=SOURCE_STEP + 1):
        batch = (update - 1) % BATCH_COUNT
        positions = list(range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE))
        _require(record.get("update") == update and record.get("batch_index") == batch and record.get("fit_positions") == positions, f"resume history batch differs at {update}")
        _require(record.get("training_evaluators") == ["E_S", "E_I", "E_C"], f"resume history evaluator set differs at {update}")
    _require(set(value.get("trainable_parameter_state", {})) == {"encoder.phase", "generator.phase"}, "resume parameters are incomplete")
    for name, tensor in value["trainable_parameter_state"].items():
        _require(tuple(tensor.shape) == tuple(source["trainable_parameter_state"][name].shape) and tensor.dtype == source["trainable_parameter_state"][name].dtype and bool(torch.isfinite(tensor).all()), f"resume parameter state differs for {name}")
    slot = value["optimizer_state"]["state"][0]
    group = value["optimizer_state"]["param_groups"]
    _require(len(group) == 1 and group[0].get("params") == [0] and group[0].get("lr") == capacity.GENERATOR_LEARNING_RATE and tuple(group[0].get("betas", ())) == capacity.GENERATOR_BETAS and group[0].get("eps") == capacity.GENERATOR_EPS and group[0].get("weight_decay") == 0.0, "resume Adam hyperparameters differ")
    _require(set(value["optimizer_state"]["state"]) == {0}, "resume Adam contains an unexpected parameter state")
    _require(float(slot["step"].item()) == step and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "resume Adam state shape or step differs")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "resume Adam state is nonfinite")
    _require(isinstance(value.get("rng_state"), Mapping) and len(value["rng_state"]) >= 3, "resume RNG state is incomplete")
    _require(_tree_equal(value["frozen_runtime_state"], source["frozen_runtime_state"]), "resume changed frozen runtime state")
    _require(_tree_equal(value["references"], source["references"]) and _tree_equal(value["weights"], source["weights"]), "resume changed BOGT references or weights")
    if not measurement_baseline:
        later = [p.name for p in target.glob("generator_step*.pt") if (match := CHECKPOINT_RE.fullmatch(p.name)) and int(match.group(1)) > step]
        _require(not later, f"resume checkpoint predates saved progress: {sorted(later)}")
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


def _preflight(fit_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], paths: Mapping[str, Path], base_state: Mapping[str, Any], source: Mapping[str, Any], target: Path) -> dict[str, Any]:
    _EXECUTED_CHECKS.clear()
    checks: dict[str, Any] = {}
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device("cuda:0")
    _record_check(checks, "cuda_available", {"executed": True, "passed": True, "device": str(device)})
    disk = _disk_probe(target, _resolve(_root(), population["source_checkpoint"]))
    _record_check(checks, "disk_and_external_io", disk | {"executed": True, "passed": True})
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    new_target = _target_positive_control(evaluators, measure_inputs)
    target_payload = {
        "source_row_indices": measure_inputs["source_indices"][OLD_ROWS:].clone(),
        "labels": measure_inputs["labels"][OLD_ROWS:].clone(),
        **{f"{name.lower()}_logits": new_target["logits"][name] for name in EVALUATOR_NAMES},
    }
    target_path = target / "target_image_evaluator_positive_controls.pt"
    if target_path.exists():
        _require(_tree_equal(torch.load(target_path, map_location="cpu", weights_only=True), target_payload), "saved new confirmation target controls differ")
    else:
        _write_torch(target_path, target_payload, "four evaluator target-image controls on new confirmation rows")
    target_check = {"executed": True, "passed": new_target["passed"], "role": "new_confirmation", "evaluators": new_target["summary"]}
    _record_check(checks, "new_confirmation_target_positive_control", target_check)
    _require(new_target["passed"], "new TRAIN target positive control is below the predeclared evaluator qualification")
    trials: list[dict[str, Any]] = []
    frozen_report: dict[str, Any] = {}
    source_before = _cpu_tree(source)
    for trial_index in range(2):
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, ARM, device)
        encoder_before = params["encoder.phase"].detach().cpu().clone()
        runtime_before = capacity._frozen_runtime_state(consumer)
        evaluator_state = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        records = []
        for update_index in range(SOURCE_STEP, SOURCE_STEP + 3):
            record = ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, update_index, arm=ARM)
            records.append(record)
            _require(record["generator_adam_step"] == update_index + 1 and record["old_100_plane_gradient_norm"] > 0 and record["new_400_plane_gradient_norm"] > 0, "preflight did not update both phase supports")
            _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "preflight changed frozen Encoder")
            capacity._assert_state_equal(runtime_before, capacity._frozen_runtime_state(consumer), "preflight frozen OLS/runtime state")
            for name, model in evaluators.items():
                capacity._assert_state_equal(evaluator_state[name], capacity._state_snapshot(model), f"preflight evaluator {name}")
                _require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()), f"preflight evaluator {name} lost frozen isolation")
        trials.append({"records": records, "parameters": capacity._snapshot_parameters(params), "adam": _cpu_tree(optimizer.state_dict())})
        frozen_report = {"encoder_unchanged": True, "runtime_unchanged": True, "evaluators_unchanged": True, "encoder_and_ols_buffers_unchanged": True}
        del consumer, optimizer, params
        gc.collect()
        torch.cuda.empty_cache()
    source_unchanged = _tree_equal(source_before, source)
    _record_check(checks, "source_checkpoint_immutability", {"executed": True, "passed": source_unchanged, "source_adam_step": SOURCE_STEP})
    _require(source_unchanged, "preflight mutated source model or Adam state")
    aa_passed, aa_detail = _aa_repeatability(trials[0], trials[1])
    _record_check(checks, "three_step_production_AA_update", {"executed": True, "passed": aa_passed, "from_step": SOURCE_STEP, "to_step": SOURCE_STEP + 3, "repeatability": aa_detail, **frozen_report, "both_generator_phase_groups_updated": True})
    _require(aa_passed, "A/A repeated three-step trajectory exceeds the declared numeric tolerance")
    wrong = ablation._wrong_label_control(fit_inputs, base_state, source, evaluators)
    _record_check(checks, "wrong_label_ce_negative_control", {"executed": True, **wrong})
    with tempfile.TemporaryDirectory(prefix="lg_tenclass_resume_") as scratch:
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, ARM, device)
        history = [dict(row) for row in source["training_history"]]
        record = ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, SOURCE_STEP, arm=ARM)
        history.append(record)
        temp_payload = _checkpoint_payload(consumer, optimizer, SOURCE_STEP + 1, history, population, {**source, "checkpoint_path": population["source_checkpoint"]})
        scratch_file = Path(scratch) / "checkpoint.pt"
        torch.save(temp_payload, scratch_file)
        saved = torch.load(scratch_file, map_location="cpu", weights_only=True)
        continuous = [ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, index, arm=ARM) for index in (SOURCE_STEP + 1, SOURCE_STEP + 2)]
        continuous_state = {"parameters": capacity._snapshot_parameters(params), "adam": _cpu_tree(optimizer.state_dict()), "records": continuous}
        del consumer, optimizer, params
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, ARM, device)
        _restore_task_checkpoint(saved, consumer, optimizer)
        resumed = [ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, index, arm=ARM) for index in (SOURCE_STEP + 1, SOURCE_STEP + 2)]
        resumed_state = {"parameters": capacity._snapshot_parameters(params), "adam": _cpu_tree(optimizer.state_dict()), "records": resumed}
        resume_passed, resume_detail = _aa_repeatability(continuous_state, resumed_state)
        _record_check(checks, "interrupted_resume_equivalence", {"executed": True, "passed": resume_passed, "saved_step": SOURCE_STEP + 1, "next_batch_index": (SOURCE_STEP + 1) % BATCH_COUNT, "final_step": SOURCE_STEP + 3, "repeatability": resume_detail})
        _require(resume_passed, "continuous and resumed trajectory exceeds the declared numeric tolerance")
        del consumer, optimizer, params
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight final {name} evaluator")
    _record_check(checks, "source_checkpoint_immutability_final", {"executed": True, "passed": _tree_equal(source_before, source), "source_adam_step": SOURCE_STEP})
    _require(checks["source_checkpoint_immutability_final"]["passed"], "preflight changed source checkpoint after later checks")
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "preflight is missing a passed check")
    return {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "READY", "device": str(device), "numeric_mode": NUMERIC_MODE, "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates": ADDED_UPDATES, "checks": checks, "duration_seconds_observation_only": time.perf_counter() - _STARTED}


def _make_run_control(population: Mapping[str, Any], source: Mapping[str, Any], preflight: Mapping[str, Any], fit_inputs: Mapping[str, Any], base_state: Mapping[str, Any]):
    consumer, optimizer, params = capacity._new_endpoint(fit_inputs, base_state, "B_large", torch.device("cpu"))
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-large-generator-three-evaluator-1700-to-3000",
        semantic_authority_id=PARENT_TASK_ID,
        data_role="canonical TRAIN: fit=200 updated; historic=100 and prior/new confirmation=600 excluded from gradients",
        config_identity={"grid": 2048, "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "added_updates": ADDED_UPDATES, "batch_size": BATCH_SIZE, "cycles": ADDED_UPDATES // BATCH_COUNT, "numeric_mode": NUMERIC_MODE, "training_evaluators": ["E_S", "E_I", "E_C"]},
        model_identity={"checkpoint": population["source_checkpoint"], "parameter_shapes": {key: list(value.shape) for key, value in source["trainable_parameter_state"].items()}, "architecture": source["architecture_layout"]},
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": capacity.GENERATOR_LEARNING_RATE, "betas": list(capacity.GENERATOR_BETAS), "eps": capacity.GENERATOR_EPS, "weight_decay": 0.0, "reset": False},
        sampler_identity={"fit_rows": population["source_row_indices"][:FIT_ROWS], "new_confirmation_rows": population["source_row_indices"][OLD_ROWS:], "confirmation_seed": CONFIRMATION_SEED, "fit_batch_size": BATCH_SIZE, "fit_batch_count": BATCH_COUNT},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )
    del consumer, optimizer, params
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01")
    checks = preflight["checks"]
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY receipt cannot include missing or unpassed checks")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "source_checkpoint_and_population": True, "new_confirmation_target_control": checks["new_confirmation_target_positive_control"]["passed"], "three_real_AA_updates": checks["three_step_production_AA_update"]["passed"], "wrong_label_negative_control": checks["wrong_label_ce_negative_control"]["passed"], "resume_equivalence": checks["interrupted_resume_equivalence"]["passed"]},
        resource={"passed": checks["disk_and_external_io"]["passed"], "status": "PASS", "cuda_available": True, "disk_and_io": checks["disk_and_external_io"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_generator_only_1700_to_3000"},
        selected_engine="cuda_2048_generator_only_1700_to_3000",
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
            _write_json(population_path, population, "fixed 900-row canonical TRAIN roles")
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
        actual_summary[name] = {"per_class_correct": correct, "macro_recall": macro, "passed": macro >= 0.90 and correct[1] >= 24 and correct[7] >= 24}
    saved_check = preflight.get("checks", {}).get("new_confirmation_target_positive_control", {})
    _require(saved_check.get("executed") is True and saved_check.get("passed") is True, "READY receipt does not contain a passed target positive-control check")
    _require(actual_summary == saved_check.get("evaluators"), "saved target positive-control logits do not reproduce READY qualification")
    return {"executed": True, "passed": True, "role": "new_confirmation", "evaluators": actual_summary}


def _consume_saved_ready(target: Path, population: Mapping[str, Any], measure_inputs: Mapping[str, Any], preflight: Mapping[str, Any], control: ResearchRunControl, expected: PreflightReceipt) -> PreflightReceipt:
    ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY" and ready.get("preflight") == preflight, "saved preflight evidence is missing or differs")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == dict(population), "saved population differs from the READY attempt")
    saved = PreflightReceipt.from_mapping(ready.get("receipt", {}))
    _require(saved.source_semantics_id == TASK_ID and saved.authorization_id == expected.authorization_id and saved.status == "READY", "saved READY receipt is bound to another task")
    _require(all(value.get("executed") is True and value.get("passed") is True for value in preflight["checks"].values()), "saved READY receipt omits an executed passing check")
    _validate_saved_target_positive_control(target, measure_inputs, population, preflight)
    control.mark_preflight_ready(saved)
    return saved


def _training_loop(target: Path, fit_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module], start_from: Mapping[str, Any] | None) -> dict[str, Any]:
    consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, ARM, torch.device("cuda:0"))
    if start_from is not None:
        _restore_task_checkpoint(start_from, consumer, optimizer)
    else:
        start_from = _checkpoint_payload(consumer, optimizer, SOURCE_STEP, source["training_history"], population, {**source, "checkpoint_path": population["source_checkpoint"]})
    completed = int(start_from["completed_updates"])
    history = list(start_from["training_history"])
    _require(SOURCE_STEP <= completed <= FINAL_STEP and len(history) == completed, "training start checkpoint count differs")
    checkpoint_path = target / _checkpoint_name(completed)
    if not checkpoint_path.exists():
        _save_checkpoint(target, consumer, optimizer, completed, history, population, {**source, "checkpoint_path": population["source_checkpoint"]})
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
            record = ablation._train_step(consumer, optimizer, params, evaluators, fit_inputs, completed, arm=ARM)
            completed += 1
            history.append(record)
            if completed % CHECKPOINT_INTERVAL == 0 or stop_requested:
                path = _save_checkpoint(target, consumer, optimizer, completed, history, population, {**source, "checkpoint_path": population["source_checkpoint"]})
                _resource(target, "checkpoint", completed)
                _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed frozen Encoder")
                capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "training frozen OLS/runtime state")
                for name, model in evaluators.items():
                    capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"training evaluator {name}")
                    _require(all(param.grad is None and not param.requires_grad for param in model.parameters()), f"training evaluator {name} isolation changed")
                gc.collect()
                torch.cuda.empty_cache()
                if stop_requested:
                    raise TrainingInterrupted(f"saved checkpoint {path.name} at a completed update boundary")
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    _require(completed == FINAL_STEP and len(history) == FINAL_STEP, "training did not finish exactly at 3000 updates")
    _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed Encoder at completion")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "frozen runtime at completion")
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"evaluator {name} at completion")
    final_path = target / _checkpoint_name(FINAL_STEP)
    if not final_path.exists():
        _save_checkpoint(target, consumer, optimizer, FINAL_STEP, history, population, {**source, "checkpoint_path": population["source_checkpoint"]})
    history_payload = {"task_id": TASK_ID, "arm": ARM, "source_step": SOURCE_STEP, "additional_updates": ADDED_UPDATES, "final_step": FINAL_STEP, "training_evaluators": ["E_S", "E_I", "E_C"], "records": history}
    history_path = target / "training_history_A_three_evaluator.json"
    if history_path.exists():
        _require(json.loads(history_path.read_text(encoding="utf-8")) == history_payload, "saved final training history differs")
    else:
        _write_json(history_path, history_payload, "source plus 1300 additional Generator updates")
    result = torch.load(final_path, map_location="cpu", weights_only=True)
    del consumer, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _endpoint(target: Path, name: str, checkpoint: Mapping[str, Any], fit_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, torch.nn.Module]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = target / f"readouts_{name}.pt"
    if path.is_file():
        readouts = torch.load(path, map_location="cpu", weights_only=True)
        _require(torch.equal(readouts["source_indices"], measure_inputs["source_indices"]) and torch.equal(readouts["labels"], measure_inputs["labels"]), f"saved {name} readout population differs")
    else:
        consumer, optimizer, params = ablation._load_arm(fit_inputs, base_state, source, ARM, torch.device("cuda:0"))
        _restore_task_checkpoint(checkpoint, consumer, optimizer)
        param_before = capacity._snapshot_parameters(params)
        adam_before = _cpu_tree(optimizer.state_dict())
        frozen_before = capacity._frozen_runtime_state(consumer)
        readouts = _measure_forward(consumer, evaluators, measure_inputs)
        _require(_tree_equal(param_before, capacity._snapshot_parameters(params)), f"{name} forward changed model parameters")
        _require(_tree_equal(adam_before, _cpu_tree(optimizer.state_dict())), f"{name} forward changed Adam state")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{name} forward changed frozen runtime")
        _write_torch(path, readouts, f"900-row frozen production readout {name}")
        del consumer, optimizer, params
        gc.collect()
        torch.cuda.empty_cache()
    summary, rows = _row_and_image_metrics(readouts, population)
    metric_path = target / f"metrics_{name}.json"
    row_path = target / f"row_metrics_{name}.pt"
    if metric_path.exists():
        _require(json.loads(metric_path.read_text(encoding="utf-8")) == summary, f"saved {name} metrics differ")
    else:
        _write_json(metric_path, summary, "900-row class and image measurements")
    if row_path.exists():
        _require(_tree_equal(torch.load(row_path, map_location="cpu", weights_only=True), rows), f"saved {name} row metrics differ")
    else:
        _write_torch(row_path, rows, "900-row float64 per-row scoring and image metrics")
    return summary, readouts


def _main(args: argparse.Namespace) -> dict[str, Any]:
    global _STARTED
    _STARTED = time.perf_counter()
    root = _root()
    parent_dir = _resolve(root, args.ablation_dir)
    target = _resolve(root, args.output_dir)
    _require(bool(args.resume) == (args.checkpoint is not None), "--resume requires an explicit --checkpoint")
    _require(not (args.preflight_only and args.resume), "--preflight-only cannot be combined with resume")
    fit_inputs, measure_inputs, population, paths, base_state, source, source_checkpoints, parent_readouts, source_report = _population(root, parent_dir)
    created_target = _prepare_target(root, target, resume=args.resume)
    source["checkpoint_path"] = population["source_checkpoint"]
    if args.resume:
        _require((target / "input_population.json").is_file() and (target / "preflight.json").is_file(), "resume lacks task population or READY receipt")
        _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == population, "resume population differs")
        ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
        _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY", "resume READY receipt is missing")
        preflight = ready["preflight"]
        control, expected_receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
        receipt = _consume_saved_ready(target, population, measure_inputs, preflight, control, expected_receipt)
        selected = _read_checkpoint(_resolve(root, args.checkpoint), target, population, source)
        report_path = target / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _require(selected["completed_updates"] == FINAL_STEP and report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "terminal recovery requires the explicit final checkpoint")
            return report
    else:
        staged = not created_target
        if staged:
            ready_stage_files = {"input_population.json", "preflight.json", "target_image_evaluator_positive_controls.pt"}
            _require({path.name for path in target.iterdir()} == ready_stage_files, "existing output is not an untouched READY stage")
            _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == population, "staged population differs")
            ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
            preflight = ready["preflight"]
            control, expected_receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
            receipt = _consume_saved_ready(target, population, measure_inputs, preflight, control, expected_receipt)
        else:
            _write_json(target / "input_population.json", population, "fixed canonical TRAIN population and role mapping")
            with capacity.strict_fp32_mode() as mode:
                _require(mode == NUMERIC_MODE, "active preflight numeric mode differs from strict FP32")
                preflight = _preflight(fit_inputs, measure_inputs, population, paths, base_state, source, target)
            control, receipt = _make_run_control(population, source, preflight, fit_inputs, base_state)
            _ready_files(target, population, preflight, receipt, resume=False)
            selected = None
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}
    control.mark_preflight_ready(receipt)
    control.start_execution()
    control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == NUMERIC_MODE, "active scientific execution numeric mode differs from strict FP32")
        evaluators = diagnostic.load_evaluators(paths, torch.device("cuda:0"))
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        if not args.resume:
            _write_torch(target / "source_checkpoint_A_step1700.pt", source, "immutable source A1700 checkpoint copy")
            start_checkpoint_path = target / _checkpoint_name(SOURCE_STEP)
            if start_checkpoint_path.exists():
                start_checkpoint = _read_checkpoint(start_checkpoint_path, target, population, source)
            else:
                consumer, optimizer, _params = ablation._load_arm(fit_inputs, base_state, source, ARM, torch.device("cuda:0"))
                saved_path = _save_checkpoint(target, consumer, optimizer, SOURCE_STEP, source["training_history"], population, source)
                start_checkpoint = torch.load(saved_path, map_location="cpu", weights_only=True)
                del consumer, optimizer, _params
            checkpoint = _training_loop(target, fit_inputs, population, base_state, source, evaluators, start_checkpoint)
        else:
            checkpoint = _training_loop(target, fit_inputs, population, base_state, source, evaluators, selected)
        _require(checkpoint.get("completed_updates") == FINAL_STEP, "final checkpoint does not end at step 3000")
        start_task_checkpoint = _read_checkpoint(target / _checkpoint_name(SOURCE_STEP), target, population, source, measurement_baseline=True)
        start_summary, start_readout = _endpoint(target, "A_step1700", start_task_checkpoint, fit_inputs, measure_inputs, population, base_state, source, evaluators)
        final_summary, final_readout = _endpoint(target, "A_step3000", checkpoint, fit_inputs, measure_inputs, population, base_state, source, evaluators)
    replay = _source_readout_replay(start_readout, parent_readouts["A_step1700"])
    decision = _final_decision(start_summary, final_summary, valid=True)
    role_transitions = {}
    for role, bounds in population["role_positions"].items():
        a, b = map(int, bounds)
        y = measure_inputs["labels"][a:b]
        p0 = start_readout["e_d_logits"][a:b].to(torch.float64).argmax(1)
        p1 = final_readout["e_d_logits"][a:b].to(torch.float64).argmax(1)
        role_transitions[role] = {str(c): {"correct_to_wrong": int(((y == c) & (p0 == c) & (p1 != c)).sum()), "wrong_to_correct": int(((y == c) & (p0 != c) & (p1 == c)).sum())} for c in range(10)}
    images = _comparison_pages(target, measure_inputs, start_readout, final_readout, population)
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final evaluator {name}")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": decision["status"],
        "classification": decision["classification"],
        "scientific_updates": ADDED_UPDATES,
        "parent": {"task_id": PARENT_TASK_ID, "status": source_report["status"], "classification": source_report["classification"], "source_checkpoint": population["source_checkpoint"]},
        "scope": {"source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates": ADDED_UPDATES, "fit_rows": FIT_ROWS, "new_confirmation_rows": NEW_CONFIRMATION_ROWS, "all_rows_canonical_train": True, "new_confirmation_unseen_claim": False, "updated_parameter": "generator.phase", "encoder_ols_geometry_buffers_evaluators_frozen": True, "sampler_cycles": ADDED_UPDATES // BATCH_COUNT, "training_evaluators": ["E_S", "E_I", "E_C"], "numeric_mode": NUMERIC_MODE},
        "source_readout_replay": replay,
        "endpoint_metrics": {"A_step1700": start_summary, "A_step3000": final_summary},
        "paired_correctness_transitions": role_transitions,
        **decision,
        "readout_artifacts": {"A_step1700": "readouts_A_step1700.pt", "A_step3000": "readouts_A_step3000.pt"},
        "row_metric_artifacts": {"A_step1700": "row_metrics_A_step1700.pt", "A_step3000": "row_metrics_A_step3000.pt"},
        "comparison_images": images,
        "checkpoint_artifacts": sorted(path.name for path in target.glob("generator_step*.pt")),
        "training_history_artifact": "training_history_A_three_evaluator.json",
        "preflight_artifact": "preflight.json",
        "elapsed_seconds_observation_only": time.perf_counter() - _STARTED,
        "next_action": "NONE",
        "stop": True,
    }
    control.complete() if decision["status"] == "VALID_COMPLETED" else control.valid_failed()
    report["research_run_state"] = control.state
    _write_json(target / "report.json", report, "single 3000-step ten-class presentation decision")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_PARENT)
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
                int(match.group(1))
                for candidate in target.glob("generator_step*.pt")
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
