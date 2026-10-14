"""Compare a 200-row fit bank with a 2,000-row bank at equal LightGen update budget."""

from __future__ import annotations

import argparse
import gc
import io
import json
import re
import signal
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9
from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as ablation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity


TASK_ID = "M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1"
PARENT_TASK_ID = class9.TASK_ID
REPORT_SCHEMA = "m0_generator_training_population_expansion_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_training_population_expansion_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_training_population_expansion_checkpoint.v1"
DEFAULT_PARENT = Path("outputs/20261008_m0_generator_class9_weighted_ce_comparison_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_training_population_expansion_comparison_v1/runs/attempt_01")
SOURCE_STEP, FINAL_STEP, ADDED_UPDATES = 3700, 4700, 1000
BASE_FIT_ROWS, HISTORIC_ROWS, EXPANDED_PER_CLASS = 200, 100, 180
EXPANDED_ROWS, CONFIRMATION_ROWS = 1800, 300
SOURCE_ROWS, TOTAL_ROWS = 1200, 3300
BATCH_SIZE, A_BATCH_COUNT, B_BATCH_COUNT, CHECKPOINT_INTERVAL = 10, 20, 200, 100
EXPANSION_SEED, CONFIRMATION_SEED = 2026100811, 2026100812
RTOL, ATOL = ablation.RTOL, ablation.ATOL
NUMERIC_MODE = dict(class9.NUMERIC_MODE)
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
ARM_A, ARM_B = "A_repeat_200", "B_expand_2000"
ARM_NAMES = (ARM_A, ARM_B)
CHECKPOINT_RE = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_NAME = "resource_telemetry_population_expansion.jsonl"


class PopulationExpansionError(RuntimeError):
    """Invalid source, population, measurement, or training invariant."""


class TrainingInterrupted(RuntimeError):
    """A signal was recorded after an immutable recovery checkpoint was saved."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise PopulationExpansionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _cpu_tree(value: Any) -> Any:
    return capacity._cpu_tree(value)


def _write_json(path: Path, value: Mapping[str, Any], kind: str) -> None:
    ablation._write_json_new(path, value, kind)


def _write_torch(path: Path, value: Mapping[str, Any], kind: str) -> None:
    ablation._write_torch_new(path, value, kind)


def _select_rows(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    excluded: torch.Tensor,
    per_class: int,
    seed: int,
) -> torch.Tensor:
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "canonical MNIST label metadata is invalid")
    _require(torch.unique(excluded).numel() == excluded.numel(), "excluded TRAIN row list contains duplicates")
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    excluded_mask = torch.zeros(60_000, dtype=torch.bool)
    excluded_mask[excluded.cpu()] = True
    _require(bool(train_mask.index_select(0, excluded.cpu()).all()), "a previously assigned row is outside canonical TRAIN")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class: list[torch.Tensor] = []
    for class_id in range(10):
        candidates = torch.nonzero(train_mask & ~excluded_mask & (labels == class_id), as_tuple=False).flatten().sort().values
        _require(candidates.numel() >= per_class, f"class {class_id} has too few unassigned canonical TRAIN rows")
        selected = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator)[:per_class])
        by_class.append(selected)
        excluded_mask[selected] = True
    selected = torch.stack([torch.stack([by_class[c][i] for c in range(10)]) for i in range(per_class)]).reshape(-1)
    _require(selected.numel() == 10 * per_class and torch.unique(selected).numel() == selected.numel(), "sampled rows are not unique and class-balanced")
    _require(bool(train_mask.index_select(0, selected).all()), "sampled population contains a non-TRAIN row")
    return selected


def _role_partition() -> tuple[dict[str, list[int]], dict[str, int], list[str]]:
    """Name every source and newly added row interval exactly once."""
    role_positions = {
        "fit": [0, BASE_FIT_ROWS],
        "historic": [BASE_FIT_ROWS, BASE_FIT_ROWS + HISTORIC_ROWS],
        "ce_confirmation": [300, 600],
        "presentation_confirmation": [600, 900],
        "class9_confirmation": [900, SOURCE_ROWS],
        "expanded_training": [SOURCE_ROWS, SOURCE_ROWS + EXPANDED_ROWS],
        "coverage_confirmation": [SOURCE_ROWS + EXPANDED_ROWS, TOTAL_ROWS],
    }
    roles = [""] * TOTAL_ROWS
    role_counts: dict[str, int] = {}
    for role, (start, stop) in role_positions.items():
        _require(0 <= start < stop <= TOTAL_ROWS, f"population role interval is invalid: {role}")
        _require(all(not roles[position] for position in range(start, stop)), f"population role overlaps: {role}")
        roles[start:stop] = [role] * (stop - start)
        role_counts[role] = stop - start
    _require(all(roles), "population role intervals leave rows unassigned")
    _require(sum(role_counts.values()) == TOTAL_ROWS, "population role counts do not cover all rows")
    return role_positions, role_counts, roles


def _validate_role_partition(population: Mapping[str, Any]) -> None:
    expected_positions, expected_counts, expected_roles = _role_partition()
    _require(population.get("role_positions") == expected_positions, "population role intervals differ from the seven fixed row ranges")
    _require(population.get("role_counts") == expected_counts, "population role counts do not cover all 3,300 rows")
    _require(population.get("roles") == expected_roles, "per-row roles differ from the fixed interval partition")


def _role_page_positions(population: Mapping[str, Any], *, rows_per_page: int = 30) -> dict[str, list[list[int]]]:
    _validate_role_partition(population)
    _require(rows_per_page > 0, "comparison page size must be positive")
    pages: dict[str, list[list[int]]] = {}
    for role, (start, stop) in population["role_positions"].items():
        pages[role] = [list(range(offset, min(offset + rows_per_page, stop))) for offset in range(start, stop, rows_per_page)]
        _require([position for page in pages[role] for position in page] == list(range(start, stop)), f"comparison pages omit or duplicate a {role} row")
    _require(sum(len(page) for role_pages in pages.values() for page in role_pages) == TOTAL_ROWS, "comparison pages do not cover all 3,300 rows")
    return pages


def _population(root: Path, parent_dir: Path):
    # The task directory supplies the A3700 checkpoint; its declared presentation
    # ancestor supplies the already validated optics, population, and evaluators.
    parent_population_path, parent_report_path = parent_dir / "input_population.json", parent_dir / "report.json"
    _require(parent_population_path.is_file() and parent_report_path.is_file(), "explicit class9 source population/report is missing")
    parent_population = json.loads(parent_population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    _require(parent_population.get("task_id") == PARENT_TASK_ID and parent_report.get("task_id") == PARENT_TASK_ID, "named class9 source identity differs")
    _require(parent_report.get("status") == "VALID_FAILED" and parent_report.get("classification") == "MIXED_OR_INSUFFICIENT", "named class9 source result differs")
    presentation_value = parent_population.get("source_presentation_dir")
    _require(isinstance(presentation_value, str) and presentation_value, "class9 source does not name its presentation lineage")
    presentation_dir = _resolve(root, presentation_value)
    model_inputs, _old_measure, old_population, paths, base_state, _step3000, _checkpoints, _parent_readouts, _presentation_report = class9._population(root, presentation_dir)
    expected_paths = parent_population.get("source_input_paths", {})
    _require({name: path.as_posix() for name, path in paths.items()} == expected_paths, "class9 source input paths differ from its declared lineage")
    source_path = parent_dir / "arm_a_step3700.pt"
    source_readout_path = parent_dir / "readouts_A_step3700.pt"
    _require(source_path.is_file() and source_readout_path.is_file(), "explicit original-rule A3700 source checkpoint/readout is missing")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    source_readout = torch.load(source_readout_path, map_location="cpu", weights_only=True)
    _require(source.get("task_id") == PARENT_TASK_ID and source.get("arm") == "A_original", "source checkpoint is not the class9 task's original-rule arm")
    _require(source.get("schema") == class9.CHECKPOINT_SCHEMA and source.get("completed_updates") == SOURCE_STEP and source.get("optimizer_step") == SOURCE_STEP, "source Generator or Adam step is not 3700")
    _require(source.get("next_batch_index") == 0 and source.get("next_batch_cursor") == 0, "source A3700 next-batch cursor is not zero")
    _require(source.get("optimizer_parameter_names") == ["generator.phase"] and source.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "source architecture or optimizer scope differs")
    parameters = source.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping) and tuple(parameters["generator.phase"].shape) == (340_000,) and tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100), "source phase layout differs")
    _require(all(bool(torch.isfinite(value).all()) for value in parameters.values()), "source phase state is nonfinite")
    adam = source["optimizer_state"]
    _require(len(adam["param_groups"]) == 1 and adam["param_groups"][0]["params"] == [0] and set(adam["state"]) == {0}, "source Generator Adam mapping differs")
    group = adam["param_groups"][0]
    _require(group["lr"] == capacity.GENERATOR_LEARNING_RATE and tuple(group["betas"]) == capacity.GENERATOR_BETAS and group["eps"] == capacity.GENERATOR_EPS and group["weight_decay"] == 0.0, "source Adam hyperparameters differ")
    slot = adam["state"][0]
    _require(int(float(slot["step"].item())) == SOURCE_STEP and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "source Adam moments/step differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "source Adam moments are nonfinite")
    _require(source.get("numeric_mode") == NUMERIC_MODE and len(source.get("training_history", [])) == SOURCE_STEP, "source strict FP32 mode or history differs")
    _require(torch.equal(source_readout["source_indices"], torch.tensor(old_population["source_row_indices"])) and torch.equal(source_readout["labels"], torch.tensor(old_population["labels"])), "A3700 readout row mapping differs")
    _require(_tree_equal(source.get("references"), base_state.get("references")) and _tree_equal(source.get("weights"), base_state.get("weights")), "A3700 BOGT references/weights differ from the source lineage")
    _require(all(torch.isfinite(value.real if value.is_complex() else value).all() for value in source_readout.values() if torch.is_tensor(value)), "A3700 source readout contains a nonfinite tensor")
    source_metrics, _ = class9._row_and_image_metrics(source_readout, old_population)
    _require(source_metrics == parent_report["endpoint_metrics"]["A_step3700"], "A3700 saved readout does not reproduce its source report")

    image_store = diagnostic._load_torch(paths["training_data"], mmap=True)
    image_rows, labels = diagnostic._read_image_labels(image_store)
    train_indices, _ = diagnostic.stratified_mnist_indices(labels, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    prior_rows = torch.tensor(old_population["source_row_indices"], dtype=torch.long)
    _require(torch.equal(labels.index_select(0, prior_rows), torch.tensor(old_population["labels"])), "source labels differ from canonical TRAIN metadata")
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, prior_rows).all()), "one or more source rows are outside canonical TRAIN")
    for update, record in enumerate(source["training_history"], start=1):
        batch_index = (update - 1) % A_BATCH_COUNT
        positions = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        _require(record.get("update") == update and record.get("batch_index") == batch_index and record.get("fit_positions") == positions, f"A3700 source history batch differs at update {update}")
        _require(record.get("source_row_indices") == [old_population["source_row_indices"][position] for position in positions] and record.get("labels") == [old_population["labels"][position] for position in positions], f"A3700 source history rows differ at update {update}")
    expanded_rows = _select_rows(labels, train_indices, prior_rows, EXPANDED_PER_CLASS, EXPANSION_SEED)
    confirmation_excluded = torch.cat((prior_rows, expanded_rows))
    confirmation_rows = _select_rows(labels, train_indices, confirmation_excluded, 30, CONFIRMATION_SEED)
    all_rows = torch.cat((prior_rows, expanded_rows, confirmation_rows))
    all_labels = labels.index_select(0, all_rows).contiguous()
    _require(all_rows.numel() == TOTAL_ROWS and torch.unique(all_rows).numel() == TOTAL_ROWS, "3,300-row population overlaps or is incomplete")
    _require(torch.bincount(all_labels[-CONFIRMATION_ROWS:], minlength=10).tolist() == [30] * 10, "new confirmation rows are not balanced")

    # The named source already materialized only its 1,200 rows; materialize only the 2,100 new rows here.
    new_images = image_rows.index_select(0, torch.cat((expanded_rows, confirmation_rows))).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    images = torch.cat((_old_measure["images"], new_images), dim=0)
    _require(tuple(images.shape) == (TOTAL_ROWS, 1, 28, 28) and bool(torch.isfinite(images).all()) and float(images.min()) >= 0 and float(images.max()) <= 1, "fixed 3,300-row image tensor is invalid")
    role_positions, role_counts, roles = _role_partition()
    measure_inputs = {"images": images, "source_indices": all_rows, "labels": all_labels}
    fit_measure_positions = torch.cat((torch.arange(BASE_FIT_ROWS), torch.arange(SOURCE_ROWS, SOURCE_ROWS + EXPANDED_ROWS)))
    train_inputs = {
        "images": images.index_select(0, fit_measure_positions).contiguous(),
        "labels": all_labels.index_select(0, fit_measure_positions).contiguous(),
        "source_indices": all_rows.index_select(0, fit_measure_positions).contiguous(),
        "fit_measure_positions": fit_measure_positions,
    }
    _require(torch.bincount(train_inputs["labels"], minlength=10).tolist() == [200] * 10, "expanded training population does not have 200 rows per class")
    population = {
        "task_id": TASK_ID,
        "source_class9_dir": parent_dir.as_posix(),
        "source_class9_status": parent_report["status"],
        "source_checkpoint": source_path.as_posix(),
        "source_step": SOURCE_STEP,
        "source_generator_adam_step": SOURCE_STEP,
        "source_parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
        "source_input_paths": {name: value.as_posix() for name, value in paths.items()},
        "source_row_indices": all_rows.tolist(),
        "labels": all_labels.tolist(),
        "roles": roles,
        "role_counts": role_counts,
        "role_positions": role_positions,
        "fit_mapping_A_measure_positions": list(range(200)),
        "fit_mapping_B_measure_positions": fit_measure_positions.tolist(),
        "fit_mapping_B_training_positions": list(range(2000)),
        "expanded_per_class": EXPANDED_PER_CLASS,
        "expansion_seed": EXPANSION_SEED,
        "confirmation_seed": CONFIRMATION_SEED,
        "updates_from_source": SOURCE_STEP,
        "additional_updates_per_arm": ADDED_UPDATES,
        "final_generator_adam_step": FINAL_STEP,
        "batch_size": BATCH_SIZE,
        "batch_count_A": A_BATCH_COUNT,
        "batch_count_B": B_BATCH_COUNT,
        "A_row_update_count": ADDED_UPDATES // A_BATCH_COUNT,
        "B_row_update_count": ADDED_UPDATES // B_BATCH_COUNT,
        "all_rows_canonical_train_not_claimed_unseen": True,
        "numeric_mode": NUMERIC_MODE,
        "training_evaluators": ["E_S", "E_I", "E_C"],
    }
    source = dict(source)
    source["checkpoint_path"] = source_path.as_posix()
    source["source_task_id"] = PARENT_TASK_ID
    _validate_role_partition(population)
    _require(torch.equal(source_readout["source_indices"], all_rows[:SOURCE_ROWS]), "source readout and preserved first 1,200-row order differ")
    return model_inputs, train_inputs, measure_inputs, population, paths, base_state, source, source_readout, parent_report


def _target_positive_control(evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    start = SOURCE_ROWS + EXPANDED_ROWS
    images = inputs["images"][start:].to(torch.device("cuda:0"))
    labels = inputs["labels"][start:].cpu()
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_NAMES}
    with torch.no_grad():
        for offset in range(0, CONFIRMATION_ROWS, BATCH_SIZE):
            view = capacity.canonical.cmos_view(images[offset : offset + BATCH_SIZE])
            for name in EVALUATOR_NAMES:
                logits[name].append(capacity._evaluator_logits(evaluators[name], view, name).detach().cpu())
    result = {}
    for name in EVALUATOR_NAMES:
        values = torch.cat(logits[name]).to(torch.float64)
        prediction = values.argmax(1)
        correct = [int(((labels == c) & (prediction == c)).sum()) for c in range(10)]
        macro = sum(correct) / CONFIRMATION_ROWS
        result[name] = {"per_class_correct": correct, "macro_recall": macro, "passed": macro >= .90 and all(correct[c] >= 24 for c in (1, 5, 7, 9))}
    return {"summary": result, "logits": {name: torch.cat(values) for name, values in logits.items()}, "passed": all(value["passed"] for value in result.values())}


def _source_replay(readout: Mapping[str, Any], source_readout: Mapping[str, Any]) -> dict[str, Any]:
    details = {}
    for name in ("z", "raw_cmos", "evaluator_view", "target_slm_intensity", "power_by_plane", "new_phase_plane_power"):
        left, right = readout[name][:SOURCE_ROWS], source_readout[name]
        _require(left.shape == right.shape and torch.allclose(left, right, rtol=RTOL, atol=ATOL), f"A3700 source replay differs for {name}")
        diff = (left.to(torch.complex128 if left.is_complex() else torch.float64) - right.to(torch.complex128 if right.is_complex() else torch.float64)).abs()
        details[name] = {"passed": True, "max_abs_difference": float(diff.max())}
    for name in EVALUATOR_NAMES:
        key = f"{name.lower()}_logits"
        left, right = readout[key][:SOURCE_ROWS], source_readout[key]
        _require(left.shape == right.shape and torch.allclose(left, right, rtol=RTOL, atol=ATOL), f"A3700 source replay differs for {key}")
        details[key] = {"passed": True, "max_abs_difference": float((left.double() - right.double()).abs().max())}
    return {"passed": True, "rtol": RTOL, "atol": ATOL, "fields": details}


def _decision(start: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], valid: bool) -> dict[str, Any]:
    role = "coverage_confirmation"
    metrics = [item["evaluators"]["E_D"]["by_role"][role] for item in (start, a, b)]
    image_metrics = [item["image_metrics_by_role"][role] for item in (start, a, b)]
    counts = [list(item["per_class_correct"]) for item in metrics]
    totals = [int(item["correct_total"]) for item in metrics]
    macros = [float(item["macro_recall"]) for item in metrics]
    base_image, a_image, b_image = image_metrics
    conditions = {
        "B_at_least_225_and_5_points_above_A_and_not_below_start": totals[2] >= 225 and totals[2] >= totals[1] + 15 and totals[2] >= totals[0],
        "every_class_at_least_15": min(counts[2]) >= 15,
        "classes_1_and_7_not_below_start": counts[2][1] >= counts[0][1] and counts[2][7] >= counts[0][7],
        "no_class_loses_more_than_3_vs_A_or_start": all(counts[2][c] >= counts[1][c] - 3 and counts[2][c] >= counts[0][c] - 3 for c in range(10)),
        "raw_mse_clipping_and_image_integrity": b_image["raw_mse"] <= 1.05 * a_image["raw_mse"] and b_image["raw_mse"] <= 1.05 * base_image["raw_mse"] and b_image["raw_fraction_above_one"] <= .01 and all(b_image["by_class"][str(c)]["raw_fraction_above_one"] <= .01 for c in (1, 5, 7, 9)) and b_image["raw_zero_image"] == 0 and b_image["raw_constant_image"] == 0 and b.get("all_finite") is True,
        "budget_roles_freeze_and_measurement_valid": valid,
    }
    if not valid:
        status, classification = "INVALID_HARNESS", "INVALID_HARNESS"
    elif all(conditions.values()):
        status, classification = "VALID_COMPLETED", "TRAINING_POPULATION_EXPANSION_GAIN_SUPPORTED"
    elif totals[2] > totals[1] and totals[2] > totals[0]:
        status, classification = "VALID_FAILED", "MIXED_OR_INSUFFICIENT"
    else:
        status, classification = "VALID_FAILED", "NOT_SUPPORTED_WITHIN_1000_UPDATES"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "new_confirmation": {
            "correct_total": {"A3700": totals[0], "A4700": totals[1], "B4700": totals[2]},
            "macro_recall": {"A3700": macros[0], "A4700": macros[1], "B4700": macros[2]},
            "per_class_correct": {"A3700": counts[0], "A4700": counts[1], "B4700": counts[2]},
            "B_delta_vs_A": [counts[2][c] - counts[1][c] for c in range(10)],
            "B_delta_vs_start": [counts[2][c] - counts[0][c] for c in range(10)],
            "raw_mse": {"A3700": base_image["raw_mse"], "A4700": a_image["raw_mse"], "B4700": b_image["raw_mse"]},
        },
        "training_recommendation": "CONSIDER_EXPANDED_TRAINING_POPULATION_IN_SEPARATE_STUDY" if classification == "TRAINING_POPULATION_EXPANSION_GAIN_SUPPORTED" else "NONE",
        "next_action": "NONE",
        "stop": True,
    }


def _batch_indices(arm: str, local_update: int) -> tuple[int, int]:
    count = A_BATCH_COUNT if arm == ARM_A else B_BATCH_COUNT
    _require(arm in ARM_NAMES and local_update >= 0, "arm or task-local update is invalid")
    batch_index = local_update % count
    return batch_index, batch_index * BATCH_SIZE


def _uniform_ce_piece(logits: torch.Tensor, labels: torch.Tensor, *, logical_batch_size: int) -> torch.Tensor:
    """One microbatch's share of a uniform logical-batch CE mean."""
    _require(logits.ndim == 2 and logits.shape[0] == labels.shape[0], "evaluator logits and labels differ")
    _require(logits.shape[1] == 10 and logical_batch_size == BATCH_SIZE, "uniform CE class or logical batch size differs")
    return torch.nn.functional.cross_entropy(logits, labels, reduction="sum") / logical_batch_size


def _training_batch(train_inputs: Mapping[str, Any], arm: str, batch_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = A_BATCH_COUNT if arm == ARM_A else B_BATCH_COUNT
    _require(arm in ARM_NAMES and 0 <= batch_index < count, "requested batch is outside this arm's task population")
    start = batch_index * BATCH_SIZE
    positions = torch.arange(start, start + BATCH_SIZE, dtype=torch.long)
    labels = train_inputs["labels"].index_select(0, positions).cpu()
    image_positions = train_inputs["fit_measure_positions"].index_select(0, positions)
    source_rows = train_inputs["source_indices"].index_select(0, positions)
    _require(torch.bincount(labels, minlength=10).tolist() == [1] * 10, "logical batch must contain one row per class")
    return positions, image_positions, source_rows, labels


def _uniform_microbatch_loss_gradients(consumer: Any, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], images: torch.Tensor, labels: torch.Tensor):
    """Production 10-row gradient with uniform CE and the existing joint BOGT direction."""
    _require(images.shape[0] == BATCH_SIZE and labels.shape == (BATCH_SIZE,), "logical update must contain ten rows")
    _require(torch.bincount(labels.detach().cpu(), minlength=10).tolist() == [1] * 10, "logical batch must have one row per class")
    refs = consumer.references
    _require(refs is not None, "frozen BOGT references are missing")
    _require(labels.shape == (BATCH_SIZE,), "uniform logical-batch labels differ")
    z_parts = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
            stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
            result = consumer.runtime.forward_from_image(images[start:stop])
            z_parts.append(result.z_ols.coefficients.detach())
            del result
    z_all = torch.cat(z_parts)
    loss_dist, d_dist = capacity._dist_loss_and_ols_gradient(z_all, refs.scale)
    names = ("encoder.phase", "generator.phase")
    g_dist = {key: torch.zeros_like(parameters[key]) for key in names}
    g_bogt = {key: torch.zeros_like(parameters[key]) for key in names}
    g_sem = {ev: {key: torch.zeros_like(parameters[key]) for key in names} for ev in ("E_S", "E_I", "E_C")}
    losses = {"L_rec": 0.0, "L_op": 0.0, **{f"{ev}_ce": 0.0 for ev in ("E_S", "E_I", "E_C")}}
    powers, extra_powers = [], []
    for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
        fraction = (stop - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw, target, z = result.raw_intensity, result.image_input.slm_intensity, result.z_ols.coefficients
        proxy = torch.real(torch.conj(d_dist[start:stop].to(z)) * z).sum()
        l_rec = (raw - target).square().mean()
        plane = capacity.bogt.contracted_plane_powers(result)
        l_op = capacity.bogt.optical_power_loss_from_powers(plane, refs.power_reference)
        view = capacity.canonical.cmos_view(raw)
        ce = {ev: _uniform_ce_piece(capacity._evaluator_logits(evaluators[ev], view, ev), labels[start:stop], logical_batch_size=BATCH_SIZE) for ev in ("E_S", "E_I", "E_C")}
        gd = capacity._gradient(proxy, parameters, retain_graph=True)
        l_bogt = capacity.semantic.BETA * l_rec + capacity.semantic.GAMMA * l_op
        gb = capacity._gradient(l_bogt, parameters, retain_graph=True)
        gs = {ev: capacity._gradient(ce[ev], parameters, retain_graph=i < 2) for i, ev in enumerate(("E_S", "E_I", "E_C"))}
        for key in names:
            g_dist[key] += gd[key]
            g_bogt[key] += fraction * gb[key]
            for ev in ("E_S", "E_I", "E_C"):
                g_sem[ev][key] += gs[ev][key]
        losses["L_rec"] += fraction * float(l_rec.detach().double().cpu())
        losses["L_op"] += fraction * float(l_op.detach().double().cpu())
        for ev in ("E_S", "E_I", "E_C"):
            losses[f"{ev}_ce"] += float(ce[ev].detach().double().cpu())
        powers.append(plane.detach())
        extra = consumer.runtime.generator.last_extra_plane_powers
        _require(extra is not None, "expanded Generator omitted the new-plane power readout")
        extra_powers.append(extra.detach())
        del result, raw, target, z, proxy, l_rec, plane, l_op, view, ce, gd, l_bogt, gb, gs
    semantic, unavailable = ablation._canonical_semantic_gradient(g_sem, ("E_S", "E_I", "E_C"))
    _require(semantic is not None, f"uniform semantic direction unavailable: {unavailable}")
    _require(capacity._vector_norm(g_dist["generator.phase"]) == 0.0, "L_dist produced a Generator gradient")
    combined = {key: capacity.semantic.ALPHA * g_dist[key] + g_bogt[key] + capacity.semantic.KAPPA * semantic[key] for key in names}
    _require(all(bool(torch.isfinite(value).all()) for value in combined.values()), "uniform combined direction is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only production autograd populated .grad")
    consumer.runtime.generator.last_extra_plane_powers = torch.cat(extra_powers)
    return torch.cat(powers), {"L_dist": loss_dist, **{key: torch.tensor(value, device=images.device) for key, value in losses.items()}}, {"L_dist": g_dist, "semantic": g_sem}, combined


def _train_step(consumer: Any, optimizer: torch.optim.Optimizer, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], train_inputs: Mapping[str, Any], arm: str, local_update: int, *, batch_index: int | None = None) -> dict[str, Any]:
    expected_index, _ = _batch_indices(arm, local_update)
    if batch_index is None:
        batch_index = expected_index
    _require(0 <= batch_index < (A_BATCH_COUNT if arm == ARM_A else B_BATCH_COUNT), "selected training batch index is invalid")
    positions, fit_positions, row_ids, labels_cpu = _training_batch(train_inputs, arm, batch_index)
    batch = train_inputs["images"].index_select(0, positions).to(consumer.device)
    labels = labels_cpu.to(consumer.device)
    _require(labels.shape[0] == BATCH_SIZE, "uniform semantic objective must receive ten rows")
    planes, losses, _gradients, combined = _uniform_microbatch_loss_gradients(consumer, parameters, evaluators, batch, labels)
    phase = parameters["generator.phase"]
    gradient = combined["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "read-only Encoder gradient entered Adam")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(float(optimizer.state[phase]["step"].item()))
    # Adam has already stepped; its counter must correspond to the model update just performed.
    expected_step = SOURCE_STEP + local_update + 1
    _require(step == expected_step, "Generator Adam count differs from task-local update count")
    delta = phase.detach() - before
    old_grad, new_grad = capacity._phase_parts(gradient, "B_large")
    old_delta, new_delta = capacity._phase_parts(delta, "B_large")
    record = {
        "update": step,
        "task_local_update": local_update + 1,
        "arm": arm,
        "batch_index": batch_index,
        "fit_positions": fit_positions.tolist(),
        "training_positions": positions.tolist(),
        "source_row_indices": row_ids.tolist(),
        "labels": labels_cpu.tolist(),
        "training_evaluators": ["E_S", "E_I", "E_C"],
        "semantic_weight_policy": "uniform_x1",
        "ce_denominator": 10.0,
        "ce_by_evaluator": {name: float(losses[f"{name}_ce"].detach().cpu()) for name in ("E_S", "E_I", "E_C")},
        "raw_mse": float(losses["L_rec"].detach().cpu()),
        "l_dist": float(losses["L_dist"].detach().cpu()),
        "l_op": float(losses["L_op"].detach().cpu()),
        "five_plane_power_mean_by_batch": planes.detach().to(torch.float64).mean(0).cpu().tolist(),
        "generator_gradient_norm": capacity._vector_norm(gradient),
        "generator_gradient_rms": capacity._vector_rms(gradient),
        "old_100_plane_gradient_norm": capacity._vector_norm(old_grad),
        "new_400_plane_gradient_norm": capacity._vector_norm(new_grad),
        "old_100_plane_displacement_rms": capacity._vector_rms(old_delta),
        "new_400_plane_displacement_rms": capacity._vector_rms(new_delta),
        "generator_displacement_rms": capacity._vector_rms(delta),
        "generator_adam_step": step,
        "all_parameters_finite": bool(torch.isfinite(phase).all()),
    }
    _require(record["all_parameters_finite"] and record["new_400_plane_gradient_norm"] > 0.0, "update is nonfinite or omitted the expanded phase group")
    _require(all(parameter.grad is None for parameter in parameters.values()), "optical gradients were not cleared after the read-only update")
    return record


def _checkpoint_name(arm: str, step: int) -> str:
    _require(arm in ARM_NAMES, "unknown checkpoint arm")
    return f"arm_{'a' if arm == ARM_A else 'b'}_step{step:04d}.pt"


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, local_completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    params = capacity._parameters(consumer)
    step = int(float(optimizer.state[params["generator.phase"]]["step"].item()))
    batch_count = A_BATCH_COUNT if arm == ARM_A else B_BATCH_COUNT
    _require(step == completed and local_completed == completed - SOURCE_STEP and len(history) == completed, "checkpoint update counts or history length differ")
    return {
        "schema": CHECKPOINT_SCHEMA, "task_id": TASK_ID, "source_task_id": PARENT_TASK_ID,
        "arm": arm, "source_checkpoint": source["checkpoint_path"], "source_step": SOURCE_STEP,
        "checkpoint_reason": reason or ("source_start" if completed == SOURCE_STEP else "fixed_interval_or_terminal"),
        "additional_updates": local_completed, "completed_updates": completed, "task_local_updates": local_completed,
        "optimizer_step": step, "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": source["architecture_layout"], "population": dict(population),
        "trainable_parameter_state": capacity._snapshot_parameters(params),
        "optimizer_state": _cpu_tree(optimizer.state_dict()), "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": consumer.references.to_payload(), "weights": consumer.weights.to_payload(),
        "training_history": [dict(row) for row in history], "source_history_count": SOURCE_STEP,
        "source_rng_state": source["rng_state"], "rng_state": capacity._rng_state(),
        "task_batch_count": batch_count, "next_batch_index": local_completed % batch_count,
        "next_batch_cursor": local_completed % batch_count, "numeric_mode": dict(NUMERIC_MODE),
    }


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, local_completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any], source: Mapping[str, Any], *, reason: str | None = None) -> Path:
    path = target / _checkpoint_name(arm, completed)
    _require(not path.exists(), f"checkpoint will not be overwritten: {path.name}")
    _write_torch(path, _checkpoint_payload(consumer, optimizer, arm, completed, local_completed, history, population, source, reason=reason), "population-expansion Generator Adam recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any], source: Mapping[str, Any], *, measurement_baseline: bool = False) -> dict[str, Any]:
    _require(path.resolve().parent == target.resolve() and path.is_file(), "resume requires the explicit task checkpoint inside its output directory")
    value = torch.load(path, map_location="cpu", weights_only=True)
    arm, step = value.get("arm"), value.get("completed_updates")
    _require(value.get("schema") == CHECKPOINT_SCHEMA and value.get("task_id") == TASK_ID and value.get("source_task_id") == PARENT_TASK_ID, "resume checkpoint belongs to another task")
    _require(arm in ARM_NAMES and isinstance(step, int) and SOURCE_STEP <= step <= FINAL_STEP, "resume checkpoint arm/step differs")
    reason = value.get("checkpoint_reason")
    resumable_boundary = step == SOURCE_STEP or step == FINAL_STEP or step % CHECKPOINT_INTERVAL == 0 or reason == "interrupt_boundary"
    _require(resumable_boundary, "resume checkpoint is not an interval or saved interrupt boundary")
    local = step - SOURCE_STEP
    batch_count = A_BATCH_COUNT if arm == ARM_A else B_BATCH_COUNT
    _require(path.name == _checkpoint_name(arm, step) and value.get("source_checkpoint") == source["checkpoint_path"], "resume checkpoint path/source differs")
    _require(value.get("population") == dict(population) and value.get("optimizer_step") == step and value.get("task_local_updates") == local, "resume population or update count differs")
    _require(value.get("task_batch_count") == batch_count and value.get("next_batch_index") == local % batch_count and value.get("next_batch_cursor") == local % batch_count, "resume cursor used the wrong population batch count")
    _require(value.get("numeric_mode") == NUMERIC_MODE and value.get("architecture_layout") == source["architecture_layout"], "resume numerical mode or architecture differs")
    history = value.get("training_history")
    _require(isinstance(history, list) and len(history) == step and _tree_equal(history[:SOURCE_STEP], source["training_history"]), "resume history lacks the exact source trajectory")
    source_measure = population["source_row_indices"]
    labels = population["labels"]
    fit_measure_positions = population["fit_mapping_A_measure_positions"] if arm == ARM_A else population["fit_mapping_B_measure_positions"]
    total_batches = batch_count
    for update, row in enumerate(history[SOURCE_STEP:], start=0):
        batch_index = update % total_batches
        train_pos = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        measure_pos = [fit_measure_positions[p] for p in train_pos]
        _require(row.get("update") == SOURCE_STEP + update + 1 and row.get("task_local_update") == update + 1 and row.get("arm") == arm, f"resume history step/arm differs at task update {update + 1}")
        _require(row.get("batch_index") == batch_index and row.get("training_positions") == train_pos and row.get("fit_positions") == measure_pos, f"resume history task-local batch mapping differs at update {update + 1}")
        _require(row.get("source_row_indices") == [source_measure[p] for p in measure_pos] and row.get("labels") == [labels[p] for p in measure_pos], f"resume history source row/label mapping differs at update {update + 1}")
        _require(row.get("training_evaluators") == ["E_S", "E_I", "E_C"] and row.get("semantic_weight_policy") == "uniform_x1" and row.get("ce_denominator") == 10.0, f"resume history objective differs at update {update + 1}")
    _require(value.get("source_history_count") == SOURCE_STEP, "resume source-history boundary differs")
    params = value.get("trainable_parameter_state")
    _require(isinstance(params, Mapping) and set(params) == {"encoder.phase", "generator.phase"}, "resume model state is incomplete")
    for name, tensor in params.items():
        ref = source["trainable_parameter_state"][name]
        _require(tensor.shape == ref.shape and tensor.dtype == ref.dtype and bool(torch.isfinite(tensor).all()), f"resume parameter invalid: {name}")
    _require(_tree_equal(params["encoder.phase"], source["trainable_parameter_state"]["encoder.phase"]), "resume changed the frozen Encoder")
    adam = value["optimizer_state"]
    _require(len(adam["param_groups"]) == 1 and adam["param_groups"][0]["params"] == [0] and set(adam["state"]) == {0}, "resume Adam scope differs")
    group = adam["param_groups"][0]
    _require(group.get("lr") == capacity.GENERATOR_LEARNING_RATE and tuple(group.get("betas", ())) == capacity.GENERATOR_BETAS and group.get("eps") == capacity.GENERATOR_EPS and group.get("weight_decay") == 0.0, "resume Adam hyperparameters differ")
    slot = adam["state"][0]
    _require(int(float(slot["step"].item())) == step and slot["exp_avg"].shape == (340_000,) and slot["exp_avg_sq"].shape == (340_000,), "resume Adam step/moments differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "resume Adam moments are nonfinite")
    _require(_tree_equal(value["frozen_runtime_state"], source["frozen_runtime_state"]) and _tree_equal(value["references"], source["references"]) and _tree_equal(value["weights"], source["weights"]), "resume changed frozen state/references/weights")
    _require(isinstance(value.get("rng_state"), Mapping) and len(value["rng_state"]) >= 3, "resume RNG state is incomplete")
    _require(_tree_equal(value.get("source_rng_state"), source["rng_state"]), "resume source RNG lineage differs")
    if not measurement_baseline:
        later = [int(match.group(2)) for candidate in target.glob("arm_[ab]_step*.pt") if (match := CHECKPOINT_RE.fullmatch(candidate.name)) and ("A_repeat_200" if match.group(1) == "a" else "B_expand_2000") == arm and int(match.group(2)) > step]
        _require(not later, f"selected checkpoint is behind saved progress for {arm}")
        if arm == ARM_B:
            _require((target / _checkpoint_name(ARM_A, FINAL_STEP)).is_file(), "B resume requires the completed A checkpoint")
    return value


def _restore(checkpoint: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    params = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(checkpoint["trainable_parameter_state"][name].to(parameter.device))
    ablation._load_adam_state(optimizer, params["generator.phase"], checkpoint["optimizer_state"], expected_step=int(checkpoint["completed_updates"]))
    capacity._restore_rng_state(checkpoint["rng_state"])


def _disk_probe(target: Path, source_checkpoint: Path) -> dict[str, Any]:
    _require(source_checkpoint.is_file(), "source checkpoint is missing for output estimate")
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = __import__("shutil").disk_usage(existing).free
    endpoint_raw = TOTAL_ROWS * 200 * 200 * 4 * 3
    endpoint_other = TOTAL_ROWS * (100 * 16 + 4 * 28 * 28 * 4 + 10 * 4 * 4)
    checkpoints = 24 * (source_checkpoint.stat().st_size + 18_000_000)
    estimate = endpoint_raw + endpoint_other * 3 + checkpoints + 700_000_000
    reserve = 1_610_612_736
    _require(free >= estimate + reserve, f"output volume has {free} bytes; estimated outputs plus reserve require {estimate + reserve}")
    return {"free_bytes_observed": free, "estimated_remaining_output_bytes": estimate, "reserve_bytes": reserve, "required_bytes": estimate + reserve, "passed": True}


def _io_probe(target: Path) -> dict[str, Any]:
    payload = b"lightgen-population-expansion-io-probe-v1"
    locations = {}
    for label, parent in (("task_output", target), ("system_temp", Path(tempfile.gettempdir()))):
        with tempfile.TemporaryDirectory(prefix="lg_population_io_", dir=parent) as temporary:
            path = Path(temporary) / "probe.bin"
            path.write_bytes(payload)
            locations[label] = {"write_read_equal": path.read_bytes() == payload, "bytes": path.stat().st_size}
            _require(locations[label]["write_read_equal"], f"{label} write/read probe differs")
    return {"executed": True, "passed": True, "locations": locations}


def _preflight(model_inputs: Mapping[str, Any], train_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], paths: Mapping[str, Path], base_state: Mapping[str, Any], source: Mapping[str, Any], source_readout: Mapping[str, Any], target: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the required production preflight")
    disk = _disk_probe(target, Path(population["source_checkpoint"]))
    checks["cuda_and_disk"] = {"executed": True, "passed": True, "device": str(device), "disk": disk, "resource_observation": capacity.diagnostic._resource_snapshot(device)}
    checks["output_and_external_temp_io"] = _io_probe(target)
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    rng_before = capacity._rng_state()
    try:
        target_result = _target_positive_control(evaluators, measure_inputs)
        checks["new_confirmation_target_positive_control"] = {"executed": True, "passed": target_result["passed"], "summary": target_result["summary"]}
        _require(target_result["passed"], "a frozen evaluator target positive control failed")
        _write_torch(target / "target_image_evaluator_positive_controls.pt", {"source_row_indices": measure_inputs["source_indices"][-CONFIRMATION_ROWS:], "labels": measure_inputs["labels"][-CONFIRMATION_ROWS:], **{f"{n.lower()}_logits": v for n, v in target_result["logits"].items()}}, "executed fresh TRAIN target positive control")

        start_consumer, start_optimizer, _ = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
        frozen_before = capacity._frozen_runtime_state(start_consumer)
        source_readout_replay = class9._measure_forward(start_consumer, evaluators, {"images": measure_inputs["images"][:SOURCE_ROWS], "source_indices": measure_inputs["source_indices"][:SOURCE_ROWS], "labels": measure_inputs["labels"][:SOURCE_ROWS]})
        replay = _source_replay(source_readout_replay, source_readout)
        checks["source_1200_row_production_replay"] = {"executed": True, "passed": replay["passed"], "fields": replay["fields"]}
        _require(replay["passed"], "A3700 source readout replay differs")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(start_consumer), "source replay changed frozen optical state")
        start_readout = class9._measure_forward(start_consumer, evaluators, measure_inputs)
        _require(_tree_equal(_cpu_tree(start_optimizer.state_dict()), source["optimizer_state"]), "source replay loader changed Adam state")
        capacity._assert_state_equal(source["frozen_runtime_state"], capacity._frozen_runtime_state(start_consumer), "source replay changed frozen state")
        _write_torch(target / "readouts_A_step3700.pt", start_readout, "3300-row fixed production source readout")
        start_summary, _ = class9._row_and_image_metrics(start_readout, population)
        _write_json(target / "metrics_A_step3700.json", start_summary, "3300-row source baseline metrics")
        del start_consumer, start_optimizer
        gc.collect(); torch.cuda.empty_cache()

        source_snapshot = _cpu_tree(source)

        def disposable_three(arm: str, batches: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
            consumer, optimizer, parameters = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
            original = capacity._snapshot_parameters(parameters)
            frozen = capacity._frozen_runtime_state(consumer)
            saved_rng = capacity._rng_state()
            records = []
            for index, batch_index in enumerate(batches):
                records.append(_train_step(consumer, optimizer, parameters, evaluators, train_inputs, arm, index, batch_index=batch_index))
                _require(records[-1]["generator_adam_step"] == SOURCE_STEP + index + 1, "disposable step did not advance source Adam by one")
                _require(records[-1]["old_100_plane_gradient_norm"] > 0.0 and records[-1]["new_400_plane_gradient_norm"] > 0.0, "disposable update omitted one of the Generator phase groups")
                _require(records[-1]["old_100_plane_displacement_rms"] > 0.0 and records[-1]["new_400_plane_displacement_rms"] > 0.0, "disposable Adam update did not move both Generator phase groups")
            result = {"records": records, "parameters": capacity._snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict())}
            _require(not _tree_equal(original["generator.phase"], result["parameters"]["generator.phase"]), f"{arm} disposable path did not update Generator")
            _require(_tree_equal(original["encoder.phase"], result["parameters"]["encoder.phase"]), f"{arm} disposable path changed Encoder")
            capacity._assert_state_equal(frozen, capacity._frozen_runtime_state(consumer), f"{arm} disposable update changed frozen optical state")
            _require(all(parameter.grad is None for parameter in parameters.values()), f"{arm} disposable update left optical gradients")
            capacity._restore_rng_state(saved_rng)
            del consumer, optimizer, parameters
            gc.collect(); torch.cuda.empty_cache()
            return result, {"passed": True, "updates": len(records), "batch_indices": list(batches), "adam_steps": [row["generator_adam_step"] for row in records]}

        a1, a_check = disposable_three(ARM_A, (0, 1, 2))
        a2, _ = disposable_three(ARM_A, (0, 1, 2))
        aa_passed, aa_details = capacity._repeatability({"parameters": a1["parameters"], "optimizer": a1["optimizer"], "records": a1["records"]}, {"parameters": a2["parameters"], "optimizer": a2["optimizer"], "records": a2["records"]})
        checks["A_three_step_real_update_and_AA"] = {"executed": True, "passed": aa_passed, **a_check, "repeatability": aa_details}
        _require(aa_passed, "A/A disposable updates are not reproducible within the declared numeric tolerance")
        b1, b_check = disposable_three(ARM_B, (0, 1, 2))
        checks["B_three_step_real_update"] = {"executed": True, "passed": True, **b_check}
        b_new, b_new_check = disposable_three(ARM_B, (20, 21, 22))
        checks["B_new_population_real_update"] = {"executed": True, "passed": True, **b_new_check, "source_rows": [row["source_row_indices"] for row in b_new["records"]], "expanded_rows_entered": all(all(index >= SOURCE_ROWS for index in row["fit_positions"]) for row in b_new["records"])}
        _require(checks["B_new_population_real_update"]["expanded_rows_entered"], "new expanded rows were not used in the production update")

        batch_index, offset = _batch_indices(ARM_B, 0)
        labels_cpu = train_inputs["labels"][offset : offset + BATCH_SIZE]
        wrong = labels_cpu.roll(1)
        _require(not torch.equal(wrong, labels_cpu), "wrong-label negative control did not change labels")
        control_consumer, control_optimizer, control_parameters = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
        control_batch = train_inputs["images"][:BATCH_SIZE].to(device)
        control_labels = labels_cpu.to(device)
        correct_result = _uniform_microbatch_loss_gradients(control_consumer, control_parameters, evaluators, control_batch, control_labels)
        correct_gradient = correct_result[3]["generator.phase"].detach().clone()
        wrong_result = _uniform_microbatch_loss_gradients(control_consumer, control_parameters, evaluators, control_batch, wrong.to(device))
        wrong_gradient = wrong_result[3]["generator.phase"]
        gradient_changed = not torch.allclose(correct_gradient, wrong_gradient, rtol=1e-5, atol=1e-9)
        checks["wrong_label_counterfactual"] = {"executed": True, "passed": gradient_changed, "wrong_label_batch_index": batch_index, "wrong_labels": wrong.tolist(), "gradient_difference_l2": capacity._vector_norm(correct_gradient - wrong_gradient), "expected": "wrong labels change the actual production semantic gradient"}
        _require(gradient_changed, "wrong-label counterfactual did not change production gradient")
        del control_consumer, control_optimizer, control_parameters, correct_result, wrong_result
        gc.collect(); torch.cuda.empty_cache()

        # Resume equivalence uses a real serialized B checkpoint and the production resume validator.
        uninterrupted_consumer, uninterrupted_optimizer, uninterrupted_params = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
        uninterrupted_records = [_train_step(uninterrupted_consumer, uninterrupted_optimizer, uninterrupted_params, evaluators, train_inputs, ARM_B, index) for index in (0, 1)]
        uninterrupted_state = {"parameters": capacity._snapshot_parameters(uninterrupted_params), "optimizer": _cpu_tree(uninterrupted_optimizer.state_dict()), "records": uninterrupted_records, "rng": capacity._rng_state()}
        del uninterrupted_consumer, uninterrupted_optimizer, uninterrupted_params
        gc.collect(); torch.cuda.empty_cache()
        split_consumer, split_optimizer, split_params = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
        first_record = _train_step(split_consumer, split_optimizer, split_params, evaluators, train_inputs, ARM_B, 0)
        split_history = list(source["training_history"]) + [first_record]
        interrupt_payload = _checkpoint_payload(split_consumer, split_optimizer, ARM_B, SOURCE_STEP + 1, 1, split_history, population, source, reason="interrupt_boundary")
        with tempfile.TemporaryDirectory(prefix="lg_population_resume_", dir=target) as scratch:
            scratch_path = Path(scratch)
            event_path = scratch_path / _checkpoint_name(ARM_B, SOURCE_STEP + 1)
            # B resumes only after A's final checkpoint exists. This disposable
            # marker supplies that ordering prerequisite for the parser test.
            (scratch_path / _checkpoint_name(ARM_A, FINAL_STEP)).write_bytes(b"preflight A-completion ordering marker")
            torch.save(interrupt_payload, event_path)
            saved = _read_checkpoint(event_path, scratch_path, population, source)
            del split_consumer, split_optimizer, split_params
            split_consumer, split_optimizer, split_params = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, device)
            _restore(saved, split_consumer, split_optimizer)
            resumed_record = _train_step(split_consumer, split_optimizer, split_params, evaluators, train_inputs, ARM_B, 1)
            resumed_state = {"parameters": capacity._snapshot_parameters(split_params), "optimizer": _cpu_tree(split_optimizer.state_dict()), "records": [first_record, resumed_record], "rng": capacity._rng_state()}
        resume_passed, resume_details = capacity._repeatability(uninterrupted_state, resumed_state)
        checks["population_cursor_and_recovery_mapping"] = {"executed": True, "passed": resume_passed, "A_batch_count": A_BATCH_COUNT, "B_batch_count": B_BATCH_COUNT, "A_row_updates": 50, "B_row_updates": 5, "B_saved_step": SOURCE_STEP + 1, "B_resume_cursor": 1, "A_completion_ordering_prerequisite": "temporary marker in disposable recovery directory", "next_batch_source_rows": resumed_record["source_row_indices"], "repeatability": resume_details}
        _require(resume_passed, "interrupted and uninterrupted B updates differ")
        del split_consumer, split_optimizer, split_params
        gc.collect(); torch.cuda.empty_cache()

        frozen_evaluators = all(_tree_equal(evaluator_before[name], capacity._state_snapshot(model)) and all(p.grad is None and not p.requires_grad for p in model.parameters()) for name, model in evaluators.items())
        checks["frozen_evaluators"] = {"executed": True, "passed": frozen_evaluators}
        _require(frozen_evaluators, "preflight altered or unfroze an evaluator")
        source_unchanged = _tree_equal(source_snapshot, source)
        checks["source_checkpoint_and_adam_immutability"] = {"executed": True, "passed": source_unchanged, "source_adam_step": int(float(source["optimizer_state"]["state"][0]["step"].item()))}
        _require(source_unchanged, "disposable preflight changed the source model or Adam")
        _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "preflight is missing an executed passing result")
        return {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "READY", "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates_per_arm": ADDED_UPDATES, "population_rows": TOTAL_ROWS, "numeric_mode": NUMERIC_MODE, "disk": disk, "source_replay": replay, "checks": checks, "duration_seconds_observation_only": time.perf_counter() - _STARTED}
    finally:
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight final evaluator {name}")
        capacity._restore_rng_state(rng_before)
        gc.collect(); torch.cuda.empty_cache()


def _make_control(population: Mapping[str, Any], source: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=f"{TASK_ID}-attempt-01", trajectory_id="large-generator-population-200-vs-2000-3700-to-4700",
        semantic_authority_id=PARENT_TASK_ID,
        data_role="canonical TRAIN: A updates original 200; B updates original 200 plus 1,800 additional rows; fresh 300 confirmation rows excluded",
        config_identity={"grid": 2048, "source_step": SOURCE_STEP, "final_step": FINAL_STEP, "updates": ADDED_UPDATES, "batch_size": BATCH_SIZE, "A_batches": A_BATCH_COUNT, "B_batches": B_BATCH_COUNT, "numeric_mode": NUMERIC_MODE},
        model_identity={"checkpoint": population["source_checkpoint"], "parameter_shapes": population["source_parameter_shapes"], "architecture": source["architecture_layout"]},
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": capacity.GENERATOR_LEARNING_RATE, "betas": list(capacity.GENERATOR_BETAS), "eps": capacity.GENERATOR_EPS, "weight_decay": 0.0, "reset": False},
        sampler_identity={"A_rows": population["fit_mapping_A_measure_positions"], "B_rows": population["fit_mapping_B_measure_positions"], "confirmation_rows": population["source_row_indices"][-CONFIRMATION_ROWS:], "A_batch_count": A_BATCH_COUNT, "B_batch_count": B_BATCH_COUNT},
        execution_source_semantics_id=TASK_ID, scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01")
    checks = preflight["checks"]
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY receipt cannot omit a measured check")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "target_controls": checks["new_confirmation_target_positive_control"]["passed"], "source_replay": checks["source_1200_row_production_replay"]["passed"], "A_A": checks["A_three_step_real_update_and_AA"]["passed"], "B_updates": checks["B_three_step_real_update"]["passed"], "B_expanded_rows": checks["B_new_population_real_update"]["passed"], "negative_control": checks["wrong_label_counterfactual"]["passed"], "recovery": checks["population_cursor_and_recovery_mapping"]["passed"]},
        resource={"passed": checks["cuda_and_disk"]["passed"], "status": "PASS", "cuda_available": True, "disk": checks["cuda_and_disk"]["disk"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_large_generator_training_population_expansion_3700_to_4700"},
        selected_engine="cuda_2048_large_generator_training_population_expansion_3700_to_4700", resource_envelope={"disk": checks["cuda_and_disk"]["disk"]},
    )
    return control, receipt


def _ensure_ready(target: Path, population: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    population_path, ready_path = target / "input_population.json", target / "preflight.json"
    ready = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    if resume:
        _require(json.loads(population_path.read_text(encoding="utf-8")) == dict(population) and json.loads(ready_path.read_text(encoding="utf-8")) == ready, "explicit resume population or READY evidence differs")
    else:
        if not population_path.exists(): _write_json(population_path, population, "fixed 3,300-row TRAIN population and role mapping")
        else: _require(json.loads(population_path.read_text(encoding="utf-8")) == dict(population), "staged population differs")
        if not ready_path.exists(): _write_json(ready_path, ready, "executed production READY preflight")
        else: _require(json.loads(ready_path.read_text(encoding="utf-8")) == ready, "staged READY evidence differs")


def _validate_ready(target: Path, population: Mapping[str, Any], control: ResearchRunControl, expected: PreflightReceipt) -> PreflightReceipt:
    ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY", "READY evidence missing or belongs to another task")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == dict(population), "fixed population changed after READY")
    checks = ready["preflight"].get("checks", {})
    _require(checks and all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "saved READY omits an executed check")
    receipt = PreflightReceipt.from_mapping(ready.get("receipt", {}))
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == expected.authorization_id and receipt.status == "READY", "READY receipt belongs to another authorization")
    control.mark_preflight_ready(receipt)
    return receipt


def _consume_ready_artifacts(target: Path, measure_inputs: Mapping[str, Any], population: Mapping[str, Any], source_readout: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    target_control = torch.load(target / "target_image_evaluator_positive_controls.pt", map_location="cpu", weights_only=True)
    _require(torch.equal(target_control["source_row_indices"], measure_inputs["source_indices"][-CONFIRMATION_ROWS:]) and torch.equal(target_control["labels"], measure_inputs["labels"][-CONFIRMATION_ROWS:]), "saved target control row mapping differs")
    expected_target = preflight["checks"]["new_confirmation_target_positive_control"]["summary"]
    labels = target_control["labels"].to(torch.long)
    for name in EVALUATOR_NAMES:
        logits = target_control[f"{name.lower()}_logits"].to(torch.float64)
        _require(logits.shape == (CONFIRMATION_ROWS, 10) and bool(torch.isfinite(logits).all()), f"saved {name} target logits are invalid")
        prediction = logits.argmax(1)
        correct = [int(((labels == c) & (prediction == c)).sum()) for c in range(10)]
        actual = {"per_class_correct": correct, "macro_recall": sum(correct) / CONFIRMATION_ROWS}
        _require(actual == {key: expected_target[name][key] for key in actual}, f"saved {name} target control no longer reproduces READY")
    start_readout = torch.load(target / "readouts_A_step3700.pt", map_location="cpu", weights_only=True)
    _require(torch.equal(start_readout["source_indices"], measure_inputs["source_indices"]) and torch.equal(start_readout["labels"], measure_inputs["labels"]), "saved A3700 READY readout row mapping differs")
    _require(all(torch.isfinite(value.real if value.is_complex() else value).all() for value in start_readout.values() if torch.is_tensor(value)), "saved A3700 READY readout contains nonfinite values")
    replay = _source_replay(start_readout, source_readout)
    _require(replay["passed"] and replay == preflight["source_replay"], "saved A3700 READY readout no longer reproduces source evidence")
    summary, _ = class9._row_and_image_metrics(start_readout, population)
    saved_summary = json.loads((target / "metrics_A_step3700.json").read_text(encoding="utf-8"))
    _require(summary == saved_summary, "saved A3700 metrics differ from the expanded-population readout")
    # The parent report covers its original 1,200 rows. _population already
    # revalidated that report against the parent population and readout; here
    # the newly staged metrics cover all 3,300 rows, while _source_replay
    # verifies the preserved 1,200-row prefix against the parent readout.
    return replay, summary


def _prepare_target(root: Path, target: Path, resume: bool) -> None:
    out = (root / "outputs").resolve()
    parts = target.relative_to(out).parts if out in target.parents else ()
    _require(len(parts) == 3 and parts[0].startswith("20261008_") and parts[1] == "runs" and parts[-1] != "final_pt", "output must be a new versioned run below outputs/")
    if resume:
        _require(target.is_dir(), "explicit resume output directory is missing")
    else:
        if target.exists():
            ready_stage = {"input_population.json", "target_image_evaluator_positive_controls.pt", "readouts_A_step3700.pt", "metrics_A_step3700.json", "preflight.json"}
            _require(target.is_dir() and {path.name for path in target.iterdir()} == ready_stage, "existing task output requires an exact READY stage or explicit checkpoint resume")
        else:
            target.mkdir(parents=True, exist_ok=False)


def _training_loop(target: Path, model_inputs: Mapping[str, Any], train_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, nn.Module], arm: str, start_from: Mapping[str, Any] | None) -> dict[str, Any]:
    consumer, optimizer, params = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, torch.device("cuda:0"))
    if start_from is not None:
        _restore(start_from, consumer, optimizer)
    local_completed = 0 if start_from is None else int(start_from["completed_updates"]) - SOURCE_STEP
    completed = SOURCE_STEP + local_completed
    history = list(source["training_history"] if start_from is None else start_from["training_history"])
    _require(len(history) == completed and 0 <= local_completed <= ADDED_UPDATES, "training history or task-local count differs")
    if not (target / _checkpoint_name(arm, completed)).exists(): _save_checkpoint(target, consumer, optimizer, arm, completed, local_completed, history, population, source)
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
        while local_completed < ADDED_UPDATES:
            record = _train_step(consumer, optimizer, params, evaluators, train_inputs, arm, local_completed)
            local_completed += 1; completed += 1; history.append(record)
            if completed % CHECKPOINT_INTERVAL == 0 or stop_requested:
                reason = "interrupt_boundary" if stop_requested else "fixed_interval_or_terminal"
                checkpoint_path = _save_checkpoint(target, consumer, optimizer, arm, completed, local_completed, history, population, source, reason=reason)
                class9._resource(target, f"{arm}_checkpoint", completed)
                _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed frozen Encoder")
                capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "training changed frozen optical runtime")
                for name, model in evaluators.items():
                    capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"training changed evaluator {name}")
                    _require(all(p.grad is None and not p.requires_grad for p in model.parameters()), f"evaluator {name} is not frozen")
                gc.collect(); torch.cuda.empty_cache()
                if stop_requested: raise TrainingInterrupted(checkpoint_path.name)
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    _require(local_completed == ADDED_UPDATES and completed == FINAL_STEP and len(history) == FINAL_STEP, "training missed its fixed 1,000-update budget")
    _require(torch.equal(params["encoder.phase"].detach().cpu(), encoder_before), "training changed Encoder at completion")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "frozen runtime changed at completion")
    for name, model in evaluators.items(): capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"evaluator {name} changed at completion")
    final_path = target / _checkpoint_name(arm, FINAL_STEP)
    if not final_path.exists(): _save_checkpoint(target, consumer, optimizer, arm, completed, local_completed, history, population, source)
    history_path = target / f"training_history_{'A_repeat_200' if arm == ARM_A else 'B_expand_2000'}.json"
    payload = {"task_id": TASK_ID, "arm": arm, "source_step": SOURCE_STEP, "additional_updates": ADDED_UPDATES, "final_step": FINAL_STEP, "training_evaluators": ["E_S", "E_I", "E_C"], "training_rows_per_arm": BASE_FIT_ROWS if arm == ARM_A else BASE_FIT_ROWS + EXPANDED_ROWS, "records": history}
    if history_path.exists(): _require(json.loads(history_path.read_text(encoding="utf-8")) == payload, "saved training history differs")
    else: _write_json(history_path, payload, "source history and one thousand population comparison updates")
    result = torch.load(final_path, map_location="cpu", weights_only=True)
    del consumer, optimizer, params; gc.collect(); torch.cuda.empty_cache()
    return result


def _endpoint(target: Path, name: str, checkpoint: Mapping[str, Any], arm: str, model_inputs: Mapping[str, Any], measure_inputs: Mapping[str, Any], population: Mapping[str, Any], base_state: Mapping[str, Any], source: Mapping[str, Any], evaluators: Mapping[str, nn.Module], *, verify_existing: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    path = target / f"readouts_{name}.pt"
    saved_readout = torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None
    if saved_readout is not None:
        _require(torch.equal(saved_readout["source_indices"], measure_inputs["source_indices"]) and torch.equal(saved_readout["labels"], measure_inputs["labels"]), f"saved {name} measurement population differs")
    if saved_readout is None or verify_existing:
        consumer, optimizer, params = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, torch.device("cuda:0"))
        _restore(checkpoint, consumer, optimizer)
        before_params = capacity._snapshot_parameters(params); before_adam = _cpu_tree(optimizer.state_dict()); frozen = capacity._frozen_runtime_state(consumer)
        measured = class9._measure_forward(consumer, evaluators, measure_inputs)
        _require(_tree_equal(before_params, capacity._snapshot_parameters(params)) and _tree_equal(before_adam, _cpu_tree(optimizer.state_dict())), f"{name} forward changed model or Adam")
        capacity._assert_state_equal(frozen, capacity._frozen_runtime_state(consumer), f"{name} forward changed frozen state")
        if saved_readout is not None:
            repeatable, details = capacity._repeatability(saved_readout, measured)
            _require(repeatable, f"saved {name} readout does not reproduce from its explicit checkpoint: {details}")
            readout = saved_readout
        else:
            readout = measured
            _write_torch(path, readout, f"3,300-row frozen production readout {name}")
        del consumer, optimizer, params; gc.collect(); torch.cuda.empty_cache()
    else:
        readout = saved_readout
    summary, rows = class9._row_and_image_metrics(readout, population)
    metric_path, row_path = target / f"metrics_{name}.json", target / f"row_metrics_{name}.pt"
    if metric_path.exists(): _require(json.loads(metric_path.read_text(encoding="utf-8")) == summary, f"saved {name} summary differs")
    else: _write_json(metric_path, summary, "3,300-row classification and image metrics")
    if row_path.exists(): _require(_tree_equal(torch.load(row_path, map_location="cpu", weights_only=True), rows), f"saved {name} per-row metrics differ")
    else: _write_torch(row_path, rows, "3,300-row float64 per-row scores and image errors")
    return summary, readout


def _display_image(view: torch.Tensor, scale: int = 4) -> Image.Image:
    value = view.detach().to(torch.float32).cpu().squeeze().clamp(0, 1)
    pixels = torch.round((1 - value) * 255).to(torch.uint8).numpy()
    return Image.fromarray(pixels, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _comparison_pages(target: Path, inputs: Mapping[str, Any], start: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    pages: dict[str, list[str]] = {}
    displayed_source_indices: dict[str, list[int]] = {}
    x_positions, titles = (350, 465, 580, 695), ("Target", "A3700", "A4700", "B4700")
    role_pages = _role_page_positions(population)
    for role, page_positions in role_pages.items():
        pages[role] = []
        displayed_source_indices[role] = []
        start_pos, stop_pos = population["role_positions"][role]
        for positions in page_positions:
            canvas = Image.new("RGB", (820, 58 + len(positions) * 150), "white"); draw = ImageDraw.Draw(canvas)
            offset = positions[0]
            draw.text((8, 8), f"{role}; rows {offset-start_pos+1}-{offset-start_pos+len(positions)} of {stop_pos-start_pos}; fixed [0,1]", fill="black")
            for x, title in zip(x_positions, titles, strict=True): draw.text((x, 35), title, fill="black")
            for row, position in enumerate(positions):
                y = 55 + row * 150
                draw.text((8, y + 8), f"row={int(inputs['source_indices'][position])} y={int(inputs['labels'][position])} {role}", fill="black")
                for col, panel in enumerate((inputs["images"][position], start["evaluator_view"][position], a["evaluator_view"][position], b["evaluator_view"][position])): canvas.paste(_display_image(panel), (x_positions[col], y))
                for col, readout in enumerate((start, a, b), start=1): draw.text((x_positions[col], y + 116), f"E_D={int(readout['e_d_logits'][position].double().argmax())}", fill="black")
                displayed_source_indices[role].append(int(inputs["source_indices"][position]))
            filename = f"comparison_{role}_{len(pages[role])+1:03d}.png"; path = target / filename
            if path.exists():
                with Image.open(path) as old: _require(old.size == canvas.size and old.convert("RGB").tobytes() == canvas.tobytes(), f"saved image page differs: {filename}")
            else: canvas.save(path, optimize=True)
            pages[role].append(filename)
    _require(sum(len(rows) for rows in displayed_source_indices.values()) == TOTAL_ROWS, "rendered image manifest does not cover all population rows")
    _require(all(displayed_source_indices[role] == inputs["source_indices"][start:stop].tolist() for role, (start, stop) in population["role_positions"].items()), "rendered image manifest changed row coverage or order")
    return {"pages_by_role": pages, "role_counts": population["role_counts"], "displayed_source_indices_by_role": displayed_source_indices, "displayed_row_count": sum(len(rows) for rows in displayed_source_indices.values()), "panel_order": ["target", "A3700", "A4700", "B4700"], "fixed_display_range": [0, 1], "per_image_rescaling": False}


def _paired_transitions(readouts: Mapping[str, Mapping[str, Any]], labels: torch.Tensor, roles: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    result = {}
    for role, bounds in roles.items():
        lo, hi = map(int, bounds); y = labels[lo:hi]
        preds = {name: readouts[name]["e_d_logits"][lo:hi].double().argmax(1) for name in readouts}
        result[role] = {}
        for left, right in (("A3700", "A4700"), ("A3700", "B4700"), ("A4700", "B4700")):
            result[role][f"{left}_to_{right}"] = {str(c): {"correct_to_wrong": int(((y == c) & (preds[left] == c) & (preds[right] != c)).sum()), "wrong_to_correct": int(((y == c) & (preds[left] != c) & (preds[right] == c)).sum())} for c in range(10)}
    return result


def _report(target: Path, source_report: Mapping[str, Any], population: Mapping[str, Any], replay: Mapping[str, Any], start: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], readouts: Mapping[str, Any], elapsed: float) -> dict[str, Any]:
    decision = _decision(start, a, b, valid=True)
    pages = _comparison_pages(target, readouts["inputs"], readouts["A3700"], readouts["A4700"], readouts["B4700"], population)
    report = {
        "schema": REPORT_SCHEMA, "task_id": TASK_ID, "status": decision["status"], "classification": decision["classification"],
        "scientific_updates": {"A": ADDED_UPDATES, "B": ADDED_UPDATES},
        "parent": {"task_id": PARENT_TASK_ID, "status": source_report["status"], "classification": source_report["classification"], "source_checkpoint": population["source_checkpoint"]},
        "scope": {"source_step": SOURCE_STEP, "final_step": FINAL_STEP, "additional_updates_per_arm": ADDED_UPDATES, "fit_rows_A": BASE_FIT_ROWS, "fit_rows_B": BASE_FIT_ROWS + EXPANDED_ROWS, "expanded_rows_per_class": EXPANDED_PER_CLASS, "new_confirmation_rows": CONFIRMATION_ROWS, "total_measurement_rows": TOTAL_ROWS, "all_rows_canonical_train": True, "unseen_claim": False, "updated_parameter": "generator.phase", "encoder_ols_geometry_buffers_evaluators_frozen": True, "A_cycles": ADDED_UPDATES // A_BATCH_COUNT, "B_cycles": ADDED_UPDATES // B_BATCH_COUNT, "A_row_updates": 50, "B_row_updates": 5, "training_evaluators": ["E_S", "E_I", "E_C"], "numeric_mode": NUMERIC_MODE},
        "source_readout_replay": replay, "endpoint_metrics": {"A3700": start, "A4700": a, "B4700": b},
        "new_confirmation_class_counts_by_id": {str(c): {"A3700": decision["new_confirmation"]["per_class_correct"]["A3700"][c], "A4700": decision["new_confirmation"]["per_class_correct"]["A4700"][c], "B4700": decision["new_confirmation"]["per_class_correct"]["B4700"][c]} for c in range(10)},
        "paired_correctness_transitions": _paired_transitions({"A3700": readouts["A3700"], "A4700": readouts["A4700"], "B4700": readouts["B4700"]}, readouts["inputs"]["labels"], population["role_positions"]),
        **decision,
        "readout_artifacts": {name: f"readouts_{name}.pt" for name in ("A3700", "A4700", "B4700")},
        "training_history_artifacts": {"A": "training_history_A_repeat_200.json", "B": "training_history_B_expand_2000.json"},
        "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_[ab]_step*.pt")),
        "comparison_images": pages, "preflight_artifact": "preflight.json", "elapsed_seconds_observation_only": elapsed,
        "next_action": "NONE", "stop": True,
    }
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class9-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit task checkpoint required with --resume")
    return parser.parse_args()


_STARTED = 0.0


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    global _STARTED
    args = args or _arguments(); _STARTED = time.perf_counter()
    root = _root(); parent_dir = _resolve(root, args.class9_dir); target = _resolve(root, args.output_dir)
    _require(bool(args.resume) == (args.checkpoint is not None), "--resume requires one explicit --checkpoint")
    _require(not (args.resume and args.preflight_only), "--resume cannot be combined with --preflight-only")
    try:
        model_inputs, train_inputs, measure_inputs, population, paths, base_state, source, source_readout, source_report = _population(root, parent_dir)
        _prepare_target(root, target, args.resume)
        if (target / "report.json").is_file():
            _require(args.resume and args.checkpoint is not None, "completed output requires explicit terminal resume checkpoint")
            selected = _read_checkpoint(_resolve(root, args.checkpoint), target, population, source)
            report = json.loads((target / "report.json").read_text(encoding="utf-8"))
            _require(selected["arm"] == ARM_B and selected["completed_updates"] == FINAL_STEP and report["task_id"] == TASK_ID, "terminal recovery requires explicit B4700 checkpoint")
            return report
        if args.resume:
            selected = _read_checkpoint(_resolve(root, args.checkpoint), target, population, source)
            ready_path = target / "preflight.json"
            _require(ready_path.is_file(), "resume lacks the saved READY receipt")
            preflight = json.loads(ready_path.read_text(encoding="utf-8"))["preflight"]
        else:
            selected = None
            population_path = target / "input_population.json"
            if population_path.exists(): _require(json.loads(population_path.read_text(encoding="utf-8")) == population, "staged population differs")
            else: _write_json(population_path, population, "fixed 3,300-row canonical TRAIN population")
            preflight_path = target / "preflight.json"
            if preflight_path.is_file():
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))["preflight"]
                _require(all(check.get("executed") and check.get("passed") for check in preflight.get("checks", {}).values()), "staged READY lacks executed passing checks")
            else:
                with capacity.strict_fp32_mode() as mode:
                    _require(mode == NUMERIC_MODE, "strict FP32 mode differs")
                    preflight = _preflight(model_inputs, train_inputs, measure_inputs, population, paths, base_state, source, source_readout, target)
                provisional_control, provisional_receipt = _make_control(population, source, preflight)
                _ensure_ready(target, population, preflight, provisional_receipt, resume=False)
                if args.preflight_only: return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}
        control, expected_receipt = _make_control(population, source, preflight)
        receipt = _validate_ready(target, population, control, expected_receipt)
        ready_replay, ready_summary = _consume_ready_artifacts(target, measure_inputs, population, source_readout, preflight)
        if args.preflight_only: return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}
        control.mark_preflight_ready(receipt); control.start_execution(); control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
        with capacity.strict_fp32_mode() as mode:
            _require(mode == NUMERIC_MODE, "execution strict FP32 mode differs")
            evaluators = diagnostic.load_evaluators(paths, torch.device("cuda:0"))
            evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
            if selected is None:
                starts = {}
                for arm in ARM_NAMES:
                    start_path = target / _checkpoint_name(arm, SOURCE_STEP)
                    if start_path.exists(): starts[arm] = _read_checkpoint(start_path, target, population, source, measurement_baseline=True)
                    else:
                        consumer, optimizer, _ = ablation._load_arm(model_inputs, base_state, source, class9.SOURCE_ARM, torch.device("cuda:0"))
                        payload = _checkpoint_payload(consumer, optimizer, arm, SOURCE_STEP, 0, source["training_history"], population, source)
                        _write_torch(start_path, payload, "copied A3700 source start with inherited Adam and RNG")
                        starts[arm] = payload; del consumer, optimizer, _; gc.collect(); torch.cuda.empty_cache()
                for key in ("trainable_parameter_state", "optimizer_state", "rng_state", "references", "weights", "frozen_runtime_state"):
                    _require(_tree_equal(starts[ARM_A][key], starts[ARM_B][key]), f"A/B task starts differ at {key}")
                final_a = _training_loop(target, model_inputs, train_inputs, population, base_state, source, evaluators, ARM_A, starts[ARM_A])
                final_b = _training_loop(target, model_inputs, train_inputs, population, base_state, source, evaluators, ARM_B, starts[ARM_B])
            elif selected["arm"] == ARM_A:
                final_a = _training_loop(target, model_inputs, train_inputs, population, base_state, source, evaluators, ARM_A, selected)
                bstart = _read_checkpoint(target / _checkpoint_name(ARM_B, SOURCE_STEP), target, population, source, measurement_baseline=True)
                final_b = _training_loop(target, model_inputs, train_inputs, population, base_state, source, evaluators, ARM_B, bstart)
            else:
                final_a = _read_checkpoint(target / _checkpoint_name(ARM_A, FINAL_STEP), target, population, source, measurement_baseline=True)
                final_b = _training_loop(target, model_inputs, train_inputs, population, base_state, source, evaluators, ARM_B, selected)
            _require(final_a["completed_updates"] == FINAL_STEP and final_b["completed_updates"] == FINAL_STEP, "both arms must finish at step4700")
            start_summary, start_readout = _endpoint(target, "A3700", starts[ARM_A] if selected is None else source, ARM_A, model_inputs, measure_inputs, population, base_state, source, evaluators, verify_existing=args.resume)
            _require(start_summary == ready_summary, "formal A3700 measurement differs from the consumed READY readout")
            summary_a, readout_a = _endpoint(target, "A4700", final_a, ARM_A, model_inputs, measure_inputs, population, base_state, source, evaluators, verify_existing=args.resume)
            summary_b, readout_b = _endpoint(target, "B4700", final_b, ARM_B, model_inputs, measure_inputs, population, base_state, source, evaluators, verify_existing=args.resume)
            for name, model in evaluators.items(): capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final evaluator {name} changed")
        replay = _source_replay(start_readout, source_readout)
        _require(replay == ready_replay, "final A3700 source replay differs from READY")
        readouts = {"A3700": start_readout, "A4700": readout_a, "B4700": readout_b, "inputs": measure_inputs}
        report = _report(target, source_report, population, replay, start_summary, summary_a, summary_b, readouts, time.perf_counter() - _STARTED)
        if report["status"] == "VALID_COMPLETED": control.complete()
        else: control.valid_failed()
        report["research_run_state"] = control.state
        _write_json(target / "report.json", report, "terminal population coverage decision")
        return report
    except TrainingInterrupted as error:
        return {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume from explicit task checkpoint"}
    except Exception as error:
        if target.is_dir() and not (target / "report.json").exists() and not (target / "invalid_harness.json").exists():
            checkpoints = sorted(p.name for p in target.glob("arm_[ab]_step*.pt"))
            incident = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "failure_type": type(error).__name__, "failure": str(error), "last_durable_checkpoint": checkpoints[-1] if checkpoints else None, "scientific_updates": None, "scientific_conclusion": None, "next_action": "NONE"}
            try: (target / "invalid_harness.json").write_text(json.dumps(incident, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError: pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
