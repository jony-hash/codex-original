"""Localize saved LightGen class-shape separation without model execution."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_fixed_prior_sampled_readability_v1 as readability
from scripts.research.lightgen import audit_lightgen_2d_generator_foreground_shape_localization_v1 as foreground
from scripts.research.lightgen import audit_lightgen_2d_generator_frozen_joint_prior_comparison_v1 as prior_task
from scripts.research.lightgen import audit_lightgen_2d_generator_shape_energy_counterfactual_v1 as source_audit
from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as parent
from scripts.research.lightgen import execute_lightgen_2d_generator_foreground_objective_training_comparison_v1 as training
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation


TASK_ID = "M0_GENERATOR_DIGIT_IDENTITY_LOCALIZATION_V1"
SOURCE_DIR = Path("outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03")
TRAINING_DIR = Path("outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01")
PRIOR_DIR = Path("outputs/20261009_m0_generator_frozen_joint_prior_comparison_v1/runs/attempt_02")
SAMPLING_DIR = Path("outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01")
FIT_ROWS, CONFIRM_ROWS, TOTAL_ROWS = 200, 300, 500
CLASS_COUNT, DRAW_COUNT, CHUNK_ROWS = 10, 30, 50
SEEDS = tuple(range(2026100961, 2026100966))
ENDPOINTS = ("A4700", "A5400", "B5400")
FOCUS_CLASSES = (2, 4)
DECISION_ATOL, DECISION_RTOL = 2.0e-7, 2.0e-5
REPLAY_RTOL, REPLAY_ATOL = 2.0e-6, 5.0e-8


class LocalizationError(RuntimeError):
    """Saved evidence violates the localization contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LocalizationError(f"cannot read saved JSON {path.name}: {error}") from error
    _require(isinstance(value, dict), f"saved JSON is not an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise LocalizationError(f"cannot read saved CSV {path.name}: {error}") from error


def _resolve(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _validate_output(root: Path, output: Path) -> None:
    outputs = (root / "outputs").resolve()
    resolved = output.resolve()
    _require(outputs in resolved.parents, "output must be below outputs/")
    parts = resolved.relative_to(outputs).parts
    _require(parts in {
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_01"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_02"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_03"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_04"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_05"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_06"),
        ("20261010_m0_generator_digit_identity_localization_v1", "runs", "attempt_07"),
    }, "output must be an explicitly named localization attempt")
    _require(not resolved.exists(), f"output already exists: {resolved}")


def _targets(raw_targets: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[int, torch.Tensor]]:
    result: dict[str, dict[int, torch.Tensor]] = {"fit": {}, "confirmation": {}}
    for role, start, stop, count in (("fit", 0, FIT_ROWS, 20), ("confirmation", FIT_ROWS, TOTAL_ROWS, 30)):
        for digit in range(CLASS_COUNT):
            positions = torch.tensor([i for i in range(start, stop) if int(labels[i]) == digit], dtype=torch.long)
            _require(positions.numel() == count, f"{role} target class {digit} row count differs")
            result[role][digit] = raw_targets.index_select(0, positions).to(torch.float32).contiguous()
    return result


def _vectorize(raw: torch.Tensor, size: int, view: str) -> torch.Tensor:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "image batch must be N×1×200×200")
    values = raw.to(device="cpu", dtype=torch.float64)
    _require(bool(torch.isfinite(values).all() and (values >= 0.0).all()), "image batch contains invalid intensity")
    if view == "display28":
        pooled = generation.capacity.canonical.cmos_view(raw).to(torch.float64)
    else:
        _require(view in {"shape28", "shape200"} and size in {28, 200}, "unknown localization view")
        pooled = torch.nn.functional.adaptive_avg_pool2d(values, (size, size))
    flattened = pooled.flatten(1)
    mass = flattened.sum(1)
    _require(bool(torch.isfinite(mass).all() and (mass > 0.0).all()), "image has nonpositive view mass")
    normalized = flattened / mass[:, None]
    _require(bool(torch.isfinite(normalized).all()), "normalized shape vectors are nonfinite")
    return normalized


def _mean_distance(left: torch.Tensor, right: torch.Tensor, *, row_chunk: int = 30) -> float:
    _require(left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1], "distance vectors have incompatible dimensions")
    _require(left.shape[0] > 0 and right.shape[0] > 0, "distance vectors cannot be empty")
    a, b = left.to(torch.float64), right.to(torch.float64)
    mode = "use_mm_for_euclid_dist" if left.shape[1] > 4096 else "donot_use_mm_for_euclid_dist"
    parts: list[float] = []
    for start in range(0, a.shape[0], row_chunk):
        distances = torch.cdist(a[start : start + row_chunk], b, p=2, compute_mode=mode)
        _require(bool(torch.isfinite(distances).all()), "Euclidean distances are nonfinite")
        parts.append(float(distances.sum()))
    return math.fsum(parts) / (left.shape[0] * right.shape[0])


def _energy_distance(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    cross = _mean_distance(x, y)
    within_x = _mean_distance(x, x)
    within_y = _mean_distance(y, y)
    value = 2.0 * cross - within_x - within_y
    _require(math.isfinite(value) and value >= -1.0e-10, "energy distance is invalid")
    return {"energy_distance": max(0.0, value), "cross_distance": cross, "within_output_distance": within_x, "within_target_distance": within_y}


def _class_margins(output: torch.Tensor, target_groups: Mapping[int, torch.Tensor], requested: int) -> tuple[float, int, dict[int, dict[str, float]]]:
    _require(requested in range(CLASS_COUNT), "requested class is outside 0–9")
    _require(set(target_groups) == set(range(CLASS_COUNT)), "all ten target groups are required")
    table = {digit: _energy_distance(output, target_groups[digit]) for digit in range(CLASS_COUNT)}
    margin, nearest, _competitors = _margin_from_distances(table, requested)
    requested_component = 2.0 * table[requested]["cross_distance"] - table[requested]["within_target_distance"]
    competitor_component = min(
        2.0 * table[digit]["cross_distance"] - table[digit]["within_target_distance"]
        for digit in range(CLASS_COUNT)
        if digit != requested
    )
    _require(math.isclose(margin, competitor_component - requested_component, rel_tol=0.0, abs_tol=1.0e-12), "output-within distance did not cancel from the class margin")
    return margin, nearest, table


def _margin_from_distances(distances: Mapping[int, Mapping[str, float]], requested: int) -> tuple[float, int, dict[int, float]]:
    _require(set(distances) == set(range(CLASS_COUNT)), "distance table does not contain all ten target classes")
    requested_distance = float(distances[requested]["energy_distance"])
    competitors = {digit: float(distances[digit]["energy_distance"]) for digit in range(CLASS_COUNT) if digit != requested}
    _require(len(competitors) == 9 and all(math.isfinite(value) for value in competitors.values()), "competitor distances are incomplete or nonfinite")
    nearest = min(competitors, key=lambda digit: (competitors[digit], digit))
    return competitors[nearest] - requested_distance, nearest, competitors


def _decision(rows: Sequence[Mapping[str, Any]], *, view_name: str = "shape28") -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] in {"A5400", "B5400"} and row["target_role"] == "confirmation" and int(row["requested_class"]) in FOCUS_CLASSES and row["view"] == view_name]
    expected = 2 * len(SEEDS) * len(FOCUS_CLASSES)
    _require(len(selected) == expected, f"primary contains {len(selected)} rows, expected {expected}")
    expected_keys = {(arm, seed, digit) for arm in ("A5400", "B5400") for seed in SEEDS for digit in FOCUS_CLASSES}
    actual_keys = {(str(row["arm"]), int(row["seed"]), int(row["requested_class"])) for row in selected}
    _require(actual_keys == expected_keys and len(actual_keys) == len(selected), "primary seed/class/arm allocation is incomplete or duplicated")
    means = {arm: math.fsum(float(row["margin"]) for row in selected if row["arm"] == arm) / (len(SEEDS) * len(FOCUS_CLASSES)) for arm in ("A5400", "B5400")}
    difference = means["B5400"] - means["A5400"]
    tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(means["A5400"]), abs(means["B5400"]))
    if difference > tolerance:
        classification = "CLASS_SEPARATION_INCREASED"
    elif difference < -tolerance:
        classification = "CLASS_SEPARATION_DECREASED"
    else:
        classification = "CLASS_SEPARATION_WITHIN_TOLERANCE"
    by_class = {str(digit): {arm: math.fsum(float(row["margin"]) for row in selected if int(row["requested_class"]) == digit and row["arm"] == arm) / len(SEEDS) for arm in ("A5400", "B5400")} for digit in FOCUS_CLASSES}
    return {"classification": classification, "view": view_name, "mean_margin": means, "difference_B_minus_A": difference, "tolerance": tolerance, "rows": len(selected), "seeds": list(SEEDS), "focus_classes": list(FOCUS_CLASSES), "by_class": by_class}


def _component_metrics(raw: torch.Tensor, targets_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    return foreground._measure_components(raw, targets_raw)


def _training_rows(training_dir: Path, population: Mapping[str, Any], source_readouts: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(_read_json(training_dir / "report.json").get("task_id") == training.TASK_ID, "training report task identity differs")
    _require(_read_json(training_dir / "report.json").get("status") == "VALID_COMPLETED", "training report is not validly complete")
    _require(_read_json(training_dir / "input_population.json") == population, "training population differs from source attempt_03")
    labels = torch.tensor(population["labels"], dtype=torch.long)
    roles = population["roles"]
    rows: list[dict[str, Any]] = []
    endpoint_readouts: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B"):
        scenario = f"{arm}5400"
        pieces: dict[str, list[torch.Tensor]] = {name: [] for name in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
        for chunk_index, start in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
            path = training_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt"
            _require(path.is_file(), f"{scenario} readout chunk is missing")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            stop = start + CHUNK_ROWS
            _require(payload.get("task_id") == training.TASK_ID and payload.get("scenario") == scenario, f"{scenario} chunk task/scenario differs")
            _require(torch.equal(payload.get("positions"), torch.arange(start, stop)), f"{scenario} chunk positions differ")
            _require(torch.equal(payload.get("source_row_indices"), torch.tensor(population["source_row_indices"][start:stop])), f"{scenario} chunk source rows differ")
            _require(torch.equal(payload.get("labels"), labels[start:stop]) and payload.get("roles") == roles[start:stop], f"{scenario} chunk labels or roles differ")
            for field in pieces:
                value = payload.get(field)
                _require(isinstance(value, torch.Tensor) and value.shape[0] == CHUNK_ROWS and bool(torch.isfinite(value).all()), f"{scenario} {field} is malformed")
                pieces[field].append(value.contiguous())
        result = {key: torch.cat(value) for key, value in pieces.items()}
        _require(tuple(result["raw_cmos"].shape) == (TOTAL_ROWS, 1, 200, 200), f"{scenario} raw geometry differs")
        _require(bool((result["raw_cmos"] >= 0.0).all() and (result["raw_cmos"].flatten(1).sum(1) > 0.0).all()), f"{scenario} raw mass is invalid")
        replayed = generation.capacity.canonical.cmos_view(result["raw_cmos"])
        _require(torch.allclose(replayed, result["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"{scenario} saved production view does not replay")
        _require(torch.equal(result["target_slm_intensity"], source_readouts["A4700"]["target_slm_intensity"]), f"{scenario} targets differ from the shared source")
        endpoint_readouts[scenario] = {
            "raw_cmos": result["raw_cmos"],
            "evaluator_view": replayed,
            "source_indices": torch.tensor(population["source_row_indices"], dtype=torch.long),
            "labels": labels,
            "target_slm_intensity": source_readouts["A4700"]["target_slm_intensity"],
            "plane_metadata": source_readouts["A4700"]["plane_metadata"],
        }
        component = _component_metrics(result["raw_cmos"], result["target_slm_intensity"])
        metric_pieces: dict[str, list[torch.Tensor]] = {"q28": [], "ssim28": []}
        for start in range(0, TOTAL_ROWS, CHUNK_ROWS):
            stop = start + CHUNK_ROWS
            _, measured = parent._energy_image_metrics(
                {"raw_cmos": result["raw_cmos"][start:stop], "target_slm_intensity": result["target_slm_intensity"][start:stop], "evaluator_view": replayed[start:stop], "labels": labels[start:stop], "plane_metadata": source_readouts["A4700"]["plane_metadata"]},
                {"population": {"role_positions": {"fit": [0, 0], "confirmation": [0, CHUNK_ROWS]}}},
            )
            metric_pieces["q28"].append(measured["brightness_normalized_stroke_error_Q"])
            metric_pieces["ssim28"].append(measured["original_target_ssim_28x28"])
        metrics = {key: torch.cat(value) for key, value in metric_pieces.items()}
        for index in range(TOTAL_ROWS):
            row: dict[str, Any] = {"position": index, "source_row_index": int(population["source_row_indices"][index]), "label": int(labels[index]), "role": roles[index], "endpoint": scenario}
            for name, values in component.items():
                row[name] = float(values[index])
            row["integrated_intensity"] = float(result["raw_cmos"][index].to(torch.float64).sum() * parent._pixel_area(source_readouts["A4700"], "camera_plane")[0])
            row["clip_fraction"] = float(((result["raw_cmos"][index] < 0.0) | (result["raw_cmos"][index] > 1.0)).to(torch.float64).mean())
            row["e_d_prediction"] = int(result["e_d_logits"][index].argmax())
            row["e_d_correct"] = int(row["e_d_prediction"] == row["label"])
            row["q28"] = float(metrics["q28"][index])
            row["ssim28"] = float(metrics["ssim28"][index])
            rows.append(row)
    return rows, endpoint_readouts


def _saved_source_rows(readout: Mapping[str, Any], population: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = readout["raw_cmos"]
    target = readout["target_slm_intensity"]
    evaluator_view = readout["evaluator_view"]
    labels = readout["labels"].to(torch.long)
    _require(tuple(raw.shape) == (TOTAL_ROWS, 1, 200, 200) and tuple(target.shape) == tuple(raw.shape), "A4700 source readout geometry differs")
    component = _component_metrics(raw, target)
    metric_pieces: dict[str, list[torch.Tensor]] = {"q28": [], "ssim28": []}
    for start in range(0, TOTAL_ROWS, CHUNK_ROWS):
        stop = start + CHUNK_ROWS
        _, measured = parent._energy_image_metrics(
            {"raw_cmos": raw[start:stop], "target_slm_intensity": target[start:stop], "evaluator_view": evaluator_view[start:stop], "labels": labels[start:stop], "plane_metadata": readout["plane_metadata"]},
            {"population": {"role_positions": {"fit": [0, 0], "confirmation": [0, CHUNK_ROWS]}}},
        )
        metric_pieces["q28"].append(measured["brightness_normalized_stroke_error_Q"])
        metric_pieces["ssim28"].append(measured["original_target_ssim_28x28"])
    metrics = {key: torch.cat(value) for key, value in metric_pieces.items()}
    area = parent._pixel_area(readout, "camera_plane")[0]
    rows = []
    for index in range(TOTAL_ROWS):
        row: dict[str, Any] = {"position": index, "source_row_index": int(population["source_row_indices"][index]), "label": int(labels[index]), "role": population["roles"][index], "endpoint": "A4700"}
        for name, values in component.items():
            row[name] = float(values[index])
        row["integrated_intensity"] = float(raw[index].to(torch.float64).sum() * area)
        row["clip_fraction"] = float(((raw[index] < 0.0) | (raw[index] > 1.0)).to(torch.float64).mean())
        row["e_d_prediction"] = int(readout["e_d_logits"][index].argmax())
        row["e_d_correct"] = int(row["e_d_prediction"] == row["label"])
        row["q28"] = float(metrics["q28"][index])
        row["ssim28"] = float(metrics["ssim28"][index])
        rows.append(row)
    return rows


def _validate_training_checkpoints(training_dir: Path, population: Mapping[str, Any], source_checkpoint: Path) -> dict[str, Any]:
    checkpoints = {}
    for arm in ("A", "B"):
        checkpoint = torch.load(training_dir / f"arm_{arm.lower()}_step5400.pt", map_location="cpu", weights_only=True)
        _require(checkpoint.get("schema") == training.CHECKPOINT_SCHEMA and checkpoint.get("task_id") == training.TASK_ID and checkpoint.get("arm") == arm and checkpoint.get("optimizer_step") == 5400 and checkpoint.get("completed_updates") == 5400, f"{arm}5400 checkpoint identity differs")
        _require(checkpoint.get("source_step") == 4700 and checkpoint.get("task_local_updates") == 700 and checkpoint.get("source_checkpoint") == source_checkpoint.as_posix(), f"{arm}5400 is not the named 700-step A4700 endpoint")
        _require(checkpoint.get("lambda") == training.FIXED_LAMBDA, f"{arm}5400 foreground objective weight differs")
        _require(checkpoint.get("source_population") == population and checkpoint.get("source_history_count") == 4700 and len(checkpoint.get("training_history", [])) == 700, f"{arm}5400 input population or update history differs")
        expected_shapes = population["source_parameter_shapes"]
        params = checkpoint.get("trainable_parameter_state", {})
        _require(set(params) == set(expected_shapes), f"{arm}5400 trainable update scope differs")
        for name, shape in expected_shapes.items():
            value = params[name]
            _require(tuple(value.shape) == tuple(shape) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"{arm}5400 trainable phase state differs: {name}")
        _require(set(checkpoint["frozen_runtime_state"]) == {"ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, f"{arm}5400 frozen runtime state differs")
        config = checkpoint.get("optimizer_config", {})
        _require(config == {"lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "history_reset": False}, f"{arm}5400 Adam configuration differs")
        optimizer = checkpoint.get("optimizer_state", {})
        states = optimizer.get("state", {})
        groups = optimizer.get("param_groups", [])
        _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, f"{arm}5400 Adam state scope differs")
        _require(float(states[0]["step"].item()) == 5400.0 and bool(torch.isfinite(states[0]["exp_avg"]).all() and torch.isfinite(states[0]["exp_avg_sq"]).all()), f"{arm}5400 Adam state is invalid")
        checkpoints[arm] = checkpoint
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    _require(training._tree_equal(checkpoints["A"]["frozen_runtime_state"], checkpoints["B"]["frozen_runtime_state"]) and training._tree_equal(checkpoints["A"]["frozen_runtime_state"], source["frozen_runtime_state"]), "A5400/B5400 frozen runtime state differs from shared A4700")
    return checkpoints


def _group_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def _empirical_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("q_raw200", "q_fg", "q_bg", "raw_mse", "raw_mse_fg", "raw_mse_bg", "raw_mse_fg_contribution", "raw_mse_bg_contribution", "foreground_area_fraction", "normalized_fg_underfill_mass", "normalized_fg_overfill_mass", "normalized_bg_generated_mass", "q28", "ssim28", "integrated_intensity", "clip_fraction")
    result = []
    for (endpoint, role, label), group in sorted(_group_rows(rows, ("endpoint", "role", "label")).items()):
        row: dict[str, Any] = {"endpoint": endpoint, "role": role, "label": label, "rows": len(group), "e_d_correct": sum(int(item["e_d_correct"]) for item in group)}
        for field in fields:
            values = [float(item[field]) for item in group]
            _require(all(math.isfinite(value) for value in values), f"nonfinite empirical summary field {field}")
            row[f"mean_{field}"] = math.fsum(values) / len(values)
        result.append(row)
    _require(len(result) == len(ENDPOINTS) * 2 * CLASS_COUNT and all(row["rows"] == (20 if row["role"] == "fit" else 30) for row in result), "empirical role/class summary allocation differs")
    return result


def _sample_class_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("margin", "own_class_distance", "nearest_class_distance", "own_cross_distance", "own_within_output_distance", "own_within_target_distance")
    result = []
    group_fields = ("arm", "target_role", "view", "requested_class")
    for (arm, role, view, digit), group in sorted(_group_rows(rows, group_fields).items()):
        row: dict[str, Any] = {"arm": arm, "target_role": role, "view": view, "requested_class": digit, "seed_groups": len(group)}
        for field in fields:
            row[f"mean_{field}"] = math.fsum(float(item[field]) for item in group) / len(group)
        nearest_counts: dict[str, int] = {}
        for item in group:
            nearest = str(item["nearest_class"])
            nearest_counts[nearest] = nearest_counts.get(nearest, 0) + 1
        row["nearest_class_counts"] = nearest_counts
        row["mean_distance_to_class_9"] = math.fsum(float(item["distance_class_9"]) for item in group) / len(group)
        result.append(row)
    return result


def _sample_operational_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("camera_integrated_intensity", "generator_input_power", "raw_clip_fraction")
    result = []
    for (arm, digit), group in sorted(_group_rows(rows, ("arm", "requested_class")).items()):
        item: dict[str, Any] = {"arm": arm, "requested_class": digit, "rows": len(group), "e_d_correct": sum(int(row["e_d_correct"]) for row in group)}
        for field in fields:
            values = [float(row[field]) for row in group]
            item[f"mean_{field}"] = math.fsum(values) / len(values)
            item[f"min_{field}"] = min(values)
            item[f"max_{field}"] = max(values)
        result.append(item)
    _require(len(result) == 2 * CLASS_COUNT and all(row["rows"] == len(SEEDS) * DRAW_COUNT for row in result), "sample operational role/class allocation differs")
    return result


def _human_descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == readability.PRIMARY_PAIR_COUNT, "historical human descriptive pair count differs")
    result: dict[str, Any] = {}
    for arm, answer_field in (("A5400", "A_answer"), ("B5400", "B_answer")):
        by_class = {}
        recognized_total = 0
        unreadable_total = 0
        for digit in range(CLASS_COUNT):
            group = [row for row in rows if int(row["requested_class"]) == digit]
            recognized = sum(str(row[answer_field]) == str(digit) for row in group)
            unreadable = sum(str(row[answer_field]) in {"无法辨认", "unreadable"} for row in group)
            by_class[str(digit)] = {"rows": len(group), "recognized_request": recognized, "unreadable": unreadable}
            recognized_total += recognized
            unreadable_total += unreadable
        result[arm] = {"rows": len(rows), "recognized_request": recognized_total, "unreadable": unreadable_total, "by_class": by_class}
    return result


def _sampled_decomposition(class_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(str(row["arm"]), int(row["seed"]), str(row["target_role"]), str(row["view"]), int(row["requested_class"])): row for row in class_rows}
    _require(len(keyed) == len(class_rows), "sampled class-distance rows contain duplicate identities")
    result = []
    for seed in SEEDS:
        for role in ("fit", "confirmation"):
            for view in ("shape28", "shape200", "display28"):
                for digit in range(CLASS_COUNT):
                    left = keyed.get(("A5400", seed, role, view, digit))
                    right = keyed.get(("B5400", seed, role, view, digit))
                    _require(left is not None and right is not None, "sampled class-distance pair is missing")
                    delta_s = float(right["own_class_distance"]) - float(left["own_class_distance"])
                    delta_cross = float(right["own_cross_distance"]) - float(left["own_cross_distance"])
                    delta_within_output = float(right["own_within_output_distance"]) - float(left["own_within_output_distance"])
                    residual = delta_s - (2.0 * delta_cross - delta_within_output)
                    _require(abs(residual) <= 1.0e-12, "sampled set-distance decomposition does not close")
                    _require(math.isclose(float(left["own_within_target_distance"]), float(right["own_within_target_distance"]), rel_tol=0.0, abs_tol=1.0e-14), "paired arms use different target-set within distances")
                    result.append({"seed": seed, "target_role": role, "view": view, "requested_class": digit, "delta_energy_distance_B_minus_A": delta_s, "delta_cross_distance_B_minus_A": delta_cross, "delta_within_output_distance_B_minus_A": delta_within_output, "decomposition_residual": residual})
    return result


def _focus_summary(empirical_rows: Sequence[Mapping[str, Any]], sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    empirical = [row for row in empirical_rows if row["target_role"] == "confirmation" and row["view"] == "shape28" and int(row["requested_class"]) in FOCUS_CLASSES]
    sampled = [row for row in sample_rows if row["target_role"] == "confirmation" and row["view"] == "shape28" and int(row["requested_class"]) in FOCUS_CLASSES]
    sampled_by_arm_class = []
    for arm in ("A5400", "B5400"):
        for digit in FOCUS_CLASSES:
            group = [row for row in sampled if row["arm"] == arm and int(row["requested_class"]) == digit]
            _require(len(group) == len(SEEDS), "focus sample summary lacks seed/class rows")
            sampled_by_arm_class.append({
                "arm": arm,
                "requested_class": digit,
                "seed_groups": len(group),
                "mean_margin": math.fsum(float(row["margin"]) for row in group) / len(group),
                "mean_own_class_distance": math.fsum(float(row["own_class_distance"]) for row in group) / len(group),
                "mean_nearest_class_distance": math.fsum(float(row["nearest_class_distance"]) for row in group) / len(group),
                "nearest_class_counts": {str(k): sum(int(row["nearest_class"]) == k for row in group) for k in range(CLASS_COUNT)},
                "mean_distance_to_class_9": math.fsum(float(row["distance_class_9"]) for row in group) / len(group),
            })
    _require(len(empirical) == len(ENDPOINTS) * len(FOCUS_CLASSES), "empirical focus distance summary is incomplete")
    return {"empirical_confirmation_shape28": empirical, "sampled_confirmation_shape28_by_arm_class": sampled_by_arm_class}


def _compare_training_report(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    parent_rows = report.get("endpoint_summaries")
    _require(isinstance(parent_rows, list), "training parent report summaries are incomplete")
    parent_rows = [row for row in parent_rows if row.get("scenario") in {"A5400", "B5400"}]
    _require(len(parent_rows) == 2 * 2 * CLASS_COUNT, "training parent report summaries are incomplete for measured endpoints")
    by_key = {(str(row["scenario"]), str(row["role"]), int(row["label"])): row for row in parent_rows}
    _require(len(by_key) == len(parent_rows), "training parent report summary identities are duplicated")
    grouped = _group_rows(rows, ("endpoint", "role", "label"))
    mapping = {
        "q28": "mean_production_q_28",
        "q_fg": "mean_q_fg",
        "q_bg": "mean_q_bg",
        "q_raw200": "mean_q_raw200",
        "raw_mse": "mean_raw_mse",
        "raw_mse_fg": "mean_raw_mse_fg",
        "raw_mse_bg": "mean_raw_mse_bg",
        "ssim28": "mean_ssim_28",
        "integrated_intensity": "mean_integral_intensity",
        "clip_fraction": "mean_clipped_fraction",
    }
    comparisons = 0
    for (endpoint, role, label), group in grouped.items():
        parent_row = by_key.get((str(endpoint), str(role), int(label)))
        _require(parent_row is not None and int(parent_row["rows"]) == len(group), "training parent report role/class rows differ")
        for field, parent_field in mapping.items():
            actual = math.fsum(float(row[field]) for row in group) / len(group)
            expected = float(parent_row[parent_field])
            _require(math.isclose(actual, expected, rel_tol=5.0e-6, abs_tol=1.0e-12), f"training report metric does not replay: {endpoint}/{role}/{label}/{field}")
            comparisons += 1
        correct = sum(int(row["e_d_correct"]) for row in group)
        _require(correct == int(parent_row["e_d_correct"]), f"training report E_D count does not replay: {endpoint}/{role}/{label}")
    return {"passed": True, "role_class_groups": len(grouped), "scalar_metric_comparisons": comparisons, "e_d_count_comparisons": len(grouped)}


def _compare_sample_report(class_rows: Sequence[Mapping[str, Any]], operational_rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    auxiliary = report.get("numerical_auxiliary")
    _require(isinstance(auxiliary, Mapping), "sampled parent report numerical auxiliary is missing")
    shape_reference = auxiliary["shape_statistics"]
    shape_comparisons = 0
    for role in ("fit", "confirmation"):
        for size, view in ((28, "shape28"), (200, "shape200")):
            for arm in ("A5400", "B5400"):
                group = [row for row in class_rows if row["arm"] == arm and row["target_role"] == role and row["view"] == view]
                _require(len(group) == len(SEEDS) * CLASS_COUNT, "sample class summaries do not match parent population")
                expected = shape_reference[role][str(size)][arm]
                for field, local_field in (("energy_distance", "own_class_distance"), ("cross_distance", "own_cross_distance"), ("within_output_distance", "own_within_output_distance"), ("within_target_distance", "own_within_target_distance")):
                    actual = math.fsum(float(row[local_field]) for row in group) / len(group)
                    _require(math.isclose(actual, float(expected[field]), rel_tol=2.0e-8, abs_tol=2.0e-11), f"sampled parent shape metric does not replay: {arm}/{role}/{size}/{field}")
                    shape_comparisons += 1
    ed_reference = auxiliary["E_D_auxiliary"]
    operational_reference = auxiliary["operational_auxiliary"]
    operational_comparisons = 0
    for arm in ("A5400", "B5400"):
        rows = [row for row in operational_rows if row["arm"] == arm]
        correct = sum(int(row["e_d_correct"]) for row in rows)
        _require(len(rows) == 1500 and correct == int(ed_reference[arm]["correct"]), f"sample E_D overall count does not replay: {arm}")
        for digit in range(CLASS_COUNT):
            class_rows = [row for row in rows if int(row["requested_class"]) == digit]
            expected_class = ed_reference[arm]["by_class"][str(digit)]
            _require(len(class_rows) == int(expected_class["rows"]) and sum(int(row["e_d_correct"]) for row in class_rows) == int(expected_class["correct"]), f"sample E_D class count does not replay: {arm}/{digit}")
        for field in ("camera_integrated_intensity", "generator_input_power", "raw_clip_fraction"):
            values = [float(row[field]) for row in rows]
            expected = operational_reference[arm][field]
            _require(math.isclose(math.fsum(values) / len(values), float(expected["mean"]), rel_tol=2.0e-8, abs_tol=1.0e-15), f"sample operational mean does not replay: {arm}/{field}")
            _require(math.isclose(min(values), float(expected["minimum"]), rel_tol=2.0e-8, abs_tol=1.0e-15), f"sample operational minimum does not replay: {arm}/{field}")
            _require(math.isclose(max(values), float(expected["maximum"]), rel_tol=2.0e-8, abs_tol=1.0e-15), f"sample operational maximum does not replay: {arm}/{field}")
            operational_comparisons += 3
    return {"passed": True, "shape_metric_comparisons": shape_comparisons, "operational_metric_comparisons": operational_comparisons, "e_d_counts_replayed": True}


def _targets_by_role(readout: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, dict[int, torch.Tensor]]:
    target_raw = readout["target_slm_intensity"]
    labels = torch.tensor(population["labels"], dtype=torch.long)
    return _targets(target_raw, labels)


def _roundtrip_decision(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    loaded = _read_csv(path)
    decision = _decision([{**row, "seed": int(row["seed"]) if row["seed"] != "empirical" else row["seed"], "requested_class": int(row["requested_class"]), "margin": float(row["margin"]), "target_role": row["target_role"], "view": row["view"]} for row in loaded])
    _require(decision["classification"] == expected["classification"], "CSV classification differs from machine decision")
    _require(math.isclose(decision["difference_B_minus_A"], float(expected["difference_B_minus_A"]), rel_tol=0.0, abs_tol=1e-14), "CSV main effect differs from machine decision")
    _require(math.isclose(decision["tolerance"], float(expected["tolerance"]), rel_tol=0.0, abs_tol=1e-14), "CSV tolerance differs from machine decision")
    return {"passed": True, "rows": len(loaded), "primary_rows": decision["rows"], "classification": decision["classification"], "difference_B_minus_A": decision["difference_B_minus_A"], "tolerance": decision["tolerance"]}


def _mean_distance_numpy(left: torch.Tensor, right: torch.Tensor, *, rows: int = 4) -> float:
    a, b = left.detach().cpu().numpy().astype(np.float64), right.detach().cpu().numpy().astype(np.float64)
    _require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] and a.shape[0] > 0 and b.shape[0] > 0, "NumPy reference vectors are invalid")
    partials = []
    for start in range(0, a.shape[0], rows):
        delta = a[start : start + rows, None, :] - b[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta, optimize=True))
        _require(np.isfinite(distances).all(), "NumPy reference distance is nonfinite")
        partials.append(float(distances.sum(dtype=np.float64)))
    return math.fsum(partials) / (a.shape[0] * b.shape[0])


def _numpy_primary(primary_vectors: Mapping[tuple[str, int, int], torch.Tensor], targets: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any]:
    rows = []
    for arm in ("A5400", "B5400"):
        for seed in SEEDS:
            for digit in FOCUS_CLASSES:
                output = primary_vectors[(arm, seed, digit)]
                distances = {}
                for target_class in range(CLASS_COUNT):
                    target = _vectorize(targets["confirmation"][target_class], 28, "shape28")
                    cross = _mean_distance_numpy(output, target)
                    within_output = _mean_distance_numpy(output, output)
                    within_target = _mean_distance_numpy(target, target)
                    value = 2.0 * cross - within_output - within_target
                    _require(math.isfinite(value) and value >= -1.0e-10, "NumPy reference energy distance is invalid")
                    distances[target_class] = {"energy_distance": max(0.0, value)}
                margin, nearest, _ = _margin_from_distances(distances, digit)
                rows.append({"arm": arm, "seed": seed, "requested_class": digit, "target_role": "confirmation", "view": "shape28", "margin": margin})
    return _decision(rows)


def _gallery(path: Path, rows: Sequence[Mapping[str, Any]], views: Mapping[tuple[Any, ...], torch.Tensor], titles: Mapping[str, str]) -> None:
    blocks = []
    for row in rows:
        endpoint = str(row["endpoint"])
        if "seed" in row:
            image = views[(endpoint, int(row["seed"]), int(row["label"]), int(row["draw_index"]))]
        else:
            image = views[(endpoint, int(row["position"]))]
        data, digest = readability._png_record(image[0])
        blocks.append(f"<figure><img src='{data}' alt='{digest}'><figcaption>{html.escape(titles.get(endpoint, endpoint))} · class {row['label']} · row {row.get('source_row_index', row.get('draw_index'))}</figcaption></figure>")
    style = "body{font-family:Arial,sans-serif;background:#eee;margin:16px}section{margin:24px 0}div{display:grid;grid-template-columns:repeat(5,232px);gap:12px}figure{margin:0;background:white;padding:4px}img{width:224px;height:224px;image-rendering:pixelated}figcaption{font-size:12px}"
    path.write_text(f"<!doctype html><meta charset='utf-8'><style>{style}</style><h1>{html.escape(path.stem)}</h1><div>{''.join(blocks)}</div>", encoding="utf-8", newline="\n")


def _preflight_samples(sampling_dir: Path, training_dir: Path, bank: torch.Tensor, population: Mapping[str, Any], source_readouts: Mapping[str, Mapping[str, Any]], targets: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any]:
    positions = []
    for digit in range(CLASS_COUNT):
        candidates = [index for index in range(FIT_ROWS, TOTAL_ROWS) if int(population["labels"][index]) == digit]
        candidates.sort(key=lambda index: int(population["source_row_indices"][index]))
        positions.append(candidates[0])
    for endpoint in ENDPOINTS:
        if endpoint == "A4700":
            raw = source_readouts[endpoint]["raw_cmos"].index_select(0, torch.tensor(positions))
            representative_rows = [(digit, position, raw[digit : digit + 1]) for digit, position in enumerate(positions)]
        else:
            scenario = endpoint
            cache: dict[int, Mapping[str, Any]] = {}
            representative_rows = []
            for digit, position in enumerate(positions):
                chunk_index, offset = divmod(position, CHUNK_ROWS)
                if chunk_index not in cache:
                    start = chunk_index * CHUNK_ROWS
                    stop = start + CHUNK_ROWS
                    payload = torch.load(training_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt", map_location="cpu", weights_only=True)
                    _require(payload.get("task_id") == training.TASK_ID and payload.get("scenario") == scenario, f"{scenario} representative chunk identity differs")
                    _require(torch.equal(payload.get("positions"), torch.arange(start, stop)), f"{scenario} representative chunk positions differ")
                    _require(torch.equal(payload.get("source_row_indices"), torch.tensor(population["source_row_indices"][start:stop])) and torch.equal(payload.get("labels"), torch.tensor(population["labels"][start:stop])) and payload.get("roles") == population["roles"][start:stop], f"{scenario} representative row identity differs")
                    raw_chunk, view_chunk = payload.get("raw_cmos"), payload.get("evaluator_view")
                    _require(isinstance(raw_chunk, torch.Tensor) and tuple(raw_chunk.shape) == (CHUNK_ROWS, 1, 200, 200), f"{scenario} representative raw chunk geometry differs")
                    _require(isinstance(view_chunk, torch.Tensor) and tuple(view_chunk.shape) == (CHUNK_ROWS, 1, 28, 28), f"{scenario} representative view geometry differs")
                    _require(bool(torch.isfinite(raw_chunk).all() and (raw_chunk >= 0.0).all() and (raw_chunk.flatten(1).sum(1) > 0.0).all()), f"{scenario} representative image quality differs")
                    _require(torch.allclose(generation.capacity.canonical.cmos_view(raw_chunk), view_chunk, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"{scenario} representative production view does not replay")
                    _require(torch.equal(payload.get("target_slm_intensity"), source_readouts["A4700"]["target_slm_intensity"][start:stop]), f"{scenario} representative targets differ from the shared source")
                    cache[chunk_index] = payload
                representative_rows.append((digit, position, cache[chunk_index]["raw_cmos"][offset : offset + 1]))
        for digit, position, image in representative_rows:
            output = _vectorize(image, 28, "shape28")
            target_vectors = {k: _vectorize(targets["confirmation"][k], 28, "shape28") for k in range(CLASS_COUNT)}
            margin, _, _ = _class_margins(output, target_vectors, digit)
            _require(math.isfinite(margin) and int(population["labels"][position]) == digit, f"{endpoint} representative identity/distance differs")

    seed = SEEDS[0]
    expected_bank = bank[0].reshape(300, 100)
    for arm in ("A5400", "B5400"):
        representatives: dict[int, torch.Tensor] = {}
        for chunk_index in range(6):
            payload = torch.load(sampling_dir / "readouts" / arm / f"seed_{seed}" / f"chunk_{chunk_index:04d}.pt", map_location="cpu", weights_only=True)
            start = chunk_index * CHUNK_ROWS
            _require(payload.get("task_id") == readability.TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed and payload.get("flat_start") == start, "representative sample chunk identity differs")
            _require(torch.equal(payload.get("input_z"), expected_bank[start : start + CHUNK_ROWS]), "representative OLS inputs differ from the saved joint bank")
            raw, view = payload.get("raw_cmos"), payload.get("evaluator_view")
            _require(isinstance(raw, torch.Tensor) and raw.shape[0] == CHUNK_ROWS and bool(torch.isfinite(raw).all() and (raw >= 0.0).all() and (raw.flatten(1).sum(1) > 0).all()), "representative raw sample is invalid")
            _require(torch.allclose(generation.capacity.canonical.cmos_view(raw), view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "representative sample view does not replay")
            for offset, record in enumerate(payload["rows"]):
                flat = start + offset
                digit, draw = divmod(flat, DRAW_COUNT)
                _require(record.get("label") == digit and record.get("draw_index") == draw and record.get("pair_id") == f"{seed}:{digit}:{draw}", "representative sample identity differs")
                if draw == 0:
                    representatives[digit] = raw[offset : offset + 1]
        _require(set(representatives) == set(range(CLASS_COUNT)), f"{arm} representatives do not cover all classes")
        for digit, raw in representatives.items():
            output = _vectorize(raw, 28, "shape28")
            target_vectors = {k: _vectorize(targets["confirmation"][k], 28, "shape28") for k in range(CLASS_COUNT)}
            margin, _, _ = _class_margins(output, target_vectors, digit)
            _require(math.isfinite(margin), f"{arm} representative margin is nonfinite")
    return {"passed": True, "rows": len(ENDPOINTS) * CLASS_COUNT + 2 * CLASS_COUNT, "empirical_rows": len(ENDPOINTS) * CLASS_COUNT, "sampled_rows": 2 * CLASS_COUNT, "seed": seed, "all_nine_competitors_measured": True, "production_views_replayed": True, "sampled_input_bank_matched": True}


def _run(artifact: Path, training_dir: Path, prior_dir: Path, sampling_dir: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    _validate_output(root, output)
    fixed = {"artifact_dir": _resolve(root, SOURCE_DIR), "training_dir": _resolve(root, TRAINING_DIR), "prior_dir": _resolve(root, PRIOR_DIR), "sampling_dir": _resolve(root, SAMPLING_DIR)}
    supplied = {"artifact_dir": artifact.resolve(), "training_dir": training_dir.resolve(), "prior_dir": prior_dir.resolve(), "sampling_dir": sampling_dir.resolve()}
    _require(supplied == fixed, "one or more inputs differ from the explicitly named attempts")
    output.mkdir(parents=True, exist_ok=False)
    validation: dict[str, Any] = {"task_id": TASK_ID, "status": "RUNNING", "execution": {"device": "cpu", "model_updates": 0, "optical_forwards": 0, "evaluator_forwards": 0, "gradients": 0, "sampling": 0, "human_ratings": 0}, "checks": {}}
    _write_json(output / "preflight.json", validation)
    try:
        population, source_report, source_readouts, _summaries, _source_metrics, source_checks = source_audit._load_and_verify(artifact)
        foreground._validate_population(population)
        labels = torch.tensor(population["labels"], dtype=torch.long)
        _require(source_report.get("status") == "VALID_COMPLETED", "source energy attempt is not complete")
        checks: dict[str, Any] = {"source_population_and_readouts": source_checks, "roles_and_all_ten_classes": True}
        source_readouts = {"A4700": source_readouts["A4700"]}
        targets = _targets_by_role(source_readouts["A4700"], population)

        prior_report = _read_json(prior_dir / "report.json")
        _require(prior_report.get("task_id") == prior_task.TASK_ID and prior_report.get("status") == "VALID_COMPLETED", "frozen joint-prior attempt is not complete")
        prior_inputs = readability._verify_saved_joint_bank(prior_dir, population)
        bank = prior_inputs["bank"]["joint"]
        checks["frozen_complex64_bank_reconstructs"] = True

        training_report = _read_json(training_dir / "report.json")
        _require(training_report.get("task_id") == training.TASK_ID and training_report.get("status") == "VALID_COMPLETED", "foreground training comparison is not validly complete")
        _require(_read_json(training_dir / "input_population.json") == population, "training population differs from source attempt_03")
        _validate_training_checkpoints(training_dir, population, artifact / "arm_a_step4700.pt")
        checks["A5400_B5400_same_frozen_runtime"] = True

        mapping = _read_json(sampling_dir / "private" / "review_mapping.json")["items"]
        _require(len(mapping) == readability.ITEM_COUNT, "archived review mapping row count differs")
        pair_rows = _read_csv(sampling_dir / "analysis" / "paired_rows.csv")
        _require(len(pair_rows) == readability.PRIMARY_PAIR_COUNT, "archived earliest-presentation pair count differs")
        required_pair_fields = {"seed", "requested_class", "draw_index", "A_item_id", "B_item_id", "A_answer", "B_answer"}
        _require(all(required_pair_fields.issubset(row) for row in pair_rows), "archived earliest-presentation pair columns differ")
        ratings_rows = _read_csv(sampling_dir / "ratings_submission.csv")
        _require(len(ratings_rows) == readability.ITEM_COUNT and len({row["item_id"] for row in ratings_rows}) == readability.ITEM_COUNT, "archived rating submission is incomplete")
        answer_by_item = {row["item_id"]: row["answer"] for row in ratings_rows}
        mapping_by_item = {row["item_id"]: row for row in mapping}
        _require(len(mapping_by_item) == readability.ITEM_COUNT and set(answer_by_item) == set(mapping_by_item), "archived rating submission does not match the private mapping")
        _require(all(answer in readability.ALLOWED_ANSWERS for answer in answer_by_item.values()), "archived rating file contains an invalid answer")
        human_answers: dict[tuple[int, int, int], dict[str, str]] = {}
        for row in pair_rows:
            key = (int(row["seed"]), int(row["requested_class"]), int(row["draw_index"]))
            _require(key not in human_answers, "archived human pair identity is duplicated")
            for arm in ("A5400", "B5400"):
                arm_prefix = arm[0]
                item_id = row[f"{arm_prefix}_item_id"]
                mapped = mapping_by_item.get(item_id)
                _require(mapped is not None and (int(mapped["seed"]), int(mapped["requested_class"]), int(mapped["draw_index"]), mapped["arm"]) == (*key, arm), "archived human item maps to a different seed/class/draw/arm")
                _require(answer_by_item[item_id] == row[f"{arm_prefix}_answer"], "archived paired answer differs from original submission")
            human_answers[key] = {"A5400": row["A_answer"], "B5400": row["B_answer"]}
        _require(len(human_answers) == 100 and all(sum(key[1] == digit for key in human_answers) == 10 for digit in range(CLASS_COUNT)), "archived human pair class allocation differs")
        review_status = _read_json(sampling_dir / "measurement_status.json")
        _require(review_status.get("state") == "COMPLETED" and review_status.get("review_quality_passed") is False, "historical reviewer quality state differs; do not reinterpret it")
        sample_report = _read_json(sampling_dir / "report.json")
        _require(sample_report.get("status") == "VALID_COMPLETED" and sample_report.get("human_results", {}).get("classification") == "REVIEW_INCONCLUSIVE", "fixed-prior sampled readability report or human conclusion differs")
        sample_inputs = _read_json(sampling_dir / "input_record.json")
        _require(Path(str(sample_inputs["checkpoint_A"])).resolve() == (training_dir / "arm_a_step5400.pt").resolve() and Path(str(sample_inputs["checkpoint_B"])).resolve() == (training_dir / "arm_b_step5400.pt").resolve(), "sampled outputs do not cite the named A5400/B5400 checkpoints")
        _require(Path(str(sample_inputs["source_attempt"])).resolve() == artifact.resolve() and Path(str(sample_inputs["training_attempt"])).resolve() == training_dir.resolve() and Path(str(sample_inputs["prior_attempt"])).resolve() == prior_dir.resolve(), "sampled output parent attempt identities differ")
        _require(sample_inputs.get("joint_bank_dtype") == "complex64" and sample_inputs.get("joint_bank_shape") == [5, 10, 30, 100] and sample_inputs.get("seeds") == list(SEEDS), "sampled bank configuration differs")
        _require(sample_inputs.get("population_source_rows") == population["source_row_indices"] and sample_inputs.get("roles") == population["roles"], "sampled input population identity or roles differ")
        checks["historical_ratings_linked_without_rescoring"] = True

        preflight = _preflight_samples(sampling_dir, training_dir, bank, population, source_readouts, targets)
        checks["representative_50_row_cpu_preflight"] = preflight
        validation["checks"] = checks
        validation["status"] = "READY"
        _write_json(output / "preflight.json", validation)

        training_rows, training_readouts = _training_rows(training_dir, population, source_readouts)
        training_parent_replay = _compare_training_report(training_rows, training_report)
        checks["training_parent_metrics_replayed"] = training_parent_replay
        empirical_rows = _saved_source_rows(source_readouts["A4700"], population) + training_rows

        empirical_class_rows: list[dict[str, Any]] = []
        empirical_focus_views: dict[tuple[str, int], torch.Tensor] = {}
        target_focus_views: dict[tuple[str, int], torch.Tensor] = {}
        empirical_readouts = {"A4700": source_readouts["A4700"], **training_readouts}
        for endpoint in ENDPOINTS:
            readout = empirical_readouts[endpoint]
            _require(torch.equal(readout["source_indices"], torch.tensor(population["source_row_indices"])) and torch.equal(readout["labels"], labels), f"{endpoint} source identity differs")
            for view_name in ("shape28", "shape200", "display28"):
                for role in ("fit", "confirmation"):
                    for digit in range(CLASS_COUNT):
                        selected = [i for i in range(TOTAL_ROWS) if population["roles"][i] == role and int(labels[i]) == digit]
                        raw = readout["raw_cmos"].index_select(0, torch.tensor(selected))
                        target_vectors = {k: _vectorize(targets[role][k], 200 if view_name == "shape200" else 28, view_name) for k in range(CLASS_COUNT)}
                        output_vectors = _vectorize(raw, 200 if view_name == "shape200" else 28, view_name)
                        distance_table = {k: _energy_distance(output_vectors, target_vectors[k]) for k in range(CLASS_COUNT)}
                        margin, nearest, distance_table = _class_margins(output_vectors, target_vectors, digit)
                        empirical_class_rows.append({"arm": endpoint, "seed": "empirical", "requested_class": digit, "target_role": role, "view": view_name, "output_rows": len(selected), "target_rows": len(targets[role][digit]), "margin": margin, "nearest_class": nearest, "own_class_distance": distance_table[digit]["energy_distance"], "nearest_class_distance": distance_table[nearest]["energy_distance"], **{f"distance_class_{k}": distance_table[k]["energy_distance"] for k in range(CLASS_COUNT)}, "own_cross_distance": distance_table[digit]["cross_distance"], "own_within_output_distance": distance_table[digit]["within_output_distance"], "own_within_target_distance": distance_table[digit]["within_target_distance"]})
                        if role == "confirmation" and digit in FOCUS_CLASSES and view_name == "display28":
                            for i in selected:
                                empirical_focus_views[(endpoint, i)] = readout["evaluator_view"][i:i+1].clone()
        for i in range(FIT_ROWS, TOTAL_ROWS):
            if int(labels[i]) in FOCUS_CLASSES:
                target_focus_views[("TARGET", i)] = generation.capacity.canonical.cmos_view(source_readouts["A4700"]["target_slm_intensity"][i:i+1]).clone()
        checks["empirical_all_classes_roles_and_views"] = len(empirical_class_rows) == len(ENDPOINTS) * CLASS_COUNT * 2 * 3

        sample_class_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        sample_focus_views: dict[tuple[str, int, int, int], torch.Tensor] = {}
        primary_vectors: dict[tuple[str, int, int], torch.Tensor] = {}
        sample_views = ("shape28", "shape200", "display28")
        for seed_index, seed in enumerate(SEEDS):
            seed_raw: dict[str, list[torch.Tensor]] = {arm: [] for arm in ("A5400", "B5400")}
            seed_display: dict[str, list[torch.Tensor]] = {arm: [] for arm in ("A5400", "B5400")}
            for chunk_index in range(6):
                start, stop = chunk_index * CHUNK_ROWS, (chunk_index + 1) * CHUNK_ROWS
                chunks = {}
                for arm in ("A5400", "B5400"):
                    path = sampling_dir / "readouts" / arm / f"seed_{seed}" / f"chunk_{chunk_index:04d}.pt"
                    _require(path.is_file(), f"sampled {arm} seed {seed} chunk is missing")
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    _require(payload.get("task_id") == readability.TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed and payload.get("flat_start") == start, "sample chunk identity differs")
                    _require(len(payload.get("rows", [])) == CHUNK_ROWS, "sample metadata rows are incomplete")
                    for offset, record in enumerate(payload["rows"]):
                        pos = start + offset
                        digit, draw = divmod(pos, DRAW_COUNT)
                        _require(record.get("label") == digit and record.get("draw_index") == draw and record.get("pair_id") == f"{seed}:{digit}:{draw}" and record.get("position") == seed_index * 300 + pos, "sample row identity or pair mapping differs")
                    raw = payload.get("raw_cmos")
                    view = payload.get("evaluator_view")
                    z = payload.get("input_z")
                    _require(isinstance(raw, torch.Tensor) and tuple(raw.shape) == (CHUNK_ROWS, 1, 200, 200), "sampled raw chunk geometry differs")
                    _require(isinstance(view, torch.Tensor) and tuple(view.shape) == (CHUNK_ROWS, 1, 28, 28), "sampled production view geometry differs")
                    _require(isinstance(z, torch.Tensor) and tuple(z.shape) == (CHUNK_ROWS, 100) and z.is_complex() and z.dtype == torch.complex64, "sampled OLS chunk geometry or dtype differs")
                    _require(bool(torch.isfinite(raw).all() and (raw >= 0.0).all() and (raw.flatten(1).sum(1) > 0.0).all()), "sampled raw intensities are invalid")
                    _require(torch.allclose(generation.capacity.canonical.cmos_view(raw), view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "sampled production view does not replay")
                    expected_z = bank[seed_index].reshape(300, 100)[start:stop]
                    _require(torch.equal(z, expected_z), f"{arm} sampled rows did not use the saved joint-bank inputs")
                    chunks[arm] = payload
                _require(torch.equal(chunks["A5400"]["input_z"], chunks["B5400"]["input_z"]), "A/B sampled OLS inputs differ")
                for arm in ("A5400", "B5400"):
                    payload = chunks[arm]
                    seed_raw[arm].append(payload["raw_cmos"].contiguous())
                    seed_display[arm].append(payload["evaluator_view"].contiguous())
                    for offset, record in enumerate(payload["rows"]):
                        pos = start + offset
                        digit, draw = divmod(pos, DRAW_COUNT)
                        flat = seed_index * 300 + pos
                        human = human_answers.get((seed, digit, draw), {})
                        row = {"arm": arm, "seed": seed, "position": flat, "requested_class": digit, "draw_index": draw, "pair_id": record["pair_id"], "target_role": "sampled_output", "e_d_prediction": int(record["e_d_prediction"]), "e_d_correct": int(record["e_d_correct"]), "camera_integrated_intensity": float(payload["camera_integrated_intensity"][offset]), "generator_input_power": float(payload["generator_input_power"][offset]), "raw_clip_fraction": float(payload["raw_clip_fraction"][offset]), "human_first_answer": human.get(arm, ""), "human_recognized_request": int(human.get(arm, "") == str(digit)) if human else ""}
                        sample_rows.append(row)
            for arm in ("A5400", "B5400"):
                raw = torch.cat(seed_raw[arm])
                display = torch.cat(seed_display[arm])
                _require(tuple(raw.shape) == (300, 1, 200, 200) and tuple(display.shape) == (300, 1, 28, 28), "sample seed concatenation geometry differs")
                for view_name in sample_views:
                    vectors = _vectorize(raw, 200 if view_name == "shape200" else 28, view_name)
                    for role in ("fit", "confirmation"):
                        target_vectors = {k: _vectorize(targets[role][k], 200 if view_name == "shape200" else 28, view_name) for k in range(CLASS_COUNT)}
                        for digit in range(CLASS_COUNT):
                            output_group = vectors[digit * DRAW_COUNT : (digit + 1) * DRAW_COUNT]
                            margin, nearest, table = _class_margins(output_group, target_vectors, digit)
                            sample_class_rows.append({"arm": arm, "seed": seed, "requested_class": digit, "target_role": role, "view": view_name, "output_rows": DRAW_COUNT, "target_rows": len(targets[role][digit]), "margin": margin, "nearest_class": nearest, "own_class_distance": table[digit]["energy_distance"], "nearest_class_distance": table[nearest]["energy_distance"], **{f"distance_class_{k}": table[k]["energy_distance"] for k in range(CLASS_COUNT)}, "own_cross_distance": table[digit]["cross_distance"], "own_within_output_distance": table[digit]["within_output_distance"], "own_within_target_distance": table[digit]["within_target_distance"]})
                            if role == "confirmation" and digit in FOCUS_CLASSES and view_name == "shape28":
                                primary_vectors[(arm, seed, digit)] = output_group.clone()
                    for digit in FOCUS_CLASSES:
                        for draw in range(DRAW_COUNT):
                            index = digit * DRAW_COUNT + draw
                            sample_focus_views[(arm, seed, digit, draw)] = display[index:index+1].clone()
            print(f"localized saved sampled shapes seed={seed}", flush=True)

        _require(len(empirical_rows) == 1500 and len(sample_rows) == 3000, "saved empirical or sampled rows are incomplete")
        primary = _decision(sample_class_rows)
        sample_parent_replay = _compare_sample_report(sample_class_rows, sample_rows, sample_report)
        checks["A_B_input_pairs_identical_and_bank_matched"] = True
        checks["all_4500_rows_finite_positive_and_views_replay"] = True
        checks["all_class_distance_tables_complete"] = len(sample_class_rows) == 2 * 5 * 10 * 2 * 3
        checks["sample_parent_metrics_replayed"] = sample_parent_replay
        validation["checks"] = checks
        empirical_summary = _empirical_summary(empirical_rows)
        sampled_summary = _sample_class_summary(sample_class_rows)
        sampled_operational = _sample_operational_summary(sample_rows)
        human_summary = _human_descriptive_summary(pair_rows)
        decomposition = _sampled_decomposition(sample_class_rows)
        focus_findings = _focus_summary(empirical_class_rows, sample_class_rows)
        focus_findings["empirical_confirmation_components"] = [row for row in empirical_summary if row["role"] == "confirmation" and int(row["label"]) in FOCUS_CLASSES]
        sensitivity_decisions = {view: _decision(sample_class_rows, view_name=view) for view in ("shape200", "display28")}

        empirical_path = output / "empirical_rows.csv"
        sample_path = output / "sampled_rows.csv"
        empirical_dist_path = output / "empirical_class_distances.csv"
        sample_dist_path = output / "sampled_class_distances.csv"
        _write_csv(empirical_path, empirical_rows)
        _write_csv(sample_path, sample_rows)
        _write_csv(empirical_dist_path, empirical_class_rows)
        _write_csv(sample_dist_path, sample_class_rows)
        _require(len(_read_csv(empirical_path)) == 1500 and len(_read_csv(sample_path)) == 3000, "row CSV roundtrip count differs")
        _require(len(_read_csv(empirical_dist_path)) == len(empirical_class_rows) and len(_read_csv(sample_dist_path)) == len(sample_class_rows), "class-distance CSV row count differs")
        roundtrip = _roundtrip_decision(sample_dist_path, primary)

        independent = _numpy_primary(primary_vectors, targets)
        _require(independent["classification"] == primary["classification"] and math.isclose(independent["difference_B_minus_A"], primary["difference_B_minus_A"], rel_tol=0.0, abs_tol=1e-12) and math.isclose(independent["tolerance"], primary["tolerance"], rel_tol=0.0, abs_tol=1e-14), "independent CSV decision differs")
        for digit in FOCUS_CLASSES:
            for arm in ("A5400", "B5400"):
                _require(math.isclose(independent["by_class"][str(digit)][arm], primary["by_class"][str(digit)][arm], rel_tol=0.0, abs_tol=1e-12), "independent per-class decision differs")

        ratings_path = output / "human_descriptive_pairs.csv"
        _write_csv(ratings_path, pair_rows)
        _require(len(_read_csv(ratings_path)) == 100, "historical descriptive human-pair count differs")

        empirical_gallery_rows = [row for row in empirical_rows if row["role"] == "confirmation" and row["label"] in FOCUS_CLASSES and row["endpoint"] in ENDPOINTS]
        empirical_gallery_rows.extend({"endpoint": "TARGET", "position": i, "source_row_index": int(population["source_row_indices"][i]), "label": int(labels[i])} for i in range(FIT_ROWS, TOTAL_ROWS) if int(labels[i]) in FOCUS_CLASSES)
        endpoint_order = {"TARGET": 0, "A4700": 1, "A5400": 2, "B5400": 3}
        empirical_gallery_rows.sort(key=lambda row: (int(row["source_row_index"]), endpoint_order[str(row["endpoint"])]) )
        _gallery(output / "empirical_focus_gallery.html", empirical_gallery_rows, {**empirical_focus_views, **target_focus_views}, {"TARGET": "原目标", "A4700": "A4700", "A5400": "A5400", "B5400": "B5400"})
        gallery_rows = [{"endpoint": arm, "label": digit, "seed": seed, "draw_index": draw, "position": 0} for seed in SEEDS for digit in FOCUS_CLASSES for draw in range(DRAW_COUNT) for arm in ("A5400", "B5400")]
        _gallery(output / "sampled_focus_gallery.html", gallery_rows, sample_focus_views, {"A5400": "A5400", "B5400": "B5400"})

        auxiliary_summary = {
            "empirical_by_endpoint_role_class": empirical_summary,
            "empirical_class_distances": empirical_class_rows,
            "sampled_by_arm_role_view_class": sampled_summary,
            "sampled_operational_by_arm_class": sampled_operational,
            "sampled_paired_distance_decomposition": decomposition,
            "historical_first_presentation_human_pairs": human_summary,
            "focus_findings": focus_findings,
            "primary_view_sensitivity_decisions_descriptive_only": sensitivity_decisions,
        }
        _write_json(output / "auxiliary_summary.json", auxiliary_summary)

        report = {"schema": "m0_generator_digit_identity_localization_report.v1", "task_id": TASK_ID, "status": "EVIDENCE_REVIEW_COMPLETED", "decision": primary, "view_sensitivity_decisions_descriptive_only": sensitivity_decisions, "focus_findings": focus_findings, "population": {"source_rows": TOTAL_ROWS, "fit_rows": FIT_ROWS, "confirmation_rows": CONFIRM_ROWS, "per_class_fit": 20, "per_class_confirmation": 30, "all_canonical_train": True}, "inputs": {"artifact_dir": str(artifact), "training_dir": str(training_dir), "prior_dir": str(prior_dir), "sampling_dir": str(sampling_dir), "endpoints": list(ENDPOINTS), "seeds": list(SEEDS)}, "auxiliary": {"empirical_distance_rows": len(empirical_class_rows), "sampled_distance_rows": len(sample_class_rows), "empirical_shape_and_operational_rows": len(empirical_rows), "sampled_operational_rows": len(sample_rows), "rating_rows_descriptive_only": 100, "views": ["shape28", "shape200", "display28"], "sampled_set_distance_decomposition": "delta_S = 2*delta_cross_distance - delta_within_output_distance; sampled output's within distance cancels from class-margin contrasts", "human_review_quality_state_retained": "REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED", "summary_file": "auxiliary_summary.json", "human_summary": human_summary}, "validation": {"checks": checks, "csv_roundtrip": roundtrip, "independent_decision_recheck": {"passed": True, "classification": independent["classification"], "effect": independent["difference_B_minus_A"], "tolerance": independent["tolerance"]}}, "training_recommendation": "NONE", "next_action": "NONE", "stop": True, "scientific_scope": "Saved endpoints and fixed samples on canonical TRAIN; geometric class separation is not a classifier or human readability validation."}
        (output / "experiment_plan.md").write_text(_render_plan(), encoding="utf-8", newline="\n")
        (output / "report.md").write_text(_render_report(report), encoding="utf-8", newline="\n")
        _write_json(output / "validation.json", {**validation, "status": "VALIDATED", "csv_roundtrip": roundtrip})
        _write_json(output / "report.json", report)
        _write_json(output / "completed.json", {"task_id": TASK_ID, "status": report["status"], "classification": primary["classification"], "training_recommendation": "NONE", "next_action": "NONE", "stop": True})
        return report
    except (LocalizationError, source_audit.CounterfactualError, foreground.ForegroundLocalizationError, readability.ReadabilityComparisonError, OSError, RuntimeError, ValueError) as error:
        validation["status"] = "INVALID_EVIDENCE"
        validation["stop_reason"] = f"{type(error).__name__}: {error}"
        _write_json(output / "validation.json", validation)
        result = {"task_id": TASK_ID, "status": "INVALID_EVIDENCE", "decision": None, "stop_reason": validation["stop_reason"], "training_recommendation": "NONE", "next_action": "NONE", "stop": True}
        _write_json(output / "report.json", result)
        _write_json(output / "completed.json", result)
        (output / "experiment_plan.md").write_text(_render_plan(), encoding="utf-8", newline="\n")
        (output / "report.md").write_text(_render_report(result), encoding="utf-8", newline="\n")
        return result


def _render_plan() -> str:
    return """# 类别2、4残余形态与类别分离CPU定位\n\n只读取 attempt_03 的原目标/人口/OLS、前景训练 attempt_01 的三个经验端点、冻结联合先验 attempt_02 的保存输入，以及固定先验采样 attempt_01 的A/B输出和已归档最早题号答案。\n\n主问题：B5400相对A5400的类别2、4采样输出，是否在接近请求类目标集合时也与其余九类拉开形态距离？图像统一使用未截断Pool28后逐图归一化；集合距离包含对角项。五种子、类别2/4等权。容差为2e-7+2e-5*max(|M_A|,|M_B|)。人审历史质控失败状态仅保留描述，不改变判定。\n\n另外分开报告经验解码/采样、fit/confirmation目标、28/200分辨率和生产显示截断视图，并保留前景/背景误差、能量、截断及E_D辅助结果。所有人口属于canonical TRAIN。\n\n输出定位报告和固定显示图库后停止；无训练、无人审、无新采样、无模型采用。training_recommendation=NONE，next_action=NONE。"""


def _render_report(report: Mapping[str, Any]) -> str:
    decision = report.get("decision")
    if decision is None:
        return "\n".join((f"# {TASK_ID}", "", "## 结果与状态", "保存证据核验失败，测量无效；没有科学判读。", "", "## 科学认识", "本轮没有形成有效测量，不支持或削弱类别分离解释。", "", "## 目标缺口", str(report.get("stop_reason", "证据不完整")), "", "## 工程完成与验证", "未生成可用于科学解释的完整读出分析。", "", "## 决策影响与下一步", "保持当前基线与人审质控终局；training_recommendation=NONE，next_action=NONE。", ""))
    focus = report["focus_findings"]
    empirical_distance_rows = focus["empirical_confirmation_shape28"]
    sampled_rows = focus["sampled_confirmation_shape28_by_arm_class"]
    component_rows = focus["empirical_confirmation_components"]
    human = report["auxiliary"]["human_summary"]
    empirical_lines = ["| 类别 | 端点 | 最近其他类 | 本类距离 | 分离量 |", "|---:|---|---:|---:|---:|"]
    for digit in FOCUS_CLASSES:
        for endpoint in ENDPOINTS:
            row = next(item for item in empirical_distance_rows if int(item["requested_class"]) == digit and item["arm"] == endpoint)
            empirical_lines.append(f"| {digit} | {endpoint} | {row['nearest_class']} | {row['own_class_distance']:.6g} | {row['margin']:.6g} |")
    sample_lines = ["| 类别 | 端点 | 最近其他类众数 | 本类距离 | 分离量 | 到类别9距离 |", "|---:|---|---:|---:|---:|---:|"]
    for digit in FOCUS_CLASSES:
        for arm in ("A5400", "B5400"):
            row = next(item for item in sampled_rows if int(item["requested_class"]) == digit and item["arm"] == arm)
            nearest = max(row["nearest_class_counts"].items(), key=lambda item: (item[1], -int(item[0])))[0]
            sample_lines.append(f"| {digit} | {arm} | {nearest}（{row['nearest_class_counts'][nearest]}/5） | {row['mean_own_class_distance']:.6g} | {row['mean_margin']:.6g} | {row['mean_distance_to_class_9']:.6g} |")
    component_lines = ["| 类别 | 端点 | Q前景 | Q背景 | Q总量200 | raw MSE | SSIM | E_D正确 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for digit in FOCUS_CLASSES:
        for endpoint in ("A4700", "A5400", "B5400"):
            row = next(item for item in component_rows if int(item["label"]) == digit and item["endpoint"] == endpoint)
            component_lines.append(f"| {digit} | {endpoint} | {row['mean_q_fg']:.6g} | {row['mean_q_bg']:.6g} | {row['mean_q_raw200']:.6g} | {row['mean_raw_mse']:.6g} | {row['mean_ssim28']:.6g} | {row['e_d_correct']}/{row['rows']} |")
    human_line = "; ".join(f"{arm}认可请求数字 {human[arm]['recognized_request']}/100（类别2/4分别 {human[arm]['by_class']['2']['recognized_request']}/10、{human[arm]['by_class']['4']['recognized_request']}/10）" for arm in ("A5400", "B5400"))
    sensitivity = report["view_sensitivity_decisions_descriptive_only"]
    sensitivity_line = "; ".join(f"{view}: D={value['difference_B_minus_A']:.6g}，{value['classification']}" for view, value in sensitivity.items())
    return "\n".join((
        f"# {TASK_ID}", "", "## 结果与状态",
        f"CPU保存证据定位完成。唯一主量`D_margin={decision['difference_B_minus_A']:.10g}`，容差`τ={decision['tolerance']:.10g}`，判读`{decision['classification']}`。A5400均值分离量`{decision['mean_margin']['A5400']:.10g}`，B5400为`{decision['mean_margin']['B5400']:.10g}`。",
        "", "## 科学认识",
        "本主量是形态空间里输出集合相对十类真实目标集合的距离差。正分离量表示本类目标集合比最近竞争类更近；它不是另训的分类器，也不是人眼识读指标。",
        "",
        "类别2、4采样主视图（确认人口、五种子等权）：",
        "| 类别 | 端点 | 分离量 | 本类距离 | 主要最近竞争类 |", "|---:|---|---:|---:|---|",
        *[f"| {digit} | {arm} | {decision['by_class'][str(digit)][arm]:.6g} | {next(item for item in sampled_rows if item['requested_class']==digit and item['arm']==arm)['mean_own_class_distance']:.6g} | {max(next(item for item in sampled_rows if item['requested_class']==digit and item['arm']==arm)['nearest_class_counts'].items(), key=lambda item: (item[1], -int(item[0])))[0]} |" for digit in FOCUS_CLASSES for arm in ("A5400", "B5400")],
        "",
        "类别2/4采样的最近竞争类与到类别9的辅助距离（类别4尤其需要关注2对9）：", *sample_lines,
        "逐种子、逐类的全部九个竞争类距离见`sampled_class_distances.csv`。经验解码的形态分离见：", *empirical_lines,
        "", "类别2、4经验解码端点的前景/背景辅助指标：", *component_lines,
        "", f"28×28主视图之外，视图敏感性仅作描述：{sensitivity_line}。",
        "", "已有单人评分只作定位线索：", human_line + "。实验质量检查仍为`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`，因为A端隐藏重复一致性13/20未达18/20；这些回答不改变本次数值判读。",
        "", "## 目标缺口", "本分析不能证明笔画拓扑、人眼可读性、E_D代理效度、采样覆盖/新颖性或独立泛化。经验与采样都使用canonical TRAIN；形态距离的改善也不意味着每张图都变好。类别距离仍可能受输出分布与多样性影响。",
        "", "## 工程完成与验证", f"读取并核验三个经验端点1500行、固定先验A/B采样3000行和100个历史配对答案。父训练报告关键指标复算、父采样的28/200形态统计与操作指标复算、CSV回读及NumPy独立主差核对{'均通过' if report['validation']['csv_roundtrip']['passed'] and report['validation']['independent_decision_recheck']['passed'] else '未全部通过'}。输入为相同保存complex64复OLS库；测量只用CPU保存强度，没有模型更新、光学/评价器前向、梯度、新采样或新增人审。图库包括原目标、A4700/A5400/B5400类别2/4确认图像，以及全部300组A/B采样配对。",
        "", "## 决策影响与下一步", "如果后续继续，应把结果仅用于挑选待验证假设：类别分离仍弱时先检验身份区分目标；若经验解码和采样表现不同，定位先验环节；若数值分离改善但人眼证据仍不足，先校验形态指标与具体身份缺陷的对应关系。这里是研究建议，不是训练或新增人审授权。保持A4700研究基线，B5400只保留研究候选，`training_recommendation=NONE`、`next_action=NONE`。",
        "",
    ))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--prior-dir", required=True, type=Path)
    parser.add_argument("--sampling-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    report = _run(_resolve(root, args.artifact_dir), _resolve(root, args.training_dir), _resolve(root, args.prior_dir), _resolve(root, args.sampling_dir), _resolve(root, args.output_dir))
    print(json.dumps({"status": report["status"], "decision": report.get("decision"), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2, allow_nan=False))
    if report["status"] != "EVIDENCE_REVIEW_COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
