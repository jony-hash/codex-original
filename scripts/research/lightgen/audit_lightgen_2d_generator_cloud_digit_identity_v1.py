"""Audit saved cloud foreground endpoints for digit-identity tradeoffs on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1 as source


TASK_ID = "M0_GENERATOR_CLOUD_DIGIT_IDENTITY_SAVED_EVIDENCE_V1"
DEFAULT_EVIDENCE = Path("outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9")
DEFAULT_OUTPUT = Path("outputs/20261013_m0_generator_cloud_digit_identity_saved_evidence_v1/runs/attempt_01")
TRAIN_SEEDS = source.TRAIN_SEEDS
SAMPLE_SEEDS = source.SAMPLE_SEEDS
CLASSES = tuple(range(10))
FOCUS_CLASSES = source.FOCUS_CLASSES
SCENARIOS = (
    "A4700",
    *(f"seed_{seed}_{arm}_step6700" for seed in TRAIN_SEEDS for arm in ("A", "B")),
)
SAMPLE_SCENARIOS = SCENARIOS
CHUNK_ROWS = source.CHUNK_ROWS
ANALYSIS_FIELDS = (
    "population", "training_seed", "scenario", "sample_seed", "requested_class", "draw_index", "source_row_index", "label",
    "q_fg", "q_bg", "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass",
    "raw_cmos_integral", "clipped_fraction", "generator_input_power",
    "e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction", "e_d_identity_margin",
    "e_s_correct", "e_i_correct", "e_c_correct", "e_d_correct",
)


class SavedEvidenceError(RuntimeError):
    """Saved inputs or their recomputed consumers violate the audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SavedEvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON input must be an object: {path.name}")
    return value


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(set(fields).issubset(reader.fieldnames or ()), f"CSV fields are incomplete: {path.name}")
        return list(reader)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    _require(json.loads(path.read_text(encoding="utf-8")) == value, f"JSON evidence failed readback: {path.name}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, replace: bool = False) -> None:
    _require(bool(rows), f"refusing to write an empty analysis table: {path.name}")
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    if path.exists() and not replace:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        _require(len(existing) == len(rows), f"refusing to replace different analysis row count: {path.name}")
        _require(tuple(existing[0]) == fields, f"refusing to replace different analysis columns: {path.name}")
        for expected, actual in zip(rows, existing, strict=True):
            _require(all(str(expected.get(key, "")) == actual[key] for key in fields), f"refusing to replace different analysis values: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        replay = list(csv.DictReader(stream))
    _require(len(replay) == len(rows) and tuple(replay[0]) == fields, f"analysis CSV failed row/field readback: {path.name}")
    for expected, actual in zip(rows, replay, strict=True):
        _require(all(str(expected.get(key, "")) == actual[key] for key in fields), f"analysis CSV failed value readback: {path.name}")


def _identity_margin(logits: torch.Tensor, requested: int) -> float:
    _require(logits.shape == (1, 10) and bool(torch.isfinite(logits).all()), "saved E_D logits are incomplete or invalid")
    competitor = torch.cat((logits[0, :requested], logits[0, requested + 1 :]))
    return float((logits[0, requested] - competitor.max()).item())


def _numpy_qfg(raw: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    image = raw.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    truth = target.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    _require(image.shape == truth.shape == (40_000,), "raw and target images must contain 40,000 pixels")
    _require(np.isfinite(image).all() and np.isfinite(truth).all() and (image >= 0).all() and (truth >= 0).all(), "raw or target image contains invalid intensity")
    image_mass = float(image.sum())
    target_mass = float(truth.sum())
    _require(image_mass > 0 and target_mass > 0, "raw or target image has zero total intensity")
    p_image, p_target = image / image_mass, truth / target_mass
    mask = truth > 0.1
    _require(mask.any() and (~mask).any(), "target foreground or background mask is empty")
    denominator = float(np.square(p_target).sum())
    _require(math.isfinite(denominator) and denominator > 0, "normalized target denominator is invalid")
    squared = np.square(p_image - p_target)
    q_fg = float(squared[mask].sum() / denominator)
    q_bg = float(squared[~mask].sum() / denominator)
    q_raw = float(squared.sum() / denominator)
    underfill = float(np.maximum(p_target - p_image, 0.0)[mask].sum())
    overfill = float(np.maximum(p_image - p_target, 0.0)[mask].sum())
    generated_background = float(p_image[~mask].sum())
    _require(math.isclose(q_fg + q_bg, q_raw, rel_tol=1e-12, abs_tol=1e-14), "NumPy Q_fg decomposition failed")
    return q_fg, q_bg, q_raw, underfill, overfill, generated_background


def _validate_parent(evidence: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, str]]], dict[tuple[str, int], dict[tuple[int, int], dict[str, str]]]]:
    report = _read_json(evidence / "report.json")
    validation = _read_json(evidence / "validation.json")
    population = _read_json(evidence / "input_population.json")
    _require(report.get("status") == "VALID_COMPLETED" and report.get("classification") == "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS", "parent report is not the named completed scientific result")
    _require(validation.get("status") == "VALID" and validation.get("scientific_updates") == 12_000, "parent validation is incomplete or has a different update count")
    decision = validation.get("decision_recomputed_from_saved_chunks")
    _require(isinstance(decision, dict) and decision.get("all_three_seeds_pass") is True, "parent saved decision record is not valid")
    _require(population.get("task_id") == source.TASK_ID and population.get("source_checkpoint_step") == source.SOURCE_STEP, "population source or checkpoint step differs")
    _require(population.get("dev_rows") == source.DEV_ROWS and population.get("dev_rows_per_class") == source.DEV_PER_CLASS, "DEV population dimensions differ")
    _require(set(int(row) for row in population.get("dev_selection", {}).get("source_row_indices", [])) == set(int(row) for row in population.get("source_row_indices", [])[source.FIT_ROWS :]), "DEV row identity differs from its declared population")
    dev_rows = _read_csv(evidence / "paired_rows.csv", source.DEV_FIELDS)
    sample_rows = _read_csv(evidence / "conditional_sample_rows.csv", source.SAMPLE_FIELDS)
    _require(len(dev_rows) == 9 * source.DEV_ROWS and len(sample_rows) == 7 * len(SAMPLE_SEEDS) * 10 * source.SAMPLES_PER_CLASS, "parent CSV row counts differ")
    dev_by_scenario: dict[str, list[dict[str, str]]] = {}
    for scenario in (*SCENARIOS, "historical_A5400", "historical_B5400"):
        rows = [row for row in dev_rows if row["scenario"] == scenario]
        if scenario in SCENARIOS:
            _index_dev_rows(rows)
            dev_by_scenario[scenario] = rows
    _require(set(row["scenario"] for row in dev_rows) == set((*SCENARIOS, "historical_A5400", "historical_B5400")), "DEV scenarios are missing or unexpected")
    sample_by_key: dict[tuple[str, int], dict[tuple[int, int], dict[str, str]]] = {}
    for scenario in SAMPLE_SCENARIOS:
        for seed in SAMPLE_SEEDS:
            rows = [row for row in sample_rows if row["scenario"] == scenario and int(row["sample_seed"]) == seed]
            sample_by_key[(scenario, seed)] = _index_sample_rows(rows, expected_per_class=source.SAMPLES_PER_CLASS)
    _require(set(row["scenario"] for row in sample_rows) == set(SAMPLE_SCENARIOS), "conditional sample scenarios are missing or unexpected")
    return report, validation, dev_by_scenario, sample_by_key


def _index_dev_rows(rows: Sequence[Mapping[str, str]]) -> dict[int, Mapping[str, str]]:
    _require(len(rows) == source.DEV_ROWS and {row["role"] for row in rows} == {"dev"}, "DEV role or row count differs")
    by_position = {int(row["position"]): row for row in rows}
    source_rows = [int(row["source_row_index"]) for row in rows]
    _require(set(by_position) == set(range(source.DEV_ROWS)), "DEV positions are missing or repeated")
    _require(len(source_rows) == len(set(source_rows)), "DEV source rows repeat")
    counts = {digit: sum(int(row["label"]) == digit for row in rows) for digit in CLASSES}
    _require(counts == {digit: source.DEV_PER_CLASS for digit in CLASSES}, "DEV class population differs")
    return by_position


def _index_sample_rows(rows: Sequence[Mapping[str, str]], *, expected_per_class: int) -> dict[tuple[int, int], Mapping[str, str]]:
    keys = [(int(row["requested_class"]), int(row["draw_index"])) for row in rows]
    expected = {(digit, draw) for digit in CLASSES for draw in range(expected_per_class)}
    _require(len(rows) == len(expected) and len(keys) == len(set(keys)) and set(keys) == expected, "conditional sample keys collide or are incomplete")
    return {(int(row["requested_class"]), int(row["draw_index"])): row for row in rows}


def _verify_logits_predictions(logits: torch.Tensor, predictions: Sequence[Any], evaluator: str) -> None:
    _require(logits.ndim == 2 and logits.shape[1] == 10 and logits.shape[0] == len(predictions) and bool(torch.isfinite(logits).all()), f"saved {evaluator} logits are incomplete")
    _require(torch.equal(logits.argmax(1).cpu(), torch.tensor([int(value) for value in predictions])), f"saved {evaluator} logits disagree with the per-row CSV")


def _shape_vectors(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "DEV raw chunk geometry differs")
    value = raw.to(device="cpu", dtype=torch.float64)
    _require(bool(torch.isfinite(value).all() and (value >= 0).all()), "DEV raw chunk contains invalid pixels")
    pooled = F.adaptive_avg_pool2d(value, (28, 28)).flatten(1)
    mass = pooled.sum(1)
    _require(bool((mass > 0).all()), "DEV shape has zero mass")
    return pooled / mass[:, None]


def _mean_distance(left: torch.Tensor, right: torch.Tensor, chunk_rows: int = 20) -> float:
    parts = []
    for start in range(0, left.shape[0], chunk_rows):
        distance = torch.cdist(left[start : start + chunk_rows], right, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        _require(bool(torch.isfinite(distance).all()), "DEV shape distance is nonfinite")
        parts.append(float(distance.sum()))
    return math.fsum(parts) / (left.shape[0] * right.shape[0])


def _energy_distance(output: torch.Tensor, target: torch.Tensor, target_within: float) -> dict[str, float]:
    cross = _mean_distance(output, target)
    within_output = _mean_distance(output, output)
    value = 2.0 * cross - within_output - target_within
    _require(math.isfinite(value) and value >= -1e-10, "DEV normalized shape energy distance is invalid")
    return {"energy_distance": max(0.0, value), "cross_distance": cross, "within_output_distance": within_output, "within_target_distance": target_within}


def _preflight(evidence: Path) -> dict[str, Any]:
    report, validation, dev_rows, sample_rows = _validate_parent(evidence)
    scenario = SCENARIOS[0]
    rows = sorted(dev_rows[scenario], key=lambda row: int(row["position"]))
    raw = torch.load(evidence / "readouts" / scenario / "chunk_0000.pt", map_location="cpu", weights_only=True)
    _require(raw.get("task_id") == source.TASK_ID and raw.get("scenario") == scenario and len(raw.get("positions", [])) == CHUNK_ROWS, "representative raw DEV chunk is incomplete")
    positions = raw["positions"].tolist()
    _require(positions == list(range(CHUNK_ROWS)) and raw.get("roles") == ["dev"] * CHUNK_ROWS, "representative raw DEV identity or role differs")
    view = diagnostic.canonical.cmos_view(raw["raw_cmos"])
    _require(torch.allclose(view, raw["evaluator_view"], rtol=source.REPLAY_RTOL, atol=source.REPLAY_ATOL), "production CMOS view does not replay from raw intensity")
    for evaluator in source.EVALUATOR_NAMES:
        _require(torch.equal(raw["logits"][evaluator].argmax(1), torch.tensor([int(rows[pos][f"{evaluator.lower()}_prediction"]) for pos in positions])), f"representative {evaluator} logits disagree with the CSV")
    sample_scenario, sample_seed = SCENARIOS[0], SAMPLE_SEEDS[0]
    sample_raw = torch.load(evidence / "readouts" / "conditional_samples" / sample_scenario / f"seed_{sample_seed}" / "chunk_0000.pt", map_location="cpu", weights_only=True)
    sample_view = diagnostic.canonical.cmos_view(sample_raw["raw_cmos"])
    _require(torch.allclose(sample_view, sample_raw["evaluator_view"], rtol=source.REPLAY_RTOL, atol=source.REPLAY_ATOL), "representative conditional CMOS view does not replay")
    return {
        "task_id": TASK_ID,
        "status": "READY",
        "science_updates": 0,
        "cloud_adam_updates": 0,
        "dev_rows": len(dev_rows[scenario]),
        "representative_dev_raw_rows": len(positions),
        "sample_rows": len(sample_rows[(sample_scenario, sample_seed)]),
        "representative_sample_raw_rows": len(sample_raw["rows"]),
        "parent_decision_valid": report.get("classification") == "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS" and validation["decision_recomputed_from_saved_chunks"]["all_three_seeds_pass"],
    }


def _scenario_rows(evidence: Path, scenario: str, csv_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    by_position = _index_dev_rows(csv_rows)
    records: list[dict[str, Any]] = []
    output_vectors: list[torch.Tensor] = []
    target_vectors: list[torch.Tensor] = []
    pixel_area = 1.0
    for chunk_number, start in enumerate(range(0, source.DEV_ROWS, CHUNK_ROWS)):
        path = evidence / "readouts" / scenario / f"chunk_{chunk_number:04d}.pt"
        _require(path.is_file(), f"raw DEV chunk is missing: {scenario}/{path.name}")
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        positions = list(range(start, min(start + CHUNK_ROWS, source.DEV_ROWS)))
        expected_positions = torch.tensor(positions, dtype=torch.long)
        _require(chunk.get("task_id") == source.TASK_ID and chunk.get("scenario") == scenario and torch.equal(chunk.get("positions"), expected_positions), f"raw DEV chunk identity or ordering differs: {scenario}/{chunk_number}")
        _require(chunk.get("roles") == ["dev"] * len(positions), f"raw DEV chunk contains a non-DEV role: {scenario}/{chunk_number}")
        expected_rows = torch.tensor([int(by_position[pos]["source_row_index"]) for pos in positions], dtype=torch.long)
        expected_labels = torch.tensor([int(by_position[pos]["label"]) for pos in positions], dtype=torch.long)
        _require(torch.equal(chunk["source_row_indices"], expected_rows) and torch.equal(chunk["labels"], expected_labels), f"raw DEV source rows or labels differ from CSV: {scenario}/{chunk_number}")
        _require(tuple(chunk["raw_cmos"].shape) == (len(positions), 1, 200, 200) and torch.isfinite(chunk["raw_cmos"]).all() and (chunk["raw_cmos"] >= 0).all(), f"raw DEV pixels are invalid: {scenario}/{chunk_number}")
        replay_view = diagnostic.canonical.cmos_view(chunk["raw_cmos"])
        _require(torch.allclose(replay_view, chunk["evaluator_view"], rtol=source.REPLAY_RTOL, atol=source.REPLAY_ATOL), f"saved evaluator view does not replay from raw: {scenario}/{chunk_number}")
        _require(tuple(chunk["target_slm_intensity"].shape) == tuple(chunk["raw_cmos"].shape), f"DEV target intensity geometry differs: {scenario}/{chunk_number}")
        shapes = _shape_vectors(chunk["raw_cmos"])
        target_shapes = _shape_vectors(chunk["target_slm_intensity"])
        output_vectors.append(shapes)
        target_vectors.append(target_shapes)
        logits = chunk["logits"]
        for evaluator in source.EVALUATOR_NAMES:
            expected_prediction = torch.tensor([int(by_position[pos][f"{evaluator.lower()}_prediction"]) for pos in positions])
            _verify_logits_predictions(logits[evaluator], expected_prediction.tolist(), evaluator)
        ed = logits["E_D"].to(torch.float64)
        for offset, position in enumerate(positions):
            row = by_position[position]
            label = int(row["label"])
            record = source._dev_row(
                chunk["raw_cmos"][offset : offset + 1], chunk["target_slm_intensity"][offset : offset + 1],
                {name: logits[name][offset : offset + 1] for name in source.EVALUATOR_NAMES},
                scenario=scenario, position=position, source_row=int(row["source_row_index"]), label=label, pixel_area_m2=1.0,
            )
            numpy_values = _numpy_qfg(chunk["raw_cmos"][offset], chunk["target_slm_intensity"][offset])
            for field, observed in zip(("q_fg", "q_bg", "q_raw200"), numpy_values[:3], strict=True):
                _require(math.isclose(float(record[field]), observed, rel_tol=1e-12, abs_tol=1e-14), f"independent NumPy {field} differs: {scenario}/{position}")
                _require(math.isclose(observed, float(row[field]), rel_tol=1e-12, abs_tol=1e-14), f"saved CSV {field} differs from raw recomputation: {scenario}/{position}")
            for field in ("e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction"):
                _require(int(record[field]) == int(row[field]), f"saved CSV {field} differs from logits: {scenario}/{position}")
            pixel_area = float(row["raw_cmos_integral"]) / float(chunk["raw_cmos"][offset].to(torch.float64).sum())
            q_fg, q_bg, q_raw, underfill, overfill, background = numpy_values
            record.update({
                "population": "empirical_dev",
                "training_seed": "shared" if scenario == "A4700" else int(scenario.split("_")[1]),
                "scenario": scenario,
                "sample_seed": "",
                "requested_class": label,
                "draw_index": "",
                "source_row_index": int(row["source_row_index"]),
                "label": label,
                "q_fg": q_fg,
                "q_bg": q_bg,
                "q_raw200": q_raw,
                "foreground_underfill_mass": underfill,
                "foreground_overfill_mass": overfill,
                "background_generated_mass": background,
                "raw_cmos_integral": float(chunk["raw_cmos"][offset].to(torch.float64).sum()) * pixel_area,
                "clipped_fraction": float((chunk["raw_cmos"][offset] > 1.0).to(torch.float64).mean()),
                "generator_input_power": "",
                "e_d_identity_margin": _identity_margin(ed[offset : offset + 1].to(torch.float32), label),
                "e_s_correct": int(int(row["e_s_prediction"]) == label),
                "e_i_correct": int(int(row["e_i_prediction"]) == label),
                "e_c_correct": int(int(row["e_c_prediction"]) == label),
                "e_d_correct": int(int(row["e_d_prediction"]) == label),
            })
            records.append({key: record[key] for key in ANALYSIS_FIELDS})
    _require(len(records) == source.DEV_ROWS, f"raw DEV readback row count differs: {scenario}")
    return records, torch.cat(output_vectors), torch.cat(target_vectors)


def _sample_scenario_rows(evidence: Path, scenario: str, seed: int, csv_rows: Mapping[tuple[int, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    directory = evidence / "readouts" / "conditional_samples" / scenario / f"seed_{seed}"
    records: list[dict[str, Any]] = []
    for chunk_number in range(6):
        path = directory / f"chunk_{chunk_number:04d}.pt"
        _require(path.is_file(), f"raw conditional chunk is missing: {scenario}/{seed}/{path.name}")
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        expected_start = chunk_number * CHUNK_ROWS
        expected_rows = 50 if chunk_number < 5 else 50
        metadata = chunk.get("rows")
        _require(chunk.get("task_id") == source.TASK_ID and chunk.get("scenario") == scenario and chunk.get("sample_seed") == seed and chunk.get("flat_start") == expected_start, f"conditional chunk identity differs: {scenario}/{seed}/{chunk_number}")
        _require(len(metadata) == expected_rows and len(chunk["raw_cmos"]) == expected_rows, f"conditional raw chunk row count differs: {scenario}/{seed}/{chunk_number}")
        _require(chunk["raw_cmos"].shape[1:] == (1, 200, 200), "conditional raw geometry differs")
        replay_view = diagnostic.canonical.cmos_view(chunk["raw_cmos"])
        _require(torch.allclose(replay_view, chunk["evaluator_view"], rtol=source.REPLAY_RTOL, atol=source.REPLAY_ATOL), f"conditional evaluator view does not replay from raw: {scenario}/{seed}/{chunk_number}")
        logits = chunk["logits"]
        keys: set[tuple[int, int]] = set()
        for name in source.EVALUATOR_NAMES:
            _require(tuple(logits[name].shape) == (expected_rows, 10) and torch.isfinite(logits[name]).all(), f"conditional {name} logits are incomplete")
        for offset, entry in enumerate(metadata):
            key = (int(entry["requested_class"]), int(entry["draw_index"]))
            _require(key not in keys, f"conditional draw key repeats within chunk: {scenario}/{seed}/{key}")
            keys.add(key)
            observed = csv_rows.get(key)
            _require(observed is not None, f"conditional draw is missing from CSV: {scenario}/{seed}/{key}")
            _require(int(observed["e_d_prediction"]) == int(logits["E_D"][offset].argmax()), f"conditional E_D prediction disagrees with saved logits: {scenario}/{seed}/{key}")
            for name in source.EVALUATOR_NAMES:
                _require(int(observed[f"{name.lower()}_prediction"]) == int(logits[name][offset].argmax()), f"conditional {name} prediction disagrees with saved logits: {scenario}/{seed}/{key}")
            label = key[0]
            row = {
                "population": "conditional_sample",
                "training_seed": "shared" if scenario == "A4700" else int(scenario.split("_")[1]),
                "scenario": scenario,
                "sample_seed": seed,
                "requested_class": label,
                "draw_index": key[1],
                "source_row_index": "",
                "label": label,
                "q_fg": "",
                "q_bg": "",
                "q_raw200": "",
                "foreground_underfill_mass": "",
                "foreground_overfill_mass": "",
                "background_generated_mass": "",
                "raw_cmos_integral": float(observed["raw_cmos_integral"]),
                "clipped_fraction": float(observed["clipped_fraction"]),
                "generator_input_power": float(observed["generator_input_power"]),
                "e_s_prediction": int(observed["e_s_prediction"]),
                "e_i_prediction": int(observed["e_i_prediction"]),
                "e_c_prediction": int(observed["e_c_prediction"]),
                "e_d_prediction": int(observed["e_d_prediction"]),
                "e_d_identity_margin": _identity_margin(logits["E_D"][offset : offset + 1], label),
                "e_s_correct": int(int(observed["e_s_prediction"]) == label),
                "e_i_correct": int(int(observed["e_i_prediction"]) == label),
                "e_c_correct": int(int(observed["e_c_prediction"]) == label),
                "e_d_correct": int(int(observed["e_d_prediction"]) == label),
            }
            records.append(row)
    _require(len(records) == 300, f"conditional readback does not contain 300 unique draws: {scenario}/{seed}")
    observed_keys = {(int(row["requested_class"]), int(row["draw_index"])) for row in records}
    _require(observed_keys == set(csv_rows), f"conditional raw and CSV draw key sets differ: {scenario}/{seed}")
    return records


def _class_separation(scenarios: Mapping[str, torch.Tensor], targets: Mapping[int, torch.Tensor], labels: torch.Tensor) -> list[dict[str, Any]]:
    target_within = {digit: _mean_distance(targets[digit], targets[digit]) for digit in CLASSES}
    summaries = []
    for scenario, vectors in scenarios.items():
        _require(labels.shape == (source.DEV_ROWS,) and torch.bincount(labels, minlength=10).tolist() == [source.DEV_PER_CLASS] * 10, "DEV labels do not define the fixed balanced target population")
        by_class = {digit: vectors[labels == digit] for digit in CLASSES}
        per_label: dict[int, dict[int, dict[str, float]]] = {}
        for label in CLASSES:
            output = by_class[label]
            per_label[label] = {
                digit: _energy_distance(output, targets[digit], target_within[digit])
                for digit in CLASSES
            }
        for requested in FOCUS_CLASSES:
            own = per_label[requested][requested]["energy_distance"]
            nearest = min((digit for digit in CLASSES if digit != requested), key=lambda digit: (per_label[requested][digit]["energy_distance"], digit))
            margin = per_label[requested][nearest]["energy_distance"] - own
            summaries.append({
                "scenario": scenario,
                "requested_class": requested,
                "own_class_energy_distance": own,
                "nearest_competing_class": nearest,
                "nearest_competing_energy_distance": per_label[requested][nearest]["energy_distance"],
                "class_separation_margin": margin,
                "own_cross_distance": per_label[requested][requested]["cross_distance"],
                "own_within_output_distance": per_label[requested][requested]["within_output_distance"],
                "own_within_dev_target_distance": target_within[requested],
            })
    return summaries


def _summaries(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_rows = []
    for population in ("empirical_dev", "conditional_sample"):
        subset = [row for row in records if row["population"] == population]
        for scenario in dict.fromkeys(str(row["scenario"]) for row in subset):
            scenario_rows = [row for row in subset if row["scenario"] == scenario]
            for digit in CLASSES:
                rows = [row for row in scenario_rows if int(row["requested_class"]) == digit]
                _require(bool(rows), f"class summary has no observations: {population}/{scenario}/{digit}")
                summary = {"population": population, "scenario": scenario, "requested_class": digit, "rows": len(rows)}
                for key in ("e_s_correct", "e_i_correct", "e_c_correct", "e_d_correct", "e_d_identity_margin", "raw_cmos_integral", "clipped_fraction", "generator_input_power", "q_fg", "q_bg", "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass"):
                    values = [float(row[key]) for row in rows if row[key] != ""]
                    if values:
                        summary[f"mean_{key}"] = math.fsum(values) / len(values)
                        if key == "e_d_identity_margin":
                            summary[f"min_{key}"] = min(values)
                            summary[f"max_{key}"] = max(values)
                            summary[f"std_{key}"] = float(np.std(np.asarray(values, dtype=np.float64)))
                            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
                                summary[f"p{int(quantile * 100):02d}_{key}"] = float(np.quantile(values, quantile))
                class_rows.append(summary)
    paired_rows = []
    transitions = []
    for population in ("empirical_dev", "conditional_sample"):
        subset = [row for row in records if row["population"] == population and row["scenario"] != "A4700"]
        grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
        for row in subset:
            if population == "empirical_dev":
                key = (int(row["training_seed"]), int(row["source_row_index"]))
            else:
                key = (int(row["training_seed"]), int(row["sample_seed"]), int(row["requested_class"]), int(row["draw_index"]))
            arm = "A" if "_A_" in str(row["scenario"]) else "B"
            _require(arm not in grouped.setdefault(key, {}), f"paired arm is duplicated: {population}/{key}/{arm}")
            grouped[key][arm] = row
        expected = source.DEV_ROWS * 3 if population == "empirical_dev" else 1_500 * 3
        _require(len(grouped) == expected and all(set(pair) == {"A", "B"} for pair in grouped.values()), f"paired {population} A/B population is incomplete")
        by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in TRAIN_SEEDS}
        for key, pair in grouped.items():
            seed = int(key[0])
            a, b = pair["A"], pair["B"]
            row = {"population": population, "training_seed": seed, "source_row_index": key[1] if population == "empirical_dev" else "", "sample_seed": key[1] if population == "conditional_sample" else "", "requested_class": int(a["requested_class"]), "draw_index": int(a["draw_index"]) if population == "conditional_sample" else ""}
            for metric in ("e_s_correct", "e_i_correct", "e_c_correct", "e_d_correct", "e_d_identity_margin", "raw_cmos_integral", "clipped_fraction", "generator_input_power", "q_fg", "q_bg", "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass"):
                for arm, value in (("A", a), ("B", b)):
                    row[f"{arm}_{metric}"] = value[metric]
                if a[metric] != "" and b[metric] != "":
                    row[f"delta_B_minus_A_{metric}"] = float(b[metric]) - float(a[metric])
            for evaluator in source.EVALUATOR_NAMES:
                a_prediction, b_prediction = int(a[f"{evaluator.lower()}_prediction"]), int(b[f"{evaluator.lower()}_prediction"])
                label = int(a["requested_class"])
                a_correct, b_correct = a_prediction == label, b_prediction == label
                outcome = "both_correct" if a_correct and b_correct else "A_correct_B_wrong" if a_correct else "A_wrong_B_correct" if b_correct else "both_wrong"
                transitions.append({"population": population, "training_seed": seed, "requested_class": label, "evaluator": evaluator, "A_prediction": a_prediction, "B_prediction": b_prediction, "outcome": outcome, "count": 1})
            by_seed[seed].append(row)
        for seed, rows in by_seed.items():
            summary = {
                "population": population,
                "training_seed": seed,
                "pairs": len(rows),
                "focus_class_pairs": sum(int(row["requested_class"]) in FOCUS_CLASSES for row in rows),
            }
            metrics = (
                "e_s_correct", "e_i_correct", "e_c_correct", "e_d_correct", "e_d_identity_margin",
                "raw_cmos_integral", "clipped_fraction", "generator_input_power", "q_fg", "q_bg",
                "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass",
            )
            for metric in metrics:
                field = f"delta_B_minus_A_{metric}"
                values = [float(row[field]) for row in rows if field in row]
                if values:
                    summary[f"mean_{field}"] = math.fsum(values) / len(values)
            focus_rows = [row for row in rows if int(row["requested_class"]) in FOCUS_CLASSES]
            summary["focus_class_pairs"] = len(focus_rows)
            focus_margin_deltas = [float(row["delta_B_minus_A_e_d_identity_margin"]) for row in focus_rows if "delta_B_minus_A_e_d_identity_margin" in row]
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
                if focus_margin_deltas:
                    summary[f"focus_p{int(quantile * 100):02d}_delta_B_minus_A_e_d_identity_margin"] = float(np.quantile(focus_margin_deltas, quantile))
            for metric in ("e_d_correct", "e_d_identity_margin", "raw_cmos_integral", "clipped_fraction", "generator_input_power", "q_fg", "q_bg", "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass"):
                for arm in ("A", "B"):
                    values = [float(row[f"{arm}_{metric}"]) for row in focus_rows if row[f"{arm}_{metric}"] != ""]
                    if values:
                        summary[f"focus_mean_{arm}_{metric}"] = math.fsum(values) / len(values)
            for evaluator in source.EVALUATOR_NAMES:
                key = f"{evaluator.lower()}_correct"
                summary[f"{key}_both_correct"] = sum(int(row[f"A_{key}"]) == 1 and int(row[f"B_{key}"]) == 1 for row in rows)
                summary[f"{key}_A_correct_B_wrong"] = sum(int(row[f"A_{key}"]) == 1 and int(row[f"B_{key}"]) == 0 for row in rows)
                summary[f"{key}_A_wrong_B_correct"] = sum(int(row[f"A_{key}"]) == 0 and int(row[f"B_{key}"]) == 1 for row in rows)
                summary[f"{key}_both_wrong"] = sum(int(row[f"A_{key}"]) == 0 and int(row[f"B_{key}"]) == 0 for row in rows)
                focus = focus_rows
                summary[f"focus_{key}_both_correct"] = sum(int(row[f"A_{key}"]) == 1 and int(row[f"B_{key}"]) == 1 for row in focus)
                summary[f"focus_{key}_A_correct_B_wrong"] = sum(int(row[f"A_{key}"]) == 1 and int(row[f"B_{key}"]) == 0 for row in focus)
                summary[f"focus_{key}_A_wrong_B_correct"] = sum(int(row[f"A_{key}"]) == 0 and int(row[f"B_{key}"]) == 1 for row in focus)
                summary[f"focus_{key}_both_wrong"] = sum(int(row[f"A_{key}"]) == 0 and int(row[f"B_{key}"]) == 0 for row in focus)
            paired_rows.append(summary)
    collapsed: dict[tuple[Any, ...], int] = {}
    for row in transitions:
        key = (row["population"], row["training_seed"], row["requested_class"], row["evaluator"], row["A_prediction"], row["B_prediction"], row["outcome"])
        collapsed[key] = collapsed.get(key, 0) + 1
    transition_rows = [{"population": key[0], "training_seed": key[1], "true_or_requested_class": key[2], "evaluator": key[3], "A_prediction": key[4], "B_prediction": key[5], "paired_outcome": key[6], "count": count} for key, count in sorted(collapsed.items(), key=lambda item: tuple(str(x) for x in item[0]))]
    return class_rows, paired_rows, transition_rows


def _focus_identity_pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        if row["scenario"] == "A4700" or int(row["requested_class"]) not in FOCUS_CLASSES:
            continue
        population = str(row["population"])
        seed = int(row["training_seed"])
        if population == "empirical_dev":
            key = (population, seed, int(row["source_row_index"]))
        else:
            key = (population, seed, int(row["sample_seed"]), int(row["requested_class"]), int(row["draw_index"]))
        arm = "A" if "_A_" in str(row["scenario"]) else "B"
        _require(arm not in grouped.setdefault(key, {}), f"focus identity pair repeats an arm: {key}/{arm}")
        grouped[key][arm] = row
    _require(all(set(pair) == {"A", "B"} for pair in grouped.values()), "focus identity rows do not form complete A/B pairs")
    output = []
    for key, pair in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        a, b = pair["A"], pair["B"]
        label = int(a["requested_class"])
        a_prediction, b_prediction = int(a["e_d_prediction"]), int(b["e_d_prediction"])
        a_correct, b_correct = a_prediction == label, b_prediction == label
        outcome = "both_correct" if a_correct and b_correct else "A_correct_B_wrong" if a_correct else "A_wrong_B_correct" if b_correct else "both_wrong"
        row = {
            "population": key[0], "training_seed": key[1],
            "source_row_index": key[2] if key[0] == "empirical_dev" else "",
            "sample_seed": key[2] if key[0] == "conditional_sample" else "",
            "requested_class": label,
            "draw_index": key[3] if key[0] == "conditional_sample" else "",
            "A_e_d_prediction": a_prediction, "B_e_d_prediction": b_prediction,
            "A_correct": int(a_correct), "B_correct": int(b_correct),
            "paired_outcome": outcome,
            "A_e_d_identity_margin": float(a["e_d_identity_margin"]),
            "B_e_d_identity_margin": float(b["e_d_identity_margin"]),
            "delta_B_minus_A_e_d_identity_margin": float(b["e_d_identity_margin"]) - float(a["e_d_identity_margin"]),
            "A_raw_cmos_integral": float(a["raw_cmos_integral"]),
            "B_raw_cmos_integral": float(b["raw_cmos_integral"]),
            "delta_B_minus_A_raw_cmos_integral": float(b["raw_cmos_integral"]) - float(a["raw_cmos_integral"]),
            "A_q_fg": a["q_fg"], "B_q_fg": b["q_fg"],
            "delta_B_minus_A_q_fg": float(b["q_fg"]) - float(a["q_fg"]) if a["q_fg"] != "" and b["q_fg"] != "" else "",
        }
        output.append(row)
    return output


def _verify_focus_identity_pair_readback(
    row_records: Sequence[Mapping[str, str]],
    saved_focus_pairs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    expected = _focus_identity_pairs(row_records)
    _require(len(expected) == len(saved_focus_pairs) == 1_500, "focus identity sample table does not cover all 1,500 predefined paired observations")
    _require(tuple(expected[0]) == tuple(saved_focus_pairs[0]), "focus identity table columns changed on readback")
    for left, right in zip(expected, saved_focus_pairs, strict=True):
        _require(all(str(left[key]) == right[key] for key in left), "focus identity table differs from paired source-row readback")
    return {"passed": True, "paired_observations": len(expected), "all_predefined_focus_rows_included": True}


def _validate_written_output(output: Path, expected_counts: Mapping[str, int]) -> dict[str, list[dict[str, str]]]:
    readback: dict[str, list[dict[str, str]]] = {}
    for filename, count in expected_counts.items():
        path = output / filename
        _require(path.is_file(), f"analysis output is missing on readback: {filename}")
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            _require(bool(reader.fieldnames), f"analysis output has no columns: {filename}")
            rows = list(reader)
        _require(len(rows) == count, f"analysis output row count changed on readback: {filename}")
        readback[filename] = rows
    return readback


def _verify_saved_pair_summaries(
    pair_rows: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
    transition_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, tuple[Any, ...]], dict[str, Mapping[str, str]]] = {}
    for row in pair_rows:
        population = row["population"]
        if row["scenario"] == "A4700":
            continue
        if population == "empirical_dev":
            key = (population, int(row["training_seed"]), (int(row["source_row_index"]),))
        elif population == "conditional_sample":
            key = (population, int(row["training_seed"]), (int(row["sample_seed"]), int(row["requested_class"]), int(row["draw_index"])))
        else:
            continue
        arm = "A" if "_A_" in row["scenario"] else "B"
        pair = grouped.setdefault(key, {})
        _require(arm not in pair, f"saved row readback repeats a paired arm: {population}/{key}/{arm}")
        pair[arm] = row
    _require(all(set(pair) == {"A", "B"} for pair in grouped.values()), "saved pair rows do not contain exactly one A and B for every pairing key")
    summary_index = {(row["population"], int(row["training_seed"])): row for row in summary_rows}
    _require(len(summary_index) == 6, "saved paired summary does not contain two populations for all three seeds")
    comparisons = 0
    for population in ("empirical_dev", "conditional_sample"):
        for seed in TRAIN_SEEDS:
            pairs = [pair for (item_population, item_seed, _), pair in grouped.items() if item_population == population and item_seed == seed]
            summary = summary_index[(population, seed)]
            expected_pairs = 1_000 if population == "empirical_dev" else 1_500
            _require(len(pairs) == expected_pairs and int(summary["pairs"]) == expected_pairs, f"saved pair count differs from its summary: {population}/{seed}")
            focus_pairs = [pair for pair in pairs if int(pair["A"]["requested_class"]) in FOCUS_CLASSES]
            _require(int(summary["focus_class_pairs"]) == len(focus_pairs), f"saved focus-class pair count differs: {population}/{seed}")
            for metric in ("e_d_correct", "e_d_identity_margin", "raw_cmos_integral", "clipped_fraction", "generator_input_power", "q_fg", "q_bg", "q_raw200", "foreground_underfill_mass", "foreground_overfill_mass", "background_generated_mass"):
                deltas = [float(pair["B"][metric]) - float(pair["A"][metric]) for pair in pairs if pair["A"][metric] != "" and pair["B"][metric] != ""]
                field = f"mean_delta_B_minus_A_{metric}"
                if deltas:
                    _require(field in summary and math.isclose(float(summary[field]), math.fsum(deltas) / len(deltas), rel_tol=1e-12, abs_tol=1e-14), f"saved paired mean differs from row readback: {population}/{seed}/{metric}")
                    comparisons += 1
                focus = [pair for pair in focus_pairs if pair["A"][metric] != "" and pair["B"][metric] != ""]
                for arm in ("A", "B"):
                    values = [float(pair[arm][metric]) for pair in focus if pair[arm][metric] != ""]
                    field = f"focus_mean_{arm}_{metric}"
                    if values:
                        _require(field in summary and math.isclose(float(summary[field]), math.fsum(values) / len(values), rel_tol=1e-12, abs_tol=1e-14), f"saved focus mean differs from row readback: {population}/{seed}/{metric}/{arm}")
                        comparisons += 1
            for evaluator in source.EVALUATOR_NAMES:
                key = f"{evaluator.lower()}_correct"
                for focus_only, selected in ((False, pairs), (True, focus_pairs)):
                    for outcome, predicate in (
                        ("both_correct", lambda a, b: int(a[key]) == int(b[key]) == 1),
                        ("A_correct_B_wrong", lambda a, b: int(a[key]) == 1 and int(b[key]) == 0),
                        ("A_wrong_B_correct", lambda a, b: int(a[key]) == 0 and int(b[key]) == 1),
                        ("both_wrong", lambda a, b: int(a[key]) == int(b[key]) == 0),
                    ):
                        expected = sum(predicate(pair["A"], pair["B"]) for pair in selected)
                        field = f"{'focus_' if focus_only else ''}{key}_{outcome}"
                        _require(int(summary[field]) == expected, f"saved paired outcome count differs from raw row readback: {population}/{seed}/{field}")
                        comparisons += 1
                observed_transition_count = sum(
                    int(row["count"])
                    for row in transition_rows
                    if row["population"] == population and int(row["training_seed"]) == seed and row["evaluator"] == evaluator
                )
                _require(observed_transition_count == expected_pairs, f"saved confusion transitions are incomplete: {population}/{seed}/{evaluator}")
    return {"passed": True, "summary_rows": len(summary_rows), "row_to_summary_comparisons": comparisons, "paired_identity_keys": len(grouped)}


def _identity_summary_from_readback(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    summary = []
    for row in rows:
        item: dict[str, Any] = {
            "population": row["population"],
            "training_seed": int(row["training_seed"]),
            "pairs": int(row["pairs"]),
            "focus_class_pairs": int(row["focus_class_pairs"]),
        }
        for field in (
            "mean_delta_B_minus_A_e_d_correct", "mean_delta_B_minus_A_e_d_identity_margin",
            "focus_mean_A_e_d_correct", "focus_mean_B_e_d_correct",
            "focus_mean_A_e_d_identity_margin", "focus_mean_B_e_d_identity_margin",
            "focus_e_d_correct_both_correct", "focus_e_d_correct_A_correct_B_wrong",
            "focus_e_d_correct_A_wrong_B_correct", "focus_e_d_correct_both_wrong",
        ):
            if field in row and row[field] != "":
                item[field] = float(row[field]) if "mean_" in field else int(row[field])
        if "focus_mean_A_e_d_identity_margin" in item and "focus_mean_B_e_d_identity_margin" in item:
            item["focus_delta_B_minus_A_e_d_identity_margin"] = item["focus_mean_B_e_d_identity_margin"] - item["focus_mean_A_e_d_identity_margin"]
        if "focus_mean_A_e_d_correct" in item and "focus_mean_B_e_d_correct" in item:
            item["focus_delta_B_minus_A_e_d_correct"] = item["focus_mean_B_e_d_correct"] - item["focus_mean_A_e_d_correct"]
        summary.append(item)
    return summary


def _verify_saved_report_consumers(
    report: Mapping[str, Any],
    identity_summary: Sequence[Mapping[str, Any]],
    q_fg_seed_decisions: Mapping[str, Any],
) -> dict[str, Any]:
    _require(report.get("paired_identity_summary_from_saved_csv") == list(identity_summary), "saved report differs from the identity summary read from CSV")
    _require(report.get("q_fg_seed_decisions_replayed") == dict(q_fg_seed_decisions), "saved report differs from the raw-replayed parent Q_fg decisions")
    _require(set(q_fg_seed_decisions) == {str(seed) for seed in TRAIN_SEEDS}, "raw-replayed parent Q_fg decisions do not cover all training seeds")
    _require(all(bool(item["passes_both_conditions"]) for item in q_fg_seed_decisions.values()), "raw-replayed parent Q_fg decisions do not preserve the three-seed result")
    _require(report.get("parent_q_fg_decision_replayed_from_raw") is True, "saved report does not record the raw parent decision replay")
    _require(report.get("parent_classification_preserved") == "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS", "saved report changed the parent classification")
    return {"passed": True, "identity_summary_rows": len(identity_summary), "q_fg_seed_decisions": len(q_fg_seed_decisions)}


def _next_step_from_descriptive_summary(identity_summary: Sequence[Mapping[str, Any]], shape_rows: Sequence[Mapping[str, Any]]) -> str:
    by_population: dict[str, list[Mapping[str, Any]]] = {}
    for row in identity_summary:
        by_population.setdefault(str(row["population"]), []).append(row)
    empirical = by_population["empirical_dev"]
    sampled = by_population["conditional_sample"]
    empirical_margin = math.fsum(float(row["focus_delta_B_minus_A_e_d_identity_margin"]) for row in empirical) / len(empirical)
    empirical_accuracy = math.fsum(float(row["focus_delta_B_minus_A_e_d_correct"]) for row in empirical) / len(empirical)
    sample_margin = math.fsum(float(row["focus_delta_B_minus_A_e_d_identity_margin"]) for row in sampled) / len(sampled)
    sample_accuracy = math.fsum(float(row["focus_delta_B_minus_A_e_d_correct"]) for row in sampled) / len(sampled)
    if empirical_margin > 0 and empirical_accuracy > 0 and sample_margin > 0 and sample_accuracy > 0:
        return "both_empirical_and_sampled_focus_class_identity_metrics_improve: review the fixed readability-method pilot as the next measurement step"
    if empirical_margin > 0 and empirical_accuracy > 0 and (sample_margin <= 0 or sample_accuracy <= 0):
        return "empirical_identity_improves_but_sampled_identity_does_not: prioritize the saved OLS-prior versus empirical-distribution comparison"
    shape_delta = []
    for seed in TRAIN_SEEDS:
        a_rows = [row for row in shape_rows if str(row["scenario"]) == f"seed_{seed}_A_step6700"]
        b_rows = [row for row in shape_rows if str(row["scenario"]) == f"seed_{seed}_B_step6700"]
        if a_rows and b_rows:
            shape_delta.append(math.fsum(float(row["class_separation_margin"]) for row in b_rows) / len(b_rows) - math.fsum(float(row["class_separation_margin"]) for row in a_rows) / len(a_rows))
    own_distance_deltas = []
    for seed in TRAIN_SEEDS:
        a_rows = [row for row in shape_rows if str(row["scenario"]) == f"seed_{seed}_A_step6700"]
        b_rows = [row for row in shape_rows if str(row["scenario"]) == f"seed_{seed}_B_step6700"]
        if a_rows and b_rows:
            own_distance_deltas.append(math.fsum(float(row["own_class_energy_distance"]) for row in b_rows) / len(b_rows) - math.fsum(float(row["own_class_energy_distance"]) for row in a_rows) / len(a_rows))
    focus_identity_not_improved = all(float(row["focus_delta_B_minus_A_e_d_correct"]) <= 0 and float(row["focus_delta_B_minus_A_e_d_identity_margin"]) <= 0 for row in (*empirical, *sampled))
    if own_distance_deltas and math.fsum(own_distance_deltas) / len(own_distance_deltas) < 0 and focus_identity_not_improved:
        return "DEV_shape_distance_improves_while_focus_identity_proxies_do_not: inspect class-specific confusion and critical strokes, then design one controlled objective diagnostic before any readability scoring"
    if shape_delta and math.fsum(shape_delta) / len(shape_delta) > 0 and empirical_accuracy <= 0 and sample_accuracy <= 0:
        return "shape_separation_improves_without_focus_identity_gain: inspect confusion and foreground localization before considering a single-factor target diagnostic"
    if any(float(row.get("mean_delta_B_minus_A_e_d_correct", 0.0)) < 0 for row in identity_summary):
        return "identity_or_class_tradeoffs_remain: review class-specific intensity and confusion tables before designing a controlled counterfactual"
    return "identity changes are mixed or small: use the paired tables to choose one bounded follow-up; do not add training steps automatically"


def _task_plan(evidence: Path) -> str:
    return (
        "# Saved cloud foreground identity audit\n\n"
        f"Task: `{TASK_ID}`. Status: diagnostic-only.\n\n"
        f"Evidence: `{evidence.as_posix()}`.\n\n"
        "This task replays the prior Q_fg decision from saved DEV raw chunks and describes paired evaluator, foreground/background, intensity, and shape measurements. "
        "It performs no model update, new sampling, human rating, new scientific gate, or model selection. E_D and geometric distances are diagnostic proxies, not readability evidence.\n\n"
        "The proposed follow-up is a fixed 160-question readability-method pilot; it is not executed by this task.\n"
    )


def _readability_pilot_plan() -> str:
    return """# 固定样本识读方法试点方案

用途：检查人工识读测量流程是否稳定，并描述固定A/B样本上的配对表现。此试点不验证模型采用，也不作为正式效应检验。

## 固定配对样本

60组A/B，共120张：训练种子2026101201、2026101202、2026101203 × 类别2、4 × 采样种子2026101221至2026101225 × draw 0、1。使用全部预先指定组合；不按图像质量筛选，也不挑训练种子。样本来自已保存的共同条件采样库。配对只比较对应同一采样种子、类别和draw的A/B图像。

## 控题与展示

- 10张真实数字正控：类别2与4各5张，来源固定DEV行；选择前五行以源行号排序，不再抽样。
- 6张全黑控题。
- 隐藏重复24张：A、B各12张，从上述60组按训练种子和类别轮转取样；每臂每种子每类2张。重复图使用同一像素内容，随机改变题目位置。
- 总计160题。所有图像先按生产28×28 CMOS视图显示，再用固定灰度和最近邻放大至224×224；展示图仅保留统一边框，不显示端点、类别、随机种子或题目来源。
- A/B顺序和题序由固定的展示清单随机化；评分分析消费每张原始图第一次出现的回答，重复题只用于质控。

## 建议的试点质控

正控至少9/10正确；全黑控6/6识别为空白；A、B重复题分别至少11/12一致。未达到任一项时，试点结论标为测量无效或不稳定，不解释为模型科学失败。阈值只判断方法是否可用，不判断A/B谁更好。

配对描述同时报告A对/B错、A错/B对、两者均对、两者均错，以及每类和每训练种子的结果。试点不作显著性通过判定，也不据其筛选终点。若需正式识读效应检验，另根据最小有意义差异和成对不一致率确定样本量与唯一主分析。

执行前才制作匿名展示清单、随机顺序和答案映射；本文件不含样本或答案映射，当前不启动评分。
"""


def _execute(evidence: Path, output: Path) -> dict[str, Any]:
    evidence = evidence.resolve()
    output = output.resolve()
    root = Path(__file__).resolve().parents[3]
    _require(evidence.is_dir() and root in evidence.parents, "evidence directory must be an explicit repository-local directory")
    expected_output = (root / DEFAULT_OUTPUT).resolve()
    _require(output == expected_output, "output directory differs from the task's fixed attempt path")
    _require(not (output / "report.json").is_file(), "completed saved-evidence attempt already exists")
    preflight = _preflight(evidence)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "task_plan.md"
    plan = _task_plan(evidence)
    if plan_path.exists():
        _require(plan_path.read_text(encoding="utf-8") == plan, "existing output belongs to a different evidence input")
    else:
        plan_path.write_text(plan, encoding="utf-8", newline="\n")
    _write_json(output / "preflight.json", {**preflight, "evidence_dir": evidence.as_posix()})
    report, validation, dev_csv, sample_csv = _validate_parent(evidence)
    all_rows: list[dict[str, Any]] = []
    scenario_vectors: dict[str, torch.Tensor] = {}
    reference_targets: dict[int, list[torch.Tensor]] = {digit: [] for digit in CLASSES}
    q_means: dict[str, float] = {}
    dev_labels = torch.tensor([int(row["label"]) for row in dev_csv["A4700"]], dtype=torch.long)
    reference_target_vectors: torch.Tensor | None = None
    for scenario in SCENARIOS:
        rows, vectors, target_vectors = _scenario_rows(evidence, scenario, dev_csv[scenario])
        all_rows.extend(rows)
        scenario_vectors[scenario] = vectors
        if scenario == "A4700":
            labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
            reference_target_vectors = target_vectors
            for digit in CLASSES:
                reference_targets[digit] = [target_vectors[labels == digit]]
        else:
            _require(reference_target_vectors is not None and torch.equal(target_vectors, reference_target_vectors), f"DEV target images change across endpoints: {scenario}")
        if scenario != "A4700":
            selected = [row for row in rows if int(row["label"]) in FOCUS_CLASSES]
            q_means[scenario] = math.fsum(float(row["q_fg"]) for row in selected) / len(selected)
    for seed in TRAIN_SEEDS:
        a = q_means[f"seed_{seed}_A_step6700"]
        b = q_means[f"seed_{seed}_B_step6700"]
        original = validation["decision_recomputed_from_saved_chunks"]["seed_decisions"][str(seed)]
        _require(math.isclose(a, original["A"], rel_tol=0, abs_tol=1e-12) and math.isclose(b, original["B"], rel_tol=0, abs_tol=1e-12), f"saved-raw Q_fg decision differs from the parent report for seed {seed}")
        _require(b - a < -(source.DECISION_ATOL + source.DECISION_RTOL * max(abs(a), abs(b))) and b < original["Q0"], f"original Q_fg decision does not replay for seed {seed}")
    for scenario in SAMPLE_SCENARIOS:
        for seed in SAMPLE_SEEDS:
            all_rows.extend(_sample_scenario_rows(evidence, scenario, seed, sample_csv[(scenario, seed)]))
    shape_targets = {digit: torch.cat(reference_targets[digit]) for digit in CLASSES}
    shape_summary = _class_separation(scenario_vectors, shape_targets, dev_labels)
    class_rows, paired_rows, transitions = _summaries(all_rows)
    focus_pairs = _focus_identity_pairs(all_rows)
    _require(len(focus_pairs) == 1_500, "focus identity pair table must contain every predefined class-2/4 paired observation")
    _write_csv(output / "paired_rows.csv", all_rows, replace=True)
    _write_csv(output / "class_summary.csv", class_rows, replace=True)
    _write_csv(output / "paired_seed_summary.csv", paired_rows, replace=True)
    _write_csv(output / "confusion_transitions.csv", transitions, replace=True)
    _write_csv(output / "dev_shape_separation.csv", shape_summary, replace=True)
    _write_csv(output / "focus_identity_pairs.csv", focus_pairs, replace=True)
    readback = _validate_written_output(output, {"paired_rows.csv": 17_500, "class_summary.csv": 140, "paired_seed_summary.csv": 6, "confusion_transitions.csv": len(transitions), "dev_shape_separation.csv": 14, "focus_identity_pairs.csv": 1_500})
    summary_readback = _verify_saved_pair_summaries(
        readback["paired_rows.csv"], readback["paired_seed_summary.csv"], readback["confusion_transitions.csv"],
    )
    focus_pair_readback = _verify_focus_identity_pair_readback(readback["paired_rows.csv"], readback["focus_identity_pairs.csv"])
    identity_summary = _identity_summary_from_readback(readback["paired_seed_summary.csv"])
    pilot_plan_path = output / "readability_pilot_plan.md"
    pilot_plan = _readability_pilot_plan()
    if pilot_plan_path.exists():
        _require(pilot_plan_path.read_text(encoding="utf-8") == pilot_plan, "existing readability pilot plan differs")
    else:
        pilot_plan_path.write_text(pilot_plan, encoding="utf-8", newline="\n")
    report_value = {
        "task_id": TASK_ID,
        "status": "EVIDENCE_REVIEW_COMPLETED",
        "classification": "diagnostic-only",
        "scientific_updates": 0,
        "cloud_adam_updates": 0,
        "parent_classification_preserved": report["classification"],
        "parent_q_fg_decision_replayed_from_raw": True,
        "q_fg_seed_decisions_replayed": validation["decision_recomputed_from_saved_chunks"]["seed_decisions"],
        "paired_identity_summary_from_saved_csv": identity_summary,
        "saved_table_consumer_validation": summary_readback,
        "focus_identity_pair_readback": focus_pair_readback,
        "next_step_recommendation": _next_step_from_descriptive_summary(identity_summary, readback["dev_shape_separation.csv"]),
        "populations": {"raw_empirical_dev_rows_rechecked": 7_000, "raw_conditional_rows_rechecked": 10_500, "paired_empirical_dev_rows_per_seed": source.DEV_ROWS, "paired_conditional_sample_rows_per_seed": 1_500, "training_seeds": list(TRAIN_SEEDS), "sample_seeds": list(SAMPLE_SEEDS), "focus_classes": list(FOCUS_CLASSES)},
        "shape_reference": "28x28 area-pooled unit-mass DEV target distributions; descriptive only and distinct from the original TRAIN-target conditional-sample distances",
        "human_readability_claim": False,
        "model_adoption": False,
        "outputs": ["paired_rows.csv", "focus_identity_pairs.csv", "class_summary.csv", "paired_seed_summary.csv", "confusion_transitions.csv", "dev_shape_separation.csv", "readability_pilot_plan.md"],
        "stop": True,
    }
    _write_json(output / "report.json", report_value)
    saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    report_readback = _verify_saved_report_consumers(
        saved_report,
        identity_summary,
        validation["decision_recomputed_from_saved_chunks"]["seed_decisions"],
    )
    validation_value = {
        "status": "VALID_DESCRIPTIVE_EVIDENCE",
        "raw_dev_rows_rechecked": 7_000,
        "raw_sample_rows_rechecked": 10_500,
        "numpy_q_fg_rows_rechecked": 7_000,
        "all_saved_logits_rechecked": True,
        "all_csv_outputs_read_back": True,
        "saved_table_consumer_validation": summary_readback,
        "focus_identity_pair_readback": focus_pair_readback,
        "report_readback_validation": report_readback,
        "parent_q_fg_decision_replayed_from_raw": True,
        "parent_q_fg_seed_decisions_replayed": validation["decision_recomputed_from_saved_chunks"]["seed_decisions"],
        "parent_q_fg_all_three_seeds_pass": all(
            bool(item["passes_both_conditions"])
            for item in validation["decision_recomputed_from_saved_chunks"]["seed_decisions"].values()
        ),
        "scientific_gate_added": False,
    }
    _write_json(output / "validation.json", validation_value)
    return saved_report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.preflight_only:
        evidence = args.evidence_dir.resolve()
        output = args.output_dir.resolve()
        root = Path(__file__).resolve().parents[3]
        _require(output == (root / DEFAULT_OUTPUT).resolve(), "output directory differs from the task's fixed attempt path")
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "task_plan.md"
        plan = _task_plan(evidence)
        if plan_path.exists():
            _require(plan_path.read_text(encoding="utf-8") == plan, "existing output belongs to a different evidence input")
        else:
            plan_path.write_text(plan, encoding="utf-8", newline="\n")
        result = {**_preflight(evidence), "evidence_dir": evidence.as_posix()}
        _write_json(output / "preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = _execute(args.evidence_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
