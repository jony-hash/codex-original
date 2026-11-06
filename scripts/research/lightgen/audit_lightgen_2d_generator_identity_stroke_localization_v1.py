"""Localize saved A/B identity changes by confusion, shape, and display scale."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from ldm_project.v2.application.research.lightgen.canonical_semantics import cmos_view


TASK_ID = "M0_GENERATOR_IDENTITY_STROKE_LOCALIZATION_V1"
OUTPUT_RUN_ROOT = Path("outputs/20261014_m0_generator_identity_stroke_localization_v1/runs")
DEFAULT_OUTPUT = OUTPUT_RUN_ROOT / "attempt_01"
FOCUS_CLASSES = (1, 2, 4, 7, 8, 9)
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
TRANSITIONS = (
    "both_correct",
    "A_correct_B_wrong",
    "A_wrong_B_correct",
    "both_wrong",
)
RAW_CHUNK_ROWS = 50
DEV_ROWS = 1_000
DEV_ROWS_PER_CLASS = 100
SAMPLE_ROWS_PER_CLASS = 30
SAMPLE_ROWS = 300
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8


class EvidenceError(RuntimeError):
    """An input artifact or its production consumer failed validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            _require(set(required).issubset(fields), f"required columns are missing from {path.name}")
            rows = list(reader)
    except OSError as error:
        raise EvidenceError(f"cannot read CSV {path}: {error}") from error
    return rows


def _number(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"invalid numeric field {field}") from error
    _require(math.isfinite(value), f"nonfinite field {field}")
    return value


def _scenario_arm(row: Mapping[str, str]) -> tuple[int, str] | None:
    scenario = str(row["scenario"])
    pieces = scenario.split("_")
    if len(pieces) != 4 or pieces[0] != "seed" or pieces[2] not in {"A", "B"} or pieces[3] != "step6700":
        return None
    try:
        return int(pieces[1]), pieces[2]
    except ValueError as error:
        raise EvidenceError(f"malformed A/B scenario: {scenario}") from error


def _correct_transition(a_prediction: int, b_prediction: int, target: int) -> str:
    a_correct = a_prediction == target
    b_correct = b_prediction == target
    if a_correct and b_correct:
        return "both_correct"
    if a_correct:
        return "A_correct_B_wrong"
    if b_correct:
        return "A_wrong_B_correct"
    return "both_wrong"


def _record_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    population = row["population"]
    seed = int(row["training_seed"])
    if population == "empirical_dev":
        return (population, seed, int(row["source_row_index"]))
    if population == "conditional_sample":
        return (
            population,
            seed,
            int(row["sample_seed"]),
            int(row["requested_class"]),
            int(row["draw_index"]),
        )
    raise EvidenceError(f"unexpected population: {population}")


def _target_class(row: Mapping[str, str]) -> int:
    return int(row["label"] if row["population"] == "empirical_dev" else row["requested_class"])


def _pair_rows(
    rows: Sequence[Mapping[str, str]],
    dev_positions: Mapping[tuple[str, int, int], Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Join only the six declared step-6700 arms on their source identity."""
    arms: dict[tuple[Any, ...], dict[str, Mapping[str, str]]] = {}
    seeds: set[int] = set()
    active_counts: Counter[str] = Counter()
    seen_scenarios: set[str] = set()
    for row in rows:
        arm = _scenario_arm(row)
        if arm is None:
            if row["scenario"] == "A4700":
                continue
            raise EvidenceError(f"unexpected analyzed endpoint: {row['scenario']}")
        seed, endpoint = arm
        seeds.add(seed)
        seen_scenarios.add(str(row["scenario"]))
        active_counts[str(row["population"])] += 1
        if int(row["training_seed"]) != seed:
            raise EvidenceError("scenario and row training seed disagree")
        key = _record_key(row)
        pair = arms.setdefault(key, {})
        if endpoint in pair:
            raise EvidenceError(f"duplicate A/B row key: {key}/{endpoint}")
        pair[endpoint] = row

    _require(len(seeds) == 3, "the paired evidence must contain all three training seeds")
    _require(len(seen_scenarios) == 6, "the paired evidence must contain all six A/B endpoints")
    _require(active_counts == Counter({"empirical_dev": 6_000, "conditional_sample": 9_000}), "active A/B row populations differ")
    _require(len(arms) == 7_500, "the active evidence must contain exactly 7,500 A/B pairs")

    pairs: list[dict[str, Any]] = []
    for key, pair in arms.items():
        _require(set(pair) == {"A", "B"}, f"A/B pair is incomplete: {key}")
        a, b = pair["A"], pair["B"]
        target = _target_class(a)
        _require(_target_class(b) == target, f"A/B target class differs: {key}")
        _require(a["population"] == b["population"], f"A/B data roles differ: {key}")
        if a["population"] == "empirical_dev":
            a_position_key = (str(a["scenario"]), int(a["training_seed"]), int(a["source_row_index"]))
            b_position_key = (str(b["scenario"]), int(b["training_seed"]), int(b["source_row_index"]))
            a_position_row = dev_positions.get(a_position_key)
            b_position_row = dev_positions.get(b_position_key)
            _require(a_position_row is not None and b_position_row is not None, f"DEV source key has no raw position: {key}")
            _require(a_position_row["label"] == target and b_position_row["label"] == target, f"DEV raw position label differs: {key}")
            a_position = a_position_row["position"]
            b_position = b_position_row["position"]
            _require(a_position == b_position, f"paired DEV positions differ: {key}")
        else:
            a_position = target * SAMPLE_ROWS_PER_CLASS + int(a["draw_index"])
            b_position = target * SAMPLE_ROWS_PER_CLASS + int(b["draw_index"])
        pair_row: dict[str, Any] = {
            "population": a["population"],
            "training_seed": int(a["training_seed"]),
            "target_class": target,
            "source_row_index": a["source_row_index"],
            "sample_seed": a["sample_seed"],
            "draw_index": a["draw_index"],
            "sample_key": ":".join(str(item) for item in key),
            "a_scenario": a["scenario"],
            "b_scenario": b["scenario"],
            "a_position": a_position,
            "b_position": b_position,
            "a_e_d_identity_margin": _number(a, "e_d_identity_margin"),
            "b_e_d_identity_margin": _number(b, "e_d_identity_margin"),
            "delta_e_d_identity_margin_b_minus_a": _number(b, "e_d_identity_margin") - _number(a, "e_d_identity_margin"),
            "a_raw_cmos_integral": _number(a, "raw_cmos_integral"),
            "b_raw_cmos_integral": _number(b, "raw_cmos_integral"),
            "delta_raw_cmos_integral_b_minus_a": _number(b, "raw_cmos_integral") - _number(a, "raw_cmos_integral"),
            "a_clipped_fraction": _number(a, "clipped_fraction"),
            "b_clipped_fraction": _number(b, "clipped_fraction"),
        }
        pair_row["raw_cmos_integral_ratio_b_over_a"] = (
            pair_row["b_raw_cmos_integral"] / pair_row["a_raw_cmos_integral"]
            if pair_row["a_raw_cmos_integral"] > 0
            else "A_ZERO_MASS"
        )
        if a["population"] == "empirical_dev":
            pair_row["a_q_fg"] = _number(a, "q_fg")
            pair_row["b_q_fg"] = _number(b, "q_fg")
            pair_row["delta_q_fg_b_minus_a"] = _number(b, "q_fg") - _number(a, "q_fg")
        else:
            pair_row["a_q_fg"] = ""
            pair_row["b_q_fg"] = ""
            pair_row["delta_q_fg_b_minus_a"] = ""
        for evaluator in EVALUATORS:
            suffix = evaluator.lower()
            a_prediction = int(a[f"{suffix}_prediction"])
            b_prediction = int(b[f"{suffix}_prediction"])
            pair_row[f"a_{suffix}_prediction"] = a_prediction
            pair_row[f"b_{suffix}_prediction"] = b_prediction
            pair_row[f"{suffix}_transition"] = _correct_transition(a_prediction, b_prediction, target)
        pairs.append(pair_row)

    return sorted(
        pairs,
        key=lambda row: (
            row["population"],
            row["training_seed"],
            row["target_class"],
            row["sample_seed"],
            row["draw_index"],
            row["source_row_index"],
        ),
    )


def _confusion_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter()
    for row in pairs:
        for evaluator in EVALUATORS:
            suffix = evaluator.lower()
            key = (
                row["population"],
                row["training_seed"],
                evaluator,
                row["target_class"],
                row[f"a_{suffix}_prediction"],
                row[f"b_{suffix}_prediction"],
                row[f"{suffix}_transition"],
            )
            counts[key] += 1
    fields = ("population", "training_seed", "evaluator", "target_or_requested_class", "a_prediction", "b_prediction", "correctness_transition", "pairs")
    return [dict(zip(fields, (*key, count), strict=True)) for key, count in sorted(counts.items())]


def _population_identity_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for population in ("empirical_dev", "conditional_sample"):
        rows = [row for row in pairs if row["population"] == population]
        _require(bool(rows), f"no paired rows for population {population}")
        transitions = Counter(row["e_d_transition"] for row in rows)
        a_correct = transitions["both_correct"] + transitions["A_correct_B_wrong"]
        b_correct = transitions["both_correct"] + transitions["A_wrong_B_correct"]
        by_seed: list[dict[str, Any]] = []
        for seed in sorted({int(row["training_seed"]) for row in rows}):
            seed_rows = [row for row in rows if int(row["training_seed"]) == seed]
            seed_counts = Counter(row["e_d_transition"] for row in seed_rows)
            seed_a = seed_counts["both_correct"] + seed_counts["A_correct_B_wrong"]
            seed_b = seed_counts["both_correct"] + seed_counts["A_wrong_B_correct"]
            by_seed.append(
                {
                    "training_seed": seed,
                    "pairs": len(seed_rows),
                    "a_correct": seed_a,
                    "b_correct": seed_b,
                    "delta_correct_b_minus_a": seed_b - seed_a,
                    "a_accuracy": seed_a / len(seed_rows),
                    "b_accuracy": seed_b / len(seed_rows),
                    "transitions": {name: seed_counts[name] for name in TRANSITIONS},
                }
            )

        class_profile: list[dict[str, Any]] = []
        for digit in range(10):
            digit_rows = [row for row in rows if int(row["target_class"]) == digit]
            digit_counts = Counter(row["e_d_transition"] for row in digit_rows)
            digit_a = digit_counts["both_correct"] + digit_counts["A_correct_B_wrong"]
            digit_b = digit_counts["both_correct"] + digit_counts["A_wrong_B_correct"]
            class_profile.append(
                {
                    "target_or_requested_class": digit,
                    "pairs": len(digit_rows),
                    "a_correct": digit_a,
                    "b_correct": digit_b,
                    "delta_correct_b_minus_a": digit_b - digit_a,
                    "a_accuracy": digit_a / len(digit_rows),
                    "b_accuracy": digit_b / len(digit_rows),
                    "transitions": {name: digit_counts[name] for name in TRANSITIONS},
                }
            )

        regressions: Counter[tuple[int, int]] = Counter()
        recoveries: Counter[tuple[int, int]] = Counter()
        persistent: Counter[tuple[int, int, int]] = Counter()
        for row in rows:
            target = int(row["target_class"])
            a_prediction, b_prediction = int(row["a_e_d_prediction"]), int(row["b_e_d_prediction"])
            if row["e_d_transition"] == "A_correct_B_wrong":
                regressions[(target, b_prediction)] += 1
            elif row["e_d_transition"] == "A_wrong_B_correct":
                recoveries[(target, a_prediction)] += 1
            elif row["e_d_transition"] == "both_wrong":
                persistent[(target, a_prediction, b_prediction)] += 1

        ratios = [
            float(row["raw_cmos_integral_ratio_b_over_a"])
            for row in rows
            if row["raw_cmos_integral_ratio_b_over_a"] != "A_ZERO_MASS"
        ]
        margins = [float(row["delta_e_d_identity_margin_b_minus_a"]) for row in rows]
        population_summary: dict[str, Any] = {
            "pairs": len(rows),
            "e_d_transitions": {name: transitions[name] for name in TRANSITIONS},
            "a_correct": a_correct,
            "b_correct": b_correct,
            "delta_correct_b_minus_a": b_correct - a_correct,
            "a_accuracy": a_correct / len(rows),
            "b_accuracy": b_correct / len(rows),
            "mean_delta_e_d_identity_margin_b_minus_a": statistics.fmean(margins),
            "median_b_over_a_raw_cmos_integral_ratio": statistics.median(ratios),
            "seed_profiles": by_seed,
            "class_profiles": class_profile,
            "largest_b_regressions_target_to_competitor": [
                {"target_class": key[0], "b_competitor": key[1], "pairs": count}
                for key, count in regressions.most_common(5)
            ],
            "largest_b_recoveries_from_competitor": [
                {"target_class": key[0], "a_competitor": key[1], "pairs": count}
                for key, count in recoveries.most_common(5)
            ],
            "largest_persistent_wrong_confusions": [
                {"target_class": key[0], "a_prediction": key[1], "b_prediction": key[2], "pairs": count}
                for key, count in persistent.most_common(5)
            ],
            "class_losses": sorted(class_profile, key=lambda item: (item["delta_correct_b_minus_a"], item["target_or_requested_class"]))[:3],
            "class_gains": sorted(class_profile, key=lambda item: (-item["delta_correct_b_minus_a"], item["target_or_requested_class"]))[:3],
        }
        if population == "empirical_dev":
            q_a = [float(row["a_q_fg"]) for row in rows]
            q_b = [float(row["b_q_fg"]) for row in rows]
            population_summary["dev_q_fg"] = {
                "a_mean": statistics.fmean(q_a),
                "b_mean": statistics.fmean(q_b),
                "delta_b_minus_a": statistics.fmean(q_b) - statistics.fmean(q_a),
            }
        result[population] = population_summary
    return result


def _class_summary_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        for evaluator in EVALUATORS:
            grouped[(row["population"], row["training_seed"], row["target_class"], evaluator)].append(row)
    result: list[dict[str, Any]] = []
    for (population, seed, digit, evaluator), rows in sorted(grouped.items()):
        suffix = evaluator.lower()
        counts = Counter(row[f"{suffix}_transition"] for row in rows)
        margins = [row["delta_e_d_identity_margin_b_minus_a"] for row in rows]
        intensities = [row["delta_raw_cmos_integral_b_minus_a"] for row in rows]
        q_deltas = [float(row["delta_q_fg_b_minus_a"]) for row in rows if row["delta_q_fg_b_minus_a"] != ""]
        result.append(
            {
                "population": population,
                "training_seed": seed,
                "target_or_requested_class": digit,
                "evaluator": evaluator,
                "pairs": len(rows),
                **{f"{name}_pairs": counts[name] for name in TRANSITIONS},
                "mean_delta_e_d_identity_margin_b_minus_a": statistics.fmean(margins),
                "mean_delta_raw_cmos_integral_b_minus_a": statistics.fmean(intensities),
                "median_delta_raw_cmos_integral_b_minus_a": statistics.median(intensities),
                "mean_delta_q_fg_b_minus_a": statistics.fmean(q_deltas) if q_deltas else "",
            }
        )
    return result


def _select_pairs(pairs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        if int(row["target_class"]) not in FOCUS_CLASSES:
            continue
        strata[(row["population"], row["training_seed"], row["target_class"], row["e_d_transition"])].append(row)

    selected: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for population in ("empirical_dev", "conditional_sample"):
        for seed in sorted({int(row["training_seed"]) for row in pairs}):
            for digit in FOCUS_CLASSES:
                for transition in TRANSITIONS:
                    key = (population, seed, digit, transition)
                    candidates = strata.get(key, [])
                    if population == "empirical_dev":
                        candidates = sorted(candidates, key=lambda row: int(row["source_row_index"]))
                    else:
                        candidates = sorted(candidates, key=lambda row: (int(row["sample_seed"]), int(row["draw_index"])))
                    chosen = candidates[:2]
                    inventory.append(
                        {
                            "population": population,
                            "training_seed": seed,
                            "target_or_requested_class": digit,
                            "e_d_transition": transition,
                            "candidate_pairs": len(candidates),
                            "selected_pairs": len(chosen),
                            "selection_status": "selected" if chosen else "empty_stratum",
                        }
                    )
                    for rank, row in enumerate(chosen, start=1):
                        selected.append(
                            {
                                **row,
                                "selection_rank": rank,
                                "stratum_pairs": len(candidates),
                            }
                        )
    _require(0 < len(selected) <= 288, "fixed selection count is outside the declared bound")
    return selected, inventory


def _source_report(evidence_dir: Path, analysis_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = _read_json(evidence_dir / "report.json")
    parent_validation = _read_json(evidence_dir / "validation.json")
    parent_population = _read_json(evidence_dir / "input_population.json")
    analysis = _read_json(analysis_dir / "report.json")
    _require(parent.get("status") == "VALID_COMPLETED", "parent training result is not the named valid completion")
    _require(parent.get("classification") == "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS", "parent training classification differs")
    _require(parent_validation.get("status") == "VALID" and parent_validation.get("scientific_updates") == 12_000, "parent validation or update count differs")
    decision = parent_validation.get("decision_recomputed_from_saved_chunks")
    _require(isinstance(decision, dict) and decision.get("all_three_seeds_pass") is True, "parent three-seed decision is incomplete")
    _require(parent_population.get("task_id") == "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1", "parent population task identity differs")
    _require(analysis.get("task_id") == "M0_GENERATOR_CLOUD_DIGIT_IDENTITY_SAVED_EVIDENCE_V1", "saved identity analysis task identity differs")
    _require(analysis.get("status") == "EVIDENCE_REVIEW_COMPLETED", "saved identity analysis did not complete")
    _require(analysis.get("scientific_updates") == 0 and analysis.get("human_readability_claim") is False, "saved identity analysis exceeds its authorized scope")
    return parent, parent_validation, analysis


def _load_population_rows(
    analysis_dir: Path,
    evidence_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[tuple[str, int, int], dict[str, int]]]:
    paired = _read_csv(
        analysis_dir / "paired_rows.csv",
        (
            "population", "training_seed", "scenario", "sample_seed", "requested_class", "draw_index",
            "source_row_index", "label", "q_fg", "raw_cmos_integral", "clipped_fraction",
            "e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction", "e_d_identity_margin",
        ),
    )
    transitions = _read_csv(
        analysis_dir / "confusion_transitions.csv",
        ("population", "training_seed", "true_or_requested_class", "evaluator", "A_prediction", "B_prediction", "paired_outcome", "count"),
    )
    _require(len(paired) == 17_500, "saved identity table does not contain the declared 17,500 rows")
    active = [row for row in paired if _scenario_arm(row) is not None]
    _require(len(active) == 15_000, "saved identity table does not contain the 15,000 active A/B endpoint rows")
    _require(Counter(row["population"] for row in paired if row["scenario"] != "A4700") == Counter({"empirical_dev": 6_000, "conditional_sample": 9_000}), "A/B population counts differ")

    source_rows = _read_csv(
        evidence_dir / "paired_rows.csv",
        ("scenario", "position", "source_row_index", "label", "role"),
    )
    positions: dict[tuple[str, int, int], dict[str, int]] = {}
    position_sets: dict[str, set[int]] = defaultdict(set)
    for row in source_rows:
        arm = _scenario_arm({"scenario": row["scenario"]})
        if arm is None or row["role"] != "dev":
            continue
        seed, _endpoint = arm
        key = (row["scenario"], seed, int(row["source_row_index"]))
        _require(key not in positions, f"duplicate parent DEV source key: {key}")
        position = int(row["position"])
        _require(position not in position_sets[row["scenario"]], f"duplicate parent DEV position: {row['scenario']}/{position}")
        positions[key] = {"position": position, "label": int(row["label"])}
        position_sets[row["scenario"]].add(position)
    _require(len(position_sets) == 6, "parent DEV row table does not contain all six step-6700 A/B endpoints")
    _require(all(positions_count == set(range(DEV_ROWS)) for positions_count in position_sets.values()), "parent DEV raw positions are incomplete or unordered")
    _require(len(positions) == 6_000, "parent DEV row table does not contain exactly 6,000 endpoint rows")
    return paired, transitions, positions


def _verify_source_confusions(pairs: Sequence[Mapping[str, Any]], source_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    generated = _confusion_rows(pairs)
    generated_map: dict[tuple[Any, ...], int] = {}
    for row in generated:
        key = (
            row["population"], str(row["training_seed"]), row["evaluator"],
            str(row["target_or_requested_class"]), str(row["a_prediction"]),
            str(row["b_prediction"]), row["correctness_transition"],
        )
        generated_map[key] = int(row["pairs"])
    source_map: dict[tuple[Any, ...], int] = {}
    for row in source_rows:
        key = (
            row["population"], row["training_seed"], row["evaluator"],
            row["true_or_requested_class"], row["A_prediction"], row["B_prediction"], row["paired_outcome"],
        )
        if key in source_map:
            raise EvidenceError(f"duplicate source confusion row: {key}")
        source_map[key] = int(row["count"])
    _require(generated_map == source_map, "recomputed confusion table differs from saved full-population analysis")
    return {"source_confusion_rows": len(source_map), "recomputed_confusion_rows": len(generated_map), "exact_match": True}


def _load_chunk(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"selected raw chunk is missing: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise EvidenceError(f"selected raw chunk cannot be loaded: {path}: {error}") from error
    _require(isinstance(value, Mapping), f"selected raw chunk is not a mapping: {path}")
    return value


def _expected_logits(row: Mapping[str, str], logits: Mapping[str, torch.Tensor], offset: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for evaluator in EVALUATORS:
        _require(evaluator in logits, f"selected raw chunk is missing {evaluator} logits")
        values = logits[evaluator]
        _require(tuple(values.shape) == (RAW_CHUNK_ROWS, 10), f"selected {evaluator} logits have the wrong shape")
        one = values[offset].to(torch.float64)
        _require(bool(torch.isfinite(one).all()), f"selected {evaluator} logits are nonfinite")
        prediction = int(one.argmax().item())
        _require(prediction == int(row[f"{evaluator.lower()}_prediction"]), f"selected {evaluator} prediction differs from its CSV row")
        if evaluator == "E_D":
            label = _target_class(row)
            competitor = torch.cat((one[:label], one[label + 1 :]))
            result["recomputed_e_d_identity_margin"] = float(one[label] - competitor.max())
    return result


def _selected_raw_path(evidence_dir: Path, row: Mapping[str, Any], endpoint: str = "A") -> tuple[Path, int]:
    scenario = str(row[f"{endpoint.lower()}_scenario"])
    position = int(row[f"{endpoint.lower()}_position"])
    chunk_index, offset = divmod(position, RAW_CHUNK_ROWS)
    if row["population"] == "empirical_dev":
        return evidence_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt", offset
    sample_seed = int(row["sample_seed"])
    return (
        evidence_dir / "readouts" / "conditional_samples" / scenario / f"seed_{sample_seed}" / f"chunk_{chunk_index:04d}.pt",
        offset,
    )


def _read_selected_endpoint(
    evidence_dir: Path,
    row: Mapping[str, Any],
    endpoint: str,
    csv_row: Mapping[str, str],
) -> dict[str, Any]:
    scenario = str(row[f"{endpoint.lower()}_scenario"])
    position = int(row[f"{endpoint.lower()}_position"])
    chunk_index, offset = divmod(position, RAW_CHUNK_ROWS)
    if row["population"] == "empirical_dev":
        path = evidence_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt"
    else:
        sample_seed = int(row["sample_seed"])
        path = evidence_dir / "readouts" / "conditional_samples" / scenario / f"seed_{sample_seed}" / f"chunk_{chunk_index:04d}.pt"
    chunk = _load_chunk(path)
    _require(chunk.get("task_id") == "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1", f"selected task identity differs: {path}")
    _require(chunk.get("scenario") == scenario, f"selected raw scenario differs: {path}")
    raw_values = chunk.get("raw_cmos")
    saved_view = chunk.get("evaluator_view")
    _require(isinstance(raw_values, torch.Tensor) and raw_values.ndim == 4 and tuple(raw_values.shape[1:]) == (1, 200, 200), f"selected raw CMOS shape differs: {path}")
    _require(len(raw_values) == RAW_CHUNK_ROWS and offset < len(raw_values), f"selected row is outside its raw chunk: {path}")
    raw = raw_values[offset : offset + 1].to(torch.float64).contiguous()
    _require(bool(torch.isfinite(raw).all() and (raw >= 0).all()), f"selected raw CMOS is invalid: {path}")
    logits = chunk.get("logits")
    _require(isinstance(logits, Mapping), f"selected logits missing from raw chunk: {path}")
    margin = _expected_logits(csv_row, logits, offset)
    saved_margin = _number(csv_row, "e_d_identity_margin")
    _require(math.isclose(margin["recomputed_e_d_identity_margin"], saved_margin, rel_tol=2e-6, abs_tol=5e-8), f"selected E_D identity margin differs from its CSV row: {path}")
    production = cmos_view(raw.to(raw_values.dtype)).to(torch.float64)
    _require(isinstance(saved_view, torch.Tensor) and tuple(saved_view.shape[-2:]) == (28, 28), f"selected saved production view is missing: {path}")
    _require(
        torch.allclose(production, saved_view[offset : offset + 1].to(torch.float64), rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
        f"selected raw-to-production view does not replay: {path}",
    )
    target: torch.Tensor | None = None
    source_index = ""
    if row["population"] == "empirical_dev":
        positions = chunk.get("positions")
        source_rows = chunk.get("source_row_indices")
        labels = chunk.get("labels")
        roles = chunk.get("roles")
        target_values = chunk.get("target_slm_intensity")
        _require(isinstance(positions, torch.Tensor) and int(positions[offset]) == position, f"selected DEV position differs: {path}")
        _require(isinstance(source_rows, torch.Tensor) and int(source_rows[offset]) == int(row["source_row_index"]), f"selected DEV source row differs: {path}")
        _require(isinstance(labels, torch.Tensor) and int(labels[offset]) == int(row["target_class"]), f"selected DEV label differs: {path}")
        _require(isinstance(roles, list) and len(roles) == RAW_CHUNK_ROWS and roles[offset] == "dev", f"selected DEV role differs: {path}")
        _require(isinstance(target_values, torch.Tensor) and tuple(target_values.shape) == (RAW_CHUNK_ROWS, 1, 200, 200), f"selected DEV target is missing: {path}")
        target = target_values[offset : offset + 1].to(torch.float64).contiguous()
        _require(bool(torch.isfinite(target).all() and (target >= 0).all()), f"selected DEV target is invalid: {path}")
        source_index = str(int(source_rows[offset]))
    else:
        metadata = chunk.get("rows")
        _require(isinstance(metadata, list) and len(metadata) == RAW_CHUNK_ROWS, f"selected conditional metadata is incomplete: {path}")
        expected_position = int(row["target_class"]) * SAMPLE_ROWS_PER_CLASS + int(row["draw_index"])
        item = metadata[offset]
        _require(item.get("position") == expected_position, f"selected conditional position differs: {path}")
        _require(item.get("sample_seed") == int(row["sample_seed"]), f"selected conditional seed differs: {path}")
        _require(item.get("requested_class") == int(row["target_class"]), f"selected conditional request differs: {path}")
        _require(item.get("draw_index") == int(row["draw_index"]), f"selected conditional draw differs: {path}")
    return {
        "raw": raw,
        "view": production,
        "target": target,
        "source_row_index": source_index,
        "recomputed_e_d_identity_margin": margin["recomputed_e_d_identity_margin"],
        "raw_path": path.as_posix(),
    }


def _normalized(raw: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    values = raw.to(torch.float64)
    mass = float(values.sum())
    if mass == 0.0:
        return None, mass
    _require(math.isfinite(mass) and mass > 0.0, "raw CMOS has negative or invalid total mass")
    return values / mass, mass


def _qfg_numpy(raw: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    image = raw.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    truth = target.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    _require(image.shape == truth.shape == (40_000,), "DEV raw and target must each contain 40,000 pixels")
    image_mass, target_mass = float(image.sum()), float(truth.sum())
    _require(image_mass > 0.0 and target_mass > 0.0, "Q_fg is undefined for zero-mass images")
    p_image, p_target = image / image_mass, truth / target_mass
    mask = truth > 0.1
    denominator = float(np.square(p_target).sum())
    _require(bool(mask.any() and (~mask).any() and denominator > 0.0), "DEV foreground/background mask or denominator is invalid")
    delta = p_image - p_target
    qfg = float(np.square(delta[mask]).sum() / denominator)
    underfill = float(np.maximum(-delta[mask], 0.0).sum())
    overfill = float(np.maximum(delta[mask], 0.0).sum())
    background = float(p_image[~mask].sum())
    return {"q_fg": qfg, "foreground_underfill_mass": underfill, "foreground_overfill_mass": overfill, "background_generated_mass": background}


def _png_data(value: torch.Tensor, *, scale: float, difference: bool = False, size: int = 140) -> str:
    image = value.detach().cpu().to(torch.float64).squeeze()
    _require(image.ndim == 2 and image.numel() > 0, "atlas image is not a 2D tensor")
    if difference:
        normalized = image.clamp(-scale, scale) / scale if scale > 0.0 else torch.zeros_like(image)
        pixels = ((normalized + 1.0) * 127.5).round().to(torch.uint8)
    else:
        _require(math.isfinite(scale) and scale > 0.0, "atlas image scale must be positive")
        pixels = (image.clamp(0.0, scale) / scale * 255.0).round().to(torch.uint8)
    width, height = int(pixels.shape[1]), int(pixels.shape[0])
    pil = Image.frombytes("L", (width, height), bytes(pixels.reshape(-1).tolist()))
    pil = pil.resize((size, size), resample=Image.Resampling.NEAREST)
    stream = io.BytesIO()
    pil.save(stream, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _pair_metrics_and_panels(
    selected: Mapping[str, Any],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    raw_a, raw_b = a["raw"], b["raw"]
    shape_a, mass_a = _normalized(raw_a)
    shape_b, mass_b = _normalized(raw_b)
    both_positive = shape_a is not None and shape_b is not None
    view_a, view_b = a["view"], b["view"]
    matched_b: torch.Tensor | None = None
    matched_view_b: torch.Tensor | None = None
    if mass_b > 0.0:
        factor = mass_a / mass_b if mass_a > 0.0 else 0.0
        matched_b = raw_b * factor
        matched_view_b = cmos_view(matched_b.to(raw_b.dtype)).to(torch.float64)
    else:
        factor = math.nan
    display_gap = float((view_a - view_b).abs().mean())
    matched_gap = float((view_a - matched_view_b).abs().mean()) if matched_view_b is not None else math.nan
    shape_tv = 0.5 * float((shape_a - shape_b).abs().sum()) if both_positive else math.nan
    shape_l2 = float(torch.linalg.vector_norm(shape_a - shape_b)) if both_positive else math.nan
    qfg_a: dict[str, float] | None = None
    qfg_b: dict[str, float] | None = None
    qfg_error_a: float | str = ""
    qfg_error_b: float | str = ""
    if selected["population"] == "empirical_dev":
        _require(a["target"] is not None and b["target"] is not None, "DEV target is missing")
        _require(torch.equal(a["target"], b["target"]), "paired DEV targets differ between A and B")
        qfg_a = _qfg_numpy(raw_a, a["target"])
        qfg_b = _qfg_numpy(raw_b, b["target"])
        qfg_error_a = qfg_a["q_fg"] - float(selected["a_q_fg"])
        qfg_error_b = qfg_b["q_fg"] - float(selected["b_q_fg"])
        _require(abs(qfg_error_a) <= 1e-12 and abs(qfg_error_b) <= 1e-12, "selected raw Q_fg does not reproduce the saved analysis table")

    metrics: dict[str, Any] = {
        "sample_key": selected["sample_key"],
        "population": selected["population"],
        "training_seed": selected["training_seed"],
        "target_or_requested_class": selected["target_class"],
        "e_d_transition": selected["e_d_transition"],
        "selection_rank": selected["selection_rank"],
        "source_row_index": selected["source_row_index"],
        "sample_seed": selected["sample_seed"],
        "draw_index": selected["draw_index"],
        "a_raw_mass": mass_a,
        "b_raw_mass": mass_b,
        "b_to_a_energy_match_factor": factor if math.isfinite(factor) else "B_ZERO_MASS",
        "a_raw_clipped_pixel_fraction": float((raw_a > 1.0).to(torch.float64).mean()),
        "b_raw_clipped_pixel_fraction": float((raw_b > 1.0).to(torch.float64).mean()),
        "a_clipped_intensity_fraction": float((raw_a - raw_a.clamp(max=1.0)).sum() / mass_a) if mass_a > 0.0 else "A_ZERO_MASS",
        "b_clipped_intensity_fraction": float((raw_b - raw_b.clamp(max=1.0)).sum() / mass_b) if mass_b > 0.0 else "B_ZERO_MASS",
        "production_view_a_b_mean_absolute_gap": display_gap,
        "production_view_a_energy_matched_b_mean_absolute_gap": matched_gap if math.isfinite(matched_gap) else "B_ZERO_MASS",
        "unit_mass_shape_total_variation_a_b": shape_tv if math.isfinite(shape_tv) else "ZERO_MASS_SHAPE_UNDEFINED",
        "unit_mass_shape_l2_a_b": shape_l2 if math.isfinite(shape_l2) else "ZERO_MASS_SHAPE_UNDEFINED",
        "a_zero_mass": mass_a == 0.0,
        "b_zero_mass": mass_b == 0.0,
        "normalized_shape_status": "VALID" if both_positive else "ZERO_MASS_SHAPE_UNDEFINED",
        "a_recomputed_e_d_identity_margin": a["recomputed_e_d_identity_margin"],
        "b_recomputed_e_d_identity_margin": b["recomputed_e_d_identity_margin"],
        "a_raw_path": a["raw_path"],
        "b_raw_path": b["raw_path"],
        "raw_q_fg_a_recomputed": qfg_a["q_fg"] if qfg_a else "",
        "raw_q_fg_b_recomputed": qfg_b["q_fg"] if qfg_b else "",
        "q_fg_saved_replay_error_a": qfg_error_a,
        "q_fg_saved_replay_error_b": qfg_error_b,
        "a_foreground_underfill_mass": qfg_a["foreground_underfill_mass"] if qfg_a else "",
        "b_foreground_underfill_mass": qfg_b["foreground_underfill_mass"] if qfg_b else "",
        "a_foreground_overfill_mass": qfg_a["foreground_overfill_mass"] if qfg_a else "",
        "b_foreground_overfill_mass": qfg_b["foreground_overfill_mass"] if qfg_b else "",
        "a_generated_background_mass": qfg_a["background_generated_mass"] if qfg_a else "",
        "b_generated_background_mass": qfg_b["background_generated_mass"] if qfg_b else "",
    }

    shared_shape_scale = max(float(shape_a.max()) if shape_a is not None else 0.0, float(shape_b.max()) if shape_b is not None else 0.0)
    raw_scale = 1.0
    difference_scale = float((raw_a - raw_b).abs().max())
    target = a["target"]
    panels = [
        {"label": "A raw CMOS (display clipped at 1)", "data": _png_data(raw_a, scale=raw_scale)},
        {"label": "B raw CMOS (same display scale)", "data": _png_data(raw_b, scale=raw_scale)},
        {"label": "A production clamp then 28×28 pool", "data": _png_data(view_a, scale=1.0)},
        {"label": "B production clamp then 28×28 pool", "data": _png_data(view_b, scale=1.0)},
        {"label": "A unit-mass spatial shape", "data": _png_data(shape_a, scale=shared_shape_scale) if shape_a is not None else ""},
        {"label": "B unit-mass spatial shape (shared pair scale)", "data": _png_data(shape_b, scale=shared_shape_scale) if shape_b is not None else ""},
        {"label": "B energy-matched to A, then production view", "data": _png_data(matched_view_b, scale=1.0) if matched_view_b is not None else ""},
        {"label": "A minus B raw difference (symmetric scale)", "data": _png_data(raw_a - raw_b, scale=difference_scale, difference=True)},
    ]
    if target is not None:
        panels.insert(0, {"label": "DEV target (display clipped at 1)", "data": _png_data(target, scale=1.0)})
    else:
        panels.insert(0, {"label": "Conditional sample: no per-image target", "data": ""})
    return metrics, panels


def _read_selected_raw(evidence_dir: Path, selected: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    requests: dict[Path, list[tuple[Mapping[str, Any], str, Mapping[str, str], int]]] = defaultdict(list)
    for pair in selected:
        for endpoint in ("A", "B"):
            csv_row = pair["_a_csv_row"] if endpoint == "A" else pair["_b_csv_row"]
            scenario = str(pair[f"{endpoint.lower()}_scenario"])
            position = int(pair[f"{endpoint.lower()}_position"])
            chunk_index, offset = divmod(position, RAW_CHUNK_ROWS)
            if pair["population"] == "empirical_dev":
                path = evidence_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt"
            else:
                path = evidence_dir / "readouts" / "conditional_samples" / scenario / f"seed_{int(pair['sample_seed'])}" / f"chunk_{chunk_index:04d}.pt"
            requests[path].append((pair, endpoint, csv_row, offset))

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(requests):
        chunk = _load_chunk(path)
        _require(chunk.get("task_id") == "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1", f"selected raw task identity differs: {path}")
        raw_values = chunk.get("raw_cmos")
        _require(isinstance(raw_values, torch.Tensor) and tuple(raw_values.shape) == (RAW_CHUNK_ROWS, 1, 200, 200), f"selected raw chunk geometry differs: {path}")
        for pair, endpoint, csv_row, offset in requests[path]:
            scenario = str(pair[f"{endpoint.lower()}_scenario"])
            _require(chunk.get("scenario") == scenario, f"selected raw scenario differs: {path}")
            raw = raw_values[offset : offset + 1].to(torch.float64).contiguous()
            _require(bool(torch.isfinite(raw).all() and (raw >= 0.0).all()), f"selected raw intensity is invalid: {path}")
            saved_view = chunk.get("evaluator_view")
            _require(isinstance(saved_view, torch.Tensor) and tuple(saved_view.shape) == (RAW_CHUNK_ROWS, 1, 28, 28), f"selected saved production view is missing: {path}")
            production = cmos_view(raw.to(raw_values.dtype)).to(torch.float64)
            _require(
                torch.allclose(production, saved_view[offset : offset + 1].to(torch.float64), rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                f"selected raw-to-production view does not replay: {path}",
            )
            logits = chunk.get("logits")
            _require(isinstance(logits, Mapping), f"selected logits are missing: {path}")
            ed_margin = _expected_logits(csv_row, logits, offset)["recomputed_e_d_identity_margin"]
            _require(math.isclose(ed_margin, _number(csv_row, "e_d_identity_margin"), rel_tol=2e-6, abs_tol=5e-8), f"selected E_D margin differs from the CSV: {path}")
            target: torch.Tensor | None = None
            source_row_index = ""
            if pair["population"] == "empirical_dev":
                position_values = chunk.get("positions")
                source_rows = chunk.get("source_row_indices")
                labels = chunk.get("labels")
                roles = chunk.get("roles")
                targets = chunk.get("target_slm_intensity")
                position = int(pair[f"{endpoint.lower()}_position"])
                _require(isinstance(position_values, torch.Tensor) and int(position_values[offset]) == position, f"selected DEV position differs: {path}")
                _require(isinstance(source_rows, torch.Tensor) and int(source_rows[offset]) == int(pair["source_row_index"]), f"selected DEV source row differs: {path}")
                _require(isinstance(labels, torch.Tensor) and int(labels[offset]) == int(pair["target_class"]), f"selected DEV label differs: {path}")
                _require(isinstance(roles, list) and roles[offset] == "dev", f"selected DEV role differs: {path}")
                _require(isinstance(targets, torch.Tensor) and tuple(targets.shape) == (RAW_CHUNK_ROWS, 1, 200, 200), f"selected DEV target is missing: {path}")
                target = targets[offset : offset + 1].to(torch.float64).contiguous()
                _require(bool(torch.isfinite(target).all() and (target >= 0).all()), f"selected DEV target is invalid: {path}")
                source_row_index = str(int(source_rows[offset]))
            else:
                metadata = chunk.get("rows")
                _require(isinstance(metadata, list) and len(metadata) == RAW_CHUNK_ROWS, f"selected conditional row metadata is incomplete: {path}")
                item = metadata[offset]
                expected_position = int(pair["target_class"]) * SAMPLE_ROWS_PER_CLASS + int(pair["draw_index"])
                _require(item.get("position") == expected_position, f"selected conditional position differs: {path}")
                _require(item.get("sample_seed") == int(pair["sample_seed"]), f"selected conditional sample seed differs: {path}")
                _require(item.get("requested_class") == int(pair["target_class"]), f"selected conditional request differs: {path}")
                _require(item.get("draw_index") == int(pair["draw_index"]), f"selected conditional draw differs: {path}")
            result[(str(pair["sample_key"]), endpoint)] = {
                "raw": raw,
                "view": production,
                "target": target,
                "saved_e_d_logits": logits["E_D"][offset].detach().cpu().clone(),
                "source_row_index": source_row_index,
                "recomputed_e_d_identity_margin": ed_margin,
                "raw_path": path.as_posix(),
            }
    _require(len(result) == 2 * len(selected), "selected raw chunks did not provide both A and B rows")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty CSV: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        _require(sum(1 for _ in csv.DictReader(stream)) == len(rows), f"CSV row count failed readback: {path.name}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    _require(saved == payload, f"JSON readback differs: {path.name}")


def _write_atlas(path: Path, entries: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, str]], Mapping[str, Any]]]) -> None:
    cards: list[str] = []
    for selected, panels, metrics in entries:
        title = (
            f"{selected['population']} · seed {selected['training_seed']} · class {selected['target_class']} · "
            f"{selected['e_d_transition']} · rank {selected['selection_rank']} · {selected['sample_key']}"
        )
        metadata = (
            f"A/B margin: {metrics['a_recomputed_e_d_identity_margin']:.6g} / {metrics['b_recomputed_e_d_identity_margin']:.6g}; "
            f"raw mass: {metrics['a_raw_mass']:.6g} / {metrics['b_raw_mass']:.6g}; "
            f"unit-shape TV: {metrics['unit_mass_shape_total_variation_a_b']}; "
            f"production gap: {metrics['production_view_a_b_mean_absolute_gap']:.6g}; "
            f"energy-matched gap: {metrics['production_view_a_energy_matched_b_mean_absolute_gap']}"
        )
        cells = []
        for panel in panels:
            if panel["data"]:
                image = f'<img loading="lazy" alt="{html.escape(panel["label"])}" src="{panel["data"]}">'
            else:
                image = '<div class="missing">undefined for zero mass / no target</div>'
            cells.append(f'<figure>{image}<figcaption>{html.escape(panel["label"])}</figcaption></figure>')
        cards.append(
            f'<article><h2>{html.escape(title)}</h2><p>{html.escape(metadata)}</p><div class="grid">{"".join(cells)}</div></article>'
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saved A/B identity and stroke localization</title>
<style>
body{font:14px system-ui,sans-serif;margin:20px;background:#f4f6f8;color:#18212b}article{background:white;border:1px solid #ccd4dc;border-radius:8px;margin:16px 0;padding:14px;break-inside:avoid}h2{font-size:16px;margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}figure{margin:0;padding:6px;border:1px solid #e0e5e9}img{display:block;width:140px;height:140px;image-rendering:pixelated;background:#eee;margin:auto}.missing{width:140px;height:140px;display:grid;place-items:center;text-align:center;color:#687786;background:#eef1f4;margin:auto}figcaption{margin-top:5px;font-size:12px}p{color:#435261}
</style><body><h1>Saved A/B identity and stroke localization</h1>
<p>Diagnostic atlas only. Images use shared pair display scales. Raw displays are clipped at 1; unit-mass panels use one shared scale within each pair. Counterfactual panels are not evaluator outputs.</p>
""" + "\n".join(cards) + "\n</body></html>\n"
    path.write_text(document, encoding="utf-8", newline="\n")


def _recommendation(metrics: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in metrics if isinstance(row["production_view_a_energy_matched_b_mean_absolute_gap"], (int, float))]
    if not valid:
        return {
            "status": "INSUFFICIENT_SELECTED_RAW_EVIDENCE",
            "recommendation": "none",
            "reason": "no selected pair had two positive raw intensities for the predeclared energy-matched readout comparison",
        }
    display_gap = statistics.fmean(float(row["production_view_a_b_mean_absolute_gap"]) for row in valid)
    matched_gap = statistics.fmean(float(row["production_view_a_energy_matched_b_mean_absolute_gap"]) for row in valid)
    reduced = sum(float(row["production_view_a_energy_matched_b_mean_absolute_gap"]) < float(row["production_view_a_b_mean_absolute_gap"]) for row in valid)
    by_population: dict[str, dict[str, Any]] = {}
    for population in ("empirical_dev", "conditional_sample"):
        members = [row for row in valid if row["population"] == population]
        gaps = [float(row["production_view_a_b_mean_absolute_gap"]) for row in members]
        matched_gaps = [float(row["production_view_a_energy_matched_b_mean_absolute_gap"]) for row in members]
        tv_values = [
            float(row["unit_mass_shape_total_variation_a_b"])
            for row in members
            if isinstance(row["unit_mass_shape_total_variation_a_b"], (int, float))
        ]
        by_population[population] = {
            "selected_pairs": len(members),
            "energy_matched_gap_smaller_pairs": sum(matched < gap for matched, gap in zip(matched_gaps, gaps, strict=True)),
            "mean_production_view_gap": statistics.fmean(gaps) if gaps else "NO_VALID_ROWS",
            "mean_energy_matched_production_view_gap": statistics.fmean(matched_gaps) if matched_gaps else "NO_VALID_ROWS",
            "mean_unit_mass_shape_total_variation": statistics.fmean(tv_values) if tv_values else "ZERO_MASS_SHAPE_UNDEFINED",
            "median_unit_mass_shape_total_variation": statistics.median(tv_values) if tv_values else "ZERO_MASS_SHAPE_UNDEFINED",
        }
    shape_rows = [
        row for row in metrics
        if isinstance(row["unit_mass_shape_total_variation_a_b"], (int, float))
    ]
    by_transition: dict[str, dict[str, Any]] = {}
    for transition in TRANSITIONS:
        members = [row for row in shape_rows if row["e_d_transition"] == transition]
        if not members:
            by_transition[transition] = {"selected_pairs": 0, "shape_status": "NO_SELECTED_ROWS"}
            continue
        gaps = [float(row["production_view_a_b_mean_absolute_gap"]) for row in members]
        matched_gaps = [float(row["production_view_a_energy_matched_b_mean_absolute_gap"]) for row in members]
        tv_values = [float(row["unit_mass_shape_total_variation_a_b"]) for row in members]
        by_transition[transition] = {
            "selected_pairs": len(members),
            "mean_unit_mass_shape_total_variation": statistics.fmean(tv_values),
            "median_unit_mass_shape_total_variation": statistics.median(tv_values),
            "mean_production_view_gap": statistics.fmean(gaps),
            "mean_energy_matched_production_view_gap": statistics.fmean(matched_gaps),
            "energy_match_reduced_view_gap_pairs": sum(matched < gap for matched, gap in zip(matched_gaps, gaps, strict=True)),
        }
    return {
        "status": "DESCRIPTIVE_EVIDENCE_AVAILABLE",
        "interpretation": "MIXED_OR_UNRESOLVED",
        "interpretation_limit": "The audit can compare saved structure and rendered intensity but does not run an evaluator on counterfactual inputs or establish human readability/causality.",
        "recommendation": "frozen_output_energy_match_evaluator_sensitivity",
        "decision_use": "measure whether B-to-A scalar energy matching changes frozen E_D predictions and identity margins on the same selected pairs",
        "single_changed_factor": "B raw CMOS scalar amplitude, set per pair so the pre-clipping integrated intensity matches A",
        "fixed_inputs": "the selected A/B saved raw CMOS rows, targets for DEV only, original evaluator weights, and the production clamp-then-pool view",
        "measurements": "production view replay, frozen E_D predictions and identity margins before and after B energy matching; no retraining, new samples, or human ratings",
        "stop": "stop after the fixed selected population; interpret any change as evaluator sensitivity to this readout counterfactual, not human readability or physical causality",
        "observed_selected_pairs": len(valid),
        "pairs_where_matched_view_gap_is_smaller": reduced,
        "mean_unmatched_production_view_gap": display_gap,
        "mean_energy_matched_production_view_gap": matched_gap,
        "population_counts": by_population,
        "selected_profile_by_e_d_transition": by_transition,
        "missing_measurement": "E_D predictions and margins on energy-matched B views",
        "added_scientific_gate": False,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    recommendation = report["next_diagnostic"]
    dev = report["population_identity_summary"]["empirical_dev"]
    conditional = report["population_identity_summary"]["conditional_sample"]

    def population_lines(name: str, summary: Mapping[str, Any]) -> str:
        seed_delta = ", ".join(
            f"{row['training_seed']}: {row['delta_correct_b_minus_a']:+d}"
            for row in summary["seed_profiles"]
        )
        transitions = summary["e_d_transitions"]
        class_losses = ", ".join(
            f"{row['target_or_requested_class']} ({row['delta_correct_b_minus_a']:+d})"
            for row in summary["class_losses"]
        )
        class_gains = ", ".join(
            f"{row['target_or_requested_class']} ({row['delta_correct_b_minus_a']:+d})"
            for row in summary["class_gains"]
        )
        return (
            f"- **{name}**：A `{summary['a_correct']}/{summary['pairs']}`（{summary['a_accuracy']:.2%}），"
            f"B `{summary['b_correct']}/{summary['pairs']}`（{summary['b_accuracy']:.2%}），净变化 "
            f"`{summary['delta_correct_b_minus_a']:+d}`；四类转移 `both_correct={transitions['both_correct']}`、"
            f"`A_correct_B_wrong={transitions['A_correct_B_wrong']}`、"
            f"`A_wrong_B_correct={transitions['A_wrong_B_correct']}`、`both_wrong={transitions['both_wrong']}`。"
            f"逐种子净变化 `{seed_delta}`；较大类别退化 `{class_losses}`、改善 `{class_gains}`。"
        )

    population_report = "\n".join(
        (
            population_lines("经验 DEV", dev),
            population_lines("条件采样", conditional),
        )
    )
    dev_qfg = dev["dev_q_fg"]
    selected_profile = recommendation["population_counts"]
    dev_selected = selected_profile["empirical_dev"]
    sample_selected = selected_profile["conditional_sample"]
    dev_regressions = ", ".join(
        f"{row['target_class']}→{row['b_competitor']} ({row['pairs']})"
        for row in dev["largest_b_regressions_target_to_competitor"][:3]
    )
    sample_regressions = ", ".join(
        f"{row['target_class']}→{row['b_competitor']} ({row['pairs']})"
        for row in conditional["largest_b_regressions_target_to_competitor"][:3]
    )
    dev_persistent = ", ".join(
        f"{row['target_class']}→{row['a_prediction']}/{row['b_prediction']} ({row['pairs']})"
        for row in dev["largest_persistent_wrong_confusions"][:3]
    )
    sample_persistent = ", ".join(
        f"{row['target_class']}→{row['a_prediction']}/{row['b_prediction']} ({row['pairs']})"
        for row in conditional["largest_persistent_wrong_confusions"][:3]
    )
    return f"""# 前景目标与数字身份变化：保存证据定位

## 结果与状态

状态：`{report['status']}`，本任务为零更新描述性审计。配对表覆盖 `{report['paired_rows']}` 行并重建 `{report['paired_count']}` 个 A/B 键；全人口混淆汇总与既存分析 `{report['source_confusion_validation']['source_confusion_rows']}` 行逐项一致。固定选样 `{report['selected_pair_count']}` 对，最多 288 对。没有重训、模型前向、重新采样或人眼评分。

## 科学认识

全人口 E_D 配对结果（描述性计数，不作显著性或泛化推断）：

{population_report}

DEV 全十类平均 Q_fg 为 A `{dev_qfg['a_mean']:.6f}`、B `{dev_qfg['b_mean']:.6f}`（B−A `{dev_qfg['delta_b_minus_a']:.6f}`）；全人口 B/A raw CMOS 积分强度中位数为 DEV `{dev['median_b_over_a_raw_cmos_integral_ratio']:.6f}`、条件采样 `{conditional['median_b_over_a_raw_cmos_integral_ratio']:.6f}`。较大的 B 新发混淆方向为 DEV `{dev_regressions}`、条件采样 `{sample_regressions}`；两臂持续错误方向示例为 DEV `{dev_persistent}`、条件采样 `{sample_persistent}`。

所选图册的 B→A 能量匹配前/后生产视图平均绝对差：DEV `{dev_selected['mean_production_view_gap']:.6f}`→`{dev_selected['mean_energy_matched_production_view_gap']:.6f}`（{dev_selected['energy_matched_gap_smaller_pairs']}/{dev_selected['selected_pairs']} 对缩小，单位总强度形态 TV 均值 `{dev_selected['mean_unit_mass_shape_total_variation']:.6f}`）；条件采样 `{sample_selected['mean_production_view_gap']:.6f}`→`{sample_selected['mean_energy_matched_production_view_gap']:.6f}`（{sample_selected['energy_matched_gap_smaller_pairs']}/{sample_selected['selected_pairs']} 对缩小，TV 均值 `{sample_selected['mean_unit_mass_shape_total_variation']:.6f}`）。这是强度呈现变化与仍存在的形态差异并存的混合描述。能量匹配图没有重新运行 E_D，因此不说明分类是否随呈现改变。

每对保留 A/B 预测、四种正确性转移、E_D 身份余量、DEV 的 Q_fg 及 raw CMOS 积分强度。固定样本 raw 到生产 clamp→28×28 pooling 视图逐项复核；DEV 的 Q_fg 又用独立 NumPy 公式重算。单位总强度形态、原始生产视图和 B 匹配 A 总强度后的呈现只作描述性对照，不等同于 E_D 对反事实输入的预测，也不证明因果或人眼识读。

全人口混淆见 `confusion_transitions.csv` 和 `class_summary.csv`；固定图册见 `review_atlas.html`；所选行的 raw、目标、强度与形态读数见 `selected_sample_metrics.csv`。图册样本由稳定键分层选择，样本比例不能解释为总体发生率。条件采样没有逐图目标，因此不计算其目标笔画误差。

## 目标缺口

此结果仍不能把混淆变化归因于结构或读出，也不验证人眼可读性、泛化或模型采用。由于DEV与条件采样的身份净变化方向相反、能量匹配后呈现差异略收缩而归一化形态差异仍在，定位结论为 `MIXED_OR_UNRESOLVED`。唯一后续建议为 `{recommendation['recommendation']}`：`{recommendation.get('decision_use', recommendation.get('reason', '证据无效，停止并修复输入消费'))}`。该建议只用固定选中的 `{recommendation.get('observed_selected_pairs', 0)}` 对，且不是本轮授权。

## 工程完成与验证

原始证据消费路径（本次执行读取）：`{report['parent_evidence_dir']}`；保存身份分析表路径：`{report['identity_analysis_dir']}`。结果确认三种子前景收益父判定保持不变，科学更新数为 0。机器验证见 `validation.json`，执行预检见 `preflight.json`。本报告不把云端命令退出码单独视为证据有效性。

## 决策影响与下一步

后续单因素诊断问题：对同一固定 A/B raw 行，只将 B raw CMOS 乘以逐对强度匹配系数，再复用生产 clamp→pool 视图和冻结 E_D，观察预测转移与身份余量是否改变；空间形态、模型、权重、样本和目标保持不变。只测本次选中的 `{recommendation.get('observed_selected_pairs', 0)}` 对，完成后停止。它不增加训练、采样、人眼评分、TEST或模型采用授权。本任务到此停止。
"""


def _preflight(evidence_dir: Path, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    evidence_dir, analysis_dir, output_dir = evidence_dir.resolve(), analysis_dir.resolve(), output_dir.resolve()
    _require(root in evidence_dir.parents and root in analysis_dir.parents, "evidence and analysis tables must be inside the project")
    _require(
        output_dir.parent == (root / OUTPUT_RUN_ROOT).resolve()
        and output_dir.name.startswith("attempt_")
        and output_dir.name.removeprefix("attempt_").isdigit(),
        "output path must be an explicit numbered attempt under the task run root",
    )
    _require(not output_dir.exists(), "output path already exists; refusing to overwrite any prior attempt")
    parent, parent_validation, analysis = _source_report(evidence_dir, analysis_dir)
    rows, source_confusions, dev_positions = _load_population_rows(analysis_dir, evidence_dir)
    pairs = _pair_rows(rows, dev_positions)
    confusion_check = _verify_source_confusions(pairs, source_confusions)
    selected, inventory = _select_pairs(pairs)
    _require(len(inventory) == 2 * 3 * len(FOCUS_CLASSES) * len(TRANSITIONS), "selection inventory omits declared strata")
    expected_seeds = {int(seed) for seed in parent_validation["decision_recomputed_from_saved_chunks"]["seed_decisions"]}
    _require({int(row["training_seed"]) for row in pairs} == expected_seeds, "paired training seed set differs from the parent scientific decision")
    raw_checks = 0
    if selected:
        pair = selected[0]
        source_lookup = {
            (str(row["scenario"]), _record_key(row)): row
            for row in rows
            if _scenario_arm(row) is not None
        }
        key = (
            "empirical_dev", int(pair["training_seed"]), int(pair["source_row_index"])
        ) if pair["population"] == "empirical_dev" else (
            "conditional_sample", int(pair["training_seed"]), int(pair["sample_seed"]),
            int(pair["target_class"]), int(pair["draw_index"]),
        )
        csv_by_scenario = {
            "A": source_lookup[(str(pair["a_scenario"]), key)],
            "B": source_lookup[(str(pair["b_scenario"]), key)],
        }
        for endpoint in ("A", "B"):
            row = csv_by_scenario[endpoint]
            _read_selected_endpoint(evidence_dir, pair, endpoint, row)
            raw_checks += 1
    return {
        "status": "PREFLIGHT_VALID",
        "task_id": TASK_ID,
        "parent_status": parent["status"],
        "parent_classification": parent["classification"],
        "parent_scientific_updates": parent_validation["scientific_updates"],
        "identity_analysis_status": analysis["status"],
        "paired_rows": len(rows),
        "active_ab_rows": 15_000,
        "paired_count": len(pairs),
        "source_confusion_validation": confusion_check,
        "strata_count": len(inventory),
        "selected_pair_count": len(selected),
        "selected_raw_endpoints_checked": raw_checks,
        "raw_readout_replay": "saved production clamp-then-pool view and evaluator logits",
        "scientific_updates": 0,
        "model_forward_or_evaluator_call": False,
        "output_path": output_dir.as_posix(),
    }


def _execute(evidence_dir: Path, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    preflight = _preflight(evidence_dir, analysis_dir, output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    parent, parent_validation, analysis = _source_report(evidence_dir, analysis_dir)
    source_rows, source_confusions, dev_positions = _load_population_rows(analysis_dir, evidence_dir)
    pairs = _pair_rows(source_rows, dev_positions)
    confusion_validation = _verify_source_confusions(pairs, source_confusions)
    selected, strata = _select_pairs(pairs)
    source_lookup = {
        (str(row["scenario"]), _record_key(row)): row
        for row in source_rows
        if _scenario_arm(row) is not None
    }
    for pair in selected:
        if pair["population"] == "empirical_dev":
            a_key = ("empirical_dev", int(pair["training_seed"]), int(pair["source_row_index"]))
            b_key = a_key
        else:
            a_key = (
                "conditional_sample", int(pair["training_seed"]), int(pair["sample_seed"]),
                int(pair["target_class"]), int(pair["draw_index"]),
            )
            b_key = a_key
        a_scenario = str(pair["a_scenario"])
        a_source = source_lookup.get((a_scenario, a_key))
        _require(a_source is not None, f"selected A row disappeared from source table: {a_key}")
        pair["_a_csv_row"] = a_source
        b_scenario = str(pair["b_scenario"])
        b_source = source_lookup.get((b_scenario, b_key))
        _require(b_source is not None, f"selected B row disappeared from source table: {b_key}")
        pair["_b_csv_row"] = b_source

    raw = _read_selected_raw(evidence_dir.resolve(), selected)
    metric_rows: list[dict[str, Any]] = []
    atlas_entries: list[tuple[Mapping[str, Any], Sequence[Mapping[str, str]], Mapping[str, Any]]] = []
    for pair in selected:
        a = raw[(str(pair["sample_key"]), "A")]
        b = raw[(str(pair["sample_key"]), "B")]
        metrics, panels = _pair_metrics_and_panels(pair, a, b)
        metric_rows.append(metrics)
        atlas_entries.append((pair, panels, metrics))

    confusion = _confusion_rows(pairs)
    class_summary = _class_summary_rows(pairs)
    population_summary = _population_identity_summary(pairs)
    next_diagnostic = _recommendation(metric_rows, pairs)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "VALID_DESCRIPTIVE_EVIDENCE",
        "classification": "diagnostic-only",
        "parent_classification_preserved": parent["classification"],
        "parent_seed_decisions_preserved": parent_validation["decision_recomputed_from_saved_chunks"]["seed_decisions"],
        "identity_analysis_status": analysis["status"],
        "paired_rows": len(source_rows),
        "paired_count": len(pairs),
        "selected_pair_count": len(selected),
        "strata_count": len(strata),
        "empty_strata": sum(row["selection_status"] == "empty_stratum" for row in strata),
        "source_confusion_validation": confusion_validation,
        "population_identity_summary": population_summary,
        "next_diagnostic": next_diagnostic,
        "human_readability_claim": False,
        "model_adoption": False,
        "scientific_updates": 0,
        "training_or_evaluator_forward": 0,
        "parent_evidence_dir": evidence_dir.resolve().as_posix(),
        "identity_analysis_dir": analysis_dir.resolve().as_posix(),
        "outputs": [
            "paired_identity_rows.csv", "confusion_transitions.csv", "class_summary.csv",
            "selection_strata.csv", "selected_samples.csv", "selected_sample_metrics.csv",
            "review_atlas.html", "preflight.json", "validation.json", "report.md", "report.json",
        ],
        "stop": True,
    }

    output_pairs = [{key: value for key, value in row.items() if not key.startswith("_")} for row in pairs]
    output_selected = [{key: value for key, value in row.items() if not key.startswith("_")} for row in selected]
    _write_csv(output_dir / "paired_identity_rows.csv", output_pairs)
    _write_csv(output_dir / "confusion_transitions.csv", confusion)
    _write_csv(output_dir / "class_summary.csv", class_summary)
    _write_csv(output_dir / "selection_strata.csv", strata)
    _write_csv(output_dir / "selected_samples.csv", output_selected)
    _write_csv(output_dir / "selected_sample_metrics.csv", metric_rows)
    _write_atlas(output_dir / "review_atlas.html", atlas_entries)

    validation = {
        "status": "VALID_DESCRIPTIVE_EVIDENCE",
        "source_rows": len(source_rows),
        "paired_rows": len(output_pairs),
        "paired_keys_unique_and_complete": True,
        "source_confusion_recomputed_exactly": True,
        "selected_pair_count": len(selected),
        "selection_upper_bound": 288,
        "selection_strata_count": len(strata),
        "selected_raw_blocks": len({
            str(_selected_raw_path(evidence_dir, pair, endpoint)[0])
            for pair in selected
            for endpoint in ("A", "B")
        }),
        "selected_raw_endpoints": 2 * len(selected),
        "raw_to_production_view_replayed": 2 * len(selected),
        "selected_logits_and_margins_replayed": 2 * len(selected),
        "dev_q_fg_independent_numpy_replays": sum(row["population"] == "empirical_dev" for row in selected) * 2,
        "zero_mass_rows_are_explicit": sum(bool(row["a_zero_mass"] or row["b_zero_mass"]) for row in metric_rows),
        "conditional_targets_invented": False,
        "model_or_evaluator_forward": 0,
        "scientific_updates": 0,
        "human_ratings": 0,
        "new_gate_added": False,
    }
    _write_json(output_dir / "validation.json", validation)
    _write_json(output_dir / "report.json", report)
    report_readback = json.loads((output_dir / "report.json").read_text(encoding="utf-8")) == report
    _require(report_readback, "saved final report does not round-trip")
    report["report_readback"] = True
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    _require((output_dir / "review_atlas.html").is_file() and (output_dir / "review_atlas.html").stat().st_size > 0, "review atlas was not written")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.preflight_only:
        result = _preflight(args.evidence_dir, args.analysis_dir, args.output_dir)
    else:
        result = _execute(args.evidence_dir, args.analysis_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
