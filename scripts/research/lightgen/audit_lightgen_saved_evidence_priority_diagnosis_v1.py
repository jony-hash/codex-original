"""Audit fixed LightGen camera and identity evidence without optical updates."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ldm_project.v2.application.research.lightgen.canonical_semantics import cmos_view
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
)
from scripts.research.lightgen import audit_lightgen_2d_generator_identity_stroke_localization_v1 as identity


TASK_ID = "LIGHTGEN_SAVED_EVIDENCE_PRIORITY_DIAGNOSIS_V1"
OUTPUT_RUN_ROOT = Path("outputs/20261019_lightgen_saved_evidence_priority_diagnosis_v1/runs")
EXPECTED_CANDIDATES = ("candidate_2x2_C", "candidate_3x3_B")
EXPECTED_STATES = (("step_000_initial", "initial"), ("step_000500_terminal", "terminal"))
EXPECTED_MODE = "gaussian_s12"
CAMERA_RTL = 2.0e-6
CAMERA_ATOL = 5.0e-7
ED_RTOL = 2.0e-6
ED_ATOL = 5.0e-8
SOURCE_POWER_W = 249.3e-6
V8_UNIFORM_N_EFF = {"2x2-C Gaussian S12": 4.0, "3x3-B Gaussian S12": 9.0}
EXPECTED_CLASS_COUNTS = {"fit": 10, "confirmation": 20}


class EvidenceError(RuntimeError):
    """Saved evidence is missing, mismatched, or numerically invalid."""


class IdentityReplayError(EvidenceError):
    """The fixed CPU evaluator replay failed its predeclared source-logit tolerance."""

    def __init__(self, message: str, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV evidence is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceError(f"refusing to write an empty result table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixed_mean_mse(targets: np.ndarray, role: np.ndarray) -> tuple[float, np.ndarray]:
    fit = targets[role == "fit"]
    _require(len(fit) == 100 and bool(np.isfinite(fit).all()), "v12 fit targets are incomplete or nonfinite")
    mean_image = fit.mean(axis=0, dtype=np.float64)
    confirm = targets[role == "confirmation"]
    errors = np.square(confirm.astype(np.float64) - mean_image[None, ...]).mean(axis=(1, 2, 3))
    return float(errors.mean()), errors


def _class_balanced_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    labels = sorted({int(row["label"]) for row in rows})
    _require(bool(labels), "cannot class-balance an empty sample")
    return statistics.fmean(
        statistics.fmean(float(row[field]) for row in rows if int(row["label"]) == label)
        for label in labels
    )


def _camera_replay(
    power_w_per_pixel: np.ndarray,
    wavelength_m: float,
    exposure_s: float,
    quantum_efficiency: float,
    full_well_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    constants = (wavelength_m, exposure_s, quantum_efficiency, full_well_e)
    _require(all(math.isfinite(float(value)) for value in constants)
             and wavelength_m > 0.0 and exposure_s > 0.0
             and 0.0 <= quantum_efficiency <= 1.0 and full_well_e > 0.0,
             "camera conversion settings are physically invalid")
    _require(bool(np.isfinite(power_w_per_pixel).all()) and np.all(power_w_per_pixel >= 0.0),
             "camera power must be finite and nonnegative")
    photon_energy = 6.62607015e-34 * 299_792_458.0 / wavelength_m
    expected_electrons = np.asarray(power_w_per_pixel, dtype=np.float64) * exposure_s * quantum_efficiency / photon_energy
    task_output = np.clip(expected_electrons / full_well_e, 0.0, 1.0)
    return expected_electrons, task_output


def _camera_error_partition(
    target: np.ndarray,
    expected_electrons: np.ndarray,
    task_output: np.ndarray,
    full_well_e: float,
) -> dict[str, float]:
    _require(target.shape == expected_electrons.shape == task_output.shape,
             "camera partition inputs must have identical pixel geometry")
    _require(bool(np.isfinite(target).all() and np.isfinite(expected_electrons).all()
                  and np.isfinite(task_output).all()), "camera partition inputs must be finite")
    _require(bool(np.all((target >= 0.0) & (target <= 1.0)) and np.all(expected_electrons >= 0.0)),
             "camera partition target/electrons are outside their valid ranges")
    clipped = expected_electrons >= full_well_e
    error = np.square(task_output - target)
    total_error = float(error.sum(dtype=np.float64))
    clipped_count = int(clipped.sum())
    unclipped_count = int((~clipped).sum())
    clipped_error = float(error[clipped].sum(dtype=np.float64))
    unclipped_error = float(error[~clipped].sum(dtype=np.float64))
    return {
        "at_or_above_full_well_pixel_fraction": float(clipped.mean()),
        "at_or_above_full_well_error_fraction": clipped_error / total_error if total_error else 0.0,
        "below_full_well_error_fraction": unclipped_error / total_error if total_error else 0.0,
        "at_or_above_full_well_sq_error_per_pixel": clipped_error / clipped_count if clipped_count else 0.0,
        "below_full_well_sq_error_per_pixel": unclipped_error / unclipped_count if unclipped_count else 0.0,
        "at_or_above_full_well_sq_error_per_image": clipped_error,
        "below_full_well_sq_error_per_image": unclipped_error,
        "total_sq_error_per_image": total_error,
        "task_mse": float(error.mean(dtype=np.float64)),
    }


def _require_same_camera_population(
    reference: Mapping[str, np.ndarray],
    candidate_arrays: Mapping[str, np.ndarray],
    context: str,
) -> None:
    for field in ("rows", "labels", "role", "target"):
        _require(np.array_equal(reference[field], candidate_arrays[field]),
                 f"camera fixed population {field} differs: {context}")


def _power_lower_bound_sorted(target: np.ndarray, available_electrons: float, full_well: float) -> float:
    values = np.asarray(target, dtype=np.float64).clip(0.0, 1.0).reshape(-1)
    if not math.isfinite(available_electrons) or available_electrons < 0.0:
        raise EvidenceError("available electron budget must be finite and nonnegative")
    if not math.isfinite(full_well) or full_well <= 0.0:
        raise EvidenceError("camera full well must be finite and positive")
    budget = min(available_electrons / full_well, float(values.size))
    target_mass = float(values.sum())
    if budget >= target_mass:
        return 0.0
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    valid = np.flatnonzero(ordered - (cumulative - budget) / ranks > 0.0)
    if valid.size == 0:
        return float(np.square(values).mean())
    rho = int(valid[-1])
    theta = (cumulative[rho] - budget) / float(rho + 1)
    ideal = np.maximum(values - theta, 0.0)
    return float(np.square(ideal - values).mean())


def _power_lower_bound_bisection(target: np.ndarray, available_electrons: float, full_well: float) -> float:
    values = np.asarray(target, dtype=np.float64).clip(0.0, 1.0).reshape(-1)
    _require(math.isfinite(available_electrons) and available_electrons >= 0.0,
             "available electron budget must be finite and nonnegative")
    _require(math.isfinite(full_well) and full_well > 0.0,
             "camera full well must be finite and positive")
    budget = min(available_electrons / full_well, float(values.size))
    if budget >= float(values.sum()):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) * 0.5
        total = float(np.clip(values - middle, 0.0, 1.0).sum(dtype=np.float64))
        if total > budget:
            low = middle
        else:
            high = middle
    ideal = np.clip(values - high, 0.0, 1.0)
    return float(np.square(ideal - values).mean(dtype=np.float64))


def _camera_rows(evidence_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    machine = _read_json(evidence_dir / "report.json")
    _require(machine.get("task_id") == "OLS_DECODER_MNIST_RECONSTRUCTION_V12"
             and machine.get("status") == "VALID_FAILED", "named v12 scientific result identity changed")
    settings = machine.get("fixed_config")
    _require(isinstance(settings, dict) and isinstance(settings.get("optics"), dict)
             and isinstance(settings.get("camera"), dict) and isinstance(settings.get("training"), dict),
             "v12 fixed camera settings are missing")
    wavelength = float(settings["optics"]["wavelength_m"])
    source_power = float(settings["optics"]["source_power_w"])
    exposure = float(settings["camera"]["primary_exposure_s"])
    qe = float(settings["camera"]["quantum_efficiency"])
    full_well = float(settings["camera"]["full_well_e"])
    _require(math.isclose(source_power, SOURCE_POWER_W, rel_tol=0.0, abs_tol=1.0e-12),
             "v12 fixed source-power setting differs")
    _require(settings["training"]["primary_mode"] == EXPECTED_MODE
             and tuple(settings["training"]["candidates"]) == EXPECTED_CANDIDATES,
             "v12 fixed candidate or Gaussian mode differs")
    camera_settings = {
        "wavelength_m": wavelength,
        "source_power_w": source_power,
        "primary_exposure_s": exposure,
        "camera_QE": qe,
        "camera_full_well_e": full_well,
    }
    metrics_root = evidence_dir / "measurements"
    rows: list[dict[str, Any]] = []
    gallery_by_candidate: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    reference_population: dict[str, np.ndarray] | None = None
    reference_max_errors: dict[str, float] = {}

    for candidate in EXPECTED_CANDIDATES:
        candidate_arrays: dict[str, dict[str, np.ndarray]] = {}
        candidate_gallery: dict[str, dict[int, dict[str, Any]]] = {}
        for state_dir, state_name in EXPECTED_STATES:
            base = metrics_root / state_dir / candidate / EXPECTED_MODE
            array_path = base / "camera_arrays_4096_1ms.npz"
            table_path = base / "per_image_measurements.csv"
            _require(array_path.is_file(), f"camera arrays are missing: {array_path}")
            table = _read_csv(table_path)
            table_ids = [int(item["row_id"]) for item in table]
            _require(len(table_ids) == len(set(table_ids)), f"measurement table has duplicate row IDs: {candidate}/{state_name}")
            table_by_row = {int(item["row_id"]): item for item in table}
            with np.load(array_path, allow_pickle=False) as archive:
                required = {"rows", "labels", "role", "target", "power_w_per_pixel", "expected_electrons", "task_output"}
                _require(required.issubset(archive.files), f"camera archive schema is incomplete: {array_path}")
                arrays = {key: archive[key].copy() for key in required}
            n = len(arrays["rows"])
            _require(n == 300 and len(table) == n, f"camera population size differs: {candidate}/{state_name}")
            _require(len(np.unique(arrays["rows"])) == n, f"camera archive has duplicate sample rows: {candidate}/{state_name}")
            _require(arrays["labels"].shape == (n,) and arrays["role"].shape == (n,), "camera labels/roles have wrong shape")
            _require(arrays["target"].shape == (n, 1, 200, 200), "camera target geometry differs")
            for key in ("power_w_per_pixel", "expected_electrons", "task_output"):
                _require(arrays[key].shape == (n, 200, 200), f"camera {key} geometry differs")
            _require(bool(np.isfinite(arrays["target"]).all() and np.isfinite(arrays["power_w_per_pixel"]).all()
                          and np.isfinite(arrays["expected_electrons"]).all()
                          and np.isfinite(arrays["task_output"]).all()), "camera archive has nonfinite numeric values")
            _require(np.all((arrays["target"] >= 0.0) & (arrays["target"] <= 1.0)), "camera targets are outside [0,1]")
            _require(np.all(arrays["power_w_per_pixel"] >= 0.0), "camera power contains negative values")
            _require(np.array_equal(arrays["rows"], np.asarray([int(item["row_id"]) for item in table])),
                     f"CSV/NPZ row identity differs: {candidate}/{state_name}")
            _require(np.array_equal(arrays["labels"], np.asarray([int(item["label"]) for item in table])),
                     f"CSV/NPZ label identity differs: {candidate}/{state_name}")
            _require(np.array_equal(arrays["role"], np.asarray([item["role"] for item in table])),
                     f"CSV/NPZ population role differs: {candidate}/{state_name}")
            _require(set(arrays["role"].tolist()) == set(EXPECTED_CLASS_COUNTS),
                     f"camera population roles differ: {candidate}/{state_name}")
            _require(set(int(value) for value in arrays["labels"]) == set(range(10)),
                     f"camera labels do not cover all ten classes: {candidate}/{state_name}")
            _require(all(item["candidate"] == candidate and item["mode"] == EXPECTED_MODE
                         and item["state"] == state_dir for item in table),
                     f"camera table candidate/mode/state identity differs: {candidate}/{state_name}")
            _require(all(int((arrays["role"] == role).sum()) == count * 10
                         for role, count in EXPECTED_CLASS_COUNTS.items()),
                     f"camera fit/confirmation role counts differ: {candidate}/{state_name}")
            for role, expected_per_class in EXPECTED_CLASS_COUNTS.items():
                for label in range(10):
                    _require(int(((arrays["role"] == role) & (arrays["labels"] == label)).sum()) == expected_per_class,
                             f"camera role/class population differs: {candidate}/{state_name}/role={role}/label={label}")
            if state_name == "initial":
                if reference_population is None:
                    reference_population = {
                        field: arrays[field].copy()
                        for field in ("rows", "labels", "role", "target")
                    }
                else:
                    _require_same_camera_population(reference_population, arrays, f"{candidate}/{state_name}")
            else:
                _require_same_camera_population(candidate_arrays["initial"], arrays, f"{candidate}/{state_name}")
                _require(np.array_equal(arrays["target"], candidate_arrays["initial"]["target"].astype(arrays["target"].dtype)),
                         f"initial/terminal target population differs: {candidate}")
                _require(np.array_equal(arrays["rows"], candidate_arrays["initial"]["rows"]),
                         f"initial/terminal rows differ: {candidate}")
            expected_electrons, clipped = _camera_replay(
                arrays["power_w_per_pixel"].astype(np.float64), wavelength, exposure, qe, full_well
            )
            saved_electrons = arrays["expected_electrons"].astype(np.float64)
            electron_error = float(np.max(np.abs(expected_electrons - saved_electrons)))
            _require(np.allclose(expected_electrons, saved_electrons, rtol=4.0e-6, atol=1.0e-3),
                     f"W/pixel-to-electron conversion does not replay: {candidate}/{state_name}")
            _require(np.allclose(clipped, arrays["task_output"], rtol=CAMERA_RTL, atol=CAMERA_ATOL),
                     f"power-to-task camera view does not replay: {candidate}/{state_name}")
            confirm = arrays["role"] == "confirmation"
            _require(int(confirm.sum()) == 200, f"confirmation population differs: {candidate}/{state_name}")
            _fixed_fit_mse, mean_errors = _fixed_mean_mse(arrays["target"], arrays["role"])
            confirm_indices = np.flatnonzero(confirm)
            candidate_gallery[state_name] = {}
            for index in confirm_indices:
                label = int(arrays["labels"][index])
                if label not in candidate_gallery[state_name]:
                    candidate_gallery[state_name][label] = {
                        "row_id": int(arrays["rows"][index]),
                        "target": arrays["target"][index, 0].astype(np.float32),
                        "task_output": clipped[index].astype(np.float32),
                    }
            per_image: list[dict[str, Any]] = []
            for confirm_position, index in enumerate(confirm_indices):
                target = arrays["target"][index, 0].astype(np.float64)
                task = clipped[index]
                partition = _camera_error_partition(target, saved_electrons[index], task, full_well)
                image_mse = partition["task_mse"]
                table_record = table_by_row[int(arrays["rows"][index])]
                saved_mse = float(table_record["primary_task_mse"])
                _require(math.isclose(image_mse, saved_mse, rel_tol=CAMERA_RTL, abs_tol=CAMERA_ATOL),
                         f"camera MSE does not replay: {candidate}/{state_name}/row={arrays['rows'][index]}")
                camera_power = float(arrays["power_w_per_pixel"][index].sum(dtype=np.float64))
                _require(math.isclose(camera_power, float(table_record["camera_pixel_integrated_power_w"]),
                                      rel_tol=2.0e-5, abs_tol=1.0e-15),
                         f"camera power sum does not replay: {candidate}/{state_name}/row={arrays['rows'][index]}")
                total_electrons = float(expected_electrons[index].sum(dtype=np.float64))
                _require(math.isclose(total_electrons, float(table_record["camera_expected_electrons"]),
                                      rel_tol=2.0e-6, abs_tol=1.0),
                         f"camera electron total does not replay: {candidate}/{state_name}/row={arrays['rows'][index]}")
                channel = np.asarray(json.loads(table_record["fiber_channel_power_w"]), dtype=np.float64)
                port_count = 4 if candidate == "candidate_2x2_C" else 9
                _require(channel.shape == (port_count,) and bool(np.isfinite(channel).all())
                         and np.all(channel >= 0.0),
                         f"fiber channel power vector is invalid: {candidate}/{state_name}/row={arrays['rows'][index]}")
                fiber_total = float(channel.sum(dtype=np.float64))
                _require(fiber_total > 0.0 and math.isclose(
                    fiber_total, float(table_record["fiber_total_power_w"]), rel_tol=3.0e-7, abs_tol=1.0e-18
                ), f"fiber channel sum does not replay: {candidate}/{state_name}/row={arrays['rows'][index]}")
                n_eff = 1.0 / float(np.square(channel / fiber_total).sum(dtype=np.float64))
                saved_n_eff = float(table_record["fiber_N_eff"])
                _require(math.isclose(n_eff, saved_n_eff, rel_tol=1.0e-10, abs_tol=1.0e-10),
                         f"fiber N_eff does not replay: {candidate}/{state_name}/row={arrays['rows'][index]}")
                system_coupling = fiber_total / source_power
                before_fiber = float(table_record["before_fiber_roi_power_w"])
                pre_fiber_coupling = fiber_total / before_fiber
                _require(before_fiber > 0.0 and math.isclose(
                    system_coupling, float(table_record["fiber_coupling_system"]), rel_tol=3.0e-7, abs_tol=1.0e-12
                ), f"fiber-to-source coupling does not replay: row={arrays['rows'][index]}")
                _require(math.isclose(
                    pre_fiber_coupling, float(table_record["fiber_coupling_before_fiber"]), rel_tol=3.0e-7, abs_tol=1.0e-12
                ), f"pre-fiber coupling does not replay: row={arrays['rows'][index]}")
                table_lower = float(table_record["ideal_power_constrained_mse_lower_bound"])
                bound = _power_lower_bound_sorted(target, total_electrons, full_well)
                bisection_bound = _power_lower_bound_bisection(target, total_electrons, full_well)
                reference_max_errors[f"{candidate}_{state_name}_sorted_vs_bisection_max_abs"] = max(
                    reference_max_errors.get(f"{candidate}_{state_name}_sorted_vs_bisection_max_abs", 0.0),
                    abs(bound - bisection_bound),
                )
                reference_max_errors[f"{candidate}_{state_name}_sorted_vs_saved_max_abs"] = max(
                    reference_max_errors.get(f"{candidate}_{state_name}_sorted_vs_saved_max_abs", 0.0),
                    abs(bound - table_lower),
                )
                _require(math.isclose(bound, bisection_bound, rel_tol=0.0, abs_tol=1.0e-10),
                         f"sorted projection and original bisection differ: row={arrays['rows'][index]}")
                _require(math.isclose(bound, table_lower, rel_tol=0.0, abs_tol=2.0e-8),
                         f"independent power lower bound does not match saved value: row={arrays['rows'][index]}")
                entry = {
                    "candidate": candidate,
                    "state": state_name,
                    "row_id": int(arrays["rows"][index]),
                    "label": int(arrays["labels"][index]),
                    "source_power_w": source_power,
                    "camera_power_w": camera_power,
                    "expected_electrons": total_electrons,
                    **partition,
                    "fiber_total_power_w": fiber_total,
                    "fiber_coupling_system_recomputed": system_coupling,
                    "fiber_coupling_before_fiber_recomputed": pre_fiber_coupling,
                    "fiber_n_eff_recomputed": n_eff,
                    "fiber_channel_power_w_json": json.dumps(channel.tolist(), separators=(",", ":")),
                    "task_mse": image_mse,
                    "fixed_mean_image_mse": float(mean_errors[confirm_position]),
                    "ideal_power_constrained_mse_lower_bound": bound,
                    "sorted_vs_bisection_abs_error": abs(bound - bisection_bound),
                }
                rows.append(entry)
                per_image.append(entry)
            _require(len(per_image) == 200, "confirmation camera rows were omitted")
            candidate_arrays[state_name] = arrays
            reference_max_errors[f"{candidate}_{state_name}_electron_max_abs"] = electron_error
            reference_max_errors[f"{candidate}_{state_name}_task_max_abs"] = float(
                np.max(np.abs(clipped - arrays["task_output"].astype(np.float64)))
            )
        for label in range(10):
            _require(
                candidate_gallery["initial"][label]["row_id"] == candidate_gallery["terminal"][label]["row_id"],
                f"camera localization row changed across states: {candidate}/label={label}",
            )
        gallery_by_candidate[candidate] = candidate_gallery
        del candidate_arrays

    summaries: list[dict[str, Any]] = []
    for candidate in EXPECTED_CANDIDATES:
        for _, state_name in EXPECTED_STATES:
            subset = [row for row in rows if row["candidate"] == candidate and row["state"] == state_name]
            summaries.append({
                "candidate": candidate,
                "state": state_name,
                "confirmation_rows": len(subset),
                "task_mse_mean": statistics.fmean(float(row["task_mse"]) for row in subset),
                "task_mse_class_balanced": _class_balanced_mean(subset, "task_mse"),
                "fixed_mean_image_mse_mean": _class_balanced_mean(subset, "fixed_mean_image_mse"),
                "power_lower_bound_mean": _class_balanced_mean(subset, "ideal_power_constrained_mse_lower_bound"),
                "at_or_above_full_well_pixel_fraction_class_balanced": _class_balanced_mean(subset, "at_or_above_full_well_pixel_fraction"),
                "at_or_above_full_well_error_fraction_class_balanced": _class_balanced_mean(subset, "at_or_above_full_well_error_fraction"),
                "below_full_well_error_fraction_class_balanced": _class_balanced_mean(subset, "below_full_well_error_fraction"),
                "at_or_above_full_well_sq_error_per_pixel_class_balanced": _class_balanced_mean(subset, "at_or_above_full_well_sq_error_per_pixel"),
                "below_full_well_sq_error_per_pixel_class_balanced": _class_balanced_mean(subset, "below_full_well_sq_error_per_pixel"),
                "at_or_above_full_well_sq_error_per_image_class_balanced": _class_balanced_mean(subset, "at_or_above_full_well_sq_error_per_image"),
                "below_full_well_sq_error_per_image_class_balanced": _class_balanced_mean(subset, "below_full_well_sq_error_per_image"),
                "camera_power_w_mean": statistics.fmean(float(row["camera_power_w"]) for row in subset),
                "expected_electrons_mean": statistics.fmean(float(row["expected_electrons"]) for row in subset),
            })
    transition_summaries: list[dict[str, Any]] = []
    for candidate in EXPECTED_CANDIDATES:
        initial = next(row for row in summaries if row["candidate"] == candidate and row["state"] == "initial")
        terminal = next(row for row in summaries if row["candidate"] == candidate and row["state"] == "terminal")
        transition_summaries.append({
            "candidate": candidate,
            "initial_to_terminal_task_mse_class_balanced_delta": terminal["task_mse_class_balanced"] - initial["task_mse_class_balanced"],
            "initial_to_terminal_at_or_above_full_well_pixel_fraction_delta": terminal["at_or_above_full_well_pixel_fraction_class_balanced"] - initial["at_or_above_full_well_pixel_fraction_class_balanced"],
            "initial_to_terminal_at_or_above_full_well_error_fraction_delta": terminal["at_or_above_full_well_error_fraction_class_balanced"] - initial["at_or_above_full_well_error_fraction_class_balanced"],
            "initial_to_terminal_at_or_above_full_well_sq_error_per_pixel_delta": terminal["at_or_above_full_well_sq_error_per_pixel_class_balanced"] - initial["at_or_above_full_well_sq_error_per_pixel_class_balanced"],
            "initial_to_terminal_below_full_well_sq_error_per_pixel_delta": terminal["below_full_well_sq_error_per_pixel_class_balanced"] - initial["below_full_well_sq_error_per_pixel_class_balanced"],
        })
    return rows, {
        "fixed_camera_settings": camera_settings,
        "max_replay_errors": reference_max_errors,
        "summary": summaries,
        "initial_to_terminal_class_balanced_changes": transition_summaries,
        "scientific_updates": 0,
        "optical_forwards": 0,
    }, {"gallery": gallery_by_candidate}


def _load_selected_pairs(analysis_dir: Path) -> list[dict[str, Any]]:
    report = _read_json(analysis_dir / "report.json")
    _require(report.get("task_id") == "M0_GENERATOR_IDENTITY_STROKE_LOCALIZATION_V1"
             and report.get("identity_analysis_status") == "EVIDENCE_REVIEW_COMPLETED",
             "identity localization report identity differs")
    _require(int(report.get("empty_strata", -1)) == 16, "identity localization empty strata count differs")
    rows = _read_csv(analysis_dir / "selected_samples.csv")
    _validate_selected_pair_table(
        rows,
        expected_count=230,
        expected_population_counts={"empirical_dev": 113, "conditional_sample": 117},
    )
    selected: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for key in ("training_seed", "target_class", "a_position", "b_position", "selection_rank", "stratum_pairs"):
            row[key] = int(row[key])
        for key in ("source_row_index", "sample_seed", "draw_index"):
            row[key] = int(row[key]) if row[key] else ""
        for endpoint in ("a", "b"):
            source_row: dict[str, Any] = {
                "population": row["population"],
                "label": row["target_class"] if row["population"] == "empirical_dev" else "",
                "requested_class": row["target_class"] if row["population"] == "conditional_sample" else "",
                "e_d_identity_margin": row[f"{endpoint}_e_d_identity_margin"],
            }
            for evaluator_name in ("e_s", "e_i", "e_c", "e_d"):
                source_row[f"{evaluator_name}_prediction"] = row[f"{endpoint}_{evaluator_name}_prediction"]
            row[f"_{endpoint}_csv_row"] = source_row
        selected.append(row)
    return selected


def _validate_selected_pair_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_population_counts: Mapping[str, int],
) -> None:
    sample_keys = [str(row.get("sample_key", "")) for row in rows]
    _require(len(rows) == expected_count and all(sample_keys) and len(set(sample_keys)) == expected_count,
             f"selected-pair table must contain {expected_count} unique complete samples")
    actual_counts = {
        population: sum(row.get("population") == population for row in rows)
        for population in expected_population_counts
    }
    _require(actual_counts == dict(expected_population_counts),
             f"selected population allocation differs: {actual_counts}")
    required = (
        "target_class", "a_scenario", "b_scenario", "a_position", "b_position",
        "a_e_d_prediction", "b_e_d_prediction", "a_e_d_identity_margin", "b_e_d_identity_margin",
        "e_d_transition",
    )
    _require(all(all(str(row.get(key, "")) != "" for key in required) for row in rows),
             "selected-pair table has an incomplete endpoint or target row")


def _ed_margin(logits: torch.Tensor, label: int) -> float:
    one = logits.to(torch.float64)
    other = torch.cat((one[:label], one[label + 1 :]))
    return float((one[label] - other.max()).item())


def _run_evaluator(
    evaluator: PixelMLPConfirmationEvaluatorV1,
    raw: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    source = raw.to(torch.float32)
    if scale != 1.0:
        if not math.isfinite(scale) or scale <= 0.0:
            raise EvidenceError("energy match requires a finite positive scale")
        source = source * float(scale)
    view = cmos_view(source)
    with torch.no_grad():
        result = evaluator(view)
    _require(tuple(result.shape) == (len(raw), 10) and bool(torch.isfinite(result).all()),
             "frozen E_D returned invalid logits")
    _require(all(parameter.grad is None for parameter in evaluator.parameters()), "E_D accumulated gradients")
    return result.detach().cpu()


def _energy_match_scale(a: torch.Tensor, b: torch.Tensor) -> float:
    mass_a = float(a.to(torch.float64).sum())
    mass_b = float(b.to(torch.float64).sum())
    _require(math.isfinite(mass_a) and math.isfinite(mass_b) and mass_a > 0.0 and mass_b > 0.0,
             "energy match is undefined for nonfinite or zero-mass A/B input")
    scale = mass_a / mass_b
    _require(math.isfinite(scale) and scale > 0.0, "energy match scale must be finite and positive")
    return scale


def _identity_measurements(
    evidence_dir: Path,
    analysis_dir: Path,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _load_selected_pairs(analysis_dir)
    parent = _read_json(evidence_dir / "report.json")
    _require(parent.get("task_id") == "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1"
             and parent.get("classification") == "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS",
             "identity source training report differs")
    saved = identity._read_selected_raw(evidence_dir, selected)
    saved_logits = {
        key: value["saved_e_d_logits"]
        for key, value in saved.items()
    }
    _require(len(saved_logits) == 460, "saved source logits do not cover all fixed endpoints")
    evaluator = PixelMLPConfirmationEvaluatorV1()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping)
             and payload.get("architecture_kind") == PixelMLPConfirmationEvaluatorV1.architecture_kind,
             "frozen E_D checkpoint architecture differs")
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "frozen E_D checkpoint state is missing")
    evaluator.load_state_dict(state, strict=True)
    evaluator.cpu().freeze().eval()
    before = {name: value.detach().clone() for name, value in evaluator.state_dict().items()}
    record_by_key: dict[tuple[str, str], torch.Tensor] = {}
    all_rows: list[dict[str, Any]] = []
    replay_attempts = 0
    replay_matches = 0

    def predict(row: Mapping[str, Any], endpoint: str, raw: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return _run_evaluator(evaluator, raw, scale)[0]

    for row in selected:
        key = str(row["sample_key"])
        target = int(row["target_class"])
        for endpoint in ("A", "B"):
            source = saved[(key, endpoint)]["raw"]
            old_logits = saved_logits[(key, endpoint)].to(torch.float64)
            _require(bool(torch.isfinite(old_logits).all()), f"saved E_D logits are nonfinite: {key}/{endpoint}")
            pred = int(old_logits.argmax().item())
            csv_pred = int(row[f"{endpoint.lower()}_e_d_prediction"])
            _require(pred == csv_pred, f"saved E_D prediction differs from selected table: {key}/{endpoint}")
            old_margin = _ed_margin(old_logits, target)
            csv_margin = float(row[f"{endpoint.lower()}_e_d_identity_margin"])
            _require(math.isclose(old_margin, csv_margin, rel_tol=ED_RTOL, abs_tol=ED_ATOL),
                     f"saved E_D margin differs from selected table: {key}/{endpoint}")
            fresh = predict(row, endpoint, source, 1.0)
            replay_attempts += 1
            old_float = old_logits.to(torch.float32)
            absolute_delta = (fresh - old_float).abs()
            permitted_delta = ED_ATOL + ED_RTOL * old_float.abs()
            if not torch.allclose(fresh, old_float, rtol=ED_RTOL, atol=ED_ATOL):
                state_unchanged = all(
                    torch.equal(before[name], value)
                    for name, value in evaluator.state_dict().items()
                )
                raise IdentityReplayError(
                    f"CPU E_D replay logits differ from saved CUDA logits: {key}/{endpoint}",
                    {
                        "selected_pairs_validated": len(selected),
                        "raw_endpoints_validated": len(saved),
                        "saved_source_logits_validated": len(saved_logits),
                        "matched_pairs_completed": len(all_rows),
                        "saved_endpoint_cpu_replay_attempts": replay_attempts,
                        "saved_endpoint_cpu_replay_matches": replay_matches,
                        "matched_b_evaluator_forwards": 0,
                        "mismatch_sample_key": key,
                        "mismatch_endpoint": endpoint,
                        "e_d_checkpoint_loaded": True,
                        "mismatch_prediction_agrees": int(fresh.argmax().item()) == pred,
                        "mismatch_max_abs_logit_error": float(absolute_delta.max()),
                        "mismatch_max_excess_over_logit_tolerance": float((absolute_delta - permitted_delta).max()),
                        "logit_tolerance": {"rtol": ED_RTOL, "atol": ED_ATOL},
                        "evaluator_state_unchanged_at_stop": state_unchanged,
                        "evaluator_gradients_absent_at_stop": all(parameter.grad is None for parameter in evaluator.parameters()),
                    },
                )
            replay_matches += 1
            fresh_pred = int(fresh.argmax().item())
            _require(fresh_pred == pred, f"CPU E_D replay prediction differs: {key}/{endpoint}")
            fresh_margin = _ed_margin(fresh, target)
            _require(math.isclose(fresh_margin, old_margin, rel_tol=ED_RTOL, abs_tol=ED_ATOL),
                     f"CPU E_D replay margin differs: {key}/{endpoint}")
            record_by_key[(key, endpoint)] = fresh

        a = saved[(key, "A")]["raw"][0].to(torch.float64)
        b = saved[(key, "B")]["raw"][0].to(torch.float64)
        mass_a, mass_b = float(a.sum()), float(b.sum())
        try:
            scale = _energy_match_scale(a, b)
        except EvidenceError as error:
            raise EvidenceError(f"energy match is undefined for zero-mass A/B pair: {key}") from error
        matched = predict(row, "B", saved[(key, "B")]["raw"], scale)
        old_a, old_b = record_by_key[(key, "A")], record_by_key[(key, "B")]
        pred_a, pred_b, pred_matched = int(old_a.argmax()), int(old_b.argmax()), int(matched.argmax())
        margin_a = _ed_margin(old_a, target)
        margin_b = _ed_margin(old_b, target)
        margin_matched = _ed_margin(matched, target)
        _require(bool(all(torch.isfinite(parameter).all() for parameter in evaluator.parameters())),
                 "E_D parameters became nonfinite")
        b_correct_before = pred_b == target
        b_correct_after = pred_matched == target
        all_rows.append({
            "sample_key": key,
            "population": row["population"],
            "training_seed": int(row["training_seed"]),
            "target_class": target,
            "e_d_transition_original": row["e_d_transition"],
            "selection_rank": int(row["selection_rank"]),
            "stratum_pairs": int(row["stratum_pairs"]),
            "source_row_index": row["source_row_index"],
            "sample_seed": row["sample_seed"],
            "draw_index": row["draw_index"],
            "a_mass": mass_a,
            "b_mass": mass_b,
            "b_to_a_scale_before_clipping": scale,
            "a_prediction": pred_a,
            "b_prediction": pred_b,
            "matched_b_prediction": pred_matched,
            "a_margin": margin_a,
            "b_margin": margin_b,
            "matched_b_margin": margin_matched,
            "b_margin_change_after_matching": margin_matched - margin_b,
            "b_correct_before_matching": b_correct_before,
            "b_correct_after_matching": b_correct_after,
            "b_became_wrong_after_matching": b_correct_before and not b_correct_after,
            "b_recovered_after_matching": not b_correct_before and b_correct_after,
            "prediction_changed_after_matching": pred_matched != pred_b,
            "a_raw_clip_fraction": float((a > 1.0).to(torch.float64).mean()),
            "b_raw_clip_fraction": float((b > 1.0).to(torch.float64).mean()),
        })
    after = evaluator.state_dict()
    _require(all(torch.equal(before[name], after[name]) for name in before), "frozen E_D state changed during evaluation")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters()),
             "frozen E_D gradient or trainability isolation failed")
    summary: dict[str, Any] = {
        "selected_pairs": len(all_rows),
        "empirical_dev_pairs": sum(row["population"] == "empirical_dev" for row in all_rows),
        "conditional_sample_pairs": sum(row["population"] == "conditional_sample" for row in all_rows),
        "empty_strata_preserved": 16,
        "saved_endpoint_cpu_replay_attempts": replay_attempts,
        "saved_endpoint_cpu_replay_matches": replay_matches,
        "matched_b_evaluator_forwards": 230,
        "b_predictions_changed": sum(bool(row["prediction_changed_after_matching"]) for row in all_rows),
        "b_correct_before": sum(bool(row["b_correct_before_matching"]) for row in all_rows),
        "b_correct_after": sum(bool(row["b_correct_after_matching"]) for row in all_rows),
        "b_became_wrong_after_matching": sum(bool(row["b_became_wrong_after_matching"]) for row in all_rows),
        "b_recovered_after_matching": sum(bool(row["b_recovered_after_matching"]) for row in all_rows),
        "scientific_updates": 0,
        "optical_forwards": 0,
        "sampling_calls": 0,
        "human_ratings": 0,
        "official_test_reads": 0,
        "evaluator_state_unchanged": True,
    }
    for population in ("empirical_dev", "conditional_sample"):
        group = [row for row in all_rows if row["population"] == population]
        summary[population] = {
            "pairs": len(group),
            "prediction_changes": sum(bool(row["prediction_changed_after_matching"]) for row in group),
            "b_correct_before": sum(bool(row["b_correct_before_matching"]) for row in group),
            "b_correct_after": sum(bool(row["b_correct_after_matching"]) for row in group),
            "b_became_wrong_after_matching": sum(bool(row["b_became_wrong_after_matching"]) for row in group),
            "b_recovered_after_matching": sum(bool(row["b_recovered_after_matching"]) for row in group),
            "mean_b_margin_change": statistics.fmean(float(row["b_margin_change_after_matching"]) for row in group),
        }
    for transition in sorted({str(row["e_d_transition_original"]) for row in all_rows}):
        group = [row for row in all_rows if row["e_d_transition_original"] == transition]
        summary.setdefault("by_original_transition", {})[transition] = {
            "pairs": len(group),
            "prediction_changes": sum(bool(row["prediction_changed_after_matching"]) for row in group),
            "mean_b_margin_change": statistics.fmean(float(row["b_margin_change_after_matching"]) for row in group),
        }
    for field in ("population", "training_seed", "target_class", "e_d_transition_original"):
        summary[f"by_{field}"] = {}
        for value in sorted({str(row[field]) for row in all_rows}):
            group = [row for row in all_rows if str(row[field]) == value]
            summary[f"by_{field}"][value] = {
                "pairs": len(group),
                "prediction_changes": sum(bool(row["prediction_changed_after_matching"]) for row in group),
                "b_correct_before": sum(bool(row["b_correct_before_matching"]) for row in group),
                "b_correct_after": sum(bool(row["b_correct_after_matching"]) for row in group),
                "b_became_wrong_after_matching": sum(bool(row["b_became_wrong_after_matching"]) for row in group),
                "b_recovered_after_matching": sum(bool(row["b_recovered_after_matching"]) for row in group),
                "mean_b_margin_change": statistics.fmean(float(row["b_margin_change_after_matching"]) for row in group),
            }
    return all_rows, summary


def _power_rows(camera_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_image = [
        {
            "candidate": row["candidate"],
            "state": row["state"],
            "row_id": row["row_id"],
            "label": row["label"],
            "fiber_total_power_w_recomputed": row["fiber_total_power_w"],
            "camera_power_w": row["camera_power_w"],
            "camera_to_fiber_power_fraction": row["camera_power_w"] / row["fiber_total_power_w"],
            "fiber_coupling_system_recomputed": row["fiber_coupling_system_recomputed"],
            "fiber_coupling_before_fiber_recomputed": row["fiber_coupling_before_fiber_recomputed"],
            "fiber_channel_power_w_json": row["fiber_channel_power_w_json"],
            "n_eff_recomputed": row["fiber_n_eff_recomputed"],
            "power_lower_bound": row["ideal_power_constrained_mse_lower_bound"],
            "task_mse": row["task_mse"],
            "fixed_mean_image_mse": row["fixed_mean_image_mse"],
            "sorted_vs_bisection_abs_error": row["sorted_vs_bisection_abs_error"],
        }
        for row in camera_rows
        if row["candidate"] == "candidate_3x3_B"
    ]
    by_state: dict[str, dict[str, Any]] = {}
    for state_name in ("initial", "terminal"):
        state_rows = [row for row in per_image if row["state"] == state_name]
        _require(len(state_rows) == 200, f"nine-channel {state_name} power population is incomplete")
        by_state[state_name] = {
            "confirmation_rows": len(state_rows),
            "mean_fiber_power_w": statistics.fmean(row["fiber_total_power_w_recomputed"] for row in state_rows),
            "mean_camera_power_w": statistics.fmean(row["camera_power_w"] for row in state_rows),
            "mean_camera_to_fiber_power_fraction": statistics.fmean(row["camera_to_fiber_power_fraction"] for row in state_rows),
            "mean_fiber_coupling_system": statistics.fmean(row["fiber_coupling_system_recomputed"] for row in state_rows),
            "mean_fiber_coupling_before_fiber": statistics.fmean(row["fiber_coupling_before_fiber_recomputed"] for row in state_rows),
            "mean_n_eff": statistics.fmean(row["n_eff_recomputed"] for row in state_rows),
            "n_eff_quantiles": np.quantile([row["n_eff_recomputed"] for row in state_rows], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
            "rows_n_eff_below_4": sum(row["n_eff_recomputed"] < 4.0 for row in state_rows),
            "rows_n_eff_below_2": sum(row["n_eff_recomputed"] < 2.0 for row in state_rows),
            "power_lower_bound_class_balanced": _class_balanced_mean(state_rows, "power_lower_bound"),
            "actual_task_mse_class_balanced": _class_balanced_mean(state_rows, "task_mse"),
            "fixed_mean_image_mse_class_balanced": _class_balanced_mean(state_rows, "fixed_mean_image_mse"),
            "power_lower_bound_per_image_mean": statistics.fmean(row["power_lower_bound"] for row in state_rows),
            "actual_task_mse_per_image_mean": statistics.fmean(row["task_mse"] for row in state_rows),
            "fixed_mean_image_mse_per_image_mean": statistics.fmean(row["fixed_mean_image_mse"] for row in state_rows),
            "per_class": {
                str(label): {
                    "power_lower_bound": statistics.fmean(float(row["power_lower_bound"]) for row in state_rows if int(row["label"]) == label),
                    "actual_task_mse": statistics.fmean(float(row["task_mse"]) for row in state_rows if int(row["label"]) == label),
                    "fixed_mean_image_mse": statistics.fmean(float(row["fixed_mean_image_mse"]) for row in state_rows if int(row["label"]) == label),
                    "mean_n_eff": statistics.fmean(float(row["n_eff_recomputed"]) for row in state_rows if int(row["label"]) == label),
                }
                for label in range(10)
            },
        }
    nine_channel_summary = {
        **by_state["terminal"],
        "v8_uniform_input_neff_reference": V8_UNIFORM_N_EFF["3x3-B Gaussian S12"],
        "v8_reference_population": "synthetic uniform t=0.5 command, not MNIST population",
        "interpretation": "v8均匀输入仅作通道均匀性的参照；输入与人口不同，不能直接作同人口效果比较。",
    }
    return per_image, {"nine_channel_task_input": nine_channel_summary, "nine_channel_by_state": by_state}


def _recommendation(camera: Mapping[str, Any], power: Mapping[str, Any], identity_summary: Mapping[str, Any]) -> dict[str, Any]:
    terminal = {(row["candidate"], row["state"]): row for row in camera["summary"]}
    four = terminal[("candidate_2x2_C", "terminal")]
    nine = power["nine_channel_task_input"]
    # This is a prioritization of already measured gaps, not a new scientific gate.
    if (
        float(four["at_or_above_full_well_pixel_fraction_class_balanced"]) > 0.0
        and float(four["at_or_above_full_well_error_fraction_class_balanced"])
        > float(four["at_or_above_full_well_pixel_fraction_class_balanced"])
    ):
        chosen = "four_channel_fixed_readout_dynamic_range_diagnostic"
        reason = "四通道终点的满阱及以上像素承担的类别等权误差份额，高于其面积份额；先验证固定读出是否能保留这些像素的有效误差信号。"
    elif float(nine["power_lower_bound_class_balanced"]) > float(nine["fixed_mean_image_mse_class_balanced"]):
        chosen = "nine_channel_fixed_power_feasibility_diagnostic"
        reason = "九通道当前捕获功率下的理想空间分配下界高于固定均值图误差；先验证固定读出功率是否足以支撑该基准。"
    elif int(identity_summary["b_predictions_changed"]) > 0:
        chosen = "frozen_output_readout_scale_sensitivity"
        reason = "固定230对中有B样本在A/B总强度匹配后改变E_D预测；读出尺度应纳入后续生成对照。"
    else:
        chosen = "lightgen400_fast_operator_qualification"
        reason = "现有读出与功率证据未单独指定一个可立即短训的解释；先完成大架构传播值、梯度及真实批次成本资格。"
    return {"single_priority": chosen, "reason": reason, "new_scientific_gate": False}


def _png_data(image: np.ndarray) -> str:
    from PIL import Image

    pixels = (np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    tile = Image.fromarray(pixels, mode="L").resize((96, 96), resample=Image.Resampling.NEAREST)
    stream = io.BytesIO()
    tile.save(stream, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _html_report(
    report: Mapping[str, Any],
    camera_gallery: Mapping[str, Any],
    camera_rows: Sequence[Mapping[str, Any]],
    power_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> str:
    summaries = report["camera"]["summary"]
    body = "\n".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row["candidate"], row["state"], f"{row['task_mse_class_balanced']:.6g}",
                f"{row['fixed_mean_image_mse_mean']:.6g}", f"{row['power_lower_bound_mean']:.6g}",
                f"{row['at_or_above_full_well_pixel_fraction_class_balanced']:.4f}",
                f"{row['at_or_above_full_well_error_fraction_class_balanced']:.4f}",
                f"{row['camera_power_w_mean']:.4g}",
            )
        ) + "</tr>"
        for row in summaries
    )
    if report["branches"]["identity_scale"]["status"] == "VALID":
        identity_stats = report["identity"]["summary"]
        conditional = identity_stats["conditional_sample"]
        dev = identity_stats["empirical_dev"]
        identity_section = f"""<h2>固定230对强度反事实</h2><table><tr><th>人口</th><th>样本对数</th><th>B预测改变数</th><th>B正确数匹配前</th><th>匹配后</th><th>B余量平均变化</th></tr>
<tr><td>经验DEV</td><td>{dev['pairs']}</td><td>{dev['prediction_changes']}</td><td>{dev['b_correct_before']}</td><td>{dev['b_correct_after']}</td><td>{dev['mean_b_margin_change']:.6g}</td></tr>
<tr><td>条件采样</td><td>{conditional['pairs']}</td><td>{conditional['prediction_changes']}</td><td>{conditional['b_correct_before']}</td><td>{conditional['b_correct_after']}</td><td>{conditional['mean_b_margin_change']:.6g}</td></tr></table>"""
    else:
        detail = report["branches"]["identity_scale"]
        identity_section = (
            "<h2>固定230对强度反事实</h2><p>本分支因冻结E_D CPU重放超出预设logit容差而停止；"
            f"样本 {html.escape(str(detail.get('mismatch_sample_key', '未定位')))} / "
            f"{html.escape(str(detail.get('mismatch_endpoint', '')))}，最大容差超额 "
            f"{html.escape(str(detail.get('mismatch_max_excess_over_logit_tolerance', '未知')))}。"
            "未执行B的总强度匹配；本分支不作尺度敏感结论。</p>"
        )
    gallery_rows: list[str] = []
    labels = sorted(camera_gallery["candidate_2x2_C"]["terminal"])
    headings = ("目标", "2×2 初始", "2×2 终点", "3×3 初始", "3×3 终点")
    for label in labels:
        reference = camera_gallery["candidate_2x2_C"]["initial"][label]
        cells = [f"<td>{label} / row {reference['row_id']}</td>", f"<td><img src=\"{_png_data(reference['target'])}\"></td>"]
        for candidate, state in (
            ("candidate_2x2_C", "initial"), ("candidate_2x2_C", "terminal"),
            ("candidate_3x3_B", "initial"), ("candidate_3x3_B", "terminal"),
        ):
            item = camera_gallery[candidate][state][label]
            _require(item["row_id"] == reference["row_id"], f"HTML camera gallery row mismatch: label={label}")
            cells.append(f"<td><img src=\"{_png_data(item['task_output'])}\"></td>")
        gallery_rows.append("<tr>" + "".join(cells) + "</tr>")
    gallery_header = "<tr><th>类别／样本</th><th>目标</th>" + "".join(
        f"<th>{heading}</th>" for heading in headings[1:]
    ) + "</tr>"
    power = report["power"]["nine_channel_task_input"]
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LightGen saved evidence diagnosis</title>
<style>body{{font:15px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#18202a}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}td,th{{border:1px solid #ccd2da;padding:.45rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{color:#435368}}img{{width:72px;height:72px;image-rendering:pixelated}}.gallery td{{text-align:center}}</style>
<h1>LightGen保存证据诊断</h1><p>固定样本回顾；无光学前向、训练更新、采样或人评。状态：{html.escape(str(report['status']))}</p>
<h2>四通道与相机功率</h2><table><thead><tr><th>候选／状态</th><th>类别等权 MSE</th><th>固定均值图 MSE</th><th>理想功率下界</th><th>满阱及以上像素</th><th>这些像素承担的误差份额</th><th>相机功率(W)</th></tr></thead><tbody>{body}</tbody></table>
<p class="note">功率下界允许理想空间分配，仅约束当前捕获功率，不表示Decoder可达。裁剪像素分解为描述性统计，不证明全网梯度贡献。</p>
<h2>固定样本图像定位</h2><p class="note">每类取固定确认人口中的首个样本；所有图使用共同的灰度范围 [0,1]。</p><table class="gallery"><thead>{gallery_header}</thead><tbody>{"".join(gallery_rows)}</tbody></table>
<h2>九通道功率与利用</h2><p>终点均值：捕获功率 {power['mean_fiber_power_w']:.4g} W，相机功率 {power['mean_camera_power_w']:.4g} W，系统耦合 {power['mean_fiber_coupling_system']:.4g}，N_eff {power['mean_n_eff']:.4g}；N_eff&lt;4：{power['rows_n_eff_below_4']}/200，N_eff&lt;2：{power['rows_n_eff_below_2']}/200。</p>
<p>终点类别等权：功率下界 {power['power_lower_bound_class_balanced']:.6g}，实际 MSE {power['actual_task_mse_class_balanced']:.6g}，固定均值图 {power['fixed_mean_image_mse_class_balanced']:.6g}。v8 的 N_eff=9 来自不同的均匀输入，不作同人口效果比较。</p>
{identity_section}
<h2>优先行动</h2><p><b>{html.escape(report['decision']['single_priority'])}</b>：{html.escape(report['decision']['reason'])}</p>
<p>相机逐图记录 {len(camera_rows)} 条；九通道逐图记录 {len(power_rows)} 条；身份反事实 {len(identity_rows)} 对。所选样本和分层不能外推为总体发生率或人眼识读。</p>"""


def _closeout_markdown(report: Mapping[str, Any]) -> str:
    four = next(row for row in report["camera"]["summary"]
                if row["candidate"] == "candidate_2x2_C" and row["state"] == "terminal")
    nine = report["power"]["nine_channel_task_input"]
    identity_summary = report["identity"]["summary"]
    four_change = next(
        row["initial_to_terminal_task_mse_class_balanced_delta"]
        for row in report["camera"]["initial_to_terminal_class_balanced_changes"]
        if row["candidate"] == "candidate_2x2_C"
    )
    power_conclusion = (
        "支持当前捕获功率预算会阻断固定均值基准"
        if nine["power_lower_bound_class_balanced"] > nine["fixed_mean_image_mse_class_balanced"]
        else "削弱仅凭当前捕获功率预算即可解释固定均值基准差距"
    )
    identity_conclusion = (
        f"支持样本级读出尺度敏感：匹配后 {identity_summary['b_predictions_changed']} 个B预测改变"
        if identity_summary["b_predictions_changed"]
        else "未观察到本次匹配改变B预测；这只表示本次匹配未消除分类差异"
    )
    return f"""# LightGen保存证据优先级诊断

结果与状态：{report["status"]}。三个分支消费固定保存证据；相机确认记录 {report["counts"]["camera_confirmation_rows"]} 条，九通道功率记录 {report["counts"]["power_rows"]} 条，身份强度反事实 {report["counts"]["identity_selected_pairs"]} 对。实际执行 {report["counts"]["saved_endpoint_evaluator_forwards"]} 次原始端点 E_D CPU 重放和 {report["counts"]["energy_matched_e_d_forwards"]} 次匹配强度前向；光学前向、采样、梯度与训练更新均为 0。

科学认识：A（四通道）以区域误差份额和面积份额的比较判断是否集中：终点类别等权的满阱及以上像素面积份额为 {four["at_or_above_full_well_pixel_fraction_class_balanced"]:.4f}，其平方误差份额为 {four["at_or_above_full_well_error_fraction_class_balanced"]:.4f}，低于满阱区域误差份额为 {four["below_full_well_error_fraction_class_balanced"]:.4f}；初始到终点 MSE 变化为 {four_change:.6g}。严格超过满阱的裁剪局部功率导数为零，边界点处不可微；上述描述不推出整网梯度或因果贡献。B（九通道）{power_conclusion}：类别等权的功率下界 {nine["power_lower_bound_class_balanced"]:.6g}，实际 MSE {nine["actual_task_mse_class_balanced"]:.6g}，固定均值图 MSE {nine["fixed_mean_image_mse_class_balanced"]:.6g}；终点 mean N_eff={nine["mean_n_eff"]:.4g}，N_eff&lt;4 的样本为 {nine["rows_n_eff_below_4"]}/200。下界不是Decoder可达预测，实际差距仍可来自空间分配、表征或优化。C（固定230对）{identity_conclusion}，其中经验DEV {identity_summary["empirical_dev"]["prediction_changes"]}/113、条件采样 {identity_summary["conditional_sample"]["prediction_changes"]}/117；不外推总体发生率，也不证明结构因果。

目标缺口：本诊断没有测量真实相机/硬件的动态范围和噪声，也没有取得相位Decoder可达性或复场干涉证据；没有补足完整生成质量、人眼可读性、跨人口泛化和模型采用所需证据。下界约束当前捕获功率，不说明相位Decoder一定能达到它；无复场时不归因背景透射或相位相消。

工程完成与验证：核验固定样本行号、标签、fit/confirmation角色、类别数、候选与状态、数组几何和有限值；从W/pixel独立复算电子数、裁剪输出与逐图MSE，并复核相机功率和通道功率。排序投影与原二分法、保存的功率下界逐图对照；九通道总功率、系统/入纤耦合及N_eff从通道向量独立复算。固定E_D使用CPU、eval、无梯度前向，参数与buffer保持逐项相同；原端点logits容差为 rtol=2e-6、atol=5e-8。最近RS工程成本参考为单跳缓存complex128 10.43秒、complex64 5.27秒；完整剩余链路估算4.20小时后停止，batch10未测。本记录只解释工程停止点，不是传播资格或训练性能证据。证据见 report.json、逐图CSV及HTML。耗时 {report["elapsed_seconds_observation"]:.1f} 秒。报告是保存证据诊断，不构成新的模型采用Gate、训练许可或硬件就绪结论。

决策影响与下一步：本轮唯一优先项为 {report["decision"]["single_priority"]}：{report["decision"]["reason"]}。这是后继工作的建议；本轮已授权内容到此停止，尚未授权的算子验收或短训不自动启动。若本轮证据未指向单一读出/功率限制，则先做大架构快速算子的实际成本与数值资格，再决定是否安排效果短训。
"""


def _partial_closeout_markdown(report: Mapping[str, Any]) -> str:
    branches = report["branches"]
    branch_lines = "\n".join(
        f"- {name}: {value['status']}；{value.get('error', '数据与数值检查通过')}"
        for name, value in branches.items()
    )
    camera_text = "相机分支无有效结果。"
    power_text = "九通道功率分支无有效结果。"
    if branches["camera_saturation"]["status"] == "VALID":
        four = next(row for row in report["camera"]["summary"]
                    if row["candidate"] == "candidate_2x2_C" and row["state"] == "terminal")
        change = next(row["initial_to_terminal_task_mse_class_balanced_delta"]
                      for row in report["camera"]["initial_to_terminal_class_balanced_changes"]
                      if row["candidate"] == "candidate_2x2_C")
        camera_text = (
            f"终点四通道满阱及以上像素面积份额 {four['at_or_above_full_well_pixel_fraction_class_balanced']:.4f}，"
            f"误差份额 {four['at_or_above_full_well_error_fraction_class_balanced']:.4f}，"
            f"未满阱区域误差份额 {four['below_full_well_error_fraction_class_balanced']:.4f}，"
            f"类别等权MSE初始到终点变化 {change:.6g}。"
        )
    if branches["nine_channel_power"]["status"] == "VALID":
        nine = report["power"]["nine_channel_task_input"]
        power_text = (
            f"九通道终点类别等权功率下界 {nine['power_lower_bound_class_balanced']:.6g}，"
            f"实际MSE {nine['actual_task_mse_class_balanced']:.6g}，"
            f"固定均值图 {nine['fixed_mean_image_mse_class_balanced']:.6g}；"
            f"捕获功率均值 {nine['mean_fiber_power_w']:.4g} W，mean N_eff={nine['mean_n_eff']:.4g}，"
            f"N_eff&lt;4 为 {nine['rows_n_eff_below_4']}/200。"
        )
    identity = branches["identity_scale"]
    mismatch = ""
    if "mismatch_max_abs_logit_error" in identity:
        mismatch = (
            f"首个失败为 {identity['mismatch_sample_key']}/{identity['mismatch_endpoint']}；"
            f"最大logit绝对差 {identity['mismatch_max_abs_logit_error']:.6g}，"
            f"超过预设逐logit容差最大值 {identity['mismatch_max_excess_over_logit_tolerance']:.6g}；"
            f"预测一致={identity['mismatch_prediction_agrees']}。"
        )
    decision = report["decision"]
    return f"""# LightGen保存证据优先级诊断

结果与状态：{report["status"]}，不是整体完成。分支情况：
{branch_lines}

科学认识：A（四通道）结果：{camera_text}面积加权误差比较支持固定代理相机下饱和区集中承担误差，但不推出整网梯度或因果。B（九通道）结果：{power_text}下界若低于均值图只削弱“功率预算单独阻断基准”的解释，不证明Decoder可达，也不排除分配、表征或优化缺口。C（固定230对）未形成有效结论：{mismatch} 没有执行强度匹配，不声称读出尺度敏感或不敏感。

目标缺口：身份尺度分支未完成，因此还缺固定230对的匹配前后预测、余量及正确性转移。诊断也没有测量真实相机硬件动态范围/噪声、相位Decoder可达性、复场干涉、完整生成质量、人眼可读性或跨人口泛化。没有复场证据时不归因背景透射或相位相消。

工程完成与验证：相机/功率分支按固定保存样本核验行号、标签、人口角色、数组几何、W/pixel到电子数/裁剪/MSE重放、通道总功率、耦合与N_eff；两候选及两状态的行号、标签、角色与目标数组逐项一致。排序投影逐图与原二分法相合，排序结果与v12原表差异不超过相机数组float32序列化所致的观测范围。身份分支完整加载并核验固定230对、460个原始端点与保存logits，再加载原E_D checkpoint，按原生产视图运行CPU eval、无梯度重放；在预设 rtol=2e-6、atol=5e-8 下停止，{mismatch} 参数与buffer在停止点不变；匹配强度前向为0。保存端点复放尝试 {report["counts"]["saved_endpoint_evaluator_forwards"]} 次、在容差内 {report["counts"]["saved_endpoint_replays_within_tolerance"]} 次；完整配对结果 {report["counts"]["identity_pair_results_completed"]} 对，匹配强度前向 {report["counts"]["energy_matched_e_d_forwards"]} 次。光学前向、训练更新、梯度、采样均为0。最新RS缓存报告只作成本背景：单跳缓存complex128/complex64分别10.43/5.27秒，完整剩余链路估算4.20小时后工程停止，batch10成本未测；它不证明整链资格或训练速度。

决策影响与下一步：当前唯一优先项为 {decision["single_priority"]}：{decision["reason"]}。{decision.get("decision_scope", "")}。本轮只建议后继工作，没有自动启动光纤/相机可行性诊断、快速算子验收或效果短训；科学结果仍限于有效分支及其固定人口。
"""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-evidence-dir", required=True, type=Path)
    parser.add_argument("--identity-evidence-dir", required=True, type=Path)
    parser.add_argument("--identity-analysis-dir", required=True, type=Path)
    parser.add_argument("--ed-checkpoint", required=True, type=Path)
    parser.add_argument("--rs-stop-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _rs_stop_context(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"latest cached RS stop report is missing: {path}")
    content = path.read_text(encoding="utf-8")
    required_markers = (
        "ENGINEERING_BUDGET_STOP_AFTER_OPERATOR",
        "15,119 秒（4.20 小时）",
        "10.43秒",
        "5.27秒",
        "11.14 GiB",
        "batch10",
    )
    _require(all(marker in content for marker in required_markers),
             "latest cached RS stop report identity or declared cost boundary differs")
    return {
        "path": path.resolve().as_posix(),
        "status": "ENGINEERING_BUDGET_STOP_AFTER_OPERATOR",
        "cache_complex128_single_hop_seconds": 10.43,
        "cache_complex64_single_hop_seconds": 5.27,
        "complex128_peak_reserved_gib": 11.14,
        "estimated_remaining_seconds": 15119,
        "estimated_remaining_hours": 4.20,
        "batch10_cost_measured": False,
        "use_limit": "仅解释最近工程停止点和成本边界；不作为本轮相机、功率或身份科学证据，也不验证整链、梯度或训练资格。",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    v12_dir = args.v12_evidence_dir.resolve()
    identity_dir = args.identity_evidence_dir.resolve()
    analysis_dir = args.identity_analysis_dir.resolve()
    checkpoint = args.ed_checkpoint.resolve()
    rs_stop_report_path = args.rs_stop_report.resolve()
    output_dir = args.output_dir.resolve()
    root = Path(__file__).resolve().parents[3]
    _require(root in v12_dir.parents and root in identity_dir.parents and root in analysis_dir.parents
             and root in output_dir.parents, "all evidence and output paths must stay inside the project")
    expected_output_dir = (root / OUTPUT_RUN_ROOT / "attempt_01").resolve()
    _require(output_dir == expected_output_dir, f"output target must be the named attempt_01: {expected_output_dir}")
    _require(checkpoint.is_file(), f"explicit E_D checkpoint is missing: {checkpoint}")
    rs_stop_context = _rs_stop_context(rs_stop_report_path)
    if args.execute:
        _require(not output_dir.exists(), f"refusing to overwrite an existing output directory: {output_dir}")
    branches: dict[str, dict[str, Any]] = {}
    camera_rows: list[dict[str, Any]] = []
    camera_summary: dict[str, Any] = {}
    camera_context: dict[str, Any] = {}
    power_rows: list[dict[str, Any]] = []
    power_summary: dict[str, Any] = {}
    identity_rows: list[dict[str, Any]] = []
    identity_summary: dict[str, Any] = {}
    identity_failure_details: dict[str, Any] = {}

    try:
        camera_rows, camera_summary, camera_context = _camera_rows(v12_dir)
        branches["camera_saturation"] = {"status": "VALID"}
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        branches["camera_saturation"] = {"status": "INVALID_HARNESS", "error": f"{type(error).__name__}: {error}"}

    if branches["camera_saturation"]["status"] == "VALID":
        try:
            power_rows, power_summary = _power_rows(camera_rows)
            branches["nine_channel_power"] = {"status": "VALID"}
        except (EvidenceError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            branches["nine_channel_power"] = {"status": "INVALID_HARNESS", "error": f"{type(error).__name__}: {error}"}
    else:
        branches["nine_channel_power"] = {"status": "BLOCKED_DEPENDENCY", "error": "camera evidence did not pass identity/replay checks"}

    try:
        identity_rows, identity_summary = _identity_measurements(identity_dir, analysis_dir, checkpoint)
        branches["identity_scale"] = {"status": "VALID"}
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if isinstance(error, IdentityReplayError):
            identity_failure_details = error.details
        branches["identity_scale"] = {
            "status": "INVALID_HARNESS",
            "error": f"{type(error).__name__}: {error}",
            **identity_failure_details,
        }

    all_branches_valid = all(value["status"] == "VALID" for value in branches.values())
    if all_branches_valid:
        decision = _recommendation(camera_summary, power_summary, identity_summary)
        decision["decision_scope"] = "all three saved-evidence branches valid"
    elif branches["camera_saturation"]["status"] == "VALID" and branches["nine_channel_power"]["status"] == "VALID":
        ab_priority = _recommendation(camera_summary, power_summary, {"b_predictions_changed": 0})
        if ab_priority["single_priority"] in {
            "four_channel_fixed_readout_dynamic_range_diagnostic",
            "nine_channel_fixed_power_feasibility_diagnostic",
        }:
            decision = {
                **ab_priority,
                "decision_scope": "valid camera/power branches only; identity scale branch invalid and unresolved",
                "identity_scale_result_available": False,
            }
        else:
            decision = {
                "single_priority": "identity_cpu_evaluator_replay_parity_diagnostic",
                "reason": "相机/功率分支未单独指定读出或功率可行性；身份尺度分支在CPU logit复放处超差，先查明数值复放差异以完成固定证据诊断。",
                "decision_scope": "engineering prerequisite only; identity scale branch invalid",
                "identity_scale_result_available": False,
                "new_scientific_gate": False,
            }
    else:
        decision = {
            "single_priority": "complete_invalid_saved_evidence_branch",
            "reason": "至少一个保存证据分支未通过输入/生产重放检查；先修复并复核该工程路径，再决定科学后继行动。",
            "decision_scope": "no cross-branch scientific conclusion",
            "identity_scale_result_available": False,
            "new_scientific_gate": False,
        }
    selected_pairs_validated = int(identity_summary.get(
        "selected_pairs",
        identity_failure_details.get("selected_pairs_validated", 0),
    ))
    raw_endpoints_validated = int(identity_summary.get(
        "saved_endpoint_cpu_replay_matches",
        identity_failure_details.get("raw_endpoints_validated", 0),
    ))
    report: dict[str, Any] = {
        "schema": "lightgen_saved_evidence_priority_diagnosis.v1",
        "task_id": TASK_ID,
        "status": "VALID_DESCRIPTIVE_EVIDENCE" if all_branches_valid else "PARTIAL_INVALID_HARNESS",
        "branches": branches,
        "input_paths": {
            "v12_evidence_dir": v12_dir.as_posix(),
            "identity_evidence_dir": identity_dir.as_posix(),
            "identity_analysis_dir": analysis_dir.as_posix(),
            "ed_checkpoint": checkpoint.as_posix(),
            "rs_stop_report": rs_stop_report_path.as_posix(),
        },
        "counts": {
            "camera_confirmation_rows": len(camera_rows),
            "power_rows": len(power_rows),
            "identity_selected_pairs": selected_pairs_validated,
            "identity_pair_results_completed": len(identity_rows),
            "identity_raw_endpoints_validated": raw_endpoints_validated,
            "identity_saved_endpoints": 460,
            "saved_endpoint_evaluator_forwards": int(identity_summary.get(
                "saved_endpoint_cpu_replay_attempts", identity_failure_details.get("saved_endpoint_cpu_replay_attempts", 0)
            )),
            "saved_endpoint_replays_within_tolerance": int(identity_summary.get(
                "saved_endpoint_cpu_replay_matches", identity_failure_details.get("saved_endpoint_cpu_replay_matches", 0)
            )),
            "energy_matched_e_d_forwards": int(identity_summary.get(
                "matched_b_evaluator_forwards", identity_failure_details.get("matched_b_evaluator_forwards", 0)
            )),
            "scientific_updates": 0,
            "optical_forwards": 0,
            "sampling_calls": 0,
            "human_ratings": 0,
            "official_test_reads": 0,
        },
        "camera": camera_summary,
        "power": power_summary,
        "identity": {"summary": identity_summary, "partial_failure_details": identity_failure_details},
        "engineering_stop_reference_only": rs_stop_context,
        "decision": decision,
        "scope_limit": "Fixed saved populations and proxy camera; descriptive diagnosis only, not hardware, readability, generalization, or model adoption.",
        "elapsed_seconds_observation": time.perf_counter() - started,
    }
    if args.preflight_only:
        ready = all_branches_valid
        return {
            "status": "PREFLIGHT_READY" if ready else "PREFLIGHT_BLOCKED",
            "task_id": TASK_ID,
            "output_dir": output_dir.as_posix(),
            "branches": branches,
            "camera_rows_checked": len(camera_rows),
            "power_rows_checked": len(power_rows),
            "identity_pairs_checked": selected_pairs_validated,
            "e_d_checkpoint_loaded": bool(identity_rows) or bool(identity_failure_details.get("e_d_checkpoint_loaded")),
            "scientific_updates": 0,
            "optical_forwards": 0,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    if camera_rows:
        _write_csv(output_dir / "camera_per_image.csv", camera_rows)
    if power_rows:
        _write_csv(output_dir / "nine_channel_power_per_image.csv", power_rows)
    if identity_rows:
        _write_csv(output_dir / "identity_energy_match_per_pair.csv", identity_rows)
    _write_json(output_dir / "report.json", report)
    if branches["camera_saturation"]["status"] == "VALID" and branches["nine_channel_power"]["status"] == "VALID":
        (output_dir / "report.html").write_text(
            _html_report(report, camera_context["gallery"], camera_rows, power_rows, identity_rows),
            encoding="utf-8",
        )
    else:
        (output_dir / "report.html").write_text(
            "<!doctype html><meta charset=\"utf-8\"><title>LightGen saved evidence diagnosis</title>"
            "<h1>LightGen保存证据诊断：部分完成</h1><pre>"
            + html.escape(json.dumps(branches, ensure_ascii=False, indent=2))
            + "</pre>",
            encoding="utf-8",
        )
    closeout = _closeout_markdown(report) if all_branches_valid else _partial_closeout_markdown(report)
    (output_dir / "closeout_report.md").write_text(closeout, encoding="utf-8")
    report["output_files"] = sorted(path.name for path in output_dir.iterdir())
    report["elapsed_seconds_observation"] = time.perf_counter() - started
    _write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    args = _arguments()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
