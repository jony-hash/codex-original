"""Diagnose one globally shared proxy-camera exposure from fixed v12 evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_ID = "LIGHTGEN_FIXED_READOUT_FEASIBILITY_V1"
SCHEMA = "lightgen_fixed_readout_feasibility.v1"
OUTPUT_RELATIVE = Path("outputs/20261019_lightgen_fixed_readout_feasibility_v1/runs/attempt_01")
CANDIDATE = "candidate_2x2_C"
MODE = "gaussian_s12"
STATES = (("step_000_initial", "initial"), ("step_000500_terminal", "terminal"))
CAMERA_REPLAY_RTOL = 2.0e-6
CAMERA_REPLAY_ATOL = 5.0e-7
ELECTRON_REPLAY_RTOL = 4.0e-6
ELECTRON_REPLAY_ATOL = 1.0e-3
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026101803
H_PLANCK = 6.62607015e-34
C_LIGHT = 299_792_458.0
GRID_INDEXES = np.arange(301, dtype=np.int64)
EXPOSURES_S = np.power(10.0, -5.0 + 0.01 * GRID_INDEXES)
EXPOSURES_S[200] = 1.0e-3


class EvidenceError(RuntimeError):
    """Fixed evidence is missing, inconsistent, or numerically invalid."""


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camera_replay(
    power_w_per_pixel: np.ndarray,
    wavelength_m: float,
    exposure_s: float,
    quantum_efficiency: float,
    full_well_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(power_w_per_pixel, dtype=np.float64)
    settings = (wavelength_m, exposure_s, quantum_efficiency, full_well_e)
    _require(all(math.isfinite(float(value)) for value in settings), "camera settings must be finite")
    _require(wavelength_m > 0.0 and exposure_s > 0.0 and full_well_e > 0.0
             and 0.0 <= quantum_efficiency <= 1.0, "camera settings are outside valid ranges")
    _require(bool(np.isfinite(values).all()) and bool(np.all(values >= 0.0)),
             "camera power must be finite and nonnegative")
    photon_energy = H_PLANCK * C_LIGHT / wavelength_m
    electrons = values * (quantum_efficiency * exposure_s / photon_energy)
    task_output = np.clip(electrons / full_well_e, 0.0, 1.0).astype(np.float32)
    return electrons, task_output


def _validate_population_vectors(
    rows: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    table: Sequence[Mapping[str, str]],
    state_dir: str,
) -> None:
    _require(rows.shape == labels.shape == roles.shape == (300,), f"population vector shape differs: {state_dir}")
    _require(np.issubdtype(rows.dtype, np.integer) and np.issubdtype(labels.dtype, np.integer)
             and np.issubdtype(roles.dtype, np.str_), f"population metadata dtype differs: {state_dir}")
    _require(len(np.unique(rows)) == 300, f"duplicate sample row: {state_dir}")
    table_rows = [int(record["row_id"]) for record in table]
    _require(len(table_rows) == 300 and len(set(table_rows)) == 300,
             f"measurement table has missing or duplicate rows: {state_dir}")
    _require(np.array_equal(rows, np.asarray(table_rows, dtype=np.int64)),
             f"array/table row identity differs: {state_dir}")
    _require(np.array_equal(labels, np.asarray([int(record["label"]) for record in table], dtype=np.int64)),
             f"array/table label identity differs: {state_dir}")
    _require(np.array_equal(roles, np.asarray([record["role"] for record in table], dtype=str)),
             f"array/table population role differs: {state_dir}")
    _require(set(int(value) for value in labels) == set(range(10)), f"labels do not cover 0-9: {state_dir}")
    _require(set(str(value) for value in roles) == {"fit", "confirmation"}, f"roles differ: {state_dir}")
    for role, per_class in (("fit", 10), ("confirmation", 20)):
        _require(int((roles == role).sum()) == per_class * 10, f"{role} population count differs: {state_dir}")
        for label in range(10):
            count = int(((roles == role) & (labels == label)).sum())
            _require(count == per_class, f"{role} class count differs: {state_dir}/label={label}")
    _require(all(record["candidate"] == CANDIDATE and record["mode"] == MODE
                 and record["state"] == state_dir for record in table),
             f"candidate/mode/state identity differs: {state_dir}")


def _validate_camera_values(
    target: np.ndarray,
    power: np.ndarray,
    saved_electrons: np.ndarray,
    saved_output: np.ndarray,
    state_name: str,
) -> None:
    _require(bool(np.isfinite(target).all() and np.isfinite(power).all()
                  and np.isfinite(saved_electrons).all() and np.isfinite(saved_output).all()),
             f"camera archive contains nonfinite values: {state_name}")
    _require(bool(np.all((target >= 0.0) & (target <= 1.0)) and np.all(power >= 0.0)),
             f"target/power values are outside valid ranges: {state_name}")


def _validate_endpoint(
    evidence_dir: Path,
    state_dir: str,
    state_name: str,
    settings: Mapping[str, float],
) -> dict[str, np.ndarray]:
    base = evidence_dir / "measurements" / state_dir / CANDIDATE / MODE
    array_path = base / "camera_arrays_4096_1ms.npz"
    table_path = base / "per_image_measurements.csv"
    table = _read_csv(table_path)
    _require(len(table) == 300, f"measurement row count differs: {state_name}")
    with np.load(array_path, allow_pickle=False) as archive:
        required = {"rows", "labels", "role", "target", "power_w_per_pixel",
                    "expected_electrons", "task_output"}
        _require(required.issubset(archive.files), f"camera archive fields are incomplete: {array_path}")
        rows = archive["rows"].copy()
        labels = archive["labels"].copy()
        roles = archive["role"].copy()
        target = archive["target"].copy()
        power = archive["power_w_per_pixel"].copy()
        saved_electrons = archive["expected_electrons"].copy()
        saved_output = archive["task_output"].copy()

    _validate_population_vectors(rows, labels, roles, table, state_dir)
    _require(target.shape == (300, 1, 200, 200), f"target geometry differs: {state_name}")
    _require(power.shape == saved_electrons.shape == saved_output.shape == (300, 200, 200),
             f"camera array geometry differs: {state_name}")
    _validate_camera_values(target, power, saved_electrons, saved_output, state_name)

    electrons, output = camera_replay(
        power, settings["wavelength_m"], settings["primary_exposure_s"],
        settings["quantum_efficiency"], settings["full_well_e"],
    )
    _require(np.allclose(electrons, saved_electrons, rtol=ELECTRON_REPLAY_RTOL,
                         atol=ELECTRON_REPLAY_ATOL), f"W/pixel-to-electron replay differs: {state_name}")
    _require(np.allclose(output, saved_output, rtol=CAMERA_REPLAY_RTOL,
                         atol=CAMERA_REPLAY_ATOL), f"1 ms camera-view replay differs: {state_name}")
    table_by_row = {int(record["row_id"]): record for record in table}
    for index, row_id in enumerate(rows.tolist()):
        record = table_by_row[int(row_id)]
        camera_power_w = float(power[index].sum(dtype=np.float64))
        total_electrons = float(electrons[index].sum(dtype=np.float64))
        mse = float(np.square(output[index].astype(np.float64) - target[index, 0].astype(np.float64)).mean())
        _require(math.isclose(camera_power_w, float(record["camera_pixel_integrated_power_w"]),
                              rel_tol=2.0e-5, abs_tol=1.0e-15),
                 f"W/pixel sum does not match integrated power: {state_name}/row={row_id}")
        _require(math.isclose(total_electrons, float(record["camera_expected_electrons"]),
                              rel_tol=2.0e-6, abs_tol=1.0),
                 f"electron sum does not match measurement table: {state_name}/row={row_id}")
        _require(math.isclose(mse, float(record["primary_task_mse"]),
                              rel_tol=CAMERA_REPLAY_RTOL, abs_tol=CAMERA_REPLAY_ATOL),
                 f"1 ms per-image MSE replay differs: {state_name}/row={row_id}")
    return {"rows": rows, "labels": labels, "roles": roles, "target": target[:, 0], "power": power}


def load_and_preflight(evidence_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    report = _read_json(evidence_dir / "report.json")
    _require(report.get("task_id") == "OLS_DECODER_MNIST_RECONSTRUCTION_V12"
             and report.get("status") == "VALID_FAILED", "fixed v12 evidence identity/status differs")
    config = report.get("fixed_config")
    _require(isinstance(config, dict), "v12 fixed configuration is missing")
    _require(config.get("training", {}).get("primary_mode") == MODE
             and CANDIDATE in config.get("training", {}).get("candidates", []),
             "fixed Gaussian S12 candidate is absent from v12 report")
    _require(config.get("data", {}).get("fit_per_label") == 10
             and config.get("data", {}).get("confirmation_per_label") == 20,
             "fixed fit/confirmation allocation differs")
    _require(config.get("measurement", {}).get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
             and config.get("measurement", {}).get("bootstrap_seed") == BOOTSTRAP_SEED,
             "fixed bootstrap settings differ from v12 report")
    optics, camera = config.get("optics", {}), config.get("camera", {})
    settings = {
        "wavelength_m": float(optics["wavelength_m"]),
        "source_power_w": float(optics["source_power_w"]),
        "primary_exposure_s": float(camera["primary_exposure_s"]),
        "quantum_efficiency": float(camera["quantum_efficiency"]),
        "full_well_e": float(camera["full_well_e"]),
    }
    _require(settings["primary_exposure_s"] == 1.0e-3 and settings["quantum_efficiency"] == 0.5
             and settings["full_well_e"] == 30000.0 and settings["wavelength_m"] == 532.0e-9,
             "v12 proxy camera settings differ from the fixed diagnostic scope")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state_dir, state_name in STATES:
        loaded[state_name] = _validate_endpoint(evidence_dir, state_dir, state_name, settings)
    initial, terminal = loaded["initial"], loaded["terminal"]
    for field in ("rows", "labels", "roles", "target"):
        _require(np.array_equal(initial[field], terminal[field]),
                 f"initial/terminal sample {field} differ")
    fit_count = int((terminal["roles"] == "fit").sum())
    confirm_count = int((terminal["roles"] == "confirmation").sum())
    preflight = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "READY",
        "evidence_dir": str(evidence_dir),
        "candidate": CANDIDATE,
        "mode": MODE,
        "states": [state_name for _, state_name in STATES],
        "fit_images": fit_count,
        "confirmation_images": confirm_count,
        "camera_replay": "PASSED",
        "row_role_label_alignment": "PASSED",
        "power_unit_sum_check": "PASSED: sum(W/pixel) matches integrated W per image",
        "forbidden_operations": {
            "optical_forward_calls": 0,
            "evaluator_forward_calls": 0,
            "sampling_calls": 0,
            "gradient_calls": 0,
            "optimizer_updates": 0,
        },
    }
    return settings, loaded, preflight


def global_mean_image(targets: np.ndarray, roles: np.ndarray) -> np.ndarray:
    fit = np.asarray(roles) == "fit"
    _require(int(fit.sum()) == 100, "global mean baseline requires exactly 100 fit targets")
    values = np.asarray(targets, dtype=np.float64)
    _require(bool(np.isfinite(values[fit]).all()), "fit targets contain nonfinite values")
    return values[fit].mean(axis=0, dtype=np.float64)


def select_fit_exposure(
    fit_power_w_per_pixel: np.ndarray,
    fit_targets: np.ndarray,
    fit_labels: np.ndarray,
    wavelength_m: float,
    quantum_efficiency: float,
    full_well_e: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    power = np.asarray(fit_power_w_per_pixel, dtype=np.float64)
    targets = np.asarray(fit_targets, dtype=np.float64)
    labels = np.asarray(fit_labels, dtype=np.int64)
    _require(power.ndim == 3 and power.shape[0] == 100 and targets.shape == power.shape
             and labels.shape == (100,),
             "exposure selection requires the fixed 100-image fit population")
    _require(set(labels.tolist()) == set(range(10)) and all(int((labels == label).sum()) == 10 for label in range(10)),
             "exposure fit population is not balanced 10 per class")
    photon_energy = H_PLANCK * C_LIGHT / wavelength_m
    electron_rate = power * (quantum_efficiency / photon_energy)
    curve: list[dict[str, Any]] = []
    for grid_index, exposure_s in enumerate(EXPOSURES_S):
        electrons = electron_rate * float(exposure_s)
        prediction = np.clip(electrons / full_well_e, 0.0, 1.0).astype(np.float32)
        image_mse = np.square(prediction.astype(np.float64) - targets).mean(axis=(1, 2))
        class_mse = {str(label): float(image_mse[labels == label].mean()) for label in range(10)}
        curve.append({
            "grid_index": int(grid_index),
            "exposure_s": float(exposure_s),
            "exposure_ms": float(exposure_s * 1000.0),
            "fit_class_balanced_mse": float(np.mean(list(class_mse.values()))),
            "fit_mean_mse": float(image_mse.mean()),
            **{f"fit_class_{label}_mse": class_mse[str(label)] for label in range(10)},
        })
    selected = min(curve, key=lambda row: (row["fit_class_balanced_mse"], row["exposure_s"]))
    lock = {
        "schema": SCHEMA,
        "selected_exposure_s": selected["exposure_s"],
        "selected_exposure_ms": selected["exposure_ms"],
        "grid_index": selected["grid_index"],
        "grid_size": len(EXPOSURES_S),
        "selection_metric": "terminal fit-set class-balanced MSE against fixed 100 targets",
        "tie_break": "shorter exposure",
        "fit_class_balanced_mse": selected["fit_class_balanced_mse"],
        "original_exposure_s": 1.0e-3,
        "confirmation_used_for_selection": False,
    }
    return curve, lock


def read_exposure_lock(path: Path) -> dict[str, Any]:
    lock = _read_json(path)
    _require(lock.get("schema") == SCHEMA and lock.get("confirmation_used_for_selection") is False,
             "exposure lock schema or selection provenance is invalid")
    index = int(lock["grid_index"])
    _require(0 <= index < len(EXPOSURES_S) and len(EXPOSURES_S) == 301,
             "locked exposure is outside the predefined grid")
    exposure_s = float(lock["selected_exposure_s"])
    _require(exposure_s == float(EXPOSURES_S[index]), "locked exposure does not match its fit-grid row")
    return lock


def _camera_metrics(
    target: np.ndarray,
    power: np.ndarray,
    exposure_s: float,
    settings: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    electrons, prediction = camera_replay(
        power, settings["wavelength_m"], exposure_s,
        settings["quantum_efficiency"], settings["full_well_e"],
    )
    error = prediction.astype(np.float64) - np.asarray(target, dtype=np.float64)
    return electrons, prediction, error


def _metric_record(
    row_id: int,
    label: int,
    state_name: str,
    condition: str,
    exposure_s: float | None,
    target: np.ndarray,
    prediction: np.ndarray,
    error: np.ndarray,
    electrons: np.ndarray | None,
    full_well_e: float,
) -> dict[str, Any]:
    squared = np.square(error)
    total_sse = float(squared.sum(dtype=np.float64))
    record: dict[str, Any] = {
        "row_id": int(row_id),
        "label": int(label),
        "state": state_name,
        "condition": condition,
        "exposure_s": exposure_s,
        "mse": float(squared.mean(dtype=np.float64)),
        "saturation_area_fraction": None,
        "saturated_error_share": None,
        "unsaturated_error_share": None,
        "saturated_mse_per_pixel": None,
        "unsaturated_mse_per_pixel": None,
        "mean_electrons_per_pixel": None,
        "electron_p50": None,
        "electron_p95": None,
        "electron_p99": None,
        "electron_max": None,
    }
    if electrons is not None:
        saturated = electrons >= full_well_e
        sat_count = int(saturated.sum())
        unsat_count = int(saturated.size - sat_count)
        sat_sse = float(squared[saturated].sum(dtype=np.float64))
        unsat_sse = float(squared[~saturated].sum(dtype=np.float64))
        record.update({
            "saturation_area_fraction": float(saturated.mean()),
            "saturated_error_share": sat_sse / total_sse if total_sse else 0.0,
            "unsaturated_error_share": unsat_sse / total_sse if total_sse else 0.0,
            "saturated_mse_per_pixel": sat_sse / sat_count if sat_count else 0.0,
            "unsaturated_mse_per_pixel": unsat_sse / unsat_count if unsat_count else 0.0,
            "mean_electrons_per_pixel": float(electrons.mean(dtype=np.float64)),
            "electron_p50": float(np.percentile(electrons, 50)),
            "electron_p95": float(np.percentile(electrons, 95)),
            "electron_p99": float(np.percentile(electrons, 99)),
            "electron_max": float(electrons.max()),
        })
    _require(bool(np.isfinite(np.asarray(prediction)).all() and np.isfinite(error).all()),
             f"nonfinite camera result: {state_name}/{condition}/row={row_id}")
    _require(np.asarray(target).shape == np.asarray(prediction).shape,
             f"prediction/target shape differs: {state_name}/{condition}/row={row_id}")
    return record


def _group_summary(
    metrics: Sequence[Mapping[str, Any]],
    endpoint_data: Mapping[str, np.ndarray],
    settings: Mapping[str, float],
    exposure_s: float,
    condition: str,
    state_name: str,
) -> list[dict[str, Any]]:
    rows = endpoint_data["rows"]
    labels = endpoint_data["labels"]
    roles = endpoint_data["roles"]
    confirm = np.flatnonzero(roles == "confirmation")
    condition_rows = [row for row in metrics if row["state"] == state_name and row["condition"] == condition]
    _require(len(condition_rows) == 200, f"confirmation metrics incomplete: {state_name}/{condition}")
    groups: list[tuple[str, np.ndarray]] = [("all", confirm)]
    groups.extend((str(label), confirm[labels[confirm] == label]) for label in range(10))
    output: list[dict[str, Any]] = []
    for group_name, indices in groups:
        row_ids = {int(rows[index]) for index in indices.tolist()}
        subset = [row for row in condition_rows if int(row["row_id"]) in row_ids]
        _require(len(subset) == len(indices), f"summary rows do not align: {state_name}/{condition}/{group_name}")
        mse = float(np.mean([float(row["mse"]) for row in subset]))
        summary: dict[str, Any] = {
            "state": state_name,
            "condition": condition,
            "group": group_name,
            "image_count": len(subset),
            "mse": mse,
            "saturation_area_fraction": None,
            "saturated_error_share": None,
            "unsaturated_error_share": None,
            "electron_mean_per_pixel": None,
            "electron_p50": None,
            "electron_p95": None,
            "electron_p99": None,
            "electron_max": None,
        }
        if condition != "global_mean_baseline":
            power = endpoint_data["power"][indices]
            target = endpoint_data["target"][indices]
            electrons, prediction, error = _camera_metrics(
                target, power, exposure_s, settings,
            )
            saturated = electrons >= settings["full_well_e"]
            squared = np.square(error)
            total_sse = float(squared.sum(dtype=np.float64))
            sat_sse = float(squared[saturated].sum(dtype=np.float64))
            flat_e = electrons.reshape(-1)
            summary.update({
                "saturation_area_fraction": float(saturated.mean()),
                "saturated_error_share": sat_sse / total_sse if total_sse else 0.0,
                "unsaturated_error_share": 1.0 - (sat_sse / total_sse) if total_sse else 0.0,
                "electron_mean_per_pixel": float(flat_e.mean(dtype=np.float64)),
                "electron_p50": float(np.percentile(flat_e, 50)),
                "electron_p95": float(np.percentile(flat_e, 95)),
                "electron_p99": float(np.percentile(flat_e, 99)),
                "electron_max": float(flat_e.max()),
            })
        output.append(summary)
    return output


def _stratified_bootstrap(
    differences: Mapping[int, float],
    labels_by_row: Mapping[int, int],
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_class = {
        label: np.asarray([float(value) for row_id, value in differences.items()
                           if labels_by_row[int(row_id)] == label], dtype=np.float64)
        for label in range(10)
    }
    _require(all(values.shape == (20,) for values in per_class.values()),
             "stratified bootstrap requires 20 paired confirmations in each class")
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        class_means = []
        for label in range(10):
            values = per_class[label]
            class_means.append(float(values[rng.integers(0, len(values), size=20)].mean()))
        draws[replicate] = float(np.mean(class_means))
    return {
        "mean_gain": float(np.mean([values.mean() for values in per_class.values()])),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "stratification": "10 labels; sample 20 paired rows with replacement within each label",
    }


def validate_saved_measurements(output_dir: Path) -> dict[str, str]:
    """Independently reconcile the persisted exposure lock and diagnostic tables."""
    lock = read_exposure_lock(output_dir / "locked_exposure.json")
    curve = _read_csv(output_dir / "fit_exposure_curve.csv")
    _require(len(curve) == 301, "saved exposure curve is incomplete")
    selected_curve_row = curve[int(lock["grid_index"])]
    _require(float(selected_curve_row["exposure_s"]) == float(lock["selected_exposure_s"]),
             "saved exposure lock does not point to the selected fit-curve row")
    metrics = _read_csv(output_dir / "per_image_metrics.csv")
    _require(len(metrics) == 1200, "saved per-image confirmation table is incomplete")
    keys = [(row["state"], int(row["row_id"]), row["condition"]) for row in metrics]
    _require(len(keys) == len(set(keys)), "saved per-image table has duplicate endpoint/row/condition keys")
    for state_name in ("initial", "terminal"):
        for condition in ("locked_exposure", "original_1ms", "global_mean_baseline"):
            subset = [row for row in metrics if row["state"] == state_name and row["condition"] == condition]
            _require(len(subset) == 200, f"saved confirmation metrics are missing: {state_name}/{condition}")
            _require(all(math.isfinite(float(row["mse"])) and float(row["mse"]) >= 0.0 for row in subset),
                     f"saved MSE values are invalid: {state_name}/{condition}")
    summaries = _read_csv(output_dir / "summary_by_group.csv")
    _require(len(summaries) == 66, "saved class-summary table is incomplete")
    summary_map = {(row["state"], row["condition"], row["group"]): float(row["mse"])
                   for row in summaries}
    _require(len(summary_map) == 66, "saved class-summary keys are duplicated")
    for key, expected in summary_map.items():
        subset = [float(row["mse"]) for row in metrics
                  if row["state"] == key[0] and row["condition"] == key[1]
                  and (key[2] == "all" or row["label"] == key[2])]
        _require(len(subset) == (200 if key[2] == "all" else 20),
                 f"saved class summary does not have the expected population: {key}")
        _require(math.isclose(float(np.mean(subset)), expected, rel_tol=1.0e-12, abs_tol=1.0e-12),
                 f"saved group MSE does not reconcile to per-image values: {key}")
    return {
        "saved_exposure_lock_and_curve": "PASSED",
        "saved_per_image_population": "PASSED",
        "saved_group_MSE_reconciliation": "PASSED",
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_gray_png(path: Path, image: np.ndarray) -> None:
    values = np.rint(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    _require(values.ndim == 2 and values.shape == (200, 200), "HTML gallery expects 200x200 grayscale images")
    scanlines = b"".join(b"\x00" + values[row].tobytes() for row in range(values.shape[0]))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 200, 200, 8, 0, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _build_html(
    output_dir: Path,
    endpoint_data: Mapping[str, np.ndarray],
    mean_image: np.ndarray,
    locked_exposure_s: float,
    settings: Mapping[str, float],
) -> None:
    image_dir = output_dir / "gallery"
    image_dir.mkdir(exist_ok=True)
    fit_mean = np.asarray(mean_image, dtype=np.float32)
    labels = endpoint_data["labels"]
    roles = endpoint_data["roles"]
    rows = endpoint_data["rows"]
    target = endpoint_data["target"]
    power = endpoint_data["power"]
    cards: list[str] = []
    for label in range(10):
        index = int(np.flatnonzero((roles == "confirmation") & (labels == label))[0])
        _, one_ms = camera_replay(power[index], settings["wavelength_m"], 1.0e-3,
                                  settings["quantum_efficiency"], settings["full_well_e"])
        _, selected = camera_replay(power[index], settings["wavelength_m"], locked_exposure_s,
                                    settings["quantum_efficiency"], settings["full_well_e"])
        variants = (("Target", target[index]), ("1 ms", one_ms),
                    ("Locked exposure", selected), ("Global fit mean", fit_mean))
        cells: list[str] = []
        for title, image in variants:
            name = f"class_{label}_{title.lower().replace(' ', '_')}.png"
            _write_gray_png(image_dir / name, image)
            cells.append(f'<figure><img src="gallery/{html.escape(name)}" alt=""><figcaption>{html.escape(title)}</figcaption></figure>')
        cards.append(
            f'<section><h2>类别 {label} · confirmation row {int(rows[index])}</h2>'
            f'<div class="images">{"".join(cells)}</div></section>'
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LightGen 固定全局曝光可行性诊断</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;background:#f6f7f9;color:#202124}
header,section{background:white;border:1px solid #d8dce2;border-radius:8px;padding:14px;margin:12px 0}
.images{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px}
figure{margin:0;text-align:center}img{width:100%;max-width:240px;image-rendering:pixelated;background:#eee}
figcaption{padding:5px;color:#444}@media(max-width:720px){.images{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><h1>LightGen 固定全局曝光可行性诊断</h1>
<p>灰度均统一为 [0,1]，无逐图增强；锁定曝光仅由终点100张拟合集选择。对照为同一100张拟合目标生成的一张全局均值图。</p>
<p>锁定曝光：__EXPOSURE__ ms。图像用于定位；不代表真实相机或硬件资格。</p></header>
__CARDS__</body></html>"""
    document = document.replace("__EXPOSURE__", f"{locked_exposure_s * 1000.0:.8g}")
    document = document.replace("__CARDS__", "\n".join(cards))
    (output_dir / "comparison.html").write_text(document, encoding="utf-8")


def _aggregate_confirmation(
    loaded: Mapping[str, Mapping[str, np.ndarray]],
    settings: Mapping[str, float],
    mean_image: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lock_path = output_dir / "locked_exposure.json"
    lock = read_exposure_lock(lock_path)
    locked_exposure_s = float(lock["selected_exposure_s"])
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    endpoint_mses: dict[str, dict[str, dict[int, float]]] = {}
    labels_by_row: dict[int, int] = {}

    for state_name in ("initial", "terminal"):
        endpoint_data = loaded[state_name]
        confirm_indices = np.flatnonzero(endpoint_data["roles"] == "confirmation")
        endpoint_mses[state_name] = {"locked_exposure": {}, "original_1ms": {}, "global_mean_baseline": {}}
        for index in confirm_indices.tolist():
            row_id = int(endpoint_data["rows"][index])
            label = int(endpoint_data["labels"][index])
            target = endpoint_data["target"][index]
            power = endpoint_data["power"][index]
            labels_by_row[row_id] = label
            for condition, exposure_s in (("locked_exposure", locked_exposure_s),
                                          ("original_1ms", 1.0e-3)):
                electrons, prediction, error = _camera_metrics(target, power, exposure_s, settings)
                record = _metric_record(row_id, label, state_name, condition, exposure_s,
                                        target, prediction, error, electrons, settings["full_well_e"])
                metric_rows.append(record)
                endpoint_mses[state_name][condition][row_id] = float(record["mse"])
            mean_error = mean_image - target.astype(np.float64)
            baseline_record = _metric_record(
                row_id, label, state_name, "global_mean_baseline", None, target,
                mean_image, mean_error, None, settings["full_well_e"],
            )
            metric_rows.append(baseline_record)
            endpoint_mses[state_name]["global_mean_baseline"][row_id] = float(baseline_record["mse"])

        for condition, exposure in (("locked_exposure", locked_exposure_s),
                                    ("original_1ms", 1.0e-3),
                                    ("global_mean_baseline", 1.0e-3)):
            summary_rows.extend(_group_summary(metric_rows, endpoint_data, settings,
                                                exposure, condition, state_name))

    terminal = endpoint_mses["terminal"]
    gains = {
        "locked_vs_global_mean_baseline": _stratified_bootstrap(
            {row_id: terminal["global_mean_baseline"][row_id] - mse
             for row_id, mse in terminal["locked_exposure"].items()}, labels_by_row),
        "locked_vs_original_1ms": _stratified_bootstrap(
            {row_id: terminal["original_1ms"][row_id] - mse
             for row_id, mse in terminal["locked_exposure"].items()}, labels_by_row),
    }
    baseline_gain = gains["locked_vs_global_mean_baseline"]
    one_ms_gain = gains["locked_vs_original_1ms"]
    if baseline_gain["ci95_lower"] > 0.0:
        decision = "SUPPORTS_PROXY_READOUT_CROSSING_GLOBAL_MEAN_BASELINE"
    elif one_ms_gain["ci95_lower"] > 0.0:
        decision = "READOUT_IMPROVES_VS_1MS_BUT_NOT_GLOBAL_MEAN_BASELINE"
    else:
        decision = "NO_CONFIRMATORY_SUPPORT_FOR_EXPOSURE_IMPROVEMENT"
    comparison = {
        "decision": decision,
        "bootstrap_gains": gains,
        "terminal_mse": {
            condition: float(np.mean([
                np.mean([value for row_id, value in values.items() if labels_by_row[row_id] == label])
                for label in range(10)
            ]))
            for condition, values in terminal.items()
        },
        "initial_mse": {
            condition: float(np.mean([
                np.mean([value for row_id, value in values.items() if labels_by_row[row_id] == label])
                for label in range(10)
            ]))
            for condition, values in endpoint_mses["initial"].items()
        },
        "locked_exposure_consumed_from_saved_file": str(lock_path),
    }
    _build_html(output_dir, loaded["terminal"], mean_image, locked_exposure_s, settings)
    return metric_rows, summary_rows, comparison


def _check_output_dir(output_dir: Path, expected: Path) -> None:
    _require(output_dir.resolve() == expected.resolve(),
             f"output must be the authorized new run path: {expected}")
    if output_dir.exists():
        allowed = {"preflight_receipt.json"}
        _require(all(path.name in allowed and path.is_file() for path in output_dir.iterdir()),
                 f"refusing to overwrite non-preflight output content: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _closeout_markdown(report: Mapping[str, Any]) -> str:
    fit = report["fit_selection"]
    comparison = report["confirmation_comparison"]
    terminal = comparison["terminal_mse"]
    gain = comparison["bootstrap_gains"]
    decision = comparison["decision"]
    if decision == "SUPPORTS_PROXY_READOUT_CROSSING_GLOBAL_MEAN_BASELINE":
        scientific = (
            f"在预设无噪声代理相机和10 μs–10 ms网格内，拟合集选出的单一全局曝光为"
            f"{fit['selected_exposure_ms']:.6g} ms。终点确认MSE为"
            f"{terminal['locked_exposure']:.6g}，低于同一100张拟合目标产生的一张全局均值图"
            f"（{terminal['global_mean_baseline']:.6g}）；配对分层bootstrap收益95%区间下界为"
            f"{gain['locked_vs_global_mean_baseline']['ci95_lower']:.6g}。这支持所测代理读出设置"
            "足以跨过该静态MSE基准；不支持真实相机、输入特异重建、生成可读性或模型采用。"
        )
    elif decision == "READOUT_IMPROVES_VS_1MS_BUT_NOT_GLOBAL_MEAN_BASELINE":
        scientific = (
            f"锁定全局曝光为{fit['selected_exposure_ms']:.6g} ms。终点确认结果相对原1 ms有配对改善"
            f"（收益95%区间下界{gain['locked_vs_original_1ms']['ci95_lower']:.6g}），但没有可靠优于"
            f"全局均值图基线（MSE {terminal['global_mean_baseline']:.6g}）；支持读出限制有贡献，"
            "同时保留空间分配、表征和优化缺口。"
        )
    else:
        scientific = (
            f"锁定全局曝光为{fit['selected_exposure_ms']:.6g} ms，但终点确认结果未显示相对原1 ms或"
            "同一全局均值图的明确配对收益。当前预设网格内不支持该曝光选择改善确认结果；"
            "这不排除网格外、噪声/增益模型或其他相机条件带来的差异。"
        )
    return f"""# LightGen 固定全局曝光可行性诊断

结果与状态：完整有效测量，状态 `VALID_COMPLETED`；科学判读为 `{decision}`。仅使用保存的2×2-C Gaussian S12初始/500步终点相机功率，曝光拟合100张、确认200张。

科学认识：{scientific}本报告的全局均值图严格定义为100张拟合目标的逐像素平均，只有一张，对所有类别共用；按类别列出的MSE是确认图相对这同一张图的误差，不是各类别各自均值图。读出采用无噪声代理公式，无真实相机噪声、暗电流或硬件测量。

目标缺口：仍缺真实相机参数与噪声验证、物理光路读出、输入特异重建和人眼/生成可读性证据。本次没有验证重新训练时改变曝光的效果。

工程完成与验证：1 ms W/pixel→电子数→满阱裁剪→MSE重放、样本/角色/标签对齐、301点曝光网格、拟合集唯一选择及锁定文件回读均已完成。初始与终点在同一锁定曝光及原1 ms下评估；配对区间使用2000次等类分层bootstrap，seed `{BOOTSTRAP_SEED}`。光学前向、E_D前向、采样、梯度和优化器更新均为0。实际耗时 {report['duration_seconds']:.2f} 秒。

决策影响与下一步：当前判读为 `{decision}`。建议下一步：{report['recommended_next_action']}。该建议不构成训练、硬件采用或后续执行授权；本轮到此停止。
"""


def run_diagnostic(
    evidence_dir: Path,
    output_dir: Path,
    settings: Mapping[str, float],
    loaded: Mapping[str, Mapping[str, np.ndarray]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    terminal = loaded["terminal"]
    fit_mask = terminal["roles"] == "fit"
    mean_image = global_mean_image(terminal["target"], terminal["roles"])
    curve, fit_lock = select_fit_exposure(
        terminal["power"][fit_mask], terminal["target"][fit_mask], terminal["labels"][fit_mask],
        settings["wavelength_m"], settings["quantum_efficiency"], settings["full_well_e"],
    )
    _write_csv(output_dir / "fit_exposure_curve.csv", curve)
    # The lock is persisted before any confirmation metric is constructed or read.
    _write_json(output_dir / "locked_exposure.json", fit_lock)
    metric_rows, summary_rows, comparison = _aggregate_confirmation(
        loaded, settings, mean_image, output_dir,
    )
    _write_csv(output_dir / "per_image_metrics.csv", metric_rows)
    _write_csv(output_dir / "summary_by_group.csv", summary_rows)
    saved_table_checks = validate_saved_measurements(output_dir)
    _write_json(output_dir / "confirmation_comparison.json", comparison)
    decision = comparison["decision"]
    if decision == "SUPPORTS_PROXY_READOUT_CROSSING_GLOBAL_MEAN_BASELINE":
        next_action = (
            "只规划一次固定曝光下的代理相机噪声/读出扰动稳健性诊断，沿用同一确认人口与单一全局曝光；"
            "在完成前不选择真实相机、不训练。"
        )
    elif decision == "READOUT_IMPROVES_VS_1MS_BUT_NOT_GLOBAL_MEAN_BASELINE":
        next_action = (
            "优先完成大架构快速算子的实际成本与数值资格，继续保留空间分配/表征/优化缺口；"
            "暂不安排效果短训。"
        )
    else:
        next_action = (
            "优先完成大架构快速算子的实际成本与数值资格，当前曝光网格不提供增加短训或转向硬件的依据。"
        )
    duration = time.perf_counter() - started
    report = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "scientific_decision": decision,
        "evidence_dir": str(evidence_dir),
        "output_dir": str(output_dir),
        "scope": {
            "candidate": CANDIDATE,
            "mode": MODE,
            "states": ["initial", "terminal"],
            "fit_images": 100,
            "confirmation_images": 200,
            "baseline": "one image: pixelwise mean of all 100 fit targets; shared across every category",
            "proxy_readout": "noiseless expected-electron conversion followed by full-well clipping",
            "optical_forward_calls": 0,
            "evaluator_forward_calls": 0,
            "sampling_calls": 0,
            "gradient_calls": 0,
            "optimizer_updates": 0,
        },
        "camera_settings": settings,
        "fit_selection": read_exposure_lock(output_dir / "locked_exposure.json"),
        "confirmation_comparison": comparison,
        "preflight": preflight,
        "validation": {
            "saved_1ms_camera_replay": "PASSED",
            "power_unit_and_table_reconciliation": "PASSED",
            "population_alignment": "PASSED",
            "exposure_lock_written_before_confirmation": "PASSED",
            "selection_is_fit_only": "PASSED",
            "paired_bootstrap": "PASSED",
            **saved_table_checks,
        },
        "duration_seconds": duration,
        "recommended_next_action": next_action,
        "limitations": [
            "noiseless proxy readout only; not a real-camera selection or hardware-feasibility claim",
            "no inference about retraining under the selected exposure",
            "exposure conclusion is limited to the predefined 301-point grid",
            "confirmation population is used once after the fit exposure lock",
        ],
    }
    _write_json(output_dir / "report.json", report)
    validation_receipt = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASSED",
        "preflight": preflight,
        "checks": report["validation"],
        "scientific_decision_consumer": "report.json reads locked_exposure.json from disk before confirmation aggregation",
        "scientific_decision": decision,
        "duration_seconds": duration,
    }
    _write_json(output_dir / "validation_receipt.json", validation_receipt)
    (output_dir / "closeout_report.md").write_text(_closeout_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, help="explicit saved v12 evidence directory")
    parser.add_argument("--output-dir", required=True, help="explicit new attempt_01 output directory")
    parser.add_argument("--preflight-only", action="store_true", help="validate inputs and write a receipt; do not scan exposure")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    evidence_dir = Path(args.evidence_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    expected_output = root / OUTPUT_RELATIVE
    try:
        _check_output_dir(output_dir, expected_output)
        settings, _loaded, preflight = load_and_preflight(evidence_dir)
        receipt = {
            **preflight,
            "status": "READY",
            "camera_settings": settings,
            "duration_seconds": None,
            "preflight_only": bool(args.preflight_only),
        }
        _write_json(output_dir / "preflight_receipt.json", receipt)
        if args.preflight_only:
            print(json.dumps(receipt, ensure_ascii=False))
            return 0
        report = run_diagnostic(evidence_dir, output_dir, settings, _loaded, preflight)
        print(json.dumps({"status": report["status"], "decision": report["scientific_decision"],
                          "output_dir": str(output_dir), "duration_seconds": report["duration_seconds"]},
                         ensure_ascii=False))
        return 0
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "INVALID_HARNESS",
            "error_type": type(error).__name__,
            "error": str(error),
            "evidence_dir": str(evidence_dir),
        }
        if output_dir.resolve() == expected_output.resolve():
            output_dir.mkdir(parents=True, exist_ok=True)
            if all(path.name in {"preflight_receipt.json"} for path in output_dir.iterdir()):
                _write_json(output_dir / "preflight_receipt.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
