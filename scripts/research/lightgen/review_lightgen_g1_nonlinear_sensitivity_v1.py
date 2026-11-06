"""Independently reconsume saved G1 nonlinear-sensitivity evidence on Windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
TOL = 1.0e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected.ravel())), 1.0e-300)
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def shape_error(actual: np.ndarray, reference: np.ndarray) -> float:
    a = actual.astype(np.float64, copy=False)
    b = reference.astype(np.float64, copy=False)
    a = a / max(float(a.sum()), 1.0e-300)
    b = b / max(float(b.sum()), 1.0e-300)
    return relative_error(a, b)


def phase_error(observed: float, expected: float) -> float:
    delta = observed - expected
    return math.atan2(math.sin(delta), math.cos(delta))


def adaptive_average_28(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    output = np.empty((28, 28), dtype=np.float64)
    for oy in range(28):
        y0 = math.floor(oy * height / 28)
        y1 = math.ceil((oy + 1) * height / 28)
        for ox in range(28):
            x0 = math.floor(ox * width / 28)
            x1 = math.ceil((ox + 1) * width / 28)
            output[oy, ox] = float(image[y0:y1, x0:x1].mean())
    return output.astype(np.float32)


def load_metric_rows(path: Path) -> dict[tuple[int, str, int, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2640, f"expected 2640 production rows, got {len(rows)}")
    result: dict[tuple[int, str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["grid"]), row["arm"], int(row["population_position"]), row["condition_id"])
        require(key not in result, f"duplicate production metric row: {key}")
        result[key] = row
    return result


def check_close(actual: float, expected: float, name: str, tolerance: float = TOL) -> float:
    absolute = abs(actual - expected)
    allowed = tolerance * max(abs(expected), 1.0e-6)
    require(math.isfinite(absolute) and absolute <= allowed,
            f"{name} absolute difference {absolute:g} exceeds {allowed:g}")
    return absolute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path, required=True,
                        help="Local autodl_backup_attempt_01 evidence directory")
    args = parser.parse_args()
    config_path = args.config.resolve()
    measurement_dir = args.measurement_dir.resolve()
    require(config_path == (ROOT / "configs/research/lightgen/g1_nonlinear_sensitivity_v1.yaml").resolve(),
            "review must use the fixed sensitivity configuration")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight = read_json(measurement_dir / "preflight.json")
    measurement = read_json(measurement_dir / "measurement.json")
    metadata = read_json(measurement_dir / "field_metadata.json")
    require(preflight.get("status") == "PREFLIGHT_PASSED"
            and measurement.get("status") == "COMPLETE_DIAGNOSTIC_ONLY"
            and measurement.get("optimizer_updates") == 0
            and measurement.get("scientific_updates") == 0,
            "preflight or measurement state is not admissible")
    require(all(item.get("passed") is True for item in preflight.get("checks", []))
            and len(preflight.get("checks", [])) == 6,
            "preflight receipt contains missing or failed checks")
    require(metadata.get("schema") == "lightgen_g1_nonlinear_sensitivity_field_metadata.v1"
            and metadata.get("complex_dtype") == "complex128",
            "saved coordinate metadata identity is invalid")
    conditions = measurement["conditions"]
    require(len(conditions) == 12 and len({x["condition_id"] for x in conditions}) == 12,
            "measurement does not contain the twelve declared conditions")
    identity = measurement["identity"]
    q0 = complex(float(identity["q0_real"]), float(identity["q0_imag"]))
    i_ref = float(identity["i_ref_w_m2"])
    require(math.isfinite(i_ref) and i_ref > 0.0, "saved I_ref is invalid")
    metric_rows = load_metric_rows(measurement_dir / "sample_metrics.csv")
    config_base = yaml.safe_load((ROOT / config["base_config"]).read_text(encoding="utf-8"))
    source_fields_path = (ROOT / config["source"]["attempt04_fields"]).resolve()
    source_metadata_path = (ROOT / config["source"]["attempt04_metadata"]).resolve()
    source_fields = np.load(source_fields_path, allow_pickle=False)
    source_metadata = read_json(source_metadata_path)
    require(metadata.get("bare_field_storage") == source_fields_path.relative_to(ROOT).as_posix(),
            "saved output metadata points to a different bare-field source")
    source_te = read_json((ROOT / config_base["source"]["te_measurement"]).resolve())
    bare_transmission = float(source_te["bare_reference"]["T"])
    bare_t = complex(float(source_te["bare_reference"]["tE_z50"]["real"]),
                     float(source_te["bare_reference"]["tE_z50"]["imag"]))
    nominal = next(row for row in source_te["nominal_10nm_rows"] if row["state"] == "solid_25C")
    film_t = complex(float(nominal["tE_z50_real"]), float(nominal["tE_z50_imag"]))
    source_q0 = film_t / bare_t
    check_close(q0.real, source_q0.real, "ADMA2025 q0 real")
    check_close(q0.imag, source_q0.imag, "ADMA2025 q0 imaginary")
    check_close(abs(q0) ** 2, float(nominal["T"]) / bare_transmission,
                "ADMA2025 q0 power normalization", 1.0e-9)
    direct_fields = source_fields["2048_direct_bare_generator_entry"]
    reference_plane = source_metadata["grids"]["2048"]["arms"]["direct"]["planes"]["generator_entry"]["plane"]
    reference_area = math.prod(float(x) for x in reference_plane["pixel_pitch_m"])
    effective_means = []
    for entry in direct_fields:
        image = np.abs(entry) ** 2
        denominator = float(image.sum() * reference_area)
        numerator = float((image ** 2).sum() * reference_area)
        require(denominator > 0.0, "zero field in I_ref population")
        effective_means.append(numerator / denominator)
    source_i_ref = float(np.median(np.asarray(effective_means, dtype=np.float64)))
    check_close(i_ref, source_i_ref, "frozen I_ref", 1.0e-9)
    per_row_max: dict[str, float] = {}
    response_summaries: list[dict[str, Any]] = []
    checked_rows = 0

    for grid in (2048, 4096):
        archive = np.load(measurement_dir / f"fields_grid_{grid}.npz", allow_pickle=False)
        expected_count = 100 if grid == 2048 else 10
        for arm in ("direct", "mapped"):
            prefix = f"{grid}_{arm}"
            positions = archive[f"{prefix}_population_positions"].astype(np.int64)
            indices = archive[f"{prefix}_native_capture_indices"].astype(np.int64)
            labels = archive[f"{prefix}_labels"].astype(np.int64)
            storage_rows = positions if grid == 2048 else np.arange(expected_count)
            source_prefix = f"{grid}_{arm}"
            bare_entry = source_fields[f"{source_prefix}_bare_generator_entry"][storage_rows]
            bare_camera = source_fields[f"{source_prefix}_bare_camera"][storage_rows]
            entry_meta = metadata["grids"][str(grid)]["arms"][arm]["generator_entry"]
            camera_meta = metadata["grids"][str(grid)]["arms"][arm]["camera"]
            entry_pitch = tuple(float(x) for x in entry_meta["pixel_pitch_m"])
            camera_pitch = tuple(float(x) for x in camera_meta["pixel_pitch_m"])
            require(entry_meta["shape"] == [100, 100] and entry_pitch == (2.0e-6, 2.0e-6)
                    and float(entry_meta["wavelength_m"]) == 532e-9,
                    f"generator-entry physical coordinates differ for {prefix}")
            require(camera_meta["shape"] == [200, 200] and camera_pitch == (2.5e-6, 2.5e-6),
                    f"numerical CMOS coordinates differ for {prefix}")
            require(np.array_equal(positions, np.arange(100) if grid == 2048
                                   else np.arange(0, 100, 10)),
                    f"population position map differs for {prefix}")
            require(np.array_equal(indices, np.asarray(identity["native_capture_indices"], dtype=np.int64)[positions])
                    and np.array_equal(labels, np.asarray(identity["labels"], dtype=np.int64)[positions]),
                    f"sample identities differ from the frozen population for {prefix}")
            out_entry = archive[f"{prefix}_hypothetical_entry"]
            out_camera = archive[f"{prefix}_hypothetical_camera"]
            saved_raw = archive[f"{prefix}_hypothetical_raw_w_m2"]
            saved_fixed = archive[f"{prefix}_hypothetical_fixed_native_u8"]
            saved_view = archive[f"{prefix}_hypothetical_fixed_view28"]
            require(out_entry.shape == (expected_count, 12, 100, 100)
                    and out_entry.dtype == np.complex128
                    and out_camera.shape == (expected_count, 12, 200, 200)
                    and out_camera.dtype == np.complex128
                    and saved_raw.shape == (expected_count, 12, 200, 200)
                    and saved_raw.dtype == np.float64
                    and saved_fixed.shape == (expected_count, 12, 200, 200)
                    and saved_fixed.dtype == np.uint8
                    and saved_view.shape == (expected_count, 12, 28, 28),
                    f"saved field or image array shape/dtype differs for {prefix}")
            entry_area = math.prod(entry_pitch)
            camera_area = math.prod(camera_pitch)
            gain_w_inverse = float(identity["camera_gain_w_inverse"])
            by_condition: dict[str, list[tuple[dict[str, float], int]]] = {
                x["condition_id"]: [] for x in conditions}

            for sample_pos, population_pos in enumerate(positions.tolist()):
                e0 = bare_entry[sample_pos]
                c0 = bare_camera[sample_pos]
                intensity = np.abs(e0) ** 2
                bare_input_power = float(intensity.sum() * entry_area)
                bare_raw = np.abs(c0) ** 2
                bare_camera_power = float(bare_raw.sum() * camera_area)
                for condition_index, condition in enumerate(conditions):
                    identifier = condition["condition_id"]
                    u = intensity / (float(condition["a"]) * i_ref)
                    h = u / (1.0 + u)
                    gain = 1.0 + float(condition["sigma"]) * float(condition["m"]) * h
                    qmap = q0 * np.sqrt(gain) * np.exp(1j * float(condition["psi_rad"]) * h)
                    expected_entry = e0 * qmap
                    absolute_transmission = bare_transmission * np.abs(qmap) ** 2
                    require(np.isfinite(absolute_transmission).all()
                            and float(absolute_transmission.max()) <= 1.0 + 1.0e-12,
                            f"hypothetical transmission violates passive bounds for {prefix}/{identifier}")
                    stored_entry = out_entry[sample_pos, condition_index]
                    entry_error = relative_error(stored_entry, expected_entry)
                    camera = out_camera[sample_pos, condition_index]
                    recomputed_raw = np.abs(camera) ** 2
                    raw_error = relative_error(saved_raw[sample_pos, condition_index], recomputed_raw)
                    fixed = np.clip(recomputed_raw * camera_area * gain_w_inverse, 0.0, 1.0)
                    fixed_u8 = np.rint(fixed * 255.0).astype(np.uint8)
                    fixed_error = float(np.count_nonzero(fixed_u8 != saved_fixed[sample_pos, condition_index]))
                    view_error = float(np.max(np.abs(adaptive_average_28(fixed)
                                                     - saved_view[sample_pos, condition_index])))
                    require(entry_error <= 1.0e-12 and raw_error <= 1.0e-12
                            and fixed_error == 0.0 and view_error <= 1.0e-6,
                            f"saved production fields/readout fail independent recomputation: {prefix}/{population_pos}/{identifier}")
                    control_phase = float(np.angle(np.vdot(e0, expected_entry)))
                    input_ratio = float((np.abs(expected_entry) ** 2).sum()
                                        / max(float(intensity.sum()), 1.0e-300))
                    control = math.sqrt(input_ratio) * np.exp(1j * control_phase)
                    predicted_control = c0 * control
                    control_raw = np.abs(predicted_control) ** 2
                    fit_alpha = np.vdot(c0, camera) / max(float(np.vdot(c0, c0).real), 1.0e-300)
                    fit_residual = float(np.linalg.norm((camera - fit_alpha * c0).ravel())
                                         / max(float(np.linalg.norm(camera.ravel())), 1.0e-300))
                    observed_phase = float(np.angle(np.vdot(c0, camera)))
                    output_power = float(recomputed_raw.sum() * camera_area)
                    input_power = float((np.abs(expected_entry) ** 2).sum() * entry_area)
                    control_camera_power = float(control_raw.sum() * camera_area)
                    control_input_power = float((np.abs(e0 * control) ** 2).sum() * entry_area)
                    control_input_power_residual = abs(input_power - control_input_power) / max(input_power, 1.0e-300)
                    clipped = float(np.mean(recomputed_raw * camera_area * gain_w_inverse >= 1.0))
                    row = metric_rows[(grid, arm, int(population_pos), identifier)]
                    values = {
                        "input_power_before_w": bare_input_power,
                        "input_power_after_w": input_power,
                        "membrane_power_ratio": input_power / max(bare_input_power, 1.0e-300),
                        "cmos_power_before_w": bare_camera_power,
                        "cmos_power_after_w": output_power,
                        "cmos_power_ratio": output_power / max(bare_camera_power, 1.0e-300),
                        "equal_power_control_cmos_power_w": control_camera_power,
                        "equal_power_control_cmos_power_ratio": control_camera_power / max(bare_camera_power, 1.0e-300),
                        "equal_power_control_scalar_real": float(control.real),
                        "equal_power_control_scalar_imag": float(control.imag),
                        "equal_power_control_input_power_residual": control_input_power_residual,
                        "normalized_shape_error_vs_bare": shape_error(recomputed_raw, bare_raw),
                        "normalized_shape_error_vs_equal_power": shape_error(recomputed_raw, control_raw),
                        "best_uniform_complex_fit_residual": fit_residual,
                        "fixed_exposure_clipped_fraction": clipped,
                    }
                    row_errors = {key: check_close(float(row[key]), value, key)
                                  for key, value in values.items()}
                    row_errors["equal_power_control_scalar_real"] = check_close(
                        float(row["equal_power_control_scalar_real"]), float(control.real),
                        "equal-power scalar real")
                    row_errors["equal_power_control_scalar_imag"] = check_close(
                        float(row["equal_power_control_scalar_imag"]), float(control.imag),
                        "equal-power scalar imaginary")
                    require(abs(phase_error(observed_phase, float(row["cmos_phase_vs_bare_rad"]))) <= 1.0e-8,
                            f"phase metric differs for {prefix}/{population_pos}/{identifier}")
                    row_errors["cmos_phase_vs_bare_rad"] = abs(
                        phase_error(observed_phase, float(row["cmos_phase_vs_bare_rad"])))
                    row_errors["field_from_declared_q"] = entry_error
                    row_errors["raw_from_complex_camera"] = raw_error
                    row_errors["fixed_native_u8_pixel_mismatches"] = fixed_error
                    row_errors["fixed_view28_max_abs"] = view_error
                    for name, error in row_errors.items():
                        per_row_max[name] = max(per_row_max.get(name, 0.0), float(error))
                    by_condition[identifier].append((values, int(labels[sample_pos])))
                    checked_rows += 1

            for condition in conditions:
                identifier = condition["condition_id"]
                rows = by_condition[identifier]
                for label in ["all", *range(10)]:
                    selected = [values for values, saved_label in rows
                                if label == "all" or saved_label == label]
                    metric_subset = [metric_rows[(grid, arm, int(positions[sample_pos]), identifier)]
                                     for sample_pos in range(len(positions))
                                     if label == "all" or int(labels[sample_pos]) == label]
                    if not selected:
                        continue
                    response_summaries.append({
                        "grid": grid, "arm": arm, "condition_id": identifier,
                        "class_label": label, "sample_count": len(selected),
                        "ED_prediction_changed_count": sum(x["ED_prediction_changed"].lower() == "true"
                                                             for x in metric_subset),
                        "EC_prediction_changed_count": sum(x["EC_prediction_changed"].lower() == "true"
                                                             for x in metric_subset),
                        "normalized_shape_error_vs_equal_power_median": float(np.median(
                            [x["normalized_shape_error_vs_equal_power"] for x in selected])),
                        "membrane_power_ratio_median": float(np.median([x["membrane_power_ratio"] for x in selected])),
                        "cmos_power_ratio_median": float(np.median([x["cmos_power_ratio"] for x in selected])),
                        "best_uniform_complex_fit_residual_median": float(np.median(
                            [x["best_uniform_complex_fit_residual"] for x in selected])),
                    })

    cross_checks: list[dict[str, Any]] = []
    for arm in ("direct", "mapped"):
        with np.load(measurement_dir / "fields_grid_2048.npz", allow_pickle=False) as a, \
             np.load(measurement_dir / "fields_grid_4096.npz", allow_pickle=False) as b:
            for condition_index, condition in enumerate(conditions):
                for reference_index, position in enumerate(range(0, 100, 10)):
                    raw_a = a[f"2048_{arm}_hypothetical_raw_w_m2"][position, condition_index]
                    raw_b = b[f"4096_{arm}_hypothetical_raw_w_m2"][reference_index, condition_index]
                    raw_error = relative_error(raw_a, raw_b)
                    shape = shape_error(raw_a, raw_b)
                    cross_checks.append({"arm": arm, "condition_id": condition["condition_id"],
                                         "population_position": position,
                                         "raw_intensity_relative_error": raw_error,
                                         "normalized_shape_relative_error": shape})
    max_cross_raw = max(x["raw_intensity_relative_error"] for x in cross_checks)
    max_cross_shape = max(x["normalized_shape_relative_error"] for x in cross_checks)
    require(max_cross_raw <= 1.0e-3, f"independent 2048/4096 raw mismatch is {max_cross_raw:g}")
    with (measurement_dir / "cross_grid_checks.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_cross = list(csv.DictReader(stream))
    require(len(saved_cross) == len(cross_checks), "saved cross-grid table row count differs")
    saved_cross_map = {(x["arm"], x["condition_id"], int(x["population_position"])): x
                       for x in saved_cross}
    for item in cross_checks:
        key = (item["arm"], item["condition_id"], item["population_position"])
        saved = saved_cross_map[key]
        check_close(float(saved["raw_intensity_relative_error"]),
                    item["raw_intensity_relative_error"], "cross-grid raw error", 1.0e-6)
        check_close(float(saved["normalized_shape_relative_error"]),
                    item["normalized_shape_relative_error"], "cross-grid normalized shape error", 1.0e-6)
    require(checked_rows == 2640, f"independent consumer checked {checked_rows} rows")
    require(all(x["frozen_model_state_unchanged"] is True for x in measurement["grid_receipts"]),
            "production record does not confirm frozen optical state")
    wrong = [x["wrong_power_as_field_raw_relative_discrepancy"]
             for x in preflight["production_control_rows"]
             if x.get("wrong_power_as_field_raw_relative_discrepancy") is not None]
    require(len(wrong) == 8 and min(wrong) > 1.0e-2,
            "production negative-control evidence is incomplete")

    summary_path = measurement_dir / "windows_condition_summary.csv"
    with summary_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(response_summaries[0]))
        writer.writeheader()
        writer.writerows(response_summaries)
    receipt = {
        "schema": "lightgen_g1_nonlinear_sensitivity_windows_reconsumption.v1",
        "status": "SAVED_EVIDENCE_VERIFIED", "scientific_gate": "none; diagnostic sensitivity only",
        "optimizer_updates": 0, "scientific_updates": 0, "checked_sample_rows": checked_rows,
        "conditions": 12, "primary_rows_2048": 2400, "reference_rows_4096": 240,
        "I_ref_w_m2": i_ref, "q0_real": q0.real, "q0_imag": q0.imag,
        "maximum_reconsumption_differences": per_row_max,
        "cross_grid": {"maximum_raw_intensity_relative_error": max_cross_raw,
                       "maximum_normalized_shape_relative_error": max_cross_shape,
                       "raw_tolerance": 1.0e-3},
        "negative_control": {"rows": len(wrong), "minimum_wrong_power_raw_discrepancy": min(wrong)},
        "all_frozen_state_receipts_true": True,
        "limitations": ["all twelve local response functions are mathematical sensitivity assumptions",
                        "not measured Te response, not hardware prediction, not a training or adoption result"],
    }
    with (measurement_dir / "windows_reconsumption.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
