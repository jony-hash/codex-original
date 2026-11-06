"""Directly replay equal-power controls for the shared ten-sample G1 subset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.lightgen import diagnose_lightgen_g1_nonlinear_sensitivity_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_POSITIONS = list(range(0, 100, 10))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm((actual - expected).ravel())
                 / max(float(np.linalg.norm(expected.ravel())), 1.0e-300))


def normalized_shape_error(actual: np.ndarray, expected: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    a = a / max(float(a.sum()), 1.0e-300)
    b = b / max(float(b.sum()), 1.0e-300)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-300))


def run(config_path: Path, measurement_dir: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    measurement_dir = measurement_dir.resolve()
    output_dir = output_dir.resolve()
    config = diagnostic._load_config(config_path)
    expected_measurement = (ROOT / config["output"]["run_dir"]).resolve()
    require(measurement_dir == expected_measurement,
            "control replay must consume the fixed attempt 01 measurement")
    require(output_dir.parent == measurement_dir.parent
            and output_dir.name.startswith("diagnostic-only")
            and not output_dir.exists(),
            "control replay output must be a new diagnostic-only sibling run directory")
    measurement = diagnostic._read_json(measurement_dir / "measurement.json")
    require(measurement.get("status") == "COMPLETE_DIAGNOSTIC_ONLY"
            and measurement.get("optimizer_updates") == 0
            and measurement.get("scientific_updates") == 0,
            "the frozen sensitivity measurement is incomplete or has updates")
    preflight = diagnostic._read_json(measurement_dir / "preflight.json")
    require(preflight.get("status") == "PREFLIGHT_PASSED",
            "the production preflight is not passed")
    conditions = measurement["conditions"]
    require(conditions == diagnostic._conditions(config),
            "saved measurement conditions differ from the declared twelve assumptions")

    torch = diagnostic.torch
    require(torch.cuda.is_available(), "direct control replay requires the authorized CUDA host")
    device = torch.device("cuda")
    base = diagnostic._load_base(config, device)
    identity = measurement["identity"]
    native_indices = np.asarray(identity["native_capture_indices"], dtype=np.int64)
    labels = np.asarray(identity["labels"], dtype=np.int64)
    require(native_indices.shape == (100,) and labels.shape == (100,),
            "saved measurement identity does not contain the frozen 100-row population")
    rows: list[dict[str, Any]] = []
    thresholds = {
        "nonlinear_camera_replay_error": diagnostic.PRIMARY_TOLERANCE,
        "relative_field_error": diagnostic.PRIMARY_TOLERANCE,
        "input_power_residual": diagnostic.PRIMARY_TOLERANCE,
        "camera_power_residual": diagnostic.PRIMARY_TOLERANCE,
        "camera_shape_error": diagnostic.PRIMARY_TOLERANCE,
    }

    for grid in (2048, 4096):
        count = 100 if grid == 2048 else 10
        selected_positions = REFERENCE_POSITIONS
        fields_archive = np.load(measurement_dir / f"fields_grid_{grid}.npz", allow_pickle=False)
        for arm in ("direct", "mapped"):
            prefix = f"{grid}_{arm}"
            positions_saved = np.asarray(fields_archive[f"{prefix}_population_positions"], dtype=np.int64)
            indices_saved = np.asarray(fields_archive[f"{prefix}_native_capture_indices"], dtype=np.int64)
            labels_saved = np.asarray(fields_archive[f"{prefix}_labels"], dtype=np.int64)
            expected_storage_positions = (np.arange(100, dtype=np.int64) if grid == 2048
                                          else np.asarray(REFERENCE_POSITIONS, dtype=np.int64))
            require(positions_saved.shape == (count,)
                    and np.array_equal(positions_saved, expected_storage_positions)
                    and np.array_equal(indices_saved, native_indices[expected_storage_positions])
                    and np.array_equal(labels_saved, labels[expected_storage_positions]),
                    f"saved population identity is invalid for {prefix}")
            source_entry = base["arrays"][f"{prefix}_bare_generator_entry"]
            source_camera = base["arrays"][f"{prefix}_bare_camera"]
            saved_entry = fields_archive[f"{prefix}_hypothetical_entry"]
            saved_camera = fields_archive[f"{prefix}_hypothetical_camera"]
            entry_plane = base["metadata"]["grids"][str(grid)]["arms"][arm]["planes"]["generator_entry"]["plane"]
            camera_plane = base["metadata"]["grids"][str(grid)]["arms"][arm]["planes"]["camera"]["plane"]
            adapter = diagnostic.mapper_ref._new_optical_model(
                base["context"]["optical_checkpoint"], grid, device)
            adapter.eval().requires_grad_(False)
            diagnostic._plane_matches(adapter.runtime.generator_entrance_plane,
                                      entry_plane, f"{prefix}.generator_entry")
            diagnostic._plane_matches(adapter.runtime.camera_plane,
                                      camera_plane, f"{prefix}.camera")
            model_before = diagnostic.linear_ref._snapshot(adapter)
            pixel_area = math.prod(adapter.runtime.camera_plane.pixel_pitch_m)
            entry_area = math.prod(adapter.runtime.generator_entrance_plane.pixel_pitch_m)

            for selected_position in selected_positions:
                local_pos = selected_position if grid == 2048 else REFERENCE_POSITIONS.index(selected_position)
                e0_np = np.asarray(source_entry[local_pos], dtype=np.complex128)
                c0_np = np.asarray(source_camera[local_pos], dtype=np.complex128)
                e0 = diagnostic._complex_field(
                    torch.as_tensor(e0_np[None, None], device=device),
                    adapter.runtime.generator_entrance_plane, "attempt04_bare_entry_control_replay")
                bare_entry_power = diagnostic._power(e0.tensor, entry_area)
                for condition_index, condition in enumerate(conditions):
                    nonlinear_entry_np = np.asarray(saved_entry[local_pos, condition_index], dtype=np.complex128)
                    saved_nonlinear_camera_np = np.asarray(saved_camera[local_pos, condition_index], dtype=np.complex128)
                    nonlinear_entry = diagnostic._complex_field(
                        torch.as_tensor(nonlinear_entry_np[None, None], device=device),
                        adapter.runtime.generator_entrance_plane,
                        "saved_nonlinear_entry_direct_replay")
                    nonlinear_camera = diagnostic._camera_from_entry(adapter.runtime, nonlinear_entry)
                    nonlinear_camera_np = nonlinear_camera.tensor[0, 0].detach().cpu().numpy()
                    nonlinear_replay_error = relative_error(nonlinear_camera_np,
                                                            saved_nonlinear_camera_np)
                    require(nonlinear_replay_error <= thresholds["nonlinear_camera_replay_error"],
                            f"saved nonlinear camera field does not replay for {prefix}/{selected_position}/{condition['condition_id']}: "
                            f"{nonlinear_replay_error:g}")
                    overlap = np.vdot(e0_np, nonlinear_entry_np)
                    require(abs(overlap) > 1.0e-30 and bare_entry_power > 0.0,
                            f"equal-power control phase/power undefined for {prefix}/{selected_position}/{condition['condition_id']}")
                    control_phase = float(np.angle(overlap))
                    nonlinear_entry = torch.as_tensor(nonlinear_entry_np[None, None], device=device)
                    nonlinear_power = diagnostic._power(nonlinear_entry, entry_area)
                    scalar = math.sqrt(nonlinear_power / bare_entry_power) * complex(
                        math.cos(control_phase), math.sin(control_phase))
                    control_entry = diagnostic._complex_field(
                        e0.tensor * scalar, e0.plane, "equal_power_uniform_complex_control_replay")
                    control_input_power = diagnostic._power(control_entry.tensor, entry_area)
                    control_camera = diagnostic._camera_from_entry(adapter.runtime, control_entry)
                    expected_camera = c0_np * scalar
                    control_camera_np = control_camera.tensor[0, 0].detach().cpu().numpy()
                    field_error = relative_error(control_camera_np, expected_camera)
                    expected_raw = np.abs(expected_camera) ** 2
                    control_raw = np.abs(control_camera_np) ** 2
                    camera_power = float(control_raw.sum() * pixel_area)
                    expected_camera_power = float(expected_raw.sum() * pixel_area)
                    input_residual = abs(control_input_power - nonlinear_power) / max(nonlinear_power, 1.0e-300)
                    camera_power_residual = abs(camera_power - expected_camera_power) / max(expected_camera_power, 1.0e-300)
                    shape_error = normalized_shape_error(control_raw, expected_raw)
                    require(field_error <= thresholds["relative_field_error"]
                            and input_residual <= thresholds["input_power_residual"]
                            and camera_power_residual <= thresholds["camera_power_residual"]
                            and shape_error <= thresholds["camera_shape_error"],
                            f"direct equal-power replay failed for {prefix}/{selected_position}/{condition['condition_id']}: "
                            f"field={field_error:g}, input_power={input_residual:g}, "
                            f"camera_power={camera_power_residual:g}, shape={shape_error:g}")
                    rows.append({
                        "grid": grid, "arm": arm, "population_position": selected_position,
                        "native_capture_index": int(native_indices[selected_position]),
                        "label": int(labels[selected_position]),
                        "condition_id": condition["condition_id"],
                        "control_scalar_real": scalar.real, "control_scalar_imag": scalar.imag,
                        "nonlinear_entry_power_w": nonlinear_power,
                        "control_entry_power_w": control_input_power,
                        "relative_input_power_residual": input_residual,
                        "nonlinear_camera_replay_error": nonlinear_replay_error,
                        "direct_control_camera_power_w": camera_power,
                        "linearity_predicted_camera_power_w": expected_camera_power,
                        "relative_camera_power_residual": camera_power_residual,
                        "relative_camera_complex_field_error": field_error,
                        "normalized_camera_shape_error": shape_error,
                    })
            diagnostic.linear_ref._assert_unchanged(adapter, model_before, f"G1 control replay grid {grid}")
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_rows = 10 * 2 * 2 * len(conditions)
    require(len(rows) == expected_rows == 480,
            f"expected 480 direct control propagation rows, got {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "equal_power_control_direct_replay.csv").open(
            "x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_errors = {
        key: max(float(row[column]) for row in rows)
        for key, column in (
            ("nonlinear_camera_replay_error", "nonlinear_camera_replay_error"),
            ("relative_field_error", "relative_camera_complex_field_error"),
            ("input_power_residual", "relative_input_power_residual"),
            ("camera_power_residual", "relative_camera_power_residual"),
            ("camera_shape_error", "normalized_camera_shape_error"),
        )
    }
    result = {
        "schema": "lightgen_g1_equal_power_control_direct_replay.v1",
        "status": "DIRECT_REPLAY_PASSED",
        "evidence_tier": "diagnostic_only",
        "optimizer_updates": 0,
        "scientific_updates": 0,
        "population_positions": REFERENCE_POSITIONS,
        "grids": [2048, 4096],
        "arms": ["direct", "mapped"],
        "conditions": [item["condition_id"] for item in conditions],
        "rows": len(rows),
        "maximum_errors": max_errors,
        "thresholds": thresholds,
        "frozen_optical_state_unchanged": True,
        "interpretation": "Verifies the equal-power uniform complex control by direct propagation for the shared ten-row subset; this is a numerical control validation, not a Te measurement or material model.",
    }
    with (output_dir / "equal_power_control_direct_replay.json").open(
            "x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.measurement_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
