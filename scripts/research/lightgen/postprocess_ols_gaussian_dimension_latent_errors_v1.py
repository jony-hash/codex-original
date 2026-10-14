"""Compute per-fiber complex-latent error arrays from saved G3 ROI fields only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen.diagnose_ols_gaussian_dimension_reaudit_v1 import (
    EPS,
    PHASE_NAMES,
    _array_coordinates,
    _load_config,
    _make_runtime,
    _mode_coefficients,
    _phase_alignment,
    _rel_l2,
)


def _complex_pairs(value: torch.Tensor) -> list[list[float]]:
    return torch.stack((value.real, value.imag), dim=-1).detach().cpu().tolist()


def run(config_path: Path, raw_fields_path: Path, reference_report_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    if not raw_fields_path.is_file():
        raise FileNotFoundError(f"saved G3 fields are missing: {raw_fields_path}")
    if not reference_report_path.is_file():
        raise FileNotFoundError(f"previous G3 report is missing: {reference_report_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite latent-error output: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("saved-field modal projection requires the configured AutoDL CUDA device")

    payload = torch.load(raw_fields_path, map_location="cpu", weights_only=True)
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    grids = tuple(int(value) for value in config["optics"]["propagation_grid_sizes"])
    input_power_w = float(config["optics"]["input_power_w"])
    phase_data: dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]] = {}
    device = torch.device("cuda")
    for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
        runtime = _make_runtime(grids[0], real_dtype, device)
        phase_data[precision] = {}
        try:
            for phase_name in PHASE_NAMES:
                phase_data[precision][phase_name] = {"gaussian_s12": {}, "lp01": {}}
                for grid in grids:
                    field_tensor = payload[phase_name][precision][str(grid)].to(device=device)
                    field = ComplexField(
                        field_tensor[None, None],
                        runtime.ols_plane,
                        OperationProvenance("saved_g3_common_roi_latent_readout", runtime.ols_plane.plane_id, None),
                    )
                    for mode in ("gaussian_s12", "lp01"):
                        coefficients = _mode_coefficients(field, runtime, mode)[0]
                        phase_data[precision][phase_name][mode][grid] = (
                            coefficients.detach().cpu().to(torch.complex128)
                        )
                    del field, field_tensor, coefficients
        finally:
            del runtime
            torch.cuda.empty_cache()

    record: dict[str, Any] = {
        "schema": "ols_gaussian_dimension_latent_readout.v1",
        "status": "COMPLETE_FROM_SAVED_G3_FIELDS",
        "source_saved_fields": str(raw_fields_path),
        "source_previous_report": str(reference_report_path),
        "fixed_phases": list(PHASE_NAMES),
        "window_grid_sizes": list(grids),
        "fiber_pitch_m": float(config["fiber"]["reference_pitch_m"]),
        "fiber_number": 100,
        "fiber_centers_xy_m_row_major": [],
        "common_phase_alignment": "one scalar per 100-channel complex latent vector; no independent fiber alignment",
        "amplitude_denominator_floor": "max(max_abs_reference_complex_coefficient * 1e-12, 1e-30 sqrt(W))",
        "absolute_error_normalization": "per-fiber |delta c_j| / sqrt(configured source power)",
        "propagation_rerun": False,
        "projection_pixel_area_m2": (2.0e-6) ** 2,
        "phase_optimizer_updates": 0,
        "formal_training_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "comparisons": {},
        "previous_report_crosscheck": {},
    }
    coordinates = _array_coordinates(10, float(config["fiber"]["reference_pitch_m"]), torch.device("cpu"))
    record["fiber_centers_xy_m_row_major"] = [[float(x), float(y)] for y in coordinates for x in coordinates]
    for precision in ("complex64", "complex128"):
        record["comparisons"][precision] = {}
        for phase_name in PHASE_NAMES:
            record["comparisons"][precision][phase_name] = {}
            for left_index, reference_grid in enumerate(grids):
                for test_grid in grids[left_index + 1 :]:
                    pair = f"{reference_grid}_vs_{test_grid}"
                    record["comparisons"][precision][phase_name][pair] = {}
                    for mode in ("gaussian_s12", "lp01"):
                        reference = phase_data[precision][phase_name][mode][reference_grid]
                        test = phase_data[precision][phase_name][mode][test_grid]
                        global_phase, aligned_test = _phase_alignment(reference, test)
                        raw_abs = (test - reference).abs()
                        aligned_abs = (aligned_test - reference).abs()
                        magnitude = reference.abs()
                        floor = max(float(magnitude.max()) * 1.0e-12, EPS)
                        reference_power = reference.abs().square()
                        test_power = test.abs().square()
                        power_delta = test_power - reference_power
                        power_floor = max(float(reference_power.max()) * 1.0e-12, EPS)
                        power_relative = power_delta.abs() / reference_power.clamp_min(power_floor)
                        current_raw_error = _rel_l2(reference, test)
                        current_aligned_error = _rel_l2(reference, aligned_test)
                        previous = reference_report["G3"]["comparisons"][precision][phase_name][pair]["fiber_modes"][mode]
                        previous_power_delta = previous["per_fiber_power_absolute_error_w"]
                        previous_power_relative = previous["per_fiber_power_relative_error"]
                        if len(previous_power_delta) != int(reference.numel()) or len(previous_power_relative) != int(reference.numel()):
                            raise ValueError(f"previous report fiber count mismatch: {precision}/{phase_name}/{pair}/{mode}")
                        power_delta_error = max(
                            abs(float(value) - float(previous_value))
                            for value, previous_value in zip(power_delta.detach().cpu().tolist(), previous_power_delta, strict=True)
                        )
                        power_relative_error = max(
                            abs(float(value) - float(previous_value))
                            for value, previous_value in zip(power_relative.detach().cpu().tolist(), previous_power_relative, strict=True)
                        )
                        raw_scalar_error = abs(current_raw_error - float(previous["complex_latent_raw_relative_error"]))
                        aligned_scalar_error = abs(
                            current_aligned_error - float(previous["complex_latent_common_phase_aligned_relative_error"])
                        )
                        record["comparisons"][precision][phase_name][pair][mode] = {
                            "shared_global_phase_rad": global_phase,
                            "complex_latent_vector_raw_relative_l2_error": current_raw_error,
                            "complex_latent_vector_aligned_relative_l2_error": current_aligned_error,
                            "per_fiber_reference_complex_coefficient_sqrt_w": _complex_pairs(reference),
                            "per_fiber_test_complex_coefficient_sqrt_w": _complex_pairs(test),
                            "per_fiber_test_aligned_complex_coefficient_sqrt_w": _complex_pairs(aligned_test),
                            "per_fiber_raw_absolute_complex_error_sqrt_w": raw_abs.tolist(),
                            "per_fiber_aligned_absolute_complex_error_sqrt_w": aligned_abs.tolist(),
                            "per_fiber_raw_error_over_sqrt_source_power": (raw_abs / math.sqrt(input_power_w)).tolist(),
                            "per_fiber_aligned_error_over_sqrt_source_power": (aligned_abs / math.sqrt(input_power_w)).tolist(),
                            "per_fiber_raw_relative_error_to_reference_amplitude": (raw_abs / magnitude.clamp_min(floor)).tolist(),
                            "per_fiber_aligned_relative_error_to_reference_amplitude": (aligned_abs / magnitude.clamp_min(floor)).tolist(),
                            "reference_amplitude_denominator_floor_sqrt_w": floor,
                            "per_fiber_reference_power_w": reference_power.tolist(),
                            "per_fiber_test_power_w": test_power.tolist(),
                            "per_fiber_power_absolute_error_w": power_delta.tolist(),
                            "per_fiber_power_relative_error": power_relative.tolist(),
                            "per_fiber_power_denominator_floor_w": power_floor,
                            "previous_report_crosscheck": {
                                "raw_latent_scalar_absolute_difference": raw_scalar_error,
                                "aligned_latent_scalar_absolute_difference": aligned_scalar_error,
                                "per_fiber_power_absolute_error_max_abs_difference_w": power_delta_error,
                                "per_fiber_power_relative_error_max_abs_difference": power_relative_error,
                                "passes_serialized_readout_consistency_1e-12": max(
                                    raw_scalar_error, aligned_scalar_error, power_delta_error, power_relative_error
                                ) <= 1.0e-12,
                            },
                        }
                        record["previous_report_crosscheck"].setdefault(precision, {}).setdefault(phase_name, {})[
                            f"{pair}/{mode}"
                        ] = record["comparisons"][precision][phase_name][pair][mode]["previous_report_crosscheck"]
    all_crosschecks = [
        item["previous_report_crosscheck"]["passes_serialized_readout_consistency_1e-12"]
        for precision_values in record["comparisons"].values()
        for phase_values in precision_values.values()
        for pair_values in phase_values.values()
        for item in pair_values.values()
    ]
    if not all(all_crosschecks):
        raise RuntimeError("saved-field per-fiber readout does not reproduce the previous report aggregates")
    record["previous_report_crosscheck_pass"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-fields", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.raw_fields, args.reference_report, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "propagation_rerun": False, "previous_report_crosscheck_pass": result["previous_report_crosscheck_pass"]}))


if __name__ == "__main__":
    main()
