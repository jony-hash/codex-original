"""Execute the frozen-phase HW207 P0-P7 diagnostic, stopping before P8."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.lightgen import hw207_physical_cmos_v1 as cmos
from scripts.research.lightgen import hw207_physical_frozen_phase_v1 as model


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run evidence: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run evidence: {path}")
    model.write_json(path, value)


def _run_manifest(config_path: Path, run_dir: Path, stage: str, device: str) -> dict[str, Any]:
    return {
        "task": "HW207_P0_P7_PHYSICAL_FROZEN_PHASE_MNIST4",
        "stage": stage,
        "run_directory": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256_provenance_only": model.sha256_file(config_path),
        "source_session_id": "01a18b9f-206d-7c61-96a3-0da3a49efb8c",
        "device_requested": device,
        "phase_updates": 0,
        "hardware_operation": False,
        "official_test_opened": False,
        "started_unix_seconds": time.time(),
        "repository_head": os.environ.get("CODEX_REPOSITORY_HEAD", "not captured"),
        "working_tree_dirty_entries_at_start": 359,
        "working_tree_staged_entries_at_start": 0,
        "evidence_scope": "ASSUMED_DIAGNOSTIC_NUMERICAL_MODEL_ONLY",
    }


def run_p0(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = model.read_json(config_path)
    source_config = model.read_json(model.SOURCE_CONFIG)
    data = model.load_c0_validation(config)
    candidates = model.candidate_metadata(config)
    p0_dir = run_dir / "p0"
    p0_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(
        p0_dir / "c0_validation_0_3.npz",
        original_indices=data["indices"], labels=data["labels"], images=data["images"],
    )
    data_ledger = {
        "evaluation_role": "C0 validation only",
        "population_source": data["source"],
        "sample_count": int(len(data["indices"])),
        "class_counts": data["class_counts"],
        "ordered_original_indices": data["indices"].tolist(),
        "indices_unique": bool(np.unique(data["indices"]).size == len(data["indices"])),
        "train_validation_index_overlap_count": len(set(
            model.read_json(model.repo_path(config["source_configs"]["c0_index_manifest"]))["train"]["original_indices"]
        ).intersection(data["indices"].tolist())),
        "validation_labels_verified_from_raw_idx": True,
        "c0_index_manifest": data["index_manifest_path"],
        "c0_index_manifest_sha256_provenance_only": data["index_manifest_sha256"],
        "raw_images": {"path": data["images_path"], "sha256_provenance_only": data["images_sha256"]},
        "raw_labels": {"path": data["labels_path"], "sha256_provenance_only": data["labels_sha256"]},
        "official_test": "not opened; historically used and not independent",
        "data_order": "C0 validation original_indices order, filtered to labels 0..3 without sorting",
        "historical_use_audit": {
            "current_population": "the same C0 validation indices were reused by C3.4-C3.8 development reports; this run is not an independent evaluation",
            "official_mnist_test": "previously opened in C0/C1/C2 workflows; no independent-test claim is available, and this run does not read test IDX files",
            "evidence_sources": [
                "docs/research/lightgen/hw207_mnist4_c0_report.md",
                "docs/research/lightgen/hw207_mnist4_c1_report.md",
                "docs/research/lightgen/hw207_mnist4_c2_report.md",
                "docs/research/lightgen/hw207_mnist4_c34_report.md",
                "docs/research/lightgen/hw207_mnist4_c35_report.md",
                "docs/research/lightgen/hw207_mnist4_c37_report.md",
                "docs/research/lightgen/hw207_mnist4_c38_report.md",
            ],
            "test_images_or_labels_loaded_by_this_entrypoint": False,
        },
    }
    if data_ledger["train_validation_index_overlap_count"] != 0:
        raise ValueError("C0 train and validation source indices overlap")
    geometry = model.focal_geometry_summary(config)
    p0 = {
        "status": "P0_PASS_DIAGNOSTIC_ASSET_AUDIT",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "data_ledger": data_ledger,
        "hardware_parameters": {
            "DMD": {
                "family": config["dmd"]["family"],
                "native_shape_rows_columns": config["dmd"]["native_shape_rows_columns"],
                "pitch_um": config["dmd"]["micromirror_pitch_um"],
                "fill_factor": config["dmd"]["fill_factor"],
                "tilt_deg": config["dmd"]["micromirror_tilt_deg"],
                "status": "MANUFACTURER_SPEC_OR_USER_SPEC_IMAGE; installed identity/port/orientation/mode UNKNOWN",
            },
            "input_relay": config["input_relay"],
            "phase_layer": config["phase_layer"],
            "output_relay": config["output_relay"],
            "CMOS": config["cmos"],
            "fixed_ROI": config["fixed_roi_geometry"],
        },
        "ABCD_diagnostic": geometry,
        "historical_phase_assets_are_immutable": True,
        "candidate_phase_arrays_copied_or_modified": False,
        "trained_weights_or_optimizer_opened": False,
        "parameter_evidence_states": {
            "wavelength_532_nm": {"value": 532.0, "status": "USER_DECLARED_NOMINAL; used as a theoretical numerical input, not a spectrometer measurement"},
            "DMD_family_and_chip": {"value": "F9000 / DLPC900 + DLP9000", "status": "MANUFACTURER_SPEC_OR_USER_SPEC_IMAGE; serial, firmware and installed orientation UNKNOWN"},
            "DMD_pitch_fill_tilt": {"value": {"pitch_um": 7.56, "fill_factor": 0.92, "tilt_deg": 12.0}, "status": "MANUFACTURER_SPEC_OR_USER_SPEC_IMAGE; actual ON/OFF collection and selected diffraction order UNKNOWN"},
            "phase_plane_target": {"value": "100x100 at 2 um pitch, 200x200 um", "status": "HW207_TARGET_NUMERICAL_GEOMETRY; device identity, stack, physical mapping and complex transmission UNKNOWN"},
            "tape_distances_01_to_13": {"status": "PHOTO_ESTIMATED; centimeter-rounded initial geometry, not calibrated measurements"},
            "relay_lens_focal_lengths_and_apertures": {"status": "BOM_NOMINAL; installed serial, principal planes and clear apertures UNKNOWN"},
            "CMOS_family_pixel_pitch_and_shape": {"value": "GSENSE2020 family, 6.5 um, 2048x2048", "status": "MANUFACTURER_SPEC_OR_USER_SPEC_IMAGE; camera-unit identity, readout and affine registration UNKNOWN"},
            "CMOS_affine_and_roi_registration": {"status": "ASSUMED_DIAGNOSTIC; not measured"},
            "phase_layer_complex_transmission": {"status": "UNKNOWN; numerical forward uses unit-amplitude exp(i*phase)"},
        },
        "source_metadata": {
            "hardware_geometry_config": "configs/research/lightgen/hw207_optical_geometry_v1.json",
            "CMOS_and_candidate_inventory_config": "configs/research/lightgen/hw207_physical_cmos_v1.json",
            "input_encoding_source": "scripts/research/lightgen/hw207_mnist4_c0_v1.py::features",
            "candidate_inventory_source": "configs/research/lightgen/hw207_physical_cmos_v1.json::phase_candidates_to_audit_when_model_is_calibrated",
            "historical_candidate_source_count": len(source_config["phase_candidates_to_audit_when_model_is_calibrated"]),
        },
        "historical_coordinate_and_mapping_audit": {
            "new_fixed_mapping": "class 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right in CMOS physical y-down coordinates",
            "new_fixed_roi_source": "configs/research/lightgen/hw207_physical_cmos_v1.json and user P0-P7 plan",
            "historical_c38_mapping_issue": "C3.8 figures used physical y-up with origin=lower while ROI names/config imply top-left origin; numerical historical prediction rows are not relabeled",
            "historical_phase_array_registration": "array row/column order retained; DMD-to-phase affine/rotation remains UNKNOWN",
            "this_run_reoriented_candidates": False,
        },
    }
    _json_once(p0_dir / "p0_asset_and_data_audit.json", p0)
    return p0


def _direct_fresnel_reference(field: np.ndarray, input_pitch_um: float,
                              output_shape: tuple[int, int], output_pitch_um: float,
                              distance_um: float, wavelength_um: float) -> np.ndarray:
    """Slow direct Riemann-sum reference independent of the CZT implementation."""
    in_y = (np.arange(field.shape[-2]) - (field.shape[-2] - 1) / 2.0) * input_pitch_um
    in_x = (np.arange(field.shape[-1]) - (field.shape[-1] - 1) / 2.0) * input_pitch_um
    out_y = (np.arange(output_shape[0]) - (output_shape[0] - 1) / 2.0) * output_pitch_um
    out_x = (np.arange(output_shape[1]) - (output_shape[1] - 1) / 2.0) * output_pitch_um
    yy_in, xx_in = np.meshgrid(in_y, in_x, indexing="ij")
    yy_out, xx_out = np.meshgrid(out_y, out_x, indexing="ij")
    source = field * np.exp(1j * math.pi * (xx_in**2 + yy_in**2) / (wavelength_um * distance_um))
    kernel = np.exp(-2j * math.pi * (xx_out[..., None, None] * xx_in + yy_out[..., None, None] * yy_in)
                    / (wavelength_um * distance_um))
    result = np.einsum("yx,abyx->ab", source, kernel, optimize=True)
    result *= np.exp(1j * math.pi * (xx_out**2 + yy_out**2) / (wavelength_um * distance_um))
    result *= -1j * input_pitch_um**2 / (wavelength_um * distance_um)
    return result


def _preflight_small_reference(device: torch.device) -> dict[str, Any]:
    rows, cols = np.mgrid[0:5, 0:6]
    random_field = (np.sin(rows * 0.7 + cols * 0.3) + 1j * np.cos(rows * 0.2 - cols * 0.5)).astype(np.complex128)
    expected = _direct_fresnel_reference(random_field, 3.1, (4, 7), 2.4, 120.0, model.WAVELENGTH_UM)
    tensor = torch.as_tensor(random_field[None], device=device, dtype=torch.complex128)
    actual = model.c34.scaled_fresnel(
        tensor, 3.1, (4, 7), 2.4, 120.0, wavelength_um=model.WAVELENGTH_UM,
        plan_cache={}, phase_dtype=torch.complex128,
    )[0].cpu().numpy()
    rel_l2 = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
    wrong_sign = _direct_fresnel_reference(random_field, 3.1, (4, 7), 2.4, -120.0, model.WAVELENGTH_UM)
    wrong_separation = float(np.linalg.norm(expected - wrong_sign) / max(np.linalg.norm(expected), 1e-30))

    # Axis-swap check uses a non-square field and output so a transposed
    # implementation cannot pass merely because the production grid is square.
    transposed = model.c34.scaled_fresnel(
        tensor.transpose(-2, -1), 3.1, (7, 4), 2.4, 120.0,
        wavelength_um=model.WAVELENGTH_UM, plan_cache={}, phase_dtype=torch.complex128,
    )[0].transpose(-2, -1).cpu().numpy()
    axis_swap_rel_l2 = float(np.linalg.norm(transposed - actual) / max(np.linalg.norm(actual), 1e-30))

    # Compare the production CZT path to the closed-form paraxial propagation
    # of a tilted Gaussian (a finite, smooth single-frequency probe).
    count, pitch_um, distance_um, waist_um = 65, 2.0, 500.0, 20.0
    fx, fy = 0.02, -0.015
    coordinate = (np.arange(count, dtype=np.float64) - (count - 1) / 2.0) * pitch_um
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    gaussian = np.exp(-(xx**2 + yy**2) / waist_um**2) * np.exp(2j * math.pi * (fx * xx + fy * yy))
    gaussian_tensor = torch.as_tensor(gaussian[None], device=device, dtype=torch.complex128)
    gaussian_actual = model.c34.scaled_fresnel(
        gaussian_tensor, pitch_um, (count, count), pitch_um, distance_um,
        wavelength_um=model.WAVELENGTH_UM, plan_cache={}, phase_dtype=torch.complex128,
    )[0].cpu().numpy()
    rayleigh_um = math.pi * waist_um**2 / model.WAVELENGTH_UM
    q = 1.0 + 1j * distance_um / rayleigh_um
    shifted_x = xx - model.WAVELENGTH_UM * distance_um * fx
    shifted_y = yy - model.WAVELENGTH_UM * distance_um * fy
    gaussian_expected = (
        np.exp(-(shifted_x**2 + shifted_y**2) / (waist_um**2 * q)) / q
        * np.exp(2j * math.pi * (fx * xx + fy * yy)
                - 1j * math.pi * model.WAVELENGTH_UM * distance_um * (fx**2 + fy**2))
    )
    gaussian_rel_l2 = float(
        np.linalg.norm(gaussian_actual - gaussian_expected)
        / max(np.linalg.norm(gaussian_expected), 1e-30)
    )

    actual32 = model.c34.scaled_fresnel(
        torch.as_tensor(random_field.astype(np.complex64)[None], device=device),
        3.1, (4, 7), 2.4, 120.0, wavelength_um=model.WAVELENGTH_UM,
        plan_cache={}, phase_dtype=torch.complex64,
    )[0].to(torch.complex128).cpu().numpy()
    precision_rel_l2 = float(np.linalg.norm(actual32 - actual) / max(np.linalg.norm(actual), 1e-30))
    energy64 = float(np.sum(np.abs(actual) ** 2, dtype=np.float64) * 2.4**2)
    energy32 = float(np.sum(np.abs(actual32) ** 2, dtype=np.float64) * 2.4**2)
    precision_energy_rel = abs(energy32 - energy64) / max(energy64, 1e-30)

    tolerances = {
        "direct_reference_relative_l2": 1e-8,
        "axis_swap_relative_l2": 1e-10,
        "analytic_gaussian_relative_l2": 1e-4,
        "fp32_vs_fp64_relative_l2": 1e-5,
        "fp32_vs_fp64_energy_relative_difference": 1e-5,
    }
    checks_pass = (
        rel_l2 <= tolerances["direct_reference_relative_l2"]
        and axis_swap_rel_l2 <= tolerances["axis_swap_relative_l2"]
        and gaussian_rel_l2 <= tolerances["analytic_gaussian_relative_l2"]
        and precision_rel_l2 <= tolerances["fp32_vs_fp64_relative_l2"]
        and precision_energy_rel <= tolerances["fp32_vs_fp64_energy_relative_difference"]
        and wrong_separation > 1e-3
    )
    return {
        "direct_fresnel_relative_l2": rel_l2,
        "direct_reference_tolerance": tolerances["direct_reference_relative_l2"],
        "wrong_propagation_sign_relative_difference": wrong_separation,
        "wrong_sign_counterexample_rejected": wrong_separation > 1e-3,
        "axis_swap_relative_l2": axis_swap_rel_l2,
        "analytic_tilted_gaussian_relative_l2": gaussian_rel_l2,
        "fp32_vs_fp64_complex_field_relative_l2": precision_rel_l2,
        "fp32_vs_fp64_energy_relative_difference": precision_energy_rel,
        "tolerances": tolerances,
        "status": "PASS" if checks_pass else "FAIL",
    }


def _rows_first_per_class(data: dict[str, Any], count: int) -> np.ndarray:
    ids = []
    for label in range(4):
        matched = np.flatnonzero(data["labels"] == label)
        if matched.size < count:
            raise ValueError(f"validation population lacks {count} rows for class {label}")
        ids.extend(matched[:count].tolist())
    return np.asarray(ids, dtype=np.int64)


def _sampling_admission(p2: dict[str, Any], consumer: str) -> bool:
    """Return whether downstream results are sampling-limited; reject invalid runs."""
    if p2.get("status") != "PREFLIGHT_READY" or p2.get("acceptance_status") not in {
        "P2_PASS", "P2_PARTIAL_PASS",
    }:
        raise RuntimeError(
            f"{consumer} requires a completed P2 preflight; failed/invalid harnesses remain blocked"
        )
    return p2.get("acceptance_status") == "P2_PARTIAL_PASS"


def _compare_cmos_maps(left: np.ndarray, right: np.ndarray, label: int,
                       input_energy: float, phase_energy: float,
                       config: dict[str, Any], roi_weights: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    right_total = float(right.sum(dtype=np.float64))
    left_total = float(left.sum(dtype=np.float64))
    left_metrics = model.measure_cmos(left, label, input_energy, phase_energy, config, roi_weights)
    right_metrics = model.measure_cmos(right, label, input_energy, phase_energy, config, roi_weights)
    roi_left = np.asarray(left_metrics["roi_scores"], dtype=np.float64)
    roi_right = np.asarray(right_metrics["roi_scores"], dtype=np.float64)
    roi_denominator = np.maximum(roi_right, max(float(np.mean(roi_right)) * 1e-12, 1e-30))
    roi_absolute_fraction = float(
        np.max(np.abs(roi_left - roi_right)) / max(right_total, 1e-30)
    )
    return {
        "intensity_relative_l1": float(np.abs(left - right).sum(dtype=np.float64) / max(right_total, 1e-30)),
        "total_energy_relative_difference": float(abs(left_total - right_total) / max(right_total, 1e-30)),
        "maximum_roi_score_relative_difference": float(np.max(np.abs(roi_left - roi_right) / roi_denominator)),
        "maximum_roi_score_difference_fraction_of_cmos_energy": roi_absolute_fraction,
        "prediction_left": left_metrics["prediction"],
        "prediction_right": right_metrics["prediction"],
        "prediction_agrees": bool(left_metrics["prediction"] == right_metrics["prediction"]),
    }


def _sampling_convergence(config: dict[str, Any], data: dict[str, Any], phase: np.ndarray,
                          selected_rows: np.ndarray, scenario: dict[str, Any],
                          device: torch.device, roi_weights: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    tolerance = float(config["numerics"]["sampling_relative_tolerance"])
    plan_cache: dict = {}
    input_pairs = list(zip(
        config["numerics"]["input_lens_grids_for_convergence"][:-1],
        config["numerics"]["input_lens_grids_for_convergence"][1:],
        strict=True,
    ))
    output_pairs = list(zip(
        config["numerics"]["output_lens_grids_for_convergence"][:-1],
        config["numerics"]["output_lens_grids_for_convergence"][1:],
        strict=True,
    ))
    common_output_grid = int(config["numerics"]["output_lens_grids_for_convergence"][-1])
    camera_factors = [int(value) for value in config["numerics"]["cmos_oversample_factors_for_convergence"]]
    if not camera_factors or camera_factors[0] < 1 or any(
        right <= left for left, right in zip(camera_factors, camera_factors[1:])
    ):
        raise ValueError("CMOS oversample factors must be a non-empty strictly increasing sequence")
    convergence_cmos_factor = camera_factors[-1]

    def compare_input_grids(n_low: int, n_high: int) -> list[dict[str, Any]]:
        records = []
        for row in selected_rows.tolist():
            image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
            low = model.propagate_dmd_to_phase(image, config, device=device, model_level=1,
                                               mode="static_binary", plan_cache=plan_cache,
                                               override_lens_grid=n_low)
            high = model.propagate_dmd_to_phase(image, config, device=device, model_level=1,
                                                mode="static_binary", plan_cache=plan_cache,
                                                override_lens_grid=n_high)
            low_phase = low["phase_field"][0].detach().cpu().numpy()
            high_phase = high["phase_field"][0].detach().cpu().numpy()
            overlap = np.vdot(high_phase.astype(np.complex128), low_phase.astype(np.complex128))
            global_phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0 + 0.0j
            phase_field_rel_l2 = float(
                np.linalg.norm(low_phase - global_phase * high_phase)
                / max(np.linalg.norm(high_phase), 1e-30)
            )
            low_phase_energy = float(low["phase_plane_energy"][0].detach().cpu())
            high_phase_energy = float(high["phase_plane_energy"][0].detach().cpu())
            phase_energy_rel = abs(low_phase_energy - high_phase_energy) / max(high_phase_energy, 1e-30)
            records.append({
                "validation_position": int(row),
                "input_grid_pair": [n_low, n_high],
                "phase_field_relative_l2": phase_field_rel_l2,
                "best_fit_global_phase_difference_rad": float(np.angle(global_phase)),
                "phase_plane_energy_relative_difference": float(phase_energy_rel),
            })
        return records

    def input_records_pass(records: list[dict[str, Any]]) -> bool:
        return all(
            row["phase_field_relative_l2"] <= tolerance
            and row["phase_plane_energy_relative_difference"] <= tolerance
            for row in records
        )

    chosen_input = int(config["numerics"]["input_lens_grids_for_convergence"][-1])
    input_comparisons = []
    input_pass = False
    for n_low, n_high in input_pairs:
        records = compare_input_grids(int(n_low), int(n_high))
        passed = input_records_pass(records)
        input_comparisons.append({
            "grid_pair": [int(n_low), int(n_high)],
            "rows": records,
            "passed": passed,
        })
        chosen_input = int(n_high)
        if passed:
            input_pass = True
            break

    chosen_output = int(config["numerics"]["output_lens_grids_for_convergence"][-1])
    output_comparisons = []
    output_pass = False
    input_fields = []
    for row in selected_rows.tolist():
        image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
        input_fields.append(model.propagate_dmd_to_phase(
            image, config, device=device, model_level=1, mode="static_binary",
            plan_cache=plan_cache, override_lens_grid=chosen_input,
        ))

    for n_low, n_high in output_pairs:
        records = []
        for row, input_field in zip(selected_rows.tolist(), input_fields, strict=True):
            values = {}
            for out_grid in (int(n_low), int(n_high)):
                output = model.output_cmos_field(
                    input_field["phase_field"], phase, config, scenario,
                    plan_cache=plan_cache, output_lens_grid=out_grid,
                    cmos_oversample=convergence_cmos_factor,
                )
                values[out_grid] = output["cmos_intensity"][0].detach().cpu().numpy()
            comparison = _compare_cmos_maps(
                values[int(n_low)], values[int(n_high)], int(data["labels"][row]),
                float(input_field["dmd_input_energy"][0].detach().cpu()),
                float(input_field["phase_plane_energy"][0].detach().cpu()),
                config, roi_weights,
            )
            records.append({
                "validation_position": int(row),
                "output_grid_pair": [int(n_low), int(n_high)],
                **comparison,
            })
        passed = all(
            row["intensity_relative_l1"] <= tolerance
            and row["maximum_roi_score_difference_fraction_of_cmos_energy"] <= tolerance
            and row["prediction_agrees"]
            for row in records
        )
        output_comparisons.append({
            "grid_pair": [int(n_low), int(n_high)],
            "rows": records,
            "passed": passed,
        })
        chosen_output = int(n_high)
        if passed:
            output_pass = True
            break

    camera_comparisons = []
    camera_pass = False
    chosen_cmos_oversample = convergence_cmos_factor
    output_grid_for_camera = chosen_output
    for factor_low, factor_high in zip(camera_factors[:-1], camera_factors[1:]):
        records = []
        for row, input_field in zip(selected_rows.tolist(), input_fields, strict=True):
            values = {}
            for factor in (factor_low, factor_high):
                output = model.output_cmos_field(
                    input_field["phase_field"], phase, config, scenario,
                    plan_cache=plan_cache, output_lens_grid=output_grid_for_camera,
                    cmos_oversample=factor,
                )
                values[factor] = output["cmos_intensity"][0].detach().cpu().numpy()
            comparison = _compare_cmos_maps(
                values[factor_low], values[factor_high], int(data["labels"][row]),
                float(input_field["dmd_input_energy"][0].detach().cpu()),
                float(input_field["phase_plane_energy"][0].detach().cpu()),
                config, roi_weights,
            )
            records.append({
                "validation_position": int(row),
                "cmos_oversample_pair": [factor_low, factor_high],
                "output_lens_grid": output_grid_for_camera,
                **comparison,
            })
        passed = all(
            item["intensity_relative_l1"] <= tolerance
            and item["maximum_roi_score_difference_fraction_of_cmos_energy"] <= tolerance
            and item["prediction_agrees"]
            for item in records
        )
        camera_comparisons.append({
            "factor_pair": [factor_low, factor_high],
            "rows": records,
            "passed": passed,
        })
        chosen_cmos_oversample = factor_high
        if passed:
            camera_pass = True
            break

    input_rows = input_comparisons[-1]["rows"]
    output_rows = output_comparisons[-1]["rows"]
    return {
        "tolerance": tolerance,
        "selected_input_lens_grid": chosen_input,
        "selected_output_lens_grid": chosen_output,
        "selected_cmos_oversample": chosen_cmos_oversample,
        "input_grid_comparisons": input_comparisons,
        "output_grid_comparisons": output_comparisons,
        "cmos_oversample_comparisons": camera_comparisons,
        "maximum_observed_input_phase_field_relative_l2": max(row["phase_field_relative_l2"] for row in input_rows),
        "maximum_observed_input_phase_energy_relative_difference": max(row["phase_plane_energy_relative_difference"] for row in input_rows),
        "maximum_observed_output_roi_energy_fraction_difference": max(row["maximum_roi_score_difference_fraction_of_cmos_energy"] for row in output_rows),
        "input_grid_converged": bool(input_pass),
        "output_grid_converged": bool(output_pass),
        "cmos_pixel_integration_converged": bool(camera_pass),
        "status": "PASS" if input_pass and output_pass and camera_pass else "PARTIAL_NOT_CONVERGED_AT_MAX_GRID",
    }


def run_preflight(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = model.read_json(config_path)
    data = model.load_c0_validation(config)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the production preflight but is unavailable")
    device = torch.device(device_name)
    p2_dir = run_dir / "p2"
    p2_dir.mkdir(parents=True, exist_ok=True)
    reference = _preflight_small_reference(device)
    candidates = model.candidate_metadata(config)
    phase = model.phase_array(candidates[0])
    selected = _rows_first_per_class(data, int(config["numerics"]["preflight_rows_per_class"]))
    one_per_class = _rows_first_per_class(data, 1)
    roi_weights = model.fixed_sensor_roi_weights(config)
    scenario = config["output_relay"]["scenarios"]["PHOTO_ESTIMATE_AND_C37_C38_REFERENCE"]
    plan_cache: dict = {}
    sampling_started = time.perf_counter()
    sampling = _sampling_convergence(config, data, phase, one_per_class, scenario, device, roi_weights)
    sampling["elapsed_seconds_observed"] = time.perf_counter() - sampling_started
    selected_input_grid = int(sampling["selected_input_lens_grid"])
    selected_output_grid = int(sampling["selected_output_lens_grid"])
    selected_cmos_oversample = int(sampling["selected_cmos_oversample"])
    production_rows = []
    production_started = time.perf_counter()
    all_planes_finite = True
    representative_planes: dict[str, np.ndarray] | None = None
    plane_inventory: dict[str, dict[str, Any]] | None = None
    condition_defs = (
        ("binary_level1", selected, "static_binary", 1),
        ("binary_level0", one_per_class, "static_binary", 0),
        ("gray_effective_level1", one_per_class, "gray_effective_amplitude", 1),
    )
    for condition, rows, mode, level in condition_defs:
        for sample_row in rows.tolist():
            image = torch.as_tensor(data["images"][sample_row:sample_row + 1], device=device)
            forward = model.forward_field(image, phase, config, scenario, device=device,
                                          model_level=level, mode=mode, plan_cache=plan_cache,
                                          input_lens_grid=selected_input_grid,
                                          output_lens_grid=selected_output_grid,
                                          cmos_oversample=selected_cmos_oversample)
            tensors_finite = all(bool(torch.isfinite(value).all().item()) for value in forward.values()
                                 if isinstance(value, torch.Tensor) and value.dtype != torch.bool)
            all_planes_finite &= tensors_finite
            intensity = forward["cmos_intensity"][0].detach().cpu().numpy()
            measurement = model.measure_cmos(
                intensity, int(data["labels"][sample_row]),
                float(forward["dmd_input_energy"][0].detach().cpu()),
                float(forward["phase_plane_energy"][0].detach().cpu()), config, roi_weights,
            )
            production_rows.append({
                "condition": condition,
                "dmd_model_level": level,
                "input_encoding": mode,
                "validation_position": int(sample_row),
                "original_index": int(data["indices"][sample_row]),
                "label": int(data["labels"][sample_row]),
                "finite_all_named_planes": tensors_finite,
                "measurement": measurement,
                "energy_ledger_model_units_um2": {
                    key: float(forward[key][0].detach().cpu())
                    for key in ("dmd_input_energy", "l2_before_aperture_energy", "l2_after_aperture_energy",
                                "l1_before_aperture_energy", "l1_after_aperture_energy", "phase_plane_energy",
                                "output_lens_energy_before_pupil", "output_lens_energy_after_pupil", "cmos_energy")
                },
            })
            if representative_planes is None and condition == "binary_level1":
                plane_inventory = {
                    key: {"shape_batch_inclusive": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in forward.items()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2
                }
                representative_planes = {}
                retained_keys = {"dmd_field", "phase_field", "phase_field_after_screen",
                                 "cmos_intensity", "output_aperture_mask"}
                for key in retained_keys:
                    value = forward[key][0].detach()
                    if key == "cmos_intensity":
                        representative_planes[key] = value.cpu().numpy()
                    else:
                        representative_planes[key] = value.cpu().numpy()
                camera_field = forward["cmos_field"][0].detach()
                sensor_rows, sensor_columns = config["cmos"]["sensor_shape_rows_columns"]
                camera_factor = selected_cmos_oversample
                camera_window = config["fixed_roi_geometry"]["physical_window_um"]
                pitch_subpixel = float(config["cmos"]["pixel_pitch_um"]) / camera_factor
                crop_rows = int(math.ceil(float(camera_window[1]) / pitch_subpixel))
                crop_columns = int(math.ceil(float(camera_window[0]) / pitch_subpixel))
                center_row, center_column = camera_field.shape[-2] // 2, camera_field.shape[-1] // 2
                r0, c0 = center_row - crop_rows // 2, center_column - crop_columns // 2
                representative_planes["cmos_field_center_fixed_window"] = camera_field[
                    r0:r0 + crop_rows, c0:c0 + crop_columns
                ].cpu().numpy()
                representative_planes["cmos_field_center_window_metadata"] = np.asarray(
                    [sensor_rows, sensor_columns, camera_factor, pitch_subpixel,
                     r0, c0, crop_rows, crop_columns], dtype=np.float64,
                )
    elapsed = time.perf_counter() - production_started
    if not all(row["finite_all_named_planes"] for row in production_rows):
        raise FloatingPointError("production forward emitted a nonfinite CMOS map")
    # CPU/GPU implementation comparison on a reduced optical-grid diagnostic.
    cpu_gpu: dict[str, Any] = {"status": "NOT_RUN"}
    if device.type == "cuda":
        sample = torch.as_tensor(data["images"][:1], dtype=torch.uint8)
        cpu_cache: dict = {}
        gpu_cache: dict = {}
        cpu_out = model.forward_field(
            sample, phase, config, scenario, device="cpu", model_level=1,
            input_lens_grid=32, output_lens_grid=32, plan_cache=cpu_cache,
        )
        gpu_out = model.forward_field(
            sample, phase, config, scenario, device=device, model_level=1,
            input_lens_grid=32, output_lens_grid=32, plan_cache=gpu_cache,
        )
        a = cpu_out["cmos_field"].detach().cpu().numpy()
        b = gpu_out["cmos_field"].detach().cpu().numpy()
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        cpu_gpu = {"field_relative_l2": rel, "tolerance": 1e-4,
                   "status": "PASS" if rel <= 1e-4 else "FAIL"}
    expected_count = len(selected) + 2 * len(one_per_class)
    preflight_core_pass = (reference["status"] == "PASS" and len(production_rows) == expected_count
                           and all_planes_finite and cpu_gpu.get("status") == "PASS")
    p2_acceptance = (
        "P2_PASS" if preflight_core_pass and sampling.get("status") == "PASS"
        else "P2_PARTIAL_PASS" if preflight_core_pass
        else "P2_INVALID_HARNESS"
    )
    result = {
        "status": "PREFLIGHT_READY" if preflight_core_pass else "PREFLIGHT_FAILED",
        "acceptance_status": p2_acceptance,
        "stage": "P1_P2_NO_UPDATE_PRODUCTION_PREFLIGHT",
        "device": str(device),
        "production_forward": {
            "candidate_id": candidates[0]["candidate_id"],
            "dmd_mode": "static_binary",
            "dmd_model_level": 1,
            "output_geometry": "PHOTO_ESTIMATE_AND_C37_C38_REFERENCE",
            "selected_input_lens_grid": selected_input_grid,
            "selected_output_lens_grid": selected_output_grid,
            "selected_cmos_oversample": selected_cmos_oversample,
            "rows": production_rows,
            "row_count": len(production_rows),
            "condition_counts": {key: sum(row["condition"] == key for row in production_rows)
                                 for key, _, _, _ in condition_defs},
            "elapsed_seconds_observed": elapsed,
            "all_named_plane_outputs_finite": bool(all_planes_finite),
            "all_roi_integrations_completed": True,
            "forward_api": "hw207_physical_frozen_phase_v1.forward_field",
            "named_plane_inventory": plane_inventory,
        },
        "independent_reference": reference,
        "sampling_convergence": sampling,
        "sampling_convergence_is_required_for_claims": True,
        "sampling_partial_allows_only_labeled_diagnostic_continuation": True,
        "cpu_gpu_consistency": cpu_gpu,
        "parameter_updates": 0,
        "negative_controls": {"fresnel_sign": reference["wrong_sign_counterexample_rejected"]},
        "official_test_opened": False,
        "status_scope": "engineering preflight only; no scientific candidate selection",
    }
    if representative_planes is not None:
        _save_npz(p2_dir / "representative_plane_fields.npz", **representative_planes)
    _json_once(p2_dir / "production_preflight.json", result)
    return result


def _build_input_cache(config: dict[str, Any], data: dict[str, Any], run_dir: Path,
                       device: torch.device, input_lens_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = run_dir / "p4" / "input_phase_fields_level1_binary.npy"
    energy_path = run_dir / "p4" / "input_dmd_energy_level1_binary.npy"
    phase_energy_path = run_dir / "p4" / "input_phase_plane_energy_level1_binary.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() or energy_path.exists() or phase_energy_path.exists():
        raise FileExistsError("refusing to overwrite the frozen input-field cache")
    n = len(data["indices"])
    field_mm = np.lib.format.open_memmap(cache_path.with_suffix(".tmp.npy"), mode="w+",
                                         dtype=np.complex64, shape=(n, 100, 100))
    energy_mm = np.lib.format.open_memmap(energy_path.with_suffix(".tmp.npy"), mode="w+",
                                          dtype=np.float32, shape=(n,))
    phase_energy_mm = np.lib.format.open_memmap(phase_energy_path.with_suffix(".tmp.npy"), mode="w+",
                                                dtype=np.float32, shape=(n,))
    batch_size = 2
    plan_cache: dict = {}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            images = torch.as_tensor(data["images"][start:stop], dtype=torch.uint8, device=device)
            forward = model.propagate_dmd_to_phase(
                images, config, device=device, model_level=1, mode="static_binary",
                plan_cache=plan_cache, override_lens_grid=input_lens_grid,
            )
            field_mm[start:stop] = forward["phase_field"].detach().cpu().numpy().astype(np.complex64)
            energy_mm[start:stop] = forward["dmd_input_energy"].detach().cpu().numpy().astype(np.float32)
            phase_energy_mm[start:stop] = forward["phase_plane_energy"].detach().cpu().numpy().astype(np.float32)
    field_mm.flush()
    energy_mm.flush()
    phase_energy_mm.flush()
    del field_mm, energy_mm, phase_energy_mm
    Path(str(cache_path.with_suffix(".tmp.npy"))).replace(cache_path)
    Path(str(energy_path.with_suffix(".tmp.npy"))).replace(energy_path)
    Path(str(phase_energy_path.with_suffix(".tmp.npy"))).replace(phase_energy_path)
    return (np.load(cache_path, mmap_mode="r"), np.load(energy_path, mmap_mode="r"),
            np.load(phase_energy_path, mmap_mode="r"))


def run_frozen_inference(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = model.read_json(config_path)
    p0_path = run_dir / "p0" / "p0_asset_and_data_audit.json"
    p2_path = run_dir / "p2" / "production_preflight.json"
    if not p0_path.is_file() or not p2_path.is_file():
        raise RuntimeError("P0 audit and production P2 preflight must complete before frozen inference")
    p2 = model.read_json(p2_path)
    sampling_limited = _sampling_admission(p2, "frozen inference")
    selected_input_grid = int(p2["sampling_convergence"]["selected_input_lens_grid"])
    selected_output_grid = int(p2["sampling_convergence"]["selected_output_lens_grid"])
    selected_cmos_oversample = int(p2["sampling_convergence"]["selected_cmos_oversample"])
    data = model.load_c0_validation(config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for full frozen inference but is unavailable")
    candidates = model.candidate_metadata(config)
    roi_weights = model.fixed_sensor_roi_weights(config)
    mapping_matrix, mapping_offset = model.cmos_pixel_to_window_affine(config)
    roi_dir = run_dir / "p4"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_arrays: dict[str, np.ndarray] = {
        "pixel_to_window_matrix_um_per_pixel": mapping_matrix,
        "pixel_to_window_offset_um": mapping_offset,
    }
    roi_contracts = []
    for item in roi_weights:
        roi_id = int(item["roi"])
        roi_arrays[f"roi_{roi_id}_rows"] = item["rows"]
        roi_arrays[f"roi_{roi_id}_columns"] = item["columns"]
        roi_arrays[f"roi_{roi_id}_area_fractions"] = item["fractions"]
        roi_contracts.append({
            "roi": roi_id,
            "roi_area_um2": float(item["roi_area_um2"]),
            "represented_area_um2": float(np.sum(item["fractions"], dtype=np.float64)
                                          * float(item["pixel_area_um2"])),
            "candidate_pixel_count": int(item["rows"].size),
        })
    _save_npz(roi_dir / "fixed_cmos_roi_pixel_weights.npz", **roi_arrays)
    _json_once(roi_dir / "cmos_physical_mapping_and_calibration_interface.json", {
        "sensor_shape_rows_columns": config["cmos"]["sensor_shape_rows_columns"],
        "pixel_pitch_um_specification": config["cmos"]["pixel_pitch_um"],
        "pixel_pitch_status": "USER_SPEC_IMAGE_OR_MANUFACTURER_SPEC; camera unit identity UNKNOWN",
        "pixel_to_window_matrix_um_per_pixel": mapping_matrix.tolist(),
        "pixel_to_window_offset_um": mapping_offset.tolist(),
        "affine_status": config["cmos"]["pixel_to_window_affine"]["status"],
        "affine_support": "general invertible affine supports scale, rotation, x/y flips, and subpixel offset; current values are assumed diagnostic",
        "pixel_integral": (
            "CMOS subpixel irradiance is area-averaged over each physical pixel at the selected oversampling; "
            "ROI boundaries then use exact affine pixel-area overlap. Full-frame convergence remains separately reported."
        ),
        "fixed_roi_area_audit": roi_contracts,
        "dark_flat_gain_exposure_interface": {
            "function": "hw207_physical_frozen_phase_v1.apply_cmos_calibration",
            "configured_inputs": config["cmos"]["calibration_interface"],
            "calibration_applied_to_this_run": False,
        },
        "absolute_photoelectron_or_count_calibration": "UNKNOWN_NOT_MODELED",
    })
    scenario = config["output_relay"]["scenarios"]["PHOTO_ESTIMATE_AND_C37_C38_REFERENCE"]
    cache_start = time.perf_counter()
    input_fields, input_energies, phase_energies = _build_input_cache(
        config, data, run_dir, device, selected_input_grid,
    )
    cache_seconds = time.perf_counter() - cache_start
    dmd_examples = _rows_first_per_class(data, 1)
    native_dmd = np.stack([
        model.native_dmd_pattern(data["images"][row], mode="static_binary")
        for row in dmd_examples
    ])
    _save_npz(
        run_dir / "p4" / "native_dmd_representative_commands.npz",
        original_indices=data["indices"][dmd_examples], labels=data["labels"][dmd_examples],
        commands_uint8=native_dmd,
    )
    all_candidate_summaries = []
    representative_records: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        candidate_dir = run_dir / "p5" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        phase = model.phase_array(candidate)
        n = len(data["indices"])
        predictions = np.full(n, -1, dtype=np.int8)
        processed_indices = np.empty(n, dtype=np.int64)
        processed_labels = np.empty(n, dtype=np.int64)
        scores = np.zeros((n, 4), dtype=np.float64)
        cmos_energy = np.zeros(n, dtype=np.float64)
        dmd_efficiency = np.zeros(n, dtype=np.float64)
        phase_efficiency = np.zeros(n, dtype=np.float64)
        outside_energy = np.zeros(n, dtype=np.float64)
        correct_energy = np.zeros(n, dtype=np.float64)
        strongest_wrong = np.zeros(n, dtype=np.float64)
        peak_rc = np.zeros((n, 2), dtype=np.int32)
        centroid_rc = np.full((n, 2), np.nan, dtype=np.float64)
        peak_correct_roi = np.zeros(n, dtype=np.uint8)
        centroid_correct_roi = np.zeros(n, dtype=np.uint8)
        roi_fractions = np.zeros((n, 4), dtype=np.float32)
        class_sums = np.zeros((4, 2048, 2048), dtype=np.float32)
        class_counts = np.zeros(4, dtype=np.int64)
        representative_images = np.zeros((4, 2048, 2048), dtype=np.float32)
        window_half = math.ceil(
            float(config["fixed_roi_geometry"]["physical_window_um"][0])
            / (2.0 * float(config["cmos"]["pixel_pitch_um"]))
        )
        center_row, center_col = (
            int(v) // 2 for v in config["cmos"]["sensor_shape_rows_columns"]
        )
        window_bounds = (
            center_row - window_half, center_row + window_half,
            center_col - window_half, center_col + window_half,
        )
        misclassified_window_records: dict[int, dict[str, Any]] = {}
        representative_positions = _rows_first_per_class(data, 1)
        representative_lookup = {int(row): int(label) for row, label in
                                 zip(representative_positions, range(4))}
        plan_cache: dict = {}
        started = time.perf_counter()
        with torch.no_grad():
            for row in range(n):
                image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
                cached = {
                    "phase_field": torch.as_tensor(input_fields[row:row + 1], device=device),
                    "dmd_input_energy": torch.as_tensor([input_energies[row]], dtype=torch.float32, device=device),
                    "phase_plane_energy": torch.as_tensor([phase_energies[row]], dtype=torch.float32, device=device),
                }
                forward = model.forward_field(
                    image, phase, config, scenario, device=device,
                    model_level=1, mode="static_binary", plan_cache=plan_cache,
                    input_relay_cache=cached, output_lens_grid=selected_output_grid,
                    cmos_oversample=selected_cmos_oversample,
                )
                intensity = forward["cmos_intensity"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                label = int(data["labels"][row])
                processed_indices[row] = int(data["indices"][row])
                processed_labels[row] = label
                measurement = model.measure_cmos(intensity, label, float(input_energies[row]),
                                                 float(phase_energies[row]),
                                                 config, roi_weights)
                predictions[row] = int(measurement["prediction"])
                scores[row] = np.asarray(measurement["roi_scores"], dtype=np.float64)
                cmos_energy[row] = measurement["cmos_energy_model_units_um2"]
                dmd_efficiency[row] = measurement["dmd_to_cmos_efficiency"] or 0.0
                phase_efficiency[row] = measurement["phase_to_cmos_efficiency"] or 0.0
                outside_energy[row] = measurement["outside_four_roi_energy_model_units_um2"]
                correct_energy[row] = measurement["correct_roi_energy_model_units_um2"]
                strongest_wrong[row] = measurement["strongest_wrong_roi_energy_model_units_um2"]
                peak_rc[row] = measurement["peak_row_col"]
                if measurement["centroid_row_col"][0] is not None:
                    centroid_rc[row] = measurement["centroid_row_col"]
                peak_correct_roi[row] = int(measurement["peak_in_correct_roi"])
                centroid_correct_roi[row] = int(measurement["centroid_in_correct_roi"])
                roi_fractions[row] = measurement["roi_fractions_of_cmos"]
                class_sums[label] += intensity
                class_counts[label] += 1
                if row in representative_lookup:
                    representative_images[representative_lookup[row]] = intensity
                    representative_records[str(label)] = {
                        "validation_position": row,
                        "original_index": int(data["indices"][row]),
                        "true_label": label,
                        **measurement,
                    }
                if int(measurement["prediction"]) != label and label not in misclassified_window_records:
                    r0, r1, c0, c1 = window_bounds
                    misclassified_window_records[label] = {
                        "validation_position": row,
                        "original_index": int(data["indices"][row]),
                        "true_label": label,
                        "prediction": int(measurement["prediction"]),
                        "roi_scores": measurement["roi_scores"],
                        "dmd_image": data["images"][row].copy(),
                        "cmos_window_intensity": intensity[r0:r1, c0:c1].copy(),
                        "peak_xy_window_um": measurement["peak_x_y_in_window_um"],
                        "centroid_xy_window_um": measurement["centroid_x_y_in_window_um"],
                        "dmd_to_cmos_efficiency": measurement["dmd_to_cmos_efficiency"],
                    }
                if (row + 1) % 100 == 0:
                    print(f"{candidate_id}: {row + 1}/{n}", flush=True)
        elapsed = time.perf_counter() - started
        class_means = class_sums / class_counts[:, None, None]
        mean_path = candidate_dir / "class_mean_cmos_float32.npy"
        if mean_path.exists():
            raise FileExistsError(mean_path)
        np.save(mean_path, class_means.astype(np.float32), allow_pickle=False)
        reps_path = candidate_dir / "representative_cmos_float32.npy"
        if reps_path.exists():
            raise FileExistsError(reps_path)
        np.save(reps_path, representative_images, allow_pickle=False)
        _save_npz(
            candidate_dir / "validation_rows.npz",
            original_indices=processed_indices, true_labels=processed_labels, predictions=predictions,
            roi_scores_model_units_um2=scores, roi_fractions_of_cmos=roi_fractions,
            cmos_energy_model_units_um2=cmos_energy, input_dmd_energy_model_units_um2=input_energies,
            dmd_to_cmos_efficiency=dmd_efficiency,
            phase_to_cmos_efficiency=phase_efficiency,
            outside_roi_energy_model_units_um2=outside_energy,
            correct_roi_energy_model_units_um2=correct_energy,
            strongest_wrong_roi_energy_model_units_um2=strongest_wrong,
            peak_row_col=peak_rc, centroid_row_col=centroid_rc,
            peak_in_correct_roi=peak_correct_roi, centroid_in_correct_roi=centroid_correct_roi,
        )
        error_rows = [misclassified_window_records[label]
                      for label in sorted(misclassified_window_records)]
        _save_npz(
            candidate_dir / "misclassified_examples_200um_windows.npz",
            validation_positions=np.asarray([row["validation_position"] for row in error_rows], dtype=np.int32),
            original_indices=np.asarray([row["original_index"] for row in error_rows], dtype=np.int64),
            true_labels=np.asarray([row["true_label"] for row in error_rows], dtype=np.int8),
            predictions=np.asarray([row["prediction"] for row in error_rows], dtype=np.int8),
            roi_scores_model_units_um2=np.asarray([row["roi_scores"] for row in error_rows], dtype=np.float64).reshape(-1, 4),
            dmd_images=np.asarray([row["dmd_image"] for row in error_rows], dtype=np.uint8).reshape(-1, 28, 28),
            cmos_window_intensity=np.asarray(
                [row["cmos_window_intensity"] for row in error_rows], dtype=np.float32
            ).reshape(-1, 2 * window_half, 2 * window_half),
            peak_xy_window_um=np.asarray([row["peak_xy_window_um"] for row in error_rows], dtype=np.float64).reshape(-1, 2),
            centroid_xy_window_um=np.asarray([row["centroid_xy_window_um"] for row in error_rows], dtype=np.float64).reshape(-1, 2),
            dmd_to_cmos_efficiency=np.asarray([row["dmd_to_cmos_efficiency"] for row in error_rows], dtype=np.float64),
        )
        _json_once(candidate_dir / "representative_records.json", {
            "candidate_id": candidate_id,
            "selection": "first C0 validation row of each true class, in original index order",
            "records_by_true_class": representative_records,
            "absolute_cmos_intensity_units": "model arbitrary units; not calibrated counts",
            "phase_display_rule": "source array unchanged; only visualization may show phase modulo 2pi",
        })
        order_verified = bool(np.array_equal(processed_indices, data["indices"])
                              and np.array_equal(processed_labels, data["labels"]))
        if not order_verified:
            raise ValueError(f"processed row order differs from the frozen C0 validation order: {candidate_id}")
        metrics = cmos.fixed_roi_classification_metrics(processed_labels, predictions)
        cm = np.asarray(metrics["confusion_matrix_true_rows_predicted_columns"], dtype=np.int64)
        summary = {
            "candidate_id": candidate_id,
            "source_type": candidate["source_type"],
            "source_phase_path": candidate["phase_path"],
            "sample_count": int(n),
            "selected_numerical_grids": {
                "input_lens_grid": selected_input_grid,
                "output_lens_grid": selected_output_grid,
                "cmos_oversample": selected_cmos_oversample,
            },
            "rows_and_labels_order_verified": order_verified,
            "classification": metrics,
            "confusion_matrix_true_rows_predicted_columns": cm.tolist(),
            "mean_dmd_to_cmos_efficiency": float(np.mean(dmd_efficiency)),
            "p10_dmd_to_cmos_efficiency": float(np.quantile(dmd_efficiency, 0.1)),
            "minimum_dmd_to_cmos_efficiency": float(np.min(dmd_efficiency)),
            "mean_phase_to_cmos_efficiency": float(np.mean(phase_efficiency)),
            "p10_phase_to_cmos_efficiency": float(np.quantile(phase_efficiency, 0.1)),
            "minimum_phase_to_cmos_efficiency": float(np.min(phase_efficiency)),
            "mean_correct_roi_fraction_of_cmos": float(np.mean(scores[np.arange(n), data["labels"]] / np.maximum(cmos_energy, 1e-30))),
            "mean_correct_roi_energy_model_units_um2": float(np.mean(correct_energy)),
            "mean_strongest_wrong_roi_energy_model_units_um2": float(np.mean(strongest_wrong)),
            "mean_correct_to_strongest_wrong_roi_ratio": float(np.mean(correct_energy / np.maximum(strongest_wrong, 1e-30))),
            "mean_outside_four_roi_fraction_of_cmos": float(np.mean(outside_energy / np.maximum(cmos_energy, 1e-30))),
            "peak_localization_accuracy": float(np.mean(peak_correct_roi)),
            "centroid_localization_accuracy": float(np.mean(centroid_correct_roi)),
            "unresolved_count": int(metrics["unresolved_count"]),
            "elapsed_seconds_observed": elapsed,
            "same_fixed_physical_roi_mapping": True,
            "numeric_status": "ASSUMED_DIAGNOSTIC_ONLY",
        "sampling_limited": sampling_limited,
        "sampling_limitation": (
            "P2 completed but output-lens and/or CMOS full-frame sampling did not meet convergence tolerance; "
            "results are conditional on the recorded maximum grid"
            if sampling_limited else None
        ),
        }
        _json_once(candidate_dir / "candidate_summary.json", summary)
        all_candidate_summaries.append(summary)
    _save_npz(
        run_dir / "p4" / "input_field_cache_metadata.npz",
        original_indices=data["indices"], labels=data["labels"],
        input_dmd_energy_model_units_um2=np.asarray(input_energies),
        input_phase_plane_energy_model_units_um2=np.asarray(phase_energies),
    )
    overall = {
        "status": "P5_FROZEN_INFERENCE_COMPLETE",
        "acceptance_status": "P5_PASS",
        "completion_definition": "all seven frozen candidates processed over all 2,475 ordered C0 validation rows; no model updates",
        "evaluation_role": "reused C0 validation; not independent",
        "row_count": int(len(data["indices"])),
        "class_counts": data["class_counts"],
        "original_indices_order": "identical C0 validation 0..3 filtered order for each candidate",
        "candidate_count": len(all_candidate_summaries),
        "candidate_summaries": all_candidate_summaries,
        "primary_geometry": "PHOTO_ESTIMATE_AND_C37_C38_REFERENCE",
        "selected_input_lens_grid": selected_input_grid,
        "selected_output_lens_grid": selected_output_grid,
        "input": "static binary DMD Level 1; coherent square mirror aperture with pitch/fill factor and ideal ON collection assumption",
        "output_readout": "four fixed 50x50 um ROI integrations followed by argmax; no digital classifier",
        "official_test_opened": False,
        "input_cache_seconds_observed": cache_seconds,
        "phase_updates": 0,
        "hardware_operation": False,
        "physical_mapping_status": "HW207_CURRENT_MAPPING_BLOCKED",
        "p2_acceptance_status": p2.get("acceptance_status"),
        "sampling_convergence_status": p2["sampling_convergence"]["status"],
        "sampling_limited_diagnostic_only": sampling_limited,
        "candidate_ranking_allowed": not sampling_limited,
        "performance_interpretation": (
            "complete maximum-grid numerical diagnostic; values are conditional on the recorded discretization "
            "and do not establish a stable candidate ranking"
            if sampling_limited else "numerically converged under the registered P2 sampling criteria"
        ),
        "candidate_selection_on_reused_validation": False,
    }
    _json_once(run_dir / "p5" / "frozen_inference_summary.json", overall)
    return overall


def run_geometry_sensitivity(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = model.read_json(config_path)
    p2_path = run_dir / "p2" / "production_preflight.json"
    inference_path = run_dir / "p5" / "frozen_inference_summary.json"
    field_path = run_dir / "p4" / "input_phase_fields_level1_binary.npy"
    energy_path = run_dir / "p4" / "input_dmd_energy_level1_binary.npy"
    phase_energy_path = run_dir / "p4" / "input_phase_plane_energy_level1_binary.npy"
    if not all(path.is_file() for path in (p2_path, inference_path, field_path, energy_path, phase_energy_path)):
        raise RuntimeError("P0-P5 production evidence and input caches are required before P3 sensitivity")
    p2 = model.read_json(p2_path)
    sampling_limited = _sampling_admission(p2, "geometry sensitivity")
    selected_output_grid = int(p2["sampling_convergence"]["selected_output_lens_grid"])
    selected_cmos_oversample = int(p2["sampling_convergence"]["selected_cmos_oversample"])
    data = model.load_c0_validation(config)
    input_fields = np.load(field_path, mmap_mode="r")
    input_energies = np.load(energy_path, mmap_mode="r")
    phase_energies = np.load(phase_energy_path, mmap_mode="r")
    device = torch.device(device_name)
    candidates = model.candidate_metadata(config)
    selected = _rows_first_per_class(data, int(config["numerics"]["sensitivity_rows_per_class"]))
    roi_weights = model.fixed_sensor_roi_weights(config)
    p3_dir = run_dir / "p3"
    p3_dir.mkdir(parents=True, exist_ok=False)
    conditions: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    plan_cache: dict = {}
    scenarios = config["output_relay"]["scenarios"]
    for candidate in candidates:
        phase = model.phase_array(candidate)
        for scenario_name, scenario in scenarios.items():
            for row in selected.tolist():
                cached = {
                    "phase_field": torch.as_tensor(input_fields[row:row + 1], device=device),
                    "dmd_input_energy": torch.as_tensor([input_energies[row]], dtype=torch.float32, device=device),
                    "phase_plane_energy": torch.as_tensor([phase_energies[row]], dtype=torch.float32, device=device),
                }
                image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
                with torch.no_grad():
                    forward = model.forward_field(image, phase, config, scenario, device=device,
                                                  model_level=1, mode="static_binary",
                                                  input_relay_cache=cached, plan_cache=plan_cache,
                                                  output_lens_grid=selected_output_grid,
                                                  cmos_oversample=selected_cmos_oversample)
                intensity = forward["cmos_intensity"][0].detach().cpu().numpy()
                measurement = model.measure_cmos(intensity, int(data["labels"][row]),
                                                 float(input_energies[row]), float(phase_energies[row]),
                                                 config, roi_weights)
                all_rows.append({
                    "condition": f"{candidate['candidate_id']}::{scenario_name}",
                    "candidate_id": candidate["candidate_id"],
                    "scenario": scenario_name,
                    "validation_position": row,
                    "original_index": int(data["indices"][row]),
                    "true_label": int(data["labels"][row]),
                    "prediction": int(measurement["prediction"]),
                    "roi_scores": measurement["roi_scores"],
                    "dmd_to_cmos_efficiency": measurement["dmd_to_cmos_efficiency"],
                    "phase_to_cmos_efficiency": measurement["phase_to_cmos_efficiency"],
                    "peak_in_correct_roi": measurement["peak_in_correct_roi"],
                    "centroid_in_correct_roi": measurement["centroid_in_correct_roi"],
                    "correct_roi_fraction_of_cmos": measurement["roi_fractions_of_cmos"][int(data["labels"][row])],
                })
            records = [x for x in all_rows if x["candidate_id"] == candidate["candidate_id"]
                       and x["scenario"] == scenario_name]
            predictions = [x["prediction"] for x in records]
            labels = [x["true_label"] for x in records]
            class_metrics = cmos.fixed_roi_classification_metrics(labels, predictions)
            conditions.append({
                "candidate_id": candidate["candidate_id"],
                "scenario": scenario_name,
                "row_count": len(records),
                "classification": class_metrics,
                "mean_dmd_to_cmos_efficiency": float(np.mean([x["dmd_to_cmos_efficiency"] for x in records])),
                "mean_phase_to_cmos_efficiency": float(np.mean([x["phase_to_cmos_efficiency"] for x in records])),
                "peak_localization_accuracy": float(np.mean([x["peak_in_correct_roi"] for x in records])),
                "centroid_localization_accuracy": float(np.mean([x["centroid_in_correct_roi"] for x in records])),
                "mean_correct_roi_fraction_of_cmos": float(np.mean([x["correct_roi_fraction_of_cmos"] for x in records])),
                "geometry_status": scenario["status"],
            })
    # Aperture and defocus sensitivity: same frozen phase and rows, no candidate search.
    sensitivity_candidates = [candidates[0], candidates[-1]]
    primary = scenarios["PHOTO_ESTIMATE_AND_C37_C38_REFERENCE"]
    for candidate in sensitivity_candidates:
        phase = model.phase_array(candidate)
        for aperture_mm in config["output_relay"]["aperture_sensitivity_mm"]:
            for row in selected.tolist():
                cached = {
                    "phase_field": torch.as_tensor(input_fields[row:row + 1], device=device),
                    "dmd_input_energy": torch.as_tensor([input_energies[row]], dtype=torch.float32, device=device),
                    "phase_plane_energy": torch.as_tensor([phase_energies[row]], dtype=torch.float32, device=device),
                }
                image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
                with torch.no_grad():
                    forward = model.forward_field(image, phase, config, primary, device=device,
                                                  input_relay_cache=cached, plan_cache=plan_cache,
                                                  output_aperture_mm=float(aperture_mm),
                                                  output_lens_grid=selected_output_grid,
                                                  cmos_oversample=selected_cmos_oversample)
                intensity = forward["cmos_intensity"][0].detach().cpu().numpy()
                measurement = model.measure_cmos(intensity, int(data["labels"][row]),
                                                 float(input_energies[row]), float(phase_energies[row]),
                                                 config, roi_weights)
                all_rows.append({
                    "condition": f"{candidate['candidate_id']}::APERTURE_{aperture_mm:g}mm",
                    "candidate_id": candidate["candidate_id"],
                    "scenario": f"APERTURE_{aperture_mm:g}mm",
                    "validation_position": row,
                    "original_index": int(data["indices"][row]),
                    "true_label": int(data["labels"][row]),
                    "prediction": int(measurement["prediction"]),
                    "roi_scores": measurement["roi_scores"],
                    "dmd_to_cmos_efficiency": measurement["dmd_to_cmos_efficiency"],
                    "phase_to_cmos_efficiency": measurement["phase_to_cmos_efficiency"],
                    "peak_in_correct_roi": measurement["peak_in_correct_roi"],
                    "centroid_in_correct_roi": measurement["centroid_in_correct_roi"],
                    "correct_roi_fraction_of_cmos": measurement["roi_fractions_of_cmos"][int(data["labels"][row])],
                })
        for delta_mm in config["output_relay"]["distance_sensitivity_offsets_mm"]:
            if float(delta_mm) == 0.0:
                continue
            shifted = dict(primary)
            shifted["lens_to_cmos_mm"] = float(primary["lens_to_cmos_mm"]) + float(delta_mm)
            for row in selected.tolist():
                cached = {
                    "phase_field": torch.as_tensor(input_fields[row:row + 1], device=device),
                    "dmd_input_energy": torch.as_tensor([input_energies[row]], dtype=torch.float32, device=device),
                    "phase_plane_energy": torch.as_tensor([phase_energies[row]], dtype=torch.float32, device=device),
                }
                image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
                with torch.no_grad():
                    forward = model.forward_field(image, phase, config, shifted, device=device,
                                                  input_relay_cache=cached, plan_cache=plan_cache,
                                                  output_lens_grid=selected_output_grid,
                                                  cmos_oversample=selected_cmos_oversample)
                intensity = forward["cmos_intensity"][0].detach().cpu().numpy()
                measurement = model.measure_cmos(intensity, int(data["labels"][row]),
                                                 float(input_energies[row]), float(phase_energies[row]),
                                                 config, roi_weights)
                all_rows.append({
                    "condition": f"{candidate['candidate_id']}::CMOS_DISTANCE_OFFSET_{delta_mm:+g}mm",
                    "candidate_id": candidate["candidate_id"],
                    "scenario": f"CMOS_DISTANCE_OFFSET_{delta_mm:+g}mm",
                    "validation_position": row,
                    "original_index": int(data["indices"][row]),
                    "true_label": int(data["labels"][row]),
                    "prediction": int(measurement["prediction"]),
                    "roi_scores": measurement["roi_scores"],
                    "dmd_to_cmos_efficiency": measurement["dmd_to_cmos_efficiency"],
                    "phase_to_cmos_efficiency": measurement["phase_to_cmos_efficiency"],
                    "peak_in_correct_roi": measurement["peak_in_correct_roi"],
                    "centroid_in_correct_roi": measurement["centroid_in_correct_roi"],
                    "correct_roi_fraction_of_cmos": measurement["roi_fractions_of_cmos"][int(data["labels"][row])],
                })
    _save_npz(
        p3_dir / "geometry_sensitivity_rows.npz",
        candidate_id=np.asarray([x["candidate_id"] for x in all_rows], dtype="U64"),
        scenario=np.asarray([x["scenario"] for x in all_rows], dtype="U64"),
        validation_position=np.asarray([x["validation_position"] for x in all_rows], dtype=np.int32),
        original_index=np.asarray([x["original_index"] for x in all_rows], dtype=np.int64),
        true_label=np.asarray([x["true_label"] for x in all_rows], dtype=np.int8),
        prediction=np.asarray([x["prediction"] for x in all_rows], dtype=np.int8),
        roi_scores=np.asarray([x["roi_scores"] for x in all_rows], dtype=np.float64),
        dmd_to_cmos_efficiency=np.asarray([x["dmd_to_cmos_efficiency"] for x in all_rows], dtype=np.float64),
        phase_to_cmos_efficiency=np.asarray([x["phase_to_cmos_efficiency"] for x in all_rows], dtype=np.float64),
        peak_in_correct_roi=np.asarray([x["peak_in_correct_roi"] for x in all_rows], dtype=np.uint8),
        centroid_in_correct_roi=np.asarray([x["centroid_in_correct_roi"] for x in all_rows], dtype=np.uint8),
        correct_roi_fraction_of_cmos=np.asarray([x["correct_roi_fraction_of_cmos"] for x in all_rows], dtype=np.float64),
    )
    summary = {
        "status": "P3_SENSITIVITY_COMPLETE_DIAGNOSTIC",
        "acceptance_status": "P3_PARTIAL_PASS" if sampling_limited else "P3_PASS",
        "sampling_limited_diagnostic_only": sampling_limited,
        "candidate_ranking_allowed": not sampling_limited,
        "sampling_convergence_status": p2["sampling_convergence"]["status"],
        "selected_output_lens_grid": selected_output_grid,
        "rows_per_class_for_geometry_comparison": int(config["numerics"]["sensitivity_rows_per_class"]),
        "validation_positions": selected.tolist(),
        "original_indices": data["indices"][selected].tolist(),
        "full_frozen_candidate_output_geometries": conditions,
        "aperture_and_defocus_sensitivities": "saved row-level in geometry_sensitivity_rows.npz; fixed candidates C3.5 P1 and C3.8 T3",
        "not_a_measurement_uncertainty_interval": True,
        "not_independent_evaluation": True,
    }
    _json_once(p3_dir / "geometry_sensitivity_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p0", "preflight", "evaluate", "sensitivity"), required=True)
    parser.add_argument("--config", type=Path, default=model.DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "p0":
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            _json_once(manifest_path, _run_manifest(config_path, run_dir, args.stage, args.device))
        result = run_p0(config_path, run_dir)
    elif args.stage == "preflight":
        if not (run_dir / "p0" / "p0_asset_and_data_audit.json").is_file():
            raise RuntimeError("P0 asset and data audit must be present before preflight")
        result = run_preflight(config_path, run_dir, args.device)
    elif args.stage == "sensitivity":
        result = run_geometry_sensitivity(config_path, run_dir, args.device)
    else:
        result = run_frozen_inference(config_path, run_dir, args.device)
    print(json.dumps({"stage": args.stage, "status": result["status"], "run_dir": str(run_dir)}, ensure_ascii=False))
    accepted_statuses = {
        "P0_PASS_DIAGNOSTIC_ASSET_AUDIT", "PREFLIGHT_READY",
        "P5_FROZEN_INFERENCE_COMPLETE", "P3_SENSITIVITY_COMPLETE_DIAGNOSTIC",
    }
    return 0 if result["status"] in accepted_statuses else 2


if __name__ == "__main__":
    raise SystemExit(main())
