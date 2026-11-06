"""C2.1 spatial alignment and C3 interface diagnostics for HW207.

The executable chain ends at the modeled LightGen input plane.  It does not
produce a LightGen output or claim camera classification without a fixed,
traceable LightGen layer model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

import hw207_mnist4_c2_v1 as c2
from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geo


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _candidate_table(config: dict) -> list[dict]:
    candidates = []
    for item in config["relay_candidates"]:
        input_f1, input_f2 = item["input_focal_lengths_mm"]
        output_f1, output_f2 = item["output_focal_lengths_mm"]
        input_4f = geo.ideal_4f_segments(input_f1, input_f2)
        output_4f = geo.ideal_4f_segments(output_f1, output_f2)
        intended = item["intended_abs_magnifications"]
        calculated = [abs(input_4f["signed_magnification"]),
                      abs(output_4f["signed_magnification"])]
        if any(abs(actual - target) > 1e-9 for actual, target in zip(calculated, intended)):
            raise ValueError(f"Relay candidate focal lengths do not match intended magnification: {item['id']}")
        candidates.append({
            "id": item["id"], "input_focal_lengths_mm": [input_f1, input_f2],
            "output_focal_lengths_mm": [output_f1, output_f2],
            "input_4f": input_4f, "output_4f": output_4f,
            "input_magnification_signed": input_4f["signed_magnification"],
            "output_magnification_signed": output_4f["signed_magnification"],
            "fill_factor": float(config["dmd"]["fill_factor"]), "status": item["status"],
        })
    return candidates


def _relay_distance_audit(physical_geometry: dict) -> dict:
    segments = physical_geometry.get("distance_segments")
    if not isinstance(segments, list):
        raise ValueError("Physical geometry baseline has no distance segment table")
    source_rows = [item for item in segments if item.get("id") != "06_return"]
    if len(source_rows) != 13:
        raise ValueError("Physical geometry baseline must preserve all 13 source distance rows")
    if any(item.get("status") != "PHOTO_ESTIMATE_NOT_CALIBRATED" for item in source_rows):
        raise ValueError("Source distances must remain photo estimates, not calibrated values")
    by_id = {item["id"]: item for item in segments}
    return_path = by_id.get("06_return")
    if (return_path is None or return_path.get("distance_type") != "separate_return_path_assumption"
            or return_path.get("value_role") != "derived_return_copy_not_independent_measurement"
            or return_path.get("distance_mm") != by_id["06_outbound"].get("distance_mm")):
        raise ValueError("SLM return path must stay a separately labeled copy of the single-pass estimate")
    input_lens_gap = by_id["09"]["distance_mm"]
    input_l1_to_chip = by_id["10"]["distance_mm"] + by_id["11"]["distance_mm"]
    temporary_mirror_to_l2 = by_id["06_return"]["distance_mm"] + by_id["07"]["distance_mm"] + by_id["08"]["distance_mm"]
    input_abcd = geo.ray_matrix(temporary_mirror_to_l2, input_lens_gap, input_l1_to_chip, 150, 150)
    output_object = by_id["12"]["distance_mm"]
    output_sensor = by_id["13"]["distance_mm"]
    output_abcd = geo.ray_matrix(output_object, 0, output_sensor, 30)
    image_distance_conjugate = 1 / (1 / 30 - 1 / output_object)
    return {
        "source_segments": [{"id": item["id"], "branch": item["branch"], "from": item["from"],
                             "to": item["to"], "distance_mm": item["distance_mm"],
                             "status": item["status"], "source_cm": item.get("source_cm"),
                             "photo": item.get("photo")} for item in source_rows],
        "input_placeholder_path": {
            "distances_mm": {"temporary_mirror_return_to_BS": by_id["06_return"]["distance_mm"],
                             "BS_to_Mirror2": by_id["07"]["distance_mm"],
                             "Mirror2_to_RelayL2": by_id["08"]["distance_mm"],
                             "RelayL2_to_RelayL1": input_lens_gap,
                             "RelayL1_to_chip_reserved_position": input_l1_to_chip},
            "ABCD": [list(input_abcd[0]), list(input_abcd[1])],
            "conjugate_B_mm": input_abcd[0][1],
            "status": "TEMPORARY_MIRROR_PHOTO_ESTIMATE_DIAGNOSTIC; NOT THE INSTALLED DMD PATH",
        },
        "output_single_lens_photo_estimate": {
            "object_to_lens_mm": output_object, "lens_to_sensor_mm": output_sensor,
            "focal_length_mm": 30, "ABCD": [list(output_abcd[0]), list(output_abcd[1])],
            "thin_lens_image_distance_mm": image_distance_conjugate,
            "thin_lens_signed_magnification": -image_distance_conjugate / output_object,
            "status": "PHOTO_ESTIMATE_AND_PROVISIONAL_LENS; NOT A CALIBRATED CONJUGATE",
        },
        "distance_semantics": "All 13 source values remain centimeter-rounded tape-photo estimates; theoretical 4f distances are separate candidates.",
        "slm_return_semantics": "The extra 70 mm segment is a derived same-path return assumption, not a fourteenth independent measurement and not the installed DMD return path.",
    }


def _pilot_positions(labels: torch.Tensor, per_class: int) -> torch.Tensor:
    if per_class <= 0:
        raise ValueError("Pilot size per class must be positive")
    selected = []
    for label in range(4):
        positions = torch.where(labels == label)[0]
        if len(positions) < per_class:
            raise ValueError(f"Validation set has fewer than {per_class} examples for class {label}")
        selected.extend(positions[:per_class].tolist())
    return torch.tensor(sorted(selected), dtype=torch.int64)


def _readout_summary(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
    result = {}
    for seed_index, seed in enumerate((11, 23, 37)):
        pred = predictions[seed_index].to(torch.int64)
        cm = torch.bincount(labels.to(torch.int64) * 4 + pred, minlength=16).reshape(4, 4)
        tp = cm.diagonal().double()
        recall = tp / cm.sum(1).clamp_min(1)
        f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
        result[str(seed)] = {"accuracy": float(tp.sum() / len(labels)),
                             "macro_f1": float(f1.mean()), "per_class_recall": recall.tolist(),
                             "confusion_4x4": cm.tolist()}
    result["mean_accuracy"] = sum(result[str(seed)]["accuracy"] for seed in (11, 23, 37)) / 3
    result["mean_macro_f1"] = sum(result[str(seed)]["macro_f1"] for seed in (11, 23, 37)) / 3
    return result


def _aggregate_condition(saved: dict) -> dict:
    values = saved["per_sample"]
    numeric = {}
    for name, tensor in values.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype == torch.bool:
            continue
        finite = tensor[torch.isfinite(tensor)]
        numeric[name] = {"mean": float(finite.mean()) if finite.numel() else None,
                         "std_population": float(finite.std(unbiased=False)) if finite.numel() else None,
                         "finite_count": int(finite.numel()), "total_count": int(tensor.numel())}
    labels = saved["labels"].to(torch.int64)
    classification = {
        mode: _readout_summary(labels, saved["predictions"][mode])
        for mode in ("raw", "normalized")
    }
    return {"condition": saved["condition"], "population": saved["population"],
            "sample_count": int(len(labels)), "original_indices_first_last":
            [int(saved["rows"][0]), int(saved["rows"][-1])],
            "class_counts": torch.bincount(labels, minlength=4).tolist(),
            "metrics": numeric, "frozen_c1_input_plane_readout": classification,
            "readout_scope": "frozen C1 digital head on DMD-to-target-plane normalized pattern; not CMOS or optical classification"}


def _build_cache(config: dict, device: torch.device) -> tuple[list[dict], dict, dict]:
    candidates = _candidate_table(config)
    cache: dict[tuple[str, int, str], dict] = {}
    geometry = {"dmd_array": {"rows": geo.DMD_ROWS, "columns": geo.DMD_COLUMNS,
                               "pitch_um": geo.DMD_PITCH_UM,
                               "width_um": geo.DMD_COLUMNS * geo.DMD_PITCH_UM,
                               "height_um": geo.DMD_ROWS * geo.DMD_PITCH_UM},
                "target": config["lightgen_target"], "candidate_spatial_scales": {},
                "dmd_window_and_target_intersection": {}, "legacy_26_28_comparison": {},
                "energy_accounting": {
                    "reference_plane": "DMD active input plane",
                    "on_active_area_um2": "binary mirror count x DMD pitch area x fill factor",
                    "projected_on_active_area_um2": "input active area multiplied by affine area Jacobian |det(M)|",
                    "target_energy_relative_um2": "target-window irradiance integral divided by |det(M)|; input-plane equivalent area",
                    "outside_energy_relative_um2": "input-plane active area minus captured target-window equivalent area",
                    "fill_factor_geometry": "square active mirror with side pitch*sqrt(fill_factor)",
                    "absolute_optical_power": "UNKNOWN; this is a relative geometric ledger only",
                },
                "relay_distance_audit": None,
                "classification_roi_mapping": ["top_left->0", "top_right->1", "bottom_left->2", "bottom_right->3"]}
    fill_factor = float(config["dmd"]["fill_factor"])
    for candidate in candidates:
        cid = candidate["id"]
        m_in = candidate["input_magnification_signed"]
        m_out = candidate["output_magnification_signed"]
        p_dmd = abs(m_in) * geo.DMD_PITCH_UM
        neuron_pitch = float(config["lightgen_target"]["neuron_pitch_um"])
        cmos_pitch = float(config["cmos"]["pixel_um"])
        geometry["candidate_spatial_scales"][cid] = {
            "input_magnification_signed": m_in, "output_magnification_signed": m_out,
            "dmd_projected_pitch_um": p_dmd,
            "dmd_periods_across_200um": geo.TARGET_WIDTH_UM / p_dmd,
            "mnist_pixel_um": geo.TARGET_WIDTH_UM / 28,
            "dmd_mirrors_per_mnist_pixel": (geo.TARGET_WIDTH_UM / 28) / p_dmd,
            "lightgen_neurons_per_dmd_mirror": p_dmd / neuron_pitch,
            "cmos_pitch_per_lightgen_neuron": abs(m_out) * neuron_pitch / cmos_pitch,
            "projected_lightgen_width_um": abs(m_out) * geo.TARGET_WIDTH_UM,
            "projected_lightgen_width_pixels": abs(m_out) * geo.TARGET_WIDTH_UM / cmos_pitch,
            "cmos_fov_um": [geo.CMOS_COLUMNS * cmos_pitch, geo.CMOS_ROWS * cmos_pitch],
            "projected_field_fits_cmos": (abs(m_out) * geo.TARGET_WIDTH_UM <= geo.CMOS_COLUMNS * cmos_pitch),
            "candidate_status": candidate["status"],
        }
        geometry["legacy_26_28_comparison"][cid] = {
            "historical_conditions": {
                "C2_26_at_1x": {"projected_span_um": 26 * geo.DMD_PITCH_UM,
                                  "target_width_fraction": 26 * geo.DMD_PITCH_UM / geo.TARGET_WIDTH_UM,
                                  "historical_mapping": "26 mirrors at unit magnification"},
                "C2_28_width_matched": {"magnification_abs": geo.TARGET_WIDTH_UM / (28 * geo.DMD_PITCH_UM),
                                         "projected_span_um": geo.TARGET_WIDTH_UM,
                                         "historical_mapping": "28 mirrors scaled to 200 um; not equal to candidate A at 1x"},
            },
            **{str(count): {"projected_span_um": count * p_dmd,
                            "target_width_fraction": count * p_dmd / geo.TARGET_WIDTH_UM,
                            "mirrors_required_for_200um": geo.TARGET_WIDTH_UM / p_dmd}
               for count in (26, 28)}
        }
        geometry["dmd_window_and_target_intersection"][cid] = {}
        for angle in config["dmd"]["installation_angle_candidates_deg"]:
            affine = geo.scaled_rotation(m_in, float(angle))
            coverage_matrix = None
            methods = {}
            lattice = None
            for method in ("nearest", "area"):
                source_matrix, current_lattice = geo.build_source_sampling_matrix(
                    affine, config["coordinate_model"]["default_offset_um"], method,
                    pitch_um=geo.DMD_PITCH_UM)
                if lattice is None:
                    lattice = current_lattice
                    coverage_matrix = geo.build_target_area_matrix(
                        lattice, affine, config["coordinate_model"]["default_offset_um"], fill_factor,
                        pitch_um=geo.DMD_PITCH_UM)
                methods[method] = {"source": source_matrix.to(device), "target": coverage_matrix.to(device)}
            counts = geo.dmd_coverage_counts(lattice, affine,
                                             config["coordinate_model"]["default_offset_um"],
                                             fill_factor=fill_factor)
            geometry["dmd_window_and_target_intersection"][cid][str(angle)] = {
                "local_window_shape_rows_columns": list(lattice["shape"]), **counts,
                "method_source_matrix_nnz": {method: int(methods[method]["source"]._nnz()) for method in methods},
                "target_area_matrix_nnz": int(coverage_matrix._nnz()),
            }
            for method in methods:
                cache[(cid, int(angle), method)] = {
                    "affine": affine, "source_matrix": methods[method]["source"],
                    "target_matrix": methods[method]["target"], "lattice": lattice,
                }
    geometry["relay_distance_audit"] = _relay_distance_audit(_read_json(Path(config["source_files"]["physical_geometry"])))
    geometry["derived_units"] = {"all_lengths": "µm unless field name ends in _mm", "magnification": "signed 4f transverse magnification"}
    return candidates, cache, geometry


def _output_cmos_sampling_audit(candidates: list[dict], config: dict) -> dict:
    """Exercise the ideal output-relay -> finite CMOS-area -> 2x2 ROI path.

    The input is a synthetic uniform field at the LightGen target output plane.
    It checks geometry and energy integration only; no LightGen transfer or
    MNIST class signal is available to this component diagnostic.
    """
    target = config["lightgen_target"]
    camera = config["cmos"]
    coordinate = config["coordinate_model"]
    object_rows, object_cols = int(target["rows"]), int(target["columns"])
    object_pitch = float(target["neuron_pitch_um"])
    if object_rows != object_cols or object_rows <= 0 or object_pitch <= 0:
        raise ValueError("CMOS sampling audit requires a square positive LightGen target grid")
    sensor_shape = (int(camera["rows"]), int(camera["columns"]))
    sensor_pitch = float(camera["pixel_um"])
    if min(*sensor_shape) <= 0 or sensor_pitch <= 0:
        raise ValueError("CMOS sampling audit requires a positive sensor grid and pixel pitch")
    rotation_deg = float(coordinate["output_rotation_deg_diagnostic"])
    offset_um = coordinate["output_offset_um_diagnostic"]
    sensor_fov_um = (sensor_shape[1] * sensor_pitch, sensor_shape[0] * sensor_pitch)
    source_width_um = object_cols * object_pitch
    source = torch.ones((object_rows, object_cols), dtype=torch.float64)
    source_energy = float(source.sum()) * object_pitch**2
    roi_rectangles = geo.equal_quadrant_rois(sensor_shape)
    conditions = {}
    for candidate in candidates:
        magnification = float(candidate["output_magnification_signed"])
        affine = geo.scaled_rotation(magnification, rotation_deg)
        area_matrix, area_info = geo.build_cmos_area_matrix(
            affine,
            offset_um,
            object_pixels=object_rows,
            object_pixel_um=object_pitch,
            sensor_shape=sensor_shape,
            sensor_pixel_um=sensor_pitch,
        )
        camera_energy = geo.integrate_cmos_pixels(source, area_matrix).reshape(sensor_shape)
        roi = geo.roi_scores(camera_energy, roi_rectangles)
        matrix = [[float(value) for value in row] for row in affine]
        half_x = source_width_um / 2 * (abs(matrix[0][0]) + abs(matrix[0][1]))
        half_y = source_width_um / 2 * (abs(matrix[1][0]) + abs(matrix[1][1]))
        roi_scores = [float(value) for value in roi["scores"]]
        camera_energy_total = float(camera_energy.sum())
        capture = area_info["source_area_capture_fraction"]
        conditions[candidate["id"]] = {
            "output_magnification_signed": magnification,
            "coordinate_transform": {
                "matrix": matrix,
                "offset_um": [float(offset_um[0]), float(offset_um[1])],
                "rotation_deg_diagnostic": rotation_deg,
                "status": coordinate["output_transform_status"],
            },
            "projected_field_bounds_um": [
                float(offset_um[0]) - half_x,
                float(offset_um[0]) + half_x,
                float(offset_um[1]) - half_y,
                float(offset_um[1]) + half_y,
            ],
            "projected_field_width_um": 2 * half_x,
            "projected_field_height_um": 2 * half_y,
            "projected_field_fits_cmos": (
                float(offset_um[0]) - half_x >= -sensor_fov_um[0] / 2
                and float(offset_um[0]) + half_x <= sensor_fov_um[0] / 2
                and float(offset_um[1]) - half_y >= -sensor_fov_um[1] / 2
                and float(offset_um[1]) + half_y <= sensor_fov_um[1] / 2
            ),
            "cmos_pixels_per_lightgen_neuron_by_source_axis": [
                math.hypot(matrix[0][axis], matrix[1][axis]) * object_pitch / sensor_pitch
                for axis in (0, 1)
            ],
            "sensor_fov_um": list(sensor_fov_um),
            "source_relative_energy_um2": source_energy,
            "camera_integrated_relative_energy_um2": camera_energy_total,
            "energy_closure_relative_error": abs(camera_energy_total - source_energy) / max(source_energy, 1.0),
            "minimum_cell_capture_fraction": float(capture.min()),
            "maximum_cell_capture_fraction": float(capture.max()),
            "roi_scores_for_uniform_field_relative_um2": roi_scores,
            "roi_prediction_for_uniform_field": int(roi["prediction"]),
            "uniform_roi_score_tie": int(roi["prediction"]) == -1 and sum(roi_scores) > 0,
            "outside_roi_relative_energy_um2": max(0.0, float(roi["outside_energy"])),
            "interpretation": "synthetic uniform-field integration check only; ROI tie is expected and is not a classification result",
        }
    return {
        "status": "COMPONENT_DIAGNOSTIC_ONLY_NO_LIGHTGEN_FIELD",
        "source_field": "all-ones intensity on the 100x100, 2um output-plane target grid; arbitrary relative units",
        "source_plane": "LightGen output plane (synthetic geometry fixture, not a modeled physical LightGen output)",
        "sensor_shape": list(sensor_shape),
        "sensor_pixel_um": sensor_pitch,
        "sensor_fov_um": list(sensor_fov_um),
        "roi_layout": "four equal non-overlapping quadrants over the full virtual sensor",
        "roi_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "roi_to_label": {"top_left": 0, "top_right": 1, "bottom_left": 2, "bottom_right": 3},
        "conditions": conditions,
        "excluded": ["LightGen propagation or transmission", "DMD coherent response", "CMOS QE/noise/gain/ADC", "MNIST classification", "hardware calibration"],
    }


def _flatfield_uniformity(cache: dict, config: dict) -> dict:
    results = {}
    fill = float(config["dmd"]["fill_factor"])
    for key, item in cache.items():
        cid, angle, method = key
        if method != "nearest":
            continue
        mirrors = item["source_matrix"].shape[0]
        all_on = torch.ones((mirrors, 1), dtype=torch.float64, device=item["target_matrix"].device)
        coverage = torch.sparse.mm(item["target_matrix"], all_on).reshape(100, 100)
        affine = item["affine"]
        det = abs(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0])
        irradiance = coverage / det
        results[f"{cid}/angle_{angle}"] = {
            "model": "all-mirrors-on geometric flatfield; square active mirror fill assumption",
            "mean_irradiance_relative": float(irradiance.mean()),
            "cv_across_100x100_pixels": float(irradiance.std(unbiased=False) / irradiance.mean()),
            "min_relative": float(irradiance.min()), "max_relative": float(irradiance.max()),
            "fill_factor": fill,
        }
    return results


def _grating_sensitivity(candidates: list[dict], config: dict, device: torch.device) -> dict:
    cutoffs = config["relay_diagnostic_scenarios"]["pupil_cutoff_cycles_per_um"]
    defocus_values = config["relay_diagnostic_scenarios"]["defocus_um"]
    periods = config["relay_diagnostic_scenarios"]["dmd_grating_periods_in_mirrors"]
    fill = float(config["dmd"]["fill_factor"])
    wavelength_um = float(config["lightgen_target"]["wavelength_nm"]) / 1000
    cases = []
    for candidate in candidates:
        pitch = abs(candidate["input_magnification_signed"]) * geo.DMD_PITCH_UM
        for period_mirrors in periods:
            n = max(64, int(period_mirrors) * 4 * 16)
            period_um = period_mirrors * pitch
            sample_um = period_um / (16 * period_mirrors)
            x = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * sample_um
            mirror_index = torch.floor(x / pitch).to(torch.int64)
            local = x / pitch - mirror_index
            on_mirrors = max(1, (period_mirrors + 1) // 2)
            off_mirrors = period_mirrors - on_mirrors
            commands_on = mirror_index.remainder(period_mirrors) < on_mirrors
            active = commands_on & (local < math.sqrt(fill))
            line = active.to(torch.complex128)
            field = line[None, None, :].expand(1, n, n).contiguous()
            for cutoff in cutoffs:
                for defocus_um in defocus_values:
                    filtered = c2.coherent_pupil(field, sample_um, float(cutoff),
                                                 defocus_um=float(defocus_um),
                                                 wavelength_um=wavelength_um)
                    intensity = filtered.abs().square().mean(dim=-2).flatten()
                    maximum, minimum = float(intensity.max()), float(intensity.min())
                    modulation = (maximum - minimum) / (maximum + minimum) if maximum + minimum else 0.0
                    cases.append({"candidate": candidate["id"], "input_projected_pitch_um": pitch,
                                  "command_period_mirrors": period_mirrors,
                                  "command_profile": ("all_on" if off_mirrors == 0 else
                                                      f"{on_mirrors}-on/{off_mirrors}-off"),
                                  "command_on_mirrors_per_period": on_mirrors,
                                  "command_off_mirrors_per_period": off_mirrors,
                                  "modulation_origin": ("finite active-mirror aperture lattice" if off_mirrors == 0 else
                                                        "ON/OFF command pattern multiplied by active-mirror aperture"),
                                  "spatial_period_um": period_um, "fundamental_cycles_per_um": 1 / period_um,
                                  "fundamental_semantics": "resultant field supercell, including active-mirror aperture",
                                  "pupil_cutoff_cycles_per_um": cutoff, "defocus_um": defocus_um,
                                  "sample_pitch_um": sample_um, "modulation_depth": modulation,
                                  "model_status": "UNPHYSICAL_DIAGNOSTIC_ONLY",
                                  "interpretation": "ideal coherent pupil scenario; not measured HW207 NA or relay prediction"})
    return {"status": "UNPHYSICAL_DIAGNOSTIC_ONLY", "cases": cases,
            "pupil_scenarios_source": "legacy C2 sensitivity values, reused as explicit scenarios"}


def _write_geometry_artifacts(out_dir: Path, candidates: list[dict], cache: dict, geometry: dict,
                              config: dict, device: torch.device,
                              test_receipt: dict | None = None) -> None:
    camera_audit = _output_cmos_sampling_audit(candidates, config)
    geometry["output_camera_sampling"] = camera_audit
    geometry["all_on_flatfield_uniformity"] = _flatfield_uniformity(cache, config)
    geometry["relay_candidates"] = candidates
    geometry["c3_statuses"] = {
        "C3_DMD_SCALAR_BINARY": "PILOT_DIAGNOSTIC_COMPLETE; full-validation audit pending",
        "C3_DMD_COHERENT_MICROMIRROR": "BLOCKED",
        "C3_INPUT_RELAY_ACTUAL_GEOMETRY": "BLOCKED; only ideal 4f and explicitly unphysical pupil scenarios are computed",
        "C3_LIGHTGEN_FIXED_STACK": "BLOCKED",
        "C3_OUTPUT_RELAY_GEOMETRY": "THEORY_ONLY; candidate coordinate transform available",
        "C3_CMOS_PIXEL_INTEGRATION_COMPONENT": (
            "NUMERICAL_DIAGNOSTIC_PASS"
            if test_receipt and test_receipt.get("status") == "PASS"
            else "TARGETED_TEST_REQUIRED"
        ),
        "C3_CMOS_NOISE_AND_READOUT": "BLOCKED_PARAMETERS_UNKNOWN",
        "C3_2X2_CMOS_CLASSIFICATION": "BLOCKED_NO_LIGHTGEN_OUTPUT_FIELD_OR_CALIBRATED_ROI",
        "HW207_CURRENT_MAPPING": "BLOCKED",
    }
    geometry["targeted_test_receipt"] = test_receipt
    _atomic_json(out_dir / "spatial_geometry.json", geometry)
    _atomic_json(out_dir / "output_camera_sampling_audit.json", camera_audit)
    _atomic_json(out_dir / "relay_pupil_sensitivity.json", _grating_sensitivity(candidates, config, device))


def _compute_candidate_batch(images: torch.Tensor, reference: torch.Tensor,
                             candidate: dict, cache: dict, heads: dict) -> dict:
    outputs: dict[tuple[int, str], dict] = {}
    for angle in (0, 45):
        for method in ("nearest", "area"):
            item = cache[(candidate["id"], angle, method)]
            rendered = geo.map_binary_images(images, item["source_matrix"], item["target_matrix"], item["affine"])
            quality = geo.image_quality_metrics(reference, rendered["relative_pattern"])
            energy = geo.target_energy_account(rendered["commands"], rendered["irradiance"],
                                               candidate["fill_factor"], item["affine"])
            readouts = {}
            for mode in ("raw", "normalized"):
                predictions = c2.frozen_scores(rendered["relative_pattern"], heads[mode], mode == "normalized")[0]
                readouts[mode] = torch.stack([row["prediction"].detach().cpu().to(torch.int64)
                                               for row in predictions])
            outputs[(angle, method)] = {"rendered": rendered, "quality": quality,
                                        "energy": energy, "readouts": readouts}
    result = {}
    ref_sum = reference.double().sum((1, 2)).clamp_min(1)
    for angle in (0, 45):
        near = outputs[(angle, "nearest")]
        area = outputs[(angle, "area")]
        command_change = (near["rendered"]["commands"] != area["rendered"]["commands"]).double().mean(1)
        mapping_l1 = (near["rendered"]["relative_pattern"] - area["rendered"]["relative_pattern"]).abs().double().sum((1, 2)) / ref_sum
        orientation_l1 = {}
        for method in ("nearest", "area"):
            out0 = outputs[(0, method)]["rendered"]["relative_pattern"]
            out45 = outputs[(45, method)]["rendered"]["relative_pattern"]
            orientation_l1[method] = (out0 - out45).abs().double().sum((1, 2)) / ref_sum
        for method in ("nearest", "area"):
            item = outputs[(angle, method)]
            per_sample = {key: value.detach().cpu() for key, value in item["quality"].items()}
            per_sample.update({key: value.detach().cpu() for key, value in item["energy"].items()})
            per_sample["command_fraction_different_vs_area_mapping"] = command_change.detach().cpu()
            per_sample["nearest_area_pattern_l1_over_reference"] = mapping_l1.detach().cpu()
            per_sample["orientation_0_vs_45_pattern_l1_over_reference"] = orientation_l1[method].detach().cpu()
            result[(angle, method)] = {
                "per_sample": per_sample,
                "predictions": {mode: prediction for mode, prediction in item["readouts"].items()},
            }
    return result


def _run_population(images: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor,
                    candidates: list[dict], cache: dict, heads: dict, device: torch.device,
                    batch_size: int, output_dir: Path, population_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_results: dict[str, dict[str, list[torch.Tensor]]] = {}
    for candidate in candidates:
        for angle in (0, 45):
            for method in ("nearest", "area"):
                key = f"{candidate['id']}/angle_{angle}/{method}"
                row_results[key] = {"metrics": {}, "predictions": {"raw": [], "normalized": []}}
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        raw_images = images[start:end]
        binary = features(raw_images, True).to(device)
        reference = map_image(binary, True).to(device)
        for candidate in candidates:
            outputs = _compute_candidate_batch(binary, reference, candidate, cache, heads)
            for (angle, method), value in outputs.items():
                key = f"{candidate['id']}/angle_{angle}/{method}"
                target = row_results[key]
                for name, tensor in value["per_sample"].items():
                    target["metrics"].setdefault(name, []).append(tensor.detach().cpu())
                for mode, prediction in value["predictions"].items():
                    target["predictions"][mode].append(prediction)
        print(f"{population_name.upper()} rows={end}/{len(rows)}", flush=True)
    summary = {"population": population_name, "sample_count": len(rows), "conditions": {}}
    for candidate in candidates:
        for angle in (0, 45):
            for method in ("nearest", "area"):
                key = f"{candidate['id']}/angle_{angle}/{method}"
                data = row_results[key]
                saved = {"identity": "HW207_C21_C3_SPATIAL_DIAGNOSTIC",
                         "condition": key, "population": population_name,
                         "rows": rows.detach().cpu(), "labels": labels.detach().cpu(),
                         "per_sample": {name: torch.cat(values) for name, values in data["metrics"].items()},
                         "predictions": {name: torch.cat(values, dim=1) for name, values in data["predictions"].items()},
                         "readout_scope": "frozen C1 digital head on DMD-to-target-plane normalized pattern; not CMOS or optical classification"}
                result_path = output_dir / f"{candidate['id']}_angle_{angle}_{method}.pt"
                if result_path.exists():
                    existing = torch.load(result_path, map_location="cpu", weights_only=True)
                    if (existing.get("condition") != key or existing.get("population") != population_name
                            or not torch.equal(existing.get("rows"), saved["rows"])
                            or not torch.equal(existing.get("labels"), saved["labels"])
                            or set(existing.get("per_sample", {})) != set(saved["per_sample"])
                            or any(not torch.equal(existing["per_sample"][name], saved["per_sample"][name])
                                   for name in saved["per_sample"])
                            or any(not torch.equal(existing["predictions"][mode], saved["predictions"][mode])
                                   for mode in ("raw", "normalized"))):
                        raise ValueError(f"Existing result file does not match explicit condition: {result_path}")
                    saved = existing
                else:
                    _atomic_torch(result_path, saved)
                item_summary = _aggregate_condition(saved)
                item_summary["result_file"] = str(result_path)
                summary["conditions"][key] = item_summary
    return summary


def _load_heads(readout_root: Path) -> dict:
    if not (readout_root / "readout_selection.json").is_file():
        raise FileNotFoundError(f"Missing explicit C1 readout selection: {readout_root}")
    return c2.bind_identity_binary_heads(readout_root, torch.device("cpu"))


def _input_provenance(paths: dict[str, Path], heads: dict, device: torch.device) -> dict:
    imported = {
        "geometry_module": Path(geo.__file__),
        "c0_module": Path(__import__("hw207_mnist4_c0_v1").__file__),
        "c1_module": Path(__import__("hw207_mnist4_c1_v1").__file__),
        "c2_module": Path(c2.__file__),
        "v2_contracts_module": Path(__import__("ldm_project.v2.contracts",
                                                  fromlist=["PlaneSpec"]).__file__),
        "v2_propagation_module": Path(__import__("ldm_project.v2.optics.propagation",
                                                     fromlist=["angular_spectrum_propagate"]).__file__),
    }
    return {"explicit_inputs": {name: {"path": str(path.resolve()), "sha256_provenance_only": _sha256(path)}
            for name, path in paths.items() if path.is_file()} | {
                "frozen_c1_readouts": {
                    mode: [{"seed": model["seed"], "path": model["path"],
                            "sha256_provenance_only": model["sha256"]} for model in heads[mode]["models"]]
                    for mode in ("raw", "normalized")
                }
            },
            "production_sources": {
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256_provenance_only": _sha256(Path(__file__)),
                "imported_modules": {name: {"path": str(path.resolve()),
                                             "sha256_provenance_only": _sha256(path)}
                                     for name, path in imported.items()},
            },
            "execution_environment": {"torch": torch.__version__, "device": str(device),
                                      "cuda_available": torch.cuda.is_available(),
                                      "cuda_device_name": torch.cuda.get_device_name(0)
                                      if device.type == "cuda" else None}}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "validation", "audit", "geometry-only"), required=True)
    for name in ("config", "physical-geometry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("indices", "data", "train", "test", "c1-readouts"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _load_effective_config(config_path: Path) -> tuple[dict, dict, Path | None]:
    raw = _read_json(config_path)
    if raw.get("identity") != "HW207_MNIST4_C21_C3_ALIGNED_GEOMETRY_V2":
        return raw, raw, None
    base_path = config_path.with_name(raw["base_config_file"])
    base = _read_json(base_path)
    if base.get("identity") != "HW207_MNIST4_C21_C3_SPATIAL_V1" or not base.get("non_runtime"):
        raise ValueError("Aligned-geometry configuration must reference the explicit C2.1/C3 v1 base")
    effective = dict(base)
    effective["identity"] = raw["identity"]
    effective["purpose"] = raw["purpose"]
    effective["source_files"] = {**base["source_files"], **raw["source_files"]}
    effective["coordinate_model"] = {
        **base["coordinate_model"],
        "output_rotation_deg_diagnostic": float(raw["output_coordinate_assumption"]["rotation_deg"]),
        "output_offset_um_diagnostic": list(raw["output_coordinate_assumption"]["offset_um"]),
        "output_transform_status": raw["output_coordinate_assumption"]["status"],
    }
    effective["aligned_geometry_v2"] = raw
    return raw, effective, base_path


def _prior_c21_validation_evidence(config: dict, config_path: Path) -> dict:
    """Require the already completed C2.1 validation before geometry closeout."""
    spec = config["aligned_geometry_v2"]["prior_c21_evidence"]
    repo_root = config_path.resolve().parents[3]
    validation_path = repo_root / config["source_files"]["prior_c21_validation_summary"]
    audit_path = repo_root / config["source_files"]["prior_c21_independent_audit"]
    validation = _read_json(validation_path)
    audit = _read_json(audit_path)
    candidates = _candidate_table(config)
    expected = [f"{candidate['id']}/angle_{angle}/{method}"
                for candidate in candidates for angle in (0, 45) for method in ("nearest", "area")]
    if (validation.get("population") != "validation"
            or validation.get("sample_count") != spec["sample_count"]
            or list(validation.get("conditions", {})) != expected
            or validation.get("official_test_evaluated") is not False):
        raise ValueError("Saved C2.1 validation summary does not match the declared population/condition matrix")
    for key, condition in validation["conditions"].items():
        if (condition.get("condition") != key or condition.get("population") != "validation"
                or condition.get("sample_count") != spec["sample_count"]
                or sum(condition.get("class_counts", [])) != spec["sample_count"]
                or "not CMOS or optical classification" not in condition.get("readout_scope", "")):
            raise ValueError(f"Saved C2.1 condition has invalid population or readout scope: {key}")
    population = audit.get("population", {})
    if (audit.get("status") != spec["required_local_independent_audit_status"]
            or audit.get("condition_count") != spec["condition_count"]
            or population.get("validation_rows") != spec["sample_count"]
            or population.get("validation_order_matches_c0_indices") is not True
            or population.get("labels_match_local_mnist_train_file") is not True
            or population.get("train_validation_disjoint") is not True
            or audit.get("official_test_evaluated") is not False
            or audit.get("c2_original_status_retained") != spec["c2_original_status"]):
        raise ValueError("Saved C2.1 independent audit is incomplete or disagrees with the declared status")
    return {
        "status": "REUSED_AND_LOCALLY_AUDITED",
        "validation_summary": str(validation_path.resolve()),
        "independent_audit": str(audit_path.resolve()),
        "population": validation["population"],
        "sample_count_per_condition": validation["sample_count"],
        "condition_count": len(validation["conditions"]),
        "class_counts_0_to_3": population["class_counts_0_to_3"],
        "original_validation_order_matches_c0_indices": population["validation_order_matches_c0_indices"],
        "official_test_evaluated_in_saved_run": validation["official_test_evaluated"],
        "readout_scope": spec["result_scope"],
        "local_independent_audit_status": audit["status"],
        "c2_original_status": spec["c2_original_status"],
        "c21_status": spec["c21_status"],
    }


def _run_geometry_only(args, raw_config: dict, config: dict, base_config_path: Path | None,
                       physical_geometry: dict, device: torch.device) -> None:
    if raw_config.get("identity") != "HW207_MNIST4_C21_C3_ALIGNED_GEOMETRY_V2":
        raise ValueError("geometry-only requires the versioned aligned-geometry v2 config")
    if not args.output.is_dir() or {item.name for item in args.output.iterdir()} != {"tests.xml"}:
        raise FileExistsError("Geometry-only output must be a fresh attempt directory containing only tests.xml")
    test_receipt = _test_receipt(args.output / "tests.xml")
    prior_validation = _prior_c21_validation_evidence(config, args.config)
    candidates, _, geometry = _build_cache(config, device)
    geometry["relay_distance_audit"] = _relay_distance_audit(physical_geometry)
    geometry["dmd_array_bounds_by_installation_angle"] = {
        str(angle): geo.dmd_array_bounds_um(float(angle))
        for angle in config["dmd"]["installation_angle_candidates_deg"]
    }
    geometry["prior_c21_validation_evidence"] = prior_validation
    camera_audit = _output_cmos_sampling_audit(candidates, config)
    geometry["output_camera_sampling"] = camera_audit
    geometry["relay_candidates"] = candidates
    geometry["targeted_test_receipt"] = test_receipt
    geometry["c3_statuses"] = {
        "overall": "C3_PARTIAL_PASS",
        "C3_DMD_SCALAR_BINARY": "NUMERICAL_DIAGNOSTIC_PASS_INPUT_PLANE_ONLY; prior 12-condition validation reused",
        "C3_DMD_COHERENT_MICROMIRROR": "BLOCKED_UNKNOWN_INCIDENCE_AND_ON_OFF_COLLECTION",
        "C3_INPUT_RELAY_ACTUAL_GEOMETRY": "BLOCKED; ideal 4f candidate geometry only",
        "C3_LIGHTGEN_FIXED_STACK": "BLOCKED_NO_FIXED_LAYER_TRANSMISSION_OR_FIELD",
        "C3_OUTPUT_RELAY_GEOMETRY": "THEORY_ONLY_IDEAL_4F_COORDINATE_CANDIDATES",
        "C3_CMOS_PIXEL_INTEGRATION_COMPONENT": "NUMERICAL_DIAGNOSTIC_PASS_SYNTHETIC_FIELD_AND_TARGETED_TESTS",
        "C3_CMOS_NOISE_AND_READOUT": "BLOCKED_PARAMETERS_UNKNOWN",
        "C3_2X2_CMOS_CLASSIFICATION": "BLOCKED_NO_LIGHTGEN_OUTPUT_FIELD_OR_CALIBRATED_ROI",
        "C2_HARDWARE_CALIBRATION": "BLOCKED",
        "C2_COHERENT_DMD_MODEL": "BLOCKED_OR_DIAGNOSTIC_ONLY",
        "C2_original": "C2_PARTIAL_PASS",
        "C21": "C2.1_GEOMETRY_DIAGNOSTIC_COMPLETE",
        "HW207_CURRENT_MAPPING": "BLOCKED",
    }
    geometry["execution_scope"] = raw_config["execution_scope"]
    geometry["execution_scope"]["new_mnist_rows_processed"] = 0
    geometry["execution_scope"]["prior_validation_reprocessed"] = False
    geometry["execution_scope"]["test_receipt"] = test_receipt
    geometry["source_provenance"] = {
        "config": str(args.config.resolve()),
        "base_config": str(base_config_path.resolve()) if base_config_path else None,
        "physical_geometry": str(args.physical_geometry.resolve()),
        "geometry_module": str(Path(geo.__file__).resolve()),
        "parameter_baseline": str((args.config.resolve().parents[3] / config["source_files"]["parameter_baseline"]).resolve()),
        "original_bom": str((args.config.resolve().parents[3] / config["source_files"]["original_bom"]).resolve()),
    }
    _atomic_json(args.output / "spatial_geometry.json", geometry)
    _atomic_json(args.output / "output_camera_sampling_audit.json", camera_audit)
    _atomic_json(args.output / "c3_geometry_closeout_summary.json", {
        "identity": "HW207_C21_C3_ALIGNED_GEOMETRY_V2",
        "status": "C3_PARTIAL_PASS",
        "c2_original_status": "C2_PARTIAL_PASS",
        "c21_status": "C2.1_GEOMETRY_DIAGNOSTIC_COMPLETE",
        "prior_c21_validation": prior_validation,
        "output_cmos_sampling": {
            "status": camera_audit["status"],
            "candidate_count": len(camera_audit["conditions"]),
            "max_energy_closure_relative_error": max(
                item["energy_closure_relative_error"] for item in camera_audit["conditions"].values()
            ),
            "all_uniform_roi_cases_abstain": all(
                item["roi_prediction_for_uniform_field"] == -1
                for item in camera_audit["conditions"].values()
            ),
        },
        "targeted_test_receipt": test_receipt,
        "official_test_evaluated": False,
        "new_mnist_rows_processed": 0,
        "hardware_controlled": False,
        "lightgen_transmission_or_output_field_loaded": False,
        "c3_full_physical_classification": "BLOCKED_NO_LIGHTGEN_OUTPUT_FIELD_OR_CALIBRATED_ROI",
        "hw207_current_mapping": "BLOCKED",
    })
    print("GEOMETRY_ONLY_COMPLETE c3=C3_PARTIAL_PASS new_mnist_rows=0", flush=True)


def _test_receipt(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Targeted spatial test receipt is required: {path}")
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root)
    else:
        suites = [root]
    counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] < 16 or counts["failures"] or counts["errors"]:
        raise ValueError(f"Spatial test receipt is incomplete or failing: {counts}")
    return {"path": str(path.resolve()), **counts, "status": "PASS"}


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    raw_config, config, base_config_path = _load_effective_config(args.config)
    physical_geometry = _read_json(args.physical_geometry)
    if not config.get("non_runtime") or config.get("identity") not in (
            "HW207_MNIST4_C21_C3_SPATIAL_V1", "HW207_MNIST4_C21_C3_ALIGNED_GEOMETRY_V2"):
        raise ValueError("Unexpected diagnostic configuration identity")
    if args.phase == "geometry-only" and config["identity"] != "HW207_MNIST4_C21_C3_ALIGNED_GEOMETRY_V2":
        raise ValueError("geometry-only requires the versioned aligned-geometry v2 configuration")
    if args.phase != "geometry-only" and config["identity"] != "HW207_MNIST4_C21_C3_SPATIAL_V1":
        raise ValueError("MNIST data phases continue to use the frozen C2.1/C3 v1 configuration")
    if ((config["dmd"]["rows"], config["dmd"]["columns"], config["dmd"]["pitch_um"])
            != (geo.DMD_ROWS, geo.DMD_COLUMNS, geo.DMD_PITCH_UM)):
        raise ValueError("DMD configuration and production geometry constants disagree")
    if ((config["lightgen_target"]["rows"], config["lightgen_target"]["columns"],
            config["lightgen_target"]["neuron_pitch_um"], config["lightgen_target"]["active_width_um"])
            != (geo.TARGET_PIXELS, geo.TARGET_PIXELS, geo.TARGET_PIXEL_UM, geo.TARGET_WIDTH_UM)):
        raise ValueError("LightGen target grid and production geometry constants disagree")
    if ((config["cmos"]["rows"], config["cmos"]["columns"], config["cmos"]["pixel_um"])
            != (geo.CMOS_ROWS, geo.CMOS_COLUMNS, geo.CMOS_PIXEL_UM)):
        raise ValueError("CMOS configuration and production geometry constants disagree")
    config["source_files"]["physical_geometry"] = str(args.physical_geometry)
    if args.phase == "geometry-only":
        _run_geometry_only(args, raw_config, config, base_config_path,
                            physical_geometry, device)
        return
    missing = [name for name in ("indices", "data", "train", "test", "c1_readouts")
               if getattr(args, name) is None]
    if missing:
        raise ValueError(f"MNIST phases require explicit inputs: {missing}")
    paths = {"config": args.config, "physical_geometry": args.physical_geometry,
             "indices": args.indices, "data_roles": args.data, "mnist_train": args.train,
             "mnist_test_role_check_only": args.test}
    if args.phase == "pilot":
        if args.output.exists():
            if (not args.output.is_dir()
                    or not {item.name for item in args.output.iterdir()}.issubset(
                        {"tests.xml", "inputs", "source_overlay", "approved_command_pilot.sh"})
                    or "tests.xml" not in {item.name for item in args.output.iterdir()}):
                raise FileExistsError(args.output)
        else:
            args.output.mkdir(parents=True)
        test_receipt = _test_receipt(args.output / "tests.xml")
    elif not args.output.is_dir():
        raise FileNotFoundError(f"Explicit pilot output directory is required: {args.output}")
    else:
        test_receipt = _test_receipt(args.output / "tests.xml")
    if args.phase == "validation" and not (args.output / "pilot_summary.json").is_file():
        raise FileNotFoundError("Validation phase requires the completed pilot summary")
    heads = _load_heads(args.c1_readouts)
    samples = population(args.indices, args.data, args.train, args.test)
    validation_images, validation_labels, validation_rows = samples["validation"]
    if len(validation_rows) != 2475 or len(torch.unique(validation_rows)) != 2475:
        raise ValueError("C0 validation population identity is invalid")
    candidates, cache, geometry = _build_cache(config, device)
    if args.phase == "pilot":
        positions = _pilot_positions(validation_labels, int(config["mnist"]["pilot_per_class"]))
        started = time.perf_counter()
        pilot = _run_population(validation_images[positions], validation_labels[positions],
                                validation_rows[positions], candidates, cache, heads, device,
                                args.batch_size, args.output / "stage1_pilot", "validation_pilot")
        pilot["selection_rule"] = "retain all 12 relay x mapping x installation-angle conditions; no predeclared evidence-based exclusion criterion exists, and A/B/C plus 0/45 are all planned contrasts"
        pilot["elapsed_seconds_observed"] = time.perf_counter() - started
        pilot["selected_condition_count_for_full_validation"] = len(pilot["conditions"])
        pilot["test_receipt"] = test_receipt
        pilot["execution_environment"] = {"torch": torch.__version__, "device": str(device),
                                          "cuda_available": torch.cuda.is_available(),
                                          "cuda_device_name": torch.cuda.get_device_name(0)
                                          if device.type == "cuda" else None}
        geometry["relay_distance_audit"] = _relay_distance_audit(physical_geometry)
        geometry["pilot_evidence"] = {"status": "COMPLETED", "rows": len(positions),
                                      "per_class": int(config["mnist"]["pilot_per_class"]),
                                      "conditions": len(pilot["conditions"])}
        _write_geometry_artifacts(args.output, candidates, cache, geometry, config, device, test_receipt)
        _atomic_json(args.output / "pilot_summary.json", pilot)
        _atomic_json(args.output / "input_provenance.json", _input_provenance(paths, heads, device))
        print(f"PILOT_COMPLETE conditions={len(pilot['conditions'])} rows_per_condition=256", flush=True)
        return
    if args.phase == "validation":
        pilot = _read_json(args.output / "pilot_summary.json")
        retained = list(pilot.get("conditions", {}))
        expected = [f"{candidate['id']}/angle_{angle}/{method}"
                    for candidate in candidates for angle in (0, 45) for method in ("nearest", "area")]
        if retained != expected:
            raise ValueError("Pilot condition list differs from the predeclared A/B/C x mapping x angle matrix")
        started = time.perf_counter()
        full = _run_population(validation_images, validation_labels, validation_rows,
                               candidates, cache, heads, device, args.batch_size,
                               args.output / "stage2_validation", "validation")
        full["retained_by_pilot"] = retained
        full["official_test_evaluated"] = False
        full["test_history_note"] = config["mnist"]["test_history_note"]
        full["elapsed_seconds_observed"] = time.perf_counter() - started
        full["test_receipt"] = test_receipt
        full["execution_environment"] = {"torch": torch.__version__, "device": str(device),
                                         "cuda_available": torch.cuda.is_available(),
                                         "cuda_device_name": torch.cuda.get_device_name(0)
                                         if device.type == "cuda" else None}
        _atomic_json(args.output / "validation_summary.json", full)
        print(f"VALIDATION_COMPLETE conditions={len(full['conditions'])} rows_per_condition={len(validation_rows)}", flush=True)
        return
    _audit(args, config, physical_geometry, samples, candidates, cache, heads, device)


def _audit(args, config: dict, physical_geometry: dict, samples: dict,
           candidates: list[dict], cache: dict, heads: dict, device: torch.device) -> None:
    summary_path = args.output / "validation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("C2.1 full validation summary is missing")
    saved_summary = _read_json(summary_path)
    images, labels, rows = samples["validation"]
    if saved_summary.get("sample_count") != len(rows) or len(saved_summary.get("conditions", {})) != 12:
        raise ValueError("Full validation population or condition count is incomplete")
    if not (args.output / "pilot_summary.json").is_file():
        raise FileNotFoundError("Completed stratified pilot summary is missing")
    test_receipt = _test_receipt(args.output / "tests.xml")
    checked = 0
    for condition, report in saved_summary["conditions"].items():
        result_path = Path(report["result_file"])
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        saved = torch.load(result_path, map_location="cpu", weights_only=True)
        if (saved.get("condition") != condition or saved.get("population") != "validation"
                or not torch.equal(saved["rows"], rows) or not torch.equal(saved["labels"], labels)):
            raise ValueError(f"Population or condition mismatch: {result_path}")
        recomputed = _aggregate_condition(saved)
        if recomputed["frozen_c1_input_plane_readout"] != report["frozen_c1_input_plane_readout"]:
            raise ValueError(f"Saved readout summary does not match per-sample predictions: {condition}")
        for metric, stats in recomputed["metrics"].items():
            reported = report["metrics"][metric]
            if stats["finite_count"] != reported["finite_count"] or abs(stats["mean"] - reported["mean"]) > 1e-12:
                raise ValueError(f"Saved per-sample metric does not match summary: {condition}/{metric}")
        checked += 1
    # Recompute the entire fixed validation population through the production
    # mapper in a separate pass, then compare every saved sample statistic.
    replay = _run_population(images, labels, rows, candidates, cache, heads, device,
                             args.batch_size, args.output / "audit_replay", "validation_replay")
    maximum_metric_difference = 0.0
    for condition, item in replay["conditions"].items():
        original = torch.load(saved_summary["conditions"][condition]["result_file"],
                              map_location="cpu", weights_only=True)
        replay_item = torch.load(item["result_file"], map_location="cpu", weights_only=True)
        if (not torch.equal(original["rows"], replay_item["rows"])
                or not torch.equal(original["labels"], replay_item["labels"])):
            raise ValueError(f"Independent replay population identity mismatch: {condition}")
        for mode in ("raw", "normalized"):
            if not torch.equal(original["predictions"][mode], replay_item["predictions"][mode]):
                raise ValueError(f"Independent replay prediction mismatch: {condition}/{mode}")
        for metric, saved_values in original["per_sample"].items():
            replay_values = replay_item["per_sample"][metric]
            difference = (saved_values.double() - replay_values.double()).abs()
            finite_difference = difference[torch.isfinite(difference)]
            maximum = float(finite_difference.max()) if finite_difference.numel() else 0.0
            maximum_metric_difference = max(maximum_metric_difference, maximum)
            if not torch.allclose(saved_values, replay_values, atol=1e-7, rtol=1e-7, equal_nan=True):
                raise ValueError(f"Independent replay geometry metric mismatch: {condition}/{metric}")
        checked += 1
    pilot_geometry = _read_json(args.output / "spatial_geometry.json")
    statuses = dict(pilot_geometry["c3_statuses"])
    statuses["C3_DMD_SCALAR_BINARY"] = "NUMERICAL_DIAGNOSTIC_PASS"
    statuses["C3_CMOS_PIXEL_INTEGRATION_COMPONENT"] = "NUMERICAL_DIAGNOSTIC_PASS"
    c3_summary = {
        "identity": config["identity"], "c2_original_status": "C2_PARTIAL_PASS",
        "c2_1_status": "C2.1_GEOMETRY_DIAGNOSTIC_COMPLETE",
        "c3_status": "C3_PARTIAL_PASS", "c3_statuses": statuses,
        "validation_population": {"count": len(rows), "original_indices_first_last":
                                   [int(rows[0]), int(rows[-1])]},
        "c21_condition_count": len(saved_summary["conditions"]),
        "c21_conditions": saved_summary["conditions"],
        "c3_end_to_end_classification": "BLOCKED: fixed LightGen output field/complex transmission map and calibrated CMOS ROI are unavailable",
        "test_history_note": config["mnist"]["test_history_note"],
        "targeted_tests": test_receipt,
        "independent_replay": {"conditions": checked, "rows_per_condition": len(rows),
                               "predictions_match": True,
                               "max_abs_per_sample_metric_difference": maximum_metric_difference},
        "energy_ledger": {"DMD_incident_absolute_power_W": "UNKNOWN",
                          "DMD_collected_power_W": "UNKNOWN: incidence, ON/OFF port and collection NA are unmeasured",
                          "input_relay_loss": "UNKNOWN: installed relay aperture/transmission/aberration are unmeasured",
                          "chip_input_relative_geometric_capture": "reported per C2.1 condition under SCALAR_BINARY_APPROXIMATION",
                          "LightGen_output_and_transmission": "UNKNOWN: no fixed physical layer map or measured stack",
                          "output_relay_loss": "UNKNOWN: installed relay and field are unmeasured",
                          "CMOS_absolute_photons_or_electrons": "UNKNOWN: optical power, 532 nm QE and readout mode are not fixed",
                          "CMOS_ROI_energy": "NOT_COMPUTED: no supported LightGen output field or calibrated ROIs"},
        "official_test_evaluated": False,
        "science_boundary": "C2.1 finite-area binary DMD-to-target-plane geometry and isolated CMOS area-integration component were numerically checked. No full DMD-to-LightGen-to-CMOS classification or hardware performance is established.",
    }
    _atomic_json(args.output / "c3_summary.json", c3_summary)
    audit = {
        "identity": "HW207_C21_C3_INDEPENDENT_LOCAL_AUDIT",
        "status": "PASS",
        "full_validation_conditions_checked": 12,
        "rows_per_condition": len(rows),
        "all_original_validation_indices_order_verified": True,
        "confusion_and_classification_summary_recomputed_from_per_sample_predictions": True,
        "per_sample_geometry_metrics_reaggregated": True,
        "independent_production_replay": {"rows": len(rows), "conditions": checked - 12,
                                          "prediction_match": True,
                                          "max_abs_per_sample_metric_difference": maximum_metric_difference},
        "official_test_evaluated": False,
        "c2_original_status_retained": "C2_PARTIAL_PASS",
        "c3_full_physical_chain": "BLOCKED_LIGHTGEN_STACK_AND_CURRENT_MAPPING_UNKNOWN",
        "targeted_tests": test_receipt,
    }
    _atomic_json(args.output / "independent_audit.json", audit)
    print(f"AUDIT_PASS conditions_checked={checked} validation_rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
