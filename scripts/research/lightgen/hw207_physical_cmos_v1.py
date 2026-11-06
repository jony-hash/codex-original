"""HW207 fixed physical CMOS ROI geometry and calibration preparation.

This diagnostic entrypoint creates no optical intensity and never operates
hardware. A pixel-to-window affine transform must be explicit for ROI energy
integration. Detector values are per-pixel integrated signal; fractional pixel
weights allocate boundary pixels by physical area overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geometry


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_physical_cmos_v1.json"
IDENTITY = "HW207_PHYSICAL_CMOS_2X2_V1"
DEFAULT_ROIS_UM = (
    (0, 35.0, 85.0, 35.0, 85.0),
    (1, 115.0, 165.0, 35.0, 85.0),
    (2, 35.0, 85.0, 115.0, 165.0),
    (3, 115.0, 165.0, 115.0, 165.0),
)


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("identity") != IDENTITY:
        raise ValueError("unexpected HW207 physical CMOS configuration identity")
    roi_cfg = config["fixed_cmos_geometry"]
    if tuple(roi_cfg["window_um"]) != (200.0, 200.0):
        raise ValueError("the fixed physical ROI contract requires a 200 x 200 um window")
    actual = tuple(
        (item["roi"], *item["x_um"], *item["y_um"])
        for item in roi_cfg["roi_layout"]
    )
    if actual != DEFAULT_ROIS_UM:
        raise ValueError("the fixed ROI coordinates or numbering changed")
    return config


def _validated_rois(
    roi_rectangles_um: Sequence[Sequence[float]], window_um: Sequence[float]
) -> tuple[tuple[int, float, float, float, float], ...]:
    if len(window_um) != 2:
        raise ValueError("window_um must be (width, height)")
    width, height = (float(value) for value in window_um)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("window dimensions must be positive and finite")
    if len(roi_rectangles_um) != 4:
        raise ValueError("exactly four physical ROI rectangles are required")
    result = []
    for expected_id, rectangle in enumerate(roi_rectangles_um):
        if len(rectangle) != 5:
            raise ValueError("each ROI must be (id, x0, x1, y0, y1)")
        roi_id = int(rectangle[0])
        x0, x1, y0, y1 = (float(value) for value in rectangle[1:])
        if roi_id != expected_id:
            raise ValueError("ROI ids must be ordered 0, 1, 2, 3")
        if not all(math.isfinite(value) for value in (x0, x1, y0, y1)):
            raise ValueError("ROI coordinates must be finite")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI is outside the fixed physical window or empty")
        result.append((roi_id, x0, x1, y0, y1))
    for index, first in enumerate(result):
        for second in result[index + 1 :]:
            overlap_x = min(first[2], second[2]) - max(first[1], second[1])
            overlap_y = min(first[4], second[4]) - max(first[3], second[3])
            if overlap_x > 0 and overlap_y > 0:
                raise ValueError("physical ROI rectangles must not overlap")
    areas = {(x1 - x0) * (y1 - y0) for _, x0, x1, y0, y1 in result}
    if len(areas) != 1:
        raise ValueError("the four physical ROIs must have equal areas")
    return tuple(result)


def _transform_point(
    point: tuple[float, float], matrix: np.ndarray, offset: np.ndarray
) -> tuple[float, float]:
    mapped = matrix @ np.asarray(point, dtype=np.float64) + offset
    return float(mapped[0]), float(mapped[1])


def _clip_polygon(
    polygon: list[tuple[float, float]], axis: int, edge: float, keep_greater: bool
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= edge if keep_greater else previous[axis] <= edge
    for current in polygon:
        current_inside = current[axis] >= edge if keep_greater else current[axis] <= edge
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = 0.0 if denominator == 0 else (edge - previous[axis]) / denominator
            output.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def _clip_rectangle(
    polygon: list[tuple[float, float]], x0: float, x1: float, y0: float, y1: float
) -> list[tuple[float, float]]:
    polygon = _clip_polygon(polygon, 0, x0, True)
    polygon = _clip_polygon(polygon, 0, x1, False)
    polygon = _clip_polygon(polygon, 1, y0, True)
    return _clip_polygon(polygon, 1, y1, False)


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            x * polygon[(index + 1) % len(polygon)][1]
            - y * polygon[(index + 1) % len(polygon)][0]
            for index, (x, y) in enumerate(polygon)
        )
    ) / 2.0


def pixel_to_window_affine(
    sensor_shape: Sequence[int],
    pixel_pitch_um: float,
    *,
    signed_magnification: float,
    window_um: Sequence[float] = (200.0, 200.0),
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a declared scenario mapping detector pixels back to an object plane.

    This helper is for geometric scenarios and tests, not for registering the
    fixed physical CMOS ROIs. A real CMOS integration must pass a measured
    sensor-edge-pixel to CMOS-plane affine directly to ``build_roi_pixel_weights``;
    optical magnification affects the irradiance pattern arriving at CMOS, not
    the physical pixel pitch or the CMOS ROI coordinates.
    """
    if len(sensor_shape) != 2 or min(int(value) for value in sensor_shape) <= 0:
        raise ValueError("sensor_shape must contain two positive dimensions")
    rows, columns = (int(value) for value in sensor_shape)
    if len(window_um) != 2:
        raise ValueError("window_um must contain width and height")
    width_um, height_um = (float(value) for value in window_um)
    if not all(math.isfinite(value) and value > 0 for value in (width_um, height_um)):
        raise ValueError("window dimensions must be positive and finite")
    if not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0:
        raise ValueError("pixel_pitch_um must be positive and finite")
    if not math.isfinite(signed_magnification) or signed_magnification == 0:
        raise ValueError("signed_magnification must be finite and nonzero")
    if not math.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")
    scale = pixel_pitch_um / signed_magnification
    angle = math.radians(rotation_deg)
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    matrix = scale * rotation
    center = np.asarray((columns / 2.0, rows / 2.0), dtype=np.float64)
    target_center = np.asarray((width_um / 2.0, height_um / 2.0), dtype=np.float64)
    offset = target_center - matrix @ center
    return matrix, offset


def build_roi_pixel_weights(
    sensor_shape: Sequence[int],
    pixel_to_window_matrix: Sequence[Sequence[float]],
    pixel_to_window_offset_um: Sequence[float],
    *,
    roi_rectangles_um: Sequence[Sequence[float]] = DEFAULT_ROIS_UM,
    window_um: Sequence[float] = (200.0, 200.0),
) -> tuple[dict[str, Any], ...]:
    """Build sparse per-ROI fractions of detector pixels assigned by area.

    Each detector pixel value represents energy integrated over the pixel.
    A boundary pixel is allocated by the physical area of its transformed
    parallelogram intersecting the ROI. The affine is mandatory and explicit.
    """
    if len(sensor_shape) != 2:
        raise ValueError("sensor_shape must be (rows, columns)")
    rows, columns = (int(value) for value in sensor_shape)
    if min(rows, columns) <= 0:
        raise ValueError("sensor dimensions must be positive")
    matrix = np.asarray(pixel_to_window_matrix, dtype=np.float64)
    offset = np.asarray(pixel_to_window_offset_um, dtype=np.float64)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("pixel-to-window affine must be a 2x2 matrix and 2-vector")
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("pixel-to-window affine must be finite")
    determinant = float(np.linalg.det(matrix))
    pixel_area_um2 = abs(determinant)
    if pixel_area_um2 <= 1e-15:
        raise ValueError("pixel-to-window affine must be invertible")
    rois = _validated_rois(roi_rectangles_um, window_um)
    inverse = np.linalg.inv(matrix)
    result: list[dict[str, Any]] = []

    for roi_id, x0, x1, y0, y1 in rois:
        roi_polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        pixel_corners = [inverse @ (np.asarray(point) - offset) for point in roi_polygon]
        cols = [float(point[0]) for point in pixel_corners]
        row_values = [float(point[1]) for point in pixel_corners]
        col_start = max(0, math.floor(min(cols)) - 1)
        col_stop = min(columns, math.ceil(max(cols)) + 1)
        row_start = max(0, math.floor(min(row_values)) - 1)
        row_stop = min(rows, math.ceil(max(row_values)) + 1)

        roi_rows: list[int] = []
        roi_columns: list[int] = []
        fractions: list[float] = []
        for row in range(row_start, row_stop):
            for column in range(col_start, col_stop):
                cell = [
                    _transform_point(point, matrix, offset)
                    for point in (
                        (column, row),
                        (column + 1, row),
                        (column + 1, row + 1),
                        (column, row + 1),
                    )
                ]
                cell_area = _polygon_area(cell)
                overlap = _polygon_area(_clip_rectangle(cell, x0, x1, y0, y1))
                fraction = overlap / cell_area
                if fraction > 1e-12:
                    if fraction > 1.0 + 1e-10:
                        raise ArithmeticError("pixel overlap fraction exceeds one")
                    roi_rows.append(row)
                    roi_columns.append(column)
                    fractions.append(min(1.0, fraction))

        covered_area = math.fsum(fractions) * pixel_area_um2
        roi_area = (x1 - x0) * (y1 - y0)
        if not math.isclose(covered_area, roi_area, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(
                f"CMOS sensor does not fully cover ROI-{roi_id}: "
                f"covered {covered_area:.9g} of {roi_area:.9g} um^2"
            )
        result.append(
            {
                "roi": roi_id,
                "rows": np.asarray(roi_rows, dtype=np.int64),
                "columns": np.asarray(roi_columns, dtype=np.int64),
                "fractions": np.asarray(fractions, dtype=np.float64),
                "roi_area_um2": roi_area,
                "pixel_area_um2": pixel_area_um2,
                "effective_pixel_count": math.fsum(fractions),
            }
        )

    occupied: dict[tuple[int, int], float] = {}
    for roi in result:
        for row, column, fraction in zip(roi["rows"], roi["columns"], roi["fractions"]):
            key = (int(row), int(column))
            occupied[key] = occupied.get(key, 0.0) + float(fraction)
    if any(value > 1.0 + 1e-10 for value in occupied.values()):
        raise ArithmeticError("nonoverlapping ROI areas assigned more than one pixel")
    return tuple(result)


def integrate_roi_energies(
    pixel_energy: np.ndarray, roi_weights: Sequence[dict[str, Any]]
) -> np.ndarray:
    """Integrate four ROI energies from nonnegative per-pixel integrated data."""
    image = np.asarray(pixel_energy)
    if image.ndim != 2 or image.dtype.kind not in "buif":
        raise ValueError("pixel_energy must be a numeric two-dimensional array")
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError("pixel_energy must be finite and nonnegative; apply a measured background model first")
    if len(roi_weights) != 4 or [item.get("roi") for item in roi_weights] != [0, 1, 2, 3]:
        raise ValueError("ordered weights for ROI 0, 1, 2, 3 are required")
    scores = []
    for item in roi_weights:
        rows = item["rows"]
        columns = item["columns"]
        if (rows.size == 0 or np.any(rows < 0) or np.any(rows >= image.shape[0])
                or np.any(columns < 0) or np.any(columns >= image.shape[1])):
            raise ValueError("ROI pixel weights are empty or outside the sensor")
        scores.append(float(np.sum(image[rows, columns].astype(np.float64) * item["fractions"])))
    return np.asarray(scores, dtype=np.float64)


def predict_from_roi_energies(energies: Sequence[float]) -> int | None:
    values = np.asarray(energies, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("four finite nonnegative ROI energies are required")
    ordered = np.sort(values)
    if ordered[-1] <= 0:
        return None
    # The tie rule must not depend on the arbitrary units or global scale of
    # the detector signal. Relative tolerance preserves argmax under any
    # positive rescaling while still rejecting round-off-level ties.
    tolerance = 16 * np.finfo(np.float64).eps * float(ordered[-1])
    if ordered[-1] - ordered[-2] <= tolerance:
        return None
    return int(np.argmax(values))


def fixed_roi_classification_metrics(
    true_labels: Sequence[int], predictions: Sequence[int | None]
) -> dict[str, Any]:
    """Summarize four-ROI decisions; unresolved ties/zero-energy count as errors."""
    labels = np.asarray(true_labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("true_labels must be a nonempty one-dimensional integer sequence")
    if np.any((labels < 0) | (labels > 3)):
        raise ValueError("true_labels must be in the fixed four-class range 0..3")

    raw_predictions = np.asarray(predictions, dtype=object)
    if raw_predictions.shape != labels.shape:
        raise ValueError("predictions must have the same one-dimensional shape as true_labels")
    normalized: list[int] = []
    for value in raw_predictions.tolist():
        if value is None:
            normalized.append(-1)
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("predictions must be class ids 0..3 or None for abstention")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("predictions must be class ids 0..3 or None for abstention") from exc
        if not math.isfinite(numeric) or numeric != math.floor(numeric):
            raise ValueError("predictions must be class ids 0..3 or None for abstention")
        prediction = int(numeric)
        if prediction not in (-1, 0, 1, 2, 3):
            raise ValueError("predictions must be class ids 0..3 or -1 for abstention")
        normalized.append(prediction)

    confusion = np.zeros((4, 4), dtype=np.int64)
    abstentions = np.zeros(4, dtype=np.int64)
    for label, prediction in zip(labels.tolist(), normalized):
        if prediction < 0:
            abstentions[int(label)] += 1
        else:
            confusion[int(label), prediction] += 1

    per_class: list[dict[str, Any]] = []
    defined_f1: list[float] = []
    for class_id in range(4):
        true_positive = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum() + abstentions[class_id])
        predicted_count = int(confusion[:, class_id].sum())
        recall = true_positive / support if support else None
        precision = true_positive / predicted_count if predicted_count else (0.0 if support else None)
        denominator = support + predicted_count
        f1 = 2.0 * true_positive / denominator if denominator else None
        if f1 is not None:
            defined_f1.append(f1)
        per_class.append(
            {
                "class": class_id,
                "support": support,
                "correct": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    correct = sum(int(confusion[index, index]) for index in range(4))
    unresolved = int(sum(abstentions))
    return {
        "sample_count": int(labels.size),
        "correct_count": correct,
        "accuracy": correct / int(labels.size),
        "macro_f1": float(np.mean(defined_f1)) if defined_f1 else None,
        "unresolved_count": unresolved,
        "unresolved_fraction": unresolved / int(labels.size),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "abstentions_by_true_class": abstentions.tolist(),
        "per_class": per_class,
        "abstentions_count_as_incorrect": True,
    }


def intensity_concentration_metrics(
    intensity: np.ndarray, pixel_pitch_um: float
) -> dict[str, float]:
    """Return full-map RMS radius and inverse-participation effective area.

    RMS radius is about the intensity centroid. Effective area is
    ``(sum(I)**2 / sum(I**2)) * pixel_area``; it is a concentration statistic,
    not a fitted spot boundary or a claim about the physical PSF.
    """
    image = np.asarray(intensity)
    pitch = float(pixel_pitch_um)
    if image.ndim != 2 or image.size == 0 or image.dtype.kind not in "buif":
        raise ValueError("intensity must be a nonempty numeric two-dimensional array")
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("pixel_pitch_um must be positive and finite")
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError("intensity must be finite and nonnegative")

    values = image.astype(np.float64, copy=False)
    total = float(np.sum(values, dtype=np.float64))
    if total <= 0:
        raise ValueError("intensity must have positive total energy")
    squared_total = float(np.sum(np.square(values), dtype=np.float64))
    if not math.isfinite(squared_total) or squared_total <= 0:
        raise ValueError("intensity square sum must be positive and finite")

    row_profile = np.sum(values, axis=1, dtype=np.float64)
    column_profile = np.sum(values, axis=0, dtype=np.float64)
    row_centers = np.arange(image.shape[0], dtype=np.float64) + 0.5
    column_centers = np.arange(image.shape[1], dtype=np.float64) + 0.5
    centroid_row = float(np.dot(row_profile, row_centers) / total)
    centroid_column = float(np.dot(column_profile, column_centers) / total)
    second_moment_pixels2 = float(
        np.dot(row_profile, np.square(row_centers - centroid_row))
        + np.dot(column_profile, np.square(column_centers - centroid_column))
    ) / total
    pixel_area = pitch**2
    effective_pixel_count = total**2 / squared_total
    return {
        "intensity_rms_radius_um": math.sqrt(max(second_moment_pixels2, 0.0)) * pitch,
        "inverse_participation_ratio_per_pixel": squared_total / (total**2),
        "effective_pixel_count": effective_pixel_count,
        "effective_area_um2": effective_pixel_count * pixel_area,
        "effective_area_fraction_of_map": effective_pixel_count / int(image.size),
    }


def nominal_geometry_summary(config: dict[str, Any]) -> dict[str, Any]:
    optics = config["optics"]
    camera = optics["cmos"]
    pixel_um = float(camera["pixel_pitch_um"])
    rows, columns = int(camera["rows"]), int(camera["columns"])
    active_um = (rows * pixel_um, columns * pixel_um)
    window_um = tuple(config["fixed_cmos_geometry"]["window_um"])
    roi_side_um = 50.0
    dmd_pitch = float(optics["dmd"]["micromirror_pitch_um"])
    target_pixel = float(optics["phase_layer"]["target_pitch_um"])
    relay = config["relay_diagnostic_scenarios"]
    dmd_comparisons = []
    for name, scenario in relay.items():
        magnification = abs(float(scenario["magnification_signed"]))
        projected_pitch = dmd_pitch * magnification
        dmd_comparisons.append(
            {
                "scenario": name,
                "signed_magnification": scenario["magnification_signed"],
                "projected_mirror_pitch_um": projected_pitch,
                "mirrors_per_200um_axis": 200.0 / projected_pitch,
                "mirrors_per_2um_phase_pixel": target_pixel / projected_pitch,
                "status": "IDEAL_4F_CALCULATION_NOT_MEASURED_RELAY",
            }
        )

    input_matrix = geometry.ray_matrix(230.0, 350.0, 150.0, 150.0, 150.0)
    output_at_100mm = geometry.ray_matrix(80.0, 0.0, 100.0, 30.0)
    output_conjugate_mm = 30.0 * 80.0 / (80.0 - 30.0)
    output_at_conjugate = geometry.ray_matrix(80.0, 0.0, output_conjugate_mm, 30.0)
    object_for_camera_at_100mm = 30.0 * 100.0 / (100.0 - 30.0)
    return {
        "sensor_active_area_um": list(active_um),
        "sensor_active_area_mm": [value / 1000.0 for value in active_um],
        "fixed_window_um": list(window_um),
        "fixed_window_area_um2": window_um[0] * window_um[1],
        "roi_side_um": roi_side_um,
        "roi_area_um2": roi_side_um**2,
        "four_roi_total_area_um2": 4 * roi_side_um**2,
        "four_roi_fraction_of_window": 4 * roi_side_um**2 / (window_um[0] * window_um[1]),
        "fixed_cmos_pixel_equivalents_from_reference_pitch": {
            "window_side_px": window_um[0] / pixel_um,
            "roi_side_px": roi_side_um / pixel_um,
            "gap_px": 30.0 / pixel_um,
            "symmetric_border_px": 35.0 / pixel_um,
            "status": "CMOS_PHYSICAL_GEOMETRY_FROM_REFERENCE_PIXEL_PITCH; NOT_CAMERA_REGISTRATION",
        },
        "nominal_0p6x_if_phase_layer_is_object": {
            "projected_200um_object_field_side_px": window_um[0] * 0.6 / pixel_um,
            "projected_50um_object_feature_side_px": roi_side_um * 0.6 / pixel_um,
            "fixed_cmos_roi_side_px": roi_side_um / pixel_um,
            "status": "IDEAL_THIN_LENS_CASE_ONLY; f30, s=80 mm gives s'=48 mm, not the historical 100 mm plane",
        },
        "dmd_relay_sampling_comparisons": dmd_comparisons,
        "historical_estimate_input_relay_abcd": [list(input_matrix[0]), list(input_matrix[1])],
        "historical_estimate_input_relay_conjugate_B_mm": input_matrix[0][1],
        "historical_estimate_output_at_100mm_abcd": [list(output_at_100mm[0]), list(output_at_100mm[1])],
        "ideal_output_conjugate_for_s80_f30_mm": output_conjugate_mm,
        "ideal_output_magnification_at_conjugate": output_at_conjugate[0][0],
        "object_distance_if_camera_is_100mm_from_f30_lens_mm": object_for_camera_at_100mm,
        "all_abcd_values_status": "THIN_LENS_AND_HISTORICAL_DISTANCE_DIAGNOSTICS_ONLY",
        "classification_plane_status": "UNKNOWN",
    }


def geometry_sensitivity_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate explicit ideal-geometry scenarios, never inferred hardware values."""
    optics = config["optics"]
    camera = optics["cmos"]
    dmd_pitch_um = float(optics["dmd"]["micromirror_pitch_um"])
    phase_pitch_um = float(optics["phase_layer"]["target_pitch_um"])
    sensor_pitch_um = float(camera["pixel_pitch_um"])
    relay_cases = []
    for name, item in config["relay_diagnostic_scenarios"].items():
        magnification = abs(float(item["magnification_signed"]))
        projected_pitch_um = dmd_pitch_um * magnification
        relay_cases.append(
            {
                "scenario": name,
                "magnification_abs": magnification,
                "dmd_pitch_at_phase_um": projected_pitch_um,
                "dmd_periods_across_200um": 200.0 / projected_pitch_um,
                "phase_pixels_per_projected_mirror_period": projected_pitch_um / phase_pitch_um,
                "mnist_28px_nominal_pitch_um": 200.0 / 28.0,
                "dmd_periods_per_nominal_mnist_pixel": (200.0 / 28.0) / projected_pitch_um,
                "status": "IDEAL_4F_SCENARIO_ONLY_NOT_CURRENT_RELAY_CALIBRATION",
            }
        )

    output = config["output_lens_sensitivity_scenarios"]
    focal_length_mm = float(output["focal_length_mm"])
    reference_camera_distance_mm = float(output["camera_distance_reference_mm"])
    output_cases = []
    for object_distance_mm in output["object_distances_mm"]:
        object_distance_mm = float(object_distance_mm)
        if object_distance_mm <= focal_length_mm:
            raise ValueError("thin-lens sensitivity object distances must exceed focal length")
        image_distance_mm = (
            focal_length_mm * object_distance_mm / (object_distance_mm - focal_length_mm)
        )
        magnification_abs = image_distance_mm / object_distance_mm
        image_plane_offset_mm = image_distance_mm - reference_camera_distance_mm
        if abs(image_plane_offset_mm) < 1e-12:
            image_plane_offset_mm = 0.0
        output_cases.append(
            {
                "object_distance_from_pi_cls_mm": object_distance_mm,
                "image_distance_from_principal_plane_mm": image_distance_mm,
                "magnification_signed": -magnification_abs,
                "image_plane_offset_from_100mm_reference_mm": image_plane_offset_mm,
                "projected_50um_object_feature_width_sensor_pixels": 50.0 * magnification_abs / sensor_pitch_um,
                "fixed_cmos_roi_width_pixels": 50.0 / sensor_pitch_um,
                "projected_window_width_sensor_pixels": 200.0 * magnification_abs / sensor_pitch_um,
                "scenario_basis": output["scenario_basis"][str(object_distance_mm)],
                "status": "IDEAL_THIN_LENS_SCENARIO_ONLY_NOT_MEASURED_CONJUGACY",
            }
        )
    return {
        "status": "ILLUSTRATIVE_SENSITIVITY_ONLY",
        "uncertainty_intervals_measured": False,
        "relay_scenarios": relay_cases,
        "f30_thin_lens_scenarios": output_cases,
        "limitations": [
            "No current optical reference-plane distances, actual focal lengths, apertures, aberrations, or Pi_cls position were measured.",
            "The 80 mm value refers to a historical phase-layer-position estimate and does not establish that Pi_cls is at the phase surface.",
            "The pixel counts are geometric conversions and do not model PSF, diffraction, detector noise, or camera registration.",
        ],
    }


def inspect_phase_candidate_arrays(
    config: dict[str, Any], repo_root: Path = ROOT
) -> list[dict[str, Any]]:
    """Read small saved phase/height arrays for provenance metadata only; never propagate them."""
    expected_shape = tuple(int(value) for value in config["optics"]["phase_layer"]["target_grid"])
    results: list[dict[str, Any]] = []
    for candidate in config["phase_candidates_to_audit_when_model_is_calibrated"]:
        record: dict[str, Any] = {
            "name": candidate["name"],
            "source_type": candidate["source_type"],
            "phase_path": candidate["path"],
            "source_report": candidate.get("source_report"),
            "source_report_status": "NOT_PROVIDED" if "source_report" not in candidate else "NOT_CHECKED",
            "phase_array_status": "NOT_INSPECTED",
            "expected_shape_rows_columns": list(expected_shape),
            "height_path": candidate.get("height_path"),
            "height_array_status": "NOT_APPLICABLE" if "height_path" not in candidate else "NOT_INSPECTED",
            "propagation_or_roi_evaluation_performed": False,
            "physical_complex_transmission_verified": False,
        }
        if candidate.get("source_report"):
            report_path = (repo_root / candidate["source_report"]).resolve()
            try:
                report_path.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ValueError(f"candidate report path escapes the repository: {candidate['source_report']}") from exc
            record["source_report_status"] = "PRESENT" if report_path.is_file() else "MISSING"
        for key, path_key, status_key in (
            ("phase", "path", "phase_array_status"),
            ("height", "height_path", "height_array_status"),
        ):
            relative = candidate.get(path_key)
            if relative is None:
                continue
            source_path = (repo_root / relative).resolve()
            try:
                source_path.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ValueError(f"candidate path escapes the repository: {relative}") from exc
            if not source_path.is_file():
                record[status_key] = "MISSING"
                continue
            size_bytes = source_path.stat().st_size
            item = {"path": relative, "bytes": size_bytes}
            if size_bytes > 10_000_000:
                record[status_key] = "UNEXPECTEDLY_LARGE_NOT_LOADED"
                record[f"{key}_array"] = item
                continue
            values = np.load(source_path, mmap_mode="r", allow_pickle=False)
            if not isinstance(values, np.ndarray) or values.dtype.kind not in "buif":
                record[status_key] = "INVALID_NONNUMERIC_OR_NON_NPY"
                record[f"{key}_array"] = item
                continue
            item.update(
                {
                    "shape_rows_columns": list(values.shape),
                    "dtype": str(values.dtype),
                    "finite": bool(np.isfinite(values).all()),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
            record[f"{key}_array"] = item
            if not item["finite"]:
                record[status_key] = "INVALID_NONFINITE"
            elif key == "phase" and tuple(values.shape) != expected_shape:
                record[status_key] = "SHAPE_DIFFERS_FROM_TARGET_GRID"
            else:
                record[status_key] = "PRESENT_METADATA_ONLY"
        if candidate.get("height_path"):
            phase = record.get("phase_array", {})
            height = record.get("height_array", {})
            record["phase_height_shape_match"] = (
                phase.get("shape_rows_columns") == height.get("shape_rows_columns")
                if phase and height else False
            )
            record["phase_height_physical_relation_verified"] = False
        results.append(record)
    return results


def make_dmd_calibration_patterns() -> dict[str, np.ndarray]:
    rows, columns = 1600, 2560
    row = np.arange(rows, dtype=np.int32)[:, None]
    column = np.arange(columns, dtype=np.int32)[None, :]
    off = np.zeros((rows, columns), dtype=np.uint8)
    on = np.full((rows, columns), 255, dtype=np.uint8)
    checker = ((((row // 32) + (column // 32)) % 2) * 255).astype(np.uint8)
    vertical = np.broadcast_to((((column // 8) % 2) * 255).astype(np.uint8),
                               (rows, columns)).copy()
    horizontal = np.broadcast_to((((row // 8) % 2) * 255).astype(np.uint8),
                                 (rows, columns)).copy()
    fiducials = np.zeros((rows, columns), dtype=np.uint8)
    # Unequal L, single cross, double-dot and bar marks make flips/rotation identifiable.
    fiducials[180:430, 180:220] = 255
    fiducials[390:430, 180:470] = 255
    fiducials[270:330, 1180:1380] = 255
    fiducials[900:970, 1810:1880] = 255
    fiducials[1080:1130, 2020:2080] = 255
    fiducials[1080:1130, 2140:2200] = 255
    fiducials[1370:1390, 400:680] = 255
    return {
        "00_full_off": off,
        "01_full_on": on,
        "02_checkerboard_32_mirror_cells": checker,
        "03_vertical_grating_16_mirror_period": vertical,
        "04_horizontal_grating_16_mirror_period": horizontal,
        "05_asymmetric_fiducials": fiducials,
    }


def _mpl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _save_triplet(fig: Any, path_without_suffix: Path, *, dpi: int = 300) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")


def draw_cmos_roi_geometry(path_without_suffix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _mpl_style()
    colors = ("#3775BA", "#D89A28", "#42949E", "#B64342")
    fig = plt.figure(figsize=(10.2, 5.6), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1000)
    ax.set_ylim(540, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    sensor_x, sensor_y, sensor_side = 35.0, 95.0, 400.0
    ax.add_patch(Rectangle((sensor_x, sensor_y), sensor_side, sensor_side,
                           facecolor="#FAFBFC", edgecolor="#2B3540", linewidth=1.3))
    active_mm = 13.312
    window_side = sensor_side * 0.2 / active_mm
    window_x = sensor_x + (sensor_side - window_side) / 2
    window_y = sensor_y + (sensor_side - window_side) / 2
    ax.add_patch(Rectangle((window_x, window_y), window_side, window_side,
                           facecolor="none", edgecolor="#D87800", linewidth=1.5))
    ax.text(sensor_x + sensor_side / 2, sensor_y - 20,
            "13.312 mm × 13.312 mm active sensor", ha="center", va="bottom", fontsize=9)
    ax.annotate("200 µm window\nshown at scale", xy=(window_x + window_side / 2, window_y + window_side / 2),
                xytext=(450, 295), ha="left", va="center", fontsize=7.0, color="#7A4B13",
                arrowprops={"arrowstyle": "-", "color": "#D87800", "lw": 0.8})

    detail_x, detail_y, detail_side = 570.0, 105.0, 360.0
    ax.add_patch(Rectangle((detail_x, detail_y), detail_side, detail_side,
                           facecolor="#FFFFFF", edgecolor="#2B3540", linewidth=1.2))
    scale = detail_side / 200.0
    rois = _validated_rois(DEFAULT_ROIS_UM, (200.0, 200.0))
    for roi_id, x0, x1, y0, y1 in rois:
        x = detail_x + x0 * scale
        y = detail_y + y0 * scale
        w = (x1 - x0) * scale
        h = (y1 - y0) * scale
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[roi_id] + "22",
                               edgecolor=colors[roi_id], linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, f"ROI {roi_id}\n50 × 50 µm",
                ha="center", va="center", fontsize=8, color="#20262D")
    ax.text(detail_x + detail_side / 2, detail_y - 20,
            "Fixed window coordinates (µm; X → right, Y → down)",
            ha="center", va="bottom", fontsize=8.5)
    ax.text(detail_x + detail_side / 2, detail_y + detail_side + 20,
            "Margins 35 µm  ·  gaps 30 µm", ha="center", va="top", fontsize=8)
    ax.text(sensor_x, sensor_y + sensor_side + 10,
            "Window placement is illustrative; pixel registration is UNKNOWN",
            ha="left", va="top", fontsize=7.0, color="#7A4B13")
    ax.text(25, 44, "a", fontsize=11, fontweight="bold", va="top")
    ax.text(560, 44, "b", fontsize=11, fontweight="bold", va="top")
    fig.text(0.04, 0.035,
             "CMOS reference-pitch geometry: window 30.77 px; each ROI 7.69 px; gap 4.62 px; margins 5.38 px.",
             ha="left", va="bottom", fontsize=7.2, color="#333333")
    fig.text(0.04, 0.012,
             "Optical magnification changes the projected spot, not the fixed physical CMOS ROI; registration and PSF remain unmeasured.",
             ha="left", va="bottom", fontsize=7.2, color="#333333")
    _save_triplet(fig, path_without_suffix)
    plt.close(fig)


def draw_geometry_sensitivity(path_without_suffix: Path, sensitivity: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), facecolor="white")
    magnification = np.linspace(0.15, 1.2, 300)
    feature_pixels = 50.0 * magnification / 6.5
    axes[0].plot(magnification, feature_pixels, color="#275D8C", linewidth=1.8,
                 label="50 µm feature in Π_cls projected to CMOS")
    fixed_roi_pixels = 50.0 / 6.5
    axes[0].axhline(fixed_roi_pixels, color="#B24A35", linestyle="--", linewidth=1.0,
                    label=f"fixed 50 µm CMOS ROI = {fixed_roi_pixels:.2f} px")
    cases = sensitivity["f30_thin_lens_scenarios"]
    for case in cases:
        m_abs = abs(case["magnification_signed"])
        if 0.15 <= m_abs <= 1.2:
            axes[0].scatter(m_abs, case["projected_50um_object_feature_width_sensor_pixels"],
                            s=25, zorder=3, color="#C2512D")
            axes[0].annotate(f"s={case['object_distance_from_pi_cls_mm']:.1f} mm",
                             (m_abs, case["projected_50um_object_feature_width_sensor_pixels"]),
                             xytext=(4, 5), textcoords="offset points", fontsize=6.4)
    axes[0].set_xlabel("Absolute output magnification |M|")
    axes[0].set_ylabel("Width on 6.5 µm CMOS pixels")
    axes[0].set_title("Projected feature vs fixed CMOS ROI", loc="left", fontweight="bold")
    axes[0].grid(True, linewidth=0.45, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    axes[0].text(0.02, 0.02, "Object-plane feature changes width; physical CMOS ROI does not",
                 transform=axes[0].transAxes, fontsize=6.4, color="#555555")

    object_distance = np.linspace(35.0, 110.0, 400)
    focal_length = 30.0
    image_distance = focal_length * object_distance / (object_distance - focal_length)
    axes[1].plot(object_distance, image_distance, color="#275D8C", linewidth=1.8,
                 label="Thin lens, f = 30 mm")
    axes[1].plot(object_distance, object_distance, color="#888888", linestyle="--",
                 linewidth=0.9, label="1:1 reference (s′ = s)")
    for case in cases:
        s = case["object_distance_from_pi_cls_mm"]
        sp = case["image_distance_from_principal_plane_mm"]
        axes[1].scatter(s, sp, s=23, zorder=3, color="#C2512D")
    axes[1].scatter([60.0], [60.0], marker="o", facecolors="none", edgecolors="#C2512D",
                    s=65, linewidths=1.1, zorder=4)
    axes[1].annotate("1:1 at 60/60 mm", (60.0, 60.0), xytext=(8, 8),
                     textcoords="offset points", fontsize=6.7)
    axes[1].annotate("80 mm object → 48 mm image",
                     (80.0, 30.0 * 80.0 / (80.0 - 30.0)),
                     xytext=(8, -16), textcoords="offset points", fontsize=6.4)
    axes[1].annotate("42.86 mm object → 100 mm image",
                     (30.0 * 100.0 / (100.0 - 30.0), 100.0),
                     xytext=(-95, 8), textcoords="offset points", fontsize=6.4)
    axes[1].set_xlabel("Object distance from Π_cls, s (mm)")
    axes[1].set_ylabel("Image distance from lens principal plane, s′ (mm)")
    axes[1].set_title("Conjugacy depends on Π_cls position", loc="left", fontweight="bold")
    axes[1].grid(True, linewidth=0.45, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=6.3, loc="upper right")
    fig.suptitle("Illustrative thin-lens and pixel-sampling sensitivity — not measured HW207 geometry",
                 fontsize=9.2, y=1.02)
    fig.tight_layout()
    _save_triplet(fig, path_without_suffix)
    plt.close(fig)


def draw_topology(path_without_suffix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _mpl_style()
    fig = plt.figure(figsize=(12.0, 4.9), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 430)
    ax.axis("off")
    top = [
        "Laser\n532 nm nominal",
        "Beam\nexpander",
        "Mirror-1\nfold",
        "PF-1\npolarization",
        "PBS",
        "Beam\nsplitter",
    ]
    bottom = [
        "DMD\nF9000 / DLP9000",
        "Input relay\n2 × f150 mm",
        "LightGen phase\n100 × 100 target",
        "Π_cls\nposition unknown",
        "Lens\nf = 30 mm nominal",
        "CMOS\n2048² × 6.5 µm",
    ]
    top_centers = np.linspace(85, 915, 6)
    bottom_centers = top_centers[::-1]
    box_w, box_h = 128.0, 76.0

    def draw_box(center_x: float, y: float, label: str, *, unknown: bool = False) -> None:
        patch = FancyBboxPatch(
            (center_x - box_w / 2, y), box_w, box_h,
            boxstyle="round,pad=5,rounding_size=5",
            facecolor="#FFF8EC" if unknown else "#F4F6F8",
            edgecolor="#C27500" if unknown else "#425466",
            linewidth=1.0,
            linestyle="--" if unknown else "-",
        )
        ax.add_patch(patch)
        ax.text(center_x, y + box_h / 2, label, ha="center", va="center", fontsize=8.1)

    for index, (center, label) in enumerate(zip(top_centers, top)):
        draw_box(float(center), 275.0, label, unknown=True)
        if index < len(top) - 1:
            ax.add_patch(FancyArrowPatch((center + box_w / 2, 313),
                                         (top_centers[index + 1] - box_w / 2, 313),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                         linestyle="--", color="#687482"))
    for index, (center, label) in enumerate(zip(bottom_centers, bottom)):
        draw_box(float(center), 90.0, label, unknown=True)
        if index < len(bottom) - 1:
            ax.add_patch(FancyArrowPatch((center - box_w / 2, 128),
                                         (bottom_centers[index + 1] + box_w / 2, 128),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                         linestyle="--", color="#687482"))
    ax.add_patch(FancyArrowPatch((top_centers[-1], 275), (bottom_centers[0], 170),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 linestyle="--", color="#687482"))
    ax.text(500, 405,
            "Upper path: historical 2026-11-03 layout record; current ports and geometry not reverified",
            ha="center", va="center", fontsize=8.2, color="#68450E")
    ax.text(500, 195,
            "Lower chain order: user-confirmed assembled classifier path; component identities and distances remain uncalibrated",
            ha="center", va="center", fontsize=7.8, color="#68450E")
    ax.text(500, 34,
            "Dashed arrows indicate the stated propagation order, not measured ray directions. Amber boxes mark unverified identity/geometry; Π_cls is unknown.",
            ha="center", va="center", fontsize=7.8, color="#555555")
    _save_triplet(fig, path_without_suffix)
    plt.close(fig)


def save_dmd_patterns(output_dir: Path, plate_path: Path) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    patterns = make_dmd_calibration_patterns()
    metadata: dict[str, dict[str, Any]] = {}
    for name, pixels in patterns.items():
        path = output_dir / f"{name}.png"
        Image.fromarray(pixels).save(path, format="PNG")
        metadata[name] = {
            "file": path.name,
            "shape_rows_columns": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "png_mode": "L",
            "gray_levels": sorted(int(v) for v in np.unique(pixels)),
            "on_fraction": float(np.mean(pixels > 0)),
            "use": "static binary calibration input; no optical capture is included",
        }

    _mpl_style()
    fig = plt.figure(figsize=(10.0, 5.8), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 520)
    ax.axis("off")
    x_positions = (35.0, 355.0, 675.0)
    y_positions = (300.0, 65.0)
    tile_w, tile_h = 290.0, 181.25
    for index, (name, pixels) in enumerate(patterns.items()):
        row, col = divmod(index, 3)
        x, y = x_positions[col], y_positions[1 - row]
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(x, x + tile_w, y + tile_h, y), origin="upper")
        ax.add_patch(FancyBboxPatch((x, y), tile_w, tile_h,
                                    boxstyle="square,pad=0", facecolor="none",
                                    edgecolor="#45515D", linewidth=0.7))
        ax.text(x + tile_w / 2, y - 12, name.replace("_", " "),
                ha="center", va="top", fontsize=7.5)
    ax.text(500, 500, "Static DMD calibration inputs · native 2560 × 1600 pixels · 1-bit values",
            ha="center", va="center", fontsize=9)
    ax.text(500, 15,
            "Patterns establish orientation, scale, and contrast only after manual optical capture; they are not CMOS measurements.",
            ha="center", va="center", fontsize=7.3, color="#555555")
    _save_triplet(fig, plate_path)
    plt.close(fig)
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_distance_csv(path: Path, distances: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("segment", "from", "to", "value_mm", "source", "current", "reference_surface", "uncertainty")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in distances:
            writer.writerow({
                **item,
                "reference_surface": "photo/mechanical estimate; not a lens principal plane",
                "uncertainty": "not quantified; historical pre-install position estimate",
            })


def _write_current_distance_template(path: Path) -> None:
    rows = (
        ("01", "Laser output reference", "Beam expander input reference"),
        ("02", "Beam expander output reference", "Mirror-1 reflecting surface"),
        ("03", "Mirror-1 reflecting surface", "PF-1 reference surface"),
        ("04", "PF-1 reference surface", "PBS input reference"),
        ("05", "PBS used output reference", "Beam splitter used input reference"),
        ("06", "Beam splitter used output reference", "DMD micromirror plane"),
        ("07", "DMD micromirror plane", "Input relay L2 principal plane"),
        ("08", "Input relay L2 principal plane", "Input relay L1 principal plane"),
        ("09", "Input relay L1 principal plane", "PF-2 reference surface"),
        ("10", "PF-2 reference surface", "LightGen phase surface"),
        ("11", "LightGen phase surface", "Classification plane Π_cls"),
        ("12", "Classification plane Π_cls", "f30 lens object-side principal plane"),
        ("13", "f30 lens image-side principal plane", "CMOS active sensor plane"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "segment", "from_reference", "to_reference", "value_mm", "measurement_method",
        "reference_surface_confirmation", "uncertainty_mm", "source", "confidence",
        "used_in_forward_model", "status",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for segment, start, end in rows:
            writer.writerow(
                {
                    "segment": segment,
                    "from_reference": start,
                    "to_reference": end,
                    "value_mm": "",
                    "measurement_method": "",
                    "reference_surface_confirmation": "PENDING",
                    "uncertainty_mm": "",
                    "source": "no current-session measurement record supplied",
                    "confidence": "UNKNOWN",
                    "used_in_forward_model": "NO",
                    "status": "CALIBRATION_REQUIRED",
                }
            )


def _write_acquisition_protocol(path: Path, pattern_metadata: dict[str, Any]) -> None:
    pattern_lines = "\n".join(
        f"- `{name}.png` — {item['shape_rows_columns'][1]}×{item['shape_rows_columns'][0]} native DMD pixels; ON fraction {item['on_fraction']:.4f}."
        for name, item in pattern_metadata.items()
    )
    content = f"""# HW207 手动标定采集说明

本文件只准备采集顺序；本轮没有打开激光、DMD或CMOS，也没有采集硬件数据。请实验人员在已批准的激光安全规程和硬件控制流程内执行，并保留原始帧、设备设置和时间戳。

## DMD 输入端

静态输入图案：

{pattern_lines}

1. 先记录 F9000 铭牌/序列号、DLPC900 固件、DMD 安装角、图案位深、ON/OFF 定义、触发配置和相机型号/序列号。
2. 使用安全、受控的静态二值图案采集全 OFF、全 ON；报告暗背景、信号、饱和像素比例和重复帧波动。
3. 采集非对称 fiducials 的至少 3 个重复帧；以已知 DMD 行列坐标拟合 DMD 到 200 µm 输入窗口的仿射矩阵、平移、行列翻转和旋转。另留未参与拟合的点计算残差。
4. 采集横向与纵向光栅及 checkerboard，报告每方向投影周期、调制度和可分辨最高空间频率；据此评价 Relay 的有效带宽，不把标称倍率等同于分辨率。
5. 对 PBS/BS 的各输出端分别记录 ON/OFF 光束方向与功率比例；不要把临时反射镜路径当作 DMD 反射几何。

## 输出平面与 30 mm 镜头

1. 以镜片主平面或可追踪机械基准记录 LightGen、候选分类面、f30 镜头和 CMOS 的轴向距离、横向偏心、倾角与口径；每段记录量具、参考面、不确定度和重复测量。
2. 在不改变衍射相位的条件下，用固定点/光栅输入对候选分类面做预先声明的轴向扫描，保存每个位置的完整原始 CMOS 帧；不得按测试标签挑选位置。
3. 通过像面标尺或已知点距的标定靶独立拟合实际倍率、旋转、翻转和畸变；把未参与拟合的点作为注册残差检查。
4. 30 mm 理想薄透镜的 1:1 参考是物距、像距各60 mm（相对主平面）。这是计算参考，不是现有硬件距离。

## CMOS 读出

1. 用铭牌/设备枚举确认相机整机型号、传感器、位深、快门、binning/ROI、增益、曝光和触发；保留 SDK 元数据。
2. 在每个计划曝光和增益下采集不少于100帧暗场；在不饱和均匀照明下采集不少于100帧平场。报告暗均值/方差、平场响应、坏点、读出噪声估计、满阱/饱和比例和帧间漂移。
3. 532 nm 的 QE、光功率到电子数/数字量转换和曝光同步须来自相机校准或独立功率测量；峰值 QE@595 nm 不等于 QE@532 nm。
4. 固定200×200 µm窗口和四个50×50 µm ROI 的窗口到像素仿射变换后，将它冻结；每次采集连同变换、ROI、曝光、暗场/平场版本一起记录。

## 相位候选与数据角色

每个候选需绑定实际安装器件的高度/表面或复透射测量。C3.5/C3.7/C3.8 已用开发人口只可作开发结果；官方 MNIST test 已有历史使用记录，不能再冒充独立测试。独立评价样本须先经数据使用台账核实并在采集/推理前冻结。

推荐每次保存：原始未裁剪帧、相机元数据、暗/平场帧、DMD命令、触发日志、器件与几何记录、温度/功率参考、样本原始索引/标签、相位/高度来源和校准变换。不要只保存裁剪图或截图。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_report(
    path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    sensitivity: dict[str, Any],
    candidate_inventory: Sequence[dict[str, Any]],
) -> None:
    run_id = config["session_source_id"]
    content = f"""# HW207 物理 CMOS 固定 ROI：本会话执行记录

来源标识：`{run_id}`  
状态：**PREFLIGHT_BLOCKED / DIAGNOSTIC_ONLY**

## 结果与状态

固定 200×200 µm 窗口和四个50×50 µm ROI已编码；采集图案、几何换算、1:1输出成像参考和面积重叠积分程序已生成。尚未建立经过测量的 P0–P4 光学/CMOS模型，因此本轮不运行候选传播、冻结相位比较或相位优化。此状态是输入证据不足，不是有效光学失败。

## 科学认识

当前C3.5/C3.7/C3.8数值结果对应约669.5 µm边长的ROI，不能推出本计划50 µm ROI结果。按 CMOS 参考像元间距6.5 µm换算，固定在感光面上的50 µm ROI宽{summary['fixed_cmos_pixel_equivalents_from_reference_pitch']['roi_side_px']:.3f}个像元，200 µm窗口宽{summary['fixed_cmos_pixel_equivalents_from_reference_pitch']['window_side_px']:.3f}个像元；这些尺寸不随输出光学倍率改变。倍率会改变分类面光斑映到传感器后的尺寸与位置；当前倍率、PSF、像元响应、串扰、读噪和相机实测注册仍未知。

用历史照片估读暂代主平面距离时，150/150 mm relay 的估读ABCD矩阵有 B={summary['historical_estimate_input_relay_conjugate_B_mm']:.3f} mm，非共轭；f30、物距80 mm的薄透镜像距为{summary['ideal_output_conjugate_for_s80_f30_mm']:.3f} mm，不是历史相机位置100 mm。由于这些距离不是当前主平面测量且分类面 Π_cls 未定位，这些量只指出待校准接口，不能认定实物失焦或倍率。

## 目标缺口

- P0：接受“已装好并接通”为用户确认状态；实际型号/序列号、当前端口方向、器件数量和当下参考面距离仍缺独立记录。两份BOM对 f30 镜头采购数记录为2与3，采购量不等同于安装数量。
- P1：没有当前DMD到芯片输入面的标定帧、仿射映射、ON/OFF收光方向或有效空间分辨率测量。
- P2：分类平面Π_cls、主平面距离、实际像差/口径/PSF及30 mm镜头共轭未测。
- P3：相机整机、读出模式、暗平场、增益、曝光、532 nm响应和窗口到像素映射未闭合。
- P4：固定物理ROI坐标已冻结；映射矩阵及有效探测读出尚未标定。
- P5：MNIST开发人口已用于C3.5–C3.8选择/评估，官方测试集也有历史使用记录；在数据台账未确认新独立人口前，不生成独立准确率。

## 对计划验收问题的逐项回答

| 问题 | 当前证据结论 |
|---|---|
| HW207是否有可信的实物数字模型？ | 否。拓扑顺序按用户确认和历史资料整理；当前参考面、DMD映射、Π_cls、镜头共轭及相机坐标没有测量闭环。 |
| 输入Relay是否满足分类采样？ | 未验证。理想150/150倍率下，200 µm约覆盖26.46个DMD镜周期；与100×100相位网格和28×28输入的关系只作采样比较，不能代替实际NA、方向、映射和MTF测量。 |
| 30 mm镜头能否做1:1成像？ | 理论上可在物距、像距各60 mm（相对主平面）时实现；当前Π_cls到主平面及CMOS距离未测，实际能否实现未知。 |
| 200×200 µm窗口是否正确映射到CMOS？ | 未验证。2048²×6.5 µm对应13.312 mm方形感光面是规格换算；实际相机身份、倍率、窗口偏移、旋转和读出模式未知。 |
| 四个50×50 µm ROI是否固定？ | 物理设计坐标已固定为(35–85,35–85)、(115–165,35–85)、(35–85,115–165)、(115–165,115–165) µm；其CMOS像素注册尚未测量。四ROI合计10,000 µm²，占200×200 µm窗口25%。 |
| 现有C3.5/C3.7相位能否形成目标亮斑？ | 未在新几何和新ROI下传播；当前证据不足，不能回答“能”或“不能”。候选小数组只作文件元数据核验，未做forward。 |
| 若不满足，损失发生在哪个接口？ | 尚不能归因。最先缺少的是可验证的几何/坐标接口；没有该接口时，不能把模型差异归因于单一光学部件。 |
| 是否需要重新训练？ | 当前不需要也不准入。先闭合P0–P4并用未用过的数据评价冻结候选，再依据预定门槛判断。 |
| 当前最佳候选是什么？ | 对本计划50 µm ROI没有已评估的最佳候选。历史C3.8 T3 seed23的91.05%仅是旧数值模型/旧大ROI结果，不能迁移为当前候选排名。 |
| 新物理任务是否达到90%？ | 未测；不能把旧模型准确率作为本任务通过。 |
| 是否已证明四类对应亮斑？ | 否。没有当前CMOS实测帧，也未运行经物理接口准入的forward。 |

敏感性计算只覆盖显式理想情景：f30薄透镜在`object_distance_mm`={', '.join(f"{v['object_distance_from_pi_cls_mm']:.3f}" for v in sensitivity['f30_thin_lens_scenarios'])}时的共轭，以及指定150 mm relay倍率对DMD节距的投影。它们不是测量不确定度区间。检查了{len(candidate_inventory)}个历史相位/高度候选条目的文件元数据；此操作没有生成或读取CMOS强度图。

## 工程完成与验证

生成了光路拓扑图、CMOS窗口/固定ROI比例图、Relay与f30几何敏感性图、静态DMD标定图案、当前距离采集模板、CMOS像元面积重叠积分与argmax接口、历史相位候选元数据清单和手动采集步骤。ROI积分由独立矩形交叠公式复核；自动化测试及图件QA结果另存于本会话QA目录。原C3.5–C3.8数据和运行结果未改写；未运行光场forward、CUDA、训练、激光、DMD或CMOS硬件。

## 决策影响与下一步

当前只支持“用户确认光路已装配”“固定ROI坐标与软件积分定义已固化”“理论校准采集方案已准备”。不支持“物理数字孪生闭合”“f30已实现1:1”“当前50 µm ROI准确率达标”或“冻结相位已形成四类亮斑”。建议实验人员按 `p1/acquisition_protocol.md` 和 `p0/current_distance_measurement_template.csv` 补齐原始帧、参考面距离和相机标定；这些是后续物理结论的必要输入，不代表已授权硬件操作。相位优化只在物理接口和独立评价数据通过准入后考虑。

## 本地状态

起始快照在本输出目录上一级的 `../git_status_before.txt`；分支和HEAD以该快照为准。本会话不暂存、不提交、不推送。
"""
    path.write_text(content, encoding="utf-8")


def generate_outputs(config_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    geometry_summary = nominal_geometry_summary(config)
    sensitivity = geometry_sensitivity_summary(config)
    candidate_inventory = inspect_phase_candidate_arrays(config)

    directories = ("p0", "p1", "p2", "p3", "p4", "p5", "figures")
    for name in directories:
        (output / name).mkdir()
    _write_json(output / "config_snapshot.json", config)
    _write_json(output / "p0" / "nominal_device_and_geometry_audit.json", {
        "assembly_state": "ASSEMBLED_USER_CONFIRMED",
        "identity_and_geometry_status": "PARTIAL_OR_UNKNOWN",
        "inventory_note": "BOM sheet quantities differ; they are not installed-item counts.",
        "historical_distances_are_current": False,
        "hardware_operation_performed": False,
    })
    _write_json(output / "p0" / "device_identity_inventory.json", {
        "assembly_state": "ASSEMBLED_USER_CONFIRMED",
        "devices": [
            {"device": "532 nm laser", "status": "USER_SPECIFIED_WAVELENGTH; unit identity and power calibration unknown"},
            {"device": "Beam expander and folding mirrors", "status": "PRESENT_IN_HISTORICAL_TOPOLOGY; current identity and geometry unverified"},
            {"device": "PF-1 / PBS / beam splitter / PF-2", "status": "HISTORICAL_RECORD; current orientation, ports and polarization unverified"},
            {"device": "DMD", "status": "USER_CONFIRMED_F9000_FAMILY; DLP9000 unit identity, mounted axis and ON/OFF collection unverified"},
            {"device": "Input relay", "status": "NOMINAL_150_MM_LENSES_IN_BOM; installed focal lengths, principal planes, aperture and conjugacy unverified"},
            {"device": "LightGen phase layer", "status": "USER_CONFIRMED_ASSEMBLED; material stack and complex transmission unverified"},
            {"device": "Output lens", "status": "30_MM_DESIGN_TARGET; installed serial/focal length/principal planes/aperture unverified; BOM quantity conflict is unresolved"},
            {"device": "CMOS", "status": "2048_SQUARED_X_6.5_UM_REFERENCE_SPEC; actual camera unit, readout, pixel map and response unverified"},
        ],
        "bom_conflict": "Two BOM sheets list f30-lens quantity 2 and 3; purchase list quantities do not prove installed count.",
        "hardware_operated": False,
    })
    _write_json(output / "p0" / "bom_conflict_audit.json", {
        "part_number": "GCL-010650A",
        "description": "f30 mm, 25 mm diameter achromatic lens",
        "rows": [
            {"workbook": "research_assets/lightgen/HW207_实物与距离基线_20261103/05_器件参考/LightGen光路-1014-v5.xlsx", "sheet": "Sheet1", "cell_range": "A3:H3", "quantity_cell": "F3", "quantity": 2},
            {"workbook": "research_assets/lightgen/HW207_实物与距离基线_20261103/05_器件参考/LightGen光路-1014-v5.xlsx", "sheet": "Sheet2", "cell_range": "A3:H3", "quantity_cell": "F3", "quantity": 3},
            {"workbook": "research_assets/lightgen/HW207_实物与距离基线_20261103/05_器件参考/LightGen_optical_setup_BOM_reference.xlsx", "sheet": "Sheet1", "cell_range": "A3:H3", "quantity_cell": "F3", "quantity": 2},
            {"workbook": "research_assets/lightgen/HW207_实物与距离基线_20261103/05_器件参考/LightGen_optical_setup_BOM_reference.xlsx", "sheet": "Sheet2", "cell_range": "A3:H3", "quantity_cell": "F3", "quantity": 3},
        ],
        "conclusion": "The two workbook copies each contain a Sheet1/Sheet2 quantity conflict. BOM procurement quantities do not establish how many lenses are currently installed.",
        "installed_count": "UNKNOWN_NOT_MEASURED",
    })
    _write_distance_csv(output / "p0" / "historical_distance_table.csv",
                        config["historical_distance_estimates_mm"])
    _write_current_distance_template(output / "p0" / "current_distance_measurement_template.csv")
    _write_json(output / "p2" / "ideal_relay_and_conjugacy_diagnostics.json", geometry_summary)
    _write_json(output / "p2" / "geometry_sensitivity_scenarios.json", sensitivity)
    _write_json(output / "p5" / "historical_phase_candidate_file_audit.json", {
        "inventory": candidate_inventory,
        "all_candidates_forwarded": False,
        "phase_or_height_arrays_copied": False,
    })
    _write_json(output / "p3" / "nominal_cmos_sampling.json", {
        "sensor_shape": [config["optics"]["cmos"]["rows"], config["optics"]["cmos"]["columns"]],
        "pixel_pitch_um": config["optics"]["cmos"]["pixel_pitch_um"],
        "fixed_geometry": config["fixed_cmos_geometry"],
        "nominal_conversion": geometry_summary["fixed_cmos_pixel_equivalents_from_reference_pitch"],
        "integration_rule": config["fixed_cmos_geometry"]["integration"],
        "actual_registration_calibrated": False,
        "camera_dark_flat_gain_noise_qe_at_532nm_calibrated": False,
    })
    _write_json(output / "p4" / "fixed_roi_classifier_contract.json", {
        "roi_layout": config["fixed_cmos_geometry"]["roi_layout"],
        "prediction": config["fixed_cmos_geometry"]["classification"],
        "fractional_area_weighting_implementation": "scripts/research/lightgen/hw207_physical_cmos_v1.py::build_roi_pixel_weights",
        "physical_affine_supplied": False,
    })
    missing = config["preflight"]["missing_inputs"]
    preflight = {
        "status": "PREFLIGHT_BLOCKED",
        "outer_gate": config["preflight"]["outer_gate"],
        "missing_inputs": missing,
        "decision": config["preflight"]["decision_if_missing"],
        "frozen_phase_candidates_reprocessed": 0,
        "new_samples_forwarded": 0,
        "new_phase_updates": 0,
        "hardware_operated": False,
    }
    _write_json(output / "p5" / "frozen_phase_preflight.json", preflight)

    topology = output / "figures" / "figure_01_optical_topology"
    geometry_figure = output / "figures" / "figure_02_cmos_roi_geometry"
    pattern_plate = output / "figures" / "figure_03_dmd_calibration_patterns"
    sensitivity_figure = output / "figures" / "figure_04_geometry_sensitivity"
    draw_topology(topology)
    draw_cmos_roi_geometry(geometry_figure)
    draw_geometry_sensitivity(sensitivity_figure, sensitivity)
    pattern_metadata = save_dmd_patterns(output / "p1" / "calibration_patterns", pattern_plate)
    _write_json(output / "p1" / "calibration_pattern_manifest.json", {
        "array_rows_columns": [1600, 2560],
        "pixel_format": "8-bit PNG containing only 0 and 255 values; intended as static 1-bit DMD commands",
        "patterns": pattern_metadata,
        "hardware_capture_performed": False,
    })
    _write_acquisition_protocol(output / "p1" / "acquisition_protocol.md", pattern_metadata)
    _write_report(output / "closeout_report.md", config, geometry_summary, sensitivity, candidate_inventory)

    files = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    files.append("manifest.json")
    manifest = {
        "identity": IDENTITY,
        "session_source_id": config["session_source_id"],
        "status": "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED",
        "config_source": str(Path(config_path)),
        "script_source": str(Path(__file__)),
        "measurement_data_copied": False,
        "historical_phase_files_copied": False,
        "hardware_operated": False,
        "forward_simulation_executed": False,
        "training_executed": False,
        "phase_candidate_metadata_entries": len(candidate_inventory),
        "phase_candidates_forwarded": 0,
        "git_status_snapshot": "../git_status_before.txt",
        "generated_files": files,
        "references": config["references"],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = generate_outputs(args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "generated_files": len(result["generated_files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
