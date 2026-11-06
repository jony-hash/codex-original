"""Reintegrate saved C3.8 CMOS model maps using the current fixed 50 um ROIs.

This is a post-hoc diagnostic of existing numerical arrays. It performs no
optical forward propagation, training, hardware operation, or per-image
normalization. The assumed center and orientation are explicit scenarios, not
a measured registration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

from audit_panel_alignment import require_matplotlib_panel_alignment

from hw207_physical_cmos_v1 import (
    DEFAULT_ROIS_UM,
    build_roi_pixel_weights,
    fixed_roi_classification_metrics,
    integrate_roi_energies,
    intensity_concentration_metrics,
    predict_from_roi_energies,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_RUN = (
    ROOT
    / "outputs/20261105_hw207_mnist4_c38_v1/runs/autodl_backup_attempt_05_closeout_01"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/20261106_hw207_physical_cmos_v1/runs/diagnostic-only_01a1a9b1-5c54-7d81-9999-0cb3b09557a2/artifacts_attempt_05/small_roi_reintegration"
)
EXPECTED_SHAPE = (2048, 2048)
PIXEL_PITCH_UM = 6.5
WINDOW_UM = (200.0, 200.0)
WINDOW_CENTER_UM = (100.0, 100.0)
ROI_AREA_UM2 = 2500.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


ORIENTATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "array_indices_as_y_down",
        "description": "raw row index increases toward CMOS y-down",
        "matrix": ((1, 0), (0, 1)),
        "apply": lambda image: image,
    },
    {
        "name": "model_lower_origin_to_cmos_upper_origin",
        "description": "vertical reflection: model origin-lower to CMOS origin-upper",
        "matrix": ((1, 0), (0, -1)),
        "apply": lambda image: np.flipud(image),
    },
    {
        "name": "horizontal_reflection",
        "description": "horizontal reflection relative to the fixed CMOS frame",
        "matrix": ((-1, 0), (0, 1)),
        "apply": lambda image: np.fliplr(image),
    },
    {
        "name": "rotation_180",
        "description": "180 degree rotation relative to the fixed CMOS frame",
        "matrix": ((-1, 0), (0, -1)),
        "apply": lambda image: np.rot90(image, 2),
    },
    {
        "name": "transpose",
        "description": "transpose of row and column axes",
        "matrix": ((0, 1), (1, 0)),
        "apply": lambda image: image.T,
    },
    {
        "name": "anti_transpose",
        "description": "reflected transpose of row and column axes",
        "matrix": ((0, -1), (-1, 0)),
        "apply": lambda image: np.flipud(np.fliplr(image.T)),
    },
    {
        "name": "rotation_90_ccw_array",
        "description": "90 degree counterclockwise array rotation",
        "matrix": ((0, 1), (-1, 0)),
        "apply": lambda image: np.rot90(image, 1),
    },
    {
        "name": "rotation_90_cw_array",
        "description": "90 degree clockwise array rotation",
        "matrix": ((0, -1), (1, 0)),
        "apply": lambda image: np.rot90(image, 3),
    },
)


def _source_paths(source_run: Path) -> dict[str, Path]:
    return {
        "manifest": source_run / "p5/c38_visualization_manifest.json",
        "selected": source_run / "p5/c38_selected_cmos_maps_raw_float32.npz",
        "means": source_run / "p4/c38_selected_cmos_group_means.npz",
        "rows": source_run / "p4/T3_seed23_validation_rows.npz",
    }


def _load_source(source_run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paths = _source_paths(source_run)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required C3.8 saved input(s): " + ", ".join(missing))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("selected_candidate") != "T3_seed23":
        raise ValueError("source P5 manifest is not the expected C3.8 T3_seed23 candidate")
    if manifest.get("selected_population_role") != (
        "development validation reused for model selection, not independent"
    ):
        raise ValueError("source population role differs from the documented C3.8 manifest")
    if manifest.get("coordinate_mapping") != (
        "row=y, column=x; origin lower; CMOS axes centered at optical axis"
    ):
        raise ValueError("source coordinate convention differs from the reviewed C3.8 manifest")

    with np.load(paths["selected"], allow_pickle=False) as data:
        selected = {key: data[key] for key in data.files}
    with np.load(paths["means"], allow_pickle=False) as data:
        means = {key: data[key] for key in data.files}
    with np.load(paths["rows"], allow_pickle=False) as data:
        rows = {key: data[key] for key in data.files}

    if selected["intensity"].shape != (36, *EXPECTED_SHAPE):
        raise ValueError(f"unexpected selected-map shape: {selected['intensity'].shape}")
    if manifest.get("raw_map_shape") != list(selected["intensity"].shape):
        raise ValueError("P5 raw-map shape differs from the saved manifest")
    if not np.array_equal(selected["main_rows"], np.asarray(manifest["main_rows"])):
        raise ValueError("P5 main-row list differs from the saved manifest")
    if means["all_class_means"].shape != (4, *EXPECTED_SHAPE):
        raise ValueError("all-class mean array has an unexpected shape")
    if means["correct_class_means"].shape != (4, *EXPECTED_SHAPE):
        raise ValueError("correct-only mean array has an unexpected shape")
    if rows["rows"].shape != (2000,) or rows["labels"].shape != (2000,):
        raise ValueError("P4 row/label arrays have an unexpected shape")
    if rows["predictions"].shape != (2000,):
        raise ValueError("P4 predictions have an unexpected shape")
    if rows["input_energy_um2"].shape != (2000,):
        raise ValueError("P4 input-energy array has an unexpected shape")
    if not np.isfinite(rows["input_energy_um2"]).all() or (rows["input_energy_um2"] <= 0).any():
        raise ValueError("P4 input energies must be finite and positive")
    if len(np.unique(selected["rows"])) != len(selected["rows"]):
        raise ValueError("selected P5 source rows are not unique")
    if not np.isin(np.asarray(manifest["main_rows"]), selected["rows"]).all():
        raise ValueError("one or more manifest main rows are absent from P5 saved maps")
    if len(np.unique(rows["rows"])) != len(rows["rows"]):
        raise ValueError("P4 source row table contains duplicate row identifiers")

    row_lookup = {
        int(row): (int(label), int(prediction), float(input_energy))
        for row, label, prediction, input_energy in zip(
            rows["rows"], rows["labels"], rows["predictions"], rows["input_energy_um2"]
        )
    }
    selected_labels: list[int] = []
    selected_predictions: list[int] = []
    for row in selected["rows"]:
        if int(row) not in row_lookup:
            raise ValueError(f"selected P5 row {int(row)} is absent from the P4 row table")
        label, prediction, _ = row_lookup[int(row)]
        selected_labels.append(label)
        selected_predictions.append(prediction)
    selected["labels"] = np.asarray(selected_labels, dtype=np.int64)
    selected["old_predictions"] = np.asarray(selected_predictions, dtype=np.int64)

    for key in ("intensity",):
        array = selected[key]
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"source selected map {key} contains invalid values")
    for key in ("all_class_means", "correct_class_means", "overall_mean"):
        array = means[key]
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"source mean array {key} contains invalid values")
    if not np.array_equal(means["rows"], rows["rows"]):
        raise ValueError("P4 means and P4 row table have different row order")
    if not np.array_equal(means["labels"], rows["labels"]):
        raise ValueError("P4 means and P4 row table have different labels")
    if not np.array_equal(means["predictions"], rows["predictions"]):
        raise ValueError("P4 means and P4 row table have different predictions")
    if not np.array_equal(means["all_class_counts"], np.full(4, 500)):
        raise ValueError("expected 500 reused development rows in each all-class mean")
    if not np.array_equal(means["correct_class_counts"], np.asarray(manifest["p4_group_mean_counts"]["correct_by_class"])):
        raise ValueError("P4 correct-only mean counts differ from the P5 manifest")
    return manifest, {"selected": selected, "means": means, "rows": rows}


def _build_weights(shape: tuple[int, int]) -> tuple[dict[str, Any], ...]:
    rows, columns = shape
    pitch = PIXEL_PITCH_UM
    matrix = np.asarray(((pitch, 0.0), (0.0, pitch)), dtype=np.float64)
    sensor_center = np.asarray((columns / 2.0, rows / 2.0), dtype=np.float64)
    offset = np.asarray(WINDOW_CENTER_UM, dtype=np.float64) - matrix @ sensor_center
    return build_roi_pixel_weights(
        shape,
        matrix,
        offset,
        roi_rectangles_um=DEFAULT_ROIS_UM,
        window_um=WINDOW_UM,
    )


def _direct_axis_aligned_integral(image: np.ndarray, roi: tuple[int, float, float, float, float]) -> float:
    """Independent rectangle/pixel overlap reference for axis-aligned cases."""
    _, x0, x1, y0, y1 = roi
    rows, columns = image.shape
    x_left = WINDOW_CENTER_UM[0] + PIXEL_PITCH_UM * (
        np.arange(columns, dtype=np.float64) - columns / 2.0
    )
    y_top = WINDOW_CENTER_UM[1] + PIXEL_PITCH_UM * (
        np.arange(rows, dtype=np.float64) - rows / 2.0
    )
    x_overlap = np.maximum(
        0.0,
        np.minimum(x_left + PIXEL_PITCH_UM, x1) - np.maximum(x_left, x0),
    )
    y_overlap = np.maximum(
        0.0,
        np.minimum(y_top + PIXEL_PITCH_UM, y1) - np.maximum(y_top, y0),
    )
    row_ids = np.flatnonzero(y_overlap)
    column_ids = np.flatnonzero(x_overlap)
    values = np.asarray(image[np.ix_(row_ids, column_ids)], dtype=np.float64)
    return float(np.sum(values * y_overlap[row_ids, None] * x_overlap[None, column_ids]))


def _orientation_point(
    source_x: float,
    source_y: float,
    matrix: tuple[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[float, float]:
    rows, columns = shape
    centered = np.asarray(
        (source_x - columns / 2.0, source_y - rows / 2.0), dtype=np.float64
    )
    physical = np.asarray(matrix, dtype=np.float64) @ centered
    return (
        WINDOW_CENTER_UM[0] + PIXEL_PITCH_UM * float(physical[0]),
        WINDOW_CENTER_UM[1] + PIXEL_PITCH_UM * float(physical[1]),
    )


def _source_peak_centroid(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("source image has no positive total intensity")
    rows, columns = values.shape
    row_sum = np.sum(values, axis=1, dtype=np.float64)
    column_sum = np.sum(values, axis=0, dtype=np.float64)
    centroid_row = float(np.dot(row_sum, np.arange(rows, dtype=np.float64) + 0.5) / total)
    centroid_column = float(
        np.dot(column_sum, np.arange(columns, dtype=np.float64) + 0.5) / total
    )
    peak_row, peak_column = np.unravel_index(int(np.argmax(values)), values.shape)
    concentration = intensity_concentration_metrics(values, PIXEL_PITCH_UM)
    return {
        "total_density_sum": total,
        "centroid_source_x": centroid_column,
        "centroid_source_y": centroid_row,
        "peak_source_x": float(peak_column) + 0.5,
        "peak_source_y": float(peak_row) + 0.5,
        "peak_intensity": float(values[peak_row, peak_column]),
        **concentration,
    }


def _point_in_roi(x: float, y: float, roi_id: int) -> bool:
    _, x0, x1, y0, y1 = DEFAULT_ROIS_UM[roi_id]
    return x0 <= x < x1 and y0 <= y < y1


def _image_record(
    image: np.ndarray,
    *,
    weights: tuple[dict[str, Any], ...],
    orientation: dict[str, Any],
    true_label: int,
    row: int | None,
    old_prediction: int | None,
    input_energy: float | None,
    sample_group: str,
    source_geometry: dict[str, float],
) -> dict[str, Any]:
    oriented = orientation["apply"](image)
    pixel_area = PIXEL_PITCH_UM**2
    scores = integrate_roi_energies(
        np.asarray(oriented, dtype=np.float64) * pixel_area, weights
    )
    prediction = predict_from_roi_energies(scores)
    total = source_geometry["total_density_sum"] * pixel_area
    roi_sum = float(np.sum(scores))
    target = float(scores[true_label])
    wrong = float(np.max(np.delete(scores, true_label)))
    outside = total - roi_sum
    outside_tolerance = max(1e-10, abs(total) * 1e-12)
    if outside < -outside_tolerance:
        raise ArithmeticError("the four fixed nonoverlapping ROIs exceed full CMOS energy")
    outside = max(outside, 0.0)
    matrix = orientation["matrix"]
    peak_x, peak_y = _orientation_point(
        source_geometry["peak_source_x"],
        source_geometry["peak_source_y"],
        matrix,
        image.shape,
    )
    centroid_x, centroid_y = _orientation_point(
        source_geometry["centroid_source_x"],
        source_geometry["centroid_source_y"],
        matrix,
        image.shape,
    )
    target_weights = weights[true_label]
    target_pixels = oriented[target_weights["rows"], target_weights["columns"]]
    target_roi_max = float(np.max(target_pixels)) if target_pixels.size else math.nan
    return {
        "row": row,
        "sample_group": sample_group,
        "true_label": true_label,
        "old_c38_prediction": old_prediction,
        "orientation_scenario": orientation["name"],
        "roi0_energy_au_um2": float(scores[0]),
        "roi1_energy_au_um2": float(scores[1]),
        "roi2_energy_au_um2": float(scores[2]),
        "roi3_energy_au_um2": float(scores[3]),
        "small_roi_prediction": prediction,
        "small_roi_correct": bool(prediction == true_label),
        "target_roi_energy_au_um2": target,
        "strongest_wrong_roi_energy_au_um2": wrong,
        "target_over_strongest_wrong_ratio": target / wrong if wrong > 0 else math.inf,
        "target_fraction_of_four_rois": target / roi_sum if roi_sum > 0 else math.nan,
        "target_fraction_of_full_cmos": target / total if total > 0 else math.nan,
        "roi_outside_energy_au_um2": outside,
        "roi_outside_fraction_of_full_cmos": outside / total if total > 0 else math.nan,
        "model_input_energy_um2": input_energy,
        "target_roi_to_input_model_fraction": (
            target / input_energy if input_energy is not None and input_energy > 0 else None
        ),
        "target_roi_mean_intensity_au": target / ROI_AREA_UM2,
        "target_roi_max_pixel_intensity_au": target_roi_max,
        "full_cmos_integral_au_um2": total,
        "four_roi_integral_au_um2": roi_sum,
        "global_peak_intensity_au": source_geometry["peak_intensity"],
        "global_peak_x_um_in_window": peak_x,
        "global_peak_y_um_in_window": peak_y,
        "global_peak_in_target_roi": _point_in_roi(peak_x, peak_y, true_label),
        "intensity_centroid_x_um_in_window": centroid_x,
        "intensity_centroid_y_um_in_window": centroid_y,
        "centroid_in_target_roi": _point_in_roi(centroid_x, centroid_y, true_label),
        "intensity_rms_radius_um_about_centroid": source_geometry["intensity_rms_radius_um"],
        "inverse_participation_ratio_per_pixel": source_geometry[
            "inverse_participation_ratio_per_pixel"
        ],
        "effective_intensity_area_um2": source_geometry["effective_area_um2"],
        "effective_intensity_area_fraction_of_full_map": source_geometry[
            "effective_area_fraction_of_map"
        ],
    }


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "nan_count": int(np.isnan(array).sum()),
        "mean": float(np.mean(finite)) if finite.size else None,
        "p10": float(np.quantile(finite, 0.10)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p90": float(np.quantile(finite, 0.90)) if finite.size else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def _check_independent_integrals(
    images: list[tuple[np.ndarray, str]],
    orientations: tuple[dict[str, Any], ...],
    weights: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    maximum_absolute_error = 0.0
    comparisons = 0
    for image, _ in images:
        for orientation in orientations:
            oriented = orientation["apply"](image)
            library_scores = integrate_roi_energies(
                np.asarray(oriented, dtype=np.float64) * PIXEL_PITCH_UM**2,
                weights,
            )
            direct_scores = np.asarray(
                [
                    _direct_axis_aligned_integral(oriented, roi)
                    for roi in DEFAULT_ROIS_UM
                ],
                dtype=np.float64,
            )
            error = float(np.max(np.abs(library_scores - direct_scores)))
            maximum_absolute_error = max(maximum_absolute_error, error)
            comparisons += 4
            if not np.allclose(library_scores, direct_scores, rtol=2e-12, atol=1e-10):
                raise ArithmeticError(
                    f"independent ROI integration mismatch for {orientation['name']}: "
                    f"max absolute error {error:.9g}"
                )
    return {
        "status": "PASS",
        "method": "independent axis-aligned pixel/rectangle overlap integration",
        "roi_comparisons": comparisons,
        "maximum_absolute_error_au_um2": maximum_absolute_error,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_class_means(
    *,
    means: dict[str, np.ndarray],
    mean_counts: dict[str, list[int]],
    source_shared_vmax: float,
    output_dir: Path,
) -> dict[str, Any]:
    x_edges = WINDOW_CENTER_UM[0] + PIXEL_PITCH_UM * (
        np.arange(EXPECTED_SHAPE[1] + 1, dtype=np.float64) - EXPECTED_SHAPE[1] / 2.0
    )
    y_edges = WINDOW_CENTER_UM[1] + PIXEL_PITCH_UM * (
        np.arange(EXPECTED_SHAPE[0] + 1, dtype=np.float64) - EXPECTED_SHAPE[0] / 2.0
    )
    c0 = max(0, int(np.searchsorted(x_edges, 0.0, side="right") - 1))
    c1 = min(EXPECTED_SHAPE[1], int(np.searchsorted(x_edges, 200.0, side="left")))
    r0 = max(0, int(np.searchsorted(y_edges, 0.0, side="right") - 1))
    r1 = min(EXPECTED_SHAPE[0], int(np.searchsorted(y_edges, 200.0, side="left")))

    plotted: list[tuple[str, int, np.ndarray]] = []
    for kind, key in (("all samples", "all_class_means"), ("correct-only", "correct_class_means")):
        for class_id in range(4):
            oriented = np.flipud(means[key][class_id])
            plotted.append((kind, class_id, oriented[r0:r1, c0:c1]))
    pooled = np.concatenate([crop.ravel() for _, _, crop in plotted])
    contrast_vmax = float(np.quantile(pooled, 0.999))
    if contrast_vmax <= 0:
        raise ValueError("small-window class mean maps have no positive intensity")
    clipped_fraction = float(np.count_nonzero(pooled > contrast_vmax) / pooled.size)
    absolute_clipped_fraction = float(np.count_nonzero(pooled > source_shared_vmax) / pooled.size)

    ext = (float(x_edges[c0]), float(x_edges[c1]), float(y_edges[r1]), float(y_edges[r0]))
    roi_labels = ("ROI 0", "ROI 1", "ROI 2", "ROI 3")
    written: dict[str, Any] = {}
    plot_specs = (
        {
            "name": "historical_absolute",
            "transform": lambda values: values,
            "vmax": source_shared_vmax,
            "colorbar": "model intensity (a.u.; shared linear scale)",
            "scale": "shared absolute linear intensity across all eight class-mean panels; no per-image normalization",
            "transform_name": "identity",
            "clipped_fraction": absolute_clipped_fraction,
            "display_note": "Absolute model intensity; shared full-map linear range. Display enlargement only; not camera counts.",
        },
        {
            "name": "shared_small_window_sqrt_p99_9",
            "transform": np.sqrt,
            "vmax": math.sqrt(contrast_vmax),
            "colorbar": "sqrt(model intensity) (sqrt(a.u.); shared range)",
            "scale": "same square-root transform across all eight class-mean panels; common pooled-window P99.9 range; no per-image normalization",
            "transform_name": "sqrt(model intensity)",
            "clipped_fraction": clipped_fraction,
            "display_note": (
                "Contrast-enhanced: sqrt(model intensity), one shared pooled-window P99.9 range; "
                f"{clipped_fraction:.4%} of pixels clipped. Display enlargement only; not camera counts."
            ),
        },
    )
    for spec in plot_specs:
        scale_name = spec["name"]
        vmax = float(spec["vmax"])
        fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.7), constrained_layout=True)
        last = None
        for index, (kind, class_id, crop) in enumerate(plotted):
            ax = axes.flat[index]
            last = ax.imshow(
                spec["transform"](crop),
                origin="upper",
                extent=ext,
                interpolation="nearest",
                aspect="equal",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            for roi_id, x0, x1, y0, y1 in DEFAULT_ROIS_UM:
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="#e8f0f8",
                        linewidth=0.8,
                    )
                )
                ax.text(
                    (x0 + x1) / 2.0,
                    y0 - 7.0 if roi_id < 2 else y1 + 7.0,
                    roi_labels[roi_id],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6.3,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.2, "edgecolor": "none"},
                )
            count = mean_counts["all_by_class" if kind == "all samples" else "correct_by_class"][class_id]
            ax.set_title(f"class {class_id} · {kind} · n={count}", fontsize=9)
            ax.set_xlim(0, 200)
            ax.set_ylim(200, 0)
            ax.set_xlabel("CMOS x (µm)")
            ax.set_ylabel("CMOS y (µm, down)")
            ax.set_xticks((0, 50, 100, 150, 200))
            ax.set_yticks((0, 50, 100, 150, 200))
            ax.tick_params(labelsize=7)
        assert last is not None
        colorbar = fig.colorbar(
            last,
            ax=axes.ravel().tolist(),
            shrink=0.88,
            label=spec["colorbar"],
        )
        fig.suptitle(
            "C3.8 saved class means under fixed physical 50 × 50 µm ROIs\n"
            "centered 200 × 200 µm sensor window · origin-lower → origin-upper scenario",
            fontsize=12,
        )
        fig.text(
            0.5,
            -0.012,
            spec["display_note"]
            + " 6.5 µm model pixels; center/orientation uncalibrated; population reused for development.",
            ha="center",
            fontsize=8,
        )
        stem = f"class_mean_small_roi_{scale_name}"
        panel_ids = list("abcdefgh")
        for ax, label in zip(axes.flat, panel_ids):
            offset = ScaledTranslation(-5 / 72, 4 / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                label,
                transform=ax.transAxes + offset,
                ha="left",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
            )
        alignment = require_matplotlib_panel_alignment(
            fig,
            axes=list(axes.flat),
            panel_ids=panel_ids,
            row_groups=[panel_ids[:4], panel_ids[4:]],
            column_groups=[[panel_ids[index], panel_ids[index + 4]] for index in range(4)],
            exclude_axes=[colorbar.ax],
            require_panel_labels=True,
            json_out=output_dir / f"{stem}.alignment.json",
            overlay_svg=output_dir / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
        for ext_name in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{stem}.{ext_name}", dpi=300 if ext_name == "png" else None, bbox_inches="tight")
        plt.close(fig)
        written[scale_name] = {
            "vmin": 0.0,
            "vmax": float(vmax),
            "scale": spec["scale"],
            "transform": spec["transform_name"],
            "pixels_clipped_above_vmax_fraction": float(spec["clipped_fraction"]),
            "alignment_verdict": alignment["verdict"],
            "alignment_json": f"{stem}.alignment.json",
            "alignment_overlay_svg": f"{stem}.alignment.svg",
            "files": [f"{stem}.{ext_name}" for ext_name in ("png", "pdf", "svg")],
        }
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic output: {output_dir}")
    manifest, data = _load_source(source_run)
    selected = data["selected"]
    means = data["means"]
    rows = data["rows"]
    output_dir.mkdir(parents=True, exist_ok=False)

    shape = tuple(int(value) for value in selected["intensity"].shape[-2:])
    weights = _build_weights(shape)
    roi_areas = [float(item["roi_area_um2"]) for item in weights]
    effective_counts = [float(item["effective_pixel_count"]) for item in weights]
    if not np.allclose(roi_areas, ROI_AREA_UM2, rtol=0, atol=1e-10):
        raise ArithmeticError(f"physical ROI areas are not 2500 um^2: {roi_areas}")
    constant_scores = integrate_roi_energies(
        np.ones(shape, dtype=np.float64) * PIXEL_PITCH_UM**2,
        weights,
    )
    if not np.allclose(constant_scores, ROI_AREA_UM2, rtol=0, atol=1e-7):
        raise ArithmeticError("constant-intensity analytic area check failed")

    sample_records: list[dict[str, Any]] = []
    independent_images: list[tuple[np.ndarray, str]] = []
    source_geometry_by_row: dict[int, dict[str, float]] = {}
    row_lookup = {
        int(row): (int(label), int(prediction), float(input_energy))
        for row, label, prediction, input_energy in zip(
            rows["rows"], rows["labels"], rows["predictions"], rows["input_energy_um2"]
        )
    }
    for index, image in enumerate(selected["intensity"]):
        row = int(selected["rows"][index])
        label = int(selected["labels"][index])
        old_prediction = int(selected["old_predictions"][index])
        if row_lookup[row][:2] != (label, old_prediction):
            raise ValueError(f"selected-map labels/predictions do not match P4 row {row}")
        input_energy = row_lookup[row][2]
        group = "selected display sample; biased by C3.8 selection rules"
        geometry = _source_peak_centroid(image)
        source_geometry_by_row[row] = geometry
        independent_images.append((image, f"selected row {row}"))
        for orientation in ORIENTATIONS:
            sample_records.append(
                _image_record(
                    image,
                    weights=weights,
                    orientation=orientation,
                    true_label=label,
                    row=row,
                    old_prediction=old_prediction,
                    input_energy=input_energy,
                    sample_group=group,
                    source_geometry=geometry,
                )
            )

    mean_records: list[dict[str, Any]] = []
    mean_geometry: dict[tuple[str, int], dict[str, float]] = {}
    for mean_kind, key, counts_key in (
        ("all_true_class_samples", "all_class_means", "all_class_counts"),
        ("correct_class_samples_only", "correct_class_means", "correct_class_counts"),
    ):
        for class_id, image in enumerate(means[key]):
            geometry = _source_peak_centroid(image)
            mean_geometry[(mean_kind, class_id)] = geometry
            independent_images.append((image, f"{mean_kind} class {class_id}"))
            for orientation in ORIENTATIONS:
                record = _image_record(
                    image,
                    weights=weights,
                    orientation=orientation,
                    true_label=class_id,
                    row=None,
                    old_prediction=None,
                    input_energy=None,
                    sample_group=mean_kind,
                    source_geometry=geometry,
                )
                record["mean_sample_count"] = int(means[counts_key][class_id])
                mean_records.append(record)

    independent_check = _check_independent_integrals(
        independent_images,
        ORIENTATIONS,
        weights,
    )
    _write_csv(output_dir / "selected_sample_small_roi_integrals.csv", sample_records)
    _write_csv(output_dir / "class_mean_small_roi_integrals.csv", mean_records)

    class_means_for_plot = {
        "all_class_means": means["all_class_means"],
        "correct_class_means": means["correct_class_means"],
    }
    count_manifest = {
        "all_by_class": [int(value) for value in means["all_class_counts"]],
        "correct_by_class": [int(value) for value in means["correct_class_counts"]],
    }
    display = _plot_class_means(
        means=class_means_for_plot,
        mean_counts=count_manifest,
        source_shared_vmax=float(selected["shared_linear_vmax"]),
        output_dir=output_dir,
    )

    orientation_summary: list[dict[str, Any]] = []
    summary_metric_keys = (
        "target_roi_energy_au_um2",
        "strongest_wrong_roi_energy_au_um2",
        "target_over_strongest_wrong_ratio",
        "target_fraction_of_four_rois",
        "target_fraction_of_full_cmos",
        "roi_outside_energy_au_um2",
        "roi_outside_fraction_of_full_cmos",
        "target_roi_to_input_model_fraction",
        "intensity_rms_radius_um_about_centroid",
        "effective_intensity_area_um2",
        "effective_intensity_area_fraction_of_full_map",
    )
    for orientation in ORIENTATIONS:
        name = orientation["name"]
        records = [record for record in sample_records if record["orientation_scenario"] == name]
        mean_rows = [record for record in mean_records if record["orientation_scenario"] == name]
        valid_predictions = [record for record in records if record["small_roi_prediction"] is not None]
        all_mean_predictions = {
            record["true_label"]: record["small_roi_prediction"]
            for record in mean_rows
            if record["sample_group"] == "all_true_class_samples"
        }
        correct_mean_predictions = {
            record["true_label"]: record["small_roi_prediction"]
            for record in mean_rows
            if record["sample_group"] == "correct_class_samples_only"
        }
        classification = fixed_roi_classification_metrics(
            [int(record["true_label"]) for record in records],
            [record["small_roi_prediction"] for record in records],
        )
        distributions = {
            key: _distribution_summary(
                [float(record[key]) for record in records if record.get(key) is not None]
            )
            for key in summary_metric_keys
        }
        orientation_summary.append(
            {
                "orientation_scenario": name,
                "description": orientation["description"],
                "selected_sample_count": len(records),
                "selected_sample_correct_count": sum(bool(record["small_roi_correct"]) for record in records),
                "selected_sample_accuracy_descriptive_only": (
                    sum(bool(record["small_roi_correct"]) for record in records) / len(records)
                ),
                "selected_rows_with_unresolved_ties": len(records) - len(valid_predictions),
                "classification_metrics_descriptive_only": classification,
                "peak_in_target_roi_rate": sum(
                    bool(record["global_peak_in_target_roi"]) for record in records
                ) / len(records),
                "centroid_in_target_roi_rate": sum(
                    bool(record["centroid_in_target_roi"]) for record in records
                ) / len(records),
                "diagnostic_distributions": distributions,
                "all_sample_class_mean_predicted_roi": [all_mean_predictions[i] for i in range(4)],
                "correct_only_class_mean_predicted_roi": [correct_mean_predictions[i] for i in range(4)],
                "all_sample_class_mean_matches_true_class_count": sum(
                    all_mean_predictions[i] == i for i in range(4)
                ),
                "correct_only_class_mean_matches_true_class_count": sum(
                    correct_mean_predictions[i] == i for i in range(4)
                ),
            }
        )

    physical_mapping = {
        "map_data": "saved C3.8 model intensity at the modeled CMOS plane; not camera counts",
        "source_pixel_pitch_um": PIXEL_PITCH_UM,
        "window_um": list(WINDOW_UM),
        "window_placement": "assumed centered on the saved model optical axis at the center of the 2048 x 2048 array",
        "window_orientation": "not hardware-calibrated; eight D4 array orientation scenarios reported separately",
        "roi_contract": [
            {"roi": roi_id, "x_um": [x0, x1], "y_um_down": [y0, y1], "size_um": [x1-x0, y1-y0]}
            for roi_id, x0, x1, y0, y1 in DEFAULT_ROIS_UM
        ],
        "model_pixel_area_um2": PIXEL_PITCH_UM**2,
        "effective_model_pixels_per_roi": effective_counts,
        "spot_concentration_definition": (
            "full-map intensity RMS radius about the centroid plus inverse-participation effective area; "
            "not a fitted PSF or measured spot boundary"
        ),
        "relative_efficiency_definition": (
            "small fixed target-ROI model energy divided by the same C3.8 row's saved input_energy_um2; "
            "dimensionless model ratio, not calibrated optical power efficiency"
        ),
        "optical_magnification": "not applied to sensor-plane ROI integration; maps already represent modeled CMOS-plane samples",
    }
    report = {
        "identity": "HW207_C38_FIXED_50UM_ROI_POSTHOC_REINTEGRATION_V1",
        "status": "DIAGNOSTIC_ONLY_PHYSICAL_MAPPING_UNCALIBRATED",
        "source_run": str(source_run),
        "source_files": {key: str(value) for key, value in _source_paths(source_run).items()},
        "source_candidate": manifest["selected_candidate"],
        "source_phase_sha256_provenance": manifest["phase_sha256"],
        "source_population_role": manifest["selected_population_role"],
        "source_model_coordinate_mapping": manifest["coordinate_mapping"],
        "source_map_shape": list(selected["intensity"].shape),
        "source_rows": [int(value) for value in selected["rows"]],
        "selected_rows_per_true_class": {
            str(class_id): int(np.count_nonzero(selected["labels"] == class_id))
            for class_id in range(4)
        },
        "source_population_mean_counts": count_manifest,
        "physical_mapping_scenario": physical_mapping,
        "orientation_scenarios": orientation_summary,
        "checks": {
            "source_row_label_prediction_crosswalk": "PASS",
            "fixed_roi_area_each_um2": roi_areas,
            "constant_density_integral_each_au_um2": [float(value) for value in constant_scores],
            "independent_integral_recalculation": independent_check,
            "roi_pixel_fractions_do_not_overlap": True,
            "forward_propagation_rerun": False,
            "training_rerun": False,
            "hardware_operation": False,
        },
        "display": display,
        "interpretation": [
            "The 36 saved P5 maps were selected for display from a reused development-validation population; their descriptive small-ROI accuracy is not a population accuracy estimate.",
            "Per-orientation confusion, accuracy, macro-F1, class recall, target-energy advantage, outside-ROI energy, peak/centroid localization, RMS spread, inverse-participation effective area, and model-relative target/input energy are reported for the 36 selected rows only.",
            "The target/input ratio uses the saved C3.8 numerical input-energy denominator and is not a measured optical power efficiency.",
            "Spatial concentration is summarized by an intensity-weighted RMS radius and inverse-participation effective area; neither is a fitted spot edge or physical PSF measurement.",
            "The P4 class-average maps include all 500 true-class rows per class; correct-only means are shown separately and are not substituted for all-sample means.",
            "Orientation scenarios are reported without choosing the scenario that maximizes accuracy. A measured camera/optics registration is still required.",
            "The fixed 50 x 50 um regions occupy about 7.69 x 7.69 modeled 6.5 um pixels each; fractional pixel overlap is integrated exactly.",
            "No physical CMOS counts, dark/flat correction, measured sensor mapping, or hardware result is present in this analysis.",
            "Only 36 raw full-resolution CMOS maps are available for the new small-ROI post-hoc calculation; the full 2,000-row small-ROI metrics remain unavailable without a new forward pass.",
        ],
        "outputs": [
            "selected_sample_small_roi_integrals.csv",
            "class_mean_small_roi_integrals.csv",
            "class_mean_small_roi_historical_absolute.png",
            "class_mean_small_roi_historical_absolute.pdf",
            "class_mean_small_roi_historical_absolute.svg",
            "class_mean_small_roi_shared_small_window_sqrt_p99_9.png",
            "class_mean_small_roi_shared_small_window_sqrt_p99_9.pdf",
            "class_mean_small_roi_shared_small_window_sqrt_p99_9.svg",
        ],
    }
    (output_dir / "small_roi_reintegration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# C3.8 saved maps re-integrated with the fixed 50 µm CMOS ROIs",
        "",
        "**Status: diagnostic only.** This post-hoc analysis reads saved model intensity arrays. It does not rerun propagation, training, or hardware.",
        "",
        f"Source candidate: `{manifest['selected_candidate']}`; rows: {len(selected['rows'])} selected display maps. The source population is reused development validation, not an independent evaluation set.",
        "",
        "## Physical-coordinate scenario",
        "",
        "The saved maps have 2048 × 2048 samples at a modeled 6.5 µm CMOS pixel pitch. The 200 × 200 µm window is assumed centered on the saved optical axis. Each fixed ROI is 50 × 50 µm; fractional pixel overlap is integrated as intensity density × overlap area. This is a nominal coordinate scenario, not a measured sensor registration.",
        "",
        "The source manifest says the saved array is displayed with origin lower. Results therefore include the conventional vertical-reflection scenario into a top-left, y-down CMOS frame and seven other rotations/reflections. No orientation is selected by best accuracy.",
        "",
        "## Orientation sensitivity",
        "",
        "| Scenario | Selected-map correct / 36 (descriptive only) | All-sample class means matching true ROI | Correct-only means matching true ROI |",
        "|---|---:|---:|---:|",
    ]
    for item in orientation_summary:
        lines.append(
            f"| `{item['orientation_scenario']}` | {item['selected_sample_correct_count']}/36 | "
            f"{item['all_sample_class_mean_matches_true_class_count']}/4 | "
            f"{item['correct_only_class_mean_matches_true_class_count']}/4 |"
        )
    lines.extend(
        [
            "",
            "The 36 maps are a biased display selection (typical correct, difficult correct, and representative errors), so their fraction is not an accuracy estimate. Class-mean ROI argmax is also not individual-sample accuracy.",
            "",
            "## Descriptive P4 metrics for the 36 selected maps",
            "",
            "All metrics below are model-array diagnostics for a display-selected subset; accuracy, macro-F1, and recall are not population estimates.",
            "",
            "| Orientation | Accuracy | Macro-F1 | Recall 0 / 1 / 2 / 3 | Peak in target ROI | Centroid in target ROI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in orientation_summary:
        classification = item["classification_metrics_descriptive_only"]
        recalls = " / ".join(
            "n/a" if value["recall"] is None else f"{value['recall']:.3f}"
            for value in classification["per_class"]
        )
        lines.append(
            f"| `{item['orientation_scenario']}` | {classification['accuracy']:.3f} | "
            f"{classification['macro_f1']:.3f} | {recalls} | "
            f"{item['peak_in_target_roi_rate']:.3f} | {item['centroid_in_target_roi_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The JSON and per-map CSV also report fixed-ROI energies, strongest wrong-ROI energy, target advantage, energy outside all four ROIs, total CMOS model energy, target/input model fraction, peak/centroid coordinates, intensity RMS radius, and inverse-participation effective area. The latter two concentration metrics are definitions over the saved full map, not a fitted spot boundary or measured PSF.",
            "The target/input ratio uses the saved C3.8 model input energy; it is not a calibrated optical power efficiency.",
            "",
            "## Checks and limitations",
            "",
            f"- Exact physical ROI area check: {roi_areas[0]:.1f} µm² per ROI.",
            f"- Independent axis-aligned overlap recalculation: {independent_check['roi_comparisons']} ROI comparisons; maximum absolute difference {independent_check['maximum_absolute_error_au_um2']:.3g} model a.u.·µm².",
            "- Intensity scale remains model a.u.; no camera counts, dark/flat correction, sensor registration, or real CMOS measurement is available.",
            "- Full numerical accuracy under the new fixed ROIs remains unmeasured for the entire 2,000-row population because only 36 raw maps were saved; computing it would require another forward pass.",
            "- The absolute export uses one shared full-map linear scale. The contrast export applies sqrt(model intensity) to every panel and uses one pooled-window P99.9 transformed range; no panel has independent normalization.",
            "",
            "## Outputs",
            "",
            "See the JSON report and CSV tables in this folder. The PNG/PDF/SVG figures display the 200 × 200 µm window enlarged for visibility; the enlargement is display-only.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "orientation_summary": orientation_summary}, indent=2))


if __name__ == "__main__":
    main()
