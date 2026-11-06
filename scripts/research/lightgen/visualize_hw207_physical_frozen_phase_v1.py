"""Generate fixed-scale CMOS figures and QA evidence for the HW207 P0-P7 run."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

from scripts.research.lightgen import hw207_physical_frozen_phase_v1 as model


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_physical_frozen_phase_p0_p7_v2.json"
ROI_COLORS = ("#43a2ca", "#fdae61", "#66bd63", "#d7191c")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _save_triplet(fig: plt.Figure, target: Path) -> list[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in (".png", ".pdf", ".svg"):
        path = target.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        saved.append(path)
    plt.close(fig)
    return saved


def _affine_axes(ax: plt.Axes, matrix: np.ndarray, offset: np.ndarray,
                 rows: int, columns: int, crop: tuple[int, int, int, int] | None = None) -> None:
    transform_matrix = np.asarray(
        ((matrix[0, 0], matrix[0, 1], offset[0]),
         (matrix[1, 0], matrix[1, 1], offset[1]),
         (0.0, 0.0, 1.0)), dtype=np.float64,
    )
    transform = Affine2D(transform_matrix) + ax.transData
    if crop is None:
        r0, r1, c0, c1 = 0, rows, 0, columns
    else:
        r0, r1, c0, c1 = crop
    corners = np.asarray(((c0, r0), (c1, r0), (c0, r1), (c1, r1)), dtype=np.float64)
    physical = corners @ matrix.T + offset
    extent = (float(c0), float(c1), float(r1), float(r0))
    ax._hw207_image_extent = extent
    ax._hw207_image_transform = transform
    ax._hw207_physical_bounds = physical


def _draw_rois(ax: plt.Axes, config: dict[str, Any], target_class: int,
               matrix: np.ndarray, offset: np.ndarray, *, local: bool) -> None:
    for key, item in config["fixed_roi_geometry"]["class_to_roi"].items():
        x0, x1 = (float(x) for x in item["x_um"])
        y0, y1 = (float(y) for y in item["y_um"])
        is_target = int(key) == int(target_class)
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="#ffe34d" if is_target else "white",
            linewidth=2.3 if is_target else 0.9,
            linestyle="-" if is_target else "--", zorder=5,
        ))
        if local:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"ROI {key}",
                    ha="center", va="center", color="white", fontsize=7,
                    weight="bold", zorder=6,
                    bbox={"facecolor": "black", "alpha": 0.35, "pad": 1})


def _imshow_cmos(ax: plt.Axes, intensity: np.ndarray, config: dict[str, Any],
                 matrix: np.ndarray, offset: np.ndarray, vmax: float,
                 *, transform: str, crop: tuple[int, int, int, int] | None = None,
                 target_class: int = 0, markers: dict[str, Any] | None = None) -> Any:
    rows, columns = intensity.shape
    if crop is None:
        display = intensity
    else:
        r0, r1, c0, c1 = crop
        display = intensity[r0:r1, c0:c1]
    _affine_axes(ax, matrix, offset, rows, columns, crop)
    values = np.sqrt(np.maximum(display, 0.0)) if transform == "sqrt" else display
    norm_vmax = math.sqrt(vmax) if transform == "sqrt" else vmax
    image = ax.imshow(
        values, origin="upper", extent=ax._hw207_image_extent,
        transform=ax._hw207_image_transform, cmap="inferno",
        norm=Normalize(vmin=0.0, vmax=max(norm_vmax, 1e-30)), interpolation="nearest",
        rasterized=True,
    )
    bounds = ax._hw207_physical_bounds
    xmin, xmax = float(bounds[:, 0].min()), float(bounds[:, 0].max())
    ymin, ymax = float(bounds[:, 1].min()), float(bounds[:, 1].max())
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    _draw_rois(ax, config, target_class, matrix, offset, local=crop is not None)
    if markers:
        peak = markers.get("peak_x_y_in_window_um")
        centroid = markers.get("centroid_x_y_in_window_um")
        if peak and peak[0] is not None:
            ax.scatter([peak[0]], [peak[1]], marker="x", s=52, color="#00ffff",
                       linewidths=1.8, zorder=8, label="global peak")
        if centroid and centroid[0] is not None:
            ax.scatter([centroid[0]], [centroid[1]], marker="+", s=70, color="white",
                       linewidths=2.0, zorder=8, label="intensity centroid")
    ax.set_xlabel("CMOS physical x (µm)", fontsize=7)
    ax.set_ylabel("CMOS physical y (µm, down)", fontsize=7)
    ax.tick_params(labelsize=6)
    return image


def _format_candidate(candidate_id: str) -> str:
    return candidate_id.replace("_", "\n")


def _candidate_inputs(run_dir: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    p0 = _load_json(run_dir / "p0" / "p0_asset_and_data_audit.json")
    p5 = _load_json(run_dir / "p5" / "frozen_inference_summary.json")
    if p5.get("candidate_count") != 7 or p5.get("row_count") != 2475:
        raise ValueError("P5 must contain all seven candidates and 2,475 validation rows")
    if p5.get("official_test_opened") is not False or p5.get("phase_updates") != 0:
        raise ValueError("P5 evidence violates the frozen-run boundary")
    c0 = np.load(run_dir / "p0" / "c0_validation_0_3.npz", allow_pickle=False)
    indices = c0["original_indices"].astype(np.int64)
    labels = c0["labels"].astype(np.int64)
    images = c0["images"]
    if len(indices) != 2475 or len(np.unique(indices)) != len(indices):
        raise ValueError("P0 population is incomplete or has duplicate original indices")
    if any(int(np.sum(labels == k)) != int(p0["data_ledger"]["class_counts"][str(k)]) for k in range(4)):
        raise ValueError("P0 class counts disagree with the saved C0 rows")
    candidates = []
    for summary in p5["candidate_summaries"]:
        candidate_id = summary["candidate_id"]
        folder = run_dir / "p5" / candidate_id
        records = _load_json(folder / "representative_records.json")
        rows = np.load(folder / "validation_rows.npz", allow_pickle=False)
        if rows["original_indices"].shape != indices.shape or not np.array_equal(rows["original_indices"], indices):
            raise ValueError(f"P5 original-index order mismatch: {candidate_id}")
        if not np.array_equal(rows["true_labels"], labels):
            raise ValueError(f"P5 labels mismatch: {candidate_id}")
        if summary.get("sample_count") != len(indices) or not summary.get("rows_and_labels_order_verified"):
            raise ValueError(f"P5 candidate row ledger is incomplete: {candidate_id}")
        means = np.load(folder / "class_mean_cmos_float32.npy", mmap_mode="r", allow_pickle=False)
        representatives = np.load(folder / "representative_cmos_float32.npy", mmap_mode="r", allow_pickle=False)
        if means.shape != (4, 2048, 2048) or representatives.shape != means.shape:
            raise ValueError(f"P5 full CMOS map shapes are invalid: {candidate_id}")
        candidates.append({
            "summary": summary, "candidate_id": candidate_id, "folder": folder,
            "records": records["records_by_true_class"], "rows": rows,
            "means": means, "representatives": representatives,
        })
    if len(candidates) != 7:
        raise ValueError("expected exactly seven unique candidate summaries")
    return candidates, p5, indices, labels, images


def _global_intensity_max(candidates: list[dict[str, Any]]) -> float:
    maximum = 0.0
    for candidate in candidates:
        for maps in (candidate["means"], candidate["representatives"]):
            value = float(np.max(maps))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid intensity values in {candidate['candidate_id']}")
            maximum = max(maximum, value)
    if maximum <= 0:
        raise ValueError("all representative and class-mean CMOS maps are zero")
    return maximum


def _save_overviews(candidates: list[dict[str, Any]], config: dict[str, Any],
                    images: np.ndarray, labels: np.ndarray, vmax: float,
                    output_dir: Path) -> list[Path]:
    matrix, offset = model.cmos_pixel_to_window_affine(config)
    rows, columns = (int(x) for x in config["cmos"]["sensor_shape_rows_columns"])
    window_um = config["fixed_roi_geometry"]["physical_window_um"]
    half = int(math.ceil(float(window_um[0]) / (2 * float(config["cmos"]["pixel_pitch_um"]))))
    center_row, center_col = rows // 2, columns // 2
    crop = (center_row - half, center_row + half, center_col - half, center_col + half)
    dmd_rows = [int(np.flatnonzero(labels == label)[0]) for label in range(4)]
    all_score_max = max(
        float(max(item["roi_scores"]))
        for candidate in candidates for item in candidate["records"].values()
    )
    saved: list[Path] = []
    for candidate in candidates:
        fig, axes = plt.subplots(4, 7, figsize=(21.5, 12.5), squeeze=False)
        fig.suptitle(
            f"{candidate['candidate_id']} | fixed 2×2 CMOS ROI, C0 validation representative rows\n"
            f"Common absolute intensity limits: linear 0–{vmax:.4g}; sqrt 0–{math.sqrt(vmax):.4g} (model units)",
            fontsize=12,
        )
        for label in range(4):
            row = dmd_rows[label]
            measurement = candidate["records"][str(label)]
            frame = np.asarray(candidate["representatives"][label], dtype=np.float32)
            phase = np.load(
                model.repo_path(next(x["phase_path"] for x in model.candidate_metadata(config)
                                     if x["candidate_id"] == candidate["candidate_id"])),
                allow_pickle=False,
            )
            ax = axes[label, 0]
            ax.imshow(images[row], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title("DMD command crop\n28×28 mirrors; outside OFF", fontsize=8)
            ax.set_ylabel(f"True {label}\nidx {measurement['original_index']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[label, 1]
            phase_image = ax.imshow(np.mod(phase, 2 * np.pi), cmap="twilight",
                                    vmin=0, vmax=2 * np.pi, interpolation="nearest")
            ax.set_title("Frozen phase\nwrapped for display only", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if label == 0:
                fig.colorbar(phase_image, ax=ax, fraction=0.046, pad=0.03, label="rad")

            ax = axes[label, 2]
            linear_image = _imshow_cmos(ax, frame, config, matrix, offset, vmax,
                                        transform="linear", target_class=label, markers=measurement)
            ax.set_title("Full CMOS · linear absolute", fontsize=8)
            if label == 0:
                fig.colorbar(linear_image, ax=ax, fraction=0.046, pad=0.03, label="intensity")

            ax = axes[label, 3]
            sqrt_image = _imshow_cmos(ax, frame, config, matrix, offset, vmax,
                                      transform="sqrt", target_class=label, markers=measurement)
            ax.set_title("Full CMOS · sqrt; same absolute scale", fontsize=8)
            if label == 0:
                fig.colorbar(sqrt_image, ax=ax, fraction=0.046, pad=0.03, label="sqrt(intensity)")

            ax = axes[label, 4]
            local_linear = _imshow_cmos(ax, frame, config, matrix, offset, vmax,
                                        transform="linear", crop=crop, target_class=label, markers=measurement)
            ax.set_title("200 µm region · linear absolute\nfixed ROI overlay", fontsize=8)
            if label == 0:
                fig.colorbar(local_linear, ax=ax, fraction=0.046, pad=0.03, label="intensity")

            ax = axes[label, 5]
            local_sqrt = _imshow_cmos(ax, frame, config, matrix, offset, vmax,
                                      transform="sqrt", crop=crop, target_class=label, markers=measurement)
            ax.set_title("200 µm region · sqrt; same absolute scale", fontsize=8)
            if label == 0:
                fig.colorbar(local_sqrt, ax=ax, fraction=0.046, pad=0.03, label="sqrt(intensity)")

            ax = axes[label, 6]
            scores = np.asarray(measurement["roi_scores"], dtype=np.float64)
            ax.bar(np.arange(4), scores, color=ROI_COLORS)
            ax.set_ylim(0, all_score_max * 1.12)
            ax.set_xticks(range(4), ["ROI 0", "ROI 1", "ROI 2", "ROI 3"], rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("integrated energy\n(model units·µm²)", fontsize=7)
            ax.set_title(
                f"true={label}, pred={measurement['prediction']}\n"
                f"ηDMD→CMOS={measurement['dmd_to_cmos_efficiency']:.3g}", fontsize=8,
            )
            ax.tick_params(labelsize=7)
        fig.text(
            0.5, 0.006,
            "Full-frame and local maps share one absolute scale across all seven candidates. "
            "ROI classes are fixed 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right; "
            "× is global peak, + is intensity centroid. Geometry and detector response are diagnostic assumptions.",
            ha="center", fontsize=8,
        )
        saved.extend(_save_triplet(fig, output_dir / f"{candidate['candidate_id']}_overview"))
    return saved


def _save_class_means(candidates: list[dict[str, Any]], config: dict[str, Any],
                      vmax: float, output_dir: Path) -> list[Path]:
    matrix, offset = model.cmos_pixel_to_window_affine(config)
    fig, axes = plt.subplots(7, 4, figsize=(13.5, 21.0), squeeze=False)
    fig.suptitle("Mean full CMOS intensity by true class · all C0 validation rows", fontsize=14)
    for row, candidate in enumerate(candidates):
        for label in range(4):
            ax = axes[row, label]
            img = _imshow_cmos(ax, np.asarray(candidate["means"][label]), config, matrix, offset,
                               vmax, transform="sqrt", target_class=label)
            if label == 0:
                ax.set_ylabel(candidate["candidate_id"], fontsize=7)
            ax.set_title(f"class {label} mean", fontsize=8)
            if row == 0:
                fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03, label="sqrt(intensity)")
    fig.text(0.5, 0.006,
             f"Common absolute square-root scale 0–{math.sqrt(vmax):.4g}; no per-map normalization.",
             ha="center", fontsize=9)
    return _save_triplet(fig, output_dir / "all_candidates_class_mean_cmos")


def _save_confusions(candidates: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.8), squeeze=False)
    fig.suptitle("Fixed CMOS ROI confusion matrices · abstentions shown as column 5", fontsize=14)
    last = None
    for index, candidate in enumerate(candidates):
        ax = axes.flat[index]
        metrics = candidate["summary"]["classification"]
        confusion = np.asarray(metrics["confusion_matrix_true_rows_predicted_columns"], dtype=np.int64)
        abstain = np.asarray(metrics["abstentions_by_true_class"], dtype=np.int64).reshape(4, 1)
        matrix = np.concatenate((confusion, abstain), axis=1)
        last = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        ax.set_title(f"{candidate['candidate_id']}\nAcc {metrics['accuracy']:.3f} · F1 {metrics['macro_f1']:.3f}", fontsize=8)
        ax.set_xticks(range(5), ["0", "1", "2", "3", "abst."], fontsize=7)
        ax.set_yticks(range(4), ["0", "1", "2", "3"], fontsize=7)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
        for y in range(4):
            for x in range(5):
                ax.text(x, y, str(int(matrix[y, x])), ha="center", va="center",
                        color="white" if matrix[y, x] > matrix.max() * 0.55 else "black", fontsize=7)
    axes.flat[-1].axis("off")
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, label="sample count")
    fig.text(0.5, 0.01, "Abstentions count as errors for accuracy and as false negatives for Macro-F1.",
             ha="center", fontsize=9)
    return _save_triplet(fig, output_dir / "all_candidates_confusion_matrices")


def _save_energy_and_localization(candidates: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    labels = [_format_candidate(x["candidate_id"]) for x in candidates]
    x = np.arange(len(candidates))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    fig.suptitle("Classification, energy delivery and spot-routing diagnostics", fontsize=14)
    accuracy = [c["summary"]["classification"]["accuracy"] for c in candidates]
    f1 = [c["summary"]["classification"]["macro_f1"] for c in candidates]
    axes[0, 0].bar(x - 0.18, accuracy, 0.36, label="Accuracy")
    axes[0, 0].bar(x + 0.18, f1, 0.36, label="Macro-F1")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Fixed 2×2 ROI classification")
    axes[0, 0].legend()
    axes[0, 0].set_xticks(x, labels, fontsize=7)

    mean_eta = np.asarray([c["summary"]["mean_dmd_to_cmos_efficiency"] for c in candidates])
    p10_eta = np.asarray([c["summary"]["p10_dmd_to_cmos_efficiency"] for c in candidates])
    axes[0, 1].scatter(mean_eta, accuracy, s=70, c=colors, label="mean η")
    axes[0, 1].scatter(p10_eta, accuracy, s=42, c=colors, marker="v", alpha=0.8, label="P10 η")
    for i, label in enumerate(labels):
        axes[0, 1].annotate(label.replace("\n", " "), (mean_eta[i], accuracy[i]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
    axes[0, 1].set_xlabel("DMD-to-full-CMOS energy ratio (model units)")
    axes[0, 1].set_ylabel("fixed ROI accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Accuracy–energy trade-off (no threshold asserted)")
    axes[0, 1].legend(fontsize=8)

    peak = [c["summary"]["peak_localization_accuracy"] for c in candidates]
    centroid = [c["summary"]["centroid_localization_accuracy"] for c in candidates]
    axes[1, 0].bar(x - 0.18, peak, 0.36, label="global peak in correct ROI")
    axes[1, 0].bar(x + 0.18, centroid, 0.36, label="CMOS intensity centroid in correct ROI")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Spot localization")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xticks(x, labels, fontsize=7)

    correct = [c["summary"]["mean_correct_roi_fraction_of_cmos"] for c in candidates]
    outside = [c["summary"]["mean_outside_four_roi_fraction_of_cmos"] for c in candidates]
    wrong_ratio = [c["summary"]["mean_correct_to_strongest_wrong_roi_ratio"] for c in candidates]
    axes[1, 1].bar(x - 0.24, correct, 0.24, label="correct ROI / full CMOS")
    axes[1, 1].bar(x, outside, 0.24, label="outside all four ROIs / full CMOS")
    axes[1, 1].bar(x + 0.24, wrong_ratio, 0.24, label="correct ROI / strongest wrong ROI")
    axes[1, 1].set_title("Energy concentration and ROI contrast")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_xticks(x, labels, fontsize=7)
    fig.text(0.5, 0.01,
             "All values are conditional on the assumed optical model and maximum tested sampling; "
             "no absolute efficiency or accuracy gate was specified.", ha="center", fontsize=9)
    return _save_triplet(fig, output_dir / "candidate_accuracy_energy_localization")


def _save_roi_distributions(candidates: list[dict[str, Any]], labels: np.ndarray,
                            output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    fig.suptitle("ROI energy shares across the complete C0 validation population", fontsize=14)
    class_colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    x = np.arange(len(candidates))
    for true_class, ax in enumerate(axes.flat):
        correct_values, wrong_values = [], []
        for candidate in candidates:
            fractions = np.asarray(candidate["rows"]["roi_fractions_of_cmos"], dtype=np.float64)
            scores = np.asarray(candidate["rows"]["roi_scores_model_units_um2"], dtype=np.float64)
            selection = labels == true_class
            correct_values.append(fractions[selection, true_class])
            mask_scores = scores[selection].copy()
            mask_scores[:, true_class] = -np.inf
            wrong_index = np.argmax(mask_scores, axis=1)
            wrong_values.append(fractions[selection, wrong_index])
        width = 0.34
        bp1 = ax.boxplot(correct_values, positions=x - width / 2, widths=0.28,
                         patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(wrong_values, positions=x + width / 2, widths=0.28,
                         patch_artist=True, showfliers=False)
        for patch in bp1["boxes"]:
            patch.set_facecolor(class_colors[true_class])
            patch.set_alpha(0.72)
        for patch in bp2["boxes"]:
            patch.set_facecolor("#bdbdbd")
            patch.set_alpha(0.85)
        ax.set_title(f"True class {true_class}")
        ax.set_ylabel("ROI energy / full CMOS energy")
        ax.set_ylim(bottom=0)
        ax.set_xticks(x, [_format_candidate(c["candidate_id"]) for c in candidates], fontsize=6)
        ax.grid(axis="y", alpha=0.2)
        if true_class == 0:
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
                      ["correct ROI", "strongest wrong ROI"], fontsize=8)
    fig.text(0.5, 0.01,
             "Each box contains all rows of that true class for a candidate; the ROI mapping is fixed.",
             ha="center", fontsize=9)
    return _save_triplet(fig, output_dir / "roi_energy_share_distributions")


def _save_error_examples(candidates: list[dict[str, Any]], config: dict[str, Any],
                         vmax: float, output_dir: Path) -> list[Path]:
    matrix, offset = model.cmos_pixel_to_window_affine(config)
    fig, axes = plt.subplots(7, 4, figsize=(12.0, 19.0), squeeze=False)
    fig.suptitle("First fixed-population misclassification example per true class", fontsize=14)
    for i, candidate in enumerate(candidates):
        with np.load(candidate["folder"] / "misclassified_examples_200um_windows.npz", allow_pickle=False) as error:
            labels = error["true_labels"].astype(int)
            for cls in range(4):
                ax = axes[i, cls]
                matches = np.flatnonzero(labels == cls)
                if not matches.size:
                    ax.text(0.5, 0.5, "no error for class", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                j = int(matches[0])
                image = error["cmos_window_intensity"][j]
                full_r0 = 1024 - image.shape[0] // 2
                full_c0 = 1024 - image.shape[1] // 2
                local_crop = (full_r0, full_r0 + image.shape[0], full_c0, full_c0 + image.shape[1])
                _affine_axes(ax, matrix, offset, 2048, 2048, local_crop)
                ax.imshow(np.sqrt(np.maximum(image, 0)), origin="upper", extent=ax._hw207_image_extent,
                          transform=ax._hw207_image_transform, cmap="inferno",
                          norm=Normalize(0, math.sqrt(vmax)), interpolation="nearest", rasterized=True)
                for key, roi in config["fixed_roi_geometry"]["class_to_roi"].items():
                    x0, x1 = roi["x_um"]
                    y0, y1 = roi["y_um"]
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           edgecolor="white", linewidth=0.8, linestyle="--"))
                ax.set_title(
                    f"true {cls} → pred {int(error['predictions'][j])}\n"
                    f"idx {int(error['original_indices'][j])}", fontsize=7,
                )
                bounds = ax._hw207_physical_bounds
                ax.set_xlim(float(bounds[:, 0].min()), float(bounds[:, 0].max()))
                ax.set_ylim(float(bounds[:, 1].max()), float(bounds[:, 1].min()))
                ax.tick_params(labelsize=6)
        axes[i, 0].set_ylabel(candidate["candidate_id"], fontsize=7)
    fig.text(0.5, 0.01, "Common square-root scale shared with candidate overview figures.",
             ha="center", fontsize=9)
    return _save_triplet(fig, output_dir / "first_misclassification_examples")


def _qa(files: list[Path], output_dir: Path, vmax: float,
        p5: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"figure output missing or empty: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
                if image.width < 1000 or image.height < 700:
                    raise ValueError(f"figure resolution too small: {path}")
                dimensions = [image.width, image.height]
        elif path.suffix == ".pdf":
            if not path.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"invalid PDF header: {path}")
            dimensions = None
        else:
            ET.parse(path)
            dimensions = None
        checks.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                       "format": path.suffix[1:], "dimensions_px": dimensions})
    return {
        "status": "P6_PASS",
        "figure_count": len(files),
        "candidate_count": len(candidates),
        "all_candidates_have_overview_triplet": all(
            all((output_dir / f"{c['candidate_id']}_overview{ext}").is_file()
                for ext in (".png", ".pdf", ".svg")) for c in candidates
        ),
        "common_absolute_intensity_vmax_model_units": vmax,
        "per_image_auto_normalization": False,
        "display_transforms": ["linear absolute intensity", "square-root of absolute intensity"],
        "human_visual_review": "generated; manual image visual inspection recorded separately in closeout",
        "files": checks,
        "sampling_limited": bool(p5.get("sampling_limited_diagnostic_only")),
        "status_scope": "figure generation, source binding and raster/vector file QA; not optical validity",
    }


def generate(run_dir: Path, output_dir: Path, config_path: Path) -> dict[str, Any]:
    config = model.read_json(config_path)
    candidates, p5, indices, labels, images = _candidate_inputs(run_dir, config)
    vmax = _global_intensity_max(candidates)
    output_dir.mkdir(parents=True, exist_ok=False)
    files: list[Path] = []
    files.extend(_save_overviews(candidates, config, images, labels, vmax, output_dir))
    files.extend(_save_class_means(candidates, config, vmax, output_dir))
    files.extend(_save_confusions(candidates, output_dir))
    files.extend(_save_energy_and_localization(candidates, output_dir))
    files.extend(_save_roi_distributions(candidates, labels, output_dir))
    files.extend(_save_error_examples(candidates, config, vmax, output_dir))
    result = _qa(files, output_dir, vmax, p5, candidates)
    result.update({
        "run_directory": str(run_dir.relative_to(ROOT)),
        "config_path": str(config_path.relative_to(ROOT)),
        "evaluation_population_original_indices": {
            "count": int(len(indices)), "first": int(indices[0]), "last": int(indices[-1]),
            "ordered_sha256_provenance_only": model.sha256_file(run_dir / "p0" / "c0_validation_0_3.npz"),
        },
        "class_counts": {str(k): int(np.sum(labels == k)) for k in range(4)},
        "p5_sampling_status": p5.get("sampling_convergence_status"),
    })
    model.write_json(output_dir / "figure_qa_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "p6" / "figures").resolve()
    result = generate(run_dir, output_dir, args.config.resolve())
    print(json.dumps({"status": result["status"], "figure_count": result["figure_count"],
                      "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0 if result["status"] == "P6_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
