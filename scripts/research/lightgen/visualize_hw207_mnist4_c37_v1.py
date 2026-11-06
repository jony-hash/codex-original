"""HW207 MNIST4 C3.7 P5: CMOS 2x2 spot visualisation.

This entry point only consumes arrays exported by
``execute_hw207_mnist4_c37_v1.py vizdata``.  It never synthesises or brightens a
spot: every displayed CMOS map is a real frozen-phase forward result written by
the propagation path.

Two intensity conventions are produced for every intensity figure:

* ``linear``  one absolute linear colour scale shared by all panels of a figure;
* ``sqrt``    the same arrays under a common square-root display mapping.

Samples are never normalised individually, so panel brightness stays comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.research.lightgen import hw207_mnist4_c37_v1 as c37


FRAME_EXTENT_UM = (-c37.CMOS / 2.0 * c37.CMOS_PITCH_UM, c37.CMOS / 2.0 * c37.CMOS_PITCH_UM,
                   -c37.CMOS / 2.0 * c37.CMOS_PITCH_UM, c37.CMOS / 2.0 * c37.CMOS_PITCH_UM)
ZOOM_UM = 1100.0
DPI = 300


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("matplotlib is required for the C3.7 P5 figures") from error
    return plt, Rectangle


def load_intensity(path: Path) -> np.ndarray:
    """Read one exported CMOS intensity map without rescaling it."""
    data = np.load(path, allow_pickle=False)
    array = data["intensity"] if "intensity" in getattr(data, "files", []) else data
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (c37.CMOS, c37.CMOS):
        raise ValueError(f"{path} is {array.shape}, expected a full {c37.CMOS}x{c37.CMOS} frame")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite intensities")
    return array


def roi_pixel_extent(bounds: Sequence[int]) -> tuple[float, float, float, float]:
    """Rectangle in physical micrometres for one frozen ROI or core box."""
    r0, r1, c0, c1 = bounds
    y0 = (r0 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM
    y1 = (r1 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM
    x0 = (c0 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM
    x1 = (c1 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM
    return (x0, y0, x1 - x0, y1 - y0)


def draw_regions(axis, record: dict[str, Any], spec: dict[str, Any]) -> None:
    """Draw the frozen four ROIs, the target core, the peak and the centroid."""
    _, Rectangle = _plt()
    target = int(record["true_label"])
    prediction = int(record["prediction"])
    for index, roi in enumerate(spec["records"]):
        x, y, width, height = roi_pixel_extent(roi["roi_bounds_px"])
        style = "solid" if index == target else "solid"
        colour = "#22c55e" if index == target else "#ffffff"
        width_line = 1.6 if index == target else 0.8
        axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=colour,
                                 linewidth=width_line, linestyle=style, alpha=0.95))
        if index == prediction and prediction != target:
            axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ef4444",
                                     linewidth=1.6, linestyle="--", alpha=0.95))
        cx, cy, core_width, core_height = roi_pixel_extent(roi["core_bounds_px"])
        axis.add_patch(Rectangle((cx, cy), core_width, core_height, fill=False,
                                 edgecolor="#fbbf24" if index == target else "#9ca3af",
                                 linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.95))
        axis.text(x + 8, y + height - 40, f"ROI{index}", color=colour, fontsize=6.5,
                  ha="left", va="top")
    peak_um = ((record["global_peak_col"] - c37.FRAME_CENTER_PX) * c37.CMOS_PITCH_UM,
               (record["global_peak_row"] - c37.FRAME_CENTER_PX) * c37.CMOS_PITCH_UM)
    centroid_um = ((record["centroid_col"] - c37.FRAME_CENTER_PX) * c37.CMOS_PITCH_UM,
                   (record["centroid_row"] - c37.FRAME_CENTER_PX) * c37.CMOS_PITCH_UM)
    axis.plot(peak_um[0], peak_um[1], marker="x", markersize=5.0, color="#f472b6",
              markeredgewidth=1.2, label="global peak")
    axis.plot(centroid_um[0], centroid_um[1], marker="+", markersize=6.0, color="#38bdf8",
              markeredgewidth=1.2, label="centroid")


def _display(array: np.ndarray, mode: str) -> np.ndarray:
    return array if mode == "linear" else np.sqrt(np.maximum(array, 0.0))


def _save(figure, path_stem: Path) -> list[str]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf", ".svg"):
        target = path_stem.with_suffix(suffix)
        figure.savefig(target, dpi=DPI, bbox_inches="tight", facecolor="white")
        written.append(str(target))
    return written


def _colour_limits(arrays: Sequence[np.ndarray], mode: str) -> tuple[float, float]:
    stacked = np.stack([_display(array, mode) for array in arrays])
    return 0.0, float(stacked.max())


def figure_main_rows(payload: dict[str, Any], work_dir: Path, candidate: str,
                     mode: str, out: Path) -> dict[str, Any]:
    plt, _ = _plt()
    spec = payload["core_spec"]
    exemplars = [item for item in payload["class_exemplars"] if item["candidate"] == candidate]
    exemplars.sort(key=lambda item: item["class"])
    if len(exemplars) != 4:
        raise ValueError(f"{candidate} does not have four class exemplars")
    maps = [load_intensity(work_dir / "viz" / item["map_path"]) for item in exemplars]
    vmin, vmax = _colour_limits(maps, mode)
    phase = np.load(work_dir / "viz" / f"phase_{candidate}.npy", allow_pickle=False)
    figure = plt.figure(figsize=(15.0, 12.0), constrained_layout=True)
    grid = figure.add_gridspec(4, 4, width_ratios=[1.0, 0.9, 1.25, 1.25])
    for row, (item, image) in enumerate(zip(exemplars, maps)):
        record = item["record"]
        axis_input = figure.add_subplot(grid[row, 0])
        digit = np.array(record.get("input_image", []), dtype=np.float64)
        if digit.size:
            axis_input.imshow(digit.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        axis_input.set_title(f"input {item['true_label']}", fontsize=8)
        axis_input.set_xticks([])
        axis_input.set_yticks([])
        axis_phase = figure.add_subplot(grid[row, 1])
        image_phase = axis_phase.imshow(np.mod(phase, 2 * math.pi), cmap="twilight",
                                        vmin=0, vmax=2 * math.pi)
        axis_phase.set_title("frozen phase (rad)", fontsize=8)
        axis_phase.set_xticks([])
        axis_phase.set_yticks([])
        if row == 3:
            colourbar = figure.colorbar(image_phase, ax=axis_phase, fraction=0.046, pad=0.02)
            colourbar.ax.tick_params(labelsize=6)
        for column, (extent, title) in enumerate((
                (FRAME_EXTENT_UM, "full CMOS frame"),
                ((-ZOOM_UM, ZOOM_UM, -ZOOM_UM, ZOOM_UM), "ROI block zoom"))):
            axis = figure.add_subplot(grid[row, 2 + column])
            axis.imshow(_display(image, mode), cmap="magma", vmin=vmin, vmax=vmax,
                        origin="lower", extent=FRAME_EXTENT_UM)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            draw_regions(axis, {**record, "true_label": item["true_label"],
                                "prediction": item["prediction"]}, spec)
            axis.set_title(f"{title} | true {item['true_label']} pred {item['prediction']}",
                           fontsize=8)
            axis.tick_params(labelsize=6)
            axis.set_xlabel("x (um)", fontsize=7)
            axis.set_ylabel("y (um)", fontsize=7)
        energies = item["roi_energies_um2"]
        axis_zoom = figure.axes[-1]
        axis_zoom.text(0.02, 0.02,
                       "ROI energy (um^2): " + " ".join(f"{value:.0f}" for value in energies)
                       + f"\ncore fraction {record['core_fraction']:.3f}"
                       + f" | share {record['target_roi_share_of_cmos']:.3f}",
                       transform=axis_zoom.transAxes, fontsize=6.2, color="white",
                       va="bottom", ha="left",
                       bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5})
    figure.suptitle(f"C37 {candidate} | CMOS four-ROI spots ({mode} display, shared colour scale)",
                    fontsize=11)
    files = _save(figure, out / f"c37_fig01_four_class_spots_{candidate}_{mode}")
    plt.close(figure)
    return {"figure": "c37_fig01", "candidate": candidate, "mode": mode, "files": files,
            "vmin": vmin, "vmax": vmax,
            "samples": [{"class": item["class"], "row": item["row"]} for item in exemplars]}


def figure_before_after(payload: dict[str, Any], work_dir: Path, candidates: Sequence[str],
                        mode: str, out: Path) -> dict[str, Any]:
    plt, _ = _plt()
    spec = payload["core_spec"]
    exemplars = {name: {item["class"]: item for item in payload["class_exemplars"]
                        if item["candidate"] == name} for name in candidates}
    rows = max(len(mapping) for mapping in exemplars.values())
    maps = [load_intensity(work_dir / "viz" / mapping[cls]["map_path"])
            for mapping in exemplars.values() for cls in sorted(mapping)]
    vmin, vmax = _colour_limits(maps, mode)
    figure = plt.figure(figsize=(4.2 * len(candidates) * 2, 4.0 * rows), constrained_layout=True)
    grid = figure.add_gridspec(rows, len(candidates) * 2)
    for column, name in enumerate(candidates):
        for cls in sorted(exemplars[name]):
            item = exemplars[name][cls]
            image = load_intensity(work_dir / "viz" / item["map_path"])
            record = {**item["record"], "true_label": item["true_label"],
                      "prediction": item["prediction"]}
            for offset, (extent, title) in enumerate((
                    ((-ZOOM_UM, ZOOM_UM, -ZOOM_UM, ZOOM_UM), f"{name} zoom"),
                    (FRAME_EXTENT_UM, f"{name} full"))):
                axis = figure.add_subplot(grid[cls, column * 2 + offset])
                axis.imshow(_display(image, mode), cmap="magma", vmin=vmin, vmax=vmax,
                            origin="lower", extent=FRAME_EXTENT_UM)
                axis.set_xlim(extent[0], extent[1])
                axis.set_ylim(extent[2], extent[3])
                draw_regions(axis, record, spec)
                axis.set_title(f"class {cls} | {title}", fontsize=8)
                axis.tick_params(labelsize=6)
                axis.text(0.02, 0.02,
                          f"core {record['core_fraction']:.3f} | share "
                          f"{record['target_roi_share_of_cmos']:.3f} | peak "
                          f"{'in' if record['global_peak_in_target_roi'] else 'out'} ROI",
                          transform=axis.transAxes, fontsize=6.0, color="white", va="bottom",
                          bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5})
    figure.suptitle(f"C37 before/after compare ({mode} display, shared colour scale)", fontsize=11)
    files = _save(figure, out / f"c37_fig02_before_after_{mode}")
    plt.close(figure)
    return {"figure": "c37_fig02", "mode": mode, "candidates": list(candidates), "files": files}


AVERAGE_CROP = (640, 768)
AVERAGE_EXTENT_UM = (-AVERAGE_CROP[1] / 2.0 * c37.CMOS_PITCH_UM,
                     AVERAGE_CROP[1] / 2.0 * c37.CMOS_PITCH_UM,
                     -AVERAGE_CROP[1] / 2.0 * c37.CMOS_PITCH_UM,
                     AVERAGE_CROP[1] / 2.0 * c37.CMOS_PITCH_UM)


def load_average(path: Path) -> np.ndarray:
    """Read one exported class-average map (central 768x768 window)."""
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if array.shape != (AVERAGE_CROP[1], AVERAGE_CROP[1]):
        raise ValueError(f"{path} is {array.shape}, expected the central "
                         f"{AVERAGE_CROP[1]}x{AVERAGE_CROP[1]} window")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite intensities")
    return array


def figure_class_averages(payload: dict[str, Any], work_dir: Path, candidate: str,
                          mode: str, out: Path) -> dict[str, Any]:
    plt, Rectangle = _plt()
    spec = payload["core_spec"]
    entry = payload["class_averages"][candidate]
    all_maps = [load_average(work_dir / "viz" / entry["paths"][str(cls)]) for cls in range(4)]
    correct_maps = [load_average(work_dir / "viz" / entry["correct_paths"][str(cls)])
                    for cls in range(4)]
    vmin, vmax = _colour_limits(all_maps + correct_maps, mode)
    figure = plt.figure(figsize=(12.0, 12.0), constrained_layout=True)
    grid = figure.add_gridspec(4, 2)
    for cls in range(4):
        for column, (image, label) in enumerate((
                (all_maps[cls], "all development validation samples"),
                (correct_maps[cls],
                 f"correct only (n={entry['correct_counts'][str(cls)]})"))):
            axis = figure.add_subplot(grid[cls, column])
            axis.imshow(_display(image, mode), cmap="inferno", vmin=vmin, vmax=vmax,
                        origin="lower", extent=AVERAGE_EXTENT_UM)
            axis.set_xlim(-ZOOM_UM, ZOOM_UM)
            axis.set_ylim(-ZOOM_UM, ZOOM_UM)
            for index, roi in enumerate(spec["records"]):
                x, y, width, height = roi_pixel_extent(roi["roi_bounds_px"])
                axis.add_patch(Rectangle((x, y), width, height, fill=False,
                                         edgecolor="#22c55e" if index == cls else "#ffffff",
                                         linewidth=1.3 if index == cls else 0.7))
                cx, cy, cw, ch = roi_pixel_extent(roi["core_bounds_px"])
                axis.add_patch(Rectangle((cx, cy), cw, ch, fill=False,
                                         edgecolor="#fbbf24" if index == cls else "#9ca3af",
                                         linewidth=0.8, linestyle=(0, (2, 2))))
            axis.set_title(f"class {cls} mean CMOS | {label}", fontsize=8)
            axis.tick_params(labelsize=6)
            axis.set_xlabel("x (um)", fontsize=7)
            axis.set_ylabel("y (um)", fontsize=7)
    figure.suptitle(f"C37 {candidate} class average CMOS ({mode}, shared colour scale, "
                    f"n={entry['rows_per_class']}/class)", fontsize=11)
    files = _save(figure, out / f"c37_fig03_class_average_{candidate}_{mode}")
    plt.close(figure)
    return {"figure": "c37_fig03", "candidate": candidate, "mode": mode, "files": files}


def figure_energy_routing(records: dict[str, np.ndarray], candidate: str, out: Path
                          ) -> dict[str, Any]:
    plt, _ = _plt()
    labels = records["labels"].astype(int)
    scores = records["roi_scores"]
    predictions = records["predictions"].astype(int)
    core = records["core_fraction"]
    eta = records["eta_phase_to_cmos"]
    cmos = records["cmos_energy_um2"]
    matrix = np.zeros((4, 4))
    for truth in range(4):
        mask = labels == truth
        matrix[truth] = scores[mask].mean(0) / max(cmos[mask].mean(), 1e-12)
    confusion = np.zeros((4, 4), dtype=int)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    image = axes[0, 0].imshow(matrix, cmap="viridis")
    axes[0, 0].set_title("mean ROI energy / mean full CMOS energy (normalised in rows)")
    axes[0, 0].set_xlabel("ROI index")
    axes[0, 0].set_ylabel("true class")
    for row in range(4):
        for column in range(4):
            axes[0, 0].text(column, row, f"{matrix[row, column]:.2f}", ha="center",
                            va="center", color="white", fontsize=8)
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)
    image = axes[0, 1].imshow(confusion, cmap="Blues")
    axes[0, 1].set_title("confusion matrix (row = true, column = predicted)")
    for row in range(4):
        for column in range(4):
            axes[0, 1].text(column, row, str(confusion[row, column]), ha="center", va="center",
                            color="black" if confusion[row, column] < confusion.max() / 2 else "white",
                            fontsize=8)
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)
    for cls in range(4):
        mask = labels == cls
        axes[1, 0].plot(np.sort(core[mask]), np.linspace(0, 1, mask.sum()),
                        label=f"class {cls}")
    axes[1, 0].set_title("target core energy fraction CDF")
    axes[1, 0].set_xlabel("core fraction")
    axes[1, 0].set_ylabel("cumulative fraction")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].hist(eta, bins=40, color="#0ea5e9")
    axes[1, 1].set_title(f"phase-to-CMOS efficiency (mean {eta.mean():.4f}, "
                         f"p10 {np.percentile(eta, 10):.4f})")
    axes[1, 1].set_xlabel("eta")
    axes[1, 1].grid(alpha=0.3)
    figure.suptitle(f"C37 {candidate} energy routing summary (n={len(labels)})", fontsize=11)
    files = _save(figure, out / f"c37_fig04_energy_routing_{candidate}")
    plt.close(figure)
    return {"figure": "c37_fig04", "candidate": candidate, "files": files,
            "roi_energy_matrix": matrix.tolist(), "confusion_matrix": confusion.tolist(),
            "eta_mean": float(eta.mean()), "eta_p10": float(np.percentile(eta, 10)),
            "core_fraction_median": float(np.median(core))}


def figure_examples(payload: dict[str, Any], work_dir: Path, candidate: str, mode: str,
                    out: Path) -> dict[str, Any]:
    plt, _ = _plt()
    spec = payload["core_spec"]
    items = payload.get("error_exemplars", {}).get(candidate, [])
    items = items + [item for item in payload["class_exemplars"] if item["candidate"] == candidate]
    if not items:
        raise ValueError(f"no exemplars available for {candidate}")
    maps = [load_intensity(work_dir / "viz" / item["map_path"]) for item in items]
    vmin, vmax = _colour_limits(maps, mode)
    figure = plt.figure(figsize=(4.0 * 4, 3.9 * math.ceil(len(items) / 4)),
                        constrained_layout=True)
    grid = figure.add_gridspec(math.ceil(len(items) / 4), 4)
    for index, (item, image) in enumerate(zip(items, maps)):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        axis.imshow(_display(image, mode), cmap="magma", vmin=vmin, vmax=vmax,
                    origin="lower", extent=FRAME_EXTENT_UM)
        axis.set_xlim(-ZOOM_UM, ZOOM_UM)
        axis.set_ylim(-ZOOM_UM, ZOOM_UM)
        draw_regions(axis, {**item["record"], "true_label": item["true_label"],
                            "prediction": item["prediction"]}, spec)
        status = "correct" if item["true_label"] == item["prediction"] else "WRONG"
        axis.set_title(f"row {item['row']} true {item['true_label']} "
                       f"pred {item['prediction']} ({status})", fontsize=8)
        axis.tick_params(labelsize=6)
    figure.suptitle(f"C37 {candidate} typical correct and misclassified samples "
                    f"({mode}, shared colour scale)", fontsize=11)
    files = _save(figure, out / f"c37_fig05_examples_{candidate}_{mode}")
    plt.close(figure)
    return {"figure": "c37_fig05", "candidate": candidate, "mode": mode, "files": files,
            "rows": [item["row"] for item in items]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--compare", default=None,
                        help="comma separated candidate list for the before/after figure")
    args = parser.parse_args(argv)
    work_dir: Path = args.work_dir
    payload = json.loads((work_dir / "c37_vizdata.json").read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    selected = args.candidate or candidates[-1]
    compare = (args.compare.split(",") if args.compare
               else [name for name in candidates if "E0" in name or "E5" in name][:1] + [selected])
    out = work_dir / "figures"
    manifest: dict[str, Any] = {"identity": "C3.7_P5_VISUALISATION", "dpi": DPI,
                                "work_dir": str(work_dir), "candidate": selected,
                                "compare": compare, "figures": [],
                                "intensity_conventions": {
                                    "linear": "absolute intensity, one shared linear scale per figure",
                                    "sqrt": "same arrays under a shared square-root display mapping",
                                },
                                "note": "no per-sample normalisation and no synthetic spot is used"}
    for mode in ("linear", "sqrt"):
        manifest["figures"].append(figure_main_rows(payload, work_dir, selected, mode, out))
        manifest["figures"].append(figure_before_after(payload, work_dir, compare, mode, out))
        manifest["figures"].append(figure_class_averages(payload, work_dir, selected, mode, out))
        manifest["figures"].append(figure_examples(payload, work_dir, selected, mode, out))
    records_path = work_dir / "viz" / f"validation_records_{selected}.npz"
    if records_path.is_file():
        records = np.load(records_path, allow_pickle=False)
        manifest["figures"].append(figure_energy_routing(records, selected, out))
    c37.write_json(work_dir / "c37_visualization_manifest.json", manifest)
    print(json.dumps({"figures": len(manifest["figures"]), "candidate": selected,
                      "compare": compare}, indent=2))


if __name__ == "__main__":
    main()