"""Replot frozen HW207 C3.6 CMOS arrays with a large four-ROI display window."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.lightgen import visualize_hw207_mnist4_cmos_demo_v1 as plotter


ROOT = Path(__file__).resolve().parents[3]
EXPORT_FORMATS = (".pdf", ".svg", ".png")
RASTER_DPI = 300
FIGURE_STEMS = (
    "01_input_1_cmos_demo",
    "01_input_1_cmos_demo_p1",
    "02_four_class_optical_routing",
    "02_four_class_optical_routing_p1",
    "02_first_error_cases_p1",
    "02_first_error_cases_p2",
    "03_class_average_cmos",
    "03_class_average_cmos_p1",
    "04_roi_energy_matrix",
    "04_roi_energy_matrix_p1",
    "05_confusion_matrix",
    "05_confusion_matrix_p1",
    "06_c35_candidate_comparison",
)
SPATIAL_STEMS = set(FIGURE_STEMS[:8]) | {"06_c35_candidate_comparison"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment-tool", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_run.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source run does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    source_reports = source / "reports"
    source_arrays = source / "arrays"
    source_figures = source / "figures"
    manifest = read_json(source_reports / "visualization_manifest.json")
    selection = read_json(source_arrays / "sample_indices.json")
    if manifest.get("identity") != "HW207_C3.6_CMOS_2X2_OPTICAL_CLASSIFICATION_VISUALIZATION":
        raise ValueError("source run is not the expected HW207 C3.6 visualization")
    rectangles = tuple(tuple(int(v) for v in row) for row in manifest["roi"]["bounds"])
    expected_rectangles = plotter.assert_roi_contract(ROOT / "configs/research/lightgen")
    if rectangles != expected_rectangles:
        raise ValueError("source ROI bounds differ from the frozen C3.4 ROI contract")

    with np.load(source_arrays / "cmos_intensity_maps.npz", allow_pickle=False) as payload:
        map_payload = {key: payload[key] for key in payload.files}
    with np.load(source_arrays / "roi_energy_scores.npz", allow_pickle=False) as payload:
        score_payload = {key: payload[key] for key in payload.files}
    with np.load(source_arrays / "input_dmd_patterns.npz", allow_pickle=False) as payload:
        dmd_rows = payload["original_rows"].astype(np.int64)
        dmd_commands = payload["binary_commands"]
    dmd_by_row = {int(row): dmd_commands[i] for i, row in enumerate(dmd_rows)}

    results: dict[str, dict[str, Any]] = {}
    maps: dict[str, dict[int, np.ndarray]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for key in ("p1", "p2"):
        rows = score_payload[f"{key}_original_rows"].astype(np.int64)
        labels = score_payload[f"{key}_labels"].astype(np.int64)
        predictions = score_payload[f"{key}_predictions"].astype(np.int64)
        scores = score_payload[f"{key}_roi_scores_um2"]
        selected_rows = map_payload[f"{key}_sample_rows"].astype(np.int64)
        intensities = map_payload[f"{key}_sample_intensity"]
        if len(rows) != 2000 or scores.shape != (2000, 4):
            raise ValueError(f"{key} ROI score population is incomplete")
        if len(np.unique(rows)) != len(rows) or len(np.unique(selected_rows)) != len(selected_rows):
            raise ValueError(f"{key} contains duplicate row identifiers")
        if intensities.shape != (len(selected_rows), 2048, 2048):
            raise ValueError(f"{key} saved CMOS maps have unexpected shape: {intensities.shape}")
        positions = {int(row): i for i, row in enumerate(rows)}
        selected_positions = [positions.get(int(row)) for row in selected_rows]
        if any(position is None for position in selected_positions):
            raise ValueError(f"{key} CMOS maps contain a row absent from the ROI score population")
        if not np.array_equal(predictions, np.argmax(scores, axis=1)):
            raise ValueError(f"{key} saved predictions differ from fixed ROI argmax")
        maps[key] = {int(row): intensities[i] for i, row in enumerate(selected_rows)}
        for row in selection["selected_raw_cmos_rows_by_candidate"][key]:
            if int(row) not in maps[key]:
                raise ValueError(f"{key} selected row {row} is missing its saved CMOS map")
        if any(int(row) not in dmd_by_row for row in selected_rows):
            raise ValueError(f"{key} selected row is missing its saved DMD command")
        mean_all = map_payload[f"{key}_class_mean_all"]
        mean_correct = map_payload[f"{key}_class_mean_correct"]
        counts = map_payload[f"{key}_class_mean_counts"].astype(np.int64)
        result = {
            "rows": rows,
            "labels": labels,
            "predictions": predictions,
            "roi_scores": scores,
            "class_means": np.stack((mean_all, mean_correct), axis=1),
            "class_counts": counts,
        }
        results[key] = result
        model_entry = manifest["models"][key]
        candidates[key] = {"name": model_entry["name"]}
        stats[key] = model_entry["metrics"]

    images, labels_all, split = plotter.c34_runner._split_and_data()
    labels_np = labels_all.detach().cpu().numpy().astype(np.int64)
    validation_rows = split["validation"].detach().cpu().numpy().astype(np.int64)
    if len(validation_rows) != 2000 or not np.array_equal(np.bincount(labels_np[validation_rows], minlength=4), np.full(4, 500)):
        raise ValueError("loaded C3.4 validation population no longer matches the saved C3.6 population")
    for key in ("p1", "p2"):
        rows = results[key]["rows"]
        if not np.array_equal(rows, validation_rows) or not np.array_equal(results[key]["labels"], labels_np[rows]):
            raise ValueError(f"{key} saved rows or labels differ from current C3.4 validation data")

    output.mkdir(parents=True, exist_ok=False)
    figures_dir = output / "figures"
    reports_dir = output / "reports"
    qa_dir = reports_dir / "figure_qa"
    figures_dir.mkdir()
    qa_dir.mkdir(parents=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    require_alignment = plotter.load_alignment_checker(args.alignment_tool.resolve())
    rectangles = expected_rectangles
    vmax = float(manifest["checks"]["shared_color_scale"]["vmax"])
    norm = plotter.Normalize(0.0, vmax)
    demo_row = int(selection["input_1_demo_original_row"])
    for stem in FIGURE_STEMS:
        if stem not in SPATIAL_STEMS:
            for suffix in ("pdf", "png", "svg", "alignment.json", "alignment.svg"):
                source_path = source_figures / f"{stem}.{suffix}"
                if not source_path.is_file():
                    raise FileNotFoundError(f"unchanged figure component is missing: {source_path}")
                shutil.copy2(source_path, figures_dir / source_path.name)

    for key in ("p2", "p1"):
        result = results[key]
        cmos_maps = maps[key]
        candidate = candidates[key]
        image_stem = "01_input_1_cmos_demo" if key == "p2" else "01_input_1_cmos_demo_p1"
        plotter.figure_input_one(
            candidate, result, cmos_maps, dmd_by_row, images, rectangles, vmax, demo_row,
            figures_dir / image_stem, "HW207 MNIST4 · frozen single-layer optical classification",
            require_alignment,
        )
        plotter.figure_four_class(
            key, candidate, result, cmos_maps, selection, images, rectangles, vmax,
            figures_dir / ("02_four_class_optical_routing" if key == "p2" else "02_four_class_optical_routing_p1"),
            require_alignment,
        )
        plotter.figure_failures(
            key, candidate, result, cmos_maps, selection, images, rectangles, vmax,
            figures_dir / f"02_first_error_cases_{key}", require_alignment,
        )
        plotter.figure_class_means(
            candidate, result, rectangles, vmax,
            figures_dir / ("03_class_average_cmos" if key == "p2" else "03_class_average_cmos_p1"),
            require_alignment,
        )

    comparison = plotter.figure_comparison(
        candidates, results, stats, maps, demo_row, rectangles, vmax,
        figures_dir / "06_c35_candidate_comparison", require_alignment,
    )
    alignment_results = {}
    for path in sorted(figures_dir.glob("*.alignment.json")):
        report = read_json(path)
        if report.get("verdict") not in ("PASS", "NOT APPLICABLE"):
            raise RuntimeError(f"panel alignment failed: {path.name}: {report.get('verdict')}")
        alignment_results[path.name] = {"verdict": report["verdict"], "summary": report.get("summary", {})}
    if len(alignment_results) != 13:
        raise RuntimeError(f"expected alignment audit for 13 figure panels, found {len(alignment_results)}")

    output_manifest = {
        "identity": "HW207_C3.6_CMOS_2X2_OPTICAL_CLASSIFICATION_VISUALIZATION_ZOOM",
        "status": "NUMERICAL_DIAGNOSTIC_ONLY",
        "source_run": str(source),
        "source_artifact_references": [
            "arrays/cmos_intensity_maps.npz", "arrays/input_dmd_patterns.npz",
            "arrays/roi_energy_scores.npz", "arrays/sample_indices.json",
        ],
        "source_arrays_copied": False,
        "display": {
            "zoom_window_row_column_half_open": [824, 1224, 824, 1224],
            "zoom_window_shape_pixels": [400, 400],
            "full_cmos_shape_pixels": [2048, 2048],
            "approximate_linear_magnification": 2048 / 400,
            "color_scale": manifest["checks"]["shared_color_scale"],
            "roi_bounds_and_class_mapping": manifest["roi"],
            "source_pixel_array_preserved": True,
        },
        "checks": {
            "saved_rows_labels_roi_argmax_and_cmos_shapes": "PASS",
            "validation_dataset_rows_and_labels": "PASS",
            "panel_alignment": alignment_results,
            "optical_forward": "NONE",
            "optimizer_update": "NONE",
            "physical_dmd_or_cmos_operation": "NONE",
        },
        "models": manifest["models"],
        "outputs": {
            "figures": sorted(path.name for path in figures_dir.iterdir()),
            "report": "reports/visualization_report.md",
            "figure_qa_summary": "reports/figure_qa_summary.json",
            "figure_qa_reports": "reports/figure_qa/",
        },
    }
    write_json(reports_dir / "visualization_manifest.json", output_manifest)
    report_text = f"""# HW207 C3.6 CMOS 四区大窗口可视化重绘

## 科学认识

所有数值和判决均沿用来源 run 中的冻结结果，仅调整图件的显示窗口与版式。全幅 CMOS 为 2048×2048 像素；主显示采用行、列半开窗口 `[824:1224)`，大小 400×400 像素，约 5.12 倍线性放大。ROI 坐标、方向、线性色标、样本选择、积分分数和预测保持来源记录。

## 目标缺口

来源人口为已复用的 C3.4 开发验证集，不是独立测试。数值模拟不构成 CMOS 实测或硬件标定。P2 在共同线性色标下仍可能比 P1 暗，局部放大未改变强度映射。

## 工程检查

- 每个放大视图直接取自保存的全幅强度数组，输入像素与绘图数组逐像素一致；无重采样、转置或翻转。
- 四框仍使用固定 C3.4 半开坐标和 ROI 到类别映射；标签移至框外。局部图均配全幅定位图并标记窗口。
- 保存的 2,000 行、标签、ROI argmax、DMD 命令和 2048×2048 CMOS 图形状均通过重绘入口检查。
- 重绘只加载 C3.4 图像用于展示输入；没有运行光学 forward、模型更新或物理 DMD/CMOS 操作。
- 13 张图均生成 panel alignment 审查；结果记录于新 manifest 和对应 alignment JSON。
- PDF 字号、文字与图形碰撞及 1920×1080 屏幕显示尺寸验收写入 `reports/figure_qa_summary.json`。
- 完整强度数组继续引用来源 run，没有复制大型 NPZ 文件。

## 决策影响与下一步

大窗口提高四区光强及相邻泄漏的可辨识性，同时全幅定位图保留探测区域空间关系。共同色标保证 P1/P2 及样本间可比；如 P2 仍暗，应将其作为显示限制报告，不按图自动归一化。

## 本地状态

- 新输出目录：`{output}`；原 C3.6 输出未改写。
- 来源图件与数组：`{source}`。
- 输出含 13 组 PDF、SVG、300 dpi PNG，其中 9 张 CMOS 空间图按 400×400 窗口重绘，4 张统计图从来源 run 原样复制。
- 各候选准确率：P1 {stats['p1']['accuracy']:.2%}，P2 {stats['p2']['accuracy']:.2%}；均来自来源 run。
"""
    (reports_dir / "visualization_report.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({"status": "RENDERED", "output": str(output), "spatial_figures": len(SPATIAL_STEMS),
                      "figure_count": len(FIGURE_STEMS), "forward": "NONE", "updates": "NONE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
