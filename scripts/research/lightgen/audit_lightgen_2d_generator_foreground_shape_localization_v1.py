"""Localize saved LightGen shape error to target foreground and background regions."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_saved_evidence_v1 as saved_evidence,
)
from scripts.research.lightgen import (
    audit_lightgen_2d_generator_shape_energy_counterfactual_v1 as source_audit,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as energy_task,
)

TASK_ID = "M0_GENERATOR_FOREGROUND_SHAPE_LOCALIZATION_V1"
SOURCE_ATTEMPT = Path(
    "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03"
)
OUTPUT_VERSION = "20261009_m0_generator_foreground_shape_localization_v1"
DEFAULT_OUTPUT = Path(f"outputs/{OUTPUT_VERSION}/runs/attempt_01")
ENDPOINTS = source_audit.ENDPOINTS
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
BATCH_ROWS = 50
FOCUS_CLASSES = (2, 4)
DECOMPOSITION_RTOL = 1.0e-12
DECOMPOSITION_ATOL = 1.0e-14
DECISION_ATOL = 2.0e-7
DECISION_RTOL = 2.0e-5
COMPONENT_METRICS = (
    "q_raw200",
    "q_fg",
    "q_bg",
    "foreground_area_fraction",
    "raw_mse",
    "raw_mse_fg",
    "raw_mse_bg",
    "raw_mse_fg_contribution",
    "raw_mse_bg_contribution",
    "normalized_fg_underfill_mass",
    "normalized_fg_overfill_mass",
    "normalized_bg_generated_mass",
)


class ForegroundLocalizationError(RuntimeError):
    """Saved inputs or a measured foreground component violate the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ForegroundLocalizationError(message)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_output_path(root: Path, output_dir: Path) -> None:
    outputs = (root / "outputs").resolve()
    resolved = output_dir.resolve()
    _require(outputs in resolved.parents, "output must be a child of outputs/")
    relative = resolved.relative_to(outputs)
    _require(
        len(relative.parts) == 3
        and relative.parts[:2] == (OUTPUT_VERSION, "runs"),
        "output must be under the named experiment version and runs directory",
    )
    _require(
        relative.parts[2].startswith("attempt_")
        and relative.parts[2][8:].isdigit(),
        "output attempt must have an explicit numeric suffix",
    )
    _require(
        not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in relative.parts),
        "protected, restored, or temporary output namespace is forbidden",
    )
    _require(not resolved.exists(), f"refusing to overwrite existing output directory: {resolved}")


def _validate_population(population: Mapping[str, Any]) -> None:
    source_rows = population.get("source_row_indices")
    labels = population.get("labels")
    roles = population.get("roles")
    _require(
        isinstance(source_rows, list)
        and isinstance(labels, list)
        and isinstance(roles, list)
        and len(source_rows) == len(labels) == len(roles) == TOTAL_ROWS,
        "source population must contain exactly 500 row identities, labels, and roles",
    )
    _require(len(set(int(value) for value in source_rows)) == TOTAL_ROWS, "source population contains duplicate row identities")
    _require(all(0 <= int(label) <= 9 for label in labels), "source population contains an invalid class label")
    _require(
        population.get("role_positions") == {"fit": [0, FIT_ROWS], "confirmation": [FIT_ROWS, TOTAL_ROWS]},
        "source population fit and confirmation roles differ from the fixed split",
    )
    _require(roles[:FIT_ROWS] == ["fit"] * FIT_ROWS, "first 200 population rows are not all fit-role rows")
    _require(roles[FIT_ROWS:] == ["confirmation"] * CONFIRM_ROWS, "last 300 population rows are not all confirmation-role rows")
    for role, start, stop, expected in (("fit", 0, FIT_ROWS, 20), ("confirmation", FIT_ROWS, TOTAL_ROWS, 30)):
        for label in range(10):
            count = sum(int(labels[index]) == label for index in range(start, stop))
            _require(count == expected, f"{role} role does not contain exactly {expected} rows for class {label}")


def _verify_source_row_identity(
    readout: Mapping[str, Any], population: Mapping[str, Any], endpoint: str
) -> None:
    expected_rows = torch.as_tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.as_tensor(population["labels"], dtype=torch.long)
    _require(readout["source_indices"].dtype == torch.long, f"{endpoint} source row IDs are not integers")
    _require(readout["labels"].dtype == torch.long, f"{endpoint} labels are not integers")
    _require(torch.equal(readout["source_indices"], expected_rows), f"{endpoint} source row IDs differ from population")
    _require(torch.equal(readout["labels"], expected_labels), f"{endpoint} labels differ from population")


def _verify_production_view(readout: Mapping[str, Any], endpoint: str) -> None:
    raw = readout["raw_cmos"]
    saved = readout["evaluator_view"]
    replayed = energy_task.diagnostic.canonical.cmos_view(raw)
    _require(
        replayed.shape == saved.shape
        and bool(torch.allclose(replayed.to(torch.float64), saved.to(torch.float64), rtol=saved_evidence.FLOAT_RTOL, atol=saved_evidence.FLOAT_ATOL)),
        f"{endpoint} saved production view differs from replay",
    )


def _measure_components(generated: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(generated.shape == target.shape, "generated images and targets have different shapes")
    _require(
        generated.ndim == 4 and tuple(generated.shape[1:]) == (1, 200, 200),
        "saved intensity must be N×1×200×200",
    )
    image = generated.to(torch.float64)
    truth = target.to(torch.float64)
    _require(bool(torch.isfinite(image).all() and torch.isfinite(truth).all()), "saved intensity contains nonfinite values")
    _require(bool((image >= 0.0).all() and (truth >= 0.0).all()), "saved intensity contains negative values")

    image_flat = image.flatten(1)
    truth_flat = truth.flatten(1)
    image_mass = image_flat.sum(1)
    target_mass = truth_flat.sum(1)
    _require(bool((image_mass > 0.0).all()), "generated image has zero intensity mass")
    _require(bool((target_mass > 0.0).all()), "target image has zero intensity mass")
    p_image = image_flat / image_mass[:, None]
    p_target = truth_flat / target_mass[:, None]
    mask = truth_flat > 0.1
    foreground_count = mask.sum(1)
    background_count = (~mask).sum(1)
    _require(bool((foreground_count > 0).all() and (background_count > 0).all()), "target foreground or background is empty")
    denominator = p_target.square().sum(1)
    _require(bool((denominator > 0.0).all()), "normalized target has a zero shape-error denominator")

    normalized_error = (p_image - p_target).square()
    q_fg = (normalized_error * mask).sum(1) / denominator
    q_bg = (normalized_error * (~mask)).sum(1) / denominator
    q_raw = normalized_error.sum(1) / denominator
    _require(
        bool(torch.allclose(q_fg + q_bg, q_raw, rtol=DECOMPOSITION_RTOL, atol=DECOMPOSITION_ATOL)),
        "Q_raw200 does not equal foreground plus background error",
    )

    raw_error = (image_flat - truth_flat).square()
    fg_mse = (raw_error * mask).sum(1) / foreground_count
    bg_mse = (raw_error * (~mask)).sum(1) / background_count
    fraction = foreground_count.to(torch.float64) / image_flat.shape[1]
    raw_mse = raw_error.mean(1)
    fg_contribution = fraction * fg_mse
    bg_contribution = (1.0 - fraction) * bg_mse
    _require(
        bool(torch.allclose(fg_contribution + bg_contribution, raw_mse, rtol=DECOMPOSITION_RTOL, atol=DECOMPOSITION_ATOL)),
        "raw MSE does not equal area-weighted foreground plus background contribution",
    )

    return {
        "q_raw200": q_raw,
        "q_fg": q_fg,
        "q_bg": q_bg,
        "foreground_area_fraction": fraction,
        "raw_mse": raw_mse,
        "raw_mse_fg": fg_mse,
        "raw_mse_bg": bg_mse,
        "raw_mse_fg_contribution": fg_contribution,
        "raw_mse_bg_contribution": bg_contribution,
        "normalized_fg_underfill_mass": ((p_target - p_image).clamp_min(0.0) * mask).sum(1),
        "normalized_fg_overfill_mass": ((p_image - p_target).clamp_min(0.0) * mask).sum(1),
        "normalized_bg_generated_mass": (p_image * (~mask)).sum(1),
    }


def _mean(values: list[float]) -> float:
    _require(bool(values) and all(math.isfinite(value) for value in values), "cannot summarize empty or nonfinite values")
    return math.fsum(values) / len(values)


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2.0,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _classify_primary(difference: float, mean_b: float, mean_a: float) -> dict[str, Any]:
    _require(all(math.isfinite(value) for value in (difference, mean_b, mean_a)), "primary decision inputs are nonfinite")
    tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(mean_b), abs(mean_a))
    if difference < -tolerance:
        classification = "FOREGROUND_SHAPE_ERROR_REDUCED"
        interpretation = "类别2、4目标前景的未截断归一化误差下降；保留前景形态改善的局部证据。"
    elif difference > tolerance:
        classification = "FOREGROUND_SHAPE_ERROR_INCREASED"
        interpretation = "类别2、4目标前景的未截断归一化误差增加；削弱整体形态收益代表目标前景改善的解释。"
    else:
        classification = "FOREGROUND_SHAPE_CHANGE_WITHIN_TOLERANCE"
        interpretation = "未观察到超出预设数值容差的前景收益；仍不能确定原因。"
    return {
        "classification": classification,
        "difference_q_fg_B6700_minus_A6700": difference,
        "tolerance": tolerance,
        "equal_class_mean_q_fg_A6700": mean_a,
        "equal_class_mean_q_fg_B6700": mean_b,
        "interpretation": interpretation,
    }


def _primary_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["role"] == "confirmation" and int(row["label"]) in FOCUS_CLASSES
    ]
    class_results: dict[str, Any] = {}
    class_differences: list[float] = []
    means_a: list[float] = []
    means_b: list[float] = []
    for label in FOCUS_CLASSES:
        group = [row for row in selected if int(row["label"]) == label]
        _require(len(group) == 30, f"primary class {label} must contain 30 confirmation rows")
        mean_a = _mean([float(row["A6700_q_fg"]) for row in group])
        mean_b = _mean([float(row["B6700_q_fg"]) for row in group])
        delta = mean_b - mean_a
        class_results[str(label)] = {
            "rows": len(group),
            "mean_q_fg_A6700": mean_a,
            "mean_q_fg_B6700": mean_b,
            "difference_B6700_minus_A6700": delta,
        }
        class_differences.append(delta)
        means_a.append(mean_a)
        means_b.append(mean_b)
    difference = _mean(class_differences)
    mean_a, mean_b = _mean(means_a), _mean(means_b)
    result = _classify_primary(difference, mean_b, mean_a)
    return {**result, "focus_classes": list(FOCUS_CLASSES), "primary_rows": len(selected), "by_class": class_results}


def _build_paired_rows(
    population: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, Any]],
    source_metrics: Mapping[str, Mapping[str, torch.Tensor]],
    component_metrics: Mapping[str, Mapping[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(TOTAL_ROWS):
        row: dict[str, Any] = {
            "position": index,
            "source_row_index": int(population["source_row_indices"][index]),
            "label": int(population["labels"][index]),
            "role": str(population["roles"][index]),
        }
        for endpoint in ENDPOINTS:
            components = component_metrics[endpoint]
            for metric in COMPONENT_METRICS:
                row[f"{endpoint}_{metric}"] = float(components[metric][index])
            source = source_metrics[endpoint]
            row[f"{endpoint}_q28"] = float(source["brightness_normalized_stroke_error_Q"][index])
            row[f"{endpoint}_ssim28"] = float(source["original_target_ssim_28x28"][index])
            row[f"{endpoint}_integrated_intensity"] = float(source["cmos_integrated_intensity_joule_proxy"][index])
            row[f"{endpoint}_clip_fraction"] = float(source["raw_clip_fraction_outside_0_1"][index])
            row[f"{endpoint}_e_d_prediction"] = int(source["e_d_predictions"][index])
            row[f"{endpoint}_e_d_correct"] = int(int(source["e_d_predictions"][index]) == int(population["labels"][index]))
        rows.append(row)
    return rows


def _class_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metrics = (*COMPONENT_METRICS, "q28", "ssim28", "integrated_intensity", "clip_fraction")
    for role in ("fit", "confirmation"):
        for label in range(10):
            group = [row for row in rows if row["role"] == role and int(row["label"]) == label]
            for endpoint in ENDPOINTS:
                summary: dict[str, Any] = {"role": role, "label": label, "endpoint": endpoint, "rows": len(group)}
                for metric in metrics:
                    summary[f"mean_{metric}"] = _mean([float(row[f"{endpoint}_{metric}"]) for row in group])
                correct = sum(int(row[f"{endpoint}_e_d_correct"]) for row in group)
                summary["e_d_correct"] = correct
                summary["e_d_accuracy"] = correct / len(group)
                result.append(summary)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), "refusing to write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recheck_csv(
    path: Path,
    population: Mapping[str, Any],
    expected_primary: Mapping[str, Any],
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    _require(len(records) == TOTAL_ROWS, "paired CSV does not contain all 500 rows")
    _require([int(row["source_row_index"]) for row in records] == [int(value) for value in population["source_row_indices"]], "paired CSV source row order differs")
    _require([int(row["label"]) for row in records] == [int(value) for value in population["labels"]], "paired CSV labels differ")
    _require([row["role"] for row in records] == list(population["roles"]), "paired CSV roles differ")

    maximum_q_error = 0.0
    maximum_mse_error = 0.0
    for row in records:
        for endpoint in ENDPOINTS:
            qfg = float(row[f"{endpoint}_q_fg"])
            qbg = float(row[f"{endpoint}_q_bg"])
            qraw = float(row[f"{endpoint}_q_raw200"])
            q_error = abs(qfg + qbg - qraw)
            maximum_q_error = max(maximum_q_error, q_error)
            _require(math.isclose(qfg + qbg, qraw, rel_tol=DECOMPOSITION_RTOL, abs_tol=DECOMPOSITION_ATOL), "CSV Q component identity does not close")
            fraction = float(row[f"{endpoint}_foreground_area_fraction"])
            fg = float(row[f"{endpoint}_raw_mse_fg"])
            bg = float(row[f"{endpoint}_raw_mse_bg"])
            raw_mse = float(row[f"{endpoint}_raw_mse"])
            weighted = fraction * fg + (1.0 - fraction) * bg
            mse_error = abs(weighted - raw_mse)
            maximum_mse_error = max(maximum_mse_error, mse_error)
            _require(math.isclose(weighted, raw_mse, rel_tol=DECOMPOSITION_RTOL, abs_tol=DECOMPOSITION_ATOL), "CSV area-weighted MSE identity does not close")

    class_means: dict[int, tuple[float, float]] = {}
    for label in FOCUS_CLASSES:
        group = [row for row in records if row["role"] == "confirmation" and int(row["label"]) == label]
        _require(len(group) == 30, f"CSV primary class {label} does not contain 30 rows")
        a_mean = math.fsum(float(row["A6700_q_fg"]) for row in group) / len(group)
        b_mean = math.fsum(float(row["B6700_q_fg"]) for row in group) / len(group)
        class_means[label] = (a_mean, b_mean)
    mean_a = math.fsum(value[0] for value in class_means.values()) / len(FOCUS_CLASSES)
    mean_b = math.fsum(value[1] for value in class_means.values()) / len(FOCUS_CLASSES)
    difference = math.fsum(class_means[label][1] - class_means[label][0] for label in FOCUS_CLASSES) / len(FOCUS_CLASSES)
    tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(mean_a), abs(mean_b))
    if difference < -tolerance:
        classification = "FOREGROUND_SHAPE_ERROR_REDUCED"
    elif difference > tolerance:
        classification = "FOREGROUND_SHAPE_ERROR_INCREASED"
    else:
        classification = "FOREGROUND_SHAPE_CHANGE_WITHIN_TOLERANCE"
    _require(classification == expected_primary["classification"], "CSV-recomputed primary classification differs")
    _require(math.isclose(difference, float(expected_primary["difference_q_fg_B6700_minus_A6700"]), rel_tol=0.0, abs_tol=1.0e-15), "CSV-recomputed primary effect differs")
    _require(math.isclose(tolerance, float(expected_primary["tolerance"]), rel_tol=0.0, abs_tol=1.0e-15), "CSV-recomputed primary tolerance differs")
    return {
        "passed": True,
        "rows": len(records),
        "primary_rows": 60,
        "primary_classification_recomputed": classification,
        "primary_effect_recomputed": difference,
        "primary_tolerance_recomputed": tolerance,
        "maximum_q_decomposition_absolute_error": maximum_q_error,
        "maximum_mse_decomposition_absolute_error": maximum_mse_error,
    }


def _png_data_uri(image: torch.Tensor) -> str:
    values = image.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    _require(len(values) == 28 * 28 and all(math.isfinite(value) for value in values), "gallery view must be finite 28×28")
    pixels = bytes(
        max(0, min(255, int(math.floor(max(0.0, min(1.0, value)) * 255.0 + 0.5))))
        for value in values
    )
    scanlines = b"".join(b"\x00" + pixels[row * 28 : (row + 1) * 28] for row in range(28))

    def chunk(name: bytes, payload: bytes) -> bytes:
        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 28, 28, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _render_focus_gallery(
    population: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]
) -> str:
    positions = [
        index for index in range(FIT_ROWS, TOTAL_ROWS)
        if int(population["labels"][index]) in FOCUS_CLASSES
    ]
    positions.sort(key=lambda index: int(population["source_row_indices"][index]))
    _require(len(positions) == 60, "focus gallery must contain all 60 confirmation rows for classes 2 and 4")
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><meta charset="utf-8"><title>类别2、4前景形态定位图库</title>',
        "<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:8px;text-align:center}img{width:224px;height:224px;image-rendering:pixelated;image-rendering:crisp-edges;background:#777}.note{max-width:1100px}</style>",
        "<body><h1>类别2、4确认人口定位图库</h1>",
        '<p class="note">共60行，按保存源行号排序；固定使用保存28×28生产视图、灰度[0,1]和8倍最近邻显示。该页用于揭盲后的定位复盘，不是新的评分任务，图片未按结果挑选。</p>',
        "<table><thead><tr><th>源行号</th><th>请求类别</th><th>目标</th><th>A4700</th><th>A6700</th><th>B6700</th></tr></thead><tbody>",
    ]
    target_readout = readouts["A4700"]
    target_views = energy_task.diagnostic.canonical.cmos_view(target_readout["target_slm_intensity"])
    for index in positions:
        source_row = int(population["source_row_indices"][index])
        label = int(population["labels"][index])
        images = [("目标", target_views[index, 0])]
        for endpoint in ENDPOINTS:
            images.append((endpoint, readouts[endpoint]["evaluator_view"][index, 0]))
        cells = "".join(
            f'<td><img alt="{html.escape(name)} source row {source_row}" src="{_png_data_uri(image)}"></td>'
            for name, image in images
        )
        lines.append(f"<tr><td>{source_row}</td><td>{label}</td>{cells}</tr>")
    lines.extend(("</tbody></table></body></html>", ""))
    return "\n".join(lines)


def _render_plan() -> str:
    return """# 第一阶段：保存读出的前景形态定位（执行记录）

唯一问题：B6700相对A6700的归一化形态收益，是否涉及类别2、4的目标前景区域？固定读取能量目标对照attempt_03三端点的500行保存读出；确认人口300行作主分析，拟合人口200行单独描述。数据均属于canonical TRAIN。

对原始200×200目标使用固定掩码`T>0.1`，生成图和目标各自归一化后计算Q_fg、Q_bg与Q_raw200。逐行同时核对`Q_raw200=Q_fg+Q_bg`及`MSE=f·MSE_fg+(1-f)·MSE_bg`。主效应为类别2、4各自确认人口均值的B6700−A6700差再等权平均，容差为`2e-7+2e-5·max(|mean Q_fg B|,|mean Q_fg A|)`。负值超出容差表示目标前景误差降低；正值表示增加；容差内未观察到超容差变化。输入核验失败时不形成科学判定。

辅助结果报告十类、两种角色、背景与总误差、前景/背景 raw MSE贡献、归一化前景强度不足/过量与背景强度占比、原生产28×28 Q、SSIM、积分强度、截断比例和保存E_D读数。所有结论只定位保存强度分布，不直接测量笔画拓扑或人眼可读性，不证明梯度因果或确定损失权重。

CPU预检先重放全部保存来源与生产视图，再从确认人口每类按源行号取第一行组成10行代表样本；全部通过后按50行批次复算500行。重读逐行CSV，独立重算主效应、容差和终局。输出类别2、4的全部60行固定视图图库。停止于定位报告；不改目标、不训练、不运行梯度或新前向，`training_recommendation=NONE`、`next_action=NONE`。
"""


def _render_report(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    focus = decision["by_class"]
    lines = [
        f"# {TASK_ID}",
        "",
        "## 结果与状态",
        f"保存读出的CPU定位测量有效完成。主效应`D_fg={decision['difference_q_fg_B6700_minus_A6700']:.10g}`，容差`τ={decision['tolerance']:.10g}`，终局`{decision['classification']}`。",
        "",
        "## 科学认识",
        decision["interpretation"],
        f"类别2：Q_fg A6700={focus['2']['mean_q_fg_A6700']:.8g}，B6700={focus['2']['mean_q_fg_B6700']:.8g}，差={focus['2']['difference_B6700_minus_A6700']:.8g}；类别4：A6700={focus['4']['mean_q_fg_A6700']:.8g}，B6700={focus['4']['mean_q_fg_B6700']:.8g}，差={focus['4']['difference_B6700_minus_A6700']:.8g}。两类各30行、等权进入主量。",
        "背景、总Q、raw MSE面积加权贡献、归一化前景不足/过量、背景质量、28×28 Q、SSIM、强度、截断和E_D均作为辅助结果，见`report.json`、`class_summary.csv`与`paired_rows.csv`。前景误差是固定目标掩码内的强度分布误差，不等同于笔画拓扑或可读性。",
        "",
        "## 目标缺口",
        "该保存证据定位不能证明前景梯度被背景压制，不能确定最佳损失权重，也不证明人眼可读性、分类代理效度、联合先验可读性或独立泛化。全部500行属于canonical TRAIN；本结果不改变实验一的审阅质控失败或实验二、三既有终局。",
        "",
        "## 工程完成与验证",
        f"复现父attempt_03三端点、人口角色和原生产视图；测量500行（拟合200、确认300），通过10行代表性CPU预检、两项逐行分解恒等式及CSV回读独立重算。CSV终局为`{result['validation']['csv_roundtrip']['primary_classification_recomputed']}`。计算设备CPU；模型更新、光学/评价器前向、梯度、采样、人审均为0。类别2、4确认人口60行揭盲定位图库按源行号输出。",
        "",
        "## 决策影响与下一步",
        "本结果只决定是否保留“类别2、4前景误差下降”的局部证据。若前景误差下降，后续仍需拆解残留缺陷与背景代价；若没有下降，可将前景/背景目标作为待检验假设；若下降但人眼识读仍差，应先验证指标与具体形态缺陷的对应关系。后续建议依次为候选目标梯度核验、单因素训练对照、固定先验下采样及关键人审、独立人口确认；这些不在本轮授权内。停止，不训练或采用端点；`training_recommendation=NONE`、`next_action=NONE`。",
        "",
    ]
    return "\n".join(lines)


def _run(artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    _validate_output_path(root, output_dir)
    _require(artifact_dir.resolve() == (root / SOURCE_ATTEMPT).resolve(), "artifact directory differs from the explicitly named attempt_03")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    validation: dict[str, Any] = {
        "schema": "m0_generator_foreground_shape_localization_validation.v1",
        "task_id": TASK_ID,
        "artifact_directory": str(artifact_dir),
        "output_directory": str(output_dir),
        "execution": {"device": "cpu", "model_updates": 0, "optical_forwards": 0, "evaluator_forwards": 0, "gradients": 0, "sampling": 0, "human_ratings": 0},
        "checks": {},
    }
    write_json = saved_evidence._write_json
    write_json(output_dir / "preflight.json", {**validation, "status": "RUNNING"})
    try:
        population, parent_report, readouts, _source_summaries, source_metrics, source_checks = source_audit._load_and_verify(artifact_dir)
        _validate_population(population)
        for endpoint, readout in readouts.items():
            _verify_source_row_identity(readout, population, endpoint)
            _verify_production_view(readout, endpoint)
        checks = {"parent_saved_evidence_replay": source_checks, "population_roles_and_class_counts": True}

        representative_positions: list[int] = []
        for label in range(10):
            candidates = [index for index in range(FIT_ROWS, TOTAL_ROWS) if int(population["labels"][index]) == label]
            candidates.sort(key=lambda index: int(population["source_row_indices"][index]))
            representative_positions.append(candidates[0])
        representative = torch.tensor(representative_positions, dtype=torch.long)
        target_raw = readouts["A4700"]["target_slm_intensity"]
        representative_components: dict[str, dict[str, torch.Tensor]] = {}
        for endpoint in ENDPOINTS:
            representative_components[endpoint] = _measure_components(
                readouts[endpoint]["raw_cmos"].index_select(0, representative),
                target_raw.index_select(0, representative),
            )
            for values in representative_components[endpoint].values():
                _require(len(values) == 10, f"{endpoint} 10-row preflight is incomplete")
        checks["representative_ten_row_component_preflight"] = True
        checks["representative_q_and_mse_decompositions"] = True
        write_json(
            output_dir / "preflight.json",
            {**validation, "checks": checks, "status": "READY", "representative_positions": representative_positions, "representative_source_row_indices": [int(population["source_row_indices"][i]) for i in representative_positions]},
        )

        component_rows: dict[str, dict[str, list[float]]] = {endpoint: {metric: [] for metric in COMPONENT_METRICS} for endpoint in ENDPOINTS}
        for start in range(0, TOTAL_ROWS, BATCH_ROWS):
            stop = min(start + BATCH_ROWS, TOTAL_ROWS)
            for endpoint in ENDPOINTS:
                measured = _measure_components(
                    readouts[endpoint]["raw_cmos"][start:stop],
                    target_raw[start:stop],
                )
                for metric, values in measured.items():
                    component_rows[endpoint][metric].extend(float(value) for value in values.tolist())
        _require(
            all(len(values) == TOTAL_ROWS for endpoint in component_rows.values() for values in endpoint.values()),
            "component computation did not include all 500 rows",
        )
        for endpoint in ENDPOINTS:
            for metric, expected in representative_components[endpoint].items():
                for local_index, position in enumerate(representative_positions):
                    actual = component_rows[endpoint][metric][position]
                    _require(math.isclose(actual, float(expected[local_index]), rel_tol=DECOMPOSITION_RTOL, abs_tol=DECOMPOSITION_ATOL), f"{endpoint} batch result differs from representative replay for {metric}")
        checks["all_500_rows_measured_in_fixed_batches"] = True
        checks["batch_consistency"] = True

        paired = _build_paired_rows(population, readouts, source_metrics, component_rows)
        paired_path = output_dir / "paired_rows.csv"
        _write_csv(paired_path, paired)
        primary = _primary_decision(paired)
        class_rows = _class_summary(paired)
        class_path = output_dir / "class_summary.csv"
        _write_csv(class_path, class_rows)
        csv_check = _recheck_csv(paired_path, population, primary)
        primary = {**primary, "csv_recheck": csv_check}

        population_summary: dict[str, Any] = {}
        for role in ("fit", "confirmation"):
            role_rows = [row for row in paired if row["role"] == role]
            population_summary[role] = {"rows": len(role_rows), "by_class": {}}
            for label in range(10):
                group = [row for row in role_rows if int(row["label"]) == label]
                endpoint_summary: dict[str, Any] = {}
                for endpoint in ENDPOINTS:
                    endpoint_summary[endpoint] = {
                        metric: _stats([float(row[f"{endpoint}_{metric}"]) for row in group])
                        for metric in (*COMPONENT_METRICS, "q28", "ssim28", "integrated_intensity", "clip_fraction")
                    }
                    correct = sum(int(row[f"{endpoint}_e_d_correct"]) for row in group)
                    endpoint_summary[endpoint]["e_d_accuracy"] = {"correct": correct, "rows": len(group), "accuracy": correct / len(group)}
                population_summary[role]["by_class"][str(label)] = endpoint_summary
            population_summary[role]["endpoints"] = {}
            for endpoint in ENDPOINTS:
                endpoint_rows = [row for row in role_rows]
                correct = sum(int(row[f"{endpoint}_e_d_correct"]) for row in endpoint_rows)
                population_summary[role]["endpoints"][endpoint] = {
                    metric: _stats([float(row[f"{endpoint}_{metric}"]) for row in endpoint_rows])
                    for metric in (*COMPONENT_METRICS, "q28", "ssim28", "integrated_intensity", "clip_fraction")
                }
                population_summary[role]["endpoints"][endpoint]["e_d_accuracy"] = {"correct": correct, "rows": len(endpoint_rows), "accuracy": correct / len(endpoint_rows)}

        class2_4_rows = [row for row in paired if row["role"] == "confirmation" and int(row["label"]) in FOCUS_CLASSES]
        changes: dict[str, Any] = {}
        for endpoint in ("A6700", "B6700"):
            changes[endpoint] = {
                "q_raw200_difference_from_A4700_mean": _mean([float(row[f"{endpoint}_q_raw200"]) - float(row["A4700_q_raw200"]) for row in paired[FIT_ROWS:]]),
                "q28_difference_from_A4700_mean": _mean([float(row[f"{endpoint}_q28"]) - float(row["A4700_q28"]) for row in paired[FIT_ROWS:]]),
                "integrated_intensity_ratio_to_A4700_mean": _mean([float(row[f"{endpoint}_integrated_intensity"]) / float(row["A4700_integrated_intensity"]) for row in paired[FIT_ROWS:]]),
            }

        result: dict[str, Any] = {
            "schema": "m0_generator_foreground_shape_localization_report.v1",
            "task_id": TASK_ID,
            "status": "EVIDENCE_REVIEW_COMPLETED",
            "source_task_id": source_audit.SOURCE_TASK_ID,
            "source_parent_classification": parent_report["classification"],
            "decision": primary,
            "population": {"total_rows": TOTAL_ROWS, "fit_rows": FIT_ROWS, "confirmation_rows": CONFIRM_ROWS, "per_class_fit": 20, "per_class_confirmation": 30, "data_role": "canonical TRAIN; no independent generalization claim", "summary": population_summary},
            "measurement": {
                "raw_intensity_shape": "N×1×200×200",
                "foreground_mask": "saved target raw intensity > 0.1",
                "normalization": "each generated image and target independently divided by its raw 200×200 sum",
                "q_component_definition": "sum_region((p_image-p_target)^2)/sum(p_target^2), with foreground and background masks partitioning all pixels",
                "raw_mse_decomposition": "MSE = foreground_area_fraction*MSE_fg + (1-foreground_area_fraction)*MSE_bg",
                "main_population": "confirmation classes 2 and 4; each class mean weighted equally",
                "decision_atol": DECISION_ATOL,
                "decision_rtol": DECISION_RTOL,
                "displayed_views": "saved production 28×28 view; fixed [0,1] grayscale; 8× nearest-neighbor gallery",
            },
            "auxiliary_endpoint_changes_from_A4700": changes,
            "auxiliary_outputs": {"paired_rows_csv": paired_path.name, "class_summary_csv": class_path.name, "focus_gallery": "focus_gallery.html", "focus_gallery_rows": len(class2_4_rows)},
            "validation": {"checks": checks, "csv_roundtrip": csv_check},
            "training_recommendation": "NONE",
            "next_action": "NONE",
            "stop": True,
            "scientific_scope": "Only saved A4700/A6700/B6700 endpoints and named 500-row canonical TRAIN population; foreground components are intensity-distribution diagnostics, not readability or topology measures.",
        }
        _write_text(output_dir / "experiment_plan.md", _render_plan())
        _write_text(output_dir / "focus_gallery.html", _render_focus_gallery(population, readouts))
        write_json(output_dir / "validation.json", {**validation, "status": "VALIDATED", "checks": checks, "source_rows": TOTAL_ROWS, "measured_rows": TOTAL_ROWS, "csv_roundtrip": csv_check})
        write_json(output_dir / "report.json", result)
        _write_text(output_dir / "report.md", _render_report(result))
        write_json(output_dir / "preflight.json", {**validation, "status": "READY", "checks": checks, "representative_positions": representative_positions})
        write_json(output_dir / "completed.json", {"task_id": TASK_ID, "status": result["status"], "classification": primary["classification"], "training_recommendation": "NONE", "next_action": "NONE", "stop": True})
        return result
    except (ForegroundLocalizationError, saved_evidence.SavedEvidenceAuditError, source_audit.CounterfactualError, energy_task.EnergyTargetError, OSError) as error:
        validation["status"] = "INVALID_EVIDENCE"
        validation["stop_reason"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "validation.json", validation)
        write_json(output_dir / "preflight.json", {**validation, "status": "INVALID_EVIDENCE"})
        result = {"task_id": TASK_ID, "status": "INVALID_EVIDENCE", "decision": None, "stop_reason": validation["stop_reason"], "training_recommendation": "NONE", "next_action": "NONE", "stop": True}
        write_json(output_dir / "report.json", result)
        write_json(output_dir / "completed.json", result)
        _write_text(output_dir / "experiment_plan.md", _render_plan())
        _write_text(
            output_dir / "report.md",
            "\n".join((
                f"# {TASK_ID}", "", "## 结果与状态", "测量无效；没有科学判定。", "",
                "## 科学认识", "没有形成有效前景形态测量，不能据此支持或削弱训练解释。", "",
                "## 目标缺口", f"输入或测量核验停止：{validation['stop_reason']}", "",
                "## 工程完成与验证", "未形成可用于科学判断的完整500行配对结果。", "",
                "## 决策影响与下一步", "保持现有端点和训练判断；修复证据或测量问题后重新核验。未授权训练或后继实验；training_recommendation=NONE，next_action=NONE。", "",
            )),
        )
        return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Explicit energy target comparison attempt_03")
    parser.add_argument("--output-dir", required=True, type=Path, help="New versioned run directory")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    artifact_dir = saved_evidence._resolve(root, args.artifact_dir)
    output_dir = saved_evidence._resolve(root, args.output_dir)
    result = _run(artifact_dir, output_dir)
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "output_directory": str(output_dir)}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "EVIDENCE_REVIEW_COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
