"""Measure saved LightGen readouts under per-image numerical energy counterfactuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_saved_evidence_v1 as saved_evidence,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as energy_task,
)

TASK_ID = "M0_GENERATOR_SHAPE_ENERGY_COUNTERFACTUAL_V1"
SOURCE_TASK_ID = energy_task.TASK_ID
SOURCE_ATTEMPT = Path(
    "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03"
)
DEFAULT_OUTPUT = Path(
    "outputs/20261009_m0_generator_shape_energy_counterfactual_v1/runs/attempt_01"
)
ENDPOINTS = saved_evidence.ENDPOINTS
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
BATCH_ROWS = 50
ENERGY_RTOL = 1.0e-12
REPLAY_RTOL = saved_evidence.FLOAT_RTOL
REPLAY_ATOL = saved_evidence.FLOAT_ATOL
FIXED_ATOL = 2.0e-7
FIXED_RTOL = 2.0e-5
CELL_ORDER = ("AA", "AB", "BA", "BB")


class CounterfactualError(RuntimeError):
    """Saved inputs or the numerical counterfactual violate the measurement contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CounterfactualError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required evidence file is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CounterfactualError(f"evidence JSON is malformed: {path.name}: {error}") from error
    _require(isinstance(value, dict), f"evidence JSON must contain an object: {path.name}")
    return value


def _load_readout(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CounterfactualError(f"saved readout cannot be loaded: {path.name}: {error}") from error
    _require(isinstance(value, Mapping), f"saved readout is not a mapping: {path.name}")
    return value


def _integrated_energy(raw: torch.Tensor, pixel_area_m2: float) -> torch.Tensor:
    _require(math.isfinite(pixel_area_m2) and pixel_area_m2 > 0.0, "camera pixel area is invalid")
    _require(raw.ndim == 4 and raw.shape[1] == 1, "raw CMOS image must be NCHW single-channel")
    values = raw.to(torch.float64)
    _require(bool(torch.isfinite(values).all()), "raw CMOS image contains nonfinite values")
    _require(bool((values >= 0.0).all()), "raw CMOS image contains negative intensity")
    energy = values.flatten(1).sum(1) * pixel_area_m2
    _require(bool(torch.isfinite(energy).all() and (energy > 0.0).all()), "raw CMOS image has zero or invalid integrated intensity")
    return energy


def _four_images(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    pixel_area_m2: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _require(raw_a.shape == raw_b.shape, "A6700 and B6700 image shapes differ")
    _require(raw_a.ndim == 4 and tuple(raw_a.shape[1:]) == (1, 200, 200), "raw CMOS images must be N×1×200×200")
    a = raw_a.to(torch.float64)
    b = raw_b.to(torch.float64)
    energy_a = _integrated_energy(a, pixel_area_m2)
    energy_b = _integrated_energy(b, pixel_area_m2)
    images = {
        "AA": a,
        "AB": a * (energy_b / energy_a).view(-1, 1, 1, 1),
        "BA": b * (energy_a / energy_b).view(-1, 1, 1, 1),
        "BB": b,
    }
    expected = {"AA": energy_a, "AB": energy_b, "BA": energy_a, "BB": energy_b}
    measured: dict[str, torch.Tensor] = {}
    for cell, image in images.items():
        _require(bool(torch.isfinite(image).all() and (image >= 0.0).all()), f"counterfactual {cell} contains invalid intensity")
        measured[cell] = _integrated_energy(image, pixel_area_m2)
        relative_error = (measured[cell] - expected[cell]).abs() / expected[cell]
        _require(bool((relative_error <= ENERGY_RTOL).all()), f"counterfactual {cell} does not meet its pre-clipping energy constraint")
    _require(torch.equal(images["AA"], a) and torch.equal(images["BB"], b), "counterfactual diagonal differs from the saved image")
    return images, measured


def _normalized_shape_q(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require(generated.shape == target.shape and generated.ndim == 4, "shape-Q tensors must have matching NCHW dimensions")
    x = generated.to(torch.float64).flatten(1)
    y = target.to(torch.float64).flatten(1)
    sum_x, sum_y = x.sum(1), y.sum(1)
    _require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), "shape-Q input contains nonfinite values")
    _require(bool((sum_x > 0.0).all() and (sum_y > 0.0).all()), "shape-Q is undefined for a zero-mass image")
    px, py = x / sum_x[:, None], y / sum_y[:, None]
    denominator = py.square().sum(1)
    _require(bool((denominator > 0.0).all()), "shape-Q target denominator is zero")
    return (px - py).square().sum(1) / denominator


def _verify_production_view(readout: Mapping[str, Any], endpoint: str) -> None:
    raw = readout["raw_cmos"]
    replayed = energy_task.diagnostic.canonical.cmos_view(raw).to(torch.float64)
    saved = readout["evaluator_view"].to(torch.float64)
    _require(bool(torch.allclose(replayed, saved, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)), f"{endpoint} saved production view does not replay")


def _verify_row_identity(readout: Mapping[str, Any], population: Mapping[str, Any], endpoint: str) -> None:
    expected_rows = torch.as_tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.as_tensor(population["labels"], dtype=torch.long)
    _require(readout["source_indices"].dtype == torch.long, f"{endpoint} source row IDs are not stored as integers")
    _require(readout["labels"].dtype == torch.long, f"{endpoint} labels are not stored as integers")
    _require(torch.equal(readout["source_indices"], expected_rows), f"{endpoint} source row IDs differ from the population")
    _require(torch.equal(readout["labels"], expected_labels), f"{endpoint} labels differ from the population")


def _classify_difference(difference: float, q_ba: float, q_aa: float) -> dict[str, Any]:
    _require(all(math.isfinite(value) for value in (difference, q_ba, q_aa)), "main decision inputs are nonfinite")
    tolerance = FIXED_ATOL + FIXED_RTOL * max(abs(q_ba), abs(q_aa))
    if difference < -tolerance:
        classification = "FIXED_ENERGY_Q_IMPROVEMENT_RETAINED"
        interpretation = "固定A能量后Q改善保留；削弱原改善完全来自输出变暗的解释。"
    elif difference > tolerance:
        classification = "FIXED_ENERGY_Q_IMPROVEMENT_REVERSED"
        interpretation = "固定A能量后Q变化反转；削弱原Q改善代表同能量形态改善的解释。"
    else:
        classification = "FIXED_ENERGY_Q_CHANGE_WITHIN_TOLERANCE"
        interpretation = "未观察到超出预设数值容差的固定能量Q变化；保留未决。"
    return {
        "classification": classification,
        "difference_q_ba_minus_q_aa": difference,
        "tolerance": tolerance,
        "mean_q_ba": q_ba,
        "mean_q_aa": q_aa,
        "interpretation": interpretation,
    }


def _make_protocol() -> str:
    return """# 实验二：固定能量下的形态收益（执行方案）

## 问题与边界

检验 B6700 相对 A6700 的生产 Q 改善在逐图恢复到 A 的积分强度后是否保留。唯一主分析为 300 行固定确认人口；200 行拟合人口单列报告。数据属于 canonical TRAIN 的经验 OLS 解码人口，不能与实验一的条件采样总体合并。所有能量反事实都是保存读出上的数值构造，不是光学输出。

## 计算

读取能量对照 attempt_03 的 A4700、A6700、B6700 三个读出、人口、父报告和 A/B 保存指标。令 E(I)=ΔA·sum(I)，每行构造 I_AA=I_A、I_AB=I_A·E_B/E_A、I_BA=I_B·E_A/E_B、I_BB=I_B。四格统一经过 Pool28(clip(I,0,1))，再以保存的原生产目标视图计算 Q=sum[(p(I)-p(T))²]/sum[p(T)²]，其中 p(x)=x/sum(x)。

唯一主量为确认人口 mean(Q_BA-Q_AA)，容差为 2e-7+2e-5·max(|mean Q_BA|,|mean Q_AA|)。低于负容差表示固定 A 能量后收益保留；高于正容差表示收益反转；容差内表示未观察到超出数值容差的变化。输入或生产核验失败时终止并标记测量无效。

报告四格Q、逐图能量和截断比例、未截断200×200及池化未截断28×28形态Q、原目标重建误差和SSIM、类别及拟合人口描述、实际端点分类器读数。对称形态/能量贡献及交互仅辅助解释；分类器不评价反事实图像。

## 验证与停止

全500行身份、角色、坐标、非负有限强度、正质量、保存视图、Q与父报告先通过核验；代表样本四格预检、能量约束和分解恒等式通过后才计算完整人口。CSV须重读并独立复算主判读。聚焦回归覆盖截断前后纯缩放、等能真实形态变化、三种判读及无效证据拒绝。

范围限于这批冻结端点和 TRAIN 人口，不证明人眼可读性、独立泛化、梯度因果或模型采用；不改变实验一质控结论。完成本报告后停止，不启动训练或后继实验。
"""


def _cell_metrics(
    image: torch.Tensor,
    target_raw: torch.Tensor,
    labels: torch.Tensor,
    plane_metadata: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    view = energy_task.diagnostic.canonical.cmos_view(image)
    target_view = energy_task.diagnostic.canonical.cmos_view(target_raw)
    population = {"role_positions": {"confirmation": [0, int(labels.shape[0])]}}
    summary, rows = energy_task._energy_image_metrics(
        {
            "raw_cmos": image,
            "target_slm_intensity": target_raw,
            "evaluator_view": view,
            "labels": labels,
            "plane_metadata": plane_metadata,
        },
        {"population": population},
    )
    _require(summary["metrics_by_role"]["confirmation"]["Q_undefined_rows"] == 0, "production Q is undefined for a counterfactual row")
    pooled_unclipped = F.adaptive_avg_pool2d(image, (28, 28))
    target_pooled_unclipped = F.adaptive_avg_pool2d(target_raw.to(torch.float64), (28, 28))
    return {
        "q": rows["brightness_normalized_stroke_error_Q"],
        "energy": rows["cmos_integrated_intensity_joule_proxy"],
        "clip_fraction": rows["raw_clip_fraction_outside_0_1"],
        "raw_mse": rows["original_target_raw_mse"],
        "foreground_mse": rows["original_target_raw_foreground_mse"],
        "background_mse": rows["original_target_raw_background_mse"],
        "ssim": rows["original_target_ssim_28x28"],
        "unclipped_q_200": _normalized_shape_q(image, target_raw),
        "unclipped_q_28": _normalized_shape_q(pooled_unclipped, target_pooled_unclipped),
    }


def _measure_batch(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    target_raw: torch.Tensor,
    labels: torch.Tensor,
    plane_metadata: Mapping[str, Any],
    pixel_area_m2: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    images, energies = _four_images(raw_a, raw_b, pixel_area_m2)
    cells = {
        cell: _cell_metrics(image, target_raw, labels, plane_metadata)
        for cell, image in images.items()
    }
    q = {cell: metrics["q"] for cell, metrics in cells.items()}
    d_shape = 0.5 * ((q["BA"] - q["AA"]) + (q["BB"] - q["AB"]))
    d_energy = 0.5 * ((q["AB"] - q["AA"]) + (q["BB"] - q["BA"]))
    interaction = q["BB"] - q["BA"] - q["AB"] + q["AA"]
    total_change = q["BB"] - q["AA"]
    _require(bool(torch.isfinite(d_shape).all() and torch.isfinite(d_energy).all() and torch.isfinite(interaction).all()), "factorial decomposition contains nonfinite values")
    _require(bool(torch.allclose(d_shape + d_energy, total_change, rtol=1e-12, atol=1e-14)), "shape and energy decomposition does not reproduce the diagonal Q change")
    _require(bool(torch.allclose((q["BB"] - q["AB"]) - (q["BA"] - q["AA"]), interaction, rtol=1e-12, atol=1e-14)), "interaction does not reproduce the difference-in-differences")
    return cells, {"d_shape": d_shape, "d_energy": d_energy, "interaction": interaction, "total_change": total_change, **{f"energy_{key.lower()}": value for key, value in energies.items()}}


def _summary_values(values: list[float]) -> dict[str, float | int]:
    _require(bool(values) and all(math.isfinite(value) for value in values), "summary values are empty or nonfinite")
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": math.fsum(values) / len(values),
        "median": (ordered[(len(values) - 1) // 2] + ordered[len(values) // 2]) / 2.0,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _load_and_verify(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    _require(source_dir.is_dir(), f"source attempt directory is missing: {source_dir}")
    population = saved_evidence._load_json(source_dir / "input_population.json")
    source_report = saved_evidence._load_json(source_dir / "report.json")
    checks = saved_evidence._verify_attempt(population, source_report)
    readouts = {endpoint: _load_readout(source_dir / filename) for endpoint, filename in ENDPOINTS.items()}
    checks.update(saved_evidence._verify_readouts(population, readouts))
    for endpoint, readout in readouts.items():
        _verify_row_identity(readout, population, endpoint)

    pixel_area, pixel_pitch = energy_task._pixel_area(readouts["A4700"], "camera_plane")
    checks["camera_pixel_area_m2"] = pixel_area
    checks["camera_pixel_pitch_m"] = pixel_pitch
    summaries: dict[str, dict[str, Any]] = {}
    row_values: dict[str, dict[str, torch.Tensor]] = {}
    for endpoint, readout in readouts.items():
        raw = readout["raw_cmos"]
        target_raw = readout["target_slm_intensity"]
        _require(bool((raw >= 0.0).all() and (target_raw >= 0.0).all()), f"{endpoint} has negative saved intensity")
        _integrated_energy(raw, pixel_area)
        target_energy = target_raw.to(torch.float64).flatten(1).sum(1)
        _require(bool(torch.isfinite(target_energy).all() and (target_energy > 0.0).all()), f"{endpoint} contains a zero-mass or invalid target")
        _verify_production_view(readout, endpoint)
        checks[f"{endpoint}_production_view_replay"] = True
        summary, rows = saved_evidence._derive_readout_metrics(readout, population, endpoint)
        summaries[endpoint], row_values[endpoint] = summary, rows
    checks["parent_report_metrics_and_decision"] = saved_evidence._verify_stored_metrics(
        source_dir, population, source_report, readouts, summaries, row_values
    )
    checks["complete_source_population_valid"] = True
    return population, source_report, readouts, summaries, row_values, checks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(len(rows) == TOTAL_ROWS, "paired output does not contain all 500 rows")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(
    population: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, Any]],
    endpoint_rows: Mapping[str, Mapping[str, torch.Tensor]],
    cell_rows: Mapping[str, Mapping[str, list[float]]],
    decomp_rows: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    roles = population["roles"]
    rows: list[dict[str, Any]] = []
    for index in range(TOTAL_ROWS):
        row: dict[str, Any] = {
            "position": index,
            "source_row_index": int(population["source_row_indices"][index]),
            "label": int(population["labels"][index]),
            "role": str(roles[index]),
            "image_status": "AA_and_BB_saved_endpoints; AB_and_BA_numerical_counterfactuals",
        }
        for endpoint in ENDPOINTS:
            values = endpoint_rows[endpoint]
            for metric, column in (("brightness_normalized_stroke_error_Q", "q"), ("cmos_integrated_intensity_joule_proxy", "energy"), ("raw_clip_fraction_outside_0_1", "clip_fraction"), ("original_target_raw_mse", "raw_mse"), ("original_target_ssim_28x28", "ssim")):
                row[f"actual_{endpoint}_{column}"] = float(values[metric][index])
            row[f"actual_{endpoint}_e_d_prediction"] = int(values["e_d_predictions"][index])
            row[f"actual_{endpoint}_e_d_margin"] = float(values["e_d_margin_float64"][index])
        for cell in CELL_ORDER:
            for metric, values in cell_rows[cell].items():
                row[f"{cell}_{metric}"] = float(values[index])
        row["energy_ratio_b_over_a"] = row["BB_energy"] / row["AA_energy"]
        for metric, values in decomp_rows.items():
            row[metric] = float(values[index])
        rows.append(row)
    return rows


def _recheck_csv(path: Path, population: Mapping[str, Any], expected_decision: Mapping[str, Any]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    _require(len(records) == TOTAL_ROWS, "paired CSV row count differs from the declared population")
    expected_indices = [int(value) for value in population["source_row_indices"]]
    expected_labels = [int(value) for value in population["labels"]]
    expected_roles = list(population["roles"])
    _require([int(row["source_row_index"]) for row in records] == expected_indices, "paired CSV source row order differs from the source population")
    _require([int(row["label"]) for row in records] == expected_labels, "paired CSV labels differ from the source population")
    _require([row["role"] for row in records] == expected_roles, "paired CSV population roles differ from the source population")
    confirm = records[FIT_ROWS:]
    _require(len(confirm) == CONFIRM_ROWS and all(row["role"] == "confirmation" for row in confirm), "primary confirmatory rows are incomplete")
    q_aa = math.fsum(float(row["AA_q"]) for row in confirm) / CONFIRM_ROWS
    q_ba = math.fsum(float(row["BA_q"]) for row in confirm) / CONFIRM_ROWS
    row_differences = [float(row["BA_q"]) - float(row["AA_q"]) for row in confirm]
    _require(all(delta == float(row["d_fixed_a_energy"]) for delta, row in zip(row_differences, confirm, strict=True)), "paired CSV fixed-energy row differences do not match its Q cells")
    difference = math.fsum(row_differences) / CONFIRM_ROWS
    decision = _classify_difference(difference, q_ba, q_aa)
    _require(decision["classification"] == expected_decision["classification"], "CSV-recomputed primary decision differs from the paired-row calculation")
    _require(math.isclose(decision["difference_q_ba_minus_q_aa"], expected_decision["difference_q_ba_minus_q_aa"], rel_tol=0.0, abs_tol=1e-15), "CSV-recomputed main effect differs from the paired-row calculation")
    _require(math.isclose(decision["tolerance"], expected_decision["tolerance"], rel_tol=0.0, abs_tol=1e-15), "CSV-recomputed tolerance differs from the paired-row calculation")
    max_decomposition_error = max(
        abs(float(row["d_shape_symmetric"]) + float(row["d_energy_symmetric"]) - float(row["q_total_bb_minus_aa"]))
        for row in records
    )
    _require(max_decomposition_error <= 1.0e-12, "CSV-recomputed factorial decomposition does not close")
    return {
        "passed": True,
        "rows": len(records),
        "primary_confirmation_rows": len(confirm),
        "primary_classification_recomputed": decision["classification"],
        "primary_effect_recomputed": difference,
        "primary_decision": decision,
        "maximum_decomposition_error": max_decomposition_error,
    }


def _summarize(report_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("fit", "confirmation"):
        selected = [row for row in report_rows if row["role"] == role]
        result[role] = {"rows": len(selected), "metrics": {}}
        result[role]["mean_energy_ratio_b_over_a"] = _summary_values([float(row["energy_ratio_b_over_a"]) for row in selected])
        for name in CELL_ORDER:
            result[role]["metrics"][name] = {
                metric: _summary_values([float(row[f"{name}_{metric}"]) for row in selected])
                for metric in ("q", "energy", "clip_fraction", "raw_mse", "foreground_mse", "background_mse", "ssim", "unclipped_q_200", "unclipped_q_28")
            }
        for label in range(10):
            group = [row for row in selected if int(row["label"]) == label]
            result[role].setdefault("by_class", {})[str(label)] = {
                "rows": len(group),
                "q_by_cell": {cell: _summary_values([float(row[f"{cell}_q"]) for row in group]) for cell in CELL_ORDER},
                "energy_ratio_b_over_a": _summary_values([float(row["energy_ratio_b_over_a"]) for row in group]),
            }
        if role == "confirmation":
            q_aa = result[role]["metrics"]["AA"]["q"]["mean"]
            q_ba = result[role]["metrics"]["BA"]["q"]["mean"]
            difference = math.fsum(float(row["d_fixed_a_energy"]) for row in selected) / len(selected)
            result[role]["primary_decision"] = _classify_difference(difference, q_ba, q_aa)
            for endpoint in ENDPOINTS:
                prediction_key = f"actual_{endpoint}_e_d_prediction"
                labels = [int(row["label"]) for row in selected]
                predictions = [int(row[prediction_key]) for row in selected]
                result[role].setdefault("actual_endpoint_classifier", {})[endpoint] = {
                    "correct": sum(pred == label for pred, label in zip(predictions, labels, strict=True)),
                    "rows": len(selected),
                    "accuracy": sum(pred == label for pred, label in zip(predictions, labels, strict=True)) / len(selected),
                }
    for role in ("fit", "confirmation"):
        selected = [row for row in report_rows if row["role"] == role]
        result[role]["actual_endpoint_changes_from_A4700"] = {}
        for endpoint in ("A6700", "B6700"):
            q_differences = [float(row[f"actual_{endpoint}_q"]) - float(row["actual_A4700_q"]) for row in selected]
            energy_ratios = [float(row[f"actual_{endpoint}_energy"]) / float(row["actual_A4700_energy"]) for row in selected]
            transitions = {"wrong_to_right": 0, "right_to_wrong": 0}
            for row in selected:
                before = int(row["actual_A4700_e_d_prediction"]) == int(row["label"])
                after = int(row[f"actual_{endpoint}_e_d_prediction"]) == int(row["label"])
                transitions["wrong_to_right"] += int(not before and after)
                transitions["right_to_wrong"] += int(before and not after)
            result[role]["actual_endpoint_changes_from_A4700"][endpoint] = {
                "mean_delta_q": math.fsum(q_differences) / len(selected),
                "mean_energy_ratio_to_A4700": math.fsum(energy_ratios) / len(selected),
                "e_d_correctness_transitions": transitions,
            }
    return result


def _render_report(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    primary = result["population"]["confirmation"]["primary_decision"]
    summary = result["population"]
    return "\n".join(
        (
            f"# {TASK_ID}",
            "",
            "## 结果与状态",
            f"测量有效完成。唯一主比较判定：`{decision['classification']}`；确认人口 D_fixed={decision['difference_q_ba_minus_q_aa']:.10g}，数值容差 τ={decision['tolerance']:.10g}。",
            "",
            "## 科学认识",
            decision["interpretation"],
            f"确认人口 Q_BA={primary['mean_q_ba']:.10g}、Q_AA={primary['mean_q_aa']:.10g}。该结论只适用于冻结的A6700/B6700端点和300行canonical TRAIN确认人口；数值反事实不等于真实光学读出。",
            "",
            "辅助分解 D_shape={:.10g}、D_energy={:.10g}、交互 J={:.10g}。A/B能量、截断、未截断形态、重建误差、SSIM、逐类和实际端点分类读数见 report.json 与 paired_rows.csv。".format(
                summary["confirmation"]["decomposition"]["d_shape_symmetric"],
                summary["confirmation"]["decomposition"]["d_energy_symmetric"],
                summary["confirmation"]["decomposition"]["interaction"],
            ),
            "",
            "## 目标缺口",
            "本结果不证明人眼可读性、独立泛化、梯度因果、保能量训练可行性或完整生成成功；也不能补足实验一的审阅者重复一致性质控。",
            "",
            "## 工程完成与验证",
            f"核验 {result['validation']['source_rows']} 行来源身份与保存指标；四格数值测量覆盖 {result['validation']['counterfactual_rows']} 行。CSV回读和独立重算：{'通过' if result['validation']['csv_roundtrip']['passed'] else '失败'}。模型更新、光学／评价器前向、梯度、采样和人工评分均为0。",
            "",
            "## 决策影响与下一步",
            "本复盘只给出当前固定端点的数值证据，不构成B6700采用依据。停止于本报告；training_recommendation=NONE，next_action=NONE。",
            "",
        )
    )


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_output_path(root: Path, output_dir: Path) -> None:
    outputs = (root / "outputs").resolve()
    _require(outputs in output_dir.parents, "output must be a child of outputs/")
    relative = output_dir.relative_to(outputs)
    _require(len(relative.parts) == 3 and relative.parts[:2] == ("20261009_m0_generator_shape_energy_counterfactual_v1", "runs"), "output must be under the named experiment version and runs directory")
    _require(relative.parts[2].startswith("attempt_") and relative.parts[2][8:].isdigit(), "output attempt must have an explicit numeric suffix")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in relative.parts), "protected, restored, or temporary output namespace is forbidden")
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")


def _run(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    _validate_output_path(root, output_dir)
    _require(source_dir.resolve() == (root / SOURCE_ATTEMPT).resolve(), "source directory differs from the explicitly named attempt_03")
    output_dir.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "schema": "m0_generator_shape_energy_counterfactual_validation.v1",
        "task_id": TASK_ID,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "execution": {"device": "cpu", "model_updates": 0, "optical_forwards": 0, "evaluator_forwards": 0, "gradients": 0, "sampling": 0, "human_ratings": 0},
        "checks": checks,
    }
    _write_json = saved_evidence._write_json
    _write_json(output_dir / "preflight.json", {**validation, "status": "RUNNING"})
    try:
        population, source_report, readouts, summaries, endpoint_values, checks = _load_and_verify(source_dir)
        validation["checks"] = checks
        pixel_area = float(checks["camera_pixel_area_m2"])
        sample_indices = [FIT_ROWS + 10 * label for label in range(10)]
        sample = torch.tensor(sample_indices, dtype=torch.long)
        sample_grids, _sample_decomp = _measure_batch(
            readouts["A6700"]["raw_cmos"].index_select(0, sample),
            readouts["B6700"]["raw_cmos"].index_select(0, sample),
            readouts["A4700"]["target_slm_intensity"].index_select(0, sample),
            readouts["A4700"]["labels"].index_select(0, sample).to(torch.long),
            readouts["A4700"]["plane_metadata"],
            pixel_area,
        )
        _require(all(len(values["q"]) == len(sample_indices) for values in sample_grids.values()), "representative preflight did not measure all four cells")
        checks["representative_four_cell_preflight"] = True
        checks["representative_decomposition"] = True
        _write_json(output_dir / "preflight.json", {**validation, "status": "READY", "representative_source_positions": sample_indices})

        cell_values: dict[str, dict[str, list[float]]] = {cell: {} for cell in CELL_ORDER}
        decomposition: dict[str, list[float]] = {key: [] for key in ("d_fixed_a_energy", "d_fixed_b_energy", "d_shape_symmetric", "d_energy_symmetric", "interaction", "q_total_bb_minus_aa")}
        for start in range(0, TOTAL_ROWS, BATCH_ROWS):
            stop = min(start + BATCH_ROWS, TOTAL_ROWS)
            selection = slice(start, stop)
            batch_grids, batch_decomp = _measure_batch(
                readouts["A6700"]["raw_cmos"][selection],
                readouts["B6700"]["raw_cmos"][selection],
                readouts["A4700"]["target_slm_intensity"][selection],
                readouts["A4700"]["labels"][selection].to(torch.long),
                readouts["A4700"]["plane_metadata"],
                pixel_area,
            )
            for cell in CELL_ORDER:
                for metric, tensor in batch_grids[cell].items():
                    cell_values[cell].setdefault(metric, []).extend(float(value) for value in tensor.tolist())
            batch_q = {cell: batch_grids[cell]["q"] for cell in CELL_ORDER}
            batch_components = {
                "d_fixed_a_energy": batch_q["BA"] - batch_q["AA"],
                "d_fixed_b_energy": batch_q["BB"] - batch_q["AB"],
                "d_shape_symmetric": batch_decomp["d_shape"],
                "d_energy_symmetric": batch_decomp["d_energy"],
                "interaction": batch_decomp["interaction"],
                "q_total_bb_minus_aa": batch_decomp["total_change"],
            }
            for key, tensor in batch_components.items():
                decomposition[key].extend(float(value) for value in tensor.tolist())
        _require(all(len(values) == TOTAL_ROWS for cell in cell_values.values() for values in cell.values()), "counterfactual computation did not include all 500 rows")
        _require(all(len(values) == TOTAL_ROWS for values in decomposition.values()), "factorial decomposition did not include all 500 rows")

        paired = _paired_rows(population, readouts, endpoint_values, cell_values, decomposition)
        paired_path = output_dir / "paired_rows.csv"
        _write_csv(paired_path, paired)
        report_population = _summarize(paired)
        confirm_rows = paired[FIT_ROWS:]
        decomposition_by_role: dict[str, dict[str, float]] = {}
        for role, selected_rows in (("fit", paired[:FIT_ROWS]), ("confirmation", confirm_rows)):
            decomposition_by_role[role] = {
                key: math.fsum(float(row[key]) for row in selected_rows) / len(selected_rows)
                for key in ("d_shape_symmetric", "d_energy_symmetric", "interaction", "q_total_bb_minus_aa", "d_fixed_a_energy", "d_fixed_b_energy")
            }
            _require(abs(decomposition_by_role[role]["d_shape_symmetric"] + decomposition_by_role[role]["d_energy_symmetric"] - decomposition_by_role[role]["q_total_bb_minus_aa"]) <= 1e-12, f"{role} mean factorial decomposition does not close")
        report_population["fit"]["decomposition"] = decomposition_by_role["fit"]
        report_population["confirmation"]["decomposition"] = decomposition_by_role["confirmation"]
        expected_primary = report_population["confirmation"]["primary_decision"]
        csv_check = _recheck_csv(paired_path, population, expected_primary)
        primary = csv_check["primary_decision"]
        report_population["confirmation"]["primary_decision"] = primary
        result: dict[str, Any] = {
            "schema": "m0_generator_shape_energy_counterfactual_report.v1",
            "task_id": TASK_ID,
            "status": "EVIDENCE_REVIEW_COMPLETED",
            "source_task_id": SOURCE_TASK_ID,
            "source_report_classification": source_report["classification"],
            "population": {"total_rows": TOTAL_ROWS, "fit_rows": FIT_ROWS, "confirmation_rows": CONFIRM_ROWS, "per_class_confirmation": 30, "data_role": "canonical TRAIN empirical OLS decode; not claimed unseen", **report_population},
            "measurement": {"energy_definition": "camera_pixel_area_m2 * sum(raw_cmos_before_clipping)", "camera_pixel_area_m2": pixel_area, "production_view": "adaptive_avg_pool2d(clamp(raw_cmos,0,1),(28,28))", "q_definition": "sum((p(generated_view)-p(saved_target_view))^2)/sum(p(saved_target_view)^2); each view normalized by its own sum", "fixed_energy_atol": FIXED_ATOL, "fixed_energy_rtol": FIXED_RTOL, "counterfactuals_are_numeric_not_physical": True},
            "decision": primary,
            "auxiliary_decomposition": decomposition_by_role,
            "training_recommendation": "NONE",
            "next_action": "NONE",
            "stop": True,
            "scientific_scope": "Only the named frozen endpoints on the saved canonical TRAIN OLS population; no human-readability or generalization claim.",
            "paired_rows_csv": paired_path.name,
            "validation": {"source_rows": TOTAL_ROWS, "counterfactual_rows": TOTAL_ROWS, "csv_roundtrip": csv_check},
        }
        checks["csv_roundtrip"] = csv_check
        checks["counterfactual_rows"] = TOTAL_ROWS
        checks["source_rows"] = TOTAL_ROWS
        validation["status"] = "VALIDATED"
        _write_json(output_dir / "validation.json", validation)
        _write_json(output_dir / "report.json", result)
        _write_text(output_dir / "experiment_02_plan.md", _make_protocol())
        _write_text(output_dir / "report.md", _render_report(result))
        _write_json(output_dir / "preflight.json", {**validation, "status": "READY"})
        completion = {"task_id": TASK_ID, "status": result["status"], "classification": primary["classification"], "training_recommendation": "NONE", "next_action": "NONE", "stop": True}
        _write_json(output_dir / "completed.json", completion)
        return result
    except (CounterfactualError, saved_evidence.SavedEvidenceAuditError, energy_task.EnergyTargetError, OSError) as error:
        validation["status"] = "INVALID_EVIDENCE"
        validation["stop_reason"] = f"{type(error).__name__}: {error}"
        _write_json(output_dir / "validation.json", validation)
        _write_json(output_dir / "preflight.json", {**validation, "status": "INVALID_EVIDENCE"})
        failed = {"task_id": TASK_ID, "status": "INVALID_EVIDENCE", "scientific_decision": None, "training_recommendation": "NONE", "next_action": "NONE", "stop": True, "stop_reason": validation["stop_reason"]}
        _write_json(output_dir / "completed.json", failed)
        _write_text(output_dir / "report.md", "\n".join((f"# {TASK_ID}", "", "## 结果与状态", "测量无效；没有科学结论。", "", "## 科学认识", "本轮未形成有效测量，未判定形态或能量解释。", "", "## 目标缺口", f"证据核验停止：{validation['stop_reason']}", "", "## 工程完成与验证", "没有生成完整配对表或主统计。", "", "## 决策影响与下一步", "保留现有模型和实验判断；修复输入或测量错误后再重新核验，不自动启动训练或后继实验。", "")), encoding="utf-8")
        return failed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Explicit attempt_03 saved-evidence directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New versioned output run directory")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    source_dir = saved_evidence._resolve(root, args.artifact_dir)
    output_dir = saved_evidence._resolve(root, args.output_dir)
    result = _run(source_dir, output_dir)
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "output_directory": str(output_dir)}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "EVIDENCE_REVIEW_COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
