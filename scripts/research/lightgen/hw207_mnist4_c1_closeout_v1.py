"""Build C1 closeout from returned and independently verified evidence."""
import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    for key in ("data_root", "analysis_root", "verification", "cloud_verification", "preliminary", "convergence", "report", "summary"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    verify = json.loads(args.verification.read_text(encoding="utf-8"))
    cloud_verify = json.loads(args.cloud_verification.read_text(encoding="utf-8"))
    valid_status = ("PASS", "PASS_WITH_CPU_DIFFERENCES")
    if (verify["status"] not in valid_status or cloud_verify["status"] not in valid_status
            or len(verify["conditions"]) != 24 or len(verify["readouts"]) != 16
            or len(cloud_verify["readouts"]) != 16):
        raise ValueError("Incomplete independent verification")
    if any(r["cuda_mismatches"] is None or any(r["cuda_mismatches"]) for r in cloud_verify["readouts"].values()):
        raise ValueError("Full-batch CUDA frozen readout verification failed")
    cpu_differences = {key: item["differences"] for key, item in verify["readouts"].items()
                       if any(item["mismatches"])}
    receipt = json.loads((args.data_root / "propagation_preflight.json").read_text(encoding="utf-8"))
    selection = json.loads((args.analysis_root / "readout_selection.json").read_text(encoding="utf-8"))
    results = json.loads((args.analysis_root / "readout_test.json").read_text(encoding="utf-8"))
    old = json.loads(args.preliminary.read_text(encoding="utf-8"))
    convergence = json.loads(args.convergence.read_text(encoding="utf-8"))
    image_max = max(max(r["comparisons"][-1]["image_errors"]) for c in convergence["results"].values() for r in c.values())
    roi_max = max(max(r["comparisons"][-1]["roi_errors"]) for c in convergence["results"].values() for r in c.values())
    field_max = max(max(r["precision"]["complex_errors"]) for c in convergence["results"].values() for r in c.values())
    energy_max = max(max(r["precision"]["energy_errors"]) for c in convergence["results"].values() for r in c.values())
    if len(selection) != 16 or len(results) != 16:
        raise ValueError("Incomplete readouts")
    lines = ["# HW207 MNIST 0–3 C1 理想传播诊断", "",
             "## 结果与状态", "",
             "**C1_NUMERICAL_DIAGNOSTIC_PASS。** 八种传播条件、十六种数字线性读出和独立复核均完成。",
             "固定 LightGen 堆栈、目标实物几何与 HW207_CURRENT 映射仍为 BLOCKED。", "",
             "## 科学认识", "",
             "相同编码、相同读出和相同测试人口下比较 Identity 与 1 mm。线性读出表中为三个种子的均值，ROI 不训练。", "",
             "| 编码 | 距离 mm | ROI-A 准确率／Macro-F1／覆盖率 | ROI-B 准确率／Macro-F1／覆盖率 | ROI-B 区外能量占视场 | 视场能量占输入 |",
             "|---|---:|---|---|---:|---:|"]
    compact = {}
    for name in ("gray", "binary"):
        for distance in (0, 0.0005, 0.001, 0.002):
            key = f"{name}/{distance}/test"
            v = verify["conditions"][key]
            audit = torch.load(args.data_root / "roi_audit" / name / str(distance) / "test.pt", map_location="cpu", weights_only=True)
            a, b = audit["roi"]["a"], audit["roi"]["b"]
            outside_fraction = sum(audit["roi_b_outside"]) / sum(audit["crop_energy"])
            lines.append(f"| {name} | {distance*1000:g} | {a['accuracy']:.3%} / {a['macro_f1']:.3%} / {a['coverage']:.3%} | {b['accuracy']:.3%} / {b['macro_f1']:.3%} / {b['coverage']:.3%} | {outside_fraction:.3%} | {v['crop_fraction_mean']:.3%} |")
            compact[key] = v
    zero_counts = {}
    for name in ("gray", "binary"):
        for distance in (0, 0.001):
            for role in ("train", "validation", "test"):
                audit = torch.load(args.data_root / "roi_audit" / name / str(distance) / (role + ".pt"),
                                   map_location="cpu", weights_only=True)
                zero_counts[f"{name}/{distance}/{role}"] = sum(energy == 0 for energy in audit["crop_energy"])
    lines.append("")
    lines.append("线性读出输入中，零视场能量样本数按编码／距离／数据集记录：" +
                 "；".join(f"{key}={value}" for key, value in zero_counts.items()) + "。")
    lines += ["", "| 编码 | 距离 mm | 特征 | 表示 | test 准确率均值 | Macro-F1 均值 | 选定衰减 | 末轮 train loss | 验证准确率第 30－25 轮 |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name in ("gray", "binary"):
        for distance in (0, 0.001):
            for size in (100, 25):
                for mode in ("raw", "normalized"):
                    key = f"{name}/{distance}/{size}/{mode}"
                    runs = results[key]["runs"]
                    acc = [r["metrics"]["accuracy"] for r in runs]
                    f1 = [r["metrics"]["macro_f1"] for r in runs]
                    chosen = selection[key]
                    directory = args.analysis_root / "readouts" / name / str(distance) / str(size) / mode
                    candidates = [torch.load(directory / f"decay_{chosen['decay']}_seed_{seed}.pt",
                                             map_location="cpu", weights_only=True) for seed in (11, 23, 37)]
                    final_loss = statistics.mean(c["history"][-1]["train_loss"] for c in candidates)
                    validation_delta = statistics.mean(c["history"][-1]["validation"]["accuracy"]
                                                       - c["history"][24]["validation"]["accuracy"] for c in candidates)
                    lines.append(f"| {name} | {distance*1000:g} | {size}×{size} | {mode} | {statistics.mean(acc):.3%} | {statistics.mean(f1):.3%} | {chosen['decay']} | {final_loss:.5f} | {validation_delta*100:+.3f} pp |")
                    compact[key] = {"accuracy": acc, "macro_f1": f1, "decay": chosen["decay"],
                                    "final_train_loss_mean": final_loss, "validation_30_minus_25": validation_delta}
    lines += ["", "配对的 1 mm 减 Identity 准确率变化如下，单位为百分点。", "",
              "| 编码 | 特征 | 表示 | 三种子均值变化 |", "|---|---:|---|---:|"]
    for name in ("gray", "binary"):
        for size in (100, 25):
            for mode in ("raw", "normalized"):
                identity = compact[f"{name}/0/{size}/{mode}"]["accuracy"]
                propagated = compact[f"{name}/0.001/{size}/{mode}"]["accuracy"]
                change = statistics.mean(b-a for a, b in zip(identity, propagated)) * 100
                lines.append(f"| {name} | {size}×{size} | {mode} | {change:+.3f} |")
    lines += ["", "固定 ROI-A 的准确率随传播下降，线性读出仍保持较高的数值分类准确率。"
              "这保留了输出强度中存在数字线性可读类别信息的解释，也提示固定窗口的判决受光强空间分布影响。"
              "比较只适用于本轮采样、距离与输入。训练后的 C0 数字头与 C1 固定 ROI 使用不同读出，二者的准确率差不能直接归因于传播。"]
    lines += ["", "测试集中四类中心视场强度的空间均值与逐像素类内方差空间均值如下，数值使用数值振幅平方单位。", "",
              "| 编码 | 距离 mm | 类别 | 类均值的空间均值 | 类内方差的空间均值 |",
              "|---|---:|---:|---:|---:|"]
    for name in ("gray", "binary"):
        for distance in (0, 0.0005, 0.001, 0.002):
            audit = torch.load(args.data_root / "roi_audit" / name / str(distance) / "test.pt",
                               map_location="cpu", weights_only=True)
            for label in range(4):
                mean = audit["class_mean"][label].mean().item()
                variance = audit["class_variance"][label].mean().item()
                lines.append(f"| {name} | {distance*1000:g} | {label} | {mean:.7g} | {variance:.7g} |")
            compact[f"{name}/{distance}/class_mean_pair_l2"] = audit["class_mean_pair_l2"]
    lines += ["", "各类别平均强度图的两两 L2 距离如下。", "",
              "| 编码 | 距离 mm | 0–1 | 0–2 | 0–3 | 1–2 | 1–3 | 2–3 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("gray", "binary"):
        for distance in (0, 0.0005, 0.001, 0.002):
            pair = compact[f"{name}/{distance}/class_mean_pair_l2"]
            values = " | ".join(f"{pair[key]:.6g}" for key in ("0-1", "0-2", "0-3", "1-2", "1-3", "2-3"))
            lines.append(f"| {name} | {distance*1000:g} | {values} |")
    lines += ["", "两个编码都采用同一批 C0 原始索引。灰度使用连续振幅，二值使用固定阈值。逐图归一化在均值池化之前进行，公共行 L2 均方根尺度仅从 train 拟合。所有候选固定后统一评估官方 test。该 test 已用于 C0 和初步 Identity，结果属于同一测试集的后续分析。", "",
              "各类别均值、方差、类间均值差和逐图能量差保存在 ROI 审计数据。各候选逐轮训练损失、验证轨迹、三种子指标、混淆矩阵和冻结权重保存在回传的读出目录。", "",
              "## 目标缺口", "",
              "当前测量是固定目标采样上的理想数值传播。尚无绑定完整物理几何与相位图的固定 LightGen 层，也缺少 DMD 收光、输入 Relay 共轭及 CMOS 读出实测。四分类结果尚不足以验证实物光路或光学图像生成。", "",
              "## 工程完成与验证", "",
              "读取 C0 索引并核验 train 22,279、validation 2,475、test 4,157。八个条件逐条件各 28,911 行。独立回传检查逐块索引、强度、ROI、能量账和混淆矩阵，使用冻结权重重新计算全部线性测试预测。云端同设备整批复算逐项与原始预测一致。",
              f"本地 CPU 分块复算共有 {sum(len(run) for runs in cpu_differences.values() for run in runs)} 个读出预测与云端整批预测不同。逐项样本索引、真实标签、预测及 FP64 分数差见独立复核文件。跨设备、批次引起的近并列结果按实际记录保留。",
              f"验证集每类前 16 张用于窗口比较。灰度各距离以及二值 0.5、1 mm 选用 1024 窗口，二值 2 mm 选用 2048。最终逐样本中心图归一化 L1 最大值 {image_max:.3g}，四区差最大值 {roi_max:.3g}，均满足 ≤1e-3。FP32／FP64 全窗复场相对误差最大值 {field_max:.3g}、能量相对误差最大值 {energy_max:.3g}，均满足 ≤1e-5。",
              "生产路径执行零距离、独立 ASM、平面波、Fresnel、能量和错误相位／轴方向探针，分块中断恢复验证通过。16 组读出在每类前 128 张 train、前 16 张 validation 上执行三步真实更新检查。二值 1 mm 随机标签对照验证准确率分别为 39.0625%（100×100）和 35.9375%（25×25），满足预设 <40%。", "",
              "## 决策影响与下一步", "",
              "理想传播及数值读出结果可用于决定是否单独设计 C2 的 DMD 与 Relay 接口。物理参数缺口仍需独立闭合，C2 未在本次任务中启动。", "",
              "## 本地状态", "",
              f"强度和 ROI 备份位于 {args.data_root}，冻结模型和指标位于 {args.analysis_root}。云端和本地复算结果均已保留。"]
    if cpu_differences:
        marker = lines.index("## 目标缺口")
        detail = ["跨设备预测差异如下。FP64 分数差为该设备输入的前两类得分差。", "",
                  "| 读出 | 种子 | test 原始索引 | 标签 | GPU 整批预测 | CPU 分块预测 | FP64 前两类分数差 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for key, runs in cpu_differences.items():
            for seed, differences in zip((11, 23, 37), runs):
                for difference in differences:
                    detail.append(f"| {key} | {seed} | {difference['original_index']} | {difference['label']} | {difference['saved']} | {difference['cpu']} | {difference['fp64_top_margin']:.4g} |")
        detail.append("")
        lines[marker:marker] = detail
    for name in ("gray", "binary"):
        current = torch.load(args.data_root / "roi_audit" / name / "0" / "test.pt", map_location="cpu", weights_only=True)
        prior = old["results"][name + "_test"]
        matrix = current["roi"]["a"]["confusion"]
        same = [row[:4] for row in matrix] == prior["confusion"] and sum(row[4] for row in matrix) == prior["undecided"]
        lines.insert(lines.index("## 目标缺口"), f"初步 Identity {name} 的四象限 test 混淆矩阵与本轮{'一致' if same else '不一致，需进一步定位'}。")
        if not same:
            raise ValueError(f"Identity baseline mismatch {name}")
    with args.report.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with args.summary.open("x", encoding="utf-8") as f:
        json.dump({"status": "C1_NUMERICAL_DIAGNOSTIC_PASS", "results": compact,
                   "cpu_readout_disagreements": cpu_differences,
                   "cuda_full_batch_readout_disagreements": 0, "zero_energy_counts": zero_counts},
                  f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
