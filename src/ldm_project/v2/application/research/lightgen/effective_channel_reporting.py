from __future__ import annotations

from typing import Any


def preflight_status(report: dict[str, Any]) -> str:
    return "通过" if report["preflight"]["status"] == "READY" and report["preflight"]["all_100_port_complex_camera_bases_compared_to_reference"] and report["preflight"]["direct_replay_and_ablation_checks_passed"] else "未通过"


def render_effective_channel_report(report: dict[str, Any]) -> str:
    endpoint_rows = []
    for endpoint, summary in report["decoder_ablation"].items():
        native = summary["native_population_metrics"]
        raw = summary["controls"]["raw_zero"]["population_mean"]["mse28"]
        matched = summary["controls"]["equal_ols_power"]["population_mean"]["mse28"]
        endpoint_rows.append(
        f"| {endpoint} | {summary['input_role']} n={summary['sample_count']} | {native['mse28']['mean']:.6g} [{native['mse28']['ci95_low']:.6g}, {native['mse28']['ci95_high']:.6g}] | {raw['mean_delta']:.6g} [{raw['ci95_low']:.6g}, {raw['ci95_high']:.6g}] | {matched['mean_delta']:.6g} [{matched['ci95_low']:.6g}, {matched['ci95_high']:.6g}] | {summary['mean_modal_participation_neff']:.3f} | {summary['population_mean_power_vector_neff']:.3f} | {summary['population_mean_port_power_spread_db']:.2f} |"
        )
    noise_rows = []
    for endpoint, controls in report["noise_observability"].items():
        for control, exposures in controls.items():
            for exposure, stats in exposures.items():
                noise_rows.append(f"| {endpoint} | {control} | {float(exposure):g} | {stats['mean_observable_roots_snr_ge_1']:.2f} | {stats['mean_observable_roots_snr_ge_3']:.2f} | {stats['mean_observable_roots_snr_ge_5']:.2f} |")
    covariance_rows = []
    for population, stats in report["covariance"].items():
        covariance_rows.append(f"| {population} | {stats['sample_count']} | {stats['effective_participation']:.2f} | {stats['directions_95_percent']} | {stats['directions_99_percent']} | {stats.get('within_class_fraction_of_total', float('nan')):.3f} | {stats.get('between_class_fraction_of_total', float('nan')):.3f} |")
    retention_rows = []
    for endpoint, values in report["cumulative_retention"]["tolerance_minimum_roots"].items():
        retention_rows.append(f"| {endpoint} | {values['0.01'] if values['0.01'] is not None else '未达到'} | {values['0.05'] if values['0.05'] is not None else '未达到'} |")
    local_rows = []
    for endpoint, payload in report["noise_weighted_local_direction_spectra"].items():
        for exposure, spectrum in payload["exposures"].items():
            counts = spectrum["relative_dimension_counts"]
            local_rows.append(f"| {endpoint} | {float(exposure):g} | {counts['0.1']} | {counts['0.01']} | {counts['0.001']} | {counts['0.0001']} | {spectrum['global_phase_direction_relative_response_max']:.3g} |")
    lines = [
        "# Gaussian OLS有效通道：协方差、代理噪声与冻结消融",
        "",
        "## 结果与状态",
        "",
        f"状态为 **{report['status']}**。四个Decoder端点均冻结；小架构每个端点使用固定5,000张DEV，大架构使用保存的300张TRAIN系数。55,000张TRAIN只用于恢复历史人口协方差。训练更新、优化器创建、先验采样、官方TEST读取和硬件操作均为0。",
        "",
        "| 端点 | 人口角色 | 原生28×28 MSE（95%分层Bootstrap CI） | 单根原功率消融ΔMSE | 单根OLS等总功率消融ΔMSE | 平均单图N_eff | 人口平均功率向量N_eff | 平均功率跨度(dB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *endpoint_rows,
        "",
        "等总功率条件只按剩余OLS系数共同缩放，使OLS模态总功率回到原值；没有匹配相机功率或重拟合数字增益。ΔMSE为移除该根后的MSE减去原生MSE，正值代表变差，负值代表改善。所有100根、两种消融的逐图结果在`ablation/per_image_per_port_ablation.csv.gz`。",
        "",
        "| 端点 | 协方差人口 | 200维Gauge实虚协方差参与数 | 95%方差方向数 | 99%方差方向数 | 类内方差占比 | 类间方差占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *covariance_rows,
        "",
        "| 端点 | 消融控制 | 曝光(s) | 平均SNR≥1根数 | 平均SNR≥3根数 | 平均SNR≥5根数 |",
        "|---|---|---:|---:|---:|---:|",
        *noise_rows,
        "",
        "噪声数值是条件代理扫描。使用QE=0.5、读噪声1 e⁻ RMS、暗电流0、满阱30,000 e⁻；按Poisson光子/暗电流计数与Gaussian读噪声卷积，并对每个像素施加单侧满阱截断，直接计算均值和方差。读噪声截断只在满阱附近的有界计数区间求和，区间外尾概率低于预设数值界。固定曝光扫描10 μs、100 μs、1 ms、10 ms，保留全部曝光。它不是相机实测的有效通道数。",
        "",
        "| 端点 | 最少保留根数：平均相对MSE误差≤1% | 最少保留根数：平均相对MSE误差≤5% |",
        "|---|---:|---:|",
        *retention_rows,
        "",
        "| 端点 | 曝光(s) | 相对奇异值≥0.1方向数 | ≥0.01 | ≥0.001 | ≥0.0001 | 公共相位方向相对响应 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *local_rows,
        "",
        "局部谱是在每类固定20张图上累计的噪声加权Decoder输出方向谱；计数按200个实/虚系数方向及给定相对阈值计算。公共相位方向应接近零。该维数描述冻结端点和读出局部敏感度，不等于光纤数、独立信息容量或可部署压缩维数。",
        "",
        "## 科学认识",
        "",
        "功率跨度、功率参与数、相机噪声下可观察端口数、逐端口Decoder消融效应和噪声加权局部方向数是不同量。约48 dB与约32 dB是各自固定人口平均功率向量内的最大/最小跨度；大架构来自TRAIN保存矩与固定TRAIN子样本，小架构来自DEV，Decoder和几何也不同，因此不能把两者差值直接解释为架构有效通道损失。",
        "",
        "若等总功率消融显著缩小单根原功率消融造成的损失，说明总功率变化解释了部分影响；若等总功率后仍恶化，说明端口对应的空间/相位分布含有任务相关贡献。单根影响小而累计保留曲线快速恶化，说明存在联合冗余。协方差方向多而局部相机谱低，支持Decoder/读出限制；两谱都高而代理SNR占用下降，支持代理噪声限制。最终逐端点数值见上表和机器报告，不把中间条件外推到硬件。",
        "",
        "## 目标缺口",
        "",
        "真实暗帧/读噪声、平场、QE、曝光、满阱、增益、波长、ROI、binning和有效像素面积均未实测齐备，所以真实相机下有多少根可区分仍未确定。大架构55,000 TRAIN协方差仅保存了最大端口公共相位规范后的实虚统计；原始逐样本公共相位信息不能从这些矩恢复。数值模拟也没有证明100根实体光纤可独立控制或达成相近耦合。",
        "",
        "## 工程完成与验证",
        "",
        f"生产/参考网格100端口复场与相机功率检查：{preflight_status(report)}。生产Decoder直算与基场合成的最大MSE绝对差为{report['preflight']['direct_replay_max_mse_absolute_error']:.3g}；生产W/pixel与复场面积分功率的最大相对差为{report['preflight']['production_w_per_pixel_power_identity_max_relative_error']:.3g}。错序、错行、漏面积、漏相干交叉项、满阱截断反例均在预检中核对。分层配对Bootstrap为2,000次、seed `2026102101`；逐根检验按每端点/每消融控制分别做100根BH-FDR，q=0.05。每类图像量和全部逐根输出分块保存并回读验证。",
        "",
        "主要结果：`machine_report.json`；复协方差和特征值：`covariance/`；逐图逐根消融：`ablation/per_image_per_port_ablation.csv.gz`；逐根逐类：`ablation/per_port_per_class.csv`；逐根功率：`ablation/per_port_power_summary.csv`；逐图代理可观察根数：`ablation/per_image_noise_observable_port_counts.csv`；累计保留：`ablation/cumulative_root_retention.csv`；噪声方向谱：`noise/local_direction_spectra.json`。",
        "",
        "## 决策影响与下一步",
        "",
        "本轮只支持固定模型、固定人口和指定代理噪声参数下的条件结论。建议取得与相机/波长/曝光/增益/ROI/binning/像素平面匹配的暗帧、平场和标定参数后，复用保存的Decoder基场补算真实噪声观察数；这项建议不构成启动训练、调整损失/几何或采用删减通道模型的授权。历史`VALID_FAILED`不变，真实噪声标定仍待补。",
        "",
    ]
    return "\n".join(lines)
