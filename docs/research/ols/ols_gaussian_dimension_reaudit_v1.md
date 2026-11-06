# Gaussian 主参考下的 OLS 零更新复核

## 结果与状态

本轮云端数值任务以 shell 退出码 0 完成，测量状态为 **`VALID_FAILED / G3_FAIL`**。在未改变的复场相对误差门槛 `1e-4` 下，complex64 生产精度和 complex128 高精度诊断均未通过共同物理 ROI 的复场收敛比较。按计划，OLS 响应矩阵、奇异值、维度排序和候选推荐均未生成；它们被 G3 Gate 阻止，而不是测量失败后改用替代输入。

唯一决策是：**现有证据不足以支持下一轮受控的 OLS 维度优化。** 此结论不推翻此前 Gaussian 公式验证，也不改变既有维度扫描的历史记录。

模型采用记录：用户已选定后续 LightGen 光纤 OLS 仿真以完整 Gaussian S12 为主参考、同几何 LP01 为模式敏感性对照。此选择只界定未来仿真参考，不表示 Gaussian 与实物更吻合；正式运行默认模型、旧 checkpoint、旧训练参考和其他研究分支均未迁移。

正式训练更新、phase optimizer 更新、数据集读取、任务评价器/FID 调用和 loss 修改均为 0。

## 科学认识

本轮固定 532 nm、2 μm 采样、两层 100×100（200 μm）相位屏、1.5 mm 层间距、末层至 fiber plane 1.5 mm；10×10 阵列假设 pitch 125 μm、MFD 14.222 μm。总 mask-to-fiber 传播距离为 3.0 mm。所有 fiber 参数仍是 `PROJECT_GEOMETRY_UNMEASURED` 仿真假设。相位屏只取此前保存的零相位、末层 x 向 1°、末层 x/y 各 1°三种状态，没有更新。

窗口尺寸分别为 2048²、3072²、4096²，均保持 `dx=2 μm`，即 4.096、6.144、8.192 mm。三者都裁到相同的 2048² / 4.096 mm 物理 ROI：较大窗口对应索引 512:2560 和 1024:3072。没有 resize；复场、强度、功率及 fiber latent 均在这个共同 ROI 上比较。

各窗口对的最坏共同全局相位对齐复场误差如下。每个误差都是三个固定相位状态中的最大值；Gate 仍为 `1e-4`。

| 精度 | ROI 窗口对 | 最坏相位状态 | 对齐复场相对误差 | 结果 |
| --- | --- | --- | ---: | --- |
| complex64 | 4.096 vs 6.144 mm | x 向 1° | 6.9225e-4 | FAIL |
| complex64 | 4.096 vs 8.192 mm | x/y 各 1° | 5.9049e-4 | FAIL |
| complex64 | 6.144 vs 8.192 mm | x 向 1° | 5.5205e-4 | FAIL |
| complex128 | 4.096 vs 6.144 mm | x/y 各 1° | 1.2435e-4 | FAIL |
| complex128 | 4.096 vs 8.192 mm | x/y 各 1° | 1.6014e-4 | FAIL |
| complex128 | 6.144 vs 8.192 mm | x/y 各 1° | 3.6067e-5 | PASS |

complex64 全部 9 个相位/窗口比较的最坏对齐误差为 `6.9225e-4`，raw 最坏值 `6.9380e-4`；complex128 最坏对齐误差为 `1.6014e-4`，raw 最坏值 `1.6014e-4`。最大共同全局相位分别只有 `9.44e-5 rad` 和 `3.89e-7 rad`，相位对齐几乎没有降低误差。因此 global phase 不是主要解释。complex128 仍有超过门槛的项，也排除了“只因 complex64 累积精度”这一解释。

其他读数稳定得多：complex64 的最大强度相对误差为 `4.4104e-5`、共同 ROI 积分功率误差为 `2.3639e-6`；complex128 分别为 `3.6802e-5` 和 `2.9119e-8`。Gaussian/LP01 两种投影的 complex128 逐 fiber 功率 L2 差最大 `6.40e-8`，共同相位对齐后的 latent 差最大 `7.07e-8`；complex64 对应最大值为 `3.95e-5` 和 `2.20e-4`。所以总功率和 fiber 读出接近，并不能替代全复场的严格 G3 Gate。

绝对仿真功率读数为：均匀相干输入在 encoder entrance 为约 `249.3 μW`，经本轮两层/传播后的 fiber 前平面约 `184.675 μW`；各窗口裁到共同 ROI 后保留的总功率比例接近 1。该值由配置的仿真输入功率和场幅计算得出，不是激光器或器件实测值。

对来源的判别：

- **裁切不匹配**没有证据支持：像素间距相同、ROI 物理坐标与索引明确一致；单像素错位负控产生 `0.1247` 相对误差并被拒绝。
- **全局相位**不是主因：对齐前后误差接近，且实测共同相位很小。
- **复数精度**有贡献，但不是全部原因：complex64 误差普遍更大；complex128 仍在涉及 4.096 mm 窗口的非零相位比较中失败。
- 最支持的来源是**有限窗口下的离散频率采样差异**。固定 `dx` 时，三个窗口的频率间距分别为 `244.14`、`162.76`、`122.07 cycles/m`；complex128 的 6.144–8.192 mm 比较通过，而 4.096 mm 与较大窗口的部分比较失败。
- 采样 Nyquist 频率及其二维角点均低于 `1/λ`，因此本离散频带没有采到 evanescent 分量；evanescent 截断不是当前证据支持的原因。各窗口 20 μm 边界带功率比例很低，但这不足以单独证明或排除 FFT 周期边界贡献，故不将结果简单归因于 wrap-around。

独立 NumPy CPU 复算在三个固定状态下与 GPU complex128 单窗口传播的相对误差为 `1.17e-15`、`1.20e-15`、`1.21e-15`；解析平面波正控误差为 `2.22e-15`，错误传递相位负控误差为 `0.0362`。这验证了各单窗口实现和解析传播核，不会把跨窗口复场差异变成 PASS。

## 目标缺口

本轮没有建立 `c=K a` 输入—fiber 响应矩阵，也没有计算复响应秩、实幅度控制奇异值、探针可控性或新的 N_eff 排序。不能据此选择 fiber 通道数或声称某个 OLS 维度可用。此前 compact 维度优化没有保存优化后相位屏，旧优化结果仍只能作为历史测量；本轮没有重放或替代它。

逐通道复 latent 的误差向量也尚未补齐：已回传报告包含100通道 latent 向量的整体 raw/aligned 相对 L2 误差和逐 fiber 功率误差，但没有落盘每根 fiber 的复系数误差数组。为避免把整体误差误称为逐通道复误差，这项交付明确保留为未完成。

Gaussian S12 的公式与模式接口已有独立数值验证，且用户已选择它作为未来模拟主参考；本轮新发现是跨窗口复场的数值收敛仍未闭合。二者并不矛盾：模式定义通过，不等于当前生产传播离散化已经满足 G3，也不等于实物 fiber 参数已经标定。

此前 [physical reachability/convergence attempt_06](../../../outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_06/physical_reachability_and_convergence_v2.json) 的 G3 对齐复场误差约 `6.62e-4` 保留为历史测量；本轮结果写入新的 attempt，没有覆盖旧报告。既有 compact 维度扫描也保留在[原始结果](../../../outputs/20261013_compact_200um_ols_dimension_optimization_v1/runs/autodl_backup_results/full_measurement/compact_200um_ols_dimension_results.json)。该扫描没有保存优化相位屏，故不把其优化端点声称为本轮已验证或可重放结果。

fiber 几何、MFD、真实复传输矩阵、探测噪声、相机响应与实际功率校准仍未知。因此本轮不支持真实阵列可达性、实验有效维数、非线性材料阈值或硬件部署结论。

## 工程完成与验证

已在 AutoDL `autodl-monthly`、`/root/miniconda3/bin/python`、PyTorch 2.5.1+cu124、RTX 4080 上运行。零更新代表性预检通过，包含生产有限裁切相位栈、Gaussian/LP01 投影、模式范数/被动性、complex64 投影、complex128 NumPy 参考、解析平面波和相位/裁切负控。完整测量耗时观测 86.37 s，云端命令退出码 0，遥测采集 2 次、采集失败 0 次；科学状态仍是 `VALID_FAILED`。

18 个共同 ROI 原始复场（3 相位 × 2 精度 × 3 窗口）已回传并加载验证：每个张量为 2048×2048，complex64/complex128 类型正确，全部有限。主回传 26 个文件、共 907,134,033 bytes；独立预检回传 7 个文件、共 5,700 bytes；两份收据均为 `COMPLETE` 且逐文件大小匹配。完整逐 fiber 误差、功率、相位、输入配置、命令、日志、退出状态、遥测、部署源码以及云端原源码备份见 [attempt_01 证据目录](../../../outputs/20261013_ols_gaussian_dimension_reaudit_v1/runs/autodl_backup_attempt_01) 和 [preflight_01 证据目录](../../../outputs/20261013_ols_gaussian_dimension_reaudit_v1/runs/autodl_backup_preflight_01)。原始 complex ROI 张量是 `g3_common_roi_fields.pt`；维度响应矩阵未生成，因为 G3 门控失败。

云端原有的物理运行时、fiber operator 和相位栈各自备份后临时覆盖；运行结束后均由备份恢复并逐字节比较通过。临时诊断入口、配置及云端原先不存在的 Gaussian/材料模块已移除。云端保留本轮输入、日志和可重放源码副本；本地新增执行器与配置分别为 [诊断脚本](../../../scripts/research/lightgen/diagnose_ols_gaussian_dimension_reaudit_v1.py) 和 [冻结配置](../../../configs/research/lightgen/ols_gaussian_dimension_reaudit_v1.yaml)。本地 `compileall`、CLI、配置/相位输入核验及 Ruff `F821/F841` 检查通过；未运行全仓 pytest。

为补齐上述逐通道复 latent 误差，已准备独立的 saved-field 后处理入口 [postprocessor](../../../scripts/research/lightgen/postprocess_ols_gaussian_dimension_latent_errors_v1.py)，本地 `py_compile` 和 Ruff 检查通过。云端后处理没有启动：本轮追加的 `autodl_run -Action Check` 在提交命令前以 SSH 255 退出，提示无法解析 `connect.westb.seetacloud.com`；本机系统 DNS（127.0.0.1）、显式 1.1.1.1 查询均超时，HTTPS DNS 查询也因本地代理 127.0.0.1 不可用而失败。因此没有上传脚本、投影复场或产生后处理结果/收据。原始复场仍已完整回传；恢复 AutoDL 网络/SSH 解析后，只需对已保存 G3 common-ROI 字段执行这个零更新投影，不需重跑传播或维度测量。

本地没有修改训练入口、production runtime、OLS 几何或 loss；本轮新增的生产代码文件仅是独立诊断脚本，另有固定配置、机器报告和状态导航更新。

功率训练接口审计没有修改训练入口：`physical_plane_power` 通过 `dy·dx·Σ|E|²` 积分；在场振幅已标定为 √W/m 时单位才是 W。当前 `L_op` 仍是五平面相对冻结参考功率的 `mean(log(P/P_ref)^2)`，并通过既有权重进入总 loss；它约束功率相对参考的变化，不是最大化 fiber coupling 的目标。当前重建项消费 `result.raw_intensity`。物理前向另有 W/pixel 的 `physical_sensor_output_w_per_pixel`，以及在固定 `camera_gain_w_inverse` 已设置时生成的裁剪 `task_sensor_output`，但该 task 输出目前不是重建项的消费接口。

## 决策影响与下一步

**2026-10-14连接纠正：** 相同AutoDL Check在Codex沙箱内报SSH255，在沙箱外成功；screen、CUDA工程探针和SCP回传均通过。此前“本机DNS/代理需要修复”的推断撤回，实际应使用已授权的沙箱外网络通道，详见[连接修复记录](../../operations/autodl_connection_repair_v1.md)。逐通道latent后处理仍未执行，但连接不再阻塞；本轮连接修复不改变以下G3科学结果。

后续仿真主参考采用 Gaussian S12，LP01 作为相同几何下的模式敏感性对照；这项用户选择已记录，不自动修改生产默认模型或旧 checkpoint。当前 G3 继续 FAIL，因此不对 OLS 维度作结构排序，也不开始训练。

后续训练中，功率会进入**物理前向、过程监控和最终报告**：记录光源、SLM 后、fiber 前、逐 fiber、decoder 和 sensor 功率及耦合效率/N_eff；最终同时给出任务指标、逐通道 min/p10/p50/mean/p90/max 和固定 camera gain 下的裁剪比例。**不新增功率损失。**保留现有 `L_op`，但在 Gaussian 训练获单独授权前，先核实绝对标定、basis-specific 冻结参考值、单位和梯度兼容性；不沿用 LP01 的旧功率参考。camera gain 仍是固定的仿真校准参数。当前 W/pixel → task 输出 → reconstruction 消费接口尚未闭合，本轮没有修改它。

2026-10-14沙箱外AutoDL连接已恢复；后续可沿用该通道补齐基于已保存复场的逐通道 latent 误差数组，随后另行设计 G3 离散频率/窗口误差定位与修复验证，保持 `1e-4` 门槛、传播几何和输入相位不变。只有生产 complex64 G3 通过后，才能规划受控维度响应测量。该建议不是对后续诊断、正式训练、评价器、loss、材料模型或光路变更的授权；本轮在记录当前阻塞和结果后停止。

## 本地状态

仓库分支为 `research/lightgen-ols-native-semantic-generation-v1`，HEAD 为 `b9d865ebce77934216e0dee1a884c44fc3775992`；未创建提交。工作树在本任务开始前已有多处用户修改。本轮新增执行器、配置、复核报告和机器 artifact，并在既有 `CURRENT_STATUS.md`、`current_handoff.md`、`project_map.md` 与 Gaussian 对齐报告中作了定点更新；没有更改 `PROJECT_MEMORY.md`、训练入口或共享物理前向源码。
