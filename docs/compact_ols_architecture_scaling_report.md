# Compact OLS architecture scaling study

## 结果与状态

结果为 **VALID_COMPLETED / SIMULATION_SCREEN_MET_SENSITIVITY_PITCH**。本轮在云端完成了 280 个计划扫描点和 4 个固定 1.5 mm 对照点；每点运行 total-coupling、uniform-distribution、effective-dimension 三个 phase-only 目标，各 24 步，共 20,448 个诊断优化步。训练更新、数据集读取、任务 evaluator 调用、loss 改动、材料非线性前向和生产几何改动均为 0 / false。

配置固定为 532 nm、2 μm 像素间距、两层 100×100 phase mask、2048×2048 global propagation grid（4.096 mm 视场）；两层间距 1.5 mm，当前末层至 SMF 面距离 1.5 mm。扫描 aperture 为 100/150/200/250/300/400/500 像素（200–1000 μm），末层至纤面距离为 0.5/1/2/3/5/8/10/15 mm，fiber pitch 为 125/100/75/62.5/50 μm，通道数始终为 10×10。125 μm pitch 是参考几何，其他 pitch 都是敏感性假设；fiber MFD 14.222 μm、125 μm 包层等也未经过本项目实物测量。

参考 pitch 下的 fiber 中心坐标轴为 −562.5、−437.5、−312.5、−187.5、−62.5、62.5、187.5、312.5、437.5、562.5 μm。最外中心为 x/y=±562.5 μm，中心跨度 1.125 mm；按 125 μm 包层计算的阵列外廓为 1.25 mm。采样相位极限为 sin(theta_max)=532 nm/(2×2 μm)=0.133，theta_max=7.643°。在 z=1.5 mm、125 μm pitch 的原点直接转向判据下，只有 16/100 个中心 PASS、84 个 FAIL：

```text
FFFFFFFFFF
FFFFFFFFFF
FFFFFFFFFF
FFFPPPPFFF
FFFPPPPFFF
FFFPPPPFFF
FFFPPPPFFF
FFFFFFFFFF
FFFFFFFFFF
```
FFFFFFFFFF

P/F 图是按 theta_x=atan(|x|/z)、theta_y=atan(|y|/z) 和采样角上限逐中心计算的必要条件筛查，不代表完整双相位面系统的严格可达性证明；各架构的逐通道角度与 map 均保存在 JSON 中。

当前 A（100×100、200 μm aperture、125 μm pitch、z=1.5 mm）的零相位基线结果为 eta_total=6.471%、N_eff=4.000、−20 dB utilization=4%、corner/center power=9.62×10⁻¹¹、P10/P90=0.00201。相同架构做 24 步 total-coupling 优化后 eta_total 升至 55.645%（相对传播前光功率；输入到 fiber 为 38.725%），但 N_eff 仍为 4.000、边缘功率比仍约 2.05×10⁻¹⁰。Uniform 和 effective-dimension 目标在该点没有取得指标改善。由此可见，提高总耦合本身会进一步集中在中央少数通道，不能作为 100 维 OLS 可用性的代理。

在 z=1.5 mm、125 μm pitch 下，按 effective-dimension 目标的四个架构比较如下。P10/P90 越接近 1 越均匀；optimization difficulty 是每个目标 24 步的实测秒数。

| 架构 | aperture | eta_total | N_eff | utilization（−40/−30/−20/−10 dB） | edge retention | P10/P90 | 24 步时间 |
|---|---:|---:|---:|---|---:|---:|---:|
| A | 200 μm | 6.471% | 4.00 | 4/4/4/4% | 9.62×10⁻¹¹ | 0.00201 | 1.184 s |
| B | 400 μm | 4.329% | 15.99 | 16/16/16/16% | 2.05×10⁻⁸ | 5.43×10⁻¹⁰ | 1.211 s |
| C | 600 μm | 1.876% | 34.00 | 36/36/36/36% | 5.76×10⁻⁸ | 6.11×10⁻⁹ | 1.163 s |
| D | 800 μm | 4.989% | 36.41 | 60/60/36/36% | 1.36×10⁻⁸ | 6.69×10⁻¹⁰ | 1.170 s |

全扫描的工程筛选标准是 N_eff≥80、eta_total≥20%、edge retention≥1e-3；它仅用于筛选，不是物理验收门槛。共有 59 个 phase-only 结果同时达到三项，其中 125 μm 参考 pitch 为 0 个；59 个结果均依赖非参考 pitch：50 μm 为 27 个、62.5 μm 为 13 个、75 μm 为 12 个、100 μm 为 7 个。48 个结果位于记录的 Pareto frontier。

扫描中最小的筛选命中为 250 像素（500 μm）aperture、z=0.5 mm、50 μm pitch、maximize-effective-dimension：eta_total=21.891%，输入到 fiber 总效率=16.859%，N_eff=99.993，edge retention=0.98975，P10/P90=0.98376，CV=0.843%。在基准输入功率 249.3 μW 时，fiber 面功率约 191.986 μW，捕获到 100 个模式的总功率约 42.028 μW。该 50 μm pitch 小于本配置的 125 μm 包层直径，故这是敏感性仿真命中，不是可直接装配或加工的 fiber array 候选。

只对上述敏感性候选做了强度覆盖换算。下表每格按 min/p10/p50/mean/p90/max 排列，功率单位为 μW、峰值强度单位为 W/cm²；完整 10×10 P_i 与 I_i 图都在 JSON 中。

| 输入功率 | P_i 分布 min/p10/p50/mean/p90/max（μW） | I_i 分布 min/p10/p50/mean/p90/max（W/cm²） |
|---:|---|---|
| 0.2493 mW | 0.4113 / 0.4158 / 0.4217 / 0.4203 / 0.4227 / 0.4266 | 0.5178 / 0.5235 / 0.5309 / 0.5291 / 0.5321 / 0.5371 |
| 1 mW | 1.6497 / 1.6678 / 1.6916 / 1.6859 / 1.6954 / 1.7114 | 2.0769 / 2.0998 / 2.1297 / 2.1225 / 2.1344 / 2.1546 |
| 10 mW | 16.4965 / 16.6784 / 16.9159 / 16.8585 / 16.9537 / 17.1139 | 20.7688 / 20.9978 / 21.2968 / 21.2245 / 21.3444 / 21.5460 |
| 100 mW | 164.965 / 166.784 / 169.159 / 168.585 / 169.537 / 171.139 | 207.688 / 209.978 / 212.968 / 212.245 / 213.444 / 215.460 |
| 300 mW | 494.895 / 500.352 / 507.477 / 505.756 / 508.612 / 513.416 | 623.063 / 629.933 / 638.904 / 636.736 / 640.332 / 646.381 |

按 active 定义 0.1<I_i/I_sat<10，coverage 全部以该敏感性候选的 100 个通道计算：

| I_sat（W/cm²） | 0.2493 mW | 1 mW | 10 mW | 100 mW | 300 mW |
|---:|---|---|---|---|---|
| 10 | 100% linear | 100% active | 100% active | 100% saturated | 100% saturated |
| 100 | 100% linear | 100% linear | 100% active | 100% active | 100% active |
| 520 | 100% linear | 100% linear | 100% linear | 100% active | 100% active |
| 1,000 | 100% linear | 100% linear | 100% linear | 100% active | 100% active |
| 10,000 | 100% linear | 100% linear | 100% linear | 100% linear | 100% linear |
| 100,000 / 1,000,000 / 30,000,000 | 全部 100% linear | 全部 100% linear | 全部 100% linear | 全部 100% linear | 全部 100% linear |

这里的强度公式为 2P_i/[pi×(MFD/2)^2]。这是使用未测 MFD、未校准输入功率的物理敏感性计算，不构成材料或器件结论。

## 科学认识

**Q1：200 μm 方案是否物理可扩展？** 在本次有限步、当前传播器和给定 illumination 下，200 μm aperture 没有在任何扫描结果中同时满足三项工程筛选标准。对 125 μm 参考 pitch，200 μm aperture 最佳 N_eff=33.72（z=15 mm），但 eta_total=3.72%；最大 eta_total=55.40% 时 N_eff=5.14。将 pitch 放宽到 50 μm 后，200 μm aperture 的最好 N_eff=63.18（z=10 mm），eta_total=12.33%、edge retention=0.0169，仍未覆盖 100 维筛选目标。优化到高总耦合可明显改善吞吐，但功率仍集中；当前失败不是单一的“相位还没调好”。

**Q2：达到 100 维 latent 的最小接口是什么？** 在本次采样点和筛选条件下，最小仿真命中是 500 μm aperture、0.5 mm 传播距离、50 μm pitch。它依赖包层相互重叠的敏感性假设。对 125 μm 参考 pitch，本轮没有任何结果同时通过三项筛选；1 mm aperture、15 mm 距离可以达到 N_eff=99.08、edge retention=0.994，但 eta_total 只有 6.36%。因此未建立适用于当前 fiber 几何的最小 100 维接口，也没有可直接加工的推荐尺寸。

**Q3：非线性材料应基于哪个 architecture 评估？** 本轮仅能把 500 μm/0.5 mm/50 μm 作为后续仿真敏感性 profile，不能据此指定真实 Te/TaP/CoSi 器件的工作点。下一次材料研究必须先使用实测 fiber MFD、pitch、传输与真实可装配几何，再按实测通道功率评估覆盖。

**Q4：最终实验加工尺寸是否合理？** 尚不能判断。实测 relay magnification、fiber pitch/MFD/包层、照明尺寸与损耗均缺失；敏感性候选 pitch 还与当前 125 μm 包层不兼容。现有结果只能支持仿真中的结构趋势，不能支持实验制造或部署结论。

关于失败来源，本轮证据更支持“紧凑有限 aperture 与 fiber 阵列跨度/模式重叠共同造成通道效率和均匀性的限制”，并显示距离、pitch 与 aperture 有明显交互；无法将影响唯一归因于其中一个因素。采样相位角判据能筛出几何压力，但不是双层有限孔径系统的充分条件：例如 500 μm/0.5 mm/50 μm 候选原点转向图仅 4/100 PASS，而完整传播与相位优化却得到 N_eff≈99.993。这种差异说明不能把 kinematic FAIL 解释成物理绝对不可达。反过来，125 μm pitch、500 μm aperture、15 mm 的样本显示高 N_eff 仍可伴随低捕获率。

数值上，预检的短距离解析平面波误差为 3.71×10⁻⁷，错误距离负控误差为 0.2357；实际 1.5 mm 长距离平面波对照的 FP32 相位误差约 1.953×10⁻³。该误差作为当前传播器的数值限制如实保留，没有用放宽容差掩盖。长距离扫描的精细趋势应在更高精度独立传播参考下复核。

## 目标缺口

- 尚未测量当前设备的实际 fiber pitch、MFD、包层排列、encoder/relay 几何、照明尺寸、层透射与总光功率。
- phase-only 每个目标仅从零相位初始化并运行 24 步；它给出有限步实测结果，不是全局最优或严格耦合上界。高精度传播器对照、更多初始化与收敛核验仍缺失。
- 对 125 μm 参考几何没有通过全筛选的点；对敏感性 pitch 的命中还缺少与不重叠纤芯/包层及真实装配相符的光纤规格。
- 本轮没有检验语义生成、训练收益、任务 evaluator 指标、器件材料非线性或实验可制造性。

## 工程完成与验证

诊断配置为 configs/architecture_sweep/compact_ols_architecture_scaling_v1.yaml，测量入口为 scripts/research/lightgen/diagnose_compact_ols_architecture_scaling_v1.py，聚焦测试为 tests/research/lightgen/test_compact_ols_architecture_scaling_v1.py。云端预检 PASS，完整命令以 exit 0 结束，结果状态为 VALID_COMPLETED；资源 telemetry 有 18 条记录、0 次采集失败。主证据为 artifacts/compact_ols_scaling_results.json；云端回传、运行状态与原文件恢复证据保存在 outputs/20261013_compact_ols_architecture_scaling_v1/runs/ 下的 attempt_05 和 autodl_backup_compact_ols_scaling_attempt_05。

本轮验证覆盖配置驱动参数、生产传播与 LP01 port 基线一致性、解析传播检查、错误距离负控、相位梯度有限性、完整结果 JSON 和 progress 架构键一一对应；聚焦测试 5 项通过，Ruff E4/E9/F821、compileall 与 git diff --check 通过。云端临时部署的依赖源码已按备份恢复；本地既有工作区改动和历史输出均未清理或覆盖。

## 决策影响与下一步

建议先测量真实 relay、illumination 和 fiber 几何，再单独设计高精度、无训练的 phase-only validation；这会决定是扩大 aperture、采用 relay，还是重新定义 fiber pitch/通道布局。可讨论的架构替代项及其代价为：

| 备选项 | 所需改变与预期优势 | 主要成本 | 是否保留 100×100、2 μm、2+2 |
|---|---|---|---|
| 增大 trainable aperture | 扫描到 500 μm–1 mm 能提高参考 pitch 下的 latent 覆盖潜力 | 需要更大 mask/照明孔径和计算面；仍须解决捕获效率 | 否，保持 2 μm 时需增加像素数 |
| 增大 encoder-to-fiber 距离 | 可降低所需转向角；参考 pitch 的 1 mm/15 mm 样本达到 N_eff≈99 | 约 15 mm 光路失去紧凑性，且样本 eta_total 仅 6.36% | 可保留两层与 100×100，但不保留 compact spacing |
| 增加 magnifying relay | 约 6.25× 可把 200 μm 尺度映射到约 1.25 mm 阵列外廓 | 增加 relay、孔径匹配、损耗、像差、对准与标定 | encoder 可保留，系统不再是无 relay 的紧凑 2+2 |
| 更小 pitch / 替代 fiber geometry | 减小阵列跨度；本次 50–100 μm 敏感性点出现筛选命中 | 必须更换并实测 fiber core/cladding/MFD；50 μm 与当前 125 μm 包层冲突 | 未证明可保留当前 200 μm 接口与 100 通道 |
| 减少 OLS 通道数 | 缩小有效阵列，降低空间覆盖要求 | 直接降低 latent channel 数、改变目标 | 可保留 mask 像素，但不保留 100 通道目标 |
| 专用 fan-out DOE | 增加定制分束/扇出自由度，可能绕开紧凑 mask 的覆盖瓶颈 | 增加元件、损耗、对准和校准；需另行建模 | 可保留原 mask，但增加光学层，改变当前 2+2 系统 |

若必须给出本轮的 Architecture_candidate_A，它只指 500 μm aperture / 0.5 mm / 50 μm pitch 的仿真敏感性命中，不是实验候选或制造建议。本轮允许的后续动作是查看 Pareto 候选、测量物理 relay/illumination/fiber 几何、另行规划高保真验证。正式训练、数据集读取、evaluator 调用、loss 改动、材料器件结论、生产几何修改与加工验收仍被禁止；本轮完成后停止。
