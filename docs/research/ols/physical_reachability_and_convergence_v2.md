# LightGen 物理可达性与收敛诊断 v2

日期：2026-10-13。对象是当前 532 nm LightGen 2+2 研究前向。所有测量都使用 CPU 合成场；没有读取训练数据、加载 checkpoint、调用任务评价器或执行生成任务训练。双层 phase-only 搜索是零更新光学诊断，不是任务训练。相位优化只报告有限步局部达到值，不声称是全局最优上界。fiber pitch、cladding、MFD 和器件间距仍是项目参考值，非当前器件实测。

机器结果：[`physical_reachability_and_convergence_v2.json`](../../../artifacts/physical_reachability_and_convergence_v2.json)。原始完整运行：[`attempt_06`](../../../outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_06/physical_reachability_and_convergence_v2.json)。修正后的 clipped camera-gain 预检：[`attempt_05`](../../../outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_05/camera_gain_preflight.json)。

## 结果与状态：

本轮有效完成，整体为 **VALID_FAILED**。R1 的逐轴 sampled-phase 可达性只有 16/100 根通过；C2 去除全局相位后仍超过 `1e-4`，所以 R5/G3 失败。G9a、G9b、G9c 分别通过 `1e-4` 单标量或 identity 等价标准。R2–R4 的逐端口及阵列分布测量已生成；R9 报告固定 camera gain 的测量残差，不设置请求中未给出的相机精度门槛。

上一轮原始 G1–G10 记录完整保留，未覆盖：G1/G2/G4/G5/G6/G7/G8/G10 PASS，G3/G9 FAIL；新的 G9a/b/c 是分步定义，不改写旧的 G9 结果。

架构分类为 **ARCHITECTURE_MARGINAL**，限于当前项目参考几何与仿真前向：采样相位判据只覆盖内侧 4×4；均匀目标双层优化后约 `99.999984%` 的被捕获功率仍在中心四根 fiber，edge/center 均值耦合比约 `1.67×10⁻¹⁷`。这不是外侧场严格为零的证明，也不是硬件结论。没有人为添加耦合率 PASS 百分比。

## 科学认识：

### C1：fiber 阵列的采样可达性与实际耦合

当前光路模型参数为：`λ=532 nm`、`dx=2 μm`、可训练 encoder 为 `100×100`（物理口径 `200×200 μm`）；encoder mask 1→mask 2 为 `1.5 mm`，mask 2→fiber reference plane 再传播 `1.5 mm`。fiber pitch 为 `125 μm`，cladding 参考直径为 `125 μm`。10×10 偶数阵列没有位于原点的 fiber：每轴中心为 `−562.5, −437.5, −312.5, −187.5, −62.5, +62.5, +187.5, +312.5, +437.5, +562.5 μm`。最外中心为 `|x|=|y|=562.5 μm`；含 cladding 的阵列外宽为 `1.25 mm`（半宽 `0.625 mm`）。这些中心被放在当前 `4.096 mm` global plane 上，没有 resize 或移动 fiber。

Nyquist phase-steering 上限为 `sin(θmax)=λ/(2dx)=0.133`、`θmax=7.642985°`。以最后一片 encoder 到 fiber 的 `1.5 mm` 为 `z`，外侧可达中心坐标约 `201.3 μm`；因此 `±62.5` 和 `±187.5 μm` 每轴通过，`±312.5 μm` 及更外侧失败。此判断仅测试 sampled phase 空间频率，不等同于高耦合保证。

逐纤维 `PASS/FAIL` map（列为 x，行为 y，P=PASS，F=FAIL）：

| y\\x (μm) | -562.5 | -437.5 | -312.5 | -187.5 | -62.5 | +62.5 | +187.5 | +312.5 | +437.5 | +562.5 |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -562.5 | F | F | F | F | F | F | F | F | F | F |
| -437.5 | F | F | F | F | F | F | F | F | F | F |
| -312.5 | F | F | F | F | F | F | F | F | F | F |
| -187.5 | F | F | F | P | P | P | P | F | F | F |
| -62.5 | F | F | F | P | P | P | P | F | F | F |
| +62.5 | F | F | F | P | P | P | P | F | F | F |
| +187.5 | F | F | F | P | P | P | P | F | F | F |
| +312.5 | F | F | F | F | F | F | F | F | F | F |
| +437.5 | F | F | F | F | F | F | F | F | F | F |
| +562.5 | F | F | F | F | F | F | F | F | F | F |

C1.2 的 phase ramp 放在第二片 100×100 encoder mask 上，随后走真实两段 `1.5 mm` angular-spectrum propagation 和 LP01 complex-overlap producer；越过 Nyquist 的分量被裁到 Nyquist 边界并明确标记，不能把它当成准确的外侧 steering。下表是单 fiber 的初始解析 ramp 与 8 步双层 phase-only 局部优化结果；target-to-crosstalk 是目标 eta 除以其余 99 根 eta 总和。中心选最靠近原点的 row 4/column 4（`−62.5,−62.5 μm`），不是不存在的正中心端口；对称的四个边中点与四角结果相同到表中精度。

| 目标 | 初始 target η | 优化后 target η | 优化后总 η | target/其余99根 | ramp 是否 Nyquist 裁剪 |
|---|---:|---:|---:|---:|---|
| 最近中心 fiber | `8.329983×10⁻⁷` | `6.452104×10⁻³` | `6.483857×10⁻³` | `2.031962×10²` | 否 |
| 四个 edge-center（每根） | `4.4345×10⁻¹³` | `4.4345×10⁻¹³` | `3.3741×10⁻⁸` | `1.3143×10⁻⁵` | 是 |
| 四个 corner（每根） | `7.7556×10⁻¹⁵` | `7.7556×10⁻¹⁵` | `1.6770×10⁻⁸` | `4.6248×10⁻⁷` | 是 |

逐根 all-100 analytic phase-ramp target η map（`ηtarget=Ptarget/Pbefore_fiber`；这是单 mask ramp 的实测图，双层优化锚点另列于机器 JSON）：

| y\\x (μm) | -562.5 | -437.5 | -312.5 | -187.5 | -62.5 | +62.5 | +187.5 | +312.5 | +437.5 | +562.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -562.5 | 7.8e-15 | 2.5e-14 | 5.1e-13 | 2.7e-12 | 4.4e-13 | 4.4e-13 | 2.7e-12 | 5.1e-13 | 2.5e-14 | 7.8e-15 |
| -437.5 | 2.5e-14 | 8.1e-14 | 1.7e-12 | 1.0e-11 | 6.9e-13 | 6.9e-13 | 1.0e-11 | 1.7e-12 | 8.1e-14 | 2.5e-14 |
| -312.5 | 5.1e-13 | 1.7e-12 | 3.2e-11 | 1.4e-10 | 7.9e-11 | 7.9e-11 | 1.4e-10 | 3.2e-11 | 1.7e-12 | 5.1e-13 |
| -187.5 | 2.7e-12 | 1.0e-11 | 1.4e-10 | 1.6e-9 | 4.3e-9 | 4.3e-9 | 1.6e-9 | 1.4e-10 | 1.0e-11 | 2.7e-12 |
| -62.5 | 4.4e-13 | 6.9e-13 | 7.9e-11 | 4.3e-9 | 8.3e-7 | 8.3e-7 | 4.3e-9 | 7.9e-11 | 6.9e-13 | 4.4e-13 |
| +62.5 | 4.4e-13 | 6.9e-13 | 7.9e-11 | 4.3e-9 | 8.3e-7 | 8.3e-7 | 4.3e-9 | 7.9e-11 | 6.9e-13 | 4.4e-13 |
| +187.5 | 2.7e-12 | 1.0e-11 | 1.4e-10 | 1.6e-9 | 4.3e-9 | 4.3e-9 | 1.6e-9 | 1.4e-10 | 1.0e-11 | 2.7e-12 |
| +312.5 | 5.1e-13 | 1.7e-12 | 3.2e-11 | 1.4e-10 | 7.9e-11 | 7.9e-11 | 1.4e-10 | 3.2e-11 | 1.7e-12 | 5.1e-13 |
| +437.5 | 2.5e-14 | 8.1e-14 | 1.7e-12 | 1.0e-11 | 6.9e-13 | 6.9e-13 | 1.0e-11 | 1.7e-12 | 8.1e-14 | 2.5e-14 |
| +562.5 | 7.8e-15 | 2.5e-14 | 5.1e-13 | 2.7e-12 | 4.4e-13 | 4.4e-13 | 2.7e-12 | 5.1e-13 | 2.5e-14 | 7.8e-15 |

C1.3 的双层 encoder phase-only 分别以零相位开始，优化 uniform-100、center-weighted、random-nonuniform 三种目标各 12 步。shape objective 在优化内部比较 captured-port distribution；真实场、`P_i`、eta 和报告输出均未归一化。结果是局部优化达到值，不是全局上界：

| 目标 | 总耦合 η | η 的 CV | uniform 目标 L2 初始→优化后 | edge/center 均值 η 比 | corner/center 均值 η 比 |
|---|---:|---:|---:|---:|---:|
| A uniform 100-channel | `0.06464969` | `4.89898` | `0.48989792 → 0.48989788` | `1.6729e-17` | `8.5671e-24` |
| B center-weighted | `0.06465189` | `4.89898` | `0.44081281 → 0.44081278` | `1.68e-17` | `8.54e-24` |
| C random nonuniform | `0.06471423` | `4.89898` | `0.48893262 → 0.48893262` | `2.00e-17` | `7.47e-24` |

uniform-target 输出 eta 的 min/p10/p50/mean/p90/max 为 `1.38e-25 / 1.02e-24 / 1.50e-19 / 6.46e-4 / 1.29e-9 / 1.62e-2`。统计分布高度偏斜；mean 不是用来替代 100 个实测通道值，完整 `η_i`、`P_i` 和 complex overlap 均在 JSON 中。C1.4 对旧 `6.47%` 的主分类为 **TYPE-B（当前合成输入几乎只把功率送进中心四根）**；其外侧失败同时符合 TYPE-C 的 Nyquist steering 判据。外侧 overlap 是非零小数，不能说成数学上的零或光学场绝对不可达。现有数据不支持 TYPE-A“只是未优化”，也没有将其归因于单一 TYPE-D 衍射因素。

C1.5 只列备选方案，没有选择或实施：

| 备选 | 所需变化与预期优势 | 代价 | 保留100×100 / 2 μm / compact 2+2 |
|---|---|---|---|
| 增大 trainable aperture | 在 dx 不变时增加可训练 cell 数，提高空间自由度和角聚焦分辨；单纯增口径不会提升 Nyquist steering 角 | 面积、驱动单元、校准和封装增大 | 否 / 是 / 否 |
| 增大 encoder-to-fiber 距离 | 外侧中心满足当前逐轴准则需末端距离至少约 `4.18 mm`（现为1.5mm） | 更长光路、占地、振动与新收敛验证 | 是 / 是 / 否 |
| 插入放大 relay | 200 μm mask 到 1.25 mm 阵列外宽的一阶倍率约 `6.25×`，须按共轭面重算 | 新光学件、像差/损耗/对准和标定 | 是 / 是 / 否 |
| 降低 fiber pitch 或替换 fiber geometry | 当前距离下按中心跨度估算需 pitch 不高于约 `44.7 μm` | 与125 μm参考cladding冲突，需新fiber及新模式模型 | 是 / 是 / 是（紧凑概念可保留） |
| 减少 OLS channel 数 | 当前参考布局直接 steering 判据最多为中心4×4=16通道 | 降低通道数并改变100-port合同 | 是 / 是 / 是 |
| 增加专用 fan-out diffractive optic | 增加专门将光分到10×10中心的衍射元件 | 插损、量化、串扰、对准及新增物理验证 | 是 / 是 / 否 |

### C2：G3 phase-aligned complex-field convergence

比较固定 `dx=2 μm` 的 `2048²/3072²/4096²`（窗口 `4.096/6.144/8.192 mm`），把大窗口场裁到和 reference 相同的中心 `4.096 mm` ROI。相位按 `arg(sum(conj(Eref)*Etest))` 只对每对整场去一个共同相位；per-fiber latent 也只去一个共同相位。没有放宽 `1e-4`。

| 窗口对 | 全局相位(rad) | raw complex 相对误差 | phase-aligned complex | intensity 相对误差 | common-ROI 功率相对误差 | Pbefore-fiber 相对误差 | 总 eta 绝对差 | per-fiber power L2 | per-fiber latent raw/aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 vs 3072 | `8.1522e-5` | `6.6714e-4` | `6.6214e-4` | `4.9895e-5` | `2.7724e-6` | `2.2851e-6` | `6.3542e-7` | `7.5758e-6` | `1.7472e-4 / 1.6538e-4` |
| 2048 vs 4096 | `-1.5683e-6` | `5.2873e-4` | `5.2873e-4` | `3.4455e-5` | `5.8914e-6` | `5.5945e-6` | `3.2262e-7` | `6.3567e-7` | `1.2440e-4 / 1.2140e-4` |
| 3072 vs 4096 | `-8.3090e-5` | `4.8496e-4` | `4.7779e-4` | `3.2092e-5` | `3.1190e-6` | `3.3095e-6` | `3.1280e-7` | `8.2038e-6` | `1.2324e-4 / 9.0647e-5` |

最大 phase-aligned global-ROI complex error 为 `6.6214e-4`，仍高于 `1e-4`；**G3/R5 FAIL**。同一 2048 设置对同一 synthetic uniform 输入再独立执行的 raw complex error 为0，排弱了同一设置下的随机/非确定性累积差异。40 μm 外沿功率比例分别为 `3.53e-8 / 1.81e-8 / 7.96e-9`，不支持把主要差异简单归因于明显的窗口边界 wrap-around，但不能完全排除有限周期边界影响。三个中心 crop 的物理坐标、dx 和 ROI 相同，crop mismatch 不成立。轴向 Nyquist 空间频率为 `250,000 cycles/m`，采样最大径向 `λf≈0.188<1`，因此离散网格上没有 evanescent bins。窗口增大改变 FFT 频率间隔（`244.14/162.76/122.07 cycles/m`）；当前证据将跨窗口频率采样/有限窗口离散作为候选来源，不能唯一归因。全局相位很小且对齐后仍未过门槛；intensity、积分功率、总耦合相对稳定不能代替复场收敛。接近零的 per-fiber 功率通道会使逐通道相对误差很大，机器 JSON 同时保存绝对误差向量。

### C3：stepwise G9 与双输出 camera interface

| 检查 | 结果 | 含义 |
|---|---|---|
| G9a A0 vs A1 | `α=6.2324987e-9`；残差 `5.7393e-7`，PASS | A1 仅加绝对单位标定，和 A0 保持 compact/local fiber geometry；整批合成像素用一个 α |
| G9b A1 vs A2 | `α=0.6246073`；残差 `5.7418e-7`，PASS | A2 只加所有2个 encoder和2个 decoder层的统一 `T=0.889` |
| A2 vs A3 | scale-adjusted残差 `0.9434`；raw相对差 `5.2365` | 作为 `physical_model_delta` 报告，不要求数值等价；A3 改为global fiber中心/几何 |
| G9c A3 vs A4(identity) | 相对差 `0`，PASS | identity material 保持严格等价，门槛仍 `1e-4` |

旧的 G9 失败结果留在 [`physical_power_integration_v1`](physical_power_integration_v1.md) 与其 JSON，没有覆盖或删除。相机正式物理输出保留 `physical_sensor_output_w_per_pixel = raw W/m² × pixel area`；当前相机像素 pitch 是 `2.5 μm`，pixel area `6.25e-12 m²`。新增任务口为 `task_sensor_output=clip(camera_gain×physical_W_per_pixel,0,1)`。配置冻结值 `camera_gain_proxy=1.6044928876×10⁸ W⁻¹`，来源是 A0/A1 同几何合成校准子集，不由 evaluator 调节，也不逐样本重估。它只代表 simulation calibration parameter。

首次误把 A0/A3 用于 gain 拟合，因同时改变单位和阵列几何，残差 `0.9366`，保留但弃用；A0/A1 未裁剪的标量残差 `5.71e-7` 也保留但不用于设置最终 gain，因为实际接口会 clip。按真实 clip 映射重新拟合后，两个 calibration patterns 的整体残差为 `0.30987`，校准子集裁剪像素比例 `7.505%`。validation 没有 refit：A1 同几何下 diagonal shape/scale residual=`5.42e-7/2.63e-7`，center-disk=`0.15088/0.04491`（裁剪2.73%）；A3 global geometry下两图的 shape residual=`1.0545/0.5347`，scale residual=`0.9317/0.9988`，包含 A2→A3 的物理模型差异。shape residual 的按图总和归一化只用于离线比较形状，传感器输出本身没有 max/L2/样本归一化，也没有 fiber 后功率归一化。请求没有给相机精度阈值，因此 R9 记为 `MEASURED_NO_ACCEPTANCE_TOLERANCE`，不声称通过精度验收。

### C4：绝对功率和非线性覆盖

本表使用 C1 uniform-target finite-step encoder phase 和本轮实测的 per-channel overlap；没有使用旧的50%耦合假设，没有以 `mean/100` 替代通道功率。fiber 参数仍是 `MFD=14.222 μm`、衰减 `0 dB/km` 的未测 reference。每组 `P_i` 与 `I_i` 列按 `min/p10/p50/mean/p90/max` 排列，单位分别为 W 和 W/cm²。

| Pinput | Pbefore-fiber (W) | Pfiber total (W) | ηtotal | `P_i` 六统计量 (W) | `I_i` 六统计量 (W/cm²) |
|---:|---:|---:|---:|---|---|
| 249.3 μW | `1.84676e-4` | `1.19392e-5` | `0.0646497` | `2.545e-29 / 1.888e-28 / 2.763e-23 / 1.194e-7 / 2.376e-13 / 2.986e-6` | `3.204e-23 / 2.376e-22 / 3.479e-17 / 1.503e-1 / 2.992e-7 / 3.759` |
| 1 mW | `7.40776e-4` | `4.78910e-5` | `0.0646497` | `1.021e-28 / 7.572e-28 / 1.109e-22 / 4.789e-7 / 9.532e-13 / 1.198e-5` | `1.285e-22 / 9.533e-22 / 1.396e-16 / 6.029e-1 / 1.200e-6 / 15.079` |
| 10 mW | `7.40776e-3` | `4.78910e-4` | `0.0646497` | `1.021e-27 / 7.572e-27 / 1.109e-21 / 4.789e-6 / 9.532e-12 / 1.198e-4` | `1.285e-21 / 9.533e-21 / 1.396e-15 / 6.029 / 1.200e-5 / 150.787` |
| 100 mW | `7.40776e-2` | `4.78910e-3` | `0.0646497` | `1.021e-26 / 7.572e-26 / 1.109e-20 / 4.789e-5 / 9.532e-11 / 1.198e-3` | `1.285e-20 / 9.533e-20 / 1.396e-14 / 60.294 / 1.200e-4 / 1.508e3` |
| 300 mW | `2.22233e-1` | `1.43673e-2` | `0.0646497` | `3.062e-26 / 2.272e-25 / 3.326e-20 / 1.437e-4 / 2.860e-10 / 3.593e-3` | `3.855e-20 / 2.860e-19 / 4.187e-14 / 180.881 / 3.600e-4 / 4.524e3` |

下面每格为 `linear/active/saturated` 比例，active 使用既有 sensitivity 定义 `0.1≤I/I_sat≤10`。完整45组逐通道比值在 JSON；没有器件接受阈值。

| Pinput | Isat 10 | 100 | 520 | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 | 3e7 W/cm² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 249.3 μW | 96/4/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% |
| 1 mW | 96/4/0% | 96/4/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% |
| 10 mW | 96/0/4% | 96/4/0% | 96/4/0% | 96/4/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% |
| 100 mW | 96/0/4% | 96/0/4% | 96/4/0% | 96/4/0% | 96/4/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% |
| 300 mW | 96/0/4% | 96/0/4% | 96/4/0% | 96/4/0% | 96/4/0% | 100/0/0% | 100/0/0% | 100/0/0% | 100/0/0% |

这只是用通道实际分布做的物理敏感性扫值，不说明 Te/TaP/CoSi 当前器件在 532 nm 的响应。

## 目标缺口：

- 外侧 fiber pitch/MFD、最后一片到端面的距离和实际 relay/端面坐标均未测；当前125 μm、14.222 μm、1.5 mm是 reference/project geometry。
- G3 的 common-ROI 复场误差尚未达到 `1e-4`。频率采样变化是候选原因而非已证实根因，继续保持 fail-closed。
- C1 phase optimization 是固定初始条件、有限步数的局部诊断，不能建立全局可达上界、训练后可达性或硬件光斑验收。
- 没有真实 CMOS 响应、曝光、量子效率和绝对标定；gain proxy 的 clipped calibration/validation 残差也显示不可直接采用为实验参数。
- 本轮不含任务图像质量、语义表现、独立泛化、材料动态或实物器件证据。

## 工程完成与验证：

- 新增合成诊断入口 [`diagnose_physical_reachability_and_convergence_v2.py`](../../../scripts/research/lightgen/diagnose_physical_reachability_and_convergence_v2.py) 与固定 gain 配置 [`physical_reachability_convergence_v2.yaml`](../../../configs/research/lightgen/physical_reachability_convergence_v2.yaml)。传播使用生产 angular-spectrum encoder 和 LP01 overlap；没有缩小场或移动 fiber。C1 的9个单通道锚点共执行72个 phase-only Adam steps，三种 full-array 目标共36步；合计108个诊断相位更新，任务训练更新仍为0。
- LightGen physical forward 保留 `physical_sensor_output_w_per_pixel`，并仅在显式配置 scalar gain 时提供 `task_sensor_output`。这条接口目前只被合成诊断和边界测试消费；未接入任务训练器、loss 或 evaluator。
- 新增回归测试覆盖 W/pixel 的物理像素面积换算、固定 gain clip 映射以及未配置 gain 时不生成 task 输出。本轮和既有 LightGen 前向测试一起运行：`10 passed`。脚本语法检查和 focused Ruff（`E4,E7,E9,F`）通过；`git diff --check` 通过。未跑全仓 suite。
- 输出保留在 `outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_01...06`；attempt_01 与 attempt_02 的 gain 拟合口径存在混杂/未按clip拟合，已明确标为探索并保留；最终值来自 attempt_05 clipped fit 与 attempt_06 full run。没有覆盖上一轮 `physical_power_integration_v1`。
- 当前物理模块、配置、报告与 artifact 都未改变 fiber pitch、光路距离、loss、正式训练、数据或 evaluator。停止条件已到：输出 `VALID_FAILED` 后没有继续执行后继任务。

## 决策影响与下一步：

结果支持把当前参考仿真判为 **ARCHITECTURE_MARGINAL**，而不是“6.47%只因没优化”：优化后的 uniform/center/random 目标仍集中在中心四根；逐轴 Nyquist 判据只有16/100通过。外侧 field overlap 仍是非零值，有限步局部优化不是不可达性证明。G3 仍是正式数值阻断，G9a/b/c 工程定义已拆开并通过；R9 只记录固定 gain 与裁剪残差，不是相机验收。

允许的后续建议是先取得 fiber pitch/MFD、真实层间距离和 relay/端面坐标，再决定是否另行设计 reachability 复核；针对 G3 继续以共同物理 ROI 和 `1e-4` 门槛定位频率离散来源。若要改变架构，可比较上列六种备选方案，但本报告没有替用户选择或实施任何方案。

当前授权到此停止。正式训练、任务 evaluator、训练数据读取、loss 修改、光路几何/架构修改、材料器件结论和硬件部署均为 blocked；本报告不构成这些动作的授权。

### 本地状态：

分支 `research/lightgen-ols-native-semantic-generation-v1`，HEAD `b9d865e`；沿用已有 dirty checkout，未创建分支、commit 或清理其它更改。本轮交付集中于物理 camera 输出接口、其测试、C1–C4 诊断脚本/配置、指定报告与 artifact，以及 diagnostic-only attempt 目录。
