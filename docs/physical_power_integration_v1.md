# LightGen 物理功率与非线性前向集成记录

日期：2026-10-13。输入为上一轮 `physical_power_audit` 和 `configs/reference/lightgen_power_proxy.yaml`。范围限于小型 2+2 LightGen 研究前向；正式 v2 条件光路、损失函数、评价器、数据集和任务定义均未改。

## 结果与状态：

状态为**有效未通过**。A1–A4 的可配置研究前向已实现，零更新物理诊断有效；G1、G2、G4–G8、G10 通过，G3 和 G9 未通过。依照本轮计划，条件性 A0–A4 训练 smoke **未运行**，训练更新、评价器调用、数据集读取均为 0；没有正式长训练。

执行顺序说明：一次组合式诊断在汇总 G3 之前就运行了 generic SA 扫描和 identity/A0 对照；这不符合“G1–G8 全部通过后再做 A4.2”的顺序要求。因此这些 SA/G9 数值只保留为 attempt_02 的探索性结果，不作为顺序合规的 A4.2 接受证据，也没有据此训练。G3 是正式停止原因；后续 gate 不会获得训练许可。

G3 的功率积分和 fiber efficiency 随窗口扩大变化很小，但复数场及 decoder 输入场相对变化超过预先设定的 `1e-4`。G9 中 A4 identity 精确复现 A3 线性物理链；A3 与原 A0 的相机数据却没有共同的物理量纲或相同 OLS 几何，因而要求的 A0 数值等价没有成立。现有训练消费者把 `raw_camera` 与无量纲 `SLM intensity` 直接作 MSE，尚无 W/m² 到目标/探测器响应的标定。

## 科学认识：

大白话说，10×10 阵列现在确实被放到了 4.096 mm 的真实数值传播平面里，100 个中心坐标也都落在窗口内；但“坐标放得下”不等于“光照到了整个阵列”。四个合成控制的 `r99` 只有 0.107–0.143 mm，远小于阵列中心到边缘的 0.5625 mm，也小于含包层的 0.625 mm 半宽。均匀输入下总 fiber coupling efficiency 约 6.47%，通道功率高度不均匀。该结果只属于零相位、合成控制和代理参数场景，不能外推到训练 checkpoint 或实物器件。

本次配置中，2048×2048、2 μm 采样覆盖 4.096 mm；数组中心跨度 1.125 mm，含 125 μm 参考包层的外宽 1.25 mm，四边余量各 1.423 mm。请求的 array pitch 与 cladding diameter 是两个独立字段，即使此参考场景数值相同也未绑定。100×100、2 μm 的可训练 mask 仍为 200 μm，没有放大。

阵列外方形范围内的场功率比例是 0.99999964–0.99999970，范围外约 `3.0–3.6×10⁻⁷`。这个指标只表示数值光没有越过阵列外边界；它不表示外围 fiber 有足够输入。四个控制的 `r90/r95/r99` 分别为：均匀输入 `0.1058/0.1142/0.1358 mm`，竖条 `0.0870/0.0958/0.1254 mm`，对角线 `0.1135/0.1218/0.1429 mm`，中心圆 `0.0611/0.0731/0.1074 mm`。因此外圈通道仍需用实际照明及器件几何验证，不能靠窗口边界指标替代。

窗口扩大时，2048 与 4096 的 `P_before_fiber` 相对变化为 `5.59×10⁻⁶`，每通道 eta 向量相对 L2 变化 `4.98×10⁻⁶`，总 eta 相对变化 `4.95×10⁻⁶`；但 fiber-plane 复场 L2 为 `5.29×10⁻⁴`、generator 输入复场 L2 为 `1.24×10⁻⁴`。相对变化不证明存在某一种特定周期卷回机理，但复场未达到本次预先设置的稳定容差，G3 因此失败。

## 目标缺口：

- G3 需要达到窗口扩展后的复场稳定标准。当前差异很小且功率/耦合接近稳定，但不能在观察结果后放宽 `1e-4` 容差，也不能把它写成已经通过。
- G9 需要定义 A0 和 A1–A4 共用的相机输出量纲与比较边界。当前 A0 没有绝对功率标定、使用旧的局部 200 μm coupling plane；A3/A4 使用 W 标定和 125 μm array。现有 `L_rec=(raw-target)^2` 将物理相机强度与无量纲图像直接相减；尚无曝光、探测器响应或等效目标转换依据。为遵守“不改 loss/评价器/任务定义”，本轮不补造换算常数。
- reference config 的 `P_input=249.3 μW`、每层 `T=0.889`、125 μm pitch、125 μm cladding、14.222 μm MFD、8.5 mm 长度和零 dB/km 衰减均不是当前项目器件的实测值。MFD 沿用既有 LP01 模型；attenuation=0 是可配置的零损耗参考场景。
- Fiber 输出到 decoder 的模式面仍使用既有紧凑 100×100 输出基底，没有已测 relay/放大率/端面映射。材料处理是基于 MFD 的 lumped Gaussian 峰值近似，不是膜面 `I(x,y)` 空间积分；没有薄膜的 532 nm `T(I)`、相位、响应时间、脉冲条件或损伤阈值测量。
- 本次没有加载训练 checkpoint，也没有读取 MNIST 或运行 evaluator。4 个控制是统一合成图案，不构成可读性、泛化、任务质量或实物部署证据。

## 工程完成与验证：

### A1–A4 的实现前后

| 项 | 修改前 | 修改后与文件 |
|---|---|---|
| A1：global fiber reference plane | `FiniteCropPhaseStack` 每层传播后都裁回 100×100；SMF 只在 200 μm 局部 plane 上计算。 | 在 [`d2nn.py`](../src/ldm_project/v2/optics/d2nn.py) 增加默认关闭的全局末层输出；第一层仍保留 100×100 有限 mask/crop。[`fiber_operator.py`](../src/ldm_project/v2/optics/fiber_operator.py) 复用现有 LP01 径向模型和复数面积重叠，在显式 pitch 的 global plane 上投影；输入与紧凑输出 plane 分开。[`lightgen_2d_semantic_generation_input_ols_alignment.py`](../src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py) 为物理选项连通全局 OLS plane 并输出 100 个 row-major 中心坐标。mask 仍 `(2,1,100,100)`、dx=2 μm。 |
| A2：absolute power | 输入场只有相对振幅。 | 入口在振幅 SLM 前一次按 `P_input_W=Σ|U|² dxdy` 标定；图像 SLM 命令之后仅乘入射场，不在 encoder/fiber/material/decoder 前重归一。功率 trace 记录输入、SLM、各层、fiber 和 CMOS。 |
| A3：器件 transmission | 相位调制幅度为 1，未记录器件功率 transmission。 | [`FiniteCropPhaseStack`](../src/ldm_project/v2/optics/d2nn.py) 以 `√T_layer·exp(iφ)` 乘场，并分别记录层输入、器件后、传播后、孔径/crop 后功率。reference YAML 设 `T=0.889`，明确非本设备实测。 |
| A4：complex SMF、intensity 和 material | overlap 有 100 个复系数，但没有全局阵列坐标、绝对 W、fiber attenuation 与材料消费者。 | 复用 `FiberArrayOLS` 的复数 overlap 和系数传输；输出 `a_i`、`P_i=|a_i|²`、每通道 eta、fiber attenuation、峰值强度与 material 后功率。新模块 [`material_nonlinearity.py`](../src/ldm_project/v2/application/research/lightgen/material_nonlinearity.py) 接受 identity/SA，位置为 fiber output 后、decoder 前。训练消费者只增加可选物理配置接线，没有增加 loss。 |
| 配置与诊断 | 参考几何和参数散落在现有代理里。 | 新增 [`lightgen_power_proxy.yaml`](../configs/reference/lightgen_power_proxy.yaml) 与 [`integrate_lightgen_physical_power_v1.py`](../scripts/research/lightgen/integrate_lightgen_physical_power_v1.py)；材料覆盖复用 [`material_nonlinearity_diagnostic.py`](../src/ldm_project/v2/application/research/lightgen/material_nonlinearity_diagnostic.py)，几何/radius 诊断在 [`physical_power_diagnostics.py`](../src/ldm_project/v2/application/research/lightgen/physical_power_diagnostics.py)。 |

此次只接入 LightGen 研究路径；没有改变正式 v2 的 identity OLS 合同。全局输入 mode 使用以 fiber pitch 划分的互不重叠 Voronoi 数值支持并逐端口单位功率归一。cladding diameter 单独记录为几何值，不将其伪装成 mode kernel 或实测端面边界。输出 mode 仍为未校准项目基底。

### 公式与单位

- 输入场定义为 `√W/m`，功率 `P=Σ|U|² ΔxΔy`，单位 W；只在 SLM 之前一次按 `U_phys=U_relative·√(P_target/P_relative)` 标定。配置中的 uniform plane-wave 实现具有 `|U|=√(P_target/A_SLM)`。
- 中心坐标为 `x_c=(c−4.5)p+o_x`、`y_r=(r−4.5)p+o_y`，`r,c=0,…,9`；复 overlap `a_i=Σ U u_i* ΔxΔy`，`Σ|u_i|²ΔxΔy=1`，`P_i=|a_i|²`，`η_total=Σ_iP_i/P_before_fiber`。代码不均分 100 个通道功率。
- 每个 D2NN 层 `U_after_device=√T_layer exp(iφ) U_before`；传播使用同一 dx 的 angular-spectrum FFT。功率从器件后到传播后检查守恒，传播后到 retained/cropped field 的减少单列为 aperture/crop。衍射是空间重分配，不计为吸收损耗。
- 光纤功率衰减为 `T_fiber=10^(−α[dB/km]·L[m]/1000/10)`，`P_fiber_out=T_fiber P_coupled`，和 SMF overlap coupling、器件 transmission 分开。此 reference 的 α=0，故本次 `T_fiber=1`。
- Gaussian 峰值强度采用 `w=MFD/2`、`I_peak=2P_i/(πw²)`，先得到 W/m²，再除以 `10⁴` 转成 W/cm²。它是局部 Gaussian 估计，不表示真实膜层的空间响应。
- SA 用功率透射 `T(I)=T_ns+ΔT·I/(I+I_sat)`，`E_out=√T(I)E_in`；`T_ns` 是低强度透射率，`ΔT` 是饱和增加量，`I_sat` 单位 W/cm²，`φ(I)=0`。参考扫值 `T_ns=0.7, ΔT=0.2` 仅为 generic scenario；代码做 `I=0`、高强度、passivity、解析导数和有限差分校验。
- 覆盖按 `r=I/I_sat` 定义：linear `r<0.1`、active `0.1≤r≤10`、saturated `r>10`。`nonlinear_activation_coverage` 等于 active 通道比例，只报告分布，不要求所有端口 active。

### 实测功率与耦合

下表是零相位、四个合成图案、CPU 物理诊断；功率来自 YAML reference config。均匀图案输入 `249.300 μW`，SLM 后仍 `249.300 μW`；Encoder 入口 `249.300 μW`、Encoder1 retained `207.736 μW`、Encoder2/ fiber 前 `184.677 μW`。fiber 总耦合 `11.950 μW`，η=`0.0647083`；零 attenuation 时 fiber 输出总功率相同。decoder1、decoder2、camera 分别约 `10.593 μW`、`9.196 μW`、`9.196 μW`。其他合成图案 η 分别为竖条 `0.001551`、对角线 `0.059192`、中心圆 `0.001279`。

四个样本×100端口合并的 channel power 分布（W）：min `6.64×10⁻¹⁸`、mean `3.49×10⁻⁸`、median `3.71×10⁻¹⁶`、p10 `5.65×10⁻¹⁷`、p50 `3.71×10⁻¹⁶`、p90 `7.22×10⁻¹⁵`、max `2.99×10⁻⁶`。相应 `I_peak` 分布（W/cm²）：min `8.36×10⁻¹²`、mean `0.04388`、median `4.67×10⁻¹⁰`、p10 `7.12×10⁻¹¹`、p50 `4.67×10⁻¹⁰`、p90 `9.09×10⁻⁹`、max `3.761`。完整 4×100 复 latent、Pi、eta 与 intensity 均保存在机器 JSON 中。

功率损耗被分开记录：每层的 `T_layer` 从输入/器件后功率验证；propagation error 由器件后/传播后功率验证；裁切量由传播后/retained 功率阶段隔离；fiber coupling 由 `ΣPi` 和 `P_before_fiber` 对比；fiber attenuation 由独立长度公式给出；材料透射单独记录于 `P_after_material_each_W`。input/output relay 的净功率变化可从相邻物理面功率读取，不能混称为材料吸收。

### 窗口收敛与 gates

窗口比较固定 dx=2 μm，grid=2048/3072/4096，对应 4.096/6.144/8.192 mm；比较 fiber plane 公共 ROI intensity、复场、`P_before_fiber`、100 个 eta 和 decoder 输入复场。预设相对容差 `1e-4`，完整逐项差值见 JSON。

| Gate | 结果 | 核验值/含义 |
|---|---|---|
| G1 global geometry | PASS | 2048² global plane，4.096 mm；center span 1.125 mm、外宽 1.25 mm、四边余量 1.423 mm。 |
| G2 100 centers inside | PASS | 100 个 x/y 坐标均在 `[-2.048,2.048] mm`，中心对称并 row-major。 |
| G3 window convergence | **FAIL** | 最大公共区 intensity L1 `5.03×10⁻⁵`、`P_before_fiber` `5.59×10⁻⁶`、eta 向量 L2 `9.96×10⁻⁶`；但 fiber-plane complex L2 `6.67×10⁻⁴`、decoder input field L2 `1.75×10⁻⁴`，高于 `1e-4`。 |
| G4 absolute input | PASS | 四个控制的输入功率均 `249.3000356 μW`，相对误差最大 `1.17×10⁻⁷`。 |
| G5 free-space conservation | PASS | encoder 最大相对差 `3.22×10⁻⁷`，decoder `2.84×10⁻⁷`；裁切另外记录。 |
| G6 layer transmission | PASS | 4 层功率比均按 `T=0.889`，最大相对差 0；场振幅乘 `√T`。 |
| G7 fiber passivity | PASS | 四个合成控制 `ΣPi≤P_before_fiber`；最大总 eta `0.0647083`。 |
| G8 matched Gaussian | PASS | 已有 Gaussian S12 模态的匹配通道 eta=`1.00000000000002`，100 通道 eta 总和同值、不增能。它只校验数值 overlap，不校验实物 MFD。 |
| G9 identity vs original linear baseline | **FAIL** | A4 identity 与 A3 逐元素完全相同；但 A0→A3 归一相机图 L1=`1.2534`，raw 数值相对差=`0.9972`，且二者单位不可比、OLS 几何不同，严格 A0 数值一致未建立。 |
| G10 no hidden renormalization | PASS | 输入功率加倍时所有记录功率级的二倍缩放误差最大 `5.16×10⁻⁷`；来源标定边界位于 SLM 前。 |

### Nonlinear sweep

执行 5 个输入功率（`0.2493/1/10/100/300 mW`）×5 个 layer transmission（`1/0.95/0.889/0.8/0.7`）得到 25 个 power scenarios；再乘 9 个 `I_sat` (`10,100,10³,10⁴,10⁵,10⁶,10⁷,3×10⁷,520 W/cm²`) 得到 225 组完整覆盖分布。每组保留合成 batch 中每样本、每端口的 `r_i`、比例、材料后 Pi 与各类统计。

reference 的 `249.3 μW, T=0.889, I_sat=520 W/cm²` 场景中，全部通道为 linear，最大 `r=0.00723`。`300 mW,T=1,I_sat=520 W/cm²` 条件扫描中，四样本共 400 个端口有 active `0.5%`、saturated `1%`、linear `98.5%`；仅对角线控制有 2% 端口 active。`300 mW,T=1,I_sat=3×10⁷ W/cm²` 场景所有通道 linear，最大 `r=1.91×10⁻⁴`。这些是通用数值场景：520 W/cm² 是其它波长/结构 TaP 的条件性 benchmark，不是 TaP@532 nm 测量；3×10⁷ W/cm² 是 Te@532 nm 高阈值文献参考场景，不是当前热蒸镀 Te 膜参数。CoSi 没有指定物理 I_sat，只参与 generic sweep 的建模范围。

集成 forward 在 fiber output 后用 `520 W/cm²` SA 条件点做 consumer 核对：参数 `T_ns=0.7, ΔT=0.2`，材料后每通道功率与 SA 公式相对误差 `9.30×10⁻¹⁵` 且被动。该条件点不代表材料识别；名义输入下最高 channel intensity 仅 `3.761 W/cm²`，最高 `r=0.00723`。

### 执行与验证

- Windows 支持解释器 `D:\ANACONDA\envs\optical-generative-models-win\python.exe` 的 focused pytest：`35 passed`（integration、既有物理审计、LightGen runtime、fiber operator、propagation reference）；Ruff focused rules、AST parse 与 `git diff --check` 通过。随后按仓库规范尝试全套 pytest，但解释器在收集期导入 `numpy._core.overrides` 时触发 Windows fatal exception `0xc0000008`，进程未给出完整 pytest 汇总；全套检查未完成，日志保存在 `C:\Users\24567\.codex\visualizations\2026\10\13\01a1331f-4464-78c1-9ecf-103867133cc1\pytest-full-final-physical-power.log`。它不改变物理诊断的 Gate 结果。
- CUDA 可见，但实际第一算子触发 `nvrtc-builtins64_121.dll` 缺失，首个 attempt 判作 `INVALID_HARNESS`，没有产生科学测量。随后在同一 Windows 解释器使用显式 CPU 运行，测量有效并保存；因此结果明确标记为 CPU diagnostic，不冒充 CUDA training/acceptance。
- 有效结果和每样本/每通道/每 sweep JSON 位于 [`attempt_02`](../outputs/20261013_physical_power_integration_v1/runs/diagnostic-only_attempt_02/physical_power_integration.json)；失败 CUDA 尝试保存在 [`attempt_01/failure.json`](../outputs/20261013_physical_power_integration_v1/runs/diagnostic-only_attempt_01/failure.json)。最终交付 JSON 是 [`physical_power_integration_v1.json`](../artifacts/physical_power_integration_v1.json)。
- 未完成全套 pytest、正式训练、评价器、真实样本生成或硬件测试。没有改 L_op、没有增加 `L_power/L_coupling/L_activation_coverage`，也没有改 optimizer、数据集或标签定义。

## 决策影响与下一步：

当前可以保留 A1–A4 的**研究模拟实现**，但不能把本次结果当作物理设备已校准或正式训练 ready。A1 说明坐标映射不再受 200 μm 窄面限制；传播能量和 coupling 对窗口变化接近稳定；A4 identity 对 A3 线性物理链严格等价。G3 仍未满足复场收敛阈值，G9 也没有定义 A0 与物理 W/m² CMOS 输出间的可比映射，因此依照用户条件不启动 A0–A4 训练 smoke。

建议下一步先给出或测出输出面到 decoder/CMOS 的物理 irradiance-to-target 标定，并明确 A0/A3 比较所用的共同 plane 和单位；另外针对 G3 复数输出场差异做一次独立参考核验，保留 `1e-4` 标准，不能事后放宽。只有这些缺口在新授权范围内关闭且全部原定 gate 通过，才考虑另行安排最小 smoke。上述是建议，**不是本报告授权开始训练**。Formal long training、loss/evaluator/data/task 改动、材料采用和硬件部署均保持 blocked。

### 本地状态

本任务在现有 dirty checkout 上进行，未清理或覆盖其它工作，未创建分支、提交或启动后继任务。代码、诊断与测试变更均限于本报告列出的 LightGen 研究路径、共享光学模块的默认关闭 opt-in 分支及其回归测试；完整本地 Git 状态以任务结束时的仓库为准。
