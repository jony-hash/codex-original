# 当前系统物理功率与非线性审计

审计日期：2026-10-13。授权范围为审计、独立 CPU diagnostics、功率预算与材料参数扫描、下一轮计划；科学更新为零。本任务未改正式训练、评价器或checkpoint。

**结果与状态：** 审计和独立诊断完成，最终为 `AUDIT_COMPLETED_WITH_PHYSICAL_GAPS`。现有主链的绝对功率、每层材料透射、真实125 μm阵列映射及物理材料层未闭合；大FFT窗口和复数SMF重叠已经存在。14项为4 PASS、4 PARTIAL、6 FAIL。新增工具支持条件功率／材料扫描，未接入训练。

**科学认识：** “整个传播窗口只有200 μm”和“fiber只是10×10下采样”均被当前代码与运行否定。保留的关键解释是**200 μm耦合平面与实际端面未对应、场没有W标定、材料响应未知**。无孔径Gaussian守恒误差2.33×10⁻¹⁵；小窗口负控仍守恒却把约99.99%光卷到另一侧，说明守恒不能代替传播准确性。指定3×10⁷ W/cm²场景在本轮未训练光束中均近线性；这不否定所有Te器件。

**目标缺口：** 尚缺明确的端面／像面、pitch/MFD/倍率、source/SLM功率与透射、8.5 mm复传输、532 nm薄膜T(I)/φ(I)及时间条件。峰值按数值模态计算；不能视为真实端面实测值。未加载训练checkpoint、未测MNIST任务质量或真实硬件；这些不属于本轮完成项。

**工程完成与验证：** 新增两个独立模块、两个CLI入口、14项聚焦测试；实际测试14 passed，Ruff检查和AST编译、`git diff --check`通过。A/C/D/E数值组件检查及保存回读通过；B拒绝名义实物阵列在200 μm耦合平面上的支撑。200条模型预算、20,000条通道功率、450组完整覆盖分布、7,200条四模型响应及125条总耦合敏感性均保存。本轮未运行全套pytest；未改共享光学实现。

**决策影响与下一步：** 先闭合功率入口和阵列参考面，再逐项加入透射、transport和材料；保留现有相对L_op。建议A0–A5分步验证，不同时新增多项loss。本轮授权已完成，交付后停止；下轮接入正式光学链、训练、采用、可读性验收或部署须另行授权。

## 范围与真实光路

主审计对象是 `LightGen2DSemanticGenerationInputOLSAlignment` 的小架构，它最接近本次指定的 2+2 光路。必须区分三个现存分支：

1. 小架构：归一化图像 → 200×200、2.5 μm 振幅 SLM → 相干 B=0 relay（倍率 −0.4、有 pupil）→ 两层 100×100、2 μm phase mask，每层 pad 至2048后ASM传播1.5 mm并crop回100 → 100个复LP01 overlap → `T=I_100` → 同平面模态重发射 → 两层同样的Generator → 倍率−2.5后端relay → 原始CMOS `|U|²`。**没有材料层和激光功率标定。**
2. 当前保存的云端前景研究采用 `ExpandedGenerator`：Encoder仍为两层100×100；Generator为两张100×100旧相位面加两张400×400新增相位面，共340,000参数。它不是本次目标的两层100×100 Decoder。本轮不加载或评价这些训练端点。
3. 维护的正式 v2 条件运行时使用 identity OLS，合同明确没有fiber。`optics/lightgen.py` 和 `fiber_ols.py` 是另一个2+4架构代理，不是当前LP01研究链。

本轮源码实例化确认小架构 phase shape `(2,1,100,100)`、传播shape `(2048,2048)`。开场零更新 `forward_from_image(ones)` 在 `ComplexOLSState.__post_init__` 因未导入 `Mapping` 报 `NameError`，见[开场观察](../../../outputs/20261013_physical_power_audit_v1/runs/opening_entrypoint_probe.json)。审计期间该工作树新增 `from typing import Any, Mapping`（当前第14行，非本任务所写）；随后attempt_02/03的完整未训练入口探针通过。**开场失败是工程观察，当前已不能列为仍未解决的物理缺口。** 两次source状态均保留，不归功于本轮诊断代码，也不因此声称训练端点通过。

## 关键文件与证据索引

以下路径相对仓库根；行号对应本轮审计的当前工作树。

| 文件 | 类／函数与关键位置 | 消费者 |
| --- | --- | --- |
| `src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py` | 常量39起；`LightGenImageInputAdapter.__call__`181；`LightGenAlignedComplexOLS`220；`mode_identity`253；`write_to_generator_input`323；主模型379、phase stacks444、`forward_from_image`543 | 小架构生产研究路径 |
| `src/ldm_project/v2/optics/propagation.py` | `_transfer_function`19、`angular_spectrum_propagate`37、独立reference70 | SI波长ASM、orthonormal FFT |
| `src/ldm_project/v2/optics/d2nn.py` | `FiniteCropPhaseStack`13；phase调制47、pad62、crop84、retention100 | 有限器件孔径与层间传播 |
| `src/ldm_project/v2/optics/relays.py` | `CoherentImagingRelay`105、`__call__`133、固定Jacobian振幅因子141 | 前／后相干成像、有源／靶pupil |
| `src/ldm_project/v2/optics/fiber_operator.py` | `FiberArraySpec`20；LP01解57；`_mode_bank`157；`FiberArrayOLS`197；project239、transport246、reemission254、energy检查263 | 当前100端口复场overlap |
| `src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py` | `GaussianFiberGeometry`24、`GaussianFiberArrayOLS`33、Gram被动性检查71、project100 | opt-in Gaussian，未标定项目几何 |
| `src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py` | `physical_plane_power`651、`contracted_plane_powers`666、`optical_power_loss_from_powers`675、`losses`生产消费者约1070 | 五平面的积分和冻结相对参考L_op |
| `scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py` | `ExpandedGenerator`148、`phase_layout`177 | 四张混合窗口相位面 |
| `scripts/research/lightgen/execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1.py` | Generator形状核验566、723；合成目标1126 | 当前保存云研究的架构身份 |
| `scripts/research/optical_manifold/execute_lightgen_2d_nonlinear_fiber_facet_activation_canary_v1.py` | `FacetMaterialProbe.apply`335、ReLU342 | 历史算法端面代理，未接入当前基线 |
| `src/ldm_project/v2/optics/fiber_ols.py` | `gaussian_single_mode_bank`44、project63 | 另一架构代理，离散sum归一化，不能混用SI系数 |
| `src/ldm_project/v2/optics/fiber_pool.py` | `FiberResearchSpec`16、`_make`74 | 合成dB幅度/相位/offset扫描，不是8.5 mm实测模型 |
| `src/ldm_project/v2/contracts/planes.py` | `PlaneSpec`12、`fov_m`34 | `shape × pixel_pitch_m`坐标合同；未声明场的W标定 |

已定向扫描 `src/`、`scripts/`、`configs/` 中的传播、fiber、power、normalization、material词项，并核对当前状态、代码地图及历史入口。归档文本和其他代理的功能不能算作当前链已接入。

## 14项检查

状态针对本次目标：PASS=已实现并符合被检查的数值合同；PARTIAL=存在子功能但缺物理闭环；FAIL=目标缺失或不一致；UNKNOWN=无足够证据。PASS不表示实际硬件通过。

| 检查 | 状态 | 类／函数、关键变量及实际行为 | 目标符合程度与风险 |
| --- | --- | --- | --- |
| 1 532 nm物理传播 | PASS | alignment `WAVELENGTH_M=532e-9`；`PlaneSpec.wavelength_m`被ASM频率radicand和`2πz/λ`读取 | `532e-9 m=0.532 μm`，pitch、z均m。主链未发现nm/μm/m混算；并非只存配置。v1另有Fresnel/ASM，当前主链不用RS |
| 2 100×100与2 μm含义 | PASS | 小架构Encoder/Generator参数`(2,1,100,100)`；`OLS_PIXEL_PITCH_M=2e-6`；active width200 μm | mask与pad网格已分开。当前大Generator有400×400额外相位面，不能声称它也是本目标2+2 |
| 3 独立大窗口 | PASS | `FiniteCropPhaseStack.propagation_shape=(2048,2048)`，同pitch补零 | `Lx=Ly=4.096 mm`；只在每次ASM内部保留，随后crop回200 μm。没有把mm阵列接在全局canvas上的实现 |
| 4 有限孔径与损耗 | PARTIAL | phase→pad→ASM→hard crop；relay有circular source/target pupil | 有真实数值孔径损耗；FFT仍周期性，不是无限自由空间。能量守恒不能排除wrap-around；需窗口收敛和传播采样诊断 |
| 5 绝对功率 | FAIL | input adapter对[0,1]图像resize/clamp再sqrt；SLM产生零相位复场；无`P_laser` | `∑|U|²dxdy`已有，但未赋予`U`单位√W/m，积分不能称W。新增诊断入口标定不意味着原模型已修复 |
| 6 每层器件透射 | FAIL | phase mask为`polar(ones,phase)`，振幅透射1 | crop/pupil不是材料吸收；无每层可配置`√T_layer`。未发现当前链将0.889错乘振幅；该参数根本未接入 |
| 7 SMF模式重叠 | PARTIAL | `FiberArrayOLS.project_coefficients`用`einsum(u*.U)*dxdy`；LP01、V=2.2、waist7.111 μm、每端口矩形cell、面积归一化；Gaussian可选20/3.5 μm | 不是pixel downsample，复相位保留。截断LP01是合成近似；MFD/core/NA/端面像面未标定。Gaussian候选不自动替换LP01 |
| 8 10×10真实几何 | FAIL | 100端口row-major；LP01中心−80…80 μm，间距17.7778 μm；Gaussian−90…90 μm，间距20 μm | 非125 μm真实端面阵列。暂定125 μm pitch/cladding的中心跨度1.125 mm、外宽1.25 mm，超过**耦合平面**200 μm；单边余量−525 μm。pad宽4.096 mm不能解决耦合平面错配 |
| 9 coupling与fiber传播分开 | PARTIAL | project→`transmit_coefficients(T)`→reemission三API分开，当前`T=I_100` | 结构分开；没有fiber length、βL、8.5 mm对应attenuation。可把共同相位视为规范，但差分相位、损耗未知，不能说已建模8.5 mm |
| 10 每通道W及不增能 | FAIL | 系数`|a_i|²`和energy/coupling ratios可计算，transport SVD及forward能量门存在 | 原输入无瓦特标定；独立diagnostic已从实际project/transport写出100个W场景功率并检查不增能，但原训练链仍无W。aligned writer未调用operator.forward的整场energy门 |
| 11 材料层 | FAIL | 当前`_run_generator`从writer直接入Generator；历史FacetMaterialProbe为normalized ReLU代理 | 无当前物理材料层；数字评价器ReLU也不属于光路 |
| 12 绝对材料光强 | FAIL | 没有当前链W/cm²、I_sat或物理端面强度消费者 | waist是项目模态坐标，不足以推定真实端面强度；不能直接将tensor作为W/cm² |
| 13 防过暗loss | PARTIAL | `L_op=mean(log((P+eps)/(P_ref+eps))²)`；五个crop/OLS/Generator平面，参考来自TRAIN、更新前冻结；当前组合目标仍消费L_op | 相对积分尺度有约束，且高／低偏离都罚；无激光W、绝对power floor/ceiling或材料覆盖。必须保留原约束，不能据此声称光强够 |
| 14 功率归一化泄漏 | PASS | 当前层后、fiber前后、Generator前无`U/norm(U)`；raw CMOS不按max归一化；下节逐项区分输入／基底／loss／显示处理 | 物理链未发现逐层偷归一化，但边界没有绝对标定。loss的形态归一化会消除亮度敏感性，不会修改传播场；不能混称 |

## 单位、窗口与几何

- 波长、采样、距离、waist、FOV：m。相位：rad。图像和SLM command：[0,1]无量纲。现有场：未作绝对光度标定的数值振幅；`physical_plane_power`这个函数名不能代替单位证据。
- 若把场定义为 `U=√(nε₀c/2) E_phys`，则`|U|²`为W/m²，离散功率`P=∑|U|²dxdy`为W。诊断只在入射边界设置一次这种标定，传播中不重新归一化。
- trainable active grid100² @2 μm：200 μm；ASM pad2048² @2 μm：4.096 mm；OLS/crop plane100²：200 μm；SLM/CMOS200² @2.5 μm：500 μm。
- LP01的`FiberArraySpec.pitch_m=2e-6`是**采样pitch**，不是fiber中心间距。中心间距由`FOV×fill/9`产生。LP01被矩形cell截断并离散归一；waist7.111 μm是该LP01轮廓的1/e振幅半径参数，不能无条件套Gaussian峰值公式。
- Gaussian候选是全Gaussian，场半径3.5 μm、MFD7 μm、pitch20 μm，没有端口cell或离散重归一化。必须检查Gram与完整project+reemission被动性，不能认为所有未正交Gaussian模式自动构成独立fiber。
- 实际125 μm中心pitch只是本轮指定的名义场景；LightGen补充仅给125 μm**外径**，没有完整pitch/MFD/relay标定。若数值平面是像面，放大率同时缩放pitch与MFD，必须保持二者比例；不能只改pitch。

## 所有相关归一化与幅度处理

| 位置 | 处理 | 是否改光场功率 |
| --- | --- | --- |
| image adapter181起 / IdealAmplitudeSLM | 输入要求[0,1]，resize、clamp、sqrt，零照明相位 | 输入边界决定相对入射功率；不是瓦特标定 |
| `_mode_bank`188–194 | 模式除以`√(∑u²dxdy)`；可选Löwdin | 基底归一化；未将每个输入场归一化，但截断模式与实物模式可能不一致 |
| Gaussian baseline | 解析无限平面单位模功率；检查离散积分 | 不对输入场或每端口离散重归一 |
| relay141 | 固定`1/abs(M)`，面积随M²变化 | 物理坐标Jacobian，不依赖样本功率；不能删除 |
| ASM FFT | fft2/ifft2均`norm=ortho` | 数值DFT规范；不逐场归一 |
| D2NN crop retention | 只计算分子／分母比值 | 不反写场，不补回裁失功率 |
| BOGT `u_j=|z_j|²/(s_j+eps)`，约547 | TRAIN冻结尺度用于分布loss | 统计变量；writer仍使用原复系数 |
| L_op675 | 相对冻结参考的log ratio | loss；不是传播归一化，不提供W |
| `_qfg_components` scripts/…foreground_objective_gradient_v1.py114 | raw/target各除图像总质量 | loss／形态指标对整体亮度不敏感，不回写raw场 |
| `canonical_semantics.cmos_view`62 | clamp[0,1]、avg_pool28² | 评价器/显示边界截断，非物理材料；不用于本轮调参 |
| `normalized_feature_loss`71 | feature L2 normalization | 特征loss，非光学传播 |
| 梯度RMS／参数方向归一化 | 用于优化/方向比较 | 改梯度的标度，不是功率归一化 |
| 另一代理 `fiber_ols.py`58、68 | 模式sum-unit norm，overlap无dxdy | 离散向量约定；系数不能直接当W，不能与面积归一LP01混用 |

## 最严重的五项风险与修改顺序

1. **P0：功率量纲未闭合。** 有积分没有激光标定。先明确P_laser测量平面、入射beam/SLM透射与单次入口标定，保持传播复场和面积不变。
2. **P0：端面／像面与阵列支撑错配。** 耦合FOV200 μm不能容纳名义1.25 mm阵列。先决定物理参考面，再联合设pitch、MFD、倍率与coupling support；不直接把所有层放大成1024。
3. **P0：材料模型、绝对I及时间条件缺失。** 单532 nm功率扫描不验证CW/脉冲响应、热损伤、SA/RSA和相位；没有本项目薄膜曲线，不能把Te/TaP/CoSi当ReLU。
4. **P1：缺器件透射与fiber transport的物理参数。** crop/pupil并不能替代吸收，`T=I`没有长度相关传播。新增模型需区分coupling、√T_layer、fiber attenuation和固定／差分相位。
5. **P1：传播精度和架构身份不能由守恒替代。** 本轮只验证Gaussian窗口收敛，未验证所有训练相位的采样／bandlimit；当前大Generator也不是目标2+2。开场`Mapping`工程问题随后在当前工作树消失，保留历史观察。现有数值／采样端点不可冒充目标硬件架构。

## 文献参考与未知参数

- 本地 `research_assets/lightgen/science_adv7434/paper/sciencesm.pdf`，PDF第11页给入射相干光249.3 μW=0.2493 mW；第2/26页给MSL-U-532-300mW型号；第12页给实测平均**功率**透射0.889；第3页给8.5 mm和125 μm外径。这些是Chen系统证据，预算只使用其参考场景。[论文](https://doi.org/10.1126/science.adv7434)
- 520 W/cm²来自TaP量子点、tapered-fiber、Er激光条件，摘要给0.52 kW/cm²和调制度10.8%；不是TaP@532 nm端面薄膜参数。[作者预印本](https://preprints.opticaopen.org/articles/preprint/Passive_mode_locking_Erbium-doped_fiber_laser_based_on_semimetal_TaP_quantum_dots/24599145)
- 指定3×10⁷ W/cm²保留作高阈值扫描，但本轮尚未核实它的Te@532 nm准确出处，禁止标成已核实文献数。已找到的Te纳米线溶液532 nm、7 ns文献Table1实际给1.6×10¹³ W/m²=1.6×10⁹ W/cm²，并含SA/ESA/scattering；它也不是热蒸镀Te膜。[原始论文作者PDF](https://pradeepresearch.org/pdf-files/listpdf/204.pdf)
- 未知：实物pitch、MFD、core、NA、端面像面倍率、input beam/SLM绝对功率传输、每层T及相位透射、8.5 mm损耗与复传输矩阵、材料膜厚/形貌/覆盖、532 nm T(I)/φ(I)、时间响应、脉宽/重复率、损伤及热上限。无CoSi参数赋值。

## 独立诊断的实际结果

最终诊断：[attempt_03/diagnostics.json](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03/diagnostics.json)；原始通过的A/B/C在attempt_01保存，并在02/03显式复用。attempt_01的D因诊断封装未重绑Generator出口平面而失败，修复诊断后重跑D/E；[failure.json](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_01/failure.json)保持不变。02/03逐级读数完整相等；03在完整入口探针之后复核所有模型state_dict不变。

| Diagnostic | 实测结果 | 解释 |
| --- | --- | --- |
| A 无孔径Gaussian功率守恒 | `Pout/Pin−1`绝对值2.3315×10⁻¹⁵；NumPy独立积分相对差2.1684×10⁻¹⁶ | orthonormal FFT与unit phase不产生功率损耗；2 μm网格的可采频段无evanescent成分 |
| A pad收敛 | 同20 μm waist、532 nm、1.5 mm传播，2048/3072的中心100² ROI复场相对差6.6971×10⁻¹⁴ | 只支持此Gaussian输入及距离，不能推广为任意训练相位准确 |
| A wrap负控 | 256网格中tilted Gaussian传播后左半功率占99.9902%，守恒误差仍4.44×10⁻¹⁶；2048对应左半占8.96×10⁻²⁵ | 小窗口周期卷回明确；守恒测试并不能侦测它 |
| B 名义阵列支撑 | 125 μm pitch/直径外宽1.25 mm；200 μm耦合平面四边margin均−525 μm，FAIL | pad4.096 mm四边margin+1.423 mm，但pad不是coupling plane |
| C LP01匹配 | 目标通道η=1.00000000001285 | 浮点误差量级，在声明容差内；没有强行把结果归一到1 |
| C Gaussian匹配／偏差 | η：匹配0.999998907、x/y offset各0.367878956、1° tilt0.877991998、2倍waist0.6399999998；解析参考最大绝对差3.5523×10⁻⁶ | 位移／角度／错waist正确降低重叠；100端口功率全部保留且不增能 |
| D 输入单次1 mW标定 | 四种Gaussian照明，unit SLM command；现有relay、Encoder、project/transport/reemit、Generator和后relay运行 | 零相位、无checkpoint，不是训练端点复现。完整原图像入口的另一个all-ones smoke通过 |
| E 不增能 | LP01／Gaussian所有样本的通道功率和、重发射功率均不超过输入容差；逐级功率单调不增 | 比较actual complex overlap输出，未按总功率重归一化；1 W→1 mW线性缩放最大通道误差低于10⁻¹² W |

1 mW、center Gaussian（SLM waist60 μm；T_layer=1）的现有LP01组件trace如下。单位均W，正是诊断数值光源所赋单位，原模型并未因此获得激光标定。

| stage | 功率W |
| --- | --- |
| P_input | 0.0010000000000000002 |
| P_encoder_entrance | 0.0009999999999999998 |
| P_after_encoder_1 | 0.0009999999999999538 |
| P_after_encoder_2 / P_before_fiber | 0.0009999999991964995 |
| P_fiber_total = ∑P_i | 0.0007169631239182994 |
| P_before_nonlinearity（实际没有材料） | 0.0007169631222382161 |
| P_after_decoder_1 | 0.0007159615925700581 |
| P_after_decoder_2 | 0.0006825294607206623 |
| P_camera | 0.0006825292523939066 |

各样本／100通道完整记录见[absolute_power.json](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03/absolute_power.json)与[复系数证据](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03/component_power_evidence.pt)。LP01四种输入coupling为71.6963%、71.6794%、71.6742%、74.8294%；Gaussian为18.8413%、18.8415%、18.8414%、19.1656%。两者pitch、waist和mode support都不同，此差异不能单独归因于模式家族，更不能据此选择模型。

## 功率预算、通用材料与覆盖扫描

最终[预算报告](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/report.json)以attempt_03实际复系数为输入。扫描P_laser为0.2493/1/10/100/300 mW、每层功率T为1/0.95/0.889/0.8/0.7；不重新运行未改变的线性光路，而使用已验证的线性缩放。两个Encoder的幅度因子为`(√T)²=T`，因此材料前功率乘`T²`；四层无材料出口乘`T⁴`。

- [power_budget.csv](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/power_budget.csv)：200条模型预算，保留每种输入的overlap coupling。
- [channel_power.csv](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/channel_power.csv)：20,000条通道功率及峰值光强，row-major。没有100通道等功率假设。
- [coupling_sensitivity.csv](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/coupling_sensitivity.csv)：125条单独的总耦合0.1/0.3/0.5/0.7/0.9假设。仅用于尚无真实125 μm端面overlap模型的预算；不替代当前模型算出的coupling，也不产生逐通道均分。
- [coverage.csv](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/coverage.csv)和[coverage_distributions.jsonl](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/coverage_distributions.jsonl)：450组覆盖。每组保留4×100的r_i及batch、每个样本、每个通道的mean/median/p10/p50/p90/max/min。`r<0.1`、`0.1≤r≤10`、`r>10`三类完整覆盖；非线性覆盖只作分布诊断，不设全通道硬阈值。
- [material_response.csv](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/material_response.csv)：7,200条四模型响应。I_sat为10、100、520、10³、10⁴、10⁵、10⁶、10⁷、3×10⁷ W/cm²。

新增 `material_nonlinearity_diagnostic.MaterialNonlinearity.forward(field, intensity_w_cm2, params)` 只在诊断中使用。四模型公式：

1. identity：`U_out=U`。
2. SA：`T_SA(I)=T_ns+ΔT I/(I+I_sat)`，`U_out=√T_SA U`；这里T_ns是低强度**透射率**，不是文献常见的nonsaturable loss符号。
3. SA+RSA：`T=T_SA/(1+I/I_RSA)`，`U_out=√T U`；这是被动现象模型，不是对ESA微观方程的拟合。
4. 幅度+相位：`U_out=√T_SA exp(i φ_max I/(I+I_sat)) U`。

扫描使用`T_ns=0.7, ΔT=0.2, I_RSA=10 I_sat, φ_max=0.5 rad`，全部标为任意通用假设，未给任何Te/TaP/CoSi器件赋值。被动性要求`0≤T_ns, ΔT`且和≤1；拒绝非有限、负强度、缺失阈值／RSA参数或不匹配shape。没有将T当作幅度、没有clip掩盖增益。

光强使用：Gaussian为`I_peak=2P_i/(πw²)/10⁴` W/cm²；LP01为`P_i max|u_i|²/10⁴`（采样且经cell截断的模态），没有把LP01无条件套成Gaussian。**这里材料平面假定等于数值OLS平面；实际端面映射未知。** 扫描将每通道峰值用于lumped传输，是敏感性近似；不是在空间分布I(x,y)上积分材料损耗，也没有处理非线性后的模式畸变。接口本身支持local field，独立验证使用`I=|U|²/10⁴`；后续真实薄膜建模应使用空间积分和模式重投影。

[材料独立验证](../../../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/material_validation.json)实际核对四模型功率透射、相位、解析I导数／符号、中心差分和不增能；测试还注入错用T振幅，最终值检查会拒绝，未修改生产接口作“修复”。

| 场景（4种输入×100通道） | linear | nonlinear coverage | saturated | r_max |
| --- | --- | --- | --- | --- |
| LP01，1 mW，T=.889，I_sat=520 | 97% | 3% | 0% | .20943 |
| Gaussian，1 mW，T=.889，I_sat=520 | 96% | 4% | 0% | .29439 |
| LP01，300 mW，T=1，I_sat=520 | 56.5% | 30.5% | 13% | 79.497 |
| Gaussian，300 mW，T=1，I_sat=520 | 62.5% | 25% | 12.5% | 111.747 |
| LP01，300 mW，T=1，I_sat=3×10⁷ | 100% | 0% | 0% | .0013780 |
| Gaussian，300 mW，T=1，I_sat=3×10⁷ | 100% | 0% | 0% | .0019370 |

此处低阈值与高阈值的差异支持“材料阈值和逐通道分布是决定性条件”，不能证明TaP在532 nm优于Te，也不证明低coverage导致MNIST生成失败。峰值lumped处理可能高估SA透射收益；未做器件时间／空间完整仿真。

## 新增工具与复核入口

| 新增文件 | 功能与隔离边界 |
| --- | --- |
| [physical_power_diagnostics.py](../../../src/ldm_project/v2/application/research/lightgen/physical_power_diagnostics.py) | A–E、入口单次标定及现有组件trace；不建optimizer、不读数据／checkpoint、不调用evaluator |
| [material_nonlinearity_diagnostic.py](../../../src/ldm_project/v2/application/research/lightgen/material_nonlinearity_diagnostic.py) | 四种静态被动材料方程、physical intensity接口、覆盖分布、独立值／导数／符号复核；不被正式训练链导入 |
| [check_physical_power_v1.py](../../../scripts/research/lightgen/check_physical_power_v1.py) | 现有研究目录的独立CPU CLI，统一执行建议的多项diagnostics，拒绝已有output目录 |
| [power_budget.py](../../../scripts/research/lightgen/power_budget.py) | 读取显式通过的组件证据，核对复系数／100通道W与不增能，再执行条件扫描；不修改光学模型 |
| [test_physical_power_audit_v1.py](../../../tests/research/lightgen/test_physical_power_audit_v1.py) | 14项回归，包含遗漏面积、错乘T振幅、保存证据增能的负例，生产producer一直通到最终诊断consumer |

本轮最终实际命令（在仓库根、支持的Windows环境执行）：

```powershell
python -B -m scripts.research.lightgen.check_physical_power_v1 --output-dir outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03 --reuse-checked-dir outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_02
python -B -m scripts.research.lightgen.power_budget --diagnostic-dir outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03 --output-dir outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02
```

这些目录已经存在，命令会拒绝覆盖；今后复现应使用新的明确run路径。首次完整A/B/C命令不带reuse参数，已保存在attempt_01。验收记录见[delivery_verification.json](../../../outputs/20261013_physical_power_audit_v1/runs/delivery_verification.json)：从20,000行CSV独立复算所有通道和，不增能；450组JSONL覆盖分类复算误差0；每组100通道非均匀功率完整保留。CSV功率总和最大差5.55×10⁻¹⁷ W。链接和五段反馈顺序核对通过。没有新训练或后继自动动作。

## 下一轮最小计划与停止点

问题：能否在目标2+2／单532 nm范围内建立量纲、阵列坐标和被动非线性都一致的前向链？输入必须是显式物理参考面、beam、pitch/MFD/relay、层透射、fiber transport和材料T/φ的场景；实测缺失项只能sweep。

推荐A0原模型→A1仅absolute入口/传播（不新增loss）→A2层透射→A3已明确几何的overlap/transport→A4材料→A5可选power-aware loss。A3的“real”只有测量支持时才能使用；现有overlap已存在，工作重点是几何闭环。在只做仿真的下一轮，可以把参数明确写为假设并sweep，而不声称实测。A1需固定入射beam、P_laser定义平面和SLM command造成的真实功率变化，不能将每张图都归一到同样的透射功率。L_op参考应与新单位一致，epsilon也有尺度，不能直接把旧无量纲参考当成W。

建议保留L_op。只有A4的冻结前向分布显示任务相关的功率／覆盖问题，再考虑A5。候选power-aware项可用`L_power=mean(ReLU(P_min−P)/P_scale)²+mean(ReLU(P−P_max)/P_scale)²`，`L_coupling=mean(ReLU(η_min−η))²`；P_min/P_max来自指定应用／材料条件，η_min来自独立耦合需求，不从评价器指标选取。覆盖若要训练需设计平滑、可导的目标并明确其决策用途，不能直接优化本轮硬区间计数。未启用任何新loss或给lambda默认值。

`permitted_next_actions`（本轮已完成，以下仅建议、须另行授权）：先为目标2+2建立单次入口W标定与明确端面/像面方案；验证200 μm器件口径、global传播和宽coupling plane之间的坐标/采样；加入被动√T和长度/传输参数；对空间T(I)/φ(I)的材料前向做独立值、导数、被动性和冻结trace核对；随后才规划单变量A1–A4和可选A5。不得把已有preflight或空闲GPU视为许可。

`blocked_actions`：本任务内正式训练、改评价器、同时更换架构和loss、依据本轮图像指标调参（本轮根本未调用评价器）、材料/模型采用、将文献proxy写成本项目实测、物理部署。

本轮新增文件仅为报告／JSON、两个诊断模块、两个研究CLI、一个测试文件及CURRENT_STATUS本轮短索引；证据位于新版本化run目录。预先存在的dirty改动保留，本任务没有commit、branch、Sources同步或云执行。原始开场状态与后续工作树的导入变化已分开记录。

通过几何/量纲/被动性检查，只支持进一步单独设计冻结前向评估；几何失败需要重定义参考面，不能启动训练；仅材料阈值不可达意味着该**扫描条件**弱响应，不否定全部材料；源码入口错误为工程阻断，修复并复核后才能进行相关完整链动作。所有训练、模型采用、正式可读性验收与部署仍需后续明确授权。本轮报告和诊断交付后停止。
