# HW207 Optical Twin 对比实验矩阵

## 状态和统一规则

本矩阵是后续标定/对比实验的设计，不是已执行结果。当前没有执行 E0–E9，也没有运行生成或调用 evaluator。每个配置必须明确关联 PAPER_REFERENCE、HW207_TARGET 或 HW207_CURRENT 之一、硬件身份、F0–F6 fidelity、标定 ID、input encoding ID、固定图案和 raw 数据路径；三种配置身份不自动继承参数。

光路、硬件来源、坐标和模型层级定义见[数字孪生设计总览](207_optical_twin_design.md)；参数测量顺序与标定完成条件见[标定计划](207_calibration_plan.md)。

实验从单因素比较开始，使用共同波长、输入功率参考面、输入光斑、测试图案、芯片坐标和输出测量面。相机增益/曝光、relay pupil 或图像归一化若随组变化，必须被列为变化因子或协变量；否则该对比不能解释为单一因素。结果同时报告原始功率与图样归一化结果。

E1 的主要对照只使用 binary-compatible pattern：all ON、all OFF、horizontal/vertical binary grating、checkerboard、binary random、binary aperture。Gaussian illumination 由 DMD all ON 测入射光束包络；binary Gaussian aperture 按预先固定阈值二值化；gray Gaussian 只进入 PWM 或选定的 spatial dithering。每种图案及转换使用独立 pattern ID / encoding ID，并固定视场/ROI、占空比或目标振幅、DMD方向、照明和曝光；重复采集数应在执行前登记。偏振角、DMD时序、relay tolerance、相机阈值等没有依据的数字不在本文中指定。

## 指标与统一判定语义

### 物理接口指标

- 输入强度误差：对齐、相同 ROI 后计算 NRMSE；参考范数接近零时使用预先定义的显式 epsilon 稳定分母，或同时报告 absolute RMSE。物理强度和按固定规则归一化的图样误差并列报告。
- 功率效率：指定两个物理参考面后计算 \(\eta=P_{out}/P_{in}\)，分别列出绝对功率与透过/收集效率。
- 均匀性：在预先定义的有效照明 ROI 上报告均值、标准差和变异系数；区域不能按结果事后挑选。
- 对比度：对黑白周期靶报告 Michelson contrast，并标记空间频率和方向。
- 空间频响/MTF：使用周期光栅或分辨靶记录频率响应；明确区分 DMD/relay optical transfer 与 camera measurement-chain MTF。
- Clipping / range：分开报告 intentional dark pixels、ADC floor clipping、full-well/high-end saturation 和 valid dynamic-range utilization。不能把所有零值像素都当作低端裁剪；同时保存未裁切前功率信息。
- 配准误差、仿射残差和图像畸变：均换算为芯片平面物理单位，并给测量不确定度。
- 复场重叠：仅当干涉/数字全息等相位测量可追溯时计算；否则记为 N/A，不由强度图推断。

### 固定网络映射指标

只有 E8 才查看冻结网络输出：MSE、PSNR、SSIM、归一化相关性和输出能量偏差。输出图像指标和物理功率必须并列；图像归一化不得掩盖光功率下降。该轮不运行 FID、Semantic Accuracy、Precision、Recall 或 LPIPS intra，也不以输出图像指标证明硬件测量已通过。

### 有效、通过、失败和未分辨

每个实验执行前要登记性能接受阈值、测量重复数、样本/图案用途和判定消费者。本文没有收到这些具体阈值，所以当前矩阵只定义解释逻辑，不预填任意合格数值。

- **测量有效**：身份、配置、输入面、传感器状态、重复采集和不确定度完整；否则 INVALID/UNRESOLVED，不能算科学负结果。
- **可分辨效应**：组间变化超过该对比的重复性与合成测量不确定度，才归因到受控因子。
- **未检测到效应**：在当前灵敏度/视场/噪声范围内变化不超过不确定度。保留简单模型作为当前可支持解释，不写成“物理上没有效应”。
- **有效失败**：只有事先有用于该决策的接受阈值，并且完整测量越界时，才能称相对于该阈值失败。没有阈值时报告读数和未决，不补造 Gate。

## 完整实验矩阵

### E0：Ideal amplitude reference

- **目的**：以 HW207 自己的理想目标输入接口建立数值上限，回答同一目标 LightGen 芯片/传播模型下理想振幅图样的基线响应。
- **输入/固定条件**：HW207_TARGET/F0_IDEAL_AMPLITUDE；A(x,y)∈[0,1]、532 nm、100×100 / 2 μm / 200×200 μm 目标芯片、共同 E0、固定下游传播模型、目标芯片面 ROI、测试 pattern 和功率参考面。
- **唯一接口条件**：理想连续振幅映射。它不是 HES6001 实物复现，不包含 HES6001 灰阶 LUT、偏振、非均匀性或液晶复传输。
- **观察与指标**：理想目标芯片输入面强度、参考功率、NRMSE基点、功率、均匀性、contrast/频响和有效区域。
- **通过/失败判定**：作为数值参考，需验证输入范围、坐标、单位和共同下游接口一致；它不构成实物硬件通过。若照明归一化/边界未定义，则 E0 无效。
- **依赖/决策影响**：为 E1/E2 提供共同理想输入基线；PAPER_REFERENCE 保持文献参照，不直接作为 E1 数值对照芯片，也不说明 HES6001 或 HW207 已复现。

### E1：DMD binary

- **目的**：比较实际二值 DMD 前端和 E0 连续振幅输入造成的场/强度与功率差异。
- **输入/固定条件**：HW207_CURRENT/F1 起步；E1 使用与 E0 相同的 HW207_TARGET 100×100、2 μm LightGen 目标和同一下游传播模型。固定 binary-compatible pattern、illumination、chip input plane、relay/camera calibration；DMD 二值 pattern 经镜面状态和 relay 收集传播。每个 pattern transformation 有独立 encoding ID。
- **唯一变化因子**：输入接口从 E0 连续振幅映射变为 DMD Binary。Gaussian 只按统一规则转换为 all-ON beam-envelope 测量或 binary threshold aperture；不得作为未经编码的 static binary gray pattern。光源 power 在 DMD 入射面记录；不要通过对每幅结果单独归一化隐藏功率差。
- **观察与指标**：芯片输入面原始强度、DMD/芯片功率、NRMSE、效率、均匀性、contrast、横/纵频响、clipping、仿射残差；无复场相位测量时不算 overlap。
- **通过/失败判定**：有效后报告差异与不确定度。若主要是总功率变化，后续优先标定效率；若图样/频响变化可分辨，保留 DMD 离散/衍射/relay 机制作为必要模型；若测量无分辨率则保持未决。
- **依赖/决策影响**：依赖步骤 1–6 的身份、拓扑、倍率和 binary 基线；它决定是否可用简化标量输入模型代表目标范围。

### E2：DMD naive gray（负对照）

- **目的**：测出把数字 0–255 灰度直接当连续场振幅的误差，不将此路径当成 F9000 实际行为。
- **输入/固定条件**：与 E0 使用相同目标振幅图和下游数值模型；对照命令/平均 duty 与 E1 所用目标对应。
- **唯一变化因子**：灰度映射采用 A=g/255 的理想连续振幅近似，配置必须同时标为 **UNPHYSICAL_DIAGNOSTIC_ONLY** 与 **NOT_PHYSICAL_DMD_RESULT**。
- **观察与指标**：E0/E1 指标加上 naive gray 与 Binary/PWM/dither 的归一化图样差和绝对功率差。
- **通过/失败判定**：此项的“通过”是负对照能暴露理想灰度近似与真实机制的差异，或在已知精度范围内显示差异未分辨；不得用它宣布 DMD 是连续振幅器件。若它和 E1 一致，仅能说明当前测试图案/视场/灵敏度下未分辨。
- **依赖/决策影响**：依赖 E0/E1；如果差异显著，禁止后续使用 naive gray 作为实物输入模型。若无差异，则保留其诊断用途，不提升物理身份。

### E3：DMD temporal PWM

- **目的**：验证真实位平面时间积分和相机曝光如何形成灰度响应，评估以平均复振幅代替时序积分的误差。
- **输入/固定条件**：E3-A `global-exposure-equivalent diagnostic` 假设所有 sensor pixels 共用同一曝光窗，I=∫|H[D(t)E]|²dt。E3-B `real rolling-shutter integration` 按第 r 行的实际开始时间积分，I_r=∫_(t_r)^(t_r+T_exp)|H[D(t)E]|²dt。同一目标图案、DMD bit-plane sequence、relay/camera 设置。
- **唯一变化因子**：引入实测 Temporal PWM 时间序列，并按对应曝光窗对强度积分；不使用平均复振幅替代时序强度积分。
- **观察与指标**：逐帧/同步波形、累计强度、等效灰度等级、功率、图样误差、频响、曝光窗内重复性和饱和。
- **通过/失败判定**：E3-A 仅为全局曝光等效诊断，不表示相机为 global shutter。E3-B 只有 camera line time、row exposure delay、frame timing、DMD bit-plane timing 和 trigger relation 均已核实后才可执行；任一未知即 BLOCKED，不得用 DLP9000 名义最大帧率猜测。只有实际序列与相机曝光窗能对齐、重复性误差可量化时才有效。
- **依赖/决策影响**：依赖 E1 与步骤7；决定是否需要 F5。若 F9000 使用外部输入或预载模式，报告对应速率，不外推到其他模式。

### E4：DMD spatial dithering

- **目的**：用 E4-A 可行性筛选后保留的 superpixel 尺寸，比较空间分辨率与图样等级的权衡。
- **输入/固定条件**：E4-A 根据实际 relay magnification、p_proj、chip pitch、MTF 和目标空间频率计算物理尺寸与带宽损失；E4-B 使用同一芯片面视场、dither duty、DMD方向、照明、relay pupil 和 camera mapping。
- **唯一变化因子**：仅改变 E4-A 选中的 superpixel 尺寸。二值子图案经相干传播和空间滤波后计算，不直接把组内均值当场振幅；若尺寸超过目标空间细节尺度，标记 `resolution_limited`。
- **观察与指标**：芯片面归一化与原始功率图、有效频率响应/MTF、输入强度 NRMSE、pattern contrast、空间细节和重复性。
- **通过/失败判定**：根据完整传播后的实测/模型图样和不确定度报告 trade-off；如果较大 superpixel 平滑低频但丢失空间细节，模型结论是分辨率换等级，而非灰度增加等同于连续振幅。如果 detector 不能解析 DMD pattern，使用大周期靶/频谱测量。
- **依赖/决策影响**：依赖 E1、E5 relay映射和步骤7；限制可用 superpixel 尺寸，并决定 F1/F2 是否足以表达结果。

### E5：Relay magnification

- **E5-A current relay calibration**：只标定 HW207_CURRENT 的实际 Mx/My、rotation、shear、offset、distortion residual、pupil、NA 和 throughput。必须报告几何映射与独立残差。
- **E5-B magnification diagnostic**：目的为比较 DMD 投影尺度/采样影响。M0=150/150，仅在当前两片透镜实测满足 4f 共轭后代表当前配置；M1=150/30、倍率0.2×，M2=150/~39.68、倍率约0.26455×，均为 `DIAGNOSTIC_ONLY`。M2 焦距仅为尺度估计；M1 的 30 mm 镜头当前归属输出链，调入输入端会改变/占用输出链。
- **E5-C pupil/NA/confound analysis**：变更倍率时同时检查 pupil、NA、cutoff、aberration 和 power collection。只有这些量足以匹配或控制时，才可称 `magnification_only_comparison`；若焦距变化连带改变其中任一项且无法匹配，结果标为 `relay_configuration_comparison`。
- **共同固定条件与观察**：波长、illumination、标记图案、目标芯片面和 camera 条件；报告实测 (Mx,My)、旋转/剪切/偏移、畸变残差、投影周期、频率响应、NA/cutoff、图样强度、功率和注册误差。名义 26.46/132.28/约100 周期只表示几何关系，不是有效自由度。
- **判定与依赖**：E5-A 完成后才可称 M0 当前倍率已测；M1/M2 只有相应透镜真实可用并重新标定后才能作物理诊断。对照依赖设备身份/拓扑、E1 和标定步骤5；30 mm 输出镜链变化必须随结果记录。

### E6：Polarization（PF/PBS）

- **目的**：区分偏振模块只改变总功率、改变空间图样/DMD反射效率，还是引入 PBS 损耗。
- **固定条件**：同一 laser/DMD binary pattern、波长、光路、输入功率参考面、relay、曝光和 ROI；BS 按确认后的实际拓扑固定。P0=no PF/no PBS，P1=PF only，P3=PF+PBS；仅在实际光路允许且 PBS 不承担不可替代转向功能时增加 P2=PBS only。
- **唯一变化因子**：偏振模块配置。每组测 DMD 入射、DMD ON-state、芯片面和 CMOS 接收功率，采用 all ON/OFF、horizontal/vertical binary grating、checkerboard、binary random、binary aperture。若移除 PBS 改变几何，须补偿或标记 `configuration_comparison`，不得称偏振单因素实验。
- **观察与指标**：s/p端口功率、消光比、总效率、空间 NRMSE、均匀性、方向 contrast/MTF、DMD ON/OFF 对比、偏振态和不确定度。
- **通过/失败判定**：功率变化超过不确定度但空间图样稳定，支持 scalar throughput；图样或方向频响有可分辨改变，支持增加偏振/角度相关的接口；无法测偏振或对准不可重复时保持未分辨。不得把 HES6001 论文链的偏振角度移植到 207。
- **依赖/决策影响**：依赖步骤2、6、8。只有数据需要时再做 Jones matrix，避免无证据增加复杂度。

### E7：Detector

- **目的**：从理想强度面逐层量化 sensor pixel integration 和 GSENSE2020 相机读出的影响。
- **固定条件**：使用相同的芯片输出场、output relay、视场、实际倍率和原始功率；相同相机 ROI、binning、曝光、增益、位深。先在同一输入面离线比较理想强度和像素积分，再叠加实测传感器参数。
- **唯一变化因子**：探测器模型层级（理想强度 → 像素积分 → QE/电子计数与噪声 → 满阱/ADC/ROI/binning）。不要同时改变场/relay/输入。
- **观察与指标**：相机 W/pixel、期望电子数、噪声与饱和、像素响应、输出强度 NRMSE、能量偏差、动态范围、camera measurement-chain MTF/可采样频率，并与 DMD/relay optical transfer 分开。
- **通过/失败判定**：只有 532 nm QE、camera gain/noise、full-well/bit depth 和 output magnification 关闭冲突后才称 F6 硬件模型。sensor pitch 为 6.5 μm，芯片等效采样 p_object=6.5/M_out μm；对 2 μm target pitch，M_out≥3.25 只是采样下限，不保证分辨率。若 M_out<3.25 标记 `undersampled`，使用更高倍率、long-period targets 或 Fourier-plane diagnostics；不得声称解析单个 neuron/DMD pixel。
- **依赖/决策影响**：依赖步骤9、E5和 E1；决定相机采样是否成为输入/输出接口限制，不给生成质量结论。

### E8：Frozen-weight substitution

- **目的**：权重完全不更新，只把 ideal input interface 替换为经测量校准的 HW207 输入接口，观察这个指定模型/checkpoint 的固定网络输出响应。
- **固定条件**：明确 checkpoint/step、网络配置、目标输出/映射、输入样本或 pattern、芯片传播、相机参数和预先登记的人工样本数；每个输入记录 `input_encoding_id`，只允许 `IDEAL_CONTINUOUS`、`BINARY_THRESHOLD`、`PWM_REAL` 或 `DITHER_SELECTED`，并说明连续输入如何变为实际 DMD command。同组所有权重保持冻结。不调用 evaluator，不以未批准的语义分类器给出结论。
- **唯一变化因子**：输入接口模型从理想 amplitude 变为 HW207 标定 DMD/relay 输出来到的物理输入。权重、数据角色、功率/归一化规则和读出模型不改。
- **观察与指标**：物理功率/能量偏差及归一化图样同时报告；在已指定任务图像条件下可记录 MSE/PSNR/SSIM/normalized correlation。仅对少量固定输入的输出响应，不外推为完整生成能力。
- **通过/失败判定**：执行前必须确定 checkpoint、样本数、决策用途和接受阈值；否则阻断。完整有效数据越过预先阈值才可称该窄范围有效失败。人工审查只回答固定权重接口是否值得继续讨论，不代表硬件部署或生成质量验收。
- **依赖/决策影响**：依赖 E0–E7 对使用中的物理接口完成标定并完成步骤10的冻结状态检查。E8完成后进入人工审查，不自动进入 E9。

### E9：Hardware-aware retraining

- **目的**：如 E8 及人工审查显示有必要，单独设计硬件感知训练是否可行。
- **固定条件/唯一变化因子**：现在尚未定义。将来必须由新计划明确训练数据角色、模型和权重初始化、更新参数组/预算、物理模型、训练目标、验证/确认集及唯一最终 gate。
- **观察指标**：只在新计划中定义，可能包括物理接口指标和语义生成质量；不能从本文件默认继承训练阈值、样本或 evaluator 使用授权。
- **判定与依赖**：当前状态为 **未授权 / 不执行**。仅当 E0–E8 有效、人工审查完成且用户明确批准独立训练计划后，才可计划。E8 和 E9 严格分开，不可混合成一个结果。

## 依赖图与执行优先级

1. **设备身份 → 光路拓扑 → 距离/角度/功率**：没有这三项，不得给 HW207 几何命名。
2. **DMD 静态 Binary → Relay/仿射校准 → E1**：建立当前输入物理基线；E0 与 E1 的比较才有意义。
3. **Binary 基线 → PWM/dither → E3/E4**：真实控制时序未知时，PWM 组阻断；空间 dither 以二值图案传播，不以平均值代替。
4. **Binary + 固定功率参考 → P0/P1/P3（条件允许时加 P2）→ E6**：先辨识简单透过率/图样影响，再决定是否需要偏振场模型；若 PBS 拆除改变几何，须补偿或降级为 configuration comparison。
5. **输出倍率 + 传感器标定 → E7**：未解决相机参数冲突前不拟合 F6。
6. **E0–E7 + 固定权重/消费者 + 预登记人工样本数 → E8 → 人工审查**。
7. **E8 人工审查 + 新的明确用户批准 → E9 另案计划**。

不进行一开始就把倍率、dither、偏振和曝光全部组合的大规模全因子扫描。先做上述主效应；只有当单因素结果显示需要解释某个交互时，另行缩小交互实验并写出其决策用途。

## 当前未决项

- 207 原始器件表、台面照片、光束方向、器件位置和各实际距离/角度；机械尺寸不能替代光学共轭、实际 relay Mx/My 和仿射映射残差。
- 激光完整型号、扩束后 beam diameter / 2D intensity profile / ellipticity / collimation，以及 DMD 入射功率与角度。
- F9000 控制器、固件、DMD bit-plane 时序、camera line time、row exposure delay、frame timing 和 trigger relation；因此 E3-B 当前 BLOCKED。
- 两片 f=150 mm 是否实现当前 4f relay、实际倍率/NA/pupil/throughput；30 mm 输出镜是否仍在输出链。身份和距离未闭合时 E1 实物芯片面结论及 E5-A 当前几何仍 BLOCKED。
- PF 缩写和型号对应的真实器件功能；PF/PBS/BS 端口与偏振响应。移除 PBS 是否改变几何也待实测确认。
- GSENSE2020 相机整机型号、M_out、532 nm QE、读出模式、满阱 13.8 ke⁻ / 43 ke⁻ 冲突、增益/噪声和 sensor plane 映射；这些未闭合时 E7/F6 BLOCKED。
- 100×100、2 μm 芯片目标是否有最终版图/实物，以及 HW207 中实际安装的 chip 是否相同；芯片缺席时只允许 `HW207_TARGET_INPUT_PLANE` relay 标定，不构成 LightGen 芯片实验。


当前阻断关系：组件距离/参考面、relay 实际倍率和映射未确认前，HW207_CURRENT 的 E1 实物芯片面结论及 E5-A 仍 BLOCKED；芯片缺席时只能做 `HW207_TARGET_INPUT_PLANE` relay 标定，不能称真实芯片实验。camera line time、row exposure delay、frame timing、DMD bit-plane timing 或 trigger relation 任一未知时 E3-B BLOCKED。M_out、532 nm QE、实际读出模式、满阱/增益/噪声尚未闭合时 E7 的 F6 硬件模型 BLOCKED；M_out<3.25 时对 2 μm target 标记 `undersampled`。

在这些项完成或被明确圈定前，实验结果只能回答其有效测量范围，不能声称完整 HW207 光路已验证，也不启动主线修改、模型训练或物理部署。
