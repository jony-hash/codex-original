# HW207 光路数字孪生标定计划

## 状态与适用范围

本文件规定 HW207_CURRENT 由“任务记录/规格”转为“可用于实物模型”的标定顺序；若目标芯片尚未制造或安装，则先使用 `HW207_TARGET_INPUT_PLANE` 闭合 DMD→relay→目标输入面。它是计划，不是测量记录。本轮未接触实验台、未生成数据、未运行 DMD/相机，也未进行网络推理或参数更新。仪器和器件是否在实验室可用尚未核实；以下仪器均是建议的仪器类别。

硬件身份、参数来源和 F0–F6 模型定义见[数字孪生设计总览](207_optical_twin_design.md)。

只有按顺序闭合相应参数，才能解锁依赖它的下一步。若设备身份、光路、曝光同步、坐标标记或量测精度未知，记录 UNKNOWN / CALIBRATION_REQUIRED 并暂停该依赖项。不得用论文规格、相似型号或名义倍率补齐实测值。

## 标定顺序

| 顺序 | 标定任务 | 方法与建议仪器 | 必须记录 | 完成条件及下游影响 |
|---:|---|---|---|---|
| 1 | 硬件身份 | 对每件器件拍摄完整标签、接口/端口和安装状态；抄录制造商、型号、序列号、修订版。DMD 还记录控制器板、固件、DMD 芯片料号、图案输入方式与软件版本；相机记录整机型号、传感器、固件、读出模式。 | Hardware ID、标签原文、来源链接/页码、照片路径、序列号、配置/固件版本、状态。 | 关键光源、DMD、relay、相机身份都能唯一对应到实物或清楚标 UNKNOWN；型号来源冲突已列出。DMD 时序与相机标定不得依赖未确认的控制器型号。 |
| 2 | 光路拓扑和方向 | 由实验人员沿实际光束逐段确认；用台面照片、安装图、器件端口和可见光低功率对准记录闭合 Mirror、expander、PF/PBS/BS、DMD 的入射/出射路径。不得根据名称/单张照片推断端口。 | 每段起点/终点、局部坐标、波束方向、反射/透射端口、组件先后、实际位置和照片视角。 | 所有器件与光束方向有连续、无歧义的证据；不能确认的节点标 UNKNOWN 并阻断受影响测量。由此决定 P0 组中 BS 是否固定保留。 |
| 3 | 长度、角度、入射与功率 | 轨尺/卡尺作粗测，光学导轨刻度或位置量具记录器件基准面，测角仪/反射几何确定角度；校准波长适配的功率计/探头依次测 laser 出射、DMD 入射、DMD ON-state、LightGen 输入和 CMOS 端功率。以安全低功率完成几何核查。 | 量具型号/校准日期、距离定义的表面/主平面、角度参考轴、波长、偏振、重复读数、功率范围和不确定度。 | 每个距离和角度都指出基准面/符号/单位；功率读数带有仪器范围和不确定度。只有标称功率或未校准表读数时，保持 PROVISIONAL。 |
| 4 | DMD 几何和静态二值响应 | 用已知二值图案（全黑/全 ON、水平/垂直光栅、checkerboard、binary random）确认图像朝向和 ON/OFF 收光。用显微镜、已标尺相机或频谱法确认 pitch 投影、有效区、像素方向、可见杂散级次。传感器像素不足以采样时用显微物镜/高倍率成像或 Fourier-plane 测量。 | Pattern 文件/哈希非必需，记录图案 ID、bit depth、控制模式、曝光/触发、波长、照明角、光阑、中心位置、采样尺标、功率和采集帧。 | 至少能区分 DMD 状态命令和光学收集结果；pitch/朝向与官方芯片型号相符或偏差被说明。数据表 fill factor 可引用为规格，不能伪装成实验实测。 |
| 5 | 输入 relay 倍率、pupil 和映射 | 在 DMD/芯片共轭面使用已知周期靶或网格；用经标定的成像镜头/显微镜和相机采集。验证实际物像共轭、两镜主平面距离、有效倍率、清孔径、光阑、NA、旋转、偏移、剪切、场曲/畸变。用覆盖全视场的非共线 fiducials 拟合二维仿射变换并保留独立残差。 | 两透镜型号、焦距/公差来源、主平面与间距、光阑和 NA 定义、拟合系数矩阵、点集坐标、训练/验证点分组、残差分布、重复性。 | 仿射模型在预先设定的量测误差内描述全视场；如果存在有结构的残差，再考虑高阶模型。倍率只作仿射矩阵的摘要，不得代替残差。M0 完成后才能称为“当前倍率已测”。 |
| 6 | DMD Binary 基线 | 固定光源、光路、曝光和芯片面坐标，主对照仅使用 binary-compatible pattern：all ON、all OFF、horizontal/vertical binary grating、checkerboard、binary random、binary aperture。Gaussian illumination 用 DMD all ON 只测入射光束包络；binary Gaussian aperture 按预先固定阈值二值化；gray Gaussian 只进入 PWM 或已选定的 spatial dithering。 | 各面 raw power、未归一化强度图、背景/暗帧、ROI、坐标、对比度、均匀度、频响/MTF、裁切和重复测量不确定度；每种图案/转换使用独立 pattern ID 与 encoding ID。 | 可形成可重复的 HW207 binary 参考数据；黑/白状态可区分且测量范围未因曝光饱和/噪声失效。未编码的连续灰度 Gaussian 不得作为静态 Binary DMD 对照。失败时先修复采样/照明/对准，不进入灰度模型归因。 |
| 7 | 灰度编码比较 | 先完成控制器/固件/同步信息核验。E3-A 为共享全局曝光窗的诊断积分；E3-B 为真实 rolling-shutter 行积分，只有 line time、row exposure delay、frame timing、DMD bit-plane timing 和 trigger relation 全部已知才可执行。E4-A 先按 M、p_proj、chip pitch、MTF 和目标空间频率筛选候选 superpixel；E4-B 只测选定的可行尺寸。另以 DMD-B naive gray 作双标签负对照。 | 控制器配置、逐帧/位平面时间、曝光窗、同步信号、图案 duty、选定 superpixel、空间频率/有效等级。 | E3-B 任一时序参数未知即 BLOCKED，不得用 DLP9000 名义最高帧率猜测。E4-A 若候选超出目标空间细节尺度则标 `resolution_limited`，不把它解释为单纯灰度提升。所有 dither 需经传播/滤波后比较，不能以子像素占空比宣称场振幅。 |
| 8 | PF/PBS/BS 偏振和透过 | 先记录偏振器方向；使用已确认的 polarizer/analyzer 或偏振计（可用性需确认）和 532 nm 功率计；P0=no PF/no PBS，P1=PF only，P3=PF+PBS。只有实际光路允许且 PBS 不承担不可替代的转向功能时，才增加 P2=PBS only。BS 按已核实拓扑固定。每次改组后重新维持光束中心/角度；若移除 PBS 改变几何，须补偿或标记 `configuration_comparison`。 | 入射偏振态、器件方位角、s/p 标识、各端口功率、消光/漏光、beam profile、光栅方向响应、重复误差。 | 几何未补偿的 PBS 拆除对照不能称为偏振单因素实验。若只有总功率差超过重复不确定度而空间分布/频响无可测变化，先保留 scalar loss/throughput 模型；若图样也变化，再拟定偏振/Jones 模型需求。低精度测量只得“未分辨”，不等于无偏振效应；不推断 Science 论文 PF/PBS 角度。 |
| 9 | CMOS 标定 | 确认整机型号和读出模式。采 dark frames、平场和 532 nm 已标定光功率序列；测曝光响应、系统增益、read noise、dark current、线性区、饱和/满阱、bit depth、ROI/binning、像素响应及 output relay 倍率。QE 由可追溯 photon-transfer/光子通量方法或准确响应曲线获取。 | 每个曝光、增益、温度、读出模式、位深、ROI/binning、暗帧/平场、像元面积/倍率、功率计读数、电子转换增益、均值-方差曲线、置信/不确定度。 | 532 nm QE、噪声、满阱/裁切及 sensor plane 空间映射的来源互相一致；解决 13.8 ke⁻ 与 43 ke⁻ 冲突。未闭合时 F6 和 E7 只能标未定，不可报真实相机预测。 |
| 10 | Frozen-weight substitution | 仅在物理接口参数有效、固定模型和 checkpoint 明确、数据角色与唯一比较指标预先登记后，保持权重不变，把理想输入接口替换为标定后的 HW207 输入。对比原始功率和归一化图样；不改权重、不调用 evaluator。 | 固定权重身份/步数、唯一输入/输出映射、测量样本与图案集、原始输出、MSE/PSNR/SSIM/相关性/能量偏差等预先选定项、人工审查用途。 | 结果只回答这个冻结模型/接口的替换范围。没有物理校准或消费者定义则阻断；物理接口可用不等于生成质量可接受。进入正式审查前预先定义人工样本数和接受目的。 |
| 11 | 人工审查 | 对 E8 预先指定的一组原始/归一化输出和对应功率、注册、饱和信息进行盲化或并列审查。 | 审查样本数、样本选择方法、评价问题、评审者、对照和决策用途。 | 审查只能决定 E8 当前接口/固定权重是否进入后续研究讨论；不能回写训练成功或泛化结论。 |
| 12 | Hardware-aware retraining | 本轮禁止。只有前序物理接口工作和 E8 审查完成后，单独提交训练计划，明确训练数据/角色、模型与 checkpoint、更新范围/预算、gate、预检和停止点。 | 独立批准和计划字段。 | 无用户批准前不启动；本文件不提供训练授权。 |

## 参考面、几何和入射光束记录

每个组件的几何记录至少包含以下字段：

| 字段 | 定义 |
|---|---|
| component_id | 组件的唯一身份。 |
| reference_surface | 组件几何记录采用的明确物理参考面。 |
| x / y / z | 相对本地坐标框架的位置，单位和原点一并记录。 |
| theta_x / theta_y / theta_z | 绕对应轴的角度、正方向和单位。 |
| uncertainty | 各位置/角度量的不确定度。 |
| measurement_method / instrument_id | 测量方法、仪器型号与可追溯身份。 |

参考面按物理定义填写：DMD 使用 active micromirror plane；透镜已知时使用 principal plane，若只能测机械表面则明确标成 mechanical reference，不得称为主平面；LightGen 使用 target input plane 或 physical active plane；CMOS 使用 photosensitive sensor plane。

机械几何（台面、镜架、组件机械表面）与光学几何分开记录。光学几何需通过共轭、focus、magnification 和 fiducials 验证；机械位置测量不能单独证明光学共轭。

入射光束至少记录 beam diameter、beam center、2D intensity profile、ellipticity、collimation、DMD incidence angle 和 polarization state。完整 phase/wavefront 测量是可选的更高保真标定，不是 Binary 第一阶段的强制条件。

## 芯片缺席时的目标输入面模式

若实际 100×100、2 μm LightGen 尚未制造或安装，标定模式命名为 `HW207_TARGET_INPUT_PLANE`。可在预计芯片输入位置放置 calibration target，使用 microscope、high-magnification camera 或 Fourier-plane diagnostic，先闭合 DMD→relay→target input plane。记录该替代靶标及采样参考面；结果只证明输入面 relay 标定，不得描述为真实 LightGen 芯片实验。

## E5 Relay 拆分与归因

- **E5-A current relay calibration**：只测 HW207_CURRENT 的实际 Mx/My、rotation、shear、offset、distortion residual、pupil、NA 和 throughput。
- **E5-B magnification diagnostic**：在条件允许时比较 M0=150/150、M1=150/30、M2=150/~39.68；M1/M2 保持 `DIAGNOSTIC_ONLY`，不是已安装状态。30 mm 透镜当前记录为输出镜头，转作输入端会改变/占用输出链。
- **E5-C pupil/NA/confound analysis**：若变更焦距同时改变 NA、pupil、cutoff、aberration 或 power collection，且不能匹配，则结果标为 `relay_configuration_comparison`，不得称 `magnification_only_comparison`。

## E3 时序和相机曝光窗

E3-A `global-exposure-equivalent diagnostic` 假设所有 sensor pixels 共用同一曝光窗，积分量为 `I=∫|H[D(t)E]|²dt`。E3-B `real rolling-shutter integration` 按行使用 `I_r=∫_(t_r)^(t_r+T_exp)|H[D(t)E]|²dt`。只有 camera line time、row exposure delay、frame timing、DMD bit-plane timing 与 trigger relation 均有实测或可核查记录时才解锁 E3-B；未知任一项即 BLOCKED，不用名义最大 DMD 帧率推算。

## CMOS 目标面采样条件

GSENSE2020 传感器 pitch 记录为 6.5 μm。以输出倍率 M_out 计算芯片面的等效采样 p_object=6.5/M_out μm。对 HW207_TARGET 的 2 μm 节距，几何下限为 M_out≥3.25；这只是每个目标节距至少一个 sensor sample 的下限，不保证分辨率。实际还受 optical MTF、sensor MTF、aberration 和 SNR 限制。若 M_out<3.25，标记 `undersampled`；改用更高倍率、long-period targets 或 Fourier-plane diagnostics，不声称相机解析单个 neuron 或 DMD pixel。

## E8 输入编码身份

E8 每个输入额外记录 `input_encoding_id`：`IDEAL_CONTINUOUS`、`BINARY_THRESHOLD`、`PWM_REAL` 或 `DITHER_SELECTED`。记录从连续目标输入到真实 DMD command 的具体编码步骤与参数；只写“SLM→DMD”不足以定义输入接口。

## 建议设备类别（实际库存待确认）

| 用途 | 建议设备类别 | 关键要求 |
|---|---|---|
| 功率 | 532 nm 适用且校准有效的功率计/探头 | 量程覆盖激光输出和各级损耗后功率；记录不确定度和探头位置。 |
| 光斑/成像 | 光束分析仪、带标尺显微成像或高倍率相机 | 要能覆盖目标周期/边缘；低倍率 CMOS 不能直接认定解析了 1.512–2 μm 投影细节。 |
| 空间频响 | 靶标、网格、周期光栅、显微物镜或 Fourier-plane 成像 | 记录视场、焦面、采样标尺和仪器 MTF。 |
| 偏振 | 偏振计或经标定的偏振片/分析器组合 | 能在实际光路端口测入射偏振、s/p 和消光；未知器件功能先查标签。 |
| 时序 | F9000/DMD 控制软件日志、触发输入输出、光电二极管/示波器 | 实测位平面时间、重复率、灯/激光同步和相机曝光窗。 |
| 相机 | 完整型号手册、暗场/均匀光源、已标定光功率、成像标尺 | 输出 QE/增益/噪声/满阱/线性和像面倍率；不能仅用传感器网页。 |

上述库存和是否经过计量校准都未确认。需先盘点，缺少合适设备时再决定是借用、外测还是把相应项保留 UNKNOWN；不默认采购或启动外部服务。

## 数据记录格式

每个采集批次需有一份可读记录（YAML/CSV 均可）及未修改的 raw 数据路径。建议字段：

| 字段 | 内容 |
|---|---|
| calibration_id / configuration_id | 如 HW207_CURRENT/DMD_BINARY/M0，唯一指向一次具体条件。 |
| 日期、操作者、实验台 | 时间、人员、台位/温度等环境信息。 |
| hardware_id / model / serial / firmware | 对应唯一器件和控制器、相机读出配置。 |
| source_ref / status | 参数来源、页码/标签/实测 ID；CONFIRMED、DATASHEET、MEASURED、PROVISIONAL、UNKNOWN、CALIBRATION_REQUIRED、DIAGNOSTIC_ONLY 之一。 |
| plane_id / coordinate_frame | 物理面、原点、轴向、单位、波长、网格大小和 pitch。 |
| method / instrument_id | 仪器型号、序列号、校准日期、设置和测量方法。 |
| value / unit / uncertainty | 数值、单位、统计/仪器不确定度和样本数；未测必须显式为 UNKNOWN。 |
| pattern / timing / exposure | pattern ID、bit depth、PWM/dither 模式、触发和曝光/ROI/binning。 |
| input_encoding_id | IDEAL_CONTINUOUS、BINARY_THRESHOLD、PWM_REAL、DITHER_SELECTED 之一；记录连续目标如何映射为实际 DMD command。 |
| raw_data_path / processing | 原始图像/波形/表格路径及可复现处理说明。 |
| notes / anomalies | 饱和、背景、移位、光斑漂移、缺帧、设备冲突等。 |

同一物理量如有多个源头或不同统计口径，保留多行和冲突，禁止覆盖旧记录或只留“最终选值”。测量记录中的 digest 可作为来源的附加 provenance，但不作为准入锁或成功条件。

## 标定完成判定

一个参数只有在设备身份和测量面明确、单位/方法可追溯、重复性和量测不确定度被报告、结果与来源不冲突、原始数据可定位时，才可从 UNKNOWN/CALIBRATION_REQUIRED 更新为 MEASURED。数据表参数标 DATASHEET，设计目标留 PROVISIONAL，诊断假设标 DIAGNOSTIC_ONLY。

物理测量的“完成”不等于模型通过或实验成功。每个需要做选择的数值阈值（例如可接受注册误差、MTF、功率损耗或输出图样差异）必须在对应实验前，依据任务用途由负责人登记；本计划不代设性能接受阈值。若测量误差大于所要分辨的效应，结论为未分辨/阻断，不推断“没有效应”。

## 第一批实测优先级

1. 207 实物标签和完整光路拓扑，特别是 F9000/DLP9000、控制器/固件、DMD 入射/ON-OFF 收光方向。
2. 激光完整型号、DMD 入射功率、扩束后二维光斑直径/均匀性。
3. 两片 150 mm relay 的主平面间距、实际倍率、有效清孔径/NA、DMD 到芯片面的仿射配准和残差。
4. DMD 实际控制模式和时序/硬触发同步；先完成静态 binary，再决定 E3 能否开展。
5. 相机完整型号和读出模式、输出倍率、532 nm QE、实际满阱/增益/噪声及满阱规格冲突的来源。

PF/PBS/BS 的透过率和偏振状态在拓扑核验后、binary 基线稳定后测。最小的首轮解锁点是器件身份、拓扑、尺度映射和 DMD Binary 有效采集；未经这些确认，不进入灰度、偏振/相机拟合、E8 或 E9。
