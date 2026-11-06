# HW207 MNIST4 C3 全光路数字孪生状态报告

## 结果与状态

总体状态为 **`C3_PARTIAL_PASS`**。C3 完成了静态二值 DMD 到 LightGen 目标输入面的空间几何映射、三个理想 relay 候选的坐标/倍率换算，以及无相机噪声的 CMOS 像元面积积分组件核验；C2 原始 **`C2_PARTIAL_PASS`** 保持不变，C2.1 空间几何诊断已完成。

| 子项 | 状态 | 本轮可支持的范围 |
|---|---|---|
| DMD 标量 Binary 图案与目标面几何 | `NUMERICAL_DIAGNOSTIC_PASS` | 只在二值、ON=1/OFF=0、均匀照明及方形有效微镜假设下 |
| DMD 相干微镜/衍射级响应 | `BLOCKED` | 无真实倾角、入射向量、ON/OFF 收光端与相干场传输模型 |
| 输入 Relay 实际几何 | `BLOCKED` | 计算了理想 4f 候选；另有 pupil 情境仅为不具实物依据的诊断 |
| 固定 LightGen 物理层 | `BLOCKED` | 无已绑定的实物/打印结构及复透射函数 |
| 输出 Relay 几何 | `THEORY_ONLY` | 仅有坐标变换候选，不是实测倍率/像差/NA |
| CMOS 像元面积积分代码组件 | `NUMERICAL_DIAGNOSTIC_PASS` | 积分矩阵的几何/能量数值测试通过；不是相机标定 |
| CMOS 噪声、QE、增益、读出 | `BLOCKED_PARAMETERS_UNKNOWN` | 缺少可用于 532 nm 的整机参数 |
| CMOS 2×2 四区分类 | `BLOCKED_NO_LIGHTGEN_OUTPUT_FIELD_OR_CALIBRATED_ROI` | 未产生受支持的 LightGen 输出场或实体 ROI 映射 |
| HW207 当前实物映射 | `BLOCKED` | 当前 DMD→LightGen→CMOS 端到端传递仍无闭环证据 |

因此，**本轮没有得到全光分类、实物分类或 CMOS 四区准确率**。`C3_PARTIAL_PASS` 只表示若干数值组件按定义完成；不能表述为现有 HW207 光路已验证。

2026-11-04 attempt06 将三个理想输出 Relay 候选接到 2048×2048、6.5 μm 虚拟 CMOS 面积积分。对 100×100、2 μm 网格上的人工均匀场，三种候选均落入名义视场，积分能量闭合误差为 0；四个全幅象限积分相同并按规则拒判。该项只验证输出坐标、像元面积积分及 ROI 消费接口，不产生 MNIST 结果，也没有 LightGen 输出场。

实验矩阵中的 C3-D/E/F 相干微镜传播条件没有完成，保持 `BLOCKED`：实装微镜入射角、ON/OFF 收集端、照明/接收 NA 等不足以定义可解释的相干场。已有 pupil 灵敏度结果使用显式假设参数，只是 relay 孔径诊断，不替代这些相干条件。

### 器件参数等级与来源

| 项目 | 当前记录 | 来源等级与边界 |
|---|---|---|
| DMD 芯片几何 | DLP9000 系列：2560×1600、7.56 μm pitch、±12° mirror tilt、92% fill factor、400–700 nm 设计波段 | [TI DLP9000 datasheet](https://www.ti.com/lit/ds/symlink/dlp9000.pdf) 芯片级规格；它没有确认当前装机身份、倾角端口或 532 nm 实际衍射效率。TI 说明 DLP9000 需双 DLPC900，DLP9000X 则配 DLPC910。 |
| DMD 图案时序 | DLP9000 1-bit 最高 9523 Hz；8-bit 预载并配合照明调制时最高 1031 Hz；外部输入 8-bit 最高 360 Hz | TI 数据表中的器件模式指标；不得当作 PC 可实时传送不同图案的保证，也不把 1031 Hz 当作外部输入刷新率。 |
| F9000 整机信息 | 用户提供规格图称 DLPC900 + DLP9000、400/40 张预载图案、USB 1.1/HDMI/DisplayPort 与同步支持 | 用户图像/供应商页面参数声明。[FLDiscovery F9000 页面](https://www.fldiscovery.com/product/f9000/)本轮返回 HTTP 403，400/40 与接口信息未能从页面独立复核；F9000 实机身份、控制器数、固件模式与当前安装状态均为 `UNKNOWN`。 |
| CMOS 候选 | 用户截图标注 Gpixel GSENSE2020、2048×2048、6.5 μm；截图给出的峰值 QE 为 595 nm 下 72%。这些组合与鑫图 Dhyana 401D 的规格字段高度匹配 | 用户提供的规格截图与厂家规格书交叉匹配，不证明现场整机身份；仍需铭牌/设备枚举确认。72% 只对应 595 nm，不能用作 532 nm QE。实际 QE、读出模式、曝光、增益和噪声仍为 `UNKNOWN`；现有满阱值 43 ke⁻ 与 13.8 ke⁻ 冲突，模型不任选其一。详见[实物与闭环基线](../../../research_assets/lightgen/HW207_实物与距离基线_20261103/实物信息与闭环状态.md)。 |

设备供应资料只固定模拟输入的候选尺度和参数来源。该表不把供应商/用户规格升级为 HW207 当前硬件实测值。

TI 的 ±12° 是微镜相对平态的器件倾角；几何计算中的 0°/45° 是 DMD 阵列在实验坐标系中的安装旋转，两者独立。DMD 通过反射方向调制光，模型中的 `OFF=0` 仅是标量代理约定，不代表实际 OFF 态无光或其反射光已被排除。

原始[器件清单工作簿](../../../research_assets/lightgen/HW207_实物与距离基线_20261103/05_器件参考/LightGen光路-1014-v5.xlsx)已作为来源核对；其中两个工作表的 30 mm 镜数量分别记为 2 和 3（150 mm 镜均记为 2）。本报告保留这项清单差异，不据其推定当前已安装镜片或镜片位置。

## 科学认识

这次数字孪生覆盖到 DMD 图案经假设坐标变换与有效面积映射后，在 100×100、200×200 μm LightGen **目标输入面**的二值相对图样。输出端只单独验证了“已知输入平面强度如何积分到有限 CMOS 像元面积”的数值组件；DMD 反射级次、实际输入 Relay、LightGen 层传播、输出 Relay 误差、CMOS 响应没有串成一个物理前向算子。

新增输出接口仍以人工均匀场为输入，并采用候选理想 4f 变换、中心零偏移、轴对齐坐标。输出倍率保留负号，因此图像旋转 180°；独立点斑反例显示左上输入经 −1× 映射后到 CMOS 右下象限。A/B/C 的投影宽度为 200/1,000/1,333.333 μm，名义 CMOS 等效采样约为 0.308/1.538/2.051 像素/目标单元。实际倍率、偏移和方向未知，不能据此冻结实物 ROI。均匀场产生平局，验证的是拒判逻辑，不表示四类信号编码或分类成功。

结合用户设定的“预留位放置一层双光子打印超表面”方向，该层仍是 **HW207_TARGET/拟验证架构**：仓库证据未提供最终版微结构、打印文件、实际层数、复振幅/相位透射图、材料与加工批次、层面波长响应或对准结果。仿真不能用一个理想 phase mask 代替这些未知项并将结果命名为 HW207 实物验证。

三个理想 4f 候选的空间比较和冻结 C1 读出见[C2.1 报告](hw207_mnist4_c21_spatial_alignment_report.md)。其 C1 头准确率只在 validation 上读取模拟的 DMD→LightGen 输入图样，且 C1 也用该 validation 选择过数字读出；该读出既不是光学算子，也不是 CMOS 输出。DMD/CMOS 参数等级见[实物与闭环基线](../../../research_assets/lightgen/HW207_实物与距离基线_20261103/实物信息与闭环状态.md)及[本轮显式配置](../../../configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json)。

### 计划问题逐项结论

| # | 问题 | 本轮证据支持的结论 |
|---:|---|---|
| 1 | 现有 1×/1× 配置能否完成四分类？ | 不能据本轮证据判定。C3-A 只得到输入图样几何映射与冻结数字头辅助读出，没有 LightGen 输出场或固定 2×2 ROI 预测。 |
| 2 | 交叉 Relay 是否改善输入和输出空间采样？ | 在理想倍率假设下，B 将 DMD 投影节距从 7.56 μm 缩至 1.512 μm、候选 CMOS 像素/目标网格从 0.308 提至 1.538；但其微镜对齐形态优于 A 的结论仅限模拟映射，实际镜片分配和成像性能未知。 |
| 3 | 空间匹配方案是否值得继续加工？ | C 是后续设计候选：投影节距 2.016 μm 接近 2 μm 目标格点，输出坐标比例约 2.051 候选像素/格点。40 mm 与 200 mm 镜的实装、复透射数据和制造容差均未知，因此本轮不构成加工放行。 |
| 4 | DMD 微镜空间粒度是否匹配 MNIST？ | 若把 200 μm 分为 28 个 MNIST 像素，名义间距为 7.143 μm；A/B/C 分别约为 0.945/4.724/3.543 个 DMD 周期每个 MNIST 像素。A 接近一周期/像素，B/C 在模型上过采样；这不等于 DMD 的光学分辨率。 |
| 5 | DMD 到 LightGen 是否存在严重空间信息损失？ | 仅对二值图样的几何重采样作了诊断：B/C 的模拟 NRMSE、IoU 优于 A 的部分条件。相干微镜衍射、ON/OFF 收光、真实孔径和 relay 像差缺失，实际光路的信息损失未知。 |
| 6 | LightGen 输出到 CMOS 是否存在采样瓶颈？ | 只能给出倍率候选下 A/B/C 约 0.308/1.538/2.051 个候选像素/目标格点的坐标比例。没有 PSF、NA、像差及实际倍率证据，不能判断真实采样瓶颈。 |
| 7 | 2×2 ROI 能否稳定区分四类？ | 本轮未执行。无 LightGen 输出场、已标定输出 relay 或冻结实物 ROI，也无重复帧；因此没有 ROI 分类指标或稳定性结论。 |
| 8 | 当前主要瓶颈来自 DMD、Relay、LightGen 还是 CMOS？ | 不能归因或排序。各环节没有被同一实测前向链连接，且关键传递参数未知。 |
| 9 | 哪些结论仅是数值诊断、不能推广到实物？ | A/B/C 倍率、像素/微镜覆盖、nearest/area 形态误差、冻结头验证读出和 CMOS 面积积分均依赖明示假设；它们不验证 DMD 收光、实际 Relay、打印层、相机响应或 HW207 当前映射。 |

因此，C3-A/B/C 均未通过 2×2 光学四分类验证；B/C 的模拟几何优势只能用于后续候选比较，不能据此选择实物方案或进入加工。

## 目标缺口

端到端 C3 需要同一坐标/方向定义下的实际输入和输出关系，还需要能被前向模型消费的固定 LightGen 复透射/传播描述。目前仍缺：

- DMD 实物铭牌/控制器身份、安装角、镜面实际开关方向、照明偏振/角度、有效光束与收集 NA；用户规格图不能证明当前安装与端口。
- Relay 实装镜片身份、主平面位置、共轭、倍率、有效孔径、像差和二维仿射/畸变；估读尺带值不提供这些标定。
- 一层双光子打印 LightGen 的制造几何、材料/工艺和 532 nm 复透射图；若未来可调参数只在数字模型中出现，还需说明能否在制造后实现。
- CMOS 整机铭牌或枚举身份、532 nm QE、读出模式、曝光、转换增益、位深、满阱/读噪声与饱和行为。用户截图中的 72% QE 对应 595 nm，不能移作 532 nm 参数；满阱来源有冲突。
- 实际输出面到 CMOS 的坐标标定、暗场/平场、候选四 ROI 的冻结，以及稳定重复帧的区域能量、外泄漏和串扰。

报告仍保留 DMD F9000 用户规格和候选 CMOS 截图规格作为 **参数来源**，实际器件/装机身份维持 UNKNOWN。原 13 段距离按照片估读、厘米取整保留为初始化：`30,100,130,70,90,70,100,60,350,90,60,80,100 mm`，未改成精密实测。`06_return` 的 70 mm 是去程估读的同路径返回假设，不是第 14 段独立测量，也没有被当作已安装 DMD 支路。完整来源见[实物与闭环基线](../../../research_assets/lightgen/HW207_实物与距离基线_20261103/实物信息与闭环状态.md)及[13 段几何配置](../../../configs/research/lightgen/hw207_optical_geometry_v1.json)。

逐段来源如下；每个数字均为对应卷尺照片读数按厘米取整得到的初始几何估值，状态统一为 `PHOTO_ESTIMATE_NOT_CALIBRATED`，无逐段不确定度。当前没有逐段精密机械测量、镜片主平面间距、展开后的校准光程或 BFL 记录。端点是资料记录的近似光学中心/反射点，不等同于镜片主平面或 CMOS 感光面。

| 段 | 支路 | 起点 → 终点 | 初始距离 | 距离类型 / 证据等级 | 来源照片 |
|---|---|---|---:|---|---|
| 01 | 输入 | Laser 出光口 → Beam expander 入光口 | 30 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M1 |
| 02 | 输入 | Beam expander → Mirror-1 | 100 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M1 |
| 03 | 输入 | Mirror-1 反射点 → PF-1 | 130 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M2 |
| 04 | 输入 | PF-1 → PBS | 70 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M2 |
| 05 | 输入 | PBS → BS | 90 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M2 |
| 06（去程） | SLM | BS → SLM 位置（临时反射镜） | 70 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M2 |
| 07 | 成像 | BS → Mirror-2（折转镜） | 100 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M3 |
| 08 | 成像 | Mirror-2 → Relay L2 | 60 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |
| 09 | 成像 | Relay L2 → Relay L1 | 350 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |
| 10 | 成像 | Relay L1 → PF-2 | 90 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |
| 11 | 成像 | PF-2 → LightGen 芯片预留位 | 60 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |
| 12 | 成像 | LightGen 芯片预留位 → Relay lens | 80 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |
| 13 | 成像 | Relay lens → CMOS 预留位 | 100 mm | 卷尺照片估读、cm 取整 / `PHOTO_ESTIMATE_NOT_CALIBRATED` | M4 |

06 回程另在计算图中写成 70 mm，是依据 TSV 对“单程、沿同一路径返回”的文字展开的同路径假设；原表只有一个第 06 段记录，故它不是独立的第 14 个估读。真实 DMD 的 ON/OFF 反射方向和回程轴未知，不能直接照搬临时镜回路。

### 照片路径估值与理想 4f 间距并列

临时反射镜回程到 Relay L2 的照片估值路径为 `70+100+60=230 mm`；Relay L2→L1 为 350 mm；Relay L1→芯片预留位为 `90+60=150 mm`。将这三段分别与候选 4f 的物面→H1、H1→H2、H2→像面相减，只作几何差异提示：照片端点是估读的器件位置，理论端点是主平面，尚未建立一一标定关系。

| 候选输入 Relay | 照片路径估值 (mm) | 理想 4f 主平面段 (mm) | 估值减理论值 (mm) |
|---|---:|---:|---:|
| C3-A，150/150 mm | 230 / 350 / 150 | 150 / 300 / 150 | +80 / +50 / 0 |
| C3-B，150/30 mm | 230 / 350 / 150 | 150 / 180 / 30 | +80 / +170 / +120 |
| C3-C，150/40 mm | 230 / 350 / 150 | 150 / 190 / 40 | +80 / +160 / +110 |

这组差值不用于调整或覆盖原 13 段估读。输出照片只有单片 `f≈30 mm` 的候选 Relay：芯片→镜约 80 mm、镜→CMOS 预留位约 100 mm。薄透镜关系给出像距 48 mm，与 100 mm 照片估值相差约 52 mm；它与 A/B/C 两片式理想 4f 输出 Relay 的距离定义不同，不能互换使用。

## 工程完成与验证

- 建立/复用独立 C2.1/C3 入口、空间几何模块和定向测试；所有输入路径显式传入。输入沿用 C0 原始 train/validation/test 角色：22,279/2,475/4,157；C1 binary raw 与 normalized 冻结权重作为数字辅助读出。数据来源与源码 SHA256 仅用于结果溯源，不用作任务准入锁。
- 目标计算包含 DMD 的 2560×1600、7.56 μm 用户规格，对应完整阵列外形 19.3536 mm × 12.096 mm；同时比较 0°/45° 安装候选、nearest/area 两种静态二值像素映射并作有效面积裁切积分。零平移为诊断假设；`0.92` 填充率来自用户规格，但方形有效镜面形状仍是假设。候选输入 relay 以理想 affine 坐标映射和面积重采样表示，并未模拟真实透镜间的复场自由传播、瞳面截断或像差，也不含实测偏移/倾斜/剪切。
- attempt04 中 54 个 pupil 光栅情景的最大调制度如下（每个 cutoff 数值取 defocus 0/50 μm 中较大者）。本轮又用当前生产函数重算这 54 个情景，并将 0°/45° 全阵列包围框和命令图样语义写入[attempt05 补充机器记录](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_05/c21_c3_followup_audit.json)；数值与 attempt04 一致，没有重跑 MNIST 映射或 validation：

  | 候选 | 周期 | 输入命令图样 | 基频 (cycles/μm) | cutoff 0.05 | cutoff 0.1 | cutoff 0.2 |
  |---|---:|---|---:|---:|---:|---:|
  | A | 1 微镜 | 全 ON；有效孔径晶格 | 0.1323 | 0.0000 | 0.0000 | 0.2620 |
  | A | 2 微镜 | 1-on/1-off | 0.0661 | 0.0000 | 0.9993 | 0.9999 |
  | A | 4 微镜 | 2-on/2-off | 0.0331 | 1.0000 | 1.0000 | 1.0000 |
  | B | 1 微镜 | 全 ON；有效孔径晶格 | 0.6614 | 0.0000 | 0.0000 | 0.0000 |
  | B | 2 微镜 | 1-on/1-off | 0.3307 | 0.0000 | 0.0000 | 0.0000 |
  | B | 4 微镜 | 2-on/2-off | 0.1653 | 0.0000 | 0.0000 | 1.0000 |
  | C | 1 微镜 | 全 ON；有效孔径晶格 | 0.4960 | 0.0000 | 0.0000 | 0.0000 |
  | C | 2 微镜 | 1-on/1-off | 0.2480 | 0.0000 | 0.0000 | 0.0000 |
  | C | 4 微镜 | 2-on/2-off | 0.1240 | 0.0000 | 0.0000 | 1.0000 |

  1 微镜空间周期的命令是全 ON，调制度来自有限填充率的有效镜面孔径晶格；2/4 微镜周期才施加 ON/OFF 命令图样。cutoff 取旧 C2 数值情景，不是 HW207 实测 NA；这些结果属于 `UNPHYSICAL_DIAGNOSTIC_ONLY`，只核对理想频率通过/截止方向。
- 输出端建立了 `100×100 → 候选 relay 坐标 → 2048×2048/6.5 μm CMOS 候选像元面积`积分接口，并测试能量守恒、均匀场、四个等面积非重叠象限、区外能量以及零强度/并列最高拒判。它没有应用 QE、光子散粒噪声、暗电流、读噪声、增益、ADC、曝光或 532 nm 相机响应。2×2 ROI 函数接口通过测试，但没有对不存在的 LightGen 输出场执行 ROI 分类。
- attempt06 用同一生产函数生成 A/B/C 全传感器面积积分；最大相对能量闭合误差为 0，单元捕获率相对 1 的最大数值偏差小于 `4.8×10⁻¹³`。C3-C 均匀场四区积分存在约 `1.27×10⁻¹¹` 的舍入差，旧精确相等规则曾误判为类别 1。修复后按双精度累加及 `1e-12` 相对容差识别平局，并将不可为负的区外能量舍入残差清零。新增回归先复现误判，修复后 20 项 C2.1/C3 定向测试通过。
- attempt06 几何执行只复用 attempt04 已审计的 12 条件 × 2,475 张 validation 汇总；没有重新处理 MNIST。新 MNIST 行数、训练、官方 test、硬件控制均为 0。历史 C2.1 结果和 C2 的 `C2_PARTIAL_PASS` 未更改。
- 原输出 Relay 照片估读为物距约 80 mm、镜到 CMOS 预留位约 100 mm，候选 f=30 mm。薄透镜关系给出共轭像距约 48 mm、倍率约 −0.60，与 100 mm 估读不一致，因此该 100 mm 不作为成像共轭或相机倍率；C3 的 30/30、30/150、30/200 4f 输出候选都独立标成理论配置。
- attempt03 之后的本地独立复核发现旧能量账跨参考面相减错误，因此 attempt03 虽通过同实现重播自检，其能量/捕获结果已明确作废。修复后的 attempt04 用 18 项定向测试、256 张分层预检、完整 2,475 张 validation 和 12 条件再播放完成复核；官方 test 不推断。局部独立复核验证 C0 索引顺序与标签、逐条件指标/混淆矩阵、旧函数反例和修正版能量守恒；最大能量闭合相对误差 6.8×10⁻⁹，倍率面积相对误差 3.81×10⁻¹⁶，原结果与全量重播的逐样本最大差为 0。
- 对当前生产 pupil 诊断调用另做了定向回归：18 项通过、失败/错误/跳过均为 0，覆盖 A/B/C、1/2/4 微镜周期和 0.05/0.1/0.2 cycles/μm cutoff。首次执行遇到项目临时目录权限拒绝，使用新建仓库外 scratch 复跑后通过；证据见[attempt05 定向测试复跑](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_05/tests_retry2.xml)。该局部回归没有重跑 attempt04 全量映射或 validation。
- 修正版全量归档有 66 个逐文件收据，状态 `COMPLETE`；被替代的 attempt03 有独立的 23 项历史备份并标作无效旧账。pilot、validation、audit 控制日志和 RunId 也保存在 attempt04。

证据入口：[C3 状态汇总](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/c3_summary.json)、[全量验证与能量统计](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/validation_summary.json)、[本地独立审计](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_04/independent_local_audit.json)、[生产重播审计](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/independent_audit.json)、[空间映射与倍率](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/spatial_geometry.json)、[pupil 假设诊断](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/relay_pupil_sensitivity.json)、[attempt05 补充机器记录](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_05/c21_c3_followup_audit.json)、[66 项回传收据](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/transfer_receipt.json)、[旧 attempt03 收据](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_superseded_attempt_03/transfer_receipt.json)、[attempt04 原测试](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/autodl_backup_attempt_04/tests.xml)、[attempt05 pupil 回归复跑](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_05/tests_retry.xml)。历史 pupil 参数来自旧 C2 敏感性数值，只用于明确标出的不具实物依据诊断。

### attempt06 新增证据

[几何收尾摘要](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_06/c3_geometry_closeout_summary.json) · [CMOS 面积积分结果](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_06/output_camera_sampling_audit.json) · [统一网格与来源记录](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_06/spatial_geometry.json) · [20 项定向测试](../../../outputs/20261104_hw207_mnist4_c3_v1/runs/attempt_06/tests.xml) · [ROI 判决实现](../../../src/ldm_project/v2/application/research/lightgen/hw207_spatial_geometry.py) · [aligned geometry v2 配置](../../../configs/research/lightgen/hw207_mnist4_c21_c3_aligned_geometry_v2.json)

## 决策影响与下一步

当前 HW207/LightGen **有条件地具备继续开展组件级空间光学建模与信息处理研究的工程基础**：输入网格如何映到目标面、相机有限像元如何积分、代码如何逐行复算，已有可复用实现和数值证据。它**尚不具备端到端光学分类或图像生成已获得实物实验支撑的证据基础**；缺的正是实物映射、已制造 LightGen 层的复传递、输出场/CMOS 标定和重复采集证据。

在当前“不补测、不操作硬件”的约束下，下一步若继续推进，只做显式假设下的数值敏感性分析和单层双光子打印模型的参数边界整理；未知材料与复透射继续标为 UNKNOWN，不把结果提升为 HW207_CURRENT。若未来要作实物结论，再单独决定是否需要取得器件身份、坐标映射、LightGen 复透射和 CMOS 标定。13 段估读距离保持原值；本轮没有补测、硬件操作或后继实验。

**最终判断：**本轮证明了若干理想化空间映射与探测积分组件可计算、可测试、可复核；没有证明完整 HW207/LightGen 实物光路已闭环，不能据此判断其实物四分类可行，也不能把类别读出外推为图像生成能力。

## 本地状态

分支 `research/lightgen-ols-native-semantic-generation-v1`，HEAD `043e2544594642669914355d35040124b64459b7`。本轮新增 attempt06 输出 Relay→CMOS 几何面积积分、修复数值平局拒判并通过 20 项定向测试；attempt04 全量 C2.1 validation 未重跑，模型权重和生成主线未改，未控制硬件或提交内容。工作区既有未提交内容均保留。
