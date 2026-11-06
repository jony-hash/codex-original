# HW207 实物约束光路数字孪生设计

## 文档状态与边界

本文件定义独立的 HW207 Optical Twin 设计线，供后续光学测量与模型实现审查使用。当前交付仅为只读调查后的设计文档：未改训练代码、模型、数据集或现有配置；未运行传播测量、生成实验、训练或 evaluator。本文中的 HW207 参数来自本轮任务记录或公开器件规格，不代表已完成实验室实测。

三种配置身份严格分开，不自动继承彼此的器件参数、传播参数或实测值：

- **PAPER_REFERENCE**：Science LightGen 论文实际报告的装置与样品信息，仅作文献参照；Science 的器件和 400×400/600×600、3 μm 芯片尺度不自动进入 HW207。连续理想振幅接口属于 HW207_TARGET/F0 的项目目标上限，不是 HES6001 实物复现。
- **HW207_TARGET**：项目自己的理想目标架构：532 nm、100×100 LightGen、2 μm 光子神经元节距、200×200 μm 有效区，允许理想连续振幅输入。它不含 DMD、relay misalignment 或 camera noise，仍是设计目标，不代表芯片已经制造或实测。
- **HW207_CURRENT**：依据 207 实验室实际器件和后续实测几何建立的配置。未知参数保留为 UNKNOWN 或 CALIBRATION_REQUIRED；不从 PAPER_REFERENCE 或 HW207_TARGET 自动继承。

数字孪生的目标是解释物理接口如何改变光场、功率和传感器读数，不替换现有 LightGen 论文参考、v1/v2 运行时或研究主线，也不授权网络更新。当前代码主线中已有理想 SLM 与数值 relay，不等于 HW207 硬件已被校准。E0 使用 HW207_TARGET 的理想输入接口；PAPER_REFERENCE 只作为论文器件/规模来源，不作为 HW207 的数值芯片配置。

## 论文参考光路 PAPER_REFERENCE

Science 补充材料第 2 页记录：单模 532 nm CNI MSL-U-532-300mW 激光器、级联扩束系统、HOLOEYE HES6001 振幅调制 SLM、将输入成像到芯片面的 4f relay、LightGen 芯片、Tucsen Dhyana 400BSI sCMOS；芯片到检测面的距离为 3 mm，因此使用镜头将输出成像到相机。Meadowlark P1920-400-800-PCIE phase-only SLM 用于制造前的单层可重构验证，不属于 Science 正式输入链路。

论文实验链路：

**MSL-U-532-300mW → 级联扩束 → HES6001 振幅 SLM → 4f 输入 relay → Science LightGen 芯片 → 输出成像镜头 → Dhyana 400BSI sCMOS**

论文补充材料第 3 页还报告了 400×400 或 600×600、每层 3 μm × 3 μm 的元表面神经元，以及 10×10 单模光纤 OLS。它们是论文样品信息，不得替换或覆盖 HW207 的芯片设计目标。论文未给出可直接作为 HW207 实测值的 HES6001 灰度到复振幅 LUT、光路镜头具体焦距、完整台面坐标、相机 532 nm QE 或本实验室器件参数。参考来源：[Science 补充材料本地副本](../../research_assets/lightgen/literature_library/papers/science_lightgen_2025/original/sciencesm.pdf)；[论文资料包说明](../../research_assets/lightgen/literature_library/papers/science_lightgen_2025/metadata/project_evidence_readme.md)；DOI [10.1126/science.adv7434](https://doi.org/10.1126/science.adv7434)。

Science 参考链中没有依据将 Nature 版 phase-only SLM 输入方式并入 HES6001 正式输入链路，也没有依据把 PF/PBS 的角度、偏振功能或位置补画进论文配置。

## 207 实物光路 HW207_CURRENT

当前用户资料明确了三条待核分支，且提供了原光路照片、M1–M4 卷尺照片、BOM 与 13 段照片估读表。拓扑作为用户提供的布局记录保存；照片估读不是机械或光学标定。各段独立沿该支路候选光轴定义，BS 分支与反射返回使用独立有向段，方向余未校准。

- 输入链：`Laser → Beam expander → Mirror-1 → PF-1 → PBS → BS`
- SLM 支路：`BS ↔ SLM 临时反射镜`。现为临时镜代位，70 mm 是单程。展开往返时保存两个 70 mm 段。
- 成像支路：`BS → Mirror-2（折转镜）→ Relay L2 → Relay L1 → PF-2 → LightGen 芯片预留位 → Relay lens → CMOS 预留位`

详细端点定义、来源和状态见本文末尾新增的光路固化记录及配套非运行时 JSON。未确认 PBS/BS 端口、反射角和 DMD ON/OFF 收光方向时，不把几何距离解释为已知传播光程。

## 当前项目结构与可复用性调查

| 能力 | 当前仓库位置 / 现状 | 对 HW207 的处理 |
|---|---|---|
| 复场和物理采样 | v1 有 `optics/fields.py`、`sampling.py`；v2 有物理平面、复场和强度帧契约 `contracts/planes.py`。v2 平面带形状、像素间距、波长、z 和坐标框架。 | 复用物理坐标、平面元数据、复场/强度分型的设计思想；HW207 必须有自己的平面身份和实测配置，不把张量形状相同当作空间对应。 |
| 传播 | v1 提供 Fresnel 与 angular-spectrum 传播；v2 提供带物理平面的 angular-spectrum 传播。 | 可复用传播数值核作为候选算子；波长、采样、传播距离、裁切和单位必须由 HW207 配置提供并校核。 |
| Relay / Fourier 光学 | v1 `optics/relays.py` 有 4f、pupil/NA 截止和焦距参数；v2 `optics/relays.py` 有 ABCD/B=0 成像、相干 pupil 过滤和显式源/目标采样。 | 可复用 relay 和 pupil 的物理接口；当前参数是模型输入，不是 207 的镜头间距、清孔径或实测像差。需增加 207 仿射配准、畸变残差和器件标定记录。 |
| 输入振幅 SLM | v2 `optics/devices.py` 的 `IdealAmplitudeSLM` 把 $[0,1]$ 振幅命令变为零残余相位复场；LightGen 研究链也有连续振幅输入。v1 有基于复数传输的 reflective amplitude SLM 抽象。 | 理想连续振幅输入只用于 HW207_TARGET/F0_IDEAL_AMPLITUDE 目标上限。v1 的复传输接口可供将来承载已标定的复响应，但现有实现没有真实 DMD 微镜状态/衍射/时序物理，不能直接命名为 DMD。 |
| 相位调制和衍射层 | v1/v2 `optics/d2nn.py` 有理想相位层和传播；v2 正式 D2NN 路径使用有限裁切。 | 仅在芯片侧且与其物理合同一致时复用；这不是 DMD 输入模型，也不代表 207 芯片的相位屏或器件已核验。 |
| DMD | 在 v1/v2 optics 与 LightGen 模块中未找到 F9000、DLP9000 微镜、ON/OFF 角度、像素填充、PWM 位平面或空间抖动的 HW207 实物接口。 | 需要独立的 DMD 命令/二值状态、镜面几何与状态效率、位平面时序和同步接口。未进入代码实现。 |
| 相机 | LightGen 研究计算可输出 (|E|^2)，部分路径还计算 W/pixel 和固定增益裁剪；正式 v2 的强度帧及 current/exposure 主要提供测量/曝光链路，不是 GSENSE2020 已标定传感器模型。现有 v10 相机接口报告明确记录真实 QE、暗帧、噪声、满阱等尚未齐备。 | 可复用像素功率积分、曝光边界和原始强度数据结构；需独立增加 532 nm QE、电子计数、噪声、满阱、ADC、ROI/binning 等以硬件标定为输入的模型。 |
| 配置/入口/证据 | 主入口包括 `train.py`、`python -m ldm_project.v2`；研究入口主要在 `scripts/research/lightgen/`，配置在 `configs/research/lightgen/`，测试在 `tests/` 和 `tests/research/lightgen/`。 | 当前只新增本目录三份设计文档。后续实现如获批准，应放入独立硬件研究命名空间和独立配置，不能借用主线配置自动覆盖主模型。 |

参考：[v2 运行时边界](../architecture/v2_runtime.md)、[理想输入/relay 研究模型](../../src/ldm_project/v2/optics/devices.py)、[v1 relay](../../src/ldm_project/v1/optics/relays.py)、[v2 relay](../../src/ldm_project/v2/optics/relays.py)、[相机接口审计 v9](../research/ols/ols_decoder_camera_interface_v9.md)、[相机接口收尾 v10](../research/ols/ols_camera_training_interface_closure_v10.md)。

仓库另有 HES6001 8-bit replay 配置和诊断脚本（`configs/research/lightgen/frozen_slm_8bit_replay_v1.yaml`、`scripts/research/lightgen/diagnose_lightgen_frozen_slm_8bit_replay_v1.py`），用于既有 SLM 编码假设的数值回放；它们不是 HW207 DMD 接口，也不是 HES6001 或 F9000 实物灰阶标定。

### 现成模块不能直接代表真实 DMD 的原因

理想振幅 SLM 的数学操作是场振幅乘连续 $A(x,y)$。DMD 是倾转微镜阵列：二值图案控制镜面 ON/OFF 落位，镜面几何会改变反射方向和空间频谱；实际收集到的场还取决于照明角、ON/OFF 出射方向、光阑、衍射级次、像差与相机/relay。TI 的 DLP9000 规格可作芯片数据表输入，但其效率数值有规定的照明角和 F/# 条件，不能当成本系统传输率。8-bit 命令也可能由多个二值时间片或空间图案实现，不能替代连续静态振幅。

## Hardware Source of Truth

**状态**沿用本任务定义：CONFIRMED、DATASHEET、MEASURED、PROVISIONAL、UNKNOWN、CALIBRATION_REQUIRED、DIAGNOSTIC_ONLY。状态按参数而非整台设备赋予。**置信度**是来源质量的说明，不等价于参数已测：HIGH 表示可定位的正式论文/制造商资料；MEDIUM 表示可定位的用户规格图片但设备完整身份未确认；LOW 表示只有本轮任务记录。Physical location 未被实物标签、照片和台面坐标共同确认的，一律不填具体光路位置。

### PAPER_REFERENCE（论文装置，不是 207 器件）

| Hardware ID | 类别 / 型号 | 可引用信息与状态 | 来源 / 置信度 | 实验室位置与备注 |
|---|---|---|---|---|
| PAPER-LASER | CNI MSL-U-532-300mW | 532 nm、单模；CONFIRMED 为论文中报告的型号。论文标称功率不代表到达 HES6001 的功率。 | Science SI 第 2 页；HIGH | 仅 PAPER_REFERENCE。 |
| PAPER-SLM | HOLOEYE HES6001 | 振幅调制 SLM；CONFIRMED 为论文器件身份。灰阶响应、复传输、空间均匀性、器件序列号和本地单位均 UNKNOWN。 | Science SI 第 2 页；HIGH | 仅 PAPER_REFERENCE，第一阶段采用理想连续振幅近似。 |
| PAPER-RELAY | 4f relay | 输入共轭到 LightGen 输入面；CONFIRMED 为论文光路功能。焦距、间距、NA、孔径和实测误差 UNKNOWN。 | Science SI 第 2 页；HIGH | 不将任何 207 焦距自动移入此链。 |
| PAPER-CHIP | Science LightGen | SI 第 3 页报告 400×400 或 600×600、3 μm 元表面像素和 10×10 SMF OLS；CONFIRMED 为论文样品信息。 | Science SI 第 3 页；HIGH | 不等同 HW207 100×100、2 μm 设计目标。 |
| PAPER-CAMERA | Tucsen Dhyana 400BSI sCMOS | CONFIRMED 为论文探测相机型号；像素响应/曝光的实验室参数未在本任务中作为 HW207 值使用。 | Science SI 第 2 页；HIGH | 仅 PAPER_REFERENCE。 |
| PAPER-PHASE-SLM | Meadowlark P1920-400-800-PCIE | SI 记为单层可重构验证用 phase-only SLM；CONFIRMED 为该用途报告。 | Science SI 第 2 页；HIGH | 不混入 Science 正式振幅输入链路。 |

### HW207_TARGET（项目理想目标，不是实物测量）

| Target ID | 目标量 | 当前定义与状态 | 边界 |
|---|---|---|---|
| HW207-TARGET-INPUT | 波长 / 输入接口 | 532 nm；理想连续振幅 $E_{in}=A(x,y)E_0(x,y)$，$A\in[0,1]$；PROVISIONAL（项目设计目标） | 不包含 DMD 命令、微镜、relay 误差、偏振器件响应或传感器噪声；不代表 HES6001 实物复现。 |
| HW207-TARGET-CHIP | LightGen 目标芯片 | 100×100、2 μm 光子神经元节距、200×200 μm 有效区；PROVISIONAL（项目设计目标） | 不替换 Science 的 400×400 / 600×600、3 μm 样品；制造状态和实物尺寸未在本轮确认。 |
| HW207-TARGET-PROPAGATION | 下游传播 | E0 与 E1 使用同一 HW207 目标芯片和同一已定义下游传播模型 | 模型参数须在实验/计算前明确；不得从 PAPER_REFERENCE 自动带入芯片尺寸或传播参数。 |

### HW207_CURRENT（当前实物记录，需现场确认）

| Hardware ID | 类别 / 制造商与型号 | 参数、状态与来源 | 置信度 | 校准状态、位置和注意事项 |
|---|---|---|---|---|
| HW207-LASER | 激光器；任务记录 PGL-FS-532-50mW；制造商/序列号 UNKNOWN | 532 nm、50 mW：PROVISIONAL，仅来自任务记录；到达 DMD 的功率和稳定性 UNKNOWN。 | LOW | CALIBRATION_REQUIRED：实物标签、完整型号、功率计读数、光束直径和扩束后剖面。仓库已有 CNI PGL-FS-III-532 系列 PDF，但型号不同；该系列 PDF 为 532±1 nm、TEM00 CW，输出档位列为 >1/10/30…80 mW，不能确认它就是计划中的 PGL-FS-532-50mW。 |
| HW207-MIRROR | 反射镜；GCC-101044 | 高反、532 nm：PROVISIONAL，来自任务记录；反射率曲线、基底/角度和实际入射面 UNKNOWN。 | LOW | CALIBRATION_REQUIRED：核对标签、入射角、实际光束方向和透射/反射功率。位置 UNKNOWN。 |
| HW207-EXPANDER | 扩束系统；GCO-140113 | 8×、532 nm：PROVISIONAL，来自任务记录；型号数据表未在仓库中找到。出射光斑直径、椭圆率、均匀性 UNKNOWN。 | LOW | CALIBRATION_REQUIRED：记入前后束径、距离和剖面。位置 UNKNOWN。 |
| HW207-PF | PF；GCL-050003（器件类别/缩写待确认） | 消光比 500:1：PROVISIONAL，来自任务记录；透过率、偏振轴、角度和 PF 的准确器件功能 UNKNOWN。 | LOW | CALIBRATION_REQUIRED：先确认器件标签和功能，再测透过/消光/方向。位置 UNKNOWN；不推断论文偏振设置。 |
| HW207-PBS | PBS；GCC-M402103 | 波段 420–680 nm：PROVISIONAL，来自任务记录；s/p 透射、反射、消光比、入射/出射端口和摆放方向 UNKNOWN。 | LOW | CALIBRATION_REQUIRED：逐端口测功率和偏振；位置 UNKNOWN。 |
| HW207-BS | 分光器；MBS1455-A（制造商 UNKNOWN） | 分光比、波段、偏振属性和方向均 UNKNOWN；型号记录 PROVISIONAL。 | LOW | CALIBRATION_REQUIRED：实物身份、端口/角度、偏振依赖和功率比。位置 UNKNOWN。 |
| HW207-DMD | DMD 系统；F9000；任务记录芯片为 TI DLP9000 | DLP9000 数据表：2560×1600、7.56 μm pitch、±12° 镜面倾角、400–700 nm、92% array fill；DATASHEET。名义活性阵列尺寸由行列和 pitch 推得约 19.3536 mm × 12.096 mm，属于数据表几何推算，不是实测安装视场。数据表另列微镜反射率 88%、array diffraction efficiency 86% 和窗口透过率 95%；其中效率数据假设 24° 照明角、F/2.4 照明/收集、100% 镜面电子 duty cycle，且不含光学 overfill loss，均不能当作 HW207 的系统效率。TI 对 DLP9000/DLPC900 给出 1-bit 最高 9523 Hz；8-bit 的 1031 Hz 对应预载图案及 illumination modulation，外部输入最高 360 Hz，表格的视频/按需模式 8-bit 为 247 Hz。该模式差异均为 DATASHEET。 | HIGH（芯片规格）；LOW（F9000 实物系统身份） | F9000 整机/控制器、固件、图案加载模式、实际 bit-plane 顺序/权重、触发延迟、镜面朝向、入射角、OFF 光收集、损伤阈值均 UNKNOWN / CALIBRATION_REQUIRED。不要把 DLP9000X 的更高速规格套用到 DLP9000。 |
| HW207-IN-RELAY | 输入 relay；2× GCL-010605A，任务记录 f=150 mm | 焦距和数量：PROVISIONAL，来自任务记录；清孔径、焦距公差、厚度主平面、实际间距、工作距离、是否构成 4f UNKNOWN。 | LOW | CALIBRATION_REQUIRED：量测主平面间距、物像共轭、倍率、NA、倾斜/像差与配准。位置 UNKNOWN。 |
| HW207-OUT-LENS | 输出镜头；GCL-010650A，任务记录 f=30 mm | 型号/f=30 mm：PROVISIONAL，来自任务记录；焦距验证、孔径、相机共轭和实际倍率 UNKNOWN。 | LOW | 暂归输出成像链；不得默认借作输入 relay。M1 使用此透镜时必须记录输出链变化。位置 UNKNOWN。 |
| HW207-CHIP | LightGen 芯片目标 | 100×100、2 μm 节距、200×200 μm 有效区：PROVISIONAL 的项目设计目标，不是制造完成或实物测量。 | MEDIUM（任务目标） | CALIBRATION_REQUIRED：拿到版图/芯片图纸、批次、层结构、面向和实际有效边界；论文的 3 μm 芯片不是替代值。 |
| HW207-CMOS | 相机系统；传感器记录 Gpixel GSENSE2020，整机型号 UNKNOWN | 本地用户规格图片列出 FSI sCMOS、2048×2048、6.5 μm 像元、72%@595 nm、读噪声典型 2.1 e⁻、典型满阱 43 ke⁻、8/12/16 bit、45 fps@8 bit、40 fps@16 bit、10 μs–10 s、rolling shutter、ROI/binning 支持：PROVISIONAL（截图资料）。本轮任务文字另记约 13.8 ke⁻ 满阱，与截图 43 ke⁻ 冲突；当前实际相机满阱值为 UNKNOWN，不选其中一值。532 nm QE UNKNOWN。 | MEDIUM（截图参数）；LOW（相机整机身份） | CALIBRATION_REQUIRED：相机整机、增益/读出模式、532 nm QE、读噪声/暗帧、线性区、满阱、曝光、ADC、ROI、binning、像元响应和输出倍率。相机像元不等于芯片神经元。 |

来源：TI [DLP9000 datasheet](https://www.ti.com/lit/ds/symlink/dlp9000.pdf)；CNI [PGL-FS-III-532 datasheet](https://www.cnilaser.com/PDF/PGL-FS-III-532.pdf)；GSENSE2020 用户规格图[本地副本](../../research_assets/lightgen/literature_library/devices/gsense2020_user_spec.png)及[既有来源比较记录](../../research_assets/lightgen/hardware_g1_material_evidence_20261028_v1/device_and_smf_comparison.md)。供应商系列资料只能支持对应系列/芯片规格，不证明本实验室具体器件或测量值。

### 明确的来源冲突和缺口

| 项目 | 证据 A | 证据 B | 当前处理 |
|---|---|---|---|
| 207 激光器型号/功率 | 本轮任务记录 PGL-FS-532-50mW | 仓库/制造商 PDF 是 PGL-FS-III-532，输出档位列 1/10/30…80 mW | 保留计划项为 PROVISIONAL；不把相似型号、50 mW 或系列规格写为 HW207 已确认身份。 |
| CMOS 满阱 | 本轮任务记录约 13.8 ke⁻ | GSENSE2020 用户规格图列典型约 43 ke⁻ | 当前满阱为 UNKNOWN；核验相机整机型号、读出模式和规格来源后再确定。 |
| DMD 图案速率 | 任务记录 9523 Hz / 1031 Hz | TI 表明速率取决于控制器、预载/外部输入和图案模式 | 只登记数据表模式上限；实际系统速率、PWM权重和同步 UNKNOWN。 |
| 207 光路顺序 | 本轮任务的暂定文字拓扑 | 未见现场照片、器件原始清单和实测位置/角度记录 | 所有位置均 UNKNOWN；拓扑仍待确认。 |
| 207 芯片 | 当前目标 100×100 @ 2 μm | Science 论文样品 400×400 或 600×600 @ 3 μm | 两者身份、目标和来源分开。 |

## 坐标、光路映射和尺度模型

每个物理面记录：配置 ID、平面 ID、波长、单位、坐标原点、坐标正向、形状、采样间距、传播距离和来源状态。DMD 的列/行坐标先定义为自身像素索引坐标；相对于镜头座、芯片版图和相机 sensor 的旋转、翻转及偏移，需由实际封装方向和标记图案测定后再赋值。反射前后用不同局部 frame，并将像面奇偶翻转保留在矩阵行列式符号内。

仿射起始模型为

\[
\begin{bmatrix}x_{chip}\\y_{chip}\end{bmatrix}
=\mathbf{A}\begin{bmatrix}x_{DMD}\\y_{DMD}\end{bmatrix}+\mathbf{b},
\]

其中 \(\mathbf A\) 包含 X/Y 缩放、旋转和剪切，\(\mathbf b\) 为平移。先用覆盖全视场的非共线标记拟合并保留独立残差；残差显示超出测量不确定度的系统畸变时再考虑高阶映射。标量倍率只作理想 4f 名义估计，不代替仿射校准。

在理想两透镜 4f 条件下，带符号横向倍率为 M=-f2/f1，投影像素 pitch 的名义值为 p_proj=|M|p_DMD，每个 200 μm 宽度内的 DMD 周期数为 N=W_chip/p_proj。相对于 HW207_TARGET 的 2 μm neuron 节距，每个 neuron 对应的名义 DMD 周期数为 N_DMD periods per neuron=p_chip/p_proj。对于 150 mm 第一片：

| 组别 | 配置假设 | 名义倍率 | 名义投影 pitch | 200 μm 内周期数 | 每个 2 μm neuron 的 DMD 周期数 | 状态与限定 |
|---|---|---:|---:|---:|---:|---|
| M0 | 150/150 mm | 1.0× | 7.56 μm | 26.46 | 0.2646 | 仅在现有两片 150 mm 透镜确实满足 4f 共轭时才成立；名义采样数，不是有效自由度测量。 |
| M1 | 150/30 mm | 0.2× | 1.512 μm | 132.28 | 1.3228 | DIAGNOSTIC_ONLY。30 mm 镜头当前记录为输出镜头；调入输入端会改变/占用输出链。 |
| M2 | 150/约39.68 mm | 0.26455× | 约2.00 μm | 约100 | 约1.0 | DIAGNOSTIC_ONLY。39.68 mm 是尺度匹配所需的理想焦距，不代表库存或实装。 |

这些周期数只描述几何采样。实际输入有效分辨率还受 DMD 衍射、relay pupil/NA、像差、功率、投影对准、像素积分和相机传感器影响。

下表给出单个 superpixel 映射到芯片面的名义边长；数字按上表投影 pitch 计算，仅用于可行性分析，不代表已实现的有效灰度或分辨率：

| Superpixel | M0 芯片面边长 | M1 芯片面边长 | M2 芯片面边长 |
|---|---:|---:|---:|
| 1×1 | 7.56 μm | 1.512 μm | 约2 μm |
| 2×2 | 15.12 μm | 3.024 μm | 约4 μm |
| 4×4 | 30.24 μm | 6.048 μm | 约8 μm |
| 8×8 | 60.48 μm | 12.096 μm | 约16 μm |

superpixel 尺寸同时改变灰度表达和空间分辨率。较大的 superpixel 不必然是更好的灰度方案；若其芯片面尺寸超过目标细节尺度，应标为 `resolution_limited`。

模型应显式包含：DMD 活性区域/封装边界、微镜 pitch 与填充率、relay 焦距和主平面位置、清孔径与 NA、相干传播和衍射、像素积分、二维配准、畸变、以及输入/输出参考面。焦距与孔径未知时不填默认值；NA 可在空气近轴近似下由实测清孔径和有效焦距估计，但报告使用的精确定义和测量几何。

## DMD 调制模型

概念物理接口分为四层：

`DMDCommand → MicromirrorState → ReflectedComplexField → RelayCollectedField`

命令图案、微镜状态、反射复场和 relay 实际收集场是不同物理量；不能把命令数组直接解释成芯片面连续复振幅。

### DMD-A：Binary，实际基础模式

F1 是 DMD pixelated binary。第一阶段可用标量二值近似，必须明确标记为 `scalar_binary_approximation`；它不代表完整 DLP9000 微镜物理。

F2 才加入完整微镜与收集物理：±12° micromirror tilt、finite micromirror aperture、fill factor、ON/OFF reflected direction、diffraction orders、complex reflection coefficient，以及 relay pupil / collection geometry。每项数值均须绑定对应数据表条件或 HW207 实测；不能把条件化数据表效率直接当作系统收集效率。

命令图案为 \(D_{mn}\in\{0,1\}\)。为保留微镜几何，后续模型的局部场应按镜面状态和有限像素孔径形成，再经过相干 relay/pupil 传播。实际 ON/OFF 状态的反射方向、复反射系数、离轴照明、镜面方位和被收集衍射级次须由芯片数据表与 207 实测共同决定。先做 scalar binary 近似时，必须注明遗漏的矢量/偏振及微镜倾角；不得把它写成连续振幅 SLM。

### DMD-B：Naive Gray，非物理负对照

\[
A(x,y)=g(x,y)/255.
\]

必须同时标记为 **UNPHYSICAL_DIAGNOSTIC_ONLY** 与 **NOT_PHYSICAL_DMD_RESULT**。它只回答错误地把 DMD 0–255 命令当连续振幅时的乐观程度或偏差，不代表 F9000 的真实灰度行为。

### DMD-C：Temporal PWM

E3-A 是 `global-exposure-equivalent diagnostic`，所有 sensor pixels 共用同一曝光窗：

`I = ∫ |H[D(t)E]|² dt`

E3-B 是 `real rolling-shutter integration`，第 r 行使用其实际曝光起始时间和曝光时长：

`I_r = ∫_(t_r)^(t_r+T_exp) |H[D(t)E]|² dt`

只有在 camera line time、row exposure delay、frame timing、DMD bit-plane timing 和 trigger relation 均已知后才可运行 E3-B；否则 BLOCKED，不得用 DLP9000 名义最大帧率猜测。上述离散位平面加权式只表示对应曝光窗内的强度积分，不可用平均复振幅替代。
以实测位平面序列 \(D_k\)、时间片权重 \(w_k\) 和相机曝光/同步为输入：

\[
I_{PWM}=\sum_k w_k\left|H[D_kE_{in}]\right|^2.
\]

不使用 \(\left|H[\langle D\rangle E_{in}]\right|^2\) 替代曝光期间的强度积分。若控制器、位平面时间、权重、触发和相机曝光同步任一项未知，则该配置 UNKNOWN，不生成猜测时序。

E3-A 的离散权重对应全局共用曝光窗；E3-B 必须按每行实际曝光区间裁切位平面权重，不能沿用单一全局权重。

### DMD-D：Spatial Dithering

E4-A 先根据实测 relay magnification、p_proj、chip pitch、MTF 和目标空间频率，计算候选 superpixel 的物理尺寸与空间带宽损失；E4-B 只测 E4-A 后仍有物理意义的选定尺寸，不要求 2×2、4×4、8×8 全部执行。每个二值子图案经过微镜孔径、相干传播和实际 relay 空间滤波后，在芯片面计算复场/强度和频率响应。superpixel 占空比只是命令统计，不能直接当作芯片面复振幅或真实有效灰度等级；若 superpixel 大于目标空间细节尺度，标记 `resolution_limited`。

## 偏振与功率接口

PF/PBS/BS 先用器件身份确认后的简单端口透过/反射功率模型；不先验假设其有益、无效或复现论文光路。基线组为 P0（no PF / no PBS）、P1（PF only）、P3（PF+PBS）。只有实际光路允许且 PBS 不承担不可替代的转向功能时，才增加 P2（PBS only）。若移除 PBS 会改变几何光路，必须几何补偿，或标记为 `configuration_comparison`，不得称为偏振单因素实验。BS 是否固定保留应由核实后的拓扑决定。先比较总功率、光斑形状、条纹对比和方向频响；若形状变化超过重复测量不确定度，再讨论 Jones 矩阵。未测偏振前不填 Jones 系数，也不把供应商消光比等同在系统中的实际表现；不推断 Science 论文中 PF/PBS 的具体角度。

各组测 DMD 入射功率、DMD ON-state 功率、LightGen 输入面功率和 CMOS 接收功率；分别记录未归一化功率与归一化图样。这样可区分纯功率损耗、图样改变、DMD 反射效率变化和 PBS 端口损耗。

## CMOS 探测模型

sensor pitch 为 6.5 μm。定义输出成像倍率 M_out 后，芯片面的等效 detector sampling 为 p_object=6.5/M_out μm。对 2 μm target chip pitch，几何采样下限是 M_out≥3.25；这是每个目标节距至少一个 sensor sample 的几何下限，不是解析 2 μm 结构的充分条件。实际解析还受 optical MTF、sensor MTF、aberration 与 SNR 限制。

若 M_out<3.25，针对单个 2 μm neuron 的采样标记为 `undersampled`；可使用更高倍率、long-period targets 或 Fourier-plane diagnostics。不得声称相机能解析单个 neuron 或 DMD pixel。

先保留理想强度面 \(I(x,y)=|E(x,y)|^2\)，并声明场到 W/m² 的标定。接着以输出成像倍率把芯片面映射到 sensor 面，按 6.5 μm sensor pitch 做有限像素面积积分，才进入功率/光子量。电子数期望可写为

\[
\mu_e = \mathrm{QE}(532\,\mathrm{nm})\,P_{pix}t_{exp}/(hc/\lambda),
\]

之后依次增加 photon shot noise、实测 read noise/暗电流、满阱裁剪、ADC 量化、ROI 和 binning。传感器像素积分和相机成像倍率必须分开：仅当相机与相应光学面共轭且倍率已知，才可讨论芯片或 DMD 周期在 sensor 上是否可分辨。1.512–2 μm 投影周期小于 6.5 μm sensor pitch；若在该成像面直接采样，像素会合并多个空间细节。当前输出 relay magnification UNKNOWN，因此这只是条件性风险，不能据此断言现有 CMOS 具体解析能力。

不将 72%@595 nm 代替 532 nm QE，也不在 13.8 ke⁻ / 43 ke⁻ 满阱冲突未解决时计算饱和。绝对功率、归一化图样和传感器归一化输出分别报告，避免归一化掩盖功率损失。

## 模型保真度与配置命名

每一层独立开关、独立配置 ID，低保真参考保留为只读基线；高保真层不得覆盖低保真基线。

| Fidelity | 内容 | 前置数据 / 限定 |
|---|---|---|
| F0 | HW207_TARGET ideal amplitude input：连续 $A\in[0,1]$ 乘已定义 $E_0$，目标芯片为 100×100、2 μm pitch。 | `F0_IDEAL_AMPLITUDE`；不是 HES6001 复现或 DMD 行为。 |
| F1 | DMD pixelated binary：二值命令的标量近似，标记 `scalar_binary_approximation`。 | DMD 身份、方向、pitch 和二值有效面；不代表完整 DLP9000 微镜物理。 |
| F2 | DMD reflected field：微镜倾角/有限孔径/填充率/ON-OFF 方向/衍射级次/复反射及 relay pupil 收集。 | 微镜角、照明/收集几何、NA/孔径；数据表效率不替代系统标定。 |
| F3 | Real relay：实测焦距/间距、倍率、仿射配准、畸变、像差和 pupil。 | 物像面和几何标定完成；未测像差保留未知。 |
| F4 | Polarization：PF/PBS/BS 的实测透反及偏振状态；必要时 Jones 模型。 | 只有图样变化证据支持时增加 Jones 层。 |
| F5 | Temporal modulation：实测 PWM/位平面时序、权重、触发与相机曝光积分。 | 控制器/固件/同步资料和实测时序齐备；否则配置 UNKNOWN。 |
| F6 | CMOS：像素积分、倍率、QE、shot/read/dark noise、满阱、曝光、ADC、ROI/binning。 | 相机整机、532 nm 标定和读出参数齐备；规格冲突关闭。 |

建议配置命名为 HW207_TARGET/F0_IDEAL_AMPLITUDE 与 HW207_CURRENT/F1…F6；PAPER_REFERENCE 保持文献身份，不自动作为 E1 数值对照芯片。本文只规定命名和边界，不创建配置代码或运行配置文件。

## 与现有主线隔离规则

1. HW207 独立于 PAPER_REFERENCE、v1 LDM teacher 线、v2 teacher-free formal runtime 和现有 LightGen 研究线。
2. 不修改模型/训练代码、权重、数据集、Evaluator、正式主线配置或现有输出。
3. v2 formal chain 的理想 amplitude SLM、encoder metasurface、identity OLS、decoder metasurface、CMOS exposure 不自动成为 HW207；其合同没有 F9000 DMD。
4. 真实 DMD 接口实现、公开 CLI、SLM/芯片行为或 OLS/CMOS 校准合同变更都需要单独审查与批准。
5. 任何将来网络对照均先冻结权重。Hardware-aware retraining 是 E9，必须在接口标定、E8 冻结替换复核后另行批准；当前设计文档不授权它。

## 首批用户确认/实测问题

1. 提供 207 台面照片/光路图和原始器件表；逐器件确认标签、完整型号、序列号、端口与实际位置/方向。
2. 确认激光器到底是 PGL-FS-532-50mW 还是其他 PGL-FS 系列型号，并测 DMD 入射功率和扩束后二维光斑。
3. 确认 F9000 内部 DMD 芯片、控制器/固件、实际 Pattern 模式、位平面时序与硬触发链路。
4. 测两片 150 mm 输入镜的主平面间距、清孔径、实际 4f 共轭、当前倍率和像面变形；确认 30 mm 输出镜是否仍在输出链。
5. 确认相机整机型号/读出模式，解决满阱 13.8 ke⁻ 与 43 ke⁻ 冲突，取得 532 nm QE/增益/噪声/曝光和输出成像倍率。

**决策边界：** 以上任一关键输入缺失时，仍可保存设计或做标为 diagnostic-only 的敏感性分析，但不得称为 HW207 实物数字孪生验证。人工审阅本文后再决定是否开展标定；本文不自动进入代码实现、实验或训练。

## HW207 光路参数固化（来源快照 2026-11-03）

本节优先记录用户提交的实物资料包。来源次序为原始工作簿/距离 Excel 与 TSV、照片及用户规格图、可核验厂家资料、理论推导。工作簿的“数量”是采购表数量，不证明库存或实装。`LightGen光路-1014-v5.xlsx` 的 Sheet1 与 Sheet2 数量不同，原始行和数量均保存在 [独立非运行时几何记录](../../configs/research/lightgen/hw207_optical_geometry_v1.json)。完整零件行包括机械支撑及定制转接件，不仅包含光学件。

### 来源清单与证据等级

| 来源 | 证据等级 | 用途与边界 |
|---|---|---|
| README_先读.md、实物信息与闭环状态.md、原光路图与标注图 | 用户资料包说明/照片可见信息 | 拓扑、估距口径、预留/临时位置；不建立实测光轴或主平面 |
| LightGen_adjacent_optics_distance_table_v1.xlsx、distance_table_copy_to_excel.tsv | 用户表格原始记录 | 13 段估距；Excel 与 TSV 的 13 条备注文本逐字不同，分别保留 |
| LightGen光路-1014-v5.xlsx、LightGen_optical_setup_BOM_reference.xlsx | BOM 原始单元格 | 两个工作簿文件字节相同；Sheet1/Sheet2 数量仍分别保留。采购数不表示库存或实装 |
| 原光路、标注图、M1–M4 卷尺照片 | 用户照片 | 仅支持初始几何与组件可见性，不给出量化误差、主平面或装机身份 |
| F9000、CMOS 用户规格图 | 用户提供的规格图 | 不能证明本机安装、接线或型号 |
| TI DLP9000 Rev. B 数据手册与产品页 | 厂家芯片资料 | 芯片级参数，不等于 F9000 整机或 HW207 实测 |
| FLDiscovery F9000 产品页 | 厂家页面，本轮返回 HTTP 403 | 网页字段未能独立复核，只转录用户规格图 |

### 原始光路与 13 段照片估距

距离表逐项按源值存为毫米，状态均为 `PHOTO_ESTIMATE_NOT_CALIBRATED`。表称器件中心/反射点估距并按厘米取整，不等同于镜片主平面、镜架外壳面或探测面间距。资料不提供不确定度，本文件不添加估读误差范围。

Excel 与 TSV 分别保留原始备注。中心距是卷尺照片估读并按厘米取整。Excel 数据行 5–17，TSV 数据行 2–14。

Excel 与 TSV 分别保留原始端点与备注。中心距是卷尺照片估读并按厘米取整。Excel 数据行 5–17，TSV 数据行 2–14。

| 段 | 分支 | Excel 原始相邻器件 | TSV 起点→终点 | 距离 | 照片 | Excel 原始备注 | TSV 原始备注 |
|---|---|---|---|---:|---|---|---|
| 01 | 输入链 | Laser → Beam expander | Laser→Beam expander | 30 mm | M1 | 出光口—入光口中心投影估读 | 卷尺照片估读；出光口至入光口中心投影 |
| 02 | 输入链 | Beam expander → Mirror-1 | Beam expander→Mirror-1 | 100 mm | M1 | 卷尺刻度投影估读 | 卷尺照片估读 |
| 03 | 输入链 | Mirror-1 → PF-1 | Mirror-1→PF-1 | 130 mm | M2 | 卷尺端点中心投影估读 | 卷尺照片估读 |
| 04 | 输入链 | PF-1 → PBS | PF-1→PBS | 70 mm | M2 | 卷尺端点中心投影估读 | 卷尺照片估读 |
| 05 | 输入链 | PBS → BS | PBS→BS | 90 mm | M2 | 卷尺端点中心投影估读 | 卷尺照片估读 |
| 06 | SLM支路 | BS → SLM位（临时反射镜） | BS→SLM位置（临时反射镜） | 70 mm | M2 | 临时反射镜代位；实际SLM安装后复测 | 单程距离；SLM反射后沿同一路径返回 |
| 07 | 成像支路 | BS → 折转镜（Mirror-2） | BS→Mirror-2（折转镜） | 100 mm | M3 | 垂直尺读数按光轴中心投影估读 | 卷尺照片估读 |
| 08 | 成像支路 | 折转镜（Mirror-2） → Relay L2 | Mirror-2（折转镜）→Relay L2 | 60 mm | M4 | 全景尺带投影估读 | 卷尺照片估读 |
| 09 | 成像支路 | Relay L2 → Relay L1 | Relay L2→Relay L1 | 350 mm | M4 | 全景尺带刻度估读 | 卷尺照片估读；两片中继透镜间距 |
| 10 | 成像支路 | Relay L1 → PF-2 | Relay L1→PF-2 | 90 mm | M4 | 全景尺带投影估读 | 卷尺照片估读 |
| 11 | 成像支路 | PF-2 → LightGen芯片位 | PF-2→LightGen芯片预留位置 | 60 mm | M4 | LightGen预留位；芯片安装后复测 | 位置估读；芯片安装后复测 |
| 12 | 成像支路 | LightGen芯片位 → Relay lens | LightGen芯片预留位置→Relay lens | 80 mm | M4 | LightGen预留位；芯片安装后复测 | 位置估读；芯片安装后复测 |
| 13 | 成像支路 | Relay lens → CMOS位 | Relay lens→CMOS预留位置 | 100 mm | M4 | CMOS预留位；相机安装后复测 | 位置估读；相机安装后复测 |

来源差异：编号、支路、照片与厘米换算后的数值对应一致。Excel 将端点写成一个相邻器件字符串，TSV 拆成起点/终点两列；13 行端点文本格式均不相同，但器件顺序可对应。13 条备注文本也逐字不同，以上并列原文。段 06 尤其不同：Excel 写“临时反射镜代位；实际SLM安装后复测”，TSV 写“单程距离；SLM反射后沿同一路径返回”。源表只含一条 70 mm 单程估值；配置中的 70 mm 回程是将 TSV 的返回描述显式化所得建模假设，不是第二次测量。安装 DMD 后的实际 ON/OFF 路径仍未知。Excel 第 20–22 行为测量口径/支路/预留位置说明，第 24–28 行为 M1–M4 照片索引。

算术核对：01–05 合计 420 mm，07–13 合计 840 mm，13 行各加一次为 1330 mm。若假设 BS→临时镜往返后再接 07–13，条件性几何估值为 `420 + 2×70 + 840 = 1400 mm`。上述合计均不是一条已闭合光线的总长或 OPL。临时镜回程到 L2 的照片路径和为 70+100+60=230 mm；L2→L1 为 350 mm；L1→芯片预留位为 90+60=150 mm；临时镜到预留芯片位的表值合计 730 mm。仅作估距算术，不作为主平面距离。

仿真自由空间段应连接明确光学参考面并沿段局部光轴传播。现有照片只支持端点中心/反射点的初始几何估值，尚未建立实际入射方向、折叠坐标、介质折射率和元件内长度。故 `mechanical_gap_mm`、`principal_plane_gap_mm`、`geometric_path_mm`、`optical_path_length_mm` 分开保存。前三者中只有照片端点估距已知；真实 OPL 为 `UNKNOWN`。BS→SLM 70 mm 不代表已装 DMD 的真实 ON/OFF 路径；芯片和 CMOS 安装后距离仍为预留位估读。本轮不安排硬件补测。

### 器件身份、BOM 与冲突

主工作簿两表逐行数据以原始单元格行号保存在 JSON，型号后缀和空型号均原样留存。Sheet1 / Sheet2 采购数量差异示例：30 mm 镜 2/3、三维镜架 2/3、偏振片每行 2/3 及 1/3、干板夹 1/4、150 mm 镜均 2、简易镜架 2/3、400 mm 支杆 1/3、Mirror 架 1/3、高反镜 1/4、BS 1/4、PBS 1/4、扩束镜 1/3、V 架 1/3、102 mm 支杆均 10、102 mm 调节座均 10、76 mm 支杆 6/12、76 mm 调节座 6/12、叉式压板均 16、激光器均 1、定制激光器转接板均 1。各 Sheet 的金额及行值不用于推断安装数量。


完整 BOM 原始行（Excel 第 3–27 行）如下。空单元格用“（空）”标记。源工作簿只有文献名称、品牌、类型、型号、参数、数量、单价、总价八列；没有数量单位、币种或独立备注列，这些字段保留为未提供。价格仅为逐行保留来源，不据此推断库存或实装。

| 行 | 文献名称 | 品牌 | 类型 | 型号 | 参数（原单元格） | 数量 S1 | 单价 S1 | 总价 S1 | 数量 S2 | 单价 S2 | 总价 S2 |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | relay lens | 大恒 | 消色差透镜 | GCL-010650A | f30 φ25 | 2 | 370 | 740 | 3 | 680 | 2040 |
| 4 | （空） | 大恒 | 三维镜架 | GCM-082814M | φ25.4 | 2 | 450 | 900 | 3 | 725 | 2175 |
| 5 | PF | 大恒 | 偏振片 | GCL-050003 | φ25.4    消光比500∶1 | 2 | 280 | 560 | 3 | 423 | 1269 |
| 6 | （空） | 大恒 | 偏振镜架 | GCM-0902M | φ25.4 | 2 | 350 | 700 | 3 | 700 | 2100 |
| 7 | （空） | 大恒 | 干板夹 | GCM-132101M | 夹持厚度≤10mm | 1 | 140 | 140 | 4 | 360 | 1440 |
| 8 | relay system | 大恒 |  消色差透镜 | GCL-010605A | f150   φ25.4 | 2 | 325 | 650 | 2 | 680 | 1360 |
| 9 | （空） | 大恒 | 简易镜架 | GCT-080104 | φ25.4 | 2 | 140 | 280 | 3 | 280 | 840 |
| 10 | （空） | 大恒 | 支杆 | GCT-010107-P4 | 400mm | 1 | 595 | 595 | 3 | 730 | 2190 |
| 11 | Mirror | 大恒 | 二维镜架 | GCM-080817M | φ40 | 1 | 310 | 310 | 3 | 600 | 1800 |
| 12 | （空） | 大恒 | 单波长高反射镜 | GCC-101044 | φ40     532nm | 1 | 190 | 190 | 4 | 640 | 2560 |
| 13 | BS | 麓邦 | 非偏振分光棱镜 | MBS1455-A | 400-700mm | 1 | 850 | 850 | 4 | 850 | 3400 |
| 14 | PBS | 大恒 | 偏振分光棱镜 | GCC-M402103 | 420-680nm | 1 | 960 | 960 | 4 | 955 | 3820 |
| 15 | PF | 大恒 | 偏振片 | GCL-050003 | φ25.4    消光比500∶1 | 1 | 280 | 280 | 3 | 423 | 1269 |
| 16 | （空） | 大恒 | 偏振镜架 | GCM-0902M | φ25.4 | 1 | 350 | 350 | 3 | 700 | 2100 |
| 17 | Mirror | 大恒 | 二维镜架 | GCM-080817M | φ40 | 1 | 310 | 310 | 4 | 600 | 2400 |
| 18 | （空） | 大恒 | 单波长高反射镜 | GCC-101044 | φ40     532nm | 1 | 190 | 190 | 4 | 640 | 2560 |
| 19 | Beam expander | 大恒 | 扩束镜 | GCO-140113 | 8×    532nm | 1 | 956 | 956 | 3 | 964 | 2892 |
| 20 | （空） | 大恒 | V型夹持 | GCM-060202M | （空） | 1 | 170 | 170 | 3 | 370 | 1110 |
| 21 | 支撑 | 大恒 | 支杆 | GCM-030113M | 102mm | 10 | 40 | 400 | 10 | 50 | 500 |
| 22 | （空） | 大恒 | 调节支座 | GCM-030345M | 102mm | 10 | 200 | 2000 | 10 | 200 | 2000 |
| 23 | （空） | 大恒 | 支杆 | GCM-030112M | 76mm | 6 | 40 | 240 | 12 | 40 | 480 |
| 24 | （空） | 大恒 | 调节支座 | GCM-030344M | 76mm | 6 | 200 | 1200 | 12 | 190 | 2280 |
| 25 | （空） | 大恒 | 叉式压板 | GCM-5328M | / | 16 | 60 | 960 | 16 | 80 | 1280 |
| 26 | Laser | 长春光机所 | 激光器 | PGL-FS-532-50mW | 532nm | 1 | 869 | 869 | 1 | 935 | 935 |
| 27 | （空） | 定制 | 激光器转接板 | （空） | （空） | 1 | 500 | 500 | 1 | 500 | 500 |

主要光学采购行保留为：PGL-FS-532-50mW（532 nm）；GCO-140113（8×、532 nm）；GCC-101044（φ40、532 nm 单波长高反）；GCL-050003（φ25.4、消光比500:1，分行采购数量）；GCC-M402103（420–680 nm）；MBS1455-A（50:50 的用户规格图记录，工作簿原文 `400–700mm`）；GCL-010605A（f=150、φ25.4）；GCL-010650A（f=30、φ25）。标签参数来源分别是 BOM、用户规格图或待确认型号资料，不是光路实测值。


F9000 用户规格图字段（未确认装机）：型号 F9000；芯片组 DLPC900+DLP9000；0.9 inch；微镜尺寸 7.56 μm；2560×1600；VIS 400–700 nm；PC USB；同步触发接口支持；软排线 31 cm；安装方向候选 0°/45°；1–8 bit 可调；图示 1-bit 最大 9523 Hz、8-bit 最大 1031 Hz；存储 512Mb×2；倾转角 ±12°；填充因子 92%；损伤阈值图示 >20 W/cm²（并注明与散热、波段有关）；控制软件 TI GUI。以上是用户提供的规格图字段，不是本机已核实的运行状态。

CMOS 用户规格图字段（候选传感器资料，不闭合整机身份）：FSI sCMOS、Gpixel GSENSE2020、黑白、对角线 18.8 mm、有效面积 13.3×13.3 mm、2048×2048、像元 6.5×6.5 μm、峰值量子效率 72%@595 nm、典型满阱 43 ke⁻、典型读出噪声 2.1 e⁻（Median）、16-bit 40 fps/8-bit 45 fps、rolling shutter、曝光 10 μs–10 s、2×2/4×4 binning、支持 ROI、硬件/软件触发、触发输出为曝光开始/读出结束/触发准备、Hirose 触发接口、USB 3.0 数据与供电、8/12/16 bit、C-mount、<4 W、机身 56×56×61.7 mm、305 g。该图可能对应 Dhyana 401D，但相机整机铭牌未核实；这些值不作为 HW207 实测项。F9000 型号和参数来自独立用户规格图；厂家页面本轮返回 403，字段未独立复核。它不属于上述 BOM；设备资料称当前位置由临时反射镜代替，现场安装、接线和控制状态 UNKNOWN。

厂家来源复核日期为 2026-11-03。TI DLP9000 Rev. B 数据手册记录 DLP9000 为 0.9 inch、2560×1600、7.56 μm pitch、400–700 nm 和 ±12° 微镜倾角。DLP9000/DLPC900 最高 1-bit 9523 Hz；8-bit 1031 Hz 对应预载图案与照明调制，外部输入最高 360 Hz。TI 同时把 DLP9000X 与 DLPC910 单独列出，不可混用。DLP9000 芯片资料列出窗口单程透过率 95%（经两个窗口表面）、微镜反射率 88%、阵列衍射效率 86%、填充因子 92%。效率条件为 24° 照明、F/2.4 照明及收集光锥、均匀光谱与瞳面照明、100% 电子镜面 duty，且不含光学过填充损失；不能作为本机效率。TI 数据表还说明微镜铰链方向相对阵列为 45°，这与用户规格图的 0°/45° 安装选项含义不同。厂家资料未证明本机角度正负、坐标原点、手性、图像翻转、ON/OFF 光向或收光端口。FLDiscovery 产品页本次请求返回 403，因此 F9000 页面字段只标供应商资料转录，网页内容未独立复核。[TI DLP9000 数据手册 Rev. B](https://www.ti.com/lit/ds/symlink/dlp9000.pdf)；[TI DLP9000 产品页](https://www.ti.com/product/DLP9000)；[FLDiscovery F9000 页面](https://www.fldiscovery.com/product/f9000/)。

F9000 候选局部阵列坐标以像素中心为原点，`u=(column−(2560−1)/2)×7.56 μm`、`v=(row−(1600−1)/2)×7.56 μm`。候选安装角变换为 `R(α)=[[cosα,−sinα],[sinα,cosα]]`、`α∈{0°,45°}`。矩阵只定义候选正向数学坐标，实际旋转符号、台面基准、封装手性、行列翻转及原点偏移均 UNKNOWN。微镜铰链轴 45°和 ±12°状态倾角单独记录，不补造 ON/OFF 出射向量。

CMOS 用户图片列 Gpixel GSENSE2020，黑白 FSI sCMOS、2048×2048、6.5 μm、72%@595 nm、读噪典型 2.1 e⁻、满阱典型 43 ke⁻、8/12/16 bit、rolling shutter 等。图片参数组合可能对应 Dhyana 401D，但整机铭牌未闭合。其他旧记录有 13.8 ke⁻ 满阱，故实际满阱和 532 nm QE 仍 UNKNOWN，不选其中一个作为实测。曝光、增益、读出模式、行时序及相机映射也 UNKNOWN。

### 主平面理论方案和来源冲突

理想 4f thin-lens 模型以物面、H1/H2 主平面和像面为参考：物面至 H1 为 f1，H1 至 H2 为 f1+f2，H2 至像面为 f2，总长 2(f1+f2)，有符号横向倍率 `M=−f2/f1`。以下全部标为 `THEORY_4F`，非实装或标定值：

| 方案 | 输入 relay（物面/H1/H2/芯片面） | 输入 M | 输出 relay（芯片面/H1/H2/像面） | 输出 M | 总长 | 镜片需求/限制 |
|---|---|---:|---|---:|---:|---|
| 150/150 + 30/30 | 150/300/150 mm | −1 | 30/60/30 mm | −1 | 720 mm | 输入需 2 片 150 mm，输出需 2 片 30 mm。采购表数量不是库存证明。 |
| 交叉 A：150/30、30/150 | 150/180/30 mm | −0.2 | 30/180/150 mm | −5 | 720 mm | 各需 1 片 150 mm 和 1 片 30 mm，需确认可用镜片，改变 30 mm 输出配置。 |
| 交叉 B：30/150、150/30 | 30/180/150 mm | −5 | 150/180/30 mm | −0.2 | 720 mm | 各需 1 片 30 mm 和 1 片 150 mm，需确认可用镜片，改变输出链用途。 |

原始记录与先前理论值差异表：

| 项目 | 来源原值/状态 | 先前理论或其他来源 | 本次处理 |
|---|---|---|---|
| Relay L2–L1 | 距离表约 350 mm，照片估读 | 150/150 理想 4f 的 H1–H2 为 300 mm | 参考面不同，不据此推断失焦；保留 350 mm 源值 |
| 输出侧 Relay | 芯片预留位→单片 30 mm relay lens 约 80 mm；镜→CMOS 预留位约 100 mm | 30/30 理想 4f 为 30/60/30 mm | 不作实装倍率推断 |
| 720 mm | 不属于距离表测量 | 三种理想 4f 候选各自输入+输出的理论总长 | 仅为 THEORY_4F |
| 1330 mm | 13 行源值各加一次 | — | 算术合计，不是一条连续光路或 OPL |
| 1400 mm | 420+2×70+840 | 假定 BS→临时反射镜往返后接成像支路 | 条件路径假设，不是已验证 DMD 光路或 OPL |
| 30 mm 镜片数 | BOM Sheet1 为 2、Sheet2 为 3 | 旧计算曾按 2 片配置 | 保留逐表数量；采购数不等于库存或实装 |
| BS 波段字段 | BOM 原文 400-700mm | 用户规格图及 TI 芯片资料写 400–700 nm | 同时保留，记录单位冲突；实装标签待核 |
| CMOS 满阱 | 用户规格图 43 ke⁻ 典型值；旧资料另记 13.8 ke⁻ | 无可确认整机身份/同一配置依据 | 不择一；实装相机满阱与 532 nm QE 均 UNKNOWN |
| 理想芯片规格 | 100×100、2 μm、200×200 μm | HW207 目标定义 | 不表示芯片已安装或实测 |



距离 Excel 与 TSV 的 13 个编号、数值和照片对应一致；支路与端点语义可对应，但备注存在逐项差异，原文已在上表并列保留。Excel 的数据行为 5–17，备注/照片说明位于第 20–28 行；TSV 数据行为第 2–14 行。BOM 工作簿原始单元格 A:H 与行号均在 JSON 中保存。光路原图和 M1–M4 照片编号见资料包 README。校准先后与停机门槛沿用 [校准计划](207_calibration_plan.md) 和 [实验矩阵](207_experiment_matrix.md)。本 JSON 只描述几何和证据，不接入训练或运行时。
