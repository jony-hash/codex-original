# 开题 PPT 证据盘点

审计日期：2026-10-12（Asia/Shanghai）。仓库：`D:\codex-original`。课题暂定名：《基于非线性光纤复合器件的紧凑光学语义生成架构用于智能视觉生成》。本文件供聊天模式理解背景、核查开题表述使用，不是训练、模型采用或物理部署授权。

**范围与方法。** 只读检索源码、配置、文档和保存证据；对 `src/scripts/configs/docs/experiments/tools/research_assets/outputs` 使用含 ignored 文件的路径盘点，列出 62,119 个可见文件，得到 69 条历史输出目录访问错误。这个数字包含环境及辅助文件，不表示逐份读完所有文件。随后沿当前状态索引追踪模型、生产损失、推理入口，定向读取有效报告、训练历史、CSV、坐标和论文补充 PDF。以 CPU `torch.load(map_location='cpu', weights_only=True)` 解包 A4700、A5400、B5400 三个检查点，核对相位形状、任务身份和 Adam 步数；独立汇总保存的 3,500 行 CSV。收尾时新的云端终局回传到本地，另核对报告、完成标记、人口及 9,000 行 DEV CSV，补入第 8、9 节。没有模型实例化、光学/评价器前向、采样、梯度、训练或昂贵实验重跑；新报告记载的12,000次历史科学更新属于另一项已完成任务，不是本次审计执行。

根目录没有独立的 `logs/results/runs/checkpoints/figures/reports/evidence/metrics` 文件夹，相关证据主要在日期化 `outputs/.../runs/...`。没有修改代码、已有结果、状态文件或 Sources。存在访问受限历史目录，因此“未检出”仅表示可访问范围未检出，不能推断实验室里没有实物或数据。所有数值结论均采用下方明确结果路径，不使用自动 latest。

标记含义：**CURRENT** = 当前维护实现或当前研究证据，须注明是哪一种；**LEGACY** = 历史路线/结果；**UNUSED** = 此开题主线不消费；**DEBUG** = 诊断或组件候选。CURRENT 不等于通过正式科学验收。置信度“高”表示代码、检查点或保存结果可核对；“中”表示文档/示意素材中的意向或有口径限制；“缺失”表示没有相应直接证据。

# 1. Executive summary

1. 你的新主线固定为 SMF / 10×10 OLS，研究端面材料非线性的作用；“2+2、100×100、2 μm、双光子打印、类别/形态/角度控制”是拟定研究目标，不能整组当成已完成事实。来源：本次用户提示词及下述实现核对。
2. 仓库正式 v2 是 `label + 8D code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS`，没有 SMF；LightGen SMF 研究线独立于它。[E01][E02]
3. LightGen 基础运行时确实为两层 Encoder、两层 Generator、各 100×100 相位参数；532 nm、2 μm 光学平面采样、1.50 mm 传播间隔是项目参数。[E03]
4. 当前保存研究基线 A4700 及候选 B5400 实际使用两层 Encoder 加自定义四层 Generator；Generator 相位窗按光路顺序为 100×100、400×400、100×100、400×400。不是四层全部 400×400。[E04][E05][E06]
5. 当前研究模型有 20,000 个 Encoder 相位参数及 340,000 个 Generator 相位参数，总计 360,000；近期 Generator-only 对照实际只更新后者。[E04][E05][E06]
6. 图像训练路径输入 MNIST 图像，标签供软件语义损失使用；光学运行时本身没有 label/controller 输入。图像路径主要是重建/图像映射，不能仅凭 Generator 名称称为随机生成。[E03][E07]
7. 另有真实执行并保存的按类别复 OLS 采样生成：软件先按类别拟合/选择先验，再生成 100 维复系数，直接送入光学 Generator。此路径绕过 Encoder。[E08][E09]
8. 类别条件采样已有数值证据；没有检出当前 SMF 链的旋转角度条件编码、连续角度插值及相应验收结果。[E03][E08][E09]
9. 当前 A4700/A5400/B5400 主结果没有端面材料非线性。B5400 的改善来自重建目标增加前景项，不能作为 Te 或 ReLU 的收益。[E06][E07]
10. 历史 Strict ReLU90 使用固定强度阈值、保留复场相位的像素级算法代理；2 层和 4 层对照都没有支持语义收益。[E10][E11][E12]
11. 更新的历史非线性实验是 Generator 内部的强度依赖相位代理，300 步后 DEV 宏召回 0.1030→0.1024；只支持局部响应，不支持数据集语义恢复，更不是实测材料。[E13][E14]
12. 既有训练端点仍用截断、离散归一化的 step-index LP01 模型。2026-10-12 新 Gaussian 组件验证通过，但参数未实测、未替换默认 LP01 或既有检查点。[E15][E16][E17]
13. 固定先验 A5400/B5400 人审为 65/100 与 69/100，但 A 重复一致性 13/20 未达 18/20，终局为 `REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`；实测材料响应、STL 和打印实物证据也未闭合。[E09][E18][E19]
14. 最新回传的云端前景目标对照已完成：2,000行TRAIN拟合、1,000行DEV确认，三个训练seed各A/B追加2,000步至6700，类别2/4 Q_fg三个seed均重复改善。它补充了固定起点的DEV与重复轨迹证据，仍不是材料、人眼可读、TEST盲测或模型采用证据。[E47][E48]

# 2. Current authoritative configuration

| 项目 | 当前值 | 证据路径 | 置信度 | 备注 |
| --- | --- | --- | --- | --- |
| 开题方向 | 固定 SMF / 10×10 OLS；研究材料端面非线性 | 本次用户提示词 | 高（意向） | 方向更新不自动证明已有实现 |
| 正式 v1 入口 | `train.py`，LDM teacher/distillation | [E46]，`Runtime Boundaries` | 高 | LEGACY/UNUSED_FOR_THIS_TOPIC；不定义 SMF 当前研究 |
| 正式 v2 入口/配置 | `python -m ldm_project.v2`；`formal_identity_logit_amplitude_finite_crop_identity.yaml` | [E02]，配置 `system.kind` | 高 | CURRENT 正式兼容运行时，但 identity OLS 无光纤 |
| LightGen 维护的基础训练边界 | `execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py`；`LightGen2DChenBOGTComplexOLSTrainingConsumer` | [E20][E21] | 高 | CURRENT 维护实现；不是现在获准继续运行的任务 |
| LightGen 独立 YAML | 没有当前独立 LightGen research YAML；主要参数在运行时常量、任务脚本和检查点 | [E20] 的 research code map | 高 | `configs/v2/diagnostic/lightgen_amplitude_fiber_ols_400.yaml` 是另一个诊断路径，不能替代 M0 配置 |
| 当前研究状态 | A4700 保留为研究基线；B5400 保留为研究候选 | [E01]，`当前决策概览/当前停止点` | 高 | 不等于正式采用或完成语义生成 |
| 最新完成的追加研究证据 | 三seed云端前景目标DEV确认，`FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS` | [E47]，`decision/completed.json` | 高 | 本地收尾时已回传；新的六个seed-specific 6700端点未自动替换基线；不能与旧能量B6700混名 |
| 基础主模型 | `LightGen2DSemanticGenerationInputOLSAlignment` | [E03]，290–377 行 | 高 | 基础类默认 2+2；不能覆盖已扩容检查点的结构 |
| 当前端点 Generator 类 | `ExpandedGenerator` | [E04]，148–194 行 | 高 | 2+4 混合窗口版本 |
| Encoder 层数/每层尺寸 | 2 层；100×100 | [E03]，34–42、356–361 行；[E05] 检查点相位形状 | 高 | 每层 10,000 个相位参数 |
| Decoder 层数/每层尺寸 | 当前端点 4 层：100²→400²→100²→400² | [E04]，`phase_layout`；[E06] 人口记录 `source_parameter_shapes` | 高 | 计划 2+2 与已保存大模型冲突；开题中需分别陈述 |
| pixel pitch / neuron size | 相位参数格点 2×2 μm；SLM/CMOS 格点 2.5×2.5 μm | [E03]，34–42 行；[E22] `pixel_pitch_m` | 高（仿真） | 未验证打印的有效最小特征尺寸 |
| 相位窗横向尺寸 | 100×2 μm=200 μm；400×2 μm=800 μm | [E04] `new_phase_support_fov_m`；[E22] | 高（推导） | 不是完整器件/封装尺寸 |
| Encoder 相位面间隔 | 1.50 mm | [E03]，40、356–361 行 | 高（仿真） | 每个相位面之后还传播一次；详见第 4 节 |
| 末 Encoder 相位面→OLS | 1.50 mm | [E03]，313–325 行；[E23] | 高（仿真） | 不能将“层后复场节点”当作相位面位置 |
| OLS 输出→首 Decoder 相位面 | 同一数值平面，距离为 0 | [E03]，327–335、237–247 行 | 高（实现） | 没有实际光纤长度/端面间距建模 |
| 当前 Decoder 相位面位置 | 相对 OLS 为 0、0.75、1.50、2.25 mm；最终读出出口为 3.00 mm | [E04]，183–194 行；[E22] | 高（仿真） | 新面窗外复透射率 1，无新增中间硬裁切 |
| OLS 通道/端口顺序 | 10×10=100 个复系数；`j=10r+c` | [E03]，25–32、92–108 行 | 高 | 100 通道不等于 100×100 像素，也不是 100 个相位参数 |
| 当前端点光纤类型 | step-index LP01 single-mode synthetic ports；`V=2.2` | [E03]，200–215 行；[E15] | 高（模型） | 没有实测芯径、NA、MFD、端口间距 |
| FMF/MMF 残留 | `FewModeFiberArrayOLS`、`NestedMultimodeFiberArrayOLS` 及旧图件仍在仓库 | [E24][E25] | 高 | LEGACY/DEBUG/UNUSED；当前 LightGen 运行时未选它们；正式 v2 也没有光纤 selector |
| 当前新增 Gaussian | 20 μm 端口中心间距；3.5 μm Gaussian 1/e 振幅半径；532 nm | [E16][E17] | 高（组件验证） | DEBUG 候选，`PROJECT_GEOMETRY_UNMEASURED`；未替换上表端点 |
| 当前材料非线性 | 主研究端点没有材料层/Strict ReLU90 | [E04][E07] | 高 | 不可给当前端点填写历史 τ 值 |

**可代表当前研究证据的输入组合**：运行时 [E03] + `ExpandedGenerator` [E04] + A4700 [E05] 或 A5400/B5400 [E06] 检查点 + 同任务报告/人口。代表当前近期对照的入口是 [E07]，其 `DEFAULT_CHECKPOINT`、`SOURCE_STEP`、`FIXED_LAMBDA` 与终点文件共同定义任务。单独引用基础类默认值、旧 YAML、脚本文件名或日期最新目录都不充分。本审计不提供启动训练命令。

最新已完成对照另由云端入口 [E48] 与本地明确回传目录 [E47] 定义；参数仍是同一340,000相位参数Generator、无材料代理，新增的是拟合人口、训练seed和DEV测量。其源A4700路径见该任务`input_population.json.source_checkpoint`，不能凭同名文件认定不同任务所有元数据一致。

**已发现并解决的口径冲突：**

- 2+2 仍是基础类默认和拟定紧凑目标；A4700/B5400 实际为 2+4。检查点 `encoder.phase=[2,1,100,100]`、`generator.phase=[340000]` 与 `phase_layout` 一致。
- 当前前景任务的 `input_population.json` 保留父能量匹配任务的 `task_id`、`final_generator_adam_step=6700`、`updates_per_arm=2000` 及 energy-matched target 描述。该文件是**继承人口/父任务元数据**，不是本任务执行预算。当前 `fixed_parameters.json`、A/B 检查点、700 行/臂训练历史、实际 `_train_step` 与报告一致为 4700→5400、各 700 步。当前梯度入口实际用原始 `result.image_input.slm_intensity`，没有消费父任务的逐图能量缩放系数。[E06][E07][E26]
- 新 Gaussian 的 20 μm/3.5 μm 不能回填到旧 LP01 端点；二者模场、中心和归一化不同。[E15][E16]
- [E27] 可视化验证报告中的 `WAIT_FOR_RATINGS` 是交付时历史状态；当前已归档质控失败，以 [E01][E09] 为准。旧状态不改写。

# 3. Generation task definition

| 路径/判断 | 训练或推理输入 | 输出与关系 | 证据及结论 |
| --- | --- | --- | --- |
| 真实图像路径 | 训练：image，外加软件损失用 label；推理：image | 同一输入图像的 raw CMOS 强度图，通常与输入强度目标配对 | [E03] `forward_from_image`，451–477 行；[E07][E26]。CURRENT 重建/图像映射；不是以噪声为输入的端到端光学采样 |
| 经验 latent 解码 | 保存的 `z∈C^100`，来自 Encoder | 模态重发→Generator→CMOS；可和该图像目标配对 | [E03] `forward_from_ols`，479–491 行。是 latent replay/解码，不自动证明新颖生成 |
| 对角高斯类别生成 | 请求数字类别；软件随机数产生实/虚噪声，采样类条件复 OLS | 不给定输入数字图像，直接生成 CMOS；没有单张配对真值 | [E08]，`scope/sampling_parameters/results`；2000 张已保存。CURRENT 保存生成证据，但验收范围仅数值筛选 |
| 冻结联合先验类别生成 | 相同类别和固定随机噪声；每类 20 行拟合的联合复高斯 | A5400/B5400 共用 1500 个 OLS 输入，输出配对比较 | [E09][E28]。类别条件是先验选择，不是光学运行时显式 label 输入 |
| 数字类别控制 | 存在软件按类选择先验和请求类别评分 | E_D 请求类别正确数、隐藏标签的人审 | 支持“实现了按类别采样入口”；尚不足以称稳定可读的十类语义可控生成 |
| 旋转角度控制 | 当前路径没有 angle/θ 条件输入 | 未检出按指定角度生成的结果 | MISSING — REQUIRE MANUAL INPUT。不能用 S12 入射倾斜角或梯度夹角代替图像旋转角 |
| 连续角度插值 | 无当前角度编码、训练数据角色或误差验收 | 无可核验角度连续控制输出 | MISSING — REQUIRE MANUAL INPUT |
| 形态/属性操控 | 有类别形态距离、固定 latent/局部梯度响应研究 | 没有当前已验收的笔画粗细、旋转等属性控制协议 | 诊断响应不是属性可控生成验收。[E18][E26] |
| latent interpolation / semantic manipulation | 本地论文图源含作者插值/语义数据；当前工程有 latent 入口和局部扰动 | 未检出当前 A4700/B5400 的已验收连续语义插值实验 | [E29] 明确论文 CSV 为 A.U. 图源，不能当本项目复 OLS bank；当前相关结论为 INSUFFICIENT EVIDENCE |
| 正式 v2 条件生成 | label + 8D code，经电子 amplitude generator 后进 SLM | identity OLS 单次曝光路径 | [E02] 是另一运行时，不能移植成 SMF 链已完成的功能 |

对角采样定义可写为：

\[
z_{c,j}=(\mu^{R}_{c,j}+\sigma^{R}_{c,j}\epsilon^{R}_{j})
+i(\mu^{I}_{c,j}+\sigma^{I}_{c,j}\epsilon^{I}_{j}),\qquad \epsilon\sim\mathcal N(0,1).
\]

均值、总体标准差只用 200 行拟合人口（每类 20 行）；标准差下界 `1e-8`，seed `2026100813`，先抽实部、再抽虚部。该采样分布不是原始 BOGT 固定截断强度先验。[E08]

联合先验用 `x=[Re z_0,Im z_0,...,Re z_99,Im z_99]∈R^200`。每类样本总体协方差除以 20；标准化相关矩阵为 R，保留每端口 2×2 实虚 block 为 B，采用 `R_joint=0.5R+0.5B`，再 Cholesky 采样。该先验不是材料激活。[E28]，122–153 行。

样例及脚本：`execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1.py`；[E08] 所在目录的 `conditional_ols_bank.pt`、`generation_gallery_01.png` 至 `generation_gallery_20.png`；[E09] 所在目录的 `blind_review.html`、`ratings_submission.csv`；[E18] 所在目录的 `empirical_focus_gallery.html` 和 `sampled_focus_gallery.html`。路径索引见第 14 节。

# 4. Optical architecture

**训练/真实图像光路：**

```text
MNIST 28×28 image（[0,1]）
→ bilinear resize 到 200×200 intensity
→ sqrt 得到 amplitude，初始相位 0
→ amplitude SLM（200×200，2.5 μm）
→ coherent input relay（倍率 -0.4）
→ Encoder phase 1 + ASM + finite crop
→ Encoder phase 2 + ASM + finite crop
→ 100 个复 overlap coefficients（10×10 LP01 ports）
→ coefficient transport T=I100
→ W_OLS modal re-emission（100×100 complex field）
→ Generator（当前端点为 4 个混合大小相位窗）
→ output imaging relay（倍率 -2.5，B=0）
→ raw CMOS I=|E|²（200×200，2.5 μm）
→ 配对图像 loss / 软件评价器语义梯度
```

**类别采样推理：** `class + software random draw → complex OLS z → T/W_OLS → 同一个 Generator/relay → raw CMOS`。直接 OLS 推理没有 Encoder 输出；软件先验、latent 加载/写入也不能被叙述成已完成的全光随机源。[E03][E08][E09]

生产评价视图为 `adaptive_avg_pool2d(clamp(raw,0,1), (28,28))`，供冻结 E_S/E_I/E_C/E_D 软件模型使用；raw CMOS 与该视图必须区分。评价器不属于最终光学推理器件。[E30]，62–68 行。

| 位置/区间 | 当前数值 | 代码/坐标证据 | 物理边界 |
| --- | --- | --- | --- |
| SLM | z=0 | [E03] `slm_plane` | 仿真坐标起点 |
| Front relay 路段 | 25、125、73.2143、312.5、89.2857 mm；合计 625 mm | [E03]，46–58 行 | 项目 relay 约定，不是完整论文复现 |
| Encoder 相位面 1/2 | z=625 / 626.5 mm | [E03][E23] | 相位面之后的记录节点分别在 626.5 / 628 mm |
| OLS 输入/模态输出/Decoder 入口 | z=628 mm，共面 | [E03][E22] | T=I 没有真实光纤长度/传播相位 |
| 当前 Generator 相位面 | z=628 / 628.75 / 629.5 / 630.25 mm | [E04] `phase_layout`；[E22] | 100 / 400 / 100 / 400 像素相位窗 |
| Generator 出口 | z=631 mm | [E03][E22] | 最后相位面后还传播 0.75 mm |
| Back relay | 100+250=350 mm | [E03]，56–58 行 | 理想成像描述 |
| CMOS | z=981 mm | [E22] `cmos_incident_field.z_m` | 含 relay 的数值 SLM→CMOS 长度 981 mm |

基础两层相位栈是 `phase modulation → center pad → angular-spectrum propagation → center crop`，每层后中央裁剪。`2048×2048` 是传播网格，不是每层可训练光学神经元数。[E23]。大 Generator 在两个 1.5 mm 传播区间中各插入一个 400×400 相位窗，窗外透射率为 1，不在新增面追加硬裁切；原 100×100 端点裁剪边界仍在。[E04]，205–265 行。

**物理光路尚未闭合**：复系数表示、模式重发和理想 relay 已实现；真实 SMF 阵列、材料膜、衍射层制造误差及器件装配未由这些仿真文件验证。不得将共面 OLS 接口解释为实物光纤长度为零。

# 5. Nonlinear module

| 实现 | 位置/输入输出 | 公式与参数 | 训练/权重/状态 |
| --- | --- | --- | --- |
| 当前 A4700/A5400/B5400 | OLS→Generator，没有额外材料变换 | 复场由线性 T 和 W_OLS 写入；相位掩模 `E'=E exp(iφ)` | CURRENT。没有当前材料阈值；B5400 是目标函数候选 |
| N6 facet canary | 模态重发后、Decoder 前；complex field→complex field | `I'=gain·ReLU(slope·(I-0.05))`；ε=`1e-12`；复场按强度比平方根缩放 | LEGACY。固定材料代理，不是实测传输曲线；[E31] |
| Strict ReLU90 | 同上；重发的二维复场各像素计算 intensity | `I=|E|²; I_t=max(I-τ,0); E_out=E sqrt((I_t+ε)/(I+ε))`；τ=`0.10682560942718147`，gain=slope=1，ε=`1e-12` | LEGACY。τ 用 TRAIN 校准至初始功率保留 90%，非 trainable；训练内包含。[E10][E11] |
| NearIdentityNonlinearPhaseStack | 第一次 Generator 传播后、第二相位面前；complex field→complex field | `E_out=E exp(iη |E|²/(I_ref+1e-12))`；`I_ref=0.9120797964019747`；η 初始 0 | LEGACY/DEBUG。η 为可训练标量，相位改变、振幅不变；不是 SMF 端面材料。[E13][E14] |
| 评价器里的 `nn.ReLU` | E_D 等软件网络内部 | 电子神经网络激活 | UNUSED_FOR_MATERIAL_CLAIM；[E32] |
| `ConditionalNonlinearExpandedManifoldGenerator` | 另一电子 amplitude-SLM manifold 路径 | 电子生成器非线性 | DEBUG/UNUSED；不能借其名称证明端面光学非线性 |

Strict ReLU 代码证据：[E10] `LongFacetProbe.apply`，323–360 行；`_install_probe`，363–371 行，通过包装 `generator.forward` 在每次训练/测量时应用，而不是只在训练后插入。`_new_candidate` 载入共同起点，但 A/B 各自训练后终点权重不同；因此终点对照不是“完全相同权重下只开关非线性”。[E11] 的 A 复用审计说明继承已保存线性对照，而不是本轮再次训练 A。

phase shift=0 的 ReLU 缩放因子为非负实数，因此非零复场相位保留；ε 使被抑制像素的实际强度为 `I·ε/(I+ε)`，并非严格为 0。报告 `output_intensity=ReLU(I-τ)` 是理想统计量，复场实现是带 ε 的代理。τ 是仿真强度单位，未换算为 mW 或 W/cm²。[E10][E11]

**不是“每根光纤独立激活”**：现有 facet hook 接收 `W_OLS(z)` 后的 100×100 空间场，以像素计算强度；没有对 `[B,100]` 系数逐通道直接实施 ReLU。截断 LP01 ports 的空间支撑近似分离，也不能据此声称已实现 100 个实物独立非线性器件。[E03][E10][E15]

最新的已保存显式非线性实验是 [E14] 的内部相位代理；最新的当前研究端点使用 **NONE（无端面材料代理）**。当前未发现由实测 Te/CoSi/TaP 曲线拟合得到并接入训练的 `T(P_s,P_c)`、材料相位响应或可饱和吸收参数。结论为 **INSUFFICIENT EVIDENCE**，不是这些材料不能工作。[E19][E31]

# 6. SMF/OLS model

**当前训练端点：step-index LP01，不能写成已使用论文 Gaussian。** [E03] `LightGenAlignedComplexOLS` 显式选择 `mode_model='step_index_lp01_ports'`、`waist_m=7.111e-6`、`fill=0.8`、`v_number=2.2`、`facet_magnification=1`、T=I。底层 `FiberArrayOLS` 标为 synthetic diagnostic-only；上层研究接口中的 `diagnostic_only=False` 不会把其几何变成实测光纤。[E15]

标量 LP01 径向形状（[E15]，49–101、156–194 行）：

\[
\psi(\rho)=\begin{cases}
J_0(u\rho),&\rho\le1,\\
J_0(u)\,K_0(w\rho)/K_0(w),&\rho>1,
\end{cases}\quad
u^2+w^2=V^2,\quad
\frac{uJ_1(u)}{J_0(u)}=\frac{wK_1(w)}{K_0(w)}.
\]

代码取 `V=2.2<2.4048255577`，根据振幅 1/e 半径 `rho_e` 令 `a=waist/rho_e`。每根模式按所属端口单元裁切，再用离散面积积分归一化至单位功率。默认 ports 分支不做 Löwdin；`step_index_lp01_lowdin` 是另一个可选诊断模式。

\[
u_j[p]=\frac{\psi(|\mathbf r_p-\mathbf c_j|/a)\,\chi_{C_j}[p]}
{\sqrt{\Delta A\sum_p|\psi(|\mathbf r_p-\mathbf c_j|/a)\chi_{C_j}[p]|^2}},\quad
z_j=\Delta A\sum_p u_j[p]^* E[p].
\]

\[
y=Tz,\qquad E_{\rm out}[p]=\sum_{j=0}^{99}y_jv_j[p],\qquad T=I_{100}.
\]

其中 `ΔA=(2 μm)^2`。输入、输出模式缓冲保留复系数乘法，振幅和相位都传递。投影、传输、重发分别在 [E15] `project_coefficients` 239–244 行、`transmit_coefficients` 246–252 行、`reemit_coefficients` 254–261 行。

| 问题 | 当前实现/边界 | 证据 |
| --- | --- | --- |
| fundamental mode 是否 Gaussian | 旧端点不是：是 Bessel LP01+端口裁切；新 opt-in 组件是 Gaussian | [E15][E16] |
| mode radius / MFD | 旧模型振幅 1/e 半径 7.111 μm，等效直径 14.222 μm；不是实测 MFD。新 Gaussian 为 3.5 / 7 μm | [E15] `FiberArraySpec`；[E17] `geometry.json` |
| wavelength | 532 nm，模型参数 | [E03][E15][E17]；不代表 980/1550 nm 实物 |
| core/cladding | a 从模场形状反推；没有实际芯径、包层折射率/外径、NA 或厂家参数 | [E15]，169 行；[E29]；MISSING 实测 |
| 100 根空间坐标 | `c_y(r)=-80+160r/9 μm`，`c_x(c)=-80+160c/9 μm`，r,c=0…9，row-major；相邻约 17.7778 μm | [E15]，164–183 行；是数值端面坐标 |
| 输入 field→coefficients | 面积加权复 overlap integral，不是单点采样/平均池化 | [E15]，239–244 行 |
| incident angle | 生产接口无独立入射角旋钮；输入复场的倾斜相位会自然改变 overlap。没有额外矢量/偏振/NA 角度模型 | [E15]；新 Gaussian 已保存倾斜平面波组件核对 [E17] |
| 输出复场重构 | 100 个复系数乘输出模式相干求和 | [E15]，254–261 行 |
| fiber 内传播 | 不显式模拟长度、βL、色散、衰减、双折射、模间耦合；当前 T=I | [E03][E15] |
| 学习/校准 | 模式、T、relay 均 frozen；没有实测复矩阵 | [E21] `MODEL_CONTRACT_PAYLOAD.frozen`；[E29] |

**新增 Gaussian 组件** [E16] `GaussianFiberArrayOLS` 使用论文 S12 的解析归一化：

\[
g_j(\mathbf r)=\sqrt{2/\pi}\,w^{-1}\exp(-|\mathbf r-\mathbf c_j|^2/w^2),\quad
\mathbf c_{r,c}=((r-4.5)20,(c-4.5)20)\;\mu m,\quad w=3.5\;\mu m.
\]

保留共同 200 μm FOV，无每端口裁切、无离散重归一化、无正交化；仍以面积加权 overlap 和 T=I 重发。`report.md` 明确：旧 LP01 默认、latent bank、prior、power reference、训练检查点均未切换。数值组件验证不等于论文实物复现或生成收益。[E17]

**与论文的复现关系**：本地补充 PDF 第 7 页 S12/S14 为 Gaussian 模式/耦合积分，第 8 页 S22 引入入射倾斜相位；第 3 页描述 10×10 SMF、8.5 mm 长度、125 μm 外径。旧模型只借鉴 100 端口/复 overlap/重发概念，新 Gaussian 只对齐解析形式。论文没有在本地资料中闭合项目数值端口到实物阵列的倍率、实测 MFD 和传输矩阵。[E33][E29] 因而不能写“参数级复现 LightGen”。

# 7. Compactness metrics

口径：每个可训练相位格点计 1 个 optical neuron / 1 个实参数。不计 2048² 传播网格、冻结模式矩阵、电子评价器、先验均值/协方差或旧 adapter。下面比较的是**相位格点数量**，不等于同质量计算容量或封装紧凑性。

| 架构 | Encoder 参数 | Decoder 参数（逐层） | 总相位参数 | 总衍射层数 | 证据 |
| --- | ---: | ---: | ---: | ---: | --- |
| 项目基础小架构 / 拟定 2+2 | 2×100²=20,000 | 10,000+10,000=20,000 | 40,000 | 4 | [E03][E23] |
| 当前 A4700/B5400 混合大架构 | 2×100²=20,000 | 10,000+160,000+10,000+160,000=340,000 | 360,000 | 6 | [E04][E05][E06] |
| LightGen 单通道 400² 案例 | 2×400²=320,000 | 4×400²=640,000 | 960,000 | 6 | [E33] 第 2–3、38 页 |
| LightGen 单通道 600² 案例 | 2×600²=720,000 | 4×600²=1,440,000 | 2,160,000 | 6 | [E33] 第 2–3、38 页 |

近期任务的 trainable/update scope 是 Generator-only，实际 Adam 更新 340,000 个相位参数；Encoder 的 20,000 个相位参数被保存但不更新。因此“系统有 360,000 个相位自由参数”和“该次训练更新 340,000 个参数”都成立，口径不同。[E07]，220–225 行。

\[
N_{\rm phase}=\sum_\ell H_\ell W_\ell,\quad
L_x=W\Delta x,\quad L_y=H\Delta y.
\]

100²×2 μm 窗为 0.2×0.2 mm，面积 0.04 mm²；400²×2 μm 为 0.8×0.8 mm，面积 0.64 mm²。论文 400²×3 μm 为 1.2×1.2 mm；600²×3 μm 为 1.8×1.8 mm。这里的面积是相位活动区，不是基板或 holder。[E03][E04][E33]

同口径数量比仅供审计：960,000/40,000=24；2,160,000/40,000=54；960,000/360,000≈2.67；2,160,000/360,000=6。可表述为“相位格点数量分别约为参考案例的 1/24、1/54 或 1/2.67、1/6”；**不可表述为整个器件小这些倍**。当前大架构和论文之间 Encoder/Generator 分配、FOV、边界、pixel pitch、量化及损耗不同，没有相同性能条件。[E04][E33]

数值轴向跨度：Encoder 入口→OLS=`2×1.5=3 mm`；Generator 入口→出口=3 mm；共面 OLS 不增加长度；入口到最终出口=6 mm。当前大架构第一至最后相位面间距=5.25 mm。含两侧 relay 的 SLM→CMOS=625+6+350=981 mm。[E03][E04][E22] 这些数值省略实际光纤长度、基板/膜厚及封装间隙，不能当作已制造的 6 mm 器件。

项目 holder / packaging 外形：**MISSING — REQUIRE MANUAL INPUT**。论文自身 holder 为 27.3×5 mm=136.5 mm²，Encoder/OLS/Generator footprint 分别 6.2×2.7、8.5×3.25、11.9×2.7 mm（补充第 39 页 Table S4）。这是论文实物，不是本项目封装。[E33]

# 8. Training protocol

本节先列已归档前景任务（A4700→A5400/B5400）的协议，再给出基础BOGT及最新回传云端三seed协议；避免将不同人口、预算、v1 distillation或历史adapter训练合并。

| 项目 | 当前近期对照的真实取值 | 证据/边界 |
| --- | --- | --- |
| Dataset | 本地 MNIST；image `[N,1,28,28]`；类别 0–9 | [E06][E07]；`data/MNIST/processed/training.pt` 已存在 |
| 基础角色划分 | 60,000 train-source 中分层取每类 500 作 DEV，共 5,000；其余 55,000 TRAIN；official test 10,000 protected | [E21]，39–41 行；[E34] 1340–1373 行，seed=1234 |
| 本次拟合/确认人口 | 拟合 200（20/类），确认 300（30/类），共 500；两者都属 canonical TRAIN | [E06] `input_population.json`；[E07] 35–46 行 |
| 确认人口语义 | 本任务未参与更新、调权、终点选择；不代表独立 TEST 泛化 | [E06] `report.md` |
| 输入/输出分辨率 | 数据 28²；SLM/CMOS 200²；OLS 空间场 100²；评价视图 28² | [E03][E30] |
| Augmentation | 当前入口没有随机旋转、裁切等 augmentation；uint8/255，确定性 bilinear resize | [E03] 156–164 行；[E35] `load_local_mnist` |
| Optimizer | Adam；betas=(0.9,0.999)，eps=1e-8，weight_decay=0，amsgrad=False | [E06] `fixed_parameters.json`；本轮实际检查点解包一致 |
| Learning rate | 0.001 | 同上 |
| Batch | 逻辑 batch=10，每批每类 1 行；20 个固定拟合批次循环；最新梯度入口逐行生产前向后合成 batch 梯度 | [E07] 190–225 行；[E26] 244–307 行 |
| Epochs / updates | 不是整套 55,000 数据的 epoch 训练；各追加 700 更新，Generator Adam step 4700→5400 | [E06][E07]；每臂保存历史 700 行、4701→5400 |
| 类平衡重复次数 | 700/20=35 次/拟合批次；200 行各呈现 35 次/臂 | 当前调度 `batch_index=local_update%20` 推导；[E07] |
| A 臂重建 | `R=raw200 MSE+0.1·(1−SSIM28)` | [E36] 230–238 行；[E26] 273–285 行 |
| B 臂重建 | `R+λQ_fg`，λ=0.11159457624315472 | [E06] `training_arm_objectives/lambda`；[E26] 159–168 行 |
| 光功率约束 | 五个既定平面的 frozen-reference log-power penalty | [E21] 632–671 行 |
| KL / L_dist | 存在：归一化 OLS 强度的 factorized positive-support truncated-normal KL；对 Generator 的梯度严格为 0 | [E21] `MODEL_CONTRACT_PAYLOAD.q/p`；[E04] 589–598 行；[E26] 301 行 |
| Semantic 项 | 三个冻结软件评价器 E_S/E_I/E_C 的类标签 cross-entropy 梯度，归一化后组合；E_D 不参与训练 | [E26] 279–302 行；[E37] 800–815 行 |
| Loss/梯度权重 | α=0.01596223111734585；β=2.4509953404250195；γ=0.5330424284576345；κ=0.6936758419764932；δ_SIC=1.365988374680088e-5 | [E38] 44–50 行；生产组合 [E26] 302 行 |
| Diversity loss | 当前本次训练没有显式 diversity loss、LPIPS 或 FID loss | [E26]，生产组合全部项；不能把采样库像素方差当训练 loss |
| teacher/DDPM/VAE/student | 当前 LightGen 没有 teacher、DDPM、电子 VAE decoder 或 teacher/student 蒸馏链；有光学 Encoder/Generator 和冻结评价器 | [E03][E21][E26]；v1 另行维护 [E01] |
| Trainable/update | 仅 Generator 相位 340,000；Encoder、OLS 模式/T、relay、参考量、评价器 frozen | [E07] 220–225 行；[E06] 检查点/预检 |
| Numeric mode | strict FP32 optics，Q_fg/统计使用 float64；不用 AMP/TF32 | [E07] `NUMERIC_MODE`；[E26] |
| Checkpoint 选择 | 预设固定终点 5400；中间量只作观测，不按最好 loss/最大分类器分数选模 | [E07] 35–46、740–760 行；[E06] |
| 当前阶段 | 保存证据诊断/冻结生成与识读分析；A4700 研究基线、B5400 候选；非正式部署就绪 | [E01][E18] |

生产的更新方向不能简写成普通标量“CE 加权总 loss”：

\[
\widehat g_i=\nabla_{(\theta_E,\theta_G)}CE_i/\|\nabla CE_i\|_2,
\quad g_{\rm can}=\delta_{\rm SIC}\,
\frac{\widehat g_S+\widehat g_I+\widehat g_C}
{\operatorname{RMS}(\widehat g_S+\widehat g_I+\widehat g_C)}.
\]

\[
g_A=\alpha g_{\rm dist,G}+\beta\nabla_G R+\gamma\nabla_G L_{op}+\kappa g_{\rm can,G},
\qquad g_B=g_A+\beta\lambda\nabla_GQ_{fg},\quad g_{\rm dist,G}=0.
\]

代码先计算两组参数的语义梯度，在组合后只把 Generator 分量写入 `.grad` 给 Adam。α 项保留统计/历史定义，不会因 Decoder-only 训练改善 Encoder 的 latent 分布。[E26][E37]

设 raw intensity 为 I、目标为 t，`p=I/ΣI`，`q=t/Σt`，`F={p: t[p]>0.1}`，则：

\[
Q_{fg}=\frac{\sum_{p\in F}(p_p-q_p)^2}{\sum_p q_p^2}.
\]

该归一化含整幅图像的总强度，前景项不能忽略分母对所有像素的依赖。它是形态误差，不是效率、可读性或多样性指标。[E26] 113–142 行。

基础 BOGT 的 `u_j=|z_j|²/(s_j+1e-12)`，s_j 是 θ0 TRAIN 平均模态功率。Q 用 class-balanced batch 的总体矩和方差下界 1e-8 定义，P 是 `TN_+(loc=1,variance=0.25)`，KL 使用真实截断归一化项；基础 `L_total=αL_dist+βL_rec+γL_op`，原始 `L_rec` 为配对 raw MSE，后续目标已如上改变。[E21] 86–129、601–625 行。

**最新云端三seed协议（已回传终局）**：审计收尾时本地新增 [E47] 的完整报告、`completed.json`、CSV及检查点；转移收据状态为COMPLETE。本轮只读取本地回传，不读取远端、启动/暂停或重跑任务。

| 项目 | 最新云端任务 | 证据 |
| --- | --- | --- |
| 入口/身份 | `execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1.py`；`M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1` | [E48]，44–90行 |
| 来源/架构 | 同一A4700模型与Adam step4700；Generator=340,000参数，2048传播网格 | [E47] `preflight.checks.source_model_and_optimizer` |
| 人口 | 2,000 TRAIN拟合、每类200；1,000 canonical DEV确认、每类100 | [E47] `input_population.json`；TRAIN/DEV角色分离 |
| 训练 | 三seed 2026101201/1202/1203，每seed同起点A/B同批次序列；各2,000更新，固定200拟合批次×10 epochs，batch10；六条轨迹共12,000更新 | [E47][E48] |
| 目标与优化器 | A保留生产目标，B加固定λ=0.11159457624315472的Q_fg；Adam lr0.001、(0.9,0.999)、eps1e-8、weight_decay0，继承Adam不reset | [E47] `fixed_parameters.json/training_arm_objectives` |
| 冻结范围 | 仅Generator相位更新；Encoder autograd保留用于全参数语义梯度归一化，但Encoder不由优化器更新；四评价器冻结 | [E47]；不是移除Encoder语义导数的代理 |
| 固定终点 | 六个端点均Adam step6700，不按DEV最佳点选模 | [E47] `final_generator_adam_step` |
| 主判定 | 类别2/4共200行DEV，三seed分别要求D=Q_B−Q_A低于负容差且Q_B<Q0 | [E47] `decision`；DEV未用于该次更新/调权 |
| 正式回读 | 9,000行DEV读出/CSV（起点+两个历史端点+六个新端点）；10500行固定OLS采样（7情景×1500） | [E47] `validation.json/sampling` |
| 采样 | 固定joint prior，sample seeds 2026101221…2026101225；仅DESCRIPTIVE_ONLY | [E47] `conditional_sample_summary.json`；不是新识读gate |
| 状态边界 | `VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`；`training_recommendation=NONE`、`next_action=NONE` | [E47] `report.json/completed.json` |

DEV曾用于评价器开发，因此这次是Generator的held-out DEV确认，不是全项目盲测或canonical TEST验收。此前跨平台严格重放仍为`REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`，没有放宽原容差；本云端任务使用独立本机预检和重复性，不声称严格复现Windows。[E47][E45]

# 9. Existing quantitative results

## 9.1 Linear vs nonlinear：可验证的历史对照

这些是研究任务的有效保存结果，不是实物材料结果，也不是当前已验收正式生成模型。

| 同任务条件 | Linear | Nonlinear | 对照结论/来源 |
| --- | ---: | ---: | --- |
| Strict ReLU90，2 层 Generator，各历史新增 2000 步；5000 DEV | Smooth macro recall 0.6170；`L_image=0.0457815547` | 0.6138；`L_image=0.0463036330` | Δrecall=-0.0032，paired 95% CI=[-0.00641,0]；`STRICT_TRAIN_CALIBRATED_RELU90_NOT_SUPPORTED`。[E11]；两臂终点权重不同 |
| LightGen-aligned 4 层历史 Generator，各 2000 步；5000 DEV | C: 0.5774 | D ReLU90: 0.5724 | Δ=-0.005，paired 95% CI=[-0.0094,-0.0004]；`LIGHTGEN_ALIGNED_4L_RELU_GAIN_NOT_SUPPORTED`。[E12]。这个历史 4 层模型不是当前混合窗口 ExpandedGenerator |
| 4 层 linear 单独延长 2000→3000 步 | 0.5774→0.5790 | 未运行匹配 nonlinear 延长 | 仅+0.0016；不支持自动到4000；不能与上行2000步D作公平新对照。[E40] |
| 内部相位 nonlinear，2 层、300 步 | 自身 η=0 起点 DEV Smooth recall=0.1030；MSE=0.0687408655 | 终点0.1024；MSE=0.0671431288 | Δrecall=-0.0006，CI=[-0.0020,0.0006]；`NONLINEAR_LOCAL_GAIN_WITHOUT_DATASET_RECOVERY`。[E14]；是起点→终点，不能冒充平行线性臂 |

ReLU τ、初始功率保留：TRAIN calibration 1000 行与 disjoint TRAIN validation 1000 行，τ=0.10682560942718147，保留率约 0.9000000 / 0.9015110。[E11] 的 calibration 文件。它证明代理在初始校准场上造成非零功率变化；不是物理衍射效率、绝对阈值或重复训练收益。

历史检查点与 run id（均已定位）：

- A linear：`D:/codex-original/outputs/20260915_ideal_facet_relu_long/runs/lightgen_2d_ideal_facet_relu_long_a_linear_attempt_05_minimal_ondemand/a_linear_step_02000_checkpoint.v1.pt`；任务步 2000、历史 global step 2600。[E11] `candidates.A_LINEAR_REUSED.checkpoint`。
- B Strict ReLU90：`D:/codex-original/outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_b_relu90_attempt_03/b_relu90_step_02000_checkpoint.v2.pt`；任务步 2000、global step 2600。[E11]。
- D 4L ReLU90：`D:/codex-original/outputs/20260915_lightgen_aligned_4l_relu_gain_v1/runs/lightgen_aligned_4l_relu_gain_v1_d_relu90_attempt_01/d_4l_relu90_step_02000_checkpoint.v1.pt`；[E12] 的 D 报告。
- 内部相位 nonlinear：`D:/codex-original/outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/scientific_step_000300.pt`；[E14]。

评价器：历史对照主要使用冻结 `SmoothMNISTEvaluator v2.1` 的 DEV 宏召回，不能与当前 E_D 数值合并。Strict paired bootstrap seed=271828，partition seed=1931；历史 sampler seed 与模型 RNG 必须消费对应合同/检查点，不从一个统计 seed 推断训练 seed。[E11][E14]。

**最优 nonlinear 配置：INSUFFICIENT EVIDENCE。** 上表仅有固定代理、预算和起点的比较，没有验证的参数优化搜索、跨训练种子稳定收益或真实材料排名。τ90 是校准定义，不是“最佳阈值”。

## 9.2 当前无材料代理的研究端点

下表同一 300 行确认人口，类别各 30 行，均属 canonical TRAIN；本轮从 [E06] `paired_rows.csv` 独立汇总并与报告相符。Q_fg 的主量只消费类别2/4的60行，其余指标消费全300行。

| 指标 | A4700 基线 | A5400 原目标 | B5400 前景附加项 |
| --- | ---: | ---: | ---: |
| E_D 请求类别正确数 | 230/300 | 237/300 | 242/300 |
| E_D accuracy / balanced macro recall | 76.667% | 79.000% | 80.667% |
| 类别2/4等权 Q_fg | 0.44411628 | 0.42808681 | 0.36127857 |
| Raw CMOS MSE，全300行 | 0.05541527 | 0.05291420 | 0.05015811 |
| SSIM28，全300行 | 0.28556578 | 0.32990552 | 0.33519938 |
| 生产 Q28，全300行 | 0.44996242 | 0.40816197 | 0.35291378 |
| 模拟 CMOS 积分强度均值 | 1.89506408e-8 | 1.89043146e-8 | 1.88004282e-8 |

主差 `Q_fg(B5400)−Q_fg(A5400)=-0.06680824`，相对下降约15.61%，仅支持该任务固定起点、λ、700步和人口下的相对数值收益。积分强度有 pixel-area 因子，但光场没有实验绝对功率标定，**不能标为 W、J 或实测 diffraction efficiency**。[E06][E26]

近期训练继承共同模型、Adam/RNG 起点，每臂只有一条预设训练轨迹；五个采样 seeds `2026100961`…`2026100965` 不是五条训练轨迹。人口扩展的 split/row seed 为2026100811/2026100812，基础初始化/角色 seed=1234；实际复现需相应检查点内 RNG 和显式 sample rows。[E06][E28][E34]

E_D 是冻结 `PixelMLPConfirmationEvaluatorV1`：784→512→256→10，两个电子 ReLU，只用于确认读数；检查点 `D:/codex-original/outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt`。E_S/E_I/E_C 是训练语义梯度提供者，不与 E_D 混用。[E32][E06] `source_input_paths`。

其他已归档结果：

| 结果人口/任务 | 已保存数量/主量 | 正式边界 |
| --- | --- | --- |
| A4700 对角类条件采样 2000 张 | E_D=1513/2000，macro recall=75.65%；固定 seed2026100813 | `CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED` 是数值入口筛选，不是人眼可读/独立泛化验收。[E08] |
| 固定联合先验 A5400/B5400，各1500张 | E_D=1202/1500 vs1214/1500；shape28集合距离0.02666559 vs0.02296276 | 样本没有逐张目标，不能填写配对重建MSE；集合距离不是FID。[E09] |
| 同任务100对、270题单人评分 | 65/100 vs69/100；仅A3、仅B7、双方62、双方不认可28；exact p=0.34375 | A隐藏重复13/20未达18/20；B19/20；`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`。[E09] |
| 当前类别2/4保存证据定位 | `D_margin=0.0007479456`；类别2近邻8、类别4近邻9 | `CLASS_SEPARATION_INCREASED` 是数值形态集合分类分离，仍不证明人工识读。[E18] |
| 公平小/大架构1000步对照 | E_D=114/300 vs166/300；MSE=0.0582866 vs0.0584632；类别8受损 | 支持有限分类器读数改善，不能称全面成像质量改善；不是 nonlinear 对照。[E27] |
| 能量匹配 A6700/B6700 | E_D=243/300 vs257/300；Q约改善5.14%，未达10%；候选对A4700积分强度约94.71% | 父任务 `MIXED_OR_INSUFFICIENT`；不得作为已采用最优模型。[E27][E41] |

| 用户要求的指标 | 当前相关运行的证据状态 |
| --- | --- |
| FID | 未发现当前所选 SMF/材料对照的有效 FID 报告；MISSING，不填0；论文图源或其他v1指标不能移用 |
| Semantic/classification accuracy | 当前有 E_D 数量；历史有 Smooth macro recall，见上表；均需保留评价器、人口与视图口径 |
| Precision | 当前所选正式研究报告未列标准 precision 汇总；MISSING；本轮没有创建新分类评测 |
| Recall | 上述宏召回存在；类别均衡时 accuracy 数量与 macro recall数值一致；不能当成人工识读率 |
| LPIPS | 当前所选任务未检出；MISSING |
| Diversity | A4700对角采样保存描述性 mean_pixel_variance=0.0046577104、同库最近生成样本view RMSE均值0.048527848、8-bit exact duplicate excess=0；不是覆盖/新颖性或LPIPS验收。[E08] |
| Reconstruction | 经验图像输入有raw MSE/SSIM；随机采样没有逐图配对真值，配对reconstruction不适用 |
| Output diffraction efficiency | 有模拟积分强度、mode coupling、crop/power retention辅助量；没有已标定光学输入输出功率/材料损耗的系统效率验收 |
| 当前最新材料 integrated run | 包括新云端任务在内，未检出以实测Te/CoSi/TaP响应为输入的有效运行；INSUFFICIENT EVIDENCE |

## 9.3 最新云端三seed DEV确认：无材料代理

[E47] 的主量以同一类别2/4 DEV人口（100行/类，共200行）计算。共同起点Q0=0.4377299823。下表Q_fg与E_D计数均由本轮读取的9,000行CSV独立汇总，符合报告；E_D辅助量则使用完整1,000行DEV。

| 训练seed | A原目标 Q_fg | B前景项 Q_fg | B−A | E_D全DEV A/B |
| --- | ---: | ---: | ---: | ---: |
| 2026101201 | 0.3896659281 | 0.3191550831 | -0.0705108450 | 824/1000；840/1000 |
| 2026101202 | 0.3901176451 | 0.3195266620 | -0.0705909831 | 822/1000；839/1000 |
| 2026101203 | 0.3898325656 | 0.3194221251 | -0.0704104404 | 827/1000；839/1000 |

三个seed均满足预设主规则，B相对A前景误差降低约18.1%；每个B也低于共同起点。结论现在可以写成“固定A4700起点、固定λ、扩大拟合人口及每臂2000步范围内，类别2/4前景目标的DEV收益在三个训练seed上重复”。**不应继续把该固定目标的DEV/重复轨迹证据写成完全缺失。** 但不能外推为多起点普遍稳定性、物理材料增益、全项目盲测、人眼可读或模型采用。DEV历史上用于评价器开发；canonical TEST仍未用于这次验收。[E47]

历史A5400/B5400在相同新DEV人口上的E_D为793/1000与808/1000，只作描述，不参与新主判定。新端点名称是`seed_2026101201/02/03_A/B_step6700`；它们不是原能量对照的A6700/B6700。

新检查点位于`D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/checkpoints/seed_2026101201_arm_a_step6700.pt`及对应`arm_b`、seed1202/1203六份文件。本轮只核对存在性和报告绑定，没有加载这些新模型或重跑其原始光学读出；云端保存的独立回读验证消费9,000行DEV及10,500行采样，本轮额外核对主CSV和完成标记。采样仍为描述性，不新增人审。

对重点句子的审计结论：**历史算法代理可以改变场/强度，已有对照没有建立稳定、显著的语义性能收益；这句话不能扩写为实测非线性光纤复合器件已证明上述效果。** 对实测材料系统，输出变化和语义收益均为 **INSUFFICIENT EVIDENCE**。最新无材料候选的数值收益也不能改写这个结论。

# 10. Fabrication readiness

| 问题 | 当前证据 | 结论 |
| --- | --- | --- |
| phase→height 转换 | 当前训练是 `exp(iφ)` 理想薄相位面；未检出当前SMF模型的已标定height exporter | MISSING — REQUIRE MANUAL INPUT |
| 理论换算 | 透射薄元件可写 `h(φ)=λφ/[2π(n_print−n_env)]`，`Δh_2π=λ/(n_print−n_env)` | 仅条件公式；n、基准高度、工艺相位标定缺失，不能计算项目实际高度 |
| 打印材料折射率 | 未检出当前resist在工作波长下的实测/厂家参数 | MISSING — REQUIRE MANUAL INPUT；不能借用论文SiO₂折射率 |
| 设计波长 | 仿真532nm | 高置信仿真参数；实际打印/材料实验波长仍待用户确认 |
| 连续高度/量化 | 当前相位参数连续；没有当前打印量化规则和height-map | 不能由连续相位推出连续高度已加工 |
| 100×100 STL / 2μm STL | 可访问文件路径盘点未检出 `.stl` | MISSING — REQUIRE MANUAL INPUT；访问受限目录不作存在性推断 |
| 2μm严格定义 | phase-plane格点/像素边长，即100格点对应200μm活动窗 | 不是打印线宽/体素大小/最小特征尺寸或fiber pitch。[E03][E04] |
| 器件横向尺寸 | 基础活动区200×200μm；当前新增窗800×800μm | 不是基板/封装实测；后者MISSING |
| 层厚/基底厚度 | 模型零厚度相位面；1.5mm是传播间隔 | 不能作为打印层厚；MISSING |
| 当前打印参数 | resin、设备/物镜、NA、laser power、scan speed、hatching、slicing、support均未检出直接实验记录 | MISSING — REQUIRE MANUAL INPUT |
| 打印时间 | 当前版本无估算依据或实际计时日志 | MISSING — REQUIRE MANUAL INPUT |
| 实物打印记录 | 没有检出对应当前模型的工艺记录、显微/SEM/AFM、形貌标定与测量数据 | 未验证；不能断言实验室未打印 |
| 2+2全部加工文件 | 没有当前4层、2μm、完整geometry/phase-height绑定的可加工文件包 | INSUFFICIENT EVIDENCE；当前研究大模型实际又是6层 |

找到的旧工具 [E19] `export_v67_phase_candidate.py` 只将四个100×100相位图模2π后写为 `.npy`；manifest显式 `height_map_exported=False`、`height_calibration='requires measured n(532nm) and height-phase calibration'`。这是 LEGACY diagnostic-only 的相位候选导出，不是当前已生成STL，更不是打印完成证据。

论文加工（与本项目分开）：补充第3页 S1 是 `φ=2π(n−n0)d/λ`，3μm像素、JGS1/SiO₂、8级刻蚀、150nm级差、0.6mm基板；相位级0至5.71rad；工艺为overlay photolithography。不能将论文的多次光刻加工或其holder打印等同于本项目双光子打印衍射层。[E33]

# 11. Nonlinear fiber-composite device readiness

| 问题 | 可核验状态 | 已完成/计划区分 |
| --- | --- | --- |
| 当前优先材料 | 汇报素材把Te标为“基线候选”，CoSi/TaP为其他候选 | 仅候选定位；没有验证材料排名。[E42]，171–173、427–429行 |
| 是否制备 | 可访问仓库未检出当前样品编号、制备批次/工艺、实物测量记录 | MISSING — REQUIRE MANUAL INPUT；不是“确认未制备” |
| 制备工艺 | 未检出样品对应的sputtering/沉积方法、设备、recipe | MISSING |
| 镀膜位置 | 10×10 SMF输出端面材料层在概念图中出现 | 设计/示意位置，不是实际端面沉积验证。[E43] |
| 膜厚 | 未检出样品厚度/均匀性/表征 | MISSING |
| fiber类型 | 研究方向是SMF；实际型号、单模工作范围、core/NA/MFD未核对 | 模型选择≠样品规格 |
| 工作波长 | 代码532nm；材料器件signal/pump波长无当前样品实验记录 | 不把980/1550nm自动赋予pump/signal角色；MISSING |
| 单光束/pump+signal | 没有可核验样品光路及两束光角色/功率/脉冲条件 | MISSING |
| transmission vs power | 素材中的`t_m(I)`是统一接口示意；未检出带单位、原始功率、参考归一化和样品身份的实测曲线 | MISSING；ReLU90校准数据不是材料测量。[E42] |
| nonlinear absorption | 没有当前样品α(I)、饱和/阈值参数及误差报告 | MISSING |
| phase response | 未检出Δφ(P)、interferometry、hysteresis或mode变形证据 | MISSING |
| 原始数据/图 | 已找到材料机制SVG/PNG和visual QA；不是实测透过率、吸收或相位数据 | 示例为 [E43]；图件质量QA不是材料实验验收 |
| 100-port阵列器件 | 没有当前复合阵列测量、通道一致性/串扰、热稳定性和mode保持数据 | MISSING |
| 与仿真耦合 | 当前无由实测材料曲线拟合、与复场接口绑定的模型 | INSUFFICIENT EVIDENCE |

可用于解释缺口的**条件接口**为：

\[
E_{out,j}=\sqrt{T_j(P_s,P_c,\lambda,pol,\ldots)}\,
e^{i\Delta\phi_j(P_s,P_c,\ldots)}E_{in,j}.
\]

这只是所需变量的表达式；仓库没有当前样品的T、Δφ及功率到仿真单位映射。透过率曲线本身不是ReLU输出关系：单光束条件下 `P_out=P_in·T(P_in)`，归一化T、非线性吸收及阈值神经元响应不可直接等同。没有测量时不补参数，不把候选材料写成已验证器件。

# 12. Legacy branches that must NOT enter the opening presentation

这里的“不得进入”指不得作为当前主线已完成事实或当前有效配置；必要的历史负结果可带日期、路径和边界呈现。

| 内容 | 标记 | 为什么不能混入当前 |
| --- | --- | --- |
| SMF/FMF/MMF横向扩展为正式主线 | LEGACY/UNUSED | 本次用户明确撤出；存在few/nested-mode实现不构成当前主线 |
| FMF文献模式示意与旧方向资产 | LEGACY | 参考LP11/LP21图不是当前SMF数据或硬件 |
| v1 teacher/DDPM/VAE/student distillation | LEGACY/UNUSED_FOR_TOPIC | 正式维护但与当前teacher-free LightGen研究链分离 |
| 正式v2 label+8D identity OLS | CURRENT compatibility / UNUSED_FOR_SMF_CLAIM | 不能借其标签输入证明SMF研究链已有同一控制功能 |
| N3 latent producer、rank-9/16 adapter、controller映射 | LEGACY/support | 支持分支的latent/软件变换结果不等于当前纯光学Generator成功。[E44] |
| Strict ReLU90/N6 facet | LEGACY algorithmic proxy | 与实测Te/CoSi/TaP和当前无材料大模型不同 |
| Generator内部η相位代理 | LEGACY/DEBUG | 插在Decoder内部；不是光纤端面材料；未证明数据集恢复 |
| 旧400×400诊断YAML或既有step600 replay | DEBUG/reference | 与当前混合大模型不同，不可用“400”文件名推断全系统每层400² |
| old transparent_identity、phase-amplitude upper bound、v67相位导出 | LEGACY/DEBUG | 不能当当前finite_crop实物加工/正式acceptance |
| 失败工程attempt、缺失终局的启动日志 | DEBUG/INVALID_HARNESS | 工程失败不产生科学负结果；启动不证明完成 |
| `VALID_FAILED/MIXED_OR_INSUFFICIENT`候选 | 历史有效负/不足结果 | 能支持受限判断，但不授权promotion或“最佳模型” |
| B5400/B6700的局部指标收益 | CURRENT研究候选 | 不能写成材料收益、稳定人工可读、TEST泛化或已采用模型 |
| 新Gaussian组件 | DEBUG未标定候选 | 组件独立核对不自动迁移latent/prior/checkpoint或替代旧训练证据 |
| 论文source_data与论文SEM/AFM/holder参数 | 外部参考 | 作者结果和硬件不能算本项目已完成实验 |

# 13. Missing information

以下各项描述需要补全的背景事实/证据，不构成新实验方案或执行请求。

## 13.1 研究目标架构与已保存大模型的最终定位

- **MISSING ITEM:** 开题最终采用2+2紧凑目标还是将2+4大模型列为目前诊断基线；是否已有正式冻结版本。
- **WHY IT MATTERS FOR THE OPENING PPT:** 决定层数、neuron数和“紧凑”表述，避免把计划2+2和360,000参数结果拼成同一系统。
- **CAN IT BE FOUND IN REPO:** 两种实现/检查点都可找到；最终开题定位未由数值结果决定。[E03][E04][E05]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 对两种架构在开题中的身份说明，以及如有的正式设计版本/已批准几何文件。

## 13.2 实物SMF阵列几何与型号

- **MISSING ITEM:** 样品型号、core/cladding尺寸、NA、工作单模波段、实测MFD、端口中心间距、阵列FOV、端面relay倍率。
- **WHY IT MATTERS FOR THE OPENING PPT:** 判断数值17.7778μm或20μm中心间距如何对应实物；2μm不是fiber pitch。
- **CAN IT BE FOUND IN REPO:** 有synthetic LP01与未标定Gaussian参数；没有该样品测量。[E15][E17][E29]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 厂家规格、样品照片/尺度及实测几何/MFD记录；无需凭经验补值。

## 13.3 光纤内部/阵列复传输

- **MISSING ITEM:** 长度、通道损耗/相位、串扰、偏振依赖、复传输矩阵及不确定度。
- **WHY IT MATTERS FOR THE OPENING PPT:** T=I只是一种仿真假设，不能证明100通道硬件相位保持和材料集成后的模态行为。
- **CAN IT BE FOUND IN REPO:** 现有T=I或synthetic pool；没有实测矩阵。[E03][E15][E25]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 对应阵列的测量记录、仪器和校准条件、原始数据。

## 13.4 材料样品身份与制备

- **MISSING ITEM:** 优先材料是否正式选定，是否已制备、样品编号、沉积方法、膜厚、位置和均匀性。
- **WHY IT MATTERS FOR THE OPENING PPT:** 决定可以写“候选/拟集成”还是“已制备/已测量”。
- **CAN IT BE FOUND IN REPO:** 只有Te基线候选、CoSi/TaP候选素材。[E42][E43]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 样品清单、工艺recipe、膜厚/形貌表征、日期和原始图；若仅计划，明确说明。

## 13.5 材料实验光路与波长

- **MISSING ITEM:** signal/pump角色、波长、单/双光束、绝对功率、光斑、CW/脉冲、偏振、探测器通道。
- **WHY IT MATTERS FOR THE OPENING PPT:** 确定532nm仿真与材料实验是否同一条件；避免错误指定980/1550nm角色。
- **CAN IT BE FOUND IN REPO:** 没有当前样品完整实验条件；仿真532nm不补足该信息。
- **IF NOT, WHAT THE USER MUST PROVIDE:** 实验光路图、器件表、beam role、功率/时域/探测器设置记录。

## 13.6 实测强度非线性传输

- **MISSING ITEM:** 带单位与样品身份的T(P)或T(P_s,P_c)、P_out(P_in)、参照归一化、误差/重复性和工作范围。
- **WHY IT MATTERS FOR THE OPENING PPT:** 才能区分非线性吸收、饱和响应与ReLU式阈值；不能用理想τ90代替。
- **CAN IT BE FOUND IN REPO:** 未检出；facet脚本明确`measured_transfer_curve=False`。[E31][E42]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 原始CSV/仪器导出及曲线、无材料参照、单位和归一化方法；如数据在仓库外给具体来源。

## 13.7 材料相位、动力学与模式保持

- **MISSING ITEM:** Δφ、response time、hysteresis/drift、近远场、LP01保持、邻通道串扰。
- **WHY IT MATTERS FOR THE OPENING PPT:** 光学生成消费复场，仅透过率不能闭合相位/模态传输。
- **CAN IT BE FOUND IN REPO:** 未检出对应实测；零phase_shift为代理假设。[E10][E14]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 相干相位/时域/模式测量资料，或明确尚未测量。

## 13.8 双光子打印材料与phase-height标定

- **MISSING ITEM:** resist、n(λ)、损耗、h↔φ标定、基准/最大高度、量化、加工公差。
- **WHY IT MATTERS FOR THE OPENING PPT:** 无n无法计算2π高度；连续φ不代表高度已可制造。
- **CAN IT BE FOUND IN REPO:** 旧导出器明确height未导出且需要n/height标定。[E19]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 厂家/实测n、实际设计波长、phase-height曲线和工艺限制。

## 13.9 当前版本完整加工文件

- **MISSING ITEM:** 对应各层的STL/height-map、像素节距单位、active aperture、substrate和版本绑定。
- **WHY IT MATTERS FOR THE OPENING PPT:** 决定能否写“完成可加工设计”；当前100²/400²混合窗尤其不能用旧4层npy代替。
- **CAN IT BE FOUND IN REPO:** 可访问范围未检出STL；旧phase npy exporter不是height/STL。[E19]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 实际导出文件及几何/材料参数说明，若在外部目录需明确路径。

## 13.10 打印与封装实物证据

- **MISSING ITEM:** 打印设备/NA/power/speed/hatching/slicing、时间、实物测量、层厚、holder尺寸和对准误差。
- **WHY IT MATTERS FOR THE OPENING PPT:** 计算真实紧凑性、加工可行性与完成状态；不能将6mm仿真跨度当封装。
- **CAN IT BE FOUND IN REPO:** 未检出当前样品的工艺/装配记录；论文holder只能作参考。[E33]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 工艺日志、照片/SEM/AFM、计时记录、CAD及尺寸测量。

## 13.11 当前角度/属性控制证据

- **MISSING ITEM:** θ编码、角度标注数据、控制入口、指定角度样本和连续插值验证；形态属性控制也缺同类证据。
- **WHY IT MATTERS FOR THE OPENING PPT:** 当前按类别先验采样不能证明角度/笔画属性可控。
- **CAN IT BE FOUND IN REPO:** 当前运行时/采样器没有角度条件；论文插值CSV不是本项目结果。[E03][E08][E29]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 已有实现/数据/样例路径；若不存在则只将其列为研究目标。

## 13.12 稳定人工可读性

- **MISSING ITEM:** 通过预定审阅质量要求的可读性结果，以及指标与识读之间的有效性证据。
- **WHY IT MATTERS FOR THE OPENING PPT:** E_D、Q、shape距离不是人眼可读性；当前人审不能支持收益。
- **CAN IT BE FOUND IN REPO:** 原始回答及质控失败已齐全；缺的是通过要求的结论，不是少一份日志。[E09][E18]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 若仓库外已有有效评价，提供人口/评分/质控和协议；否则明确未验证，不自动追加人审。

## 13.13 TEST泛化、多起点稳定性与采样覆盖

- **MISSING ITEM:** 全项目未用过的TEST验收、多模型起点的普遍稳定性、生成覆盖和新颖性。
- **WHY IT MATTERS FOR THE OPENING PPT:** 固定起点三条训练seed的DEV前景收益，仍不能覆盖上述更强结论；五个采样seed也不是五次训练。
- **CAN IT BE FOUND IN REPO:** 新云端报告已提供固定起点的三seed DEV收益，需纳入已完成事实；DEV曾用于评价器开发、TEST和新颖性仍缺。[E47]。旧200/300 TRAIN结论保留。[E06]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 若另有更强验收，提供对应模型、人群角色和协议；否则只保留这里的限定缺口，不补跑昂贵实验。

## 13.14 FID/precision/LPIPS及物理效率

- **MISSING ITEM:** 当前声明任务的有效FID、precision、LPIPS、系统功率效率及各自评价协议。
- **WHY IT MATTERS FOR THE OPENING PPT:** 指标名相近也不能借用论文、其他runtime或不同人口的数据。
- **CAN IT BE FOUND IN REPO:** 相关任务有E_D/SSIM/Q/描述性方差与模拟能量；没有上述完整结果。[E08][E09]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 如已有则提供数据人口、评价器/特征空间、样本数、公式与原始报告；否则标MISSING。

## 13.15 实测材料相对匹配baseline的系统结果

- **MISSING ITEM:** 将实测材料响应接入同一复OLS链后的完整比较及公平权重/损耗/预算说明。
- **WHY IT MATTERS FOR THE OPENING PPT:** 这是支持“非线性光纤复合器件改善语义生成”的直接证据，目前理想代理不够。
- **CAN IT BE FOUND IN REPO:** 没有检出；当前数值收益来自loss/容量/先验变化，不是材料。[E06][E11][E14]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 若已在仓库外完成，提供明确实现、baseline、权重、损耗、种子和终局数据；若未完成就写研究问题。

## 13.16 跨平台严格重放差异

- **MISSING ITEM:** 此前跨平台raw/logit差异的解释/正式处理；新云端终局本身已到达，不再列为缺失。
- **WHY IT MATTERS FOR THE OPENING PPT:** 回传或环境验收不能当成新科学结果；迁移差异不能因分类相同被自动判通过。
- **CAN IT BE FOUND IN REPO:** 20261011保存`REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`；新任务保留其终局和容差，以独立云端本机预检准入，新报告并未声称消除跨平台差异。[E47][E45]
- **IF NOT, WHAT THE USER MUST PROVIDE:** 若已有差异定位/正式处置记录，提供明确证据；否则保留重放边界。本轮不访问云端、不变更容差或续跑。

# 14. Recommended evidence files for PPT

只列路径和用途；这是证据定位表，不是PPT目录。

| 文件路径 | 用途/限制 |
| --- | --- |
| [CURRENT_STATUS.md](D:/codex-original/docs/CURRENT_STATUS.md) | 当前研究基线、候选、终局和工程迁移状态；状态索引不是原始测量 |
| [最新云端终局](D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/report.json) | 三个训练seed的DEV前景收益；无材料、人审或模型采用结论 |
| [最新云端收尾](D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/closeout.md) | 人口、固定预算、独立本机预检与严格跨平台重放未通过的边界 |
| [最新DEV逐行数据](D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/paired_rows.csv) | 9000行读数，可核对三seed主量及ED辅助计数；旧端点仅作描述 |
| [最新采样汇总](D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/conditional_sample_summary.json) | 7情景×1500张固定OLS采样，仅描述性指标 |
| [当前交接](D:/codex-original/docs/status/current_handoff.md) | 保存证据停止点、科学边界、无自动后继动作 |
| [基础光学运行时](D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py:25) | 输入/OLS/relay和2+2默认参数；与大模型区分 |
| [ExpandedGenerator](D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py:148) | 当前混合窗口层数、位置、参数数量 |
| [LP01实现](D:/codex-original/src/ldm_project/v2/optics/fiber_operator.py:156) | 模式、坐标、复overlap、T和重发公式 |
| [Gaussian组件几何](D:/codex-original/outputs/20261012_lightgen_gaussian_fiber_geometry_v1/runs/diagnostic-only_attempt_01/geometry.json) | 未标定Gaussian候选，不能当已有训练默认 |
| [Gaussian组件报告](D:/codex-original/outputs/20261012_lightgen_gaussian_fiber_geometry_v1/runs/diagnostic-only_attempt_01/report.md) | 组件数值验证与未切换/未实测边界 |
| [A4700检查点](D:/codex-original/outputs/20261008_m0_generator_training_population_expansion_comparison_v1/runs/attempt_03/arm_a_step4700.pt) | 当前研究基线的保存模型/Adam状态 |
| [前景训练报告](D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/report.json) | 当前A/B数值主差、人口和固定预算 |
| [前景逐行数据](D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/paired_rows.csv) | 3500行经验图像输出指标；能核对主量和分类器计数 |
| [A5400检查点](D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/arm_a_step5400.pt) | 同起点同700步的原目标对照 |
| [B5400检查点](D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/arm_b_step5400.pt) | 前景项研究候选；不是材料候选或已采用模型 |
| [A4700条件采样报告](D:/codex-original/outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_02_microbatch1/report.json) | 定义“生成”的推理输入和2000张数值入口结果 |
| [条件采样图库首页](D:/codex-original/outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_02_microbatch1/generation_gallery_01.png) | 固定显示范围、带E_D预测的示例；完整20页应配合report，不能代表人眼认可 |
| [固定联合先验识读报告](D:/codex-original/outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/report.json) | 65/69描述、重复质控失败和最终不确定结论 |
| [固定先验评分明细](D:/codex-original/outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/ratings_submission.csv) | 原始单人评分，须按盲映射/首次展示消费 |
| [类别2/4定位报告](D:/codex-original/outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/report.md) | 数值类别分离改善与仍近邻8/9的事实 |
| [经验输出图库](D:/codex-original/outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/empirical_focus_gallery.html) | 类别2/4目标及A4700/A5400/B5400经验输出 |
| [采样配对图库](D:/codex-original/outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/sampled_focus_gallery.html) | 固定先验焦点类别A/B保存样本，不新增人审结论 |
| [架构复场坐标](D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/field_coordinate_metadata.json) | 真实保存节点的shape/pitch/FOV/z；场节点≠相位面 |
| [架构可视化验证报告](D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/validation_report.md) | 小/大架构与候选展示的口径限制；其中旧等待评分状态已过时 |
| [前景CMOS例图](D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/figures/foreground_cmos_cases.png) | 当前候选经验图像示例；不是按类别随机生成成功证据 |
| [前景Q_fg对照图](D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/figures/foreground_qfg_comparison.svg) | 有固定任务范围的数值收益图，不是nonlinear材料收益 |
| [历史Strict ReLU90汇总](D:/codex-original/outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03/strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json) | 需要说明历史代理负结果时使用，标LEGACY |
| [历史4L ReLU对照](D:/codex-original/outputs/20260915_lightgen_aligned_4l_relu_gain_v1/runs/lightgen_aligned_4l_relu_gain_v1_aggregate_attempt_01/lightgen_aligned_4l_relu_gain_v1_aggregate_report.v1.json) | 限定历史架构的未支持结果 |
| [内部相位代理报告](D:/codex-original/outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/report.json) | 排除把局部nonlinear响应等同于数据集恢复或端面材料 |
| [材料候选示意](D:/codex-original/outputs/_legacy_2026q3/smf/smf_array_composite_device_v1/smf_array_composite_device.svg) | 仅概念图，必须标“拟定/示意”；不是器件照片 |
| [材料素材边界](D:/codex-original/outputs/_legacy_2026q3/ppt/ppt_slide04_material_candidates_v1/slide04_text_layout.md) | 证明素材没有材料响应曲线、排名或性能结论 |
| [旧相位导出器](D:/codex-original/experiments/archive/v57_v67/tools/export_v67_phase_candidate.py:28) | 说明相位.npy与height/STL/打印之间的缺口 |
| [论文证据包说明](D:/codex-original/research_assets/lightgen/science_adv7434/README.md) | 图源语义、未知物理参数及论文/项目边界 |
| [LightGen补充材料](D:/codex-original/research_assets/lightgen/science_adv7434/paper/sciencesm.pdf) | 参考层数/格点/加工及Gaussian公式；重点PDF第2–3、7–8、12、38–39页 |

### 正文引用定位

下列链接的行号来自本次实际读取；JSON以所写字段定位。不能用链接数量代替科学验证。

[E01]: D:/codex-original/docs/CURRENT_STATUS.md
[E02]: D:/codex-original/docs/architecture/v2_runtime.md:23
[E03]: D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py:25
[E04]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py:148
[E05]: D:/codex-original/outputs/20261008_m0_generator_training_population_expansion_comparison_v1/runs/attempt_03/arm_a_step4700.pt
[E06]: D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/report.json
[E07]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_foreground_objective_training_comparison_v1.py:35
[E08]: D:/codex-original/outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_02_microbatch1/report.json
[E09]: D:/codex-original/outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/report.json
[E10]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_ideal_facet_relu_long_v1.py:299
[E11]: D:/codex-original/outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03/strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json
[E12]: D:/codex-original/outputs/20260915_lightgen_aligned_4l_relu_gain_v1/runs/lightgen_aligned_4l_relu_gain_v1_aggregate_attempt_01/lightgen_aligned_4l_relu_gain_v1_aggregate_report.v1.json
[E13]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py:203
[E14]: D:/codex-original/outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/report.json
[E15]: D:/codex-original/src/ldm_project/v2/optics/fiber_operator.py:19
[E16]: D:/codex-original/src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py:23
[E17]: D:/codex-original/outputs/20261012_lightgen_gaussian_fiber_geometry_v1/runs/diagnostic-only_attempt_01/report.md
[E18]: D:/codex-original/outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/report.json
[E19]: D:/codex-original/experiments/archive/v57_v67/tools/export_v67_phase_candidate.py:28
[E20]: D:/codex-original/docs/research/project_map.md:29
[E21]: D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py:86
[E22]: D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/field_coordinate_metadata.json
[E23]: D:/codex-original/src/ldm_project/v2/optics/d2nn.py:13
[E24]: D:/codex-original/src/ldm_project/v2/optics/few_mode_fiber_operator.py:78
[E25]: D:/codex-original/docs/status/v2_fiber_ols_research_v1.md
[E26]: D:/codex-original/scripts/research/lightgen/diagnose_lightgen_2d_generator_foreground_objective_gradient_v1.py:244
[E27]: D:/codex-original/outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/validation_report.md
[E28]: D:/codex-original/scripts/research/lightgen/audit_lightgen_2d_generator_frozen_joint_prior_comparison_v1.py:122
[E29]: D:/codex-original/research_assets/lightgen/science_adv7434/README.md
[E30]: D:/codex-original/src/ldm_project/v2/application/research/lightgen/canonical_semantics.py:62
[E31]: D:/codex-original/scripts/research/optical_manifold/execute_lightgen_2d_nonlinear_fiber_facet_activation_canary_v1.py:219
[E32]: D:/codex-original/src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py:17
[E33]: D:/codex-original/research_assets/lightgen/science_adv7434/paper/sciencesm.pdf
[E34]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py:1340
[E35]: D:/codex-original/src/ldm_project/v2/student/evaluator.py:115
[E36]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_stroke_fidelity_comparison_v1.py:230
[E37]: D:/codex-original/scripts/research/lightgen/diagnose_lightgen_2d_class17_mapping_v2.py:800
[E38]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_semantic_target_comparison_v1.py:44
[E39]: D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/attempt_01
[E40]: D:/codex-original/outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_aggregate_attempt_01/4l_linear_optimization_ceiling_v1_aggregate_report.v1.json
[E41]: D:/codex-original/outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/report.json
[E42]: D:/codex-original/scripts/ppt_assets/render_slide04_material_candidates.py:171
[E43]: D:/codex-original/outputs/_legacy_2026q3/smf/smf_array_composite_device_v1/smf_array_composite_device.svg
[E44]: D:/codex-original/docs/research/lightgen_research_route_registry.md:22
[E45]: D:/codex-original/outputs/20261011_autodl_fixed_endpoint_replay_v1/runs/attempt_01/report.json
[E46]: D:/codex-original/docs/PROJECT_MEMORY.md:8
[E47]: D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/report.json
[E48]: D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1.py:44
