# 开题背景：可直接发送到聊天模式

以下是截至2026-10-12对 `D:\codex-original` 的只读证据盘点。请用这些事实理解课题背景，严格区分“已经实现”“有限证据支持”“研究目标”和“缺失信息”。不要把计划补成事实，不要把数值指标改善写成材料或实物系统成功；当前不需要设计PPT目录或提出新的研究路线。

课题暂定标题为《基于非线性光纤复合器件的紧凑光学语义生成架构用于智能视觉生成》。研究方向已固定为单模光纤SMF、10×10 OLS，不再以SMF/FMF/MMF横向比较为正式主线。重点是SMF输出端集成非线性材料后，光学潜在表示与数字类别、形态、旋转等语义/属性生成如何变化。衍射层拟采用双光子打印，目标紧凑架构约为2层Encoder+2层Decoder、每层100×100格点、2μm像素。**这些是目标，其中部分与当前研究模型不同。**

## 已有架构与目标的区别

仓库有三类容易混淆的系统：v1的teacher/distillation历史线；正式v2的`label+8D code→振幅SLM→Encoder→identity OLS→Decoder→CMOS`兼容运行时；以及本课题相关的独立LightGen SMF研究线。正式v2的identity OLS没有光纤，不能拿其功能证明SMF路线已有同样能力。

LightGen基础运行时确实是2+2、100×100相位面。当前保存研究基线A4700与候选B5400已经换成**两层Encoder+四层自定义Generator**：Encoder各100×100；Generator按光路顺序为100×100、400×400、100×100、400×400。Encoder有20,000个相位参数，Generator有340,000个，全系统共360,000个；近期任务仅更新Generator。拟定2+2小系统则共40,000个相位参数。不能把40,000参数的小架构和大模型结果拼成同一系统。

仿真波长532nm；衍射面格点2μm，100×100活动窗为200×200μm，400×400新增窗为800×800μm。SLM/CMOS为200×200、2.5μm采样。Encoder相位面间隔1.5mm；大Generator相位面相对入口位于0、0.75、1.5、2.25mm，出口为3mm。2048×2048是传播计算网格，不是每层可训练神经元数。

模型中OLS与Decoder入口共面，没有显式模拟实物光纤长度。Encoder入口至最终Generator出口的数值传播跨度约6mm；若计入两侧relay，SLM到CMOS的数值距离为981mm。因此尚不能声称整个已制造系统长6mm或给出封装缩小倍数。

## 当前“生成”的实际含义

真实图像训练路径是：MNIST 28×28图像→resize为200×200强度→sqrt振幅SLM→Encoder→100个复OLS系数→模式重发→Generator→CMOS强度。数字标签用于软件语义损失，光学运行时自身没有label或角度输入。这条路径主要是重建/图像映射。

另有实际保存的类别条件采样生成：软件按请求类别选择拟合的复OLS先验，用随机数产生100维复系数，再绕过Encoder直接进入同一Generator。它确实没有输入目标图像，但类别控制来自软件先验选择，不能写成已完成的全光随机源或label直接驱动光路。

已经保存A4700的2000张对角高斯采样，以及A5400/B5400共用固定联合先验的各1500张采样。当前没有检出此SMF模型的旋转角度编码、连续角度插值或已验收的形态属性控制。论文插值图源、光束入射倾角、梯度夹角不能替代本项目图像旋转控制。

## 非线性和SMF实现

当前A4700/A5400/B5400主结果没有材料非线性层。B5400的数值改善来自增加前景重建项，不能归因于Te或ReLU。

历史Strict ReLU90代理插在OLS模式重发后、Decoder前，对二维复场逐像素计算：

`I=|E|²; E_out=E·sqrt((max(I−τ,0)+1e−12)/(I+1e−12))`

固定τ=0.10682560942718147，按TRAIN校准使初始功率保留约90%；不是绝对功率阈值，不可训练，保留非零复场相位。它作用于重发后的二维场，不能直接称为100根光纤分别独立激活。A/B共同起点但分别训练，终点权重不同。

历史另有Decoder内部强度依赖相位代理`E_out=E·exp(iη|E|²/(I_ref+1e−12))`，η可训练。它不是SMF端面、Te或已测量Kerr模型。

当前已有训练检查点仍采用synthetic step-index LP01模式，V=2.2、振幅1/e半径7.111μm；在端口单元内裁切并离散归一化。自由空间复场通过面积加权overlap积分映射为100个复系数，T=I100，再相干重发，保留系数振幅与相位。数值中心间距约17.7778μm；2μm是光场采样，不能当作光纤中心间距。实物core/NA/MFD、阵列倍率和复传输矩阵尚未标定。

2026-10-12新增Gaussian组件使用20μm中心间距、3.5μm振幅半径和论文S12解析归一化，完成了组件数值验证；明确属于未标定候选，没有替换旧LP01、prior或已有检查点。不能将它的参数回填到旧结果。

## 已有结果及其限度

历史2层Generator线性/Strict ReLU90对照，在5000 DEV上的Smooth评价器宏召回为61.70%/61.38%；历史4层对照为57.74%/57.24%。均未支持非线性收益。内部相位代理300步后DEV宏召回10.30%→10.24%，仅支持局部响应，未恢复数据集语义性能。

近期无材料代理的A4700→A5400/B5400对照，从相同模型、Adam和RNG起点各追加700步，仅更新Generator。拟合200行、确认300行均属canonical TRAIN；确认行未用于该次更新、调权或选终点，但不是独立TEST泛化。

| 同一300行确认人口 | A4700 | A5400 | B5400 |
| --- | ---: | ---: | ---: |
| 冻结E_D请求类别正确数 | 230/300 | 237/300 | 242/300 |
| 类别2/4前景归一化误差Q_fg | 0.44411628 | 0.42808681 | 0.36127857 |
| Raw CMOS MSE | 0.05541527 | 0.05291420 | 0.05015811 |
| SSIM28 | 0.28556578 | 0.32990552 | 0.33519938 |

B5400相对A5400的类别2/4 Q_fg下降约15.61%，这是固定起点、目标权重和700步范围内的数值收益，每臂只有一条训练轨迹。五个采样seed不是五次独立训练。E_D是冻结软件分类器，不是光学推理器件，也不等于人眼识读。

固定联合先验的100对单人识读描述为A65/100、B69/100，精确p=0.34375。但A隐藏重复一致性13/20未达18/20，最终为`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`；不能声称B已改善可读性。之后保存证据显示类别2/4的数值形态分离改善，但最接近竞争类仍为8/9，也不能替代人审。

**最新补充：2026-10-12云端终局已在本次盘点收尾时回传。** 同一A4700起点扩大到2000行TRAIN拟合，使用1000行canonical DEV确认；三个训练seed各A/B追加2000步至6700，共六条轨迹、12000次历史更新。B仅增加相同固定前景项，没有材料非线性。类别2/4 DEV前景误差如下：

| 训练seed | A原目标 Q_fg | B前景项 Q_fg | E_D全1000行DEV A/B |
| --- | ---: | ---: | ---: |
| 2026101201 | 0.38966593 | 0.31915508 | 824/840 |
| 2026101202 | 0.39011765 | 0.31952666 | 822/839 |
| 2026101203 | 0.38983257 | 0.31942213 | 827/839 |

共同起点Q0=0.43772998，三个seed均满足B低于A及起点的预设要求，终局为`VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`，相对A降低约18.1%。这补充了**固定起点下三seed重复的DEV前景收益**，不能再把该范围的DEV/重复轨迹证据列为完全缺失。DEV此前用于评价器开发，因此不是全项目盲测；TEST、人眼可读、材料作用和模型采用仍未验证。新seed-specific六个6700端点不能与旧能量任务的A6700/B6700混名；附属采样也仅为描述性。此次审计只回读本地报告和CSV，没有执行这12000次更新。

当前可用表述是：**历史算法非线性代理能够改变光场或强度，但尚未建立稳定、显著的语义性能收益；对实测SMF-材料复合器件的系统效果，证据不足。** 不能把前景loss、容量或采样先验的收益转写成材料收益。

## 需要补充的实验室背景

Te在现有素材中是基线候选，CoSi/TaP是其他候选；这些SVG/PNG是机制示意，未提供材料响应或性能排名。仓库未检出当前样品的制备批次、膜厚、端面位置实测、带单位的transmission-vs-power、非线性吸收、相位响应、动力学和100通道一致性数据。不能据此断言实验室尚未制备；需由用户补充已有实物/数据的事实。980nm与1550nm的pump/signal角色也不能自动指定。

双光子打印方面，当前是连续理想相位参数；旧导出工具只写100×100相位`.npy`，明确没有height-map。可访问范围未检出当前版本STL、打印材料n(λ)、phase-height标定、完整各层加工包、工艺参数、打印时间或实物表征。理论上`Δh_2π=λ/(n_print−n_env)`，但n缺失，不能给实际高度。2μm目前仅是仿真像素节距。

还缺可靠人工可读性、TEST盲测/多起点普遍稳定性、采样覆盖和新颖性、角度属性控制、当前任务FID/precision/LPIPS以及已标定系统衍射效率。固定起点三seed DEV前景收益已如上得到支持，不能将这个局部完成事实与剩余缺口混写。此前Windows/云端严格冻结重放仍为`REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`，新云端任务用独立本机预检准入，没有声称严格复现Windows。

目前保留A4700为研究基线、B5400为研究候选。完成背景整理不授权继续训练、采用候选、追加人审或物理部署。

## 证据来源

完整14节审计及路径/行号见 [OPENING_PPT_EVIDENCE_AUDIT.md](D:/codex-original/OPENING_PPT_EVIDENCE_AUDIT.md)。若聊天模式无法访问本地路径，请同时上传这两份文件；下面这些路径是来源索引，不假设聊天模式能直接打开D盘。

- 当前状态：`D:/codex-original/docs/CURRENT_STATUS.md`。
- 运行时：`D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py`。
- 大Generator：`D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py`。
- 当前对照：`D:/codex-original/outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/report.json`及`paired_rows.csv`。
- 最新三seed DEV终局：`D:/codex-original/outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/report.json`、`closeout.md`及`paired_rows.csv`。
- 识读终局：`D:/codex-original/outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/report.json`。
- 当前定位：`D:/codex-original/outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/report.json`。
- Gaussian组件：`D:/codex-original/outputs/20261012_lightgen_gaussian_fiber_geometry_v1/runs/diagnostic-only_attempt_01/geometry.json`及`report.md`。
- 论文参考：`D:/codex-original/research_assets/lightgen/science_adv7434/paper/sciencesm.pdf`，与本项目结果分开使用。
