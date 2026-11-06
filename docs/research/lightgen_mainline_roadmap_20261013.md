# LightGen 主线路线图：前景收益后的数字身份定位与工程状态

## 2026-10-30 G1固定零相位教师资格：TRAINING_REFERENCE_INSUFFICIENT

`LIGHTGEN_G1_FIXED_PHASE_TEACHER_DISTILLATION_V1`按计划生成训练2000条、开发200条和新留出2000条固定人口；新留出在训练期间保持未测量。零相位合法命令教师资格处理到第1,394条后停止：云端汇总为1,393/2,000通过，索引1393（类别6、proposal 193）在5,000步预算下未通过指定相位和完整双网格误差标准；剩余606条未求解。Windows从保存的目标、float32命令、两网格复系数/CMOS及直接Decoder参考独立复算，系数误差约`8.126e-2`、CMOS误差约`1.487e-1`，保存值与独立指标差不超过`2.8e-16`。

首次教师运行的CUDA/CPU设备错配故障被归为`INVALID_HARNESS`并保留；未保存第500步FISTA游标，故目标0按499步保守上界扣减，修复后实际125步通过，累计上界624。该工程修复没有改变科学阈值。独立验证了一个完整失败教师，足以阻断全体教师准入；**没有READY、学生科学更新、终态留出测量或语义判断**。因此当前结论只针对固定零相位教师在固定训练人口与单目标5,000步预算下未能覆盖该目标，不表示其他相位、总体G1或真实光路不可达。

本候选按计划停止，不扩大相位搜索、不更换样本、不放宽阈值、不启动蒸馏。后续若研究教师策略或相位分布，需要另行明确方案、预算和独立留出。历史`VALID_FAILED`、`TRAINING_REFERENCE_INSUFFICIENT`、`PROJECT_GEOMETRY_UNMEASURED`和340张待评分状态保持。详见[五部分收尾报告](../../outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/windows_verify_failure_01/closeout_report.md)、[Windows独立复核](../../outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/windows_verify_failure_01/independent_review.json)、[教师失败回传收据](../../outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/autodl_backup_teacher_failure_01/transfer_receipt.json)及[固定人口回传收据](../../outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/autodl_backup_population_01/transfer_receipt.json)。

## 2026-10-30 G1来源、语义与20目标相位诊断：完成并停止

`LIGHTGEN_G1_FIXED_FRAME_TRANSFER_DIAGNOSIS_V1`已完成零更新复核。修订报告确认2,400条固定物理相位终点命令及2,000条V2留出命令与CUDA独立进程重放逐位一致；Windows CPU命令核对最大差`1.97e-6`。历史完整双网格见证属于训练人口，不属于开发人口。

在新物理相位先验下的已消费开发集20条中，冻结V2、训练终点和零相位前向消融均为0/20通过六项`1e-4`标准；相位0固定求解、终点相位固定求解及八起点自由相位求解均为20/20。每类两条全部通过。该结果支持这20条在冻结数值模型内存在合法振幅命令，且允许目标公共相位不是这些见证的必要条件；当前一次前向映射仍未产生这些命令。它不证明其他开发目标、总体G1人口或实物光路可达。

语义复盘显示，在同一新先验留出上，V2和15,000步终点的E_D/E_C净变化仅-0.4/-0.2个百分点。旧G1直接Decoder的75.60%/78.45%与新先验输出不是同一系数人口，不能作配对因果归因。语义评价器不是人工可读性证据。

Windows消费者完成来源绑定、实际测量重算、分解独立核对与回传大小检查，最终`SAVED_EVIDENCE_VERIFIED`；模型、光学、先验和评价器更新均为0。见[本轮五部分收尾报告](../../outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/closeout_report_01.md)、[Windows复核](../../outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/windows_verify_02/saved_verification.json)、[云端测量摘要](../../outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_attempt_02/measurement_summary.json)、[来源修订](../../outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_source_review_01/report_correction.md)及[语义归因摘要](../../outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/semantic_attribution_01/semantic_attribution_summary.json)。

本诊断已停止。下一项可决策的研究可考虑在另行批准后，用合法振幅命令训练一次前向命令映射，并新建未消费人口；本结果不授权该训练。历史`VALID_FAILED`、`TRAINING_REFERENCE_INSUFFICIENT`、`PROJECT_GEOMETRY_UNMEASURED`和340张待评分状态不变。

## 2026-10-30 G1固定物理相位语义候选：VALID_FAILED

`LIGHTGEN_G1_FIXED_FRAME_SEMANTIC_V1`已完成一次固定预算候选，选择物理相位单高斯`alpha=1.0`，映射器更新15,000步。训练人口200/200通过六项数值`1e-4`标准；历史开发200条0/200；新固定留出2,000条仅19/2,000通过，且19条全为class 1。两网格间的系数与原生CMOS误差全部通过，失败来自目标重现误差。

语义代理门槛通过：E_D/E_C准确率85.90%/86.75%，分别比旧G1直接Decoder参照高10.30/8.30个百分点；每类图像距离最低保留95.88%。这支持该人口上的评价器代理保留，不等同于人工可读性。训练人口全过、开发和留出明显较弱，说明当前候选在人口外的数值制备不足，但本次不能将差异唯一归因于高斯先验、优化暴露或网络泛化。

云端判定`VALID_FAILED`经Windows全量重算确认；回传768项共8,136,362,299字节、收据逐项匹配，Windows `SAVED_EVIDENCE_VERIFIED`。详见[收尾报告](../../outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/windows_verify_01/closeout_report.md)、[Windows验证摘要](../../outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/windows_verify_01/verification_summary.json)、[重算判定](../../outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/windows_verify_01/recomputed_final_decision.json)、[原始测量及报告备份](../../outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01)和[回传收据](../../outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/transfer_receipt.json)。

本候选到此停止，不自动续训、调alpha、扩人口、量化、推进G4、人工评分或实物实验。旧`LIGHTGEN_G1_SEMANTIC_EXPOSURE200_V1`的`TRAINING_REFERENCE_INSUFFICIENT`、历史有效失败、`PROJECT_GEOMETRY_UNMEASURED`与340张待评分均保持；后继研究须另行设计并授权。

## 前序停止点：2026-10-29 G1语义生成200目标训练参照资格不足

`LIGHTGEN_G1_SEMANTIC_EXPOSURE200_V1`经连续公共相位修订后，Windows消费者判定`REFERENCE_INSUFFICIENT_VERIFIED`。固定训练人口只检查5/200条：4条通过完整六项`1e-4`标准，class 0、proposal 4未通过，余下195条未求解。未生成READY、未创建正式attempt、未启动15,000步训练，科学更新为0。

proposal 4新增4次固定相位FISTA求解、共20,000步；38个对偶方向覆盖完整公共相位周期，最小归一化下界`1.521607e-3`高于`1e-4`。此下界仅适用于冻结响应矩阵及`a∈[0,1]`的数值模型，不证明实物不可达或G1总体不可制备。保存指令完整光路系数误差约`9.286e-3`/`9.283e-3`，原生CMOS误差约`1.680e-2`/`1.679e-2`，跨网格两项通过。首轮复数比较器缺陷已修复并有回归；矩阵训练路径和恢复工程资格此前已通过，但不构成本次训练READY。历史语义复盘的直接Decoder、冻结V2、B5000 E_D/E_C准确率相近且配对变化混合，不能把语义变化归因于映射器。

当前停止原因是200/200参照条件未满足，不是`VALID_FAILED`。不继续余下195条、不扩求解预算、不换人口、不启动15,000步训练。`PROJECT_GEOMETRY_UNMEASURED`、340张待评分、G4、量化、人工评分和实物实验边界均保持。后续若改变训练参照或人口，须先另行明确可改变决策的新方案。证据见[收尾报告](../../outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/audit_extension_02/qualification_verification_repair_01/closeout_report.md)、[Windows资格复核](../../outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/audit_extension_02/qualification_verification_repair_01/qualification_verification.json)、[修订回传收据](../../outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/autodl_backup_reference_continuous_phase_repair_01/transfer_receipt.json)、[矩阵/恢复资格摘要](../../outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/autodl_backup_engine_qualification_repair_01/engine_qualification_summary.json)、[历史语义复盘](../../outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/audit_repair_01/semantic_review.md)和[当前交接](../status/current_handoff.md)。

主线内容更新日期：2026-10-14；Gaussian/OLS物理分支更新：2026-10-18。三种子Generator训练的前景收益判定保持不变；全人口身份混淆、保存raw形态/强度读数和固定样本图册已完成，但具体笔画解释仍不足。云端恢复工程attempt_02未通过冻结终点包络，不能作为后续长训练的恢复验收。三项工作阶段停点见[收尾报告](../status/lightgen_three_workstreams_closeout_20261014.md)；身份定位明细见[本轮分析报告](../../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/report.md)及[AutoDL回传副本](../../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/autodl_backup_cloud_attempt_01)。Gaussian/OLS物理分支最新v12固定前端MNIST重建有效未通过；这不改变上述Generator结论。

## Gaussian/OLS 物理分支（2026-10-18 v12 当前停止点）

固定前端MNIST对照已在AutoDL完成：2×2-C与3×3-B各500步Decoder相位更新，共1,000次重建更新。任务退出码0，主结论`VALID_FAILED`；没有正式Generator训练、task evaluator/FID、loss修改、材料前向或官方DEV/TEST读取。Gaussian S12为主参考，LP01继续是同相位敏感性对照。

两候选初始和500步终点均通过4096/6144/8192网格共同ROI数值资格，complex64最大误差分别为2×2-C初始/终点`4.83e-6 / 4.91e-5`、3×3-B`1.02e-5 / 4.55e-5`。然而，终点MSE分别为`0.49972`、`0.08354`，都高于100张拟合集生成的固定均值图基线`0.05724`；与类内循环错配输出的配对收益区间也跨0。因此，这轮没有证明输入相关重建收益，初始到终点loss下降本身不足以通过Gate。

功率与读出限制仍明显不同：2×2-C平均Fiber输出约`16.10 μW`，但1 ms代理相机终点平均57.55%像素饱和；3×3-B约`76.64 nW`到Fiber、`39.92 nW`到相机，基本未饱和但终点仍不如拟合集均值图。相机及光纤参数均为未测量假设。结果只约束本轮数据人口、冻结前端、相机公式与500步预算；不证明架构容量不足、不选择OLS维数，也不验证真实硬件或语义生成。详见[v12报告](../research/ols/ols_decoder_mnist_reconstruction_v12.md)、[机器artifact](../../artifacts/ols_decoder_mnist_reconstruction_v12.json)、[51项回传证据](../../outputs/20261018_ols_decoder_mnist_reconstruction_v12/runs/autodl_backup_attempt_01_evidence)。

本阶段停止。可建议后续另行做固定相机下的饱和/功率/相位表征误差归因；建议不授权增加步数、调曝光/gain、接入材料、采用OLS维数或启动正式生成训练。历史G3 FAIL和旧Generator恢复`39/45`保持不变且独立。

## Gaussian/OLS 物理分支历史（2026-10-18 v11 可微链工程验收）

固定v9 2×2-C与3×3-B候选、代理相机和显式4f主路径后，AutoDL v11完成Gaussian S12可微前向、相机损失反向、Decoder两相位屏三步更新及step-1检查点独立恢复。两个候选均通过，合计10次工程Adam调用；训练集、评价器/FID、材料和正式生成训练均未使用。初始值、双层梯度和负控通过，4096/6144/8192共同ROI全部低于`1e-4`；连续与恢复终点状态一致。

本轮确认的只是当前仿真假设下的工程链资格。代理曝光1 ms下四通道终点仍约94.4–94.9%像素饱和；九通道未饱和但传感器端功率低。没有重建图像质量、语义生成、相机实测或OLS最终维度证据。Gaussian S12仍是仿真主参考，LP01为同相位敏感性对照；历史2048窗口G3 FAIL及Generator恢复39/45保持原状。详见[v11报告](../research/ols/ols_decoder_autograd_update_preflight_v11.md)、[机器artifact](../../artifacts/ols_decoder_autograd_update_preflight_v11.json)、[精确回传副本](../../outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/autodl_backup_v11_return)及[首次失败日志回传](../../outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/autodl_backup_v11_failure_logs)。

下一步建议另行制定短程Decoder重建对照，预先固定训练人口、相机读出、预算和图像质量判据。该建议不授权自动训练，也不选择曝光、gain、OLS维数或实验结构。

## Gaussian/OLS 物理分支历史：v10相机读出闭合与训练接口审计（2026-10-17）

Gaussian S12仍为后续OLS仿真主参考，LP01为同几何敏感性对照；模式、传输、相机参数均未实测。v10在AutoDL完成16组候选/模式/路线/冻结Decoder上下文的50图逐像素裁剪后处理。1 ms代理曝光下，2×2-C每上下文50图平均饱和率89.71–98.09%；3×3-B均匀`t=0.5`基点没有饱和，但控制图样最高饱和22.04%。相机像素输出的导数中心差分与解析值相对误差`3.43e-11`，边界单侧检查及负控通过。

v10更正了v9“九通道所有图样均不饱和”的表述。三个训练消费者也没有统一读出：旧语义OLS直接消费单位未声明的`|E|²`并把`L_optical`置零；匹配架构与Chen400消费各自W/pixel增益链，未接入v10 QE/曝光/满阱输出。故本结果支持另行制定受控Decoder训练计划，但没有迁移正式接口、训练图像或验证实物。详细数据见[v10报告](../research/ols/ols_camera_training_interface_closure_v10.md)、[机器artifact](../../artifacts/ols_camera_training_interface_closure_v10.json)和[AutoDL attempt_02](../../outputs/20261017_ols_camera_training_interface_closure_v10/runs/attempt_02)。

后续若设计训练，需先明确固定camera gain/曝光及任务输出消费者，并在真实生产反向路径核验值和梯度；本轮没有选择gain或曝光、修改loss、迁移参考或checkpoint，也未授权启动训练。历史G3失败与Generator恢复39/45不变。

## Gaussian/OLS 物理分支历史：v9 Decoder/camera接口验证（2026-10-17）

Gaussian S12继续作为OLS仿真主参考，LP01作为同几何模式敏感性对照；所有模式、光纤、relay、Decoder及相机参数仍是未测量仿真假设。v9已在AutoDL完成冻结2×2-C/3×3-B相位的16种接口组合，各50个固定命令；4096/6144/8192共同ROI的complex64窗口比较全部通过，最大对齐复场误差1.075e-5，原1e-4门槛未变。唯一判定为SUPPORTED_WITHIN_QUALIFIED_CONTEXTS。

t=0.5时，2×2-C光纤输出约42.44 μW，3×3-B约77.57 nW。显式−0.5倍4f假设将九通道相机视场功率从直连约8–10 nW提高到约33–35 nW；四通道在1 ms代理曝光下约94–98%像素饱和。相机强度Jacobian在1e-3相对阈值下为7/17个数值方向，但方向数依赖阈值；噪声SNR在满阱饱和时只代表裁剪前线性估计。故两种候选均保留作接口研究，尚未决定最终OLS维数，也没有语义图像或硬件结论。详见[v9报告](../research/ols/ols_decoder_camera_interface_v9.md)和[机器结果](../../artifacts/ols_decoder_camera_interface_v9.json)。

下一步只建议另行规划：用可追溯的相机/光纤参数范围评估接口敏感性，再决定是否进入材料响应研究。此建议不授权材料前向、Decoder或相位优化、正式生成训练或任务评价。

## Gaussian/OLS 物理分支历史：v8（2026-10-15）

本分支与Generator训练/数字身份路线分开维护。Gaussian S12继续作为OLS仿真主参考，LP01是同几何、同一冻结复场下的模式敏感性对照；模型和光纤参数仍是未测量的仿真假设，不迁移正式默认模型或旧checkpoint。

最新证据为[四通道—九通道快速数值闭合v8](../research/ols/ols_four_nine_window_closure_v8.md)：冻结v7的2×2-C与3×3-B相位，使用4096/6144/8192窗口和共同4.096 mm ROI。两个候选各50个图样的所有complex64窗口比较通过原`1e-4`门槛，矩阵预测与生产前向最大误差`7.40e-7`；全部通道幅度控制矩阵、逐通道直接控制和相位Jacobian已检查。主任务与补充共同活塞预检退出码均为0，回传收据完整。[机器artifact](../../artifacts/ols_four_nine_window_closure_v8.json)记录报告、补充预检、失败attempt、源恢复和回传信息。

在固定249.3 μW源、t=0.5下，2×2-C Gaussian耦合42.443 μW、`N_eff≈4`；3×3-B为77.569 nW、`N_eff≈9`。LP01对四通道功率趋势接近，但九通道分布明显更不均匀（`N_eff≈8.91`）。因此两候选均获得限定范围内的局部幅度控制证据：四路代表高功率低维候选，九路代表低功率高名义维数候选。它们不是相机可观测维度、语义输入能力或实验容量证明；没有选定最终OLS维数。

历史边界保持：v4原2048窗口G3 FAIL没有被改写，v7的九通道3072↔6144最坏探针失败仍保留。v8是较大研究窗口下的新资格，不能回写成旧生产网格通过。v6相位优化的完整权衡仍见[v6报告](../research/ols/ols_phase_dimension_power_v6.md)。旧Generator恢复39/45与本物理分支独立。

该Decoder/camera坐标与单位接口对照已由v9完成；其数值结论与代理相机限制见v9收尾报告。后续材料敏感性、Decoder优化、训练入口修改或生成训练仍需分别计划和授权。

## 1. 结果与状态

此前的Generator-only三种子A/B训练完成12000次科学更新，正式结果保持`VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`。保存证据分析没有新增更新。其后云端工程诊断新增36次Adam更新、历史计数3次，累计39/45；科学更新仍为0，工程状态为`INVALID_HARNESS / INCOMPLETE`。

保存证据定位状态为`EVIDENCE_REVIEW_COMPLETED / diagnostic-only`。输入为[父任务备份](../../outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9)，完整分析记录覆盖7000行DEV raw和10500行条件采样raw。当前报告另从已保存CSV重新核验配对键、汇总和原主判定值；该刷新没有重复加载全部raw像素。

| 批次种子 | Q_A | Q_B | Q_B−Q_A | Q0 | 旧规则 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026101201 | 0.3896659281 | 0.3191550831 | −0.0705108450 | 0.4377299823 | 通过 |
| 2026101202 | 0.3901176451 | 0.3195266620 | −0.0705909831 | 0.4377299823 | 通过 |
| 2026101203 | 0.3898325656 | 0.3194221251 | −0.0704104404 | 0.4377299823 | 通过 |

这些结果沿用类别2、4各100行等类权重Q_fg、既定容差和共同起点比较，维持唯一旧判定，没有增加科学Gate。

## 2. 科学认识

固定前景目标降低了焦点前景误差，却没有同步改善焦点数字身份代理。经验DEV每个种子的类别2、4 E_D正确率由A臂192/200变为B臂191/200；配对为190对均正确、2个A对/B错、1个A错/B对、7个均错。条件采样对应每种子296/300变为293/300；293对均正确、3个A对/B错、没有反向转移、4个均错。身份余量是请求类别logit减最大竞争logit，其三种子均值在DEV约4.701→4.204、条件采样约4.521→3.844。

类别表现并不一致。经验DEV中，类别1约88.0%→80.3%、类别7约85.7%→81.0%，类别8约73.7%→85.0%、类别9约55.0%→63.7%。条件采样中，类别1约86.0%→64.9%、类别7约79.8%→72.4%，类别8约72.2%→80.4%。这说明十类平均值会掩盖类别转移；这些仍是冻结E_D代理，不是人眼识读。

固定DEV目标上的28×28归一化形态分析显示：类别2自身能量距离均值为0.013984→0.011804，类别4为0.013901→0.012467；但类别2与最近竞争类的分离量略降（0.009731→0.009146），类别4略升（0.004180→0.004428）。这与原条件采样报告的TRAIN目标集合距离是不同参考，不能混用。

焦点raw CMOS积分强度约下降1.17%（DEV）和1.24%（条件采样）。前景欠填、过填代理下降，背景生成质量略升。故目前可支持的解释是“前景目标改善且伴随身份与强度权衡”，不能说可读性提高，也不能只靠形态距离选择模型。

## 3. 目标缺口

全人口混淆方向和预先分层的保存raw图册已定位；当前仍不能区分笔画形态与输出强度各自对E_D变化的贡献，也不能单独归因于冻结OLS先验。下一项缺失测量是固定输出能量匹配后的冻结E_D响应。三种训练种子共享初始化，不代表三次独立随机初始化。

人眼识读、独立TEST泛化、模型采用和物理部署均未验收。评价器正确率与几何指标不能代替这些结论。

工程方面，attempt_02已在云端部署当前入口，并完成输入核验、六轨迹短更新、零lambda控制、连续重复、资源采集和一次真实独立进程文件恢复。保存边界模型/Adam/RNG/批次/历史精确重载，但恢复后终点Generator相位超出预先冻结的连续运行包络；SIGINT/SIGTERM及下游保存测量没有执行。微批1双臂梯度通过；2、5、10未通过等价性，故不采用。微批1相对旧生产路径测得约31.24%提速，但不构成更大微批切换依据。预算为历史3+本次36=39/45，余6次。恢复来源仍未定位，不能称恢复或当前入口长训练预检完成；该句仅记录attempt_02当时的SSH状态；2026-10-14 v4已通过沙箱外AutoDL通道完成云端长验证与SCP回传，当前云端连接已验证。

## 4. 主线推进路线

### P0：保存证据定位（已完成）

- **问题：** Q_fg下降是否与数字身份、形态、强度和类别混淆同向？
- **输入：** 固定父备份中的共同A4700、六训练终点、DEV raw与条件采样库；不选择最好种子。
- **动作与证据：** 逐行重算指标并保留两类配对键；消费7000行DEV、10500行条件采样；从已保存表独立复核7500个配对身份键、327项行到汇总比较和1500条预先定义的焦点身份配对。原Q_fg规则按三个种子原值复放。
- **结论与停止：** Q_fg收益重现，但焦点身份代理未改善，形态集合靠近DEV目标；停止于描述性定位，不训练、不采样、不评分、不采用模型。

### P1：全人口混淆、形态/强度量化与固定图册（量化完成；笔画解释未闭合）

**量化审计和图册完成，笔画解释未完成。** 输入为[父任务完整备份](../../outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9)及[保存身份分析](../../outputs/20261013_m0_generator_cloud_digit_identity_saved_evidence_v1/runs/attempt_01)；结果见[Windows CPU收尾](../../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/report.md)、[AutoDL原结果回传](../../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/autodl_backup_cloud_attempt_01)和[固定样本图册](../../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/review_atlas.html)。分析消费17,500行保存记录、验证7,500个配对键和2,070行混淆汇总，按人口×种子×类别×四种E_D正确性转移分层，选出230对；16层为空并保留，不补挑样本。

三种子DEV的E_D正确数为2473/3000→2518/3000（+45），每种子均增加；条件采样为3596/4500→3526/4500（−70），每种子均减少。变化由类别驱动：条件采样类别1减少95/450、类别7减少33/450；类别8增加37/450。主要错误方向包括采样类别1→3（60对）和类别7→9（23对）；两臂持续错误里，DEV类别9→4为91对、采样类别9→4为139对。raw积分强度中位B/A约0.988。固定样本能量匹配后生产视图差距略有收缩，而单位总强度形态差异仍存在；这保留强度与结构两种解释，没有证明因果、人眼识读或生成质量改善。

图册提供固定样本的并列视图，但没有系统地将具体连接、缺口、欠填/过填笔画逐项标注；全人口混淆表也不能提供这类空间解释。故不将“关键笔画定位”列为已完成。

### P2：条件式识读方法试点（已有历史方案，当前不作为下一步）

已有[160题方案](../../outputs/20261013_m0_generator_cloud_digit_identity_saved_evidence_v1/runs/attempt_01/readability_pilot_plan.md)：60组预定A/B（3训练种子×类别2、4×5采样种子×draw 0、1），10张真实数字正控、6张黑屏控题、24张隐藏重复。质控建议为正控至少9/10、黑屏6/6、A/B重复各至少11/12一致。方案用于检验方法，不是正式效应检验。

该方案保留作历史方法材料。当前身份线的唯一建议是先对P1固定230对保存输出做单因素强度反事实：只把B raw CMOS积分强度匹配到A，再经生产clamp→pool视图和冻结E_D读取预测转移与身份余量。固定空间形态、模型/权重、样本和DEV目标；不训练、不采样、不启用TEST。此诊断尚未执行，不由本报告授权。只有其结果形成明确且值得回答的人眼问题后，才另行讨论识读试点；正式效应检验还需预设最小有意义差异、样本量和唯一主分析。

### P3：下一项训练与恢复工程（仅在提出明确科学问题后）

只有当固定输出诊断形成更明确、可反驳的训练问题后，再另行明确起点、人口角色、更新范围、固定预算、唯一主量和停止规则。启动任何依赖恢复的长更新前，必须先定位attempt_02的终点差异，并在新的明确工程预算内完成独立进程恢复及必要的信号保存、隔离、测量回读。attempt_02保留历史3次加新增36次，共39/45；余6次不够完成失败恢复复测和两种信号保存（至少还需9次）。不能通过放宽包络、自动延长预算或将微批候选标为等价来收尾。

新的训练结果只能用于其预先指定的问题，不能回头改写旧Q_fg结论或按结果追加步数。恢复或生产保存测量无效时按工程问题修复，不作科学负结果。

### P4：独立TEST、正式识读与采用评审

开发阶段先锁定候选、选择规则、阈值和分析流程，再一次性消费预留TEST；不把TEST用于调权。正式识读、模型采用和物理部署是不同里程碑，分别需要人眼效应、模型决策和实测物理参数证据。

## 5. 决策影响与授权边界

当前不建议无新问题地继续增加Generator步数。身份线唯一保留的下一项建议是上文固定230对输出的强度反事实；识读方法试点和新训练均不列为当前后继动作。OLS/G3 v5的限定系统复核和独立获准的v6相位诊断均已完成，v6结果及下一项物理建议见本路线图上方；这些物理证据不改变Generator分支状态。Generator云端恢复仍须在任何依赖该断点的长更新前解决。

本路线图给出的身份线建议不授权训练、调权、采样、人工评分、TEST启用、模型采用或物理部署。OLS v6的独立授权已经用于完成该诊断，不构成下一项物理测量或生成训练的授权，也不改变本路线图对身份/Generator工作的边界。恢复诊断停止于39/45次预算，保留失败阶段、传输收据和CPU回归证据；云端恢复、信号保存和生产保存测量仍未闭环。

## 完整光路语义生成：终点 Encoder 投影配准（2026-11-02，已完成）

### 结果与状态

在冻结 E1G1、mapper、先验、模态尺度和相机增益后，比较 P0 原响应投影与终点 Encoder 响应右逆 P1。已消费人口上的最终语义结果为 SEMANTIC_BENEFIT_NOT_SUPPORTED：E_D -0.30 pp（95% 区间 [-0.80, +0.20] pp），E_C +0.10 pp，最小逐类多样性保留 100.07%。未满足 E_D 区间下界大于零的预设规则。

### 科学认识

P1 将 G 从 0.13052 降至 0.12291，配对 Delta G=-0.00761（95% 区间 [-0.00803, -0.00718]）；公共相位对齐目标误差均值由 0.08357 降至 0.01269。这支持投影配准改善输入/OLS 对齐，但训练分布距离仍大于零，且完整光路语义收益未建立。P1 总模态功率均值增加约 1.09%；预裁剪越界比例均值 4.27%、最大 31.59%。

### 目标缺口

本结论限于已消费的 2,000 行诊断人口、冻结模型和当前数值光路。独立人口上的语义复核、人眼可读性、陈一同论文设计等价性、器件实测及物理部署仍未完成。

### 工程完成与验证

AutoDL 方案内接受的测量为 2,023 条 2048 前向、20 条 4096 参考及 200 次输入标量导数；全量回传 38 个文件且大小核对通过。Windows 独立消费者以保存命令和 raw CMOS 重算命令、投影、评价器、距离、2,000 次 bootstrap 和最终 Gate；P0/P1 命令最大重放差 7.45e-9 / 1.49e-8，右逆误差 1.47e-14，跨网格复系数/raw CMOS 误差 2.73e-5 / 1.98e-5。聚焦测试 10 项、Ruff 和编译检查通过。恢复过程中的无效尝试保存在[执行历史](../../outputs/20261102_lightgen_terminal_projection_alignment_v1/runs/windows_review_01/execution_history.json)。

### 决策影响与下一步

保留 P0、P1 和所有测量证据。该结果确认配准机制变化，但未支持数值语义候选胜出；不追加训练、调参、先验重拟合、采样或硬件操作。若后续继续，应另行设计独立人口上的语义确认并重新预注册唯一 Gate；此建议不构成本轮授权。

## 完整光路语义生成：D/P0/P1路径影响闭环（2026-11-02，已完成）

### 问题与路线位置

复用近期 E1G1 权重及已保存 P0/P1 完整路径输出，比较它们相对同目标的直接 OLS 后端参照是否保留数值语义表现。路线位置为：近期训练成果复用→模块归因→训练/生成 OLS 分布定位→输入投影配准→本次直接 OLS 边界比较与训练入口固化。

### 固定设计与唯一结论

人口为已消费诊断人口 2,000 条（每类 200），同一组保存先验目标及 proposal；D 是直接目标复系数进入同一 OLS/Generator/CMOS 后端，P0/P1 为真实保存的完整 SLM 路径输出。对两条路径分别使用 2,000 次、seed `202611021170` 的等类配对 bootstrap；门槛为 E_D/E_C 区间下界均不低于 −2 pp、每类显示图像距离保留至少 80%。结果为 `BOTH_FULL_PATHS_WITHIN_PREDECLARED_MARGIN`：

| 条件 | E_D 差（95% CI） | E_C 差（95% CI） | 最低逐类多样性保留 | 最终判断 |
|---|---:|---:|---:|---|
| P0−D | +0.70 pp [+0.20, +1.25] | 0.00 pp [−0.60, +0.60] | 97.25% | 在预设界内 |
| P1−D | +0.40 pp [+0.10, +0.70] | +0.10 pp [−0.15, +0.40] | 97.73% | 在预设界内 |

这说明本人口中 P0/P1 没有超过预设语义非劣界的下降；并不表示路径没有物理影响或输出相同。raw CMOS RMS 相对 D 分别变化 2.22279 和 1.70670 W/m²；P0 的 E_C 类 8 下降 4 pp。D 只是边界参照，不能替代完整推理路径。

### Chen 等论文方法对应与项目差异

参照 Chen et al., *Science* 390, 1259–1265 (2025), DOI [10.1126/science.adv7434](https://doi.org/10.1126/science.adv7434)：后续训练沿用“真实图像经光学 Encoder 学习 OLS、同一完整前向的输出参与训练”的原则。当前联合更新入口核对 TRAIN 来源行、振幅 SLM、两层 Encoder、当前 Gaussian OLS 系数、两层 Generator、功率/CMOS 单位，并确认 BOGT+固定 SSIM 来自同一完整前向；后续研究仍需保持完整 SLM→Encoder→OLS→Generator→CMOS 前向。

项目与论文的差异保持明示：当前 2+2、100×100 架构与物理参数为项目规模；固定 SSIM 是 BOGT 之外的项目附加项；mapper+投影是将 OLS 目标制备为合法 SLM 振幅的输入适配器，不等于论文原生 OLS 采样。该任务不修改损失、先验或训练权重，因此不称为精确论文复现。

### 工程与证据

AutoDL 保存 D 的 2,000 条 2048 输出以及预检/控制证据；D 的 20 条 4096 参考 raw CMOS 相对误差为 1.4301e−5，P1 实际 OLS 系数重注入误差为 0。终点模型上的 10 张无更新 TRAIN 完整路径检查确认 Encoder、Generator 的 CMOS 梯度均非零，分别为 0.005646、0.002462；模型、Adam、mapper、先验、投影和评价器均冻结。Windows 从 20 个 raw CMOS 分片重建显示与评价器，NumPy/生产显示差为 0，独立 bootstrap 与最终判断完全匹配；影响路径测试 88 项通过，Ruff 与 `py_compile` 通过。

详情见[独立收尾报告](../../outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/windows_review_01/closeout_report.md)、[Windows 复核](../../outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/windows_review_01/independent_review.json)、[完整云端备份](../../outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/autodl_backup_attempt_01)和[生产训练护栏](../../scripts/research/lightgen/execute_lightgen_small_joint_update_readability_pilot_v1.py)。

### 结论边界与下一步

独立人口语义确认、人眼可读性、总体 G1 分布、实物器件映射和部署尚未完成；本结果不支持模型采用或硬件验收。下一步建议是另行预注册一项独立人口确认；任何后继训练仍须经过完整真实光路、固定预算和语义判据。本轮无科研更新，不自动开始后继工作。

## 完整光路工程闭环修复（2026-11-03，attempt_02）

### 结果与状态

工程验收与 Windows 独立复核通过。attempt_01 `INVALID_HARNESS` 及 attempt_02 首次恢复 worker 失败均保留。成功工程更新调用为 5，实际 Adam 实例为 3（父进程 1、失败恢复 worker 1、成功恢复进程 1）；失败 worker 在更新前退出。旧报告声称优化器实例为零已勘误；历史总数为未知。本轮科研更新、先验重拟合/采样和官方 TEST 读取均为 0，源 checkpoint 未改变。

### 科学认识

修复后的保存证据消费者独立复算后，D/P0/P1 仍为 `BOTH_FULL_PATHS_WITHIN_PREDECLARED_MARGIN`，与先前结论一致且仅适用于已消费的 2,000 条生成诊断人口；没有新增科学候选。D、P1 已保存的 2048/4096 控制通过 1e-4；P0 缺配对 4096 控制，不能声称 P0 跨网格验证通过。该限制不替代语义人口判据，但须保留为数值复核范围限制。

### 工程完成与验证

CMOS 重构/SSIM 梯度消费者、断梯度负控、三步真实更新、第四步独立恢复、更新见证与源状态回滚均通过。P1 20 条实际 OLS 重注入的 raw CMOS 与 W/pixel 最大绝对差均为 0。Windows 保存数组复算、NumPy bootstrap/多样性与最终决策复算一致；只读复核没有创建优化器或更新。定向测试 36 项通过，Ruff、编译及差异空白检查通过。有限差分复用既有生产记录，正负扰动向量未归档。

### 目标缺口与路线

下一项建议为独立人口零更新验证：同一冻结终点和适配器，新人口 2,000 条、每类 200 条，D/P0/P1 同行配对；两评价器 95% 配对区间下界均须 ≥−2 pp，且每类多样性保留 ≥80%。人口与 bootstrap 种子需在单独授权和执行预检时固定。草案见[独立人口验证计划](../../outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/windows_review_02/next_stage_independent_population_plan.md)，当前不授权采样或训练。后续训练继续强制真实图像→SLM→Encoder→OLS→Generator→CMOS 完整前向；单组冻结训练入口尚未支持。实物阶段另需完成 SLM/器件标定与坐标注册、OLS/光纤复传输、相机读出及端到端采集，本轮没有硬件证据。

详情见[收尾报告](../../outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/windows_review_02/closeout_report.md)、[独立复核](../../outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/windows_review_02/repair_independent_review.json)及[AutoDL 回传收据](../../outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/autodl_backup_attempt_02/transfer_receipt.json)。
