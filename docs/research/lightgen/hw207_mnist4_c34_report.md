# HW207 MNIST4 C3.4 收尾报告

日期：2026-11-04<br>
会话来源：`01a18b9f-206d-7c61-96a3-0da3a49efb8c`<br>
执行边界：本报告整合 C3.4 历史数值运行与本轮 `attempt_05` 加工感知高度训练；新增 attempt05 未读取官方 MNIST test、未打开 holdout、未控制 DMD/CMOS、未打印器件。新增结果仅是复用 C3.4 validation 的参数化诊断。

## 结果与状态

| 工作包 | 收尾状态 | 证据边界 |
|---|---|---|
| `C3.4_ARTIFACT_AUDIT_PASS` | PASS | C3.3 文件审计覆盖 201 个文件、57,711,932 字节；19/19 训练相位可用于推理，24/24 STL 文件有回传记录。历史优化器状态缺失，19 个训练不能独立重放。 |
| `C3.4_OPTICAL_MODEL_NUMERICAL_PASS` | PASS（数值候选） | 生产传播和独立数值检查完成；模型为 `P(80 mm)L(30 mm)P(100 mm)`、D=2 mm 孔径假设，不是 HW207 标定。输入映射与 C3.3 完全一致。 |
| `C3.4_FROZEN_PHASE_INFERENCE_PASS` | PASS（测量完成） | 固定 C3.3 相位在历史 2,000 行上，模型 A 为 95.80%，模型 B（D=2 mm）为 48.30%；配对差异触发几何适配。这里的旧 holdout 已用于 C3.3。 |
| `C3.4_GEOMETRY_ADAPTED_TRAINING_PASS` | PARTIAL | 三种子训练和固定 ROI 分类完成；最高验证准确率 96.10%，C3.4 holdout 为 95.75%。但 holdout 全传感器效率仅 3.245%，低于配置的 7.708% 吞吐下限，故不能作为完整训练通过。 |
| `C3.4_FABRICATION_AWARE_PASS` | DEGRADED | 156 个同算子情景完成。高度 64 级量化 + 20 nm 噪声单独影响小；加入假设 PSF σ=0.8 px 和最小特征半径 1 px 后，几何适配相位准确率降至约 19–20%。不是打印机实测。 |
| `C3.4_PROCESS_AWARE_PREFLIGHT_PASS` | PASS | 生产前向一致性、独立导数检查、3 次真实更新、随机标签负对照和检查点恢复对照均通过。首次 `attempt_04` 因检查点恢复测试共享优化器状态而判为 `INVALID_HARNESS`；修复后在 `attempt_05` 重跑。 |
| `C3.4_PROCESS_AWARE_HEIGHT_TRAINING` | `ASSUMPTION_ONLY_DIAGNOSTIC_COMPLETE` | 三个种子在假设加工算子下完成；入选 seed 1/epoch 5，validation 平均准确率 95.265%。同一高度图在 Δn=0.6 敏感性情景为 90.42%。validation 已用于 P3 选模，不是独立泛化结果。 |
| `C3.4_PROCESS_AWARE_STL_DIAGNOSTIC` | PASS（参数化几何/往返） | 两个 STL 各 41,612 面；闭合二流形、法向一致、无边界/非流形边或退化面，高度恢复误差低于 `5.0×10⁻⁶ µm`。通用三角面相交算法未运行，文件不是加工放行件。 |
| `C3.4_STL_GEOMETRY_PASS` | PASS（参数化诊断 STL） | 四个 STL 各 41,612 个三角面；独立解析显示封闭二流形、边配对和法向一致，边界/非流形边与退化面为 0。新增拓扑专用高度场结构证明通过；通用三角面对相交算法仍未运行。 |
| `C3.4_PRINTED_STRUCTURE_SIMULATION_PASS` | PASS（参数化数值重仿真；性能 DEGRADED） | 从 STL 恢复高度/相位并重算；理想往返为 93.75%，名义加工往返为 18.75%（Δn=0.5）和 20.31%（Δn=0.6）。对应的是 64 行验证子集。 |
| `C3.4_FABRICATION_READY` | BLOCKED | 材料折射率差、打印机横向响应和实际结构均未知；参数化 STL 不可作为最终加工文件。 |
| `HW207_CURRENT_MAPPING` | BLOCKED | DMD、真实器件复透射、Relay 共轭和 CMOS 坐标仍没有实物闭环证据。 |

**总判定仍为 `C3.4_PARTIAL_PASS`。** 新增分支在假设加工模型中恢复了四区分类，但没有补足真实材料、打印机与 HW207 光路证据；全传感器效率也显著低于未加工的 P3 基线。

加工感知分支在同一 2,000 行平衡 validation 上，对 5 个噪声种子与正负 20 nm 边缘高度偏差组成的 10 个情景进行成对评估：

| Δn 假设 | P3 几何相位：Accuracy / 全传感器效率 | 加工感知高度：Accuracy / Macro-F1 / 全传感器效率 | Accuracy 配对提升 |
|---|---:|---:|---:|
| 0.5 | 21.635% / 39.373% | 95.265% / 95.254% / 11.008% | +73.630 pp |
| 0.6 | 21.835% / 38.896% | 90.420% / 90.369% / 10.500% | +68.585 pp |

上表是在相同 validation 和相同硬加工情景下比较；P3/加工感知分别平均 10 个实现。准确率跨实现标准差分别为 0.133/0.058 pp（Δn=0.5）和 0.100/0.333 pp（Δn=0.6）。这不是独立测试集结果。

几何适配训练 holdout 的混淆矩阵（行是真实类别，列为 ROI 预测，最后一列为弃判）为：

| 真值 \ 预测 | ROI-0 | ROI-1 | ROI-2 | ROI-3 | 弃判 |
|---|---:|---:|---:|---:|---:|
| 0 | 489 | 1 | 5 | 5 | 0 |
| 1 | 0 | 487 | 10 | 3 | 0 |
| 2 | 12 | 5 | 477 | 6 | 0 |
| 3 | 5 | 6 | 27 | 462 | 0 |

该矩阵给出 1,915/2,000 正确，Accuracy=95.75%，Macro-F1=95.7519%，覆盖率 100%。汇总指标可由混淆矩阵复算；回传材料没有逐样本 holdout 分数、预测或强度，因此无法逐行重算该集合。

补充的可视化均从已保存产物生成，没有重新传播或训练：[P2 类均值 CMOS 图样](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p2_class_average_cmos_crops.png)、[P2 固定示例](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p2_fixed_cmos_examples.png)、[P2 固定 ROI 指标](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p2_fixed_roi_metric_comparison.png)、[P3 入选相位](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p3_selected_wrapped_phase.png)、[P3 三种子训练轨迹](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p3_validation_training_trajectories.png)和[holdout 混淆矩阵](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/p3_holdout_confusion_matrix.png)。P2 图像为形状展示而逐图归一到单位 crop 能量，不用于比较模型间绝对能量。

## 科学认识

**固定相位不能直接迁移到估算几何。** C3.3 相位在历史理想 4f 模型上为 95.80%，换到估算 `P(80)L(30)P(100)` 模型后为 48.30%。两者配对不一致方向为 A 正确/B 错误 976 行、A 错误/B 正确 26 行；单侧精确检验 `p=4.51×10⁻²⁵¹`，按预设触发了 C3.4 几何适配训练。该比较的 2,000 行来自已消费的 C3.3 holdout。

**几何适配恢复了分类，但消耗了大量光能。** 三个种子的验证准确率为 96.05%、96.05%、96.10%，按验证准确率选择 seed 3、epoch 15；固定相位 holdout Accuracy=95.75%。然而 holdout 平均全传感器效率为输入能量的 3.245%，而配置下限为 7.708%（训练零相位参考的 10%）。验证子集上的理想相位效率约 3.324%，也低于该下限。

这不是一个可以忽略的副指标。生产损失把 ROI 分数除以裁切能量形成分类 logits，同时只用权重 0.1 的平方 hinge 项惩罚低吞吐；模型可以在降低绝对能量后仍维持相近的归一化类别分数。验证集只按分类准确率选 seed/epoch，没有硬性吞吐准入。因此报告高准确率时必须同时报告低能量，`C3.4_GEOMETRY_ADAPTED_TRAINING_PASS` 只能记为部分通过。实现见[训练损失与选模代码](../../../scripts/research/lightgen/execute_hw207_mnist4_c34_v1.py#L747)。

**假设加工扫描把瓶颈定位到横向细节。** 几何适配相位在 64 行验证集的理想加工映射为 93.75%；仅 64 级高度量化和 20 nm 噪声时，五次扰动平均为 94.69%（Δn=0.5）和 95.00%（Δn=0.6）。但联合假设 PSF σ=0.8 px、最小特征半径 1 px、64 级量化和 20 nm 噪声时，平均准确率分别为 19.38%（SD 0.86 个百分点）和 20.31%（SD 0 个百分点）。这支持“横向带宽/最小特征可能是主瓶颈”的假设，不证明真实打印机性能。P4/P6 的 64 行都来自参与 P3 选模的验证集。

**加工感知再优化改变了假设模型内的分类结果，但没有验证真实加工。** attempt05 把唯一可训练量设为 100×100 物理高度图，前向训练使用 PSF σ=0.8 px、最小特征半径 1 px、64 级量化、20 nm 高度噪声及 ±20 nm 边缘偏差；训练只用 18,279 行，validation 2,000 行用于早停、种子选择和结果比较。入选 seed 1/epoch 5 的 Δn=0.5 平均 Accuracy 为 95.265%，三个种子的入选分数为 95.265%、94.950%、94.930%。这说明约 20% 的旧 P3 结果并非这些数值假设下无法优化的分类上限；它不能证明真实打印机的 PSF/最小特征等于假设，也不能说明实物器件会达到该准确率。

加工感知高度图在 Δn=0.6 下保持同一物理高度图，平均 Accuracy 降至 90.420%，说明折射率差假设仍影响相位映射。分类改善伴随效率下降：Δn=0.5/0.6 的全传感器效率约为输入能量的 11.01%/10.50%，而匹配情景下 P3 几何相位为 39.37%/38.90%。因此，这条分支解决的是假设加工算子下的固定 ROI 分类退化，不是能量效率、材料或制造不确定性。

STL 往返也区分了“设计高度图”和“假设加工后高度图”：直接对设计 STL 的理想无加工读数为 48.30%（Δn=0.5）和 36.90%（Δn=0.6）；名义 PSF/最小特征/量化转换后的 STL 读数为 95.35% 和 90.60%。高度往返误差小只说明软件高度场编码一致。加工后高度 STL 表示假设变换结果，不是对打印结构的测量。

**STL 往返验证了数值接口，不验证打印。** Δn=0.5/0.6 下的高度矩阵和 STL 均按参数化模型生成；理想相位经高度映射、STL 编解码后保持 93.75% 验证准确率。加入名义横向加工算子后再走 STL 往返，准确率为 18.75%/20.31%；加工相位与 STL 恢复相位的预测一致率为 100%。高一致性说明两个软件表示接近，不代表打印结构接近模型。

## 目标缺口

- C3.4 的光学传播是标量近似：DMD 使用 C3.3/C2.1 二值输入代理；D=2 mm 孔径是假设；距离 `P(80)L(30)P(100)` 仍源于估算几何。不能称作 HW207 实测传播模型。
- C3.4 的 18,279/2,000/2,000 train/validation/holdout 在内部互斥，三个训练检查点保存的行索引与固定分层表完全一致。但 P3 使用的 C0 train 样本池已在 C3.3 各角色中使用；旧 P2/C3.3 holdout 与 C3.4 train/validation/holdout 又分别重叠 **1,647/181/172** 行。因此 C3.4 holdout 没有参与 C3.4 自身拟合或选模，却不是项目级独立测试；官方 test 也没有读取。
- CMOS 坐标、相机身份/读出噪声、输出 Relay 实际倍率和像差、DMD ON/OFF 收光、LightGen 实物复透射均未测。`HW207_CURRENT_MAPPING` 仍为 `BLOCKED`。
- 材料和打印机身份、532 nm 的 Δn、横向 PSF、最小特征、边缘误差及收缩均为 `UNKNOWN` 或情景假设。四个 STL 的 sidecar 声明单位为 µm，STL 本体没有固有单位。新审计独立解析 STL 并验证 102×102 网格的 top/bottom 单元三角剖分、单值高度和边界侧面四边形；这对本次高度场网格给出结构性无自交证明，但通用三角面对相交算法仍为 `NOT_RUN`。不能交付为 `FABRICATION_READY`。
- C3.4 holdout 仅回传聚合混淆矩阵/指标，缺逐样本结果文件，降低了复算粒度。P4/P6 使用选模验证集，不能用于独立泛化声明。
- 分类只证明在指定数字数据、假设传播和固定 ROI 下可产生类别读数；不等于图像生成质量、语义生成能力、人眼可读性或实物光路通过。
- 加工感知分支的 95.265%/90.420% 使用了已参与 P3 选模的同一 validation；三个种子用于初始化敏感性，不构成三个独立数据集。官方 MNIST test 和项目级独立人口均未读取。
- 材料 Δn、实际打印 PSF/最小特征/边缘误差、固定 LightGen 复透射、DMD 收光、Relay/CMOS 映射仍未知。两个 STL 的通用三角面对相交检查为 `NOT_RUN`，它们继续标记为参数化诊断文件。

## 工程完成与验证

- P0 审计了 C3.3 输出：201 个文件、57.7 MB；唯一验证集选中相位来自 `train_dz52000_zero_s3/phase.pt`（seed 3、epoch 13、validation Accuracy 96.05%、loss 0.54618），P2 在历史 holdout 的 Model A 复现 Accuracy 为 95.80%。19/19 相位训练产物可用于推理，但缺优化器状态，无法独立重放；24/24 STL 文件均被清点。独立 STL 解析显示这些 C3.3 网格的平面边界为 198×198 µm，即 100 个采样中心的跨度；它们没有达到 200×200 µm 单元边界目标。C3.4 新参数化网格将采样中心放在 1–199 µm，并将物理边界扩展到 0–200 µm，未改写历史 STL。P1 检查模型输入映射、传播、相位与精度；结果状态是 `NUMERICAL_CANDIDATE_MODEL_NOT_HW207_CALIBRATION`。
- P3 三个种子正式运行在 AutoDL RTX 4080 上退出码为 0。seed 3/epoch 15 入选；相位 `phase_rad.npy` 与 `selected_phase.pt` 完全相同。三个 checkpoint 的 train/validation/holdout 行索引均与固定 split 完全一致，三组彼此无交集。官方 MNIST test 未加载。
- P4 156 组、每组 64 行的分数均有限。独立读取原始 MNIST 标签后，重新检查了全部 156 组：存储预测均等于四 ROI 分数 `argmax`，Accuracy 和 Macro-F1 与报告值一致。样本为每类 16 行的 C3.4 validation 子集。
- P5 独立解析四个 STL：每个 41,612 面、闭合二流形、法向一致、无边界/非流形边和退化面；高度往返误差最大约 `5.0×10⁻⁶ µm`。新增结构审计确认每个顶/底网格单元由两个不重叠投影三角形划分，侧面按边界段闭合，四个网格的拓扑专用无自交证明均通过；通用三角相交程序没有运行。相位图/高度矩阵与 STL 只对应 Δn 假设。
- P6 保存六条汇总记录和四组可逐行核对的读出数组；四组的 argmax/准确率与原始 MNIST 标签复算一致，另两条加工相位与 STL 恢复相位的预测一致率均为 1.0。
- 工程修复：修正 P3 phase 文件位于 `formal/` 子目录的加载路径；兼容 C3.3 checkpoint 的 `[1,1,100,100]` 相位形状；修正 P4 图表字段 `delta_n_assumption`；支持系统盘外部 STL sidecar 的审计路径。
- C3.4 定向测试远端状态为退出码 0、20 项通过（31.58 s）；P4 preflight 与 P4–P6 运行均退出码 0。P3/P4–P6 与 P0–P2 两次回传收据分别为 65/65、13/13 文件 `COMPLETE`。收尾补图所需的四个 P2 数组另行回传，4/4 文件尺寸核对通过（[补充回传收据](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/transfer_receipt.json)）。
- 云端缓存把 22,279 张固定输入场（complex64，1,782,320,000 字节，约 1.66 GiB）放在系统盘，缓存不含标签或相位。训练逻辑 batch=64、microbatch=64，选定评估 batch=32。三步预检吞吐为 177.15 张/秒，直接 microbatch=2 基线下三次更新 13.55 秒；该三步预检的速度比为 12.50×，不能解释成整次正式训练提速 12.50×。
- P3 运行中 71 个 telemetry 点的 GPU 利用率均值为 82.10%、峰值 100%，显存约 18,229/32,760 MiB（约 55.6%）；主机最少可用内存约 448.5 GiB。历史采集没有 CPU 字段，故无法回推出训练期间 CPU 利用率。2026-11-04 20:42:34 的 8 秒空闲采样显示容器 CPU 配额为 16 核、期间使用 0.633 CPU 秒，即配额的 0.495%；当时无训练进程、GPU 利用率为 0%。这是空闲状态，不能替代训练期测量。
- 2026-11-04 20:09 的云端磁盘检查显示系统盘空闲 14.3 GiB，`/root/autodl-tmp` 数据盘只剩 0.61 GiB；1.66 GiB 输入场缓存已置于系统盘。后续若有获准运行，应继续把缓存、临时文件和恢复检查点放系统盘，并先处理数据盘空间约束；当前没有活动训练任务。既有训练期间 GPU 已较繁忙，CPU 缺少训练期遥测且数据已缓存，没有证据表明增加 CPU worker 或盲目占满资源会提速；逻辑 batch 大于 64 会改变训练条件。
- attempt05 在 AutoDL RTX 4080 完成三种子训练及加工情景/STL收尾，命令退出码 0；总任务墙钟约 88 分钟。174 个 30 秒采样点中，GPU 利用率均值 91.40%（中位数 97%、峰值 100%），显存稳定在 18,235/32,760 MiB；CPU 按容器 16 核配额修正后均值 38.45%、中位数 26.00%、P95 99.78%。因此 CPU 在部分时段会吃满配额，但不是持续瓶颈；GPU 忙碌度较高。仅凭显存未满不能推出增加 batch 会加速，逻辑 batch=64 保持不变。
- attempt05 系统盘空闲最低 14.32 GiB，数据盘空闲最低 0.609 GiB，主机可用内存最低约 407.96 GiB。原遥测按主机 128 核归一，因未读取 cgroup v1 的 16 核配额而低估进程 CPU 百分比；原始 CSV 保留，修正值另存为 [校正遥测 CSV](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/resource_telemetry_corrected.csv) 与[校正摘要](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/resource_summary_corrected.json)。本地采样器已补 cgroup v1 配额读取，并有 2 个定向测试通过；attempt05 运行期仍用旧采集器，故其原始值按 8 倍离线校正。
- 对保存的 `validation_process_comparison.npz` 独立复算 40 个“模型 × Δn × 加工实现”读出：每项均有 2,000 行、标签为 500/类；存储预测与分数 argmax、Accuracy、Macro-F1 和混淆矩阵全部一致，无不匹配。完整回传收据为 27/27 文件、7,792,568 字节： [attempt05 回传目录](../../../outputs/20261104_c34/runs/autodl_backup_processaware05)。其中也包含预检、三种子检查点、对照数组和两个 STL 审计。

机器可读复核记录：[C3.4 独立收尾审计](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/windows_review_closeout_01a18b9f/c34_independent_closeout_review.json)、[STL 高度场结构审计](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/c34_stl_heightfield_structure_audit.json)、[图表来源/边界记录](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p2_visuals_01a18b9f/figures/figure_sources.json)。**时间范围说明：** `c34_independent_closeout_review.json` 是 attempt05 加工感知分支完成前的 P0–P6 独立复核快照；其中“未进行加工感知训练”只描述该快照的范围。attempt05 的补充证据见[正式训练汇总](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/formal/training_summary.json)、[STL 往返审计](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/formal/stl_roundtrip_validation.json)、[资源校正摘要](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/resource_summary_corrected.json)和[回传收据](../../../outputs/20261104_c34/runs/autodl_backup_processaware05/transfer_receipt.json)。本报告与状态页汇总基线审计和 attempt05，原独立审计 JSON 保持原样。光学细节见独立的 [C3.4 P1 光学模型报告](hw207_mnist4_c34_optical_model_report.md)。主要输入：[C3.3 文件审计](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p0_p2_closeout_01a18b9f/p0_p2/c33_artifact_audit.json)、[候选光学模型审计](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p0_p2_closeout_01a18b9f/p0_p2/optical_model_audit.json)、[冻结相位配对结果](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p0_p2_closeout_01a18b9f/p2_frozen_inference/frozen_phase_comparison.json)、[P3 训练汇总](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_closeout_01a18b9f/p3_formal/training_summary.json)、[P4 加工敏感性扫描](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_closeout_01a18b9f/p4_p6/p4_fabrication_sweep/p4_fabrication_sweep.json)、[P5 STL 审计](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_closeout_01a18b9f/p4_p6/p5_parameterized_stl/p5_height_stl_report.json)、[P6 STL 重仿真](../../../outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_closeout_01a18b9f/p4_p6/p6_stl_recovered_simulation/p6_printed_structure_resimulation.json)。

## 决策影响与下一步

本轮新增支持一个更窄的数值结论：在给定的估算几何、物理高度映射与假设加工算子下，单层高度图可重新优化，使四区 validation 分类显著优于未经加工感知训练的 P3 相位。与此同时，效率下降，Δn敏感性仍可见，数据重复使用且真实材料/打印机未知。总体状态仍为 `C3.4_PARTIAL_PASS`；这不支持“当前 HW207 已验证”、不支持打印放行或图像生成能力。

建议先测量材料 Δn 与打印机横向响应，再决定是否把实测加工算子纳入下一轮固定设计；泛化评价需预先冻结此前未参与 C3.3/P2/C3.4 的评估人口。硬件映射还需 DMD 收光、Relay/CMOS 坐标与倍率、器件复透射的实测。**这是建议，不授权新的训练、打印或硬件实验。**
