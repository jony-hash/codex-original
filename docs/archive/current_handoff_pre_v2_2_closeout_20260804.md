# 当前主线交接：v6.4 近共轭 relay 诊断闭环（2026-07-30）

## Evaluator v2.1 已资格化：2+2 transport 的前置仪器门通过

## v1 design/admission harness invalidation 与 v1.1 修复（2026-07-31）

运行 `20260731_005811_685228` 的 design 误把 scale 数值范围当成双符号准入；类别 1、5 的
16 个方向在所有二分探针中均未满足 smooth 语义，故该 artifact 不具备下游授权资格。随后
`20260731_010954_585046` 的 admission 保留为诊断，不能作为模型失败；历史 outputs 不变。

v1.1 将 primary 作为唯一语义硬门，smooth 只作独立表示/rank/分布验证；scale 必须先过
`0.05` floor、固定二分并在最终 scale 独立复核。tangent 与双-seed smoke 在 v1.1 静态验证
通过前不得启动。

### 2026-07-31 工程执行政策：先实现全链，再运行单一阶段序列

从本条主线起，不再以“实现一个 stage 就立即启动 CUDA”的方式推进。先完成一个共享的
evaluation-qualified runtime（双 evaluator bundle、单次 raw panel snapshot、三表示指标、
anti-template、checkpoint/evidence 绑定），然后一次性完成 design、admission、tangent 和双-seed
smoke 的实现、定向测试、targeted baseline、完整 pytest 与全仓 Ruff。只有代码冻结后才按
`design → admission → tangent → smoke` 运行；每个有效失败仍立即停止。

完整性门（hash/context、系统 provenance、state、CUDA、zero-code、saturation、最终两-seed
验收）保持硬门。candidate/operational curvature 保持诊断；逐面板小样本 correlation、单面板
effective rank 和“3/4 tangent 面板改善”改为诊断。anti-template 以四面板合并后的每类统计为
硬门；tangent 的硬授权改为存在安全、aggregate composite 改善且无灾难性面板下降的方向。
这些调整不降低最终任务质量，而是避免小样本和局部一阶统计造成假阴性。

当前实现已完成 shared runtime、design、admission、tangent 和预注册双-seed smoke 的 CLI
与 fail-closed training stub；尚未启动新的 CUDA stage。定向 v2 验证、两组 v2 回归、
student/teacher 回归、全仓 Ruff 与 `git diff --check` 已通过。该执行环境对单条超过约28秒
的 pytest 命令不保留完成状态，故 application/diagnostics 的完整单文件回归和聚合 `pytest -q`
仍待在没有该会话上限的终端确认；在此之前不得启动 design。

Evaluator-qualified 2+2 曲率基线已完成：
`outputs/20260731_v2/runs/20260731_001324_194392_diagnostic_v2_transport_evaluation_qualified_v1/`。
该 evidence 为 `diagnostic`，仅授权新的 evaluation-qualified design。它复核了 4 次
near-conjugate relay、identity OLS、固定 `tau=0.80`、state 不变、无 optimizer/checkpoint
与 CUDA 显存门。candidate/operational 的 command/raw 最大曲率分别仅约
`0.008/0.015`，但 primary evaluator feature 的最大曲率为 `0.319/0.252`；smooth feature
仅为 `0.072/0.058`。真实 PCA 和同类 chord 在两个 evaluator feature 中都显示更高曲率，
因此该现象是表示的局部几何，不是当前 2+2 光路的可用性失败。曲率继续只作诊断，不再作为
design 一票否决门。

历史 evaluator-v2 evidence `20260730_164205_687065` 仍为有效失败，且其 smooth
checkpoint 不可复用。v2.1 将该失败作为内容绑定的校准输入，保持 primary evaluator、数据
split、固定曝光和阈值不变；smooth evaluator 从零开始以 seed `2234`、10 epochs、batch 256、
Adam `1e-3` 训练。校准 evidence 位于 `20260730_171048_272046`，资格化 evidence 位于
`20260730_171126_556355`，两者均为 `passed`。

资格化的 primary/smooth validation accuracy 为 `0.9876/0.9880`，macro recall 为
`0.9876/0.9880`，最差类别 recall 为 `0.954/0.976`，预测一致率为 `0.986`。强度 ±2%
和 x/y 单像素平移均通过硬门；morphology 的 dilation/erosion 指标分别被记录但不参与资格化，
因为它们会改变细笔画的任务语义。该通过只授权新的 evaluator-qualified 曲率基线和独立 2+2
design；不复活任何 v7.1/v7.2 evidence，也不授权训练。

## 已关闭：v6.4 phase-only；v7 准入失败

重放 evidence `20260730_014137_778801` 已完整验证旧 smoke 的 artifact、context、400-step
schedule、CUDA前向、generator冻结、phase更新和relay crop。它将 v6.4 分类为
`valid_failed_phase_only_distribution`：类别7 recall为 `0.8824`，最终最低 feature ratio为
`0.3982/0.5833/0.6479`，relative-diversity与delta的最后50步趋势也失败。唯一的
phase-distribution tangent `20260730_014836_346059` 在安全限制内完成，但encoder、decoder和joint
均未让三面板最差 ratio同时提升0.01；正式关闭冻结PCA generator、仅训练四层phase的v6.4路线。

随后 v7 `conditional_nonlinear_expanded_manifold_v7` 保持相同near-conjugate relay、`tau=0.80`、
identity OLS和零phase，只把SLM前电子端扩展为32个低通PCA模式与无bias类条件MLP。容量证据
`20260730_030257_695213` 通过；CUDA admission `20260730_030608_413918` 有效失败：fraction、
Jacobian/八方向响应、非线性、饱和、crop、CUDA和state门通过，但axis类别8 recall为`0.8235`，
feature ratio为axis `0.4231`、uniform-1234 `0.5223`、holdout `0.5419`，且防模板坍缩门失败。
没有optimizer、checkpoint、smoke或训练授权。当前无活动训练主线；不得调参、重跑v7 admission、
导出或promotion，后续只能另立新的电子流形设计合同。

当前分支为 `experiment/v2-near-conjugate-relay-v64`；v6.4 smoke 的科学状态已由
`44981b1` 归档。此前用于 v6、宏观可行性、v6.2 和 v6.3 的本地实验分支指针已精简为
`archive/v2/*` tags；提交和 evidence 均未删除，也未推送远端。
v6.4 保留宏观链路 `label + 8D code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS`，
但使用四个可搭建的两透镜近共轭 1:1 relay：每模块机械段 `25.00/50.00/24.75 mm`、两片
`f=25 mm` 透镜、ABCD `[-1,+0.25 mm;0,-1]`，四模块总机械长度 `399 mm`。通过的 v6.2
PCA electronic state 被显式复制并冻结；仅四层 phase 可训练；全局照明强度透过率固定 `tau=0.80`。

relay equivalence、admission、semantic tangent 和 observability 都已通过。初始 axis/uniform-1234/
uniform-5678 的 accuracy 为 `0.9765/0.9882/0.9706`，camera fraction 为
`0.2623/0.4604/0.4938`，实际 pre-clamp saturation 为 `0.0449/0.0613/0.0621`；trace 明确含
四次 `near_conjugate_relay_lct`、Encoder、identity OLS、Decoder 与 CMOS。

唯一400-step CUDA smoke 已完成：
`outputs/20260729_v2/runs/20260729_233809_929370_diagnostic_v2_pca_near_conjugate_relay/`。
evidence 为 `failed`，没有后续授权。uniform-5678 的类别7 recall 是 `0.8824 < 0.90`，最终
最低 feature ratio 为 axis `0.3982`、uniform `0.5833/0.6479`，不满足预注册 smoke 下限。
同时，现有 smoke harness 未把 ratio、delta correlation、relay crop retention 和最后50步
distribution-loss 趋势完整写成 GateResult。因此它能阻止正式训练，却**不能作为 v6.4 的有效架构失败归因**；
归档类别为 `invalid_harness_evidence`。禁止重跑 smoke、正式5-epoch训练、5000 validation、60000导出
或从 smoke-final warm-start。若继续，必须先立独立的 harness 修复与重新资格化计划，且不得创建v6.5/v7、
调整 B/tau/LR/loss或降低质量门。

v6.3 是已停止的正式1.50 mm历史主线：`conditional_pca_formal_spacing_self_imaging_v6_3` 保留
`label + code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS` 与正式
1.50 mm 层距；只显式接续已通过的 v6.2 电子 PCA 载波、basis 与逐类逐方向 scale，四层相位从零开始。
0.25 mm 链已因 full-connect 可搭建性、uniform retention 和真实 pre-clamp 饱和门而关闭，只能作为未来单片堆叠器件的诊断资料。

v6.3 的唯一 400-step smoke 已执行并失败，故没有 5-epoch 训练授权。最终 axis camera fraction 为
`0.2203 < 0.25`，而 uniform-1234/5678 为 `0.4149/0.4339`；三面板 accuracy 与零码 accuracy
均为 `1.0`，饱和约 `0.054–0.059`，方向响应、五组更新和 CUDA 门均通过。结果说明 phase+carrier
联合优化可恢复语义，却把最严格的轴向随机码物理响应从初始 `0.2531` 压低。smoke final 仅为诊断，
不得 warm-start；不重复、不延长或调参。后续只能明确选择 1:1 relay 重设计或单片多层器件路线。

历史候选 `conditional_multi_anchor_local_modes_v5` 已完成实现和无更新容量审计，但容量准入失败，现为
`rejected candidate / diagnostic-only`，不是当前活动候选。v1 的 LDM 蒸馏线仍由 `train.py` 维护，
但不定义 v2 的训练或验收输入。

## 当前 v6.4 诊断合同

- 宏观链仍为 `label + 8D code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS`。
- 电子端使用通过 v6.2 资格化的逐类 PCA carrier/basis/scale，并在 v6.4 中永久冻结。
- 四层 phase 是唯一训练参数；层间使用四个两透镜近共轭 relay，而不是 1.50 mm 自由传播或
  0.25 mm 器件间隔。
- relay 的机械段为 `25.00/50.00/24.75 mm`，有效 ABCD 为
  `[-1,+0.25 mm;0,-1]`；全局照明强度透过率固定 `tau=0.80`。
- 532 nm、SLM/相位面规格、2+2 phase-only、finite-crop、identity OLS、后端 relay 和固定曝光
  `0.4913053810596466` 保持不变。
- evaluator 只用于训练和验收；最终推理不运行 teacher、diffusion、VAE 或迭代优化。
- v6.4 当前仍是 diagnostic-only，不能被称为正式硬件或数据集已通过。

## 当前停止规则

v6.4 唯一400-step smoke 的 evidence 为 `failed` 且无后续授权；同时 harness 遗漏了
feature-ratio、delta-correlation、relay crop-retention 和 distribution-loss trend 的独立
GateResult。因此当前同时满足两条停止理由：

1. 不得启动正式5-epoch、5000张自动评估、60,000张导出、memorization、utility或promotion。
2. 不得把本次结果直接宣称为 v6.4 架构失败；应先用现有只读 history/raw artifacts 补齐并重放
   gate，生成 superseding evidence。禁止使用 smoke-final warm-start，也不重复同一400-step CUDA。

历史 identity、aligned、reanchor、v3–v6.3 的失败证据均保持不可变；它们不能直接授权 v6.4 或
任何后续正式训练。

## 当前合同运行结果（2026-07-24）

- schema-v2 evaluator qualification 通过；旧 schema-v1 evaluator 被内容绑定检查拒绝。
- injection admission、observability 和 control isolation 通过 harness；control isolation 分类为
  `semantic_alignment_required`（command semantic accuracy 1.0、raw code fraction 0.3202、饱和 0.0583）。
- 只读 semantic localization 分类为 `local_semantic_alignment_plausible`，参数 state hash 未变化。
- 当前合同唯一一次 100-step smoke 失败：raw accuracy 0.217647，code fraction 0.320250，饱和 0.057841；
  六组梯度/更新及 Wz 恢复门通过。失败后不得重复 smoke 或直接训练。
- 运行目录：`outputs/20260724_v2/runs/20260724_164030_016157_formal_v2_identity_logit_amplitude_finite_crop_identity/`。
- 400-step semantic recovery smoke 已通过：raw accuracy 0.435294，前/后50步 classification loss
  为 3.849249/1.876344，code fraction 0.310312，饱和 0.056744，raw 相对变化 0.089787；
  六组梯度/更新、Wz 恢复、provenance 和 schedule/context 门全部通过。
- recovery 运行目录：`outputs/20260724_v2/runs/20260724_172048_171701_formal_v2_identity_logit_amplitude_finite_crop_identity/`。

任何 evidence 的 system/config/evaluator/split/checkpoint/action 不匹配，都必须在创建
optimizer 前失败。旧 global-amplitude、teacher/prototype/capacity、direct-template 和
LightGen evidence 不能授权 identity 主线。

## 多样性恢复断点（2026-07-25）

- 旧400-step recovery：camera fraction `0.1436`、最低距离比 `0.310`。
- aligned smoke：训练batch camera fraction约`0.242`，固定170面板仅`0.1581`，最低距离比`0.268`。
- Wy=0只读反事实：轴向camera fraction `0.3034`、最低距离比 `0.5406`、accuracy `0.8941`；uniform accuracy约`0.77`。
- identity-reanchor smoke：轴向camera fraction `0.2961`、最低距离比 `0.5226`、accuracy `0.8941`；uniform accuracy `0.8118`。
- 结论：自由Wy/Wz会压缩独立code，global modes单独训练又不足以恢复非零code语义；问题已定位为物理多样性与类别语义的结构容量冲突。

## 历史证据

旧运行手册与完整指标保留在 [`docs/archive`](../archive/)，阶段摘要在
[`experiment_history.md`](../status/experiment_history.md)。它们具有参考价值，但不提供当前执行权。

## 历史下一动作记录：v5 后的连续模式审计

以下小节保留 2026-07-27 当时的决策，不是 2026-07-30 的当前执行授权。

当前没有正式训练授权。v3/v4和v5均保持失败证据只读。v5容量审计已经证明固定四锚点等权硬路由不满足当前MNIST类内几何，
因此不得运行v5初始化准入、可观测性、400-step smoke或正式训练。下一步仅允许运行
`audit conditional-continuous-mode-capacity`，先比较数据对齐的连续PCA模式与v5硬锚点基线；该审计不创建optimizer或checkpoint。

2026-07-27 的首次v3 smoke因派生代码遗漏复制已训练的Encoder/Decoder相位而作废，
不作为模型失败证据。派生准入现已增加step-0 raw-CMOS等价门；只有该门通过后运行的smoke
才是该主线唯一有效的400-step结果。

修复后有效v3 smoke已完成并失败：axis accuracy `0.8941`、最低recall `0.5882`、
camera fraction `0.2997`、最低距离比`0.5273`；uniform-1234 accuracy `0.8059`，
uniform-5678 accuracy `0.7412`。三面板的camera fraction、饱和、rank、零码、冻结参数和
class-mode更新门均通过，但语义门、axis距离比及15%分类loss下降门未通过。该evidence不授权
984-step训练、自动评估或导出；下一步必须另立结构方案，不得重复该v3 smoke或延长相同训练。

## v4 parity audit（2026-07-27）

已实现候选 `conditional_parity_stabilized_modes_v4`：冻结v3奇模式，增加零初始化、仅非零码生效的偶语义稳定器。该候选尚未获得训练授权。

只读 parity conflict audit 的结果为 `failed / parity_cause_unconfirmed`：奇梯度冲突为 `71/80`，类别5和7均命中，偶梯度正向对齐通过；但真实梯度向量抵消率仅从奇分量 `0.1721` 改善到偶分量 `0.0691`，改善 `0.1030`，未达到合同要求的 `0.15`。state hash 未变化，未创建optimizer，未派生v4 checkpoint，未运行v4 smoke。

该失败证据只说明偶稳定器方向尚未充分证明，不授权任何训练、正式评估或数据集导出。不得放宽抵消率门或重复相同审计。

## v4 semantic tangent audit（2026-07-27）

新增的无更新 `conditional-semantic-tangent` 审计已完成，证据位于
`outputs/20260727_v2/runs/20260727_203639_023816_formal_v2_parity_stabilized_modes_finite_crop_identity/`。

evaluator validation accuracy 为 `0.9876`，state hash 前后一致，未创建 optimizer。Q 方向对所有低margin样本均为正，functional probe 的面板 accuracy 分别由 `0.8941/0.7706/0.7059` 提高到 `0.9118/0.8412/0.8000`；但每类特征距离比下降超过允许的 `0.03`，违反多样性保护门。因此最终分类为 `local_semantic_unreachable`，evidence 状态为 failed，`permitted_next_actions=[]`。

该结果表明语义修正方向存在，但在保持当前物理多样性、零码和饱和约束的切空间内不可达。不得运行 v4 admission、400-step smoke、984-step正式训练、自动评估或数据集导出；继续扩展生成器前必须另立结构/目标合同。

## v5 容量审计（2026-07-27）

v5代码、配置和CLI已实现，但无更新四锚点容量审计未通过：平均类内特征离散度下降 `0.1691`，最低类别下降 `0.1243`，低于固定平均 `0.30`/最低 `0.15` 门。证据位于
`outputs/20260727_v2/runs/20260727_211242_776413_formal_v2_multi_anchor_local_modes_finite_crop_identity/`。

该失败是有效停止证据；不得降低门限、运行v5准入/可观测性/400-step smoke或启动正式训练。数字1还存在156张小簇，后续容量gate必须显式报告最小cluster count。
下一步仅允许连续模式容量审计；若其通过，才能另立candidate admission合同。

## 连续模式容量审计（2026-07-28）

无更新数据审计已完成，证据位于
`outputs/20260728_v2/runs/20260728_180501_490139_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。
固定55k训练子集上，PCA-8相对v5 K=4基线的平均优势为 `0.4269`、最低类别优势为 `0.3052`，振幅空间PCA-8最小解释收益为 `0.4732`，十类模式均为rank 8；数据容量门全部通过。

同一证据的物理probe因当前环境 `CUDA unavailable` 分类为 `harness_invalid`，未创建optimizer、未写入checkpoint、未改变任何参数。该结果只能授权在CUDA可用且内容上下文一致的环境中重跑同一无更新物理probe，不授权连续模式训练或正式评估。

## 连续模式CUDA probe合同复核（2026-07-29）

20260729 CUDA运行完成了旧版连续模式容量/物理probe，但旧版evidence的`passed`不能作为准入依据：axis accuracy为 `0.8235`，uniform-1234/5678 accuracy为 `0.5824/0.5706`，低于原语义门；类别5/7 recall显著不足。旧实现还复用了v3固定gain `0.5`，没有执行预注册的 `0.05` SLM-logit RMS安全探针、逐方向响应、梯度和stateless切空间门。

因此该run标记为 `probe_gain_contract_mismatch / harness_contract_incomplete`，原始文件保持只读，不修改、不迁移其`permitted_next_actions`。下一步必须运行新的
`audit conditional-continuous-mode-reachability`，固定 `alpha=0.05/tanh(1)`，补齐语义、梯度、逐方向响应和完整10×17 contact sheet门。该命令仍不创建optimizer或checkpoint。

代码侧已补齐该只读合同：新审计固定 basis hash
`de283cad951b1fddb54499c16c032fc44f6beb5239734d78795475effbe890bb`，生成三个完整
`10×17` contact sheet，并对 class carrier、continuous PCA modes、Encoder phase、Decoder
phase 分别执行可逆的一次性切空间 probe。probe 将分类 margin 梯度与分布保持梯度归一化组合，
实际限制 SLM-logit RMS `≤0.05`、相位 RMS `≤0.005 rad` 和 raw-CMOS 相对 RMS `1%`；不创建
optimizer、不写回参数、不生成 checkpoint。修复前当前会话 CUDA 不可用，故尚未运行该新物理审计；
CUDA 可见性恢复后仍须重新执行该审计，才能宣称连续模式语义可达或授权后续架构/训练。

## WSL CUDA 环境修复（2026-07-29）

此前的 `torch.cuda.is_available() == false` 不是模型或审计失败：WSL 的 NVIDIA 驱动库目录
`/usr/lib/wsl/lib` 没有在导入 PyTorch 前进入动态链接器路径。已为 Conda 环境
`optical-generative-models` 安装 activation/deactivation hook；新的 shell 执行
`conda activate optical-generative-models` 后会自动设置 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`，
并已验证 `torch.cuda.is_available() == true`、`device_count == 1`、设备为 RTX 3060 Laptop GPU。
此修复只恢复硬件可见性，不改变任何训练、审计或物理合同。

## v6 PCA transport smoke 失败闭环（2026-07-29）

v6 admission 与 observability 已通过，唯一一次 400-step CUDA smoke 已完成但失败。最终固定面板
accuracy 为 `1.0`，saturation 为 `0.0467/0.0474/0.0482`；但 raw-CMOS code fraction 仅为
`0.0106/0.0325/0.0354`，最低归一化特征距离比为 `0.288/0.399/0.442`。step 200→400 的
camera fraction 增量只有 `0.0013/0.0031/0.0041`，不满足 `0.03` 门；因此不得运行 v6 正式
5 epochs、自动评估或数据集导出。

原 smoke 的 updates gate 还错误地把固定 `lowpass_mask` 当作必须更新参数。只读复核已修正为只检查
五个可训练组；其余固定状态保持不变。复核证据位于
`outputs/20260729_v2/runs/20260729_172156_692157_diagnostic_v2_pca_transport_codesign_finite_crop_identity/`，
最终结论仍是 `valid_v6_model_failure_with_harness_defects`。

随后完成一次流式宏观光路局部可行性审计，未创建 optimizer、未写回 checkpoint，state hash 前后一致。
在 band-limited SLM、SLM+共享 phase counterfactual 和 identity-phase reference 三个局部切空间中，
均无法同时达到语义和物理多样性门，分类为 `local_macro_generation_unreachable`。审计产物位于
`outputs/20260729_v2/runs/20260729_173629_136389_diagnostic_v2_pca_transport_codesign_finite_crop_identity/`。

该分类是当前 v6 局部合同下的停止证据，不是对任意未来光学系统的数学不可能性证明。当前不创建 v7，
不重复 smoke，不调整 gain/LR/loss/scale，不启动正式训练。下一阶段只能进行宏观物理合同复审：明确是否
允许改变相位初始化、层数/间距/孔径或任务门限；在此决定前不得进入 5000 张 validation、60,000 张导出
或 utility 闭环。

## 连续 PCA 安全尺度可达性审计（2026-07-29）

修复 WSL CUDA 路径后，`conditional-continuous-mode-reachability` 已在 RTX 3060 Laptop GPU
完成，证据位于
`outputs/20260729_v2/runs/20260729_131120_150834_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。
审计初版在 6 GB 显存上错误地同时保留 170 个完整光路反向图而 OOM；实现现已改为每类17个样本
立即消费并累计梯度，固定三块 10×17 面板、无 optimizer、无 checkpoint、参数哈希不变的合同保持不变。

安全增益 `alpha=0.05/tanh(1)` 准确得到 axis SLM-logit RMS `0.0500`。语义在安全尺度下显著恢复：
axis/uniform-1234/uniform-5678 accuracy 为 `0.9824/0.9529/0.9529`，饱和均约 `0.0076`；四组
gradient observability 均通过，零码和光路 trace 均有效。但物理随机码传输不足：raw CMOS code fraction
仅 `0.0071/0.0212/0.0240`，物理 SLM `A²` code fraction 仅 `0.0107/0.0323/0.0352`，低于固定 `0.10`。
最终分类为 `failed / safe_scale_transport_insufficient`，不授权连续模式 admission、smoke 或训练。

这说明旧 gain `0.5` 的语义退化主要来自过大扰动；但在符合预注册安全 logit 范围的单位 RMS PCA 模式下，
变化尚未以足够强度穿过完整光路。不得恢复 gain `0.5` 或进行 gain sweep。该失败现为不可变 v6 输入证据：
v6 联合训练载波、20 cycles/mm 固定低通 PCA 模式、有界模式尺度和四层相位，建立新的传输合同。
只允许按 `admission → observability → 一次400-step smoke → 用户5 epoch` 推进；任一门失败即停止。

## v6.2 短层距自成像合同（2026-07-29）

v6 的有效停止证据已归档到 `docs/status/v6_failure_attribution.md`。用户选择新的
diagnostic-only 物理合同：层距从 1.50 mm 改为 0.25 mm，保持 2+2 层、relay、identity
OLS、finite_crop、曝光和 2048 网格不变，并加入同 code 自成像 `κA²` 与 delta 传输损失。

当前只允许执行 `conditional-short-spacing-target-admission`；它通过后才能执行 optical
admission、observability 和唯一一次 400-step smoke。短层距 checkpoint 与旧 v6 checkpoint
严格不兼容，旧 smoke final 禁止 warm-start。正式训练、5000 validation 和 60000 导出仍未授权。

## v6.2.1 短层距光学准入失败（2026-07-29）

每类每方向 `(0.05,0.50)` 电子标定已经通过语义、`A²` code fraction和方向响应门；首次v6.2
target evidence因此标为 `invalid_harness_evidence`。零phase短层距CUDA准入随后完成，冻结
`κ=0.997967`；原报告的“无饱和”结论无效，因为它在固定曝光 clamp 后测量饱和。Encoder layer 2 retention相对1.50 mm仅为axis `+27.4%`、
uniform `+16.0%/+14.3%`，未满足三个面板均需`+25%`的物理门。

该历史 evidence 已标记 `invalid_saturation_harness`，必须用 pre-clamp 高饱和比例重跑一次 optical admission；
无论结果如何都不得运行observability、400-step smoke、正式训练或任何数据集验收。`0.25 mm`只保留给未来
单片堆叠器件诊断，当前实验台主线回到正式 `1.50 mm` 合同。
## v6.4 near-conjugate relay recovery（已归档为 harness-invalid）

## v7 supersession 与 v7.1 受控替代（2026-07-30）

v7 的 admission evidence 是有效的未准入结果：axis 类别 8 recall 为
`0.8235`，四面板 feature-ratio 最低值为 `0.4231/0.5223/...`，均独立违反
不可降低的门。该 run 的 anti-template 字段曾错误取中心化 17 样本矩阵的最后
奇异值；现更正为 `s[7]/s[0]`，因此 supersession 分类为
`valid_v7_failure_with_rank_gate_defect`：rank 子门无效，但不改变 recall 与
feature-ratio 的独立失败结论。历史 outputs/evidence 不重跑、不重写。

唯一允许的后续电子候选是诊断架构
`conditional_transport_aligned_manifold_v7_1`。它冻结 carrier、v7 同源的
32 个 train-only 低通 PCA 候选模式和每方向尺度，仅训练 `[10,32,8]`
transport mixing 与四层 phase；零码严格回到 carrier。固定顺序为 design →
CUDA no-update admission → tangent → 一次 400-step smoke → 5 epoch。任何前置
门失败都产生空 `permitted_next_actions`，直接停止电子端版本迭代并转向物理容量重设计。

v7.1 的执行合同已预注册：design 必须同时校验正负 scale 的语义、pre-clamp
saturation 与 raw/feature response；admission 使用四个固定 170 样本面板，复核
command/raw `s[7]/s[0]`、feature rank、170 raw 唯一性、delta correlation、crop、
CUDA 和严格 checkpoint 回读；tangent 只审计 mixing/encoder/decoder/joint 的安全
无更新方向；唯一 smoke 固定为 seed=1234、400 steps、五个 probe。未通过任一门不得
创建可 warm-start checkpoint 或进入 5 epoch。本轮只运行至 smoke 的有效结论。

执行结果：v7.1 design 已在 CUDA 完成并有效失败。所有类别的 transport response
rank=8 且 `s[7]/s[0]=0.2692--0.5902`，双符号 scale、zero carrier、relay provenance、
state 与显存门通过；但 central-difference nonlinearity 为 `0.1132--0.3192`，全部不能
满足预注册的 `≤0.10`。因此未发布 design artifact，evidence 的
`permitted_next_actions=[]`，admission/tangent/smoke 和 5 epoch 均停止。

## v7.1 失败归因与 v7.2 物理容量合同（2026-07-30）

v7.1 的失败是有效科学证据，不是 rank 缺陷：每类 32 个 candidate physical-Jacobian
方向都以最大二阶/一阶响应比聚合，得到 `0.1132--0.3192`（最低值仍高于 `0.10`）。该完整测量
同时包含 sigmoid amplitude command、相干传播后的 intensity square 和冻结 evaluator 的分段非线性，
故不能据此归因为单一器件。rank、双符号 scale、relay 与显存均通过，说明信息可以传输，但不支持
“一阶 transport mixing 足够线性近似局部物理/evaluator 映射”的假设。不得改变 epsilon、聚合规则或门限，
亦不得补跑 v7.1 admission。

下一条且唯一的受控候选为独立 `conditional_transport_aligned_phase_capacity_v7_2`：4+4 phase-only、
amplitude SLM、near-conjugate relay、identity OLS、固定曝光与 `tau=.80` 保持不变。其内部机械距离锁为
`(4+4)*99.75 mm = 798 mm`，trace 必须有八个 relay 与八个 finite crop。v7.1 failed evidence 仅作为内容
绑定的停止事实，绝不授权下游。v7.2 的 design → admission → tangent → 唯一400-step smoke 各有独立
架构与 evidence/hash；任一有效门失败即关闭当前电子+phase 家族，不创建 v7.3 或进行门限、seed、epsilon、LR/gain 扫描。

## v7.2 曲率归因：冻结 evaluator 首次越门（2026-07-30）

新的 no-update `conditional-transport-curvature-attribution` 严格绑定 v7.1/v7.2 failed evidence、
v6.2 initialization、v7 train-only 32-mode basis、split、frozen evaluator、固定曝光与 `epsilon=.01`。
320 个 candidate directions 在 command、OLS field、camera field、raw intensity 与 fixed-exposure
边界的逐类最大曲率均小于 `.10`：raw intensity 为 `.0074--.0103`，fixed-exposure 为 `.0101--.0925`。
首次越门只发生在 frozen evaluator feature，逐类最大值为 `.0839--.2674`；最终八个 mixed
directions 的 evaluator-feature 最大值为 `.1035--.3218`，十类全失败。所有 state/CUDA/provenance
门通过，未创建 optimizer、checkpoint 或 artifact。

因此 v7.2 的 design failure 仍保持有效，但其物理归因修正为 `evaluator_feature_local_nonlinearity`，
而非 CMOS square-law 或 relay 传输非线性。根据预注册决策，禁止启动 balanced coherent readout、
OLS/readout 或更多 phase 层扩容；下一步只能另行决定是否重定义冻结 evaluator 的局部可线性特征合同，
或收缩任务。证据位于 `outputs/20260730_v2/runs/20260730_161035_715678_diagnostic_v2_transport_aligned_phase_capacity_v72/`。

## Evaluator Qualification v2 停止（2026-07-30）

新 evaluator-v2 合同在任何新 transport design 前有效失败。历史 ReLU evaluator 的总体 validation
accuracy 为 `.9876`，但类别4 recall 为 `.954 < .97`；独立 smooth evaluator 的 accuracy 为
`.9846 < .985`，类别9 recall 为 `.966 < .97`。两者原始 validation agreement `.9834` 通过，
但固定 erosion 扰动的一致率仅 `.5064/.5346`，dilation 也仅 `.9508/.9152`，不满足 `.98`。
该结果表明新合同把原 evaluator qualification 未曾要求的类别下界和形态侵蚀鲁棒性提升为硬门，
因此不能作为允许重启 transport 的已资格化验收仪器。未运行 evaluator curvature baseline、
2+2 design、admission、tangent 或 smoke；无新 transport artifact/checkpoint 被创建。
证据位于 `outputs/20260730_v2/runs/20260730_164205_687065_diagnostic_v2_transport_aligned_manifold_v71/`。

The v6.3 formal-spacing smoke is a valid failed diagnostic: allowing the
electronic carrier, PCA modes and scales to update restored classification but
reduced the axis code fraction from 0.2531 to 0.2203.  It must not warm-start
another run.  The active branch is `experiment/v2-near-conjugate-relay-v64`.

v6.4 freezes the passed v6.2 electronic PCA target and trains only the four
phase surfaces.  Its inter-layer transport is a mechanically buildable,
two-lens 1:1 near-conjugate relay (25.00/50.00/24.75 mm segments, 25 mm
lenses, 99.75 mm per module, effective ABCD `[-1, +0.25 mm; 0, -1]`).  A
global intensity transmission of 0.80 is part of this physical contract; it
is not per-sample exposure normalization.  The required sequence is relay
equivalence → admission → semantic tangent → observability → one CUDA
400-step smoke. 该 smoke 已完成；详细结果与停止理由见本文件开头。它是阻止训练的
diagnostic-only evidence，不能升级为可推广的架构失败结论。
