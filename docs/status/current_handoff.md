# 当前主线交接：v6.3 正式 1.50 mm 自成像恢复（2026-07-29）

当前实验台主线是 `conditional_pca_formal_spacing_self_imaging_v6_3`。它保留
`label + code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS` 与正式
1.50 mm 层距；只显式接续已通过的 v6.2 电子 PCA 载波、basis 与逐类逐方向 scale，四层相位从零开始。
0.25 mm 链已因 full-connect 可搭建性、uniform retention 和真实 pre-clamp 饱和门而关闭，只能作为未来单片堆叠器件的诊断资料。

v6.3 的唯一 400-step smoke 已执行并失败，故没有 5-epoch 训练授权。最终 axis camera fraction 为
`0.2203 < 0.25`，而 uniform-1234/5678 为 `0.4149/0.4339`；三面板 accuracy 与零码 accuracy
均为 `1.0`，饱和约 `0.054–0.059`，方向响应、五组更新和 CUDA 门均通过。结果说明 phase+carrier
联合优化可恢复语义，却把最严格的轴向随机码物理响应从初始 `0.2531` 压低。smoke final 仅为诊断，
不得 warm-start；不重复、不延长或调参。后续只能明确选择 1:1 relay 重设计或单片多层器件路线。

当前活动候选 `conditional_multi_anchor_local_modes_v5` 已完成实现和无更新容量审计，但容量准入失败，现为
`rejected candidate / diagnostic-only`。v1 的 LDM 蒸馏线仍由 `train.py` 维护，
但不定义 v2 的训练或验收输入。

## 已锁定合同

- amplitude SLM、532 nm、完整 4f relay、2+2 phase-only finite-crop D2NN、identity
  OLS、1.50 mm layer spacing、B=0 backend relay。
- 输入为 `label + 8D code`；identity-anchor generator 的 gain 为 `0.5`，seed 为
  `1234`，固定曝光为 `0.4913053810596466`。
- evaluator 只用于训练和验收；最终推理不运行 teacher、diffusion、VAE 或迭代优化。
- 当前配置位于 [`configs/v2/current`](../../configs/v2/current/)，旧配置由迁移 manifest 追溯。

## 当前停止规则

evaluator、injection、observability、control isolation、semantic localization 和 schedule
completion 的内容绑定链已经通过。语义训练已完成，但多样性恢复的 aligned 和 identity-reanchor
两条路径均未通过。因此不运行984-step正式恢复、5000张自动评估、60,000张导出、memorization、
utility或promotion。所有failed diversity smoke均为不可变诊断证据。

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
[`experiment_history.md`](experiment_history.md)。它们具有参考价值，但不提供当前执行权。

## 下一允许动作

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
