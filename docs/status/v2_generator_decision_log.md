# v2 generator 决策记录

本记录只总结当前teacher-free全光路生成主线的结构决策，不授权任何训练或数据集导出。

## v2.2 closeout（当前有效状态，2026-08-04）

`E2:O0_declared_teacher_free_objectives` 已以
`closed_conditional_negative` 封存。三个 dual-bundle confirm 均未通过完整的
qualified dual-evaluator 门（`0/3`），因此 O1--O5、Teacher/WGAN、Controlled
Shapes 与 Fashion-MNIST 分支被 `stopped_by_o0_gate` 停止。

这只支持“当前电子 O0 与已声明 teacher-free objective family 不可达”的条件性
结论；不支持 phase-only、光学容量、空间 DOF 或普适 semantic--diversity conflict。
历史 v6/v7、revision-9 和本文件以下的历史记录不能重新授权物理训练。完整证据位于
[`experiments/archive/v2_2_capacity_map_closeout/`](../../experiments/archive/v2_2_capacity_map_closeout/)。

任何未来研究必须另立电子 O0 合同与新的 objective family，并重新通过数字 oracle 的
multi-seed dual-evaluator 确认；不得恢复当前 O1--O5 的长训练。

## v2.3 电子数字上界分支（2026-08-05）

为避免把 v2.2 的双评价器错配直接外推为光学失败，已从
`archive/v2/capacity-map-v2.2-conditional-negative` 建立独立分支
`experiment/v2-o0-feasibility-v23`。该分支只实现 teacher-free、单次前馈的 D0/C0--C3
电子候选、projection discriminator 训练目标、四固定面板测量、三 seed 阶段门和
locked-final evidence 治理；不导入光学 operator、Teacher、diffusion 或外部数据。

revision-2 validate、CUDA preflight 和 controls 均通过；D0 seed-1234 probe 在 step 50
有效失败。worst normalized margin 未改善，digital code fraction 从约 0.0323 降至约
0.0020；finite、safety、saturation 仍通过。依据停止规则，smoke、confirm、locked-final
以及 C0--C3 全部停止，唯一分类为
`training_objective_or_evaluator_inconclusive`，仅适用于 D0 image-domain objective。
这不支持任何光学、phase-only、空间 DOF 或普适容量结论。

## v2.3-r2 correction / v2.4（2026-08-06）

逆向审计发现 v2.3-r2 的 D0 probe 为 `invalid_harness`：模型初始化未绑定
全局 seed，且 classifier/Q 梯度未执行对应 optimizer 更新。原始 evidence 保持
不可变并仅作运行事实；不得据此宣称 feature-matching 主导、目标冲突或 generator
capacity failure。v2.3 的 locked-final、C0--C3 和光学分支从未执行。

v2.4 新建 `o0-identifiability` 合同，只验证 seed/init、DQ/G 参数边界、恢复和
100-step 训练可识别性；结果无论如何均不授权物理推进或 formal promotion。

v2.4 CUDA preflight 通过，唯一 D0 seed-1234 100-step probe 在首个有效 health
gate 关闭：real auxiliary classifier accuracy 从 0.10 提升到 0.26，但 Q latent
error 未达到 10% 改善门，分类为 `discriminator_identifiability_failure`。这只说明
训练可识别性仍未满足，不说明 D0 分布不可达或任何光学容量限制；C0--C3 与物理链路
均保持未执行。

## 已停止的结构

| 候选 | 停止原因 |
| --- | --- |
| Identity v2 `Wz/Wy` | 能保持类别语义，但通过跨维系数映射压缩随机码。 |
| Wy=0 / v3共享或类条件模式 | 多样性恢复后，非零码语义与类别5/7冲突。 |
| v4 parity stabilizer | 只读切空间能改善margin，但会破坏特征多样性；奇偶抵消改善不足。 |
| v5四锚点局部模式 | 四簇平均收益仅0.1691，最低类别0.1243；数字1有156张小簇，不支持等权硬路由。 |
| v6.4 冻结PCA + phase-only relay | replay确认ratio/分布趋势失败；安全encoder/decoder/joint tangent均不能改善三面板瓶颈。 |
| v7 32-mode nonlinear manifold | 容量通过，但固定CUDA admission的axis语义、ratio与防模板坍缩失败。 |

## 当前有效边界

上述失败均不否定宏观物理链路：

```text
label + code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS
```

它们否定的是电子端入口的参数化和数据几何假设。失败evidence保持不可变，只能用于下一次诊断定位，不能授权训练、自动评估或导出。

## 当前状态（2026-07-30）

2026-07-31 起，evaluation-qualified 2+2 主线采用“完整 runner 先行”的工程策略：先冻结
共享 runtime 和全链测试，再执行一次有阶段门的 CUDA 序列。曲率、单面板小样本 correlation 和
局部 tangent 的逐面板改善只承担诊断职责；最终双-seed、语义、物理安全、分布和证据完整性门不变。

该链的 shared runtime 和四个 stage runner 已完成实现，但尚未运行新的 CUDA design。先前的
曲率 evidence `20260731_001324_194392` 是唯一允许消费的曲率输入；历史两份曲率 evidence
均不重跑。当前剩余的是完整 pytest 的环境受限确认，而非任何新的科学门失败。

Evaluator v2.1 已通过独立资格化。它不改变历史 v7.1/v7.2 的有效失败：那些 evidence
仍绑定各自的 evaluator 合同。v2.1 仅证明新的双-evaluator bundle 可以作为下一条独立 2+2
transport 诊断链的测量仪器；后续仍须依次通过曲率基线、design、admission、tangent 与预注册的
两 seed smoke，才可能授权正式训练。

v7使用v6.2的carrier与前8模式，新增训练split PCA 9--32和无bias类条件MLP；不读取v6.4
diagnostic final。容量 evidence `20260730_030257_695213` 通过，但 admission
`20260730_030608_413918` 为 `nonlinear_expanded_manifold_not_admitted`。因此当前没有可训练
generator候选；下一步只能提出新结构合同，不可重试、调scale或启动smoke。

v7.1 是唯一受控替代：冻结 v6.2 carrier、v7 同源 32-mode pool 和逐方向 scale，
仅允许训练 `[10,32,8]` transport mixing 与 2+2 phase。它必须完成
`design → no-update admission → no-update tangent → 唯一400-step smoke`；任何失败
均停止电子生成器迭代并转向物理容量合同，不执行本轮 5-epoch。

该候选已在 design 门停止：完整 CUDA central-difference response 的每类 rank 均为8，
但最小 nonlinearity 为 `0.1132`，高于锁定上限 `0.10`。这是一条有效的 v7.1
transport-linearization 失败，不授权 admission 或后续训练。

## 当前 v6 决策门

连续 PCA 容量审计已通过，但安全尺度物理传输审计以 `safe_scale_transport_insufficient` 失败：语义安全范围与
可验收 raw-CMOS code response 没有重叠。当前候选为 v6 `conditional_pca_transport_codesign_v6`：它明确共同优化
载波、20 cycles/mm 固定低通 PCA 模式、有界尺度和四层相位；不是 gain sweep，也不复用旧 generator 权重。
v6 只可按 `admission → observability → 一次400-step smoke → 用户5 epoch` 推进。任一门失败即转入宏观光路
生成可行性复审，禁止 v7、降门限、重复 smoke 或延长相同训练。

## v6 smoke 与宏观可行性复审结果（2026-07-29）

v6 的 admission/observability 通过，但唯一 400-step smoke 真实失败：语义 accuracy 达到 `1.0`，
而三个面板 code fraction 仅 `0.0106/0.0325/0.0354`，最低 feature ratio 为 `0.288/0.399/0.442`。
分类 loss 和 evaluator 特征损失下降没有转化为足够 raw-CMOS 物理变化。固定 `lowpass_mask` 被错误纳入旧
updates gate，已由只读复核工具排除；五组可训练参数仍均有有限非零更新，因此该缺陷不改变模型失败结论。

流式宏观审计在 band-limited SLM、SLM+共享 phase 和 identity-phase reference 三个局部 counterfactual
空间均未满足联合门，结果为 `local_macro_generation_unreachable`。该证据只表示当前 v6 初始化、20 cycles/mm
输入带宽、2+2 phase 和固定曝光合同的局部可达性不足，不授权创建 v7 或继续训练。

当前停止条件：必须先完成宏观物理合同复审，明确是否允许改变 phase 初始化、光学层容量/间距/孔径或质量门；
在得到明确合同前，禁止正式训练、自动评估、candidate 导出、防记忆、utility 和最终准入。

## v6.2 短层距自成像（2026-07-29）

v6 失败归因为 1.50 mm 层距下相位职责与随机码传输目标错位，而不是 MNIST 容量或注入失败。
真实 MNIST 同一 ANOVA 指标的 camera fraction 约 0.710，故不降低 0.25 门限。新 diagnostic
架构 `conditional_pca_self_imaging_transport_v6_2` 将层距固定为 0.25 mm，并让 raw CMOS
对同 code 的 `κA²` 与 delta 变化负责；target 分支 detach，禁止 teacher/paired-MSE。

旧 v6 证据不能迁移。短层距链必须重新通过 target admission、optical admission、observability
和唯一 400-step smoke；失败后停止，不创建 v7。

### v6.2.1 optical admission结果

修复每类每方向scale与CLI返回码后，电子target通过；但0.25 mm零phase的Encoder layer 2
retention相对1.50 mm只在axis达到`+27.4%`，两组uniform仅`+16.0%/+14.3%`。原报告还在
fixed-exposure clamp 后计算饱和，因此“无饱和”字段无效；历史 evidence 只能作为
`invalid_saturation_harness`归档。修复后将重跑一次 optical admission，但短层距自成像训练
不得通过调loss、重复smoke或降低门限绕过。

### v6.3 实验台主线（2026-07-29）

只读复算表明，使用同一通过电子标定和零相位时，正式1.50 mm raw CMOS code fraction 已为
axis `0.2531`、uniform `0.4603/0.4814`，且实际高饱和比例为`0.054–0.059`；缺口是语义
组织而非随机码消失。下一条受控主线因此是 `conditional_pca_formal_spacing_self_imaging_v6_3`：
保留正式2+2、1.50 mm、identity OLS和固定曝光，用冻结 `κA²` self-imaging/delta target 先做
无更新切空间准入，再最多执行一次400-step smoke。若该切空间或smoke失败，停止当前生成器扩展。

v6.3 不把 0.25 mm 数值结果提升为实验台配置：它使用 formal 1.50 mm、零相位和已通过的电子 PCA
状态重新建立 `κA²` / 同 code delta 自成像合同。先以无更新 semantic-tangent 验证 carrier 与四层相位
的局部语义方向，再运行一次 400-step phase-warmup + joint smoke；不降低语义、多样性或饱和门，也不
复用任何 short-spacing 或 smoke-final 状态。

### v6.3 smoke 结果（2026-07-29）

admission、semantic tangent 和 observability 都通过；tangent 在约 1% raw-CMOS 限制内使三面板的
低 margin 样本正向改善比例达到 `0.94–0.97`，且 state hash 不变。唯一 smoke 则在 step 400 取得
三个面板和零码 accuracy `1.0`，uniform camera fraction `0.4149/0.4339`，但 axis fraction 只有
`0.2203`，低于不可降低的 `0.25`。分类 loss 前50/后50均值从 `1.5472` 降至 `0.0606`，说明训练
确实向语义模板收敛，却未保持最难轴向 code 的物理差分。该 evidence 为有效核心物理门失败，不授权
5-epoch、validation、导出或任何数据集准入；不得重跑同一 smoke。
## v6.4: near-conjugate relay（2026-07-30，harness-invalid diagnostic）

The exact B=0 relay preserves command intensity but leaves phase-only masks
unable to create useful intensity changes; the 0.25 mm free-space alternative
is not a normal free-space bench contract and also failed its repaired
retention/saturation gate.  v6.4 therefore uses a physically positive-distance
two-lens relay whose LCT is `[-1, +0.25 mm; 0, -1]`.  It retains the electronic
PCA response while preserving a limited phase-to-intensity channel.  This is a
new architecture and evidence chain; no v6.3 final state is compatible.

实现分支为 `experiment/v2-near-conjugate-relay-v64`，实现提交范围 `07cd1f9` 至 `8fb6174`，
科学状态由 `44981b1` 归档。电子 PCA
carrier/basis/[10,8] scale 来自通过的 v6.2 target 并永久冻结；四层 phase 是唯一可训练参数；
全局照明透过率为 `tau=0.80`。relay equivalence、admission、semantic tangent 与 observability
通过，且初始化三面板同时满足语义、camera fraction、方向响应与实际饱和门。

唯一400-step CUDA smoke 已结束。其 fraction 为 `0.2625/0.4606/0.4941`，saturation 为
`0.0451/0.0613/0.0621`，delta correlation 为 `0.8760/0.9281/0.9256`；但 uniform-5678
类别7 recall `0.8824`，且最低 feature ratio 为 axis `0.3982`、uniform-1234 `0.5833`、
uniform-5678 `0.6479`。该 evidence 的 status 是 failed，故不授权正式训练。

不过 smoke 未把预注册的 ratio、delta-correlation、relay crop-retention、最后50步分布损失趋势
全部写为 GateResult。因此不得将它写成 v6.4 模型失败；正式分类为 `invalid_harness_evidence`。
禁止重跑、延长、调 B/tau/LR/loss、放宽门限或从 diagnostic final warm-start。后续只能先制定独立的
harness 修复与重新资格化计划。
