# D2NN 实验与验收历史

## 2026-08-24：H-SCED lifecycle screen closeout与 H-PGC 注册

`H-SCED` 注册为 H-SWAE 之后的新 electronic O0 假设。唯一变量是把每类小批量
quantile-SWD prior alignment 替换为 Gaussian sliced Cramér/energy U-statistic，目标是
避免 `n=10` 条件下的 variance-shrink optimum，同时保持 reconstruction、cycle、novelty
和 dual-evaluator semantic gates。

实现、配置和测试已提交于 `0d83a4d`；paired-batch preflight 修复提交于 `9290016`，
lineage-bound lifecycle 修复提交于 `70b8e32`。clean HEAD 上的 manifest-bound
validation、512 个固定 replication 的 synthetic audit 和三步 CUDA preflight 均为
`status=valid, passed=true`，evaluator calls 为零；对应 evidence 位于
`outputs/20260824_v2/runs/20260824_electronic_o0_prior_native_sced_v1/`。

preflight 的实际结果包括 encoder/decoder finite 更新、SCED `0.0211737 → 0.0117856`、
reference SWD `0.280985 → 0.264564`、cycle `0.0221913 → 0.0220756`，所有模型、
optimizer、CPU/CUDA RNG、sampler 和 partial-history replay 均 exact；peak allocation
为 `685118976` bytes，storage 为 `26619` bytes。后续 seed-8601 screen 已在本文件
下方记录为有效科学失败，H-SCED 不再有 confirm 或下游权限。

H-PGC（Prior Gradient Conflict）随后注册为新的无更新归因 lifecycle。它固定消费
WAE seed-7301、H-SWAE/H-SCED seed-8601 final checkpoints 和两个 projection banks，
测量 blockwise/classwise objective gradients、固定权重 cycle 冲突及解析 half-space
projection；无 optimizer、无 evaluator、无训练。最终 validation、audit、report、
closeout 均为 valid，SHA256 依次为 `ac2f28…b14f`、`7496ca…6beb`、
`105d57…2476`、`d777af…e1e0`。SCED full-joint cycle rho 在两个 bank 的四个
batch 均为负（范围约 `-0.663` 到 `-0.239`），H-SWAE paired arm 同样为负（约
`-0.595` 到 `-0.230`），所以分类为 `gradient_conflict_not_supported`，不能用一阶
固定权重冲突解释历史 cycle 恶化。SCED blockwise decoder aggregate rho 仍为负，
SWAE decoder upsample/output block 出现局部正值但 full-joint 未冲突；classwise 10
类报告完整，SCED class 8 的 projected retention 约 `0.874`，与 class 6 为定位上的
最低两类，不单独决定分类。peak allocation 为 `821096448` bytes，storage 为
`1183663` bytes，参数 hash、FD/KKT、optimizer/evaluator 隔离均通过。唯一下一动作
是注册 trajectory/curvature attribution；不授权任何训练或下游路线。

## 2026-08-23：prior-native H-SWAE screen stop

`electronic_o0_prior_native_swae_v1` 使用冻结 WAE v2 seed-7301 权重和全新 optimizer，
以 deterministic classwise sliced-Wasserstein quantile objective 替换 MMD。validation、
synthetic scale-direction audit 和 CUDA interrupted/resumed preflight 均通过；source WAE、
flow 和 mixture artifacts 保持只读，训练期间没有 evaluator selection。

唯一 seed-8601 screen 是有效科学失败 `prior_alignment_fit_failed`：aggregate SWD
reduction 为 `34.62% < 50%`，类别 6/7 退化且 cycle 指标变差。依据 fail-fast 合同，
没有进入 confirm、8602/8603、locked-final、O0-G、SMF、E1 或 fiber-aware training。
closeout 固定下一动作 `close_current_prior_native_swae_route`；该结果只关闭 H-SWAE，
不支持 decoder、光学、fiber、phase-only 或硬件的普遍失败结论。

## 2026-08-24：H-SCED seed-8601 screen stop

H-SCED 在冻结 WAE v2 seed-7301 权重上使用 Gaussian sliced Cramér/energy U-statistic，
完成了 validation、512-replication synthetic audit、CUDA preflight 和一次完整的
seed-8601 full-coverage screen。训练期间 evaluator calls 为零；screen checkpoint、
training history、decoder hash 和 report/closeout consumer chain 均完整。

该 screen 是有效科学失败 `prior_alignment_fit_failed`：training-bank aggregate SWD
reduction 为 `39.7259% < 50%`，reference-bank reduction 为 `36.6652% < 50%`，cycle
由 source `0.0164912` 变为 `0.0225073`。SCED aggregate 从 `0.0390378` 降至
`0.0204746`，reconstruction objective 从 `0.2676530` 降至 `0.2533920`，但不足以
满足完整 screen gate。novelty exact duplicates 为 `0`、aggregate memorization pass
fraction 为 `0.9246`、prior saturation 为 `1.28e-6`、Jacobian rank pass fraction 为
`1.0`；这些通过项不能抵消 SWD/cycle 失败。

screen 后仅执行了纯消费者 `report → closeout`。closeout 分类为
`prior_alignment_fit_failed`，下一动作为 `close_current_prior_native_sced_route`。
没有进入 confirm、seed-8602/8603、locked-final、O0-G、SMF、E1 或 fiber-aware training；
该结果只关闭 H-SCED，不构成 decoder、光学或 fiber 容量失败结论。

## 2026-08-24：H-TCA prior trajectory/curvature attribution 注册

H-PGC 的有效 paired attribution 否定了终点固定权重的一阶 cycle conflict：H-SWAE 与
H-SCED 的 full-joint cycle rho 在两个 projection bank、四个 batch 上均未达到冲突阈值。
但两条旧路线没有保留中间 checkpoint，因而无法定位 cycle 恶化是在梯度场旋转、AdamW
实际位移、有限步长曲率还是 observer 分布偏差中产生。

H-TCA 注册为 diagnostic-only lifecycle。它将从 WAE seed-7301 source 克隆临时模型，
按两条旧路线各自的原始 objective、seed-8601、schedule 和 625 steps 做确定性 replay，
在 0/50/.../600/625 landmark 记录 raw 与 AdamW 方向、固定 observer 方向导数和
实际 cycle 漂移，并在选定点做 HVP 与有限差分。终态必须逐项匹配旧 final checkpoint；
evaluator calls 为零，旧 checkpoint/evidence 只读，且不生成可训练 consumer 的模型。

H-TCA 实现及加固提交为 `84fbb7d`、`1ba714b`、`f18d71b`、`c616c7a`、`4486033`、
`3634221`、`6bd465c`、`44c032e`、`7929779`。retry7 validation/preflight/audit/report/
closeout 均 valid，SHA256 依次为 `0f5c40…c2b`、`e1d6be…c024`、`8d17ef…aa44`、
`b424f1…6707`、`53d0d5…3a9c`。两条 arm 的 625-step replay 均 exact 重现旧 final
encoder/decoder/optimizer/RNG/sampler/schedule，SCED history 13 个窗口 exact；14 个
landmark（含终点 625）、peak allocation `1771866112` bytes、storage `10217971` bytes、
optimizer 只在临时副本创建，evaluator calls 为零。

有效分类为 `trajectory_attribution_underdetermined`，两条路线的四种机制均未满足
连续 landmark、双 projection bank 和 observer-batch 归因门；因此不能把历史 cycle
恶化归因于 trajectory rotation、AdamW preconditioning、finite-step curvature 或
observer distribution mismatch。H-TCA closeout 的下一动作是
`register_denser_local_trajectory_attribution`，仅允许更密集的局部诊断注册，不得直接
启动 projected-gradient、SGD、trust-region、adapter 或任何新训练；confirm、O0-G、
光学、fiber 和 formal promotion 继续关闭。

## 2026-08-24：H-DLTA dense local trajectory attribution 注册

H-DLTA 注册为 H-TCA 的独立 follow-up diagnostic-only lifecycle。它不改变模型、objective、
seed、训练 schedule 或 checkpoint policy，只把 landmark 固定加密为 39 个点，并在相同
历史 replay 中解析 AdamW 的 raw gradient、momentum、adaptive preconditioner、weight
decay 与实际 displacement，同时测量 actual-step cycle curvature 和 observer mismatch。

H-DLTA 只读绑定 WAE seed-7301、H-SWAE/H-SCED seed-8601、H-PGC 与 H-TCA retry7 lineage，
不创建 evaluator、不生成候选模型、不修改旧 evidence。它必须先完成独立的 validate →
preflight → audit → report → closeout；任一 harness/资源门失败均停止并从新 clean commit
重建 lineage。所有有效分类只能注册无更新 counterfactual 假设；confirm、projected-gradient、
SGD、trust-region、adapter、O0-G、光学和 fiber training 继续关闭。

## 2026-08-24：H-DLTA dense local attribution closeout

H-DLTA 实现提交为 `95e61c5`，manifest lineage alias 修复为 `f3d5962`，SCED
13th partial history window flush 修复为 `d5670e3`。首轮 audit 因 SCED 最终 history
只有 12 个窗口而 fail-closed 为 `invalid_harness`（audit SHA256
`0eba966b0e5644ccf1518b45d3522fc5c078d5e98f88a1d85a68b5acaeec2fe0`）；该尝试保留，
未被后续 evidence 消费。修复后从新 clean HEAD 重建 lineage，retry validation、preflight、
audit、report、closeout 的 SHA256 分别为：

`a2045d600065a87c4db89cb305d19931a646241363cd53eb61750ba387d50edf`、
`1207f1a9426876300008b4e237c505b56ab016668130801bb915175d49fd1502`、
`d168f8bd23e6ed60c8a84175af88408d3a73f6a220b5df7bba4749018740b18e`、
`797e5aacc817ca3bb763a170b8310645f3a25547c52985c4ab9ae5c1f22e2506`、
`a80ef182b88c5d16e2400ca80ded1053c7ede0ffc7387dbbe601f8b3109abd55`。

retry replay 覆盖固定 39 个 landmark：early `0,2,…,20`、W250 `230–270`、W350
`330–370`、W500 `480–520`、anchor `625`。两条 arm 的 encoder、decoder、optimizer、
CPU/CUDA RNG、sampler、schedule 和终态 canonical hash exact；SCED 13 个 history windows
exact；AdamW predicted/actual displacement、reverse-direction control、HVP 和
`α=0.25/0.5/1.0` finite-difference 均通过。peak allocation 为 `3457404416` bytes，
run storage 为 `22407988` bytes，evaluator calls 为零，旧 source/route checkpoint 与
upstream evidence hash 未改变。

最终分类为 `dense_local_attribution_underdetermined`：H-SWAE 与 H-SCED 在四个窗口均未
支持 raw gradient rotation、momentum、adaptive preconditioning、weight decay、
finite-step curvature 或 observer sample distribution mismatch；没有 first-supported
landmark。classwise locator 固定记录于 `0/250/350/500/625`，class 6/8 仅作定位字段，
不改变主分类。closeout 下一动作是 `close_optimizer_trajectory_attribution_family`；
这只允许登记无更新 counterfactual hypothesis，不授权修改 optimizer/objective/架构、
任何新训练、confirm、其他 seed、O0-G、光学、fiber 或 formal promotion。

## 2026-08-24：H-CCF prior cycle composition factorial 注册

H-DLTA 已有效关闭 optimizer/trajectory attribution family，未发现可重复的 raw-gradient、
momentum、adaptive preconditioning、weight-decay、finite-step curvature 或 observer
distribution 机制。下一条只读反事实注册为 `H-CCF`：对 WAE source、H-SWAE final 和
H-SCED final 的 encoder/decoder 做 2×2 factorial 组合，测量
`L00/L10/L01/L11` cycle，并计算 encoder、decoder、interaction 与 Shapley 贡献。

H-CCF 不创建 optimizer、不加载 evaluator、不生成候选 checkpoint、不修改旧 evidence，
只允许登记下一条 no-update diagnostic hypothesis。正式执行顺序固定为
`validate → preflight → audit → report → closeout`；任一 lineage、null swap、historical
cycle reproduction、finite、hash 或资源门失败均停止，不形成科学结论。confirm、其他 seed、
projected-gradient、SGD、trust-region、adapter、O0-G、光学、fiber training 和 formal
promotion 继续关闭。

## 2026-08-23：electronic O0 mixture transport screen stop

在冻结 WAE v2 encoder/decoder 的条件下，建立了
`electronic_o0_mixture_transport_v1`。每个类别拟合 `K=2/4/8` 的
full-covariance Gaussian mixture，使用 validation NLL 选择 component 数，
并以独立 design seed `173205` 选择 covariance temperature；evaluator 在
拟合与选择阶段调用次数为零。

clean commit `67f9d50` 上的 CUDA validation 与 preflight 通过：EM 三次更新
finite、分量更新完整、interrupted/resumed replay exact。seed `8501` screen
为有效科学失败：mixture validation NLL 为 `9.364574`，standard Gaussian
为 `12.044989`，但 classwise sliced-distance reduction 只有 `12.5758%`，
低于 `30%`；temperature `1.35` 下独立 novelty memorization pass fraction
为 `0.894`，低于 `0.95`，exact duplicate 为零。

依据 fail-fast 合同，没有运行 evaluator confirm、8502/8503、locked-final、
O0-G、SMF、E1 或 fiber-aware training。closeout classification 为
`mixture_latent_fit_failed`，唯一下一动作是
`close_current_wae_posterior_sampler_family`。该结果只关闭当前 mixture
posterior-sampler 假设，不外推为 decoder、光纤或 phase-only 的普遍失败；
所有结果保持 diagnostic-only，未提交 outputs。

## 2026-08-23：electronic O0 latent transport attribution and flow stop

在不修改 WAE v2 encoder/decoder、无 evaluator 训练调用的条件下，冻结
55,000-row train posterior 与 5,000-row validation posterior，完成
`electronic_o0_latent_transport_attribution.v1`。P0 标准 Gaussian 与
Aglobal/Aclass affine transport 未通过完整 dual-evaluator gate；独立
Eposterior 与 Rencoded 上界各有 2/3 locked panels 通过，第三 panel
secondary 同方向改善，因此只授权一次 latent-only RealNVP preflight。

preflight 的三次 CUDA update、WAE 隔离和 deterministic interrupted/resumed
replay 通过；seed8401 screen 的 validation NLL 为 `10.0161`，P0 为
`12.0450`，flow sliced distance 为 `0.05688`，P0 为 `0.15620`。固定
10-epoch seed8401 confirm 是有效科学失败
`latent_flow_semantic_failed`，不是 harness failure。locked panels 的
primary/secondary accuracy 为 `0.982–0.986`/`0.972–0.992`，但 minimum
recall 分别为 `0.88–0.94`/`0.90–0.96`；memorization pass fraction 为
`0.7876`，低于 `0.95`，虽 exact duplicate 为 0。官方 10k accuracy 为
`0.9869/0.9798`，不能覆盖 locked-panel 与 memorization 失败。

依据 seed8401 fail-fast 合同，没有训练 8402/8403，没有 locked-final、
O0-G、SMF、E1 或 fiber-aware training。该记录关闭的是当前 electronic
latent-transport family，不外推为 decoder、光纤、phase-only 或硬件的
普遍不可能性；outputs 保持未提交的诊断证据，当前 action 为
`no_o0_g/no_smf/no_e1/no_fiber_training/no_formal_promotion`。

## 2026-08-22：WAE electronic O0 v1 invalid-harness correction and v2 stop

旧 v1 WAE evidence 经只读复核后归档为 `invalid_harness`，原因是重建分支
误用 random prior code，且 camera fraction、diversity、coverage、shuffled
control、official test、Jacobian、memorization 和 resume contract 均不完整；
旧 `wae_image_manifold_failed` 不再具有下游权限。随后在 clean commits
`17ee681`、`e6673b0`、`ed03cf5` 上完成 v2 修复并执行 CUDA validate、preflight、
完整覆盖 screen 和 seed-7301 五 epoch confirm。v2 confirm 的 primary 为
`0.930–0.952`，secondary 为 `0.826–0.852`，secondary minimum recall 为
`0.36–0.44`，agreement 为 `0.828–0.856`；官方 10k secondary 为 `0.8374`、
minimum recall `0.3683`。camera fraction 约 `0.637`，真实随机标签 shuffle
primary `0.090–0.102`，30 个 Jacobian 点均为 rank 8，deterministic replay
通过。该结果是有效的 `wae_image_manifold_failed`，不是 harness 失败。

失败发生在 O0-I，故没有运行 7302/7303、locked-final、O0-G、SMF 100/196/400
端口密度上界、ridge readout、E1 或任何 fiber-aware training。它说明修复后的
电子图像流形仍未通过 qualified dual-evaluator semantic contract，不能外推为
单模/少模/多模光纤性能结论；outputs 仅作运行证据，未提交。

本文件保存带日期的实验事实和验收证据，不定义当前执行契约。迁移或续接当前工作时先读 [当前正式基线与活动研究交接](current_handoff.md)；当前架构见 [D2NN 单次曝光蒸馏主线](../architecture/v1_single_shot_distillation.md)，stage 命令与输入要求见 [Stage Guide](../operations/stage_guide.md)，完整导航见 [Documentation](../README.md)。本文件的“当前/下一步”均表示各记录当时的结论；历史证据保留，不应覆盖现行合同。

## 2026-08-22：synthetic fiber route formally archived without E1

The v2 synthetic fiber attribution chain was closed and archived at
`experiments/archive/v2_fiber_route_no_e1_closeout/` after a read-only reverse
necessity check. v3 remained `valid_failed /
no_reproducible_fiber_gain`; interface attribution retained class structure
and local port observability; global exposure explained approximately 95.4%
of the per-sample oracle gain but did not restore minimum recall; and decoder
controllability did not recover secondary semantics. The final decision is
`no_fiber_training`, `no_e1`, `no_modal_regularizer`, and
`no_formal_promotion`.

This record is synthetic diagnostic-only and is not a formal optical contract.
No new CUDA measurement, training, evaluator comparison, checkpoint, or
hardware claim was created. Restart requires a new electronic O0 hypothesis
that can change the rejection decision, independent qualification, and a
multi-seed dual-evaluator digital oracle before any new synthetic fiber route.

## 2026-07-30：v7.1 transport-aligned manifold 合同预注册

在 v7 的有效 CUDA admission 失败后，唯一允许的电子候选为 v7.1：冻结 carrier、
32-mode candidate pool 和 calibrated scales，以完整物理 relay Jacobian 拟合 8D
transport mixing。其运行顺序、阈值、固定 seed、唯一 smoke 与停止规则均写入
runner/evidence；此条目仅记录预注册合同，不记录任何 CUDA 结果。

随后 design 在 CUDA 上完成。双符号 scale 的语义、饱和和 raw/feature response 全部合格，
response rank 也满足8维；但类别 nonlinearity 落在 `0.1132--0.3192`，超过
`≤0.10` 的预注册上限。design artifact 未发布，后续 admission、tangent、smoke 未运行。

## 2026-07-30：v6.4 近共轭 1:1 relay 诊断（harness-invalid）

分支 `experiment/v2-near-conjugate-relay-v64` 实现了四个 `25.00/50.00/24.75 mm` 的两透镜
近共轭模块，总机械路径399 mm，对应有效 ABCD `[-1,+0.25 mm;0,-1]`。通过的 v6.2 PCA
electronic state 被冻结，统一使用 `tau=0.80`，仅训练四层 phase。equivalence、admission、
semantic tangent 和 observability 通过；400-step smoke 保持高 code fraction 和低饱和，
但 uniform-5678 类别7 recall 为0.8824，最终 feature-ratio 下限也未满足。

smoke 漏掉若干预注册 GateResult（ratio、delta、relay crop retention、分布损失趋势），故记录为
`invalid_harness_evidence`，而非模型失败。它不授权正式5-epoch、validation、导出、防记忆或utility，
也不得用 diagnostic final warm-start。下一步若继续必须先创建独立的 harness 修复和重新资格化合同。

## 2026-07-29：v6.2 短层距归档与 v6.3 正式间距接续

v6.2 的逐类逐方向电子标定已证明 SLM 前端可以同时保留语义与强 code 响应；但 0.25 mm optical
admission 的 uniform retention 相对 1.50 mm 仅 `+16.0%/+14.3%`，低于三个面板均需 `+25%` 的门。
历史报告还在 clamp 后统计饱和，故其“0 饱和”字段无效；pre-clamp 复算的高饱和比例约为
`9.4%–10.4%`，超过 `0.08`。该链归档为 diagnostic-only，不授权训练。

同一电子 command 在正式 1.50 mm 零相位合同下仍保留 raw-CMOS code fraction：axis `0.2531`、
uniform `0.4603/0.4814`，实际饱和 `0.054–0.059`；但分类 accuracy 仅
`0.6412/0.5000/0.4941`，零码 accuracy `0.80`。这说明随机变化没有被物理传播消除，断点是零相位
尚未把高响应变化组织为可读数字。v6.3 因而只接续电子状态并重置四层相位，以 formal 1.50 mm
self-imaging semantic-tangent → 一次 smoke 的新证据链检验该可达性；它不是层距或增益 sweep。

该链的 admission、无更新 tangent、observability 和唯一 400-step smoke 都已实际执行。tangent
显示局部语义方向存在，五个训练组都有有限梯度；smoke 的三面板 accuracy 与零码 accuracy 最终均为
`1.0`，类别 loss 前50/后50均值 `1.5472 → 0.0606`。但 axis camera fraction 从初始 `0.2531`
降为 `0.2203`，未达到 `0.25`；uniform-1234/5678 仍为 `0.4149/0.4339`。饱和、方向响应、显存与
更新门均通过。因此这不是未收敛或 CUDA/harness 问题，而是当前 self-imaging 目标把语义恢复与 axis
物理差分推向不同方向。v6.3 正式训练、自动评估和导出均停止。

## 状态快照：2026-06-10

- **已验收稳定基线**：`3k/fast200` teacher distill dataset + bs128 matched teacher + seed202 one-shot distill，学生复现实验使用 `one_shot_distill_epoch_0020.pt` 做 sample/eval。
- **默认最佳 checkpoint 口径**：常规 distilled run 使用 `one_shot_distill_best.pt`；seed sweep 或历史验收复现可使用文档明确指定的 epoch checkpoint。
- **候选主线**：`15k_fast200` export、`guidance_scale=4.0`、`gs25/30/40` sweep、label-strict filter、readable split、teacher oracle/latent audit 当时作为候选链路使用。它们完成 `one_shot_distill -> one_shot_sample -> one_shot_distill_eval` 全链路验收前，不替代已验收稳定基线。
- **诊断/探索工具**：teacher label audit、strict filter、readable split、oracle audit、latent audit 用于筛选和定位 teacher targets，不属于最终硬件推理路径。

## 2026-06-03 全链路验收

**基线提交**：`c99f559 refactor: training orchestration into stage modules`

**环境**：RTX 3060 Laptop GPU (6GB), CUDA 12.1, PyTorch 2.5.1+cu121, WSL2 Linux

**证据文件**：[acceptance_baseline_c99f559.json](../archive/acceptance_baseline_c99f559.json)

| Stage | 配置 | Epochs | 关键指标 | Exit |
| --- | --- | --- | --- | --- |
| `teacher_first_stage` | `configs/train_ldm.yaml` | 35 | `camera_corr=0.9547`, `balanced=0.8770`, `mnist_score=0.8144` | 0 |
| `one_shot_distill` | `experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml` | 20 | `quality=0.1614`, `camera_dice=0.4575` | 0 |
| `one_shot_sample` | 同上 config | - | 可视化、诊断和 tensor bundle 完整 | 0 |
| `one_shot_distill_eval` | 同上 config | - | `mse=0.020`, `corr=0.807`, `dice=0.756` | 0 |

- Teacher 蒸馏数据为 3000 samples，源自 `teacher_distill_export`（batch size 96、guidance scale 2.0、steps 200）。
- 训练和采样均使用 CUDA。
- 原始验收使用 `one_shot_distill_best.pt`；2026-06-04 seed sweep 复现使用 `one_shot_distill_epoch_0020.pt`。
- 当前 run artifact/checkpoint 位于 `outputs/YYYYMMDD_<version>/runs/<run>/`，不纳入 git；旧记录中的历史路径已迁移为真实带日期前缀的版本目录。

### 评估结果汇总工具

`ldm_project.eval_reports` 可只读汇总已有 run，返回 `stage`、`checkpoint`、`distill_dataset`、`best_loss`、`eval_metrics` 和 `sample_metrics`。缺失 optional artifact 时对应字段为 `null`，缺失 run metadata 时记录 error；该工具不生成 checkpoint、图片或新 artifact。

当前命令模板和回归比较方式由 [Stage Guide](../operations/stage_guide.md) 维护。

### 回归比较

比较方向保持如下：

- `eval_mse`、`eval_sat`、`eval_slm_tv`、`best_loss`：lower is better。
- `eval_corr`、`eval_dice`、`eval_diversity_ratio`、`sample_corr`、`sample_dice`：higher is better。
- 多个 candidate run 按 best-of 合并：lower-is-better 取最小值，higher-is-better 取最大值。
- Baseline 值为 `null` 的指标跳过；candidate 缺少 baseline 要求的指标视为失败。
- 阈值优先使用 baseline JSON 的 `tolerances`，其次使用代码内默认值。

### bs128 匹配扩散基线

**验收日期**：2026-06-03

**证据文件**：[acceptance_baseline_bs128_matched_diffusion.json](../archive/acceptance_baseline_bs128_matched_diffusion.json)

bs128 first-stage 相比旧基线将训练吞吐从 563 step/epoch 改善到 71 step/epoch，同时保持或超过核心重建指标。

| 指标 | c99f559 | bs128 matched | 变化 |
| --- | --- | --- | --- |
| `eval_corr` | 0.807 | 0.809 | +0.002 |
| `eval_dice` | 0.756 | 0.777 | +0.021 |
| `eval_mse` | 0.0204 | 0.0209 | +0.0005 |
| `best_loss` | 0.161 | 0.143 | -0.018 |
| `eval_slm_tv` | 0.382 | 0.277 | -0.105 |
| `sample_corr` | 0.588 | 0.550 | -0.038 |
| `sample_dice` | 0.625 | 0.616 | -0.009 |
| `diversity_ratio` | 0.531 | 0.269 | -0.262 |

核心重建指标持平或提升，SLM pattern 更平滑。`diversity_ratio` 下降的记录结论是 teacher target 多样性提高而学生未完全复现，不作为该基线的阻塞项。

## Matched First-Stage 与 Diffusion 证据

稳定规则是 `first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自同一 first-stage 训练的 latent/decoder 分布；规则和原因由 [架构文档](../architecture/v1_single_shot_distillation.md) 维护。

| 配置 | `eval_corr` | `eval_dice` | `eval_mse` |
| --- | --- | --- | --- |
| bs128 first-stage + **匹配** diffusion | 0.809 | 0.777 | 0.0209 |
| bs128 first-stage + **旧 grid_size=32** diffusion | 0.421 | 0.483 | 0.0404 |

不匹配时 corr 从 0.809 降到 0.421，dice 从 0.777 降到 0.483，mse 约翻倍。`best_loss` 反而从 0.161 降到 0.154，说明 shape/edge/SLM-TV regularizer 可能掩盖重建退化，不能用单一 loss 判断 checkpoint 是否配套。

## Diversity 实验（负结果）

以下实验以 bs128 matched-diffusion 基线（`diversity_ratio=0.269`）为起点，记录在此以避免重复尝试已经验证的方向。

### code_diversity_weight 翻倍

- 变更：`code_diversity_weight` 从 0.03 增至 0.06。
- 结果：`diversity_ratio=0.295`，但 corr 降 3.2%、dice 降 3.0%、`best_loss=0.160`。
- 结论：重建代价过大，该方向不可行。

### teacher_pairwise_diversity_weight 翻倍

- 变更：`teacher_pairwise_diversity_weight` 从 0.005 增至 0.01。
- 结果：`diversity_ratio=0.312`、`corr=0.776`、`dice=0.750`、`best_loss=0.163`。
- 结论：重建指标低于验收标准，损害大于 code-diversity 调整。

### Multiclass Batch

- 变更：`paired_class_batches: false`、`multiclass_batches: true`。
- 结果：`diversity_ratio=0.292`、`corr=0.783`、`dice=0.757`。
- 结论：diversity 几乎不变，重建轻微退化，视为中性方向。

### 同配置重新训练

| 指标 | 原基线 | Retrain ep20 |
| --- | --- | --- |
| `eval_corr` | 0.809 | 0.810 |
| `eval_dice` | 0.777 | 0.774 |
| `eval_mse` | 0.0209 | 0.0205 |
| `diversity_ratio` | 0.269 | **0.322** |
| `sample_corr` | 0.550 | 0.561 |
| `sample_dice` | 0.616 | 0.609 |
| `best_loss` | 0.143 | 0.142 |

记录结论是 diversity 的改善来自 D2NN phase 初始化差异。后续方向应使用多 seed 验证波动范围，而不是继续增加上述 loss 权重。

## 诊断与复现记录

### Epoch Checkpoint

`training.one_shot_distill_epoch_checkpoint_interval` 控制周期 checkpoint；默认 0 表示关闭。启用后生成 `one_shot_distill_epoch_XXXX.pt`，并记录到 `run_meta["epoch_checkpoints"]`。历史诊断配置为 `experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_debug_epoch_ckpt.yaml`，interval 为 2。

### Seed Sweep

`training.seed` 或 CLI `--seed` 固定 one-shot 训练随机性；默认 `null` 保持随机行为。历史流程固定 teacher dataset 和正式 student 配置，分别运行 seed 101、202、303，再对每个 ep20 checkpoint 执行 sample/eval。

通过标准为三个 seed 至少两个同时满足：`eval_corr >= 0.79`、`eval_dice >= 0.765`、`sample_corr >= 0.53`、`sample_dice >= 0.60`、`eval_diversity_ratio >= 0.31`。

**记录提交**：`ebba8e9`

**证据文件**：[acceptance_seed_sweep_ebba8e9.json](../archive/acceptance_seed_sweep_ebba8e9.json)

| 指标 | 阈值 | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- | --- |
| `eval_corr` | >=0.79 | 0.805 | 0.809 | 0.810 |
| `eval_dice` | >=0.765 | 0.771 | 0.780 | 0.779 |
| `sample_corr` | >=0.53 | 0.553 | 0.558 | 0.524 |
| `sample_dice` | >=0.60 | 0.607 | 0.612 | 0.582 |
| `eval_diversity_ratio` | >=0.31 | 0.357 | 0.333 | 0.301 |

三个 seed 中两个达标。seed 202 的重建与 diversity 最稳；seed 101 的 diversity 最高；seed 303 是 eval 重建通过但 sample 和 diversity 略低的 near-miss。

### Post-Refactor One-Shot Sample Smoke

`one_shot.py` 拆分为 `one_shot_model.py` 与 `one_shot_optical_layers.py` 后，使用 seed202 ep20 checkpoint 跑通 CUDA `one_shot_sample`。

**记录提交**：`8a307f3`

**证据文件**：[acceptance_post_refactor_one_shot_smoke_8a307f3.json](../archive/acceptance_post_refactor_one_shot_smoke_8a307f3.json)

| 项 | 值 |
| --- | --- |
| run | `outputs/20260514_unversioned/runs/20260604_160035_377670_train_post_refactor_one_shot_sample_seed202` |
| checkpoint | `outputs/20260514_unversioned/runs/20260604_145621_832064_train_one_shot_distill_seed202/checkpoints/one_shot_distill_epoch_0020.pt` |
| `sample_corr` | 0.5448 |
| `sample_dice` | 0.6095 |

结论：拆分后学生采样路径仍可用，sample metrics、tensor bundle、D2NN diagnostics 和可视化 artifact 均正常生成。

## 稳定基线复现记录

当时推荐使用 bs128 matched teacher、filtered distill dataset、seed202 one-shot distill，并以 ep20 checkpoint 作为复现 checkpoint。默认 config 不固定 seed，正式复现实验通过 CLI 显式传入 `--seed 202`。

维护结论：

- 不继续提高 `code_diversity_weight`、`teacher_pairwise_diversity_weight` 或改变 batch composition；记录结果显示这些方向会用重建质量换 diversity。
- 不把 `training.seed` 写入默认 config；复现实验通过 CLI 固定 seed。
- 更换 first-stage、diffusion、teacher distill dataset 或 D2NN 网格配置后，必须重新跑全链路验收。
- `15k_fast200`、guidance sweep、label-strict filter 和 readable split 在该快照中仍是候选链路。
- Teacher oracle/latent audit 只能说明 teacher target 或 latent 链路质量，不能证明 one-shot student 已达到最终硬件推理就绪。

历史使用的具体 run 路径保留在本文件其他记录和 managed `outputs/` metadata 中；当前命令模板见 [Stage Guide](../operations/stage_guide.md)。

## v9.8 full-relay 受控复验

`front relay_coherent_psf + back relay_psf` audit 只证明 relay/readout 保像，不证明随机初始化的 `generator_e` 可以冻结后直接训练 D2NN。

只要配置设置 `one_shot.generator_base_lr_scale: 0.0`，`one_shot_distill` 就必须显式传入 `--one-shot-generator-checkpoint` 或完整的 `--one-shot-checkpoint`，缺失时应 fail fast。

历史诊断最初先运行 1 epoch，再决定是否运行完整 10 epoch；当时的 scratch 输出写到 `/tmp/codex-v9-8-corrected-runs`。当前工作区可追溯的 managed v9.8 artifact 已整理到 `outputs/20260620_v9_8/`，见下一节。

若 warm-start summary 未加载 `generator_e.*`、首轮 label accuracy 仍接近随机，或 strict eval 没有改善，不得解释为 D2NN 可训练性结论；应先检查 checkpoint 与 generator 输入语义。

## 2026-06-20 至 2026-06-22 光路定位与 v9.10 候选

### v9.8 本地可追溯证据

当前工作区可追溯的 artifact 是：

`outputs/20260620_v9_8/runs/20260620_102234_647492_train_one_shot_d2nn_v9_8_front_coherent_psf_back_psf_relay_audit_v2/artifacts/one_shot_identity_audit_metrics.json`

其中正式几何为 D2NN grid `200`、propagation grid `1024`、spacing `0.5 mm`、encoder/decoder 各一层。`front_relay_coherent_psf_sigma1` 的 `corr_to_slm=0.9676`、dice `0.9814`；`back_relay_psf_sigma1` 的 `corr_to_slm=0.9589`、dice `0.9804`。zero-mask 检查通过。该证据说明受控 relay/identity audit 下光路可读，不等同于 code-conditioned student 验收。

迁移交接摘要还报告了另一组 v9.8 formal re-verification 数值（camera corr `0.9476`、dice `0.8328`、foreground hole `0.1245`），但当前工作区未找到对应 artifact，因此这些数值不作为本文件的已复验证据。

### v9.9 迁移交接结论（当前未复验）

迁移摘要报告 v9.9 validation 曾达到 corr `0.9821`、dice `0.9421`、hole `0.0110`，但独立 code 导致类别形态漂移，固定 zero code 的最小 margin 也不足。当前工作区没有对应 checkpoint、metrics 或 `/tmp` artifact；因此这里只记录交接结论，不把这些数值晋级为本地验收事实。维护决策仍是：v9.9 不能作为鲁棒 code-conditioned baseline。

### v9.10 状态

v9.10 新增 label-only template generator 与 `intensity_to_amplitude` command transform。checkpoint 自包含 `[num_classes, 1, H, W]` intensity template bank，接口保留 code 但明确忽略；模板和 D2NN 均冻结。该版本只能称为 `prototype_baseline`。

2026-06-22 的受控 CUDA audit 已完成并通过。冻结参数检查中 generator、encoder/decoder phase 与 amplitude 的 `max_abs_change` 均为 `0.0`；独立 test split 得到 `paired_corr=0.99944`、`paired_dice=0.98052`、`student_high_saturation_ratio=0.005975`，三项 required gate 全部通过。脱敏后的持久证据见 [acceptance_v9_10_label_template_inverse_20260622.json](../archive/acceptance_v9_10_label_template_inverse_20260622.json)。

该结果只接受为 label-only `prototype_baseline`：code conditioning 仍未启用，不能解释为同类多样性或最终 generative student readiness。2026-06-22 这次历史审计的原始运行和 checkpoint 仍是 `/tmp` 记录；当前及后续 v9.10 复验默认写入持久的 `outputs/YYYYMMDD_v9_10/` 管理目录。

### 2026-06-11 v7.2 受保护参考 run

`outputs/20260514_unversioned/runs/20260611_154426_446321_train_one_shot_distill_eval` 使用 phase-only D2NN、encoder/decoder 各两层、`200 × 200` 神经元、`2.5 µm` pitch、`5 mm` 层间距。它保留为后续物理尺寸实验的行为参考，目录及其引用的 student checkpoint 和 teacher dataset 不得删除或回填。

该 run 的 `paired_corr=0.82863`、`paired_dice=0.75583`，但 strict readability 未通过：最差类别 label margin 为 `0.01852 < 0.03`，background speckle ratio 为 `0.08374 > 0.07`。因此它是受保护参考证据，不是已接受基线。

### 双光子打印 100 × 100 尺寸候选

新尺寸候选保留 v7.2 的 phase-only、encoder/decoder 各两层结构和 `200 × 200` CMOS/loss 接口，仅把物理 D2NN 改为 `100 × 100` 神经元、`2 µm` 边长/pitch、`1024` propagation padding。有效口径为 `200 µm × 200 µm`。

按最大半锥角全连接公式计算，`532 nm` 下理论最小层间距为 `1.490 mm`，配置采用 `1.50 mm`。该几何与旧 `200 × 200` 相位 checkpoint 不兼容，后续必须从头训练。

2026-06-24 的 diagnostic-only 延长验证显示，phase-amplitude upper-bound Probe K 达到 strict gate：`paired_corr=0.8755`、`paired_dice=0.8408`、`student_high_saturation_ratio=0.00548`。该结果仅授权 `diagnostic-only/phase-amplitude prototype` sample，不证明 phase-only 双光子打印 ready，也不证明 code-conditioned generative readiness。

同轮 phase-only padded/crop 分支未达 strict：`1.50 mm` Probe L 为 `paired_corr=0.3933`、`paired_dice=0.5288`，`0.75 mm` Probe M 为 `paired_corr=0.4213`、`paired_dice=0.5340`。因此当前 `100 × 100` phase-only 快速验收训练暂停，下一步转向更大 aperture、短 spacing/readout plane 重设、relay/crop 对齐等几何/读出设计诊断。

后续 phase-only geometry/readout 诊断确认，离线 readout sweep 不能单独解释失败：L 最佳 corrected `corr=0.4149`、`dice=0.5772`，M 最佳 corrected `corr=0.4703`、`dice=0.6070`。短 spacing warm-start 有局部改善但未接近 strict：`0.50 mm` Probe N 为 `paired_corr=0.5123`、`paired_dice=0.6163`，`0.25 mm` Probe O 为 `paired_corr=0.4870`、`paired_dice=0.5974`。`150 × 150` aperture upper-bound Probe P 从头 3 epoch 失败：`paired_corr=-0.3256`、`paired_dice=0.1453`。本轮未运行 phase-only sample，结论仍是停止当前 phase-only 快速训练，转硬件几何/读出重设计。

按重新标定的硬约束复核后，2026-06-24 的 full-connect 主线延长只保留 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，并且只从符合该几何的 Probe L checkpoint warm-start。Probe Q 再训 20 epoch 后达到 `paired_corr=0.6483`、`paired_dice=0.6440`、`student_high_saturation_ratio=0.00561`；Probe R 降低 phase LR 到 `1e-4` 再训 20 epoch 后为 `paired_corr=0.4924`、`paired_dice=0.5848`、`student_high_saturation_ratio=0.00562`。两者均未达到 strict gate，也未达到继续延长到 50 epoch 的门槛；未运行 phase-only `one_shot_sample`，未写 accepted summary。结论是当前固定 full-connect phase-only 主线不能靠本轮继续训练完成快速验收。

2026-06-25 的 Probe S-50 是 pre-v15 unversioned milestone：它从 Probe S warm-start，仍严格保持 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，并首次让该 full-connect 几何通过 strict readability gate：`paired_corr=0.8467`、`paired_dice=0.8023`、`student_high_saturation_ratio=0.00576`，crop retention mean 为 `0.8577`。但它的 label audit 回退到 `student_label_correct_ratio=0.50`，失败类集中为 `0→5`、`1→8`、`3→5`、`7→9`、`9→8`；因此只证明几何可读性可行，不授权 accepted summary。后续 100 × 100 phase-only label-semantics 批次从 `v15` 版本名开始，避免继续写入 `unversioned`，也避免混入 2026-05-23 已存在的旧 `v10` grid16 批次。

v15 label-repair 从 S-50 checkpoint warm-start，输出进入 `outputs/20260625_v15/runs/`。Probe U 训练 5 epoch 后保持 strict readability：`paired_corr=0.8540`、`paired_dice=0.8092`、`student_high_saturation_ratio=0.00585`，但 `student_label_correct_ratio=0.60`；Probe V 训练 10 epoch 后为 `paired_corr=0.8557`、`paired_dice=0.8099`、`student_high_saturation_ratio=0.00590`，但 `student_label_correct_ratio=0.50`。两者均未达到 label-repair gate `label_acc>=0.80`，因此未运行 sample，未写 accepted summary。当前结论是 full-connect phase-only 可读性已经稳定通过，但 label semantics 仍不稳定，下一步需要重设计 label audit/readout objective 或使用 alternating label/shape schedule。

2026-06-25 的 v15 pair-contrast 继续保持同一硬约束，并把此前误放入旧 `outputs/20260523_v10/runs/` 和 `unversioned` 的 U/V/W/X/Y 新批次 run 迁入 `outputs/20260625_v15/runs/`。Probe X 是本轮 config-only 最好 label 结果：strict readability 仍通过，`student_label_correct_ratio=0.70`，但 `3→5` 与 `7→9` 仍为零类；Probe Y 提高 `[1,8]` margin 后修复 class 1，却破坏 class 4/9。结论是 pair-contrast 只能局部修复且会转移错误，后续 prototype CE / negative-class CE 必须使用 `v15_probe_z*` 命名，不能再复用旧 `v10`。

v15 prototype CE / negative-class margin 已作为下一步 label-semantics 训练路线准备，但尚未执行 CUDA。该路线新增默认关闭的 `student_prototype_ce_*` 和 `student_negative_label_margin_*` loss 配置，目标是直接优化 label-audit prototype 判别边界，而不是继续调 batch mean pair-contrast。Probe Z1 将从 Probe X warm-start 修复 `[1,3,7,9]`，Probe Z2 将从 pre-v15 S-50 warm-start 修复 `[0,1,3,4,7,9]`；两者都必须继续保持 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，输出只能进入 `outputs/20260625_v15/runs/`。除非 strict readability、`label_acc>=0.80` 且 `[3,7]` 不再为零类同时通过，否则不运行 sample、不写 accepted summary。

后续 v15 prototype CE CUDA 结果仍未突破 label gate：Probe Z1 从 Probe X warm-start 后为 `paired_corr=0.8550`、`paired_dice=0.8114`、`student_label_correct_ratio=0.70`，失败为 `0→5`、`3→5`、`4→9`；Probe Z2 从 S-50 warm-start 后为 `paired_corr=0.8507`、`paired_dice=0.8056`、`student_label_correct_ratio=0.70`，失败为 `1→8`、`3→5`、`7→9`。两者都保持 strict readability，但未过三重 gate，未运行 sample。离线 label-audit boundary 诊断 `outputs/reports/twophoton_100x100_v15_label_audit_boundary_summary_20260625_215412.json` 显示 teacher target 在 class-average audit 下为 `1.0` 正确率，student class-average 仍为 `0.70`，multi-prototype student 降至 `0.30-0.40`，而 ROI-weighted student audit 达到 `1.0` 并修复 `3→5`、`7→9`、`4→9`、`1→8` 等失败对。当前结论是 label semantics blocker 主要来自非 ROI / 背景成分污染 prototype correlation；下一步应实现 ROI-weighted label audit/objective，而不是继续 prototype CE 或 multi-prototype CE。

v15 ROI/readout-calibrated 收口进一步确认 full-image standard label gate 不应继续用 loss 调参追逐。ROI label repair 后，Probe AB 保持 strict readability：`paired_corr=0.8505`、`paired_dice=0.8034`、`student_high_saturation_ratio=0.00558`，standard `student_label_correct_ratio=0.70`，ROI audit 为 `1.0`。随后 Probe AC 从 AB warm-start 做背景抑制，背景 speckle 从 `0.1063` 降到 `0.0981`，同时保持 `paired_corr=0.8532`、`paired_dice=0.8138`、standard `label_acc=0.70`、ROI audit `1.0`；因此 AC 是当前最佳 `diagnostic-only/phase-only ROI-readable label-template baseline` checkpoint 候选。Probe AD 继续加入 weak standard label repair 后虽提高到 `paired_corr=0.8589`、`paired_dice=0.8221`、背景 speckle `0.0920`，但 standard `label_acc` 退化到 `0.50`，作为 standard full-image label repair 会破坏类别平衡的负证据。当前收口结论：停止 v15 standard full-image label-gate 训练；若后续要 sample，必须先显式批准 ROI/readout-calibrated diagnostic sample，使用 AC checkpoint，并继续标记 diagnostic-only，不写 accepted summary。

## 2026-07-03 v16-v31 主线推进复盘

本轮对话当前可见 transcript 中可确认发生过 **1 次上下文压缩**：可见上下文开头存在一次自动生成的 conversation summary。若更早、当前不可见片段中还发生过压缩，现有 transcript 无法验证；迁移到新对话时应写作“可确认 1 次”。

所有 v16-v31.1 结果均保持 `diagnostic-only`，没有写入 accepted summary。最终人眼验收仍只看 v4-style preview 的真实最终 `CMOS intensity`，中间 plane、selected readout、software oracle 或 classifier 可分性均不能替代最终验收。

迁移到新对话的重要事实：

- v16-v17 增加 readout/sample calibration 与 terminal readability 诊断；后处理和读出校准未能恢复人眼可读数字，v16/v17 均保持 `diagnostic-only`。
- v18-v19 转向 front topology / front imaging；SLM/encoder 可读性改善，但后续 D2NN/readout 仍塌缩。
- v20-v21 完成 downstream plane gate 与 physical CMOS readout 隔离；确认 chip/probe 与最终 true CMOS 不能混用，最终验收必须看真实 `CMOS intensity`。
- v22-v23 尝试 physical CMOS semantic separation 与 decoder/chip semantic preservation；指标小幅改善，但 prototype audit 和人眼 topology 仍失败。
- v24-v25 做 OLS/decoder expressivity 与 decoder capacity upper bound；decoder bypass、phase-amplitude、加深 decoder 等都未让最终 CMOS 人眼可读。
- v26 新增 offline OLS software oracle；结论为 `ols-decodable`，`encoded_amplitude` oracle accuracy 达到 `1.0`，说明上游表征可被软件读出。
- v27-v29 聚焦 decoder synthesis / controllability / trainability；训练 harness 正常、参数确实更新，但 decoder/chip 从人眼 topology 角度仍不可达。
- v30-v31 测试 target-code / low-frequency bridge；v30 显示 target-code 有一定 rescue 迹象，但 sample/human gate 失败；v31 real encoded/OLS -> target-code bridge 仍为 `metric-proxy-failure`。
- v31.1 运行 bridge observability same-batch hard gate，三路 `encoded->lowfreq`、`OLS->lowfreq`、`OLS->spatial` 的 `target_code_amplitude` / `decoder_input_amplitude` 均未达到 bridge topology gate，最终仍为 `metric-proxy-failure`。

| 版本 | 主线问题 | 关键效果 | 结论 |
| --- | --- | --- | --- |
| v16 | sample/readout 后处理 | calibration-negative | 停止 v16 loss/config 微调 |
| v17 | readout/camera/OLS | terminal CMOS 仍 blob-like | eval strict 不能代表 sample 人眼可读 |
| v18 | front topology | collapse 起点定位到前/编码链路附近 | 转向 front-path 模型 |
| v19 | front imaging | SLM -> encoder 改善，但后级仍塌 | 瓶颈转到 downstream D2NN/readout |
| v20 | downstream plane gate | 定位 camera-path / downstream failure | chip/probe 与 final CMOS 必须分开 |
| v21 | physical CMOS readout | true CMOS corr/dice 提升但不可读 | readout PSF 不足以解决语义 |
| v22 | physical CMOS semantic | corr/dice 小升，audit 仍约 `0.30` | semantic loss 不够 |
| v23 | decoder/chip semantic | CMOS 指标继续小升，人眼仍失败 | collapse 在 decoder/chip topology |
| v24 | OLS/decoder expressivity | decoder-bottleneck / no rescue | 不继续普通 decoder loss |
| v25 | decoder capacity | no-capacity-rescue | 加容量/phase-amplitude upper bound 仍不够 |
| v26 | OLS software oracle | `encoded_amplitude` oracle accuracy `1.0` | OLS/encoded 有可解码语义 |
| v27 | decoder synthesis bridge | physical-decoder-unreachable | 软件可分不等于物理 topology 可合成 |
| v28 | decoder controllability | metric-proxy-failure | classifier 可分但人眼 topology 失败 |
| v28.1 | same-batch overfit | decoder-topology-unreachable | 同批次也无法强行过 topology gate |
| v29 | trainability/target encoding | harness 正常但仍 metric-proxy-failure | 不是参数没更新，是目标/架构问题 |
| v30 | target-code architecture | frequency-encoding-bottleneck；sample gate 失败 | target-code 有方向，但未过人眼验收 |
| v30.1 | sample gate recovery | eval-positive-sample-failed | eval 正例不能提升为 human accepted |
| v31 | real OLS -> target-code bridge | metric-proxy-failure | bridge 输出需单独可视化和 hard gate |
| v31.1 | bridge observability | same-batch 三路均 `metric-proxy-failure` | 转向 v32 stronger bridge / generator-code diversity / target-code encoding redesign |

v31.1 证据：summary 为 `outputs/reports/twophoton_100x100_v31_1_bridge_same_batch_summary_20260703_104933.json`，bridge topology report 为 `outputs/reports/twophoton_100x100_v31_1_bridge_topology_20260703_104933.json`。三路 same-batch eval run 分别是 `outputs/20260703_v31_1/runs/20260703_105222_241271_train_v31_1_probe_a_encoded_to_lowfreq_bridge_same_batch_e8_same_eval`、`outputs/20260703_v31_1/runs/20260703_105606_359124_train_v31_1_probe_b_ols_to_lowfreq_bridge_same_batch_e8_same_eval`、`outputs/20260703_v31_1/runs/20260703_105951_197276_train_v31_1_probe_c_ols_to_spatial_bridge_same_batch_e8_same_eval`。三路 `same_batch_bridge_gate_passed=false`，未触发 test eval 或 sample，`accepted_summary_written=false`。

当前主线结论：不是继续堆 terminal/sample loss，也不是回到 decoder/chip/camera 微调；v31.1 已显示 bridge 输出本身在 same-batch 上仍不可读，下一步应转向 v32 stronger bridge adapter、generator/code diversity 或 target-code encoding redesign。只有 bridge code 本身可读后，才值得继续判断 decoder/chip/camera 哪一段破坏最终 true CMOS。

## 2026-07-10 v32-v47 主线推进复盘

所有 v32-v47 结果均保持 `diagnostic-only`，`accepted_summary_written=false`，没有 checkpoint 被提升为 accepted baseline。最终验收仍只看 v4-style preview 或 CMOS-only review 中真实最终 `CMOS intensity / physical_camera_plane`；中间 plane、topology proxy、classifier accuracy、same/test eval strict gate 和 software oracle 均不能替代人眼验收。v38 已由人工明确判定 final CMOS 仍为 blob/纹理团块，因此截至 v47 仍没有 accepted summary。

### 关键实现与诊断能力

- bridge 增加 `one_shot.target_code_bridge_arch`，支持兼容旧 checkpoint 的 `shallow`，以及 diagnostic `deep_residual`、`multiscale` adapter。
- generator 增加 `label_template_code_residual`：保留 label prototype base，并由 code-conditioned residual network 产生样本变化；eval/sample tensor bundle 同步加入 `generator_base_command`、`generator_residual_map`、`generator_residual_abs_mean` 和背景 residual 统计。
- target-code 依次增加 `thick_lowfreq_digit_code`、`sharp_stroke_digit_code`、`small_sharp_stroke_digit_code`，并增加 `target_code_spatial_scale` 以显式控制前景占比和笔画空间尺度。
- 训练目标增加默认关闭的 `sample_code_diversity_weight`；只作为 sample/code diversity 诊断训练项，不能替代 final CMOS 人眼 gate。
- sample 路径增加 diagnostic-only low-frequency SLM command override，用于直接测试 front encoder/OLS 是否能够承载人为注入的 code variation。
- decoder 增加 `propagation_only_decoder`，与既有 `bypass_decoder` 配合，将 decoder mask、层间自由传播和 back path/CMOS 分开隔离。
- artifact/report 体系增加 residual observability、plane shape、同标签 diversity、class correlation、top confused pairs、CMOS-only review、topology proxy v2 和人工 verdict。v37.1 修复中间 plane `100 x 100` 与 final CMOS `200 x 200` 被错误要求同尺寸的 harness 问题。
- 新增 v32.1-v47 离线诊断工具及对应 persistent launcher/config；报告统一写入 `outputs/reports/`，长 CUDA 训练写入真实版本目录 `outputs/YYYYMMDD_<version>/runs/`。

### 主线推进过程

#### v32-v33.2：bridge 与 code-conditioned generator

- v32 以 `deep_residual`/`multiscale` bridge 和 `thick_lowfreq` target-code 做 stronger bridge upper bound。Probe A/B 在 same/test eval 上出现 `bridge-code-candidate-needs-human-review`，但 sample 仍失败。
- v32.1 no-retrain sample-gap isolation 将失败定位为 `label-template-reference-mismatch`：eval/generated 与 sample/generated 近似固定于 label prototype，而 random sample reference 是另一组真实数字；同时记录 `code_insensitive=true`。
- v33 引入 `label_template_code_residual`，但 A/B/C 因 bridge-only trainability audit 与 `target_code_bridge_source=none` 冲突而无可训练参数；D 只更新 bridge、没有更新 residual。该轮属于 harness-invalid/`residual-inactive` 证据，不能作为模型结论。
- v33.1 关闭错误的 bridge-only audit，并补齐 residual tensor observability。A/B/C/D eval/test 均达到 `code-conditioned-target-rescue`，证明 residual harness 和 code-conditioned target variation 可以工作；C sample 的 arbitrary reference exact metrics 被正确标记为 `sample-reference-metric-invalid`。
- v33.2 强制对 real-OLS Probe D 做 no-retrain sample，结果为 `sample-code-distribution-gap`：code 和 residual 有变化，但从 SLM/target-code 开始同标签输出仍近似固定。

#### v34-v36：sample-time code coupling

- v34 对 `code_residual_scale=0.05/0.10/0.20/0.50` 做 no-retrain sweep。`0.50` 已使 SLM command variation active，但 encoded/OLS/target-code/CMOS 仍近似固定，报告状态为 `command-to-ols-bottleneck`。
- v35 将 generator gain 提高到 `1.0/2.0/4.0`，并加入 direct-SLM low-frequency perturbation oracle。`scale=2.0/4.0` 与 direct-SLM oracle 均能把 code variation 传到 `physical_camera_plane`，说明 front/OLS carrier 在足够强的 command variation 下可工作；最终状态为 `code-coupling-recovered-needs-human-review`，但 CMOS 仍不可读。
- v36 将 high gain 正式纳入 residual+real-OLS bridge 训练。三路均为 `high-gain-training-rescue-needs-human-review`，code coupling 已到达 final CMOS，但最终 preview 仍偏纹理团块，不能 accepted。

#### v37-v40：人工 gate、false positive 与 sharp target

- v37 最初因错误要求 `target_code/decoder` 的 `100 x 100` plane 与 `physical_camera_plane` 的 `200 x 200` shape 一致而得到 `harness-invalid`，不是模型失败。
- v37.1 允许各 plane 独立归一化和独立 topology 计算，只要求 batch/sample 数一致。A/B/C 均成为 `manual-cmos-review-candidate`，但该状态只授权人工审查。
- v38 生成只展示 true final CMOS 的正式 review package。人工 verdict 为 final CMOS 仍是 blob/纹理团块，未通过验收。
- v39 将旧 topology proxy positive、人工失败正式记录为 `topology-proxy-false-positive`，并确认同标签与跨 probe 输出高度同源。v39 曾把局部现象报告为 `camera-texture-amplification`，但该定位随后被 v42 bypass 和 v45 near-field 证据进一步收敛：camera/back path 可以显示数字，当前根因不在 camera。
- v40 增加 `sharp_stroke_digit_code` 和 sample-code diversity loss，状态为 `label-template-collapse-recovered`。sharp target/decoder input 已形成更明确数字，但 final CMOS 仍未通过人眼验收；主线因此转向 decoder physical reachability。

#### v41-v45：decoder/back-path 与传播几何

- v41 对 direct sharp target 测试 phase-only decoder、physical-CMOS loss、phase-amplitude upper bound 和 4-layer decoder，四路均为 `decoder-physical-unreachable`。断点收敛到 `decoder_input -> decoder_output -> physical_camera_plane`。
- v42 no-retrain decoder isolation 显示 `bypass_decoder` 路径下 true final CMOS 可以显示可读数字，而 `propagation_only_decoder` 仍退化为光斑。该证据排除了 back path/CMOS 无法显示数字，首要问题转为 decoder propagation geometry、spacing、aperture 或 target scale。
- v43 对 propagation-only decoder 做 spacing、padding 和 target scale sweep。增大 propagation grid 没有形成主要改善，缩小/变细的 `small_sharp` target 给出正信号，报告为 `target-scale-bottleneck`。
- v44 将 `small_sharp_stroke_digit_code` 和显式 `target_code_spatial_scale` 推进到完整 non-bypass decoder 训练。`scale=0.72` 优于 `0.60`；`0.50 mm` short-spacing Probe E 明显优于 `1.50 mm`，状态为 `target-scale-decoder-recovered-needs-human-review`，但仍未人眼通过。
- v45 固定 direct small-sharp target，扫描 `0.25/0.50/0.75/1.00 mm`。`0.25 mm` Probe A 是当前最佳 direct-decoder 几何：decoder topology proxy `0.851`、final CMOS proxy `0.708`、decoder-vs-target corr 约 `0.877`、CMOS-vs-decoder corr 约 `0.954`。短 spacing 明显优于其他 spacing，但 final CMOS 仍有纹理和类别混淆。

#### v46-v47：near-field 语义分离与 capacity upper bound

- v46 固定 `0.25 mm`，尝试 longer training、cross-class pairwise loss、hard-pair contrast 和 final-CMOS topology/contrast loss。四路训练均健康、decoder gradient 非零，但最大类间相关仍约 `0.912`，最终状态为 `semantic-collapse-still-high`。
- v47 从 v40 direct-sharp checkpoint 分别测试 2-layer phase-amplitude、4-layer phase-only、`propagation_grid_size=1536` 和 phase-amplitude physical-CMOS loss。四路 trainability audit 均通过，上游保持冻结，但 strict test 全部失败：paired corr 约 `0.501-0.508`、paired dice 约 `0.552-0.558`，类别 `0/3/5` label audit 正确率均为 `0`。
- v47 A/C/D final CMOS 高度同源；A-C corr 为 `0.999999`，且 decoder crop retention 均约 `0.9998`，因此 `1536` padding/crop 可基本排除为当前主因。`physical_camera_plane` 与 decoder output corr 约 `0.95`，CMOS 主要继承 decoder 产生的纹理形态。
- v47 最终状态为 `capacity-still-collapsed`。但 phase-amplitude 和 4-layer 结论保留利用率限制：A/D decoder mask amplitude 均值约 `0.982`、空间 std 仅 `9e-5` 至 `1.7e-4`，幅度自由度实际接近未激活；B 新增 L3/L4 phase std 约 `0.032`，明显低于 L1/L2 的 `0.28-0.36`，且总训练只有 36 optimizer steps。因此只能否定当前配置，不能宣称 amplitude modulation 或 decoder depth 已被充分穷尽。

| 版本 | 关键状态 | 主线结论 |
| --- | --- | --- |
| v32 | `bridge-code-candidate-needs-human-review` | stronger bridge 有 eval 正信号，但 sample 未通过 |
| v32.1 | `label-template-reference-mismatch` | eval/sample target 与 arbitrary reference 不一致，generator 仍 code-insensitive |
| v33 | `residual-inactive` / harness-invalid | bridge-only audit 使 residual 未被训练，不能下模型结论 |
| v33.1 | `code-conditioned-target-rescue` | 修复 harness 后 residual active，code-conditioned target 可工作 |
| v33.2 | `sample-code-distribution-gap` | sample variation 未稳定传到 SLM/target-code |
| v34 | `command-to-ols-bottleneck` | scale 0.50 激活 command，但 downstream variation 仍弱 |
| v35 | `code-coupling-recovered-needs-human-review` | scale 2/4 与 direct-SLM oracle 证明 carrier 可承载 code variation |
| v36 | `high-gain-training-rescue-needs-human-review` | 训练后 variation 到达 CMOS，但最终仍 blob-like |
| v37 | `harness-invalid` | 100/200 plane shape gate 错误 |
| v37.1 | `manual-cmos-review-candidate` | 修复 gate 后 proxy positive，只授权人工审查 |
| v38 | `manual-review-required`；人工失败 | true final CMOS 仍为 blob/纹理团块 |
| v39 | `manual-review-failed-final-cmos-blob` | 旧 topology proxy 为 false positive，存在同源/模板 collapse |
| v40 | `label-template-collapse-recovered` | sharp target 恢复类差异，但 final CMOS 仍未过人眼 gate |
| v41 | `decoder-physical-unreachable` | direct target 可读，完整 decoder 输出不可达 |
| v42 | decoder bypass positive | back path/CMOS 能显示数字，传播式 decoder 是首要问题 |
| v43 | `target-scale-bottleneck` | smaller/sharper target 有正信号，padding不是主因 |
| v44 | `target-scale-decoder-recovered-needs-human-review` | small target 有改善，0.50 mm 优于 1.50 mm |
| v45 | `near-field-spacing-recovered-needs-human-review` | 0.25 mm 是当前最佳 direct-decoder spacing |
| v46 | `semantic-collapse-still-high` | longer/semantic/final-plane losses 未拉开类别 |
| v47 | `capacity-still-collapsed` | 当前 phase-amplitude/depth/padding upper bounds 均未救回人眼可读性 |

### 代表性持久证据

- v32 bridge summary：`outputs/reports/twophoton_100x100_v32_bridge_redesign_summary_20260703_150903.json`
- v32.1 sample-gap report：`outputs/reports/twophoton_100x100_v32_1_sample_gap_isolation_20260703_154715.json`
- v33.1 harness-fix summary：`outputs/reports/twophoton_100x100_v33_1_code_conditioned_target_summary_20260703_203645.json`
- v35 code-carrier report：`outputs/reports/twophoton_100x100_v35_optical_code_carrier_isolation_20260703_215703.json`
- v38 formal CMOS review：`outputs/reports/twophoton_100x100_v38_formal_cmos_manual_review_20260706_170459.json`
- v39 false-positive report：`outputs/reports/twophoton_100x100_v39_human_topology_false_positive_20260707_142536.json`
- v42 decoder-bypass summary：`outputs/reports/twophoton_100x100_v42_decoder_bypass_backpath_isolation_summary_20260707_171439.json`
- v45 near-field report：`outputs/reports/twophoton_100x100_v45_decoder_spacing_near_field_recovery_20260709_151115.json`
- v46 semantic report：`outputs/reports/twophoton_100x100_v46_near_field_semantic_separation_20260709_195630.json`
- v47 capacity summary/report：`outputs/reports/twophoton_100x100_v47_near_field_decoder_capacity_upper_bound_summary_20260710_050842.json`、`outputs/reports/twophoton_100x100_v47_near_field_decoder_capacity_upper_bound_20260710_050842.json`

### 当前断点与下一 gate

当前可确认链路状态为：

`code -> generator residual -> SLM command -> target_code -> decoder_input` 已能形成清楚数字；real OLS bridge 曾显示 sample/code distribution 问题，但 v44-v47 为隔离 decoder 而使用 direct target，尚未重新接回完整 real-OLS inference path。当前首要断点是 `decoder_input -> decoder_output`；`physical_camera_plane` 以约 `0.95` correlation 继承 decoder output，结合 v42 bypass positive，camera/readout 不是当前根因。

下一步不应继续普通 loss、padding 或 camera 微调。推荐先做 decoder capacity-utilization audit，明确 amplitude mask 和新增 decoder layer 是否真正被激活；随后执行一个真正利用 amplitude/新增层的 complex-mask 或 non-propagation decoder upper bound。若该 upper bound 通过，主线回到可制造 modulation/传播式 decoder architecture；若仍失败，再转向更大 aperture、pixel size 或传播网格/物理尺度重设计。即使 direct decoder 后续通过，也仍必须单独恢复 real OLS bridge，并在完整单曝光链路上重新进行 true final CMOS 人眼验收。

## 2026-07-11 v48 transparent-decoder 对照

v48 使用同一个 v47 Probe A checkpoint、相同 `small_sharp_stroke_digit_code`、`target_code_spatial_scale=0.72` 和 true `physical_camera_plane`，对比 `0.10 mm` transparent、`0.25 mm` transparent 与 `0.25 mm` trained mask。最初的 `20260710_213539` 和 `20260711_020721` 报告因输入 metadata 不完整而是 `harness-invalid`，不构成模型证据；修复后的 fixed report 无 harness error，并生成十类 CMOS-only review。

人工 verdict 为 `t025_readable_m025_unreadable`，最终状态 `decoder-mask-induced-artifact`：T025 transparent 人眼可读，而同层距 M025 trained mask 不可读。自动指标只用于解释：T025 `decoder_output_vs_target corr=0.9557`、`physical_vs_decoder corr=0.9695`，M025 分别为 `0.8783` 和 `0.9536`。所有结果仍为 `diagnostic-only`，`accepted_summary_written=false`；0.25 mm 低于正式 1.50 mm 全连接几何，不能 promotion。

- fixed report：`outputs/reports/twophoton_100x100_v48_transparent_decoder_control_20260711_020721_fixed.json`
- 人工 verdict：`outputs/reports/twophoton_100x100_v48_transparent_decoder_control_20260711_020721_verdict.json`
- localized issue：trained decoder mask，而不是 0.25 mm transparent propagation 或 camera/back path。
- ruled-out causes：T025 条件下的纯传播、最终 relay/CMOS 无法显示数字。
- still-plausible causes：decoder phase、近 unity amplitude、phase-amplitude interaction 或具体责任层。
- gate validity：fixed/verdict report 无 error，十类 true CMOS 人工审查有效；早期两个 harness-invalid report 无效。
- next gate：v49 无训练 P025/A025/L1P025/L2P025 mask attribution；在人工 verdict 前不得启动 v50。

## 2026-07-11 至 2026-07-14 v49-v57：mask 归因、严格复场与物理 relay 合同

本节记录当前主线从 v48 之后的事实。所有 v49-v57 结果仍为 `diagnostic-only`，`accepted_summary_written=false`。真实最终验收始终只看 `physical_camera_plane` 的 CMOS intensity；复场/振幅/相位诊断、proxy、eval 指标和人工对中间 plane 的判断都不能替代该 gate。

### v49-v50：0.25 mm direct-decoder mask 归因（历史 diagnostic）

- v49 在同一个 v47 Probe A checkpoint、相同 target/seed/0.25 mm 下复用 transparent `T025` 和 trained `M025`，新增 `P025`（仅训练后 phase）及 `A025`（仅训练后 amplitude）。人工 verdict 为 `phase_only_unreadable`，状态为 `phase-mask-induced-artifact`：`T025/A025` 可读，而 `P025/M025` 出现同源横纹/碎裂；`P025–M025` final CMOS correlation 约 `0.9999996`，`A025–T025` 约 `1.0`。当时 amplitude 只形成约 `0.9306` 的近似均匀强度缩放，不能解释主要 artifact。
- v50 增加 `decoder_diagnostic_phase_layer_mask`，分别生成 `L1P025=[true,false]` 与 `L2P025=[false,true]`。该报告仍停在 `manual-review-required`，没有完整的人工 verdict，故不得归因到 L1、L2 或它们的 interaction。
- 以上 0.25 mm/direct-target probe 低于正式 1.50 mm 全连接合同；它们只说明旧合同下 trained decoder phase 曾是 artifact 来源，不能外推到现行正式 relay 路径。

代表证据：`outputs/reports/twophoton_100x100_v49_decoder_mask_component_attribution_20260711_031107_verdict.json`、`outputs/reports/twophoton_100x100_v50_decoder_phase_layer_attribution_20260711_034458.json`。

### v51：严格数字孪生复场可视化修复

- 停止将 SLM command、target code、mask 参数、人工零相位、CMOS intensity 或逐面归一化 display tensor 称为“光学振幅/相位”。历史 v48-v50 的完整 physical-chain amplitude/phase 图被追溯标记为 `visualization-harness-invalid`，历史 run 不改写。
- 正式光场图唯一源自同一传播复场 `E` 的原始 `real/imag`：振幅为 `abs(E)`（a.u.，共享线性色标），相位为 `angle(E)`（固定 `[-pi, pi]` rad）；PNG 只在峰值强度以下 -60 dB 处遮罩相位，未遮罩 raw tensor 仍保存。
- intensity-only relay 之后不再伪造 camera phase。mask transmittance 仅作为 device-mask diagnostic，明确不是传播光场。
- v51 sample 证明了该 artifact/provenance 接口可写入，但它仍是旧光路/旧 checkpoint 下的诊断证据，不能证明现行完整物理链。

代表证据：`outputs/reports/v51_legacy_optical_field_visualization_correction_20260711_102325.json`、`outputs/20260711_v51/runs/20260711_132217_854523_train_v51_real_complex_field_probe_a_s025_sample/`。

### v52-v54：物理坐标合同与旧前端 aperture 失败定位

- v52 建立正式物理 sampling metadata：每个严格复场 plane 必须携带 shape、pitch、FOV、aperture、origin、wavelength、upstream、distance、magnification 与 operation；正式 OLS 固定为复恒等映射 `Eout=Ein` 并通过线性叠加测试。generic complex resize 被正式禁止。
- 当时的 v52/v53 前端曾采用“自由传播后从 500 um FOV 中截取中央 200 um aperture 再得到 encoder sampling”的解释。它的 CUDA preflight 通过的是该旧合同的数值一致性，并不等于其为合适的实际光路模型。
- v53 在该旧合同下从零训练一次 5 epoch：validation loss `0.231066`，camera corr `0.5728`，dice `0.6253`，diversity ratio 约 `2.59e-7`；final CMOS 仍同源、不可读。v54 只读复场诊断曾报告 `macro-front-aperture-first-collapse`，其候选点正是旧 `lightgen_input` 中央截取。这是随后替换前端模型的依据，不是当前 relay 的结论。

代表证据：`outputs/reports/v53_cuda_physical_preflight.json`、`outputs/reports/v54_first_collapse_complex_field.json`。v53/v54 均已被 v55 前端 relay 合同取代，旧 checkpoint 禁止加载到新合同。

### v55-v55.1：正式相干缩小 relay 合同

现行正式理想数字孪生光路为：SLM 完整 `500 um` FOV 经相干 4f 缩小成像到 encoder 完整 `200 um` FOV，禁止中心截取。

| Relay | 对象与目标 | 倍率 | 设计参数 |
| --- | --- | ---: | --- |
| 前端 4f | SLM `200x200 @2.5 um` -> encoder `100x100 @2 um` | `-0.4` | `f1=223.214 mm`，`f2=89.286 mm`，clear aperture `>=23.75 mm` |
| 后端有限共轭 | chip `200 um` -> CMOS `500 um` | `-2.5` | `u=100 mm`，`f=71.4285714 mm`，`v=250 mm`，clear aperture `>=26.60 mm` |

- 前端机械折转距离保持 SLM->BS->M3 `150.000 mm`、M3->L1 `73.214 mm`、L1->L2 `312.500 mm`、L2->encoder `89.286 mm`，总计 `625 mm`；ABCD 为 `A=-0.4, B=0`。
- v55 CUDA structural preflight 通过，确认 sampling/FOV/ABCD/provenance/无 crop 路径可完整 forward：`outputs/reports/v55_coherent_relay_cuda_preflight.json`。
- v55.1 原报告曾标记 reference validation 通过，但其参考实现与生产 B=0 采样同构，现降级为实现自洽证据。v55.2 的自适应 scaled-Fresnel/direct-RS 参考仍因中间面离散采样不足返回 `reference-harness-invalid`。v58.2 的 CUDA 数值自洽运行保留为历史证据：它使用 `128→64`、未实测 Airy PSF，不能作为正式验收。v58.3 才以真实 `A=-0.4, B=0, C=0`、完整 `200→100` sampling、闭式 B=0 场、实测 pupil PSF 和故障注入审计签发 gate；实际尺寸低分辨率 direct-RS 继续排除出 acceptance。pupil/NA 仍是设计下限，不是已实测硬件参数。

### v56-v57：现行 relay 合同的一次从零训练与首塌缩定位

- v56 在当时 v55.1 结构/自洽门通过的配置上只做了一次从零、5 epoch、seed 1234 的 CUDA 训练，复用 readable-split teacher 数据，仅保留 camera reconstruction/distillation objective；没有加载任何 v47/v51/v53 checkpoint，也没有增加 loss、sweep 或 override。v55.2 重新独立验证未通过，因此 v56 继续仅作为 diagnostic 历史。
- 训练 best checkpoint 为 epoch 5：validation loss 从 `0.2715` 降至 `0.2276`，paired corr 从 `0.4913` 升至 `0.5870`，paired dice 从 `0.5704` 升至 `0.6351`，但 diversity ratio 从 `0.00146` 降至 `0.00095`，cross-class ratio 从 `0.000138` 降至 `0.000103`。正式 test eval 为 corr `0.5789`、dice `0.6298`、diversity ratio `0.000968`，strict readability 未通过。
- 同一个 epoch-5 best checkpoint 的 class-balanced sample（0-9、每类两张）具有 `field_chain_status=valid`；真实 `physical_camera_plane` 仍是同源圆斑/纹理，人眼不可读。因此这是有效失败，不是 harness failure。
- v57 新增只读、流式 pairwise 复场诊断。它保留 complex `real/imag`，先为每对样本去掉全局相位，再计算 amplitude、complex 与 intensity 的同类/跨类中位距离分离。初版发现过 complex tensor 被 `.double()` 时丢虚部的问题；已立即修正为 `complex128` 并重新运行，当前报告中的复场/相位指标有效。
- v57 结果没有满足“至少两类分离指标低于 SLM 的 10% 且在后续平面持续”的局部首塌缩门。相对于 SLM 的跨类减同类分离，前端 relay 保留约 `47%`，encoder/OLS/decoder/chip 的 complex 分离仍约 `39%-49%`，camera intensity 约 `11.7%`；OLS 输入输出连续。最终决策为 `aperture-throughput-limited-no-localized-collapse`，不是某一层已证实的语义首塌缩。
- 数值传播每层能量比约 `0.999999`，但有限 metasurface aperture crop 保留率依次为 encoder L1 `25.70%`、encoder L2 `27.65%`、decoder L1 `30.83%`、decoder L2 `33.42%`，累计为 `0.7322%`。这说明有高吞吐损失，但其在类别间近似相同；目前只能列为主导候选，不能直接宣称它是类别塌缩根因。

代表证据：

- train：`outputs/20260714_v56/runs/20260714_204958_697374_train_v56_coherent_relay_from_scratch/`
- eval：`outputs/20260714_v56/runs/20260714_210206_240133_train_v56_coherent_relay_eval/artifacts/one_shot_distill_eval_metrics.json`
- sample/provenance：`outputs/20260714_v56/runs/20260714_210416_032022_train_v56_coherent_relay_sample/`
- v56 preflight：`outputs/reports/v56_coherent_relay_cuda_preflight.json`
- v57 corrected report：`outputs/reports/v57_first_collapse_complex_field.json`

### 当前状态与下一决策门

当前正式合同是：`laser -> M1 -> BS -> amplitude SLM -> BS -> M3 -> coherent 4f relay -> encoder(2) -> complex identity OLS -> decoder(2) -> finite-conjugate relay -> CMOS intensity`。SLM/CMOS 为 `200x200 @2.5 um`、500 um FOV；encoder/decoder 每层为 `100x100 @2 um`、200 um FOV、1.50 mm spacing、phase-only。fiber 尚未设计或实现。

下一轮在 v57 结论前不得继续训练、延长 epoch、增加 diversity/semantic loss，也不得把 aperture/pupil 直接改为新的正式硬件合同。应先设计一个明确标注 `diagnostic-only` 的 aperture-throughput broad-isolation upper bound：在同一 v56 checkpoint、相同输入与相同 formal relay 下，对有限 metasurface crop 的反事实保留进行离线/无训练比较，并同时报告类别分离与能量吞吐。只有它能把类别分离或最终 true CMOS 的人眼信号恢复，才可将 aperture 认定为可操作的根因；否则应转向 upstream generator/objective 的低分离问题。无论结果如何，当前 v56 不得 promotion。

### v58-v59：aperture upper bound 与 upstream separation broad isolation

- v58 使用同一 v56 best checkpoint、同一 class-balanced sample、相同前后 relay、零训练，唯一反事实变化是移除 D2NN 层间 finite crop；训练好的 `100x100` mask 保持居中，外部扩展域为 identity。chip 能量由 `126.18` 增至 `17232.99`，camera field 能量仅由 `66.27` 增至 `69.78`；最终 CMOS separation gain 仅为 amplitude `1.0227x`、complex `1.0173x`、intensity `1.0101x`。人工复核 baseline/upper-bound 都是不可读圆斑/纹理。结论为 `aperture-upper-bound-no-separation-recovery`，不是 harness failure。
- v59 使用同一正式 v56 config/checkpoint，在 `readability_strict_split` validation 数据上做 100 样本（每类 10 张）的三组零训练 forward：paired baseline、label 循环移位、同类 code 循环移位。报告确认 identity OLS、coherent 4f/free-space relay、正式 D2NN geometry 与参数零变化。
- teacher target prototype CV `1.00`、permutation `p=0.00775`；SLM command 与前端 4f 输出 prototype CV 分别 `0.94/0.94`、均显著高于 permutation null。SLM complex separation 仅为 teacher 的 `0.6787%`，但 4f 相对 SLM 仍保留 `41.0%`，因此不是“4f 首塌缩”。SLM 类间频谱约 `80.2%` 落在形式通带外，但前端 separation 没有跌到 SLM 的 `10%` 以下，故不满足 bandwidth mismatch gate。
- label-shift 与 code-rotation 对最终 target MSE 的相对变化分别为 `0.155%` 与 `0.0036%`；最终 true CMOS 仍不可读。当前决策为 `objective-representation-mismatch-candidate`，下一门是 offline decodability upper bound；不得继续 decoder/loss 微调或 promotion v56。

代表证据：

- v58：`outputs/reports/v58_aperture_throughput_upper_bound.json`、`outputs/reports/v58_aperture_throughput_upper_bound.pt`
- v59：`outputs/reports/v59_upstream_separation_isolation.json`、`outputs/reports/v59_upstream_separation_isolation.pt`、`outputs/reports/v59_upstream_separation_isolation_preview.png`

### v60-v61：复制训练、稳健首塌缩审计与离线可解码上界

- v60 使用 v56 相同配置、seed `1234`、5 epoch 和正式物理合同完成复制训练。eval corr/dice 为 `0.5733/0.6235`，sample corr 为 `0.4914`，均未超过 v56；最终 CMOS 仍不可读。v58.3 只加固 relay 验收和 preflight，不改变正式传播算子。
- v61 robust tool 冻结 v60 best checkpoint，对 readable train/val/test `800/100/100` 做完整 formal forward；缓存包含所有严格复场 plane、CMOS intensity、provenance 和 parameter hash。使用固定 32×32 诊断网格、2,000 次 pair bootstrap 与 permutation null；前后参数 hash 一致，报告结论为 `no-robust-localized-collapse`。
- 因此 v60 的 20-sample camera intensity retention `0.0981` 仍是阈值边缘证据，不能推翻 v59 的 100-sample broad-isolation 结论。早期 `tools/diagnose_v61_offline_decodability.py` 是历史软件 readout：它使用降采样/PCA 与额外对照，不能充当下述原生采样上界的替代或正式单曝光读出。
- v61 offline report 的历史 linear plane probe 在 test 上得到 corr `0.874-0.892`、dice `0.676-0.710`，但相对 label-only paired corr/dice delta 为负；code-only/code+label corr 仅 `0.716/0.731`。固定 nonlinear probe 的三个 seed 未达到 strict gate，且部分 seed 输出零多样性。该结果支持“plane 仍有类别模板/低阶信号，但没有证明 paired instance-level CMOS 生成”，不授权继续堆 loss 或 promotion。
- 随后的 v61 native dual-channel upper bound 冻结 v60 epoch-5 checkpoint，以 readable train/val/test `800/100/100` 提取真实 chip/camera fields；无 optimizer、`state_dict` 前后无变化、无 target-code/bridge/override。特征保持 native sampling：chip `100x100 @2 um`，camera/teacher target `200x200 @2.5 um`；逐批验证 `physical_camera_intensity_raw == |E_camera|²`，而 probe 使用真实最终 detector-normalized `physical_camera_plane`。它没有 resize、PCA、神经网络或保存完整复场 bundle。
- 四个严格 held-out linear ridge probes 只用 train 拟合、只由 val 的 raw MSE 从 `{1e-4,1e-3,1e-2,1e-1,1}` 选择、再 train+val 重拟合并仅 test 一次。chip intensity / chip complex / camera intensity / camera complex 的 test `(corr, dice, prototype-accuracy)` 分别为 `(0.9453, 0.8829, 1.00)`、`(0.9456, 0.8835, 1.00)`、`(0.9452, 0.8829, 1.00)`、`(0.9456, 0.8835, 1.00)`；均选 `lambda=0.1`。按预注册门，结论为 `camera-intensity-decodable`。
- 这不是 raw CMOS 人眼 gate 通过，也不是可部署的 optical readout：preview 中的数字来自使用 900 个配对 teacher target 拟合的离线软件映射。complex oracle 相对 intensity 无实质增益，chip/camera intensity 均达标，故本结果不支持仅 phase-to-intensity 或仅 backend relay 失配。它也尚未区分逐实例信息与类别模板；下一门只能做零训练 native conditional-control audit（label-only、code-only、within-class target permutation、逐实例残差），不得调整 decoder/loss 或 promotion。

代表证据：

- `outputs/20260715_v61/runs/robust_collapse/v61_robust_collapse.json`
- `outputs/20260715_v61/runs/robust_collapse/v61_frozen_optical_cache.pt`
- `outputs/20260715_v61/runs/offline_decodability/v61_offline_decodability.json`
- `outputs/reports/v61_offline_decodability_upper_bound.json`
- `outputs/reports/v61_offline_decodability_upper_bound.pt`
- `outputs/reports/v61_offline_decodability_upper_bound_preview.png`
### v58：现行合同去旧化与独立 relay 重验（进行中）

- formal validator 已收紧为当前 v55 coherent 4f 前端、identity OLS、1.50 mm D2NN 和后端薄透镜 B=0 relay；v52/v53 free-space+aperture 配置保留为 legacy diagnostic，不再作为 formal contract。
- 前端 provenance 正在改为具名的 `SLM→BS→M3→L1→L2→Encoder` 距离链。原 v55.1 reference 使用与生产实现同构的插值/相位表达，现降级为实现自洽证据；v55.2 需要独立逐段 Fresnel/Collins 与低分辨率 direct-RS 4f 参考通过后，才能恢复 formal-valid 状态。
- 本轮合同重验不训练、不重复运行 v58 aperture upper bound，也不修改历史 outputs 或 checkpoint。

### v62：native conditional-control audit（2026-07-15）

- 在冻结 v60 epoch-5 checkpoint、正式 v56 config、同一 readable split 和 v58.3 relay gate 上运行一次零训练 CUDA audit。train/val/test 为 `800/100/100`，参数 hash 前后相同，`optimizer_created=false`，所有结果仍为 `diagnostic-only`。
- 原生 forward 对照包括 baseline、固定 readable-train code 的 label-only、固定 label `0` 的 code-only；target 对照为每类一位确定性循环置换。v61 已封存的 camera-intensity held-out ridge prediction 直接复用，未重新拟合。
- label-only prototype CV accuracy `1.00`、permutation `p=0.00775`，类内 residual energy `3.14e-15`；baseline accuracy `0.94`，code-only accuracy `0.23`（同样 `p=0.00775`），但 code-only 类内 residual energy 仅 `5.09e-5`。baseline→label-only camera response `1.040%`，baseline→fixed-label code-only `0.389%`。
- v61 offline camera-intensity prediction 对 paired teacher target 的 corr 为 `0.9452`，within-class target rotation 后 `0.8909`，paired delta `0.0544`（95% CI `[0.0469,0.0628]`）。leave-one-out class-template residual corr 为 `0.4931`，旋转 target 后 `-0.0890`，paired residual delta `0.5821`（95% CI `[0.5321,0.6325]`）。因此上界包含逐实例 residual，但仍不是 raw CMOS 人眼 gate。
- 当时决策写为 `paired-instance-residual-with-weak-explicit-code-control` / `manual-review-required`。该解释已被后续 v63 修正：code-only classifier 用原始 readable labels 分组，readable filter 对 IID code 造成 code-label selection confound；且 `baseline→label-only` 是替换 code、保持 label 的响应。因此 `0.23` classifier 和固定 `2%` 门不能用作“code 弱”或根因结论。已封存 paired-residual 对照仍是有效的关联证据，但不授权修改 generator、decoder、loss 或硬件合同。

代表证据：

- 工具：`tools/diagnose_v62_native_conditional_control_audit.py`
- 报告：`outputs/reports/v62_native_conditional_control_audit.json`
- tensor：`outputs/reports/v62_native_conditional_control_audit.pt`
- preview：`outputs/reports/v62_native_conditional_control_audit_preview.png`

### v63：code-conditioning provenance 与 crossed-factor audit（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay、同一 readable `800/100/100` split 与 v61/v62 工件，执行一次零训练 CUDA audit。identity OLS、正式 geometry/field chain、camera intensity 语义、参数快照、code replay 与 teacher-noise replay 全部有效；`parameter_update_count=0`、`optimizer_created=false`、`diagnostic_only=true`。
- 全局 `code_seed=1234` 的 15k IID 导出 code prototype CV 为 `0.09713`，相对于 permutation null 的 `p=0.94118`。按 readability filter 选择后的 train/val/test 分别为 `0.3225/0.1700/0.2200`，对应 p 值 `0.00775/0.02326/0.00775`。这是筛选造成的 code-label selection confound，不能当作模型 code-conditioning 能力或训练许可证。
- 在 100 held-out test code × 10 labels 的平衡网格上，SLM command 的 code/label/interaction 方差份额为 `80.62%/16.07%/3.31%`；pre-encoder intensity 为 `80.51%/16.13%/3.35%`；chip intensity 为 `79.52%/17.71%/2.77%`；最终 detector-normalized CMOS 为 `80.39%/15.55%/4.06%`。因此正式光路没有把 code 因子衰减到 SLM 的 10% 以下，且 label 因子没有形成局部保留而 code 丢失的模式。
- 预注册门给出 `generator-code-dominant` 和 `native-code-conditioned-subvisual-representation-confirmed`：generator 的 code 因子很强，并保持到 final CMOS，且 v62 paired-residual 对照仍有效。它不改变 raw final CMOS 人工不可读的硬 gate；结论是原生 code-conditioned representation 为 subvisual，下一门只能进行 representation/readability isolation，不得重新微调 decoder/loss、修改 optical contract、训练或 promotion checkpoint。

代表证据：

- 工具：`tools/diagnose_v63_code_conditioning_provenance_audit.py`
- 报告：`outputs/reports/v63_code_conditioning_provenance_audit.json`
- tensor：`outputs/reports/v63_code_conditioning_provenance_audit.pt`
- preview：`outputs/reports/v63_code_conditioning_provenance_audit_preview.png`

### v64：semantic-band carriage 与 readability isolation（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay、同一 readable `800/100/100` split 及 v61/v63 工件，执行零训练 CUDA audit。15k code/noise provenance、identity OLS、formal geometry/field chain、camera `|E|²`/detector normalization 和参数快照均通过；`parameter_update_count=0`、`optimizer_created=false`，临时 native memmap 在诊断后删除。
- 以 camera `200x200 @2.5 µm` 的 native DFT lattice 定义 `0–20 cycles/mm` semantic、`20–80` mid、`>80` high。train teacher target 在 semantic band 的中心化能量份额为 `95.96%`，通过预注册 `90%` cutoff；v63 camera code-main 频带份额为 `20.16%/67.47%/12.37%`，label-main 为 `18.24%/69.56%/12.20%`。
- 独立 held-out band probes（raw v61 probe 保持冻结）得到：semantic `0–20` corr/dice `0.9421/0.8782`，paired rotation drop `0.0547`、95% CI `[0.0474,0.0626]`；non-semantic `>20` corr/dice `0.9454/0.8832`，drop `0.0545`、CI `[0.0469,0.0625]`；冻结 v61 raw reference 为 `0.9452/0.8829`。两路均保留 paired held-out information。
- 因此 v64 决策为 `distributed-subvisual-carrier` / `manual-review-required`：paired representation 同时分布于 semantic 与 non-semantic 频带，不存在单一主载波。raw CMOS 人眼不可读的硬 gate 不变；v64 不授权 detector/display calibration、generator/loss/optical-contract 修改、训练或 promotion。下一门只能做跨频带表示归因与固定显示/人工审查关联。
- v64 preview 人工复核显示 teacher 行可读，raw CMOS 仍是同源颗粒纹理，semantic 行接近相同低频模板，non-semantic 行主要是颗粒纹理；因此最终 raw CMOS 人眼 gate 仍失败。

代表证据：

- 工具：`tools/diagnose_v64_semantic_band_carriage_audit.py`
- 报告：`outputs/reports/v64_semantic_band_carriage_audit.json`
- tensor：`outputs/reports/v64_semantic_band_carriage_audit.pt`
- preview：`outputs/reports/v64_semantic_band_carriage_audit_preview.png`
### v65：common-mode 与 readout-locality attribution audit（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 config、v58.3 relay、readable `800/100/100` split 及 v61/v63/v64 工件，完成一次零训练 CUDA audit。provenance、identity OLS、formal field-chain、camera `|E|²`/detector normalization 与参数快照通过；`parameter_update_count=0`、`optimizer_created=false`，临时 native memmap 已删除，结果均为 `diagnostic-only`。
- train-only common mean/gain 的参数为 RMS `0.230195`、residual abs q995 `0.0071021`、gain `63.3615`。固定 common-mode display 的 test corr/dice/prototype accuracy 为 `0.04295/0.33365/0.46`，人工 preview 仍不可读。
- 冻结 v61 raw reference 为 corr/dice `0.94523/0.88293`。pointwise affine 为 `0.68988/0.68732`；3x3、7x7、15x15、31x31 shared local convolution 分别为 `0.62136/0.64443`、`0.65647/0.67172`、`0.66580/0.67585`、`0.66830/0.67637`；full-support Fourier/Wiener 为 `0.68419/0.68491`。所有 target-fitted observer 的 within-class rotation CI 下界不为正，均未达到 v61 raw corr/dice 的 90% paired retention。
- 预注册决策为 `nonlocal-codebook-objective-representation-mismatch-confirmed`，因为 fixed display、pointwise、全部 `<=31x31` local kernel 与 Fourier/Wiener 均失败，而冻结 v61 dense reference 仍可解码。raw CMOS 人眼 gate 仍失败；该结果只授权架构讨论，不授权训练、改 loss/decoder/generator/D2NN/OLS/relay/aperture 或 promotion。下一门应在不改变 formal path 的前提下讨论 nonlocal codebook/objective 设计与可读性约束，不直接启动训练。

代表工件：

- `tools/diagnose_v65_common_mode_readout_locality_audit.py`
- `outputs/reports/v65_common_mode_readout_locality_audit.json`
- `outputs/reports/v65_common_mode_readout_locality_audit.pt`
- `outputs/reports/v65_common_mode_readout_locality_audit_preview.png`

### v66：semantic-gradient budget audit（CUDA 已完成，diagnostic-only）

- 新增只读 `tools/diagnose_v66_semantic_gradient_budget_audit.py` 与 `tests/test_v66_semantic_gradient_budget_audit.py`。工具固定 v60 epoch-5、v56 formal config、v58.3 relay、readable `800/100/100` 及 v61–v65 工件；使用正式未 clamp camera reconstruction residual，对 generator、encoder、decoder 参数块进行逐样本 VJP 与流式 cancellation/cosine 统计。
- train-only global/label templates、common/label/instance residual 精确分解、foreground/background 权重、参数零变化和四个决策分支均已覆盖。工具不创建训练对象、不写入参数梯度、不保存完整复场或梯度。
- 已按预注册输入完成一次 CUDA audit：`parameter_update_count=0`、`optimizer_created=false`，报告、tensor 与 preview 均已生成。结果为 `mixed-semantic-gradient-evidence`：label/instance drive 未同时低于 suppression gate，且未满足 semantic-gradient-available 或 batch-cancellation gate；它不授权继续 v56/v60 同配置训练，只保留为透明薄相位 diagnostic 的历史前置证据。
- 实现验证：v57–v66 定向 `58 passed`；完整仓库 `522 passed`、`148 subtests passed`；compileall、ruff、`git diff --check` 通过。

## 2026-07-16 至 2026-07-22 v67–v79：透明诊断、code transport 上界与冻结结论

以下记录均为历史 `diagnostic-only` 证据，不改变 stable formal contract，也不构成 student promotion。需要逐次查看原始交接正文时，使用 [v68–v79 原始主线交接归档](../archive/current_handoff_v68_v79_archive.md)；本节保留面向检索的浓缩索引。

### v67–v67.4：透明薄相位与全场传播数值门

v67 引入 `transparent_identity` 作为 Formal 明确拒绝的独立诊断边界。v67.1 的 ASM 零相位和 backend relay reference 通过，但冻结相位在 `1024/1536` 网格的 chip complex L2 为 `0.02107/0.005816`，未满足 L2≤`1e-3` 与 guard≤`1e-4`，状态为 `grid-not-converged`（`outputs/20260716_v671/runs/preflight/v67_transparent_convergence_v2.json`）。

v67.2/v67.3 修正 padded propagation、FOV、共同 ROI 与 phase gauge 后，候选 `1536/3072` 对 reference 的 zero/small-init gauge-aligned complex L2 仍约 `0.001408/0.001412`，首失败为 `decoder_layer_1`；没有进入梯度、显存或训练门（`outputs/20260718_v673/runs/preflight/v67_transparent_convergence_v4.json`）。v67.4 的全场算子隔离继续为 periodic stress evidence；它不修改 `finite_crop`、v56/v60 checkpoint 或默认 YAML。

### v68–v69：prototype 通过不等于 code generation

v68 A2 的 locked raw audit 达到 clean corr/dice `0.9910/0.9548`、prototype accuracy `1.0`、high saturation `0.0032`，十类 raw-linear 人审通过；其后 v68.6 B 的 clean corr/dice 约 `0.9951/0.9676`、accuracy `1.0`，但 raw-camera code fraction 仅 `7.14e-08 < 0.25`，有效决定为 `b-prototype-imaging-only`。工件位于 `outputs/20260718_v685/runs/` 与 `outputs/20260719_v686/runs/`。

v68.7 证明 generator injection 是首个退化面：residual map factor `0.74447`，SLM 仅 `8.25e-06`，即使 tanh 上界也仅 `0.003595 < 0.25`。v69 的 paired teacher-camera 合同将 raw-camera factor 提升到 `0.0002712`，仍未达门；paired-teacher advantage 的 bootstrap 95% 下界为 `-0.00559`。状态 `v69-prototype-only`，冻结 checkpoint，禁止重训、sweep、test、导出与 promotion（`outputs/20260720_v69/runs/`）。

### v70–v77：冻结 D2NN 的传输归因

v70 将 SLM command intervention 校准到 `0.25000003`，但 camera code factor 降至 `0.19396`；v71 改在 physical `slm_amplitude²` 校准后，camera 为 `0.22021`，首个有效衰减面为 `encoder_layer_2`。v72 的 identity ablation 可提高 code factor（joint 为 `0.29503`），但破坏数字可读性，结论是 `phase-readability-code-tradeoff`。代表工件：`outputs/20260720_v70/runs/isolation/v70_slm_transport_isolation_v1.json`、`outputs/20260720_v71/runs/isolation/v71_slm_intensity_transport_v1.json`、`outputs/20260720_v72/runs/isolation/v72_d2nn_phase_attribution_v1.json`。

v73 semantic-support intervention 保持 prototype readability，却使 camera factor 为 `0.16961`；v74/v75 进一步确认主要机制是冻结 D2NN 的 carrier/residual phase-to-intensity redistribution，而不是 full-state exterior escape。v76 双正交读出虽得到 joint factor `0.740539`，但端口饱和 `0.156618 > 0.08`，无法作为部署上界。v77 只证明局部 phase gradient compatibility，不授权训练。上述结论及工件位于 `outputs/20260720_v73/` 至 `outputs/20260720_v77/`。

### v78–v79：八维 supported-basis 家族关闭

v78 的 semantic-gated direct-command CPU admission 虽通过 provenance 与有限性，但 physical-SLM code factor `0.00049967 < 0.25`、support energy `0.94566 < 0.95`，状态为 `v78-command-code-factor-insufficient`。v78.1 定位到 template support 与逐样本 normalization 的 code suppression；它关闭 template multiplication 加逐样本 normalization 合同（`outputs/20260721_v781/runs/isolation/v781_semantic_command_bottleneck_v2.json`）。

v79 使用固定 label-only normalizer，对 frozen learned replay 与 deterministic capacity arm 作 generator-only CPU 上界。前者 factor 为 `6.03e-7`、absolute code SS `0.496`；后者 factor 为 `0.05455`、absolute code SS `48292.58`，均低于 `0.25` 和 v73 的 physical-SLM absolute baseline `289721.29`。虽 support energy `0.97080`、低频能量 `0.99814` 通过，但 clipping `0.20547 > 0.08`。有效决定为 `label-fixed-supported-basis-insufficient`，冻结该八维 basis 家族及 residual＋phase 联合训练；工件为 `outputs/20260721_v79/runs/admission/v79_label_fixed_supported_basis_v1.json` 与 `v79_joint_decision_v1.json`。
## 2026-07-27 至 2026-07-28：v2 v5 多锚点容量准入失败

`conditional_multi_anchor_local_modes_v5` 只完成了代码实现和无更新容量审计，没有启动optimizer、CUDA训练或数据集导出。固定55k MNIST训练子集上的四簇特征审计得到平均类内离散度降低 `0.1691328704`、最低类别降低 `0.1243160963`，均低于预注册的 `0.30`/`0.15` 门。数字1的四个簇大小为 `2027/1855/156/2204`，最小簇也低于256张要求。证据保存在 `outputs/20260727_v2/runs/20260727_211242_776413_formal_v2_multi_anchor_local_modes_finite_crop_identity/`，原始evidence保持只读。

该结果否定的是每类四个等权硬锚点和象限路由的容量假设，不否定完整 `label+code → SLM → Encoder → identity OLS → Decoder → CMOS` 宏观光路。特别是数字1的小簇若被强制等概率路由，会把约2.5%的真实形态放大到25%，因此不能通过放宽最小簇或离散度门继续训练。下一动作是diagnostic-only的连续PCA模式容量审计；在该审计通过前不得派生v5 checkpoint或启动任何smoke。

## 2026-07-28：连续PCA模式容量数据门通过，物理probe等待CUDA

`audit conditional-continuous-mode-capacity` 使用当前内容绑定evaluator、55k训练split和语义checkpoint完成无更新数据审计。PCA-8相对v5 K=4基线的平均优势为 `0.4269000490`、最低类别优势为 `0.3052000363`，振幅PCA-8最小解释收益为 `0.4731594795`，十类物理basis均为rank 8。数据门通过，说明连续类内模式比四锚点等权硬路由更符合当前MNIST类内几何。

运行目录为 `outputs/20260728_v2/runs/20260728_180501_490139_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。后续物理probe未因指标失败，而是因当前环境 `CUDA unavailable` 被标记为 `harness_invalid`；未创建optimizer、未修改参数、未生成checkpoint。恢复时必须在CUDA可用环境中使用相同配置、evaluator、split和语义checkpoint重跑无更新probe，不能直接进入训练。

## 2026-07-29：连续模式旧CUDA probe合同不完整

20260729运行目录为 `outputs/20260729_v2/runs/20260729_104105_417380_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。完整光路、camera code fraction和saturation通过，但语义结果为 axis `0.8235`、uniform-1234 `0.5824`、uniform-5678 `0.5706`，类别5/7 recall最低；因此不满足原定语义门。该版本还直接使用v3的gain `0.5`，没有执行预注册安全logit RMS、逐方向响应、梯度和stateless切空间门。

旧evidence标记为 `probe_gain_contract_mismatch / harness_contract_incomplete`，不修改历史JSON，不消费其准入动作。新合同固定 `alpha=0.05/tanh(1)`，新增 `conditional-continuous-mode-reachability` 只读审计；在新审计通过前不得派生连续模式checkpoint或训练。

实现状态：新审计已固定旧basis hash，输出三个完整 `10×17` contact sheet，并分别对
class carrier、continuous PCA modes、Encoder phase、Decoder phase 做无状态可逆单步
margin/distribution tangent probe；电子扰动限制 SLM-logit RMS `0.05`，相位扰动限制
`0.005 rad`，raw-CMOS目标变化 `1%`。该改动只增加诊断代码和测试，不修改旧run/evidence，
也不产生checkpoint。由于当前执行环境 CUDA 不可用，必须由用户在 CUDA 环境运行新命令后，才能
判断 `continuous_semantic_alignment_reachable`、objective conflict 或 harness invalid。

## 2026-07-29：连续 PCA 安全尺度 CUDA 可达性审计失败

WSL CUDA 动态链接路径修复后，RTX 3060 Laptop GPU 完成了新的只读审计：
`outputs/20260729_v2/runs/20260729_131120_150834_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。
实现同时修复了诊断本身的内存合同：170样本反向图改为每个类别17样本即时消费与梯度累计，以适配6 GB显存；
没有 optimizer、checkpoint 或参数更新。

安全尺度准确达到 axis SLM-logit RMS `0.0500`，并保留高语义 accuracy
`0.9824/0.9529/0.9529` 和低饱和约 `0.0076`。但 raw CMOS code fraction 仅
`0.0071/0.0212/0.0240`，物理 `A²` fraction 仅 `0.0107/0.0323/0.0352`，均未达到 `0.10`。
结果为 `failed / safe_scale_transport_insufficient`，无下游授权。这是有效物理传输停止证据：不能恢复
gain `0.5`、不能做 gain sweep、不能派生连续PCA正式checkpoint或训练；若继续，只能审计带 PCA
特征值尺度的物理模式参数化。

## 2026-07-29：v6.2 短层距自成像恢复合同

v6 smoke 与局部宏观审计失败后，保留完整宏观光路但建立独立 diagnostic 合同：层距
`1.50 mm → 0.25 mm`，2+2 相位层、relay、identity OLS、finite_crop、曝光和网格不变。
新增 `conditional_pca_self_imaging_transport_v6_2`、固定全局 `κA²` target 和同 code
delta transport loss。旧 v6 checkpoint/evidence 不迁移；执行顺序为 target admission →
optical admission → observability → 唯一 400-step smoke。正式训练与数据集闭环在 smoke
通过前继续禁止。

## 2026-07-29：v6.2.1 合同修复后短层距 retention 失败

首次v6.2 target admission的全局scale/0.20上限/错误退出码被归为无效harness。改为每类每方向
`(0.05,0.50)`确定性标定后，电子target通过三块面板的语义、fraction和方向响应。随后0.25 mm
零phase CUDA admission计算冻结 `κ=0.997967`，但Encoder layer 2相对1.50 mm的retention为
axis `+27.4%`、uniform `+16.0%/+14.3%`，低于每面板`+25%`门。其饱和字段后来发现是在显示
clamp后计算，故历史 evidence 标为 `invalid_saturation_harness` 并需一次pre-clamp重跑；短层距
训练和导出始终未获授权。0.25 mm也不满足实验台1.50 mm full-connect几何下界，后续只作为单片堆叠
器件诊断保留。

## 2026-07-30：v6.4 replay关闭与v7扩展流形准入失败

只读CUDA replay补齐v6.4的ratio、delta correlation、crop和最后50步趋势门；harness有效但
feature ratio、类别7 recall和distribution trend失败。唯一phase-only distribution tangent在
phase RMS `<=0.005`、raw RMS `<=1%`内分别测试encoder、decoder和joint，三者均不能让三面板
最差feature ratio提升`0.01`且保持保护门，故关闭该路线。

v7保持near-conjugate光路，使用32个20 cycles/mm以下的数据对齐PCA模式和无bias类条件MLP。
PCA-32相对PCA-8的容量门通过；固定CUDA admission失败于axis类别8 recall `0.8235`、axis/
uniform-1234/holdout feature ratio `0.4231/0.5223/0.5419`及防模板坍缩。无optimizer、checkpoint、
smoke或训练被创建；结果为 `nonlinear_expanded_manifold_not_admitted`。

## 2026-08-24：H-CCF prior cycle composition factorial closeout

H-DLTA 的有效分类为 `dense_local_attribution_underdetermined`，因此注册独立的
H-CCF（Prior Cycle Composition Factorial）无更新诊断。实现提交为 `dcd9528`，
direction/source consistency gate 修复为 `629e1aa`，reconstruction locator 补齐为
`2d5a288`，只读
绑定 WAE seed-7301、H-SWAE/H-SCED seed-8601 final/screen/closeout 和 H-DLTA retry7
audit/report/closeout；不创建 optimizer、不加载 evaluator、不生成候选 checkpoint。

首次 preflight 发现 chunk-128/257 radial reference 的约 `2.6e-7` 浮点差超出原等价门，
按 fail-closed 保留为 `invalid_harness`，未进入 audit；修复提交为 `c306f46`，从新
clean HEAD 重建 validate→preflight lineage。最终 retry3 validation、preflight、audit、
report、closeout 的 immutable SHA256 为 `9163c5…a714`、`d605fa…2e2e`、
`559af9…b476`、`30e629…107b`、`65eaa0…c342`。

最终 audit valid 且通过所有 harness 门。历史 validation bank 的 source/final cycle
复现为 SWAE `0.01649122442/0.02642731884`、SCED `0.01649122442/0.02250728432`；
factorial identity、四个 stability banks、source/source null swap、chunk equivalence、
primary/MSE direction consistency、reconstruction/image locator、finite、hash isolation 和
resource gate 均通过。SWAE
total drift 为 `0.00993609441`，SCED 为 `0.00601605990`；两条路线的 encoder、decoder、
interaction 与 mixed component 均未达到稳定分类门，最终分类为
`cycle_composition_attribution_underdetermined`，next action 为
`close_prior_native_attribution_family`。peak allocation `147424768` bytes、storage
`416895` bytes、optimizer_created=false、evaluator_calls=0；report→closeout 为纯消费者。

该结果关闭 prior-native attribution family。不得据此启动 encoder geometry、decoder
invertibility、interface Jacobian、hybrid mediation、projected-gradient、SGD、trust-region、
adapter、confirm、其他 seed、O0-G、SMF/FMF/MMF、E1、光学、fiber training 或 formal
promotion；任何继续工作必须另行注册无更新假设并建立新的 manifest lineage。

## 2026-08-24：电子 O0 综合收口与 H-DPGC 注册

WAE、Flow、mixture、H-SWAE、H-SCED、H-PGC、H-TCA、H-DLTA 和 H-CCF 的可比证据已整理至
`docs/status/electronic_o0_research_synthesis.md`。语义、先验、cycle、记忆化和治理有效性
分别保留，不形成不可解释的总分。WAE 的主要失败轴是双 evaluator 语义覆盖和类别召回；
后续 prior-native 目标与归因路线均未形成可消费的通过结果。

因此固定 `prior_native_attribution_family_closed=true`，所有既有 checkpoint 保持只读且
不得 promotion。下一注册假设为 H-DPGC（Direct Prior Generator Capacity Canary）：只
测试一个固定的、类条件 rank-8 logit-Gaussian 直接生成 canary，保持
`label + 8D N(0,I)` 和单次推理接口；不创建 optimizer、不更新模型、不保存候选
checkpoint。H-DPGC 的 valid pass 只允许注册未来架构 preflight，valid failure 只关闭该
线性 canary，均不能授权神经训练、confirm、O0-G、光学或 fiber 路线。

## 2026-08-24：H-DPGC direct-prior capacity canary closeout

H-DPGC 实现提交为 `9d4c73c`；三次设备边界 harness 修复分别提交为
`1b5f1ac`（coverage CPU/CUDA）、`46b1ebb`（camera-code labels）和 `43e008c`
（memorization reference distances）。早期 preflight 尝试均在完整 measurement 前异常退出，
按 invalid harness 保留且未生成科学结论；retry3 从新 clean validation lineage 开始。

retry3 validation、preflight、audit、report、closeout 的 immutable SHA256 依次为
`af09945e6d085891c6113e19a5330ad480b3ee589e13b99d70fc2a4f1a354ef4`、
`890ad59a26f0c4f3e76099270d38d17da01c60940d4525dbcde8f1d2927a8818`、
`7fc08c89c8d86f31b93873c742496f9e219365980a022b3aea1b0db5eecf501`、
`e3d5737c01886854a65a4d732bc3dba143e13914641313134c10b537bbcbe58b` 和
`e61e40c18a6f09a272a16414ae369699356cd7ca6fc56e666d4f5735b121d075`。

preflight harness 通过，peak allocation `517880320` bytes、storage `24479` bytes；
audit peak allocation 同样为 `517880320` bytes、storage `41238` bytes，optimizer-created
为 false。audit/closeout 为 `status=valid, measurement_valid=true, passed=false`，分类
`direct_prior_linear_semantic_capacity_failed`，next action
`close_direct_prior_linear_capacity_canary`。三个 panel primary minimum recall 为
`0.76/0.80/0.92`，secondary minimum recall 为 `0.90/0.96/0.96`；Jacobian rank pass
fraction `1.00`，exact duplicates 为零，但 memorization pass fraction 为
`0.914/0.932/0.908`。official 10k primary/secondary accuracy 为 `0.9544/0.9761`，
primary minimum recall `0.8768`，agreement `0.9470`。

该 valid failure 只关闭固定线性 canary，不证明所有非线性 direct generator 不可行；在
新的无训练容量证据之前，不得注册或启动 neural direct-generator training。confirm、
O0-G、光学、fiber 和 formal promotion 继续禁止。

## 2026-08-24：H-DIDM direct image distribution matching 注册

H-DPGC 的 valid failure 仅否定固定线性 rank-8 logit-Gaussian canary，未否定现有 nonlinear
decoder 在直接分布训练下的可行性。注册 H-DIDM（Direct Image Distribution Matching）作为
下一条独立 electronic O0 训练结构假设：WAE seed-7301 decoder 只读 warm-start，删除
encoder、reconstruction、latent alignment 和 cycle，保持 `label + 8D N(0,I)` 单次生成
接口，并以 raw/pyramid/Sobel 多尺度 classwise sliced-energy U-statistic 对齐真实图像分布。

固定 seeds 为 harness `8781`、future screen `8801`、balanced batches `11/23/37/51`、
prior `1011/1023/1037/1051`、projection banks `271829/314159`。正式顺序为
`validate → audit → preflight`；audit 不创建 optimizer 或 evaluator，只能授权三步
preflight。preflight 通过后才准备 screen manifest，本注册不授权长 screen、confirm、
O0-G、光学、fiber 或 formal promotion。

## 2026-08-24：H-DIDM retry3 harness evidence 废弃

H-DIDM 实现提交为 `db391ef`；保护门修复提交为 `85c155f`、`e78951d` 和 `66d974f`。
修复后的 retry3 从 clean HEAD `66d974f` 重建 lineage，旧尝试保留但不复用。validation、
audit、preflight evidence SHA256 分别为
`5b0f35aea831cebbece92ccd22a20607ab29978d0c5f9a8fc1c4497475540543`、
`4a3f350bc79a1a80f9464a1a3dd5552fe5dd53a8bf203163d5968554c20b58a3` 和
`c88c7b2911ea209a76bee10421023a359f0f3e97d59413a6d1df81a39ee72cd4`。后审发现 audit
未绑定 collapse/blur/label-shuffle 与 chunk/formula controls；preflight 的恢复点在
step 3 后没有重放更新。因此 retry3 全部标记为 `superseded/invalid_harness`，不形成
科学结论，也不授权 screen。

审查只保留上述 hash 作为不可变历史记录，不消费其 direction-admitted 字段。

下一动作是修复 controls、真实 step-2→step-3 replay、625-step screen/checkpoint contract，
然后从新 clean commit 重建 validate → audit → preflight；在此之前不得启动 seed-8801。

这只是 direct image-distribution 目标的局部方向与恢复性证据，不是 semantic screen 或模型
准入。seed-8801 screen manifest 已生成并通过只读 hash/lineage 校验，但本轮不启动 screen；
不得延长训练、调权重、换 seed、confirm、O0-G、光学、fiber 或 formal promotion。

## 2026-08-24：H-DIDM retry4 lifecycle 与 seed-8801 screen closeout

retry3 的 audit/preflight 因缺少 collapse/blur/label-shuffle、chunk/formula controls 以及
step-2→step-3 真实恢复更新而被标记 `superseded/invalid_harness`，未形成科学结论。修复
提交 `727568b` 后，从新 clean HEAD 建立 retry4 `validate → audit → preflight → screen`
lineage；旧 artifacts 与 evidence 保持只读。retry4 validation、audit、preflight、screen、
report、closeout 的 immutable SHA256 依次为：

`1c38fcebe218823246ec9f253769bf95f15049ce0657d7faaffd6992ea333862`、
`c76d69c974ca6ed89ef19935c358688495b3c4234c91f1ae6a8e6f9ea3303b99`、
`338e150c10c0cd8f1b8d20eccf6848900be28ea59080cbe838af51425f1af363`、
`f2ccc0cb7ce72c8e6866bac5fa48be3d873cd636ecc52e47f2eb31374041d369`、
`7e1626c98a089f4db632d0170bcb9f430cdad6e4f549ec45083c9d3ce2c7afa9`、
`d65a36cdc7637f76799c5bd763bf2ea51f1973bea55cd55ed7f6845bbb07fab3`。

Audit 通过 real-vs-real matched、class-mean collapse、blur、label-shuffle、朴素 U-statistic
逐值/逐梯度、chunk invariance 和 projection repeat controls；四个 batch 的两个 projection
bank 上 `-g` 方向均通过，`+g` 负对照不通过，optimizer/evaluator calls 均为零。Preflight
使用 seed-8801 正式 schedule，step 2 checkpoint 恢复后真实执行 step 3；decoder、optimizer、
CPU/CUDA RNG、sampler、schedule 与 partial history exact，两个 observer bank 均下降，peak
allocation `1515885568` bytes、storage `64606` bytes。

唯一授权的 seed-8801 screen 执行 625 steps，训练期间 evaluator calls 为零，training
checkpoint 为 `electronic_o0_direct_image_distribution_training.v1`。两 projection bank
aggregate distribution loss 均低于 source（`-0.0126091`、`-0.0113928` vs source
`0.0142719`、`0.0145378`），且每类不退化；但 locked panels `4242/104729/161803` 的
primary accuracy 为 `0.918/0.910/0.894`、minimum recall 为 `0.72/0.78/0.78`，secondary
accuracy 为 `0.898/0.896/0.870`、minimum recall 为 `0.80/0.80/0.74`，双 evaluator
panel pass count 为 `0/0`。memorization pass fraction `0.998687`、exact duplicates `0`、
Jacobian rank pass fraction `1.0`，camera-code/diversity/saturation/coverage/label-shuffle
保护门通过；official 10k 仅报告不参与选择。screen 为 valid scientific failure，分类
`direct_image_distribution_failed`，next action `close_current_direct_image_distribution_route`。

screen decoder/training checkpoint SHA256 为
`c066349955608e2c170f7732a7ecc63be4fb97f6673271373016544b10b5b6f2` /
`df58662371220c5305e48b65ea71f51378f8637860fcc2e7ead3172f4fa72f6a`；screen peak allocation
`646068224` bytes、storage `18826955` bytes。report→closeout 均为纯消费者，H-DIDM 路线
现已关闭，不得重跑、调参、换 seed、confirm 或设计 decoder v2；后续如有必要只能另行
提出无训练容量证据并建立新 lineage。

## 2026-08-25：H-SCF semantic code-source factorial closeout

`electronic_o0_semantic_code_factorial_v1` 完成 retry4 的
`validate → preflight → audit → report → closeout`。此前三次 clean retry 的 harness
失败均保留在各自 `failed-runs/`：canonical MNIST train-count 假设、decoder/camera label
device 边界、64/257 `torch.cdist` bitwise 差异和 permutation finite 结构均已最小修复并
重新验证；未覆盖或改写失败 evidence。最终 lineage commit 为 `54771ff`。

retry4 五个 evidence 全部 `status=valid`，最终分类为
`semantic_counterfactual_safety_failed`，唯一 next action 为
`close_frozen_decoder_derived_semantic_route`。preflight peak allocation 为 `720658432`
bytes、wall time `9.480835378999473` seconds、storage `10316` bytes；audit peak allocation
为 `720658432` bytes、storage `110601` bytes。历史三 panel view hashes 与双 evaluator
correct/recall/agreement counts、posterior hashes、finite、chunk、参数不变和 no-optimizer
门均通过。

四 arm 结果按每 bank `primary correct / secondary correct / agreement count` 记录为：S-G
`2330/2115/2117`, `2320/2129/2121`（两 bank safety 均失败）；S-P `2466/2463/2454`,
`2459/2462/2442`（semantic 均通过但 safety 均失败）；D-G `2270/2244/2303`,
`2290/2240/2308`（semantic 均失败）；D-P `2311/2321/2364`, `2305/2329/2381`
（semantic 均失败，两个 within-class permutation controls 也失败）。分类按 safety 优先级
fail-closed，不把后续 D-P 结果提升为唯一 decoder 或 pairing 结论。该 valid scientific
failure 不授权 nonlinear canary、训练、confirm、decoder v2、O0-G、光学、fiber training
或 formal promotion。

最终 evidence SHA256：validation
`252b0ebf226859e618c3fdb342a69e80269098bc09d1deebe4b411378d1ab491`；preflight
`25fd79f3b1194299c32f7c0e963849523caf3d4f3cfb596c461bb9a72da05104`；audit
`aca75439aa2fce3f5414f697f34505e91bcc0052c9532aee8ff8b8e24eaef930`；report
`4a60eda420884d7a19d30aa6712482de4f4394aec6cac7e9c8035b3f6c0da9f6`；closeout
`59d927f7d7ed3b9633fe613e5b9ce14814ba9b5e15776e9ef47352594026ef94`。

## 2026-08-25：H-NDMC registered/unexecuted

H-NDMC（Nonlinear Direct Manifold Capacity）登记为下一条独立的无更新容量
诊断。它固定 `label + 8D N(0,I) → image` 接口，在 MNIST training split 上
使用 exact classwise PCA-8、稳定 Gaussian copula 和全同类训练行的连续
Gaussian-kernel reconstruction，配套 H-DPGC historical replay、H-SCF
closed-route binding、双 evaluator、安全门、label-shift、zero-code、chunk、
Jacobian、curvature 和资源门。

该路线不复用 WAE/H-DIDM decoder，不重开 v5/v7，不创建 optimizer、checkpoint、
候选模型或生成图像全集。当前状态为 `registered/unexecuted`；在形成有效
closeout 前不允许 neural training、confirm、decoder v2、O0-G、光学、fiber
或 promotion。

## 2026-08-25：H-NDMC nonlinear manifold capacity closeout

H-NDMC retry3 从 clean commit `acc22f5` 完成
`validate → preflight → audit → report → closeout`。retry1 的 classwise bank
长度 harness 错误和 retry2 的参数 hash 符号引用错误均保留为
`invalid_harness`，没有创建下游 consumer；修复后没有覆盖旧 evidence。

最终五阶段 evidence 均为 `status=valid`，分类为
`nonlinear_manifold_safety_failed`，唯一 next action 为
`close_nonlinear_direct_manifold_capacity_canary`。两 Gaussian banks 的
primary correct 为 `2500/2500`，secondary correct 为 `2000/2500`，agreement
为 `2000/2500`；camera-code fraction 约 `1.48e-5`，diversity 约
`0.004–0.007`，coverage `1.67–2.13`。memorization pass fraction 为 `1.0`、
exact duplicates 为 `0`；H-DPGC historical replay、identity-copula、
label-shift、zero-code、64/257 chunk、finite、Jacobian、curvature、参数不变、
no-optimizer 和资源门均已记录。

retry3 evidence SHA256 依次为 validation
`3d3a8cf1c3d30e3923172bc937b6f21295083d2d8963815de1a1ca9e37480d32`、preflight
`6e0a52880b6e222e79ce9a9b63f368f0d9fb6afb62b73f19297aed0114c45783`、audit
`e0fdf0b32fff2159c735f92f67c4c621eaca02aac8c26270fc82e506d5aa6544`、report
`b82d5379863068a8dfa22fcb362e36a1dcfaf37305c1692f728b9719007f267e`、closeout
`f0e21b76723c187056cfb930c306036fbf58e390ebadbda96ccf04a8dcbee43b`。

该 valid failure 关闭 H-NDMC 经验流形 canary，不授权 direct-generator
architecture preflight、neural training、confirm、decoder v2、O0-G、光学、
fiber 或 promotion。

## 2026-08-26：H-JQOC joint quadrature optical capacity closeout

H-JQOC（Joint Quadrature Optical Capacity）从最终 clean lineage
`266083e4752d947cbbd28dd16d8b0152ad4a5e4c` 完成 retry7 的
`validate → preflight → audit → report → closeout`。retry1–retry6 的 harness
失败均保留为独立 attempt，未覆盖失败 evidence，也未创建有效下游 consumer；修复只涉及
量化分位数、reference panel device、tuple 解包、evaluator count device 和 qualification
scalar serialization。

retry7 五个 evidence 均为 `status=valid`。SHA256 依次为 validation
`f8a6231b9910cc3cab0a95c61594237ec4e72c3fa2c02e9d6891556a70e11d5a`、preflight
`43351251e919981c98eb37698a01b660ed87a3b7ded4734f51eb039b8d8ae7e0`、audit
`08b0eb64c7a02e4b0ebec7dd3f7f23818abc98d29431033efcf3cd5b17296fa7`、report
`c56d4bb4d172753ab6c97d7d40171628f3a823fecdfa96deb1ef6a3aa227b982`、closeout
`07806d105e8745a83fba65cc65258adb8733c8da53ed8d1a0c70e574dd69699c`。

正式审计包含 8/16/32D 两个 Gaussian banks、电子 E-d、raw-intensity I-d 和四端口
I/Q Q-d，以及 label shift、zero-code、LO-off、sign/swap、packing、replay 和
finite-difference Jacobian controls。三种维度的 electronic primary minimum recall
范围分别为 `0.73–0.78`、`0.69–0.78`、`0.57–0.69`；secondary 范围分别为
`0.92–0.98`、`0.96–0.98`、`0.91–0.95`，均未通过输入 semantic/safety gate。raw
intensity primary/secondary accuracy 为 `0.075–0.092`/`0.421–0.436`。I/Q 的满秩、
能量恒等式、非饱和、LO-off、port packing、sign/swap 和重复执行 controls 通过，但
按预注册优先级不用于补救 E-d 失败。

最终 classification 为 `electronic_structured_input_capacity_failed`，唯一 next action
为 `close_joint_quadrature_capacity_route`。preflight peak/wall/storage 为
`3353651200` bytes / `32.109134116` s / `11065` bytes；audit 为
`4788746752` bytes / `260.960760050` s / `87355` bytes，均满足资源门。该路线不授权
intensity 或 quadrature architecture preflight、neural training、checkpoint、confirm、
decoder v2、O0-G、optical、fiber 或 promotion；outputs、manifests 和视图保持未跟踪只读。

## 2026-08-27：v3 spatial quadrature electronic admission closeout

`conditional_spatial_quadrature_v3` was implemented in clean code lineage
`d9b7a02`; the final consumer-side harness repairs ended at `e1b16c8`. The
route uses a 32D hierarchical generator (8D global + 24D local) and a fixed
single-exposure four-port receiver, but its optical stages are downstream of
electronic admission.

Retry6 electronic preflight passed after preserving earlier invalid attempts.
The fixed 10-epoch generator-only stage completed `9840` finite steps. The
valid admission used two disjoint 250/class banks, the fixed primary evaluator
`46741e…ec57`, and fixed secondary evaluator `458799…e26c`. Primary accuracy/
minimum recall were `1.000/1.000` in both banks. Secondary accuracy/minimum
recall were `0.900/0.176` and `0.896/0.116`; diversity and coverage also failed.

Final classification is `hierarchical_spatial_input_capacity_failed`, with the
unique next action `close_spatial_quadrature_v3_input_route`. Preflight evidence
SHA256 is `7bd1cc72b96dfd1ca55c39b75c824197dad8f8a992e6d0ba7a0e8895d660cc7e`;
training evidence is `58ace8919d74aec2952fd2cf908c5ab28a85b825ead2b96e641de8f34da4b161`;
admission report is `d097f46f7fc8b1addca64536c3dbed885aa434ef40f1acb37989b08e1868b17f`;
admission evidence is `d897980e47f92ecf5506616f9b316b35667990da595478f80a7d7fa0b6669449`.
No quadrature preflight, smoke, joint training, formal acceptance, candidate
export, or promotion was run.

## 2026-08-26：Package A electronic distribution recovery closeout

Package A completed retry12 from clean commit `a1842b8` through
`validate → gate-audit → preflight → audit → report → closeout`. All six
evidence records were `status=valid`; the H-GFA two-evaluator gate, CUDA
three-step preflight, negative control, resource limits, generator-state
invariance, H-LIR inversion, and H-LGS geometry/finite-difference harness all
passed. The valid scientific classification was
`generator_range_incomplete`: random (`R`), constrained (`C`), and
unconstrained (`U`) inversion arms all failed the registered dual-evaluator
range combination gate in both disjoint 250/class banks.

The real H-GFA banks passed and the cyclic label-shift control was rejected.
Across the six bank/arm measurements, primary accuracy/minimum recall ranged
from `0.9888–1.0000`/`0.9600–1.0000`; secondary accuracy/minimum recall ranged
from `0.8956–0.9172`/`0.1160–0.5360`; agreement ranged from `0.8956–0.9168`.
All six `range_gate` values were false. Memorization anomaly was zero,
generator state before/after was identical, and no model checkpoint was
created. This is a valid scientific failure, not an invalid harness, and does
not authorize retuning, seed substitution, or a downstream optical route.

Evidence is at
`outputs/20260825_electronic_distribution_recovery/runs/20260825_package_a_v1_retry12/`.
The immutable SHA256 values are validation
`3d4c71193455c5a1d8afcd62569f4e08dbc741046793418ff8a6841f74340737`, gate-audit
`99de4cae5f83cda643fd5accd09e54795d24b74d1629539f033f4156e47ac0bf`, preflight
`be07633ca2deffec1fc3b65226172743c7a0770a00b13087b420516ceee09e06`, audit
`2253c9ed0f4284b2e81818ba6089fee1ecb82430989edfa9cf678b1c84c957dd`, report
`fc2d110903b3eaa0a42234a646d4be9249e2ebe2962f58254c2e1dfb9048ae36`, and
closeout `70ed0fe02e9242ec630b44f3e46dd386950ef8d10d1ab5769664b5cb211a8cab`.

The real consumer is the separately reviewed Package B teacher audit and the
future student architecture decision. Package B was not started; all
DDPM/student/optical/phase/quadrature/OLS/fiber training, promotion, and
candidate export remain forbidden.

## 2026-08-27：Package B DDPM–VAE teacher distribution admission closeout

Package B completed the locked v1 producer lifecycle and v2 consumer lifecycle
from clean producer commit `381941f` and consumer commit `ac7baec`. The selected
first-stage and matched diffusion checkpoints were `e553bf…afdd` and
`60f759…cd0`, with explicit 2026-06-03 source-run and acceptance-manifest
lineage. The producer used complete `[1,25,25]` Gaussian noise, guidance `2.0`,
200-step DDIM, two independent banks, and 250 samples/class. All 320 shards
were generated and resumed through an atomic state manifest; preflight and
generation used no evaluator, optimizer, model checkpoint, candidate export, or
promotion.

All lifecycle records were `status=valid`. The audit rejected the cyclic
label-shift control (accuracy `0.0598`) and passed its finite/state/hash/resource
controls, but both teacher banks failed the locked dual-evaluator range contract.
Bank 0 primary/secondary accuracy was `0.3524/0.2212`, minimum recall was `0.036/0.008`,
agreement `0.1676`, and coverage `3.0909/2.8327`; bank 1 was
`0.3428/0.2248`, minimum recall `0.036/0.008`, agreement `0.1716`, and coverage
`3.1483/2.8622`. Memorization anomaly was zero in both banks, but diversity and
semantic gates also failed. The final classification is the valid scientific
failure `teacher_distribution_failed`, first failed gate
`teacher_dual_evaluator_range_gate`, with next action `close_teacher_route`.

Immutable evidence is under
`outputs/20260827_teacher_distribution_admission/runs/20260827_package_b_v1/`:
lineage `b1e385c2b3cba80b91c81e958c1a26804a39554c09daddb6bca0ba5e73bb1a6c`,
preflight `66b65fb5ccd04951201a9634fe101e4cff982f2c1a9d9cecd6c3abd6a280d922`,
bank manifest `c9aebd1e21e8ed185eb1c0db35902a7a4c67c90d01ebd25153a45ca496139ccb`,
validation `ad9f4896f8d43fa03230c5e0d00b10c7f7c5b0e0c590a171f76b4d8a2084f099`,
audit `0369e9500158a94a675e393674f3deff6edf8de0e5d6775e617750b17dbfea92`,
report `0cdd29f6b92656355165926e0ef530fa1e31b40444875476b7c8ca553d1dea5e`,
and closeout `2620311e2bedb6ba6188afc2eae5af5fb9947410e8d70dac892971e2fdc3c6f7`.

The real consumer is Package C's multiscale student admission and architecture
choice. That downstream decision is rejected: student distillation, teacher
training, optical/phase/quadrature/OLS/fiber training, candidate export and
promotion are not authorized. This closeout does not automatically open Package
C; the current DDPM–VAE teacher route remains closed until a separately reviewed
teacher design establishes new lineage and admission evidence.

## 2026-08-27：Package B2 teacher failure attribution closeout

Package B2 is a no-update attribution package whose real consumer is the next
teacher redesign decision. It locked the existing matched first-stage and DDPM
checkpoints, Package B banks, Package A H-GFA banks, qualified dual evaluators,
MNIST split, and Package B thresholds. It did not search for a checkpoint and did
not train or promote a teacher, student, or optical model.

The final run was executed from clean commit `36597b9` after two preserved,
non-consumed harness failures. The first preflight compared a batch-10 replay to
historical batch-16 Package B shards; the corrected same-batch replay passed. The
next run exposed a producer schema omission (`labels`), so it was stopped before
consumer validation. A new v3 run regenerated all 320 shards under the repaired
HEAD rather than splicing either earlier run. Lineage, CUDA preflight, generation,
consumer validation, report, and closeout all passed their structural gates.

The scientific result is the valid failure
`teacher_first_stage_distribution_failed`, first failed gate
`first_stage_reconstruction_distribution_gate`:

| Arm | Bank 0 | Bank 1 | Interpretation |
| --- | --- | --- | --- |
| R real positive control | passed; P/S acc `0.9880/0.9888`, min recall `0.9640/0.9800`, agreement `0.9868` | passed; P/S acc `0.9872/0.9872`, min recall `0.9440/0.9720`, agreement `0.9852` | evaluator and bank harness valid |
| V `decode(encode(real).mu)` | failed; P/S acc `0.9692/0.9444`, min recall `0.8920/0.8520`, agreement `0.9316` | failed; P/S acc `0.9704/0.9464`, min recall `0.8960/0.8840`, agreement `0.9380` | first-stage reconstruction distribution gate failed |
| D locked DDPM replay | failed; P/S acc `0.3524/0.2212`, min recall `0.036/0.008`, agreement `0.1676` | failed; P/S acc `0.3428/0.2248`, min recall `0.036/0.008`, agreement `0.1716` | reproduces Package B; not independently attributed because V failed first |

The cyclic label-shift controls were rejected with accuracy `0.0006`, `0.0024`,
and `0.0598` for R, V, and D respectively. All state hashes were invariant;
evaluator calls were zero in producer/preflight and `2500` in the single audit;
CUDA peaks were `213146112` bytes for generation and `276390912` bytes for audit.
The latent-only comparison was evaluator-independent and recorded classwise
mean/covariance/effective-rank/nearest-distance/multiscale statistics; its mean
distance between normalized encoded-real `mu` and DDPM latent covered
`5.03–11.08` across classes and banks.

The immutable v3 evidence is under
`outputs/20260827_teacher_failure_attribution/runs/20260827_package_b2_v3/`:
lineage `9c91e346127576143fe49bd7ea5f14b479ffd8647cce5b4b24c818bee0db2b00`,
preflight `c67d195f9c7ce1f9917982c3fc747fbf9a73ddabee0eb049431a84d039a6ad3b`,
generation state `48b3115441713d0203978403e97f87b2f0fe7a1173f17ed66e72b6e2ae31e1d1`,
artifact `5f018c0f261d9bd17f6be19cd6f242b1cc2f980caa01200c2664c05d8b3bca0f`,
validation `f8455fb8f369e9e96a53bca532c3862c1bfae5c5b0b8323cbc1d1de6e34aa307`,
audit `fffd9be3a03880b76780f168ce54c7aea3bda8193db7c4459c9cad398b8065e3`,
report `5ad140a9ee8cb431b50c258d6141159fa69b4acb24364ccadb5c10146bfa8fdb`,
and closeout `0fed705e605fd540309e6904a4214eb7de9f668d2d1496868087e9fda6e58515`.

The only permitted next action is a separately scoped `Package B3-VAE` redesign
and first-stage training, followed by a reconstruction distribution gate. A
`Package B3-DDPM` route is not authorized by this attribution, because the
first-stage gate failed first. Package C and all optical or promotion actions
remain blocked; a new teacher must pass the original Package B dual-bank
admission before Package C can be considered.

### B2 evidence-schema amendment (authoritative v4)

The earlier v3 directory is retained as a non-consumed schema diagnostic because
the producer manifest omitted `seed` and `input_artifact_hashes`. After the
producer repair, v4 reran the complete lifecycle from clean commit `9dddb78` and
is the authoritative B2 evidence. Its SHA256 values for lineage, preflight,
generation state, artifact, validation, audit, report, and closeout are,
respectively:

`4fae3feabb1bac774827e4c46719095d5084e75ab41df2789df5cb8906587ae3`,
`812e28350315bddb26023d8f56dd40b65a306d83c68c2ecf119e1b8081e73643`,
`3f7915386bfe8b2c78b0a9aef8a549b03e5165ce609def3609eb51d0c4684a1d`,
`af04f5824784943b0d3d0a33f2404802644f4d96bb7cbb9dfb53ef2d56683c45`,
`9b9a79d3efcd6fb1b8f10eac152acf2d72cf27d6371742999deee48749eae6bc`,
`a1d64208e3c55cc468229ca11552f43dc55f93c2dae3e163435b06bcdd06a71f`,
`aecb9b9a4f3a8e521c22b3cf51f11330fcf25678d3788502caebac0742e9b0e7`,
and `c760ddcc76f378f5517e280d876edaa9015bc3ea7f6216f846ecaf1813388c75`.
The v4 classification and all arm metrics are unchanged from the preceding
valid measurement: `teacher_first_stage_distribution_failed`.

## Package B3-T.2：FP32 semantic-preserving compute cleanup（2026-08-27）

本轮是 B3-VAE 的纯工程吞吐诊断，不训练或评估正式 VAE。实现先在 clean
commit `32e2b34` 提交，随后修复两处只影响 evidence writer 的参数传递缺陷，
最终 authoritative run 使用 clean commit `83e0d74`；中间的 `b3_t2_v1`、
`b3_t2_v2` 目录保留为未消费的 diagnostic-only 记录，没有与最终 run 拼接。

T2-A 使用与 T1 完全相同的 8 warmup、200 train、100 development、FP32、
microbatch 16 和 gradient accumulation 4。T1 baseline replay 通过：总窗口
相对变化 `1.93%`，development 相对变化 `2.16%`，各 phase 均在预注册容差内。
T2-B/C/D 的静态+动态审查没有找到可安全接受的新增优化：finite 检查保持
fail-closed；显式 CUDA synchronize 只在 profiler/preflight 边界；T1 已有的
GPU metric 延迟物化不再重复实施；backward 和 model forward 仍是主要 GPU
开销。

最终分类为 `b3_fp32_semantic_preserving_optimization_insufficient`，不是模型
质量失败。T1 投影 epoch `30.543 min`，最终 FP32 投影 `31.135 min`，总比值
`0.981x`，距 `25 min` 合同 `6.135 min`；backward/model forward/sync 占比
分别为 `76.86%/15.92%/8.85%`。CUDA 峰值 `2.589 GiB`，host RSS 峰值
`1.751 GiB`，evaluator calls `0`，finite 与继承的 T1 resume exactness 均通过。

权威 evidence 位于
`outputs/20260827_teacher_first_stage_recovery/b3_t2_v3/`：baseline replay、
synchronization attribution、optimization matrix、equivalence evidence、
final timing、closeout 的 SHA256 依次为：

`64a531ac55a5b8059ad28ac3f040c96c036766db01cffdaa9877dc646eb27dd8`、
`ab932c6f3da07bbf5c31d69132902eee8663405064992f46aaeefc864b00abc5`、
`a0d3f6bfaa555cd2ab3ce0d3d5a0392f1e8914b26329d246902ab1fa7668c86a`、
`7e6674e99bee213e77b0927b021653b530882cd7f9e99abe58ae9c291572d484`、
`2447c30fe2cb0082c4974097dd950f33e101286299d351d39b2aa987156d26b4`、
`351b6175deb930f1d7bf6d70ca9f3726f9fee34c0aaaa5f74713744a342341d6`。

本轮唯一允许的后续是另立并审查 `numerical_gpu_acceleration_preflight`；不自动
打开 AMP/TF32/compile，不开始 formal B3-VAE、B3-DDPM、Package C 或任何光学
阶段。T2 在此 hard stop。

## Package B3-T.3：numerical GPU acceleration preflight（2026-08-27）

本轮从 Package B3-T.2 的 `b3_t2_v3` authority 启动，目标是隔离验证
TF32、BF16 autocast、FP16 AMP 是否同时满足 numerical equivalence 和
`25 min/epoch` throughput contract。硬件审计通过：RTX 3060 Laptop GPU、
compute capability `8.6`、CUDA `12.1`、cuDNN `90100`、PyTorch
`2.5.1+cu121`，TF32/BF16/FP16 Tensor Core 和 GradScaler 均可用。

FP32 reference 已完成 100 optimizer steps；finite、三步 interrupted/resumed
model/optimizer/scheduler/RNG exactness、parameter update 和 zero evaluator
calls 均通过。随后唯一 canary 进程在候选 arm 尚未完成前被 WSL Python
memory guard 主动终止：process RSS `2,992,820 KiB`，total Python RSS
`6,473,368 KiB`，guard limit `6,291,456 KiB`。因此最终分类是
`resource_aborted`，不是 numerical equivalence failure，也不是模型质量
失败；TF32/BF16/FP16 没有形成有效结论，FP32 throughput、candidate throughput
和第二阶段 `torch.compile` 均未运行。

权威目录为
`outputs/20260827_numerical_gpu_acceleration_preflight/runs/20260827_package_b3_t3_v1/`。
其中 closeout SHA256 为
`eade8b4f013dff8d90e54779d5f6ab0637f34943c69c233158c73b5719a50ae7`，candidate
comparison 为
`a3e478ac5d5fa82bbb2fa763aa4c7899d59576df7004b4e9e9f84427b96ac060`，FP32
reference 为
`631c34f87dacca77c8dbf35862163cf7c757fd72e236d210fe0158c7d5be8e16`。本轮未
启动正式 B3-VAE、DDPM、student 或任何光学阶段，未修改 25 分钟合同。

本轮唯一允许的下一步是另立并审查 `resource_contract_revalidation`；不得把
partial arm 当作数值成功/失败，不得自动 retry、扩大资源合同、打开 AMP 或
组合 compile。T3 在此 hard stop。

## Package B3 research line：temporary pause（2026-08-27）

在 Package B3-T.3 的 `resource_aborted` closeout 后，暂停当前研究推进。
本次暂停不是新的实验，也不改变既有结论：T3 只完成了硬件能力审计和
FP32 reference/resume 验证，候选 numerical arms 未完成，因此 scientific
status 仍为 `not_evaluated`，没有选出 numerical candidate，也没有开始
formal B3-VAE 训练。

暂停期间不启动 CUDA、训练、retry、资源合同扩张、AMP/TF32/BF16/FP16、
compile 组合或任何下游阶段；B3-VAE、B3-DDPM、Package C、student/optical
训练、promotion 和 candidate export 继续禁止。权威证据保留在
`outputs/20260827_numerical_gpu_acceleration_preflight/runs/20260827_package_b3_t3_v1/`。
若恢复，只能先由用户明确决定并另行审查
`resource_contract_revalidation`，不得把 partial arm 证据解释为数值或模型结论，
也不得自动 retry。

## H-LOMA-I0 registration (2026-08-27)

Registered an independent LightGen-like optical-manifold Identity research
line on the clean code-equivalent descendant of `bf9acf0`. This registration
does not resume the paused Package B3 line. The first implementation is limited
to contract/configuration, real-MNIST rendering, formal 2+2 Identity optical
autoencoder validation, and a three-step CUDA preflight. It creates no formal
training command and enters no evaluator, KL, latent sampling, SMF, or FMF
stage. The required decision is manual review of the Identity reconstruction
training contract; failed gates are `invalid_harness` and cannot support a
capacity conclusion.

### H-LOMA-I0 首轮 preflight closeout（2026-08-27）

在 clean commit `20b0301` 上完成 validation 与三步 CUDA preflight。validation
SHA256 为 `9c763024021707314decb945d5b7a692d1b86e20421047a89d81746b70dfe57e`，
preflight SHA256 为
`67a4ff53b86e329596234e8dd7154ce85ac2f22cc6974f089a2acf728eefe166`，resume
artifact SHA256 为
`1e888e5fd8020e09df5dbc6ee9ef876c31d168da2e51ccc0501fe7dc67e88267`。完整
harness 通过：loss `0.0727686286 -> 0.0724386498`，四层 phase 更新、finite、
fixed formal relay/Identity OLS、2+1 exact、zero evaluator calls 和资源门通过。
本结果保持 diagnostic-only，不授权正式训练；唯一下一动作是人工审查
`manual_review_identity_reconstruction_training_contract`。

### H-LOMA-I0 1000 步 Identity Reconstruction Pilot 授权（2026-08-27）

项目负责人基于首轮 preflight evidence SHA256
`67a4ff53b86e329596234e8dd7154ce85ac2f22cc6974f089a2acf728eefe166`，授权
`research/v2-lightgen-optical-manifold-pilot-v1` 分支最多执行 1000 步
diagnostic research training。合同预注册 held-out panel 相对 zero-phase
baseline 的 MSE 至少下降 5%，并要求 paired MSE 低于 cyclic target permutation
对照；三步 paired/负对照/2+1 resume preflight 是前置门。

该授权不包括完整 epoch、SMF/FMF、latent sampling、evaluator、生成实验或
formal promotion。最终 evidence 必须保留 `training_authorized=false` 和
`formal_promotion=false`；pilot 通过后的唯一下一动作是人工审查
`manual_review_full_epoch_identity_reconstruction_contract`，失败则停止且不
形成模型容量结论。

### H-LOMA-I0 1000 步 pilot closeout（2026-08-27）

修复 held-out cyclic permutation harness 后，attempt-03 在 clean commit
`45e6f8d` 完成 1000 步 diagnostic research training。training validation、
training preflight、final training evidence 的 SHA256 分别为
`6d138a40f1d780f1d8ce1cf6ed6536d523c54e4908833e64d4d4cb84fdc76a6b`、
`cfa7ed3452ff41800e39a5a35aecb73df2905de2c3bc970fe863f5affe1649c7`、
`4127f80194f02750737c2e2a1e3f8f7ee4b90df8798cbdd53c4c20369b4a681f`；preflight
resume artifact SHA256 为
`d5d16fef76659ca8b236275feeef4f08d760c59bc6a44301bab86f21e013cf9a`。

运行 wall `877.2158s`、peak CUDA `481032704` bytes、storage `5959746` bytes，
evaluator calls 为 `0`。固定 1000-image held-out panel 的 MSE 从 baseline
`0.0737866908` 降至 `0.0606269501`（相对下降 `17.8348%`），并低于 cyclic
permutation control `0.0694154128`，最终状态为 `passed`。这只支持当前
Identity reconstruction pilot 的预注册诊断门，不授权完整 epoch、SMF/FMF、
latent sampling、evaluator、生成实验或 formal promotion；evidence 仍为
`training_authorized=false`、`formal_promotion=false`。唯一下一动作是人工
审查 `manual_review_full_epoch_identity_reconstruction_contract`。

### H-LOMA-I0 Identity Reconstruction Full-Epoch Continuation 授权（2026-08-27）

基于 pilot attempt-03 的 passed evidence，项目负责人授权从 pilot final step
`1000` 继续执行一次 full epoch，至 global step `5500`。剩余 45,000 张 train
样本与 pilot 已用 10,000 张组成一次无重复的 55k 数据遍历；continuation
schedule SHA256 为
`d15608c4f7f8a5f7709b021155fa252270844bdfab67fc796743014e9897bacf`。

该授权绑定 pilot training evidence SHA256
`4127f80194f02750737c2e2a1e3f8f7ee4b90df8798cbdd53c4c20369b4a681f` 与 final
checkpoint SHA256
`bb072c7b2193b24eb0d989ba731957b52b382c3ff1c83a2be2ddfd7d6d6a2d20`。成功门
是最终 paired held-out MSE 严格低于 `0.06062695011496544` 且低于 cyclic
permutation control；full epoch wall 上限为 75 分钟。该阶段不进入额外 epoch、
latent/OLS、SMF/FMF、evaluator、生成或 promotion；通过后的唯一下一动作是
人工审查 `manual_review_optical_latent_ols_contract`。

### H-LOMA-I0 Identity Reconstruction Full-Epoch Continuation closeout（2026-08-27）

在本地分支 `research/v2-lightgen-optical-manifold-full-epoch-v1` 上，先后
保留四个 attempt。attempt-01 的 preflight 因非零 CUDA Adam 跨独立 runtime
分配不能 bitwise exact 而记为 `invalid_harness`；attempt-02 在 CPU-shadow
修复后仍发现 source optimizer state 被 runtime 共享 mutation；attempt-03
的 preflight 通过，但训练入口在创建 optimizer/run 前拒绝了缺少
`full_epoch_manifest` 的 preflight evidence。以上失败均没有启动 full epoch，
也不形成模型结论。修复提交依次为 `ab868d8`、`83bba25`、`4bbe438`。

attempt-04 在 clean commit `4bbe438` 上完成。validation evidence SHA256 为
`258af9978653a3552f8786b0fc7d76e8434347ed8c7bd5833f41f0f21f9351cd`，preflight
evidence SHA256 为
`e14f733bd8a83738892404a71452972eff11a7c6014d174aff1a33703e721a75`，preflight
resume artifact SHA256 为
`d58f7101d5241853abd20795a84274f6a67451142b113a073abc6d074f8b7d92`。preflight
的 finite、四层 phase gradient/update、冻结隔离、固定 exposure、Identity
OLS/formal relay、evaluator `0`、cyclic negative control、资源和 2+1 exact
resume gates 全部通过。

唯一一次 continuation 从 pilot global step `1000` 运行到 `5500`，完成
`4500` 次 Adam 更新。为满足 exact resume，formal CUDA forward/backward
保持不变，phase Adam 状态和 update 使用注册的 CPU-shadow implementation；
pilot checkpoint、schedule、formal optics 和 trainable parameter set 均未改变。
最终 training evidence 为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_training.json`
（SHA256
`c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948`）。

固定 1000-image held-out panel 的 paired MSE 从 pilot 的
`0.06062695011496544` 降至 `0.03437697887420654`，最终 cyclic-permuted
target MSE 为 `0.06315752118825912`，因此预注册的两个科学门均通过。运行
wall `3809.5485s`、peak CUDA `480711168` bytes、run storage `75434116` bytes，
evaluator calls 为 `0`。final checkpoint
`full_epoch_final_step_005500.pt` SHA256 为
`56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f`。

最终 100-image visual panel 为 review-only：tensor artifact SHA256 为
`9328818e029852ff3ee8e9f05071ce30ff325ebcb89ba077773023466dad2a6e`，不参与
loss、gate 或 checkpoint 选择。最终状态为 `passed`，但仍是
diagnostic-only；evidence 保持 `training_authorized=false`、
`formal_promotion=false`。唯一下一动作是
`manual_review_optical_latent_ols_contract`；不启动额外 epoch、latent/OLS、
SMF/FMF、evaluator、生成或 promotion。

### H-LOMA-I0 PLAN 5 regularization pilot（2026-08-28）

项目负责人授权的 `regularized` / `mse_only` 双臂 pilot 在本地分支
`research/v2-lightgen-optical-manifold-regularization-pilot-v1` 完成。attempt-02
preflight 因 `378.4344s` 超过 `300s` 资源门而保留为 `invalid_harness`；去除
重复 control replay 后，attempt-03 preflight 通过，随后唯一一次 pilot
完成每臂 100 步。validation、preflight、training evidence SHA256 分别为
`e5e15d4d7f7e170b3a3d8b48ad74b00c29e786ccc25bb684542bebe683a5acb3`、
`769b88d114157563bfba1ea6864b627c6227f329af080bce50594743cae8a12e`、
`319526199d0b3ffe26e6ac26bd1e51d984a9a066b4dac7891e50179e91afcab9`。

状态为 `valid_failed`：regularized SWD 从
`0.25427815318107605` 降至 `0.2532535195350647`，仅改善 `0.40296%`，未达
预注册 `10%`；但它低于 MSE-only 的 `0.25692102313041687`。regularized
paired MSE 为 `0.03347916901111603`，source 为 `0.03437697887420654`，
cyclic 为 `0.0630621686577797`；结构和全部 harness gates 通过。运行 wall
`2078.0805s`，peak CUDA `492630016` bytes，storage `12952577` bytes，
evaluator calls `0`。最终 regularized / MSE-only checkpoint SHA256 为
`cbbf4cc579187e6c7d2d81f782741c73fd7f0050cdeab1025a1590fd2fa0c2f3` /
`2ee89473c1cee8aadc9153fc318ed7bf4c34f2924cd9e1c5cb19d7acbdc94856`。

该结果是有效的科学门失败，不支持继续正则化训练；唯一下一动作是
`manual_review_identity_regularization_objective_contract`。所有后续 epoch、
KL、SMF/FMF、生成和 promotion 继续禁止。

### H-LOMA-I0 PLAN 5b regularization objective attribution audit（2026-08-28）

The no-update objective attribution audit completed on branch
`research/v2-lightgen-optical-manifold-regularization-objective-audit-v1` at
clean commit `aece5c2`. Validation evidence SHA256 is
`138b5e5e7bfb16849c88012faf9673c4f31a338e864fa47cf4fe2254fdf67a4f`;
objective audit evidence SHA256 is
`e2d9c079f09f37203f2556e43836d237326740a673d7ae6ec633c9e1570908b5`; and
the projection-drift artifact SHA256 is
`13d65c4a12c6b5cb352d861e7874b4d6b5c94a90279cfd7581ebef6aff81afff`.

The result is `valid_failed` because the fixed 32D source projection was not
stable on the fresh train-only panel: source, regularized-final and MSE-only
principal-subspace overlaps were `0.92070`, `0.91819` and `0.91816`, below the
registered `0.95` gate. Source reconstruction replay, formal optical trace,
finite values, model-state isolation, evaluator calls, peak CUDA
`651952128` bytes, wall `202.4904s` and storage `7193` bytes all passed. The
audit did not select a dose, update a model or write a checkpoint.

The sole next action is `manual_review_adaptive_projection_contract`.
Continuation, KL, SMF/FMF, generation and promotion remain prohibited.

### H-LOMA-I0 PLAN 5b regularization objective attribution audit（2026-08-28）

Following the valid scientific failure of the fixed aggregate-prior pilot, the
project owner authorized one diagnostic-only, no-update objective attribution
audit on branch
`research/v2-lightgen-optical-manifold-regularization-objective-audit-v1`.
The audit is bound to the authoritative pilot attempt-03 evidence, the
step-5500 source checkpoint, both pilot final checkpoints, the fixed PLAN 5
projection and the MNIST manifest. It uses fresh train-only panels after
excluding every prior calibration, gradient, pilot-schedule and held-out
index.

The registered measurements are fixed-projection drift and frozen-model
gradient reports for `1x`, `2x`, and `4x` the source lambda, with a fixed
rank-permuted negative control. The audit creates no optimizer, updates no
parameter, writes no checkpoint and cannot authorize KL, SMF/FMF, generation
or promotion. Its real consumer is the manual decision on a possible bounded
dose-response pilot; all other outcomes stop the line under their registered
manual review action.

### H-LOMA-I0 PLAN 5c adaptive projection audit（2026-08-28）

Following the valid fixed-projection stability failure, the project owner
authorized one diagnostic-only, no-update cross-fitted projection audit on
`research/v2-lightgen-optical-manifold-adaptive-projection-audit-v1`. It uses
three disjoint balanced calibration panels, an independent assessment panel
and three gradient-control panels after excluding every prior panel and the
held-out split. Candidate ranks are 16/24/32; the selected consensus projection
is fit only from calibration data.

The audit will distinguish finite-sample PCA instability from genuine model
drift, then retest deterministic sliced-Wasserstein dose compatibility and a
rank-permuted control. It cannot update a model or create a checkpoint and
does not authorize KL, SMF/FMF, generation or promotion.

### H-LOMA-I0 PLAN 5c adaptive projection audit closeout（2026-08-28）

The no-update cross-fitted projection audit completed on branch
`research/v2-lightgen-optical-manifold-adaptive-projection-audit-v1`. An
initial CUDA invocation was invalid before evidence creation because of a
wrong internal helper reference; it was repaired on clean commit `c7878c6`.
The new attempt-02 validation and audit evidence SHA256 values are
`7e11e3d989e2dd558d11f4d13d96b2cbd33c628e337806ef27d888ff071d99b5` and
`98f40cc3312c01f22056424e6a4baa25d6bfd618bb0eb4f3dcbab75934fbb7a3`.

All candidate ranks 16/24/32 failed the registered stable-projection gate;
none was selected. The audit was a valid scientific `valid_failed` result,
not a resource or harness failure: source MSE replay, formal Identity trace,
state isolation, evaluator calls `0`, peak CUDA `446995968` bytes and wall
`205.8172s` all passed. This closes the deterministic fixed-projection route
for the current attempt and routes to `manual_review_probabilistic_latent_contract`.

### H-LOMA-I0 SMF integration canary (2026-08-28)

The project owner authorized one diagnostic-only engineering canary on
`research/v2-lightgen-optical-manifold-smf-integration-canary-v1` after PLAN 5c
closed as a valid stable-projection failure. The canary inserted the existing
100-port LP01 SMF research OLS between the formal 2-layer encoder and decoder
and ran 20 steps each for `smf_rank32_swd` and `smf_mse_only`. The rank-32
projection remained marked `diagnostic_unstable_projection`; the run did not
authorize stable-projection training, FMF or generation.

Attempts 01--06 were harness/consumer failures. Attempt-07 at clean commit
`da5c13c` completed validation, CUDA preflight, exact resume checks and both
training arms. Validation, preflight, training and panel SHA256 values are
`bc87f1c996331dbe88f059b71faac2da0ce891c32969dd195c881a9eeff93bc2`,
`70fc113ca98a4776d88837e3aab4293d0676d965ab98dbcc19f631820d1fe70f`,
`271f934515bde2fddaab413aba05e82472751b1664053c3a27d28de3c76a1860`, and
`2b08105870853a81bc8cc59c866de411b8c78cd06cc0da1bac408fc89500058f`.

The evidence outcome is `diagnostic_completed`, not a scientific `passed`.
Fixed-panel paired MSE was `0.0639665723` at SMF initialization and
`0.0631253868`/`0.0631285161` after 20 steps for MSE-only/rank-32-SWD; cyclic
controls were `0.0748243704`/`0.0748260617`. Final SWD was
`0.0294856094`/`0.0288933087`, but the two final arm tensors differed by only
`1.86e-8` MSE. Coupling efficiency was `0.5339` and active-port fraction was
`0.9317`. Peak CUDA was `2375665664` bytes, wall time `97.0598s`, storage
`86251566` bytes, and evaluator calls were `0`.

The canary establishes executable SMF transport, gradient flow, resume and
state-isolation contracts. It does not establish SMF scientific superiority
or authorize FMF. The sole next action is
`manual_review_smf_canary_visual_and_metric_trends`; all evidence retains
`training_authorized=false` and `formal_promotion=false`.

## H-LOMA-I0：SMF bottleneck attribution audit registration（2026-08-29）

固定 OLS recovery attempt-02 的工程门通过，但 coupling 臂相对 MSE control
的功率保持率仅提升 `0.2014` 个百分点，未达到预注册 `+5pp`，因此停止
1000-step continuation。project owner 授权一次 no-update、diagnostic-only
归因审计，其真实消费者是后续 coupling-dose、FMF mode-capacity、decoder
mismatch 或 projection-objective 人工合同；lineage、数据隔离、状态冻结、
负对照、provenance 或资源 gate 失败时 fail-closed。

新分支为
`research/v2-lightgen-optical-manifold-smf-bottleneck-attribution-v1`。审计
绑定 attempt-02 validation/preflight/training/panel 和 source/final checkpoints，
使用新的 balanced assessment/gradient panels，比较冻结 LP01 SMF 与既有 LP01+
LP11a FMF M2 的 observation-only captured power，执行 control/coupling
encoder-decoder `2×2` cross-swap，并测试 registered coupling lambda 的
`1×/2×/4×/8×` 无更新梯度剂量。FMF 不训练，历史 gate 不追溯放宽，所有 evidence
保持 `training_authorized=false`、`formal_promotion=false`。

## 2026-08-29：H-LOMA-I0 SMF bottleneck attribution audit closeout

The project-owner-authorized no-update audit consumed the fixed-OLS recovery
attempt-02 evidence and checkpoints. Its real consumer was the manual choice
among a higher coupling dose, FMF mode-capacity reconstruction, passive decoder
mismatch, or projection-objective replacement. Source replay, data isolation,
mode-bank, gradient-boundary, freeze, provenance and resource failures reject
all downstream training.

The first CUDA attempt is retained as `invalid_harness`: the initial implementation
exceeded available memory before writing audit evidence. After serializing model
and capacity evaluation, authoritative attempt-02 completed on clean commit
`3e608ee`. Validation, audit, decomposition and panel hashes are
`4185875318a0a301ab966f99c8ab4eedde517b14c5354864e0520f12fedacb81`,
`6f9b87adff81069b0b08d2d86e623d5c49565f30b2ca903640aa59b2c9e7b766`,
`5ad548c4c258f70ada5979c8dfb5e67021a90a3c6e71e923ef3a963a8abac1dd`, and
`047001891a3e2de4d07d33d85ed9782371b2790927095c5b159904458f7357d5`.

All harness and resource gates passed, including authoritative terminal MSE
replay, unchanged model/operator state, evaluator calls `0`, `2.305 GiB` peak
CUDA memory and `201.16 s` wall time. Observation-only LP01 M1 retained
`65.3577%` power and LP01+LP11a M2 retained `71.8676%`, a `6.5099` percentage-point
capacity difference. Only the registered `1x` coupling dose was safe across all
three gradient panels; no higher dose was eligible.

The authoritative outcome is `valid_failed_mode_capacity`; the sole next action
is `manual_review_fmf_mode_capacity_reconstruction_contract`. This supports a
mode-capacity review but does not establish FMF reconstruction superiority.
FMF training, continuation, prior sampling, generation and promotion remain
unauthorized; evidence retains `training_authorized=false` and
`formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 fixed OLS projection-loss recovery canary

在 corrected KL dose reaudit 和 matched KL effect pilot 之后，执行了从 PLAN 6 SMF
source checkpoint 开始的四臂 100-step fixed-OLS canary。固定 100-port LP01 mode
bank、identity transport、formal IdentityOLS、relay、exposure 和四层 phase-only
参数；新增 projection objective `1 - P_smf_out/P_smf_in`，KL 臂使用已注册 beta。

attempt-01 因 terminal KL metrics 缺失而登记为 `invalid_harness`；attempt-02 在
修复 evaluator metrics 后成为 authoritative retry。validation、preflight、training
和 panel SHA256 为：
`18d63ff5e9c3d18db7c669fc8c596423d8a2cc02ed50338cf8043fcc609d06f6`、
`abf063fa6e39511b01a661923f755e84c1eefd8285c91bdce0651a80066d8833`、
`a73b3917d64049509b2c6f025e0025bd506db4d6f95271598c5dc95c43c0b65e`、
`c432221ba0725da3744f66459a3c513ef632d5aa202864123007137aa5798b61`。

所有 arm 均完成，四层更新、exact `2+1` resume、状态隔离和资源门通过。coupling
臂 projection residual 下降，但最终 power retention 仅比 optical MSE control
高 `0.2014` 个百分点，低于 `5` 个百分点门；candidate KL 臂则低于自身初始 KL
和 no-KL KL。权威状态为 `valid_failed`，表示固定 OLS coupling recovery 尚未
成立，而非 harness/resource 失败。唯一下一动作是
`manual_review_smf_bottleneck_attribution_contract`；因此不执行 1000-step
continuation，不进入可训练 transport、FMF、生成或 promotion。

### 2026-08-28：H-LOMA-I0 corrected KL effect pilot registration

The corrected no-update KL dose reaudit completed with exact source MSE replay,
finite gradients, explicit four-layer checks, KL encoder/posterior isolation,
model freeze and the unique compatible `8x` multiplier. The earlier
invalid-harness dose attempt remains preserved and is not scientific evidence.

The next authorized consumer is one four-arm, 100-step matched pilot using the
PLAN 6 SMF checkpoint, a fresh disjoint balanced schedule, shared epsilon for
posterior arms, reset Adam and persisted `2+1` resume. Its rejection decision
is the registered KL/reconstruction/collapse gate. No trainable OLS transport,
FMF long training, prior-only sampling, generation or promotion is authorized.

## 2026-08-28：H-LOMA-I0 SMF VAE KL chain attribution audit closeout

The authoritative no-update KL audit completed on clean commit `f47e8c6` in
attempt-06. Attempts 01--05 remain retained as superseded validation or audit
artifacts: the first used the pre-fix context, attempts 02--03 stopped on
harness errors, attempt-04 had an incorrect outcome classification, and
attempt-05 had an incorrect comparison evidence kind.

The validation, audit, horizontal-comparison and panel SHA256 values are
`e7de7e683b2650cd9ad6e622ac0bc76899f880eface6732fb86ac2cba4f31a74`,
`40d612e20ffc7044c0a8e21bce86c7d07f3076a31ea0d6aadee5fa6f53ea22ce`,
`678e9b37d14ef84cacc94819f8cc75f18d83d5bee6ac140481e7b1679667de2b`, and
`1cc913068aa389e87122abd59bcd9b4e9115e09d86cf5b850d3cd769398c6361`.

The source MSE reproduced with difference `0`; beta KL was `3.0639548`, lower
than no-KL `3.3382955` but higher than initialization `2.9956918`. Posterior
mean variance increased to `1.4764290` from `0.9877587`, all 200 latent
dimensions remained active, and the model/optical state was unchanged. The
matched three-arm comparison produced mean-path MSE values `0.0380180`
(beta-VAE), `0.0379458` (no-KL) and `0.0503602` (deterministic).

The diagnostic outcome is `diagnostic_completed_parameterization_limited`:
the relative KL constraint is real, but every tested 1x/2x/4x combined
direction conflicted with reconstruction across all three independent panels.
The sole next action is
`manual_review_smf_vae_posterior_parameterization_contract`. No KL dose
response, new training, prior-only generation, FMF extension or promotion is
authorized; evidence remains `training_authorized=false` and
`formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 PLAN 6 Identity--SMF reconstruction pilot closeout

The authoritative Identity--SMF reconstruction comparison completed on clean
commit `bced59f` in attempt-04. Attempts 01–03 remain retained invalid
harness attempts and produced no training result: the first had an incorrect
preflight gate access, the second rejected a valid preflight because of a
self-referential hash requirement, and the third failed before updates when
the interpolation helper received a field without its batch dimension.

The validation, preflight, training and comparison-panel SHA256 values are
`d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5`,
`722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623`,
`a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e`, and
`9de2fc9d9c2d43ed9752d8609afcaa5455cfb05b9c411d7d0cca9b2a8cf50fdf`.
The final SMF and Identity checkpoint hashes are
`949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94` and
`9258eada74bf04e114d5f74fc3722450f56fad684fd978c065fd3f94e0b1683b`.

The fixed held-out panel gave SMF paired MSE `0.0659028739 → 0.0508093871`
(`22.90%`), final cyclic MSE `0.0671605766`, centered correlation
`0.6562196 → 0.7100222`, variance retention `0.0627982 → 0.1317942`, and
power/coupling retention `0.5294868 → 0.6535921`. Identity control paired MSE
was `0.0344723202 → 0.0303227697`, below cyclic MSE `0.0630837232`; its
centered correlation and variance retention also improved. Both arms completed
1000 steps with all four phase layers updating. Source/formal state, exact
resume, finite/safety/resource gates and evaluator isolation passed; evaluator
calls were `0`, peak CUDA `2677763072` bytes, wall `2212.9581s`, storage
`817444396` bytes.

The outcome is `passed` only for the registered bounded SMF reconstruction
question. It does not authorize FMF, extra epochs, generation or promotion.
The active-port fraction was unavailable from the reused diagnostic helper and
is explicitly left non-gating. The sole next action is
`manual_review_fmf_m2_integration_canary_contract`; all evidence retains
`training_authorized=false` and `formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 PLAN 6 Identity--SMF reconstruction pilot

The project owner authorized a new diagnostic-only 1000-step matched
reconstruction pilot on branch
`research/v2-lightgen-optical-manifold-smf-reconstruction-pilot-v1`. It is
bound to the completed canary evidence, the step-5500 Identity checkpoint and
the authoritative MNIST split. Both arms start from the same source weights
with reset CPU-shadow Adam and share a fresh balanced schedule; the only
structural variable is formal IdentityOLS versus the existing 100-port LP01
SMF research transport between the fixed 2-layer encoder and decoder.

The pre-registered run is MSE-only, microbatch 1, activation-checkpointed and
requires CUDA three-step plus persisted 2+1 resume preflight. It measures SMF
reconstruction recovery against its same-run initialization and an Identity
control, with cyclic MSE and safety/resource gates. It does not authorize
SWD/projection, KL, latent sampling, extra epochs, FMF, generation or
promotion; all evidence remains `training_authorized=false` and
`formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 PLAN 7a FMF M=2 integration canary closeout

The project owner authorized one diagnostic-only M=2 FMF integration canary
on `research/v2-lightgen-optical-manifold-fmf-m2-integration-canary-v1`.
It used research-only nested `LP01+LP11a` identity transport, with
`smf_m1_mse_control` and `fmf_m2_mse_canary` sharing reset Adam and an exact
20-by-10 balanced schedule. No SWD, KL, latent sampling, random coupling,
FMF long training, M=3, generation or promotion was authorized.

Attempts 01--07 are retained validation/preflight harness attempts. Attempt-08
passed preflight but failed the storage gate due to panel backing-storage
retention. Authoritative attempt-09 completed on clean commit `f53bd4e` with
validation/preflight/training/panel hashes
`f4af9a87a2f953a4873a11a1f09814e1ab9826dc336638940bb05148a4abacce`,
`14abce541b0961ca848678f44d600e48c70e13e22b6cbdbf2ef8c32ba38100ca`,
`f699fccce7da7fb5f1560c5823ee76e2c713ba9e726fb9f6213fe967da36afb6`, and
`709b1ff2f716a4a77f733ef3e77a376d9f4c234d7fe4d4d3ddd1156f28a5c195`.

M2 held-out paired MSE changed `0.0498638563 → 0.0496286638`, final cyclic
MSE was `0.0667282492`, centered correlation changed `0.6993410 → 0.7016757`,
variance retention `0.1480903 → 0.1489086`, effective rank `8.59913`,
LP01/LP11a power fractions `0.9160477/0.0839523`, and coupling `0.7191151`.
M1 paired MSE changed `0.0508093871 → 0.0507159270`, with cyclic MSE
`0.0671152920`. The 200-mode bank, LP01 preservation, LP11a excitation,
passivity, no-bypass controls, four-layer updates, exact resume, state
isolation and evaluator calls `0` all passed. Peak CUDA was `2661837824`
bytes, wall time `123.8385s`, and storage `229115402` bytes.

The record is `diagnostic` with outcome `diagnostic_completed`. It establishes
an executable M=2 integration chain and short-term update trend only; it does
not establish FMF superiority or authorize longer FMF, M=3, mode coupling,
generation or promotion. The sole next action is
`manual_review_fmf_m2_reconstruction_pilot_contract`.

## 2026-08-28：H-LOMA-I0 PLAN 5 probabilistic latent recovery pilot

The project owner authorized one bounded diagnostic SMF hybrid-VAE pilot on
`research/v2-lightgen-optical-manifold-smf-vae-pilot-v1`. This is a controlled
return to the originally planned PLAN 5 probability interface after the
deterministic PCA+SWD aggregate-prior approximation. The source is the passed
PLAN 6 SMF step-1000 checkpoint; formal IdentityOLS, the existing SMF/FMF
operators, old evidence and the current v2 runtime remain unchanged.

The research posterior is located between the 100-port LP01 projection and
mode-bank re-emission. It represents each complex coefficient by real and
imaginary parts, uses fixed train-only per-port normalization, and applies a
shared `Linear(2,16) -> SiLU` head for identity-residual `mu` and `logvar`.
The reparameterization path is `z=mu+exp(0.5*logvar)*epsilon`, followed by
fixed inverse normalization and complex SMF re-emission. This is explicitly a
`project_approximation` and `research_hybrid_active_latent_injector`, not a
formal all-optical hardware implementation.

The three arms are `smf_deterministic_mse`, `smf_stochastic_no_kl`, and
`smf_beta_vae`, with a shared balanced schedule and epsilon stream. The pilot
is gated by clean lineage, a real CUDA three-step and persisted 2+1 resume,
finite/gradient/isolation checks, and the registered KL/reconstruction
decision gates. It does not authorize prior-only generation, FMF extension,
additional epochs, evaluator use or promotion. All evidence retains
`training_authorized=false` and `formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 PLAN 5 probabilistic latent recovery pilot closeout

The project owner authorized one bounded diagnostic three-arm SMF hybrid-VAE
pilot on branch `research/v2-lightgen-optical-manifold-smf-vae-pilot-v1`.
The research-only posterior was inserted between the 100-port LP01 SMF
projection and re-emission. It used fixed real/imaginary normalization, a
shared per-port `Linear(2,16) -> SiLU` head, identity-residual `mu`, bounded
`logvar`, fixed epsilon reparameterization, and standard 200-dimensional
diagonal-Gaussian KL. This is a `project_approximation`, not a LightGen source
or formal all-optical hardware implementation.

Attempts 01--04 were retained harness attempts. Attempt-05 completed but was
superseded because the implementation omitted the mandatory final-KL-below-
initialization gate. Authoritative attempt-06 ran after repair on clean commit
`7a415ea`. Validation, preflight, training and panel hashes are
`d50919acb75ffe68453db52a6d87ebdfeb8ae9963f157b0bd3afb582f17de04c`,
`5e1f8569e7e58ce086db37a318df76fa81107820323128dafbedd02ad0e8a936`,
`ec6afc5b15274ce9a7e2446bef43bb6346007480da23408fe1399b412200c476`, and
`e8134f326dfdbdc33e900be17292f17c01d675427f1409193a27418ce64cb664`.

All arms completed 100 steps with exact persisted 2+1 resume, finite values,
four-layer phase updates, posterior updates, direct-SMF mean replay MSE
`2.72e-15`, evaluator calls `0`, peak CUDA `2362135552` bytes, wall time
`416.0924s`, and storage `116038584` bytes. Deterministic MSE ended at
`0.0503602326`; stochastic no-KL ended at `0.0379457697` with KL
`3.3382956982`; beta-VAE ended at paired MSE `0.0380179696`, cyclic MSE
`0.0669108853`, and KL `3.0639550686` versus its initial `2.9956917763`.

The result is `valid_failed`, not a harness or resource failure: the beta-VAE
KL did not fall below its own initialization, although it remained below the
stochastic no-KL arm and all reconstruction/variance/finite gates passed.
The next and only consumer is
`manual_review_identity_posterior_collapse_contract`. No KL retuning, extra
training, FMF, prior-only generation, generation or promotion is authorized;
`training_authorized=false` and `formal_promotion=false` remain mandatory.

## 2026-08-28：H-LOMA-I0 SMF VAE KL chain attribution audit authorization

The project owner authorized one no-update, diagnostic-only audit on branch
`research/v2-lightgen-optical-manifold-smf-kl-attribution-audit-v1`. Its real
consumer is the manual decision on whether to authorize a bounded KL dose or
posterior-parameterization experiment. The rejection decision is fail-closed
when source replay, KL formula, gradient isolation, state freeze, provenance
or resource gates fail.

The audit consumes only the authoritative attempt-06 VAE validation,
preflight, training, normalization, panel and final checkpoints. It performs
no optimizer creation, parameter update or checkpoint creation. The matched
comparison is limited to deterministic SMF MSE, stochastic SMF without KL and
SMF beta-VAE. Identity, PLAN 6 SMF and FMF M2 are context-only references with
different budgets and cannot define an architecture ranking.

The audit preserves the prior valid-failed conclusion: beta-VAE KL rose from
`2.9956917763` to `3.0639550686`, but remained below no-KL `3.3382956982`.
It will decompose mean and variance KL terms, verify shared epsilon and
reparameterization, measure cross-swapped phase/posterior states, and inspect
1x/2x/4x no-update gradient directions. All evidence remains
`training_authorized=false` and `formal_promotion=false`.

## 2026-08-28：H-LOMA-I0 KL dose reaudit and fixed-OLS loss-recovery registration

The project-owner-authorized continuation starts from clean commit `a62b264` on
`research/v2-lightgen-optical-manifold-kl-ols-loss-recovery-v1`. The first
no-update consumer repairs the attempt-06 dose formula to
`g_mse + multiplier * registered_beta * g_kl`; attempt-06 remains immutable and
its dose classification is not treated as scientific evidence because beta was
omitted. The subsequent matched KL pilot and fixed-OLS coupling-loss canary are
independent consumers with fresh schedules and explicit preflight gates.

SMF mode banks, identity transport, formal optics and exposure remain fixed.
Projection loss is encoder-only, KL is encoder/posterior-only, and CMOS MSE is
the end-to-end reconstruction objective. No trainable OLS transport, prior-only
generation, FMF long training or promotion is authorized; evidence remains
`training_authorized=false` and `formal_promotion=false`.

## 2026-09-02：H-LOMA-CBOGT-S6-DSFA data-weighted spectral functional attribution

The project owner authorized one frozen, no-update consumer of the RMAL detector readout after
LSDA and LRRC. The real consumer is manual review of whether low operator-energy LSDA modes have
large functional effect on real detector data. The rejection decision is fail-closed against
interpreting spectral concentration or data-weighted attribution as reconstruction sufficiency,
minimum rank, training readiness, generation readiness, or promotion authority.

Implementation/config/review/tests/CLI were committed at
`d032738c0f02af5ada85554224e045d238416e0a`; RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC evidence, probe,
renderer, checkpoint, OLS, detector and formal runtime were not modified. The authoritative
attempt is `outputs/20260902_v2/runs/lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/`.
Validation, audit and compact SHA256 are
`ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe`,
`2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e`, and
`8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3`.

The result is `passed / data_weighted_spectral_functional_attribution_completed`. Selection and
confirmation full-probe MSE are `0.009812927922827133` and `0.010185289342506197`; total signed
functional gains are `0.0885958794604826` and `0.07750253052523498`. The B256 tail carries only
`0.0007625489` of operator energy but `0.3517092/0.3451470` of data-weighted output energy in
selection/confirmation. Exact additive identities, LRRC registered replay, finite/state/no-update,
streaming/no-bank and resource gates passed. The only next action is `manual_review_DSFA_result`;
training, generation and promotion remain unauthorized.

## 2026-09-02：H-LOMA-CBOGT-S7-MPCDAR mean-preserving centered data-aware readout recovery

The project owner authorized one frozen RMAL consumer after MCFA, with the scientific question
restricted to the first confirmed rank in the preregistered family
`[1,2,4,8,16,32,64,128,256]`. The mean branch `c_mean=b+mu_fit W` was retained, only the centered
branch was truncated using the fit-derived `A_fit=C_fit^(1/2)W` right-output basis, and the matched
mean-preserving raw-SVD path was descriptive only. The rejection decision is fail-closed against
calling rank 16 a minimum/intrinsic/physical rank, general reconstruction sufficiency, or any
decoder/training/generation/promotion authorization.

Implementation/config/review/tests/CLI and the required preflight support were committed before the
formal run; the code used for the result is clean commit
`b59172750d0c88bbc9f7426ab7a21a1d63639f57`. The authoritative attempt is
`outputs/20260902_v2/runs/lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/`.
Validation, audit and compact SHA256 are
`b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2`,
`43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785`, and
`9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9`; the frozen RMAL probe hash
remains `3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`. attempt-01/02 were
retained as validation-only or pre-execution-stop records and are not consumed.

The fit replay used 1000 unlabeled fit samples with float64 population moments; both mu/C hashes
matched DMSA, the moment identity relative residual was `2.98546e-16`, and the PSD raw minimum
eigenvalue was `3.23645e-09` with zero material/tiny negative counts. Data-aware cumulative energy
at registered landmarks `1/2/4/8/16/32/64/128/256` was
`0.118309/0.207018/0.355819/0.541212/0.715164/0.868603/0.963351/0.994898/1.000000`; effective/
stable rank telemetry was `37.184239/8.452439`. Boundary gaps used the A_fit/G_fit spectrum and
remained descriptive; LSDA raw gaps were labeled as comparator telemetry.

The rank-256 positive anchor passed first, including V orthogonality and A_fit/W projector and
prediction identities. The sequential data-aware search evaluated `1,2,4,8,16`; rank 1 failed
strict template crossing, ranks 2/4 failed instance-sensitive and strict template crossing, rank 8
passed selection but failed confirmation strict template crossing, and rank 16 passed both. Thus the
result is `passed / mean_preserving_centered_data_aware_readout_recovery_completed` with
`first_confirmed_data_aware_candidate_rank=16` and previous registered candidate `8`; rank 256 was
not repeated. Rank-16 selection/confirmation probe MSE was `0.0210726835/0.0212836881`, with
relative improvement `0.660383/0.631703`.

For every evaluated rank, the same stream recorded old LRRC raw, mean-preserving raw and
mean-preserving data-aware MSE plus A→B/B→C/A→C deltas; the comparator did not affect search or
confirmation. Audit passed `14/14` gates. The replay was CMOS-only, streaming, no-bank and
no-target-fitting; model/OLS state, no-update, finite and resource controls passed, with peak CUDA
`2274099712` bytes and wall time `183.266 s`. Governance flags remain
`reconstruction_manifold_supported=false`, `generative_manifold_supported=false`,
`generation_readiness=not_ready`, `training_authorized=false`, `formal_promotion=false` and
`closed_task_relevant_bottleneck_not_supported`. The only next action is
`manual_review_MPCDAR_result`; no training, candidate export, generation or promotion is authorized.

## 2026-09-02：H-LOMA-CBOGT-S8-MPCDAR-R16-IQ frozen rank-16 readout integration qualification

The project owner authorized one post-detector, frozen integration consumer of the S7 MPCDAR
rank-16 artifact. The question was limited to whether a write-once artifact can be reloaded in a
fresh process and reproduce the exact RMAL selection and confirmation result; the rejection
decision remains fail-closed against minimum/intrinsic-rank, native-optical-reconstruction,
general sufficiency, decoder, training, generation, or promotion claims.

The implementation is bound to clean commit
`c0140577fddb56af79d72a6d473cc49ef04f5ffc`. The authoritative attempt-03 is
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/`.
Validation, audit and compact SHA256 are
`844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb`,
`3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4`, and
`12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271`; artifact manifest and
payload SHA256 are `e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45` and
`e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e`. Earlier attempts remain
immutable and are not consumed.

The artifact contains four CPU float64 tensors: `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`,
and `V16[40000,16]`; scalar count is `684352`, raw tensor bytes `5474816`, and serialized
payload bytes `5476856`. Fit replay used `N=1000`; mu/C hashes matched the parent contract and
the moment identity relative residual was `2.98546e-16`. Rank-256 positive replay had max
absolute/relative-Frobenius residual `5.68008e-11/1.17382e-13`.

Process A exited before Process B. Fresh-process loaded/reference residuals were zero for both
panels. Selection and confirmation rank-16 probe MSE were
`0.02107268348957069/0.02128368811373826`, relative improvement was
`0.6603834534521028/0.6317033169406975`, and confirmation bootstrap 95% CI was
`[0.03618517890572548,0.036830976605415344]`; both RMAL Gate vectors matched the S7 parent and
passed. The result is `passed / frozen_mean_preserving_data_aware_rank16_integration_qualified`,
audit `16/16`, with `integration_only=true`.

Factorized/full-W deterministic operation-count telemetry was `644096/10240000` MAC/sample,
ratio `15.9`; this is not a latency, throughput, or energy-efficiency measurement. Runtime
independence, default-disabled, explicit-artifact, streaming/no-bank, state/no-update, finite,
and resource controls passed; peak CUDA was `2274099712` bytes and wall time `133 s`. Governance
flags remain reconstruction/generative false, `generation_readiness=not_ready`,
`training_authorized=false`, `formal_promotion=false`, and
`closed_task_relevant_bottleneck_not_supported`. The only next action is
`manual_review_MPCDAR_R16_IQ_result`; no automatic default switch, training, decoder, generation,
or promotion is authorized.

## 2026-09-02：H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ frozen rank-16 float32 numerical qualification

S9 consumed only the S8 rank-16 frozen artifact. The registered question was limited to deterministic
float64→float32 tensor casting, fresh-process filesystem reload, CPU readout, and preservation of the
RMAL selection/confirmation Gate vectors on the same CMOS feature stream. It did not refit, train, search
ranks, modify physical forward, or make a GPU/production claim. The rejection boundary remains minimum/
intrinsic rank, native optical reconstruction, general sufficiency, decoder, training, generation, and
promotion authorization.

Implementation/fix commits were `4b75e4d67efc1dc90a01d1d33a6aa3c1c243c652` and
`ebbd0d232c56ee45a69dea830911eff4f697ad47`. Attempt-01 is preserved as an invalid harness serialization
attempt; authoritative attempt-02 is
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/`.
Validation/audit/compact SHA256 are
`5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137`,
`0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde`, and
`5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940`. F32 manifest/payload SHA256
are `1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` and
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`.

The artifact has `684352` scalars, raw float32 tensor bytes `2737408`, serialized payload bytes
`2739448`, and tested rank family `[16]`. Selection/confirmation max prediction differences are
`7.8160793304e-7/7.8570211703e-7`; normalized L2 differences are
`1.5589117128e-7/1.5627624021e-7`; aggregate MSE absolute drift is
`8.2576945282e-11/2.8491427967e-11`; relative drift is
`3.9186724995e-9/1.3386508868e-9`. All 35 audit gates passed, including exact cast identity, numerical
tolerances, RMAL Gate-vector identity, fresh reload, CPU boundary, state/no-update, no-bank and runtime
independence. Final status is `passed`, classification is
`frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified`, and `integration_only=true`.
The only next action is `manual_review_MPCDAR_R16_F32_NQ_result`; no training, decoder, generation,
promotion, or automatic default switch is authorized.

## 2026-09-02：H-LOMA-CBOGT-S9V-FULL-PATH-VIS full-path optical representation visualization

S9V was a descriptive-only visualization consumer of the frozen RMAL detector readout. Its consumer
was manual review of the fixed confirmation panel's full optical path, reconstruction families, detail/
error subset, and MPCDAR R16 F64/F32 display; it did not create reconstruction/generative-manifold,
performance, training, decoder, generation, or promotion authority. The closeout commit was
`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`; output remained outside Git.

The authoritative visualization manifest is
`outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01/visualization_manifest.json`,
SHA256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`, with status
`completed`, classification `descriptive_visualization_completed`, 26 files, and forward-consistency
max residual `0`. The S9V targeted test passed (`9 passed`); parent/visualization regressions and the
full Windows-native suite were green at that checkpoint. Governance flags remained not-ready and no
training/generation/promotion. The next action was
`manual_review_MPCDAR_R16_RUNTIME_PQ_implementation_contract`.

## 2026-09-03：H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ frozen rank-16 float32 runtime/performance characterization

S10 consumed only the immutable S9 rank-16 float32 artifact and measured a CPU-only, readout-only
runtime contract. The rejection boundary was explicit: no measured-speedup, GPU/hardware/production-
readiness, reconstruction-sufficiency, decoder, training, generation, or promotion claim. S10 started
from HEAD `8680c12be955df72906b955433e6daeabfdfec4a`; implementation/fix commits were
`dd1c38fa965e32deedd655f00f3cd974e1534cd8`, `488f12fbcf19e3663c7aa685bc4d4ad717db2a30`, and
`d0655652cd86bdc9f3852a9bbf4faa45f59f069b`.

The first formal attempt was retained write-once after a namespace-isolation source-layout correction;
authoritative attempt-02 is
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_runtime_performance_qualification_v1_attempt_02/`.
Validation/audit/compact SHA256 are
`8293e549b1ae48447e28be924406525aa03fa8843c0f1a71fa1ec21e3f5894c5`,
`7b519b3a75cdf59ecfe2678d46a8f74ae3084cb7b9de90c95763efecc0f2f4ee`, and
`ded0959097c88a74031edc1c0aa925d1a95a2df7f1abc5c6d9ba6ca7abdc3173`. Status/classification are
`passed / frozen_rank16_float32_runtime_performance_characterized`; audit `26/26`.

The frozen formula remained `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`.
The deterministic dense CPU fixture was `float32[32,256]`, seed `20260902`, SHA256
`45ca137458856a7305ecf9500220c120aa5b4947e6f58d476b791e49b6337ab6`; direct-formula and repeated-call
replay residuals were zero for B=`1/8/32`. Fresh-process initialization manifest/payload/consumer/total
timings were `0.2792/9.1288/20.8566/30.286 ms`, with fresh-process wall `3657.2196 ms`; this was not
a true cold-disk claim. Fixed steady-state measurement used 50 warmups, 200 repetitions, CPU eager
synchronous `time.perf_counter_ns`, and linear `(n-1)*q` quantiles. Batch mean latency for B=`1/8/32`
was `0.5601775/0.8914525/3.277882 ms`; per-sample mean was
`0.5601775/0.1114315625/0.1024338125 ms`; mean throughput was
`1836.8588342/9338.7841702/9874.3015643 samples/s`. Full median/p95/p99/min/max telemetry remains in
the compact evidence.

Storage was manifest=`3219`, serialized=`2739448`, raw/runtime tensors=`2737408` bytes; process memory
delta/peak/RSS was `not_recorded`. Analytical operation counts were factorized/full-W=`644096/10240000`
MAC/sample with ratio `15.89825119236884`; `measured_speedup_claim=false`. GPU, full-W runtime and
end-to-end pipeline were `not_evaluated`. The actual observed Windows version was `10.0.26100` versus
the task's expected `10.0.26200.0`; the mismatch was recorded, while the registered Python/Torch/
Windows CPU execution stack passed. Physical cores and RAM were `not_recorded`.

S10 targeted/layout/related regression was `30 passed`; final full Windows-native pytest was
`1357 passed, 12 deselected, 25 subtests passed`, with one existing pytest-cache ACL warning.
Compileall, Ruff and `git diff --check` passed. Governance remains reconstruction/generative false,
`generation_readiness=not_ready`, `training_authorized=false`, and `formal_promotion=false`; Bridge
remains `invalid / cloud_sources_sync_blocked`. The only next action is
`manual_review_MPCDAR_R16_RUNTIME_PQ_result`.

## 2026-09-03：FINAL_MAINLINE_CLOSEOUT H-LOMA-CBOGT Scaled-Chen reconstruction-repair mainline

The final mainline closeout consumes the immutable authority chain
`RMAL → S7 MPCDAR → S8 F64 R16-IQ → S9 F32-NQ → S9V full-path visualization → S10 runtime
characterization`. Later consumers are not replaced by earlier attempts; all write-once attempts
remain available as dated provenance. This entry records no new scientific run, model execution,
CUDA run, evidence generation, or status mutation outside the closeout documents.

The current state is `MAINLINE CLOSEOUT`: scientific diagnosis `closed`, repair `confirmed`,
integration `qualified`, F32 `qualified`, visualization `completed`, and runtime performance
`characterized`. The hypothesis/experiment thread is `closed / resolved to current evidence
boundary`; current mainline next scientific task is `none`.

Frozen conclusion at the evidence boundary: task-relevant information survives the existing optical
path through CMOS; reconstruction failure is primarily supported as mapping/readout misalignment;
simple locality/raw Euclidean W-SVD is inadequate; useful mean/DC material is present; centered
detector information is better represented by a fit-derived data-aware basis; and registered MPCDAR
rank 16 is the first confirmed candidate in its preregistered powers-of-two family. Rank 16 is not described as a
minimum, intrinsic, or true rank.

The frozen formula is `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`, with S9
F32 tensors `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, `V16[40000,16]`, scalar count
`684352`, raw bytes `2737408`, and serialized bytes `2739448`. S8 remains integration-qualified;
S9 remains numerically qualified; S9V remains descriptive-only with no new Gate. The allowed
architecture wording is `optical front end → CMOS intensity sensing → lightweight frozen
electronic rank-16 readout`: hybrid optical-electronic reconstruction, not a native CMOS optical
image.

S10's primary CPU record remains Python `3.11.16`, PyTorch `2.5.1`, observed Windows `10.0.26100`,
threads `8/8`, logical cores `16`, formal CPU string
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; expected Windows `10.0.26200.0` did not
match. Batch mean/median/p95/p99 latency in ms for B=`1/8/32` were
`0.5601775/0.53135/0.784155/0.943756`, `0.8914525/0.82855/1.25965/1.362798`, and
`3.277882/3.21995/4.026955/4.434511`; mean throughput was
`1836.8588342/9338.7841702/9874.3015643` samples/s. Fresh initialization
manifest/payload/consumer/total was `0.2792/9.1288/20.8566/30.286 ms`, not a true cold-disk
claim; analytical MAC ratio `15.89825119236884` was not measured speedup.

Governance remains `reconstruction_manifold_supported=false`, `generative_manifold_supported=false`,
`generation_readiness=not_ready`, `training_authorized=false`, `formal_promotion=false`, and
`closed_task_relevant_bottleneck_not_supported`. Native optical target-like CMOS reconstruction,
real physical CMOS validation, Raspberry Pi benchmarking, sim-to-real detector alignment,
end-to-end hardware latency, GPU/full-W runtime, quantization, generative manifold and generation
readiness remain unestablished. Future work is independent and requires new authorization:
`REAL-HARDWARE DEPLOYMENT` / `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`, real CMOS calibration/alignment,
separately justified optimization, native optical redesign, or separate generative-manifold evidence.

After the final clean-head check, the bridge `refresh` stopped fail-closed with
`invalid Sources manifest: C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`;
`status` remained `invalid` for the same reason. `bridge_recovery_blocked=true`; no automatic upload
occurred and the existing package was not deleted or overwritten.
## H-LOMA-GEN-T1-RCAWR：Repaired-Coordinate BOGT Weight Registration（2026-09-04）

RCAWR 是 RCWV 的 analytic-only consumer。它只读取并 hash-bind R2 v2 calibration、historical
Set B preflight registration、Set A step-5000 replay context、parent step-5000 checkpoint identity
和 RCWV attempt-01；没有重新运行正式 gradient measurement，也没有 optimizer/backward/update、
training、checkpoint、refit、prior generation 或 decoder/CMOS/MPCDAR。

对 RCWV repaired v2 的三组 panel，先用 raw `G_BOGT/G_MSE` 验证冻结 effective ratio，再以冻结
serialized ratio 作为跨进程 Decimal source 重建 `K_i`。历史注册规则为
`alpha_i=0.10/K_i`、`alpha_registered=mean(alpha_i)`；结果为
`alpha=0.00017905883237862847`，三组 predicted ratios=`0.1166303093/0.0925572168/0.0941462530`，
全部通过 `[0.05,0.40]`，classification=`RCAWR_REGISTRATION_PASS`。这只是 repaired-coordinate
weight registration，不是 training readiness 或 generative-manifold evidence。

attempt-01 因 manifest 发布顺序检查过早而只留下 4 个中间 JSON，保留但不消费；正式 write-once
authority path 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_02/`，
输出树严格为 5 个 JSON；manifest、metrics、panel details、old/new comparison 与 compact report
均记录 source binding、provenance、state isolation 和 governance。下一动作固定为
`IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY`，治理仍为 closed evidence boundary、reconstruction/
generative unsupported、generation not ready、training/promotion=false。

## 2026-09-04：H-LOMA-GEN-T1-GAC repaired-coordinate generative alignment canary

This was one preregistered, bounded single-intervention CUDA canary. The initial clean task state was
branch `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1` at HEAD
`88a09b5f737f4eb9246ad97351b367a5e3a20841`; implementation/config/test commit
`5913a045b0bd932aff11e335f41a1f246b78a0f3` was created before the formal run. The run then consumed only
the exact step-5000 parent checkpoint, restored its bound Adam state, and trained 1000 steps to fixed
logical step 6000. No sweep, best-checkpoint choice, additional steps, calibration/prior/latent/BOGT/optical
change, prior/decoder/CMOS generation, FID/classifier/diversity/novelty, or G2 was performed.

The three-step real-update preflight passed, including phase-only updates, frozen OLS, exact optimizer
continuation, finite/resource checks, and a frozen-parent negative control that rejected the strict primary
Gate. The write-once authority is
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_attempt_01/`;
it contains 10 JSON/PT files and `17055800` bytes. Training manifest SHA256 is
`48d3e883355b329c80072c50cb2b06187d4cc7454911556f9b124115ab927630`; end checkpoint SHA256 is
`b8ed605d0ed4506c0415f8199ac16fd1f4274cc7efa419f00af785a2ce4f9542`. Runtime was `1694.4757 s` and
peak CUDA allocation was `2330944000` bytes. The repaired calibration payload remained
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`; the only trainable parameters were
`system.encoder.phase` and `system.decoder.phase`; start/end learned-state digests differed and OLS was
unchanged.

Endpoint G1-SA was R50=`1.8142581`, 95% CI=`[1.7888644,1.8446554]`, and R95=`1.0970011`, 95% CI=
`[1.0733978,1.1168799]`. The frozen parent lower bounds were `1.8142740` and `1.0679024`; therefore
the preregistered joint CI-separated primary Gate failed. Endpoint MRA selection/confirmation z means
were `1.0134035/0.9309632`, variances `1.4821789/1.3868347`, Gaussian KLs `3.0508981/2.6405313`,
numerical KLs `2.9807997/2.5035856`, complex ranks `12.2222137/12.8533897`, and power ranks
`8.2595968/8.4111090`. Mean-scale and rank safety passed. Current-sample safety failed because paired
MSE=`0.05678915` was above the template control=`0.04197700` (while below cyclic control=`0.07559964`).

The valid final classification is `T1_ALIGNMENT_HARMFUL`, with next action
`REVIEW_ALIGNMENT_TRADEOFF_FAILURE`. This is bounded training-direction/trade-off evidence only; it does
not establish a global training failure, a generative manifold, prior decoder accessibility, generation
readiness, or formal promotion. GAC and direct-dependency regression was `42 passed`; compileall, Ruff,
and `git diff --check` passed. Governance remains closed at the current evidence boundary with
reconstruction/generative support false, generation not ready, training authorization false, and formal
promotion false.

## 2026-09-04：H-LOMA-GEN-PCR-PSD repaired-coordinate prior shape decomposition observer

本条补录从旧当前状态入口迁出的唯一 PCR-PSD 历史事实。该观测只执行
`IMPLEMENT_PRIOR_SHAPE_DECOMPOSITION_OBSERVER`，起始/结束 HEAD 为
`02fb1f636aa17e0e2f7f01a3fa70275b8f19dbdf`，复用 RCR-A 的 selection/confirmation panel、数据身份、
MNIST preprocessing、100-port OLS、mode bank 与 repaired coordinate；没有重新选择 panel，也没有扩大到
G1 55k support。冻结 step-5000 checkpoint SHA 为
`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`，R2 calibration payload SHA 为
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`，RCR-A manifest/metrics/parent
panel SHA 分别为
`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`、
`5d2cc5c0c7b6e80f80d4f07239bb737f87327860381c0bf151cbaa686001b612`、
`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`。

registered factorized truncated-normal prior 的 sampler/config/KL/coordinate 语义检查通过；正式观测使用
selection=`1000`、confirmation=`5000`，matched-size CPU null 使用 seeds=`260904/260905`、
`128` replicates 和 `99%` central interval。两 panel 都显示 per-port marginal heterogeneity、
cross-port dependence 与 prior-defined tail/quantile deviation：between-variance fraction 为
`0.2639763/0.2228631`，最大 off-diagonal correlation 为 `0.8836253/0.8846894`，prior-null primary
percentile position 均为 `100%`。decomposition 为 `P-S3_SUPPORTED`、`P-S4_SUPPORTED`、
`P-S5_SUPPORTED`，replication status 均为 `REPLICATED_DIRECTION`。

最终 classification=`PCR1_PRIOR_CONTRACT_INTERNALLY_COHERENT`：registered prior contract 在
sampler/config/KL/coordinate 语义上内部一致，而 frozen step-5000 `mse_only` q(z) 不匹配该 target；
这不是 q 已匹配 p 的结论，不授权 per-port prior fitting、prior 修改或训练。权威输出目录为
`outputs/20260904_v2/runs/lightgen_scaled_chen_prior_shape_decomposition_v1_attempt_01/`，manifest、
metrics、compact report SHA 分别为
`228679fda073b2e4e1e46c931b743b490930dc895313e5c201283d244e22d97a`、
`4ea665c5a021a09c9ce2d44adfc31ed0cdb593de30b0ed3dbb3b10986bfbe9d2`、
`cbd0ceda6a878bcf295ef0252936edf35a6525937b76472610f2c9e384d85f94`；state unchanged，
optimizer/backward/refit/checkpoint/prior_fitted/decoder/CMOS/MPCDAR/training 全部为 false。专项测试
`7 passed`，相关回归 `36 passed`，compileall、Ruff 和 `git diff --check` 均通过；历史下一动作
`IMPLEMENT_REPAIRED_COORDINATE_G1_SA_REPLAY` 仅表示当时状态，不能覆盖当前 N3 阻塞或授予执行权限。

## 2026-09-07—09-08：H-LOMA-GEN N3 OLS-native research line

本节集中维护 N3 研究链的日期化事实、结果和被取代的下一步。`CURRENT_STATUS.md` 只保留当前
决策摘要，`current_handoff.md` 只保留最新增量与允许动作，`PROJECT_MEMORY.md` 只保留稳定导航；
它们不再复制本节的逐项实验正文。以下记录均为 research/diagnostic evidence，不改变 formal v2、
training authorization、generation readiness 或 promotion 边界。

### H-LOMA-GEN OLS-native branch genesis（2026-09-07）

`OLS_NATIVE_RESEARCH_BRANCH_ESTABLISHED` 建立活动分支
`research/lightgen-ols-native-semantic-generation-v1`，父分支为
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，branch point 为
`113ca1c2d1e2e2ffa458ebd9c2fb7ddf898c4831`。该分支是 OLS-native research mainline，
不是 formal v3；N1/N2 已完成，N3 是活动阶段，N4 未决，N5 blocked/future，N6 是独立冻结的
物理实现边界。建立动作没有执行 positivity、fit/SVD、rank selection、sampling、CUDA、forward、
evaluator 或 generation。旧 Scaled-Chen/BOGT、S1-S3、Code8、controller-writer 与
OLS-discovery lineage 保持为历史参考。`DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1` 仅为
当时的下一规划，已被后续 N3 记录取代。

### H-LOMA-GEN-N3 producer data authority（2026-09-07）

`OLS_PRODUCER_DATA_AUTHORITY_MATERIALIZED` 只读复用既有 S1 `R3_COMPLEX` source，物化了
500/500 行、50/class、`N=100` 的 `torch.complex64` fit 与 internal-validation banks。
partition manifest、data firewall、source/provenance alias 和 canonical checkpoint 均绑定在
`outputs/20260907_v2/runs/lightgen_ols_producer_data_authority_v1_attempt_01/`；S1 selection、
S1 confirmation、S2 validation 与 formal S3 panel 的独立 overlap 为 0，controller-repair
internal panel 对 N3 为 `NOT_APPLICABLE`。本条没有 optical forward、CUDA、optimizer、
backward、producer fitting 或 novel OLS generation；labels 仍是 provenance/partition metadata，
不是 producer input。

### H-LOMA-GEN-N3 low-rank model-order selection（2026-09-07）

`N3_LOW_RANK_MODEL_ORDER_SELECTED` 在冻结的
`R2_AMPLITUDE=|c|/sqrt(s)` 坐标、`d=100`、候选 `r=1..99` 和 CPU `float64` 下有效完成。
`r_min=99`，one-standard-error acceptable set=`[99]`，selected rank=`99`；结果入口为
`outputs/20260907_v2/runs/lightgen_n3_low_rank_model_order_selection_v1_attempt_01/model_order_result.v1.json`。
这只登记低秩振幅 baseline 的模型阶数，不支持 intrinsic dimension、compact latent structure、
Gaussianity、novel OLS 或 generation。

### H-LOMA-GEN-N3 amplitude producer fit and validation（2026-09-07—09-08）

`N3_AMPLITUDE_PRODUCER_BASELINE_FITTED` 固定 `r=99`、`d=100`、`n_fit=500`，只使用 fit bank
拟合 `mu_fit`、`W_99` 与 `lambda_1_to_99`；state 与 manifest 位于
`outputs/20260907_v2/runs/lightgen_n3_amplitude_producer_baseline_v1_attempt_01/`。这是 near-full-
dimensional strict-proper-subspace baseline，不是 compact low-rank 结构或 intrinsic dimension
结论，也没有自动授权 downstream sampling。

独立 validation `VALIDATE_N3_AMPLITUDE_PRODUCER_BASELINE_V1` 为 `valid_completed`，classification
为 `N3_AMPLITUDE_PRODUCER_BASELINE_NOT_SUPPORTED`。四轴为
`SCORE_STRUCTURE=CALIBRATED_MISMATCH`、`GAUSSIAN_BASELINE=CALIBRATED_MISMATCH`、
`POSITIVITY_FEASIBILITY=SUPPORTED`、`AMPLITUDE_SUPPORT=CALIBRATED_MISMATCH`；positivity 只证明
固定 proposal 操作可执行，不能推出 distribution、semantic generation 或 generation readiness。
producer state、rank、fit/validation partition、score law 和 data firewall 均冻结，不在该 bank 上
retune。

### H-LOMA-GEN-N3 mismatch attribution and score geometry（2026-09-08）

`ATTRIBUTE_N3_PRODUCER_MISMATCH_AND_SELECT_NEXT_FAMILY_V1` 的 classification 为
`N3_PRODUCER_MISMATCH_ATTRIBUTION_MIXED`。within-class even/odd observer split 下，partition
confound 没有按预注册规则成为唯一 dominant mechanism；class-centering 能改善部分 validation
radial/covariance 指标，但 centered dependence 与 radial mismatch 仍在。A/B/C/D 均未被唯一支持，
没有选择新 producer family；labels 仍是 diagnostic observers，未执行 label-conditioned producer、
refit、rank reselection、score sampling、phase-source、novel OLS 或 training。

`RESOLVE_N3_SCORE_GEOMETRY_ATTRIBUTION_V1` 的 authority 是
`outputs/20260908_v2/runs/lightgen_n3_score_geometry_attribution_v1_attempt_02/`，classification
为 `N3_SCORE_GEOMETRY_ATTRIBUTION_UNRESOLVED`，dominant mechanism observer 为
`DATA_PARTITION_OR_MOMENT_STABILITY_CONFOUND`。在 `z=(q-mu_fit)W_99/sqrt(lambda)`、`r=99` 下，
validation `R^2` excess 为 `37.6551453`，其中 population mean-shift 为 `1.4180415`、variance-
scale 为 `36.2371038`；来源顺序呈现每类一致的正向位置趋势（10/10 classes），因此不能把差异
唯一解释为 radial-mixture family。moment normalization 后仍有高阶 radial/dependence residual，
class-profile stability 也不足以选择 local-mixture 或 non-Gaussian-dependence family。没有注册
新的 producer family；N4、novel OLS 与 generation 继续 blocked。

前一轮 attempt-01 保留为 write-once provenance，但不被消费：post-run self-audit 发现 `N-1` 与
population radial identity 实现错误且 family-selection predicate 过宽；attempt-02 是唯一 authority。
这不改变原始 firewall、membership、rank、producer state 或任何历史 report。

### H-LOMA-GEN-N3 selector calibration lineage（2026-09-08）

旧的 `CALIBRATE_N3_ATTRIBUTION_DECISION_RULE_V1` 是 synthetic-only bounded calibration，
classification=`N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED`，NULL false-positive 为 `6/20=0.30`，
真实 N3 未加载，也未授权 real-data re-attribution。

随后执行的 `CALIBRATE_AND_CONFIRM_N3_ATTRIBUTION_SELECTOR_V2` 采用了历史 planning 草案中建议的
`200/200/100` replicate 预算、NULL FPR `≤0.05`、单侧 95% 上界 `≤0.10`、机制正确率 `≥0.80` 与
单侧下界 `≥0.60`；这些数值的权威来源是该 V2 consumer 的合同和报告，不是草案本身，也不因
文档合并而追溯授予下一阶段权限。V2 使用 synthetic-only、disjoint development/confirmation
seeds，在 confirmation 前冻结 selector specification；真实 OLS/N3 未加载。

V2 machine report 位于
`outputs/20260908_v2/runs/lightgen_n3_attribution_selector_v2_calibration_attempt_01/n3_attribution_selector_v2_calibration_report.v1.json`，
classification=`N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`：NULL=`0/200` false positives，
partition=`100/100`，radial=`100/100`，nonlinear=`82/100`，local=`78/100`，local 未达到预注册
`0.80` correct-rate target。因而 selector 不能声明 calibrated，不能消费真实 N3 做 family
selection，不创建 V3；当前 selector-refinement line 在该 MIXED 边界关闭。

### Durable boundary after N3 selector V2

没有 producer-family confirmation、真实 N3 attribution、new fit、rank/producer change、N4、
phase-source、novel OLS、optical forward、CUDA、optimizer、backward、training、evaluator 或
generation。`training_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false` 保持。若未来重新研究来源归因，必须使用 fresh confirmation source，
另立独立合同；当前交接入口不提供该科学授权。

### H-LOMA-GEN-N3 sealed confirmation source reservation（2026-09-08）

`DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1` 登记了一个 fresh balanced confirmation source，
classification=`N3_SEALED_CONFIRMATION_SOURCE_RESERVED`，状态为
`RESERVED_NOT_MATERIALIZED`。权威 machine manifest 位于
`outputs/20260908_v2/runs/n3_sealed_confirmation_source_v1_attempt_01/`；适用 source-index
overlap 为 0。该记录只建立 data-firewall 和一次性 future-consumer reservation，不是 OLS、
producer、rank/family、selector 或 generation 结果；完整 lineage 以 manifest 为准。

## 2026-09-07：H-LOMA-GEN-S3 controller semantic-geometry lineage

本节补录 2026-09-07 的 S3 controller-only 研究事实；它们属于已关闭的历史/diagnostic lineage，
不定义当前 N3 研究动作。完整 machine evidence、checkpoint 和 manifest 仍由各自 immutable
output 目录维护。

### Data authority and canary

`CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED` 完成 safe reload、protected sample/code
overlap=0 以及 no-update registration；此前 step-0 `INVALID_HARNESS` attempt 没有 optimizer、
backward、forward、CUDA 或 checkpoint 写入。其后一次显式授权的 controller-only、frozen-optical、
fixed-horizon canary 通过三步真实更新 preflight，执行 492 个 logical optimizer/backward steps，
只写 final-step checkpoint；仅 generator phase 参数更新，冻结 optical/evaluator/controller authority
保持隔离。首次 invocation 在 optimizer 前因 CUDA alias/allocator 条件停止，未写 artifact，保留为
write-once 非权威 attempt。

### Frozen S3 consumer and postmortem

独立 frozen S3 consumer 未训练、未创建 optimizer、未写 candidate checkpoint，classification 为
`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`。interface pass，但 selection/confirmation targeting
与 reachability 仍 fail；code effect 非退化、无 class collapse、duplicate=0、unique states=1000，
Jacobian 为 `CODE8_LOCAL_CONTROL_FULL_RANK`，state isolation 和 authority hashes 未变。相对历史
S3 的小幅 retrieval 变化只作 descriptive evidence，不作选优。

postmortem classification=`S3_ONE_SHOT_REPAIR_POSTMORTEM_COMPLETE`，primary attribution 为
`FIRST_SECOND_ORDER_GEOMETRY_OBJECTIVE_INSUFFICIENT_SUPPORTED`：centroid alignment 与 covariance
magnitude/coverage 改善，但 absolute reachability、covariance shape/tangent residual、gain
compression 与 class-conditioned compression 仍未关闭。没有足够证据支持全局 folding/topology 或
当前 controller parameterization limitation；class9 改善仅为 secondary。exact one-shot repair line
关闭，不 retune、second canary、S3 rerun 或 class9 repair，S4 blocked；`KEEP_8D_PROVISIONALLY`、
`training_authorized=false`、`generation_readiness=not_ready` 与 `formal_promotion=false` 保持。
