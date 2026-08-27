# 当前主线交接（2026-08-26）

本文件是当前执行状态的稳定入口。架构合同见
[v2 runtime](../architecture/v2_runtime.md)，研究假设与唯一下一动作见
[capacity-map hypothesis ledger](capacity_map_hypothesis_ledger.md)，带日期的实验事实见
[experiment history](experiment_history.md)。2026-08-24 重整前的完整长篇记录保存在
[archived handoff](../archive/current_handoff_pre_sced_20260824.md)，仅供溯源。

## Formal identity 状态

当前正式架构仍为 `conditional_identity_logit_amplitude_v2`：`label + 8D code`
生成 amplitude SLM command，经完整 SLM FOV 的相干 4f relay、2+2 phase-only
finite-crop D2NN、complex identity OLS 和 B=0 backend relay，在 CMOS 完成单次曝光。
冻结 evaluator 只用于训练与验收，不属于最终光学推理。

正式 identity 链已在完整 raw-CMOS evaluation 上形成有效模型验收失败。该 checkpoint
不具备 promotion 权限，也不能授权 manual review、candidate export、memorization、
utility、dataset admission 或 accepted manifest。项目治理状态保持
`closed_historical_valid_failure`；旧 evidence 不补签为新的 `EvidenceRecordV2`。

固定边界如下：

- `formal_promotion=false`；
- formal optical training 未授权；
- fiber training、E1 和 modal regularizer 未授权；
- 不得重复已经关闭的 Objective A 正式链，或从 diagnostic checkpoint warm start。

## 已关闭的 research 路线

| 路线 | 当前结论 | 停止位置 |
| --- | --- | --- |
| Synthetic fiber route | `closed_synthetic_fiber_route_no_e1` | 无 E1、无 fiber training、无 formal promotion |
| Corrected WAE O0 | `wae_image_manifold_failed` | seed 7301 confirm 后停止 |
| Latent RealNVP transport | `latent_flow_semantic_failed` | seed 8401 confirm 后停止 |
| Class-conditional mixture transport | `mixture_latent_fit_failed` | seed 8501 screen 后停止 |
| Prior-native H-SWAE | `prior_alignment_fit_failed` | seed 8601 screen 后停止，未进入 confirm |
| Prior-native H-SCED | `prior_alignment_fit_failed` | seed 8601 screen 后停止，未进入 confirm |
| H-CCF cycle composition | `cycle_composition_attribution_underdetermined` | prior-native attribution family closed |
| H-DPGC direct-prior canary | `direct_prior_linear_semantic_capacity_failed` | linear canary closed; no neural training |
| H-DIDM direct image distribution | `direct_image_distribution_failed` on the sole authorized seed-8801 screen; route closed | same nonlinear decoder; no encoder/cycle; no confirm or promotion |
| H-SCF semantic code-source factorial | `semantic_counterfactual_safety_failed` | valid retry4 closeout; unique next action `close_frozen_decoder_derived_semantic_route`; no evidence-derived model, training, or promotion consumer |
| H-NDMC nonlinear direct manifold capacity | `nonlinear_manifold_safety_failed` | valid retry3 closeout; unique next action `close_nonlinear_direct_manifold_capacity_canary`; empirical canary closed, no architecture preflight or training consumer |
| H-JQOC joint quadrature optical capacity | `electronic_structured_input_capacity_failed` | valid retry7; input gate stopped before optical interpretation; no architecture preflight or training consumer |
| v3 spatial quadrature input | `hierarchical_spatial_input_capacity_failed` | valid retry6 electronic admission; secondary evaluator and diversity/coverage gates failed; optical/phase/receiver stages were not started |

H-SWAE 的 validation、synthetic direction audit 和 CUDA preflight 均通过。唯一执行的
seed-8601 screen 是有效科学失败：aggregate SWD reduction 为 `34.62% < 50%`，类别
6/7 退化且 cycle 指标变差。closeout 的固定下一动作是
`close_current_prior_native_swae_route`；该结论只关闭 H-SWAE，不证明 decoder、光学、
fiber、phase-only 或硬件的普遍不可行。

## 已关闭的 H-SCED 与新注册 H-PGC

下一研究假设是 `H-SCED`：以 Gaussian sliced Cramér/energy U-statistic 替换
finite-batch quantile-SWD，检验其是否避免每类小批量下的 variance-shrink optimum，
同时保持 reconstruction、cycle、novelty 和 dual-evaluator electronic O0 合同。

H-SCED 实现、配置和测试已提交于 `0d83a4d`；paired-batch preflight 修复于
`9290016`，lineage-bound lifecycle 修复于 `70b8e32`。validate、512-replication
synthetic audit 和 CUDA preflight 均已通过，且 evaluator calls 为零。当前尚无
screen、confirm、locked-final、O0-G 或 fiber 结果；这些通过项只授权预注册的
seed-8601 screen，不代表科学假设已通过或获得正式 promotion。

当前 immutable evidence 位于
`outputs/20260824_v2/runs/20260824_electronic_o0_prior_native_sced_v1/`：
validation `b6f4f4…16850`、audit `5e7884…f36b1`、preflight `2aea7a…75bb`。

seed-8601 screen 是完整有效的科学失败：training-bank aggregate SWD reduction
`39.73% < 50%`，reference-bank reduction `36.67% < 50%`，cycle
`0.0225073 > 0.0164912`；screen evidence、report 和 closeout 均保持 evaluator
calls 为零，且 closeout 固定下一动作为 `close_current_prior_native_sced_route`。
该结果只关闭 H-SCED，不证明 WAE decoder、光学、fiber、phase-only 或硬件的普遍
不可行性。

## 当前注册假设：H-PGC → H-TCA

H-SCED 当前路线已按有效失败关闭。不得继续 seed-8601 confirm、seed-8602/8603、
locked-final、O0-G、SMF、E1、fiber-aware training 或 formal promotion。任何未来
探索必须注册新的 electronic O0 假设、变量和 lifecycle，不得在本路线内调 epoch、
seed、权重或阈值。

H-PGC（Prior Gradient Conflict）是 diagnostic-only、无更新归因路线。它只读消费
WAE seed-7301、H-SWAE seed-8601 和 H-SCED seed-8601 的完整 valid closeout，比较
reconstruction/alignment/cycle 梯度、固定权重方向与解析约束投影；不创建 optimizer、
不加载 evaluator、不修改旧 evidence，也不能授权训练。H-PGC validation、audit 和
纯消费者 report/closeout 已完成，均为 valid；immutable SHA256 分别为
`ac2f28…b14f`、`7496ca…6beb`、`105d57…2476` 和
`d777af…e1e0`。最终分类为 `gradient_conflict_not_supported`：两个 projection
bank 的 SCED full-joint cycle rho 均未达到 `0.05`，H-SWAE paired arm 也未达到，
故不能把历史 cycle 恶化归因于一阶固定权重冲突。peak allocation 为
`821096448` bytes、run storage `1183663` bytes，optimizer/evaluator calls 均为零，
参数 hash、finite-difference、KKT 和 classwise 10 类报告均通过。下一动作只注册
`H-TCA`，不授权 projected-gradient、可逆 adapter、confirm、O0-G、光学或 fiber 路线。

H-TCA（Prior Trajectory/Curvature Attribution）已注册为独立的临时副本诊断路线。
它只读绑定 WAE seed-7301、H-SWAE/H-SCED seed-8601 final checkpoints 和 H-PGC
closeout；在不加载 evaluator、不修改旧 artifacts、不产生候选 checkpoint 的条件下，
精确重放两条 625-step 历史轨迹，并测量 landmark 的 raw gradient、AdamW 实际位移、
observer 方向导数、Taylor remainder 与 cycle HVP。唯一变量是路径归因机制：gradient
rotation、AdamW preconditioning、finite-step curvature 或 observer distribution
mismatch。有效结果只能被 report/closeout 和 hypothesis ledger 消费；不能直接授权
任何 optimizer 替换、trust-region、adapter、confirm 或训练。

H-TCA 实现及加固提交为 `84fbb7d`、`1ba714b`、`f18d71b`、`c616c7a`、`4486033`、
`3634221`、`6bd465c`、`44c032e`、`7929779`；早期 projection-bank、gradient layout、
classwise、history、terminal-landmark、attribution-gate 和 preflight-replay-contract
缺陷均按 fail-closed 保留，未消费旧 retry evidence，并从 clean HEAD 重建 lineage。
retry7 validation、preflight、audit、report、closeout 均 valid，immutable SHA256 依次为
`0f5c40…c2b`、`e1d6be…c024`、`8d17ef…aa44`、`b424f1…6707` 和 `53d0d5…3a9c`。

retry7 的 H-TCA 分类为 `trajectory_attribution_underdetermined`，
`alignment_estimator_independent=false`：H-SWAE 与 H-SCED 均未在两个 projection bank、
连续 landmark 和至少 3/4 observer batches 上支持 trajectory gradient rotation、AdamW
preconditioning conflict、finite-step curvature cycle drift 或 observer distribution
mismatch。两条 arm 的 encoder/decoder/optimizer/RNG/sampler/schedule 终态均 exact，SCED
13 个 history windows exact，14 个 landmark（含 625）完整；peak allocation 为
`1771866112` bytes，storage 为 `10217971` bytes，optimizer 只存在于临时 replay 副本，
evaluator calls 为零。该结果只授权 `register_denser_local_trajectory_attribution`，不
授权 optimizer 替换、trust-region、adapter、projected-gradient、SGD、confirm、O0-G、
光学或 fiber 路线。正式 promotion、
O0-G、SMF/FMF/MMF、E1、光学和 fiber training 继续禁止。

H-DLTA（Dense Local Trajectory Attribution）已完成其 diagnostic-only lifecycle。实现与
harness 修复提交为 `95e61c5`、`f3d5962`、`d5670e3`；首轮 SCED history-window mismatch
（audit `0eba96…6d3a`）按 fail-closed 保留，修复后从新 clean HEAD 重建 lineage。固定的
0–20、230–270、330–370、480–520 和 625 共 39 个 landmark、AdamW 四分量分解、
actual-step HVP/有限差分、reverse-direction negative control、paired replay/resume 和
SCED 13 个 history windows 均通过。H-DLTA validation、preflight、audit、report、closeout
的 retry evidence SHA256 依次为 `a2045d…0edf`、`1207f1…1502`、`d168f8…b18e`、
`797e5a…2506`、`a80ef1…bd55`；peak allocation 为 `3457404416` bytes，run storage
为 `22407988` bytes，evaluator calls 为零，旧 artifacts/checkpoints hash 未改变。

最终分类为 `dense_local_attribution_underdetermined`：H-SWAE 与 H-SCED 在 early、W250、
W350、W500 四个预注册窗口均没有任何机制同时满足三个连续 landmark、至少 3/4 observer
batches 和两个 projection banks 的支持门；没有 first-supported landmark，classwise
locator（0/250/350/500/625，含 class 6/8 定位字段）也不改变主分类。raw gradient rotation、
momentum、adaptive preconditioning、weight decay、finite-step curvature 和 observer sample
distribution mismatch 均 unsupported。下一动作固定为
`close_optimizer_trajectory_attribution_family`，只允许登记无更新 counterfactual hypothesis；
不得修改 optimizer、objective、阈值或架构，不得启动 projected-gradient、SGD、trust-region、
adapter、confirm、其他 seed、O0-G、SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion。

下一注册假设为 `H-CCF`（Prior Cycle Composition Factorial）：在不创建 optimizer、
不加载 evaluator、不生成 checkpoint 的条件下，将 source/final encoder 与 decoder 做
2×2 factorial 组合，分解 cycle 恶化的 encoder、decoder 和交互贡献。它只允许 no-update
counterfactual attribution，不重新打开 optimizer/trajectory attribution family。

## H-CCF 已关闭：cycle composition attribution underdetermined

H-CCF 实现提交于 `dcd9528`，direction/source consistency gate 修复于 `629e1aa`，
reconstruction locator 补齐于 `2d5a288`；首次
preflight 因 chunk-128/257 radial reference 的 `2.6e-7` 浮点差超过原门槛而按
`invalid_harness` 保留，修复提交为 `c306f46`，并从新 clean HEAD 重建 validate lineage。
最终 retry3 validation、preflight、audit、report、closeout 的 SHA256 分别为
`9163c5…a714`、`d605fa…2e2e`、`559af9…b476`、`30e629…107b` 和 `65eaa0…c342`。

最终 audit 为 `status=valid, passed=true`，classification 为
`cycle_composition_attribution_underdetermined`，next action 为
`close_prior_native_attribution_family`。SWAE 与 SCED 的 full-bank source/final cycle
分别复现为 `0.01649122442/0.02642731884` 与 `0.01649122442/0.02250728432`，均在
`atol=1e-8, rtol=1e-6` 内；四个 stability banks、factorial identity、source/source
null swap、chunk equivalence、primary/MSE direction consistency 与 hash isolation 均通过。SWAE 的
cycle drift 为 `0.00993609441`，SCED 为 `0.00601605990`；两条路线均未达到单一
encoder/decoder/interaction 或 mixed component 的稳定归因门，故不能把 cycle 恶化归因
到某一个组合因子。reconstruction objective 与 decoder image displacement 定位字段均已记录。
peak allocation 为 `147424768` bytes，storage 为 `416895` bytes，
optimizer_created=false、evaluator_calls=0，旧 artifacts/checkpoints 保持只读。

H-CCF report/closeout 是纯消费者，未创建候选模型。该结果关闭 prior-native attribution
family；不得启动 encoder geometry、decoder invertibility、interface Jacobian、hybrid
mediation、projected-gradient、SGD、trust-region、adapter、confirm、其他 seed、O0-G、
SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion。若继续研究，必须另行注册
无更新诊断假设并建立新的 lineage。

## Electronic O0 综合结论与 H-DPGC

跨路线的正式比较记录见
[electronic O0 research synthesis](electronic_o0_research_synthesis.md)。WAE 的主瓶颈是
双 evaluator 语义覆盖和类别召回；Flow、mixture、H-SWAE 与 H-SCED 只改善局部轴，未
同时满足语义、先验、cycle 和记忆化门槛。H-PGC、H-TCA、H-DLTA 未找到稳定的梯度、
AdamW、曲率或 observer 分布机制，H-CCF 又显示 encoder、decoder 与 interaction 存在
抵消，因此不再继续修补 prior-native 训练路线。

当前固定状态为 `prior_native_attribution_family_closed=true`，所有现有 checkpoint
均不得 promotion。H-DPGC（Direct Prior Generator Capacity Canary）已完成无 optimizer 的
固定 rank-8 logit-Gaussian canary；它只测试线性直接容量，不代表所有非线性直接生成器。

H-DPGC retry3 的 validation/preflight/audit/report/closeout SHA256 依次为
`af0994…54ef`、`890ad5…8818`、`7fc08c…f501`、`e3d573…be58` 和
`e61e4…d075`。三次早期 preflight 尝试分别保留了 coverage device、camera-label device
和 memorization device 的 harness 异常；每次均未形成 evidence，修复后从新 clean commit
重建 lineage。最终 audit/closeout 为 valid scientific failure：分类
`direct_prior_linear_semantic_capacity_failed`，next action 为
`close_direct_prior_linear_capacity_canary`。三个 locked panels 的 primary minimum recall
为 `0.76/0.80/0.92`，未达到 `0.90`；secondary minimum recall 为 `0.90/0.96/0.96`，
Jacobian rank-1e-3 pass fraction 为 `1.00`，exact duplicates 均为 `0`，但 memorization
pass fraction 为 `0.914/0.932/0.908`。official 10k primary/secondary accuracy 为
`0.9544/0.9761`，primary minimum recall `0.8768`，agreement `0.9470`。peak allocation
为 `517880320` bytes，audit storage `41238` bytes，optimizer-created 为 false。

该 valid failure 仅关闭固定线性 canary，不外推为所有非线性直接生成器失败；但在获得新的
无训练容量证据前，不得注册或启动 neural direct-generator training。正式 promotion、
confirm、O0-G、SMF/FMF/MMF、E1、光学和 fiber training 继续禁止。

## H-DIDM（retry3 历史状态；已由 retry4 closeout 取代）

H-DPGC 只否定了固定线性 rank-8 logit-Gaussian canary，未否定现有 nonlinear decoder
在直接分布训练下的可行性；当时注册的 H-DIDM（Direct Image Distribution Matching）为：
只读 warm-start WAE seed-7301 decoder，删除训练期 encoder、reconstruction、latent
alignment 和 cycle，直接用 `label + 8D N(0,I)` 生成 O0-I image，并以固定多尺度
classwise sliced-energy U-statistic 对齐真实图像分布。

H-DIDM 是 diagnostic-only 训练结构实验。evaluator 不进入 audit、训练、权重或 checkpoint
选择；不改变 latent 维度、decoder backbone、光学结构或最终单次推理接口。retry3 的
direction audit/preflight 曾通过，但后审发现 preflight 在 step 3 后没有重放更新，audit
也未绑定 collapse/blur/label-shuffle 与 chunk/formula controls；这些 evidence 现标记为
superseded/invalid_harness，不形成科学结论，也不再授权 screen。

retry3 的局部证据不得授权 encoder geometry、decoder invertibility、interface Jacobian、
hybrid mediation、projected-gradient、SGD、trust-region、adapter、confirm、其他 seed、
O0-G、SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion；其 screen 需另行明确
授权并沿用 immutable manifest lineage。最终 retry4 结果见下方 closeout。

## 2026-08-24：H-DIDM retry3 harness evidence 废弃

H-DIDM 实现提交为 `db391ef`；保护门修复分别提交为 `85c155f`、`e78951d` 和 `66d974f`。
按 fail-closed 规则，修复后从 clean HEAD `66d974f` 重建 retry3 lineage；此前三次
validation/audit/preflight 尝试保留但不被消费。

retry3 validation、audit 和 preflight evidence 的 immutable SHA256 分别为
`5b0f35aea831cebbece92ccd22a20607ab29978d0c5f9a8fc1c4497475540543`、
`4a3f350bc79a1a80f9464a1a3dd5552fe5dd53a8bf203163d5968554c20b58a3` 和
`c88c7b2911ea209a76bee10421023a359f0f3e97d59413a6d1df81a39ee72cd4`；这些证据现为
`superseded/invalid_harness`，screen manifest 不再可消费。

审查确认 audit 未绑定计划要求的 real-vs-real matched、class-mean collapse、blur、
label-shuffle、chunk invariance 与朴素公式 controls，因此不形成 direction-admitted 结论。

preflight 虽记录了三步更新和保护指标，但所谓 interrupted/resumed replay 在恢复后执行
了零步，未证明 step-2 checkpoint 到 step-3 的真实续接，因此按 harness 规则废弃。

该历史下一动作已由 `727568b` 的 controls、真实 replay、625-step checkpoint/screen lifecycle
修复执行；retry4 lineage 与最终分类见下方 closeout。retry3 evidence 不得被消费。

## 2026-08-24：H-DIDM retry4 screen closeout

retry3 的 audit/preflight 已按 fail-closed 规则标记为 `superseded/invalid_harness`；它们
不形成科学结论且不被 retry4 消费。修复提交为 `727568b`，在 clean HEAD 上重新建立
`validate → audit → preflight → screen` lineage。retry4 的 validation、audit、preflight、
screen、report、closeout SHA256 分别为：

`1c38fcebe218823246ec9f253769bf95f15049ce0657d7faaffd6992ea333862`、
`c76d69c974ca6ed89ef19935c358688495b3c4234c91f1ae6a8e6f9ea3303b99`、
`338e150c10c0cd8f1b8d20eccf6848900be28ea59080cbe838af51425f1af363`、
`f2ccc0cb7ce72c8e6866bac5fa48be3d873cd636ecc52e47f2eb31374041d369`、
`7e1626c98a089f4db632d0170bcb9f430cdad6e4f549ec45083c9d3ce2c7afa9`、
`d65a36cdc7637f76799c5bd763bf2ea51f1973bea55cd55ed7f6845bbb07fab3`。

retry4 audit 绑定了 real-vs-real、class-mean collapse、blur、label-shuffle、朴素公式、
chunk 和 projection-determinism controls，四个 batch 的 `-g` 双 bank 方向通过，且
optimizer/evaluator calls 均为零。三步 preflight 的 step-2→step-3 interrupted/resumed
replay、decoder/optimizer/RNG/sampler/schedule/history exact gate 通过；两个 observer
bank 均下降，peak allocation 为 `1515885568` bytes，storage 为 `64606` bytes。

唯一授权的 seed-8801、625-step full-coverage screen 正常完成，training checkpoint
schema 为 `electronic_o0_direct_image_distribution_training.v1`，训练期间 evaluator calls
为零。两 projection bank 的 aggregate distribution loss 均低于 source，且十类均未退化；
但三 locked panels 的语义门全部未同时通过（panel pass count primary/secondary 为
`0/0`）：primary accuracy 为 `0.918/0.910/0.894`、minimum recall 为
`0.72/0.78/0.78`，secondary accuracy 为 `0.898/0.896/0.870`、minimum recall 为
`0.80/0.80/0.74`（panel 顺序 `4242/104729/161803`）。memorization pass fraction
为 `0.998687`、exact duplicates 为 `0`，Jacobian rank pass fraction 为 `1.0`，camera-code、
diversity、saturation、coverage 和 label-shuffle 保护门通过；official 10k 仅作完整报告，
不参与选择。screen 为 `status=valid, passed=false`，固定分类
`direct_image_distribution_failed`，next action 为
`close_current_direct_image_distribution_route`。

screen decoder/training checkpoint SHA256 分别为
`c066349955608e2c170f7732a7ecc63be4fb97f6673271373016544b10b5b6f2` 和
`df58662371220c5305e48b65ea71f51378f8637860fcc2e7ead3172f4fa72f6a`；screen peak allocation
为 `646068224` bytes，storage 为 `18826955` bytes。report/closeout 为纯消费者，旧
WAE、H-DPGC 与全部 prior-native artifacts 保持只读。该 valid failure 关闭 H-DIDM，禁止
重跑、延长 epoch、调 loss、换 seed、confirm、decoder v2、O0-G、光学、fiber training
或 formal promotion；若继续研究必须另行注册无训练证据和新 lineage。

## H-SCF closeout（valid scientific failure）

H-SCF（Semantic Code-source Factorial）已完成 retry4 的
`validate → preflight → audit → report → closeout`。五阶段均为 `status=valid`，最终分类为
`semantic_counterfactual_safety_failed`，唯一 next action 为
`close_frozen_decoder_derived_semantic_route`。这是一项 diagnostic-only、no-update 的有效科学
失败，不授权任何训练、confirm、decoder v2、O0-G、光学、fiber 或 promotion。

最终 evidence 位于
`outputs/20260824_v2/runs/20260824_electronic_o0_semantic_code_factorial_v1_retry4/`，SHA256 为：

- validation `252b0ebf226859e618c3fdb342a69e80269098bc09d1deebe4b411378d1ab491`
- preflight `25fd79f3b1194299c32f7c0e963849523caf3d4f3cfb596c461bb9a72da05104`
- audit `aca75439aa2fce3f5414f697f34505e91bcc0052c9532aee8ff8b8e24eaef930`
- report `4a60eda420884d7a19d30aa6712482de4f4394aec6cac7e9c8035b3f6c0da9f6`
- closeout `59d927f7d7ed3b9633fe613e5b9ce14814ba9b5e15776e9ef47352594026ef94`

preflight 资源为 peak allocation `720658432` bytes、storage `10316` bytes、wall time
`9.480835378999473` seconds；audit peak allocation `720658432` bytes、storage `110601` bytes。
历史三 panel 的 view hash 与双 evaluator correct/recall/agreement counts、posterior hashes、
finite、64/257 chunk、参数不变和 `optimizer_created=false` 均通过。

audit arm 摘要：S-G 两 Gaussian banks 的 safety gate 均失败；S-P 两 posterior banks 的
semantic gate 均通过但 safety gate 均失败；D-G 两 Gaussian banks semantic gate 均失败；D-P
两 posterior banks semantic gate 均失败，within-class permutation 两 banks 也均失败。计数为
S-G `2330/2115/2117`、`2320/2129/2121`，S-P `2466/2463/2454`、`2459/2462/2442`，
D-G `2270/2244/2303`、`2290/2240/2308`，D-P `2311/2321/2364`、`2305/2329/2381`
（每项依次为 primary correct、secondary correct、agreement count）。分类按 safety 优先级
停止，不将 D-P 的后续失败重新解释为单一 decoder 或 pairing 结论。旧 WAE、H-DIDM 和 evaluator
artifacts 仍为只读；本轮不执行任何下游路线。

## H-NDMC closeout（valid scientific failure）

H-NDMC（Nonlinear Direct Manifold Capacity）已完成 retry3 的
`validate → preflight → audit → report → closeout`。retry1 因 classwise training
bank 长度不等而错误 `stack`、retry2 因参数 hash 符号引用错误，均在 evidence
创建前作为 `invalid_harness` 保留；修复后从 clean commit `acc22f5` 重建
lineage。

retry3 五个 evidence 均为 `status=valid`，最终分类为
`nonlinear_manifold_safety_failed`，唯一 next action 为
`close_nonlinear_direct_manifold_capacity_canary`。audit 使用 exact classwise
PCA-8、Gaussian copula、连续全同类 kernel reconstruction、两 Gaussian banks、
identity-copula ablation、H-DPGC historical replay、zero-code、label-shift、
64/257 chunk、finite-difference Jacobian 和 curvature controls；不创建
optimizer、checkpoint、candidate 或生成图像全集。

retry3 evidence 位于
`outputs/20260825_v2/runs/20260825_electronic_o0_nonlinear_direct_manifold_capacity_v1_retry3/`，SHA256 为：

- validation `3d3a8cf1c3d30e3923172bc937b6f21295083d2d8963815de1a1ca9e37480d32`
- preflight `6e0a52880b6e222e79ce9a9b63f368f0d9fb6afb62b73f19297aed0114c45783`
- audit `e0fdf0b32fff2159c735f92f67c4c621eaca02aac8c26270fc82e506d5aa6544`
- report `b82d5379863068a8dfa22fcb362e36a1dcfaf37305c1692f728b9719007f267e`
- closeout `f0e21b76723c187056cfb930c306036fbf58e390ebadbda96ccf04a8dcbee43b`

两正式 bank 的 primary correct 均为 `2500/2500`，secondary 为 `2000/2500`，
agreement 为 `2000/2500`；camera-code fraction 为 `1.48e-5`，primary diversity
约 `0.0042–0.0070`，coverage 为 `1.67–2.13`，因此安全门失败。memorization
pass fraction 为 `1.0`、exact duplicates 为 `0`；preflight peak allocation
`922167808` bytes、wall time `14.303609110007528` seconds、storage `10120`
bytes；audit storage `54366` bytes。该结果关闭 H-NDMC 经验流形 canary，不能
授权 direct-generator architecture preflight、neural training、confirm、decoder
v2、O0-G、optical、fiber 或 promotion。

## H-JQOC closeout（valid scientific failure）

H-JQOC（Joint Quadrature Optical Capacity）已完成 retry7 的
`validate → preflight → audit → report → closeout`。retry1–retry6 的 harness
失败均保留在各自 attempt，且在 evidence 创建前或下游 consumer 创建前停止；修复内容仅限
量化分位数、reference panel device、tuple 解包、evaluator count device 和 qualification
scalar serialization。最终从 clean lineage `266083e4752d947cbbd28dd16d8b0152ad4a5e4c` 重启。

retry7 五个 evidence 均为 `status=valid`，最终分类为
`electronic_structured_input_capacity_failed`，唯一 next action 为
`close_joint_quadrature_capacity_route`。两份正式 bank 中，8D/16D/32D 的 primary
minimum recall 分别为 `0.73–0.78`、`0.69–0.78`、`0.57–0.69`，secondary minimum
recall 分别为 `0.92–0.98`、`0.96–0.98`、`0.91–0.95`；三个维度均未通过预注册
electronic semantic/safety gate。因此不能用后续 raw-intensity 或四端口 I/Q 结果补救输入
合同。raw-intensity primary/secondary accuracy 仅为 `0.075–0.092`/`0.421–0.436`；
I/Q 的 port packing、energy identity、LO-off、sign/swap、replay、finite-difference rank
和资源门通过，但这些结果只属于诊断记录。

retry7 evidence 位于
`outputs/20260826_v2/runs/20260826_joint_quadrature_optical_capacity_v1_retry7/`，SHA256 为：

- validation `f8a6231b9910cc3cab0a95c61594237ec4e72c3fa2c02e9d6891556a70e11d5a`
- preflight `43351251e919981c98eb37698a01b660ed87a3b7ded4734f51eb039b8d8ae7e0`
- audit `08b0eb64c7a02e4b0ebec7dd3f7f23818abc98d29431033efcf3cd5b17296fa7`
- report `c56d4bb4d172753ab6c97d7d40171628f3a823fecdfa96deb1ef6a3aa227b982`
- closeout `07806d105e8745a83fba65cc65258adb8733c8da53ed8d1a0c70e574dd69699c`

preflight peak allocation 为 `3353651200` bytes、wall time `32.109134116` seconds、
storage `11065` bytes；audit peak allocation 为 `4788746752` bytes、wall time
`260.960760050` seconds、storage `87355` bytes，均低于预注册资源门。结果关闭当前
joint-quadrature route，不授权 intensity 或 quadrature architecture preflight、neural
training、checkpoint、confirm、decoder v2、O0-G、optical、fiber 或 promotion。

## v3 spatial quadrature closeout（valid scientific failure）

`conditional_spatial_quadrature_v3` 已完成 electronic preflight、generator-only
training 和双 evaluator electronic admission。retry1–retry5 的 harness 失败均
保留且未创建下游 consumer；retry6 在 clean lineage `e1b16c8` 上通过 preflight
后完成 admission。该 admission 的 classification 为
`hierarchical_spatial_input_capacity_failed`，唯一 next action 为
`close_spatial_quadrature_v3_input_route`。

两份 250/class bank 中，primary accuracy/minimum recall 均为 `1.000/1.000`；
secondary accuracy/minimum recall 为 `0.900/0.176` 与 `0.896/0.116`。secondary
diversity ranges 为 `0.401–0.611` 与 `0.385–0.614`，coverage 为 `1.551` 与
`1.532`，均未达 electronic admission gates。primary code fraction 为
`0.3226/0.3250`、coverage 为 `1.001/0.997`、memorization anomalous rate 为
`0/0`，但不能抵消 secondary 失败。

本轮没有执行 quadrature preflight、400-step smoke、phase-mask 更新、联合训练、
formal-test、candidate export 或 promotion。v3 checkpoint 仅是 electronic
training artifact，不兼容旧 identity checkpoint；所有 v2 identity、H-DIDM、
H-SCF、H-NDMC 和 H-JQOC evidence 保持只读。

## 2026-08-26：Package A electronic distribution recovery closeout

Package A（Electronic Distribution Recovery）在 clean commit `a1842b8` 上完成
`validate → gate-audit → preflight → audit → report → closeout`。retry12 的六份
evidence 均为 `status=valid`；H-GFA 双 evaluator gate、三步 CUDA preflight、
H-LIR/H-LGS geometry 与 finite-difference harness、负对照和资源门均通过。
完整 audit 的有效科学分类为 `generator_range_incomplete`：R/C/U 三个 inversion arm
在两份严格互斥的 250/class validation bank 上均未同时满足双 evaluator semantics、
agreement、diversity、coverage 和 memorization 组合门。H-GFA 本身通过，label-shift
负对照被拒绝；这是 valid scientific failure，不是 invalid harness，不支持继续调参或换 seed。

六次 bank/arm 测量中，primary accuracy/minimum recall 为 `0.9888–1.0000`/
`0.9600–1.0000`，secondary 为 `0.8956–0.9172`/`0.1160–0.5360`，agreement
为 `0.8956–0.9168`；R/C/U 的两 bank `range_gate` 均为 false。memorization anomaly
为零，generator state 前后 hash 完全一致，geometry/finite-difference gate 通过，
未创建模型 checkpoint。

retry12 evidence 位于
`outputs/20260825_electronic_distribution_recovery/runs/20260825_package_a_v1_retry12/`。
SHA256 依次为 validation
`3d4c71193455c5a1d8afcd62569f4e08dbc741046793418ff8a6841f74340737`、gate-audit
`99de4cae5f83cda643fd5accd09e54795d24b74d1629539f033f4156e47ac0bf`、preflight
`be07633ca2deffec1fc3b65226172743c7a0770a00b13087b420516ceee09e06`、audit
`2253c9ed0f4284b2e81818ba6089fee1ecb82430989edfa9cf678b1c84c957dd`、report
`fc2d110903b3eaa0a42234a646d4be9249e2ebe2962f58254c2e1dfb9048ae36`、closeout
`70ed0fe02e9242ec630b44f3e46dd386950ef8d10d1ab5769664b5cb211a8cab`。

Package A 的真实消费者是 Package B teacher audit 与未来 student 架构选择；本轮不授权
Package B 自动启动，必须另行审查并建立新 lineage。DDPM、student、optical、phase、
quadrature、OLS、fiber training、promotion 和 candidate export 均保持禁止。

## 2026-08-27：Package B teacher distribution admission closeout（valid scientific failure）

Package B（DDPM–VAE Teacher Distribution Admission）在 clean producer commit
`381941f` 上完成 v1 `lineage → preflight → generate`，随后在 clean consumer
commit `ac7baec` 上完成 v2 `validate → audit → report → closeout`。producer
锁定并验证了 first-stage checkpoint `e553bf…afdd`、matched diffusion
checkpoint `60f759…cd0`、2026-06-03 source runs、acceptance manifest、Package A
H-GFA evidence、双 evaluator qualification 和 MNIST split。完整 Gaussian latent
noise、200-step DDIM、guidance 2.0 以及两个独立 250/class bank 均按 manifest
生成；生成过程 320/320 shard、producer/preflight 无 evaluator calls、无 optimizer
或 checkpoint，preflight peak CUDA allocation 为 `173581824` bytes，generation
peak 为 `197254656` bytes。

lineage/preflight/bank/consumer evidence 位于
`outputs/20260827_teacher_distribution_admission/runs/20260827_package_b_v1/`，
SHA256 为：

- lineage `b1e385c2b3cba80b91c81e958c1a26804a39554c09daddb6bca0ba5e73bb1a6c`
- preflight `66b65fb5ccd04951201a9634fe101e4cff982f2c1a9d9cecd6c3abd6a280d922`
- teacher banks manifest `c9aebd1e21e8ed185eb1c0db35902a7a4c67c90d01ebd25153a45ca496139ccb`
- validation `ad9f4896f8d43fa03230c5e0d00b10c7f7c5b0e0c590a171f76b4d8a2084f099`
- audit `0369e9500158a94a675e393674f3deff6edf8de0e5d6775e617750b17dbfea92`
- report `0cdd29f6b92656355165926e0ef530fa1e31b40444875476b7c8ca553d1dea5e`
- closeout `2620311e2bedb6ba6188afc2eae5af5fb9947410e8d70dac892971e2fdc3c6f7`
- synthesis `66f29aff3bf5c915c25195f83914c265335a6f5068ed3e9dbec71b64770b97ec`

The audit was `status=valid`, with negative-control rejection, finite values,
state/lineage/hash binding and resource gates passing. Its scientific classification
is `teacher_distribution_failed`, first failed gate
`teacher_dual_evaluator_range_gate`, and the sole permitted closeout action is
`close_teacher_route`. Bank 0 had primary/secondary accuracy `0.3524/0.2212`,
agreement `0.1676`, primary/secondary coverage `3.0909/2.8327`; bank 1 had
`0.3428/0.2248`, agreement `0.1716`, and coverage `3.1483/2.8622`. Primary
minimum recall was `0.036` for primary and `0.008` for secondary in both banks; multiple diversity values also missed the
registered `[0.70, 1.30]` range. Memorization anomalous rate was `0` in both
banks, but does not rescue the failed semantic/range gate. The cyclic label-shift
negative control had accuracy `0.0598` and was rejected. Audit peak CUDA allocation
was `276411392` bytes over `2300` evaluator calls.

This is a valid teacher-distribution failure, not an invalid harness. The named
consumer is Package C's multiscale student admission and architecture decision;
that decision is rejected by this evidence. Package C student distillation is
forbidden, and the current DDPM–VAE teacher route is closed pending a separately
designed teacher route with new lineage. DDPM/VAE/student/optical/phase/quadrature/
OLS/fiber training, promotion and candidate export remain forbidden. No Package C
or other downstream stage was started.

## 2026-08-27：Package B2 teacher failure attribution closeout

Package B2（Teacher Failure Attribution）在 clean commit `36597b9` 上完成独立的
`lineage → preflight → generate → validate → audit → report → closeout`。本轮锁定
Package B 的 matched first-stage checkpoint `e553bf…afdd`、matched diffusion
checkpoint `60f759…cd0`、原 Package B banks、Package A H-GFA banks、双 evaluator、
MNIST split 和全部正式阈值；没有搜索、选择或训练新模型。

正式结论为有效科学归因 `teacher_first_stage_distribution_failed`，首个失败门为
`first_stage_reconstruction_distribution_gate`，不是 `invalid_harness`。R（真实
MNIST 正对照）两个 bank 同时通过；primary accuracy/minimum recall 为
`0.9880/0.9640` 与 `0.9872/0.9440`，secondary 为 `0.9888/0.9800` 与
`0.9872/0.9720`，agreement 为 `0.9868/0.9852`。cyclic label-shift 负对照被拒绝，
R/V/D accuracy 分别为 `0.0006/0.0024/0.0598`。

V=`decode(encode(real).mu)` 在两个 bank 都未通过：bank 0 的 primary/secondary
accuracy/minimum recall 为 `0.9692/0.8920`、`0.9444/0.8520`，agreement
`0.9316`；bank 1 为 `0.9704/0.8960`、`0.9464/0.8840`，agreement `0.9380`。
D 与锁定 Package B teacher replay 完全一致，并重现其 failure（bank 0/1 的
primary/secondary accuracy `0.3524/0.2212`、`0.3428/0.2248`，minimum recall
`0.036/0.008`，coverage 约 `3.09/2.83` 和 `3.15/2.86`，agreement 约 `0.17`）。
由于 V 先失败，B2 不把 D 单独解释为 diffusion failure。evaluator-independent
latent statistics 已记录；normalized encoded-real `mu` 与 DDPM latent 的
classwise mean L2 distance 覆盖约 `5.03–11.08`。

v3 evidence 位于
`outputs/20260827_teacher_failure_attribution/runs/20260827_package_b2_v3/`：
lineage `9c91e346127576143fe49bd7ea5f14b479ffd8647cce5b4b24c818bee0db2b00`，
preflight `c67d195f9c7ce1f9917982c3fc747fbf9a73ddabee0eb049431a84d039a6ad3b`，
generation state `48b3115441713d0203978403e97f87b2f0fe7a1173f17ed66e72b6e2ae31e1d1`，
artifact `5f018c0f261d9bd17f6be19cd6f242b1cc2f980caa01200c2664c05d8b3bca0f`，
validation `f8455fb8f369e9e96a53bca532c3862c1bfae5c5b0b8323cbc1d1de6e34aa307`，
audit `fffd9be3a03880b76780f168ce54c7aea3bda8193db7c4459c9cad398b8065e3`，
report `5ad140a9ee8cb431b50c258d6141159fa69b4acb24364ccadb5c10146bfa8fdb`，
closeout `0fed705e605fd540309e6904a4214eb7de9f668d2d1496868087e9fda6e58515`，
synthesis `3a70799aea9dad02f9126b36b5337f6e6f21c9ab54de2c9792268fcf7266312f`。

此前 v1 preflight 的 batch-10/batch-16 replay mismatch 和 v2 artifact 漏写 labels
均已 fail-closed 且未被消费；修复后 v3 重新生成 320/320 shards。producer/preflight
无 evaluator calls，生成峰值 `213146112` bytes，audit 峰值 `276390912` bytes，
state hash 前后不变。

Package B2 的真实消费者是 teacher redesign 决策；唯一允许的下一步是另立
`Package B3-VAE`，重新设计/训练 first-stage，并先通过 reconstruction distribution
gate。当前结果不授权 `Package B3-DDPM`、Package C、student/teacher/optical/
phase/quadrature/OLS/fiber training、candidate export 或 promotion。若未来 first-stage
通过，仍须用新 teacher lineage 重新通过原 Package B 双 bank admission，只有得到
`teacher_distribution_qualified` 才可审查 Package C。

### B2 evidence-schema amendment

上述 v3 目录仅保留为未消费的 schema-diagnostic run：其 producer manifest 缺少
`seed` 和 `input_artifact_hashes`，因此不承担最终科学证据。修复后的最终权威 run
是 `outputs/20260827_teacher_failure_attribution/runs/20260827_package_b2_v4/`，在
clean commit `9dddb78` 上完成全流程；其 attribution artifact 已通过顶层 required
evidence-field 核验。v4 的 lineage/preflight/state/artifact/validation/audit/report/
closeout SHA256 依次为：

`4fae3feabb1bac774827e4c46719095d5084e75ab41df2789df5cb8906587ae3`、
`812e28350315bddb26023d8f56dd40b65a306d83c68c2ecf119e1b8081e73643`、
`3f7915386bfe8b2c78b0a9aef8a549b03e5165ce609def3609eb51d0c4684a1d`、
`af04f5824784943b0d3d0a33f2404802644f4d96bb7cbb9dfb53ef2d56683c45`、
`9b9a79d3efcd6fb1b8f10eac152acf2d72cf27d6371742999deee48749eae6bc`、
`a1d64208e3c55cc468229ca11552f43dc55f93c2dae3e163435b06bcdd06a71f`、
`aecb9b9a4f3a8e521c22b3cf51f11330fcf25678d3788502caebac0742e9b0e7`、
`c760ddcc76f378f5517e280d876edaa9015bc3ea7f6216f846ecaf1813388c75`。

## 2026-08-27：Package B3-T.2 FP32 semantic-preserving compute cleanup closeout

Package B3-T.2 was completed as a research-only throughput diagnostic. It replayed
the locked B3-T1 CUDA contract (8 warmup batches, 200 measured training batches,
100 development batches, FP32, microbatch 16, gradient accumulation 4) before
reviewing synchronization, exact same-batch reuse, and Python/module dispatch.
The T1 baseline replay was reproducible: total profiling-window deviation was
`1.93%`, development deviation was `2.16%`, and all registered phase tolerances
passed.

The authoritative run is
`outputs/20260827_teacher_first_stage_recovery/b3_t2_v3/`, executed from clean
commit `83e0d74`. Its classification is
`b3_fp32_semantic_preserving_optimization_insufficient`; this is an engineering
throughput result, not a B3 model-quality result. The final FP32 projected epoch
was `31.135 min` versus the locked `25 min` contract, with total replay ratio
`0.981x`. Backward remained `76.86%`, model forward `15.92%`, and measured CUDA
synchronization wait `8.85%`. The low-risk candidate matrix accepted no new
intervention: safety finite checks remain fail-closed, profiler-only synchronizes
are not formal training overhead, and the already-small data/host/loss portions
do not justify unscoped work.

The run remained resource-safe with CUDA peak `2.589 GiB`, host RSS peak
`1.751 GiB`, finite values verified, inherited T1 resume exactness, and zero
evaluator calls. The T2 implementation commits are `32e2b34`, `b3a6253`, and
`83e0d74`. Earlier `b3_t2_v1` and `b3_t2_v2` directories are retained as
non-consumed diagnostic records: each completed the GPU profiling window but
hit a T2 evidence-writer programming defect, so neither is a timing or model
conclusion and neither was spliced into v3.

The six required v3 evidence hashes are:

- baseline replay `64a531ac55a5b8059ad28ac3f040c96c036766db01cffdaa9877dc646eb27dd8`
- synchronization attribution `ab932c6f3da07bbf5c31d69132902eee8663405064992f46aaeefc864b00abc5`
- optimization matrix `a0d3f6bfaa555cd2ab3ce0d3d5a0392f1e8914b26329d246902ab1fa7668c86a`
- equivalence evidence `7e6674e99bee213e77b0927b021653b530882cd7f9e99abe58ae9c291572d484`
- final timing `2447c30fe2cb0082c4974097dd950f33e101286299d351d39b2aa987156d26b4`
- closeout `351b6175deb930f1d7bf6d70ca9f3726f9fee34c0aaaa5f74713744a342341d6`

The only permitted follow-up is a separately reviewed
`numerical_gpu_acceleration_preflight`. T2 itself does not authorize AMP,
BF16/FP16, TF32 enablement, `torch.compile`, formal B3-VAE training,
B3-DDPM, Package C, student/optical/phase/quadrature/OLS/fiber training,
candidate export, or promotion. The T2 closeout is a hard stop.
