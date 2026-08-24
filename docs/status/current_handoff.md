# 当前主线交接（2026-08-24）

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

H-CCF 实现提交于 `dcd9528`；首次 preflight 因 chunk-128/257 radial reference 的
`2.6e-7` 浮点差超过原门槛而按 `invalid_harness` 保留，修复提交为 `c306f46`，并从
新 clean HEAD 重建 validate lineage。最终 retry validation、preflight、audit、report、
closeout 的 SHA256 分别为 `5076dc…159d6`、`7bed53…4839`、`e89061…c2bf`、
`a0c5d5…21d7` 和 `702c6f…d61f`。

最终 audit 为 `status=valid, passed=true`，classification 为
`cycle_composition_attribution_underdetermined`，next action 为
`close_prior_native_attribution_family`。SWAE 与 SCED 的 full-bank source/final cycle
分别复现为 `0.01649122442/0.02642731884` 与 `0.01649122442/0.02250728432`，均在
`atol=1e-8, rtol=1e-6` 内；四个 stability banks、factorial identity、source/source
null swap、chunk equivalence、primary/MSE finite 与 hash isolation 均通过。SWAE 的
cycle drift 为 `0.00993609441`，SCED 为 `0.00601605990`；两条路线均未达到单一
encoder/decoder/interaction 或 mixed component 的稳定归因门，故不能把 cycle 恶化归因
到某一个组合因子。peak allocation 为 `147424256` bytes，storage 为 `50846` bytes，
optimizer_created=false、evaluator_calls=0，旧 artifacts/checkpoints 保持只读。

H-CCF report/closeout 是纯消费者，未创建候选模型。该结果关闭 prior-native attribution
family；不得启动 encoder geometry、decoder invertibility、interface Jacobian、hybrid
mediation、projected-gradient、SGD、trust-region、adapter、confirm、其他 seed、O0-G、
SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion。若继续研究，必须另行注册
无更新诊断假设并建立新的 lineage。
