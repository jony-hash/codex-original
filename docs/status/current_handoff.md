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

## 当前注册假设：H-SCED

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

## 唯一允许的下一动作

H-SCED 当前路线已按有效失败关闭。不得继续 seed-8601 confirm、seed-8602/8603、
locked-final、O0-G、SMF、E1、fiber-aware training 或 formal promotion。任何未来
探索必须注册新的 electronic O0 假设、变量和 lifecycle，不得在本路线内调 epoch、
seed、权重或阈值。

除上述 H-SCED clean lifecycle validation 外，没有其他已授权研究或训练动作。
