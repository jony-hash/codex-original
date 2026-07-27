# 当前主线交接：identity v3 无蒸馏（2026-07-27）

当前活动运行时是 `rewrite/v2` 上的
`conditional_class_conditioned_modes_v3`（实现已写入工作区但尚未取得CUDA准入证据）。v1 的 LDM 蒸馏线仍由 `train.py` 维护，
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

当前没有正式训练授权。已实现但尚未运行的下一主线是
`conditional_class_conditioned_modes_v3`：每个类别拥有8个全视场物理模式，系数固定为
`tanh(z)`，删除Wz/Wy。必须先运行 shared-mode capacity conflict audit，再完成v3派生准入、
observability和唯一400-step smoke。当前机器无可用NVIDIA驱动，因此CUDA审计/光学smoke暂缓；
不得用CPU结果替代正式CUDA证据，不得启动984-step训练或数据集导出。
