# 当前主线交接：identity v2 无蒸馏（2026-07-24）

当前活动运行时是 `rewrite/v2` 上的
`conditional_identity_logit_amplitude_v2`。v1 的 LDM 蒸馏线仍由 `train.py` 维护，
但不定义 v2 的训练或验收输入。

## 已锁定合同

- amplitude SLM、532 nm、完整 4f relay、2+2 phase-only finite-crop D2NN、identity
  OLS、1.50 mm layer spacing、B=0 backend relay。
- 输入为 `label + 8D code`；identity-anchor generator 的 gain 为 `0.5`，seed 为
  `1234`，固定曝光为 `0.4913053810596466`。
- evaluator 只用于训练和验收；最终推理不运行 teacher、diffusion、VAE 或迭代优化。
- 当前配置位于 [`configs/v2/current`](../../configs/v2/current/)，旧配置由迁移 manifest 追溯。

## 当前停止规则

当前 identity 主线尚未完成 evaluator qualification、injection admission、observability、
control isolation 和语义定位的全新内容绑定链。因此不运行 5-epoch 长训练、60,000 张导出、
memorization、utility 或 promotion。旧 100-step smoke 只保留为历史诊断，不能迁移到当前合同。

任何 evidence 的 system/config/evaluator/split/checkpoint/action 不匹配，都必须在创建
optimizer 前失败。旧 global-amplitude、teacher/prototype/capacity、direct-template 和
LightGen evidence 不能授权 identity 主线。

## 历史证据

旧运行手册与完整指标保留在 [`docs/archive`](../archive/)，阶段摘要在
[`experiment_history.md`](experiment_history.md)。它们具有参考价值，但不提供当前执行权。

## 下一允许动作

在 clean commit 上完成当前 evaluator qualification 和 identity admission/observability/
control-isolation 链。若 isolation 判定为 semantic_alignment_required，必须先完成
只读 conditional-semantic-localization；只有其结果为 local_semantic_alignment_plausible
时，才允许当前内容绑定链中的唯一一次 100-step smoke。smoke 通过后，才交给用户运行
预注册的 5-epoch CUDA training。
