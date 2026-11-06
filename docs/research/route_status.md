# 研究路线状态

本表回答"这条线现在算什么"。状态只引用现有文档，不新增判定；文档无法确认的写 `UNCONFIRMED`。代码、配置或测试目录的存在本身不构成执行授权或科学通过。

| 路线 | 状态 | 入口 | 当前消费者 | 禁止事项 | 来源 |
| --- | --- | --- | --- | --- | --- |
| 正式 v1（LDM teacher 蒸馏） | `ACTIVE`（保留运行时） | `train.py --stage` | v1 distillation / one-shot evidence | 不能签发 v2 evidence；不是当前默认主线 | [v1 合同](../architecture/v1_single_shot_distillation.md)、[README](../../README.md) |
| 正式 v2（teacher-free conditional） | `ACTIVE`（当前正式） | `python -m ldm_project.v2` | `application/current/` 的 formal identity chain | 不得 import v1 teacher/target 或 research/diagnostic 模块 | [v2 合同](../architecture/v2_runtime.md)、[taxonomy](../repository_taxonomy.md) |
| LightGen OLS-native | `ACTIVE`（研究主线） | `scripts/research/lightgen/` | 当前 HW207 / C3 系列 | 数值通过不等于实物闭环；不得自动升级 formal 或加工放行 | [project map](project_map.md)、[CURRENT_STATUS](../CURRENT_STATUS.md) |
| scaled_chen（Chen400 / finite-pixel RS） | `REFERENCE_LEGACY`（2026-11-07 签字） | `scripts/research/scaled_chen/` | 无命名消费者；保留为重放与参考实现 | 不得当作独立正式路线；新工作需先建立命名消费者 | [project map](project_map.md) |
| optical_manifold / fiber | `DIAGNOSTIC_ONLY` | `scripts/research/optical_manifold/` | fiber 与 manifold 归因诊断 | 不得声明 fiber 特异性增益，不得修改 formal identity 合同 | [fiber lane status](../status/v2_fiber_image_generation_v1.md)、[corrected status](../status/v2_fiber_ols_corrected_status.md) |
| capacity-map（O0 / H-SCED） | `CLOSED`（不授予物理容量归因） | `scripts/research/capacity/` | 历史归因索引 | 不得再以容量归因授权新实验或正式验收 | [capacity ledger](../status/capacity_map_hypothesis_ledger.md) |

## 状态标签

- `ACTIVE`：有当前消费者与一个已声明的下一步；只能在该消费者授权内执行。
- `DIAGNOSTIC_ONLY`：只产生诊断证据，不进入正式验收。
- `CLOSED`：已关闭，仅保留溯源与重放价值。
- `REFERENCE_LEGACY`：保留为参考实现或重放路径，没有当前消费者；不得作为默认路线。
- `UNCONFIRMED`：本仓库文档没有给出可引用的状态结论；使用前先补状态文档。

## 使用规则

1. 新任务开始前先在本表定位路线，再读对应状态文档；本表不替代状态文档。
2. 本表与状态文档冲突时，以状态文档正文为准，并修正本表。
3. 关闭支线的代码仍留在 runtime 包内（`application/research/`），这是有意的隔离；移动它们是另一项需要单独授权的变更。