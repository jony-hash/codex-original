# Documentation index

先读当前状态，再读架构与运行说明；archive 只用于溯源，不能覆盖当前合同。

| 任务 | 主文档 |
| --- | --- |
| 当前停止规则、有效证据与下一门 | [Current Handoff](status/current_handoff.md) |
| v1 架构与最终推理边界 | [v1 single-shot architecture](architecture/v1_single_shot_distillation.md) |
| v2 runtime、evidence 和 LightGen proxy | [v2 runtime](architecture/v2_runtime.md) |
| 命令、stage 和回档 | [Stage Guide](operations/stage_guide.md) / [Git branch management](operations/git_branch_management.md) |
| 输出与 artifact 语义 | [Output notes](operations/output_artifact_notes.md) |
| 实验历史 | [Experiment history](status/experiment_history.md) |
| 历史配置、工具、测试与原始交接 | [Experiment archive](../experiments/archive/manifest.yaml) / [Docs archive](archive/) |
| 目录归类和兼容策略 | [Repository taxonomy](repository_taxonomy.md) |

`architecture/` 定义稳定合同；`operations/` 定义如何运行与回退；`status/` 记录当前状态；`archive/` 保留只读历史正文和 JSON。文档整理不触碰 `outputs/`、`data/raw/`、`.tmp/` 或 `legacy/`。
