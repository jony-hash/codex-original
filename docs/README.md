# Documentation index

先读当前状态，再读架构与运行说明；archive 只用于溯源，不能覆盖当前合同。

| 任务 | 主文档 |
| --- | --- |
| ChatGPT 讨论、项目记忆与 Codex 本地交接 | [Project Memory](PROJECT_MEMORY.md) / [ChatGPT ↔ Codex workflow](chatgpt_codex_workflow.md) / [Delta + One Gate + Short Prompt](operations/delta_one_gate_short_prompt.md) |
| 当前 LightGen M0 结果与停止点 | [Current Status](CURRENT_STATUS.md) / [Current Handoff](status/current_handoff.md) |
| 当前停止规则、有效证据与允许动作/停止边界 | [Current Handoff](status/current_handoff.md) |
| OLS-native N3 历史治理状态 | [Research governance state](status/research_governance_state.json) / [Experiment History](status/experiment_history.md) |
| 已关闭的 O0/H-SCED 假设与历史归因 | [Capacity-map hypothesis ledger](status/capacity_map_hypothesis_ledger.md) |
| v1 架构与最终推理边界 | [v1 single-shot architecture](architecture/v1_single_shot_distillation.md) |
| 当前 v2 identity 无蒸馏 runtime、evidence 和停止规则 | [v2 runtime](architecture/v2_runtime.md) |
| Research 脚本、运行模块、配置与测试的分类 | [Research Project Map](research/project_map.md) |
| electronic O0 与 synthetic fiber 路线状态 | [Capacity-map hypothesis ledger](status/capacity_map_hypothesis_ledger.md) / [Fiber image-generation status](status/v2_fiber_image_generation_v1.md) / [Corrected fiber OLS status](status/v2_fiber_ols_corrected_status.md) |
| v2 旧蒸馏、prototype/capacity 与 LightGen 历史 | [archive](archive/)；活动兼容实现仍在显式 `legacy`/`diagnostic` 命名空间 |
| LightGen 论文补充材料与图源数据 | [`research_assets/lightgen/science_adv7434/`](../research_assets/lightgen/science_adv7434/)；只读 reference-only 证据包 |
| 命令、stage 和回档 | [Stage Guide](operations/stage_guide.md) / [Git branch management](operations/git_branch_management.md) |
| 长训练、模型选择和科学结论预检 | [Long-training preflight](operations/long-training-preflight.md) |
| 输出与 artifact 语义 | [Output notes](operations/output_artifact_notes.md) |
| 实验历史 | [Experiment history](status/experiment_history.md) |
| 历史配置、工具、测试与原始交接 | [Experiment archive](../experiments/archive/manifest.yaml) / [Docs archive](archive/) |
| 目录归类和兼容策略 | [Repository taxonomy](repository_taxonomy.md) |

`architecture/` 定义稳定合同；`operations/` 定义如何运行与回退；`status/` 提供当前入口；`archive/` 保留只读历史正文和 JSON。2026-10-01 前的长状态和完整实验历史快照位于 `archive/20261001_project_refactor/`。文档整理不触碰 `outputs/`、`data/raw/`、`.tmp/` 或显式标记的 legacy namespaces。
