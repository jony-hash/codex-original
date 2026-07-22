# Documentation

本目录按读者任务组织文档。先找到当前问题的唯一主文档，再沿链接查看设计依据或历史证据；不要用历史实验记录覆盖当前执行契约。

## Start Here

- 项目概览与最小安全入口：[README](../README.md)
- 架构与最终推理边界：[D2NN 单次曝光蒸馏主线](ldm_migration/single_shot_ldm_distillation.md)
- Stage 与 config 选择：[Stage Guide](ldm_migration/stage_guide.md)
- 输出与可视化解释：[D2NN 主线输出与可视化说明](output_JIESHI.md)
- 迁移或续接当前实验：[当前主线交接（v79 冻结与 v2）](ldm_migration/current_handoff.md)（时效性状态快照，下一决策门后刷新）
- 历史实验与验收证据：[D2NN 实验与验收历史](ldm_migration/experiment_history.md)
- 理解分支、回档与发布状态：[Git 分支治理与回档指南](git_branch_management.md)
- 归档配置与诊断源码：[实验归档清单](../experiments/archive/manifest.yaml)（仅供溯源，`diagnostic-only`）
- 旁路重写运行时：[v2 Contract-First Runtime](v2/README.md)（不替换 v1 默认入口）

## Choose By Task

| 任务 | 主文档 | 权威范围 |
| --- | --- | --- |
| 理解项目并安全开始 | [README](../README.md) | 稳定入口、最小命令、目录地图 |
| 理解架构和最终推理边界 | [D2NN 单次曝光蒸馏主线](ldm_migration/single_shot_ldm_distillation.md) | 架构契约、teacher/student 边界、硬件化顺序 |
| 选择或运行 stage | [Stage Guide](ldm_migration/stage_guide.md) | 命令、必需输入、checkpoint 与验收顺序 |
| 解释图片和 tensor bundle | [输出与可视化说明](output_JIESHI.md) | artifact 路径、张量语义、诊断图读取方式 |
| 迁移或续接当前实验 | [当前主线交接（v79 冻结与 v2）](ldm_migration/current_handoff.md) | 当前合同、有效证据、停止规则与下一决策门 |
| 查阅过去的结果 | [实验与验收历史](ldm_migration/experiment_history.md) | 日期、指标、负结果、回归证据 |
| 使用旁路 v2 runtime | [v2 Contract-First Runtime](v2/README.md) | 类型化物理合同、evidence gate 与独立 CLI |
| 理解分支与安全回档 | [Git 分支治理与回档指南](git_branch_management.md) | 分支谱系、远端状态、v1/v2 差异与 revert 策略 |
| 约束 Codex 执行 | [AGENTS.md](../AGENTS.md) | 仓库执行策略与不可变边界 |

## Source Of Truth

- `AGENTS.md` 是执行策略和项目不可变边界的权威来源。
- `single_shot_ldm_distillation.md` 是架构与最终推理契约的权威来源。
- `stage_guide.md` 是 stage 选择、命令模板与验收顺序的权威来源。
- `output_JIESHI.md` 是输出 artifact 和诊断张量语义的权威来源。
- `experiment_history.md` 保存历史证据，不定义当前默认行为。
- `current_handoff.md` 只保存最新状态、停止规则和下一决策门；它不是架构合同，下一决策门完成后必须刷新。

同一事实只在一个主文档中展开；其他文档仅做摘要并链接到主文档。

## Protected Paths

文档整理不触碰 `outputs/`、`data/raw/`、仓库 `.tmp/` 或 `legacy/`。输出迁移只能通过受控工具执行：真实 run 目录统一位于 `outputs/YYYYMMDD_<version>/runs/<run>/`，并按版本内最早 surviving run 日期排序；`outputs/final_pt/` 不参与迁移。完整约束见 [AGENTS.md](../AGENTS.md)。
