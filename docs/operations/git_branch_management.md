# Git 分支治理与回档指南

> 状态快照：2026-07-24。本文的远端引用和提交关系仍是带日期快照；后续操作前应重新执行只读 Git 核验。

## 当前拓扑

```text
main (660763e, scaffold)
  └─ experiment-first-stage-complex-recon (19c611c)
      ├─ 只完成第一阶段训练-… (e4e1553, early side branch)
      └─ encoder-decoder-主要依赖电子域-… (9cd76ca)
          └─ 完成全阶段训练-…-双通道 (5783b6e)
              └─ 基于-LDM-蒸馏的单次曝光光学生成模型 (161555b)
                  └─ 适用于实验室双光子打印加工的尺寸 (e21dba1, v79 snapshot)
                      └─ rewrite/v2 (724f389 baseline, active identity-v2 line)
```

`rewrite/v2` 从 `e21dba1` 线性分出，当前活动合同是 teacher-free identity conditional generation。旧 v2 teacher/prototype/capacity 内容仍可从 archive 追溯，但不再是当前命令来源。

为便于不切换分支地查阅证据，`e21dba1` 中原有的 v68–v79 交接全文已逐字归档到 [v68–v79 原始主线交接归档](../archive/current_handoff_v68_v79_archive.md)。归档不定义当前行为；当前状态仍以 [当前主线交接](../status/current_handoff.md) 为准。

## 本地分支

| 分支 | 提交 | 角色与状态 |
| --- | --- | --- |
| `main` | `660763e` | 初始项目骨架；本地历史锚点。 |
| `experiment-first-stage-complex-recon` | `19c611c` | 第一阶段复杂复场重建实验；本地历史分支。 |
| `只完成第一阶段训练-encoder为两层-decoder为4层-encoder和decoder均是线性` | `e4e1553` | 早期 2-layer encoder／4-layer decoder 线性旁支。 |
| `encoder-decoder-主要依赖电子域-不是贴合超表面设计` | `9cd76ca` | LDM 迁移、checkpoint 与电子域基线。 |
| `完成全阶段训练-encoder为1层-decoder为1层-encoder和decoder均是线性，激活函数为gelu-双通道` | `5783b6e` | 完整单次曝光工作流基线。 |
| `基于-LDM-蒸馏的单次曝光光学生成模型` | `161555b` | one-shot 蒸馏、评估与审计主线。 |
| `适用于实验室双光子打印加工的尺寸` | `e21dba1` | 已封存的 v68–v79 诊断研究和正式物理几何快照。 |
| `rewrite/v2` | `724f389` baseline | 当前活动分支；identity teacher-free v2，未 promotion。 |

## 远端发布状态

远端 `origin` 只存在以下四个历史分支：

| 远端分支 | 提交 | 本地关系 |
| --- | --- | --- |
| `encoder-decoder-主要依赖电子域-不是贴合超表面设计` | `9cd76ca` | 与同名本地分支一致。 |
| `只完成第一阶段训练-encoder为两层-decoder为4层-encoder和decoder均是线性` | `e4e1553` | 与同名本地分支一致。 |
| `基于-LDM-蒸馏的单次曝光光学生成模型` | `4e2ac3f` | 本地 `161555b` 比它领先 16 个提交。 |
| `完成全阶段训练-encoder为1层-decoder为1层-encoder和decoder均是线性，激活函数为gelu` | `ff70108` | 双通道本地分支配置为跟踪该分支，领先 1 个提交。 |

`rewrite/v2`、v79 快照、`main` 和 complex-recon 均未发布到 `origin`。本轮不推送；发布前先执行 `git fetch --prune origin`，再复核本文中的提交与领先关系。

## v79 快照与 rewrite/v2 的区别

两线共享 amplitude SLM、`finite_crop`、2+2 phase-only D2NN、identity OLS 与现行 relay 几何。它们的边界不同：

| 维度 | v79 快照（v1） | `rewrite/v2` |
| --- | --- | --- |
| 入口 | `train.py --stage` 与既有 `ldm_project.*` | `python -m ldm_project.v2` |
| 目的 | 保存 v16–v79 的版本化诊断和失败证据 | 独立、可审计的正式链路重写 |
| 物理/配置合同 | 既有 formal 与 diagnostic 分支共存 | 类型化 plane/field、严格 schema、独立 production/reference 实现 |
| 工件与 checkpoint | 历史 checkpoint、审计工具和版本专用 gate | 内容寻址 teacher manifest、精确 checkpoint manifest、evidence 状态机 |
| 科学状态 | v79 的八维 supported basis 未达到 capacity gate；learned code transport 未证明 | identity v2 尚未完成当前 evidence 链和长训练；旧 v2 capacity 结果仅为历史参考 |

v2 不替换 `train.py`，不导入 v1 teacher、旧 prototype/capacity 或 diagnostics。正式 acceptance 以当前 identity evaluator/evidence/5-epoch 链为准；旧 v79 与 v2 诊断不能授予新主线权限。

## 操作与回档规则

- `rewrite/v2` 是后续 v2 工作的活动分支；v79 分支是 v1 研究快照。两线在正式 acceptance 前并行，不整分支合并。
- 历史分支默认只读。需要复用已验证改动时，先审阅具体提交，再使用 `git cherry-pick <commit>`。
- 切换整体回档点使用 `git switch rewrite/v2` 或 `git switch 适用于实验室双光子打印加工的尺寸`。
- 撤销已提交的单项改动使用 `git revert <commit>`；不使用 `git reset --hard`、强推或改写历史。
- 每次计划发布或恢复前，执行 `git status --short --branch`、`git log --oneline --decorate -n 12`，并以实际 Git 输出而不是本文快照为准。
