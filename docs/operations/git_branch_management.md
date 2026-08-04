# Git 分支治理与回档指南

> 状态快照：2026-07-30。本文的远端引用和提交关系仍是带日期快照；后续操作前应重新执行只读 Git 核验。

## 当前拓扑

```text
main (660763e, scaffold)
  └─ experiment-first-stage-complex-recon (19c611c)
      ├─ 只完成第一阶段训练-… (e4e1553, early side branch)
      └─ encoder-decoder-主要依赖电子域-… (9cd76ca)
          └─ 完成全阶段训练-…-双通道 (5783b6e)
              └─ 基于-LDM-蒸馏的单次曝光光学生成模型 (161555b)
                  └─ 适用于实验室双光子打印加工的尺寸 (e21dba1, v79 snapshot)
                      └─ rewrite/v2 (135b5ed, stable identity-v2 baseline)
                          └─ v5/v6/v6.2/v6.3 linear experiment history
                              └─ experiment/v2-near-conjugate-relay-v64
                                  (44981b1 scientific-state anchor, active diagnostic line)
```

`rewrite/v2` 从 `e21dba1` 线性分出，是 teacher-free identity conditional generation 的稳定
基线。当前活动诊断位于 `experiment/v2-near-conjugate-relay-v64`；旧 v2
teacher/prototype/capacity 内容仍可从 archive 追溯，但不再是当前命令来源。

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
| `rewrite/v2` | `135b5ed` | teacher-free identity v2 稳定基线；当前不直接运行实验。 |
| `experiment/v2-near-conjugate-relay-v64` | `44981b1` scientific-state anchor | 当前活动诊断分支；v6.4 smoke 已归档为 harness-invalid，未 promotion。 |

此前的 v6、宏观可行性、v6.2 和 v6.3 实验提交已经线性包含在 v6.4 分支中。为减少重复 branch
指针，它们已改为以下本地只读 tags：

| Tag | 提交 | 归档断点 |
| --- | --- | --- |
| `archive/v2/pca-transport-v6` | `0b265e3` | v6 PCA transport smoke 失败闭环。 |
| `archive/v2/optical-feasibility` | `e9072f6` | 宏观光路局部可行性审计。 |
| `archive/v2/short-spacing-v6.2` | `25d3c4a` | 0.25 mm 短层距与饱和 harness 闭环。 |
| `archive/v2/formal-spacing-v6.3` | `74a41e8` | 1.50 mm self-imaging smoke 失败闭环。 |
| `archive/v2/capacity-map-v2.2-conditional-negative` | closeout milestone | E2:O0 teacher-free objective-family 条件性负结果封存。 |

这些 tags 只替代冗余本地 branch 指针或标记已关闭研究里程碑；提交、输出目录和 evidence 未删除，远端引用未修改。v2.2 closeout 的可审阅 JSON 快照位于 `experiments/archive/v2_2_capacity_map_closeout/`；它不授权物理训练或 promotion。

## 远端发布状态

远端 `origin` 只存在以下四个历史分支：

| 远端分支 | 提交 | 本地关系 |
| --- | --- | --- |
| `encoder-decoder-主要依赖电子域-不是贴合超表面设计` | `9cd76ca` | 与同名本地分支一致。 |
| `只完成第一阶段训练-encoder为两层-decoder为4层-encoder和decoder均是线性` | `e4e1553` | 与同名本地分支一致。 |
| `基于-LDM-蒸馏的单次曝光光学生成模型` | `4e2ac3f` | 本地 `161555b` 比它领先 16 个提交。 |
| `完成全阶段训练-encoder为1层-decoder为1层-encoder和decoder均是线性，激活函数为gelu` | `ff70108` | 双通道本地分支配置为跟踪该分支，领先 1 个提交。 |

`rewrite/v2`、当前 v6.4 分支、归档 tags、v79 快照、`main` 和 complex-recon 均未在本轮推送。
发布前先执行 `git fetch --prune origin`，再复核本文中的提交与领先关系。

## v79 快照与 rewrite/v2 的区别

两线共享 amplitude SLM、`finite_crop`、2+2 phase-only D2NN、identity OLS 与现行 relay 几何。它们的边界不同：

| 维度 | v79 快照（v1） | `rewrite/v2` |
| --- | --- | --- |
| 入口 | `train.py --stage` 与既有 `ldm_project.*` | `python -m ldm_project.v2` |
| 目的 | 保存 v16–v79 的版本化诊断和失败证据 | 独立、可审计的正式链路重写 |
| 物理/配置合同 | 既有 formal 与 diagnostic 分支共存 | 类型化 plane/field、严格 schema、独立 production/reference 实现 |
| 工件与 checkpoint | 历史 checkpoint、审计工具和版本专用 gate | 内容寻址 teacher manifest、精确 checkpoint manifest、evidence 状态机 |
| 科学状态 | v79 的八维 supported basis 未达到 capacity gate；learned code transport 未证明 | v6.4 可生成可辨认图像并保留 code response，但 distribution gate 与 harness 完整性均未通过 |

v2 不替换 `train.py`，不导入 v1 teacher、旧 prototype/capacity 或 diagnostics。正式 acceptance 以当前 identity evaluator/evidence/5-epoch 链为准；旧 v79 与 v2 诊断不能授予新主线权限。

## 操作与回档规则

- `rewrite/v2` 是 v2 稳定基线；`experiment/v2-near-conjugate-relay-v64` 是当前活动诊断分支；
  v79 分支是 v1 研究快照。两线在正式 acceptance 前并行，不整分支合并。
- 已被 `archive/v2/*` tag 替代的 v2 实验 branch 不应重新创建；从 tag 检查历史，后续新实验仅在
  当前停止门明确允许时从选定的干净提交创建一个分支。
- 历史分支默认只读。需要复用已验证改动时，先审阅具体提交，再使用 `git cherry-pick <commit>`。
- 切换稳定基线使用 `git switch rewrite/v2`；检查历史 v2 断点使用
  `git show archive/v2/<name>` 或在显式需要时从 tag 创建临时 detached checkout。
- 撤销已提交的单项改动使用 `git revert <commit>`；不使用 `git reset --hard`、强推或改写历史。
- 每次计划发布或恢复前，执行 `git status --short --branch`、`git log --oneline --decorate -n 12`，并以实际 Git 输出而不是本文快照为准。
