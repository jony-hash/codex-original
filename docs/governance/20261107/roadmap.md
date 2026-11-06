# 收尾总路线图（2026-11-07）

## 现状判断

治理的表层已经收口：工作树干净、`docs/` 根 5 文件、一屏快照可用、路线表有签字、产物有布局规则、验证有唯一入口。项目层还差发布、全量证据与少量遗留项。

## 阶段

### P0 已完成

工作树、根目录、docs 归位、契约瘦身、验证入口、当前快照、路线表、产物布局、资产说明、分支规则、legacy outputs 归并。

### P1 本批执行

| 包 | 内容 | 产出 |
| --- | --- | --- |
| A1 | 本地 `--full`（3549 测试），只记录不修 | `full_suite.md` |
| A2 | AutoDL 全量交接 | `docs/operations/full_suite_autodl.md` |
| A3 | 路线签字：scaled_chen → `REFERENCE_LEGACY` | `route_status.md` |
| A4 | 14 个非日期 outputs 归并到 `_legacy_2026q3/` | `legacy_moves.jsonl` + 文档 |
| A5 | archive 既有断链统计（794 条 / 4 文件，不修） | `full_suite.md` 附录 |
| A6 | CI 边界说明 | `tests/AGENTS.md` |

### P2 需要决策（未排期）

- 关闭支线的代码是否出包（115 个文件、175 处 import）；当前决定是不移动。
- `reports/` 833 MB 是否外部归档；当前决定是冻结在 legacy 容器内。
- 12 个空目录壳需要管理员权限删除。

### P3 发布

- `git push -u origin research/lightgen-ols-native-semantic-generation-v1`。
- worktree 与空壳只报告，不处理。

### P4 长期纪律

每个里程碑：更新快照、复核路线表、`git fetch --prune`、检查 outputs 是否出现新的非日期顶层目录、确认文献 bundle 已归档。

## 收尾判据

| # | 判据 | 状态 |
| --- | --- | --- |
| 1 | 工作树干净 | ✅ |
| 2 | 已推送且有 upstream | 本批完成 |
| 3 | 一屏状态可信 | ✅（需按里程碑维护） |
| 4 | 每条路线有签字 | ✅ |
| 5 | 每类产物有归宿 | ✅（legacy 容器 + 日期版本目录） |
| 6 | 共享变更验证有全量证据 | 本批采集 |
| 7 | 残余项有 owner | ⚠️ 空壳需管理员，worktree 待 Codex 收尾 |

## 边界

不移动 `application/research/` 代码；不改 `final_pt/` 与恢复元数据；不删除任何分支；不对外发布未经授权的证据。