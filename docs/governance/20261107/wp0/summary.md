# WP0 基线快照（2026-11-07）

## 仓库状态

- 分支：`research/lightgen-ols-native-semantic-generation-v1`
- 未提交：365 项（352 提交类 / 8 迁移类 / 5 删除类）
- 未提交集中：scripts/research 148、tests/research 88、configs/research 66、src 16、research_assets 13、artifacts 8
- 结论：11 月 3–6 日的研究工作全部未入库；本次治理不替这些工作做科学判断，只做归置。

## 分类结果

处置清单见 `disposition.jsonl`：

- `commit`：352 项，均为研究执行器、测试、配置、源码、研究资产与新文档。
- `move`：8 项，`docs/research/ols/ols_*`、`docs/research/ols/physical_*`、`docs/research/ols/compact_*`、`docs/operations/storage_cleanup_20261015.md`。
- `delete`：5 项，根目录 `_tmp_*.png` / `.jpeg`。

## 根目录 scratch 扫描

23 个 pytest/tmp 目录中，18 个为空；`test_scratch` 仅 4 个 `.pyc`；`.ruff_cache` 为缓存。
5 个 `_pytest_g1_fit20_*` 共 275 MB，内容是测试生成的 `.pt` 基线与 JSON（测试 basetemp），非研究证据，可回收。
4 个 `_pytest_gaussian_jdiag_recovery*` 为测试输出（json/npz/csv.gz），可回收。
`_tmp_*.png` 仅出现在 `artifacts/.../git_preflight.json` 的"未跟踪文件清单"里，无实际引用。
`.tmp/` 按仓库契约保留，不清理。

## worktree 扫描

5 个游离 worktree：4 个有未提交工作（136/38/38/156 项），1 个干净（5dcb, 113ca1c）。
结论：只执行 `git worktree prune`（清理失效登记），不删除任何 worktree 目录，避免破坏其他会话。

## 边界扫描（供 WP5）

`src/ldm_project/v2/application/current/` 中 `frozen_mpcdar_rank16_*` 只包含 artifact 类型字符串，**没有 import** `application.research.optical_manifold`。当前 formal 与 research 命名空间之间没有代码级依赖，补边界测试是安全的。

## 本包未改动

未删除、未移动、未修改任何文件；仅新增本目录。