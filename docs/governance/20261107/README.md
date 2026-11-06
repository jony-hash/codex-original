# 工作区治理执行报告（2026-11-07）

## 1. 结果

| 包 | 结果 |
| --- | --- |
| WP0 基线 | 365 个未提交条目分类完毕，`wp0/disposition.jsonl` 记录逐项处置；worktree、scratch、import 面已扫描 |
| WP1 卫生 | 回收 26 个 scratch 目标、187 个文件（约 276 MB）；删除 14 个目录与 5 个根目录临时图片；`git worktree prune` 无副作用 |
| WP2 验证 | `pyproject.toml` 新增仓库级 Ruff（`E9,F82`）；`run_windows_cpu_checks.py` 新增 `--full`；文档写明唯一验证入口 |
| WP3 文档 | 19 个 OLS/物理功率/紧凑系统报告移入 `docs/research/ols/`；5 个运维文档移入 `docs/operations/`；`docs/` 根只剩 5 个入口文件；入链与内部链接全部重算 |
| WP4 契约 | 根 `AGENTS.md` 312 → 165 行，修复首行乱码，新增规则映射 `wp4/rule_map.md`；新增 `docs/AGENTS.md` 放置规则 |
| WP5 边界 | `v2/AGENTS.md` 明确 research 命名空间只读隔离；确认边界测试已存在于 `tests/test_architecture_boundaries.py`（current 禁 import research） |
| WP6 收尾 | 验证通道全绿；本报告 |

## 2. 验证证据

- `python scripts/run_windows_cpu_checks.py`：**176 passed**，仓库级 Ruff 通过，focused Ruff 通过，`compileall` 通过。
- `docs/` 根文件数：5（AGENTS、CURRENT_STATUS、PROJECT_MEMORY、README、repository_taxonomy）。
- 指向被移动文件的断链：**0**（不含 `docs/archive/` 内既有的历史快照断链）。
- `AGENTS.md`：165 行，无 U+FFFD；18 个链接目标全部存在。
- 根目录 scratch：从 26 个目标降到 12 个空壳（见 §3）。

## 3. 剩余与未完成

1. **12 个空目录壳**无法删除：`.pytest-*`、`.pytest_cache_*`、`.tmp_pytest_cache_phase_repair`、`mpcdar-preflight-tmp`。当前账户对这些对象没有读取权限（`Get-Acl` 被拒），无法取得删除权。需要管理员执行 `takeown`/`icacls` 后删除，或忽略——它们为空且被 `.gitignore` 覆盖。
2. **5 个 worktree 未删除**：4 个有未提交工作（136 / 38 / 38 / 156 项），1 个干净。删除会破坏其他 Codex 会话，保留原状。
3. **工作树仍未提交**：`git status -uall` 为 796 条目（757 个未跟踪文件、23 个修改、16 个删除），其中绝大多数是 11 月 3–6 日的研究工作与本次治理改动。按契约提交权在 Codex；本计划未提交。
4. **`docs/archive/` 既有断链**：历史快照指向不存在的 `outputs/` 与旧路径，属既有债务，archive 不可改写，未处理。
5. **`AGENTS.md` 165 行**，未达计划自设的 150 行；差异来自保留 8 条执行规则与 5 条不可协商项的完整语义，未删规则凑数。
6. `.ruff_cache/` 会在每次验证后于根目录重建；已被 `.gitignore` 覆盖，属工具行为。

## 4. 边界（未触碰）

- `outputs/`（含 `final_pt/`、`ppt_*`、`reports/`）、`data/raw/`、`.tmp/`、`experiments/archive/` 零改动。
- 科学结论未改写：`CURRENT_STATUS.md`、`current_handoff.md`、`project_map.md`、路线图与收尾报告只改了指向被移动文档的链接路径，日期条目与判定文字未动。
- 未新增 Gate、未改运行时代码、未改配置语义。

## 5. 提交门与建议

建议由 Codex 分 5 个提交落地：

1. `docs:` 移动 OLS 系列与运维文档，更新入链与索引（含 `docs/AGENTS.md`、`docs/research/ols/README.md`）。
2. `build:` 新增 Ruff 配置与 `--full` 验证通道。
3. `docs(contract):` 根 `AGENTS.md` 瘦身与规则映射。
4. `docs(contract):` `v2/AGENTS.md` research 命名空间边界。
5. `chore(governance):` 本目录的扫描、处置与报告。

提交前先处理 12 个空目录壳（可选）与 4 个脏 worktree（不要删）。

## 附：产物

- `wp0/summary.md`、`wp0/disposition.jsonl`
- `wp1/deleted_dirs.jsonl`
- `wp4/rule_map.md`
- `docs/research/ols/README.md`（索引）

## 第二批：结构层（2026-11-07）

### 完成

- **G1 当前快照**：新增 [`docs/status/current_snapshot.md`](../status/current_snapshot.md)，固定四段（活跃路线 / 当前停止点 / 允许动作 / 未验证缺口）；`docs/README.md` 首行与 `CURRENT_STATUS.md` 头部改为"先读快照"。
- **G2 产物规则**：`output_artifact_notes.md` 新增顶层布局——`runs/`、`assets/`、`reports/` 都必须挂在日期版本目录下；根 `AGENTS.md` §7 同步一句；现有 16 个非日期顶层目录标记 legacy/frozen，不移动。
- **G3 路线状态**：新增 [`docs/research/route_status.md`](../research/route_status.md)，六行覆盖 v1、v2、LightGen、scaled_chen、optical_manifold/fiber、capacity；状态只引用现有文档，scaled_chen 标 `UNCONFIRMED`，未擅自下结论。
- **G4 资产来源**：[`research_assets/README.md`](../../research_assets/README.md) 纳入版本库，列出 14 个 bundle 与归档要求；`.gitignore` 增加 README negation。工作区未检测到 `D:\LDM-Archive`，已写入文档。
- **G5 分支**：`git fetch --prune` 清理失效远端引用（远端分支 37 → 7）；`git_branch_management.md` 新增清理规则——合并分支删除必须经用户授权。

### 验证

- `python scripts/run_windows_cpu_checks.py`：176 passed，两轮 Ruff 通过，`compileall` 通过。
- 新增/改动文档的断链为 0；`docs/` 根仍为 5 个入口文件。

### 未做与边界

- 不移动 `application/research/` 的 115 个文件：175 处 import 且边界测试已隔离，移动收益不抵风险。
- 不删除任何本地/远端分支，不 push；本地分支仍无 upstream。
- 12 个空目录壳与 5 个 worktree 保持原状。