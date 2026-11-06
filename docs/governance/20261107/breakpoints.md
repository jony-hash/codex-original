# 断点处置（2026-11-07）

本页对应 `docs/governance` 审计出的 31 个断点。状态只有四类：`FIXED`（本轮修复）、`BLOCKED`（缺权限或数据已被清除）、`IMMUTABLE`（归档/证据不可改写）、`DEFERRED`（需要所有者决策）。

## 本轮修复

| # | 断点 | 修复 |
| --- | --- | --- |
| 3 | `migration_windows_native_20260830.md:79` 指向已移动的 `outputs/migration_smoke/...` | 改为 `outputs/_legacy_2026q3/migration/migration_smoke/...` |
| 15 | 旧路线注册表（472 行）与新 `route_status.md` 并存、无指向 | `lightgen_research_route_registry.md` 顶部加"历史注册表，状态见 route_status"声明 |
| 22 | `--full` 在长路径临时目录清理时崩溃，退出码不可信 | `run_windows_cpu_checks.py` 改用短前缀 `ldm-pytest-` 并加 `ignore_cleanup_errors=True` |
| 24 | 云端环境（ruff/pymupdf/版本）未记录 | `full_suite_autodl.md` 增加环境版本与看门狗命令 |

## 阻塞：缺权限

| # | 断点 | 原因 |
| --- | --- | --- |
| 10 | `outputs/` 下 78 个目录不可读（`os.scandir` 直接 `PermissionError`，加 `\\?\` 前缀仍拒绝） | 目录 ACL 属于另一 SID；当前用户无 `READ_CONTROL`，需管理员 `takeown`/`icacls` |
| 11 | 根目录 12 个空壳 `.pytest-*` | 同上，且根目录有 `Everyone Deny DeleteSubdirectoriesAndFiles` 安全地板 |

## 阻塞：数据已被清除

| # | 断点 | 原因 |
| --- | --- | --- |
| 9 | 全量 16 failed + 7 errors 指向的 `conditional_optical_generator.pt`、`20261106_.../cloud_archive/...` | 本地全盘与 AutoDL `/root` 均未找到；该 checkpoint 与云归档已被 2026-11-06 存储清理删除，无法伪造 |
| 1b | `ols_gaussian_lp01_model_alignment_v1.md` 指向 `complex_field_readouts.pt` | 全盘未找到，同属已清理证据 |

## 不可改写：归档/证据

| # | 断点 | 处置理由 |
| --- | --- | --- |
| 1a | `CURRENT_STATUS.md` 的乱码绝对路径 | 科学状态记录，属既有条目；改路径会改历史正文 |
| 2 | `docs/archive/` 793 条死链 | archive 只读，保留原始快照 |
| 4 | 2 个 `artifacts/*.json` 仍写迁移前 doc 路径 | 证据元数据不可改写；`human_readable_report` 只作 provenance |
| 5 | 50 个 `experiments/archive/pre_v57/tools` 引用 `outputs/reports` | 归档工具冻结；恢复该顶层目录会违反新布局规则 |
| 27 | `wp0/disposition.jsonl` 记录迁移前路径 | 历史处置记录，保留原始路径 |

## 待决策

| # | 断点 | 需要什么 |
| --- | --- | --- |
| 14 | 无 `main` 分支（本地/远端都没有），文档仍称 main 为历史骨架 | 是否创建 `main` 指向 `043e254` 并推送；这是仓库主干决策 |
| 18 | 4 个脏 worktree（412 项）停在 `043e254` | 保留还是让 Codex 收尾 |
| 19 | 云端仓库停在 `b9d865e` 且有 345 项脏改动 | 是否让 Codex 对齐到当前 HEAD |

## 接受（记录风险，不改）

| # | 断点 | 说明 |
| --- | --- | --- |
| 6 | `ppt_inference_flow_demo.py` 默认 `outputs/reports_DEMO-ppt/` | 非研究主链脚本；运行前需显式 `--output`，规则已在 `output_artifact_notes.md` 说明 |
| 7 | 10 个 py 文件 25 行绝对路径 | 研究与测试固定路径，属历史实现；新代码禁止 |
| 8 | 25 个 md 198 行绝对 `D:` 路径 | 既有文档；新文档禁止（`docs/AGENTS.md`） |
| 12 | `research_assets/` 701.6 MB 本地-only | 已记入 `research_assets/README.md`；外部归档缺失是已知风险 |
| 13 | 172 MB 大文件被跟踪 | 改历史需 rewrite，收益不抵风险 |
| 16 | 16 个 status 文件 | 已用 `current_snapshot.md` 收敛入口，正文不动 |
| 17 | hash 政策张力 | AGENTS 已明确 digest 仅 provenance；`test_governance_closeout.md` 保留完整性用途，语义兼容 |
| 23 | CI 只跑 focused | `tests/AGENTS.md` 已写明；全量靠手工/夜间 |
| 25 | 云端全量停滞 | 已记录宿主负载与看门狗 |
| 26 | 70 处 pending 标记 | 多为科学 `UNKNOWN`，非待办 |
| 28 | 三份状态文档仍是流水账 | 快照承担当前入口 |
| 29–31 | 路线标签与失败测试/证据粒度不完全一致 | `route_status.md` 使用规则已说明以状态文档为准 |

## 复核方式

- 链接：对全部 tracked Markdown 跑一次相对链接检查（排除 `docs/archive/`），应仍为 2 处既有断点。
- 目录权限：`os.scandir` 扫描 `outputs/` 统计 `PermissionError`，应仍为 78。
- 缺失工件：`find` 全盘与云端 `/root`，应仍为 0 命中。
- 治理产物：`legacy_moves.jsonl` 14/14 目标存在。