# Codex 历史会话迁移运行手册

更新时间：2026-08-30

## 当前状态

迁移已于 2026-08-30 完成。旧 WSL Codex 历史仍以不可变归档保留在 `D:\LDM-Archive\codex-original\backup\wsl-codex-history.tar`，217 个唯一会话已安全追加到 Windows Codex Home，未覆盖现有 Windows 会话。

工具位置：`scripts/maintenance/codex_history_migration.py`

外部知识层：`C:\Users\24567\Documents\Codex\codex-original-history-context`

## 固定数据边界

- Windows 主项目：`D:\codex-original`
- Windows Codex Home：`%USERPROFILE%\.codex`
- 旧归档：`D:\LDM-Archive\codex-original\backup\wsl-codex-history.tar`
- 迁移快照：`%USERPROFILE%\Documents\Codex\codex-history-migration-snapshot`
- 不迁入旧认证、`config.toml`、日志、缓存、插件或 WSL 用户配置。
- 不修改原始 tar，不覆盖现有 Windows session，不推送 Git，不注销 Ubuntu。

## 已完成迁移结果

- canary 导入 3 条代表性会话并通过标题、消息、工具记录和工作目录抽查。
- complete 阶段导入其余 214 条会话，累计为 217 条唯一会话。
- 最终校验：SQLite `integrity_check=ok`；217 个导入 JSONL 文件均存在且哈希匹配；147 条可见主聊天、70 条归档内部线程；10 条旧父子关系完整追加。
- 迁移前快照保留在 `%USERPROFILE%\Documents\Codex\codex-history-migration-snapshot`。回滚窗口内不得删除它或修改原 WSL tar。

## 已执行顺序（留作审计）

1. 关闭 GPT App、Codex CLI、VS Code 及 VS Code Codex 扩展窗口。迁移工具会再次检查进程，发现任何 `ChatGPT.exe`、`Codex.exe`、`codex-cli.exe` 或 `Code.exe` 就停止。
4. 在 PowerShell 7 中进入 `D:\codex-original`，先运行：

   ```powershell
   $py = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'
   & $py scripts\maintenance\codex_history_migration.py dry-run
   ```

5. 执行三条 canary：

   ```powershell
   & $py scripts\maintenance\codex_history_migration.py apply --canary
   ```

6. 打开 GPT App，检查最早、最近和最大的一条历史会话。确认标题、消息、工具记录、工作目录和项目归属正确；检查完成后再次关闭 GPT App。
7. 继续全量迁移：

   ```powershell
   & $py scripts\maintenance\codex_history_migration.py apply --complete
   & $py scripts\maintenance\codex_history_migration.py verify
   ```

8. 重新打开 GPT App，使用 Search chats 按标题或短语查找历史会话；本地项目归类可在界面中后续添加 `D:\codex-original` 文件夹，不影响已迁入会话的工作目录。

## 预期分类

- 147 条 `source=vscode` 主聊天保持可见。
- 其中 144 条工作目录属于旧 `codex-original`，归入 `D:\codex-original` 本地项目。
- 另外 3 条主聊天保留可见，但不错误归入代码项目。
- 70 条 guardian、spawned subagent 和 exec 内部线程完整保留并默认归档。
- 旧归档中的 1 个重复 JSONL 内容哈希相同，只保留 1 份。

## 回滚

canary 或全量验收失败时，关闭 GPT App 后运行：

```powershell
& $py scripts\maintenance\codex_history_migration.py rollback
```

回滚前若发现快照后产生了新的 Windows Codex 线程，工具会拒绝回滚，避免覆盖新工作。回滚会把当前相关文件移入快照目录下的 quarantine，再恢复迁移前的 sessions 和 SQLite 文件。

## 验收边界

SQLite、JSONL、哈希、ID 和路径由工具自动验证；GPT App 中的项目显示、长会话渲染、搜索和继续会话必须人工检查。知识层是脱敏导航，不替代原始 transcript；新聊天按需检索历史，不能把全部历史一次性加载进上下文窗口。
