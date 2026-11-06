# Codex 历史会话迁移运行手册

更新时间：2026-08-30

## 当前状态

迁移已于 2026-08-30 完成。旧 WSL Codex 历史仍以不可变归档保留在 `D:\LDM-Archive\codex-original\backup\wsl-codex-history.tar`，217 个唯一会话已安全追加到 Windows Codex Home，未覆盖现有 Windows 会话。历史会话的 folder-bound 项目修复 v2 已在真实 Codex Home 完成 canary/full apply，并通过 `project-bind --verify` 与顶层 `verify`；GPT App 的项目显示、搜索、长会话滚动和继续会话已由用户完成人工验收。

注销前审计提交为 `75d0584c379b145b0f0eb45ece14fda160741f23`，已推送到目标 GitHub 分支，
工作树干净。最终 Git bundle 位于
`C:\Users\24567\Documents\Codex\codex-original-final-20260830.bundle`，已通过
`git bundle verify`，包含当前 HEAD 和 55 个 refs。备份时间后的 WSL 增量位于
`C:\Users\24567\Documents\Codex\wsl-post-backup-increment-20260830.tar`，包含 5 个
用户范围路径，SHA-256 为
`072212068E3F6D02FFD3B762B8670CB2B5E6A83F967137C6459874D99A07494B`；Ubuntu 已在最终审计
通过后注销。

工具位置：`scripts/maintenance/codex_history_migration.py`

外部知识层：`C:\Users\24567\Documents\Codex\codex-original-history-context`

跨产品交接入口：[`docs/PROJECT_MEMORY.md`](../PROJECT_MEMORY.md)；云端项目
`codex-original · ChatGPT` 与本地项目 `codex-original · Windows` 的受控 Sources 桥接、
构建/验证步骤见 [`docs/operations/chatgpt_codex_workflow.md`](../operations/chatgpt_codex_workflow.md)。本手册的
历史迁移规则仍优先于任何 ChatGPT 讨论或 Sources 副本。

## 固定数据边界

- Windows 主项目：`D:\codex-original`
- Windows Codex Home：`%USERPROFILE%\.codex`
- 旧归档：`D:\LDM-Archive\codex-original\backup\wsl-codex-history.tar`
- 迁移快照：`%USERPROFILE%\Documents\Codex\codex-history-migration-snapshot`
- 项目绑定快照：`%USERPROFILE%\Documents\Codex\codex-project-binding-snapshot-20260830`
- 不迁入旧认证、`config.toml`、日志、缓存、插件或 WSL 用户配置。
- 不修改原始 tar，不覆盖现有 Windows session；GitHub 推送已完成，Ubuntu 注销已完成，
  不手动删除或改写历史 VHD/归档。
- `project-bind` v2 不查询或写入 `projects`、`project_roots` 或 `threads.project_id`；桌面端实际消费者是会话及 transcript 中的 `cwd`。

## 已完成迁移结果

- canary 导入 3 条代表性会话并通过标题、消息、工具记录和工作目录抽查。
- complete 阶段导入其余 214 条会话，累计为 217 条唯一会话。
- 最终校验：SQLite `integrity_check=ok`；217 个导入 JSONL 文件均存在且哈希匹配；147 条可见主聊天、70 条归档内部线程；10 条旧父子关系完整追加。
- 迁移前快照保留在 `%USERPROFILE%\Documents\Codex\codex-history-migration-snapshot`，不自动删除或修改它及原始 WSL tar。

Ubuntu 已注销后，以上快照、原始 WSL tar、用户配置 tar、增量 tar 和 Git bundle 仍作为
离线恢复材料保留；它们不是当前工作区，也不会被自动展开。

## 实际 folder-bound 修复结果（2026-08-30）

- `project-bind --apply --canary` 修复 3 条固定代表；`project-bind --apply --complete` 修复剩余 211 条，累计 214/214。
- `project-bind --verify` 通过：phase=`complete`、SQLite `integrity=ok`、候选 214、可见 144、归档 70、杂项 3、父子边 16；217 个迁移文件均通过修复后哈希验证。
- 顶层 `verify` 首次暴露 CLI 参数注册缺陷（`verify` 子命令未注册 `binding_manifest`）；已补齐参数并加入回归测试，修复后的真实顶层 `verify` 通过：SQLite `ok`、217/217 文件、147 可见、70 归档。
- GPT App 重开后将一个既有线程的 `rollout_path` 规范化为带 `\\?\` 前缀的等价 Windows extended path，验证器曾因此安全拒绝。全量只读字段比对确认只有这一处等价差异；现仅对 `rollout_path` 使用 `canonical_path()`，修复后的真实 `project-bind --verify` 已再次通过。
- `session_index.jsonl` 保持不变，34 行、候选命中 0 条，仅作为审计观察；数据库 `projects`/`project_roots` 为空、候选 `threads.project_id` 保持为空均符合 folder-bound 语义。

## 本地项目归属修复（folder-bound v2）

会话迁移和本地项目归类是两个独立阶段。GPT App 使用本地项目 `codex-original · Windows`，主文件夹为 `D:\codex-original`，辅助文件夹为 `C:\Users\24567\Documents\Codex\codex-original-history-context`。桌面端按 `cwd` 将会话映射到文件夹项目；`projects`/`project_roots` 为空和 `threads.project_id=NULL` 都是正常状态，工具不会伪造 Project ID。辅助目录的历史入口名为 `HISTORY_CONTEXT.md`，不再使用与仓库权威记忆同名的外部 `PROJECT_MEMORY.md`。

旧实现只检查 transcript 首行，导致后续 `session_meta`/`turn_context` 再次把会话的 `cwd` 漂移回 WSL。v2 对 manifest 指定的 214 条候选流式处理：只改 `session_meta.payload.cwd` 和 `turn_context.payload.cwd` 中的精确旧路径 `/home/dingzhen/projects/codex-original`，将其写为 `D:\codex-original`；`event_msg` 中的历史命令 `cwd` 保持原样，作为负控。状态库只在一个事务中把精确候选行的 `threads.cwd` 写为目标目录，`project_id` 必须继续为空。3 条杂项可见聊天、原生 Windows 新聊天、标题、消息、工具输出、source/thread_source、归档状态、父子边和 `session_index.jsonl` 均不改。

项目创建完成后，关闭 GPT App、Codex CLI、VS Code 和 VS Code Codex 扩展，在 PowerShell 7 运行：

```powershell
$py = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'
& $py scripts\maintenance\codex_history_migration.py project-bind --dry-run
& $py scripts\maintenance\codex_history_migration.py project-bind --apply --canary
```

`--dry-run` 必须报告 214/144/70/3 分类、217 个文件、SQLite `integrity=ok`、待改 `session_meta`/`turn_context` 数量，并报告 `event_msg` 负控；未知 `cwd`、候选缺失、哈希不符或数据库非 `ok` 会停止。canary 固定为：

- `01a04b61-4eca-7260-8437-161edde533cb`：当前已漂移的可见会话；
- `019dd835-d845-7031-b572-2999535d377f`：只需修 `turn_context`，保留历史 `event_msg`；
- `019f927b-dc1c-7d42-b37a-a8af0b597320`：归档 subagent，保留 source/thread_source 和归档身份。

重开 GPT App 后抽查这三条会话：漂移会话应映射到唯一的 `codex-original · Windows` 项目；可见/归档身份应正确；会话应能读取 `D:\codex-original\AGENTS.md`。再次完全退出 App 后运行：

```powershell
& $py scripts\maintenance\codex_history_migration.py project-bind --verify
& $py scripts\maintenance\codex_history_migration.py project-bind --apply --complete
& $py scripts\maintenance\codex_history_migration.py project-bind --verify
& $py scripts\maintenance\codex_history_migration.py verify
```

`project-bind` v2 manifest 独立于原迁移 manifest，记录每条候选文件的修复前/后哈希、原始数据库 `cwd`、改写计数和基线父子边；原迁移 manifest 保持不可变。`session_index.jsonl` 只产生审计观察，不参与成功判定，也不自动重建。若 App 在 canary 或 complete 后仍找不到正确 `cwd` 的历史会话，立即停止，不降级写 `project_id`，单独处理索引问题。

## 已执行顺序（留作审计）

1. 关闭 GPT App、Codex CLI、VS Code 及 VS Code Codex 扩展窗口。迁移工具会再次检查进程，发现任何 `ChatGPT.exe`、`Codex.exe`、`codex-cli.exe` 或 `Code.exe` 就停止。
2. 在 PowerShell 7 中进入 `D:\codex-original`，先运行：

   ```powershell
   $py = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'
   & $py scripts\maintenance\codex_history_migration.py dry-run
   ```

3. 执行三条 canary：

   ```powershell
   & $py scripts\maintenance\codex_history_migration.py apply --canary
   ```

4. 打开 GPT App，检查最早、最近和最大的一条历史会话。确认标题、消息、工具记录、工作目录和项目归属正确；检查完成后再次关闭 GPT App。
5. 继续全量迁移：

   ```powershell
   & $py scripts\maintenance\codex_history_migration.py apply --complete
   & $py scripts\maintenance\codex_history_migration.py verify
   ```

6. 重新打开 GPT App，使用 Search chats 按标题或短语查找历史会话；项目归类按上面的 folder-bound v2 步骤执行。

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

项目绑定 canary 或 complete 失败时，关闭 GPT App 后运行：

```powershell
& $py scripts\maintenance\codex_history_migration.py project-bind --rollback
```

项目绑定回滚只恢复快照中记录的 214 条候选 transcript 和候选行原始 `cwd`，不写入任何 `project_id`，不覆盖绑定后产生的新聊天。被替换版本先放入独立 `rollback-quarantine`，独立绑定快照、原迁移 manifest 和已迁入的 transcript 均保留。回滚前若当前候选文件哈希既不是修复后哈希也不是原始哈希，工具会拒绝覆盖。

## 验收边界

SQLite、JSONL、哈希、ID、路径、元数据改写范围和父子/归档不变量由工具自动验证；GPT App 中的项目显示、长会话渲染、搜索和继续会话必须人工检查。项目归属使用本地 Codex 的 folder-bound `cwd` 过滤，不是 OpenAI 公开的 Codex→Project 导入 API，因此必须保留快照。知识层是脱敏导航，不替代原始 transcript；新聊天按需检索历史，不能把全部历史一次性加载进上下文窗口。
