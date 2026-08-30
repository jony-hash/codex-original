# ChatGPT ↔ Codex 项目桥接手册

更新时间：2026-08-30

本手册描述一个受控的“双项目桥接”，不是本地文件的自动云同步。ChatGPT 项目保存
讨论、规划和总结；Codex 本地项目访问真实工作树并执行修改和验证。ChatGPT 项目使用
Sources，用户必须人工上传或替换由本地工具生成的 Markdown 包。

## 两个项目

| 角色 | 项目 | 内容 |
| --- | --- | --- |
| 讨论层 | `codex-original · ChatGPT` | 讨论、规划、总结、知识问答；只读受控 Sources |
| 执行层 | `codex-original · Windows` | 主目录 `D:\codex-original`；读代码、改代码、测试 |

本地项目的主目录是 `D:\codex-original`。历史上下文目录只能作为辅助目录和导航层：
`C:\Users\24567\Documents\Codex\codex-original-history-context`。它不替代 Git 工作树，
也不应把旧 `.codex` 项目副本或原始 transcript 当作当前事实源。

## 第一次设置 ChatGPT 项目

在 GPT App 中：

1. 新建标准 ChatGPT 项目，命名为 `codex-original · ChatGPT`。
2. 使用以下项目指令：

   ```text
   本项目用于讨论、规划和总结 codex-original。
   PROJECT_MEMORY.md 是入口，但不能覆盖 AGENTS.md、CURRENT_STATUS.md、
   CURRENT_HANDOFF.md、Git 或正式实验证据。

   不要声称已经读取或修改本机文件。需要实施时，输出 Codex 交接块。
   旧 WSL 路径只表示历史记录，不是当前工作目录。
   历史摘要必须按 Session ID 选择，不得一次性假定全部历史有效。
   ```

3. 在本地执行 Sources 工具的 `build`，将生成包中的六个 Markdown 文件上传到该
   ChatGPT 项目：`PROJECT_MEMORY.md`、`CURRENT_STATUS.md`、`CURRENT_HANDOFF.md`、
   `V2_RUNTIME.md`、`HISTORY_CONTEXT.md`、`CHAT_INDEX.md`。
4. 不上传内部 manifest、`CHAT_CATALOG.jsonl`、全部摘要、原始 transcript、代码树或
   `.pytest-tmp` 文件。
5. 所有桥接对话都从这个项目内部新建，不使用 Quick chat。Quick chat 不属于该项目
   的 Sources/聊天边界。

## 生成和验证 Sources 包

在 PowerShell 7 中使用用户实际的 Windows Conda 解释器：

```powershell
Set-Location D:\codex-original
$py = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'

& $py scripts\maintenance\chatgpt_project_memory.py build `
  --project-root D:\codex-original `
  --history-root C:\Users\24567\Documents\Codex\codex-original-history-context `
  --output-root C:\Users\24567\Documents\Codex\codex-original-history-context\chatgpt-sources

& $py scripts\maintenance\chatgpt_project_memory.py verify `
  --project-root D:\codex-original `
  --history-root C:\Users\24567\Documents\Codex\codex-original-history-context `
  --output-root C:\Users\24567\Documents\Codex\codex-original-history-context\chatgpt-sources
```

`build` 只复制明确列出的来源，记录 Git HEAD、dirty paths、来源/输出哈希和选中的
Session ID。`verify` 会拒绝来源变化、输出篡改、Git 状态变化、秘密模式、未知文件、
缺失摘要、错误项目根目录或包内临时目录。验证失败时不要上传旧包。

需要历史上下文时显式选择 Session ID，例如：

```powershell
& $py scripts\maintenance\chatgpt_project_memory.py build `
  --project-root D:\codex-original `
  --history-root C:\Users\24567\Documents\Codex\codex-original-history-context `
  --output-root C:\Users\24567\Documents\Codex\codex-original-history-context\chatgpt-sources `
  --session-id 019daa26-d2b2-7501-9990-ae38838e34a6
```

这只加入对应的一份 `summaries/<session-id>.md`，不会默认加入 147 份摘要。

## 标准交接

ChatGPT 不声称访问本机，也不直接宣称已经实施。需要 Codex 执行时，使用：

```text
目标：
依据文件：
范围内：
范围外：
约束：
验收标准：
需要核实的不确定项：
建议更新的项目记忆：
```

Codex 完成后返回：

```text
结果：
修改文件：
验证结果：
未验证内容：
已知风险：
项目记忆更新：
```

用户随后用 Codex 重新 `build`/`verify`，在 ChatGPT 项目中删除旧 Sources 并上传新包中的
六个 Markdown 文件。上传是人工确认步骤；本地工具不调用 ChatGPT 上传 API。

## 故障处理

- ChatGPT 报告的根目录不是 `D:\codex-original`：先检查 Sources 是否过期并重新 verify，
  不要把旧 WSL 路径当作修复目标。
- `verify` 报 Git 状态或来源变化：重新 build，确认变化属于预期，再人工替换 Sources。
- `verify` 报秘密、未知文件或 `.pytest-tmp`：停止上传，检查来源和输出目录；不要静默
  脱敏权威文件或覆盖未知文件。
- 历史问题只按 Session ID 加入摘要；若仍无法定位，会话索引问题必须单独审计，不能
  修改 Codex SQLite、`project_id` 或自动重建索引。
- Codex 读取本地文件时，以 `AGENTS.md`、Git 和验证结果为准；ChatGPT Sources 仅是
  受控上下文副本。

## 当前人工验收

新项目设置后，分别验证：ChatGPT 能从 Sources 说出真实工作区、v2 正式边界、当前测试
状态和旧 WSL 路径语义；Codex 能读取 `AGENTS.md`、`docs/PROJECT_MEMORY.md` 和 Git 状态；
选定一个 Session ID 后 ChatGPT 能定位该摘要但不把它当成当前事实。历史迁移的 SQLite、
217 条会话、214 条 folder-bound 修复、归档状态和父子关系不应因 Sources 桥接改变。
