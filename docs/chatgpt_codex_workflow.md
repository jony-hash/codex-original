# ChatGPT ↔ Codex Windows 双项目桥接手册

更新时间：2026-08-30

本手册描述当前的双项目工作流。它解决“本地 Codex 项目不能直接进行普通 ChatGPT 对话”的
界面限制，但不把两个项目伪装成实时共享上下文。

## 两个项目的职责

| 项目 | 类型 | 用途 |
| --- | --- | --- |
| `codex-original · ChatGPT` | 云端 ChatGPT 项目 | 讨论、联网检索、方案设计、总结和交接块 |
| `codex-original · Windows` | GPT App 本地 Codex 项目 | 源码、Git、测试、CUDA、实现和本地事实核验 |

本地项目的主文件夹是 `D:\codex-original`，辅助文件夹是
`C:\Users\24567\Documents\Codex\codex-original-history-context`。云端项目不能直接
读取这些路径；它只使用人工上传的 Sources 快照。项目名称相似只便于识别，不提供技术
绑定。

## 事实优先级

发生冲突时依次使用：

1. `AGENTS.md` 和适用的 scoped `AGENTS.md`；
2. 当前 Git、源码、配置、测试和正式 evidence；
3. `docs\CURRENT_STATUS.md` 与 `docs\status\current_handoff.md`；
4. `docs\PROJECT_MEMORY.md`；
5. 外部历史索引和指定 Session 摘要；
6. 云端 ChatGPT 对话和 Sources 快照。

云端 Sources 必须被视为某一个 Git HEAD 的只读快照。没有快照信息、快照过期或与本地
事实冲突时，停止使用云端结论并刷新 Sources。

## 云端 ChatGPT 项目指令

将以下规则放入 `codex-original · ChatGPT` 的项目指令：

```text
本项目只负责 codex-original 的讨论、检索、规划和总结。

Sources 是某个 Git HEAD 的只读快照，不代表实时访问本机。不要声称已经读取、修改、
测试或运行 D:\codex-original，也不要把云端聊天结论当作本地事实。

规划前先识别 Sources 中的 snapshot_id、git_head 和生成时间。若缺失、过期或互相冲突，
明确要求刷新 Sources。

需要交给本地 Codex 执行时，始终输出：
来源快照：
目标：
依据文件：
范围内：
范围外：
约束：
验收标准：
需要核实的不确定项：
建议更新的项目记忆：
```

所有云端讨论应从该云端项目内部新建聊天；不要使用 Quick chat 作为长期项目记录。

## ChatGPT 到 Codex

1. 在 `codex-original · ChatGPT` 中讨论问题并形成计划。
2. ChatGPT 输出带有 `来源快照` 的交接块。
3. 在 `codex-original · Windows` 中新建 Codex 任务，将交接块作为任务输入。
4. Codex 读取 `AGENTS.md`、`docs\PROJECT_MEMORY.md`、当前 Git 状态和适用的 scoped 合同。
5. Codex 先核对交接块中的 `git_head`；不一致时以本地 Git 为准并报告差异。
6. Codex 实现、测试、审查并更新权威状态文档。

Codex 完成后返回：

```text
结果：
修改文件：
验证结果：
未验证内容：
已知风险：
项目记忆更新：
```

## Sources 刷新

活动桥接目录固定为：

```text
C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current
```

该目录不属于源码仓库，也不应添加为本地项目辅助文件夹。使用当前 Windows 环境运行：

```powershell
Set-Location D:\codex-original
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$env:Path = "$envRoot;$envRoot\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\Lib\site-packages\torch\lib;$env:Path"
$py = Join-Path $envRoot 'python.exe'
$historyRoot = 'C:\Users\24567\Documents\Codex\codex-original-history-context'
$bridgeRoot = 'C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current'

& $py scripts\maintenance\chatgpt_project_memory.py refresh `
  --project-root D:\codex-original `
  --history-root $historyRoot `
  --output-root $bridgeRoot

& $py scripts\maintenance\chatgpt_project_memory.py status `
  --project-root D:\codex-original `
  --history-root $historyRoot `
  --output-root $bridgeRoot
```

只有 `refresh_ok` 和 `status=current` 时才允许上传。默认上传且仅上传：

```text
PROJECT_MEMORY.md
CURRENT_STATUS.md
CURRENT_HANDOFF.md
V2_RUNTIME.md
HISTORY_CONTEXT.md
CHAT_INDEX.md
```

不要上传 `.chatgpt-sources-manifest.json`、`CHAT_CATALOG.jsonl`、原始 transcript、全部
历史摘要、`.codex` 数据、Git 元数据、outputs、临时目录或隔离目录。

需要历史细节时，在刷新命令中追加一个明确的 `--session-id <id>`；上传该临时摘要后，
任务结束即从云端项目移除。未知 Session ID、过期包、dirty 状态变化、密钥扫描命中或
输出目录中出现未知文件时，停止上传。

## VS Code 与运行环境

查看或手工编辑代码时，VS Code 直接打开 `D:\codex-original`。它与本地 Codex 使用同一
工作树，不创建副本；不要打开已注销的 WSL 路径，也不要使用旧 Conda 环境。

当前解释器固定为：

```text
D:\ANACONDA\envs\optical-generative-models-win\python.exe
```

VS Code 是可选编辑器；GPT App 本地项目是 Codex 执行入口。Ubuntu 已注销，
`docker-desktop` 不是项目运行时。

## 停止条件与恢复

- Sources 状态不是 `current`：不上传，先刷新。
- 云端快照 HEAD 与本地 HEAD 不一致：以本地 Git 为准，重新生成交接。
- 需要旧 WSL 文件或历史输出：只从不可变归档恢复明确命名的材料。
- D: 可用空间低于 10 GiB：停止大型任务。
- 不修改 Codex SQLite、历史 transcript、训练算法或正式 outputs。

桥接工具只负责本地生成和验证，永不自动上传或修改云端项目。云端项目的 Sources 替换
必须由用户人工确认。
