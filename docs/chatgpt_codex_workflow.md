# ChatGPT ↔ Codex Windows 原生工作流

更新时间：2026-08-30

本手册描述唯一主要入口：GPT App 中的本地项目 `codex-original · Windows`。它不是本机
文件的自动云同步；当前事实仍以项目文件、Git 和验证结果为准。

## 单一本地项目

| 项目 | 位置 | 用途 |
| --- | --- | --- |
| `codex-original · Windows` 主文件夹 | `D:\codex-original` | 源码、Git、测试、训练和 `AGENTS.md` |
| 同项目辅助文件夹 | `C:\Users\24567\Documents\Codex\codex-original-history-context` | `HISTORY_CONTEXT.md`、`CHAT_INDEX.md`、`CHAT_CATALOG.jsonl` 和摘要导航 |

主文件夹决定默认工作目录、Git 和 `AGENTS.md` 发现。辅助目录只是历史导航层，不替代
源码、`docs\PROJECT_MEMORY.md` 或原始 Codex transcript。外部历史目录刻意使用
`HISTORY_CONTEXT.md`，避免与仓库权威的 `docs\PROJECT_MEMORY.md` 混淆。

ChatGPT 模式用于讨论、规划和总结；Codex 模式在同一 GPT App 本地项目中读取工作树、
实现修改、运行验证和更新项目记忆。聊天上下文不会自动覆盖本地事实，长期状态必须写入
`AGENTS.md`、`docs\CURRENT_STATUS.md`、`docs\PROJECT_MEMORY.md` 或适用的交接文档。

## 打开和查看源码

在 GPT App 中选择 `codex-original · Windows`，新建 Codex 任务即可执行。查看或手工编辑
代码时，用 VS Code 打开 `D:\codex-original` 这个同一目录即可；VS Code 只是可选查看器，
不要重新打开已经注销的 WSL 工作区，也不要让它切换到旧 Conda 环境。首次打开后，在
VS Code 执行 `Python: Select Interpreter`，选择：

```text
D:\ANACONDA\envs\optical-generative-models-win\python.exe
```

用户设置中不应保留不存在的 `D:\ANACONDA\python.exe`。不需要 VS Code 才能运行 GPT App
Codex；两者都直接作用于 `D:\codex-original`，因此编辑前后用 Git diff 检查即可。

## Windows 执行入口

PowerShell 7 中推荐显式使用当前解释器：

```powershell
Set-Location D:\codex-original
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$env:Path = "$envRoot;$envRoot\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\Lib\site-packages\torch\lib;$env:Path"
$py = Join-Path $envRoot 'python.exe'

& $py -m pytest -q
& $py -m ldm_project.v2 --help
```

不要在未激活环境的终端使用裸 `python` 或裸 `pytest`；当前它们可能解析为系统 Python
3.14。旧环境 `D:\ANACONDA\envs\optical-generative-models` 保留作 Legacy 恢复参考，
不属于本项目运行时。

## ChatGPT 讨论到 Codex 执行

ChatGPT 讨论结果应落到一个清晰的交接块，再在同一本地项目中新建 Codex 任务：

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

Codex 执行前读取 `AGENTS.md`、`docs\PROJECT_MEMORY.md` 和当前 Git 状态；执行后回写：

```text
结果：
修改文件：
验证结果：
未验证内容：
已知风险：
项目记忆更新：
```

新任务不要依赖旧聊天自动携带全部上下文。需要历史细节时，先在辅助目录的
`CHAT_INDEX.md` 找到 Session ID，再在 GPT App 的本地 Codex 项目打开原始会话；历史摘要
只用于导航，不能覆盖当前 Git 和验证结果。

## 可选的离线 Sources 导出

Sources 不是当前项目的主工作流，也不是自动同步机制。只有用户明确需要把受控摘要提供给
另一个 ChatGPT 对话时，才在 PowerShell 7 中运行：

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

导出器现在从辅助目录读取 `HISTORY_CONTEXT.md`，并将其输出为 Sources 包内的
`HISTORY_CONTEXT.md`。验证失败或包过期时停止使用，不上传内部 manifest、原始 transcript、
全部摘要、代码树或临时文件；上传和替换仍由用户人工完成。

## 故障与停止条件

- 默认 Python/pytest 版本不是当前 Windows 环境：立即停止，检查解释器和 PATH，不运行训练。
- Git 工作树、`AGENTS.md`、当前状态或实验事实与交接不一致：以本地文件和验证结果为准，
  先更新交接，不把聊天内容当成授权。
- 需要旧 WSL 路径、旧 checkpoint 或历史输出：只从保留的归档中恢复明确命名的材料，不把
  它们展开成当前工作树。
- D: 可用空间低于 10 GiB：停止大型作业；不要删除归档、旧 Conda 环境或 Codex 历史。
- Ubuntu 已注销，`docker-desktop` 不是项目运行时；不从 WSL 开发、训练或生成输出。

## 验收边界

代码、Git、环境、测试和 CUDA 由 Codex 本地验证；GPT App 中项目显示、历史搜索、长会话
渲染和继续会话需要用户人工检查。VS Code 只查看同一 Windows 工作区，不产生第二份源码或
第二套运行入口。
