# ChatGPT ↔ Codex Windows 双项目桥接手册

更新时间：2026-09-08

本手册描述双项目的传输与职责边界。它不记录当前科研结论；当前 formal runtime、活动
OLS-native 研究和允许动作/停止边界只以本地 `CURRENT_STATUS.md` 与 `current_handoff.md`
为准，也不把两个项目伪装成实时共享上下文。

## 两个项目的职责

| 项目 | 类型 | 用途 |
| --- | --- | --- |
| `codex-original · ChatGPT` | 云端 ChatGPT 项目 | 讨论、联网检索、方案设计、总结和交接块 |
| `codex-original · Windows` | GPT App 本地 Codex 项目 | 源码、Git、测试、CUDA、实现和本地事实核验；formal runtime 与活动 research 均在此核验 |

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

六个 Sources 文件必须作为同一个 `snapshot_id + git_head` 的完整集合使用。旧副本、重复
同名文件或混合快照不进行调和，直接视为历史/无效并要求完整替换；不能按时间戳自动
选择“最新”副本。

每次交给本地 Codex 的任务必须使用 Delta + One Gate：
1. 标注 `MODE=VERIFY|PLAN|IMPLEMENT|REPORT`；`MODE=IMPLEMENT` 表示请求范围已批准。
   Codex 不得扩大研究问题、统计门槛、正式权限或计算预算；在原范围内可跳过已完成工作
   或采用等效的更小实现，超出这些边界才需要新的 `PLAN` 交接。
2. 先写状态增量，只列最多五项会改变本轮决策的变化。
3. 只定义一个会改变下一动作的主 Gate，并写明消费者、输入 lineage、
   允许/禁止动作、阈值、失败语义和唯一下一动作。
4. 函数位置、当前 HEAD、helper 可用性和实际路径交给本地 Codex 核验，
   不在提示中重复粘贴可发现事实。
5. Short Prompt 默认不预填 artifact SHA-256 清单；由 Codex 自动核验
   branch、HEAD、worktree 和 Git 增量。
6. 只有当 Gate 的 evidence consumer 要求精确 artifact 身份时，才核验指定
   Hash；未要求时反馈 hash_not_required，缺失或歧义返回 blocked，不一致返回
   invalid_harness，不自动选择 latest。
7. 先标注模式，再使用短提示的五段结构：状态增量、One Gate、执行边界、验收与停止、
   反馈格式。
8. Short Prompt 通常为 300–1,000 tokens；超过 1,500 不再称为 Short Prompt，必须拆 Gate，
   或满足完整规范的升级条件后改用 Long Prompt；Long Prompt 仍只能有一个主 Gate。

完整规范：
docs/operations/delta_one_gate_short_prompt.md
```

所有云端讨论应从该云端项目内部新建聊天；不要使用 Quick chat 作为长期项目记录。

## ChatGPT 到 Codex

1. 在 `codex-original · ChatGPT` 中只压缩基准之后的决策相关 Delta；同一对话内快照
   未变时不重新初始化，快照变化时只读取新快照头和当前 Gate 所需摘要。
2. 把本轮问题压缩为一个 One Gate，指定真实消费者和唯一下一动作，并标注模式。
3. 生成 Short Codex Prompt，并将完整细节放在本地规范或明确命名的输入中。
4. 在 `codex-original · Windows` 中新建 Codex 任务，将短提示作为任务输入。
5. Codex 读取 `AGENTS.md`、`docs\PROJECT_MEMORY.md`、当前 Git 状态和适用的 scoped 合同，
   只按 Gate 需要读取状态小节和命名 evidence。
6. Codex 只执行一次 bounded preflight：核对 branch/HEAD/worktree、Gate 输入 lineage、
   授权和保护条件；之后执行已授权范围、在原权限内缩小/跳过，或返回 `BLOCKED`/差异报告，
   不得静默转入重新规划。
7. Codex 先核对真实 HEAD、branch、dirty scope、lineage 和授权；快照冲突时以本地事实为准。
   Hash 仅按 Gate consumer 需要定向核验，不自动选择 latest。
8. Codex 只执行授权范围，运行最小充分验证，并按固定反馈格式返回。

Codex 完成后返回：

```text
结果：
本地增量核验：
- Hash：<hash_not_required|hash_verified|hash_mismatch|hash_required_missing>
- 模式/预检：<MODE=...；preflight=passed|blocked>
Gate 判定与唯一下一动作：
修改文件：
验证结果：
未验证内容与已知风险：
项目记忆与 Sources 状态：
- publication_state=<current_clean|current_dirty_planning_only|stale_or_conflicting>
```

短提示、Long Prompt 升级、失败语义和预算规则见
[Delta + One Gate + Short Codex Prompt](operations/delta_one_gate_short_prompt.md)。

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

只有工作树已按 bridge 合同验证、`refresh_ok`、`status=current` 且 manifest 的
`allow_dirty=false`（即 `publication_state=current_clean`）时，才允许用户人工上传。
默认上传且仅上传：

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
