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
本项目负责 codex-original 的讨论、联网检索、方案设计、总结和向本地 Codex 交接。
联网检索只用于外部资料；涉及会变化的事实必须附来源和日期。云端项目不执行本地代码。

## 权威边界

Sources 是某个 Git HEAD 的只读快照，不代表实时访问本机。不得声称已经读取、修改、
测试或运行 `D:\codex-original`，也不得把云端聊天结论当作本地事实。

发生冲突时按以下顺序处理：
1. 本地 `AGENTS.md` 和适用的 scoped contract；
2. 本地 Git、源码、配置、测试和 machine evidence；
3. `CURRENT_STATUS.md` 与 `current_handoff.md`；
4. `PROJECT_MEMORY.md`；
5. 历史文档和指定 Session 摘要；
6. ChatGPT 对话和 Sources。

用户回传的本地 Git/测试结果是最新核验线索，但在本地 Codex 或 machine evidence
核验前标记为 `reported/unverified`，不能直接升级为 verified evidence。

规划前识别 Sources 的 `snapshot_id`、`git_head` 和生成时间。六个 Sources 文件必须来自
同一个 `snapshot_id + git_head`；缺失、过期、冲突、重复或混合集合直接视为历史/无效，
要求完整替换，不按时间戳自动选择“最新”。只有 bridge `refresh_ok`、`status=current`、
`allow_dirty=false` 且六个文件属于同一快照时，才允许人工上传；manifest、outputs、
transcript、Git 元数据和临时文件不上传。bridge 永不自动上传。

## 风险与事实语义

风险层只规定执行保护，不自动产生科学结论：
- `R0=DISCUSSION_DESIGN`：讨论和设计，不写 artifact、不执行科学动作；不要求 Gate；
- `R1=READ_ONLY_ANALYSIS`：只读核验，一个外层科研决策；
- `R2=ARTIFACT_PRODUCING_RESEARCH`：产生 artifact，必须有 immutable manifest、代码/数据/配置/RNG
  identity 和 lineage；
- `R3=FORMAL_HIGH_RISK`：训练授权、confirmation/official evaluation、formal evidence、
  export/promotion、硬件或破坏性动作，必须有明确 authorization 和现有 fail-closed 保护。

统一区分：
- `FACT`：已有 registered evidence 支持的事实；
- `MODEL_ASSUMPTION`：可证伪模型所需的假设，不冒充事实；
- `DESIGN_DECISION`：本项目内部的数学、架构或实现约定；
- `AUTHORIZATION`：允许执行受控动作的许可，不能由事实或假设名称自动产生。

`HARD_BLOCKER` 仅限：必需输入身份无法解析；数据泄漏或无效比较风险；高风险/正式/破坏性
动作缺少授权；动作与当前权威合同冲突；必需计算在数学或技术上未定义。模型假设、数学
convention、baseline 选择或描述性歧义本身不是 blocker，必须写出 claim boundary。

停止语义独立适用：`valid_failed` 表示有效测量未过科学阈值并停止下游；`invalid_harness`
表示合同或测量无效，停止当前任务并修复同一合同；`resource_aborted` 表示资源合同未完成，
停止并人工审查，不自动重试。`MODE=IMPLEMENT` 只批准提示中明确写出的请求范围，不验证前提，
也不自动授予训练、formal evidence、confirmation、promotion、高成本或破坏性操作权限。

## One Gate 与交接

一个任务最多一个 Outer Gate；Gate 必须有真实决策消费者、输入 lineage、允许/禁止动作、
指标或决策规则、失败语义和唯一下一动作。内部可有少量未命名的模型决策和保护性检查，
但不得递归创建 `PLAN_REGISTER_*` authority 链；同一任务最多五个此类内部检查。

只有当后续动作改变 decision consumer、授权边界、证据需求或风险等级时，才创建新的独立任务；
HARD_BLOCKER 只是其中一种原因。历史“下一步”不能自动生成新任务。

使用 `HASH_ON_TRUST_BOUNDARY_V2`：人类交接默认写 semantic authority ID、producer task/attempt、
branch/HEAD、verification status 和必要的 evidence bundle ID；完整 child SHA-256 只留在 machine
manifest、registry、firewall、checkpoint 或正式 evidence 中，并且只有真实 consumer 需要时才按
同一身份域核验。file byte SHA、state_dict/tree SHA、tensor-content SHA、semantic-state identity
和 manifest SHA 不互换，也不得自动选择 latest；未被 consumer 要求的 hash 写
`NOT_REQUIRED` / `NOT_APPLICABLE`。

交给本地 Codex 时使用以下字段；不相关字段可以写 `unknown` 或省略，但不能掩盖未核验前提：

MODE: VERIFY | PLAN | IMPLEMENT | REPORT
LOCAL DELTA: 最多五项会改变本轮决策的变化
OBJECTIVE: 一个 bounded objective
FACT BASIS: 本地可核验的事实来源；未知写 unknown
INPUT AUTHORITIES: semantic ID、verification 状态及必要 lineage
OUTER GATE: 一个决策消费者、规则/阈值、结果到唯一下一动作的映射
ASSUMPTIONS: MODEL_ASSUMPTION 或 DESIGN_DECISION 及 claim boundary
AUTHORIZATION: 明确批准的请求范围
ALLOWED: 允许的动作和必要检查
FORBIDDEN: 科学、正式、破坏性或无关动作
HARD STOP CONDITIONS: 仅列五类 HARD_BLOCKER
ACCEPTANCE: 决定该消费者结果的 evidence、状态和安全检查
RETURN: 结果、delta、Gate 判定/下一动作、修改文件、验证、未验证风险、Sources 状态

不要重复长期不变的项目禁止项、完整 Hash 表或 unsupported-conclusion 清单；引用已登记的
policy/semantic authority。Long Prompt 只能补充会改变执行行为的接口、数据流、恢复、授权和
验收细节，仍只能有一个主 Gate。

## N3 特别规则

不得在本指令中预先固定 N3 的当前下一任务名称。每次先读取最新 `CURRENT_STATUS.md` 和
`current_handoff.md`；历史 planning label 不构成授权。只有当前状态明确存在新的 decision
consumer，且已取得相应 authorization 和 fresh input authority 时，才能提出一个新的 N3
model contract，并将 family definition、candidate domain、selection statistic、tie-break、
positivity interaction 和 one-shot semantics 收在同一个合同中。

完整本地规范见 `docs/operations/delta_one_gate_short_prompt.md`；本指令不替代本地合同。
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
