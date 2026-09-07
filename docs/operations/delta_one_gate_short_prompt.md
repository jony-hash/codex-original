# Delta + One Gate + Short Codex Prompt

版本：v1.1
更新时间：2026-09-07
性质：GPT → Codex 的项目治理与交接协议，不是科研证据、训练授权或 formal promotion 合同。

## 1. 目的与边界

本协议解决三个重复成本：

- ChatGPT 和 Codex 重复初始化同一个项目；
- 一个任务同时携带多个独立科研问题，导致执行代理重新规划；
- 交接提示重复粘贴已知规则、历史正文和可由本地核验的事实。

协议只压缩交接和上下文，不压缩必须保留的目标、证据、阈值、授权边界、失败语义或停止条件。它不改变 AGENTS.md、scoped AGENTS.md、科研 hypothesis ledger、正式 evidence、训练 Gate 或 outputs 管理规则。普通研究交接采用风险分层、语义 evidence alias 和一个外层 Gate；R3 formal/high-risk 仍使用现有严格的 fail-closed 合同。

权威顺序仍由 AGENTS.md 定义：本地执行合同、当前 Git/源码/配置/测试/evidence、当前状态与 handoff、项目记忆、历史导航，最后才是云端 ChatGPT 与 Sources。Sources 是带 Git 快照的只读规划材料，不是实时工作树。

完整桥接说明见 ../chatgpt_codex_workflow.md。本文件是 Delta、One Gate、短提示和反馈格式的唯一完整规范；PROJECT_MEMORY.md 只保留供云端快速消费的摘要模板。

## 2. 两端职责

| 参与方 | 负责 | 不负责 |
| --- | --- | --- |
| ChatGPT 项目 | 压缩问题、形成一个 Gate、列出精确输入/边界/验收、输出短交接 | 声称已读取或运行本地工作树；代替 Codex 判断当前 Git/evidence |
| 本地 Codex 项目 | 核对真实 Git 和文件、复用既有实现、执行授权动作、验证、回传事实 | 重新设计已确定的 Gate；用旧 Sources 覆盖本地事实；自动扩大科研范围 |
| Sources bridge | 生成带快照的六文件上传包 | 自动上传、修改云端项目、提供实时同步 |

如果本地事实与 Sources、历史摘要或聊天结论冲突，使用本地事实，记录差异，并将快照标为 stale；不要把旧结论重写成当前结论。

### 2.1 执行模式与单次预检

交接可标注一个执行模式：`MODE=VERIFY|PLAN|IMPLEMENT|REPORT`。其中
`MODE=IMPLEMENT` 表示 One Gate、允许动作、阈值、停止条件和反馈要求已经批准；
Codex 不得因为发现更多背景、历史或可能性而重新设计 Gate。只有人工批准的新交接
才能把任务重新置于 `PLAN`。

`MODE=IMPLEMENT` 任务只执行一次 bounded preflight。预检只允许核对：

1. branch、HEAD、worktree 和交接基准之后的 Git 增量；
2. Gate 明确列出的输入、producer/lineage 和授权；
3. 本任务直接适用的 scoped contract；
4. 必须的安全、资源、finite 和负控保护条件。

预检结束后只允许进入 `IMPLEMENT` 或 `BLOCKED`：满足条件就执行已批准动作；不满足
就返回最小解阻动作。禁止第二轮开放式项目初始化、为“更全面理解”继续搜索历史，或
从 `IMPLEMENT` 静默转入 `PLAN`。

## 3. 标准生命周期

### 3.1 ChatGPT 规划阶段

ChatGPT 只做以下压缩：

1. 如果当前对话已经核验相同的 `snapshot_id @ git_head`，只消费新的状态 Delta；
   快照变化时只读取新快照头、工作流摘要和当前 Gate 输入；
2. 从上一个已确认快照或 handoff 提取基准，并只列出基准之后会改变当前决策的状态差分；
3. 把本轮压缩为一个会改变下一动作的主 Gate；
4. 明确允许、禁止、验收、停止和反馈字段；
5. 生成 Short Codex Prompt。

函数位置、helper 是否存在、当前 HEAD、dirty paths 和实际 artifact 路径属于本地可发现事实，除非它们是 Gate 输入，否则不应在提示中预先展开。

### 3.2 Codex 执行阶段

收到短提示后，Codex 只执行一次 bounded preflight：

1. 读取根 AGENTS.md、docs/PROJECT_MEMORY.md 和本任务适用的 scoped contract；
2. 运行最小 Git 核验：status、rev-parse HEAD，必要时查看基准之后的提交；
3. 只读取 CURRENT_STATUS.md、current_handoff.md、hypothesis ledger 和 Gate 明确列出的相关小节/证据；
4. 核对 Gate 输入、lineage、授权以及安全/资源保护条件；
5. 预检结束后：满足条件则执行已授权动作并验证，不满足则返回 blocked 或差异报告；
6. 按固定反馈格式返回结果，不重新输出完整项目历史。

除非 Gate 明确需要，不默认读取完整 current_handoff.md、全部历史摘要、全部日志或全部 outputs。

## 4. Delta 状态增量头

每次 ChatGPT → Codex 交接必须以以下结构开始：

~~~
状态增量：
- 云端基准：<snapshot_id> @ <git_head>
- 本地核验：<branch> @ <head>；worktree=<clean|dirty>
- 决策相关变化：<最多 5 项>
- 保持不变：<关闭路线、冻结边界、授权标志>
- 权威性：<verified_local|stale_snapshot|blocked>
~~~

规则：

- 云端基准只能填写实际已知的 Sources 快照；缺失时写 unknown，不能猜测。
- 本地核验在交接阶段可写“待 Codex 核验”；Codex 返回时必须填写真实 branch、HEAD 和 dirty scope。
- 决策相关变化只记录会改变本轮 Gate、输入 lineage、权限或唯一下一动作的变化，最多五项。
- 保持不变只记录仍然有效的高价值不变量，例如 training_authorized=false、冻结 checkpoint、关闭路线或不改 formal OLS。
- stale_snapshot 表示快照落后但本地可以继续核验；blocked 表示缺少输入、授权、资源或安全条件，不能安全继续。

### 4.1 按需哈希与本地增量核验

- Short Prompt 默认不携带 artifact SHA-256 清单。交接只填写实际已知的
  snapshot_id @ git_head；未知值写 unknown，不猜测。
- Codex 每次启动自动核验 branch、HEAD、worktree，以及交接中已知基准之后的
  Git 增量。Git 增量核验不等于对所有 artifact 做 Hash 扫描。
- ChatGPT 必须命名 Gate 所需的输入、artifact 类型和真实 evidence consumer，
  但不必预先填写 producer digest。Codex 从本地权威状态和 evidence 解析
  producer/lineage，并且不自动选择 latest。
- 只有当 Gate 的 evidence consumer 要求精确 artifact 身份时，Codex 才核验
  指定 artifact 的 Hash；该操作在本次任务中只做一次。反馈在本地增量核验中
  写 hash_verified。
- Gate 不要求 Hash 时，反馈写 hash_not_required。要求 Hash 但输入缺失或有
  歧义时返回 blocked；记录的 Hash 与实际 artifact 不一致时返回
  invalid_harness，不得解释为模型科学失败。
- 正式 evidence 和 Sources bridge manifest 中已经存在的 Hash 是 provenance
  与完整性边界，继续保留；本规则只禁止把无关 Hash 清单复制进每次交接。

本节的固定政策 ID 为 `HASH_ON_TRUST_BOUNDARY`。人类交接只显示 semantic authority ID、
branch、适量的 Git HEAD、evidence bundle ID 与 verification status；不重复 child artifact
的完整 SHA-256 表。外部不可变数据、data partition/firewall、calibration payload、
checkpoint、不可变 experiment artifact 和 evidence bundle 仍由其 machine manifest/registry
保存完整 SHA-256。Git-tracked source/docs 通常由 Git lineage 标识，除非已有 consumer 明确
要求逐文件 digest。

当前审计分类如下；分类只改变呈现与停止语义，不删除历史 evidence：

| 类别 | 当前用途 |
| --- | --- |
| `MACHINE_REQUIRED` | `research_run_manifest.v1` 的输入、`EvidenceRecordV2.input_hashes`/context、`ArtifactRefV2`、data firewall、checkpoint、不可变 artifact 与 Sources manifest 的完整性校验 |
| `HUMAN_DISPLAY_ONLY` | branch、HEAD、dirty scope、semantic evidence ID、consumer、verification status 和短反馈 |
| `DUPLICATED` | 在 PROJECT_MEMORY、CURRENT_STATUS、current_handoff 与 handoff prompt 中重复出现且没有 consumer 读取的 child SHA 表、未变化 flags 与 unsupported-conclusion catalog |
| `NOT_REQUIRED` | R0 讨论设计中的 hash、普通 Git-tracked 文档的逐文件 SHA，以及 hash 数量/文件数量本身作为成功标准 |

### 4.2 Sources 活动快照选择

- 六个 Sources 文件必须作为同一个 `snapshot_id + git_head` 的完整集合处理；不能把
  不同快照的同名文件拼接成活动状态。
- 旧副本、重复副本或无法确认属于同一集合的文件不参与调和，直接视为历史/无效，
  要求用完整六文件集合替换；不能按时间戳自动选择“最新”文件。
- 本地 bridge manifest 是集合完整性校验材料，不上传到 ChatGPT Sources；人工上传时
  只选择该 manifest 对应的六个 Markdown 文件。

### 4.3 风险分层、Gate 分类与 HARD_BLOCKER

普通研究按以下风险层执行；风险层是执行边界，不是新的科研结论：

| 层级 | 名称 | 必需保护 |
| --- | --- | --- |
| `R0` | `DISCUSSION_DESIGN` | 不写 artifact、不做 scientific execution、不要求 Hash；不需要 Gate，假设可直接声明 |
| `R1` | `READ_ONLY_ANALYSIS` | 核验 branch/HEAD/worktree；使用 semantic input authority ID；只有 consumer 要求时才核验精确 Hash；一个外层科研决策 |
| `R2` | `ARTIFACT_PRODUCING_RESEARCH` | immutable output manifest、Git/code identity、data/config/RNG identity、root evidence bundle digest 与严格 lineage |
| `R3` | `FORMAL_HIGH_RISK` | training authorization、official/confirmation evaluation、formal evidence consumption、candidate export/promotion、hardware 或 destructive action；继续使用现有 exact evidence 和 fail-closed 规则 |

现有 Gate 按消费者分类为：`HIGH_RISK_AUTHORIZATION`（formal gate、训练授权、promotion
或硬件动作）、`SCIENTIFIC_DECISION`（完整测量改变下游科学动作）、`MODEL_ASSUMPTION`
（可证伪模型假设）和 `INTERNAL_CHECK`（finite、lineage、冻结状态、资源与静态检查）。
后两类不能仅因名称带有 authority 就自动升级为 R3 或新 Gate；正式消费者仍只能通过
`src/ldm_project/v2/application/current/gates.py` 及其既有 evidence readers。

`HARD_BLOCKER` 仅限以下五类：

1. required input identity 无法解析；
2. 会造成 scientific data leakage 或 invalid comparison；
3. destructive/high-cost/formal action 缺少明确 authorization；
4. 当前 authoritative contract 与 proposed action 冲突；
5. 必需计算在数学或技术上未定义。

未被历史 evidence 证明的 model family、distribution、数学 convention、falsifiable baseline
assumption 或 descriptive ambiguity，不自动是 blocker；它们分别登记为 `MODEL_ASSUMPTION`
或 `DESIGN_DECISION`，并写出 claim boundary。已批准的 bounded task 只有遇到上述
`HARD_BLOCKER` 才提前停止；`valid_failed` 仍表示有效测量失败并停止下游，不得改写成模型成功。

### 4.4 规范词汇与 Gate 预算

| 词汇 | 语义 | 允许的结论边界 |
| --- | --- | --- |
| `FACT` | 已由 registered evidence 支持的事实 | 可被 consumer 直接消费 |
| `MODEL_ASSUMPTION` | 构造候选模型所需的、可证伪的 modeling choice | 可登记为 test baseline，不得冒充 FACT |
| `DESIGN_DECISION` | project-specific 的实现或数学 convention | 只约束该 contract，不声称普遍真理 |
| `AUTHORIZATION` | 执行 bounded high-risk 或 evidence-producing action 的许可 | 不能由 FACT/assumption 名称自动产生 |

普通 Codex research task 只允许一个 outer Gate，最多五个未命名的 internal checks/decisions，
最多返回一个 `HARD_BLOCKER`。内部的 model vocabulary、domain、candidate enumeration、
deterministic tie-break 和 assumption registration 在同一 model contract 内闭合，不得递归创建
`PLAN_REGISTER_*` 任务。只有当新的 `HARD_BLOCKER` 需要外部 evidence、执行授权或不同
decision consumer 时，才可创建新的 named Gate。Gate 的定义是“结果改变下游授权的决策”；
单纯的 model definition 不是 Gate。

### 4.5 Semantic evidence alias 与现有 bundle

本项目不新增与现有 evidence/manifest 并行的全局 evidence registry。`research_run_manifest.v1` 的
manifest input binding、`EvidenceRecordV2` 的 context/input hashes、`ArtifactRefV2` 的
path/role/sha256，以及 `canonical_sha256`/`file_sha256` 是现有 machine authority primitives。
`docs/status/research_governance_state.json` 只是这些既有 machine artifacts 的紧凑 alias
index，不承载或替代 evidence。人类可以使用稳定 alias，例如 `OLS_PRODUCER_DATA_V1` 或
`LOW_RANK_SCORE_MODEL_V1`；实际 consumer 必须把 alias 解析到一个明确的 manifest/evidence
record，而不是自动选择 latest。

当 alias 被 machine consumer 注册时，解析记录至少包含：

- semantic ID、exact path、role、schema；
- producer/parent lineage；
- child digest（仅 machine manifest）；
- root/canonical digest（该 bundle 有此概念时）；
- verification status 与允许的 consumer。

人类 handoff 只需写 `input_authority=<semantic-id>` 和
`verification=required|verified|not_required`。若没有真实 consumer 需要 alias index，不得为
展示目的新建 registry 或复制 child hash 表；full Hash 仍留在原 machine manifest。

## 5. One Gate 科研决策合同

一个任务只能有一个会改变下一动作的主科研问题。以下字段必须在交接中给出：

~~~
Gate：
- ID 与单一问题：
- 决策消费者：
- 输入及 producer/lineage（Hash 按需核验；由 Codex 发现并验证）：
- 固定不变量：
- 允许动作：
- 范围外：
- 指标与预注册阈值：
- 结果映射：
  passed -> 唯一下一动作
  valid_failed -> 关闭或人工审查动作
  invalid_harness -> 修复同一 Gate，不作科研结论
  resource_aborted -> 停止并人工审查，不自动重试
  blocked -> 最小解阻动作
- evidence/输出：
- 时间、CUDA、存储预算：
~~~

具体约束：

- 决策消费者必须是真实的人工审查、状态 closeout、后续合同或明确的拒绝决策；计数、hash 数量、文件数量不能单独充当成功标准。
- 负控、lineage、冻结状态、资源、finite 和安全检查是该 Gate 的保护条件，不应伪装成第二个科研问题。
- 若两个结果会分别决定不同后续动作，拆成两个任务；不能在一个提示中并列 Gate A/Gate B。
- passed 只授权合同中写明的唯一下一动作；不自动授权长训练、promotion、生成或新路线。
- valid_failed 是完整有效测量未过科学阈值；保留证据并停止下游消费者。
- invalid_harness 是测量、实现、lineage 或验证合同无效；不把它解释为模型失败，也不通过换数据或放宽阈值重试。
- resource_aborted 只说明资源合同未完成；不把它转换成科学成功/失败。
- 缺少 checkpoint、producer/lineage、数据、硬件或项目负责人授权时，返回 blocked，
  不得静默替换输入。只有合同明确要求精确 artifact 身份时，缺少该 Hash 或
  Hash 与记录不一致才分别按 blocked 或 invalid_harness 处理。

## 6. Short Codex Prompt

推荐的可复制交接格式如下：

~~~
MODE: <VERIFY|PLAN|IMPLEMENT|REPORT>
LOCAL DELTA:
- cloud snapshot (if known): <snapshot_id @ git_head | unknown>
- local branch/HEAD/worktree: <Codex verifies; do not guess>
- decision-changing changes: <at most five>
OBJECTIVE: <one bounded objective>
INPUT AUTHORITIES:
- <semantic-id>; verification=<required|verified|not_required>
- <producer/lineage only when it changes the decision>
OUTER GATE:
- ID/question: <one decision consumer>
- metric/threshold or decision rule: <exact>
- passed/valid_failed/invalid_harness/resource_aborted/blocked -> <single next action>
ASSUMPTIONS:
- <MODEL_ASSUMPTION or DESIGN_DECISION; claim boundary>
ALLOWED:
- <bounded actions and required checks>
FORBIDDEN:
- <scientific, formal, destructive, or unrelated actions>
HARD STOP CONDITIONS:
- <only the five HARD_BLOCKER classes in policy; no child hash table>
ACCEPTANCE:
- <evidence, state, and safety checks that determine the consumer decision>
RETURN:
- <result, local delta, Gate decision/next action, modified files, validation,
  unverified risks, memory/Sources publication state>
~~~

短提示只携带本地不能安全推导且会改变本轮决策的变量。路径、符号、helper、当前状态和
producer digest 由 Codex 按最小搜索核验；已明确的科研选择不重新讨论。完整 child SHA 表、
重复 unchanged flags、unsupported-conclusion catalog 和递归 single-next-Gate prose 不属于
默认 handoff。Hash、失败分类、发布资格和完整反馈字段由本协议固定，不需要每次重新粘贴。

## 7. Token 与上下文预算

这些预算是效率保护线，不得用来删掉 lineage、阈值、停止条件、失败分类或用户授权边界。token 估算依赖实际模型和语言混合情况；无法精确估算时使用字符/行数作为保守代理，并在反馈中说明。

| 内容 | 目标 | 升级/停止线 |
| --- | ---: | ---: |
| Short Prompt | 通常 300–1,000 tokens | 1,000–1,500 仅在无法用规范引用压缩时允许；超过 1,500 不再称为 Short Prompt，必须拆 Gate 或按第 8 节升级为 Long Prompt |
| Long Prompt | 1,500–3,000 tokens | 超过 4,000 tokens：引用独立规范或拆成多个任务 |
| Codex 执行反馈 | 不超过 700 tokens | 保留结论、证据、风险、下一动作，删除过程复述 |
| 本地初始上下文 | 约 12,000 tokens 以内 | 超出时只增加 Gate 必需的文件，并说明原因 |

本地初始上下文的默认组成是根执行合同、项目记忆、最小 Git 差分、相关状态小节和最多三个命名 evidence。完整历史只在 Gate 输入或升级条件明确要求时读取。
Short Prompt 不要求达到 300 tokens，也不应为了达到下限填充固定协议；只保留当前 Gate
必需的变量、边界和验收条件。

连续三项代表性任务记录以下过程指标：

- 交接提示估算 tokens；
- 初始读取的文件/小节数量；
- 规划轮次和首次实质动作前耗时；
- 是否发生返工及返工原因；
- Gate 字段是否完整、反馈是否可消费。

这些指标只由“工作流复盘”消费，不能替代科研 Gate。若三项中至少两项因短提示缺少决策必要字段而返工，修订模板；如果只是本地事实发生变化，更新 Delta，不扩写固定协议。

## 8. 何时升级为 Long Prompt

满足任一条件即可升级，但升级后仍只保留一个主 Gate：

- HEAD 差异改变 Gate、producer/lineage、冻结边界或唯一下一动作；
- 任务跨越公共接口、数据迁移、checkpoint/output policy 或高风险训练/验收语义；
- 短提示已经因缺少关键接口或失败恢复细节发生一次实质返工；
- 必须恢复归档、协调外部资源、处理重叠 dirty changes 或验证硬件；
- 一个目标无法在不丢失决策消费者的情况下压缩为单一 Gate。

Long Prompt 只增加会改变执行行为的细节：接口、数据流、迁移兼容、失败恢复、审批边界和验收样例。它不能重新粘贴完整项目历史，也不能把多个独立科研问题合并为一个“总 Gate”。
仅以“问题复杂”作为理由不足以升级；不满足上述条件时应拆分 Gate，而不是继续堆叠历史正文。

## 9. 失败、回退与恢复

### 9.1 Sources 发布状态

bridge 的 `status` 与 manifest 的 `allow_dirty` 一起解释为发布状态：

| 发布状态 | 判定 | 允许动作 |
| --- | --- | --- |
| `current_clean` | `status=current` 且 `allow_dirty=false` | 可由用户人工替换云端六文件 Sources |
| `current_dirty_planning_only` | `status=current` 且 `allow_dirty=true` | 仅供本地检查和讨论，不得上传或宣称正式同步 |
| `stale_or_conflicting` | `status` 为 stale/invalid/missing，或六文件不属于同一快照 | 停止依赖，先完整刷新/替换集合 |

`refresh_ok` 只表示本次构建和验证完成；正式上传还必须满足
`publication_state=current_clean`。不得把 `allow_dirty=true` 的 `current` 包解释成
clean 快照，也不得用 `--allow-dirty` 绕过发布条件。

| 情况 | 必须动作 | 禁止动作 |
| --- | --- | --- |
| Sources stale | 以本地 Git/evidence 为准，生成新 Delta | 沿用旧 HEAD 或旧结论声称 current |
| Sources invalid | 可以继续已授权的本地工作；报告 bridge 阻塞 | 上传、声明同步成功或用 --allow-dirty 掩盖问题 |
| dirty 与范围无关 | 记录并保护原改动 | 清理、reset、checkout 或覆盖用户改动 |
| dirty 与范围重叠 | 停止并返回最小解阻动作 | 猜测归属或强行合并 |
| 缺失输入/授权或必需的精确 artifact 身份 | 返回 blocked | 自动寻找“最新”替代物或扩大权限 |
| invalid_harness | 修复同一合同并重新验证；Hash 不一致不作科学结论 | 形成科学结论或放宽门槛 |
| valid_failed | 保留 evidence，停止下游 | 自动重训、sweep、promotion 或改写失败 |
| resource_aborted | 停止并人工审查资源合同 | 自动长重试或解释成模型结论 |

协议变更或回退通过后续文档修订/提交完成；不改写正式 evidence、历史 handoff、outputs、SQLite 或 transcript。交接协议本身不授权 commit、Sources refresh 或云端上传；这些动作按用户授权和现有 bridge 合同执行。

## 10. Codex → ChatGPT 固定反馈

~~~
结果：<complete|blocked|passed|valid_failed|invalid_harness|resource_aborted>

本地增量核验：
- HEAD/branch/worktree：
- 相对交接基准的变化：
- Hash：<hash_not_required|hash_verified|hash_mismatch|hash_required_missing>
- Sources 是否 stale/invalid：
- 模式/预检：<MODE=...；preflight=passed|blocked>

Gate 判定与唯一下一动作：
- Gate：
- 判定：
- 消费者：
- 下一动作：

修改文件：
- <绝对路径；没有修改则写 none>

验证结果：
- <最小充分验证及结果>

未验证内容与已知风险：
- <事实、缺口、资源或人工验收>

项目记忆与 Sources 状态：
- <是否建议更新>
- <publication_state=current_clean|current_dirty_planning_only|stale_or_conflicting；是否需要 clean commit 后 refresh/status>
~~~

反馈必须先给结论，再给支持结论所需的最小证据。工具日志、完整 traceback 和过程性搜索结果只在它们改变判定时附上。

## 11. 本项目的落地边界

- 本协议的完整规范位于 docs/operations/；PROJECT_MEMORY.md 只保存稳定导航、durable
  decisions 和本协议入口，不保存每次 handoff 的完整 telemetry。
- CURRENT_STATUS.md 记录当前 branch、objective、state 与 high-level governance；
  docs/status/current_handoff.md 只记录相对上一次交接的 delta、immediate objective/blocker
  与唯一下一动作；详细历史 Hash、telemetry 和 child lineage 留在 machine evidence registry
  或 experiment history。历史正文不因压缩而删除。
- 六文件 Sources 合同保持不变：PROJECT_MEMORY.md、CURRENT_STATUS.md、CURRENT_HANDOFF.md、V2_RUNTIME.md、HISTORY_CONTEXT.md、CHAT_INDEX.md。
- 六个文件必须作为同一 `snapshot_id + git_head` 的活动集合；旧副本不调和，重复或混合集合按无效处理。
- Short Prompt 不要求复制 bridge manifest 或正式 evidence 的完整 Hash；这些 Hash
  只在其既有完整性/复现消费者需要时由 Codex 定向核验。
- 修改记忆或当前状态后，只有在工作树 clean、bridge `refresh_ok`、`status=current` 且
  `allow_dirty=false`（即 `publication_state=current_clean`）时，才允许用户人工替换云端 Sources。
- 如果 bridge 目录缺失、不可读、过期、dirty 或含未知文件，保持 fail-closed；本地文档可以交付，但不能声称已同步。

### 11.1 现有维护工具与 compact report

`scripts/maintenance/chatgpt_project_memory.py` 继续负责六文件 Sources 的
`build|verify|status|refresh` 和其 fail-closed manifest，并新增只读 `summary` 与
`verify-evidence <semantic-id>`。`summary` 只输出 branch、HEAD、clean/dirty、active
objective、semantic evidence aliases、authorization tier 和 HARD_BLOCKER；它不从自由文本
推断状态，也不自动选择 latest artifact。`verify-evidence` 只解析
`docs/status/research_governance_state.json` 中的一个已声明 alias，复用现有 manifest
identity，full Hash 仅在 `--verbose` 或 machine manifest output 中出现。本轮不改变 bridge
或 formal consumer。

### 11.2 N3 planning representation

当前 N3 的低秩相关微 Gate 不再逐项展开为
`PLAN_REGISTER_LOW_RANK_*` 链。未来唯一 planning task 为
`DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1`；其内部一次性决定 low-rank family/domain、
admissible rank candidates、selection statistic、complexity handling、deterministic tie-break、
positivity interaction 和 one-shot semantics。实际 rank selection 只有在另行授权并产生新
evidence 时才执行；本规划标签本身不授权 fit、SVD、sampling、forward、CUDA、training、
generation 或 evaluator。
