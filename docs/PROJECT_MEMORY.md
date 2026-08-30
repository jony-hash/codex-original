# codex-original 项目记忆

更新时间：2026-08-30

## 目的与事实优先级

这是 `codex-original` 的跨 ChatGPT/Codex 交接入口。它只保存稳定导航、当前边界和
交接规则，不替代正式证据、实验报告或历史 transcript。

事实冲突时按以下顺序处理：

1. `AGENTS.md` 及适用的 scoped `AGENTS.md`：执行合同和禁止事项；
2. [`CURRENT_STATUS.md`](CURRENT_STATUS.md)：当前 Windows 工作区、迁移和验证状态；
3. [`status/current_handoff.md`](status/current_handoff.md)：科研状态、门控和唯一下一动作；
4. Git 状态与正式实验/evidence：可复核的实现和测量事实；
5. 外部历史索引与按 Session ID 选择的摘要：只用于历史导航，不能覆盖当前事实。

ChatGPT Sources 包中的文件是受控副本。副本过期、来源不明确、出现敏感信息或与本地
验证结果冲突时，拒绝使用该副本，回到本地权威文件。

## 当前工作区与运行边界

- 唯一当前工作区是 Windows 原生 `D:\codex-original`。
- 开发、测试和 CUDA 运行使用 Windows Conda 环境；具体环境和回滚约束见
  `AGENTS.md` 与 `CURRENT_STATUS.md`。
- v1 仍由 `train.py` 维护历史可复现性；v2 是当前 teacher-free conditional optical-generation
  runtime，入口为 `python -m ldm_project.v2`。
- 正式 v2 光路边界是：`label + 8D code` → identity-anchored electronic amplitude
  generator → amplitude SLM → coherent 4f relay → 2+2 phase-only finite-crop D2NN
  → complex identity OLS → decoder metasurface → CMOS single exposure。
- formal D2NN boundary 是 `finite_crop`；`transparent_identity` 仅为 diagnostic-only。
- 不把旧 WSL checkout、`.codex` 旧项目副本、历史输出归档或原始 transcript 当作当前工作树。

## 当前科研状态、禁止事项与唯一下一动作

- 当前 formal identity 链的有效结果是 `closed_historical_valid_failure`；它没有
  `formal_promotion` 权限，也不能授权 manual review、candidate export、memorization、
  utility、dataset admission 或 accepted manifest。
- prior-native、direct-generator、quadrature、fiber、FMF 长训练和 formal promotion
  等已关闭或未授权路线不能因为 ChatGPT 的讨论而重新打开。
- 所有科研执行仍需遵守 `AGENTS.md` 的 long-training gate；记忆文件本身不提供训练授权。
- 当前交接文档记录的唯一允许科研动作是
  `manual_review_fmf_mode_capacity_reconstruction_contract`。如果交接文档更新，
  以最新的 `CURRENT_HANDOFF.md` 和 hypothesis ledger 为准。
- 本次桥接链路的唯一允许实现动作是维护记忆入口、生成受控 Sources 包并验证其没有
  过期、秘密或未知文件；不修改训练代码、`outputs`、WSL、SQLite 或历史 transcript。

## Git 与验证状态

- 当前分支、HEAD、dirty paths 和最近验证结果以 `CURRENT_STATUS.md` 和运行时 Git
  状态为准；这里不复制容易过期的完整提交清单。
- 注销前本地收尾提交为 `0cef23dc2b1e272601f91d3bbba9d13f93215635`，工作树已验证干净；
  对应的 C: Git bundle 已通过完整历史校验。GitHub 推送仍需用户明确确认，不能把本地
  bundle 伪装成远端备份。
- 注销前默认全量 pytest 已在 Windows Conda/CUDA 环境通过：`1,028 passed, 12
  deselected, 25 subtests passed`。near-conjugate 单项只有在较长临时根下会触发 Windows
  260 字符路径边界；使用专用短 TEMP `C:\t` 后单项和全量均通过。这不改变模型、训练或
  科研结论。
- 任何 ChatGPT 结论都必须标明“已验证”“未验证”或“需要人工验收”，不能把历史
  全绿基线写成当前全绿。

## Codex 历史迁移状态

- 217 条唯一迁入会话、147 条可见主聊天和 70 条归档线程已完成迁移；214 条项目候选
  已完成 folder-bound `cwd` 修复，SQLite `integrity_check=ok`。
- `project_id` 为空是当前桌面端按 `cwd` 过滤的正常语义；不能为了“绑定”而手写
  `projects`、`project_roots`、`threads.project_id` 或重建 `session_index.jsonl`。
- 旧 WSL 路径只表示历史记录。`session_meta`/`turn_context` 的修复已将项目路径统一
  到 Windows 目标；`event_msg` 的历史命令位置保留原样，不能把它解释为当前 cwd。
- 外部历史上下文目录是导航层，不是原始 transcript 的替代品。历史只能按 Session ID
  选择摘要，默认不把全部摘要上传到 ChatGPT Sources。
- Ubuntu 注销前增量审计仅发现 5 个 WSL Codex 日志/缓存文件；它们已在 C: 独立 tar
  中保留并通过目录读取校验。Ubuntu、docker-desktop 均已停止，但 Ubuntu 尚未注销。

## ChatGPT 与 Codex 的职责边界

ChatGPT 项目 `codex-original · ChatGPT` 用于讨论、规划、总结和提出记忆更新建议；
它只读取用户人工上传的受控 Sources，不声称能直接读取或修改本机文件。

Codex 本地项目 `codex-original · Windows` 以 `D:\codex-original` 为主目录，负责读取
真实工作树、核对 Git 和验证结果、修改代码/文档、运行测试，并在事实核对后更新本文件。
Sources 包由本地工具生成，上传/替换由用户在 ChatGPT 项目中人工确认。

GPT App 中历史会话的项目显示、长会话渲染、搜索和继续会话属于人工验收；本地 Codex
不能替代该桌面检查，也不能把未知状态写成已通过。

标准方向是：

```text
ChatGPT 讨论/规划/总结
        ↓ 受控 Sources 包（人工上传）
docs/PROJECT_MEMORY.md + 当前文档
        ↓ Codex 本地读取、验证、修改
D:\codex-original
```

## 标准交接模板

ChatGPT → Codex：

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

Codex → ChatGPT：

```text
结果：
修改文件：
验证结果：
未验证内容：
已知风险：
项目记忆更新：
```

## 记忆更新规则

1. ChatGPT 只提出更新建议，不直接把建议当作事实。
2. Codex 修改前先读取本文件、`AGENTS.md`、`CURRENT_STATUS.md` 和适用的交接文档。
3. 只有被 Git、测试、正式 evidence 或明确人工验收支持的事实才能写入“当前”章节。
4. 记忆更新后重新运行 Sources `build` 和 `verify`；过期包必须在 ChatGPT 项目中人工替换。
5. 旧 WSL 路径、旧项目副本和历史摘要必须显式标为历史，不能伪装成当前状态。
