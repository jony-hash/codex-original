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

ChatGPT Sources 包（如果用户明确需要离线上传）中的文件只是受控副本。副本过期、来源
不明确、出现敏感信息或与本地验证结果冲突时，拒绝使用该副本，回到本地权威文件；
Sources 不是当前项目的同步机制。

## 当前工作区与运行边界

- 唯一当前工作区是 Windows 原生 `D:\codex-original`。
- 开发、测试和 CUDA 运行唯一使用 `D:\ANACONDA\envs\optical-generative-models-win`；
  旧 `D:\ANACONDA\envs\optical-generative-models` 仅作 Legacy 恢复参考。Ubuntu 已注销，
  当前环境与恢复边界见 `AGENTS.md` 与 `CURRENT_STATUS.md`。
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
- 本次工作流治理允许的实现动作是维护记忆入口、项目文档和环境入口；不修改训练代码、
  `outputs`、WSL、SQLite 或历史 transcript。只有用户明确需要离线上传时，才生成并验证
  受控 Sources 包。

## Git 与验证状态

- 当前分支、HEAD、dirty paths 和最近验证结果以 `CURRENT_STATUS.md` 和运行时 Git
  状态为准；这里不复制容易过期的完整提交清单。
- 注销前审计提交为 `75d0584c379b145b0f0eb45ece14fda160741f23`，已推送到
  `origin/research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`；远端 SHA 与本地
  HEAD 一致。最新 C: Git bundle 为
  `C:\Users\24567\Documents\Codex\codex-original-final-20260830.bundle`，已通过完整历史
  校验（55 refs，SHA-256 `36F1C6E6FBEE03AB7DCB17B076895606F6166D193249248CB6FB9364512D57FE`）；
  不能把本地 bundle 伪装成远端备份。
- 注销前和注销后默认全量 pytest 均在 Windows Conda/CUDA 环境通过：`1,028 passed, 12
  deselected, 25 subtests passed`。near-conjugate 单项在专用短 TEMP `C:\t` 下通过；这不
  改变模型、训练或科研结论。
- 任何 ChatGPT 结论都必须标明“已验证”“未验证”或“需要人工验收”，不能把历史
  全绿基线写成当前全绿。

## Codex 历史迁移状态

- 217 条唯一迁入会话、147 条可见主聊天和 70 条归档线程已完成迁移；214 条项目候选
  已完成 folder-bound `cwd` 修复，SQLite `integrity_check=ok`。
- `project_id` 为空是当前桌面端按 `cwd` 过滤的正常语义；不能为了“绑定”而手写
  `projects`、`project_roots`、`threads.project_id` 或重建 `session_index.jsonl`。
- 旧 WSL 路径只表示历史记录。`session_meta`/`turn_context` 的修复已将项目路径统一
  到 Windows 目标；`event_msg` 的历史命令位置保留原样，不能把它解释为当前 cwd。
- 外部历史上下文目录是导航层，不是原始 transcript 的替代品；其入口是
  `HISTORY_CONTEXT.md`，刻意不与本仓库的 `docs/PROJECT_MEMORY.md` 同名。历史只能按
  Session ID 选择摘要，默认不把全部摘要上传到 ChatGPT Sources。
- Ubuntu 注销前增量审计仅发现 5 个 WSL Codex 日志/缓存文件；它们已在 C: 独立 tar
  中保留并通过目录读取校验。2026-08-30 已执行精确的 `wsl.exe --unregister Ubuntu`；
  Ubuntu 注册项、`D:\WSL\Ubuntu` 目录和 `ext4.vhdx` 均已消失，`docker-desktop` 保留。

## ChatGPT 与 Codex 的职责边界

唯一主要本地项目是 `codex-original · Windows`：主目录为 `D:\codex-original`，辅助
目录为 `C:\Users\24567\Documents\Codex\codex-original-history-context`。ChatGPT
模式负责讨论、规划和总结；Codex 模式负责读取真实工作树、核对 Git、修改代码/文档和
运行验证。两种模式可以在同一个本地项目中连续工作，但不会把任意聊天自动变成权威事实。

Sources 工具只提供可选的离线导出边界，由用户人工决定是否上传；它不替代本地项目，也
不声称能直接同步聊天。

GPT App 中历史会话的项目显示、长会话渲染、搜索和继续会话属于人工验收；本地 Codex
不能替代该桌面检查，也不能把未知状态写成已通过。

标准方向是：

```text
ChatGPT 讨论/规划/总结
        ↓ 同一 GPT App 本地项目
AGENTS.md + docs/PROJECT_MEMORY.md + Git
        ↓ Codex 本地读取、验证、修改
D:\codex-original

（可选：用户明确需要时，离线生成受控 Sources 包）
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
4. 若用户明确维护离线 Sources 包，记忆更新后才运行 Sources `build` 和 `verify`；过期包
   必须在使用前人工替换。没有该需求时，不生成或上传 Sources 包。
5. 旧 WSL 路径、旧项目副本和历史摘要必须显式标为历史，不能伪装成当前状态。
