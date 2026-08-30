# codex-original 项目记忆

更新时间：2026-08-30

## 目的与事实优先级

这是 `codex-original` 的跨 ChatGPT/Codex 交接入口。它只保存稳定导航、当前边界和
交接规则，不替代正式证据、实验报告或历史 transcript。云端 ChatGPT 项目使用它的
受控上传副本；本文件本身仍以本地 Git 版本为准。

事实冲突时按以下顺序处理：

1. `AGENTS.md` 及适用的 scoped `AGENTS.md`：执行合同和禁止事项；
2. Git 状态、源码、配置、测试和正式实验/evidence：可复核的当前事实；
3. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 与 [`status/current_handoff.md`](status/current_handoff.md)：当前状态、门控和下一动作；
4. 本文件：稳定导航和跨产品交接规则；
5. 外部历史索引与按 Session ID 选择的摘要：只用于历史导航，不能覆盖当前事实；
6. 云端 ChatGPT 对话和 Sources：只读规划快照，不能覆盖以上来源。

ChatGPT Sources 包中的文件是由维护工具生成的受控副本。副本必须包含 Git 快照信息；
副本过期、来源不明确、出现敏感信息或与本地验证结果冲突时，拒绝使用该副本，回到
本地权威文件。Sources 是人工确认的桥接快照，不是实时同步机制。

## 当前工作区与运行边界

- 本地 Codex 项目的唯一当前工作区是 Windows 原生 `D:\codex-original`。
- 云端 ChatGPT 项目是 `codex-original · ChatGPT`，只负责讨论、检索、规划和总结；它
  不能直接访问 `D:\codex-original`。
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
- 注销前审计提交 `75d0584c379b145b0f0eb45ece14fda160741f23` 是历史恢复点；当前分支
  HEAD、远端同步状态和桥接治理提交以运行时 Git 与 `CURRENT_STATUS.md` 为准。历史 C:
  Git bundle 为
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

本地项目 `codex-original · Windows` 的主目录为 `D:\codex-original`，辅助目录为
`C:\Users\24567\Documents\Codex\codex-original-history-context`。云端项目
`codex-original · ChatGPT` 与本地项目通过六文件 Sources 快照和标准交接块连接；项目
名称相似本身不提供技术绑定。

ChatGPT 模式负责讨论、规划和总结；Codex 模式负责读取真实工作树、核对 Git、修改
代码/文档和运行验证。只有 Codex 在验证后才能写入权威项目记忆；云端 Sources 仅是
某个 Git HEAD 的只读快照。

Sources 工具的活动输出目录为
`C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current`，
上传由用户人工完成，不修改 Codex SQLite 或云端项目状态。

GPT App 中历史会话的项目显示、长会话渲染、搜索和继续会话属于人工验收；本地 Codex
不能替代该桌面检查，也不能把未知状态写成已通过。

标准方向是：

```text
云端 ChatGPT 项目：讨论/规划/总结
        ↓ 六文件 Sources 快照 + 交接块
本地 Codex 项目：AGENTS.md + Git + docs/PROJECT_MEMORY.md
        ↓ Codex 读取、验证、修改
D:\codex-original
        ↓ refresh + verify
更新下一份云端 Sources 快照
```

## 标准交接模板

ChatGPT → Codex：

```text
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
4. 记忆或当前状态变化后，先在干净工作树运行 Sources `refresh`，再运行 `status`；
   只有返回 `current` 的快照才允许人工替换云端项目中的六个 Sources。
5. Sources 的内部 manifest 不上传；需要历史细节时只按 Session ID 生成临时摘要，任务
   结束后从云端项目移除。
6. 旧 WSL 路径、旧项目副本和历史摘要必须显式标为历史，不能伪装成当前状态。
