# 当前项目状态

更新时间：2026-08-30

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- 基线提交：`019b76ef911e0bd16b8d5d4169807323427684af`
- Windows 原生迁移提交：`6f9015b1ebc8a461f48801c5280ad13a168c14fd`（本地保留，回滚期后再推送）
- Codex 历史迁移工具检查点：`c383f21`（本地保留，回滚期后再推送）
- 本地项目绑定工具提交：`6c395da`；绑定前 WSL `cwd` 漂移保护提交：`8efd88d`；folder-bound 修复 v2 当前基于 `d47da23` 的未提交工作树（均本地保留，回滚期后再推送）
- `origin`：`https://github.com/jony-hash/codex-original.git`
- Windows 环境：`D:\ANACONDA\envs\optical-generative-models-win`
- 历史输出归档：`D:\LDM-Archive\codex-original\outputs`

Windows 原生项目是当前唯一开发和训练入口。PowerShell 7、Windows Conda、CUDA 和 ChatGPT 桌面版 Codex 均指向上述项目；VS Code 只用于可选的代码查看和编辑。

## 迁移状态

- Git bundle、WSL Codex 会话归档、WSL 用户配置归档和 Conda 清单已保存到 `D:\LDM-Archive\codex-original\backup`。
- `outputs` 的 163 个一级目录均已独立归档并验证；manifest 记录 33,117 个文件、240,556,414,616 字节源数据和每个 tar 的 SHA-256。
- 归档保留 Linux 原始相对路径，包括尾随句点目录和长路径；历史输出不展开到工作树。
- WSL 的 `data` 已复制到 Windows 项目，10 个文件、121,987,494 字节，逐文件校验一致。
- 新环境已验证 Python 3.11.16、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1 和 RTX 3060 Laptop GPU。
- Windows 原生 CUDA smoke 已完成训练和验证，结果位于 `D:\codex-original\outputs\migration_smoke\20260830_052257_572717_train_windows_native`。
- 初始完整 pytest：1,010 passed、9 failed、3 deselected、25 subtests passed；9 项均依赖按归档策略未展开的历史 WSL 输出，不属于当前 Windows 运行环境失败。
- 默认 pytest 现排除 `historical_artifact` 与 `scheduled_diagnostic`；历史通道保留 9 项测试，仅在明确临时恢复命名归档后运行。
- 注销前复验：将 `TEMP`/`TMP` 指向专用短路径 `C:\t`，并将 Windows Conda 的 CUDA DLL 目录加入 PATH 后，默认 pytest 为 `1,028 passed, 12 deselected, 25 subtests passed`（0 失败）；`historical_artifact` collect-only 精确收集 9 项。near-conjugate 单项在较长临时根下会因 Windows 260 字符路径边界报 `FileNotFoundError`，短 TEMP 复验通过；这不是模型、CUDA 或测试逻辑失败。
- Codex 历史会话已于 2026-08-30 完成 canary 与全量合并：旧归档 218 个文件中的 217 个唯一会话均已迁入 Windows Codex Home；147 条主聊天保持可见（144 条 `codex-original` 聊天的工作目录为 `D:\codex-original`，3 条杂项聊天保留原目录），70 条内部线程默认归档。
- 全量会话校验通过：SQLite `integrity_check=ok`、217/217 JSONL 文件与清单哈希可读、原有 Windows 会话未被覆盖；10 条旧父子关系已追加（迁移前 6 条，迁移后 16 条）。迁移前快照和不可变 WSL tar 均保留。
- 当前状态库最新只读核验共有 353 条线程，而迁移前基线加 217 条迁入线程为 331 条；多出的 22 条是迁移后新建任务，不构成重复。顶层 verify 以基线/迁入 ID 集合和文件哈希核验，不覆盖这些后续任务。
- GPT App 已创建本地项目 `codex-original · Windows`，主目录为 `D:\codex-original`，辅助目录为 `C:\Users\24567\Documents\Codex\codex-original-history-context`；当前 App 映射按 `cwd` 工作，数据库 `projects`/`project_roots` 为空、214 条项目线程的 `project_id` 保持为空是正常状态。folder-bound 修复后，214 条候选的数据库 `cwd` 与 transcript 中 `session_meta`/`turn_context` 的项目路径均已统一为 Windows 目标；历史 `event_msg` 的 `cwd` 仍保持原样作为负控。
- folder-bound `project-bind` v2 已在真实 Codex Home 完成：canary 修复 3 条、complete 修复剩余 211 条，累计 214/214；独立 v2 manifest 与 C: 快照保留。固定 canary 为 `01a04b61-4eca-7260-8437-161edde533cb`、`019dd835-d845-7031-b572-2999535d377f`、`019f927b-dc1c-7d42-b37a-a8af0b597320`。工具没有写入项目表或 `threads.project_id`。
- live `project-bind --verify` 已通过：phase=`complete`、SQLite `integrity=ok`、候选 214、可见 144、归档 70、杂项 3、父子边 16、217 个文件全部核验；最新状态库线程数为 353。`session_index.jsonl` 仍为 34 行、命中候选 0 条，仅作审计观察，不影响成功判定。随后修复了顶层 `verify` 未注册 v2 binding 参数的 CLI 缺陷；修复后的真实顶层 verify 也已通过：217/217 文件、147 可见、70 归档、SQLite `ok`。
- GPT App 重开后将一个既有线程的 `rollout_path` 从普通 Windows 路径规范化为等价的 `\\?\` extended path，首次触发了验证器的 fail-closed 差异报告。全量只读字段比对确认只有这一处等价差异；验证器现仅对 `rollout_path` 使用已有 `canonical_path()` 比较，其他线程字段仍严格比较，修复后的真实 `project-bind --verify` 已再次通过。
- 本轮迁移 targeted pytest 为 11 passed；本轮 ChatGPT Sources targeted pytest 为 7 passed；compileall（现有 `src`、`tests`、`train.py`、迁移脚本和 Sources 工具）、ruff 与 `git diff --check` 全部通过。默认全量 pytest 复验为 `1,028 passed, 12 deselected, 25 subtests passed`；所有测试在 Windows Conda/CUDA 环境完成，未启动 WSL。
- 当前四个未提交的迁移文件为 `docs/CURRENT_STATUS.md`、`docs/codex_history_migration.md`、`scripts/maintenance/codex_history_migration.py` 和 `tests/test_codex_history_migration.py`；本轮桥接文件另行记录，不与迁移数据库或 transcript 混合。
- 脱敏跨聊天知识层位于 `C:\Users\24567\Documents\Codex\codex-original-history-context`，原有 `PROJECT_MEMORY.md`、`CHAT_INDEX.md`、`CHAT_CATALOG.jsonl` 和 147 份摘要保持不变；本轮四个 `.pytest-tmp*` 目录及 761 个测试文件已移动到同级可恢复 quarantine `C:\Users\24567\Documents\Codex\codex-original-history-context-quarantine-20260830`，不作为 Sources 来源。
- 双项目桥接已落地：权威入口为 [`docs/PROJECT_MEMORY.md`](PROJECT_MEMORY.md)，操作手册为 [`chatgpt_codex_workflow.md`](chatgpt_codex_workflow.md)，本地 Sources 工具为 `scripts/maintenance/chatgpt_project_memory.py`。注销前 Sources 包已在 `C:\Users\24567\Documents\Codex\codex-original-history-context\chatgpt-sources-final-20260830` 重建并验证通过，精确包含 `PROJECT_MEMORY.md`、`CURRENT_STATUS.md`、`CURRENT_HANDOFF.md`、`V2_RUNTIME.md`、`HISTORY_CONTEXT.md` 和 `CHAT_INDEX.md` 六个文件；内部 manifest 不上传。
- 活动项目列表中已没有旧的 `chatgpt-codex工作流`；`.codex` 中残留的旧缓存/副本仍保留为回滚窗口内的待处理数据，不读作当前事实源，也未被新 Sources 包扫描或复制。
- 归档自检：163/163 个 tar 存在；抽查 4 个 tar 的 SHA-256 与目录读取通过；5 项核心备份大小与 SHA-256 均匹配。
- 最近容量检查：D: 可用约 14.59 GiB；回滚窗口内仅运行测试和小型 smoke。

## Windows 兼容性改动

迁移所需改动仅处理 Windows 运行环境、路径表示和历史证据兼容性，不改变模型结构、公共 API 或训练算法：

- 为 Windows 提供进程峰值内存统计兼容实现。
- 将实验产物中的相对路径统一序列化为 POSIX 形式。
- 将 Windows 下的临时二进制文件以可 `fsync` 的读写方式打开。
- 允许历史 WSL 数据集 manifest 在数据根目录迁移后保持原有证据身份，同时继续严格校验张量和 split 哈希。

## 回滚状态

WSL Ubuntu 与 docker-desktop 当前均为 Stopped，未发现 VS Code WSL 后台连接或训练进程。`vmmemWSL` 仅保留低于 1 GiB 的系统服务占用，不强制终止。迁移日为 2026-08-30，回滚窗口截至 2026-09-06；窗口内不在 WSL 项目上开发、训练或生成新输出，且 D: 不得低于 10 GiB。迁移提交保持本地，不在窗口内推送。Ubuntu 注销未执行，必须在窗口结束后再次取得明确批准。

## 下一步

以 Windows 项目作为唯一最新工作树；每次任务更新本文件中的提交、测试、实验路径和下一步。自动迁移与 folder-bound 修复的 canary、complete、`project-bind --verify` 和顶层 `verify` 均已通过，Windows 默认回归也已全绿。注销前尚需完成 Ubuntu 备份时间后的增量审计，并由用户在 GPT App 中人工确认固定 canary、最早/最近/最长及含子线程会话；本地 CLI 不能替代该桌面验收。全部门槛通过后才执行精确的 `wsl --unregister Ubuntu`，不手动删除 VHD，也不注销 `docker-desktop`。
