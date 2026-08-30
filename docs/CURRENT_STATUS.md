# 当前项目状态

更新时间：2026-08-30

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- 基线提交：`019b76ef911e0bd16b8d5d4169807323427684af`
- 注销前审计提交：`75d0584c379b145b0f0eb45ece14fda160741f23`（已按用户确认推送）
- Windows 原生迁移、Codex 历史迁移和 folder-bound 修复提交均已包含在当前分支历史中；本次最终注销记录将在其后追加。
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
- Windows 原生 CUDA smoke 已完成训练和验证；注销后复验结果位于 `D:\codex-original\outputs\migration_smoke\20260830_191323_817115_train_windows_native_post_unregister`。
- 初始完整 pytest：1,010 passed、9 failed、3 deselected、25 subtests passed；9 项均依赖按归档策略未展开的历史 WSL 输出，不属于当前 Windows 运行环境失败。
- 默认 pytest 现排除 `historical_artifact` 与 `scheduled_diagnostic`；历史通道保留 9 项测试，仅在明确临时恢复命名归档后运行。
- 注销前复验：将 `TEMP`/`TMP` 指向专用短路径 `C:\t`，并将 Windows Conda 的 CUDA DLL 目录加入 PATH 后，默认 pytest 为 `1,028 passed, 12 deselected, 25 subtests passed`（0 失败）；`historical_artifact` collect-only 精确收集 9 项。near-conjugate 单项在较长临时根下会因 Windows 260 字符路径边界报 `FileNotFoundError`，短 TEMP 复验通过；这不是模型、CUDA 或测试逻辑失败。
- Codex 历史会话已于 2026-08-30 完成 canary 与全量合并：旧归档 218 个文件中的 217 个唯一会话均已迁入 Windows Codex Home；147 条主聊天保持可见（144 条 `codex-original` 聊天的工作目录为 `D:\codex-original`，3 条杂项聊天保留原目录），70 条内部线程默认归档。
- 全量会话校验通过：SQLite `integrity_check=ok`、217/217 JSONL 文件与清单哈希可读、原有 Windows 会话未被覆盖；10 条旧父子关系已追加（迁移前 6 条，迁移后 16 条）。迁移前快照和不可变 WSL tar 均保留。
- 当前状态库最新只读核验共有 362 条线程，而迁移前基线加 217 条迁入线程为 331 条；多出的 31 条是迁移后新建任务，不构成重复。顶层 verify 以基线/迁入 ID 集合和文件哈希核验，不覆盖这些后续任务。
- GPT App 已创建本地项目 `codex-original · Windows`，主目录为 `D:\codex-original`，辅助目录为 `C:\Users\24567\Documents\Codex\codex-original-history-context`；当前 App 映射按 `cwd` 工作，数据库 `projects`/`project_roots` 为空、214 条项目线程的 `project_id` 保持为空是正常状态。folder-bound 修复后，214 条候选的数据库 `cwd` 与 transcript 中 `session_meta`/`turn_context` 的项目路径均已统一为 Windows 目标；历史 `event_msg` 的 `cwd` 仍保持原样作为负控。
- folder-bound `project-bind` v2 已在真实 Codex Home 完成：canary 修复 3 条、complete 修复剩余 211 条，累计 214/214；独立 v2 manifest 与 C: 快照保留。固定 canary 为 `01a04b61-4eca-7260-8437-161edde533cb`、`019dd835-d845-7031-b572-2999535d377f`、`019f927b-dc1c-7d42-b37a-a8af0b597320`。工具没有写入项目表或 `threads.project_id`。
- live `project-bind --verify` 已通过：phase=`complete`、SQLite `integrity=ok`、候选 214、可见 144、归档 70、杂项 3、父子边 16、217 个文件全部核验；最新状态库线程数为 362。`session_index.jsonl` 仍为 34 行、命中候选 0 条，仅作审计观察，不影响成功判定。随后修复了顶层 `verify` 未注册 v2 binding 参数的 CLI 缺陷；修复后的真实顶层 verify 也已通过：217/217 文件、147 可见、70 归档、SQLite `ok`。
- GPT App 重开后将一个既有线程的 `rollout_path` 从普通 Windows 路径规范化为等价的 `\\?\` extended path，首次触发了验证器的 fail-closed 差异报告。全量只读字段比对确认只有这一处等价差异；验证器现仅对 `rollout_path` 使用已有 `canonical_path()` 比较，其他线程字段仍严格比较，修复后的真实 `project-bind --verify` 已再次通过。
- 本轮迁移 targeted pytest 为 11 passed；本轮 ChatGPT Sources targeted pytest 为 7 passed；compileall（现有 `src`、`tests`、`train.py`、迁移脚本和 Sources 工具）、ruff 与 `git diff --check` 全部通过。默认全量 pytest 复验为 `1,028 passed, 12 deselected, 25 subtests passed`；所有测试在 Windows Conda/CUDA 环境完成，未启动 WSL。
- 注销后收尾只更新迁移状态、项目记忆、运行手册和根执行合同；本轮再增加环境入口、历史文件命名、双项目桥接和测试契约治理；没有修改模型、训练代码、公共 API、outputs 正式产物、SQLite 或 transcript。
- 脱敏跨聊天知识层位于 `C:\Users\24567\Documents\Codex\codex-original-history-context`，入口为 `HISTORY_CONTEXT.md`，另有 `CHAT_INDEX.md`、`CHAT_CATALOG.jsonl` 和 147 份摘要；它不与仓库权威 `docs\PROJECT_MEMORY.md` 同名。原有 `chatgpt-sources` 与 `chatgpt-sources-final-20260830` 已移到 `C:\Users\24567\Documents\Codex\codex-original-closure-quarantine-20260830`，不作为当前项目来源。
- 本地项目 `codex-original · Windows` 仍是源码和执行入口；云端项目 `codex-original · ChatGPT` 负责讨论和规划。两者通过受控六文件 Sources 快照与标准交接块连接。操作手册为 [`chatgpt_codex_workflow.md`](chatgpt_codex_workflow.md)，不建立源码副本或自动数据库同步。
- 活动项目列表中已没有旧的 `chatgpt-codex工作流`；`.codex` 中残留的旧缓存/副本仍作为离线历史材料保留，不读作当前事实源，也未被新 Sources 包扫描或复制。
- 归档自检：163/163 个 tar 存在；抽查 4 个 tar 的 SHA-256 与目录读取通过；5 项核心备份大小与 SHA-256 均匹配。
- 注销前 WSL 增量审计：备份时间 `2026-08-29 23:33:29` 后仅发现 5 个用户范围文件（WSL Codex 日志 SQLite/WAL/SHM、`models_cache.json`、空的 `.motd_shown`），未发现 `/home` 其他用户文件、项目文件、`/root`、`/usr/local` 的新增内容；5 个文件已保存为 `C:\Users\24567\Documents\Codex\wsl-post-backup-increment-20260830.tar`，397,127,680 bytes，SHA-256 为 `072212068E3F6D02FFD3B762B8670CB2B5E6A83F967137C6459874D99A07494B`，归档目录读取通过。
- 注销前最新 Git bundle：`C:\Users\24567\Documents\Codex\codex-original-final-20260830.bundle`，50,768,910 bytes，SHA-256 为 `36F1C6E6FBEE03AB7DCB17B076895606F6166D193249248CB6FB9364512D57FE`；`git bundle verify` 报告包含当前 `75d0584c379b` 和完整历史、55 个 refs。
- 注销前的 WSL 增量归档仍位于 `C:\Users\24567\Documents\Codex\wsl-post-backup-increment-20260830.tar`，5 个路径、397,127,680 bytes，SHA-256 为 `072212068E3F6D02FFD3B762B8670CB2B5E6A83F967137C6459874D99A07494B`。
- Ubuntu 已于 2026-08-30 按精确注册名注销；`D:\WSL\Ubuntu\ext4.vhdx` 已消失，D: 可用约 `263.72 GiB`。

## Windows 原生工作流与双项目桥接治理（2026-08-30）

- 当前唯一支持运行环境为 `D:\ANACONDA\envs\optical-generative-models-win`；旧
  `D:\ANACONDA\envs\optical-generative-models`（PyTorch 2.4.1、editable install
  指向旧代码副本）仍保留为 Legacy，不参与当前测试或训练。VS Code 用户设置中的无效
  `D:\ANACONDA\python.exe` 已移除，用户已确认状态栏选择新解释器。
- 当前历史入口为 `HISTORY_CONTEXT.md`；活动辅助目录包含 147 份摘要、`CHAT_INDEX.md`
  和 `CHAT_CATALOG.jsonl`，不存在外部同名 `PROJECT_MEMORY.md`。
- 本地项目 `codex-original · Windows` 的主目录为 `D:\codex-original`；云端项目
  `codex-original · ChatGPT` 只负责讨论、检索、规划和总结。云端项目不能直接读取本机
  目录，二者通过六文件 Sources 快照和标准交接块连接。
- 活动桥接目录固定为
  `C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current`，
  不属于源码仓库或本地项目辅助目录。旧 Sources 派生目录继续保存在
  `C:\Users\24567\Documents\Codex\codex-original-closure-quarantine-20260830`。
- Sources 工具 manifest 已升级为 v2，支持 `refresh` 和只读 `status`；默认要求干净
  工作树，生成副本的 `PROJECT_MEMORY.md` 带 snapshot ID、Git HEAD、生成时间和工作树
  状态。只有 `refresh_ok` 且 `status=current` 才允许用户人工替换云端 Sources。
- 原项目 `.pytest_cache` 已先移至
  `D:\codex-original-acl-quarantine-20260830\.pytest_cache`，再由测试重新生成。短临时
  根 `D:\ct` 只含测试副产物，不是项目数据；可由普通 Windows PowerShell 在应用关闭后
  按精确路径单独清理。
- 采用短临时根的定向 near-conjugate 测试为 5/5，迁移定向测试为 11/11，Sources 桥接
  定向测试为 10/10；compileall、Ruff、v2 CLI、迁移 verify、folder-bound verify 和
  完整默认 pytest 均通过，完整结果为 `1,028 passed, 12 deselected, 25 subtests passed`。
- 环境探针确认 `sys.executable` 为上述新环境、`ldm_project` 来自
  `D:\codex-original\src\ldm_project`、PyTorch 2.5.1/CUDA 12.1、RTX 3060 和 CUDA
  张量计算可用。Sources 工具读取外部 `HISTORY_CONTEXT.md`，不复制原始 transcript、
  全部摘要或 `.codex` 数据。

## Windows 兼容性改动

迁移所需改动仅处理 Windows 运行环境、路径表示和历史证据兼容性，不改变模型结构、公共 API 或训练算法：

- 为 Windows 提供进程峰值内存统计兼容实现。
- 将实验产物中的相对路径统一序列化为 POSIX 形式。
- 将 Windows 下的临时二进制文件以可 `fsync` 的读写方式打开。
- 允许历史 WSL 数据集 manifest 在数据根目录迁移后保持原有证据身份，同时继续严格校验张量和 split 哈希。

## 注销后状态

2026-08-30 已完成 GPT App 固定 canary、最早/最近/最长会话及含子线程会话的人工验收。随后在最后精确检查通过后执行 `wsl.exe --unregister Ubuntu`，命令退出码为 0。注册表和 `wsl --list --verbose` 中已不再有 Ubuntu，`D:\WSL\Ubuntu\ext4.vhdx` 及其目录已消失；`docker-desktop` 保留且停止。Windows 项目、Codex Home、历史 transcript、迁移快照、Git bundle 和 outputs tar 均未删除；旧 Sources 派生目录现保存在 C 盘可恢复隔离区。

注销后复验未发现 Python、训练或 `wslhost` 进程；Windows Conda/PyTorch/CUDA/RTX 3060、完整测试、迁移工具和 folder-bound 工具均正常；Sources 桥接工具已支持 `refresh`/`status`，并由定向测试覆盖，云端上传仍需用户人工确认。

## 下一步

以本地 Windows 项目作为唯一开发、测试、训练和 Git 入口，以云端 ChatGPT 项目作为规划和
检索入口。两者通过经过验证的六文件 Sources 快照和标准交接块衔接；云端快照过期时必须
刷新，不能覆盖本地事实。Ubuntu 已注销，不再存在 WSL 回滚工作区；未来若需恢复 Linux
环境，只能重装 WSL 并使用已保存的 Git bundle、用户配置归档、Codex transcript 归档、
增量 tar 和 outputs tar。保留这些恢复材料，不自动删除归档或 `docker-desktop`；继续维护
当前 commit、测试、实验路径和历史上下文状态。
