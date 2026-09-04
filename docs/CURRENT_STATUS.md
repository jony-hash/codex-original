# 当前项目状态

更新时间：2026-09-04

## FINAL_MAINLINE_CLOSEOUT：H-LOMA-CBOGT Scaled-Chen reconstruction-repair mainline（2026-09-03）

- Current phase: `MAINLINE CLOSEOUT`
- Scientific diagnosis: `closed`
- Repair: `confirmed`
- Integration: `qualified`
- F32: `qualified`
- Visualization: `completed`
- Runtime performance: `characterized`
- `mainline_state=closed_at_current_evidence_boundary`
- Current mainline has no next scientific experiment. The future independent track is
  `REAL-HARDWARE DEPLOYMENT`; if explicitly reopened, use the task name
  `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`.

### Frozen authority chain

`RMAL → H-LOMA-CBOGT-S7-MPCDAR → H-LOMA-CBOGT-S8-MPCDAR-R16-F64-IQ →
H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ → H-LOMA-CBOGT-S9V-FULL-PATH-VIS →
H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ`.

Each consumer uses only its named immutable parent. Write-once non-authoritative attempts remain
for audit history and are not consumed; no later attempt is replaced by an earlier attempt.

### Frozen scientific boundary

- Within the existing optical-path/CMOS evidence boundary, task-relevant information survives
  through CMOS.
- The reconstruction failure is primarily supported as mapping/readout misalignment.
- Simple locality and raw Euclidean W-SVD are inadequate for this task; useful mean/DC material
  is present, and centered detector information is better represented by a fit-derived data-aware
  basis.
- Registered MPCDAR rank 16 is the first confirmed candidate in its preregistered powers-of-two
  family. It is
  not a minimum, intrinsic, or true rank-16 claim.

### Frozen readout and architecture

The frozen candidate is `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`,
with `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, and `V16[40000,16]`. The S9 F32 artifact
contains `684352` scalars, `2737408` raw tensor bytes, and `2739448` serialized bytes.
The allowed wording is: `optical front end → CMOS intensity sensing → lightweight frozen
electronic rank-16 readout`; this is hybrid optical-electronic reconstruction, and MPCDAR is
not a native CMOS optical image.

### Governance boundary

`reconstruction_manifold_supported=false`; `generative_manifold_supported=false`;
`generation_readiness=not_ready`; `training_authorized=false`; `formal_promotion=false`;
`closed_task_relevant_bottleneck_not_supported`. The hypothesis/experiment thread is
`closed / resolved to current evidence boundary`; historical hypotheses and write-once records
are retained and are not deleted.

- Not established: native optical target-like CMOS reconstruction, real physical CMOS validation,
  Raspberry Pi benchmarking, sim-to-real detector alignment, end-to-end hardware latency,
  GPU runtime, full-W runtime, quantization, generative manifold, or generation readiness.
- Future tracks remain independent: real CMOS/Raspberry Pi deployment, sim-to-real detector
  alignment, optional optimization only if a separately justified need appears, native optical
  redesign only under new explicit authorization, and generative-manifold/generation evidence.

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- MPCDAR 的 S7 producer commit 为 `b59172750d0c88bbc9f7426ab7a21a1d63639f57`；S8
  parent implementation commit 为 `c0140577fddb56af79d72a6d473cc49ef04f5ffc`；S9
  F32 qualification implementation/fix commits 为 `4b75e4d67efc1dc90a01d1d33a6aa3c1c243c652`
  与 `ebbd0d232c56ee45a69dea830911eff4f697ad47`，当前 clean HEAD 以 Git 核验为准。
- 当前主线已在 H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ 后完成最终收口。S10 对冻结 S9
  rank-16 float32 readout 的 CPU-only runtime/performance characterization 只报告
  fresh-process initialization、readout-only latency/throughput、storage 与 deterministic
  correctness；不把分析 MAC ratio 或 CPU telemetry 解释为 measured speedup、GPU/硬件、
  production readiness、reconstruction sufficiency 或 downstream authorization。S9V、S9、
  S8、S7、MCFA、DMSA、DSFA 与更早 evidence 保持 immutable，并继续作为父级事实。
- 基线提交：`019b76ef911e0bd16b8d5d4169807323427684af`
- 当前几何归因实现绑定提交：`6419cc523a8b5dd41cb3d3c2531193890fa1481a`
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

## Windows 原生工作流与双项目桥接治理（2026-08-31）

- 2026-08-31 已采用 Delta + One Gate + Short Codex Prompt 工作流；完整规范位于
  docs/operations/delta_one_gate_short_prompt.md，PROJECT_MEMORY.md 和桥接手册
  只保留短模板与入口。六文件 Sources 合同保持不变；本次治理不改变当前科研 handoff、
  Scaled-Chen 结论、training_authorized=false 或 formal_promotion=false。
- 默认 Short Prompt 不预填 artifact SHA-256 清单；Codex 自动核验 branch、HEAD、
  worktree 和 Git 增量，只有 evidence consumer 要求精确身份时才核验指定 Hash。
  未要求 Hash 时反馈 hash_not_required；必需 Hash 缺失或输入歧义返回 blocked，
  记录 Hash 不一致返回 invalid_harness；
  不自动选择 latest；bridge manifest 与正式 evidence 中的 Hash 保持不变。
- 本轮工作流收紧为 `MODE=VERIFY|PLAN|IMPLEMENT|REPORT`；`MODE=IMPLEMENT` 表示计划已批准。
  Codex 只执行一次 bounded preflight，预检后只能执行或返回 `BLOCKED`，不得因新增背景
  静默重新规划；该变化不修改科研 Gate、handoff 或训练授权。
- 六个 Sources 文件只能作为同一 `snapshot_id + git_head` 的完整集合使用；旧副本、重复
  同名文件和混合快照不调和，不能按时间戳自动选择“最新”。bridge 的 `status=current`
  只有在 `allow_dirty=false` 时才是 `publication_state=current_clean`；`allow_dirty=true`
  仅为 `current_dirty_planning_only`，不可上传；stale/invalid/missing 或无法确认六文件
  属于同一快照时为 `stale_or_conflicting`，必须先完整替换集合。
- Short Prompt 预算改为通常 300–1,000 tokens；超过 1,500 时必须拆 Gate，或满足完整规范
  第 8 节条件后升级为 Long Prompt。固定 Hash、失败分类和 Sources 发布规则不再重复粘贴。

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

## Scaled-Chen 秩匹配 OLS 几何归因（2026-08-30）

在当前 Windows 原生环境执行了冻结权重、无更新的九组秩匹配 LP01 OLS 审计。固定
10×10/100 端口、身份传输、7.111 µm waist、V=2.2、原 calibration、2+2 相位层和
formal Identity 几何；选择集为 MNIST test 索引 5000–9999 中每类 100 张，确认集为
horizon 原 held-out 5000 张。审计不创建 optimizer、不调用 evaluator、不写 checkpoint，
模型与算子状态前后完全一致，九组模式库均为有限 100 秩，峰值 CUDA 为
`2279883264` bytes，墙钟 `345.25s`（仅遥测）。

权威输出位于
`outputs/20260830_v2/runs/lightgen_optical_manifold_smf_bogt_geometry_attribution_v1_attempt_01/`：
validation、audit、decomposition、panel 的 SHA-256 依次为
`ff904bfab0ca63208e881d51c3e06650295ca1e09a3f3b0a9a1a79c374bae0bd`、
`923d2e233cba0f795ec80fbd536846c356cdc177e2dfe0d77957bf4835565f8a`、
`952d062c8a72b872e299da3a20e870b8c7a414b1fd27dc9d9d57a376ca74e5e4`、
`94f1ac7fc74a3bf7b78c60ccc56a7fa479ab56fa30e06dcbc27ddecdc702fd4b`。

选择出的最佳非基线为 `fill=0.80, facet_magnification=0.80`。选择集 paired MSE
相对基线改善 `4.02%`，确认集 step 0 改善 `4.48%`、step 5000 改善 `3.68%`；
逐样本差值 bootstrap 95% 区间为 `[0.00210138, 0.00215426]`。由于未达到预注册
`5%` 实质改善门槛，且 step 5000 收益没有超过 step 0，结果为
`geometry_not_supported` / evidence `valid_failed`，不支持几何匹配 canary。

唯一下一动作是 `manual_review_scaled_chen_distributed_bottleneck_closeout_contract`；
不得自动进入 FMF、继续训练、exposure retuning、生成或 formal promotion。所有证据继续
保持 `training_authorized=false`、`formal_promotion=false`。

## 下一步

以本地 Windows 项目作为唯一开发、测试、训练和 Git 入口，以云端 ChatGPT 项目作为规划和
检索入口。当前科研入口先停在上述分布式瓶颈 closeout 人工审查；在获得新的明确合同前，
不得继续训练或进入 FMF。两者通过经过验证的六文件 Sources 快照和标准交接块衔接；云端
快照过期时必须刷新，不能覆盖本地事实。Ubuntu 已注销，不再存在 WSL 回滚工作区；未来若
需恢复 Linux 环境，只能重装 WSL 并使用已保存的 Git bundle、用户配置归档、Codex transcript
归档、增量 tar 和 outputs tar。保留这些恢复材料，不自动删除归档或 `docker-desktop`；继续
维护当前 commit、测试、实验路径和历史上下文状态。

## Scaled-Chen Manifold Readiness Audit（2026-08-31）

假设 `H-LOMA-CBOGT-S0-MRA` 已在实现提交
`a041187a6ff6292f2be6602871cfd6d35ca172c` 绑定的干净工作树上完成一次冻结、无更新审计。
权威运行目录为
`outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/`；
四个输出的 SHA-256 为：`validation.json`
`b1713ee2676c4f5ab6e6f45838ed75bb1d2b1843e2ce738b53abe62d8e4b0c35`、`audit.json`
`6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556`、`compact_report.json`
`401fe2159f4342a4740b2c40955cb65fc7bb3600f2416fdc4171e1553eebeee0`、`panel.pt`
`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`。

结果为 `valid_failed`（harness 完整、科学门失败），瓶颈四分类为
`distributed_bottleneck_supported`，`generation_readiness=not_ready`，
`generative_manifold_supported=false`。selection 1000 张、confirmation 5000 张，
冻结与负对照门通过；step-0→step-5000 paired-MSE 相对改善为 `28.41%`/`27.02%`，
bootstrap 95% 差值区间分别为 `[0.02395245, 0.02525906]` 和
`[0.02125572, 0.02182155]`。审计未创建 optimizer、未调用 evaluator、未反传、未写
checkpoint 或正式产物；峰值 CUDA 为 `3,618,634,752` bytes（上限 5 GiB）。

表征定位：SLM→pre-OLS retention 为 `0.9182/0.9216`，故 encoder 不是当前主要容量
瓶颈；pre-OLS→modal 为 `0.8145/0.8092`，显示 OLS 前后的 modal 几何压缩；最大且在
两个 panel 唯一稳定的 discontinuity 是 `decoder_layer_2→CMOS`，retention
`0.5827/0.5871`。`post-OLS→decoder-input` 为 `1.0`，四个 checkpoint 的新鲜
decoder Jacobian 数值秩均为 `100`，所以不能把该段仅凭最终 MSE 宣称为 decoder
capacity-limited；多个独立段同时损失信息，维持 distributed 分类。

`reconstruction_manifold_supported=false`：paired MSE 虽优于 cyclic，但未同时优于
class-template，class/instance geometry 与 retrieval controls 也未通过。
同路线 parent contract 的 truncated-normal prior（`z_j=|c_j|²/s_j`，mean 1、variance
0.25、lower 0、fixed calibrated zero-residual phase）已正确注册且 lineage 有效，但
learned 分布的 Gaussian/numerical KL、complex/power rank gates 均失败，因而 prior 状态为
`generation_prior_not_accessible`；没有生成图像，也没有授权 generation canary。唯一下一
动作是 `manual_review_scaled_chen_distributed_bottleneck_closeout_contract`，不得继续训练、
FMF、prior-only generation 或 formal promotion。历史 SMF attempt-02 的 preflight/training
文件不存在，已作为显式零哈希背景保留；未伪造恢复。云端 Sources snapshot 仍落后，本轮未刷新。

## Scaled-Chen Decoder→CMOS 可观测性归因（2026-08-31）

假设 `H-LOMA-CBOGT-S0-DCM-OBS` 在实现提交
`d5fbe5e6efd55076aca7020e0e1cb0f1c7763b19` 的 clean HEAD 上完成了冻结、无更新的
三平面回放。它把旧 MRA 的 `decoder_layer_2→CMOS` 拆为真实的
`decoder_layer_2 complex field→CMOS incident complex field→CMOS intensity`，
继续消费同一 1000 张 selection 与 5000 张 confirmation，不修改 formal v2、OLS、
checkpoint 或训练目标。

唯一权威运行目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_decoder_cmos_attribution_v1_attempt_03/`。
attempt-01/02 的验证或失败运行产物按 write-once 规则保留，但不参与本结论。attempt-03
的 `validation.json`、`audit.json`、`compact_report.json`、`panel.pt` SHA-256 依次为：
`89ece3debd73fa3b7738b7604cafb1d1e52085562d0c70c15563a8d8c39691a7`、
`16975f7f26db13100af134fad70e510a1978093b88c619bc6c3a3e08c067ed68`、
`349ef59d07ca42c6e5446d7bcbe578110b03af27cbc838d0ebe818be38aaa5da`、
`6120c7cc1dc4832587bfe6d91b118579fbf097849d6c9d9a3668b1eb3952ce28`。

结果为 `valid_failed`（harness 完整、归因科学门形成有效结果），分类为
`relay_propagation_bottleneck_supported`。decoder L2→incident retention 为
selection `0.6571`、confirmation `0.6031`；incident→intensity 为
`0.8868/0.9735`，相对损失差 `0.3000`，因此只有 back-relay 段获得单点支持。
最大唯一 discontinuity 也是 decoder L2→incident；CMOS 强度严格由 `|E|²` 产生，
全局相位和逐像素相位不变性控制通过（最大误差 `<7.2×10⁻⁷`）。模型/OLS 状态
前后 hash 一致，`optimizer_created=false`、`evaluator_calls=0`，峰值 CUDA
`2589355520` bytes（上限 5 GiB）。

该结果不推翻已有 `distributed_bottleneck_supported`，也不把 decoder Jacobian rank=100
改写为 decoder capacity 结论；它只把旧断点中的 relay 可观测性单独定位出来。
`generation_readiness` 继续为 `not_ready`，`generative_manifold_supported=false`，
不授权生成、FMF、训练或 promotion。唯一下一动作是
`manual_review_scaled_chen_bounded_back_relay_geometry_diagnostic_contract`。

## WSL 历史 outputs 可见性恢复（2026-08-31）

按已批准的恢复边界，历史 outputs 已从只读归档安全恢复到当前 Windows 工作区
`D:\codex-original\outputs`。原始归档仍保存在
`D:\LDM-Archive\codex-original\outputs`，未被修改；当前树同时保留 Windows 迁移后的
`20260830_v2`、`20260831_v2`、`migration_smoke` 和现有正式 checkpoint。

恢复验收使用 `restore-plan.json` v4：163/163 个归档完成，恢复源文件条目
`33117/33117`，源字节数 `240556414616/240556414616`，路径映射 46 项。32 项 Linux
尾随点目录映射为 `.__linux_trailing_dot__`，14 项超长路径按确定性哈希缩短；完整映射位于
`outputs\_wsl_restore_metadata_v4\path-map.jsonl`。四个 canary 归档先通过，28 个与当前
Windows 输出重叠的文件均逐字节一致后保留现有文件，未覆盖当前结果或 checkpoint。

恢复工具定向测试为 `9 passed`；compileall、Ruff、v2 CLI 和正确 Windows CUDA/PATH 条件下
的默认回归均通过：`1055 passed, 12 deselected, 25 subtests passed`。历史通道在可写临时
目录下为 `5 passed, 4 failed`；失败项只反映历史 JSON 内嵌的 WSL checkpoint 绝对路径与当前
Windows 路径绑定不一致，未改写历史证据，也不影响默认回归。D 盘恢复完成后可用约
`40.57 GB`，仍高于 10 GiB 安全下限。

恢复状态、计划和路径映射是本次操作的审计材料；旧失败尝试的三个元数据目录和 staging
已移动到 `D:\LDM-Archive\codex-original\reports\restore-attempts-20260831`，原 staging
根目录已确认为空并删除。三个本轮 pytest 临时目录仍因当前 Windows 身份的 ACL 无法删除，
不属于项目数据，待具备相应权限时按精确路径清理。Git 不提交 outputs 恢复文件、归档或
临时材料；本次新增恢复工具和测试已在 `f61aa7b` 提交，并随当前分支普通推送到 GitHub。

## Evidence-core harness 首轮收敛（2026-08-31）

基于维护开始时 `HEAD f9e0a75`，完成 Evidence-core-first 路线的首个叶子原语：在现有
evidence IO 中新增并导出 `atomic_write_new_json`。它沿用现有 JSON 缩进、排序和换行格式，
先写同目录临时文件，再以 `os.link` 无覆盖发布；目标已存在时 fail-closed 保留原文件，
不支持 hard-link 的文件系统直接传播原始 `OSError`，成功或失败均清理临时文件。既有
`atomic_write_json` 覆盖语义保持不变。

首批仅迁移三个近期 Scaled-Chen consumer 的本地 `_write_new` 薄适配器：horizon、geometry
attribution 和 manifold readiness。未改变 `EvidenceRecordV2`、配置/CLI、科学 Gate、
输出格式、checkpoint、Git/资源策略或 Manifold Readiness 的 status、classification、
readiness 与 next action；geometry decomposition 和 manifold compact report 的既有覆盖
写入路径也未纳入本轮迁移。

本轮修改文件为 `src/ldm_project/v2/evidence/io.py`、
`src/ldm_project/v2/evidence/__init__.py`、上述三个 research consumer、
`tests/v2/test_manifests_and_checkpoints.py` 以及本文件。验证结果：compileall 通过，Ruff
通过，evidence IO 定向测试 `2 passed`，三个路线定向测试 `20 passed`，默认完整 pytest
为 `1056 passed, 12 deselected, 25 subtests passed`。完整 pytest 仅有一个非测试语义的
`PytestCacheWarning`：当前身份无权写入既有 `.pytest_cache`；未影响执行或结果。

本轮未运行 CUDA、训练、科研 audit，也未生成新 evidence；现有科研状态保持不变：
`valid_failed`、`training_authorized=false`、`formal_promotion=false`，唯一下一动作仍为
`manual_review_scaled_chen_distributed_bottleneck_closeout_contract`。用户已有的
`.pytest-tmp-full-20260831/` 和 `.pytest-tmp-history/` 未修改；本轮专用临时目录已清理。

## Git facts harness 第二轮收敛（2026-08-31）

基于首轮提交 `814c5ef`，将 Scaled-Chen BOGT、geometry attribution 和 manifold readiness
的 `_git_state()` adapter 收敛到既有 `optical_manifold._git_state()`。本轮没有新增
`GitFacts` 类型、evidence API、schema 或 policy helper；事实返回仍为 `commit`、`branch`、
`clean` 和原始 `status` 行，Git 异常继续原样传播，三个路线的 branch/clean 拒绝仍由
各自本地 policy 执行。horizon 的 BOGT 委托关系和 base helper 未改变。

本轮新增的临时 Git 仓库测试覆盖 clean、tracked/untracked dirty、detached HEAD、三个
route-local policy 拒绝以及非 Git 目录异常传播。定向测试最终为 `45 passed`；compileall、
Ruff 和默认完整 pytest 均通过，完整结果为 `1063 passed, 12 deselected, 1 warning,
25 subtests passed`。warning 仅为当前身份无法写入既有 `.pytest_cache`，不影响测试执行。

本轮未运行 CUDA、训练、科研 audit，也未生成新 evidence；`valid_failed`、
`training_authorized=false`、`formal_promotion=false` 和唯一下一动作
`manual_review_scaled_chen_distributed_bottleneck_closeout_contract` 均未改变。当前仍不
刷新 Sources；用户既有 `.pytest-tmp-full-20260831/` 与 `.pytest-tmp-history/` 保持不动，
本轮专用临时目录已清理。

## Scaled-Chen Back-Relay Axial / Conjugate 归因（2026-09-01）

基于实现提交 `0307119ee7da2ad38905fa7f20066eefd927b5c2`，完成了假设
`H-LOMA-CBOGT-S0-BRAC` 的冻结、无更新 CUDA 审计。它严格先重放 DCM baseline，
再比较 nominal relay 与 detector-side `±1.1681239556097333e-05 m` 的 axial
offset；模型、OLS、relay、formal config 和 exposure 均只读，未创建 optimizer、
未 backward、未调用 evaluator、未采样 prior、未写 checkpoint。

唯一权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_bogt_back_relay_axial_conjugate_v1_attempt_03/`。
validation、preflight、audit、report、compact、panel、closeout 的 SHA-256 依次为：
`568670d5ccd4b151335fc8cc97c9c6cd15d5dd4dfd29d40c41c9fa1ce0e9384d`、
`15116230c4fdb7e8a59593bc754b24748bfe698837a9ef1a7a7a742ebacaf984`、
`2800a7c0c61008df0450ae3ef6e4ecc9779d7796ab16b29a7b20611d241815aa`、
`6749b336127e13b70e89c26f39956e52737d40a58344e902d11a90ccdbce579b`、
`099c77193f56dfb7c5b048b19025a914469e2d9af534bc5acb88ba40a7c6c785`、
`8a76e6aaf2df491d83f9cd322a6dffb7597de3fb52bb0fd0fc739a1bbf750ed0`、
`0dd3c67fcedb67df3da7bb4916723c905aa31e3491305522d222f8cee7d7c124`。

preflight、baseline replay、finite/frozen/resource 和 phase/square-law controls 全部通过，
审计状态为 `valid_failed`，分类为 `relay_axial_conjugate_not_supported`；
`aperture_effectively_inactive=true`，`numerical_boundary_not_evaluated=true`。
step-5000 selection 的 relay retention 为 negative/baseline/positive
`0.656313/0.657125/0.658047`，paired MSE 为 `0.0620113/0.0620485/0.0621269`。
相对 baseline 的最大 retention 改善仅 `0.0922` 个百分点，paired MSE 反而恶化约
`0.1265%`，没有达到预注册的实质改善门，故没有 candidate、没有 confirmation arm。
confirmation panel 未被错误地升级为因果支持；本次 panel 明确保留为
`mechanism_evidence_only`。

峰值 CUDA `2588578304` bytes（低于 5 GiB），运行目录约 `195799` bytes，D: 安全余量、
状态 hash、relay fingerprint 和数据隔离门均通过。该结果不推翻父级
`relay_propagation_bottleneck_supported`，只说明当前 nominal relay 附近的微小轴向/共轭
偏移未提供可归因的改善证据。`distributed_bottleneck_supported`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false` 和
`generation_readiness=not_ready` 均保持不变。

唯一下一动作是 `manual_review_scaled_chen_local_relay_tuning_closeout_contract`；在人工
审查前不训练、不启动 FMF、不采样 prior、不生成、不 promotion，也不把 axial/conjugate
诊断结果写成新的 decoder 或 geometry 结论。attempt-01/02 产物按 write-once 规则保留，
但不作为本次权威证据。

## Scaled-Chen Backend Observable Subspace Attribution（2026-09-01）

假设 `H-LOMA-CBOGT-S1-BOSA` 在实现提交
`8ece7fb6f0eb8208e4b50ada87bbfaaf981d648b` 上完成了冻结、无更新的 backend
observable-subspace 审计。它固定 Identity step-5500 与 Scaled-Chen step-5000，复用
MRA/DCM 的 selection panel、native replay 和四格 representation/backend 设计：
`I→I`、`I→S`、`S→I`、`S→S`。没有重新运行 OLS、没有交换/重采样/归一化，也没有创建
optimizer、调用 evaluator、反传、采样 prior 或写 checkpoint。

权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_backend_observable_subspace_v1_attempt_05/`；
`validation.json`、`audit.json`、`compact_report.json`、`panel.pt` 的 SHA-256 分别为
`105606ac5886c2374d09d49443faa878f130811181f68ccd758d239e64f4a7d7`、
`1131860efc1500c4807d773eed6e75e49737972eff05e46bbbefea9e9e993baf`、
`67fbf72f2d7775d87f9c415babadc60f3794e4e3eb251a8ccdda0df1ff2ec6be`、
`a11a3c356ce7146bd0f491bc29c87f4219f65131e62af08595d27dae7eb08fc7`。attempt-01--04
按 write-once 规则保留，但不参与结论。

结果为 `valid_failed`，分类为
`backend_observable_subspace_mismatch_not_supported`，唯一允许的下一动作是
`manual_review_high_fidelity_relay_numerical_reference_contract`。selection 预注册门只在
representation contrast 上失败：`R_prop` 为 `I→I=0.9912`、`I→S=0.3256`、
`S→I=1.0086`、`S→S=0.6571`；Identity 相对 Scaled-Chen 的传播差为
`-0.01736/-0.33149`，方向与要求的稳定 Identity 优势相反；detector-side 差为
`+0.11389/+0.11386`，但不能替代四个 representation gap。两 backend 的功率差均低于
`5%` confound 门（约 `5.5×10⁻⁵` 与 `0`），native replay、平方律/相位控制、finite、
冻结状态和资源门全部通过。由于 selection gate 失败，confirmation 没有执行；不能把
该结果写成 backend mismatch 的支持性因果结论。

该结果不改变 parent `distributed_bottleneck_supported`、DCM 的
`relay_propagation_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false` 或 `generation_readiness=not_ready`。结合
geometry `geometry_not_supported`、BRAC `aperture_effectively_inactive` 与
`relay_axial_conjugate_not_supported`，当前本地 Scaled-Chen v1 relay 微调族登记为
`scaled_chen_v1_local_relay_tuning_family=closed_no_local_correction_supported`；这不
关闭高保真数值 relay reference 或架构重设计，也不授权训练、FMF、prior sampling、
generation 或 promotion。CUDA 峰值为 `2575680000` bytes，输出约 `176147` bytes，
均在资源合同内。

## Scaled-Chen High-Fidelity Relay Numerical Reference（2026-09-01）

基于实现提交 `8835bb228f074408a596118a907248e0f7c9c609`，完成假设
`H-LOMA-CBOGT-S1-HFRR` 的冻结、无更新高保真数值 reference 审计。独立 reference
使用 CPU `complex128` 的直接二维可分离 Fourier-series evaluator，不调用生产
`_fourier_resample`、`_quadratic_phase` 或 `reference_b0_imaging`；生产 Identity
`back_b0` 与 Scaled-Chen `chen_relay` 分别按原 formal 坐标、ABCD、pupil、采样、
detector 与 exposure 回放。没有创建 optimizer、backward、evaluator、prior 或
checkpoint，也没有修改物理 forward、formal runtime、OLS 或训练目标。

权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/`；
`validation.json`、`audit.json`、`compact_report.json`、`panel.pt` 的 SHA-256 分别为
`248983c83cdd122f4a6aa7cf7f38bf6b415d3f080eb8136bb0a4fdb0b8e1c84f`、
`4045e8e0469a745c637c8dae151d073e5b65e06395bf175297d93245f1031e48`、
`d85e82d9a98fb36568d16c63d0fc1a4e5e3a311a007c512f2b07be7472427e7b`、
`dc75d62ed53a2699aa0949fd8bd8a465f0cec2abe46b64279858a304bde9301b`。总产物约
`179883` bytes，低于 `512 MiB`；attempt-01 已占用，后续不得覆盖。

12 组固定合成 probe 以及 Identity/Scaled-Chen native task replay 均通过数值门：
Identity 最大 complex relative L2 为 `8.17e-14`、Scaled-Chen 为 `6.15e-14`，
两者最小 normalized correlation 均高于 `0.99999999999999`，最大 phase RMS 分别为
`2.27e-13`/`2.46e-13`；task replay 最大 complex relative L2 为
`9.23e-7`（I→I）/`8.81e-6`（I→S）/`1.03e-6`（S→I）/`7.67e-6`（S→S），均低于
预注册 `2e-3`。动态 ABCD/pupil 检查与参考一致：Identity
`(A,B,C,D)=(-2.5,-1.94e-16,-14,-0.4)`，source/target pupil occupancy
`7843/10000`、`7845/40000`；Scaled-Chen
`(A,B,C,D)=(-2.5,2.17e-19,-1633.333333333333,-0.4)`，source/target 均满 pupil。

审计状态为 `valid_failed`，科学分类为
`relay_complex_retention_metric_confound_supported`（Class B）；数值子分类为
`relay_numerical_reference_consistent`，`relay_numerical_reference_mismatch=false`，
`numerical_boundary_not_evaluated=false`。也就是说，高保真 reference 没有支持
生产实现错配；但原始复场 retention 的 backend contrast 明显受坐标/二次相位口径
影响：Identity raw gap `0.6517181` 经 analytic-only canonicalization 降为
`0.0081707`（减少 `0.6435474`），Scaled-Chen raw gap `0.3329323` 降为
`0.0193076`（减少 `0.3136248`）。canonicalization 未进入 physical forward、
detector、MSE、训练或任何晋级决策，cross-backend 仍仅作 pupil-aware diagnostic。

CUDA 回放峰值为 `1323107840` bytes，运行前可用显存 `5368709120` bytes，D: 保持
高于 `10 GiB`；模型与 OLS state hash 前后完全一致，`optimizer_created=false`、
`backward=false`、`evaluator_calls=0`、`prior_sampled=false`。相邻测试 48 项和单项
CUDA relay 回归通过；预加载当前 Windows 环境 CUDA DLL 后默认完整 pytest 为
`1102 passed, 12 deselected, 1 warning, 25 subtests passed`。未预加载 DLL 的首轮
失败仅是环境搜索路径问题，不构成代码或科学 evidence 失败。

HFRR 不删除 DCM 的 `relay_propagation_bottleneck_supported`；它将该结论的后续
决策限定为“复场 retention 指标的物理坐标/相位口径复核”，而非直接改写为数值
实现错误或 decoder capacity 结论。`distributed_bottleneck_supported`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`
和 `generation_readiness=not_ready` 均保持不变。唯一下一动作是
`manual_review_relay_retention_metric_reinterpretation_contract`；在人工审查前不训练、
不启动 FMF、不采样 prior、不生成、不 promotion。

## H-LOMA-CBOGT-S1-RMRI：Relay Metric Reinterpretation Audit（2026-09-01）

项目负责人批准了一个 retrospective、same-panel、same-threshold、no-update 的 RMRI，
用于在 HFRR 已验证的 analytic B=0 canonicalization 下重新解释 DCM/MRA/BOSA；不修改
physical forward、生产 relay、OLS、detector、checkpoint 或历史 evidence。实现提交为
`16896f06b31549120adf6235587619fe2820af83`。静态 validation、冻结 CUDA replay 和证据
结构校验均绑定该提交；attempt-01/02 仅保留了前一版实现的 validation，未产生 audit，
不参与当前结论。

权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/`；
`validation.json`、`audit.json`、`compact_report.json`、`panel.pt` 的 SHA-256 分别为
`a87769fb638d5bffa1f77bc31c931fd3999c398a23c1a1f17d51d01eb91882eb`、
`d409729f9b111c024c1913cef90c6cd2ebe778efa9990b836a955783f378ba00`、
`cd39f2fed5719572a558be2bbbf139cdb7cc657a08d74c2be4cd8b69b71c35b8`、
`6b7f9225624fa404a0743a417903b27a7113d100315ea9b1804043c63aafb079`。四个 parent
evidence 的 hashes、两个 checkpoint hashes、selection/confirmation 隔离、HFRR
canonicalizer identity 和 threshold transfer 均通过；BOSA confirmation 按合同未执行。

RMRI 的 raw replay 与历史值完全落在注册 tolerance 内。Scaled-Chen DCM selection/
confirmation 的 raw relay retention 为 `0.65712498/0.60306906`，readout 为
`0.88675454/0.97353794`；canonicalized relay retention 为
`0.98158839/0.98122396`，相应 canonicalized relay loss 为 `0.01841161/0.01877604`，
detector/readout loss 为 `0.11324546/0.02646206`。canonicalized 复场仍保留 label/
retrieval geometry（selection/confirmation 的 label-signal drop 为负或接近零），因此
原始传播断点被 HFRR 解析口径移除，DCM 当前解释为
`relay_propagation_bottleneck_reinterpreted_metric_confound`（D1）；历史
`relay_propagation_bottleneck_supported` 和 HFRR 的
`relay_complex_retention_metric_confound_supported` 均保留。

MRA 的独立段不变：Scaled-Chen `slm→pre-OLS` 为 `0.91817469/0.92159733`，
`pre-OLS→modal` 为 `0.81449253/0.80923885`，`modal→post-OLS` 为
`0.91388854/0.91189128`，`post-OLS→decoder input` 为 `1/1`，
`decoder input→decoder L2` 为 `0.82043698/0.81888384`。因此 OLS 投影与 decoder
内部仍是两个独立 material failures，RMRI MRA 当前解释为
`distributed_bottleneck_supported_canonicalized`（M1），不是 encoder-only 或
decoder-capacity-only 结论；decoder Jacobian rank=100 的历史事实不变。

BOSA selection 四格 raw `R_prop` 仍为 `I→I=0.99121143`、`I→S=0.32563974`、
`S→I=1.00856667`、`S→S=0.65712498`；canonicalized 后为
`1.00012124/0.99160306/1.00003910/0.98158839`。canonicalized contrasts 仅为
`0.00008214/0.01001467`，未达到 `0.05`，故当前解释为
`backend_observable_subspace_mismatch_not_supported_reinterpreted`（B1）；confirmation
仍为 `false`，不能把 selection 变成新的支持性 architecture conclusion。

RMRI overall status 为 `valid_failed`（科学重解释完成但不构成晋级），唯一下一动作是
`manual_review_relay_metric_confound_closeout_contract`。原 MRA 的
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready` 继续有效；不授权训练、FMF、prior sampling、generation
或 formal promotion。CUDA 峰值 `3626808320` bytes（约 3.63 GiB），输出总量约
`388685` bytes，均在资源合同内；flags 仍为 optimizer/backward/evaluator/checkpoint/
prior 全部为零或 false，physical forward 与 historical evidence 未修改。

## H-LOMA-CBOGT-S2-TRBA：Task-Relevant Bottleneck Attribution（2026-09-01）

本 Gate 是 RMRI relay closeout 之后的冻结、无更新 task-relevant 复核；真实消费者是
项目负责人决定当前 Scaled-Chen v1 是否仍有可归因的 OLS 或 decoder-internal 任务瓶颈。
它只测两个独立段：`pre-OLS → modal_complex` 与 `decoder_input → decoder_layer_2`，
不把 relay/CMOS 混入，不修改 formal runtime、FiberArrayOLS、checkpoint、训练目标或
历史 evidence。实现与 validation/audit 绑定 clean implementation commit
`7c1c0f8dcfb878049260043441a117456e5ebe6a`；TRBA 的结果/状态由本次文档提交绑定。

attempt-01 保留为 write-once 的发布失败记录：完整 CUDA 测量已完成，但初版 JSON
contract 错把逐样本 Tensor 带入序列化；该目录只有 validation、panel、compact，未写入
audit，不参与结论。修复后的权威 attempt-02 为
`outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/`，
四个产物 SHA-256 依次为 validation
`3bec36f315c872a05bcb3381929987d8fff57914351a2d5bef4ff1eebda0c659`、audit
`b6be2c6f97dbdcb1d02ad0f2395eca438ebeb619196758a57eb8cc26f31745c0`、compact
`a6015a3fa594515fc24ac1af6cb684d3aa70cd703ae22ae9e8ab3924c86f633a`、panel
`ec76f0760a743ef8fd845b6b93969fac727619e439b4f03109fb40218cc46319`。总输出
`517995` bytes，低于 `512 MiB`；D: 运行后仍约 `36.5 GiB` 可用。

TRBA 的 invariant sample geometry 使用
`S_ij=|<z_i,z_j>|²/[(||z_i||²+eps)(||z_j||²+eps)]`，按完整 complex tensor/注册
port order flatten，逐块计算，不 resize/crop、phase-fit 或 HFRR canonicalize。C0
no-op、C1 per-sample global phase、C2 common reversal permutation、C3 common diagonal
unit-modulus phase 均通过；最大 geometry drift `1.48e-7`、label drift `5.63e-7`，
retrieval drift `0`，均低于预注册容差。MRA 的 panel、bootstrap helper、冻结 state
检查和 RMRI/HFRR lineage 均被复用。

真实结果（selection/confirmation）：

- OLS geometry retention=`0.828977/0.830021`，即 degradation=`0.171023/0.169979`；
  power retention=`0.718075/0.703888`。true-label inter/intra
  `10.873794→11.488753`、`10.590649→11.169436`，permuted-label
  `9.086486→9.085960`、`8.988460→8.988677`；task-advantage degradation 为
  `-0.615485/-0.578570`，invariant retrieval degradation 为 `0/0.002600`。因此
  OLS 只有 `generic_geometry_change_only`，没有 ≥5pp 的 task-relevant secondary loss。
- decoder-internal geometry retention=`0.989736/0.989383`，degradation=`0.010264/
  0.010617`；power retention=`0.759560/0.758735`。true-label inter/intra
  `11.488753→11.519594`、`11.169436→11.199390`，permuted-label
  `9.085960→9.085983`、`8.988677→8.988709`；task-advantage degradation
  `-0.030818/-0.029922`，retrieval degradation `-0.002000/-0.001200`。该段未达到
  material primary 或 secondary 门，分类为 `not_supported`。

两个 panel 的 finite、冻结和 transform controls 全部有效，CUDA 峰值
`2281484800` bytes（约 2.28 GiB），运行前可用显存 `5368709120` bytes，wall time
`225.297` 秒仅作观察。`optimizer_created=false`、`backward=false`、
`evaluator_calls=0`、`checkpoint_written=false`、`prior_sampled=false`、
`physical_forward_modified=false`、`historical_evidence_modified=false`。

TRBA 整体科学分类为 T4：
`task_relevant_bottleneck_not_supported`，唯一下一动作是
`manual_review_scaled_chen_bottleneck_attribution_closeout_contract`。这表示当前数据
不支持 OLS-only、decoder-only 或 task-relevant distributed bottleneck；OLS 的几何变化
仍是机制观察，但没有被 task-relevant control 确认为信息瓶颈。历史 RMRI/MRA 的
`distributed_bottleneck_supported_canonicalized` 继续保留为 raw/rank-context 历史事实，
不被删除或静默改写。TRBA 不提供 reconstruction 或 generation evidence，
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready` 保持；不授权训练、FMF、prior sampling、generation
或 formal promotion。

默认完整 pytest 为 `1121 passed, 12 deselected, 1 failed`；唯一失败是既有
`test_near_conjugate_relay_equivalence_workflow_writes_bound_passed_evidence` 在较长
临时路径下创建缺失父目录，与 TRBA 文件无关。TRBA/邻近 MRA、RMRI、HFRR、BOSA、
geometry、fiber 测试及 compileall/Ruff/diff-check 均通过。

## H-LOMA-CBOGT-S3-RMFA：Reconstruction-Manifold Failure Attribution（2026-09-01）

这是 TRBA closeout 后唯一获准执行的冻结、无更新 bounded-recoverability 审计。真实消费者是项目负责人决定“当前 physical mapping 是否与 Scaled-Chen representation 错配”，拒绝决定是：若一个在 selection 前锁定、低自由度的 closed-form readout 不能跨过既有 reconstruction barrier，不得把失败写成 representation insufficiency，也不得训练新 decoder 或继续当前 Scaled-Chen 生命周期。实现提交为 `2445d946fd26867d1c9bd98b04f262b50cd0c90e`；没有修改 formal v2 runtime、OLS、checkpoint、训练目标、detector、prior 或历史 evidence。

权威 CUDA attempt-01 位于 `outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/`。validation、audit、compact、probe SHA-256 分别为 `08e6afd841574a5ac6ba126d11af24b560b6f85bf2d5365d42fd229b373625c0`、`e0a53625171144b7aba9dbdc3258ff94ca2686219a80d0725f535d2625c7d468`、`174bdf09c8b5a33896b3a22396f8dfaea6dd22200836f9bbbe46d53b57c403f5`、`25d9eba0d6203027b3a6be00a9d0a5538225510e45c0927f63cf8f78caaf4a09`。输出总量为 `82223871` bytes，低于 `512 MiB`；峰值 CUDA `3588494848` bytes，运行前可用显存 `5368709120` bytes，D: 安全余量与写一次目录门均通过。

probe 只在预注册、与两个评估 panel 不相交的 train fit subset（1000 个、每类 100）上构造一次，输入是复数 pre-OLS 的固定 `16×16` real/imag adaptive-average-pool 投影加 bias，`input_dim=513`；目标是原 paired MNIST reconstruction target，输出 `40000`。求解器为 float64 Moore–Penrose minimum-norm pseudoinverse，固定 `rcond=1e-12`，无 lambda sweep、无 adaptive rank、无 optimizer/backward。矩阵 rank 与 effective rank 均为 `513`，condition number `7945.41016660053`，probe 在 fit 后冻结。

selection（MNIST test 5000–9999，每类 100）native physical anchors 完整复现：paired/template/cyclic MSE=`0.0620484613/0.0324799940/0.0799509138`；冻结 probe paired MSE=`0.0033369721`，相对 physical 改善 `94.62198%`，严格低于 template 和 cyclic，instance-sensitive geometry correlation=`0.97861648`，selection gate 全部通过。confirmation（train 55000–59999，5000）anchors 为 `0.0577895157/0.0277411994/0.0756082982`；同一个 probe paired MSE=`0.0033930303`，相对改善 `94.12865%`，strict template crossing 与 instance control 继续通过，paired-MSE difference 的固定 2000 次 bootstrap 95% CI 为 `[0.0540225357, 0.0547624826]`，下界严格大于零。selection/confirmation 未重新拟合。

因此 RMFA 为完整 harness 下的 `passed` / R1：`reconstruction_mapping_misalignment_supported`。这只证明预注册的冻结线性 readout 能从当前 representation 恢复 physical mapping 丢失的 instance-level target 信息；不证明 representation 对任意非线性 decoder 充分、不证明光学架构已修正，也不授权 generation。RMFA 的 bounded diagnostic field 可记为 recoverability positive，但原 formal/physical `reconstruction_manifold_supported=false` 结论不被静默升级；`generative_manifold_supported=false`、`generation_readiness=not_ready` 保持。

本 Gate 同时正式关闭 task-relevant bottleneck attribution 生命周期：`closed_task_relevant_bottleneck_not_supported`。历史 MRA raw contraction、DCM raw relay attribution、HFRR/RMRI metric reinterpretation 与 TRBA task-relevant negative result 全部保留；RMFA 不是 OLS/decoder 训练授权。唯一下一动作是 `manual_review_reconstruction_mapping_alignment_contract`，不得自动训练、FMF、prior sampling、generation canary、candidate export 或 formal promotion。

治理 flags 全部符合 no-update 合同：`optimizer_created=false`、`backward=false`、`evaluator_calls=0`、`checkpoint_written=false`、`prior_sampled=false`、`generation_started=false`、`physical_forward_modified=false`、`mechanism_evidence_only=true`。

实现后的默认 pytest 仍受同一 Windows 临时根权限问题影响：默认根目录运行得到 `1015 passed, 12 deselected, 116 errors`，错误均为访问 `C:\TEMP\pytest-of-24567`；在新的可写 `C:\Windows\Temp\codex-rmfa-full-20260901` 下得到 `1131 passed, 12 deselected, 25 subtests passed`，无 RMFA 新失败。RMFA/相邻研究测试、compileall、Ruff 与 `git diff --check` 均通过。

## H-LOMA-CBOGT-S4-RMAL：Reconstruction Mapping Alignment Localization（2026-09-01）

RMAL 是 RMFA 之后唯一获准的冻结、无更新、下游到上游 bounded-recoverability 审计。真实消费者是项目负责人判断当前 Scaled-Chen step-5000 的 physical reconstruction mapping 是否存在可由冻结 readout 恢复的错配；拒绝决定是不得把某个 stage 的 bounded recoverability 结果升级为通用 representation sufficiency、架构修正或 generation 授权。实现 commit 为 `5b274b926474784a1ad49744e75190afff31747c`，没有修改 formal v2 runtime、OLS、checkpoint、训练目标、detector、prior 或历史 evidence。

权威 CUDA attempt-01 位于
`outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/`；validation、audit、compact、probes 的 SHA-256 分别为
`affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec`、
`5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa`、
`c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137`、
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt 总输出为 `41230970` bytes，低于 `512 MiB`；运行前可用显存 `5368709120` bytes，峰值 CUDA `2274099712` bytes。

审计固定一次 fit（1000、每类 100）的 float64 `torch.linalg.pinv(rcond=1e-12)` minimum-norm probe，并按
`cmos_intensity → cmos_incident_complex_field → decoder_layer_2 → decoder_input → modal_complex_100 → pre_ols`
执行。RMFA pre-OLS positive-anchor selection/confirmation replay 通过；RMAL 首个阶段
`cmos_intensity` 的 selection physical/probe/template/cyclic paired MSE 为
`0.0620484613/0.0098129278/0.0324799940/0.0799509138`，relative improvement=`84.18505%`；confirmation 为
`0.0577895157/0.0101852901/0.0277411994/0.0756082982`，relative improvement=`82.37519%`，固定 bootstrap 95% CI=`[0.0472728200,0.0479231887]`。两个 panel 的 selection/confirmation gates 全部通过，因此
`furthest_downstream_recoverable_stage=cmos_intensity`，其余五个阶段按 stop rule 跳过；classification 为
`bounded_recoverability_established_at_detector`。

这是 bounded diagnostic recoverability positive，不是 formal reconstruction 或 generation 通过：
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready` 保持。全程 `optimizer_created=false`、`backward_calls=0`、
`evaluator_calls=0`、`checkpoint_written=false`、`prior_sampled=false`、
`generation_started=false`、`physical_forward_modified=false`、`mechanism_evidence_only=true`；
唯一下一动作是 `manual_review_RMAL_recoverability_result`，不得自动训练、FMF、prior sampling、generation 或 promotion。

实现文件为：RMAL producer、研究 YAML、review contract、RMAL tests，以及 research CLI schema/router；
无关既有全套 pytest 中仍有 1 项 near-conjugate 临时 run 目录 `FileNotFoundError`（`1144 passed, 1 failed, 12 deselected, 25 subtests passed`），与本次文件无关。RMAL 定向测试为 `14 passed`，直接相关回归为 `56 passed`，compileall、Ruff 与 diff-check 均通过。

## H-LOMA-CBOGT-S5-ZCSA（2026-09-01）

ZCSA 在起始 HEAD `0dead1b73655b3556450cedbb5d0a2c579385209` 上完成；实现提交为
`1f5e553f6c7ad80217427ab83f5669e7b9758acb`。新增零列语义审计 producer、研究 YAML、review
contract、18 项测试及 research CLI schema/router；未修改 RMAL/RWSA 既有 producer/probe/evidence、
checkpoint、OLS、detector、target renderer 或 formal runtime。审计严格复用冻结 RMAL probe 与固定
MNIST renderer，CPU target-only，无 optical replay、CUDA forward、model load、optimizer、backward 或 evaluator。

正式 attempt 为 `outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/`：
`validation.json` SHA256=`d422f1e31e8298f9662896f832b207ad896fb619c5a656bccf54b099cd419954`，
`audit.json` SHA256=`0cba5990462e9f8cc238885b14c90c907a7c6f085877b6c8ad0c8b95cbf0bfb8`，
`compact_report.json` SHA256=`772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6`；
冻结 probe SHA256=`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed` 未变。
RMAL/RWSA 上游身份与旧 RWSA attempt-02 保持 immutable；旧 RWSA audit 仍为
`invalid_harness`，不被本结果覆盖。

ZCSA status=`passed`、classification=`zero_column_semantics_attribution_completed`。40000 列 taxonomy
完整：`dynamic_active=33502`、`target_zero_static=6498`、`bias_only_constant=0`、
`bias_only_variable_target=0`、`zero_W_constant_target_bias_mismatch=0`。zero-W 与 fit always-zero
交集为 `6498`，与 selection/confirmation ever-nonzero 的交集分别为 `1495/2097`；fit-zero→active
转移同为 `1495/2097`。zero-W 的 b 全部 exact-zero（6498），bias-only fit MSE 为 `0.0`，
selection/confirmation MSE 为 `1.9136665239926873e-05 / 2.4601896459000674e-05`。
空间摘要为 pixel=`6498`、edge=`660`、interior=`5838`、4-connected components=`2`，最大/次大
component=`5105/1393`；fit/selection/confirmation overlap=`0/1495/2097`。

验证：ZCSA 定向测试 `18 passed`，直接相关 RMAL/RWSA/renderer/MRA 回归 `57 passed`，compileall、
Ruff、diff-check 通过；使用显式短 basetemp 的全套默认 pytest 为 `1197 passed, 12 deselected,
25 subtests passed`。`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 保持；
`closed_task_relevant_bottleneck_not_supported` 保持。唯一下一动作是 `manual_review_ZCSA_result`；
不得训练、FMF、prior sampling、generation、candidate export 或 formal promotion。结果/状态 commit
后需 refresh bridge 并检查 status，不自动上传。

## H-LOMA-CBOGT-S5-RWSA-ACTIVE：Detector Readout Structure Attribution（Active-Domain Amendment，2026-09-01）

本次是用户批准的 RWSA active-domain amendment。真实消费者是项目负责人判断冻结 RMAL detector readout 在 dynamic-active 输出域上的空间结构、符号与 bias attribution；拒绝决定是不能把 descriptive structural result 写成 universal representation sufficiency、架构修正、reconstruction/generation readiness 或 training/promotion 授权。

起始 lineage HEAD 为 86e0862690cdb32173e453869ed33b8eb2b4c017。实现与修补依次落在 eed3bb3、6521893、ca372c9、e0c442f；正式 validation/audit 的 clean implementation/result commit 为 e0c442f92dd23b71d048ceb647570b76406c0435。新增 active-domain producer、研究 YAML、review contract、测试和 CLI schema/router；旧 RWSA attempt-02、ZCSA/RMAL 产物、冻结 probe、target renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 均未修改。

因 write-once 规则，attempt-01 只保留 validation-only retry 记录：目录为 outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_01/，validation SHA-256 为 8c20e5629e36e2d88d2358512b95f5cd0858999e21d981f864ea8f541f44a539；run 在 transition harness 修复前停止，未写 audit，不消费该 attempt。权威 attempt-02 为 outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_02/，validation/audit/compact SHA-256 分别为 686d99f331c99b30a73cb3a70cc1002aa0a21a33b014e2a59a5e6c33f8a5e15d、cb3feabbfc0b70e23d43ae18a354dabc112319eb11d2c8cea2c13f6bd5c0337、23b38d3231594cb0eb3ebab21972b8979ffaf825e17c901f4b86d46eeddc1c11；active config canonical SHA-256 为 2115380c676fc710470429abed9438d154a1758db99e5635d6295eeaae03a25d。

冻结 probe 仍为 RMAL 的 [257, 40000] float32 full readout（W=[256,40000]、b=[40000]），probe SHA-256 为 3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed。dynamic mass 定义为 d_j=sum_i abs(W_ij)，active 仅取 d_j>0，inactive 取 d_j==0；完整域为 40000，dynamic-active 为 33502，zero-W 为 6498，W 与 b 在 inactive 列均 exact zero，active mask SHA-256 为 afbda35eca136c1275c5771bc9540f7985dd66f77a4c6ea596940649848352ea。inactive 概率不计算、不用 epsilon、不静默删除，所有 inactive 列保留在 decomposition。

fit 使用预注册 1000 个 train fit 样本（每类 100），只定义 RMAL/ZCSA identity；selection 是 MNIST test 5000–9999 的 1000 个样本，confirmation 是 train 55000–59999 的 5000 个样本，未重拟合 readout，candidate prefix 未消费。target 仍由固定 base.render_intensity_batch 产生；16×16 adaptive-average-pool 使用重叠 bins (0,13)、(12,25)、…、(187,200)，坐标为 centered row=y、column=x、pixel pitch 2.5e-6 m、FOV 0.5 mm。D4 controls 为 identity、x_flip、y_flip、xy_flip、transpose、transpose_x_flip、transpose_y_flip、transpose_xy_flip。permutation null 为 source-feature-row-only、同一 active mask、256 次、seed=0，仅描述不设 scientific threshold。

正式结果为 status=passed、classification=descriptive_structural_attribution_completed，且 conditional_on_dynamic_active_domain=true。L_active=0.003260666999；top-1/top-4 mass=0.0806218534/0.2238490484，expected-support locality=0.0032606670，radius 0/1/2 locality=0.0032606670/0.0241852561/0.0619379424；spatial variance=7.1791287e-08 m²，RMS support radius=267.8815 µm，effective support=44.3347，normalized entropy=0.8016757，row/column concentration=0.2319700/0.2541995，positive/negative mass=0.4990808/0.5009192，cancellation fraction=0.9248245。null mean/std=0.0042417098/0.0003771318，observed empirical rank=1/256；这些数值是 descriptive structural statistics，不构成 bottleneck 或 readiness 判定。

两组 exact CMOS-only no-update replay 的 registered physical anchors 均通过：selection physical paired/template/cyclic MSE=0.0620484594/0.0324799946/0.0799509163，confirmation=0.0577895189/0.0277411913/0.0756082949，均在既有 tolerance 内。selection 的 all_40000 paired/full/dynamic/bias MSE=0.0620484594/0.0098129277/0.0126702857/0.0984088076，dynamic_active=0.0740748917/0.0117125174/0.0151240844/0.1174923265，zero_W=4.3298663e-05/1.9136666e-05/1.9136666e-05/1.9136666e-05。confirmation 对应为 all_40000=0.0577895189/0.0101852894/0.0130028048/0.0876878199，dynamic_active=0.0689895374/0.0121560417/0.0155200385/0.1046908519，zero_W=4.5133395e-05/2.4601896e-05/2.4601896e-05/2.4601896e-05。

zero-W error contribution（paired/full/dynamic-only/bias-only）为 selection=0.0001133609/0.0003168016/0.0002453576/0.0000315902，confirmation=0.0001268728/0.0003923873/0.0003073628/0.0000455773。fit-zero 与 zero-W mask 完全一致；fit-zero→selection target-ever-nonzero transition=1495 列，fit-zero→confirmation target-ever-nonzero transition=2097 列，transition error 仍只归入 zero_W_transition_columns。zero-W/active masked class-template、cyclic 和 instance-sensitive metrics 按合同均为 not_evaluated_without_unambiguous_masked_output_semantics；未保存 raw CMOS bank 或大 feature bank。

回放只执行 exact panel CMOS model replay：compact 记录 exact_panel_cmos_only_replay_executed=true、optical_replay_executed=false；optimizer_created=false、backward_calls=0、evaluator_calls=0、checkpoint_written=false、prior_sampled=false、generation_started=false、physical_forward_modified=false，模型/OLS/checkpoint/artifact integrity 全部保持不变。资源峰值 CUDA=303070720 bytes，wall time=183.036 s，最终三文件总量=240004 bytes，resource/storage gates 通过。

全套默认 pytest 在显式短 basetemp 下为 1214 passed、12 deselected、25 subtests passed；新增 active-domain 定向测试为 17 passed，直接相关回归为 96 passed；compileall、Ruff、git diff --check 与 CLI contract checks 均通过。唯一 warning 是既有 .pytest_cache ACL（C:\TEMP / D:\codex-original\.pytest_cache），不是测试失败。

本结果仍保持 reconstruction_manifold_supported=false、generative_manifold_supported=false、generation_readiness=not_ready、training_authorized=false、formal_promotion=false 与 closed_task_relevant_bottleneck_not_supported。唯一下一动作是 manual_review_RWSA_active_domain_result；不得训练、FMF、prior sampling、generation、candidate export 或 formal promotion。

## H-LOMA-CBOGT-S6-LSDA：Linear Readout Structure Decomposition Audit（2026-09-02）

本轮是 RWSA-ACTIVE 之后唯一获准的 frozen-W、CPU-only descriptive spectral/operator attribution。真实消费者是项目负责人审查冻结 RMAL detector readout 的谱能量、source/target singular-mode 结构与 bias/subspace 归因；拒绝决定是不得把描述性 mode/spectrum 结果写成新的 readiness、架构或训练/晋级授权。起始 lineage HEAD 为 `bd967f64ffb046677fa4204e81e911b226047e6e`，实现提交为 `bed235283d1fa47f4a5a38d41368d87ce2b5d218`。

新增 LSDA producer、YAML、review contract、测试及 research CLI schema/router；RMAL/ZCSA/RWSA-ACTIVE、冻结 probe、renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 均未修改。权威 attempt 为
`outputs/20260901_v2/runs/lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/`；validation/audit/compact SHA-256 分别为
`3f24dcc7d56e098ec778998bb8ddc7871973131f1d7269c86da729395426c2f2`、
`97842d5707541516c79adb87c3a1e609ee533156895908c9e99c015cde0f9b13`、
`e281daf9a4a62f85828a3bc8055ce044027d0c0fb1af0a522febcf3c2b68185b`。冻结 RMAL probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

validation 的 10 个 gate 与 audit 的 8 个 completeness/finite/no-update/storage gate 全部通过。audit status=`passed`、classification=`linear_readout_spectral_attribution_completed`；`W=[256,40000]`、`b=[40000]`，source 为 float32，analysis copy 为 CPU float64，full/active Gram 均为 `[256,256]`，未构造 `40000×40000` Gram。full/active 的前五个 singular values 均为 `12227.8671591/10625.3503547/8724.4504853/8126.3713707/7844.9489846`，top-16 cumulative energy 为 `0.8441783201`；effective-rank telemetry 为 `21.88110383398559/21.881103833985602`，stable-rank telemetry 为 `5.528997319328641/5.5289973193286395`，condition profile 为 finite，全部 256 项谱字段已保存。

full-vs-active identity 保持：full/active/inactive columns=`40000/33502/6498`，inactive exact-zero identity=`true`；最大 eigenvalue absolute/relative difference=`4.470348358154297e-08/8.328885261127632e-13`，最大 singular-value difference=`2.900790718740609e-12`，effective/stable telemetry difference=`1.0658141036401503e-14/1.7763568394002505e-15`。PSD raw spectrum 无负值，full/active minimum eigenvalue=`20.898962877319427/20.898962877313448`，tiny-negative clamp=`0`，未使用 cutoff。

K=`16` 仅为 descriptive preview；source/target mode descriptor 均标记 basis-dependent，所有 target preview mode 均可用。source/target preview orthogonality residual=`1.5695833278775036e-15/3.095197802817193e-14`；每个 mode 均保存 sign-canonicalized source、target 的 spatial/finite-difference、FFT low/mid/high、center/middle/edge、D4 sign-invariant residual 与 signed summary，D4 controls 为 identity、x_flip、y_flip、xy_flip、transpose 及四个 transpose+flip 变体，不据此形成额外分类。mode-0 的 source/target spatial smoothness score=`0.2219516623/0.9468270025`，source/target RMS radius=`184.8835/133.7382` µm；对应 source high-frequency fraction=`0.8366757445`、target low-frequency fraction=`0.9982443165`，均为描述性字段。

global reference 严格为 RMAL fit panel 的 1000 个 unlabeled targets，streaming mean hash=`513a696e21dafebebe4a862ab814b06294beb662db2afcfd679899fd19b27daf`，selection/confirmation 未消费，class-template reference=`not_evaluated`，未保存 target bank。bias norm²=`114.5276114768678`，preview top-16 projection energy=`21.21861242530154`，preview 外 residual bias energy=`93.30899905156626`，未 refit。

全程 `cuda_used=false`、`optimizer_created=false`、`backward_calls=0`、`probe_refit=false`、`decoder_called=false`、`physical_forward_called=false`、`rank_truncated_reconstruction=false`；`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported` 保持。新增 LSDA 定向测试 `18 passed`，直接相关回归 `101 passed`，全套 Windows-native pytest（显式可写 basetemp）为 `1232 passed, 12 deselected, 25 subtests passed`；compileall、Ruff、git diff --check 均通过。唯一 warning 为既有 `.pytest_cache` ACL。唯一下一动作是 `manual_review_LSDA_spectral_result`；不得训练、decoder、RMAL replay、generation 或 promotion。

## H-LOMA-CBOGT-S6-LRRC：Low-Rank Readout Recovery Confirmation（2026-09-02）

LRRC 的真实消费者是项目负责人确认冻结 RMAL detector readout 的 functional low-dimensional
recoverability；拒绝决定是不得把 LSDA spectral concentration、任一 candidate rank 或本结果
升级为 mathematical minimum、intrinsic/physical/architecture rank、training、generation 或
formal promotion。实现/config/review/test/CLI 的 clean implementation commit 为
`2a061db61910a847118e38663d05c05bd9447063`；RMAL/ZCSA/RWSA-ACTIVE/LSDA、冻结 probe、
renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 未修改。

权威 attempt-04 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/`；
validation/audit/compact SHA-256 分别为
`b305c38dc3c774ff995c98907977d67cfc396d56009b8a88f726e214f9efa004`、
`5ffcb3550e0a1d635f70cdf965d12c93ded6e99fadf8f0f6ef0cda829b4a6b58`、
`4ae704104accd8cebfcb0e3c19dba7deb90b4febf38bace3c7e578bd484951b8`。输入保持
`W=[256,40000]`、`b=[40000]`，RMAL probe SHA-256=`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`，active/inactive=`33502/6498`。

预注册 candidate family 固定为 `[1,2,4,8,16,32,64,128,256]`，rank-0 只作 bias-only control。
rank-256 full positive anchor 首先执行且 selection/confirmation 均通过；factorized-vs-full
operator max residual 为 `2.4333037798e-06/2.5858223358e-06`（selection/confirmation），
probe replay abs delta 为 `1.1276390065e-10/7.8544505795e-10`。随后 rank `1,2,4,8,16,32,64,128`
严格按注册顺序执行 selection，全部失败，confirmation 均未执行；因此
`first_confirmed_candidate_rank=256`、`previous_registered_candidate=128`、
`rank_256_reused_as_fallback=true`，且没有第二次 rank-256 run。这里的 256 是候选家族内的
首个 confirmed/fallback 结果，不是对一般 rank sufficiency 的声明。

LSDA boundary telemetry 被完整继承：`k=1/2/4/8/16/32/64/128` 的 relative gap 为
`0.131054483/0.178902324/0.034630756/0.105670493/0.104494812/0.061605380/0.043964196/0.052600464`；
对应 `sigma_k` 为 `12227.867159/10625.350355/8126.371370/5321.871383/3216.147739/1630.578030/676.140671/169.436259`，
`sigma_(k+1)` 为 `10625.350355/8724.450485/7844.948985/4759.506609/2880.076987/1530.125651/646.414690/160.523833`。
full spectrum 最小 gap 仍是 rank-36 的 `0.0025793411`；无新 degeneracy threshold，boundary
ambiguity 只作 descriptive telemetry。

结果为 `status=passed`、`classification=low_rank_readout_recovery_confirmation_completed`。
CMOS-only exact panel replay、state/no-update、streaming/no-bank 与 resource gates 全部通过；
`cmos_only_replay_executed=true`、`optical_replay_executed=false`、optimizer/backward/refit/
decoder/generation/physical-forward modification 全部为 false/zero。当前 flags 仍为
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。正式下一动作是 `manual_review_LRRC_result`；
不得训练、candidate export、generation 或 promotion。

本轮最终代码核验：LRRC 定向 `26 passed`，相关 RMAL/LSDA/RWSA-ACTIVE/ZCSA/shared-lifecycle
回归 `130 passed`，默认 Windows-native 全套 pytest 为 `1258 passed, 12 deselected,
25 subtests passed`；compileall、Ruff、git diff --check 通过。强制纳入 historical_artifact
marker 的诊断 lane 中 4 个既有 archive-only tests 仍因历史 checkpoint 路径绑定失败，未计入默认
lane，也未改变 LRRC 证据。此前 attempt-01/02/03 记录保持 immutable；attempt-04 为当前权威。

## H-LOMA-CBOGT-S6-DSFA：Data-Weighted Spectral Functional Attribution（2026-09-02）

DSFA 消费冻结 RMAL `W=[256,40000]`、`b=[40000]`、ZCSA/RWSA-ACTIVE/LSDA 及 LRRC attempt-04，
真实消费者是项目负责人审查低 operator-energy mode 在真实 detector data 上的 functional
effect；拒绝决定是不得把 data-weighted attribution 升级为 reconstruction sufficiency、
最低 rank、training、generation 或 promotion。实现提交为
`d032738c0f02af5ada85554224e045d238416e0a`，未修改上游 evidence、probe、renderer、checkpoint、
OLS、detector、formal runtime 或物理光路。

权威 attempt 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/`；
validation/audit/compact SHA-256 分别为
`ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe`、
`2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e`、
`8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3`。probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

结果为 `passed / data_weighted_spectral_functional_attribution_completed`。LSDA exact ordered
CPU float64 spectral identity 与 q-norm identity 通过；selection bias/full MSE 为
`0.09840880738327583/0.009812927922827133`，confirmation 为
`0.08768781986771307/0.010185289342506197`，对应 total signed functional gain 为
`0.0885958794604826/0.07750253052523498`。B256（modes 129–256）operator-energy share 为
`0.0007625489`，但 data-weighted output-energy share 为 selection/confirmation
`0.3517092/0.3451470`；该差异仅作 descriptive attribution。所有 256 mode 与固定 B1–B256
band 字段均保存，selection registered-prefix 与 confirmation full-rank LRRC replay 均通过。

正式 replay 为 CUDA CMOS-only no-update streaming；selection/confirmation additive identities、
finite、state/OLS unchanged、no-bank、peak CUDA `2274099712` bytes 与 resource gates 全部通过。
optimizer/backward/refit/decoder/physical-forward modification/generation 均为 false/zero；
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported` 保持。DSFA 定向测试 `10 passed`，compileall、
Ruff、diff-check 通过；唯一下一动作是 `manual_review_DSFA_result`。

## H-LOMA-CBOGT-S6-DMSA：Detector Data-Metric Spectral Attribution（2026-09-02）

DMSA 消费冻结 RMAL `W=[256,40000]`、`b=[40000]`、LSDA exact ordered CPU float64
spectrum、LRRC attempt-04 与 DSFA attempt-01；真实消费者是项目负责人审查 detector
data second moment 与 centered variance 对冻结 readout spectral structure 的描述性分解。
拒绝决定是不得把 weighted spectral telemetry 升级为 mechanism、reconstruction sufficiency、
minimum rank、training、generation 或 promotion。起始 lineage HEAD 为
`39b0ac449986df31b9102c0b1296dda912f7b39c`，正式 validation/audit 绑定 clean
implementation/result commit 为 `9818e87441eb2c86eef06c18c1aa646f3e57dea5`；上游
RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC/DSFA、probe、renderer、checkpoint、OLS、detector、
formal runtime 与历史 evidence 未修改。

DMSA attempt-01 仅保留 validation-only 的 invalid harness 记录（`three_panel_isolation`
修复前停止，validation SHA-256=`94a29d7ad6c065ac0253700524de1f0f1e23c0e92fe3959b340ae04e26ac63b3`）；
权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/`，
validation/audit/compact SHA-256 分别为
`f0eebcb978851b9ab58453e1b28e920eddc93e86a2b82133af0bd7ec8e8ce93d`、
`c9cc448f1b58040d45b8b77c9cfaa9facd536cdcf40d0b4f152efe42e16411fe`、
`39564f0ce3973bd738d196191a06c7990407f9b7b6ab7e6e8a2eb3366dc2d0ac`。研究 config/review/source
SHA-256 分别为
`48e3e34d47a486fd50052f2d805d387c84fc216a9e9498775059c2de45560828`、
`2899715f961a19cdee83b53f709c9b77498ab618afe5963c6c63a5492ea2501a`、
`88356728b165e7c1ff91fb825292a635901da2d02c60841da972b6434d36112f`。

结果为 `passed / detector_data_metric_spectral_attribution_completed`。fit/selection/confirmation
分别使用 1000/1000/5000 样本、population `N`、float64 streaming `S=C+mu muT`；三组
moment identity 通过，selection/confirmation 的 `E_mean/E_var` fractions 为
`0.4019555898/0.5980444102` 与 `0.3704005497/0.6295994503`。冻结 raw spectrum 与 LSDA
identity residual 为零；second-moment effective/stable rank 为 selection/confirmation
`14.8728108/2.2436135`、`17.4264177/2.4233991`，centered-variance 为
`36.6563260/8.1270337`、`39.8650006/8.4787742`。这些均为 descriptive telemetry，未设置
scientific dominance 或 panel-agreement gate。

正式 replay 为 exact-panel CMOS-only、no-update、streaming、no-bank；15/15 audit gates
通过，peak CUDA=`2274099712` bytes、wall=`84.922 s`。`optimizer_created=false`、
`backward_calls=0`、`probe_refit=false`、`decoder_created=false`、`optical_replay_executed=false`、
`generation=false`，state/OLS unchanged；`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported`
保持不变。

DMSA 定向测试 `21 passed`，直接相关回归 `111 passed, 1 deselected`，compileall、Ruff、
git diff-check 与 CLI contract checks 通过。全套 pytest 为 `1285 passed, 12 deselected,
25 subtests passed`，另有 4 个与 DMSA 无关的既有环境失败（3 个 NVRTC DLL/运行时失败、
1 个 near-conjugate 临时路径失败）。唯一下一动作是 `manual_review_DMSA_result`；不得训练、
 decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S8-MPCDAR-R16-IQ：Frozen MPCDAR Rank-16 Readout Integration Qualification（2026-09-02）

S8 的真实消费者是项目负责人确认 S7 rank-16 mean-preserving data-aware readout 的
frozen artifact、fresh-process reload 与完整 RMAL bounded selection/confirmation integration。
拒绝决定是不得将该 post-detector repair candidate 解释为 minimum/intrinsic rank、native
optical reconstruction、一般 reconstruction sufficiency，或 decoder/training/generation/
promotion authorization。S7 producer、RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC/DSFA/DMSA/MCFA、
probe、renderer、checkpoint、OLS、detector 与正式 physical runtime 未修改。

实现最终绑定 clean commit `c0140577fddb56af79d72a6d473cc49ef04f5ffc`；此前的
`f48afb89286f2051732c8c64e6ee91aa593f7a52` 与
`5b90b1097522e7b68ffdf80f7f6348307a2b7df8` 为同一实现的前置修复 commits。权威
attempt-03 位于
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/`；
validation/audit/compact SHA-256 分别为
`844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb`、
`3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4`、
`12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271`；manifest SHA-256 为
`e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45`，payload SHA-256 为
`e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e`，probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01/02 按
write-once 规则保留；attempt-03 是当前权威结果。

Artifact 为四个 CPU float64 tensors：`mu_fit[256]`、`c_mean[40000]`、`T16[256,16]`、
`V16[40000,16]`；scalar count=`684352`，raw float64 tensor bytes=`5474816`，serialized
payload bytes=`5476856`。mu/C fit replay `N=1000`，hash 分别匹配
`187bb9a64c2142dcb7eb79325f3720bc8d86a6c854c9ddcc4725add4c4069496` /
`be2ac0c63a6bcf290ee04fde074508d823c462368c9360078f99b9556b2b4d52`；moment identity
relative residual=`2.98546e-16`。rank-256 positive replay max absolute residual=`5.68008e-11`、
relative Frobenius residual=`1.17382e-13`，construction identities 全部通过。

Process A build 后退出，Process B 为 fresh process；selection/confirmation loaded-artifact
与 reference prediction residuals 均为零。selection `N=1000` 的 loaded/reference probe MSE
均为 `0.02107268348957069`，full RMAL reference=`0.009812927917651195`，relative
improvement=`0.6603834534521028`，RMAL selection Gate vector 全部通过；confirmation
`N=5000` loaded/reference probe MSE 均为 `0.02128368811373826`，full RMAL reference=
`0.010185289342547572`，relative improvement=`0.6317033169406975`，bootstrap 95% CI=
`[0.03618517890572548,0.036830976605415344]`，confirmation Gate vector 全部通过。audit
为 `16/16` gates，classification=`frozen_mean_preserving_data_aware_rank16_integration_qualified`，
`integration_only=true`。

factorized matmul deterministic operation-count telemetry 为 `644096 MAC/sample`，full-W
comparator 为 `10240000 MAC/sample`，operation-count ratio=`15.9`；这不是 latency、throughput
或 energy-efficiency 测量。runtime independence 中 W/b/C_fit/dataset/probe/research imports
均为 false，default-disabled 与 explicit-artifact contract 通过；peak CUDA=`2274099712`
bytes，wall=`133 s`，CMOS-only streaming/no-bank、state/no-update、finite/resource gates
通过。状态 flags 保持 `reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_MPCDAR_R16_IQ_result`，不得自动 make rank16 default、训练、decoder、
generation 或 promotion。

## H-LOMA-CBOGT-S7-MPCDAR：Mean-Preserving Centered Data-Aware Readout Recovery（2026-09-02）

MPCDAR 的真实消费者是项目负责人确认：在保留冻结 RMAL useful mean branch
`c_mean=b+mu_fit W` 的条件下，fit-derived centered data-aware output subspace 是否在预注册
候选族内保留完整 RMAL bounded selection + confirmation recovery。拒绝决定是不得把结果升级为
minimum/intrinsic/physical rank、一般 reconstruction sufficiency、decoder、training、generation
或 promotion authorization。实现/config/review/test/CLI 及本轮 preflight 支持改动已提交；RMAL、
probe、renderer、checkpoint、OLS、detector、formal runtime 与父级 evidence 未修改。

正式 replay 使用 code commit `b59172750d0c88bbc9f7426ab7a21a1d63639f57`，权威 attempt-03 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/`；
validation/audit/compact SHA-256 分别为
`b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2`、
`43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785`、
`9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9`；RMAL probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01/02 仅为
validation-only 或正式执行前停止记录，均未覆盖或改变旧 evidence。

## H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ：Frozen MPCDAR Rank-16 Float32 Numerical Qualification（2026-09-02）

S9 是 S8 rank-16 artifact 的独立、窄范围、no-update float32 consumer。真实消费者是项目负责人审查：
S8 CPU float64 artifact 在独立 Process A→B、文件系统 fresh reload 和同一 CMOS feature stream 下，
通过严格 preregistered float32 cast/runtime tolerance，并保持 RMAL selection/confirmation Gate
vector identity。拒绝决定是不得将数值资格升级为 GPU-ready、production、native optical reconstruction、
minimum/intrinsic rank、一般 reconstruction sufficiency 或 decoder/training/generation/promotion
authorization。

实现绑定 commits 为 `4b75e4d67efc1dc90a01d1d33a6aa3c1c243c652` 与修复后的
`ebbd0d232c56ee45a69dea830911eff4f697ad47`。attempt-01 按 write-once 规则保留为 GateResult
evidence serialization harness failure，不消费；权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/`。
validation/audit/compact SHA-256 分别为
`5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137`、
`0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde`、
`5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940`；F32 manifest/payload
SHA-256 为 `1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`。

F32 artifact 仅包含从 S8 四个 tensor 的逐元素确定性 cast：scalar count=`684352`，raw
float32 bytes=`2737408`，serialized payload bytes=`2739448`；rank family=`[16]`，无 rank search、
refit、训练、backward 或 physical-forward modification。selection/confirmation readout 均在 CPU
执行，candidate/reference prediction residual、fresh-reload residual、state/no-update 与 runtime
independence gates 通过。audit `35/35` gates 通过，classification=
`frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified`，`integration_only=true`。

selection 的 max prediction difference / normalized L2 / aggregate MSE absolute-relative drift 为
`7.8160793304e-7` / `1.5589117128e-7` / `8.2576945282e-11` / `3.9186724995e-9`；confirmation
分别为 `7.8570211703e-7` / `1.5627624021e-7` / `2.8491427967e-11` /
`1.3386508868e-9`，均满足 elementwise、normalized-L2、absolute-MSE 与 relative-MSE gates；
confirmation bootstrap 95% CI 保持 `[0.03618517890572548,0.036830976605415344]`。治理 flags
保持 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_MPCDAR_R16_F32_NQ_result`，不得自动切换 default、训练、decoder、generation 或
promotion。Bridge refresh/status=`invalid`，原因是既有目标目录的
`.chatgpt-sources-manifest.json` 不合法；按 fail-closed 规则未删除、覆盖或上传该包。

结果为 `status=passed`、`classification=mean_preserving_centered_data_aware_readout_recovery_completed`，
audit `14/14` gates 通过。fit replay 使用 `N=1000`、float64 population moments；DMSA 的 mu/C
hash 均匹配，moment identity relative residual=`2.98546e-16`。C_fit PSD raw minimum eigenvalue
为 `3.23645e-09`，material/tiny negative count=`0/0`；A_fit spectrum 的 cumulative energy
at `1/2/4/8/16/32/64/128/256` 为 `0.118309/0.207018/0.355819/0.541212/0.715164/0.868603/
0.963351/0.994898/1.000000`，effective/stable rank=`37.184239/8.452439`。DA boundary gaps
使用 A_fit/G_fit 自有 spectrum，仅作 descriptive telemetry；raw LSDA gaps 只存在于 comparator。

rank-256 positive anchor first passed；V orthogonality max residual=`1.70464e-12`，A_fit/W
projector max residual=`4.53214e-14/5.68008e-11`，selection/confirmation full prediction identity
与 RMAL gates 均通过。随后 primary DA search 按 `[1,2,4,8,16,32,64,128]` 顺序实际评估
`1,2,4,8,16`：rank 1 仅 strict-template crossing 失败；rank 2/4 为 instance-sensitive 与
strict-template crossing 失败；rank 8 selection 通过但 confirmation strict-template crossing
失败；rank 16 selection/confirmation 均通过。因此
`first_confirmed_data_aware_candidate_rank=16`、`previous_registered_candidate=8`，没有重复
rank-256。rank-16 probe MSE selection/confirmation=`0.0210726835/0.0212836881`，relative
improvement=`0.660383/0.631703`。

每个实际 evaluated rank 同流记录 old LRRC raw (A)、mean-preserving raw (B) 与 mean-preserving
data-aware (C) MSE 及 A→B/B→C/A→C delta；matched control 不参与 stop rule 或 confirmation。
正式回放为 CMOS-only streaming/no-bank/no-target-fitting，state/no-update、finite、resource
gates 通过，peak CUDA=`2274099712` bytes、wall=`183.266 s`。flags 继续为
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是 `manual_review_MPCDAR_result`，
不得训练、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S6-MCFA：Detector Mean-Centered Functional Attribution（2026-09-02）

MCFA 消费冻结 RMAL `W=[256,40000]`、`b=[40000]` 及 LSDA、LRRC、DSFA、DMSA、
ZCSA/RWSA-ACTIVE 的已验证身份；真实消费者是项目负责人区分 panel mean response
`mu W` 与 centered instance response `(z-mu)W` 对 reconstruction functional MSE gain
的描述性贡献。拒绝决定是不得把 attribution 升级为 mean/centered dominance、
reconstruction sufficiency、training、generation 或 promotion。最终实现/config/review/
test/CLI clean commit 为 `069cfa4b3aeda3943d754d739daf6daeba2a54a6`；本轮改动文件为：
`src/ldm_project/v2/application/research/optical_manifold_smf_bogt_detector_mean_centered_functional_attribution.py`、
`configs/research/lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_v1.yaml`、
`configs/research/lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_review_v1.json`、
`tests/v2/test_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution.py`、
`src/ldm_project/v2/application/_cli_research_optical_manifold.py` 与
`src/ldm_project/v2/application/_cli_schema_research.py`；上游 evidence/probe/renderer/
checkpoint/OLS/detector/formal runtime 未修改。

attempt-01 为 validation-only failed harness，正式 replay 在 panel image runtime binding
处停止，validation SHA-256=`e27c273780387ceb36f1c63cb34746231356236d209ac122f61a2a23a35bb3bb`，
按 write-once 规则保留且不消费。权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/`；
validation/audit/compact SHA-256 分别为
`35ad479aa931cb2e605bb9b582306549512106a5c2dcd070bfde35a2e552e7f1`、
`f69664cf9da1d61e90742a09614c4535388402f9228e401d2451e3ebeeefdc0b`、
`e95c072bde8306dd3f7cb8d6c5af1a671a778cede8f84c285c450af854059494`；probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

结果为 `passed / detector_mean_centered_functional_attribution_completed`。fit/selection/
confirmation 分别使用 1000/1000/5000 独立 panel 样本，population denominator=`N`，
两遍 CMOS-only streaming replay 共 14000 sample visits；mean/centered output-energy
fractions 分别为 fit `0.3829179441/0.6170820559`、selection `0.4019555898/0.5980444102`、
confirmation `0.3704005497/0.6295994503`。对应 mean/centered/total functional gains 为
fit `0.0313323622/0.0504929026/0.0818252648`、selection
`0.0396463978/0.0489494817/0.0885958795`、confirmation
`0.0316138044/0.0458887261/0.0775025305`；full-probe MSE 为
`0.0044580647/0.0098129279/0.0101852893`。

三 panel 的 centered feature/dynamic mean、dynamic decomposition、energy identity、
MSE/gain identity、DSFA/DMSA/LRRC scalar cross-identity、finite、panel isolation、
state/no-update、streaming/no-bank 与 attribution boundary gates 全部通过；audit 为
15/15 gates，peak CUDA=`2274099712` bytes，wall=`183.266 s`。optimizer/backward/refit/
decoder/physical-forward modification/generation 均为 false/zero；
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported` 保持。最终核验为 MCFA 定向 `21 passed`、
相关回归 `179 passed, 1 warning`、完整 pytest `1310 passed, 12 deselected, 25 subtests passed`，
compileall、Ruff、git diff-check 通过。唯一下一动作是 `manual_review_MCFA_result`；不得训练、
decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S9V-FULL-PATH-VIS：Full-Path Optical Representation Visualization（2026-09-02）

S9V 是基于冻结 RMAL detector readout 的 descriptive-only 可视化 consumer，真实消费者是项目负责人对固定
confirmation 10-sample panel 的完整 optical path、重建方法族、fixed-class error detail 与 MPCDAR R16
F64/F32 数值显示进行人工审查；拒绝决定是不把图像显示解释为 reconstruction/generative manifold、性能、
bottleneck、training、decoder、generation 或 formal-promotion Gate。实现从预期 HEAD
`7da2d066d8a8f1b3dbcff7c94393c51d11d7bcb7` 执行，S9V closeout 已提交为
`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`，当前 worktree clean。

本轮新增/修改文件为：
`configs/research/lightgen_optical_manifold_smf_bogt_full_path_visualization_v1.yaml`、
`configs/research/lightgen_optical_manifold_smf_bogt_full_path_visualization_review_v1.json`、
`scripts/visualization/render_scaled_chen_full_pipeline.py` 与
`tests/v2/test_scaled_chen_visualization.py`。父级只读绑定为 S7
`H-LOMA-CBOGT-S7-MPCDAR_attempt-03`、S8 `H-LOMA-CBOGT-S8-MPCDAR-R16-IQ`、S9
`H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ`；S7/S8/S9 validation/audit/compact 及 S8/S9 artifact
哈希均由 manifest 记录并在运行前逐项核验。RMAL checkpoint/config 身份仍为已登记的
`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de` /
`c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f`。

正式回放使用 CUDA、固定样本索引 `{0:55006,1:55000,2:55001,3:55008,4:55004,5:55010,6:55012,7:55003,8:55002,9:55007}`，
无训练、refit、optimizer、backward、evaluator、prior sampling 或 scientific Gate。图包固定包含：full optical
path 两页、reconstruction family、detail/error classes `[0,2,3,5,8]`、MPCDAR R16 F64/F32；阶段与 alias
按合同保存，`modal_complex` 仅作 10×10 modal coefficient display，`pre_ols`/`post_ols` 不作为额外传播步。
幅度使用 panel-wide p99.5，phase 使用 raw `[-pi,pi]` 与每 stage/sample 的 1% mask，real SLM/CMOS readout
明确为 phase N/A，reconstruction/error display 均固定 `[0,1]`。manifest classification=
`descriptive_visualization_completed`，schema=`lightgen_optical_manifold_smf_bogt_full_path_visualization.v1`，
SHA-256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`；输出目录为
`outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01/`，26 files、
`16184241` bytes。manifest 的 forward consistency max residual=`0`，model/OLS state hashes unchanged。
描述性 panel MSE 为 physical=`0.06047516688704491`、raw-SVD R16=`0.08698886632919312`、MP-RAW R16=
`0.05387931689620018`、MPCDAR R16 F64/F32=`0.024362197145819664`、frozen full R256/RMAL=
`0.011573356576263905`；这些 scalar 仅服务本图包显示，不构成新 Gate。

S9V 定向可视化测试为 `9 passed`，父级 artifact/回归定向测试为 `48 passed`；Windows-native 默认全量 pytest
为 `1347 passed, 12 deselected, 25 subtests passed`，compileall、Ruff、git diff-check 均通过。代表性 PNG 已完成人工
视觉核验。治理 flags 保持 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_MPCDAR_R16_RUNTIME_PQ_implementation_contract`，
不得继续训练、decoder、candidate export、generation 或 promotion。Bridge 仍为 `invalid / cloud_sources_sync_blocked`，
未在本轮修复或上传。

## H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ：Frozen MPCDAR Rank-16 Float32 Runtime / Performance Characterization（2026-09-03）

S10 是冻结 S9 MPCDAR rank-16 float32 artifact 的 CPU-only、readout-only 工程 characterization。
真实消费者是项目负责人审查固定运行环境中的 fresh-process initialization、确定性 correctness、
latency、throughput 与 storage telemetry；拒绝决定是不把这些结果升级为 measured speedup、GPU/
硬件/production readiness、reconstruction sufficiency、decoder、training、generation 或 promotion。
S9V closeout commit=`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`，S10 starting HEAD=`8680c12be955df72906b955433e6daeabfdfec4a`；S10 implementation/fix commits
为 `dd1c38fa965e32deedd655f00f3cd974e1534cd8`、`488f12fbcf19e3663c7aa685bc4d4ad717db2a30`、
`d0655652cd86bdc9f3852a9bbf4faa45f59f069b`；父级 S9/S9V、runtime、模型、OLS、probe 与 detector
均未修改。

write-once 权威目录为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_runtime_performance_qualification_v1_attempt_02/`；
validation/audit/compact SHA-256 分别为
`8293e549b1ae48447e28be924406525aa03fa8843c0f1a71fa1ec21e3f5894c5`、
`7b519b3a75cdf59ecfe2678d46a8f74ae3084cb7b9de90c95763efecc0f2f4ee`、
`ded0959097c88a74031edc1c0aa925d1a95a2df7f1abc5c6d9ba6ca7abdc3173`。早先 attempt-01 在
namespace-isolation 修复前完成，按 write-once 保留但不作为当前权威；attempt-02 绑定修复后的
clean commit `d0655652cd86bdc9f3852a9bbf4faa45f59f069b`。父级 S9 manifest/payload SHA-256 为
`1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`；S9V manifest SHA-256 为
`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`。

父级 artifact contract 保持 rank=`16`、dtype=`torch.float32`、tensor shapes
`mu_fit[256]`、`c_mean[40000]`、`T16[256,16]`、`V16[40000,16]`，公式为
`centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`。固定 fixture 为 CPU dense finite
`float32[32,256]`，seed=`20260902`，SHA-256=`45ca137458856a7305ecf9500220c120aa5b4947e6f58d476b791e49b6337ab6`。
独立 direct-formula replay 与 repeated-call determinism 在 B=`1/8/32` 均通过，max absolute 与
normalized-L2 residual 均为 `0`。

正式环境记录为 Python `3.11.16`（`D:\ANACONDA\envs\optical-generative-models-win\python.exe`）、
PyTorch `2.5.1`、torch threads `8/8`、logical cores=`16`、CPU model=`AMD64 Family 25 Model 68
Stepping 1, AuthenticAMD`。任务预期 Windows=`10.0.26200.0`，实测=`10.0.26100`，因此
`windows_version_match=false`、`registered_match=false`，但 registered Python/Torch/Windows CPU
execution stack=`true`；physical cores/RAM=`not_recorded`。这是事实记录差异，不是 benchmark failure。

fresh-process initialization（非 true cold-disk claim）为 manifest parse/load=`0.2792 ms`、payload
load=`9.1288 ms`、consumer initialization=`20.8566 ms`、total artifact initialization=`30.286 ms`、
fresh-process wall=`3657.2196 ms`；filesystem cache flush=`false`，stage timers may overlap。
稳态计时固定 warmup=`50`、measured repetitions=`200`、`time.perf_counter_ns`、CPU eager synchronous，
quantile method=`linear: index=(n-1)*q`：B=1/8/32 batch mean latency=`0.5601775/0.8914525/3.277882 ms`，
per-sample mean=`0.5601775/0.1114315625/0.1024338125 ms`，throughput mean=
`1836.8588342/9338.7841702/9874.3015643 samples/s`。compact evidence 保存每批的 median/p95/p99/min/max
完整统计；未保存 raw latency samples 或 prediction bank。

storage telemetry 为 manifest=`3219` bytes、serialized artifact=`2739448` bytes、raw/runtime tensors
均为 `2737408` bytes；process-memory delta/peak/RSS=`not_recorded`。保留的只是 analytical operation
count：factorized=`644096 MAC/sample`、full-W=`10240000 MAC/sample`、ratio=`15.89825119236884`；
`measured_speedup_claim=false`。GPU、full-W runtime 与 end-to-end software pipeline 均为
`not_evaluated`。

结果为 `status=passed / frozen_rank16_float32_runtime_performance_characterized`，audit=`26/26`，
diagnostic-only；`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 保持不变。
S10 定向/布局/相关回归=`30 passed`，Windows-native 全套 pytest=`1357 passed, 12 deselected,
25 subtests passed`，compileall、Ruff、git diff-check 通过；唯一 warning 为既有 pytest cache ACL。
Bridge 仍为 `invalid / cloud_sources_sync_blocked`，未修复、未上传。唯一下一动作是
`manual_review_MPCDAR_R16_RUNTIME_PQ_result`；不得训练、decoder、generation 或 promotion。

## FINAL_MAINLINE_CLOSEOUT registry and primary runtime record

The following is the frozen read-only registry for the final mainline. Paths are relative to this
repository; hashes are SHA-256. Older attempts remain write-once history and are not substituted for
the authoritative attempt named here.

- RMAL (`H-LOMA-CBOGT-S4-RMAL`): authoritative directory
  `outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/`;
  validation/audit/compact/probe SHA-256
  `affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec` /
  `5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa` /
  `c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137` /
  `3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`; checkpoint SHA-256
  `ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`, config SHA-256
  `c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f`.
- S7 `attempt-03`: `outputs/20260902_v2/runs/lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/`; validation/audit/compact SHA-256
  `b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2` /
  `43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785` /
  `9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9`.
- S8 `attempt-03`: `outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/`; validation/audit/compact SHA-256
  `844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb` /
  `3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4` /
  `12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271`; artifact manifest/payload
  `e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45` /
  `e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e`.
- S9 `attempt-02`: `outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/`; validation/audit/compact SHA-256
  `5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137` /
  `0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde` /
  `5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940`; artifact manifest/payload
  `1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
  `480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`.
- S9V `attempt-01` manifest: `outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01/visualization_manifest.json`, SHA-256
  `ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`.
- S10 `attempt-02`: `outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_runtime_performance_qualification_v1_attempt_02/`; validation/audit/compact SHA-256
  `8293e549b1ae48447e28be924406525aa03fa8843c0f1a71fa1ec21e3f5894c5` /
  `7b519b3a75cdf59ecfe2678d46a8f74ae3084cb7b9de90c95763efecc0f2f4ee` /
  `ded0959097c88a74031edc1c0aa925d1a95a2df7f1abc5c6d9ba6ca7abdc3173`.

S10 primary record: Python `3.11.16`, PyTorch `2.5.1`, Windows observed `10.0.26100`,
`torch_num_threads=8`, `torch_num_interop_threads=8`, logical cores `16`, and formal CPU string
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; expected Windows `10.0.26200.0` did not
match. Batch latency mean/median/p95/p99 in ms for B=`1/8/32` were respectively
`0.5601775/0.53135/0.784155/0.943756`, `0.8914525/0.82855/1.25965/1.362798`, and
`3.277882/3.21995/4.026955/4.434511`; per-sample mean was
`0.5601775/0.1114315625/0.1024338125`, `1836.8588342/9338.7841702/9874.3015643` samples/s
mean throughput. Fresh-process artifact initialization was `0.2792/9.1288/20.8566/30.286 ms`
(manifest/payload/consumer/total), with `3657.2196 ms` process wall; it was not true cold-disk.
Analytical factorized/full-W MACs were `644096/10240000` per sample, ratio
`15.89825119236884`, not measured speedup. GPU, full-W runtime, end-to-end latency and process
memory were `not_evaluated`/`not_recorded` as applicable.

Bridge closeout: `refresh` stopped fail-closed with
`invalid Sources manifest: C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`;
`status=invalid` returned the same reason. `bridge_recovery_blocked=true`; the existing package was
not deleted or overwritten, and no automatic upload occurred.

## H-LOMA-CBOGT-FIBER-MMF-M6（2026-09-03，diagnostic-only）

本次 bounded implementation 在分支
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、起始/结束 HEAD
`daf9d7f5871eedc8950833e90ca2ff95d3ad56e6` 上完成；起始工作树 clean，结束工作树只包含
本 operator、导出和专项测试的实现增量，以及本状态登记和迁移附录。SMF、FMF、formal
IdentityOLS、MPCDAR、S7/S8/S9/S9V/S10 和既有 outputs 均未修改。

新增 `NestedMultimodeFiberArraySpec` / `NestedMultimodeFiberArrayOLS`，固定为
`synthetic_nested_mmf_m6_second_order_identity_transport`、`diagnostic_only=true` 的合成嵌套
MMF 诊断 operator。它复用现有 FMF 的 200-mode prefix，再按 degree-2 candidate block
追加 `ETA1/XI2/XIETA/ETA2`，mode bank 为 `[600,100,100]`、transport 为
`[600,600]` complex64 identity；registered buffers 的静态大小为两个各 `48,000,000`
bytes 的 mode bank 加 `2,880,000` bytes transport。无 fiber length、beta、dispersion、loss、
random/learned coupling 或 propagation。

验证结果：M6 专项 `9 passed`；既有 SMF `4 passed`；既有 FMF M2 `8 passed`。SMF/FM F
回归使用当前 Windows Conda 环境；FMF 文件测试为解决默认 `tmp_path` 的 Windows ACL
问题显式指定了任务临时根，未改变测试或生产代码。changed modules/tests 的 compileall、
Ruff 和 `git diff --check` 均通过。该结果只消费于人工审查和拒绝后续 VIS1/训练/晋升的
边界决策，不产生 experiment output 或正式 evidence。

Gate=`synthetic_mmf_m6_operator_implemented_and_validated`。下一动作唯一为
`manual_review_synthetic_mmf_m6_implementation`；`mainline_state=closed_at_current_evidence_boundary`、
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`
保持不变。该 operator 不代表 real MMF propagation、physical guided-mode count、hardware
MMF、real capacity 或 generation readiness。

## H-LOMA-CBOGT-FIBER-VIS1（2026-09-03，descriptive-only）

本轮在分支
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting HEAD
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7` 上执行已批准的 SMF/FMF/合成 MMF-M6 frozen
horizontal visualization。实施前工作树 clean；本轮之后只保留 renderer、VIS1 契约测试以及本状态/迁移登记的预期增量，
没有额外用户修改。S7/S8/S9/S9V/S10、MPCDAR、formal runtime、checkpoint、训练代码与既有 outputs 均未修改。

S9V full-path authority、RMAL、固定 step-5000 checkpoint 与 confirmation panel 均在运行前重新核验；checkpoint SHA256 为
`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`，RMAL config SHA256 为
`c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f`。三臂只改变 frozen modal OLS：SMF=`100`、
FMF=`200`、MMF-M6=`600` coefficients；同一 encoder replay 的同一 `pre_ols` tensor 通过三臂，decoder/relay/CMOS 与
现有 frozen MPCDAR R16 F32 不变。MMF-M6 的 `ETA1/XI2/XIETA/ETA2` 在图和 metadata 中均标为 synthetic mode families，
不作 physical LP11b/LP21/LP02、guided-mode 或 real-MMF propagation claim。

上一轮权威 write-once 输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_06/`，包含
`fiber_contracts.json`、`comparison_manifest.json`、`comparison_metrics.json`、`panel.pt`、Figure A/B 的 PNG/TIFF/PDF/SVG
及 alignment JSON/overlay。Figure A 是固定 digits 0–9 的七列 Target、SMF CMOS/MPCDAR、FMF CMOS/MPCDAR、MMF-M6
CMOS/MPCDAR；Figure B 保存 shared pre-OLS amplitude/phase、三臂 modal power、post-fiber amplitude/phase 和
`post_fiber = decoder_input` alias。`panel.pt` 只保存 captured coefficients/family power/fields/z，不保存完整 600-mode bank。
attempt-01 至 attempt-05 按 write-once 保留但不消费；attempt-06 是通过最终 FMF family-axis 与最终版式 QA 的当前候选。

attempt-06 描述性指标的 MPCDAR R16 F32 mean MSE 为 SMF=`0.024362197145819664`、FMF=`0.033275894820690155`、
MMF-M6=`0.04628327488899231`；detector z 相对 SMF 的 mean L2 delta 为 FMF=`0.22263486683368683`、
MMF-M6=`0.4845358431339264`，cosine-like similarity 为 FMF=`0.9905906915664673`、MMF-M6=`0.9773411750793457`。
这些值只描述固定下游 distribution mismatch，不支持“fiber worse/superior”结论。

固定 CUDA 10-sample run 为 descriptive telemetry：elapsed=`2.2810000000008586 s`、peak CUDA allocation=
`1748934144` bytes；model learned-state digest before/after 均为
`fe8bd9ede571917cf622a89cff36c3e6236c416774f3783e95c5ee9800fedef0`，all parameters frozen，optimizer/backward/evaluator/
refit/rank search 均未执行。最终 targeted tests=`37 passed`（VIS1 7、既有 S9V 9、SMF 4、FMF M2 8、MMF-M6 9）；
compileall、Ruff、`git diff --check` 通过。Nature Figure QA 为两张 alignment report PASS、两张 PDF collision PASS、
PDF text minimum=`7.0/5.4 pt` 且无低于 5 pt 文本、source validator=`20 PASS/0 FAIL/1 intentional WARN`；该 WARN 是
固定单 panel 不提供统计不确定性条，避免伪造 inferential uncertainty。两张图已完成人工 final-size panel inspection。

Gate=`descriptive_fiber_visualization_completed`。治理 flags 仍为
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`closed_task_relevant_bottleneck_not_supported`。唯一下一动作固定为
`IMPLEMENT_H_LOMA_CBOGT_FIBER_VIS1`。Bridge 仍为 `invalid / cloud_sources_sync_blocked`，本轮未刷新、未修复、未上传。

## H-LOMA-CBOGT-FIBER-VIS2（2026-09-03，Windows-native，diagnostic-only）

本轮完成已批准的 canonical clear 0–9 full-pipeline SMF/FMF/MMF-M6 qualitative visualization。
分支为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7`。执行前已核验并保留上一轮 VIS1 的四个预期改动，
没有额外用户修改；执行后 worktree 只包含两份状态文档、VIS1 renderer/test 与 VIS2 renderer/test
六个预期路径。

Stage A 只读取同一 MNIST train split 的 raw images/labels，在 S9V confirmation train interval
`[55000,60000)` 中按 dataset index ascending、每 digit 最多 32 个候选，未读取任何 fiber、CMOS、
reconstruction 或 MPCDAR 输出。人工复核通过并冻结 clear 0–9：
`{0:55297,1:55310,2:55040,3:55054,4:55053,5:55322,6:55066,7:55052,8:55165,9:55120}`。
`clear_panel_manifest.json` SHA-256 为
`dbf9aa6741d29e264543f387edf7a4125893ec282c7a3897a57682d2e90388b9`。

当前权威 write-once 输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_08/`；
要求的 15 个 artifact 全部存在，output file count=`58`，写入 visualization manifest 前总字节数
`396579365`。包含 clear contact/approval、Figure 0、full-pipeline index 与 0–4/5–9 split pages、
fiber close-up、CMOS/reconstruction、SMF/FMF/MMF-M6 full paths、comparison/visualization manifests
与 panel payload。comparison manifest SHA-256=`5a1645f9beb67bb55e8ecaf7141acb2f81ed8586f9d808136c901d47a0cc3d10`；
visualization manifest SHA-256=`d2456ded7574a33b1941169e6efdac46384e4205438aeeb18cc0850d3c1c7f29`。
attempt-01 至 attempt-07 按 write-once 保留但不消费；attempt-08 是排版碰撞修正后的上一轮权威，随后由下方
panel-only compact attempt-09 替代。

三条 frozen arms 为 native SMF=`100`、existing FMF M2=`200`、synthetic nested MMF-M6=`600`。
raw input → SLM command → SLM output → encoder input → encoder layer 1 → shared pre-OLS 后分支，
同一 frontend replay 只执行一次，并由三臂复用同一个 pre-OLS tensor；post-fiber 是 decoder input alias，
后续为 decoder layer 1/2、CMOS incident complex field、CMOS intensity `|E|^2`、z[256] 与冻结 MPCDAR R16 F32
reconstruction。幅度采用同 stage joint p99.5，phase 为 raw `[-pi,pi]` 加 display-only 1% mask，target/reconstruction
固定 `[0,1]`，CMOS intensity 使用共同尺度。MMF-M6 的 `ETA1/XI2/XIETA/ETA2` 仅标为 synthetic families，
不作 physical LP-mode、guided-mode、real-MMF propagation 或 capacity claim；panel 不保存完整 600-mode bank。

固定 CUDA replay 为 10 samples，elapsed=`2.7030000000013388 s`、peak CUDA allocation=`3095443968` bytes，
learned model state unchanged=`true`。治理拒绝决定保持不变：
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`closed_task_relevant_bottleneck_not_supported`。VIS1 attempt-06 authority
`not_overwritten=true`、`not_recomputed=true`、`not_modified=true`。

验证结果：VIS2 + VIS1 + existing S9V/SMF/FMF-M2/MMF-M6 targeted suite=`44 passed`（唯一 warning 为既有
pytest cache ACL）；changed Python compileall、Ruff、`git diff --check` 均通过。Nature Figure QA 为 10/10
alignment reports PASS；10 个 PDF 均无低于 5 pt 文本，最低检测值=`5.0 pt`；collision audit 无 FAIL，4 个含
modal family 标签的图共 66 个 intentional text-to-image/fill edge WARN，已完成人工 final-size inspection。
source validator=`19 PASS/0 FAIL/2 intentional WARN`，WARN 为 diagnostic full-plate 宽度与固定描述性样本不编码
inferential uncertainty。Gate=`canonical_clear_full_pipeline_visualization_completed`；唯一下一动作固定为
`manual_review_VIS2_figures`，不授权训练、decoder、candidate export、generation 或 formal promotion。

## H-LOMA-CBOGT-FIBER-VIS2 compact rerender（2026-09-03，Windows-native，diagnostic-only）

基于已批准且未修改的 VIS2 attempt-08 `panel.pt` 完成 panel-only compact rerender；未重新 CUDA replay，未改变任何
optical data、shared frontend、joint p99.5 scales、phase mask、三臂 contracts 或 VIS1 attempt-06 authority。分支为
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7`；执行前后均未发现额外用户修改。

当前 VIS2 输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_09/`。source panel
SHA-256=`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；comparison manifest SHA-256=
`509aec7df5d9cbee2fe80d57658e9efa2cdd35640ce587d88580babd0232c522`（文件值），visualization manifest
SHA-256=`c5b1001527c31c54ad51dcca319d3c8f612f9a3c99a44651067ddb5223a4654c`。输出共 `115` 个文件、
`70354448` bytes：主图 `21` 个 bundle，PNG/PDF/SVG 分别 `22/22/44`，alignment JSON `22`，无 TIFF、隐藏文件或
`.tmp` 残留；panel.pt 保留在最终版中。详细页使用 `compact_arm_separated_v2`，SMF/FMF/MMF-M6 分别蓝/绿/橙，
每页 digits `0–4` 或 `5–9`，frontend/fiber_decoder/detector_reconstruction 分组，最多 6 个阶段列，空间场等比例，
modal family×port 明确作为非空间矩阵显示，MMF-M6 `ETA1/XI2/XIETA/ETA2` 继续显式标注 synthetic。

panel loader 已实际重载 source 并验证 schema、task id、APPROVED review、clear-manifest hash、source comparison/
visualization bindings、panel SHA-256 与三臂 payload；回归测试覆盖 review/hash/payload-schema mismatch rejection。VIS2
targeted tests=`13 passed`；alignment=`22/22 PASS`，PDF collision=`22/22 PASS`（0 WARN/0 FAIL），PDF text=`22/22`
无低于 `5.0 pt` 文本，minimum=`5.0 pt`。Nature source validator=`17 PASS/0 FAIL/4 intentional WARN`：PNG-only
submission raster、300 dpi floor、Stage-A 候选 contact sheet 的非标准宽度及固定样本无 inferential uncertainty，均为本轮
明确的诊断/输出约束；compileall、Ruff、`git diff --check` 在最终复核中通过。

按批准清理精确删除 VIS2 attempt-01–attempt-08 与 VIS1 attempt-01–attempt-05，共释放 `2942680973` bytes
（约 `2.7406 GiB`）；最终仅保留 VIS1
`lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_06` 与 VIS2 attempt-09，合计 `297510614` bytes，
无隐藏文件、无 `.tmp` 残留。治理仍为 `mainline_state=closed_at_current_evidence_boundary`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。Gate=
`canonical_clear_full_pipeline_visualization_completed`；唯一下一动作固定为 `manual_review_VIS2_figures`。

## H-LOMA-GEN-G1-SA prior support accessibility observer（2026-09-03，Windows-native，diagnostic-only）

在 VIS2 完成后，按已批准的 G1-SA 合同对当前 Scaled-Chen step-5000 做了一次 no-update support-accessibility
observer。执行前 branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、HEAD=
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`，worktree clean；未修改 VIS1/VIS2、generation code、formal runtime、bridge
或任何 checkpoint。MRA attempt-05 的 config/review/audit/compact/panel、step-5000 checkpoint 与 calibration 均按
固定 SHA-256 重新核验；MRA panel 未加载为新样本选择依据。

观测使用精确 100D `z_j=abs(c_j)^2/s_j`：`S_train` 为注册 train split `[0,55000)` 的全部 `55000` 个 encoded
support，`Q_holdout` 为注册 horizon held-out `[55000,60000)` 的 `5000` 个 query，unconditional truncated-normal
prior 也固定为 `5000` 个 query、seed=`1237`、mean=`1.0`、variance=`0.25`、lower=`0`。距离为 exact chunked
normalized Euclidean，仅除以 `sqrt(100)`；没有 PCA、whitening、learned metric、ANN、class conditioning 或 subset
tuning。仅执行 SLM → front → encoder → pre-OLS coefficient → normalized power prefix，未进入 decoder、relay-after-
pre-OLS、CMOS、MPCDAR、prior decode 或 evaluator。

当前 QA 通过的结果目录为
`outputs/20260903_v2/runs/lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_04/`。attempt-01 是同一
次 CUDA measurement 的原始 artifact，attempt-02/03 为保持 write-once 的 render-only QA 迭代，attempt-04 从同一个
`support_distances.pt` 无 CUDA 重绘并作为当前 figures/manifest；没有重新运行 fiber replay。attempt-04 共 `12` 个文件、
`294949` bytes；`support_distances.pt` SHA-256=`05002920d1b73ab8dd1243ce93f44659ba95ad5d4eb35e3f4e737c0f9d8b2c1b`，
`observer_manifest.json` SHA-256=`257e6b786714b6f0a76fe93e4bba6ddf2f6bc3b691d0ef3f582ebd7c7521f239`。输出只有 distance vectors、
metrics/report、PNG/PDF/SVG 两组图和 single-panel alignment records，不含完整 dataset images、TIFF、隐藏文件或
`.tmp`。

主统计为 holdout `H50=0.0006687190`、`H95=0.0011482780`，prior `P50=1.1241731`、`P95=1.2017229`；
`R50=1681.0843`，95% bootstrap CI=`[1657.9121,1705.0805]`；`R95=1046.5435`，95% bootstrap CI=
`[1019.9279,1063.9357]`，bootstrap replicates=`2000`、seed=`1236`。描述性 `prior_exceedance_H95=1.0` 不参与
主 gate。两个 ratio CI 下界均大于 1，因此 Gate=`P4_SUPPORTED`：当前 registered unconditional prior 与真实 encoded
support 显著不对齐；这不是 decoder accessibility、generation quality 或 training result。

Replay anchor 与资源：step-5000 normalized-z mean=`0.0014779161`，与 MRA confirmation anchor 一致；anchor batch
mean=`0.0015653975`；prior empirical mean=`1.0277389`。device=`cuda`，latent dtype=`torch.float32`，distance
dtype=`torch.float64`，runtime=`372.0281 s`，peak CUDA allocation=`1574747648` bytes，启动前 free CUDA=
`5368709120` bytes。learned-state digest before/after 均为
`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`；state unchanged，optimizer/backward/refit/
checkpoint/evaluator 均为 false/0。

验证：G1 targeted tests=`8 passed`；changed Python compileall、Ruff、`git diff --check` 通过。两张 PDF 均通过 5 pt
text audit（minimum=`8.0 pt` 与 `5.6 pt`），collision audit=`2/2 PASS`（0 WARN/0 FAIL），single-panel alignment
records=`2/2 NOT_APPLICABLE_SINGLE_PANEL`，PNG 代表页面已实际查看；summary 使用 log y 轴以同时显示约三数量级的
holdout/prior 距离。治理继续为 `mainline_state=closed_at_current_evidence_boundary`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。唯一下一动作：
`PLAN_GENERATIVE_TRAINING_FOR_SUPPORT_ALIGNMENT`（仅规划，不授权执行训练）。

## H-LOMA-CBOGT-FIBER-VIS2 attempt-10（2026-09-03，Windows-native，diagnostic-only）

基于未修改的 VIS2 attempt-09 `panel.pt` 完成新的 panel-only 分层重绘；未重新 CUDA replay，未修改任何光学 tensor、
display scale、shared frontend、phase mask、三臂 contract 或 VIS1 attempt-06。attempt-09 保留为不可变审计源，
attempt-10 为当前可审查结果。branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，
starting/ending HEAD 均为 `4d2933cb5b49302dd2114d49b56c905ef9d10ca7`，未创建 Git commit，也未发现额外用户修改。

当前输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_10/`。
它按 `digit_visualization/`、`pipeline_overview/`、`fiber_information/{SMF,FMF,MMF-M6}/` 和 `metadata_qa/`
分层；主图共 21 个 bundle，PNG/PDF/SVG=`22/22/44`，alignment JSON=`22`，总文件数=`115`、总大小=`70212115`
bytes（visualization manifest 写入前=`70181140` bytes）。无 TIFF、隐藏文件或 `.tmp` 残留。每个详细页仅覆盖
digits `0–4` 或 `5–9`，最多 6 个阶段列；SMF/FMF/MMF-M6 颜色分别为蓝/绿/橙。

方向策略已固化到 manifest：MNIST image-space 使用 `origin=upper` 且 display rotation=`0`；光学空间场使用显式
`origin=lower`，保留既有按 stage 的 display-only 180° mapping；modal family×port 使用显式非空间矩阵策略；
所有 digit/index/amplitude/phase 行标签 rotation=`0`。attempt-10 panel SHA-256 与 attempt-09 完全一致：
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`。comparison manifest SHA-256=
`d4229e4f966f911fee510ed4bcfa385eb46821ecf84289c0e6b7a745f679fdd9`，visualization manifest SHA-256=
`8dcc62732b213c42cd12539b6f9329310ae2ec252113261154ba5f9fe2ed8648`，manifest 显式记录
`supersedes=attempt_09` 和 source manifest/panel binding。

回归与 QA：VIS2 renderer tests=`19 passed`；VIS2/VIS1/S9V/SMF/FMF-M2/MMF-M6 targeted suite=`64 passed`，唯一
warning 为既有 pytest cache ACL；changed Python compileall、Ruff、`git diff --check` 通过。全部 `22/22` alignment
reports PASS；`22/22` PDF collision audit PASS（0 WARN/0 FAIL）；`22/22` PDF text audit 无低于 `5.0 pt` 字符，
minimum=`5.0 pt`；Nature source validator=`17 PASS/0 FAIL/4 intentional WARN`，WARN 为 PNG-only raster、300 dpi
floor、Stage-A 候选 contact sheet 非标准宽度及固定样本不编码 inferential uncertainty。代表性的总览、SMF、FMF、
MMF-M6 页面已完成人工 final-size inspection。

治理仍为 `mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。VIS1 authority 保持 unchanged；
本轮未运行 CUDA、训练、refit、rank search、formal runtime 或 bridge 操作。Gate=
`canonical_clear_full_pipeline_visualization_completed`；唯一下一动作固定为 `manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 three-photo matched-input comparison attempt-11（2026-09-03，Windows-native，diagnostic-only）

针对 attempt-10 难以阅读的问题，基于未修改的 attempt-10 `metadata_qa/panel.pt` 完成新的 panel-only 三页重绘；未重新
CUDA replay，未改变任何 optical tensor、display scale、shared frontend、phase mask、三臂 contract 或 VIS1
attempt-06。attempt-10 保留为不可覆盖 source，attempt-11 写入独立目录。执行前后 branch 均为
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`；当前 G1 prior-support 的三个未提交路径及两份状态文档均作为受保护基线，
未发现额外 dirty path。

输出目录为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_11/`。
新 layout schema=`same_digit_three_arm_comparison_v1`，三张主图严格按标准数字分为 `0–3`、`4–6`、`7–9`；每行固定为
`Standard digit input | SMF output | FMF output | MMF-M6 output`，同一行复用同一个 panel position。标准输入来自
`images_28x28`，三臂输出唯一来自 `mpcdar_rank16_float32`；图中只显示 `digit 0` 等清晰标签，dataset index 保留在
`matched_input_comparison` manifest。SMF/FMF/MMF-M6 使用蓝/绿/橙边框，空间轴均为等比例显示。

attempt-11 共 `25` 个文件、`38798939` bytes：三张 PNG/PDF/SVG 主图各一份、三张 alignment JSON 与三张 alignment SVG，
以及 `metadata_qa/` 下复制的 approved clear-panel artifacts、panel、comparison manifest 和 root visualization manifest。
无 TIFF、隐藏文件或 `.tmp`。source panel SHA-256 与批准值一致：
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；comparison manifest SHA-256 与 visualization
manifest SHA-256 分别为
`e9b3c329bd3b30449cccf618d54d3b9b22c3edd04e46adb2057597c0ca4a3a99` 与
`2b457baa83b85620c9ea9374bf5b6bc47389fadcf94991c95064ab325ebc07b`；manifest 明确记录 `source_manifest_binding`、`render_only_from_panel=true`、
`cuda_replay=false`、`optical_data_changed=false` 与 `supersedes=attempt_10`。

QA：VIS2 renderer targeted=`23 passed`；Windows-native compileall、Ruff、`git diff --check` 通过。三页 alignment=`3/3 PASS`
（alignment canvas=`504 pt`，每页 4 列）；PDF text audit=`3/3`，最低字号=`5.4 pt`；collision audit=`3/3 PASS`（0 WARN/0 FAIL）。
实际 PDF tight-bbox 宽度为 `413.4/430.5 pt`，均小于 `7.2 in`；PNG 尺寸为
`1722×1399`、`1793×1399`、`1793×1399`，三张代表页面已实际查看。Nature source validator 非 strict 为 `18 PASS/0 FAIL/3 WARN`；
warning 仅来自旧 renderer 的 300 dpi、宽度启发式与固定样本 uncertainty 检查，不影响本次三页无 TIFF 输出。

治理继续为 `mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。本轮未运行 CUDA、训练、refit、rank search、
formal runtime 或 bridge；Gate=`canonical_clear_full_pipeline_visualization_completed`。唯一下一动作固定为
`manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 reference-style process-flow rerender attempt-14（2026-09-03，Windows-native，diagnostic-only）

针对 attempt-11 仅展示最终输出、无法表达完整链路的问题，复用未修改的 attempt-10
`metadata_qa/panel.pt`，新增并执行 `--render-process-flow-from-panel` panel-only 入口。未重新 CUDA
replay，未改变任何 optical tensor、display scale、shared frontend、phase mask、三臂 contract 或 VIS1
attempt-06 authority；attempt-10/11 保持不可覆盖，attempt-12/13 作为未通过严格 collision QA 的 write-once
布局迭代保留。当前候选输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_14/`。

新 layout schema=`same_digit_three_arm_process_flow_v1`，三张图按标准数字固定为 `0–3`、`4–6`、`7–9`。
每页是参考图式的完整流程列：`Input → SLM command → SLM output → Encoder in → Encoder L1 →
Pre-OLS → Modal → Post-fiber → Decoder L1 → Decoder L2 → CMOS field → CMOS intensity → MPCDAR R16`；
每个数字在 SMF、FMF、MMF-M6 三条上下对齐的 lane 中各有相邻 amplitude/phase 行。同一 panel position
保持同一输入和 dataset index，shared frontend 在三条 lane 中重复显示以便横向对照；SMF/FMF/MMF-M6 使用
蓝/绿/橙边框与行标签区分。Modal 为明确的 family×port 非空间矩阵，MMF-M6 的 `ETA1* / XI2* /
XIETA* / ETA2*` 保持 synthetic 标识，脚注明确 `*` 含义；不作 fiber ranking、physical MMF、capacity、
reconstruction-manifold、generation、training 或 promotion 结论。

attempt-14 共 `25` 个文件、`56214710` bytes：3 个 PNG、3 个 PDF、3 个主 SVG、3 个 alignment JSON、3 个
alignment SVG，以及 `metadata_qa/` 下 9 个 approved source/manifest artifacts。无 TIFF、隐藏文件或
`.tmp`；PNG 尺寸为 `3363×4204`、`3366×3303`、`3366×3303`，PDF 均为单页。source panel SHA-256 保持批准值
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；source comparison/visualization
manifest SHA-256 分别为 `d4229e4f966f911fee510ed4bcfa385eb46821ecf84289c0e6b7a745f679fdd9` 与
`8dcc62732b213c42cd12539b6f9329310ae2ec252113261154ba5f9fe2ed8648`。attempt-14 comparison/visualization
manifest SHA-256 分别为 `71ed3ccd75e7ee2748fd30ba6bc4f60d9fabfe7f1bcd6ec71d22619144e08f1c` 与
`705ef3d7e041219dac3090ede56701d5f41767860a985ef87c5e5dfb481ce1f5`，并记录
`render_only_from_panel=true`、`cuda_replay=false`、`optical_data_changed=false`、`supersedes=attempt_11`。

QA：Windows-native relevant suite=`51 passed`；compileall、Ruff、`git diff --check` 通过。3/3 alignment
audit=`PASS`（0 fail/0 warn），3/3 PDF text audit 无低于 `5.0 pt` 文本，3/3 strict collision audit=`PASS`
（0 fail/0 warn）；三张 PNG 已实际查看。Nature source validator=`18 PASS/0 FAIL/3 WARN`，WARN 仅为既有
renderer 的 300 dpi、宽度启发式和固定诊断样本 uncertainty 启发式，不改变本次实际输出的无 TIFF 与严格图形
QA 结论。

执行前后 branch 均为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD
均为 `d07fdaa17aa91d9f3e5498e9b1f10d91db766777`。worktree 仍有且只有受保护的 G1 prior-support config/
observer/test、两份状态文档、当前 VIS2 renderer/test；未发现额外用户修改，未创建 Git commit。治理保持
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。本轮未运行 CUDA、训练、refit、
 rank search、formal runtime 或 bridge；唯一下一动作固定为 `manual_review_VIS2_figures`。


## H-LOMA-GEN-PCR-PSD：Repaired-Coordinate Prior Shape Decomposition Observer（2026-09-04，Windows-native，diagnostic-only）

本轮只执行 `IMPLEMENT_PRIOR_SHAPE_DECOMPOSITION_OBSERVER`。初始状态为 clean，branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD=`02fb1f636aa17e0e2f7f01a3fa70275b8f19dbdf`；实现、config、tests 及本状态记录为本轮新增 dirty paths，未创建 Git commit。没有训练、optimizer、backward、refit、checkpoint creation、prior modification/fitting、G1-SA support-bank replay、prior decoder、generation、CMOS、MPCDAR、FID、classifier、rank search 或 architecture search。

观测严格复用 RCR-A selection/confirmation panel、dataset identity、MNIST preprocessing、100-port OLS、mode bank 与 repaired coordinate；没有重新选择 panel，也没有扩大到 G1 55k support。冻结 step-5000 checkpoint SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`；R2 calibration 为 `scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`，payload SHA-256=`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`；RCR-A manifest SHA-256=`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`，RCR-A metrics SHA-256=`5d2cc5c0c7b6e80f80d4f07239bb737f87327860381c0bf151cbaa686001b612`。RCR-A parent panel SHA-256=`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`。

registered prior sanity 通过：base Normal=`N(1.0, 0.25)`、lower=`0`、factorized/unconditional；exact truncated moments 为 mean=`1.0276239313`、variance=`0.2216129871`、skew=`0.2209357048`，q01/q50/q95/q99=`0.0775265/1.0142585/1.8279922/2.1674849`。固定 sampler sanity sample count=`5000`、seed=`260903`，KL identity 为零。正式观测使用 selection=`1000`、confirmation=`5000`；matched-size CPU null 使用 seeds=`260904/260905`、replicates=`128`、central interval=`99%`，不保存 replica samples。

selection 的 per-port mean heterogeneity std/range=`0.7455607/3.5774960`；within/between/total pooled variance=`1.5498618/0.5558607/2.1057225`，between fraction=`0.2639763`。confirmation 对应为 std/range=`0.6458730/2.9762450`，within/between/total=`1.4546335/0.4171519/1.8717854`，between fraction=`0.2228631`。per-port variance ratio median 为 selection/confirmation=`2.8765427/2.6247470`，范围最大值为 `74.5849404/57.8657887`。两 panel 的 covariance decomposition identity residual 均为数值零。

cross-port dependence 的 mean absolute off-diagonal correlation 为 selection/confirmation=`0.1600160/0.1617629`，p95=`0.4607387/0.4692335`，max=`0.8836253/0.8846894`，off-diagonal covariance fraction=`0.7362989/0.7411037`。prior-null central-99% reference 的 primary percentile position 两 panel 均为 `100%`，但该 null 只作为 finite-sample diagnostic calibration，不把超出 percentile 自动解释为 prior contract bug。

prior-defined tail masses（thresholds 固定为 prior q01/q99，另记录 q95 exceedance）selection 的 lower/upper/central/q95=`0.156820/0.134430/0.708750/0.168730`，confirmation=`0.168390/0.114900/0.716710/0.147722`；两 panel 均相对 matched-size prior null 显示同一方向的 tail/quantile deviation。最终 decomposition 为 `P-S3_SUPPORTED`、`P-S4_SUPPORTED`、`P-S5_SUPPORTED`，三者 replication status 均为 `REPLICATED_DIRECTION`。

最终分类为 `PCR1_PRIOR_CONTRACT_INTERNALLY_COHERENT`：这表示 registered factorized truncated-normal target 在 sampler/config/KL/coordinate 语义上内部一致，同时 frozen step-5000 `mse_only` q(z) 通过 port-wise marginal heterogeneity、cross-port dependence 和 marginal tail mismatch 偏离该 target；不表示 q 已匹配 p，不授权 per-port prior fitting、prior 修改或训练。

权威新输出为 `outputs/20260904_v2/runs/lightgen_scaled_chen_prior_shape_decomposition_v1_attempt_01/`，manifest SHA-256=`228679fda073b2e4e1e46c931b743b490930dc895313e5c201283d244e22d97a`，metrics SHA-256=`4ea665c5a021a09c9ce2d44adfc31ed0cdb593de30b0ed3dbb3b10986bfbe9d2`，compact report SHA-256=`cbd0ceda6a878bcf295ef0252936edf35a6525937b76472610f2c9e384d85f94`；8 个 artifact 共 `639294` bytes，包含完整 per-port moments/quantiles、covariance、correlation、prior-null summary 与 compact report，无隐藏文件、`.tmp` 或 TIFF。learned-state digest before/after 均为 `cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`，state unchanged；optimizer/backward/refit/checkpoint/prior_fitted/decoder/CMOS/MPCDAR/training 全部 false。

新增 PSD 定向测试=`7 passed`；PSD+RCR-A+Scaled-Chen BOGT 相关回归=`36 passed`；compileall、Ruff、`git diff --check` 均通过。单次 bounded no-update CUDA prefix extraction runtime=`48.7801 s`，peak allocation=`1599782400` bytes；CPU prior-null runtime=`6.9309 s`。治理继续保持 `mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`。唯一下一动作固定为 `IMPLEMENT_REPAIRED_COORDINATE_G1_SA_REPLAY`。

## H-LOMA-GEN-RCR-A：Repaired-Coordinate MRA Replay（2026-09-04，Stage A，diagnostic-only）

在初始 clean state（branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting
HEAD=`ef1b336984b701f1905a06fc868c68c4b4892057`）完成 RCR-A harness/config/tests 独立提交后，使用
冻结的 step-5000 checkpoint（SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`）和
不可变 parent MRA attempt-05 panel，执行 repaired-coordinate Stage A latent-prefix CUDA replay。只改变
calibration identity：旧 `scaled_chen_bogt_calibration.v1`（epsilon-dominated）对比新
`scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`，新 payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`，R2 manifest SHA-256=
`0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6`。calibration indices、dataset、mode bank、
zero-phase reference 和 pre-OLS plane identity 均通过显式绑定；不允许 v1 fallback。

Stage A 只执行 `SLM → front → encoder layer 1/2 → pre-OLS → complex coefficients`，未进入 decoder、relay、
CMOS、MPCDAR、evaluator、prior generation、optimizer、backward、training 或 refit。selection 的 repaired
`z` mean/variance=`1.0060573 / 1.5498618`，confirmation 为 `0.9250944 / 1.4546335`；per-port mean range
分别为 `[0.0882697, 0.7385800, 3.6657655]` 与 `[0.0805473, 0.7471590, 3.0567923]`，per-port variance range
分别为 `[0.0073358, 0.6374792, 16.5289917]` 与 `[0.0066992, 0.5816780, 12.8238106]`。Gaussian KL 为
`3.2833059 / 2.8498273`，numerical truncated-normal KL 为 `3.2492599 / 2.7428405`，均较旧坐标降低；
power rank 为 `7.2165284 / 7.3683505`，为 coordinate-dependent 描述，不作为新 material-improvement Gate。
complex rank 保持精确旧 MRA 值 `12.2361650 / 12.8739576`，raw coefficient/power 由同一前缀复用；checkpoint
hash 和 learned-state digest 均保持不变（state digest=`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`）。

新的完整 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_02/`，manifest SHA-256=
`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`，含 4 个 JSON 文件、总计 `53575` bytes，
无 TIFF、隐藏文件或 `.tmp`。attempt-01 因首次实现的 manifest 发布顺序错误仅保留 3 个中间 JSON，不作为
authority，未被删除或覆盖；attempt-02 为唯一完整审计结果。

两组 panel 的 mean 均在 descriptive tolerance `±0.10` 内，但 variance 分别为注册 prior variance `0.25` 的
`6.1994×` 与 `5.8185×`；两组 Gaussian-KL 分解均由 variance-shape term 主导，且 numerical KL=`3.2493/2.7428`。
因此 Gate=`MRA_PRIOR_CONTRACT_REVIEW_REQUIRED`，dominant assumption 为
`unconditional_truncated_normal_variance=0.25_with_lower=0.0`，下一步为 `PLAN_PRIOR_SHAPE_CONTRACT_REVIEW`；
这不等同于 replay invalid，也不自动启动 Stage B G1-SA。RCR-A targeted/直接依赖回归=`46 passed`；compileall、
Ruff、`git diff --check` 通过。CUDA 资源记录为 free-before=`5368709120` bytes、peak allocated=`1599782400`
bytes、runtime=`39.8108 s`，全程 `stage_b_g1_replay_executed=false`。

RCR-A harness/config/tests 独立 commit 为 `c0bbe12db909d4ceb5176f72dbdb69df5c6c8ff5`；当前治理保持
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`。唯一下一动作固定为 `PLAN_PRIOR_SHAPE_CONTRACT_REVIEW`。

## H-LOMA-GEN-P1-R2-UACR unit-aware calibration contract repair attempt-02（2026-09-04，Windows-native，calibration-only）

R2 只修复 C6 已确认的 modal coefficient-power calibration contract：旧 v1 的
`torch.finfo(torch.float32).eps` absolute denominator floor 被移除；对同一 C6 raw replay 中每个
finite positive `mean_cal(|c_j|^2)` 直接登记为 v2 `port_scale`。旧 v1 producer、reader 与 artifact
保持不变；没有重新运行 CUDA、没有 optical forward、没有 downstream replay，也没有修改 G1/P4 evidence。

新 contract 为 `scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`，同时保存
`raw_port_mean` 与 registered `port_scale` 的一对一审计字段，绑定同一 1000-sample train-only
zero-phase population、dataset manifest、calibration indices、scaled LP01 mode bank、pre-OLS
plane identity、C6 source manifest/payload 和 parent v1 SHA。v2 consumer 要求显式 artifact SHA 与
这些 lineage bindings，并对历史 v1 payload fail-closed；历史 v1 reader 仍可读。

当前 write-once 结果为
`outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/`；
attempt-01 保留为不可覆盖的初始 write-once 结果，attempt-02 为当前 descriptive authority。attempt-02
包含 `repaired_calibration.pt`、`calibration_manifest.json`、`calibration_metrics.json`、
`old_new_calibration_comparison.json`、单张 `calibration_scale_summary.png` 与 `compact_report.json`，
共 `159393` bytes（manifest 外 artifact `153516` bytes），无 TIFF、隐藏文件或 `.tmp`。新 payload SHA-256
为 `ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`；parent v1 SHA
`a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959` 前后不变；C6 manifest SHA
为 `dfe880fecf297099c8c573579c7c064887ac1ded9fbd7b745e5bbadefcafedfe`。

R2 raw FP64 min/median/mean/max=`1.508997e-12 / 1.127986e-10 / 1.741773e-10 /
6.304158e-10`；registered scale 与 raw mean exact equal，100 个 port 保留独立 variation，且不再
全为 float32 eps。raw-reference z 的 global mean=`0.9999999986`，最大 per-port mean deviation
`1.451667e-7`，finite；z variance/range 仅作 calibration sanity description，不改变 prior
mean=`1.0`、variance=`0.25`、lower=`0.0`。

Gate=`R2_CALIBRATION_REPAIR_PASS`，A–O calibration-only checks 全部通过：raw finite/positive、
raw/registered equality、variation、global/per-port z mean、v1 immutability、new identity、C6
state-digest equality，以及 optimizer/backward/checkpoint/refit/training/prior-generation/decoder/
CMOS/MPCDAR 全部 false。R2 定向、C6、Scaled-Chen BOGT 与 G1 v1 preservation tests=`48 passed`；
compileall、Ruff、`git diff --check` 通过。治理继续为
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false`。P4 仍是旧 registered coordinate 下的
support-mismatch measurement，P1 contract mismatch 仍要求后续校准语义审查；R2 不声称 generation
fixed 或 generative manifold supported。唯一下一动作是 `PLAN_REPAIRED_COORDINATE_REPLAY`。

## H-LOMA-GEN-P1-C6-RSC raw-scale canary attempt-01（2026-09-03，Windows-native，diagnostic-only）

在不修改 calibration producer 的 floor/clamp 语义、不写 repaired calibration、不进入 decoder/CMOS/MPCDAR、
不训练或生成 prior 的前提下，使用注册的 `scaled_chen_bogt_calibration.v1`、同一 1000-sample train-only
zero-phase calibration population、MNIST preprocessing、SMF mode bank 与 optical prefix，完成一次 raw
per-port power canary。旧 calibration artifact 未覆盖；其 SHA-256=`a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959`。

输出为 `outputs/20260903_v2/runs/lightgen_scaled_chen_p1_calibration_raw_scale_canary_v1_attempt_01/`，
包含 `raw_scale_manifest.json`、`raw_scale_metrics.json`、`raw_port_means.pt`、`raw_scale_summary.png` 与
`compact_report.json`，共 `547210` bytes；无 repaired calibration、隐藏文件、`.tmp` 或 TIFF。raw mean
FP64 的 min/median/mean/max 为 `1.508997e-12 / 1.127986e-10 / 1.741773e-10 / 6.304158e-10`；
100/100 为正且有限，100/100 小于 float32 eps，0/100 小于 float32 tiny，0/100 小于既有
`median×0.001` relative floor。`eps=1.192093e-7`、`tiny=1.175494e-38`；FP32 reduction 与 FP64
reference 最大相对差 `1.451667e-7`，FP64→FP32→FP64 最大相对 cast error `5.324898e-8`，均在登记容差
`9.536743e-7` 内。旧 absolute floor 为 eps，median/p95/max distortion 为 `1045.03 / 18814.48 /
78999.02`，且 registered 100-port eps scale 被 raw replay 精确重放。zero-phase reference contract=`YES`；
raw coordinate sanity 的 global mean=`1.0`，仅作定义恒等式检查；learned-state comparison 跳过。

冻结状态 digest before/after 相同，optimizer/backward/checkpoint/refit/decoder/prior decode/generation/
CMOS/MPCDAR 均为 false。结果 Gate=`C6_CANARY_PASS`，支持 `C6_ABSOLUTE_EPSILON_FLOOR_INVALID_FOR_UNITS`，
但不改变现有 P4/P1 authority，也不授权训练或 formal promotion。Windows-native C6/相关 contract tests
为 `42 passed`；compileall、Ruff、`git diff --check` 通过；PNG 已实际查看。starting/ending HEAD 均为
`2fcd16a0bef7de2a6483f5ec42070a895bbcbcac`，branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`。
本轮唯一下一动作固定为 `IMPLEMENT_R2_UNIT_AWARE_CALIBRATION_REPAIR`；本记录不实施 R2。

## H-LOMA-CBOGT-S9V-FULL-PATH-VIS upright eight-stage rerender attempt-02（2026-09-03，Windows-native，diagnostic-only）

针对 reference-style 反馈，保留不可覆盖的 full-path visualization attempt-01，使用冻结 step-5000
checkpoint、RMAL/S7/S8/S9 parent chain 和同一 confirmation panel 执行一次受控 CUDA replay；未训练、未 refit、
未 rank search、未修改光学参数或 checkpoint。新输出为
`outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_02/`，attempt-01
manifest SHA-256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`，新 manifest
SHA-256=`393ebefe6c11537f5e1841a905eb0eed4365d7db57879cd823d477338b146703`。

主流程 layout schema=`full_optical_path_eight_stage_upright_v1`，严格为
`SLM output → Front 4f → Encoder layer 1 → Encoder layer 2 / Pre-OLS → OLS output / Decoder input →
Decoder layer 1 → Decoder layer 2 → Camera incident field`。每个数字为 amplitude/phase 两行，分为
`0–4` 与 `5–9` 两页；主图宽度约 `7.18 in`，PNG 尺寸均为 `2153×2387`。Supporting reconstruction、
repair/error 与 F64/F32 figures 保持原有科学语义；总计 `26` 个文件、`13413276` bytes，仅含 PNG/PDF/SVG/JSON，
无 TIFF、隐藏文件或 `.tmp`。

方向解释已写入 v2 contract 与 manifest：`IdealAmplitudeSLM` 将 command 原坐标映射为零相位复场，
所以 SLM output 不应自行倒置；旧图的 SLM output 倒置来自 renderer 错误的 display-only 180° rotation。
真实前向 4f relay 的倍率为 `-0.4`，后向 relay 为 `-2.5`，各产生一次倒像，净倍率为 `+1.0`。新图仅对
中间平面绘图副本做显式 display-only 摆正，SLM output 与 camera incident field 保留 native coordinate；
raw tensor、CMOS intensity、reconstruction 和实际生成内容均不受影响。

QA：full-path contract/layout 回归与相关 v2 optical tests=`32 passed`；compileall、Ruff、`git diff --check`
通过。5/5 alignment=`PASS`（0 fail/0 warn），5/5 PDF text audit 最小字号=`5.0 pt`且无低于阈值文本，
5/5 strict collision=`PASS`（0 fail/0 warn）；两张主流程 PNG 已实际查看。Nature source validator=`20 PASS/0 FAIL/1 WARN`，
唯一 WARN 为 renderer 中保留的 legacy export/宽度启发式，不影响 attempt-02 的实际无 TIFF 输出与严格图形 QA。

attempt-01 26 个文件逐文件 SHA-256 与运行前完全一致。执行前后 branch 均为
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`；当前 worktree 仅含既有 G1 prior-support 修改、本次 full-path
renderer/test/config 与两份状态文档，未发现额外 dirty path，未创建 Git commit。manifest 标记
`diagnostic_only=true`、`cuda_replay=true`、`optical_data_changed=false`、`training_authorized=false`、
`formal_promotion=false`；治理仍为 `mainline_state=closed_at_current_evidence_boundary`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`bridge=invalid / cloud_sources_sync_blocked`。唯一下一动作固定为
`manual_review_FULL_PATH_VIS_result`。

## H-LOMA-CBOGT-FIBER-VIS2 reference-style process-flow rerender attempt-15（2026-09-03，Windows-native，diagnostic-only）

针对人工反馈，将 attempt-14 的按光纤整组排列改为同一数字内交错排列。复用未修改的 attempt-10
`metadata_qa/panel.pt`，通过 `--render-process-flow-from-panel` panel-only 入口生成
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_15/`；
attempt-10 与 attempt-14 均保持不可覆盖，attempt-15 supersedes attempt-14。未重新 CUDA replay，未改变
任何 optical tensor、display scale、shared frontend、phase mask、三臂 contract 或 VIS1 attempt-06 authority。

新 layout schema=`same_digit_interleaved_three_arm_process_flow_v1`，artifact layout schema=
`vis2_process_flow_interleaved_v1`，row order=`digit_then_arm_then_amplitude_phase`。三页仍固定为标准数字
`0–3`、`4–6`、`7–9`；每个数字严格按 `SMF amplitude/phase → FMF amplitude/phase → MMF-M6 amplitude/phase`
排列，再进入下一个数字。完整流程列保持 `Input → SLM command → SLM output → Encoder in → Encoder L1 →
Pre-OLS → Modal → Post-fiber → Decoder L1 → Decoder L2 → CMOS field → CMOS intensity → MPCDAR R16`。
每个 arm 使用蓝/绿/橙边框与行标签区分，空间场保持等比例显示；Modal 继续使用显式 family×port 非空间矩阵，
MMF-M6 的 `ETA1* / XI2* / XIETA* / ETA2*` synthetic 标识保持不变。图中仅显示 digit/arm/amplitude/phase，
dataset index 写入 manifest；不作 fiber ranking、physical MMF、capacity、reconstruction-manifold、generation、
training 或 promotion 结论。

attempt-15 共 `25` 个文件、`56234887` bytes：3 个 PNG、3 个 PDF、3 个主 SVG、3 个 alignment JSON、3 个
alignment SVG，以及 `metadata_qa/` 下 9 个 approved source/manifest artifacts。无 TIFF、隐藏文件或 `.tmp`；
PNG 尺寸为 `3363×4204`、`3366×3303`、`3366×3303`，PDF 均为单页。source panel SHA-256 保持批准值
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；source comparison/visualization
manifest SHA-256 分别为 `d4229e4f966f911fee510ed4bcfa385eb46821ecf84289c0e6b7a745f679fdd9` 与
`8dcc62732b213c42cd12539b6f9329310ae2ec252113261154ba5f9fe2ed8648`。attempt-15 comparison/visualization
manifest SHA-256 分别为 `2d9946b29f1bd24a34081e0cef3dde6a872f894c3d5d2681ac9752198148196a` 与
`2cfe75f7bf6248504e65ecbf0a769508eba968a38408eb695a23ba95970509f3`；manifest 明确记录
`render_only_from_panel=true`、`cuda_replay=false`、`optical_data_changed=false`、`rows_per_digit=6`、
`row_records` 与 `supersedes=attempt_14`。

QA：新增交错行序回归测试先失败后通过；Windows-native relevant suite=`52 passed`；compileall、Ruff、
`git diff --check` 通过。3/3 alignment audit=`PASS`（0 fail/0 warn），3/3 PDF text audit 最小字号=`5.0 pt`
且无低于阈值文本，3/3 strict collision audit=`PASS`（0 fail/0 warn）；三张 PNG 已实际查看。Nature source
validator=`18 PASS/0 FAIL/3 WARN`，WARN 仅为既有 renderer 的 300 dpi、宽度启发式与固定诊断样本 uncertainty
启发式，不改变本次输出的严格图形 QA 结论。

执行前后 branch 均为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`。worktree 仍只有受保护的 G1 prior-support config/observer/test、
两份状态文档、当前 VIS2 renderer/test；未发现额外用户修改，未创建 Git commit。治理保持
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`。本轮未运行 CUDA、训练、refit、
rank search、formal runtime 或 bridge；唯一下一动作固定为 `manual_review_VIS2_figures`。

## H-LOMA-GEN-T1-RCWV repaired-coordinate BOGT weight validity canary attempt-01（2026-09-04）

在 branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting HEAD=
`47b9aa0896cae88516b3872e6f3355ff12434054` 的初始 clean preflight 后，使用固定
step-5000 parent checkpoint 与同一注册三组 panel，执行唯一允许的 no-update CUDA gradient
measurement。只比较旧 calibration.v1 与已批准的 `scaled_chen_bogt_calibration.v2` /
`unit_aware_raw_mean_v2` 坐标；alpha=`0.002996375650268832`、gamma=`0.0`、目标比值=`0.10`、
注册边界为 `[0.05, 0.40]`，gradient norm 仍为
`sqrt(sum(parameter_gradient**2))` over `system.encoder.phase` and `system.decoder.phase`。

旧 alpha 的 immutable provenance 已闭合：旧 preflight/pilot/panel artifact、旧 calibration、formal
config、review、dataset/schedule/panel indices、batch=`10`、microbatch=`1`、Adam/lr=`0.001` 与
三组 panel identity 全部匹配。注册旧 preflight ratios 为
`[0.1102407, 0.0855945, 0.1081556]`；在同一 step-5000 parent 上的 v1 replay ratios 为
`[0.1864269, 0.1105396, 0.1395263]`。修复坐标的同一 parent replay ratios 为
`[1.9516949, 1.5488551, 1.5754461]`，median/min/max=`1.5754461 / 1.5488551 / 1.9516949`。
因此 Gate=`RC_ALPHA_TOO_STRONG`：当前 alpha 在 repaired coordinate 下超出注册上界，不把该结果
解释为唯一 training/BOGT 失败，也不实施 alignment 或训练。

当前 write-once 结果为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_v1_attempt_01/`，
包含 `weight_validity_manifest.json`、`gradient_panel_metrics.json`、`old_new_gradient_comparison.json`
和 `compact_report.json`，共 `38814` bytes；manifest 引用 hash 全部通过，无隐藏文件、`.tmp`、
checkpoint 或生成图像。repaired payload SHA-256=`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`，
parent checkpoint SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`。

optimizer state 只读检查为 `AVAILABLE_AND_BOUND`，但没有创建 optimizer；state digest、checkpoint
hash、learned-state digest 前后一致，`optimizer_step_called=false`、`optimizer_state_mutated=false`、
`checkpoint_created=false`、`phase_updated=false`、`refit=false`、`training=false`、
`prior_generation=false`。T1-RCWV targeted regression=`8 passed`；compileall、Ruff 与
`git diff --check` 通过。治理继续为 `mainline_state=closed_at_current_evidence_boundary`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`。
唯一下一动作固定为 `PLAN_REPAIRED_COORDINATE_ALIGNMENT_WEIGHT_REGISTRATION`。
## H-LOMA-GEN-T1-RCAWR：Repaired-Coordinate BOGT Alignment Weight Registration（2026-09-04）

本轮只登记 repaired-coordinate BOGT alignment weight；没有修改 calibration/prior、没有创建
optimizer、没有执行 backward/gradient measurement、training、checkpoint、refit、prior generation
或 decoder/CMOS/MPCDAR。registration consumer 严格绑定 R2
`scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`（payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`）、step-5000 parent checkpoint
（SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`）和冻结 RCWV
attempt-01。历史 Set B registered preflight global-alpha replay 是 registration authority；Set A
step-5000 replay 只作 context。

从 RCWV repaired panel 的冻结 effective ratios 与 raw norm identity 重建
`K_i=G_BOGT/G_MSE`，按历史规则 `alpha_i=0.10/K_i` 并取三 panel 算术平均，得到
`alpha_repaired=0.00017905883237862847`（canonical Decimal=
`0.000179058832378628469663887774247392581098542509215070071798828`）。三组预测 effective
ratios=`0.1166303093/0.0925572168/0.0941462530`，均落在 inclusive `[0.05,0.40]`，Gate=
`RCAWR_REGISTRATION_PASS`。该 Gate 只登记一个候选 alpha，不授权 training 或 generation。

首次 attempt-01 因 manifest 发布顺序检查过早而只留下 4 个中间 JSON，保留但不消费；正式
write-once output 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_02/`，
仅含 `weight_registration_manifest.json`、`weight_registration_metrics.json`、
`panel_registration_details.json`、`old_new_alpha_comparison.json` 和 `compact_report.json`，无图像。
manifest 记录所有源 artifact/path/hash、Set A/B provenance、alpha 推导、state isolation 和
governance；若目录已存在只能递增到下一个 attempt，禁止覆盖。RCAWR focused regression、
compileall、Ruff 与 `git diff --check` 均通过；当前治理继续为
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false`。唯一下一动作固定为
`IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY`。

## H-LOMA-GEN-T1-GAC：Repaired-Coordinate Generative Alignment Canary（2026-09-04，Windows-native，diagnostic-only）

本轮只执行一个预注册的 bounded single-intervention canary：从 exact step-5000 parent continuation，
恢复绑定的 Adam state，使用 RCAWR attempt-02 登记的 repaired-coordinate BOGT alpha，训练固定 1000
steps 至 logical step 6000；没有 alpha/horizon/seed sweep、best-checkpoint selection、追加训练、
calibration/prior/latent/BOGT/optical/OLS/fiber/architecture/MPCDAR 修改，也没有 prior/decoder/CMOS
generation、FID/classifier/diversity/novelty 或 G2。正式 canary 执行时的 clean implementation HEAD
为 `5913a045b0bd932aff11e335f41a1f246b78a0f3`；任务最初 preflight 的 clean starting HEAD 为
`88a09b5f737f4eb9246ad97351b367a5e3a20841`。

3-step real-update preflight 输出为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_preflight_attempt_01/preflight.json`，
status=`passed`；5001–5003 的 phase 参数真实更新、OLS 保持冻结、Adam state 从 parent step 5000
恢复并推进到 5003，finite/resource 检查通过；冻结 parent negative control 按预期拒绝严格 primary
Gate。正式 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_attempt_01/`，
共 10 个 JSON/PT 文件、`17055800` bytes；training manifest SHA-256=
`48d3e883355b329c80072c50cb2b06187d4cc7454911556f9b124115ab927630`，end checkpoint SHA-256=
`b8ed605d0ed4506c0415f8199ac16fd1f4274cc7efa419f00af785a2ce4f9542`，compact report SHA-256=
`85d296f5419d8c85b2ba71dae03c80d597a1126c9c547f908823073b38cc6af6`。输出无隐藏文件、`.tmp` 或
生成内容。

训练 contract 保持 `scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`，payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`；parent step-5000 checkpoint
SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`；prior 仍为
unconditional factorized truncated normal `(mean=1.0, variance=0.25, lower=0)`；alpha 为注册值
`0.00017905883237862847`（旧 alpha `0.002996375650268832` 被拒绝），gamma=`0`，objective=`mse_kl`，
唯一 trainable parameters 为 `system.encoder.phase` 与 `system.decoder.phase`。seed=`1234`，Adam
`lr=0.001`、scheduler=`none`；runtime=`1694.4757 s`，peak CUDA allocation=`2330944000` bytes。
start/end learned-state digest 为
`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5` /
`e8fa50507e500a7f82ce5675080bd769f0bc5c8edcf59ae6e43c841e0fa7d1c0`；OLS 未变，只有注册的 phase
参数发生变化，end optimizer step=`6000`。

端点 G1-SA 为 R50=`1.8142581`，95% CI=`[1.7888644,1.8446554]`；R95=`1.0970011`，95% CI=
`[1.0733978,1.1168799]`。frozen parent baseline 为 R50=`1.8389779`、CI lower=`1.8142740`，
R95=`1.0954389`、CI lower=`1.0679024`；两个严格 CI-separated 条件均未通过。端点 MRA selection/
confirmation 的 `(z_mean, z_variance, Gaussian KL, numerical KL, complex rank, power rank)` 分别为
`(1.0134035,1.4821789,3.0508981,2.9807997,12.2222137,8.2595968)` /
`(0.9309632,1.3868347,2.6405313,2.5035856,12.8533897,8.4111090)`。mean-scale safety 与 0.90
rank-retention safety 均通过；current-sample confirmation same-route safety 未通过：paired MSE=
`0.05678915` 高于 template control=`0.04197700`（低于 cyclic control=`0.07559964`），虽 finite。

因此最终 classification=`T1_ALIGNMENT_HARMFUL`，Gate 依据是 primary support-alignment 未通过且
current-sample safety failure；这只是该固定 1000-step canary 的 trade-off 结论，不把结果升级为
training failure、BOGT objective 唯一失败、generative manifold、prior decoder accessibility 或
generation readiness。formal end evaluation 无 optimizer update；general training authorization 在
任务结束后恢复为 false。GAC focused/直接依赖回归=`42 passed`（首次运行仅受既有 pytest temp ACL
影响，使用显式可写 basetemp 后通过）；compileall、Ruff 与 `git diff --check` 通过。治理继续为
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`。唯一下一动作固定为 `REVIEW_ALIGNMENT_TRADEOFF_FAILURE`。

## H-LOMA-GEN-T2-SAAO-PF Support-Aware Energy-Distance Objective Preflight（2026-09-04，Windows-native，diagnostic-only）

本轮只完成 repaired-coordinate support-aware Energy Distance objective 的 preflight。实现提交为
`51edbe216a8000eb944daa6e66ae41a3dc934cb1`；随后在 clean HEAD 上使用一次受限 CUDA no-update
gradient-panel measurement，固定 parent step-5000 checkpoint（SHA-256=
`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`）、R2
`unit_aware_raw_mean_v2` calibration（payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`）、RCR-B distance artifact
（SHA-256=`d288beb4bd57a1db2a9a864dd8db1be5ef22f75f14702fa13f01b01623671466`）和 parent MRA
panel（SHA-256=`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`）。未训练、未创建
optimizer、未调用 backward、未写 checkpoint、未改变 phase/calibration/prior、未进入 decoder/CMOS/
MPCDAR/generation。

目标函数冻结为 `L_total=L_MSE+beta*L_ED+gamma*L_OP+alpha_BOGT*L_BOGT`，其中 ED 是
`[10,100]` logical batch 与 `[10,100]` prior batch 上的 ordered unbiased U-statistic，包含两个
off-diagonal within terms 与 prior-prior term，最终统一除以 `sqrt(100)`；允许有限负值且不做
epsilon smoothing。microbatch=`1` 只通过完整 logical-batch 的 latent VJP 重放，未把独立的 batch-1
ED 平均成替代目标。ED gradient boundary 为 encoder live、decoder zero direct gradient；未来 canary
contract 保留 `alpha_BOGT=0`、`gamma=0` 和日志 `[mse, ed, weighted_ed, total, encoder_grad_norm,
decoder_grad_norm, finite, z_mean, z_variance, optimizer_telemetry]`。

三组 T1-RCWV 注册 panel 的 `(G_MSE, G_ED, beta_i)` 分别为
`(0.0003082343, 0.0213519670, 0.0014435873)`、
`(0.0004745822, 0.0298346393, 0.0015907087)`、
`(0.0003725952, 0.0296749882, 0.0012555868)`。唯一 analytic registration 为
`beta_registered=0.0014299609134309638`；replay ratios=`0.0990561/0.0898946/0.1138879`，
min/median/max=`0.0898946/0.0990561/0.1138879`，全部通过 inclusive `[0.05,0.40]`。
CPU reference 与 optical panel-0 的 scalar/parameter-gradient equivalence 均通过；optical scalar
absolute delta=`2.98e-8`，parameter-gradient max absolute delta=`0`。

prior stream 已隔离：evaluation seed=`1237`、count=`5000`、sample SHA-256=
`c11402193afd71d176468760024ce7dec0d4e4ee3d6d206ac7293ca8ee77fd3e`；future training base seed=`1238`
按 `1238*100000+global_step`，panel namespace=`10000`，三组 panel seeds=`123810000/123810001/123810002`。
paired-prior endpoint 与 parent-relative MSE endpoint 均标记 ready，但尚未执行 canary，也不提供训练后
改善结论。write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_support_aware_energy_distance_preflight_v1_attempt_01/`，
9 个 JSON/PT 文件、`29852` bytes；无 TIFF、隐藏文件或 `.tmp`，manifest SHA-256=
`a9055ec7dfbccad92e683e3569ba4517becbda5d05565966ab22430d806df18a`。

T2-SAAO-PF targeted 与 R2/RCR-A/PCR/BOGT 直接依赖回归=`81 passed`；compileall、Ruff、`git diff --check`
通过。七个 gate 全部通过：`ENERGY_DISTANCE_OBJECTIVE_VALID`、`FULL_BATCH_GRADIENT_EQUIVALENCE_PASS`、
`PRIOR_STREAM_CONTRACT_PASS`、`ED_BETA_REGISTRATION_PASS`、`PAIRED_PRIOR_ENDPOINT_READY`、
`PARENT_RELATIVE_SAFETY_ENDPOINT_READY`、`STATE_ISOLATION_PASS`；最终 classification=
`SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS`，唯一下一动作固定为 `IMPLEMENT_SUPPORT_AWARE_ALIGNMENT_CANARY`。
治理继续为 `mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`。

## H-LOMA-GEN-T2-SAAC：SCAM normalized-power controllability canary（2026-09-04，Windows-native，diagnostic-only）

本轮在同一专用入口完成已 preflight 的唯一一次 T2 bounded canary：从固定 step-5000 parent 恢复
registered Adam state，使用 repaired calibration `scaled_chen_bogt_calibration.v2` /
`unit_aware_raw_mean_v2`、registered prior、Energy-Distance 与
`beta=0.0014299609134309638`，固定 1000 steps 到 logical step 6000。`alpha_BOGT=0`、`gamma=0`、
logical/prior batch=`10`、microbatch=`1`、two-pass gradient、optimizer、horizon、seed=`1234` 和
既有 Gate 均保持不变；唯一 trainable parameters 为 `system.encoder.phase` 与
`system.decoder.phase`。没有重新选择样本，没有 MMD/SWD/SCED/Chamfer/Sinkhorn、新 prior、新 beta 或
第二次 global-power objective，也没有 prior generation、semantic OLS audit、G2、bridge、formal
runtime 或 promotion。

新增实现/config/tests 的 clean execution boundary 为
`2665b9d13aedef898eff7e2b567f1988feeccf9b`；同入口 3-step real-update preflight 通过，冻结 parent
negative control 正确拒绝 primary Gate。正式 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_support_aware_alignment_canary_v1_attempt_01/`，
10 个 JSON/PT 文件、`17357744` bytes；training manifest SHA-256=`8b46943ce884143ebe24b0a10f782738d7ec0517b79f5efba0ac343c173807b2`，
end checkpoint SHA-256=`c1a834c755f7eef27b8e1b3ed7ce5b1f154eca7fe775ccd5ea65e05d14fa82a8`，compact report
SHA-256=`1d7d1a6473e141b85c2f15d92bf7ff83c97c1ae744f9bf93eec0d5377cb9a738`。逐文件 artifact
reference hash/size、目录树、无 TIFF/隐藏文件/`.tmp` 检查均通过；same-entrypoint preflight JSON
SHA-256=`e7efd967b5762ccb792a40f278e2a000f08388151a8620367e91014938582f43`，上游 T2-SAAO-PF
manifest SHA-256=`a9055ec7dfbccad92e683e3569ba4517becbda5d05565966ab22430d806df18a`。

正式训练实际完成 5001–6000 共 1000 steps，runtime=`676.8129 s`、peak CUDA allocation=`2814739456`
bytes；learned-state digest 从
`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5` 变为
`c3fbb90378dd06ec76cb6b72d05bd6d9fde57c0f64209fc6edcfad90df8a1abc`，OLS unchanged，只有注册
phase 参数改变，optimizer endpoint 为 step 6000。

T2 observer 明确是 `z_j=|c_j|^2/s_j` 的 `normalized_power_coordinate_observer`；物理 OLS 状态
仍为 `c∈C^100`，包含 amplitude 与 phase。结果为有效
`T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED`：R50=`1.7548616`，95% CI=`[1.7309163,1.7764292]`
（R50 CI-separated 条件通过）；R95=`1.0573063`，95% CI=`[1.0348356,1.0767013]`
（R95 条件未通过）。paired prior `ΔP50=0.0070921`、upper95=`0.0076903`，`ΔP95=0.0287672`、
upper95=`0.0293099`，均未满足 upper95<0；parent-relative MSE mean delta=`-0.0010066`、
95% CI=`[-0.0010208,-0.0009922]`，mean safety 与 rank safety 通过。因此这是当前
global normalized-power geometry 的 canary not-supported 结果，不推出 semantic OLS generation
失败或 OLS generation impossible。

SCAM architecture authority=`OLS_INPUT_CONTRACT_HIERARCHICAL_CONTROLLER_RECOMMENDED`，
plan=`SCAM_PLAN_READY`；`SCAM authority=FAIL` 仅对应本次 T2 Gate。`complex semantic OLS`、
`label-to-OLS` 和 `8D-to-OLS` 结论分别为 `UNESTABLISHED`、`UNRESOLVED`、`PARTIAL / UNRESOLVED`；
global-prior family=`CLOSED_AFTER_THIS_CANARY`。无论本次 valid FAIL，唯一下一动作统一为
`PLAN_S1_OLS_SEMANTIC_REPRESENTATION_AUDIT`。

新 T2-SAAC 回归与既有 T2-SAAO-PF 直接依赖测试=`17 passed`；compileall、Ruff、`git diff --check`
通过。治理保持 `mainline_state=closed_at_current_evidence_boundary`、
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`。
