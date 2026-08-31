# 当前项目状态

更新时间：2026-08-31

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
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
