# 当前项目状态

更新时间：2026-09-01

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- 当前 clean HEAD（本次 RMAL result/status commit；RMAL validation/audit 绑定 clean
  implementation commit `5b274b926474784a1ad49744e75190afff31747c`，本文件在结果/状态提交中更新）。
- 当前最新研究解释是 RMAL 对 Scaled-Chen physical reconstruction mapping 的 bounded
  recoverability 结果；旧 evidence 仍不可变，TRBA/HFRR/DCM/MRA/RMRI 的历史 classification
  继续保留为父级事实。
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
