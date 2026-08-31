# codex-original 项目记忆

更新时间：2026-08-31

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
- 当前最新有效科研结果是 Scaled-Chen decoder→CMOS 可观测性归因的
  `relay_propagation_bottleneck_supported` / `valid_failed`。唯一允许的下一动作是
  `manual_review_scaled_chen_bounded_back_relay_geometry_diagnostic_contract`；在项目负责人
  新合同前不得进入 FMF、继续训练、exposure retuning、生成或 promotion。若交接文档更新，
  以最新的 `CURRENT_STATUS.md`、`status/current_handoff.md` 和 hypothesis ledger 为准。
- 本次工作流治理允许的实现动作是维护记忆入口、项目文档和环境入口；不修改训练代码、
  `outputs`、WSL、SQLite 或历史 transcript。只有用户明确需要离线上传时，才生成并验证
  受控 Sources 包。

## Scaled-Chen 秩匹配 OLS 几何归因（2026-08-30）

在 Windows 原生 CUDA 环境完成了冻结权重的九组 LP01 几何归因审计。模式网格为
`fill={0.70,0.80,0.90}` × `facet_magnification={0.80,1.00,1.20}`，固定 100 端口、
身份传输、原 calibration 和 2+2 formal Identity 光学。选择集使用 test 后半段每类 100
张，确认集使用 horizon 原 5000 张 held-out；两者与 horizon 训练/标定分离。

最佳非基线为 `(fill=0.80, facet_magnification=0.80)`。选择集改善 `4.02%`，确认集
step 0/5000 改善 `4.48%/3.68%`，bootstrap 95% 差值区间为
`[0.00210138,0.00215426]`；未达到预注册 `5%` 门槛，且训练后收益没有超过初始收益。
所有 finite、100 秩、冻结状态、负对照、显存和 lineage 门通过；evaluator calls 为 `0`，
峰值 CUDA `2279883264` bytes。权威产物在
`outputs/20260830_v2/runs/lightgen_optical_manifold_smf_bogt_geometry_attribution_v1_attempt_01/`，
validation/audit/decomposition/panel SHA-256 分别为
`ff904bfab0ca63208e881d51c3e06650295ca1e09a3f3b0a9a1a79c374bae0bd`、
`923d2e233cba0f795ec80fbd536846c356cdc177e2dfe0d77957bf4835565f8a`、
`952d062c8a72b872e299da3a20e870b8c7a414b1fd27dc9d9d57a376ca74e5e4`、
`94f1ac7fc74a3bf7b78c60ccc56a7fa479ab56fa30e06dcbc27ddecdc702fd4b`。

该结果只支持分布式瓶颈 closeout 人工审查，不证明连续几何穷尽、FMF 优越性或正式光学
可用性；`training_authorized=false` 与 `formal_promotion=false` 保持不变。

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

GPT → Codex 的完整 Delta、One Gate、Short Prompt、预算和失败规范见
[operations/delta_one_gate_short_prompt.md](operations/delta_one_gate_short_prompt.md)。
本文件只保留云端可消费的短模板；科研事实、授权和禁止事项仍以本文件前文、
CURRENT_STATUS.md、status/current_handoff.md 和本地 evidence 为准。

本地项目 `codex-original · Windows` 的主目录为 `D:\codex-original`，辅助目录为
`C:\Users\24567\Documents\Codex\codex-original-history-context`。云端项目
`codex-original · ChatGPT` 与本地项目通过六文件 Sources 快照和标准交接块连接；项目
名称相似本身不提供技术绑定。

ChatGPT 模式负责讨论、规划和总结；Codex 模式负责读取真实工作树、核对 Git、修改
代码/文档和运行验证。只有 Codex 在验证后才能写入权威项目记忆；云端 Sources 仅是
某个 Git HEAD 的只读快照。

交接使用 `MODE=VERIFY|PLAN|IMPLEMENT|REPORT`。`MODE=IMPLEMENT` 表示计划已经批准；
Codex 只执行一次 bounded preflight，预检后只能执行或返回 `BLOCKED`，不得因发现更多
背景而重新规划。完整快照集合和发布资格见操作规范。

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

## Delta + One Gate 短交接模板

ChatGPT → Codex：

固定协议（不必复制到每次任务）：Short Prompt 默认不预填 artifact SHA-256 清单；
Hash 只在 Gate 的 evidence consumer 要求精确身份时定向核验，不自动选择 latest。
六个 Sources 文件必须来自同一 `snapshot_id + git_head` 集合。

```text
模式：
- MODE=<VERIFY|PLAN|IMPLEMENT|REPORT>
- 若为 IMPLEMENT：计划已批准；一次 preflight 后只允许执行或 BLOCKED，不重新规划。

状态增量：
- 云端基准：
- 本地核验：
- 决策相关变化：
- 保持不变：
- 权威性：

One Gate：
- ID/单一问题：
- 决策消费者：
- 精确输入及 producer/lineage（Hash 按需核验）：
- 指标/阈值：
- 通过或失败后的唯一下一动作：

执行边界：
- 允许：
- 禁止：
- 不得自动扩大到：

验收与停止：
- 必须验证：
- 停止条件：
- 缺失输入或失败时返回：

反馈格式：
结果；本地增量核验；Gate 判定与唯一下一动作；修改文件；
验证结果；未验证内容与已知风险；项目记忆与 Sources 状态。
```

Codex → ChatGPT：

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

Short Prompt 通常为 300–1,000 tokens，不要求为了达到下限填充固定协议；1,000–1,500
仅在无法用规范引用压缩时允许；超过 1,500 不再称为 Short Prompt，必须拆 Gate 或按
操作规范升级为 Long Prompt。
完整 Long Prompt 目标为 1,500–3,000 tokens，超过 4,000 tokens 应引用规范或拆分。
一个任务只能有一个主 Gate；完整预算和失败映射见上述操作规范。
未要求 Hash 时反馈 hash_not_required；必需 Hash 缺失或输入有歧义返回 blocked；
记录 Hash 与实际 artifact 不一致返回
invalid_harness；不把 Hash 不一致转换为 valid_failed。正式 evidence 和 bridge manifest
中的 Hash 保持不变。

Sources 发布资格由 bridge 的 `status` 与 manifest 的 `allow_dirty` 联合判断：
`status=current` 且 `allow_dirty=false` 为 `current_clean`，才允许人工替换云端；
`status=current` 且 `allow_dirty=true` 为 `current_dirty_planning_only`，只能讨论和检查；
其余 stale/invalid/missing 或混合集合为 `stale_or_conflicting`。旧副本不调和，不能按
时间戳自动选择最新文件。

## 记忆更新规则

1. ChatGPT 只提出更新建议，不直接把建议当作事实。
2. Codex 修改前先读取本文件、`AGENTS.md`、`CURRENT_STATUS.md` 和适用的交接文档。
3. 只有被 Git、测试、正式 evidence 或明确人工验收支持的事实才能写入“当前”章节。
4. 记忆或当前状态变化后，先在干净工作树运行 Sources `refresh`，再运行 `status`；
   只有 `refresh_ok`、`status=current` 且 manifest 的 `allow_dirty=false`（即
   `publication_state=current_clean`）才允许人工替换云端项目中的六个 Sources。
5. Sources 的内部 manifest 不上传；需要历史细节时只按 Session ID 生成临时摘要，任务
   结束后从云端项目移除。
6. 旧 WSL 路径、旧项目副本和历史摘要必须显式标为历史，不能伪装成当前状态。

## H-LOMA-CBOGT-S0-MRA：Scaled-Chen Manifold Readiness Audit（2026-08-31）

权威实现提交为 `a041187a6ff6292f2be6602871cfd6d35ca172c`，权威输出为
`outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/`。
冻结 Identity step-5500 与 Scaled-Chen step-0/1000/5000，在独立的 1000 张 selection
和 5000 张 confirmation panel 上完成了 pre-OLS、modal、post-OLS/decoder input、
decoder/CMOS、rank、retrieval、cross-swap、Jacobian、重建和同路线 prior 观测。
审计状态 `valid_failed`，harness 完整，`optimizer_created=false`、`evaluator_calls=0`、
`training_authorized=false`、`formal_promotion=false`。

当前结论只记录四项：

1. encoder 不是主要瓶颈候选：SLM→pre-OLS retention 为 `0.9182/0.9216`；OLS 前后
   modal 几何 retention 约 `0.8145/0.8092`，并与下游损失共同支持分布式而非单点归因。
2. 最大 representation compression 在 `decoder_layer_2→CMOS`，两个 panel retention
   为 `0.5827/0.5871`；但 decoder Jacobian 有效秩均为 `100`，不能单独归类为 decoder
   capacity bottleneck。
3. `reconstruction_manifold_supported=false`，因为 paired MSE 未同时胜过 cyclic 与
   class-template，且 geometry/retrieval controls 未形成稳定支持；不得把 reconstruction
   结果写成 generative manifold 结论。`generative_manifold_supported=false`。
4. parent contract prior 已注册但不可访问（Gaussian/numerical KL improvement 与
   complex/power rank gates 均失败），故 `generation_readiness=not_ready`。唯一下一步
   是 `manual_review_scaled_chen_distributed_bottleneck_closeout_contract`；不授权 canary、
   训练、FMF、prior-only generation 或 promotion。

结果证据哈希：validation
`b1713ee2676c4f5ab6e6f45838ed75bb1d2b1843e2ce738b53abe62d8e4b0c35`，audit
`6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556`，compact
`401fe2159f4342a4740b2c40955cb65fc7bb3600f2416fdc4171e1553eebeee0`，panel
`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`。历史 SMF
background 缺失的 preflight/training 文件保持零哈希记录；未借用其他路线 prior，未刷新
云端 Sources。

## H-LOMA-CBOGT-S0-DCM-OBS：Scaled-Chen Decoder→CMOS 可观测性归因（2026-08-31）

该 no-update 审计消费同一 MRA 的 Identity step-5500、Scaled-Chen step-0/1000/5000、
1000 张 selection 和 5000 张 confirmation，将旧的 decoder_layer_2→CMOS 段拆成
`decoder_layer_2 complex field→CMOS incident complex field→CMOS intensity`。实现提交为
`d5fbe5e6efd55076aca7020e0e1cb0f1c7763b19`；唯一权威运行目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_decoder_cmos_attribution_v1_attempt_03/`。

结论只记录四项：

1. `relay_propagation_bottleneck_supported`：decoder L2→incident retention 为
   selection `0.6571`、confirmation `0.6031`；incident→intensity 为 `0.8868/0.9735`，
   两段损失差 `0.3000`，最大唯一断点是 back-relay。
2. CMOS 强度确实是 incident 复场的 `|E|²`；全局相位与逐像素相位不变性控制通过，
   因而没有把 detector 平方律误当成可训练容量。模型/OLS 状态 hash 前后相同，峰值
   CUDA `2589355520` bytes，`optimizer_created=false`、`evaluator_calls=0`。
3. 该结果仅细化 MRA 的 `distributed_bottleneck_supported`，不改写 decoder Jacobian
   rank=100、`reconstruction_manifold_supported=false` 或 `generative_manifold_supported=false`。
4. `generation_readiness` 仍为 `not_ready`。唯一下一动作是
   `manual_review_scaled_chen_bounded_back_relay_geometry_diagnostic_contract`；不授权
   back-relay 之外的训练、FMF、prior sampling、generation canary 或 promotion。

attempt-03 的 validation/audit/compact/panel SHA-256 分别为
`89ece3debd73fa3b7738b7604cafb1d1e52085562d0c70c15563a8d8c39691a7`、
`16975f7f26db13100af134fad70e510a1978093b88c619bc6c3a3e08c067ed68`、
`349ef59d07ca42c6e5446d7bcbe578110b03af27cbc838d0ebe818be38aaa5da`、
`6120c7cc1dc4832587bfe6d91b118579fbf097849d6c9d9a3668b1eb3952ce28`。attempt-01/02
产物保留但不作权威证据；未修改正式 runtime、outputs/final_pt、data/raw 或历史归档。
