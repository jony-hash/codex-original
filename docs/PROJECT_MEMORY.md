# codex-original 项目记忆

更新时间：2026-09-02

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
- 当前最新有效科研结果是 H-LOMA-CBOGT-S7-MPCDAR 的 `passed` /
  `mean_preserving_centered_data_aware_readout_recovery_completed`，正式 replay 绑定
  clean implementation commit `b59172750d0c88bbc9f7426ab7a21a1d63639f57`：在冻结
  RMAL `W,b`、fit-derived `mu_fit/C_fit` 与预注册 candidate family
  `[1,2,4,8,16,32,64,128,256]` 下，rank 16 首次通过 selection + confirmation。
  该结果只表示预注册家族内的 mean-preserving fit-data-aware candidate 结果，不是
  minimum/intrinsic/physical rank，也不把 spectral energy 或 data-aware truncation
  升级为 reconstruction sufficiency。其 parent MCFA 仍为
  `detector_mean_centered_functional_attribution_completed`，DMSA/DSFA/LRRC 及更早
  RMAL/ZCSA/RWSA-ACTIVE/LSDA 身份保持不变。唯一允许的下一动作是
  `manual_review_MPCDAR_result`；在项目负责人审查前不得进入 FMF、继续训练、exposure
  retuning、生成或 promotion。relay family
  `scaled_chen_v1_local_backend_relay_attribution_family=closed_metric_confound_reinterpreted`
  保持。
  若交接文档更新，以最新的 `CURRENT_STATUS.md`、`status/current_handoff.md` 和 hypothesis
  ledger 为准。
- 本次工作流治理允许的实现动作是维护记忆入口、项目文档和环境入口；不修改训练代码、
  `outputs`、WSL、SQLite 或历史 transcript。只有用户明确需要离线上传时，才生成并验证
  受控 Sources 包。

## H-LOMA-CBOGT-S2-TRBA：Task-Relevant Bottleneck Attribution（2026-09-01）

TRBA 是 RMRI relay closeout 后唯一执行的冻结、无更新 task-relevant 归因。它消费
RMRI/MRA/HFRR 的已验证 parent evidence，在相同 selection（MNIST test 5000–9999，
每类 100）与 confirmation（train 55000–59999，5000）上只比较
`pre-OLS→modal_complex` 与 `decoder_input→decoder_layer_2`。实现提交为
`7c1c0f8dcfb878049260043441a117456e5ebe6a`；attempt-01 的发布失败产物保留但不消费，
attempt-02 是权威结果目录：
`outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/`。

Primary 使用不依赖维度、单样本全局相位、共同坐标置换或共同 unit-modulus 对角变换的
normalized complex Gram geometry。C0–C3 controls、finite、state freeze、panel isolation
和 parent lineage 全部通过；最大 control drift 为 geometry `1.48e-7`、label `5.63e-7`、
retrieval `0`。OLS geometry degradation 为 `0.171023/0.169979`，但 task-advantage
degradation 为 `-0.615485/-0.578570`、retrieval degradation 为 `0/0.002600`，所以只
记录 `generic_geometry_change_only`，不能称 task bottleneck。decoder-internal geometry
degradation 仅 `0.010264/0.010617`，task advantage 与 retrieval 同样没有下降。

TRBA 科学分类为 T4：`task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_scaled_chen_bottleneck_attribution_closeout_contract`。这不删除历史
MRA/RMRI 的 `distributed_bottleneck_supported_canonicalized`，也不把 raw complex/rank
变化重新升级为容量结论。`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 必须分开保持；
TRBA 不授权训练、FMF、prior sampling、generation、canary 或 formal promotion。

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

## H-LOMA-CBOGT-S0-BRAC：Scaled-Chen Back-Relay Axial / Conjugate 归因（2026-09-01）

该 no-update 审计消费父级 DCM 的固定 baseline、同一数据 panel 和同一 Scaled-Chen
step-5000 checkpoint，真实消费者是项目负责人对“nominal back-relay 附近的轴向/共轭
偏移是否足以支持局部 relay 调整诊断”的人工决定；拒绝决定是三臂选择门未达到实质
改善。实现提交为 `0307119ee7da2ad38905fa7f20066eefd927b5c2`，权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_bogt_back_relay_axial_conjugate_v1_attempt_03/`。

validation/preflight/audit/report/compact/panel/closeout SHA-256 依次为
`568670d5ccd4b151335fc8cc97c9c6cd15d5dd4dfd29d40c41c9fa1ce0e9384d`、
`15116230c4fdb7e8a59593bc754b24748bfe698837a9ef1a7a7a742ebacaf984`、
`2800a7c0c61008df0450ae3ef6e4ecc9779d7796ab16b29a7b20611d241815aa`、
`6749b336127e13b70e89c26f39956e52737d40a58344e902d11a90ccdbce579b`、
`099c77193f56dfb7c5b048b19025a914469e2d9af534bc5acb88ba40a7c6c785`、
`8a76e6aaf2df491d83f9cd322a6dffb7597de3fb52bb0fd0fc739a1bbf750ed0`、
`0dd3c67fcedb67df3da7bb4916723c905aa31e3491305522d222f8cee7d7c124`。

结果为 `valid_failed` / `relay_axial_conjugate_not_supported`，而不是
`invalid_harness` 或 `resource_aborted`。negative/baseline/positive relay retention
分别为 `0.656313/0.657125/0.658047`，paired MSE 分别为
`0.0620113/0.0620485/0.0621269`；最大 retention 增益只有 `0.0922` 个百分点，
MSE 方向不支持候选，故 `selected_candidate=null` 且没有 confirmation arm。
baseline replay 与父级 DCM 一致，phase/square-law、冻结和资源门通过；
`aperture_effectively_inactive=true`、`numerical_boundary_not_evaluated=true`，
panel 仅为 `mechanism_evidence_only`。

这不是对父级 relay bottleneck 的反证，而是对“微小轴向/共轭偏移可在当前 nominal relay 附近
恢复重建”的否决。父级 `relay_propagation_bottleneck_supported`、总体系的
`distributed_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false` 和 `generation_readiness=not_ready` 均保持不变。
唯一下一动作是 `manual_review_scaled_chen_local_relay_tuning_closeout_contract`；不授权
训练、FMF、prior sampling、generation 或 promotion。

## H-LOMA-CBOGT-S1-BOSA：Scaled-Chen Backend Observable Subspace Attribution（2026-09-01）

该 no-update 审计消费 MRA/DCM 的固定 Identity step-5500、Scaled-Chen step-5000、同一
selection panel、native replay 和四格 representation/backend 设计（`I→I`、`I→S`、
`S→I`、`S→S`）。实现提交为 `8ece7fb6f0eb8208e4b50ada87bbfaaf981d648b`，权威运行目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_backend_observable_subspace_v1_attempt_05/`。

validation/audit/compact/panel SHA-256 分别为
`105606ac5886c2374d09d49443faa878f130811181f68ccd758d239e64f4a7d7`、
`1131860efc1500c4807d773eed6e75e49737972eff05e46bbbefea9e9e993baf`、
`67fbf72f2d7775d87f9c415babadc60f3794e4e3eb251a8ccdda0df1ff2ec6be`、
`a11a3c356ce7146bd0f491bc29c87f4219f65131e62af08595d27dae7eb08fc7`。harness、native
replay、平方律/相位 negative controls、冻结状态和资源门通过；`optimizer_created=false`、
`evaluator_calls=0`、`training_authorized=false`、`formal_promotion=false`。

selection 的预注册科学门失败，原因是 representation contrast 的方向和实质差未同时
达到要求，而不是功率混淆或 harness 失败。`R_prop` 为 `I→I=0.9912`、`I→S=0.3256`、
`S→I=1.0086`、`S→S=0.6571`；Identity 相对 Scaled-Chen 的传播差为
`-0.01736/-0.33149`，与预注册的稳定 Identity 优势相反；detector-side 差虽为
`+0.11389/+0.11386`，但不足以替代 representation gate。功率 confound gap 仅
`5.5×10⁻⁵/0`。因此状态为 `valid_failed`、分类为
`backend_observable_subspace_mismatch_not_supported`，confirmation 按合同未执行，不能
形成 backend mismatch 支持性因果结论。

该结果保持 parent `distributed_bottleneck_supported`、DCM 的
`relay_propagation_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false` 与 `generation_readiness=not_ready`。结合前级
geometry/BRAC 结果，当前本地 relay 微调族关闭为
`scaled_chen_v1_local_relay_tuning_family=closed_no_local_correction_supported`；数值
boundary 仍未评估，故唯一下一动作是
`manual_review_high_fidelity_relay_numerical_reference_contract`。不授权训练、FMF、prior
sampling、generation 或 promotion。

## H-LOMA-CBOGT-S1-HFRR：Scaled-Chen High-Fidelity Relay Numerical Reference（2026-09-01）

该 no-update 审计的消费者是项目负责人对 relay 复场 retention 指标是否能支持
“数值实现错配”或仅是坐标/相位口径混淆的人工决定；任何结果都不能授权训练、FMF、
prior sampling、generation 或 promotion。实现提交为
`8835bb228f074408a596118a907248e0f7c9c609`，冻结 Identity step-5500 与
Scaled-Chen step-5000，生产 forward、OLS、formal config、exposure 和数据 panel
均只读。

reference 使用独立 CPU `complex128` 直接 Fourier-series evaluator，未调用生产
transform helper；12 组固定合成 probe 及 Identity/Scaled-Chen native task replay
均通过 complex、amplitude、phase、power、intensity 数值门。动态 ABCD/pupil 也与
formal replay 一致：Identity `A=-2.5,B≈-1.94e-16,C=-14,D=-0.4`，source/target
occupancy `7843/10000`、`7845/40000`；Scaled-Chen `A=-2.5,B≈2.17e-19,
C=-1633.333333333333,D=-0.4`，两侧 pupil 均满。状态 hash 前后相同，峰值 CUDA
`1323107840` bytes，optimizer/backward/evaluator/prior/checkpoint 均为零或未发生。

权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/`；
validation/audit/compact/panel SHA-256 分别为
`248983c83cdd122f4a6aa7cf7f38bf6b415d3f080eb8136bb0a4fdb0b8e1c84f`、
`4045e8e0469a745c637c8dae151d073e5b65e06395bf175297d93245f1031e48`、
`d85e82d9a98fb36568d16c63d0fc1a4e5e3a311a007c512f2b07be7472427e7b`、
`dc75d62ed53a2699aa0949fd8bd8a465f0cec2abe46b64279858a304bde9301b`。

结果为 `valid_failed` / `relay_complex_retention_metric_confound_supported`（Class B），
数值子分类为 `relay_numerical_reference_consistent`；
`relay_numerical_reference_mismatch=false`、`numerical_boundary_not_evaluated=false`。
原始 backend gap Identity `0.6517181`、Scaled-Chen `0.3329323`，analytic-only
canonicalization 后分别为 `0.0081707`、`0.0193076`，gap reduction 分别为
`0.6435474`、`0.3136248`。canonicalization 不进入 physical forward、detector、
MSE 或任何训练/晋级决策，cross-backend 仍只是 pupil-aware diagnostic。

HFRR 不抹除 DCM 的 `relay_propagation_bottleneck_supported`，而是把后续决策收敛为
`manual_review_relay_retention_metric_reinterpretation_contract`。总体系的
`distributed_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false` 与 `generation_readiness=not_ready` 均保持不变。

## H-LOMA-CBOGT-S1-RMRI：Relay Metric Reinterpretation Audit（2026-09-01）

RMRI 是对 HFRR/DCM/MRA/BOSA 的冻结、无更新、同 panel、同 threshold 重解释层。它只在
合法的 decoder-L2/chip → incident complex field relay 段应用 HFRR 的 analytic B=0
canonicalization；canonicalized field 不进入 detector、MSE、training 或 generation，旧
evidence 与历史 classification 不改写。实现提交为
`16896f06b31549120adf6235587619fe2820af83`，权威 attempt-03 位于
`outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/`。

权威 validation/audit/compact/panel SHA-256 为：
`a87769fb638d5bffa1f77bc31c931fd3999c398a23c1a1f17d51d01eb91882eb`、
`d409729f9b111c024c1913cef90c6cd2ebe778efa9990b836a955783f378ba00`、
`cd39f2fed5719572a558be2bbbf139cdb7cc657a08d74c2be4cd8b69b71c35b8`、
`6b7f9225624fa404a0743a417903b27a7113d100315ea9b1804043c63aafb079`。attempt-01/02
只含旧实现绑定的 validation，按 write-once 规则保留但不消费。

原始 replay 锚点全部通过：MRA 六段与历史 retention 一致；DCM Scaled-Chen selection/
confirmation raw relay=`0.65712498/0.60306906`、readout=`0.88675454/0.97353794`；
BOSA selection `R_prop(I→I/I→S/S→I/S→S)=0.99121143/0.32563974/1.00856667/0.65712498`。
canonicalized DCM relay retention 变为 `0.98158839/0.98122396`，canonicalized relay
loss 为 `0.01841161/0.01877604`，且 label/retrieval signal 没有出现 material drop。

因此 DCM 当前解释为 `relay_propagation_bottleneck_reinterpreted_metric_confound`（D1），
历史 `relay_propagation_bottleneck_supported` 和 HFRR
`relay_complex_retention_metric_confound_supported` 都保留；这是“raw complex-field
metric 被解析坐标/二次相位混淆”的 superseding interpretation，不是历史实验无效。
MRA 独立 OLS (`0.81449253/0.80923885`) 与 decoder-internal
(`0.82043698/0.81888384`) 两段仍同时低于 `0.90`，故为
`distributed_bottleneck_supported_canonicalized`（M1）；encoder `slm→pre-OLS`
仍为 `0.91817469/0.92159733`，不能判作 encoder-only 或 decoder-capacity-only。
BOSA canonicalized `R_prop` 为 `1.00012124/0.99160306/1.00003910/0.98158839`，
两项 representation contrast 仅 `0.00008214/0.01001467`，故为
`backend_observable_subspace_mismatch_not_supported_reinterpreted`（B1），confirmation
保持未执行。

RMRI overall=`valid_failed`（科学重解释有效但不构成晋级），唯一下一动作是
`manual_review_relay_metric_confound_closeout_contract`。reconstruction/generative
manifold 与 generation readiness 继续分别为 `false/false/not_ready`；不授权训练、FMF、
prior sampling、generation 或 formal promotion。峰值 CUDA `3626808320` bytes，输出
约 `388685` bytes；optimizer/backward/evaluator/checkpoint/prior 均为零或 false，physical
forward 与历史 evidence 未修改。

## H-LOMA-CBOGT-S3-RMFA：Reconstruction-Manifold Failure Attribution（2026-09-01）

RMFA 是 TRBA closeout 后唯一获准的 no-update bounded frozen recoverability 审计。其消费者是项目负责人判断 Scaled-Chen 的 physical reconstruction mapping 是否错配；不测试 unrestricted representation sufficiency、通用 nonlinear recoverability 或新架构性能。实现与静态 validation 绑定 clean commit `2445d946fd26867d1c9bd98b04f262b50cd0c90e`，只读使用 Scaled-Chen step-5000、固定 MRA selection/confirmation panel，并保留正式 runtime、OLS、checkpoint、target 与历史 evidence 不变。

权威目录为 `outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/`；validation/audit/compact/probe SHA-256 为 `08e6afd841574a5ac6ba126d11af24b560b6f85bf2d5365d42fd229b373625c0`、`e0a53625171144b7aba9dbdc3258ff94ca2686219a80d0725f535d2625c7d468`、`174bdf09c8b5a33896b3a22396f8dfaea6dd22200836f9bbbe46d53b57c403f5`、`25d9eba0d6203027b3a6be00a9d0a5538225510e45c0927f63cf8f78caaf4a09`。

probe 是预注册固定 `16×16` real/imag pooled pre-OLS feature（`input_dim=513`）到 `40000` 维 paired target 的一次性 float64 Moore–Penrose minimum-norm readout，`rcond=1e-12`，fit subset 为 train 前 55000 中每类 100 个、共 1000 个，rank/effective rank=`513`，condition number=`7945.41016660053`。selection 与 confirmation 均未重拟合。selection physical/probe/template/cyclic paired MSE=`0.0620484613/0.0033369721/0.0324799940/0.0799509138`；confirmation 为 `0.0577895157/0.0033930303/0.0277411994/0.0756082982`。相对 physical 改善为 `94.62198%/94.12865%`，instance geometry correlation=`0.97861648/0.97681407`，confirmation fixed bootstrap 95% CI=`[0.0540225357,0.0547624826]`，两个 panel 的 finite/frozen/disjoint/native replay gates 全部通过。

因此 RMFA classification 为 `reconstruction_mapping_misalignment_supported`（R1），唯一下一动作是 `manual_review_reconstruction_mapping_alignment_contract`。该正结果只说明当前冻结 physical mapping 与 representation 存在可恢复错配；不能写成 representation universally sufficient、decoder 已修正或 generation ready。RMFA diagnostic contract 内的 bounded recoverability 为 positive，但 formal/physical `reconstruction_manifold_supported` 仍按上游 MRA 保持 `false`，`generative_manifold_supported=false`，`generation_readiness=not_ready`。

## H-LOMA-CBOGT-S4-RMAL：Reconstruction Mapping Alignment Localization（2026-09-01）

RMAL 的消费者是项目负责人对当前 Scaled-Chen physical reconstruction mapping 是否存在可恢复错配的决定；拒绝决定是不能把 bounded probe 的 stage 结果写成通用 representation sufficiency、架构修正或 generation 授权。实现与 validation/audit 绑定 clean commit `5b274b926474784a1ad49744e75190afff31747c`；权威 attempt-01 为
`outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/`。

validation/audit/compact/probes SHA-256 为：
`affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec`、
`5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa`、
`c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137`、
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。固定一次 fit 的 probe 为 float64 Moore–Penrose minimum-norm (`torch.linalg.pinv`, `rcond=1e-12`)，不含 optimizer/backward/refit/regularization；RMFA positive-anchor replay、finite、native replay、冻结、stop-rule 与 resource gates 全部通过。

固定搜索顺序为 `cmos_intensity → cmos_incident_complex_field → decoder_layer_2 → decoder_input → modal_complex_100 → pre_ols`。首个 `cmos_intensity` stage selection physical/probe/template/cyclic=`0.0620484613/0.0098129278/0.0324799940/0.0799509138`，relative improvement=`84.18505%`；confirmation=`0.0577895157/0.0101852901/0.0277411994/0.0756082982`，relative improvement=`82.37519%`，bootstrap 95% CI=`[0.0472728200,0.0479231887]`。因此 RMAL 为 `passed` / `bounded_recoverability_established_at_detector`，`furthest_downstream_recoverable_stage=cmos_intensity`，其余五段按预注册 stop rule 跳过。

formal/physical `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready` 保持；治理 flags 全部保持 no-update（optimizer/backward/evaluator/checkpoint/prior/generation/physical-forward modification 均为零或 false，`mechanism_evidence_only=true`）。唯一下一动作是 `manual_review_RMAL_recoverability_result`，不授权训练、FMF、prior sampling、generation、candidate export 或 formal promotion。

本结果把 Scaled-Chen v1 attribution family 正式记录为 `closed_task_relevant_bottleneck_not_supported`；MRA/DCM/HFRR/RMRI/TRBA 历史与 superseding interpretation 均保留。全程 optimizer/backward/evaluator/checkpoint/prior/generation/physical-forward modification 均为零或 false，不授权任何训练、FMF、prior sampling、generation canary 或 promotion。

## H-LOMA-CBOGT-S5-ZCSA（2026-09-01）

在起始 HEAD `0dead1b73655b3556450cedbb5d0a2c579385209` 上完成零列语义审计，实现提交
`1f5e553f6c7ad80217427ab83f5669e7b9758acb`。新增 producer、YAML、review contract、测试和 CLI
路由；上游 RMAL/RWSA 产物、冻结 probe、target renderer、checkpoint、OLS、detector 与 formal runtime
均未修改。冻结 probe SHA256 为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`；旧 RWSA attempt-02 的
`invalid_harness` 状态与身份哈希保持不变。

正式 attempt `outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/`
的 validation/audit/compact SHA256 分别为
`d422f1e31e8298f9662896f832b207ad896fb619c5a656bccf54b099cd419954`、
`0cba5990462e9f8cc238885b14c90c907a7c6f085877b6c8ad0c8b95cbf0bfb8`、
`772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6`。结果为
`passed / zero_column_semantics_attribution_completed`：active=`33502`，zero-W=`6498`，且
全部 zero-W 均为 `target_zero_static`；其余四类均为 0。fit-zero→selection/confirmation active
分别为 `1495/2097`；zero-W b exact-zero=`6498`；空间 pixel/edge/interior=`6498/660/5838`，
4-connected components=`2`，最大/次大=`5105/1393`。

该结果是 target-only、mechanism-evidence-only 的零列归因，不支持 reconstruction/generative
manifold：`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`，
并保持 `closed_task_relevant_bottleneck_not_supported`。全程无 optical replay、model load、
optimizer、backward、evaluator、checkpoint、prior 或 generation。唯一下一动作为

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

LSDA 已在起始 HEAD `bd967f64ffb046677fa4204e81e911b226047e6e` 上完成，实现 commit 为
`bed235283d1fa47f4a5a38d41368d87ce2b5d218`。这是 frozen-W、CPU-only 的 descriptive spectral/operator attribution；消费者是项目负责人审查冻结 RMAL detector readout 的谱、source/target mode 与 bias 结构，不能由此产生 readiness、架构或训练/晋级授权。

权威 attempt 为 `outputs/20260901_v2/runs/lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/`，validation/audit/compact SHA256 为
`3f24dcc7d56e098ec778998bb8ddc7871973131f1d7269c86da729395426c2f2`、
`97842d5707541516c79adb87c3a1e609ee533156895908c9e99c015cde0f9b13`、
`e281daf9a4a62f85828a3bc8055ce044027d0c0fb1af0a522febcf3c2b68185b`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。结果为 `passed / linear_readout_spectral_attribution_completed`。

谱分析保持 full/active/inactive=`40000/33502/6498`，Gram=`[256,256]`，CPU float64；前五 singular values=`12227.8671591/10625.3503547/8724.4504853/8126.3713707/7844.9489846`，top-16 cumulative=`0.8441783201`，effective/stable telemetry=`21.88110383398559/21.881103833985602` 与 `5.528997319328641/5.5289973193286395`。full-vs-active identity 保持，最大 eig absolute/relative difference=`4.470348358154297e-08/8.328885261127632e-13`，最大 sigma difference=`2.900790718740609e-12`。K=16 只作 descriptive preview；global reference 为 RMAL fit panel 的 1000 个 unlabeled streaming targets，未保存 target bank；bias norm²=`114.5276114768678`，preview projection=`21.21861242530154`，preview 外 residual=`93.30899905156626`。

全程 no-update，定向/相关/全套 pytest 为 `18/101/1232 passed`（全套另有 12 deselected、25 subtests），compileall/Ruff/diff-check 通过。`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported` 保持；唯一下一动作是 `manual_review_LSDA_spectral_result`。

## H-LOMA-CBOGT-S6-LRRC：Low-Rank Readout Recovery Confirmation（2026-09-02）

LRRC 是 LSDA 之后唯一执行的 frozen-W、no-refit functional recoverability confirmation。消费者是
项目负责人判断预注册 powers-of-two candidate family 内的首个 confirmed rank；拒绝决定是不能
把谱能量或 rank-256 anchor 写成数学最小秩、intrinsic/physical/architecture rank、训练、生成或
formal promotion。实现/config/review/test/CLI 的 clean commit=`2a061db61910a847118e38663d05c05bd9447063`，
上游 RMAL/ZCSA/RWSA-ACTIVE/LSDA、probe、renderer、checkpoint、OLS、detector、formal runtime
与历史 evidence 均未修改。

权威 attempt-04 为 `outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/`；
validation/audit/compact SHA256=`b305c38dc3c774ff995c98907977d67cfc396d56009b8a88f726e214f9efa004` /
`5ffcb3550e0a1d635f70cdf965d12c93ded6e99fadf8f0f6ef0cda829b4a6b58` /
`4ae704104accd8cebfcb0e3c19dba7deb90b4febf38bace3c7e578bd484951b8`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

candidate family=`[1,2,4,8,16,32,64,128,256]`；rank-256 anchor first，selection/confirmation 均通过。
rank `1,2,4,8,16,32,64,128` 按注册顺序全部 selection 失败，confirmation 未执行；因此
`first_confirmed_candidate_rank=256`、`previous_registered_candidate=128`、
`rank_256_reused_as_fallback=true`，无第二次 rank-256 run。operator factorized-vs-full max residual
为 selection/confirmation=`2.4333037798e-06/2.5858223358e-06`，probe replay delta=`1.1276390065e-10/7.8544505795e-10`。
LSDA candidate-boundary relative gaps（1/2/4/8/16/32/64/128）为
`0.131054483/0.178902324/0.034630756/0.105670493/0.104494812/0.061605380/0.043964196/0.052600464`；
full-spectrum smallest gap 是 rank-36=`0.0025793411`，无新 threshold，仅保留 descriptive ambiguity。

结果为 `passed / low_rank_readout_recovery_confirmation_completed`。CMOS-only exact panel replay、
state/no-update、streaming/no-bank、anchor 与 resource gates 全部通过；`cmos_only_replay_executed=true`、
`optical_replay_executed=false`，optimizer/backward/refit/decoder/generation/physical-forward modification
均为 false/zero。flags 仍为 `reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported`。
唯一下一动作是 `manual_review_LRRC_result`；不得训练、candidate export、generation 或 promotion。

最终核验：LRRC 定向 `26 passed`，相关回归 `130 passed`，默认 Windows-native 全套 pytest
`1258 passed, 12 deselected, 25 subtests passed`；compileall、Ruff、diff-check 通过。历史
attempt-01/02/03 按 write-once 规则保留，attempt-04 为当前权威；结果/状态 commit 后 refresh bridge
并检查 status，不自动上传。

## H-LOMA-CBOGT-S6-DSFA：Data-Weighted Spectral Functional Attribution（2026-09-02）

DSFA 在 clean implementation commit `d032738c0f02af5ada85554224e045d238416e0a` 上执行，消费
RMAL frozen probe、ZCSA/RWSA-ACTIVE/LSDA 与 LRRC attempt-04。它是 no-update、descriptive-only
的 data-weighted mode attribution：按真实 selection/confirmation detector data 统计每个
LSDA mode 的 activation、output energy、alignment benefit、self-energy penalty 与 signed MSE
gain，不重新 fit、训练或搜索 rank；因此不把结果写成 reconstruction sufficiency 或最低 rank。

权威 attempt 为 `outputs/20260902_v2/runs/lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/`，
validation/audit/compact SHA256=`ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe` /
`2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e` /
`8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3`，结果为 `passed / data_weighted_spectral_functional_attribution_completed`。
selection bias/full MSE=`0.09840880738327583/0.009812927922827133`，confirmation
`0.08768781986771307/0.010185289342506197`；B256 operator-energy share=`0.0007625489`，
data-weighted output-energy share=`0.3517092/0.3451470`（selection/confirmation）。

LSDA q/operator、两 panel additive identity、LRRC prefix replay、state/no-update、streaming/no-bank、
finite 与 resource gates 全部通过；peak CUDA=`2274099712` bytes，wall time=`225.532 s`。所有
治理 flags 继续禁止 training/generation/promotion，唯一下一动作是 `manual_review_DSFA_result`。
DSFA 定向测试 `10 passed`，compileall/Ruff/diff-check 通过；上游 evidence 与历史 attempt 均保持 immutable。

## H-LOMA-CBOGT-S6-DMSA：Detector Data-Metric Spectral Attribution（2026-09-02）

DMSA 在 clean implementation/result commit `9818e87441eb2c86eef06c18c1aa646f3e57dea5`
上完成，消费冻结 RMAL `W=[256,40000]`、`b=[40000]`、LSDA exact ordered CPU float64
spectrum、LRRC attempt-04 与 DSFA attempt-01。它是 no-update、descriptive-only 的
detector data-metric attribution：以 population `N` 流式记录 `S=C+mu muT`，并比较
raw operator、second-moment 与 centered-variance 的 spectrum、energy decomposition、
effective/stable rank；不训练、不 refit、不做 rank search 或 reconstruction truncation。

权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/`；
validation/audit/compact SHA256=`f0eebcb978851b9ab58453e1b28e920eddc93e86a2b82133af0bd7ec8e8ce93d` /
`c9cc448f1b58040d45b8b77c9cfaa9facd536cdcf40d0b4f152efe42e16411fe` /
`39564f0ce3973bd738d196191a06c7990407f9b7b6ab7e6e8a2eb3366dc2d0ac`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01 为
validation-only invalid harness，保留且不消费。

结果为 `passed / detector_data_metric_spectral_attribution_completed`，15/15 audit gates
通过。selection/confirmation 的 mean/variance energy fractions 为
`0.4019555898/0.5980444102` 与 `0.3704005497/0.6295994503`；second-moment
effective/stable rank 为 `14.8728108/2.2436135` 与 `17.4264177/2.4233991`，
centered-variance 为 `36.6563260/8.1270337` 与 `39.8650006/8.4787742`。正式 replay
为 exact-panel CMOS-only no-update streaming/no-bank，peak CUDA=`2274099712` bytes，
state/OLS unchanged；`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported` 保持。唯一下一动作是
`manual_review_DMSA_result`；不得训练、decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S6-MCFA：Detector Mean-Centered Functional Attribution（2026-09-02）

MCFA 在 clean commit `069cfa4b3aeda3943d754d739daf6daeba2a54a6` 上完成，消费冻结 RMAL
`W=[256,40000]`、`b=[40000]` 与 LSDA/LRRC/DSFA/DMSA 及 ZCSA/RWSA-ACTIVE 的已验证身份。
它是 fit/selection/confirmation 独立 panel 上的两遍 CMOS-only、no-update、streaming
attribution，消费者是项目负责人区分 `mu W` 与 `(z-mu)W` 的功能 MSE gain；拒绝决定是
不得把结果升级为 dominance、reconstruction sufficiency、training、generation 或 promotion。

权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/`；
validation/audit/compact SHA256=`35ad479aa931cb2e605bb9b582306549512106a5c2dcd070bfde35a2e552e7f1` /
`f69664cf9da1d61e90742a09614c4535388402f9228e401d2451e3ebeeefdc0b` /
`e95c072bde8306dd3f7cb8d6c5af1a671a778cede8f84c285c450af854059494`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01 为
validation-only failed harness，按 write-once 规则保留且不消费。

fit/selection/confirmation 样本数为 `1000/1000/5000`，共 14000 visits；mean/centered
output-energy fractions=`0.3829179441/0.6170820559`、`0.4019555898/0.5980444102`、
`0.3704005497/0.6295994503`，mean/centered/total functional gains=`0.0313323622/0.0504929026/0.0818252648`、
`0.0396463978/0.0489494817/0.0885958795`、`0.0316138044/0.0458887261/0.0775025305`。
audit 为 15/15 gates，peak CUDA=`2274099712` bytes、wall=`183.266 s`；所有 energy/MSE/
gain/cross-identity、state/no-update、no-bank、finite 与 attribution boundary gates 通过。
flags 仍为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；完整 pytest 为 `1310 passed, 12 deselected,
25 subtests passed`。唯一下一动作是 `manual_review_MCFA_result`，不得训练、decoder、
candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S7-MPCDAR：Mean-Preserving Centered Data-Aware Readout Recovery（2026-09-02）

MPCDAR 在 clean implementation commit `b59172750d0c88bbc9f7426ab7a21a1d63639f57` 上完成，
消费冻结 RMAL `W=[256,40000]`、`b=[40000]`、MCFA fit moment contract 与 LSDA/LRRC/DSFA/DMSA
及 ZCSA/RWSA-ACTIVE 身份。它是 no-refit、no-training、CMOS-only 的 mean-preserving centered
data-aware truncation recovery confirmation；真实消费者是项目负责人审查预注册 rank family
内的 bounded recovery，拒绝决定是不得将其升级为 minimum/intrinsic rank、一般 sufficiency、
decoder、training、generation 或 promotion authorization。

权威 attempt-03 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/`；
validation/audit/compact SHA256=`b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2` /
`43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785` /
`9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9`；probe SHA256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01/02 仅保留
validation-only 或执行前停止记录，不消费；没有覆盖任何旧 evidence。

fit replay 使用 `N=1000`、float64 population moment contract，`mu`/`C` hash 均与 DMSA
匹配，moment identity residual `2.9855e-16`，PSD raw minimum eigenvalue=`3.23645e-09`，
material/tiny negative count=`0/0`。A_fit data-aware spectrum 的 cumulative energy at
`1/2/4/8/16/32/64/128/256` 为 `0.118309/0.207018/0.355819/0.541212/0.715164/0.868603/
0.963351/0.994898/1.000000`，effective/stable rank=`37.184239/8.452439`；registered DA
boundary gaps 只作 descriptive telemetry，未增加 degeneracy threshold。

rank-256 positive anchor 首先通过；V orthogonality max residual=`1.70464e-12`，A_fit/W
projector max residual=`4.53214e-14/5.68008e-11`，selection/confirmation full prediction
identity 与 RMAL gates 均通过。随后按 `[1,2,4,8,16,32,64,128]` 顺序执行 data-aware
selection：实际评估 `1,2,4,8,16`；rank 1/2/4 selection failed，rank 8 selection passed
但 confirmation failed，rank 16 selection+confirmation passed，故
`first_confirmed_data_aware_candidate_rank=16`、`previous_registered_candidate=8`，未重复
rank 256。selection/confirmation rank-16 probe MSE=`0.0210726835/0.0212836881`，相对改善
=`0.660383/0.631703`。

matched control 对每个实际 evaluated rank 同流计算 old LRRC raw、mean-preserving raw 与
mean-preserving data-aware 三路 MSE；A/B/C 不参与 stop rule 或 confirmation authorization。
正式结果为 `passed / mean_preserving_centered_data_aware_readout_recovery_completed`，audit
14/14 gates 通过；CMOS-only streaming、state/no-update、no-target-fitting、finite/resource
和 no-bank 条件通过，peak CUDA=`2274099712` bytes、wall=`183.266 s`。治理 flags 保持
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是 `manual_review_MPCDAR_result`，
不得训练、candidate export、generation 或 promotion。
