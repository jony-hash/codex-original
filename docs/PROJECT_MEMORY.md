# codex-original 项目记忆

更新时间：2026-09-10

## 目的与事实优先级

这是 `codex-original` 的跨 ChatGPT/Codex 交接入口。它只保存稳定导航、当前边界和
交接规则，不替代正式证据、实验报告或历史 transcript。云端 ChatGPT 项目使用它的
受控上传副本；本文件本身仍以本地 Git 版本为准。

研究交接的风险层、HARD_BLOCKER、FACT/MODEL_ASSUMPTION/DESIGN_DECISION/AUTHORIZATION
词汇、semantic evidence alias 与 one-outer-Gate 规则统一见
[`delta_one_gate_short_prompt.md`](operations/delta_one_gate_short_prompt.md)。本文件只保留
稳定入口与 durable decision，不复制 child hash 表。

`RESEARCH_GOVERNANCE_SIMPLIFICATION_V1_ESTABLISHED`：本轮治理只改变 research planning、
evidence handoff 与 provenance presentation；随后 evidence I/O 的局部异常语义修复不改变公共
函数签名、formal/high-risk evidence、training authorization、data firewall、checkpoint validation、
formal runtime 或历史 artifact。

## 唯一活动研究主线：LightGen 输入与 OLS 对齐的语义生成原型

用户于 2026-09-10 明确研究目标：学习光学系统的语义流形，尽快得到可检查的初步生成效果；
允许衍射层架构有项目差异，但输入和 OLS 处理必须以陈一同等人的 LightGen 原文为对齐依据。
这是目标与设计约束，不是已经完成的论文复现，也不是训练或物理架构变更授权。

| 路线 | 地位 | 如何使用 |
| --- | --- | --- |
| LightGen 输入/OLS 对齐的 OLS-native 语义生成 | 唯一活动研究主线 | 以真实光学输出检验语义生成；先明确训练输入、生成输入和完整 OLS 接口 |
| N3 Gaussian copula 与复 OLS 兼容性研究 | 主线已有候选组件及证据 | 保留成果；其项目自建假设不得自动写成论文方法，是否纳入对齐实现需审查 |
| 正式 v2 `label + 8D` / field identity OLS | 保留的正式兼容基线 | 解释现有 CLI/代码合同，不定义当前研究目标，不自动恢复训练 |
| Scaled-Chen 重建、rank-16 电子读出、旧 S1–S3 controller | 历史证据或已关闭路线 | 可作对照；旧下一步与 `mainline_state=closed...` 只约束原路线 |
| v1 LDM teacher | 历史复现运行时 | 与当前 teacher-free 研究分离 |

### 输入和 OLS 的稳定对齐约束

- 学习/重建路径：图像 → 振幅 SLM → optical encoder → 模式耦合 OLS → optical generator → CMOS。
  生成路径可直接从表示 OLS 的输入进入 generator；不把旧 `label + 8D` controller 设为必要前置。
- OLS 区分完整复系数、振幅和强度；保留模式投影、相对相位与模式重发。
  端口传输 `T=I` 不等于整幅光场原样通过的 identity OLS。
- 论文明确的事实与项目假设分列：`sqrt(intensity)` 的精确输入预处理、生成时的相位写入、
  缩尺度 LP01 参数、逐端口归一化、KL 统计轴及参数、经验 copula 和独立相位继承，
  均不得在缺少原文/作者实现依据时称为已对齐。
- 论文 BOGT 在训练中结合分布约束、重建与光功率约束；冻结 encoder 后拟合分布不自动等同 BOGT。
  具体未披露实现保持 unknown；不能为迅速出图静默换成软件解码或不可实现的任意复场注入。
- 真实 OLS 重放、生成的新样本、raw CMOS 和电子读出分别报告；类别可识别、视觉质量、
  多样性与新颖性不能相互替代。图像验收及执行预算须在相应实验合同中明确。

论文依据为 [已登记补充材料](../research_assets/lightgen/science_adv7434/paper/sciencesm.pdf)
第 2–3 页、第 6–9 页、第 10 页及图 S2；使用前按其 [manifest](../research_assets/lightgen/science_adv7434/manifest.json)
确认版本。当前 numerical results 与既有实验下一动作只维护在 [当前状态](CURRENT_STATUS.md)；
[当前交接](status/current_handoff.md) 负责新目标与既有执行合同之间的差异。

## 当前状态入口

本文件只维护稳定目标、长期决策和导航；当前正式基线、活动研究快照、数值结论、阻塞以及
允许动作或停止边界只在 [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 与
[`status/current_handoff.md`](status/current_handoff.md) 维护。下面带日期或明确标为历史的段落保留
决策 provenance，不是当前执行指令；其中的历史“下一步”不能覆盖上述两个当前入口。

<details>
<summary>历史 N3 导航快照（2026-09-08；不代表当前状态或下一动作）</summary>

## OLS-native N3 research navigation

活动分支 `research/lightgen-ols-native-semantic-generation-v1` 的 N3 研究链包括 producer
data authority、model-order/fit/validation、mismatch attribution、score-geometry resolution 和
selector calibration。最新 V2 classification 为
`N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`；selector refinement 在 MIXED 边界关闭，不创建
V3，随后在既有 development pool 上的两候选 minimal producer comparison 为
`N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED`，没有候选冻结；N4/N5 与 generation 的 formal
flags 不变。其后仅定义了 shared-covariance unsupervised local-location mixture contract，候选
`K={2,4,8}`；contract 已执行一次 development-only selection，但三个 K 均未通过五折 compatibility，
结果为 `N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED`，不能解释为 family supported 或 K selected。
随后定义了保留该 local partition、但为每个 hard-assigned component 引入 analytic shrinkage
covariance 的 contract；随后严格执行了一次 development-only selection。Stage-1 的 15/15
candidate-fold 均收敛，但 Stage-2 的 `n_k<2` 在 K=2/4/8 分别造成 1/2/4 个 fold failure，
其余 operational folds 也均未通过 frozen primary/covariance/norm compatibility；结果为
`N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE`，没有 selected K，不能解释为 covariance
heterogeneity supported 或 producer family established。

随后定义了唯一的 empirical-marginal Gaussian-copula candidate：每个 `R2_AMPLITUDE` 坐标使用
fit-only plotting-position empirical quantile，ties 使用 deterministic average rank，Gaussianized
scores 使用 `n-1` 标准化与同一 analytic shrinkage dependence family，sampling 直接使用
`R_shrunk`。Contract SHA 为
`3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4`，claim 仅为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED`。随后唯一一次
`EXECUTE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_V1` 在同一 frozen five-fold
development evaluation 上完成：5/5 folds 的 computation、positive-definite 与 sampling
integrity 均通过，但只有 2/5 folds 通过完整 compatibility，因此结果为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`。marginal-vs-joint decomposition
为 `MIXED`，不能把结果解释成 marginal population truth、dependence solved 或新 producer
family supported；report SHA 为
`78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d`。

该 one-shot 只使用 development 数据，sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`；final all-1000 fit、confirmation、novel OLS、generation、
CUDA、optimizer、backward、training 均未执行。后续必须先经独立 planning Gate，不能把本次
相对历史 family 的改善替代 frozen all-fold compatibility。

随后，针对 attribution 因缺少历史 generated-bank tensors 而产生的 blocker，完成了唯一一次
严格 deterministic recovery。Recovery semantic authority 为
`FROZEN_N3_COPULA_GENERATED_BANKS_V1`；60/60 banks 与原 copula report 中的 q hashes 完全相等，
manifest SHA 为 `4ed6ac4839cdb075f98d07be3162ac560ea29cc260fcbd0e272a342aed04297c`，tensor
artifact SHA 为 `6b9da40cf868fb9033e5c45aa05778c27199867430d2ec5627b36875d284b42b`。这只是历史
sample identity recovery，不是新 evidence、producer rerun 或 attribution result；sealed
confirmation 仍未 materialize。后续若继续，只能由原
`RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` read-only consumer 消费该 authority。

这些稳定边界和当前入口分别见 [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 与
[`status/current_handoff.md`](status/current_handoff.md)；所有日期化事实、结果、被取代的下一步和
machine-evidence 路径集中见 [`experiment_history.md`](status/experiment_history.md)。本文件不复制
逐次实验正文或 child hash 表，也不把历史条目解释为执行授权。

Fresh N3 producer confirmation requires a sealed source. The current reservation,
manifest identity and authorization remain in [`CURRENT_STATUS.md`](CURRENT_STATUS.md),
[`status/current_handoff.md`](status/current_handoff.md) and the machine manifest;
this stable memory records only the durable boundary: the minimal A/B development comparison and the
frozen unsupervised local-location mixture selection did not support a producer family, reservation
is not execution authorization, and no confirmation, final producer fit, OLS or generation is opened.

</details>

## S3 历史研究导航

2026-09-07 的 controller semantic-geometry data authority、one-shot canary、frozen S3 consumer
与 postmortem 已集中记录在 [`experiment_history.md`](status/experiment_history.md)。本文件只维护
S3 的长期边界：`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`、`KEEP_8D_PROVISIONALLY`、
one-shot repair line closed、S4 blocked。日期化证据和历史“下一步”不构成当前执行授权。

## Windows 工作区与运行边界

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

## 正式兼容边界与历史重建主线收口

- 当前 formal identity 链的有效结果是 `closed_historical_valid_failure`；它没有
  `formal_promotion` 权限，也不能授权 manual review、candidate export、memorization、
  utility、dataset admission 或 accepted manifest。
- prior-native、direct-generator、quadrature、fiber、FMF 长训练和 formal promotion
  等已关闭或未授权路线不能因为 ChatGPT 的讨论而重新打开。
- 所有科研执行仍需遵守 `AGENTS.md` 的 long-training gate；记忆文件本身不提供训练授权。
- 正式 reconstruction mainline 已完成 `FINAL_MAINLINE_CLOSEOUT`：H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ 为
  `passed / frozen_rank16_float32_runtime_performance_characterized`，仅完成冻结 S9 rank-16
  float32 readout 的 CPU-only runtime/performance characterization。`mainline_state=`
  `closed_at_current_evidence_boundary`，该正式主线当前没有下一科学实验；OLS-native 研究线的
  活动状态以本文件顶部和 current handoff 为准。未来若用户另行授权，独立
  任务名为 `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`。S9V、S9、S8、S7、MCFA、DMSA、DSFA 与更早
  identity 保持不变。relay family
  `scaled_chen_v1_local_backend_relay_attribution_family=closed_metric_confound_reinterpreted`
  保持。
  若交接文档更新，以最新的 `CURRENT_STATUS.md`、`status/current_handoff.md` 和 hypothesis
  ledger 为准。
- 本次工作流治理允许的实现动作是维护记忆入口、项目文档和环境入口；不修改训练代码、
  `outputs`、WSL、SQLite 或历史 transcript。只有用户明确需要离线上传时，才生成并验证
  受控 Sources 包。

## FINAL_MAINLINE_CLOSEOUT：稳定导航结论（2026-09-03）

Authority chain: `RMAL → S7 MPCDAR → S8 F64 R16-IQ → S9 F32-NQ → S9V full-path visualization
→ S10 runtime characterization`. Each stage consumed only its named immutable parent; earlier
or failed attempts remain write-once history and are not consumed in place of later authority.

The current evidence boundary supports that task-relevant information survives the existing
optical path through CMOS and primarily supports mapping/readout misalignment as the reconstruction
failure. Simple locality/raw Euclidean W-SVD is inadequate; useful mean/DC material is present;
centered detector information is better represented by a fit-derived data-aware basis. Registered
MPCDAR rank 16 is the first confirmed candidate in its preregistered powers-of-two family, not a
minimum, intrinsic, or true rank-16 claim.

Frozen readout formula: `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`.
The current S9 F32 artifact is `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, `V16[40000,16]`;
scalar count `684352`, raw tensor bytes `2737408`, serialized bytes `2739448`. S8 F64 integration
and S9 F32 numerical qualification are frozen qualifications; S9V is descriptive-only and adds
no scientific Gate. The permitted architecture wording is `optical front end → CMOS intensity
sensing → lightweight frozen electronic rank-16 readout`: hybrid optical-electronic reconstruction,
not a native CMOS optical image.

S10 is recorded only as a CPU readout characterization in the registered Windows environment:
Python `3.11.16`, PyTorch `2.5.1`, Windows observed `10.0.26100`, torch threads `8/8`, logical
cores `16`, formal CPU string `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; expected
Windows `10.0.26200.0` did not match. Batch mean latency B=`1/8/32` was
`0.5601775/0.8914525/3.277882 ms`, mean throughput was
`1836.8588342/9338.7841702/9874.3015643 samples/s`, and analytical MAC ratio was
`15.89825119236884`, not measured speedup.

Limitations remain: native optical target-like CMOS reconstruction, real physical CMOS validation,
Raspberry Pi benchmarking, sim-to-real detector alignment, end-to-end hardware latency, GPU
runtime, full-W runtime, quantization, generative manifold, and generation readiness are not
established. `reconstruction_manifold_supported=false`, `generative_manifold_supported=false`,
`generation_readiness=not_ready`, `training_authorized=false`, `formal_promotion=false`, and
`closed_task_relevant_bottleneck_not_supported` remain fixed. Future tracks are independent:
real CMOS/Raspberry Pi deployment, sim-to-real alignment, optional separately justified optimization,
newly authorized native optical redesign, and separate generative-manifold/generation evidence.

The exact evidence/artifact registry is maintained in
[`CURRENT_STATUS.md`](CURRENT_STATUS.md), [`status/current_handoff.md`](status/current_handoff.md),
the hypothesis ledger, and the dated experiment history; this memory file intentionally does not
duplicate the full hash table. Final bridge validation is `status=invalid` with
`bridge_recovery_blocked=true` because `refresh` stopped at the existing invalid manifest
`C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`;
no automatic upload occurred.

## 历史研究导航

2026-08-30—09-03 的 Scaled-Chen 几何、readout 与主线 closeout 事实统一见
[实验历史](status/experiment_history.md)；本文件只保留 durable architecture、Git/迁移规则、
跨产品职责与记忆更新协议。

## Git 与验证状态

- 当前分支、HEAD、dirty paths 和最近验证结果以 `CURRENT_STATUS.md` 和运行时 Git
  状态为准；这里不复制容易过期的完整提交清单。
- 注销前审计提交 `75d0584c379b145b0f0eb45ece14fda160741f23`、历史 C: Git bundle
  和其完整性核验均属于迁移记录；具体位置、摘要和当时的验证结果见
  [Windows-native migration](migration_windows_native_20260830.md)。它们不能被解释为当前
  branch 的 clean-head 或全量测试保证，也不能把本地 bundle 伪装成远端备份。
- 当前验证结果只写入 [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 与
  [`status/current_handoff.md`](status/current_handoff.md)；历史全绿基线不自动继承为当前全绿。
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

交接使用 `MODE=VERIFY|PLAN|IMPLEMENT|REPORT`。`MODE=IMPLEMENT` 表示请求范围已经批准；
Codex 只执行一次 bounded preflight，随后执行、在原权限内缩小/跳过，或返回 `BLOCKED`/差异报告，
不得扩大研究问题、统计门槛、正式权限或计算预算。完整快照集合和发布资格见操作规范。

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
- 若为 IMPLEMENT：请求范围已批准；一次 preflight 后可执行、在原权限内缩小/跳过，或 BLOCKED，不静默扩大范围。

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

## 历史研究导航（续）

更早的 S0–S10、I0、T1 等研究记录均按实验身份和日期集中在
[实验历史](status/experiment_history.md)；本文件不再复制逐次结果、完整 SHA 清单或历史下一步。
需要恢复某项研究时，先从历史条目定位 authority，再回到当前交接确认是否有新的授权。
