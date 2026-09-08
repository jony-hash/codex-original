# 当前项目状态

更新时间：2026-09-08

Git 基线：`research/lightgen-ols-native-semantic-generation-v1`，HEAD
`848083ee5e2ee2f6d0e3f0630fc6eeebcdb30f74`；本轮工作树保持 dirty，未自动提交。

## 当前决策摘要

正式 reconstruction mainline 已在当前 evidence boundary 收口；独立的
OLS-native semantic-generation research line 仍在活动分支
`research/lightgen-ols-native-semantic-generation-v1` 上，但不构成 formal v3。
N1/N2 已完成，N3 producer baseline 与 selector calibration 已完成但未支持新
producer family，N4 未决，N5 与 generation blocked，N6 保持独立冻结。

最新 N3 selector calibration V2 classification=`N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`。
冻结 selector 在独立 synthetic confirmation 上 NULL=`0/200` false positives，
partition=`100/100`，radial=`100/100`，nonlinear=`82/100`，local=`78/100`；local
未达到预注册 `0.80` correct-rate target，故不能宣布 calibrated。真实 N3 attribution
未执行，不能重新解释真实 N3 failure，也不能授权新模型族、N4、generation 或 formal
promotion。当前 flags 仍为 `training_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false`；本 selector refinement line 在 MIXED 边界关闭，不创建 V3。
对应 machine evidence 为
[`n3_attribution_selector_v2_calibration_report.v1.json`](../outputs/20260908_v2/runs/lightgen_n3_attribution_selector_v2_calibration_attempt_01/n3_attribution_selector_v2_calibration_report.v1.json)，
frozen spec SHA=`9ee5b7ad64221b323f1b182c2f4d74edddfad89599bdf30b352ef33cc5b4c25b`，
report SHA=`77c0a9f632b6870579a338d84dd6650530849fa6b74892acb1345567ffa8dae5`。

本文件后续带日期或明确标为历史的段落是不可覆盖的历史事实；其中的“下一步”只表示当时的状态，
不覆盖本摘要。详细数值与输入绑定由对应 machine evidence 和 experiment history 维护。

## N3 sealed confirmation source reservation（2026-09-08）

`DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1` 已完成，classification=
`N3_SEALED_CONFIRMATION_SOURCE_RESERVED`。Manifest 位于
[`n3_sealed_confirmation_source_manifest.v1.json`](../outputs/20260908_v2/runs/n3_sealed_confirmation_source_v1_attempt_01/n3_sealed_confirmation_source_manifest.v1.json)，
状态为 `RESERVED_NOT_MATERIALIZED`，manifest SHA-256 为
`6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e`。

这是 fresh balanced confirmation source reservation（500 rows、50/class、适用 overlap=0），
不是 OLS、producer、rank/family、selector 或 generation 结果。它只允许未来已冻结 N3
producer confirmation consumer 一次性消费；完整选择规则和 firewall 输入以 machine manifest
为准，日期化 provenance 见 [experiment history](status/experiment_history.md)。

## 本轮文档与研究导航治理（2026-09-08）

本轮先完成治理/导航精简，再修复一个不改变公共合同的 evidence I/O 防御性问题。文档/导航变更
覆盖 `AGENTS.md`、`README.md`、`docs/README.md`、`docs/CURRENT_STATUS.md`、
`docs/PROJECT_MEMORY.md`、`docs/status/current_handoff.md`、`docs/status/capacity_map_hypothesis_ledger.md`、
`docs/status/experiment_history.md`、`docs/status/research_governance_state.json`、
`docs/repository_taxonomy.md`、架构/迁移/桥接和输出操作文档，
以及线程可视化目录中的 `research-map.html`。代码变更仅为
`src/ldm_project/v2/evidence/io.py`、其回归测试和
`scripts/research/resolve_n3_score_geometry_attribution.py` 的解释文本一致性修正。
主线推进4已留下的 V2 calibration consumer
`scripts/research/calibrate_and_confirm_n3_attribution_selector_v2.py` 及其 machine evidence
本轮未修改，只作为最新状态的只读输入。
治理/审查部分未修改恢复 outputs、`_wsl_restore_metadata_v4`、外部 tar/index/manifest、既有
evidence JSON、N3 阈值/selector 机制、训练代码、CLI 或公共函数签名；上方 reservation
manifest 是独立当前 run 的唯一新增 machine artifact。

本次根合同收缩只移出已有操作/架构文档中的重复步骤，保留执行、失败语义、运行时和 outputs
不可变边界；`finite_crop`/`transparent_identity` 语义归档于 v2 架构文档，正式光场 provenance
字段归档于 output artifact notes。没有删除历史实验、checkpoint、evidence 或恢复元数据。

只读验收：`git diff --check` 通过；restore state 为 `163/163 complete`，声明源文件
`33117`、源字节 `240556414616`、plan/path mapping `46/46`；HTML JavaScript 语法检查通过，
并通过临时 localhost 预览验证了研究卡片、探索/自测切换、7 题反馈、重新开始和键盘焦点。
evidence I/O 定向回归为 `23 passed`，Ruff 定向检查通过；默认全套 pytest 已执行，结果为
`1609 passed, 12 deselected, 25 subtests passed, 5 failures`。失败项属于既有主线绝对导入布局、
旧分支预期、历史 controller authority 预期和长临时路径环境问题，未改变本次 I/O 定向结果；
本轮未运行 CUDA/训练。
原始 `file://` 标签仍受浏览器安全策略拦截；窄屏/具体 viewport 的布局尚未验证，不得将该部分声明为已验证。

## 恢复后 outputs 语义合同修订（2026-09-07）

本次只修订 outputs 的治理与说明文档，不修改任何恢复的 checkpoint、图像、JSON、
日志、索引或 `_wsl_restore_metadata_v4`。只读核验确认 restore plan 的
163/163 个 archive 状态为 `complete`，声明的源文件数为 `33117`、源字节数为
`240556414616`、path mapping 为 `46`。恢复历史内容到 `20260829_v2` 为止；
`20260830_v2` 及之后的日期目录是当前 Windows-native 运行空间，最终归属以
restore plan 而不是目录修改时间推断。

`outputs/version_index.json` 和 `.md` 保留原 WSL `/home/...` 路径，只能作为历史
provenance，不能作为当前 Windows 路径解析器或 formal acceptance 输入。修改文件为
`AGENTS.md`、`README.md`、`docs/repository_taxonomy.md`、
`docs/operations/output_artifact_notes.md`、本文件和迁移记录；
`git diff --check` 通过，未运行 CUDA 或完整 pytest。本条只记录 outputs 治理边界；当前
科学下一步以本文件顶部的决策摘要和最新 N3 条目为准，不进行输出清理或自动重解析。

## N3 research history navigation

N3 OLS-native 的日期化事实、数值和被取代的下一步统一维护在
[实验历史](status/experiment_history.md)；本文件只维护上方当前决策摘要、治理边界和验证状态。
历史条目中的“下一步”只描述当时的状态，不能覆盖本摘要，也不能授予当前执行权限。
旧入口中的实验身份已逐项映射到实验历史、迁移记录、假设 ledger 或 archive；其中
`H-LOMA-GEN-PCR-PSD` 已补回实验历史，`H-LOMA-CBOGT-FIBER-MMF-M6` 保留在迁移记录。
当前入口省略的是重复正文、非权威 attempt 的逐项路径清单和过时下一步，不代表 outputs 或 machine
evidence 被删除。

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

## 历史证据导航

Windows-native migration、恢复 outputs、旧 harness、Scaled-Chen/I0/T1 以及 reconstruction closeout 的
日期化事实统一见 [实验历史](status/experiment_history.md)；迁移/桥接细节见
[Windows-native migration](migration_windows_native_20260830.md) 与
[ChatGPT ↔ Codex workflow](chatgpt_codex_workflow.md)。本文件不再复制历史 SHA 表、旧下一步或
逐次实验正文；当前允许动作只看本文件顶部摘要和 `status/current_handoff.md`。
