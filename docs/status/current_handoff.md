# 当前正式基线与活动研究交接（2026-09-08）

本文件是当前执行状态的稳定入口。正式 reconstruction mainline 已在当前 evidence boundary
收口，但独立的 OLS-native research line 仍处于 N3 selector-calibration 阻塞状态。只有下面的
“当前决策摘要”定义当前允许动作；其后的带日期或明确标为历史的条目是不可覆盖的历史 provenance，其中的
“下一步”只表示当时的记录，不得直接执行。架构合同见 [v2 runtime](../architecture/v2_runtime.md)，
假设边界见 [capacity-map hypothesis ledger](capacity_map_hypothesis_ledger.md)，完整历史事实见
[experiment history](experiment_history.md)。2026-08-24 重整前的长篇记录保存在
[archived handoff](../archive/current_handoff_pre_sced_20260824.md)，仅供溯源。

## 当前决策摘要

- 正式运行时基线：v2 `conditional_identity_logit_amplitude_v2`；正式 promotion、generation
  readiness 和 current training authorization 均未建立。
- 活动研究主线：`research/lightgen-ols-native-semantic-generation-v1`（精确 HEAD 与
  worktree 以本地 Git/bridge manifest 核验；本轮 comparison consumer、report 与状态文档变更后尚未
  refresh Sources，不声明 publication 为 `current_clean`）；N1/N2 完成，N3 producer
  baseline、selector calibration、两候选 minimal producer comparison 与 frozen local-location
  mixture selection 已完成但均未支持新 producer family；新的 local-shrinkage-covariance contract
  已定义但尚未执行，N4 未决，N5 blocked，N6 保持独立冻结。
- 最新 N3 selector calibration V2 为 `N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`。
  独立 synthetic confirmation 为 NULL=`0/200` false positives、partition=`100/100`、
  radial=`100/100`、nonlinear=`82/100`、local=`78/100`；local 未达到预注册 `0.80`
  correct-rate target，因此不能宣布 calibrated。
- 对应 machine evidence：
  [`n3_attribution_selector_v2_calibration_report.v1.json`](../../outputs/20260908_v2/runs/lightgen_n3_attribution_selector_v2_calibration_attempt_01/n3_attribution_selector_v2_calibration_report.v1.json)。
  frozen spec SHA=`9ee5b7ad64221b323f1b182c2f4d74edddfad89599bdf30b352ef33cc5b4c25b`，
  report SHA=`77c0a9f632b6870579a338d84dd6650530849fa6b74892acb1345567ffa8dae5`。
- 真实 N3 attribution 未执行；不能用 selector 重新解释真实 N3 failure，也不能据此选择模型族。
  MIXED 边界关闭本 selector refinement line，不创建 V3。future producer-family confirmation
  必须使用 fresh confirmation source；当前 internal-validation bank 仅为 N3 development bank。
- `DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1` 已完成；reservation 状态为
  `RESERVED_NOT_MATERIALIZED`。Manifest 与完整 lineage 见 [machine manifest](../../outputs/20260908_v2/runs/n3_sealed_confirmation_source_v1_attempt_01/n3_sealed_confirmation_source_manifest.v1.json)。
- `COMPARE_N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1` 已在既有 1000-row development
  pool 上完成固定 5-fold、A/B 两候选 comparison；fold assignment SHA 为
  `90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32`。两候选均可运行并
  达到固定 positivity bank，但均在 5/5 folds 超出自身 model-vs-model q95 primary reference
  与 covariance/norm guardrails，结果为
  `N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED`，没有 selected candidate。Report SHA 为
  `af7611cfbca211e666d7773e81b230a1b380b91583817c3c4ab80925f2a16440`。
- `DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1` 已完成并冻结
  shared-covariance local-mixture contract，候选 `K={2,4,8}`，复用同一 frozen fold assignment；
  contract classification 为
  `DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY`，SHA 为
  `38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449`。这只是
  `N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED`，尚未 fit、sampling 或选择 K。
- `EXECUTE_N3_LOCAL_LOCATION_MIXTURE_SELECTION_V1` 已严格执行 frozen contract；K=2、4、8 均
  5/5 folds operationally valid，但均为 0/5 compatible，最终 classification 为
  `N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED`，没有 selected K。Report SHA 为
  `5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080`；这是 development-only
  valid failure，不授权 final all-1000 fit、sealed confirmation 或 generation。
- `DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1` 已完成并冻结下一
  minimal family：保留 Stage-1 unsupervised local partition，Stage-2 对 assigned fitting rows
  使用 component-local analytic shrinkage covariance。classification 为
  `DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_READY`，SHA 为
  `2cb596abe61deeaa847b90c74654c5abe0e94aae18a6f252170f8149ad1c1550`。这只是
  `N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED`，尚未 fit、sampling 或选择 K。
- 恢复 outputs、历史 evidence、producer、rank、partition、formal v2 和全局 flags 均保持不变。
- 当前允许的下一项：若继续，只能执行这个新 frozen contract 的一次 development-only consumer；
  不得修改 Stage-1 partition、增加 K、改变 `n_k<2` 语义、消费 sealed source、启动 selector V3
  或 generation。此前 local-location line 不得重跑或调参。

## 本轮 evidence I/O 修复（2026-09-08）

已修复 `src/ldm_project/v2/evidence/io.py` 中会把设备、资源或程序异常宽泛包装成
`ValueError` 的路径：只将 weights-only 的 `pickle.UnpicklingError` 转为明确的安全载荷拒绝，
其余异常保留原始类型和原因；canonical serialization 不再把任意对象的 `__dict__` 默认为
稳定 schema；artifact digest、shape、dtype 和非 tensor 输入仍 fail-closed。公共函数签名、
有效 artifact 的规范化结果和历史哈希兼容性保持不变。定向回归 `23 passed`、Ruff 定向检查
已通过；默认全套 pytest 已尝试（`1609 passed, 12 deselected, 25 subtests passed, 5 failures`），
失败项为既有分支/布局或临时路径环境问题；最终 `git diff --check` 已通过。

## 历史合同草案（已由 V2 执行结果取代）

`PLAN_N3_ATTRIBUTION_SELECTOR_CALIBRATION_REPAIR` 是已完成前的 planning contract，不是当前待批准动作。
其建议的 replicate、FPR、置信界限和机制控制已由后续
`CALIBRATE_AND_CONFIRM_N3_ATTRIBUTION_SELECTOR_V2` 明确采用并执行；权威事实、结果与限制集中见
[实验历史](experiment_history.md)。本节保留名称与取代关系，不能追溯修改 evidence，也不授予后续科学权限。

## Research governance simplification（2026-09-07）

`RESEARCH_GOVERNANCE_SIMPLIFICATION_V1_ESTABLISHED`：本轮只改变 research planning、evidence
handoff 与 provenance presentation；不改变 scientific result、formal/high-risk evidence、
training authorization 或 historical artifact。

普通交接采用 [`delta_one_gate_short_prompt.md`](../operations/delta_one_gate_short_prompt.md)
的 R0–R3 风险层、one-outer-Gate 和 semantic authority ID；完整 child Hash 只留在其
machine consumer。N3 的 `RESOLVE_N3_SCORE_GEOMETRY_ATTRIBUTION_V1` 与后续
`CALIBRATE_N3_ATTRIBUTION_DECISION_RULE_V1` 已完成；当时记录的下一步
`PLAN_N3_ATTRIBUTION_SELECTOR_CALIBRATION_REPAIR` 随后已由 V2 consumer 执行并取代，
不再是当前动作；当时仍不执行 fit、SVD、rank selection、sampling、forward、CUDA 或 generation。

## N3 研究历史导航

OLS-native N3 的 branch genesis、producer data authority、rank/fit/validation、mismatch attribution、
score-geometry resolution 与两轮 selector calibration 均已迁移到唯一的
[实验历史](experiment_history.md) 条目。本交接只保留当前摘要、阻塞与允许动作；历史条目中的下一步
不构成当前授权，恢复 outputs 和 machine evidence 仍按现有 immutable 边界读取。

## S3 历史研究导航

2026-09-07 的 controller semantic-geometry data authority、one-shot canary、frozen S3 consumer
与 postmortem 已集中记录在 [实验历史](experiment_history.md)。本交接只保留当前 N3 增量、阻塞、
允许动作和合同入口；S3 条目中的历史“下一步”不构成当前授权。

## FINAL_MAINLINE_CLOSEOUT：H-LOMA-CBOGT Scaled-Chen（2026-09-03）

### Final state

- Current phase: `MAINLINE CLOSEOUT`
- Scientific diagnosis: `closed`
- Repair: `confirmed`
- Integration: `qualified`
- F32: `qualified`
- Visualization: `completed`
- Runtime performance: `characterized`
- `mainline_state=closed_at_current_evidence_boundary`
- Current mainline next scientific task: `none`
- Future independent track: `REAL-HARDWARE DEPLOYMENT`; explicit future task name:
  `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`.

### Frozen authority and conclusion

The authority chain is `RMAL → S7 MPCDAR → S8 F64 R16-IQ → S9 F32-NQ → S9V full-path
visualization → S10 runtime characterization`. Later consumers use only their named immutable
parents; write-once earlier attempts remain historical and are never silently substituted.

Within the existing optical-path/CMOS evidence boundary, task-relevant information survives through
CMOS. The reconstruction failure is primarily supported as mapping/readout misalignment; simple
locality/raw Euclidean W-SVD is inadequate, useful mean/DC material is present, and centered
detector information is better represented by a fit-derived data-aware basis. Registered MPCDAR
rank 16 is the first confirmed candidate in its preregistered powers-of-two family, not a minimum, intrinsic, or
true rank-16 claim.

Frozen formula: `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`. The S9 F32
artifact contract is `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, `V16[40000,16]`, with
`684352` scalars, `2737408` raw tensor bytes, and `2739448` serialized bytes. The allowed system
description is `optical front end → CMOS intensity sensing → lightweight frozen electronic rank-16
readout`: hybrid optical-electronic reconstruction, not a native CMOS optical image.

S8 F64 is integration-qualified; S9 F32 is numerically qualified; S9V is
`descriptive_visualization_completed` and adds no scientific Gate; S10 is CPU-only readout
characterization, not measured speedup or production evidence. S10 recorded Python `3.11.16`,
PyTorch `2.5.1`, Windows `10.0.26100`, threads `8/8`, logical cores `16`, and CPU string
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; expected Windows `10.0.26200.0` did not
match. Batch mean latency B=`1/8/32` was `0.5601775/0.8914525/3.277882 ms`; mean throughput
was `1836.8588342/9338.7841702/9874.3015643 samples/s`; analytical MAC ratio was
`15.89825119236884`, not a measured speedup.

### Non-authorized boundary and future tracks

Native optical target-like CMOS reconstruction, real physical CMOS validation, Raspberry Pi
benchmarking, sim-to-real detector alignment, end-to-end hardware latency, GPU runtime, full-W
runtime, quantization, generative manifold, and generation readiness remain unestablished.
`reconstruction_manifold_supported=false`, `generative_manifold_supported=false`,
`generation_readiness=not_ready`, `training_authorized=false`, `formal_promotion=false`, and
`closed_task_relevant_bottleneck_not_supported` remain fixed. Future tracks are separate and require
new authority: real CMOS/Raspberry Pi deployment, sim-to-real alignment, optional optimization only
if separately justified, native optical redesign only with explicit authorization, and separate
generative-manifold/generation evidence. Older hypothesis entries remain for provenance and are not
deleted; their historical next-action text is superseded by this closeout.

Bridge closeout is `status=invalid`, `bridge_recovery_blocked=true`: the clean-HEAD `refresh`
stopped fail-closed on the existing invalid manifest
`C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`.
The package was not deleted or overwritten, and no automatic upload occurred.

## 历史与稳定合同导航

Formal v2 identity、主线 closeout、旧研究路线与迁移状态的日期化正文统一见
[实验历史](experiment_history.md)、[v2 runtime](../architecture/v2_runtime.md) 和
[Windows-native migration](../migration_windows_native_20260830.md)。本交接不再复制旧实验正文、
完整 SHA 表或历史下一步；恢复资产仍按 `AGENTS.md` 与 outputs restore metadata 的不可变边界处理。

需要恢复某项历史研究时，先从实验历史定位 authority，再回到本文件顶部确认当前 branch、阻塞、
允许动作和是否需要新的独立授权。
