# 当前主线交接（2026-09-07）

本文件是当前执行状态的稳定入口。最终主线已在当前 evidence boundary 收口；当前没有下一科学
实验。架构合同见 [v2 runtime](../architecture/v2_runtime.md)，收口边界见
[capacity-map hypothesis ledger](capacity_map_hypothesis_ledger.md)，带日期的实验事实见
[experiment history](experiment_history.md)。2026-08-24 重整前的完整长篇记录保存在
[archived handoff](../archive/current_handoff_pre_sced_20260824.md)，仅供溯源。

## H-LOMA-GEN-N3 OLS producer data authority（2026-09-07）

OLS_PRODUCER_DATA_AUTHORITY_MATERIALIZED 已完成，authority attempt 为
outputs/20260907_v2/runs/lightgen_ols_producer_data_authority_v1_attempt_01/。
它消费既有 S1 fit R3_COMPLEX，source artifact SHA=
9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279，
S1 semantic manifest SHA=
b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e，
checkpoint SHA=
ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de。

fit bank SHA=
21b989a13cbea8f24d3298424b200eb1727de8d04809bfbc77e1901daa888f77，
internal-validation bank SHA=
8397c886db73bb260723debd6d4eab332b6172972a055b28f770e7ca2bf3ee25；
两者均为 500、50/class、torch.complex64、N=100 的 exact complex
row bank。partition manifest SHA=
409b598aacae921da631adea553b339b56560231f9cb2cf75ce8d7e5b439242b，
semantic identity=
d1eaf777c27b98eb607e98ca41d2360115eb0b070156b29d21805c7332981d6d。

data-firewall manifest SHA=
b711e9d3443f3a86c803ff1f8cae1ff266634439ebe0c0d31c04e807c3841837，
firewall_pass=true。independent protected source/S3 identities 的 overlap
全部为 0；S2 aliases、historical/post S3 和 Code8 为 SUBSUMED。controller
repair internal code panel 是 CONTROLLER_CODE_SPACE、complex_states_saved=false
且对 N3 overlap 为 NOT_APPLICABLE，不记录伪造的零 overlap。未注册的外部候选
identity 未进入 authority/manifest。

本轮不包含 optical forward、CUDA、optimizer、backward、producer fitting 或
novel OLS generation；不得解释为 producer trained、rank selected 或 generation
supported。training_authorized=false、formal_promotion=false、
generation_readiness=not_ready 保持。下一唯一 Gate：
PLAN_LOW_RANK_JOINT_AMPLITUDE_PRODUCER_CONTRACT。第一次错误 materialization
已隔离为 invalid quarantine，不是 authority。

## H-LOMA-GEN-S3 Controller semantic-geometry one-shot repair postmortem（2026-09-07）

当前正式状态为 `S3_ONE_SHOT_REPAIR_POSTMORTEM_COMPLETE`，primary attribution 为
`FIRST_SECOND_ORDER_GEOMETRY_OBJECTIVE_INSUFFICIENT_SUPPORTED`。只读比较绑定 historical S3
authority=`0779d8d0ffb0a9958064bb3c05d451058cdd7a670a732d6ae1d4ba49c11eb5e6`、geometry diagnostic=
`1112b8331b3a7a31cf4814ad50c6b3ea8b60217771ef222b7182e553d8284682`、Code8 review=
`d8f073987eda234ac9b85731dad66e9d9ffc92eaf896f700ff84ba4de17263fd`、final-step-492 checkpoint=
`4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84` 与 post-training S3 manifest=
`e13603745ed9fc70ae4f47975241d7653629000090e89aef5cb38b5357e5018c`。

结论：centroid alignment 改善，covariance magnitude/coverage 改善，但 absolute reachability 仍失败；
covariance shape/tangent residual、gain compression 与 class-conditioned compression 仍存在。local
Code8 actuation 仍为 `CODE8_LOCAL_CONTROL_FULL_RANK`。class9 改善但保持 secondary；没有足够证据支持
`CURRENT_CONTROLLER_PARAMETERIZATION_LIMITATION_SUPPORTED` 或 global-folding/topology failure。S3 仍为
`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`，`KEEP_8D_PROVISIONALLY`。

exact one-shot repair line 已 closed；不 retune、不 second canary、不 S3 rerun、不 class9 repair，S4
blocked。`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。
唯一下一 research question：在冻结当前 8D、S3 panel、optical front-end 与 OLS 的条件下，哪些未被
first/second-order class moments 约束的 map-level higher-order/support/tangent quantities 决定剩余
absolute reachability failure？

## H-LOMA-GEN-S3 Controller semantic-geometry post-step-492 frozen S3 consumer（2026-09-07）

已完成一次有效的 frozen S3 consumer；不重新训练、不创建 optimizer、不 backward、不写 candidate checkpoint，
不改变 panel、metric、threshold 或 optical authority。candidate SHA=`4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84`，
preflight SHA=`ed44ba642252a7ecb08faa2541af9af55f9b7d1a72c9ea8696afd8fc689c1835`，final state hash=
`cccefcb25c674b11f64653fe1f97a0193d4896bc69c4fe01cf3812daffda1b99`，step=`492`。第一次 invocation 因
internal-validation 后 allocator cache 影响同进程资源门而停止、未写 artifact；`83436fc` 释放 CUDA cache 后，
在 clean HEAD=`83436fcccafb0c760ed81424976fe627a3ef3e2d` 上完成有效 attempt。

write-once output=`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_post_step_492_v1_attempt_01/`，
manifest SHA=`e13603745ed9fc70ae4f47975241d7653629000090e89aef5cb38b5357e5018c`。internal-validation 是
seed=`100000`、与 S3 base-code overlap=`0` 的 descriptive panel；正式 S3 仍是 seed=`1234`、
`Uniform([-1,1]^8)`、100 base codes×10 labels、1000 states。

结果：`valid_completed`，classification=`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`。interface pass；
targeting fail（selection/confirmation retrieval real-vs-null=`0.832/0.106`、`0.812/0.098`，direction=false）；
reachability fail（selection direction=false，confirmation hard criterion=false）；code effect nondegenerate、
no class collapse、duplicate=`0`、unique R1 states=`1000`；Jacobian 每类/aggregate=`CODE8_LOCAL_CONTROL_FULL_RANK`；
state isolation pass，candidate/optical/controller hashes unchanged，optimizer/backward/training/checkpoint/
decoder/evaluator/parameter update 全为零。相对 historical S3 的小幅 retrieval 变化仅 descriptive，不是选优。

当前唯一治理结论：controller semantic-geometry repair line 在本 evidence boundary 关闭；不自动进入 S4、
再训练、调参、class9 repair 或 S3 rerun。`mainline_state=closed_at_current_evidence_boundary`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 保持不变。

## H-LOMA-GEN-S3 Controller semantic-geometry one-shot canary（2026-09-07）

显式授权 `AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY` 已消费一次，
并在 clean implementation HEAD
`169fd515b7d6ff6d862222c066c44a39dc373879` 上完成已注册的 controller-only、frozen-optical、
fixed-horizon canary。attempt-01 的 preflight 通过但旧 consumer 在 optimizer 前拒绝
`cuda:0` alias，未产生任何更新或 checkpoint；该 write-once 尝试保留且不消费。

attempt-02 的 3-step preflight gate=
`CONTROLLER_SEMANTIC_GEOMETRY_3_STEP_PREFLIGHT_PASS`；证据为
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02/preflight.json`，
SHA=`ed44ba642252a7ecb08faa2541af9af55f9b7d1a72c9ea8696afd8fc689c1835`。随后准确执行
`492` logical optimizer/backward steps，唯一 final-step-only checkpoint 为
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02/controller_semantic_geometry_final_step_492.pt`，
SHA=`4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84`；loss=
`0.34952232241630554`、mean_anchor=`0.19811758399009705`、
mean_covariance=`0.5009270906448364`。

state isolation 通过：generator state hash 从
`3d06989ba6c5f9154e4d05b64cef42bd4c94b02686ff5b2120a953d9f87a7931` 变为
`cccefcb25c674b11f64653fe1f97a0193d4896bc69c4fe01cf3812daffda1b99`；final payload 只有
四个 generator state groups，冻结 optical/evaluator/controller authority bindings 未变。
run 目录仅含 preflight 与 final checkpoint，Git execution HEAD clean。

结果是 `bounded_controller_semantic_geometry_canary_completed`，不是 S3 efficacy 或
generation/promotion 结论。S3 classification 仍为
`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`；无 evaluator、decoder、S3 rerun、generation
metrics 或 formal promotion。全局治理仍保持 `training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`。后续任何 S3 效果判断须另行授权
新的 write-once S3 rerun；本次授权不成为持久 training authority。

## H-LOMA-GEN-S3 Controller semantic-geometry data authority（2026-09-07）

上一授权在 `e4e1a855f55a542d75ec5121b86dd242028c71b6` 上以 `INVALID_HARNESS`、step 0 结束；
未发生 optimizer/backward/forward/CUDA/checkpoint/S3 rerun。data-authority registrar、safe-load 与
fail-closed consumer 已提交：实现=`23a0e5ae2454949b5963a3b240aa54e14004c5ed`，artifact binding=
`8d341914bdbfa355984b13c20bfd11b1f8d8c264`，fresh reauthorization status=`0f1126660a2d2d8c497c7c35d3ba01fd5c6d13d8`。

专用 authority run：
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_data_firewall_v1_attempt_01/`。
target bank SHA=`d0f29701e4f9818c48b222e1cbfd8706257ec38356842870f154cd3fc2c8ecb0`；partition manifest
SHA=`1eae38d6086a7e395960177e24e8b1ccb7deb1312f2885c07321f3ebeb1cd7c4`；target-bank sidecar SHA=
`9eb2f032de52ec68dfda69ad3abd217ded2774656ec83c217ebe7d22ebed7916`；data-firewall manifest SHA=
`f5704a5108681a78273b918d3eda3ade3b0e68b8af6f4e98314037e1eb94d4d1`。semantic partition identity=
`b07a04d3c779e5952c06209c9d8cc8421b8193cd41f037db88f95c736ab35594`，physical file SHA 已分离绑定；
component hashes 与 S1 deterministic rule 全部复现，local material digest=
`2010a7541994c9422de26f5f646ae67e6c91f73ec49297a2a4f378589615c431`。

registration=`CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED`，safe reload 通过，保护样本/code
overlap 全为 0，registration counters=`optimizer 0 / backward 0 / forward 0`。在 clean implementation
HEAD=`0f1126660a2d2d8c497c7c35d3ba01fd5c6d13d8` 上，production no-update preflight=
`PRETRAINING_HARNESS_PREFLIGHT_PASS`，status=`IMPLEMENTATION_READY_FOR_ONE_SHOT_REAUTHORIZATION`，
steps/backward/checkpoint writes=`0/0/0`。training_authorized=false、formal_promotion=false，S3
仍为 `S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`。

下一唯一动作：等待新授权
`AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY`。未授权前禁止 3-step、492-step、
CUDA training、S3 rerun、class9 repair、architecture/LR/seed/horizon/objective redesign。

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

## Formal identity 状态

当前正式架构仍为 `conditional_identity_logit_amplitude_v2`：`label + 8D code`
生成 amplitude SLM command，经完整 SLM FOV 的相干 4f relay、2+2 phase-only
finite-crop D2NN、complex identity OLS 和 B=0 backend relay，在 CMOS 完成单次曝光。
冻结 evaluator 只用于训练与验收，不属于最终光学推理。

正式 identity 链已在完整 raw-CMOS evaluation 上形成有效模型验收失败。该 checkpoint
不具备 promotion 权限，也不能授权 manual review、candidate export、memorization、
utility、dataset admission 或 accepted manifest。项目治理状态保持
`closed_historical_valid_failure`；旧 evidence 不补签为新的 `EvidenceRecordV2`。

固定边界如下：

- `formal_promotion=false`；
- formal optical training 未授权；
- fiber training、E1 和 modal regularizer 未授权；
- 不得重复已经关闭的 Objective A 正式链，或从 diagnostic checkpoint warm start。

## 已关闭的 research 路线

| 路线 | 当前结论 | 停止位置 |
| --- | --- | --- |
| Synthetic fiber route | `closed_synthetic_fiber_route_no_e1` | 无 E1、无 fiber training、无 formal promotion |
| Corrected WAE O0 | `wae_image_manifold_failed` | seed 7301 confirm 后停止 |
| Latent RealNVP transport | `latent_flow_semantic_failed` | seed 8401 confirm 后停止 |
| Class-conditional mixture transport | `mixture_latent_fit_failed` | seed 8501 screen 后停止 |
| Prior-native H-SWAE | `prior_alignment_fit_failed` | seed 8601 screen 后停止，未进入 confirm |
| Prior-native H-SCED | `prior_alignment_fit_failed` | seed 8601 screen 后停止，未进入 confirm |
| H-CCF cycle composition | `cycle_composition_attribution_underdetermined` | prior-native attribution family closed |
| H-DPGC direct-prior canary | `direct_prior_linear_semantic_capacity_failed` | linear canary closed; no neural training |
| H-DIDM direct image distribution | `direct_image_distribution_failed` on the sole authorized seed-8801 screen; route closed | same nonlinear decoder; no encoder/cycle; no confirm or promotion |
| H-SCF semantic code-source factorial | `semantic_counterfactual_safety_failed` | valid retry4 closeout; unique next action `close_frozen_decoder_derived_semantic_route`; no evidence-derived model, training, or promotion consumer |
| H-NDMC nonlinear direct manifold capacity | `nonlinear_manifold_safety_failed` | valid retry3 closeout; unique next action `close_nonlinear_direct_manifold_capacity_canary`; empirical canary closed, no architecture preflight or training consumer |
| H-JQOC joint quadrature optical capacity | `electronic_structured_input_capacity_failed` | valid retry7; input gate stopped before optical interpretation; no architecture preflight or training consumer |
| v3 spatial quadrature input | `hierarchical_spatial_input_capacity_failed` | valid retry6 electronic admission; secondary evaluator and diversity/coverage gates failed; optical/phase/receiver stages were not started |

H-SWAE 的 validation、synthetic direction audit 和 CUDA preflight 均通过。唯一执行的
seed-8601 screen 是有效科学失败：aggregate SWD reduction 为 `34.62% < 50%`，类别
6/7 退化且 cycle 指标变差。closeout 的固定下一动作是
`close_current_prior_native_swae_route`；该结论只关闭 H-SWAE，不证明 decoder、光学、
fiber、phase-only 或硬件的普遍不可行。

## 已关闭的 H-SCED 与新注册 H-PGC

下一研究假设是 `H-SCED`：以 Gaussian sliced Cramér/energy U-statistic 替换
finite-batch quantile-SWD，检验其是否避免每类小批量下的 variance-shrink optimum，
同时保持 reconstruction、cycle、novelty 和 dual-evaluator electronic O0 合同。

H-SCED 实现、配置和测试已提交于 `0d83a4d`；paired-batch preflight 修复于
`9290016`，lineage-bound lifecycle 修复于 `70b8e32`。validate、512-replication
synthetic audit 和 CUDA preflight 均已通过，且 evaluator calls 为零。当前尚无
screen、confirm、locked-final、O0-G 或 fiber 结果；这些通过项只授权预注册的
seed-8601 screen，不代表科学假设已通过或获得正式 promotion。

当前 immutable evidence 位于
`outputs/20260824_v2/runs/20260824_electronic_o0_prior_native_sced_v1/`：
validation `b6f4f4…16850`、audit `5e7884…f36b1`、preflight `2aea7a…75bb`。

seed-8601 screen 是完整有效的科学失败：training-bank aggregate SWD reduction
`39.73% < 50%`，reference-bank reduction `36.67% < 50%`，cycle
`0.0225073 > 0.0164912`；screen evidence、report 和 closeout 均保持 evaluator
calls 为零，且 closeout 固定下一动作为 `close_current_prior_native_sced_route`。
该结果只关闭 H-SCED，不证明 WAE decoder、光学、fiber、phase-only 或硬件的普遍
不可行性。

## 当前注册假设：H-PGC → H-TCA

H-SCED 当前路线已按有效失败关闭。不得继续 seed-8601 confirm、seed-8602/8603、
locked-final、O0-G、SMF、E1、fiber-aware training 或 formal promotion。任何未来
探索必须注册新的 electronic O0 假设、变量和 lifecycle，不得在本路线内调 epoch、
seed、权重或阈值。

H-PGC（Prior Gradient Conflict）是 diagnostic-only、无更新归因路线。它只读消费
WAE seed-7301、H-SWAE seed-8601 和 H-SCED seed-8601 的完整 valid closeout，比较
reconstruction/alignment/cycle 梯度、固定权重方向与解析约束投影；不创建 optimizer、
不加载 evaluator、不修改旧 evidence，也不能授权训练。H-PGC validation、audit 和
纯消费者 report/closeout 已完成，均为 valid；immutable SHA256 分别为
`ac2f28…b14f`、`7496ca…6beb`、`105d57…2476` 和
`d777af…e1e0`。最终分类为 `gradient_conflict_not_supported`：两个 projection
bank 的 SCED full-joint cycle rho 均未达到 `0.05`，H-SWAE paired arm 也未达到，
故不能把历史 cycle 恶化归因于一阶固定权重冲突。peak allocation 为
`821096448` bytes、run storage `1183663` bytes，optimizer/evaluator calls 均为零，
参数 hash、finite-difference、KKT 和 classwise 10 类报告均通过。下一动作只注册
`H-TCA`，不授权 projected-gradient、可逆 adapter、confirm、O0-G、光学或 fiber 路线。

H-TCA（Prior Trajectory/Curvature Attribution）已注册为独立的临时副本诊断路线。
它只读绑定 WAE seed-7301、H-SWAE/H-SCED seed-8601 final checkpoints 和 H-PGC
closeout；在不加载 evaluator、不修改旧 artifacts、不产生候选 checkpoint 的条件下，
精确重放两条 625-step 历史轨迹，并测量 landmark 的 raw gradient、AdamW 实际位移、
observer 方向导数、Taylor remainder 与 cycle HVP。唯一变量是路径归因机制：gradient
rotation、AdamW preconditioning、finite-step curvature 或 observer distribution
mismatch。有效结果只能被 report/closeout 和 hypothesis ledger 消费；不能直接授权
任何 optimizer 替换、trust-region、adapter、confirm 或训练。

H-TCA 实现及加固提交为 `84fbb7d`、`1ba714b`、`f18d71b`、`c616c7a`、`4486033`、
`3634221`、`6bd465c`、`44c032e`、`7929779`；早期 projection-bank、gradient layout、
classwise、history、terminal-landmark、attribution-gate 和 preflight-replay-contract
缺陷均按 fail-closed 保留，未消费旧 retry evidence，并从 clean HEAD 重建 lineage。
retry7 validation、preflight、audit、report、closeout 均 valid，immutable SHA256 依次为
`0f5c40…c2b`、`e1d6be…c024`、`8d17ef…aa44`、`b424f1…6707` 和 `53d0d5…3a9c`。

retry7 的 H-TCA 分类为 `trajectory_attribution_underdetermined`，
`alignment_estimator_independent=false`：H-SWAE 与 H-SCED 均未在两个 projection bank、
连续 landmark 和至少 3/4 observer batches 上支持 trajectory gradient rotation、AdamW
preconditioning conflict、finite-step curvature cycle drift 或 observer distribution
mismatch。两条 arm 的 encoder/decoder/optimizer/RNG/sampler/schedule 终态均 exact，SCED
13 个 history windows exact，14 个 landmark（含 625）完整；peak allocation 为
`1771866112` bytes，storage 为 `10217971` bytes，optimizer 只存在于临时 replay 副本，
evaluator calls 为零。该结果只授权 `register_denser_local_trajectory_attribution`，不
授权 optimizer 替换、trust-region、adapter、projected-gradient、SGD、confirm、O0-G、
光学或 fiber 路线。正式 promotion、
O0-G、SMF/FMF/MMF、E1、光学和 fiber training 继续禁止。

H-DLTA（Dense Local Trajectory Attribution）已完成其 diagnostic-only lifecycle。实现与
harness 修复提交为 `95e61c5`、`f3d5962`、`d5670e3`；首轮 SCED history-window mismatch
（audit `0eba96…6d3a`）按 fail-closed 保留，修复后从新 clean HEAD 重建 lineage。固定的
0–20、230–270、330–370、480–520 和 625 共 39 个 landmark、AdamW 四分量分解、
actual-step HVP/有限差分、reverse-direction negative control、paired replay/resume 和
SCED 13 个 history windows 均通过。H-DLTA validation、preflight、audit、report、closeout
的 retry evidence SHA256 依次为 `a2045d…0edf`、`1207f1…1502`、`d168f8…b18e`、
`797e5a…2506`、`a80ef1…bd55`；peak allocation 为 `3457404416` bytes，run storage
为 `22407988` bytes，evaluator calls 为零，旧 artifacts/checkpoints hash 未改变。

最终分类为 `dense_local_attribution_underdetermined`：H-SWAE 与 H-SCED 在 early、W250、
W350、W500 四个预注册窗口均没有任何机制同时满足三个连续 landmark、至少 3/4 observer
batches 和两个 projection banks 的支持门；没有 first-supported landmark，classwise
locator（0/250/350/500/625，含 class 6/8 定位字段）也不改变主分类。raw gradient rotation、
momentum、adaptive preconditioning、weight decay、finite-step curvature 和 observer sample
distribution mismatch 均 unsupported。下一动作固定为
`close_optimizer_trajectory_attribution_family`，只允许登记无更新 counterfactual hypothesis；
不得修改 optimizer、objective、阈值或架构，不得启动 projected-gradient、SGD、trust-region、
adapter、confirm、其他 seed、O0-G、SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion。

下一注册假设为 `H-CCF`（Prior Cycle Composition Factorial）：在不创建 optimizer、
不加载 evaluator、不生成 checkpoint 的条件下，将 source/final encoder 与 decoder 做
2×2 factorial 组合，分解 cycle 恶化的 encoder、decoder 和交互贡献。它只允许 no-update
counterfactual attribution，不重新打开 optimizer/trajectory attribution family。

## H-CCF 已关闭：cycle composition attribution underdetermined

H-CCF 实现提交于 `dcd9528`，direction/source consistency gate 修复于 `629e1aa`，
reconstruction locator 补齐于 `2d5a288`；首次
preflight 因 chunk-128/257 radial reference 的 `2.6e-7` 浮点差超过原门槛而按
`invalid_harness` 保留，修复提交为 `c306f46`，并从新 clean HEAD 重建 validate lineage。
最终 retry3 validation、preflight、audit、report、closeout 的 SHA256 分别为
`9163c5…a714`、`d605fa…2e2e`、`559af9…b476`、`30e629…107b` 和 `65eaa0…c342`。

最终 audit 为 `status=valid, passed=true`，classification 为
`cycle_composition_attribution_underdetermined`，next action 为
`close_prior_native_attribution_family`。SWAE 与 SCED 的 full-bank source/final cycle
分别复现为 `0.01649122442/0.02642731884` 与 `0.01649122442/0.02250728432`，均在
`atol=1e-8, rtol=1e-6` 内；四个 stability banks、factorial identity、source/source
null swap、chunk equivalence、primary/MSE direction consistency 与 hash isolation 均通过。SWAE 的
cycle drift 为 `0.00993609441`，SCED 为 `0.00601605990`；两条路线均未达到单一
encoder/decoder/interaction 或 mixed component 的稳定归因门，故不能把 cycle 恶化归因
到某一个组合因子。reconstruction objective 与 decoder image displacement 定位字段均已记录。
peak allocation 为 `147424768` bytes，storage 为 `416895` bytes，
optimizer_created=false、evaluator_calls=0，旧 artifacts/checkpoints 保持只读。

H-CCF report/closeout 是纯消费者，未创建候选模型。该结果关闭 prior-native attribution
family；不得启动 encoder geometry、decoder invertibility、interface Jacobian、hybrid
mediation、projected-gradient、SGD、trust-region、adapter、confirm、其他 seed、O0-G、
SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion。若继续研究，必须另行注册
无更新诊断假设并建立新的 lineage。

## Electronic O0 综合结论与 H-DPGC

跨路线的正式比较记录见
[electronic O0 research synthesis](electronic_o0_research_synthesis.md)。WAE 的主瓶颈是
双 evaluator 语义覆盖和类别召回；Flow、mixture、H-SWAE 与 H-SCED 只改善局部轴，未
同时满足语义、先验、cycle 和记忆化门槛。H-PGC、H-TCA、H-DLTA 未找到稳定的梯度、
AdamW、曲率或 observer 分布机制，H-CCF 又显示 encoder、decoder 与 interaction 存在
抵消，因此不再继续修补 prior-native 训练路线。

当前固定状态为 `prior_native_attribution_family_closed=true`，所有现有 checkpoint
均不得 promotion。H-DPGC（Direct Prior Generator Capacity Canary）已完成无 optimizer 的
固定 rank-8 logit-Gaussian canary；它只测试线性直接容量，不代表所有非线性直接生成器。

H-DPGC retry3 的 validation/preflight/audit/report/closeout SHA256 依次为
`af0994…54ef`、`890ad5…8818`、`7fc08c…f501`、`e3d573…be58` 和
`e61e4…d075`。三次早期 preflight 尝试分别保留了 coverage device、camera-label device
和 memorization device 的 harness 异常；每次均未形成 evidence，修复后从新 clean commit
重建 lineage。最终 audit/closeout 为 valid scientific failure：分类
`direct_prior_linear_semantic_capacity_failed`，next action 为
`close_direct_prior_linear_capacity_canary`。三个 locked panels 的 primary minimum recall
为 `0.76/0.80/0.92`，未达到 `0.90`；secondary minimum recall 为 `0.90/0.96/0.96`，
Jacobian rank-1e-3 pass fraction 为 `1.00`，exact duplicates 均为 `0`，但 memorization
pass fraction 为 `0.914/0.932/0.908`。official 10k primary/secondary accuracy 为
`0.9544/0.9761`，primary minimum recall `0.8768`，agreement `0.9470`。peak allocation
为 `517880320` bytes，audit storage `41238` bytes，optimizer-created 为 false。

该 valid failure 仅关闭固定线性 canary，不外推为所有非线性直接生成器失败；但在获得新的
无训练容量证据前，不得注册或启动 neural direct-generator training。正式 promotion、
confirm、O0-G、SMF/FMF/MMF、E1、光学和 fiber training 继续禁止。

## H-DIDM（retry3 历史状态；已由 retry4 closeout 取代）

H-DPGC 只否定了固定线性 rank-8 logit-Gaussian canary，未否定现有 nonlinear decoder
在直接分布训练下的可行性；当时注册的 H-DIDM（Direct Image Distribution Matching）为：
只读 warm-start WAE seed-7301 decoder，删除训练期 encoder、reconstruction、latent
alignment 和 cycle，直接用 `label + 8D N(0,I)` 生成 O0-I image，并以固定多尺度
classwise sliced-energy U-statistic 对齐真实图像分布。

H-DIDM 是 diagnostic-only 训练结构实验。evaluator 不进入 audit、训练、权重或 checkpoint
选择；不改变 latent 维度、decoder backbone、光学结构或最终单次推理接口。retry3 的
direction audit/preflight 曾通过，但后审发现 preflight 在 step 3 后没有重放更新，audit
也未绑定 collapse/blur/label-shuffle 与 chunk/formula controls；这些 evidence 现标记为
superseded/invalid_harness，不形成科学结论，也不再授权 screen。

retry3 的局部证据不得授权 encoder geometry、decoder invertibility、interface Jacobian、
hybrid mediation、projected-gradient、SGD、trust-region、adapter、confirm、其他 seed、
O0-G、SMF/FMF/MMF、E1、光学、fiber training 或 formal promotion；其 screen 需另行明确
授权并沿用 immutable manifest lineage。最终 retry4 结果见下方 closeout。

## 2026-08-24：H-DIDM retry3 harness evidence 废弃

H-DIDM 实现提交为 `db391ef`；保护门修复分别提交为 `85c155f`、`e78951d` 和 `66d974f`。
按 fail-closed 规则，修复后从 clean HEAD `66d974f` 重建 retry3 lineage；此前三次
validation/audit/preflight 尝试保留但不被消费。

retry3 validation、audit 和 preflight evidence 的 immutable SHA256 分别为
`5b0f35aea831cebbece92ccd22a20607ab29978d0c5f9a8fc1c4497475540543`、
`4a3f350bc79a1a80f9464a1a3dd5552fe5dd53a8bf203163d5968554c20b58a3` 和
`c88c7b2911ea209a76bee10421023a359f0f3e97d59413a6d1df81a39ee72cd4`；这些证据现为
`superseded/invalid_harness`，screen manifest 不再可消费。

审查确认 audit 未绑定计划要求的 real-vs-real matched、class-mean collapse、blur、
label-shuffle、chunk invariance 与朴素公式 controls，因此不形成 direction-admitted 结论。

preflight 虽记录了三步更新和保护指标，但所谓 interrupted/resumed replay 在恢复后执行
了零步，未证明 step-2 checkpoint 到 step-3 的真实续接，因此按 harness 规则废弃。

该历史下一动作已由 `727568b` 的 controls、真实 replay、625-step checkpoint/screen lifecycle
修复执行；retry4 lineage 与最终分类见下方 closeout。retry3 evidence 不得被消费。

## 2026-08-24：H-DIDM retry4 screen closeout

retry3 的 audit/preflight 已按 fail-closed 规则标记为 `superseded/invalid_harness`；它们
不形成科学结论且不被 retry4 消费。修复提交为 `727568b`，在 clean HEAD 上重新建立
`validate → audit → preflight → screen` lineage。retry4 的 validation、audit、preflight、
screen、report、closeout SHA256 分别为：

`1c38fcebe218823246ec9f253769bf95f15049ce0657d7faaffd6992ea333862`、
`c76d69c974ca6ed89ef19935c358688495b3c4234c91f1ae6a8e6f9ea3303b99`、
`338e150c10c0cd8f1b8d20eccf6848900be28ea59080cbe838af51425f1af363`、
`f2ccc0cb7ce72c8e6866bac5fa48be3d873cd636ecc52e47f2eb31374041d369`、
`7e1626c98a089f4db632d0170bcb9f430cdad6e4f549ec45083c9d3ce2c7afa9`、
`d65a36cdc7637f76799c5bd763bf2ea51f1973bea55cd55ed7f6845bbb07fab3`。

retry4 audit 绑定了 real-vs-real、class-mean collapse、blur、label-shuffle、朴素公式、
chunk 和 projection-determinism controls，四个 batch 的 `-g` 双 bank 方向通过，且
optimizer/evaluator calls 均为零。三步 preflight 的 step-2→step-3 interrupted/resumed
replay、decoder/optimizer/RNG/sampler/schedule/history exact gate 通过；两个 observer
bank 均下降，peak allocation 为 `1515885568` bytes，storage 为 `64606` bytes。

唯一授权的 seed-8801、625-step full-coverage screen 正常完成，training checkpoint
schema 为 `electronic_o0_direct_image_distribution_training.v1`，训练期间 evaluator calls
为零。两 projection bank 的 aggregate distribution loss 均低于 source，且十类均未退化；
但三 locked panels 的语义门全部未同时通过（panel pass count primary/secondary 为
`0/0`）：primary accuracy 为 `0.918/0.910/0.894`、minimum recall 为
`0.72/0.78/0.78`，secondary accuracy 为 `0.898/0.896/0.870`、minimum recall 为
`0.80/0.80/0.74`（panel 顺序 `4242/104729/161803`）。memorization pass fraction
为 `0.998687`、exact duplicates 为 `0`，Jacobian rank pass fraction 为 `1.0`，camera-code、
diversity、saturation、coverage 和 label-shuffle 保护门通过；official 10k 仅作完整报告，
不参与选择。screen 为 `status=valid, passed=false`，固定分类
`direct_image_distribution_failed`，next action 为
`close_current_direct_image_distribution_route`。

screen decoder/training checkpoint SHA256 分别为
`c066349955608e2c170f7732a7ecc63be4fb97f6673271373016544b10b5b6f2` 和
`df58662371220c5305e48b65ea71f51378f8637860fcc2e7ead3172f4fa72f6a`；screen peak allocation
为 `646068224` bytes，storage 为 `18826955` bytes。report/closeout 为纯消费者，旧
WAE、H-DPGC 与全部 prior-native artifacts 保持只读。该 valid failure 关闭 H-DIDM，禁止
重跑、延长 epoch、调 loss、换 seed、confirm、decoder v2、O0-G、光学、fiber training
或 formal promotion；若继续研究必须另行注册无训练证据和新 lineage。

## H-SCF closeout（valid scientific failure）

H-SCF（Semantic Code-source Factorial）已完成 retry4 的
`validate → preflight → audit → report → closeout`。五阶段均为 `status=valid`，最终分类为
`semantic_counterfactual_safety_failed`，唯一 next action 为
`close_frozen_decoder_derived_semantic_route`。这是一项 diagnostic-only、no-update 的有效科学
失败，不授权任何训练、confirm、decoder v2、O0-G、光学、fiber 或 promotion。

最终 evidence 位于
`outputs/20260824_v2/runs/20260824_electronic_o0_semantic_code_factorial_v1_retry4/`，SHA256 为：

- validation `252b0ebf226859e618c3fdb342a69e80269098bc09d1deebe4b411378d1ab491`
- preflight `25fd79f3b1194299c32f7c0e963849523caf3d4f3cfb596c461bb9a72da05104`
- audit `aca75439aa2fce3f5414f697f34505e91bcc0052c9532aee8ff8b8e24eaef930`
- report `4a60eda420884d7a19d30aa6712482de4f4394aec6cac7e9c8035b3f6c0da9f6`
- closeout `59d927f7d7ed3b9633fe613e5b9ce14814ba9b5e15776e9ef47352594026ef94`

preflight 资源为 peak allocation `720658432` bytes、storage `10316` bytes、wall time
`9.480835378999473` seconds；audit peak allocation `720658432` bytes、storage `110601` bytes。
历史三 panel 的 view hash 与双 evaluator correct/recall/agreement counts、posterior hashes、
finite、64/257 chunk、参数不变和 `optimizer_created=false` 均通过。

audit arm 摘要：S-G 两 Gaussian banks 的 safety gate 均失败；S-P 两 posterior banks 的
semantic gate 均通过但 safety gate 均失败；D-G 两 Gaussian banks semantic gate 均失败；D-P
两 posterior banks semantic gate 均失败，within-class permutation 两 banks 也均失败。计数为
S-G `2330/2115/2117`、`2320/2129/2121`，S-P `2466/2463/2454`、`2459/2462/2442`，
D-G `2270/2244/2303`、`2290/2240/2308`，D-P `2311/2321/2364`、`2305/2329/2381`
（每项依次为 primary correct、secondary correct、agreement count）。分类按 safety 优先级
停止，不将 D-P 的后续失败重新解释为单一 decoder 或 pairing 结论。旧 WAE、H-DIDM 和 evaluator
artifacts 仍为只读；本轮不执行任何下游路线。

## H-NDMC closeout（valid scientific failure）

H-NDMC（Nonlinear Direct Manifold Capacity）已完成 retry3 的
`validate → preflight → audit → report → closeout`。retry1 因 classwise training
bank 长度不等而错误 `stack`、retry2 因参数 hash 符号引用错误，均在 evidence
创建前作为 `invalid_harness` 保留；修复后从 clean commit `acc22f5` 重建
lineage。

retry3 五个 evidence 均为 `status=valid`，最终分类为
`nonlinear_manifold_safety_failed`，唯一 next action 为
`close_nonlinear_direct_manifold_capacity_canary`。audit 使用 exact classwise
PCA-8、Gaussian copula、连续全同类 kernel reconstruction、两 Gaussian banks、
identity-copula ablation、H-DPGC historical replay、zero-code、label-shift、
64/257 chunk、finite-difference Jacobian 和 curvature controls；不创建
optimizer、checkpoint、candidate 或生成图像全集。

retry3 evidence 位于
`outputs/20260825_v2/runs/20260825_electronic_o0_nonlinear_direct_manifold_capacity_v1_retry3/`，SHA256 为：

- validation `3d3a8cf1c3d30e3923172bc937b6f21295083d2d8963815de1a1ca9e37480d32`
- preflight `6e0a52880b6e222e79ce9a9b63f368f0d9fb6afb62b73f19297aed0114c45783`
- audit `e0fdf0b32fff2159c735f92f67c4c621eaca02aac8c26270fc82e506d5aa6544`
- report `b82d5379863068a8dfa22fcb362e36a1dcfaf37305c1692f728b9719007f267e`
- closeout `f0e21b76723c187056cfb930c306036fbf58e390ebadbda96ccf04a8dcbee43b`

两正式 bank 的 primary correct 均为 `2500/2500`，secondary 为 `2000/2500`，
agreement 为 `2000/2500`；camera-code fraction 为 `1.48e-5`，primary diversity
约 `0.0042–0.0070`，coverage 为 `1.67–2.13`，因此安全门失败。memorization
pass fraction 为 `1.0`、exact duplicates 为 `0`；preflight peak allocation
`922167808` bytes、wall time `14.303609110007528` seconds、storage `10120`
bytes；audit storage `54366` bytes。该结果关闭 H-NDMC 经验流形 canary，不能
授权 direct-generator architecture preflight、neural training、confirm、decoder
v2、O0-G、optical、fiber 或 promotion。

## H-JQOC closeout（valid scientific failure）

H-JQOC（Joint Quadrature Optical Capacity）已完成 retry7 的
`validate → preflight → audit → report → closeout`。retry1–retry6 的 harness
失败均保留在各自 attempt，且在 evidence 创建前或下游 consumer 创建前停止；修复内容仅限
量化分位数、reference panel device、tuple 解包、evaluator count device 和 qualification
scalar serialization。最终从 clean lineage `266083e4752d947cbbd28dd16d8b0152ad4a5e4c` 重启。

retry7 五个 evidence 均为 `status=valid`，最终分类为
`electronic_structured_input_capacity_failed`，唯一 next action 为
`close_joint_quadrature_capacity_route`。两份正式 bank 中，8D/16D/32D 的 primary
minimum recall 分别为 `0.73–0.78`、`0.69–0.78`、`0.57–0.69`，secondary minimum
recall 分别为 `0.92–0.98`、`0.96–0.98`、`0.91–0.95`；三个维度均未通过预注册
electronic semantic/safety gate。因此不能用后续 raw-intensity 或四端口 I/Q 结果补救输入
合同。raw-intensity primary/secondary accuracy 仅为 `0.075–0.092`/`0.421–0.436`；
I/Q 的 port packing、energy identity、LO-off、sign/swap、replay、finite-difference rank
和资源门通过，但这些结果只属于诊断记录。

retry7 evidence 位于
`outputs/20260826_v2/runs/20260826_joint_quadrature_optical_capacity_v1_retry7/`，SHA256 为：

- validation `f8a6231b9910cc3cab0a95c61594237ec4e72c3fa2c02e9d6891556a70e11d5a`
- preflight `43351251e919981c98eb37698a01b660ed87a3b7ded4734f51eb039b8d8ae7e0`
- audit `08b0eb64c7a02e4b0ebec7dd3f7f23818abc98d29431033efcf3cd5b17296fa7`
- report `c56d4bb4d172753ab6c97d7d40171628f3a823fecdfa96deb1ef6a3aa227b982`
- closeout `07806d105e8745a83fba65cc65258adb8733c8da53ed8d1a0c70e574dd69699c`

preflight peak allocation 为 `3353651200` bytes、wall time `32.109134116` seconds、
storage `11065` bytes；audit peak allocation 为 `4788746752` bytes、wall time
`260.960760050` seconds、storage `87355` bytes，均低于预注册资源门。结果关闭当前
joint-quadrature route，不授权 intensity 或 quadrature architecture preflight、neural
training、checkpoint、confirm、decoder v2、O0-G、optical、fiber 或 promotion。

## v3 spatial quadrature closeout（valid scientific failure）

`conditional_spatial_quadrature_v3` 已完成 electronic preflight、generator-only
training 和双 evaluator electronic admission。retry1–retry5 的 harness 失败均
保留且未创建下游 consumer；retry6 在 clean lineage `e1b16c8` 上通过 preflight
后完成 admission。该 admission 的 classification 为
`hierarchical_spatial_input_capacity_failed`，唯一 next action 为
`close_spatial_quadrature_v3_input_route`。

两份 250/class bank 中，primary accuracy/minimum recall 均为 `1.000/1.000`；
secondary accuracy/minimum recall 为 `0.900/0.176` 与 `0.896/0.116`。secondary
diversity ranges 为 `0.401–0.611` 与 `0.385–0.614`，coverage 为 `1.551` 与
`1.532`，均未达 electronic admission gates。primary code fraction 为
`0.3226/0.3250`、coverage 为 `1.001/0.997`、memorization anomalous rate 为
`0/0`，但不能抵消 secondary 失败。

本轮没有执行 quadrature preflight、400-step smoke、phase-mask 更新、联合训练、
formal-test、candidate export 或 promotion。v3 checkpoint 仅是 electronic
training artifact，不兼容旧 identity checkpoint；所有 v2 identity、H-DIDM、
H-SCF、H-NDMC 和 H-JQOC evidence 保持只读。

## 2026-08-26：Package A electronic distribution recovery closeout

Package A（Electronic Distribution Recovery）在 clean commit `a1842b8` 上完成
`validate → gate-audit → preflight → audit → report → closeout`。retry12 的六份
evidence 均为 `status=valid`；H-GFA 双 evaluator gate、三步 CUDA preflight、
H-LIR/H-LGS geometry 与 finite-difference harness、负对照和资源门均通过。
完整 audit 的有效科学分类为 `generator_range_incomplete`：R/C/U 三个 inversion arm
在两份严格互斥的 250/class validation bank 上均未同时满足双 evaluator semantics、
agreement、diversity、coverage 和 memorization 组合门。H-GFA 本身通过，label-shift
负对照被拒绝；这是 valid scientific failure，不是 invalid harness，不支持继续调参或换 seed。

六次 bank/arm 测量中，primary accuracy/minimum recall 为 `0.9888–1.0000`/
`0.9600–1.0000`，secondary 为 `0.8956–0.9172`/`0.1160–0.5360`，agreement
为 `0.8956–0.9168`；R/C/U 的两 bank `range_gate` 均为 false。memorization anomaly
为零，generator state 前后 hash 完全一致，geometry/finite-difference gate 通过，
未创建模型 checkpoint。

retry12 evidence 位于
`outputs/20260825_electronic_distribution_recovery/runs/20260825_package_a_v1_retry12/`。
SHA256 依次为 validation
`3d4c71193455c5a1d8afcd62569f4e08dbc741046793418ff8a6841f74340737`、gate-audit
`99de4cae5f83cda643fd5accd09e54795d24b74d1629539f033f4156e47ac0bf`、preflight
`be07633ca2deffec1fc3b65226172743c7a0770a00b13087b420516ceee09e06`、audit
`2253c9ed0f4284b2e81818ba6089fee1ecb82430989edfa9cf678b1c84c957dd`、report
`fc2d110903b3eaa0a42234a646d4be9249e2ebe2962f58254c2e1dfb9048ae36`、closeout
`70ed0fe02e9242ec630b44f3e46dd386950ef8d10d1ab5769664b5cb211a8cab`。

Package A 的真实消费者是 Package B teacher audit 与未来 student 架构选择；本轮不授权
Package B 自动启动，必须另行审查并建立新 lineage。DDPM、student、optical、phase、
quadrature、OLS、fiber training、promotion 和 candidate export 均保持禁止。

## 2026-08-27：Package B teacher distribution admission closeout（valid scientific failure）

Package B（DDPM–VAE Teacher Distribution Admission）在 clean producer commit
`381941f` 上完成 v1 `lineage → preflight → generate`，随后在 clean consumer
commit `ac7baec` 上完成 v2 `validate → audit → report → closeout`。producer
锁定并验证了 first-stage checkpoint `e553bf…afdd`、matched diffusion
checkpoint `60f759…cd0`、2026-06-03 source runs、acceptance manifest、Package A
H-GFA evidence、双 evaluator qualification 和 MNIST split。完整 Gaussian latent
noise、200-step DDIM、guidance 2.0 以及两个独立 250/class bank 均按 manifest
生成；生成过程 320/320 shard、producer/preflight 无 evaluator calls、无 optimizer
或 checkpoint，preflight peak CUDA allocation 为 `173581824` bytes，generation
peak 为 `197254656` bytes。

lineage/preflight/bank/consumer evidence 位于
`outputs/20260827_teacher_distribution_admission/runs/20260827_package_b_v1/`，
SHA256 为：

- lineage `b1e385c2b3cba80b91c81e958c1a26804a39554c09daddb6bca0ba5e73bb1a6c`
- preflight `66b65fb5ccd04951201a9634fe101e4cff982f2c1a9d9cecd6c3abd6a280d922`
- teacher banks manifest `c9aebd1e21e8ed185eb1c0db35902a7a4c67c90d01ebd25153a45ca496139ccb`
- validation `ad9f4896f8d43fa03230c5e0d00b10c7f7c5b0e0c590a171f76b4d8a2084f099`
- audit `0369e9500158a94a675e393674f3deff6edf8de0e5d6775e617750b17dbfea92`
- report `0cdd29f6b92656355165926e0ef530fa1e31b40444875476b7c8ca553d1dea5e`
- closeout `2620311e2bedb6ba6188afc2eae5af5fb9947410e8d70dac892971e2fdc3c6f7`
- synthesis `66f29aff3bf5c915c25195f83914c265335a6f5068ed3e9dbec71b64770b97ec`

The audit was `status=valid`, with negative-control rejection, finite values,
state/lineage/hash binding and resource gates passing. Its scientific classification
is `teacher_distribution_failed`, first failed gate
`teacher_dual_evaluator_range_gate`, and the sole permitted closeout action is
`close_teacher_route`. Bank 0 had primary/secondary accuracy `0.3524/0.2212`,
agreement `0.1676`, primary/secondary coverage `3.0909/2.8327`; bank 1 had
`0.3428/0.2248`, agreement `0.1716`, and coverage `3.1483/2.8622`. Primary
minimum recall was `0.036` for primary and `0.008` for secondary in both banks; multiple diversity values also missed the
registered `[0.70, 1.30]` range. Memorization anomalous rate was `0` in both
banks, but does not rescue the failed semantic/range gate. The cyclic label-shift
negative control had accuracy `0.0598` and was rejected. Audit peak CUDA allocation
was `276411392` bytes over `2300` evaluator calls.

This is a valid teacher-distribution failure, not an invalid harness. The named
consumer is Package C's multiscale student admission and architecture decision;
that decision is rejected by this evidence. Package C student distillation is
forbidden, and the current DDPM–VAE teacher route is closed pending a separately
designed teacher route with new lineage. DDPM/VAE/student/optical/phase/quadrature/
OLS/fiber training, promotion and candidate export remain forbidden. No Package C
or other downstream stage was started.

## 2026-08-27：Package B2 teacher failure attribution closeout

Package B2（Teacher Failure Attribution）在 clean commit `36597b9` 上完成独立的
`lineage → preflight → generate → validate → audit → report → closeout`。本轮锁定
Package B 的 matched first-stage checkpoint `e553bf…afdd`、matched diffusion
checkpoint `60f759…cd0`、原 Package B banks、Package A H-GFA banks、双 evaluator、
MNIST split 和全部正式阈值；没有搜索、选择或训练新模型。

正式结论为有效科学归因 `teacher_first_stage_distribution_failed`，首个失败门为
`first_stage_reconstruction_distribution_gate`，不是 `invalid_harness`。R（真实
MNIST 正对照）两个 bank 同时通过；primary accuracy/minimum recall 为
`0.9880/0.9640` 与 `0.9872/0.9440`，secondary 为 `0.9888/0.9800` 与
`0.9872/0.9720`，agreement 为 `0.9868/0.9852`。cyclic label-shift 负对照被拒绝，
R/V/D accuracy 分别为 `0.0006/0.0024/0.0598`。

V=`decode(encode(real).mu)` 在两个 bank 都未通过：bank 0 的 primary/secondary
accuracy/minimum recall 为 `0.9692/0.8920`、`0.9444/0.8520`，agreement
`0.9316`；bank 1 为 `0.9704/0.8960`、`0.9464/0.8840`，agreement `0.9380`。
D 与锁定 Package B teacher replay 完全一致，并重现其 failure（bank 0/1 的
primary/secondary accuracy `0.3524/0.2212`、`0.3428/0.2248`，minimum recall
`0.036/0.008`，coverage 约 `3.09/2.83` 和 `3.15/2.86`，agreement 约 `0.17`）。
由于 V 先失败，B2 不把 D 单独解释为 diffusion failure。evaluator-independent
latent statistics 已记录；normalized encoded-real `mu` 与 DDPM latent 的
classwise mean L2 distance 覆盖约 `5.03–11.08`。

v3 evidence 位于
`outputs/20260827_teacher_failure_attribution/runs/20260827_package_b2_v3/`：
lineage `9c91e346127576143fe49bd7ea5f14b479ffd8647cce5b4b24c818bee0db2b00`，
preflight `c67d195f9c7ce1f9917982c3fc747fbf9a73ddabee0eb049431a84d039a6ad3b`，
generation state `48b3115441713d0203978403e97f87b2f0fe7a1173f17ed66e72b6e2ae31e1d1`，
artifact `5f018c0f261d9bd17f6be19cd6f242b1cc2f980caa01200c2664c05d8b3bca0f`，
validation `f8455fb8f369e9e96a53bca532c3862c1bfae5c5b0b8323cbc1d1de6e34aa307`，
audit `fffd9be3a03880b76780f168ce54c7aea3bda8193db7c4459c9cad398b8065e3`，
report `5ad140a9ee8cb431b50c258d6141159fa69b4acb24364ccadb5c10146bfa8fdb`，
closeout `0fed705e605fd540309e6904a4214eb7de9f668d2d1496868087e9fda6e58515`，
synthesis `3a70799aea9dad02f9126b36b5337f6e6f21c9ab54de2c9792268fcf7266312f`。

此前 v1 preflight 的 batch-10/batch-16 replay mismatch 和 v2 artifact 漏写 labels
均已 fail-closed 且未被消费；修复后 v3 重新生成 320/320 shards。producer/preflight
无 evaluator calls，生成峰值 `213146112` bytes，audit 峰值 `276390912` bytes，
state hash 前后不变。

Package B2 的真实消费者是 teacher redesign 决策；唯一允许的下一步是另立
`Package B3-VAE`，重新设计/训练 first-stage，并先通过 reconstruction distribution
gate。当前结果不授权 `Package B3-DDPM`、Package C、student/teacher/optical/
phase/quadrature/OLS/fiber training、candidate export 或 promotion。若未来 first-stage
通过，仍须用新 teacher lineage 重新通过原 Package B 双 bank admission，只有得到
`teacher_distribution_qualified` 才可审查 Package C。

### B2 evidence-schema amendment

上述 v3 目录仅保留为未消费的 schema-diagnostic run：其 producer manifest 缺少
`seed` 和 `input_artifact_hashes`，因此不承担最终科学证据。修复后的最终权威 run
是 `outputs/20260827_teacher_failure_attribution/runs/20260827_package_b2_v4/`，在
clean commit `9dddb78` 上完成全流程；其 attribution artifact 已通过顶层 required
evidence-field 核验。v4 的 lineage/preflight/state/artifact/validation/audit/report/
closeout SHA256 依次为：

`4fae3feabb1bac774827e4c46719095d5084e75ab41df2789df5cb8906587ae3`、
`812e28350315bddb26023d8f56dd40b65a306d83c68c2ecf119e1b8081e73643`、
`3f7915386bfe8b2c78b0a9aef8a549b03e5165ce609def3609eb51d0c4684a1d`、
`af04f5824784943b0d3d0a33f2404802644f4d96bb7cbb9dfb53ef2d56683c45`、
`9b9a79d3efcd6fb1b8f10eac152acf2d72cf27d6371742999deee48749eae6bc`、
`a1d64208e3c55cc468229ca11552f43dc55f93c2dae3e163435b06bcdd06a71f`、
`aecb9b9a4f3a8e521c22b3cf51f11330fcf25678d3788502caebac0742e9b0e7`、
`c760ddcc76f378f5517e280d876edaa9015bc3ea7f6216f846ecaf1813388c75`。

## 2026-08-27：Package B3-T.2 FP32 semantic-preserving compute cleanup closeout

Package B3-T.2 was completed as a research-only throughput diagnostic. It replayed
the locked B3-T1 CUDA contract (8 warmup batches, 200 measured training batches,
100 development batches, FP32, microbatch 16, gradient accumulation 4) before
reviewing synchronization, exact same-batch reuse, and Python/module dispatch.
The T1 baseline replay was reproducible: total profiling-window deviation was
`1.93%`, development deviation was `2.16%`, and all registered phase tolerances
passed.

The authoritative run is
`outputs/20260827_teacher_first_stage_recovery/b3_t2_v3/`, executed from clean
commit `83e0d74`. Its classification is
`b3_fp32_semantic_preserving_optimization_insufficient`; this is an engineering
throughput result, not a B3 model-quality result. The final FP32 projected epoch
was `31.135 min` versus the locked `25 min` contract, with total replay ratio
`0.981x`. Backward remained `76.86%`, model forward `15.92%`, and measured CUDA
synchronization wait `8.85%`. The low-risk candidate matrix accepted no new
intervention: safety finite checks remain fail-closed, profiler-only synchronizes
are not formal training overhead, and the already-small data/host/loss portions
do not justify unscoped work.

The run remained resource-safe with CUDA peak `2.589 GiB`, host RSS peak
`1.751 GiB`, finite values verified, inherited T1 resume exactness, and zero
evaluator calls. The T2 implementation commits are `32e2b34`, `b3a6253`, and
`83e0d74`. Earlier `b3_t2_v1` and `b3_t2_v2` directories are retained as
non-consumed diagnostic records: each completed the GPU profiling window but
hit a T2 evidence-writer programming defect, so neither is a timing or model
conclusion and neither was spliced into v3.

The six required v3 evidence hashes are:

- baseline replay `64a531ac55a5b8059ad28ac3f040c96c036766db01cffdaa9877dc646eb27dd8`
- synchronization attribution `ab932c6f3da07bbf5c31d69132902eee8663405064992f46aaeefc864b00abc5`
- optimization matrix `a0d3f6bfaa555cd2ab3ce0d3d5a0392f1e8914b26329d246902ab1fa7668c86a`
- equivalence evidence `7e6674e99bee213e77b0927b021653b530882cd7f9e99abe58ae9c291572d484`
- final timing `2447c30fe2cb0082c4974097dd950f33e101286299d351d39b2aa987156d26b4`
- closeout `351b6175deb930f1d7bf6d70ca9f3726f9fee34c0aaaa5f74713744a342341d6`

The only permitted follow-up is a separately reviewed
`numerical_gpu_acceleration_preflight`. T2 itself does not authorize AMP,
BF16/FP16, TF32 enablement, `torch.compile`, formal B3-VAE training,
B3-DDPM, Package C, student/optical/phase/quadrature/OLS/fiber training,
candidate export, or promotion. The T2 closeout is a hard stop.

## 2026-08-27：Package B3-T.3 numerical GPU acceleration preflight closeout

T3 was started from the T2 authority `b3_t2_v3` and clean commit `ce4dfe0`.
Hardware audit passed for the RTX 3060 Laptop GPU (compute capability `8.6`,
CUDA `12.1`, cuDNN `90100`, PyTorch `2.5.1+cu121`); native TF32, BF16, FP16
Tensor Core and GradScaler capabilities were detected. The FP32 reference
completed 100 optimizer steps with finite values, four parameter updates in the
three-step resume check, exact model/optimizer/scheduler/RNG restoration, and
zero evaluator calls.

The numerical arms did not reach a valid decision. The single T3 process was
terminated by the WSL Python memory guard before TF32/BF16/FP16 arm evidence
could be completed: the guard recorded process RSS `2,992,820 KiB`, total
Python RSS `6,473,368 KiB`, and limit `6,291,456 KiB`. The fail-closed
classification is therefore `resource_aborted`, not numerical equivalence
failure and not a model-quality result. No throughput comparison and no
`torch.compile` stage was run. The 25-minute resource contract was unchanged.

Authoritative evidence is under
`outputs/20260827_numerical_gpu_acceleration_preflight/runs/20260827_package_b3_t3_v1/`.
The hardware, equivalence contract, FP32 reference, TF32/BF16/FP16 arm records,
candidate comparison, throughput comparison and closeout SHA256 values are:

- hardware `3e23273842347e6d717b9cb2c3b7cfa38208e8279da8fd4b1f9b160d0f70c6a0`
- equivalence contract `f9d69526363ff9d5f4c9aa7ae2059fe0a80524c2d1d1418b068f099a74a75fda`
- FP32 reference `631c34f87dacca77c8dbf35862163cf7c757fd72e236d210fe0158c7d5be8e16`
- TF32 `51b318974cde1e217aebee78d1bef4dd2aca25e1df00c5242a760ed0063c0f64`
- BF16 `0263307acd92e1b161fdb6d29304c37e2f33e2e2f291446ab2621fc8266aa7d1`
- FP16 `0e5321ea0f9b7f57870d2f42db6c8bc35958e5db7354bc689447f9b15f91f77b`
- candidate comparison `a3e478ac5d5fa82bbb2fa763aa4c7899d59576df7004b4e9e9f84427b96ac060`
- throughput comparison `4dffcfdf198b4b6ec3d59d9ffc37526f53cb5dcff295bd65ad2521786bdebda1`
- closeout `eade8b4f013dff8d90e54779d5f6ab0637f34943c69c233158c73b5719a50ae7`

The real consumer is numerical execution-contract design, not a model-quality
claim. The only permitted next action is `resource_contract_revalidation`;
this requires a separately reviewed plan and must not silently enlarge swap,
change the 25-minute contract, or rerun partial arms. Formal B3-VAE training,
B3-DDPM, Package C, student/optical/phase/quadrature/OLS/fiber training,
AMP/compile combination, candidate export and promotion remain forbidden.

## 2026-08-27：Package B3 research line temporarily paused

The Package B3 research line is now intentionally paused after the
Package B3-T.3 resource-aborted closeout. The scientific status remains
`not_evaluated`: no numerical candidate was selected, formal B3-VAE training
was not started, and no model-quality conclusion may be drawn from the
partial canary. The authoritative T3 evidence and closeout remain at
`outputs/20260827_numerical_gpu_acceleration_preflight/runs/20260827_package_b3_t3_v1/`.

No further CUDA job, retry, resource-contract expansion, AMP/TF32/BF16/FP16
enablement, or compile combination is authorized during this pause. B3-VAE,
B3-DDPM, Package C, student/optical/phase/quadrature/OLS/fiber training,
checkpoint promotion and candidate export remain prohibited. The only
recorded resumption path is a separately reviewed
`resource_contract_revalidation`; resumption is not automatic and must not
reuse partial candidate-arm evidence as a scientific result.

## Independent H-LOMA-I0 optical-manifold line (2026-08-27)

The next implementation is an independent diagnostic v2 research line,
`research/v2-lightgen-optical-manifold-v1`; it does not resume Package B3 and
does not reopen the closed synthetic fiber route. The registered scientific
question is whether the complete current formal 2+2 phase-only system with
complex Identity OLS can learn real-MNIST reconstruction under fixed exposure.

The first delivery is limited to a research contract, configuration, data
rendering boundary, Identity optical autoencoder, validation, and a three-step
CUDA preflight. It does not provide formal reconstruction training and does
not enter KL, latent sampling, SMF, FMF, evaluator, or generation stages. Only
encoder/decoder phase masks may update; labels select deterministic one-per-
class representatives and are not model inputs.

Validation must reject dirty/stale lineage, non-formal geometry, non-Identity
OLS, coordinate or exposure mismatches, missing local MNIST, or any extra
trainable parameter before creating an optimizer or run. Preflight must use
the same 2048-grid formal path with microbatch one and activation checkpointing,
then prove finite values, all four phase-layer updates, frozen-state isolation,
loss decrease, zero evaluator calls, resource limits, and exact 3-step versus
2+1 resume. Any failed execution gate is `invalid_harness` and stops the line;
even a pass records `training_authorized=false` and
`formal_promotion=false`. The unique downstream decision is manual review of
the Identity reconstruction training contract.

### H-LOMA-I0 首轮 preflight closeout

首轮 authoritative attempt 使用 clean commit `20b0301`，validation evidence
为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_v1_attempt_02/validation.json`
（SHA256 `9c763024021707314decb945d5b7a692d1b86e20421047a89d81746b70dfe57e`），
preflight evidence 为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_v1_attempt_02/preflight.json`
（SHA256 `67a4ff53b86e329596234e8dd7154ce85ac2f22cc6974f089a2acf728eefe166`）。
resume artifact 为
`preflight.resume.pt`（SHA256
`1e888e5fd8020e09df5dbc6ee9ef876c31d168da2e51ccc0501fe7dc67e88267`）。

Validation 与 preflight 均为 `passed`，但仍是 diagnostic-only：3 步 loss
从 `0.0727686286` 降到 `0.0724386498`，四个 phase mask 均有非零 gradient 和
真实更新，2+1 resume exact，peak CUDA `380651008` bytes、wall `5.5566s`、
storage `494451` bytes，evaluator calls 为 `0`。唯一下一动作是
`manual_review_identity_reconstruction_training_contract`；本轮不启动正式
reconstruction training、SMF/FMF、latent sampling、evaluator 或 promotion。

### H-LOMA-I0 1000 步 pilot 授权

项目负责人已将首轮 preflight 结果（SHA256
`67a4ff53b86e329596234e8dd7154ce85ac2f22cc6974f089a2acf728eefe166`）绑定到
`lightgen_optical_manifold_identity_pilot_v1` 合同，并人工授权最多 1000 步
的 diagnostic research training。授权仅覆盖固定 55k/5k MNIST 划分、每步每类
一个样本、Identity OLS 下的 paired reconstruction；不授权完整 epoch、SMF/FMF、
latent sampling、evaluator、生成实验或 formal promotion。

Pilot 的预注册科学门是固定 validation 中每类 100 张的 held-out panel 相对零
相位 baseline 的 MSE 至少下降 5%，且最终 paired MSE 小于循环错配 target 对照。
训练必须先通过 training validation 与三步 training preflight（含负对照及 2+1
resume exact）。失败即停止；完整 1000 步但科学门未过记为 `valid_failed`。无论
结果如何，evidence 均保持 `training_authorized=false`、`formal_promotion=false`；
通过后的唯一下一动作是
`manual_review_full_epoch_identity_reconstruction_contract`。

### H-LOMA-I0 1000 步 pilot closeout

Pilot attempt-03 在 clean commit `45e6f8d` 上完成并通过全部 harness 与科学门。
training validation evidence 为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_pilot_v1_attempt_03/training_validation.json`
（SHA256
`6d138a40f1d780f1d8ce1cf6ed6536d523c54e4908833e64d4d4cb84fdc76a6b`），training
preflight evidence 为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_pilot_v1_attempt_03/training_preflight.json`
（SHA256
`cfa7ed3452ff41800e39a5a35aecb73df2905de2c3bc970fe863f5affe1649c7`），其
resume artifact SHA256 为
`d5d16fef76659ca8b236275feeef4f08d760c59bc6a44301bab86f21e013cf9a`。
最终 training evidence 为
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_pilot_v1_attempt_03/training.json`
（SHA256
`4127f80194f02750737c2e2a1e3f8f7ee4b90df8798cbdd53c4c20369b4a681f`）。

1000 步完整运行耗时 `877.2158s`、peak CUDA `481032704` bytes、run storage
`5959746` bytes，evaluator calls 为 `0`。固定 1000-image held-out panel 的
MSE 从 zero-phase baseline `0.0737866908` 降至 paired `0.0606269501`，相对
下降 `17.8348%`；paired MSE 低于 cyclic-permuted target 的 `0.0694154128`。
最终状态为 `passed`，但 evidence 仍记录 `training_authorized=false`、
`formal_promotion=false`。唯一下一动作是
`manual_review_full_epoch_identity_reconstruction_contract`；本轮到此停止，
不启动完整 epoch、SMF/FMF、latent sampling、evaluator 或生成实验。

### H-LOMA-I0 full-epoch continuation 授权

基于 pilot attempt-03 的 `passed` evidence，项目负责人授权在
`research/v2-lightgen-optical-manifold-full-epoch-v1` 分支执行一次受限的
Identity reconstruction full epoch。该阶段从 pilot final step `1000` 继承
模型、Adam、RNG 和 data cursor，仅消费剩余 45,000 张 train 样本，至 global
step `5500` 停止；不授权额外 epoch、latent/OLS 实验、SMF/FMF、evaluator、
生成或 formal promotion。

full-epoch 的成功门是最终 paired held-out MSE 严格低于 pilot 的
`0.06062695011496544`，并低于最终 cyclic-permuted target 对照。资源门为
preflight `300s`、full epoch `4500s`、peak CUDA `5GiB`、free CUDA 至少
`4GiB`、run storage `2GiB`。固定 visual panel 为 held-out 中每类 10 张，
仅用于人工审查。任何失败均停止；通过后的唯一下一动作是
`manual_review_optical_latent_ols_contract`。

### H-LOMA-I0 full-epoch continuation closeout

Implementation was completed on local branch
`research/v2-lightgen-optical-manifold-full-epoch-v1`. The implementation
commits are `06400b4` (authorization/config/governance), `225cf26`
(lifecycle/CLI/tests), `ab868d8` (deterministic CPU-shadow Adam contract),
`83bba25` (source optimizer-state isolation), and `4bbe438` (preflight
manifest binding). The final clean commit consumed by the authoritative run
is `4bbe438`.

Three earlier attempts were retained and not overwritten. Attempt-01
preflight was `invalid_harness` because non-zero CUDA Adam updates were not
bitwise stable across independently allocated runtimes (SHA256
`9ba6cae22aa74f0b3c1818046b1149167e5df645b2f32f3610634327102d10d7`).
Attempt-02 remained `invalid_harness` after the first corrective implementation
because a runtime mutated the shared source optimizer state (SHA256
`aa3288cf31ccc76268392cdbd977e742a23aef95d1c6b33f57b083110d9e179c`).
Attempt-03 passed preflight, but its train loader correctly rejected an
incomplete preflight manifest binding before creating an optimizer or run
(SHA256 `7c81172e08de737df2a89179d8c0086e7126a6e50217018005be45a419a1e448`);
no training occurred in that attempt. These are harness records, not model
failures.

Authoritative attempt-04 validation evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_validation.json`
(SHA256
`258af9978653a3552f8786b0fc7d76e8434347ed8c7bd5833f41f0f21f9351cd`), and
preflight evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_preflight.json`
(SHA256
`e14f733bd8a83738892404a71452972eff11a7c6014d174aff1a33703e721a75`). The
preflight resume artifact SHA256 is
`d58f7101d5241853abd20795a84274f6a67451142b113a073abc6d074f8b7d92`.
All preflight gates passed, including real CUDA formal 2+2 Identity OLS,
four phase-layer gradients and updates, frozen-state isolation, evaluator
calls `0`, cyclic mismatch control, and exact 2+1 resume. Peak allocation was
`379527680` bytes, wall time `14.4956s`, and storage `514594` bytes.

The single authorized continuation completed global steps `1001` through
`5500` in `3809.5485s`. Final training evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_training.json`
(SHA256
`c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948`). The
final checkpoint is `full_epoch_final_step_005500.pt` (SHA256
`56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f`). The
fixed 1000-image held-out paired MSE was
`0.06062695011496544` at the pilot checkpoint and
`0.03437697887420654` after continuation; the final cyclic-permuted target
MSE was `0.06315752118825912`. Thus both preregistered scientific gates
passed. Peak CUDA allocation was `480711168` bytes, run storage was
`75434116` bytes, and evaluator calls were `0`.

The review-only 100-image visual panel contains the original 28x28 inputs,
200x200 targets, raw CMOS tensors, reconstructions, and fixed target/
reconstruction PNGs. Its tensor artifact is `full_epoch_visual_panel.pt`
(SHA256 `9328818e029852ff3ee8e9f05071ce30ff325ebcb89ba077773023466dad2a6e`);
target PNG SHA256 is
`7f413d88cd0b80219a0c38c054f696295feb0b8f61c58e9213658a6bce006284` and
reconstruction PNG SHA256 is
`01d4bd3fec47d2224df560aa45684b3051eb034acb7e723212038c774ebad55e`.

Final status is `passed` for this bounded diagnostic contract. Evidence still
records `training_authorized=false` and `formal_promotion=false`. The only
permitted next action is `manual_review_optical_latent_ols_contract`; no extra
epoch, latent/OLS experiment, SMF/FMF, evaluator, generation, or promotion is
authorized.

### H-LOMA-I0 next action: Identity reconstruction attribution audit

The full-epoch result is sufficient to open one no-update PLAN-3 closeout
audit, but not to claim a learned latent manifold. The audit is registered on
branch `research/v2-lightgen-optical-manifold-identity-attribution-v1` and is
bound to full-epoch evidence SHA256
`c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948`, final
checkpoint SHA256
`56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f`, and MNIST
manifest SHA256
`64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8`.

It will use the fixed 1000-image held-out panel and phase/template controls to
test instance sensitivity, phase dependence, and within-class geometry. No
optimizer, checkpoint, latent observer, KL, SMF/FMF, evaluator, generation, or
promotion is authorized. A pass can only request
`manual_review_identity_latent_observer_contract`; a valid scientific failure
requests `manual_review_identity_reconstruction_objective_contract`.

### H-LOMA-I0 PLAN 3 attribution closeout (2026-08-27)

The first two attribution attempts were retained as `invalid_harness` because
their source-MSE comparison used a CPU offline reduction instead of the
authoritative CUDA evaluator path. Commit `f7d38c5` added a direct replay of
`optical_manifold_training._evaluate` and kept the offline attribution metric
separate. The new attempt-03 validation evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_reconstruction_audit_v1_attempt_03/reconstruction_audit_validation.json`
with SHA256
`fe86d7b1990f65d7f2cf51332bc1acca917302160afc1530471106ecdb442bd6`; the
passed authoritative audit is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_reconstruction_audit_v1_attempt_03/reconstruction_audit.json`
with SHA256
`31ee69fd050902d0191fbc1b6847ebde040d8f91e0f01d9a6773a7bd8d284905`.

The replay exactly reproduced source paired MSE `0.03437697887420654`. All
phase/template controls, instance geometry and retrieval gates passed; model
state was unchanged, evaluator calls were `0`, wall time was `290.18s`, and
peak CUDA allocation was `447156736` bytes. This closes PLAN 3's attribution
audit as a valid diagnostic result: reconstruction is phase-dependent and
instance-sensitive, but no latent-manifold claim is made.

The only permitted next action is
`manual_review_identity_latent_observer_contract`. PLAN 4 must remain
observation-only and separately bound to this passed evidence; no optimizer,
checkpoint, KL, SMF/FMF, evaluator, generation, or promotion is authorized.

### H-LOMA-I0 PLAN 4 latent observer closeout (2026-08-28)

The observer harness was repaired on clean commit `3e1476d` to require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before the Python process enters CUDA. A
first CUDA run in attempt-02 was retained only as an environment startup
failure. Attempt-03 then completed, but its endpoint comparison emitted a
broadcasting warning because the channel dimension was dropped; that evidence
is retained as superseded harness output and is not authoritative. Commit
`45a6826` added exact camera-shape checking and a regression test.

The authoritative attempt-04 validation evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_latent_observer_v1_attempt_04/latent_observer_validation.json`
with SHA256
`4dda0fb96e144c91d48c8ccbf1395e043bc215b988b7f6ac169b0c91a6530724`; the
passed observer evidence is
`outputs/20260827_v2/runs/lightgen_optical_manifold_identity_latent_observer_v1_attempt_04/latent_observer.json`
with SHA256
`36a1b918991affaf6093033b5ce5139f2d6f87517f72373b16751f357bb63507`.
The saved pre-OLS field artifact has SHA256
`c838a2c85ad4597ebde005e79ed4bceb8726beb8597d8622a578f93f455c7341`, and
the 50-image interpolation panel has SHA256
`b8c3553de57ee9e36bea463a957d82125c540667b61f572e08dc081c9a6c4954`.

The observer extracted a finite `complex64` field of shape
`[1000,1,100,100]`. Normalized pre-OLS true-label inter/intra distance ratio
was `1.2041758298873901`, above one and above the seed-1234 permuted-label
ratio `1.0003329515457153`; entropy effective rank was `38.74065017700195`
and participation rank was `20.190610885620117`. The rendered-input reference
was recorded with the same centered-Gram definitions. Authoritative source
reconstruction MSE reproduced `0.03437697887420654`; pre/post Identity OLS
fields were exact, all endpoint camera MSEs were `0.0`, model state was
unchanged, evaluator calls were `0`, peak CUDA allocation was `446995968`
bytes, wall time was `64.6113s`, and storage was `176635673` bytes.

This is a valid observation-only result: the Identity pre-OLS representation
has finite, label-associated and instance-level structure under the registered
diagnostics. It does not establish a generative latent prior, KL suitability,
SMF/FMF benefit, or formal promotion. t-SNE remains a separate non-gating
visualization task. The sole next action is
`manual_review_identity_manifold_regularization_contract`; evidence continues
to record `training_authorized=false` and `formal_promotion=false`.

### H-LOMA-I0 PLAN 5 aggregate-prior regularization pilot authorization

The current clean baseline is `154a301` on the new local branch
`research/v2-lightgen-optical-manifold-regularization-pilot-v1`. The project
owner has authorized one bounded diagnostic pilot with matched `regularized`
and `mse_only` arms, 100 updates per arm, 100 examples per update,
microbatch 1, and a shared 10,000-example no-replacement schedule. Both arms
reset Adam and inherit only the step-5500 Identity checkpoint weights.

The regularized arm consumes the passed PLAN 5 fixed 32D projection, 64 SWD
directions, and lambda `0.002733644070001058`. The lifecycle must first pass
training validation and the real CUDA three-step/2+1-resume preflight; a
preflight failure stops the pilot. The pilot is diagnostic-only and does not
authorize extra epochs, KL, latent sampling, SMF/FMF, generation, evaluator,
or promotion. All evidence continues to require
`training_authorized=false` and `formal_promotion=false`.

## H-LOMA-I0：SMF bottleneck attribution audit（2026-08-29）

当前 fixed-OLS recovery attempt-02 为有效科学失败：coupling 臂相对 MSE
control 的 power gain 只有 `0.2014` 个百分点，不能进入 1000-step continuation。
KL 路径不是当前阻塞，KL 臂已低于初始化和 no-KL 对照。

下一步从 clean `37e2b1b` 的独立分支执行 no-update attribution，消费 attempt-02
evidence 与 checkpoints。实验拆分 SMF LP01 投影容量、既有 FMF M2 observation-only
容量、encoder/decoder `2×2` cross-swap，以及 `1×/2×/4×/8×` coupling 梯度剂量；
不训练、不改 OLS transport、不放宽历史 `+5pp` 门限。审计结果只决定下一个人工
合同，不自动授权 dose pilot、FMF、生成或 promotion。

## H-LOMA-CBOGT-S2-TRBA：Task-Relevant Bottleneck Attribution（2026-09-01）

TRBA 是 RMRI metric-confound closeout 后唯一执行的冻结、无更新 task-relevant
归因，消费者是项目负责人对当前 Scaled-Chen v1 是否存在可行动的 task-relevant
OLS/decoder 瓶颈及是否关闭本生命周期的决定。它不改 physical forward、formal
runtime、OLS、checkpoint、训练目标、detector 或 prior；raw MRA retention 只作历史
上下文，不能作为本次主结论。

实现与审计绑定 clean implementation commit
`7c1c0f8dcfb878049260043441a117456e5ebe6a`。attempt-01 因首次发布时 contract
JSON 含未序列化 tensor 而保留为不可消费的部分产物；不得覆盖。权威 CUDA
attempt-02 位于
`outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/`，
validation/audit/compact/panel SHA-256 分别为
`3bec36f315c872a05bcb3381929987d8fff57914351a2d5bef4ff1eebda0c659`、
`b6be2c6f97dbdcb1d02ad0f2395eca438ebeb619196758a57eb8cc26f31745c0`、
`a6015a3fa594515fc24ac1af6cb684d3aa70cd703ae22ae9e8ab3924c86f633a`、
`ec76f0760a743ef8fd845b6b93969fac727619e439b4f03109fb40218cc46319`。

审计固定 MRA selection（MNIST test 5000–9999，每类 100）与 confirmation（train
55000–59999，5000）。主指标是单样本全局相位、共同坐标变换不变的 normalized
complex Gram geometry；C0 no-op、C1 global phase、C2 common permutation/reversal、
C3 common unit-modulus diagonal controls 均通过，最大 control drift 为 geometry
`1.48e-7`、label `5.63e-7`、retrieval `0`。OLS 段 selection/confirmation geometry
degradation 为 `0.171023/0.169979`，但 task-advantage degradation 为
`-0.615485/-0.578570`、retrieval degradation 为 `0/0.002600`，因此仅为
`generic_geometry_change_only`；decoder-input→decoder-L2 degradation 仅
`0.010264/0.010617`，任务与 retrieval 均无 material loss，段状态为
`not_supported`。

TRBA 总体是完整 harness 下的 `valid_failed`，分类 T4
`task_relevant_bottleneck_not_supported`，唯一下一动作是
`manual_review_scaled_chen_bottleneck_attribution_closeout_contract`。这不是
`distributed_bottleneck_supported` 的 task-relevant 复现，也不删除历史 MRA/RMRI
classification；`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 保持分离。
TRBA 不授权训练、FMF、exposure retuning、prior sampling、generation、canary 或
formal promotion。relay family 继续保持
`scaled_chen_v1_local_backend_relay_attribution_family=closed_metric_confound_reinterpreted`。

治理与资源门全部通过：`optimizer_created=false`、`backward=false`、
`evaluator_calls=0`、`checkpoint_written=false`、`prior_sampled=false`、
`physical_forward_modified=false`；峰值 CUDA `2281484800` bytes，输出约 `517995`
bytes。默认全量 pytest 为 `1121 passed, 12 deselected, 1 failed`，唯一失败是与
TRBA 无关的 near-conjugate Windows 长临时路径创建问题；TRBA/邻近研究测试、
compileall、Ruff、`git diff --check` 均通过。

## H-LOMA-CBOGT-S1-RMRI：Relay Metric Reinterpretation Audit closeout（2026-09-01）

RMRI 已从 clean implementation commit
`16896f06b31549120adf6235587619fe2820af83` 完成冻结、无更新、same-panel、same-threshold
回放。它消费 HFRR、MRA、DCM、BOSA 的已验证 evidence，只在 decoder L2/chip → CMOS
incident complex field 的合法 relay 段使用 HFRR analytic B=0 canonicalization；不修改
physical forward、生产 relay、OLS、detector、checkpoint 或历史 evidence。BOSA 只重放已
注册的 1000 张 selection，confirmation 仍明确为未执行。

权威 attempt-03 目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/`，
validation/audit/compact/panel SHA-256 分别为
`a87769fb638d5bffa1f77bc31c931fd3999c398a23c1a1f17d51d01eb91882eb`、
`d409729f9b111c024c1913cef90c6cd2ebe778efa9990b836a955783f378ba00`、
`cd39f2fed5719572a558be2bbbf139cdb7cc657a08d74c2be4cd8b69b71c35b8`、
`6b7f9225624fa404a0743a417903b27a7113d100315ea9b1804043c63aafb079`。四个 parent hashes、
checkpoint hashes、selection/confirmation 隔离、canonicalizer identity、state freeze 和
resource gates 均通过；attempt-01/02 仅有旧实现 validation，未形成 audit，保留但不消费。

### 真实结果

- DCM raw relay selection/confirmation=`0.65712498/0.60306906`，readout=
  `0.88675454/0.97353794`，与历史 anchors 一致；canonicalized relay=
  `0.98158839/0.98122396`，canonicalized label/retrieval signal 无 material drop。当前
  DCM interpretation 为 `relay_propagation_bottleneck_reinterpreted_metric_confound`（D1）。
  历史 `relay_propagation_bottleneck_supported` 与 HFRR
  `relay_complex_retention_metric_confound_supported` 必须继续保留。
- MRA raw retention 完全复现：encoder `slm→pre-OLS=0.91817469/0.92159733`，OLS
  `pre-OLS→modal=0.81449253/0.80923885`，`modal→post-OLS=0.91388854/0.91189128`，
  `post-OLS→decoder input=1/1`，decoder internal `decoder input→L2=0.82043698/0.81888384`。
  OLS 与 decoder internal 两个独立段仍 material，当前 MRA interpretation 为
  `distributed_bottleneck_supported_canonicalized`（M1）。decoder Jacobian rank=100
  仍只支持“不能归约为 decoder capacity-only”。
- BOSA raw `R_prop(I→I/I→S/S→I/S→S)=0.99121143/0.32563974/1.00856667/0.65712498`；
  canonicalized=`1.00012124/0.99160306/1.00003910/0.98158839`，representation
  contrasts=`0.00008214/0.01001467`，所以当前 BOSA interpretation 为
  `backend_observable_subspace_mismatch_not_supported_reinterpreted`（B1），confirmation
  为 `false`。

RMRI overall status 为 `valid_failed`：这是完整测量后的科学重解释状态，不是
`invalid_harness`，也不授予任何 promotion。`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 和
`scaled_chen_v1_local_relay_tuning_family=closed_no_local_correction_supported` 均保持。
唯一下一动作是 `manual_review_relay_metric_confound_closeout_contract`；在项目负责人
人工审查前不得训练、启动 FMF、采样 prior、生成、调整 relay 或 formal promotion。
CUDA 峰值 `3626808320` bytes、输出总量约 `388685` bytes；optimizer/backward/evaluator/
checkpoint/prior flags 全部为零或 false，physical forward 与历史 evidence 未修改。

## H-LOMA-CBOGT-S0-BRAC：Scaled-Chen Back-Relay Axial / Conjugate 归因 closeout（2026-09-01）

从 clean implementation commit `0307119ee7da2ad38905fa7f20066eefd927b5c2` 完成
冻结、无更新的 detector-side axial/conjugate boundary audit。执行顺序固定为：先重放
父级 DCM nominal baseline，再测 `-Δz`、`0`、`+Δz`，其中
`Δz=1.1681239556097333e-05 m`。权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_bogt_back_relay_axial_conjugate_v1_attempt_03/`；
validation/preflight/audit/report/compact/panel/closeout SHA-256 依次为
`568670d5ccd4b151335fc8cc97c9c6cd15d5dd4dfd29d40c41c9fa1ce0e9384d`、
`15116230c4fdb7e8a59593bc754b24748bfe698837a9ef1a7a7a742ebacaf984`、
`2800a7c0c61008df0450ae3ef6e4ecc9779d7796ab16b29a7b20611d241815aa`、
`6749b336127e13b70e89c26f39956e52737d40a58344e902d11a90ccdbce579b`、
`099c77193f56dfb7c5b048b19025a914469e2d9af534bc5acb88ba40a7c6c785`、
`8a76e6aaf2df491d83f9cd322a6dffb7597de3fb52bb0fd0fc739a1bbf750ed0`、
`0dd3c67fcedb67df3da7bb4916723c905aa31e3491305522d222f8cee7d7c124`。

preflight、baseline replay、finite、frozen state、phase/square-law negative controls、
CUDA/storage resource gates 全部通过；evaluator calls 为 `0`，无 optimizer、backward、
prior sampling 或 checkpoint 写入。权威状态为 `valid_failed`，分类为
`relay_axial_conjugate_not_supported`，并记录 `aperture_effectively_inactive=true`、
`numerical_boundary_not_evaluated=true`。

selection 上 negative/baseline/positive relay retention 为
`0.656313/0.657125/0.658047`，paired MSE 为 `0.0620113/0.0620485/0.0621269`；
最大 retention 改善只有 `0.0922` 个百分点，paired MSE 相对 baseline 恶化约 `0.1265%`，
所以选择门拒绝所有候选且没有 confirmation arm。该测量只能否决“当前 nominal relay
附近的微小轴向/共轭偏移已经恢复重建”的解释，不能否定父级
`relay_propagation_bottleneck_supported`；`mechanism_evidence_only` panel 不升级为
因果证据。

`distributed_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 继续有效。唯一
下一动作是 `manual_review_scaled_chen_local_relay_tuning_closeout_contract`；在项目负责人
审查前不训练、不启动 FMF、不采样 prior、不生成、不 promotion。attempt-01/02 仍保留，
但已明确标为非权威历史产物。

## H-LOMA-CBOGT-S1-BOSA：Scaled-Chen Backend Observable Subspace Attribution（2026-09-01）

在实现提交 `8ece7fb6f0eb8208e4b50ada87bbfaaf981d648b` 上完成冻结、无更新的 backend
observable-subspace 审计。固定 Identity step-5500 与 Scaled-Chen step-5000，复用 MRA/DCM
selection panel、native replay 和四格 representation/backend 组合 `I→I`、`I→S`、
`S→I`、`S→S`；不重新运行 OLS，不交换、重采样或归一化，不创建 optimizer、不调用
evaluator、不反传、不采样 prior、不写 checkpoint。

权威 attempt-05 位于
`outputs/20260901_v2/runs/lightgen_scaled_chen_backend_observable_subspace_v1_attempt_05/`。
validation、audit、compact、panel SHA-256 分别为
`105606ac5886c2374d09d49443faa878f130811181f68ccd758d239e64f4a7d7`、
`1131860efc1500c4807d773eed6e75e49737972eff05e46bbbefea9e9e993baf`、
`67fbf72f2d7775d87f9c415babadc60f3794e4e3eb251a8ccdda0df1ff2ec6be`、
`a11a3c356ce7146bd0f491bc29c87f4219f65131e62af08595d27dae7eb08fc7`。attempt-01--04
仅作为保留的非权威历史产物。

结果为 `valid_failed` / `backend_observable_subspace_mismatch_not_supported`。selection
科学门在 representation contrast 上失败：`R_prop` 为 `I→I=0.9912`、`I→S=0.3256`、
`S→I=1.0086`、`S→S=0.6571`；Identity 相对 Scaled-Chen 的传播差为
`-0.01736/-0.33149`，方向与要求的稳定 Identity 优势相反。detector-side 差为
`+0.11389/+0.11386`，但不能替代四格 representation gate；功率 confound gap 仅
`5.5×10⁻⁵/0`。native replay、平方律/相位控制、finite、冻结状态、lineage 和资源门均
通过，因此这是有效科学失败而非 harness/resource failure。按合同 confirmation 未执行，
不能将该结果升级为 backend mismatch 的支持性因果结论。

该结果不改变 parent `distributed_bottleneck_supported` 或 DCM 的
`relay_propagation_bottleneck_supported`，也不改变
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`。结合 geometry `geometry_not_supported`、BRAC
`aperture_effectively_inactive` 和 `relay_axial_conjugate_not_supported`，当前本地
Scaled-Chen v1 relay 微调族关闭为
`scaled_chen_v1_local_relay_tuning_family=closed_no_local_correction_supported`；
高保真数值 relay boundary 仍未评估。唯一下一动作为
`manual_review_high_fidelity_relay_numerical_reference_contract`，不授权训练、FMF、prior
sampling、generation 或 promotion。

## H-LOMA-CBOGT-S0-MRA：Scaled-Chen Manifold Readiness Audit closeout（2026-08-31）

从实现提交 `a041187a6ff6292f2be6602871cfd6d35ca172c` 的干净证据运行
`outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/`。
冻结 Identity step-5500 与 Scaled-Chen step-0/1000/5000；selection/confirmation 为
1000/5000 张且相互隔离。validation、audit、compact、panel SHA-256 分别为
`b1713ee2676c4f5ab6e6f45838ed75bb1d2b1843e2ce738b53abe62d8e4b0c35`、
`6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556`、
`401fe2159f4342a4740b2c40955cb65fc7bb3600f2416fdc4171e1553eebeee0`、
`842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724`。

结果：`valid_failed`（harness 完整），瓶颈分类
`distributed_bottleneck_supported`，最大唯一 discontinuity 为
`decoder_layer_2→CMOS`（selection/confirmation retention `0.5827/0.5871`）。
encoder 的 SLM→pre-OLS retention 为 `0.9182/0.9216`，不是主要瓶颈；pre-OLS→modal
为 `0.8145/0.8092`，OLS 与下游均有独立损失。四个 checkpoint 的 fresh decoder
Jacobian rank 均为 `100`，不支持“decoder 单独 capacity-limited”。

`reconstruction_manifold_supported=false`，`generative_manifold_supported=false`。
parent prior 已注册但 `generation_prior_not_accessible`：KL improvement、rank retention
和 reconstruction gate 均未通过，故 `generation_readiness=not_ready`。审计保持
`optimizer_created=false`、`evaluator_calls=0`、`training_authorized=false`、
`formal_promotion=false`。唯一允许的人工后续为
`manual_review_scaled_chen_distributed_bottleneck_closeout_contract`；不启动训练、FMF、
prior-only generation、canary 或 promotion。旧 SMF background 缺失文件以零哈希明确记录，
未以其他路线 prior 或不匹配历史 Jacobian 替代。

## H-LOMA-CBOGT-S0-DCM-OBS：Scaled-Chen Decoder→CMOS 可观测性归因 closeout（2026-08-31）

在实现提交 `d5fbe5e6efd55076aca7020e0e1cb0f1c7763b19` 的 clean HEAD 上，完成了冻结、
无更新的 readout attribution。该审计只消费 MRA 已注册的 Identity step-5500、Scaled-Chen
step-0/1000/5000、1000 张 selection 和 5000 张 confirmation，将旧的
`decoder_layer_2→CMOS` 拆成 decoder L2 复场、formal back-relay 后 CMOS incident 复场、
以及 detector `|E|²` 强度三个真实平面。

权威运行目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_decoder_cmos_attribution_v1_attempt_03/`。
validation、audit、compact、panel SHA-256 分别为
`89ece3debd73fa3b7738b7604cafb1d1e52085562d0c70c15563a8d8c39691a7`、
`16975f7f26db13100af134fad70e510a1978093b88c619bc6c3a3e08c067ed68`、
`349ef59d07ca42c6e5446d7bcbe578110b03af27cbc838d0ebe818be38aaa5da`、
`6120c7cc1dc4832587bfe6d91b118579fbf097849d6c9d9a3668b1eb3952ce28`。attempt-01/02
按 write-once 规则保留但不参与结论。

结果为 `valid_failed`（harness 完整），分类为
`relay_propagation_bottleneck_supported`。decoder L2→incident retention 为
`0.6571/0.6031`（selection/confirmation），incident→intensity 为 `0.8868/0.9735`，
两段平均损失差 `0.3000`，最大唯一 discontinuity 是 back-relay。平方律、全局/逐像素
相位不变性、finite、冻结、状态 hash、negative control、CUDA/存储资源门均通过；峰值
CUDA `2589355520` bytes，`optimizer_created=false`、`evaluator_calls=0`。

这只细化 MRA 已有的 `distributed_bottleneck_supported`，不把 decoder Jacobian rank=100
改写为 decoder capacity 结论。`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 均保持不变。
唯一下一动作是 `manual_review_scaled_chen_bounded_back_relay_geometry_diagnostic_contract`；
该动作也不授权训练、FMF、prior sampling、generation canary 或 promotion。

## H-LOMA-CBOGT-S0-GEOM-RM：Scaled-Chen 秩匹配 OLS 几何归因 closeout（2026-08-30）

为解决 horizon 无更新审计中“OLS 几何失配与容量不足未分离”的遗留问题，项目负责人
授权一次冻结权重、无更新的九组秩匹配 LP01 OLS 审计。网格为
`fill={0.70,0.80,0.90}` × `facet_magnification={0.80,1.00,1.20}`，基线为
`(0.80,1.00)`；100 端口、身份 transport、7.111 µm waist、V=2.2、原 calibration、
2+2 phase-only formal Identity 光学全部固定。选择集为 MNIST test 后半段的 1000 张
均衡样本，确认集为原 horizon 5000 张 held-out；没有 optimizer、backward、evaluator、
checkpoint 写入或 exposure retuning。

指定归档 `D:\LDM-Archive\codex-original\outputs\archive_0157.tar` 仅恢复了 step 0/5000
checkpoint、horizon validation/preflight/training/calibration 及四类 no-update 归因文件。
历史 manifest 与当前 Windows 数据的图像、标签、标定索引、日程和 held-out 摘要全部一致；
仅 `dataset_root` 路径字段变化，已在消费者中显式分离绑定。

权威运行位于
`outputs/20260830_v2/runs/lightgen_optical_manifold_smf_bogt_geometry_attribution_v1_attempt_01/`。
最佳非基线 `(fill=0.80, facet_magnification=0.80)` 的选择集 paired MSE 改善 `4.02%`，
确认集 step 0/5000 改善 `4.48%/3.68%`；bootstrap 95% 差值区间为
`[0.00210138,0.00215426]`。虽然差值方向稳定且 paired 优于 cyclic，但未达到预注册
`5%` 实质改善门槛，且 step 5000 收益没有超过 step 0。所有 finite、100 秩、状态冻结、
负对照、lineage、CUDA 资源门均通过；峰值 `2279883264` bytes，墙钟 `345.25s` 仅作遥测，
evaluator calls 为 `0`。

validation、audit、decomposition、panel SHA-256 分别为
`ff904bfab0ca63208e881d51c3e06650295ca1e09a3f3b0a9a1a79c374bae0bd`、
`923d2e233cba0f795ec80fbd536846c356cdc177e2dfe0d77957bf4835565f8a`、
`952d062c8a72b872e299da3a20e870b8c7a414b1fd27dc9d9d57a376ca74e5e4`、
`94f1ac7fc74a3bf7b78c60ccc56a7fa479ab56fa30e06dcbc27ddecdc702fd4b`。

权威分类为 `geometry_not_supported`，evidence status 为 `valid_failed`。唯一下一动作是
`manual_review_scaled_chen_distributed_bottleneck_closeout_contract`；不得据此自动进入
FMF、继续训练、生成、exposure retuning 或 formal promotion，所有证据继续保持
`training_authorized=false`、`formal_promotion=false`。

### SMF bottleneck attribution closeout（attempt-02，2026-08-29）

attempt-01 的 CUDA 运行在审计 evidence 生成前因显存驻留超限而停止，登记为
`invalid_harness`，不形成科学结论。修复后在 clean commit
`3e608ee6ad4a7a4c61dbd06876c4fbe50d98dc7c` 完成唯一权威 attempt-02。validation、
audit、decomposition、panel SHA256 分别为：

`4185875318a0a301ab966f99c8ab4eedde517b14c5354864e0520f12fedacb81`、
`6f9b87adff81069b0b08d2d86e623d5c49565f30b2ca903640aa59b2c9e7b766`、
`5ad548c4c258f70ada5979c8dfb5e67021a90a3c6e71e923ef3a963a8abac1dd`、
`047001891a3e2de4d07d33d85ed9782371b2790927095c5b159904458f7357d5`。

全部 source replay、capacity、cross-swap、梯度边界、冻结状态、evaluator 隔离
和资源 gate 通过；evaluator calls `0`，peak CUDA `2.305 GiB`，wall `201.16 s`。
LP01 M1 平均 power retention 为 `65.3577%`，LP01+LP11a M2 为 `71.8676%`，
同源提升 `6.5099` 个百分点。但 G1--G3 仅 `1x` coupling dose 同时满足安全
兼容门，没有更高剂量候选。权威状态为 `valid_failed_mode_capacity`，唯一下一
动作是 `manual_review_fmf_mode_capacity_reconstruction_contract`；不启动 FMF
训练、continuation、prior sampling、生成或 promotion，仍保持
`training_authorized=false`、`formal_promotion=false`。

## H-LOMA-I0 SMF VAE KL chain attribution audit closeout（2026-08-28）

The authoritative no-update audit completed on clean implementation commit
`f47e8c6` in attempt-06. Earlier attempts are retained: attempt-01 is the
pre-context-fix validation, attempts 02--03 stopped on harness errors before
evidence, and attempts 04--05 are superseded by classification/schema fixes.

Authoritative hashes are validation
`e7de7e683b2650cd9ad6e622ac0bc76899f880eface6732fb86ac2cba4f31a74`, audit
`40d612e20ffc7044c0a8e21bce86c7d07f3076a31ea0d6aadee5fa6f53ea22ce`,
horizontal comparison
`678e9b37d14ef84cacc94819f8cc75f18d83d5bee6ac140481e7b1679667de2b`, and
panel `1cc913068aa389e87122abd59bcd9b4e9115e09d86cf5b850d3cd769398c6361`.

The KL chain is valid and active: source MSE reproduced exactly, KL formula
and complex round-trip passed, KL decoder gradient was zero, encoder/posterior
gradients were non-zero, and all model/optical state remained unchanged. The
beta arm KL was `3.0639548`, below no-KL `3.3382955`, but above initialization
`2.9956918`; posterior mean variance was `1.4764290` versus `0.9877587`, with
all 200 latent dimensions active. This rules out posterior collapse under the
registered screen, but does not satisfy absolute prior matching.

The three-arm comparison is valid only within the matched SMF 100-step budget:
beta-VAE mean-path MSE `0.0380180`, no-KL `0.0379458`, deterministic MSE
`0.0503602`. Identity/PLAN 6 SMF/FMF M2 are retained as explicitly non-matched
context. All 1x/2x/4x combined directions conflicted with reconstruction on
all three gradient panels, so the outcome is
`diagnostic_completed_parameterization_limited`. The sole next action is
`manual_review_smf_vae_posterior_parameterization_contract`; no dose-response,
new training, prior-only generation, FMF extension or promotion is authorized.

### H-LOMA-I0 PLAN 5 regularization pilot closeout（2026-08-28）

The bounded two-arm pilot was executed on
`research/v2-lightgen-optical-manifold-regularization-pilot-v1` after a
passed attempt-03 validation and CUDA preflight. Attempt-02 is retained as
`invalid_harness` because its preflight exceeded the fixed `300s` limit; the
preflight was then optimized by removing duplicate MSE-control replays without
changing the formal optics, loss, or gate definitions.

Attempt-03 validation SHA256 is
`e5e15d4d7f7e170b3a3d8b48ad74b00c29e786ccc25bb684542bebe683a5acb3`; preflight
SHA256 is
`769b88d114157563bfba1ea6864b627c6227f329af080bce50594743cae8a12e`; training
evidence SHA256 is
`319526199d0b3ffe26e6ac26bd1e51d984a9a066b4dac7891e50179e91afcab9`.
Preflight passed in `264.3439s` and the pilot completed both 100-step arms in
`2078.0805s`; peak CUDA was `492630016` bytes, storage was `12952577` bytes,
and evaluator calls were `0`.

The scientific status is `valid_failed`, solely because regularized SWD fell
from `0.25427815318107605` to `0.2532535195350647`, a `0.40296%` reduction
below the fixed `10%` gate. It was still below the MSE-only final SWD
`0.25692102313041687`. Regularized paired MSE was `0.03347916901111603`, below
the source `0.03437697887420654` and cyclic `0.0630621686577797`; normalized
true-label inter/intra was `1.2041319608688354`, above permuted
`1.0003312826156616`. The four phase layers updated, frozen state and source
MSE replay passed, and no formal candidate was produced.

The final regularized and MSE-only checkpoint hashes are respectively
`cbbf4cc579187e6c7d2d81f782741c73fd7f0050cdeab1025a1590fd2fa0c2f3` and
`2ee89473c1cee8aadc9153fc318ed7bf4c34f2924cd9e1c5cb19d7acbdc94856`.
The sole next action is
`manual_review_identity_regularization_objective_contract`; do not start
continuation, extra epochs, KL, SMF/FMF, generation, or promotion.

### H-LOMA-I0 PLAN 5b regularization objective attribution audit closeout（2026-08-28）

The objective attribution audit completed on clean commit `aece5c2` on branch
`research/v2-lightgen-optical-manifold-regularization-objective-audit-v1`.
Validation evidence is
`outputs/20260828_v2/runs/lightgen_optical_manifold_identity_regularization_objective_audit_v1_attempt_01/objective_audit_validation.json`
with SHA256
`138b5e5e7bfb16849c88012faf9673c4f31a338e864fa47cf4fe2254fdf67a4f`.
The objective audit evidence is
`outputs/20260828_v2/runs/lightgen_optical_manifold_identity_regularization_objective_audit_v1_attempt_01/objective_audit.json`
with SHA256
`e2d9c079f09f37203f2556e43836d237326740a673d7ae6ec633c9e1570908b5`.
The projection artifact is
`objective_projection_drift.json` with SHA256
`13d65c4a12c6b5cb352d861e7874b4d6b5c94a90279cfd7581ebef6aff81afff`.

The run is `valid_failed`, not `invalid_harness` or `resource_aborted`.
Authoritative source replay reproduced paired MSE `0.03437697887420654`;
formal trace provenance, finite values, model-state isolation and evaluator
calls `0` passed. Peak CUDA was `651952128` bytes, wall time was `202.4904s`,
and storage was `7193` bytes. The fresh-panel fixed-projection overlap was
`0.92070` for source, `0.91819` for regularized final and `0.91816` for the
MSE-only final, below the registered `0.95` stability gate. No higher dose was
selected and no model or checkpoint was changed.

This closes PLAN 5b as a valid projection-stability failure. The sole next
action is `manual_review_adaptive_projection_contract`; do not continue the
pilot, introduce VAE KL, enter SMF/FMF, generate samples or promote a model.
Evidence remains `training_authorized=false` and `formal_promotion=false`.

### H-LOMA-I0 PLAN 5b regularization objective attribution audit authorization（2026-08-28）

The PLAN 5 bounded pilot is closed as a valid scientific failure, not as a
harness failure. The regularized arm reduced fixed held-out SWD by only
`0.40296%`, although it remained below the MSE-only control and preserved
reconstruction and instance-structure gates. The project owner authorized one
no-update objective attribution audit on branch
`research/v2-lightgen-optical-manifold-regularization-objective-audit-v1`.

The audit consumes the authoritative pilot validation, preflight, training
evidence, source step-5500 checkpoint, both pilot final checkpoints, PLAN 5
projection, and MNIST manifest. It uses new train-only panels disjoint from
all prior calibration, gradient, pilot, and held-out samples. It measures
fixed-projection drift and tests only the registered `1x`, `2x`, and `4x`
regularizer gradient doses without creating an optimizer or updating a model.

The real consumer is manual authorization of a bounded dose-response pilot.
Projection instability, lack of a safe higher dose, or failure against the
rank-permuted control each route to its registered manual review decision;
lineage, isolation, numerical, or resource failure is `invalid_harness`.
This audit does not authorize continuation, VAE KL, latent sampling, SMF/FMF,
generation, or promotion. Evidence remains
`training_authorized=false` and `formal_promotion=false`.

### H-LOMA-I0 PLAN 5c adaptive projection audit authorization（2026-08-28）

The project owner authorized one no-update, diagnostic-only cross-fitted
projection audit on branch
`research/v2-lightgen-optical-manifold-adaptive-projection-audit-v1`. It
consumes the authoritative PLAN 5b validation, audit and projection evidence,
the step-5500 source checkpoint, both 100-step pilot checkpoints, and the
MNIST manifest. The real consumer is a manual decision on a bounded
regularization pilot using a stable deterministic projection.

The audit fits candidate ranks 16/24/32 on three fresh train-only calibration
panels, constructs a fixed consensus projector, and evaluates an independent
assessment panel plus three gradient-control panels. The assessment panel is
not used for fitting or selection. The phase model, formal 2+2 Identity path,
fixed exposure and all source artifacts remain unchanged. No optimizer,
parameter update, checkpoint, KL, latent sampling, SMF/FMF, generation or
promotion is permitted.

### H-LOMA-I0 PLAN 5c adaptive projection audit closeout（2026-08-28）

The implementation was completed on branch
`research/v2-lightgen-optical-manifold-adaptive-projection-audit-v1`. The first
validation-bound CUDA launch was retained as an invalid execution attempt
because the lifecycle referenced the field-capture helper from the wrong
module; no evidence was written and the defect was repaired on clean commit
`c7878c6`.

The authoritative attempt-02 validation evidence is
`outputs/20260828_v2/runs/lightgen_optical_manifold_identity_adaptive_projection_audit_v1_attempt_02/adaptive_projection_validation.json`
with SHA256
`7e11e3d989e2dd558d11f4d13d96b2cbd33c628e337806ef27d888ff071d99b5`.
The authoritative adaptive projection audit is
`outputs/20260828_v2/runs/lightgen_optical_manifold_identity_adaptive_projection_audit_v1_attempt_02/adaptive_projection_audit.json`
with SHA256
`98f40cc3312c01f22056424e6a4baa25d6bfd618bb0eb4f3dcbab75934fbb7a3`.

The audit was a valid scientific `valid_failed` result. Candidate ranks 16,
24 and 32 all failed the registered stability contract. Their minimum
cross-fit overlaps were `0.90390`, `0.88964` and `0.92639`; their assessment
captured-energy retention values were `0.67514`, `0.74007` and `0.84228`.
Rank 32 also had SWD coefficient of variation `0.07101`, above the `0.05`
gate. No rank was selected and no consensus projection artifact was written.

Source reconstruction MSE was reproduced exactly at
`0.03437697887420654`. Formal Identity provenance, finite values, model-state
isolation and evaluator calls `0` passed. Peak CUDA allocation was
`446995968` bytes, wall time `205.8172s`, and storage `7319` bytes. The result
does not show that the optical representation is absent; it shows that the
tested deterministic PCA coordinates are not stable enough to authorize a
regularized pilot.

PLAN 5c therefore closes with `valid_failed`. The sole next action is
`manual_review_probabilistic_latent_contract`. Do not retry by changing the
rank, lambda or panel within this attempt, and do not enter VAE KL, SMF/FMF,
generation or promotion. Evidence remains
`training_authorized=false` and `formal_promotion=false`.

### H-LOMA-I0 SMF integration canary closeout (2026-08-28)

The project owner authorized a diagnostic-only 100-port LP01 SMF integration
canary on branch
`research/v2-lightgen-optical-manifold-smf-integration-canary-v1`. The canary
used the existing 2+2 phase-only optical path and inserted research-only SMF
transport between encoder and decoder. It ran exactly 20 steps for each of
`smf_rank32_swd` and `smf_mse_only`; the rank-32 projection was retained only
as `diagnostic_unstable_projection` and was not treated as a stable training
projection.

Attempts 01--06 remain preserved as harness/consumer failures. Attempt-07 is
the authoritative complete run at clean commit `da5c13c`:

- validation SHA256: `bc87f1c996331dbe88f059b71faac2da0ce891c32969dd195c881a9eeff93bc2`;
- preflight SHA256: `70fc113ca98a4776d88837e3aab4293d0676d965ab98dbcc19f631820d1fe70f`;
- training SHA256: `271f934515bde2fddaab413aba05e82472751b1664053c3a27d28de3c76a1860`;
- panel SHA256: `2b08105870853a81bc8cc59c866de411b8c78cd06cc0da1bac408fc89500058f`.

Evidence status is `diagnostic`, outcome `diagnostic_completed`. The fixed
panel paired MSE changed from SMF initialization `0.0639665723` to
`0.0631253868` for MSE-only and `0.0631285161` for rank-32-SWD; cyclic controls
were `0.0748243704` and `0.0748260617`. Final held-out probe SWD was
`0.0294856094` for MSE-only and `0.0288933087` for rank-32-SWD. The two final
arm image tensors differed by only `1.86e-8` MSE, so this short run does not
support a meaningful SWD-arm effect. Coupling efficiency was approximately
`0.5339`, active-port fraction `0.9317`, and the final output retained much
less variance than the Identity reference.

Validation, preflight, exact 2+1 resume, four-layer updates, SMF/Identity/
zero-port controls, state isolation, formal IdentityOLS preservation and
evaluator isolation all passed. Peak CUDA allocation was `2375665664` bytes,
wall time `97.0598s`, storage `86251566` bytes, and evaluator calls `0`.
This establishes that the SMF path executes and updates correctly; it does
not establish SMF superiority, stable projection, FMF readiness or formal
promotion. The only next action is
`manual_review_smf_canary_visual_and_metric_trends`; a matched PLAN 6
reconstruction pilot requires a separate project-owner authorization.

## H-LOMA-I0 PLAN 6 Identity--SMF reconstruction pilot closeout（2026-08-28）

The authoritative run is attempt-04 on clean commit `bced59f`. Attempts 01–03
are retained invalid harness attempts: wrong resume-result nesting, an
impossible self-referential preflight hash consumer, and a dropped batch
dimension in interpolation. None reached training updates or produced a
scientific result.

Evidence hashes: validation
`d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5`,
preflight
`722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623`,
training
`a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e`, and
panel
`9de2fc9d9c2d43ed9752d8609afcaa5455cfb05b9c411d7d0cca9b2a8cf50fdf`.
The SMF and Identity final checkpoint hashes are
`949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94` and
`9258eada74bf04e114d5f74fc3722450f56fad684fd978c065fd3f94e0b1683b`.

Both MSE-only arms completed 1000 updates on the same fresh schedule. SMF
held-out paired MSE improved `0.0659028739 → 0.0508093871` (`22.90%`), with
final cyclic MSE `0.0671605766`; centered correlation and variance retention
improved `0.6562196 → 0.7100222` and `0.0627982 → 0.1317942`. Identity control
improved `0.0344723202 → 0.0303227697`, stayed below cyclic MSE `0.0630837232`,
and also passed its registered control gate. Four phase layers updated,
formal IdentityOLS and source state were unchanged, evaluator calls were `0`,
peak CUDA was `2677763072` bytes, wall time `2212.9581s`, and storage
`817444396` bytes.

This is a bounded diagnostic `passed`: it establishes trainable SMF MSE
reconstruction, not SMF superiority or FMF readiness. Active-port fraction was
unavailable from the reused low-level diagnostic and remains explicitly
non-gating. The unique next action is
`manual_review_fmf_m2_integration_canary_contract`; do not start FMF, extra
training, generation or promotion. `training_authorized=false` and
`formal_promotion=false` remain mandatory.

## H-LOMA-I0 PLAN 6 Identity--SMF reconstruction pilot authorization（2026-08-28）

The project owner authorized one diagnostic-only 1000-step matched comparison
on `research/v2-lightgen-optical-manifold-smf-reconstruction-pilot-v1`.
It binds the completed SMF canary, the step-5500 Identity checkpoint and the
authoritative MNIST manifest. The two arms share a fresh 1000-by-10 balanced
schedule and reset CPU-shadow Adam; only the research OLS differs:
`smf_mse_reconstruction` inserts the existing 100-port LP01 transport and
`identity_mse_control` keeps formal IdentityOLS.

The lifecycle requires static lineage validation, a real CUDA three-step and
persisted 2+1 resume preflight before one 1000-step run. It is MSE-only and
diagnostic: no SWD/projection, KL, latent sampling, extra epoch, FMF,
generation or promotion. Its only post-run consumers are the registered SMF
reconstruction or bottleneck-attribution manual reviews. Formal v2 and all
historical evidence remain unchanged; `training_authorized=false` and
`formal_promotion=false` remain mandatory.

## H-LOMA-I0 PLAN 7a FMF M=2 integration canary closeout（2026-08-28）

PLAN 7a is complete as an engineering canary on branch
`research/v2-lightgen-optical-manifold-fmf-m2-integration-canary-v1`, clean
commit `f53bd4e`. Attempts 01--07 are retained non-authoritative validation or
preflight attempts; attempt-08 completed the run but failed storage because
the panel retained its 1000-image backing storage. Attempt-09 is authoritative
with outcome `diagnostic_completed`.

Evidence hashes are validation
`f4af9a87a2f953a4873a11a1f09814e1ab9826dc336638940bb05148a4abacce`, preflight
`14abce541b0961ca848678f44d600e48c70e13e22b6cbdbf2ef8c32ba38100ca`, training
`f699fccce7da7fb5f1560c5823ee76e2c713ba9e726fb9f6213fe967da36afb6`, and
panel `709b1ff2f716a4a77f733ef3e77a376d9f4c234d7fe4d4d3ddd1156f28a5c195`.

The nested bank preserved LP01 exactly and added finite orthonormal LP11a,
producing `[10,200]` coefficients. M2 paired MSE changed
`0.0498638563 → 0.0496286638` (cyclic `0.0667282492`), centered correlation
`0.6993410 → 0.7016757`, variance retention `0.1480903 → 0.1489086`, and
coupling efficiency `0.7191151`. M1 changed paired MSE
`0.0508093871 → 0.0507159270` (cyclic `0.0671152920`). These are
non-gating diagnostic trends.

All preflight/training gates passed: finite values, four-layer updates,
formal IdentityOLS preservation, exact persisted 2+1 resume, passivity,
zero-transport/no-bypass controls, source isolation, evaluator calls `0`,
CUDA peak `2661837824` bytes, wall `123.8385s`, and storage `229115402`
bytes. This proves the M=2 integration chain is executable and updates
correctly; it does not prove FMF superiority or authorize longer FMF, M=3,
mode coupling, generation or promotion. The unique next action is
`manual_review_fmf_m2_reconstruction_pilot_contract`.

## H-LOMA-I0 PLAN 5 probabilistic latent recovery pilot authorization（2026-08-28）

The next research branch is
`research/v2-lightgen-optical-manifold-smf-vae-pilot-v1`, based on clean
commit `a3af141`. It restores the missing standard probabilistic interface of
PLAN 5 without changing the completed Identity, SMF or FMF evidence.

The posterior is inserted at the 100-port SMF coefficient boundary. A fixed
training-only calibration maps real/imaginary coefficients to a 200-dimensional
real representation. A shared per-port `Linear(2,16) -> SiLU` head produces
identity-residual `mu` and `logvar`; `z=mu+exp(0.5*logvar)*epsilon` is mapped
back to complex coefficients and re-emitted by the unchanged SMF bank.
This is a research-only electronic hybrid and active latent injector, not a
formal all-optical inference claim.

The bounded pilot contains three matched arms: deterministic SMF MSE,
stochastic posterior reconstruction without KL, and stochastic reconstruction
with fixed beta-KL. It requires static validation, CUDA three-step and persisted
2+1 resume preflight before 100 steps. Prior-only generation, KL extension to
FMF, extra epoch, evaluator use, formal promotion and changes to
`outputs/final_pt/` remain prohibited.

## H-LOMA-I0 PLAN 5 probabilistic latent recovery pilot closeout（2026-08-28）

The project-owner-authorized bounded SMF hybrid-VAE pilot ran on branch
`research/v2-lightgen-optical-manifold-smf-vae-pilot-v1`. The posterior is a
research-only electronic `project_approximation` at the 100-port SMF
coefficient boundary: shared per-port `Linear(2,16) -> SiLU` heads produce
identity-residual `mu` and bounded `logvar`, followed by fixed
reparameterization and SMF re-emission. Formal IdentityOLS, the existing
SMF/FMF operators, `outputs/final_pt/`, and prior evidence were not changed.

Attempts 01--04 were retained validation/preflight harness attempts. Attempt-05
completed a run but was superseded because its implementation did not enforce
the pre-registered requirement that final VAE KL be below initialization. The
authoritative attempt-06 was run on clean commit `7a415ea` after that gate and
the panel contract were repaired. Its validation, preflight, training, panel,
posterior-normalization and final-checkpoint hashes are:

`d50919acb75ffe68453db52a6d87ebdfeb8ae9963f157b0bd3afb582f17de04c`,
`5e1f8569e7e58ce086db37a318df76fa81107820323128dafbedd02ad0e8a936`,
`ec6afc5b15274ce9a7e2446bef43bb6346007480da23408fe1399b412200c476`,
`e8134f326dfdbdc33e900be17292f17c01d675427f1409193a27418ce64cb664`,
`e764ceb31e9ce1aa24f2cd9ad909011550aa6ad095404916ff84be8f93d07a85`,
`d4bbad17de2c9623a37825b718aee7dc1e14f99727c1651fe9bfb8bd53f04262`,
`7885762ffd63b42f2bff3bf82b002c0a4e7cccb4e46c0b4993d4a242b183ec6d`, and
`8b508be438a73f160a956dc063e359354ccde4ac894de960297ae79eafde2b59`.

All three arms completed 100 steps with finite values, four-layer phase and
posterior updates, exact persisted 2+1 resume, direct-SMF posterior-mean
replay MSE `2.72e-15`, evaluator calls `0`, peak CUDA `2362135552` bytes,
wall time `416.0924s`, and storage `116038584` bytes. The fixed held-out
metrics were:

- deterministic MSE: `0.0508093871 → 0.0503602326`, cyclic `0.0669119731`;
- stochastic no-KL: final paired MSE `0.0379457697`, cyclic `0.0672367439`,
  final KL `3.3382956982`;
- beta-VAE: final paired MSE `0.0380179696`, cyclic `0.0669108853`, final KL
  `3.0639550686` versus initialization `2.9956917763`, centered correlation
  `0.7465408`, posterior-mean variance `1.4764290`.

The authoritative outcome is `valid_failed`: the VAE KL decreased relative to
the stochastic no-KL arm but increased relative to its own initialization.
This pilot demonstrates that the hybrid posterior, sampling path and KL
gradient flow are executable without reconstruction collapse, but it does not
demonstrate the registered KL-success objective. The sole next action is
`manual_review_identity_posterior_collapse_contract`; no KL retuning, extra
epoch, FMF extension, prior-only sampling, generation or promotion is
authorized. Evidence remains `training_authorized=false` and
`formal_promotion=false`.

## H-LOMA-I0 SMF VAE KL chain attribution audit（2026-08-28）

Implementation is registered on branch
`research/v2-lightgen-optical-manifold-smf-kl-attribution-audit-v1`, based on
clean commit `ecfa4a5`. The audit is no-update and diagnostic-only. It consumes
attempt-06 source evidence and final checkpoints, verifies the original CUDA
MSE replay and standard diagonal-Gaussian KL chain, then compares the three
matched SMF arms. Identity, PLAN 6 SMF and FMF M2 numbers are displayed only
as non-matched context.

The audit must preserve the existing `valid_failed` absolute-KL conclusion;
it may only determine whether KL is relatively constraining, whether a
posterior collapse signature exists, and whether a higher dose has compatible
local gradients. Its only downstream consumer is manual review of a future
KL dose or posterior parameterization pilot. No training, checkpoint creation,
prior-only generation, FMF extension or promotion is permitted. Evidence
continues to require `training_authorized=false` and `formal_promotion=false`.

## H-LOMA-I0：KL 因果确认与固定 OLS 流形损耗恢复（2026-08-28）

本阶段从 clean commit `a62b264` 创建分支
`research/v2-lightgen-optical-manifold-kl-ols-loss-recovery-v1`，固定 SMF
mode bank、identity transport 和 formal 光学几何。第一步修复 KL attribution
剂量审计：attempt-06 的 `1x/2x/4x` 漏乘 registered beta，故其剂量归因不得
支持 `parameterization_limited` 结论；attempt-06 的 KL 公式、source replay、
cross-swap 和梯度边界结果仍保留为输入。

本阶段后续消费者依次为 corrected KL dose reaudit、同源 100-step KL effect
pilot、固定 OLS projection/coupling-loss canary 和 1000-step continuation。
projection loss 只作用于 encoder，KL 只直接作用于 encoder/posterior，CMOS
MSE 负责全链路 reconstruction。可训练 OLS transport、先验生成、FMF 长训练、
formal promotion 和 `outputs/final_pt/` 修改均不在授权范围内。

## H-LOMA-I0 固定 OLS 流形损耗恢复 canary closeout（2026-08-28）

修正后的 KL 剂量审计选出 `8x`，同源 KL effect pilot 通过短预算门后，执行了
固定 SMF OLS 的四臂 100-step canary。固定边界为 100-port LP01 SMF mode bank、
identity transport、formal IdentityOLS、relay 和 exposure；新增
`L_projection = 1 - P_smf_out/P_smf_in`，只对 encoder 产生梯度，KL 臂使用已注册
的 candidate beta `0.014605162095621654`。

attempt-01 保留为 `invalid_harness`：终端 posterior KL 没有写入 evaluator
metrics，导致 KL gate 无法判定；不把该 attempt 的完整 checkpoint/panel 当作科学
证据。修复后的 attempt-02 为权威结果，validation、preflight、training、panel
SHA256 分别为：

`18d63ff5e9c3d18db7c669fc8c596423d8a2cc02ed50338cf8043fcc609d06f6`、
`abf063fa6e39511b01a661923f755e84c1eefd8285c91bdce0651a80066d8833`、
`a73b3917d64049509b2c6f025e0025bd506db4d6f95271598c5dc95c43c0b65e`、
`c432221ba0725da3744f66459a3c513ef632d5aa202864123007137aa5798b61`。

attempt-02 四臂均完成 100 步，finite、四层 phase 更新、固定 OLS 隔离、持久化
`2+1` resume、evaluator calls `0` 和资源门全部通过。coupling 臂 projection
residual 从 `0.34640783` 降至 `0.34150791`，paired MSE 为 `0.05036358`；但
power retention 仅从 `65.3592%` 到 `65.8492%`，相对 optical MSE control
`65.6478%` 只高 `0.2014` 个百分点，未达到预注册的 `+5` 个百分点。KL 臂的
KL 为 `2.81820757`，低于初始化 `2.99914436` 和 no-KL `3.37511785`，KL gates
在修复后通过。

权威状态为 `valid_failed`：固定 OLS projection loss 产生了小幅功率变化，但尚未
达到损耗恢复 canary 门；这不是 harness 或资源失败。唯一下一动作是
`manual_review_smf_bottleneck_attribution_contract`。不启动 1000-step continuation、
不重新调 lambda、不训练 OLS transport、不进入 FMF/生成或 promotion；继续保持
`training_authorized=false`、`formal_promotion=false`。

## H-LOMA-CBOGT-S1-HFRR：Scaled-Chen High-Fidelity Relay Numerical Reference（2026-09-01）

项目负责人批准了一个冻结、无更新的高保真 relay 数值边界审计。实现提交为
`8835bb228f074408a596118a907248e0f7c9c609`，唯一权威目录为
`outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/`。
审计固定 Identity step-5500 与 Scaled-Chen step-5000，保留生产
`CoherentImagingRelay`、`back_b0`/`chen_relay`、formal 坐标、ABCD、pupil、采样、
detector 和 exposure；独立 reference 为 CPU `complex128` 直接可分离
Fourier-series evaluator，不调用生产 transform helper。没有改 physical forward、
formal runtime、OLS、checkpoint 或训练目标，也没有 optimizer、backward、evaluator、
prior sampling 或 checkpoint 写入。

`validation.json`、`audit.json`、`compact_report.json`、`panel.pt` 的 SHA-256 分别为
`248983c83cdd122f4a6aa7cf7f38bf6b415d3f080eb8136bb0a4fdb0b8e1c84f`、
`4045e8e0469a745c637c8dae151d073e5b65e06395bf175297d93245f1031e48`、
`d85e82d9a98fb36568d16c63d0fc1a4e5e3a311a007c512f2b07be7472427e7b`、
`dc75d62ed53a2699aa0949fd8bd8a465f0cec2abe46b64279858a304bde9301b`。12 组固定合成
probe 与四种 task backend cell 的 complex/amplitude/phase/power/intensity 数值门
全部通过；Identity/Scaled-Chen 的最大 native complex relative L2 为
`8.17e-14/6.15e-14`，task 最大值均低于 `1.03e-5`（预注册上限 `2e-3`）。动态
ABCD/pupil 与 formal replay 一致，模型/OLS state hash 前后不变，峰值 CUDA 为
`1323107840` bytes，输出总量约 `179883` bytes。

权威状态是 `valid_failed` / `relay_complex_retention_metric_confound_supported`
（Class B），不是 numerical implementation mismatch：
`relay_numerical_reference_consistent=true`、`relay_numerical_reference_mismatch=false`、
`numerical_boundary_not_evaluated=false`。Identity raw backend gap `0.6517181` 经
仅供诊断的 analytic canonicalization 降至 `0.0081707`，Scaled-Chen
`0.3329323→0.0193076`；该变换不进入 physical forward、detector、MSE、训练或晋级，
cross-backend 结果仍只作 pupil-aware diagnostic。HFRR 不推翻 DCM 的
`relay_propagation_bottleneck_supported`，而是要求复核 retention metric 的坐标/相位
解释。`distributed_bottleneck_supported`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false` 与 `generation_readiness=not_ready` 保持不变。

唯一下一动作是
`manual_review_relay_retention_metric_reinterpretation_contract`。在该人工审查前，
不训练、不启动 FMF、不采样 prior、不生成、不 promotion；本记录不授权新的 relay
参数扫描或 decoder redesign。

## H-LOMA-I0：corrected KL effect pilot registration（2026-08-28）

The corrected no-update dose reaudit selected the sole common compatible
multiplier `8x`, giving effective beta `0.014605162095621654`; source replay,
gradient-boundary, freeze and resource gates passed. The earlier dose attempt
remains immutable and invalid-harness.

The project owner authorized one four-arm, 100-step diagnostic causal pilot on
this branch: deterministic MSE, stochastic no-KL, current-beta VAE and
corrected-beta VAE. It uses a fresh disjoint balanced schedule and shared
posterior epsilon. Fixed SMF modes, identity transport, formal IdentityOLS,
relay and exposure are preserved. No trainable OLS transport, prior-only
generation, FMF extension or promotion is authorized; evidence remains
`training_authorized=false` and `formal_promotion=false`.

## H-LOMA-I0：SMF bottleneck attribution audit（2026-08-29）

当前 fixed-OLS recovery attempt-02 为有效科学失败：coupling 臂相对 MSE
control 的 power gain 只有 `0.2014` 个百分点，不能进入 1000-step continuation。
KL 路径不是当前阻塞，KL 臂已低于初始化和 no-KL 对照。

下一步从 clean `37e2b1b` 的独立分支执行 no-update attribution，消费 attempt-02
evidence 与 checkpoints。实验拆分 SMF LP01 投影容量、既有 FMF M2 observation-only
容量、encoder/decoder `2×2` cross-swap，以及 `1×/2×/4×/8×` coupling 梯度剂量；
不训练、不改 OLS transport、不放宽历史 `+5pp` 门限。审计结果只决定下一个人工
合同，不自动授权 dose pilot、FMF、生成或 promotion。

## H-LOMA-CBOGT-S3-RMFA：Reconstruction-Manifold Failure Attribution（2026-09-01）

本次是 TRBA closeout 后的唯一 RMFA 实施。消费者是项目负责人判断当前 Scaled-Chen step-5000 physical reconstruction mapping 是否与 frozen representation 错配；拒绝决定是不得把 bounded probe 失败写成 representation insufficiency，也不得以 probe 结果直接训练或晋级新 decoder。实现绑定 clean commit `2445d946fd26867d1c9bd98b04f262b50cd0c90e`，没有改 formal v2 runtime、OLS、checkpoint、训练目标、detector、prior 或历史 evidence。

RMFA 只使用一次预注册 fit subset（train 前 55000 中每类 100、共 1000）构造固定 Moore–Penrose minimum-norm linear readout，然后在已有 selection（test 5000–9999，每类 100）与 confirmation（train 55000–59999，5000）上复用同一 probe；特征为固定 `16×16` real/imag adaptive-average-pool pre-OLS 加 bias，`input_dim=513`，target 为原 paired reconstruction target，`rcond=1e-12`，rank/effective rank=`513`，condition number=`7945.41016660053`。不存在 lambda sweep、optimizer、backward、evaluator、refit、prior sampling 或 physical-forward modification。

权威 attempt-01 目录为 `outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/`；validation/audit/compact/probe SHA-256 为 `08e6afd841574a5ac6ba126d11af24b560b6f85bf2d5365d42fd229b373625c0`、`e0a53625171144b7aba9dbdc3258ff94ca2686219a80d0725f535d2625c7d468`、`174bdf09c8b5a33896b3a22396f8dfaea6dd22200836f9bbbe46d53b57c403f5`、`25d9eba0d6203027b3a6be00a9d0a5538225510e45c0927f63cf8f78caaf4a09`。selection native paired/template/cyclic=`0.0620484613/0.0324799940/0.0799509138`，probe paired=`0.0033369721`，relative improvement=`94.62198%`；confirmation native=`0.0577895157/0.0277411994/0.0756082982`，probe paired=`0.0033930303`，relative improvement=`94.12865%`，fixed 2000-replicate bootstrap 95% CI=`[0.0540225357,0.0547624826]`。两个 panel 的 gate 全部通过，故 R1 classification 为 `reconstruction_mapping_misalignment_supported`。

该结论只支持“当前 physical mapping 丢失的信息可由预注册冻结线性 readout 恢复”，不支持 universal representation sufficiency、架构已修正或 generation readiness。上游 formal/physical `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready` 继续有效；RMFA 只在自身 bounded diagnostic 字段中记录 recoverability positive。Scaled-Chen v1 bottleneck-attribution family 现在收敛为 `closed_task_relevant_bottleneck_not_supported`。唯一下一动作是 `manual_review_reconstruction_mapping_alignment_contract`；不训练、不启动 FMF、不采样 prior、不生成、不 candidate export、不 formal promotion。

默认完整 pytest 的 `C:\TEMP` 访问拒绝仍是既有环境问题；在可写新临时根下为 `1131 passed, 12 deselected, 25 subtests passed`。治理状态保持 `training_authorized=false`、`formal_promotion=false`、`mechanism_evidence_only=true`。

## H-LOMA-CBOGT-S4-RMAL：Reconstruction Mapping Alignment Localization（2026-09-01）

本次是 RMFA 之后唯一执行的冻结、无更新 bounded-recoverability 审计。消费者是项目负责人判断当前 Scaled-Chen step-5000 physical reconstruction mapping 是否存在可由固定 readout 恢复的错配；拒绝决定是不能把该结果写成通用 representation sufficiency、架构修正或 generation 授权。实现绑定 clean commit `5b274b926474784a1ad49744e75190afff31747c`，没有改 formal v2 runtime、OLS、checkpoint、训练目标、detector、prior 或历史 evidence。

权威 CUDA attempt-01：
`outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/`。
validation/audit/compact/probes SHA-256 依次为
`affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec`、
`5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa`、
`c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137`、
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。
attempt 总量 `41230970` bytes；峰值 CUDA `2274099712` bytes；运行前可用显存 `5368709120` bytes；validation/audit/compact 均 `passed`，0 个 false gate。

RMFA positive-anchor selection/confirmation replay 均通过。RMAL 固定一次 1000-sample fit 的 float64
`torch.linalg.pinv(rcond=1e-12)` probe，按
`cmos_intensity → cmos_incident_complex_field → decoder_layer_2 → decoder_input → modal_complex_100 → pre_ols`
搜索。首个 `cmos_intensity` stage selection physical/probe/template/cyclic=`0.0620484613/0.0098129278/0.0324799940/0.0799509138`，relative improvement=`84.18505%`；confirmation=`0.0577895157/0.0101852901/0.0277411994/0.0756082982`，relative improvement=`82.37519%`，固定 bootstrap 95% CI=`[0.0472728200,0.0479231887]`。两 panel gate 全部通过，故 `furthest_downstream_recoverable_stage=cmos_intensity`，其余五段按 stop rule 跳过，classification 为 `bounded_recoverability_established_at_detector`。

该结果仅是 bounded diagnostic recoverability positive：`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready` 保持。治理 flags 为
`optimizer_created=false`、`backward_calls=0`、`evaluator_calls=0`、`checkpoint_written=false`、
`prior_sampled=false`、`generation_started=false`、`physical_forward_modified=false`、
`mechanism_evidence_only=true`。唯一下一动作是 `manual_review_RMAL_recoverability_result`；不训练、
不启动 FMF、不采样 prior、不生成、不 candidate export、不 formal promotion。

RMAL 定向测试 `14 passed`，直接相关回归 `56 passed`，compileall、Ruff、diff-check 通过。全套默认 pytest
为 `1144 passed, 1 failed, 12 deselected, 25 subtests passed`；唯一失败仍是既有 near-conjugate 临时
run 目录缺失的 Windows 路径问题，与 RMAL 文件无关。

## H-LOMA-CBOGT-S5-ZCSA（2026-09-01）

ZCSA 已在起始 HEAD `0dead1b73655b3556450cedbb5d0a2c579385209` 完成，实现提交为
`1f5e553f6c7ad80217427ab83f5669e7b9758acb`。改动仅包含 zero-column semantics audit producer、
研究 YAML、review contract、测试与 CLI schema/router；RMAL/RWSA 旧产物、冻结 probe、renderer、
checkpoint、OLS、detector 和 formal runtime 未改动。正式 attempt 路径为
`outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/`，
validation/audit/compact SHA256 为 `d422f1e31e8298f9662896f832b207ad896fb619c5a656bccf54b099cd419954` /
`0cba5990462e9f8cc238885b14c90c907a7c6f085877b6c8ad0c8b95cbf0bfb8` /
`772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6`，probe hash 为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

正式结果 `passed / zero_column_semantics_attribution_completed`。taxonomy 为
`dynamic_active=33502`、`target_zero_static=6498`，其余三类为 0；zero-W 与 selection/confirmation
ever-nonzero 交集为 `1495/2097`，b exact-zero=`6498`；空间 pixel/edge/interior=`6498/660/5838`，
4-connected components=`2`（最大/次大 `5105/1393`）。无 optical replay、CUDA forward、model load、
optimizer、backward 或 evaluator。定向测试 `18 passed`，直接相关回归 `57 passed`，compileall、Ruff、
diff-check 通过；全套默认 pytest（短 basetemp）`1197 passed, 12 deselected, 25 subtests passed`。

状态继续为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`，分类
`closed_task_relevant_bottleneck_not_supported`。唯一下一动作是 `manual_review_ZCSA_result`；不训练、
不 FMF、不采样 prior、不生成、不 candidate export、不 formal promotion。result/status commit 后
执行 bridge refresh/status 检查，不自动上传。

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

本轮为 frozen-W、CPU-only descriptive spectral/operator attribution；消费者是项目负责人审查冻结 RMAL detector readout 的谱能量、source/target singular-mode 结构与 bias 归因，拒绝决定是不得把描述性结果升级为 readiness、架构或训练/晋级授权。起始 HEAD=`bd967f64ffb046677fa4204e81e911b226047e6e`，实现 commit=`bed235283d1fa47f4a5a38d41368d87ce2b5d218`。新增 producer/config/review/tests/CLI schema-router，RMAL/ZCSA/RWSA-ACTIVE、probe、renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 未改。

权威 attempt：
`outputs/20260901_v2/runs/lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/`；validation/audit/compact SHA256=`3f24dcc7d56e098ec778998bb8ddc7871973131f1d7269c86da729395426c2f2` / `97842d5707541516c79adb87c3a1e609ee533156895908c9e99c015cde0f9b13` / `e281daf9a4a62f85828a3bc8055ce044027d0c0fb1af0a522febcf3c2b68185b`。probe SHA256=`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

结果为 `passed / linear_readout_spectral_attribution_completed`。W/b=`[256,40000]/[40000]`，full/active/inactive=`40000/33502/6498`；full/active Gram=`[256,256]`，CPU float64 eigensolver，未构造 40000×40000 Gram。前五 singular values=`12227.8671591/10625.3503547/8724.4504853/8126.3713707/7844.9489846`，top-16 cumulative=`0.8441783201`，effective/stable telemetry=`21.88110383398559/21.881103833985602` 与 `5.528997319328641/5.5289973193286395`。full-vs-active 最大 absolute/relative eigen difference=`4.470348358154297e-08/8.328885261127632e-13`，最大 sigma difference=`2.900790718740609e-12`，inactive identity 保持。

K=16 为 descriptive preview；source/target orthogonality residual=`1.5695833278775036e-15/3.095197802817193e-14`，每个 mode 保存固定 D4、FFT、空间、signed 与 basis-dependent descriptor。global reference 为 RMAL fit panel 1000 个 unlabeled streaming target mean，hash=`513a696e21dafebebe4a862ab814b06294beb662db2afcfd679899fd19b27daf`；class-template、selection、confirmation 与 target bank 均未使用/保存。bias norm²=`114.5276114768678`，top-16 projection energy=`21.21861242530154`，preview 外 residual=`93.30899905156626`。

全程 CPU-only/no-update：cuda、optimizer、backward、refit、decoder、physical forward 与 rank-truncated reconstruction 均为 false/zero。新增定向测试 `18 passed`，直接相关回归 `101 passed`，全套 pytest `1232 passed, 12 deselected, 25 subtests passed`（显式可写 basetemp），compileall/Ruff/diff-check 通过；唯一 warning 为既有 `.pytest_cache` ACL。`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported` 保持。唯一下一动作：`manual_review_LSDA_spectral_result`。

## H-LOMA-CBOGT-S6-LRRC：Low-Rank Readout Recovery Confirmation（2026-09-02）

LRRC 在 LSDA 之后执行 frozen RMAL `W=[256,40000]`、`b=[40000]` 的 no-refit functional
recoverability confirmation。消费者是项目负责人审查预注册 powers-of-two family 内的首个
confirmed rank；拒绝决定是不能把 LSDA energy 或本结果升级为数学最小秩、intrinsic/physical/
architecture rank、training、generation 或 formal promotion。实现/config/review/test/CLI 的
clean commit=`2a061db61910a847118e38663d05c05bd9447063`，上游 evidence/probe/renderer/checkpoint/
OLS/detector/formal runtime 未修改。

权威 attempt-04：
`outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/`；
validation/audit/compact SHA256=`b305c38dc3c774ff995c98907977d67cfc396d56009b8a88f726e214f9efa004` /
`5ffcb3550e0a1d635f70cdf965d12c93ded6e99fadf8f0f6ef0cda829b4a6b58` /
`4ae704104accd8cebfcb0e3c19dba7deb90b4febf38bace3c7e578bd484951b8`；probe hash=
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

candidate family 固定为 `[1,2,4,8,16,32,64,128,256]`，rank-0 仅为 bias-only control。rank-256
full positive anchor 首先执行，selection/confirmation 均通过；factorized-vs-full operator
max residual=`2.4333037798e-06/2.5858223358e-06`，probe replay abs delta=
`1.1276390065e-10/7.8544505795e-10`（selection/confirmation）。随后 rank `1,2,4,8,16,32,64,128`
按注册顺序全部 selection 失败，confirmation 未执行；winner=`256`、
`rank_256_reused_as_fallback=true`、无第二次 rank-256 run。LSDA candidate boundary gaps 与
rank-36 smallest-gap=`0.0025793411` 仅作为 descriptive telemetry，无新增 threshold。

结果为 `passed / low_rank_readout_recovery_confirmation_completed`。CMOS-only exact replay、
state/no-update、streaming/no-bank、anchor 和 resource gates 全部通过；`cmos_only_replay_executed=true`、
`optical_replay_executed=false`，optimizer/backward/refit/decoder/generation/physical-forward
modification 均 false/zero。flags 仍为 `reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false`、`closed_task_relevant_bottleneck_not_supported`。

最终核验：LRRC 定向 `26 passed`，相关回归 `130 passed`，默认 Windows-native 全套 pytest
`1258 passed, 12 deselected, 25 subtests passed`；compileall、Ruff、diff-check 通过，只有既有
`.pytest_cache` ACL warning。attempt-01/02/03 保留为 immutable 历史记录，attempt-04 为当前权威。
唯一下一动作：`manual_review_LRRC_result`；不得训练、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S6-DSFA：Data-Weighted Spectral Functional Attribution（2026-09-02）

DSFA 是 LRRC 之后唯一执行的 frozen-RMAL、no-update data-weighted spectral attribution。消费者是
项目负责人审查低 operator-energy tail 在真实 detector data 上的 functional effect；拒绝决定是
不得把该 descriptive attribution 升级为 reconstruction sufficiency、minimum rank、training、
generation 或 promotion。实现/config/review/test/CLI 的 clean commit 为
`d032738c0f02af5ada85554224e045d238416e0a`；RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC、probe、renderer、
checkpoint、OLS、detector、formal runtime 与历史 evidence 未修改。

权威 attempt 为 `outputs/20260902_v2/runs/lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/`；
validation/audit/compact SHA256 为
`ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe`、
`2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e`、
`8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。结果为 `passed / data_weighted_spectral_functional_attribution_completed`。

selection bias/full MSE=`0.09840880738327583/0.009812927922827133`，confirmation=
`0.08768781986771307/0.010185289342506197`；total signed functional gain=`0.0885958794604826/0.07750253052523498`。
B256（129–256）operator-energy share=`0.0007625489`，data-weighted output-energy
share=`0.3517092/0.3451470`；该结果仅为 descriptive。LSDA q/operator identity、selection/
confirmation additive identity、LRRC replay、state/no-update、streaming/no-bank、finite 与
resource gates 全部通过；peak CUDA=`2274099712` bytes，wall=`225.532 s`。唯一下一动作是
`manual_review_DSFA_result`；治理 flags 继续为 not-ready、no training、no promotion。

## H-LOMA-CBOGT-S6-DMSA：Detector Data-Metric Spectral Attribution（2026-09-02）

DMSA 是 DSFA 之后唯一执行的 frozen-RMAL、no-update detector data-metric attribution。消费者是
项目负责人审查真实 detector data 的 second-moment 与 centered-variance 如何加权冻结 readout
的 LSDA spectrum；拒绝决定是不得把 descriptive weighted-spectrum 结果升级为 mechanism、
reconstruction sufficiency、minimum rank、training、generation 或 promotion。起始 lineage HEAD 为
`39b0ac449986df31b9102c0b1296dda912f7b39c`，正式结果绑定 clean implementation/result commit 为
`9818e87441eb2c86eef06c18c1aa646f3e57dea5`；上游 RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC/DSFA、
probe、renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 未修改。

权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/`；
validation/audit/compact SHA256=`f0eebcb978851b9ab58453e1b28e920eddc93e86a2b82133af0bd7ec8e8ce93d` /
`c9cc448f1b58040d45b8b77c9cfaa9facd536cdcf40d0b4f152efe42e16411fe` /
`39564f0ce3973bd738d196191a06c7990407f9b7b6ab7e6e8a2eb3366dc2d0ac`；probe SHA256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。attempt-01 保留为
validation-only invalid harness 记录，不消费。

正式结果为 `passed / detector_data_metric_spectral_attribution_completed`，15/15 audit gates
通过。fit/selection/confirmation 为 1000/1000/5000 样本，使用 population `N` 的 float64
streaming moment contract，三组 `S=C+mu muT` identity 均通过；selection/confirmation
mean/variance energy fractions=`0.4019555898/0.5980444102`、`0.3704005497/0.6295994503`。
second-moment effective/stable rank=`14.8728108/2.2436135`、`17.4264177/2.4233991`；
centered-variance=`36.6563260/8.1270337`。raw spectrum 与 LSDA identity residual 为零，
这些均为 descriptive telemetry，未设置 dominance 或 panel-agreement gate。

回放为 exact-panel CMOS-only streaming/no-bank，peak CUDA=`2274099712` bytes、wall=`84.922 s`；
optimizer/backward/refit/decoder/optical replay/generation 均未发生，state/OLS unchanged。
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported` 保持。DMSA 定向 `21 passed`，直接相关回归
`111 passed, 1 deselected`，compileall/Ruff/diff-check/CLI checks 通过；全套 pytest 为
`1285 passed, 12 deselected, 25 subtests passed`，另有 4 个无关环境失败。唯一下一动作是
`manual_review_DMSA_result`；不得训练、decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S6-MCFA：Detector Mean-Centered Functional Attribution（2026-09-02）

MCFA 在 clean implementation/result commit `069cfa4b3aeda3943d754d739daf6daeba2a54a6` 上完成。
它消费冻结 RMAL `W=[256,40000]`、`b=[40000]` 与 LSDA/LRRC/DSFA/DMSA 及 ZCSA/RWSA-ACTIVE
身份，真实消费者是项目负责人审查独立 fit/selection/confirmation panel 上 `mu W` 与
`(z-mu)W` 的 functional MSE gain 分解；拒绝决定是不得把结果升级为 dominance、
reconstruction sufficiency、training、generation 或 promotion。上游 evidence、probe、
renderer、checkpoint、OLS、detector、formal runtime 与历史产物未修改；本轮只新增 MCFA
producer/config/review/test/CLI route。

attempt-01 是 validation-only failed harness，正式 replay 在 panel image runtime binding 处
停止并按 write-once 保留；权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/`，
validation/audit/compact SHA256=`35ad479aa931cb2e605bb9b582306549512106a5c2dcd070bfde35a2e552e7f1` /
`f69664cf9da1d61e90742a09614c4535388402f9228e401d2451e3ebeeefdc0b` /
`e95c072bde8306dd3f7cb8d6c5af1a671a778cede8f84c285c450af854059494`；probe SHA256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

正式结果为 `passed / detector_mean_centered_functional_attribution_completed`，audit
15/15 gates 通过。fit/selection/confirmation 样本数为 `1000/1000/5000`；mean/centered
output-energy fractions=`0.3829179441/0.6170820559`、`0.4019555898/0.5980444102`、
`0.3704005497/0.6295994503`；mean/centered/total functional gains=`0.0313323622/0.0504929026/0.0818252648`、
`0.0396463978/0.0489494817/0.0885958795`、`0.0316138044/0.0458887261/0.0775025305`。
两遍 CMOS-only streaming replay 共 14000 visits；energy/MSE/gain/cross-identity、
panel isolation、finite、state/no-update、streaming/no-bank 与 attribution boundary 均通过。
peak CUDA=`2274099712` bytes，wall=`183.266 s`；完整 pytest=`1310 passed, 12 deselected,
25 subtests passed`。flags 继续为 `reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、
`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。唯一下一动作是 `manual_review_MCFA_result`；
不得训练、decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S7-MPCDAR：Mean-Preserving Centered Data-Aware Readout Recovery（2026-09-02）

MPCDAR 在 clean implementation commit `b59172750d0c88bbc9f7426ab7a21a1d63639f57` 上完成，
消费冻结 RMAL `W=[256,40000]`、`b=[40000]`、MCFA fit moment contract 与 LSDA/LRRC/DSFA/DMSA
及 ZCSA/RWSA-ACTIVE 身份。其唯一科学问题是预注册
`[1,2,4,8,16,32,64,128,256]` 内，mean-preserving fit-derived data-aware centered
operator 首次通过完整 RMAL selection + confirmation Gate 的 rank；rank 0 只作 bias-only
control，matched mean-preserving raw-SVD 只作 descriptive comparator。

权威 attempt-03 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/`；
validation/audit/compact SHA256=`b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2` /
`43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785` /
`9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9`。attempt-01/02 为
validation-only 或正式执行前停止记录，按 write-once 保留且不消费；RMAL、probe、renderer、
checkpoint、OLS、detector 与父级 evidence 均未修改。

fit replay 的 mu/C hash 与 DMSA 匹配，population moment identity relative residual=`2.98546e-16`；
PSD raw minimum eigenvalue=`3.23645e-09`，material/tiny negative count=`0/0`。A_fit data-aware
cumulative energy at `1/2/4/8/16/32/64/128/256`=`0.118309/0.207018/0.355819/0.541212/
0.715164/0.868603/0.963351/0.994898/1.000000`，effective/stable rank=`37.184239/8.452439`。
DA boundary gaps 来自 A_fit/G_fit spectrum，raw LSDA gaps 只作为明确标记的 comparator，未创建
degeneracy threshold。

rank-256 anchor first 通过，V orthogonality max residual=`1.70464e-12`，A_fit/W projector
max residual=`4.53214e-14/5.68008e-11`，selection/confirmation full prediction identity 与
RMAL gates 通过。primary DA search 严格实际评估 `1,2,4,8,16`：1 的 strict-template crossing
失败；2/4 的 instance-sensitive 与 strict-template crossing 失败；8 selection 通过但
confirmation strict-template crossing 失败；16 selection 与 confirmation 均通过。因此
`first_confirmed_data_aware_candidate_rank=16`、`previous_registered_candidate=8`，未重复
rank 256。rank-16 selection/confirmation probe MSE=`0.0210726835/0.0212836881`，relative
improvement=`0.660383/0.631703`。

matched A/B/C control 对每个实际 evaluated rank 同流记录 old LRRC raw、mean-preserving raw、
mean-preserving data-aware 三路 MSE 及 A→B/B→C/A→C delta，不影响 primary stop 或 confirmation。
正式结果为 `passed / mean_preserving_centered_data_aware_readout_recovery_completed`，audit
14/14 gates 通过；CMOS-only streaming/no-bank/no-target-fitting、state/no-update、finite 与
resource gates 通过，peak CUDA=`2274099712` bytes、wall=`183.266 s`。治理 flags 保持
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。唯一下一动作是 `manual_review_MPCDAR_result`；
不得训练、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S8-MPCDAR-R16-IQ：Frozen MPCDAR Rank-16 Readout Integration Qualification（2026-09-02）

S8 消费 S7 MPCDAR rank-16 的 frozen mean-preserving data-aware artifact，唯一科学问题
限定为：四张 CPU float64 tensor 在 artifact write-once、Process A 退出、Process B fresh
reload 后，是否在同一 feature stream 上复现完整 RMAL selection + confirmation。拒绝决定是
不把 integration qualification 写成 minimum/intrinsic rank、native optical reconstruction、
一般 reconstruction sufficiency 或 decoder/training/generation/promotion authorization。

实现最终绑定 clean commit `c0140577fddb56af79d72a6d473cc49ef04f5ffc`；权威 attempt-03 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/`。
validation/audit/compact SHA256=`844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb` /
`3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4` /
`12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271`；artifact manifest/payload
SHA256=`e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45` /
`e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e`；probe hash 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。旧 attempts 按
write-once 保留，attempt-03 为当前权威。

Artifact keys/shapes/dtypes：`mu_fit[256]`、`c_mean[40000]`、`T16[256,16]`、
`V16[40000,16]`，均为 `torch.float64`；scalar count=`684352`，raw tensor bytes=`5474816`，
serialized payload bytes=`5476856`。fit replay `N=1000` 的 mu/C hash 与 parent 匹配，moment
identity relative residual=`2.98546e-16`；rank-256 positive replay max abs/
relative-Frobenius residual=`5.68008e-11/1.17382e-13`，construction identity 通过。

Process A build 后退出，Process B fresh reload；loaded/reference selection 与 confirmation
prediction residuals 均为零。selection `N=1000` loaded/reference probe MSE=`0.02107268348957069`，
full RMAL reference=`0.009812927917651195`，relative improvement=`0.6603834534521028`；
confirmation `N=5000` loaded/reference probe MSE=`0.02128368811373826`，full RMAL reference=
`0.010185289342547572`，relative improvement=`0.6317033169406975`，bootstrap 95% CI=
`[0.03618517890572548,0.036830976605415344]`。两 panel Gate vector 与 S7 parent 相同且
全部通过，audit=`16/16`，结果为 `passed / frozen_mean_preserving_data_aware_rank16_integration_qualified`。

factorized MAC=`644096/sample`，full-W comparator=`10240000/sample`，deterministic operation-count
ratio=`15.9`，不作 latency/throughput/energy-efficiency claim。runtime independence 中
W/b/C_fit/dataset/probe/research imports 均未被 runtime 使用；default-disabled、explicit artifact、
CMOS-only streaming/no-bank、state/no-update、finite/resource gates 通过，peak CUDA=
`2274099712` bytes，wall=`133 s`。治理 flags 保持 reconstruction/generative false、
generation_readiness=`not_ready`、training/promotion=false 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_MPCDAR_R16_IQ_result`，不得自动 make rank16 default、训练、decoder、generation
或 promotion。

## H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ：Frozen MPCDAR Rank-16 Float32 Numerical Qualification（2026-09-02）

S9 是 S8 rank-16 artifact 的独立 no-update float32 qualification。消费者仅审查：S8 CPU
float64 artifact 经确定性逐元素 cast、Process A→B fresh filesystem reload 后，在同一 CMOS
feature stream 上满足 preregistered numerical tolerance，并保持 RMAL selection/confirmation
Gate-vector identity。拒绝决定是不把它解释为 GPU-ready、production、native optical reconstruction、
minimum/intrinsic rank、一般 reconstruction sufficiency 或 decoder/training/generation/promotion
authorization。

实现/fix commits=`4b75e4d67efc1dc90a01d1d33a6aa3c1c243c652` /
`ebbd0d232c56ee45a69dea830911eff4f697ad47`。attempt-01 是 write-once 保留的 GateResult
serialization harness failure，未消费；权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/`。
validation/audit/compact SHA256=`5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137` /
`0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde` /
`5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940`；F32 manifest/payload
SHA256=`1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`。

F32 artifact scalar count=`684352`，raw/serialized bytes=`2737408/2739448`，candidate family=`[16]`。
selection/confirmation max prediction difference=`7.8160793304e-7/7.8570211703e-7`，normalized
L2=`1.5589117128e-7/1.5627624021e-7`，aggregate MSE absolute drift=`8.2576945282e-11/
2.8491427967e-11`，relative drift=`3.9186724995e-9/1.3386508868e-9`；数值门、RMAL Gate-vector
identity、fresh reload、CPU boundary、state/no-update、no-bank 与 runtime-independence 全部通过。
audit=`35/35`，classification=`frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified`，
`integration_only=true`。唯一下一动作是 `manual_review_MPCDAR_R16_F32_NQ_result`；不得训练、
decoder、generation、promotion 或自动切换 default。

## H-LOMA-CBOGT-S9V-FULL-PATH-VIS：Full-Path Optical Representation Visualization（2026-09-02）

本轮是 descriptive-only visualization consumer，真实消费者为项目负责人对冻结 RMAL readout 的固定
confirmation panel 完整光学路径、重建方法族、detail/error subset 与 MPCDAR R16 F64/F32 显示进行人工审查。
不得把图像显示升级为 reconstruction/generative manifold、性能、bottleneck、training、decoder、generation
或 formal-promotion 证据。执行时 HEAD=`7da2d066d8a8f1b3dbcff7c94393c51d11d7bcb7`、branch=
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，S9V closeout commit=
`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`，当前 worktree clean。

父级只读身份锁定 S7 `H-LOMA-CBOGT-S7-MPCDAR_attempt-03`、S8
`H-LOMA-CBOGT-S8-MPCDAR-R16-IQ`、S9 `H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ`，并核验各自
validation/audit/compact 与 S8/S9 artifact 哈希。固定样本为
`{0:55006,1:55000,2:55001,3:55008,4:55004,5:55010,6:55012,7:55003,8:55002,9:55007}`；
阶段、pre/post-OLS alias、10×10 modal display、p99.5 amplitude、raw phase mask、fixed reconstruction/error
range 与 real SLM/CMOS phase N/A 均按已批准合同冻结。

权威输出目录为
`outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01/`；
manifest SHA-256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`，26 files、
`16184241` bytes。manifest status=`completed`、classification=
`descriptive_visualization_completed`、forward consistency max residual=`0`、model/OLS state unchanged；
包含 full optical path 两页、reconstruction family、detail/error classes `[0,2,3,5,8]` 与 F64/F32 comparison。
代表性 PNG 已完成人工视觉核验；F64/F32 差异按固定 `[0,2e-4]` raw map 与 display-only ×10^6 保存。

S9V 定向测试=`9 passed`，父级 artifact/回归定向=`48 passed`，默认全量 pytest=
`1347 passed, 12 deselected, 25 subtests passed`；compileall、Ruff、diff-check 均通过。全程无 training/refit/
optimizer/backward/evaluator/prior sampling，治理 flags 继续为
`reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。Bridge 仍为
`invalid / cloud_sources_sync_blocked`，未修复、未上传。唯一下一动作：
`manual_review_MPCDAR_R16_RUNTIME_PQ_implementation_contract`。

## H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ：Frozen MPCDAR Rank-16 Float32 Runtime / Performance Characterization（2026-09-03）

S10 是冻结 S9 MPCDAR rank-16 float32 artifact 的 CPU-only、readout-only runtime/performance
characterization。消费者是项目负责人审查 fresh-process initialization、确定性 correctness、
latency、throughput 与 storage telemetry；拒绝决定是不把结果升级为 measured speedup、GPU/硬件/
production readiness、reconstruction sufficiency、decoder、training、generation 或 promotion。
S9V closeout commit=`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`，S10 starting HEAD=
`8680c12be955df72906b955433e6daeabfdfec4a`；S10 implementation/fix commits 为
`dd1c38fa965e32deedd655f00f3cd974e1534cd8`、`488f12fbcf19e3663c7aa685bc4d4ad717db2a30`、
`d0655652cd86bdc9f3852a9bbf4faa45f59f069b`。S9/S9V parent、frozen runtime、model、OLS、probe、
detector 与既有 evidence 未修改。

当前权威 write-once attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_runtime_performance_qualification_v1_attempt_02/`；
validation/audit/compact SHA256=`8293e549b1ae48447e28be924406525aa03fa8843c0f1a71fa1ec21e3f5894c5` /
`7b519b3a75cdf59ecfe2678d46a8f74ae3084cb7b9de90c95763efecc0f2f4ee` /
`ded0959097c88a74031edc1c0aa925d1a95a2df7f1abc5c6d9ba6ca7abdc3173`。attempt-01 在 namespace-
isolation 修复前完成，按 write-once 保留但不消费。父级 S9 manifest/payload SHA256 为
`1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`；S9V manifest SHA256 为
`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`。

冻结公式为 `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`。固定 CPU dense finite
fixture 为 `float32[32,256]`、seed=`20260902`、SHA256=`45ca137458856a7305ecf9500220c120aa5b4947e6f58d476b791e49b6337ab6`；
B=`1/8/32` direct-formula replay 与 repeated-call determinism 全部通过，max absolute 与
normalized-L2 residual 均为 `0`。fresh-process initialization 的 manifest/payload/consumer/total
timings=`0.2792/9.1288/20.8566/30.286 ms`，fresh-process wall=`3657.2196 ms`；filesystem cache flush
为 `false`，不声称 true cold-disk latency。

稳态计时为 warmup=`50`、repetitions=`200`、`time.perf_counter_ns`、CPU eager synchronous，quantile
method=`linear: index=(n-1)*q`。B=1/8/32 batch mean latency=`0.5601775/0.8914525/3.277882 ms`，
per-sample mean=`0.5601775/0.1114315625/0.1024338125 ms`，throughput mean=
`1836.8588342/9338.7841702/9874.3015643 samples/s`；compact evidence 保存其余 median/p95/p99/min/max。
storage telemetry 为 manifest=`3219`、serialized=`2739448`、raw/runtime tensors=`2737408` bytes；
process memory delta/peak/RSS=`not_recorded`。分析 operation count 为 factorized/full-W=
`644096/10240000 MAC/sample`、ratio=`15.89825119236884`，不是 measured speedup；GPU、full-W runtime、
end-to-end pipeline=`not_evaluated`。

环境记录为 Python `3.11.16`、PyTorch `2.5.1`、torch threads `8/8`、logical cores=`16`、CPU model=
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`；task expected Windows=`10.0.26200.0`，observed=
`10.0.26100`，因此 version mismatch 已记录但 registered Python/Torch/Windows CPU execution stack 通过；
physical cores/RAM=`not_recorded`。结果为 `passed / frozen_rank16_float32_runtime_performance_characterized`，
audit=`26/26`，diagnostic-only。S10 定向/布局/相关回归=`30 passed`，全套 pytest=`1357 passed, 12 deselected,
25 subtests passed`，compileall/Ruff/diff-check 通过；唯一 warning 为既有 pytest cache ACL。治理 flags
继续为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`；Bridge 仍为
`invalid / cloud_sources_sync_blocked`。当前 worktree 在状态提交后保持 clean；唯一下一动作是
`manual_review_MPCDAR_R16_RUNTIME_PQ_result`，不得训练、decoder、generation 或 promotion。

## H-LOMA-GEN-P1-R2-UACR：Scaled-Chen Unit-Aware Calibration Contract Repair（2026-09-04）

R2 仅消费不可变 C6 raw-scale replay 与 parent v1 calibration，修复 modal coefficient-power
coordinate 的 absolute float32-eps floor 语义；没有新的 optical forward/CUDA、downstream replay、
prior generation、decoder、CMOS/MPCDAR、training 或 refit。新 v2 payload 明确保存 raw mean 与
registered scale，且 registered scale 逐 port 等于 raw mean；source C6 manifest/payload、dataset
manifest、calibration indices、mode bank 和 parent v1 SHA 均显式绑定。v2 loader 要求这些 identity
bindings，并拒绝 historical v1 fallback；v1 reader 与旧 P4 measurement authority 不变。

当前 write-once authority：
`outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/`；
attempt-01 保留且未覆盖。六个输出文件共 `159393` bytes，无 TIFF、隐藏文件或 `.tmp`。Gate=
`R2_CALIBRATION_REPAIR_PASS`；R2/C6/Scaled-Chen BOGT/G1 直接回归=`48 passed`，compileall、Ruff、
`git diff --check` 通过。raw FP64 min/median/mean/max=`1.508997e-12 / 1.127986e-10 /
1.741773e-10 / 6.304158e-10`；z global mean=`0.9999999986`，最大 per-port mean deviation=
`1.451667e-7`。该结果只修 calibration contract，不声称 P4 旧坐标结论被推翻，也不授权 generation、
training 或 formal promotion。治理仍为 `mainline_state=closed_at_current_evidence_boundary`、
reconstruction/generative unsupported、generation not ready、training/promotion=false。唯一下一动作：
`PLAN_REPAIRED_COORDINATE_REPLAY`。

## H-LOMA-GEN-RCR-A：Repaired-Coordinate MRA Replay（2026-09-04）

本轮 initial state 为 clean，branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting
HEAD=`ef1b336984b701f1905a06fc868c68c4b4892057`。RCR-A harness/config/tests 已独立提交为
`c0bbe12db909d4ceb5176f72dbdb69df5c6c8ff5`。正式 Stage A 使用 parent MRA attempt-05 的固定 panel/索引和
step-5000 checkpoint SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`，仅替换
calibration identity 为 R2 `scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`；R2 payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`，manifest SHA-256=
`0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6`。

repaired coordinate 的 selection/confirmation z mean 为 `1.0060573/0.9250944`，variance 为
`1.5498618/1.4546335`；Gaussian KL 为 `3.2833059/2.8498273`，numerical truncated-normal KL 为
`3.2492599/2.7428405`。旧 MRA formal values 由 audit.json 精确读取；complex rank
`12.2361650/12.8739576` 与同一前缀 replay 一致，power rank 变为 `7.2165284/7.3683505`，仅作为
coordinate-dependent 指标。两组 mean 均满足 descriptive `±0.10` 对齐，但 variance 是 registered prior
`0.25` 的 `6.1994×/5.8185×`，且 variance-shape term 在两组 Gaussian-KL 分解中均主导，故 Gate=
`MRA_PRIOR_CONTRACT_REVIEW_REQUIRED`，dominant assumption=`unconditional_truncated_normal_variance=0.25_with_lower=0.0`。

唯一完整 authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_02/`，manifest SHA-256=
`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`；4 个 JSON 共 `53575` bytes，无 TIFF、
隐藏文件或 `.tmp`。attempt-01 是首次发布顺序缺陷留下的 3 个中间 JSON，保留但不消费。state digest
before/after=`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`，decoder/relay/CMOS/MPCDAR、
prior generation、optimizer、backward、training、refit 均未执行；Stage B G1-SA 未执行。RCR-A/直接依赖测试
`46 passed`，compileall、Ruff、diff-check 通过；peak CUDA=`1599782400` bytes，runtime=`39.8108 s`。

治理继续为 `mainline_state=closed_at_current_evidence_boundary`、reconstruction/generative manifold unsupported、
generation not ready、training/promotion=false。下一动作固定为 `PLAN_PRIOR_SHAPE_CONTRACT_REVIEW`。

## H-LOMA-GEN-PCR-PSD：Repaired-Coordinate Prior Shape Decomposition Observer（2026-09-04）

本轮只执行 `IMPLEMENT_PRIOR_SHAPE_DECOMPOSITION_OBSERVER`。正式 output=`outputs/20260904_v2/runs/lightgen_scaled_chen_prior_shape_decomposition_v1_attempt_01/`，PCR Gate=`PCR1_PRIOR_CONTRACT_INTERNALLY_COHERENT`；P-S3、P-S4、P-S5 均为 `SUPPORTED`，selection/confirmation 均为 `REPLICATED_DIRECTION`。该结论表示 registered factorized truncated-normal target 在 sampler/config/KL/coordinate 语义上内部一致，而 frozen step-5000 `mse_only` q(z) 的 port-wise marginal heterogeneity、cross-port dependence 和 marginal tail mismatch 已被量化；不表示 prior 错误、q 已匹配 p，也不授权 prior fitting 或 training。

starting/ending HEAD=`02fb1f636aa17e0e2f7f01a3fa70275b8f19dbdf`，branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`。只复用 RCR-A selection/confirmation panel、dataset identity、100-port OLS、mode bank 与 R2 repaired coordinate；checkpoint SHA-256=`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`，R2 payload SHA-256=`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`，RCR-A manifest SHA-256=`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`。selection/confirmation counts=`1000/5000`；CPU prior-null seeds=`260904/260905`、replicates=`128`、99% central interval，未保存 replica samples。

selection mean heterogeneity std/range=`0.7455607/3.5774960`、within/between/total=`1.5498618/0.5558607/2.1057225`；confirmation 对应为 `0.6458730/2.9762450`、`1.4546335/0.4171519/1.8717854`。mean absolute off-diagonal correlation=`0.1600160/0.1617629`，p95=`0.4607387/0.4692335`，max=`0.8836253/0.8846894`，off-diagonal covariance fraction=`0.7362989/0.7411037`；prior-null primary percentile 两 panel 均为 `100%`。prior-defined lower/upper/central tail masses=`0.156820/0.134430/0.708750`（selection）与 `0.168390/0.114900/0.716710`（confirmation）。

manifest SHA-256=`228679fda073b2e4e1e46c931b743b490930dc895313e5c201283d244e22d97a`，metrics SHA-256=`4ea665c5a021a09c9ce2d44adfc31ed0cdb593de30b0ed3dbb3b10986bfbe9d2`，8 个 artifact 共 `639294` bytes；无隐藏文件、`.tmp` 或 TIFF。learned-state digest before/after=`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`，unchanged；optimizer/backward/refit/checkpoint/prior_fitted/decoder/CMOS/MPCDAR/training 均 false。PSD 定向测试=`7 passed`，PSD+RCR-A+Scaled-Chen BOGT 相关回归=`36 passed`，compileall/Ruff/diff-check 通过；唯一下一动作：`IMPLEMENT_REPAIRED_COORDINATE_G1_SA_REPLAY`。治理仍保持 reconstruction/generative unsupported、generation not ready、training/promotion=false。
## H-LOMA-GEN-T1-RCAWR：Repaired-Coordinate BOGT alignment weight registration（2026-09-04）

本轮 scope 仅为从冻结 RCWV attempt-01 登记 repaired-coordinate BOGT alpha；不运行 CUDA 或
gradient measurement，不改变 calibration/prior，不创建 optimizer，不执行 training、checkpoint、
refit、generation、decoder、CMOS 或 MPCDAR。consumer 绑定 R2
`scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`、step-5000 parent checkpoint 和
RCWV artifact identity；Set B 是 registered preflight global-alpha replay authority，Set A 是
step-5000 replay context。

冻结 RCWV raw norm/effective-ratio identity 经 Decimal replay 得到
`alpha_repaired=0.00017905883237862847`，canonical Decimal 为
`0.000179058832378628469663887774247392581098542509215070071798828`。三 panel predicted ratios
为 `0.1166303093/0.0925572168/0.0941462530`，通过 inclusive `[0.05,0.40]`，Gate=
`RCAWR_REGISTRATION_PASS`。attempt-01 因 manifest 发布顺序检查过早而只留下 4 个中间 JSON，保留但
不消费；正式 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_02/`；
输出只含 5 个 JSON，manifest 绑定 source hashes 与 no-update state。

治理保持 reconstruction/generative manifold unsupported、generation not ready、training/promotion
false，下一动作是 `IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY`；下一步仍需新的 canary 授权，不得由
本次 weight registration 自动进入 training。

## H-LOMA-GEN-T1-GAC：Repaired-Coordinate Generative Alignment Canary（2026-09-04）

GAC 是 RCAWR 注册 alpha 的唯一 bounded single-intervention training-direction consumer，不是
generation 或 formal promotion consumer。任务初始 clean preflight 为 branch=
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、HEAD=
`88a09b5f737f4eb9246ad97351b367a5e3a20841`；新增 harness/config/tests 已先独立提交为
`5913a045b0bd932aff11e335f41a1f246b78a0f3`，正式 canary 在该 implementation HEAD 上执行。唯一
允许的 intervention 是 exact step-5000 parent → 1000-step continuation → fixed step-6000 endpoint，
恢复 parent Adam state，alpha=`0.00017905883237862847`、gamma=`0`、objective=`mse_kl`；旧 alpha
`0.002996375650268832` 未消费，calibration/prior/latent/BOGT formula/optics/OLS/fiber/architecture/
MPCDAR 均未修改。

3-step real-update preflight 与冻结 parent negative control 均通过。正式 write-once output 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_attempt_01/`，
10 个 JSON/PT、`17055800` bytes；training manifest SHA-256=`48d3e883355b329c80072c50cb2b06187d4cc7454911556f9b124115ab927630`，
end checkpoint SHA-256=`b8ed605d0ed4506c0415f8199ac16fd1f4274cc7efa419f00af785a2ce4f9542`。训练实际完成
5001–6000 共 1000 steps，runtime=`1694.4757 s`，peak CUDA=`2330944000` bytes；start/end learned-state
digest=`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5` /
`e8fa50507e500a7f82ce5675080bd769f0bc5c8edcf59ae6e43c841e0fa7d1c0`，OLS unchanged，只有 encoder/
decoder phase 参数改变。

端点 G1-SA：R50=`1.8142581`，95% CI=`[1.7888644,1.8446554]`；R95=`1.0970011`，95% CI=
`[1.0733978,1.1168799]`。parent R50 CI lower=`1.8142740`、R95 CI lower=`1.0679024`，故严格
CI-separated primary Gate 未通过。MRA selection/confirmation z mean=`1.0134035/0.9309632`，
variance=`1.4821789/1.3868347`，Gaussian KL=`3.0508981/2.6405313`，numerical KL=
`2.9807997/2.5035856`，complex rank=`12.2222137/12.8533897`，power rank=`8.2595968/8.4111090`；
mean-scale 与 rank safety 通过。current-sample same-route safety 未通过：paired MSE=`0.05678915`，
template control=`0.04197700`，cyclic control=`0.07559964`。

最终 Gate/classification=`T1_ALIGNMENT_HARMFUL`。这是固定 canary 的有效 trade-off failure，不是
全局训练失败或 generative-manifold 结论；没有 prior/decoder/CMOS generation、FID、G2 或 formal
promotion。GAC focused regression=`42 passed`，compileall、Ruff、`git diff --check` 通过。治理仍为
`mainline_state=closed_at_current_evidence_boundary`、reconstruction/generative unsupported、
generation not ready、training/promotion=false；唯一下一动作为 `REVIEW_ALIGNMENT_TRADEOFF_FAILURE`。

## H-LOMA-GEN-T2-SAAC：SCAM normalized-power controllability canary（2026-09-04）

T2-SAAC 在 clean implementation boundary `2665b9d13aedef898eff7e2b567f1988feeccf9b` 上完成同入口
3-step real-update preflight 后，执行唯一一次固定 step-5000→6000、1000-step bounded canary。目标为
`L_total=L_MSE+beta*L_ED`，`beta=0.0014299609134309638`、`alpha_BOGT=0`、`gamma=0`；parent、repaired
calibration、registered prior、Energy-Distance、optimizer、horizon、seed、four primary gates 与 safety
gates 均不变。只更新 encoder/decoder phase，OLS 保持冻结；无 prior decoding、semantic OLS audit、
MMD/SWD/SCED/Chamfer/Sinkhorn、new prior/beta、G2、bridge 或 promotion。

正式 authority：`outputs/20260904_v2/runs/lightgen_scaled_chen_support_aware_alignment_canary_v1_attempt_01/`，
10 个 JSON/PT、`17357744` bytes，training manifest SHA-256=`8b46943ce884143ebe24b0a10f782738d7ec0517b79f5efba0ac343c173807b2`，
end checkpoint SHA-256=`c1a834c755f7eef27b8e1b3ed7ce5b1f154eca7fe775ccd5ea65e05d14fa82a8`。R50=`1.7548616`
（CI `[1.7309163,1.7764292]`，该项通过），R95=`1.0573063`（CI `[1.0348356,1.0767013]`，该项未通过）；
paired prior ΔP50/ΔP95 upper95=`0.0076903/0.0293099`，未通过；parent-relative MSE upper95=
`-0.0009922`，通过；mean/rank safety 通过。有效 Gate=`T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED`。

SCAM 只把 `z_j=|c_j|²/s_j` 解释为 `normalized_power_coordinate_observer`，不是完整
`c∈C^100` complex semantic OLS latent。`SCAM authority=FAIL` 仅为本次 canary Gate；semantic OLS=
`UNESTABLISHED`，label-to-OLS=`UNRESOLVED`，8D-to-OLS=`PARTIAL / UNRESOLVED`，global-prior family=
`CLOSED_AFTER_THIS_CANARY`。治理保持 closed evidence boundary、reconstruction/generative unsupported、
generation not ready、training/promotion=false；唯一下一动作：
`PLAN_S1_OLS_SEMANTIC_REPRESENTATION_AUDIT`。
