# 当前主线交接：v2 formal evaluation 与 objective diagnostic 收口（2026-08-12）

## WAE electronic O0 / SMF density branch closed (2026-08-22)

The first WAE O0 implementation was reclassified as `invalid_harness` and
preserved in `experiments/archive/v2_wae_o0_v1_invalid_harness/`; its old
`wae_image_manifold_failed` record has no downstream authority. The corrected
v2 contract was then run on clean commits `17ee681`, `e6673b0`, and `ed03cf5`.
It used encoded-latent reconstruction, full-coverage balancing (625 steps and
55,000 unique train rows), deterministic CUDA replay, true random-label
rerendering, independent reference diversity/coverage, official 10,000-test
generation, JVP Jacobians, and chunked memorization checks. Evaluators were
not loaded during calibration or training.

The corrected seed-7301 confirm is a valid scientific failure, not a harness
failure. Across locked panels, primary accuracy was `0.930–0.952`, secondary
accuracy `0.826–0.852`, secondary minimum recall `0.36–0.44`, and agreement
`0.828–0.856`; official-test secondary accuracy was `0.8374` with minimum
recall `0.3683`. Camera code fraction was about `0.637`, label-shuffled
primary accuracy `0.090–0.102`, all 30 Jacobian points had rank 8, and replay
was exact. The semantic contract therefore failed upstream of any O0-G or
fiber interpretation, while the acceptance harness remained valid.

The gate stopped before 7302/7303, locked-final, O0-G, P100/P196/P400, ridge
readout, E1, and fiber-aware training. The result is
`training_authorized=false`, `formal_promotion=false`, and
`synthetic_diagnostic_only=true`; the fixed action remains
`close_fiber_restart`. This does not establish an optical, SMF, modal-density,
phase-only, hardware, or universal evaluator failure. No additional WAE
epochs, seed substitution, or downstream fiber run is authorized without a
new electronic hypothesis that can change this decision.

## v2 synthetic fiber route closed (2026-08-22)

The synthetic fiber route is closed with status
`closed_synthetic_fiber_route_no_e1`. The final evidence capsule is
`experiments/archive/v2_fiber_route_no_e1_closeout/`. Its real consumer is
this handoff and the future research-authorization gate. The fixed rejection
decision is `no_fiber_training`, `no_e1`, `no_modal_regularizer`, and
`no_formal_promotion`.

The reverse necessity audit passed before archival: v3 is a valid failed
closeout; interface attribution retained class structure and local port
Jacobian rank; global exposure recovered most oracle gain without restoring
minimum recall; and the decoder attribution ended with
`close_current_fiber_route_no_e1`. No additional CUDA measurement, evaluator
comparison, training, or E1 is authorized for this route.

The only restart condition is a new electronic O0 hypothesis that can change
the rejection decision, followed by independent qualification and a multi-seed
dual-evaluator digital oracle. Even then, a new synthetic fiber lifecycle must
be registered before any fiber work, and real hardware still requires an
independent fiber/camera calibration contract.

This state is diagnostic-only: `formal_promotion=false`,
`training_authorized=false`, and `synthetic_diagnostic_only=true`.

## 项目级 harness 封存（2026-08-16）

Phase 1 harness governance 已在干净提交 `69c5c24`（`harden v2 formal
acceptance lineage`）中独立形成。该提交包含 current gate registry、
`invalid_harness`/`valid_failed` 语义、primary→secondary evaluator lineage、
training authorization 双边绑定、official-test exposure 链，以及 current
consumer 的 fail-closed 检查。提交前验证为：current lineage/CLI 47 passed、完整
套件 527 passed（138 deselected，25 subtests）、research 70 passed、compatibility
66 passed、scheduled-diagnostic 2 passed、compileall、ruff 和 diff 检查全部通过。

本提交不迁移、不补签任何旧 evidence。旧 current training/measurement/evaluation
记录缺少新版 contract、status 或 exposure 要求时，只能作为审计材料读取，不能授权
recovery、training、evaluation、manual admission、candidate export、memorization、
utility、dataset admission 或 accepted manifest。历史 Objective A evaluation 的
输入和 raw-CMOS 测量已完成但未达到预注册科学 gate；其旧 `failed` record 保持原样，
不转换为新版 `valid_failed` evidence。

因此当前项目状态为 `closed_historical_valid_failure`。这是项目管理和审计状态，不是
`EvidenceRecordV2` 新 status，也不授予任何 permitted action。项目重启必须先提出并
记录新的 objective hypothesis，再从 primary evaluator qualification 开始签发全新
current evidence chain；不得直接重复已关闭的 Objective A。

本节封存不代表生成 accepted manifest，也不启动 CUDA、训练、正式 evaluation 或人工
review。工作区中与 research/diagnostic 实验有关的其他用户修改继续保留，未纳入
`69c5c24`。

## Objective D diagnostic implementation (2026-08-12)

The branch `experiment/v2-late-objective-v25` contains an engineering-only
diagnostic implementation of registered candidate D.  D preserves the formal
identity optical path and activates at global step 400 after an A-equivalent
warmup; it adds the camera, smooth worst-class relative-diversity, and coverage
auxiliaries with a shared deterministic companion schedule.  Runtime,
checkpoint, training, and diagnostic objective hashes use one neutral contract
builder and bind the warmup model/optimizer state plus primary and companion
schedules.

This implementation is not an objective-selection decision.  The current YAML
remains objective A, old A/B/C evidence is read-only, and no D calibration,
screen, confirmation, formal training, evaluation, promotion, or dataset action
is authorized.  The real consumers are the D calibration/screen/confirmation
evidence readers and the formal evaluation gate; the A run is the registered
negative control.  A failed calibration, screen, or confirmation closes this
objective family without changing the optical contract.

The first `f1accba` calibration passed its numeric gates, but its screen
consumer exposed an invalid warmup comparison: a step-400 milestone mapping was
compared directly with a state-hash string.  That calibration remains
read-only and superseded; it does not constitute a D failure.  The harness fix
requires a new clean commit, fresh qualification lineage, calibration, and
screen before any confirmation decision.

## Current identity status

The current evidence-lineage policy is clean ancestor plus scoped Git diff.
Consumers require clean ancestor commits and independently verify config, split,
checkpoint, and artifact hashes; documentation- and test-only descendants may
reuse evidence, while any change under `src/ldm_project/v2/`,
`configs/v2/current/`, or `pyproject.toml` requires reissue. Invalid commits,
dirty provenance, unavailable Git, or failed comparison is fail-closed.
Historical component-registry evidence remains read-only and cannot authorize a
new run; it is not migrated or backfilled.

The evaluation schedule harness was repaired at clean commit `ae3273d`. The
earlier evaluation attempt mixed the 55,000-sample training split with the
60,000-sample full label set and was therefore an `invalid_harness` attempt,
not a model failure.

The runtime-equivalent evidence-lineage guard was implemented at clean commit
`4aac9c0`. The reissued current evidence chain was then regenerated on
`ae3273d`; its upstream qualification, initialization, observability, control
isolation, localization, controllability failure, and semantic recovery remain
diagnostic lineage only.
This engineering closeout does not reopen the failed formal evaluation or
objective screen; no evidence is re-signed and no new mainline is authorized.
evaluator qualification (`outputs/20260811_v2/runs/20260811_011455_509777_formal_v2_identity_logit_amplitude_finite_crop_identity`),
initialization admission
(`outputs/20260811_v2/runs/20260811_011510_366668_formal_v2_identity_logit_amplitude_finite_crop_identity`),
observability
(`outputs/20260811_v2/runs/20260811_011528_573879_formal_v2_identity_logit_amplitude_finite_crop_identity`),
control isolation
(`outputs/20260811_v2/runs/20260811_011548_110334_formal_v2_identity_logit_amplitude_finite_crop_identity`),
and semantic-localization
(`outputs/20260811_v2/runs/20260811_011638_725751_formal_v2_identity_logit_amplitude_finite_crop_identity`)
all passed their gates.  The
100-step controllability smoke
(`outputs/20260811_v2/runs/20260811_012127_036513_formal_v2_identity_logit_amplitude_finite_crop_identity`)
is a valid diagnostic
failure in semantic accuracy only (`0.217647 < 0.30`); code fraction, finite
gradients, six-group updates, code-projection recovery, saturation, and
provenance gates passed.  The smoke checkpoint is diagnostic-only and must never
be used as a warm start.

The current final-checkpoint schedule is bound to the production balanced
iterator: 984 steps/epoch and 4,920 steps over five epochs.  The old 5,500-step
contract and historical recovery remain predictive/reference material only.
The reissued 400-step semantic recovery
(`outputs/20260811_v2/runs/20260811_013934_467132_formal_v2_identity_logit_amplitude_finite_crop_identity`)
is a passed diagnostic gate (accuracy `0.435294`,
classification loss `3.849→1.876`, code fraction `0.31031`, saturation
`0.05674`) and permits `train`. Formal training subsequently passed at
`outputs/20260812_v2/runs/20260812_012817_770553_formal_v2_identity_logit_amplitude_finite_crop_identity`,
with `dirty_patch_hash=null`, checkpoint hash
`423328bf472bb7468f131b351850d1b6f7c5a282796444ecbec5e10bad12c2b9`, and only
`evaluate` authorized.

Formal evaluation ended at
`outputs/20260812_v2/runs/20260812_132740_122683_formal_v2_identity_logit_amplitude_finite_crop_identity`
with `conditional_acceptance / acceptance / failed`, evidence hash
`7a3ded3c14bbb64936bcbd9fdfe59a3694e5e939c045b6684625b19a01a91bd8`, and no
permitted next actions. Accuracy and recall were 1.0 and saturation was
0.0501; camera code fraction was 0.1389, class diversity ratios were
0.278–0.458, and coverage was 1.758. Raw-CMOS and 5,000 validation labels are
complete, so this is a valid model acceptance failure rather than an invalid
harness.

No manual review, candidate export, memorization, utility evaluation, or
dataset admission was authorized. The failure does not establish phase-only,
optical-capacity, hardware, or universal semantic-diversity impossibility.
The formal current chain is closed at this valid acceptance failure; it does
not authorize repeating the same A-objective run or promoting an alternative.

## Objective diagnostic closeout (2026-08-12)

The earlier raw-weight calibration
(`outputs/20260812_v2/runs/20260812_180102_738872_formal_v2_identity_logit_amplitude_finite_crop_identity`)
reported camera `6578.77` and pairwise `10.93` as out of range.  Because those
raw values depend on loss units, that result only showed that the old
authorization contract was insufficient; it did not establish a scientific
objective failure.

The v3 calibration run
`outputs/20260812_v2/runs/20260812_183756_552830_formal_v2_identity_logit_amplitude_finite_crop_identity`
was reclassified as `invalid_harness`: its B/C combined metric included the
base gradient while applying an auxiliary-only budget, and its aggregate used
an arithmetic mean instead of the RMS contract. It did not establish an
objective failure. The correction was committed at `f0861e6`, with the
parameter-index and evidence-gate follow-ups at `8bbf78d` and `ab620be`.

The run
`outputs/20260812_v2/runs/20260812_185456_859190_formal_v2_identity_logit_amplitude_finite_crop_identity`
produced an artifact but no consumable evidence record. It is an
execution-invalid attempt, not a scientific endpoint and not an authorization
for screen.

The corrected calibration
`outputs/20260812_v2/runs/20260812_185926_470787_formal_v2_identity_logit_amplitude_finite_crop_identity`
passed its four-batch effective-gradient gates: single auxiliary ratios were
approximately `0.25`, B was `0.3424`, and C was `0.4162`; it authorized only
`screen`. Its artifact SHA256 is
`1401a4562cfda6aa3afb7a6650b5d56bf956f463eac1262175fc42f074991bb9`, and its
evidence SHA256 is
`7d9afe982d728071fd816e76bbf1b863d9a24648836d19b9b67ecd0ef2a634f6`.
Screen then ended at
`outputs/20260812_v2/runs/20260812_191311_132000_formal_v2_identity_logit_amplitude_finite_crop_identity`
with valid `conditional_objective_alignment_screen / diagnostic / failed`
evidence. B and C both failed the registered camera/diversity/coverage screen
gates; all six parameter groups updated and all candidates shared the same
schedule. The report SHA256 is
`a135e5b43e42a052c4308afe2f62de5d354dd42c50b6823d1f74727a1adaedc1`, and the
evidence SHA256 is
`2e68540c385931a5ec47c516de7a470ebb7607dc26827335951178146ef2ea68`.

Relative to A, B changed camera by `+0.00394`, minimum diversity by `+0.00475`,
and coverage by `+0.00757`; C changed them by `+0.00375`, `+0.00222`, and
`-0.00338`, respectively. Neither candidate reached the registered
`+0.02 / +0.05 / -0.05` screen thresholds. This is a valid diagnostic failure
of the tested B/C definitions, frozen weights, and 100-step contract. It is
not evidence of an optical-path, hardware, generator-capacity, or universal
auxiliary-objective failure. No winner, confirmation, objective promotion, or
formal training was authorized.
That historical screen was produced before the strict calibration-evidence and
report-hash consumer checks; it remains a read-only diagnostic observation and
cannot be reused for promotion or confirmation.

## Mainline decision point

The current configuration remains objective A. The prior A formal run is
already closed by a valid acceptance failure, while B and C have no promotion
qualification. A new mainline attempt must therefore begin with a separately
pre-registered objective hypothesis and repeat calibration → screen →
confirmation. Even a passed confirmation would only qualify a candidate for an
explicit objective-selection commit and a newly issued formal evidence chain;
it would not directly authorize five-epoch training or evaluation.

The remainder of this file records immutable historical closures and is retained
for context; those records must not be rewritten or used as current acceptance.

---

## Objective D 可恢复 screen 工程状态（2026-08-13）

当前分支仍保持 objective A；`7de2a87` 上的 v4 calibration 仅为只读数值
记录。此前 screen 未产生可恢复 checkpoint，且运行时未持久化阶段状态，因此
不能复用已消耗的训练步骤，也不构成 Objective D 的有效失败。

本次收口将 diagnostic contract 升级为 v5，并引入显式绝对路径
`--resume-checkpoint`。checkpoint 只服务 screen consumer，绑定干净提交、配置、
split、初始化、evaluator、measurement、injection、calibration 和自身 hash；中间
状态始终 `formal_promotion_authorized=false` 且无下一 action。恢复 lineage、状态、
schedule 或 manifest 任一不匹配即停止。训练 objective、400 步 activation、984 步
screen、物理架构、current YAML 和正式验收阈值均不变。

代码提交和全新资格链完成前，不授权 screen；新 calibration 通过后才启动 seed
1234 screen。有效 screen 失败则按预注册规则关闭 Objective D family；通过也只
允许 `confirm`，不能直接进入 objective selection、formal training 或 evaluation。

v5 screen 已完成 A/D 各 984 步并留下完整 checkpoint，但因 CUDA warmup 在
step 400 无法逐位复现而被判定为 `invalid_harness`；其指标只读，不构成
Objective D 的科学失败。后续 v6 修复启用 strict deterministic runtime，并要求
warmup 使用实际消费的零 auxiliary weights；v5 calibration、screen 与 checkpoint
不得授权 v6。

在 `b7f1638` 上的 CUDA deterministic preflight 已 fail closed：
`adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation`。
因此当前 PyTorch/CUDA 环境无法满足 v6 的逐位 warmup 合同；未重签资格链、未运行
v6 calibration 或 screen。该阻断是 invalid harness/环境能力问题，不构成 Objective D
失败；在获得支持 deterministic backward 的运行环境或批准新的等价实现前，保持停止。

后续根因收口取消了对当前 CUDA 环境无法提供的独立逐位重演要求，不替换或弱化
正式 pooling 算子。v7 calibration 将唯一的 step-400 A model/optimizer state作为
无 promotion 权限的显式 warmup artifact；screen 的 A/D 从该同一artifact分叉，
confirmation则为每个seed只计算一次warmup后在内存中精确分叉。消费者在创建run前
验证artifact、evidence、state、optimizer、history、schedule与初始化lineage。
v5/v6 evidence及checkpoint均不得授权v7。

---

# 历史主线记录：v2.2 条件性负结果已封存（2026-08-04）

## 当前状态

v2.2 capacity-map 已正式关闭为 `closed_conditional_negative`。这是
diagnostic/research-only 状态，不改变 formal current 配置，不授权 promotion、
CUDA 训练、O1--O5、Teacher/WGAN 或外部数据集工作。

唯一允许结论：当前 `E2:O0_declared_teacher_free_objectives` 未通过合格
双 evaluator 门。dual objective 的三次确认均未通过，因此
`confirmation_pass_count=0/3`，下游状态是 `stopped_by_o0_gate`。

## 证据与治理

仓库内只读证据胶囊位于
[`experiments/archive/v2_2_capacity_map_closeout/`](../../experiments/archive/v2_2_capacity_map_closeout/)。
它包含六个 smoke、三个 confirm、完整 diagnose/曝光扫描、Shape evaluator
资格化失败以及 closeout manifest、summary 和 report。原始运行仅保留在
`outputs/20260804_capacity_map_v2_2/`；该路径不是唯一的长期引用来源。

`ambiguous_revision` 与 invalid diagnosis revision 均被保留，但不得进入
Pareto 排名或因果归因。完整可主张/禁止主张和假设状态见
[capacity-map hypothesis ledger](capacity_map_hypothesis_ledger.md)。

## 禁止主张

不得把本阶段写成光学容量、phase-only 容量、空间自由度或普适
semantic--diversity conflict 的证据。尚未唯一识别的替代原因是电子生成器容量、
优化动力学和 evaluator disagreement。

## 未执行分支与未来重启条件

O1--O5 长训练、Teacher/WGAN 上界、Controlled Shapes 和 Fashion-MNIST
确认均未执行，且不得直接恢复。若重启研究，必须另立新的电子 O0 合同和新的
teacher-free objective family，从数字 oracle 重新完成双 evaluator 多 seed
确认；只有 O0 可达后才可重新考虑物理变体。

此前的长 handoff 已逐字冻结在
[current_handoff_pre_v2_2_closeout_20260804.md](../archive/current_handoff_pre_v2_2_closeout_20260804.md)，
仅作历史参考，不定义当前执行授权。

## v2.3-r2 D0 电子图像上界：harness 无效（2026-08-06）

v2.3-r2 在 `experiment/v2-o0-feasibility-v23` 分支完成了 harness 冻结、revision-2
validate、CUDA preflight 和 controls。D0 seed-1234 probe 已按预注册门运行至 step 50，
但其 evidence 已永久降级为 `invalid_harness`。实现没有在模型构造前将全局
RNG 初始化绑定到 evidence seed，且 auxiliary classifier/Q 的梯度没有由其 optimizer
真正更新。code fraction 下降仍作为运行事实保留，但不构成科学负结果。

依据 D0 停止规则，smoke、confirm、locked-final、C0--C3、O1--O5、Teacher、diffusion
和外部数据集均未执行。不得主张 feature-matching 主导、目标冲突或 generator capacity
failure，也不得解释为光学、phase-only、空间 DOF 或普适容量失败。修正证据位于
[`experiments/archive/v2_3_r2_invalid_harness/`](../../experiments/archive/v2_3_r2_invalid_harness/)。

## v2.4 D0 training identifiability

独立的 `o0-identifiability` 生命周期只验证确定性初始化、DQ/G 更新隔离、classifier/Q
是否真实学习、恢复连续性和一次 100-step optimization-health probe；它不重开 v2.3，
不运行 C0--C3 或光学实验，且 `physical_authorization=false`。CUDA preflight 通过，
唯一 probe 已完成并有效失败为 `discriminator_identifiability_failure`：real auxiliary
classifier accuracy 从 0.10 提升到 0.26，但 Q latent error 未达到预注册的 10% 改善门。
因此 v2.4 在首个有效停止门关闭；这不是 D0 分布可达性或 generator capacity 结论。

## Objective D closeout and electronic O0 v1 (2026-08-13)

The v7 Objective D screen at
`outputs/20260813_v2/runs/20260813_183155_849434_formal_v2_identity_logit_amplitude_finite_crop_identity`
is a valid diagnostic failure, not an invalid harness. Its evidence SHA256 is
`533f35a71e2569c4691e707cec9940d0ea97de9cc721ec8820ea31da84d89594` and its
report SHA256 is
`bdb163afe3ee27cfeb7d53ba7ad32cf9e81d0cdddc2a7bad5acb212b95920ff6`.
A/D shared warmup, optimizer, schedules, finite gradients, and all six real
updates passed. D improved camera by `+0.1547` and primary diversity by
`+0.0691`, but primary coverage worsened by `+0.0562`, primary accuracy fell
by `0.10`, and primary minimum recall fell by `0.1765`. No confirmation or
formal action is authorized; the D family is closed and all D artifacts are
read-only.

A new research-only `electronic_objective_o0_v1` contract is registered on
`experiment/v2-electronic-objective-o0-v26`. It uses Objective E active-set
penalties: already-satisfied camera/diversity/coverage constraints receive
zero weight and zero gradient, while violated constraints are calibrated in
the electronic domain on steps `400--403`. O0-I is a high-capacity 28x28
electronic image oracle; O0-G is the current 8D amplitude generator followed
by an ideal amplitude-square detector. O0-G is blocked until O0-I reaches a
passed screen or locked-final state. Neither candidate can authorize optical
training, objective selection, or formal promotion.

## Objective E2 / electronic O0 v2 (2026-08-13)

The electronic O0 v1 calibration result is superseded as an invalid harness:
its raw diversity coefficient (`0.0007977245`) was below a dimensional
coefficient floor, while the implementation did not yet bind exact-zero
diversity, four-batch active-set truth, or per-group gradient budgets. It is
read-only and does not close the objective, generator, or optical branches.

The v2 contract uses exact-zero bounded diversity, deterministic disjoint
primary/companion batches, scale-invariant `0.25` gradient calibration, strict
preflight determinism, and explicit closeout admission before O0-G. Current
formal YAML remains Objective A; no optical or formal training action is
authorized by this research contract.

## Objective E2 lifecycle harness closure (2026-08-13)

The earlier E2 validation, preflight, and calibration artifacts are read-only
numeric references. They were issued under the superseded lifecycle harness
and cannot authorize a screen. The current O0-I work upgrades the contract to
classwise joint schedules, independent warmup reconstruction, per-parameter
group update gates, RNG/schedule/lineage-bound checkpoints, and official
10,000-test locked-final measurement. No E2 screen result exists yet; CUDA
execution remains stopped until fresh evidence is issued from the clean
implementation commit.

The next harness revision also repairs the full-train-set preflight reference,
bounds official evaluator measurement to 250-sample microbatches, and exposes
explicit confirmation and locked-final resume checkpoints. These changes are
implementation-only until fresh qualification is issued from their clean
commit.

## Objective E O0-I v3 screen closure (2026-08-14)

The v3 O0-I calibration, preflight, and shared-warmup screen harness were
executed under the frozen research contract. The final screen evidence is
`outputs/20260814_electronic_objective_o0_v3/runs/20260814_123854_840883_research_screen_O0-I/objective_o0_screen_evidence.json`.
Both candidates reused the same step-400 model and optimizer hashes and each
completed 584 post-warmup updates (1,168 total). The shared-warmup and
saturation checks passed, but camera, diversity, coverage, accuracy, and recall
gates failed: candidate camera fraction was `0.01675`, primary diversity ratio
`0.05564`, primary coverage `1.29147`, secondary accuracy `0.90000`, and
secondary minimum recall `0.0`. This is a valid scientific screen failure,
not an invalid harness. Confirmation, locked-final, closeout, O0-G, objective
selection, and formal optical training are stopped. The v3 artifacts remain
diagnostic-only and historical evidence remains read-only.

## Objective F/F2 invalid-harness closeout (2026-08-14)

Objective F and F2 remain read-only diagnostic observations. Their numeric
outputs are preserved in
`experiments/archive/v2_electronic_objective_f_f2_invalid_harness/`, but the
evidence is reclassified as `invalid_harness`: the records do not bind the
complete execution provenance contract (exact command, commit/dirty patch,
runtime/device, elapsed time, and all upstream audit/preflight/calibration
content hashes). The reported camera, diversity, coverage, accuracy, and
recall values therefore do not establish an F/F2 objective failure or any
electronic, optical, phase-only, or universal capacity conclusion.

The original outputs and checkpoints remain untouched at their managed paths.
The F/F2 active CLI, runtime, configs, and tests are retired; the archive is
not runnable or resumable and has `permitted_next_actions=[]`. No confirmation,
locked-final, O0-G, objective selection, or formal optical training is
authorized. The current mainline has no authorized candidate; a future branch
requires a separately pre-registered hypothesis and explicit authorization.
