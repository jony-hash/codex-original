# 当前主线交接：v2 current identity evaluation closeout（2026-08-12）

## Current identity status

The current evidence-lineage policy is now component-scoped.  New evidence
records carry the registry version, the applicable runtime components, and a
Git-tree fingerprint for each component; consumers still require clean
ancestor commits and independently verify config, split, checkpoint, and
artifact hashes.  Evidence created before this registry was introduced is
fail-closed and must be reissued once on the migrated clean baseline.  A
change to an unrelated component no longer invalidates every current record,
while unknown evidence kinds conservatively retain the full-current invalidation
scope.

The evaluation schedule harness was repaired at clean commit `ae3273d`. The
earlier evaluation attempt mixed the 55,000-sample training split with the
60,000-sample full label set and was therefore an `invalid_harness` attempt,
not a model failure.

The runtime-equivalent evidence-lineage guard was implemented at clean commit
`4aac9c0`. The reissued current evidence chain was then regenerated on
`ae3273d`; its upstream qualification, initialization, observability, control
isolation, localization, controllability failure, and semantic recovery remain
diagnostic lineage only.
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
Maintenance governance is now the next permitted work; any new objective or
scientific diagnostic must be a separate research line.

## Objective diagnostic closeout (2026-08-12)

The v3 calibration run
`outputs/20260812_v2/runs/20260812_183756_552830_formal_v2_identity_logit_amplitude_finite_crop_identity`
was reclassified as `invalid_harness`: its B/C combined metric included the
base gradient while applying an auxiliary-only budget, and its aggregate used
an arithmetic mean instead of the RMS contract. It did not establish an
objective failure. The correction was committed at `f0861e6`, with the
parameter-index and evidence-gate follow-ups at `8bbf78d` and `ab620be`.

The corrected calibration
`outputs/20260812_v2/runs/20260812_185926_470787_formal_v2_identity_logit_amplitude_finite_crop_identity`
passed its four-batch effective-gradient gates: single auxiliary ratios were
approximately `0.25`, B was `0.3424`, and C was `0.4162`; it authorized only
`screen`. Screen then ended at
`outputs/20260812_v2/runs/20260812_191311_132000_formal_v2_identity_logit_amplitude_finite_crop_identity`
with valid `conditional_objective_alignment_screen / diagnostic / failed`
evidence. B and C both failed the registered camera/diversity/coverage screen
gates; all six parameter groups updated and all candidates shared the same
schedule. No winner, confirmation, objective promotion, or formal training was
authorized. This is a diagnostic objective failure, not a model-wide optical,
hardware, or execution-invalid claim.

The remainder of this file records immutable historical closures and is retained
for context; those records must not be rewritten or used as current acceptance.

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
