# 当前主线交接：v2 current identity schedule gate（2026-08-10）

## Current identity status

The pre-schedule baseline chain was clean commit `b937509`.  The schedule-corrected
runtime and its reissued current evidence chain are bound to commit `1003767`:
evaluator qualification, initialization admission, observability, control
isolation, and semantic-localization all passed their gates.  The 100-step
controllability smoke is a valid diagnostic failure in semantic accuracy only
(`0.217647 < 0.30`); code fraction, finite gradients, six-group updates,
code-projection recovery, saturation, and provenance gates passed.  The smoke
checkpoint is diagnostic-only and must never be used as a warm start.

The current final-checkpoint schedule is bound to the production balanced
iterator: 984 steps/epoch and 4,920 steps over five epochs.  The old 5,500-step
contract and historical recovery remain predictive/reference material only.
The reissued 400-step semantic recovery is a passed diagnostic gate (accuracy
`0.435294`, classification loss `3.849→1.876`, code fraction `0.31030`,
saturation `0.05674`) and permits `train`, but no formal 5-epoch training,
evaluation, dataset export, or promotion has been run or authorized here.

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
