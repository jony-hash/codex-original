# 当前主线交接：v2.2 条件性负结果已封存（2026-08-04）

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

## v2.3 电子数字上界分支（已实现，尚未训练）

v2.3 在 `experiment/v2-o0-feasibility-v23` 分支实现为独立 research 生命周期；其
基线仍是 v2.2 closeout tag，不消费 v2.2 checkpoint，也不改变 formal current。已完成
`validate`/`controls`、D0/C0--C3 合同、四面板双 evaluator 测量、microbatch 训练状态、
checkpoint/evidence 绑定和顺序停止门。CPU controls 已通过真实集、标签置换和模板重复控制。

本分支尚未启动 probe、smoke、confirm 或 locked-final CUDA 训练，因此尚无 v2.3 生成器
可达性结论。下一条命令必须先使用通过 controls 的 evidence 执行 D0 probe；D0 未通过时
不得启动 C0--C3。只有 C0/C1/C2 locked-final 通过，才可另立 O1/O2 计划；C3 永远不
授权物理推进。完整接口和配置见
`configs/research/conditional_o0_feasibility_v2_3.yaml` 及
`python -m ldm_project.v2 research o0-feasibility --help`。
