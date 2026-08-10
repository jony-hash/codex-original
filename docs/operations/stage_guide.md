# Stage Guide

本文件只定义 v1 `train.py --stage` 运行方式和通用实验调试政策。v2 当前命令和
停止规则见 [v2 runtime](../architecture/v2_runtime.md) 与
[current handoff](../status/current_handoff.md)。

## v1 入口

```bash
python train.py --train-config configs/v1/formal/train_ldm.yaml \
  --stage teacher_first_stage --device cuda
python train.py --train-config configs/v1/formal/train_ldm.yaml \
  --stage teacher_diffusion --device cuda \
  --first-stage-checkpoint <matched_checkpoint>
python train.py --train-config configs/v1/teacher/export_teacher_distill_mnist_3k_fast200.yaml \
  --stage teacher_distill_export --device cuda \
  --first-stage-checkpoint <first_stage.pt> \
  --diffusion-checkpoint <diffusion.pt>
```

v1 one-shot 训练、采样和评估只能使用 `configs/v1/**`。v1 teacher 输出、student
checkpoint 和历史 v68–v79 证据不能被解释为 v2 teacher-free acceptance。

## 变更级反向验证

任何修改先完成与风险相称的反向验证，再进入实施。任务或 PR 说明应至少写明
修改目的、受影响范围、不可破坏的不变量、验证方法和失败条件；不要求为每次修改
新增版本化计划文件。

- 低风险：文档、注释和格式修改。确认不改变运行时行为、合同或证据语义，并执行
  适用的文档/格式检查（至少包括 `git diff --check`）。
- 中风险：局部代码和非合同配置。先复现问题或建立会失败的回归用例，检查调用方
  与影响范围，并在实施前选定定向测试。
- 高风险：公共 CLI、运行时边界、训练/评估、光学参数、SLM/OLS/CMOS、
  checkpoint/output 政策或科学结论。实施前必须完成 no-update preflight、独立
  reference 或负对照，写明验收门和 fail-closed 停止条件；任一有效 gate 失败时，
  不得推进训练、promotion 或物理结论。

这套变更级检查补充现有 CUDA、硬件、raw-CMOS 和正式 acceptance 门，不能以低风险
检查或 CPU smoke 替代高风险验收。无法建立充分验证时，必须记录 blocker 及失去的
保证，不得静默降级为“已验证”。

## 通用调试规则

- 先验证输入 checkpoint、config、数据和光学 provenance，再启动 optimizer。
- 有效 gate 失败时停止下游训练、导出和 promotion；invalid harness 必须修复后重做。
- CPU smoke 只证明执行链，不能替代 CUDA、raw-CMOS、人工可读性或加工验收。
- 历史工具和配置在 `experiments/archive/` 或 `docs/archive/` 中只作参考，不能从归档路径
  直接作为活动 CLI。
