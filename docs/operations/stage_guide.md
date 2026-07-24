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

## 通用调试规则

- 先验证输入 checkpoint、config、数据和光学 provenance，再启动 optimizer。
- 有效 gate 失败时停止下游训练、导出和 promotion；invalid harness 必须修复后重做。
- CPU smoke 只证明执行链，不能替代 CUDA、raw-CMOS、人工可读性或加工验收。
- 历史工具和配置在 `experiments/archive/` 或 `docs/archive/` 中只作参考，不能从归档路径
  直接作为活动 CLI。
