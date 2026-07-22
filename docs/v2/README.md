# v2 Contract-First Runtime

v2 是与 v1 并行的重写运行时，入口为：

```bash
python -m ldm_project.v2 --help
```

它不替换 `train.py`，也不改变 v1 的现行实验结论。v2 的唯一正式系统是：

```text
code + optional label -> amplitude command -> complete 4f relay
-> 2-layer finite-crop encoder -> complex identity OLS
-> 2-layer finite-crop decoder -> B=0 backend relay -> raw CMOS intensity
```

## 运行顺序

1. 使用 `teacher manifest` 将配套 first-stage/diffusion checkpoint 固定为内容寻址 manifest。
2. 运行 `student calibrate-prototypes`。它只能写 diagnostic evidence，直到固定曝光、十类人工 review、数值/梯度/显存门都通过。
3. 运行 `audit capacity`。它必须读取已通过的 prototype evidence，并写入 8 维、label-only normalized DCT/SVD basis artifact。
4. 仅当 capacity evidence 是 clean-commit 的 acceptance `passed` 状态时，`student train` 才允许启动。
5. `student sample` 和 `student evaluate` 只产生 raw-CMOS 结果；它们不会 promotion checkpoint。

示例配置是 [formal_amplitude_finite_crop_identity.yaml](../../configs/v2/formal_amplitude_finite_crop_identity.yaml)。配置 schema 为 2，未知字段会失败；其中的 formal geometry 不允许修改。

## 重要边界

- `transparent_identity`、phase-seed、fiber/calibrated OLS 都不在 v2 正式实现中。
- v2 的显示图必须来自显式固定曝光；raw camera、loss view 和 display view 是不同类型。
- v2 正式 checkpoint 只允许完整精确加载。prototype→code basis 的载入是单独、可审计的 warm-start。
- v1 `outputs/final_pt/` 只作为 teacher adapter 输入，v2 不会写入或替换它。
- CPU 竖切和单样本 CUDA smoke 仅证明执行与数值路径，不构成正式训练、可读性、硬件或 promotion 证据。

## 用户执行的正式 CUDA 工作

本次重写没有运行长 CUDA 训练。完成 clean commit 后，用户应先准备固定的 clean-target、人工 review、匹配 teacher manifest 和数据 hash，再依序运行：

```bash
python -m ldm_project.v2 student calibrate-prototypes --config configs/v2/formal_amplitude_finite_crop_identity.yaml ...
python -m ldm_project.v2 audit capacity --config configs/v2/formal_amplitude_finite_crop_identity.yaml ...
python -m ldm_project.v2 student train --config configs/v2/formal_amplitude_finite_crop_identity.yaml ...
```

任何 prototype 或 capacity gate 失败都会停止后续训练、test、导出与 promotion；不得用 enhanced preview 或 software readout 绕过 raw-CMOS 证据。
