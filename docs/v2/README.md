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

v2 从 v79 研究快照旁路重写而来；两线的范围、回档点和发布状态见 [Git 分支治理与回档指南](../git_branch_management.md)。v1 的版本化诊断命令与 v2 子命令不互相替代。

## 运行顺序

1. `teacher manifest` 固定 first-stage/diffusion pair；`teacher export-targets` 先验证该 manifest，再原子写入内容寻址 target bundle 与 target manifest。
2. `student calibrate-prototypes` 只产生冻结 checkpoint、raw-CMOS replay、数值/显存测量和 `pending_review` evidence；它不在同一命令内伪造人工结论。
3. 人工查看该 replay 后运行 `student admit-prototypes`。review 必须绑定 replay tensor SHA-256，且 `readable_labels` 严格为 `0–9`。
4. `audit capacity` 从冻结 amplitude prototype 的 `>=0.2` support 内部构造物理 `<=20 cycles/mm` DCT 候选、SVD basis 与固定 code replay；不接受外部 command/camera/support 绕过。
5. 只有 clean-commit acceptance 的 passed capacity evidence 才能启动 `student train`。训练读取一个 target manifest，而非无关联的散装 tensor。
6. `student evaluate` exact-load checkpoint 并重放 raw CMOS。第一次可生成待审 tensor；第二次携带与该 tensor hash 匹配的 review 才能形成 passed/failed acceptance。任何命令都不会自动 promotion checkpoint。

示例配置是 [formal_amplitude_finite_crop_identity.yaml](../../configs/v2/formal_amplitude_finite_crop_identity.yaml)。配置 schema 为 2，未知字段会失败；其中的 formal geometry 不允许修改。

## 重要边界

- `transparent_identity`、phase-seed、fiber/calibrated OLS 都不在 v2 正式实现中。
- v2 的显示图必须来自显式固定曝光；raw camera、loss view 和 display view 是不同类型。
- v2 正式 checkpoint 只允许完整精确加载：system/config/dataset/teacher/calibration/basis context、state key/shape/dtype 与 state content hash 都必须一致。prototype→code basis 的载入是单独、可审计的 warm-start。
- v1 `outputs/final_pt/` 只作为 teacher adapter 输入，v2 不会写入或替换它。
- `diagnostic_amplitude_finite_crop_identity` 只用于小尺寸 CPU 竖切，任何 evidence 都不能变为 acceptance；CPU 竖切和单样本 CUDA smoke 不构成正式训练、可读性、硬件或 promotion 证据。

## 用户执行的正式 CUDA 工作

本次重写没有运行长 CUDA 训练。CPU diagnostic 全链已在临时目录中验证，但它只能生成 diagnostic evidence。完成 clean commit 后，用户应先准备固定 clean target、人工 review、matching teacher pair manifest 和 target manifest，再依序运行：

```bash
python -m ldm_project.v2 teacher manifest --config configs/v2/formal_amplitude_finite_crop_identity.yaml --split train
python -m ldm_project.v2 teacher export-targets --config configs/v2/formal_amplitude_finite_crop_identity.yaml --legacy-train-config <v1-export-yaml> --teacher-manifest <pair.json> --output <targets.pt> --target-manifest <targets.json>
python -m ldm_project.v2 student calibrate-prototypes --config configs/v2/formal_amplitude_finite_crop_identity.yaml --clean-targets <clean.pt>
python -m ldm_project.v2 student admit-prototypes --config configs/v2/formal_amplitude_finite_crop_identity.yaml --calibration-evidence <calibration.json> --prototype-checkpoint <prototype.pt> --prototype-raw-cmos <raw.pt> --measurements <measurements.json> --clean-targets <clean.pt> --manual-review <review.json>
python -m ldm_project.v2 audit capacity --config configs/v2/formal_amplitude_finite_crop_identity.yaml --prototype-evidence <prototype.json> --prototype-checkpoint <prototype.pt>
python -m ldm_project.v2 student train --config configs/v2/formal_amplitude_finite_crop_identity.yaml --prototype-evidence <prototype.json> --capacity-evidence <capacity.json> --prototype-checkpoint <prototype.pt> --code-basis <basis.pt> --teacher-target-manifest <targets.json>
```

任何 prototype 或 capacity gate 失败都会停止后续训练、test、导出与 promotion；不得用 enhanced preview 或 software readout 绕过 raw-CMOS 证据。
