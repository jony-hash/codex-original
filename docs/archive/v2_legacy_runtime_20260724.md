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

v2 从 v79 研究快照旁路重写而来；两线的范围、回档点和发布状态见 [Git 分支治理与回档指南](../operations/git_branch_management.md)。v1 的版本化诊断命令与 v2 子命令不互相替代。

## 条件全光路生成主线

历史路径 `configs/v2/formal_amplitude_finite_crop_identity.yaml` 曾经显式选择
`conditional_global_amplitude_v1`。正式输入为 `label + 8D code`：电子端
generator_e 生成 `200×200` 振幅 SLM 指令，每个样本随后完整经过 Encoder →
identity OLS → Decoder → raw CMOS。该配置拒绝 teacher、teacher target、prototype
和 DCT basis 字段；历史 v2 子命令只保留为兼容路径，不能消费此配置。

训练只读取本地 MNIST train split 的固定 55,000/5,000 分层划分。raw CMOS 使用全局
曝光 `0.4913053810596466`，并通过 area pooling 得到 `28×28` 视图；不会做逐样本
归一化。官方 MNIST test split 不进入生成器训练。

```bash
python -m ldm_project.v2 student train-evaluator \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml --dataset-root data
python -m ldm_project.v2 audit conditional-observability \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml
python -m ldm_project.v2 student controllability-smoke \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml --dataset-root data \
  --evaluator-checkpoint <frozen_evaluator.pt> --observability-evidence <observability.json>
python -m ldm_project.v2 student train \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml --dataset-root data \
  --evaluator-checkpoint <frozen_evaluator.pt> --observability-evidence <observability.json> \
  --controllability-evidence <smoke.json>
```

`student evaluate` consumes conditional training evidence, checkpoint, local MNIST and the
same frozen evaluator. Its automatic gates include semantic accuracy, per-class recall, CMOS
code fraction, saturation, per-class feature-diversity ratio and Sinkhorn coverage. A passed
automatic result becomes `pending_review`, not accepted: it writes a deterministic `10×16`
fixed-exposure raw-CMOS grid and a content-bound review template. The next formal commands are:

```bash
python -m ldm_project.v2 student admit-conditional-evaluation \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml \
  --evaluation-evidence <evaluation.json> --evaluation-raw-cmos <raw.pt> \
  --evaluation-labels <labels.pt> --evaluation-preview <review.png> \
  --manual-review <completed-review.json>
python -m ldm_project.v2 student export-dataset \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml \
  --checkpoint <generator.pt> --acceptance-evidence <generation-admission.json>
python -m ldm_project.v2 student audit-memorization \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml \
  --candidate-evidence <candidate.json> --candidate-dataset <dataset-dir> \
  --dataset-root data --evaluator-checkpoint <frozen_evaluator.pt>
python -m ldm_project.v2 student evaluate-utility \
  --config configs/v2/formal_amplitude_finite_crop_identity.yaml \
  --candidate-evidence <candidate.json> --candidate-dataset <dataset-dir> --dataset-root data
```

The manual gate requires at least 14 readable samples for each of ten labels. Exported shards
are explicitly `candidate` artifacts, not accepted datasets. `audit-memorization` uses only
their stored `28×28` views and chunked frozen-evaluator nearest neighbors; `evaluate-utility`
is the only conditional command that reads the official MNIST test split. Only `student
admit-dataset`, after passed generation, memorization and utility evidence, writes the immutable
accepted manifest. Long CUDA training, the full export, manual review and utility experiment
remain user-run tasks.

## LightGen 对齐诊断（2026-07-24）

`formal_amplitude_finite_crop_identity` 没有改变，仍是 `2+2`、identity OLS、`1.50 mm` 的正式合同。此前失败只冻结已验证的局部注入族：十类 direct-template 附近的 64 个低频 DCT candidates、8 个线性 basis 与 gain `0.1`；它不能外推为 amplitude SLM 失败。

新增的 `diagnostic_lightgen_amplitude_fiber_ols_400` 是严格隔离的 architecture proxy：`400×400 @ 3 µm` amplitude seed → 2-layer encoder → 10×10 Gaussian single-mode-fiber complex OLS → 4-layer generator → raw sensor。来源与字段置信等级记录在 [LightGen reference contract](../architecture/lightgen_reference_contract_v2.json)：论文组织、100D OLS/fiber 传输为 paper-reported；400×400、3 µm、2+4 是本项目 user-locked diagnostic 参数；波长、层距、孔径和耦合校准仍是 `unknown`。因此它永远不能自称 physical-faithful，也不授予 training 或 promotion 权限。

```bash
python -m ldm_project.v2 audit lightgen-contract-gap \
  --config configs/v2/diagnostic_lightgen_amplitude_fiber_ols_400.yaml \
  --reference-contract docs/architecture/lightgen_reference_contract_v2.json
python -m ldm_project.v2 audit lightgen-ols-observability \
  --config configs/v2/diagnostic_lightgen_amplitude_fiber_ols_400.yaml \
  --reference-contract docs/architecture/lightgen_reference_contract_v2.json
```

第二条命令无 optimizer，检查 100 个复 OLS modes、有效秩 `>=8`、`sigma8/sigma1 >= 1e-3`、完整 plane/provenance 与参数零更新。只有它通过，才可另立计划讨论 diagnostic prototype calibration；本阶段不恢复 capacity 或 student training。

## 运行顺序

1. `student export-clean-targets` 只从本地 MNIST 的 train split 构造固定 `0–9`、每类 128 张平均的 `[10,1,200,200]` clean target；不下载数据，并原子写入内容寻址 manifest。
2. `teacher manifest` 固定 first-stage/diffusion pair；`teacher export-targets` 使用 v2 的 8 维、每类 100 个 export 配置，先验证 pair manifest，再原子写入 target bundle 与 target manifest。历史 128 维 bundle 不能截断后复用。
3. `audit prototype-reachability` 先以真实 finite-crop retention、类间距离 retention 与十类逐类 margin 检验冻结方向；它只生成 diagnostic evidence。formal `student calibrate-prototypes` 还必须消费该 evidence，缺失、篡改或 context 漂移会在构造 optimizer 前失败。
4. `audit prototype-topology` 是无 optimizer 的四臂冻结隔离（sqrt/trained template × zero/trained phase）；ROI 结果只定位背景污染，不能替代 full-frame raw-CMOS gate 或人工判断。
5. 当 shape-guard 已有效失败时，`audit prototype-geometry-upper-bound` 以同一冻结 template、encoder、OLS 与 backend 比较 trained 1.50 mm decoder、zero-phase 1.50 mm decoder、decoder bypass，以及独立的 zero-phase 0.25 mm decoder。它只作传播归因：任何 ROI、bypass 或 complex-control 结果都不能授权 capacity、training 或 promotion。仅当 0.25 mm 全帧自动门和用户人工审阅均通过，才可另行建立名称隔离的 diagnostic variant。
6. 若短层距上界未通过，它只会授权 `diagnostic_control_oracle`。用户可消费该内容绑定 evidence 执行 `audit prototype-control-oracle`：两臂、十类、每类 400 固定 Adam steps，固定初始化、曝光和 objective；每 40 steps 原子记录进度。中断不会发布最终 evidence。它不是训练 sweep，也不会产生下游权限。
7. amplitude-only oracle 自动通过后，`audit direct-template-replay` 会以冻结光路逐类重放内容寻址的 amplitude command；只有 bit-exact raw replay 和完整自动门同时通过，才生成待填写的 `direct_template_replay` 人审模板。`audit admit-direct-template-replay` 复核该模板和工件后，只能授权 `audit freeze-direct-template-prototype`。
8. `audit freeze-direct-template-prototype` 将已审阅的十类 command、raw CMOS 和 `command >= 0.2` support 复制为内容寻址的只读 diagnostic artifact。它不是 student checkpoint，唯一后继是 `audit direct-template-local-capacity`。
9. `audit direct-template-local-capacity` 使用固定 64 DCT candidates、8 个 SVD directions、100 codes/class、seed `1234`、gain `0.1` 和 microbatch `1` 测量冻结 command 邻域。即使通过，它也只能证明 diagnostic local upper bound，不授权 formal capacity、training、export 或 promotion。
5. `student calibrate-prototypes` 必须消费 clean-target manifest；自动 measurement/shape gate 失败会直接签发 `failed` evidence。只有自动指标通过时，才产生冻结 checkpoint、raw-CMOS replay、固定曝光预览、未填写 review 模板和 `pending_review` evidence。
4. 人工查看该 replay 后运行 `student admit-prototypes`。review 必须同时绑定 replay tensor 与固定曝光 preview SHA-256；它可记录不可读标签子集，但只有完整 `0–9` 才能通过 gate。
4. `audit capacity` 从冻结 amplitude prototype 的 `>=0.2` support 内部构造物理 `<=20 cycles/mm` DCT 候选、SVD basis 与固定 code replay；不接受外部 command/camera/support 绕过。
5. 只有 clean-commit acceptance 的 passed capacity evidence 才能启动 `student train`。训练读取一个 target manifest，而非无关联的散装 tensor。
6. `student evaluate` exact-load checkpoint 并流式重放 raw CMOS，输出待审 tensor、固定曝光 preview 和 review 模板。`student admit-evaluation` 仅复核这些内容绑定工件，不重跑光路，才形成 passed/failed acceptance。任何命令都不会自动 promotion checkpoint。

示例配置是历史路径 [formal_global_amplitude_finite_crop_identity.yaml](../../configs/v2/legacy/formal_global_amplitude_finite_crop_identity.yaml)。配置 schema 为 2，未知字段会失败；其中的 formal geometry 不允许修改。

## 重要边界

- `transparent_identity`、phase-seed、fiber/calibrated OLS 都不在 v2 正式实现中。
- v2 的显示图必须来自显式固定曝光；raw camera、loss view 和 display view 是不同类型。
- v2 正式 checkpoint 只允许完整精确加载：system/config/dataset/teacher/calibration/basis context、state key/shape/dtype 与 state content hash 都必须一致。prototype→code basis 的载入是单独、可审计的 warm-start。
- v1 `outputs/final_pt/` 只作为 teacher adapter 输入，v2 不会写入或替换它。
- `diagnostic_amplitude_finite_crop_identity` 只用于小尺寸 CPU 竖切，任何 evidence 都不能变为 acceptance；CPU 竖切和单样本 CUDA smoke 不构成正式训练、可读性、硬件或 promotion 证据。

## 用户执行的正式 CUDA 工作

本次重写没有运行长 CUDA 训练。CPU diagnostic 全链已在临时目录中验证，但它只能生成 diagnostic evidence。正式 `student preflight` 已在 clean commit `5add68b` 通过；它只测量单样本/逻辑 batch 的显存、梯度、参数不变性和 provenance，仍是 diagnostic evidence。用户仍应在自己的 clean commit 上复跑它；仅当 CUDA 内存门通过后，再依序运行：

```bash
python -m ldm_project.v2 student preflight --config configs/v2/formal_amplitude_finite_crop_identity.yaml
python -m ldm_project.v2 student export-clean-targets --config configs/v2/formal_amplitude_finite_crop_identity.yaml --dataset-root data --output <clean.pt> --manifest <clean.json>
python -m ldm_project.v2 teacher manifest --config configs/v2/formal_amplitude_finite_crop_identity.yaml --split train
python -m ldm_project.v2 teacher export-targets --config configs/v2/formal_amplitude_finite_crop_identity.yaml --legacy-train-config configs/export_teacher_distill_mnist_v2_formal_8d.yaml --teacher-manifest <pair.json> --output <targets.pt> --target-manifest <targets.json>
python -m ldm_project.v2 audit prototype-reachability --config configs/v2/formal_amplitude_finite_crop_identity.yaml --clean-target-manifest <clean.json> --failed-calibration-evidence <failed.json> --failed-prototype-checkpoint <failed.pt>
python -m ldm_project.v2 audit prototype-topology --config configs/v2/formal_amplitude_finite_crop_identity.yaml --clean-target-manifest <clean.json> --prototype-checkpoint <failed.pt>
python -m ldm_project.v2 audit prototype-geometry-upper-bound --config configs/v2/formal_amplitude_finite_crop_identity.yaml --clean-target-manifest <clean.json> --shape-evidence <shape-failed.json> --shape-checkpoint <shape-failed.pt>
# Only with a fresh geometry evidence carrying diagnostic_control_oracle:
python -m ldm_project.v2 audit prototype-control-oracle --config configs/v2/formal_amplitude_finite_crop_identity.yaml --clean-target-manifest <clean.json> --geometry-evidence <geometry.json> --shape-checkpoint <shape-failed.pt>
python -m ldm_project.v2 student calibrate-prototypes --config configs/v2/formal_amplitude_finite_crop_identity_shape_guard.yaml --clean-target-manifest <clean.json> --reachability-evidence <reachability.json>
python -m ldm_project.v2 student admit-prototypes --config configs/v2/formal_amplitude_finite_crop_identity.yaml --calibration-evidence <calibration.json> --prototype-checkpoint <prototype.pt> --prototype-raw-cmos <raw.pt> --prototype-preview <preview.png> --measurements <measurements.json> --clean-target-manifest <clean.json> --manual-review <review.json>
python -m ldm_project.v2 audit capacity --config configs/v2/formal_amplitude_finite_crop_identity.yaml --prototype-evidence <prototype.json> --prototype-checkpoint <prototype.pt>
python -m ldm_project.v2 student train --config configs/v2/formal_amplitude_finite_crop_identity.yaml --prototype-evidence <prototype.json> --capacity-evidence <capacity.json> --prototype-checkpoint <prototype.pt> --code-basis <basis.pt> --teacher-target-manifest <targets.json>
python -m ldm_project.v2 student evaluate --config configs/v2/formal_amplitude_finite_crop_identity.yaml --training-evidence <training.json> --code-basis <basis.pt> --checkpoint <student.pt> --teacher-target-manifest <targets.json> --output <pending-evaluation.json>
python -m ldm_project.v2 student admit-evaluation --config configs/v2/formal_amplitude_finite_crop_identity.yaml --evaluation-evidence <pending-evaluation.json> --evaluation-raw-cmos <raw.pt> --evaluation-preview <preview.png> --manual-review <review.json> --code-basis <basis.pt> --checkpoint <student.pt> --teacher-target-manifest <targets.json> --output <accepted-evaluation.json>
```

任何 prototype 或 capacity gate 失败都会停止后续训练、test、导出与 promotion；不得用 enhanced preview 或 software readout 绕过 raw-CMOS 证据。

当前 v2 formal prototype 状态：旧 `212454` reachability 报告以内容绑定 supersession 明确标记为 harness-invalid——其 crop 值是平面强度、front gate 是相似度比、direction gate 未逐类验证，且 calibration 未消费 evidence。`212618` 的 5×40-step calibration（corr/dice `0.8395/0.7625`、accuracy `0.50`）保留为旧 reconstruction 合同的数值负结果，但没有有效 admission lineage。2026-07-23 修正后的 reachability 已通过（最低 crop retention `0.9887`、front distance retention `0.9992`），而 frozen topology 显示 joint 最好但 topology 未解。shape-guard 仍失败：corr/dice `0.8408/0.7696`、accuracy `0.50`、minimum full-frame margin `-0.0801`，background speckle 未过门。geometry 的 0.25 mm arm 恢复 accuracy `1.0`，却仍以 margin `0.0073` 与 speckle `0.1725` 失败；它只能内容绑定地授权 bounded control oracle。正式 1.50 mm 合同冻结，不得进行 formal calibration、capacity、training 或人工补签。

2026-07-23 的 bounded control oracle 已完成：amplitude-only arm 以 corr `0.9803`、Dice `0.9402`、accuracy `1.0`、minimum margin `0.1074`、foreground hole `0.0315`、background speckle `0.0324`、saturation `0.00037` 通过完整 raw-CMOS 自动门，且参数未改变。随后 direct-template replay 已 bit-exact，人审已确认 `0–9` 全部可读。该结果证明固定 amplitude command 的可控性，不证明 code transport；下一门只能是冻结 command artifact 与一次 diagnostic local-capacity upper bound。

该 local-capacity upper bound 已在同日完成。最初一次结果因 state-hash 算法混用且覆盖数值指标而标记为 harness-invalid，原工件保留；修复后以相同内容寻址 inputs、64 DCT candidates、8 basis、100 codes/class、seed `1234`、gain `0.1` 重跑。参数不变、finite、provenance、support energy `1.0` 与 clipping `9.875e-06` 全部通过，但 physical-SLM/raw-CMOS code fraction 分别仅为 `0.00500678`/`0.00493487`，低于 `0.25`。因此结论是 `direct-template-local-capacity-not-demonstrated`：冻结 direct-template 邻域与其 basis/normalizer 家族，不进行 gain、basis、seed、phase、spacing、exposure sweep，也不进入 formal capacity、student training 或 promotion。
