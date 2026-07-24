# 当前主线交接：v79 冻结与 v2 旁路（2026-07-23）

本文件只保存当前状态、停止规则和下一允许动作。v68–v79 的旧交接正文已完整、只读保存在 [v68–v79 原始主线交接归档](current_handoff_v68_v79_archive.md)，按阶段查阅的摘要、指标和工件索引见 [实验与验收历史](../status/experiment_history.md)；稳定架构合同见 [D2NN 单次曝光蒸馏主线](../architecture/v1_single_shot_distillation.md)；分支与回档点见 [Git 分支治理指南](../operations/git_branch_management.md)。

## 当前正式合同

```text
laser -> amplitude SLM -> coherent 4f relay -> encoder D2NN (2 layers)
      -> complex identity OLS -> decoder D2NN (2 layers)
      -> finite-conjugate coherent relay -> raw CMOS intensity
```

- `lambda=532 nm`；无 fiber；OLS 为严格复线性恒等映射。
- SLM/CMOS 为 `200×200 @ 2.5 µm`（`500 µm` FOV）。encoder/decoder 各为 `100×100 @ 2 µm`（`200 µm` FOV）、2 层 phase-only、层间距 `1.50 mm`。
- 前端为完整 FOV 的 `M=-0.4` 相干 4f relay；后端为 `M=-2.5` 的有限共轭 relay。正式 D2NN 边界为 propagation-after-crop 的 `finite_crop`，不是材料不透明性声明。
- raw CMOS、loss view 和 display view 不能混用；最终可读性只看锁定曝光的 raw CMOS，不接受增强预览或软件读出替代。

## v1 研究线的冻结结论

v68 A2 曾通过 prototype raw/readability 门，v68.6 B 与 v69 也保留了可读 prototype；但它们的 raw-camera code fraction 分别为 `7.14e-08` 和 `0.0002712`，远低于 `0.25`。v70–v77 将衰减定位为冻结 D2NN 的 phase-to-intensity/readout 权衡，未产生可部署的训练授权。

v79 以固定 label-only normalization 对八维 supported basis 做 CPU 上界：frozen replay 的 physical-SLM code factor 为 `6.03e-7`，deterministic capacity arm 为 `0.05455 < 0.25`，且 clipping `0.20547 > 0.08`。有效结论为 `label-fixed-supported-basis-insufficient`：该 basis 家族不足以证明 code transport；learned code transport 仍未证明，且不存在可 promotion 的 student checkpoint。

因此冻结 v1 semantic-supported basis、residual＋phase 联合训练、normalization/gain/amplitude/basis sweep、test、导出和 Formal promotion。不得把 prototype readability、teacher-camera 相似度或离线 software readout 表述为最终单次曝光生成能力。

## v2 旁路状态

`rewrite/v2` 从 v79 快照 `e21dba1` 线性分出，当前入口为 `python -m ldm_project.v2`；它不替换 `train.py` 或 v1 默认行为。v2 以类型化 physical field、严格配置/manifest、独立光学 reference、prototype/capacity evidence gate 和受约束 code basis 重写正式主链。其执行链已固定为 teacher pair/target manifest → calibration measurement → content-bound prototype review/admission → frozen capacity → guarded training → raw-CMOS evaluate/review。

已完成代码和 CPU diagnostic 全链竖切；该竖切使用 fake teacher pair、内部 physical-DCT capacity 和单步训练，所有输出强制为 diagnostic。v2 已具备 8 维/每类 100 个 teacher export 合同、本地 128-per-label clean target 导出、流式 CUDA 执行、固定曝光 review 工件及无更新 `student preflight`。正式 8D teacher bundle 已完成内容审计。首次 formal prototype calibration 在 `d7958bc` 上有效失败：统一 `0.5` amplitude template、零 phase、5 个实际 optimizer step 的 raw corr/dice/accuracy 为 `0.34985/0.49562/0.10`。随后 `212454` reachability 报告被内容绑定 supersession 明确标记为 harness-invalid，原文件保持不变：它把平面平均强度误当 crop retention、用相似度比替代类间距离，并未逐类验证 margin，且 calibration 未消费该 evidence。`212618` 的 5×40-step run（corr/dice `0.8395/0.7625`、accuracy `0.50`）保留为旧 reconstruction 合同的数值负结果，但无有效 reachability admission lineage。2026-07-23 修正后的 reachability 已通过（最低 crop retention `0.9887`、front class-distance retention `0.9992`），frozen 2×2 topology isolation 的 joint arm 最好但仍是 `joint-separation-improving-but-topology-unresolved`。shape-guard 的 corr/dice 为 `0.8408/0.7696`，但 accuracy `0.50`、minimum full-frame margin `-0.0801`、background speckle 均未过自动门。geometry 进一步确认 0.25 mm arm 虽 accuracy `1.0`，仍以 margin `0.0073 < 0.03` 与 speckle `0.1725 > 0.07` 失败；它不允许 short-spacing variant，只能明确授权一次内容绑定的 `diagnostic_control_oracle`。正式 shape-guard 合同继续冻结，不得人工补签、capacity、student training 或 promotion。

## 下一允许动作

v2 的正式执行顺序固定为：clean commit 上的 `student preflight` → local clean-target manifest → teacher pair manifest → 8D atomic target export/target manifest → `student calibrate-prototypes` → `student admit-prototypes` → `audit capacity` → `student train` → `student evaluate` → `student admit-evaluation`。prototype calibration 的自动 corr/dice/accuracy/saturation/finite/gradient/memory/epoch gate 任一失败会直接停止，不进入人工审阅；CPU diagnostic evidence 只能显式授权 diagnostic smoke，不能跨入 formal 链。preflight 通过只授权用户进入 prototype gate，不构成 capacity、科学或 promotion acceptance。

正式 1.50 mm 主线的下一允许动作不是继续调 loss、epoch、LR、ROI、phase 或 exposure。two-arm control oracle 已证明 amplitude-only command 可控制冻结光路；其 amplitude arm 与随后 direct-template replay 均通过完整 raw-CMOS 自动门，且用户已对 `0–9` 完成内容绑定人工审阅。该链只证明十类固定 command 的可控性，不证明 8D code transport；complex arm不推进硬件变体。

direct-template command 已冻结并完成唯一一次 `audit direct-template-local-capacity`。首次运行暴露 state-hash 算法混用和数值报告覆盖，保留为 harness-invalid；修复后在 clean commit 上按完全相同的内容寻址 inputs 与固定 64 个 `<=20 cycles/mm` DCT candidates、8 basis、100 codes/class、seed 1234、gain 0.1 重跑。参数不变、finite、provenance、support energy `1.0` 与 clipping `9.875e-06` 都通过，但 physical-SLM/raw-CMOS code fraction 只有 `0.00500678`/`0.00493487`，低于 `0.25`。有效结论为 `direct-template-local-capacity-not-demonstrated`：冻结 direct-template 邻域与此 basis/normalizer 家族；不进行 gain、basis、seed、phase、spacing 或 exposure sweep，不进入 formal capacity、student training、export 或 promotion。

2026-07-24 的唯一新诊断主线是 LightGen 对齐审计，而不是修改上述正式合同：`diagnostic_lightgen_amplitude_fiber_ols_400` 使用 amplitude seed、2-layer encoder、10×10 Gaussian single-mode-fiber complex OLS 和 4-layer generator。其 [内容寻址 reference contract](../architecture/lightgen_reference_contract_v2.json) 明确保留波长、层距、孔径及真实 fiber coupling 为 `unknown`，所以输出只能是 `architecture-proxy` diagnostic evidence。先运行 contract-gap 与无 optimizer observability；若 100 modes、rank、奇异值比、provenance 或参数不变性任何一项失败，停止，不训练。即使通过，也仅可另立 diagnostic prototype 计划，绝不授权 formal capacity、student training 或 promotion。

## Identity-Anchored 条件生成主线（2026-07-24，恢复中）

在现有 prototype/teacher 旁路均未产生可 promotion checkpoint 的前提下，v2 下一条正式候选主线切换为 `conditional_identity_logit_amplitude_v2`。该切换只改变 generator-e 的随机码注入合同，不改变正式 `amplitude SLM → encoder → identity OLS → decoder → raw CMOS` 光路、532 nm、2+2 phase layers、1.50 mm layer spacing、relay、`finite_crop` 或固定曝光。旧 `conditional_global_amplitude_v1` checkpoint、旧 controllability evidence 以及 teacher/prototype/capacity evidence 均为历史工件，禁止静默迁移。

候选 generator 的锁定公式为：

```text
h(z) = SiLU(Wz · z)
r(z,y) = Wy · h(z)
c(z,y) = tanh(z + r(z,y))
A(z,y) = sigmoid(By + 0.5 × Σk ck(z,y) P̂k)
```

其中 `Wy` 零初始化，故初始状态为 `c=tanh(z)`，`z=0` 严格复现类别载波；固定增益为 `0.5`，不以 gain、loss、LR、phase、exposure 或 basis sweep 代替准入。正式阶段必须从同一个、seed `1234` 生成且内容哈希绑定的 `conditional_initialization.pt` 加载状态，并将 system/config、MNIST split、冻结 evaluator、曝光、metric contract 和 initialization hash 绑定在每份 evidence 中。任何上下文漂移、缺失字段或旧 architecture kind 都必须显式拒绝。

### Identity 主线的停止规则

正式顺序固定为：

```text
generator-only injection admission
→ no-update optical observability
→ no-update control isolation
→ one 100-step controllability smoke
→ user-run 5-epoch joint training
→ automatic evaluation / manual review / candidate export
```

- generator-only 注入门未同时证明 command 与物理 `A²` code fraction ≥ `0.10`、八个方向 RMS ≥ `0.01`、rank=8、零码语义、有限/正交模式、确定性 state hash 和参数不变时，立即停止，不运行光路或 optimizer。
- isolation 将首个 code fraction 低于 `0.10` 的平面分类为 `generator_injection_insufficient` 或 `optical_transport_attenuated`；raw CMOS 已通过但语义准确率仍低于 `0.30` 时分类为 `semantic_readout_insufficient`。上述三类结果均不得授权 smoke 或正式训练。
- 只有分类为 `controllable_upper_bound` 的 isolation evidence 才能授权唯一一次 100-step smoke。smoke 必须从同一个 immutable initialization checkpoint 开始，并记录 step 0/10/100 的固定 probe、loss、梯度、更新、preview 和最终 state；该 state 不得作为正式训练 warm-start。
- smoke 任一 accuracy、corrected code fraction、饱和率、梯度/更新或 context gate 失败，停止，不重复训练、不导出 60,000 张、不做 utility/test promotion。只有 smoke 通过后，用户才可从同一初始化 checkpoint 运行完整 5 epoch 主线。

当前 implementation 尚未产生可 promotion 的 identity checkpoint；在 identity 注入门和上述 evidence lineage 完成前，任何 60,000 张导出、人工准入、memorization/utility 评估都属于越级操作，必须保持禁止。
