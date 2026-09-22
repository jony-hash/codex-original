# 单次曝光光学生成项目介绍

> 文档类型：产品经理初始化说明
> 快照日期：2026-08-29
> 更新方式：每次项目主线、训练合同、验收状态或迁移目标变化时，更新“最后更新日期”和对应章节；历史事实保留原日期，不用最新状态覆盖。

## 1. 一句话定位

本项目是一个单次曝光光学生成模型研究与验证平台：电子生成器将“类别条件 + 低维随机编码”转换为振幅 SLM 指令，随后由相位衍射光学系统传播，并通过 CMOS 一次曝光输出图像。

它不是传统的纯软件扩散模型，也不是当前可直接商用的通用文生图产品。当前数据内容为 MNIST，主要用于验证光学生成、物理约束、单次曝光推理和硬件迁移可行性。

## 2. 项目要解决的问题

- 用光学传播承担部分图像生成计算，探索低延迟、单次曝光的生成路径。
- 让电子输入、SLM、衍射超表面、OLS 和 CMOS 形成可追溯的数字孪生链路。
- 验证随机 code 是否真的能带来同类图像多样性，而不是只生成类别模板。
- 在进入真实硬件前，通过严格配置、checkpoint、数据 split 和 evidence 检查减少误判。
- 为后续迁移到字母、图标、纹理或工业图案建立可复用的光学与验收框架。

## 3. 当前正式生成链路

```text
label + 8D code
    ↓
identity-anchored electronic amplitude generator
    ↓
amplitude SLM command
    ↓
coherent 4f relay
    ↓
2-layer finite-crop phase-only encoder
    ↓
complex identity OLS
    ↓
2-layer finite-crop phase-only decoder
    ↓
B=0 backend relay
    ↓
raw CMOS intensity
    ↓
generated image
```

当前正式物理合同包括：532 nm 波长；200×200 SLM；100×100、2 µm 超表面；Encoder/Decoder 各 2 层；层间距 1.50 mm；finite-crop 传播；理想复数 Identity OLS；CMOS 作为最终学生输出。

需要特别区分：

- SLM 直接控制的是振幅，不应当描述成 phase-only 或 complex-amplitude SLM。
- OLS 当前是复场恒等映射，不等于已实现的真实光纤或标定矩阵。
- 冻结 evaluator 只用于训练和验收，不属于最终光学推理路径。
- 最终推理不运行 diffusion denoising、软件 VAE 解码、迭代 latent sampling 或 inference-time optimization。

## 4. 历史演进

| 日期 | 阶段 | 关键变化 | 产品含义 |
|---|---|---|---|
| 2026-06-22 | v1 单曝光蒸馏主线 | LDM teacher 生成目标，再训练 D2NN one-shot student。 | 建立最初的“软件 teacher → 光学 student”路线。 |
| 2026-07-14～15 | 物理前向合同收紧 | 引入完整 SLM FOV、相干 4f relay、2+2 phase-only D2NN、CMOS 输出。 | 从算法演示转向可审计的物理链路。 |
| 2026-07-21 | v79 容量路线关闭 | 固定 8D supported-basis 未通过 code-capacity 与 clipping 门。 | 说明该编码基族不能直接支撑正式生成。 |
| 2026-07-22 | v2 建立 | 引入类型化光场、严格 checkpoint/evidence 和独立 CLI。 | v1 负责复现，v2 成为新的架构边界。 |
| 2026-07-24 | teacher-free identity 主线 | 使用 `label + 8D code` 的 identity-anchored 振幅生成器，不依赖 LDM teacher。 | 从 teacher 蒸馏转向条件生成器直接学习。 |
| 2026-07～08 | 研究路线扩展 | 验证 multi-anchor、PCA transport、fiber、WAE/O0、quadrature 等候选。 | 各路线独立记录，失败不能反向授权正式主线。 |
| 2026-08-12 | v2 正式训练验收 | 5 epoch、4,920 step 正式训练通过；raw-CMOS 验收因多样性/覆盖率不足失败。 | 工程训练链可运行，但模型未达到推广条件。 |
| 2026-08-27 | H-LOMA-I0 重建研究 | 完整 2+2 Identity 光路的限定重建实验通过。 | 证明受限光学重建可学习，但不等于条件生成或硬件就绪。 |
| 2026-08-29 | 本文档快照 | 汇总当前产品定位、主线、历史与迁移方法。 | 作为后续移植和项目交接入口。 |

## 5. 两条运行时的关系

### v1：保留的 LDM-teacher 蒸馏路线

```text
teacher_first_stage
→ teacher_diffusion
→ teacher_distill_export
→ one_shot_distill
→ one_shot_sample
→ one_shot_distill_eval
```

v1 的 teacher、VAE、diffusion 和 distillation target 只用于训练基础设施和历史复现，不属于最终硬件推理。

### v2：当前 teacher-free 条件生成路线

v2 不替换 v1 的 `train.py`，入口为：

```bash
python -m ldm_project.v2 --help
```

v2 使用独立的 current、diagnostic、research 和 legacy 边界。旧 prototype、capacity、fiber 和 LightGen 路线只用于历史解释或研究，不能自动签发 current acceptance。

## 6. 训练方式

### v1 训练顺序

```bash
python train.py --train-config configs/v1/formal/train_ldm.yaml \
  --stage teacher_first_stage --device cuda

python train.py --train-config configs/v1/formal/train_ldm.yaml \
  --stage teacher_diffusion --device cuda \
  --first-stage-checkpoint <MATCHED_FIRST_STAGE.pt>

python train.py --train-config configs/v1/teacher/export_teacher_distill_mnist_3k_fast200.yaml \
  --stage teacher_distill_export --device cuda \
  --first-stage-checkpoint <MATCHED_FIRST_STAGE.pt> \
  --diffusion-checkpoint <MATCHED_DIFFUSION.pt>

python train.py --train-config configs/v1/formal/train_v56_coherent_relay_from_scratch.yaml \
  --stage one_shot_distill --device cuda \
  --distill-dataset <TEACHER_TARGETS.pt>
```

first-stage checkpoint 和 diffusion checkpoint 必须来自同一 teacher 链路；更换 first-stage 后必须重新训练 diffusion。

### v2 训练顺序

v2 正式训练必须先完成资格化和证据链：

```text
primary evaluator qualification
→ secondary evaluator training
→ acceptance measurement qualification
→ generator injection admission
→ optical observability
→ control isolation
→ 100-step controllability smoke 或 400-step semantic recovery smoke
→ 三步真实 CUDA preflight
→ 5 epoch joint training
→ 双 evaluator raw-CMOS evaluation
```

接口示例：

```bash
python -m ldm_project.v2 student train \
  --config configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml \
  --dataset-root <MNIST_ROOT> \
  --evaluator-checkpoint <PRIMARY_EVALUATOR.pt> \
  --initialization-checkpoint <INITIALIZATION.pt> \
  --injection-evidence <INJECTION.json> \
  --observability-evidence <OBSERVABILITY.json> \
  --semantic-recovery-evidence <PASSED_RECOVERY.json>
```

也可以用通过的 `--controllability-evidence` 替代 `--semantic-recovery-evidence`。当前正式 identity 链已经在 raw-CMOS 验收处关闭，因此上述命令是可复用的接口模板，不代表 2026-08-29 可以无条件重新启动正式训练。

训练配置当前为 5 epoch、每 epoch 984 step、总计 4,920 step，使用 `balanced_min_class_exhaustion` 调度。训练 evidence、初始化 checkpoint、配置和数据 split 必须相互绑定；输入不匹配时应 fail-closed。

## 7. 推理和图像生成

推理输入是两个 Tensor 文件：

- `codes.pt`：形状通常为 `[N, 8]` 的随机编码；
- `labels.pt`：形状通常为 `[N]` 的整数类别标签。

示例：

```python
import torch

g = torch.Generator().manual_seed(1234)
torch.save(torch.randn(10, 8, generator=g), "codes.pt")
torch.save(torch.arange(10, dtype=torch.long), "labels.pt")
```

执行 current v2 单次前向：

```bash
python -m ldm_project.v2 student sample \
  --config configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml \
  --checkpoint <EXACT_IDENTITY_CHECKPOINT.pt> \
  --codes codes.pt \
  --labels labels.pt \
  --output raw_cmos.pt
```

`raw_cmos.pt` 是原始 CMOS 强度张量，是结果源。PNG 预览只用于人工查看，不能替代 raw tensor、物理 provenance 或正式验收。

正式评估使用：

```bash
python -m ldm_project.v2 student evaluate \
  --config configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml \
  --training-evidence <TRAINING_EVIDENCE.json> \
  --checkpoint <EXACT_IDENTITY_CHECKPOINT.pt> \
  --dataset-root <MNIST_ROOT> \
  --evaluator-checkpoint <PRIMARY_EVALUATOR.pt> \
  --acceptance-evaluator-checkpoint <SECONDARY_EVALUATOR.pt> \
  --acceptance-measurement-bundle <MEASUREMENT_BUNDLE> \
  --acceptance-measurement-evidence <MEASUREMENT_EVIDENCE.json> \
  --official-test-exposure-evidence <EXPOSURE_EVIDENCE.json>
```

## 8. 当前产品状态

截至本快照：

- v2 identity formal training checkpoint 已产生；
- formal raw-CMOS evaluation 是完整、有效的模型验收失败，不是 harness 失败；
- 失败主要涉及同类多样性和覆盖率，不能外推为所有 phase-only、光学或硬件方案都不可行；
- 没有 accepted dataset、formal promotion 或 hardware-ready 结论；
- H-LOMA-I0 的 bounded reconstruction research 通过，只能支持“光学重建训练合同可行”的有限结论；
- 研究 checkpoint 不得直接作为 current identity 的正式 warm start。

项目管理状态应写为：

```text
Research platform / diagnostic maturity
Formal promotion: false
Accepted dataset: none
Production readiness: not established
```

## 9. 迁移到其他内容的初始化清单

迁移到字母、图标、纹理或工业图案时，建议新建独立配置和架构名称，不直接覆盖 MNIST current 配置：

1. 替换数据集，并固定 train/validation/official-test split。
2. 重新定义 label、条件语义和 code 的含义。
3. 重新设计内容到 SLM 的电子 carrier 映射，检查振幅范围和饱和。
4. 为新内容训练 primary/secondary evaluator。
5. 重新定义可读性、类别召回、多样性、覆盖率和记忆化阈值。
6. 先做 generator-only capacity admission，再做 optical observability。
7. 完成三步 CUDA 更新、负对照和中断恢复 preflight。
8. 用 raw CMOS 输出进行正式评估，不能只看软件 preview。
9. 新内容的 checkpoint、manifest、evidence 和 output 使用新的版本目录。
10. 只有通过 generation admission、memorization 和 utility，才可生成 accepted manifest。

保持不变的底座通常包括：复场接口、物理坐标与 provenance、SLM→CMOS 单次曝光语义、phase-only D2NN 约束和 fail-closed evidence 消费规则。若修改 SLM control、超表面可训练性、OLS、CMOS 标定或光学几何，应视为新硬件/新正式合同，而不是普通数据集迁移。

## 10. 维护记录

| 日期 | 更新内容 | 维护人/来源 |
|---|---|---|
| 2026-08-29 | 创建项目经理初始化介绍，汇总 v1/v2、生成、训练、推理、历史和迁移清单。 | 依据仓库架构、配置、CLI 和状态文档 |

后续更新时，优先核对：

- [v2 Runtime](docs/architecture/v2_runtime.md)
- [v1 Architecture](docs/architecture/v1_single_shot_distillation.md)
- [Current Handoff](docs/status/current_handoff.md)
- [Stage Guide](docs/operations/stage_guide.md)
- [Output Artifact Notes](docs/operations/output_artifact_notes.md)
