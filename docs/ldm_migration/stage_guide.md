# Stage Guide

本文件是选择和运行 stage 的操作权威来源。架构与最终推理边界见 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)，带日期的结果见 [实验与验收历史](experiment_history.md)，完整文档导航见 [Documentation](../README.md)。当前 v57 状态与停止规则见 [当前主线交接](current_handoff.md)；该 handoff 是时效性状态快照，不是架构合同，下一决策门后必须刷新。

## Golden Path

| Stage | 用途 | 何时运行 | 必需输入 | 可作为最终单次曝光证据 |
| --- | --- | --- | --- | --- |
| `teacher_distill_export` | 从 matched teacher 导出蒸馏 targets | 准备或更新 teacher dataset 时 | `--first-stage-checkpoint`、`--diffusion-checkpoint` | 否，只生成训练目标 |
| `one_shot_distill` | 训练 D2NN one-shot student | 有可用 distill dataset 后 | `--distill-dataset` | 否，只是训练过程 |
| `one_shot_sample` | 加载 one-shot checkpoint 生成样本和诊断 | 检查学生采样输出 | `--one-shot-checkpoint` | 是，若未依赖 diffusion/VAE software decode |
| `one_shot_distill_eval` | 对 teacher target 与 student camera 做指标评估 | 正式比较 checkpoint | `--distill-dataset`、`--one-shot-checkpoint` | 是，若 sample/eval 均达标 |

正式证据链固定为：

```text
teacher_first_stage -> teacher_diffusion -> teacher_distill_export
-> one_shot_distill -> one_shot_sample -> one_shot_distill_eval
```

## End-To-End Sequence

先激活环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models
```

训练或选择配套 teacher：

```bash
python train.py --train-config configs/train_ldm.yaml \
  --stage teacher_first_stage --device cuda

python train.py --train-config configs/train_ldm.yaml \
  --stage teacher_diffusion --device cuda \
  --first-stage-checkpoint <matched_first_stage_checkpoint>
```

导出 teacher 蒸馏目标：

```bash
python train.py --train-config configs/export_teacher_distill_mnist_3k_fast200.yaml \
  --stage teacher_distill_export --device cuda \
  --first-stage-checkpoint <matched_first_stage_checkpoint> \
  --diffusion-checkpoint <matched_diffusion_checkpoint>
```

训练学生（仅在已批准的 physical-forward config 和 preflight 通过后执行）：

```bash
python train.py --train-config <approved_physical_forward_config.yaml> \
  --stage one_shot_distill --device cuda \
  --distill-dataset <distill_dataset>
```

正式 config 必须以 [架构文档](single_shot_ldm_distillation.md) 为合同来源，至少锁定 identity OLS、完整 SLM-to-encoder 相干 4f relay、`100 × 100 @ 2 µm`、`1.50 mm` D2NN spacing、严格 physical-field provenance，并先通过对应 CUDA preflight。正式 stage 由 `FormalOneShotOpticalGenerator` 构造；包含 bypass、phase-amplitude、generic resize、target-code bridge 或 SLM override 的配置只能走 compatibility/diagnostic facade。当前 v56 已完成一次有效但人眼失败的训练；在 v58 决策门之前不提供新的训练命令。

采样并评估同一个 student checkpoint：

```bash
python train.py --train-config <approved_physical_forward_config.yaml> \
  --stage one_shot_sample --device cuda \
  --one-shot-checkpoint <one_shot_checkpoint>

python train.py --train-config <approved_physical_forward_config.yaml> \
  --stage one_shot_distill_eval --device cuda \
  --distill-dataset <distill_dataset> \
  --one-shot-checkpoint <one_shot_checkpoint>
```

尖括号表示必须替换的实际路径。显式 CLI checkpoint 路径优先；省略 teacher checkpoint 时仅可使用仓库已记录的 resolver。`one_shot_sample` 没有文档化的自动 resolver，必须显式提供 student checkpoint。

## Checkpoint And Acceptance Rules

- `first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自配套的 latent/decoder 分布；更换 first-stage 后必须重新训练 diffusion。
- 更换 teacher pair、teacher distill dataset 或 D2NN 网格配置后，必须重新导出并重新验证下游链路。
- 正式 student 证据必须来自加载了 one-shot checkpoint 的 `one_shot_sample` 和 `one_shot_distill_eval`；仅训练完成不构成最终证据。
- Teacher sample、teacher audit、direct MNIST `one_shot`、CPU smoke 和 compatibility alias 都不能证明最终单次曝光学生链路可用。
- 详细原因见 [架构文档](single_shot_ldm_distillation.md)；历史测量值见 [实验与验收历史](experiment_history.md)。

## Output Policy

- 训练、诊断和 pre-acceptance run 默认写到 `outputs/YYYYMMDD_<version>/runs/<run>/`；`<version>` 从 `--run-name` 或 train config 文件名提取，缺省为 `unversioned`，日期前缀取该版本当前已知最早 run 日期。
- 诊断和 pre-acceptance 结果仍必须标注为 `diagnostic-only`，不能仅因位于 `outputs/` 就视为 accepted。
- `outputs/runs/` 已移除；具体历史路径需要改写到真实的带日期前缀版本目录，`outputs/final_pt/` 不参与版本迁移。
- 整理历史 run 时先执行 `python -m tools.organize_outputs_by_version --root outputs --dry-run` 检查 manifest，确认无 blocker 后再执行 `--apply`。
- 输出图片和 tensor bundle 的语义见 [输出与可视化说明](../output_JIESHI.md)。

## Archived Experiments

v9.10 prototype、旧 100×100 two-photon、v53 及更早的 loss/relay probes 都已迁入 `experiments/archive/`，并由 [archive manifest](../../experiments/archive/manifest.yaml) 标为 `diagnostic-only`、`runnable: false`。它们只能解释历史证据，不能作为当前命令、checkpoint 或 formal physical-forward 入口；当前合同与停止规则只以 [Current Handoff](current_handoff.md) 为准。

## Diagnostics

以下流程均为 `diagnostic-only`：它们可以定位环节问题，但不能单独证明最终 one-shot student 硬件路径可用。

### Experimental Debugging Decision Policy

本节只定义通用决策流程；当前版本状态、run 路径和下一决策门以 [Current Handoff](current_handoff.md) 为准。

- 从事实开始：记录最新 summary/run 路径、gate 与 preview 状态、first-collapse plane，以及 sample/human review 是否存在。
- 人眼可读性是硬 gate。真实最终 CMOS 为 blob 时，即使 corr/dice、strict eval 或 proxy audit 良好，也不得 promotion。
- 仅在已定位子系统且指标和人类相关 preview 同时改善时进行 micro-tuning；两次聚焦尝试后人眼信号或首塌缩平面无改善，停止 micro-tuning。
- 当 eval 通过但 sample/human 失败、指标改善而 class/prototype audit 仍塌缩、gate 输入无效，或失败平面不变时，切换为 broad isolation。
- 在追加 CUDA 训练或 loss stack 前，broad isolation 必须先证伪一个子系统假设；优先 offline readout、plane-gate、oracle、ablation 或 upper-bound 诊断。
- bypass、identity relay、phase-amplitude override、software oracle 和非合同层数仅是 `diagnostic-only` 证据，不能 promotion。
- 每份实验报告必须列出：已定位问题、已排除原因、仍可能原因、gate 有效性和下一决策 gate；不得只提出“再训练”。

### v67.1 transparent thin-phase gate

`tools/diagnose_v67_transparent_boundary_convergence.py` 是独立的 diagnostic-only 前置门。它先验证零相位 ASM reference 和固定 chip 的 backend relay 坐标，再比较冻结 v60 phase 参数在 1024/1536/2048 传播网格上的中央 100×100 complex field、raw camera intensity、总能量和 guard-band。`finite_crop` 是 Formal Student 的理想硬孔径计算合同；`transparent_identity` 只允许 experimental diagnostic 使用。只要 1024/1536 未同时通过 corr、complex L2、总能量和 guard-band 门，就不得启动透明边界 A/B 训练或相位导出；v67.1 结果以 Current Handoff 和 versioned output report 为准。

### v67.2 padded transparent admission

`tools/diagnose_v672_transparent_phase.py` 是 v67.2 的 diagnostic-only 数值和训练准入门。`padded_band_limited_asm` 只在完整 state grid 外做临时 FFT padding，传播后裁回完整 state grid，不得裁回活动 100×100 区域。候选按 state/FFT `1536/3072`、`2048/4096` 顺序测试；必须同时通过逐层复场、幅度、圆相位、强度、总能量、guard/discarded energy、raw camera、独立 direct-RS、sampling/FOV、梯度和显存门，才允许 A/B 训练。候选不能与自身 reference 比较，固定 pitch 下不同 state grid 的 FOV 不等价时必须停止。v67.2 当前报告为 `grid-not-converged`，未运行训练；不得用 camera corr 单项通过替代完整复场门。

v67.3 将 `1536/3072` 固定为唯一训练候选、`2048/4096` 固定为扩大 reference；总 FOV 不同是允许的，只有共同 ROI 的 pitch、origin、wavelength、距离和坐标映射必须一致。复场 gate 使用 gauge-aligned L2，phase gate 使用 `-40 dB` reference-intensity 权重。zero/small-init 必须先通过，v60 phase 只作 stress test；当前 v67.3 仍在 `decoder_layer_1` 失败，未进入梯度、显存或训练阶段。

### v67.4 full-domain operator audit

`tools/diagnose_v674_transparent_operator.py` 只用于隔离透明传播算子：`1536/3072` 每层 crop 复现历史断点，`3072/3072` 是唯一候选，`4096/4096` 是只读 reference。候选 encoder、identity OLS 和 decoder 之间不得裁回 1536 或 100；普通输出可只保存中央活动视图，但 backend relay 必须接收完整 chip field。只有 zero/small-init 的逐层复场、共同频率 transfer audit 和 reference-domain 状态全部通过，才允许用户运行 gradient/memory smoke；该阶段不自动启动长 CUDA 训练。

### v68 LightGen-calibrated transparent readability

`tools/audit_v68_lightgen_contract.py` 是只读合同审计；它锁定振幅 SLM、完整相干 4f 前端、identity complex-field OLS、后端有限共轭 relay、`2+2 @ 1.50 mm` 和 `100×100 @ 2 µm`。审计中的宏观距离是用户确认的 LightGen 参照标定，不声称为 Science 正文逐项数字；identity OLS 也不声称是论文单模光纤阵列的复现。Nature 仅作为 DDPM 蒸馏/单曝光生成的来源。

`configs/train_v68_transparent_readability_a.yaml` 是硬件路径可读上界（只启用 camera reconstruction）；`..._b.yaml` 只有在 A 的 validation corr/dice 门通过后才运行，并加入最小 clean-target/readability 项。两者均为 `diagnostic-only`，不改变 Formal v56/v60，也不允许 generic complex resize 或旧 v15 中央硬裁切进入 import/forward 图。透明路径可使用 `checkpointed_active_capture`：内部仍保留完整复场并把 backend relay 接收完整 chip field，只有常规 artifact 才保存中央活动视图。先运行合同审计、短数值/显存 preflight，再由用户执行长 CUDA；任何门失败都停止训练和加工候选导出。

短 preflight（不会启动训练）如下；需要显式的 v60 generator-only checkpoint 和固定 sample bundle：

```bash
python -m tools.audit_v68_lightgen_contract \
  --config configs/train_v68_transparent_readability_a.yaml
python -m tools.audit_v68_lightgen_contract \
  --config configs/train_v68_transparent_readability_b.yaml
python -m tools.diagnose_v68_transparent_memory \
  --train-config configs/train_v68_transparent_readability_a.yaml \
  --sample-tensors <fixed_sample_bundle.pt> \
  --output <versioned_v68_preflight.json> --device cuda --max-samples 4
```

该 v68.2 streaming probe 默认从本地 MNIST train 构造 prototype，并固定 small phase seed；逐样本释放内部全场，只比较中央 artifact view 与 raw camera。v60 checkpoint 不兼容时不得完整或静默加载；如需压力测试，必须显式加 `--warm-start-mode generator_subset --checkpoint <checkpoint>`，工具会比较 generator arch/command transform 并报告不兼容。v67.4 的 periodic full-domain 结果保留为 stress test，不替代 v68.2 padded candidate gate。v68.2 只有在 zero/small-init 均报告 `numerical-converged` 后，才进入完整 one_shot_loss 梯度门和显存门。

### v68.3 fixed-exposure raw-CMOS acceptance

v68 A 的 paired 指标已经达到 validation corr `0.9633`、dice `0.9122`，但历史 `student (raw)` 图使用逐样本 percentile/gamma，不能作为硬件人工门。`tools/audit_v68_raw_cmos.py` 是只读审计：它在完整 validation split 上计算一个共享的 99.5% exposure scale，直接读取 `physical_camera_intensity_raw`，固定 gamma=1，并同时输出 raw-linear、固定曝光和历史 normalized 对照。它不看 test、不训练、不保存完整 optical-field bundle；manual review 必须逐类检查轮廓。

审计通过后可以直接进入 B；审计的固定曝光形状或人工门失败时，只能把报告中的 scale 填入 `configs/train_v68_transparent_readability_a2.yaml`，运行一次 5 epoch A2。A2 的 `diagnostic_camera_normalization=fixed_exposure` 要求正的固定 scale，`strict_readability` 选择 checkpoint；A/A2 未通过 raw 门不得运行 code residual B。v68.3 仍为 diagnostic-only，不改变 Formal v56/v60、LightGen 距离、SLM 类型、D2NN 尺寸或层间距。

数值门通过后，长 CUDA 梯度/显存准入由用户显式运行；该工具不创建 optimizer、不会更新参数，也不会替代 A/B 训练：

```bash
python -m tools.diagnose_v68_training_admission \
  --train-config configs/train_v68_transparent_readability_a.yaml \
  --sample-tensors <fixed_sample_bundle.pt> \
  --output <versioned_v68_gradient_admission.json> \
  --device cuda --max-samples 20
```

它逐样本比较 `2048/3072` candidate 与 `3072/4096` streaming reference 的完整 `one_shot_loss`、generator/encoder/decoder 梯度和 CUDA 峰值显存；只有 `gradient_pass`、`memory_pass` 和 `finite` 同时为真，才允许运行 A 的 5 epoch。A 不传入 v60 checkpoint；B 的受控 warm-start 见下方命令。

B 分支的 warm-start 只允许继承 A checkpoint 的 `generator_e.template_logits`、`encoder.*` 和 `decoder.*`；`label_embedding` 与 `residual_net` 必须保持新初始化：

```bash
python train.py --train-config configs/train_v68_transparent_readability_b.yaml \
  --stage one_shot_distill --device cuda \
  --distill-dataset <readable_train.pt> \
  --distill-validation-dataset <readable_val.pt> \
  --one-shot-generator-checkpoint <v68_A_best.pt> \
  --one-shot-optical-checkpoint <v68_A_best.pt>
```

| Stage | 用途 | 何时运行 | 必需输入 | 可作为最终单次曝光证据 |
| --- | --- | --- | --- | --- |
| `teacher_distill_filter` | 按前景/对比度等规则过滤 teacher targets | Teacher target 质量不稳时 | `--distill-dataset` | 否 |
| `teacher_distill_label_audit` | 评估 teacher target 与请求 label 的一致性 | 怀疑 label 可读性或类别混淆时 | `--distill-dataset` | 否 |
| `teacher_distill_label_strict_filter` | 保留 label 一致性更强的 teacher targets | 构造候选可读数据集时 | `--distill-dataset` | 否 |
| `teacher_distill_readable_split` | 生成 readable train/val/test split | 候选数据集进入 student 训练前 | `--distill-dataset` | 否 |
| `teacher_first_stage_oracle_audit` | 检查 first-stage camera oracle 可读性 | First-stage 或 hard-class recovery 调试时 | `--first-stage-checkpoint` | 否 |
| `teacher_diffusion_latent_audit` | 检查 diffusion latent 与 camera 解码链路 | Guidance/checkpoint 配套性调试时 | `--first-stage-checkpoint`、`--diffusion-checkpoint` | 否 |
| `one_shot_distill_eval_diagnostics` | 从 eval tensor bundle 生成 focus 诊断图 | Eval 已完成后定位类别/样本问题 | `--eval-tensor-bundle`、`--eval-metrics` | 否，只解释已有 eval |

## Legacy / Smoke / Compatibility

| Stage 或入口 | 用途 | 何时运行 | 备注 |
| --- | --- | --- | --- |
| `one_shot` | Direct MNIST baseline / CPU smoke | 快速检查 one-shot/D2NN 基础路径 | `diagnostic-only`，不是 LDM distill 主线 |
| `teacher_sample` | Teacher LDM sample baseline | 对照 teacher 生成质量 | 不是最终硬件推理 |
| `teacher_first_stage` | 训练 first-stage teacher 组件 | 更换 teacher/checkpoint 时 | Teacher 基础设施 |
| `teacher_diffusion` | 训练 diffusion teacher 组件 | 更换 matched first-stage 后 | 必须与 first-stage 配套 |
| `teacher_full` | 兼容式 full teacher training | 需要一次运行 teacher 组件时 | 非 one-shot student 入口 |
| `dense_linear` optical core | 旧 baseline/compatibility core | 旧 checkpoint 或对照测试 | 主线使用 `diffractive_phase_only` |
| `first_stage`、`diffusion`、`full`、`sample` | CLI compatibility aliases | 旧脚本/旧命令兼容 | 对应 teacher stages |

## Evaluation Reports

`ldm_project.eval_reports` 只读汇总已有 run，不生成 checkpoint、图片或 artifact：

```bash
python -m ldm_project.eval_reports summary \
  outputs/YYYYMMDD_<version>/runs/<eval_run> outputs/YYYYMMDD_<version>/runs/<sample_run>
```

将 candidate 与保存的 baseline 比较；退出码非零表示比较未通过：

```bash
python -m ldm_project.eval_reports compare \
  outputs/YYYYMMDD_<version>/runs/<eval_run> outputs/YYYYMMDD_<version>/runs/<sample_run> outputs/YYYYMMDD_<version>/runs/<distill_run> \
  --baseline docs/ldm_migration/acceptance_baseline_c99f559.json
```

Baseline 指标、阈值语义和过往结果见 [实验与验收历史](experiment_history.md)。
