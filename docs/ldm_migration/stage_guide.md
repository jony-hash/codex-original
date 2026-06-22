# Stage Guide

本文件是选择和运行 stage 的操作权威来源。架构与最终推理边界见 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)，带日期的结果见 [实验与验收历史](experiment_history.md)，完整文档导航见 [Documentation](../README.md)。

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

训练学生：

```bash
python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml \
  --stage one_shot_distill --device cuda \
  --distill-dataset <distill_dataset>
```

采样并评估同一个 student checkpoint：

```bash
python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml \
  --stage one_shot_sample --device cuda \
  --one-shot-checkpoint <one_shot_checkpoint>

python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml \
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

- 诊断和 pre-acceptance run 写到 `/tmp`。
- 已接受或正式配置的 managed run 写到 `outputs/runs/<run>/`。
- 现有 `outputs/` artifact 不得编辑、删除或回填。
- 输出图片和 tensor bundle 的语义见 [输出与可视化说明](../output_JIESHI.md)。

## v9.10 Prototype Baseline Audit

`v9.10 label-template inverse prototype baseline` 明确忽略 code，只验证 intensity-template 到 amplitude SLM command 的逆映射与完整 `relay_coherent_psf -> D2NN -> relay_psf` 光路。它不是 code-conditioned generative baseline。

先确认 WSL2 环境可见 CUDA：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && \
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

正式 audit 由用户启动，所有结果写入 `/tmp`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && \
python tools/run_v9_10_label_template_audit.py \
  --train-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_train.pt \
  --validation-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_val.pt \
  --test-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_test.pt \
  --output-root /tmp/v9_10_label_template_inverse \
  --device cuda
```

启动器先生成 self-contained checkpoint，再用独立 test split 执行 strict eval。唯一 required gate 是 `paired_corr >= 0.78`、`paired_dice >= 0.72`、`student_high_saturation_ratio <= 0.08`；模板和 D2NN 的零学习率参数还必须满足 `max_abs_change == 0`。任一条件失败时停止，不运行 sample。通过后才用同一 checkpoint 运行 `one_shot_sample`，并写入唯一的 `v9_10_label_template_audit_summary_*.json`。

只检查命令和配置、不启动 CUDA 作业时添加 `--dry-run`。不得把 `/tmp` checkpoint 复制或回填到旧 `outputs/` run。

## Two-Photon Printing 100 × 100 Experiment

该实验沿用受保护 v7.2 参考 run 的 phase-only、encoder/decoder 各两层结构，只修改 D2NN 加工几何。神经元边长与 pitch 均为 `2 µm`，`100 × 100` 神经元形成 `200 µm × 200 µm` 有效口径；数值传播 padding 保持 `1024`，CMOS/loss 接口保持 `200 × 200`。

按可见光 D2NN 全连接判据，`532 nm` 波长下理论最小层间距为：

`D_min = sqrt(10000) × 2 µm × sqrt(4 × (2 µm)^2 / (0.532 µm)^2 - 1) = 1.490 mm`

配置采用 `1.50 mm`。旧 `200 × 200` checkpoint 的相位参数形状不兼容，必须从头训练，禁止 resize 或静默部分加载。正式训练前先在 `/tmp` 做 pre-acceptance：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && \
python train.py \
  --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200_twophoton_100x100_2um_full_connect.yaml \
  --stage one_shot_distill --device cuda \
  --distill-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_train.pt \
  --distill-validation-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_val.pt \
  --output-root /tmp/twophoton_100x100_2um \
  --run-name twophoton_100x100_2um_full_connect
```

训练产生新 checkpoint 后，再使用配套 eval 配置和独立 test split；不得传入受保护 v7.2 checkpoint：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && \
python train.py \
  --train-config configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_twophoton_100x100_2um_full_connect_strict_readability_test.yaml \
  --stage one_shot_distill_eval --device cuda \
  --one-shot-checkpoint /tmp/twophoton_100x100_2um/<new-run>/checkpoints/one_shot_distill_best.pt \
  --distill-dataset outputs/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_test.pt \
  --output-root /tmp/twophoton_100x100_2um \
  --run-name twophoton_100x100_2um_full_connect_eval
```

本配置尚未完成 CUDA 训练、strict readability 或实验室加工验收，不得仅凭静态几何检查称为 accepted。

## Diagnostics

以下流程均为 `diagnostic-only`：它们可以定位环节问题，但不能单独证明最终 one-shot student 硬件路径可用。

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
  outputs/runs/<eval_run> outputs/runs/<sample_run>
```

将 candidate 与保存的 baseline 比较；退出码非零表示比较未通过：

```bash
python -m ldm_project.eval_reports compare \
  outputs/runs/<eval_run> outputs/runs/<sample_run> outputs/runs/<distill_run> \
  --baseline docs/ldm_migration/acceptance_baseline_c99f559.json
```

Baseline 指标、阈值语义和过往结果见 [实验与验收历史](experiment_history.md)。
