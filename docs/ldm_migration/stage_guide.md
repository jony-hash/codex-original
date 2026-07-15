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
