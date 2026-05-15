# Final Delivery Checklist

本文记录当前项目交付时应使用的最终模型、验收结果、复现命令和工程结构。除非验收目标发生变化，不应再用 smoke run 的 checkpoint 替换最终模型。

## 1. Final Checkpoints

最终交付 checkpoint 已复制到：

- `outputs/final_pt/first_stage_best_optical.pt`
- `outputs/final_pt/diffusion_best.pt`

原始来源如下：

- First stage: `outputs/runs/20260504_134132_767348_train_first_stage_grid32_ols1ch_artifact_w055_e8/checkpoints/first_stage_best_optical.pt`
- Diffusion: `outputs/runs/20260507_011608_845740_train_diffusion_grid32_ols1ch_artifact_w055_s20k_b128_w4_wc4x300_g45_e30_weak345n16/checkpoints/diffusion_best.pt`

## 2. Final Acceptance

正式验收以 CUDA sample run 的 nearest-neighbor 指标为准：

- Accepted run: `outputs/runs/20260507_081830_752705_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_nn_metrics`
- Latest recheck run: `outputs/runs/20260508_144922_456260_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_diffusion_visual_fix_recheck`
- Device: `cuda`
- Required thresholds: `overall_nearest_dice >= 0.80`, `min_class_nearest_dice >= 0.75`
- Latest recheck metrics: `overall_nearest_dice=0.8492`, `overall_nearest_corr=0.8675`, `min_class_nearest_dice=0.7815`

The latest recheck uses `--sample-class-labels 0 1 2 3 4 5 6 7 8 9` and `--sample-repeats-per-class 4`, so every class is represented in the final sample grid.

## 3. Final First-stage Structure

最终 first-stage checkpoint 使用 `physical_first_linear_metasurface_vae_v5`，结构以 `linear_metasurface` 为准：

| 模块 | 层数 | 结构 | 输入维度 | 输出维度 | 参数量 |
|---|---:|---|---:|---:|---:|
| Encoder | 1 | `Linear(bias=False)` | `2048 = 2 * 32 * 32` | `1250 = 2 * 1 * 25 * 25` | `2,560,000` |
| OLS | 1 | pointwise `GELU` | `1 * 25 * 25` | `1 * 25 * 25` | `0` |
| Decoder | 1 | `Linear(bias=False)` | `625 = 1 * 25 * 25` | `2048 = 2 * 32 * 32` | `1,280,000` |
| Total | 3 | Encoder + OLS + Decoder | - | - | `3,840,000` |

`2048` 不是 OLS 输入维度，而是 PF2 复光场 `32x32` 网格的实部与虚部拼接结果。若把 encoder/decoder 光场接口改成 `1x32x32`，会丢掉一个复场通道或改成强度读出，属于新的模型假设；现有 final first-stage 与 diffusion checkpoint 不能直接复用，需要重新训练并重新验收。

## 4. Visualization Smoke

第二阶段训练过程图已经修复为固定显示 `0-9`：

- Smoke run: `outputs/runs/20260508_145449_512992_train_smoke_diffusion_visual_0_9_check`
- Config: `configs/diffusion_visual_config.json`
- Required labels: `class_labels == [0,1,2,3,4,5,6,7,8,9]`
- Required reference labels: `reference_labels == [0,1,2,3,4,5,6,7,8,9]`
- Diffusion epoch figure: `images/diffusion/epoch_0001.png`

This smoke run is not a quality checkpoint. Its one-epoch model is expected to score lower than the final accepted checkpoint.

## 5. Reproduction Command

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical-generative-models

python train.py \
  --train-config configs/train_first_stage_linear_metasurface_2000_grid32.yaml \
  --stage sample \
  --device cuda \
  --first-stage-checkpoint "outputs/final_pt/first_stage_best_optical.pt" \
  --diffusion-checkpoint "outputs/final_pt/diffusion_best.pt" \
  --guidance-scale 4.5 \
  --sample-class-labels 0 1 2 3 4 5 6 7 8 9 \
  --sample-repeats-per-class 4 \
  --run-name final_pt_acceptance_recheck
```

## 6. Project Structure

Current execution responsibilities:

- `train.py`: CLI, config loading, dependency initialization, high-level stage dispatch
- `src/ldm_project/stage_runners.py`: sample, first-stage, and diffusion stage execution
- `src/ldm_project/sampling.py`: sample labels, reference selection, sample artifacts, nearest-neighbor metrics
- `src/ldm_project/evaluation.py`: validation and balanced metric summaries
- `src/ldm_project/train_utils.py`: stage plan, logging intervals, checkpoint helper decisions, latent-scale helpers
- `tests/`: CPU-friendly unit and smoke tests for the core workflow

## 8. One-Shot Student Acceptance

### 8.1 Architecture

One-Shot Student (`OneShotOpticalGenerator`) 使用单层线性超表面 encoder/decoder，保留完整复光场（振幅+相位）信息：

| 模块 | 层数 | 结构 | 输入维度 | 输出维度 | 参数量 |
|---|---:|---|---:|---:|---:|
| Encoder | 1 | `Linear(bias=False)` | `512 = 2 * 16 * 16` | `512 = 2 * 16 * 16` | `262,144` |
| OLS | 1 | pointwise `GELU` | `1 * 16 * 16` complex | `1 * 16 * 16` complex | `0` |
| Decoder | 1 | `Linear(bias=False)` | `512 = 2 * 16 * 16` | `512 = 2 * 16 * 16` | `262,144` |

- 空间通道 C=1，`2*G²` 中的 `2` 是复光场实虚部拼接
- `metasurface_grid_size = 16`（grid16 在所有指标上优于 grid32）

### 8.2 Checkpoints

| 文件 | 来源 |
|---|---|
| Student (quality v2) | `outputs/runs/20260515_161215_855773_train_one_shot_distill_quality_v2/checkpoints/one_shot_distill_best.pt` |
| Student (baseline v0) | `outputs/runs/20260515_152205_925342_train_one_shot_distill_grid16_acceptance/checkpoints/one_shot_distill_best.pt` |
| Teacher distill dataset | `outputs/runs/20260514_161039_737798_train_teacher_distill_mnist_3k_b96_fast200/artifacts/teacher_distill_dataset.pt` |

### 8.3 Quality Improvements (v0 → v2)

v2 改进点：`hidden_dim` 256→512、新增 `camera_topology_loss` + `camera_artifact_loss`、训练 60 epochs。

#### Sample Metrics 对比

| 指标 | Baseline v0 | Quality v2 | 变化 |
|---|---:|---:|---|
| `overall dice` | 0.6811 | **0.7031** | +3.2% |
| `overall hole_proxy` | 0.2385 | **0.1918** | **-19.6%** |
| `class 3 dice` | 0.4807 | **0.5464** | **+13.7%** |
| `class 3 hole` | 0.3692 | **0.2689** | **-27.2%** |
| `class 4 hole` | 0.3871 | **0.2925** | **-24.4%** |
| `class 5 dice` | 0.6527 | **0.6922** | +6.1% |

#### Distill Eval (v2, paired teacher-student)

Run: `outputs/runs/20260515_162453_319064_train_one_shot_distill_eval_quality_v2`

| 指标 | 结果 | 阈值 | 状态 |
|---|---:|---:|---|
| `paired_dice` | **0.8506** | ≥ 0.85 | ✅ |
| `paired_corr` | **0.9141** | ≥ 0.90 | ✅ |
| `min_class_paired_dice` | **0.7709** (class 7) | ≥ 0.75 | ✅ |
| `diversity_ratio` | **0.5770** | ≥ 0.20 | ✅ |
| `saturation_ratio` | **0.3517** | ≤ 0.40 | ✅ |

#### Sample (v2)

Run: `outputs/runs/20260515_162552_977757_train_one_shot_sample_quality_v2`

| 指标 | 结果 | 阈值 | 状态 |
|---|---:|---:|---|
| `sample_dice` | **0.7031** | ≥ 0.65 | ✅ |
| `sample_corr` | **0.6698** | — | — |
| `hole_proxy` | **0.1918** | ≤ 0.25 | ✅ |

### 8.4 Reproduction Commands

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical-generative-models

# Train quality v2
python train.py \
  --train-config configs/train_one_shot_distill_mnist_3k_grid16_quality_v2.yaml \
  --stage one_shot_distill \
  --device cuda \
  --distill-dataset outputs/runs/20260514_161039_737798_train_teacher_distill_mnist_3k_b96_fast200/artifacts/teacher_distill_dataset.pt \
  --run-name one_shot_distill_quality_v2

# Distill Eval
python train.py \
  --train-config configs/train_one_shot_distill_mnist_3k_grid16_quality_v2.yaml \
  --stage one_shot_distill_eval \
  --device cuda \
  --one-shot-checkpoint outputs/runs/20260515_161215_855773_train_one_shot_distill_quality_v2/checkpoints/one_shot_distill_best.pt \
  --distill-dataset outputs/runs/20260514_161039_737798_train_teacher_distill_mnist_3k_b96_fast200/artifacts/teacher_distill_dataset.pt \
  --run-name one_shot_distill_eval_quality_v2_recheck

# Sample
python train.py \
  --train-config configs/train_one_shot_distill_mnist_3k_grid16_quality_v2.yaml \
  --stage one_shot_sample \
  --device cuda \
  --one-shot-checkpoint outputs/runs/20260515_161215_855773_train_one_shot_distill_quality_v2/checkpoints/one_shot_distill_best.pt \
  --run-name one_shot_sample_quality_v2_recheck
```

## 7. Final Checks

Before handoff, run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical-generative-models

python -m compileall src tests train.py
pytest -q
ruff check src tests *.py
```
