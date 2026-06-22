# D2NN 实验与验收历史

本文件保存带日期的实验事实和验收证据，不定义当前执行契约。当前架构见 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)，stage 命令与输入要求见 [Stage Guide](stage_guide.md)，完整导航见 [Documentation](../README.md)。

## 状态快照：2026-06-10

- **已验收稳定基线**：`3k/fast200` teacher distill dataset + bs128 matched teacher + seed202 one-shot distill，学生复现实验使用 `one_shot_distill_epoch_0020.pt` 做 sample/eval。
- **默认最佳 checkpoint 口径**：常规 distilled run 使用 `one_shot_distill_best.pt`；seed sweep 或历史验收复现可使用文档明确指定的 epoch checkpoint。
- **候选主线**：`15k_fast200` export、`guidance_scale=4.0`、`gs25/30/40` sweep、label-strict filter、readable split、teacher oracle/latent audit 当时作为候选链路使用。它们完成 `one_shot_distill -> one_shot_sample -> one_shot_distill_eval` 全链路验收前，不替代已验收稳定基线。
- **诊断/探索工具**：teacher label audit、strict filter、readable split、oracle audit、latent audit 用于筛选和定位 teacher targets，不属于最终硬件推理路径。

## 2026-06-03 全链路验收

**基线提交**：`c99f559 refactor: training orchestration into stage modules`

**环境**：RTX 3060 Laptop GPU (6GB), CUDA 12.1, PyTorch 2.5.1+cu121, WSL2 Linux

**证据文件**：[acceptance_baseline_c99f559.json](acceptance_baseline_c99f559.json)

| Stage | 配置 | Epochs | 关键指标 | Exit |
| --- | --- | --- | --- | --- |
| `teacher_first_stage` | `configs/train_ldm.yaml` | 35 | `camera_corr=0.9547`, `balanced=0.8770`, `mnist_score=0.8144` | 0 |
| `one_shot_distill` | `configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml` | 20 | `quality=0.1614`, `camera_dice=0.4575` | 0 |
| `one_shot_sample` | 同上 config | - | 可视化、诊断和 tensor bundle 完整 | 0 |
| `one_shot_distill_eval` | 同上 config | - | `mse=0.020`, `corr=0.807`, `dice=0.756` | 0 |

- Teacher 蒸馏数据为 3000 samples，源自 `teacher_distill_export`（batch size 96、guidance scale 2.0、steps 200）。
- 训练和采样均使用 CUDA。
- 原始验收使用 `one_shot_distill_best.pt`；2026-06-04 seed sweep 复现使用 `one_shot_distill_epoch_0020.pt`。
- 所有 artifact/checkpoint 位于 `outputs/runs/<run>/`，不纳入 git。

### 评估结果汇总工具

`ldm_project.eval_reports` 可只读汇总已有 run，返回 `stage`、`checkpoint`、`distill_dataset`、`best_loss`、`eval_metrics` 和 `sample_metrics`。缺失 optional artifact 时对应字段为 `null`，缺失 run metadata 时记录 error；该工具不生成 checkpoint、图片或新 artifact。

当前命令模板和回归比较方式由 [Stage Guide](stage_guide.md) 维护。

### 回归比较

比较方向保持如下：

- `eval_mse`、`eval_sat`、`eval_slm_tv`、`best_loss`：lower is better。
- `eval_corr`、`eval_dice`、`eval_diversity_ratio`、`sample_corr`、`sample_dice`：higher is better。
- 多个 candidate run 按 best-of 合并：lower-is-better 取最小值，higher-is-better 取最大值。
- Baseline 值为 `null` 的指标跳过；candidate 缺少 baseline 要求的指标视为失败。
- 阈值优先使用 baseline JSON 的 `tolerances`，其次使用代码内默认值。

### bs128 匹配扩散基线

**验收日期**：2026-06-03

**证据文件**：[acceptance_baseline_bs128_matched_diffusion.json](acceptance_baseline_bs128_matched_diffusion.json)

bs128 first-stage 相比旧基线将训练吞吐从 563 step/epoch 改善到 71 step/epoch，同时保持或超过核心重建指标。

| 指标 | c99f559 | bs128 matched | 变化 |
| --- | --- | --- | --- |
| `eval_corr` | 0.807 | 0.809 | +0.002 |
| `eval_dice` | 0.756 | 0.777 | +0.021 |
| `eval_mse` | 0.0204 | 0.0209 | +0.0005 |
| `best_loss` | 0.161 | 0.143 | -0.018 |
| `eval_slm_tv` | 0.382 | 0.277 | -0.105 |
| `sample_corr` | 0.588 | 0.550 | -0.038 |
| `sample_dice` | 0.625 | 0.616 | -0.009 |
| `diversity_ratio` | 0.531 | 0.269 | -0.262 |

核心重建指标持平或提升，SLM pattern 更平滑。`diversity_ratio` 下降的记录结论是 teacher target 多样性提高而学生未完全复现，不作为该基线的阻塞项。

## Matched First-Stage 与 Diffusion 证据

稳定规则是 `first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自同一 first-stage 训练的 latent/decoder 分布；规则和原因由 [架构文档](single_shot_ldm_distillation.md) 维护。

| 配置 | `eval_corr` | `eval_dice` | `eval_mse` |
| --- | --- | --- | --- |
| bs128 first-stage + **匹配** diffusion | 0.809 | 0.777 | 0.0209 |
| bs128 first-stage + **旧 grid_size=32** diffusion | 0.421 | 0.483 | 0.0404 |

不匹配时 corr 从 0.809 降到 0.421，dice 从 0.777 降到 0.483，mse 约翻倍。`best_loss` 反而从 0.161 降到 0.154，说明 shape/edge/SLM-TV regularizer 可能掩盖重建退化，不能用单一 loss 判断 checkpoint 是否配套。

## Diversity 实验（负结果）

以下实验以 bs128 matched-diffusion 基线（`diversity_ratio=0.269`）为起点，记录在此以避免重复尝试已经验证的方向。

### code_diversity_weight 翻倍

- 变更：`code_diversity_weight` 从 0.03 增至 0.06。
- 结果：`diversity_ratio=0.295`，但 corr 降 3.2%、dice 降 3.0%、`best_loss=0.160`。
- 结论：重建代价过大，该方向不可行。

### teacher_pairwise_diversity_weight 翻倍

- 变更：`teacher_pairwise_diversity_weight` 从 0.005 增至 0.01。
- 结果：`diversity_ratio=0.312`、`corr=0.776`、`dice=0.750`、`best_loss=0.163`。
- 结论：重建指标低于验收标准，损害大于 code-diversity 调整。

### Multiclass Batch

- 变更：`paired_class_batches: false`、`multiclass_batches: true`。
- 结果：`diversity_ratio=0.292`、`corr=0.783`、`dice=0.757`。
- 结论：diversity 几乎不变，重建轻微退化，视为中性方向。

### 同配置重新训练

| 指标 | 原基线 | Retrain ep20 |
| --- | --- | --- |
| `eval_corr` | 0.809 | 0.810 |
| `eval_dice` | 0.777 | 0.774 |
| `eval_mse` | 0.0209 | 0.0205 |
| `diversity_ratio` | 0.269 | **0.322** |
| `sample_corr` | 0.550 | 0.561 |
| `sample_dice` | 0.616 | 0.609 |
| `best_loss` | 0.143 | 0.142 |

记录结论是 diversity 的改善来自 D2NN phase 初始化差异。后续方向应使用多 seed 验证波动范围，而不是继续增加上述 loss 权重。

## 诊断与复现记录

### Epoch Checkpoint

`training.one_shot_distill_epoch_checkpoint_interval` 控制周期 checkpoint；默认 0 表示关闭。启用后生成 `one_shot_distill_epoch_XXXX.pt`，并记录到 `run_meta["epoch_checkpoints"]`。历史诊断配置为 `configs/train_one_shot_distill_d2nn_mnist_3k_fast200_debug_epoch_ckpt.yaml`，interval 为 2。

### Seed Sweep

`training.seed` 或 CLI `--seed` 固定 one-shot 训练随机性；默认 `null` 保持随机行为。历史流程固定 teacher dataset 和正式 student 配置，分别运行 seed 101、202、303，再对每个 ep20 checkpoint 执行 sample/eval。

通过标准为三个 seed 至少两个同时满足：`eval_corr >= 0.79`、`eval_dice >= 0.765`、`sample_corr >= 0.53`、`sample_dice >= 0.60`、`eval_diversity_ratio >= 0.31`。

**记录提交**：`ebba8e9`

**证据文件**：[acceptance_seed_sweep_ebba8e9.json](acceptance_seed_sweep_ebba8e9.json)

| 指标 | 阈值 | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- | --- |
| `eval_corr` | >=0.79 | 0.805 | 0.809 | 0.810 |
| `eval_dice` | >=0.765 | 0.771 | 0.780 | 0.779 |
| `sample_corr` | >=0.53 | 0.553 | 0.558 | 0.524 |
| `sample_dice` | >=0.60 | 0.607 | 0.612 | 0.582 |
| `eval_diversity_ratio` | >=0.31 | 0.357 | 0.333 | 0.301 |

三个 seed 中两个达标。seed 202 的重建与 diversity 最稳；seed 101 的 diversity 最高；seed 303 是 eval 重建通过但 sample 和 diversity 略低的 near-miss。

### Post-Refactor One-Shot Sample Smoke

`one_shot.py` 拆分为 `one_shot_model.py` 与 `one_shot_optical_layers.py` 后，使用 seed202 ep20 checkpoint 跑通 CUDA `one_shot_sample`。

**记录提交**：`8a307f3`

**证据文件**：[acceptance_post_refactor_one_shot_smoke_8a307f3.json](acceptance_post_refactor_one_shot_smoke_8a307f3.json)

| 项 | 值 |
| --- | --- |
| run | `outputs/runs/20260604_160035_377670_train_post_refactor_one_shot_sample_seed202` |
| checkpoint | `outputs/runs/20260604_145621_832064_train_one_shot_distill_seed202/checkpoints/one_shot_distill_epoch_0020.pt` |
| `sample_corr` | 0.5448 |
| `sample_dice` | 0.6095 |

结论：拆分后学生采样路径仍可用，sample metrics、tensor bundle、D2NN diagnostics 和可视化 artifact 均正常生成。

## 稳定基线复现记录

当时推荐使用 bs128 matched teacher、filtered distill dataset、seed202 one-shot distill，并以 ep20 checkpoint 作为复现 checkpoint。默认 config 不固定 seed，正式复现实验通过 CLI 显式传入 `--seed 202`。

维护结论：

- 不继续提高 `code_diversity_weight`、`teacher_pairwise_diversity_weight` 或改变 batch composition；记录结果显示这些方向会用重建质量换 diversity。
- 不把 `training.seed` 写入默认 config；复现实验通过 CLI 固定 seed。
- 更换 first-stage、diffusion、teacher distill dataset 或 D2NN 网格配置后，必须重新跑全链路验收。
- `15k_fast200`、guidance sweep、label-strict filter 和 readable split 在该快照中仍是候选链路。
- Teacher oracle/latent audit 只能说明 teacher target 或 latent 链路质量，不能证明 one-shot student 已达到最终硬件推理就绪。

历史使用的具体 run 路径保留在本文件其他记录和 managed `outputs/` metadata 中；当前命令模板见 [Stage Guide](stage_guide.md)。

## v9.8 full-relay 受控复验

`front relay_coherent_psf + back relay_psf` audit 只证明 relay/readout 保像，不证明随机初始化的 `generator_e` 可以冻结后直接训练 D2NN。

只要配置设置 `one_shot.generator_base_lr_scale: 0.0`，`one_shot_distill` 就必须显式传入 `--one-shot-generator-checkpoint` 或完整的 `--one-shot-checkpoint`，缺失时应 fail fast。

历史诊断先运行 1 epoch，再决定是否运行完整 10 epoch。使用的配置是 `configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_8_front_coherent_psf_back_psf_relay.yaml`，输出写到 `/tmp/codex-v9-8-corrected-runs`。

若 warm-start summary 未加载 `generator_e.*`、首轮 label accuracy 仍接近随机，或 strict eval 没有改善，不得解释为 D2NN 可训练性结论；应先检查 checkpoint 与 generator 输入语义。

## 2026-06-20 至 2026-06-22 光路定位与 v9.10 候选

### v9.8 本地可追溯证据

当前工作区可追溯的 artifact 是：

`outputs/runs/20260620_102234_647492_train_one_shot_d2nn_v9_8_front_coherent_psf_back_psf_relay_audit_v2/artifacts/one_shot_identity_audit_metrics.json`

其中正式几何为 D2NN grid `200`、propagation grid `1024`、spacing `0.5 mm`、encoder/decoder 各一层。`front_relay_coherent_psf_sigma1` 的 `corr_to_slm=0.9676`、dice `0.9814`；`back_relay_psf_sigma1` 的 `corr_to_slm=0.9589`、dice `0.9804`。zero-mask 检查通过。该证据说明受控 relay/identity audit 下光路可读，不等同于 code-conditioned student 验收。

迁移交接摘要还报告了另一组 v9.8 formal re-verification 数值（camera corr `0.9476`、dice `0.8328`、foreground hole `0.1245`），但当前工作区未找到对应 artifact，因此这些数值不作为本文件的已复验证据。

### v9.9 迁移交接结论（当前未复验）

迁移摘要报告 v9.9 validation 曾达到 corr `0.9821`、dice `0.9421`、hole `0.0110`，但独立 code 导致类别形态漂移，固定 zero code 的最小 margin 也不足。当前工作区没有对应 checkpoint、metrics 或 `/tmp` artifact；因此这里只记录交接结论，不把这些数值晋级为本地验收事实。维护决策仍是：v9.9 不能作为鲁棒 code-conditioned baseline。

### v9.10 状态

v9.10 新增 label-only template generator 与 `intensity_to_amplitude` command transform。checkpoint 自包含 `[num_classes, 1, H, W]` intensity template bank，接口保留 code 但明确忽略；模板和 D2NN 均冻结。该版本只能称为 `prototype_baseline`。

2026-06-22 的受控 CUDA audit 已完成并通过。冻结参数检查中 generator、encoder/decoder phase 与 amplitude 的 `max_abs_change` 均为 `0.0`；独立 test split 得到 `paired_corr=0.99944`、`paired_dice=0.98052`、`student_high_saturation_ratio=0.005975`，三项 required gate 全部通过。脱敏后的持久证据见 [acceptance_v9_10_label_template_inverse_20260622.json](acceptance_v9_10_label_template_inverse_20260622.json)。

该结果只接受为 label-only `prototype_baseline`：code conditioning 仍未启用，不能解释为同类多样性或最终 generative student readiness。审计运行和 checkpoint 位于 `/tmp`，未提升为 managed `outputs/` artifact。

### 2026-06-11 v7.2 受保护参考 run

`outputs/runs/20260611_154426_446321_train_one_shot_distill_eval` 使用 phase-only D2NN、encoder/decoder 各两层、`200 × 200` 神经元、`2.5 µm` pitch、`5 mm` 层间距。它保留为后续物理尺寸实验的行为参考，目录及其引用的 student checkpoint 和 teacher dataset 不得删除或回填。

该 run 的 `paired_corr=0.82863`、`paired_dice=0.75583`，但 strict readability 未通过：最差类别 label margin 为 `0.01852 < 0.03`，background speckle ratio 为 `0.08374 > 0.07`。因此它是受保护参考证据，不是已接受基线。

### 双光子打印 100 × 100 尺寸候选

新尺寸候选保留 v7.2 的 phase-only、encoder/decoder 各两层结构和 `200 × 200` CMOS/loss 接口，仅把物理 D2NN 改为 `100 × 100` 神经元、`2 µm` 边长/pitch、`1024` propagation padding。有效口径为 `200 µm × 200 µm`。

按最大半锥角全连接公式计算，`532 nm` 下理论最小层间距为 `1.490 mm`，配置采用 `1.50 mm`。该几何与旧 `200 × 200` 相位 checkpoint 不兼容，后续必须从头训练。截至本记录，仅完成配置、公式与 CPU 回归检查，尚无 CUDA、strict readability 或加工验收结果。
