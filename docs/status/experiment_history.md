# D2NN 实验与验收历史

本文件保存带日期的实验事实和验收证据，不定义当前执行契约。迁移或续接当前主线时先读 [当前主线交接（v79 冻结与 v2）](current_handoff.md)；当前架构见 [D2NN 单次曝光蒸馏主线](../architecture/v1_single_shot_distillation.md)，stage 命令与输入要求见 [Stage Guide](../operations/stage_guide.md)，完整导航见 [Documentation](../README.md)。本文件的“当前/下一步”均表示各记录当时的结论；历史证据保留，不应覆盖现行合同。

## 状态快照：2026-06-10

- **已验收稳定基线**：`3k/fast200` teacher distill dataset + bs128 matched teacher + seed202 one-shot distill，学生复现实验使用 `one_shot_distill_epoch_0020.pt` 做 sample/eval。
- **默认最佳 checkpoint 口径**：常规 distilled run 使用 `one_shot_distill_best.pt`；seed sweep 或历史验收复现可使用文档明确指定的 epoch checkpoint。
- **候选主线**：`15k_fast200` export、`guidance_scale=4.0`、`gs25/30/40` sweep、label-strict filter、readable split、teacher oracle/latent audit 当时作为候选链路使用。它们完成 `one_shot_distill -> one_shot_sample -> one_shot_distill_eval` 全链路验收前，不替代已验收稳定基线。
- **诊断/探索工具**：teacher label audit、strict filter、readable split、oracle audit、latent audit 用于筛选和定位 teacher targets，不属于最终硬件推理路径。

## 2026-06-03 全链路验收

**基线提交**：`c99f559 refactor: training orchestration into stage modules`

**环境**：RTX 3060 Laptop GPU (6GB), CUDA 12.1, PyTorch 2.5.1+cu121, WSL2 Linux

**证据文件**：[acceptance_baseline_c99f559.json](../archive/acceptance_baseline_c99f559.json)

| Stage | 配置 | Epochs | 关键指标 | Exit |
| --- | --- | --- | --- | --- |
| `teacher_first_stage` | `configs/train_ldm.yaml` | 35 | `camera_corr=0.9547`, `balanced=0.8770`, `mnist_score=0.8144` | 0 |
| `one_shot_distill` | `experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml` | 20 | `quality=0.1614`, `camera_dice=0.4575` | 0 |
| `one_shot_sample` | 同上 config | - | 可视化、诊断和 tensor bundle 完整 | 0 |
| `one_shot_distill_eval` | 同上 config | - | `mse=0.020`, `corr=0.807`, `dice=0.756` | 0 |

- Teacher 蒸馏数据为 3000 samples，源自 `teacher_distill_export`（batch size 96、guidance scale 2.0、steps 200）。
- 训练和采样均使用 CUDA。
- 原始验收使用 `one_shot_distill_best.pt`；2026-06-04 seed sweep 复现使用 `one_shot_distill_epoch_0020.pt`。
- 当前 run artifact/checkpoint 位于 `outputs/YYYYMMDD_<version>/runs/<run>/`，不纳入 git；旧记录中的历史路径已迁移为真实带日期前缀的版本目录。

### 评估结果汇总工具

`ldm_project.eval_reports` 可只读汇总已有 run，返回 `stage`、`checkpoint`、`distill_dataset`、`best_loss`、`eval_metrics` 和 `sample_metrics`。缺失 optional artifact 时对应字段为 `null`，缺失 run metadata 时记录 error；该工具不生成 checkpoint、图片或新 artifact。

当前命令模板和回归比较方式由 [Stage Guide](../operations/stage_guide.md) 维护。

### 回归比较

比较方向保持如下：

- `eval_mse`、`eval_sat`、`eval_slm_tv`、`best_loss`：lower is better。
- `eval_corr`、`eval_dice`、`eval_diversity_ratio`、`sample_corr`、`sample_dice`：higher is better。
- 多个 candidate run 按 best-of 合并：lower-is-better 取最小值，higher-is-better 取最大值。
- Baseline 值为 `null` 的指标跳过；candidate 缺少 baseline 要求的指标视为失败。
- 阈值优先使用 baseline JSON 的 `tolerances`，其次使用代码内默认值。

### bs128 匹配扩散基线

**验收日期**：2026-06-03

**证据文件**：[acceptance_baseline_bs128_matched_diffusion.json](../archive/acceptance_baseline_bs128_matched_diffusion.json)

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

稳定规则是 `first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自同一 first-stage 训练的 latent/decoder 分布；规则和原因由 [架构文档](../architecture/v1_single_shot_distillation.md) 维护。

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

`training.one_shot_distill_epoch_checkpoint_interval` 控制周期 checkpoint；默认 0 表示关闭。启用后生成 `one_shot_distill_epoch_XXXX.pt`，并记录到 `run_meta["epoch_checkpoints"]`。历史诊断配置为 `experiments/archive/pre_v56/configs/train_one_shot_distill_d2nn_mnist_3k_fast200_debug_epoch_ckpt.yaml`，interval 为 2。

### Seed Sweep

`training.seed` 或 CLI `--seed` 固定 one-shot 训练随机性；默认 `null` 保持随机行为。历史流程固定 teacher dataset 和正式 student 配置，分别运行 seed 101、202、303，再对每个 ep20 checkpoint 执行 sample/eval。

通过标准为三个 seed 至少两个同时满足：`eval_corr >= 0.79`、`eval_dice >= 0.765`、`sample_corr >= 0.53`、`sample_dice >= 0.60`、`eval_diversity_ratio >= 0.31`。

**记录提交**：`ebba8e9`

**证据文件**：[acceptance_seed_sweep_ebba8e9.json](../archive/acceptance_seed_sweep_ebba8e9.json)

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

**证据文件**：[acceptance_post_refactor_one_shot_smoke_8a307f3.json](../archive/acceptance_post_refactor_one_shot_smoke_8a307f3.json)

| 项 | 值 |
| --- | --- |
| run | `outputs/20260514_unversioned/runs/20260604_160035_377670_train_post_refactor_one_shot_sample_seed202` |
| checkpoint | `outputs/20260514_unversioned/runs/20260604_145621_832064_train_one_shot_distill_seed202/checkpoints/one_shot_distill_epoch_0020.pt` |
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

历史使用的具体 run 路径保留在本文件其他记录和 managed `outputs/` metadata 中；当前命令模板见 [Stage Guide](../operations/stage_guide.md)。

## v9.8 full-relay 受控复验

`front relay_coherent_psf + back relay_psf` audit 只证明 relay/readout 保像，不证明随机初始化的 `generator_e` 可以冻结后直接训练 D2NN。

只要配置设置 `one_shot.generator_base_lr_scale: 0.0`，`one_shot_distill` 就必须显式传入 `--one-shot-generator-checkpoint` 或完整的 `--one-shot-checkpoint`，缺失时应 fail fast。

历史诊断最初先运行 1 epoch，再决定是否运行完整 10 epoch；当时的 scratch 输出写到 `/tmp/codex-v9-8-corrected-runs`。当前工作区可追溯的 managed v9.8 artifact 已整理到 `outputs/20260620_v9_8/`，见下一节。

若 warm-start summary 未加载 `generator_e.*`、首轮 label accuracy 仍接近随机，或 strict eval 没有改善，不得解释为 D2NN 可训练性结论；应先检查 checkpoint 与 generator 输入语义。

## 2026-06-20 至 2026-06-22 光路定位与 v9.10 候选

### v9.8 本地可追溯证据

当前工作区可追溯的 artifact 是：

`outputs/20260620_v9_8/runs/20260620_102234_647492_train_one_shot_d2nn_v9_8_front_coherent_psf_back_psf_relay_audit_v2/artifacts/one_shot_identity_audit_metrics.json`

其中正式几何为 D2NN grid `200`、propagation grid `1024`、spacing `0.5 mm`、encoder/decoder 各一层。`front_relay_coherent_psf_sigma1` 的 `corr_to_slm=0.9676`、dice `0.9814`；`back_relay_psf_sigma1` 的 `corr_to_slm=0.9589`、dice `0.9804`。zero-mask 检查通过。该证据说明受控 relay/identity audit 下光路可读，不等同于 code-conditioned student 验收。

迁移交接摘要还报告了另一组 v9.8 formal re-verification 数值（camera corr `0.9476`、dice `0.8328`、foreground hole `0.1245`），但当前工作区未找到对应 artifact，因此这些数值不作为本文件的已复验证据。

### v9.9 迁移交接结论（当前未复验）

迁移摘要报告 v9.9 validation 曾达到 corr `0.9821`、dice `0.9421`、hole `0.0110`，但独立 code 导致类别形态漂移，固定 zero code 的最小 margin 也不足。当前工作区没有对应 checkpoint、metrics 或 `/tmp` artifact；因此这里只记录交接结论，不把这些数值晋级为本地验收事实。维护决策仍是：v9.9 不能作为鲁棒 code-conditioned baseline。

### v9.10 状态

v9.10 新增 label-only template generator 与 `intensity_to_amplitude` command transform。checkpoint 自包含 `[num_classes, 1, H, W]` intensity template bank，接口保留 code 但明确忽略；模板和 D2NN 均冻结。该版本只能称为 `prototype_baseline`。

2026-06-22 的受控 CUDA audit 已完成并通过。冻结参数检查中 generator、encoder/decoder phase 与 amplitude 的 `max_abs_change` 均为 `0.0`；独立 test split 得到 `paired_corr=0.99944`、`paired_dice=0.98052`、`student_high_saturation_ratio=0.005975`，三项 required gate 全部通过。脱敏后的持久证据见 [acceptance_v9_10_label_template_inverse_20260622.json](../archive/acceptance_v9_10_label_template_inverse_20260622.json)。

该结果只接受为 label-only `prototype_baseline`：code conditioning 仍未启用，不能解释为同类多样性或最终 generative student readiness。2026-06-22 这次历史审计的原始运行和 checkpoint 仍是 `/tmp` 记录；当前及后续 v9.10 复验默认写入持久的 `outputs/YYYYMMDD_v9_10/` 管理目录。

### 2026-06-11 v7.2 受保护参考 run

`outputs/20260514_unversioned/runs/20260611_154426_446321_train_one_shot_distill_eval` 使用 phase-only D2NN、encoder/decoder 各两层、`200 × 200` 神经元、`2.5 µm` pitch、`5 mm` 层间距。它保留为后续物理尺寸实验的行为参考，目录及其引用的 student checkpoint 和 teacher dataset 不得删除或回填。

该 run 的 `paired_corr=0.82863`、`paired_dice=0.75583`，但 strict readability 未通过：最差类别 label margin 为 `0.01852 < 0.03`，background speckle ratio 为 `0.08374 > 0.07`。因此它是受保护参考证据，不是已接受基线。

### 双光子打印 100 × 100 尺寸候选

新尺寸候选保留 v7.2 的 phase-only、encoder/decoder 各两层结构和 `200 × 200` CMOS/loss 接口，仅把物理 D2NN 改为 `100 × 100` 神经元、`2 µm` 边长/pitch、`1024` propagation padding。有效口径为 `200 µm × 200 µm`。

按最大半锥角全连接公式计算，`532 nm` 下理论最小层间距为 `1.490 mm`，配置采用 `1.50 mm`。该几何与旧 `200 × 200` 相位 checkpoint 不兼容，后续必须从头训练。

2026-06-24 的 diagnostic-only 延长验证显示，phase-amplitude upper-bound Probe K 达到 strict gate：`paired_corr=0.8755`、`paired_dice=0.8408`、`student_high_saturation_ratio=0.00548`。该结果仅授权 `diagnostic-only/phase-amplitude prototype` sample，不证明 phase-only 双光子打印 ready，也不证明 code-conditioned generative readiness。

同轮 phase-only padded/crop 分支未达 strict：`1.50 mm` Probe L 为 `paired_corr=0.3933`、`paired_dice=0.5288`，`0.75 mm` Probe M 为 `paired_corr=0.4213`、`paired_dice=0.5340`。因此当前 `100 × 100` phase-only 快速验收训练暂停，下一步转向更大 aperture、短 spacing/readout plane 重设、relay/crop 对齐等几何/读出设计诊断。

后续 phase-only geometry/readout 诊断确认，离线 readout sweep 不能单独解释失败：L 最佳 corrected `corr=0.4149`、`dice=0.5772`，M 最佳 corrected `corr=0.4703`、`dice=0.6070`。短 spacing warm-start 有局部改善但未接近 strict：`0.50 mm` Probe N 为 `paired_corr=0.5123`、`paired_dice=0.6163`，`0.25 mm` Probe O 为 `paired_corr=0.4870`、`paired_dice=0.5974`。`150 × 150` aperture upper-bound Probe P 从头 3 epoch 失败：`paired_corr=-0.3256`、`paired_dice=0.1453`。本轮未运行 phase-only sample，结论仍是停止当前 phase-only 快速训练，转硬件几何/读出重设计。

按重新标定的硬约束复核后，2026-06-24 的 full-connect 主线延长只保留 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，并且只从符合该几何的 Probe L checkpoint warm-start。Probe Q 再训 20 epoch 后达到 `paired_corr=0.6483`、`paired_dice=0.6440`、`student_high_saturation_ratio=0.00561`；Probe R 降低 phase LR 到 `1e-4` 再训 20 epoch 后为 `paired_corr=0.4924`、`paired_dice=0.5848`、`student_high_saturation_ratio=0.00562`。两者均未达到 strict gate，也未达到继续延长到 50 epoch 的门槛；未运行 phase-only `one_shot_sample`，未写 accepted summary。结论是当前固定 full-connect phase-only 主线不能靠本轮继续训练完成快速验收。

2026-06-25 的 Probe S-50 是 pre-v15 unversioned milestone：它从 Probe S warm-start，仍严格保持 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，并首次让该 full-connect 几何通过 strict readability gate：`paired_corr=0.8467`、`paired_dice=0.8023`、`student_high_saturation_ratio=0.00576`，crop retention mean 为 `0.8577`。但它的 label audit 回退到 `student_label_correct_ratio=0.50`，失败类集中为 `0→5`、`1→8`、`3→5`、`7→9`、`9→8`；因此只证明几何可读性可行，不授权 accepted summary。后续 100 × 100 phase-only label-semantics 批次从 `v15` 版本名开始，避免继续写入 `unversioned`，也避免混入 2026-05-23 已存在的旧 `v10` grid16 批次。

v15 label-repair 从 S-50 checkpoint warm-start，输出进入 `outputs/20260625_v15/runs/`。Probe U 训练 5 epoch 后保持 strict readability：`paired_corr=0.8540`、`paired_dice=0.8092`、`student_high_saturation_ratio=0.00585`，但 `student_label_correct_ratio=0.60`；Probe V 训练 10 epoch 后为 `paired_corr=0.8557`、`paired_dice=0.8099`、`student_high_saturation_ratio=0.00590`，但 `student_label_correct_ratio=0.50`。两者均未达到 label-repair gate `label_acc>=0.80`，因此未运行 sample，未写 accepted summary。当前结论是 full-connect phase-only 可读性已经稳定通过，但 label semantics 仍不稳定，下一步需要重设计 label audit/readout objective 或使用 alternating label/shape schedule。

2026-06-25 的 v15 pair-contrast 继续保持同一硬约束，并把此前误放入旧 `outputs/20260523_v10/runs/` 和 `unversioned` 的 U/V/W/X/Y 新批次 run 迁入 `outputs/20260625_v15/runs/`。Probe X 是本轮 config-only 最好 label 结果：strict readability 仍通过，`student_label_correct_ratio=0.70`，但 `3→5` 与 `7→9` 仍为零类；Probe Y 提高 `[1,8]` margin 后修复 class 1，却破坏 class 4/9。结论是 pair-contrast 只能局部修复且会转移错误，后续 prototype CE / negative-class CE 必须使用 `v15_probe_z*` 命名，不能再复用旧 `v10`。

v15 prototype CE / negative-class margin 已作为下一步 label-semantics 训练路线准备，但尚未执行 CUDA。该路线新增默认关闭的 `student_prototype_ce_*` 和 `student_negative_label_margin_*` loss 配置，目标是直接优化 label-audit prototype 判别边界，而不是继续调 batch mean pair-contrast。Probe Z1 将从 Probe X warm-start 修复 `[1,3,7,9]`，Probe Z2 将从 pre-v15 S-50 warm-start 修复 `[0,1,3,4,7,9]`；两者都必须继续保持 `100 × 100`、`2 µm`、encoder/decoder `2+2`、`1.50 mm`、`phase_only`，输出只能进入 `outputs/20260625_v15/runs/`。除非 strict readability、`label_acc>=0.80` 且 `[3,7]` 不再为零类同时通过，否则不运行 sample、不写 accepted summary。

后续 v15 prototype CE CUDA 结果仍未突破 label gate：Probe Z1 从 Probe X warm-start 后为 `paired_corr=0.8550`、`paired_dice=0.8114`、`student_label_correct_ratio=0.70`，失败为 `0→5`、`3→5`、`4→9`；Probe Z2 从 S-50 warm-start 后为 `paired_corr=0.8507`、`paired_dice=0.8056`、`student_label_correct_ratio=0.70`，失败为 `1→8`、`3→5`、`7→9`。两者都保持 strict readability，但未过三重 gate，未运行 sample。离线 label-audit boundary 诊断 `outputs/reports/twophoton_100x100_v15_label_audit_boundary_summary_20260625_215412.json` 显示 teacher target 在 class-average audit 下为 `1.0` 正确率，student class-average 仍为 `0.70`，multi-prototype student 降至 `0.30-0.40`，而 ROI-weighted student audit 达到 `1.0` 并修复 `3→5`、`7→9`、`4→9`、`1→8` 等失败对。当前结论是 label semantics blocker 主要来自非 ROI / 背景成分污染 prototype correlation；下一步应实现 ROI-weighted label audit/objective，而不是继续 prototype CE 或 multi-prototype CE。

v15 ROI/readout-calibrated 收口进一步确认 full-image standard label gate 不应继续用 loss 调参追逐。ROI label repair 后，Probe AB 保持 strict readability：`paired_corr=0.8505`、`paired_dice=0.8034`、`student_high_saturation_ratio=0.00558`，standard `student_label_correct_ratio=0.70`，ROI audit 为 `1.0`。随后 Probe AC 从 AB warm-start 做背景抑制，背景 speckle 从 `0.1063` 降到 `0.0981`，同时保持 `paired_corr=0.8532`、`paired_dice=0.8138`、standard `label_acc=0.70`、ROI audit `1.0`；因此 AC 是当前最佳 `diagnostic-only/phase-only ROI-readable label-template baseline` checkpoint 候选。Probe AD 继续加入 weak standard label repair 后虽提高到 `paired_corr=0.8589`、`paired_dice=0.8221`、背景 speckle `0.0920`，但 standard `label_acc` 退化到 `0.50`，作为 standard full-image label repair 会破坏类别平衡的负证据。当前收口结论：停止 v15 standard full-image label-gate 训练；若后续要 sample，必须先显式批准 ROI/readout-calibrated diagnostic sample，使用 AC checkpoint，并继续标记 diagnostic-only，不写 accepted summary。

## 2026-07-03 v16-v31 主线推进复盘

本轮对话当前可见 transcript 中可确认发生过 **1 次上下文压缩**：可见上下文开头存在一次自动生成的 conversation summary。若更早、当前不可见片段中还发生过压缩，现有 transcript 无法验证；迁移到新对话时应写作“可确认 1 次”。

所有 v16-v31.1 结果均保持 `diagnostic-only`，没有写入 accepted summary。最终人眼验收仍只看 v4-style preview 的真实最终 `CMOS intensity`，中间 plane、selected readout、software oracle 或 classifier 可分性均不能替代最终验收。

迁移到新对话的重要事实：

- v16-v17 增加 readout/sample calibration 与 terminal readability 诊断；后处理和读出校准未能恢复人眼可读数字，v16/v17 均保持 `diagnostic-only`。
- v18-v19 转向 front topology / front imaging；SLM/encoder 可读性改善，但后续 D2NN/readout 仍塌缩。
- v20-v21 完成 downstream plane gate 与 physical CMOS readout 隔离；确认 chip/probe 与最终 true CMOS 不能混用，最终验收必须看真实 `CMOS intensity`。
- v22-v23 尝试 physical CMOS semantic separation 与 decoder/chip semantic preservation；指标小幅改善，但 prototype audit 和人眼 topology 仍失败。
- v24-v25 做 OLS/decoder expressivity 与 decoder capacity upper bound；decoder bypass、phase-amplitude、加深 decoder 等都未让最终 CMOS 人眼可读。
- v26 新增 offline OLS software oracle；结论为 `ols-decodable`，`encoded_amplitude` oracle accuracy 达到 `1.0`，说明上游表征可被软件读出。
- v27-v29 聚焦 decoder synthesis / controllability / trainability；训练 harness 正常、参数确实更新，但 decoder/chip 从人眼 topology 角度仍不可达。
- v30-v31 测试 target-code / low-frequency bridge；v30 显示 target-code 有一定 rescue 迹象，但 sample/human gate 失败；v31 real encoded/OLS -> target-code bridge 仍为 `metric-proxy-failure`。
- v31.1 运行 bridge observability same-batch hard gate，三路 `encoded->lowfreq`、`OLS->lowfreq`、`OLS->spatial` 的 `target_code_amplitude` / `decoder_input_amplitude` 均未达到 bridge topology gate，最终仍为 `metric-proxy-failure`。

| 版本 | 主线问题 | 关键效果 | 结论 |
| --- | --- | --- | --- |
| v16 | sample/readout 后处理 | calibration-negative | 停止 v16 loss/config 微调 |
| v17 | readout/camera/OLS | terminal CMOS 仍 blob-like | eval strict 不能代表 sample 人眼可读 |
| v18 | front topology | collapse 起点定位到前/编码链路附近 | 转向 front-path 模型 |
| v19 | front imaging | SLM -> encoder 改善，但后级仍塌 | 瓶颈转到 downstream D2NN/readout |
| v20 | downstream plane gate | 定位 camera-path / downstream failure | chip/probe 与 final CMOS 必须分开 |
| v21 | physical CMOS readout | true CMOS corr/dice 提升但不可读 | readout PSF 不足以解决语义 |
| v22 | physical CMOS semantic | corr/dice 小升，audit 仍约 `0.30` | semantic loss 不够 |
| v23 | decoder/chip semantic | CMOS 指标继续小升，人眼仍失败 | collapse 在 decoder/chip topology |
| v24 | OLS/decoder expressivity | decoder-bottleneck / no rescue | 不继续普通 decoder loss |
| v25 | decoder capacity | no-capacity-rescue | 加容量/phase-amplitude upper bound 仍不够 |
| v26 | OLS software oracle | `encoded_amplitude` oracle accuracy `1.0` | OLS/encoded 有可解码语义 |
| v27 | decoder synthesis bridge | physical-decoder-unreachable | 软件可分不等于物理 topology 可合成 |
| v28 | decoder controllability | metric-proxy-failure | classifier 可分但人眼 topology 失败 |
| v28.1 | same-batch overfit | decoder-topology-unreachable | 同批次也无法强行过 topology gate |
| v29 | trainability/target encoding | harness 正常但仍 metric-proxy-failure | 不是参数没更新，是目标/架构问题 |
| v30 | target-code architecture | frequency-encoding-bottleneck；sample gate 失败 | target-code 有方向，但未过人眼验收 |
| v30.1 | sample gate recovery | eval-positive-sample-failed | eval 正例不能提升为 human accepted |
| v31 | real OLS -> target-code bridge | metric-proxy-failure | bridge 输出需单独可视化和 hard gate |
| v31.1 | bridge observability | same-batch 三路均 `metric-proxy-failure` | 转向 v32 stronger bridge / generator-code diversity / target-code encoding redesign |

v31.1 证据：summary 为 `outputs/reports/twophoton_100x100_v31_1_bridge_same_batch_summary_20260703_104933.json`，bridge topology report 为 `outputs/reports/twophoton_100x100_v31_1_bridge_topology_20260703_104933.json`。三路 same-batch eval run 分别是 `outputs/20260703_v31_1/runs/20260703_105222_241271_train_v31_1_probe_a_encoded_to_lowfreq_bridge_same_batch_e8_same_eval`、`outputs/20260703_v31_1/runs/20260703_105606_359124_train_v31_1_probe_b_ols_to_lowfreq_bridge_same_batch_e8_same_eval`、`outputs/20260703_v31_1/runs/20260703_105951_197276_train_v31_1_probe_c_ols_to_spatial_bridge_same_batch_e8_same_eval`。三路 `same_batch_bridge_gate_passed=false`，未触发 test eval 或 sample，`accepted_summary_written=false`。

当前主线结论：不是继续堆 terminal/sample loss，也不是回到 decoder/chip/camera 微调；v31.1 已显示 bridge 输出本身在 same-batch 上仍不可读，下一步应转向 v32 stronger bridge adapter、generator/code diversity 或 target-code encoding redesign。只有 bridge code 本身可读后，才值得继续判断 decoder/chip/camera 哪一段破坏最终 true CMOS。

## 2026-07-10 v32-v47 主线推进复盘

所有 v32-v47 结果均保持 `diagnostic-only`，`accepted_summary_written=false`，没有 checkpoint 被提升为 accepted baseline。最终验收仍只看 v4-style preview 或 CMOS-only review 中真实最终 `CMOS intensity / physical_camera_plane`；中间 plane、topology proxy、classifier accuracy、same/test eval strict gate 和 software oracle 均不能替代人眼验收。v38 已由人工明确判定 final CMOS 仍为 blob/纹理团块，因此截至 v47 仍没有 accepted summary。

### 关键实现与诊断能力

- bridge 增加 `one_shot.target_code_bridge_arch`，支持兼容旧 checkpoint 的 `shallow`，以及 diagnostic `deep_residual`、`multiscale` adapter。
- generator 增加 `label_template_code_residual`：保留 label prototype base，并由 code-conditioned residual network 产生样本变化；eval/sample tensor bundle 同步加入 `generator_base_command`、`generator_residual_map`、`generator_residual_abs_mean` 和背景 residual 统计。
- target-code 依次增加 `thick_lowfreq_digit_code`、`sharp_stroke_digit_code`、`small_sharp_stroke_digit_code`，并增加 `target_code_spatial_scale` 以显式控制前景占比和笔画空间尺度。
- 训练目标增加默认关闭的 `sample_code_diversity_weight`；只作为 sample/code diversity 诊断训练项，不能替代 final CMOS 人眼 gate。
- sample 路径增加 diagnostic-only low-frequency SLM command override，用于直接测试 front encoder/OLS 是否能够承载人为注入的 code variation。
- decoder 增加 `propagation_only_decoder`，与既有 `bypass_decoder` 配合，将 decoder mask、层间自由传播和 back path/CMOS 分开隔离。
- artifact/report 体系增加 residual observability、plane shape、同标签 diversity、class correlation、top confused pairs、CMOS-only review、topology proxy v2 和人工 verdict。v37.1 修复中间 plane `100 x 100` 与 final CMOS `200 x 200` 被错误要求同尺寸的 harness 问题。
- 新增 v32.1-v47 离线诊断工具及对应 persistent launcher/config；报告统一写入 `outputs/reports/`，长 CUDA 训练写入真实版本目录 `outputs/YYYYMMDD_<version>/runs/`。

### 主线推进过程

#### v32-v33.2：bridge 与 code-conditioned generator

- v32 以 `deep_residual`/`multiscale` bridge 和 `thick_lowfreq` target-code 做 stronger bridge upper bound。Probe A/B 在 same/test eval 上出现 `bridge-code-candidate-needs-human-review`，但 sample 仍失败。
- v32.1 no-retrain sample-gap isolation 将失败定位为 `label-template-reference-mismatch`：eval/generated 与 sample/generated 近似固定于 label prototype，而 random sample reference 是另一组真实数字；同时记录 `code_insensitive=true`。
- v33 引入 `label_template_code_residual`，但 A/B/C 因 bridge-only trainability audit 与 `target_code_bridge_source=none` 冲突而无可训练参数；D 只更新 bridge、没有更新 residual。该轮属于 harness-invalid/`residual-inactive` 证据，不能作为模型结论。
- v33.1 关闭错误的 bridge-only audit，并补齐 residual tensor observability。A/B/C/D eval/test 均达到 `code-conditioned-target-rescue`，证明 residual harness 和 code-conditioned target variation 可以工作；C sample 的 arbitrary reference exact metrics 被正确标记为 `sample-reference-metric-invalid`。
- v33.2 强制对 real-OLS Probe D 做 no-retrain sample，结果为 `sample-code-distribution-gap`：code 和 residual 有变化，但从 SLM/target-code 开始同标签输出仍近似固定。

#### v34-v36：sample-time code coupling

- v34 对 `code_residual_scale=0.05/0.10/0.20/0.50` 做 no-retrain sweep。`0.50` 已使 SLM command variation active，但 encoded/OLS/target-code/CMOS 仍近似固定，报告状态为 `command-to-ols-bottleneck`。
- v35 将 generator gain 提高到 `1.0/2.0/4.0`，并加入 direct-SLM low-frequency perturbation oracle。`scale=2.0/4.0` 与 direct-SLM oracle 均能把 code variation 传到 `physical_camera_plane`，说明 front/OLS carrier 在足够强的 command variation 下可工作；最终状态为 `code-coupling-recovered-needs-human-review`，但 CMOS 仍不可读。
- v36 将 high gain 正式纳入 residual+real-OLS bridge 训练。三路均为 `high-gain-training-rescue-needs-human-review`，code coupling 已到达 final CMOS，但最终 preview 仍偏纹理团块，不能 accepted。

#### v37-v40：人工 gate、false positive 与 sharp target

- v37 最初因错误要求 `target_code/decoder` 的 `100 x 100` plane 与 `physical_camera_plane` 的 `200 x 200` shape 一致而得到 `harness-invalid`，不是模型失败。
- v37.1 允许各 plane 独立归一化和独立 topology 计算，只要求 batch/sample 数一致。A/B/C 均成为 `manual-cmos-review-candidate`，但该状态只授权人工审查。
- v38 生成只展示 true final CMOS 的正式 review package。人工 verdict 为 final CMOS 仍是 blob/纹理团块，未通过验收。
- v39 将旧 topology proxy positive、人工失败正式记录为 `topology-proxy-false-positive`，并确认同标签与跨 probe 输出高度同源。v39 曾把局部现象报告为 `camera-texture-amplification`，但该定位随后被 v42 bypass 和 v45 near-field 证据进一步收敛：camera/back path 可以显示数字，当前根因不在 camera。
- v40 增加 `sharp_stroke_digit_code` 和 sample-code diversity loss，状态为 `label-template-collapse-recovered`。sharp target/decoder input 已形成更明确数字，但 final CMOS 仍未通过人眼验收；主线因此转向 decoder physical reachability。

#### v41-v45：decoder/back-path 与传播几何

- v41 对 direct sharp target 测试 phase-only decoder、physical-CMOS loss、phase-amplitude upper bound 和 4-layer decoder，四路均为 `decoder-physical-unreachable`。断点收敛到 `decoder_input -> decoder_output -> physical_camera_plane`。
- v42 no-retrain decoder isolation 显示 `bypass_decoder` 路径下 true final CMOS 可以显示可读数字，而 `propagation_only_decoder` 仍退化为光斑。该证据排除了 back path/CMOS 无法显示数字，首要问题转为 decoder propagation geometry、spacing、aperture 或 target scale。
- v43 对 propagation-only decoder 做 spacing、padding 和 target scale sweep。增大 propagation grid 没有形成主要改善，缩小/变细的 `small_sharp` target 给出正信号，报告为 `target-scale-bottleneck`。
- v44 将 `small_sharp_stroke_digit_code` 和显式 `target_code_spatial_scale` 推进到完整 non-bypass decoder 训练。`scale=0.72` 优于 `0.60`；`0.50 mm` short-spacing Probe E 明显优于 `1.50 mm`，状态为 `target-scale-decoder-recovered-needs-human-review`，但仍未人眼通过。
- v45 固定 direct small-sharp target，扫描 `0.25/0.50/0.75/1.00 mm`。`0.25 mm` Probe A 是当前最佳 direct-decoder 几何：decoder topology proxy `0.851`、final CMOS proxy `0.708`、decoder-vs-target corr 约 `0.877`、CMOS-vs-decoder corr 约 `0.954`。短 spacing 明显优于其他 spacing，但 final CMOS 仍有纹理和类别混淆。

#### v46-v47：near-field 语义分离与 capacity upper bound

- v46 固定 `0.25 mm`，尝试 longer training、cross-class pairwise loss、hard-pair contrast 和 final-CMOS topology/contrast loss。四路训练均健康、decoder gradient 非零，但最大类间相关仍约 `0.912`，最终状态为 `semantic-collapse-still-high`。
- v47 从 v40 direct-sharp checkpoint 分别测试 2-layer phase-amplitude、4-layer phase-only、`propagation_grid_size=1536` 和 phase-amplitude physical-CMOS loss。四路 trainability audit 均通过，上游保持冻结，但 strict test 全部失败：paired corr 约 `0.501-0.508`、paired dice 约 `0.552-0.558`，类别 `0/3/5` label audit 正确率均为 `0`。
- v47 A/C/D final CMOS 高度同源；A-C corr 为 `0.999999`，且 decoder crop retention 均约 `0.9998`，因此 `1536` padding/crop 可基本排除为当前主因。`physical_camera_plane` 与 decoder output corr 约 `0.95`，CMOS 主要继承 decoder 产生的纹理形态。
- v47 最终状态为 `capacity-still-collapsed`。但 phase-amplitude 和 4-layer 结论保留利用率限制：A/D decoder mask amplitude 均值约 `0.982`、空间 std 仅 `9e-5` 至 `1.7e-4`，幅度自由度实际接近未激活；B 新增 L3/L4 phase std 约 `0.032`，明显低于 L1/L2 的 `0.28-0.36`，且总训练只有 36 optimizer steps。因此只能否定当前配置，不能宣称 amplitude modulation 或 decoder depth 已被充分穷尽。

| 版本 | 关键状态 | 主线结论 |
| --- | --- | --- |
| v32 | `bridge-code-candidate-needs-human-review` | stronger bridge 有 eval 正信号，但 sample 未通过 |
| v32.1 | `label-template-reference-mismatch` | eval/sample target 与 arbitrary reference 不一致，generator 仍 code-insensitive |
| v33 | `residual-inactive` / harness-invalid | bridge-only audit 使 residual 未被训练，不能下模型结论 |
| v33.1 | `code-conditioned-target-rescue` | 修复 harness 后 residual active，code-conditioned target 可工作 |
| v33.2 | `sample-code-distribution-gap` | sample variation 未稳定传到 SLM/target-code |
| v34 | `command-to-ols-bottleneck` | scale 0.50 激活 command，但 downstream variation 仍弱 |
| v35 | `code-coupling-recovered-needs-human-review` | scale 2/4 与 direct-SLM oracle 证明 carrier 可承载 code variation |
| v36 | `high-gain-training-rescue-needs-human-review` | 训练后 variation 到达 CMOS，但最终仍 blob-like |
| v37 | `harness-invalid` | 100/200 plane shape gate 错误 |
| v37.1 | `manual-cmos-review-candidate` | 修复 gate 后 proxy positive，只授权人工审查 |
| v38 | `manual-review-required`；人工失败 | true final CMOS 仍为 blob/纹理团块 |
| v39 | `manual-review-failed-final-cmos-blob` | 旧 topology proxy 为 false positive，存在同源/模板 collapse |
| v40 | `label-template-collapse-recovered` | sharp target 恢复类差异，但 final CMOS 仍未过人眼 gate |
| v41 | `decoder-physical-unreachable` | direct target 可读，完整 decoder 输出不可达 |
| v42 | decoder bypass positive | back path/CMOS 能显示数字，传播式 decoder 是首要问题 |
| v43 | `target-scale-bottleneck` | smaller/sharper target 有正信号，padding不是主因 |
| v44 | `target-scale-decoder-recovered-needs-human-review` | small target 有改善，0.50 mm 优于 1.50 mm |
| v45 | `near-field-spacing-recovered-needs-human-review` | 0.25 mm 是当前最佳 direct-decoder spacing |
| v46 | `semantic-collapse-still-high` | longer/semantic/final-plane losses 未拉开类别 |
| v47 | `capacity-still-collapsed` | 当前 phase-amplitude/depth/padding upper bounds 均未救回人眼可读性 |

### 代表性持久证据

- v32 bridge summary：`outputs/reports/twophoton_100x100_v32_bridge_redesign_summary_20260703_150903.json`
- v32.1 sample-gap report：`outputs/reports/twophoton_100x100_v32_1_sample_gap_isolation_20260703_154715.json`
- v33.1 harness-fix summary：`outputs/reports/twophoton_100x100_v33_1_code_conditioned_target_summary_20260703_203645.json`
- v35 code-carrier report：`outputs/reports/twophoton_100x100_v35_optical_code_carrier_isolation_20260703_215703.json`
- v38 formal CMOS review：`outputs/reports/twophoton_100x100_v38_formal_cmos_manual_review_20260706_170459.json`
- v39 false-positive report：`outputs/reports/twophoton_100x100_v39_human_topology_false_positive_20260707_142536.json`
- v42 decoder-bypass summary：`outputs/reports/twophoton_100x100_v42_decoder_bypass_backpath_isolation_summary_20260707_171439.json`
- v45 near-field report：`outputs/reports/twophoton_100x100_v45_decoder_spacing_near_field_recovery_20260709_151115.json`
- v46 semantic report：`outputs/reports/twophoton_100x100_v46_near_field_semantic_separation_20260709_195630.json`
- v47 capacity summary/report：`outputs/reports/twophoton_100x100_v47_near_field_decoder_capacity_upper_bound_summary_20260710_050842.json`、`outputs/reports/twophoton_100x100_v47_near_field_decoder_capacity_upper_bound_20260710_050842.json`

### 当前断点与下一 gate

当前可确认链路状态为：

`code -> generator residual -> SLM command -> target_code -> decoder_input` 已能形成清楚数字；real OLS bridge 曾显示 sample/code distribution 问题，但 v44-v47 为隔离 decoder 而使用 direct target，尚未重新接回完整 real-OLS inference path。当前首要断点是 `decoder_input -> decoder_output`；`physical_camera_plane` 以约 `0.95` correlation 继承 decoder output，结合 v42 bypass positive，camera/readout 不是当前根因。

下一步不应继续普通 loss、padding 或 camera 微调。推荐先做 decoder capacity-utilization audit，明确 amplitude mask 和新增 decoder layer 是否真正被激活；随后执行一个真正利用 amplitude/新增层的 complex-mask 或 non-propagation decoder upper bound。若该 upper bound 通过，主线回到可制造 modulation/传播式 decoder architecture；若仍失败，再转向更大 aperture、pixel size 或传播网格/物理尺度重设计。即使 direct decoder 后续通过，也仍必须单独恢复 real OLS bridge，并在完整单曝光链路上重新进行 true final CMOS 人眼验收。

## 2026-07-11 v48 transparent-decoder 对照

v48 使用同一个 v47 Probe A checkpoint、相同 `small_sharp_stroke_digit_code`、`target_code_spatial_scale=0.72` 和 true `physical_camera_plane`，对比 `0.10 mm` transparent、`0.25 mm` transparent 与 `0.25 mm` trained mask。最初的 `20260710_213539` 和 `20260711_020721` 报告因输入 metadata 不完整而是 `harness-invalid`，不构成模型证据；修复后的 fixed report 无 harness error，并生成十类 CMOS-only review。

人工 verdict 为 `t025_readable_m025_unreadable`，最终状态 `decoder-mask-induced-artifact`：T025 transparent 人眼可读，而同层距 M025 trained mask 不可读。自动指标只用于解释：T025 `decoder_output_vs_target corr=0.9557`、`physical_vs_decoder corr=0.9695`，M025 分别为 `0.8783` 和 `0.9536`。所有结果仍为 `diagnostic-only`，`accepted_summary_written=false`；0.25 mm 低于正式 1.50 mm 全连接几何，不能 promotion。

- fixed report：`outputs/reports/twophoton_100x100_v48_transparent_decoder_control_20260711_020721_fixed.json`
- 人工 verdict：`outputs/reports/twophoton_100x100_v48_transparent_decoder_control_20260711_020721_verdict.json`
- localized issue：trained decoder mask，而不是 0.25 mm transparent propagation 或 camera/back path。
- ruled-out causes：T025 条件下的纯传播、最终 relay/CMOS 无法显示数字。
- still-plausible causes：decoder phase、近 unity amplitude、phase-amplitude interaction 或具体责任层。
- gate validity：fixed/verdict report 无 error，十类 true CMOS 人工审查有效；早期两个 harness-invalid report 无效。
- next gate：v49 无训练 P025/A025/L1P025/L2P025 mask attribution；在人工 verdict 前不得启动 v50。

## 2026-07-11 至 2026-07-14 v49-v57：mask 归因、严格复场与物理 relay 合同

本节记录当前主线从 v48 之后的事实。所有 v49-v57 结果仍为 `diagnostic-only`，`accepted_summary_written=false`。真实最终验收始终只看 `physical_camera_plane` 的 CMOS intensity；复场/振幅/相位诊断、proxy、eval 指标和人工对中间 plane 的判断都不能替代该 gate。

### v49-v50：0.25 mm direct-decoder mask 归因（历史 diagnostic）

- v49 在同一个 v47 Probe A checkpoint、相同 target/seed/0.25 mm 下复用 transparent `T025` 和 trained `M025`，新增 `P025`（仅训练后 phase）及 `A025`（仅训练后 amplitude）。人工 verdict 为 `phase_only_unreadable`，状态为 `phase-mask-induced-artifact`：`T025/A025` 可读，而 `P025/M025` 出现同源横纹/碎裂；`P025–M025` final CMOS correlation 约 `0.9999996`，`A025–T025` 约 `1.0`。当时 amplitude 只形成约 `0.9306` 的近似均匀强度缩放，不能解释主要 artifact。
- v50 增加 `decoder_diagnostic_phase_layer_mask`，分别生成 `L1P025=[true,false]` 与 `L2P025=[false,true]`。该报告仍停在 `manual-review-required`，没有完整的人工 verdict，故不得归因到 L1、L2 或它们的 interaction。
- 以上 0.25 mm/direct-target probe 低于正式 1.50 mm 全连接合同；它们只说明旧合同下 trained decoder phase 曾是 artifact 来源，不能外推到现行正式 relay 路径。

代表证据：`outputs/reports/twophoton_100x100_v49_decoder_mask_component_attribution_20260711_031107_verdict.json`、`outputs/reports/twophoton_100x100_v50_decoder_phase_layer_attribution_20260711_034458.json`。

### v51：严格数字孪生复场可视化修复

- 停止将 SLM command、target code、mask 参数、人工零相位、CMOS intensity 或逐面归一化 display tensor 称为“光学振幅/相位”。历史 v48-v50 的完整 physical-chain amplitude/phase 图被追溯标记为 `visualization-harness-invalid`，历史 run 不改写。
- 正式光场图唯一源自同一传播复场 `E` 的原始 `real/imag`：振幅为 `abs(E)`（a.u.，共享线性色标），相位为 `angle(E)`（固定 `[-pi, pi]` rad）；PNG 只在峰值强度以下 -60 dB 处遮罩相位，未遮罩 raw tensor 仍保存。
- intensity-only relay 之后不再伪造 camera phase。mask transmittance 仅作为 device-mask diagnostic，明确不是传播光场。
- v51 sample 证明了该 artifact/provenance 接口可写入，但它仍是旧光路/旧 checkpoint 下的诊断证据，不能证明现行完整物理链。

代表证据：`outputs/reports/v51_legacy_optical_field_visualization_correction_20260711_102325.json`、`outputs/20260711_v51/runs/20260711_132217_854523_train_v51_real_complex_field_probe_a_s025_sample/`。

### v52-v54：物理坐标合同与旧前端 aperture 失败定位

- v52 建立正式物理 sampling metadata：每个严格复场 plane 必须携带 shape、pitch、FOV、aperture、origin、wavelength、upstream、distance、magnification 与 operation；正式 OLS 固定为复恒等映射 `Eout=Ein` 并通过线性叠加测试。generic complex resize 被正式禁止。
- 当时的 v52/v53 前端曾采用“自由传播后从 500 um FOV 中截取中央 200 um aperture 再得到 encoder sampling”的解释。它的 CUDA preflight 通过的是该旧合同的数值一致性，并不等于其为合适的实际光路模型。
- v53 在该旧合同下从零训练一次 5 epoch：validation loss `0.231066`，camera corr `0.5728`，dice `0.6253`，diversity ratio 约 `2.59e-7`；final CMOS 仍同源、不可读。v54 只读复场诊断曾报告 `macro-front-aperture-first-collapse`，其候选点正是旧 `lightgen_input` 中央截取。这是随后替换前端模型的依据，不是当前 relay 的结论。

代表证据：`outputs/reports/v53_cuda_physical_preflight.json`、`outputs/reports/v54_first_collapse_complex_field.json`。v53/v54 均已被 v55 前端 relay 合同取代，旧 checkpoint 禁止加载到新合同。

### v55-v55.1：正式相干缩小 relay 合同

现行正式理想数字孪生光路为：SLM 完整 `500 um` FOV 经相干 4f 缩小成像到 encoder 完整 `200 um` FOV，禁止中心截取。

| Relay | 对象与目标 | 倍率 | 设计参数 |
| --- | --- | ---: | --- |
| 前端 4f | SLM `200x200 @2.5 um` -> encoder `100x100 @2 um` | `-0.4` | `f1=223.214 mm`，`f2=89.286 mm`，clear aperture `>=23.75 mm` |
| 后端有限共轭 | chip `200 um` -> CMOS `500 um` | `-2.5` | `u=100 mm`，`f=71.4285714 mm`，`v=250 mm`，clear aperture `>=26.60 mm` |

- 前端机械折转距离保持 SLM->BS->M3 `150.000 mm`、M3->L1 `73.214 mm`、L1->L2 `312.500 mm`、L2->encoder `89.286 mm`，总计 `625 mm`；ABCD 为 `A=-0.4, B=0`。
- v55 CUDA structural preflight 通过，确认 sampling/FOV/ABCD/provenance/无 crop 路径可完整 forward：`outputs/reports/v55_coherent_relay_cuda_preflight.json`。
- v55.1 原报告曾标记 reference validation 通过，但其参考实现与生产 B=0 采样同构，现降级为实现自洽证据。v55.2 的自适应 scaled-Fresnel/direct-RS 参考仍因中间面离散采样不足返回 `reference-harness-invalid`。v58.2 的 CUDA 数值自洽运行保留为历史证据：它使用 `128→64`、未实测 Airy PSF，不能作为正式验收。v58.3 才以真实 `A=-0.4, B=0, C=0`、完整 `200→100` sampling、闭式 B=0 场、实测 pupil PSF 和故障注入审计签发 gate；实际尺寸低分辨率 direct-RS 继续排除出 acceptance。pupil/NA 仍是设计下限，不是已实测硬件参数。

### v56-v57：现行 relay 合同的一次从零训练与首塌缩定位

- v56 在当时 v55.1 结构/自洽门通过的配置上只做了一次从零、5 epoch、seed 1234 的 CUDA 训练，复用 readable-split teacher 数据，仅保留 camera reconstruction/distillation objective；没有加载任何 v47/v51/v53 checkpoint，也没有增加 loss、sweep 或 override。v55.2 重新独立验证未通过，因此 v56 继续仅作为 diagnostic 历史。
- 训练 best checkpoint 为 epoch 5：validation loss 从 `0.2715` 降至 `0.2276`，paired corr 从 `0.4913` 升至 `0.5870`，paired dice 从 `0.5704` 升至 `0.6351`，但 diversity ratio 从 `0.00146` 降至 `0.00095`，cross-class ratio 从 `0.000138` 降至 `0.000103`。正式 test eval 为 corr `0.5789`、dice `0.6298`、diversity ratio `0.000968`，strict readability 未通过。
- 同一个 epoch-5 best checkpoint 的 class-balanced sample（0-9、每类两张）具有 `field_chain_status=valid`；真实 `physical_camera_plane` 仍是同源圆斑/纹理，人眼不可读。因此这是有效失败，不是 harness failure。
- v57 新增只读、流式 pairwise 复场诊断。它保留 complex `real/imag`，先为每对样本去掉全局相位，再计算 amplitude、complex 与 intensity 的同类/跨类中位距离分离。初版发现过 complex tensor 被 `.double()` 时丢虚部的问题；已立即修正为 `complex128` 并重新运行，当前报告中的复场/相位指标有效。
- v57 结果没有满足“至少两类分离指标低于 SLM 的 10% 且在后续平面持续”的局部首塌缩门。相对于 SLM 的跨类减同类分离，前端 relay 保留约 `47%`，encoder/OLS/decoder/chip 的 complex 分离仍约 `39%-49%`，camera intensity 约 `11.7%`；OLS 输入输出连续。最终决策为 `aperture-throughput-limited-no-localized-collapse`，不是某一层已证实的语义首塌缩。
- 数值传播每层能量比约 `0.999999`，但有限 metasurface aperture crop 保留率依次为 encoder L1 `25.70%`、encoder L2 `27.65%`、decoder L1 `30.83%`、decoder L2 `33.42%`，累计为 `0.7322%`。这说明有高吞吐损失，但其在类别间近似相同；目前只能列为主导候选，不能直接宣称它是类别塌缩根因。

代表证据：

- train：`outputs/20260714_v56/runs/20260714_204958_697374_train_v56_coherent_relay_from_scratch/`
- eval：`outputs/20260714_v56/runs/20260714_210206_240133_train_v56_coherent_relay_eval/artifacts/one_shot_distill_eval_metrics.json`
- sample/provenance：`outputs/20260714_v56/runs/20260714_210416_032022_train_v56_coherent_relay_sample/`
- v56 preflight：`outputs/reports/v56_coherent_relay_cuda_preflight.json`
- v57 corrected report：`outputs/reports/v57_first_collapse_complex_field.json`

### 当前状态与下一决策门

当前正式合同是：`laser -> M1 -> BS -> amplitude SLM -> BS -> M3 -> coherent 4f relay -> encoder(2) -> complex identity OLS -> decoder(2) -> finite-conjugate relay -> CMOS intensity`。SLM/CMOS 为 `200x200 @2.5 um`、500 um FOV；encoder/decoder 每层为 `100x100 @2 um`、200 um FOV、1.50 mm spacing、phase-only。fiber 尚未设计或实现。

下一轮在 v57 结论前不得继续训练、延长 epoch、增加 diversity/semantic loss，也不得把 aperture/pupil 直接改为新的正式硬件合同。应先设计一个明确标注 `diagnostic-only` 的 aperture-throughput broad-isolation upper bound：在同一 v56 checkpoint、相同输入与相同 formal relay 下，对有限 metasurface crop 的反事实保留进行离线/无训练比较，并同时报告类别分离与能量吞吐。只有它能把类别分离或最终 true CMOS 的人眼信号恢复，才可将 aperture 认定为可操作的根因；否则应转向 upstream generator/objective 的低分离问题。无论结果如何，当前 v56 不得 promotion。

### v58-v59：aperture upper bound 与 upstream separation broad isolation

- v58 使用同一 v56 best checkpoint、同一 class-balanced sample、相同前后 relay、零训练，唯一反事实变化是移除 D2NN 层间 finite crop；训练好的 `100x100` mask 保持居中，外部扩展域为 identity。chip 能量由 `126.18` 增至 `17232.99`，camera field 能量仅由 `66.27` 增至 `69.78`；最终 CMOS separation gain 仅为 amplitude `1.0227x`、complex `1.0173x`、intensity `1.0101x`。人工复核 baseline/upper-bound 都是不可读圆斑/纹理。结论为 `aperture-upper-bound-no-separation-recovery`，不是 harness failure。
- v59 使用同一正式 v56 config/checkpoint，在 `readability_strict_split` validation 数据上做 100 样本（每类 10 张）的三组零训练 forward：paired baseline、label 循环移位、同类 code 循环移位。报告确认 identity OLS、coherent 4f/free-space relay、正式 D2NN geometry 与参数零变化。
- teacher target prototype CV `1.00`、permutation `p=0.00775`；SLM command 与前端 4f 输出 prototype CV 分别 `0.94/0.94`、均显著高于 permutation null。SLM complex separation 仅为 teacher 的 `0.6787%`，但 4f 相对 SLM 仍保留 `41.0%`，因此不是“4f 首塌缩”。SLM 类间频谱约 `80.2%` 落在形式通带外，但前端 separation 没有跌到 SLM 的 `10%` 以下，故不满足 bandwidth mismatch gate。
- label-shift 与 code-rotation 对最终 target MSE 的相对变化分别为 `0.155%` 与 `0.0036%`；最终 true CMOS 仍不可读。当前决策为 `objective-representation-mismatch-candidate`，下一门是 offline decodability upper bound；不得继续 decoder/loss 微调或 promotion v56。

代表证据：

- v58：`outputs/reports/v58_aperture_throughput_upper_bound.json`、`outputs/reports/v58_aperture_throughput_upper_bound.pt`
- v59：`outputs/reports/v59_upstream_separation_isolation.json`、`outputs/reports/v59_upstream_separation_isolation.pt`、`outputs/reports/v59_upstream_separation_isolation_preview.png`

### v60-v61：复制训练、稳健首塌缩审计与离线可解码上界

- v60 使用 v56 相同配置、seed `1234`、5 epoch 和正式物理合同完成复制训练。eval corr/dice 为 `0.5733/0.6235`，sample corr 为 `0.4914`，均未超过 v56；最终 CMOS 仍不可读。v58.3 只加固 relay 验收和 preflight，不改变正式传播算子。
- v61 robust tool 冻结 v60 best checkpoint，对 readable train/val/test `800/100/100` 做完整 formal forward；缓存包含所有严格复场 plane、CMOS intensity、provenance 和 parameter hash。使用固定 32×32 诊断网格、2,000 次 pair bootstrap 与 permutation null；前后参数 hash 一致，报告结论为 `no-robust-localized-collapse`。
- 因此 v60 的 20-sample camera intensity retention `0.0981` 仍是阈值边缘证据，不能推翻 v59 的 100-sample broad-isolation 结论。早期 `tools/diagnose_v61_offline_decodability.py` 是历史软件 readout：它使用降采样/PCA 与额外对照，不能充当下述原生采样上界的替代或正式单曝光读出。
- v61 offline report 的历史 linear plane probe 在 test 上得到 corr `0.874-0.892`、dice `0.676-0.710`，但相对 label-only paired corr/dice delta 为负；code-only/code+label corr 仅 `0.716/0.731`。固定 nonlinear probe 的三个 seed 未达到 strict gate，且部分 seed 输出零多样性。该结果支持“plane 仍有类别模板/低阶信号，但没有证明 paired instance-level CMOS 生成”，不授权继续堆 loss 或 promotion。
- 随后的 v61 native dual-channel upper bound 冻结 v60 epoch-5 checkpoint，以 readable train/val/test `800/100/100` 提取真实 chip/camera fields；无 optimizer、`state_dict` 前后无变化、无 target-code/bridge/override。特征保持 native sampling：chip `100x100 @2 um`，camera/teacher target `200x200 @2.5 um`；逐批验证 `physical_camera_intensity_raw == |E_camera|²`，而 probe 使用真实最终 detector-normalized `physical_camera_plane`。它没有 resize、PCA、神经网络或保存完整复场 bundle。
- 四个严格 held-out linear ridge probes 只用 train 拟合、只由 val 的 raw MSE 从 `{1e-4,1e-3,1e-2,1e-1,1}` 选择、再 train+val 重拟合并仅 test 一次。chip intensity / chip complex / camera intensity / camera complex 的 test `(corr, dice, prototype-accuracy)` 分别为 `(0.9453, 0.8829, 1.00)`、`(0.9456, 0.8835, 1.00)`、`(0.9452, 0.8829, 1.00)`、`(0.9456, 0.8835, 1.00)`；均选 `lambda=0.1`。按预注册门，结论为 `camera-intensity-decodable`。
- 这不是 raw CMOS 人眼 gate 通过，也不是可部署的 optical readout：preview 中的数字来自使用 900 个配对 teacher target 拟合的离线软件映射。complex oracle 相对 intensity 无实质增益，chip/camera intensity 均达标，故本结果不支持仅 phase-to-intensity 或仅 backend relay 失配。它也尚未区分逐实例信息与类别模板；下一门只能做零训练 native conditional-control audit（label-only、code-only、within-class target permutation、逐实例残差），不得调整 decoder/loss 或 promotion。

代表证据：

- `outputs/20260715_v61/runs/robust_collapse/v61_robust_collapse.json`
- `outputs/20260715_v61/runs/robust_collapse/v61_frozen_optical_cache.pt`
- `outputs/20260715_v61/runs/offline_decodability/v61_offline_decodability.json`
- `outputs/reports/v61_offline_decodability_upper_bound.json`
- `outputs/reports/v61_offline_decodability_upper_bound.pt`
- `outputs/reports/v61_offline_decodability_upper_bound_preview.png`
### v58：现行合同去旧化与独立 relay 重验（进行中）

- formal validator 已收紧为当前 v55 coherent 4f 前端、identity OLS、1.50 mm D2NN 和后端薄透镜 B=0 relay；v52/v53 free-space+aperture 配置保留为 legacy diagnostic，不再作为 formal contract。
- 前端 provenance 正在改为具名的 `SLM→BS→M3→L1→L2→Encoder` 距离链。原 v55.1 reference 使用与生产实现同构的插值/相位表达，现降级为实现自洽证据；v55.2 需要独立逐段 Fresnel/Collins 与低分辨率 direct-RS 4f 参考通过后，才能恢复 formal-valid 状态。
- 本轮合同重验不训练、不重复运行 v58 aperture upper bound，也不修改历史 outputs 或 checkpoint。

### v62：native conditional-control audit（2026-07-15）

- 在冻结 v60 epoch-5 checkpoint、正式 v56 config、同一 readable split 和 v58.3 relay gate 上运行一次零训练 CUDA audit。train/val/test 为 `800/100/100`，参数 hash 前后相同，`optimizer_created=false`，所有结果仍为 `diagnostic-only`。
- 原生 forward 对照包括 baseline、固定 readable-train code 的 label-only、固定 label `0` 的 code-only；target 对照为每类一位确定性循环置换。v61 已封存的 camera-intensity held-out ridge prediction 直接复用，未重新拟合。
- label-only prototype CV accuracy `1.00`、permutation `p=0.00775`，类内 residual energy `3.14e-15`；baseline accuracy `0.94`，code-only accuracy `0.23`（同样 `p=0.00775`），但 code-only 类内 residual energy 仅 `5.09e-5`。baseline→label-only camera response `1.040%`，baseline→fixed-label code-only `0.389%`。
- v61 offline camera-intensity prediction 对 paired teacher target 的 corr 为 `0.9452`，within-class target rotation 后 `0.8909`，paired delta `0.0544`（95% CI `[0.0469,0.0628]`）。leave-one-out class-template residual corr 为 `0.4931`，旋转 target 后 `-0.0890`，paired residual delta `0.5821`（95% CI `[0.5321,0.6325]`）。因此上界包含逐实例 residual，但仍不是 raw CMOS 人眼 gate。
- 当时决策写为 `paired-instance-residual-with-weak-explicit-code-control` / `manual-review-required`。该解释已被后续 v63 修正：code-only classifier 用原始 readable labels 分组，readable filter 对 IID code 造成 code-label selection confound；且 `baseline→label-only` 是替换 code、保持 label 的响应。因此 `0.23` classifier 和固定 `2%` 门不能用作“code 弱”或根因结论。已封存 paired-residual 对照仍是有效的关联证据，但不授权修改 generator、decoder、loss 或硬件合同。

代表证据：

- 工具：`tools/diagnose_v62_native_conditional_control_audit.py`
- 报告：`outputs/reports/v62_native_conditional_control_audit.json`
- tensor：`outputs/reports/v62_native_conditional_control_audit.pt`
- preview：`outputs/reports/v62_native_conditional_control_audit_preview.png`

### v63：code-conditioning provenance 与 crossed-factor audit（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay、同一 readable `800/100/100` split 与 v61/v62 工件，执行一次零训练 CUDA audit。identity OLS、正式 geometry/field chain、camera intensity 语义、参数快照、code replay 与 teacher-noise replay 全部有效；`parameter_update_count=0`、`optimizer_created=false`、`diagnostic_only=true`。
- 全局 `code_seed=1234` 的 15k IID 导出 code prototype CV 为 `0.09713`，相对于 permutation null 的 `p=0.94118`。按 readability filter 选择后的 train/val/test 分别为 `0.3225/0.1700/0.2200`，对应 p 值 `0.00775/0.02326/0.00775`。这是筛选造成的 code-label selection confound，不能当作模型 code-conditioning 能力或训练许可证。
- 在 100 held-out test code × 10 labels 的平衡网格上，SLM command 的 code/label/interaction 方差份额为 `80.62%/16.07%/3.31%`；pre-encoder intensity 为 `80.51%/16.13%/3.35%`；chip intensity 为 `79.52%/17.71%/2.77%`；最终 detector-normalized CMOS 为 `80.39%/15.55%/4.06%`。因此正式光路没有把 code 因子衰减到 SLM 的 10% 以下，且 label 因子没有形成局部保留而 code 丢失的模式。
- 预注册门给出 `generator-code-dominant` 和 `native-code-conditioned-subvisual-representation-confirmed`：generator 的 code 因子很强，并保持到 final CMOS，且 v62 paired-residual 对照仍有效。它不改变 raw final CMOS 人工不可读的硬 gate；结论是原生 code-conditioned representation 为 subvisual，下一门只能进行 representation/readability isolation，不得重新微调 decoder/loss、修改 optical contract、训练或 promotion checkpoint。

代表证据：

- 工具：`tools/diagnose_v63_code_conditioning_provenance_audit.py`
- 报告：`outputs/reports/v63_code_conditioning_provenance_audit.json`
- tensor：`outputs/reports/v63_code_conditioning_provenance_audit.pt`
- preview：`outputs/reports/v63_code_conditioning_provenance_audit_preview.png`

### v64：semantic-band carriage 与 readability isolation（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay、同一 readable `800/100/100` split 及 v61/v63 工件，执行零训练 CUDA audit。15k code/noise provenance、identity OLS、formal geometry/field chain、camera `|E|²`/detector normalization 和参数快照均通过；`parameter_update_count=0`、`optimizer_created=false`，临时 native memmap 在诊断后删除。
- 以 camera `200x200 @2.5 µm` 的 native DFT lattice 定义 `0–20 cycles/mm` semantic、`20–80` mid、`>80` high。train teacher target 在 semantic band 的中心化能量份额为 `95.96%`，通过预注册 `90%` cutoff；v63 camera code-main 频带份额为 `20.16%/67.47%/12.37%`，label-main 为 `18.24%/69.56%/12.20%`。
- 独立 held-out band probes（raw v61 probe 保持冻结）得到：semantic `0–20` corr/dice `0.9421/0.8782`，paired rotation drop `0.0547`、95% CI `[0.0474,0.0626]`；non-semantic `>20` corr/dice `0.9454/0.8832`，drop `0.0545`、CI `[0.0469,0.0625]`；冻结 v61 raw reference 为 `0.9452/0.8829`。两路均保留 paired held-out information。
- 因此 v64 决策为 `distributed-subvisual-carrier` / `manual-review-required`：paired representation 同时分布于 semantic 与 non-semantic 频带，不存在单一主载波。raw CMOS 人眼不可读的硬 gate 不变；v64 不授权 detector/display calibration、generator/loss/optical-contract 修改、训练或 promotion。下一门只能做跨频带表示归因与固定显示/人工审查关联。
- v64 preview 人工复核显示 teacher 行可读，raw CMOS 仍是同源颗粒纹理，semantic 行接近相同低频模板，non-semantic 行主要是颗粒纹理；因此最终 raw CMOS 人眼 gate 仍失败。

代表证据：

- 工具：`tools/diagnose_v64_semantic_band_carriage_audit.py`
- 报告：`outputs/reports/v64_semantic_band_carriage_audit.json`
- tensor：`outputs/reports/v64_semantic_band_carriage_audit.pt`
- preview：`outputs/reports/v64_semantic_band_carriage_audit_preview.png`
### v65：common-mode 与 readout-locality attribution audit（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 config、v58.3 relay、readable `800/100/100` split 及 v61/v63/v64 工件，完成一次零训练 CUDA audit。provenance、identity OLS、formal field-chain、camera `|E|²`/detector normalization 与参数快照通过；`parameter_update_count=0`、`optimizer_created=false`，临时 native memmap 已删除，结果均为 `diagnostic-only`。
- train-only common mean/gain 的参数为 RMS `0.230195`、residual abs q995 `0.0071021`、gain `63.3615`。固定 common-mode display 的 test corr/dice/prototype accuracy 为 `0.04295/0.33365/0.46`，人工 preview 仍不可读。
- 冻结 v61 raw reference 为 corr/dice `0.94523/0.88293`。pointwise affine 为 `0.68988/0.68732`；3x3、7x7、15x15、31x31 shared local convolution 分别为 `0.62136/0.64443`、`0.65647/0.67172`、`0.66580/0.67585`、`0.66830/0.67637`；full-support Fourier/Wiener 为 `0.68419/0.68491`。所有 target-fitted observer 的 within-class rotation CI 下界不为正，均未达到 v61 raw corr/dice 的 90% paired retention。
- 预注册决策为 `nonlocal-codebook-objective-representation-mismatch-confirmed`，因为 fixed display、pointwise、全部 `<=31x31` local kernel 与 Fourier/Wiener 均失败，而冻结 v61 dense reference 仍可解码。raw CMOS 人眼 gate 仍失败；该结果只授权架构讨论，不授权训练、改 loss/decoder/generator/D2NN/OLS/relay/aperture 或 promotion。下一门应在不改变 formal path 的前提下讨论 nonlocal codebook/objective 设计与可读性约束，不直接启动训练。

代表工件：

- `tools/diagnose_v65_common_mode_readout_locality_audit.py`
- `outputs/reports/v65_common_mode_readout_locality_audit.json`
- `outputs/reports/v65_common_mode_readout_locality_audit.pt`
- `outputs/reports/v65_common_mode_readout_locality_audit_preview.png`

### v66：semantic-gradient budget audit（CUDA 已完成，diagnostic-only）

- 新增只读 `tools/diagnose_v66_semantic_gradient_budget_audit.py` 与 `tests/test_v66_semantic_gradient_budget_audit.py`。工具固定 v60 epoch-5、v56 formal config、v58.3 relay、readable `800/100/100` 及 v61–v65 工件；使用正式未 clamp camera reconstruction residual，对 generator、encoder、decoder 参数块进行逐样本 VJP 与流式 cancellation/cosine 统计。
- train-only global/label templates、common/label/instance residual 精确分解、foreground/background 权重、参数零变化和四个决策分支均已覆盖。工具不创建训练对象、不写入参数梯度、不保存完整复场或梯度。
- 已按预注册输入完成一次 CUDA audit：`parameter_update_count=0`、`optimizer_created=false`，报告、tensor 与 preview 均已生成。结果为 `mixed-semantic-gradient-evidence`：label/instance drive 未同时低于 suppression gate，且未满足 semantic-gradient-available 或 batch-cancellation gate；它不授权继续 v56/v60 同配置训练，只保留为透明薄相位 diagnostic 的历史前置证据。
- 实现验证：v57–v66 定向 `58 passed`；完整仓库 `522 passed`、`148 subtests passed`；compileall、ruff、`git diff --check` 通过。

## 2026-07-16 至 2026-07-22 v67–v79：透明诊断、code transport 上界与冻结结论

以下记录均为历史 `diagnostic-only` 证据，不改变 stable formal contract，也不构成 student promotion。需要逐次查看原始交接正文时，使用 [v68–v79 原始主线交接归档](../archive/current_handoff_v68_v79_archive.md)；本节保留面向检索的浓缩索引。

### v67–v67.4：透明薄相位与全场传播数值门

v67 引入 `transparent_identity` 作为 Formal 明确拒绝的独立诊断边界。v67.1 的 ASM 零相位和 backend relay reference 通过，但冻结相位在 `1024/1536` 网格的 chip complex L2 为 `0.02107/0.005816`，未满足 L2≤`1e-3` 与 guard≤`1e-4`，状态为 `grid-not-converged`（`outputs/20260716_v671/runs/preflight/v67_transparent_convergence_v2.json`）。

v67.2/v67.3 修正 padded propagation、FOV、共同 ROI 与 phase gauge 后，候选 `1536/3072` 对 reference 的 zero/small-init gauge-aligned complex L2 仍约 `0.001408/0.001412`，首失败为 `decoder_layer_1`；没有进入梯度、显存或训练门（`outputs/20260718_v673/runs/preflight/v67_transparent_convergence_v4.json`）。v67.4 的全场算子隔离继续为 periodic stress evidence；它不修改 `finite_crop`、v56/v60 checkpoint 或默认 YAML。

### v68–v69：prototype 通过不等于 code generation

v68 A2 的 locked raw audit 达到 clean corr/dice `0.9910/0.9548`、prototype accuracy `1.0`、high saturation `0.0032`，十类 raw-linear 人审通过；其后 v68.6 B 的 clean corr/dice 约 `0.9951/0.9676`、accuracy `1.0`，但 raw-camera code fraction 仅 `7.14e-08 < 0.25`，有效决定为 `b-prototype-imaging-only`。工件位于 `outputs/20260718_v685/runs/` 与 `outputs/20260719_v686/runs/`。

v68.7 证明 generator injection 是首个退化面：residual map factor `0.74447`，SLM 仅 `8.25e-06`，即使 tanh 上界也仅 `0.003595 < 0.25`。v69 的 paired teacher-camera 合同将 raw-camera factor 提升到 `0.0002712`，仍未达门；paired-teacher advantage 的 bootstrap 95% 下界为 `-0.00559`。状态 `v69-prototype-only`，冻结 checkpoint，禁止重训、sweep、test、导出与 promotion（`outputs/20260720_v69/runs/`）。

### v70–v77：冻结 D2NN 的传输归因

v70 将 SLM command intervention 校准到 `0.25000003`，但 camera code factor 降至 `0.19396`；v71 改在 physical `slm_amplitude²` 校准后，camera 为 `0.22021`，首个有效衰减面为 `encoder_layer_2`。v72 的 identity ablation 可提高 code factor（joint 为 `0.29503`），但破坏数字可读性，结论是 `phase-readability-code-tradeoff`。代表工件：`outputs/20260720_v70/runs/isolation/v70_slm_transport_isolation_v1.json`、`outputs/20260720_v71/runs/isolation/v71_slm_intensity_transport_v1.json`、`outputs/20260720_v72/runs/isolation/v72_d2nn_phase_attribution_v1.json`。

v73 semantic-support intervention 保持 prototype readability，却使 camera factor 为 `0.16961`；v74/v75 进一步确认主要机制是冻结 D2NN 的 carrier/residual phase-to-intensity redistribution，而不是 full-state exterior escape。v76 双正交读出虽得到 joint factor `0.740539`，但端口饱和 `0.156618 > 0.08`，无法作为部署上界。v77 只证明局部 phase gradient compatibility，不授权训练。上述结论及工件位于 `outputs/20260720_v73/` 至 `outputs/20260720_v77/`。

### v78–v79：八维 supported-basis 家族关闭

v78 的 semantic-gated direct-command CPU admission 虽通过 provenance 与有限性，但 physical-SLM code factor `0.00049967 < 0.25`、support energy `0.94566 < 0.95`，状态为 `v78-command-code-factor-insufficient`。v78.1 定位到 template support 与逐样本 normalization 的 code suppression；它关闭 template multiplication 加逐样本 normalization 合同（`outputs/20260721_v781/runs/isolation/v781_semantic_command_bottleneck_v2.json`）。

v79 使用固定 label-only normalizer，对 frozen learned replay 与 deterministic capacity arm 作 generator-only CPU 上界。前者 factor 为 `6.03e-7`、absolute code SS `0.496`；后者 factor 为 `0.05455`、absolute code SS `48292.58`，均低于 `0.25` 和 v73 的 physical-SLM absolute baseline `289721.29`。虽 support energy `0.97080`、低频能量 `0.99814` 通过，但 clipping `0.20547 > 0.08`。有效决定为 `label-fixed-supported-basis-insufficient`，冻结该八维 basis 家族及 residual＋phase 联合训练；工件为 `outputs/20260721_v79/runs/admission/v79_label_fixed_supported_basis_v1.json` 与 `v79_joint_decision_v1.json`。
## 2026-07-27 至 2026-07-28：v2 v5 多锚点容量准入失败

`conditional_multi_anchor_local_modes_v5` 只完成了代码实现和无更新容量审计，没有启动optimizer、CUDA训练或数据集导出。固定55k MNIST训练子集上的四簇特征审计得到平均类内离散度降低 `0.1691328704`、最低类别降低 `0.1243160963`，均低于预注册的 `0.30`/`0.15` 门。数字1的四个簇大小为 `2027/1855/156/2204`，最小簇也低于256张要求。证据保存在 `outputs/20260727_v2/runs/20260727_211242_776413_formal_v2_multi_anchor_local_modes_finite_crop_identity/`，原始evidence保持只读。

该结果否定的是每类四个等权硬锚点和象限路由的容量假设，不否定完整 `label+code → SLM → Encoder → identity OLS → Decoder → CMOS` 宏观光路。特别是数字1的小簇若被强制等概率路由，会把约2.5%的真实形态放大到25%，因此不能通过放宽最小簇或离散度门继续训练。下一动作是diagnostic-only的连续PCA模式容量审计；在该审计通过前不得派生v5 checkpoint或启动任何smoke。

## 2026-07-28：连续PCA模式容量数据门通过，物理probe等待CUDA

`audit conditional-continuous-mode-capacity` 使用当前内容绑定evaluator、55k训练split和语义checkpoint完成无更新数据审计。PCA-8相对v5 K=4基线的平均优势为 `0.4269000490`、最低类别优势为 `0.3052000363`，振幅PCA-8最小解释收益为 `0.4731594795`，十类物理basis均为rank 8。数据门通过，说明连续类内模式比四锚点等权硬路由更符合当前MNIST类内几何。

运行目录为 `outputs/20260728_v2/runs/20260728_180501_490139_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。后续物理probe未因指标失败，而是因当前环境 `CUDA unavailable` 被标记为 `harness_invalid`；未创建optimizer、未修改参数、未生成checkpoint。恢复时必须在CUDA可用环境中使用相同配置、evaluator、split和语义checkpoint重跑无更新probe，不能直接进入训练。

## 2026-07-29：连续模式旧CUDA probe合同不完整

20260729运行目录为 `outputs/20260729_v2/runs/20260729_104105_417380_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。完整光路、camera code fraction和saturation通过，但语义结果为 axis `0.8235`、uniform-1234 `0.5824`、uniform-5678 `0.5706`，类别5/7 recall最低；因此不满足原定语义门。该版本还直接使用v3的gain `0.5`，没有执行预注册安全logit RMS、逐方向响应、梯度和stateless切空间门。

旧evidence标记为 `probe_gain_contract_mismatch / harness_contract_incomplete`，不修改历史JSON，不消费其准入动作。新合同固定 `alpha=0.05/tanh(1)`，新增 `conditional-continuous-mode-reachability` 只读审计；在新审计通过前不得派生连续模式checkpoint或训练。

实现状态：新审计已固定旧basis hash，输出三个完整 `10×17` contact sheet，并分别对
class carrier、continuous PCA modes、Encoder phase、Decoder phase 做无状态可逆单步
margin/distribution tangent probe；电子扰动限制 SLM-logit RMS `0.05`，相位扰动限制
`0.005 rad`，raw-CMOS目标变化 `1%`。该改动只增加诊断代码和测试，不修改旧run/evidence，
也不产生checkpoint。由于当前执行环境 CUDA 不可用，必须由用户在 CUDA 环境运行新命令后，才能
判断 `continuous_semantic_alignment_reachable`、objective conflict 或 harness invalid。

## 2026-07-29：连续 PCA 安全尺度 CUDA 可达性审计失败

WSL CUDA 动态链接路径修复后，RTX 3060 Laptop GPU 完成了新的只读审计：
`outputs/20260729_v2/runs/20260729_131120_150834_diagnostic_v2_continuous_mode_capacity_finite_crop_identity/`。
实现同时修复了诊断本身的内存合同：170样本反向图改为每个类别17样本即时消费与梯度累计，以适配6 GB显存；
没有 optimizer、checkpoint 或参数更新。

安全尺度准确达到 axis SLM-logit RMS `0.0500`，并保留高语义 accuracy
`0.9824/0.9529/0.9529` 和低饱和约 `0.0076`。但 raw CMOS code fraction 仅
`0.0071/0.0212/0.0240`，物理 `A²` fraction 仅 `0.0107/0.0323/0.0352`，均未达到 `0.10`。
结果为 `failed / safe_scale_transport_insufficient`，无下游授权。这是有效物理传输停止证据：不能恢复
gain `0.5`、不能做 gain sweep、不能派生连续PCA正式checkpoint或训练；若继续，只能审计带 PCA
特征值尺度的物理模式参数化。

## 2026-07-29：v6.2 短层距自成像恢复合同

v6 smoke 与局部宏观审计失败后，保留完整宏观光路但建立独立 diagnostic 合同：层距
`1.50 mm → 0.25 mm`，2+2 相位层、relay、identity OLS、finite_crop、曝光和网格不变。
新增 `conditional_pca_self_imaging_transport_v6_2`、固定全局 `κA²` target 和同 code
delta transport loss。旧 v6 checkpoint/evidence 不迁移；执行顺序为 target admission →
optical admission → observability → 唯一 400-step smoke。正式训练与数据集闭环在 smoke
通过前继续禁止。
