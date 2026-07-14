# 当前主线交接：v61

更新时间：2026-07-15。此文件供新对话快速恢复当前主线；带日期的完整证据在 [Experiment History](experiment_history.md)，稳定架构合同在 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)。

## 先读的结论

- 当前所有 v49-v61 结果都是 `diagnostic-only`，且 `accepted_summary_written=false`。没有可 promotion 的学生 checkpoint。
- 人眼硬 gate 只看最终 `physical_camera_plane` 的 CMOS intensity。v56 的正式 class-balanced sample（0-9 每类两张）仍为同源圆斑/纹理，不可读；这是 strict provenance 有效条件下的模型失败，不是 harness failure。
- 不要重新使用 v47/v51/v53 checkpoint：它们与现行的 identity OLS、1.50 mm geometry 或完整前端 4f relay 合同不兼容。
- 在下一决策门前，不训练、不延长 epoch、不叠加 loss、不做 spacing/padding/pupil sweep，也不修改正式硬件合同。

代码已完成行为保持的领域重构：`optics / student / teacher / application / diagnostics` 为正式目录，stage 通过 `StageRequest` 懒构造依赖；旧 import 仍解析到同一实现。此结构状态不改变下列 v56–v59 科学结论。

## 现行正式理想数字孪生合同

```text
laser -> M1 -> BS -> amplitude SLM -> BS -> M3
      -> coherent 4f relay -> encoder D2NN (2 layers)
      -> complex identity OLS
      -> decoder D2NN (2 layers)
      -> finite-conjugate coherent relay -> CMOS intensity
```

- `lambda=532 nm`；无 fiber。OLS 是严格复线性恒等映射 `Eout=Ein`。
- SLM/CMOS：`200x200 @2.5 um`，500 um FOV。
- Encoder/decoder：各 `100x100 @2 um`，200 um FOV，各两层，phase-only，spacing `1.50 mm`。
- 前端完整 FOV 相干 4f 缩小成像：`M=-0.4`、`f1=223.214 mm`、`f2=89.286 mm`、最低 pupil `23.75 mm`；禁止 central aperture crop 或 generic complex resize。
- 后端有限共轭成像：`M=-2.5`、`u=100 mm`、`f=71.4285714 mm`、`v=250 mm`、最低 pupil `26.60 mm`。
- 严格光场 artifact 必须从同一传播复场的 `real/imag` 现场计算 `abs(E)`/`angle(E)`；camera intensity relay 不得伪造 phase。

v55 structural preflight 通过；原 v55.1 reference 仅证明生产算子与同构 Collins 实现自洽，已降级为非独立证据。v55.2 的低分辨率实际尺寸 direct-RS 参考也因采样不足而无效。v58.2 CUDA run 证明 DFT/NUDFT 可执行，但其 `128→64` sampling、硬编码 Airy error 与单段 component reference 不能签发正式 gate。当前门是 v58.3：完整 `200→100` FOV、闭式 B=0 场、实测 pupil PSF 与故障注入独立性审计。

- `outputs/reports/v55_coherent_relay_cuda_preflight.json`
- `outputs/reports/v55_1_coherent_relay_reference_cuda.json`（历史自洽检查）

这些是理想数字孪生与设计 pupil 下限的结构/历史自洽检查，不是独立物理参考或硬件实测。

v49-v54 的 mask 归因、严格复场修复和旧 central-aperture 失败定位均已降为历史证据；完整细节见 [Experiment History](experiment_history.md)。v60 是 v56 的同配置复制训练，未改善人工可读性。v61 稳健审计在完整 val/test split 上未确认稳定的局部首塌缩；下一门是已实现的离线 decodability upper bound。

## v56-v59 事实与证据

v56 best checkpoint（epoch 5）：

```text
outputs/20260714_v56/runs/20260714_204958_697374_train_v56_coherent_relay_from_scratch/checkpoints/one_shot_distill_best.pt
```

训练从 epoch 1 到 5：validation loss `0.2715 -> 0.2276`，paired corr `0.4913 -> 0.5870`，paired dice `0.5704 -> 0.6351`；但 diversity ratio `0.00146 -> 0.00095`，cross-class ratio `0.000138 -> 0.000103`。

正式 eval：corr `0.5789`，dice `0.6298`，diversity ratio `0.000968`，strict readability failed：

```text
outputs/20260714_v56/runs/20260714_210206_240133_train_v56_coherent_relay_eval/artifacts/one_shot_distill_eval_metrics.json
```

正式 sample/provenance：

```text
outputs/20260714_v56/runs/20260714_210416_032022_train_v56_coherent_relay_sample/
```

v57 corrected report：

```text
outputs/reports/v57_first_collapse_complex_field.json
```

v57 以 SLM 的跨类中位距离减同类中位距离为 100% 基准，得到：前端 relay 约 47%，encoder/OLS/decoder/chip 的 complex separation 约 39%-49%，camera intensity 约 11.7%。OLS 前后连续。没有任何 plane 满足“至少两类指标低于 10% 且持续到下游”的局部首塌缩门。

v58 aperture-throughput upper bound 已完成，且人工并排复核 baseline 与 upper-bound 的最终 CMOS 都是同源圆斑/纹理、不可读。它在同一 v56 checkpoint、相同输入/relay、零训练下移除了 D2NN 层间 finite crop；chip 能量从 `126.18` 增至 `17232.99`，但 camera field 能量仅从 `66.27` 增至 `69.78`，CMOS separation gain 只有 amplitude `1.0227x`、complex `1.0173x`、intensity `1.0101x`。因此 finite crop 是高吞吐损失，但不是类别/图像塌缩的充分根因：

```text
aperture-upper-bound-no-separation-recovery
manual-review-required
diagnostic-only
accepted_summary_written=false
```

v58 报告与 tensor：

- `outputs/reports/v58_aperture_throughput_upper_bound.json`
- `outputs/reports/v58_aperture_throughput_upper_bound.pt`

v59 upstream broad-isolation 已在同一正式合同与 v56 checkpoint 上完成，验证了 100 张 readable validation（0-9 每类 10 张）、`readability_strict_split`、identity OLS、正式 relay geometry 与参数零变化。teacher target 的 prototype CV 为 `1.00`，permutation `p=0.00775`；baseline SLM command 与前端 4f 输出的 prototype CV 均为 `0.94`，前端相对 SLM 的 complex separation 为 `41.0%`，SLM 相对 teacher target 的 complex separation 仅 `0.6787%`。SLM 类间频谱约 `80.2%` 在形式通带外，但 4f 输出仍保留显著类别 prototype signal，故未满足 bandwidth mismatch 门。label-shift 与 code-rotation 对最终 target MSE 的相对变化分别为 `0.155%` 与 `0.0036%`；最终 CMOS 仍不可读。

v59 corrected report 与 artifacts：

- `outputs/reports/v59_upstream_separation_isolation.json`
- `outputs/reports/v59_upstream_separation_isolation.pt`
- `outputs/reports/v59_upstream_separation_isolation_preview.png`

当前 v59 决策为：

```text
objective-representation-mismatch-candidate
manual-review-required
diagnostic-only
accepted_summary_written=false
```

下一门是离线 decodability upper bound；不得继续 decoder/loss 微调、延长 v56 epoch、sweep aperture/spacing/pupil，或 promotion v56。

每层 numerical propagation 能量比约 `0.999999`；但 finite aperture crop retention 是 encoder L1 `25.70%`、encoder L2 `27.65%`、decoder L1 `30.83%`、decoder L2 `33.42%`，累计 `0.7322%`。其类别差异不显著，故当前状态是：

```text
aperture-throughput-limited-no-localized-collapse
manual-review-required
diagnostic-only
accepted_summary_written=false
```

## v59 下一决策门

v59 tool：`tools/diagnose_v59_upstream_separation_isolation.py`。它只做固定 validation batch 的 baseline、label derangement、within-class code rotation forward，并将类间/类内分离、permutation null、target error、label/code sensitivity 与 formal pupil spectral audit 写入报告。当前报告指向 offline decodability upper bound；其仍是 `diagnostic-only`，不改变正式合同。

## v60-v61 当前证据与下一门

v60 使用与 v56 相同的 readable 数据、seed `1234`、5 epoch 和正式 `coherent_imaging_4f/free_space/identity OLS` 合同；它只是复制训练，不是新物理架构。相较 v56，eval corr/dice 为 `0.5733/0.6235`（v56 为 `0.5789/0.6298`），sample corr 为 `0.4914`（v56 为 `0.4945`），最终 CMOS 仍为跨类别同源圆斑/散斑。

v61 robust audit 使用冻结 v60 best checkpoint 和 readable `800/100/100` train/val/test，参数 hash 前后相同；以 32×32 诊断网格、2,000 次固定大小 pair bootstrap 和 permutation null 复核全部正式 planes。当前报告为 `no-robust-localized-collapse`，因此 v60 的单次 20 样本、`0.0981` camera 阈值边缘报告不能单独升级为 camera 首崩溃结论。

代表证据与工具：

- `outputs/20260715_v61/runs/robust_collapse/v61_robust_collapse.json`
- `outputs/20260715_v61/runs/robust_collapse/v61_frozen_optical_cache.pt`
- `tools/diagnose_v61_robust_collapse.py`
- `tools/diagnose_v61_offline_decodability.py`

离线 readout 只允许作为 `diagnostic-only` 上界：linear ridge 会在 validation 选择预注册强度，test 只评估一次；固定小型 nonlinear readout 只运行审计发现的边界两侧，并保留 code-only、code+label、label-only 与类内打乱对照。实际 v61 report 中 linear plane probe 的 test corr 约 `0.874-0.892`、dice 约 `0.676-0.710`，但相对 label-only 的 paired delta 为负；code-only/code+label 仅 `0.716/0.731` corr。nonlinear 三 seed 未达到 strict gate，且部分 seed 退化为零多样性。结论是 plane 中保留的是类别模板/低阶可预测信号，不是通过 paired software readout 证明的 instance-level single-exposure generation。软件 readout 的可读图不能替代 raw CMOS 人工硬 gate，也不能进入正式单曝光推理链。

## 已完成验证

v58.3 targeted/CUDA preflight、v61 robust audit 和离线 readout 已完成；v61 targeted tests `6 passed`，工具报告均为 `diagnostic-only`。所有输出和 cache 均位于 `outputs/`，不加入 Git；迁移时不得用 reset/checkout 清理用户改动。
