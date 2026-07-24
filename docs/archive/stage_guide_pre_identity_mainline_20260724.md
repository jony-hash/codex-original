# Stage Guide

本文件是 v1 stage 的历史快照。架构与最终推理边界见 [D2NN 单次曝光蒸馏主线](../architecture/v1_single_shot_distillation.md)，带日期的结果见 [实验与验收历史](../status/experiment_history.md)，当前文档导航见 [Documentation](../README.md)。当前运行规则以新的 [v2 runtime](../architecture/v2_runtime.md) 和 [Git 分支治理指南](../operations/git_branch_management.md) 为准。

本文件中的 `train.py --stage` 命令只适用于 v1。v2 使用独立的 `python -m ldm_project.v2` 子命令和独立 evidence gate，二者不互相替代；在 v2 正式 acceptance 前，v1 仍是默认入口。

v2 的可执行闭环为：内容寻址 teacher pair/target manifest → prototype measurement → 内容绑定的人工 admission → frozen capacity → guarded training → checkpoint-driven raw-CMOS evaluate。small diagnostic system 仅允许 `diagnostic_*_smoke` 动作，不能授权 formal train、export、evaluate acceptance 或 promotion；正式 CUDA 命令和 review 交接见 [v2 runtime](../architecture/v2_runtime.md)。

2026-07-23 v2 prototype 断点已完成一次 broad isolation 和一次修正后的 calibration。`sqrt(clean intensity)` 初始化、5×40 logical steps、固定 clean-correlation/Dice/CE objective 后，corr 达到 `0.8395`、dice `0.7625`，但 prototype accuracy 仅 `0.50`，且末两 epoch Dice 退化。此前 `212454` reachability 的 passed 状态现为 harness-invalid：crop、front separation 与逐类 direction gate 的计算不符合合同，且 calibration 未绑定该 evidence。保留 `212618` 作为旧 reconstruction 合同的数值负结果，不据此下物理不可达结论。修正后的 reachability 通过且 topology isolation 显示 joint 最好，但 shape-guard 仍以 accuracy `0.50`、minimum full-frame margin `-0.0801` 与 background-speckle failure 自动停止。无更新 geometry 上界显示 0.25 mm 虽恢复十类 accuracy，完整 raw-CMOS 仍以 margin `0.0073`、speckle `0.1725` 失败；bounded control oracle 随后证明十类固定 amplitude command 可控制冻结光路，但 direct-template 邻域的固定 DCT/8D/gain-0.1 local capacity 上界失败，仍不得训练。当前仅允许完全隔离的 LightGen architecture-proxy 对齐审计：参见 [v2 runtime](../architecture/v2_runtime.md)，它不改变 formal `2+2/identity OLS` 合同，也不授权 capacity 或 promotion。

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
python train.py --train-config configs/v1/formal/train_ldm.yaml \
  --stage teacher_first_stage --device cuda

python train.py --train-config configs/v1/formal/train_ldm.yaml \
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

正式 config 必须以 [架构文档](../architecture/v1_single_shot_distillation.md) 为合同来源，至少锁定 identity OLS、完整 SLM-to-encoder 相干 4f relay、`100 × 100 @ 2 µm`、`1.50 mm` D2NN spacing、严格 physical-field provenance，并先通过对应 CUDA preflight。正式 stage 由 `FormalOneShotOpticalGenerator` 构造；包含 bypass、phase-amplitude、generic resize、target-code bridge 或 SLM override 的配置只能走 compatibility/diagnostic facade。v56 的人眼失败和 v58 门属于历史证据；v1 已在 v79 capacity gate 后冻结，不提供新的 v1 训练命令。v2 的正式前置门与命令见 [v2 runtime](../architecture/v2_runtime.md)。

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
- 详细原因见 [架构文档](../architecture/v1_single_shot_distillation.md)；历史测量值见 [实验与验收历史](../status/experiment_history.md)。

## Output Policy

- 训练、诊断和 pre-acceptance run 默认写到 `outputs/YYYYMMDD_<version>/runs/<run>/`；`<version>` 从 `--run-name` 或 train config 文件名提取，缺省为 `unversioned`，日期前缀取该版本当前已知最早 run 日期。
- 诊断和 pre-acceptance 结果仍必须标注为 `diagnostic-only`，不能仅因位于 `outputs/` 就视为 accepted。
- `outputs/runs/` 已移除；具体历史路径需要改写到真实的带日期前缀版本目录，`outputs/final_pt/` 不参与版本迁移。
- 整理历史 run 时先执行 `python -m tools.organize_outputs_by_version --root outputs --dry-run` 检查 manifest，确认无 blocker 后再执行 `--apply`。
- 输出图片和 tensor bundle 的语义见 [输出与可视化说明](../operations/output_artifact_notes.md)。

## Archived Experiments

v9.10 prototype、旧 100×100 two-photon、v53 及更早的 loss/relay probes 都已迁入 `experiments/archive/`，并由 [archive manifest](../../experiments/archive/manifest.yaml) 标为 `diagnostic-only`、`runnable: false`。它们只能解释历史证据，不能作为当前命令、checkpoint 或 formal physical-forward 入口；当前合同与停止规则只以 [Current Handoff](../status/current_handoff.md) 为准。

## Diagnostics

以下流程均为 `diagnostic-only`：它们可以定位环节问题，但不能单独证明最终 one-shot student 硬件路径可用。

### Experimental Debugging Decision Policy

本节只定义通用决策流程；当前版本状态、run 路径和下一决策门以 [Current Handoff](../status/current_handoff.md) 为准。

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

### v68.4 双目标 raw 验收与 A2/B 准入

`tools/audit_v68_raw_cmos.py` 的 v2 报告同时计算 clean MNIST prototype 和 `teacher_camera` 两套目标；clean prototype 是 A/A2 的主可读性门，teacher camera 只作蒸馏相似度辅助，不能互换。预览固定为 clean prototype、teacher camera、locked-scale raw-linear 和历史 per-sample normalized 对照。A 使用 `calibrate`（全 validation 一次计算 99.5% scale）；A2/B 使用 `locked`，固定 scale 为 `0.4913053810596466`，禁止重新 percentile 标定。

v68 A 已通过 numerical/gradient/memory 和自动 raw 门，但 `4/8` 仍为 blob、`2/5` 仍模糊，因此状态是 `automated-pass / manual-readability-fail / prototype-upper-bound-only`。不要重训 A 或直接运行 B；先用 A template/phase warm-start 执行唯一一次 5 epoch A2。A2 只有在 clean-prototype raw corr/dice、饱和度、逐类人工门和 epoch 4→5 稳定性同时通过后才允许 B。B 必须继续使用 locked exposure、strict_readability，并只重新初始化 label embedding/residual；code factor 不足时不得称为生成模型。

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

### v68.5 A2 短准入与联合决策

A2 长训练前先运行只读 smoke（四个 validation 样本，不创建 optimizer）：

```bash
python -m tools.diagnose_v68_a2_warmstart \
  --train-config configs/train_v68_transparent_readability_a2.yaml \
  --checkpoint <v68_A_best.pt> \
  --validation-dataset <readable_val.pt> \
  --output <v68_a2_warmstart_smoke.json> --device cuda
```

smoke 必须报告 `a2-warmstart-smoke-passed`、finite、参数 hash 未改变和 memory limit 通过，才允许用户执行 5 epoch A2。A2 完成后，人工 review JSON 需要明确十类是否可读，再用 `tools/decide_v68_a2_acceptance.py` 联合判断；未通过即禁止 B、test 和加工候选导出，不以 paired corr/dice 或增强预览替代人工门。

v68.5.1 使用 smoke v2：它将 model/warm-start 后的 CUDA reserved memory 作为 baseline，分别报告 absolute 与 delta peak。只有 delta reserved 不超过 v68.2 admission 的 `1.904296875 GiB`、delta 不超过启动空闲显存的 85%、absolute reserved 不超过总显存的 90%，才算 memory pass。旧 v1 绝对/增量混比报告保留但无效；重新运行时输出必须使用新文件名 `a2_warmstart_smoke_v2.json`。

### v68.6 B code-conditioned raw-CMOS

v68.5 A2 的实际联合决定为 `a2-raw-readability-passed`（`b_allowed=true`）。B 先运行只读 `tools/diagnose_v68_b_warmstart.py`：它必须链接该 A2 决定，只加载 `generator_e.template_logits` 与 encoder/decoder phase，验证 generator `1/9`、optical `2/8` 覆盖、seed 1234 的 residual/embedding 未变，并在四个 validation 样本上完成有限 forward/backward；不创建 optimizer 或更新参数。

smoke 通过后由用户执行唯一一次 5 epoch B。有效 run 使用 `tools/audit_v68_b_crossed_cmos.py` 对 validation 的 100 codes × 10 labels 做平衡 crossed audit；它禁止 test split，保持 `return_optical_fields=false`，只流式统计 SLM、pre-encoder、chip active intensity 和 `physical_camera_intensity_raw`。camera code factor 固定为 raw-camera 平衡 ANOVA 的 `code_fraction`；预览固定使用 `0.4913053810596466`，不做逐样本 percentile/gamma。最后以 `tools/decide_v68_b_acceptance.py` 联合 A2、B smoke、epoch history、crossed report 和人工 review：通过状态为 `b-code-conditioned-readability-passed`，code factor 失败为 `b-prototype-imaging-only`，可读性失败为 `b-readability-regression`，provenance/数值异常为 `harness-invalid`。三种状态均保持 test、加工导出和 Formal promotion 为 false。

实际 v68.6 B 已完成唯一一次 5 epoch：可读性、稳定性、能量和人工十类门均通过，但 `camera_code_factor=7.14e-08 < 0.25`，最终状态为 `b-prototype-imaging-only`。这证明当前路径可形成高质量请求标签原型图像，却不证明 code-conditioned generation；停止 B 重训、任何 hyperparameter/loss/scale sweep、test、加工导出和 Formal promotion。

### v68.7 frozen bottleneck closure and v69 code distillation

`tools/diagnose_v687_b_code_bottleneck.py` 是只读 closure：它必须链接 B decision v2、B best checkpoint、validation split 和 crossed report v2，流式记录 residual logits/map、desired intensity、SLM、pre-encoder、chip 与 raw camera 的 balanced code/label/interaction fraction、seed 参数偏移和 hash。固定 `code_residual_scale=0.10` 的唯一 tanh ±1 上界仍给出 SLM code fraction `0.003595 < 0.25`，所以实际状态为 `generator-injection-contract-insufficient`；v68 至此关闭，不能 B 重训、sweep、光路 isolation、test、加工导出或 Formal promotion。

v69 只能使用 `configs/train_v69_code_distill.yaml` 的独立 experimental generator：`label_template_code_delta` 计算 `sigmoid(template_logits + delta_logits(code,label))`，并强制 scale 为 `0.0`。它从 B best 全量继承 generator/phase，但冻结 template、label embedding、encoder/decoder phase，只以 `5e-6` 训练 residual net，并使用 paired `teacher_camera` target；不增加 diversity loss。训练前必须由 `tools/diagnose_v69_code_delta_warmstart.py` 链接 v68.7 closure，验证 generator `loaded=8/missing=2`、optical `loaded=2/missing=8`、冻结 hash、四样本 finite residual gradient、零 optimizer/update 和 CUDA capacity。该 smoke 已通过。

唯一有效 v69 run 完成后，`tools/audit_v69_code_distill.py` 必须在 validation 的 `100×10` crossed grid（batch 1、`return_optical_fields=false`）上流式审计 delta logits 至 raw camera；预览使用 locked scale `0.4913053810596466`，不使用 test 或完整 2048/3072 field。`tools/decide_v69_code_distill.py` 同时要求 clean readability、camera code factor、epoch stability、十类人工审查和 paired teacher 对齐：diagonal teacher corr/dice 至少 `0.70/0.68`，每个样本优于同类其他九个 teacher 的平均相关性，seed 1234 的 10,000 bootstrap advantage 下界大于零。状态只可为 `v69-code-distillation-passed`、`v69-code-transport-unaligned`、`v69-prototype-only`、`v69-readability-regression` 或 `harness-invalid`；所有状态仍禁止 test、加工导出及 Formal promotion，失败不重训或 sweep。

实际 v69 run 已完成并通过 harness、clean readability、energy、epoch stability 和十类 locked raw 审查；联合状态为 `v69-prototype-only`。camera code factor 为 `0.0002712`，未达 `0.25`，且 paired teacher advantage 的 bootstrap 95% 下界为 `-0.00559`。factor decomposition 显示 delta logits 的 code fraction 为 `0.70649`，但在 SLM command 已降为 `0.001143`，随后在 pre-encoder/chip/raw camera 分别为 `0.000417/0.000271/0.000271`。因此首个主要退化面是未约束 logit delta 到 amplitude-SLM command 的映射；冻结该 checkpoint 和报告，不重训、不 sweep，下一步必须以新的独立合同处理这个接口瓶颈。

### v70 SLM code-transport upper-bound isolation（2026-07-20）

v70 对冻结 v69 best checkpoint 做了一次只读 `lowfreq_additive` SLM intervention。generator-only 30 次二分将平衡 SLM code fraction 校准为 `0.25000003`（绝对误差 `3.46e-8`）；未创建 optimizer、未更新参数、未保存完整 2048/3072 全场。唯一一次 100 codes × 10 labels CUDA crossed forward 的 clean prototype corr/dice/accuracy 为 `0.96495/0.93059/1.0`，人工 locked raw 复核十类全部可读，2/5 与 4/8 保持区分。

但 code fraction 在 pre-encoder 首次降至 `0.22124`，随后 chip/camera 为 `0.19430/0.19396`，camera 未达到 `0.25` 门。因此联合状态固定为 `post-slm-code-factor-attenuated`：后端只能传输约 `0.194` 的该合成 code pattern，上界隔离不授权 generator 重训、scale/loss/LR sweep、test、加工导出或 Formal promotion。v70 全部工件保持 `diagnostic-only`，报告为 `outputs/20260720_v70/runs/isolation/v70_slm_transport_isolation_v1.json`，联合决策为 `v70_joint_decision_v1.json`；首个 attenuation plane 为 `pre_encoder`。

### v71 physical SLM-intensity to D2NN isolation（2026-07-20）

v71 将唯一 intervention 的校准域改为物理 `slm_amplitude²`，30 次 generator-only 二分得到 `0.24999998`（绝对误差 `2.3e-8`，override amplitude `0.27918722`）。这确认 v70 的 command→intensity 平方变换是前一轮数值下降来源；在物理强度校准后，pre-encoder 为 `0.25000195`，front 4f relay 没有实质衰减。

逐层 intensity code fraction 为：encoder layer 1 `0.28391`、encoder layer 2/OLS `0.23731`、decoder layer 1 `0.22287`、chip `0.22058`、camera `0.22021`。因此联合状态固定为 `d2nn-code-factor-attenuated`，首个有效 attenuation plane 为 `encoder_layer_2`；camera clean corr/dice/accuracy 为 `0.96002/0.92565/1.0`，locked raw 十类均可读，2/5 与 4/8 保持区分。

encoder layer 2 的 complex-field code fraction 仍为 `0.26413`，高于其 intensity fraction `0.23731`；代码尚未从 complex field 中消失，主要退化是 phase→intensity conversion。v71 不授权训练、loss/LR/scale sweep、test、加工导出或 Formal promotion。报告为 `outputs/20260720_v71/runs/isolation/v71_slm_intensity_transport_v1.json`，联合决策为 `v71_joint_decision_v1.json`；下一门只能做冻结 D2NN 的 phase-to-intensity/readout broad isolation。

### v72 frozen D2NN phase-mask causal attribution（2026-07-20）

v72 在冻结的 v69 best checkpoint 上复用 v71 的物理 `slm_amplitude²` intervention（校准 code fraction `0.25`），执行一次完整的 2×2 phase ablation：baseline、encoder layer 2 identity、decoder layer 1 identity、joint identity。四臂均使用 validation `100×10` crossed grid、locked exposure、`return_optical_fields=false`，不创建 optimizer、不更新参数、不保存新 checkpoint 或完整 2048/3072 光场；首次因内部审计参数规格错误中止的运行标记为无效 harness，修复后四臂完整运行才作为科学证据。

baseline 保持可读（camera factor `0.22021`，clean corr/dice/accuracy `0.96002/0.92565/1.0`）。encoder-2 identity 将 camera factor 提至 `0.26085`，decoder-1 identity 为 `0.24725`，joint identity 为 `0.29503`；但三种 ablation 的 locked raw 均退化为环状/纹理斑点，自动 clean corr/dice 分别为 `0.59914/0.65107`、`0.643995/0.67713`、`0.47278/0.56174`，人工十类门均失败。以同一 requested label 的 leave-one-out mean 做 global-phase 对齐后，joint 的 decoder-layer-2 code-effect SS 为 baseline 的 `1.2938×`，说明训练 phase mask 确实抑制了可见 code，但同时承担了保持数字轮廓的功能。

联合状态固定为 `phase-readability-code-tradeoff`；不能将 phase identity 结果视为部署路径，也不启动训练、phase 导出、loss/LR/scale sweep、test、加工导出或 Formal promotion。v69–v72 工件全部保持 `diagnostic-only`。下一主线仅允许设计带 foreground/semantic-band 可读性约束的 code basis，并先做冻结光路的只读上界；不得继续做 phase neutralization sweep 或重训。

### v73 semantic-support direct-command upper bound（2026-07-20）

v73 在冻结 v69–v72 的基础上，使用 clean MNIST prototype soft support 和 weighted-zero-mean low-frequency pattern 做一次 semantic-support SLM intervention；不修改 Formal/public SLM 模式、训练配置、光学器件或 checkpoint。CPU admission 的 physical SLM code fraction 为 `0.25000002`，amplitude `0.40276851`，前景内 delta energy `0.95638`，`≤20 cycles/mm` 能量 `0.99454`，one-clipping `0.02905`，checkpoint hash 与 v72 baseline 一致。

首次三次 CUDA 尝试仅因 audit harness 的 CPU/CUDA device 和 support shape 错误中止，均未形成报告或模型变更；修复后唯一完整运行才作为证据。有效运行固定为 validation `100×10`、batch 1、locked exposure、`return_optical_fields=false`，不保存完整 2048/3072 field。semantic support 的 camera corr/dice/accuracy 为 `0.95721/0.94303/1.0`，人工十类可读；但 code factor 在 encoder layer 1 首次低于门（`0.21386`），camera 仅 `0.16961`。camera absolute code-effect SS `39513.27` 低于 v72 baseline `53141.62`，label SS retention `0.99040`，所以不是 denominator artifact。

联合状态为 `semantic-support-code-factor-attenuated`，且没有相对 v72 的 absolute code improvement。v73 关闭训练、basis/phase sweep、test、加工导出、phase 导出和 Formal promotion；下一门转入 transient full-state active/exterior code-energy partition，定位 encoder layer 1 之后 code 是离开 active window、保留在 complex phase，还是在 intensity conversion 中衰减。
对应工件为 `outputs/20260720_v73/runs/admission/v73_cpu_admission_v1.json`、`outputs/20260720_v73/runs/raw_audit/v73_semantic_support_raw_v1.json`、`v73_manual_review_v1.json` 和 `v73_joint_decision_v1.json`。

### v74 transparent full-state active/exterior attribution（2026-07-20）

v74 冻结 v69–v73，复用 v73 semantic-support command，执行一次 validation `100×10` transient full-state audit；full field 只用于内存中按 code 聚合，未保存完整 2048/3072 光场。CPU admission、provenance/hash、finite、参数不变和 v73 active replay 均通过，CUDA 峰值约 `2.82/3.00 GiB`（allocated/reserved）。

在 `encoder_full_grid_input → encoder_layer_1`，intensity code SS 的 full retention 为 `0.490237`、active retention 为 `0.489444`，exterior 仅解释 `0.00155` 的 active intensity loss；raw complex 和 gauge-aligned complex full retention 分别为 `0.999999/1.000001`。因此 code 仍保留在完整 2048 state 的 complex field 中，首个有效退化是 phase→intensity conversion，而不是 active-window 空间逃逸或 2048 crop。联合状态为 `full-state-phase-to-intensity-redistribution-confirmed`。

v74 的 camera locked corr/dice/accuracy 为 `0.95721/0.94303/1.0`，复现 v73；人工十类 review 继续沿用 v73。该状态保持 `diagnostic-only`，不授权训练、phase/basis/geometry sweep、test、加工导出或 Formal promotion。下一门只能继续冻结模型的只读 phase-to-intensity attribution。

### v75 carrier–residual phase-to-intensity attribution（2026-07-20）

v75 冻结 v69–v74，完成两遍 validation `100×10` full-state carrier/residual 审计；full field 仅瞬时驻留内存，持久化工件为 scalar statistics。一次 shape harness 中止后完整重跑，最终 CPU admission、v74 replay、alias、finite、energy、hash、参数不变和 `ΔI=R+Q` 重构（最大相对误差 `1.65e-6`）均通过。

首层 `encoder_full_grid_input→encoder_layer_1` 的 intensity code-SS retention 为 `0.490237`，residual complex retention `0.999999`；quadrature share 增加 `0.22424`，real-interference signed loss contribution `1.0496`，within-label real-interference retention `0.46035`，self-term+covariance 未解释损失。联合状态固定为 `carrier-residual-quadrature-rotation-confirmed`：首因是 carrier/residual 正交相位旋转导致 coherent phase→intensity 读出损失，后续 exterior 扩散为次级结构；camera 可读性仍为 corr/dice/accuracy `0.95721/0.94303/1.0`。

该状态仍为 `diagnostic-only`，不启动训练、test、phase/加工导出或 Formal promotion。下一步最多设计一次固定双正交 coherent-reference readout 上界，不能直接训练或 sweep；v69 learned code transport 仍未证明。工件：`outputs/20260720_v75/runs/admission/v75_interference_admission_v1.json`、`outputs/20260720_v75/runs/isolation/v75_carrier_residual_interference_v1.json`、`v75_joint_decision_v1.json`。

### v76 fixed dual-quadrature coherent-reference readout（2026-07-20）

v76 的 CPU admission 链接 v73 manual、v75 admission/report/decision、v69 best checkpoint 和固定 validation split；验证了 amplitude `0.4027685085311532`、locked exposure `0.4913053810596466`、camera complex readout、100×10/batch-1 合同及 50:50 分束恒等式。首轮 CUDA 因端口能量恒等式的残差计算错误判为无效 harness；修复后从第一遍 carrier pass 完整重跑，未创建 optimizer、未更新参数、未保存 full field 或载波。

有效 replay 的 raw camera code SS/factor 为 `39513.27116/0.16960563`，与 v75 在 `rtol=1e-5` 内一致；raw corr/dice/accuracy 为 `0.95721090/0.94303381/1.0`。双通道 `[H_R,H_J]` code SS `47158.77928`、factor `0.740539`，相对 v75 raw 提升 `19.35%`，但仍只有 v72 baseline 的 `88.74%`。H_R/H_J 恒等式和两组端口能量误差均 `≤1e-5`，finite、energy、hash、参数不变和 provenance 通过。

唯一动态范围失败为 `r_plus` high saturation `0.156618`（门限 `0.08`）；`j_plus/j_minus/r_minus` 分别为 `0.003801/0.004351/0.0`。联合状态固定为 `dual-quadrature-dynamic-range-tradeoff`。这说明理想双正交通道能回收部分 quadrature code，但固定曝光下 R+ 端口发生饱和，且绝对 SS 未达到 v72 baseline，不能宣称 coherent receiver 可部署。

v76 继续保持 `diagnostic-only`，训练、test、phase/加工导出和 Formal promotion 均为 false。下一门只能进入 phase-aware D2NN 与受限 coherent receiver 的只读设计评审；禁止 LO 增益/相位/曝光 sweep、重新训练、改变光学几何或修改 Formal v56/v60。工件：`outputs/20260720_v76/runs/admission/v76_dual_quadrature_admission_v1.json`、`outputs/20260720_v76/runs/isolation/v76_dual_quadrature_readout_v1.json`、`v76_joint_decision_v1.json`。

### v77 phase-aware D2NN local gradient compatibility（2026-07-20）

v77 CPU admission 已通过，固定链接 v73/v75/v76 有效证据、v69 best checkpoint 和 validation `100×128` codes；确认 semantic-support amplitude `0.4027685085311532`、locked exposure `0.4913053810596466`、100×10/batch-1 合同、checkpoint hash、十类 manual review 和无 test split。v76 coherent receiver 分支固定为 closed，v69 learned code transport 仍为 unproven。

四 code smoke（4 codes × 10 requested labels）仅计算 phase VJP，未创建 optimizer、未更新参数、未保存 full field。finite、非零 phase gradient、phase/non-phase hash 和 parameter invariance 通过；峰值 `1.659/1.848 GiB`（allocated/reserved），joint projected code-drive ratio `1.0`，encoder L1 ratio `1.0`，joint gradient cosine `-0.4154`，一阶 readability change 为负。该结果仅表明局部 phase code/readability 方向未冲突，不能授权训练。

完整 100×10 streaming VJP 已有效完成，`streaming_code_effect_ss=39513.27116` 复现 v75 raw camera，joint/encoder L1 projected code-drive ratio 均为 `1.0` 且一阶 readability change 为负；v77 最终状态为 `phase-aware-local-co-design-admissible`。该状态只允许设计 semantic-gated direct-command residual＋phase 合同，不授权训练、test、phase/加工导出或 Formal promotion。工件：`outputs/20260720_v77/runs/admission/v77_phase_gradient_admission_v1.json`、`outputs/20260720_v77/runs/admission/v77_phase_gradient_smoke_v1.json`、`outputs/20260720_v77/runs/isolation/v77_phase_gradient_compatibility_v1.json`、`v77_joint_decision_v1.json`。

### v78 semantic-gated direct-command contract admission（2026-07-21）

v78 只读合同已实现，CPU generator-only admission 有效完成，未启动 epoch 训练。`label_template_semantic_command_residual` 复用 v69 state keys/形状，固定 v73 amplitude `0.4027685085311532`，模型内确定八维 low-frequency basis 与 template soft support；Formal student 明确拒绝该架构。

实际指标：physical SLM intensity code factor `0.00049967<0.25`，prototype support command-delta energy `0.94566<0.95`，低频能量 `0.99817`、one-clipping `0.01558`；公式误差、finite、checkpoint/hash、参数不变和 provenance 全通过。联合状态固定为 `v78-command-code-factor-insufficient`。这是有效合同失败，不是 harness failure；不得执行四样本 CUDA smoke、不得训练或 sweep。下一步冻结 v78 direct-command 分支，保留 v69 learned code transport unproven，继续关闭 coherent receiver、test、phase/加工导出和 Formal promotion。工件：`outputs/20260721_v78/runs/admission/v78_semantic_command_admission_v1.json`、`outputs/20260721_v78/runs/admission/v78_joint_decision_v1.json`。

### v78.1 semantic-gate / normalization bottleneck closure（2026-07-21）

v78.1 generator-only postmortem 有效完成，未运行 CUDA、smoke 或训练。code factor 依次为 basis coefficients `0.56152`、unit direction `0.37579`、projected pattern `0.56182`、weighted-zero-mean `0.44844`、template-support `0.05015`、normalized pattern `0.00350`、command `0.000256`、physical SLM intensity `0.000500`；最大公式误差 `1.86e-9`，provenance、finite、hash、参数不变和无 full-field artifact 通过。

联合状态为 `mixed-support-normalization-code-suppression`。首个主要 code 衰减发生在 template support，随后 per-sample max-abs normalization 进一步消除 code 差异；因此关闭当前 support＋逐样本 normalization 合同。下一步只允许设计 label-fixed supported basis，normalization 常数必须与 label 绑定且不得依赖 code/sample；training、CUDA、test、导出和 Formal promotion 继续关闭。工件：`outputs/20260721_v781/runs/isolation/v781_semantic_command_bottleneck_v1.json`、`v781_joint_decision_v1.json`。

Evidence v2 追加真实 before/after hash、state-key equality 和 transition retention；v1 保留不覆盖。absolute code SS 的首个损失在 template support，normalization 后 absolute code SS 上升但 label SS 大幅放大，因此 normalization 主要是 ratio artifact；v2 状态仍为 `mixed-support-normalization-code-suppression`。工件：`outputs/20260721_v781/runs/isolation/v781_semantic_command_bottleneck_v2.json`、`v781_joint_decision_v2.json`。

### v79 label-fixed supported-basis upper bound（2026-07-22）

v79 完成固定 label-only normalizer 的 generator-only CPU 上界，使用 learned replay 与 deterministic capacity 两臂；没有逐样本归一化、校准/sweep、optical forward、CUDA、teacher alignment 或训练。provenance、finite、formula、state-key 和 before/after hash 全通过。

learned replay physical SLM code factor `6.03e-7`、absolute code SS `0.496`；deterministic capacity factor `0.05455`、absolute code SS `48292.58`，均未达到 `0.25` / `289721.29` 门。capacity support energy `0.97080`、低频 `0.99814`、label SS 通过，但 clipping `0.20547` 失败。联合状态为 `label-fixed-supported-basis-insufficient`；关闭八维 semantic-supported basis 家族及 residual＋phase 联合训练，不运行 CUDA/smoke/training、normalization/gain/amplitude/basis sweep、test 或导出。工件：`outputs/20260721_v79/runs/admission/v79_label_fixed_supported_basis_v1.json`、`v79_joint_decision_v1.json`。

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
  --baseline docs/archive/acceptance_baseline_c99f559.json
```

Baseline 指标、阈值语义和过往结果见 [实验与验收历史](../status/experiment_history.md)。
