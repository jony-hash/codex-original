# Single-Exposure Optical Generative Model

## What This Repository Builds

本项目把 LDM teacher 的生成能力蒸馏到 D2NN one-shot student。Teacher 负责先验、训练目标和对照；最终 inference 必须由学生在单次光学传播和单次 CMOS 曝光中完成。

```text
noise/code + optional label -> generator_e -> amplitude SLM command
-> encoder metasurface -> linear identity OLS -> decoder metasurface
-> CMOS single exposure -> generated image
```

最终路径不依赖 diffusion denoising、迭代 latent sampling、software-only VAE decoding 或 inference-time optimization。

## Supported Workflow

支持的主流程是：

```text
teacher_first_stage -> teacher_diffusion -> teacher_distill_export
-> one_shot_distill -> one_shot_sample -> one_shot_distill_eval
```

Teacher stages 只提供训练基础设施。加载 student checkpoint 的 `one_shot_sample` 和 `one_shot_distill_eval` 才能形成最终单次曝光学生证据。完整命令、必需输入和 checkpoint 规则见 [Stage Guide](docs/ldm_migration/stage_guide.md)。

## Quick Start

Python 要求和依赖来自 `pyproject.toml`。激活项目环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models
```

运行 CUDA compatibility smoke：

```bash
python train.py --train-config configs/train_one_shot_d2nn_smoke.yaml \
  --stage one_shot --device cuda
```

该 smoke 是 `diagnostic-only` compatibility check，不是正式验收，也不能替代 teacher distillation 和 student sample/eval 链路。

## Evidence Boundaries

- `teacher_sample` 只说明 teacher/baseline 可用，不能证明 final single-exposure student hardware ready。
- Direct MNIST `one_shot`、CPU/CUDA smoke 和 `dense_linear` 只用于 baseline、诊断或兼容检查。
- `one_shot_distill` 只表示训练完成；正式证据必须加载明确的 student checkpoint。
- Teacher audit、readability filter 和单环节诊断不能替代 `one_shot_sample` 与 `one_shot_distill_eval`。
- 正式 CUDA 训练和验收所需 checkpoint、dataset 或硬件缺失时应视为 blocker，不应静默降级。

## Current Mainline Status

当前正式物理合同是 `532 nm`、identity OLS、完整 SLM FOV 的相干 4f `-0.4` relay、`100 × 100 @ 2 µm` phase-only D2NN、`1.50 mm` 层间距，以及后端 `-2.5` relay。理想两镜 4f 的实际 ABCD 为 `A=-0.4, B=0, C=0`；v58.2 仅是 CUDA 数值自洽证据，v58.3 才在完整 `500→200 µm` FOV 上验收闭式场、实测 pupil PSF 与独立性。未得到 v58.3 的 `physical-forward-model-valid` 报告前不得训练。

v56/v60 在该合同上完成了两次同配置短训练；strict provenance 有效，但真实最终 CMOS 都不可读，因此没有可 promotion 的学生 checkpoint。v61 对完整 readable split 的原生双通道 ridge 上界虽显示冻结 camera intensity 可被离线软件恢复为 teacher target，却不改变 raw CMOS 人眼失败；下一门只能隔离类别模板与逐实例信息，不能继续训练。当前状态与证据见 [v61 主线交接](docs/ldm_migration/current_handoff.md)。

`v9.10 label-template inverse prototype baseline` 仅是历史 label-template prototype：它明确忽略 code，不能证明 code-conditioned 多样性或最终 generative student readiness。历史证据见 [v9.10 record](docs/ldm_migration/acceptance_v9_10_label_template_inverse_20260622.json)。

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/ldm_project/optics/` | 物理复场、传播、sampling、relay、devices、正式 D2NN 与独立 reference |
| `src/ldm_project/student/` | 正式 `FormalOneShotOpticalGenerator`、canonical generator、兼容 student、objective、training、evaluation、artifact 与 checkpoint contract |
| `src/ldm_project/student/checkpoints.py` | 完整 checkpoint 与显式 optical/generator subset warm-start |
| `src/ldm_project/student/generator.py` | `ElectronicGeneratorE` 唯一 canonical 实现 |
| `src/ldm_project/student/experimental/` | phase-amplitude、generic resize 和 diagnostic compatibility 实现 |
| `src/ldm_project/configuration/` | config schema、loading、validation 的内部归属；`config.py` 保持公开 facade |
| `src/ldm_project/teacher/` | first-stage、diffusion、teacher sampling/training/distill |
| `src/ldm_project/teacher/common.py` | teacher distillation algorithm implementation; `distill_legacy.py` and `distill.py` are compatibility facades |
| `src/ldm_project/application/` | CLI、输出布局、run metadata 与按 stage 懒构造的调度 |
| `src/ldm_project/application/stage_runtimes.py` | data-only、student-only、teacher runtime footprints; no model imports during classification |
| `configs/` | 仅活动 Golden Path、physics、teacher、smoke 与 v52/v55/v56 配置 |
| `experiments/archive/` | 不可 promotion 的历史配置/工具及其 manifest |
| `tests/` | 行为测试和仓库契约测试 |
| `docs/` | 架构、运行指南、输出语义和历史证据 |
| `tools/` | 聚焦的诊断与受控复验工具 |
| `scripts/` | 独立展示或工作流辅助脚本 |
| `outputs/` | 按真实日期前缀版本目录分组的 managed run artifacts；`outputs/final_pt/` 保持固定 |
| `data/raw/` | 不可变原始数据 |
| `legacy/` | 历史实现；不得混入 LDM core package |

训练、诊断和 pre-acceptance run 默认写入 `outputs/YYYYMMDD_<version>/runs/<run>/`；`<version>` 从 run-name 或 config 文件名提取，缺省为 `unversioned`。目录名前缀使用该版本当前已知最早 run 日期；若这是一个新版本，则首次创建时使用当天日期。`outputs/runs/` 已移除。

## Documentation

- [Documentation Index](docs/README.md)：按任务选择文档和确认 source of truth。
- [Architecture And Boundaries](docs/ldm_migration/single_shot_ldm_distillation.md)：最终推理契约、模块职责和硬件化顺序。
- [Stage Guide](docs/ldm_migration/stage_guide.md)：stage 分类、命令模板、checkpoint 和验收规则。
- [Output Guide](docs/output_JIESHI.md)：图片、tensor bundle 和光学诊断语义。
- [Current Handoff](docs/ldm_migration/current_handoff.md)：当前合同、有效证据、停止规则和下一决策门。
- [Experiment History](docs/ldm_migration/experiment_history.md)：历史 baseline、负实验、seed sweep 和复验记录。
- [AGENTS.md](AGENTS.md)：仓库执行契约与不可变边界。
