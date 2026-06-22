# Single-Exposure Optical Generative Model

## What This Repository Builds

本项目把 LDM teacher 的生成能力蒸馏到 D2NN one-shot student。Teacher 负责先验、训练目标和对照；最终 inference 必须由学生在单次光学传播和单次 CMOS 曝光中完成。

```text
noise/code + optional label -> generator_e -> amplitude SLM command
-> encoder metasurface -> OLS/fiber -> decoder metasurface
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

## Current Candidate

`v9.10 label-template inverse prototype baseline` 已通过受控 CUDA strict audit。它从 checkpoint 内的类别 intensity template 生成实际 amplitude SLM command，明确忽略 code，只验证 command calibration 与完整非理想 relay 光路；不能证明 code-conditioned 多样性，也不是最终 generative student 验收。审计证据见 [v9.10 acceptance record](docs/ldm_migration/acceptance_v9_10_label_template_inverse_20260622.json)，复验入口和失败停止规则见 [Stage Guide](docs/ldm_migration/stage_guide.md#v910-prototype-baseline-audit)。

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/ldm_project/` | Teacher、student、optics、training、evaluation 与 orchestration 核心代码 |
| `configs/` | Training、evaluation、export、audit 与 smoke 配置 |
| `tests/` | 行为测试和仓库契约测试 |
| `docs/` | 架构、运行指南、输出语义和历史证据 |
| `tools/` | 聚焦的诊断与受控复验工具 |
| `scripts/` | 独立展示或工作流辅助脚本 |
| `outputs/` | 不可变 managed run artifacts；不得编辑、删除或回填 |
| `data/raw/` | 不可变原始数据 |
| `legacy/` | 历史实现；不得混入 LDM core package |

诊断和 pre-acceptance run 写入 `/tmp`；已接受或正式配置的 managed run 写入 `outputs/runs/<run>/`。

## Documentation

- [Documentation Index](docs/README.md)：按任务选择文档和确认 source of truth。
- [Architecture And Boundaries](docs/ldm_migration/single_shot_ldm_distillation.md)：最终推理契约、模块职责和硬件化顺序。
- [Stage Guide](docs/ldm_migration/stage_guide.md)：stage 分类、命令模板、checkpoint 和验收规则。
- [Output Guide](docs/output_JIESHI.md)：图片、tensor bundle 和光学诊断语义。
- [Experiment History](docs/ldm_migration/experiment_history.md)：历史 baseline、负实验、seed sweep 和复验记录。
- [AGENTS.md](AGENTS.md)：仓库执行契约与不可变边界。
