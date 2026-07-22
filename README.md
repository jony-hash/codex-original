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

当前正式物理合同是 `532 nm`、identity OLS、完整 SLM FOV 的相干 4f `-0.4` relay、`100 × 100 @ 2 µm` phase-only D2NN、`1.50 mm` 层间距，以及后端 `-2.5` relay。理想两镜 4f 的实际 ABCD 为 `A=-0.4, B=0, C=0`；物理前向合同、raw-CMOS 人眼 gate 与严格 provenance 均保持有效。

v1 研究线已冻结在 v79：固定 label-only normalization 的八维 supported-basis capacity arm 仅得到 physical-SLM code factor `0.05455 < 0.25`，且 clipping `0.20547 > 0.08`。这不是 harness failure，而是当前 basis 家族的有效 capacity gate 失败；learned code transport 未证明，因而没有可 promotion 的 student checkpoint。完整当前状态、停止规则和下一允许动作见 [v79/v2 主线交接](docs/ldm_migration/current_handoff.md)。

`v9.10 label-template inverse prototype baseline` 仅是历史 label-template prototype：它明确忽略 code，不能证明 code-conditioned 多样性或最终 generative student readiness。历史证据见 [v9.10 record](docs/ldm_migration/acceptance_v9_10_label_template_inverse_20260622.json)。

## v2 Side-by-Side Runtime

`ldm_project.v2` 是新的 contract-first 旁路实现，使用类型化 plane/field、严格 checkpoint manifest、8 维内容绑定 teacher 数据、prototype/capacity gate、固定曝光人审工件和独立子命令 CLI。它保持 amplitude SLM、finite-crop、identity OLS 与现行正式几何，但尚未替换 `train.py`；CPU 竖切、单样本 CUDA smoke 和无更新 preflight 都不构成正式训练、人工验收或 student promotion。使用说明见 [v2 runtime](docs/v2/README.md)。

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
| `src/ldm_project/teacher/{export,label_audit,filter,readable}.py` | canonical teacher distillation workflows |
| `src/ldm_project/teacher/common.py` | compatibility facade; shared helpers live in `distill_support.py` |
| `src/ldm_project/evaluation/d2nn*.py` | neutral D2NN tensor/metric primitives used by Student and diagnostics |
| `src/ldm_project/application/` | CLI、输出布局、run metadata 与按 stage 懒构造的调度 |
| `src/ldm_project/application/stage_runtimes.py` | stage footprint registry 与真实 runtime materialization；`StageContext` 仅为兼容适配器 |
| `configs/` | 仅活动 Golden Path、physics、teacher、smoke 与 v52/v55/v56 配置 |
| `experiments/archive/` | 不可 promotion 的历史配置/工具及其 manifest |
| `tests/student/` | Student model、objective、config、checkpoint 和训练/评估测试 |
| `tests/teacher/` | Teacher distillation export、audit、filter、readable split 和兼容测试 |
| `tests/application/` | stage runtime、CLI、输出和 checkpoint resolution 测试 |
| `tests/diagnostics/` | v16–v63 diagnostic-only 版本验证 |
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
- [Git Branch Management](docs/git_branch_management.md)：分支谱系、v1/v2 差异、远端发布状态与安全回档。
- [AGENTS.md](AGENTS.md)：仓库执行契约与不可变边界。
