# Single-Exposure Optical Generative Model

## What This Repository Builds

本项目保留两条有意分离的运行时：v1 将 LDM teacher 蒸馏到 D2NN one-shot student；正式 v2 是无 LDM 蒸馏的 conditional generator 合同。OLS-native 是独立的活动研究分支，不等于 formal v3，也不改变 v2 正式合同。v1 和 v2 共享物理边界定义，但不共享训练输入或 evidence。

```text
noise/code + optional label -> generator_e -> amplitude SLM command
-> encoder metasurface -> linear identity OLS -> decoder metasurface
-> CMOS single exposure -> generated image
```

这是正式路径的设计合同示意，不是“生成能力已通过”的证据。最终路径不依赖 diffusion denoising、迭代 latent sampling、software-only VAE decoding 或 inference-time optimization。

## Supported Workflow

v1 主流程是：

```text
teacher_first_stage -> teacher_diffusion -> teacher_distill_export
-> one_shot_distill -> one_shot_sample -> one_shot_distill_eval
```

Teacher stages 只提供 v1 训练基础设施。加载 student checkpoint 的 `one_shot_sample` 和 `one_shot_distill_eval` 才能形成 v1 单次曝光证据；v2 命令见 [v2 runtime](docs/architecture/v2_runtime.md)。

## Quick Start

Python 要求和依赖来自 `pyproject.toml`。Windows-native 项目环境（PowerShell）：

```powershell
Set-Location D:\codex-original
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$env:Path = "$envRoot;$envRoot\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\Lib\site-packages\torch\lib;$env:Path"
$py = Join-Path $envRoot 'python.exe'
```

运行 CUDA compatibility smoke：

```powershell
& $py train.py --train-config configs/v1/smoke/train_one_shot_d2nn_smoke.yaml `
  --stage one_shot --device cuda
```

该 smoke 是 v1 的 `diagnostic-only` compatibility check，不是正式验收，也不能替代 v1 distillation 或 v2 conditional evidence 链路。

## Evidence Boundaries

- `teacher_sample` 只说明 teacher/baseline 可用，不能证明 final single-exposure student hardware ready。
- Direct MNIST `one_shot`、CPU/CUDA smoke 和 `dense_linear` 只用于 baseline、诊断或兼容检查。
- `one_shot_distill` 只表示训练完成；正式证据必须加载明确的 student checkpoint。
- Teacher audit、readability filter 和单环节诊断不能替代 `one_shot_sample` 与 `one_shot_distill_eval`。
- 正式 CUDA 训练和验收所需 checkpoint、dataset 或硬件缺失时应视为 blocker，不应静默降级。

## Formal Runtime Baseline And Active Research Status

正式 v2 物理合同是 `532 nm`、identity OLS、完整 SLM FOV 的相干 4f `-0.4` relay、`100 × 100 @ 2 µm` phase-only D2NN、`1.50 mm` 层间距，以及后端 `-2.5` relay。理想两镜 4f 的实际 ABCD 为 `A=-0.4, B=0, C=0`；物理前向合同、raw-CMOS 人眼 gate 与严格 provenance 均保持有效。正式 v2 架构是 `conditional_identity_logit_amplitude_v2`。

v1 研究线已冻结在 v79：固定 label-only normalization 的八维 supported-basis capacity arm 仅得到 physical-SLM code factor `0.05455 < 0.25`，且 clipping `0.20547 > 0.08`。这不是 harness failure，而是当前 basis 家族的有效 capacity gate 失败；learned code transport 未证明，因而没有可 promotion 的 student checkpoint。

活动研究分支为 `research/lightgen-ols-native-semantic-generation-v1`：N1/N2 已完成，N3 producer baseline 与两轮 selector calibration 均未得到可用的新模型族，N4 未决，N5 与生成仍 blocked，N6 物理实现保持独立冻结。最新 V2 校准已登记为 `MIXED_V2`（NULL false-positive rate `0/200`；local target 未达成），因此不存在可直接继续的 selector-refinement 任务；若要重新研究来源归因，必须另立独立合同并重新绑定 evidence。详见 [当前交接](docs/status/current_handoff.md)。

`v9.10 label-template inverse prototype baseline` 仅是历史 label-template prototype：它明确忽略 code，不能证明 code-conditioned 多样性或最终 generative student readiness。历史证据见 [v9.10 record](docs/archive/acceptance_v9_10_label_template_inverse_20260622.json)。

## v2 Side-by-Side Runtime

`ldm_project.v2` 是当前 teacher-free contract-first runtime，使用类型化 plane/field、identity-anchored 8D conditional generator、冻结 MNIST evaluator、严格 checkpoint/evidence manifest 和独立子命令 CLI。它保持 amplitude SLM、finite-crop、identity OLS 与现行正式几何，不替换 `train.py`。旧 teacher/prototype/capacity 与 LightGen 路径仍以显式 `legacy`/`diagnostic` 兼容实现存在，但不能签发 current acceptance；archive 保存不可授权的历史正文、配置和 evidence。使用说明见 [v2 runtime](docs/architecture/v2_runtime.md)。

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/ldm_project/v1/` | v1 canonical runtime；根包模块仅为旧导入兼容 facade |
| `src/ldm_project/v2/` | contract-first v2 runtime，以及隔离的 current、diagnostic、research、legacy 实现 |
| `src/ldm_project/v2/application/research/` | research-only lifecycle 实现；不改变 formal identity 合同 |
| `configs/v1/` | v1 活动配置 |
| `configs/v2/current/` | 当前 identity v2 配置 |
| `configs/v2/legacy/`, `configs/v2/diagnostic/` | v2 兼容和诊断配置，不授权当前 acceptance |
| `configs/research/` | 研究生命周期配置；按活动、冻结可复现或历史参考分类，不授权 formal promotion |
| `experiments/archive/` | 不可 promotion 的历史配置/工具及其 manifest |
| `tests/student/` | Student model、objective、config、checkpoint 和训练/评估测试 |
| `tests/teacher/` | Teacher distillation export、audit、filter、readable split 和兼容测试 |
| `tests/application/` | stage runtime、CLI、输出和 checkpoint resolution 测试 |
| `tests/diagnostics/` | 当前 diagnostic boundary 验证；版本专用测试在 archive |
| `docs/` | 架构、运行指南、输出语义和历史证据 |
| `scripts/maintenance/`, `scripts/visualization/` | 维护、桥接和展示辅助脚本 |
| `scripts/research/` | 绑定明确 evidence consumer 的研究脚本；不自动选择 latest |
| `outputs/` | 当前 Windows-native runs 与已恢复历史 outputs 的可见树；恢复元数据位于 `outputs/_wsl_restore_metadata_v4/`，`outputs/final_pt/` 保持固定 |
| `data/raw/` | 不可变原始数据 |

训练、诊断和 pre-acceptance run 默认写入 `outputs/YYYYMMDD_<version>/runs/<run>/`；`<version>` 从 run-name 或 config 文件名提取，缺省为 `unversioned`。目录名前缀使用该版本当前已知最早 run 日期；若这是一个新版本，则首次创建时使用当天日期。`outputs/runs/` 已移除。

研究和 artifact 生命周期只有三种有效阅读方式：活动条目有当前 consumer 与下一决策；冻结条目只用于可复现比较；历史条目只用于 provenance。文件名、日期、磁盘占用或测试计数不能单独把条目标为废弃。

恢复历史目录由 `outputs/_wsl_restore_metadata_v4/` 的 restore plan、state 和 path map 界定；它们只用于历史溯源，不能自动作为当前 formal acceptance 输入。恢复的 `version_index.json/.md` 保留原 WSL 索引路径，不能作为当前 Windows 路径解析器。

## Documentation

- [Documentation Index](docs/README.md)：按任务选择文档和确认 source of truth。
- [Architecture And Boundaries](docs/architecture/v1_single_shot_distillation.md)：最终推理契约、模块职责和硬件化顺序。
- [Stage Guide](docs/operations/stage_guide.md)：stage 分类、命令模板、checkpoint 和验收规则。
- [Output Guide](docs/operations/output_artifact_notes.md)：图片、tensor bundle 和光学诊断语义。
- [Current Handoff](docs/status/current_handoff.md)：当前正式基线、活动研究、有效证据、允许动作或停止边界；其中带日期的旧条目仅作历史 provenance。
- [Capacity-map Hypothesis Ledger](docs/status/capacity_map_hypothesis_ledger.md)：已关闭 O0/H-SCED capacity 假设的历史归因与边界；当前 N3 入口见 Current Handoff。
- [Fiber / Electronic O0 Status](docs/status/v2_fiber_image_generation_v1.md)：fiber 路线状态；electronic O0 当前决策以 capacity ledger 为准。
- [Experiment History](docs/status/experiment_history.md)：历史 baseline、负实验、seed sweep 和复验记录。
- [Git Branch Management](docs/operations/git_branch_management.md)：稳定分支治理、现场核验与安全回档。
- [Long-Training Preflight](docs/operations/long-training-preflight.md)：长训练、模型选择和科学结论前的 fail-closed 预检。
- [AGENTS.md](AGENTS.md)：仓库执行契约与不可变边界。
