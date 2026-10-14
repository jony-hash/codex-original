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

For a clean Windows environment, use the checked-in package lock and then run
the minimum CPU regression suite described in
[Windows development and verification](docs/operations/windows_development.md).
The existing supported environment is unchanged. CPU checks establish software
contracts only; they do not authorize or validate a CUDA research run.

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

## Formal Runtime Baseline And Active Research

正式 v2 物理合同是 `532 nm`、identity OLS、完整 SLM FOV 的相干 4f `-0.4` relay、`100 × 100 @ 2 µm` phase-only D2NN、`1.50 mm` 层间距，以及后端 `-2.5` relay。理想两镜 4f 的实际 ABCD 为 `A=-0.4, B=0, C=0`；物理前向合同、raw-CMOS 可读性条件与输入/数据角色核对均保持有效。摘要和运行记录可保留来源信息，但不作为执行锁。正式 v2 架构是 `conditional_identity_logit_amplitude_v2`。

v1 的最新研究状态、capacity gate 结果和 checkpoint promotion 边界以[当前状态](docs/CURRENT_STATUS.md)及其引用的证据为准。

LightGen 当前研究问题、最新测量、冻结/历史入口和停止点均以[当前状态](docs/CURRENT_STATUS.md)及[当前交接](docs/status/current_handoff.md)为准。可复用的 BOGT 训练实现位于 [`lightgen_2d_chen_bogt_complex_ols_training.py`](src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py)，三评价器规范梯度的共享数值函数位于 [`canonical_semantics.py`](src/ldm_project/v2/application/research/lightgen/canonical_semantics.py)。最初类别1/7归因 attempt 的 `INVALID_HARNESS` 与后续冻结诊断、Generator-only 对照是不同任务和记录；后续结果没有追认旧 attempt，也没有自动授权新的研究动作。目录分类和历史入口见[研究代码导航](docs/research/project_map.md)。

`v9.10 label-template inverse prototype baseline` 仅是历史 label-template prototype：它明确忽略 code，不能证明 code-conditioned 多样性或最终 generative student readiness。历史证据见 [v9.10 record](docs/archive/acceptance_v9_10_label_template_inverse_20260622.json)。

## v2 Side-by-Side Runtime

`ldm_project.v2` 是当前 teacher-free runtime，使用类型化 plane/field、identity-anchored 8D conditional generator、冻结 MNIST evaluator 和独立子命令 CLI。它保持 amplitude SLM、finite-crop、identity OLS 与现行正式几何，不替换 `train.py`。研究实验代码与 formal v2 runtime 分开维护。使用说明见 [v2 runtime](docs/architecture/v2_runtime.md)。

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/ldm_project/v1/` | v1 canonical runtime；根包模块仅为旧导入兼容 facade |
| `src/ldm_project/v2/` | contract-first v2 runtime，以及隔离的 current、diagnostic、research、legacy 实现 |
| `src/ldm_project/v2/application/research/{lightgen,capacity,optical_manifold,shared}/` | 按研究家族放置的实现；共享运行控制位于 `shared/` |
| `configs/v1/` | v1 活动配置 |
| `configs/v2/current/` | 当前 identity v2 配置 |
| `configs/v2/legacy/`, `configs/v2/diagnostic/` | v2 兼容和诊断配置，不授权当前 acceptance |
| `configs/research/{scaled_chen,optical_manifold,capacity}/` | 当前有跟踪文件的研究配置；LightGen 研究入口没有单独的跟踪配置目录 |
| `experiments/archive/` | 不可 promotion 的历史配置/工具及其 manifest |
| `tests/student/` | Student model、objective、config、checkpoint 和训练/评估测试 |
| `tests/teacher/` | Teacher distillation export、audit、filter、readable split 和兼容测试 |
| `tests/application/` | stage runtime、CLI、输出和 checkpoint resolution 测试 |
| `tests/diagnostics/` | 当前 diagnostic boundary 验证；版本专用测试在 archive |
| `docs/` | 架构、运行指南、输出语义和历史证据 |
| `scripts/maintenance/`, `scripts/visualization/` | 维护、桥接和展示辅助脚本 |
| `scripts/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | 按研究家族组织的命令入口 |
| `tests/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | 与研究代码同类分组的测试 |
| `scripts/research/archive/project_refactor_20261001/lightgen/m0/` | 已停止的 M0 执行器；保留供历史查阅，不是当前运行入口 |
| `scripts/research/archive/project_refactor_20261001/lightgen/n3/` | 已关闭且无当前消费者的 N3 执行器；原文归档，结果仍在 `outputs/` |
| `scripts/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | 当前按家族检索的命令入口；是否可执行、冻结重放或仅供历史查阅，以当前状态和任务凭据为准 |
| `scripts/research/archive/`, `tests/archive/`, `docs/archive/` | 冻结/归档材料；用于历史重放或溯源，不隐含执行授权 |
| `docs/archive/20261001_project_refactor/` | 精简前状态、交接、项目记忆和完整历史快照 |
| `outputs/` | 当前 Windows-native runs 与已恢复历史 outputs 的可见树；恢复元数据位于 `outputs/_wsl_restore_metadata_v4/`，`outputs/final_pt/` 保持固定 |
| `data/raw/` | 不可变原始数据 |

训练、诊断和 pre-acceptance run 默认写入 `outputs/YYYYMMDD_<version>/runs/<run>/`；`<version>` 从 run-name 或 config 文件名提取，缺省为 `unversioned`。目录名前缀使用该版本当前已知最早 run 日期；若这是一个新版本，则首次创建时使用当天日期。`outputs/runs/` 已移除。

研究代码和正式运行时保持分离。新研究命令显式指定 checkpoint、config、数据与输出；历史文档和摘要用于查阅，不自动成为执行锁。

恢复历史目录由 `outputs/_wsl_restore_metadata_v4/` 的 restore plan、state 和 path map 界定；它们只用于历史溯源，不能自动作为当前 formal acceptance 输入。恢复的 `version_index.json/.md` 保留原 WSL 索引路径，不能作为当前 Windows 路径解析器。

## Documentation

- [Documentation Index](docs/README.md)：按任务选择文档和确认 source of truth。
- [Architecture And Boundaries](docs/architecture/v1_single_shot_distillation.md)：最终推理契约、模块职责和硬件化顺序。
- [Stage Guide](docs/operations/stage_guide.md)：stage 分类、命令模板、checkpoint 和验收规则。
- [Output Guide](docs/operations/output_artifact_notes.md)：图片、tensor bundle 和光学诊断语义。
- [Current Handoff](docs/status/current_handoff.md)：当前正式基线、活动研究、有效证据、允许动作或停止边界；其中带日期的旧条目仅作历史 provenance。
- [Capacity-map Hypothesis Ledger](docs/status/capacity_map_hypothesis_ledger.md)：已关闭 O0/H-SCED capacity 假设的历史归因与边界。
- [Fiber / Electronic O0 Status](docs/status/v2_fiber_image_generation_v1.md)：fiber 路线状态；electronic O0 当前决策以 capacity ledger 为准。
- [Experiment History](docs/status/experiment_history.md)：历史 baseline、负实验、seed sweep 和复验记录。
- [Research Project Map](docs/research/project_map.md)：研究代码、配置、测试和历史文档迁移表。
- [Windows Development](docs/operations/windows_development.md)：新环境锁、安装步骤和最小 CPU 回归命令。
- [Git Branch Management](docs/operations/git_branch_management.md)：稳定分支治理、现场核验与安全回档。
- [Long-Training Preflight](docs/operations/long-training-preflight.md)：长训练、模型选择和科学结论前的 fail-closed 预检。
- [AGENTS.md](AGENTS.md)：仓库执行契约与不可变边界。
