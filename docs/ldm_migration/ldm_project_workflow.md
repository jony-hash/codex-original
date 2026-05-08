# LDM 项目工作流定义

## 1. 目标

本文定义新 `LDM` 项目的默认工程工作流。它基于当前仓库的真实习惯抽象而来，但做了适度标准化，供新项目直接沿用。

默认原则如下：

- 保留 `Python + PyTorch + 脚本入口 + 本地 GPU/单机训练` 的使用方式
- 引入 `src/tests` 标准化布局
- 引入 `pytest`、`ruff`、`mypy`
- 环境管理固定为 `Conda + pip`
- Python 版本固定为 `3.11`

当前仓库 `.vscode/settings.json` 指向的 Conda 解释器实际版本是 `Python 3.11.14`，因此新项目默认锁定到 `Python 3.11.x`。

当前仓库的环境基线定义为：

- Conda 环境名：`optical-generative-models`
- Python：`3.11.x`
- PyTorch：CUDA 版 `torch 2.5.1+cu121`

这里记录的是当前验证过的环境基线，不代表代码层面硬锁版本；后续升级时应同步更新文档，但不要把某台机器的绝对解释器路径写进规范。

## 2. Project Context

```md
## 10. Project context

### Stack
- Language and version: Python 3.11
- Framework(s): PyTorch, torchvision, PyYAML, matplotlib
- Package manager: pip inside a Conda environment
- Runtime / deployment target: Local single-machine training/inference, default single-machine GPU training with explicit `--device cuda`; CPU is retained only for smoke tests and compatibility checks, Windows/VS Code friendly

### Commands
- Install: `conda create -n optical-generative-models python=3.11 -y && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && pip install -r requirements.txt && pip install -e .`
- Build: `python -m compileall src tests`
- Test (all): `pytest -q`
- Test (single file): `pytest tests/test_scheduler.py -q`
- Lint: `ruff check src tests *.py`
- Typecheck: `mypy src`
- Run locally:
  - `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage first_stage --device cuda`
  - `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage diffusion --device cuda --first-stage-checkpoint outputs/runs/.../checkpoints/first_stage_latest.pt`
  - `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage full --device cuda` 仅作为联调 / 一键串联便捷模式

`--device auto` 仅用于兼容性检查或临时排错；`--device cpu` 仅用于本地 smoke test，不是默认训练口径。

Prefer single-file or single-test runs during iteration. Full suites are for the final verification pass.

### Layout
- Source lives in: `src/ldm_project/` with root entry script `train.py`
- `train.py` remains the CLI and high-level orchestration entry; reusable helpers live in package modules
- `src/ldm_project/stage_runners.py` owns sample, first-stage training, and diffusion training execution loops
- `src/ldm_project/sampling.py` owns sample label selection, reference images/banks, final sample artifacts, and nearest-neighbor sample metrics
- `src/ldm_project/evaluation.py` owns first-stage quality metrics, balanced validation metrics, and diffusion validation summaries
- `src/ldm_project/train_utils.py` owns pure training utilities such as stage plans, logging/validation intervals, image-fusion weights, and checkpoint helper decisions
- Tests live in: `tests/`
- Do not modify: `outputs/`, `data/raw/`, `.tmp/`, `legacy/`, generated checkpoints, generated plots, cached artifacts, except configured final acceptance artifacts under `outputs/final_pt/`

### Conventions specific to this repo
- Naming: `snake_case` for files/functions/modules, `PascalCase` for classes, explicit entrypoint names like `train.py`
- Import style: Absolute imports from `src/ldm_project`; entry scripts import from the package; no cross-tree `sys.path` hacks
- Error handling pattern: Library code raises explicit exceptions (`ValueError`, `RuntimeError`) with concrete messages; CLI/entry scripts catch only at the boundary, print concise failure context, and exit non-zero
- Testing pattern and framework: `pytest`; small CPU-friendly unit tests for utilities/scheduler/config, smoke tests for model init + forward, thin integration tests for one mini-batch train/sample path

### Forbidden
- Do not hardcode interpreter paths, Conda paths, or machine-specific directories in code
- Do not rely on `.vscode` settings as the only source of runtime configuration
- Do not write outputs/checkpoints outside managed `outputs/runs/...` directories, except final acceptance artifacts placed under the configured `outputs/final_pt/` directory
- Do not import project modules by mutating `sys.path`
- Do not mix `legacy` optical VAE code directly into the new LDM core package
- Do not use full-dataset or full-epoch runs for routine iteration; use fast configs and single-test runs first
```

## 3. 关键接口

- 唯一入口：`train.py`
- 包根目录：`src/ldm_project/`
- 测试目录：`tests/`
- `legacy` 参考代码如果保留，必须放在新核心包之外，例如 `legacy/`

## 4. 为什么这样定义

这些约束直接来自当前仓库暴露出的真实问题和可复用习惯：

- 当前仓库已经证明脚本入口适合研究迭代，因此保留单文件 `train.py` 入口
- 当前仓库没有统一依赖清单，也没有 lint/typecheck 入口，因此新项目需要补齐最小工程约束
- 当前仓库依赖 `.vscode` 的解释器与 `TEMP/TMP` 设置，说明环境假设过于隐式，新项目要把这部分转成显式工作流
- 当前仓库的 smoke test 更接近“脚本验证”，新项目则应把它标准化为 `pytest` 驱动
- 当前训练口径需要从“CUDA 可用时优先”收紧为“正式训练命令显式写 `--device cuda`”，这样团队不会在 `auto` 和 `cpu` 回退上反复猜测

## 5. 验证方式

新项目建立后，至少要满足以下检查：

- `python --version` 返回 `3.11.x`
- `python -m compileall src tests` 可通过
- `pytest -q` 可通过
- 可以单独执行 `pytest tests/test_scheduler.py -q`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage first_stage --device cuda` 能完成第一阶段正式训练
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage diffusion --device cuda --first-stage-checkpoint ...` 能完成第二阶段正式训练，并自动生成最终 sample 可视化
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage full --device cuda` 仍可作为联调便捷模式运行
- 若当前终端 `torch.cuda.is_available()` 为 `False`，应先视为 GPU 可见性异常并排查 WSL2 / 驱动 / CUDA 暴露问题，而不是直接把 CPU 回退当作正式训练流程

## 6. 当前最终验收记录

当前仓库的模型验收已经完成，不应继续盲目追加训练。正式验收口径如下：

- First-stage checkpoint: `outputs/runs/20260504_134132_767348_train_first_stage_grid32_ols1ch_artifact_w055_e8/checkpoints/first_stage_best_optical.pt`
- Diffusion checkpoint: `outputs/runs/20260507_011608_845740_train_diffusion_grid32_ols1ch_artifact_w055_s20k_b128_w4_wc4x300_g45_e30_weak345n16/checkpoints/diffusion_best.pt`
- Acceptance sample run: `outputs/runs/20260507_081830_752705_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_nn_metrics`
- Latest CUDA acceptance recheck: `outputs/runs/20260508_144922_456260_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_diffusion_visual_fix_recheck`
- Final copied checkpoint folder: `outputs/final_pt/`
- Acceptance metric source: `artifacts/post_train_sample_metrics.json`
- Acceptance device: `run_meta.device == "cuda"`
- Acceptance thresholds: `overall_nearest_dice >= 0.80` and `min_class_nearest_dice >= 0.75`
- Accepted metrics: `overall_nearest_dice=0.8503`, `overall_nearest_corr=0.8662`, `min_class_nearest_dice=0.7857`
- Latest recheck metrics: `overall_nearest_dice=0.8492`, `overall_nearest_corr=0.8675`, `min_class_nearest_dice=0.7815`

正式验收使用 `nearest_neighbor` 指标，因为手写数字生成结果可以与同类样本形态不同，但仍然有效。旧的 fixed-reference `overall` / `per_class` / `per_sample` 指标继续保留在 `post_train_sample_metrics.json` 中，用于连续性和诊断，但不作为最终是否通过的唯一依据。

第二阶段训练过程图的全类别显示也已经单独验证：

- Visualization smoke run: `outputs/runs/20260508_145449_512992_train_smoke_diffusion_visual_0_9_check`
- Smoke purpose: 验证 `images/diffusion/epoch_XXXX.png` 使用固定 `0-9` 标签，而不是最后一个训练 batch 的随机子集
- Smoke config: `configs/diffusion_visual_config.json`
- Expected labels: `class_labels == [0,1,2,3,4,5,6,7,8,9]` and `reference_labels == [0,1,2,3,4,5,6,7,8,9]`
- Smoke output: `images/diffusion/epoch_0001.png`

该 smoke run 只证明可视化路径正确；它的 `first_stage_best_optical.pt` 和 `diffusion_best.pt` 来自 1 epoch 短训，不作为最终 checkpoint。

复现验收 sample 的命令必须显式使用 CUDA：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py \
  --train-config configs/train_first_stage_linear_metasurface_2000_grid32.yaml \
  --stage sample \
  --device cuda \
  --first-stage-checkpoint outputs/runs/20260504_134132_767348_train_first_stage_grid32_ols1ch_artifact_w055_e8/checkpoints/first_stage_best_optical.pt \
  --diffusion-checkpoint outputs/runs/20260507_011608_845740_train_diffusion_grid32_ols1ch_artifact_w055_s20k_b128_w4_wc4x300_g45_e30_weak345n16/checkpoints/diffusion_best.pt \
  --guidance-scale 4.5 \
  --sample-class-labels 0 1 2 3 4 5 6 7 8 9 \
  --sample-repeats-per-class 4 \
  --run-name sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_acceptance_recheck
```

后续工作默认从这个验收 checkpoint 出发：先做工程收尾、文档化和小步结构优化；只有当新的验收目标明确改变时，才启动新一轮 GPU 训练。

## 7. 与当前仓库的关系

当前仓库中的以下内容是新工作流的来源，但不应原样继承：

- `LightGen_PyTorch/train.py`
- `LightGen_PyTorch/train_fast.py`
- `LightGen_PyTorch/inference_generate.py`
- `scripts/verify_model.py`
- `.vscode/settings.json` 中的解释器与环境变量写法

新项目应保留的是“单机研究型脚本驱动”的节奏，而不是保留 `legacy` optical VAE 的目录耦合、`sys.path` 处理和环境隐式约定。

## 8. 一句话结论

新 `LDM` 项目默认采用：`Conda + pip`、`Python 3.11`、`src/ldm_project + tests`、`train.py` 单入口、`pytest + ruff + mypy` 的轻量标准化研究工作流。
