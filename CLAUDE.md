# CLAUDE.md

- In the first assistant message of each new task, the first line must be exactly `【CLAUDE.md effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.

## 行为规则

1. **先想再做**：改代码前读文件，不确定时先确认，不要猜。
2. **最小改动**：只改被要求的部分，不顺手重构、不加注释、不改格式。
3. **外科手术**：只碰相关的文件。不要"顺路优化"相邻代码。
4. **验证不自夸**：不声称成功除非实际验证过。改完跑对应测试。
5. **匹配现有风格**：遵循当前代码库的命名、结构、类型注解习惯。
6. **读调用链再写**：新增代码前先读它的调用者和被调用者。
7. **自主运行**：代码修改、测试、lint、compileall、web搜索等操作直接执行，无需用户确认。仅在需要提权（sudo）、删除文件、force push 等破坏性操作时才询问用户。

### Claude Code 八荣八耻

1. 以认真查询为荣，以瞎猜接口为耻。
2. 以寻求确认为荣，以模糊执行为耻。
3. 以人类确认为荣，以臆想业务为耻。
4. 以复用现有为荣，以创造接口为耻。
5. 以主动测试为荣，以跳过验证为耻。
6. 以遵循规范为荣，以破坏架构为耻。
7. 以诚实无知为荣，以假装理解为耻。
8. 以谨慎重构为荣，以盲目修改为耻。

## 项目概要

单曝光光学生成模型，从 LDM 教师蒸馏到光学硬件部署。

- **推理路径**: `code + label → generator_e → SLM → encoder metasurface → OLS → decoder metasurface → CMOS → 生成图像`
- **主流程**: `teacher_first_stage → teacher_diffusion → teacher_distill_export → one_shot_distill → one_shot_sample`
- **关键约束**: 最终推理不依赖 diffusion U-Net、迭代采样或纯软件 VAE 解码
- **教师 vs 学生**: LDM 是教师基础设施，one-shot 是学生；不要将教师图像质量当作硬件就绪度

## 架构

三模型流水线，`train.py:main()` → `stage_runners.py` 调度：

- **教师**: `OpticalPathVAE` (first_stage.py) → `LatentDiffusionUNet` (diffusion.py) → 蒸馏导出
- **学生**: `ElectronicGeneratorE` → SLM → `BackOpticalPath` → CMOS（跳过扩散+VAE解码）

```text
训练: MNIST → VAE ↔ Diffusion ↔ 采样
蒸馏: teacher → (z₀, camera) pairs → teacher_distill_dataset.pt
推理: noise+label → GeneratorE → SLM → BackOpticalPath → CMOS
```

## 环境

- Python 3.11, PyTorch 2.5.1+cu121, WSL2 Linux
- Conda 环境: `optical-generative-models`
- 所有命令前缀: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models &&`
- 所有训练/采样命令必须使用 `--device cuda`（GPU 加速），禁止 `--device cpu`

## 关键文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | CLI 入口、stage 路由 |
| `src/ldm_project/config.py` | 所有配置类 |
| `src/ldm_project/first_stage.py` | `OpticalPathVAE` |
| `src/ldm_project/diffusion.py` | `LatentDiffusionUNet`, 噪声调度 |
| `src/ldm_project/one_shot.py` | `OneShotOpticalGenerator`, `ElectronicGeneratorE` |
| `src/ldm_project/optics.py` | Fresnel/角谱传播, `BackOpticalPath` |
| `src/ldm_project/training.py` | 各阶段 loss 函数 |
| `src/ldm_project/stage_runners.py` | 训练/采样循环 |
| `src/ldm_project/evaluation.py` | 质量指标 |
| `src/ldm_project/sampling.py` | 采样可视化 |
| `src/ldm_project/run_manager.py` | 运行目录、artifact 管理 |

## Stage 模式

| 类型 | Stage |
|------|-------|
| 教师 | `teacher_first_stage`, `teacher_diffusion`, `teacher_full`, `teacher_sample`, `teacher_distill_export`, `teacher_distill_filter` |
| 学生 | `one_shot`, `one_shot_distill`, `one_shot_sample`, `one_shot_distill_eval`, `one_shot_distill_eval_diagnostics` |

## 常用命令

```bash
# 构建检查
python -m compileall src tests

# 测试
pytest -q
pytest tests/test_train_script.py tests/test_one_shot.py -q

# Lint & Typecheck
ruff check src tests *.py
mypy src

# 教师训练
python train.py --stage teacher_first_stage --device cuda
python train.py --stage teacher_diffusion --device cuda

# One-shot 蒸馏
python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml \
  --stage one_shot_distill --device cuda \
  --distill-dataset outputs/runs/<run>/artifacts/teacher_distill_dataset.pt

# One-shot 采样
python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml \
  --stage one_shot_sample --device cuda \
  --one-shot-checkpoint outputs/runs/<run>/checkpoints/one_shot_distill_best.pt
```

## Checkpoint 优先级

1. CLI 显式路径 → 总是优先
2. `outputs/final_pt/` → 教师 checkpoint 缺省回退源
3. 学生 checkpoint: 蒸馏用 `one_shot_distill_best.pt`，基线用 `one_shot_best.pt`
4. 新训练的 artifacts 统一写入 `outputs/runs/<run_name>/`

## 光学模型约束

- SLM 是幅度控制型，不能静默重解释为相位或复数控制
- Encoder/decoder metasurface 是薄复调制层：训练时可变，制造后固定
- OLS 必须保持可替换接口: `complex field in → OLS/fiber → complex field out`
- CMOS 输出是主学生输出，仿真要反映真实 `SLM command → CMOS image` 行为

## 禁止行为

- 不硬编码解释器路径、Conda 路径、Windows 路径、机器特定目录
- 不修改 `sys.path`，使用 `pip install -e .`
- 不将 legacy 代码混入核心包
- 不将 checkpoint 写到 `outputs/runs/` 之外
- 不将教师 LDM 图像质量当作学生硬件就绪度
- 不将依赖迭代软件降噪的推理结果称为"单曝光"

## Superpowers 技能库

Superpowers (v5.1.0) 已安装。复杂任务按需调用：`brainstorming` → `writing-plans` → `executing-plans` → `test-driven-development`，调试用 `systematic-debugging`。

自检：每次代码修改任务前判断简单/复杂，复杂则立即调用对应 Skill。

## 何时询问

在以下情况先问再动手：
- 两种合理解释会导致不同输出
- 改动影响 stage、checkpoint、输出目录、CLI 参数或教师/学生边界
- 改动影响 SLM 类型、metasurface 可训性、OLS 行为、CMOS 标定
- 用户目标与字面要求冲突
- 需要密钥、凭证、硬件访问或不可达的外部资源
