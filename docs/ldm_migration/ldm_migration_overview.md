# LDM 迁移总览

## 1. 迁移目标

本轮迁移的目标是把当前仓库从 `legacy` 的 optical VAE 生成主线，切换为：

`OpticalPathVAE + class-conditioned latent diffusion`

一句话结论：

**retained 设定光路与工程壳层，替换 `legacy` 生成核心。**

本轮默认边界如下：

- 数据集继续使用 `MNIST`
- `guanglu.md` 继续 retained 为物理基线文档
- 首期采用标准 `latent diffusion`
- 首期把设定光路并入第一阶段 `OpticalPathVAE`
- 首期不引入硬件噪声、像差标定或随机器件误差

## 2. 术语约定

- `legacy`：当前仓库中围绕 `LightGenVAE` 的实现与流程
- `retained`：迁移到新主线后仍继续保留并维护的内容
- `OpticalPathVAE`：显式经过设定光路、并保留 `metasurface encoder -> OLS -> metasurface decoder` 的第一阶段概率自编码器
- `latent`：`OpticalPathVAE` 编码后的低维潜空间张量
- `sample`：由带类别条件与 classifier-free guidance 的逆扩散驱动后，经 `OpticalPathVAE.decode()` 在 `camera` 平面生成的样本
- `scheduler`：控制扩散训练噪声日程和采样步进的独立组件

## 3. 当前现状

当前文档中描述的旧主线是：

- 训练入口：`LightGen_PyTorch/train.py`
- 快速训练入口：`LightGen_PyTorch/train_fast.py`
- 生成入口：`LightGen_PyTorch/inference_generate.py`
- 核心模型：`LightGen_PyTorch/models/lightgen_vae.py`

当前 `legacy` 主线的核心特征：

- 主模型是 `LightGenVAE`
- 训练目标是重构质量、KL 约束与防坍塌
- 生成不是标准扩散采样，而是先选目标 latent，再反求 SLM 输入图
- `run_manager.py`、`training_history.json`、`run_meta.json` 已形成可复用的实验管理壳层

## 4. 迁移动机

迁移到 `OpticalPathVAE + latent diffusion` 的直接原因：

- 当前主线强绑定 `LightGenVAE` 和其专属损失设计，难以直接演进为标准生成研究栈
- 当前生成链路依赖 latent prior 和输入反优化，不是标准可扩展采样接口
- 新主线既保留设定光路和芯片语义，又把扩散训练统一到标准 latent 空间接口上，并补上更标准的条件 U-Net 与 CFG 语义

## 5. 本轮保留与替换边界

### 5.1 retained

- `MNIST` 数据入口
- 当前 `200x200` 输入规约
- 设定光路顺序和宏观距离字段
- 配置加载机制
- run 目录、artifact 目录和元数据写盘机制
- 历史物理平台说明文档

### 5.2 替换

- `LightGenVAE` 主模型
- `legacy` 的 VAE 重参数化实现
- optical VAE 专属损失组合
- latent prior 标定
- `proxy_input_optimization`
- 当前生成脚本的目标 latent 反求流程

### 5.3 改造

- 训练入口
- 采样入口
- 训练配置 schema
- 第一阶段模型实现
- 评估指标组织
- checkpoint 选择语义

## 6. 目标接口

迁移后统一约定以下新接口：

- 唯一入口：`train.py`
- 配置入口：
  - `configs/physics_cfg.yaml`
  - `configs/train_ldm.yaml`
- 模型分层：
  - `OpticalPathVAE`
  - `diffusion_unet`
  - `scheduler`

以下接口统一标记为 `legacy`：

- `LightGen_PyTorch/train.py`
- `train_fast.py`
- `inference_generate.py`
- `LightGenVAE`

## 7. 阶段目标

### 阶段 1

- 保持 `MNIST -> 200x200` 输入规约不变
- 建立可独立训练和推理的 `OpticalPathVAE`
- 建立 `latent diffusion` 训练闭环
- 建立标准 `scheduler` 采样闭环

### 阶段 2

- 统一 LDM 的评估指标和 checkpoint 选择语义
- 接入现有 run/artifact 管理
- 补齐新文档和 `legacy` 边界标注

### 阶段 3

- 视需要再讨论“采样也走完整前级光路”的非标准扩展模式

## 8. 成功标准

本轮迁移文档和骨架代码完成后，应满足：

- 团队能够清楚区分 `retained`、`改造`、`legacy` 三类内容
- 新主线接口采用 `OpticalPathVAE + latent diffusion` 语义
- 第一阶段训练流与采样流可以被独立描述清楚
- `guanglu.md` 继续解释物理基线，但其字段被第一阶段模型直接引用

## 9. 文档交付顺序

1. `docs/ldm_migration/guanglu.md`
2. `docs/ldm_migration/ldm_migration_overview.md`
3. `docs/ldm_migration/ldm_reuse_matrix.md`
4. `docs/ldm_migration/ldm_architecture_spec.md`
5. `docs/ldm_migration/ldm_project_workflow.md`
