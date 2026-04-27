# LDM 迁移复用矩阵

## 1. 术语约定

- `retained`：迁移后继续保留并维护
- `改造`：迁移后保留壳层或接口，但内部语义需要重写
- `废弃归档`：仅作为 `legacy` 参考，不继续作为主线维护

## 2. 复用矩阵

| 子系统 | retained | 改造 | 废弃归档 |
|---|---|---|---|
| 数据 | `MNIST` 数据入口；`200x200` 输入规约 | 后续若换数据集，再扩展 dataset 抽象 | 当前无必须直接废弃的数据模块 |
| 配置 | `physics_cfg.yaml` 中的光路顺序、宏观距离和读出定义 | 新增 `train_ldm.yaml`；将 `physics_cfg` 从背景配置改为第一阶段前向配置 | `legacy` 训练配置中的 latent prior 等字段 |
| 训练 | 训练入口分层思路；fast/full 模式作为可参考运行方式 | 新建 `train.py`；按 `OpticalPathVAE -> diffusion` 双阶段重写训练闭环，并在训练结束后自动生成 sample 可视化 | `LightGen_PyTorch/train.py`、`train_fast.py` 围绕 `LightGenVAE` 的训练闭环 |
| 推理 / 采样 | 输出保存、artifact 目录组织思路 | 将采样并入 `train.py` 的训练后自动可视化流程，再经 `OpticalPathVAE.decode()` 输出 `camera` 图 | `inference_generate.py`、`proxy_input_optimization`、目标 latent 反求流程 |
| 模型 | 设定光路顺序与 `metasurface encoder -> OLS -> metasurface decoder` 语义 | 新建 `OpticalPathVAE`、`diffusion_unet`、`scheduler` | `models/lightgen_vae.py` 及其 `legacy` VAE 主线 |
| 评估 | 历史曲线写盘和结果汇总思路 | 评估指标改成 `camera` 平面重建和 latent diffusion 语义；重写 best checkpoint 选择标准 | `optical_vae_loss.py`、latent prior readiness、当前 generation collapse 语义 |
| 实验管理 | `run_manager.py` 的 run 目录、artifact 目录、`run_meta.json`、`training_history.json` 管理思路 | 元数据字段改成 `first_stage` / `ldm` 术语；补充 `probe` 与 `camera` 读出记录 | 当前 best_recon / best_generation / latent_prior 相关字段 |
| 历史文档 | `guanglu.md` 作为物理平台基线 retained | `ldm_architecture_spec.md`、`ldm_migration_overview.md` 改写为 `OpticalPathVAE + latent diffusion` 主线说明 | 任何把 `LightGenVAE` 写成“当前唯一真实主线”的旧描述 |

## 3. 必须覆盖的七类内容

本矩阵已覆盖以下七类主子系统：

- 数据
- 配置
- 训练
- 推理
- 评估
- 实验管理
- 历史文档

另补充了一行“模型”，因为这是本次迁移变化最大的区域。

## 4. 重点判断

### 4.1 直接 retained

- `MNIST` 数据入口
- `200x200` 输入规约
- 设定光路顺序与宏观距离字段
- run / artifact 元数据管理
- 物理光路基线文档

### 4.2 必须改造

- 第一阶段模型实现
- 训练入口与采样入口
- 配置 schema
- 评估指标与 checkpoint 语义
- `run_meta.json` 的字段命名

### 4.3 应归档为 `legacy`

- `LightGenVAE`
- `legacy` VAE 专属损失
- latent prior 标定
- 反求 SLM 输入图的采样方式

## 5. 迁移时的执行原则

- 不复用 `legacy` 的 latent 语义来命名新模块
- 不把 `best_recon` 和 `best_generation` 直接照搬到新主线
- 保留工程壳层时，优先复用目录组织和元数据写盘逻辑，而不是复用旧评价逻辑
- `guanglu.md` 只解释物理背景与光路基线，不解释扩散训练细节
