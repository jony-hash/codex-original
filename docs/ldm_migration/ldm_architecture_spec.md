# LDM v1 架构规范

## 1. 目标

LDM v1 采用以下生成架构：

`Input -> OpticalPathVAE.encode -> latent -> class-conditioned diffusion_unet + scheduler + classifier-free guidance -> latent sample -> OpticalPathVAE.decode -> camera sample`

本规范定义的 v1 主线仍然是标准 `latent diffusion`，但第一阶段不再使用普通数字 `VAE`，而是显式经过设定光路的 `OpticalPathVAE`。

## 2. 术语约定

- `OpticalPathVAE`：以设定光路和 `metasurface encoder -> OLS -> metasurface decoder` 为第一阶段骨架的概率自编码器
- `latent`：由 `OpticalPathVAE.encode()` 产生的概率潜空间张量
- `sample`：由带类别条件与 `classifier-free guidance` 的逆扩散得到，并经 `OpticalPathVAE.decode()` 解码到 `camera` 平面的输出样本
- `electronic_pattern`：电子域控制图，用来生成 `SLM` 的振幅与相位调制
- `OLS`：位于 `encoder` 和 `decoder` 之间、处理潜空间向量的固定逐点非线性层；最终验收 checkpoint 使用 `GELU`
- `linear_metasurface_grid`：最终验收 checkpoint 使用 `32x32` 复光场网格，展平后为 `2 * 32 * 32 = 2048` 个实值特征，其中 `2` 分别表示实部与虚部
- `camera_plane`：主监督和主读出平面
- `probe_plane`：`chip` 后 `3 mm` 的中间监测平面，不作为主损失平面
- `legacy`：当前 `LightGenVAE` 及其相关训练/生成链路
- `retained`：迁移后继续保留并维护的工程壳层或物理参数资产

## 3. v1 目标接口

### 3.1 入口脚本

- 唯一入口：`train.py`

### 3.2 配置分层

- `configs/physics_cfg.yaml`
  - 管理波长、像素尺寸、网格尺寸和宏观光路距离
  - 波长采用米制存储，例如 `5.32e-7` 表示 `532 nm`
  - 直接参与 `OpticalPathVAE` 前向，而不再只是背景说明
- `configs/train_ldm.yaml`
  - 管理 `OpticalPathVAE` 配置
  - 管理 diffusion 训练参数
  - 管理 `scheduler` 配置
  - 管理 smoke 训练与采样参数

### 3.3 模型分层

- `OpticalPathVAE`
  - 负责 `200x200` 输入到 latent 的压缩与解压
  - 在编码时显式经过前级设定光路
  - 在解码时显式经过 `OLS`、`metasurface decoder` 和后级设定光路
  - 最终验收版本采用 `linear_metasurface`，第一阶段可训练核心为一层线性 encoder、一个无参数 OLS 激活、一层线性 decoder
- `diffusion_unet`
  - 只处理 latent 空间张量
  - 负责在 latent 空间做带类别条件的噪声预测
  - 采用轻量 LDM 风格的 U-Net、skip connection、self-attention 与 cross-attention
- `scheduler`
  - 负责正向加噪与逆向采样时间步控制

## 4. 完整数据流

### 4.1 输入预处理

- 沿用当前 `MNIST -> 200x200` 输入规约
- 输入张量归一化到 `[0, 1]`
- 归一化后的图像作为 `electronic_pattern` 控制量，再生成 `SLM` 复振幅调制

### 4.2 `OpticalPathVAE.encode`

编码路径固定为：

`electronic_pattern -> complex-valued reflective SLM approximation -> Laser -> m1 -> Beam Expander -> m2 -> PF1 -> PBS -> BS -> SLM -> BS -> m3 -> 4f Relay -> PF2 -> metasurface encoder -> mu/logvar -> reparameterize -> latent`

实现约束如下：

- `SLM` 采用复振幅工程近似：电子控制图同时生成振幅与相位，再映射为复数透过率
- 前级光路按 `physics_cfg.yaml` 中的顺序和距离执行
- `metasurface encoder` 保留其“芯片内编码层”的语义；最终验收版本采用 `linear_metasurface`，把 PF2 复光场重采样到 `32x32` 后，将实部和虚部分别展平并拼接为 `2048` 维输入
- `latent` 定义在 `encoder` 输出之后、`OLS` 之前
- 概率瓶颈由 `mu/logvar` 两个头补齐，采样使用重参数化
- 最终验收版本的 latent 形状为 `1x25x25`；encoder 的 `2048 -> 1250` 线性输出被拆成 `mu: 1x25x25` 和 `logvar: 1x25x25`

### 4.3 `OpticalPathVAE.decode`

解码路径固定为：

`latent -> OLS -> metasurface decoder -> chip output -> chip后 3 mm probe plane -> relay lens -> camera`

实现约束如下：

- `OLS` 采用固定逐点非线性代理；最终验收 checkpoint 使用 `GELU`
- `metasurface decoder` 负责把潜空间张量重建为 `chip` 输出复场
- 最终验收版本的 decoder 为 `625 -> 2048` 线性层，再把 `2048` 拆成 `32x32` 复光场的实部和虚部，并重采样到 `200x200`
- `probe_plane` 只作为中间监测量
- `camera_plane` 是 `decode()` 的正式输出和重建监督目标
- `camera_plane` 在 v1 中采用基于 relay lens 放大率的理想成像近似：对 `probe` 复场做倒像、重采样与视场裁剪，而不是继续沿用同采样 Fresnel 末端强度

### 4.3.1 最终验收 first-stage 结构

最终验收模型版本为 `physical_first_linear_metasurface_vae_v5`，对应 checkpoint 为 `final pt/first_stage_best_optical.pt`。

| 模块 | 层数 | 结构 | 输入维度 | 输出维度 | 参数量 |
|---|---:|---|---:|---:|---:|
| Encoder | 1 | `Linear(bias=False)` | `2048 = 2 * 32 * 32` | `1250 = 2 * 1 * 25 * 25` | `2,560,000` |
| OLS | 1 | pointwise `GELU` | `1 * 25 * 25` | `1 * 25 * 25` | `0` |
| Decoder | 1 | `Linear(bias=False)` | `625 = 1 * 25 * 25` | `2048 = 2 * 32 * 32` | `1,280,000` |
| 合计 | 3 | Encoder + OLS + Decoder | - | - | `3,840,000` |

这里的 encoder 输入不是 `1x32x32`，是因为最终验收版本直接读取 PF2 的复光场；复数张量在 PyTorch 训练参数中按实部和虚部两个实值通道表示，所以 `32x32` 光场网格对应 `2x32x32` 个实值特征。若改成 `1x32x32`，等价于只保留一个实值通道，例如只读强度或只读实部，会改变物理读出语义、checkpoint 形状和 diffusion latent 分布，必须作为新实验重新训练 first stage 与 diffusion。

### 4.4 第一阶段训练

- 第一阶段先独立训练 `OpticalPathVAE`
- 训练目标以 camera 平面重建为主：
  - 核心项为 `foreground_weighted_MSE(camera_hat, input) + beta * KL(mu, logvar)`
  - 最终验收训练还加入结构、形状、光场平滑、相位平滑、能量、背景泄漏、热点、拓扑、饱和与前景伪影等辅助约束
  - 其中前景区域由输入图阈值近似得到，默认用同一套前景 mask 同时驱动 loss 与监控指标
- `beta` 使用 warmup
- 第一阶段训练完成后，`OpticalPathVAE` 作为 diffusion 的固定前端

### 4.5 latent diffusion 训练

- 先用已训练好的 `OpticalPathVAE.encode()` 把输入转为 `latent`
- 使用 `MNIST` 类别标签作为条件信号，经类别编码器映射为 `context`
- 在 latent 空间按 `scheduler` 的噪声日程采样时间步
- 对 latent 加噪后连同 `context` 一起输入 `diffusion_unet`
- `diffusion_unet` 预测噪声
- 训练时按 `cond_dropout_prob` 丢弃一部分条件，构造 classifier-free guidance 所需的无条件分支
- 以标准 denoising loss 完成训练

### 4.6 采样生成

- `train.py` 在 `diffusion` / `full` 阶段结束后，从标准随机噪声 latent 开始自动生成可视化 sample
- 训练后默认按 `0-9` 循环标签生成展示样本
- `scheduler` 驱动逆扩散过程
- 每一步分别计算条件 / 无条件噪声预测，并通过 `classifier-free guidance` 组合去噪目标
- 得到最终 `latent sample` 后，交给 `OpticalPathVAE.decode()`
- `decode()` 只经过 `OLS + metasurface decoder + 后级光路 -> camera`
- 自动 sample 不再经过前级 `SLM + 4f + encoder`，以保持标准 `latent diffusion` 语义

### 4.7 输出保存与 run 元数据记录

- 继续沿用 run 目录和 artifact 目录组织
- 训练阶段记录 `first_stage` 与 `ldm` 各自的 history
- 训练完成后的自动 sample 结果保存在同一个 train run 目录
- `run_meta.json` 中不再使用 `best_recon`、`best_generation`、`latent_prior` 等 `legacy` 字段语义

## 5. 训练流程切换

### 5.1 legacy 流程

`重构 + KL + 物理先验 + latent prior + 反求输入采样`

这是当前 `legacy optical VAE` 主线，应整体标记为 `legacy`。

### 5.2 LDM v1 流程

分为两段：

1. `OpticalPathVAE` 预训练
2. `latent diffusion` denoising 训练

首期默认不做以下事情：

- 不用 latent prior 做采样入口
- 不用 `proxy_input_optimization` 反求输入
- 不引入硬件噪声、像差标定或随机器件误差
- 不把 `legacy` 的损失语义直接搬进新主线

## 6. 建议的模块职责

### 6.1 `OpticalPathVAE`

- 独立训练与独立 checkpoint
- 提供 `encode()`、`decode()`、`forward()` 风格接口
- 显式持有 `front_path`、`OLS`、`metasurface decoder`、`back_path`
- 训练完成后作为 diffusion 的固定或半固定前端

### 6.2 `diffusion_unet`

- 只处理 latent 空间张量
- 不直接感知前级 `SLM` 或相机输出
- 不依赖 `legacy` 的 optical 模块
- 显式接收时间步与类别条件 `context`
- 通过 cross-attention 接收条件语义，并支持 classifier-free guidance

### 6.3 `scheduler`

- 独立管理时间步、噪声强度和采样步数
- 训练与采样共用统一语义
- 作为可替换组件维护，不和 `unet` 写死耦合

## 7. 兼容与边界

- `guanglu.md` retained，但其顺序与距离字段被 `OpticalPathVAE` 直接引用
- `physics_cfg.yaml` retained，并正式成为第一阶段前向配置
- `train_fast.py`、`inference_generate.py`、`LightGenVAE` 统一标记为 `legacy`
- 若后续要让采样也经过完整前级光路，需要另起非标准 `physical-pattern diffusion` 方案，不属于 v1

## 8. v1 验收条件

- 文档中已经清晰区分 `OpticalPathVAE`、`diffusion_unet`、`scheduler`
- 完整数据流已经覆盖：
  - 输入预处理
  - `OpticalPathVAE.encode`
  - `OpticalPathVAE.decode`
  - 第一阶段训练
  - latent diffusion 训练
  - 采样生成
  - 输出保存与 run 元数据记录
- 文档没有把 `legacy optical VAE` 的语义混入新主线
