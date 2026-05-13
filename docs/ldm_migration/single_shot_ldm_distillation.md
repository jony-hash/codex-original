# 基于 LDM 蒸馏的单次曝光光学生成模型实验流程

## 1. 核心结论

本实验流程的目标不是继续把 `LDM` 作为最终推理模型，而是把已训练好的 `LDM` 用作教师模型，将其生成能力蒸馏到一个可实际部署的单次曝光光学生成学生模型中。

最终推理路径固定为：

```text
随机低维码 / noise + 可选类别条件
        ↓
电子 generator_e
        ↓
振幅 SLM 控制图
        ↓
encoder 单层线性超表面
        ↓
OLS：当前用 GELU 代理，后续替换为单模 / 多模光纤
        ↓
decoder 单层线性超表面
        ↓
CMOS 单次曝光
        ↓
生成图像
```

这条路径应被视为后续实际光学实验的主路径。当前仓库中的 `latent diffusion` 路径保留为教师模型、样本分布来源和蒸馏目标，不再作为最终硬件推理路径。

## 2. 实验动机

当前 `LDM v1` 的正式推理流程是：

```text
随机 latent 噪声
        ↓
diffusion U-Net 多步反向去噪
        ↓
latent sample
        ↓
OpticalPathVAE.decode
        ↓
camera_plane
```

该流程适合软件生成和模型验收，但不符合实际光学系统中“SLM 输入一次、光路传播一次、CMOS 曝光一次”的实验目标。

实际实验中，encoder 超表面、OLS 光纤和 decoder 超表面应被视为一条连续物理光路，而不是可以任意拆开的 `encode()` 与 `decode()` 两个软件模块。因此后续实验应转向：

```text
训练时：LDM teacher 教 student 如何生成。
推理时：student 通过 SLM + 超表面 + OLS/光纤 + CMOS 单次曝光生成。
```

## 3. 术语约定

- `Teacher LDM`：当前仓库已训练或后续继续训练的 `latent diffusion` 模型，用来提供生成样本、类别条件目标和分布先验。
- `Student one-shot optical generator`：最终部署的单次曝光光学生成模型。
- `generator_e`：电子域轻量生成器，输入低维随机码和可选类别条件，输出振幅 SLM 控制图。
- `振幅 SLM`：实验使用的空间光调制器，只直接控制入射光场振幅。若存在残余相位，应在标定模型中显式记录。
- `encoder metasurface`：单层线性超表面，作为光路中的第一片可设计 / 可训练薄层复调制器。
- `OLS`：位于 encoder 与 decoder 之间的非线性或光纤传播模块。当前先用 `GELU` 代理；后续可替换为单模光纤或多模光纤模型。
- `decoder metasurface`：单层线性超表面，作为光路中的第二片薄层复调制器。
- `CMOS`：最终强度读出平面，也是训练和实验验收的主输出平面。
- `digital twin`：真实光路的可微仿真代理，包含 SLM 响应、传播、超表面、OLS/光纤、相机响应和误差项。

## 4. 总体阶段图

```text
阶段 1：训练 / 准备 LDM teacher
        ↓
阶段 2：理想仿真环境下蒸馏 one-shot student
        ↓
阶段 3：加入非理想物理误差做鲁棒训练
        ↓
阶段 4：搭建实际光路原型并采集响应
        ↓
阶段 5：硬件标定 + digital twin 修正 + camera-in-the-loop 微调
        ↓
阶段 6：真实光路单次曝光推理与验收
```

注意：真实环境偏差不能只在纸面上假设。必须先搭建或确定实际光路原型，采集真实 `SLM input -> CMOS output` 响应，再进行硬件标定和闭环微调。

## 5. 阶段 1：Teacher LDM 训练与样本生成

教师模型可以沿用当前仓库的 LDM 路径：

```text
真实图像
        ↓
OpticalPathVAE.encode
        ↓
latent
        ↓
训练 diffusion U-Net 预测噪声
```

教师采样路径为：

```text
随机 latent 噪声 + 类别条件
        ↓
diffusion U-Net 多步反向去噪
        ↓
latent sample
        ↓
OpticalPathVAE.decode
        ↓
teacher camera target
```

教师模型的作用：

- 提供比单纯 MNIST 样本更平滑的生成分布。
- 提供类别条件目标，例如输入类别 `3` 时生成数字 `3` 的多种写法。
- 作为学生模型训练初期的软目标，降低直接训练光学学生模型的难度。

教师模型不进入最终硬件推理。

## 6. 阶段 2：理想仿真蒸馏 Student

学生模型训练路径为：

```text
随机低维码 z + 可选类别 label
        ↓
generator_e
        ↓
SLM command S(x,y), S ∈ [0, 1]
        ↓
振幅 SLM：E_slm = A_slm(S) · E_in
        ↓
自由空间传播
        ↓
encoder 单层线性超表面
        ↓
传播到 OLS 平面
        ↓
OLS / GELU 代理
        ↓
传播到 decoder
        ↓
decoder 单层线性超表面
        ↓
传播到 CMOS
        ↓
camera_hat
```

蒸馏目标为：

```text
camera_hat ≈ teacher_camera_target
```

也可以在早期使用真实 MNIST 作为目标：

```text
camera_hat ≈ real_mnist_image
```

推荐训练策略：

1. 先用真实 MNIST 目标训练，让学生模型学会生成可辨数字。
2. 再引入 LDM teacher 目标，让学生模型学习更丰富的生成分布。
3. 最终混合真实样本监督与 teacher 蒸馏，避免学生只复制教师伪影。

## 7. 阶段 3：非理想物理增强

理想仿真训练完成后，必须加入非理想物理增强，否则仿真到实物偏差通常会过大。

建议加入的扰动包括：

- SLM 振幅响应非线性：`A_slm(S)` 不等于理想线性 `S`。
- 振幅 SLM 残余相位：`E_slm = A_slm(S) · exp(i φ_residual(S)) · E_in`。
- 激光强度不均匀和入射波前倾斜。
- encoder / decoder 超表面加工误差。
- 超表面平移、旋转、缩放和离焦。
- 传播距离误差。
- OLS / 光纤扰动、温漂和弯曲引起的响应变化。
- CMOS 暗电流、读出噪声、饱和、gamma 响应和像素非均匀性。

鲁棒训练路径为：

```text
z + label
        ↓
generator_e
        ↓
带随机非理想参数的 digital twin
        ↓
camera_hat
        ↓
distillation / reconstruction / physical regularization losses
```

这一阶段的目标不是让仿真图像最漂亮，而是让学生模型对真实实验误差不敏感。

## 8. 阶段 4：实际光路原型与硬件标定

在正式实验前，应搭建实际光路原型：

```text
Laser
        ↓
振幅 SLM
        ↓
encoder 单层线性超表面
        ↓
OLS：单模 / 多模光纤或当前等效模块
        ↓
decoder 单层线性超表面
        ↓
CMOS
```

需要采集的标定数据：

- SLM 灰度值到实际振幅响应的 LUT。
- SLM 残余相位响应，如果可测或可间接拟合。
- 空载光强分布。
- 若干随机 SLM pattern 的 CMOS 输出。
- 若干基底 pattern 的 CMOS 输出，例如 Hadamard、Fourier、随机二值图。
- 不同曝光时间下的相机响应曲线。
- 光纤状态变化对输出的影响。

标定目标：

```text
真实硬件：SLM command -> CMOS image
        ↓ 拟合
digital twin：SLM command -> simulated CMOS image
```

标定后的 digital twin 应尽量复现实验 CMOS 输出，而不是只复现理想物理公式。

## 9. 阶段 5：Camera-in-the-loop 微调

硬件标定后，进入闭环微调。

推荐优先微调：

- `generator_e`
- SLM 输入 LUT
- 少量可学习补偿层

不建议在真实硬件阶段首先假设可以更新：

- 已经制备完成的 encoder 超表面
- 已经制备完成的 decoder 超表面
- 已经固定安装的 OLS / 光纤本体

闭环训练路径为：

```text
z + label
        ↓
generator_e
        ↓
真实振幅 SLM
        ↓
真实 encoder + OLS/光纤 + decoder 光路
        ↓
真实 CMOS 采集
        ↓
与 teacher target / real target 计算损失
        ↓
更新 generator_e / LUT / 补偿参数
```

如果无法直接对真实硬件反传梯度，可采用以下替代方式：

- 使用标定后的 digital twin 反传梯度。
- 真实硬件只用于周期性采样和误差校正。
- 使用小批量真实采集数据微调 digital twin。
- 使用黑盒优化或强化学习式策略微调少量 SLM 控制参数。

## 10. 阶段 6：最终单次曝光推理

最终推理路径为：

```text
选择随机低维码 z
选择可选类别 label
        ↓
generator_e 一次前向
        ↓
生成振幅 SLM 控制图 S(x,y)
        ↓
SLM 显示 S(x,y)
        ↓
光路自然传播一次：
encoder metasurface -> OLS/光纤 -> decoder metasurface
        ↓
CMOS 单次曝光
        ↓
生成图像
```

最终推理时不再运行：

- `diffusion U-Net` 多步去噪。
- `OpticalPathVAE.encode()`。
- 软件侧的独立 `latent -> decode` 推理。
- 通过优化输入图案反求目标图像的迭代过程。

## 11. 单层线性超表面建模

encoder 和 decoder 均建模为薄层复透过率：

```text
T_enc(x,y) = A_enc(x,y) · exp(i φ_enc(x,y))
T_dec(x,y) = A_dec(x,y) · exp(i φ_dec(x,y))
```

通过超表面时：

```text
E_after(x,y) = T(x,y) · E_before(x,y)
```

单层线性超表面本身只做线性复场调制，不提供神经网络式非线性。系统主要非线性来自：

- `generator_e`
- OLS / GELU / 光纤非线性
- CMOS 强度读出 `I = |E|^2`
- SLM 和相机的非理想响应

如果超表面仍处于设计阶段，可在仿真中训练 `T_enc` 与 `T_dec`，但需要加入实际可制造约束。如果超表面已经制备完成，则应把它们视为固定物理层，只标定其实际响应。

## 12. 振幅 SLM 建模

由于实验使用振幅 SLM，`generator_e` 输出应解释为振幅控制图，而不是相位图：

```text
S(x,y) = sigmoid(generator_e(z, label))
E_slm(x,y) = A_slm(S(x,y)) · E_in(x,y)
```

理想近似为：

```text
A_slm(S) = S
```

更真实的模型为：

```text
E_slm(x,y) = A_slm(S(x,y)) · exp(i φ_residual(S(x,y))) · E_in(x,y)
```

其中 `A_slm` 与 `φ_residual` 应尽量通过硬件标定得到。振幅 SLM 的自由度低于相位 SLM 或复振幅 SLM，因此后续 encoder 超表面、OLS 和 decoder 超表面的设计与标定更关键。

## 13. OLS / 光纤接口

OLS 在软件中必须被抽象成统一接口：

```text
complex field in -> OLS module -> complex field out
```

当前阶段可以使用 GELU 代理：

```text
E_out = GELU(Re(E_in)) + i · GELU(Im(E_in))
```

后续替换单模光纤时，可建模为：

```text
E_in
        ↓
模式投影
        ↓
非线性相位 / 幅度响应
        ↓
输出模式
```

后续替换多模光纤时，可先用线性传输矩阵近似：

```text
E_out = H_mmf · E_in
```

若考虑更强非线性和模式耦合，可进一步扩展为：

```text
E_out = H2 · nonlinear(H1 · E_in)
```

关键要求是：GELU、单模光纤、多模光纤和实验标定矩阵都占据同一个 OLS 接口位置，避免再次把 encoder 和 decoder 软件式拆开。

## 14. 训练损失建议

基础蒸馏损失：

```text
L_distill = L1(camera_hat, teacher_target)
```

可选重建 / 结构损失：

```text
L_mse
L_ssim
foreground Dice loss
edge / Sobel loss
class recognizer loss
```

物理正则：

```text
SLM command smoothness / contrast regularization
camera energy regularization
background leakage penalty
hotspot / saturation penalty
metasurface manufacturability penalty
```

总损失示例：

```text
L_total =
    λ_distill · L_distill
  + λ_real · L_real_target
  + λ_shape · L_shape
  + λ_cls · L_class
  + λ_phys · L_physical_regularization
```

早期实验应优先保证 `camera_hat` 可辨识，再逐步增加复杂损失。

## 15. 是否仍然是 LDM

修改后，最终推理模型不再是标准 `LDM`。

更准确的命名是：

```text
基于 LDM 蒸馏的单次曝光光学生成模型
```

或：

```text
LDM-distilled single-shot optical generator
```

原因：

- 训练阶段保留 LDM teacher。
- 生成分布来自 LDM 的蒸馏。
- 推理阶段不再运行 diffusion 多步去噪。
- 最终输出由 `generator_e + 振幅 SLM + encoder + OLS/光纤 + decoder + CMOS` 一次光学前向得到。

## 16. 主要解决的光计算问题

本流程解决的问题是：如何把高质量但计算昂贵的数字生成模型，迁移成实际光学系统中可部署的单次曝光生成器。

具体包括：

1. 解决 LDM 推理多步、速度慢、不适合直接硬件部署的问题。
2. 解决实际光路中 encoder、OLS/光纤、decoder 不能被软件式分离的问题。
3. 解决光学系统不知道应该显示什么 SLM 输入的问题，让 `generator_e` 学会生成可驱动光路成像的输入波前。
4. 解决纯理想仿真与真实硬件偏差的问题，通过标定、物理增强和 camera-in-the-loop 微调缩小仿真到现实的差距。
5. 保留 LDM 的分布学习能力，同时让最终推理由真实光传播完成。

## 17. 下一步工程建议

在不影响当前 `LDM v1` 验收主线的前提下，下一阶段可新增独立实验分支：

```text
one_shot
```

建议新增模块边界：

```text
ElectronicGeneratorE
AmplitudeSLMModel
SingleLayerMetasurface
OLSProxy / FiberProxy
IntegratedOpticalGenerator
OneShotDistillationRunner
```

建议新增训练阶段：

```text
train.py --stage one_shot
train.py --stage one_shot_sample
```

建议先做最小可行实验：

```text
MNIST label + random code
        ↓
generator_e
        ↓
ideal amplitude SLM
        ↓
single-layer encoder
        ↓
GELU OLS
        ↓
single-layer decoder
        ↓
ideal CMOS
        ↓
camera_hat
```

验收目标先设为：

- 单次前向生成可辨数字。
- 每个类别至少能稳定生成。
- 同一类别不同随机码能产生有限多样性。
- 加入轻微物理扰动后仍能保持可辨。

在该最小实验通过后，再引入 LDM teacher 蒸馏、复杂物理增强和真实硬件闭环。

## 18. 一句话总结

该实验流程的本质是：

```text
训练时用 LDM 教会 student 生成；
推理时只用 generator_e 生成振幅 SLM 输入；
真实光路通过 encoder 超表面、OLS/光纤、decoder 超表面和 CMOS 一次曝光完成成像。
```
