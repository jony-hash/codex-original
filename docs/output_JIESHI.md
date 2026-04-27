# 三类图的列含义与所在位置如下

**First Stage 图**
位置：`outputs/runs/<run_name>/images/first_stage/epoch_XXXX.png`

- `input`: 原始 MNIST 输入数字。
- `camera_hat`: 经过 decoder 复场、`probe` 平面再经理想相机成像近似后的相机平面读出，是物理优先版本最该看的主结果。
- `image_hat`: decoder 的直接数字图像头输出，用来辅助判断 latent 是否保留数字语义。
- `pf2`: 输入经过前段光学/物理路径后的中间强度图，通常会很像散斑纹理。
- `chip_amplitude`: 芯片平面上的复场幅度，能看到数字信息是否被编码进去。
- `chip_phase`: 芯片平面上的相位，归一化后显示，主要用于诊断光学编码。
- `probe`: 探针/传播到中间平面的光场强度。

当前 `camera_hat` 不是“probe 再继续做同采样 Fresnel 传播后的末端强度图”，而是把 `probe` 复场按 relay lens 放大率做理想成像近似，再回采样到固定输出网格。这个近似优先保证 `camera_hat` 和 `probe` 同形、同向，避免旧实现里因为放大后中心裁剪导致只剩局部轮廓。

以前 `image_hat` 和 `input` 很像，是因为旧配置使用 `dual_path_concat`，encoder 同时看 PF2 光学特征和原图特征；同时 `primary_output: image` 且 `image_recon_weight` 最大，训练目标本来就是让直接图像头重建输入。物理优先版本改成 `optical_only` 和 `primary_output: camera` 后，重点转为检查 `camera_hat` 是否像输入数字。

**Diffusion 图**
位置：`outputs/runs/<run_name>/images/diffusion/epoch_XXXX.png`

- `reference`: 当前条件标签对应的参考数字图，用来肉眼对照目标类别。
- `generated_camera`: diffusion 采样 latent 经过 first stage 后级光路与理想相机成像近似后的相机读出，是物理优先版本最该看的生成结果。
- `generated_image`: 同一个 latent 的 decoder 直接数字图像头输出，是辅助诊断列。
- `chip_amplitude`: 生成样本对应的芯片复场幅度。
- `chip_phase`: 生成样本对应的芯片复场相位。
- `probe`: 生成样本对应的中间光场强度。

**Samples 图**
位置：`outputs/runs/<run_name>/images/samples/post_train_sample_grid.png`

和 `diffusion` 图含义一样，只是它不是某个训练 epoch 的中间可视化，而是训练结束后的最终采样结果。

所以看结果时优先级可以简单记：`first_stage` 看 `input` vs `camera_hat`；`diffusion/samples` 看 `reference` vs `generated_camera`。`image_hat` / `generated_image` 现在主要用于辅助判断 latent 数字语义，而不是最终物理读出质量。
