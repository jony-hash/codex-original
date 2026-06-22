# D2NN 主线输出与可视化说明

本文件只解释生成图片、tensor bundle 和光学诊断。Stage 命令见 [Stage Guide](ldm_migration/stage_guide.md)，完整文档导航见 [Documentation](README.md)。

## Teacher 与蒸馏数据

Teacher first-stage / diffusion 只用于训练和导出蒸馏目标。相关图像仍可能出现在：

- `outputs/runs/<run_name>/images/first_stage/`
- `outputs/runs/<run_name>/images/diffusion/`
- `outputs/runs/<run_name>/images/samples/`

这些图用于检查 teacher 是否能提供可用的 `camera` target，不代表最终 D2NN 硬件推理路径。

## D2NN One-shot 输出

主线 one-shot 输出位于：

- `outputs/runs/<run_name>/images/one_shot_distill/epoch_XXXX_amplitude.png`
- `outputs/runs/<run_name>/images/one_shot_distill/epoch_XXXX_phase.png`
- `outputs/runs/<run_name>/images/one_shot_sample_amplitude.png`
- `outputs/runs/<run_name>/images/one_shot_sample_phase.png`
- `outputs/runs/<run_name>/images/one_shot_distill_eval_amplitude.png`
- `outputs/runs/<run_name>/images/one_shot_distill_eval_phase.png`

振幅图按单次光路读；SLM 列显示 amplitude-control command / out amplitude，后续复场平面显示共享尺度下的强度 `I = amplitude^2`：

```text
noise code -> reference/teacher -> slm command -> slm out amp
-> encoder in/grid/out I -> OLS I -> decoder out I -> chip I -> camera/student
```

相位图按同一光路读：

```text
slm out phase -> encoder phase diagnostics -> OLS boundary
-> decoder phase diagnostics -> chip phase
```

`camera/student` 是训练和评估的主输出。raw D2NN/grid/camera diagnostics 的尺寸由 `one_shot.d2nn.grid_size` 和 decoder 输出的 chip grid 决定；当前主线为 `N=200`，loss 和主评估仍使用 `200 × 200` `camera_plane`。不要把 raw D2NN camera diagnostics 和 loss 图像尺寸混为同一个物理平面，也不要把 raw diagnostics 当成另一条产品输出路径。

## Raw Tensor Bundle

`one_shot_sample_tensors.pt`、`one_shot_distill_eval_tensors.pt` 和训练 artifact 会保留：

- amplitude SLM command 与 SLM 出射场。
- encoder/decoder 输入输出复场。
- OLS 输入输出复场。
- `camera_plane`：用于 loss 和主评估的 `200 × 200` 相机输出，是 one-shot student 的主输出。
- D2NN raw diagnostics：内部衍射层振幅/相位，尺寸由 `one_shot.d2nn.grid_size` 决定，当前主线为 `N=200`。
- D2NN high-res camera diagnostics：`diffractive_camera_intensity_raw`、`diffractive_camera_plane_raw`、`diffractive_camera_plane_loss_raw`。
- phase-only mask diagnostics：mask 幅度应恒为 1，相位为可训练参数；mask 幅度只在 diagnostics 中保留，不进入主振幅图。

`dense_linear` 仍是保留 baseline，但主线应使用 `one_shot.optical_core_arch: diffractive_phase_only`。判断 D2NN 是否正常时，优先看 `reference/teacher` 与 `camera/student` 是否一致，再看 SLM、encoder、OLS、decoder 和 chip 的振幅/相位诊断定位问题。
