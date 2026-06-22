# D2NN 单次曝光蒸馏主线

本文件定义稳定架构、最终推理边界和模块职责，不保存带日期的实验状态。

- 运行 stage：[Stage Guide](stage_guide.md)
- 解释输出：[Output Guide](../output_JIESHI.md)
- 查阅历史证据：[Experiment History](experiment_history.md)
- 浏览全部文档：[Documentation](../README.md)

## 1. 项目定位

本项目用 teacher LDM 生成蒸馏目标，再训练 D2NN one-shot 学生模型，使最终推理成为一次 SLM 输入、一次光路传播和一次 CMOS 曝光。

最终推理路径固定为：

```text
noise/code + optional label
        ↓
generator_e
        ↓
amplitude SLM command
        ↓
encoder D2NN phase-only diffractive stack
        ↓
OLS / fiber interface
        ↓
decoder D2NN phase-only diffractive stack
        ↓
CMOS camera plane
        ↓
generated image
```

Teacher first-stage、diffusion 和 teacher distill export 是训练基础设施，不属于最终硬件推理路径。直接 MNIST `one_shot` 和 `dense_linear` 光学核心只作为 baseline、smoke 或兼容入口保留。

最终 inference 不得依赖 diffusion U-Net 多步去噪、迭代 latent sampling、software-only VAE decoding 或 inference-time optimization。

## 2. 主线配置与几何

核心配置职责：

- `configs/physics_cfg.yaml`：波长、像素尺寸、宏观光路与读出配置。
- `configs/train_ldm.yaml`：teacher first-stage 与 diffusion 训练入口。
- `configs/export_teacher_distill_mnist_3k_fast200.yaml`：稳定 teacher distill dataset 导出入口。
- `configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml`：D2NN one-shot 蒸馏主线。
- `configs/train_one_shot_d2nn_smoke.yaml`：小尺寸 D2NN compatibility smoke，不作为正式证据。

其他 export、filter、split、audit 和实验 config 的分类与使用条件见 [Stage Guide](stage_guide.md)。

D2NN 主线通过 `one_shot.optical_core_arch: diffractive_phase_only` 启用；`dense_linear` 只保留为旧 baseline 和兼容入口。

### D2NN 几何

- 工作波长 `λ = 532 nm`（0.532 µm）。
- 单神经元尺寸 / pitch `Δx = 2.5 µm`。
- 层间距 `d = 5.0 mm`（5000 µm）。
- Encoder phase-only 衍射层数为 2。
- Decoder phase-only 衍射层数为 2。
- 总衍射层数 `L = 4`。
- 调制类型为 `phase_only`。
- 传播方法为 angular spectrum。
- 内部 D2NN 网格 `N = 200`，物理孔径为 `0.5 mm × 0.5 mm`。
- Loss 输出保持 teacher target 的 `200 × 200` camera shape。
- Raw diagnostics 保留 D2NN 内部复场、phase mask 和 camera readout。

### 宏观光路

- `Laser 532 nm -> M1`：350 mm。
- `M1 -> BS`：370 mm。
- `BS -> SLM`：25 mm；SLM 为反射式振幅调制，反射光沿 `SLM -> BS` 返回 25 mm。
- `BS -> M3`：125 mm；BS 将 SLM 返回光向下导入 M3。
- `M3 -> LightGen`：475 mm；M3 将竖直向下光束折转为水平进入 LightGen。
- `LightGen -> relay lens`：100 mm。
- `relay lens -> CMOS`：250 mm；`LightGen -> CMOS` 总机械距离为 350 mm。
- Relay lens 焦距 `f = 71.4285714 mm`，由 `u = 100 mm, v = 250 mm` 的薄透镜公式反推，使 CMOS 位于合焦像面。
- LightGen 内部仍为 `Encoder x2 -> OLS -> Decoder x2`；其 5 mm 层间距不被宏观距离替代。
- PBS 只作为偏振效率建议，不作为当前数值传播节点。

## 3. 模块边界

### generator_e

电子域生成器输入低维 code 和 optional label，输出 amplitude SLM command。它不能被解释为 phase SLM 或 complex SLM。

`v9.10 label-template inverse prototype baseline` 仅用于验证 command calibration 与完整光路。它把 checkpoint 内的类别 intensity template 映射为实际 amplitude SLM command，并明确忽略 code；其结果不能解释为同类多样性或最终 generative student。

### Amplitude SLM

SLM 只直接控制振幅。残余相位只能作为显式标定项加入，不能在文档或评估中隐式假设 complex-field control。

### Encoder / Decoder D2NN

Encoder 和 decoder 由多层 phase-only mask 构成，每层执行：

```text
field -> exp(i phase) -> angular_spectrum_propagate(d)
```

Phase mask 是可训练光学参数；其幅度保持为 1。

### OLS

OLS 保持可替换的 complex-field 输入/输出接口。当前可以使用 GELU 代理，后续可替换为单模光纤、多模光纤、标定矩阵或 camera-in-the-loop 模块，但不得绕过最终单次曝光路径。

### CMOS

CMOS camera plane 是学生主输出。训练和评估优先使用下采样后的 `camera_plane`；硬件诊断同时检查 `diffractive_camera_*_raw`。具体张量和图像语义见 [Output Guide](../output_JIESHI.md)。

## 4. Teacher 配套边界

`first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自配套的 first-stage latent/decoder 分布。

First-stage 和 diffusion 共享 `OpticalPathVAE` encoder/decoder。不同 `linear_metasurface_grid_size`、随机种子、batch size、学习率、loss 权重或架构超参数会改变 latent 分布。若 diffusion 采样的 latent 分布与 first-stage decoder 期望分布不一致，`latent -> decoder -> camera` 质量会退化，teacher distill target 随之失效。

操作约束：

1. 更换 first-stage checkpoint 后必须重新训练 diffusion，不得复用不匹配的 diffusion checkpoint。
2. `teacher_distill_export` 显式接收 checkpoint 时，两者必须来自同一 teacher 链路。
3. Acceptance baseline 的 `source_runs` 必须记录配套关系，用于 checkpoint 溯源。
4. 修改 teacher latent geometry 后，必须重新运行 `first-stage -> diffusion -> export -> distill` 全链路。

匹配与不匹配 checkpoint 的历史测量对比见 [Experiment History](experiment_history.md)。

## 5. 运行与验收边界

正式主流程为：

```text
teacher_first_stage / teacher_diffusion
-> teacher_distill_export
-> one_shot_distill
-> one_shot_sample
-> one_shot_distill_eval
```

命令、必需输入、checkpoint policy、diagnostic-only 标签和 output policy 均由 [Stage Guide](stage_guide.md) 维护。

验收时必须保持以下边界：

- Teacher 生成质量只说明 teacher infrastructure 能提供目标，不代表 student hardware ready。
- `one_shot_distill` 只说明训练完成，不代表最终推理路径已验证。
- 最终证据必须来自显式加载 one-shot student checkpoint 的 sample 和 eval。
- Direct MNIST、CPU smoke、readability substitute 和单环节 audit 均不得替代正式链路。
- 有效 acceptance gate 失败后，不得继续下游训练或结果提升。

## 6. 硬件化顺序

当前实现是理想 phase-only D2NN。后续硬件化应按以下顺序加入复杂度：

1. SLM 振幅响应 LUT 与残余相位。
2. Phase mask 量化、加工误差和平滑/带宽约束。
3. 传播距离、平移、旋转、缩放和离焦扰动。
4. OLS 从 GELU 代理替换为 fiber 或 calibrated matrix 接口。
5. CMOS 噪声、饱和、gamma 和像素非均匀性。
6. 真实 `SLM command -> CMOS image` 标定与 camera-in-the-loop 微调。

每一步都必须保持最终推理路径不依赖 diffusion denoising 或 software-only VAE decoding。涉及 SLM control、metasurface trainability、OLS behavior 或 CMOS calibration 的行为变更，需要在实施前重新确认边界。

## 7. 历史证据

带日期的基线、负实验、seed sweep、post-refactor smoke 和受控复验记录已集中到 [Experiment History](experiment_history.md)。这些记录解释“过去验证了什么”，不覆盖本文件的架构契约或 [Stage Guide](stage_guide.md) 的当前操作要求。
