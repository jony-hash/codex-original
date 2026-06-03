# D2NN 单次曝光蒸馏主线

## 1. 项目定位

本项目现在服务一条主线：用 teacher LDM 生成蒸馏目标，再训练 D2NN one-shot 学生模型，让最终推理成为一次 SLM 输入、一次光路传播、一次 CMOS 曝光。

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

Teacher first-stage、diffusion、teacher distill export 是训练基础设施，不属于最终硬件推理路径。直接 MNIST `one_shot` 和 `dense_linear` 光学核心只作为 baseline/smoke/兼容测试保留。

## 2. 当前主线配置

主线配置文件：

- `configs/physics_cfg.yaml`：全局波长、像素尺寸、宏观光路与读出配置。
- `configs/train_ldm.yaml`：teacher first-stage / diffusion 训练入口。
- `configs/export_teacher_distill_mnist_3k_fast200.yaml`：导出 teacher distill dataset。
- `configs/filter_teacher_distill_mnist_3k_fast200_top2k.yaml`：可选过滤 teacher targets。
- `configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml`：D2NN one-shot 蒸馏主线。
- `configs/train_one_shot_d2nn_smoke.yaml`：CPU 小尺寸 D2NN smoke。

D2NN 主线通过 `one_shot.optical_core_arch: diffractive_phase_only` 启用；`dense_linear` 只保留为旧 baseline 和兼容测试入口。

D2NN one-shot 主线几何：

- 工作波长 `λ = 532 nm`（0.532 µm）。
- 单神经元尺寸 / pitch `Δx = 2.5 µm`。
- 层间距 `d = 5.0 mm`（5000 µm）。
- encoder phase-only 衍射层数 `2`。
- decoder phase-only 衍射层数 `2`。
- 总衍射层数 `L = 4`。
- 调制类型 `phase_only`。
- 传播方法 `angular spectrum`。
- 内部 D2NN 网格 `N = 200`，物理孔径 `0.5 mm × 0.5 mm`。
- loss 输出保持 teacher target 的 `200 × 200` camera shape。
- raw diagnostics 保留 D2NN 内部复场、相位 mask 和 camera readout。

宏观硬件光路：

- `Laser 532 nm -> M1`：350 mm。
- `M1 -> BS`：370 mm。
- `BS -> SLM`：25 mm；SLM 为反射式振幅调制，反射光沿 `SLM -> BS` 返回 25 mm。
- `BS -> M3`：125 mm；BS 将 SLM 返回光向下导入 M3。
- `M3 -> LightGen`：475 mm；M3 将竖直向下光束折转为水平进入 LightGen。
- `LightGen -> relay lens`：100 mm。
- `relay lens -> CMOS`：250 mm；`LightGen -> CMOS` 总机械距离为 350 mm。
- relay lens 焦距 `f = 71.4285714 mm`，由 `u = 100 mm, v = 250 mm` 的薄透镜公式反推，使 CMOS 位于合焦像面。
- LightGen 内部仍为 `Encoder x2 -> OLS -> Decoder x2`，其 5 mm 层间距不被宏观距离替代。
- PBS 只作为偏振效率建议，不作为当前数值传播节点。

## 3. 模块边界

- `generator_e`：电子域生成器，输入低维 code 和 label，输出 amplitude SLM command。它不被解释为 phase SLM 或 complex SLM。
- `amplitude SLM`：只直接控制振幅；残余相位只能作为显式标定项加入。
- `encoder/decoder D2NN`：多层相位-only mask，每层执行 `field -> exp(i phase) -> angular_spectrum_propagate(d)`。
- `OLS`：保持可替换接口，输入复场、输出复场。当前可用 GELU 代理，后续可替换为单模光纤、多模光纤、标定矩阵或 camera-in-the-loop 模块。
- `CMOS`：主输出平面；训练、评估优先看下采样后的 `camera_plane`，硬件诊断同时看 `diffractive_camera_*_raw`。

## 4. 推荐流程

训练或复用 teacher：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/train_ldm.yaml --stage teacher_first_stage --device cuda
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/train_ldm.yaml --stage teacher_diffusion --device cuda
```

导出 teacher 蒸馏数据：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/export_teacher_distill_mnist_3k_fast200.yaml --stage teacher_distill_export --device cuda --run-name teacher_distill_mnist_3k_fast200
```

训练 D2NN one-shot 学生：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml --stage one_shot_distill --device cuda --distill-dataset outputs/runs/<teacher_export_run>/artifacts/teacher_distill_dataset.pt --run-name one_shot_distill_d2nn_mnist_3k_fast200
```

采样 D2NN one-shot 学生：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml --stage one_shot_sample --device cuda --one-shot-checkpoint outputs/runs/<student_run>/checkpoints/one_shot_distill_best.pt
```

CPU smoke：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --train-config configs/train_one_shot_d2nn_smoke.yaml --stage one_shot --device cpu
```

## 5. 后续硬件化顺序

当前实现是理想 phase-only D2NN。后续硬件化应按以下顺序加入复杂度：

1. SLM 振幅响应 LUT 与残余相位。
2. phase mask 量化、加工误差和平滑/带宽约束。
3. 传播距离、平移、旋转、缩放和离焦扰动。
4. OLS 从 GELU 代理替换为 fiber / calibrated matrix 接口。
5. CMOS 噪声、饱和、gamma 和像素非均匀性。
6. 真实 `SLM command -> CMOS image` 标定与 camera-in-the-loop 微调。

每一步都应保持最终推理路径不依赖 diffusion U-Net 多步去噪，也不依赖软件-only VAE decoding。

## 6. 验收基线

**基线提交**: `c99f559 refactor: training orchestration into stage modules`
**验收日期**: 2026-06-03
**环境**: RTX 3060 Laptop GPU (6GB), CUDA 12.1, PyTorch 2.5.1+cu121, WSL2 Linux

### 全链路验收

| Stage | 配置 | Epochs | 关键指标 | Exit |
| ----- | ---- | ------ | -------- | ---- |
| `teacher_first_stage` | `configs/train_ldm.yaml` | 35 | `camera_corr=0.9547`, `balanced=0.8770`, `mnist_score=0.8144` | 0 |
| `one_shot_distill` | `configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml` | 20 | `quality=0.1614`, `camera_dice=0.4575` | 0 |
| `one_shot_sample` | 同上 config | — | 可视化/诊断/tensor bundle 完整 | 0 |
| `one_shot_distill_eval` | 同上 config | — | `mse=0.020`, `corr=0.807`, `dice=0.756` | 0 |

- 教师蒸馏数据: 3000 samples，源自 `teacher_distill_export` (batch_size=96, guidance_scale=2.0, steps=200)。
- 训练和采样均使用 `--device cuda`。
- `one_shot_sample` 和 `one_shot_distill_eval` 使用 `one_shot_distill_best.pt` checkpoint。
- 所有 artifact/checkpoint 位于 `outputs/runs/...`，不纳入 git。
