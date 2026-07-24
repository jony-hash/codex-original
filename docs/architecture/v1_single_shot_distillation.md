# D2NN 单次曝光蒸馏主线

本文件定义稳定架构、最终推理边界和模块职责，不保存带日期的实验状态。当前实验状态与验收结论见 [v79/v2 主线交接](../status/current_handoff.md)；分支与回档点见 [Git 分支治理指南](../operations/git_branch_management.md)。物理合同通过不等于学生模型已通过最终 CMOS 人眼验收。

- 运行 stage：[Stage Guide](../operations/stage_guide.md)
- 解释输出：[Output Guide](../operations/output_artifact_notes.md)
- 查阅历史证据：[Experiment History](../status/experiment_history.md)
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
linear identity OLS (fiber is not implemented)
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

- `configs/v1/formal/physics_cfg.yaml`：波长、像素尺寸、宏观光路与读出配置。
- `configs/v1/formal/train_ldm.yaml`：teacher first-stage 与 diffusion 训练入口。
- `configs/v1/teacher/export_teacher_distill_mnist_3k_fast200.yaml`：稳定 teacher distill dataset 导出入口。
- `configs/v1/formal/train_v56_coherent_relay_from_scratch.yaml`：最近一次批准的 coherent-relay 训练配置；当前训练已停止，详见交接文档。旧 one-shot baseline 位于 `experiments/archive/`。
- `configs/v1/smoke/train_one_shot_d2nn_smoke.yaml`：小尺寸 D2NN compatibility smoke，不作为正式证据。

其他 export、filter、split、audit 和实验 config 的分类与使用条件见 [Stage Guide](../operations/stage_guide.md)。

D2NN 主线通过 `one_shot.optical_core_arch: diffractive_phase_only` 启用；`dense_linear` 只保留为旧 baseline 和兼容入口。

### v55 正式 D2NN 几何与 relay

- 工作波长 `λ = 532 nm`（0.532 µm）。
- SLM 与 CMOS 为 `200 × 200 @ 2.5 µm`，物理 FOV 为 `500 µm`。
- Encoder 与 decoder metasurface 各为 `100 × 100 @ 2 µm`，物理 FOV 为 `200 µm`。
- 层间距 `d = 1.50 mm`，满足当前全连接几何下限。
- Encoder phase-only 衍射层数为 2。
- Decoder phase-only 衍射层数为 2。
- 总衍射层数 `L = 4`。
- 调制类型为 `phase_only`。
- 传播方法为 angular spectrum。
- 前端使用相干 4f 缩小 relay，将完整 `500 µm` SLM FOV 以倍率 `−0.4` 成像到完整 `200 µm` Encoder FOV；禁止使用自由传播后的中央 aperture 截取作为正式路径。
- 前端 relay 为 `f1=223.214 mm`、`f2=89.286 mm`，两镜间距 `312.5 mm`，SLM→Encoder 总路径 `625 mm`，最低 clear aperture `23.75 mm`。
- CMOS 输出保持 `200 × 200 @ 2.5 µm` camera shape。
- Raw diagnostics 保留 D2NN 内部复场、phase mask 和 camera readout。

### 宏观光路

- `Laser 532 nm -> M1`：350 mm。
- `M1 -> BS`：370 mm。
- `BS -> SLM`：25 mm；SLM 为反射式振幅调制，反射光沿 `SLM -> BS` 返回 25 mm。
- `BS -> M3`：125 mm；BS 将 SLM 返回光向下导入 M3。
- `M3 -> L1`：73.214 mm；随后 `L1 -> L2` 为 312.5 mm，`L2 -> Encoder` 为 89.286 mm，构成完整前端 4f relay。
- `chip -> backend relay lens`：100 mm。
- `backend relay lens -> CMOS`：250 mm；chip -> CMOS 总机械距离为 350 mm。
- Relay lens 焦距 `f = 71.4285714 mm`，由 `u = 100 mm, v = 250 mm` 的薄透镜公式反推，使 CMOS 位于合焦像面。
- LightGen（Science）提供振幅 SLM 输入、optical encoder–latent–generator 的系统组织，以及本项目采用的器件距离标定参考；当前配置中的具体距离是用户确认的 LightGen 参照标定合同，不声称 Science 正文逐项原样给出。论文的 OLS/光纤连接、层数、尺寸和间距不等同于本项目。当前项目自定义为 `phase-only Encoder x2 -> identity complex-field OLS -> phase-only Decoder x2`；100×100、2 µm、2+2 和 1.50 mm 均为本项目双光子打印缩放参数。identity OLS 是保留复场接口的理想近似，不等同于论文中的单模光纤阵列。Nature 论文只提供 DDPM teacher 蒸馏与单曝光生成思想，不替换本项目的振幅 SLM 或 relay。
- PBS 只作为偏振效率建议，不作为当前数值传播节点。
- v55 physical-forward contract 中 relay 距离、倍率、pupil 和 ABCD trace 全部进入 provenance；v52 的自由传播+中央 aperture 仅保留为历史 diagnostic，不代表当前正式路径。

## 3. 模块边界

正式代码按五个领域组织：`optics/`（复场/传播/relay/device/D2NN）、`student/`（正式单曝光模型、canonical generator、训练评估与 checkpoint）、`teacher/`（VAE/DDPM 与 target export）、`application/`（CLI、stage registry 与懒调度）和 `diagnostics/`。中立 `generation/`、`distillation/`、`evaluation/` 提供 code、payload/target、prototype/label-audit 能力。`configuration/` 实际拥有 schema、loading、validation；根 `config.py` 和 `configuration/legacy.py` 只保留兼容导出。teacher distill workflow 分别位于 `teacher/export.py`、`label_audit.py`、`filter.py` 和 `readable.py`，`teacher/common.py`、`teacher/distill.py`、`teacher/distill_legacy.py` 和根 `teacher_distill.py` 仅保留兼容入口。实验 objective term 位于 `student/objective_terms/`，`student/objectives.py` 只做编排。正式 v56 链由 `FormalOneShotOpticalGenerator` 固定；旧 `OneShotOpticalGenerator` 仅保留历史 compatibility/diagnostic 变体。依赖只可从 application 指向 teacher/student，teacher 与 student 不得彼此导入，optics 不得导入上层领域；顶层旧模块仅是兼容 shim。

`application.stage_runtimes` 在构造兼容 `StageContext` 前登记 data-only、student-only、teacher 的依赖 footprint：filter/split/audit 不构造模型或 loader，student stage 不声明 teacher 网络。`StageContext` 仍作为旧调用适配器保留，正式 CLI 的 footprint 由 registry 审计。

正式 optics 组件的唯一实现位于 `optics/devices.py` 与 `optics/d2nn.py`：SLM command 到复传递函数、identity OLS、phase-only modulation、center padding→ASM propagation→center crop 和能量 provenance 均在该层定义。`student/layers.py` 与 `one_shot_optical_layers.py` 只为旧 import 提供 aliases；`student/experimental/` 承载 phase-amplitude、generic resize、decoder bypass、target-code bridge 和 SLM override。formal model 的完全独立模块化仍需下一道 bitwise 等价门，不得以兼容继承替代。

`student/checkpoints.py` 拥有 checkpoint version、完整加载和显式 subset warm-start 合同；`student/formal_contract.py` 在构造正式模型前拒绝诊断配置。`distillation/targets.py` 是 teacher payload 与 student supervision 共享的中立 target transform，禁止 payload loader 反向导入 student objective。

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

当前正式 OLS 是严格复恒等映射 `Eout = Ein`，并通过复线性叠加测试。fiber、标定矩阵和 camera-in-the-loop 仅是未来可选实现；历史 GELU/tanh 代理只允许历史配置读取，不能进入 v55 physical-forward contract。

### CMOS

CMOS camera plane 是学生主输出。训练和评估优先使用下采样后的 `camera_plane`；硬件诊断同时检查 `diffractive_camera_*_raw`。具体张量和图像语义见 [Output Guide](../operations/output_artifact_notes.md)。

Teacher 只生成目标图像/目标 latent，不经过学生的 laser、SLM、D2NN 或 CMOS 光路；teacher 与 student 不是同一条物理路径。

第五轮收口后，正式 Student 入口是固定物理链 `FormalOneShotOpticalGenerator`；历史 phase-amplitude、bypass、generic-resize 等干预只由 `student.experimental` 兼容路径构造。应用层先按 stage 生成 `DataOnlyRuntime`、`StudentRuntime` 或 `TeacherRuntime`，再物化该阶段声明的模型和数据，旧 `StageContext` 不再拥有独立构造逻辑。正式 camera reconstruction term 位于 `student/objective_terms/formal.py`，`one_shot_loss` 仅保留兼容编排，确保正式项与实验项的边界可审计。

## 4. Teacher 配套边界

`first_stage_best_optical.pt` 与 `diffusion_best.pt` 必须来自配套的 first-stage latent/decoder 分布。

First-stage 和 diffusion 共享 `OpticalPathVAE` encoder/decoder。不同 `linear_metasurface_grid_size`、随机种子、batch size、学习率、loss 权重或架构超参数会改变 latent 分布。若 diffusion 采样的 latent 分布与 first-stage decoder 期望分布不一致，`latent -> decoder -> camera` 质量会退化，teacher distill target 随之失效。

操作约束：

1. 更换 first-stage checkpoint 后必须重新训练 diffusion，不得复用不匹配的 diffusion checkpoint。
2. `teacher_distill_export` 显式接收 checkpoint 时，两者必须来自同一 teacher 链路。
3. Acceptance baseline 的 `source_runs` 必须记录配套关系，用于 checkpoint 溯源。
4. 修改 teacher latent geometry 后，必须重新运行 `first-stage -> diffusion -> export -> distill` 全链路。

匹配与不匹配 checkpoint 的历史测量对比见 [Experiment History](../status/experiment_history.md)。

## 5. 运行与验收边界

正式主流程为：

```text
teacher_first_stage / teacher_diffusion
-> teacher_distill_export
-> one_shot_distill
-> one_shot_sample
-> one_shot_distill_eval
```

命令、必需输入、checkpoint policy、diagnostic-only 标签和 output policy 均由 [Stage Guide](../operations/stage_guide.md) 维护。

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
4. 在独立硬件变体中以保留复场接口的 fiber 或 calibrated matrix 替换 identity OLS。
5. CMOS 噪声、饱和、gamma 和像素非均匀性。
6. 真实 `SLM command -> CMOS image` 标定与 camera-in-the-loop 微调。

每一步都必须保持最终推理路径不依赖 diffusion denoising 或 software-only VAE decoding。涉及 SLM control、metasurface trainability、OLS behavior 或 CMOS calibration 的行为变更，需要在实施前重新确认边界。

## 7. 历史证据

带日期的基线、负实验、seed sweep、post-refactor smoke 和受控复验记录已集中到 [Experiment History](../status/experiment_history.md)。这些记录解释“过去验证了什么”，不覆盖本文件的架构契约或 [Stage Guide](../operations/stage_guide.md) 的当前操作要求。
