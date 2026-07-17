# v67 理想透明薄相位面边界

v67 是 diagnostic-only 的加工候选筛选，不替换 Formal Student，也不代表已完成硬件验证。

## 科学来源与本项目边界

- [Optical generative models (Nature, 2025)](https://www.nature.com/articles/s41586-025-09446-5)：本项目借鉴其将生成模型知识蒸馏到单曝光光学生成链的思想；不复制其器件层数、材料或训练目标。
- [All-optical synthesis chip for large-scale intelligent semantic vision generation (Science, 2025)](https://pubmed.ncbi.nlm.nih.gov/41411438/)：本项目参考其 encoder–latent–decoder 和光学系统级组织方式；四层 `2+2`、100×100、2 µm 双光子打印相位层是本项目自定义设计。

### 来源归属矩阵

| 内容 | 论文借鉴 | 本项目合同 |
| --- | --- | --- |
| DDPM teacher distillation、数字 seed encoder、单曝光生成 | Nature | 具体 generator、loss 和 SLM command 为本项目实现 |
| optical encoder、optical latent space、optical generator 的系统抽象 | Science LightGen | 自由空间 phase-only D2NN 与 identity complex-field OLS |
| `2+2` 层、100×100、2 µm、1.50 mm | 非论文复现参数 | 本项目设计/几何推导 |
| 透明外围 `1+0j` | 两篇论文未规定 | 由透明玻璃、无金属遮光层的第一阶段加工假设决定 |

## 理想仿真合同

Formal `finite_crop` 是传播后的中央裁剪计算合同，等价于每层之间存在理想硬孔径；它不是不透明玻璃基底的完整材料模型。v67 使用四个 phase-only 零厚度相位面在完整传播网格上传播：中央 100×100 活动区使用可学习相位，外围复透射率严格为 `1+0j`，层间不执行 destructive crop，1.50 mm 仅作为相位面间空气距离。玻璃厚度、基板折射率/吸收/反射、打印材料折射率、相位量化和装配误差均明确省略。

因此，当前结果只能命名为 `ideal-phase-fabrication-candidate`。在取得玻璃和 ATE-DIP1.52 的 `n/k@532 nm`、结构高度及高度—相位标定前，不导出高度图；`h₂π` 的任何估算都不是加工参数。

## 记录字段

透明诊断 provenance 固定包含 `glass_model=omitted`、`substrate_thickness=None`、`exterior_transmission=identity`、`physical_aperture=None`、`phase_plane_spacing_m=1.5e-3` 和 `fabrication_status=unverified`。活动区和完整传播网格使用独立 sampling/FOV 记录，避免坐标语义混淆。

## 通过条件

先完成 1024/1536/2048 CUDA 收敛门，再允许 A/B 理想训练。只有 held-out test、人工可读性和 code-factor 门全部通过，才可生成四层相位候选；软件 decoder、PCA、ridge 或推理时优化不能计入正式通过。

## v67.2 数值门结果（2026-07-17）

v67.2 在不改变 Formal Student 的前提下增加了 diagnostic `padded_band_limited_asm`：完整 state grid 临时扩大到 FFT grid，传播后只裁回完整 state grid，外围仍严格为恒等复透射，绝不裁回 100×100 活动区。该路径的 provenance 记录 state/FFT grid、输入/保留/丢弃能量和 guard-band ratio；`ideal_transparent_thin_phase_v2` 不能由 formal loader 加载。

冻结 v60 phase、20 个 readable test 样本的 CUDA 结果为 `grid-not-converged`，因此没有运行梯度 smoke、A/B 训练或相位导出：

- 1536/3072 相对 2048/4096：camera corr `0.9999993`、chip complex L2 `0.003382`，首个严格复场失败为 `encoder_layer_2`；amplitude L2 `0.002500`，phase MAE `0.2717 rad`。
- 1536/3072 discarded-energy ratio `3.33e-5`、guard ratio `2.90e-5`，能量和强度相关通过，但不能抵消复场/幅度/相位失败。
- 2048/4096 仅作为当前比较 reference，禁止与自身比较，不能视为候选已收敛。
- 固定 2 µm pitch 时 1536 与 2048 的 state FOV 分别为 `3.072 mm` 和 `4.096 mm`，sampling/FOV 等价门失败；不得把不同物理坐标网格直接当作同一 formal plane。
- 独立小网格 direct-RS 中央归一化强度门为失败（relative L2 `1.924`），所以 padded ASM 尚未取得独立非周期参考的准入证据。

结论是“透明传播实现可运行，但数值/坐标/独立参考门未通过”，不是透明模型训练失败，也不是 Formal v56/v60 失败。按照停止条件，后续必须先修正统一 FOV 的候选定义并复核 RS 参考；在此之前不得降低阈值、扩大网格、启动 A/B 训练或生成加工候选。报告保存在 `outputs/20260717_v672/runs/preflight/v67_transparent_convergence_v3.json`，仅为 diagnostic 工件。
