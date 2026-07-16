# v67 理想透明薄相位面边界

v67 是 diagnostic-only 的加工候选筛选，不替换 Formal Student，也不代表已完成硬件验证。

## 科学来源与本项目边界

- [Optical generative models (Nature, 2025)](https://www.nature.com/articles/s41586-025-09446-5)：本项目借鉴其将生成模型知识蒸馏到单曝光光学生成链的思想；不复制其器件层数、材料或训练目标。
- [All-optical synthesis chip for large-scale intelligent semantic vision generation (Science, 2025)](https://pubmed.ncbi.nlm.nih.gov/41411438/)：本项目参考其 encoder–latent–decoder 和光学系统级组织方式；四层 `2+2`、100×100、2 µm 双光子打印相位层是本项目自定义设计。

## 理想仿真合同

四个 phase-only 零厚度相位面在完整传播网格上传播。中央 100×100 活动区使用可学习相位，外围复透射率严格为 `1+0j`；层间不执行 destructive crop，1.50 mm 仅作为相位面间空气距离。玻璃厚度、基板折射率/吸收/反射、打印材料折射率、相位量化和装配误差均明确省略。

因此，当前结果只能命名为 `ideal-phase-fabrication-candidate`。在取得玻璃和 ATE-DIP1.52 的 `n/k@532 nm`、结构高度及高度—相位标定前，不导出高度图；`h₂π` 的任何估算都不是加工参数。

## 记录字段

透明诊断 provenance 固定包含 `glass_model=omitted`、`substrate_thickness=None`、`exterior_transmission=identity`、`physical_aperture=None`、`phase_plane_spacing_m=1.5e-3` 和 `fabrication_status=unverified`。活动区和完整传播网格使用独立 sampling/FOV 记录，避免坐标语义混淆。

## 通过条件

先完成 1024/1536/2048 CUDA 收敛门，再允许 A/B 理想训练。只有 held-out test、人工可读性和 code-factor 门全部通过，才可生成四层相位候选；软件 decoder、PCA、ridge 或推理时优化不能计入正式通过。
