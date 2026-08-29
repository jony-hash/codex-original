# LightGen Science 论文证据包

本目录是 Yitong Chen 等人 Science 论文
“All-optical synthesis chip for large-scale intelligent semantic vision
generation”（DOI: `10.1126/science.adv7434`）的只读研究资料包。

## 内容

- `paper/sciencesm.pdf`：用户提供的 Science 补充材料 PDF，共 46 页。
- `source_data/`：论文图源 CSV，包括 Figure 3、Figure 4-1 至 Figure 4-13、Figure S1 和 Figure S6。
- `manifest.json`：仓库相对路径、文件大小和 SHA256 完整性清单。

## 数据语义

- Figure S1 是 0–9 数字在 OLS 中的二维分布数据，可用于潜空间聚类可视化。
- Figure 4 数据主要是潜空间插值、语义特征和平均指标，可用于重画趋势图。
- Figure 3 和 Figure S6 是风格迁移的 PSNR、内容损失和风格损失等汇总数据。
- CSV 中的 `A.U.` 是论文绘图使用的归一化单位，不是 SI 光场，也不是可直接训练的 100 维复数 OLS 张量。

## 物理参数边界

补充材料确认了 10×10 单模光纤阵列、8.5 mm 光纤长度和 125 μm 外部直径，
但没有公开芯径、NA、MFD、端口间距、串扰、插入损耗或 100 端口复数传输矩阵。
这些资料不能把项目中的 synthetic LP01 S0 变成论文实测光纤模型。

补充材料还报告了 532 nm、3 μm 元表面像素、8 级量化和元表面平均透射率
0.889；项目中的其他层间距、模场半径、宏观光路和 BOGT 权重仍须按项目状态
标注为 `project_approximation` 或 `unknown`。

## 使用和边界

该资料包是论文证据和图源数据，不是作者官方训练代码，不接入正式 v1/v2
runtime，也不授权训练、真实 SMF/FMF 验证或 formal promotion。PPT、论文解读
和 LightGen 结果审计应引用本目录中的相对路径，并先通过 `manifest.json`
完整性校验。

完整补充 PDF 仅作为当前私有研究仓库的参考副本保存，不作公开再分发。
