# Compact 200 μm OLS Dimension Optimization v1

## 结果与状态

完成 synthetic physical-interface diagnostics。固定 532 nm、2 μm、100×100（200 μm）两层 phase-only encoder；encoder 间距和末层至 fiber plane 距离均为 1.5 mm。共测 29 种几何，每种分别运行 throughput、N_eff、1 mW nonlinear-usability 三种 phase-only objective，各 24 步，共 2088 次诊断相位更新；不是生成模型训练。预检 PASS，生产投影复系数/功率相对误差分别为 1.210e-08 / 2.723e-12。没有数据集读取、task evaluator/FID 调用、loss 改动或材料前向。

## 科学认识

固定标准 125 μm pitch 的 10×10 在 N_eff 优化后 N_eff=4.000/100，η=4.7937%。标准 pitch 的 raw 最高值来自 7×7（N_eff=5.126978136）；3×3、5×5、7×7 分别为 5.126974535、5.126976447、5.126978136，三者均显示为 5.127。3×3 与 raw 最大值仅差 3.601e-06，但 3×3 只有 9 根纤芯，-10 dB utilization=55.6%（5×5=20.0%，7×7=10.2%）。因此建议把 3×3 / 125 μm 作为低复杂度 standard-pitch 候选，同时保留 7×7 raw N_eff 点；这不是全局最优证明。4×4/5×5/6×6 中最高 N_eff 候选为 5×5，N_eff=5.127。

本次最大 pitch sensitivity 是 50 μm，regular_10_p50，N_eff=30.683，η=2.0406%；该 pitch 与当前 125 μm cladding 假设冲突，不能作为实验承诺。跨全部已测配置最大观察 N_eff=30.683。

#### Fiber-count sweep (125 μm, N_eff objective)

| Array | Fibers | η total | N_eff | P10/P90 | Relative complexity |
|---|---:|---:|---:|---:|---:|
| 1×1 | 1 | 0.8261% | 1.000 | 1.0000 | 0.8 |
| 2×2 | 4 | 4.7937% | 4.000 | 1.0000 | 3.3 |
| 3×3 | 9 | 0.4938% | 5.127 | 0.0171 | 7.5 |
| 4×4 | 16 | 4.7937% | 4.000 | 0.0000 | 13.3 |
| 5×5 | 25 | 0.4938% | 5.127 | 0.0000 | 20.8 |
| 6×6 | 36 | 4.7937% | 4.000 | 0.0000 | 30.0 |
| 7×7 | 49 | 0.4938% | 5.127 | 0.0000 | 40.8 |
| 8×8 | 64 | 4.7937% | 4.000 | 0.0000 | 53.3 |
| 10×10 | 100 | 4.7937% | 4.000 | 0.0020 | 83.3 |

#### Pitch sensitivity (N_eff objective)

125 μm 为 COMMERCIAL_REFERENCE。100/75/50 μm 是 CUSTOM_ARRAY_ASSUMPTION；在当前 125 μm cladding 假设下与无重叠封装不相容。

| Array | Pitch | Class | η total | N_eff | P10/P90 | Cladding overlap |
|---|---:|---|---:|---:|---:|---|
| 4×4 | 125 µm | COMMERCIAL_REFERENCE | 4.7937% | 4.000 | 0.0000 | no |
| 4×4 | 100 µm | CUSTOM_ARRAY_ASSUMPTION | 3.6903% | 4.011 | 0.0000 | yes |
| 4×4 | 75 µm | CUSTOM_ARRAY_ASSUMPTION | 2.5425% | 13.991 | 0.2694 | yes |
| 4×4 | 50 µm | CUSTOM_ARRAY_ASSUMPTION | 19.5871% | 16.000 | 0.9922 | yes |
| 5×5 | 125 µm | COMMERCIAL_REFERENCE | 0.4938% | 5.127 | 0.0000 | no |
| 5×5 | 100 µm | CUSTOM_ARRAY_ASSUMPTION | 3.3204% | 8.923 | 0.0000 | yes |
| 5×5 | 75 µm | CUSTOM_ARRAY_ASSUMPTION | 11.7139% | 9.001 | 0.0000 | yes |
| 5×5 | 50 µm | CUSTOM_ARRAY_ASSUMPTION | 7.7118% | 24.911 | 0.8173 | yes |
| 6×6 | 125 µm | COMMERCIAL_REFERENCE | 4.7937% | 4.000 | 0.0000 | no |
| 6×6 | 100 µm | CUSTOM_ARRAY_ASSUMPTION | 3.6898% | 4.011 | 0.0000 | yes |
| 6×6 | 75 µm | CUSTOM_ARRAY_ASSUMPTION | 2.5425% | 13.991 | 0.0000 | yes |
| 6×6 | 50 µm | CUSTOM_ARRAY_ASSUMPTION | 2.0406% | 30.680 | 0.2257 | yes |
| 10×10 | 125 µm | COMMERCIAL_REFERENCE | 4.7937% | 4.000 | 0.0020 | no |
| 10×10 | 100 µm | CUSTOM_ARRAY_ASSUMPTION | 3.6905% | 4.011 | 0.0000 | yes |
| 10×10 | 75 µm | CUSTOM_ARRAY_ASSUMPTION | 2.5425% | 13.991 | 0.0000 | yes |
| 10×10 | 50 µm | CUSTOM_ARRAY_ASSUMPTION | 2.0406% | 30.683 | 0.0000 | yes |

#### Layout comparison

Center compact 固定为 62.5 μm（125 μm 的 0.5 倍）。Optimized placement 在零相位下搜索 square/hexagonal lattice、rotation、translation，保持 fiber 数、125 μm 最小中心距和 ±625 μm 边界；这是有限候选搜索，不是连续全局最优。

| Array | Layout | Pitch | η total | N_eff | P10/P90 | Cladding overlap |
|---|---|---:|---:|---:|---:|---|
| 4×4 | regular_square | 125 µm | 4.7937% | 4.000 | 0.0000 | no |
| 4×4 | regular_square | 100 µm | 3.6903% | 4.011 | 0.0000 | yes |
| 4×4 | regular_square | 75 µm | 2.5425% | 13.991 | 0.2694 | yes |
| 4×4 | regular_square | 50 µm | 19.5871% | 16.000 | 0.9922 | yes |
| 4×4 | center_compact_square | 62.5 µm | 8.7226% | 15.992 | 0.9379 | yes |
| 4×4 | optimized_custom_lattice | 125 µm | 1.6694% | 4.000 | 0.0000 | no |
| 5×5 | regular_square | 125 µm | 0.4938% | 5.127 | 0.0000 | no |
| 5×5 | regular_square | 100 µm | 3.3204% | 8.923 | 0.0000 | yes |
| 5×5 | regular_square | 75 µm | 11.7139% | 9.001 | 0.0000 | yes |
| 5×5 | regular_square | 50 µm | 7.7118% | 24.911 | 0.8173 | yes |
| 5×5 | center_compact_square | 62.5 µm | 1.8487% | 20.162 | 0.0019 | yes |
| 5×5 | optimized_custom_lattice | 125 µm | 4.6832% | 4.000 | 0.0000 | no |
| 6×6 | regular_square | 125 µm | 4.7937% | 4.000 | 0.0000 | no |
| 6×6 | regular_square | 100 µm | 3.6898% | 4.011 | 0.0000 | yes |
| 6×6 | regular_square | 75 µm | 2.5425% | 13.991 | 0.0000 | yes |
| 6×6 | regular_square | 50 µm | 2.0406% | 30.680 | 0.2257 | yes |
| 6×6 | center_compact_square | 62.5 µm | 8.7203% | 15.993 | 0.0000 | yes |
| 6×6 | optimized_custom_lattice | 125 µm | 1.6694% | 4.000 | 0.0000 | no |
| 10×10 | regular_square | 125 µm | 4.7937% | 4.000 | 0.0020 | no |
| 10×10 | regular_square | 100 µm | 3.6905% | 4.011 | 0.0000 | yes |
| 10×10 | regular_square | 75 µm | 2.5425% | 13.991 | 0.0000 | yes |
| 10×10 | regular_square | 50 µm | 2.0406% | 30.683 | 0.0000 | yes |
| 10×10 | center_compact_square | 62.5 µm | 8.7203% | 15.993 | 0.0000 | yes |
| 10×10 | optimized_custom_lattice | 125 µm | 4.7937% | 4.000 | 0.0020 | no |

#### Candidate comparison: 4×4, 5×5, 6×6, 10×10

η 采用 throughput optimizer；分布列采用 N_eff optimizer，并从该解的原始 P_i 线性缩放至 1 mW 后算 nonlinear coverage。

| Array | Fibers | η max-throughput | N_eff | Util. -10 dB | Util. -20 dB | Util. -30 dB | P10/P90 | CV | Edge/center | Nonlinear coverage | Manufacturing score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4×4 | 16 | 40.4426% | 4.000 | 25.0% | 25.0% | 25.0% | 0.0000 | 1.732 | 3.222e-08 | 6.2% | 13.3 |
| 5×5 | 25 | 26.1136% | 5.127 | 20.0% | 36.0% | 36.0% | 0.0000 | 1.969 | 2.367e-08 | 2.5% | 20.8 |
| 6×6 | 36 | 40.4426% | 4.000 | 11.1% | 11.1% | 11.1% | 0.0000 | 2.828 | 2.608e-10 | 2.8% | 30.0 |
| 10×10 | 100 | 40.4425% | 4.000 | 4.0% | 4.0% | 4.0% | 0.0020 | 4.899 | 1.088e-09 | 1.0% | 83.3 |

#### Pareto frontier

光阑面积固定为 0.04 mm²，所以 D_latent=N_eff/0.04 mm²，与 N_eff 成比例。下表给出同时考虑 N_eff、η、fiber 数和相对复杂度的非支配候选。

| Geometry | Objective | Fibers | N_eff | η total | Relative complexity |
|---|---|---:|---:|---:|---:|
| regular_10_p50 | maximize_effective_dimension | 100 | 30.683 | 2.0406% | 72.0 |
| regular_6_p50 | maximize_effective_dimension | 36 | 30.680 | 2.0406% | 36.6 |
| regular_5_p50 | maximize_effective_dimension | 25 | 24.911 | 7.7118% | 30.5 |
| regular_4_p50 | maximize_effective_dimension | 16 | 16.000 | 19.5871% | 25.5 |
| regular_10_p50 | maximize_nonlinear_usability_at_1mW | 100 | 15.646 | 60.9731% | 72.0 |
| regular_6_p50 | maximize_nonlinear_usability_at_1mW | 36 | 15.645 | 60.9960% | 36.6 |
| regular_4_p50 | maximize_nonlinear_usability_at_1mW | 16 | 15.639 | 60.9890% | 25.5 |
| regular_4_p75 | maximize_effective_dimension | 16 | 13.991 | 2.5425% | 18.3 |
| regular_5_p75 | maximize_effective_dimension | 25 | 9.001 | 11.7139% | 23.8 |
| regular_5_p75 | maximize_nonlinear_usability_at_1mW | 25 | 8.641 | 53.2896% | 23.8 |
| regular_3_p125 | maximize_effective_dimension | 9 | 5.127 | 0.4938% | 7.5 |
| regular_4_p75 | maximize_nonlinear_usability_at_1mW | 16 | 4.408 | 40.0267% | 18.3 |
| regular_4_p75 | maximize_total_coupling | 16 | 4.134 | 42.5885% | 18.3 |
| regular_4_p100 | maximize_total_coupling | 16 | 4.000 | 42.5993% | 19.7 |
| regular_4_p125 | maximize_total_coupling | 16 | 4.000 | 40.4426% | 13.3 |
| regular_2_p125 | maximize_effective_dimension | 4 | 4.000 | 4.7937% | 3.3 |
| regular_2_p125 | maximize_nonlinear_usability_at_1mW | 4 | 4.000 | 40.4117% | 3.3 |
| regular_2_p125 | maximize_total_coupling | 4 | 4.000 | 40.4427% | 3.3 |
| regular_1_p125 | maximize_total_coupling | 1 | 1.000 | 26.1532% | 0.8 |

#### Nonlinear sensitivity at 1 mW

使用 I_i=2P_i/(πw²)，w=MFD/2；active 是 0.1 < I_i/I_sat < 10。指标只是功率阈值敏感性，不是 Te/TaP/CoSi 或具体器件结论。

| Array | I_sat=10 W/cm² | I_sat=100 W/cm² | I_sat=520 W/cm² | I_sat=1000 W/cm² | I_sat=10000 W/cm² | I_sat=100000 W/cm² | I_sat=1000000 W/cm² | I_sat=30000000 W/cm² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4×4 | 0.0% | 25.0% | 25.0% | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 5×5 | 16.0% | 4.0% | 4.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 6×6 | 0.0% | 11.1% | 11.1% | 11.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10×10 | 0.0% | 4.0% | 4.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |

10×10 的 N_eff 若远低于 100，说明该物理链下端口数不能当作有效 latent 维度；通道集中由 power map、utilization、P10/P90 和 CV 共同描述。此有限步结果不等价于全局理论上限，也不能单独证明 aperture、fiber pitch 或 phase optimization 是唯一原因。

## 目标缺口

没有真实 MFD、fiber placement/封装公差、camera/decoder calibration 或材料参数；phase 优化只面对一个 synthetic coherent top-hat 输入。没有测独立输入模式、语义 latent 可生成性、实验硬件耦合或材料响应。因此未支持生成质量、实验可制造或非线性器件结论。

## 工程完成与验证

锁定 production angular-spectrum finite-crop propagation 与 LP01 complex overlap，不 resize field、不重定位正式 fiber、不做通道或样本级功率归一化。通用变通道 projector 在 10×10/125 μm 上通过严格生产投影对照；执行三步真实双层 phase update、梯度、中心有限差分和反向符号负控。每种 geometry 的完整 channel power map、coordinates 与 nonlinear input distribution 保存在 JSON。制造复杂度仅按 fiber 数、占地、对准点、材料面积和 decoder 通道匹配构成相对负担分数。

同目录机器结果：compact_200um_ols_dimension_results.json；接口：compact_ols_interface.json。mode_overlap_matrix 是 disjoint-cell 数值 port basis 的 Gram 矩阵，不是实测 fiber crosstalk。

## 决策影响与下一步

**G1 — 最大有效 latent：** 全部 sensitivity 中最大观察 N_eff=30.683（regular_10_p50）；标准 125 μm pitch raw 最大值=5.126978（49 fibers）。**G2 — 最佳 fiber 数：** raw N_eff 的精确最大点是 49 fibers，但它与 3×3 只差 3.601e-06，两者在报告显示精度均为 5.127；在显示精度相同的标准-pitch 候选中，推荐复杂度更低的 9 fibers。指定的 4×4/5×5/6×6 候选中 5×5 的 N_eff 最高。**G3 — 最佳 pitch：** 测得 sensitivity 最大值在 50 μm，但该几何与125 μm cladding假设冲突；唯一 commercial reference 仍为125 μm。**G4 — 100维：** 10×10标准pitch N_eff=4.000，不支持有效100维描述；保留为基准。**G5 — 非线性通道：** 后续材料接口敏感性建议先以 9 通道、125 μm regular layout 输入；该建议不等于材料选择或实验授权。

当前最小复杂度标准 pitch 候选的推荐 channel 数为 9，N_eff=5.127、η=0.4938%。10×10 只保留为历史/对照接口，不应表述为有效 100 维。4×4/5×5/6×6 内最高 N_eff 是 5×5；全标准-pitch count scan 则提示 3×3 已达到相同报告精度的 N_eff。

本轮到此停止。Compact OLS + Nonlinear Material + Semantic Generation 是后继工作，不在本轮授权范围内。
