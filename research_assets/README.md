# Research assets

本目录存放研究证据包（论文 PDF、图源 CSV、XLSX、参考数据）。除 `lightgen/science_adv7434/` 外，这些 bundle **不纳入 Git**：`.gitignore` 只保留它们的本地副本，版本库中不保存路径、大小或哈希清单。

## 策略

1. 这些 bundle 是**本地-only 证据**。文档中指向 `research_assets/` 的链接只在当前机器可解析；全新 clone 不会包含它们。
2. 删除或移动任何 bundle 前，先复制到外部不可变归档并记录目标路径；不要在未归档的情况下清理。
3. 当前工作区未检测到 `D:\LDM-Archive`。外部归档位置必须在使用前确认，不能假定它已经存在。
4. `lightgen/science_adv7434/` 是唯一被跟踪的 bundle（Science 补充材料与图源 CSV），保持现状。

## Bundle 清单

| Bundle | 日期 | 用途（按目录名） |
| --- | --- | --- |
| `lightgen/hardware_g1_direct_facet_linear_20261028_v1/` | 2026-10-28 | G1 直接 facet 线性证据 |
| `lightgen/hardware_g1_material_evidence_20261028_v1/` | 2026-10-28 | G1 材料证据 |
| `lightgen/hardware_g1_nonlinear_sensitivity_20261030_v1/` | 2026-10-30 | G1 非线性敏感性 |
| `lightgen/hardware_g1_recent_highimpact_process_20261028_v1/` | 2026-10-28 | G1 近期高影响工艺文献 |
| `lightgen/hardware_g1_sputtered_te_532_design_20261028_v1/` | 2026-10-28 | G1 溅射 Te 532 nm 设计 |
| `lightgen/hardware_g1_stage_review_20261029_v1/` | 2026-10-29 | G1 阶段复审 |
| `lightgen/hardware_g1_te_adma2026_proxy_20261029_v1/` | 2026-10-29 | ADMA 2026 Te 代理参数 |
| `lightgen/hardware_g1_te_inserted_path_20261029_v1/` | 2026-10-29 | Te 插入路径 |
| `lightgen/hardware_g1_te_literature_proxy_20261029_v1/` | 2026-10-29 | Te 文献代理 |
| `lightgen/hardware_g1_te_nonlinear_bridge_20261029_v1/` | 2026-10-30 | Te 非线性桥接 |
| `lightgen/hardware_g1_te_nonlinear_literature_local_20261029_v1/` | 2026-10-29 | 非线性文献本地副本 |
| `lightgen/HW207_实物与距离基线_20261103/` | 2026-11-03 | HW207 实物与距离基线 |
| `lightgen/literature_library/` | 2026-10-30 | 文献库 |
| `lightgen/science_adv7434/` | 2026-08-29 | **已跟踪**：Science 补充材料与图源 CSV |

## 变更规则

- 新增 bundle 使用 `<family>/<topic>_<YYYYMMDD>_v<N>/` 命名，并在本表登记。
- 本文件是唯一被跟踪的入口；不要为了登记 bundle 而把 bundle 内容加回 Git。